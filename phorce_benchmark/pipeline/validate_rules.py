"""
validate_rules.py — external validation of EFTE rule classes vs measured PPDB ground truth.

Rule prediction = stored cls_* flag (descriptor-based, Table S2).
Ground truth     = derived from the MEASURED endpoint that represents the class property,
                   using citable regulatory thresholds. Analysis is restricted to compounds
                   where that endpoint was measured (BCF: measured-only, estimates excluded).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss

_DATA = Path(__file__).resolve().parent.parent / "data"
a = pd.read_csv(_DATA / "benchmark_assembled.csv")

# Effective soil DT50 for the Persistence ground truth: laboratory @20 C where available,
# otherwise the PPDB/VSDB "typical" field (fallback). coumaphos (VSDB report 181) contributes
# a typical-only value; 21 further compounds gain a typical value under this fallback.
a["DT50_soil_eff"] = a["DT50_soil_d__value"].fillna(a["DT50_soil_d__typical_value"])

def metrics(pred, truth):
    pred = pred.astype(bool); truth = truth.astype(bool)
    TP = int((pred & truth).sum()); FP = int((pred & ~truth).sum())
    FN = int((~pred & truth).sum()); TN = int((~pred & ~truth).sum())
    n = TP + FP + FN + TN
    prec = TP / (TP + FP) if TP + FP else np.nan
    rec  = TP / (TP + FN) if TP + FN else np.nan
    spec = TN / (TN + FP) if TN + FP else np.nan
    acc  = (TP + TN) / n if n else np.nan
    f1   = 2 * prec * rec / (prec + rec) if (prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else np.nan
    bal  = np.nanmean([rec, spec])
    den  = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc  = (TP*TN - FP*FN) / den if den else np.nan
    return dict(n=n, pos_truth=TP+FN, pos_pred=TP+FP, TP=TP, FP=FP, FN=FN, TN=TN,
                precision=prec, recall=rec, specificity=spec, accuracy=acc,
                F1=f1, balanced_acc=bal, MCC=mcc)

# measured-only BCF mask
bcf_meas = a["BCF_Lkg__meas_or_est"] == "measured"

CLASSES = {
    "Water Soluble": dict(
        rule="cls_WaterSoluble",
        endpoint="WaterSolubility_mgL__value",
        gt=lambda s: s >= 1000, gt_desc="water solubility >= 1000 mg/L (soluble)",
        mask=a["WaterSolubility_mgL__value"].notna()),
    "Aquatic Bioavailable": dict(
        rule="cls_AquaticBioavail",
        endpoint="LogKow__value",
        gt=lambda s: (s > 0.95) & (s <= 3.05), gt_desc="measured LogKow in (0.95, 3.05]",
        mask=a["LogKow__value"].notna()),
    "Bioconcentration Risk": dict(
        rule="cls_BioconcRisk",
        endpoint="BCF_Lkg__value",
        gt=lambda s: s >= 2000, gt_desc="measured BCF >= 2000 L/kg (REACH B)",
        mask=a["BCF_Lkg__value"].notna() & bcf_meas),
    "Persistence": dict(
        rule="cls_Persistence",
        endpoint="DT50_soil_eff",
        gt=lambda s: s > 60, gt_desc="soil DT50 > 60 d (lab@20 C, else typical; FOCUS persistent)",
        mask=a["DT50_soil_eff"].notna()),
    "Soil Mobility": dict(
        rule="cls_SoilMobility",
        endpoint="Koc_Lkg__value",
        gt=lambda s: s < 500, gt_desc="Koc < 500 mL/g (mobile, McCall)",
        mask=a["Koc_Lkg__value"].notna()),
}

rows = []
for name, cfg in CLASSES.items():
    m = cfg["mask"]
    sub = a[m]
    truth = cfg["gt"](sub[cfg["endpoint"]])
    pred = sub[cfg["rule"]]
    r = metrics(pred, truth)
    r["class"] = name; r["ground_truth"] = cfg["gt_desc"]
    rows.append(r)

res = pd.DataFrame(rows).set_index("class")
cols = ["n", "pos_truth", "pos_pred", "TP", "FP", "FN", "TN",
        "precision", "recall", "specificity", "accuracy", "F1", "balanced_acc", "MCC", "ground_truth"]
res = res[cols]
res.to_csv(_DATA / "rule_validation_metrics.csv")

# Bounded Brier of each deterministic 0/1 rule vs the measured ground truth, with the
# always-predict-base-rate reference. For a 0/1 rule the Brier equals the misclassification rate.
brier_rows = []
for name, cfg in CLASSES.items():
    sub = a[cfg["mask"]]
    truth = cfg["gt"](sub[cfg["endpoint"]]).astype(int).values
    pred = sub[cfg["rule"]].astype(int).values.astype(float)
    p = truth.mean()
    brier_rows.append(dict(
        **{"class": name}, n=len(truth),
        brier_rule=brier_score_loss(truth, pred),
        error_rate=float((pred != truth).mean()),
        brier_baserate=(brier_score_loss(truth, np.full_like(truth, p, dtype=float)) if 0 < p < 1 else 0.0)))
pd.DataFrame(brier_rows).to_csv(_DATA / "rule_brier_vs_benchmark.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("=== EFTE rule vs measured ground truth ===\n")
show = res[["n","pos_truth","pos_pred","TP","FP","FN","TN","precision","recall","specificity","F1","MCC"]].copy()
for c in ["precision","recall","specificity","F1","MCC"]:
    show[c] = show[c].map(lambda x: "  n/a" if pd.isna(x) else f"{x:5.2f}")
print(show.to_string())
print("\nground-truth definitions:")
for name, cfg in CLASSES.items():
    print(f"  {name:22s}: {cfg['gt_desc']}")

# threshold sensitivity for the endpoints with obvious threshold choices
print("\n=== threshold sensitivity ===")
def sens(name, rule, endpoint, mask, thresholds, direction):
    sub = a[mask]
    pred = sub[rule]
    print(f"\n{name} (rule positives in-sample = {int(pred.sum())}, n={len(sub)}):")
    for th in thresholds:
        truth = sub[endpoint] >= th if direction == "ge" else sub[endpoint] < th if direction=="lt" else sub[endpoint] > th
        r = metrics(pred, truth)
        pr = "n/a" if pd.isna(r["precision"]) else f"{r['precision']:.2f}"
        rc = "n/a" if pd.isna(r["recall"]) else f"{r['recall']:.2f}"
        mc = "n/a" if pd.isna(r["MCC"]) else f"{r['MCC']:.2f}"
        print(f"   thr={th:>7}: truth_pos={r['pos_truth']:2d}  TP={r['TP']} FP={r['FP']} FN={r['FN']}  prec={pr} rec={rc} MCC={mc}")

sens("Water Soluble (WS mg/L)", "cls_WaterSoluble", "WaterSolubility_mgL__value",
     a["WaterSolubility_mgL__value"].notna(), [100,1000,10000], "ge")
sens("Bioconcentration (BCF measured)", "cls_BioconcRisk", "BCF_Lkg__value",
     a["BCF_Lkg__value"].notna() & bcf_meas, [500,1000,2000], "ge")
sens("Persistence (soil DT50 d)", "cls_Persistence", "DT50_soil_d__value",
     a["DT50_soil_d__value"].notna(), [60,100,180], "gt")
sens("Soil Mobility (Koc mL/g)", "cls_SoilMobility", "Koc_Lkg__value",
     a["Koc_Lkg__value"].notna(), [150,500,1000], "lt")

if __name__ == "__main__":
    pass
