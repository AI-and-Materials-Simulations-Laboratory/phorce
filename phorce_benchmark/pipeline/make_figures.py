"""make_figures.py — regenerate both benchmark figures from the data artifacts.

Reproduces:
  figures/rule_validation_figure.png     (metrics + confusion breakdown per class)
  figures/benchmark_analysis_panels.png  (6-panel descriptor / endpoint overview)

All values are read from data/benchmark_assembled.csv and data/rule_validation_metrics.csv,
so the figures always reflect the current benchmark (coumaphos included; Persistence uses the
lab@20 C -> typical DT50 fallback). Run:  python pipeline/make_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / "data"
FIG = _ROOT / "figures"
FIG.mkdir(exist_ok=True)

a = pd.read_csv(DATA / "benchmark_assembled.csv")
met = pd.read_csv(DATA / "rule_validation_metrics.csv").set_index("class")
eff_dt50 = a["DT50_soil_d__value"].fillna(a["DT50_soil_d__typical_value"])

# palette
C_TP, C_FN, C_FP, C_TN = "#2e8b57", "#f4a460", "#c0392b", "#d9d9d9"
C_WS, C_BC, C_SM, C_OTHER = "#1a7f8e", "#c0392b", "#b5822e", "#cccccc"
C_PREC, C_REC, C_MCC = "#1f77b4", "#7fc9b0", "#e8543f"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True})


# ============================================================ FIGURE 1
def figure_rule_validation():
    order = ["Water Soluble", "Soil Mobility", "Aquatic Bioavailable",
             "Bioconcentration Risk", "Persistence"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: precision / recall / MCC ---
    y = np.arange(len(order))[::-1]
    h = 0.25
    for j, (col, color, lab) in enumerate([("precision", C_PREC, "Precision"),
                                           ("recall", C_REC, "Recall"),
                                           ("MCC", C_MCC, "MCC")]):
        vals = [met.loc[c, col] for c in order]
        axA.barh(y + (1 - j) * h, [0 if pd.isna(v) else v for v in vals],
                 height=h, color=color, label=lab)
        for yi, v in zip(y + (1 - j) * h, vals):
            axA.text(0.02 if pd.isna(v) else v + 0.02, yi,
                     "n/a" if pd.isna(v) else f"{v:.2f}", va="center", fontsize=9)
    axA.set_yticks(y); axA.set_yticklabels(order)
    axA.set_xlim(-0.1, 1.25); axA.set_xlabel("score")
    axA.set_title("A. EFTE rule vs measured ground truth", fontweight="bold")
    axA.legend(loc="lower right", framealpha=0.9)
    axA.axvline(0, color="k", lw=0.8)

    # --- Panel B: confusion breakdown (stacked) ---
    for yi, c in zip(y, order):
        TP, FN, FP, TN = (int(met.loc[c, k]) for k in ["TP", "FN", "FP", "TN"])
        left = 0
        for val, color in [(TP, C_TP), (FN, C_FN), (FP, C_FP), (TN, C_TN)]:
            if val:
                axB.barh(yi, val, left=left, color=color, height=0.55)
                axB.text(left + val / 2, yi, str(val), va="center", ha="center",
                         fontsize=9, color="black")
            left += val
    axB.set_yticks(y); axB.set_yticklabels(order)
    axB.set_xlabel("compounds (n with measured endpoint)")
    axB.set_title("B. Confusion breakdown per class", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [C_TP, C_FN, C_FP, C_TN]]
    axB.legend(handles, ["TP", "FN", "FP", "TN"], loc="upper center",
               bbox_to_anchor=(0.5, -0.09), ncol=4, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "rule_validation_figure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================ FIGURE 2
def figure_analysis_panels():
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("PHORCE Benchmark 0 — rule validation against measured OP-pesticide fate data",
                 fontsize=15, fontweight="bold")
    (A, B, C), (Dp, E, F) = axes

    # A. descriptor space & rule regions
    def klass(r):
        if r["cls_WaterSoluble"]: return ("Water sol.", C_WS)
        if r["cls_BioconcRisk"]:  return ("Bioconc.", C_BC)
        if r["cls_SoilMobility"]: return ("Soil mob.", C_SM)
        return ("other", C_OTHER)
    seen = {}
    for _, r in a.iterrows():
        name, color = klass(r)
        A.scatter(r["MolLogP"], r["TPSA"], c=color, s=28,
                  edgecolor="none", label=name if name not in seen else None)
        seen[name] = 1
    for x in (0.95, 3.05):
        A.axvline(x, ls=":", c="k", lw=1)
    A.set_xlabel("MolLogP"); A.set_ylabel("TPSA (Å$^2$)")
    A.set_title("A. Descriptor space & rule regions", fontweight="bold")
    A.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # B. Water Soluble: rule vs measured
    ws = a[a["WaterSolubility_mgL__value"].notna()].copy()
    ws["logS"] = np.log10(ws["WaterSolubility_mgL__value"])
    for _, r in ws.iterrows():
        pred = bool(r["cls_WaterSoluble"]); truth = r["WaterSolubility_mgL__value"] >= 1000
        color = C_TP if (pred and truth) else (C_FN if (truth and not pred) else C_OTHER)
        yv = 1 if pred else 0
        B.scatter(r["logS"], yv + np.random.uniform(-0.06, 0.06), c=color, s=26, edgecolor="none")
    B.axvline(np.log10(1000), ls="--", c="k", lw=1)
    B.set_yticks([0, 1]); B.set_yticklabels(["rule −", "rule +"])
    B.set_xlabel("log10 water solubility (mg/L)")
    B.set_title("B. Water Soluble: rule vs measured", fontweight="bold")

    # C. Persistence rule never fires
    cp = a[eff_dt50.notna()].copy(); cp["dt"] = eff_dt50[eff_dt50.notna()]
    persist = cp["dt"] > 60
    C.scatter(cp.loc[~persist, "Complexity"], cp.loc[~persist, "dt"], c=C_OTHER, s=26, edgecolor="none")
    C.scatter(cp.loc[persist, "Complexity"], cp.loc[persist, "dt"], c=C_FP, s=34, edgecolor="none")
    C.axhline(60, ls=":", c="#b5822e", lw=1.2)
    C.axvline(600, ls="--", c="k", lw=1)
    C.set_yscale("log"); C.set_xlabel("Molecular complexity"); C.set_ylabel("soil DT50 (days)")
    C.set_title("C. Persistence rule never fires", fontweight="bold")
    C.text(605, C.get_ylim()[1] * 0.6, "rule needs\ncomplexity>600", fontsize=8, va="top")

    # D. DT50 lab vs typical
    both = a[a["DT50_soil_d__value"].notna() & a["DT50_soil_d__typical_value"].notna()]
    Dp.scatter(both["DT50_soil_d__typical_value"], both["DT50_soil_d__value"],
               c=C_WS, s=28, edgecolor="none")
    mx = float(np.nanmax([both["DT50_soil_d__typical_value"].max(), both["DT50_soil_d__value"].max()]))
    Dp.plot([0, mx], [0, mx], "k--", lw=1)
    Dp.set_xlabel("typical DT50 (d)"); Dp.set_ylabel("lab@20°C DT50 (d)")
    Dp.set_title("D. DT50: lab vs typical", fontweight="bold")

    # E. BCF measured vs PPDB estimate
    bcf = a[a["BCF_Lkg__value"].notna()].copy()
    bcf["logB"] = np.log10(bcf["BCF_Lkg__value"].clip(lower=0.05))
    meas = bcf[bcf["BCF_Lkg__meas_or_est"] == "measured"]["logB"]
    est = bcf[bcf["BCF_Lkg__meas_or_est"] == "estimated"]["logB"]
    bins = np.linspace(-1.2, 3.6, 20)
    E.hist([meas, est], bins=bins, stacked=True, color=[C_TP, "#c8991f"],
           label=[f"measured ({len(meas)})", f"estimated ({len(est)})"])
    E.axvline(np.log10(2000), ls="--", c=C_FP, lw=1.2)
    E.text(np.log10(2000) + 0.05, E.get_ylim()[1] * 0.9, "B≥2000", color=C_FP, fontsize=9)
    E.set_xlabel("log10 BCF (L/kg)"); E.set_ylabel("compounds")
    E.set_title("E. BCF: measured vs PPDB estimate", fontweight="bold")
    E.legend(loc="upper left", fontsize=9)

    # F. Benchmark coverage
    N = len(a)
    cov = [("Water sol.", a["WaterSolubility_mgL__value"].notna().sum()),
           ("log Kow", a["LogKow__value"].notna().sum()),
           ("Koc", a["Koc_Lkg__value"].notna().sum()),
           ("BCF", a["BCF_Lkg__value"].notna().sum()),
           ("Soil DT50", a["DT50_soil_d__value"].notna().sum()),
           ("Water DT50", a["DT50_water_d__value"].notna().sum())]
    labs = [c[0] for c in cov]; vals = [int(c[1]) for c in cov]
    yy = np.arange(len(cov))[::-1]
    F.barh(yy, vals, color=C_WS, height=0.6)
    for yi, v in zip(yy, vals):
        F.text(v + 0.5, yi, f"{v}/{N}", va="center", fontsize=9)
    F.set_yticks(yy); F.set_yticklabels(labs)
    F.set_xlim(0, N + 4); F.set_xlabel("compounds with measured value")
    F.set_title("F. Benchmark coverage", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "benchmark_analysis_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)  # only for the small vertical jitter in panel B
    figure_rule_validation()
    figure_analysis_panels()
    print("Regenerated figures/rule_validation_figure.png and figures/benchmark_analysis_panels.png")
