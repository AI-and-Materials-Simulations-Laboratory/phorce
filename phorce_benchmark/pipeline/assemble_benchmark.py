"""
assemble_benchmark_v2.py — fill Benchmark 0 from BOTH PPDB source formats.

  * 33 ZIP archives (per-page JPEG + OCR)      -> ppdb_extract.extract
  * 27 print-to-PDF monographs                 -> ppdb_extract_pdf.extract
The real PDFs expose the PPDB quality code inline, so those rows (and the preserved
fenthion exemplar) carry per-value QC codes in the citation; the ZIP rows do not
(codes are detached in OCR) and are left for a targeted QC-code pass.

Excludes terbutryn (886-50-0; a triazine, not a benchmark compound). Leaves
coumaphos (56-72-4) empty and flagged — no source document was provided.

Outputs: benchmark_assembled.csv, benchmark_audit.csv, dt50_lab_vs_typical.csv
"""
import os
import re
import pandas as pd
import ppdb_extract as zx
import ppdb_extract_pdf as px

PROJ = "/mnt/project"
EST_KW = re.compile(r"calculat|estimat|predicted|QSAR", re.I)
CITATION = "PPDB (Lewis et al. 2016)"

FIELDS = {
    "WaterSolubility_mgL": dict(unit="mg/L", method="water 20C pH7", species=None),
    "BCF_Lkg":             dict(unit="L/kg", method=None, species="fish whole body"),
    "Koc_Lkg":             dict(unit="mL/g=L/kg", method="Koc", species=None),
    "DT50_soil_d":         dict(unit="days", method="lab 20C aerobic", species=None),
    "DT50_water_d":        dict(unit="days", method="water-sediment", species=None),
    "LogKow":              dict(unit="log", method="Kow pH7 20C"),
}

# base filename -> (CSV PesticideName). Verified via internal CAS reconciliation.
# fenthion is present but preserved (skip-fill). terbutryn deliberately excluded.
ZIP_MAP = {
    "Acephate": "acephate", "Bensulide_Ref__R_4461": "bensulide",
    "Chlorpyrifos_Ref__OMS_971": "chlorpyrifos",
    "Chlorpyrifosmethyl_Ref__OMS_1155": "chlorpyrifos-methyl",
    "DemetonSmethyl": "demeton-S-methyl", "Diazinon_Ref__OMS_469": "diazinon",
    "Dichlorvos_Ref__OMS_14": "dichlorvos", "Dicrotophos_Ref__OMS_253": "dicrotophos",
    "Dimethoate_Ref__OMS_94": "dimethoate", "Disulfoton_Ref__ENT_23347": "disulfoton",
    "EPN_Ref__OMS_219": "EPN", "Ethoprophos": "ethoprophos",
    "Fenitrothion_Ref__OMS_43": "fenitrothion", "Fenthion_Ref__OMS_2": "fenthion",
    "Glyphosate_Ref__MON_0573": "glyphosate", "Malathion_Ref__OMS_1": "malathion",
    "Mecarbam_Ref__MC_474": "mecarbam", "Methamidophos_Ref__ENT_27396": "methamidophos",
    "Methidathion_Ref__ENT_27193_": "methidathion", "Mevinphos_Ref__ENT_22374": "mevinphos",
    "Monocrotophos_Ref__ENT_27129": "monocrotophos", "Naled_Ref__OMS_75": "naled",
    "Oxydemetonmethyl_Ref__ENT_24964": "oxydemeton-methyl",
    "Parathion_Ref__OMS_19": "parathion", "Parathionmethyl_Ref__OMS_213": "methyl parathion",
    "Phorate_Ref__ENT_24042": "phorate", "Phosalone_Ref__ENT_27163": "phosalone",
    "Phosmet_Ref__OMS_232": "phosmet", "Phosphamidon_Ref__OMS_1325": "phosphamidon",
    "Terbufos_Ref__AC_92100": "terbufos", "Tetrachlorvinphos_Ref__SD_8447": "tetrachlorvinphos",
    "Tribufos": "tribufos", "Trichlorfon_Ref__OMS_800": "trichlorfon",
}
PDF_MAP = {
    "Azinphosethyl_Ref__BAY_16259": "azinphos-ethyl", "Azinphosmethyl": "azinphos-methyl",
    "Cadusafos_Ref__FMC_67825": "cadusafos", "Chlorfenvinphos_Ref__OMS_166": "chlorfenvinphos",
    "Dialifos_Ref__AC_14503": "dialifos", "Edifenphos_Ref__BAY_78418": "edifenphos",
    "Fonofos_Ref__N2790": "fonofos", "Formothion_Ref__SAN_6913I": "formothion",
    "Fosetyl": "fosetyl", "Fosthiazate_Ref__IKI_1145": "fosthiazate",
    "Glufosinate_Ref__HOE_00661": "glufosinate", "Heptenophos_Ref__OMS_1845": "heptenophos",
    "Iprobenfos": "iprobenfos", "Isofenphos_Ref__BAY_SRA_12869": "isofenphos",
    "Omethoate_Ref__ENT_25776": "omethoate", "Pirimiphosethyl_Ref__PP211": "pirimiphos-ethyl",
    "Pirimiphosmethyl_Ref__OMS_1424": "pirimiphos-methyl", "Profenofos_Ref__OMS_2004": "profenofos",
    "Propetamphos_Ref__OMS_1502": "propetamphos", "Prothiofos_Ref__OMS_2006": "prothiofos",
    "Pyrazophos_Ref__HOE_2873": "pyrazophos", "Pyridafenthion_Ref__ENT_23968": "pyridaphenthion",
    "Quinalphos_Ref__ENT_27397": "quinalphos", "Tebupirimfos_Ref__BAY_MAT_7484": "tebupirimfos",
    "Thiometon_Ref__SAN_230": "thiometon", "Triazophos_Ref__HOE_002960_": "triazophos",
    "Vamidothion_Ref__ENT_26613": "vamidothion",
}


def meas_or_est(raw):
    return "estimated" if raw and EST_KW.search(raw) else "measured"


def fill(df, idx, name, rec_all, has_qc, base, fmt, audit):
    row = idx[name]
    for ep, const in FIELDS.items():
        rec = rec_all.get(ep)
        if not rec or rec.get("value") is None:
            continue
        val, raw = rec["value"], rec.get("raw", "")
        qc = rec.get("qc")
        df.at[row, f"{ep}__value"] = val
        df.at[row, f"{ep}__unit"] = const["unit"]
        if const.get("method") is not None:
            df.at[row, f"{ep}__method"] = const["method"]
        if const.get("species") is not None:
            df.at[row, f"{ep}__species"] = const["species"]
        df.at[row, f"{ep}__meas_or_est"] = meas_or_est(raw)
        df.at[row, f"{ep}__db_source"] = "PPDB"

        cite = CITATION
        if has_qc and qc:
            cite += f"; QC {qc}"
        if ep == "DT50_soil_d":
            typ = rec_all.get("DT50_soil_typical")
            if typ and typ.get("value") is not None:
                cite += f"; typical {typ['value']:g}"
        df.at[row, f"{ep}__primary_citation"] = cite

        audit.append(dict(compound=name, endpoint=ep, value=val,
                          meas_or_est=meas_or_est(raw), qc=qc or "",
                          source_file=base + ".pdf", format=fmt,
                          page=rec.get("page", ""), line=rec.get("line", ""),
                          raw_line=raw))


def main():
    df = pd.read_csv("benchmark_template_filled.csv")
    idx = {n: i for i, n in enumerate(df.PesticideName)}
    audit, dt50 = [], []

    def record_dt50(name, rec_all, note=""):
        lab = rec_all.get("DT50_soil_d"); typ = rec_all.get("DT50_soil_typical")
        lv = lab["value"] if lab and lab.get("value") is not None else None
        tv = typ["value"] if typ and typ.get("value") is not None else None
        if lv is None and tv is not None and not note:
            note = "lab@20C MISSING - only typical available"
        dt50.append(dict(compound=name, lab_20C=lv, typical=tv,
                        diff=(None if lv is None or tv is None else round(lv - tv, 3)), note=note))

    # ZIP compounds (QC omitted)
    for base, name in ZIP_MAP.items():
        rec = zx.extract(os.path.join(PROJ, base + ".pdf"))
        if name == "fenthion":
            record_dt50(name, rec, note="exemplar (hand-authored, preserved)")
            continue
        fill(df, idx, name, rec, has_qc=False, base=base, fmt="zip", audit=audit)
        record_dt50(name, rec)

    # real-PDF compounds (QC captured inline)
    for base, name in PDF_MAP.items():
        rec = px.extract(os.path.join(PROJ, base + ".pdf"))
        fill(df, idx, name, rec, has_qc=True, base=base, fmt="pdf", audit=audit)
        record_dt50(name, rec)

    df.to_csv("benchmark_assembled.csv", index=False)
    pd.DataFrame(audit).to_csv("benchmark_audit.csv", index=False)
    pd.DataFrame(dt50).sort_values("compound").to_csv("dt50_lab_vs_typical.csv", index=False)

    filled_rows = sorted(set(a["compound"] for a in audit))
    print(f"Filled {len(audit)} values across {len(filled_rows)} compounds (+fenthion preserved).")
    print("coumaphos (56-72-4): left EMPTY - no source document.")
    print("terbutryn (886-50-0): EXCLUDED - not a benchmark compound.")
    return df


if __name__ == "__main__":
    main()
