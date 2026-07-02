"""
ppdb_extract.py  —  Benchmark 0 endpoint extractor for the PHORCE/P1M manuscript.

Reads PPDB monograph exports (ZIP archives of per-page JPEG + OCR .txt + manifest.json)
and extracts the six environmental-fate / physicochemical endpoints used in Benchmark 0,
following the conventions established by the hand-authored fenthion exemplar:

  WaterSolubility_mgL : "Solubility - In water at 20 C at pH 7 (mg l-1)"  (value on line AFTER label)
  LogKow              : "Log P <value>"                                    (value same line; sign preserved)
  Koc_Lkg             : "Koc (mL g-1) <value>"                             (value same line)
  BCF_Lkg             : "BCF (l kg-1) <value>"                             (value same line)
  DT50_soil_d         : "DT50 (lab at 20 C)"  -> lab@20C aerobic primary   (value on next content line)
                        "DT50 (typical) <value>" captured for the lab-vs-typical decision
  DT50_water_d        : "Water-sediment DT50 (days) <value>"              (value same line)

Every extracted value is returned with its source page and line so the audit trail
(value -> raw OCR line) can be regenerated for manuscript defensibility.
"""

import os
import re
import zipfile

# ---- filename base -> CSV PesticideName (verified against internal doc titles) ----
FILE_TO_NAME = {
    "Acephate": "acephate",
    "Bensulide_Ref__R_4461": "bensulide",
    "Chlorpyrifos_Ref__OMS_971": "chlorpyrifos",
    "Chlorpyrifosmethyl_Ref__OMS_1155": "chlorpyrifos-methyl",
    "DemetonSmethyl": "demeton-S-methyl",
    "Diazinon_Ref__OMS_469": "diazinon",
    "Dichlorvos_Ref__OMS_14": "dichlorvos",
    "Dicrotophos_Ref__OMS_253": "dicrotophos",
    "Dimethoate_Ref__OMS_94": "dimethoate",
    "Disulfoton_Ref__ENT_23347": "disulfoton",
    "EPN_Ref__OMS_219": "EPN",
    "Ethoprophos": "ethoprophos",
    "Fenitrothion_Ref__OMS_43": "fenitrothion",
    "Fenthion_Ref__OMS_2": "fenthion",
    "Glyphosate_Ref__MON_0573": "glyphosate",
    "Malathion_Ref__OMS_1": "malathion",
    "Mecarbam_Ref__MC_474": "mecarbam",
    "Methamidophos_Ref__ENT_27396": "methamidophos",
    "Methidathion_Ref__ENT_27193_": "methidathion",
    "Mevinphos_Ref__ENT_22374": "mevinphos",
    "Monocrotophos_Ref__ENT_27129": "monocrotophos",
    "Naled_Ref__OMS_75": "naled",
    "Oxydemetonmethyl_Ref__ENT_24964": "oxydemeton-methyl",
    "Parathion_Ref__OMS_19": "parathion",
    "Parathionmethyl_Ref__OMS_213": "methyl parathion",
    "Phorate_Ref__ENT_24042": "phorate",
    "Phosalone_Ref__ENT_27163": "phosalone",
    "Phosmet_Ref__OMS_232": "phosmet",
    "Phosphamidon_Ref__OMS_1325": "phosphamidon",
    "Terbufos_Ref__AC_92100": "terbufos",
    "Tetrachlorvinphos_Ref__SD_8447": "tetrachlorvinphos",
    "Tribufos": "tribufos",
    "Trichlorfon_Ref__OMS_800": "trichlorfon",
}

# A numeric token: optional sign, digits, optional decimals. (Scientific "x 10-0n"
# does not occur for these six endpoints in PPDB fate tables.)
NUM = re.compile(r"(-?\d+(?:\.\d+)?)")

# "no data" markers PPDB uses in the value column
NODATA = {"-", "--", ""}


def load_pages(zip_path):
    """Return list of (page_number, line_index_1based, text) for every OCR line."""
    lines = []
    with zipfile.ZipFile(zip_path) as z:
        txts = [n for n in z.namelist() if n.endswith(".txt")]
        # sort numerically by page number
        txts.sort(key=lambda n: int(re.match(r"(\d+)", os.path.basename(n)).group(1)))
        for n in txts:
            page = int(re.match(r"(\d+)", os.path.basename(n)).group(1))
            body = z.read(n).decode("utf-8", "replace")
            for i, raw in enumerate(body.splitlines(), start=1):
                lines.append((page, i, raw.rstrip("\r")))
    return lines


def _first_number(s):
    m = NUM.search(s)
    return float(m.group(1)) if m else None


def _value_same_line(lines, anchor_regex):
    """Endpoints where the value sits on the same line as the label."""
    pat = re.compile(anchor_regex)
    for page, ln, txt in lines:
        if pat.search(txt):
            tail = pat.sub("", txt, count=1)
            val = _first_number(tail)
            if val is not None:
                return dict(value=val, page=page, line=ln, raw=txt.strip())
            # label present but value is a dash / absent
            if tail.strip().lstrip("-").strip() == "" or tail.strip() in NODATA:
                return dict(value=None, page=page, line=ln, raw=txt.strip(), nodata=True)
    return None


def _value_next_line(lines, anchor_regex):
    """Endpoints whose value is on a following content line (multi-line label).

    After the anchor line, walk forward and return the FIRST line whose leading
    token is numeric (the value) or a no-data dash. Lines whose leading token is
    non-numeric label text (e.g. "pH 7 (mg l-1)", "C)") are treated as label
    continuation and skipped. Keying on the leading token (rather than skipping any
    line that merely mentions "C"/"pH") avoids discarding real value lines that
    carry a temperature qualifier, e.g. "1000000 at 25 C High".
    """
    pat = re.compile(anchor_regex)
    for idx, (page, ln, txt) in enumerate(lines):
        if pat.search(txt):
            for page2, ln2, txt2 in lines[idx + 1: idx + 7]:
                t = txt2.strip()
                if t == "":
                    continue
                first = t.split()[0]
                if re.match(r"^-?\d", first):                    # value line
                    return dict(value=_first_number(t), page=page2, line=ln2, raw=t)
                if first in {"-", "--", "---"} or set(t) <= set("- "):
                    return dict(value=None, page=page2, line=ln2, raw=t, nodata=True)
                # else: label-continuation line -> keep scanning
    return None


def extract(zip_path):
    """Extract the six endpoints (+ typical DT50 soil) with provenance."""
    lines = load_pages(zip_path)
    out = {}

    # 1) Water solubility: value on the line after "pH 7 (mg l-1)"
    out["WaterSolubility_mgL"] = _value_next_line(lines, anchor_regex=r"In water at 20")

    # 2) Log Kow: "Log P <value>" (anchor on 'Log P' to avoid the bare 'P' partition line)
    out["LogKow"] = _value_same_line(lines, r"Log P\b")

    # 3) Koc
    out["Koc_Lkg"] = _value_same_line(lines, r"Koc \(")

    # 4) BCF
    out["BCF_Lkg"] = _value_same_line(lines, r"BCF \(")

    # 5) DT50 soil — PRIMARY = lab at 20 C (value on next content line)
    out["DT50_soil_d"] = _value_next_line(lines, anchor_regex=r"DT.{0,3}\s*\(lab at 20")
    # 5b) DT50 soil typical (same line) — for the lab-vs-typical benchmark decision
    out["DT50_soil_typical"] = _value_same_line(lines, r"DT.{0,3}\s*\(typical\)")

    # 6) DT50 water — water-sediment (value same line)
    out["DT50_water_d"] = _value_same_line(lines, r"Water-sediment DT.{0,3}\s*\(days\)")

    return out


if __name__ == "__main__":
    import sys, json
    r = extract(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items()}, indent=2, default=str))
