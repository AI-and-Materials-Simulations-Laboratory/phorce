"""
ppdb_extract_pdf.py — endpoint extractor for the print-to-PDF PPDB monographs
(the 28 real PDFs added in the second upload; distinct from the ZIP/OCR archives).

Uses `pdftotext -layout` so each table row stays on one line with the value and the
PPDB quality code inline (e.g. "BCF (l kg-1)  75  Q2 Estimated  Low potential").
All six endpoints are same-line here; Log P needs block-aware handling because the
hydrophilic-compound layout staggers the value onto the row-label line.
"""
import re
import subprocess

NUM = re.compile(r"(-?\d+(?:\.\d+)?)")
QC = re.compile(r"\b([A-Z]{1,3}\d)\b")            # H4, A5, DW4, AC4, Q2, ...
EST_KW = re.compile(r"calculat|estimat|predicted|QSAR", re.I)
INTERP = re.compile(r"\b(Low|Moderate|High|Very)\b")


def pdf_lines(path):
    txt = subprocess.run(["pdftotext", "-layout", path, "-"],
                         capture_output=True, text=True).stdout
    return txt.splitlines()


def _num_and_qc(tail):
    """First plain number in tail (skip scientific 'X 10'), plus first QC code after it."""
    # remove scientific-notation partition values so they can't be mistaken for the value
    if re.search(r"X\s*10", tail):
        # keep only the part that is NOT the 'n X 10-0m Calculated' construct
        tail = re.sub(r"-?\d+(?:\.\d+)?\s*X\s*10\S*", " ", tail)
    m = NUM.search(tail)
    if not m:
        return None, None, tail
    val = float(m.group(1))
    after = tail[m.end():]
    q = QC.search(after)
    return val, (q.group(1) if q else None), tail


def _same_line(lines, anchor):
    pat = re.compile(anchor)
    for i, ln in enumerate(lines):
        if pat.search(ln):
            tail = pat.sub("", ln, count=1)
            val, qc, _ = _num_and_qc(tail)
            raw = ln.strip()
            if val is not None:
                return dict(value=val, qc=qc, raw=raw, line=i + 1)
            return dict(value=None, qc=None, raw=raw, line=i + 1, nodata=True)
    return None


LOGP_LINE = re.compile(r"(-?\d+(?:\.\d+)?)\s+([A-Z]{1,3}\d)\b")


def _logp(lines):
    """Find the Log P value within the octanol-water partition block.

    The block has a raw 'P  m X 10-n  Calculated' line (skipped) and the Log P value
    line formatted '<value>  <QC>  <Low/Moderate/High>'. Depending on wrapping the
    value sits either on the 'Log P' line or on the 'coefficient at' line just above
    the 'Log P' label. We locate the 'Octanol-water' block and return the first line
    (excluding the 'X 10 ... Calculated' raw-P line) carrying a signed decimal in
    [-12, 12] followed by a QC code, with an interpretation word present.
    """
    starts = [i for i, ln in enumerate(lines) if "Octanol-water" in ln]
    if not starts:
        starts = [i for i, ln in enumerate(lines) if "Log P" in ln]
    for s in starts:
        for j in range(s, min(len(lines), s + 9)):
            cand = lines[j]
            if "X 10" in cand or "Calculated" in cand:
                continue
            if not INTERP.search(cand):
                continue
            m = LOGP_LINE.search(cand)
            if m:
                v = float(m.group(1))
                if -12 <= v <= 12:
                    return dict(value=v, qc=m.group(2), raw=cand.strip(), line=j + 1)
    # explicit no-data
    for i, ln in enumerate(lines):
        if "Log P" in ln:
            return dict(value=None, qc=None, raw=ln.strip(), line=i + 1, nodata=True)
    return None


def extract(path):
    L = pdf_lines(path)
    out = {}
    out["WaterSolubility_mgL"] = _same_line(L, r"In water at 20")
    out["LogKow"] = _logp(L)
    out["Koc_Lkg"] = _same_line(L, r"Koc \(")
    out["BCF_Lkg"] = _same_line(L, r"BCF \(")
    out["DT50_soil_d"] = _same_line(L, r"DT.{0,3}\s*\(lab at 20")
    out["DT50_soil_typical"] = _same_line(L, r"DT.{0,3}\s*\(typical\)")
    out["DT50_water_d"] = _same_line(L, r"Water-sediment DT.{0,3}\s*\(days\)")
    return out


if __name__ == "__main__":
    import sys, json
    print(json.dumps(extract(sys.argv[1]), indent=2, default=str))
