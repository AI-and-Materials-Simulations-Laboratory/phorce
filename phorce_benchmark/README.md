# PHORCE Benchmark 0 — external validation of the P-rule against measured pesticide fate

This package externally validates the EFTE "P-rule" descriptor classifiers against **measured**
environmental-fate data for **61 organophosphorus (OP) pesticides**, curated from the Pesticide
Properties Database (PPDB), with coumaphos sourced from the AERU Veterinary Substances DataBase
(VSDB). It exists to answer the circularity critique directly: the rules are tested against data
they were never fit to.

## What's here

```
extract/     ppdb_extract.py       parse PPDB ZIP/OCR monograph exports
             ppdb_extract_pdf.py   parse PPDB print-to-PDF monographs (text layer + quality codes)
pipeline/    assemble_benchmark.py unify both extractors → benchmark_assembled.csv + audit
             validate_rules.py     rule vs measured ground truth → per-class metrics + Brier CSV
             compute_brier.py      bounded Brier (guards the >1 error from the submitted tables)
             make_figures.py       regenerate both figures/*.png from the data artifacts
data/        benchmark_assembled.csv  61 compounds × 92 columns (identity, descriptors, rule flags, 6 measured endpoints + DT50 typical)
             benchmark_audit.csv      every extracted value → source page/line (traceability)
             dt50_lab_vs_typical.csv  lab@20 °C vs typical soil DT50
             rule_validation_metrics.csv, rule_brier_vs_benchmark.csv, dash_data.json
```

## Reproduce

```bash
python pipeline/assemble_benchmark.py     # rebuild benchmark_assembled.csv + audit from PPDB exports
python pipeline/validate_rules.py         # recompute per-class metrics + bounded Brier CSV
python pipeline/make_figures.py           # regenerate figures/*.png from the data
```

Requires `pandas`, `numpy`, `scikit-learn`, `matplotlib`. The extractors additionally use
`pdftotext` (poppler) for the print-to-PDF monographs. The validation, figure, and dashboard scripts
resolve paths relative to the package root, so they can be run from anywhere; `assemble_benchmark.py`
is the raw-rebuild step and expects the (unshipped) PPDB exports in the working directory.

## Notes / caveats

- **BCF ground truth uses measured values only.** 17 of 53 BCF values in the benchmark are PPDB
  *estimates* and are flagged (`BCF_Lkg__meas_or_est == "estimated"`) and excluded from the BCF label.
- **DT50** primary value is lab @ 20 °C; where no lab value exists the "typical" field is used as a
  fallback for the Persistence ground truth (`DT50_soil_eff` in `validate_rules.py`; the raw typical
  is stored in `DT50_soil_d__typical_value` and `dt50_lab_vs_typical.csv`). This raises the
  Persistence evaluation from n=38 to n=60; the rule still flags no OP as persistent (recall 0.00),
  so the verdict is unchanged and the "no predictive value on OPs" finding is reinforced.
- **coumaphos (CAS 56-72-4)** is now sourced from the AERU **Veterinary Substances DataBase**
  (VSDB report 181, https://sitem.herts.ac.uk/aeru/vsdb/Reports/181.htm) — it is a livestock
  ectoparasiticide outside PPDB's plant-protection scope. Water solubility (1.5 mg/L), log Kow
  (3.86), Koc (18 000 mL/g) and a typical soil DT50 (152 d) are populated; BCF and a lab@20 °C
  soil DT50 remain unreported. All 61 compounds now carry a source document.

