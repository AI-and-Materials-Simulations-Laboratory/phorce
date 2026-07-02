# PHORCE: PHysicochemical ORganization and Classification for Environmental compounds

A comprehensive data pipeline for preprocessing, labeling, and analyzing chemical compound data from PubChem, with a focus on environmental domain classification.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Project Structure](#project-structure)

---

## Overview

PHORCE processes large chemical compound datasets (1M+ compounds) through a multi-stage pipeline:

1. **Preprocessing** - Clean, validate, and standardize raw chemical data
2. **Feature Engineering** - Create derived features from existing properties
3. **Domain Labeling** - Classify compounds into environmental domains using keyword matching
4. **Subset Creation** - Create physicochemical subsets based on property thresholds
5. **Analysis** - Train classifiers (XGBoost) and extract interpretable decision rules for each domain/subset

The pipeline is designed for environmental chemistry research, enabling researchers to identify compounds relevant to soil science, aquatic toxicology, and agricultural applications.

---

## Features

- **Configurable Pipeline** - All settings controlled via a single JSON configuration file
- **Robust Preprocessing** - Handles missing values, duplicates, outliers, and data type enforcement
- **Chemical Validation** - Optional SMILES/InChI validation using RDKit
- **Domain Classification** - Keyword-based labeling for environmental domains:
  - **Soil-related** - Compounds relevant to soil science and geochemistry
  - **Water-related** - Aquatic and marine compounds
  - **Crop-related** - Agricultural and pesticide compounds
- **Physicochemical Subsets** - Property-based classification:
  - Water Soluble, Aquatic Bioavailable, Bioconcentration Risk, Persistence, Soil Mobility
- **Machine Learning Analysis** - Gradient-boosted (XGBoost) and decision-tree classifiers with cross-validation
- **Rule Extraction** - Human-readable decision rules from trained models
- **Comprehensive Reporting** - JSON reports and visualizations at each step

---

## Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip

### Setup with Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/AI-and-Materials-Simulations-Laboratory/phorce.git
cd phorce

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate phorce
```

### Setup with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn tqdm

# Optional: Install RDKit for chemical validation
pip install rdkit
```

---

## Quick Start

### Run the Full Pipeline

```bash
# 1. Preprocess raw data
python scripts/preprocess_data.py

# 2. Engineer features
python scripts/engineer_features.py

# 3. Label domains
python scripts/label_domains.py

# 4. Create subsets
python scripts/create_domain_subsets.py

# 5. Run analysis (choose one or more)
python scripts/analyze_processed.py      # Analyze preprocessed data
python scripts/analyze_domains.py        # Analyze domain classifications
python scripts/analyze_subdomains.py     # Analyze subsets
```

### Run with Custom Configuration

```bash
python scripts/preprocess_data.py --config my_config.json
python scripts/preprocess_data.py --input data.csv --output processed.csv
```

---

## Pipeline Steps

### Step 1: Data Preprocessing (`preprocess_data.py`)

Cleans and standardizes raw chemical data:

| Operation | Description |
|-----------|-------------|
| Standardize Missing Values | Convert NA, null, None, ".", "-", "?" → NaN |
| Drop High-Missing Columns | Remove columns with >95% missing values |
| Drop High-Missing Rows | Remove rows with >50% missing values |
| Drop Required Missing | Remove rows missing required columns (cid, canonicalsmiles) |
| Drop Numeric Missing | Remove rows with missing numeric values |
| Clean Text | Strip whitespace from text columns |
| Clean Numeric | Remove negative values from physical properties |
| Enforce Types | Convert columns to correct data types |

**Input:** `P1M.csv` (raw PubChem data)  
**Output:** `data/processed/P1M_preprocessed.csv`

---

### Step 2: Feature Engineering (`engineer_features.py`)

Creates derived features from existing columns:

```json
{
  "name": "TotalHBond",
  "operation": "add",
  "columns": ["hbonddonor", "hbondacc"],
  "description": "Total hydrogen bond donors and acceptors"
}
```

**Supported Operations:** add, subtract, multiply, divide, log, sqrt, power, clip, bin, conditional, formula

**Input:** `data/processed/P1M_preprocessed.csv`  
**Output:** `data/engineered/P1M_engineered.csv`

---

### Step 3: Domain Labeling (`label_domains.py`)

Classifies compounds into environmental domains based on keyword matching in text fields:

| Domain | Keywords (examples) |
|--------|---------------------|
| **soil_related** | soil, sediment, humic, biochar, rhizosphere, leaching, sorption |
| **water_related** | aquatic, marine, daphnia, fish, groundwater, toxicity, ocean |
| **crop_related** | crop, plant, herbicide, pesticide, fertilizer, agriculture |

**Text fields searched:** meshheadings, annotation, cmpdsynonym, annothits, aids, sidsrcname

**Input:** `data/engineered/P1M_engineered.csv`  
**Output:** `data/labeled/domain_labeled_compounds.csv`

---

### Step 4: Subset Creation (`create_domain_subsets.py`)

Creates physicochemical subsets based on property thresholds:

| Subset | Criteria | Description |
|--------|----------|-------------|
| **Water_Soluble** | LogP ≤ 0.95, TPSA > 74.9 | Highly water-soluble compounds |
| **Aquatic_Bioavailable** | LogP 0.95-3.05, HBond ≤ 5.5 | Bioavailable in aquatic environments |
| **Bioconcentration_Risk** | LogP > 3.05, TPSA ≤ 60 | Potential for bioconcentration |
| **Persistence** | Complexity > 600, Rotbonds > 5 | Environmentally persistent |
| **Soil_Mobility** | LogP < 2.5, TPSA < 90, Charge -1 to 1 | Mobile in soil environments |

**Input:** `data/labeled/domain_labeled_compounds.csv`  
**Output:** `data/subsets/P1M_with_subsets.csv`

---

### Step 5: Analysis Scripts

#### `analyze_processed.py`
Exploratory analysis of preprocessed data:
- Feature distributions and statistics
- Correlation heatmaps
- PCA analysis (5 components)
- Box plots

#### `analyze_domains.py`
Classifier analysis (XGBoost) for each environmental domain:
- GridSearchCV hyperparameter tuning
- 5-fold cross-validation
- Feature importance ranking
- Rule extraction
- ROC curves and confusion matrices

#### `analyze_subdomains.py`
Classifier analysis (XGBoost) for physicochemical subsets:
- Same analysis as domains
- Parent domain filtering (e.g., Water_Soluble within water_related)
- Threshold validation against original criteria

---

## Configuration

All pipeline settings are controlled via `preprocessing_config.json`:

```json
{
  "input_file": "P1M.csv",
  "output_file": "data/processed/P1M_preprocessed.csv",
  
  "column_handling": {
    "required_columns": ["cid", "canonicalsmiles", "meshheadings"],
    "keep_columns": ["cid", "mw", "xlogp", "polararea", ...]
  },
  
  "missing_value_handling": {
    "strategy": "drop_rows",
    "missing_threshold_columns": 0.95,
    "missing_threshold_rows": 0.5
  },
  
  "data_type_enforcement": {
    "integer_columns": ["cid", "heavycnt", "hbonddonor", ...],
    "float_columns": ["mw", "xlogp", "polararea", ...],
    "string_columns": ["canonicalsmiles", "inchi", ...]
  },
  
  "domain_labeling": {
    "domains": {
      "soil_related": { "keywords": [...] },
      "water_related": { "keywords": [...] },
      "crop_related": { "keywords": [...] }
    }
  },
  
  "domain_subsets": {
    "subsets": [
      { "name": "Water_Soluble", "conditions": [...] },
      ...
    ]
  }
}
```

See `preprocessing_config.json` for the complete configuration reference.

---

## Output Files

```
data/
├── processed/
│   ├── P1M_preprocessed.csv          # Cleaned data
│   ├── preprocessing_report.json     # Preprocessing statistics
│   └── preprocessing.log             # Processing log
│
├── engineered/
│   ├── P1M_engineered.csv            # Data with derived features
│   └── feature_engineering_report.json
│
├── labeled/
│   ├── domain_labeled_compounds.csv  # All compounds with domain labels
│   ├── soil_related_compounds.csv    # Soil domain subset
│   ├── water_related_compounds.csv   # Water domain subset
│   ├── crop_related_compounds.csv    # Crop domain subset
│   ├── domain_labeling_report.json
│   └── plots/                        # Visualization outputs
│
├── subsets/
│   ├── P1M_with_subsets.csv          # All compounds with subset labels
│   ├── Water_Soluble.csv
│   ├── Aquatic_Bioavailable.csv
│   ├── Bioconcentration_Risk.csv
│   ├── Persistence.csv
│   ├── Soil_Mobility.csv
│   └── subset_report.json
│
└── analysis/
    ├── processed/                    # Preprocessed data analysis
    ├── domains/                      # Domain classifier analysis
    └── subdomains/                   # Subset classifier analysis
```

---

## Project Structure

```
phorce/
├── P1M.csv                           # Input: Raw PubChem data (1M compounds)
├── preprocessing_config.json         # Pipeline configuration
├── environment.yml                   # Conda environment specification
├── pyproject.toml                    # Python project metadata
├── README.md                         # This file
│
├── scripts/
│   ├── preprocess_data.py            # Step 1: Data preprocessing
│   ├── engineer_features.py          # Step 2: Feature engineering
│   ├── label_domains.py              # Step 3: Domain labeling
│   ├── create_domain_subsets.py      # Step 4: Subset creation
│   ├── analyze_processed.py          # Step 5a: Processed data analysis
│   ├── analyze_domains.py            # Step 5b: Domain analysis
│   ├── analyze_subdomains.py         # Step 5c: Subdomain analysis
│   └── analyze_decision_trees.py     # Additional tree analysis
│
├── data/                             # Output data (generated)
│   ├── processed/
│   ├── engineered/
│   ├── labeled/
│   ├── subsets/
│   └── analysis/
│
└── docs/
    └── pipeline_flowchart.md         # Visual pipeline documentation
```

---

## Key Features Tracked

### Numeric Features
| Feature | Description |
|---------|-------------|
| `mw` | Molecular Weight |
| `xlogp` | Partition coefficient (lipophilicity) |
| `polararea` | Topological Polar Surface Area (TPSA) |
| `complexity` | Molecular complexity score |
| `heavycnt` | Heavy atom count |
| `hbonddonor` | Hydrogen bond donor count |
| `hbondacc` | Hydrogen bond acceptor count |
| `rotbonds` | Rotatable bond count |
| `charge` | Formal charge |

### Engineered Features
| Feature | Formula | Description |
|---------|---------|-------------|
| `TotalHBond` | hbonddonor + hbondacc | Total H-bonding capacity |

---

## Documentation

- **Pipeline Flowcharts:** See `docs/pipeline_flowchart.md` for detailed flow diagrams of each script
- **Configuration Reference:** See comments in `preprocessing_config.json`
- **Script Help:** Run any script with `--help` for usage information

---

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

## Authors

- Patrick B. Hogsed
- Brooke K. Mayer
- Yaroslava G. Yingling
- Eric McLamore

---

## Acknowledgments

- PubChem for providing chemical compound data
- RDKit for chemical informatics functionality

- scikit-learn for machine learning capabilities
