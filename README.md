# Loan Default Scorecard — Production-Style Scoring Pipeline

A full end-to-end credit scorecard system replicating industry-standard retail banking methodology.  
Built on the **Give Me Some Credit** dataset (Kaggle).

---

## Dataset Download

1. Go to: **https://www.kaggle.com/datasets/brycecf/give-me-some-credit**
2. Click **Download** (you need a free Kaggle account)
3. Unzip the downloaded file
4. Place **`cs-training.csv`** and **`cs-test.csv`** into the `data/` folder

---

## Project Structure

```
loan_scorecard/
├── data/                        # Raw Kaggle CSV files go here
├── src/
│   ├── 01_data_quality.py       # SQL-based data validation layer
│   ├── 02_feature_engineering.py# WoE binning & feature prep
│   ├── 03_train_model.py        # XGBoost training + scorecard scaling
│   ├── 04_monitoring.py         # PSI drift detection & stability analysis
│   ├── 05_reporting.py          # Excel reports + Gini trend charts
│   ├── 06_challenger_model.py   # Champion vs Challenger evaluation
│   └── utils.py                 # Shared helpers
├── sql/
│   └── data_quality_checks.sql  # Standalone SQL validation scripts
├── outputs/
│   ├── charts/                  # Matplotlib PNGs
│   ├── excel/                   # openpyxl management reports
│   └── models/                  # Saved XGBoost & LR models
├── tests/
│   └── test_pipeline.py         # Unit tests
├── notebooks/
│   └── EDA.ipynb                # Exploratory Data Analysis
├── requirements.txt
├── run_pipeline.py              # ONE-CLICK full pipeline runner
└── README.md
```

---

## Setup Instructions (Step by Step)

### Step 1 — Install Python
Make sure you have **Python 3.9+** installed.  
Check: `python --version`

### Step 2 — Clone / Download this project
```bash
# If using git:
git clone <your-repo-url>
cd loan_scorecard

# Or just unzip the folder and cd into it1
```

### Step 3 — Create a virtual environment
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Add the dataset
Place `cs-training.csv` and `cs-test.csv` into the `data/` folder.

### Step 6 — Run the full pipeline
```bash
python run_pipeline.py
```

This runs all 6 steps automatically in order. You can also run each step individually:

```bash
python src/01_data_quality.py        # Data validation
python src/02_feature_engineering.py # Feature prep + WoE
python src/03_train_model.py         # Train + build scorecard
python src/04_monitoring.py          # PSI drift monitoring
python src/05_reporting.py           # Excel report generation
python src/06_challenger_model.py    # Champion vs Challenger
```

---

## Output Summary

| Output | Location | Description |
|--------|----------|-------------|
| Scorecard table | `outputs/scorecard_points.csv` | Integer points per feature bin |
| Model files | `outputs/models/` | Saved XGBoost & Logistic Regression |
| Score distribution chart | `outputs/charts/score_distribution.png` | Histogram 300–850 |
| PSI drift report | `outputs/charts/psi_report.png` | Population Stability Index |
| Gini trend chart | `outputs/charts/gini_trend.png` | Monthly Gini coefficients |
| Excel report | `outputs/excel/model_report.xlsx` | Multi-tab management report |
| Validation report | `outputs/excel/challenger_report.xlsx` | Champion vs Challenger |
| Data quality log | `outputs/data_quality_results.csv` | Pass/fail per check |

---

## Key Methodology

- **Scorecard scaling**: log-odds → points using `PDO=20, base score=600, base odds=1:1`
- **PSI thresholds**: Green < 0.1 | Amber 0.1–0.25 | Red > 0.25
- **Gini / KS**: Evaluated on hold-out test set
- **Champion model**: XGBoost-based scorecard
- **Challenger model**: Logistic Regression on WoE features

---

## Requirements
See `requirements.txt`. Key packages:
- `xgboost`, `scikit-learn`, `pandas`, `numpy`
- `openpyxl`, `matplotlib`, `seaborn`
- `duckdb` (SQL layer), `optbinning` (WoE)
