"""
utils.py — Shared helpers for the Loan Scorecard Pipeline
"""

import os
import pandas as pd
import numpy as np
import logging

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(ROOT, "data")
OUTPUTS_DIR   = os.path.join(ROOT, "outputs")
CHARTS_DIR    = os.path.join(OUTPUTS_DIR, "charts")
EXCEL_DIR     = os.path.join(OUTPUTS_DIR, "excel")
MODELS_DIR    = os.path.join(OUTPUTS_DIR, "models")

for d in [CHARTS_DIR, EXCEL_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Column rename map (Kaggle → clean names) ──────────────────────────────────
RENAME_MAP = {
    "SeriousDlqin2yrs":                          "target",
    "RevolvingUtilizationOfUnsecuredLines":       "revolving_util",
    "age":                                        "age",
    "NumberOfTime30-59DaysPastDueNotWorse":       "dpd_30_59",
    "DebtRatio":                                  "debt_ratio",
    "MonthlyIncome":                              "monthly_income",
    "NumberOfOpenCreditLinesAndLoans":            "open_credit_lines",
    "NumberOfTimes90DaysLate":                    "dpd_90",
    "NumberRealEstateLoansOrLines":               "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse":       "dpd_60_89",
    "NumberOfDependents":                         "dependents",
}

# Feature list (after rename)
FEATURES = [
    "revolving_util", "age", "dpd_30_59", "debt_ratio",
    "monthly_income", "open_credit_lines", "dpd_90",
    "real_estate_loans", "dpd_60_89", "dependents",
]

TARGET = "target"

# ── Scorecard scaling parameters ──────────────────────────────────────────────
BASE_SCORE = 600
BASE_ODDS  = 1.0        # p_good / p_bad at base score (1:1)
PDO        = 20         # Points to Double the Odds
SCORE_MIN  = 300
SCORE_MAX  = 850
FACTOR     = PDO / np.log(2)            # Points per log-odds unit
OFFSET     = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def load_raw_data(filename="cs-training.csv") -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n  ❌  Dataset not found at: {path}\n"
            f"  👉  Download from: https://www.kaggle.com/datasets/brycecf/give-me-some-credit\n"
            f"  👉  Place cs-training.csv and cs-test.csv in the data/ folder.\n"
        )
    df = pd.read_csv(path, index_col=0)
    df.rename(columns=RENAME_MAP, inplace=True)
    logger.info(f"Loaded {len(df):,} rows from {filename}")
    return df


def clip_score(score: np.ndarray) -> np.ndarray:
    return np.clip(score, SCORE_MIN, SCORE_MAX)


def gini_coefficient(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def ks_statistic(y_true, y_score) -> float:
    from scipy.stats import ks_2samp
    scores_good = y_score[y_true == 0]
    scores_bad  = y_score[y_true == 1]
    ks, _ = ks_2samp(scores_good, scores_bad)
    return ks
