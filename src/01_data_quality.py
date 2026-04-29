"""
01_data_quality.py
SQL-based data control layer — null checks, range validation,
referential integrity — mirrors production credit system controls.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import duckdb
from utils import load_raw_data, OUTPUTS_DIR, logger

# ── DQ rules ──────────────────────────────────────────────────────────────────
DQ_CHECKS = [
    # (check_name, sql_where_clause_for_failures, severity)
    ("null_target",          "target IS NULL",                                           "CRITICAL"),
    ("null_age",             "age IS NULL",                                              "CRITICAL"),
    ("null_monthly_income",  "monthly_income IS NULL",                                   "WARNING"),
    ("null_dependents",      "dependents IS NULL",                                       "WARNING"),
    ("age_range",            "age < 18 OR age > 110",                                    "CRITICAL"),
    ("revolving_util_range", "revolving_util < 0 OR revolving_util > 50",               "WARNING"),
    ("debt_ratio_negative",  "debt_ratio < 0",                                           "CRITICAL"),
    ("dpd_30_59_negative",   "dpd_30_59 < 0",                                           "CRITICAL"),
    ("dpd_60_89_negative",   "dpd_60_89 < 0",                                           "CRITICAL"),
    ("dpd_90_negative",      "dpd_90 < 0",                                               "CRITICAL"),
    ("income_negative",      "monthly_income < 0",                                       "CRITICAL"),
    ("target_binary",        "target NOT IN (0, 1)",                                     "CRITICAL"),
    ("open_lines_negative",  "open_credit_lines < 0",                                   "WARNING"),
    ("real_estate_negative", "real_estate_loans < 0",                                   "WARNING"),
    ("dependents_negative",  "dependents < 0",                                           "WARNING"),
]


def run_dq_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run all DQ checks via DuckDB SQL. Returns summary DataFrame."""
    con = duckdb.connect()
    con.register("applicants", df)

    results = []
    total_rows = len(df)

    for check_name, where_clause, severity in DQ_CHECKS:
        sql = f"""
            SELECT COUNT(*) AS fail_count
            FROM applicants
            WHERE {where_clause}
        """
        fail_count = con.execute(sql).fetchone()[0]
        fail_pct   = round(fail_count / total_rows * 100, 4)
        status     = "PASS" if fail_count == 0 else ("FAIL" if severity == "CRITICAL" else "WARN")

        results.append({
            "check_name":  check_name,
            "severity":    severity,
            "fail_count":  fail_count,
            "fail_pct":    fail_pct,
            "total_rows":  total_rows,
            "status":      status,
        })

        icon = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⚠️ ")
        logger.info(f"  {icon}  {check_name:<30} | {status:<4} | failures: {fail_count:,} ({fail_pct}%)")

    con.close()
    return pd.DataFrame(results)


def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Standard median imputation for missing values after DQ."""
    df = df.copy()
    df["monthly_income"].fillna(df["monthly_income"].median(), inplace=True)
    df["dependents"].fillna(df["dependents"].median(), inplace=True)
    # Cap extreme revolving utilization
    df["revolving_util"] = df["revolving_util"].clip(0, 1)
    # Cap dpd flags (erroneous values like 98)
    for col in ["dpd_30_59", "dpd_60_89", "dpd_90"]:
        df[col] = df[col].clip(0, 20)
    # Remove age outliers
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    return df


def main():
    logger.info("=" * 60)
    logger.info("STEP 1 — DATA QUALITY CHECKS")
    logger.info("=" * 60)

    df = load_raw_data("cs-training.csv")

    logger.info(f"\nRunning {len(DQ_CHECKS)} DQ checks on {len(df):,} rows...\n")
    results = run_dq_checks(df)

    # Save results
    out_path = os.path.join(OUTPUTS_DIR, "data_quality_results.csv")
    results.to_csv(out_path, index=False)
    logger.info(f"\nDQ results saved → {out_path}")

    critical_fails = results[(results["severity"] == "CRITICAL") & (results["status"] == "FAIL")]
    if len(critical_fails) > 0:
        logger.warning(f"\n⚠️  {len(critical_fails)} CRITICAL checks failed — review before scoring!")
    else:
        logger.info("\n✅  All CRITICAL checks passed. Proceeding with imputation...")

    df_clean = apply_imputation(df)
    clean_path = os.path.join(OUTPUTS_DIR, "clean_data.parquet")
    df_clean.to_parquet(clean_path, index=False)
    logger.info(f"Clean data saved → {clean_path}  ({len(df_clean):,} rows)")

    return df_clean


if __name__ == "__main__":
    main()
