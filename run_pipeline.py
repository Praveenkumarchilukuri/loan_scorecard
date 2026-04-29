"""
run_pipeline.py
One-click full pipeline runner.
Executes all 6 steps in order with timing and summary.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import logger

STEPS = [
    ("01_data_quality",       "Data Quality Checks (SQL validation layer)"),
    ("02_feature_engineering","Feature Engineering (WoE / IV binning)"),
    ("03_train_model",        "Model Training & Scorecard Scaling (XGBoost → 300-850)"),
    ("04_monitoring",         "Model Monitoring (PSI drift detection)"),
    ("05_reporting",          "Excel Report Generation (management & regulatory)"),
    ("06_challenger_model",   "Champion vs Challenger Evaluation"),
]


def run_step(module_name: str, description: str) -> bool:
    import importlib
    logger.info("")
    logger.info("─" * 60)
    logger.info(f"▶  {description}")
    logger.info("─" * 60)
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        logger.info(f"✅  Step complete  ({elapsed:.1f}s)")
        return True
    except Exception as e:
        logger.error(f"❌  Step FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: list):
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for step, desc, ok in results:
        icon = "✅" if ok else "❌"
        logger.info(f"  {icon}  {step}  —  {desc}")

    passed = sum(1 for _, _, ok in results if ok)
    logger.info(f"\n  {passed}/{len(results)} steps completed successfully.")

    if passed == len(results):
        logger.info("""
  📂 Key outputs:
     outputs/scorecard_points.csv           ← Scorecard points table
     outputs/scored_data.parquet            ← All applicants with credit scores
     outputs/excel/model_report.xlsx        ← Management Excel report (6 tabs)
     outputs/excel/challenger_report.xlsx   ← Validation report
     outputs/charts/score_distribution.png  ← Score histogram
     outputs/charts/psi_report.png          ← PSI monitoring chart
     outputs/charts/gini_trend.png          ← Monthly Gini trend
     outputs/charts/champion_vs_challenger.png ← ROC comparison
     outputs/models/xgb_model.pkl           ← Trained XGBoost model
     outputs/data_quality_results.csv       ← DQ check results
        """)
    else:
        logger.warning("\n  ⚠️  Some steps failed. Check logs above for details.")


def main():
    logger.info("=" * 60)
    logger.info("LOAN DEFAULT SCORECARD — FULL PIPELINE")
    logger.info("Production-Style Credit Scoring System")
    logger.info("=" * 60)

    total_start = time.time()
    results = []

    for module_name, description in STEPS:
        ok = run_step(module_name, description)
        results.append((module_name, description, ok))
        if not ok:
            logger.error(f"\n⛔  Pipeline halted at step: {module_name}")
            logger.error("   Fix the error above and re-run.")
            break

    total_elapsed = time.time() - total_start
    print_summary(results)
    logger.info(f"\n  Total runtime: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
