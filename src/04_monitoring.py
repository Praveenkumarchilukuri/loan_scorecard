"""
04_monitoring.py
Model Monitoring Pipeline — Population Stability Index (PSI),
characteristic stability analysis, and automated early-warning alerts.
Simulates monthly score batches to replicate production drift monitoring.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from utils import OUTPUTS_DIR, CHARTS_DIR, FEATURES, TARGET, logger


# ── PSI thresholds (Basel / IFRS9 standard) ───────────────────────────────────
PSI_GREEN  = 0.10   # Stable
PSI_AMBER  = 0.25   # Monitor closely
# > 0.25 = RED — model degradation, action required


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index between two score distributions.
    expected = development / baseline distribution
    actual   = monitoring / recent distribution
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints  = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    # Avoid division by zero
    exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 1e-4, act_counts / len(actual))

    psi_bins = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return float(np.sum(psi_bins))


def psi_flag(psi: float) -> str:
    if psi < PSI_GREEN: return "🟢 STABLE"
    if psi < PSI_AMBER: return "🟡 MONITOR"
    return "🔴 ACTION REQUIRED"


def simulate_monthly_batches(df_dev: pd.DataFrame, n_months: int = 6) -> list[pd.DataFrame]:
    """
    Simulate score drift over n months by progressively shifting
    the population (mimicking economic cycle or policy change).
    """
    batches = []
    for month in range(1, n_months + 1):
        sample = df_dev.sample(frac=0.15, random_state=month)
        # Introduce gradual shift: riskier population each month
        drift_factor = 1 + (month * 0.04)
        sample = sample.copy()
        sample["credit_score"] = (sample["credit_score"] / drift_factor).clip(300, 850)
        sample["month"] = month
        batches.append(sample)
    return batches


def run_psi_monitoring(df_dev: pd.DataFrame, batches: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute PSI for each monthly batch vs. development population."""
    baseline_scores = df_dev["credit_score"].values
    results = []

    for batch in batches:
        month  = batch["month"].iloc[0]
        psi    = compute_psi(baseline_scores, batch["credit_score"].values)
        flag   = psi_flag(psi)
        results.append({
            "month":          f"Month {month:02d}",
            "psi":            round(psi, 4),
            "status":         flag,
            "batch_mean":     round(batch["credit_score"].mean(), 1),
            "baseline_mean":  round(df_dev["credit_score"].mean(), 1),
            "batch_size":     len(batch),
        })
        logger.info(f"  Month {month:02d} | PSI={psi:.4f} | {flag}")

    return pd.DataFrame(results)


def run_characteristic_stability(df_dev: pd.DataFrame, batches: list[pd.DataFrame],
                                  binning_models: dict, selected_features: list) -> pd.DataFrame:
    """PSI per feature (characteristic stability analysis)."""
    rows = []
    combined_batch = pd.concat(batches, ignore_index=True)

    for feat in selected_features:
        dev_vals   = df_dev[feat].dropna().values
        batch_vals = combined_batch[feat].dropna().values
        psi = compute_psi(dev_vals, batch_vals, bins=8)
        rows.append({
            "feature":   feat,
            "psi":       round(psi, 4),
            "status":    psi_flag(psi),
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def plot_psi_trend(psi_df: pd.DataFrame):
    months = psi_df["month"].tolist()
    psi_vals = psi_df["psi"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for v in psi_vals:
        if v < PSI_GREEN:   colors.append("#27ae60")
        elif v < PSI_AMBER: colors.append("#f39c12")
        else:                colors.append("#e74c3c")

    ax.bar(months, psi_vals, color=colors, edgecolor="white", width=0.5)
    ax.axhline(PSI_GREEN, color="#27ae60", linestyle="--", lw=1.5, label=f"Stable threshold ({PSI_GREEN})")
    ax.axhline(PSI_AMBER, color="#e74c3c", linestyle="--", lw=1.5, label=f"Action threshold ({PSI_AMBER})")

    ax.set_ylabel("PSI", fontsize=12)
    ax.set_title("Population Stability Index (PSI) — Monthly Monitoring", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(psi_vals.max() * 1.3, 0.3))

    for i, (m, v) in enumerate(zip(months, psi_vals)):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "psi_report.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"PSI chart saved → {out}")


def plot_score_drift(df_dev: pd.DataFrame, batches: list[pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df_dev["credit_score"], bins=40, alpha=0.5, label="Development (Baseline)",
            color="#2c3e50", edgecolor="white", density=True)

    cmap = plt.cm.Reds(np.linspace(0.3, 0.9, len(batches)))
    for i, batch in enumerate(batches):
        ax.hist(batch["credit_score"], bins=40, alpha=0.4,
                label=f"Month {i+1}", color=cmap[i], density=True)

    ax.set_xlabel("Credit Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution Drift Over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "score_drift.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Score drift chart saved → {out}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 4 — MODEL MONITORING (PSI & STABILITY)")
    logger.info("=" * 60)

    scored_path = os.path.join(OUTPUTS_DIR, "scored_data.parquet")
    if not os.path.exists(scored_path):
        raise FileNotFoundError("Run 03_train_model.py first.")

    df = pd.read_parquet(scored_path)
    artifacts = joblib.load(os.path.join(OUTPUTS_DIR, "models", "binning_models.pkl"))
    selected_features = artifacts["selected_features"]
    binning_models    = artifacts["binning_models"]

    # ── Simulate monthly batches ─────────────────────────────────────────────
    logger.info("\nSimulating 6 months of scoring batches...")
    batches = simulate_monthly_batches(df, n_months=6)

    # ── PSI monitoring ───────────────────────────────────────────────────────
    logger.info("\nRunning PSI monitoring...\n")
    psi_df = run_psi_monitoring(df, batches)

    psi_path = os.path.join(OUTPUTS_DIR, "psi_results.csv")
    psi_df.to_csv(psi_path, index=False)
    logger.info(f"\nPSI results saved → {psi_path}")

    # ── Characteristic stability ─────────────────────────────────────────────
    logger.info("\nRunning characteristic stability analysis...")
    char_df = run_characteristic_stability(df, batches, binning_models, selected_features)
    char_path = os.path.join(OUTPUTS_DIR, "characteristic_stability.csv")
    char_df.to_csv(char_path, index=False)
    logger.info(f"Characteristic stability saved → {char_path}")
    print("\n", char_df.to_string(index=False))

    # ── Charts ───────────────────────────────────────────────────────────────
    plot_psi_trend(psi_df)
    plot_score_drift(df, batches)

    return psi_df, char_df


if __name__ == "__main__":
    main()
