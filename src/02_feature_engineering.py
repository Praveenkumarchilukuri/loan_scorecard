"""
02_feature_engineering.py
Weight of Evidence (WoE) binning and Information Value (IV) calculation.
Transforms raw features into WoE-encoded features for scorecard modeling.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import CHARTS_DIR, FEATURES, OUTPUTS_DIR, TARGET, logger


def compute_woe_iv(df: pd.DataFrame, feature: str, target: str) -> dict:
    """Compute quantile-based WoE and IV for a single feature."""
    x = df[feature]
    y = df[target]

    n_bins = min(10, max(2, int(np.sqrt(len(df)))))
    try:
        binned, bin_edges = pd.qcut(x, q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        binned, bin_edges = pd.cut(x, bins=n_bins, retbins=True, duplicates="drop")

    bin_df = pd.DataFrame({"bin": binned, "target": y})
    bin_stats = (
        bin_df.groupby("bin", observed=True)
        .agg(events=("target", "sum"), total=("target", "count"))
        .reset_index()
    )
    bin_stats["non_events"] = bin_stats["total"] - bin_stats["events"]

    total_events = max(bin_stats["events"].sum(), 1)
    total_non_events = max(bin_stats["non_events"].sum(), 1)

    bin_stats["pct_events"] = bin_stats["events"].replace(0, 0.0001) / total_events
    bin_stats["pct_non_events"] = bin_stats["non_events"].replace(0, 0.0001) / total_non_events
    bin_stats["woe"] = np.log(bin_stats["pct_events"] / bin_stats["pct_non_events"])
    bin_stats["iv_contrib"] = (bin_stats["pct_events"] - bin_stats["pct_non_events"]) * bin_stats["woe"]

    return {
        "binning_model": {
            "bin_edges": bin_edges,
            "woe_values": bin_stats["woe"].to_numpy(),
        },
        "iv": round(float(bin_stats["iv_contrib"].sum()), 4),
        "table": bin_stats[["bin", "events", "non_events", "total", "woe", "iv_contrib"]],
    }


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "Unpredictive"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    if iv < 0.5:
        return "Strong"
    return "Very Strong"


def apply_woe_transform(df: pd.DataFrame, binning_models: dict) -> pd.DataFrame:
    """Replace raw feature values with WoE scores using stored bin edges."""
    df_woe = df.copy()
    for feat, info in binning_models.items():
        model = info["binning_model"]
        edges = model["bin_edges"]
        woe_values = model["woe_values"]

        bin_codes = pd.cut(df_woe[feat], bins=edges, include_lowest=True, labels=False)
        code_array = bin_codes.fillna(-1).astype(int).to_numpy()

        woe_array = np.zeros(len(df_woe), dtype=float)
        valid_mask = (code_array >= 0) & (code_array < len(woe_values))
        woe_array[valid_mask] = woe_values[code_array[valid_mask]]
        df_woe[f"{feat}_woe"] = woe_array

    return df_woe


def plot_woe_charts(binning_models: dict):
    """Save WoE bar charts for each feature."""
    n = len(binning_models)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, (feat, info) in zip(axes, binning_models.items()):
        table = info["table"]
        bins = table["bin"].astype(str)
        woe = table["woe"].values

        colors = ["#e74c3c" if value < 0 else "#2ecc71" for value in woe]
        ax.barh(bins, woe, color=colors, edgecolor="white", height=0.6)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{feat}  |  IV = {info['iv']:.4f}  ({iv_strength(info['iv'])})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Weight of Evidence")
        ax.invert_yaxis()

    plt.suptitle("WoE Binning — All Features", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "woe_binning.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"WoE chart saved → {out}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 2 — FEATURE ENGINEERING (WoE / IV)")
    logger.info("=" * 60)

    clean_path = os.path.join(OUTPUTS_DIR, "clean_data.parquet")
    if not os.path.exists(clean_path):
        raise FileNotFoundError("Run 01_data_quality.py first.")

    df = pd.read_parquet(clean_path)

    logger.info("\nFitting WoE binning per feature...\n")
    binning_models = {}
    iv_summary = []

    for feat in FEATURES:
        result = compute_woe_iv(df, feat, TARGET)
        binning_models[feat] = result
        strength = iv_strength(result["iv"])
        iv_summary.append({"feature": feat, "iv": result["iv"], "strength": strength})
        logger.info(f"  {feat:<30} IV={result['iv']:.4f}  [{strength}]")

    iv_df = pd.DataFrame(iv_summary).sort_values("iv", ascending=False)
    iv_path = os.path.join(OUTPUTS_DIR, "iv_summary.csv")
    iv_df.to_csv(iv_path, index=False)
    logger.info(f"\nIV summary saved → {iv_path}")

    selected = iv_df[iv_df["iv"] >= 0.02]["feature"].tolist()
    logger.info(f"\nSelected {len(selected)} features with IV ≥ 0.02: {selected}")

    df_woe = apply_woe_transform(df, binning_models)
    woe_path = os.path.join(OUTPUTS_DIR, "woe_data.parquet")
    df_woe.to_parquet(woe_path, index=False)
    logger.info(f"WoE-encoded data saved → {woe_path}")

    model_path = os.path.join(OUTPUTS_DIR, "models", "binning_models.pkl")
    joblib.dump({"binning_models": binning_models, "selected_features": selected}, model_path)
    logger.info(f"Binning models saved → {model_path}")

    plot_woe_charts({k: v for k, v in binning_models.items() if k in selected})

    return df_woe, binning_models, selected


if __name__ == "__main__":
    main()
