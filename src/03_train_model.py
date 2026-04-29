"""
03_train_model.py
Train XGBoost classifier → convert to interpretable integer points scorecard
(300–850 range) using log-odds scaling with PDO=20, base score=600.
Replicates industry-standard retail banking scorecard methodology.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from utils import (
    FEATURES, TARGET, OUTPUTS_DIR, CHARTS_DIR,
    BASE_SCORE, BASE_ODDS, PDO, SCORE_MIN, SCORE_MAX,
    clip_score, gini_coefficient, ks_statistic, logger,
)


# ── Scorecard scaling constants ───────────────────────────────────────────────
FACTOR  = PDO / np.log(2)          # Points per log-odds unit
OFFSET  = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def log_odds_to_score(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds output to scorecard points."""
    return OFFSET + FACTOR * log_odds


def build_scorecard_table(binning_models: dict, selected_features: list,
                          xgb_model: XGBClassifier) -> pd.DataFrame:
    """
    Build a points table per feature bin.
    Each bin's WoE is weighted by the XGBoost feature importance
    then converted to integer scorecard points.
    """
    importance = xgb_model.get_booster().get_fscore()  # feature → importance count
    # Normalise importances
    total_imp = sum(importance.values()) if importance else 1
    norm_imp  = {f: importance.get(f"f{i}", 1) / total_imp
                 for i, f in enumerate(selected_features)}

    rows = []
    for feat in selected_features:
        info  = binning_models[feat]
        table = info["table"]
        imp   = norm_imp.get(feat, 1 / len(selected_features))

        for _, row in table.iterrows():
            woe   = row["woe"]
            raw   = FACTOR * woe * imp
            score = int(round(raw))
            rows.append({
                "feature":    feat,
                "bin":        str(row["bin"]),
                "count":      int(row["total"]),
                "event_rate": round(row["events"] / max(row["total"], 1), 4),
                "woe":        round(woe, 4),
                "points":     score,
            })

    return pd.DataFrame(rows)


def score_applicants(df: pd.DataFrame, scorecard_df: pd.DataFrame,
                     binning_models: dict, selected_features: list) -> np.ndarray:
    """
    Score each applicant by summing points across feature bins.
    Final score is scaled to 300–850 range.
    """
    scores = np.zeros(len(df))

    for feat in selected_features:
        model = binning_models[feat]["binning_model"]
        edges = model["bin_edges"]
        bin_ids = pd.cut(df[feat], bins=edges, include_lowest=True, labels=False)

        feat_rows = scorecard_df[scorecard_df["feature"] == feat].reset_index(drop=True)
        points_map = {i: feat_rows.loc[i, "points"] for i in range(len(feat_rows))}

        for i, bid in enumerate(bin_ids.fillna(-1).astype(int).to_numpy()):
            scores[i] += points_map.get(int(bid), 0)

    # Shift and scale to 300–850
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min) * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    else:
        scores = np.full_like(scores, BASE_SCORE, dtype=float)

    return clip_score(scores)


def plot_score_distribution(scores: np.ndarray, y: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    axes[0].hist(scores, bins=50, color="#2c3e50", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Credit Score", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Score Distribution (300–850)", fontsize=13, fontweight="bold")
    axes[0].axvline(scores.mean(), color="#e74c3c", linestyle="--", label=f"Mean: {scores.mean():.0f}")
    axes[0].legend()

    # Good vs Bad overlay
    axes[1].hist(scores[y == 0], bins=40, alpha=0.65, color="#27ae60", label="Good (No Default)", edgecolor="white")
    axes[1].hist(scores[y == 1], bins=40, alpha=0.65, color="#e74c3c", label="Bad (Default)",      edgecolor="white")
    axes[1].set_xlabel("Credit Score", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Score by Default Status", fontsize=13, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "score_distribution.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Score distribution chart saved → {out}")


def plot_roc_curve(y_test, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2980b9", lw=2, label=f"XGBoost AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Champion XGBoost Model", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "roc_curve.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved → {out}")


def main():
    logger.info("=" * 60)
    logger.info("STEP 3 — MODEL TRAINING & SCORECARD SCALING")
    logger.info("=" * 60)

    # ── Load artefacts from step 2 ───────────────────────────────────────────
    woe_path = os.path.join(OUTPUTS_DIR, "woe_data.parquet")
    if not os.path.exists(woe_path):
        raise FileNotFoundError("Run 02_feature_engineering.py first.")

    df = pd.read_parquet(woe_path)
    artifacts = joblib.load(os.path.join(OUTPUTS_DIR, "models", "binning_models.pkl"))
    binning_models   = artifacts["binning_models"]
    selected_features = artifacts["selected_features"]

    woe_cols = [f"{f}_woe" for f in selected_features]
    X = df[woe_cols].values
    y = df[TARGET].values

    # ── Train / Test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    df_train = df.loc[idx_train].copy()
    df_test  = df.loc[idx_test].copy()

    logger.info(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
    logger.info(f"Default rate — Train: {y_train.mean():.2%}  |  Test: {y_test.mean():.2%}")

    # ── XGBoost training ─────────────────────────────────────────────────────
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_prob = xgb.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    gini   = gini_coefficient(y_test, y_prob)
    ks     = ks_statistic(y_test, y_prob)

    logger.info(f"\n📊 Champion Model Performance:")
    logger.info(f"   AUC  = {auc:.4f}")
    logger.info(f"   Gini = {gini:.4f}")
    logger.info(f"   KS   = {ks:.4f}")

    # ── Build scorecard table ─────────────────────────────────────────────────
    logger.info("\nBuilding scorecard points table...")
    scorecard_df = build_scorecard_table(binning_models, selected_features, xgb)
    sc_path = os.path.join(OUTPUTS_DIR, "scorecard_points.csv")
    scorecard_df.to_csv(sc_path, index=False)
    logger.info(f"Scorecard table saved → {sc_path}")
    print("\n", scorecard_df.head(15).to_string(index=False))

    # ── Score all applicants ──────────────────────────────────────────────────
    logger.info("\nScoring all applicants...")
    all_scores = score_applicants(df, scorecard_df, binning_models, selected_features)
    df["credit_score"] = all_scores

    scored_path = os.path.join(OUTPUTS_DIR, "scored_data.parquet")
    df.to_parquet(scored_path, index=False)
    logger.info(f"Scored data saved → {scored_path}")
    logger.info(f"Score stats: min={all_scores.min():.0f}  max={all_scores.max():.0f}  "
                f"mean={all_scores.mean():.0f}  median={np.median(all_scores):.0f}")

    # ── Save test set scores for monitoring ──────────────────────────────────
    df_test["credit_score"] = score_applicants(df_test, scorecard_df, binning_models, selected_features)
    df_test["y_prob"]       = y_prob
    df_test.to_parquet(os.path.join(OUTPUTS_DIR, "test_scored.parquet"), index=False)

    # ── Save model ───────────────────────────────────────────────────────────
    model_bundle = {
        "xgb_model":        xgb,
        "selected_features": selected_features,
        "woe_cols":         woe_cols,
        "metrics":          {"auc": auc, "gini": gini, "ks": ks},
    }
    joblib.dump(model_bundle, os.path.join(OUTPUTS_DIR, "models", "xgb_model.pkl"))
    logger.info(f"Model bundle saved → outputs/models/xgb_model.pkl")

    # ── Charts ───────────────────────────────────────────────────────────────
    plot_score_distribution(all_scores, y)
    plot_roc_curve(y_test, y_prob)

    return df, scorecard_df, model_bundle


if __name__ == "__main__":
    main()
