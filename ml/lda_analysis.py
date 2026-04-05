"""
Linear Discriminant Analysis for Trade Outcome Classification

LDA finds linear combinations of features that best separate two or more
classes. Here the classes are trade outcomes: 1 = profitable, 0 = loss.

Difference from PCA:
  PCA is unsupervised — it maximizes explained variance without knowing labels.
  LDA is supervised — it maximizes the ratio of between-class variance to
  within-class variance. Given our trade labels from the meta-labeler, LDA
  identifies which feature directions most cleanly separate wins from losses.

Practical value in this pipeline:
  1. Diagnostic: which features best predict trade success? If z_at_entry
     dominates the first discriminant, extreme entries are reliable. If
     vix_percentile dominates, market regime matters more.
  2. Classification: the LDA boundary can be used as an alternative or
     complement to the Random Forest meta-labeler — it's linear, interpretable,
     and doesn't overfit on small samples.
  3. Visualization: project all trades onto the first discriminant axis (LD1)
     and plot the win vs loss distributions to visually assess separability.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import os


FEATURE_NAMES = [
    "z_at_entry",
    "spread_volatility",
    "z_velocity",
    "z_acceleration",
    "vix_level",
    "vix_percentile",
    "ou_half_life",
]


def prepare_lda_data(
    features: pd.DataFrame,
    labels:   pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Aligns features and labels, drops NaNs, and standardizes.
    LDA is sensitive to scale because it computes covariance matrices —
    standardization ensures all features contribute on equal footing.

    Returns: X_scaled, y, feature_names
    """
    aligned = features.join(labels.rename("label")).dropna()
    X       = aligned[features.columns].values
    y       = aligned["label"].values.astype(int)
    cols    = features.columns.tolist()

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, cols


def fit_lda(
    X:      np.ndarray,
    y:      np.ndarray,
    solver: str = "svd",   # SVD is numerically stable; use 'lsqr' for regularization
) -> LinearDiscriminantAnalysis:
    """
    Fits LDA. For two classes, this produces a single linear discriminant (LD1).
    The direction of LD1 is the weight vector that maximally separates wins from losses.

    solver options:
      'svd'   — no assumptions on covariance matrix, no regularization
      'lsqr'  — least squares, faster for large feature sets
      'eigen' — eigenvalue decomposition, required for shrinkage regularization
    """
    lda = LinearDiscriminantAnalysis(solver=solver, store_covariance=True)
    lda.fit(X, y)
    return lda


def cross_validate_lda(
    X:        np.ndarray,
    y:        np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Stratified k-fold cross-validation of LDA classification accuracy.

    Stratified folds preserve the win/loss ratio in each fold, which matters
    when classes are imbalanced (more losses than wins is common in stat arb).

    Returns mean accuracy, AUC, and per-fold results.
    """
    lda  = LinearDiscriminantAnalysis(solver="svd")
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracy_scores = cross_val_score(lda, X, y, cv=skf, scoring="accuracy")
    auc_scores      = cross_val_score(lda, X, y, cv=skf, scoring="roc_auc")

    return {
        "mean_accuracy": round(accuracy_scores.mean(), 4),
        "std_accuracy":  round(accuracy_scores.std(), 4),
        "mean_auc":      round(auc_scores.mean(), 4),
        "std_auc":       round(auc_scores.std(), 4),
        "fold_accuracies": accuracy_scores.tolist(),
    }


def get_discriminant_weights(
    lda:           LinearDiscriminantAnalysis,
    feature_names: list[str],
) -> pd.Series:
    """
    Returns the LD1 coefficient vector as a labeled Series.
    Large absolute values indicate features that most strongly separate
    winning from losing trades along the first discriminant axis.
    """
    weights = pd.Series(
        lda.coef_[0],
        index=feature_names,
        name="LD1_weight",
    ).sort_values(key=abs, ascending=False)

    return weights


def plot_discriminant_distributions(
    lda:       LinearDiscriminantAnalysis,
    X:         np.ndarray,
    y:         np.ndarray,
    save_path: str = "outputs/lda_distributions.png",
) -> None:
    """
    Projects all trades onto LD1 and plots the score distributions for
    winning (1) and losing (0) trades as overlapping histograms.

    Good separation = distributions barely overlap.
    Poor separation = distributions are nearly identical.
    """
    os.makedirs("outputs", exist_ok=True)

    ld1_scores = lda.transform(X)[:, 0]
    wins       = ld1_scores[y == 1]
    losses     = ld1_scores[y == 0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(losses, bins=30, alpha=0.6, color="#F44336", density=True, label="loss (0)")
    ax.hist(wins,   bins=30, alpha=0.6, color="#4CAF50", density=True, label="win (1)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="decision boundary")
    ax.set_xlabel("LD1 score")
    ax.set_ylabel("density")
    ax.set_title("LDA: trade outcome separation on first discriminant axis")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_discriminant_weights(
    weights:   pd.Series,
    save_path: str = "outputs/lda_weights.png",
) -> None:
    """
    Horizontal bar chart of LD1 feature weights.
    Positive weight: feature value above average → more likely to be a winning trade.
    Negative weight: feature value above average → more likely to be a losing trade.
    """
    os.makedirs("outputs", exist_ok=True)

    colors = ["#4CAF50" if v > 0 else "#F44336" for v in weights.values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(weights) * 0.5)))
    ax.barh(weights.index, weights.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LD1 coefficient weight")
    ax.set_title("LDA feature importance for trade outcome (LD1)")
    ax.grid(alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_roc_curve(
    lda:       LinearDiscriminantAnalysis,
    X:         np.ndarray,
    y:         np.ndarray,
    save_path: str = "outputs/lda_roc.png",
) -> float:
    """
    ROC curve and AUC for the LDA classifier.
    AUC > 0.6 on financial data is meaningful; > 0.7 is strong.
    """
    os.makedirs("outputs", exist_ok=True)

    probs = lda.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", label=f"LDA (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("LDA ROC curve — trade outcome classification")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")

    return auc_val


def run_lda_analysis(
    features:   pd.DataFrame,
    labels:     pd.Series,
    save_plots: bool = True,
) -> dict:
    """
    Full LDA analysis pipeline.
    Returns LDA model, weights, cross-validation results, and AUC.
    """
    X, y, feat_names = prepare_lda_data(features, labels)

    if len(np.unique(y)) < 2:
        print("  LDA skipped: only one class present in labels")
        return {}

    lda      = fit_lda(X, y)
    cv_result = cross_validate_lda(X, y)
    weights  = get_discriminant_weights(lda, feat_names)

    print(f"  LDA cross-validation: accuracy={cv_result['mean_accuracy']:.3f} "
          f"(+/-{cv_result['std_accuracy']:.3f}), AUC={cv_result['mean_auc']:.3f}")
    print(f"  top discriminating features:")
    for feat, val in weights.head(3).items():
        print(f"    {feat}: {val:.4f}")

    auc_val = 0.0
    if save_plots:
        plot_discriminant_distributions(lda, X, y)
        plot_discriminant_weights(weights)
        auc_val = plot_roc_curve(lda, X, y)

    return {
        "lda":          lda,
        "weights":      weights,
        "cv_result":    cv_result,
        "auc":          auc_val,
    }
