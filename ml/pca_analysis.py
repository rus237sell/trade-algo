"""
Principal Component Analysis of Return Structure

PCA decomposes the covariance matrix of asset returns into orthogonal
principal components, ordered by the amount of variance they explain.

In equity markets, the first few components have well-known interpretations:
  PC1: market factor — all stocks load positively; explains 30-60% of variance
  PC2: value/growth split — long-only loadings separate value from growth
  PC3: sector rotation factor — loadings align with cyclical vs defensive

For a pairs trading strategy, PCA is useful in two ways:

1. Residualization: project returns onto the first N PCs and trade the
   residuals. This removes the common factors and isolates idiosyncratic
   behavior — the spread becomes cleaner and more likely to be stationary.

2. Regime detection: the explained variance ratio of PC1 changes with
   market conditions. When PC1 explains more variance than usual, stocks
   are highly correlated (risk-off regime). When PC1 explains less, the
   market is stock-picker friendly and pairs strategies tend to work better.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


def run_pca(log_returns: pd.DataFrame, n_components: int = None) -> dict:
    """
    Fits PCA on the asset return matrix (T observations x N assets).

    Standardizes returns before fitting so that high-vol assets don't dominate
    the principal components purely by having larger absolute returns.

    Returns the PCA object, transformed scores, loadings, and explained variance.
    """
    symbols = log_returns.columns.tolist()

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(log_returns.dropna())

    n_comp   = n_components or log_returns.shape[1]
    pca      = PCA(n_components=n_comp, random_state=42)
    scores   = pca.fit_transform(X_scaled)   # T x n_comp — PC scores per day
    loadings = pca.components_               # n_comp x N — how each stock loads

    explained_var   = pca.explained_variance_ratio_
    cumulative_var  = np.cumsum(explained_var)

    # loadings DataFrame: rows = PCs, columns = stocks
    loadings_df = pd.DataFrame(
        loadings,
        index   = [f"PC{i+1}" for i in range(n_comp)],
        columns = symbols,
    )

    # scores DataFrame: rows = dates, columns = PCs
    scores_df = pd.DataFrame(
        scores,
        index   = log_returns.dropna().index,
        columns = [f"PC{i+1}" for i in range(n_comp)],
    )

    return {
        "pca":             pca,
        "loadings":        loadings_df,
        "scores":          scores_df,
        "explained_var":   explained_var,
        "cumulative_var":  cumulative_var,
        "n_components_95": int(np.searchsorted(cumulative_var, 0.95)) + 1,
    }


def residualize_returns(
    log_returns:  pd.DataFrame,
    n_factors:    int = 3,
) -> pd.DataFrame:
    """
    Projects returns onto the first n_factors PCs and subtracts the explained
    portion, leaving idiosyncratic residuals.

    For asset i:
      residual_i = return_i - sum_k(loading_ik * score_k)

    These residuals have the common factors removed and are used in the
    sector pairs strategy and meta-labeler feature set.
    """
    pca_result  = run_pca(log_returns, n_components=n_factors)
    scores      = pca_result["scores"].values       # T x k
    loadings    = pca_result["loadings"].values[:n_factors]  # k x N

    X_scaled    = StandardScaler().fit_transform(log_returns.dropna())
    X_explained = scores @ loadings                 # T x N reconstruction
    X_residual  = X_scaled - X_explained

    residuals_df = pd.DataFrame(
        X_residual,
        index   = log_returns.dropna().index,
        columns = log_returns.columns,
    )

    return residuals_df


def plot_scree(pca_result: dict, save_path: str = "outputs/pca_scree.png") -> None:
    """
    Scree plot: explained variance per component (bar) and cumulative (line).
    The component where cumulative variance crosses 95% is the minimum
    dimensionality needed to represent most of the market structure.
    """
    os.makedirs("outputs", exist_ok=True)

    ev  = pca_result["explained_var"][:20]   # first 20 components
    cum = pca_result["cumulative_var"][:20]
    n   = len(ev)
    pcs = [f"PC{i+1}" for i in range(n)]

    fig, ax1 = plt.subplots(figsize=(11, 4))

    ax1.bar(pcs, ev * 100, color="#2196F3", alpha=0.7, label="individual")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance (%)")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(pcs, cum * 100, color="red", marker="o", markersize=4, label="cumulative")
    ax2.axhline(95, color="red", linestyle="--", alpha=0.5, label="95% threshold")
    ax2.set_ylabel("cumulative explained variance (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(f"PCA scree plot | 95% variance at PC{pca_result['n_components_95']}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_factor_loadings(
    pca_result: dict,
    n_factors:  int  = 3,
    save_path:  str  = "outputs/pca_loadings.png",
) -> None:
    """
    Horizontal bar chart of the first n_factors PC loadings per stock.
    Stocks with high positive PC1 loading are most exposed to the market factor.
    Stocks with large positive or negative PC2/PC3 loadings have strong
    sector or style tilts.
    """
    os.makedirs("outputs", exist_ok=True)

    loadings = pca_result["loadings"].iloc[:n_factors]
    symbols  = loadings.columns.tolist()
    n        = len(symbols)

    fig, axes = plt.subplots(1, n_factors, figsize=(5 * n_factors, max(4, n * 0.3)))

    for i, pc in enumerate(loadings.index[:n_factors]):
        vals    = loadings.loc[pc].sort_values()
        colors  = ["#F44336" if v < 0 else "#4CAF50" for v in vals]
        axes[i].barh(vals.index, vals.values, color=colors)
        axes[i].axvline(0, color="black", linewidth=0.8)
        axes[i].set_title(f"{pc} ({pca_result['explained_var'][i]:.1%} var)")
        axes[i].set_xlabel("loading")
        axes[i].grid(alpha=0.2, axis="x")

    plt.suptitle("PCA factor loadings by asset", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_pc1_regime(
    pca_result:  dict,
    window:      int = 60,
    save_path:   str = "outputs/pca_pc1_regime.png",
) -> None:
    """
    Rolling explained variance of PC1 over time.

    When PC1 (market factor) explains an unusually high proportion of
    total variance, stocks are highly correlated — this is a risk-off
    environment where pairs strategies tend to perform poorly because
    pair divergences are overwhelmed by market-wide moves.

    This chart gives a regime indicator that can gate strategy allocation.
    """
    os.makedirs("outputs", exist_ok=True)

    scores    = pca_result["scores"]
    pc1_var   = scores["PC1"].rolling(window).var()
    total_var = scores.rolling(window).var().sum(axis=1)
    rolling_share = pc1_var / total_var

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(rolling_share.index, rolling_share.values,
                    alpha=0.4, color="#FF9800", label=f"{window}d rolling PC1 share")
    ax.axhline(rolling_share.mean(), color="black", linestyle="--", alpha=0.5,
               label=f"mean = {rolling_share.mean():.2f}")
    ax.set_ylabel("PC1 variance share")
    ax.set_title("market factor (PC1) rolling variance share — regime indicator")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def run_pca_analysis(
    log_returns: pd.DataFrame,
    n_factors:   int  = 5,
    save_plots:  bool = True,
) -> dict:
    """
    Full PCA analysis pipeline. Returns the PCA result dict with
    loadings, scores, and variance metrics.
    """
    pca_result = run_pca(log_returns, n_components=min(n_factors, log_returns.shape[1]))

    ev_str = ", ".join([f"PC{i+1}:{v:.1%}" for i, v in enumerate(pca_result["explained_var"][:5])])
    print(f"  PCA explained variance: {ev_str}")
    print(f"  components to reach 95% variance: {pca_result['n_components_95']}")

    if save_plots:
        plot_scree(pca_result)
        plot_factor_loadings(pca_result, n_factors=min(3, n_factors))
        plot_pc1_regime(pca_result)

    return pca_result
