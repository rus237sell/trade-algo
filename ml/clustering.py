"""
K-Means Clustering of Asset Return Profiles

Purpose: identify natural groupings among assets based on the structure of
their return correlations — not GICS sector labels, which are assigned by
committee and often lag actual market behavior (e.g., Amazon is officially
Consumer Discretionary but behaves like a technology company).

What K-means does here:
  Each asset is represented as a row of its correlation coefficients with
  all other assets (its "correlation fingerprint"). K-means clusters assets
  whose fingerprints are similar — meaning they move together for similar
  reasons. This is more data-driven than GICS and often reveals hidden
  cross-sector groupings (e.g., airlines and cruise operators in the same
  cluster even though they span different GICS sectors).

Use cases in this pipeline:
  1. Improve pair selection: only test cointegration within the same cluster
  2. Identify diversification across pairs: pick one pair per cluster to avoid
     correlated drawdowns
  3. Monitor regime shifts: re-cluster quarterly and track which stocks migrate
     between clusters — migration signals a changing correlation structure
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file, no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler


def compute_correlation_features(log_returns: pd.DataFrame) -> np.ndarray:
    """
    Builds the feature matrix for clustering.

    Each row = one asset.
    Each column = that asset's rolling 60-day correlation with every other asset,
    averaged over the last 252 days.

    Using the correlation profile rather than raw returns ensures the clustering
    is based on co-movement structure, not raw volatility or return level.
    """
    corr_matrix = log_returns.corr().fillna(0).values
    # standardize rows so high-vol assets don't dominate by distance
    scaler      = StandardScaler()
    features    = scaler.fit_transform(corr_matrix)
    return features


def optimal_k(features: np.ndarray, k_range: range = range(2, 10)) -> dict:
    """
    Determines the optimal number of clusters using two methods:

    1. Inertia (elbow method): sum of squared distances to cluster centers.
       Look for the "elbow" — the point where adding more clusters gives
       diminishing returns in inertia reduction.

    2. Silhouette score: measures how similar each point is to its own cluster
       vs other clusters. Range [-1, 1], higher is better.
       Silhouette > 0.5 is generally considered meaningful clustering.

    In practice for equity universes of 20-100 stocks, k=4 to k=8 tends to
    produce the most interpretable clusters.
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(features, labels))
        else:
            silhouettes.append(np.nan)

    best_k = list(k_range)[np.nanargmax(silhouettes)]

    return {
        "k_range":    list(k_range),
        "inertias":   inertias,
        "silhouettes": silhouettes,
        "best_k":     best_k,
    }


def fit_clusters(features: np.ndarray, k: int, symbols: list[str]) -> pd.DataFrame:
    """
    Fits K-means with the specified k and returns a DataFrame mapping
    each symbol to its cluster label and silhouette score.
    """
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)

    sil_vals = silhouette_samples(features, labels)

    result = pd.DataFrame({
        "symbol":          symbols,
        "cluster":         labels,
        "silhouette_score": sil_vals,
    }).sort_values(["cluster", "silhouette_score"], ascending=[True, False])

    return result


def plot_elbow_and_silhouette(
    optimal_k_result: dict,
    save_path:        str = "outputs/kmeans_selection.png",
) -> None:
    """
    Saves a two-panel chart: inertia elbow (left) and silhouette scores (right).
    The intersection of the elbow point and the silhouette peak gives the
    most defensible choice of k.
    """
    import os; os.makedirs("outputs", exist_ok=True)

    k_vals  = optimal_k_result["k_range"]
    inertia = optimal_k_result["inertias"]
    sils    = optimal_k_result["silhouettes"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_vals, inertia, marker="o", color="#2196F3")
    axes[0].set_title("inertia (elbow method)")
    axes[0].set_xlabel("number of clusters k")
    axes[0].set_ylabel("inertia")
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_vals, sils, marker="o", color="#4CAF50")
    axes[1].axvline(optimal_k_result["best_k"], color="red", linestyle="--",
                    label=f"best k = {optimal_k_result['best_k']}")
    axes[1].set_title("silhouette score")
    axes[1].set_xlabel("number of clusters k")
    axes[1].set_ylabel("silhouette score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def plot_cluster_heatmap(
    log_returns:    pd.DataFrame,
    cluster_result: pd.DataFrame,
    save_path:      str = "outputs/cluster_heatmap.png",
) -> None:
    """
    Plots the correlation matrix, sorted by cluster assignment.
    Within-cluster correlations should appear as darker blocks along the diagonal
    if the clustering is capturing real grouping structure.
    """
    import os; os.makedirs("outputs", exist_ok=True)

    ordered_syms = cluster_result.sort_values("cluster")["symbol"].tolist()
    corr_ordered = log_returns[ordered_syms].corr()

    # cluster boundary lines
    cluster_sizes = cluster_result.groupby("cluster")["symbol"].count().values
    boundaries    = np.cumsum(cluster_sizes)[:-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_ordered.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(ordered_syms)))
    ax.set_yticks(range(len(ordered_syms)))
    ax.set_xticklabels(ordered_syms, rotation=90, fontsize=7)
    ax.set_yticklabels(ordered_syms, fontsize=7)

    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=1.5)
        ax.axvline(b - 0.5, color="black", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="correlation")
    ax.set_title("correlation matrix sorted by K-means cluster")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {save_path}")


def run_clustering_analysis(
    log_returns: pd.DataFrame,
    k_range:     range = range(2, 9),
    save_plots:  bool  = True,
) -> dict:
    """
    Full clustering pipeline. Returns cluster assignments and optimal k metadata.
    """
    symbols  = log_returns.columns.tolist()
    features = compute_correlation_features(log_returns)

    ok_result    = optimal_k(features, k_range)
    best_k       = ok_result["best_k"]
    cluster_df   = fit_clusters(features, best_k, symbols)

    print(f"  K-means clustering: best k = {best_k} "
          f"(silhouette = {max(ok_result['silhouettes']):.3f})")
    for cid, grp in cluster_df.groupby("cluster"):
        print(f"    cluster {cid}: {', '.join(grp['symbol'].tolist())}")

    if save_plots:
        plot_elbow_and_silhouette(ok_result)
        plot_cluster_heatmap(log_returns, cluster_df)

    return {
        "cluster_assignments": cluster_df,
        "optimal_k_analysis":  ok_result,
        "best_k":              best_k,
    }
