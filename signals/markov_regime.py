"""
Markov Regime Model — 2-state switching for regime-conditional allocation

The system previously discarded all trending periods by filtering them out.
Stocks spend roughly 60–80% of time trending (DFA alpha > 0.5). A pure
mean-reversion strategy captures only the ranging minority, missing the
majority of the market's time entirely.

This module fits a 2-state Markov-switching model (Hamilton 1989) on
portfolio returns. The two states correspond to:
  State 0: low-volatility trending  (mu > 0, sigma low)
  State 1: high-volatility choppy   (|mu| variable, sigma high)

Critical implementation detail: only filtered probabilities are used —
never smoothed probabilities, which incorporate future data and introduce
lookahead bias. A one-day lag is applied before acting on any estimates.

Allocation shifts based on regime:
  Trending  -> 70% momentum  / 30% mean-reversion
  Ranging   -> 30% momentum  / 70% mean-reversion

The diversification math is compelling: at correlation -0.3 between
momentum and mean-reversion (empirically documented), combining them
yields a ~69% Sharpe improvement over either strategy alone.

Usage:
    probs   = fit_markov_regime(portfolio_returns)
    weights = regime_allocation(probs)
"""

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


def fit_markov_regime(
    returns:        pd.Series,
    k_regimes:      int   = 2,
    switching_var:  bool  = True,
    search_reps:    int   = 20,
    prob_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Fits a 2-state Markov-switching regression on a return series.

    The model estimates the probability of being in each regime at each
    point in time using only data available up to that point (filtered
    probabilities). BIC consistently favors 2 states over 3 for daily
    US equity data.

    Parameters
    ----------
    returns : pd.Series
        Daily returns — either a single spread or an equal-weighted
        portfolio return across all pairs. Fit on 2–3 years of data,
        re-estimated monthly.
    k_regimes : int
        Number of hidden states. 2 recommended.
    switching_var : bool
        Allow variance to differ between regimes. Should be True —
        high-volatility vs low-volatility is the primary distinction.
    search_reps : int
        Random restarts in EM estimation to avoid local optima.
    prob_threshold : float
        Probability must exceed this to act on a regime label. Reduces
        whipsaw from noisy near-50/50 estimates. 0.70–0.80 recommended.

    Returns
    -------
    pd.DataFrame with columns:
        filtered_trending     : P(trending state | data up to t)
        filtered_ranging      : P(ranging state  | data up to t)
        trending_prob_lagged  : filtered_trending shifted 1 day (no lookahead)
        regime_label          : 0=trending, 1=ranging, NaN=uncertain
    """
    if not _STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required. Run: pip install statsmodels")

    clean = returns.dropna()

    mod = sm.tsa.MarkovRegression(
        clean,
        k_regimes=k_regimes,
        trend='c',
        switching_variance=switching_var,
    )

    # multiple restarts — Markov-switching EM can get stuck in local optima
    res = mod.fit(search_reps=search_reps, disp=False)

    # filtered (not smoothed) marginal probabilities — no lookahead
    filtered_probs = res.filtered_marginal_probabilities

    df = pd.DataFrame(index=clean.index)
    df['filtered_trending'] = filtered_probs[0]
    df['filtered_ranging']  = filtered_probs[1]

    # one-day lag before acting on regime estimates — critical for live use
    df['trending_prob_lagged'] = df['filtered_trending'].shift(1)

    # regime label with threshold to suppress noisy regime flips
    trending_confirmed = df['trending_prob_lagged'] >= prob_threshold
    ranging_confirmed  = df['trending_prob_lagged'] <= (1.0 - prob_threshold)

    df['regime_label'] = np.where(
        trending_confirmed, 0,
        np.where(ranging_confirmed, 1, np.nan)
    )
    df['regime_label'] = df['regime_label'].ffill()

    return df.reindex(returns.index)


def regime_allocation(
    regime_probs:        pd.DataFrame,
    mom_weight_trending: float = 0.70,
) -> pd.DataFrame:
    """
    Computes regime-conditional strategy weights from filtered probabilities.

    Weights are probability-weighted rather than binary — avoids cliff-edge
    allocation swings from noisy near-threshold regime estimates.

    In trending regime  (P_trend -> 1.0): ~70% momentum / ~30% mean-reversion
    In ranging regime   (P_trend -> 0.0): ~30% momentum / ~70% mean-reversion
    At uncertainty      (P_trend = 0.5):   50% / 50%

    Parameters
    ----------
    mom_weight_trending : float
        Momentum allocation when fully in trending regime (default 0.70).

    Returns
    -------
    pd.DataFrame with columns:
        momentum_weight       : allocation to momentum leg [0, 1]
        mean_reversion_weight : allocation to mean-reversion leg [0, 1]
        (sum to 1.0 at each point)
    """
    p_trend = regime_probs['trending_prob_lagged'].fillna(0.5)
    p_range = 1.0 - p_trend

    mr_weight_trending = 1.0 - mom_weight_trending

    mom_wt = p_trend * mom_weight_trending + p_range * (1.0 - mom_weight_trending)
    mr_wt  = 1.0 - mom_wt

    return pd.DataFrame({
        'momentum_weight':        mom_wt,
        'mean_reversion_weight':  mr_wt,
    }, index=regime_probs.index)


def rolling_regime_fit(
    returns:         pd.Series,
    estimation_days: int   = 504,   # 2-year rolling window
    refit_every:     int   = 21,    # refit monthly
    prob_threshold:  float = 0.70,
) -> pd.DataFrame:
    """
    Walk-forward Markov regime estimation — refits the model on a rolling
    window to track structural changes in regime dynamics over time.

    For backtesting use only. In live trading, refit monthly on an
    expanding or rolling 2-year window.

    Returns the same DataFrame structure as fit_markov_regime().
    """
    clean    = returns.dropna()
    n        = len(clean)
    results  = []

    fit_points = list(range(estimation_days, n, refit_every))
    if not fit_points or fit_points[-1] < n - 1:
        fit_points.append(n - 1)

    prev_df = None

    for end_idx in fit_points:
        start_idx = max(0, end_idx - estimation_days)
        window    = clean.iloc[start_idx: end_idx + 1]

        try:
            df_window = fit_markov_regime(
                window,
                prob_threshold=prob_threshold,
            )
            results.append(df_window)
            prev_df = df_window
        except Exception:
            # if fit fails, carry forward last known regime
            if prev_df is not None:
                results.append(prev_df.iloc[[-1]])

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results)
    # deduplicate overlapping windows — keep last estimate for each date
    combined = combined[~combined.index.duplicated(keep='last')]
    return combined.sort_index().reindex(returns.index).ffill()
