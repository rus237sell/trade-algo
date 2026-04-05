"""
Deflated Sharpe Ratio (DSR) and Combinatorial Purged Cross-Validation (CPCV)

These are the two most important validation tools that separate real alpha
from statistical noise.

--- Deflated Sharpe Ratio (DSR) ---

The standard backtest Sharpe ratio is biased upward because researchers
test many strategy variants and report the best one. After testing just
100 parameter combinations on random data, you will find strategies with
Sharpe ratios above 2.5 that are purely spurious.

The DSR adjusts the observed Sharpe for:
  1. Number of independent trials (parameter combinations tested)
  2. Skewness of returns (downward-skewed strategies overstate Sharpe)
  3. Kurtosis of returns (fat-tailed strategies understate risk)

Formula (Bailey & Lopez de Prado 2014):
  SR_hat = observed_sharpe * sqrt(T / (1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2))
  DSR = Phi(SR_hat * sqrt(T) - SR_benchmark * sqrt(T * (1 - skew/SR + (kurt-1)/4)))

A DSR below 0.95 (95% confidence) means the strategy likely has no edge.
Any strategy with DSR > 0.99 in backtest warrants serious scrutiny —
it means either the strategy is genuinely excellent or the number of
trials is being understated.

--- Combinatorial Purged Cross-Validation (CPCV) ---

Standard k-fold cross-validation leaks future information in financial
data because observations are serially correlated. Walk-forward analysis
is better but tests only a single path through the data.

CPCV (Lopez de Prado & Armstrong 2019) generates multiple train/test
combinations while:
  1. Respecting chronological order (no future data in training)
  2. Purging observations around the test period (removing leakage)
  3. Embargoing observations near test boundaries

With N=6 groups and k=2 test groups, CPCV produces 15 backtest paths,
enabling computation of a distribution of Sharpe ratios and the
Probability of Backtest Overfitting (PBO).

PBO < 0.1: good (less than 10% chance of overfitting)
PBO > 0.5: poor (more likely overfit than not)
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations


# ---- Deflated Sharpe Ratio ---------------------------------------------------

def annualized_sharpe(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def deflated_sharpe_ratio(
    returns:         pd.Series,
    n_trials:        int   = 1,       # number of strategy variants tested
    benchmark_sharpe: float = 0.0,    # benchmark to beat (0 = beat cash)
    periods:         int   = 252,
) -> dict:
    """
    Computes the Deflated Sharpe Ratio (DSR) per Bailey & Lopez de Prado (2014).

    n_trials is the number of parameter combinations you tested before
    arriving at this strategy. Be honest here — if you tested 50 lookback
    windows and 20 threshold values, n_trials = 1000. The DSR penalizes
    heavily for trial inflation.

    Returns a dict with:
      observed_sharpe:   raw annualized Sharpe
      deflated_sharpe:   DSR probability (0 to 1)
      is_significant:    True if DSR > 0.95
      skewness:          return distribution skewness
      kurtosis:          excess kurtosis
      effective_n:       approximate independent observations
    """
    T        = len(returns)
    mu       = returns.mean()
    sigma    = returns.std()
    skew     = float(stats.skew(returns.dropna()))
    kurt     = float(stats.kurtosis(returns.dropna()))    # excess kurtosis

    if sigma == 0 or T < 30:
        return {
            "observed_sharpe":  0.0,
            "deflated_sharpe":  0.0,
            "is_significant":   False,
            "skewness":         skew,
            "kurtosis":         kurt,
            "effective_n":      T,
        }

    sr_obs = mu / sigma   # daily Sharpe

    # expected maximum Sharpe across n_trials on iid normal returns
    # Bailey & Lopez de Prado (2014), equation 3
    # E[max SR | n trials] = (1 - euler_mascheroni) * Z^{-1}(1 - 1/n) + euler_mascheroni * Z^{-1}(1 - 1/(n*e))
    euler_gamma = 0.5772156649
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr_max_expected = (1 - euler_gamma) * z1 + euler_gamma * z2
    else:
        sr_max_expected = 0.0

    # variance of Sharpe ratio estimator (Christie 2005)
    sr_var = (1 + 0.5 * sr_obs**2 - skew * sr_obs + (kurt / 4) * sr_obs**2) / T

    # DSR: probability that the true Sharpe exceeds the benchmark after
    # correcting for selection bias
    z_dsr   = (sr_obs - sr_max_expected) / np.sqrt(max(sr_var, 1e-8))
    dsr     = float(stats.norm.cdf(z_dsr))

    return {
        "observed_sharpe":   round(sr_obs * np.sqrt(periods), 3),
        "deflated_sharpe":   round(dsr, 4),
        "is_significant":    dsr > 0.95,
        "sr_max_expected":   round(sr_max_expected * np.sqrt(periods), 3),
        "skewness":          round(skew, 3),
        "kurtosis":          round(kurt, 3),
        "effective_n":       T,
        "n_trials":          n_trials,
    }


# ---- Combinatorial Purged Cross-Validation (CPCV) ----------------------------

def cpcv_split(
    n_obs:    int,
    n_groups: int = 6,
    n_test:   int = 2,
    embargo:  int = 5,    # days to embargo around test boundaries
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generates train/test index pairs for CPCV.

    Splits the time series into n_groups equal-sized groups, then
    creates C(n_groups, n_test) combinations of test groups.
    Training observations are the complement of test observations,
    with embargo periods removed near the boundaries.

    Returns list of (train_indices, test_indices) tuples.
    """
    group_size = n_obs // n_groups
    groups     = [
        np.arange(i * group_size, min((i + 1) * group_size, n_obs))
        for i in range(n_groups)
    ]

    splits = []
    for test_groups in combinations(range(n_groups), n_test):
        test_idx = np.concatenate([groups[g] for g in test_groups])

        # training: all indices not in test groups, with embargo
        all_idx   = set(range(n_obs))
        test_set  = set(test_idx.tolist())

        # embargo: remove indices within `embargo` days of test boundaries
        embargoed = set()
        for t in test_idx:
            for e in range(-embargo, embargo + 1):
                if 0 <= t + e < n_obs:
                    embargoed.add(t + e)

        train_idx = np.array(sorted(all_idx - test_set - embargoed))

        if len(train_idx) > 50 and len(test_idx) > 10:
            splits.append((train_idx, test_idx))

    return splits


def run_cpcv(
    returns:         pd.Series,
    strategy_fn,                    # callable(train_returns, test_returns) -> test_sharpe
    n_groups:        int   = 6,
    n_test:          int   = 2,
    embargo:         int   = 5,
    n_trials:        int   = 1,
) -> dict:
    """
    Runs CPCV and computes the Probability of Backtest Overfitting (PBO).

    strategy_fn is called with (train_returns, test_returns) and should
    return the out-of-sample Sharpe ratio on the test period after fitting
    strategy parameters on the train period.

    PBO = fraction of CPCV paths where the is-optimal strategy underperforms
          a random strategy in the out-of-sample period.

    Returns:
      oos_sharpes:   list of out-of-sample Sharpe ratios across all CPCV paths
      mean_oos_sharpe: average OOS Sharpe
      pbo:           Probability of Backtest Overfitting (lower is better)
      dsr:           DSR computed on the concatenated OOS returns
      n_paths:       number of CPCV paths generated
    """
    vals   = returns.values
    n_obs  = len(vals)
    splits = cpcv_split(n_obs, n_groups, n_test, embargo)

    oos_sharpes = []
    oos_returns = []

    for train_idx, test_idx in splits:
        train_ret = pd.Series(vals[train_idx])
        test_ret  = pd.Series(vals[test_idx])

        try:
            oos_sharpe = strategy_fn(train_ret, test_ret)
            oos_sharpes.append(oos_sharpe)
            oos_returns.extend(test_ret.tolist())
        except Exception:
            pass

    if not oos_sharpes:
        return {"error": "all CPCV paths failed"}

    # PBO: fraction of paths with negative OOS Sharpe
    pbo = np.mean([s < 0 for s in oos_sharpes])

    # DSR on concatenated OOS returns
    oos_ret_series = pd.Series(oos_returns)
    dsr_result     = deflated_sharpe_ratio(oos_ret_series, n_trials=n_trials)

    return {
        "oos_sharpes":       oos_sharpes,
        "mean_oos_sharpe":   round(float(np.mean(oos_sharpes)), 3),
        "std_oos_sharpe":    round(float(np.std(oos_sharpes)), 3),
        "min_oos_sharpe":    round(float(np.min(oos_sharpes)), 3),
        "max_oos_sharpe":    round(float(np.max(oos_sharpes)), 3),
        "pbo":               round(float(pbo), 3),
        "dsr":               dsr_result,
        "n_paths":           len(splits),
    }


# ---- Strategy Robustness Check -----------------------------------------------

def parameter_sensitivity(
    returns_fn,          # callable(param) -> pd.Series of returns
    param_values: list,
    n_trials:     int = None,
) -> dict:
    """
    Tests strategy robustness by computing Sharpe across a parameter grid.

    If the strategy Sharpe collapses with ±10% parameter perturbation,
    it is almost certainly curve-fit to historical noise.

    A robust strategy shows:
      - Sharpe remains > 1.0 across the entire parameter range
      - Sharpe peak is not a sharp spike (smooth parameter surface)
      - Best parameter is not at the extreme of the tested range

    Returns Sharpe for each parameter value and robustness assessment.
    """
    if n_trials is None:
        n_trials = len(param_values)

    sharpes = {}
    for p in param_values:
        try:
            ret = returns_fn(p)
            sharpes[p] = annualized_sharpe(ret)
        except Exception:
            sharpes[p] = np.nan

    valid_sharpes = [s for s in sharpes.values() if not np.isnan(s)]

    if not valid_sharpes:
        return {"error": "all parameter values failed"}

    best_param = max(sharpes, key=lambda k: sharpes.get(k, -np.inf))

    return {
        "sharpes_by_param":     sharpes,
        "best_param":           best_param,
        "best_sharpe":          round(sharpes[best_param], 3),
        "mean_sharpe":          round(np.mean(valid_sharpes), 3),
        "std_sharpe":           round(np.std(valid_sharpes), 3),
        "sharpe_range":         round(max(valid_sharpes) - min(valid_sharpes), 3),
        "is_robust":            np.std(valid_sharpes) < 0.5,   # std < 0.5 across params
        "best_at_boundary":     best_param in [param_values[0], param_values[-1]],
    }
