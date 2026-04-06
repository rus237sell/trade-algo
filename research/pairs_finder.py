import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import config


def adf_test(series: pd.Series, significance: float = 0.05) -> tuple[bool, float]:
    """
    Augmented Dickey-Fuller test for stationarity.

    H0: the series has a unit root (non-stationary / random walk)
    H1: the series is stationary

    We reject H0 if p-value < significance level.
    A stationary spread is the necessary condition for mean reversion.

    Returns: (is_stationary, p_value)
    """
    result = adfuller(series.dropna(), autolag="AIC")
    p_value = result[1]
    return p_value < significance, p_value


def engle_granger_coint(y: pd.Series, x: pd.Series, significance: float = 0.05) -> tuple[bool, float]:
    """
    Engle-Granger two-step cointegration test.

    Step 1: Estimate the long-run relationship Y = alpha + beta*X + epsilon
    Step 2: Test whether the residuals (epsilon) are stationary

    If the residuals are stationary, Y and X are cointegrated — they share
    a common stochastic trend and will not drift apart indefinitely.

    Returns: (is_cointegrated, p_value)
    """
    score, p_value, _ = coint(y, x)
    return p_value < significance, p_value


def compute_ols_beta(y: pd.Series, x: pd.Series) -> float:
    """
    OLS estimate of the hedge ratio beta in: Y = alpha + beta*X + epsilon

    This beta is the dollar hedge ratio — for every $1 long in Y,
    we short $beta of X to create a beta-neutral spread.

    We use the full-sample OLS beta for the formation period.
    Time-varying rolling beta is handled in spread_model.py.
    """
    x_const = add_constant(x.values)
    model = OLS(y.values, x_const).fit()
    alpha, beta = model.params
    return beta


def compute_half_life(spread: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck mean-reversion half-life.

    The OU process: dS_t = theta * (mu - S_t) dt + sigma * dW_t
    Discretized: delta_S_t = a + b * S_{t-1} + epsilon_t

    Where b ≈ -theta (mean-reversion speed).
    Half-life = -ln(2) / b

    A short half-life (< 30 days) means fast reversion — good.
    A long half-life (> 120 days) means the spread reverts slowly
    and ties up capital for too long.

    Returns inf if the spread is not mean-reverting (b >= 0).
    """
    spread_lag   = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()

    # align
    idx = spread_lag.index.intersection(delta_spread.index)
    spread_lag   = spread_lag.loc[idx]
    delta_spread = delta_spread.loc[idx]

    x_const = add_constant(spread_lag.values)
    model = OLS(delta_spread.values, x_const).fit()
    b = model.params[1]   # coefficient on lagged level

    if b >= 0:
        return np.inf   # not mean-reverting

    half_life = -np.log(2) / b
    return half_life


def rolling_adf_health(spread: pd.Series, window: int = None) -> float:
    """
    ADF p-value on the most recent window of the spread.
    If p > config.ADF_RETIRE_THRESHOLD the pair is flagged for retirement.
    Runs daily as a lightweight check between quarterly full rescans.
    """
    if window is None:
        window = config.ADF_HEALTH_WINDOW
    recent = spread.dropna().iloc[-window:]
    if len(recent) < 30:
        return 1.0
    _, p_value = adf_test(recent)
    return p_value


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    formation_start: int,
    formation_end: int
) -> list[dict]:
    """
    Scans all pairs in the universe for cointegration over the formation window.

    For each candidate pair (Y, X):
      1. Run Engle-Granger cointegration test
      2. If cointegrated, compute OLS beta and the spread
      3. Run ADF on the spread as a robustness check
      4. Compute OU half-life and filter out slow-reverting pairs

    Returns a list of dicts, one per accepted pair, containing:
      - y_sym, x_sym
      - beta (hedge ratio)
      - half_life
      - coint_pvalue
      - adf_pvalue
    """
    symbols  = prices.columns.tolist()
    window   = prices.iloc[formation_start:formation_end]
    accepted = []

    for y_sym, x_sym in itertools.combinations(symbols, 2):
        y = window[y_sym].dropna()
        x = window[x_sym].dropna()

        # need aligned, complete data
        aligned = pd.concat([y, x], axis=1).dropna()
        if len(aligned) < 120:
            continue

        y_aligned = aligned.iloc[:, 0]
        x_aligned = aligned.iloc[:, 1]

        # cointegration test
        is_coint, coint_p = engle_granger_coint(y_aligned, x_aligned, config.COINTEGRATION_PVALUE)
        if not is_coint:
            continue

        # OLS hedge ratio on formation window
        beta = compute_ols_beta(y_aligned, x_aligned)
        if beta <= 0:
            # negative beta doesn't make structural sense for most equity pairs
            continue

        # spread and stationarity check
        spread = y_aligned - beta * x_aligned

        is_stat, adf_p = adf_test(spread, config.COINTEGRATION_PVALUE)
        if not is_stat:
            continue

        # OU half-life filter
        half_life = compute_half_life(spread)
        if half_life > config.MAX_HALF_LIFE_DAYS or half_life <= 1:
            continue

        accepted.append({
            "y_sym":       y_sym,
            "x_sym":       x_sym,
            "beta":        round(beta, 4),
            "half_life":   round(half_life, 1),
            "coint_pvalue": round(coint_p, 4),
            "adf_pvalue":  round(adf_p, 4),
        })

    return accepted
