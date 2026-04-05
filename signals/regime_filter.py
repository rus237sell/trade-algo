"""
Regime Filter: Hurst Exponent + ADX Gate

The single biggest structural flaw in naive mean-reversion strategies is
trading in trending markets. This module implements a dual filter that
prevents new mean-reversion entries when the market is trending:

  Gate 1 — Hurst Exponent (H):
    H < 0.45: mean-reverting regime   -> mean-reversion trades ALLOWED
    H = 0.50: random walk             -> no edge
    H > 0.55: trending regime         -> mean-reversion trades BLOCKED

    The buffer (0.45 rather than 0.50) accounts for estimation error in
    the Hurst calculation. The Hurst exponent is computed via Rescaled
    Range (R/S) analysis on a rolling window.

  Gate 2 — Average Directional Index (ADX):
    ADX < 20: weak trend / ranging    -> mean-reversion trades ALLOWED
    ADX > 25: strong trend            -> mean-reversion trades BLOCKED

    ADX measures trend strength (not direction). A rising ADX means the
    trend is strengthening regardless of whether prices are going up or down.
    The 25 threshold is standard in technical analysis literature.

Combined gate: ALLOW entry only when BOTH H < 0.45 AND ADX < 25.
Either filter alone misses cases. H catches long-term trending behavior;
ADX catches short-term momentum breakouts before H adapts.

Research basis:
  - Hurst 1951 (R/S analysis), Mandelbrot & Wallis 1969 (financial application)
  - Wilder 1978 (ADX in "New Concepts in Technical Trading Systems")
  - Practical threshold values from Chan 2013 "Algorithmic Trading"
"""

import numpy as np
import pandas as pd


# ---- Hurst Exponent via Rescaled Range (R/S) --------------------------------

def hurst_rs(series: pd.Series, min_window: int = 10) -> float:
    """
    Estimates the Hurst exponent using Rescaled Range (R/S) analysis.

    Procedure:
      1. Split the series into sub-periods of decreasing length n
      2. For each sub-period: compute range R = max - min of cumulative deviations,
         compute std S of the raw values, compute R/S
      3. Regress log(R/S) on log(n) — the slope is the Hurst exponent

    H < 0.5: anti-persistent (mean reverting)
    H = 0.5: random walk (Brownian motion)
    H > 0.5: persistent (trending)

    Returns nan if insufficient data.
    """
    vals = series.dropna().values
    n    = len(vals)

    if n < min_window * 2:
        return np.nan

    lags = []
    rs   = []

    for lag in range(min_window, n // 2):
        # split into chunks of length lag
        n_chunks = n // lag
        if n_chunks < 2:
            break

        rs_vals = []
        for i in range(n_chunks):
            chunk = vals[i * lag: (i + 1) * lag]
            mean  = np.mean(chunk)
            dev   = np.cumsum(chunk - mean)
            R     = np.max(dev) - np.min(dev)
            S     = np.std(chunk, ddof=1)
            if S > 0:
                rs_vals.append(R / S)

        if rs_vals:
            lags.append(np.log(lag))
            rs.append(np.log(np.mean(rs_vals)))

    if len(lags) < 5:
        return np.nan

    # OLS regression of log(R/S) on log(n)
    A = np.vstack([lags, np.ones(len(lags))]).T
    slope, _ = np.linalg.lstsq(A, rs, rcond=None)[0]
    return float(slope)


def rolling_hurst(
    prices:     pd.Series,
    window:     int = 126,    # 6 months of daily data
    min_window: int = 20,
    step:       int = 5,      # recompute every 5 days (expensive calculation)
) -> pd.Series:
    """
    Computes Hurst exponent on a rolling window.

    Hurst is computed on log-returns (not prices) to remove the random
    walk trend component. This gives a cleaner measure of the local
    mean-reversion or trending behavior of returns.

    The step parameter reduces computation: Hurst is interpolated between
    calculation points. This is acceptable because regime changes are slow.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    n       = len(log_ret)
    hurst   = pd.Series(np.nan, index=log_ret.index)

    calc_indices = list(range(window, n, step)) + [n - 1]

    for i in calc_indices:
        window_data = log_ret.iloc[max(0, i - window): i + 1]
        h           = hurst_rs(window_data, min_window)
        if i < len(hurst):
            hurst.iloc[i] = h

    # forward-fill between calculation points (regime doesn't change intraday)
    hurst = hurst.ffill()

    # align back to original price index
    return hurst.reindex(prices.index).ffill()


# ---- Average Directional Index (ADX) ----------------------------------------

def compute_adx(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder's Average Directional Index.

    Steps:
      1. Compute +DM, -DM (directional movement)
      2. Compute True Range (TR)
      3. Smooth all three with Wilder's EMA (alpha = 1/period)
      4. +DI = 100 * smoothed(+DM) / smoothed(TR)
         -DI = 100 * smoothed(-DM) / smoothed(TR)
      5. DX  = 100 * |+DI - -DI| / (+DI + -DI)
      6. ADX = Wilder's EMA of DX

    Returns ADX series. Values > 25 indicate a trending market.
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move,  0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high  - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / period

    s_plus_dm  = pd.Series(plus_dm,  index=close.index).ewm(alpha=alpha, adjust=False).mean()
    s_minus_dm = pd.Series(minus_dm, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    s_tr       = tr.ewm(alpha=alpha, adjust=False).mean()

    plus_di  = 100 * s_plus_dm  / s_tr.replace(0, np.nan)
    minus_di = 100 * s_minus_dm / s_tr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


def compute_adx_from_close(
    close:  pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Approximates ADX when only close prices are available (no OHLC).

    Uses daily returns as a proxy for high-low range. This underestimates
    ADX in volatile markets but preserves the trend-detection signal.
    Prefer compute_adx() if OHLC data is available.
    """
    ret       = close.diff().abs()
    up_move   = close.diff().clip(lower=0)
    down_move = (-close.diff()).clip(lower=0)

    alpha = 1.0 / period

    s_up   = up_move.ewm(alpha=alpha, adjust=False).mean()
    s_down = down_move.ewm(alpha=alpha, adjust=False).mean()
    s_tr   = ret.ewm(alpha=alpha, adjust=False).mean()

    plus_di  = 100 * s_up   / s_tr.replace(0, np.nan)
    minus_di = 100 * s_down / s_tr.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


# ---- Combined Regime Gate ----------------------------------------------------

def compute_regime_gate(
    prices:         pd.Series,
    hurst_window:   int   = 126,
    adx_period:     int   = 14,
    hurst_thresh:   float = 0.45,
    adx_thresh:     float = 25.0,
    high:           pd.Series = None,
    low:            pd.Series = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Combines Hurst and ADX filters into a single binary gate.

    gate = 1: mean-reversion entries are ALLOWED (H < thresh AND ADX < thresh)
    gate = 0: mean-reversion entries are BLOCKED (trending regime detected)

    The gate only blocks NEW entries — it does not force-exit existing trades.
    Exits are still governed by the z-score mean-cross and ATR stop logic.

    Returns:
      hurst_series: rolling Hurst exponent
      adx_series:   ADX indicator
      gate:         binary Series (1 = allow entry, 0 = block entry)
    """
    hurst_series = rolling_hurst(prices, window=hurst_window)

    if high is not None and low is not None:
        adx_series = compute_adx(high, low, prices, adx_period)
    else:
        adx_series = compute_adx_from_close(prices, adx_period)

    hurst_ok = (hurst_series < hurst_thresh).astype(int)
    adx_ok   = (adx_series   < adx_thresh).astype(int)

    gate = (hurst_ok & adx_ok).fillna(0).astype(int)

    return hurst_series, adx_series, gate
