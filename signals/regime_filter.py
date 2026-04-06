"""
Regime Filter — DFA-1 + ADX, continuous sigmoid scoring

Replaces the original R/S Hurst + binary AND gate.

The R/S estimator biases Hurst upward by +0.05 to +0.10 on trending equities,
making H < 0.45 unreachable 77–87% of trading days on large-cap US equities.
DFA-1 removes linear trends within each box by construction, making it
drift-robust where R/S is not.

Binary gating (H < 0.45 AND ADX < 25) multiplies pass rates. At 40% each
the AND gate passes 16% of days. The observed system was generating ~0.6 trades
per year — below any statistically testable threshold (minimum ~30/year).
Sigmoid weighting replaces the hard gate: each indicator produces a
continuous score in [0, 1], and position size scales with the product.

Primary usage:
    dfa    = rolling_dfa(prices, window=504)
    adx    = compute_adx(high, low, close)
    score  = composite_score(dfa, adx, kalman_z)
    # score near 1.0 = strong mean-reversion conditions
    # score near 0.0 = trending regime, scale position down accordingly
"""

import numpy as np
import pandas as pd
import config

try:
    import nolds
    _NOLDS_AVAILABLE = True
except ImportError:
    _NOLDS_AVAILABLE = False


# ---- DFA-1 (Detrended Fluctuation Analysis) ----------------------------------

def dfa_exponent(series: pd.Series) -> float:
    """
    Computes the DFA-1 scaling exponent on a return series.

    DFA-1 fits and subtracts a linear trend within each non-overlapping box,
    then measures the RMS of residuals F(n) across box sizes n. The scaling
    exponent alpha from log-log regression of F(n) ~ n^alpha approximates
    the Hurst exponent for stationary signals, but is robust to deterministic
    drift by construction — the key advantage over R/S analysis.

    alpha < 0.5  : mean-reverting (anti-persistent)
    alpha ~ 0.5  : random walk
    alpha > 0.5  : trending / persistent

    Requires the `nolds` library. Returns np.nan if unavailable or insufficient data.
    """
    if not _NOLDS_AVAILABLE:
        return np.nan

    vals = series.dropna().values
    if len(vals) < 20:
        return np.nan

    try:
        return float(nolds.dfa(vals, nvals=None, overlap=True, order=1))
    except Exception:
        return np.nan


def rolling_dfa(
    prices: pd.Series,
    window: int = 504,   # 2 years — recommended for stability
    step:   int = 5,     # recompute every 5 days to reduce compute cost
) -> pd.Series:
    """
    Rolling DFA-1 exponent on log-returns.

    Window of 504 days provides stable estimates with moderate upward bias
    (+0.02 to +0.04), substantially better than R/S (+0.05 to +0.10).
    Forward-fills between computation steps — regime shifts are slow enough
    that daily recomputation adds noise without meaningful signal.

    Returns a Series aligned to prices.index.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    n       = len(log_ret)
    result  = pd.Series(np.nan, index=log_ret.index)

    calc_indices = list(range(window, n, step))
    if not calc_indices or calc_indices[-1] != n - 1:
        calc_indices.append(n - 1)

    for i in calc_indices:
        window_data = log_ret.iloc[max(0, i - window + 1): i + 1]
        result.iloc[i] = dfa_exponent(window_data)

    result = result.ffill()
    return result.reindex(prices.index).ffill()


def dfa_percentile_threshold(
    dfa_series: pd.Series,
    lookback:   int   = 504,   # 2-year rolling window for threshold
    percentile: float = 0.30,  # enter MR when DFA is in bottom 30th pct
) -> pd.Series:
    """
    Percentile-based entry threshold for the DFA signal.

    Different assets have systematically different Hurst distributions — US
    large-cap clusters at 0.48–0.54 (DFA), FX majors at 0.48–0.52, crypto
    at 0.55–0.70. A fixed cutoff ignores this. Computing the threshold from
    the asset's own trailing distribution adapts automatically.

    Returns boolean Series: True = mean-reverting conditions detected.
    """
    rolling_thresh = dfa_series.rolling(lookback, min_periods=lookback // 2).quantile(percentile)
    return dfa_series < rolling_thresh


# ---- Average Directional Index (ADX) ----------------------------------------

def compute_adx(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Wilder's Average Directional Index. Values > 25 indicate a trending market.
    Preferred over compute_adx_from_close() when OHLC data is available.
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha   = 1.0 / period
    s_plus  = pd.Series(plus_dm,  index=close.index).ewm(alpha=alpha, adjust=False).mean()
    s_minus = pd.Series(minus_dm, index=close.index).ewm(alpha=alpha, adjust=False).mean()
    s_tr    = tr.ewm(alpha=alpha, adjust=False).mean()

    plus_di  = 100 * s_plus  / s_tr.replace(0, np.nan)
    minus_di = 100 * s_minus / s_tr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    return dx.ewm(alpha=alpha, adjust=False).mean()


def compute_adx_from_close(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Approximates ADX from close-only data.
    Underestimates ADX in volatile markets but preserves the trend signal.
    Use compute_adx() if OHLC data is available.
    """
    ret     = close.diff().abs()
    up_move = close.diff().clip(lower=0)
    dn_move = (-close.diff()).clip(lower=0)
    alpha   = 1.0 / period

    s_up  = up_move.ewm(alpha=alpha, adjust=False).mean()
    s_dn  = dn_move.ewm(alpha=alpha, adjust=False).mean()
    s_tr  = ret.ewm(alpha=alpha, adjust=False).mean()

    plus_di  = 100 * s_up / s_tr.replace(0, np.nan)
    minus_di = 100 * s_dn / s_tr.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    return dx.ewm(alpha=alpha, adjust=False).mean()


# ---- Continuous Sigmoid Scoring ----------------------------------------------

def sigmoid_gate(x: pd.Series, threshold: float, sharpness: float = 10.0) -> pd.Series:
    """
    Maps an indicator to a continuous weight in [0, 1].

    Near 1.0 when x << threshold (favorable mean-reversion conditions).
    Near 0.0 when x >> threshold (trending — scale position down).

    sharpness controls transition steepness:
      high sharpness = approximates the old binary gate
      low sharpness  = gradual scaling across a wider range
    """
    return 1.0 / (1.0 + np.exp(sharpness * (x - threshold)))


def composite_score(
    dfa_series:    pd.Series,
    adx_series:    pd.Series,
    kalman_z:      pd.Series,
    dfa_threshold: float = 0.50,    # DFA ~ 0.50 is the random walk boundary
    adx_threshold: float = 25.0,
    dfa_sharpness: float = 20.0,
    adx_sharpness: float = 0.3,
) -> pd.Series:
    """
    Combines DFA and ADX into a single continuous score in [0, 1].

    Replaces the binary AND gate. Each component produces a sigmoid weight;
    the composite is their product multiplied by a z-score entry ramp.
    Scores never collapse to exactly zero — unfavorable regimes reduce
    position size, they do not eliminate it entirely.

    Components:
      w_dfa   : near 1 when DFA alpha < 0.5 (mean-reverting residuals)
      w_adx   : near 1 when ADX < 25 (weak trend / ranging market)
      w_entry : linear ramp 0→1 as |kalman_z| goes from 0 to 2-sigma

    Returns:
      score in [0, 1] — multiply by target position size to scale allocation.
    """
    w_dfa   = sigmoid_gate(dfa_series, dfa_threshold, sharpness=dfa_sharpness)
    w_adx   = sigmoid_gate(adx_series, adx_threshold, sharpness=adx_sharpness)
    w_entry = np.clip(kalman_z.abs() / 2.0, 0, 1)

    return (w_dfa * w_adx * w_entry).fillna(0.0)


def threshold_gate(signal, composite_score, min_threshold=None, full_threshold=None):
    """
    Three-zone gate — replaces raw multiplication of signal by DFA score.

    Below min_threshold: no trade (signal = 0, transaction cost avoided)
    Between thresholds:  signal scaled proportionally by score
    Above full_threshold: full signal passed unchanged

    Works on scalars or aligned pandas Series.
    """
    if min_threshold is None:
        min_threshold = config.DFA_MIN_THRESHOLD
    if full_threshold is None:
        full_threshold = config.DFA_FULL_THRESHOLD

    if isinstance(composite_score, pd.Series):
        cs  = composite_score.values
        sig = signal.values if isinstance(signal, pd.Series) else np.full(len(cs), float(signal))
        out = np.where(cs < min_threshold, 0.0,
              np.where(cs < full_threshold, sig * cs, sig))
        return pd.Series(out, index=composite_score.index)

    if composite_score < min_threshold:
        return 0.0
    if composite_score < full_threshold:
        return float(signal) * composite_score
    return signal


# ---- Legacy Binary Gate (retained for benchmark comparison) ------------------

def compute_regime_gate(
    prices:       pd.Series,
    hurst_window: int   = 126,
    adx_period:   int   = 14,
    hurst_thresh: float = 0.45,
    adx_thresh:   float = 25.0,
    high:         pd.Series = None,
    low:          pd.Series = None,
) -> tuple:
    """
    Original binary R/S Hurst + ADX gate. Retained for benchmark comparison only.

    NOTE: This estimator is biased upward by +0.05 to +0.10 on trending equities.
    The H < 0.45 gate will be unreachable 77–87% of trading days on large-cap
    US equities with typical upward drift. Use rolling_dfa() + composite_score().
    """
    hurst_series = _rolling_hurst_rs(prices, window=hurst_window)
    if high is not None and low is not None:
        adx_series = compute_adx(high, low, prices, adx_period)
    else:
        adx_series = compute_adx_from_close(prices, adx_period)

    gate = ((hurst_series < hurst_thresh) & (adx_series < adx_thresh)).fillna(0).astype(int)
    return hurst_series, adx_series, gate


def _rolling_hurst_rs(prices: pd.Series, window: int = 126, step: int = 5) -> pd.Series:
    """Legacy R/S Hurst. Retained for benchmark comparison. Use rolling_dfa() instead."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    n       = len(log_ret)
    result  = pd.Series(np.nan, index=log_ret.index)

    def _single(series):
        vals = series.dropna().values
        n_   = len(vals)
        if n_ < 20:
            return np.nan
        lags, rs = [], []
        for lag in range(10, n_ // 2):
            n_chunks = n_ // lag
            if n_chunks < 2:
                break
            rs_vals = []
            for i in range(n_chunks):
                chunk = vals[i * lag:(i + 1) * lag]
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
        A = np.vstack([lags, np.ones(len(lags))]).T
        slope, _ = np.linalg.lstsq(A, rs, rcond=None)[0]
        return float(slope)

    for i in range(window, n, step):
        result.iloc[i] = _single(log_ret.iloc[max(0, i - window):i + 1])

    return result.ffill().reindex(prices.index).ffill()
