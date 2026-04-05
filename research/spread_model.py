import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import config


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Time-varying hedge ratio via rolling OLS.

    Using a fixed formation-period beta in the trading period is a common
    mistake. As the structural relationship between two stocks shifts over
    time (e.g., one diversifies away from the other), the fixed beta
    mismodes the true hedge and introduces directional leakage into the spread.

    Rolling OLS re-estimates beta over the most recent `window` days,
    allowing the hedge to adapt.

    Returns a Series of beta estimates, indexed by date.
    """
    betas = pd.Series(index=y.index, dtype=float)

    for i in range(window, len(y) + 1):
        y_window = y.iloc[i - window: i]
        x_window = x.iloc[i - window: i]
        x_const = add_constant(x_window.values)
        model = OLS(y_window.values, x_const).fit()
        betas.iloc[i - 1] = model.params[1]   # coefficient on X

    return betas


def compute_spread(y: pd.Series, x: pd.Series, beta: pd.Series) -> pd.Series:
    """
    Beta-neutral spread: S_t = Y_t - beta_t * X_t

    When beta is time-varying (from rolling OLS), we use the prior day's
    beta to avoid lookahead. This is handled by shifting beta by 1 before
    using it.
    """
    beta_lagged = beta.shift(1)
    spread = y - beta_lagged * x
    return spread


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Z-score normalization of the spread over a rolling window.

    z_t = (S_t - mean(S_{t-window:t})) / std(S_{t-window:t})

    This converts the spread into a dimensionless measure of how far
    the current spread is from its recent mean. The z-score is what
    drives entry and exit decisions.

    Values near 0: spread near equilibrium
    Values > +2:   spread stretched high → short the spread
    Values < -2:   spread stretched low  → long the spread
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std  = spread.rolling(window).std()

    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = config.ZSCORE_ENTRY,
    exit_threshold:  float = config.ZSCORE_EXIT,
    stop_threshold:  float = config.ZSCORE_STOP
) -> pd.Series:
    """
    Generates raw Long/Short signals from the z-score time series.

    Signal logic:
      Enter long  (+1): z < -entry  (spread below equilibrium, expect upward reversion)
      Enter short (-1): z >  entry  (spread above equilibrium, expect downward reversion)
      Exit:           |z| < exit    (spread reverted to near-mean)
      Stop-loss:      |z| > stop    (spread diverging further — cut loss)

    State machine: once in a trade, hold until exit or stop is triggered.
    No pyramiding — size is 0 or ±1 at the primary signal level.

    Returns a Series of {-1, 0, 1}.
    """
    signals = pd.Series(0, index=zscore.index, dtype=float)
    position = 0

    for i, (date, z) in enumerate(zscore.items()):
        if np.isnan(z):
            signals[date] = 0
            continue

        if position == 0:
            # look for entry
            if z < -entry_threshold:
                position = 1
            elif z > entry_threshold:
                position = -1

        elif position == 1:
            # in long — exit on mean reversion or stop on further divergence
            if z > -exit_threshold or z < -stop_threshold:
                position = 0

        elif position == -1:
            # in short — exit on mean reversion or stop on further divergence
            if z < exit_threshold or z > stop_threshold:
                position = 0

        signals[date] = position

    return signals


def build_pair_model(
    prices:   pd.DataFrame,
    y_sym:    str,
    x_sym:    str,
    beta_window: int = config.ROLLING_BETA_WINDOW,
    z_window:    int = config.ZSCORE_WINDOW
) -> dict:
    """
    Full pipeline for a single pair: rolling beta → spread → z-score → signals.

    Returns a dict with all intermediate series, useful for research and
    debugging as well as passing to the backtester and meta-labeler.
    """
    y = prices[y_sym]
    x = prices[x_sym]

    beta   = rolling_ols_beta(y, x, beta_window)
    spread = compute_spread(y, x, beta)
    z      = rolling_zscore(spread, z_window)
    sigs   = generate_signals(z)

    return {
        "beta":    beta,
        "spread":  spread,
        "zscore":  z,
        "signals": sigs,
    }
