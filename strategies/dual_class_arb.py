"""
Dual-Class Share Arbitrage

Dual-class structures issue two share classes that represent identical economic
interests in the same underlying company but differ in voting rights. Because
they share the same cash flows, dividends, and book value, any price spread
between them is theoretically transient. This strategy trades that spread.

Key pairs:
  GOOGL / GOOG  — Class A (voting) vs Class C (no vote), Alphabet
  NWS   / NWSA  — Class B (voting) vs Class A (no vote), News Corp
  BRK.A / BRK.B — Class A vs Class B at a fixed 1500:1 theoretical ratio

The voting premium (Class A commanding a small persistent premium) is a known
structural feature, not an arbitrage. What IS tradeable is the deviation of
the spread from its own stable mean — not from theoretical parity.

Research basis: Wu (2015) Stanford CS229 — pairs trading on GOOG/GOOGL using
ML signal filtering achieved positive out-of-sample Sharpe after transaction
costs. The key insight was that the spread is mean-reverting within a narrow
band (~0.5–2%) and that simple z-score thresholds capture most of the alpha.

Why this is harder than it looks:
  - The spread is narrow (usually < 1%), so transaction costs eat most of the P&L
  - High-frequency participants arb this continuously during market hours
  - Execution must be nearly simultaneous on both legs or you take leg risk
  - Best suited for limit orders near the spread, not market orders
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import config


# known dual-class pairs with their theoretical conversion ratios
# ratio = shares of Class B equivalent to 1 Class A share
DUAL_CLASS_PAIRS = {
    "GOOGL_GOOG": {
        "class_a": "GOOGL",    # voting, typically commands a slight premium
        "class_b": "GOOG",     # no vote
        "ratio":   1.0,        # 1:1 economic equivalence
        "notes":   "Alphabet. Class A has 1 vote, Class C has 0. Premium ~0.1-0.8%."
    },
    "NWS_NWSA": {
        "class_a": "NWS",      # Class B, voting
        "class_b": "NWSA",     # Class A, limited vote
        "ratio":   1.0,
        "notes":   "News Corp. Murdoch family controls via Class B."
    },
    "BRK_A_B": {
        "class_a": "BRK-A",    # Class A, very high price
        "class_b": "BRK-B",    # Class B
        "ratio":   1500.0,     # 1 BRK.A = 1500 BRK.B economically
        "notes":   "Berkshire. No voting rights difference, just economic split."
    },
}


def fetch_pair_prices(pair_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads adjusted close prices for both share classes.
    For BRK.A/BRK.B, adjusts the B price by the 1500:1 ratio before computing
    the spread so both series are in the same dollar units.
    """
    pair  = DUAL_CLASS_PAIRS[pair_key]
    sym_a = pair["class_a"]
    sym_b = pair["class_b"]
    ratio = pair["ratio"]

    raw = yf.download([sym_a, sym_b], start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].ffill().dropna()

    if ratio != 1.0:
        # normalize B shares to A-share equivalent for meaningful spread
        prices[sym_b] = prices[sym_b] * ratio

    return prices


def compute_spread_and_premium(prices: pd.DataFrame, pair_key: str) -> pd.DataFrame:
    """
    Computes the price spread and voting premium percentage.

    For a 1:1 pair like GOOGL/GOOG:
      spread = GOOGL - GOOG
      premium_pct = (GOOGL - GOOG) / GOOG * 100

    A persistently positive premium_pct reflects the voting rights premium —
    this is structural and should NOT be traded away. What we trade is the
    deviation of premium_pct from its rolling mean.
    """
    pair  = DUAL_CLASS_PAIRS[pair_key]
    sym_a = pair["class_a"]
    sym_b = pair["class_b"]

    spread      = prices[sym_a] - prices[sym_b]
    premium_pct = (prices[sym_a] - prices[sym_b]) / prices[sym_b] * 100

    df = pd.DataFrame({
        "price_a":     prices[sym_a],
        "price_b":     prices[sym_b],
        "spread":      spread,
        "premium_pct": premium_pct,
    })

    return df


def rolling_zscore_premium(premium_pct: pd.Series, window: int = 30) -> pd.Series:
    """
    Z-score of the voting premium relative to its own rolling mean.

    We do NOT z-score around zero — we z-score around the rolling mean because
    the structural premium is non-zero and persistent. Trading toward the
    rolling mean captures the transient deviation; not the structural level.

    A short window (20-40 days) captures fast mean reversion.
    A long window (60-120 days) captures slower regime shifts.
    """
    rolling_mean = premium_pct.rolling(window).mean()
    rolling_std  = premium_pct.rolling(window).std()
    return (premium_pct - rolling_mean) / rolling_std


def generate_signals(
    zscore:          pd.Series,
    entry_threshold: float = 1.5,   # narrower than standard pairs — spread is tight
    exit_threshold:  float = 0.3,
    stop_threshold:  float = 2.5,
) -> pd.Series:
    """
    Signal logic identical in structure to the base pairs model but with
    tighter thresholds. The dual-class spread is narrow and reverts quickly,
    so waiting for 2.0 sigma (standard stat arb) would mean few trades.

    Long class_a / Short class_b (long spread): z < -entry
    Short class_a / Long class_b (short spread): z > +entry
    Exit: |z| < exit_threshold
    Stop:  |z| > stop_threshold (spread widening — structural break risk)
    """
    signals  = pd.Series(0, index=zscore.index, dtype=float)
    position = 0

    for date, z in zscore.items():
        if np.isnan(z):
            signals[date] = 0
            continue

        if position == 0:
            if z < -entry_threshold:
                position = 1
            elif z > entry_threshold:
                position = -1

        elif position == 1:
            if z > -exit_threshold or z < -stop_threshold:
                position = 0

        elif position == -1:
            if z < exit_threshold or z > stop_threshold:
                position = 0

        signals[date] = position

    return signals


def test_spread_stationarity(spread: pd.Series) -> dict:
    """
    Tests whether the spread (or premium) is stationary using ADF.
    For a valid dual-class arb, the spread must be stationary — if it's not,
    you're trading a random walk and the strategy has no theoretical basis.
    """
    result  = adfuller(spread.dropna(), autolag="AIC")
    p_value = result[1]

    return {
        "adf_statistic": round(result[0], 4),
        "p_value":       round(p_value, 4),
        "is_stationary": p_value < 0.05,
        "lags_used":     result[2],
    }


def run_dual_class_analysis(
    pair_key: str = "GOOGL_GOOG",
    start:    str = config.START_DATE,
    end:      str = config.END_DATE,
    window:   int = 30,
) -> dict:
    """
    Full analysis pipeline for a dual-class pair.
    Returns prices, spread metrics, stationarity test, z-scores, and signals.
    """
    pair   = DUAL_CLASS_PAIRS[pair_key]
    prices = fetch_pair_prices(pair_key, start, end)
    df     = compute_spread_and_premium(prices, pair_key)

    stat_test = test_spread_stationarity(df["premium_pct"])

    zscore  = rolling_zscore_premium(df["premium_pct"], window)
    signals = generate_signals(zscore)

    df["zscore"]  = zscore
    df["signal"]  = signals

    n_trades       = (signals.diff().abs() > 0).sum()
    mean_premium   = df["premium_pct"].mean()
    std_premium    = df["premium_pct"].std()

    print(f"\n{pair_key} dual-class arb analysis")
    print(f"  {pair['class_a']} / {pair['class_b']}")
    print(f"  mean voting premium:  {mean_premium:.3f}%")
    print(f"  std of premium:       {std_premium:.3f}%")
    print(f"  ADF p-value:          {stat_test['p_value']} "
          f"({'stationary' if stat_test['is_stationary'] else 'NON-STATIONARY'})")
    print(f"  signals generated:    {n_trades}")
    print(f"  note: {pair['notes']}")

    return {
        "pair":       pair_key,
        "data":       df,
        "stat_test":  stat_test,
        "n_trades":   n_trades,
    }
