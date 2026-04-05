"""
Position Sizing: Volatility-Targeted + Half-Kelly

Two independent sizing methods that compound each other:

1. Volatility-Targeted Sizing (primary method):
   Each position contributes equal risk to the portfolio regardless of
   the underlying asset's realized volatility.

   dollar_position = (target_vol / asset_vol) * portfolio_value

   For a $1M portfolio targeting 10% annualized vol:
     - SPY  (21% vol): position = (0.10 / 0.21) * 1M = $476K
     - NVDA (45% vol): position = (0.10 / 0.45) * 1M = $222K
     - Each contributes the same expected risk contribution to the portfolio.

   Volatility is estimated with EWMA (Exponentially Weighted Moving Average)
   using a 20-day half-life, which is more responsive than rolling std.
   This is the RiskMetrics 1994 approach, still standard at banks.

2. Half-Kelly Criterion (secondary multiplier):
   The Kelly criterion gives the theoretically optimal bet size to maximize
   long-run wealth growth. The continuous form:
     f* = mu / sigma^2
   is dangerously aggressive in practice (5x+ leverage is common).

   Half-Kelly (f = 0.5 * f*) captures 75% of the theoretical growth rate
   while dramatically reducing drawdown probability. It is used as a
   multiplier on top of volatility targeting — if Kelly says bet less than
   1.0, reduce the volatility-targeted size proportionally.

   Kelly is estimated from a rolling 100-200 trade window, not the full
   backtest, to avoid look-ahead bias.

Research basis:
  - RiskMetrics Technical Document (1994) — EWMA volatility
  - Kelly 1956, Thorp 1975 — Kelly criterion and half-Kelly
  - Lopez de Prado 2018 — position sizing in "Advances in Financial ML"
"""

import numpy as np
import pandas as pd


TARGET_VOL     = 0.10   # 10% annualized portfolio volatility target
EWMA_HALFLIFE  = 20     # days — half-life for volatility estimate
TRADING_DAYS   = 252
MAX_LEVERAGE   = 1.0    # no leverage; cap position at 1x NAV per asset
MIN_POSITION   = 0.01   # minimum 1% of NAV per trade


# ---- EWMA Volatility ---------------------------------------------------------

def ewma_volatility(
    returns:  pd.Series,
    halflife: int = EWMA_HALFLIFE,
) -> pd.Series:
    """
    Exponentially weighted moving average volatility (annualized).

    EWMA gives more weight to recent observations, making it faster to
    react to volatility regime changes than rolling standard deviation.

    The decay factor lambda = exp(-ln(2) / halflife).
    For halflife=20: lambda ≈ 0.966 (standard RiskMetrics).

    Returns annualized volatility as a fraction (e.g., 0.21 for 21% vol).
    """
    # daily EWMA variance
    ewma_var = returns.ewm(halflife=halflife, adjust=False).var()

    # annualize: multiply daily variance by 252 trading days
    return np.sqrt(ewma_var * TRADING_DAYS)


# ---- Volatility-Targeted Sizing ----------------------------------------------

def vol_targeted_weight(
    returns:      pd.Series,
    target_vol:   float = TARGET_VOL,
    halflife:     int   = EWMA_HALFLIFE,
    max_leverage: float = MAX_LEVERAGE,
) -> pd.Series:
    """
    Computes the portfolio weight (fraction of NAV) for a single asset
    using volatility-targeted sizing.

    weight_t = target_vol / asset_vol_t

    Weights are capped at max_leverage (default 1.0 = no leverage).
    When asset vol is below target_vol, the weight would exceed 1.0,
    which we cap — we don't take on leverage to hit the vol target.

    Returns a weight series aligned to the returns index.
    """
    asset_vol = ewma_volatility(returns, halflife)
    weight    = (target_vol / asset_vol).clip(upper=max_leverage, lower=MIN_POSITION)
    return weight.fillna(MIN_POSITION)


def vol_targeted_dollars(
    returns:       pd.Series,
    equity:        float,
    target_vol:    float = TARGET_VOL,
    halflife:      int   = EWMA_HALFLIFE,
    max_leverage:  float = MAX_LEVERAGE,
) -> float:
    """
    Computes the dollar position size for the most recent observation.
    Used in live trading to convert a signal into a dollar allocation.

    Returns the dollar amount to allocate to this asset (not the share count).
    """
    weight = vol_targeted_weight(returns, target_vol, halflife, max_leverage)
    return float(weight.iloc[-1]) * equity


# ---- Kelly Criterion ---------------------------------------------------------

def kelly_fraction(
    win_rate:      float,
    avg_win:       float,    # average win as a fraction of position
    avg_loss:      float,    # average loss as a fraction of position (positive number)
    kelly_scale:   float = 0.5,   # 0.5 = half-Kelly
) -> float:
    """
    Discrete Kelly fraction for a binary win/loss game.

    Formula: f* = (b*p - q) / b
      where: b = avg_win / avg_loss (win-loss ratio)
             p = win rate
             q = 1 - p (loss rate)

    Returns the fractional Kelly (kelly_scale * f*), capped at [0, 1].
    A negative Kelly means expected value is negative — do not trade.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0

    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p

    f_star = (b * p - q) / b

    return float(np.clip(kelly_scale * f_star, 0.0, 1.0))


def rolling_kelly_multiplier(
    trade_returns:  pd.Series,   # series of per-trade P&L as fractions
    window:         int   = 100,
    kelly_scale:    float = 0.5,
) -> pd.Series:
    """
    Computes a rolling Kelly fraction across the last `window` trades.

    This is used as a sizing multiplier: if Kelly says 0.6, apply 60%
    of the volatility-targeted position size.

    Using a rolling window (not full history) prevents look-ahead bias
    and captures the recent performance regime of the strategy.
    """
    fractions = []

    for i in range(len(trade_returns)):
        window_trades = trade_returns.iloc[max(0, i - window): i + 1]

        if len(window_trades) < 20:
            fractions.append(1.0)   # insufficient data: full vol-targeted size
            continue

        winners  = window_trades[window_trades > 0]
        losers   = window_trades[window_trades < 0]

        win_rate = len(winners) / len(window_trades)
        avg_win  = float(winners.mean()) if len(winners) > 0 else 0.0
        avg_loss = float(abs(losers.mean())) if len(losers) > 0 else 0.0

        f = kelly_fraction(win_rate, avg_win, avg_loss, kelly_scale)
        fractions.append(max(f, 0.10))   # floor at 10% — never go fully flat on Kelly alone

    return pd.Series(fractions, index=trade_returns.index)


# ---- Combined Sizing ---------------------------------------------------------

def combined_position_weight(
    returns:        pd.Series,
    trade_returns:  pd.Series = None,
    target_vol:     float = TARGET_VOL,
    halflife:       int   = EWMA_HALFLIFE,
    kelly_window:   int   = 100,
    kelly_scale:    float = 0.5,
    max_leverage:   float = MAX_LEVERAGE,
) -> pd.Series:
    """
    Applies volatility targeting and then scales by the Kelly fraction.

    final_weight = vol_targeted_weight * kelly_multiplier

    When trade_returns is None (no trade history yet), returns pure
    vol-targeted weights. The Kelly layer is added as trade history builds.
    """
    vol_weight = vol_targeted_weight(returns, target_vol, halflife, max_leverage)

    if trade_returns is None or len(trade_returns) < 20:
        return vol_weight

    kelly_mult = rolling_kelly_multiplier(trade_returns, kelly_window, kelly_scale)

    # align Kelly index to returns index — Kelly is trade-indexed, not date-indexed
    # we use the last available Kelly value as the current multiplier
    current_kelly = float(kelly_mult.iloc[-1]) if len(kelly_mult) > 0 else 1.0

    return (vol_weight * current_kelly).clip(upper=max_leverage, lower=MIN_POSITION)
