"""
Naive Benchmark: SMA-20 Mean Reversion with ATR Stop

This is the baseline strategy that the institutional pipeline is compared against.
It represents the simplest possible formulation of mean-reversion trading:

  Entry:  price deviates more than 1 standard deviation from a 20-bar SMA
  Exit:   price crosses back through the SMA (mean reversion achieved)
  Stop:   ATR-based stop that adapts to current market volatility

Why this is the right benchmark:
  - It is not a straw man. An ATR stop is a legitimate risk management tool
    and outperforms fixed-dollar or fixed-percentage stops because it scales
    with realized volatility.
  - A strategy that cannot beat this benchmark does not justify its complexity.
  - The specific failure mode of SMA + static z-score thresholds is that it
    enters mean-reversion trades in trending regimes (ADX > 25), which is
    the fastest way to lose capital. The quant system addresses this directly
    with a Hurst + ADX regime gate.

ATR Stop Implementation:
  ATR = Wilder's 14-period average of (high - low, |high - prev_close|, |low - prev_close|)
  Long stop  = entry_price - atr_multiplier * ATR
  Short stop = entry_price + atr_multiplier * ATR

  The multiplier is adaptive: widens when ATR is below its 6-month median
  (low vol environment — stops need more room), tightens when ATR is elevated
  (high vol — trend is already moving fast, cut faster).
  This is the Chandelier Exit concept from Chuck LeBeau, validated by
  extensive academic and practitioner research.
"""

import numpy as np
import pandas as pd


# default parameters
SMA_WINDOW      = 20
ZSCORE_WINDOW   = 20
ENTRY_THRESHOLD = 1.0   # standard deviations from mean to enter
ATR_PERIOD      = 14
ATR_BASE_MULT   = 2.0   # baseline ATR multiplier for stops
ATR_LOOK_MEDIAN = 126   # 6-month window for adaptive multiplier
SLIPPAGE_BPS    = 5
COMMISSION_BPS  = 2


def compute_atr(prices: pd.DataFrame) -> pd.Series:
    """
    Wilder's Average True Range.

    True range is the greatest of:
      (a) current high - current low
      (b) |current high - previous close|
      (c) |current low - previous close|

    Wilder smoothing: ATR_t = (ATR_{t-1} * (n-1) + TR_t) / n
    This is equivalent to EWM with alpha = 1/n.

    If OHLC is not available (daily adjusted close only), we approximate
    the true range using the absolute daily return * price, which gives
    a reasonable estimate of intraday range.
    """
    if all(col in prices.columns for col in ["high", "low", "close"]):
        high  = prices["high"]
        low   = prices["low"]
        close = prices["close"]
        prev  = close.shift(1)
        tr    = pd.concat([
            high - low,
            (high - prev).abs(),
            (low  - prev).abs(),
        ], axis=1).max(axis=1)
    else:
        # approximation using close-to-close
        close = prices.iloc[:, 0] if isinstance(prices, pd.DataFrame) else prices
        tr    = close.diff().abs()

    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return atr


def adaptive_atr_multiplier(atr: pd.Series, base_mult: float = ATR_BASE_MULT) -> pd.Series:
    """
    Adjusts the ATR stop multiplier based on whether current volatility
    is elevated or compressed relative to its recent history.

    When ATR < 0.75 * 6-month median: widen to 3.5x (low vol — give more room)
    When ATR > 1.50 * 6-month median: tighten to 1.5x (high vol — cut fast)
    Otherwise: use base multiplier

    This prevents getting stopped out on normal noise during quiet markets
    and prevents large losses during volatility spikes.
    """
    median_atr = atr.rolling(ATR_LOOK_MEDIAN, min_periods=ATR_PERIOD).median()
    ratio      = atr / median_atr.replace(0, np.nan)

    mult = pd.Series(base_mult, index=atr.index)
    mult = mult.where(ratio >= 0.75, 3.5)   # very low vol: widen
    mult = mult.where(ratio <= 1.50, 1.5)   # very high vol: tighten

    return mult


def compute_signals(
    close:          pd.Series,
    sma_window:     int   = SMA_WINDOW,
    zscore_window:  int   = ZSCORE_WINDOW,
    entry_thresh:   float = ENTRY_THRESHOLD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Generates raw entry signals from SMA mean-reversion logic.

    Z-score = (price - SMA) / rolling_std

    Returns:
      sma:    rolling mean
      zscore: normalized deviation from mean
      raw_signal: +1 (price below band, expect reversion up),
                  -1 (price above band, expect reversion down),
                   0 (inside band or no signal)
    """
    sma    = close.rolling(sma_window).mean()
    std    = close.rolling(zscore_window).std()
    zscore = (close - sma) / std.replace(0, np.nan)

    raw_signal = pd.Series(0, index=close.index)
    raw_signal[zscore < -entry_thresh] =  1   # below band: long
    raw_signal[zscore >  entry_thresh] = -1   # above band: short

    return sma, zscore, raw_signal


def run_backtest(
    close:          pd.Series,
    atr_series:     pd.Series,
    sma_window:     int   = SMA_WINDOW,
    zscore_window:  int   = ZSCORE_WINDOW,
    entry_thresh:   float = ENTRY_THRESHOLD,
    initial_capital: float = 1_000_000,
) -> tuple[pd.Series, pd.Series, list]:
    """
    State-machine backtest for the SMA mean-reversion strategy.

    State machine is necessary (vs vectorized) here because stops require
    tracking entry price and adjusting each bar — you cannot do this with
    pure pandas broadcasting.

    States: flat -> in_trade -> flat (via mean-cross exit or ATR stop)

    Returns:
      daily_returns: percent return per day
      equity_curve:  portfolio value over time
      trades:        list of (entry_date, exit_date, pnl_pct, exit_reason)
    """
    sma, zscore, _ = compute_signals(close, sma_window, zscore_window, entry_thresh)
    mult           = adaptive_atr_multiplier(atr_series)

    one_way_cost = (SLIPPAGE_BPS + COMMISSION_BPS) / 10_000

    equity        = initial_capital
    position      = 0          # +1 long, -1 short, 0 flat
    entry_price   = 0.0
    entry_date    = None
    stop_price    = 0.0

    daily_pnl     = []
    equity_curve  = []
    trades        = []

    dates = close.index

    for i, date in enumerate(dates):
        price  = close.iloc[i]
        z      = zscore.iloc[i]
        mean   = sma.iloc[i]
        atr    = atr_series.iloc[i]
        m      = mult.iloc[i]

        if np.isnan(z) or np.isnan(mean) or np.isnan(atr):
            daily_pnl.append(0.0)
            equity_curve.append(equity)
            continue

        day_ret = 0.0

        if position == 0:
            # check for entry
            if z < -entry_thresh:
                position    =  1
                entry_price = price * (1 + one_way_cost)   # pay spread on entry
                entry_date  = date
                stop_price  = entry_price - m * atr

            elif z > entry_thresh:
                position    = -1
                entry_price = price * (1 - one_way_cost)
                entry_date  = date
                stop_price  = entry_price + m * atr

        elif position == 1:
            # long position: check for mean-cross exit or stop hit
            if price <= stop_price:
                exit_price = stop_price * (1 - one_way_cost)
                day_ret    = (exit_price - entry_price) / entry_price
                trades.append((entry_date, date, day_ret, "atr_stop"))
                equity    *= (1 + day_ret)
                position   = 0

            elif price >= mean and i > 0 and close.iloc[i - 1] < sma.iloc[i - 1]:
                # price crossed up through the mean — reversion complete
                exit_price = price * (1 - one_way_cost)
                day_ret    = (exit_price - entry_price) / entry_price
                trades.append((entry_date, date, day_ret, "mean_cross"))
                equity    *= (1 + day_ret)
                position   = 0

            else:
                # mark to market
                day_ret = (price - close.iloc[i - 1]) / close.iloc[i - 1]
                # update trailing stop: raise it as the trade moves in our favor
                stop_price = max(stop_price, price - m * atr)

        elif position == -1:
            # short position: check for mean-cross exit or stop hit
            if price >= stop_price:
                exit_price = stop_price * (1 + one_way_cost)
                day_ret    = (entry_price - exit_price) / entry_price
                trades.append((entry_date, date, day_ret, "atr_stop"))
                equity    *= (1 + day_ret)
                position   = 0

            elif price <= mean and i > 0 and close.iloc[i - 1] > sma.iloc[i - 1]:
                exit_price = price * (1 + one_way_cost)
                day_ret    = (entry_price - exit_price) / entry_price
                trades.append((entry_date, date, day_ret, "mean_cross"))
                equity    *= (1 + day_ret)
                position   = 0

            else:
                day_ret    = -(price - close.iloc[i - 1]) / close.iloc[i - 1]
                stop_price = min(stop_price, price + m * atr)

        daily_pnl.append(day_ret)
        equity_curve.append(equity)

    # close any open position at end of sample
    if position != 0:
        i          = len(close) - 1
        price      = close.iloc[i]
        date       = dates[i]
        exit_price = price * (1 - one_way_cost) if position == 1 else price * (1 + one_way_cost)
        day_ret    = ((exit_price - entry_price) / entry_price) * position
        trades.append((entry_date, date, day_ret, "end_of_sample"))

    returns_series = pd.Series(daily_pnl, index=dates)
    equity_series  = pd.Series(equity_curve, index=dates)

    return returns_series, equity_series, trades


def summarize_trades(trades: list) -> dict:
    """Computes trade-level statistics from the trade log."""
    if not trades:
        return {}

    pnls      = [t[2] for t in trades]
    winners   = [p for p in pnls if p > 0]
    losers    = [p for p in pnls if p < 0]
    atr_stops = sum(1 for t in trades if t[3] == "atr_stop")

    return {
        "total_trades":    len(trades),
        "win_rate":        len(winners) / len(pnls) if pnls else 0,
        "avg_win":         np.mean(winners) if winners else 0,
        "avg_loss":        np.mean(losers)  if losers  else 0,
        "profit_factor":   sum(winners) / abs(sum(losers)) if losers else np.inf,
        "atr_stop_rate":   atr_stops / len(trades),
        "avg_trade_pnl":   np.mean(pnls),
    }
