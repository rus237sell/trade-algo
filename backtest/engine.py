import pandas as pd
import numpy as np
import config


def compute_cost_per_trade(price: float) -> float:
    """
    Round-trip cost per dollar traded.

    Slippage + commission applied on each leg (entry and exit).
    Both are one-way costs, so we multiply by 2 for round-trip.

    Slippage models the market impact of the order hitting the book.
    Commission models exchange and brokerage fees.
    """
    one_way_cost = (config.SLIPPAGE_BPS + config.COMMISSION_BPS) / 10_000
    return 2 * one_way_cost


def run_backtest(
    prices:           pd.DataFrame,
    signals:          pd.DataFrame,   # {pair_id: signal_series}, values in {-1, 0, 1}
    betas:            dict,            # {pair_id: beta_series} — time-varying hedge ratios
    pair_definitions: list[tuple],     # [(y_sym, x_sym), ...]
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Vectorized backtest engine for a long/short pairs trading strategy.

    For each active signal on a pair (y, x):
      - Long signal (+1):  long Y, short beta*X
      - Short signal (-1): short Y, long beta*X

    Daily P&L per pair (dollar-neutral, $1 on each leg):
      pnl_t = signal_{t-1} * (ret_Y_t - beta_{t-1} * ret_X_t)

    Transaction costs are applied on signal changes (entries and exits).

    Returns:
      daily_returns:  strategy daily return (normalized by capital per pair)
      equity_curve:   cumulative equity starting at INITIAL_CAPITAL
      trade_pnl:      P&L series indexed at trade close dates
    """
    log_returns = np.log(prices / prices.shift(1))

    # allocate equal capital per pair
    n_pairs = len(pair_definitions)
    capital_per_pair = config.INITIAL_CAPITAL / max(n_pairs, 1)

    all_pair_returns = pd.DataFrame(index=prices.index)

    for pair_id, (y_sym, x_sym) in enumerate(pair_definitions):
        if pair_id not in signals.columns:
            continue
        if y_sym not in log_returns.columns or x_sym not in log_returns.columns:
            continue

        sig   = signals[pair_id].reindex(prices.index).fillna(0)
        beta  = betas[pair_id].reindex(prices.index).ffill().fillna(1.0)
        ret_y = log_returns[y_sym]
        ret_x = log_returns[x_sym]

        # spread return: the return of being long Y and short beta*X
        spread_ret = ret_y - beta * ret_x

        # use signal from prior day (no lookahead)
        lagged_sig = sig.shift(1).fillna(0)

        # gross daily P&L
        gross_pnl = lagged_sig * spread_ret

        # transaction cost: applied on every signal change (entry or exit)
        signal_change = (sig != sig.shift(1)).astype(float)

        # cost is proportional to the mid-price — approximate with a flat BPS model
        avg_price = (prices[y_sym] + beta * prices[x_sym]) / 2
        cost_pct  = (config.SLIPPAGE_BPS + config.COMMISSION_BPS) / 10_000
        daily_cost = signal_change * cost_pct * 2  # *2 for round-trip on both legs

        net_pnl = gross_pnl - daily_cost

        all_pair_returns[pair_id] = net_pnl

    # aggregate: equal weight across active pairs
    if all_pair_returns.empty:
        empty = pd.Series(0.0, index=prices.index)
        return empty, pd.Series(config.INITIAL_CAPITAL, index=prices.index), pd.Series(dtype=float)

    daily_returns = all_pair_returns.sum(axis=1) / n_pairs
    equity_curve  = (1 + daily_returns).cumprod() * config.INITIAL_CAPITAL

    # extract trade-level P&L (one entry per closed trade, not per day)
    trade_pnl = _extract_trade_pnl(all_pair_returns, signals)

    return daily_returns, equity_curve, trade_pnl


def _extract_trade_pnl(
    pair_returns: pd.DataFrame,
    signals:      pd.DataFrame
) -> pd.Series:
    """
    Accumulates the daily P&L for each trade from entry to exit,
    returning a Series of per-trade net P&L values.

    This is what we use to compute win rate and profit factor —
    more meaningful than raw daily returns.
    """
    trade_pnls = []

    for pair_id in pair_returns.columns:
        if pair_id not in signals.columns:
            continue

        sig  = signals[pair_id].fillna(0)
        rets = pair_returns[pair_id].fillna(0)

        in_trade    = False
        current_pnl = 0.0

        for date in rets.index:
            s = sig.get(date, 0)
            r = rets.get(date, 0.0)

            if s != 0 and not in_trade:
                # trade entry
                in_trade    = True
                current_pnl = r

            elif s != 0 and in_trade:
                # still in trade
                current_pnl += r

            elif s == 0 and in_trade:
                # trade exit
                trade_pnls.append(current_pnl)
                in_trade    = False
                current_pnl = 0.0

        # close any open trade at end of sample
        if in_trade:
            trade_pnls.append(current_pnl)

    return pd.Series(trade_pnls)
