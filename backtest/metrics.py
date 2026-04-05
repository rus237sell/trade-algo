import numpy as np
import pandas as pd


TRADING_DAYS = 252


def annualized_sharpe(returns: pd.Series) -> float:
    """
    Sharpe = (mean daily return / std daily return) * sqrt(252)

    Uses excess return over cash; here we treat the risk-free rate as zero
    for simplicity. For a long/short strategy this is appropriate since
    the strategy is largely market-neutral and self-financing.
    """
    if returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)


def annualized_sortino(returns: pd.Series) -> float:
    """
    Sortino = (mean daily return / downside deviation) * sqrt(252)

    Downside deviation only penalizes negative returns — more appropriate
    than Sharpe for strategies with asymmetric return distributions.
    """
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    return (returns.mean() / downside.std()) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max Drawdown = max( (peak - trough) / peak ) over the full period.

    Returns a negative number representing the worst peak-to-trough decline.
    """
    rolling_peak = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_peak) / rolling_peak
    return drawdowns.min()


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """
    Calmar = annualized return / |max drawdown|

    High Calmar means strong returns relative to the worst loss period —
    the metric most useful for assessing capital preservation.
    """
    ann_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return np.nan
    return ann_return / mdd


def win_rate(trade_pnl: pd.Series) -> float:
    """
    Fraction of completed trades with positive net P&L after costs.
    """
    if len(trade_pnl) == 0:
        return np.nan
    return (trade_pnl > 0).sum() / len(trade_pnl)


def strategy_beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Beta = Cov(strategy, market) / Var(market)

    For a market-neutral pairs strategy, beta should be near zero.
    A statistically significant non-zero beta indicates unintended
    directional market exposure.
    """
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    strat = aligned.iloc[:, 0]
    mkt   = aligned.iloc[:, 1]
    return np.cov(strat, mkt)[0, 1] / np.var(mkt)


def profit_factor(trade_pnl: pd.Series) -> float:
    """
    Profit Factor = sum of winning trades / |sum of losing trades|

    A value > 1.5 is generally acceptable; > 2.0 is strong.
    """
    winners = trade_pnl[trade_pnl > 0].sum()
    losers  = abs(trade_pnl[trade_pnl < 0].sum())
    if losers == 0:
        return np.inf
    return winners / losers


def summarize(
    returns:        pd.Series,
    equity_curve:   pd.Series,
    trade_pnl:      pd.Series,
    market_returns: pd.Series
) -> dict:
    """
    Computes and returns the full performance summary as a dict.
    """
    ann_return = (1 + returns.mean()) ** TRADING_DAYS - 1

    return {
        "annualized_return":  round(ann_return, 4),
        "annualized_sharpe":  round(annualized_sharpe(returns), 3),
        "annualized_sortino": round(annualized_sortino(returns), 3),
        "max_drawdown":       round(max_drawdown(equity_curve), 4),
        "calmar_ratio":       round(calmar_ratio(returns, equity_curve), 3),
        "win_rate":           round(win_rate(trade_pnl), 4),
        "profit_factor":      round(profit_factor(trade_pnl), 3),
        "beta_vs_market":     round(strategy_beta(returns, market_returns), 4),
        "total_trades":       len(trade_pnl),
    }
