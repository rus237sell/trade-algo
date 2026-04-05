"""
Strategy Backtest Runner

Backtests the sector-based pairs and dual-class arbitrage strategies
through the existing vectorized engine. Produces a side-by-side
performance comparison so you can evaluate each strategy independently
before running them live.

Usage:
    python run_strategies.py

Outputs:
    - Performance table printed to stdout
    - outputs/equity_curves.png (if matplotlib is available)

Note on Alpaca data vs yfinance:
    Set USE_ALPACA_DATA = True if your .env has ALPACA_API_KEY and
    ALPACA_SECRET_KEY populated. This ensures the backtest price series
    matches what the live connector will see at execution time.
    Set to False to use yfinance (no API key required).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from backtest.engine import run_backtest
from backtest.metrics import summarize
from research.spread_model import build_pair_model
from strategies.sector_pairs import scan_all_sectors, find_sector_pairs
from strategies.dual_class_arb import (
    DUAL_CLASS_PAIRS,
    fetch_pair_prices,
    compute_spread_and_premium,
    rolling_zscore_premium,
    generate_signals as dual_class_signals,
)

USE_ALPACA_DATA = bool(os.getenv("ALPACA_API_KEY"))

if USE_ALPACA_DATA:
    from data.alpaca_loader import fetch_prices, fetch_market_data, fetch_vix
    print("[data] using Alpaca historical data")
else:
    from data.loader import fetch_prices, fetch_market_data, fetch_vix
    print("[data] using yfinance (set ALPACA_API_KEY to switch to Alpaca)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _print_metrics(name: str, perf: dict) -> None:
    width = 55
    print(f"\n{'=' * width}")
    print(f"  {name}")
    print(f"{'=' * width}")
    print(f"  annualized return:   {perf['annualized_return']:>8.2%}")
    print(f"  annualized sharpe:   {perf['annualized_sharpe']:>8.3f}")
    print(f"  annualized sortino:  {perf['annualized_sortino']:>8.3f}")
    print(f"  max drawdown:        {perf['max_drawdown']:>8.2%}")
    print(f"  calmar ratio:        {perf['calmar_ratio']:>8.3f}")
    print(f"  win rate:            {perf['win_rate']:>8.1%}")
    print(f"  profit factor:       {perf['profit_factor']:>8.3f}")
    print(f"  beta vs market:      {perf['beta_vs_market']:>8.4f}")
    print(f"  total trades:        {perf['total_trades']:>8}")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# strategy 1: sector-based pairs backtest
# ---------------------------------------------------------------------------

def run_sector_pairs_backtest() -> tuple[dict, pd.Series]:
    """
    Identifies sector-constrained cointegrated pairs and backtests them
    through the existing vectorized engine.

    Formation window: first 252 trading days of config.START_DATE.
    Trading window: remainder of the sample.
    """
    print("\n[sector pairs] scanning for pairs...")

    # use a focused subset for speed — run full scan_all_sectors() in production
    target_sectors = ["Energy", "Financials", "Technology", "Consumer_Staples"]
    all_pairs = []

    for sector in target_sectors:
        try:
            pairs = find_sector_pairs(
                sector         = sector,
                start          = config.START_DATE,
                end            = config.END_DATE,
                formation_days = config.FORMATION_PERIOD_DAYS,
            )
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  error in sector {sector}: {e}")

    if not all_pairs:
        print("  no sector pairs found")
        return {}, pd.Series(dtype=float)

    print(f"  {len(all_pairs)} sector pairs accepted")

    # collect all unique symbols to fetch in one call
    symbols = list({sym for p in all_pairs for sym in [p["y_sym"], p["x_sym"]]})
    prices  = fetch_prices(symbols, config.START_DATE, config.END_DATE)
    market  = fetch_market_data(config.START_DATE, config.END_DATE)
    vix     = fetch_vix(config.START_DATE, config.END_DATE)

    common  = prices.index.intersection(market.index).intersection(vix.index)
    prices  = prices.loc[common]
    market  = market.reindex(common).ffill()

    # build spread models
    pair_models       = {}
    pair_definitions  = []

    for i, pair in enumerate(all_pairs):
        y_sym = pair["y_sym"]
        x_sym = pair["x_sym"]

        if y_sym not in prices.columns or x_sym not in prices.columns:
            continue

        try:
            model = build_pair_model(prices, y_sym, x_sym)
            pair_models[i]    = model
            pair_definitions.append((y_sym, x_sym))
        except Exception as e:
            print(f"  spread model error {y_sym}/{x_sym}: {e}")

    signals_matrix = pd.DataFrame({
        pid: pair_models[pid]["signals"] for pid in pair_models
    })
    betas_dict = {
        pid: pair_models[pid]["beta"] for pid in pair_models
    }

    print(f"  backtesting {len(pair_definitions)} pairs with spread models...")

    market_returns = np.log(market / market.shift(1)).dropna()

    daily_ret, equity, trade_pnl = run_backtest(
        prices           = prices,
        signals          = signals_matrix,
        betas            = betas_dict,
        pair_definitions = pair_definitions,
    )

    perf = summarize(
        returns        = daily_ret.dropna(),
        equity_curve   = equity.dropna(),
        trade_pnl      = trade_pnl,
        market_returns = market_returns,
    )

    return perf, equity


# ---------------------------------------------------------------------------
# strategy 2: dual-class arbitrage backtest
# ---------------------------------------------------------------------------

def run_dual_class_backtest() -> tuple[dict, pd.Series]:
    """
    Backtests all dual-class pairs (GOOGL/GOOG, BRK.A/BRK.B, NWS/NWSA)
    through the vectorized engine.

    Dual-class arb is structurally different from statistical pairs:
    - No cointegration test needed (same underlying company)
    - Z-score is around a rolling mean, not zero (voting premium is structural)
    - Tighter thresholds (entry=1.5, stop=2.5)
    - BRK-A/BRK-B requires beta=1/1500 adjustment in the engine

    The backtest engine handles it identically — what differs is how we
    construct the signals, which comes from dual_class_arb.generate_signals().
    """
    print("\n[dual class arb] building signals...")

    pair_models       = {}
    pair_definitions  = []
    prices_collection = {}

    for name, info in DUAL_CLASS_PAIRS.items():
        y_sym  = info["y"]
        x_sym  = info["x"]
        ratio  = info["ratio"]  # economic equivalence (e.g., 1500 for BRK)

        try:
            prices_df = fetch_pair_prices(y_sym, x_sym, config.START_DATE, config.END_DATE)
        except Exception as e:
            print(f"  could not fetch {name}: {e}")
            continue

        spread_df = compute_spread_and_premium(
            prices_df[y_sym], prices_df[x_sym], ratio=ratio
        )
        zscore_series = rolling_zscore_premium(spread_df["premium"])
        signals_series = dual_class_signals(zscore_series)

        # store aligned prices for the backtest engine
        pair_id = len(pair_models)
        prices_collection[y_sym] = prices_df[y_sym]
        prices_collection[x_sym] = prices_df[x_sym]

        # beta for the engine: y leg vs x leg dollar-neutrality adjustment
        # for BRK-A/B: 1 share of A = 1500 shares of B, so beta = 1/1500
        beta_series = pd.Series(1.0 / ratio, index=prices_df.index)

        pair_models[pair_id] = {
            "signals": signals_series,
            "beta":    beta_series,
            "spread":  spread_df["premium"],
        }
        pair_definitions.append((y_sym, x_sym))
        print(f"  {name}: {(signals_series != 0).sum()} signal days")

    if not pair_models:
        print("  no dual-class pairs available")
        return {}, pd.Series(dtype=float)

    prices_df_all = pd.DataFrame(prices_collection).dropna()
    market        = fetch_market_data(config.START_DATE, config.END_DATE)
    common        = prices_df_all.index.intersection(market.index)
    prices_df_all = prices_df_all.loc[common]
    market        = market.reindex(common).ffill()
    market_ret    = np.log(market / market.shift(1)).dropna()

    signals_matrix = pd.DataFrame({
        pid: pair_models[pid]["signals"].reindex(prices_df_all.index).fillna(0)
        for pid in pair_models
    })
    betas_dict = {
        pid: pair_models[pid]["beta"].reindex(prices_df_all.index).ffill()
        for pid in pair_models
    }

    daily_ret, equity, trade_pnl = run_backtest(
        prices           = prices_df_all,
        signals          = signals_matrix,
        betas            = betas_dict,
        pair_definitions = pair_definitions,
    )

    perf = summarize(
        returns        = daily_ret.dropna(),
        equity_curve   = equity.dropna(),
        trade_pnl      = trade_pnl,
        market_returns = market_ret,
    )

    return perf, equity


# ---------------------------------------------------------------------------
# output: equity curves chart
# ---------------------------------------------------------------------------

def plot_equity_curves(curves: dict[str, pd.Series]) -> None:
    os.makedirs("outputs", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, curve in curves.items():
        if curve.empty:
            continue
        norm = curve / curve.iloc[0]
        ax.plot(norm.index, norm.values, label=label, linewidth=1.5)

    ax.set_title("strategy equity curves (normalized to 1.0)")
    ax.set_ylabel("normalized equity")
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig("outputs/equity_curves.png", dpi=150)
    plt.close(fig)
    print("\n  equity curve chart saved to outputs/equity_curves.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("trade-algo | strategy backtest runner")
    print("=" * 55)

    equity_curves = {}
    results       = {}

    # sector pairs
    perf_sector, equity_sector = run_sector_pairs_backtest()
    if perf_sector:
        _print_metrics("sector-based pairs trading", perf_sector)
        equity_curves["sector pairs"] = equity_sector
        results["sector_pairs"] = perf_sector

    # dual class arb
    perf_dual, equity_dual = run_dual_class_backtest()
    if perf_dual:
        _print_metrics("dual-class arbitrage", perf_dual)
        equity_curves["dual class arb"] = equity_dual
        results["dual_class_arb"] = perf_dual

    # chart
    if equity_curves:
        plot_equity_curves(equity_curves)

    # side-by-side summary
    if len(results) > 1:
        print("\n" + "=" * 55)
        print("  side-by-side comparison")
        print("=" * 55)
        metrics = ["annualized_return", "annualized_sharpe", "max_drawdown",
                   "win_rate", "profit_factor", "beta_vs_market"]
        for m in metrics:
            vals = "   ".join(
                f"{k[:14]:<14} {v[m]:>8.3f}" for k, v in results.items()
            )
            print(f"  {m:<22}  {vals}")
        print("=" * 55)

    return results


if __name__ == "__main__":
    main()
