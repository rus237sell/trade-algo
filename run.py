"""
quant-research: Statistical Arbitrage + Meta-Labeling Research Pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import config
from data.loader import fetch_prices, fetch_market_data, fetch_vix
from strategies.sector_pairs import scan_sectors_from_window, get_all_sector_symbols, SECTOR_UNIVERSE
from research.spread_model import build_pair_model
from research.meta_labeler import MetaLabeler, label_trades, build_features
from backtest.engine import run_backtest
from backtest.metrics import summarize


def main():

    print("=" * 65)
    print("quant-research | statistical arbitrage + meta-labeling")
    print("=" * 65)

    # phase 1: data acquisition — download all sector symbols at once
    print("\n[1/5] fetching price data...")

    all_symbols = get_all_sector_symbols()
    # include sector ETFs for residualization
    etf_symbols = [info["etf"] for info in SECTOR_UNIVERSE.values()]
    all_symbols_with_etfs = list(dict.fromkeys(all_symbols + etf_symbols))

    prices = fetch_prices(all_symbols_with_etfs, config.START_DATE, config.END_DATE)
    market = fetch_market_data(config.START_DATE, config.END_DATE)
    vix    = fetch_vix(config.START_DATE, config.END_DATE)

    common_idx = prices.index.intersection(market.index).intersection(vix.index)
    prices = prices.loc[common_idx]
    market = market.reindex(common_idx).ffill()
    vix    = vix.reindex(common_idx).ffill()

    print(f"    {len(prices)} trading days | {len(prices.columns)} symbols")

    # phase 2: sector-constrained pair selection on the formation window
    print("\n[2/5] scanning for cointegrated sector pairs...")

    formation_end = min(config.FORMATION_WINDOW, len(prices))
    accepted_pairs = scan_sectors_from_window(prices, 0, formation_end)

    if not accepted_pairs:
        print("    no cointegrated sector pairs found.")
        return

    print(f"    {len(accepted_pairs)} pairs accepted:")
    for p in accepted_pairs:
        print(f"      {p['y_sym']}/{p['x_sym']}  beta={p['beta']}  "
              f"half_life={p['half_life']}d  coint_p={p['coint_pvalue']}")

    # phase 3: spread modeling — rolling OLS, z-score, primary signals
    print("\n[3/5] building spread models and primary signals...")

    pair_models      = {}
    pair_definitions = []

    for pair_id, pair in enumerate(accepted_pairs):
        y_sym = pair["y_sym"]
        x_sym = pair["x_sym"]

        if y_sym not in prices.columns or x_sym not in prices.columns:
            continue

        model = build_pair_model(prices, y_sym, x_sym)
        pair_models[pair_id] = model
        pair_definitions.append((y_sym, x_sym))

        n_signals = (model["signals"] != 0).sum()
        print(f"    {y_sym}/{x_sym}: {n_signals} primary signals generated")

    # phase 4: meta-labeling — ML overlay on primary signals
    print("\n[4/5] running meta-labeling (walk-forward)...")

    signals_matrix = pd.DataFrame({
        pid: pair_models[pid]["signals"]
        for pid in pair_models
    })

    betas_dict = {
        pid: pair_models[pid]["beta"]
        for pid in pair_models
    }

    for pair_id, pair in enumerate(accepted_pairs):
        if pair_id not in pair_models:
            continue
        model = pair_models[pair_id]

        trade_labels_df = label_trades(
            signals = model["signals"],
            spread  = model["spread"],
            zscore  = model["zscore"],
        )

        if trade_labels_df.empty or len(trade_labels_df) < config.ML_MIN_TRAIN_SAMPLES:
            print(f"    {pair['y_sym']}/{pair['x_sym']}: "
                  f"insufficient labeled trades ({len(trade_labels_df)}) — "
                  f"using unfiltered primary signals")
            continue

        print(f"\n    {pair['y_sym']}/{pair['x_sym']}: "
              f"{len(trade_labels_df)} labeled trades | "
              f"base win rate = {trade_labels_df['label'].mean():.1%}")

        features = build_features(
            entry_dates = trade_labels_df.index,
            zscore      = model["zscore"],
            spread      = model["spread"],
            vix         = vix,
            prices_y    = prices[pair["y_sym"]],
            prices_x    = prices[pair["x_sym"]],
        )

        labels = trade_labels_df["label"]
        ml = MetaLabeler()
        probabilities = ml.walk_forward_predict(features, labels)

        if len(ml.oos_auc) > 0:
            print(f"    mean OOS AUC: {np.mean(ml.oos_auc):.3f}")

        primary  = model["signals"]
        filtered = ml.apply_filter_and_size(primary, probabilities)
        signals_matrix[pair_id] = filtered

    # phase 5: backtest
    print("\n[5/5] running vectorized backtest...")

    market_returns = np.log(market / market.shift(1)).dropna()

    daily_returns, equity_curve, trade_pnl = run_backtest(
        prices           = prices,
        signals          = signals_matrix,
        betas            = betas_dict,
        pair_definitions = pair_definitions,
    )

    perf = summarize(
        returns        = daily_returns.dropna(),
        equity_curve   = equity_curve.dropna(),
        trade_pnl      = trade_pnl,
        market_returns = market_returns,
    )

    print("\n" + "=" * 65)
    print("performance summary")
    print("=" * 65)
    print(f"  annualized return:   {perf['annualized_return']:>8.2%}")
    print(f"  annualized sharpe:   {perf['annualized_sharpe']:>8.3f}")
    print(f"  annualized sortino:  {perf['annualized_sortino']:>8.3f}")
    print(f"  max drawdown:        {perf['max_drawdown']:>8.2%}")
    print(f"  calmar ratio:        {perf['calmar_ratio']:>8.3f}")
    print(f"  win rate:            {perf['win_rate']:>8.1%}")
    print(f"  profit factor:       {perf['profit_factor']:>8.3f}")
    print(f"  beta vs market:      {perf['beta_vs_market']:>8.4f}")
    print(f"  total trades:        {perf['total_trades']:>8}")
    print("=" * 65)

    return perf, equity_curve, signals_matrix


if __name__ == "__main__":
    main()
