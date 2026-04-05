"""
quant-research: Statistical Arbitrage + Meta-Labeling Research Pipeline

Entry point. Runs the full pipeline sequentially:
  1. Data acquisition
  2. Cointegration-based pair selection
  3. Spread modeling and primary signal generation
  4. Meta-labeling: feature extraction, walk-forward ML training, probability filtering
  5. Vectorized backtest with transaction cost simulation
  6. Performance metrics reporting
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import config
from data.loader import fetch_prices, fetch_market_data, fetch_vix
from research.pairs_finder import find_cointegrated_pairs
from research.spread_model import build_pair_model
from research.meta_labeler import MetaLabeler, label_trades, build_features
from backtest.engine import run_backtest
from backtest.metrics import summarize


def main():

    print("=" * 65)
    print("quant-research | statistical arbitrage + meta-labeling")
    print("=" * 65)

    # ------------------------------------------------------------------
    # phase 1: data acquisition
    # ------------------------------------------------------------------
    print("\n[1/5] fetching price data...")

    prices = fetch_prices(config.UNIVERSE, config.START_DATE, config.END_DATE)
    market = fetch_market_data(config.START_DATE, config.END_DATE)
    vix    = fetch_vix(config.START_DATE, config.END_DATE)

    # align all series to the same trading calendar
    common_idx = prices.index.intersection(market.index).intersection(vix.index)
    prices = prices.loc[common_idx]
    market = market.reindex(common_idx).ffill()
    vix    = vix.reindex(common_idx).ffill()

    print(f"    {len(prices)} trading days | {len(prices.columns)} symbols")

    # ------------------------------------------------------------------
    # phase 2: cointegration — find valid pairs on the formation period
    # ------------------------------------------------------------------
    print("\n[2/5] scanning for cointegrated pairs...")

    formation_end = config.FORMATION_PERIOD_DAYS
    accepted_pairs = find_cointegrated_pairs(prices, 0, formation_end)

    if not accepted_pairs:
        print("    no cointegrated pairs found. try expanding the universe or relaxing thresholds.")
        return

    print(f"    {len(accepted_pairs)} pairs accepted:")
    for p in accepted_pairs:
        print(f"      {p['y_sym']}/{p['x_sym']}  beta={p['beta']}  "
              f"half_life={p['half_life']}d  coint_p={p['coint_pvalue']}")

    # ------------------------------------------------------------------
    # phase 3: spread modeling — rolling OLS, z-score, primary signals
    # ------------------------------------------------------------------
    print("\n[3/5] building spread models and primary signals...")

    pair_models     = {}   # pair_id -> {beta, spread, zscore, signals}
    pair_definitions = []  # [(y_sym, x_sym), ...]

    for pair_id, pair in enumerate(accepted_pairs):
        y_sym = pair["y_sym"]
        x_sym = pair["x_sym"]

        model = build_pair_model(prices, y_sym, x_sym)
        pair_models[pair_id] = model
        pair_definitions.append((y_sym, x_sym))

        n_signals = (model["signals"] != 0).sum()
        print(f"    {y_sym}/{x_sym}: {n_signals} primary signals generated")

    # ------------------------------------------------------------------
    # phase 4: meta-labeling — ML overlay on primary signals
    # ------------------------------------------------------------------
    print("\n[4/5] running meta-labeling (walk-forward)...")

    signals_matrix = pd.DataFrame({
        pid: pair_models[pid]["signals"]
        for pid in pair_models
    })

    betas_dict = {
        pid: pair_models[pid]["beta"]
        for pid in pair_models
    }

    # scale signals by ML probability for each pair
    for pair_id, pair in enumerate(accepted_pairs):
        model = pair_models[pair_id]

        # generate binary trade labels from the primary signal history
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

        # build contextual features at each trade entry
        features = build_features(
            entry_dates = trade_labels_df.index,
            zscore      = model["zscore"],
            spread      = model["spread"],
            vix         = vix,
            prices_y    = prices[pair["y_sym"]],
            prices_x    = prices[pair["x_sym"]],
        )

        labels = trade_labels_df["label"]

        # walk-forward train/predict
        ml = MetaLabeler()
        probabilities = ml.walk_forward_predict(features, labels)

        if len(ml.oos_auc) > 0:
            mean_auc = np.mean(ml.oos_auc)
            print(f"    mean out-of-sample AUC: {mean_auc:.3f}")

        # replace raw signals with ML-filtered, probability-scaled signals
        primary = model["signals"]
        filtered = ml.apply_filter_and_size(primary, probabilities)
        signals_matrix[pair_id] = filtered

    # ------------------------------------------------------------------
    # phase 5: backtest
    # ------------------------------------------------------------------
    print("\n[5/5] running vectorized backtest...")

    market_returns = np.log(market / market.shift(1)).dropna()

    daily_returns, equity_curve, trade_pnl = run_backtest(
        prices           = prices,
        signals          = signals_matrix,
        betas            = betas_dict,
        pair_definitions = pair_definitions,
    )

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
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
