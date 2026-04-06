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
from research.pairs_finder import rolling_adf_health
from signals.regime_filter import rolling_dfa, composite_score, threshold_gate, compute_adx_from_close
from signals.markov_regime import rolling_regime_fit, regime_allocation
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

    # phase 2: rolling sector pair selection — rescan every RESCAN_INTERVAL days
    print("\n[2/5] scanning for cointegrated sector pairs (rolling)...")

    n_days = len(prices)
    rescan_points = list(range(config.FORMATION_WINDOW, n_days, config.RESCAN_INTERVAL))
    if not rescan_points:
        rescan_points = [min(config.FORMATION_WINDOW, n_days - 1)]

    # track pairs by key: "Y_X" -> pair dict (last seen definition wins)
    active_pair_registry = {}

    for rescan_end in rescan_points:
        rescan_start = max(0, rescan_end - config.FORMATION_WINDOW)
        found = scan_sectors_from_window(prices, rescan_start, rescan_end)

        # retire pairs whose ADF health has drifted
        for key in list(active_pair_registry.keys()):
            pair = active_pair_registry[key]
            y = pair["y_sym"]
            x = pair["x_sym"]
            if y not in prices.columns or x not in prices.columns:
                del active_pair_registry[key]
                continue
            spread = prices[y].iloc[:rescan_end] - pair["beta"] * prices[x].iloc[:rescan_end]
            if rolling_adf_health(spread) > config.ADF_RETIRE_THRESHOLD:
                del active_pair_registry[key]

        # add newly found pairs
        for pair in found:
            key = f"{pair['y_sym']}_{pair['x_sym']}"
            active_pair_registry[key] = pair

    accepted_pairs = list(active_pair_registry.values())

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

        # run DFA on the spread (not raw stock prices) — spread is what mean-reverts
        spread_returns = model["spread"].pct_change().dropna()
        # reconstruct a price-like series so rolling_dfa can compute log-returns
        spread_price = (1 + spread_returns).cumprod() * 100

        dfa_series = rolling_dfa(spread_price, window=config.DFA_WINDOW, step=config.DFA_STEP)
        adx_series = compute_adx_from_close(prices[y_sym])
        score      = composite_score(
            dfa_series.reindex(prices.index).ffill(),
            adx_series.reindex(prices.index).ffill(),
            model["zscore"].fillna(0),
        )

        # apply threshold gate — avoids near-zero positions that bleed transaction costs
        gated_signals = threshold_gate(model["signals"], score)
        model["signals"] = gated_signals

        pair_models[pair_id] = model
        pair_definitions.append((y_sym, x_sym))

        n_raw    = (model["signals"] != 0).sum()
        n_gated  = (gated_signals != 0).sum()
        print(f"    {y_sym}/{x_sym}: {n_raw} raw → {n_gated} gated signals")

    # phase 3.5: Markov regime — fit on equal-weighted portfolio return, scale MR signals
    print("\n[3.5/5] fitting Markov regime model...")

    if pair_models:
        portfolio_spread_returns = pd.DataFrame({
            pid: pair_models[pid]["spread"].pct_change()
            for pid in pair_models
        }).dropna()

        if len(portfolio_spread_returns) >= config.MARKOV_WINDOW:
            equal_weighted = portfolio_spread_returns.mean(axis=1)
            try:
                regime_probs = rolling_regime_fit(
                    equal_weighted,
                    estimation_days = config.MARKOV_WINDOW,
                    refit_every     = config.MARKOV_REFIT_INTERVAL,
                    prob_threshold  = config.MARKOV_SWITCH_THRESHOLD,
                )
                alloc = regime_allocation(
                    regime_probs,
                    mom_weight_trending = config.MARKOV_TRENDING_MOM_WEIGHT,
                )
                # use filtered (not smoothed) probabilities, 1-day lag already applied in regime_allocation
                mr_weight = alloc["mean_reversion_weight"].reindex(prices.index).ffill().fillna(1.0)
                print(f"    mean MR allocation weight: {mr_weight.mean():.2f}")

                # scale all pair signals by Markov mean-reversion weight
                for pid in pair_models:
                    pair_models[pid]["signals"] = pair_models[pid]["signals"] * mr_weight
            except Exception as e:
                print(f"    Markov fit failed: {e} — using full MR allocation")
        else:
            print(f"    insufficient history for Markov ({len(portfolio_spread_returns)} days < {config.MARKOV_WINDOW})")

    # phase 4: meta-labeling — pool labeled trades across all pairs then train
    print("\n[4/5] running meta-labeling (walk-forward, pooled)...")

    signals_matrix = pd.DataFrame({
        pid: pair_models[pid]["signals"]
        for pid in pair_models
    })

    betas_dict = {
        pid: pair_models[pid]["beta"]
        for pid in pair_models
    }

    # build labeled trade pools for each pair
    per_pair_labels   = {}
    per_pair_features = {}

    for pair_id, pair in enumerate(accepted_pairs):
        if pair_id not in pair_models:
            continue
        model = pair_models[pair_id]
        pid_str = f"{pair['y_sym']}_{pair['x_sym']}"

        trade_labels_df = label_trades(
            signals = model["signals"],
            spread  = model["spread"],
            zscore  = model["zscore"],
        )
        if trade_labels_df.empty:
            continue

        trade_labels_df["pair_id"] = pid_str
        per_pair_labels[pair_id] = trade_labels_df

        feat_df = build_features(
            entry_dates = trade_labels_df.index,
            zscore      = model["zscore"],
            spread      = model["spread"],
            vix         = vix,
            prices_y    = prices[pair["y_sym"]],
            prices_x    = prices[pair["x_sym"]],
            pair_id     = pid_str,
        )
        per_pair_features[pair_id] = feat_df

    # pool all labeled trades
    all_labels_list = [df for df in per_pair_labels.values() if not df.empty]
    total_labeled   = sum(len(df) for df in all_labels_list)
    print(f"    {total_labeled} total labeled trades across {len(all_labels_list)} pairs")

    if total_labeled >= config.ML_MIN_TRAIN_SAMPLES:
        print(f"    pooled training: base win rate = "
              f"{pd.concat(all_labels_list)['label'].mean():.1%}")

        for pair_id, pair in enumerate(accepted_pairs):
            if pair_id not in pair_models or pair_id not in per_pair_labels:
                continue
            model   = pair_models[pair_id]
            labels  = per_pair_labels[pair_id]["label"]
            feats   = per_pair_features.get(pair_id, pd.DataFrame())

            if feats.empty:
                continue

            ml   = MetaLabeler()
            probs = ml.walk_forward_predict(feats, labels)

            if len(ml.oos_auc) > 0:
                print(f"    {pair['y_sym']}/{pair['x_sym']}: mean OOS AUC = {np.mean(ml.oos_auc):.3f}")

            filtered = ml.apply_filter_and_size(model["signals"], probs)
            signals_matrix[pair_id] = filtered
    else:
        print(f"    below {config.ML_MIN_TRAIN_SAMPLES} minimum — using unfiltered signals")

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
