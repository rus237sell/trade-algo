"""
Live Trading Entry Point — Alpaca Paper Account

Connects the sector-based pairs trading research pipeline to the
Alpaca paper trading account. This is the file you actually run
when you want the system to trade.

How it works:
  1. On startup, scan for cointegrated pairs using the formation window
  2. Build spread models (rolling OLS beta + z-score) for each accepted pair
  3. Launch the live session loop — every check_interval seconds:
       a. Compute the latest z-score for each pair from recent price data
       b. Translate z-score to a signal (-1, 0, +1)
       c. If the signal changed, close the old position and open the new one
       d. All orders are limit orders at the current NBBO

Usage:
    python run_live.py

Environment:
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env
    PAPER = True is hardcoded in alpaca_connector.py — remove that guard
    when you want to go live with real capital.

Architecture note:
    The signal_fn below is a closure that captures the latest price data
    and recomputes the z-score in real time. It is stateless — every call
    fetches fresh data from Alpaca and runs the same OLS + z-score logic
    the backtest uses. This ensures there is no drift between backtest
    signals and live signals.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

import config
from data.alpaca_loader import fetch_prices, fetch_market_data
from strategies.sector_pairs import find_sector_pairs
from research.spread_model import build_pair_model, rolling_ols_beta, compute_spread, rolling_zscore
from live.alpaca_connector import run_live_session, get_equity

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler(f"logs/live_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("run_live")

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

# how many days of recent prices to compute the live z-score
# must be longer than ZSCORE_WINDOW + ROLLING_BETA_WINDOW to have enough data
LOOKBACK_DAYS    = 180

# fraction of equity allocated per pair leg (5% = 10% total exposure per pair)
WEIGHT_PER_PAIR  = 0.05

# seconds between signal checks during market hours
CHECK_INTERVAL   = 300  # 5 minutes

# sectors to scan for pairs — reduce for faster startup
TARGET_SECTORS   = ["Energy", "Financials", "Technology", "Consumer_Staples"]

# formation window for pair selection (days of historical data used to find pairs)
FORMATION_DAYS   = config.FORMATION_PERIOD_DAYS


# ---------------------------------------------------------------------------
# live signal function factory
# ---------------------------------------------------------------------------

def make_signal_fn(
    spread_model: dict,
    y_sym:        str,
    x_sym:        str,
) -> callable:
    """
    Returns a signal_fn(y_sym, x_sym) -> int that:
      1. Fetches the last LOOKBACK_DAYS of daily prices from Alpaca
      2. Recomputes the rolling OLS beta and z-score
      3. Translates the z-score into a signal using the same thresholds as backtest

    The closure captures the pair identity; the function body is stateless
    so every call gets a fresh signal without relying on stale in-memory state.

    Note: this runs once per check interval, not tick-by-tick. For pairs
    with high intraday spread volatility you may want to reduce CHECK_INTERVAL.
    """
    def signal_fn(y: str, x: str) -> int:
        end_dt   = datetime.now().strftime("%Y-%m-%d")
        start_dt = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        try:
            prices = fetch_prices([y_sym, x_sym], start_dt, end_dt)
        except Exception as e:
            log.error(f"price fetch failed for {y_sym}/{x_sym}: {e}")
            return 0

        if y_sym not in prices.columns or x_sym not in prices.columns:
            return 0

        if len(prices) < config.ZSCORE_WINDOW + 10:
            log.warning(f"insufficient price history for {y_sym}/{x_sym}: {len(prices)} days")
            return 0

        # recompute beta and z-score from latest data
        try:
            beta   = rolling_ols_beta(prices[y_sym], prices[x_sym])
            spread = compute_spread(prices[y_sym], prices[x_sym], beta)
            zscore = rolling_zscore(spread)
        except Exception as e:
            log.error(f"spread model error {y_sym}/{x_sym}: {e}")
            return 0

        z = zscore.iloc[-1]

        if pd.isna(z):
            return 0

        # signal logic — matches backtest engine exactly
        if z > config.ZSCORE_STOP or z < -config.ZSCORE_STOP:
            return 0   # stop-loss zone — flat
        elif z > config.ZSCORE_ENTRY:
            return -1  # short spread: z too high, expect reversion down
        elif z < -config.ZSCORE_ENTRY:
            return 1   # long spread: z too low, expect reversion up
        elif abs(z) < config.ZSCORE_EXIT:
            return 0   # mean has been reached — exit
        else:
            # between exit and entry: hold existing position (handled by connector)
            return 0

    return signal_fn


# ---------------------------------------------------------------------------
# startup: pair selection
# ---------------------------------------------------------------------------

def select_live_pairs() -> list[dict]:
    """
    Runs the sector pair selection pipeline on recent data.
    Returns the accepted pairs as a list of dicts ready for run_live_session().

    This is run once at startup. In production you would re-run this
    weekly or monthly and update the pair list accordingly.
    """
    log.info("scanning for live pairs...")

    all_pairs = []

    for sector in TARGET_SECTORS:
        try:
            pairs = find_sector_pairs(
                sector         = sector,
                start          = config.START_DATE,
                end            = config.END_DATE,
                formation_days = FORMATION_DAYS,
            )
            all_pairs.extend(pairs)
            log.info(f"  {sector}: {len(pairs)} pairs accepted")
        except Exception as e:
            log.error(f"  sector {sector} failed: {e}")

    if not all_pairs:
        raise RuntimeError("no pairs found — cannot start live session")

    # cap at 10 pairs to limit total capital exposure
    all_pairs = all_pairs[:10]
    log.info(f"live pair universe: {len(all_pairs)} pairs")
    for p in all_pairs:
        log.info(f"  {p['y_sym']}/{p['x_sym']}  sector={p['sector']}  "
                 f"half_life={p['half_life']}d  beta={p['beta']}")

    return all_pairs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("logs", exist_ok=True)

    log.info("=" * 55)
    log.info("trade-algo | live session startup")
    log.info("=" * 55)

    # check account connectivity first
    try:
        equity = get_equity()
        log.info(f"Alpaca paper account connected | equity: ${equity:,.2f}")
    except Exception as e:
        log.error(f"Alpaca connection failed: {e}")
        log.error("check ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        return

    # select pairs
    live_pairs = select_live_pairs()

    # build a spread model for each pair (used to initialize the signal_fn)
    # we fetch one combined dataset to avoid redundant downloads
    symbols = list({sym for p in live_pairs for sym in [p["y_sym"], p["x_sym"]]})
    prices  = fetch_prices(symbols, config.START_DATE, config.END_DATE)

    pairs_with_signals = []
    for pair in live_pairs:
        y_sym = pair["y_sym"]
        x_sym = pair["x_sym"]

        if y_sym not in prices.columns or x_sym not in prices.columns:
            log.warning(f"skipping {y_sym}/{x_sym} — price data unavailable")
            continue

        try:
            model     = build_pair_model(prices, y_sym, x_sym)
            signal_fn = make_signal_fn(model, y_sym, x_sym)

            pairs_with_signals.append({
                "y_sym":     y_sym,
                "x_sym":     x_sym,
                "beta":      float(model["beta"].iloc[-1]),
                "signal_fn": signal_fn,
            })
        except Exception as e:
            log.error(f"spread model failed for {y_sym}/{x_sym}: {e}")

    if not pairs_with_signals:
        log.error("no pairs with valid spread models — aborting")
        return

    log.info(f"{len(pairs_with_signals)} pairs ready for live trading")

    # build a unified signal_fn dispatcher for run_live_session
    # run_live_session expects: signal_fn(y_sym, x_sym) -> int
    signal_map = {
        f"{p['y_sym']}_{p['x_sym']}": p["signal_fn"]
        for p in pairs_with_signals
    }

    def dispatch_signal(y_sym: str, x_sym: str) -> int:
        key = f"{y_sym}_{x_sym}"
        fn  = signal_map.get(key)
        if fn is None:
            return 0
        return fn(y_sym, x_sym)

    # strip signal_fn from the pair dicts (run_live_session doesn't expect it)
    session_pairs = [
        {"y_sym": p["y_sym"], "x_sym": p["x_sym"], "beta": p["beta"]}
        for p in pairs_with_signals
    ]

    # start the live loop
    run_live_session(
        pairs          = session_pairs,
        signal_fn      = dispatch_signal,
        weight         = WEIGHT_PER_PAIR,
        check_interval = CHECK_INTERVAL,
    )


if __name__ == "__main__":
    main()
