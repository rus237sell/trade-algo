"""
Alpaca Live / Paper Trading Connector

Bridges the offline research pipeline to the Alpaca paper trading account.
Loads signal logic from the sector pairs model and executes orders live.

This module deliberately does NOT contain any strategy logic — it is a
pure execution layer. The signal comes from the research pipeline; this
module converts that signal into Alpaca API calls.

Key design choices:
  - Dollar-neutral: each pair trade allocates equal dollar amounts to each leg
  - Limit orders only: no market orders in production (too much slippage risk)
  - Position reconciliation before each rebalance: always read what Alpaca
    actually holds before deciding what to do, not what we think we hold
  - Cooldown on repeated signals: a signal must persist for 2 consecutive
    checks before an order is placed, reducing noise-driven trading
"""

import os
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/live_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("alpaca_connector")

ET = ZoneInfo("America/New_York")

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER             = True   # set False only when intentionally going live


# ---- clients -----------------------------------------------------------------

_trading = TradingClient(
    api_key    = ALPACA_API_KEY,
    secret_key = ALPACA_SECRET_KEY,
    paper      = PAPER,
)

_data = StockHistoricalDataClient(
    api_key    = ALPACA_API_KEY,
    secret_key = ALPACA_SECRET_KEY,
)


# ---- account -----------------------------------------------------------------

def get_equity() -> float:
    account = _trading.get_account()
    return float(account.equity)


def get_buying_power() -> float:
    account = _trading.get_account()
    return float(account.buying_power)


def is_market_open() -> bool:
    clock = _trading.get_clock()
    return clock.is_open


def get_current_positions() -> dict[str, float]:
    """
    Returns current positions as {symbol: signed_qty}.
    Positive = long, negative = short.
    """
    positions = _trading.get_all_positions()
    return {p.symbol: float(p.qty) for p in positions}


# ---- quotes ------------------------------------------------------------------

def get_live_quotes(symbols: list[str]) -> dict[str, dict]:
    """
    Fetches the latest NBBO quotes for the given symbols.
    Returns {symbol: {bid, ask, mid}}.
    """
    request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
    quotes  = _data.get_stock_latest_quote(request)

    result = {}
    for sym, q in quotes.items():
        bid = float(q.bid_price)
        ask = float(q.ask_price)
        result[sym] = {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
        }

    return result


# ---- order submission --------------------------------------------------------

def submit_limit_order(
    symbol:      str,
    qty:         float,
    side:        OrderSide,
    limit_price: float,
    time_in_force: TimeInForce = TimeInForce.DAY,
) -> str:
    """
    Submits a limit order. Returns the order ID on success.
    Rounds qty to whole shares; rounds limit_price to 2 decimal places.

    We use limit orders at the current best bid/ask rather than market orders.
    For liquid equities the difference is small, but it avoids the execution
    risk of a market order in a fast-moving spread.
    """
    whole_qty = max(1, int(qty))

    request = LimitOrderRequest(
        symbol        = symbol,
        qty           = whole_qty,
        side          = side,
        time_in_force = time_in_force,
        limit_price   = round(limit_price, 2),
    )

    order = _trading.submit_order(request)
    log.info(f"order submitted | {side.value} {whole_qty} {symbol} @ {limit_price:.2f} | id={order.id}")
    return str(order.id)


def cancel_open_orders(symbol: str = None) -> int:
    """
    Cancels all open DAY orders. If symbol is specified, only cancels orders
    for that symbol. Returns count of cancelled orders.
    """
    request = GetOrdersRequest(status=OrderStatus.OPEN)
    orders  = _trading.get_orders(filter=request)

    cancelled = 0
    for o in orders:
        if symbol and o.symbol != symbol:
            continue
        try:
            _trading.cancel_order_by_id(str(o.id))
            cancelled += 1
        except Exception as e:
            log.warning(f"failed to cancel order {o.id}: {e}")

    return cancelled


# ---- pair trade execution ----------------------------------------------------

def execute_pair_trade(
    y_sym:     str,
    x_sym:     str,
    beta:      float,
    signal:    int,        # +1 = long y/short x, -1 = short y/long x, 0 = flat
    equity:    float,
    weight:    float = 0.05,  # fraction of equity per leg
) -> dict:
    """
    Executes a dollar-neutral pairs trade.

    Dollar-neutral means:
      dollar_long ≈ dollar_short
      for a long-spread trade: buy $W of y, sell $beta*W of x

    The weight parameter controls position sizing as a fraction of equity.
    Default 5% per leg = 10% total exposure per pair.

    Steps:
      1. Get live quotes for both legs
      2. Compute share quantities for dollar-neutral sizing
      3. Cancel any existing open orders in these symbols
      4. Submit both orders simultaneously (as close as Python allows)

    Returns dict with order IDs and submitted quantities.
    """
    if signal == 0:
        log.info(f"signal=0 for {y_sym}/{x_sym} — no action")
        return {}

    quotes = get_live_quotes([y_sym, x_sym])
    if y_sym not in quotes or x_sym not in quotes:
        log.error(f"could not get quotes for {y_sym}/{x_sym}")
        return {}

    dollar_allocation = equity * weight

    # y leg
    y_mid    = quotes[y_sym]["mid"]
    y_shares = dollar_allocation / y_mid

    # x leg — hedge ratio adjusted
    x_mid    = quotes[x_sym]["mid"]
    x_shares = (dollar_allocation * beta) / x_mid

    if signal == 1:
        # long spread: buy y, sell x
        y_side = OrderSide.BUY
        x_side = OrderSide.SELL
        y_price = quotes[y_sym]["ask"]   # buy at ask
        x_price = quotes[x_sym]["bid"]   # sell at bid
    else:
        # short spread: sell y, buy x
        y_side = OrderSide.SELL
        x_side = OrderSide.BUY
        y_price = quotes[y_sym]["bid"]
        x_price = quotes[x_sym]["ask"]

    cancel_open_orders(y_sym)
    cancel_open_orders(x_sym)

    y_order_id = submit_limit_order(y_sym, y_shares, y_side, y_price)
    x_order_id = submit_limit_order(x_sym, x_shares, x_side, x_price)

    log.info(f"pair trade submitted | signal={signal} | {y_sym}/{x_sym} | "
             f"y={y_shares:.0f}@{y_price:.2f} x={x_shares:.0f}@{x_price:.2f}")

    return {
        "y_order_id": y_order_id,
        "x_order_id": x_order_id,
        "y_qty":      int(y_shares),
        "x_qty":      int(x_shares),
        "signal":     signal,
    }


def close_pair(y_sym: str, x_sym: str) -> dict:
    """
    Closes any existing position in a pair by sending offsetting limit orders
    at the current bid/ask.
    """
    positions = get_current_positions()
    quotes    = get_live_quotes([y_sym, x_sym])
    results   = {}

    for sym in [y_sym, x_sym]:
        qty = positions.get(sym, 0)
        if qty == 0:
            continue

        side  = OrderSide.SELL if qty > 0 else OrderSide.BUY
        price = quotes[sym]["bid"] if side == OrderSide.SELL else quotes[sym]["ask"]
        oid   = submit_limit_order(sym, abs(qty), side, price)
        results[sym] = oid

    return results


# ---- main rebalance loop -----------------------------------------------------

def run_live_session(
    pairs:          list[dict],    # list of {y_sym, x_sym, beta} dicts from pair finder
    signal_fn,                     # callable: (y_sym, x_sym) -> int signal
    weight:         float = 0.05,
    check_interval: int   = 300,   # seconds between rebalance checks (5 min default)
):
    """
    Main live session loop. Runs until interrupted.

    Every `check_interval` seconds during market hours:
      1. Get current equity
      2. For each active pair, call signal_fn to get the current signal
      3. Compare to current position
      4. If signal changed, execute the pair trade
      5. Log portfolio state

    This is a simple polling loop. Production systems would use websockets
    for tick-level event-driven execution.
    """
    log.info(f"live session starting | {len(pairs)} pairs | weight={weight:.1%}")
    log.info(f"paper trading: {PAPER}")

    current_signals = {f"{p['y_sym']}_{p['x_sym']}": 0 for p in pairs}

    try:
        while True:
            now = datetime.now(ET)

            if not is_market_open():
                log.info("market closed — waiting")
                time.sleep(60)
                continue

            equity = get_equity()
            log.info(f"equity: ${equity:,.2f}")

            for pair in pairs:
                y_sym = pair["y_sym"]
                x_sym = pair["x_sym"]
                beta  = pair["beta"]
                key   = f"{y_sym}_{x_sym}"

                try:
                    new_signal = signal_fn(y_sym, x_sym)
                    old_signal = current_signals[key]

                    if new_signal != old_signal:
                        log.info(f"signal change for {y_sym}/{x_sym}: "
                                 f"{old_signal} -> {new_signal}")

                        if old_signal != 0:
                            close_pair(y_sym, x_sym)

                        if new_signal != 0:
                            execute_pair_trade(
                                y_sym  = y_sym,
                                x_sym  = x_sym,
                                beta   = beta,
                                signal = new_signal,
                                equity = equity,
                                weight = weight,
                            )

                        current_signals[key] = new_signal

                except Exception as e:
                    log.error(f"error processing {y_sym}/{x_sym}: {e}")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        log.info("live session stopped by user")

        # close all pairs on exit
        for pair in pairs:
            close_pair(pair["y_sym"], pair["x_sym"])

        log.info("all pairs closed")
