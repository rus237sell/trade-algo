"""
Alpaca Historical Data Loader

Replaces the yfinance feed in the research pipeline with Alpaca's
data API. This matters for live consistency: if your execution is on
Alpaca, your backtest should use the same data source. yfinance and
Alpaca can have slightly different split/dividend adjustments, which
shifts the spread history and invalidates backtest z-scores.

Alpaca provides:
  - Adjusted daily bars (open, high, low, close, volume, vwap)
  - Up to ~6 years of history on the free data plan
  - The same price series your limit orders will fill against

This module mirrors the interface of data/loader.py so the rest of the
pipeline does not need to change — just swap the import.
"""

import os
import logging
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

log = logging.getLogger(__name__)

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# one shared client — avoids re-authenticating on every call
_client = StockHistoricalDataClient(
    api_key    = ALPACA_API_KEY,
    secret_key = ALPACA_SECRET_KEY,
)

_DAILY = TimeFrame(1, TimeFrameUnit.Day)


def _to_date(s: str | date | datetime) -> datetime:
    if isinstance(s, str):
        return datetime.strptime(s, "%Y-%m-%d")
    if isinstance(s, date) and not isinstance(s, datetime):
        return datetime(s.year, s.month, s.day)
    return s


def fetch_prices(
    symbols:   list[str],
    start:     str,
    end:       str,
    feed:      str = "iex",    # "iex" = free tier, "sip" = paid consolidated
) -> pd.DataFrame:
    """
    Fetches adjusted daily close prices for a list of symbols.

    Returns a DataFrame indexed by date, one column per symbol.
    Missing values are forward-filled up to 3 consecutive days (holidays,
    early closes). Symbols with >10% missing data after ffill are dropped.

    feed="iex" works on Alpaca free accounts (IEX exchange data).
    feed="sip" requires a paid data subscription (consolidated tape).
    """
    request = StockBarsRequest(
        symbol_or_symbols = symbols,
        timeframe         = _DAILY,
        start             = _to_date(start),
        end               = _to_date(end),
        adjustment        = "all",    # split + dividend adjusted
        feed              = feed,
    )

    bars = _client.get_stock_bars(request).df

    if bars.empty:
        raise RuntimeError(f"Alpaca returned no bar data for {symbols} {start}:{end}")

    # bars.df has a MultiIndex (symbol, timestamp) — pivot to wide format
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.normalize()

    prices = bars.pivot_table(
        index   = "timestamp",
        columns = "symbol",
        values  = "close",
    )
    prices.index.name = "date"
    prices.columns.name = None

    # forward-fill gaps (weekends already absent; this handles data outages)
    prices = prices.ffill(limit=3)

    # drop symbols with excessive missing data
    missing_frac = prices.isna().mean()
    bad_syms = missing_frac[missing_frac > 0.10].index.tolist()
    if bad_syms:
        log.warning(f"dropping symbols with >10%% missing data: {bad_syms}")
        prices = prices.drop(columns=bad_syms)

    prices = prices.dropna()

    log.info(f"fetched {len(prices)} days x {len(prices.columns)} symbols via Alpaca")
    return prices


def fetch_market_data(start: str, end: str) -> pd.Series:
    """
    Fetches SPY adjusted close as the market benchmark.
    Returns a Series indexed by date.
    """
    prices = fetch_prices(["SPY"], start, end)
    return prices["SPY"]


def fetch_vix(start: str, end: str) -> pd.Series:
    """
    VIX is not available through Alpaca (it is not a tradeable equity).
    Falls back to yfinance for VIX only — this is acceptable because VIX
    is used as a feature, not for trade execution.
    """
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        return vix["Close"].squeeze()
    except Exception as e:
        log.warning(f"could not fetch VIX from yfinance: {e} — returning zeros")
        idx = pd.date_range(start=start, end=end, freq="B")
        return pd.Series(20.0, index=idx)  # flat 20 as neutral fallback


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from close-to-close prices."""
    return np.log(prices / prices.shift(1)).dropna()


def align_series(
    prices: pd.DataFrame,
    market: pd.Series,
    vix:    pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Aligns all three series to the intersection of their date indices.
    This avoids off-by-one errors when any series has missing dates.
    """
    common = prices.index.intersection(market.index).intersection(vix.index)
    return (
        prices.loc[common],
        market.reindex(common).ffill(),
        vix.reindex(common).ffill(),
    )
