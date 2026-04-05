import pandas as pd
import numpy as np
import yfinance as yf
import config


def fetch_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads adjusted closing prices for all symbols.

    Using adjusted close is non-negotiable for any serious backtest — it
    corrects for stock splits and dividends so that a 2:1 split doesn't
    look like a 50% drawdown in your price series.

    Returns a DataFrame indexed by date, columns = symbols.
    Drops any symbol where data is entirely missing.
    """
    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,    # adjusts for splits and dividends
        progress=False
    )

    # yfinance returns multi-level columns — we only need Close
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = symbols

    # drop symbols with no data at all
    prices = prices.dropna(axis=1, how="all")

    # forward-fill up to 3 days for market holidays, then drop remaining NaNs
    # more than 3 consecutive NaNs likely indicates a data problem
    prices = prices.ffill(limit=3).dropna()

    return prices


def fetch_market_data(start: str, end: str) -> pd.Series:
    """
    Downloads SPY as a broad market proxy.
    Used to compute strategy beta and as a market regime feature
    in the meta-labeling model.
    """
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    return spy["Close"].squeeze()


def fetch_vix(start: str, end: str) -> pd.Series:
    """
    Downloads the VIX index (^VIX) as a market volatility regime indicator.

    VIX is used as a feature in the meta-labeling model — trades entered
    during high-VIX regimes tend to have different mean-reversion
    characteristics than those entered in calm markets.
    """
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    return vix["Close"].squeeze()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns: r_t = ln(P_t / P_{t-1})

    Log returns are additive across time and better-behaved statistically
    than simple returns. For spread-based strategies the difference is minor,
    but it's the correct convention.
    """
    return np.log(prices / prices.shift(1)).dropna()


def align_series(series_dict: dict) -> dict:
    """
    Aligns all DataFrames/Series in a dict to their common date index.
    Ensures the price, VIX, and market data are all on the same trading days.
    """
    combined = pd.concat(series_dict.values(), axis=1, join="inner")
    combined.columns = list(series_dict.keys())
    return {k: combined[k] for k in series_dict}
