"""
Cross-sectional time-series momentum (TSMOM) on sector ETFs.

Provides the momentum leg for the Markov-conditional allocation framework.
Trending regime: up to 70% of capital is allocated here.
Ranging regime:  up to 30% of capital is allocated here.

Logic:
  - Compute 12-1 month cumulative return for each ETF (12-month lookback, skip most recent month)
  - Rank ETFs by return
  - Long top N, short bottom N
  - Volatility-target each position using 20-day EWMA realized vol
  - Rebalance monthly
"""

import numpy as np
import pandas as pd
import yfinance as yf
import config


class MomentumStrategy:

    def __init__(
        self,
        start:              str   = None,
        end:                str   = None,
        etfs:               list  = None,
        lookback:           int   = None,
        skip:               int   = None,
        long_n:             int   = None,
        short_n:            int   = None,
        target_vol:         float = None,
        rebalance_interval: int   = None,
    ):
        self.start              = start  or config.START_DATE
        self.end                = end    or config.END_DATE
        self.etfs               = etfs   or config.MOMENTUM_ETFS
        self.lookback           = lookback           or config.MOMENTUM_LOOKBACK
        self.skip               = skip               or config.MOMENTUM_SKIP
        self.long_n             = long_n             or config.MOMENTUM_LONG_N
        self.short_n            = short_n            or config.MOMENTUM_SHORT_N
        self.target_vol         = target_vol         or config.MOMENTUM_TARGET_VOL
        self.rebalance_interval = rebalance_interval or config.MOMENTUM_REBALANCE_INTERVAL

    def _fetch_etf_prices(self) -> pd.DataFrame:
        raw = yf.download(self.etfs, start=self.start, end=self.end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"].ffill().dropna()
        prices = raw[["Close"]]
        prices.columns = self.etfs
        return prices.ffill().dropna()

    def generate_signals(self, prices: pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by date, columns = ETF tickers.
        Values are position weights: positive = long, negative = short.
        Scaled by volatility targeting.
        """
        if prices is None:
            prices = self._fetch_etf_prices()

        log_ret = np.log(prices / prices.shift(1))

        # 20-day EWMA realized volatility for sizing
        ewma_vol = log_ret.ewm(span=20, adjust=False).std() * np.sqrt(252)

        # 12-1 month momentum signal: cumulative return from t-lookback to t-skip
        mom_ret = prices.shift(self.skip) / prices.shift(self.lookback) - 1

        positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        rebalance_days = list(range(self.lookback, len(prices), self.rebalance_interval))

        for day_idx in rebalance_days:
            if day_idx >= len(prices):
                break
            date   = prices.index[day_idx]
            signal = mom_ret.iloc[day_idx].dropna()
            if len(signal) < self.long_n + self.short_n:
                continue

            # rank and select top / bottom N
            ranked     = signal.sort_values(ascending=False)
            long_etfs  = ranked.index[:self.long_n].tolist()
            short_etfs  = ranked.index[-self.short_n:].tolist()

            # build position weights, volatility-targeted
            for etf in long_etfs:
                vol = ewma_vol.loc[date, etf] if etf in ewma_vol.columns else 0.20
                if vol > 0:
                    size = self.target_vol / vol
                    positions.loc[date, etf] = size

            for etf in short_etfs:
                if etf in long_etfs:
                    continue
                vol = ewma_vol.loc[date, etf] if etf in ewma_vol.columns else 0.20
                if vol > 0:
                    size = self.target_vol / vol
                    positions.loc[date, etf] = -size

        # forward-fill between rebalance dates (hold positions until next rebalance)
        positions = positions.replace(0.0, np.nan).ffill().fillna(0.0)
        return positions

    def compute_returns(self, positions: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """
        Daily strategy return from momentum positions on ETF prices.
        """
        log_ret   = np.log(prices / prices.shift(1))
        # lag positions by 1 day (no lookahead)
        lagged_pos = positions.shift(1).fillna(0)
        daily_ret  = (lagged_pos * log_ret).sum(axis=1)
        return daily_ret
