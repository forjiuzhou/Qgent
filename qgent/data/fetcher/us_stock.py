from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from qgent.core.constants import Freq
from qgent.data.fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

_FREQ_MAP = {
    Freq.MIN_1: "1m",
    Freq.MIN_5: "5m",
    Freq.MIN_15: "15m",
    Freq.MIN_30: "30m",
    Freq.HOUR_1: "1h",
    Freq.DAILY: "1d",
    Freq.WEEKLY: "1wk",
    Freq.MONTHLY: "1mo",
}

_YF_COL_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Adj Close": "adj_close",
}


class USStockFetcher(BaseFetcher):
    """Fetch US stock OHLCV data via yfinance."""

    def fetch(
        self,
        symbol: str,
        freq: str | Freq = Freq.DAILY,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if isinstance(freq, Freq):
            interval = _FREQ_MAP.get(freq, freq.value)
        else:
            interval = freq

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = df.rename(columns=_YF_COL_MAP)
        return self._validate_ohlcv(df)

    def list_symbols(self) -> list[str]:
        # yfinance doesn't provide a symbol listing API;
        # return popular indices and tickers as reference.
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "SPY", "QQQ", "IWM", "DIA",
        ]
