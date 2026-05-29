from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from qgent.core.constants import Freq


class BaseFetcher(ABC):
    """Abstract base class for all data fetchers.

    Subclasses must implement `fetch()` which returns a standardized OHLCV DataFrame
    with columns: ['open', 'high', 'low', 'close', 'volume'] and a DatetimeIndex.
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        freq: str | Freq = Freq.DAILY,
        start: Optional[str] = None,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a given symbol.

        Args:
            symbol: Trading pair or ticker (e.g. 'BTC/USDT', 'AAPL').
            freq: Bar frequency.
            start: Start date string (e.g. '2023-01-01').
            end: End date string. Defaults to now.

        Returns:
            DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
        """

    @abstractmethod
    def list_symbols(self) -> list[str]:
        """Return available symbols for this data source."""

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize OHLCV DataFrame."""
        required = ["open", "high", "low", "close", "volume"]
        df.columns = [c.lower().strip() for c in df.columns]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        df = df[required].copy()
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        df = df.sort_index()
        return df
