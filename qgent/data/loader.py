from __future__ import annotations

from typing import Optional

import pandas as pd

from qgent.core.constants import AssetClass, Freq
from qgent.data.storage import ParquetStorage


class DataLoader:
    """High-level data loading interface. Combines fetcher + storage."""

    def __init__(self, storage: Optional[ParquetStorage] = None):
        self.storage = storage or ParquetStorage()

    def load(
        self,
        symbol: str,
        freq: str | Freq = Freq.DAILY,
        asset_class: str | AssetClass = AssetClass.CRYPTO,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data from local storage."""
        ac = asset_class.value if isinstance(asset_class, AssetClass) else asset_class
        return self.storage.load(symbol, freq, ac, start=start, end=end)

    def load_multiple(
        self,
        symbols: list[str],
        freq: str | Freq = Freq.DAILY,
        asset_class: str | AssetClass = AssetClass.CRYPTO,
        start: Optional[str] = None,
        end: Optional[str] = None,
        field: str = "close",
    ) -> pd.DataFrame:
        """Load a single field for multiple symbols into a wide DataFrame.

        Returns a DataFrame with DatetimeIndex and one column per symbol.
        """
        ac = asset_class.value if isinstance(asset_class, AssetClass) else asset_class
        frames = {}
        for sym in symbols:
            df = self.storage.load(sym, freq, ac, start=start, end=end)
            frames[sym] = df[field]
        return pd.DataFrame(frames)

    def list_available(
        self,
        asset_class: str | AssetClass = AssetClass.CRYPTO,
        freq: str | Freq = Freq.DAILY,
    ) -> list[str]:
        ac = asset_class.value if isinstance(asset_class, AssetClass) else asset_class
        return self.storage.list_symbols(ac, freq)
