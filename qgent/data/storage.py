from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from qgent.core.constants import Freq
from qgent.core.errors import DataError
from qgent.core.utils import symbol_to_filename

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")


class ParquetStorage:
    """Manages OHLCV data persistence using Parquet files.

    Directory layout:
        {data_dir}/{asset_class}/{freq}/{symbol}.parquet

    Example:
        data/crypto/1d/btc_usdt.parquet
        data/us_stock/1d/aapl.parquet
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    def _build_path(self, symbol: str, freq: str | Freq, asset_class: str) -> Path:
        freq_str = freq.value if isinstance(freq, Freq) else freq
        filename = symbol_to_filename(symbol) + ".parquet"
        return self.data_dir / asset_class / freq_str / filename

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        freq: str | Freq,
        asset_class: str = "crypto",
        append: bool = True,
    ) -> Path:
        """Save OHLCV DataFrame to Parquet. Appends by default (deduplicates on index)."""
        path = self._build_path(symbol, freq, asset_class)
        path.parent.mkdir(parents=True, exist_ok=True)

        if append and path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        df.to_parquet(path, engine="pyarrow")
        logger.info(f"Saved {len(df)} bars to {path}")
        return path

    def load(
        self,
        symbol: str,
        freq: str | Freq,
        asset_class: str = "crypto",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from Parquet, with optional date range filtering."""
        path = self._build_path(symbol, freq, asset_class)
        if not path.exists():
            raise DataError(f"No data found at {path}")

        df = pd.read_parquet(path)
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df

    def list_symbols(self, asset_class: str = "crypto", freq: str | Freq = Freq.DAILY) -> list[str]:
        """List all saved symbols for a given asset class and frequency."""
        freq_str = freq.value if isinstance(freq, Freq) else freq
        dir_path = self.data_dir / asset_class / freq_str
        if not dir_path.exists():
            return []
        return [f.stem.replace("_", "/").upper() for f in dir_path.glob("*.parquet")]

    def exists(self, symbol: str, freq: str | Freq, asset_class: str = "crypto") -> bool:
        return self._build_path(symbol, freq, asset_class).exists()

    def delete(self, symbol: str, freq: str | Freq, asset_class: str = "crypto") -> None:
        path = self._build_path(symbol, freq, asset_class)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted {path}")
