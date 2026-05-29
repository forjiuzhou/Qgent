from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Union

import pandas as pd


def to_datetime(value: Union[str, int, float, datetime, pd.Timestamp]) -> datetime:
    """Convert various time representations to a timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        if value > 1e12:
            value = value / 1000
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        dt = pd.Timestamp(value).to_pydatetime()
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"Cannot convert {type(value)} to datetime")


def symbol_to_filename(symbol: str) -> str:
    """Convert a trading symbol to a safe filename. e.g. 'BTC/USDT' -> 'btc_usdt'."""
    return symbol.lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def generate_order_id(symbol: str, timestamp: datetime, side: str) -> str:
    raw = f"{symbol}_{timestamp.isoformat()}_{side}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV DataFrame to a lower frequency."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample(freq).agg(cols)
    return resampled.dropna(subset=["open"])
