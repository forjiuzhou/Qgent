"""Volume factor library."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qgent.factor.registry import register_factor


@register_factor("volume_ratio")
def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume relative to N-period average volume."""
    return df["volume"] / df["volume"].rolling(period).mean()


@register_factor("obv")
def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV)."""
    direction = np.sign(df["close"].diff())
    return (direction * df["volume"]).cumsum()


@register_factor("vwap_deviation")
def vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Price deviation from rolling VWAP."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()
    return (df["close"] - vwap) / vwap


@register_factor("volume_momentum")
def volume_momentum(df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """Ratio of short-term to long-term average volume."""
    return df["volume"].rolling(short).mean() / df["volume"].rolling(long).mean()
