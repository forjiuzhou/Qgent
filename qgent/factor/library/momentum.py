"""Momentum factor library."""
from __future__ import annotations

import pandas as pd

from qgent.factor.registry import register_factor


@register_factor("momentum")
def returns(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Simple price momentum (percentage return over N periods)."""
    return df[col].pct_change(period)


@register_factor("rate_of_change")
def rate_of_change(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    """Rate of Change (ROC)."""
    return (df[col] - df[col].shift(period)) / df[col].shift(period) * 100


@register_factor("rsi")
def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    """Relative Strength Index."""
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


@register_factor("macd_signal")
def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.Series:
    """MACD histogram (MACD line - Signal line)."""
    ema_fast = df[col].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df[col].ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return macd_line - signal_line


@register_factor("moving_average_ratio")
def ma_ratio(df: pd.DataFrame, short: int = 10, long: int = 50, col: str = "close") -> pd.Series:
    """Ratio of short MA to long MA. > 1 means bullish trend."""
    ma_short = df[col].rolling(short).mean()
    ma_long = df[col].rolling(long).mean()
    return ma_short / ma_long
