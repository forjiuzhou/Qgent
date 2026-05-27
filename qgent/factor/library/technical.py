"""Classic technical indicators as factors."""
from __future__ import annotations

import pandas as pd

from qgent.factor.registry import register_factor


@register_factor("sma")
def sma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Simple Moving Average."""
    return df[col].rolling(period).mean()


@register_factor("ema")
def ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Exponential Moving Average."""
    return df[col].ewm(span=period, min_periods=period).mean()


@register_factor("bollinger_pct")
def bollinger_pct(df: pd.DataFrame, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger %B — where price is relative to Bollinger Bands (0 = lower, 1 = upper)."""
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return (df["close"] - lower) / (upper - lower)


@register_factor("stochastic_k")
def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic %K oscillator."""
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    return (df["close"] - low_min) / (high_max - low_min) * 100


@register_factor("stochastic_d")
def stochastic_d(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
    """%D line (smoothed %K)."""
    k = stochastic_k(df, k_period)
    return k.rolling(d_period).mean()


@register_factor("williams_r")
def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R oscillator."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return (high_max - df["close"]) / (high_max - low_min) * -100


@register_factor("cci")
def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    ma = typical.rolling(period).mean()
    md = typical.rolling(period).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    return (typical - ma) / (0.015 * md)
