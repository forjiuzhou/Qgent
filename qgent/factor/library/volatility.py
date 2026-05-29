"""Volatility factor library."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qgent.factor.registry import register_factor


@register_factor("realized_vol")
def realized(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Annualized realized volatility based on log returns."""
    log_ret = np.log(df[col] / df[col].shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252)


@register_factor("atr")
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


@register_factor("atr_pct")
def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as percentage of close price."""
    return atr(df, period) / df["close"]


@register_factor("bollinger_width")
def bollinger_width(df: pd.DataFrame, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """Bollinger Band width as percentage of middle band."""
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return (upper - lower) / ma


@register_factor("garman_klass_vol")
def garman_klass(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Garman-Klass volatility estimator (uses OHLC data for better estimates)."""
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(gk.rolling(period).mean() * 252)
