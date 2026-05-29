from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Utilities for cleaning and preprocessing OHLCV data."""

    @staticmethod
    def fill_missing(df: pd.DataFrame, method: str = "ffill", limit: int = 5) -> pd.DataFrame:
        """Fill missing values in OHLCV data.

        Args:
            df: OHLCV DataFrame.
            method: 'ffill' or 'interpolate'.
            limit: Max consecutive gaps to fill.
        """
        original_missing = df.isna().sum().sum()
        if method == "ffill":
            df = df.ffill(limit=limit)
        elif method == "interpolate":
            df = df.interpolate(method="time", limit=limit)
        else:
            raise ValueError(f"Unknown fill method: {method}")

        remaining = df.isna().sum().sum()
        if original_missing > 0:
            logger.info(f"Filled {original_missing - remaining}/{original_missing} missing values")
        return df

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: list[str] | None = None,
        n_std: float = 5.0,
    ) -> pd.DataFrame:
        """Remove rows where values exceed n standard deviations from mean."""
        if columns is None:
            columns = ["open", "high", "low", "close"]
        columns = [c for c in columns if c in df.columns]
        mask = pd.Series(True, index=df.index)
        for col in columns:
            returns = df[col].pct_change()
            z_scores = (returns - returns.mean()) / returns.std()
            mask &= z_scores.abs() <= n_std
        removed = (~mask).sum()
        if removed > 0:
            logger.info(f"Removed {removed} outlier rows (>{n_std} std)")
        return df[mask].copy()

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV consistency: high >= low, high >= open/close, etc."""
        issues = pd.DataFrame(index=df.index)
        issues["high_lt_low"] = df["high"] < df["low"]
        issues["high_lt_open"] = df["high"] < df["open"]
        issues["high_lt_close"] = df["high"] < df["close"]
        issues["low_gt_open"] = df["low"] > df["open"]
        issues["low_gt_close"] = df["low"] > df["close"]
        issues["negative_volume"] = df["volume"] < 0

        problem_mask = issues.any(axis=1)
        n_problems = problem_mask.sum()
        if n_problems > 0:
            logger.warning(f"Found {n_problems} rows with OHLCV consistency issues")
            df = df[~problem_mask].copy()
        return df

    @staticmethod
    def add_returns(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """Add return columns to the DataFrame."""
        if periods is None:
            periods = [1]
        for p in periods:
            df[f"return_{p}"] = df["close"].pct_change(p)
        return df

    @staticmethod
    def add_log_returns(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        if periods is None:
            periods = [1]
        for p in periods:
            df[f"log_return_{p}"] = np.log(df["close"] / df["close"].shift(p))
        return df
