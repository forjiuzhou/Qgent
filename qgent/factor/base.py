from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Factor(ABC):
    """Base class for all factors.

    A factor takes OHLCV data and produces a signal Series.
    Implement `compute()` to define custom factors.
    """

    name: str = ""
    description: str = ""

    def __init__(self, **params):
        self.params = params
        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute factor values from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].

        Returns:
            Series of factor values, aligned with df's index.
        """

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        result = self.compute(df)
        result.name = self.name
        return result

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


class FunctionalFactor(Factor):
    """Wraps a plain function as a Factor object."""

    def __init__(self, func, name: str = "", **params):
        super().__init__(**params)
        self._func = func
        self.name = name or func.__name__

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return self._func(df, **self.params)
