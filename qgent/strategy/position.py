from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PositionSizer(ABC):
    """Base class for position sizing algorithms."""

    @abstractmethod
    def size(self, signal_weight: float, price: float, portfolio_value: float, **kwargs) -> float:
        """Compute position size (in units of the asset).

        Args:
            signal_weight: Signal strength from strategy, typically in [-1, 1].
            price: Current asset price.
            portfolio_value: Current total portfolio value.

        Returns:
            Number of units to buy/sell (positive = buy, negative = sell).
        """


class FixedSizer(PositionSizer):
    """Fixed fraction of portfolio per trade."""

    def __init__(self, fraction: float = 0.1):
        self.fraction = fraction

    def size(self, signal_weight: float, price: float, portfolio_value: float, **kwargs) -> float:
        dollar_amount = portfolio_value * self.fraction * abs(signal_weight)
        units = dollar_amount / price
        return units * np.sign(signal_weight)


class KellySizer(PositionSizer):
    """Kelly Criterion-based position sizing.

    Requires historical win_rate and win_loss_ratio to compute optimal fraction.
    """

    def __init__(self, win_rate: float = 0.55, win_loss_ratio: float = 1.5, max_fraction: float = 0.25):
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.max_fraction = max_fraction

    @property
    def kelly_fraction(self) -> float:
        """Full Kelly fraction: f* = p - q/b where p=win_rate, q=1-p, b=win_loss_ratio."""
        q = 1 - self.win_rate
        f = self.win_rate - (q / self.win_loss_ratio)
        return max(0, min(f, self.max_fraction))

    def size(self, signal_weight: float, price: float, portfolio_value: float, **kwargs) -> float:
        fraction = self.kelly_fraction * abs(signal_weight)
        dollar_amount = portfolio_value * fraction
        units = dollar_amount / price
        return units * np.sign(signal_weight)


class VolTargetSizer(PositionSizer):
    """Volatility-targeting position sizer.

    Adjusts position size to target a specific annualized volatility.
    """

    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback

    def size(
        self,
        signal_weight: float,
        price: float,
        portfolio_value: float,
        returns_history: pd.Series | None = None,
        **kwargs,
    ) -> float:
        if returns_history is None or len(returns_history) < self.lookback:
            return FixedSizer(0.1).size(signal_weight, price, portfolio_value)

        realized_vol = returns_history.tail(self.lookback).std() * np.sqrt(252)
        if realized_vol <= 0:
            return 0.0

        vol_scalar = self.target_vol / realized_vol
        dollar_amount = portfolio_value * vol_scalar * abs(signal_weight)
        dollar_amount = min(dollar_amount, portfolio_value)
        units = dollar_amount / price
        return units * np.sign(signal_weight)
