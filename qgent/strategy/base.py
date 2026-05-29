from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from qgent.strategy.signal import Signal, SignalType


class Strategy(ABC):
    """Base class for all trading strategies.

    Supports both vectorized and event-driven modes:
    - Vectorized: implement `generate_signals()` to return a signal Series for the whole dataset.
    - Event-driven: implement `on_bar()` to process each bar one at a time.
    """

    name: str = ""

    def __init__(self, **params):
        self.params = params
        if not self.name:
            self.name = self.__class__.__name__
        self._signals: list[Signal] = []

    # --- Vectorized interface ---

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate a signal Series for the entire DataFrame.

        Returns:
            Series with values from {-1, 0, 1} representing short/flat/long,
            or continuous weights in [-1, 1].
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_signals() or on_bar()"
        )

    # --- Event-driven interface ---

    def on_bar(self, idx: int, row: pd.Series, history: pd.DataFrame) -> Optional[Signal]:
        """Process a single bar in event-driven mode.

        Args:
            idx: Current bar index (integer position).
            row: Current bar data (open, high, low, close, volume, + any factors).
            history: All data up to and including current bar.

        Returns:
            A Signal object, or None for no action.
        """
        return None

    def on_start(self, df: pd.DataFrame) -> None:
        """Called once before backtesting starts. Use for precomputation."""

    def on_end(self, df: pd.DataFrame) -> None:
        """Called once after backtesting ends. Use for cleanup."""

    def emit_signal(self, signal_type: SignalType, weight: float = 1.0, **meta) -> Signal:
        signal = Signal(signal_type=signal_type, weight=weight, metadata=meta)
        self._signals.append(signal)
        return signal

    def buy(self, weight: float = 1.0, **meta) -> Signal:
        return self.emit_signal(SignalType.BUY, weight, **meta)

    def sell(self, weight: float = 1.0, **meta) -> Signal:
        return self.emit_signal(SignalType.SELL, weight, **meta)

    def flat(self) -> Signal:
        return self.emit_signal(SignalType.FLAT, 0.0)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
