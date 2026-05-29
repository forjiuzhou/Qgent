from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    FLAT = "flat"


@dataclass
class Signal:
    """Represents a trading signal from a strategy."""
    signal_type: SignalType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.signal_type == SignalType.BUY

    @property
    def is_sell(self) -> bool:
        return self.signal_type == SignalType.SELL

    @property
    def direction(self) -> int:
        if self.signal_type == SignalType.BUY:
            return 1
        elif self.signal_type == SignalType.SELL:
            return -1
        return 0

    @property
    def weighted_direction(self) -> float:
        return self.direction * self.weight
