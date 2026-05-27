from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def spread(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    commission: float = 0.0

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def filled(self) -> bool:
        return self.status == OrderStatus.FILLED


@dataclass
class Trade:
    """A completed trade (entry + exit)."""
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    commission: float = 0.0

    @property
    def pnl(self) -> float:
        direction = 1 if self.side == OrderSide.BUY else -1
        gross = direction * (self.exit_price - self.entry_price) * self.quantity
        return gross - self.commission

    @property
    def return_pct(self) -> float:
        direction = 1 if self.side == OrderSide.BUY else -1
        return direction * (self.exit_price - self.entry_price) / self.entry_price

    @property
    def duration(self) -> pd.Timedelta:
        return pd.Timedelta(self.exit_time - self.entry_time)


@dataclass
class Position:
    """Tracks a current open position."""
    symbol: str
    side: OrderSide
    quantity: float
    avg_price: float
    timestamp: datetime
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float) -> None:
        direction = 1 if self.side == OrderSide.BUY else -1
        self.unrealized_pnl = direction * (current_price - self.avg_price) * self.quantity


@dataclass
class Portfolio:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        return sum(p.quantity * p.avg_price for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.market_value + sum(
            p.unrealized_pnl for p in self.positions.values()
        )
