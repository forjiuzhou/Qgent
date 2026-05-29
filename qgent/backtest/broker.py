from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from qgent.core.types import Order, OrderSide, OrderType, OrderStatus, Trade
from qgent.core.utils import generate_order_id

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_order_size: float = 0.0


class SimulatedBroker:
    """Simulates order execution with commission and slippage."""

    def __init__(self, initial_cash: float = 100_000.0, config: Optional[BrokerConfig] = None):
        self.config = config or BrokerConfig()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, float] = {}
        self.avg_prices: dict[str, float] = {}
        self.orders: list[Order] = []
        self.trades: list[Trade] = []

    def reset(self):
        self.cash = self.initial_cash
        self.positions.clear()
        self.avg_prices.clear()
        self.orders.clear()
        self.trades.clear()

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        timestamp: pd.Timestamp,
        order_type: OrderType = OrderType.MARKET,
    ) -> Order:
        order = Order(
            id=generate_order_id(symbol, timestamp, side.value),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )
        self.orders.append(order)
        if order_type == OrderType.MARKET:
            self._fill_order(order, price, timestamp)
        return order

    def _fill_order(self, order: Order, market_price: float, timestamp: pd.Timestamp) -> None:
        slippage = market_price * self.config.slippage_rate
        if order.side == OrderSide.BUY:
            fill_price = market_price + slippage
        else:
            fill_price = market_price - slippage

        cost = fill_price * order.quantity
        commission = cost * self.config.commission_rate

        if order.side == OrderSide.BUY:
            if self.cash < cost + commission:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: insufficient cash ({self.cash:.2f} < {cost + commission:.2f})")
                return

            current_qty = self.positions.get(order.symbol, 0.0)
            current_avg = self.avg_prices.get(order.symbol, 0.0)

            if current_qty >= 0:
                new_qty = current_qty + order.quantity
                self.avg_prices[order.symbol] = (
                    (current_avg * current_qty + fill_price * order.quantity) / new_qty
                    if new_qty > 0 else 0
                )
                self.positions[order.symbol] = new_qty
            else:
                close_qty = min(order.quantity, abs(current_qty))
                if close_qty > 0:
                    self.trades.append(Trade(
                        symbol=order.symbol,
                        side=OrderSide.SELL,
                        entry_price=current_avg,
                        exit_price=fill_price,
                        quantity=close_qty,
                        entry_time=timestamp,
                        exit_time=timestamp,
                        commission=commission * (close_qty / order.quantity),
                    ))
                remaining = order.quantity - close_qty
                self.positions[order.symbol] = current_qty + order.quantity
                if remaining > 0:
                    self.avg_prices[order.symbol] = fill_price
            self.cash -= cost + commission

        else:  # SELL
            current_qty = self.positions.get(order.symbol, 0.0)
            if current_qty > 0:
                close_qty = min(order.quantity, current_qty)
                if close_qty > 0:
                    entry_price = self.avg_prices.get(order.symbol, fill_price)
                    self.trades.append(Trade(
                        symbol=order.symbol,
                        side=OrderSide.BUY,
                        entry_price=entry_price,
                        exit_price=fill_price,
                        quantity=close_qty,
                        entry_time=timestamp,
                        exit_time=timestamp,
                        commission=commission * (close_qty / order.quantity),
                    ))
            self.positions[order.symbol] = current_qty - order.quantity
            if self.positions[order.symbol] < 0:
                self.avg_prices[order.symbol] = fill_price
            self.cash += cost - commission

        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_timestamp = timestamp
        order.commission = commission

    @property
    def portfolio_value(self) -> float:
        return self.cash

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Calculate total portfolio value at given prices."""
        value = self.cash
        for symbol, qty in self.positions.items():
            if symbol in prices:
                value += qty * prices[symbol]
        return value
