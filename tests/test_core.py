"""Tests for core module."""

import pytest
from datetime import datetime, timezone

from qgent.core.types import Bar, Order, Trade, Position, Portfolio, OrderSide, OrderType, OrderStatus
from qgent.core.constants import Freq, AssetClass
from qgent.core.utils import to_datetime, symbol_to_filename, resample_ohlcv

import pandas as pd
import numpy as np


class TestBar:
    def test_bar_properties(self):
        bar = Bar("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), 100, 110, 90, 105, 1000)
        assert bar.mid == 100.0
        assert bar.spread == 20
        assert bar.body == 5
        assert bar.is_bullish is True

    def test_bar_bearish(self):
        bar = Bar("BTC/USDT", datetime(2024, 1, 1, tzinfo=timezone.utc), 105, 110, 90, 100, 1000)
        assert bar.is_bullish is False


class TestOrder:
    def test_order_defaults(self):
        order = Order(id="test", symbol="BTC/USDT", side=OrderSide.BUY,
                      order_type=OrderType.MARKET, quantity=1.0)
        assert order.is_buy is True
        assert order.filled is False
        assert order.status == OrderStatus.PENDING


class TestTrade:
    def test_trade_pnl(self):
        trade = Trade(
            symbol="BTC/USDT", side=OrderSide.BUY,
            entry_price=100, exit_price=110, quantity=1.0,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            commission=1.0,
        )
        assert trade.pnl == 9.0
        assert trade.return_pct == pytest.approx(0.1)


class TestPortfolio:
    def test_portfolio_value(self):
        p = Portfolio(timestamp=datetime.now(timezone.utc), cash=50_000)
        pos = Position("BTC/USDT", OrderSide.BUY, 1.0, 40_000, datetime.now(timezone.utc))
        p.positions["BTC/USDT"] = pos
        assert p.market_value == 40_000
        assert p.total_value == 90_000


class TestUtils:
    def test_to_datetime_string(self):
        dt = to_datetime("2024-01-01")
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_to_datetime_timestamp(self):
        dt = to_datetime(1704067200)
        assert dt.year == 2024

    def test_to_datetime_millis(self):
        dt = to_datetime(1704067200000)
        assert dt.year == 2024

    def test_symbol_to_filename(self):
        assert symbol_to_filename("BTC/USDT") == "btc_usdt"
        assert symbol_to_filename("AAPL") == "aapl"

    def test_resample_ohlcv(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "open": range(10), "high": range(10, 20),
            "low": range(10), "close": range(10),
            "volume": [100] * 10,
        }, index=dates)
        resampled = resample_ohlcv(df, "W")
        assert len(resampled) < len(df)
