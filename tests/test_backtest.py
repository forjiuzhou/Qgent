"""Tests for backtest module."""

import numpy as np
import pandas as pd
import pytest

from qgent.backtest.engine import BacktestEngine
from qgent.backtest.metrics import compute_metrics
from qgent.backtest.broker import SimulatedBroker, BrokerConfig
from qgent.core.types import OrderSide
from qgent.strategy.base import Strategy


def make_synthetic_data(n=200, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.uniform(1e6, 5e6, n),
    }, index=dates)


class AlwaysLongStrategy(Strategy):
    name = "AlwaysLong"

    def generate_signals(self, df):
        return pd.Series(1.0, index=df.index)


class FlipFlopStrategy(Strategy):
    name = "FlipFlop"

    def generate_signals(self, df):
        signals = pd.Series(0.0, index=df.index)
        signals.iloc[::2] = 1.0
        signals.iloc[1::2] = -1.0
        return signals


class TestComputeMetrics:
    def test_basic_metrics(self):
        equity = pd.Series([100, 110, 105, 120, 115], dtype=float)
        metrics = compute_metrics(equity)
        assert metrics.total_return == pytest.approx(0.15)
        assert metrics.max_drawdown < 0

    def test_flat_equity(self):
        equity = pd.Series([100.0] * 10)
        metrics = compute_metrics(equity)
        assert metrics.total_return == 0.0
        assert metrics.max_drawdown == 0.0


class TestSimulatedBroker:
    def test_buy_and_sell(self):
        broker = SimulatedBroker(initial_cash=10_000, config=BrokerConfig(commission_rate=0, slippage_rate=0))
        ts = pd.Timestamp("2024-01-01", tz="UTC")

        broker.submit_order("TEST", OrderSide.BUY, 10, 100, ts)
        assert broker.get_position("TEST") == 10
        assert broker.cash == pytest.approx(9_000)

        broker.submit_order("TEST", OrderSide.SELL, 10, 110, ts)
        assert broker.get_position("TEST") == 0
        assert broker.cash == pytest.approx(10_100)

    def test_insufficient_cash(self):
        broker = SimulatedBroker(initial_cash=100, config=BrokerConfig(commission_rate=0, slippage_rate=0))
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        order = broker.submit_order("TEST", OrderSide.BUY, 10, 100, ts)
        from qgent.core.types import OrderStatus
        assert order.status == OrderStatus.REJECTED


class TestBacktestEngine:
    def test_vectorized_always_long(self):
        df = make_synthetic_data()
        engine = BacktestEngine(strategy=AlwaysLongStrategy(), initial_cash=100_000)
        result = engine.run(df, mode="vectorized")
        assert result.metrics.total_return != 0
        assert len(result.equity_curve) == len(df)
        assert result.strategy_name == "AlwaysLong"

    def test_vectorized_flipflop(self):
        df = make_synthetic_data()
        engine = BacktestEngine(strategy=FlipFlopStrategy(), initial_cash=100_000)
        result = engine.run(df, mode="vectorized")
        assert result.metrics.total_trades > 0

    def test_event_driven(self):
        df = make_synthetic_data(n=100)

        class SimpleEventStrategy(Strategy):
            name = "SimpleEvent"

            def on_bar(self, idx, row, history):
                if idx == 10:
                    return self.buy()
                if idx == 50:
                    return self.sell()
                return None

        engine = BacktestEngine(strategy=SimpleEventStrategy(), initial_cash=100_000)
        result = engine.run(df, mode="event_driven", symbol="TEST")
        assert len(result.equity_curve) == 100
