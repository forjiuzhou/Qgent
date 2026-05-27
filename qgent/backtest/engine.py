from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from qgent.backtest.broker import SimulatedBroker, BrokerConfig
from qgent.backtest.metrics import compute_metrics
from qgent.backtest.report import BacktestResult
from qgent.core.errors import BacktestError
from qgent.core.types import OrderSide
from qgent.strategy.base import Strategy
from qgent.strategy.position import PositionSizer, FixedSizer
from qgent.strategy.risk import RiskManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtest engine supporting both vectorized and event-driven modes.

    Usage:
        engine = BacktestEngine(strategy=MyStrategy(), initial_cash=100000)
        result = engine.run(df)
        result.report()
        result.plot()
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        sizer: Optional[PositionSizer] = None,
        risk_manager: Optional[RiskManager] = None,
        periods_per_year: int = 252,
    ):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.sizer = sizer or FixedSizer(fraction=0.95)
        self.risk_manager = risk_manager
        self.periods_per_year = periods_per_year
        self.broker = SimulatedBroker(
            initial_cash=initial_cash,
            config=BrokerConfig(commission_rate=commission, slippage_rate=slippage),
        )

    def run(self, df: pd.DataFrame, mode: str = "vectorized", symbol: str = "asset") -> BacktestResult:
        """Run backtest on OHLCV data.

        Args:
            df: OHLCV DataFrame.
            mode: 'vectorized' or 'event_driven'.
            symbol: Symbol name for the asset.

        Returns:
            BacktestResult with metrics, equity curve, and trades.
        """
        if df.empty:
            raise BacktestError("Cannot run backtest on empty DataFrame")

        if mode == "vectorized":
            return self._run_vectorized(df, symbol)
        elif mode == "event_driven":
            return self._run_event_driven(df, symbol)
        else:
            raise BacktestError(f"Unknown mode: {mode}. Use 'vectorized' or 'event_driven'")

    def _run_vectorized(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Fast vectorized backtest."""
        signals = self.strategy.generate_signals(df)

        positions = signals.shift(1).fillna(0)
        returns = df["close"].pct_change().fillna(0)

        commission_cost = positions.diff().abs().fillna(0) * self.broker.config.commission_rate
        slippage_cost = positions.diff().abs().fillna(0) * self.broker.config.slippage_rate

        strategy_returns = positions * returns - commission_cost - slippage_cost
        equity_curve = self.initial_cash * (1 + strategy_returns).cumprod()

        position_changes = positions.diff().fillna(0)
        n_trades = (position_changes != 0).sum()

        metrics = compute_metrics(
            equity_curve,
            risk_free_rate=0.0,
            periods_per_year=self.periods_per_year,
        )
        metrics.total_trades = int(n_trades)

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            returns=strategy_returns,
            positions=positions,
            signals=signals,
            trades=self.broker.trades,
            df=df,
            strategy_name=self.strategy.name,
        )

    def _run_event_driven(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Event-driven backtest with order simulation."""
        self.broker.reset()
        if self.risk_manager:
            self.risk_manager.reset()

        self.strategy.on_start(df)

        equity_values = []
        signal_values = []
        position_values = []

        for i in range(len(df)):
            row = df.iloc[i]
            history = df.iloc[: i + 1]
            current_price = row["close"]
            timestamp = df.index[i]

            if self.risk_manager and self.broker.get_position(symbol) != 0:
                pos_qty = self.broker.get_position(symbol)
                entry_price = self.broker.avg_prices.get(symbol, current_price)
                if pos_qty > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                equity = self.broker.mark_to_market({symbol: current_price})
                peak = max(equity_values) if equity_values else self.initial_cash
                dd = (equity - peak) / peak if peak > 0 else 0

                if self.risk_manager.should_close(pnl_pct, {"drawdown": abs(dd)}):
                    close_qty = abs(pos_qty)
                    side = OrderSide.SELL if pos_qty > 0 else OrderSide.BUY
                    self.broker.submit_order(symbol, side, close_qty, current_price, timestamp)
                    signal_values.append(0)
                    pos = self.broker.get_position(symbol)
                    position_values.append(pos)
                    equity_values.append(self.broker.mark_to_market({symbol: current_price}))
                    continue

            signal = self.strategy.on_bar(i, row, history)

            if signal is not None:
                signal_values.append(signal.weighted_direction)
                direction = signal.weighted_direction
                current_pos = self.broker.get_position(symbol)

                if direction > 0 and current_pos <= 0:
                    if current_pos < 0:
                        self.broker.submit_order(
                            symbol, OrderSide.BUY, abs(current_pos), current_price, timestamp
                        )
                    portfolio_val = self.broker.mark_to_market({symbol: current_price})
                    units = self.sizer.size(direction, current_price, portfolio_val)
                    if abs(units) > 0:
                        self.broker.submit_order(
                            symbol, OrderSide.BUY, abs(units), current_price, timestamp
                        )

                elif direction < 0 and current_pos >= 0:
                    if current_pos > 0:
                        self.broker.submit_order(
                            symbol, OrderSide.SELL, abs(current_pos), current_price, timestamp
                        )
                    portfolio_val = self.broker.mark_to_market({symbol: current_price})
                    units = self.sizer.size(direction, current_price, portfolio_val)
                    if abs(units) > 0:
                        self.broker.submit_order(
                            symbol, OrderSide.SELL, abs(units), current_price, timestamp
                        )

                elif direction == 0 and current_pos != 0:
                    side = OrderSide.SELL if current_pos > 0 else OrderSide.BUY
                    self.broker.submit_order(
                        symbol, side, abs(current_pos), current_price, timestamp
                    )
            else:
                signal_values.append(0)

            pos = self.broker.get_position(symbol)
            position_values.append(pos)
            equity_values.append(self.broker.mark_to_market({symbol: current_price}))

        self.strategy.on_end(df)

        equity_curve = pd.Series(equity_values, index=df.index, name="equity")
        strategy_returns = equity_curve.pct_change().fillna(0)
        signals_series = pd.Series(signal_values, index=df.index, name="signal")
        positions_series = pd.Series(position_values, index=df.index, name="position")

        metrics = compute_metrics(
            equity_curve,
            trades=self.broker.trades,
            risk_free_rate=0.0,
            periods_per_year=self.periods_per_year,
        )

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            returns=strategy_returns,
            positions=positions_series,
            signals=signals_series,
            trades=self.broker.trades,
            df=df,
            strategy_name=self.strategy.name,
        )
