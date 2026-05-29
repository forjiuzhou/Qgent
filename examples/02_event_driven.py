"""Qgent Example: Event-driven backtest with risk management.

Demonstrates the event-driven engine with stop-loss and trailing stop.
"""

import numpy as np
import pandas as pd

from qgent.strategy.base import Strategy
from qgent.strategy.signal import Signal, SignalType
from qgent.strategy.risk import RiskManager, StopLoss, TrailingStop
from qgent.strategy.position import VolTargetSizer
from qgent.backtest.engine import BacktestEngine
from qgent.factor.library import momentum, technical


# --- 1. Synthetic data ---
np.random.seed(123)
n_days = 300
dates = pd.bdate_range("2023-01-01", periods=n_days, tz="UTC")
returns = np.random.normal(0.0003, 0.015, n_days)
close = 50 * np.exp(np.cumsum(returns))

df = pd.DataFrame({
    "open": close * (1 + np.random.uniform(-0.003, 0.003, n_days)),
    "high": close * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
    "low": close * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
    "close": close,
    "volume": np.random.uniform(5e5, 3e6, n_days),
}, index=dates)

df["rsi"] = momentum.rsi(df, period=14)
df["bb_pct"] = technical.bollinger_pct(df, period=20)


# --- 2. Event-driven strategy ---
class MeanReversionStrategy(Strategy):
    name = "MeanReversion_BB"

    def on_bar(self, idx, row, history):
        if idx < 25:
            return None

        if pd.isna(row.get("bb_pct")) or pd.isna(row.get("rsi")):
            return None

        if row["bb_pct"] < 0.1 and row["rsi"] < 35:
            return self.buy(weight=1.0)
        elif row["bb_pct"] > 0.9 and row["rsi"] > 65:
            return self.sell(weight=1.0)
        return None


# --- 3. Risk management ---
risk_mgr = RiskManager([
    StopLoss(threshold=0.05),
    TrailingStop(trail_pct=0.03),
])

# --- 4. Run ---
engine = BacktestEngine(
    strategy=MeanReversionStrategy(),
    initial_cash=100_000,
    commission=0.001,
    sizer=VolTargetSizer(target_vol=0.15),
    risk_manager=risk_mgr,
)

result = engine.run(df, mode="event_driven", symbol="SYN/USD")
result.report()

print(f"\nTrade log ({len(result.trades)} trades):")
trade_df = result.trade_log()
if not trade_df.empty:
    print(trade_df[["side", "entry_price", "exit_price", "pnl", "return_pct"]].to_string())
