"""Qgent Quick Start: Vectorized momentum strategy on synthetic data.

This example runs entirely on synthetic data, no API keys required.
"""

import numpy as np
import pandas as pd

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine
from qgent.factor.library import momentum, volatility


# --- 1. Generate synthetic OHLCV data ---
np.random.seed(42)
n_days = 500
dates = pd.bdate_range("2022-01-01", periods=n_days, tz="UTC")
returns = np.random.normal(0.0005, 0.02, n_days)
close = 100 * np.exp(np.cumsum(returns))

df = pd.DataFrame({
    "open": close * (1 + np.random.uniform(-0.005, 0.005, n_days)),
    "high": close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
    "low": close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
    "close": close,
    "volume": np.random.uniform(1e6, 5e6, n_days),
}, index=dates)


# --- 2. Add factors ---
df["mom_20"] = momentum.returns(df, period=20)
df["rsi_14"] = momentum.rsi(df, period=14)
df["vol_20"] = volatility.realized(df, period=20)


# --- 3. Define a vectorized strategy ---
class MomentumStrategy(Strategy):
    name = "Momentum_RSI"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        long = (data["mom_20"] > 0) & (data["rsi_14"] < 70) & (data["vol_20"] < 0.4)
        short = (data["mom_20"] < -0.02) | (data["rsi_14"] > 80)
        signal = pd.Series(0.0, index=data.index)
        signal[long] = 1.0
        signal[short] = -1.0
        return signal


# --- 4. Run backtest ---
engine = BacktestEngine(
    strategy=MomentumStrategy(),
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
)

result = engine.run(df, mode="vectorized")
result.report()

print("\nMonthly returns:")
print(result.monthly_returns())
