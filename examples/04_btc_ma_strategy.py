"""BTC 日线均线策略回测

策略逻辑：双均线交叉
- 短期均线上穿长期均线 → 做多
- 短期均线下穿长期均线 → 平仓
支持多组参数对比
"""

import pandas as pd
import numpy as np
import yfinance as yf

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine
from qgent.factor.library import momentum, technical, volatility


# --- 1. 获取 BTC 真实日线数据 ---
print("正在获取 BTC-USD 日线数据...")
ticker = yf.Ticker("BTC-USD")
raw = ticker.history(start="2020-01-01", end="2026-05-27", interval="1d")

df = raw.rename(columns={
    "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Volume": "volume",
})[["open", "high", "low", "close", "volume"]].copy()
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
df = df[df["close"] > 0]

print(f"数据范围: {df.index[0].date()} ~ {df.index[-1].date()}, 共 {len(df)} 根日线\n")
print(f"价格范围: ${df['close'].min():,.0f} ~ ${df['close'].max():,.0f}")
print(f"期间涨幅: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}\n")


# --- 2. 定义均线策略 ---
class MAStrategy(Strategy):
    """双均线交叉策略"""

    def __init__(self, short_period: int = 10, long_period: int = 50):
        super().__init__(short_period=short_period, long_period=long_period)
        self.short_period = short_period
        self.long_period = long_period
        self.name = f"MA_{short_period}_{long_period}"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ma_short = data["close"].rolling(self.short_period).mean()
        ma_long = data["close"].rolling(self.long_period).mean()

        signal = pd.Series(0.0, index=data.index)
        signal[ma_short > ma_long] = 1.0   # 金叉持多
        signal[ma_short <= ma_long] = 0.0   # 死叉平仓（不做空）
        return signal


# --- 3. 多参数对比回测 ---
param_sets = [
    (5, 20),
    (10, 30),
    (10, 50),
    (20, 60),
    (20, 120),
    (50, 200),
]

results = []

print("=" * 80)
print(f"{'策略':<15} {'总收益':>10} {'年化':>10} {'Sharpe':>8} {'MaxDD':>10} {'Calmar':>8} {'交易次数':>8} {'胜率':>8}")
print("=" * 80)

for short_p, long_p in param_sets:
    strategy = MAStrategy(short_period=short_p, long_period=long_p)
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=100_000,
        commission=0.001,
        slippage=0.0005,
        periods_per_year=365,
    )
    result = engine.run(df, mode="vectorized")
    m = result.metrics

    results.append({
        "策略": strategy.name,
        "总收益": m.total_return,
        "年化收益": m.annual_return,
        "Sharpe": m.sharpe_ratio,
        "MaxDD": m.max_drawdown,
        "Calmar": m.calmar_ratio,
        "交易次数": m.total_trades,
        "胜率": m.win_rate,
        "波动率": m.volatility,
    })

    print(
        f"{strategy.name:<15} "
        f"{m.total_return:>10.2%} "
        f"{m.annual_return:>10.2%} "
        f"{m.sharpe_ratio:>8.3f} "
        f"{m.max_drawdown:>10.2%} "
        f"{m.calmar_ratio:>8.3f} "
        f"{m.total_trades:>8d} "
        f"{m.win_rate:>8.2%}"
    )

# --- 4. Buy & Hold 基准对比 ---
class BuyAndHold(Strategy):
    name = "BuyAndHold"
    def generate_signals(self, data):
        return pd.Series(1.0, index=data.index)

bh_engine = BacktestEngine(
    strategy=BuyAndHold(),
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
    periods_per_year=365,
)
bh_result = bh_engine.run(df, mode="vectorized")
bh_m = bh_result.metrics

print("-" * 80)
print(
    f"{'BuyAndHold':<15} "
    f"{bh_m.total_return:>10.2%} "
    f"{bh_m.annual_return:>10.2%} "
    f"{bh_m.sharpe_ratio:>8.3f} "
    f"{bh_m.max_drawdown:>10.2%} "
    f"{bh_m.calmar_ratio:>8.3f} "
    f"{bh_m.total_trades:>8d} "
    f"{bh_m.win_rate:>8.2%}"
)
print("=" * 80)

# --- 5. 找出最优参数 ---
results_df = pd.DataFrame(results)
best_sharpe = results_df.loc[results_df["Sharpe"].idxmax()]
best_calmar = results_df.loc[results_df["Calmar"].idxmax()]
best_return = results_df.loc[results_df["总收益"].idxmax()]

print(f"\n最佳 Sharpe:  {best_sharpe['策略']} (Sharpe={best_sharpe['Sharpe']:.3f})")
print(f"最佳 Calmar:  {best_calmar['策略']} (Calmar={best_calmar['Calmar']:.3f})")
print(f"最高收益:     {best_return['策略']} (Return={best_return['总收益']:.2%})")

# --- 6. 最优策略详细报告 ---
best_name = best_sharpe["策略"]
parts = best_name.split("_")
best_strategy = MAStrategy(short_period=int(parts[1]), long_period=int(parts[2]))
best_engine = BacktestEngine(
    strategy=best_strategy,
    initial_cash=100_000,
    commission=0.001,
    slippage=0.0005,
    periods_per_year=365,
)
best_result = best_engine.run(df, mode="vectorized")
best_result.report()

print("月度收益表:")
print(best_result.monthly_returns().to_string(float_format=lambda x: f"{x:.1%}"))
