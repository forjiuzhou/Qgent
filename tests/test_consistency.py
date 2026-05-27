"""验证向量化 vs 事件驱动回测一致性"""

import numpy as np
import pandas as pd

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine


def rsi(series, period):
    d = series.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


def make_data(n=500, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-01", periods=n, tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.DataFrame({
        "open": close * (1 + np.random.uniform(-0.003, 0.003, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.008, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.008, n))),
        "close": close,
        "volume": np.random.uniform(1e6, 5e6, n),
    }, index=dates)


class RSIMAKD(Strategy):
    name = "RSIMAKD"
    def generate_signals(self, data):
        r = rsi(data["close"], 6)
        diff = r.rolling(34).mean() - r.rolling(50).mean()
        K = diff.rolling(14).mean()
        D = K.rolling(36).mean()
        delta = K - D
        sig = pd.Series(np.nan, index=data.index)
        sig[(delta > 0) & (delta.shift(1) <= 0)] = 1.0
        sig[(delta < 0) & (delta.shift(1) >= 0)] = 0.0
        return sig.ffill().fillna(0)


class AlwaysLong(Strategy):
    name = "AlwaysLong"
    def generate_signals(self, data):
        return pd.Series(1.0, index=data.index)


# ============================================================

def compare(strategy, label, comm=0.001, slip=0.0005):
    df = make_data(500)

    vec_engine = BacktestEngine(strategy=strategy, initial_cash=100_000,
                                commission=comm, slippage=slip)
    vec = vec_engine.run(df, mode="vectorized")

    # Re-create to reset state
    evt_engine = BacktestEngine(strategy=type(strategy)(), initial_cash=100_000,
                                commission=comm, slippage=slip)
    evt = evt_engine.run(df, mode="event_driven", symbol="TEST")

    vm, em = vec.metrics, evt.metrics
    diff_ret = abs(vm.total_return - em.total_return)
    diff_eq = abs(vec.equity_curve.iloc[-1] - evt.equity_curve.iloc[-1])
    rel_diff = diff_eq / vec.equity_curve.iloc[-1] * 100

    print(f"\n{'=' * 70}")
    print(f"  {label}  (commission={comm}, slippage={slip})")
    print(f"{'=' * 70}")
    print(f"{'指标':<16} {'向量化':>14} {'事件驱动':>14} {'差异':>14}")
    print(f"{'-' * 70}")
    print(f"{'总收益':<16} {vm.total_return:>14.4%} {em.total_return:>14.4%} {diff_ret:>14.4%}")
    print(f"{'Sharpe':<16} {vm.sharpe_ratio:>14.4f} {em.sharpe_ratio:>14.4f} {abs(vm.sharpe_ratio - em.sharpe_ratio):>14.4f}")
    print(f"{'MaxDD':<16} {vm.max_drawdown:>14.4%} {em.max_drawdown:>14.4%} {abs(vm.max_drawdown - em.max_drawdown):>14.4%}")
    print(f"{'最终净值':<16} ${vec.equity_curve.iloc[-1]:>13,.2f} ${evt.equity_curve.iloc[-1]:>13,.2f}    {rel_diff:.3f}%")
    print(f"{'交易次数':<16} {vm.total_trades:>14d} {em.total_trades:>14d}")
    return rel_diff


# 测试 1: RSIMAKD 零费用
d1 = compare(RSIMAKD(), "RSIMAKD 零费用", comm=0.0, slip=0.0)

# 测试 2: RSIMAKD 正常费用
d2 = compare(RSIMAKD(), "RSIMAKD 正常费用", comm=0.001, slip=0.0005)

# 测试 3: AlwaysLong 零费用
d3 = compare(AlwaysLong(), "AlwaysLong 零费用", comm=0.0, slip=0.0)

# 测试 4: AlwaysLong 正常费用
d4 = compare(AlwaysLong(), "AlwaysLong 正常费用", comm=0.001, slip=0.0005)

# 汇总
print(f"\n\n{'=' * 50}")
print("一致性验证汇总")
print(f"{'=' * 50}")
print(f"RSIMAKD   零费用 净值差:  {d1:.3f}%")
print(f"RSIMAKD   正常费用 净值差: {d2:.3f}%")
print(f"AlwaysLong 零费用 净值差: {d3:.3f}%")
print(f"AlwaysLong 正常费用 净值差: {d4:.3f}%")
all_pass = all(d < 1.0 for d in [d1, d2, d3, d4])
print(f"\n结论: {'全部 < 1%，两种模式一致' if all_pass else '存在较大差异，需要检查'}")
