"""BTC 周线超跌修复策略回测

核心逻辑：
- RSI(6) -> SMA(6)=慢线 -> SMA(2)=快线
- 买入：慢线 4 周最低 < 28 + 慢线上穿快线（超卖反转）
- 买入后持有不卖（投资逻辑，不是交易）
"""

import numpy as np
import pandas as pd
import yfinance as yf

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine


# ============================================================
# 1. 获取数据
# ============================================================

print("获取 BTC-USD 日线 -> 重采样周线...\n")
raw = yf.Ticker("BTC-USD").history(start="2014-01-01", end="2026-05-27", interval="1d")
df_daily = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
df_daily = df_daily[["open", "high", "low", "close", "volume"]].copy()
df_daily.index = pd.to_datetime(df_daily.index, utc=True)
df_daily = df_daily[df_daily["close"] > 0].sort_index()
print(f"  日线: {df_daily.index[0].date()} ~ {df_daily.index[-1].date()}, {len(df_daily)} bars")

df = df_daily.resample("W-MON", label="left", closed="left").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna(subset=["open"])
print(f"  周线: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} bars\n")


# ============================================================
# 2. 指标计算
# ============================================================

def rsi(series, period):
    d = series.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


RSI_LEN = 6
SMA_SLOW = 6
SMA_FAST = 2
BUY_ZONE = 28

rsi_val = rsi(df["close"], RSI_LEN)
rsi_sma = rsi_val.rolling(SMA_SLOW).mean()
rsi_sma2 = rsi_sma.rolling(SMA_FAST).mean()
rsi_low = rsi_sma.rolling(4).min()
rsi_low2 = rsi_val.rolling(3).min()

df["rsi_val"] = rsi_val
df["rsi_sma"] = rsi_sma
df["rsi_sma2"] = rsi_sma2
df["rsi_low"] = rsi_low
df["rsi_low2"] = rsi_low2

print("指标计算完成\n")


# ============================================================
# 3. 策略：买入后持有
# ============================================================

class WeeklyOversold_Hold(Strategy):
    """超跌买入 + 持有不卖"""
    name = "Oversold_Hold"

    def generate_signals(self, data):
        slow = data["rsi_sma"]
        fast = data["rsi_sma2"]
        cross_up = (slow > fast) & (slow.shift(1) <= fast.shift(1))
        buy = (data["rsi_low"] < BUY_ZONE) & cross_up

        signal = pd.Series(np.nan, index=data.index)
        signal[buy] = 1.0
        # 没有卖出信号，买入后一直持有
        signal = signal.ffill().fillna(0)
        return signal


class BuyAndHold(Strategy):
    name = "BuyAndHold"
    def generate_signals(self, data):
        return pd.Series(1.0, index=data.index)


# ============================================================
# 4. 回测
# ============================================================

df_test = df.dropna(subset=["rsi_sma2"]).copy()

print(f"回测区间: {df_test.index[0].date()} ~ {df_test.index[-1].date()}")
print(f"有效K线:  {len(df_test)} 根周线 ({len(df_test)/52:.1f} 年)")
print(f"起始价:   ${df_test['close'].iloc[0]:,.0f}")
print(f"结束价:   ${df_test['close'].iloc[-1]:,.0f}")
print(f"Buy&Hold: {(df_test['close'].iloc[-1] / df_test['close'].iloc[0] - 1):.2%}\n")

strategies = [WeeklyOversold_Hold(), BuyAndHold()]
results = {}
for strat in strategies:
    engine = BacktestEngine(
        strategy=strat, initial_cash=100_000,
        commission=0.001, slippage=0.0005, periods_per_year=52,
    )
    results[strat.name] = engine.run(df_test, mode="vectorized")

# 对比
print("=" * 75)
print(f"{'策略':<18} {'总收益':>10} {'年化':>8} {'Sharpe':>8} {'MaxDD':>10} {'Calmar':>8}")
print("=" * 75)
for name, result in results.items():
    m = result.metrics
    print(f"{name:<18} {m.total_return:>10.2%} {m.annual_return:>8.2%} "
          f"{m.sharpe_ratio:>8.3f} {m.max_drawdown:>10.2%} {m.calmar_ratio:>8.3f}")
print("=" * 75)

# 买入信号时间点
strat = WeeklyOversold_Hold()
signals = strat.generate_signals(df_test)
buy_bars = (signals == 1) & (signals.shift(1) == 0)
holding_start = signals.index[buy_bars][0] if buy_bars.any() else None

print(f"\n--- 买入信号触发记录 ---")
print(f"{'日期':>12} {'价格':>10} {'RSI':>6} {'慢线':>6} {'快线':>6}  信号后走势")
if buy_bars.any():
    for idx in df_test.index[buy_bars]:
        r = df_test.loc[idx]
        loc = df_test.index.get_loc(idx)
        # 之后 4/12/26/52 周的价格
        future = []
        for w, label in [(4, "1月"), (12, "3月"), (26, "半年"), (52, "1年")]:
            if loc + w < len(df_test):
                fp = df_test["close"].iloc[loc + w]
                ret = (fp - r["close"]) / r["close"]
                future.append(f"{label}{ret:+.0%}")
            else:
                future.append(f"{label}:N/A")
        print(f"{str(idx.date()):>12} ${r['close']:>9,.0f} {r['rsi_val']:>6.1f} "
              f"{r['rsi_sma']:>6.1f} {r['rsi_sma2']:>6.1f}  {', '.join(future)}")

# 持仓期与空仓期
if holding_start:
    hold_bars = (signals == 1).sum()
    total_bars = len(signals)
    wait_bars = total_bars - hold_bars
    print(f"\n空仓等待: {wait_bars} 周 ({wait_bars/52:.1f} 年)")
    print(f"持仓至今: {hold_bars} 周 ({hold_bars/52:.1f} 年)")
    print(f"首次入场: {holding_start.date()}, 价格 ${df_test.loc[holding_start, 'close']:,.0f}")
    print(f"如果在首次信号买入并持有到现在: {(df_test['close'].iloc[-1] / df_test.loc[holding_start, 'close'] - 1):.2%}")
