"""BTC 30m RSIMAKD_THETA 策略回测

原策略来自 TradingView Pine Script，核心逻辑：
1. 30m 级别：RSI(6) → 双重 SMA 平滑 → MAKD 震荡指标 (K-D = delta)
2. 日线级别：多组均线和 RSI 作为过滤条件
3. 入场：delta 上穿 0 轴 + 日线 RSI<30 过滤（非极端熊市）
4. 出场：delta 下穿 0 轴

多周期融合：日线指标 forward-fill 到 30m（模拟 Pine 的 barmerge.lookahead_off）
"""

import numpy as np
import pandas as pd
import ccxt
import time

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine


# ============================================================
# 1. 获取数据 (OKX)
# ============================================================

def fetch_ohlcv(symbol: str, timeframe: str, since: str, exchange_id: str = "okx") -> pd.DataFrame:
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    all_data = []
    limit = 300 if exchange_id == "okx" else 1000

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except Exception as e:
            print(f"  retry due to: {e}")
            time.sleep(2)
            continue

        if not ohlcv:
            break
        all_data.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        if last_ts >= now_ms - 60_000:
            break
        if len(ohlcv) < limit:
            break
        since_ms = last_ts + 1
        time.sleep(0.15)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  {timeframe}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} bars")
    return df


print("正在从 OKX 获取 BTC/USDT 数据...\n")
df_30m = fetch_ohlcv("BTC/USDT", "30m", "2022-01-01")
df_daily = fetch_ohlcv("BTC/USDT", "1d", "2021-01-01")
print()


# ============================================================
# 2. 辅助函数
# ============================================================

def rsi(series: pd.Series, period: int) -> pd.Series:
    """Pine Script 风格 RSI（RMA 平滑）"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ============================================================
# 3. 计算日线指标 → forward-fill 到 30m
# ============================================================

daily = pd.DataFrame(index=df_daily.index)
daily["daily_ma_1"] = df_daily["close"].rolling(1).mean()
daily["daily_ma_2"] = df_daily["close"].rolling(2).mean()
daily["daily_ma_5"] = df_daily["close"].rolling(5).mean()
daily["daily_ma_9"] = df_daily["close"].rolling(9).mean()
daily["daily_ma_37"] = df_daily["close"].rolling(37).mean()

daily_rsi_6 = rsi(df_daily["close"], 6)
daily["daily_rsi"] = daily_rsi_6.rolling(6).mean()   # SMA(RSI(6), 6)

daily_rsi_5 = rsi(df_daily["close"], 5)
daily["daily_rsi_5"] = daily_rsi_5
daily["daily_rsi_5_ma"] = daily_rsi_5.rolling(5).mean()

daily_rsi_30 = rsi(df_daily["close"], 30)
daily["daily_rsi_30"] = daily_rsi_30
daily["daily_rsi_ma"] = daily_rsi_30.rolling(30).mean()

# Pine barmerge.lookahead_off → 使用已完成的前一根日线
daily_shifted = daily.shift(1)

# forward-fill 到 30m 时间轴
daily_on_30m = daily_shifted.reindex(df_30m.index, method="ffill")

df = df_30m.copy()
for col in daily_on_30m.columns:
    df[col] = daily_on_30m[col]


# ============================================================
# 4. 计算 30m 级别指标
# ============================================================

# RSIMAKD 核心
rsi_6 = rsi(df["close"], 6)
fast_rsi = rsi_6.rolling(34).mean()
slow_rsi = rsi_6.rolling(50).mean()
diff = fast_rsi - slow_rsi
K = diff.rolling(14).mean()
D = K.rolling(36).mean()
df["delta"] = K - D

# 30m RSI lowest(14, 12)
rsi_14_30m = rsi(df["close"], 14)
df["rsi_30m_lowest"] = rsi_14_30m.rolling(12).min()

print("指标计算完成\n")


# ============================================================
# 5. 定义策略
# ============================================================

class RSIMAKD_THETA(Strategy):
    name = "RSIMAKD_THETA"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        delta = data["delta"]
        close = data["close"]

        # flag_1: 极端熊市（不入场）
        flag_1 = (
            (data["daily_ma_2"] < data["daily_ma_9"]) &
            (data["daily_ma_1"] < data["daily_ma_37"]) &
            (close < data["daily_ma_2"]) &
            (close < data["daily_ma_5"]) &
            (data["daily_rsi"] < 30) &
            (data["rsi_30m_lowest"] < 25)
        )

        # long_flag3: 日线 RSI 偏低（超卖区域，反弹概率大）
        long_flag3 = data["daily_rsi"] < 30

        # delta 穿越信号
        delta_cross_up = (delta > 0) & (delta.shift(1) <= 0)
        delta_cross_down = (delta < 0) & (delta.shift(1) >= 0)

        # 生成持仓信号
        signal = pd.Series(np.nan, index=data.index)
        entry = delta_cross_up & long_flag3 & (~flag_1)
        signal[entry] = 1.0
        signal[delta_cross_down] = 0.0
        signal = signal.ffill().fillna(0)
        return signal


# ============================================================
# 6. 回测
# ============================================================

df_clean = df.dropna(subset=["delta", "daily_ma_37", "daily_rsi"]).copy()
print(f"回测区间: {df_clean.index[0]} ~ {df_clean.index[-1]}")
print(f"有效K线:  {len(df_clean)} 根 30m")
print(f"起始价:   ${df_clean['close'].iloc[0]:,.0f}")
print(f"结束价:   ${df_clean['close'].iloc[-1]:,.0f}")
print(f"Buy&Hold: {(df_clean['close'].iloc[-1] / df_clean['close'].iloc[0] - 1):.2%}\n")

strategy = RSIMAKD_THETA()
engine = BacktestEngine(
    strategy=strategy,
    initial_cash=100_000,
    commission=0.0005,
    slippage=0.0002,
    periods_per_year=365 * 48,
)
result = engine.run(df_clean, mode="vectorized")
result.report()

# Buy & Hold 基准
class BuyAndHold(Strategy):
    name = "BuyAndHold"
    def generate_signals(self, data):
        return pd.Series(1.0, index=data.index)

bh_engine = BacktestEngine(
    strategy=BuyAndHold(),
    initial_cash=100_000,
    commission=0.0005,
    slippage=0.0002,
    periods_per_year=365 * 48,
)
bh_result = bh_engine.run(df_clean, mode="vectorized")

# 对比表
print("\n" + "=" * 60)
print(f"{'指标':<20} {'RSIMAKD_THETA':>15} {'BuyAndHold':>15}")
print("=" * 60)
m1, m2 = result.metrics, bh_result.metrics
rows = [
    ("总收益", f"{m1.total_return:.2%}", f"{m2.total_return:.2%}"),
    ("年化收益", f"{m1.annual_return:.2%}", f"{m2.annual_return:.2%}"),
    ("Sharpe", f"{m1.sharpe_ratio:.3f}", f"{m2.sharpe_ratio:.3f}"),
    ("Sortino", f"{m1.sortino_ratio:.3f}", f"{m2.sortino_ratio:.3f}"),
    ("最大回撤", f"{m1.max_drawdown:.2%}", f"{m2.max_drawdown:.2%}"),
    ("Calmar", f"{m1.calmar_ratio:.3f}", f"{m2.calmar_ratio:.3f}"),
    ("波动率", f"{m1.volatility:.2%}", f"{m2.volatility:.2%}"),
    ("交易次数", f"{m1.total_trades}", f"{m2.total_trades}"),
    ("胜率(bar级)", f"{m1.win_rate:.2%}", f"{m2.win_rate:.2%}"),
]
for label, v1, v2 in rows:
    print(f"{label:<20} {v1:>15} {v2:>15}")
print("=" * 60)

# 月度收益
print("\n月度收益表:")
print(result.monthly_returns().to_string(float_format=lambda x: f"{x:.1%}"))

# 信号统计
signals = strategy.generate_signals(df_clean)
entries = (signals == 1) & (signals.shift(1) == 0)
exits = (signals == 0) & (signals.shift(1) == 1)
holding_bars = (signals == 1).sum()
total_bars = len(signals)

print(f"\n--- 信号统计 ---")
print(f"入场次数:     {entries.sum()}")
print(f"出场次数:     {exits.sum()}")
print(f"持仓占比:     {holding_bars / total_bars:.1%} ({holding_bars}/{total_bars} bars)")
avg_hold = holding_bars / max(entries.sum(), 1)
print(f"平均持仓时长: {avg_hold:.0f} bars (~{avg_hold * 0.5:.0f} 小时)")
