"""BTC 30m RSIMAKD 纯信号策略回测

仅使用 RSIMAKD 核心指标：
- RSI(6) → SMA(34) / SMA(50) → diff → SMA(14)=K / SMA(36)=D → delta=K-D
- delta 上穿 0 → 买入
- delta 下穿 0 → 卖出
"""

import numpy as np
import pandas as pd
import ccxt
import time

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine


# ============================================================
# 1. 获取数据
# ============================================================

def fetch_ohlcv(symbol: str, timeframe: str, since: str, exchange_id: str = "okx") -> pd.DataFrame:
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    all_data = []
    limit = 300

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except Exception as e:
            print(f"  retry: {e}")
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


print("正在从 OKX 获取 BTC/USDT 30m 数据...\n")
df = fetch_ohlcv("BTC/USDT", "30m", "2022-01-01")
print()


# ============================================================
# 2. 计算 RSIMAKD 指标
# ============================================================

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


rsi_6 = rsi(df["close"], 6)
fast = rsi_6.rolling(34).mean()
slow = rsi_6.rolling(50).mean()
diff = fast - slow
K = diff.rolling(14).mean()
D = K.rolling(36).mean()
df["delta"] = K - D

print("指标计算完成\n")


# ============================================================
# 3. 策略：delta 金叉买入，死叉卖出
# ============================================================

class RSIMAKD_Pure(Strategy):
    name = "RSIMAKD_Pure"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        delta = data["delta"]
        cross_up = (delta > 0) & (delta.shift(1) <= 0)
        cross_down = (delta < 0) & (delta.shift(1) >= 0)

        signal = pd.Series(np.nan, index=data.index)
        signal[cross_up] = 1.0
        signal[cross_down] = 0.0
        signal = signal.ffill().fillna(0)
        return signal


# ============================================================
# 4. 回测
# ============================================================

df_clean = df.dropna(subset=["delta"]).copy()

print(f"回测区间: {df_clean.index[0]} ~ {df_clean.index[-1]}")
print(f"有效K线:  {len(df_clean)} 根 30m")
print(f"起始价:   ${df_clean['close'].iloc[0]:,.0f}")
print(f"结束价:   ${df_clean['close'].iloc[-1]:,.0f}")
bh_return = df_clean['close'].iloc[-1] / df_clean['close'].iloc[0] - 1
print(f"Buy&Hold: {bh_return:.2%}\n")

strategy = RSIMAKD_Pure()
engine = BacktestEngine(
    strategy=strategy,
    initial_cash=100_000,
    commission=0.0005,
    slippage=0.0002,
    periods_per_year=365 * 48,
)
result = engine.run(df_clean, mode="vectorized")

# Buy & Hold
class BuyAndHold(Strategy):
    name = "BuyAndHold"
    def generate_signals(self, data):
        return pd.Series(1.0, index=data.index)

bh_result = BacktestEngine(
    strategy=BuyAndHold(), initial_cash=100_000,
    commission=0.0005, slippage=0.0002, periods_per_year=365 * 48,
).run(df_clean, mode="vectorized")

# 输出
result.report()

m1, m2 = result.metrics, bh_result.metrics
print("=" * 60)
print(f"{'指标':<20} {'RSIMAKD_Pure':>15} {'BuyAndHold':>15}")
print("=" * 60)
for label, v1, v2 in [
    ("总收益", f"{m1.total_return:.2%}", f"{m2.total_return:.2%}"),
    ("年化收益", f"{m1.annual_return:.2%}", f"{m2.annual_return:.2%}"),
    ("Sharpe", f"{m1.sharpe_ratio:.3f}", f"{m2.sharpe_ratio:.3f}"),
    ("Sortino", f"{m1.sortino_ratio:.3f}", f"{m2.sortino_ratio:.3f}"),
    ("最大回撤", f"{m1.max_drawdown:.2%}", f"{m2.max_drawdown:.2%}"),
    ("Calmar", f"{m1.calmar_ratio:.3f}", f"{m2.calmar_ratio:.3f}"),
    ("波动率", f"{m1.volatility:.2%}", f"{m2.volatility:.2%}"),
    ("交易次数", f"{m1.total_trades}", f"{m2.total_trades}"),
]:
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

print(f"\n--- 信号统计 ---")
print(f"入场次数:     {entries.sum()}")
print(f"出场次数:     {exits.sum()}")
print(f"持仓占比:     {holding_bars / len(signals):.1%}")
avg_hold = holding_bars / max(entries.sum(), 1)
print(f"平均持仓时长: {avg_hold:.0f} bars (~{avg_hold * 0.5:.0f} 小时)")

# ============================================================
# 5. 逐笔交易记录 → 导出 CSV
# ============================================================

entry_idx = signals.index[entries]
exit_idx = signals.index[exits]
n_trades = min(len(entry_idx), len(exit_idx))

if n_trades > 0:
    trades = []
    for i in range(n_trades):
        ep = df_clean.loc[entry_idx[i], "close"]
        xp = df_clean.loc[exit_idx[i], "close"]
        ret = (xp - ep) / ep
        dur = (exit_idx[i] - entry_idx[i]).total_seconds() / 3600
        trades.append({
            "entry_time": entry_idx[i],
            "exit_time": exit_idx[i],
            "entry_price": ep,
            "exit_price": xp,
            "return": ret,
            "hours": dur,
        })
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["return"] > 0]
    losses = tdf[tdf["return"] <= 0]

    print(f"\n--- 逐笔统计 ({n_trades} 笔) ---")
    print(f"盈利笔数:     {len(wins)}")
    print(f"亏损笔数:     {len(losses)}")
    print(f"胜率:         {len(wins)/n_trades:.1%}")
    print(f"平均收益:     {tdf['return'].mean():.2%}")
    if len(wins) > 0:
        print(f"平均盈利:     {wins['return'].mean():.2%}")
    if len(losses) > 0:
        print(f"平均亏损:     {losses['return'].mean():.2%}")
    if len(wins) > 0 and len(losses) > 0:
        print(f"盈亏比:       {abs(wins['return'].mean() / losses['return'].mean()):.2f}")
    print(f"最大单笔盈利: {tdf['return'].max():.2%}")
    print(f"最大单笔亏损: {tdf['return'].min():.2%}")
    print(f"平均持仓:     {tdf['hours'].mean():.0f} 小时")

    # 导出 CSV
    export = pd.DataFrame()
    export["买入时间"] = tdf["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    export["卖出时间"] = tdf["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    export["买入价"] = tdf["entry_price"].round(2)
    export["卖出价"] = tdf["exit_price"].round(2)
    export["收益率%"] = (tdf["return"] * 100).round(3)
    export["持仓小时"] = tdf["hours"].round(1)
    export["累计收益%"] = ((1 + tdf["return"]).cumprod() * 100 - 100).round(2)
    export.index = range(1, len(export) + 1)
    export.index.name = "序号"

    out_path = "examples/rsimakd_trades.csv"
    export.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n交易记录已导出: {out_path} ({len(export)} 笔)")

    print("\n最近 20 笔交易:")
    print(export.tail(20).to_string())
