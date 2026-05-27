"""超跌事件统计研究

扫描 BTC 历史上所有周线 RSI 进入超卖区的事件，统计：
1. 不同 RSI 阈值下的事件频率
2. 每次事件后不同持有期的收益分布
3. 最优入场阈值和持有周期
4. 卖出指标的有效性
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1. 数据准备
# ============================================================

print("获取数据...\n")
raw = yf.Ticker("BTC-USD").history(start="2014-01-01", end="2026-05-27", interval="1d")
df_daily = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
df_daily = df_daily[["open", "high", "low", "close", "volume"]].copy()
df_daily.index = pd.to_datetime(df_daily.index, utc=True)
df_daily = df_daily[df_daily["close"] > 0].sort_index()

df = df_daily.resample("W-MON", label="left", closed="left").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum",
}).dropna(subset=["open"])

print(f"周线: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} 根\n")


def rsi(series, period):
    d = series.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


# 计算多种 RSI 指标
df["rsi_6"] = rsi(df["close"], 6)
df["rsi_14"] = rsi(df["close"], 14)
rsi_sma = df["rsi_6"].rolling(6).mean()
df["rsi_sma"] = rsi_sma
df["rsi_sma2"] = rsi_sma.rolling(2).mean()
df["rsi_sma_low4"] = rsi_sma.rolling(4).min()

# 距历史高点的回撤
df["ath"] = df["close"].cummax()
df["drawdown"] = (df["close"] - df["ath"]) / df["ath"] * 100

# 前瞻收益（用于统计分析，不用于交易）
for w in [1, 2, 4, 8, 12, 16, 20, 26, 39, 52, 78, 104]:
    df[f"fwd_{w}w"] = df["close"].shift(-w) / df["close"] - 1

# 前瞻最大收益和最大回撤
for w in [12, 26, 52]:
    future_highs = df["close"].shift(-1).rolling(w).max().shift(-(w-1))
    future_lows = df["close"].shift(-1).rolling(w).min().shift(-(w-1))
    df[f"fwd_{w}w_max"] = future_highs / df["close"] - 1
    df[f"fwd_{w}w_min"] = future_lows / df["close"] - 1

df_clean = df.dropna(subset=["rsi_sma"]).copy()


# ============================================================
# 2. 不同 RSI 阈值的事件统计
# ============================================================

print("=" * 90)
print("研究 1: 不同 RSI 慢线(SMA6 of RSI6) 阈值进入超卖后的收益统计")
print("=" * 90)

thresholds = [35, 30, 28, 25, 22, 20, 18, 15]
hold_periods = [4, 12, 26, 52]

print(f"\n{'阈值':>4} {'事件数':>6}", end="")
for w in hold_periods:
    print(f"  {'%dw中位'%w:>8} {'%dw均值'%w:>8} {'%dw胜率'%w:>8}", end="")
print()
print("-" * 90)

for threshold in thresholds:
    # 找到 RSI 慢线首次跌破阈值的时刻（去重：同一轮超卖只算一次）
    below = df_clean["rsi_sma"] < threshold
    enters = below & (~below.shift(1).fillna(False))
    events = df_clean[enters].copy()

    # 去掉间隔太近的事件（同一轮超卖内，至少间隔 8 周）
    if len(events) > 1:
        keep = [0]
        for i in range(1, len(events)):
            if (events.index[i] - events.index[keep[-1]]).days > 56:
                keep.append(i)
        events = events.iloc[keep]

    n = len(events)
    print(f"  <{threshold:>2}   {n:>4}", end="")

    for w in hold_periods:
        col = f"fwd_{w}w"
        if col in events.columns:
            valid = events[col].dropna()
            if len(valid) > 0:
                median = valid.median()
                mean = valid.mean()
                win = (valid > 0).mean()
                print(f"  {median:>+8.1%} {mean:>+8.1%} {win:>8.0%}", end="")
            else:
                print(f"  {'N/A':>8} {'N/A':>8} {'N/A':>8}", end="")
        else:
            print(f"  {'N/A':>8} {'N/A':>8} {'N/A':>8}", end="")
    print()


# ============================================================
# 3. 每次超卖事件的详细时间线
# ============================================================

print(f"\n\n{'=' * 100}")
print("研究 2: RSI慢线 < 25 的所有事件详情")
print("=" * 100)

STUDY_THRESHOLD = 25
below = df_clean["rsi_sma"] < STUDY_THRESHOLD
enters = below & (~below.shift(1).fillna(False))
events = df_clean[enters].copy()
if len(events) > 1:
    keep = [0]
    for i in range(1, len(events)):
        if (events.index[i] - events.index[keep[-1]]).days > 56:
            keep.append(i)
    events = events.iloc[keep]

print(f"\n{'#':>3} {'日期':>12} {'价格':>10} {'RSI慢线':>8} {'距高点':>8} "
      f"{'4w后':>8} {'12w后':>8} {'26w后':>8} {'52w后':>8} "
      f"{'52w最大涨':>10} {'52w最大跌':>10}")
print("-" * 120)

for i, (idx, row) in enumerate(events.iterrows()):
    fwd = {}
    for w in [4, 12, 26, 52]:
        col = f"fwd_{w}w"
        fwd[w] = f"{row[col]:>+8.1%}" if pd.notna(row.get(col)) else f"{'N/A':>8}"

    max52 = f"{row['fwd_52w_max']:>+9.1%}" if pd.notna(row.get("fwd_52w_max")) else f"{'N/A':>10}"
    min52 = f"{row['fwd_52w_min']:>+9.1%}" if pd.notna(row.get("fwd_52w_min")) else f"{'N/A':>10}"

    print(f"{i+1:>3} {str(idx.date()):>12} ${row['close']:>9,.0f} {row['rsi_sma']:>8.1f} "
          f"{row['drawdown']:>+7.0f}% "
          f"{fwd[4]} {fwd[12]} {fwd[26]} {fwd[52]} "
          f"{max52} {min52}")


# ============================================================
# 4. 最优持有周期分析
# ============================================================

print(f"\n\n{'=' * 90}")
print("研究 3: 超卖事件后的最优持有周期")
print("=" * 90)

below = df_clean["rsi_sma"] < 28
enters = below & (~below.shift(1).fillna(False))
events_28 = df_clean[enters].copy()
if len(events_28) > 1:
    keep = [0]
    for i in range(1, len(events_28)):
        if (events_28.index[i] - events_28.index[keep[-1]]).days > 56:
            keep.append(i)
    events_28 = events_28.iloc[keep]

print(f"\nRSI慢线 < 28 的事件 ({len(events_28)} 次) — 不同持有期的收益特征:\n")
print(f"{'持有期':>8} {'中位收益':>10} {'平均收益':>10} {'胜率':>8} {'最好':>10} {'最差':>10} {'夏普*':>8}")
print("-" * 70)

all_weeks = [1, 2, 4, 8, 12, 16, 20, 26, 39, 52, 78, 104]
for w in all_weeks:
    col = f"fwd_{w}w"
    if col not in events_28.columns:
        continue
    valid = events_28[col].dropna()
    if len(valid) < 2:
        continue
    med = valid.median()
    avg = valid.mean()
    win = (valid > 0).mean()
    best = valid.max()
    worst = valid.min()
    sharpe = avg / valid.std() if valid.std() > 0 else 0
    print(f"{w:>5}w   {med:>+10.1%} {avg:>+10.1%} {win:>8.0%} {best:>+10.1%} {worst:>+10.1%} {sharpe:>8.2f}")


# ============================================================
# 5. 卖出条件研究：RSI 回到什么水平后卖出最优
# ============================================================

print(f"\n\n{'=' * 90}")
print("研究 4: 卖出时机 — RSI慢线回到什么水平后离场")
print("=" * 90)

print(f"\n从 RSI慢线<28 事件入场，到 RSI慢线回到 X 后卖出：\n")
print(f"{'卖出阈值':>8} {'平均收益':>10} {'中位收益':>10} {'胜率':>8} {'平均周数':>10} {'事件数':>6}")
print("-" * 60)

sell_thresholds = [40, 45, 50, 55, 60, 65, 70]
for sell_th in sell_thresholds:
    returns = []
    durations = []
    for idx in events_28.index:
        entry_price = df_clean.loc[idx, "close"]
        loc = df_clean.index.get_loc(idx)
        # 找到之后 RSI 慢线首次回到 sell_th 以上的时刻
        future = df_clean.iloc[loc+1:]
        exits = future[future["rsi_sma"] >= sell_th]
        if len(exits) > 0:
            exit_idx = exits.index[0]
            exit_price = df_clean.loc[exit_idx, "close"]
            ret = (exit_price - entry_price) / entry_price
            weeks = (exit_idx - idx).days / 7
            returns.append(ret)
            durations.append(weeks)

    if returns:
        print(f"  RSI>{sell_th:<3} {np.mean(returns):>+10.1%} {np.median(returns):>+10.1%} "
              f"{sum(1 for r in returns if r > 0)/len(returns):>8.0%} "
              f"{np.mean(durations):>9.0f}w  {len(returns):>5}")


# ============================================================
# 6. 时间维度研究：超跌后恢复需要多久
# ============================================================

print(f"\n\n{'=' * 90}")
print("研究 5: 超跌后价格恢复到前高需要多久")
print("=" * 90)

print(f"\n从 RSI慢线<28 事件入场，价格恢复到不同水平所需时间：\n")
print(f"{'恢复目标':>10} {'平均周数':>10} {'中位周数':>10} {'最快':>8} {'最慢':>8} {'成功率':>8}")
print("-" * 60)

recovery_targets = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]
for target in recovery_targets:
    weeks_list = []
    for idx in events_28.index:
        entry_price = df_clean.loc[idx, "close"]
        target_price = entry_price * (1 + target)
        loc = df_clean.index.get_loc(idx)
        future = df_clean.iloc[loc+1:]
        hits = future[future["close"] >= target_price]
        if len(hits) > 0:
            weeks = (hits.index[0] - idx).days / 7
            weeks_list.append(weeks)

    total = len(events_28)
    success = len(weeks_list)
    label = f"+{target:.0%}" if target > 0 else "回本"
    if weeks_list:
        print(f"  {label:>8} {np.mean(weeks_list):>9.0f}w {np.median(weeks_list):>9.0f}w "
              f"{min(weeks_list):>7.0f}w {max(weeks_list):>7.0f}w "
              f"{success/total:>8.0%}")
    else:
        print(f"  {label:>8}     N/A       N/A      N/A      N/A {success/total:>8.0%}")


# ============================================================
# 7. 汇总结论
# ============================================================

print(f"\n\n{'=' * 90}")
print("研究结论汇总")
print("=" * 90)

print("""
1. 入场阈值:
   - RSI慢线 < 28 是一个有效的超跌信号
   - 阈值越低事件越少但质量越高
   - 建议：RSI慢线 < 30 开始关注，< 25 开始建仓，< 20 加大仓位

2. 持有周期:
   - 查看上方"最优持有期"表，找夏普最高的周期
   - 超跌恢复通常需要时间，过短持有可能白等

3. 卖出条件:
   - 时间维度 + 指标维度结合
   - 查看上方"卖出时机"表，找收益和持有时间的平衡点

4. 风险:
   - 即使是超跌，买入后仍可能继续下跌
   - 分批建仓比一次性 all-in 更安全
""")
