"""RSIMAKD 参数优化 — 美股日线

在多只美股上做参数网格搜索，找到最优的 RSIMAKD 参数组合。

参数空间：
- rsi_period:  RSI 周期
- fast_period: RSI 快线 SMA 周期
- slow_period: RSI 慢线 SMA 周期
- k_period:    K 线 SMA 周期
- d_period:    D 线 SMA 周期
"""

import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine


# ============================================================
# 1. 获取美股数据
# ============================================================

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
START = "2015-01-01"
END = "2026-05-27"

print("正在获取美股日线数据...\n")
stock_data = {}
for sym in SYMBOLS:
    try:
        raw = yf.Ticker(sym).history(start=START, end=END, interval="1d")
        df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df["close"] > 0].sort_index()
        if len(df) > 500:
            stock_data[sym] = df
            print(f"  {sym}: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)} bars")
    except Exception as e:
        print(f"  {sym}: 获取失败 - {e}")

print(f"\n成功加载 {len(stock_data)} 只股票\n")


# ============================================================
# 2. RSI 计算
# ============================================================

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ============================================================
# 3. 参数化策略
# ============================================================

class RSIMAKD(Strategy):
    def __init__(self, rsi_p=6, fast_p=34, slow_p=50, k_p=14, d_p=36):
        super().__init__(rsi_p=rsi_p, fast_p=fast_p, slow_p=slow_p, k_p=k_p, d_p=d_p)
        self.rsi_p = rsi_p
        self.fast_p = fast_p
        self.slow_p = slow_p
        self.k_p = k_p
        self.d_p = d_p
        self.name = f"RSIMAKD({rsi_p},{fast_p},{slow_p},{k_p},{d_p})"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        r = rsi(data["close"], self.rsi_p)
        fast = r.rolling(self.fast_p).mean()
        slow = r.rolling(self.slow_p).mean()
        diff = fast - slow
        K = diff.rolling(self.k_p).mean()
        D = K.rolling(self.d_p).mean()
        delta = K - D

        cross_up = (delta > 0) & (delta.shift(1) <= 0)
        cross_down = (delta < 0) & (delta.shift(1) >= 0)

        signal = pd.Series(np.nan, index=data.index)
        signal[cross_up] = 1.0
        signal[cross_down] = 0.0
        return signal.ffill().fillna(0)


# ============================================================
# 4. 单组参数在所有股票上的平均表现
# ============================================================

def evaluate_params(rsi_p, fast_p, slow_p, k_p, d_p, data_dict):
    """在所有股票上回测，返回平均指标。"""
    if fast_p >= slow_p:
        return None

    strategy = RSIMAKD(rsi_p, fast_p, slow_p, k_p, d_p)
    sharpes = []
    returns = []
    calmars = []
    max_dds = []
    win_rates = []
    trade_counts = []

    for sym, df in data_dict.items():
        engine = BacktestEngine(
            strategy=RSIMAKD(rsi_p, fast_p, slow_p, k_p, d_p),
            initial_cash=100_000,
            commission=0.001,
            slippage=0.0005,
            periods_per_year=252,
        )
        try:
            result = engine.run(df, mode="vectorized")
        except Exception:
            continue

        m = result.metrics
        if np.isnan(m.sharpe_ratio) or np.isinf(m.sharpe_ratio):
            continue

        sharpes.append(m.sharpe_ratio)
        returns.append(m.total_return)
        calmars.append(m.calmar_ratio)
        max_dds.append(m.max_drawdown)

        signals = strategy.generate_signals(df)
        entries = ((signals == 1) & (signals.shift(1) == 0))
        exits = ((signals == 0) & (signals.shift(1) == 1))
        entry_idx = signals.index[entries]
        exit_idx = signals.index[exits]
        n = min(len(entry_idx), len(exit_idx))
        trade_counts.append(n)
        if n > 0:
            pnls = [(df.loc[exit_idx[i], "close"] - df.loc[entry_idx[i], "close"])
                    / df.loc[entry_idx[i], "close"] for i in range(n)]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            win_rates.append(wr)

    if not sharpes:
        return None

    return {
        "rsi_p": rsi_p, "fast_p": fast_p, "slow_p": slow_p,
        "k_p": k_p, "d_p": d_p,
        "avg_sharpe": np.mean(sharpes),
        "avg_return": np.mean(returns),
        "avg_calmar": np.mean(calmars),
        "avg_maxdd": np.mean(max_dds),
        "avg_win_rate": np.mean(win_rates) if win_rates else 0,
        "avg_trades": np.mean(trade_counts),
        "n_stocks": len(sharpes),
    }


# ============================================================
# 5. 参数网格搜索
# ============================================================

param_grid = {
    "rsi_p":  [4, 6, 8, 10, 14],
    "fast_p": [15, 20, 25, 30, 34, 40],
    "slow_p": [35, 40, 50, 60, 70],
    "k_p":    [8, 10, 14, 18, 22],
    "d_p":    [20, 28, 36, 44],
}

all_combos = list(itertools.product(
    param_grid["rsi_p"],
    param_grid["fast_p"],
    param_grid["slow_p"],
    param_grid["k_p"],
    param_grid["d_p"],
))
# fast < slow
all_combos = [(r, f, s, k, d) for r, f, s, k, d in all_combos if f < s]

print(f"参数组合数: {len(all_combos)}")
print("开始优化...\n")

results = []
total = len(all_combos)
for idx, (rsi_p, fast_p, slow_p, k_p, d_p) in enumerate(all_combos):
    res = evaluate_params(rsi_p, fast_p, slow_p, k_p, d_p, stock_data)
    if res is not None:
        results.append(res)
    if (idx + 1) % 100 == 0:
        print(f"  进度: {idx+1}/{total} ({(idx+1)/total:.0%})")

results_df = pd.DataFrame(results)
print(f"\n有效组合: {len(results_df)}\n")


# ============================================================
# 6. 结果排序与输出
# ============================================================

print("=" * 100)
print("Top 20 — 按平均 Sharpe 排序")
print("=" * 100)

top = results_df.sort_values("avg_sharpe", ascending=False).head(20).reset_index(drop=True)
top.index = range(1, len(top) + 1)
top.index.name = "排名"

display_cols = {
    "rsi_p": "RSI", "fast_p": "Fast", "slow_p": "Slow",
    "k_p": "K", "d_p": "D",
    "avg_sharpe": "Sharpe", "avg_return": "收益",
    "avg_calmar": "Calmar", "avg_maxdd": "MaxDD",
    "avg_win_rate": "胜率", "avg_trades": "交易数",
}
top_display = top.rename(columns=display_cols)
for col in ["Sharpe", "Calmar"]:
    top_display[col] = top_display[col].map(lambda x: f"{x:.3f}")
for col in ["收益", "MaxDD", "胜率"]:
    top_display[col] = top_display[col].map(lambda x: f"{x:.2%}")
top_display["交易数"] = top_display["交易数"].map(lambda x: f"{x:.0f}")
top_display = top_display.drop(columns=["n_stocks"], errors="ignore")
print(top_display.to_string())

# 原始参数对比
print("\n\n--- 原始参数 (6,34,50,14,36) 的表现 ---")
original = results_df[
    (results_df["rsi_p"] == 6) &
    (results_df["fast_p"] == 34) &
    (results_df["slow_p"] == 50) &
    (results_df["k_p"] == 14) &
    (results_df["d_p"] == 36)
]
if len(original) > 0:
    o = original.iloc[0]
    print(f"Sharpe:  {o['avg_sharpe']:.3f}")
    print(f"收益:    {o['avg_return']:.2%}")
    print(f"Calmar:  {o['avg_calmar']:.3f}")
    print(f"MaxDD:   {o['avg_maxdd']:.2%}")
    print(f"胜率:    {o['avg_win_rate']:.2%}")
    print(f"交易数:  {o['avg_trades']:.0f}")
else:
    print("原始参数不在搜索空间中")

# 最优参数在每只股票上的表现
best = results_df.sort_values("avg_sharpe", ascending=False).iloc[0]
print(f"\n\n--- 最优参数 ({int(best['rsi_p'])},{int(best['fast_p'])},{int(best['slow_p'])},{int(best['k_p'])},{int(best['d_p'])}) 在每只股票上的表现 ---")
print(f"{'股票':<8} {'总收益':>10} {'年化':>10} {'Sharpe':>8} {'MaxDD':>10} {'Calmar':>8} {'交易数':>6} {'胜率':>8}")
print("-" * 72)

for sym, df in stock_data.items():
    strat = RSIMAKD(int(best["rsi_p"]), int(best["fast_p"]),
                    int(best["slow_p"]), int(best["k_p"]), int(best["d_p"]))
    engine = BacktestEngine(strategy=strat, initial_cash=100_000,
                            commission=0.001, slippage=0.0005, periods_per_year=252)
    result = engine.run(df, mode="vectorized")
    m = result.metrics

    signals = strat.generate_signals(df)
    entries = (signals == 1) & (signals.shift(1) == 0)
    exit_s = (signals == 0) & (signals.shift(1) == 1)
    entry_idx = signals.index[entries]
    exit_idx = signals.index[exit_s]
    n = min(len(entry_idx), len(exit_idx))
    if n > 0:
        pnls = [(df.loc[exit_idx[i], "close"] - df.loc[entry_idx[i], "close"])
                / df.loc[entry_idx[i], "close"] for i in range(n)]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
    else:
        wr = 0

    bh_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    print(f"{sym:<8} {m.total_return:>10.2%} {m.annual_return:>10.2%} {m.sharpe_ratio:>8.3f} "
          f"{m.max_drawdown:>10.2%} {m.calmar_ratio:>8.3f} {n:>6} {wr:>8.1%}  (B&H: {bh_ret:.2%})")

# 导出结果
results_df.to_csv("examples/rsimakd_optimization.csv", index=False)
print(f"\n完整优化结果已导出: examples/rsimakd_optimization.csv ({len(results_df)} 组参数)")
