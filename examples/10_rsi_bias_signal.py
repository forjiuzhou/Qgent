"""RSI + 乖离率 信号策略回测（Pine Script 原样移植）

来源 Pine 指标："RSI策略信号 [基本面不会被破坏的大标的]"

信号逻辑（与 Pine 一致）：
  rsiValue  = RSI(close, 6)
  rsiSMA    = SMA(rsiValue, 6)        # 慢线
  rsiSMA2   = SMA(rsiSMA, 2)          # 快线（对慢线再平滑，实际更滞后）
  rsi_low   = lowest(rsiSMA, 4)
  rsi_high  = lowest(rsiSMA, 2)       # 注意：Pine 里用的是 lowest，不是 highest

  买入 buySignal = (rsi_low < 28) and crossover(rsiSMA, rsiSMA2)
  卖出 sellSignal = (rsi_high > 70) and crossunder(rsiSMA, rsiSMA2)
                    and (biasBull > 10)
       biasBull = (low - SMA(close,250)) / SMA(close,250) * 100

回测约定：买入信号 -> 开多；卖出信号 -> 清仓；之间持有。
（Pine 原脚本只画箭头不管理仓位，这里把箭头翻译成多/空仓位。）

用 250 日均线 => 日线级别。在 BTC 和注释点名的"基本面稳健"标的上测试。
"""

import numpy as np
import pandas as pd
import yfinance as yf

from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine


# ============================================================
# 指标
# ============================================================

def rsi(series, period):
    d = series.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


# 参数（与 Pine 默认一致）
RSI_LEN = 6
SMA_SLOW = 6
SMA_FAST = 2
BUY_ZONE = 28
SELL_ZONE = 70
MA_LEN = 250
BIAS_BULL = 10.0


class RSIBiasSignal(Strategy):
    """买入信号开多，卖出信号清仓，中间持有。"""
    name = "RSI_Bias"

    def generate_signals(self, data):
        close = data["close"]
        low = data["low"]

        rsi_val = rsi(close, RSI_LEN)
        slow = rsi_val.rolling(SMA_SLOW).mean()        # rsiSMA
        fast = slow.rolling(SMA_FAST).mean()           # rsiSMA2
        rsi_low = slow.rolling(4).min()
        rsi_high = slow.rolling(2).min()               # Pine: lowest(rsiSMA, 2)

        cross_up = (slow > fast) & (slow.shift(1) <= fast.shift(1))
        cross_dn = (slow < fast) & (slow.shift(1) >= fast.shift(1))

        ma = close.rolling(MA_LEN).mean()
        bias_bull = (low - ma) / ma * 100

        buy = (rsi_low < BUY_ZONE) & cross_up
        sell = (rsi_high > SELL_ZONE) & cross_dn & (bias_bull > BIAS_BULL)

        sig = pd.Series(np.nan, index=data.index)
        sig[buy] = 1.0
        sig[sell] = 0.0
        return sig.ffill().fillna(0)


class BuyAndHold(Strategy):
    name = "BuyAndHold"
    def generate_signals(self, data):
        return pd.Series(1.0, index=data.index)


# ============================================================
# 数据 & 回测
# ============================================================

def load(sym, start, end):
    raw = yf.Ticker(sym).history(start=start, end=end, interval="1d")
    df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df[df["close"] > 0].sort_index()


SYMBOLS = ["BTC-USD", "SPY", "KO", "AAPL", "MSFT"]
START, END = "2014-01-01", "2026-05-27"

print("获取数据并回测（日线）...\n")

rows = []
trade_logs = {}
for sym in SYMBOLS:
    df = load(sym, START, END)
    if len(df) < MA_LEN + 50:
        print(f"  {sym}: 数据不足，跳过")
        continue

    for label, strat in [("RSI_Bias", RSIBiasSignal()), ("B&H", BuyAndHold())]:
        engine = BacktestEngine(strategy=strat, initial_cash=100_000,
                                commission=0.001, slippage=0.0005,
                                periods_per_year=365 if sym == "BTC-USD" else 252)
        m = engine.run(df, mode="vectorized").metrics

        # 交易统计
        if label == "RSI_Bias":
            sig = strat.generate_signals(df)
            entries = (sig == 1) & (sig.shift(1) == 0)
            exits = (sig == 0) & (sig.shift(1) == 1)
            ei, xi = sig.index[entries], sig.index[exits]
            n = min(len(ei), len(xi))
            pnls = [(df.loc[xi[i], "close"] - df.loc[ei[i], "close"]) / df.loc[ei[i], "close"]
                    for i in range(n)]
            wr = (sum(1 for p in pnls if p > 0) / n) if n else np.nan
            open_pos = len(ei) - n  # 当前是否还持仓未平
            trade_logs[sym] = (ei, xi, df)
        else:
            wr, n, open_pos = np.nan, np.nan, np.nan

        rows.append({
            "sym": sym, "策略": label,
            "总收益": m.total_return, "年化": m.annual_return,
            "Sharpe": m.sharpe_ratio, "MaxDD": m.max_drawdown,
            "Calmar": m.calmar_ratio, "胜率": wr,
            "交易数": n, "未平": open_pos,
        })
    print(f"  {sym} done")

res = pd.DataFrame(rows)
strat = res[res["策略"] == "RSI_Bias"].set_index("sym")
bh = res[res["策略"] == "B&H"].set_index("sym")
syms = [s for s in SYMBOLS if s in strat.index]


# ============================================================
# 对比表
# ============================================================

print("\n" + "=" * 104)
print(f"{'':9} {'-- 总收益 --':^20} {'-- 年化 --':^18} {'-- Sharpe --':^18} {'-- MaxDD --':^18} {'交易/胜率':^14}")
print(f"{'标的':9} {'策略':>9} {'B&H':>9} {'策略':>8} {'B&H':>8} {'策略':>8} {'B&H':>8} "
      f"{'策略':>8} {'B&H':>8} {'笔数':>6} {'胜率':>7}")
print("=" * 104)
for s in syms:
    a, b = strat.loc[s], bh.loc[s]
    nt = f"{int(a['交易数'])}" if not np.isnan(a['交易数']) else "-"
    wr = f"{a['胜率']:.0%}" if not np.isnan(a['胜率']) else "-"
    print(f"{s:9} {a['总收益']:>9.1%} {b['总收益']:>9.1%} "
          f"{a['年化']:>8.1%} {b['年化']:>8.1%} "
          f"{a['Sharpe']:>8.2f} {b['Sharpe']:>8.2f} "
          f"{a['MaxDD']:>8.1%} {b['MaxDD']:>8.1%} {nt:>6} {wr:>7}")
print("=" * 104)


# ============================================================
# 交易明细（每个标的）
# ============================================================

for s in syms:
    if s not in trade_logs:
        continue
    ei, xi, df = trade_logs[s]
    print(f"\n--- {s} 交易记录 ---")
    if len(ei) == 0:
        print("  无买入信号")
        continue
    n = min(len(ei), len(xi))
    for i in range(len(ei)):
        ep = df.loc[ei[i], "close"]
        if i < n:
            xp = df.loc[xi[i], "close"]
            ret = (xp - ep) / ep
            days = (xi[i] - ei[i]).days
            print(f"  买 {ei[i].date()} ${ep:>10,.2f}  ->  卖 {xi[i].date()} ${xp:>10,.2f}  "
                  f"{ret:>+7.1%}  ({days}天)")
        else:
            cur = df["close"].iloc[-1]
            ret = (cur - ep) / ep
            print(f"  买 {ei[i].date()} ${ep:>10,.2f}  ->  [持仓中] ${cur:>10,.2f}  {ret:>+7.1%}")
