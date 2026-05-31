"""美股买点扫描器 — RSI 超卖反转择时（带历史验证）

策略来源：Pine 指标 "RSI策略信号 [基本面不会被破坏的大标的]" 的买入信号 buy1。

哲学：只在"基本面稳的大标的"上做超跌择时。
  -> Universe 直接用 S&P 500 成分股：天然满足"市值大/流动性好/上市久/非仙股"，
     这正是严格过滤想要的结果，且贴合策略前提。

买点定义（与 Pine buy1 一致）：
  rsiValue = RSI(close, 6)
  slow     = SMA(rsiValue, 6)
  fast     = SMA(slow, 2)
  rsi_low  = lowest(slow, 4)
  buy      = (rsi_low < 28) and crossover(slow, fast)

默认周线（贴合"长期持有"低频投资；信号比日线稀疏 7 倍但质量≥日线）。

三层分析（决策漏斗，区分"打折好公司"与"接飞刀烂公司"）：
  第1层 下跌归因（定性，输出研究清单辅助人工）
  第2层 基本面体检 qgent.fundamental.health_report → ✅healthy/⚠️review/❌avoid
        非周期股看营收塌缩；周期股换周期位置+PB+资产负债表（不豁免）
  第3层 量价质地：放量恐慌 vs 缩量阴跌、乖离率历史分位

输出：候选按 ✅/⚠️/❌ 分档 + 历史验证 + 研究清单。详见 docs/buy_scanner.md。

用法：
  .venv/bin/python examples/11_us_buy_scanner.py            # 周线（默认），用缓存
  .venv/bin/python examples/11_us_buy_scanner.py --daily    # 日线
  .venv/bin/python examples/11_us_buy_scanner.py --weeks N  # 回看 N 根（默认4）
  .venv/bin/python examples/11_us_buy_scanner.py --refresh  # 强制重新下载行情
"""

import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from qgent.fundamental import fetch_many, health_report

# ============================================================
# 配置
# ============================================================

START = "2014-01-01"
END = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# 周期：默认周线（更贴合"长期持有大标的"的低频投资哲学）。加 --daily 切回日线。
WEEKLY = "--daily" not in sys.argv

# 买点参数（与 Pine 默认一致）
RSI_LEN, SMA_SLOW, SMA_FAST, BUY_ZONE = 6, 6, 2, 28

if WEEKLY:
    MA_LEN = 52                       # 周线 52 ≈ 1 年（年线）；不用 250(≈5年)
    HORIZONS = [("1月", 4), ("3月", 13), ("半年", 26), ("1年", 52)]
    RECENT_BARS = 4                   # 周线信号稀疏，默认回看最近 4 周
else:
    MA_LEN = 250                      # 日线 250 ≈ 1 年
    HORIZONS = [("1月", 21), ("3月", 63), ("半年", 126), ("1年", 252)]
    RECENT_BARS = 5

# 命令行覆盖回看窗口：--weeks N / --bars N
for _flag in ("--weeks", "--bars"):
    if _flag in sys.argv:
        RECENT_BARS = int(sys.argv[sys.argv.index(_flag) + 1])


# ============================================================
# 数据
# ============================================================

def get_universe() -> list[str]:
    f = CACHE_DIR / "sp500.csv"
    if f.exists():
        return pd.read_csv(f)["symbol"].tolist()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req).read().decode()
    tbl = pd.read_html(io.StringIO(html))[0]
    syms = tbl["Symbol"].str.replace(".", "-", regex=False).tolist()
    pd.DataFrame({"symbol": syms}).to_csv(f, index=False)
    return syms


def get_prices(symbols: list[str], refresh: bool) -> dict[str, pd.DataFrame]:
    """返回 {字段: DataFrame[日期 x 代码]}，字段含 close/high/low/volume。带当日缓存。"""
    cache = CACHE_DIR / "sp500_ohlcv.parquet"
    today = pd.Timestamp.now().normalize()
    if cache.exists() and not refresh:
        meta = pd.read_parquet(cache)
        last = pd.to_datetime(meta.index.get_level_values(0)).max().tz_localize(None)
        if last >= today - pd.Timedelta(days=4):  # 4天内的缓存直接用（含周末）
            print(f"使用缓存 {cache.name}（最新数据 {last.date()}）")
            return _unstack(meta)

    print(f"下载 {len(symbols)} 只 S&P500 日线 ({START} ~ {END}) ...")
    t = time.time()
    raw = yf.download(symbols, start=START, end=END, interval="1d",
                      progress=False, auto_adjust=True, threads=True)
    print(f"  下载耗时 {time.time()-t:.0f}s")
    # 转成长表存盘
    long = raw.stack(level=1, future_stack=True).rename_axis(["date", "symbol"])
    long.to_parquet(cache)
    return _unstack(long)


def _unstack(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for field in ["Close", "High", "Low", "Volume"]:
        if field in long.columns:
            out[field.lower()] = long[field].unstack("symbol")
    return out


# ============================================================
# 指标
# ============================================================

def rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    d = df.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


def compute_signals(close: pd.DataFrame):
    """向量化：对所有股票同时算 buy1 信号 + 辅助指标。"""
    rsi_val = rsi(close, RSI_LEN)
    slow = rsi_val.rolling(SMA_SLOW).mean()
    fast = slow.rolling(SMA_FAST).mean()
    rsi_low = slow.rolling(4).min()
    cross_up = (slow > fast) & (slow.shift(1) <= fast.shift(1))
    buy = (rsi_low < BUY_ZONE) & cross_up
    ma = close.rolling(MA_LEN).mean()
    bias = (close - ma) / ma * 100
    return buy, slow, rsi_low, bias


def validate_history(close: pd.Series, buy: pd.Series) -> dict:
    """该票历史所有 buy 信号触发后的前向收益统计（排除最近 RECENT_BARS 根，避免未来数据不足）。"""
    idx = np.where(buy.values)[0]
    idx = idx[idx < len(close) - 1]  # 至少留 1 根
    res = {"n": len(idx)}
    prices = close.values
    for label, h in HORIZONS:
        rets = []
        for i in idx:
            j = i + h
            if j < len(prices) and not np.isnan(prices[i]) and not np.isnan(prices[j]):
                rets.append(prices[j] / prices[i] - 1)
        if rets:
            res[label] = (np.mean(rets), np.mean([r > 0 for r in rets]), len(rets))
        else:
            res[label] = (np.nan, np.nan, 0)
    return res


# ============================================================
# 第 3 层：量价质地
# ============================================================

def vol_regime(vol_series: pd.Series, signal_date, lookback: int = 12):
    """信号那根K线的成交量 / 之前 lookback 根均量。
    >1.5 放量恐慌(抛售可能出尽,机会)；<0.8 缩量阴跌(没人接,危险)。"""
    if vol_series is None or signal_date not in vol_series.index:
        return None
    loc = vol_series.index.get_loc(signal_date)
    if loc < lookback:
        return None
    avg = vol_series.iloc[loc - lookback:loc].mean()
    if not avg or np.isnan(avg):
        return None
    return vol_series.iloc[loc] / avg


def vol_label(ratio):
    if ratio is None:
        return "—"
    if ratio >= 1.5:
        return f"放量{ratio:.1f}x"
    if ratio <= 0.8:
        return f"缩量{ratio:.1f}x"
    return f"平量{ratio:.1f}x"


# 行业 -> 已知结构性风险提示（第 1 层定性归因的线索，辅助人工）
INDUSTRY_RISK = {
    "Packaged Foods": "⚠️ GLP-1 减肥药对食品需求的结构性冲击",
    "Confectioners": "⚠️ GLP-1 减肥药 + 控糖趋势对糖果需求",
    "Beverages": "⚠️ 含糖饮料受控糖/GLP-1 影响",
    "Tobacco": "⚠️ 烟草需求长期萎缩 + 监管",
    "Department Stores": "⚠️ 线下零售被电商侵蚀",
    "Drug Manufacturers": "⚠️ 专利悬崖 / 集采降价",
    "Oil & Gas E&P": "⚠️ 能源转型长期需求顶 + 油价周期",
    "Semiconductors": "⚠️ 硅周期(库存/产能)波动剧烈",
}


# ============================================================
# 主流程
# ============================================================

def main():
    refresh = "--refresh" in sys.argv
    syms = get_universe()
    px = get_prices(syms, refresh)
    close, low = px["close"], px["low"]
    vol = px.get("volume")

    tf = "周线" if WEEKLY else "日线"
    if WEEKLY:
        # 日线 -> 周线（与例子08一致：W-MON）
        close = close.resample("W-MON", label="left", closed="left").last()
        low = px["low"].resample("W-MON", label="left", closed="left").min()
        if vol is not None:
            vol = vol.resample("W-MON", label="left", closed="left").sum()

    # 丢掉数据太短的票
    valid = close.columns[close.notna().sum() > MA_LEN + 30]
    close, low = close[valid], low[valid]
    if vol is not None:
        vol = vol[[c for c in valid if c in vol.columns]]
    print(f"周期: {tf} | 有效标的: {len(valid)} 只 | 数据截止 {close.index[-1].date()}\n")

    buy, slow, rsi_low, bias = compute_signals(close)

    # 最近 RECENT_BARS 根内触发买点的票
    recent = buy.iloc[-RECENT_BARS:].any()
    hits = recent[recent].index.tolist()

    bar = "周" if WEEKLY else "日"
    print("=" * 100)
    print(f"扫描结果：最近 {RECENT_BARS} 根{tf}K线内触发买点的 S&P500 股票")
    print(f"（买点 = RSI慢线4{bar}最低<{BUY_ZONE} 且 慢线上穿快线，超卖反转 | {tf}）")
    print("=" * 100)

    if not hits:
        print("\n今日无买点。市场没有超卖反转信号——这通常意味着大盘不在恐慌区。")
        return

    # --- 第 2 层：基本面体检（只对候选拉数据，~10 只）---
    print(f"\n拉取 {len(hits)} 只候选的基本面数据做体检...")
    fund = fetch_many(hits)

    # 收集每个候选的指标 + 历史验证 + 基本面 + 量价
    rows = []
    for s in hits:
        recent_buy = buy[s].iloc[-RECENT_BARS:]
        last_buy_date = recent_buy[recent_buy].index[-1]
        sig_price = close[s].loc[last_buy_date]
        cur_price = close[s].iloc[-1]
        v = validate_history(close[s], buy[s])

        # 第 2 层：体检（传完整价格历史，供周期位置/估值用）
        rpt = health_report(fund[s], price_history=close[s].dropna())

        # 第 3 层：量价质地
        ratio = vol_regime(vol[s], last_buy_date) if vol is not None and s in vol.columns else None
        bias_rank = (bias[s].dropna() < bias[s].iloc[-1]).mean()  # 当前乖离在自身历史的分位

        rows.append({
            "代码": s,
            "现价": cur_price,
            "信号日": last_buy_date.date(),
            "信号至今%": (cur_price / sig_price - 1) * 100,
            "乖离率%": bias[s].iloc[-1],
            "乖离分位": bias_rank,
            "量价": vol_label(ratio),
            "健康分": rpt["score"],
            "判定": rpt["verdict"],
            "类型": rpt["kind"],
            "红旗": "; ".join(rpt["flags"][:2]),
            "历史信号数": v["n"],
            **{lbl: v[lbl] for lbl, _ in HORIZONS},
        })

    out = pd.DataFrame(rows)

    # --- 分档输出：✅健康打折 / ⚠️需人工核 / ❌疑似陷阱 ---
    VERDICT_GROUPS = [("healthy", "✅ 健康打折（被错杀的好公司）"),
                      ("review", "⚠️ 需人工核（拿不准，看红旗）"),
                      ("avoid", "❌ 疑似陷阱（接飞刀，避开）")]
    print("（'信号至今'=买点至今涨跌；'乖离分位'=当前乖离在自身历史的百分位,越低越极端；")
    print(" '量价'=信号那周成交量/前12周均量,放量=抛售可能出尽,缩量=无人接）\n")

    for vkey, vtitle in VERDICT_GROUPS:
        grp = out[out["判定"] == vkey].sort_values("乖离率%")
        if grp.empty:
            continue
        print(f"{vtitle}  —  {len(grp)} 只")
        hdr = (f"  {'代码':<6}{'现价':>9}{'信号至今':>9}{'乖离%':>7}{'分位':>6}"
               f"{'健康分':>7}{'类型':>5}{'量价':>9}  {'半年胜/均':>11}{'1年胜/均':>11}  红旗")
        print(hdr)
        for _, r in grp.iterrows():
            half, yr = r["半年"], r["1年"]
            half_s = f"{half[1]:.0%}/{half[0]:+.0%}" if half[2] > 0 else "—"
            yr_s = f"{yr[1]:.0%}/{yr[0]:+.0%}" if yr[2] > 0 else "—"
            hs = f"{r['健康分']:.0f}" if not np.isnan(r['健康分']) else "—"
            print(f"  {r['代码']:<6}{r['现价']:>9.2f}{r['信号至今%']:>+8.1f}%{r['乖离率%']:>7.1f}"
                  f"{r['乖离分位']:>6.0%}{hs:>7}{r['类型']:>5}{r['量价']:>9}  "
                  f"{half_s:>11}{yr_s:>11}  {r['红旗']}")
        print()
    print("注：健康分0-100(≥60健康)；'胜/均'=该票历史同类买点后该窗口上涨概率/平均收益。")

    # 汇总：所有候选的历史信号合并统计（更大样本）
    print("\n" + "=" * 100)
    print("候选股票池 历史买点信号 合并统计（衡量这个买点信号整体有没有 edge）：")
    print("=" * 100)
    for lbl, h in HORIZONS:
        all_rets = []
        for s in hits:
            idx = np.where(buy[s].values)[0]
            idx = idx[idx < len(close) - 1]
            p = close[s].values
            for i in idx:
                j = i + h
                if j < len(p) and not np.isnan(p[i]) and not np.isnan(p[j]):
                    all_rets.append(p[j] / p[i] - 1)
        if all_rets:
            ar = np.array(all_rets)
            print(f"  {lbl:>4}后  样本{len(ar):>5}  胜率{np.mean(ar>0):>6.1%}  "
                  f"平均{np.mean(ar):>+7.2%}  中位{np.median(ar):>+7.2%}")

    # --- 第 1 层：研究清单（定性归因无法自动化，输出人工核查模板）---
    print("\n" + "=" * 100)
    print("研究清单（第1层定性归因，需人工/AI 读财报与新闻确认）：")
    print("=" * 100)
    for s in hits:
        f = fund[s]
        sector, industry = f.get("sector"), f.get("industry")
        rev = f.get("quarterly_revenue")
        if rev and len(rev) >= 5 and rev[0] and rev[4]:
            yoy = rev[0] / rev[4] - 1
            trend = f"营收最新季同比 {yoy:+.1%}"
        else:
            trend = "营收趋势数据不足"
        risk = INDUSTRY_RISK.get(industry or "", "")
        mrq = f.get("mostRecentQuarter")
        mrq_s = pd.to_datetime(mrq, unit="s").date() if mrq else "?"
        print(f"\n  {s}  [{sector}/{industry}]  最近财报季末 {mrq_s}")
        print(f"     {trend}" + (f"   {risk}" if risk else ""))
        print(f"     ❓ 这次下跌属于：系统性/板块回调 ? 一次性事件 ? 还是结构性恶化 ?")
        if risk:
            print(f"     ❗ 该行业有已知结构性风险，务必确认本次下跌是否与之相关")

    print("\n" + "-" * 100)
    print("免责：基于历史统计与公开财务数据的研究信号，非投资建议。财务数据滞后季报最多3个月，")
    print("      第2层体检是安全网，第1层定性归因（读电话会/新闻）才抢在价格之前。")


if __name__ == "__main__":
    main()
