"""买点扫描器 — 单股详细报告（每只候选一份）

复用 11_us_buy_scanner.py 的内部函数，对当前扫描出的每只候选股，
输出一份完整报告（不再压成一行表格）：

  ① 技术买点      信号日/现价/信号至今涨跌/乖离率+历史分位/RSI 慢线
  ② 基本面体检    health_report 全字段 + 关键原始指标（PE/PB/ROE/利润率/D-E/FCF...）
  ③ 量价质地      信号那根量 / 前12根均量
  ④ 历史胜率      该票历史同类买点 各时段 胜率/平均/中位/样本
  ⑤ 定性研究清单  行业/财报季末/营收趋势/已知结构性风险 + 待答问题

用法（与扫描器一致的开关，会复用同一份缓存）：
  .venv/bin/python examples/12_buy_scanner_detail.py            # 周线（默认）
  .venv/bin/python examples/12_buy_scanner_detail.py --daily    # 日线
  .venv/bin/python examples/12_buy_scanner_detail.py --weeks N  # 回看 N 根
  .venv/bin/python examples/12_buy_scanner_detail.py --only AAPL,MSFT   # 只看指定票
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# 11_xxx 以数字开头不能直接 import，手动按文件路径加载模块
# （导入不触发 main，main 受 __main__ 保护）
import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("11_us_buy_scanner.py")
_spec = importlib.util.spec_from_file_location("buy_scanner", _path)
scan = importlib.util.module_from_spec(_spec)
sys.modules["buy_scanner"] = scan
_spec.loader.exec_module(scan)

from qgent.fundamental import fetch_many, health_report  # noqa: E402


VERDICT_LABEL = {
    "healthy": "✅ 健康打折（被错杀的好公司）",
    "review": "⚠️ 需人工核（拿不准）",
    "avoid": "❌ 疑似陷阱（接飞刀，避开）",
}


def _fmt_pct(x, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.1%}" if plus else f"{x:.1%}"


def _fmt_num(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    for unit, div in (("万亿", 1e12), ("亿", 1e8)):
        if abs(x) >= div:
            return f"{x/div:.1f}{unit}"
    return f"{x:.0f}"


def report_one(s: str, close: pd.Series, buy: pd.Series, bias: pd.Series,
               slow: pd.Series, vol: pd.Series | None, f: dict):
    recent_buy = buy.iloc[-scan.RECENT_BARS:]
    last_buy_date = recent_buy[recent_buy].index[-1]
    sig_price = close.loc[last_buy_date]
    cur_price = close.iloc[-1]
    sig_to_now = cur_price / sig_price - 1

    rpt = health_report(f, price_history=close.dropna())
    ratio = scan.vol_regime(vol, last_buy_date) if vol is not None else None
    bias_now = bias.iloc[-1]
    bias_rank = (bias.dropna() < bias_now).mean()
    v = scan.validate_history(close, buy)

    print("\n" + "=" * 80)
    print(f"  {s}    现价 {cur_price:.2f}    {rpt['detail']}    "
          f"市值 {_fmt_money(f.get('marketCap'))}")
    print("=" * 80)

    # ① 技术买点
    print("① 技术买点")
    print(f"   信号日 {last_buy_date.date()}（{scan.RECENT_BARS} 根回看内最近一次）   "
          f"信号至今 {sig_to_now:+.1%}")
    print(f"   乖离率 {bias_now:+.1f}%   历史分位 {bias_rank:.0%}"
          f"（越低越极端，{bias_rank:.0%}=比历史 {1-bias_rank:.0%} 的时候更超卖）")
    print(f"   RSI 慢线现值 {slow.iloc[-1]:.1f}（买点要求近4根最低 < {scan.BUY_ZONE}）")

    # ② 基本面体检
    print(f"\n② 基本面体检 → {VERDICT_LABEL.get(rpt['verdict'], rpt['verdict'])}")
    hs = f"{rpt['score']:.0f}" if not (isinstance(rpt['score'], float) and np.isnan(rpt['score'])) else "—"
    print(f"   健康分 {hs}/100（≥60 健康）   类型 {rpt['kind']}")
    if rpt["flags"]:
        print(f"   红旗：{'; '.join(rpt['flags'])}")
    else:
        print("   红旗：无")
    print("   关键指标（yfinance，已知会缺失/自相矛盾，仅供交叉参考）：")
    print(f"     估值   PE {_fmt_num(f.get('trailingPE'))}   "
          f"PB {_fmt_num(f.get('priceToBook'))}   "
          f"EV/EBITDA {_fmt_num(f.get('enterpriseToEbitda'))}")
    print(f"     盈利   ROE {_fmt_pct(f.get('returnOnEquity'))}   "
          f"净利率 {_fmt_pct(f.get('profitMargins'))}   "
          f"毛利率 {_fmt_pct(f.get('grossMargins'))}")
    print(f"     成长   营收增速 {_fmt_pct(f.get('revenueGrowth'), True)}   "
          f"盈利增速 {_fmt_pct(f.get('earningsGrowth'), True)}")
    print(f"     安全   D/E {_fmt_num(f.get('debtToEquity'), 0)}   "
          f"FCF {_fmt_money(f.get('freeCashflow'))}   "
          f"现金 {_fmt_money(f.get('totalCash'))} / 债 {_fmt_money(f.get('totalDebt'))}")
    # 营收逐季（最新->旧）
    rev = f.get("quarterly_revenue")
    if rev:
        qs = " ".join(_fmt_money(x) for x in rev[:5])
        print(f"     季度营收(新→旧)  {qs}")

    # ③ 量价
    print(f"\n③ 量价质地   {scan.vol_label(ratio)}"
          + (f"   （信号那根量 / 前12根均量 = {ratio:.2f}）" if ratio else ""))

    # ④ 历史胜率
    print("\n④ 历史同类买点回测（该票自身，样本偏少，仅供参考）")
    print(f"   历史信号 {v['n']} 次")
    for lbl, _ in scan.HORIZONS:
        a = v[lbl]  # (mean, winrate, n) 约定
        if a[2] > 0:
            print(f"     {lbl:>3}后  胜率 {a[1]:.0%}  平均 {a[0]:+.1%}  样本 {a[2]}")

    # ⑤ 定性研究清单
    print("\n⑤ 定性研究清单（第1层，需人工/AI 读财报与新闻确认）")
    industry = f.get("industry")
    risk = scan.INDUSTRY_RISK.get(industry or "", "")
    if rev and len(rev) >= 5 and rev[0] and rev[4]:
        yoy = rev[0] / rev[4] - 1
        print(f"   营收最新季同比 {yoy:+.1%}")
    if risk:
        print(f"   {risk}")
        print("   ❗ 该行业有已知结构性风险，务必确认本次下跌是否与之相关")
    print("   ❓ 这次下跌属于：系统性/板块回调 ? 一次性事件 ? 还是结构性恶化（5年后还重要）?")


def main():
    refresh = "--refresh" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = {x.strip().upper() for x in sys.argv[sys.argv.index("--only") + 1].split(",")}

    syms = scan.get_universe()
    px = scan.get_prices(syms, refresh)
    close, low = px["close"], px["low"]
    vol = px.get("volume")

    if scan.WEEKLY:
        close = close.resample("W-MON", label="left", closed="left").last()
        low = px["low"].resample("W-MON", label="left", closed="left").min()
        if vol is not None:
            vol = vol.resample("W-MON", label="left", closed="left").sum()

    valid = close.columns[close.notna().sum() > scan.MA_LEN + 30]
    close, low = close[valid], low[valid]
    if vol is not None:
        vol = vol[[c for c in valid if c in vol.columns]]

    buy, slow, rsi_low, bias = scan.compute_signals(close)
    recent = buy.iloc[-scan.RECENT_BARS:].any()
    hits = recent[recent].index.tolist()
    if only:
        hits = [h for h in hits if h in only]

    tf = "周线" if scan.WEEKLY else "日线"
    print(f"周期: {tf} | 数据截止 {close.index[-1].date()} | "
          f"最近 {scan.RECENT_BARS} 根内候选 {len(hits)} 只：{', '.join(hits)}")
    if not hits:
        print("无候选。")
        return

    print(f"\n拉取 {len(hits)} 只候选基本面数据...")
    fund = fetch_many(hits)

    # 按健康分降序排报告
    order = sorted(
        hits,
        key=lambda s: (health_report(fund[s], price_history=close[s].dropna())["score"]
                       if not np.isnan(health_report(fund[s], price_history=close[s].dropna())["score"])
                       else -1),
        reverse=True,
    )
    for s in order:
        v = vol[s] if vol is not None and s in vol.columns else None
        report_one(s, close[s], buy[s], bias[s], slow[s], v, fund[s])

    print("\n" + "-" * 80)
    print("免责：基于历史统计与公开财务数据的研究信号，非投资建议。"
          "财务数据滞后季报最多3个月。")


if __name__ == "__main__":
    main()
