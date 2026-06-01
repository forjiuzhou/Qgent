"""买点扫描器 — 大模型增强研究报告（每只候选一个 md 文件）

在 11/12 的三层量化分析之上，接入大模型补「第1层定性归因」：
  抓个股近期新闻 → 连同量化上下文喂给大模型 →
  判断下跌是『系统/板块回调·一次性事件·结构性恶化』、回答『5年后还重要吗』→
  落地为综合判定，写成 reports/{TICKER}.md。

大模型配置走环境变量（OpenAI 兼容，不写进代码）：
  export QGENT_LLM_API_KEY=sk-...                 # 必填
  export QGENT_LLM_BASE_URL=https://.../v1        # 可选，默认 OpenAI 官方
  export QGENT_LLM_MODEL=gpt-4o-mini              # 可选
未配置 key 也能跑：量化层照出，定性层标注「未配置」。

用法（与扫描器共用同一份缓存）：
  .venv/bin/python examples/13_buy_scanner_report.py            # 全部候选，周线
  .venv/bin/python examples/13_buy_scanner_report.py --daily
  .venv/bin/python examples/13_buy_scanner_report.py --weeks 8
  .venv/bin/python examples/13_buy_scanner_report.py --only BSX,GIS
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# 11_xxx 数字开头不能直接 import，按文件路径加载（不触发其 main）
_path = Path(__file__).with_name("11_us_buy_scanner.py")
_spec = importlib.util.spec_from_file_location("buy_scanner", _path)
scan = importlib.util.module_from_spec(_spec)
sys.modules["buy_scanner"] = scan
_spec.loader.exec_module(scan)

from qgent.fundamental import fetch_many, health_report  # noqa: E402
from qgent.research import (  # noqa: E402
    analyze_qualitative, fetch_recent_news, llm_available,
)

REPORTS_DIR = Path("reports")
VERDICT_LABEL = {  # 基本面体检（规则层，给明确判定）
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


def build_quant_context(s, close, buy, bias, slow, vol, f):
    """汇总一只票的量化上下文 —— 既写进报告，也作为大模型的输入。"""
    recent_buy = buy.iloc[-scan.RECENT_BARS:]
    last_buy_date = recent_buy[recent_buy].index[-1]
    sig_price = close.loc[last_buy_date]
    cur_price = close.iloc[-1]
    rpt = health_report(f, price_history=close.dropna())
    ratio = scan.vol_regime(vol, last_buy_date) if vol is not None else None
    bias_now = bias.iloc[-1]
    bias_rank = (bias.dropna() < bias_now).mean()
    v = scan.validate_history(close, buy)

    return {
        "rpt": rpt, "v": v, "ratio": ratio,
        "last_buy_date": last_buy_date, "cur_price": cur_price,
        "sig_to_now": cur_price / sig_price - 1,
        "bias_now": bias_now, "bias_rank": bias_rank,
        "slow_now": slow.iloc[-1],
    }


def _llm_context_dict(s, ctx, f):
    """喂给大模型的精简上下文（人读友好，避免它臆造）。"""
    rpt, v = ctx["rpt"], ctx["v"]
    half, yr = v["半年"], v["1年"]
    biz = (f.get("longBusinessSummary") or "").strip()
    return {
        "公司业务(英文原文,供你翻译成中文一句话)": biz[:600] if biz else "（无）",
        "行业": f"{f.get('sector')}/{f.get('industry')}",
        "现价/市值": f"{ctx['cur_price']:.2f} / {_fmt_money(f.get('marketCap'))}",
        "技术信号": f"周线超卖反转，信号日{ctx['last_buy_date'].date()}，信号至今{ctx['sig_to_now']:+.1%}",
        "超卖程度": f"乖离{ctx['bias_now']:+.1f}%，历史分位{ctx['bias_rank']:.0%}（越低越极端）",
        "基本面体检": f"{rpt['verdict']}（健康分{rpt['score']}/100，类型{rpt['kind']}）；红旗：{('；'.join(rpt['flags']) or '无')}",
        "量价": scan.vol_label(ctx["ratio"]),
        "估值": f"PE {_fmt_num(f.get('trailingPE'))} / PB {_fmt_num(f.get('priceToBook'))}",
        "盈利": f"ROE {_fmt_pct(f.get('returnOnEquity'))} / 净利率 {_fmt_pct(f.get('profitMargins'))}",
        "成长": f"营收增速 {_fmt_pct(f.get('revenueGrowth'), True)} / 盈利增速 {_fmt_pct(f.get('earningsGrowth'), True)}",
        "历史同类买点": f"半年胜率{half[1]:.0%}均{half[0]:+.0%}，1年胜率{yr[1]:.0%}均{yr[0]:+.0%}（样本{yr[2]}，偏少）",
    }


def render_markdown(s, ctx, f, news, qview, today):
    rpt, v = ctx["rpt"], ctx["v"]
    L = []
    L.append(f"# {s} 买点研究报告")
    L.append(f"> 生成 {today} · 周线 · {rpt['detail']} · 市值 {_fmt_money(f.get('marketCap'))}\n")

    # 最朴素的两问先答：这是什么公司？为什么跌？
    if not qview.error:
        L.append("## 这是什么公司\n")
        L.append((qview.business or "（未获取到业务简介）") + "\n")
        L.append("## 为什么跌\n")
        L.append((qview.why_drop or "（信息不足）") + "\n")

    # 决策要点 —— 客观摆事实+优势+风险，判断权交给使用者
    L.append("## 🧭 决策要点（判断权在你）\n")
    L.append(f"- **技术面**：周线超卖反转，信号日 {ctx['last_buy_date'].date()}，"
             f"乖离 {ctx['bias_now']:+.1f}%（历史分位 {ctx['bias_rank']:.0%}）")
    L.append(f"- **基本面**：{VERDICT_LABEL.get(rpt['verdict'], rpt['verdict'])}，健康分 {rpt['score']}/100")
    L.append(f"- **下跌归因(AI)**：{qview.drop_reason}（证据充分度 {qview.evidence_strength}）")
    if qview.advantages:
        L.append("- **✅ 优势 / 利多**：")
        for a in qview.advantages:
            L.append(f"  - {a}")
    if qview.risks:
        L.append("- **⚠️ 风险 / 利空**：")
        for r in qview.risks:
            L.append(f"  - {r}")
    if qview.to_verify:
        L.append("- **❓ 待你核实**：")
        for q in qview.to_verify:
            L.append(f"  - {q}")
    L.append("\n_本工具只客观呈现优势与风险，不替你下买/卖判断。_\n")

    # 速览表
    L.append("## 0. 速览\n")
    L.append("| 维度 | 结论 |")
    L.append("| --- | --- |")
    L.append(f"| 技术买点 | 信号日 {ctx['last_buy_date'].date()}，信号至今 {ctx['sig_to_now']:+.1%} |")
    L.append(f"| 超卖程度 | 乖离 {ctx['bias_now']:+.1f}%，历史分位 {ctx['bias_rank']:.0%} |")
    L.append(f"| 基本面体检 | {VERDICT_LABEL.get(rpt['verdict'], rpt['verdict'])}，健康分 {rpt['score']}/100 |")
    L.append(f"| 量价质地 | {scan.vol_label(ctx['ratio'])} |")
    L.append(f"| 下跌归因(AI) | {qview.drop_reason}（证据充分度 {qview.evidence_strength}）|")
    L.append("")

    # 1 技术
    L.append("## 1. 技术买点（第0层入选门槛）\n")
    L.append(f"- 信号日 **{ctx['last_buy_date'].date()}**（{scan.RECENT_BARS} 根回看内最近一次），信号至今 **{ctx['sig_to_now']:+.1%}**")
    L.append(f"- 乖离率 **{ctx['bias_now']:+.1f}%**，历史分位 **{ctx['bias_rank']:.0%}**（比历史 {1-ctx['bias_rank']:.0%} 的时候更超卖）")
    L.append(f"- RSI 慢线现值 {ctx['slow_now']:.1f}（买点要求近4根最低 < {scan.BUY_ZONE}）\n")

    # 2 基本面
    L.append(f"## 2. 基本面体检（第2层）→ {VERDICT_LABEL.get(rpt['verdict'], rpt['verdict'])}\n")
    L.append(f"- 健康分 **{rpt['score']}/100**（≥60 健康），类型 **{rpt['kind']}**")
    L.append(f"- 红旗：{('；'.join(rpt['flags']) or '无')}")
    L.append("- 关键指标（yfinance，已知会缺失/自相矛盾，仅供交叉参考）：\n")
    L.append("| | | |")
    L.append("| --- | --- | --- |")
    L.append(f"| 估值 PE {_fmt_num(f.get('trailingPE'))} | PB {_fmt_num(f.get('priceToBook'))} | EV/EBITDA {_fmt_num(f.get('enterpriseToEbitda'))} |")
    L.append(f"| 盈利 ROE {_fmt_pct(f.get('returnOnEquity'))} | 净利率 {_fmt_pct(f.get('profitMargins'))} | 毛利率 {_fmt_pct(f.get('grossMargins'))} |")
    L.append(f"| 成长 营收 {_fmt_pct(f.get('revenueGrowth'), True)} | 盈利 {_fmt_pct(f.get('earningsGrowth'), True)} | |")
    L.append(f"| 安全 D/E {_fmt_num(f.get('debtToEquity'), 0)} | FCF {_fmt_money(f.get('freeCashflow'))} | 现金 {_fmt_money(f.get('totalCash'))}/债 {_fmt_money(f.get('totalDebt'))} |")
    rev = f.get("quarterly_revenue")
    if rev:
        L.append(f"\n- 季度营收(新→旧)：{' '.join(_fmt_money(x) for x in rev[:5])}")
    L.append("")

    # 3 量价
    ratio = ctx["ratio"]
    L.append("## 3. 量价质地（第3层）\n")
    L.append(f"- {scan.vol_label(ratio)}" + (f"（信号那根量 / 前12根均量 = {ratio:.2f}）" if ratio else ""))
    L.append("- 放量=恐慌抛售可能出尽（机会）；缩量=无人接（危险）\n")

    # 4 历史
    L.append("## 4. 历史同类买点回测（该票自身，样本偏少）\n")
    L.append(f"历史信号 {v['n']} 次：\n")
    L.append("| 持有 | 胜率 | 平均 | 样本 |")
    L.append("| --- | --- | --- | --- |")
    for lbl, _ in scan.HORIZONS:
        a = v[lbl]
        if a[2] > 0:
            L.append(f"| {lbl}后 | {a[1]:.0%} | {a[0]:+.1%} | {a[2]} |")
    L.append("")

    # 5 定性（AI）—— 客观列优势/风险/待核实，不下买卖判断
    L.append("## 5. 定性归因（第1层 · AI 联网研究）\n")
    if qview.error:
        L.append(f"> ⚠️ 定性层未生效：{qview.error}\n")
    else:
        L.append(f"- **下跌归因**：{qview.drop_reason}（证据充分度 {qview.evidence_strength}）")
        L.append(f"- **5年视角**：{qview.five_year_view}")
        L.append("- **✅ 优势 / 利多**：")
        for a in (qview.advantages or ["（未列出）"]):
            L.append(f"  - {a}")
        L.append("- **⚠️ 风险 / 利空**：")
        for r in (qview.risks or ["（未列出）"]):
            L.append(f"  - {r}")
        if qview.to_verify:
            L.append("- **❓ 待你核实**：")
            for q in qview.to_verify:
                L.append(f"  - {q}")
        L.append("")
    L.append("**喂给模型的近期新闻：**\n")
    if news:
        for n in news:
            d = n.published.date().isoformat() if n.published else "日期未知"
            pub = f" · {n.publisher}" if n.publisher else ""
            L.append(f"- [{d}{pub}] {n.title}")
    else:
        L.append("- （未取到近期新闻）")
    L.append("")

    L.append("---")
    L.append("_基于历史统计、公开财务数据与新闻的研究信号，非投资建议。"
             "财务数据滞后季报最多3个月；定性结论由大模型基于上述有限信息生成，需自行复核。_")
    return "\n".join(L)


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
    close = close[valid]
    if vol is not None:
        vol = vol[[c for c in valid if c in vol.columns]]

    buy, slow, rsi_low, bias = scan.compute_signals(close)
    recent = buy.iloc[-scan.RECENT_BARS:].any()
    hits = recent[recent].index.tolist()
    if only:
        hits = [h for h in hits if h in only]

    today = date.today().isoformat()
    print(f"候选 {len(hits)} 只：{', '.join(hits)}")
    if not hits:
        print("无候选。")
        return

    if not llm_available():
        print("⚠️ 未配置 QGENT_LLM_API_KEY —— 量化层照出，定性层将标注未配置。\n")

    print(f"拉取 {len(hits)} 只基本面...")
    fund = fetch_many(hits)
    REPORTS_DIR.mkdir(exist_ok=True)

    for s in hits:
        v = vol[s] if vol is not None and s in vol.columns else None
        ctx = build_quant_context(s, close[s], buy[s], bias[s], slow[s], v, fund[s])
        print(f"  {s}: 抓新闻 + 定性分析...", end="", flush=True)
        news = fetch_recent_news(s)
        qview = analyze_qualitative(s, _llm_context_dict(s, ctx, fund[s]), news)
        md = render_markdown(s, ctx, fund[s], news, qview, today)
        out = REPORTS_DIR / f"{s}.md"
        out.write_text(md, encoding="utf-8")
        if qview.error:
            tag = "定性跳过"
        else:
            tag = f"{qview.drop_reason}·优势{len(qview.advantages)}/风险{len(qview.risks)}"
        print(f" 新闻{len(news)}条，{tag} → {out}")

    print(f"\n完成。报告在 {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
