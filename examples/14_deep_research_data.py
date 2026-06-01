"""深度研究 — 数据采集层(可自动化的那一段)。

打印每只票的「深度财务底座」digest(多期三表 + 分析师/估值),作为
Agent/大模型做第1层定性深研的结构化输入。联网归因 + 综合判断那段需 Agent
在环,见 docs/deep_research.md。

用法:
  # 指定标的
  .venv/bin/python examples/14_deep_research_data.py BSX PODD GIS
  # 不给标的则跑扫描器,对当前周线候选逐只输出 digest
  .venv/bin/python examples/14_deep_research_data.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from qgent.research.financials import fetch_deep_financials, format_digest


def _scanner_candidates() -> list[str]:
    """复用 11_us_buy_scanner 的内部函数,取当前周线候选。"""
    path = Path(__file__).with_name("11_us_buy_scanner.py")
    spec = importlib.util.spec_from_file_location("buy_scanner", path)
    scan = importlib.util.module_from_spec(spec)
    sys.modules["buy_scanner"] = scan
    spec.loader.exec_module(scan)

    syms = scan.get_universe()
    px = scan.get_prices(syms, refresh=False)
    close = px["close"]
    if scan.WEEKLY:
        close = close.resample("W-MON", label="left", closed="left").last()
    valid = close.columns[close.notna().sum() > scan.MA_LEN + 30]
    close = close[valid]
    buy, *_ = scan.compute_signals(close)
    recent = buy.iloc[-scan.RECENT_BARS:].any()
    return recent[recent].index.tolist()


def main():
    tickers = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not tickers:
        print("未指定标的,跑扫描器取当前候选...")
        tickers = _scanner_candidates()
        print(f"候选 {len(tickers)} 只:{', '.join(tickers)}\n")

    for s in tickers:
        f = fetch_deep_financials(s)
        print(format_digest(f))
        print()


if __name__ == "__main__":
    main()
