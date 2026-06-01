"""深度财务底座:从 yfinance 拉多期三表 + 分析师/估值,汇成可读 digest。

这是深度研究流程里**可自动化**的一段(数据采集);联网归因与综合判断那段
需要 Agent 在环(见 docs/deep_research.md)。digest 既给人读,也作为喂给
大模型/Agent 做定性研究的结构化输入。

yfinance 报表行名在不同标的/版本间会变,取行用多别名兜底;缺失即降级留空,
绝不抛错中断。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _row(df: Optional[pd.DataFrame], *names: str) -> Optional[list[float]]:
    """按别名取报表行,返回列表(yfinance 默认列序:新→旧)。"""
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            return [float(v) if pd.notna(v) else None for v in df.loc[n].values]
    return None


@dataclass
class DeepFinancials:
    symbol: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    price: Optional[float] = None
    business_summary: Optional[str] = None
    fiscal_years: list = field(default_factory=list)      # 年报列日期(新→旧)
    revenue: Optional[list] = None
    operating_income: Optional[list] = None
    net_income: Optional[list] = None
    diluted_eps: Optional[list] = None
    free_cash_flow: Optional[list] = None
    quarterly_revenue: Optional[list] = None
    total_debt: Optional[list] = None
    cash: Optional[list] = None
    equity: Optional[list] = None
    goodwill: Optional[list] = None
    valuation: dict = field(default_factory=dict)         # fwdPE/PEG/margins/growth
    analyst_targets: Optional[dict] = None
    recommendation: Optional[str] = None
    num_analysts: Optional[int] = None
    error: Optional[str] = None


def fetch_deep_financials(symbol: str) -> DeepFinancials:
    """拉一只票的深度财务底座。任何失败返回带 error 的对象,不抛错。"""
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        inc, cf, bs = t.income_stmt, t.cashflow, t.balance_sheet
        qi = t.quarterly_income_stmt
        try:
            targets = t.analyst_price_targets
        except Exception:
            targets = None
    except Exception as e:  # 网络/限流/解析
        logger.warning("fetch_deep_financials(%s) failed: %s", symbol, e)
        return DeepFinancials(symbol=symbol, error=str(e))

    cols = []
    if inc is not None and not inc.empty:
        cols = [c.date().isoformat() for c in inc.columns]

    return DeepFinancials(
        symbol=symbol,
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap"),
        price=info.get("currentPrice"),
        business_summary=info.get("longBusinessSummary"),
        fiscal_years=cols,
        revenue=_row(inc, "Total Revenue", "Operating Revenue"),
        operating_income=_row(inc, "Operating Income", "Total Operating Income As Reported"),
        net_income=_row(inc, "Net Income", "Net Income Common Stockholders"),
        diluted_eps=_row(inc, "Diluted EPS"),
        free_cash_flow=_row(cf, "Free Cash Flow"),
        quarterly_revenue=_row(qi, "Total Revenue", "Operating Revenue"),
        total_debt=_row(bs, "Total Debt"),
        cash=_row(bs, "Cash And Cash Equivalents",
                  "Cash And Cash Equivalents And Short Term Investments"),
        equity=_row(bs, "Stockholders Equity", "Total Equity Gross Minority Interest"),
        goodwill=_row(bs, "Goodwill"),
        valuation={
            "forwardPE": info.get("forwardPE"),
            "trailingPE": info.get("trailingPE"),
            "pegRatio": info.get("pegRatio"),
            "priceToBook": info.get("priceToBook"),
            "grossMargins": info.get("grossMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "freeCashflow": info.get("freeCashflow"),
        },
        analyst_targets=targets if isinstance(targets, dict) else None,
        recommendation=info.get("recommendationKey"),
        num_analysts=info.get("numberOfAnalystOpinions"),
    )


def _b(vals: Optional[list], n: int = 4) -> str:
    """格式化一行数字为 $十亿(取前 n 列,新→旧)。"""
    if not vals:
        return "—"
    out = []
    for v in vals[:n]:
        if v is None:
            out.append("·")
        elif abs(v) >= 1e7:
            out.append(f"{v/1e9:.2f}")
        else:
            out.append(f"{v:.2f}")
    return " ".join(out)


def format_digest(f: DeepFinancials) -> str:
    """汇成人/模型可读的 digest 文本块。"""
    if f.error:
        return f"{f.symbol}: 数据获取失败 — {f.error}"
    mc = f"{f.market_cap/1e9:.0f}B" if f.market_cap else "—"
    val = f.valuation
    L = [
        f"==== {f.symbol} | {f.sector}/{f.industry} | 市值 {mc} | 现价 {f.price} ====",
        f"年报列(新→旧): {f.fiscal_years[:4]}",
        f"营收     {_b(f.revenue)}",
        f"营业利润  {_b(f.operating_income)}",
        f"净利     {_b(f.net_income)}",
        f"稀释EPS   {_b(f.diluted_eps)}",
        f"自由现金流 {_b(f.free_cash_flow)}",
        f"债/现金/权益/商誉  {_b(f.total_debt)} | {_b(f.cash)} | {_b(f.equity)} | {_b(f.goodwill)}",
        f"季营收(新→旧) {_b(f.quarterly_revenue)}",
        f"估值 fwdPE={val.get('forwardPE')} PEG={val.get('pegRatio')} PB={val.get('priceToBook')} "
        f"毛利={val.get('grossMargins')} 营业利润率={val.get('operatingMargins')} "
        f"营收增速={val.get('revenueGrowth')} 盈利增速={val.get('earningsGrowth')}",
        f"分析师目标 {f.analyst_targets} | 评级 {f.recommendation} 分析师{f.num_analysts}",
    ]
    if f.business_summary:
        L.append(f"业务: {f.business_summary[:400]}")
    return "\n".join(L)
