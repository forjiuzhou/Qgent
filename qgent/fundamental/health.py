"""Fundamental health check — the decision funnel.

Separates "discounted good company" from "collapsing bad company (falling knife)".
Two real analysis paths (no exemptions — cyclicals get a tailored ruler, not a skip):

  Path A (normal: growth/defensive)  — value-collapse veto via revenue/margin trend.
  Path B (cyclical: energy/semis/...) — judge cycle position (revenue & price历史分位),
                                        value via PB not PE, weight balance-sheet resilience.

Both paths share a financial-blowup veto and produce a definitive verdict:
  healthy ✅ / review ⚠️ / avoid ❌.

Thresholds are heuristic defaults, collected here as constants for easy tuning.
yfinance `info` fields are noisy/contradictory (e.g. totalDebt=0 with debtToEquity=217),
so checks cross-validate and degrade gracefully on missing data — never a wrong `avoid`
just because a field is absent.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# --- cyclical classification ---
CYCLICAL_SECTORS = {"Energy", "Basic Materials"}
CYCLICAL_INDUSTRY_KEYWORDS = (
    "semiconductor", "oil & gas", "oil&gas", "mining", "steel",
    "auto", "copper", "aluminum", "coal", "drilling", "metals",
)

# --- scoring / veto thresholds (heuristic) ---
ROE_GOOD = 0.15
NET_MARGIN_GOOD = 0.12
INTEREST_COVER_MIN = 2.0          # EBIT / interest below this = distress
DEBT_EQUITY_EXTREME = 400.0       # debtToEquity (%) above this = extreme leverage
DEBT_EQUITY_HIGH = 200.0          # levered enough that negative FCF is dangerous
HEALTHY_SCORE = 60                # >= this (and no veto) => healthy
REVIEW_SCORE = 40                 # >= this => review, else also review (never auto-avoid on score)


def is_cyclical(f: dict) -> bool:
    sector = (f.get("sector") or "")
    industry = (f.get("industry") or "").lower()
    if sector in CYCLICAL_SECTORS:
        return True
    return any(kw in industry for kw in CYCLICAL_INDUSTRY_KEYWORDS)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _yoy(series: Optional[list], i: int, lag: int = 4) -> Optional[float]:
    """Year-over-year change for quarter i (series is newest->oldest)."""
    if not series or i + lag >= len(series):
        return None
    a, b = series[i], series[i + lag]
    if a is None or b is None or b == 0:
        return None
    return a / b - 1


def _interest_coverage(f: dict) -> Optional[float]:
    ebit = f.get("quarterly_ebit")
    interest = f.get("quarterly_interest")
    if ebit and interest and ebit[0] is not None and interest[0]:
        denom = abs(interest[0])
        if denom > 0:
            return ebit[0] / denom
    return None


def _blowup_veto(f: dict, flags: list) -> bool:
    """Shared financial-blowup veto (both paths). Returns True if it should force avoid."""
    de = f.get("debtToEquity")
    fcf = f.get("freeCashflow")
    cover = _interest_coverage(f)
    sector = f.get("sector") or ""

    veto = False
    if cover is not None and cover < INTEREST_COVER_MIN:
        flags.append(f"利息覆盖{cover:.1f}<2")
        veto = True
    if de is not None and sector != "Financial Services" and de > DEBT_EQUITY_EXTREME:
        flags.append(f"杠杆极高D/E{de:.0f}")
        veto = True
    if fcf is not None and fcf < 0 and de is not None and de > DEBT_EQUITY_HIGH:
        flags.append("FCF为负+高杠杆")
        veto = True
    elif fcf is not None and fcf < 0:
        flags.append("FCF为负")  # yellow flag only, not a veto on its own
    return veto


def _score_profitability(f: dict) -> tuple[float, bool]:
    roe, nm = f.get("returnOnEquity"), f.get("profitMargins")
    if roe is None and nm is None:
        return 0.0, False
    s = 0.0
    if roe is not None:
        s += 12.5 * _clip(roe / ROE_GOOD)
    if nm is not None:
        s += 12.5 * _clip(nm / NET_MARGIN_GOOD)
    return s, True


def _score_safety(f: dict) -> tuple[float, bool]:
    fcf, de = f.get("freeCashflow"), f.get("debtToEquity")
    cover = _interest_coverage(f)
    if fcf is None and de is None and cover is None:
        return 0.0, False
    s = 0.0
    s += 10.0 if (fcf is not None and fcf > 0) else 0.0
    if de is not None:
        s += 8.0 * _clip(1 - de / DEBT_EQUITY_HIGH)   # less leverage = better
    if cover is not None:
        s += 7.0 * _clip(cover / 8.0)                  # comfortable coverage = better
    return s, True


def _price_percentile(price_history: Optional[pd.Series], f: dict) -> Optional[float]:
    """Where the latest price sits in its own range. 0=cheapest ever, 1=dearest.

    Prefers full price history (cycle position); falls back to 52-week range from info.
    """
    if price_history is not None and len(price_history.dropna()) > 20:
        s = price_history.dropna()
        last = s.iloc[-1]
        return float((s < last).mean())
    lo, hi = f.get("fiftyTwoWeekLow"), f.get("fiftyTwoWeekHigh")
    last = price_history.dropna().iloc[-1] if price_history is not None and len(price_history.dropna()) else None
    if lo and hi and hi > lo and last is not None:
        return _clip((last - lo) / (hi - lo))
    return None


# ---------------------------------------------------------------------------
# Path A: normal (growth / defensive) stocks
# ---------------------------------------------------------------------------

def _normal_path(f: dict, price_history, flags: list) -> tuple[float, str]:
    # Veto: value collapse = revenue YoY negative 2 quarters AND margins compressing
    rev = f.get("quarterly_revenue")
    ni = f.get("quarterly_net_income")
    yoy0, yoy1 = _yoy(rev, 0), _yoy(rev, 1)
    ni_yoy0 = _yoy(ni, 0)
    if yoy0 is not None and yoy1 is not None and yoy0 < 0 and yoy1 < 0:
        margin_compress = ni_yoy0 is not None and ni_yoy0 < yoy0  # profit shrinking faster
        if margin_compress:
            flags.append(f"营收连降{yoy0:+.0%}+利润降更快(价值塌缩)")
            return 0.0, "avoid"
        flags.append(f"营收连降{yoy0:+.0%}")

    # Score
    prof, _ = _score_profitability(f)
    safe, _ = _score_safety(f)

    # Growth (25)
    growth = 0.0
    eg = f.get("earningsGrowth")
    if yoy0 is not None:
        growth += 12.5 * _clip(0.5 + yoy0 / 0.2)   # 0% YoY -> half marks
    if eg is not None:
        growth += 12.5 * _clip(0.5 + eg / 0.2)

    # Valuation (25): cheap in own range = good, but penalize value traps
    val = 0.0
    pos = _price_percentile(price_history, f)
    if pos is not None:
        val += 15.0 * (1 - pos)                     # lower in range = cheaper
    pb = f.get("priceToBook")
    if pb is not None:
        val += 10.0 * _clip(1 - (pb - 1) / 9)       # PB~1 good, PB>=10 no marks
    if eg is not None and eg < 0 and yoy0 is not None and yoy0 < 0:
        val *= 0.4                                   # value trap: cheap but shrinking
        flags.append("疑似价值陷阱(低估值+盈利收缩)")

    score = prof + safe + growth + val
    verdict = "healthy" if score >= HEALTHY_SCORE else "review"
    return score, verdict


# ---------------------------------------------------------------------------
# Path B: cyclical stocks (energy / semis / materials ...)
# ---------------------------------------------------------------------------

def _cyclical_path(f: dict, price_history, flags: list) -> tuple[float, str]:
    # Veto: structural decline = revenue at a NEW multi-year low (broke below历史底部),
    # not merely a cyclical dip that revisits prior lows.
    ann = f.get("annual_revenue")
    if ann and len([a for a in ann if a is not None]) >= 3:
        vals = [a for a in ann if a is not None]
        newest, prior = vals[0], vals[1:]
        if newest < min(prior) and newest < 0.8 * pd.Series(prior).median():
            flags.append("营收跌破历史底部(疑似结构性衰败,需核行业替代风险)")
            return 0.0, "avoid"

    # Cycle position (25): near bottom of own price history = opportunity
    pos = _price_percentile(price_history, f)
    cycle = 0.0
    if pos is not None:
        cycle = 25.0 * (1 - pos)
        where = "周期底部" if pos < 0.33 else ("周期中部" if pos < 0.66 else "周期顶部")
        flags.append(f"{where}(价格分位{pos:.0%})")
    else:
        flags.append("周期位置数据不足,已按可得近似")

    # Balance-sheet resilience (35, weighted up — cyclicals must survive the trough)
    safe, _ = _score_safety(f)
    safe *= 35.0 / 25.0

    # Valuation by PB, not PE (20)
    val = 0.0
    pb = f.get("priceToBook")
    if pb is not None:
        val += 20.0 * _clip(1 - (pb - 0.8) / 4)     # PB<=0.8 full, PB>=4.8 none

    # Margin level as cycle thermometer (20)
    margin = 0.0
    om = f.get("operatingMargins")
    if om is not None:
        margin += 20.0 * _clip(0.3 + om / 0.3)

    score = cycle + safe + val + margin
    verdict = "healthy" if score >= HEALTHY_SCORE else "review"
    return score, verdict


# ---------------------------------------------------------------------------

def health_report(f: dict, price_history: Optional[pd.Series] = None) -> dict:
    """Run the decision funnel on one stock's fundamentals.

    Args:
        f: dict from fetch_fundamentals().
        price_history: optional close-price Series (for cycle position / valuation).

    Returns:
        {verdict: healthy|review|avoid, score: 0-100, flags: [str], kind: 常规|周期, detail: str}
    """
    flags: list[str] = []
    if f.get("error") or f.get("sector") is None:
        return {"verdict": "review", "score": float("nan"), "flags": ["数据不足"],
                "kind": "未知", "detail": "无法获取基本面数据"}

    cyclical = is_cyclical(f)
    kind = "周期" if cyclical else "常规"

    blowup = _blowup_veto(f, flags)

    if cyclical:
        score, verdict = _cyclical_path(f, price_history, flags)
    else:
        score, verdict = _normal_path(f, price_history, flags)

    if blowup:
        verdict = "avoid"

    detail = f"{kind} | {f.get('sector')}/{f.get('industry')}"
    return {"verdict": verdict, "score": round(score, 1), "flags": flags,
            "kind": kind, "detail": detail}
