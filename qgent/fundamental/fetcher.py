"""Fetch fundamental data (financials + valuation) for stocks via yfinance.

Separate from the OHLCV fetchers in qgent.data — this pulls the `info` snapshot
and quarterly/annual statements needed for the health check, with JSON caching.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache") / "fundamentals"
CACHE_TTL_DAYS = 7  # financials update slowly; reuse cache within a week

# info fields we care about (valuation, profitability, growth, safety, classification)
_INFO_FIELDS = [
    "sector", "industry", "longBusinessSummary", "marketCap", "averageVolume",
    "trailingPE", "priceToBook", "enterpriseToEbitda",
    "profitMargins", "grossMargins", "operatingMargins", "returnOnEquity",
    "revenueGrowth", "earningsGrowth",
    "debtToEquity", "freeCashflow", "totalDebt", "totalCash", "ebitda",
    "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "dividendYield", "payoutRatio",
    "lastFiscalYearEnd", "mostRecentQuarter",
]


def _statement_row(df: pd.DataFrame, *names: str) -> Optional[list[float]]:
    """Return a statement row (newest->oldest) by trying several possible labels."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            vals = df.loc[name].tolist()
            return [None if pd.isna(v) else float(v) for v in vals]
    return None


def fetch_fundamentals(symbol: str, cache_dir: Path = DEFAULT_CACHE_DIR,
                       refresh: bool = False) -> dict:
    """Fetch one symbol's fundamentals, with on-disk JSON cache.

    Returns a dict with: the `_INFO_FIELDS` from `info`, plus
      - quarterly_revenue / quarterly_net_income : list newest->oldest (~6 quarters)
      - annual_revenue / annual_net_income       : list newest->oldest (~4 years)
      - error : str, present only if the fetch failed
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol.replace('/', '_')}.json"

    if cache_file.exists() and not refresh:
        try:
            cached = json.loads(cache_file.read_text())
            age_days = (time.time() - cached.get("_fetched_at", 0)) / 86400
            if age_days < CACHE_TTL_DAYS:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    out: dict = {"symbol": symbol}
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        for k in _INFO_FIELDS:
            out[k] = info.get(k)

        qf = t.quarterly_financials
        out["quarterly_revenue"] = _statement_row(qf, "Total Revenue", "Operating Revenue")
        out["quarterly_net_income"] = _statement_row(qf, "Net Income", "Net Income Common Stockholders")
        out["quarterly_ebit"] = _statement_row(qf, "EBIT", "Operating Income")
        out["quarterly_interest"] = _statement_row(qf, "Interest Expense")

        af = t.financials
        out["annual_revenue"] = _statement_row(af, "Total Revenue", "Operating Revenue")
        out["annual_net_income"] = _statement_row(af, "Net Income", "Net Income Common Stockholders")
    except Exception as e:  # noqa: BLE001 — yfinance raises many ad-hoc errors
        logger.warning(f"fundamental fetch failed for {symbol}: {e}")
        out["error"] = str(e)

    out["_fetched_at"] = time.time()
    try:
        cache_file.write_text(json.dumps(out))
    except OSError:
        pass
    return out


def fetch_many(symbols: list[str], cache_dir: Path = DEFAULT_CACHE_DIR,
               refresh: bool = False) -> dict[str, dict]:
    """Fetch fundamentals for several symbols. One failure does not stop the rest."""
    out = {}
    for sym in symbols:
        out[sym] = fetch_fundamentals(sym, cache_dir=cache_dir, refresh=refresh)
    return out
