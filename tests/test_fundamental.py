"""Tests for the fundamental health-check decision funnel."""

import numpy as np
import pandas as pd

from qgent.fundamental.health import health_report, is_cyclical


def price_series(trend="up", n=300):
    """Build a price history; 'up' = near highs, 'down' = near lows."""
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    if trend == "up":
        p = np.linspace(50, 150, n)          # latest = highest
    elif trend == "down":
        p = np.linspace(150, 50, n)          # latest = lowest
    else:
        p = np.full(n, 100.0)
    return pd.Series(p, index=idx)


def base_normal(**over):
    f = dict(
        sector="Healthcare", industry="Medical Devices",
        returnOnEquity=0.25, profitMargins=0.18, grossMargins=0.55,
        operatingMargins=0.25, revenueGrowth=0.08, earningsGrowth=0.10,
        debtToEquity=60.0, freeCashflow=2e9, priceToBook=4.0,
        quarterly_revenue=[110, 108, 105, 103, 100, 99],
        quarterly_net_income=[20, 19, 18, 17, 16, 15],
        quarterly_ebit=[25, 24], quarterly_interest=[2, 2],
        annual_revenue=[420, 400, 380, 360],
    )
    f.update(over)
    return f


class TestIsCyclical:
    def test_energy_and_semis_are_cyclical(self):
        assert is_cyclical({"sector": "Energy", "industry": "Oil & Gas Integrated"})
        assert is_cyclical({"sector": "Technology", "industry": "Semiconductors"})

    def test_healthcare_defensive_not_cyclical(self):
        assert not is_cyclical({"sector": "Healthcare", "industry": "Medical Devices"})
        assert not is_cyclical({"sector": "Consumer Defensive", "industry": "Packaged Foods"})


class TestNormalPath:
    def test_healthy_company(self):
        r = health_report(base_normal(), price_history=price_series("down"))
        assert r["kind"] == "常规"
        assert r["verdict"] == "healthy"
        assert r["score"] >= 60

    def test_value_collapse_is_avoid(self):
        # revenue down 2 quarters YoY AND profit shrinking faster -> falling knife
        f = base_normal(
            quarterly_revenue=[88, 90, 95, 98, 100, 102],   # yoy0,yoy1 negative
            quarterly_net_income=[5, 7, 12, 14, 16, 18],     # net falling faster
            earningsGrowth=-0.30, revenueGrowth=-0.12,
        )
        r = health_report(f, price_history=price_series("down"))
        assert r["verdict"] == "avoid"
        assert any("塌缩" in x for x in r["flags"])

    def test_blowup_veto_overrides(self):
        # weak interest coverage -> financial blowup veto -> avoid
        f = base_normal(quarterly_ebit=[3], quarterly_interest=[5], freeCashflow=-1e9,
                        debtToEquity=350.0)
        r = health_report(f, price_history=price_series("down"))
        assert r["verdict"] == "avoid"


class TestCyclicalPath:
    def test_cyclical_at_trough_not_auto_avoid(self):
        # Energy, revenue YoY negative, but price near lows + healthy balance sheet.
        # Must NOT be auto-avoided just for negative YoY, and must get a real verdict.
        f = base_normal(
            sector="Energy", industry="Oil & Gas Integrated",
            quarterly_revenue=[80, 85, 90, 95, 100, 105],   # YoY negative (cyclical dip)
            annual_revenue=[360, 400, 380, 350],            # within historical range
            debtToEquity=40.0, freeCashflow=3e9, priceToBook=1.2,
        )
        r = health_report(f, price_history=price_series("down"))
        assert r["kind"] == "周期"
        assert r["verdict"] in ("healthy", "review")        # real verdict, not avoid
        assert any("周期" in x for x in r["flags"])          # judged cycle position

    def test_cyclical_structural_breakdown_is_avoid(self):
        # revenue at a NEW multi-year low (broke below历史底部) -> structural -> avoid
        f = base_normal(
            sector="Energy", industry="Oil & Gas Integrated",
            annual_revenue=[200, 400, 380, 360],            # newest far below prior min
        )
        r = health_report(f, price_history=price_series("down"))
        assert r["verdict"] == "avoid"
        assert any("结构性" in x for x in r["flags"])


class TestMissingData:
    def test_missing_sector_does_not_crash_or_wrong_avoid(self):
        r = health_report({"symbol": "X"})
        assert r["verdict"] == "review"
        assert r["verdict"] != "avoid"

    def test_error_field_handled(self):
        r = health_report({"symbol": "X", "error": "boom"})
        assert r["verdict"] == "review"
        assert np.isnan(r["score"])
