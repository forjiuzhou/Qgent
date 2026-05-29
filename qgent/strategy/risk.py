from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class RiskRule(ABC):
    """Base class for risk management rules."""

    @abstractmethod
    def check(self, position_pnl_pct: float, portfolio_state: dict) -> bool:
        """Return True if position should be closed."""


class StopLoss(RiskRule):
    """Fixed stop-loss: close position if loss exceeds threshold."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def check(self, position_pnl_pct: float, portfolio_state: dict) -> bool:
        return position_pnl_pct <= -self.threshold


class TakeProfit(RiskRule):
    """Fixed take-profit: close position if gain exceeds threshold."""

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def check(self, position_pnl_pct: float, portfolio_state: dict) -> bool:
        return position_pnl_pct >= self.threshold


class TrailingStop(RiskRule):
    """Trailing stop: close if price drops from peak by threshold."""

    def __init__(self, trail_pct: float = 0.05):
        self.trail_pct = trail_pct
        self._peak_pnl: float = 0.0

    def check(self, position_pnl_pct: float, portfolio_state: dict) -> bool:
        self._peak_pnl = max(self._peak_pnl, position_pnl_pct)
        drawdown_from_peak = self._peak_pnl - position_pnl_pct
        return drawdown_from_peak >= self.trail_pct

    def reset(self):
        self._peak_pnl = 0.0


class MaxDrawdownStop(RiskRule):
    """Stop trading if portfolio drawdown exceeds threshold."""

    def __init__(self, max_drawdown: float = 0.20):
        self.max_drawdown = max_drawdown

    def check(self, position_pnl_pct: float, portfolio_state: dict) -> bool:
        current_dd = portfolio_state.get("drawdown", 0.0)
        return current_dd >= self.max_drawdown


class RiskManager:
    """Manages multiple risk rules for a strategy."""

    def __init__(self, rules: list[RiskRule] | None = None):
        self.rules = rules or []

    def add_rule(self, rule: RiskRule) -> None:
        self.rules.append(rule)

    def should_close(self, position_pnl_pct: float, portfolio_state: dict) -> bool:
        """Return True if any risk rule triggers."""
        return any(rule.check(position_pnl_pct, portfolio_state) for rule in self.rules)

    def reset(self) -> None:
        for rule in self.rules:
            if hasattr(rule, "reset"):
                rule.reset()
