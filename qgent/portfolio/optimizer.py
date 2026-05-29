from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qgent.portfolio.allocation import Allocation


def equal_weight(returns: pd.DataFrame) -> Allocation:
    """Equal-weight allocation across all assets."""
    n = returns.shape[1]
    weights = np.ones(n) / n
    return Allocation(
        weights=dict(zip(returns.columns, weights)),
        method="equal_weight",
    )


def min_variance(
    returns: pd.DataFrame,
    constraints: Optional[dict] = None,
) -> Allocation:
    """Minimum variance portfolio."""
    cov = returns.cov().values
    n = len(returns.columns)
    x0 = np.ones(n) / n

    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        lambda w: w @ cov @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )
    return Allocation(
        weights=dict(zip(returns.columns, result.x)),
        method="min_variance",
    )


def max_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Allocation:
    """Maximum Sharpe ratio portfolio."""
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(returns.columns)
    x0 = np.ones(n) / n

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 0
        return -(port_ret - risk_free_rate) / port_vol

    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return Allocation(
        weights=dict(zip(returns.columns, result.x)),
        method="max_sharpe",
    )


def mean_variance(
    returns: pd.DataFrame,
    target_return: Optional[float] = None,
    risk_aversion: float = 1.0,
) -> Allocation:
    """Mean-variance optimization.

    If target_return is specified, minimizes variance subject to achieving that return.
    Otherwise, maximizes utility = return - (risk_aversion/2) * variance.
    """
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(returns.columns)
    x0 = np.ones(n) / n

    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if target_return is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target_return})
        objective = lambda w: w @ cov @ w
    else:
        objective = lambda w: -(w @ mu - risk_aversion / 2 * w @ cov @ w)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return Allocation(
        weights=dict(zip(returns.columns, result.x)),
        method="mean_variance",
    )


def risk_parity(returns: pd.DataFrame) -> Allocation:
    """Risk parity: each asset contributes equally to portfolio risk."""
    cov = returns.cov().values
    n = len(returns.columns)
    x0 = np.ones(n) / n

    def risk_contribution_error(w):
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e10
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib / np.sqrt(port_var)
        target = np.sqrt(port_var) / n
        return np.sum((risk_contrib - target) ** 2)

    bounds = [(0.01, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(risk_contribution_error, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return Allocation(
        weights=dict(zip(returns.columns, result.x)),
        method="risk_parity",
    )


class PortfolioOptimizer:
    """High-level interface for portfolio optimization."""

    METHODS = {
        "equal_weight": equal_weight,
        "min_variance": min_variance,
        "max_sharpe": max_sharpe,
        "mean_variance": mean_variance,
        "risk_parity": risk_parity,
    }

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def optimize(self, method: str = "max_sharpe", **kwargs) -> Allocation:
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.METHODS.keys())}")
        return self.METHODS[method](self.returns, **kwargs)

    def compare_methods(self, methods: list[str] | None = None) -> pd.DataFrame:
        """Compare allocations from different methods side by side."""
        methods = methods or list(self.METHODS.keys())
        results = {}
        for method in methods:
            try:
                alloc = self.optimize(method)
                results[method] = alloc.weights
            except Exception as e:
                results[method] = {col: float("nan") for col in self.returns.columns}
        return pd.DataFrame(results).T
