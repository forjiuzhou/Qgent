"""Qgent Example: Portfolio optimization across multiple assets."""

import numpy as np
import pandas as pd

from qgent.portfolio.optimizer import PortfolioOptimizer


# --- 1. Generate synthetic multi-asset returns ---
np.random.seed(42)
n_days = 500
dates = pd.bdate_range("2022-01-01", periods=n_days)

assets = {
    "BTC": np.random.normal(0.001, 0.03, n_days),
    "ETH": np.random.normal(0.0008, 0.035, n_days),
    "AAPL": np.random.normal(0.0005, 0.015, n_days),
    "MSFT": np.random.normal(0.0006, 0.014, n_days),
    "GOOGL": np.random.normal(0.0004, 0.016, n_days),
}

# Add some correlations
assets["ETH"] = assets["ETH"] + 0.5 * assets["BTC"]
assets["MSFT"] = assets["MSFT"] + 0.3 * assets["AAPL"]

returns = pd.DataFrame(assets, index=dates)

# --- 2. Compare optimization methods ---
optimizer = PortfolioOptimizer(returns)

print("=== Portfolio Optimization Comparison ===\n")

comparison = optimizer.compare_methods()
print("Weights by method:")
print(comparison.to_string(float_format=lambda x: f"{x:.2%}"))

print("\n--- Individual Allocations ---")
for method in ["equal_weight", "max_sharpe", "risk_parity", "min_variance"]:
    alloc = optimizer.optimize(method)
    print(f"\n{alloc}")

    dollar_alloc = alloc.apply(100_000)
    print(f"  $100k allocation: {', '.join(f'{k}: ${v:,.0f}' for k, v in dollar_alloc.items())}")
