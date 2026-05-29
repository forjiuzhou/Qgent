from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Allocation:
    """Represents a portfolio allocation."""
    weights: dict[str, float]
    method: str = ""

    @property
    def assets(self) -> list[str]:
        return list(self.weights.keys())

    @property
    def weight_array(self) -> np.ndarray:
        return np.array(list(self.weights.values()))

    def normalize(self) -> Allocation:
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total == 0:
            return self
        return Allocation(
            weights={k: v / total for k, v in self.weights.items()},
            method=self.method,
        )

    def filter(self, min_weight: float = 0.01) -> Allocation:
        """Remove assets with weight below threshold and renormalize."""
        filtered = {k: v for k, v in self.weights.items() if v >= min_weight}
        alloc = Allocation(weights=filtered, method=self.method)
        return alloc.normalize()

    def apply(self, portfolio_value: float) -> dict[str, float]:
        """Convert weights to dollar allocations."""
        return {k: v * portfolio_value for k, v in self.weights.items()}

    def to_series(self) -> pd.Series:
        return pd.Series(self.weights, name="weight").sort_values(ascending=False)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {v:.2%}" for k, v in sorted(self.weights.items(), key=lambda x: -x[1]))
        return f"Allocation({self.method}: {items})"
