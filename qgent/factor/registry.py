from __future__ import annotations

from typing import Callable

from qgent.factor.base import Factor, FunctionalFactor

_REGISTRY: dict[str, type[Factor] | Callable] = {}


class FactorRegistry:
    """Central registry for all available factors."""

    @staticmethod
    def register(name: str, factor: type[Factor] | Callable) -> None:
        _REGISTRY[name] = factor

    @staticmethod
    def get(name: str) -> type[Factor] | Callable:
        if name not in _REGISTRY:
            raise KeyError(f"Factor '{name}' not found. Available: {list(_REGISTRY.keys())}")
        return _REGISTRY[name]

    @staticmethod
    def list_factors() -> list[str]:
        return list(_REGISTRY.keys())

    @staticmethod
    def all() -> dict[str, type[Factor] | Callable]:
        return dict(_REGISTRY)


def register_factor(name: str | None = None):
    """Decorator to register a function or class as a factor.

    Usage:
        @register_factor("momentum_20d")
        def momentum_20d(df, period=20):
            return df['close'].pct_change(period)

        @register_factor()
        class VolatilityFactor(Factor):
            ...
    """
    def decorator(obj):
        factor_name = name or getattr(obj, "name", None) or obj.__name__
        _REGISTRY[factor_name] = obj
        return obj
    return decorator
