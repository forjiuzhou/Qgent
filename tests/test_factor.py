"""Tests for factor module."""

import numpy as np
import pandas as pd
import pytest

from qgent.factor.registry import FactorRegistry
from qgent.factor.library import momentum, volatility, volume, technical


def make_ohlcv(n=200, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n, tz="UTC")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
    return pd.DataFrame({
        "open": close * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": close,
        "volume": np.random.uniform(1e6, 5e6, n),
    }, index=dates)


class TestFactorRegistry:
    def test_registered_factors(self):
        factors = FactorRegistry.list_factors()
        assert "momentum" in factors
        assert "rsi" in factors
        assert "realized_vol" in factors
        assert "atr" in factors
        assert "volume_ratio" in factors
        assert "sma" in factors

    def test_get_factor(self):
        mom = FactorRegistry.get("momentum")
        assert callable(mom)


class TestMomentumFactors:
    def test_returns(self):
        df = make_ohlcv()
        result = momentum.returns(df, period=20)
        assert len(result) == len(df)
        assert result.iloc[:20].isna().all()

    def test_rsi(self):
        df = make_ohlcv()
        result = momentum.rsi(df, period=14)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_macd(self):
        df = make_ohlcv()
        result = momentum.macd(df)
        assert len(result) == len(df)


class TestVolatilityFactors:
    def test_realized(self):
        df = make_ohlcv()
        result = volatility.realized(df, period=20)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr(self):
        df = make_ohlcv()
        result = volatility.atr(df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_garman_klass(self):
        df = make_ohlcv()
        result = volatility.garman_klass(df)
        assert len(result) == len(df)


class TestVolumeFactors:
    def test_volume_ratio(self):
        df = make_ohlcv()
        result = volume.volume_ratio(df, period=20)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_obv(self):
        df = make_ohlcv()
        result = volume.on_balance_volume(df)
        assert len(result) == len(df)


class TestTechnicalFactors:
    def test_sma(self):
        df = make_ohlcv()
        result = technical.sma(df, period=20)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_bollinger_pct(self):
        df = make_ohlcv()
        result = technical.bollinger_pct(df, period=20)
        assert len(result) == len(df)

    def test_stochastic(self):
        df = make_ohlcv()
        k = technical.stochastic_k(df, period=14)
        valid = k.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
