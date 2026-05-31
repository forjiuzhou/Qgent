# Qgent

Quantitative research framework for crypto and US stocks.

## Features

- **Data Layer** — Unified data fetching (crypto via ccxt, US stocks via yfinance), Parquet storage, data cleaning
- **Factor Engine** — Extensible factor framework with 20+ built-in factors (momentum, volatility, volume, technical)
- **Strategy Framework** — Base class supporting both vectorized and event-driven strategies
- **Backtest Engine** — Vectorized (fast) and event-driven (realistic) modes with commission/slippage simulation
- **Portfolio Optimization** — Mean-variance, max Sharpe, min variance, risk parity, equal weight
- **Fundamental Health Check** — Decision funnel that separates "discounted good company" from "falling knife"; tailored cyclical vs. normal analysis paths (see [docs/buy_scanner.md](docs/buy_scanner.md))
- **Visualization** — Interactive Plotly charts and Streamlit dashboard

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
import pandas as pd
from qgent.strategy.base import Strategy
from qgent.backtest.engine import BacktestEngine
from qgent.factor.library import momentum

# Generate synthetic data
np.random.seed(42)
dates = pd.bdate_range("2022-01-01", periods=500, tz="UTC")
close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 500)))
df = pd.DataFrame({
    "open": close * 0.999, "high": close * 1.01,
    "low": close * 0.99, "close": close,
    "volume": np.random.uniform(1e6, 5e6, 500),
}, index=dates)

df["mom"] = momentum.returns(df, period=20)

class SimpleStrategy(Strategy):
    name = "Momentum"
    def generate_signals(self, data):
        signal = pd.Series(0.0, index=data.index)
        signal[data["mom"] > 0] = 1.0
        signal[data["mom"] < -0.02] = -1.0
        return signal

engine = BacktestEngine(strategy=SimpleStrategy(), initial_cash=100_000)
result = engine.run(df)
result.report()
```

## Project Structure

```
qgent/
├── core/           # Types, constants, errors, utilities
├── data/           # Fetchers (crypto, US stock), storage, loader, cleaner
├── factor/         # Factor base, registry, evaluator, built-in library
├── strategy/       # Strategy base, signals, position sizing, risk management
├── backtest/       # Engine, simulated broker, metrics, report
├── portfolio/      # Optimization (MVO, risk parity, etc.), allocation
├── fundamental/    # Fundamental fetcher + health-check decision funnel
├── viz/            # Plotly charts, Streamlit dashboard
└── config.py       # Global configuration
```

## Built-in Factors

| Category | Factors |
|----------|---------|
| Momentum | `momentum`, `rate_of_change`, `rsi`, `macd_signal`, `moving_average_ratio` |
| Volatility | `realized_vol`, `atr`, `atr_pct`, `bollinger_width`, `garman_klass_vol` |
| Volume | `volume_ratio`, `obv`, `vwap_deviation`, `volume_momentum` |
| Technical | `sma`, `ema`, `bollinger_pct`, `stochastic_k`, `stochastic_d`, `williams_r`, `cci` |

## Examples

See `examples/` directory:

- `01_quick_start.py` — Vectorized momentum strategy
- `02_event_driven.py` — Event-driven backtest with risk management
- `03_portfolio_optimization.py` — Multi-asset portfolio optimization
- `10_rsi_bias_signal.py` — Pine RSI strategy port (buy-long / sell-flat), backtest
- `11_us_buy_scanner.py` — S&P 500 oversold buy-point scanner + 3-layer analysis ([docs](docs/buy_scanner.md))

## License

MIT
