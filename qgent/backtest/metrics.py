from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Comprehensive performance metrics for a backtest."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    skewness: float
    kurtosis: float

    def __repr__(self) -> str:
        return (
            f"BacktestMetrics(\n"
            f"  Total Return:   {self.total_return:>10.2%}\n"
            f"  Annual Return:  {self.annual_return:>10.2%}\n"
            f"  Sharpe Ratio:   {self.sharpe_ratio:>10.3f}\n"
            f"  Sortino Ratio:  {self.sortino_ratio:>10.3f}\n"
            f"  Max Drawdown:   {self.max_drawdown:>10.2%}\n"
            f"  Max DD Duration:{self.max_drawdown_duration:>10d} bars\n"
            f"  Calmar Ratio:   {self.calmar_ratio:>10.3f}\n"
            f"  Win Rate:       {self.win_rate:>10.2%}\n"
            f"  Profit Factor:  {self.profit_factor:>10.3f}\n"
            f"  Total Trades:   {self.total_trades:>10d}\n"
            f"  Volatility:     {self.volatility:>10.2%}\n"
            f")"
        )

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_return": self.avg_trade_return,
            "volatility": self.volatility,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


def compute_metrics(
    equity_curve: pd.Series,
    trades: list | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """Compute performance metrics from an equity curve.

    Args:
        equity_curve: Series of portfolio values over time.
        trades: Optional list of Trade objects for trade-level stats.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading periods per year (252 for daily, 365 for crypto).
    """
    returns = equity_curve.pct_change().dropna()

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_periods = len(equity_curve)
    annual_return = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1

    vol = returns.std() * np.sqrt(periods_per_year)
    excess_return = returns.mean() - risk_free_rate / periods_per_year
    sharpe = excess_return / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = excess_return / downside_std * np.sqrt(periods_per_year)

    # Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()

    # Max drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    if in_drawdown.any():
        max_dd_duration = in_drawdown.groupby(dd_groups).sum().max()
    else:
        max_dd_duration = 0

    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    # Trade stats
    if trades:
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss
        avg_trade = np.mean(pnls) if pnls else 0.0
        total_trades = len(pnls)
    else:
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
        profit_factor = 0.0
        avg_trade = returns.mean()
        total_trades = 0

    return BacktestMetrics(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=int(max_dd_duration),
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        avg_trade_return=avg_trade,
        volatility=vol,
        skewness=float(returns.skew()),
        kurtosis=float(returns.kurtosis()),
    )
