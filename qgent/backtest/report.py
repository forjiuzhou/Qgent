from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from qgent.backtest.metrics import BacktestMetrics
from qgent.core.types import Trade


@dataclass
class BacktestResult:
    """Container for backtest results with analysis and plotting utilities."""

    metrics: BacktestMetrics
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    signals: pd.Series
    trades: list[Trade]
    df: pd.DataFrame
    strategy_name: str = ""

    def report(self) -> str:
        """Print a formatted performance report."""
        header = f"{'=' * 50}"
        title = f"  Backtest Report: {self.strategy_name}"
        output = f"\n{header}\n{title}\n{header}\n{self.metrics}\n{header}\n"
        print(output)
        return output

    def trade_log(self) -> pd.DataFrame:
        """Return trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append({
                "symbol": t.symbol,
                "side": t.side.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "commission": t.commission,
            })
        return pd.DataFrame(records)

    def drawdown_series(self) -> pd.Series:
        """Compute drawdown series from equity curve."""
        cummax = self.equity_curve.cummax()
        return (self.equity_curve - cummax) / cummax

    def monthly_returns(self) -> pd.DataFrame:
        """Compute monthly returns table."""
        monthly = self.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame(monthly)
        monthly_df.columns = ["return"]
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        pivot = monthly_df.pivot_table(values="return", index="year", columns="month")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:len(pivot.columns)]
        return pivot

    def plot(self, show: bool = True):
        """Plot equity curve and drawdown using plotly."""
        from qgent.viz.charts import plot_backtest_result
        fig = plot_backtest_result(self)
        if show:
            fig.show()
        return fig
