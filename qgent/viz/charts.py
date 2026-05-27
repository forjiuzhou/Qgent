from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from qgent.backtest.report import BacktestResult
    from qgent.portfolio.allocation import Allocation


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Portfolio Value",
        line=dict(color="#2196F3", width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> go.Figure:
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown",
        fill="tozeroy",
        line=dict(color="#F44336", width=1),
        fillcolor="rgba(244, 67, 54, 0.3)",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=300,
    )
    return fig


def plot_backtest_result(result: BacktestResult) -> go.Figure:
    """Comprehensive backtest result visualization."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price", "Equity Curve", "Drawdown", "Positions"),
        row_heights=[0.3, 0.3, 0.2, 0.2],
    )

    # Price chart
    fig.add_trace(go.Scatter(
        x=result.df.index, y=result.df["close"],
        mode="lines", name="Close Price",
        line=dict(color="#9E9E9E", width=1),
    ), row=1, col=1)

    # Equity curve
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index, y=result.equity_curve.values,
        mode="lines", name="Portfolio Value",
        line=dict(color="#2196F3", width=2),
    ), row=2, col=1)

    # Drawdown
    dd = result.drawdown_series()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown",
        fill="tozeroy",
        line=dict(color="#F44336", width=1),
        fillcolor="rgba(244, 67, 54, 0.3)",
    ), row=3, col=1)

    # Positions
    fig.add_trace(go.Scatter(
        x=result.positions.index, y=result.positions.values,
        mode="lines", name="Position",
        line=dict(color="#4CAF50", width=1),
        fill="tozeroy",
        fillcolor="rgba(76, 175, 80, 0.2)",
    ), row=4, col=1)

    metrics = result.metrics
    title = (
        f"{result.strategy_name} | "
        f"Return: {metrics.total_return:.2%} | "
        f"Sharpe: {metrics.sharpe_ratio:.2f} | "
        f"MaxDD: {metrics.max_drawdown:.2%}"
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=900,
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".1%", row=3, col=1)
    return fig


def plot_factor_analysis(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> go.Figure:
    """Plot factor quantile returns analysis."""
    from qgent.factor.evaluator import FactorEvaluator

    quant_ret = FactorEvaluator.quantile_returns(factor_values, forward_returns, n_quantiles)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Quantile Mean Returns", "Factor vs Returns Scatter"),
    )

    colors = ["#F44336", "#FF9800", "#9E9E9E", "#4CAF50", "#2196F3"]
    n_colors = min(len(colors), len(quant_ret))

    fig.add_trace(go.Bar(
        x=[f"Q{i+1}" for i in range(len(quant_ret))],
        y=quant_ret["mean"].values,
        marker_color=colors[:n_colors],
        name="Mean Return",
    ), row=1, col=1)

    aligned = pd.concat([factor_values, forward_returns], axis=1).dropna()
    if not aligned.empty:
        aligned.columns = ["factor", "returns"]
        sample = aligned.sample(min(1000, len(aligned)))
        fig.add_trace(go.Scatter(
            x=sample["factor"], y=sample["returns"],
            mode="markers", name="Scatter",
            marker=dict(size=3, opacity=0.5, color="#2196F3"),
        ), row=1, col=2)

    fig.update_layout(
        title=f"Factor Analysis: {getattr(factor_values, 'name', 'unnamed')}",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def plot_monthly_heatmap(result: BacktestResult) -> go.Figure:
    """Plot monthly returns heatmap."""
    monthly = result.monthly_returns()

    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=monthly.columns,
        y=monthly.index,
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in monthly.values],
        texttemplate="%{text}",
        textfont={"size": 10},
    ))

    fig.update_layout(
        title="Monthly Returns",
        template="plotly_white",
        height=max(300, len(monthly) * 40),
    )
    return fig


def plot_allocation(allocation: Allocation) -> go.Figure:
    """Plot portfolio allocation as a pie chart."""
    weights = allocation.to_series()
    weights = weights[weights > 0.001]

    fig = go.Figure(data=[go.Pie(
        labels=weights.index,
        values=weights.values,
        textinfo="label+percent",
        textposition="inside",
    )])
    fig.update_layout(
        title=f"Portfolio Allocation ({allocation.method})",
        template="plotly_white",
        height=400,
    )
    return fig
