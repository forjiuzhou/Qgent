from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorStats:
    """Summary statistics for a factor."""
    name: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_pct: float
    turnover: float
    auto_corr: float

    def __repr__(self) -> str:
        return (
            f"FactorStats({self.name}: IC={self.ic_mean:.4f}, "
            f"IR={self.ir:.4f}, Turnover={self.turnover:.4f})"
        )


class FactorEvaluator:
    """Evaluate factor quality using IC analysis, turnover, and quantile returns."""

    @staticmethod
    def information_coefficient(
        factor: pd.Series | pd.DataFrame,
        forward_returns: pd.Series,
        method: str = "spearman",
    ) -> pd.Series:
        """Compute rolling IC between factor values and forward returns.

        For a single-asset factor (Series), computes rolling rank correlation.
        For cross-sectional (DataFrame with multiple columns), computes
        cross-sectional rank correlation at each timestamp.
        """
        if isinstance(factor, pd.DataFrame):
            ic_values = []
            for ts in factor.index:
                if ts in forward_returns.index if isinstance(forward_returns, pd.Series) else True:
                    row = factor.loc[ts].dropna()
                    if isinstance(forward_returns, pd.DataFrame):
                        ret_row = forward_returns.loc[ts].dropna()
                    else:
                        continue
                    common = row.index.intersection(ret_row.index)
                    if len(common) >= 3:
                        corr = row[common].corr(ret_row[common], method=method)
                        ic_values.append({"timestamp": ts, "ic": corr})
            return pd.DataFrame(ic_values).set_index("timestamp")["ic"] if ic_values else pd.Series(dtype=float)

        aligned = pd.concat([factor, forward_returns], axis=1).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)
        aligned.columns = ["factor", "returns"]
        return aligned["factor"].rolling(60, min_periods=20).corr(aligned["returns"])

    @staticmethod
    def compute_stats(
        factor: pd.Series,
        forward_returns: pd.Series,
        method: str = "spearman",
    ) -> FactorStats:
        """Compute comprehensive factor statistics."""
        ic_series = FactorEvaluator.information_coefficient(factor, forward_returns, method)
        ic_series = ic_series.dropna()

        ic_mean = ic_series.mean() if len(ic_series) > 0 else 0.0
        ic_std = ic_series.std() if len(ic_series) > 0 else 1.0
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        ic_positive_pct = (ic_series > 0).mean() if len(ic_series) > 0 else 0.0

        factor_clean = factor.dropna()
        rank_current = factor_clean.rank(pct=True)
        rank_prev = factor_clean.shift(1).rank(pct=True)
        turnover = (rank_current - rank_prev).abs().mean()
        auto_corr = factor_clean.autocorr(lag=1)

        return FactorStats(
            name=getattr(factor, "name", "unnamed"),
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            ic_positive_pct=ic_positive_pct,
            turnover=turnover if not np.isnan(turnover) else 0.0,
            auto_corr=auto_corr if not np.isnan(auto_corr) else 0.0,
        )

    @staticmethod
    def quantile_returns(
        factor: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """Compute mean returns by factor quantile (for single-asset time-series factors)."""
        aligned = pd.concat([factor, forward_returns], axis=1).dropna()
        aligned.columns = ["factor", "returns"]
        aligned["quantile"] = pd.qcut(aligned["factor"], n_quantiles, labels=False, duplicates="drop")
        result = aligned.groupby("quantile")["returns"].agg(["mean", "std", "count"])
        result["sharpe"] = result["mean"] / result["std"] * np.sqrt(252)
        return result
