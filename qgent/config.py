from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QgentConfig:
    """Global configuration for Qgent."""

    data_dir: Path = field(default_factory=lambda: Path("data"))
    default_initial_cash: float = 100_000.0
    default_commission: float = 0.001
    default_slippage: float = 0.0005
    crypto_periods_per_year: int = 365
    stock_periods_per_year: int = 252
    log_level: str = "INFO"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)


# Singleton config instance
config = QgentConfig()
