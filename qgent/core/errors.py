class QgentError(Exception):
    """Base exception for all Qgent errors."""


class DataError(QgentError):
    """Raised when data fetching, loading, or validation fails."""


class StrategyError(QgentError):
    """Raised when strategy logic encounters an error."""


class BacktestError(QgentError):
    """Raised when the backtest engine encounters an error."""


class FactorError(QgentError):
    """Raised when factor computation fails."""
