from enum import Enum


class Freq(str, Enum):
    """Supported bar frequencies."""
    TICK = "tick"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    US_STOCK = "us_stock"


class Exchange(str, Enum):
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    COINBASE = "coinbase"
    NYSE = "nyse"
    NASDAQ = "nasdaq"
