from __future__ import annotations

import logging
from typing import Optional

import ccxt
import pandas as pd

from qgent.core.constants import Freq
from qgent.data.fetcher.base import BaseFetcher

logger = logging.getLogger(__name__)

_FREQ_MAP = {
    Freq.MIN_1: "1m",
    Freq.MIN_5: "5m",
    Freq.MIN_15: "15m",
    Freq.MIN_30: "30m",
    Freq.HOUR_1: "1h",
    Freq.HOUR_4: "4h",
    Freq.DAILY: "1d",
    Freq.WEEKLY: "1w",
    Freq.MONTHLY: "1M",
}


class CryptoFetcher(BaseFetcher):
    """Fetch cryptocurrency OHLCV data via ccxt (supports 100+ exchanges)."""

    def __init__(self, exchange: str = "binance", **exchange_kwargs):
        exchange_class = getattr(ccxt, exchange, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{exchange}' not found in ccxt")
        self.exchange: ccxt.Exchange = exchange_class(exchange_kwargs)
        self.exchange.load_markets()

    def fetch(
        self,
        symbol: str,
        freq: str | Freq = Freq.DAILY,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
        **kwargs,
    ) -> pd.DataFrame:
        if isinstance(freq, Freq):
            timeframe = _FREQ_MAP.get(freq, freq.value)
        else:
            timeframe = freq

        since = None
        if start:
            since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

        end_ms = None
        if end:
            end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

        all_ohlcv = []
        while True:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=limit
            )
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]

            if end_ms and last_ts >= end_ms:
                break
            if len(ohlcv) < limit:
                break

            since = last_ts + 1
            logger.debug(f"Fetched {len(all_ohlcv)} bars so far for {symbol}")

        if not all_ohlcv:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")

        if end_ms:
            df = df[df.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]

        return self._validate_ohlcv(df)

    def list_symbols(self) -> list[str]:
        return list(self.exchange.markets.keys())
