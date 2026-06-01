"""抓取个股近期新闻（yfinance），结构化成喂给大模型的素材。

yfinance `Ticker.news` 的结构在不同版本间变过：
  - 新版：每条嵌在 item["content"] 里（title/summary/pubDate/provider）
  - 旧版：扁平（title/summary/publisher/providerPublishTime）
两种都兼容；字段缺失就降级，绝不抛错中断报告生成。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    title: str
    summary: str
    publisher: str
    published: Optional[datetime]  # UTC

    def age_days(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.published is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (now - self.published).total_seconds() / 86400


def _parse_time(raw) -> Optional[datetime]:
    """pubDate 字符串(ISO) 或 providerPublishTime(epoch 秒) → aware datetime。"""
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        # ISO 字符串，可能带 Z
        return pd.to_datetime(raw, utc=True).to_pydatetime()
    except Exception:
        return None


def _extract(item: dict) -> Optional[NewsItem]:
    """从一条原始 news dict 提取，兼容新旧两种结构。"""
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = (c.get("title") or "").strip()
    if not title:
        return None
    summary = (c.get("summary") or c.get("description") or "").strip()

    # publisher: 新版 content.provider.displayName / 旧版 publisher
    prov = c.get("provider")
    if isinstance(prov, dict):
        publisher = prov.get("displayName") or prov.get("name") or ""
    else:
        publisher = c.get("publisher") or item.get("publisher") or ""

    published = _parse_time(
        c.get("pubDate") or c.get("displayTime") or item.get("providerPublishTime")
    )
    return NewsItem(title=title, summary=summary, publisher=publisher, published=published)


def fetch_recent_news(symbol: str, max_age_days: float = 45,
                      limit: int = 8) -> list[NewsItem]:
    """返回该股最近 max_age_days 天内、至多 limit 条新闻，按时间倒序。

    抓取失败返回空列表（不抛错），让上层报告照常生成、只标注「无新闻」。
    """
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception as e:  # 网络/限流/结构异常都不该中断整批报告
        logger.warning("fetch news for %s failed: %s", symbol, e)
        return []

    items: list[NewsItem] = []
    for it in raw:
        ni = _extract(it)
        if ni is None:
            continue
        age = ni.age_days()
        if age is not None and age > max_age_days:
            continue
        items.append(ni)

    # 有时间的排前面（新→旧），无时间的垫后
    items.sort(key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc),
               reverse=True)
    return items[:limit]
