"""第1层定性归因的自动化支持：抓新闻 + 调大模型 + 产出研究结论。

买点扫描器的三层漏斗里，第1层「下跌归因」（这利空5年后还重要吗？）
文档明确标注为「不可自动化、需人工读财报与新闻」。本子包用大模型补这一层：
  news.py    抓近期新闻并结构化
  llm.py     OpenAI 兼容客户端（环境变量配置，requests 实现，零额外依赖）
  analyst.py 组 prompt → 调模型 → 解析为结构化定性结论

全部为研究信号，非投资建议。
"""

from __future__ import annotations

from .analyst import QualitativeView, analyze_qualitative
from .financials import DeepFinancials, fetch_deep_financials, format_digest
from .llm import LLMClient, llm_available
from .news import NewsItem, fetch_recent_news

__all__ = [
    "fetch_recent_news",
    "NewsItem",
    "LLMClient",
    "llm_available",
    "analyze_qualitative",
    "QualitativeView",
    "fetch_deep_financials",
    "DeepFinancials",
    "format_digest",
]
