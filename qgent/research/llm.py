"""OpenAI 兼容的大模型客户端（requests 实现，零额外依赖）。

为什么不用 openai SDK：兼容端点五花八门（国内中转/DeepSeek/Kimi/通义…），
直接打 /chat/completions 最稳，也省一个重依赖。

配置走环境变量（你的 key 不写进代码/仓库）：
  QGENT_LLM_API_KEY   必填（也回退读 OPENAI_API_KEY）
  QGENT_LLM_BASE_URL  默认 https://api.openai.com/v1（回退 OPENAI_BASE_URL）
  QGENT_LLM_MODEL     默认 gpt-4o-mini（回退 OPENAI_MODEL）
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def llm_available() -> bool:
    """是否配了 key —— 没配则上层跳过定性层、优雅降级。"""
    return bool(_env("QGENT_LLM_API_KEY", "OPENAI_API_KEY"))


class LLMClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 90.0):
        self.base_url = (base_url or _env("QGENT_LLM_BASE_URL", "OPENAI_BASE_URL",
                                           default="https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or _env("QGENT_LLM_API_KEY", "OPENAI_API_KEY")
        self.model = model or _env("QGENT_LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")
        self.timeout = timeout

    def chat(self, system: str, user: str, temperature: float = 0.2,
             max_tokens: int = 1500) -> str:
        """单轮对话，返回 assistant 文本。失败抛 RuntimeError（上层捕获后降级）。"""
        if not self.api_key:
            raise RuntimeError("未配置 LLM API key（QGENT_LLM_API_KEY / OPENAI_API_KEY）")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"LLM 请求失败: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(f"LLM 返回 {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"LLM 响应解析失败: {e}; body={resp.text[:300]}") from e
