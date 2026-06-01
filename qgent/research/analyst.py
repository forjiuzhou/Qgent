"""第1层定性归因：把量化上下文 + 公司业务 + 近期新闻喂给大模型。

输出**不用 JSON**（实测大模型的中文长文本 JSON 极易因引号/截断而解析失败）。
改用「标签：内容」的纯文本块，逐行容错解析——截断只丢尾部字段，不会整体崩溃。

报告优先回答两个最朴素的问题：这是什么公司？为什么跌？
再客观摆出优势 / 风险 / 待核实，**不输出买/卖/回避的推荐**——最终判断交给使用者。
（确定性的规则层——基本面体检——仍给明确 verdict；二者分工不同。）

模型只基于「我们喂进去的」业务简介 + 量化指标 + 新闻摘要推理，不臆造；
信息不足时必须如实说明，绝不编结论。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .llm import LLMClient
from .news import NewsItem

logger = logging.getLogger(__name__)

_SYSTEM = """你是严谨中立的基本面研究员，服务一套「在基本面稳的大标的上做超跌择时」的策略。
策略想买「因不影响长期价值的利空而被错杀的好公司」，回避「价值正在毁灭的公司（接飞刀）」。
你的任务是第1层定性归因：先讲清这是什么公司、为什么跌，再判断这次下跌价值动没动
（核心问句「这利空5年后还重要吗？」）。

铁律：
- 你**不做买入/卖出/回避的推荐**，最终投资判断由使用者自己下。你只把「公司是干啥的」
  「为什么跌」讲清楚，并客观分列「优势/利多」和「风险/利空」。
- 只能基于用户给你的业务简介、量化指标、新闻摘要推理，不得臆造未提供的数字或事实。
- 信息不足以判断时如实说明，绝不编结论。
- 区分「股价跌」和「价值跌」：系统/板块回调、一次性事件 → 价值多半未动；
  需求被永久替代、护城河破裂、监管永久改变、盈利结构性塌缩 → 价值可能毁灭。"""

# 严格规定输出格式：标签开头、逐项一行；列表项用「- 」。不要 JSON、不要多余文字。
_FORMAT = """请严格按下面的纯文本格式输出（不要用 JSON，不要加代码块标记，不要任何额外说明）。
每个标签独占一行，列表项每行以「- 」开头：

公司简介：用一句话说清这家公司靠什么赚钱（基于给你的业务简介，翻成通顺中文）
为什么跌：用2-4句综合新闻讲清这轮下跌的直接导火索和市场担忧的是什么
下跌归因：从【系统性回调/板块回调/一次性事件/结构性恶化/信息不足】里选一个
5年视角：这利空5年后还重要吗，1-2句客观陈述
优势：
- 支持买入的客观利多（按证据列，1-4条）
风险：
- 反对的客观利空（按证据列，1-4条）
待核实：
- 建议使用者进一步核实的关键问题（1-3条）
证据充分度：从【高/中/低】里选一个（指你掌握信息的充分程度）"""


@dataclass
class QualitativeView:
    business: str = ""        # 公司是干啥的
    why_drop: str = ""        # 为什么跌
    drop_reason: str = "信息不足"
    five_year_view: str = ""
    advantages: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    to_verify: list[str] = field(default_factory=list)
    evidence_strength: str = "低"
    error: Optional[str] = None  # 非空表示这层失败/跳过，上层据此标注降级


def _build_user_prompt(symbol: str, quant: dict, news: list[NewsItem]) -> str:
    lines = [f"标的：{symbol}", ""]
    lines.append("【公司与量化上下文（系统已算好，供你参考，勿改写数字）】")
    for k, v in quant.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("【近期新闻摘要（你判断『为什么跌』与下跌归因的主要外部信息）】")
    if news:
        for n in news:
            date = n.published.date().isoformat() if n.published else "日期未知"
            pub = f"·{n.publisher}" if n.publisher else ""
            lines.append(f"- [{date}{pub}] {n.title}")
            if n.summary:
                lines.append(f"    摘要：{n.summary[:300]}")
    else:
        lines.append("- （未取到近期新闻；『为什么跌』据此说明信息不足，下跌归因倾向『信息不足』）")
    lines.append("")
    lines.append(_FORMAT)
    return "\n".join(lines)


# 标签 → 字段名（单行值）。列表字段单独处理。
_SINGLE = {
    "公司简介": "business",
    "为什么跌": "why_drop",
    "下跌归因": "drop_reason",
    "5年视角": "five_year_view",
    "证据充分度": "evidence_strength",
}
_LIST = {"优势": "advantages", "风险": "risks", "待核实": "to_verify"}
_ALL_LABELS = list(_SINGLE) + list(_LIST)
# 容忍中英文冒号
_LABEL_RE = re.compile(r"^\s*(" + "|".join(map(re.escape, _ALL_LABELS)) + r")\s*[:：]\s*(.*)$")


def _parse_labeled(text: str) -> dict:
    """逐行容错解析「标签：内容」块。截断只丢尾部字段，不整体失败。"""
    out: dict = {}
    cur_list: Optional[str] = None  # 当前正在收集的列表字段名
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _LABEL_RE.match(line)
        if m:
            label, val = m.group(1), m.group(2).strip()
            if label in _SINGLE:
                out[_SINGLE[label]] = val
                cur_list = None
            else:  # 列表标签
                cur_list = _LIST[label]
                out.setdefault(cur_list, [])
                if val:  # 同行就跟了内容也收
                    out[cur_list].append(val.lstrip("-－• ").strip())
            continue
        # 列表项行
        if cur_list is not None:
            item = line.strip()
            if item.startswith(("-", "－", "•", "*")):
                item = item.lstrip("-－•* ").strip()
                if item:
                    out[cur_list].append(item)
            elif item == "":
                continue
            # 非列表项的杂行：忽略
    return out


def analyze_qualitative(symbol: str, quant: dict, news: list[NewsItem],
                        client: Optional[LLMClient] = None) -> QualitativeView:
    """调大模型做第1层定性归因（纯文本格式）。失败返回带 error 的 view，不抛错。"""
    client = client or LLMClient()
    try:
        raw = client.chat(_SYSTEM, _build_user_prompt(symbol, quant, news), max_tokens=2500)
        d = _parse_labeled(raw)
    except Exception as e:
        logger.warning("qualitative analysis for %s failed: %s", symbol, e)
        return QualitativeView(error=str(e))

    if not d:  # 一个标签都没解析到 —— 视为格式异常，降级但保留原文片段
        return QualitativeView(error="模型输出未匹配到任何标签", why_drop=raw.strip()[:400])

    return QualitativeView(
        business=d.get("business", ""),
        why_drop=d.get("why_drop", ""),
        drop_reason=d.get("drop_reason", "信息不足"),
        five_year_view=d.get("five_year_view", ""),
        advantages=d.get("advantages", [])[:4],
        risks=d.get("risks", [])[:4],
        to_verify=d.get("to_verify", [])[:3],
        evidence_strength=d.get("evidence_strength", "低"),
    )
