# Qgent — Claude Code 上手指南

量化研究框架（crypto + 美股）：数据层、因子引擎、策略/回测、组合优化、基本面体检、可视化。
本文件记「不显而易见、会立刻卡住」的约定；项目总览见 `README.md`，最近主要工作见 `docs/buy_scanner.md`。

## 环境（最重要，先读）

- **跑任何代码都用 `.venv/bin/python`**。系统 `python`/`python3` 没有依赖（pandas/yfinance 等），会直接失败。
- venv 是 python3.11（满足 `requires-python>=3.10`）。若 `.venv/` 不存在就重建：
  ```bash
  python3.11 -m venv .venv && .venv/bin/pip install -e .
  ```
- 测试：`.venv/bin/python -m pytest tests/ -q`
- 跑示例：`.venv/bin/python examples/NN_xxx.py`

## 数据

- 行情 via yfinance（美股）/ ccxt（crypto）；基本面 via yfinance。
- 缓存在 `.cache/`（已 gitignore）：S&P500 行情首次下载约 1 分钟，之后秒出；基本面缓存 7 天。
- ⚠️ yfinance `info` 字段会缺失或自相矛盾（如 `totalDebt=0` 却 `debtToEquity=217`）——处理时交叉校验、缺失降级，别盲信单字段。
- RSI 用 Wilder 平滑（`ewm(alpha=1/period)`）对齐 TradingView；要逐位对齐 TV 用 `adjust=False`。

## 代码约定

- 模块风格：`from __future__ import annotations`、类型注解、`logging`（参考 `qgent/data/storage.py`）。
- 包结构对应一个职责一个子目录；可复用逻辑进 `qgent/`，研究脚本进 `examples/`。
- Git：改动先开分支，不要直接提交 `main`；提交信息结尾带 `Co-Authored-By`。

## 最近工作（买点扫描器）

分支 `feat/buy-scanner-fundamental`。S&P500 RSI 超卖买点扫描 + 三层分析（技术买点 → 基本面体检漏斗 → 量价质地），区分「打折好公司」与「接飞刀」。
- 主入口：`examples/11_us_buy_scanner.py`（默认周线，`--daily`/`--weeks N`/`--refresh`）
- 体检模块：`qgent/fundamental/`（`health_report` 决策漏斗，周期股走专门口径而非豁免）
- 设计/方法论/局限：`docs/buy_scanner.md`
