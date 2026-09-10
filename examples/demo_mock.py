"""零依赖离线演示：不需要安装任何第三方库、不需要 API Key 即可跑通。

运行（在项目根目录）：
    python examples/demo_mock.py

演示内容：
1. 单 Agent：规则大脑 + 工具调用循环（计算、查时间）；
2. 多智能体团队：Planner 拆解 → Worker 并行执行 → 自动生成 Markdown 报告并落盘。

提示：ToyLLM 只是"规则大脑"，用于展示流程；真实智能回答请配置 .env 后运行
`python -m pocketagent chat`（DeepSeek / OpenAI / Ollama 均可）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证从任意目录执行都能找到 pocketagent 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pocketagent.agent import Agent, render_event  # noqa: E402
from pocketagent.builtin_tools import default_tools  # noqa: E402
from pocketagent.team import AgentTeam  # noqa: E402
from pocketagent.toy import ToyLLM  # noqa: E402

SEP = "=" * 68


def part1_single_agent() -> None:
    print(SEP)
    print("第 1 部分 · 单 Agent（思考 → 调用工具 → 回复）")
    print(SEP)
    agent = Agent(
        llm=ToyLLM(),
        tools=default_tools(),
        on_event=lambda ev: print(render_event(ev), flush=True),
    )
    agent.run("请帮我计算 12 * 8 - 4 等于多少？")
    print()
    agent.run("现在几点了？")


def part2_multi_agent() -> None:
    print()
    print(SEP)
    print("第 2 部分 · 多智能体研究团队（Planner → Workers → 报告）")
    print(SEP)
    team = AgentTeam(
        llm=ToyLLM(),
        tools=default_tools(),
        on_event=lambda ev: print(render_event(ev), flush=True),
    )
    result = team.run(
        "请计算 21 * 2 与 100 / 4 的结果，并获取当前时间，汇总成一份报告。",
        polish=False,
    )
    print()
    print(f"[完成] 报告共 {len(result.final_report)} 字符")
    print(f"[完成] 报告已保存：{result.report_path}")


if __name__ == "__main__":
    part1_single_agent()
    part2_multi_agent()
    print()
    print(SEP)
    print("全部跑通 ✅")
    print("真实智能体验：复制 .env.example 为 .env 填写 Key，然后运行 `python -m pocketagent chat`")
    print(SEP)
