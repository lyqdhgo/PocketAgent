"""ToyLLM：规则型"大脑"（离线演示 / 自检用，无真实智能）。

用途：
- 没有 API Key 时，也能完整体验「思考 → 调用工具 → 观察结果 → 回复」的 Agent 循环；
- 让多智能体编排（Planner → Workers → 报告）在没有网络的情况下跑通全流程。

它不是生产模型：真实使用时请配置 DeepSeek / OpenAI / Ollama（见 README）。
"""

from __future__ import annotations

from typing import Optional

from .builtin_tools import extract_expressions
from .llm import BaseLLM
from .models import Message, ToolCall, ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER

_TIME_WORDS = ("时间", "几点", "日期", "星期", "几号", "today", "now", "time", "date", "clock")


class ToyLLM(BaseLLM):
    """能识别简单的数学题与时间问题，并驱动对应工具。"""

    def chat(
        self,
        messages: list,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Message:
        tool_names = {t.name for t in (tools or [])} if tools else set()

        last_user = ""
        last_tool_content = None
        for m in messages:
            if m.role == ROLE_USER and m.content:
                last_user = m.content
            if m.role == ROLE_TOOL:
                last_tool_content = m.content

        # 场景 1：作为 Planner —— 要求调用 submit_plan 提交拆解计划
        if "submit_plan" in tool_names:
            return Message.assistant(
                tool_calls=[
                    ToolCall(id="call_plan", name="submit_plan", arguments={"tasks": self._plan_tasks(last_user)})
                ]
            )

        # 场景 2：上一步工具已成功返回结果 → 收尾总结
        if last_tool_content is not None and "错误" not in last_tool_content:
            return Message.assistant(
                f"已完成。根据工具结果：{last_tool_content.strip()[:300]}"
            )

        # 场景 3：需要先调用工具
        if last_tool_content is None:
            if "calculator" in tool_names:
                exprs = extract_expressions(last_user)
                if exprs:
                    return Message.assistant(
                        tool_calls=[
                            ToolCall(id="call_calc", name="calculator", arguments={"expression": exprs[0]})
                        ]
                    )
            if "current_time" in tool_names and any(w in last_user for w in _TIME_WORDS):
                return Message.assistant(
                    tool_calls=[ToolCall(id="call_time", name="current_time", arguments={})]
                )

        # 场景 4：兜底回复
        return Message.assistant(
            "（离线演示）我没有真实推理模型，无法智能回答该问题。"
            "请配置 API Key 后运行 `python -m pocketagent chat` 获得智能回答。"
        )

    def _plan_tasks(self, question: str) -> list[dict]:
        """Toy 版规划：按表达式 / 时间关键词拆成相互独立的子任务。"""
        tasks: list[dict] = []
        for expr in extract_expressions(question):
            tasks.append(
                {"title": f"计算表达式 {expr}", "description": f"请计算并说明结果：{expr}"}
            )
        if any(w in question for w in _TIME_WORDS):
            tasks.append(
                {"title": "获取当前时间", "description": "请通过工具获取当前日期与时间，并说明今天是星期几。"}
            )
        if not tasks:
            tasks.append({"title": "整体处理", "description": question})
        return tasks
