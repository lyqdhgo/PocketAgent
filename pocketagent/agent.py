"""单 Agent：ReAct 式「思考 → 工具调用 → 观察结果 → 回答」主循环。

工作方式：
1. 把用户输入写入记忆，连同 system prompt 一起发给 LLM；
2. 若模型返回 tool_calls，则逐个执行工具并把结果以 role="tool" 消息回填，
   再次调用模型（模型据此继续思考或给出最终答案）；
3. 直到模型给出不含 tool_calls 的最终回复，或达到 max_steps 上限。

每一步都会产生结构化事件（event），可通过 on_event 回调实时输出到
终端 / GUI，方便演示与调试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from .llm import BaseLLM
from .memory import RollingMemory
from .models import Message, ROLE_SYSTEM
from .tools import Tool, ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人、严谨可靠的 AI Agent。\n"
    "规则：\n"
    "1. 如果问题需要事实检索、计算或联网，请优先调用合适的工具，不要凭空编造数据。\n"
    "2. 工具返回后，根据结果组织最终回答；若工具出错，可换一种方式重试。\n"
    "3. 用与用户相同的语言回答；最终回答简洁、直接、结构化。\n"
    "4. 若无需工具即可回答，直接给出答案。"
)


# ---------------------------------------------------------------------------
# 事件
# ---------------------------------------------------------------------------

def make_event(typ: str, **fields: Any) -> dict:
    """构造一条结构化事件。"""
    return {"type": typ, **fields}


def render_event(ev: dict) -> str:
    """把事件渲染为人类可读文本（CLI / GUI 日志共用）。"""
    t = ev["type"]
    prefix = f"[{ev.get('worker')}] " if ev.get("worker") else ""
    if t == "user":
        return f"\n[用户] {ev.get('content', '')}"
    if t == "assistant":
        content = ev.get("content") or ""
        return prefix + (f"[思考] {content}" if content else "[思考] (决定调用工具)")
    if t == "tool_call":
        return prefix + f"  🔧 调用工具 {ev.get('name')}({ev.get('arguments', '')})"
    if t == "tool_result":
        return prefix + f"  📥 工具返回：{ev.get('content', '')}"
    if t == "final":
        return prefix + f"\n[回复] {ev.get('content', '')}"
    if t == "notice":
        return f"[提示] {ev.get('content', '')}"
    if t == "error":
        return f"[错误] {ev.get('content', '')}"
    if t == "plan":
        return f"\n[规划] 拆解为 {ev.get('task_count')} 个子任务："
    if t == "plan_task":
        return f"  · {ev.get('title')}：{ev.get('description', '')}"
    if t == "worker_start":
        return f"\n[Worker {ev.get('name')}] 开始执行：{ev.get('description', '')}"
    if t == "worker_done":
        return f"[Worker {ev.get('name')}] 完成"
    if t == "final_report":
        return f"\n[最终报告]\n{ev.get('content', '')}"
    return str(ev)


# ---------------------------------------------------------------------------
# Agent 与运行结果
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """一次 run 的结果。"""

    final_output: str
    steps: list[dict] = field(default_factory=list)
    truncated: bool = False
    steps_used: int = 0
    error: Optional[str] = None

    def trace_text(self) -> str:
        return "\n".join(render_event(e) for e in self.steps)


class Agent:
    """一个可独立完成任务的 Agent（含记忆、工具、最大步数）。"""

    def __init__(
        self,
        name: str = "agent",
        llm: Optional[BaseLLM] = None,
        tools: Optional[Iterable[Tool]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory: Optional[RollingMemory] = None,
        max_steps: int = 8,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or [])
        self.max_steps = max(1, max_steps)
        self.memory = memory if memory is not None else RollingMemory()
        self.on_event = on_event
        if llm is None:
            raise ValueError("Agent 需要提供 llm（真实模型或 Mock/Toy）")
        self._ensure_system()

    # -- 内部工具 --------------------------------------------------------

    def _ensure_system(self) -> None:
        if self.system_prompt and not any(m.role == ROLE_SYSTEM for m in self.memory.messages):
            self.memory.add(Message.system(self.system_prompt))

    def _emit(self, ev: dict) -> None:
        if self.on_event:
            self.on_event(ev)

    # -- 主入口 ----------------------------------------------------------

    def run(self, user_input: str, reset_memory: bool = True) -> AgentResult:
        """执行一轮完整任务：从用户输入到最终回答（或达到步数上限）。"""
        if reset_memory:
            self.memory.reset()
            self._ensure_system()

        events: list[dict] = []
        emit = self._emit
        record = events.append

        self.memory.add(Message.user(user_input))
        user_ev = make_event("user", content=user_input)
        record(user_ev)
        emit(user_ev)

        steps_used = 0
        truncated = False
        error: Optional[str] = None
        last_output = ""

        while steps_used < self.max_steps:
            steps_used += 1
            try:
                msg = self.llm.chat(messages=self.memory.messages, tools=self.registry.as_list())
            except Exception as exc:  # 模型/网络异常不中断程序
                error = str(exc)
                notice = make_event("error", content=f"调用模型失败：{error}")
                record(notice)
                emit(notice)
                last_output = f"（调用模型失败：{error}）"
                break

            self.memory.add(msg)
            thought_ev = make_event("assistant", content=msg.content)
            record(thought_ev)
            emit(thought_ev)

            if not msg.tool_calls:
                last_output = msg.content
                final_ev = make_event("final", content=msg.content)
                record(final_ev)
                emit(final_ev)
                break

            for tc in msg.tool_calls:
                tool_ev = make_event("tool_call", name=tc.name, arguments=str(tc.arguments)[:200])
                record(tool_ev)
                emit(tool_ev)

                tool = self.registry.get(tc.name)
                if tool is None:
                    result = f"[错误] 未知工具：{tc.name}"
                else:
                    try:
                        result = tool.run(**tc.arguments)
                    except Exception as exc:  # 工具自身异常也要喂回给模型
                        result = f"[错误] 工具执行异常：{type(exc).__name__}: {exc}"
                result_ev = make_event("tool_result", name=tc.name, content=result[:500])
                record(result_ev)
                emit(result_ev)

                self.memory.add(Message.tool_result(tc.id, result))
        else:
            # while 正常结束（未 break）说明步数耗尽
            truncated = True
            notice_text = f"已达最大迭代步数 {self.max_steps}，任务可能未完全完成。"
            last_output = notice_text
            notice_ev = make_event("notice", content=notice_text)
            record(notice_ev)
            emit(notice_ev)

        return AgentResult(
            final_output=last_output,
            steps=events,
            truncated=truncated,
            steps_used=steps_used,
            error=error,
        )
