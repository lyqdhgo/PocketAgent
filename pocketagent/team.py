"""多智能体编排：Planner（拆解任务）→ Workers（独立执行）→ Reporter（汇总报告）。

这是本项目面向面试的"亮点"模块之一，体现了经典的 Agent 团队模式：
- Planner 把一个大问题拆成相互独立的子任务（通过结构化 tool_call 提交，规避 JSON 解析脆弱性）；
- 每个 Worker 是一个独立 Agent（各自持有工具与记忆），可串行或并行执行；
- Reporter（可选）把各 Worker 结论润色成一份 Markdown 报告；失败时自动降级为模板汇总，
  保证流程永不中断；
- 报告自动落盘到 outputs/ 目录，方便存档与展示。

所有过程事件通过 on_event 逐条送出，CLI / GUI 可以实时展示每个角色的动作。
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .agent import Agent, make_event
from .llm import BaseLLM
from .models import Message
from .tools import Tool, ToolRegistry

PLANNER_PROMPT = (
    "你是一名严谨的任务规划器。请把用户的问题拆解为若干相互独立、"
    "可由单个工作 Agent 在至多几步工具调用内完成的子任务。\n"
    "要求：1) 覆盖问题的所有要点；2) 子任务之间尽量无依赖；"
    "3) 每个子任务描述自包含、可直接执行。\n"
    "必须通过调用 submit_plan 提交计划，不要输出任何其它内容。"
)

WORKER_PROMPT_TMPL = (
    "你是「{team}」研究团队中的 Worker #{index}，擅长调用工具独立完成任务。\n"
    "当前分配到的子任务：{description}\n"
    "执行要求：需要事实/计算/联网时先调用工具，不要编造；"
    "完成后用与问题相同的语言给出简明结论。"
)

REPORTER_SYSTEM = (
    "你是一名研究报告撰写者。请把若干子任务的结论整理成结构清晰的中文 Markdown 报告，"
    "包含：# 标题、## 摘要、## 各子任务小节、## 总结 等结构。只输出报告正文。"
)


class SubmitPlanTool(Tool):
    """Planner 专用的结构化提交工具（不对外暴露给 Worker）。"""

    name = "submit_plan"
    description = "提交任务拆解计划（内部工具）"
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "子任务标题"},
                        "description": {"type": "string", "description": "子任务详细描述，需自包含"},
                    },
                    "required": ["title", "description"],
                },
            }
        },
        "required": ["tasks"],
    }

    def run(self, **_: Any) -> str:
        return "internal"


@dataclass
class TeamResult:
    """一次团队研究的结果。"""

    question: str
    tasks: list[dict] = field(default_factory=list)
    workers: list[dict] = field(default_factory=list)
    final_report: str = ""
    report_path: Optional[str] = None
    events: list[dict] = field(default_factory=list)
    polished: bool = False


class AgentTeam:
    """研究团队：Planner → Workers → Reporter。"""

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[Iterable[Tool]] = None,
        name: str = "研究团队",
        max_workers: int = 4,
        max_worker_steps: int = 8,
        output_dir: str | Path = "outputs",
        on_event: Optional[Callable[[dict], None]] = None,
        parallel: bool = False,
    ):
        self.llm = llm
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or [])
        self.name = name
        self.max_workers = max(1, max_workers)
        self.max_worker_steps = max(1, max_worker_steps)
        self.output_dir = Path(output_dir)
        self.on_event = on_event
        self.parallel = parallel
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def _emit(self, ev: dict, events: list[dict]) -> None:
        with self._lock:
            events.append(ev)
        if self.on_event:
            self.on_event(ev)

    # ------------------------------------------------------------------
    # 规划
    # ------------------------------------------------------------------

    def _extract_tasks(self, question: str) -> list[dict]:
        """调用 Planner 拆解任务。返回 [{id,title,description}, ...]。"""
        messages = [
            Message.system(PLANNER_PROMPT),
            Message.user(f"用户问题：{question}\n请拆解为子任务并通过 submit_plan 提交。"),
        ]
        reply = self.llm.chat(messages=messages, tools=[SubmitPlanTool()], temperature=0.1)

        tasks: Any = None
        for tc in reply.tool_calls or []:
            if tc.name == "submit_plan":
                tasks = tc.arguments.get("tasks")
                break
        if tasks is None:
            # 兼容模型把 JSON 直接放在 content 里的情况
            try:
                tasks = json.loads(reply.content).get("tasks")
            except Exception:
                tasks = None

        if not isinstance(tasks, list) or not tasks:
            return [{"id": 1, "title": "整体任务", "description": question}]

        cleaned: list[dict] = []
        for i, t in enumerate(tasks, start=1):
            if not isinstance(t, dict):
                continue
            cleaned.append(
                {
                    "id": i,
                    "title": str(t.get("title") or f"子任务 {i}"),
                    "description": str(t.get("description") or question),
                }
            )
        return cleaned or [{"id": 1, "title": "整体任务", "description": question}]

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def _run_one_worker(self, task: dict, index: int, events: list[dict]) -> dict:
        sys_prompt = WORKER_PROMPT_TMPL.format(
            team=self.name, index=index, description=task["description"]
        )
        agent = Agent(
            name=f"worker-{index}",
            llm=self.llm,
            tools=self.registry,
            system_prompt=sys_prompt,
            max_steps=self.max_worker_steps,
        )
        start_ev = make_event(
            "worker_start", name=agent.name, description=task["description"], index=index
        )
        self._emit(start_ev, events)

        res = agent.run(user_input=task["description"])

        self._emit(make_event("worker_done", name=agent.name, index=index), events)
        return {
            "id": index,
            "name": agent.name,
            "title": task["title"],
            "description": task["description"],
            "output": res.final_output,
            "truncated": res.truncated,
            "tool_calls": [e for e in res.steps if e["type"] in ("tool_call", "tool_result")],
        }

    def _run_workers(self, tasks: list[dict], events: list[dict]) -> list[dict]:
        # 控制并发数，避免一次性起太多请求
        worker_specs = tasks[: self.max_workers]
        if not self.parallel or len(worker_specs) <= 1:
            return [self._run_one_worker(t, i, events) for i, t in enumerate(worker_specs, start=1)]

        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(worker_specs)) as pool:
            futures = {
                pool.submit(self._run_one_worker, t, i, events): i
                for i, t in enumerate(worker_specs, start=1)
            }
            ordered = {}
            for fut in futures:
                ordered[futures[fut]] = fut.result()
            results = [ordered[i].result() for i in sorted(ordered)]
        return results

    # ------------------------------------------------------------------
    # 汇总 / 报告
    # ------------------------------------------------------------------

    def _call_reporter(self, question: str, workers: list[dict]) -> Optional[str]:
        transcript = "\n\n".join(
            f"### 子任务 {w['id']}：{w['title']}\n结论：{w['output'] or '（无输出）'}"
            for w in workers
        )
        reply = self.llm.chat(
            messages=[
                Message.system(REPORTER_SYSTEM),
                Message.user(f"研究问题：{question}\n\n各子任务结论：\n{transcript}"),
            ],
            temperature=0.3,
        )
        content = (reply.content or "").strip()
        return content or None

    def _worker_annex(self, workers: list[dict]) -> list[str]:
        """子任务执行过程附录（工具调用明细），便于复核与存档。"""
        lines: list[str] = []
        for w in workers:
            lines.append(f"#### {w['id']}. {w['title']}（Worker {w['name']}）")
            lines.append(f"- 结论：{w['output'] or '（无输出）'}")
            if w["truncated"]:
                lines.append("- ⚠ 该子任务达到最大步数被截断")
            if w["tool_calls"]:
                lines.append("- 工具调用过程：")
                for e in w["tool_calls"]:
                    if e["type"] == "tool_call":
                        lines.append(f"  - 调用 {e['name']}({e.get('arguments', '')})")
                    else:
                        lines.append(f"    → {e['content']}")
        return lines

    def _assemble_md(self, question: str, tasks: list[dict], workers: list[dict]) -> str:
        """不调用 LLM 的模板报告（离线 / 降级场景）。"""
        parts = [
            f"# 研究报告：{question}",
            f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S} · 模式：模板汇总（未调用 LLM 润色）",
            "",
            "## 任务拆解",
        ]
        for t in tasks:
            parts.append(f"{t['id']}. **{t['title']}**：{t['description']}")
        parts += ["", "## 各子任务结论", ""]
        for w in workers:
            parts.append(f"**{w['id']}. {w['title']}**：{w['output'] or '（无输出）'}")
        parts += ["", "## 附录：执行过程", ""]
        parts += self._worker_annex(workers)
        return "\n".join(parts)

    def _save_report(
        self,
        question: str,
        tasks: list[dict],
        workers: list[dict],
        final_report: str,
        polished: bool,
    ) -> Path:
        parts = [
            f"# 研究报告：{question}",
            f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}"
            + (" · 模式：LLM 润色报告" if polished else " · 模式：模板汇总"),
            "",
            "## 任务拆解",
        ]
        for t in tasks:
            parts.append(f"{t['id']}. **{t['title']}**：{t['description']}")
        parts += ["", "## 报告正文", "", final_report, "", "## 附录：子任务执行过程", ""]
        parts += self._worker_annex(workers)
        md = "\n".join(parts) + "\n"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"research_{datetime.now():%Y%m%d_%H%M%S}.md"
        path.write_text(md, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def run(self, question: str, polish: bool = True) -> TeamResult:
        """执行完整的研究流程：规划 → 执行 → 汇总 → 落盘。"""
        events: list[dict] = []
        emit = lambda ev: self._emit(ev, events)  # noqa: E731

        emit(make_event("notice", content=f"团队「{self.name}」开始处理问题"))

        # 1) 规划
        tasks = self._extract_tasks(question)
        emit(make_event("plan", task_count=len(tasks)))
        for t in tasks:
            emit(make_event("plan_task", title=t["title"], description=t["description"]))

        # 2) 执行
        workers = self._run_workers(tasks, events)

        # 3) 汇总
        polished = False
        final_report = ""
        if polish:
            try:
                report = self._call_reporter(question, workers)
                if report:
                    final_report = report
                    polished = True
            except Exception as exc:  # 润色失败不阻断流程
                emit(make_event("notice", content=f"报告润色失败，已降级为模板汇总：{exc}"))
        if not final_report:
            final_report = self._assemble_md(question, tasks, workers)

        # 4) 落盘
        path = None
        try:
            path = self._save_report(question, tasks, workers, final_report, polished)
            emit(make_event("notice", content=f"报告已保存：{path}"))
        except OSError as exc:
            emit(make_event("notice", content=f"报告保存失败（不影响结果）：{exc}"))

        result_ev = make_event("final_report", content=final_report)
        emit(result_ev)

        return TeamResult(
            question=question,
            tasks=tasks,
            workers=workers,
            final_report=final_report,
            report_path=str(path) if path else None,
            events=events,
            polished=polished,
        )
