"""多智能体编排单元测试：Planner 拆解 → Workers 执行 → Reporter 汇总 → 报告落盘。"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import unittest
from pathlib import Path

from pocketagent.builtin_tools import CalculatorTool, CurrentTimeTool
from pocketagent.llm import MockLLM
from pocketagent.models import Message, ToolCall
from pocketagent.team import AgentTeam


@contextlib.contextmanager
def _tmpdir():
    """临时目录上下文。rmtree 忽略错误：兼容只读/受限环境，测试结果不受影响。"""
    d = tempfile.mkdtemp(prefix="pocketagent_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

PLAN_ARGS = {
    "tasks": [
        {"title": "计算", "description": "请计算并说明结果：21 * 2"},
        {"title": "时间", "description": "请通过工具获取当前时间"},
    ]
}


class TestAgentTeam(unittest.TestCase):
    def _scripts(self) -> list[Message]:
        return [
            # 1) Planner：通过 submit_plan 提交 2 个子任务
            Message.assistant(
                tool_calls=[ToolCall(id="p1", name="submit_plan", arguments=PLAN_ARGS)]
            ),
            # 2) Worker1：先调用计算器
            Message.assistant(
                tool_calls=[
                    ToolCall(id="w1", name="calculator", arguments={"expression": "21 * 2"})
                ]
            ),
            # 3) Worker1：给出结论
            Message.assistant("结果是 42。"),
            # 4) Worker2：直接回答（未调用工具）
            Message.assistant("当前时间为 2026-09-08 12:00:00，星期二。"),
            # 5) Reporter：汇总
            Message.assistant("# 最终报告\n\n两个子任务均已完成，详见上文。"),
        ]

    def test_full_pipeline(self) -> None:
        with _tmpdir() as tmp:
            llm = MockLLM(self._scripts())
            team = AgentTeam(
                llm=llm,
                tools=[CalculatorTool(), CurrentTimeTool()],
                output_dir=tmp,
            )
            result = team.run("请计算 21*2 并获取当前时间", polish=True)

            self.assertEqual(len(result.tasks), 2)
            self.assertEqual(len(result.workers), 2)
            self.assertEqual(result.workers[0]["output"], "结果是 42。")
            self.assertIn("最终报告", result.final_report)
            self.assertTrue(result.polished)
            self.assertIsNotNone(result.report_path)
            self.assertTrue(Path(result.report_path).exists())

            types = [e["type"] for e in result.events]
            for expected in ("notice", "plan", "plan_task", "worker_start", "worker_done", "final_report"):
                self.assertIn(expected, types)

    def test_plan_fallback_from_content_json(self) -> None:
        scripts = [
            Message.assistant('{"tasks":[{"title":"整体","description":"处理问题"}]}'),
            Message.assistant("处理完成。"),
            Message.assistant("报告完成。"),
        ]
        with _tmpdir() as tmp:
            team = AgentTeam(llm=MockLLM(scripts), tools=[CalculatorTool()], output_dir=tmp)
            result = team.run("处理问题", polish=True)
        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(result.final_report, "报告完成。")

    def test_offline_template_report(self) -> None:
        """polish=False 时不调用 Reporter，使用模板汇总（离线演示场景）。"""
        scripts = [
            Message.assistant(
                tool_calls=[
                    ToolCall(
                        id="p1",
                        name="submit_plan",
                        arguments={
                            "tasks": [{"title": "计算", "description": "请计算 6 * 7"}]
                        },
                    )
                ]
            ),
            Message.assistant(
                tool_calls=[
                    ToolCall(id="w1", name="calculator", arguments={"expression": "6 * 7"})
                ]
            ),
            Message.assistant("结果是 42。"),
        ]
        with _tmpdir() as tmp:
            team = AgentTeam(llm=MockLLM(scripts), tools=[CalculatorTool()], output_dir=tmp)
            result = team.run("算一下 6*7", polish=False)
        self.assertFalse(result.polished)
        self.assertIn("42", result.final_report)
        self.assertIn("附录", result.final_report)


if __name__ == "__main__":
    unittest.main()
