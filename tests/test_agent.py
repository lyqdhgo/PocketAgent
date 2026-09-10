"""Agent 主循环单元测试：工具调用闭环、未知工具容错、工具异常容错、步数上限。"""

from __future__ import annotations

import unittest

from pocketagent.agent import Agent
from pocketagent.builtin_tools import CalculatorTool
from pocketagent.llm import MockLLM
from pocketagent.models import Message, ToolCall
from pocketagent.tools import Tool


class TestAgent(unittest.TestCase):
    def test_tool_loop_and_final(self) -> None:
        llm = MockLLM(
            [
                Message.assistant(
                    tool_calls=[
                        ToolCall(id="c1", name="calculator", arguments={"expression": "1 + 2"})
                    ]
                ),
                Message.assistant("结果是 3。"),
            ]
        )
        agent = Agent(llm=llm, tools=[CalculatorTool()], max_steps=5)
        res = agent.run("计算 1+2")
        self.assertEqual(res.final_output, "结果是 3。")
        self.assertFalse(res.truncated)
        # 工具确实被执行并把结果喂回了模型
        tool_results = [e for e in res.steps if e["type"] == "tool_result"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("3", tool_results[0]["content"])
        # 记忆里应包含 role=tool 的消息
        tool_msgs = [m for m in agent.memory.messages if m.role == "tool"]
        self.assertEqual(len(tool_msgs), 1)

    def test_unknown_tool_survives(self) -> None:
        llm = MockLLM(
            [
                Message.assistant(tool_calls=[ToolCall(id="c1", name="not_exist", arguments={})]),
                Message.assistant("抱歉，我无法使用该工具。"),
            ]
        )
        res = Agent(llm=llm, tools=[CalculatorTool()], max_steps=5).run("你好")
        self.assertEqual(res.final_output, "抱歉，我无法使用该工具。")
        self.assertFalse(res.truncated)
        results = [e for e in res.steps if e["type"] == "tool_result"]
        self.assertIn("未知工具", results[0]["content"])

    def test_tool_exception_survives(self) -> None:
        class BoomTool(Tool):
            name = "boom"
            description = "总是抛异常的工具"
            parameters = {"type": "object", "properties": {}}

            def run(self, **kwargs):
                raise RuntimeError("boom-boom")

        llm = MockLLM(
            [
                Message.assistant(tool_calls=[ToolCall(id="c1", name="boom", arguments={})]),
                Message.assistant("工具出错了，我换一种方式。"),
            ]
        )
        res = Agent(llm=llm, tools=[BoomTool()], max_steps=4).run("触发异常")
        self.assertEqual(res.final_output, "工具出错了，我换一种方式。")
        self.assertFalse(res.truncated)
        results = [e for e in res.steps if e["type"] == "tool_result"]
        self.assertIn("异常", results[0]["content"])

    def test_max_steps_truncates(self) -> None:
        endless = [
            Message.assistant(
                tool_calls=[ToolCall(id="x", name="calculator", arguments={"expression": "1"})]
            )
        ] * 20
        llm = MockLLM(endless)
        res = Agent(llm=llm, tools=[CalculatorTool()], max_steps=3).run("一直调用工具")
        self.assertTrue(res.truncated)
        self.assertEqual(res.steps_used, 3)
        self.assertIn("最大迭代步数", res.final_output)

    def test_reply_without_tools(self) -> None:
        llm = MockLLM([Message.assistant("直接回答，无需工具。")])
        res = Agent(llm=llm, tools=[CalculatorTool()], max_steps=3).run("你好")
        self.assertEqual(res.final_output, "直接回答，无需工具。")
        self.assertFalse(res.truncated)
        self.assertEqual(res.steps_used, 1)


if __name__ == "__main__":
    unittest.main()
