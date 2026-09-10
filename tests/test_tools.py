"""工具层单元测试：计算器安全/正确性、时间工具、注册表、表达式抽取。"""

from __future__ import annotations

import unittest

from pocketagent.builtin_tools import (
    CalculatorTool,
    CurrentTimeTool,
    extract_expressions,
)
from pocketagent.tools import Tool, ToolRegistry


class TestCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CalculatorTool()

    def test_basic_priority(self) -> None:
        self.assertEqual(self.tool.run(expression="1 + 2 * 3"), "7")

    def test_float_division(self) -> None:
        self.assertEqual(self.tool.run(expression="10 / 4"), "2.5")

    def test_paren_and_pow(self) -> None:
        self.assertEqual(self.tool.run(expression="(1 + 2) ** 2"), "9")

    def test_math_functions(self) -> None:
        self.assertEqual(self.tool.run(expression="sqrt(16) + 1"), "5")
        self.assertEqual(self.tool.run(expression="floor(3.7)"), "3")

    def test_constants(self) -> None:
        self.assertEqual(self.tool.run(expression="pi * 0"), "0")

    def test_division_by_zero(self) -> None:
        out = self.tool.run(expression="1 / 0")
        self.assertTrue(out.startswith("计算失败"), out)

    def test_injection_rejected(self) -> None:
        # 含引号/下划线等非法字符 → 直接拒绝
        out = self.tool.run(expression="__import__('os').system('ls')")
        self.assertTrue(out.startswith("计算失败"), out)
        # 纯字母函数但不在白名单 → 拒绝
        out2 = self.tool.run(expression="exec('1')")
        self.assertTrue(out2.startswith("计算失败"), out2)

    def test_pow_guard(self) -> None:
        out = self.tool.run(expression="2 ** 100000")
        self.assertTrue(out.startswith("计算失败"), out)

    def test_missing_argument(self) -> None:
        self.assertTrue(self.tool.run().startswith("计算失败"))


class TestCurrentTime(unittest.TestCase):
    def test_time_format(self) -> None:
        out = CurrentTimeTool().run()
        self.assertRegex(out, r"20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}，星期[一二三四五六日]")


class TestToolRegistry(unittest.TestCase):
    def test_duplicate_name_raises(self) -> None:
        reg = ToolRegistry()
        reg.add(CalculatorTool())
        with self.assertRaises(ValueError):
            reg.add(CalculatorTool())

    def test_get_unknown_returns_none(self) -> None:
        self.assertIsNone(ToolRegistry().get("not_exist"))

    def test_openai_schema_shape(self) -> None:
        schemas = ToolRegistry([CalculatorTool()]).schemas()
        self.assertEqual(schemas[0]["type"], "function")
        fn = schemas[0]["function"]
        self.assertEqual(fn["name"], "calculator")
        self.assertIn("parameters", fn)

    def test_tool_without_name_raises(self) -> None:
        class NoName(Tool):
            pass

        with self.assertRaises(ValueError):
            ToolRegistry().add(NoName())


class TestExtractExpressions(unittest.TestCase):
    def test_extract(self) -> None:
        found = extract_expressions("请计算 21 * 2 与 100 / 4 的结果")
        self.assertIn("21*2", found)
        self.assertIn("100/4", found)

    def test_extract_empty(self) -> None:
        self.assertEqual(extract_expressions("现在几点了"), [])


if __name__ == "__main__":
    unittest.main()
