"""LLM 客户端单元测试：REST 请求构造、响应解析、错误处理（不发起真实网络请求）。"""

from __future__ import annotations

import json
import unittest
from unittest import mock as umock

from pocketagent.builtin_tools import CalculatorTool
from pocketagent.llm import LLMError, OpenAICompatibleLLM
from pocketagent.models import Message


class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _make_urlopen(captured: list, *, error_payload: bool = False):
    """生成假 urlopen：第一次含 tools → 回工具调用；有 tool 消息 → 回最终答案。"""

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured.append(req)
        if error_payload:
            return FakeResponse({"error": {"message": "invalid api key"}})
        payload = json.loads(req.data)
        msgs = payload["messages"]
        if payload.get("tools") and not any(m.get("role") == "tool" for m in msgs):
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "calculator",
                                            "arguments": '{"expression": "1 + 2"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"total_tokens": 42},
                }
            )
        return FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "最终答案"}}], "usage": {}}
        )

    return fake_urlopen


class TestOpenAICompatibleLLM(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = OpenAICompatibleLLM(
            model="test-model", api_key="sk-12345678", base_url="https://api.deepseek.com"
        )
        self.captured: list = []

    def _patch(self, error_payload: bool = False):
        return umock.patch(
            "pocketagent.llm.urllib.request.urlopen",
            _make_urlopen(self.captured, error_payload=error_payload),
        )

    def test_request_and_parse_tool_call(self) -> None:
        with self._patch():
            reply = self.llm.chat([Message.user("计算 1+2")], tools=[CalculatorTool()])
        req = self.captured[0]
        self.assertTrue(req.full_url.endswith("/chat/completions"))
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.headers.get("Authorization"), "Bearer sk-12345678")

        payload = json.loads(req.data)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["tools"][0]["function"]["name"], "calculator")
        self.assertEqual(payload["tool_choice"], "auto")

        self.assertIsNotNone(reply.tool_calls)
        self.assertEqual(reply.tool_calls[0].name, "calculator")
        self.assertEqual(reply.tool_calls[0].arguments["expression"], "1 + 2")

    def test_tool_result_message_serialization(self) -> None:
        with self._patch():
            first = self.llm.chat([Message.user("计算")], tools=[CalculatorTool()])
            reply = self.llm.chat(
                [Message.user("计算"), first, Message.tool_result("call_1", "3")]
            )
        self.assertEqual(reply.content, "最终答案")
        payload = json.loads(self.captured[1].data)
        roles = [m["role"] for m in payload["messages"]]
        self.assertIn("tool", roles)
        tool_msg = [m for m in payload["messages"] if m["role"] == "tool"][0]
        self.assertEqual(tool_msg["tool_call_id"], "call_1")
        self.assertEqual(tool_msg["content"], "3")

    def test_error_payload_raises(self) -> None:
        with self._patch(error_payload=True):
            with self.assertRaises(LLMError):
                self.llm.chat([Message.user("hi")])

    def test_url_without_double_slash(self) -> None:
        llm = OpenAICompatibleLLM(model="m", base_url="https://api.deepseek.com/")
        self.assertEqual(llm.chat_url, "https://api.deepseek.com/chat/completions")


if __name__ == "__main__":
    unittest.main()
