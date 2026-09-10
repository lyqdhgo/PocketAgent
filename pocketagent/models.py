"""消息与数据结构定义。

所有模块共享的最小数据结构：
- Message：与 LLM 交互的通用消息（支持 function calling）
- ToolCall：模型请求调用工具的结构化描述
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# 角色常量
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


@dataclass
class ToolCall:
    """一次工具调用请求（由模型给出）。"""

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "ToolCall":
        """从 OpenAI 兼容的 tool_calls 元素解析（兼容 arguments 为 JSON 字符串）。"""
        fn = raw.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {"_raw": str(raw_args)}
        if not isinstance(arguments, dict):
            arguments = {"_raw": str(arguments)}
        return cls(
            id=str(raw.get("id") or ""),
            name=str(fn.get("name") or ""),
            arguments=arguments,
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    def __str__(self) -> str:  # 便于日志展示
        return f"{self.name}({self.arguments})"


@dataclass
class Message:
    """内部统一消息对象，可序列化为 OpenAI 兼容格式。"""

    role: str
    content: str = ""
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=ROLE_SYSTEM, content=text)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=ROLE_USER, content=text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: Optional[list[ToolCall]] = None) -> "Message":
        return cls(role=ROLE_ASSISTANT, content=text, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role=ROLE_TOOL, content=content, tool_call_id=tool_call_id)

    def to_api(self) -> dict[str, Any]:
        """转换为 Chat Completions API 的 message 对象。"""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_api() for tc in self.tool_calls]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def __repr__(self) -> str:  # 便于日志
        head = self.content[:80].replace("\n", " ")
        if self.tool_calls:
            head += f" | tool_calls={[tc.name for tc in self.tool_calls]}"
        return f"Message({self.role}, {head!r})"
