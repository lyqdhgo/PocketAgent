"""对话记忆：滑动窗口 + 系统消息保护。

当前实现按消息条数截断（可配置），足够支撑本地演示与单测；
更完整的方案（按 token 数截断、LLM 摘要压缩、向量检索等）见 README「后续规划」。
"""

from __future__ import annotations

from typing import Optional

from .models import Message, ROLE_SYSTEM


class RollingMemory:
    """定长滑动窗口记忆：保留全部 system 消息 + 最近 max_messages 条其它消息。"""

    def __init__(self, max_messages: int = 30):
        self.max_messages = max(4, max_messages)
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def reset(self) -> None:
        self._messages.clear()

    def to_api(self) -> list[dict]:
        """返回裁剪后的 API 消息列表（system 始终保留，其它取最近 N 条）。"""
        if not self._messages:
            return []
        system_msgs = [m for m in self._messages if m.role == ROLE_SYSTEM]
        others = [m for m in self._messages if m.role != ROLE_SYSTEM]
        if len(others) > self.max_messages:
            others = others[-self.max_messages:]
        return [m.to_api() for m in (system_msgs + others)]

    def __len__(self) -> int:
        return len(self._messages)
