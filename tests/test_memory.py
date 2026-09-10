"""记忆层单元测试：滑动窗口裁剪 + system 消息保护。"""

from __future__ import annotations

import unittest

from pocketagent.memory import RollingMemory
from pocketagent.models import Message


class TestRollingMemory(unittest.TestCase):
    def test_window_trim_keeps_system_and_recent(self) -> None:
        mem = RollingMemory(max_messages=5)
        mem.add(Message.system("sys"))
        for i in range(10):
            mem.add(Message.user(f"u{i}"))
        api = mem.to_api()
        self.assertEqual(api[0]["role"], "system")
        self.assertEqual(len(api), 6)  # 1 system + 5 最近消息
        self.assertEqual(api[-1]["content"], "u9")
        contents = [m["content"] for m in api]
        self.assertNotIn("u0", contents)
        self.assertNotIn("u4", contents)
        self.assertIn("u5", contents)

    def test_reset(self) -> None:
        mem = RollingMemory()
        mem.add(Message.user("hi"))
        mem.reset()
        self.assertEqual(mem.to_api(), [])

    def test_within_window_no_trim(self) -> None:
        mem = RollingMemory(max_messages=10)
        for i in range(5):
            mem.add(Message.user(f"u{i}"))
        self.assertEqual(len(mem.to_api()), 5)


if __name__ == "__main__":
    unittest.main()
