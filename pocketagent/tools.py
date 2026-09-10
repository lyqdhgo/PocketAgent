"""工具抽象与注册表。

设计要点：
- Tool 采用"显式 schema"而非装饰器魔法，代码一目了然，便于面试讲解；
- ToolRegistry 负责名称唯一性与 schema 导出（OpenAI function calling 格式）；
- 工具执行异常统一转为字符串返回给模型（而不是中断对话），模型可以自我纠正。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class Tool:
    """工具基类：子类只需声明 name / description / parameters 并实现 run。"""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def run(self, **kwargs: Any) -> str:
        """执行工具，返回给模型的文本结果。"""
        raise NotImplementedError

    def openai_schema(self) -> dict[str, Any]:
        """导出为 OpenAI function calling 的 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：add / get / 批量导出 schema。"""

    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("工具必须声明 name")
        if tool.name in self._tools:
            raise ValueError(f"工具名重复: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def as_list(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
