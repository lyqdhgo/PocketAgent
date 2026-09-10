"""PocketAgent —— 面试展示级的多智能体（Agent）项目。

核心特性：
- 自研 ReAct 式 Agent 循环（思考 → 工具调用 → 观察 → 回答），非现成框架封装；
- 可插拔工具系统（Tool / ToolRegistry），内置安全计算器、时间、Wikipedia、网页抓取；
- 多智能体编排：Planner 拆解 → Workers 执行 → Reporter 汇总报告；
- 兼容任意 OpenAI 格式的模型服务：DeepSeek / OpenAI / Ollama / vLLM；
- 零第三方依赖核心（仅标准库），无 API Key 也能用 ToyLLM 离线跑通全流程；
- 提供 CLI、Gradio GUI、单元测试，适合本地演示与上传 GitHub 展示。

快速开始见 README.md。
"""

__version__ = "0.1.0"

from .agent import Agent, AgentResult, DEFAULT_SYSTEM_PROMPT, render_event  # noqa: F401
from .builtin_tools import (  # noqa: F401
    CalculatorTool,
    CurrentTimeTool,
    FetchUrlTool,
    WikipediaTool,
    default_tools,
)
from .llm import (  # noqa: F401
    BaseLLM,
    LLMError,
    MockLLM,
    OpenAICompatibleLLM,
    llm_from_env,
)
from .memory import RollingMemory  # noqa: F401
from .models import Message, ToolCall  # noqa: F401
from .team import AgentTeam, TeamResult  # noqa: F401
from .tools import Tool, ToolRegistry  # noqa: F401

__all__ = [
    "__version__",
    "Agent",
    "AgentResult",
    "DEFAULT_SYSTEM_PROMPT",
    "render_event",
    "CalculatorTool",
    "CurrentTimeTool",
    "FetchUrlTool",
    "WikipediaTool",
    "default_tools",
    "BaseLLM",
    "LLMError",
    "MockLLM",
    "OpenAICompatibleLLM",
    "llm_from_env",
    "RollingMemory",
    "Message",
    "ToolCall",
    "AgentTeam",
    "TeamResult",
    "Tool",
    "ToolRegistry",
]
