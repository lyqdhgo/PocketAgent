"""LLM 抽象层。

- BaseLLM：统一 chat 接口（一轮对话，可能返回 tool_calls）
- OpenAICompatibleLLM：通过 OpenAI 兼容 REST 接口调用任意模型
  （DeepSeek / OpenAI / Ollama / vLLM / 各类中转服务均可）
- MockLLM：脚本化假模型，用于单元测试（离线、确定性、零成本）
- llm_from_env：按环境变量构造真实客户端

核心只使用标准库 urllib，因此整个核心包无需任何第三方依赖即可运行与测试。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from .models import Message, ToolCall, ROLE_ASSISTANT


class LLMError(RuntimeError):
    """所有与模型服务相关的错误统一封装。"""


class BaseLLM:
    """所有 LLM 实现的统一接口（一次调用 = 一轮模型回复）。"""

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> Message:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    """基于 Chat Completions API 的通用客户端（REST + urllib，零第三方依赖）。"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
        temperature: float = 0.3,
        max_tokens: Optional[int] = 2048,
    ):
        self.model = model
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _build_payload(
        self,
        messages: list[Message],
        tools: Optional[list],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
        }
        tokens = self.max_tokens if max_tokens is None else max_tokens
        if tokens:
            payload["max_tokens"] = tokens
        if tools:
            payload["tools"] = [t.openai_schema() for t in tools]
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, url: str, payload: dict[str, Any]) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"无法连接模型服务（{self.base_url}）：{exc.reason}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(f"模型返回了非 JSON 内容：{body[:300]}") from exc

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Message:
        payload = self._build_payload(messages, tools, temperature, max_tokens)
        data = self._post(self.chat_url, payload)
        if "error" in data:
            raise LLMError(f"模型服务错误：{data['error']}")
        try:
            raw_msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            snippet = json.dumps(data, ensure_ascii=False)[:300]
            raise LLMError(f"响应缺少 choices：{snippet}") from exc
        content = raw_msg.get("content") or ""
        if isinstance(content, list):  # 部分多模态实现返回内容片段数组
            content = "".join(
                seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content
            )
        tool_calls = [ToolCall.from_api(tc) for tc in raw_msg.get("tool_calls") or []]
        return Message(
            role=ROLE_ASSISTANT,
            content=content,
            tool_calls=tool_calls or None,
            usage=data.get("usage"),
        )


class MockLLM(BaseLLM):
    """脚本化假模型：每次 chat 依次弹出一条预设回复，用于单元测试与离线演示。"""

    def __init__(self, responses: list[Message]):
        if not responses:
            raise ValueError("MockLLM 至少需要一条预设回复")
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Message:
        if not self.responses:
            raise LLMError("MockLLM 预设回复已耗尽（循环步数过多？）")
        msg = self.responses.pop(0)
        self.calls.append(
            {"messages": [m.to_api() for m in messages], "tools": bool(tools)}
        )
        return msg


def llm_from_env(
    env: Optional[dict] = None, *, overrides: Optional[dict] = None
) -> OpenAICompatibleLLM:
    """从环境变量构造真实客户端。

    环境变量（也可写在 .env 中）：
      AGENT_API_KEY / AGENT_BASE_URL / AGENT_MODEL
      兼容常见命名：DEEPSEEK_API_KEY、OPENAI_API_KEY。
    overrides 中的非空值优先级最高（用于 GUI / CLI 参数）。
    """
    merged = dict(os.environ if env is None else env)
    if overrides:
        for k, v in overrides.items():
            if v:
                merged[k] = v

    api_key = (merged.get("AGENT_API_KEY") or "").strip()
    base_url = (merged.get("AGENT_BASE_URL") or "").strip().rstrip("/")
    model = (merged.get("AGENT_MODEL") or "").strip()

    if not api_key:
        api_key = (merged.get("DEEPSEEK_API_KEY") or merged.get("OPENAI_API_KEY") or "").strip()
    if not base_url:
        if "OPENAI_API_KEY" in merged and "DEEPSEEK_API_KEY" not in merged:
            base_url = "https://api.openai.com/v1"
        else:
            base_url = "https://api.deepseek.com"
    if not model:
        if "openai.com" in base_url:
            model = "gpt-4o-mini"
        elif "deepseek" in base_url:
            model = "deepseek-chat"
        else:
            raise LLMError(
                "无法推断模型名，请显式设置 AGENT_MODEL（例如本地 Ollama 的 qwen2.5:7b）"
            )

    is_local = any(k in base_url for k in ("127.0.0.1", "localhost", "0.0.0.0"))
    if not api_key and not is_local:
        raise LLMError(
            "缺少 API Key。请复制 .env.example 为 .env 并填写 AGENT_API_KEY，"
            "或改用本地 Ollama（AGENT_BASE_URL=http://127.0.0.1:11434/v1），"
            "或加 --mock 离线演示。"
        )
    return OpenAICompatibleLLM(model=model, api_key=api_key, base_url=base_url)
