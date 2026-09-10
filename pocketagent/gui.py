"""Gradio 图形界面（可离线体验，也可接入真实模型）。

启动：python -m pocketagent.gui   （需要先：pip install -r requirements.txt）
浏览器访问 http://127.0.0.1:7860

- 「单 Agent 对话」页：可勾选工具、切换模型来源，实时展示思考与工具调用日志；
- 「多智能体研究」页：Planner → Workers → 报告，一键生成 Markdown 研究报告。
"""

from __future__ import annotations

import os
from typing import Any, Optional

import gradio as gr

from .agent import Agent, render_event
from .builtin_tools import default_tools
from .envfile import load_dotenv
from .llm import LLMError, llm_from_env
from .team import AgentTeam
from .tools import ToolRegistry

# 模块导入时读取一次 .env（GUI 启动场景）
load_dotenv()

PROVIDER_LABELS = {
    "mock": "Mock（离线演示，无需 Key）",
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "ollama": "Ollama（本地）",
}

# provider -> (默认模型, 默认 base_url, 优先读取的 Key 环境变量)
_PROVIDER_META = {
    "deepseek": ("deepseek-chat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openai": ("gpt-4o-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("qwen2.5:7b", "http://127.0.0.1:11434/v1", ""),
}


def make_llm(provider: str, api_key_text: str = ""):
    """按 UI 选择构造 LLM。provider == 'mock' 时返回离线 ToyLLM。"""
    if provider == "mock":
        from .toy import ToyLLM

        return ToyLLM()

    default_model, default_base, key_env = _PROVIDER_META[provider]
    key = (api_key_text or "").strip() or os.environ.get("AGENT_API_KEY", "") or (
        os.environ.get(key_env, "") if key_env else ""
    )
    overrides = {
        "AGENT_MODEL": os.environ.get("AGENT_MODEL") or default_model,
        "AGENT_BASE_URL": os.environ.get("AGENT_BASE_URL") or default_base,
        "AGENT_API_KEY": key,
    }
    return llm_from_env(overrides=overrides)


def _keep_tools(names: Optional[list[str]]) -> ToolRegistry:
    reg = default_tools()
    if not names:
        return reg
    keep = set(names)
    return ToolRegistry([t for t in reg.as_list() if t.name in keep])


# ---------------------------------------------------------------------------
# 单 Agent 对话
# ---------------------------------------------------------------------------

def chat_respond(user: str, history: list, provider: str, api_key: str, tool_names: list):
    history = list(history or [])
    history.append({"role": "user", "content": user})
    logs: list[str] = []
    try:
        llm = make_llm(provider, api_key)
        agent = Agent(
            name="assistant",
            llm=llm,
            tools=_keep_tools(tool_names),
            on_event=lambda ev: logs.append(render_event(ev)),
        )
        result = agent.run(user)
        logs.append(f"[完成] 步数={result.steps_used} 截断={result.truncated}")
        answer = result.final_output or "（模型没有返回内容）"
        history.append({"role": "assistant", "content": answer})
    except LLMError as exc:
        logs.append(f"[错误] {exc}")
        history.append(
            {
                "role": "assistant",
                "content": f"调用模型失败：{exc}\n\n提示：可在上方把模型切换为"
                "「Mock（离线演示）」体验，或检查 .env 配置。",
            }
        )
    except Exception as exc:  # noqa: BLE001 —— UI 兜底，避免页面崩溃
        logs.append(f"[错误] {exc}")
        history.append({"role": "assistant", "content": f"发生异常：{exc}"})
    return history, "\n".join(logs)


# ---------------------------------------------------------------------------
# 多智能体研究
# ---------------------------------------------------------------------------

def research_run(question: str, provider: str, api_key: str, progress=gr.Progress()):
    logs: list[str] = []
    if not (question or "").strip():
        return "请输入研究问题。", "", "-"
    try:
        llm = make_llm(provider, api_key)
        team = AgentTeam(
            llm=llm,
            tools=default_tools(),
            output_dir="outputs",
            on_event=lambda ev: logs.append(render_event(ev)),
        )
        result = team.run(question, polish=provider != "mock")
        report = result.final_report or "（无报告）"
        return "\n".join(logs), report, str(result.report_path or "-")
    except LLMError as exc:
        logs.append(f"[错误] {exc}")
        return "\n".join(logs), f"调用模型失败：{exc}", "-"
    except Exception as exc:  # noqa: BLE001
        logs.append(f"[错误] {exc}")
        return "\n".join(logs), f"发生异常：{exc}", "-"


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="PocketAgent · 多智能体演示") as demo:
        gr.Markdown(
            "# 🧩 PocketAgent — 面试展示级多智能体项目\n"
            "自研 ReAct 工具调用循环 + Planner/Worker/Reporter 团队编排。"
            "默认 Mock 模式**无需任何 API Key** 即可体验；选择 DeepSeek/OpenAI/Ollama 需配置对应 Key。"
        )
        with gr.Row():
            provider = gr.Dropdown(
                choices=list(PROVIDER_LABELS.values()),
                value="Mock（离线演示，无需 Key）",
                label="模型来源",
            )
            api_key = gr.Textbox(
                label="API Key（留空则读 .env / 环境变量）",
                placeholder="sk-...",
                type="password",
            )
        provider_key = gr.State("mock")

        def on_provider_change(label: str):
            for key, lab in PROVIDER_LABELS.items():
                if lab == label:
                    return key
            return "mock"

        provider.change(on_provider_change, provider, provider_key)

        with gr.Tab("单 Agent 对话"):
            gr.Markdown("示例：`帮我计算 12*8-4`、`现在几点了`、`用中文介绍下 Python 的 GIL`")
            tool_names = gr.CheckboxGroup(
                choices=default_tools().names(),
                value=default_tools().names(),
                label="启用工具（可勾选开关）",
            )
            chatbot = gr.Chatbot(height=420, label="对话")
            logs_box = gr.Textbox(label="思考 / 工具调用日志", lines=8, max_lines=15)
            with gr.Row():
                msg = gr.Textbox(placeholder="输入你的问题…", show_label=False, scale=5)
                send = gr.Button("发送", variant="primary", scale=1)
            clear_btn = gr.Button("清空对话", size="sm")

            def respond(user, history, prov_key, key, tools):
                if not (user or "").strip():
                    return history, "", "", ""
                return (
                    chat_respond(user, history, prov_key, key, tools)
                    + ("",)  # 追加：清空输入框
                )

            send.click(respond, [msg, chatbot, provider_key, api_key, tool_names], [chatbot, logs_box, msg])
            msg.submit(respond, [msg, chatbot, provider_key, api_key, tool_names], [chatbot, logs_box, msg])
            clear_btn.click(lambda: ([], ""), None, [chatbot, logs_box])

        with gr.Tab("多智能体研究"):
            gr.Markdown(
                "描述一个开放问题，团队会自动：**拆解子任务 → 并行 Worker 执行（可调用工具）→ 汇总 Markdown 报告**。\n"
                "离线示例：`请计算 21*2 与 100/4 的结果，并获取当前时间，汇总成报告`"
            )
            question = gr.Textbox(label="研究问题", lines=3, placeholder="例如：对比 Python 与 Go 的优缺点并给出选型建议")
            run_btn = gr.Button("开始研究", variant="primary")
            team_logs = gr.Textbox(label="实时日志", lines=14, max_lines=30)
            report_out = gr.Markdown(label="报告")
            path_out = gr.Textbox(label="报告文件路径", interactive=False)
            run_btn.click(research_run, [question, provider_key, api_key], [team_logs, report_out, path_out])
            question.submit(research_run, [question, provider_key, api_key], [team_logs, report_out, path_out])

        gr.Markdown(
            "---\n配置参考（写入项目根目录 `.env`）：`AGENT_API_KEY=sk-...`、"
            "`AGENT_BASE_URL=https://api.deepseek.com`、`AGENT_MODEL=deepseek-chat`；"
            "Ollama 本地：`AGENT_BASE_URL=http://127.0.0.1:11434/v1`、`AGENT_MODEL=qwen2.5:7b`。"
            "完整文档见 README.md。"
        )
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
