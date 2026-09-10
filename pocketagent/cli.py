"""命令行入口。

用法示例：
    python -m pocketagent chat                 # 用真实模型（需配置 .env）
    python -m pocketagent chat --mock          # 离线规则大脑（无需任何 Key）
    python -m pocketagent research "你的问题"   # 多智能体研究
    python -m pocketagent doctor               # 环境自检
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from .agent import Agent, render_event
from .builtin_tools import default_tools
from .envfile import load_dotenv
from .llm import BaseLLM, LLMError, llm_from_env
from .team import AgentTeam


def _banner() -> str:
    return (
        "\n===========================================================\n"
        " PocketAgent · 面试展示级多智能体项目\n"
        " 输入 /quit 退出 | 提示：先配置 .env 或使用 --mock\n"
        "==========================================================="
    )


def make_llm(mock: bool, overrides: Optional[dict] = None) -> BaseLLM:
    """按参数构造 LLM：--mock 用 ToyLLM，否则读环境变量。"""
    if mock:
        from .toy import ToyLLM

        return ToyLLM()
    try:
        return llm_from_env(overrides=overrides)
    except LLMError as exc:
        print(f"[配置提示] {exc}")
        print("  可选方案：")
        print("    1) 复制 .env.example 为 .env，填入 AGENT_API_KEY（DeepSeek/OpenAI 均可）")
        print("    2) 使用本地 Ollama：AGENT_BASE_URL=http://127.0.0.1:11434/v1")
        print("    3) 先离线体验：python -m pocketagent chat --mock")
        sys.exit(2)


# ---------------------------------------------------------------------------

def cmd_chat(args: argparse.Namespace) -> None:
    overrides = {
        "AGENT_MODEL": args.model,
        "AGENT_BASE_URL": args.base_url,
        "AGENT_API_KEY": args.api_key,
    }
    llm = make_llm(args.mock, overrides)
    agent = Agent(
        name="assistant",
        llm=llm,
        tools=default_tools(),
        on_event=lambda ev: print(render_event(ev), flush=True),
    )
    print(_banner())
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "退出"):
            print("再见！")
            break
        if user_input in ("/reset", "/clear"):
            agent.memory.reset()
            print("[系统] 对话记忆已清空")
            continue
        agent.run(user_input, reset_memory=False)


def cmd_research(args: argparse.Namespace) -> None:
    question = (args.question or "").strip()
    if not question:
        question = input("请输入你的研究问题：").strip()
    if not question:
        print("问题不能为空。")
        sys.exit(1)

    overrides = {
        "AGENT_MODEL": args.model,
        "AGENT_BASE_URL": args.base_url,
        "AGENT_API_KEY": args.api_key,
    }
    llm = make_llm(args.mock, overrides)
    team = AgentTeam(
        llm=llm,
        tools=default_tools(),
        output_dir=args.output_dir,
        on_event=lambda ev: print(render_event(ev), flush=True),
    )
    result = team.run(question, polish=not args.mock)
    if result.report_path:
        print(f"\n[完成] 报告已保存：{result.report_path}")


def cmd_doctor(args: argparse.Namespace) -> None:
    import platform

    load_dotenv()
    print("== PocketAgent 环境自检 ==")
    print(f"Python   : {platform.python_version()}")
    print(f"平台     : {platform.platform()}")
    from . import __version__

    print(f"版本     : {__version__}")

    tools = default_tools()
    print(f"工具集   : {', '.join(tools.names())}")

    if args.mock:
        print("模式     : mock（离线规则大脑，无需 Key）")
        return
    try:
        llm = llm_from_env()
        masked = (llm.api_key[:4] + "****" + llm.api_key[-4:]) if len(llm.api_key) > 8 else "****"
        print(f"Base URL : {llm.base_url}")
        print(f"Model    : {llm.model}")
        print(f"API Key  : {masked}")
        print("配置检查 : OK")
    except LLMError as exc:
        print(f"配置检查 : 未就绪 —— {exc}")
        print("提示：使用 --mock 可离线自检 Agent 循环。")
        sys.exit(1)


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # 公共选项：挂在每个子命令上（chat --mock 与 --mock chat 均可读，子命令后更符合直觉）
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mock", action="store_true", help="使用离线规则大脑，无需 API Key")
    common.add_argument("--model", default=None, help="覆盖模型名（如 deepseek-chat）")
    common.add_argument("--base-url", default=None, help="覆盖 API 地址（如 https://api.deepseek.com）")
    common.add_argument("--api-key", default=None, help="覆盖 API Key")

    parser = argparse.ArgumentParser(
        prog="pocketagent",
        description="PocketAgent：面试展示级多智能体项目（DeepSeek / OpenAI / Ollama 兼容）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_chat = sub.add_parser("chat", parents=[common], help="交互式单 Agent 对话")
    p_chat.set_defaults(func=cmd_chat)

    p_research = sub.add_parser("research", parents=[common], help="多智能体研究（Planner→Workers→报告）")
    p_research.add_argument("question", nargs="?", help="研究问题")
    p_research.add_argument("--output-dir", default="outputs", help="报告输出目录")
    p_research.set_defaults(func=cmd_research)

    p_doc = sub.add_parser("doctor", parents=[common], help="环境自检")
    p_doc.set_defaults(func=cmd_doctor)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv()  # 读 .env（不覆盖已有环境变量）
    args.func(args)


if __name__ == "__main__":
    main()
