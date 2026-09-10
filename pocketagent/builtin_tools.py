"""内置工具集：安全计算器 / 当前时间 / Wikipedia 检索 / 网页正文抓取。

安全与可用性设计：
- calculator 使用 ast 白名单求值，绝不执行任意代码（防注入、防资源耗尽）；
- Wikipedia / 网页抓取采用懒加载与异常兜底，网络不可用时优雅返回错误文本，
  模型读到错误后可自行换一种方式完成任务。
"""

from __future__ import annotations

import ast
import math
import operator
import re
import urllib.request
from datetime import datetime
from typing import Any, Optional

from .tools import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# 工具 1：安全计算器
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY: dict[type, Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "degrees": math.degrees,
    "radians": math.radians,
}
_ALLOWED_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_MAX_ABS_EXPONENT = 512  # 防止 2**10**6 之类导致资源耗尽
_MAX_INT_BITS = 1 << 20  # 结果整数过大时报错（约 30 万位十进制，已很宽裕）
_MAX_EXPR_LEN = 300

# 用于从自然语言中粗略抽取简单算术表达式（供离线规则大脑与演示使用）
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*[+\-*/]\s*)+\d+(?:\.\d+)?")


def extract_expressions(text: str) -> list[str]:
    """从自然语言中粗略抽取形如 21 * 2 / 100 + 4 的算术表达式。"""
    found: list[str] = []
    for m in _NUMBER_RE.finditer(text or ""):
        expr = m.group(0).replace(" ", "")
        if expr not in found:
            found.append(expr)
    return found


def format_number(value: Any) -> str:
    """把结果格式化为简洁字符串：整数不带小数点，浮点去掉多余尾零。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _check(value: Any) -> Any:
    """每个数值节点求值后做资源防护。"""
    if isinstance(value, int):
        if value.bit_length() > _MAX_INT_BITS:
            raise OverflowError("结果数值过大，已超出安全上限")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise OverflowError("结果不是有限数值")
    return value


def _eval_node(node: ast.AST) -> Any:
    """递归求值：仅允许数字字面量 + 白名单运算/函数，遇到其它一律抛错。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字字面量")
        return _check(node.value)
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, (int, float)) or abs(right) > _MAX_ABS_EXPONENT:
                raise ValueError("指数过大（已限制在 ±512 以内）")
        try:
            result = _ALLOWED_BINOPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ZeroDivisionError("除数不能为 0") from None
        return _check(result)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY:
            raise ValueError("不支持的运算符")
        return _check(_ALLOWED_UNARY[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("只允许调用白名单数学函数")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        return _check(_ALLOWED_FUNCS[node.func.id](*args))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _check(_ALLOWED_CONSTS[node.id])
        raise ValueError(f"未知变量: {node.id}")
    raise ValueError(f"不支持的表达式片段: {type(node).__name__}")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "执行安全的数学表达式计算。支持 + - * / // % ** 与括号，"
        "以及数学函数 sqrt/log/log2/log10/exp/sin/cos/tan/floor/ceil/abs 等。"
        "示例：(1+2)*3/4、sqrt(16)+2**10"
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "要计算的数学表达式，如 (1+2)*3"}
        },
        "required": ["expression"],
    }

    def run(self, expression: Optional[str] = None, **_: Any) -> str:
        expr = (expression or "").strip()
        if not expr:
            return "计算失败：缺少 expression 参数"
        if len(expr) > _MAX_EXPR_LEN:
            return "计算失败：表达式过长"
        if not re.fullmatch(r"[0-9+\-*/().,\sA-Za-z_]*", expr):
            return "计算失败：包含不支持的字符"
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval_node(tree)
            return format_number(value)
        except ZeroDivisionError as exc:
            return f"计算失败：{exc}"
        except (ValueError, OverflowError, SyntaxError, TypeError) as exc:
            return f"计算失败：{exc}"


# ---------------------------------------------------------------------------
# 工具 2：当前时间
# ---------------------------------------------------------------------------

class CurrentTimeTool(Tool):
    name = "current_time"
    description = "获取服务器当前日期、时间与星期几。适合回答「现在几点 / 今天几号 / 星期几」等问题。"
    parameters = {"type": "object", "properties": {}}

    def run(self, **_: Any) -> str:
        now = datetime.now()
        weekday = "一二三四五六日"[now.weekday()]
        return f"{now:%Y-%m-%d %H:%M:%S}，星期{weekday}（本地时区）"


# ---------------------------------------------------------------------------
# 工具 3：Wikipedia 检索（可选依赖 wikipedia）
# ---------------------------------------------------------------------------

class WikipediaTool(Tool):
    name = "wikipedia"
    description = "检索 Wikipedia 词条摘要，用于获取常识 / 背景知识。lang 可选 zh / en / auto。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要查询的关键词或词条名"},
            "lang": {"type": "string", "description": "语言，auto = 先中文后英文（默认 auto）"},
            "sentences": {"type": "integer", "description": "摘要句子数，默认 3"},
        },
        "required": ["query"],
    }

    def run(self, query: str = "", lang: str = "auto", sentences: int = 3, **_: Any) -> str:
        try:
            import wikipedia as _wk  # 可选依赖，惰性导入
        except ImportError:
            return "[错误] 需要先安装 wikipedia：pip install wikipedia（或当前环境无网络）"
        query = (query or "").strip()
        if not query:
            return "[错误] 缺少 query 参数"
        try:
            sentences = max(1, min(int(sentences), 10))
        except (TypeError, ValueError):
            sentences = 3
        langs = ["zh", "en"] if lang == "auto" else [str(lang)]
        last_err: Optional[Exception] = None
        for cur in langs:
            try:
                _wk.set_lang(cur)
                summary = _wk.summary(query, sentences=sentences)
                if summary:
                    text = summary.strip().replace("\n", " ")
                    return f"[Wikipedia·{cur}] {text[:2000]}"
            except Exception as exc:  # noqa: BLE001 —— 词条不存在/网络错误都继续尝试下一语言
                last_err = exc
        reason = str(last_err or "未知错误")[:120]
        return f"[错误] Wikipedia 查询失败（{reason}）。可尝试更精确的词条名。"


# ---------------------------------------------------------------------------
# 工具 4：网页正文抓取（联网获取资料）
# ---------------------------------------------------------------------------

class FetchUrlTool(Tool):
    name = "fetch_url"
    description = "抓取一个 http(s) 网页并返回去除标签后的纯文本片段，用于联网获取资料。"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "完整网址，如 https://example.com"},
            "max_chars": {"type": "integer", "description": "最多返回字符数，默认 3000"},
        },
        "required": ["url"],
    }

    def run(self, url: str = "", max_chars: int = 3000, **_: Any) -> str:
        url = (url or "").strip()
        if not re.match(r"^https?://", url):
            return "[错误] url 必须以 http(s):// 开头"
        try:
            max_chars = max(200, min(int(max_chars), 20000))
        except (TypeError, ValueError):
            max_chars = 3000
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (PocketAgent/0.1)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read(200 * 1024).decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001 —— 网络问题返回给模型处理
            return f"[错误] 网页抓取失败：{str(exc)[:150]}"
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "[错误] 未能从页面提取到文本"
        return text[:max_chars]


def default_tools() -> ToolRegistry:
    """默认工具集：注册全部内置工具。"""
    return ToolRegistry([CalculatorTool(), CurrentTimeTool(), WikipediaTool(), FetchUrlTool()])
