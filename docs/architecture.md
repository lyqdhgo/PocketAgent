# PocketAgent 架构说明

本文面向"想快速看懂代码 / 准备面试讲解"的读者，按依赖顺序自底向上介绍各模块。

## 分层视图

```
┌───────────────────────────────────────────────────────────────┐
│ 入口层：python -m pocketagent(.cli)  ·  python -m pocketagent.gui │
│         examples/demo_mock.py（离线演示脚本）                  │
├───────────────────────────────────────────────────────────────┤
│ 编排层  team.py   AgentTeam（Planner → Workers → Reporter）    │
│          - Planner：LLM + submit_plan 结构化工具，拆解任务      │
│          - Worker：独立 Agent 实例，串行/线程池并行            │
│          - Reporter：LLM 润色 or 模板降级 → Markdown 落盘      │
├───────────────────────────────────────────────────────────────┤
│ 单 Agent agent.py   Agent.run()                                │
│          ReAct 循环：chat → tool_calls? → 执行工具→回填→chat   │
│          （最大步数、未知工具/异常兜底、事件流输出）            │
├───────────────────────────────────────────────────────────────┤
│ 组件层                                                         │
│  llm.py    BaseLLM / OpenAICompatibleLLM / MockLLM / llm_from_env │
│  toy.py    ToyLLM（规则大脑，离线演示）                        │
│  memory.py RollingMemory（滑动窗口 + system 保护）             │
│  tools.py  Tool 基类 / ToolRegistry（schema 导出）             │
│  builtin_tools.py Calculator / CurrentTime / Wikipedia / FetchUrl │
├───────────────────────────────────────────────────────────────┤
│ 基础层  models.py（Message / ToolCall） envfile.py（.env 读取） │
└───────────────────────────────────────────────────────────────┘
```

## 关键流程

### 1. 单 Agent ReAct 循环（`agent.py`）

```
用户输入 ──► 记忆(含system) ──► LLM.chat(messages, tools)
                                      │
                      有 tool_calls？ ┼─否─► 记录 final，结束
                                      │是
                       对每个调用：执行工具（异常捕获）
                              │
                    role="tool" 消息回填记忆 ──► 回到 LLM.chat
（循环受 max_steps 限制，超出标记 truncated）
```

要点：
- `messages` 全程是内部 `Message` 对象，仅在发请求时 `to_api()` 序列化；
- 工具结果**必须**携带对应 `tool_call_id`，这是各家 function calling 协议的硬性要求（有单测断言）；
- 每个动作生成事件 dict，`on_event` 回调供 CLI/GUI 实时展示。

### 2. Planner 结构化拆解（`team.py`）

不用"请输出 JSON 文本再解析"，而是让 Planner 通过内置工具
`submit_plan(tasks=[{title, description}, ...])` 以 function calling 提交计划：
- 结构由 JSON Schema 约束，天然规避换行/引号转义问题；
- 兼容层：若模型把 JSON 写在 content 里也能解析；都失败则降级为单个整体任务。

### 3. Worker 并行与隔离

每个 Worker 是完整 `Agent`（独立 RollingMemory），共享只读工具注册表与 LLM 客户端：
- 无共享可变状态 → 线程池并行安全；
- 事件追加用锁保护；
- `parallel=False`（默认）保证演示输出顺序可读。

### 4. Reporter 与降级

- 成功：把各 Worker 结论拼成 transcript 交给 LLM 润色成 Markdown；
- 失败（无 Key / 网络 / 解析）：自动使用模板汇总（`_assemble_md`）；
- 最终产物统一落盘 `outputs/research_<时间戳>.md`，并附带"任务拆解 + 工具调用附录"。

## 离线如何跑通（ToyLLM）

`toy.py` 是一个规则大脑：识别"数学表达式 / 时间类问题"，按需触发工具；
面对 Planner 场景时生成确定的子任务列表。因此**没有真实模型也能看到
Agent 循环与多智能体编排完整运行**，代价是回答是模板化的。

## 兼容哪些模型

一切以 OpenAI `chat/completions` 为协议的 endpoint：
DeepSeek、OpenAI、Ollama、vLLM、one-api 等中转。通过 `llm_from_env()`
读取 `AGENT_BASE_URL / AGENT_MODEL / AGENT_API_KEY`（支持 .env）。

## 测试策略

- `MockLLM` 脚本化回复 → 测试确定性、离线、零成本；
- 网络层用 `unittest.mock` 打桩 `urlopen`，校验请求载荷与解析逻辑；
- 目录：`tests/test_tools.py`、`test_memory.py`、`test_llm.py`、
  `test_agent.py`、`test_team.py`，共 31 例。
