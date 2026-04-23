# Codex Working Notes

## Current Work

当前正在把旧项目的非记忆能力迁移到新的 LANGVIDEO 多 Agent 框架骨架。

旧项目迁移来源：
- `/Users/apexwave/Desktop/Projects/AGENTS`

当前新项目目录：
- `/Users/apexwave/Desktop/Agents`

已确认的方向：
- 顶层继续使用 `Deepagents/` + `MainServer/`。
- `Deepagents/*/Agent/` 内部不再使用旧的 `Capabilities/` 作为目标目录。
- `Agent/` 内部采用 `Models/`、`SubAgent/`、`middlewares/`、`tools/`、`store/`、`server/`、`sandbox.py`。
- 通讯协议以 `MainServer/state.py` 的 `AgentMail` 为准，旧 `AgentMessage` 仅作为兼容别名。
- Agent 身份只使用 `agent_name`，不再使用单独的 `agent_id`。
- Sandbox 按 `agent_name` 隔离，容器名包含 agent name，并只挂载对应 `Deepagents/<agent_name>/workspace` 到容器内 `/workspace`。
- `AgentMail.type="task"` 只是另一种消息格式，不单独注册和管理任务状态。
- 附件传递使用 `Link`，模型工具里本地文件必须写成 `/workspace/...`；网络 URL 可以直接传。
- 模型配置使用项目内 `Deepagents/*/Agent/Models/model_config.json`；该文件包含本地凭据，默认不提交。
- MainAgent backend 统一通过当前项目定义的 `sandbox.py` 创建。
- 本轮暂时不迁移记忆系统，旧项目的 `MemoryManage/*`、RAG、索引构建、记忆召回先不进入骨架。

## Before Editing

每次改代码或文档前，先阅读这些文档中和任务相关的部分：
- `constraint.MD`：当前项目结构与红线，以它为最高约束。
- `PROJECT_OVERVIEW.md`：面向项目入口的简版说明。
- `GENERIC_TOOL_FRAMEWORK.md`：tool 单文件写法。
- `GENERIC_MIDDLEWARE_FRAMEWORK.md`：middleware 单文件写法。

## Documentation Sync Rule

改动一个文档后，必须检查其他文档是否需要同步：
- 结构变更先改 `constraint.MD`，再同步 `PROJECT_OVERVIEW.md`。
- tool 规则变更同步 `GENERIC_TOOL_FRAMEWORK.md`。
- middleware 规则变更同步 `GENERIC_MIDDLEWARE_FRAMEWORK.md`。
- 文件改名后，全文检索旧文件名和旧路径引用。

除 `constraint.MD` 外，其他文档不要维护完整目录树，只保留稳定摘要和关键路径，避免目录细节频繁变动。

## Testing Rule

真实模型测试必须流式监控执行过程，不能只看最终结果。测试时要观察：
- 模型是否真的发起 tool call。
- tool writer 是否输出关键事件。
- middleware writer 是否输出 hook trace。
- state 是否按声明更新。
- `ToolMessage` 是否写入 `Command(update=...)`。

## Stream Output Rule

官方 LangChain / LangGraph 文档把 stream mode 分成不同用途：
- `updates`：每个 agent / graph step 后输出状态更新，适合观察块级进度。
- `custom`：由 graph node、middleware 或 tool writer 主动发出的自定义事件，适合观察关键业务事件。
- `messages`：LLM token / message chunk 和 metadata，适合聊天 UI 或调试 token 级输出。

本项目真实模型测试的原则：
- 内部必须逐块消费真实 stream，保证可以发现中途 tool call、middleware hook 和 writer 事件。
- 对用户展示时不要逐 token 打印 `messages`，只输出块级摘要和关键信息。
- 常规真实模型测试优先使用 `stream_mode=["updates", "custom"]`。
- 只有需要验证模型是否真的发起 tool call 或最终 assistant 内容时，才附加 `messages`；附加后必须聚合成块级摘要再输出。
- 用户可见输出只保留这些块：
  - setup：本次测试的 agent、receiver。
  - middleware：`before_agent`、`wrap_model_call`、`wrap_tool_call`、`after_agent` 的 start/end 摘要。
  - model：是否发起 tool call，tool 名称、目标 agent、关键参数。
  - tool：tool start / success / error 和 `ToolMessage` 摘要。
  - result：接收方 inbox 是否收到目标 `AgentMail`，以及关键 content 命中情况。
  - error：异常类型、阶段和可操作原因。

参考官方文档：
- https://docs.langchain.com/oss/python/langchain/streaming
- https://docs.langchain.com/langgraph-platform/streaming
