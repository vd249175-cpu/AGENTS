# Codex Working Notes

## Current Work

当前任务已经切换为：按照 `demo/NESTED_DEMO_FRAMEWORK_CONSTRAINTS.md` 的
新规范，重构当前项目里的中间键和工具，并继续保持多 Agent 通讯骨架可运行。

旧项目迁移来源：
- `/Users/apexwave/Desktop/Projects/AGENTS`

当前新项目目录：
- `/Users/apexwave/Desktop/Agents`

已确认的方向：
- 顶层继续使用 `Deepagents/` + `MainServer/`。
- `Deepagents/*/Agent/` 内部不再使用旧的 `Capabilities/` 作为目标目录。
- `Agent/` 内部采用 `Models/`、`SubAgent/`、`middlewares/`、`tools/`、`store/`、`server/`、`sandbox.py`。
- 中间键和工具要按 demo 规范重构成 `Config` / `SubState` / `Schema` /
  `helper` / `wrapper` / `default entry` 这套单文件形态。
- 上层 agent 要把完整 `context` 继续向下透传给 middleware / tool / nested agent。
- middleware 只挂工具，不在 `create_agent(...)` 里重复传同一批 tools。
- 通讯协议以 `MainServer/state.py` 的 `AgentMail` 为准，旧 `AgentMessage` 仅作为兼容别名。
- Agent 身份只使用 `agent_name`，不再使用单独的 `agent_id`。
- Sandbox 按 `agent_name` 隔离，容器名包含 agent name，并只挂载对应 `Deepagents/<agent_name>/workspace` 到容器内 `/workspace`。
- `AgentMail.type="task"` 只是另一种消息格式，不单独注册和管理任务状态。
- 附件传递使用 `Link`，模型工具里本地文件必须写成 `/workspace/...`；网络 URL 可以直接传。
- 浏览器自动化通过 `workspace/skills/agent-browser/SKILL.md` 给 agent 使用，运行在 Docker sandbox 里的 `agent-browser` + Chromium，不是 Codex 自身能力。
- 模型配置使用项目内 `Deepagents/*/Agent/Models/model_config.json`；该文件包含本地凭据，默认不提交。
- MainAgent backend 统一通过当前项目定义的 `sandbox.py` 创建。
- 当前已开始接入复制到本项目的 `memory/` 包；先在 `SeedAgent` 上挂载
  `ingest_knowledge_document` 工具和 `manage_knowledge` 中间键，用于把
  `workspace/knowledge` 中的长文档切分进入记忆图，并管理记忆内容。
- SeedAgent 的 chunk 入库配置关闭文档级 run_id 派生，确保
  `ingest_knowledge_document` 和 `manage_knowledge` 共用 `SeedAgent-knowledge`
  这一套 Agent 知识库。
- 旧项目的 `MemoryManage/*`、RAG、索引构建、记忆召回仍不直接迁入骨架；
  记忆能力通过当前 `memory/` 包的 demo 风格 wrapper 接入。
- demo 规范是当前 middleware/tool 重构的最高实现参照，后续若和旧实现冲突，优先服从 demo 规范再同步其他约束文档。

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

## Test Output Rule

真实模型测试既要完整观察内部执行，也要避免控制台刷屏。

控制台输出规则：
- 不打印重复前缀；同一阶段只用一次短标签，例如 `setup:`、`planner:`、`worker:`、`verify:`。
- 不逐条打印完整中间过程；只输出关键里程碑、最终结果和日志路径。
- 不打印完整 state、完整 message、完整 event payload、完整模型 chunk。
- 失败时只输出阶段、简短原因和下一步可操作线索。

日志规则：
- 测试必须把完整监控过程保存到 `tests/logs/` 下的 `.jsonl` 文件。
- 完整日志里保存 health、status、event、request outcome、verification 等原始或近原始信息。
- 控制台摘要必须包含本次完整日志路径，便于需要时回放。
- 查看日志时默认只看末尾，例如 `tail -n 80 <log.jsonl>`；只有定位具体问题时才按关键字检索或打开更多内容。

断言规则：
- 真实模型测试不要用 `assert` 在中途卡死流程。
- 每个阶段都收集 `ok/error/reason`，测试尽量继续执行到 summary。
- 最后统一给出 `planner_ok`、`worker_ok`、`delivery_ok` 和退出码。
