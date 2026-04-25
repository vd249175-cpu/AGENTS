# LANGVIDEO 框架约束

> 本文档是当前项目的结构与规则约束。
> 所有后续开发、新增 Agent、新增协议、重构模块，都必须先对照本文档。

当前项目顶层沿用 `Deepagents/` 与 `MainServer/`，但 `Agent/` 内部采用
`middlewares/`、`tools/`、`server/` 这套新结构，不再使用旧的
`Capabilities/` 作为目标目录。

当前不直接迁移旧项目中的 `MemoryManage/*`、RAG、索引构建、记忆召回等历史实现。
记忆能力通过复制到当前项目的 `memory/` 包接入，优先使用它的 demo 风格公开入口：
`ChunkApplyTool` 和 `KnowledgeManagerCapabilityMiddleware`。主 agent 仍可接入
deepagents 自带的 `MemoryMiddleware` 与 `SkillsMiddleware`，分别读取
`workspace/brain/AGENTS.md` 和 `workspace/skills/*/SKILL.md`。

---

## 1. 项目顶层结构

```text
LANGVIDEO/
├── Deepagents/                  # 多 Agent 服务目录
│   ├── __init__.py
│   └── SeedAgent/               # 种子 Agent，示例/模板
├── MainServer/                  # 中心化注册 / 路由 / 调度服务
│   ├── __init__.py
│   ├── main_server.py           # FastAPI 应用：注册、状态、事件、消息路由
│   ├── state.py                 # 全局状态契约
│   ├── protocol.py              # 消息构造工具
│   ├── comm.py                  # Agent 端通讯 SDK
│   ├── mail_router.py           # 附件/邮件资产路由
│   ├── admin_config.py          # MainServer 本地 agent/discovery 配置读写
│   ├── agent_templates.py       # 从 SeedAgent 模板复制创建新 Agent
│   ├── config/                  # 本地运行配置示例；真实本地配置默认不提交
│   └── test_comm.py             # 通讯链路端到端验证
├── tests/                       # 单元测试、集成测试、真实模型流式测试
├── main.py                      # 本地运行入口
├── sandbox.py                   # 通用沙盒实现
└── deepagent_sandbox.Dockerfile
```

> [!IMPORTANT]
> `Deepagents/` 是多 Agent 服务目录，不是单个 Agent 的业务目录。
> 每个 `XxxAgent/` 都是独立服务，单独启动、单独注册、单独上报状态。

---

## 2. Deepagents — Agent 目录规范

### 2.1 SeedAgent 目录树

```text
SeedAgent/
├── workspace/                    # Agent 运行时工作目录
│   ├── brain/                    # Agent 自身行为约束，文件会进入系统提示词
│   ├── knowledge/                # 业务知识文件
│   ├── notes/                    # 笔记
│   └── skills/                   # 技能文件
├── AgentServer/                  # Layer 1：服务外壳
│   ├── __init__.py
│   ├── __main__.py
│   ├── service.py                # FastAPI app + lifespan + MainServer 上报
│   ├── AgentCard.json
│   └── ServiceConfig.json
├── Agent/                        # Layer 2：Agent 组装
│   ├── __init__.py
│   ├── MainAgent.py              # 主 agent 构建
│   ├── SeedAgentConfig.example.json # agent runtime 完整配置示例
│   ├── protocol.py               # 通讯/任务层共享契约
│   ├── Models/                   # init_chat_model 模型 + JSON 模型配置
│   ├── SubAgent/                 # 工具或 middleware 中会用到的 agent
│   ├── middlewares/              # 中间件和中间件配置 JSON
│   ├── tools/                    # agent 工具和工具配置 JSON
│   ├── store/                    # checkpoint 和中间数据存储区域
│   ├── server/                   # Agent 内部公共服务区
│   └── sandbox.py                # 沙盒环境
└── ...
```

### 2.2 分层职责

| 层级 | 目录 | 职责 | 红线 |
|---|---|---|---|
| Layer 1 | `AgentServer/` | 服务生命周期、MainServer 注册、状态上报、异常上报、stream 事件转发 | 不写业务逻辑，不直接做模型调用 |
| Layer 2 | `Agent/` | 构建 agent、挂载模型、middleware、tools、checkpoint，暴露 invoke/stream | 不感知 HTTP 层，不直接请求 MainServer |
| Layer 2.1 | `Agent/Models/` | 模型配置和模型初始化 | 主 agent、工具内部模型、子 agent 模型配置不混用 |
| Layer 2.2 | `Agent/middlewares/` | LangChain `AgentMiddleware` 实现，注册工具、state schema、system message、hook | 不把工具业务逻辑直接写进 middleware |
| Layer 2.3 | `Agent/tools/` | Agent 工具实现 | 一个工具只做一类事情，通过 `ToolRuntime` 获取运行时信息 |
| Layer 2.4 | `Agent/server/` | Agent 内部复用逻辑和声明 | 不放 MainServer 运行时，不做全局调度 |
| Layer 2.5 | `Agent/SubAgent/` | 工具或 middleware 需要用到的 agent / 主 agent 子 agent | 不跨 Agent 直接 import 其他服务 |
| Test | `tests/` | 声明测试、本地行为测试、真实模型流式测试 | 真实模型测试必须观察 stream 过程 |

---

## 3. MainServer — 服务端结构与职责

```text
MainServer/
├── __init__.py
├── main_server.py     # FastAPI app + 注册、状态、事件、消息路由
├── state.py           # 全局状态契约
├── protocol.py        # make_message：消息构造
├── comm.py            # AgentComm：Agent 端通讯 SDK
├── mail_router.py     # 附件/邮件资产路由
├── admin_config.py    # MainServer 本地 agent/discovery 配置读写
├── agent_templates.py # 从 SeedAgent 模板复制创建新 Agent
├── config/            # 本地配置示例；agents.local.json 默认不提交
└── test_comm.py       # 通讯链路端到端验证
```

| 文件 | 职责 | 红线 |
|---|---|---|
| `state.py` | 定义 `AgentCard`, `TaskInfo`, `AgentMail`, `CollaborationState` 等 TypedDict | 这是跨 Agent 公共状态契约，结构变更必须同步评估所有 Agent |
| `protocol.py` | `make_message()`，构造标准 `AgentMail` | 只负责消息构造，不发送 |
| `comm.py` | `AgentComm` 类，提供 `send / recv / peers` | 纯 SDK，不持有状态，不引入 `requests` / `httpx` |
| `mail_router.py` | 路由本地附件到接收方 workspace | 不下载网络资源，不执行 Agent 任务 |
| `admin_config.py` | 读取和写入 MainServer 本地 agent/discovery 配置 | 不导入 Agent 运行时，不保存模型凭据 |
| `agent_templates.py` | 从 SeedAgent 模板复制创建新 Agent 目录 | 不启动模型，不改已有 Agent，除非显式 overwrite |
| `main_server.py` | FastAPI 应用，注册、状态更新、事件收集、消息路由 | 不持有 Agent 运行时，不做任务执行 |

---

## 4. 状态与协议契约

### 4.1 公共状态真相来源

`MainServer/state.py` 是跨 Agent 公共状态契约的真相来源。

`Deepagents/*/Agent/protocol.py` 可以复用或镜像通讯/任务层共享契约，但不应单方面
定义与 `MainServer/state.py` 冲突的结构。

### 4.2 核心类型

```python
MessageType = Literal["task", "message"]

Link              # {"link", "summary"}
Deliverable       # Link 的兼容别名/子类
CapabilityCard    # {"title", "content"}
AgentCard         # {"agent_name", "capabilities"}
TaskInfo          # {"title", "goal", "description", "owner", "deliverables"}
AgentMail         # {"message_id", "from", "to", "type", "content", "attachments"}
AgentMessage      # AgentMail 的兼容别名
AgentMessageBox   # {"inbox", "outbox"}，内部保存 AgentMail
CollaborationState       # {"agentcard", "agent_messages"}
```

约束：
- `AgentMail.from` 字段保持 JSON 协议名 `"from"`，TypedDict 中用函数式声明绕过 Python 保留字。
- `AgentMail.from/to` 均使用 agent name，不再使用 agent id。
- `AgentMail.type` 新增取值时必须同步更新 MainServer、Agent 协议和测试。
- `type="task"` 只是另一种消息格式，不单独维护任务注册表和任务状态字段。
- `CollaborationState` 顶级字段不得随意扩展；需要扩展时先更新约束和测试。

---

## 5. 通讯协议约束

### 5.1 消息路由

| 端点 | 方法 | 描述 |
|---|---|---|
| `POST /send` | 发送 | `to=agent_name` 点对点，当前不做广播 |
| `GET /recv/{agent_name}` | 接收 | 一次性清空信箱，pull 模型，非 push |
| `GET /agents/peers/{agent_name}` | 发现 | 返回当前 agent 可见的 peer |

### 5.2 SDK 约束

- Agent 端只通过 `AgentComm` 调用，不直接 HTTP 请求 MainServer。
- `AgentComm` 依赖 `urllib.request`，不引入 `requests` / `httpx`。
- 默认超时不超过 5 秒。
- `AgentComm` 初始化和收发目标只使用 agent name。
- `send_message_to_agent` 同时负责发送普通信息和任务信息：`msgType="message"` 发送文本，`msgType="task"` 发送 `TaskInfo`。
- 附件通过 `attachments: list[Link]` 发送。工具入口中，本地文件路径必须以 `/workspace` 开头，工具会转换为当前 Agent workspace 下的宿主机路径；网络 URL 可直接传。
- 接收方 workspace 相对位置固定为 `Deepagents/<agent_name>/workspace`，MainServer 不依赖注册 payload 中的 workspace 字段。
- 浏览器自动化是 agent 侧能力，不是 Codex 自身能力；它通过 `workspace/skills/agent-browser/SKILL.md` 接入，Docker sandbox 提供 `agent-browser` + `/usr/bin/chromium`，真实流程由 CLI 的 `skills get core` 提供。
- SeedAgent 记忆能力通过 `memory/` 包接入：`ingest_knowledge_document` 将
  `/workspace/knowledge/...` 文件切分写入记忆图，`manage_knowledge` 管理、查询、
  修正和关联记忆内容。

### 5.3 注册链路

```text
Agent 启动
→ AgentServer lifespan
→ AgentObserver.register()
→ POST /agents/register
→ AgentObserver.set_status("running")
```

每个 Agent 必须在 lifespan 中完成注册，启动完成后状态为 `running`。

### 5.4 Scope 管道限制

scope 由 MainServer 管理，Agent 不在本地私自声明全局可达性。
MainServer 默认读取 `MainServer/config/agents.local.json`，该文件是本地运行配置，
不提交；可通过 `MAIN_SERVER_AGENT_CONFIG` 指向其他配置文件。
仓库只保留 `MainServer/config/agents.example.json` 作为示例。

```python
# scope=None 表示全部可达
# scope=[] 表示与谁都不能通讯
# scope=["agent-b", "agent-c"] 表示只能和指定 agent 通讯
# scope=[[], "agent-b", ["agent-c", []]] 这类嵌套结构会保留原始形态，
# 路由判断时递归解析出其中的 agent name。
```

Scope 影响三个维度：

| 维度 | 规则 |
|---|---|
| 注册 | 注册 payload 可带兼容 scope，但若 MainServer 本地配置中存在该 agent 的 scope，则以 MainServer 配置为准 |
| 发送 | 发送方只能向自己 scope 内的 agent 发消息；scope 外目标返回 403 |
| 接收 | 接收方只保留来自自己 scope 内的消息；scope 外来源被过滤 |
| 发现 | `GET /agents/peers/{agent_name}` 只返回 scope 内已注册 agent，且不包含自己 |

`/agents/online` 不受 scope 影响，仍返回全局列表，作为管理员视角。

MainServer 后端管理接口：

| 端点 | 方法 | 描述 |
|---|---|---|
| `/admin/agents/config` | `GET/PUT` | 读取或替换中心 agent/discovery 配置 |
| `/admin/agents/create` | `POST` | 从 `SeedAgent` 或指定模板复制创建新 Agent 目录，并写入中心配置 |
| `/admin/agents/{agent_name}/config` | `GET/PATCH/PUT` | 读取、合并或替换单个 Agent 的配置 |
| `/admin/agents/{agent_name}/scope` | `GET/PUT` | 读取或替换单个 Agent 的原始 scope，并返回解析后的可达 agent |

---

## 6. Middleware / Tool 约束

当前项目不再使用 `Capabilities/` 作为目标目录。能力边界由
`Agent/middlewares/` + `Agent/tools/` + 配置 JSON 表达。

### 6.1 middleware

- middleware 文件优先单文件结构。
- middleware、tool 和 agent 装配文件开头不要使用 `from __future__ import annotations`；公共协议文件允许使用它来表达新 TypedDict 协议。
- 声明 `MiddlewareCapabilityPrompt`、`MiddlewareRuningConfig`、`MiddlewareToolConfig`、`MiddlewareStateTydict`。
- 暴露 `state_schema`。
- 暴露 `tools`，并由 middleware 自动挂载工具。
- 使用命名 `SystemMessage.name` 作为能力 slot。
- middleware hook 的真实类型要保持可见。
- 中间件内部网络错误等应 fail-silent，不阻断 Agent 主链路。
- 顶层 agent 不维护自由的 `enabledTools` 列表；工具由所属 middleware 挂载。
- `ingest_knowledge_document` 由 `KnowledgeIngestMiddleware` 挂载，不在
  `create_deep_agent(...)` 的 `tools=` 中重复传入。
- 当前主 agent 除自定义 middleware 外，还应接入 deepagents 内置：
  - `MemoryMiddleware`：读取 `workspace/brain/AGENTS.md`
  - `SkillsMiddleware`：读取 `workspace/skills/*/SKILL.md`

middleware 文件内部推荐统一维护：

- `MiddlewareCapabilityPrompt`
- `middleware_capability_prompts`
- `MiddlewareRuningConfig`
- `middleware_runingconfig`
- `MiddlewareToolConfig`
- `MiddlewareStateTydict`
- middleware class

当前 demo 重构补充约束：

- middleware / tool / nested agent 按 `Config` / `SubState` / `Schema`
  / helper / wrapper / default entry 组织。
- 共享的 demo 级配置加载、named system message、stream emit、计数更新 helper
  放在 `Agent/server/demo_server.py`。
- 外层 wrapper 只负责持有当前配置和装配结果；真正挂载到 agent 上的是标准
  middleware / tool 对象。
- 上层 agent 在创建中间键时要把同一份 `context` 继续向下透传。

### 6.2 tool

- tool 文件优先单文件结构。
- middleware、tool 和 agent 装配文件开头不要使用 `from __future__ import annotations`；公共协议文件允许使用它来表达新 TypedDict 协议。
- 声明 `Config`、`SubState`、`ToolInputSm`、`ToolDescriptionSc`、`ToolReturnSc`、`ToolStateTydict`、`ToolRuningConfigSc`。
- 工具函数签名保留 `runtime: ToolRuntime`。
- 工具只从 `runtime.state / runtime.stream_writer / runtime.tool_call_id` 读取运行时信息。
- 工具返回 `Command(update=...)`，并在 `update["messages"]` 中包含 `ToolMessage`。
- 工具报错必须返回详细原因和建议。

tool 文件内部推荐统一维护：

- `ToolInputSm`
- `ToolDescriptionSc`
- `ToolReturnSc`
- `ToolStateTydict`
- `ToolRuningConfigSc`
- `ToolSpec`
- `tool_runingconfig`
- tool class

### 6.3 MainAgent 装配顺序

```text
ReceiveMessagesMiddleware
→ BusinessMiddlewares(*)
→ AgentStepTraceMiddleware
→ [DebugTraceMiddleware]
```

说明：
- `ReceiveMessagesMiddleware` 必须最先挂载，先拉取信箱并追加一条包含 `<Inbox>` 的 user message。
- 业务 middleware 负责挂载自己的提示词、工具、state 和内部运行逻辑。
- `AgentStepTraceMiddleware` 负责 step timing / tool timing 流式事件。
- `DebugTraceMiddleware` 只在 debug 模式挂载。
- 创建 agent 时只挂 middleware，不重复传同一批 tools。

---

## 7. 模型配置约束

Agent runtime 完整配置放在 `Deepagents/*/Agent/SeedAgentConfig.local.json`，
示例为 `SeedAgentConfig.example.json`。该配置包含：

- agent 身份、角色、职责和补充系统提示词。
- chat model、embedding model、API key、base URL。
- Neo4j URI、用户名、密码和 database。
- middleware 开关。
- middleware 所属 tool 的参数，例如 `send_message_to_agent` 默认目标和
  `ingest_knowledge_document` 分片参数。
- `memory/README.md` 中两个公开入口的配置面：
  `tools.ChunkApplyTool` 的 identity/public/runtime/chunking/document edge/keyword embedding
  参数，以及 `middleware.KnowledgeManagerCapabilityMiddleware` 的 trace、内部 manager、
  discovery、document query/write、graph query/write 和 graph query preset 参数。

`SeedAgentConfig.local.json` 包含本地凭据，默认不提交。`AgentServer/ServiceConfig.json`
只保存独立服务配置，例如 `agent_name`、`host`、`port`、`main_server_url`。

旧模型配置放在 `Deepagents/*/Agent/Models/`，只作为兼容配置保留。

当前骨架至少包含：

| 配置文件 | 消费方 | 用途 |
|---|---|---|
| `SeedAgentConfig.local.json` | MainAgent / memory bridge | agent runtime 完整本地配置，默认不提交 |
| `SeedAgentConfig.example.json` | MainAgent / template | agent runtime 完整配置示例 |
| `model_config.json` | 旧兼容 | 旧主 agent 模型结构 |
| `AgentApi.json` | 旧兼容 | 旧 MainAgent API 配置 |
| `ToolModel.json` | tools / SubAgent | 工具或子 agent 可选模型 |

当前不迁入旧项目的 `MemoryManage/*`、RAG、索引构建和记忆召回历史实现；
记忆能力通过 `memory/README.md` 推荐的两个公开入口接入。

---

## 8. 持久化约束

所有 Agent 本地运行时数据放在 `Deepagents/*/Agent/store/` 或
`Deepagents/*/workspace/`。

| 存储 | 路径 | 所有者 | 说明 |
|---|---|---|---|
| LangGraph Checkpoint | `Agent/store/state/` | MainAgent | 线程级会话状态 |
| 工具 / middleware 中间数据 | `Agent/store/**` | Agent 内部组件 | sqlite cache、staging、中间结果 |
| Agent 工作目录 | `workspace/` | Agent 运行时 | 文件读写、业务知识、笔记、技能 |

`Agent/store/` 和 `workspace/` 不提交真实运行内容。

---

## 9. 异常上报约束

`AgentServer` 在 lifespan 中注册两个异常钩子：

1. `sys.excepthook`：同步未捕获异常 → `POST /agents/{id}/error`
2. asyncio loop exception handler：异步未捕获异常 → `POST /agents/{id}/error`

所有 Agent 必须复用此机制。

业务代码不得自行 `sys.exit()`，应抛异常由服务外壳统一捕获上报。

---

## 10. workspace 边界约束

`workspace/` 是 Agent 的本地工作目录，不是框架说明存放地。

每个 Agent 的 sandbox 必须按 `agent_name` 隔离：
- 容器名包含 agent name 和 workspace 路径 hash。
- 宿主机只挂载 `Deepagents/<agent_name>/workspace`。
- 容器内统一挂载为 `/workspace`，供模型和工具使用。
- 不允许把项目根目录或其他 Agent workspace 挂进同一个 Agent sandbox。

| 可以放 | 不可以放 |
|---|---|
| Agent 读写的业务文件 | 服务框架说明 |
| `brain/AGENTS.md` | MainServer 协议文档 |
| notes / skills / knowledge | 跨 Agent 调度逻辑 |

---

## 11. 旧项目到新骨架的迁移映射

旧项目中的 `Capabilities/` 只作为迁移来源参考，不作为新目标目录。

| Source Path | Target Path | Treatment |
|---|---|---|
| `Deepagents/SeedAgent/Agent/MainAgent.py` | `Deepagents/SeedAgent/Agent/MainAgent.py` | 保留位置，按新通用 middleware/tool 约束重写 |
| `Deepagents/SeedAgent/AgentServer/service.py` | `Deepagents/SeedAgent/AgentServer/service.py` | 保留位置，只做生命周期和事件转发 |
| `Deepagents/SeedAgent/Agent/Capabilities/base.py` | `Deepagents/SeedAgent/Agent/middlewares/base.py` | 只迁移通用思想，不保留 `Capabilities` 路径 |
| `Deepagents/SeedAgent/Agent/Capabilities/ReceiveMessages/` | `Deepagents/SeedAgent/Agent/middlewares/receive_messages.py` | 收信注入 middleware |
| `Deepagents/SeedAgent/Agent/Capabilities/SendMessages/` | `Deepagents/SeedAgent/Agent/middlewares/send_messages.py` + `Agent/tools/send_message_tool.py` | 发信 middleware 自动挂 tool |
| `Deepagents/SeedAgent/Agent/Capabilities/AgentStepTrace/` | `Deepagents/SeedAgent/Agent/middlewares/agent_step_trace.py` | step/tool timing trace |
| `Deepagents/SeedAgent/Agent/Capabilities/DebugTrace/` | `Deepagents/SeedAgent/Agent/middlewares/debug_trace.py` | debug trace |
| `Deepagents/SeedAgent/Agent/Capabilities/MemoryManage/*` | 暂不迁移 | 记忆系统本轮不进入骨架 |
| `Deepagents/SeedAgent/Agent/Models/` | 同路径 | 保留模型配置隔离 |
| `MainServer/*` | `MainServer/*` | 保留中心服务路径 |

---

## 12. 初始构建顺序

1. 创建顶层 `Deepagents/`、`MainServer/`、`tests/`。
2. 创建 `MainServer/state.py`、`MainServer/protocol.py`、`MainServer/comm.py` 的可导入版本。
3. 创建 `Deepagents/SeedAgent/Agent/middlewares/base.py`。
4. 创建最小 `receive_messages.py`、`send_messages.py`、`agent_step_trace.py`、`debug_trace.py` middleware 骨架。
5. 创建 `Deepagents/SeedAgent/Agent/tools/send_message_tool.py`。
6. 创建 `Deepagents/SeedAgent/Agent/Models/` 配置和 runtime 占位。
7. 创建 `Deepagents/SeedAgent/Agent/MainAgent.py`，只做 agent 装配。
8. 创建 `Deepagents/SeedAgent/AgentServer/service.py`，只做服务生命周期。
9. 创建三层测试，尤其保留真实模型流式测试入口。

---

## 13. 绝对禁止事项

| # | 禁止事项 | 原因 |
|---|---|---|
| 1 | 在 `AgentServer/` 中执行业务逻辑 | 违反分层，Layer 1 只管服务生命周期 |
| 2 | Agent 绕过 `AgentComm` 直接 HTTP 发消息 | 破坏通讯层统一性 |
| 3 | 在 Agent 内部 `import MainServer.main_server` | MainServer 是独立进程，不可被 Agent import |
| 4 | 单方面修改 `MainServer/state.py` 类型结构 | 必须同步评估所有 Agent 协议和测试 |
| 5 | 在中间件中抛异常中断 Agent | middleware 必须 fail-silent |
| 6 | 把 `Agent/store/` 或 `workspace/` 真实运行内容提交到 git | 运行时产物，仅本地 |
| 7 | 引入 `requests` / `httpx` 作为通讯依赖 | `comm.py` 基于 `urllib.request`，零外部依赖 |
| 8 | 在 `workspace/` 中存放框架文档 | workspace 只服务于 Agent 业务 |
| 9 | 把全局编排 / 调度逻辑放进某一个 Agent | 调度由 MainServer 负责 |
| 10 | 跨 Agent 直接 import 对方代码 | Agent 之间只通过消息通讯 |
| 11 | 新建 `Capabilities/` 作为目标目录 | 当前项目改用 `middlewares/` + `tools/` |
