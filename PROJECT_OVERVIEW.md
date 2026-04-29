# Long River Agent

Long River Agent 是基于 LangChain Deep Agents 搭建的本地多 Agent 编排项目。
项目目标是让特定 agent 负责特定任务，再通过 MainServer 的通讯空间组合成复杂
工作流。每个 agent 都可以保留完整 Deep Agents 能力，并通过独立 workspace 与
sandbox 隔离运行。

项目假设单个 agent 更适合处理边界清晰的具体任务，并不天然擅长持续的信息减商
和元认知自检。因此推荐在同一类任务上反复训练一组 agent：执行 agent 完成任务，
监督 agent 评分反馈，知识 agent 沉淀经验，直到流程稳定地产生好结果后再投入使用。
前端控制台用于长期管理这些 agent、通讯空间、提示词、运行细节和 checkpoint。

顶层沿用 `Deepagents/` + `MainServer/` 这套服务结构，但
`Deepagents/*/Agent/` 内部不再使用旧的 `Capabilities/` 目录，改为当前项目的新约束：

- `middlewares/`：LangChain `AgentMiddleware` 实现和 middleware 配置 JSON。
- `tools/`：agent 工具实现和工具配置 JSON。
- `server/`：Agent 内部公共服务区，放可复用逻辑和声明。
- `SubAgent/`：工具或 middleware 中会用到的 agent，或主 agent 的子 agent。
- 当前中间键和工具正在按 `demo/NESTED_DEMO_FRAMEWORK_CONSTRAINTS.md`
  收拢成 `Config / SubState / Schema / helper / wrapper / default entry`
  的单文件形态；共享的 demo 级 helper 现在放在 `Agent/server/demo_server.py`。

主 agent 的装配入口是 `deepagents.create_deep_agent(...)`。项目尽量保持 LangChain
原生接口：tool 是标准 LangChain tool，middleware 是标准 LangChain
`AgentMiddleware`，自定义能力应优先从 `MainAgent.py` 的 `build_middlewares(...)`
或 `create_deep_agent(...)` 附近接入。

当前有两套基础 seed 模板：
- `SeedAgent`：普通基础 agent，默认不挂文档切分入库、chunkApply 或知识管理工具。
- `KnowledgeSeedAgent`：知识库模板，接入复制到本项目的 `memory/` 包；
  `ingest_knowledge_document` 负责把 `workspace/knowledge` 中的长文档切分进入
  记忆图，`manage_knowledge` 负责查询、整理、修正和关联记忆内容。
  `memory/` 内部是一套围绕文档处理的可变、可修复图 RAG：文档被尽量无损地
  切分为可追溯来源的文档节点，也就是 `Chunk`；agent 还可以写入普通图节点
  `GraphNode`，并在 Neo4j 中管理、发现、思考、补边、修复和生成新的文档结构。
旧项目中的 `MemoryManage/*`、RAG、索引构建、记忆召回不直接迁入骨架。

## 目录摘要

```text
Long River Agent/
├── Deepagents/<XxxAgent>/
│   ├── workspace/
│   ├── AgentServer/
│   └── Agent/
│       ├── Models/
│       ├── SubAgent/
│       ├── middlewares/
│       ├── tools/
│       ├── store/
│       ├── server/
│       ├── MainAgent.py
│       ├── <XxxAgent>Config.example.json
│       ├── protocol.py
│       └── sandbox.py
├── MainServer/
├── frontend/                    # React 管理台
├── tests/
└── project docs
```

完整结构和红线以 `constraint.MD` 为准；其他文档只保留稳定摘要，避免目录细节频繁改动。

## 分层职责

- `Deepagents/`
  - 多 Agent 服务目录，不是单个 Agent 的业务目录。
  - 每个 `XxxAgent/` 都是独立服务，单独启动、单独注册、单独上报状态。

- `Deepagents/<XxxAgent>/AgentServer/`
  - Layer 1，服务外壳。
  - 负责 FastAPI 生命周期、MainServer 注册、状态上报、异常上报和 stream 事件转发。
  - `ServiceConfig.json` 管独立服务启动配置，例如 `agent_name`、`host`、`port`、
    `main_server_url`。
  - 不写业务逻辑，不直接做模型调用。

- `Deepagents/<XxxAgent>/Agent/`
  - Layer 2，Agent 组装层。
  - `MainAgent.py` 负责构建主 agent，挂载模型、backend、middleware、checkpoint，
    并把当前 agent 的 `Config` 作为 context schema 透传给 middleware/tool。
  - `<XxxAgent>Config.local.json` 是完整本地 runtime 配置，包含 chat/embedding/Neo4j
    凭据、middleware 开关和 tool 参数；真实本地配置默认不提交。
    `<XxxAgent>Config.example.json` 是仓库内的可运行模板。`KnowledgeSeedAgent` 的配置覆盖
    `memory/README.md` 推荐公开入口 `ChunkApplyTool` 与
    `KnowledgeManagerCapabilityMiddleware` 的主要 JSON 配置面。
  - `protocol.py` 放通讯/任务层共享契约。
  - `Models/` 放模型初始化和旧兼容模型配置。
  - `middlewares/` 放中间件和中间件配置 JSON。
  - `tools/` 放 agent 工具和工具配置 JSON。
  - `server/` 放 Agent 内部可复用逻辑和声明。
  - `store/` 放 checkpoint 和中间数据，不提交真实运行内容。
  - `Models/model_config.json` 保留为模型结构展示和旧入口兼容；当前 AgentServer
    主要读取 runtime config 中的 `chatBaseUrl/chatApiKey` 等字段。

- `MainServer/`
  - 中心化注册、状态同步、事件收集、消息路由服务。
  - 不持有 Agent 运行时，不做任务执行。
  - Agent 端只通过 `AgentComm` 通讯，不直接 HTTP 请求 MainServer。
  - Agent 身份只使用 `agent_name`；`AgentMail.type="task"` 只是另一种消息格式。
  - 附件使用 `Link`，工具内本地文件路径必须以 `/workspace` 开头，网络 URL 可直接传。
  - 面向用户/前端的聊天入口是 `POST /user/chat`。`mode=direct` 会把文本、
    content blocks、图片 URL/data URL 等归一成 user message 并代理目标
    AgentServer `/invoke`；`mode=mail` 会构造 `AgentMail` 写入目标 inbox。
    邮件投递后，MainServer 会在目标 AgentServer 在线时自动触发或排队重试一轮
    mail wake invocation，让 `receive_messages` 中间件立刻拉取 inbox。
    AgentServer 注册时应把 `service_url` 写入 metadata，便于 MainServer 代理。
  - 前端可通过 `GET/PUT /user/chat/config/{agent_name}` 独立管理默认
    `thread_id`、`run_id`、`stream_mode` 和 `version`；请求级参数会覆盖默认配置。
    主 agent 内部会把 `run_id` 和 `thread_id` 拼成真实 checkpoint key：
    `<run_id>:<thread_id>`，所以二者共同决定对话 checkpoint 隔离。
  - 邮件唤醒使用接收方用户聊天配置中的主 `thread_id/run_id`；邮件 metadata 中的
    `thread_id/run_id` 只用于审计和 UI 展示，不覆盖接收方 checkpoint。
    `receive_messages` 会把 `<Inbox>` 写入 state messages，使邮件内容进入 sqlite
    checkpoint，而不是只在当轮模型调用里临时可见。
  - 前端或管理接口写入 Agent runtime config 后，MainServer 会在 AgentServer 在线时
    调用 `/reload-config`，让内存中的主 agent 重新读取配置。
  - 主 agent 对话 checkpoint 默认落在
    `Agent/store/checkpoints/langgraph.sqlite3`；runtime config 的 `checkpointPath`
    可覆盖该位置。清 checkpoint 后 MainServer 会请求 AgentServer 重载，避免
    sqlite 连接继续握着已删除的旧文件句柄。
  - `/admin/agents/{agent_name}/config` 是合并配置视图：中心字段仍存在
    `MainServer/config/agents.local.json`，模型/API/Neo4j/middleware/tool 字段会写回
    Agent runtime config。
  - agent 发现和通讯关系由 MainServer 的全局 `communication.spaces` 管理；默认本地配置是
    `MainServer/config/agents.local.json`，示例为 `MainServer/config/agents.example.json`。
  - 每个 space 是一组可互相通讯的 agent；多个 space 可以重叠，通讯边由“是否至少
    共享一个 space”推导。没有配置任何 space 时，兼容为已注册 agent 全可达。
  - 通讯空间是当前主要编排机制；不要把跨 agent 的全局流程硬写进某一个 agent。
  - 后端管理接口支持读取/修改中心配置、保留 legacy scope 兼容接口、读取/修改
    给其他 agent 看的 `AgentServer/AgentCard.json`，以及真正进入当前 agent 上下文的
    `workspace/brain/AGENTS.md`，并从任意已有 Agent 目录复制创建新 agent 目录；
    `SeedAgent` 是普通基础模板，`KnowledgeSeedAgent` 是带 chunkApply/知识管理工具的知识模板。
  - 复制新 agent 时不复制 `Agent/store`、mail、pycache，以及 `workspace` 中的业务文件、
    notes 和 knowledge 内容；只保留 `workspace/brain/AGENTS.md` 与 `workspace/skills/**`。
    也可通过
    `POST /admin/agents/{agent_name}/runtime/clear` 手动清理本地缓存和临时数据库。
- `frontend/`
  - React + Vite 管理台，默认通过 `/api` proxy 连接本地 MainServer。
  - 支持复制任意已有 Agent 来新建 Agent、删除 Agent、用图编辑全局通讯空间、持久化 agent 卡片位置、查看邮件和
    运行状态、改 Agent runtime 配置、改公开 `AgentCard.json`、改顶层入口
    `brain/AGENTS.md`、配置独立 `thread_id/run_id`、并进入 Agent 对话。
  - Agent 图中同一空间内的成员自动连线；框选图上的多个 Agent 会加入当前空间。
  - 对话内容缓存在 MainServer `ui.chat_sessions`，切换页面或重新点开 agent 后不会丢失。
  - 对话页把聊天和细节分成两列：聊天列只显示用户和 agent 回复，细节列显示工具调用、
    邮件和 Agent 行为卡片，不裸露完整 status/update payload。
  - “清缓存”按钮只清 agent checkpoint 数据，并清空该 agent 的前端历史对话缓存；
    不删除 mail、knowledge 或 memory cache。
  - UI 中的 `thread_id/run_id` 控制对话 invocation 和 checkpoint 隔离；Agent
    记忆图身份仍由 runtime config 的 `knowledgeRunId` 管理，换 `knowledgeRunId`
    等于切换知识库视角。
- `workspace/skills/agent-browser/SKILL.md`
  - agent 侧浏览器自动化入口。
  - 只是 discovery stub，真正的浏览器流程由 `agent-browser skills get core` 提供。
- `memory/`
  - 复制进当前项目的记忆管理包。
  - 当前通过 `KnowledgeSeedAgent` 的 bridge 使用公开入口 `ChunkApplyTool` 和
    `KnowledgeManagerCapabilityMiddleware`。
  - `ChunkApplyTool` 负责读取单文件、切分、缓存、恢复和写入 Neo4j；
    `KnowledgeManagerCapabilityMiddleware` 负责挂载 `manage_knowledge(target)`，
    由内部 manager agent 使用 document / graph 工具修复和管理知识图。

## Tool / Middleware 约束

- tool 和 middleware 都优先使用单文件结构。
- middleware、tool 和 agent 装配文件开头不要使用 `from __future__ import annotations`；公共协议文件可使用它来表达新 TypedDict 协议。
- `runtime: ToolRuntime` 和 middleware hook 的真实类型要保持可见。
- 工具由 middleware 自动注册，不要在 `create_agent(...)` 里重复传同一批 tools。
- 顶层 agent 不维护自由的 `enabledTools` 列表；启用 middleware 后由该 middleware
  挂载自己的工具。
- `ingest_knowledge_document` 由 `KnowledgeIngestMiddleware` 挂载。
- 相同的一组 tool，尤其是共享统一 state 的 tool，应由同一个 middleware 统一挂载。
- system message 使用稳定的 `name` 做能力 slot。
- tool 返回 `Command(update=...)`，并在 `update["messages"]` 中放 `ToolMessage`。
- middleware 内部网络错误等应 fail-silent，不阻断 Agent 主链路。
- 浏览器自动化不要做成 Codex 自身能力；它是 agent workspace skill，运行在 sandbox Docker 中。

## 测试注意事项

测试分三层：

1. 声明测试
   - middleware 是否暴露 `state_schema`
   - middleware 是否暴露 `tools`
   - tool 是否暴露标准 LangChain tool
   - tool / middleware 的运行配置是否有默认值

2. 本地行为测试
   - 工具调用是否返回 `Command(update=...)`
   - `Command.update["messages"]` 中是否包含 `ToolMessage`
   - state 更新字段是否和声明一致
   - middleware hook 是否返回预期 state update

3. 真实模型流式测试
   - 必须逐块消费 `agent.stream(...)` 或 `agent.astream(...)`
   - 必须观察模型是否真的发起 tool call
   - 必须观察 tool writer 是否输出关键事件
   - 必须观察 middleware writer 是否输出 hook trace
   - 不可以只断言最终回答文本

推荐测试命令：

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```
