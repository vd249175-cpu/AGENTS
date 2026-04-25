# LANGVIDEO Agents

本项目是新的多 Agent 框架骨架。顶层沿用 `Deepagents/` + `MainServer/`
这套服务结构，但 `Deepagents/*/Agent/` 内部不再使用旧的 `Capabilities/`
目录，改为当前项目的新约束：

- `middlewares/`：LangChain `AgentMiddleware` 实现和 middleware 配置 JSON。
- `tools/`：agent 工具实现和工具配置 JSON。
- `server/`：Agent 内部公共服务区，放可复用逻辑和声明。
- `SubAgent/`：工具或 middleware 中会用到的 agent，或主 agent 的子 agent。
- 当前中间键和工具正在按 `demo/NESTED_DEMO_FRAMEWORK_CONSTRAINTS.md`
  收拢成 `Config / SubState / Schema / helper / wrapper / default entry`
  的单文件形态；共享的 demo 级 helper 现在放在 `Agent/server/demo_server.py`。

当前已开始接入复制到本项目的 `memory/` 包。第一步接在 `SeedAgent`：
`ingest_knowledge_document` 负责把 `workspace/knowledge` 中的长文档切分进入
记忆图，`manage_knowledge` 负责查询、整理、修正和关联记忆内容。旧项目中的
`MemoryManage/*`、RAG、索引构建、记忆召回不直接迁入骨架。

## 目录摘要

```text
LANGVIDEO/
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
│       ├── SeedAgentConfig.example.json
│       ├── protocol.py
│       └── sandbox.py
├── MainServer/
├── tests/
└── project docs
```

完整结构和红线以 `constraint.MD` 为准；其他文档只保留稳定摘要，避免目录细节频繁改动。

## 分层职责

- `Deepagents/`
  - 多 Agent 服务目录，不是单个 Agent 的业务目录。
  - 每个 `XxxAgent/` 都是独立服务，单独启动、单独注册、单独上报状态。

- `Deepagents/SeedAgent/AgentServer/`
  - Layer 1，服务外壳。
  - 负责 FastAPI 生命周期、MainServer 注册、状态上报、异常上报和 stream 事件转发。
  - `ServiceConfig.json` 管独立服务启动配置，例如 `agent_name`、`host`、`port`、
    `main_server_url`。
  - 不写业务逻辑，不直接做模型调用。

- `Deepagents/SeedAgent/Agent/`
  - Layer 2，Agent 组装层。
  - `MainAgent.py` 负责构建主 agent，挂载模型、backend、middleware、checkpoint，
    并把 `SeedAgentConfig` 作为 context schema 透传给 middleware/tool。
  - `SeedAgentConfig.local.json` 是完整本地 runtime 配置，包含 chat/embedding/Neo4j
    凭据、middleware 开关和 tool 参数；它覆盖 `memory/README.md` 推荐公开入口
    `ChunkApplyTool` 与 `KnowledgeManagerCapabilityMiddleware` 的主要 JSON 配置面。
    真实本地配置默认不提交；`SeedAgentConfig.example.json` 是可复制模板。
  - `protocol.py` 放通讯/任务层共享契约。
  - `Models/` 放模型初始化和旧兼容模型配置。
  - `middlewares/` 放中间件和中间件配置 JSON。
  - `tools/` 放 agent 工具和工具配置 JSON。
  - `server/` 放 Agent 内部可复用逻辑和声明。
  - `store/` 放 checkpoint 和中间数据，不提交真实运行内容。

- `MainServer/`
  - 中心化注册、状态同步、事件收集、消息路由服务。
  - 不持有 Agent 运行时，不做任务执行。
  - Agent 端只通过 `AgentComm` 通讯，不直接 HTTP 请求 MainServer。
  - Agent 身份只使用 `agent_name`；`AgentMail.type="task"` 只是另一种消息格式。
  - 附件使用 `Link`，工具内本地文件路径必须以 `/workspace` 开头，网络 URL 可直接传。
  - agent 发现和通讯 scope 由 MainServer 中心配置管理；默认本地配置是
    `MainServer/config/agents.local.json`，示例为 `MainServer/config/agents.example.json`。
  - scope 可以是 `None`、扁平 agent name 列表，也可以是嵌套列表；MainServer
    保存原始结构，路由判断时递归解析出可达 agent name。
  - 后端管理接口支持读取/修改中心配置、读取/修改单个 agent scope，并从
    `SeedAgent` 模板复制创建新 agent 目录。
- `workspace/skills/agent-browser/SKILL.md`
  - agent 侧浏览器自动化入口。
  - 只是 discovery stub，真正的浏览器流程由 `agent-browser skills get core` 提供。
- `memory/`
  - 复制进当前项目的记忆管理包。
  - 当前通过 SeedAgent 的 bridge 使用公开入口 `ChunkApplyTool` 和
    `KnowledgeManagerCapabilityMiddleware`。

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
