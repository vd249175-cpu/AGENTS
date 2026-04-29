# LANGVIDEO Agents

这是一个本地多 Agent 框架项目，核心分成三部分：

- `MainServer/`：负责注册、路由、通讯图、管理接口和 UI 状态。
- `Deepagents/`：每个 `XxxAgent/` 都是独立 Agent 服务，包含自己的 `AgentServer/`、`Agent/` 和 `workspace/`。
- `frontend/`：React 控制台，用来创建、配置、查看和对话 Agent。

当前基础模板分工如下：

- `SeedAgent`：普通基础 Agent，不默认挂知识入库或知识管理工具。
- `KnowledgeSeedAgent`：知识模板，挂文档切分入库和知识管理能力。

## 环境要求

- Python `3.12+`
- Node.js 和 npm
- `uv`
- 本地 Neo4j 服务
- 本地可用的 Docker 环境，用于 Agent sandbox

默认模型配置采用 OpenAI 风格接口。如果你要直接用这些默认值，需要自己填 API Key 和 Base URL；也可以切换成仓库里支持的其他模型后端。

## 安装

在仓库根目录执行：

```bash
uv sync
cd frontend
npm install
```

## 配置文件

常用的本地配置文件有：

- `MainServer/config/agents.local.json`
- `Deepagents/<AgentName>/Agent/<AgentName>Config.local.json`
- `Deepagents/<AgentName>/AgentServer/ServiceConfig.json`
- `Deepagents/<AgentName>/Agent/Models/model_config.json`

其中：

- `*.example.json` 是可复制模板。
- `*.local.json` 保存本地端口、模型凭据、Neo4j 凭据、checkpoint 路径、中间键和工具开关，默认不提交。

常见环境变量覆盖：

- `MAIN_SERVER_AGENT_CONFIG`：MainServer 配置文件路径
- `LANGVIDEO_AGENT_SERVICE_CONFIG`：AgentServer 服务配置路径
- `LANGVIDEO_SEED_AGENT_CONFIG`：SeedAgent 运行时配置路径
- `LANGVIDEO_KNOWLEDGE_SEED_AGENT_CONFIG`：KnowledgeSeedAgent 运行时配置路径
- `LANGVIDEO_AGENT_MODEL_CONFIG`：模型配置路径
- `LANGVIDEO_OPENAI_MODEL`、`LANGVIDEO_OPENAI_BASE_URL`、`LANGVIDEO_OPENAI_API_KEY`
- `LANGVIDEO_OLLAMA_MODEL`、`LANGVIDEO_MODEL_PROVIDER`

## 启动

仓库根目录一键启动：

```bash
./scripts/start_langvideo.sh
```

这个脚本会启动 MainServer、前端和默认 Agent，并自动打开浏览器。日志默认写到 `tests/logs/dev-start/`。

脚本常用环境变量：

- `LANGVIDEO_HOST`：默认 `127.0.0.1`
- `LANGVIDEO_MAIN_PORT`：默认 `8000`
- `LANGVIDEO_FRONTEND_PORT`：默认 `5173`
- `LANGVIDEO_LOG_DIR`：默认 `tests/logs/dev-start`
- `LANGVIDEO_START_AGENTS`：默认 `1`
- `LANGVIDEO_OPEN_BROWSER`：默认 `1`
- `LANGVIDEO_AGENTS`：默认 `KnowledgeSeedAgent SeedAgent`
- `LANGVIDEO_PYTHON`：可选，自定义 Python 可执行文件

手动启动方式：

```bash
uv run python main.py mainserver
```

```bash
cd frontend
npm run dev
```

`main.py seedagent` 只会启动 `SeedAgent` 服务。`KnowledgeSeedAgent` 通常通过管理接口或一键启动脚本启动。

## 访问地址

- MainServer：`http://127.0.0.1:8000`
- Frontend：`http://127.0.0.1:5173`

## 建议先看

- `constraint.MD`：当前结构和红线
- `PROJECT_OVERVIEW.md`：模块总览
- `frontend/README.md`：前端说明
