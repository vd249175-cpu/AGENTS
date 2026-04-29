# Long River Agent 前端

这是本地 MainServer 的 React + Vite 控制台。

前端的职责比较薄，主要做这些事情：

- 读取和写入 MainServer 状态
- 查看和编辑 Agent 卡片
- 查看和编辑 Agent 顶层提示词
- 修改 Agent 运行时配置
- 编辑全局通讯图 / 通讯空间
- 查看聊天、邮件和运行细节
- 进入单个 Agent 对话

`/api/*` 会代理到 `http://127.0.0.1:8000`。

## 启动

仓库根目录一键启动：

```bash
./scripts/start_langvideo.sh
```

如果只想单独启动前端：

```bash
cd frontend
npm install
npm run dev
```

浏览器地址：

```text
http://127.0.0.1:5173
```

## 页面说明

- 创建页可以从任意已存在的 Agent 目录复制。
- `SeedAgent` 是普通基础模板。
- `KnowledgeSeedAgent` 是知识模板，带文档切分入库和知识管理能力。
- 提示词页把公开的 `AgentServer/AgentCard.json` 和真正进入上下文的 `workspace/brain/AGENTS.md` 分开。
- 配置页展示的是 MainServer 和 Agent runtime 的合并视图；模型服务类字段会写回 runtime 配置，通讯和 UI 类字段留在 MainServer 配置里。
- 通讯图页编辑全局通讯空间。空间可以重叠，效果类似维恩图：共享成员可以跨多个空间通讯，不共享的成员不会自动互通。
- 聊天页会保存历史会话，并把工具调用和 Agent 行为放到右侧细节栏，不直接裸露完整状态 payload。
- 保存运行时配置后，如果对应 AgentServer 在线，会触发 `/reload-config` 让主 agent 重新加载配置。
- 清空 checkpoint 后也会触发重载，保证 LangGraph sqlite checkpointer 重新连接到 `Agent/store/checkpoints/langgraph.sqlite3`。
- “清缓存”只清 checkpoint 和前端保存的聊天历史，不会删邮件、知识或 memory cache。
