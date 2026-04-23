# Codex Working Notes

## Current Work

当前正在搭建新的 LANGVIDEO 多 Agent 框架骨架。

旧项目迁移来源：
- `/Users/apexwave/Desktop/Projects/AGENTS`

当前新项目目录：
- `/Users/apexwave/Desktop/Agents`

已确认的方向：
- 顶层继续使用 `Deepagents/` + `MainServer/`。
- `Deepagents/*/Agent/` 内部不再使用旧的 `Capabilities/` 作为目标目录。
- `Agent/` 内部采用 `Models/`、`SubAgent/`、`middlewares/`、`tools/`、`store/`、`server/`、`sandbox.py`。
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
