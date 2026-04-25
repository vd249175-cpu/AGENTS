# Memory

`memory` 是一个面向 LangChain agent 的知识库管理包。当前对外只推荐两个入口：

- `tools.ChunkApplyTool`：把单个文件读取、切分并写入 Neo4j。
- `middleware.KnowledgeManagerCapabilityMiddleware`：给主 agent 挂载 `manage_knowledge(target)`，由内部 manager agent 管理文档和图数据。

这两个入口现在都采用 demo 风格 wrapper：实例只负责配置和装配，真正交给 LangChain 的标准对象必须通过 `.tool` 或 `.middleware` 取得。README 和 examples 只展示这种新封装用法。

document / graph 的细粒度工具仍然保留在源码里，但它们是内部 manager agent 使用的能力面，不作为外部主入口推荐。

## 1. 安装

项目使用 `uv`：

```bash
uv sync
```

核心依赖见 [pyproject.toml](/Users/apexwave/Desktop/memory/pyproject.toml:1)，包括 `langchain`、`langgraph`、`neo4j`、`llama-index-core`、`langchain-openai` 和 `langchain-ollama`。

## 2. 配置文件

对外只需要直接关注两个配置文件：

- [tools/chunk_apply.json](/Users/apexwave/Desktop/memory/tools/chunk_apply.json:1)
- [middleware/knowledge_manager.json](/Users/apexwave/Desktop/memory/middleware/knowledge_manager.json:1)

其它 tool / middleware 的 json 文件服务内部能力，不是推荐的外部配置入口。

### 2.1 `tools/chunk_apply.json`

默认结构：

```json
{
  "identity": {
    "base_run_id": "chunk-run",
    "base_thread_id": "chunk-thread",
    "derive_document_run_id": true
  },
  "public": {
    "neo4j": {
      "uri": "neo4j://localhost:7687",
      "username": "neo4j",
      "password": "1575338771",
      "database": null
    },
    "checkpoint_path": "store/checkpoint/chunk_checkpoint.sqlite3"
  },
  "runtime": {
    "resume": true,
    "cache_path": "store/cache/chunk_cache.sqlite3",
    "staging_path": "store/staging/chunk_staging.sqlite3",
    "recursion_limit": null,
    "shard_count": 4,
    "reference_bytes": 6000,
    "max_retries": 3,
    "max_workers": 2
  },
  "chunking": {
    "history_line_count": 4,
    "active_line_count": 8,
    "preview_line_count": 4,
    "line_wrap_width": 30,
    "window_back_bytes": 1200,
    "window_forward_bytes": 2400,
    "trace_limit": 16,
    "max_retries": 3
  },
  "document_edge_distance": 0.3,
  "persist_keyword_embeddings": true
}
```

字段说明：

- `identity.base_run_id`：知识库级身份前缀。同一套知识库导入的文档应共用它。
- `identity.base_thread_id`：chunk workflow 会话身份前缀，主要用于 checkpoint 和内部执行身份派生。
- `identity.derive_document_run_id`：默认 `true`，会按文档名派生文档级 run_id；作为 Agent 统一知识库使用时可设为 `false`，让导入和 `manage_knowledge` 共用同一个 run_id。
- `public.neo4j.uri`：Neo4j 连接地址，例如 `neo4j://localhost:7687`。
- `public.neo4j.username`：Neo4j 用户名。
- `public.neo4j.password`：Neo4j 密码。
- `public.neo4j.database`：Neo4j database，`null` 表示使用默认库。
- `public.checkpoint_path`：workflow checkpoint SQLite 路径，用于失败恢复。
- `public.embedding_provider`：可选，关键词 embedding provider，例如 `openai` 或 `ollama`。不填时读取模型默认配置。
- `public.embedding_model`：可选，关键词 embedding 模型名，例如 `text-embedding-3-small`。
- `public.embedding_base_url`：可选，embedding 服务地址。
- `public.embedding_api_key`：可选，embedding 鉴权密钥，本地 Ollama 可以留空。
- `public.embedding_dimensions`：可选，embedding 维度。它会参与关键词向量 profile 一致性判断。
- `runtime.resume`：默认是否允许恢复同一文档的未完成执行。
- `runtime.cache_path`：chunk cache SQLite 路径。
- `runtime.staging_path`：chunk staging SQLite 路径。
- `runtime.recursion_limit`：长文档并行工作流预留字段；当前不会改变执行步数。
- `runtime.shard_count`：默认长文档分片数量，默认 `4`；小于 `reference_bytes` 的文档不会强行分片。
- `runtime.reference_bytes`：长文档 shard 参考窗口字节数，默认 `6000`。
- `runtime.max_retries`：workflow 级重试次数。
- `runtime.max_workers`：默认并行 worker 上限，默认 `2`；实际不会超过本次 shard 数量。
- `chunking.history_line_count`：Passed 区域显示多少行历史，不带行号。
- `chunking.active_line_count`：当前切分窗口显示多少行，带行号且首行从 `0` 开始。
- `chunking.preview_line_count`：前瞻区域显示多少行，不带行号。
- `chunking.line_wrap_width`：窗口显示每行最大字符数，默认 `30`。
- `chunking.window_back_bytes`：历史窗口最多回看多少字节。
- `chunking.window_forward_bytes`：当前窗口和前瞻最多读取多少字节。
- `chunking.trace_limit`：chunking 运行 trace 保留条数。
- `chunking.max_retries`：单次 `split_chunk` 重试次数。
- `document_edge_distance`：文档内部 `DOCUMENT_NEXT` 边默认距离，默认 `0.3`。
- `persist_keyword_embeddings`：是否把关键词向量写入 `KeywordNode.embedding`。

### 2.2 `middleware/knowledge_manager.json`

默认结构：

```json
{
  "neo4j": {
    "uri": "neo4j://localhost:7687",
    "username": "neo4j",
    "password": "1575338771",
    "database": null
  },
  "run_id": null,
  "trace_limit": 16,
  "tool": {
    "temperature": 0.0,
    "debug": false,
    "stream_inner_agent": false,
    "inner_recursion_limit": 64,
    "agent_overrides": {
      "model": {},
      "embedding": {},
      "discovery": {},
      "document_query": {},
      "document_write": {},
      "graph_query": {
        "capability_preset": {}
      },
      "graph_write": {}
    }
  }
}
```

字段说明：

- `neo4j.uri`：知识管理中间键默认连接的 Neo4j 地址。
- `neo4j.username`：Neo4j 用户名。
- `neo4j.password`：Neo4j 密码。
- `neo4j.database`：Neo4j database，`null` 表示使用默认库。
- `run_id`：知识库隔离 id。内部 document / graph 工具都会沿用它读写同一套数据。正式使用建议显式传入。
- `trace_limit`：中间键自己的 trace 保留条数。
- `tool.temperature`：内部 manager agent 的默认温度。
- `tool.debug`：内部 manager agent debug 开关。
- `tool.stream_inner_agent`：是否把内部 manager agent 的 updates 透传成外层 custom event。
- `tool.inner_recursion_limit`：内部 manager agent 最大步数。
- `tool.agent_overrides.model`：覆盖内部 manager agent 的 chat model。
- `tool.agent_overrides.embedding`：覆盖内部 document / graph 写入与召回共用的 embedding。
- `tool.agent_overrides.discovery`：覆盖管理发现桶容量和扫描策略。
- `tool.agent_overrides.document_query`：覆盖内部 document query middleware。
- `tool.agent_overrides.document_write`：覆盖内部 document write middleware。
- `tool.agent_overrides.graph_query.capability_preset`：覆盖关键词召回、距离召回、useful / blocked 桶上限和 graph query embedding。常用字段包括 `keyword_top_k`、`keyword_top_k_limit`、`distance_top_k`、`distance_top_k_limit`、`distance_max_distance`、`useful_max_items`、`useful_max_total_chars`、`blocked_max_items`、`blocked_max_total_chars`、`embedding_provider`、`embedding_model`、`embedding_base_url`、`embedding_api_key`、`embedding_dimensions`。
- `tool.agent_overrides.graph_write`：覆盖内部 graph write middleware。

召回工具会把当前 `top_k_limit` 注入到工具 description 中。agent 请求超过上限时，工具返回 `status="error"`、`requested_top_k`、`top_k_limit` 和 `suggested_top_k`，不会抛出异常中断整条 manager agent 链路。

## 3. 模型配置

默认模型配置在 [models/model_config.json](/Users/apexwave/Desktop/memory/models/model_config.json:1)，读取程序在 [models/model_registry.py](/Users/apexwave/Desktop/memory/models/model_registry.py:1)。

可用函数：

- `get_chat_model_config()`
- `get_embedding_model_config()`
- `resolve_embedding_model_config()`
- `list_available_models()`

涉及 embedding 时，`provider`、`model`、`dimensions` 会写入关键词节点，用于判断切换模型后是否需要重新 embedding。

## 4. 文件读取规则

`ChunkApplyTool` 不使用固定后缀白名单。

实际读取顺序：

1. `.txt` 和 `.md` 用 `Path.read_text(encoding="utf-8")` 直接读取。
2. 其它单文件用 `llama_index.core.SimpleDirectoryReader(input_files=[...]).load_data()` 抽取文本。
3. 如果抽取出多个 `Document`，会把非空文本用 `\n\n` 拼接。

当前支持单文件路径，不支持目录路径和一次传多个文件。非文本文件能否成功读取，取决于当前环境中 LlamaIndex reader 的能力。

## 5. `ChunkApplyTool`

`ChunkApplyTool` 是 wrapper。外界实例化 wrapper 后，只把 `chunk_apply_wrapper.tool` 交给 LangChain agent。工具运行时通过 `ToolRuntime` 读取 context，返回 `Command(update=...)`，因此 README 不展示裸调用 wrapper 或裸调用工具函数。

工具业务入参只有：

- `path: str`
- `resume: bool = True`
- `chunking_requirement: str | None = None`
- `shard_count: int | None = None`
- `max_workers: int | None = None`
- `reference_bytes: int | None = None`

`run_id` / `thread_id` 不属于工具业务入参，它们来自构造配置和 agent 运行配置。`shard_count` / `max_workers` / `reference_bytes` 是本次调用级别的长文档调参。

最小组装方式：

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from tools import ChunkApplyTool, ChunkApplyToolConfig

chunk_apply_config = ChunkApplyToolConfig.load_config_chunk_apply_tool({
    "identity": {"base_run_id": "demo-kb", "base_thread_id": "demo-kb-thread"},
    "public": {
        "neo4j": {
            "uri": "neo4j://localhost:7687",
            "username": "neo4j",
            "password": "1575338771",
            "database": None
        },
        "checkpoint_path": "/Users/apexwave/Desktop/memory/store/checkpoint/demo_chunk.sqlite3",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "embedding_base_url": "https://api.openai.com/v1",
        "embedding_api_key": "YOUR_API_KEY",
        "embedding_dimensions": 1536
    },
    "runtime": {
        "resume": True,
        "cache_path": "/Users/apexwave/Desktop/memory/store/cache/demo_chunk.sqlite3",
        "staging_path": "/Users/apexwave/Desktop/memory/store/staging/demo_chunk.sqlite3",
        "shard_count": 4,
        "reference_bytes": 6000,
        "max_retries": 3,
        "max_workers": 2
    },
    "chunking": {
        "history_line_count": 4,
        "active_line_count": 8,
        "preview_line_count": 4,
        "line_wrap_width": 30,
        "window_back_bytes": 1200,
        "window_forward_bytes": 2400,
        "trace_limit": 16,
        "max_retries": 3
    },
    "document_edge_distance": 0.3,
    "persist_keyword_embeddings": True
})

chunk_apply_wrapper = ChunkApplyTool(config=chunk_apply_config)
model = init_chat_model(model="gpt-5.4-nano", model_provider="openai", temperature=0.0)
agent = create_agent(
    model=model,
    tools=[chunk_apply_wrapper.tool],
    checkpointer=InMemorySaver(),
    context_schema=type(chunk_apply_config),
    system_prompt="你是文件入库 agent。用户要求入库文件时，必须调用 chunk_apply。",
)

for event in agent.stream(
    {
        "messages": [{
            "role": "user",
            "content": (
                "请调用 chunk_apply 入库 /Users/apexwave/Desktop/memory/workspace/knowledge/demo.txt。"
                "resume=false，chunking_requirement=按语义完整段落切分，"
                "shard_count=4，max_workers=2，reference_bytes=6000。"
            )
        }]
    },
    config={"configurable": {"thread_id": "demo-kb-thread"}, "recursion_limit": 80},
    context=chunk_apply_config,
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(event)

chunk_apply_wrapper.close()
```

真实行为：

- 文档名统一使用去后缀文件名。
- 同一知识库下，`a.txt` 和 `a.md` 会视为同一个 `document_name=a`。
- `resume=False` 且同名文档已存在时会拒绝。
- `resume=True` 时允许恢复未完成执行；如果 completed cache 的源文件 fingerprint 未变化且文档已入库，会直接返回 Neo4j 现有 chunks。
- `resume=True` 且源文件 fingerprint 已变化时，不会用旧 cache 覆盖已入库文档，会返回“当前文档已经在memory中”。
- `shard_count > 1` 且文档超过 `reference_bytes` 时，会按行边界拆成多个 shard 并发切分，最后按原文位置合并。
- 写入 Neo4j 后自动维护当前文档内部 `DOCUMENT_NEXT`。
- `DOCUMENT_NEXT.dist` 默认 `0.3`。
- 关键词默认写入 `KeywordNode.embedding`。

## 6. `KnowledgeManagerCapabilityMiddleware`

`KnowledgeManagerCapabilityMiddleware` 也是 wrapper。外界实例化 wrapper 后，只把 `knowledge_manager_wrapper.middleware` 交给 `create_agent(...)`。中间键会给主 agent 挂载一个工具：

```python
manage_knowledge(target: str)
```

只有 `target` 是业务入参。`run_id`、Neo4j、模型、embedding、trace、内部递归上限等都通过构造配置和运行 context 设置。

内部 manager agent 当前使用的主要工具：

- `list_chunk_documents`
- `query_chunk_positions`
- `create_chunk_document`
- `insert_chunks`
- `update_chunks`
- `delete_chunks`
- `keyword_recall`
- `graph_distance_recall`
- `graph_mark_useful`
- `graph_mark_blocked`
- `graph_clear_blocked`
- `graph_create_nodes`
- `graph_update_node`
- `graph_delete_nodes`
- `read_nodes`

`manage_knowledge` 返回：

- `operation`：新增、修改、删除、读取、useful 标记的数量摘要。
- `message`：manager agent 的总结。
- `useful_items`：正文、节点 id 和边；如果是 chunk，还包含 `document_name` 和 `chunk_index`。

最小组装方式：

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from middleware import KnowledgeManagerCapabilityMiddleware, KnowledgeManagerMiddlewareConfig

knowledge_manager_config = KnowledgeManagerMiddlewareConfig.load_config_knowledge_manager_middleware({
    "neo4j": {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "1575338771",
        "database": None
    },
    "run_id": "demo-kb",
    "trace_limit": 16,
    "tool": {
        "temperature": 0.0,
        "debug": False,
        "stream_inner_agent": True,
        "inner_recursion_limit": 64,
        "agent_overrides": {
            "model": {
                "model": "gpt-5.4-nano",
                "model_provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "YOUR_API_KEY",
                "temperature": 0.0
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "base_url": "https://api.openai.com/v1",
                "api_key": "YOUR_API_KEY",
                "dimensions": 1536
            },
            "graph_query": {
                "capability_preset": {
                    "keyword_top_k": 6,
                    "keyword_top_k_limit": 10,
                    "distance_top_k": 6,
                    "distance_top_k_limit": 10,
                    "distance_max_distance": 1.5,
                    "useful_max_items": 12,
                    "useful_max_total_chars": 3000,
                    "blocked_max_items": 12,
                    "blocked_max_total_chars": 3000
                }
            }
        }
    }
})

knowledge_manager_wrapper = KnowledgeManagerCapabilityMiddleware(config=knowledge_manager_config)
model = init_chat_model(model="gpt-5.4-nano", model_provider="openai", temperature=0.0)
agent = create_agent(
    model=model,
    middleware=[knowledge_manager_wrapper.middleware],
    checkpointer=InMemorySaver(),
    context_schema=type(knowledge_manager_config),
    system_prompt="你是主 agent。知识库操作必须交给 manage_knowledge。",
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": "整理当前知识库里关于公开入口的内容，并把关键节点标记 useful。"}]},
    config={"configurable": {"thread_id": "demo-thread"}, "recursion_limit": 80},
    context=knowledge_manager_config,
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(event)
```

## 7. Examples

- [examples/chunk_apply_example.py](/Users/apexwave/Desktop/memory/examples/chunk_apply_example.py:1)：文件入库顶层 tool 示例，按 demo 风格通过 agent 调用 `chunk_apply_wrapper.tool`。
- [examples/knowledge_manager_middleware_example.py](/Users/apexwave/Desktop/memory/examples/knowledge_manager_middleware_example.py:1)：知识管理中间键示例，通过 `knowledge_manager_wrapper.middleware` 挂载 `manage_knowledge`。

运行：

```bash
uv run python examples/chunk_apply_example.py
uv run python examples/knowledge_manager_middleware_example.py
```

## 8. 推荐验证

普通回归：

```bash
uv run python -m unittest tests.test_chunk_apply tests.test_document_query tests.test_document_write tests.test_graph_tools
uv run python -m unittest tests.test_manage_knowledge_tool tests.test_management_discovery tests.test_knowledge_manager_runtime
uv run python -m unittest tests.test_knowledge_manager_middleware_public_api tests.test_unified_agent_public_api
```

真实模型和真实 Neo4j：

```bash
uv run python tests/test_live_chunk_stream.py
uv run python tests/test_live_graph_link_focus.py
uv run python tests/test_live_public_entrypoints.py
```

live 测试必须观察流式 updates / custom events，不能只看最终结果。
