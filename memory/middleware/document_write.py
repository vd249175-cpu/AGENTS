"""Document write middleware declaration."""

import json
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from pydantic import BaseModel, Field
from server.component_config import config_from_external
from server.neo4j import DocumentStore, Neo4jConnectionConfig, resolve_neo4j_connection
from tools.create_chunk_document import (
    CreateChunkDocumentToolConfig,
    CreateChunkDocumentToolStateTydict,
    ToolConfig as CreateChunkDocumentToolBundle,
    build_create_chunk_document_tool,
)
from tools.manage_chunks import (
    ManageChunksToolConfig,
    ManageChunksToolStateTydict,
    build_delete_chunks_tool,
    build_insert_chunks_tool,
    build_update_chunks_tool,
)


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("document_write.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class DocumentWriteMiddlewareConfig(BaseModel):
    neo4j: Neo4jConnectionConfig | None = Field(default=None, description="显式的 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    trace_limit: int = Field(default=16, ge=1, description="运行时 trace 的最大保留条数。")
    embedding_provider: str | None = Field(default=None, description="可选的 embedding provider override。")
    embedding_model: str | None = Field(default=None, description="可选的 embedding model override。")
    embedding_base_url: str | None = Field(default=None, description="可选的 embedding base_url override。")
    embedding_api_key: str | None = Field(default=None, description="可选的 embedding api_key override。")
    embedding_dimensions: int | None = Field(default=None, description="可选的 embedding dimensions override。")

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "DocumentWriteMiddlewareConfig":
        payload = _load_json(path)
        return cls(
            neo4j=Neo4jConnectionConfig.model_validate(payload["neo4j"]) if isinstance(payload.get("neo4j"), dict) else None,
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            trace_limit=max(1, int(payload.get("trace_limit", 16))),
            embedding_provider=payload.get("embedding_provider"),
            embedding_model=payload.get("embedding_model"),
            embedding_base_url=payload.get("embedding_base_url"),
            embedding_api_key=payload.get("embedding_api_key"),
            embedding_dimensions=payload.get("embedding_dimensions"),
        )

    @classmethod
    def load_config_document_write_middleware(cls, source=None) -> "DocumentWriteMiddlewareConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


class DocumentWriteStateTydict(AgentState, total=False):
    pass


middleware_runingconfig = DocumentWriteMiddlewareConfig.load()
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="document_write.guidance",
        prompt=(
            "当前任务需要写入 Chunk 文档时，只操作明确的文档名和 chunk 目标。"
            "新建文档、插入、更新、删除是不同动作；每次选择与当前意图匹配的写入口。"
            "create_chunk_document 只用于创建新文档和首个 chunk；已有文档追加内容用 insert_chunks。"
            "update_chunks 用 id 或 chunk_index 定位已有 chunk；关键词细粒度修改使用 keyword_ops 的 +、-、replace。"
            "delete_chunks 只删除已定位的 chunk。"
            "写入后系统会维护 chunk_index 和 DOCUMENT_NEXT；工具入参只描述要写的内容、关键词和普通图边。"
        ),
    )
]
MiddlewareToolConfig = {
    "tools": {
        **CreateChunkDocumentToolBundle["tools"],
    },
    "toolStateTydicts": {
        **CreateChunkDocumentToolBundle["toolStateTydicts"],
        "insert_chunks": ManageChunksToolStateTydict,
        "update_chunks": ManageChunksToolStateTydict,
        "delete_chunks": ManageChunksToolStateTydict,
    },
}


def _build_tool_config(config: DocumentWriteMiddlewareConfig) -> dict[str, object]:
    neo4j = resolve_neo4j_connection(connection=config.neo4j, path=config.neo4j_config_path)
    resolved_config_path = config.neo4j_config_path or (PROJECT_ROOT / "workspace/config/database_config.json").resolve()
    store = DocumentStore(
        uri=neo4j.uri,
        username=neo4j.username,
        password=neo4j.password,
        database=neo4j.database,
        run_id=config.run_id,
        embedding_config_override=_embedding_override_from_config(config),
    )
    create_config = CreateChunkDocumentToolConfig.load().model_copy(
        update={"neo4j_config_path": resolved_config_path, "run_id": config.run_id}
    )
    manage_config = ManageChunksToolConfig.load().model_copy(
        update={"neo4j_config_path": resolved_config_path, "run_id": config.run_id}
    )
    create_tool = build_create_chunk_document_tool(config=create_config, store=store)
    insert_tool = build_insert_chunks_tool(config=manage_config, store=store)
    update_tool = build_update_chunks_tool(config=manage_config, store=store)
    delete_tool = build_delete_chunks_tool(config=manage_config, store=store)
    return {
        "store": store,
        "tools": {
            create_tool.name: create_tool,
            insert_tool.name: insert_tool,
            update_tool.name: update_tool,
            delete_tool.name: delete_tool,
        },
        "toolStateTydicts": {
            "create_chunk_document": CreateChunkDocumentToolStateTydict,
            "insert_chunks": ManageChunksToolStateTydict,
            "update_chunks": ManageChunksToolStateTydict,
            "delete_chunks": ManageChunksToolStateTydict,
        },
    }


class DocumentWriteCapabilityMiddleware(AgentMiddleware):
    name = "document_write"
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = middleware_runingconfig
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = DocumentWriteStateTydict  # type: ignore[assignment]

    def __init__(self, *, config: DocumentWriteMiddlewareConfig | None = None) -> None:
        super().__init__()
        self.config = config or self.runingConfig
        self.capability_prompts = self.capabilityPromptConfigs
        runtime_tool_config = _build_tool_config(self.config)
        self._store = runtime_tool_config["store"]
        self.toolConfig = runtime_tool_config
        self.toolStateTydicts = runtime_tool_config["toolStateTydicts"]
        self.tools = list(runtime_tool_config["tools"].values())
        self.middleware = self

    def close(self) -> None:
        self._store.close()


def _embedding_override_from_config(config: DocumentWriteMiddlewareConfig) -> dict[str, object] | None:
    override = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "base_url": config.embedding_base_url,
        "api_key": config.embedding_api_key,
        "dimensions": config.embedding_dimensions,
    }
    cleaned = {
        key: value
        for key, value in override.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }
    return cleaned or None
