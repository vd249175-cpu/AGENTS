"""Document query middleware declaration."""

import json
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from pydantic import BaseModel, Field
from server.component_config import config_from_external
from server.neo4j import DocumentStore, Neo4jConnectionConfig, resolve_neo4j_connection
from tools.list_chunk_documents import (
    ListChunkDocumentsToolConfig,
    ListChunkDocumentsToolStateTydict,
    ToolConfig as ListChunkDocumentsToolBundle,
    build_list_chunk_documents_tool,
)
from tools.query_chunk_positions import (
    QueryChunkPositionsToolConfig,
    QueryChunkPositionsToolStateTydict,
    ToolConfig as QueryChunkPositionsToolBundle,
    build_query_chunk_positions_tool,
)


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("document_query.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class DocumentQueryMiddlewareConfig(BaseModel):
    neo4j: Neo4jConnectionConfig | None = Field(default=None, description="显式的 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    trace_limit: int = Field(default=16, ge=1, description="运行时 trace 的最大保留条数。")

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "DocumentQueryMiddlewareConfig":
        payload = _load_json(path)
        return cls(
            neo4j=Neo4jConnectionConfig.model_validate(payload["neo4j"]) if isinstance(payload.get("neo4j"), dict) else None,
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            trace_limit=max(1, int(payload.get("trace_limit", 16))),
        )

    @classmethod
    def load_config_document_query_middleware(cls, source=None) -> "DocumentQueryMiddlewareConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


class DocumentQueryStateTydict(AgentState, total=False):
    pass


middleware_runingconfig = DocumentQueryMiddlewareConfig.load()
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="document_query.guidance",
        prompt=(
            "当前任务需要查看已有 Chunk 文档时，先确认文档名称，再按位置读取需要的 chunk。"
            "读取时只使用文档名、chunk 索引或索引范围；不要把任务目标写进底层查询参数。"
        ),
    )
]
MiddlewareToolConfig = {
    "tools": {
        **ListChunkDocumentsToolBundle["tools"],
        **QueryChunkPositionsToolBundle["tools"],
    },
    "toolStateTydicts": {
        **ListChunkDocumentsToolBundle["toolStateTydicts"],
        **QueryChunkPositionsToolBundle["toolStateTydicts"],
    },
}


def _build_tool_config(config: DocumentQueryMiddlewareConfig) -> dict[str, object]:
    neo4j = resolve_neo4j_connection(connection=config.neo4j, path=config.neo4j_config_path)
    resolved_config_path = config.neo4j_config_path or (PROJECT_ROOT / "workspace/config/database_config.json").resolve()
    store = DocumentStore(
        uri=neo4j.uri,
        username=neo4j.username,
        password=neo4j.password,
        database=neo4j.database,
        run_id=config.run_id,
    )
    list_config = ListChunkDocumentsToolConfig.load().model_copy(
        update={"neo4j_config_path": resolved_config_path, "run_id": config.run_id}
    )
    query_config = QueryChunkPositionsToolConfig.load().model_copy(
        update={"neo4j_config_path": resolved_config_path, "run_id": config.run_id}
    )
    list_tool = build_list_chunk_documents_tool(config=list_config, store=store)
    query_tool = build_query_chunk_positions_tool(config=query_config, store=store)
    return {
        "store": store,
        "tools": {
            list_tool.name: list_tool,
            query_tool.name: query_tool,
        },
        "toolStateTydicts": {
            "list_chunk_documents": ListChunkDocumentsToolStateTydict,
            "query_chunk_positions": QueryChunkPositionsToolStateTydict,
        },
    }


class DocumentQueryCapabilityMiddleware(AgentMiddleware):
    name = "document_query"
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = middleware_runingconfig
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = DocumentQueryStateTydict  # type: ignore[assignment]

    def __init__(self, *, config: DocumentQueryMiddlewareConfig | None = None) -> None:
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
