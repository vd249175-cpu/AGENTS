"""Create a new Chunk document."""

import json
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState
from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator
from server.component_config import config_from_external
from server.neo4j import DocumentStore

from ._output import strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("create_chunk_document.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class CreateChunkDocumentToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    document_edge_distance: float = Field(default=0.3, ge=0.0, description="文档内部顺序边写入的默认距离值。")
    persist_keyword_embeddings: bool = Field(default=True, description="是否持久化关键词向量。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "CreateChunkDocumentToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            document_edge_distance=float(payload.get("document_edge_distance", 0.3)),
            persist_keyword_embeddings=bool(payload.get("persist_keyword_embeddings", True)),
        )

    @classmethod
    def load_config_create_chunk_document_tool(cls, source=None) -> "CreateChunkDocumentToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class CreateChunkDocumentInput(BaseModel):
    document_name: str = Field(description="要创建的文档名，使用不带文件后缀的名称；同名文档已存在时会失败。")
    summary: str = Field(description="首个 chunk 的摘要。")
    body: str = Field(description="首个 chunk 的正文。")
    keywords: list[str] = Field(default_factory=list, description="首个 chunk 的关键词。")

    @model_validator(mode="after")
    def validate_payload(self) -> "CreateChunkDocumentInput":
        if not self.document_name.strip():
            raise ValueError("document_name is required")
        self.document_name = self.document_name.strip()
        self.summary = self.summary.strip()
        self.body = self.body.strip()
        self.keywords = [keyword.strip() for keyword in self.keywords if keyword.strip()]
        return self


class CreateChunkDocumentToolStateTydict(AgentState, total=False):
    pass


CREATE_CHUNK_DOCUMENT_DESCRIPTION = (
    "创建一篇新的 Chunk 文档，并写入它的首个 chunk。"
    "只处理新文档的初始化；已存在同名文档时会返回错误。"
    "document_name 使用不带文件后缀的名称。需要提供 summary、body 和 keywords；keywords 用于后续召回，可以为空。"
)


class CreateChunkDocumentToolFeedback(BaseModel):
    successText: str = Field(default="已创建 Chunk 文档。")
    failureText: str = Field(default="创建 Chunk 文档失败：{error}")


class CreateChunkDocumentToolSchema:
    name = "create_chunk_document"
    args_schema = CreateChunkDocumentInput
    description = CREATE_CHUNK_DOCUMENT_DESCRIPTION
    toolfeedback = CreateChunkDocumentToolFeedback


def build_create_chunk_document_tool(
    *,
    config: CreateChunkDocumentToolConfig | None = None,
    store: DocumentStore | None = None,
):
    active_config = config or CreateChunkDocumentToolConfig.load()
    active_store = store or DocumentStore(
        config_path=active_config.neo4j_config_path,
        run_id=active_config.run_id,
        document_edge_distance=active_config.document_edge_distance,
        persist_keyword_embeddings=active_config.persist_keyword_embeddings,
    )

    @tool(CreateChunkDocumentToolSchema.name, args_schema=CreateChunkDocumentToolSchema.args_schema, description=CreateChunkDocumentToolSchema.description)
    def create_chunk_document(
        document_name: str,
        summary: str,
        body: str,
        keywords: list[str],
    ) -> dict[str, Any]:
        """Create a new Chunk document."""
        result = active_store.create_document(
            document_name=document_name,
            summary=summary,
            body=body,
            keywords=keywords,
            run_id=active_config.run_id,
        )
        return {
            "operation": "create_chunk_document",
            "status": "success" if result.get("ok") else "error",
            **strip_internal_run_context(result),
        }

    return create_chunk_document


class CreateChunkDocumentTool:
    name = CreateChunkDocumentToolSchema.name
    config = CreateChunkDocumentToolConfig
    substate = CreateChunkDocumentToolStateTydict
    toolschema = CreateChunkDocumentToolSchema

    def __init__(self, config: CreateChunkDocumentToolConfig | None = None, store: DocumentStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_create_chunk_document_tool(config=self.config, store=self.store)


tool_runingconfig = CreateChunkDocumentToolConfig.load()
tools = {}
toolStateTydicts = {
    "create_chunk_document": CreateChunkDocumentToolStateTydict,
}
ToolConfig = {
    "inputSm": CreateChunkDocumentInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
