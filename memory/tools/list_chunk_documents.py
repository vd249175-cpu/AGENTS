"""List existing Chunk documents."""

import json
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState
from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external
from server.neo4j import DocumentStore

from ._output import limit_items, strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("list_chunk_documents.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_RETURN_DOCUMENTS = 20


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ListChunkDocumentsToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "ListChunkDocumentsToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
        )

    @classmethod
    def load_config_list_chunk_documents_tool(cls, source=None) -> "ListChunkDocumentsToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class ListChunkDocumentsInput(BaseModel):
    limit: int | None = Field(default=None, ge=0, description="可选返回数量上限。")

    @model_validator(mode="after")
    def validate_limit(self) -> "ListChunkDocumentsInput":
        if self.limit is not None:
            self.limit = max(0, int(self.limit))
        return self


class ListChunkDocumentsToolStateTydict(AgentState, total=False):
    pass


class ListChunkDocumentsToolFeedback(BaseModel):
    successText: str = Field(default="已列出 Chunk 文档。")
    failureText: str = Field(default="列出 Chunk 文档失败：{error}")


class ListChunkDocumentsToolSchema:
    name = "list_chunk_documents"
    args_schema = ListChunkDocumentsInput
    description = "列出当前知识库中已有的 Chunk 文档名称和基础信息。limit 可限制返回数量。"
    toolfeedback = ListChunkDocumentsToolFeedback


def build_list_chunk_documents_tool(
    *,
    config: ListChunkDocumentsToolConfig | None = None,
    store: DocumentStore | None = None,
):
    active_config = config or ListChunkDocumentsToolConfig.load()
    active_store = store or DocumentStore(config_path=active_config.neo4j_config_path, run_id=active_config.run_id)

    @tool(
        "list_chunk_documents",
        args_schema=ListChunkDocumentsInput,
        description=ListChunkDocumentsToolSchema.description,
    )
    def list_chunk_documents(limit: int | None = None) -> dict[str, Any]:
        """List existing Chunk documents."""
        documents = active_store.list_documents(run_id=active_config.run_id)
        requested_limit = MAX_RETURN_DOCUMENTS if limit is None else max(0, int(limit))
        limited_documents, total_documents, truncated = limit_items(documents, requested_limit)
        return {
            "operation": "list_chunk_documents",
            "status": "success",
            "document_count": total_documents,
            "returned_document_count": len(limited_documents),
            "documents_truncated": truncated,
            "documents": strip_internal_run_context(limited_documents),
        }

    return list_chunk_documents


class ListChunkDocumentsTool:
    name = ListChunkDocumentsToolSchema.name
    config = ListChunkDocumentsToolConfig
    substate = ListChunkDocumentsToolStateTydict
    toolschema = ListChunkDocumentsToolSchema

    def __init__(self, config: ListChunkDocumentsToolConfig | None = None, store: DocumentStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_list_chunk_documents_tool(config=self.config, store=self.store)


tool_runingconfig = ListChunkDocumentsToolConfig.load()
tools = {}
toolStateTydicts = {
    "list_chunk_documents": ListChunkDocumentsToolStateTydict,
}
ToolConfig = {
    "inputSm": ListChunkDocumentsInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
