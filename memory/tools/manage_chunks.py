"""Manage chunks inside an existing Chunk document."""

import json
from pathlib import Path
from typing import Any, Literal

from langchain.agents.middleware import AgentState
from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external
from server.neo4j import DocumentStore

from ._output import limit_items, strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("manage_chunks.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_RETURN_RESULTS = 20
MAX_RETURN_CHUNKS = 20


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ManageChunksToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    document_edge_distance: float = Field(default=0.3, ge=0.0, description="文档内部顺序边写入的默认距离值。")
    persist_keyword_embeddings: bool = Field(default=True, description="是否持久化关键词向量。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "ManageChunksToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            document_edge_distance=float(payload.get("document_edge_distance", 0.3)),
            persist_keyword_embeddings=bool(payload.get("persist_keyword_embeddings", True)),
        )

    @classmethod
    def load_config_manage_chunks_tool(cls, source=None) -> "ManageChunksToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class KeywordOp(BaseModel):
    op: Literal["+", "-", "replace"] = Field(description="关键词操作，只支持 +、- 或 replace。")
    keywords: list[str] = Field(default_factory=list, description="要新增、删除或替换的关键词列表。")


class EdgeOp(BaseModel):
    op: Literal["+", "-"] = Field(description="边操作，只支持 + 或 -。")
    targets: str | list[int | str | list[int] | list[str]] = Field(
        description="普通图边目标，可以是真实 node id、chunk_index、chunk_index 区间，或者它们组成的列表。"
    )
    dist: float | None = Field(default=None, description="边的距离值。新增边时默认使用文档距离配置。")


class ChunkManageAction(BaseModel):
    op: Literal["insert", "update", "delete"] = Field(
        description="动作类型。insert 用于新增 chunk，update 用于修改已有 chunk，delete 用于删除已有 chunk。"
    )
    id: str | None = Field(
        default=None,
        description="目标 chunk 的 id。update 和 delete 时通常需要；insert 时可不传，系统会自动生成。",
    )
    chunk_index: int | None = Field(
        default=None,
        description="目标 chunk 的索引。可以和 id 二选一定位已有 chunk；insert 时可用来描述插入位置，update/delete 时也可用。",
    )
    insert_after: int | None = Field(
        default=None,
        description="仅 insert 使用。表示把新 chunk 插在某个 chunk 索引之后；不传时会追加到末尾。",
    )
    summary: str | None = Field(
        default=None,
        description="chunk 摘要。insert 时必填；update 时可选；delete 时不要传。",
    )
    body: str | None = Field(
        default=None,
        description="chunk 正文。insert 时必填；update 时可选；delete 时不要传。",
    )
    keywords: list[str] | None = Field(
        default=None,
        description="chunk 关键词。insert 和 update 时可选；delete 时不要传。",
    )
    keyword_ops: list[KeywordOp] = Field(
        default_factory=list,
        description="关键词增删改动作列表。必须是数组；insert 和 update 时可用；delete 时不要传。",
    )
    edge_ops: list[EdgeOp] = Field(
        default_factory=list,
        description="普通图边增删改动作列表。必须是数组；insert 和 update 时可用；delete 时不要传。",
    )

    @model_validator(mode="after")
    def validate_action(self) -> "ChunkManageAction":
        return self


class ManageChunksInput(BaseModel):
    document_name: str = Field(description="要管理的目标文档名。")
    actions: list[ChunkManageAction] = Field(
        description=(
            "按顺序执行的 chunk 管理动作列表。可以混合 insert、update 和 delete；"
            "insert 用来新增 chunk，update 用来改内容、关键词或普通边，delete 用来删除已有 chunk。"
            "insert 通常提供 insert_after、summary、body、keywords， 可选 edge_ops；"
            "update 通常提供 id 或 chunk_index，再加要修改的字段；delete 只保留定位字段。"
        )
    )

    @model_validator(mode="after")
    def validate_input(self) -> "ManageChunksInput":
        self.document_name = self.document_name.strip()
        if not self.document_name:
            raise ValueError("document_name is required")
        if not self.actions:
            raise ValueError("actions must not be empty")
        return self


class InsertChunkAction(BaseModel):
    id: str | None = Field(default=None, description="可选的新 chunk id；不传时系统自动生成。")
    insert_after: int | None = Field(default=None, description="把新 chunk 插在该 chunk_index 之后；不传时追加到末尾。")
    summary: str = Field(description="新 chunk 摘要。")
    body: str = Field(description="新 chunk 正文。")
    keywords: list[str] = Field(default_factory=list, description="新 chunk 关键词列表。")
    keyword_ops: list[KeywordOp] = Field(default_factory=list, description="可选关键词操作；会在 keywords 基础上继续应用。")
    edge_ops: list[EdgeOp] = Field(default_factory=list, description="可选普通图边操作；每个对象描述一组目标。")


class InsertChunksInput(BaseModel):
    document_name: str = Field(description="要插入 chunk 的已有文档名。")
    items: list[InsertChunkAction] = Field(description="要按顺序插入的新 chunk 列表。")

    @model_validator(mode="after")
    def validate_input(self) -> "InsertChunksInput":
        self.document_name = self.document_name.strip()
        if not self.document_name:
            raise ValueError("document_name is required")
        if not self.items:
            raise ValueError("items must not be empty")
        return self


class UpdateChunkAction(BaseModel):
    id: str | None = Field(default=None, description="目标 chunk id；和 chunk_index 二选一。")
    chunk_index: int | None = Field(default=None, description="目标 chunk 索引；和 id 二选一。")
    summary: str | None = Field(default=None, description="可选的新摘要。")
    body: str | None = Field(default=None, description="可选的新正文。")
    keywords: list[str] | None = Field(default=None, description="可选的全量关键词替换列表。")
    keyword_ops: list[KeywordOp] = Field(default_factory=list, description="关键词细粒度操作，支持 +、-、replace。")
    edge_ops: list[EdgeOp] = Field(
        default_factory=list,
        description="普通图边操作，支持新增或删除非 DOCUMENT_NEXT 边；不能用来调整文档顺序。",
    )

    @model_validator(mode="after")
    def validate_action(self) -> "UpdateChunkAction":
        if self.id is None and self.chunk_index is None:
            raise ValueError("id or chunk_index is required")
        return self


class UpdateChunksInput(BaseModel):
    document_name: str = Field(description="要更新 chunk 的已有文档名。")
    items: list[UpdateChunkAction] = Field(description="要按顺序更新的 chunk 列表。")

    @model_validator(mode="after")
    def validate_input(self) -> "UpdateChunksInput":
        self.document_name = self.document_name.strip()
        if not self.document_name:
            raise ValueError("document_name is required")
        if not self.items:
            raise ValueError("items must not be empty")
        return self


class DeleteChunkAction(BaseModel):
    id: str | None = Field(default=None, description="目标 chunk id；和 chunk_index 二选一。")
    chunk_index: int | None = Field(default=None, description="目标 chunk 索引；和 id 二选一。")

    @model_validator(mode="after")
    def validate_action(self) -> "DeleteChunkAction":
        if self.id is None and self.chunk_index is None:
            raise ValueError("id or chunk_index is required")
        return self


class DeleteChunksInput(BaseModel):
    document_name: str = Field(description="要删除 chunk 的已有文档名。")
    items: list[DeleteChunkAction] = Field(description="要按顺序删除的 chunk 列表。")

    @model_validator(mode="after")
    def validate_input(self) -> "DeleteChunksInput":
        self.document_name = self.document_name.strip()
        if not self.document_name:
            raise ValueError("document_name is required")
        if not self.items:
            raise ValueError("items must not be empty")
        return self


class ManageChunksToolStateTydict(AgentState, total=False):
    pass


class ManageChunksToolFeedback(BaseModel):
    successText: str = Field(default="已管理 Chunk。")
    failureText: str = Field(default="管理 Chunk 失败：{error}")


class ManageChunksToolSchema:
    name = "manage_chunks"
    args_schema = ManageChunksInput
    description = (
        "在一篇已有 Chunk 文档内批量插入、更新或删除 chunk。"
        "insert 新增 chunk，update 修改已有 chunk 的摘要、正文、关键词或普通图边，delete 删除已有 chunk。"
        "edge_ops 只处理普通图边的新增或删除，不处理文档顺序边。"
    )
    toolfeedback = ManageChunksToolFeedback


class InsertChunksToolSchema:
    name = "insert_chunks"
    args_schema = InsertChunksInput
    description = (
        "向一篇已有 Chunk 文档插入新的 chunk。"
        "只执行插入动作；不会更新或删除已有 chunk。"
        "插入后系统会重排 chunk_index 并重建该文档内部 DOCUMENT_NEXT。"
    )
    toolfeedback = ManageChunksToolFeedback


class UpdateChunksToolSchema:
    name = "update_chunks"
    args_schema = UpdateChunksInput
    description = (
        "更新一篇已有 Chunk 文档中的 chunk 正文、摘要、关键词或普通图边。"
        "每个 item 必须用 id 或 chunk_index 定位目标；不要用它插入或删除。"
        "keyword_ops 支持 +、-、replace；edge_ops 只处理普通 graph 边，不处理 DOCUMENT_NEXT。"
    )
    toolfeedback = ManageChunksToolFeedback


class DeleteChunksToolSchema:
    name = "delete_chunks"
    args_schema = DeleteChunksInput
    description = (
        "从一篇已有 Chunk 文档中删除 chunk。每个 item 必须用 id 或 chunk_index 定位目标；"
        "删除后会自动重排 chunk_index 并重建 DOCUMENT_NEXT。"
    )
    toolfeedback = ManageChunksToolFeedback


def _active_document_store(
    *,
    config: ManageChunksToolConfig | None = None,
    store: DocumentStore | None = None,
) -> tuple[ManageChunksToolConfig, DocumentStore]:
    active_config = config or ManageChunksToolConfig.load()
    active_store = store or DocumentStore(
        config_path=active_config.neo4j_config_path,
        run_id=active_config.run_id,
        document_edge_distance=active_config.document_edge_distance,
        persist_keyword_embeddings=active_config.persist_keyword_embeddings,
    )
    return active_config, active_store


def _run_manage_chunks(
    *,
    operation: str,
    active_config: ManageChunksToolConfig,
    active_store: DocumentStore,
    document_name: str,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    result = active_store.manage_chunks(
        document_name=document_name,
        actions=actions,
        run_id=active_config.run_id,
    )
    limited_results, total_results, results_truncated = limit_items(result.get("results") or [], MAX_RETURN_RESULTS)
    limited_chunks, total_chunks, chunks_truncated = limit_items(result.get("chunks") or [], MAX_RETURN_CHUNKS)
    success_count = sum(1 for item in limited_results if isinstance(item, dict) and item.get("status") == "success")
    failure_count = sum(1 for item in limited_results if isinstance(item, dict) and item.get("status") != "success")
    return {
        "operation": operation,
        "status": "success" if result.get("ok") else "error",
        **strip_internal_run_context(
            {
                **result,
                "results": limited_results,
                "result_count": total_results,
                "returned_result_count": len(limited_results),
                "results_truncated": results_truncated,
                "success_count": success_count,
                "failure_count": failure_count,
                "chunks": limited_chunks,
                "chunk_count": total_chunks,
                "returned_chunk_count": len(limited_chunks),
                "chunks_truncated": chunks_truncated,
            }
        ),
    }


def build_manage_chunks_tool(
    *,
    config: ManageChunksToolConfig | None = None,
    store: DocumentStore | None = None,
):
    active_config, active_store = _active_document_store(config=config, store=store)

    @tool(
        ManageChunksToolSchema.name,
        args_schema=ManageChunksToolSchema.args_schema,
        description=ManageChunksToolSchema.description,
    )
    def manage_chunks(document_name: str, actions: list[ChunkManageAction]) -> dict[str, Any]:
        """Manage chunks inside an existing Chunk document."""
        return _run_manage_chunks(
            operation="manage_chunks",
            active_config=active_config,
            active_store=active_store,
            document_name=document_name,
            actions=[_action_to_dict(action) for action in actions],
        )

    return manage_chunks


def build_insert_chunks_tool(
    *,
    config: ManageChunksToolConfig | None = None,
    store: DocumentStore | None = None,
):
    active_config, active_store = _active_document_store(config=config, store=store)

    @tool(
        InsertChunksToolSchema.name,
        args_schema=InsertChunksToolSchema.args_schema,
        description=InsertChunksToolSchema.description,
    )
    def insert_chunks(document_name: str, items: list[InsertChunkAction]) -> dict[str, Any]:
        """Insert chunks into an existing Chunk document."""
        actions = []
        for item in items:
            payload = _action_to_dict(item.model_dump())
            payload["op"] = "insert"
            actions.append(payload)
        return _run_manage_chunks(
            operation="insert_chunks",
            active_config=active_config,
            active_store=active_store,
            document_name=document_name,
            actions=actions,
        )

    return insert_chunks


def build_update_chunks_tool(
    *,
    config: ManageChunksToolConfig | None = None,
    store: DocumentStore | None = None,
):
    active_config, active_store = _active_document_store(config=config, store=store)

    @tool(
        UpdateChunksToolSchema.name,
        args_schema=UpdateChunksToolSchema.args_schema,
        description=UpdateChunksToolSchema.description,
    )
    def update_chunks(document_name: str, items: list[UpdateChunkAction]) -> dict[str, Any]:
        """Update chunks inside an existing Chunk document."""
        actions = []
        for item in items:
            payload = _action_to_dict(item.model_dump(exclude_none=True))
            payload["op"] = "update"
            actions.append(payload)
        return _run_manage_chunks(
            operation="update_chunks",
            active_config=active_config,
            active_store=active_store,
            document_name=document_name,
            actions=actions,
        )

    return update_chunks


def build_delete_chunks_tool(
    *,
    config: ManageChunksToolConfig | None = None,
    store: DocumentStore | None = None,
):
    active_config, active_store = _active_document_store(config=config, store=store)

    @tool(
        DeleteChunksToolSchema.name,
        args_schema=DeleteChunksToolSchema.args_schema,
        description=DeleteChunksToolSchema.description,
    )
    def delete_chunks(document_name: str, items: list[DeleteChunkAction]) -> dict[str, Any]:
        """Delete chunks from an existing Chunk document."""
        actions = []
        for item in items:
            payload = item.model_dump(exclude_none=True)
            payload["op"] = "delete"
            actions.append(payload)
        return _run_manage_chunks(
            operation="delete_chunks",
            active_config=active_config,
            active_store=active_store,
            document_name=document_name,
            actions=actions,
        )

    return delete_chunks


class ManageChunksTool:
    name = ManageChunksToolSchema.name
    config = ManageChunksToolConfig
    substate = ManageChunksToolStateTydict
    toolschema = ManageChunksToolSchema

    def __init__(self, config: ManageChunksToolConfig | None = None, store: DocumentStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_manage_chunks_tool(config=self.config, store=self.store)


class InsertChunksTool(ManageChunksTool):
    name = InsertChunksToolSchema.name
    toolschema = InsertChunksToolSchema

    def create_tool(self):
        return build_insert_chunks_tool(config=self.config, store=self.store)


class UpdateChunksTool(ManageChunksTool):
    name = UpdateChunksToolSchema.name
    toolschema = UpdateChunksToolSchema

    def create_tool(self):
        return build_update_chunks_tool(config=self.config, store=self.store)


class DeleteChunksTool(ManageChunksTool):
    name = DeleteChunksToolSchema.name
    toolschema = DeleteChunksToolSchema

    def create_tool(self):
        return build_delete_chunks_tool(config=self.config, store=self.store)


tool_runingconfig = ManageChunksToolConfig.load()
tools = {}
toolStateTydicts = {
    "manage_chunks": ManageChunksToolStateTydict,
}
ToolConfig = {
    "inputSm": ManageChunksInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}


def _action_to_dict(action: ChunkManageAction) -> dict[str, Any]:
    raw_action = action if isinstance(action, dict) else action.model_dump()
    payload = dict(raw_action)
    keyword_ops = payload.get("keyword_ops")
    if keyword_ops is not None:
        payload["keyword_ops"] = [
            dict(keyword_op if isinstance(keyword_op, dict) else keyword_op.model_dump())
            for keyword_op in keyword_ops
        ]
    edge_ops = payload.get("edge_ops")
    if edge_ops is not None:
        payload["edge_ops"] = [
            dict(edge_op if isinstance(edge_op, dict) else edge_op.model_dump())
            for edge_op in edge_ops
        ]
    return payload
