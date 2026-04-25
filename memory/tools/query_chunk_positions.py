"""Query Chunk records by document position."""

import json
from pathlib import Path
from typing import Any
from typing import Literal

from langchain.agents.middleware import AgentState
from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external
from server.neo4j import DocumentStore

from ._output import limit_items, strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("query_chunk_positions.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_RETURN_CHUNKS = 20
MAX_RETURN_POSITIONS = 50


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class QueryChunkPositionsToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "QueryChunkPositionsToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
        )

    @classmethod
    def load_config_query_chunk_positions_tool(cls, source=None) -> "QueryChunkPositionsToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class QueryChunkPositionsItem(BaseModel):
    document_name: str = Field(description="要查询的文档名。")
    positions: int | list[int | list[int]] = Field(
        description="要查询的 chunk 索引，或由 [start, end] 组成的索引区间列表。"
    )
    mode: Literal["summary", "detail"] = Field(
        default="summary",
        description="summary 只保留摘要和边，detail 返回完整 chunk。",
    )

    @model_validator(mode="after")
    def validate_item(self) -> "QueryChunkPositionsItem":
        self.document_name = self.document_name.strip()
        self.mode = self.mode.strip()
        return self


class QueryChunkPositionsInput(BaseModel):
    document_name: str | None = Field(default=None, description="单文档查询时要读的文档名。")
    positions: int | list[int | list[int]] | None = Field(
        default=None,
        description="单文档查询时的 chunk 索引，或由 [start, end] 组成的索引区间列表。"
    )
    items: list[QueryChunkPositionsItem] | None = Field(
        default=None,
        description="批处理查询项。若提供 items，就忽略 document_name 和 positions。"
    )
    mode: Literal["summary", "detail"] = Field(
        default="detail",
        description="单文档模式下的返回模式。summary 或 detail。",
    )

    @model_validator(mode="after")
    def validate_input(self) -> "QueryChunkPositionsInput":
        if self.items is None:
            if self.document_name is None or not self.document_name.strip():
                raise ValueError("document_name is required when items is not provided")
            if self.positions is None:
                raise ValueError("positions is required when items is not provided")
            self.document_name = self.document_name.strip()
            self.mode = self.mode.strip()
        elif not self.items:
            raise ValueError("items must not be empty")
        return self


class QueryChunkPositionsToolStateTydict(AgentState, total=False):
    pass


class QueryChunkPositionsToolFeedback(BaseModel):
    successText: str = Field(default="已读取 Chunk 位置。")
    failureText: str = Field(default="读取 Chunk 位置失败：{error}")


class QueryChunkPositionsToolSchema:
    name = "query_chunk_positions"
    args_schema = QueryChunkPositionsInput
    description = (
        "当你已经知道文档名和 chunk 位置，想读取具体 chunk、确认顺序或检查相邻边时使用。"
        "单文档模式填写 document_name + positions；批处理模式填写 items。"
        "mode=summary 只返回摘要和边，mode=detail 返回正文、关键词和边。"
    )
    toolfeedback = QueryChunkPositionsToolFeedback


def build_query_chunk_positions_tool(
    *,
    config: QueryChunkPositionsToolConfig | None = None,
    store: DocumentStore | None = None,
):
    active_config = config or QueryChunkPositionsToolConfig.load()
    active_store = store or DocumentStore(config_path=active_config.neo4j_config_path, run_id=active_config.run_id)

    @tool(
        "query_chunk_positions",
        args_schema=QueryChunkPositionsInput,
        description=QueryChunkPositionsToolSchema.description,
    )
    def query_chunk_positions(
        document_name: str | None = None,
        positions: int | list[int | list[int]] | None = None,
        items: list[QueryChunkPositionsItem] | None = None,
        mode: Literal["summary", "detail"] = "detail",
    ) -> dict[str, Any]:
        """Query Chunk records by document position."""
        tool_input = QueryChunkPositionsInput(
            document_name=document_name,
            positions=positions,
            items=items,
            mode=mode,
        )
        query_items = _normalize_query_items(tool_input)
        results = [_run_query_item(active_store, item) for item in query_items]
        limited_results, total_results, results_truncated = limit_items(results, MAX_RETURN_CHUNKS)
        status = "success" if all(result["status"] == "success" for result in results) else "error"
        if len(results) == 1 and tool_input.items is None:
            result = results[0]
            payload = {
                "operation": "query_chunk_positions",
                "status": result["status"],
                "document_name": result["document_name"],
                "mode": result["mode"],
                "requested_positions": result["requested_positions"],
                "found_positions": result["found_positions"],
                "missing_positions": result["missing_positions"],
                "total_chunks": result["total_chunks"],
                "chunk_count": len(result["chunks"]),
                "chunks": result["chunks"],
            }
        else:
            payload = {
                "operation": "query_chunk_positions",
                "status": status,
                "item_count": total_results,
                "returned_item_count": len(limited_results),
                "items_truncated": results_truncated,
                "chunk_count": sum(len(result["chunks"]) for result in limited_results),
                "total_chunk_count": sum(len(result["chunks"]) for result in results),
                "results": limited_results,
            }
        return strip_internal_run_context(payload)

    return query_chunk_positions


class QueryChunkPositionsTool:
    name = QueryChunkPositionsToolSchema.name
    config = QueryChunkPositionsToolConfig
    substate = QueryChunkPositionsToolStateTydict
    toolschema = QueryChunkPositionsToolSchema

    def __init__(self, config: QueryChunkPositionsToolConfig | None = None, store: DocumentStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_query_chunk_positions_tool(config=self.config, store=self.store)


tool_runingconfig = QueryChunkPositionsToolConfig.load()
tools = {}
toolStateTydicts = {
    "query_chunk_positions": QueryChunkPositionsToolStateTydict,
}
ToolConfig = {
    "inputSm": QueryChunkPositionsInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}


def _run_query_item(store: DocumentStore, item: QueryChunkPositionsItem) -> dict[str, Any]:
    positions = _flatten_positions(item.positions)
    active_run_id = getattr(store, "run_id", None)
    chunks = store.query_positions(document_name=item.document_name, positions=positions, run_id=active_run_id)
    all_chunks = store.query_positions(document_name=item.document_name, positions=None, run_id=active_run_id)
    found_positions = [int(chunk["chunk_index"]) for chunk in chunks]
    missing_positions = [position for position in positions if position not in set(found_positions)]
    rendered_chunks = [_render_chunk(chunk, mode=item.mode) for chunk in chunks]
    limited_requested_positions, total_requested_positions, requested_truncated = limit_items(positions, MAX_RETURN_POSITIONS)
    limited_found_positions, total_found_positions, found_truncated = limit_items(found_positions, MAX_RETURN_POSITIONS)
    limited_missing_positions, total_missing_positions, missing_truncated = limit_items(missing_positions, MAX_RETURN_POSITIONS)
    limited_chunks, total_returned_chunks, chunks_truncated = limit_items(rendered_chunks, MAX_RETURN_CHUNKS)
    status = "success" if all_chunks else "error"
    if status == "success":
        message = f"Found {len(rendered_chunks)} chunk(s)."
        if chunks_truncated:
            message = f"Found {total_returned_chunks} chunk(s), returned {len(limited_chunks)}."
    else:
        message = f"Document {item.document_name} not found."
    return {
        "operation": "query_chunk_positions",
        "status": status,
        "message": message,
        "document_name": item.document_name,
        "mode": item.mode,
        "requested_positions": limited_requested_positions,
        "requested_position_count": total_requested_positions,
        "requested_positions_truncated": requested_truncated,
        "found_positions": limited_found_positions,
        "found_position_count": total_found_positions,
        "found_positions_truncated": found_truncated,
        "missing_positions": limited_missing_positions,
        "missing_position_count": total_missing_positions,
        "missing_positions_truncated": missing_truncated,
        "total_chunks": len(all_chunks),
        "chunk_count": total_returned_chunks,
        "returned_chunk_count": len(limited_chunks),
        "chunks_truncated": chunks_truncated,
        "chunks": limited_chunks,
    }

def _normalize_query_items(tool_input: QueryChunkPositionsInput) -> list[QueryChunkPositionsItem]:
    if tool_input.items is not None:
        if not tool_input.items:
            raise ValueError("items must not be empty")
        return list(tool_input.items)
    if tool_input.document_name is None:
        raise ValueError("document_name is required when items is not provided")
    if tool_input.positions is None:
        raise ValueError("positions is required when items is not provided")
    return [
        QueryChunkPositionsItem(
            document_name=tool_input.document_name,
            positions=tool_input.positions,
            mode=tool_input.mode,
        )
    ]


def _flatten_positions(positions: int | list[int | list[int]]) -> list[int]:
    if isinstance(positions, int):
        flattened = [positions]
    else:
        flattened = []
        for position in positions:
            if isinstance(position, int):
                flattened.append(position)
                continue
            if len(position) != 2:
                raise ValueError("range chunk indexes must contain exactly two integers")
            start, end = int(position[0]), int(position[1])
            if start > end:
                raise ValueError("range end must be greater than or equal to start")
            flattened.extend(range(start, end + 1))
    if not flattened:
        raise ValueError("positions must not be empty")
    if any(position < 0 for position in flattened):
        raise ValueError("positions must be non-negative chunk indexes")
    return sorted(set(flattened))


def _render_chunk(chunk: dict[str, object], *, mode: str) -> dict[str, object]:
    base_payload = {
        "id": chunk["id"],
        "document_name": chunk["document_name"],
        "chunk_index": chunk["chunk_index"],
        "summary": chunk["summary"],
        "edges": strip_internal_run_context(chunk.get("edges", [])),
    }
    if mode == "detail":
        base_payload["keywords"] = chunk.get("keywords", [])
        base_payload["body"] = chunk.get("body", "")
        return base_payload
    if mode != "summary":
        raise ValueError("mode must be summary or detail")
    return base_payload
