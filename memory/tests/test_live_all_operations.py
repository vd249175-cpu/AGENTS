"""Live end-to-end check for real model access and all migrated operations.

Run with:
    uv run python tests/test_live_all_operations.py
"""

from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
import sys
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_chat_model_config, get_embedding_model_config
from tests.real_model_smoke import _post_json, _stream_chat_completion
from tools.chunk_apply import ChunkApplyInput, ChunkApplyTool
from tools.create_chunk_document import CreateChunkDocumentInput, CreateChunkDocumentToolConfig, build_create_chunk_document_tool
from tools.graph_clear_blocked import GraphClearBlockedInput, GraphClearBlockedToolConfig, build_graph_clear_blocked_tool
from tools.graph_distance_recall import GraphDistanceRecallInput, GraphDistanceRecallToolConfig, build_graph_distance_recall_tool
from tools.graph_manage_nodes import (
    GraphEdgeOp,
    GraphKeywordOp,
    GraphManageNodesInput,
    GraphManageNodesToolConfig,
    GraphNodeAction,
    build_graph_manage_nodes_tool,
)
from tools.graph_mark_blocked import GraphMarkBlockedInput, GraphMarkBlockedToolConfig, build_graph_mark_blocked_tool
from tools.graph_mark_useful import GraphMarkUsefulInput, GraphMarkUsefulToolConfig, build_graph_mark_useful_tool
from tools.keyword_recall import KeywordRecallInput, KeywordRecallToolConfig, build_keyword_recall_tool
from tools.list_chunk_documents import ListChunkDocumentsInput, ListChunkDocumentsToolConfig, build_list_chunk_documents_tool
from tools.manage_chunks import EdgeOp, KeywordOp, ChunkManageAction, ManageChunksInput, ManageChunksToolConfig, build_manage_chunks_tool
from tools.query_chunk_positions import QueryChunkPositionsInput, QueryChunkPositionsItem, QueryChunkPositionsToolConfig, build_query_chunk_positions_tool
from tools.read_nodes import ReadNodesInput, ReadNodesToolConfig, build_read_nodes_tool


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(_without_embedding_values(payload), ensure_ascii=False, indent=2), flush=True)


def _without_embedding_values(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"embedding", "keyword_vectors", "node_label"}:
                if isinstance(item, list) and item and isinstance(item[0], list):
                    cleaned[f"{key}_dimensions"] = [len(vector) for vector in item]
                elif isinstance(item, list):
                    cleaned[f"{key}_dimension"] = len(item)
                else:
                    cleaned[key] = "<omitted>"
                continue
            cleaned[key] = _without_embedding_values(item)
        return cleaned
    if isinstance(value, list):
        return [_without_embedding_values(item) for item in value]
    return value


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _call_tool(tool, *, name: str, payload: dict[str, Any], state: dict[str, Any] | None = None):
    _print_event({"type": "tool", "tool": name, "event": "start"})
    kwargs = dict(payload)
    if state is not None:
        kwargs["runtime"] = SimpleNamespace(state=state, tool_call_id=f"{name}-manual")
    result = tool.func(**kwargs)
    _print_event({"type": "tool", "tool": name, "event": "success"})
    if isinstance(result, Command):
        update = dict(result.update or {})
        payload_dict: dict[str, Any] = {}
        for message in update.get("messages") or []:
            if isinstance(message, ToolMessage) and isinstance(message.content, str):
                payload_dict = json.loads(message.content)
                break
        state_update = {key: value for key, value in update.items() if key != "messages"}
        if state_update:
            payload_dict["state_update"] = state_update
        return payload_dict
    return result


def _document_edge_snapshot(*, run_id: str, document_name: str) -> dict[str, Any]:
    with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
        with driver.session() as session:
            record = session.run(
                """
                MATCH (:Chunk {run_id: $run_id, document_name: $document_name})-[edge:DOCUMENT_NEXT]->
                      (:Chunk {run_id: $run_id, document_name: $document_name})
                RETURN count(edge) AS edge_count, collect(DISTINCT edge.dist) AS dists
                """,
                run_id=run_id,
                document_name=document_name,
            ).single()
    return {"edge_count": int(record["edge_count"]), "dists": [float(item) for item in record["dists"]]} if record else {"edge_count": 0, "dists": []}


def _keyword_embedding_snapshot(*, run_id: str, document_name: str) -> dict[str, Any]:
    with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
        with driver.session() as session:
            record = session.run(
                """
                MATCH (:Chunk {run_id: $run_id, document_name: $document_name})-[:HAS_KEYWORD]->
                      (keyword:KeywordNode {run_id: $run_id, document_name: $document_name})
                RETURN count(keyword) AS keyword_count, collect(DISTINCT keyword.dimension) AS dimensions
                """,
                run_id=run_id,
                document_name=document_name,
            ).single()
    return (
        {"keyword_count": int(record["keyword_count"]), "dimensions": [int(item) for item in record["dimensions"]]}
        if record
        else {"keyword_count": 0, "dimensions": []}
    )


def _normal_edge_count(*, document_name: str, first_id: str, third_id: str) -> int:
    with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
        with driver.session() as session:
            record = session.run(
                """
                MATCH (source:Chunk {document_name: $document_name, chunk_id: $first_id})
                MATCH (target:Chunk {document_name: $document_name, chunk_id: $third_id})
                OPTIONAL MATCH (source)-[edge:LINKS]-(target)
                RETURN count(edge) AS edge_count
                """,
                document_name=document_name,
                first_id=first_id,
                third_id=third_id,
            ).single()
    return int(record["edge_count"]) if record else 0


def _run_real_model_checks() -> None:
    chat = get_chat_model_config()
    embedding = get_embedding_model_config()
    base_url = str(chat["base_url"]).rstrip("/")
    reply = _stream_chat_completion(
        f"{base_url}/chat/completions",
        api_key=str(chat["api_key"]),
        payload={
            "model": str(chat["model"]),
            "messages": [
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "Confirm the live all-operations test can start."},
            ],
            "temperature": 0,
        },
    )
    _assert(bool(reply.strip()), "real chat model did not stream any content")
    embedding_response = _post_json(
        f"{str(embedding['base_url']).rstrip('/')}/embeddings",
        api_key=str(embedding["api_key"]),
        payload={
            "model": str(embedding["model"]),
            "input": "live all operations embedding dimension check",
            "dimensions": int(embedding["dimensions"]),
        },
    )
    dimensions_returned = len(embedding_response["data"][0]["embedding"])
    _print_event(
        {
            "type": "embedding",
            "event": "dimension_check",
            "model": embedding["model"],
            "dimensions_requested": int(embedding["dimensions"]),
            "dimensions_returned": dimensions_returned,
        }
    )
    _assert(dimensions_returned == int(embedding["dimensions"]), "embedding dimension check failed")


def _run_chunk_apply() -> None:
    sample_path = PROJECT_ROOT / "workspace" / "knowledge" / f"live_all_apply_{token_hex(4)}.txt"
    sample_path.write_text(
        "真实模型测试第一行。\n"
        "需要完成切分入库。\n"
        "入库后自动生成文档内部边。\n"
        "\n"
        "随后验证重复文档名会被拒绝。\n",
        encoding="utf-8",
    )
    tool = ChunkApplyTool()
    try:
        result = tool.run(
            tool_input=ChunkApplyInput(path=str(sample_path), resume=False, chunking_requirement="按较粗颗粒度切分。"),
            stream_writer=_print_event,
            progress_callback=_print_event,
        )
        _print_result(result)
        _assert(result["status"] == "success", "chunk_apply failed")
        document = result["results"][0]
        chunks = list(document["chunks"])
        with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (chunk:Chunk {document_name: $document_name})
                    RETURN chunk.run_id AS run_id
                    LIMIT 1
                    """,
                    document_name=str(document["document_name"]),
                ).single()
        assert record is not None
        run_id = str(record["run_id"])
        edge_snapshot = _document_edge_snapshot(run_id=run_id, document_name=str(document["document_name"]))
        _assert(edge_snapshot["edge_count"] == max(0, len(chunks) - 1), "chunk_apply DOCUMENT_NEXT edge count mismatch")
        _assert(edge_snapshot["dists"] == [0.3], "chunk_apply DOCUMENT_NEXT dist mismatch")
        keyword_snapshot = _keyword_embedding_snapshot(run_id=run_id, document_name=str(document["document_name"]))
        _print_event({"type": "neo4j", "event": "keyword_embedding_check", **keyword_snapshot})
        _assert(keyword_snapshot["keyword_count"] > 0, "chunk_apply keyword embeddings were not persisted")
        _assert(keyword_snapshot["dimensions"] == [1536], "chunk_apply keyword embedding dimension mismatch")

        duplicate = tool.run(
            tool_input=ChunkApplyInput(path=str(sample_path), resume=False, chunking_requirement="重复检测。"),
            stream_writer=_print_event,
        )
        _print_result(duplicate)
        _assert(duplicate["status"] == "error", "duplicate chunk_apply should be rejected")
        _assert(duplicate["message"] == "当前文档已经在memory中", "duplicate rejection message mismatch")
    finally:
        tool.close()


def _run_document_tools() -> None:
    document_name = f"live_all_document_{token_hex(4)}"
    run_id = f"live_all_document_run_{token_hex(4)}"
    config_path = PROJECT_ROOT / "workspace" / "config" / "database_config.json"
    create_tool = build_create_chunk_document_tool(
        config=CreateChunkDocumentToolConfig(neo4j_config_path=config_path, run_id=run_id, persist_keyword_embeddings=True)
    )
    manage_tool = build_manage_chunks_tool(
        config=ManageChunksToolConfig(neo4j_config_path=config_path, run_id=run_id, persist_keyword_embeddings=True)
    )
    query_tool = build_query_chunk_positions_tool(config=QueryChunkPositionsToolConfig(neo4j_config_path=config_path, run_id=run_id))
    list_tool = build_list_chunk_documents_tool(config=ListChunkDocumentsToolConfig(neo4j_config_path=config_path, run_id=run_id))
    try:
        create_result = _call_tool(
            create_tool,
            name="create_chunk_document",
            payload=CreateChunkDocumentInput(
                document_name=document_name,
                summary="第一段",
                body="第一段内容。",
                keywords=["live", "create"],
            ).model_dump(),
        )
        _print_result(create_result)
        _assert(create_result["status"] == "success", "create_chunk_document failed")

        insert_result = _call_tool(
            manage_tool,
            name="manage_chunks",
            payload=ManageChunksInput(
                document_name=document_name,
                actions=[
                    ChunkManageAction(op="insert", insert_after=0, summary="第二段", body="第二段内容。", keywords=["insert"]),
                    ChunkManageAction(op="insert", insert_after=1, summary="第三段", body="第三段内容。", keywords=["insert"]),
                ],
            ).model_dump(),
        )
        _print_result(insert_result)
        _assert(insert_result["status"] == "success", "manage_chunks insert failed")
        _assert([chunk["chunk_index"] for chunk in insert_result["chunks"]] == [0, 1, 2], "insert did not reindex chunks")
        first_id = str(insert_result["chunks"][0]["id"])
        third_id = str(insert_result["chunks"][2]["id"])

        edge_result = _call_tool(
            manage_tool,
            name="manage_chunks",
            payload=ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="update", chunk_index=0, edge_ops=[EdgeOp(op="+", targets=[2], dist=0.7)])],
            ).model_dump(),
        )
        _print_result(edge_result)
        _assert(edge_result["status"] == "success", "manage_chunks edge_ops create failed")
        _assert(_normal_edge_count(document_name=document_name, first_id=first_id, third_id=third_id) == 1, "edge_ops did not create LINKS")

        update_result = _call_tool(
            manage_tool,
            name="manage_chunks",
            payload=ManageChunksInput(
                document_name=document_name,
                actions=[
                    ChunkManageAction(
                        op="update",
                        chunk_index=1,
                        summary="第二段更新",
                        body="第二段更新内容。",
                        keyword_ops=[KeywordOp(op="+", keywords=["update"])],
                    )
                ],
            ).model_dump(),
        )
        _print_result(update_result)
        _assert(update_result["status"] == "success", "manage_chunks update failed")
        _assert(update_result["chunks"][1]["body"] == "第二段更新内容。", "update body mismatch")
        _assert(update_result["chunks"][1]["keywords"] == ["insert", "update"], "keyword ops append failed")

        delete_result = _call_tool(
            manage_tool,
            name="manage_chunks",
            payload=ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="delete", chunk_index=1)],
            ).model_dump(),
        )
        _print_result(delete_result)
        _assert(delete_result["status"] == "success", "manage_chunks delete failed")
        _assert([chunk["chunk_index"] for chunk in delete_result["chunks"]] == [0, 1], "delete did not reindex chunks")

        query_result = _call_tool(
            query_tool,
            name="query_chunk_positions",
            payload=QueryChunkPositionsInput(
                items=[QueryChunkPositionsItem(document_name=document_name, positions=[[0, 1]], mode="detail")]
            ).model_dump(),
        )
        _print_result(query_result)
        _assert(query_result["chunk_count"] == 2, "query_chunk_positions returned wrong chunk count")
        _assert(query_result["results"][0]["requested_positions"] == [0, 1], "query range expansion failed")

        list_result = _call_tool(
            list_tool,
            name="list_chunk_documents",
            payload=ListChunkDocumentsInput(limit=10).model_dump(),
        )
        _print_result(list_result)
        _assert(list_result["status"] == "success", "list_chunk_documents failed")

        edge_snapshot = _document_edge_snapshot(run_id=run_id, document_name=document_name)
        _assert(edge_snapshot["edge_count"] == 1, "document edge rebuild failed")
        _assert(edge_snapshot["dists"] == [0.3], "document edge dist mismatch")
        keyword_snapshot = _keyword_embedding_snapshot(run_id=run_id, document_name=document_name)
        _print_event({"type": "neo4j", "event": "document_keyword_embedding_check", **keyword_snapshot})
        _assert(keyword_snapshot["keyword_count"] > 0, "document keyword embeddings were not persisted")
        _assert(keyword_snapshot["dimensions"] == [1536], "document keyword embedding dimension mismatch")
        _assert(
            _normal_edge_count(document_name=document_name, first_id=first_id, third_id=third_id) == 1,
            "normal graph edge was changed by document operations",
        )
    finally:
        pass

def _run_graph_tools() -> None:
    run_id = f"live_all_graph_{token_hex(4)}"
    first_id = f"live_all_graph_a_{token_hex(4)}"
    second_id = f"live_all_graph_b_{token_hex(4)}"
    document_name = f"live_all_graph_doc_{token_hex(4)}"
    config_path = PROJECT_ROOT / "workspace" / "config" / "database_config.json"
    graph_manage_tool = build_graph_manage_nodes_tool(
        config=GraphManageNodesToolConfig(neo4j_config_path=config_path, run_id=run_id, persist_keyword_embeddings=True)
    )
    read_tool = build_read_nodes_tool(config=ReadNodesToolConfig(neo4j_config_path=config_path, run_id=run_id))
    keyword_tool = build_keyword_recall_tool(config=KeywordRecallToolConfig(neo4j_config_path=config_path, run_id=run_id))
    distance_tool = build_graph_distance_recall_tool(config=GraphDistanceRecallToolConfig(neo4j_config_path=config_path, run_id=run_id))
    useful_tool = build_graph_mark_useful_tool(config=GraphMarkUsefulToolConfig(neo4j_config_path=config_path, run_id=run_id))
    blocked_tool = build_graph_mark_blocked_tool(config=GraphMarkBlockedToolConfig(neo4j_config_path=config_path, run_id=run_id))
    clear_tool = build_graph_clear_blocked_tool(config=GraphClearBlockedToolConfig(run_id=run_id))
    create_document_tool = build_create_chunk_document_tool(
        config=CreateChunkDocumentToolConfig(neo4j_config_path=config_path, run_id=run_id, persist_keyword_embeddings=False)
    )
    try:
        create_result = _call_tool(
            graph_manage_tool,
            name="graph_manage_nodes",
            payload=GraphManageNodesInput(
                actions=[
                    GraphNodeAction(
                        operation="create",
                        ids=[first_id],
                        summary="live all graph alpha",
                        body="用于验证 graph keyword recall。",
                        keyword_ops=[GraphKeywordOp(op="+", keywords=["live all alpha", "graph recall"])],
                    ),
                    GraphNodeAction(
                        operation="create",
                        ids=[second_id],
                        summary="live all graph beta",
                        body="用于验证 graph distance recall。",
                        keyword_ops=[GraphKeywordOp(op="+", keywords=["live all beta", "graph distance"])],
                        edge_ops=[GraphEdgeOp(op="+", targets=[first_id], dist=0.4)],
                    ),
                ],
            ).model_dump(),
        )
        _print_result(create_result)
        _assert(create_result["status"] == "success", "graph_manage_nodes create failed")

        read_result = _call_tool(
            read_tool,
            name="read_nodes",
            payload=ReadNodesInput(ids=[first_id, second_id], detail_mode="detail").model_dump(),
        )
        _print_result(read_result)
        _assert(read_result["status"] == "success", "read_nodes failed")

        keyword_result = _call_tool(
            keyword_tool,
            name="keyword_recall",
            payload=KeywordRecallInput(query_keywords=["graph recall"], top_k=3).model_dump(),
        )
        _print_result(keyword_result)
        _assert(any(item["node_id"] == first_id for item in keyword_result["results"]), "keyword_recall did not find graph node")

        state: dict[str, Any] = {}
        useful_result = _call_tool(
            useful_tool,
            name="graph_mark_useful",
            payload=GraphMarkUsefulInput(node_ids=[first_id], rationale="matched keyword").model_dump(),
            state=state,
        )
        _print_result(useful_result)
        state.update(useful_result["state_update"])
        blocked_result = _call_tool(
            blocked_tool,
            name="graph_mark_blocked",
            payload=GraphMarkBlockedInput(node_ids=[first_id], rationale="avoid duplicate").model_dump(),
            state=state,
        )
        _print_result(blocked_result)
        state.update(blocked_result["state_update"])
        blocked_keyword = _call_tool(
            keyword_tool,
            name="keyword_recall",
            payload=KeywordRecallInput(query_keywords=["graph recall"], top_k=3).model_dump(),
            state=state,
        )
        _print_result(blocked_keyword)
        _assert(all(item["node_id"] != first_id for item in blocked_keyword["results"]), "blocked graph node was returned")
        clear_result = _call_tool(
            clear_tool,
            name="graph_clear_blocked",
            payload=GraphClearBlockedInput().model_dump(),
            state=state,
        )
        _print_result(clear_result)
        state.update(clear_result["state_update"])

        distance_result = _call_tool(
            distance_tool,
            name="graph_distance_recall",
            payload=GraphDistanceRecallInput(anchor_node_id=second_id, max_distance=0.5).model_dump(),
            state=state,
        )
        _print_result(distance_result)
        _assert(any(item["node_id"] == first_id for item in distance_result["results"]), "graph_distance_recall did not find linked node")

        document_result = _call_tool(
            create_document_tool,
            name="create_chunk_document",
            payload=CreateChunkDocumentInput(
                document_name=document_name,
                summary="graph backed chunk",
                body="用于验证 graph 工具连接 Chunk。",
                keywords=["graph chunk"],
            ).model_dump(),
        )
        _print_result(document_result)
        _assert(document_result["status"] == "success", "create graph chunk document failed")
        chunk_id = str(document_result["chunks"][0]["id"])
        chunk_link_result = _call_tool(
            graph_manage_tool,
            name="graph_manage_nodes",
            payload=GraphManageNodesInput(
                actions=[GraphNodeAction(operation="update", ids=[chunk_id], edge_ops=[GraphEdgeOp(op="+", targets=[second_id], dist=0.6)])],
            ).model_dump(),
        )
        _print_result(chunk_link_result)
        _assert(chunk_link_result["status"] == "success", "graph edge op on chunk failed")
        chunk_delete_result = _call_tool(
            graph_manage_tool,
            name="graph_manage_nodes",
            payload=GraphManageNodesInput(actions=[GraphNodeAction(operation="delete", ids=[chunk_id])]).model_dump(),
        )
        _print_result(chunk_delete_result)
        _assert(chunk_delete_result["status"] == "error", "graph delete should reject Chunk")
    finally:
        pass


def main() -> None:
    _run_real_model_checks()
    _run_chunk_apply()
    _run_document_tools()
    _run_graph_tools()
    _print_event({"type": "live_all_operations", "event": "success"})


if __name__ == "__main__":
    main()
