"""Manual live stream check for graph-side tools.

Run with:
    uv run python tests/test_live_graph_stream.py
"""

import json
from pathlib import Path
from secrets import token_hex
import sys
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_chat_model_config, get_embedding_model_config
from tests.real_model_smoke import _post_json, _stream_chat_completion
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


def _call_tool(tool, *, name: str, payload: dict[str, object], state: dict[str, object] | None = None):
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


def _run_real_model_gate() -> None:
    chat = get_chat_model_config()
    embedding = get_embedding_model_config()
    reply = _stream_chat_completion(
        f"{str(chat['base_url']).rstrip('/')}/chat/completions",
        api_key=str(chat["api_key"]),
        payload={
            "model": str(chat["model"]),
            "messages": [
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "Confirm the graph live stream test can start."},
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
            "input": "graph live embedding dimension check",
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


def main() -> None:
    _run_real_model_gate()
    run_id = f"live_graph_{token_hex(4)}"
    document_name = f"live_graph_doc_{token_hex(4)}"
    first_id = f"live_graph_a_{token_hex(4)}"
    second_id = f"live_graph_b_{token_hex(4)}"
    config_path = PROJECT_ROOT / "workspace" / "config" / "database_config.json"

    create_document_tool = build_create_chunk_document_tool(
        config=CreateChunkDocumentToolConfig(neo4j_config_path=config_path, run_id=run_id, persist_keyword_embeddings=False)
    )
    manage_tool = build_graph_manage_nodes_tool(
        config=GraphManageNodesToolConfig(neo4j_config_path=config_path, run_id=run_id, persist_keyword_embeddings=True)
    )
    read_tool = build_read_nodes_tool(config=ReadNodesToolConfig(neo4j_config_path=config_path, run_id=run_id))
    keyword_tool = build_keyword_recall_tool(config=KeywordRecallToolConfig(neo4j_config_path=config_path, run_id=run_id))
    distance_tool = build_graph_distance_recall_tool(config=GraphDistanceRecallToolConfig(neo4j_config_path=config_path, run_id=run_id))
    useful_tool = build_graph_mark_useful_tool(config=GraphMarkUsefulToolConfig(neo4j_config_path=config_path, run_id=run_id))
    blocked_tool = build_graph_mark_blocked_tool(config=GraphMarkBlockedToolConfig(neo4j_config_path=config_path, run_id=run_id))
    clear_tool = build_graph_clear_blocked_tool(config=GraphClearBlockedToolConfig(run_id=run_id))

    try:
        graph_create = _call_tool(
            manage_tool,
            name="graph_manage_nodes",
            payload=GraphManageNodesInput(
                actions=[
                    GraphNodeAction(
                        operation="create",
                        ids=[first_id],
                        summary="live graph alpha 可观测性节点",
                        body="这个节点用于验证 graph keyword recall 和状态桶。",
                        keyword_ops=[GraphKeywordOp(op="+", keywords=["live graph alpha", "可观测性"])],
                    ),
                    GraphNodeAction(
                        operation="create",
                        ids=[second_id],
                        summary="live graph beta 链接节点",
                        body="这个节点用于验证 graph distance recall。",
                        keyword_ops=[GraphKeywordOp(op="+", keywords=["live graph beta", "距离召回"])],
                        edge_ops=[GraphEdgeOp(op="+", targets=[first_id], dist=0.4)],
                    ),
                ],
            ).model_dump(),
        )
        _print_result(graph_create)
        _assert(graph_create["status"] == "success", "graph_manage_nodes create failed")

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
            payload=KeywordRecallInput(query_keywords=["可观测性"], top_k=3, detail_mode="summary").model_dump(),
        )
        _print_result(keyword_result)
        _assert(keyword_result["status"] == "success", "keyword_recall failed")
        _assert(any(item["node_id"] == first_id for item in keyword_result["results"]), "keyword recall did not find first graph node")

        state: dict[str, Any] = {}
        useful_result = _call_tool(
            useful_tool,
            name="graph_mark_useful",
            payload=GraphMarkUsefulInput(node_ids=[first_id], rationale="keyword recall matched").model_dump(),
            state=state,
        )
        _print_result(useful_result)
        state.update(useful_result["state_update"])
        _assert(first_id in state["useful_items"], "graph_mark_useful did not update useful_items")

        blocked_result = _call_tool(
            blocked_tool,
            name="graph_mark_blocked",
            payload=GraphMarkBlockedInput(node_ids=[first_id], rationale="avoid returning it again").model_dump(),
            state=state,
        )
        _print_result(blocked_result)
        state.update(blocked_result["state_update"])
        _assert(first_id in state["blocked_items"], "graph_mark_blocked did not update blocked_items")
        _assert(first_id not in state["useful_items"], "blocked node was not removed from useful_items")

        blocked_keyword = _call_tool(
            keyword_tool,
            name="keyword_recall",
            payload=KeywordRecallInput(query_keywords=["可观测性"], top_k=3, detail_mode="summary").model_dump(),
            state=state,
        )
        _print_result(blocked_keyword)
        _assert(all(item["node_id"] != first_id for item in blocked_keyword["results"]), "blocked node was returned by keyword recall")

        clear_result = _call_tool(
            clear_tool,
            name="graph_clear_blocked",
            payload=GraphClearBlockedInput().model_dump(),
            state=state,
        )
        _print_result(clear_result)
        state.update(clear_result["state_update"])
        _assert(not state["blocked_items"], "graph_clear_blocked did not clear blocked_items")

        distance_result = _call_tool(
            distance_tool,
            name="graph_distance_recall",
            payload=GraphDistanceRecallInput(anchor_node_id=second_id, max_distance=0.5, top_k=3).model_dump(),
            state=state,
        )
        _print_result(distance_result)
        _assert(any(item["node_id"] == first_id for item in distance_result["results"]), "distance recall did not find linked node")

        document_result = _call_tool(
            create_document_tool,
            name="create_chunk_document",
            payload=CreateChunkDocumentInput(
                document_name=document_name,
                summary="live graph chunk",
                body="这个 chunk 用于验证 graph 能连到文档背书节点。",
                keywords=["graph chunk"],
            ).model_dump(),
        )
        _print_result(document_result)
        _assert(document_result["status"] == "success", "create chunk document for graph failed")
        chunk_id = str(document_result["chunks"][0]["id"])

        link_chunk = _call_tool(
            manage_tool,
            name="graph_manage_nodes",
            payload=GraphManageNodesInput(
                actions=[GraphNodeAction(operation="update", ids=[chunk_id], edge_ops=[GraphEdgeOp(op="+", targets=[second_id], dist=0.6)])],
            ).model_dump(),
        )
        _print_result(link_chunk)
        _assert(link_chunk["status"] == "success", "graph edge op on chunk failed")

        illegal_chunk_delete = _call_tool(
            manage_tool,
            name="graph_manage_nodes",
            payload=GraphManageNodesInput(actions=[GraphNodeAction(operation="delete", ids=[chunk_id])]).model_dump(),
        )
        _print_result(illegal_chunk_delete)
        _assert(illegal_chunk_delete["status"] == "error", "graph delete should reject Chunk")

        delete_graph = _call_tool(
            manage_tool,
            name="graph_manage_nodes",
            payload=GraphManageNodesInput(actions=[GraphNodeAction(operation="delete", ids=[first_id])]).model_dump(),
        )
        _print_result(delete_graph)
        _assert(delete_graph["status"] == "success", "graph physical delete failed")

        _print_event({"type": "live_graph_stream", "event": "success", "run_id": run_id})
    finally:
        pass

if __name__ == "__main__":
    main()
