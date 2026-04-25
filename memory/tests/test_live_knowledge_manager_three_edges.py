"""Focused live check for top-level knowledge manager three-edge writes.

Run with:
    uv run python tests/test_live_knowledge_manager_three_edges.py
"""

from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
import sys
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from middleware import KnowledgeManagerCapabilityMiddleware, KnowledgeManagerMiddlewareConfig
from models import get_chat_model_config, get_embedding_model_config
from server.neo4j import Neo4jConnectionConfig
from tools import ChunkApplyTool, ChunkApplyToolConfig
from tests.test_live_public_entrypoints import (
    _derived_run_id,
    _fetch_chunk_id,
    _parse_inner_tool_payloads,
    _parse_tool_payloads,
)


LIVE_RECURSION_LIMIT = 80
LOG_DIR = PROJECT_ROOT / "workspace" / "logs"


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def _without_embedding_values(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"embedding", "keyword_vectors"}:
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
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _without_embedding_values(model_dump())
        except Exception:
            return str(value)
    return value


def _write_full_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_without_embedding_values(payload), ensure_ascii=False, default=str, indent=2), encoding="utf-8")


def _run_manager_phase(
    agent: Any,
    *,
    prompt: str,
    thread_id: str,
    label: str,
    full_log: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": LIVE_RECURSION_LIMIT,
    }
    update_count = 0
    custom_count = 0
    custom_events: list[dict[str, Any]] = []
    for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode=["updates", "custom"]):
        mode, payload = chunk if isinstance(chunk, tuple) and len(chunk) == 2 else ("updates", chunk)
        if mode == "updates":
            update_count += 1
            continue
        custom_count += 1
        if isinstance(payload, dict):
            custom_events.append(payload)
    final_state = dict(agent.get_state(config).values)
    outer_payloads = _parse_tool_payloads(final_state.get("messages", []))
    inner_payloads = _parse_inner_tool_payloads(custom_events)
    inner_tools = sorted({str(payload.get("_tool_name")) for payload in inner_payloads if payload.get("_tool_name")})
    phase_record = {
        "label": label,
        "thread_id": thread_id,
        "prompt": prompt,
        "update_count": update_count,
        "custom_count": custom_count,
        "outer_payloads": outer_payloads,
        "inner_payloads": inner_payloads,
        "custom_events": custom_events,
    }
    full_log.setdefault("phases", []).append(phase_record)
    _print_event(
        {
            "phase": label,
            "updates": update_count,
            "custom": custom_count,
            "outer_tools": [payload.get("_tool_name") for payload in outer_payloads],
            "inner_tools": inner_tools,
        }
    )
    manage_payload = next((payload for payload in outer_payloads if payload.get("_tool_name") == "manage_knowledge"), {})
    return manage_payload, inner_payloads


def _edge_count(*, neo4j: Neo4jConnectionConfig, run_id: str, graph_id: str, chunk_id: str) -> int:
    auth = (neo4j.username, neo4j.password)
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        with driver.session(database=neo4j.database) as session:
            record = session.run(
                """
                MATCH (graph:GraphNode {run_id: $run_id, node_id: $graph_id})-[edge:LINKS]->(chunk:Chunk {run_id: $run_id})
                WHERE coalesce(chunk.chunk_id, chunk.id) = $chunk_id
                RETURN count(edge) AS edge_count
                """,
                run_id=run_id,
                graph_id=graph_id,
                chunk_id=chunk_id,
            ).single()
    return int(record["edge_count"]) if record else 0


def _cleanup_run(*, neo4j: Neo4jConnectionConfig, run_id: str) -> None:
    auth = (neo4j.username, neo4j.password)
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        with driver.session(database=neo4j.database) as session:
            session.run("MATCH (node {run_id: $run_id}) DETACH DELETE node", run_id=run_id).consume()


def main() -> None:
    token = token_hex(4)
    log_path = LOG_DIR / f"live_knowledge_manager_three_edges_{token}.json"
    full_log: dict[str, Any] = {"token": token, "phases": []}
    imported_document_name = f"three_edge_import_{token}"
    managed_document_name = f"three_edge_managed_{token}"
    graph_marker = f"three_edge_graph_{token}"
    chunk_run_base = f"three-edge-{token}"
    source_path = PROJECT_ROOT / "workspace" / "knowledge" / f"{imported_document_name}.txt"
    source_path.write_text(
        "三边测试导入文档。\n"
        "这一段用于确认 ChunkApplyTool 的顶级入口可以先完成文档落库。\n"
        "后续知识管理者会创建普通图节点，并一次性连接三个 chunk。\n",
        encoding="utf-8",
    )

    chat = get_chat_model_config()
    embedding = get_embedding_model_config()
    neo4j = Neo4jConnectionConfig.load(PROJECT_ROOT / "workspace" / "config" / "database_config.json")
    imported_run_id = _derived_run_id(chunk_run_base, imported_document_name)
    chunk_tool = ChunkApplyTool(
        config=ChunkApplyToolConfig.load_config_chunk_apply_tool(
            {
                "identity": {
                    "base_run_id": chunk_run_base,
                    "base_thread_id": f"{chunk_run_base}-thread",
                },
                "public": {
                    "neo4j": neo4j.model_dump(),
                    "checkpoint_path": str(PROJECT_ROOT / "store" / "checkpoint" / f"three_edge_{token}.sqlite3"),
                    "embedding_provider": embedding["provider"],
                    "embedding_model": embedding["model"],
                    "embedding_base_url": embedding["base_url"],
                    "embedding_api_key": embedding["api_key"],
                    "embedding_dimensions": int(embedding["dimensions"]),
                },
                "runtime": {
                    "resume": True,
                    "cache_path": str(PROJECT_ROOT / "store" / "cache" / f"three_edge_{token}.sqlite3"),
                    "staging_path": str(PROJECT_ROOT / "store" / "staging" / f"three_edge_{token}.sqlite3"),
                    "max_retries": 3,
                    "shard_count": 2,
                    "reference_bytes": 6000,
                    "max_workers": 2,
                },
                "chunking": {
                    "history_line_count": 4,
                    "active_line_count": 8,
                    "preview_line_count": 4,
                    "line_wrap_width": 30,
                    "window_back_bytes": 1200,
                    "window_forward_bytes": 2400,
                    "trace_limit": 16,
                    "max_retries": 3,
                },
                "document_edge_distance": 0.3,
                "persist_keyword_embeddings": True,
            }
        )
    )
    try:
        chunk_events: list[dict[str, Any]] = []
        chunk_result = chunk_tool.invoke(
            {
                "path": str(source_path),
                "resume": False,
                "chunking_requirement": "保持单段也可以，摘要中保留“三边测试”和“知识管理者”。",
            },
            stream_writer=chunk_events.append,
            progress_callback=chunk_events.append,
        )
        full_log["chunk_apply"] = {
            "result": chunk_result,
            "events": chunk_events,
        }
        _print_event(
            {
                "phase": "chunk_apply",
                "status": chunk_result.get("status"),
                "success_count": chunk_result.get("success_count"),
                "event_count": len(chunk_events),
            }
        )
        if chunk_result.get("status") != "success" or chunk_result.get("success_count") != 1:
            full_log["final"] = {"status": "setup_failed", "reason": "chunk_apply_failed"}
            _write_full_log(log_path, full_log)
            _print_event({"phase": "final", "status": "setup_failed", "log_path": str(log_path)})
            return

        model = init_chat_model(
            model=str(chat["model"]),
            model_provider=str(chat["provider"]),
            base_url=str(chat["base_url"]),
            api_key=str(chat["api_key"]),
            temperature=0.0,
        )
        middleware = KnowledgeManagerCapabilityMiddleware(
            config=KnowledgeManagerMiddlewareConfig.load_config_knowledge_manager_middleware(
                {
                    "neo4j": neo4j.model_dump(),
                    "run_id": imported_run_id,
                    "trace_limit": 16,
                    "tool": {
                        "temperature": 0.0,
                        "debug": False,
                        "stream_inner_agent": True,
                        "inner_recursion_limit": 72,
                        "agent_overrides": {
                            "model": {
                                "model": str(chat["model"]),
                                "model_provider": str(chat["provider"]),
                                "base_url": str(chat["base_url"]),
                                "api_key": str(chat["api_key"]),
                                "temperature": 0.0,
                            },
                            "embedding": {
                                "provider": str(embedding["provider"]),
                                "model": str(embedding["model"]),
                                "base_url": str(embedding["base_url"]),
                                "api_key": str(embedding["api_key"]),
                                "dimensions": int(embedding["dimensions"]),
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
                                    "blocked_max_total_chars": 3000,
                                }
                            },
                        },
                    },
                }
            )
        )
        agent = create_agent(
            model=model,
            middleware=[middleware],
            checkpointer=InMemorySaver(),
            debug=False,
            name="three-edge-parent-agent",
            system_prompt="你是主 agent。知识库任务必须调用 manage_knowledge 一次，然后只用一句话总结。",
        )

        _run_manager_phase(
            agent,
            prompt=(
                "请把下面目标原样交给 manage_knowledge，一次调用即可："
                f"创建文档 {managed_document_name} 的第一段，然后 insert_chunks 插入第二段。"
                "第一段总结公开入口，第二段总结三边连接测试。"
                "最后 query_chunk_positions 读取该文档 [0,1] 范围。"
            ),
            thread_id=f"three-edge-setup-{token_hex(4)}",
            label="setup_document",
            full_log=full_log,
        )
        try:
            imported_chunk0_id = _fetch_chunk_id(
                neo4j=neo4j,
                run_id=imported_run_id,
                document_name=imported_document_name,
                chunk_index=0,
            )
            managed_chunk0_id = _fetch_chunk_id(
                neo4j=neo4j,
                run_id=imported_run_id,
                document_name=managed_document_name,
                chunk_index=0,
            )
            managed_chunk1_id = _fetch_chunk_id(
                neo4j=neo4j,
                run_id=imported_run_id,
                document_name=managed_document_name,
                chunk_index=1,
            )
        except AssertionError as error:
            full_log["final"] = {"status": "setup_failed", "reason": str(error)}
            _write_full_log(log_path, full_log)
            _print_event({"phase": "final", "status": "setup_failed", "reason": str(error), "log_path": str(log_path)})
            return

        _, graph_inner_payloads = _run_manager_phase(
            agent,
            prompt=(
                "请把下面目标原样交给 manage_knowledge，一次调用即可："
                f"1) graph_create_nodes 创建一个 GraphNode，不要传 ids，summary 和 body 都包含唯一标记 {graph_marker}，keyword_ops 显式新增关键词“三边连接”；"
                "2) 从 graph_create_nodes 的返回 results[0].ids[0] 取真实 node id；"
                "3) graph_update_node 更新同一个真实 node id，一次传入三个 edge_ops，分别连接到 "
                f"{imported_chunk0_id}、{managed_chunk0_id}、{managed_chunk1_id}，dist 都是 0.3；"
                "4) read_nodes detail 读取该真实 node id。"
            ),
            thread_id=f"three-edge-graph-{token_hex(4)}",
            label="create_three_edges",
            full_log=full_log,
        )
        create_payload = next((payload for payload in graph_inner_payloads if payload.get("_tool_name") == "graph_create_nodes"), {})
        results = create_payload.get("results")
        if not isinstance(results, list) or not results:
            full_log["final"] = {"status": "needs_fix", "reason": "graph_create_nodes_missing_results"}
            _write_full_log(log_path, full_log)
            _print_event({"phase": "final", "status": "needs_fix", "reason": "graph_create_nodes_missing_results", "log_path": str(log_path)})
            return
        ids = results[0].get("ids")
        if not isinstance(ids, list) or not ids:
            full_log["final"] = {"status": "needs_fix", "reason": "graph_create_nodes_missing_ids"}
            _write_full_log(log_path, full_log)
            _print_event({"phase": "final", "status": "needs_fix", "reason": "graph_create_nodes_missing_ids", "log_path": str(log_path)})
            return
        graph_id = str(ids[0])
        counts = {
            "imported": _edge_count(neo4j=neo4j, run_id=imported_run_id, graph_id=graph_id, chunk_id=imported_chunk0_id),
            "managed_0": _edge_count(neo4j=neo4j, run_id=imported_run_id, graph_id=graph_id, chunk_id=managed_chunk0_id),
            "managed_1": _edge_count(neo4j=neo4j, run_id=imported_run_id, graph_id=graph_id, chunk_id=managed_chunk1_id),
        }
        expected = {"imported": 1, "managed_0": 1, "managed_1": 1}
        missing = [key for key, value in counts.items() if value < expected[key]]
        graph_update_payloads = [
            payload for payload in graph_inner_payloads if payload.get("_tool_name") == "graph_update_node"
        ]
        final_status = "pass" if not missing else "needs_fix"
        full_log["final"] = {
            "status": final_status,
            "graph_id": graph_id,
            "expected_edge_counts": expected,
            "actual_edge_counts": counts,
            "missing_edges": missing,
            "graph_update_payloads": graph_update_payloads,
        }
        _write_full_log(log_path, full_log)
        _print_event(
            {
                "phase": "neo4j_edges",
                "status": final_status,
                "graph_id": graph_id,
                "edge_counts": counts,
                "missing_edges": missing,
                "graph_update_calls": len(graph_update_payloads),
                "log_path": str(log_path),
            }
        )
    finally:
        if "final" not in full_log:
            full_log["final"] = {"status": "interrupted"}
            _write_full_log(log_path, full_log)
        chunk_tool.close()
        _cleanup_run(neo4j=neo4j, run_id=imported_run_id)


if __name__ == "__main__":
    main()
