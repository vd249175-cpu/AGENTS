"""Focused live check for split graph tools linking one graph node to two chunk nodes.

Run with:
    uv run python tests/test_live_graph_link_focus.py
"""

from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path
from secrets import token_hex
import sys
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from middleware import KnowledgeManagerCapabilityMiddleware, KnowledgeManagerMiddlewareConfig
from models import get_chat_model_config, get_embedding_model_config
from server.neo4j import Neo4jConnectionConfig
from tools import ChunkApplyTool, ChunkApplyToolConfig


def _derived_run_id(base_run_id: str, document_name: str) -> str:
    digest = sha1(document_name.encode("utf-8")).hexdigest()[:12]
    return f"{base_run_id}-{digest}"


def _compact(value: Any, *, max_len: int = 180) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item, max_len=max_len) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item, max_len=max_len) for item in value]
    if isinstance(value, str) and len(value) > max_len:
        return value[: max_len - 1] + "…"
    return value


def _print(event: dict[str, Any]) -> None:
    print(json.dumps(_compact(event), ensure_ascii=False, default=str), flush=True)


def _parse_tool_payloads(messages: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload["_tool_name"] = message.name
            payloads.append(payload)
    return payloads


def _parse_inner_tool_payloads(custom_events: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for event in custom_events:
        if not isinstance(event, dict):
            continue
        inner = event.get("event")
        if not isinstance(inner, dict):
            continue
        tools_payload = inner.get("tools")
        if not isinstance(tools_payload, dict):
            continue
        messages = tools_payload.get("messages")
        if not isinstance(messages, list):
            continue
        payloads.extend(_parse_tool_payloads(messages))
    return payloads


def _fetch_chunk_id(*, neo4j: Neo4jConnectionConfig, run_id: str, document_name: str, chunk_index: int) -> str:
    auth = (neo4j.username, neo4j.password)
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        with driver.session(database=neo4j.database) as session:
            record = session.run(
                """
                MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name, chunk_index: $chunk_index})
                RETURN coalesce(chunk.chunk_id, chunk.id) AS node_id
                LIMIT 1
                """,
                run_id=run_id,
                document_name=document_name,
                chunk_index=chunk_index,
            ).single()
    assert record is not None and record["node_id"], f"chunk not found: {document_name}[{chunk_index}]"
    return str(record["node_id"])


def _fetch_graph_id(*, neo4j: Neo4jConnectionConfig, run_id: str, graph_tag: str) -> str:
    auth = (neo4j.username, neo4j.password)
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        with driver.session(database=neo4j.database) as session:
            record = session.run(
                """
                MATCH (graph:GraphNode {run_id: $run_id})
                WHERE graph.summary CONTAINS $graph_tag OR graph.body CONTAINS $graph_tag
                RETURN graph.node_id AS node_id
                ORDER BY graph.updated_at DESC, graph.created_at DESC
                LIMIT 1
                """,
                run_id=run_id,
                graph_tag=graph_tag,
            ).single()
    assert record is not None and record["node_id"], f"graph node not found: {graph_tag}"
    return str(record["node_id"])


def _edge_counts(
    *,
    neo4j: Neo4jConnectionConfig,
    run_id: str,
    graph_id: str,
    imported_document_name: str,
    managed_document_name: str,
) -> tuple[int, int]:
    auth = (neo4j.username, neo4j.password)
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        with driver.session(database=neo4j.database) as session:
            imported = session.run(
                """
                MATCH (:GraphNode {run_id: $run_id, node_id: $graph_id})-[edge:LINKS]-
                      (:Chunk {run_id: $run_id, document_name: $document_name, chunk_index: 0})
                RETURN count(edge) AS edge_count
                """,
                run_id=run_id,
                graph_id=graph_id,
                document_name=imported_document_name,
            ).single()
            managed = session.run(
                """
                MATCH (:GraphNode {run_id: $run_id, node_id: $graph_id})-[edge:LINKS]-
                      (:Chunk {run_id: $run_id, document_name: $document_name, chunk_index: 0})
                RETURN count(edge) AS edge_count
                """,
                run_id=run_id,
                graph_id=graph_id,
                document_name=managed_document_name,
            ).single()
    return (
        int(imported["edge_count"]) if imported else 0,
        int(managed["edge_count"]) if managed else 0,
    )


def _run_agent_phase(agent: Any, *, prompt: str, thread_id: str) -> tuple[list[dict[str, Any]], list[Any]]:
    state = {"messages": [HumanMessage(content=prompt)]}
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 64}
    custom_events: list[Any] = []
    for chunk in agent.stream(state, config=config, stream_mode=["updates", "custom"]):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, payload = chunk
        else:
            mode, payload = "updates", chunk
        if mode == "custom":
            custom_events.append(payload)
    final_state = dict(agent.get_state(config).values)
    return _parse_tool_payloads(final_state.get("messages", [])), custom_events


def main() -> None:
    token = token_hex(4)
    imported_document_name = f"focus_public_doc_{token}"
    managed_document_name = f"focus_managed_doc_{token}"
    graph_tag = f"focus_graph_{token}"
    chunk_run_base = f"focus-public-{token}"
    source_path = PROJECT_ROOT / "workspace" / "knowledge" / f"{imported_document_name}.txt"
    source_path.write_text(
        "第一段：公开入口说明。\n"
        "这里说明 chunk_apply 走 invoke。\n\n"
        "第二段：useful 和 blocked 是运行态桶。\n",
        encoding="utf-8",
    )

    chat = get_chat_model_config()
    embedding = get_embedding_model_config()
    neo4j = Neo4jConnectionConfig.load(PROJECT_ROOT / "workspace" / "config" / "database_config.json")

    chunk_tool = ChunkApplyTool(
        config=ChunkApplyToolConfig(
            identity={"base_run_id": chunk_run_base, "base_thread_id": f"{chunk_run_base}-thread"},
            public={
                "neo4j": neo4j,
                "checkpoint_path": PROJECT_ROOT / "store" / "checkpoint" / f"focus_live_{token}.sqlite3",
                "embedding_provider": embedding["provider"],
                "embedding_model": embedding["model"],
                "embedding_base_url": embedding["base_url"],
                "embedding_api_key": embedding["api_key"],
                "embedding_dimensions": int(embedding["dimensions"]),
            },
            runtime={
                "resume": True,
                "cache_path": PROJECT_ROOT / "store" / "cache" / f"focus_live_{token}.sqlite3",
                "staging_path": PROJECT_ROOT / "store" / "staging" / f"focus_live_{token}.sqlite3",
                "max_retries": 3,
            },
            chunking={
                "history_line_count": 4,
                "active_line_count": 8,
                "preview_line_count": 4,
                "line_wrap_width": 30,
                "window_back_bytes": 1200,
                "window_forward_bytes": 2400,
                "trace_limit": 16,
                "max_retries": 3,
            },
            document_edge_distance=0.3,
            persist_keyword_embeddings=True,
        )
    )
    middleware = None
    try:
        chunk_result = chunk_tool.invoke({"path": str(source_path), "resume": False, "chunking_requirement": "按语义切分"})
        _print(
            {
                "phase": "chunk_apply",
                "status": chunk_result["status"],
                "document": imported_document_name,
                "chunk_count": chunk_result["results"][0]["chunk_count"],
            }
        )

        run_id = _derived_run_id(chunk_run_base, imported_document_name)
        model = init_chat_model(
            model=str(chat["model"]),
            model_provider=str(chat["provider"]),
            base_url=str(chat["base_url"]),
            api_key=str(chat["api_key"]),
            temperature=0.0,
        )
        middleware = KnowledgeManagerCapabilityMiddleware(
            config=KnowledgeManagerMiddlewareConfig(
                neo4j=neo4j,
                run_id=run_id,
                trace_limit=16,
                tool={
                    "temperature": 0.0,
                    "debug": False,
                    "stream_inner_agent": True,
                    "inner_recursion_limit": 64,
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
                    },
                },
            )
        )
        agent = create_agent(
            model=model,
            middleware=[middleware],
            checkpointer=InMemorySaver(),
            debug=False,
            name="focus-parent-agent",
            system_prompt="你是主 agent。每次只调用一次 manage_knowledge。",
        )

        payloads_setup, custom_setup = _run_agent_phase(
            agent,
            prompt=(
                "必须调用 manage_knowledge 一次，并且内部只做两步："
                f"1) create_chunk_document 创建 {managed_document_name} 的第一段，"
                "summary 为“管理侧文档入口”，"
                "body 为“这是一段由知识管理者创建的文档内容，用于连接图节点和公开入口 chunk。”，"
                "keywords 为 [\"管理侧文档\", \"图连接\"]；"
                "2) 返回一句结论。"
            ),
            thread_id=f"focus-setup-{token_hex(4)}",
        )
        _print(
            {
                "phase": "setup_managed_doc",
                "outer_tools": [payload.get("_tool_name") for payload in payloads_setup],
                "custom_count": len(custom_setup),
                "inner_payloads": [
                    {
                        "tool": payload.get("_tool_name"),
                        "operation": payload.get("operation"),
                        "status": payload.get("status"),
                        "message": payload.get("message"),
                    }
                    for payload in _parse_inner_tool_payloads(custom_setup)
                ],
            }
        )

        imported_chunk0_id = _fetch_chunk_id(
            neo4j=neo4j,
            run_id=run_id,
            document_name=imported_document_name,
            chunk_index=0,
        )
        managed_chunk0_id = _fetch_chunk_id(
            neo4j=neo4j,
            run_id=run_id,
            document_name=managed_document_name,
            chunk_index=0,
        )

        payloads_graph, custom_graph = _run_agent_phase(
            agent,
            prompt=(
                "必须调用 manage_knowledge 一次，并且内部严格只做三步："
                f"1) graph_create_nodes 创建一个图节点，summary 和 body 都包含 {graph_tag}，并提供 keyword_ops；"
                "2) graph_update_node 更新同一个图节点时，必须按下面这个 shape 原样组织参数，不要改写字段名："
                f"{{\"id\":\"<创建出来的图节点id>\",\"edge_ops\":[{{\"op\":\"+\",\"targets\":[\"{imported_chunk0_id}\"],\"dist\":0.3}},{{\"op\":\"+\",\"targets\":[\"{managed_chunk0_id}\"],\"dist\":0.3}}]}}"
                "其中 id 只替换成上一步创建出来的图节点 id，其余字段保持这个结构。"
                "3) read_nodes detail 读取这个图节点并返回结论。"
            ),
            thread_id=f"focus-graph-{token_hex(4)}",
        )
        graph_payload = next(payload for payload in payloads_graph if payload.get("_tool_name") == "manage_knowledge")
        inner_graph_payloads = _parse_inner_tool_payloads(custom_graph)
        graph_id = _fetch_graph_id(neo4j=neo4j, run_id=run_id, graph_tag=graph_tag)
        imported_links, managed_links = _edge_counts(
            neo4j=neo4j,
            run_id=run_id,
            graph_id=graph_id,
            imported_document_name=imported_document_name,
            managed_document_name=managed_document_name,
        )
        _print(
            {
                "phase": "graph_link_update",
                "custom_count": len(custom_graph),
                "operation": graph_payload.get("operation"),
                "inner_payloads": [
                    {
                        "tool": payload.get("_tool_name"),
                        "operation": payload.get("operation"),
                        "status": payload.get("status"),
                        "message": payload.get("message"),
                        "first_result": (
                            payload.get("results", [{}])[0].get("message")
                            if isinstance(payload.get("results"), list) and payload.get("results")
                            else None
                        ),
                    }
                    for payload in inner_graph_payloads
                    if payload.get("_tool_name") in {"graph_create_nodes", "graph_update_node", "read_nodes"}
                ],
                "graph_node_id": graph_id,
                "imported_links": imported_links,
                "managed_links": managed_links,
            }
        )

        assert imported_links >= 1
        assert managed_links >= 1
    finally:
        if middleware is not None:
            middleware.close()
        chunk_tool.close()


if __name__ == "__main__":
    main()
