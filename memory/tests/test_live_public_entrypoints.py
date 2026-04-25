"""Live end-to-end check for the two public entrypoints.

Run with:
    uv run python tests/test_live_public_entrypoints.py
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


LIVE_RECURSION_LIMIT = 80
INNER_TOOL_NAMES = {
    "list_chunk_documents",
    "query_chunk_positions",
    "create_chunk_document",
    "insert_chunks",
    "update_chunks",
    "delete_chunks",
    "graph_create_nodes",
    "graph_update_node",
    "graph_delete_nodes",
    "read_nodes",
    "keyword_recall",
    "graph_distance_recall",
    "graph_mark_useful",
    "graph_mark_blocked",
    "graph_clear_blocked",
}


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
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "{[" and ('"embedding"' in stripped or '"keyword_vectors"' in stripped):
            try:
                parsed = json.loads(value)
            except Exception:
                return value
            return json.dumps(_without_embedding_values(parsed), ensure_ascii=False)
        return value
    return value


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(_without_embedding_values(event), ensure_ascii=False, default=str), flush=True)


def _short_text(value: Any, *, max_len: int = 160) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) <= max_len:
            return stripped
        return stripped[: max_len - 1] + "…"
    if isinstance(value, dict):
        keys = sorted(value.keys())
        summary: dict[str, Any] = {"keys": keys[:12]}
        if len(keys) > 12:
            summary["more_keys"] = len(keys) - 12
        return summary
    return _without_embedding_values(value)


def _stream_event_summary(stream: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"value": _short_text(payload)}

    if stream == "updates":
        return {"keys": sorted(payload.keys())[:12]}

    if stream == "custom":
        inner_event = payload.get("event")
        if isinstance(inner_event, dict):
            summary: dict[str, Any] = {"event_keys": sorted(inner_event.keys())[:12]}
            inner_tools = inner_event.get("tools")
            if isinstance(inner_tools, dict):
                summary["tool_event_keys"] = sorted(inner_tools.keys())[:12]
            return summary
        if isinstance(payload.get("tools"), dict):
            return {"tool_keys": sorted(payload["tools"].keys())[:12]}
        return {"keys": sorted(payload.keys())[:12]}

    summary: dict[str, Any] = {}
    for key in ("phase", "status", "message", "error", "type", "tool", "middleware", "stage", "event"):
        if key in payload:
            value = payload[key]
            summary[key] = _short_text(value)
    for key in ("update_count", "custom_count", "beforeModelCount", "wrapModelCallCount", "toolUseCount"):
        if key in payload:
            summary[key] = payload[key]
    if "tools" in payload and isinstance(payload["tools"], (list, tuple, set)):
        summary["tools"] = list(payload["tools"])[:12]
    if "outer_tool_names" in payload and isinstance(payload["outer_tool_names"], list):
        summary["outer_tool_names"] = payload["outer_tool_names"][:12]
    if "operation" in payload and isinstance(payload["operation"], dict):
        operation = payload["operation"]
        summary["operation"] = {
            key: operation[key]
            for key in sorted(operation)
            if key.endswith("_count") or key in {"summary", "action"}
        }
    if len(summary) == 0:
        summary["keys"] = sorted(payload.keys())[:12]
    return summary


def _parse_tool_payloads(messages: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        content = message.content.strip()
        if not content:
            continue
        try:
            payload = json.loads(content)
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


def _extract_inner_tool_names(custom_events: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for event in custom_events:
        serialized = json.dumps(_without_embedding_values(event), ensure_ascii=False, default=str)
        for tool_name in INNER_TOOL_NAMES:
            if tool_name in serialized:
                names.add(tool_name)
    return names


def _derived_run_id(base_run_id: str, document_name: str) -> str:
    digest = sha1(document_name.encode("utf-8")).hexdigest()[:12]
    return f"{base_run_id}-{digest}"


def _run_stream(agent: Any, *, prompt: str, thread_id: str) -> tuple[int, int, list[dict[str, Any]], dict[str, Any]]:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": LIVE_RECURSION_LIMIT,
    }
    state = {"messages": [HumanMessage(content=prompt)]}
    update_count = 0
    custom_count = 0
    custom_events: list[dict[str, Any]] = []
    for chunk in agent.stream(state, config=config, stream_mode=["updates", "custom"]):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, payload = chunk
        else:
            mode, payload = "updates", chunk
        if mode == "updates":
            update_count += 1
            _print_event(_stream_event_summary("updates", payload))
            continue
        custom_count += 1
        if isinstance(payload, dict):
            custom_events.append(payload)
        else:
            custom_events.append({"value": payload})
        _print_event(_stream_event_summary("custom", payload))
    final_state = dict(agent.get_state(config).values)
    return update_count, custom_count, custom_events, final_state


def _normal_links_snapshot(*, neo4j: Neo4jConnectionConfig, run_id: str, graph_id: str) -> dict[str, int]:
    auth = (neo4j.username, neo4j.password)
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        with driver.session(database=neo4j.database) as session:
            imported = session.run(
                """
                MATCH (graph:GraphNode {run_id: $run_id, node_id: $graph_id})-[edge:LINKS]->
                      (chunk:Chunk {run_id: $run_id, document_name: $document_name, chunk_index: 0})
                RETURN count(edge) AS edge_count
                """,
                run_id=run_id,
                graph_id=graph_id,
                document_name=f"{graph_id}_import_anchor",
            ).single()
    return {"imported_links": int(imported["edge_count"]) if imported else 0}


def _run_manager_phase(
    agent: Any,
    *,
    prompt: str,
    thread_id: str,
    label: str,
) -> tuple[dict[str, Any], set[str], list[dict[str, Any]], dict[str, Any]]:
    update_count, custom_count, custom_events, final_state = _run_stream(
        agent,
        prompt=prompt,
        thread_id=thread_id,
    )
    outer_payloads = _parse_tool_payloads(final_state.get("messages", []))
    _print_event(
        {
            "type": "phase_summary",
            "phase": label,
            "update_count": update_count,
            "custom_count": custom_count,
            "outer_tool_names": [payload.get("_tool_name") for payload in outer_payloads],
        }
    )
    assert update_count > 0
    assert custom_count > 0, f"{label}: knowledge manager did not emit forwarded inner stream events"
    assert any(payload.get("_tool_name") == "manage_knowledge" for payload in outer_payloads), f"{label}: manage_knowledge not called"
    inner_tool_names = _extract_inner_tool_names(custom_events)
    _print_event({"type": "phase_inner_tools", "phase": label, "tools": sorted(inner_tool_names)})
    manage_payload = next(payload for payload in outer_payloads if payload.get("_tool_name") == "manage_knowledge")
    assert isinstance(manage_payload.get("message"), str) and manage_payload["message"], f"{label}: missing summary message"
    assert isinstance(manage_payload.get("operation"), dict), f"{label}: missing operation summary"
    return manage_payload, inner_tool_names, custom_events, final_state


def _fetch_graph_node_id(*, neo4j: Neo4jConnectionConfig, run_id: str, graph_marker: str) -> str:
    auth = (neo4j.username, neo4j.password)
    with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
        with driver.session(database=neo4j.database) as session:
            record = session.run(
                """
                MATCH (graph:GraphNode {run_id: $run_id})
                WHERE graph.summary CONTAINS $graph_marker OR graph.body CONTAINS $graph_marker
                RETURN graph.node_id AS node_id
                ORDER BY graph.updated_at DESC, graph.created_at DESC
                LIMIT 1
                """,
                run_id=run_id,
                graph_marker=graph_marker,
            ).single()
    assert record is not None and record["node_id"], f"graph node not found for marker: {graph_marker}"
    return str(record["node_id"])


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


def main() -> None:
    token = token_hex(4)
    imported_document_name = f"public_entry_doc_{token}"
    managed_document_name = f"public_entry_managed_{token}"
    graph_marker = f"public_entry_graph_{token}"
    temp_graph_marker = f"public_entry_graph_temp_{token}"
    chunk_run_base = f"public-entry-{token}"
    source_path = PROJECT_ROOT / "workspace" / "knowledge" / f"{imported_document_name}.txt"
    source_path.write_text(
        "统一公开入口的第一段。\n"
        "这里强调 chunk_apply 应该走标准 invoke。\n"
        "同时知识管理中间键应该通过外层 agent 统一委派。\n"
        "\n"
        "第二段强调 useful 和 blocked 是运行态桶，不落库。\n"
        "第三段强调图边距离和文档顺序边都必须保留。\n",
        encoding="utf-8",
    )

    chat = get_chat_model_config()
    embedding = get_embedding_model_config()
    neo4j = Neo4jConnectionConfig.load(PROJECT_ROOT / "workspace" / "config" / "database_config.json")

    chunk_tool_config = ChunkApplyToolConfig.load_config_chunk_apply_tool(
        {
            "identity": {
                "base_run_id": chunk_run_base,
                "base_thread_id": f"{chunk_run_base}-thread",
            },
            "public": {
                "neo4j": neo4j.model_dump(),
                "checkpoint_path": str(PROJECT_ROOT / "store" / "checkpoint" / f"live_public_{token}.sqlite3"),
                "embedding_provider": embedding["provider"],
                "embedding_model": embedding["model"],
                "embedding_base_url": embedding["base_url"],
                "embedding_api_key": embedding["api_key"],
                "embedding_dimensions": int(embedding["dimensions"]),
            },
            "runtime": {
                "resume": True,
                "cache_path": str(PROJECT_ROOT / "store" / "cache" / f"live_public_{token}.sqlite3"),
                "staging_path": str(PROJECT_ROOT / "store" / "staging" / f"live_public_{token}.sqlite3"),
                "max_retries": 3,
                "shard_count": 4,
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
    chunk_tool = ChunkApplyTool(config=chunk_tool_config)
    try:
        chunk_events: list[dict[str, Any]] = []
        chunk_result = chunk_tool.invoke(
            {
                "path": str(source_path),
                "resume": False,
                "chunking_requirement": "按语义完整段落切分，摘要里保留“公开入口”和“知识管理”线索。",
            },
            stream_writer=chunk_events.append,
            progress_callback=chunk_events.append,
        )
        for event in chunk_events:
            _print_event(_stream_event_summary("chunk_apply", event))
        _print_event(_stream_event_summary("chunk_apply_final", chunk_result))
        assert chunk_result["status"] == "success"
        assert chunk_result["success_count"] == 1
        assert chunk_events, "chunk_apply did not emit any stream/progress event"

        imported_run_id = _derived_run_id(chunk_run_base, imported_document_name)
        model = init_chat_model(
            model=str(chat["model"]),
            model_provider=str(chat["provider"]),
            base_url=str(chat["base_url"]),
            api_key=str(chat["api_key"]),
            temperature=0.0,
        )
        middleware_config = KnowledgeManagerMiddlewareConfig.load_config_knowledge_manager_middleware(
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
                                "keyword_top_k": 8,
                                "keyword_top_k_limit": 12,
                                "distance_top_k": 8,
                                "distance_top_k_limit": 12,
                                "distance_max_distance": 1.5,
                                "useful_max_items": 16,
                                "useful_max_total_chars": 4000,
                                "blocked_max_items": 16,
                                "blocked_max_total_chars": 4000,
                            }
                        },
                    },
                },
            }
        )
        middleware = KnowledgeManagerCapabilityMiddleware(config=middleware_config)
        agent = create_agent(
            model=model,
            middleware=[middleware],
            checkpointer=InMemorySaver(),
            debug=False,
            name="public-entrypoints-parent-agent",
            system_prompt=(
                "你是主 agent。遇到知识库管理任务时，必须调用 manage_knowledge 一次。"
                "收到结果后只用一句话总结，不要再调用其他工具。"
            ),
        )

        seen_inner_tools: set[str] = set()

        phase_a_payload, phase_a_tools, _, _ = _run_manager_phase(
            agent,
            prompt=(
                "请完成一次文档侧管理委派。必须调用 manage_knowledge 一次，且内部严格只做这些步骤："
                f"1) list_chunk_documents 确认已有文档；"
                f"2) query_chunk_positions 读取 {imported_document_name} 的 chunk 0 与 [0,1] 范围；"
                f"3) create_chunk_document 创建 {managed_document_name} 的第一段，正文需要总结公开入口与知识管理链路；"
                "4) 返回一句结论和 useful。不要做任何图操作。"
            ),
            thread_id=f"public-entry-phase-a-{token_hex(4)}",
            label="phase_a",
        )
        seen_inner_tools.update(phase_a_tools)

        phase_b_payload, phase_b_tools, _, _ = _run_manager_phase(
            agent,
            prompt=(
                "请完成一次文档编辑委派。必须调用 manage_knowledge 一次，且内部严格只做这些步骤："
                f"1) insert_chunks 针对 {managed_document_name} 插入一段第二段临时内容，insert_after=0；"
                "2) update_chunks 更新 chunk_index=0，在正文末尾追加“已核对公开链路”，并通过 keyword_ops 新增关键词“公开链路核对”；"
                f"3) query_chunk_positions 读取 {managed_document_name} 的 chunk 0；"
                "4) 返回一句结论和 useful。不要做任何图操作。"
            ),
            thread_id=f"public-entry-phase-b-{token_hex(4)}",
            label="phase_b",
        )
        seen_inner_tools.update(phase_b_tools)
        assert phase_b_payload["operation"]["update_count"] >= 1

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

        phase_c_payload, phase_c_tools, phase_c_custom, _ = _run_manager_phase(
            agent,
            prompt=(
                "请完成一次图侧检索与状态桶委派。必须调用 manage_knowledge 一次，且内部只做这些步骤："
                f"1) read_nodes detail 读取现有 chunk 节点 {imported_chunk0_id}；"
                f"2) graph_mark_useful 把现有 chunk 节点 {imported_chunk0_id} 放进 useful；"
                f"3) keyword_recall 以“公开入口”“知识管理”“DOCUMENT_NEXT”召回，top_k=3，detail_mode=summary；"
                f"4) graph_distance_recall 以 {imported_chunk0_id} 为 anchor 做 nearby 查询，top_k=3，detail_mode=summary；"
                "5) 返回一句结论和 useful。"
            ),
            thread_id=f"public-entry-phase-c-{token_hex(4)}",
            label="phase_c",
        )
        seen_inner_tools.update(phase_c_tools)
        assert "read_nodes" in phase_c_tools, f"phase_c did not read the graph node: {sorted(phase_c_tools)}"
        imported_chunk1_id = _fetch_chunk_id(
            neo4j=neo4j,
            run_id=imported_run_id,
            document_name=imported_document_name,
            chunk_index=1,
        )

        phase_d_payload, phase_d_tools, _, _ = _run_manager_phase(
            agent,
            prompt=(
                "请完成一次图清理与状态桶委派。必须调用 manage_knowledge 一次，且内部严格只做这些步骤："
                f"1) graph_create_nodes 创建一个持久的 GraphNode，items 只包含一个对象，ids 直接写成 [\"{graph_marker}\"], summary 和 body 里都包含唯一标记 {graph_marker}, keyword_ops 也要显式提供；"
                f"2) read_nodes detail 读取现有图节点 {graph_marker}；"
                f"3) graph_mark_useful 把现有图节点 {graph_marker} 放进 useful；"
                f"4) keyword_recall 以“公开入口”“知识管理”“DOCUMENT_NEXT”召回，top_k=3，detail_mode=summary；"
                f"5) graph_distance_recall 以 {graph_marker} 为 anchor 做 nearby 查询，top_k=3，detail_mode=summary；"
                f"6) graph_create_nodes 新建一个临时图节点，items 只包含一个对象，ids 直接写成 [\"{temp_graph_marker}\"], summary 和 body 里都包含唯一标记 {temp_graph_marker}, keyword_ops 也要显式提供；"
                f"7) graph_delete_nodes 删掉刚才那个临时图节点，ids 直接写成 [\"{temp_graph_marker}\"]；"
                f"8) graph_mark_blocked 临时屏蔽节点 {imported_chunk1_id}；"
                f"9) graph_clear_blocked 清除 {imported_chunk1_id} 的 blocked 状态；"
                "10) 返回一句结论和 useful。"
            ),
            thread_id=f"public-entry-phase-d-{token_hex(4)}",
            label="phase_d",
        )
        seen_inner_tools.update(phase_d_tools)
        assert phase_d_payload["operation"]["delete_count"] >= 1

        phase_e_payload, phase_e_tools, phase_e_custom, _ = _run_manager_phase(
            agent,
            prompt=(
                "请完成一次持久图节点创建与补边委派。必须调用 manage_knowledge 一次，且内部严格只做这些步骤："
                f"1) graph_create_nodes 创建一个 GraphNode，items 只包含一个对象，ids 直接写成 [\"{graph_marker}\"], summary 和 body 里都包含唯一标记 {graph_marker}, keyword_ops 也要显式提供；"
                "2) graph_update_node 更新同一个图节点时，必须按下面这个 shape 原样组织参数，不要改写字段名："
                f"{{\"id\":\"<创建出来的图节点id>\",\"edge_ops\":[{{\"op\":\"+\",\"targets\":[\"{imported_chunk0_id}\"],\"dist\":0.3}},{{\"op\":\"+\",\"targets\":[\"{managed_chunk0_id}\"],\"dist\":0.3}}]}}"
                "其中 id 只替换成上一步创建出来的图节点 id，其余字段保持这个结构。"
                f"3) read_nodes detail 读取 {graph_marker}；"
                "4) 返回一句结论和 useful。"
            ),
            thread_id=f"public-entry-phase-e-{token_hex(4)}",
            label="phase_e",
        )
        seen_inner_tools.update(phase_e_tools)
        assert "update_count" in phase_e_payload["operation"]
        phase_e_inner_payloads = _parse_inner_tool_payloads(phase_e_custom)
        graph_create_payload = next(
            payload for payload in phase_e_inner_payloads if payload.get("_tool_name") == "graph_create_nodes"
        )
        graph_node_results = graph_create_payload.get("results")
        assert isinstance(graph_node_results, list) and graph_node_results, "phase_e did not create a graph node"
        graph_node_id = str(graph_node_results[0].get("ids", [None])[0] or "").strip()
        assert graph_node_id, "phase_e did not return a graph node id"

        phase_f_payload, phase_f_tools, _, _ = _run_manager_phase(
            agent,
            prompt=(
                "请完成一次临时图节点创建与删除委派。必须调用 manage_knowledge 一次，且内部严格只做这些步骤："
                f"1) graph_create_nodes 创建一个临时图节点，items 只包含一个对象，ids 直接写成 [\"{temp_graph_marker}\"], summary 和 body 里都包含唯一标记 {temp_graph_marker}, keyword_ops 也要显式提供；"
                f"2) graph_delete_nodes 删除刚才那个临时图节点，ids 直接写成 [\"{temp_graph_marker}\"]；"
                "3) 返回一句结论和 useful。"
            ),
            thread_id=f"public-entry-phase-f-{token_hex(4)}",
            label="phase_f",
        )
        seen_inner_tools.update(phase_f_tools)
        assert phase_f_payload["operation"]["delete_count"] >= 1
        _print_event({"type": "inner_tools_seen", "tools": sorted(seen_inner_tools)})
        missing_tools = sorted(INNER_TOOL_NAMES - seen_inner_tools)
        assert not missing_tools, f"missing inner tool invocations: {missing_tools}"

        auth = (neo4j.username, neo4j.password)
        with GraphDatabase.driver(neo4j.uri, auth=auth) as driver:
            with driver.session(database=neo4j.database) as session:
                docs_record = session.run(
                    """
                    MATCH (chunk:Chunk {run_id: $run_id})
                    RETURN collect(DISTINCT chunk.document_name) AS document_names
                    """,
                    run_id=imported_run_id,
                ).single()
                managed_count = session.run(
                    """
                    MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
                    RETURN count(chunk) AS chunk_count
                    """,
                    run_id=imported_run_id,
                    document_name=managed_document_name,
                ).single()
                graph_record = session.run(
                    """
                    MATCH (graph:GraphNode {run_id: $run_id, node_id: $graph_id})
                    RETURN count(graph) AS graph_count
                    """,
                    run_id=imported_run_id,
                    graph_id=graph_node_id,
                ).single()
                temp_record = session.run(
                    """
                    MATCH (graph:GraphNode {run_id: $run_id})
                    WHERE graph.summary CONTAINS $graph_marker OR graph.body CONTAINS $graph_marker
                    RETURN count(graph) AS graph_count
                    """,
                    run_id=imported_run_id,
                    graph_marker=temp_graph_marker,
                ).single()
                imported_edge_record = session.run(
                    """
                    MATCH (graph:GraphNode {run_id: $run_id, node_id: $graph_id})-[edge:LINKS]->
                          (chunk:Chunk {run_id: $run_id, document_name: $document_name, chunk_index: 0})
                    RETURN count(edge) AS edge_count
                    """,
                    run_id=imported_run_id,
                    graph_id=graph_node_id,
                    document_name=imported_document_name,
                ).single()
                managed_edge_record = session.run(
                    """
                    MATCH (graph:GraphNode {run_id: $run_id, node_id: $graph_id})-[edge:LINKS]->
                          (chunk:Chunk {run_id: $run_id, document_name: $document_name, chunk_index: 0})
                    RETURN count(edge) AS edge_count
                    """,
                    run_id=imported_run_id,
                    graph_id=graph_node_id,
                    document_name=managed_document_name,
                ).single()
        document_names = list(docs_record["document_names"]) if docs_record is not None else []
        _print_event(
            {
                "type": "neo4j_summary",
                "document_names": document_names,
                "managed_chunk_count": int(managed_count["chunk_count"]) if managed_count else 0,
                "graph_count": int(graph_record["graph_count"]) if graph_record else 0,
                "temp_graph_count": int(temp_record["graph_count"]) if temp_record else 0,
                "imported_links": int(imported_edge_record["edge_count"]) if imported_edge_record else 0,
                "managed_links": int(managed_edge_record["edge_count"]) if managed_edge_record else 0,
            }
        )
        assert imported_document_name in document_names
        assert managed_document_name in document_names
        assert int(managed_count["chunk_count"]) == 1
        assert int(graph_record["graph_count"]) == 1
        assert int(temp_record["graph_count"]) == 0
        assert int(imported_edge_record["edge_count"]) >= 1
        assert int(managed_edge_record["edge_count"]) >= 1
    finally:
        chunk_tool.close()


if __name__ == "__main__":
    main()
