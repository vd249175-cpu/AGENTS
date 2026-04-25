"""Live smoke test for the unified database agent.

Run with:
    uv run python tests/test_live_unified_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
import sys
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import create_unified_agent


LIVE_RECURSION_LIMIT = 64


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(_without_embedding_values(event), ensure_ascii=False, default=str), flush=True)


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


def _run_stream(
    agent: Any,
    *,
    prompt: str,
    thread_id: str,
    recursion_limit: int = LIVE_RECURSION_LIMIT,
) -> tuple[int, dict[str, Any]]:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    state = {"messages": [HumanMessage(content=prompt)]}
    update_count = 0
    for event in agent.stream(state, config=config, stream_mode="updates"):
        update_count += 1
        _print_event(event)
    final_state = agent.get_state(config).values
    return update_count, final_state


def _tool_payloads(final_state: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in final_state.get("messages", []):
        if not isinstance(message, ToolMessage):
            continue
        if not isinstance(message.content, str):
            continue
        content = message.content.strip()
        if not content:
            continue
        try:
            payloads.append(json.loads(content))
        except json.JSONDecodeError:
            continue
    return payloads


def main() -> None:
    run_id = f"unified-live-{token_hex(4)}"
    document_name = f"unified_agent_doc_{token_hex(4)}"
    graph_first_id = f"unified_graph_a_{token_hex(4)}"
    graph_second_id = f"unified_graph_b_{token_hex(4)}"
    agent = create_unified_agent(
        run_id=run_id,
        checkpointer=InMemorySaver(),
    )
    prompt = (
        "请完成一个联合数据库测试，尽量按顺序完成，每一步只调用一个工具。"
        f"1. 用 create_chunk_document 创建文档 {document_name} 的第一段，内容围绕“联合 agent 数据库测试”，关键词也围绕这个主题。"
        "2. 用 insert_chunks 在这篇文档里插入第二段，insert_after=0。"
        "3. 必须用 query_chunk_positions 读取 positions=[0,1]，mode=summary，不要用 read_nodes 代替。"
        f"4. 用 graph_create_nodes 创建两个 GraphNode，id 分别是 {graph_first_id} 和 {graph_second_id}，关键词请包含“联合 agent”与“数据库”。"
        f"   然后用 graph_update_node 把 {graph_second_id} 和 {graph_first_id} 连接起来。"
        "5. 用 read_nodes 读取这两个节点，detail_mode=summary。"
        "6. 用 keyword_recall 检索关键词 [\"联合 agent\", \"数据库\"]，detail_mode=summary。"
        f"7. 用 graph_distance_recall 从 {graph_second_id} 扩展，max_distance=0.5，top_k=3，detail_mode=summary。"
        "最后只用一句话总结你完成了什么，不要再调用其他工具。"
    )

    update_count, final_state = _run_stream(
        agent,
        prompt=prompt,
        thread_id=f"unified-agent-thread-{token_hex(4)}",
    )

    payloads = _tool_payloads(final_state)
    operations = [payload.get("operation") for payload in payloads]
    _print_event({"type": "final", "update_count": update_count, "operations": operations})

    assert 0 < update_count < 40
    assert "create_chunk_document" in operations
    assert "insert_chunks" in operations
    assert "query_chunk_positions" in operations
    assert "graph_create_nodes" in operations
    assert "graph_update_node" in operations
    assert "read_nodes" in operations
    assert "keyword_recall" in operations
    assert "graph_distance_recall" in operations

    query_payload = next(payload for payload in payloads if payload.get("operation") == "query_chunk_positions")
    assert query_payload["chunk_count"] == 2

    graph_distance_payload = next(payload for payload in payloads if payload.get("operation") == "graph_distance_recall")
    assert any(item["node_id"] == graph_first_id for item in graph_distance_payload["results"])

    graph_create_payload = next(payload for payload in payloads if payload.get("operation") == "graph_create_nodes")
    created_ids: list[str] = []
    for item in graph_create_payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        created_ids.extend(item.get("created_ids") or item.get("ids") or [])
    if not created_ids:
        created_ids = graph_create_payload.get("created_ids") or graph_create_payload.get("ids") or []
    assert graph_first_id in created_ids
    assert graph_second_id in created_ids


if __name__ == "__main__":
    main()
