"""Live baseline data seeding for the unified database agent.

Run with:
    uv run python tests/test_live_base_dataset.py
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
from models import get_chat_model_config, get_embedding_model_config
from server.neo4j import DocumentStore, GraphStore
from tests.real_model_smoke import _post_json, _stream_chat_completion


LIVE_RECURSION_LIMIT = 80
MANIFEST_PATH = PROJECT_ROOT / "workspace" / "logs" / "live_base_dataset_manifest.json"


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(_without_embedding_values(event), ensure_ascii=False), flush=True)


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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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


def _compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for chunk in chunks:
        compacted.append(
            {
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "chunk_index": chunk.get("chunk_index"),
                "summary": chunk.get("summary"),
                "document_name": chunk.get("document_name"),
            }
        )
    return compacted


def _compact_graph_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for node in nodes:
        compacted.append(
            {
                "node_id": node.get("node_id"),
                "summary": node.get("summary"),
                "edges": node.get("edges") or [],
            }
        )
    return compacted


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
                {"role": "user", "content": "Confirm the baseline database seeding test can start."},
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
            "input": "baseline dataset embedding dimension check",
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

    run_id = f"base-dataset-{token_hex(4)}"
    thread_id = f"base-dataset-thread-{token_hex(4)}"
    source_document = f"base_article_source_{token_hex(4)}"
    refined_document = f"base_article_refined_{token_hex(4)}"
    graph_topic_id = f"base_graph_topic_{token_hex(4)}"
    graph_method_id = f"base_graph_method_{token_hex(4)}"

    agent = create_unified_agent(
        run_id=run_id,
        checkpointer=InMemorySaver(),
    )
    prompt = (
        "请完成一个可复用的数据库基线构建任务，按顺序执行，尽量每一步只调用一个工具。"
        f"1. 用 create_chunk_document 创建文档 {source_document} 的第一个 chunk，主题是“联合 agent 数据库测试基线”。"
        "这一段必须非常短，只写总纲，不超过 2 句；关键词覆盖联合 agent、数据库、基线三个方向即可。"
        "2. 用 insert_chunks 在同一文档里依次插入 4 个更细的 chunk，每个 chunk 只讲一个原子观点，分别是："
        "数据准备、查询与事务、删除与保留、图结构与复用。"
        "每个 body 都控制在 2 句以内，避免再合并多个观点。"
        "3. 用 query_chunk_positions 读取 positions=[0,1,2,3,4]，detail_mode=summary，确认五段都已经落库。"
        f"4. 用 create_chunk_document 再创建一篇新文档 {refined_document}，它要基于前一篇文章提炼出来，"
        f"并明确引用 {source_document} 作为来源；然后再用 insert_chunks 插入 2 个更细的 chunk，"
        "分别补充“方法论提炼”和“删改原则”。"
        "5. 用 graph_create_nodes 创建两个 GraphNode，id 分别是 "
        f"{graph_topic_id} 和 {graph_method_id}。第一个节点写总主题，第二个节点写方法节点，再用 graph_update_node 把第二个节点连接到第一个节点。"
        "注意：keyword_ops 必须写成数组，例如 keyword_ops=[{op:+, keywords:[...]}]；"
        "edge_ops 也必须写成数组，即使只有一条边也不要直接写对象。"
        "6. 用 read_nodes 读取这两个 GraphNode，detail_mode=summary，确认图结构已经落库。"
        "7. 回到第一篇文档，用 delete_chunks 删除 chunk_index=2 的中间段，只删内容，不改图逻辑。"
        "8. 再用 query_chunk_positions 读取 positions=[0,1,2,3]，detail_mode=summary，确认文档删改后仍然通顺。"
        f"9. 用 graph_distance_recall 从 {graph_method_id} 出发，max_distance=0.5，top_k=5，detail_mode=summary，确认图节点仍然连通。"
        "10. 最后只用一句话总结你完成了什么，不要再调用其他工具。"
    )

    update_count, final_state = _run_stream(
        agent,
        prompt=prompt,
        thread_id=thread_id,
    )

    payloads = _tool_payloads(final_state)
    operations = [payload.get("operation") for payload in payloads]
    _print_event({"type": "final", "update_count": update_count, "operations": operations})

    _assert(0 < update_count < 48, "unexpected update count")
    for required in {
        "create_chunk_document",
        "insert_chunks",
        "delete_chunks",
        "query_chunk_positions",
        "graph_create_nodes",
        "graph_update_node",
        "read_nodes",
        "graph_distance_recall",
    }:
        _assert(required in operations, f"missing expected tool operation: {required}")

    document_store = DocumentStore(config_path=PROJECT_ROOT / "workspace" / "config" / "database_config.json", run_id=run_id)
    graph_store = GraphStore(config_path=PROJECT_ROOT / "workspace" / "config" / "database_config.json", run_id=run_id)
    try:
        source_chunks = document_store.list_chunks(document_name=source_document, run_id=run_id)
        refined_chunks = document_store.list_chunks(document_name=refined_document, run_id=run_id)
        graph_nodes = graph_store.read_nodes(ids=[graph_topic_id, graph_method_id], run_id=run_id, detail_mode="summary")
        graph_distance = graph_store.distance_recall(
            anchor_node_id=graph_method_id,
            run_id=run_id,
            max_distance=0.5,
            top_k=5,
            detail_mode="summary",
        )

        _print_result(
            {
                "source_document": {
                    "document_name": source_document,
                    "chunk_count": len(source_chunks),
                    "chunks": _compact_chunks(source_chunks),
                },
                "refined_document": {
                    "document_name": refined_document,
                    "chunk_count": len(refined_chunks),
                    "chunks": _compact_chunks(refined_chunks),
                },
                "graph_nodes": _compact_graph_nodes(graph_nodes.get("results") or []),
                "graph_distance": _without_embedding_values(graph_distance),
            }
        )

        _assert(len(source_chunks) == 4, "source document should keep exactly four chunks after deletion")
        _assert(len(refined_chunks) >= 3, "refined document should keep at least three chunks")
        _assert(len(graph_nodes.get("results") or []) == 2, "graph nodes should remain readable")
        _assert(
            any(
                edge.get("neighbor_node_id") == graph_topic_id
                for node in graph_nodes.get("results") or []
                if node.get("node_id") == graph_method_id
                for edge in node.get("edges") or []
            ),
            "graph edge between topic and method nodes should still exist",
        )
        _assert(
            any(item.get("node_id") == graph_topic_id for item in graph_distance.get("results") or []),
            "graph distance recall lost the topic node",
        )

        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "source_document_name": source_document,
                    "refined_document_name": refined_document,
                    "graph_topic_id": graph_topic_id,
                    "graph_method_id": graph_method_id,
                    "source_chunks": _compact_chunks(source_chunks),
                    "refined_chunks": _compact_chunks(refined_chunks),
                    "graph_nodes": _compact_graph_nodes(graph_nodes.get("results") or []),
                    "operations": operations,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        document_store.close()
        graph_store.close()


if __name__ == "__main__":
    main()
