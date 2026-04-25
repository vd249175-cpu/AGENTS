"""Live smoke test for the knowledge manager middleware.

Run with:
    uv run python tests/test_live_knowledge_manager_middleware.py
"""

from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
import sys
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from middleware import KnowledgeManagerCapabilityMiddleware, KnowledgeManagerMiddlewareConfig
from models import get_chat_model_config


LIVE_RECURSION_LIMIT = 32


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
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payload["_tool_name"] = message.name
            payloads.append(payload)
    return payloads


def main() -> None:
    run_id = f"knowledge-middleware-live-{token_hex(4)}"
    document_name = f"knowledge_mw_doc_{token_hex(4)}"
    graph_id = f"knowledge_mw_graph_{token_hex(4)}"
    chat = get_chat_model_config()
    model = init_chat_model(
        model=str(chat["model"]),
        model_provider=str(chat["provider"]),
        base_url=str(chat["base_url"]),
        api_key=str(chat["api_key"]),
        temperature=0.0,
    )
    middleware = KnowledgeManagerCapabilityMiddleware(
        config=KnowledgeManagerMiddlewareConfig(
            neo4j_config_path=PROJECT_ROOT / "workspace" / "config" / "database_config.json",
            run_id=run_id,
            trace_limit=16,
        )
    )
    agent = create_agent(
        model=model,
        middleware=[middleware],
        checkpointer=InMemorySaver(),
        debug=False,
        name="knowledge-manager-parent-agent",
        system_prompt=(
            "你是主 agent。遇到知识库管理子任务时，调用 manage_knowledge 一次并等待它回传 useful 信息与交接结论。"
            "收到结果后只做一句话总结，不要再调用其他工具。"
        ),
    )
    prompt = (
        "请完成一次知识库管理委派。"
        "必须调用 manage_knowledge 一次。"
        f"target 要求：创建文档 {document_name} 的第一段，正文强调“useful 节点要保留正文和边”；"
        f"再创建图节点 {graph_id}；"
        f"读取这个节点；"
        f"然后用 graph_mark_useful 标记 {graph_id}；"
        "最后把 useful 信息和关键发现交回主 agent。"
    )

    update_count, final_state = _run_stream(
        agent,
        prompt=prompt,
        thread_id=f"knowledge-manager-parent-thread-{token_hex(4)}",
    )
    payloads = _tool_payloads(final_state)
    operations = [payload.get("_tool_name") for payload in payloads]
    _print_event({"type": "final", "update_count": update_count, "operations": operations})

    assert 0 < update_count < 20
    assert "manage_knowledge" in operations
    payload = next(payload for payload in payloads if payload.get("_tool_name") == "manage_knowledge")
    assert isinstance(payload.get("message"), str) and payload["message"]
    assert isinstance(payload.get("operation"), dict)
    assert "summary" in payload["operation"]
    assert "useful_count" not in payload
    assert len(payload["useful_items"]) >= 1
    useful_item = payload["useful_items"][0]
    assert set(useful_item.keys()) == {"node_id", "body", "edges"}
    assert useful_item.get("body")
    assert "edges" in useful_item


if __name__ == "__main__":
    main()
