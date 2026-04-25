from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
import sys
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from middleware import KnowledgeManagerCapabilityMiddleware, KnowledgeManagerMiddlewareConfig
from models import get_chat_model_config, get_embedding_model_config
from server.neo4j import Neo4jConnectionConfig


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def _stream_summary(mode: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        keys = sorted(payload.keys())
        if mode == "custom" and isinstance(payload.get("event"), dict):
            return {"mode": mode, "event_keys": sorted(payload["event"].keys())[:8]}
        return {"mode": mode, "keys": keys[:8]}
    return {"mode": mode, "value_type": type(payload).__name__}


def _stream_mode(chunk: Any) -> tuple[str, Any]:
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return chunk
    if isinstance(chunk, dict) and isinstance(chunk.get("type"), str):
        return str(chunk["type"]), chunk
    return "updates", chunk


def main() -> None:
    token = token_hex(4)
    run_id = f"example-manager-{token}"
    thread_id = f"{run_id}-thread"
    document_name = f"example_manager_doc_{token}"
    graph_marker = f"example_manager_graph_{token}"
    log_path = PROJECT_ROOT / "workspace" / "logs" / f"knowledge_manager_example_{token}.json"

    neo4j = Neo4jConnectionConfig(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="1575338771",
        database=None,
    )
    chat = get_chat_model_config()
    embedding = get_embedding_model_config()
    model = init_chat_model(
        model=str(chat["model"]),
        model_provider=str(chat["provider"]),
        base_url=str(chat["base_url"]),
        api_key=str(chat["api_key"]),
        temperature=0.0,
    )

    knowledge_manager = KnowledgeManagerCapabilityMiddleware(
        config=KnowledgeManagerMiddlewareConfig.load_config_knowledge_manager_middleware(
            {
                "neo4j": neo4j.model_dump(),
                "run_id": run_id,
                "trace_limit": 16,
                "tool": {
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
        middleware=[knowledge_manager.middleware],
        checkpointer=InMemorySaver(),
        context_schema=type(knowledge_manager.config),
        system_prompt="你是主 agent。知识库管理任务必须调用 manage_knowledge 一次，拿到结果后只做简短总结。",
        debug=False,
        name="knowledge-manager-example-parent",
    )

    prompt = (
        "请调用 manage_knowledge 一次。target："
        f"创建 Chunk 文档 {document_name}，首段说明公开入口和 manager agent；"
        "再插入第二段说明 useful 桶和普通图边；"
        f"创建一个 GraphNode，summary/body 包含唯一标记 {graph_marker}，关键词包含“公开入口”；"
        "读取刚创建的文档和图节点，把最有帮助的节点放进 useful；"
        "最后用一句话总结完成情况。"
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
    stream_log: list[dict[str, Any]] = []
    update_count = 0
    custom_count = 0
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
        context=knowledge_manager.config,
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        mode, payload = _stream_mode(chunk)
        if mode == "updates":
            update_count += 1
        else:
            custom_count += 1
        stream_log.append({"mode": mode, "payload": payload})

    final_state = dict(agent.get_state(config).values)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"prompt": prompt, "stream": stream_log, "final_state": final_state}, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    _print_event(
        {
            "example": "knowledge_manager",
            "updates": update_count,
            "custom": custom_count,
            "final_keys": sorted(final_state.keys()),
            "log_path": str(log_path),
            "last_event": _stream_summary(stream_log[-1]["mode"], stream_log[-1]["payload"]) if stream_log else None,
        }
    )


if __name__ == "__main__":
    main()
