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

from models import get_chat_model_config, get_embedding_model_config
from server.neo4j import Neo4jConnectionConfig
from tools import ChunkApplyTool, ChunkApplyToolConfig


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def _summarize_stream(mode: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if mode == "custom":
            return {"mode": mode, "keys": sorted(payload.keys())[:8], "type": payload.get("type"), "event": payload.get("event")}
        return {"mode": mode, "keys": sorted(payload.keys())[:8]}
    return {"mode": mode, "value_type": type(payload).__name__}


def _stream_mode(chunk: Any) -> tuple[str, Any]:
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return chunk
    if isinstance(chunk, dict) and isinstance(chunk.get("type"), str):
        return str(chunk["type"]), chunk
    return "updates", chunk


def main() -> None:
    token = token_hex(4)
    run_id = f"example-chunk-{token}"
    thread_id = f"{run_id}-thread"
    source_path = PROJECT_ROOT / "workspace" / "knowledge" / f"example_chunk_apply_{token}.txt"
    source_path.write_text(
        "公开入口示例第一段。\n"
        "ChunkApplyTool 会读取单个文件，切分成 chunk，并写入 Neo4j。\n"
        "\n"
        "公开入口示例第二段。\n"
        "长文档可以通过 shard_count、max_workers 和 reference_bytes 调整处理速度。\n",
        encoding="utf-8",
    )

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
    chunk_apply_config = ChunkApplyToolConfig.load_config_chunk_apply_tool(
        {
            "identity": {
                "base_run_id": run_id,
                "base_thread_id": thread_id,
            },
            "public": {
                "neo4j": neo4j.model_dump(),
                "checkpoint_path": str(PROJECT_ROOT / "store" / "checkpoint" / f"chunk_apply_example_{token}.sqlite3"),
                "embedding_provider": embedding["provider"],
                "embedding_model": embedding["model"],
                "embedding_base_url": embedding["base_url"],
                "embedding_api_key": embedding["api_key"],
                "embedding_dimensions": int(embedding["dimensions"]),
            },
            "runtime": {
                "resume": True,
                "cache_path": str(PROJECT_ROOT / "store" / "cache" / f"chunk_apply_example_{token}.sqlite3"),
                "staging_path": str(PROJECT_ROOT / "store" / "staging" / f"chunk_apply_example_{token}.sqlite3"),
                "shard_count": 4,
                "reference_bytes": 6000,
                "max_retries": 3,
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
    chunk_apply_wrapper = ChunkApplyTool(config=chunk_apply_config)
    agent = create_agent(
        model=model,
        tools=[chunk_apply_wrapper.tool],
        checkpointer=InMemorySaver(),
        context_schema=type(chunk_apply_config),
        name="chunk-apply-example-agent",
        system_prompt="你是文件入库 agent。用户要求入库文件时，必须调用 chunk_apply。",
    )
    prompt = (
        "请调用 chunk_apply 入库这个文件："
        f"{source_path}。resume=false，chunking_requirement=按语义完整段落切分，"
        "shard_count=4，max_workers=2，reference_bytes=6000。"
    )
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
    stream_log: list[dict[str, Any]] = []
    update_count = 0
    custom_count = 0
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            context=chunk_apply_config,
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            mode, payload = _stream_mode(chunk)
            stream_log.append({"mode": mode, "payload": payload})
            if mode == "updates":
                update_count += 1
            else:
                custom_count += 1
    finally:
        chunk_apply_wrapper.close()

    final_state = dict(agent.get_state(config).values)
    log_path = PROJECT_ROOT / "workspace" / "logs" / f"chunk_apply_example_{token}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps({"prompt": prompt, "stream": stream_log, "final_state": final_state}, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    _print_event(
        {
            "example": "chunk_apply",
            "updates": update_count,
            "custom": custom_count,
            "last_event": _summarize_stream(stream_log[-1]["mode"], stream_log[-1]["payload"]) if stream_log else None,
            "log_path": str(log_path),
        }
    )


if __name__ == "__main__":
    main()
