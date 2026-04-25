"""Bridge SeedAgent to the copied memory package."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from .path_resolver import AGENT_ROOT, STORE_ROOT


PROJECT_ROOT = AGENT_ROOT.parents[2]
MEMORY_ROOT = PROJECT_ROOT / "memory"
MODEL_CONFIG_PATH = AGENT_ROOT / "Models" / "model_config.json"
MEMORY_STORE_ROOT = STORE_ROOT / "memory"


def ensure_memory_package_path() -> None:
    memory_path = str(MEMORY_ROOT)
    if memory_path not in sys.path:
        sys.path.append(memory_path)


def _load_model_config() -> dict[str, Any]:
    data = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model config: {MODEL_CONFIG_PATH}")
    return data


def _model_section(name: str) -> dict[str, Any]:
    section = _load_model_config().get(name)
    return section if isinstance(section, dict) else {}


def _neo4j_payload() -> dict[str, Any]:
    return {
        "uri": os.getenv("LANGVIDEO_NEO4J_URI") or os.getenv("NEO4J_URI") or "neo4j://localhost:7687",
        "username": os.getenv("LANGVIDEO_NEO4J_USERNAME") or os.getenv("NEO4J_USERNAME") or "neo4j",
        "password": os.getenv("LANGVIDEO_NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD") or "1575338771",
        "database": os.getenv("LANGVIDEO_NEO4J_DATABASE") or os.getenv("NEO4J_DATABASE") or None,
    }


def _chat_override() -> dict[str, Any]:
    chat = _model_section("chat_model")
    return {
        "model": chat.get("model"),
        "model_provider": chat.get("provider"),
        "base_url": chat.get("base_url"),
        "api_key": chat.get("api_key"),
        "temperature": float(chat.get("temperature", 0.0)),
    }


def _embedding_override() -> dict[str, Any]:
    embedding = _model_section("embedding_model")
    return {
        "provider": embedding.get("provider"),
        "model": embedding.get("model"),
        "base_url": embedding.get("base_url"),
        "api_key": embedding.get("api_key"),
        "dimensions": int(embedding.get("dimensions") or 1536),
    }


def _knowledge_run_id(agent_name: str) -> str:
    return os.getenv("LANGVIDEO_KNOWLEDGE_RUN_ID") or f"{agent_name}-knowledge"


def build_chunk_apply_config(*, agent_name: str):
    ensure_memory_package_path()
    from tools import ChunkApplyToolConfig

    MEMORY_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoint", "cache", "staging"):
        (MEMORY_STORE_ROOT / child).mkdir(parents=True, exist_ok=True)

    embedding = _embedding_override()
    return ChunkApplyToolConfig.load_config_chunk_apply_tool(
        {
            "identity": {
                "base_run_id": _knowledge_run_id(agent_name),
                "base_thread_id": f"{_knowledge_run_id(agent_name)}-thread",
                "derive_document_run_id": False,
            },
            "public": {
                "neo4j": _neo4j_payload(),
                "checkpoint_path": str(MEMORY_STORE_ROOT / "checkpoint" / "chunk_apply.sqlite3"),
                "embedding_provider": embedding["provider"],
                "embedding_model": embedding["model"],
                "embedding_base_url": embedding["base_url"],
                "embedding_api_key": embedding["api_key"],
                "embedding_dimensions": embedding["dimensions"],
            },
            "runtime": {
                "resume": True,
                "cache_path": str(MEMORY_STORE_ROOT / "cache" / "chunk_cache.sqlite3"),
                "staging_path": str(MEMORY_STORE_ROOT / "staging" / "chunk_staging.sqlite3"),
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


def build_chunk_apply_tool(*, agent_name: str):
    ensure_memory_package_path()
    from tools import ChunkApplyTool

    return ChunkApplyTool(config=build_chunk_apply_config(agent_name=agent_name))


def build_knowledge_manager_middleware(*, agent_name: str):
    ensure_memory_package_path()
    from middleware import KnowledgeManagerCapabilityMiddleware, KnowledgeManagerMiddlewareConfig

    config = KnowledgeManagerMiddlewareConfig.load_config_knowledge_manager_middleware(
        {
            "neo4j": _neo4j_payload(),
            "run_id": _knowledge_run_id(agent_name),
            "trace_limit": 16,
            "tool": {
                "temperature": 0.0,
                "debug": False,
                "stream_inner_agent": True,
                "inner_recursion_limit": 64,
                "agent_overrides": {
                    "model": _chat_override(),
                    "embedding": _embedding_override(),
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
    return KnowledgeManagerCapabilityMiddleware(config=config)
