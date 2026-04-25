"""Bridge SeedAgent to the copied memory package."""

import sys
from pathlib import Path
from typing import Any

from .path_resolver import AGENT_ROOT, STORE_ROOT


PROJECT_ROOT = AGENT_ROOT.parents[2]
MEMORY_ROOT = PROJECT_ROOT / "memory"
MEMORY_STORE_ROOT = STORE_ROOT / "memory"


def ensure_memory_package_path() -> None:
    memory_path = str(MEMORY_ROOT)
    if memory_path not in sys.path:
        sys.path.append(memory_path)


def _store_path(config: Any, attr_name: str, default_relative: str) -> str:
    value = getattr(config, attr_name, None)
    if value:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return str(path)
        return str((STORE_ROOT / path).resolve())
    return str((MEMORY_STORE_ROOT / default_relative).resolve())


def _neo4j_payload(config: Any) -> dict[str, Any]:
    return {
        "uri": getattr(config, "neo4jUri", "neo4j://localhost:7687"),
        "username": getattr(config, "neo4jUsername", "neo4j"),
        "password": getattr(config, "neo4jPassword", ""),
        "database": getattr(config, "neo4jDatabase", None),
    }


def _chat_override(config: Any) -> dict[str, Any]:
    return {
        "model": getattr(config, "chatModel", None),
        "model_provider": getattr(config, "chatModelProvider", None),
        "base_url": getattr(config, "chatBaseUrl", None),
        "api_key": getattr(config, "chatApiKey", None),
        "temperature": float(getattr(config, "chatTemperature", 0.0)),
    }


def _embedding_override(config: Any) -> dict[str, Any]:
    return {
        "provider": getattr(config, "embeddingProvider", None),
        "model": getattr(config, "embeddingModel", None),
        "base_url": getattr(config, "embeddingBaseUrl", None),
        "api_key": getattr(config, "embeddingApiKey", None),
        "dimensions": int(getattr(config, "embeddingDimensions", 1536) or 1536),
    }


def _knowledge_run_id(config: Any, agent_name: str) -> str:
    return str(getattr(config, "knowledgeRunId", None) or f"{agent_name}-knowledge")


def build_chunk_apply_config(*, agent_name: str, config: Any):
    ensure_memory_package_path()
    from tools import ChunkApplyToolConfig

    MEMORY_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoint", "cache", "staging"):
        (MEMORY_STORE_ROOT / child).mkdir(parents=True, exist_ok=True)

    embedding = _embedding_override(config)
    run_id = _knowledge_run_id(config, agent_name)
    return ChunkApplyToolConfig.load_config_chunk_apply_tool(
        {
            "identity": {
                "base_run_id": run_id,
                "base_thread_id": f"{run_id}-thread",
                "derive_document_run_id": bool(getattr(config, "chunkApplyDeriveDocumentRunId", False)),
            },
            "public": {
                "neo4j": _neo4j_payload(config),
                "checkpoint_path": _store_path(
                    config,
                    "chunkApplyCheckpointPath",
                    "checkpoint/chunk_apply.sqlite3",
                ),
                "embedding_provider": embedding["provider"],
                "embedding_model": embedding["model"],
                "embedding_base_url": embedding["base_url"],
                "embedding_api_key": embedding["api_key"],
                "embedding_dimensions": embedding["dimensions"],
            },
            "runtime": {
                "resume": bool(getattr(config, "resume", True)),
                "cache_path": _store_path(config, "chunkApplyCachePath", "cache/chunk_cache.sqlite3"),
                "staging_path": _store_path(config, "chunkApplyStagingPath", "staging/chunk_staging.sqlite3"),
                "recursion_limit": getattr(config, "chunkApplyRecursionLimit", None),
                "shard_count": int(getattr(config, "shardCount", 4)),
                "reference_bytes": int(getattr(config, "referenceBytes", 6000)),
                "max_retries": int(getattr(config, "chunkApplyMaxRetries", 3)),
                "max_workers": int(getattr(config, "maxWorkers", 2)),
            },
            "chunking": {
                "history_line_count": int(getattr(config, "chunkHistoryLineCount", 4)),
                "active_line_count": int(getattr(config, "chunkActiveLineCount", 8)),
                "preview_line_count": int(getattr(config, "chunkPreviewLineCount", 4)),
                "line_wrap_width": int(getattr(config, "chunkLineWrapWidth", 30)),
                "window_back_bytes": getattr(config, "chunkWindowBackBytes", 1200),
                "window_forward_bytes": getattr(config, "chunkWindowForwardBytes", 2400),
                "trace_limit": int(getattr(config, "chunkTraceLimit", 16)),
                "max_retries": int(getattr(config, "chunkMaxRetries", 3)),
            },
            "document_edge_distance": float(getattr(config, "documentEdgeDistance", 0.3)),
            "persist_keyword_embeddings": bool(getattr(config, "persistKeywordEmbeddings", True)),
        }
    )


def build_chunk_apply_tool(*, agent_name: str, config: Any):
    ensure_memory_package_path()
    from tools import ChunkApplyTool

    return ChunkApplyTool(config=build_chunk_apply_config(agent_name=agent_name, config=config))


def build_knowledge_manager_middleware(*, agent_name: str, config: Any):
    ensure_memory_package_path()
    from middleware import KnowledgeManagerCapabilityMiddleware, KnowledgeManagerMiddlewareConfig

    middleware_config = KnowledgeManagerMiddlewareConfig.load_config_knowledge_manager_middleware(
        build_knowledge_manager_config_payload(agent_name=agent_name, config=config)
    )
    return KnowledgeManagerCapabilityMiddleware(config=middleware_config)


def build_knowledge_manager_config_payload(*, agent_name: str, config: Any) -> dict[str, Any]:
    return {
        "neo4j": _neo4j_payload(config),
        "run_id": _knowledge_run_id(config, agent_name),
        "trace_limit": int(getattr(config, "knowledgeTraceLimit", 16)),
        "tool": {
            "temperature": float(getattr(config, "knowledgeManagerTemperature", 0.0)),
            "debug": bool(getattr(config, "knowledgeManagerDebug", False)),
            "stream_inner_agent": bool(getattr(config, "streamInnerKnowledgeAgent", True)),
            "inner_recursion_limit": int(getattr(config, "innerKnowledgeRecursionLimit", 64)),
            "agent_overrides": {
                "model": _chat_override(config),
                "embedding": _embedding_override(config),
                "system_prompt": getattr(config, "knowledgeManagerSystemPrompt", None),
                "debug": bool(getattr(config, "knowledgeManagerDebug", False)),
                "discovery": {
                    "max_items": getattr(config, "managementDiscoveryMaxItems", None),
                    "max_total_chars": getattr(config, "managementDiscoveryMaxTotalChars", None),
                    "max_summary_chars": getattr(config, "managementDiscoveryMaxSummaryChars", None),
                    "scan_message_limit": getattr(config, "managementDiscoveryScanMessageLimit", None),
                },
                "document_query": {
                    "trace_limit": int(getattr(config, "documentQueryTraceLimit", 16)),
                },
                "document_write": {
                    "trace_limit": int(getattr(config, "documentWriteTraceLimit", 16)),
                },
                "graph_query": {
                    "trace_limit": int(getattr(config, "graphQueryTraceLimit", 16)),
                    "capability_preset": {
                        "keyword_top_k": int(getattr(config, "graphQueryKeywordTopK", 6)),
                        "keyword_top_k_limit": int(getattr(config, "graphQueryKeywordTopKLimit", 10)),
                        "distance_top_k": int(getattr(config, "graphQueryDistanceTopK", 6)),
                        "distance_top_k_limit": int(getattr(config, "graphQueryDistanceTopKLimit", 10)),
                        "distance_max_distance": float(getattr(config, "graphQueryDistanceMaxDistance", 1.5)),
                        "useful_max_items": int(getattr(config, "graphQueryUsefulMaxItems", 12)),
                        "useful_max_total_chars": int(getattr(config, "graphQueryUsefulMaxTotalChars", 3000)),
                        "blocked_max_items": int(getattr(config, "graphQueryBlockedMaxItems", 12)),
                        "blocked_max_total_chars": int(getattr(config, "graphQueryBlockedMaxTotalChars", 3000)),
                    },
                },
                "graph_write": {
                    "trace_limit": int(getattr(config, "graphWriteTraceLimit", 16)),
                },
            },
        },
    }
