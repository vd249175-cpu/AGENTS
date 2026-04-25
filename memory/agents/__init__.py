"""Internal agent exports."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "build_chunk_apply_agent",
    "build_chunking_agent",
    "KnowledgeManagerAgent",
    "KnowledgeManagerAgentConfig",
]


_LAZY_EXPORTS = {
    "ChunkApplyAgent": ("agents.chunk_apply_agent", "ChunkApplyAgent"),
    "build_chunk_apply_agent": ("agents.chunk_apply_agent", "build_chunk_apply_agent"),
    "ChunkingAgent": ("agents.chunking_agent", "ChunkingAgent"),
    "build_chunking_agent": ("agents.chunking_agent", "build_chunking_agent"),
    "KnowledgeManagerAgent": ("agents.knowledge_manager_agent", "KnowledgeManagerAgent"),
    "KnowledgeManagerAgentConfig": ("agents.knowledge_manager_agent", "KnowledgeManagerAgentConfig"),
    "DEFAULT_UNIFIED_AGENT_SYSTEM_PROMPT": ("agents.unified_agent", "DEFAULT_UNIFIED_AGENT_SYSTEM_PROMPT"),
    "create_unified_agent": ("agents.unified_agent", "create_unified_agent"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
