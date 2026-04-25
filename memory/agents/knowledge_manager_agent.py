"""Compatibility wrapper for the internal knowledge manager agent builder."""

from server.knowledge_manager_runtime import (
    DEFAULT_KNOWLEDGE_MANAGER_SYSTEM_PROMPT,
    KnowledgeManagerAgent,
    KnowledgeManagerAgentConfig,
    create_knowledge_manager_agent,
)


__all__ = [
    "DEFAULT_KNOWLEDGE_MANAGER_SYSTEM_PROMPT",
    "KnowledgeManagerAgent",
    "KnowledgeManagerAgentConfig",
    "create_knowledge_manager_agent",
]
