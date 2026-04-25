"""Public middleware exports."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "KnowledgeManagerCapabilityOverrides",
    "KnowledgeManagerCapabilityMiddleware",
    "KnowledgeManagerMiddlewareConfig",
    "KnowledgeManagerMiddlewareOverride",
    "KnowledgeManagerMiddlewareSchema",
    "KnowledgeManagerToolConfig",
    "KnowledgeManagerStateTydict",
]


_LAZY_EXPORTS = {
    "ChunkingCapabilityMiddleware": ("middleware.chunking", "ChunkingCapabilityMiddleware"),
    "ChunkingStateTydict": ("middleware.chunking", "ChunkingStateTydict"),
    "DocumentQueryCapabilityMiddleware": ("middleware.document_query", "DocumentQueryCapabilityMiddleware"),
    "DocumentQueryMiddlewareConfig": ("middleware.document_query", "DocumentQueryMiddlewareConfig"),
    "DocumentQueryStateTydict": ("middleware.document_query", "DocumentQueryStateTydict"),
    "DocumentWriteCapabilityMiddleware": ("middleware.document_write", "DocumentWriteCapabilityMiddleware"),
    "DocumentWriteMiddlewareConfig": ("middleware.document_write", "DocumentWriteMiddlewareConfig"),
    "DocumentWriteStateTydict": ("middleware.document_write", "DocumentWriteStateTydict"),
    "GraphQueryCapabilityMiddleware": ("middleware.graph_query", "GraphQueryCapabilityMiddleware"),
    "GraphQueryCapabilityPreset": ("middleware.graph_query", "GraphQueryCapabilityPreset"),
    "GraphQueryMiddlewareConfig": ("middleware.graph_query", "GraphQueryMiddlewareConfig"),
    "GraphQueryStateTydict": ("middleware.graph_query", "GraphQueryStateTydict"),
    "GraphWriteCapabilityMiddleware": ("middleware.graph_write", "GraphWriteCapabilityMiddleware"),
    "GraphWriteMiddlewareConfig": ("middleware.graph_write", "GraphWriteMiddlewareConfig"),
    "GraphWriteStateTydict": ("middleware.graph_write", "GraphWriteStateTydict"),
    "KnowledgeManagerCapabilityOverrides": ("middleware.knowledge_manager", "KnowledgeManagerCapabilityOverrides"),
    "KnowledgeManagerCapabilityMiddleware": ("middleware.knowledge_manager", "KnowledgeManagerCapabilityMiddleware"),
    "KnowledgeManagerMiddlewareConfig": ("middleware.knowledge_manager", "KnowledgeManagerMiddlewareConfig"),
    "KnowledgeManagerMiddlewareOverride": ("middleware.knowledge_manager", "KnowledgeManagerMiddlewareOverride"),
    "KnowledgeManagerMiddlewareSchema": ("middleware.knowledge_manager", "KnowledgeManagerMiddlewareSchema"),
    "KnowledgeManagerToolConfig": ("middleware.knowledge_manager", "KnowledgeManagerToolConfig"),
    "KnowledgeManagerStateTydict": ("middleware.knowledge_manager", "KnowledgeManagerStateTydict"),
    "ManagementDiscoveryItem": ("middleware.management_discovery", "ManagementDiscoveryItem"),
    "ManagementDiscoveryMiddleware": ("middleware.management_discovery", "ManagementDiscoveryMiddleware"),
    "ManagementDiscoveryMiddlewareConfig": ("middleware.management_discovery", "ManagementDiscoveryMiddlewareConfig"),
    "ManagementDiscoveryStateTydict": ("middleware.management_discovery", "ManagementDiscoveryStateTydict"),
    "MemoryAgentConfig": ("middleware.memory", "MemoryAgentConfig"),
    "MemoryCapabilityMiddleware": ("middleware.memory", "MemoryCapabilityMiddleware"),
    "MemoryCapabilityPreset": ("middleware.memory", "MemoryCapabilityPreset"),
    "MemoryMiddlewareConfig": ("middleware.memory", "MemoryMiddlewareConfig"),
    "MemoryStateTydict": ("middleware.memory", "MemoryStateTydict"),
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
