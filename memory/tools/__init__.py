"""Public tool exports."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "ChunkApplyInput",
    "ChunkApplyOverrides",
    "ChunkApplySubState",
    "ChunkApplyTool",
    "ChunkApplyToolConfig",
    "ChunkApplyToolFeedback",
    "ChunkApplyToolSchema",
    "ManageKnowledgeInput",
    "ManageKnowledgeTool",
    "ManageKnowledgeToolConfig",
    "ManageKnowledgeToolFeedback",
    "ManageKnowledgeToolOverride",
    "ManageKnowledgeToolSchema",
    "ManageKnowledgeToolStateTydict",
]


_LAZY_EXPORTS = {
    "ChunkApplyInput": ("tools.chunk_apply", "ChunkApplyInput"),
    "ChunkApplyOverrides": ("tools.chunk_apply", "ChunkApplyOverrides"),
    "ChunkApplySubState": ("tools.chunk_apply", "ChunkApplySubState"),
    "ChunkApplyTool": ("tools.chunk_apply", "ChunkApplyTool"),
    "ChunkApplyToolConfig": ("tools.chunk_apply", "ChunkApplyToolConfig"),
    "ChunkApplyToolFeedback": ("tools.chunk_apply", "ChunkApplyToolFeedback"),
    "ChunkApplyToolSchema": ("tools.chunk_apply", "ChunkApplyToolSchema"),
    "ManageKnowledgeInput": ("tools.manage_knowledge", "ManageKnowledgeInput"),
    "ManageKnowledgeTool": ("tools.manage_knowledge", "ManageKnowledgeTool"),
    "ManageKnowledgeToolConfig": ("tools.manage_knowledge", "ManageKnowledgeToolConfig"),
    "ManageKnowledgeToolFeedback": ("tools.manage_knowledge", "ManageKnowledgeToolFeedback"),
    "ManageKnowledgeToolOverride": ("tools.manage_knowledge", "ManageKnowledgeToolOverride"),
    "ManageKnowledgeToolSchema": ("tools.manage_knowledge", "ManageKnowledgeToolSchema"),
    "ManageKnowledgeToolStateTydict": ("tools.manage_knowledge", "ManageKnowledgeToolStateTydict"),
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
