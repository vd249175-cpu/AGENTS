"""Memory operations copied from the stable implementation."""

from random import randint
from typing import Any, Mapping, TypedDict

from langchain.agents.middleware import AgentState


class MemoryEntry(TypedDict):
    id: str
    label: str
    content: str


class MemoryToolStateTydict(AgentState, total=False):
    memory: list[MemoryEntry]


def normalize_memories(memories: list[MemoryEntry] | None) -> list[MemoryEntry]:
    return list(memories or [])


def get_state_memory(state: Mapping[str, Any] | None) -> list[MemoryEntry]:
    if state is None:
        return []
    value = state.get("memory")
    return list(value) if isinstance(value, list) else []


def find_memory_indexes(memories: list[MemoryEntry], target_id: str) -> list[int]:
    resolved_target = target_id.strip()
    if not resolved_target:
        return []
    return [index for index, item in enumerate(memories) if item["id"] == resolved_target]


def add_memory(memories: list[MemoryEntry], *, label: str, content: str, memory_id: str) -> list[MemoryEntry]:
    updated = list(memories)
    updated.append(
        {
            "id": memory_id,
            "label": label.strip(),
            "content": content.strip(),
        }
    )
    return updated


def generate_unique_four_digit_id(memories: list[MemoryEntry]) -> str:
    used_ids = {item["id"] for item in memories}
    available_ids = [str(number) for number in range(1000, 10000) if str(number) not in used_ids]
    if not available_ids:
        raise ValueError("No available 4-digit ids remain for new memory items.")
    return available_ids[randint(0, len(available_ids) - 1)]


def modify_memory(
    memories: list[MemoryEntry],
    *,
    target: str,
    content: str | None = None,
    label: str | None = None,
) -> tuple[list[MemoryEntry], int]:
    indexes = find_memory_indexes(memories, target)
    if not indexes:
        return memories, 0

    updated = list(memories)
    for index in indexes:
        next_item = dict(updated[index])
        if content is not None:
            next_item["content"] = content.strip()
        if label is not None:
            next_item["label"] = label.strip()
        updated[index] = next_item
    return updated, len(indexes)


def delete_memory(memories: list[MemoryEntry], *, target: str) -> tuple[list[MemoryEntry], int]:
    indexes = find_memory_indexes(memories, target)
    if not indexes:
        return memories, 0
    index_set = set(indexes)
    updated = [item for index, item in enumerate(memories) if index not in index_set]
    return updated, len(indexes)


def memory_summary(memories: list[MemoryEntry] | None) -> str:
    normalized = normalize_memories(memories)
    if not normalized:
        return "Memory is empty."
    lines = [f'- {item["label"]} [{item["id"]}]: {item["content"]}' for item in normalized]
    return "\n".join(lines)


def memory_total_chars(memories: list[dict[str, str]]) -> int:
    return sum(len(str(item.get("label", ""))) + len(str(item.get("content", ""))) for item in memories)


def render_memory_item(memory: MemoryEntry) -> str:
    safe_label = memory["label"].strip() or "memory"
    content = memory["content"].strip()
    return f'<MemoryItem id="{memory["id"]}" label="{safe_label}">{content}</MemoryItem>'


def render_memory_block(memories: list[MemoryEntry] | None) -> str:
    normalized = normalize_memories(memories)
    if not normalized:
        return '<CapabilityState name="memory">\n<Memory>\n</Memory>\n</CapabilityState>'
    body = "\n".join(render_memory_item(memory) for memory in normalized)
    return f'<CapabilityState name="memory">\n<Memory>\n{body}\n</Memory>\n</CapabilityState>'
