"""Memory service copied from the stable implementation and adapted to dict returns."""

from dataclasses import dataclass
from typing import Any, Mapping

from server.memory_state import (
    add_memory,
    delete_memory,
    generate_unique_four_digit_id,
    get_state_memory,
    memory_summary,
    memory_total_chars,
    modify_memory,
    normalize_memories,
)


@dataclass(frozen=True)
class MemoryCapabilityPreset:
    max_items: int = 32
    max_total_chars: int = 4096

    def __post_init__(self) -> None:
        if self.max_items <= 0:
            raise ValueError("max_items must be greater than 0")
        if self.max_total_chars <= 0:
            raise ValueError("max_total_chars must be greater than 0")


class MemoryService:
    def __init__(self, *, preset: MemoryCapabilityPreset | None = None) -> None:
        self.preset = preset or MemoryCapabilityPreset()

    def get_state_memory(self, state: Mapping[str, Any] | None) -> list[dict[str, str]]:
        return get_state_memory(state)

    def summarize_state(self, state: Mapping[str, Any] | None) -> str:
        return memory_summary(self.get_state_memory(state))

    def normalize_state(self, state: Mapping[str, Any] | None) -> dict[str, object]:
        raw_state = state or {}
        monitor_trace = raw_state.get("monitor_trace")
        normalized_trace = list(monitor_trace) if isinstance(monitor_trace, list) else []
        return {
            "memory": normalize_memories(self.get_state_memory(state)),
            "monitor_trace": normalized_trace,
        }

    def apply_operations(
        self,
        *,
        state: Mapping[str, Any] | None,
        items: list[Any],
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        if not items:
            return self._error_payload(tool_call_id=tool_call_id, message="items must not be empty")

        current_memories = self.get_state_memory(state)
        results: list[dict[str, str]] = []

        for action in items:
            operation = str(getattr(action, "operation", "") or _get(action, "operation") or "")
            if operation == "create":
                memory_id = getattr(action, "id", None) or _get(action, "id") or generate_unique_four_digit_id(current_memories)
                if any(item["id"] == memory_id for item in current_memories):
                    return self._error_payload(tool_call_id=tool_call_id, message=f"Memory id '{memory_id}' already exists.")
                label = getattr(action, "label", None) or _get(action, "label")
                content = getattr(action, "content", None) or _get(action, "content")
                candidate_memories = add_memory(
                    current_memories,
                    label=str(label),
                    content=str(content),
                    memory_id=str(memory_id),
                )
                if not self._capacity_within_limits(candidate_memories):
                    return self._capacity_error_payload(tool_call_id=tool_call_id, memories=candidate_memories)
                current_memories = candidate_memories
                message = f"Added memory {memory_id} for label '{label}'."
            elif operation == "update":
                memory_id = str(getattr(action, "id", None) or _get(action, "id") or "")
                content = getattr(action, "content", None) if getattr(action, "content", None) is not None else _get(action, "content")
                label = getattr(action, "label", None) if getattr(action, "label", None) is not None else _get(action, "label")
                candidate_memories, count = modify_memory(
                    current_memories,
                    target=memory_id,
                    content=str(content) if content is not None else None,
                    label=str(label) if label is not None else None,
                )
                if count == 0:
                    return self._error_payload(tool_call_id=tool_call_id, message=f"No memory found for id '{memory_id}'.")
                if not self._capacity_within_limits(candidate_memories):
                    return self._capacity_error_payload(tool_call_id=tool_call_id, memories=candidate_memories)
                current_memories = candidate_memories
                message = f"Modified {count} memory item(s) for id '{memory_id}'."
            elif operation == "delete":
                memory_id = str(getattr(action, "id", None) or _get(action, "id") or "")
                candidate_memories, count = delete_memory(current_memories, target=memory_id)
                if count == 0:
                    return self._error_payload(tool_call_id=tool_call_id, message=f"No memory found for id '{memory_id}'.")
                if not self._capacity_within_limits(candidate_memories):
                    return self._capacity_error_payload(tool_call_id=tool_call_id, memories=candidate_memories)
                current_memories = candidate_memories
                message = f"Deleted {count} memory item(s) for id '{memory_id}'."
            else:
                return self._error_payload(tool_call_id=tool_call_id, message=f"unsupported memory operation: {operation}")

            results.append({"operation": operation, "message": message})

        return {
            "operation": "manage_memory",
            "status": "success",
            "message": "Batch completed successfully.",
            "results": results,
            "memory": current_memories,
            "state_update": {"memory": current_memories},
            "tool_call_id": tool_call_id,
            "usage": {"items": len(current_memories), "total_chars": memory_total_chars(current_memories)},
            "limits": {"max_items": self.preset.max_items, "max_total_chars": self.preset.max_total_chars},
        }

    def _capacity_within_limits(self, memories: list[dict[str, str]]) -> bool:
        return len(memories) <= self.preset.max_items and memory_total_chars(memories) <= self.preset.max_total_chars

    def _capacity_error_payload(self, *, tool_call_id: str | None, memories: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "operation": "manage_memory",
            "status": "error",
            "message": "记忆容量太多，请压缩或删除一些记忆后再添加。",
            "results": [],
            "state_update": {},
            "tool_call_id": tool_call_id,
            "limits": {"max_items": self.preset.max_items, "max_total_chars": self.preset.max_total_chars},
            "usage": {"items": len(memories), "total_chars": memory_total_chars(memories)},
        }

    @staticmethod
    def _error_payload(*, tool_call_id: str | None, message: str) -> dict[str, Any]:
        return {
            "operation": "manage_memory",
            "status": "error",
            "message": message,
            "results": [],
            "state_update": {},
            "tool_call_id": tool_call_id,
        }


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None

