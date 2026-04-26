"""Agent-local state helpers."""

from typing import Any


def state_keys(state: Any) -> list[str]:
    if isinstance(state, dict):
        return sorted(str(key) for key in state.keys())
    return []

