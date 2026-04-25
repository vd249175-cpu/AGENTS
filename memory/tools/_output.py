from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_INTERNAL_KEYS = {
    "run_id",
    "run_ids",
    "thread_id",
    "base_run_id",
    "base_thread_id",
    "ok",
    "node_label",
    "char_start",
    "char_end",
    "created_at",
    "updated_at",
    "keyword_embedding_provider",
    "keyword_embedding_model",
    "keyword_embedding_dimensions",
}
_VECTOR_KEYS = {"embedding", "keyword_vectors"}


def strip_internal_run_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in _INTERNAL_KEYS:
                continue
            if key in _VECTOR_KEYS:
                continue
            cleaned[key] = strip_internal_run_context(item)
        return cleaned
    if isinstance(value, list):
        return [strip_internal_run_context(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_internal_run_context(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [strip_internal_run_context(item) for item in value]
    return value


def limit_items(value: Sequence[Any], limit: int) -> tuple[list[Any], int, bool]:
    items = list(value)
    total = len(items)
    max_items = max(0, int(limit))
    if total <= max_items:
        return items, total, False
    return items[:max_items], total, True


def top_k_limit_error(*, operation: str, requested_top_k: int, top_k_limit: int) -> dict[str, Any]:
    limit = max(1, int(top_k_limit))
    requested = int(requested_top_k)
    return {
        "operation": operation,
        "status": "error",
        "message": f"top_k 超过工具上限：请求 {requested}，当前 top_k_limit={limit}。请用 top_k <= {limit} 重新调用。",
        "requested_top_k": requested,
        "top_k_limit": limit,
        "suggested_top_k": limit,
    }
