"""Helpers for merging nested override configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two config dicts without dropping untouched branches."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_dicts(current, value)
            continue
        merged[key] = value
    return merged


def merge_model(
    base_model: ModelT,
    override: BaseModel | Mapping[str, Any] | None,
) -> ModelT:
    """Apply a partial override onto a pydantic model instance."""
    if override is None:
        return base_model
    override_payload = (
        override.model_dump(exclude_none=True)
        if isinstance(override, BaseModel)
        else {key: value for key, value in override.items() if value is not None}
    )
    if not override_payload:
        return base_model
    payload = deep_merge_dicts(base_model.model_dump(), override_payload)
    return type(base_model).model_validate(payload)


__all__ = ["deep_merge_dicts", "merge_model"]
