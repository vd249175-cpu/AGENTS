"""Simple JSON-backed model registry for the chunk module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_DIR = Path(__file__).resolve().parent


def _load_model_config(filename: str) -> dict[str, Any]:
    path = MODEL_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["config_path"] = str(path)
    return payload


def get_shared_model_config() -> dict[str, Any]:
    return _load_model_config("model_config.json")


def get_chat_model_config() -> dict[str, Any]:
    shared = get_shared_model_config()
    chat_model = dict(shared.get("chat_model", {}))
    chat_model["config_path"] = shared["config_path"]
    return chat_model


def get_embedding_model_config() -> dict[str, Any]:
    shared = get_shared_model_config()
    embedding_model = dict(shared.get("embedding_model", {}))
    embedding_model["config_path"] = shared["config_path"]
    return embedding_model


def resolve_embedding_model_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(get_embedding_model_config())
    for key, value in (override or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        config[key] = value
    return config


def get_chunking_model_config() -> dict[str, Any]:
    payload = _load_model_config("chunking_model.json")
    payload["chat_model"] = get_chat_model_config()
    payload["embedding_model"] = get_embedding_model_config()
    return payload


def get_chunk_apply_model_config() -> dict[str, Any]:
    payload = _load_model_config("chunk_apply_model.json")
    payload["chat_model"] = get_chat_model_config()
    payload["embedding_model"] = get_embedding_model_config()
    return payload


def list_available_models() -> list[dict[str, Any]]:
    return [
        get_chunking_model_config(),
        get_chunk_apply_model_config(),
    ]
