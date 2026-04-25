"""Keyword embedding helpers backed by the shared model registry."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.embeddings import init_embeddings
from models import resolve_embedding_model_config


def keyword_embedding_profile(*, config_override: dict[str, Any] | None = None) -> dict[str, Any]:
    config = resolve_embedding_model_config(config_override)
    return {
        "provider": str(config.get("provider") or "openai"),
        "model": str(config.get("model") or "embedding"),
        "dimensions": int(config.get("dimensions") or 1536),
    }


def keyword_embedding_index_name(*, config_override: dict[str, Any] | None = None) -> str:
    profile = keyword_embedding_profile(config_override=config_override)
    provider = _slugify(str(profile["provider"]))
    model = _slugify(str(profile["model"]))
    dimensions = int(profile["dimensions"])
    return f"chunk_keyword_embedding_index__{provider}__{model}__{dimensions}"


def keyword_embedding_dimensions(*, config_override: dict[str, Any] | None = None) -> int:
    return int(keyword_embedding_profile(config_override=config_override)["dimensions"])


def embed_keywords(keywords: list[str], *, salt: str = "", config_override: dict[str, Any] | None = None) -> list[list[float]]:
    normalized = [keyword.strip() for keyword in keywords if keyword.strip()]
    if not normalized:
        return []
    config = resolve_embedding_model_config(config_override)
    embeddings = _build_embeddings(
        model=str(config["model"]),
        provider=str(config.get("provider") or "openai"),
        base_url=str(config["base_url"]),
        api_key=str(config["api_key"]),
        dimensions=int(config["dimensions"]),
    )
    return [[float(value) for value in vector] for vector in embeddings.embed_documents([f"{salt}{keyword}" for keyword in normalized])]


@lru_cache(maxsize=16)
def _build_embeddings(
    *,
    model: str,
    provider: str,
    base_url: str,
    api_key: str,
    dimensions: int,
):
    normalized_provider = provider.strip().lower()
    kwargs: dict[str, Any] = {}
    if base_url.strip():
        kwargs["base_url"] = base_url
    if normalized_provider in {"openai", "azure_openai"}:
        if api_key.strip():
            kwargs["api_key"] = api_key
        kwargs["dimensions"] = dimensions
    return init_embeddings(
        model,
        provider=provider,
        **kwargs,
    )


def _slugify(value: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    return "_".join(part for part in value.split("_") if part) or "default"
