"""Neo4j config helpers for the migrated project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class Neo4jConfig(TypedDict, total=False):
    uri: str
    username: str
    password: str
    database: str


class DatabaseConfig(TypedDict, total=False):
    neo4j: Neo4jConfig
    sqlite: dict[str, Any]


DEFAULT_DATABASE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "workspace" / "config" / "database_config.json"


class Neo4jConnectionConfig(BaseModel):
    uri: str = Field(description="Neo4j 连接 URI，例如 neo4j://localhost:7687。")
    username: str = Field(description="Neo4j 用户名。")
    password: str = Field(description="Neo4j 密码。")
    database: str | None = Field(default=None, description="可选的 Neo4j database 名称。")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Neo4jConnectionConfig":
        payload = get_neo4j_config(path=path)
        return cls(
            uri=str(payload.get("uri") or ""),
            username=str(payload.get("username") or ""),
            password=str(payload.get("password") or ""),
            database=str(payload.get("database")) if payload.get("database") else None,
        )


def resolve_database_config_path(path: str | Path | None = None) -> Path:
    return Path(path).expanduser().resolve() if path is not None else DEFAULT_DATABASE_CONFIG_PATH


def load_database_config(path: str | Path | None = None) -> DatabaseConfig:
    config_path = resolve_database_config_path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def get_neo4j_config(*, path: str | Path | None = None) -> Neo4jConfig:
    payload = load_database_config(path)
    value = payload.get("neo4j")
    return value if isinstance(value, dict) else {}


def resolve_neo4j_connection(
    *,
    connection: Neo4jConnectionConfig | dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Neo4jConnectionConfig:
    if isinstance(connection, Neo4jConnectionConfig):
        return connection
    if isinstance(connection, dict) and connection:
        return Neo4jConnectionConfig.model_validate(connection)
    return Neo4jConnectionConfig.load(path)
