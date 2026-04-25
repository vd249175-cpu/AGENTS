"""Recall graph and chunk nodes through global keyword embeddings."""

import json
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, model_validator
from server.graph_query_state import blocked_ids
from server.embedding_keywords import keyword_embedding_dimensions, keyword_embedding_index_name
from server.neo4j import GraphStore

from ._output import strip_internal_run_context, top_k_limit_error


TOOL_CONFIG_PATH = Path(__file__).with_name("recall_nodes_by_keywords.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class RecallNodesByKeywordsToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    default_top_k: int = Field(default=5, ge=1, description="未显式指定时使用的召回条数。")
    top_k_limit: int = Field(default=20, ge=1, description="允许的最大召回条数。")
    embedding_provider: str | None = Field(default=None, description="可选的 embedding provider override。")
    embedding_model: str | None = Field(default=None, description="可选的 embedding model override。")
    embedding_base_url: str | None = Field(default=None, description="可选的 embedding base_url override。")
    embedding_api_key: str | None = Field(default=None, description="可选的 embedding api_key override。")
    embedding_dimensions: int | None = Field(default=None, description="可选的 embedding dimensions override。")

    @model_validator(mode="after")
    def normalize_limits(self) -> "RecallNodesByKeywordsToolConfig":
        if self.default_top_k > self.top_k_limit:
            self.default_top_k = self.top_k_limit
        return self

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "RecallNodesByKeywordsToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            default_top_k=max(1, int(payload.get("default_top_k", 5))),
            top_k_limit=max(1, int(payload.get("top_k_limit", 20))),
            embedding_provider=payload.get("embedding_provider"),
            embedding_model=payload.get("embedding_model"),
            embedding_base_url=payload.get("embedding_base_url"),
            embedding_api_key=payload.get("embedding_api_key"),
            embedding_dimensions=payload.get("embedding_dimensions"),
        )


class RecallNodesByKeywordsInput(BaseModel):
    query_keywords: list[str] = Field(description="查询关键词列表。")
    top_k: int | None = Field(default=None, description="返回条数上限；不能超过工具声明中的 top_k_limit。")
    detail_mode: str = Field(default="summary", description="summary 或 detail。")

    @model_validator(mode="after")
    def validate_input(self) -> "RecallNodesByKeywordsInput":
        self.query_keywords = [keyword.strip() for keyword in self.query_keywords if keyword.strip()]
        self.detail_mode = self.detail_mode.strip()
        return self


class RecallNodesByKeywordsToolStateTydict(AgentState, total=False):
    pass


def build_recall_nodes_by_keywords_tool(
    *,
    config: RecallNodesByKeywordsToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config = config or RecallNodesByKeywordsToolConfig.load()
    embedding_override = _embedding_override_from_config(active_config)
    _assert_store_alignment(store, embedding_override)
    active_store = store or GraphStore(
        config_path=active_config.neo4j_config_path,
        run_id=active_config.run_id,
        embedding_config_override=embedding_override,
    )
    description = (
        "Recall graph and chunk nodes through global keyword embeddings. "
        f"default_top_k={active_config.default_top_k}; top_k_limit={active_config.top_k_limit}. "
        "If a call requests more than top_k_limit, the tool returns a structured error instead of executing recall."
    )

    @tool("recall_nodes_by_keywords", args_schema=RecallNodesByKeywordsInput, description=description)
    def recall_nodes_by_keywords(
        query_keywords: list[str],
        top_k: int | None = None,
        detail_mode: str = "summary",
        runtime: ToolRuntime | None = None,
    ) -> dict[str, Any]:
        """Recall graph and chunk nodes through global keyword embeddings."""
        resolved_top_k = active_config.default_top_k if top_k is None else int(top_k)
        if resolved_top_k > active_config.top_k_limit:
            return top_k_limit_error(
                operation="recall_nodes_by_keywords",
                requested_top_k=resolved_top_k,
                top_k_limit=active_config.top_k_limit,
            )
        result = active_store.recall_nodes_by_keywords(
            query_keywords=query_keywords,
            run_id=active_config.run_id,
            top_k=resolved_top_k,
            detail_mode=detail_mode,
            blocked_ids=blocked_ids(runtime.state if runtime is not None else None),
        )
        return {
            "operation": "recall_nodes_by_keywords",
            "status": "success" if result.get("ok") else "error",
            **strip_internal_run_context(result),
        }

    return recall_nodes_by_keywords


def _embedding_override_from_config(config: RecallNodesByKeywordsToolConfig) -> dict[str, Any] | None:
    override = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "base_url": config.embedding_base_url,
        "api_key": config.embedding_api_key,
        "dimensions": config.embedding_dimensions,
    }
    cleaned = {
        key: value
        for key, value in override.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }
    return cleaned or None


def _assert_store_alignment(store: GraphStore | None, embedding_override: dict[str, Any] | None) -> None:
    if store is None or embedding_override is None:
        return
    expected_index_name = keyword_embedding_index_name(config_override=embedding_override)
    expected_dimensions = keyword_embedding_dimensions(config_override=embedding_override)
    store_index_name = getattr(store, "keyword_index_name", None)
    if store_index_name is not None and str(store_index_name) != expected_index_name:
        raise ValueError(
            "recall_nodes_by_keywords embedding config does not match graph_store keyword index: "
            f"store={store_index_name!r}, query={expected_index_name!r}"
        )
    store_dimensions = getattr(store, "keyword_dimensions", None)
    if store_dimensions is not None and int(store_dimensions) != expected_dimensions:
        raise ValueError(
            "recall_nodes_by_keywords embedding dimensions do not match graph_store keyword dimensions: "
            f"store={store_dimensions!r}, query={expected_dimensions!r}"
        )


tool_runingconfig = RecallNodesByKeywordsToolConfig.load()
tools = {}
toolStateTydicts = {
    "recall_nodes_by_keywords": RecallNodesByKeywordsToolStateTydict,
}
ToolConfig = {
    "inputSm": RecallNodesByKeywordsInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
