"""Recall graph and chunk nodes through keyword embeddings."""

from pathlib import Path
from typing import Any
from typing import Literal

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external
from server.graph_query_state import blocked_ids
from server.neo4j import GraphStore

from .recall_nodes_by_keywords import RecallNodesByKeywordsToolConfig, _assert_store_alignment, _embedding_override_from_config

from ._output import strip_internal_run_context, top_k_limit_error


TOOL_CONFIG_PATH = Path(__file__).with_name("keyword_recall.json")


class KeywordRecallToolConfig(RecallNodesByKeywordsToolConfig):
    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "KeywordRecallToolConfig":
        base = RecallNodesByKeywordsToolConfig.load(path)
        return cls(
            neo4j_config_path=base.neo4j_config_path,
            run_id=base.run_id,
            default_top_k=base.default_top_k,
            top_k_limit=base.top_k_limit,
            embedding_provider=base.embedding_provider,
            embedding_model=base.embedding_model,
            embedding_base_url=base.embedding_base_url,
            embedding_api_key=base.embedding_api_key,
            embedding_dimensions=base.embedding_dimensions,
        )

    @classmethod
    def load_config_keyword_recall_tool(cls, source=None) -> "KeywordRecallToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class KeywordRecallInput(BaseModel):
    query_keywords: list[str] = Field(description="查询关键词列表。")
    top_k: int | None = Field(default=None, description="返回条数上限；不能超过工具声明中的 top_k_limit。")
    detail_mode: Literal["summary", "detail"] = Field(
        default="summary",
        description="summary 只返回摘要和最小边信息；detail 额外返回正文和关键词。",
    )

    @model_validator(mode="after")
    def validate_input(self) -> "KeywordRecallInput":
        self.query_keywords = [keyword.strip() for keyword in self.query_keywords if keyword.strip()]
        self.detail_mode = self.detail_mode.strip()
        return self


class KeywordRecallToolStateTydict(AgentState, total=False):
    pass


class KeywordRecallToolFeedback(BaseModel):
    successText: str = Field(default="已完成关键词召回。")
    failureText: str = Field(default="关键词召回失败：{error}")


class KeywordRecallToolSchema:
    name = "keyword_recall"
    args_schema = KeywordRecallInput
    description = (
        "当你要按关键词召回图节点或 chunk 节点时使用。"
        "detail_mode=summary 时只返回 node_id、summary 和最小边信息；detail_mode=detail 时再返回正文和关键词。"
    )
    toolfeedback = KeywordRecallToolFeedback


def build_keyword_recall_tool(*, config: KeywordRecallToolConfig | None = None, store=None):
    active_config = config or KeywordRecallToolConfig.load()
    embedding_override = _embedding_override_from_config(active_config)
    _assert_store_alignment(store, embedding_override)
    active_store = store or GraphStore(
        config_path=active_config.neo4j_config_path,
        run_id=active_config.run_id,
        embedding_config_override=embedding_override,
    )
    description = (
        f"{KeywordRecallToolSchema.description}"
        f" 默认 top_k={active_config.default_top_k}；top_k_limit={active_config.top_k_limit}。"
        "如果请求超过 top_k_limit，工具会返回结构化错误，不会执行召回。"
    )

    @tool(
        KeywordRecallToolSchema.name,
        args_schema=KeywordRecallToolSchema.args_schema,
        description=description,
    )
    def keyword_recall(
        query_keywords: list[str],
        top_k: int | None = None,
        detail_mode: Literal["summary", "detail"] = "summary",
        runtime: ToolRuntime | None = None,
    ) -> dict[str, Any]:
        """Recall graph and chunk nodes through keyword embeddings."""
        resolved_top_k = active_config.default_top_k if top_k is None else int(top_k)
        if resolved_top_k > active_config.top_k_limit:
            return top_k_limit_error(
                operation="keyword_recall",
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
            "operation": "keyword_recall",
            "status": "success" if result.get("ok") else "error",
            **strip_internal_run_context(result),
        }

    return keyword_recall


class KeywordRecallTool:
    name = KeywordRecallToolSchema.name
    config = KeywordRecallToolConfig
    substate = KeywordRecallToolStateTydict
    toolschema = KeywordRecallToolSchema

    def __init__(self, config: KeywordRecallToolConfig | None = None, store=None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_keyword_recall_tool(config=self.config, store=self.store)


tool_runingconfig = KeywordRecallToolConfig.load()
tools = {}
toolStateTydicts = {
    "keyword_recall": KeywordRecallToolStateTydict,
}
ToolConfig = {
    "inputSm": KeywordRecallInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
