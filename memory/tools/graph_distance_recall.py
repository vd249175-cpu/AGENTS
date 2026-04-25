"""Recall nearby graph items by accumulated edge distance."""

import json
from pathlib import Path
from typing import Any, Literal, cast

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external
from server.graph_query_state import GraphQueryToolStateTydict, blocked_ids
from server.neo4j import GraphStore

from ._output import strip_internal_run_context, top_k_limit_error


TOOL_CONFIG_PATH = Path(__file__).with_name("graph_distance_recall.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class GraphDistanceRecallToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    default_top_k: int = Field(default=5, ge=1, description="未显式指定时使用的召回条数。")
    top_k_limit: int = Field(default=20, ge=1, description="允许的最大召回条数。")
    default_max_distance: float = Field(default=1.0, ge=0.0, description="未显式指定时使用的最大累计距离。")

    @model_validator(mode="after")
    def normalize_limits(self) -> "GraphDistanceRecallToolConfig":
        if self.default_top_k > self.top_k_limit:
            self.default_top_k = self.top_k_limit
        return self

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "GraphDistanceRecallToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            default_top_k=max(1, int(payload.get("default_top_k", 5))),
            top_k_limit=max(1, int(payload.get("top_k_limit", 20))),
            default_max_distance=float(payload.get("default_max_distance", 1.0)),
        )

    @classmethod
    def load_config_graph_distance_recall_tool(cls, source=None) -> "GraphDistanceRecallToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class GraphDistanceRecallInput(BaseModel):
    anchor_node_id: str = Field(description="召回锚点节点 id。")
    max_distance: float | None = Field(default=None, description="最大累计距离。")
    top_k: int | None = Field(default=None, description="返回条数上限；不能超过工具声明中的 top_k_limit。")
    detail_mode: Literal["summary", "detail"] = Field(
        default="summary",
        description="summary 只返回摘要和最小边信息；detail 额外返回正文和关键词。",
    )

    @model_validator(mode="after")
    def validate_input(self) -> "GraphDistanceRecallInput":
        self.anchor_node_id = self.anchor_node_id.strip()
        self.detail_mode = self.detail_mode.strip()
        return self


class GraphDistanceRecallToolStateTydict(AgentState, GraphQueryToolStateTydict, total=False):
    pass


class GraphDistanceRecallToolFeedback(BaseModel):
    successText: str = Field(default="已完成距离召回。")
    failureText: str = Field(default="距离召回失败：{error}")


class GraphDistanceRecallToolSchema:
    name = "graph_distance_recall"
    args_schema = GraphDistanceRecallInput
    description = (
        "当你已经有一个锚点节点，想按累计边距离扩展附近节点时使用。"
        "需要 anchor_node_id；max_distance、top_k 和 detail_mode 可选。"
        "detail_mode=summary 时只返回 node_id、summary、距离和最小边信息；detail_mode=detail 时再返回正文和关键词。"
    )
    toolfeedback = GraphDistanceRecallToolFeedback


def build_graph_distance_recall_tool(
    *,
    config: GraphDistanceRecallToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config = config or GraphDistanceRecallToolConfig.load()
    active_store = store or GraphStore(config_path=active_config.neo4j_config_path, run_id=active_config.run_id)
    description = (
        f"{GraphDistanceRecallToolSchema.description}"
        f" 默认 top_k={active_config.default_top_k}；top_k_limit={active_config.top_k_limit}；"
        f"默认 max_distance={active_config.default_max_distance}。"
        "如果请求超过 top_k_limit，工具会返回结构化错误，不会执行召回。"
    )

    @tool(
        GraphDistanceRecallToolSchema.name,
        args_schema=GraphDistanceRecallToolSchema.args_schema,
        description=description,
    )
    def graph_distance_recall(
        anchor_node_id: str,
        max_distance: float | None = None,
        top_k: int | None = None,
        detail_mode: Literal["summary", "detail"] = "summary",
        runtime: ToolRuntime | None = None,
    ) -> dict[str, Any]:
        """Recall nearby graph items by accumulated edge distance."""
        resolved_top_k = active_config.default_top_k if top_k is None else int(top_k)
        if resolved_top_k > active_config.top_k_limit:
            return top_k_limit_error(
                operation="graph_distance_recall",
                requested_top_k=resolved_top_k,
                top_k_limit=active_config.top_k_limit,
            )
        resolved_distance = active_config.default_max_distance if max_distance is None else float(max_distance)
        state = cast(GraphDistanceRecallToolStateTydict, runtime.state if runtime is not None else {})
        result = active_store.distance_recall(
            anchor_node_id=anchor_node_id,
            run_id=active_config.run_id,
            max_distance=resolved_distance,
            top_k=resolved_top_k,
            detail_mode=detail_mode,
            blocked_ids=blocked_ids(state),
        )
        return {
            "operation": "graph_distance_recall",
            "status": "success" if result.get("ok") else "error",
            **strip_internal_run_context(result),
        }

    return graph_distance_recall


class GraphDistanceRecallTool:
    name = GraphDistanceRecallToolSchema.name
    config = GraphDistanceRecallToolConfig
    substate = GraphDistanceRecallToolStateTydict
    toolschema = GraphDistanceRecallToolSchema

    def __init__(self, config: GraphDistanceRecallToolConfig | None = None, store: GraphStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_graph_distance_recall_tool(config=self.config, store=self.store)


tool_runingconfig = GraphDistanceRecallToolConfig.load()
tools = {}
toolStateTydicts = {
    "graph_distance_recall": GraphDistanceRecallToolStateTydict,
}
ToolConfig = {
    "inputSm": GraphDistanceRecallInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
