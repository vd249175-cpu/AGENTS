"""Mark graph query items as useful in runtime state."""

import json
from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from server.component_config import config_from_external
from server.graph_query_state import GraphQueryToolStateTydict, body_chars, mark_useful
from server.neo4j import GraphStore

from ._output import strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("graph_mark_useful.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class GraphMarkUsefulToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    max_items: int = Field(default=32, ge=1, description="状态桶允许保存的最大条目数。")
    max_total_chars: int = Field(default=4096, ge=1, description="状态桶允许保存的最大正文字符数。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "GraphMarkUsefulToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            max_items=max(1, int(payload.get("max_items", 32))),
            max_total_chars=max(1, int(payload.get("max_total_chars", 4096))),
        )

    @classmethod
    def load_config_graph_mark_useful_tool(cls, source=None) -> "GraphMarkUsefulToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class GraphMarkUsefulInput(BaseModel):
    node_ids: list[str] = Field(description="要标记的节点 id 列表。")
    rationale: str = Field(description="标记理由。")

    @model_validator(mode="after")
    def validate_input(self) -> "GraphMarkUsefulInput":
        self.node_ids = [node_id.strip() for node_id in self.node_ids if node_id.strip()]
        self.rationale = self.rationale.strip()
        return self


class GraphMarkUsefulToolStateTydict(AgentState, GraphQueryToolStateTydict, total=False):
    pass


class GraphMarkUsefulToolFeedback(BaseModel):
    successText: str = Field(default="已标记 useful 节点。")
    failureText: str = Field(default="标记 useful 失败：{error}")


class GraphMarkUsefulToolSchema:
    name = "graph_mark_useful"
    args_schema = GraphMarkUsefulInput
    description = (
        "当你确认某些节点对当前查询有帮助时使用。"
        "node_ids 需要提供要标记的节点 id 列表，rationale 说明为什么把它们记为 useful。"
    )
    toolfeedback = GraphMarkUsefulToolFeedback


def build_graph_mark_useful_tool(
    *,
    config: GraphMarkUsefulToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config = config or GraphMarkUsefulToolConfig.load()
    active_store = store or GraphStore(config_path=active_config.neo4j_config_path, run_id=active_config.run_id)

    @tool(
        GraphMarkUsefulToolSchema.name,
        args_schema=GraphMarkUsefulToolSchema.args_schema,
        description=GraphMarkUsefulToolSchema.description,
    )
    def graph_mark_useful(
        node_ids: list[str],
        rationale: str,
        runtime: ToolRuntime | None = None,
    ) -> Command[None] | dict[str, Any]:
        """Mark graph query items as useful in runtime state."""
        items = _fetch_items(active_store, run_id=active_config.run_id, node_ids=node_ids)
        state = cast(GraphMarkUsefulToolStateTydict, runtime.state if runtime is not None else {})
        update = mark_useful(state, items, rationale=rationale)
        useful = update["useful_items"]
        status = "success"
        message = f"Marked {len(items)} node(s) as useful."
        if len(useful) > active_config.max_items or body_chars(useful) > active_config.max_total_chars:
            status = "error"
            message = (
                "useful bucket capacity exceeded: "
                f"items={len(useful)}/{active_config.max_items}, "
                f"body_chars={body_chars(useful)}/{active_config.max_total_chars}."
            )
        payload = {
            "operation": "graph_mark_useful",
            "status": status,
            "message": message,
            "marked_count": len(items),
            "node_ids": [str(item.get("node_id") or "") for item in items if str(item.get("node_id") or "").strip()],
            "items": items,
        }
        if runtime is None:
            return payload
        tool_call_id = runtime.tool_call_id
        command_update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False, indent=2),
                    tool_call_id=tool_call_id,
                    status=status,
                )
            ],
        }
        if status == "success":
            command_update.update(strip_internal_run_context(update))
        return Command(
            update=command_update
        )

    return graph_mark_useful


class GraphMarkUsefulTool:
    name = GraphMarkUsefulToolSchema.name
    config = GraphMarkUsefulToolConfig
    substate = GraphMarkUsefulToolStateTydict
    toolschema = GraphMarkUsefulToolSchema

    def __init__(self, config: GraphMarkUsefulToolConfig | None = None, store: GraphStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_graph_mark_useful_tool(config=self.config, store=self.store)


tool_runingconfig = GraphMarkUsefulToolConfig.load()
tools = {}
toolStateTydicts = {
    "graph_mark_useful": GraphMarkUsefulToolStateTydict,
}
ToolConfig = {
    "inputSm": GraphMarkUsefulInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}


def _fetch_items(store: GraphStore, *, run_id: str | None, node_ids: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for node_id in node_ids:
        payload = store.fetch_node(run_id=store._resolve_run_id(run_id), node_id=node_id, detail_mode="detail")
        if payload is not None:
            items.append(payload)
    return items
