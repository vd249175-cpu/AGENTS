"""Clear blocked graph query items from runtime state."""

import json
from pathlib import Path
from typing import Any, cast

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from server.component_config import config_from_external
from server.graph_query_state import GraphQueryToolStateTydict, clear_blocked

from ._output import strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("graph_clear_blocked.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class GraphClearBlockedToolConfig(BaseModel):
    run_id: str | None = Field(default=None, description="运行隔离 id。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "GraphClearBlockedToolConfig":
        payload = _load_json(path)
        return cls(run_id=payload.get("run_id"))

    @classmethod
    def load_config_graph_clear_blocked_tool(cls, source=None) -> "GraphClearBlockedToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class GraphClearBlockedInput(BaseModel):
    node_ids: list[str] | None = Field(default=None, description="要清理的节点 id 列表。")

    @model_validator(mode="after")
    def validate_input(self) -> "GraphClearBlockedInput":
        if self.node_ids is not None:
            self.node_ids = [node_id.strip() for node_id in self.node_ids if node_id.strip()]
        return self


class GraphClearBlockedToolStateTydict(AgentState, GraphQueryToolStateTydict, total=False):
    pass


class GraphClearBlockedToolFeedback(BaseModel):
    successText: str = Field(default="已清理 blocked 节点。")
    failureText: str = Field(default="清理 blocked 失败：{error}")


class GraphClearBlockedToolSchema:
    name = "graph_clear_blocked"
    args_schema = GraphClearBlockedInput
    description = (
        "当你想清除当前查询里的 blocked 节点时使用。"
        "node_ids 可选；不传时会清掉所有 blocked 节点，传入时只清除指定节点。"
    )
    toolfeedback = GraphClearBlockedToolFeedback


def build_graph_clear_blocked_tool(*, config: GraphClearBlockedToolConfig | None = None):
    _ = config or GraphClearBlockedToolConfig.load()

    @tool(
        GraphClearBlockedToolSchema.name,
        args_schema=GraphClearBlockedToolSchema.args_schema,
        description=GraphClearBlockedToolSchema.description,
    )
    def graph_clear_blocked(node_ids: list[str] | None = None, runtime: ToolRuntime | None = None) -> Command[None] | dict[str, Any]:
        """Clear blocked graph query items from runtime state."""
        state = cast(GraphClearBlockedToolStateTydict, runtime.state if runtime is not None else {})
        update = clear_blocked(state, node_ids=node_ids)
        payload = {
            "operation": "graph_clear_blocked",
            "status": "success",
            "message": f"Cleared {len(update['cleared_ids'])} blocked node(s).",
            "cleared_count": len(update["cleared_ids"]),
            "cleared_ids": list(update["cleared_ids"]),
        }
        if runtime is None:
            return payload
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False, indent=2),
                        tool_call_id=tool_call_id,
                        status="success",
                    )
                ],
                **strip_internal_run_context(
                    {
                        "useful_items": update["useful_items"],
                        "blocked_items": update["blocked_items"],
                    }
                ),
            }
        )

    return graph_clear_blocked


class GraphClearBlockedTool:
    name = GraphClearBlockedToolSchema.name
    config = GraphClearBlockedToolConfig
    substate = GraphClearBlockedToolStateTydict
    toolschema = GraphClearBlockedToolSchema

    def __init__(self, config: GraphClearBlockedToolConfig | None = None):
        self.config = config or self.config.load()
        self.tool = self.create_tool()

    def create_tool(self):
        return build_graph_clear_blocked_tool(config=self.config)


tool_runingconfig = GraphClearBlockedToolConfig.load()
tools = {}
toolStateTydicts = {
    "graph_clear_blocked": GraphClearBlockedToolStateTydict,
}
ToolConfig = {
    "inputSm": GraphClearBlockedInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
