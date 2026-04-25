"""Read GraphNode or Chunk records by node id."""

import json
from pathlib import Path
from typing import Any, Literal

from langchain.agents.middleware import AgentState
from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external
from server.neo4j import GraphStore

from ._output import limit_items, strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("read_nodes.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_RETURN_NODES = 20


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class ReadNodesToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "ReadNodesToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
        )

    @classmethod
    def load_config_read_nodes_tool(cls, source=None) -> "ReadNodesToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class ReadNodesInput(BaseModel):
    ids: list[str] | None = Field(default=None, description="要读取的节点 id 列表。")
    detail_mode: Literal["summary", "detail"] = Field(
        default="summary",
        description="summary 只返回摘要和边；detail 额外返回正文和关键词。",
    )

    @model_validator(mode="after")
    def validate_input(self) -> "ReadNodesInput":
        if self.ids is not None:
            self.ids = [node_id.strip() for node_id in self.ids if node_id.strip()]
        return self


class ReadNodesToolStateTydict(AgentState, total=False):
    pass


class ReadNodesToolFeedback(BaseModel):
    successText: str = Field(default="已读取节点。")
    failureText: str = Field(default="读取节点失败：{error}")


class ReadNodesToolSchema:
    name = "read_nodes"
    args_schema = ReadNodesInput
    description = (
        "当你需要确认某个图节点是否存在、查看它连着哪些边，或者核对刚创建/更新的图节点时使用。"
        "ids 可以省略；省略时只会返回提醒信息，不会扫全库。"
        "detail_mode=summary 时只返回 node_id、summary 和最小边信息；detail_mode=detail 时再返回正文和关键词。"
    )
    toolfeedback = ReadNodesToolFeedback


def build_read_nodes_tool(
    *,
    config: ReadNodesToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config = config or ReadNodesToolConfig.load()
    active_store = store or GraphStore(config_path=active_config.neo4j_config_path, run_id=active_config.run_id)

    @tool(
        ReadNodesToolSchema.name,
        args_schema=ReadNodesToolSchema.args_schema,
        description=ReadNodesToolSchema.description,
    )
    def read_nodes(ids: list[str] | None = None, detail_mode: Literal["summary", "detail"] = "summary") -> dict[str, Any]:
        """Read GraphNode or Chunk records by node id."""
        result = active_store.read_nodes(ids=ids, run_id=active_config.run_id, detail_mode=detail_mode)
        limited_ids, total_ids, ids_truncated = limit_items(result.get("ids") or [], MAX_RETURN_NODES)
        limited_results, total_results, truncated = limit_items(result.get("results") or [], MAX_RETURN_NODES)
        missing_ids = list(result.get("missing_ids") or [])
        limited_missing_ids, total_missing_ids, missing_truncated = limit_items(missing_ids, MAX_RETURN_NODES)
        return {
            "operation": "read_nodes",
            "status": "success" if result.get("ok") else "error",
            **strip_internal_run_context(
                {
                    **result,
                    "ids": limited_ids,
                    "id_count": total_ids,
                    "returned_id_count": len(limited_ids),
                    "ids_truncated": ids_truncated,
                    "results": limited_results,
                    "result_count": total_results,
                    "returned_result_count": len(limited_results),
                    "results_truncated": truncated,
                    "missing_ids": limited_missing_ids,
                    "missing_id_count": total_missing_ids,
                    "returned_missing_id_count": len(limited_missing_ids),
                    "missing_ids_truncated": missing_truncated,
                }
            ),
        }

    return read_nodes


class ReadNodesTool:
    name = ReadNodesToolSchema.name
    config = ReadNodesToolConfig
    substate = ReadNodesToolStateTydict
    toolschema = ReadNodesToolSchema

    def __init__(self, config: ReadNodesToolConfig | None = None, store: GraphStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_read_nodes_tool(config=self.config, store=self.store)


tool_runingconfig = ReadNodesToolConfig.load()
tools = {}
toolStateTydicts = {
    "read_nodes": ReadNodesToolStateTydict,
}
ToolConfig = {
    "inputSm": ReadNodesInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
