"""Manage GraphNode records and ordinary graph edges."""

import json
from pathlib import Path
from typing import Any, Literal

from langchain.agents.middleware import AgentState
from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external
from server.neo4j import GraphStore

from ._output import limit_items, strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("graph_manage_nodes.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_RETURN_ACTIONS = 20
MAX_RETURN_IDS = 20
MAX_RETURN_ERRORS = 20


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class GraphManageNodesToolConfig(BaseModel):
    neo4j_config_path: Path = Field(description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    default_edge_distance: float = Field(default=0.3, ge=0.0, description="普通图边新增时使用的默认距离值。")
    persist_keyword_embeddings: bool = Field(default=True, description="是否持久化关键词向量。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "GraphManageNodesToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            default_edge_distance=float(payload.get("default_edge_distance", 0.3)),
            persist_keyword_embeddings=bool(payload.get("persist_keyword_embeddings", True)),
        )

    @classmethod
    def load_config_graph_manage_nodes_tool(cls, source=None) -> "GraphManageNodesToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class GraphKeywordOp(BaseModel):
    op: Literal["+", "-", "replace"] = Field(description="关键词操作，只支持 +、- 或 replace。")
    keywords: list[str] = Field(default_factory=list, description="要新增、删除或替换的关键词列表。")


class GraphEdgeOp(BaseModel):
    op: Literal["+", "-"] = Field(description="边操作，只支持 + 或 -。")
    targets: str | list[int | str | list[int] | list[str]] = Field(description="边目标，可以是 id、chunk_index 或索引范围。")
    document_name: str | None = Field(default=None, description="可选的文档上下文；当 targets 使用 chunk_index 时需要。")
    dist: float | None = Field(default=None, description="边的距离值。新增边时默认使用图写配置的距离。")


class GraphNodeAction(BaseModel):
    operation: Literal["create", "update", "delete"] = Field(
        description="动作类型。create 用于新增图节点，update 用于修改已有节点，delete 用于删除已有节点。"
    )
    ids: list[str] | None = Field(
        default=None,
        description="目标节点 id 列表。create 时可选，update/delete 时需要。Chunk 节点如果是文档背书节点，不要用这个工具改内容或删除。",
    )
    summary: str | None = Field(
        default=None,
        description="节点摘要。create 时必填；update 时可选；delete 时不要传。",
    )
    body: str | None = Field(
        default=None,
        description="节点正文。create 时必填；update 时可选；delete 时不要传。",
    )
    keyword_ops: list[GraphKeywordOp] = Field(
        default_factory=list,
        description="关键词增删改动作列表。必须是数组；create 时通常需要；update 时可选；delete 时不要传。",
    )
    edge_ops: list[GraphEdgeOp] = Field(
        default_factory=list,
        description="普通图边增删改动作列表。必须是数组；create/update 时可用；delete 时不要传。",
    )

    @model_validator(mode="after")
    def validate_action(self) -> "GraphNodeAction":
        return self


class GraphManageNodesInput(BaseModel):
    actions: list[GraphNodeAction] = Field(
        description=(
            "按顺序执行的图节点动作列表。可以混合 create、update 和 delete；"
            "create 负责新建图节点，update 负责改摘要、正文、关键词或普通边，delete 负责删除非 Chunk 的图节点。"
            "create 时通常需要 keyword_ops，且 keyword_ops 必须写成数组；update 时需要 ids，并至少提供一个可修改字段；delete 时只保留 ids。"
        )
    )

    @model_validator(mode="after")
    def validate_input(self) -> "GraphManageNodesInput":
        if not self.actions:
            raise ValueError("actions must not be empty")
        return self


class GraphCreateNodeAction(BaseModel):
    ids: list[str] | None = Field(
        default=None,
        description=(
            "可选的新 GraphNode id 列表。不传时系统自动生成真实 node id；"
            "工具返回结果会包含实际写入的 ids。summary/body 中的标记词不是 node id。"
        ),
    )
    summary: str = Field(description="新 GraphNode 摘要。")
    body: str = Field(description="新 GraphNode 正文。")
    keyword_ops: list[GraphKeywordOp] = Field(description="关键词操作数组；create 必须显式提供。")
    edge_ops: list[GraphEdgeOp] = Field(
        default_factory=list,
        description="可选普通图边操作；每个对象描述一组目标。targets 必须是真实 node id 或带 document_name 的 chunk_index。",
    )

    @model_validator(mode="after")
    def validate_action(self) -> "GraphCreateNodeAction":
        if not self.keyword_ops:
            raise ValueError("keyword_ops are required for create")
        return self


class GraphCreateNodesInput(BaseModel):
    items: list[GraphCreateNodeAction] = Field(description="要按顺序创建的 GraphNode 列表。")

    @model_validator(mode="after")
    def validate_input(self) -> "GraphCreateNodesInput":
        if not self.items:
            raise ValueError("items must not be empty")
        return self


class GraphUpdateNodeInput(BaseModel):
    id: str = Field(
        description=(
            "要更新的单个真实 node id。不要使用 summary/body 中的唯一标记词。"
            "更新 Chunk 时只允许普通 graph 边，不允许改正文、摘要或关键词。"
        )
    )
    summary: str | None = Field(default=None, description="可选的新摘要；仅 GraphNode 可用。")
    body: str | None = Field(default=None, description="可选的新正文；仅 GraphNode 可用。")
    keyword_ops: list[GraphKeywordOp] = Field(default_factory=list, description="可选关键词操作；仅 GraphNode 可用。")
    edge_ops: list[GraphEdgeOp] = Field(
        default_factory=list,
        description=(
            "普通图边操作数组。一次连接多个目标时，推荐写成多个对象："
            "[{\"op\":\"+\",\"targets\":[\"node_a\"],\"dist\":0.3},{\"op\":\"+\",\"targets\":[\"node_b\"],\"dist\":0.3}]。"
            "targets 必须是真实 node id 或带 document_name 的 chunk_index，不要写文档名、摘要或标记词。"
        ),
    )

    @model_validator(mode="after")
    def validate_input(self) -> "GraphUpdateNodeInput":
        self.id = self.id.strip()
        if not self.id:
            raise ValueError("id is required")
        if self.summary is None and self.body is None and not self.keyword_ops and not self.edge_ops:
            raise ValueError("one of summary, body, keyword_ops, or edge_ops is required")
        return self


class GraphDeleteNodesInput(BaseModel):
    ids: list[str] = Field(description="要物理删除的 GraphNode id 列表。不能删除 Chunk。")

    @model_validator(mode="after")
    def validate_input(self) -> "GraphDeleteNodesInput":
        self.ids = [node_id.strip() for node_id in self.ids if node_id.strip()]
        if not self.ids:
            raise ValueError("ids must not be empty")
        return self


class GraphManageNodesToolStateTydict(AgentState, total=False):
    pass


class GraphManageNodesToolFeedback(BaseModel):
    successText: str = Field(default="已管理图节点。")
    failureText: str = Field(default="管理图节点失败：{error}")


class GraphManageNodesToolSchema:
    name = "graph_manage_nodes"
    args_schema = GraphManageNodesInput
    description = (
        "批量创建、更新或删除普通 GraphNode，并可增删普通图边。"
        "create 新建 GraphNode，update 修改 GraphNode 内容或普通边，delete 物理删除 GraphNode。"
        "keyword_ops 和 edge_ops 都必须是数组；edge_ops 只处理普通图边。"
    )
    toolfeedback = GraphManageNodesToolFeedback


class GraphCreateNodesToolSchema:
    name = "graph_create_nodes"
    args_schema = GraphCreateNodesInput
    description = (
        "创建一个或多个新的 GraphNode。"
        "每个 item 需要提供 summary、body 和 keyword_ops 数组；不会修改或删除已有节点。"
        "返回 results[*].ids，里面是实际写入的真实 node id。"
    )
    toolfeedback = GraphManageNodesToolFeedback


class GraphUpdateNodeToolSchema:
    name = "graph_update_node"
    args_schema = GraphUpdateNodeInput
    description = (
        "更新一个已有节点，或为一个已有节点增删普通图边。"
        "id 是单个真实 node id，不是摘要、正文、文档名或临时标记词。"
        "GraphNode 可更新摘要、正文和关键词，Chunk 只允许普通图边操作。"
        "edge_ops 用于新增或删除普通图边；连接多条边时，把每条边写成一个 edge_ops 对象。"
    )
    toolfeedback = GraphManageNodesToolFeedback


class GraphDeleteNodesToolSchema:
    name = "graph_delete_nodes"
    args_schema = GraphDeleteNodesInput
    description = "只用于物理删除 GraphNode。不能删除 Chunk，也不会做语义判断或修复。"
    toolfeedback = GraphManageNodesToolFeedback


def _active_graph_store(
    *,
    config: GraphManageNodesToolConfig | None = None,
    store: GraphStore | None = None,
) -> tuple[GraphManageNodesToolConfig, GraphStore]:
    active_config = config or GraphManageNodesToolConfig.load()
    active_store = store or GraphStore(
        config_path=active_config.neo4j_config_path,
        run_id=active_config.run_id,
        default_edge_distance=active_config.default_edge_distance,
        persist_keyword_embeddings=active_config.persist_keyword_embeddings,
    )
    return active_config, active_store


def _run_graph_manage_nodes(
    *,
    operation: str,
    active_config: GraphManageNodesToolConfig,
    active_store: GraphStore,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    result = active_store.manage_nodes(actions=actions, run_id=active_config.run_id)
    action_results = _summarize_action_results(result.get("results"))
    limited_action_results, total_action_results, actions_truncated = limit_items(action_results, MAX_RETURN_ACTIONS)
    success_count = sum(1 for item in action_results if item.get("status") == "success")
    failure_count = sum(1 for item in action_results if item.get("status") != "success")
    return {
        "operation": operation,
        "status": "success" if result.get("ok") else "error",
        "message": str(result.get("message") or ""),
        "action_count": total_action_results,
        "returned_action_count": len(limited_action_results),
        "actions_truncated": actions_truncated,
        "success_count": success_count,
        "failure_count": failure_count,
        "results": limited_action_results,
    }


def build_graph_manage_nodes_tool(
    *,
    config: GraphManageNodesToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config, active_store = _active_graph_store(config=config, store=store)

    @tool(
        GraphManageNodesToolSchema.name,
        args_schema=GraphManageNodesToolSchema.args_schema,
        description=GraphManageNodesToolSchema.description,
    )
    def graph_manage_nodes(actions: list[GraphNodeAction]) -> dict[str, Any]:
        """Manage GraphNode records and ordinary graph edges."""
        return _run_graph_manage_nodes(
            operation="graph_manage_nodes",
            active_config=active_config,
            active_store=active_store,
            actions=[_action_to_dict(action) for action in actions],
        )

    return graph_manage_nodes


def build_graph_create_nodes_tool(
    *,
    config: GraphManageNodesToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config, active_store = _active_graph_store(config=config, store=store)

    @tool(
        GraphCreateNodesToolSchema.name,
        args_schema=GraphCreateNodesToolSchema.args_schema,
        description=GraphCreateNodesToolSchema.description,
    )
    def graph_create_nodes(items: list[GraphCreateNodeAction]) -> dict[str, Any]:
        """Create GraphNode records."""
        actions = []
        for item in items:
            payload = _action_to_dict(item.model_dump())
            payload["operation"] = "create"
            actions.append(payload)
        return _run_graph_manage_nodes(
            operation="graph_create_nodes",
            active_config=active_config,
            active_store=active_store,
            actions=actions,
        )

    return graph_create_nodes


def build_graph_update_node_tool(
    *,
    config: GraphManageNodesToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config, active_store = _active_graph_store(config=config, store=store)

    @tool(
        GraphUpdateNodeToolSchema.name,
        args_schema=GraphUpdateNodeToolSchema.args_schema,
        description=GraphUpdateNodeToolSchema.description,
    )
    def graph_update_node(
        id: str,
        summary: str | None = None,
        body: str | None = None,
        keyword_ops: list[GraphKeywordOp] | None = None,
        edge_ops: list[GraphEdgeOp] | None = None,
    ) -> dict[str, Any]:
        """Update one GraphNode or its ordinary graph edges."""
        payload: dict[str, Any] = {
            "operation": "update",
            "ids": [id],
            "summary": summary,
            "body": body,
            "keyword_ops": keyword_ops or [],
            "edge_ops": edge_ops or [],
        }
        return _run_graph_manage_nodes(
            operation="graph_update_node",
            active_config=active_config,
            active_store=active_store,
            actions=[_action_to_dict(payload)],
        )

    return graph_update_node


def build_graph_delete_nodes_tool(
    *,
    config: GraphManageNodesToolConfig | None = None,
    store: GraphStore | None = None,
):
    active_config, active_store = _active_graph_store(config=config, store=store)

    @tool(
        GraphDeleteNodesToolSchema.name,
        args_schema=GraphDeleteNodesToolSchema.args_schema,
        description=GraphDeleteNodesToolSchema.description,
    )
    def graph_delete_nodes(ids: list[str]) -> dict[str, Any]:
        """Physically delete GraphNode records."""
        return _run_graph_manage_nodes(
            operation="graph_delete_nodes",
            active_config=active_config,
            active_store=active_store,
            actions=[{"operation": "delete", "ids": ids}],
        )

    return graph_delete_nodes


class GraphManageNodesTool:
    name = GraphManageNodesToolSchema.name
    config = GraphManageNodesToolConfig
    substate = GraphManageNodesToolStateTydict
    toolschema = GraphManageNodesToolSchema

    def __init__(self, config: GraphManageNodesToolConfig | None = None, store: GraphStore | None = None):
        self.config = config or self.config.load()
        self.store = store
        self.tool = self.create_tool()

    def create_tool(self):
        return build_graph_manage_nodes_tool(config=self.config, store=self.store)


class GraphCreateNodesTool(GraphManageNodesTool):
    name = GraphCreateNodesToolSchema.name
    toolschema = GraphCreateNodesToolSchema

    def create_tool(self):
        return build_graph_create_nodes_tool(config=self.config, store=self.store)


class GraphUpdateNodeTool(GraphManageNodesTool):
    name = GraphUpdateNodeToolSchema.name
    toolschema = GraphUpdateNodeToolSchema

    def create_tool(self):
        return build_graph_update_node_tool(config=self.config, store=self.store)


class GraphDeleteNodesTool(GraphManageNodesTool):
    name = GraphDeleteNodesToolSchema.name
    toolschema = GraphDeleteNodesToolSchema

    def create_tool(self):
        return build_graph_delete_nodes_tool(config=self.config, store=self.store)


tool_runingconfig = GraphManageNodesToolConfig.load()
tools = {}
toolStateTydicts = {
    "graph_manage_nodes": GraphManageNodesToolStateTydict,
}
ToolConfig = {
    "inputSm": GraphManageNodesInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}


def _action_to_dict(action: GraphNodeAction) -> dict[str, Any]:
    raw_action = action if isinstance(action, dict) else action.model_dump()
    payload = dict(raw_action)
    keyword_ops = payload.get("keyword_ops")
    if keyword_ops:
        payload["keyword_ops"] = [
            dict(keyword_op if isinstance(keyword_op, dict) else keyword_op.model_dump())
            for keyword_op in keyword_ops
        ]
    else:
        payload["keyword_ops"] = None
    edge_ops = payload.get("edge_ops")
    if edge_ops:
        payload["edge_ops"] = [
            dict(edge_op if isinstance(edge_op, dict) else edge_op.model_dump())
            for edge_op in edge_ops
        ]
    else:
        payload["edge_ops"] = None
    return payload


def _summarize_action_results(results: object) -> list[dict[str, Any]]:
    if not isinstance(results, list):
        return []
    summarized: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        payload: dict[str, Any] = {
            "operation": str(item.get("operation") or "unknown"),
            "status": str(item.get("status") or "error"),
            "message": str(item.get("message") or ""),
        }
        ids = item.get("ids")
        if isinstance(ids, list):
            cleaned_ids = [str(node_id).strip() for node_id in ids if str(node_id).strip()]
            limited_ids, total_ids, ids_truncated = limit_items(cleaned_ids, MAX_RETURN_IDS)
            if limited_ids:
                payload["ids"] = limited_ids
                payload["id_count"] = total_ids
                payload["returned_id_count"] = len(limited_ids)
                payload["ids_truncated"] = ids_truncated
        chunk_context = _find_chunk_context(item)
        if chunk_context is not None:
            payload.update(chunk_context)
        errors = item.get("errors")
        if isinstance(errors, list):
            cleaned_errors = [str(error).strip() for error in errors if str(error).strip()]
            limited_errors, total_errors, errors_truncated = limit_items(cleaned_errors, MAX_RETURN_ERRORS)
            if limited_errors:
                payload["errors"] = limited_errors
                payload["error_count"] = total_errors
                payload["returned_error_count"] = len(limited_errors)
                payload["errors_truncated"] = errors_truncated
        summarized.append(strip_internal_run_context(payload))
    return summarized


def _find_chunk_context(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        document_name = value.get("document_name")
        chunk_index = value.get("chunk_index")
        if document_name is not None and chunk_index is not None:
            return {
                "document_name": str(document_name),
                "chunk_index": int(chunk_index),
            }
        for key in ("results", "before"):
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    context = _find_chunk_context(item)
                    if context is not None:
                        return context
            elif isinstance(nested, dict):
                context = _find_chunk_context(nested)
                if context is not None:
                    return context
    return None
