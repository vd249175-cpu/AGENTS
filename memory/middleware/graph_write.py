"""Graph write middleware declaration."""

import json
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from pydantic import BaseModel, Field
from server.component_config import config_from_external
from server.neo4j import GraphStore, Neo4jConnectionConfig, resolve_neo4j_connection
from tools.graph_manage_nodes import (
    GraphManageNodesToolConfig,
    GraphManageNodesToolStateTydict,
    build_graph_create_nodes_tool,
    build_graph_delete_nodes_tool,
    build_graph_update_node_tool,
)
from tools.read_nodes import (
    ReadNodesToolConfig,
    ReadNodesToolStateTydict,
    ToolConfig as ReadNodesToolBundle,
    build_read_nodes_tool,
)


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("graph_write.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class GraphWriteMiddlewareConfig(BaseModel):
    neo4j: Neo4jConnectionConfig | None = Field(default=None, description="显式的 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    trace_limit: int = Field(default=16, ge=1, description="运行时 trace 的最大保留条数。")
    embedding_provider: str | None = Field(default=None, description="可选的 embedding provider override。")
    embedding_model: str | None = Field(default=None, description="可选的 embedding model override。")
    embedding_base_url: str | None = Field(default=None, description="可选的 embedding base_url override。")
    embedding_api_key: str | None = Field(default=None, description="可选的 embedding api_key override。")
    embedding_dimensions: int | None = Field(default=None, description="可选的 embedding dimensions override。")

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "GraphWriteMiddlewareConfig":
        payload = _load_json(path)
        return cls(
            neo4j=Neo4jConnectionConfig.model_validate(payload["neo4j"]) if isinstance(payload.get("neo4j"), dict) else None,
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            trace_limit=max(1, int(payload.get("trace_limit", 16))),
            embedding_provider=payload.get("embedding_provider"),
            embedding_model=payload.get("embedding_model"),
            embedding_base_url=payload.get("embedding_base_url"),
            embedding_api_key=payload.get("embedding_api_key"),
            embedding_dimensions=payload.get("embedding_dimensions"),
        )

    @classmethod
    def load_config_graph_write_middleware(cls, source=None) -> "GraphWriteMiddlewareConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


class GraphWriteStateTydict(AgentState, total=False):
    pass


middleware_runingconfig = GraphWriteMiddlewareConfig.load()
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="graph_write.guidance",
        prompt=(
            "当前任务需要写图时，只处理 GraphNode 和普通图边。"
            "创建、更新、删除、读取是不同动作；每次选择与当前意图匹配的图入口。"
            "Chunk 可参与普通图边，但不能在图写能力里修改正文、摘要、关键词、删除 Chunk 或改写 DOCUMENT_NEXT。"
            "graph_create_nodes 返回 results[*].ids；如果创建时没有显式传 ids，后续更新、读取、连边必须从返回结果里取真实 id。"
            "如果创建时显式传了 ids，也优先以返回结果确认为准。不要把 summary/body 中的唯一标记词当作 node id。"
            "一次新增多条普通图边时，在 graph_update_node.edge_ops 中为每条边写一个 {op, targets, dist} 对象。"
        ),
    )
]
MiddlewareToolConfig = {
    "tools": {
        **ReadNodesToolBundle["tools"],
    },
    "toolStateTydicts": {
        "graph_create_nodes": GraphManageNodesToolStateTydict,
        "graph_update_node": GraphManageNodesToolStateTydict,
        "graph_delete_nodes": GraphManageNodesToolStateTydict,
        **ReadNodesToolBundle["toolStateTydicts"],
    },
}


def _build_tool_config(config: GraphWriteMiddlewareConfig) -> dict[str, object]:
    neo4j = resolve_neo4j_connection(connection=config.neo4j, path=config.neo4j_config_path)
    resolved_config_path = config.neo4j_config_path or (PROJECT_ROOT / "workspace/config/database_config.json").resolve()
    store = GraphStore(
        uri=neo4j.uri,
        username=neo4j.username,
        password=neo4j.password,
        database=neo4j.database,
        run_id=config.run_id,
        embedding_config_override=_embedding_override_from_config(config),
    )
    manage_config = GraphManageNodesToolConfig.load().model_copy(
        update={"neo4j_config_path": resolved_config_path, "run_id": config.run_id}
    )
    read_config = ReadNodesToolConfig.load().model_copy(
        update={"neo4j_config_path": resolved_config_path, "run_id": config.run_id}
    )
    create_tool = build_graph_create_nodes_tool(config=manage_config, store=store)
    update_tool = build_graph_update_node_tool(config=manage_config, store=store)
    delete_tool = build_graph_delete_nodes_tool(config=manage_config, store=store)
    read_tool = build_read_nodes_tool(config=read_config, store=store)
    return {
        "store": store,
        "tools": {
            create_tool.name: create_tool,
            update_tool.name: update_tool,
            delete_tool.name: delete_tool,
            read_tool.name: read_tool,
        },
        "toolStateTydicts": {
            "graph_create_nodes": GraphManageNodesToolStateTydict,
            "graph_update_node": GraphManageNodesToolStateTydict,
            "graph_delete_nodes": GraphManageNodesToolStateTydict,
            "read_nodes": ReadNodesToolStateTydict,
        },
    }


class GraphWriteCapabilityMiddleware(AgentMiddleware):
    name = "graph_write"
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = middleware_runingconfig
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = GraphWriteStateTydict  # type: ignore[assignment]

    def __init__(self, *, config: GraphWriteMiddlewareConfig | None = None) -> None:
        super().__init__()
        self.config = config or self.runingConfig
        self.capability_prompts = self.capabilityPromptConfigs
        runtime_tool_config = _build_tool_config(self.config)
        self._store = runtime_tool_config["store"]
        self.toolConfig = runtime_tool_config
        self.toolStateTydicts = runtime_tool_config["toolStateTydicts"]
        self.tools = list(runtime_tool_config["tools"].values())
        self.middleware = self

    def close(self) -> None:
        self._store.close()


def _embedding_override_from_config(config: GraphWriteMiddlewareConfig) -> dict[str, object] | None:
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
