"""Graph query middleware declaration."""

import json
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from pydantic import BaseModel, Field, model_validator
from server.component_config import config_from_external
from server.graph_query_state import GraphQueryToolStateTydict
from server.neo4j import GraphStore, Neo4jConnectionConfig, resolve_neo4j_connection
from tools.graph_clear_blocked import (
    GraphClearBlockedToolConfig,
    GraphClearBlockedToolStateTydict,
    ToolConfig as GraphClearBlockedToolBundle,
    build_graph_clear_blocked_tool,
)
from tools.graph_distance_recall import (
    GraphDistanceRecallToolConfig,
    GraphDistanceRecallToolStateTydict,
    ToolConfig as GraphDistanceRecallToolBundle,
    build_graph_distance_recall_tool,
)
from tools.graph_mark_blocked import (
    GraphMarkBlockedToolConfig,
    GraphMarkBlockedToolStateTydict,
    ToolConfig as GraphMarkBlockedToolBundle,
    build_graph_mark_blocked_tool,
)
from tools.graph_mark_useful import (
    GraphMarkUsefulToolConfig,
    GraphMarkUsefulToolStateTydict,
    ToolConfig as GraphMarkUsefulToolBundle,
    build_graph_mark_useful_tool,
)
from tools.keyword_recall import (
    KeywordRecallToolConfig,
    KeywordRecallToolStateTydict,
    ToolConfig as KeywordRecallToolBundle,
    build_keyword_recall_tool,
)


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("graph_query.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class GraphQueryMiddlewareConfig(BaseModel):
    neo4j: Neo4jConnectionConfig | None = Field(default=None, description="显式的 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    trace_limit: int = Field(default=16, ge=1, description="运行时 trace 的最大保留条数。")
    capability_preset: "GraphQueryCapabilityPreset" = Field(default_factory=lambda: GraphQueryCapabilityPreset())

    @model_validator(mode="after")
    def _sync_legacy_run_id(self) -> "GraphQueryMiddlewareConfig":
        if self.capability_preset.run_id is None and self.run_id is not None:
            self.capability_preset.run_id = self.run_id
        elif self.capability_preset.run_id is not None and self.run_id is None:
            self.run_id = self.capability_preset.run_id
        return self

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "GraphQueryMiddlewareConfig":
        payload = _load_json(path)
        preset_payload = payload.get("capability_preset") if isinstance(payload.get("capability_preset"), dict) else {}
        return cls(
            neo4j=Neo4jConnectionConfig.model_validate(payload["neo4j"]) if isinstance(payload.get("neo4j"), dict) else None,
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            trace_limit=max(1, int(payload.get("trace_limit", 16))),
            capability_preset=GraphQueryCapabilityPreset(
                run_id=preset_payload.get("run_id", payload.get("run_id")),
                keyword_top_k=max(1, int(preset_payload.get("keyword_top_k", 5))),
                keyword_top_k_limit=max(1, int(preset_payload.get("keyword_top_k_limit", 20))),
                distance_top_k=max(1, int(preset_payload.get("distance_top_k", 5))),
                distance_top_k_limit=max(1, int(preset_payload.get("distance_top_k_limit", 20))),
                distance_max_distance=float(preset_payload.get("distance_max_distance", 1.0)),
                useful_max_items=max(1, int(preset_payload.get("useful_max_items", 32))),
                useful_max_total_chars=max(1, int(preset_payload.get("useful_max_total_chars", 4096))),
                blocked_max_items=max(1, int(preset_payload.get("blocked_max_items", 32))),
                blocked_max_total_chars=max(1, int(preset_payload.get("blocked_max_total_chars", 4096))),
                embedding_provider=preset_payload.get("embedding_provider"),
                embedding_model=preset_payload.get("embedding_model"),
                embedding_base_url=preset_payload.get("embedding_base_url"),
                embedding_api_key=preset_payload.get("embedding_api_key"),
                embedding_dimensions=preset_payload.get("embedding_dimensions"),
            ),
        )

    @classmethod
    def load_config_graph_query_middleware(cls, source=None) -> "GraphQueryMiddlewareConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


class GraphQueryCapabilityPreset(BaseModel):
    run_id: str | None = Field(default=None, description="graph query 的运行隔离 id。")
    keyword_top_k: int = Field(default=5, ge=1, description="关键词召回默认返回条数。")
    keyword_top_k_limit: int = Field(default=20, ge=1, description="关键词召回允许的最大返回条数。")
    distance_top_k: int = Field(default=5, ge=1, description="距离召回默认返回条数。")
    distance_top_k_limit: int = Field(default=20, ge=1, description="距离召回允许的最大返回条数。")
    distance_max_distance: float = Field(default=1.0, ge=0.0, description="距离召回默认最大累计距离。")
    useful_max_items: int = Field(default=32, ge=1, description="useful 状态桶允许保存的最大条目数。")
    useful_max_total_chars: int = Field(default=4096, ge=1, description="useful 状态桶允许保存的最大正文字符数。")
    blocked_max_items: int = Field(default=32, ge=1, description="blocked 状态桶允许保存的最大条目数。")
    blocked_max_total_chars: int = Field(default=4096, ge=1, description="blocked 状态桶允许保存的最大正文字符数。")
    embedding_provider: str | None = Field(default=None, description="可选的 graph query embedding provider override。")
    embedding_model: str | None = Field(default=None, description="可选的 graph query embedding model override。")
    embedding_base_url: str | None = Field(default=None, description="可选的 graph query embedding base_url override。")
    embedding_api_key: str | None = Field(default=None, description="可选的 graph query embedding api_key override。")
    embedding_dimensions: int | None = Field(default=None, description="可选的 graph query embedding dimensions override。")


class GraphQueryStateTydict(AgentState, GraphQueryToolStateTydict, total=False):
    pass


middleware_runingconfig = GraphQueryMiddlewareConfig.load()
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="graph_query.guidance",
        prompt=(
            "当前任务需要从图中找相关信息时，用短而聚焦的关键词召回，或从已知节点按边距离扩展。"
            "useful / blocked 只服务本次运行：把确实有帮助的节点留下，把干扰当前路径的节点临时屏蔽。"
            "召回工具只接收查询词、锚点、距离和返回上限，不接收任务目标或全局计划。"
        ),
    )
]
MiddlewareToolConfig = {
    "tools": {
        **KeywordRecallToolBundle["tools"],
        **GraphDistanceRecallToolBundle["tools"],
        **GraphMarkUsefulToolBundle["tools"],
        **GraphMarkBlockedToolBundle["tools"],
        **GraphClearBlockedToolBundle["tools"],
    },
    "toolStateTydicts": {
        **KeywordRecallToolBundle["toolStateTydicts"],
        **GraphDistanceRecallToolBundle["toolStateTydicts"],
        **GraphMarkUsefulToolBundle["toolStateTydicts"],
        **GraphMarkBlockedToolBundle["toolStateTydicts"],
        **GraphClearBlockedToolBundle["toolStateTydicts"],
    },
}


def _build_tool_config(config: GraphQueryMiddlewareConfig) -> dict[str, object]:
    preset = config.capability_preset
    resolved_run_id = preset.run_id if preset.run_id is not None else config.run_id
    resolved_config_path = config.neo4j_config_path or (PROJECT_ROOT / "workspace/config/database_config.json").resolve()
    neo4j = resolve_neo4j_connection(connection=config.neo4j, path=config.neo4j_config_path)
    embedding_override = {
        "provider": preset.embedding_provider,
        "model": preset.embedding_model,
        "base_url": preset.embedding_base_url,
        "api_key": preset.embedding_api_key,
        "dimensions": preset.embedding_dimensions,
    }
    cleaned_embedding_override = {
        key: value
        for key, value in embedding_override.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }
    store = GraphStore(
        uri=neo4j.uri,
        username=neo4j.username,
        password=neo4j.password,
        database=neo4j.database,
        run_id=resolved_run_id,
        embedding_config_override=cleaned_embedding_override or None,
    )
    keyword_config = KeywordRecallToolConfig.load().model_copy(
        update={
            "neo4j_config_path": resolved_config_path,
            "run_id": resolved_run_id,
            "default_top_k": preset.keyword_top_k,
            "top_k_limit": preset.keyword_top_k_limit,
        }
    )
    distance_config = GraphDistanceRecallToolConfig.load().model_copy(
        update={
            "neo4j_config_path": resolved_config_path,
            "run_id": resolved_run_id,
            "default_top_k": preset.distance_top_k,
            "top_k_limit": preset.distance_top_k_limit,
            "default_max_distance": preset.distance_max_distance,
        }
    )
    useful_config = GraphMarkUsefulToolConfig.load().model_copy(
        update={
            "neo4j_config_path": resolved_config_path,
            "run_id": resolved_run_id,
            "max_items": preset.useful_max_items,
            "max_total_chars": preset.useful_max_total_chars,
        }
    )
    blocked_config = GraphMarkBlockedToolConfig.load().model_copy(
        update={
            "neo4j_config_path": resolved_config_path,
            "run_id": resolved_run_id,
            "max_items": preset.blocked_max_items,
            "max_total_chars": preset.blocked_max_total_chars,
        }
    )
    clear_config = GraphClearBlockedToolConfig.load().model_copy(update={"run_id": resolved_run_id})
    keyword_tool = build_keyword_recall_tool(config=keyword_config, store=store)
    distance_tool = build_graph_distance_recall_tool(config=distance_config, store=store)
    useful_tool = build_graph_mark_useful_tool(config=useful_config, store=store)
    blocked_tool = build_graph_mark_blocked_tool(config=blocked_config, store=store)
    clear_tool = build_graph_clear_blocked_tool(config=clear_config)
    return {
        "store": store,
        "tools": {
            keyword_tool.name: keyword_tool,
            distance_tool.name: distance_tool,
            useful_tool.name: useful_tool,
            blocked_tool.name: blocked_tool,
            clear_tool.name: clear_tool,
        },
        "toolStateTydicts": {
            "keyword_recall": KeywordRecallToolStateTydict,
            "graph_distance_recall": GraphDistanceRecallToolStateTydict,
            "graph_mark_useful": GraphMarkUsefulToolStateTydict,
            "graph_mark_blocked": GraphMarkBlockedToolStateTydict,
            "graph_clear_blocked": GraphClearBlockedToolStateTydict,
        },
    }


class GraphQueryCapabilityMiddleware(AgentMiddleware):
    name = "graph_query"
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = middleware_runingconfig
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = GraphQueryStateTydict  # type: ignore[assignment]

    def __init__(self, *, config: GraphQueryMiddlewareConfig | None = None) -> None:
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
