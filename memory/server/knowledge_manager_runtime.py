"""Internal runtime builder for the knowledge manager agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState
from langchain.chat_models import BaseChatModel, init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict, Field

from middleware.document_query import DocumentQueryCapabilityMiddleware, DocumentQueryMiddlewareConfig
from middleware.document_write import DocumentWriteCapabilityMiddleware, DocumentWriteMiddlewareConfig
from middleware.graph_query import GraphQueryCapabilityMiddleware, GraphQueryCapabilityPreset, GraphQueryMiddlewareConfig
from middleware.graph_write import GraphWriteCapabilityMiddleware, GraphWriteMiddlewareConfig
from middleware.management_discovery import ManagementDiscoveryMiddleware, ManagementDiscoveryMiddlewareConfig
from server.component_config import config_from_external
from models import get_chat_model_config
from server.config_overrides import merge_model
from server.neo4j.database_config import DEFAULT_DATABASE_CONFIG_PATH, Neo4jConnectionConfig


DEFAULT_KNOWLEDGE_MANAGER_SYSTEM_PROMPT = """\
你是一个相对独立的知识库管理者。你的工作对象是当前委派目标，以及你能通过工具读写到的文档 chunk 和图节点。
所有管理动作都必须通过当前可用工具完成；不要把未执行的计划说成已经完成。
先确认已有内容，再决定读取、创建、更新、删除或关联；不确定时优先读取，不要凭记忆猜测节点或文档状态。
如果委派目标明确点名一个当前可用的操作，可以按该操作执行，但仍必须补足该操作 schema 所需的具体内容。
工具入参只填写该工具 schema 要求的局部信息，不要把主 agent 目标、run_id、thread_id 或全局策略塞进底层工具参数。
写文档时保持 chunk 顺序、正文、摘要、关键词和普通图边各自清楚；写图时只处理 GraphNode 和普通图边，不修改 Chunk 正文、摘要或关键词。
创建 GraphNode 后，先从 graph_create_nodes 的工具结果中确认真实 ids；后续更新、读取或连边使用这些真实 ids，不要把摘要、正文里的唯一标记词当作 node id。
一次给同一个节点连接多个目标时，调用一次 graph_update_node，并在 edge_ops 中为每条普通图边写一个独立对象；每个 targets 填真实 node id，或填带 document_name 语境的 chunk_index。
useful / blocked 是当前运行态控制桶：只把对主 agent 后续判断确实有价值的节点放进 useful；把会干扰当前查询路径的节点放进 blocked。
管理过程中参考 ManagementDiscoveries 中记录的已确认发现、风险和待跟进事项，但不要把它们当成长期事实。
结束时简洁说明：完成了哪些操作、确认了哪些关键事实、仍有哪些风险或建议。不要输出内部工具调用细节，除非它直接影响主 agent 决策。
"""


class KnowledgeManagerModelOverride(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str | BaseChatModel | None = Field(default=None, description="覆盖内部 manager agent 使用的 chat model。")
    model_provider: str | None = Field(default=None, description="覆盖模型 provider。")
    base_url: str | None = Field(default=None, description="覆盖模型 base_url。")
    api_key: str | None = Field(default=None, description="覆盖模型 api_key。")
    temperature: float | None = Field(default=None, description="覆盖内部模型 temperature。")


class KnowledgeManagerEmbeddingOverride(BaseModel):
    provider: str | None = Field(default=None, description="覆盖内部 embedding provider。")
    model: str | None = Field(default=None, description="覆盖内部 embedding model。")
    base_url: str | None = Field(default=None, description="覆盖内部 embedding base_url。")
    api_key: str | None = Field(default=None, description="覆盖内部 embedding api_key。")
    dimensions: int | None = Field(default=None, description="覆盖内部 embedding dimensions。")


class ManagementDiscoveryMiddlewareOverride(BaseModel):
    max_items: int | None = Field(default=None, ge=1, description="覆盖发现桶最大条目数。")
    max_total_chars: int | None = Field(default=None, ge=1, description="覆盖发现桶最大字符数。")
    max_summary_chars: int | None = Field(default=None, ge=32, description="覆盖单条发现摘要长度上限。")
    scan_message_limit: int | None = Field(default=None, ge=1, description="覆盖发现扫描消息上限。")


class DocumentQueryMiddlewareOverride(BaseModel):
    neo4j_config_path: Path | None = Field(default=None, description="覆盖 document query 的 Neo4j 配置路径。")
    run_id: str | None = Field(default=None, description="覆盖 document query 的 run_id。")
    trace_limit: int | None = Field(default=None, ge=1, description="覆盖 document query trace 上限。")


class DocumentWriteMiddlewareOverride(BaseModel):
    neo4j_config_path: Path | None = Field(default=None, description="覆盖 document write 的 Neo4j 配置路径。")
    run_id: str | None = Field(default=None, description="覆盖 document write 的 run_id。")
    trace_limit: int | None = Field(default=None, ge=1, description="覆盖 document write trace 上限。")
    embedding_provider: str | None = Field(default=None, description="覆盖 document write embedding provider。")
    embedding_model: str | None = Field(default=None, description="覆盖 document write embedding model。")
    embedding_base_url: str | None = Field(default=None, description="覆盖 document write embedding base_url。")
    embedding_api_key: str | None = Field(default=None, description="覆盖 document write embedding api_key。")
    embedding_dimensions: int | None = Field(default=None, description="覆盖 document write embedding dimensions。")


class GraphQueryCapabilityPresetOverride(BaseModel):
    run_id: str | None = Field(default=None, description="覆盖 graph query preset 的 run_id。")
    keyword_top_k: int | None = Field(default=None, ge=1, description="覆盖关键词召回默认 top_k。")
    keyword_top_k_limit: int | None = Field(default=None, ge=1, description="覆盖关键词召回最大 top_k。")
    distance_top_k: int | None = Field(default=None, ge=1, description="覆盖距离召回默认 top_k。")
    distance_top_k_limit: int | None = Field(default=None, ge=1, description="覆盖距离召回最大 top_k。")
    distance_max_distance: float | None = Field(default=None, ge=0.0, description="覆盖距离召回最大累计距离。")
    useful_max_items: int | None = Field(default=None, ge=1, description="覆盖 useful 桶最大条目数。")
    useful_max_total_chars: int | None = Field(default=None, ge=1, description="覆盖 useful 桶最大字符数。")
    blocked_max_items: int | None = Field(default=None, ge=1, description="覆盖 blocked 桶最大条目数。")
    blocked_max_total_chars: int | None = Field(default=None, ge=1, description="覆盖 blocked 桶最大字符数。")
    embedding_provider: str | None = Field(default=None, description="覆盖 graph query embedding provider。")
    embedding_model: str | None = Field(default=None, description="覆盖 graph query embedding model。")
    embedding_base_url: str | None = Field(default=None, description="覆盖 graph query embedding base_url。")
    embedding_api_key: str | None = Field(default=None, description="覆盖 graph query embedding api_key。")
    embedding_dimensions: int | None = Field(default=None, description="覆盖 graph query embedding dimensions。")


class GraphQueryMiddlewareOverride(BaseModel):
    neo4j_config_path: Path | None = Field(default=None, description="覆盖 graph query 的 Neo4j 配置路径。")
    run_id: str | None = Field(default=None, description="覆盖 graph query 的 run_id。")
    trace_limit: int | None = Field(default=None, ge=1, description="覆盖 graph query trace 上限。")
    capability_preset: GraphQueryCapabilityPresetOverride = Field(
        default_factory=GraphQueryCapabilityPresetOverride,
        description="覆盖 graph query preset。",
    )


class GraphWriteMiddlewareOverride(BaseModel):
    neo4j_config_path: Path | None = Field(default=None, description="覆盖 graph write 的 Neo4j 配置路径。")
    run_id: str | None = Field(default=None, description="覆盖 graph write 的 run_id。")
    trace_limit: int | None = Field(default=None, ge=1, description="覆盖 graph write trace 上限。")
    embedding_provider: str | None = Field(default=None, description="覆盖 graph write embedding provider。")
    embedding_model: str | None = Field(default=None, description="覆盖 graph write embedding model。")
    embedding_base_url: str | None = Field(default=None, description="覆盖 graph write embedding base_url。")
    embedding_api_key: str | None = Field(default=None, description="覆盖 graph write embedding api_key。")
    embedding_dimensions: int | None = Field(default=None, description="覆盖 graph write embedding dimensions。")


class KnowledgeManagerAgentOverrides(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: KnowledgeManagerModelOverride = Field(default_factory=KnowledgeManagerModelOverride)
    embedding: KnowledgeManagerEmbeddingOverride = Field(default_factory=KnowledgeManagerEmbeddingOverride)
    system_prompt: str | None = Field(default=None, description="覆盖内部 manager agent 的 system prompt。")
    debug: bool | None = Field(default=None, description="覆盖内部 manager agent 的 debug 开关。")
    checkpointer: Any | None = Field(default=None, description="覆盖内部 manager agent 的 checkpointer。")
    discovery: ManagementDiscoveryMiddlewareOverride = Field(default_factory=ManagementDiscoveryMiddlewareOverride)
    document_query: DocumentQueryMiddlewareOverride = Field(default_factory=DocumentQueryMiddlewareOverride)
    document_write: DocumentWriteMiddlewareOverride = Field(default_factory=DocumentWriteMiddlewareOverride)
    graph_query: GraphQueryMiddlewareOverride = Field(default_factory=GraphQueryMiddlewareOverride)
    graph_write: GraphWriteMiddlewareOverride = Field(default_factory=GraphWriteMiddlewareOverride)


class KnowledgeManagerAgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_default=True)

    model: str | BaseChatModel | None = Field(default=None, description="内部 manager agent 使用的聊天模型。")
    run_id: str = Field(default="knowledge-manager", description="内部 manager agent 的运行隔离 id。")
    neo4j: Neo4jConnectionConfig | None = Field(default=None, description="显式的 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="Neo4j 配置文件路径。")
    system_prompt: str | None = Field(default=None, description="覆盖内部 manager agent 的 system prompt。")
    temperature: float = Field(default=0.0, description="内部 manager agent 的模型温度。")
    debug: bool = Field(default=False, description="是否开启内部 manager agent 调试。")
    checkpointer: Any | None = Field(default=None, description="覆盖内部 manager agent 的 checkpointer。")
    overrides: KnowledgeManagerAgentOverrides = Field(
        default_factory=KnowledgeManagerAgentOverrides,
        description="覆盖内部 manager agent 的中间键、工具和模型构造参数。",
    )

    @classmethod
    def load_config_knowledge_manager_agent(cls, source=None) -> "KnowledgeManagerAgentConfig":
        if source is None:
            return cls()
        return config_from_external(cls, source)


def _resolve_database_config_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_DATABASE_CONFIG_PATH
    return Path(path).expanduser().resolve()


def _embedding_override_payload(config: KnowledgeManagerEmbeddingOverride) -> dict[str, Any]:
    override = {
        "embedding_provider": config.provider,
        "embedding_model": config.model,
        "embedding_base_url": config.base_url,
        "embedding_api_key": config.api_key,
        "embedding_dimensions": config.dimensions,
    }
    return {
        key: value
        for key, value in override.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }


def _build_chat_model(
    model: str | BaseChatModel | None,
    *,
    temperature: float,
    model_provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str | BaseChatModel:
    if isinstance(model, BaseChatModel):
        return model
    chat_config = get_chat_model_config()
    resolved_model = str(model).strip() if isinstance(model, str) and model.strip() else str(chat_config["model"])
    return init_chat_model(
        model=resolved_model,
        model_provider=str(model_provider or chat_config["provider"]),
        base_url=str(base_url or chat_config["base_url"]),
        api_key=str(api_key or chat_config["api_key"]),
        temperature=temperature,
    )


class KnowledgeManagerAgentStateTydict(AgentState, total=False):
    pass


class KnowledgeManagerAgentSchema:
    name = "knowledge-manager-agent"
    systemPrompt = DEFAULT_KNOWLEDGE_MANAGER_SYSTEM_PROMPT
    middlewares = {
        ManagementDiscoveryMiddleware.name: ManagementDiscoveryMiddleware,
        DocumentQueryCapabilityMiddleware.name: DocumentQueryCapabilityMiddleware,
        DocumentWriteCapabilityMiddleware.name: DocumentWriteCapabilityMiddleware,
        GraphQueryCapabilityMiddleware.name: GraphQueryCapabilityMiddleware,
        GraphWriteCapabilityMiddleware.name: GraphWriteCapabilityMiddleware,
    }


class KnowledgeManagerAgent:
    name = KnowledgeManagerAgentSchema.name
    config = KnowledgeManagerAgentConfig
    substate = KnowledgeManagerAgentStateTydict
    agentschema = KnowledgeManagerAgentSchema

    def __init__(self, config: KnowledgeManagerAgentConfig | None = None) -> None:
        self.config = config or KnowledgeManagerAgentConfig()
        self.agent = self.create_agent()

    def create_agent(self):
        return _build_knowledge_manager_agent(
            model=self.config.model,
            run_id=self.config.run_id,
            neo4j=self.config.neo4j,
            neo4j_config_path=self.config.neo4j_config_path,
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            debug=self.config.debug,
            checkpointer=self.config.checkpointer,
            overrides=self.config.overrides,
        )


def _build_knowledge_manager_agent(
    model: str | BaseChatModel | None = None,
    *,
    run_id: str,
    neo4j: Neo4jConnectionConfig | dict[str, Any] | None = None,
    neo4j_config_path: str | Path | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    debug: bool = False,
    checkpointer: Any | None = None,
    overrides: KnowledgeManagerAgentOverrides | dict[str, Any] | None = None,
):
    resolved_config_path = _resolve_database_config_path(neo4j_config_path)
    if isinstance(overrides, dict):
        overrides = KnowledgeManagerAgentOverrides.model_validate(overrides)
    active_overrides = overrides or KnowledgeManagerAgentOverrides()
    embedding_override = _embedding_override_payload(active_overrides.embedding)
    discovery_config = merge_model(ManagementDiscoveryMiddlewareConfig.load(), active_overrides.discovery)
    document_query_config = merge_model(
        DocumentQueryMiddlewareConfig(
            neo4j=Neo4jConnectionConfig.model_validate(neo4j) if isinstance(neo4j, dict) else neo4j,
            neo4j_config_path=resolved_config_path,
            run_id=run_id,
            trace_limit=16,
        ),
        active_overrides.document_query,
    )
    document_write_config = merge_model(
        DocumentWriteMiddlewareConfig(
            neo4j=Neo4jConnectionConfig.model_validate(neo4j) if isinstance(neo4j, dict) else neo4j,
            neo4j_config_path=resolved_config_path,
            run_id=run_id,
            trace_limit=16,
            **embedding_override,
        ),
        active_overrides.document_write,
    )
    graph_query_config = merge_model(
        GraphQueryMiddlewareConfig(
            neo4j=Neo4jConnectionConfig.model_validate(neo4j) if isinstance(neo4j, dict) else neo4j,
            neo4j_config_path=resolved_config_path,
            run_id=run_id,
            trace_limit=16,
            capability_preset=GraphQueryCapabilityPreset(run_id=run_id, **embedding_override),
        ),
        active_overrides.graph_query,
    )
    graph_write_config = merge_model(
        GraphWriteMiddlewareConfig(
            neo4j=Neo4jConnectionConfig.model_validate(neo4j) if isinstance(neo4j, dict) else neo4j,
            neo4j_config_path=resolved_config_path,
            run_id=run_id,
            trace_limit=16,
            **embedding_override,
        ),
        active_overrides.graph_write,
    )
    middlewares: list[Any] = [
        ManagementDiscoveryMiddleware(config=discovery_config),
        DocumentQueryCapabilityMiddleware(config=document_query_config),
        DocumentWriteCapabilityMiddleware(config=document_write_config),
        GraphQueryCapabilityMiddleware(config=graph_query_config),
        GraphWriteCapabilityMiddleware(config=graph_write_config),
    ]
    model_override = active_overrides.model
    chat_model = _build_chat_model(
        model_override.model if model_override.model is not None else model,
        temperature=model_override.temperature if model_override.temperature is not None else temperature,
        model_provider=model_override.model_provider,
        base_url=model_override.base_url,
        api_key=model_override.api_key,
    )
    resolved_checkpointer = active_overrides.checkpointer or checkpointer or InMemorySaver()
    agent = create_agent(
        model=chat_model,
        system_prompt=active_overrides.system_prompt or system_prompt or DEFAULT_KNOWLEDGE_MANAGER_SYSTEM_PROMPT,
        middleware=middlewares,
        checkpointer=resolved_checkpointer,
        debug=active_overrides.debug if active_overrides.debug is not None else debug,
        name="knowledge-manager-agent",
    )
    def close() -> None:
        for middleware in middlewares:
            close_method = getattr(middleware, "close", None)
            if callable(close_method):
                close_method()
    try:
        object.__setattr__(agent, "close", close)
    except Exception:
        pass
    return agent


def create_knowledge_manager_agent(
    model: str | BaseChatModel | None = None,
    *,
    run_id: str,
    neo4j: Neo4jConnectionConfig | dict[str, Any] | None = None,
    neo4j_config_path: str | Path | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    debug: bool = False,
    checkpointer: Any | None = None,
    overrides: KnowledgeManagerAgentOverrides | dict[str, Any] | None = None,
):
    wrapper = KnowledgeManagerAgent(
        KnowledgeManagerAgentConfig(
            model=model,
            run_id=run_id,
            neo4j=Neo4jConnectionConfig.model_validate(neo4j) if isinstance(neo4j, dict) else neo4j,
            neo4j_config_path=neo4j_config_path,
            system_prompt=system_prompt,
            temperature=temperature,
            debug=debug,
            checkpointer=checkpointer,
            overrides=KnowledgeManagerAgentOverrides.model_validate(overrides) if isinstance(overrides, dict) else (overrides or KnowledgeManagerAgentOverrides()),
        )
    )
    return wrapper.agent


__all__ = [
    "DEFAULT_KNOWLEDGE_MANAGER_SYSTEM_PROMPT",
    "DocumentQueryMiddlewareOverride",
    "DocumentWriteMiddlewareOverride",
    "GraphQueryCapabilityPresetOverride",
    "GraphQueryMiddlewareOverride",
    "GraphWriteMiddlewareOverride",
    "KnowledgeManagerAgent",
    "KnowledgeManagerAgentConfig",
    "KnowledgeManagerAgentOverrides",
    "KnowledgeManagerEmbeddingOverride",
    "KnowledgeManagerModelOverride",
    "KnowledgeManagerAgentSchema",
    "KnowledgeManagerAgentStateTydict",
    "ManagementDiscoveryMiddlewareOverride",
    "create_knowledge_manager_agent",
]

Config = KnowledgeManagerAgentConfig
SubState = KnowledgeManagerAgentStateTydict
AgentSchema = KnowledgeManagerAgentSchema
Agent = KnowledgeManagerAgent
