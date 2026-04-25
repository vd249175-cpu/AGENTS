"""Knowledge manager capability middleware declaration."""

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field
from server.component_config import config_from_external
from server.config_overrides import merge_model
from server.neo4j import Neo4jConnectionConfig
from server.knowledge_manager_runtime import KnowledgeManagerAgentOverrides
from tools.manage_knowledge import (
    ManageKnowledgeToolConfig,
    ManageKnowledgeToolOverride,
    ManageKnowledgeTool,
    ManageKnowledgeToolStateTydict,
    build_manage_knowledge_tool,
)


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("knowledge_manager.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class KnowledgeManagerToolConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_default=True)

    temperature: float = Field(default=0.0, description="内部 manager agent 的模型温度。")
    debug: bool = Field(default=False, description="是否开启内部 manager agent 调试。")
    stream_inner_agent: bool = Field(default=False, description="是否在工具调用时流式跑内部 manager agent。")
    inner_recursion_limit: int = Field(default=64, ge=1, description="内部 manager agent 的递归上限。")
    agent_overrides: KnowledgeManagerAgentOverrides = Field(
        default_factory=KnowledgeManagerAgentOverrides,
        description="覆盖内部 manager agent 的模型、embedding 和子能力构造参数。",
    )


class KnowledgeManagerMiddlewareConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_default=True)

    neo4j: Neo4jConnectionConfig | None = Field(
        default=None,
        description="显式的 Neo4j 连接参数。对外推荐直接传这组字段，不再依赖 database_config.json 中转。",
    )
    neo4j_config_path: Path | None = Field(
        default=None,
        description=(
            "兼容旧链路的 Neo4j 配置文件路径。"
            "通常指向 workspace/config/database_config.json；对外不再推荐使用这层中转。"
        )
    )
    run_id: str | None = Field(
        default=None,
        description=(
            "知识库隔离 id。内部 document/graph 工具默认都会继承它来决定读写哪一套数据。"
            "同一个 Neo4j 里要并存多套知识库时，应显式设置。"
        ),
    )
    trace_limit: int = Field(
        default=16,
        ge=1,
        description="运行时最多保留多少条 trace，用于调试知识管理链路；一般保持 16 即可。",
    )
    tool: KnowledgeManagerToolConfig = Field(
        default_factory=KnowledgeManagerToolConfig,
        description="manage_knowledge 工具和内部 manager agent 的默认构造配置。",
    )

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "KnowledgeManagerMiddlewareConfig":
        payload = _load_json(path)
        tool_payload = payload.get("tool") if isinstance(payload.get("tool"), dict) else {}
        return cls(
            neo4j=Neo4jConnectionConfig.model_validate(payload["neo4j"]) if isinstance(payload.get("neo4j"), dict) else None,
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            trace_limit=max(1, int(payload.get("trace_limit", 16))),
            tool=KnowledgeManagerToolConfig.model_validate(tool_payload),
        )

    @classmethod
    def load_config_knowledge_manager_middleware(cls, source=None) -> "KnowledgeManagerMiddlewareConfig":
        """Load the demo-style top-level knowledge manager middleware context."""

        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class KnowledgeManagerMiddlewareOverride(BaseModel):
    neo4j: Neo4jConnectionConfig | dict[str, object] | None = Field(
        default=None,
        description="临时覆盖知识管理中间键的显式 Neo4j 连接参数。",
    )
    neo4j_config_path: Path | None = Field(
        default=None,
        description="临时覆盖知识管理中间键的 Neo4j 配置路径，适合同一套装配切换到另一套数据库环境。",
    )
    run_id: str | None = Field(
        default=None,
        description="临时覆盖知识管理中间键的 run_id，适合同一个 agent 模板复用到不同知识库实例。",
    )
    trace_limit: int | None = Field(
        default=None,
        ge=1,
        description="临时覆盖知识管理中间键的 trace 上限。",
    )


class KnowledgeManagerCapabilityOverrides(BaseModel):
    middleware: KnowledgeManagerMiddlewareOverride = Field(
        default_factory=KnowledgeManagerMiddlewareOverride,
        description="覆盖 KnowledgeManagerCapabilityMiddleware 这一层自己的配置。",
    )
    tool: ManageKnowledgeToolOverride = Field(
        default_factory=ManageKnowledgeToolOverride,
        description="覆盖它挂出的 manage_knowledge 工具，以及内部 manager agent 的配置。",
    )


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


class AffectedPrompt(MiddlewareCapabilityPrompt):
    pass


class AffectedPrompts(BaseModel):
    Prompts: list[AffectedPrompt]


class KnowledgeManagerStateTydict(AgentState, total=False):
    pass


middleware_runingconfig = KnowledgeManagerMiddlewareConfig.load()
middleware_capability_prompts = [
    AffectedPrompt(
        name="knowledge_manager.guidance",
        prompt=(
            "如果当前任务需要查询、整理、修正或关联知识库内容，把目标交给 manage_knowledge。"
            "主 agent 只需要写清 target，不要替内部知识管理者规划底层工具步骤。"
            "收到结果后，根据 message 和 useful_items 继续完成当前对话任务。"
        ),
    )
]
affected_prompts = AffectedPrompts(Prompts=middleware_capability_prompts)


class KnowledgeManagerMiddlewareSchema:
    name = "knowledge_manager"
    affectedPrompts = affected_prompts
    tools = {
        "manage_knowledge": ManageKnowledgeTool,
    }


def _build_tool_config(
    config: KnowledgeManagerMiddlewareConfig,
    *,
    overrides: ManageKnowledgeToolOverride | None = None,
) -> dict[str, object]:
    manage_config = ManageKnowledgeToolConfig(
        neo4j=config.neo4j,
        neo4j_config_path=config.neo4j_config_path,
        run_id=config.run_id,
        temperature=config.tool.temperature,
        debug=config.tool.debug,
        stream_inner_agent=config.tool.stream_inner_agent,
        inner_recursion_limit=config.tool.inner_recursion_limit,
        agent_overrides=config.tool.agent_overrides,
    )
    manage_config = merge_model(manage_config, overrides)
    manage_tool = build_manage_knowledge_tool(config=manage_config)
    return {
        "tools": {
            manage_tool.name: manage_tool,
        },
        "toolStateTydicts": {
            "manage_knowledge": ManageKnowledgeToolStateTydict,
        },
    }


MiddlewareToolConfig = _build_tool_config(middleware_runingconfig)


class KnowledgeManagerCapabilityMiddleware(AgentMiddleware):
    name = KnowledgeManagerMiddlewareSchema.name
    config = KnowledgeManagerMiddlewareConfig
    substate = KnowledgeManagerStateTydict
    middlewareschema = KnowledgeManagerMiddlewareSchema
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = middleware_runingconfig
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = KnowledgeManagerStateTydict  # type: ignore[assignment]

    def __init__(
        self,
        *,
        config: KnowledgeManagerMiddlewareConfig | None = None,
        overrides: KnowledgeManagerCapabilityOverrides | dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(overrides, dict):
            overrides = KnowledgeManagerCapabilityOverrides.model_validate(overrides)
        base_config = config or self.runingConfig
        self.config = merge_model(base_config, overrides.middleware if overrides is not None else None)
        self.capability_prompts = self.capabilityPromptConfigs
        runtime_tool_config = _build_tool_config(self.config, overrides=overrides.tool if overrides is not None else None)
        self.toolConfig = runtime_tool_config
        self.toolStateTydicts = runtime_tool_config["toolStateTydicts"]
        self.tools = list(runtime_tool_config["tools"].values())
        self.middleware = self

    def close(self) -> None:
        for tool in self.tools:
            close_method = getattr(tool, "close", None)
            if callable(close_method):
                close_method()

    def before_model(
        self,
        state: KnowledgeManagerStateTydict,
        runtime: Runtime[KnowledgeManagerMiddlewareConfig],
    ) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        return {"messages": self._with_guidance_message(messages)}

    def _with_guidance_message(self, messages: list[Any]) -> list[Any]:
        guidance = self.capabilityPromptConfigs[0].prompt
        new_message = SystemMessage(name="knowledge_manager.guidance", content=guidance)
        replaced = False
        next_messages: list[Any] = []
        for message in messages:
            if isinstance(message, SystemMessage) and message.name == "knowledge_manager.guidance":
                next_messages.append(new_message)
                replaced = True
            else:
                next_messages.append(message)
        if not replaced:
            next_messages.insert(0, new_message)
        return next_messages

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage:
        return handler(request.override(messages=self._with_guidance_message(list(request.messages))))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        return await handler(request.override(messages=self._with_guidance_message(list(request.messages))))


Config = KnowledgeManagerMiddlewareConfig
SubState = KnowledgeManagerStateTydict
MiddlewareSchema = KnowledgeManagerMiddlewareSchema
Middleware = KnowledgeManagerCapabilityMiddleware
