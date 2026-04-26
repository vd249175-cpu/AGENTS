"""Knowledge ingest middleware that owns ingest_knowledge_document."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import AIMessage
from pydantic import Field

from .base import BaseAgentMiddleware, MiddlewareCapabilityPrompt, MiddlewareRuningConfig
from ..tools.ingest_knowledge_tool import Config as IngestKnowledgeToolConfig
from ..tools.ingest_knowledge_tool import IngestKnowledgeTool
from ..tools.ingest_knowledge_tool import SubState as IngestKnowledgeToolState


class Config(IngestKnowledgeToolConfig, MiddlewareRuningConfig):
    guidancePrompt: str = Field(
        default=(
            "如果你需要记忆长文档，请使用ingest_knowledge_document工具将文档写入graphrag知识库中"
            "长文档的存储位置统一为 /workspace/knowledge，"
            "当文档入库后，你可以通过manage_knowledge 查询或操作识库中的内容。"
            "manage_knowledge 工具还可以记住零碎的关键内容，也可以将其放入知识库中的合适位置"
        )
    )

    @classmethod
    def load_config_knowledge_ingest(cls, source=None):
        return cls.load(source)


KnowledgeIngestRuningConfig = Config


class SubState(AgentState, IngestKnowledgeToolState, total=False):
    knowledgeIngestGuidanceInjected: bool


MiddlewareStateTydict = SubState


class MiddlewareSchema:
    name = "knowledge_ingest"
    tools = {"ingest_knowledge_document": IngestKnowledgeTool}
    state_schema = SubState


class Middleware(BaseAgentMiddleware):
    name = MiddlewareSchema.name
    state_schema = SubState

    def __init__(
        self,
        *,
        runingConfig: KnowledgeIngestRuningConfig | None = None,
    ) -> None:
        self.config = runingConfig or Config()
        tool_config = IngestKnowledgeToolConfig(
            resume=self.config.resume,
            shardCount=self.config.shardCount,
            maxWorkers=self.config.maxWorkers,
            referenceBytes=self.config.referenceBytes,
            agentName=self.config.agentName,
        )
        ingest_tool = IngestKnowledgeTool(config=tool_config)
        self.toolConfig = {
            "tools": {"ingest_knowledge_document": ingest_tool.tool},
            "toolStateTydicts": {"ingest_knowledge_document": IngestKnowledgeToolState},
        }
        super().__init__(
            runingConfig=self.config,
            capabilityPromptConfigs=[
                MiddlewareCapabilityPrompt(
                    name="middleware.knowledge_ingest.guidance",
                    prompt=self.config.guidancePrompt,
                )
            ],
            tools=list(self.toolConfig["tools"].values()),
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        return handler(self._with_guidance(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        return await handler(self._with_guidance(request))


class KnowledgeIngestMiddleware:
    name = MiddlewareSchema.name
    config = Config
    substate = SubState
    middlewareschema = MiddlewareSchema

    def __init__(self, *, runingConfig: KnowledgeIngestRuningConfig | None = None) -> None:
        self.config = runingConfig or Config()
        self.middleware = Middleware(runingConfig=self.config)


knowledge_ingest_middleware = KnowledgeIngestMiddleware().middleware
