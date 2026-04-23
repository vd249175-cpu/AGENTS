from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from MainServer.comm import AgentComm
from MainServer.state import AgentMail
from Deepagents.WorkerAgent.Agent.middlewares.base import (
    BaseAgentMiddleware,
    MiddlewareCapabilityPrompt,
    MiddlewareRuningConfig,
    set_named_system_message,
)


MIDDLEWARE_DIR = Path(__file__).resolve().parent


def _format_inbox(
    messages: list[AgentMail],
) -> str:
    if not messages:
        return ""
    lines = ["<Inbox>"]
    for message in messages:
        source = message.get("from", "unknown")
        message_type = message.get("type", "")
        content = str(message.get("content", "")).strip()
        lines.append(f"[{message_type}] from={source}: {content}")
    lines.append("</Inbox>")
    return "\n".join(lines)


class MiddlewareStateTydict(AgentState, total=False):
    receiveMessagesLastCount: int


class ReceiveMessagesRuningConfig(MiddlewareRuningConfig):
    guidancePrompt: str = Field(
        default="当收到 `<Inbox>` 用户消息时，先阅读内容，再决定是否回复、委派或继续当前任务。"
    )
    maxInboxItems: int = Field(default=20, ge=1)
    peekPeersInGuidance: bool = Field(default=True)


MiddlewareCapabilityPromptSc = MiddlewareCapabilityPrompt

middleware_runingconfig = ReceiveMessagesRuningConfig.load(
    MIDDLEWARE_DIR / "receive_messages_config.json"
)
middleware_capability_prompts = [
    MiddlewareCapabilityPromptSc(
        name="middleware.receive_messages.guidance",
        prompt=middleware_runingconfig.guidancePrompt,
    )
]
MiddlewareToolConfig = {"tools": {}, "toolStateTydicts": {}}


class ReceiveMessagesMiddleware(BaseAgentMiddleware):
    name = "receive_messages"
    state_schema = MiddlewareStateTydict
    tools = []

    def __init__(
        self,
        *,
        comm: AgentComm | None = None,
        runingConfig: ReceiveMessagesRuningConfig = middleware_runingconfig,
    ) -> None:
        self.comm = comm
        super().__init__(
            runingConfig=runingConfig,
            capabilityPromptConfigs=middleware_capability_prompts,
            tools=[],
        )

    def _guidance_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        guided = self._with_guidance(request)
        if self.comm is None or not self.runingConfig.peekPeersInGuidance:
            return guided
        try:
            peers = self.comm.peers()
        except Exception:
            peers = []
        peers_text = ", ".join(peers) if peers else "未知或空"
        messages = set_named_system_message(
            list(guided.messages),
            name="middleware.receive_messages.peers",
            text=f"当前可达 peers={peers_text}",
        )
        return guided.override(messages=messages)

    def _with_inbox(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        if not self.runingConfig.enabled or self.comm is None:
            return request
        try:
            inbox_messages = list(self.comm.recv() or [])[: self.runingConfig.maxInboxItems]
        except Exception:
            inbox_messages = []
        inbox_text = _format_inbox(inbox_messages)
        if not inbox_text:
            return request
        messages = list(request.messages)
        messages.append(HumanMessage(content=inbox_text, name="agent_inbox"))
        return request.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        return handler(self._with_inbox(self._guidance_request(request)))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        return await handler(self._with_inbox(self._guidance_request(request)))
