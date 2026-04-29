"""Single-file receive-messages middleware following the demo wrapper pattern."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from MainServer.comm import AgentComm
from MainServer.state import AgentMail
from ..server.demo_server import set_named_system_message
from ..server.path_resolver import workspace_visible_path
from .base import BaseAgentMiddleware, MiddlewareCapabilityPrompt, MiddlewareRuningConfig


MIDDLEWARE_DIR = Path(__file__).resolve().parent


class Config(MiddlewareRuningConfig):
    guidancePrompt: str = Field(
        default="当收到 `<Inbox>` 用户消息时，先阅读内容，再决定是否回复、委派或继续当前任务。如果你需要和其他agent交流，你总是需要调用工具，否则交流不会生效"
    )
    maxInboxItems: int = Field(default=20, ge=1)
    peekPeersInGuidance: bool = Field(default=True)

    @classmethod
    def load_config_receive_messages(cls, source=None):
        return cls.load(source)


ReceiveMessagesRuningConfig = Config


class SubState(AgentState, total=False):
    receiveMessagesLastCount: int


MiddlewareStateTydict = SubState


class MiddlewareSchema:
    name = "receive_messages"
    tools = {}
    state_schema = SubState


middleware_runingconfig = Config.load_config_receive_messages(MIDDLEWARE_DIR / "receive_messages_config.json")
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="middleware.receive_messages.guidance",
        prompt=middleware_runingconfig.guidancePrompt,
    )
]


def _visible_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str):
        if key in {"link", "value"}:
            return workspace_visible_path(value)
        return value
    if isinstance(value, list):
        return [_visible_value(item) for item in value]
    if isinstance(value, dict):
        return {item_key: _visible_value(item, key=str(item_key)) for item_key, item in value.items()}
    return value


def _format_attachments(message: AgentMail) -> list[str]:
    attachments = message.get("attachments") or []
    if not attachments:
        return []
    lines = ["attachments:"]
    for index, attachment in enumerate(attachments, 1):
        link = workspace_visible_path(str(attachment.get("link") or attachment.get("value") or ""))
        summary = str(attachment.get("summary") or "").strip()
        if summary:
            lines.append(f"- {index}. {link} ({summary})")
        else:
            lines.append(f"- {index}. {link}")
    return lines


def _format_inbox(messages: list[AgentMail]) -> str:
    if not messages:
        return ""
    lines = ["<Inbox>"]
    for message in messages:
        source = message.get("from", "unknown")
        message_type = message.get("type", "")
        content = str(_visible_value(message.get("content", ""))).strip()
        lines.append(f"[{message_type}] from={source}: {content}")
        lines.extend(_format_attachments(message))
    lines.append("</Inbox>")
    return "\n".join(lines)


class Middleware(BaseAgentMiddleware):
    name = MiddlewareSchema.name
    state_schema = SubState
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
            text=f"当前可达 peers={peers_text}你可以向他们发送邮件，peers中为他们的agent name",
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


class ReceiveMessagesMiddleware:
    name = MiddlewareSchema.name
    config = Config
    substate = SubState
    middlewareschema = MiddlewareSchema

    def __init__(
        self,
        *,
        comm: AgentComm | None = None,
        runingConfig: ReceiveMessagesRuningConfig | None = None,
    ) -> None:
        self.config = runingConfig or middleware_runingconfig
        self.middleware = Middleware(comm=comm, runingConfig=self.config)


receive_messages_middleware = ReceiveMessagesMiddleware().middleware
