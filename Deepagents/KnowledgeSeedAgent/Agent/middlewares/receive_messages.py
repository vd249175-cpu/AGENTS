"""Single-file receive-messages middleware following the demo wrapper pattern."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse, Runtime
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
        default=(
            "当收到 `<Inbox>` 用户消息时，先阅读内容，再决定是否回复、委派或继续当前任务。"
            "如果邮件要求你回信，必须调用 send_message_to_agent 回复发件人；只在最终回答里说“已回复”不会产生通讯。"
            "邮件读取、状态上报和继续对话都属于当前主 agent 的同一个 checkpoint 线程；不要为邮件创建或切换新线程。"
        )
    )
    maxInboxItems: int = Field(default=20, ge=1)
    peekPeersInGuidance: bool = Field(default=True)

    @classmethod
    def load_config_receive_messages(cls, source=None):
        return cls.load(source)


ReceiveMessagesRuningConfig = Config


class SubState(AgentState, total=False):
    receiveMessagesLastCount: int
    receiveMessagesLastInbox: str | None
    receiveMessagesLastSender: str | None
    receiveMessagesLastMessageId: str | None
    receiveMessagesLastMailMetadataThreadId: str | None
    receiveMessagesLastMailMetadataRunId: str | None


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
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        content = str(_visible_value(message.get("content", ""))).strip()
        metadata_text = ""
        if metadata.get("thread_id") or metadata.get("run_id"):
            metadata_text = f" metadata(thread_id={metadata.get('thread_id')}, run_id={metadata.get('run_id')})"
        lines.append(f"[{message_type}] from={source} message_id={message.get('message_id', '')}{metadata_text}: {content}")
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

    def _inbox_update(self) -> dict[str, Any] | None:
        if not self.runingConfig.enabled or self.comm is None:
            return None
        try:
            inbox_messages = list(self.comm.recv() or [])[: self.runingConfig.maxInboxItems]
        except Exception:
            inbox_messages = []
        inbox_text = _format_inbox(inbox_messages)
        if not inbox_text:
            return None
        last_message = inbox_messages[-1]
        metadata = last_message.get("metadata") if isinstance(last_message.get("metadata"), dict) else {}
        mail_metadata_thread_id = (
            str(metadata.get("thread_id") or metadata.get("conversation_thread_id") or "").strip() or None
        )
        mail_metadata_run_id = (
            str(metadata.get("run_id") or metadata.get("conversation_run_id") or "").strip() or None
        )
        return {
            "messages": [HumanMessage(content=inbox_text, name="agent_inbox")],
            "receiveMessagesLastCount": len(inbox_messages),
            "receiveMessagesLastInbox": inbox_text,
            "receiveMessagesLastSender": str(last_message.get("from") or "").strip() or None,
            "receiveMessagesLastMessageId": str(last_message.get("message_id") or "").strip() or None,
            "receiveMessagesLastMailMetadataThreadId": mail_metadata_thread_id,
            "receiveMessagesLastMailMetadataRunId": mail_metadata_run_id,
        }

    def before_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        del state
        update = self._inbox_update()
        if update and runtime.stream_writer is not None:
            runtime.stream_writer(
                {
                    "type": "middleware",
                    "middleware": self.name,
                    "event": "inbox_persisted",
                    "count": update["receiveMessagesLastCount"],
                }
            )
        return update

    async def abefore_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        return handler(self._guidance_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        return await handler(self._guidance_request(request))


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
