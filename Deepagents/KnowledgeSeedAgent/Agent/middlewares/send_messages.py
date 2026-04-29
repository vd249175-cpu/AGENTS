"""Single-file send-messages middleware following the demo wrapper pattern."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import AIMessage
from pydantic import Field

from MainServer.comm import AgentComm
from MainServer.state import MessageType
from ..server.demo_server import emit, set_named_system_message
from ..server.path_resolver import WORKSPACE_ROOT
from .base import BaseAgentMiddleware, MiddlewareCapabilityPrompt, MiddlewareRuningConfig
from ..tools.send_message_tool import (
    SendMessageTool,
    ToolRuningConfigSc,
    ToolStateTydict,
)


MIDDLEWARE_DIR = Path(__file__).resolve().parent


class Config(MiddlewareRuningConfig):
    guidancePrompt: str = Field(
        default=(
            "当你需要与其他 Agent 或 MainServer 通讯时，使用 `send_message_to_agent`。"
            "发送普通信息用 msgType=message，发送任务用 msgType=task；两者是同一个 AgentMail 通道。"
            "发送前确认目标 agent name。传本地文件时 attachments.link 必须以 /workspace 开头；网络 URL 可直接传。"
        )
    )
    defaultDestination: str | None = Field(default=None)
    defaultMessageType: MessageType = Field(default="message")
    peekPeersInGuidance: bool = Field(default=True)
    blockSelfTarget: bool = Field(default=True)

    @classmethod
    def load_config_send_messages(cls, source=None):
        return cls.load(source)


SendMessagesRuningConfig = Config


class SubState(AgentState, ToolStateTydict, total=False):
    sendMessagesGuidanceInjected: bool


MiddlewareStateTydict = SubState


class MiddlewareSchema:
    name = "send_messages"
    tools = {"send_message_to_agent": SendMessageTool}
    state_schema = SubState


middleware_runingconfig = Config.load_config_send_messages(MIDDLEWARE_DIR / "send_messages_config.json")
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="middleware.send_messages.guidance",
        prompt=middleware_runingconfig.guidancePrompt,
    )
]


def _current_agent_name(comm: AgentComm | None) -> str | None:
    if comm is None:
        return None
    return str(comm.agent_name or "").strip() or None


class Middleware(BaseAgentMiddleware):
    name = MiddlewareSchema.name
    state_schema = SubState

    def __init__(
        self,
        *,
        comm: AgentComm | None = None,
        runingConfig: SendMessagesRuningConfig = middleware_runingconfig,
    ) -> None:
        self.comm = comm
        tool_config = ToolRuningConfigSc(
            defaultDestination=runingConfig.defaultDestination,
            defaultMessageType=runingConfig.defaultMessageType,
            currentAgentName=comm.agent_name if comm is not None else None,
            blockSelfTarget=runingConfig.blockSelfTarget,
            defaultRunId=runingConfig.defaultRunId,
            defaultThreadId=runingConfig.defaultThreadId,
        )
        send_tool = SendMessageTool(comm=comm, runingConfig=tool_config)
        self.toolConfig = {
            "tools": {"send_message_to_agent": send_tool.tool},
            "toolStateTydicts": {"send_message_to_agent": ToolStateTydict},
        }
        super().__init__(
            runingConfig=runingConfig,
            capabilityPromptConfigs=middleware_capability_prompts,
            tools=list(self.toolConfig["tools"].values()),
        )

    def _with_peers(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        guided = self._with_guidance(request)
        agent_name = _current_agent_name(self.comm)
        messages = set_named_system_message(
            list(guided.messages),
            name="middleware.send_messages.identity",
            text=(
                f"你当前的 agent name 是 {agent_name}。"
                "默认不要给自己发消息；只有用户明确要求时才允许自发自收。"
            )
            if agent_name
            else None,
        )
        if self.comm is None or not self.runingConfig.peekPeersInGuidance:
            return guided.override(messages=messages)
        try:
            peers = self.comm.peers()
        except Exception:
            peers = []
        if agent_name:
            peers = [peer for peer in peers if str(peer).strip() and peer != agent_name]
        peers_text = ", ".join(peers) if peers else "未知或空"
        messages = set_named_system_message(
            messages,
            name="middleware.send_messages.peers",
            text=f"当前可达 peers={peers_text}",
        )
        return guided.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        return handler(self._with_peers(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        return await handler(self._with_peers(request))


class SendMessagesMiddleware:
    name = MiddlewareSchema.name
    config = Config
    substate = SubState
    middlewareschema = MiddlewareSchema

    def __init__(
        self,
        *,
        comm: AgentComm | None = None,
        runingConfig: SendMessagesRuningConfig | None = None,
    ) -> None:
        self.config = runingConfig or middleware_runingconfig
        self.middleware = Middleware(comm=comm, runingConfig=self.config)


send_messages_middleware = SendMessagesMiddleware().middleware
