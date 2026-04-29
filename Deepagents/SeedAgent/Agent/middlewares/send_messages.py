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
            "发送前确认目标 agent name。"
            "如果你要让对方读取、检查、评价、继续处理、入库或对照某个文档/文件/图片/生成物，"
            "必须把该资源放进 attachments，格式为 attachments=[{\"link\":\"/workspace/...\",\"summary\":\"...\"}]。"
            "只在 content 里写 `/workspace/...` 路径不会传输文件，对方不会收到附件。"
            "传本地文件时 attachments.link 必须以 /workspace 开头；网络 URL 可直接传。"
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


def _format_peer_directory(peers: list[str], peer_details: list[dict[str, Any]] | None = None) -> str:
    details = [item for item in peer_details or [] if isinstance(item, dict)]
    if not details:
        peers_text = ", ".join(peers) if peers else "无"
        return (
            f"MainServer 当前在线且可通讯的其他 agent：{peers_text}。"
            "如果需要联系它们，必须调用 send_message_to_agent，并把 dst 设置为对应 agent name；"
            "只在对话里提到它们不会产生通讯。"
        )

    lines = ["MainServer 当前在线且可通讯的其他 agent："]
    for detail in details:
        name = str(detail.get("agent_name") or "").strip()
        if not name:
            continue
        online_text = "online" if detail.get("online") else "offline"
        status = str(detail.get("status") or "unknown")
        phase = str(detail.get("phase") or "").strip()
        status_text = f"{online_text}, status={status}" + (f", phase={phase}" if phase else "")
        card = detail.get("card") if isinstance(detail.get("card"), dict) else {}
        capabilities = card.get("capabilities") if isinstance(card, dict) else []
        capability_parts: list[str] = []
        if isinstance(capabilities, list):
            for capability in capabilities[:4]:
                if not isinstance(capability, dict):
                    continue
                title = str(capability.get("title") or "").strip()
                content = str(capability.get("content") or "").strip()
                if title and content:
                    capability_parts.append(f"{title}: {content}")
                elif title:
                    capability_parts.append(title)
                elif content:
                    capability_parts.append(content)
        capability_text = "；".join(capability_parts) if capability_parts else "未提供 AgentCard 能力描述"
        lines.append(f"- {name} ({status_text})：{capability_text}")
    lines.append(
        "如果需要联系这些 agent，必须调用 send_message_to_agent，并把 dst 设置为对应 agent name；"
        "只在对话里提到它们不会产生通讯。"
    )
    return "\n".join(lines)


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
        peer_details: list[dict[str, Any]] | None = None
        try:
            peer_details = self.comm.peer_directory()
            peers = [
                str(item.get("agent_name") or "").strip()
                for item in peer_details
                if isinstance(item, dict) and str(item.get("agent_name") or "").strip()
            ]
        except Exception:
            try:
                peers = self.comm.peers()
            except Exception:
                peers = []
        if agent_name:
            peers = [peer for peer in peers if str(peer).strip() and peer != agent_name]
        messages = set_named_system_message(
            messages,
            name="middleware.send_messages.peers",
            text=_format_peer_directory(peers, peer_details),
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
