from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import AIMessage
from pydantic import Field

from MainServer.comm import AgentComm
from MainServer.state import MessageType
from Deepagents.SeedAgent.Agent.middlewares.base import (
    BaseAgentMiddleware,
    MiddlewareCapabilityPrompt,
    MiddlewareRuningConfig,
    set_named_system_message,
)
from Deepagents.SeedAgent.Agent.tools.send_message_tool import (
    SendMessageTool,
    ToolRuningConfigSc,
    ToolStateTydict,
)


MIDDLEWARE_DIR = Path(__file__).resolve().parent


class MiddlewareStateTydict(AgentState, ToolStateTydict, total=False):
    sendMessagesGuidanceInjected: bool


class SendMessagesRuningConfig(MiddlewareRuningConfig):
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


middleware_runingconfig = SendMessagesRuningConfig.load(
    MIDDLEWARE_DIR / "send_messages_config.json"
)
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="middleware.send_messages.guidance",
        prompt=middleware_runingconfig.guidancePrompt,
    )
]
MiddlewareToolConfig = {"tools": {}, "toolStateTydicts": {"send_message_to_agent": ToolStateTydict}}


class SendMessagesMiddleware(BaseAgentMiddleware):
    name = "send_messages"
    state_schema = MiddlewareStateTydict

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
        if self.comm is None or not self.runingConfig.peekPeersInGuidance:
            return guided
        try:
            peers = self.comm.peers()
        except Exception:
            peers = []
        peers_text = ", ".join(peers) if peers else "未知或空"
        messages = set_named_system_message(
            list(guided.messages),
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
