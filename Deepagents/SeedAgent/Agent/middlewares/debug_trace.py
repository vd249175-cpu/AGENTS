"""Single-file debug trace middleware following the demo wrapper pattern."""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import Field

from ..server.demo_server import StrictConfig, config_from_external, emit
from .base import BaseAgentMiddleware, MiddlewareRuningConfig


MIDDLEWARE_DIR = Path(__file__).resolve().parent


class Config(MiddlewareRuningConfig):
    eventType: str = Field(default="agent_debug_trace")
    truncateLimit: int = Field(default=220, ge=20)

    @classmethod
    def load_config_debug_trace(cls, source=None):
        return config_from_external(cls, source)


DebugTraceRuningConfig = Config


class SubState(AgentState, total=False):
    debugTraceLastPhase: str | None


MiddlewareStateTydict = SubState


class MiddlewareSchema:
    name = "debug_trace"
    tools = {}
    state_schema = SubState


middleware_runingconfig = Config.load_config_debug_trace(MIDDLEWARE_DIR / "debug_trace_config.json")
middleware_capability_prompts: list[Any] = []
MiddlewareToolConfig = {"tools": {}, "toolStateTydicts": {}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Middleware(BaseAgentMiddleware):
    name = MiddlewareSchema.name
    state_schema = SubState
    tools = []

    def __init__(
        self,
        *,
        runingConfig: DebugTraceRuningConfig = middleware_runingconfig,
    ) -> None:
        super().__init__(
            runingConfig=runingConfig,
            capabilityPromptConfigs=middleware_capability_prompts,
            tools=[],
        )

    def _truncate(self, value: Any) -> str:
        text = " ".join(str(value).split())
        if len(text) <= self.runingConfig.truncateLimit:
            return text
        return text[: self.runingConfig.truncateLimit] + "..."

    def _emit(self, writer: Any, payload: dict[str, Any]) -> None:
        emit(
            writer,
            {
                "type": self.runingConfig.eventType,
                "middleware": self.name,
                **payload,
            },
        )

    def _message_summary(self, message: Any) -> dict[str, Any]:
        role = getattr(message, "type", None)
        if not isinstance(role, str) or not role:
            role = message.__class__.__name__.replace("Message", "").lower() or "message"
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        return {
            "role": role.lower(),
            "content_preview": self._truncate(content),
            "tool_calls": [call.get("name") for call in tool_calls] if isinstance(tool_calls, list) else None,
        }

    def _state_summary(self, state: Any) -> dict[str, Any]:
        if isinstance(state, dict):
            summary = {"type": "dict", "keys": sorted(state.keys())}
            if "messages" in state and isinstance(state["messages"], list):
                summary["message_count"] = len(state["messages"])
            return summary
        return {"type": type(state).__name__, "repr": self._truncate(state)}

    def _request_summary(self, request: ModelRequest[Any]) -> dict[str, Any]:
        messages = list(request.messages)
        return {
            "message_count": len(messages),
            "tool_count": len(request.tools),
            "has_system_message": request.system_message is not None,
            "tool_choice": request.tool_choice,
            "message_tail": [self._message_summary(message) for message in messages[-4:]],
            "system_message_preview": self._truncate(
                getattr(request.system_message, "content", "") if request.system_message else ""
            ),
        }

    def _tool_call_summary(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        args = tool_call.get("args") if isinstance(tool_call, dict) else None
        return {
            "tool_call_id": tool_call.get("id"),
            "tool_name": tool_call.get("name"),
            "tool_args_preview": self._truncate(args if args is not None else ""),
        }

    def _result_summary(self, result: Any) -> dict[str, Any]:
        if isinstance(result, AIMessage):
            return {
                "type": "AIMessage",
                "content_preview": self._truncate(result.content),
                "tool_calls": [call.get("name") for call in getattr(result, "tool_calls", [])],
            }
        if isinstance(result, ToolMessage):
            return {"type": "ToolMessage", "content_preview": self._truncate(result.content)}
        if isinstance(result, ExtendedModelResponse):
            return {"type": "ExtendedModelResponse", "repr": self._truncate(result)}
        if isinstance(result, ModelResponse):
            return {"type": "ModelResponse", "repr": self._truncate(result)}
        if isinstance(result, Command):
            return {"type": "Command", "repr": self._truncate(result)}
        return {"type": type(result).__name__, "repr": self._truncate(result)}

    def before_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._emit(
            runtime.stream_writer,
            {
                "phase": "before_agent",
                "event": "start",
                "started_at": _now(),
                "state": self._state_summary(state),
                "runtime": {
                    "context_type": type(runtime.context).__name__,
                    "has_store": runtime.store is not None,
                },
            },
        )
        return None

    async def abefore_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def after_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._emit(
            runtime.stream_writer,
            {
                "phase": "after_agent",
                "event": "end",
                "finished_at": _now(),
                "state": self._state_summary(state),
                "runtime": {
                    "context_type": type(runtime.context).__name__,
                    "has_store": runtime.store is not None,
                },
            },
        )
        return None

    async def aafter_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        self._emit(
            writer,
            {
                "phase": "wrap_model_call",
                "event": "start",
                "started_at": _now(),
                "request": self._request_summary(request),
                "state": self._state_summary(request.state),
            },
        )
        result = handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_model_call",
                "event": "end",
                "finished_at": _now(),
                "result": self._result_summary(result),
                "state": self._state_summary(request.state),
            },
        )
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        self._emit(
            writer,
            {
                "phase": "wrap_model_call",
                "event": "start",
                "started_at": _now(),
                "request": self._request_summary(request),
                "state": self._state_summary(request.state),
            },
        )
        result = await handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_model_call",
                "event": "end",
                "finished_at": _now(),
                "result": self._result_summary(result),
                "state": self._state_summary(request.state),
            },
        )
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        self._emit(
            writer,
            {
                "phase": "wrap_tool_call",
                "event": "start",
                "started_at": _now(),
                "tool_call": self._tool_call_summary(request.tool_call),
                "state": self._state_summary(request.state),
            },
        )
        result = handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_tool_call",
                "event": "end",
                "finished_at": _now(),
                "result": self._result_summary(result),
                "state": self._state_summary(request.state),
            },
        )
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        writer = getattr(getattr(request, "runtime", None), "stream_writer", None)
        self._emit(
            writer,
            {
                "phase": "wrap_tool_call",
                "event": "start",
                "started_at": _now(),
                "tool_call": self._tool_call_summary(request.tool_call),
                "state": self._state_summary(request.state),
            },
        )
        result = await handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_tool_call",
                "event": "end",
                "finished_at": _now(),
                "result": self._result_summary(result),
                "state": self._state_summary(request.state),
            },
        )
        return result


class DebugTraceMiddleware:
    name = MiddlewareSchema.name
    config = Config
    substate = SubState
    middlewareschema = MiddlewareSchema
    tools = []

    def __init__(self, runingConfig: DebugTraceRuningConfig | None = None) -> None:
        self.config = runingConfig or middleware_runingconfig
        self.middleware = Middleware(runingConfig=self.config)


debug_trace_middleware = DebugTraceMiddleware().middleware
