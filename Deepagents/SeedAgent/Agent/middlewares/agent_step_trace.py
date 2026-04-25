"""Single-file agent-step trace middleware following the demo wrapper pattern."""

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
    eventType: str = Field(default="agent_step_timing")
    emitFullState: bool = Field(default=True)

    @classmethod
    def load_config_agent_step_trace(cls, source=None):
        return config_from_external(cls, source)


AgentStepTraceRuningConfig = Config


class SubState(AgentState, total=False):
    agentStepTraceLastPhase: str | None


MiddlewareStateTydict = SubState


class MiddlewareSchema:
    name = "agent_step_trace"
    tools = {}
    state_schema = SubState


middleware_runingconfig = Config.load_config_agent_step_trace(
    MIDDLEWARE_DIR / "agent_step_trace_config.json"
)
middleware_capability_prompts: list[Any] = []
MiddlewareToolConfig = {"tools": {}, "toolStateTydicts": {}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_payload(state: Any, emit_full_state: bool) -> Any:
    if not emit_full_state:
        if isinstance(state, dict):
            return {"keys": sorted(state.keys())}
        return {"type": type(state).__name__}
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        try:
            return state.model_dump()
        except Exception:
            return {"type": type(state).__name__}
    return state


def _tool_call_summary(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = tool_call.get("args") if isinstance(tool_call, dict) else None
    return {
        "tool_call_id": tool_call.get("id"),
        "tool_name": tool_call.get("name"),
        "tool_args_keys": list(args.keys()) if isinstance(args, dict) else None,
    }


def _model_request_summary(request: ModelRequest[Any]) -> dict[str, Any]:
    return {
        "message_count": len(request.messages),
        "tool_count": len(request.tools),
        "has_system_message": request.system_message is not None,
        "tool_choice": request.tool_choice,
    }


def _result_type(result: Any) -> str:
    if isinstance(result, ExtendedModelResponse):
        return "ExtendedModelResponse"
    if isinstance(result, ModelResponse):
        return "ModelResponse"
    if isinstance(result, AIMessage):
        return "AIMessage"
    if isinstance(result, ToolMessage):
        return "ToolMessage"
    if isinstance(result, Command):
        return "Command"
    return type(result).__name__


class Middleware(BaseAgentMiddleware):
    name = MiddlewareSchema.name
    state_schema = SubState
    tools = []

    def __init__(
        self,
        *,
        runingConfig: AgentStepTraceRuningConfig = middleware_runingconfig,
    ) -> None:
        super().__init__(
            runingConfig=runingConfig,
            capabilityPromptConfigs=middleware_capability_prompts,
            tools=[],
        )

    def _emit(self, writer: Any, payload: dict[str, Any]) -> None:
        emit(
            writer,
            {
                "type": self.runingConfig.eventType,
                "middleware": self.name,
                **payload,
            },
        )

    def before_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._emit(
            runtime.stream_writer,
            {
                "phase": "before_agent",
                "event": "start",
                "started_at": _now(),
                "state": _state_payload(state, self.runingConfig.emitFullState),
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
                "state": _state_payload(state, self.runingConfig.emitFullState),
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
                "request": _model_request_summary(request),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
            },
        )
        result = handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_model_call",
                "event": "end",
                "finished_at": _now(),
                "result_type": _result_type(result),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
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
                "request": _model_request_summary(request),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
            },
        )
        result = await handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_model_call",
                "event": "end",
                "finished_at": _now(),
                "result_type": _result_type(result),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
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
                "tool_call": _tool_call_summary(request.tool_call),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
            },
        )
        result = handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_tool_call",
                "event": "end",
                "finished_at": _now(),
                "result_type": _result_type(result),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
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
                "tool_call": _tool_call_summary(request.tool_call),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
            },
        )
        result = await handler(request)
        self._emit(
            writer,
            {
                "phase": "wrap_tool_call",
                "event": "end",
                "finished_at": _now(),
                "result_type": _result_type(result),
                "state": _state_payload(request.state, self.runingConfig.emitFullState),
            },
        )
        return result


class AgentStepTraceMiddleware:
    name = MiddlewareSchema.name
    config = Config
    substate = SubState
    middlewareschema = MiddlewareSchema
    tools = []

    def __init__(self, runingConfig: AgentStepTraceRuningConfig | None = None) -> None:
        self.config = runingConfig or middleware_runingconfig
        self.middleware = Middleware(runingConfig=self.config)


agent_step_trace_middleware = AgentStepTraceMiddleware().middleware
