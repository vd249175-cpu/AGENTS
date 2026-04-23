from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config, get_stream_writer
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import Field

from Deepagents.SeedAgent.Agent.middlewares.base import (
    BaseAgentMiddleware,
    MiddlewareRuningConfig,
)


MIDDLEWARE_DIR = Path(__file__).resolve().parent


class MiddlewareStateTydict(AgentState, total=False):
    agentStepTraceLastPhase: str | None


class AgentStepTraceRuningConfig(MiddlewareRuningConfig):
    eventType: str = Field(default="agent_step_timing")
    emitFullState: bool = Field(default=True)


middleware_runingconfig = AgentStepTraceRuningConfig.load(
    MIDDLEWARE_DIR / "agent_step_trace_config.json"
)
middleware_capability_prompts = []
MiddlewareToolConfig = {"tools": {}, "toolStateTydicts": {}}


class AgentStepTraceMiddleware(BaseAgentMiddleware):
    name = "agent_step_trace"
    state_schema = MiddlewareStateTydict
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

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _metadata(self) -> dict[str, Any]:
        try:
            return dict((get_config().get("metadata", {}) or {}))
        except RuntimeError:
            return {}

    def _agent_name(self) -> str | None:
        metadata = self._metadata()
        return metadata.get("lc_agent_name") or metadata.get("agent_name")

    def _write_event(self, payload: dict[str, Any]) -> None:
        try:
            writer = get_stream_writer()
            identity = self.runtime_identity()
            writer(
                {
                    "type": self.runingConfig.eventType,
                    "middleware": self.name,
                    "agent_name": self._agent_name(),
                    "run_id": identity.runId,
                    "thread_id": identity.threadId,
                    **payload,
                }
            )
        except Exception:
            return

    def _state_payload(self, state: Any) -> Any:
        if not self.runingConfig.emitFullState:
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

    @staticmethod
    def _tool_call_summary(tool_call: dict[str, Any]) -> dict[str, Any]:
        args = tool_call.get("args") if isinstance(tool_call, dict) else None
        return {
            "tool_call_id": tool_call.get("id"),
            "tool_name": tool_call.get("name"),
            "tool_args_keys": list(args.keys()) if isinstance(args, dict) else None,
        }

    @staticmethod
    def _model_request_summary(request: ModelRequest[Any]) -> dict[str, Any]:
        return {
            "message_count": len(request.messages),
            "tool_count": len(request.tools),
            "has_system_message": request.system_message is not None,
            "tool_choice": request.tool_choice,
        }

    @staticmethod
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

    def before_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        self._write_event(
            {
                "phase": "before_agent",
                "event": "start",
                "started_at": self._now(),
                "state": self._state_payload(state),
                "runtime": {
                    "context_type": type(runtime.context).__name__,
                    "has_store": runtime.store is not None,
                },
            }
        )
        return None

    async def abefore_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        self._write_event(
            {
                "phase": "after_agent",
                "event": "end",
                "finished_at": self._now(),
                "state": self._state_payload(state),
                "runtime": {
                    "context_type": type(runtime.context).__name__,
                    "has_store": runtime.store is not None,
                },
            }
        )
        return None

    async def aafter_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        self._write_event(
            {
                "phase": "wrap_model_call",
                "event": "start",
                "started_at": self._now(),
                "request": self._model_request_summary(request),
                "state": self._state_payload(request.state),
            }
        )
        result = handler(request)
        self._write_event(
            {
                "phase": "wrap_model_call",
                "event": "end",
                "finished_at": self._now(),
                "result_type": self._result_type(result),
                "state": self._state_payload(request.state),
            }
        )
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        self._write_event(
            {
                "phase": "wrap_model_call",
                "event": "start",
                "started_at": self._now(),
                "request": self._model_request_summary(request),
                "state": self._state_payload(request.state),
            }
        )
        result = await handler(request)
        self._write_event(
            {
                "phase": "wrap_model_call",
                "event": "end",
                "finished_at": self._now(),
                "result_type": self._result_type(result),
                "state": self._state_payload(request.state),
            }
        )
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        self._write_event(
            {
                "phase": "wrap_tool_call",
                "event": "start",
                "started_at": self._now(),
                "tool_call": self._tool_call_summary(request.tool_call),
                "state": self._state_payload(request.state),
            }
        )
        result = handler(request)
        self._write_event(
            {
                "phase": "wrap_tool_call",
                "event": "end",
                "finished_at": self._now(),
                "result_type": self._result_type(result),
                "state": self._state_payload(request.state),
            }
        )
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        self._write_event(
            {
                "phase": "wrap_tool_call",
                "event": "start",
                "started_at": self._now(),
                "tool_call": self._tool_call_summary(request.tool_call),
                "state": self._state_payload(request.state),
            }
        )
        result = await handler(request)
        self._write_event(
            {
                "phase": "wrap_tool_call",
                "event": "end",
                "finished_at": self._now(),
                "result_type": self._result_type(result),
                "state": self._state_payload(request.state),
            }
        )
        return result

