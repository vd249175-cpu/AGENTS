"""Memory middleware declaration."""

import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from server.memory_state import MemoryToolStateTydict, render_memory_block
from server.memory_service import MemoryCapabilityPreset, MemoryService
from tools.manage_memory import (
    ManageMemoryToolConfig,
    MemoryStateTydict as ManageMemoryToolStateTydict,
    ToolConfig as ManageMemoryToolConfigBundle,
    build_manage_memory_tool,
)


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("memory.json")

DEFAULT_MEMORY_SYSTEM_PROMPT = """<SystemPrompt>
<CapabilityGuidance name="memory">
使用 manage_memory 维护持久的用户偏好、稳定事实和重要修正。
只保留有长期价值的记忆。
每轮最多调用一次 manage_memory。
回答前先阅读 <Memory> 区块。
把记忆区块视为当前对话线程的权威会话记忆。
</CapabilityGuidance>
</SystemPrompt>"""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class MemoryCapabilityPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


middleware_capability_prompts = [
    MemoryCapabilityPrompt(name="memory.guidance", prompt=DEFAULT_MEMORY_SYSTEM_PROMPT),
    MemoryCapabilityPrompt(name="memory.state", prompt=render_memory_block([])),
]


class MemoryCapabilityPreset(BaseModel):
    max_items: int = Field(default=32, ge=1, description="记忆条目允许的最大数量。")
    max_total_chars: int = Field(default=4096, ge=1, description="记忆条目允许的最大字符总数。")


class MemoryAgentConfig(BaseModel):
    memory: MemoryCapabilityPreset = Field(default_factory=MemoryCapabilityPreset, description="memory 能力的运行配置。")
    model: object | None = Field(default=None, description="预留的模型对象。")
    base_url: str | None = Field(default=None, description="预留的模型 base_url。")
    temperature: float | None = Field(default=0.0, description="预留的模型温度参数。")
    debug: bool = Field(default=False, description="是否开启调试。")
    checkpointer: object | None = Field(default=None, description="预留的 checkpoint 存储。")
    store: object | None = Field(default=None, description="预留的共享存储。")


class MemoryMiddlewareConfig(BaseModel):
    max_items: int = Field(default=32, ge=1, description="记忆条目允许的最大数量。")
    max_total_chars: int = Field(default=4096, ge=1, description="记忆条目允许的最大字符总数。")

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "MemoryMiddlewareConfig":
        payload = _load_json(path)
        return cls(
            max_items=max(1, int(payload.get("max_items", 32))),
            max_total_chars=max(1, int(payload.get("max_total_chars", 4096))),
        )


class MemoryStateTydict(AgentState, MemoryToolStateTydict, total=False):
    monitor_trace: list[str]


middleware_runingconfig = MemoryMiddlewareConfig.load()
MiddlewareToolConfig = ManageMemoryToolConfigBundle


class MemoryCapabilityMiddleware(AgentMiddleware):
    capability_name = "memory"
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = middleware_runingconfig
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = MemoryStateTydict  # type: ignore[assignment]

    def __init__(
        self,
        *,
        config: MemoryMiddlewareConfig | None = None,
        preset: MemoryCapabilityPreset | None = None,
    ) -> None:
        super().__init__()
        loaded = config or MemoryMiddlewareConfig.load()
        active_preset = preset or MemoryCapabilityPreset(max_items=loaded.max_items, max_total_chars=loaded.max_total_chars)
        self._service = MemoryService(preset=active_preset)
        self.config = loaded
        self.preset = active_preset
        self.system_prompt = DEFAULT_MEMORY_SYSTEM_PROMPT
        self.capability_prompts = self.capabilityPromptConfigs
        self.capability_prompt_configs = self.capabilityPromptConfigs
        self.middleware_runingconfig = loaded
        self.runing_config = loaded
        runtime_tool_config = {
            "tools": {
                "manage_memory": build_manage_memory_tool(
                    config=ManageMemoryToolConfig(
                        max_items=active_preset.max_items,
                        max_total_chars=active_preset.max_total_chars,
                    )
                ),
            },
            "toolStateTydicts": {
                "manage_memory": ManageMemoryToolStateTydict,
            },
        }
        self.toolConfig = runtime_tool_config
        self.toolStateTydicts = runtime_tool_config["toolStateTydicts"]
        self.tools = list(runtime_tool_config["tools"].values())

    @staticmethod
    def _trace(message: str) -> None:
        if os.getenv("RESTRUCTURE_TRACE_MEMORY") == "1":
            print(f"[MEMORY_TRACE] {message}", flush=True)

    def before_agent(self, state: MemoryStateTydict, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._trace(f"before_agent state_keys={sorted(state.keys())}")
        return self._service.normalize_state(state)

    def _upsert_named_system_message(self, messages: list[Any], *, name: str, text: str) -> list[Any]:
        new_message = SystemMessage(name=name, content=text)
        replaced = False
        new_messages = []
        for message in messages:
            if isinstance(message, SystemMessage) and message.name == name:
                new_messages.append(new_message)
                replaced = True
            else:
                new_messages.append(message)
        if not replaced:
            new_messages.insert(0, new_message)
        return new_messages

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler,
    ) -> ModelResponse[Any] | AIMessage:
        self._trace("wrap_model_call enter")
        memories = list(cast(MemoryStateTydict, request.state).get("memory") or [])
        messages = self._upsert_named_system_message(
            list(request.messages),
            name="memory.guidance",
            text=self.system_prompt,
        )
        messages = self._upsert_named_system_message(
            messages,
            name="memory.state",
            text=render_memory_block(memories),
        )
        result = handler(request.override(messages=messages))
        self._trace(
            f"wrap_model_call exit result_type={type(result).__name__} "
            f"tool_calls={len(getattr(result, 'tool_calls', []) or [])}"
        )
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        self._trace("awrap_model_call enter")
        memories = list(cast(MemoryStateTydict, request.state).get("memory") or [])
        messages = self._upsert_named_system_message(
            list(request.messages),
            name="memory.guidance",
            text=self.system_prompt,
        )
        messages = self._upsert_named_system_message(
            messages,
            name="memory.state",
            text=render_memory_block(memories),
        )
        result = await handler(request.override(messages=messages))
        self._trace(
            f"awrap_model_call exit result_type={type(result).__name__} "
            f"tool_calls={len(getattr(result, 'tool_calls', []) or [])}"
        )
        return result

    def after_model(self, state: MemoryStateTydict, runtime: Runtime[Any]) -> dict[str, Any] | None:
        self._trace(f"after_model state_keys={sorted(state.keys())}")
        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            self._trace("after_model no tool_calls")
            return None

        memory_calls = [tool_call for tool_call in last_ai_msg.tool_calls if tool_call["name"] == "manage_memory"]
        self._trace(f"after_model manage_memory_calls={len(memory_calls)} total_tool_calls={len(last_ai_msg.tool_calls)}")
        if len(memory_calls) <= 1:
            return None

        error_messages = [
            ToolMessage(
                content=(
                    "错误：`manage_memory` 工具在每个模型回合中最多只能调用一次。"
                    "请只提交一次记忆更新后重试。"
                ),
                tool_call_id=tool_call["id"],
                status="error",
            )
            for tool_call in memory_calls
        ]
        return {"messages": error_messages}

    def render_state_prompt(self, state: Mapping[str, Any] | None = None) -> str:
        return render_memory_block((state or {}).get("memory"))
