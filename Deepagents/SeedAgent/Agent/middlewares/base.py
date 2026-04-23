import json
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware import AgentState
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.config import get_config
from pydantic import BaseModel, ConfigDict, Field


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(default="generic.guidance")
    prompt: str = Field(default="")


class MiddlewareRuningConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=True)
    defaultRunId: str = Field(default="seedagent-run")
    defaultThreadId: str = Field(default="default")

    @classmethod
    def load(cls, path: Path) -> "MiddlewareRuningConfig":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Middleware config must be a JSON object: {path}")
        return cls.model_validate(data)


class MiddlewareStateTydict(AgentState, total=False):
    lastMiddlewareError: str | None


class RuntimeIdentity(BaseModel):
    runId: str
    threadId: str


def safe_current_config() -> dict[str, Any]:
    try:
        return get_config()
    except RuntimeError:
        return {}


def current_runtime_identity(
    *,
    defaultRunId: str,
    defaultThreadId: str,
) -> RuntimeIdentity:
    config = safe_current_config()
    configurable = config.get("configurable", {}) or {}
    run_id = str(
        configurable.get("run_id")
        or configurable.get("runId")
        or config.get("run_id")
        or defaultRunId
    ).strip() or defaultRunId
    thread_id = str(
        configurable.get("thread_id")
        or configurable.get("threadId")
        or config.get("thread_id")
        or defaultThreadId
    ).strip() or defaultThreadId
    return RuntimeIdentity(runId=run_id, threadId=thread_id)


def make_named_system_message(name: str, text: str) -> SystemMessage:
    slot_name = name.strip()
    if not slot_name:
        raise ValueError("SystemMessage name must not be empty")
    return SystemMessage(content=text, name=slot_name)


def remove_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
) -> list[AnyMessage]:
    output: list[AnyMessage] = []
    for message in messages or []:
        if isinstance(message, SystemMessage) and message.name == name:
            continue
        output.append(message)
    return output


def upsert_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
    text: str,
) -> list[AnyMessage]:
    replacement = make_named_system_message(name, text)
    output: list[AnyMessage] = []
    replaced = False
    inserted = False

    for message in messages or []:
        if isinstance(message, SystemMessage):
            if message.name == replacement.name:
                if not replaced:
                    output.append(replacement)
                    replaced = True
                continue
            output.append(message)
            continue

        if not inserted:
            if not replaced:
                output.append(replacement)
                replaced = True
            inserted = True
        output.append(message)

    if not replaced:
        output.append(replacement)
    return output


def set_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
    text: str | None,
) -> list[AnyMessage]:
    cleaned = (text or "").strip()
    if not cleaned:
        return remove_named_system_message(messages, name=name)
    return upsert_named_system_message(messages, name=name, text=cleaned)


class BaseAgentMiddleware(AgentMiddleware):
    name: str = "base_middleware"
    promptSlotPrefix: str = "middleware"
    state_schema = MiddlewareStateTydict
    tools: list[Any] = []

    def __init__(
        self,
        *,
        runingConfig: MiddlewareRuningConfig,
        capabilityPromptConfigs: list[MiddlewareCapabilityPrompt] | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        super().__init__()
        self.runingConfig = runingConfig
        self.capabilityPromptConfigs = capabilityPromptConfigs or []
        self.tools = tools or []

    def runtime_identity(self) -> RuntimeIdentity:
        return current_runtime_identity(
            defaultRunId=self.runingConfig.defaultRunId,
            defaultThreadId=self.runingConfig.defaultThreadId,
        )

    def guidance_prompt_slot(self, prompt: MiddlewareCapabilityPrompt) -> str:
        return prompt.name or f"{self.promptSlotPrefix}.{self.name}.guidance"

    def _with_guidance(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        messages = list(request.messages)
        if not self.runingConfig.enabled:
            for prompt in self.capabilityPromptConfigs:
                messages = remove_named_system_message(
                    messages,
                    name=self.guidance_prompt_slot(prompt),
                )
            return request.override(messages=messages)

        for prompt in self.capabilityPromptConfigs:
            messages = set_named_system_message(
                messages,
                name=self.guidance_prompt_slot(prompt),
                text=prompt.prompt,
            )
        return request.override(messages=messages)

