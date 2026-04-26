from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware import AgentState
from pydantic import BaseModel, ConfigDict, Field

from ..server.demo_server import (
    RuntimeIdentity,
    build_configured_children,
    config_from_external,
    current_runtime_identity,
    make_named_system_message,
    remove_named_system_message,
    set_named_system_message,
    upsert_named_system_message,
)


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(default="generic.guidance")
    prompt: str = Field(default="")


class MiddlewareRuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    enabled: bool = Field(default=True)
    defaultRunId: str = Field(default="knowledgeseedagent-run")
    defaultThreadId: str = Field(default="default")

    @classmethod
    def load(cls, source) -> "MiddlewareRuningConfig":
        return config_from_external(cls, source)


class MiddlewareStateTydict(AgentState, total=False):
    lastMiddlewareError: str | None


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
