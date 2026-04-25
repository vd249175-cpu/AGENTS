"""单文件中间件模板。

这个文件的声明顺序固定为：
1. Config
2. SubState
3. AffectedPrompt
4. MiddlewareSchema
5. 本地 helper 函数
6. Middleware 类
7. 包裹类

这里同步补上中间件框架里的关键约束：
- 不要在这个文件中开启 ``from __future__ import annotations``。
- ``before_model`` / ``after_model`` 只从 ``state`` 和 ``runtime`` 取动态值。
- ``wrap_model_call`` 只从 ``request`` 取模型调用相关动态值。
- ``Runtime`` 不直接包含 ``config``，如果要读 ``RunnableConfig``，
  使用 ``langgraph.config.get_config()``。
- 中间件挂工具时，在标准 middleware 类上暴露 ``tools``，创建 agent 时只挂 middleware，
  不要重复把同一批工具传给 ``create_agent(...)``。
- middleware 自己的默认配置在实例化时传递给工具包裹类；
  真实运行时优先从 ``runtime.context`` / ``request.runtime.context`` 读取配置。
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse, ModelRequest
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.prebuilt.tool_node import ToolCallRequest
from pydantic import BaseModel, Field

from demo_nested_inner_tool import Config as SplitTextSegmentsConfig
from demo_nested_inner_tool import SplitTextSegmentsTool
from demo_server import (
    build_configured_children,
    config_from_external,
    emit,
    get_prompt_tag_content,
    get_nested_count,
    update_nested_count,
    update_prompt_tag_content,
    upsert_system_message,
)


class Config(SplitTextSegmentsConfig):
    """中间件运行配置。"""

    maxToolUseCount: int = Field(default=3, ge=1, description="允许模型调用挂载工具的最大次数。")

    @classmethod
    def load_config_demo_split_text_middleware(cls, source=None):
        """从外部配置加载当前中间件可用的 context。"""

        return config_from_external(cls, source)


class SubState(AgentState, SplitTextSegmentsTool.substate, total=False):
    """当前中间件自己负责读写的状态字段。"""

    demoSplitTextMiddlewareStats: dict[str, int]


class AffectedPrompt(BaseModel):
    """中间件会影响到的提示词片段声明。"""

    name: str = Field(description="对应一个 SystemMessage.name 的稳定注册名。")
    prompt: str = Field(
        description=(
            "与该 name 一一对应的提示词模板。"
            "middleware 会根据 state 渲染它，并更新到对应 name 的 message 上。"
        )
    )


class AffectedPrompts(BaseModel):
    """中间件会影响到的提示词片段列表。"""

    Prompts: list[AffectedPrompt]


default_split_text_affected_prompt = AffectedPrompt(
    name="demo_split_text_middleware.split_text",
    prompt=(
        "<SplitCapability>\n"
        "<instruction>当任务需要把一段文本切分成多段时，调用 split_text_segments，不要自己手工模拟切分结果。</instruction>\n"
        "</SplitCapability>"
    ),
)


default_affected_prompts = AffectedPrompts(
    Prompts=[
        default_split_text_affected_prompt,
    ]
)


class MiddlewareSchema:
    """中间件对外声明集中放在这里。"""

    name = "demo_split_text_middleware"
    affectedPrompts = default_affected_prompts
    tools = {
        SplitTextSegmentsTool.name: SplitTextSegmentsTool,
    }


def render_affected_prompt(affected_prompt: AffectedPrompt, state: SubState) -> str:
    """根据当前 state 生成需要写回到 <SplitCapability> 内部的内容。"""

    completed_input_count = get_nested_count(state, "splitTextToolStats", "completedInputCount")
    instruction_content = get_prompt_tag_content(affected_prompt.prompt, "instruction")
    if instruction_content is None:
        instruction_content = affected_prompt.prompt.strip()

    return (
        f"<processedCount>{completed_input_count}</processedCount>\n"
        f"<instruction>{instruction_content}</instruction>"
    )


class Middleware(AgentMiddleware[SubState, Config]):
    """标准 LangChain 中间件入口。"""

    config = Config()
    substate = SubState
    middlewareschema = MiddlewareSchema
    affectedPromptConfigs = MiddlewareSchema.affectedPrompts.Prompts
    state_schema = SubState
    tools = []
    name = MiddlewareSchema.name

    def __init__(self, config: Config | None = None, tools: list | None = None):
        super().__init__()
        self.config = config or Config()
        self.tools = tools or build_configured_children(
            self.config,
            self.middlewareschema.tools,
            "tool",
        )

    def before_model(self, state: SubState, runtime: Runtime[Config]) -> dict[str, Any] | None:
        """模型调用前的状态更新和流式 trace。"""

        before_model_count = (
            get_nested_count(state, "demoSplitTextMiddlewareStats", "beforeModelCount") + 1
        )
        completed_input_count = get_nested_count(state, "splitTextToolStats", "completedInputCount")
        tool_use_count = get_nested_count(state, "demoSplitTextMiddlewareStats", "toolUseCount")
        agent_name = get_config().get("metadata", {}).get("lc_agent_name")
        messages = state.get("messages", [])

        for affected_prompt in self.affectedPromptConfigs:
            updated_prompt_content = render_affected_prompt(affected_prompt, state)
            updated_prompt = update_prompt_tag_content(
                affected_prompt.prompt,
                "SplitCapability",
                updated_prompt_content,
            )
            messages = upsert_system_message(messages, affected_prompt.name, updated_prompt)

        emit(
            runtime.stream_writer,
            {
                "type": "middleware",
                "middleware": self.name,
                "stage": "before_model",
                "beforeModelCount": before_model_count,
                "completedInputCount": completed_input_count,
                "toolUseCount": tool_use_count,
                "affectedPromptCount": len(self.affectedPromptConfigs),
                "agentName": agent_name,
            },
        )

        return {
            **update_nested_count(
                state,
                "demoSplitTextMiddlewareStats",
                "beforeModelCount",
                before_model_count,
            ),
            "messages": messages,
        }

    def wrap_model_call(self, request: ModelRequest[Config], handler) -> ExtendedModelResponse:
        """记录 wrap_model_call 阶段的状态。"""

        wrap_model_call_count = (
            get_nested_count(request.state, "demoSplitTextMiddlewareStats", "wrapModelCallCount") + 1
        )
        agent_name = get_config().get("metadata", {}).get("lc_agent_name")
        writer = request.runtime.stream_writer

        emit(
            writer,
            {
                "type": "middleware",
                "middleware": self.name,
                "stage": "wrap_model_call",
                "wrapModelCallCount": wrap_model_call_count,
                "toolCount": len(request.tools or []),
                "agentName": agent_name,
            },
        )

        response = handler(request)

        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    **update_nested_count(
                        request.state,
                        "demoSplitTextMiddlewareStats",
                        "wrapModelCallCount",
                        wrap_model_call_count,
                    ),
                }
            ),
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """包装工具调用，累计次数并在达到上限时短路。"""

        tool_use_count = (
            get_nested_count(request.state, "demoSplitTextMiddlewareStats", "toolUseCount") + 1
        )
        tool_name = request.tool_call["name"]
        tool_call_id = request.tool_call["id"]
        writer = request.runtime.stream_writer
        current_config = request.runtime.context or self.config

        emit(
            writer,
            {
                "type": "middleware",
                "middleware": self.name,
                "stage": "wrap_tool_call",
                "toolName": tool_name,
                "toolUseCount": tool_use_count,
                "maxToolUseCount": current_config.maxToolUseCount,
            },
        )

        if tool_use_count > current_config.maxToolUseCount:
            error_text = f"工具调用次数已达到上限 {current_config.maxToolUseCount}，本次调用被拒绝。"
            emit(
                writer,
                {
                    "type": "middleware",
                    "middleware": self.name,
                    "stage": "wrap_tool_call",
                    "event": "error",
                    "toolName": tool_name,
                    "toolUseCount": tool_use_count,
                    "error": error_text,
                },
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=error_text, tool_call_id=tool_call_id, status="error"),
                    ],
                    **update_nested_count(
                        request.state,
                        "demoSplitTextMiddlewareStats",
                        "toolUseCount",
                        tool_use_count,
                    ),
                }
            )

        result = handler(request)

        if isinstance(result, Command):
            update = result.update if isinstance(result.update, dict) else {"result": result.update}
            return Command(
                graph=result.graph,
                update={
                    **update,
                    **update_nested_count(
                        request.state,
                        "demoSplitTextMiddlewareStats",
                        "toolUseCount",
                        tool_use_count,
                    ),
                },
                resume=result.resume,
                goto=result.goto,
            )

        return Command(
            update={
                "messages": [result],
                **update_nested_count(
                    request.state,
                    "demoSplitTextMiddlewareStats",
                    "toolUseCount",
                    tool_use_count,
                ),
            }
        )


class DemoSplitTextMiddleware:
    """当前中间件的外层包裹声明。"""

    name = MiddlewareSchema.name
    config = Config
    substate = SubState
    middlewareschema = MiddlewareSchema
    middlewareclass = Middleware

    def __init__(self, config: Config | None = None):
        self.config = config or self.config()
        self.middleware = self.create_middleware()

    def create_middleware(self):
        """根据当前实例配置构造标准 LangChain middleware 实例。"""

        current_config = self.config
        configured_middleware_cls = type(
            f"{self.middlewareclass.__name__}Configured",
            (self.middlewareclass,),
            {
                "substate": self.substate,
                "middlewareschema": self.middlewareschema,
                "affectedPromptConfigs": self.middlewareschema.affectedPrompts.Prompts,
                "state_schema": self.substate,
                "name": self.middlewareschema.name,
            },
        )
        return configured_middleware_cls(
            config=current_config,
            tools=build_configured_children(current_config, self.middlewareschema.tools, "tool"),
        )


demo_split_text_middleware = DemoSplitTextMiddleware
