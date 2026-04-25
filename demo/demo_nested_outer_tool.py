"""把 demo agent 封装成标准工具。

这个文件演示最外层嵌套：
outer tool -> nested agent -> middleware -> inner tool
"""

from typing import Any

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from demo_nested_agent import Config as DemoNestedAgentConfig
from demo_nested_agent import DemoNestedAgent
from demo_server import config_from_external, emit, get_nested_count, update_nested_count


class Config(DemoNestedAgentConfig):
    """外层工具运行配置。"""

    @classmethod
    def load_config_run_demo_nested_agent_tool(cls, source=None):
        """从外部配置加载当前外层工具可用的 context。"""

        return config_from_external(cls, source)


class SubState(AgentState, DemoNestedAgent.substate, total=False):
    """当前外层工具自己负责写回的状态字段。"""

    nestedAgentToolStats: dict[str, int]


class Input(BaseModel):
    """暴露给模型的外层工具入参。"""

    task: str = Field(description="交给内层 agent 完成的任务。")


class ToolFeedback(BaseModel):
    """外层工具执行过程中的反馈文案。"""

    successText: str = Field(default="内层 agent 已完成任务。")
    failureText: str = Field(default="内层 agent 执行失败：{error}")


class ToolSchema:
    """外层工具对外声明集中放在这里。"""

    name = "run_demo_nested_agent"
    args_schema = Input
    description = "调用 demo_nested_agent 完成一个任务。"
    toolfeedback = ToolFeedback


def summarize_inner_stream_chunk(chunk: dict[str, Any] | tuple) -> dict[str, Any]:
    """把内层 agent stream chunk 压缩成适合外层 tool 转发的结构。"""

    if isinstance(chunk, tuple) and len(chunk) == 2:
        stream_type, payload = chunk
    else:
        stream_type = chunk.get("type")
        payload = chunk.get("data", chunk)

    if stream_type == "custom":
        return {"stream": "custom", "data": payload}

    if stream_type == "updates" and isinstance(payload, dict):
        return {
            "stream": "updates",
            "nodes": {
                node_name: sorted(update.keys())
                for node_name, update in payload.items()
                if isinstance(update, dict)
            },
        }

    return {"stream": stream_type}


class DemoNestedAgentTool:
    """将 demo agent 包成工具的外层包裹声明。"""

    name = ToolSchema.name
    config = Config
    substate = SubState
    toolschema = ToolSchema

    def __init__(self, config: Config | None = None):
        self.config = config or self.config()
        self.tool = self.create_tool()

    def create_tool(self):
        """根据当前实例配置构造标准 LangChain tool。"""

        current_config = self.config
        current_toolschema = self.toolschema
        current_feedback_cls = current_toolschema.toolfeedback

        @tool(
            current_toolschema.name,
            args_schema=current_toolschema.args_schema,
            description=current_toolschema.description,
        )
        def run_demo_nested_agent(runtime: ToolRuntime[Config, SubState], task: str) -> Command:
            """标准 LangChain 工具入口。"""

            feedback = current_feedback_cls()
            writer = runtime.stream_writer
            completed_task_count = (
                get_nested_count(runtime.state, "nestedAgentToolStats", "completedTaskCount") + 1
            )

            emit(
                writer,
                {
                    "type": "tool",
                    "tool": current_toolschema.name,
                    "event": "start",
                    "completedTaskCount": completed_task_count,
                },
            )

            try:
                context = runtime.context or current_config
                agent = DemoNestedAgent(context).agent
                final_preview = ""
                for chunk in agent.stream(
                    {"messages": [HumanMessage(content=task)]},
                    context=context,
                    stream_mode=["custom", "updates"],
                    version="v2",
                ):
                    rendered = summarize_inner_stream_chunk(chunk)
                    emit(
                        writer,
                        {
                            "type": "nested_agent",
                            "tool": current_toolschema.name,
                            **rendered,
                        },
                    )
                    if rendered.get("stream") == "updates":
                        final_preview = str(rendered)
            except Exception as exc:
                error_text = feedback.failureText.format(error=str(exc))
                emit(
                    writer,
                    {
                        "type": "tool",
                        "tool": current_toolschema.name,
                        "event": "error",
                        "error": error_text,
                    },
                )
                return Command(
                    update={
                        "messages": [ToolMessage(content=error_text, tool_call_id=runtime.tool_call_id)],
                    }
                )

            message_text = feedback.successText
            if final_preview:
                message_text = f"{message_text}\n{final_preview}"
            emit(
                writer,
                {
                    "type": "tool",
                    "tool": current_toolschema.name,
                    "event": "success",
                    "completedTaskCount": completed_task_count,
                },
            )
            return Command(
                update={
                    "messages": [ToolMessage(content=message_text, tool_call_id=runtime.tool_call_id)],
                    **update_nested_count(
                        runtime.state,
                        "nestedAgentToolStats",
                        "completedTaskCount",
                        completed_task_count,
                    ),
                }
            )

        return run_demo_nested_agent


run_demo_nested_agent = DemoNestedAgentTool().tool
