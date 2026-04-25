"""单文件工具模板。

这个文件的声明顺序固定为：
1. Config
2. SubState
3. Input
4. ToolFeedback
5. ToolSchema
6. 本地 helper 函数
7. 包裹类
8. 默认工具实例

这里同步补上工具框架里的关键约束：
- 不要在这个文件中开启 ``from __future__ import annotations``。
- 工具签名里保留 ``runtime: ToolRuntime`` 的真实类型。
- 运行时动态值只从 ``runtime.state``、``runtime.stream_writer``、
  ``runtime.tool_call_id`` 读取。
- 当前版本的 ``ToolRuntime`` 还可以拿到：
  ``runtime.config``、``runtime.context``、``runtime.store``、
  ``runtime.execution_info``、``runtime.server_info``。
- 始终返回 ``Command(update=...)``，并在 ``update["messages"]`` 中
  放入 ``ToolMessage``。
- 始终输出结构化的 start / success / error 事件，便于真实模型测试时
  流式观察工具执行过程。
- ``@tool`` 返回的对象保持标准 LangChain tool，不再往它上面硬塞自定义属性。
- 配置、state 声明和 schema 声明放在外层包裹类上。
- 工具构建时会捕获默认配置；真实运行时优先读取 ``runtime.context``，
  这样上层 agent 透传的 context 可以覆盖默认配置。
"""

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from demo_server import (
    StrictConfig,
    config_from_external,
    emit,
    get_nested_count,
    update_nested_count,
)


class Config(StrictConfig):
    """可被外部参数覆盖的运行配置。"""

    # 每个配置项都应该带默认值，这样没有外部 json 时工具仍然可运行。
    uppercase: bool = Field(default=False, description="是否把返回片段转成大写。")

    @classmethod
    def load_config_split_text_segments_tool(cls, source=None):
        """从外部配置加载当前工具可用的 context。"""

        return config_from_external(cls, source)


class SubState(AgentState, total=False):
    """当前工具自己负责写回的状态字段。"""

    # 这里只声明本工具会通过 Command(update=...) 回写的字段。
    splitTextToolStats: dict[str, int]


class Input(BaseModel):
    """暴露给模型的工具入参。"""

    # 把这个模板改造成新工具时，优先替换这里的字段。
    text: str = Field(description="要切分的文本。")
    splitCount: int = Field(description="要切分成多少段。", ge=1)


class ToolFeedback(BaseModel):
    """工具执行过程中使用的反馈文案。"""

    # 成功 / 失败文案集中放在这里，避免散落在工具函数内部。
    successText: str = Field(default="已将文本切分为 {count} 段。")
    failureText: str = Field(default="文本切分失败：{error}")


class ToolSchema:
    """工具对外声明集中放在这里。"""

    # 外层通过包裹类的 ``toolschema`` 读取这组声明。
    name = "split_text_segments"
    args_schema = Input
    description = "把输入文本按要求切分成多段。"
    toolfeedback = ToolFeedback


def split_text(text: str, split_count: int, uppercase: bool = False) -> list[str]:
    """供工具主体调用的纯 helper 函数。"""

    # 把这个模板改造成新工具时，通常替换这里的业务逻辑。
    content = text.strip()
    if not content:
        return []

    if uppercase:
        content = content.upper()

    segment_count = min(split_count, len(content))
    segments = []
    for index in range(segment_count):
        start = round(index * len(content) / segment_count)
        end = round((index + 1) * len(content) / segment_count)
        segments.append(content[start:end])
    return segments


class SplitTextSegmentsTool:
    """当前工具的外层包裹声明。"""

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
        def split_text_segments(runtime: ToolRuntime[Config, SubState], text: str, splitCount: int) -> Command:
            """标准 LangChain 工具入口。"""

            feedback = current_feedback_cls()
            context = runtime.context or current_config

            # ToolRuntime 当前版本可直接读取：
            # - runtime.state: 当前图状态
            # - runtime.stream_writer: 自定义流式输出 writer
            # - runtime.tool_call_id: 当前工具调用 id
            # - runtime.config: 当前执行的 RunnableConfig
            # - runtime.context: 上层 agent 通过 context_schema 透传下来的运行期配置
            # - runtime.store: 持久化 store 这个项目中不使用
            # - runtime.execution_info / runtime.server_info: 运行信息 这个项目中不使用
            #
            # 如果需要 agent name，当前版本通常不从 runtime 顶层字段直接取。
            # 对 create_agent(name="xxx") 来说，langchain 会把这个名字写进：
            # runtime.config.get("metadata", {}).get("lc_agent_name")
            # 所以更稳妥的做法是从 config.metadata 里读。
            writer = runtime.stream_writer
            completed_input_count = (
                get_nested_count(runtime.state, "splitTextToolStats", "completedInputCount") + 1
            )

            # 用结构化事件输出过程，便于真实模型测试时做流式观察。
            emit(
                writer,
                {
                    "type": "tool",
                    "tool": current_toolschema.name,
                    "event": "start",
                    "splitCount": splitCount,
                    "uppercase": context.uppercase,
                    "completedInputCount": completed_input_count,
                },
            )

            try:
                segments = split_text(text, splitCount, uppercase=context.uppercase)
            except Exception as exc:
                error_text = feedback.failureText.format(error=str(exc))

                # 错误事件尽量保持紧凑和结构化。
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
                        # 必须返回 ToolMessage，这样 agent 才能走标准工具调用链路。
                        "messages": [ToolMessage(content=error_text, tool_call_id=runtime.tool_call_id)],
                    }
                )

            message_text = feedback.successText.format(count=len(segments)) + "\n" + "\n".join(segments)
            emit(
                writer,
                {
                    "type": "tool",
                    "tool": current_toolschema.name,
                    "event": "success",
                    "segmentCount": len(segments),
                },
            )
            return Command(
                update={
                    # 这里写回的字段必须和上面的 SubState 保持一致。
                    "messages": [ToolMessage(content=message_text, tool_call_id=runtime.tool_call_id)],
                    **update_nested_count(
                        runtime.state,
                        "splitTextToolStats",
                        "completedInputCount",
                        completed_input_count,
                    ),
                }
            )

        return split_text_segments


split_text_segments = SplitTextSegmentsTool().tool
