# Generic Tool Framework

<!-- Flywheel Prefix: 工具框架文档，负责定义单文件 tool 的声明、配置、运行时、state 和流式输出写法。 -->

## Purpose

这份文档只描述工具层。当前项目使用单文件结构：一个工具对应一个 Python 文件，工具的声明、默认配置和执行类都放在这个文件中。

在当前 LANGVIDEO 项目中，工具文件放在：

```text
Deepagents/<XxxAgent>/Agent/tools/
```

工具配置 JSON 与对应工具放在同一层或同一工具子目录中，例如：

```text
Deepagents/SeedAgent/Agent/tools/send_message_tool.py
Deepagents/SeedAgent/Agent/tools/send_message_tool_config.json
```

工具做三件事：

- 向模型暴露一个清晰能力。
- 通过 `ToolRuntime` 读取运行时 state、writer 和 tool call id。
- 用 `Command(update=...)` 回写结果。

工具不负责 agent 装配、系统提示词注入和 middleware hook。

## Single File Shape

工具文件内部按这个顺序组织：

```text
generic_tool.py
  imports
  ToolInputSm
  ToolDescriptionSc
  ToolReturnSc
  ToolStateTydict
  ToolRuningConfigSc
  ToolSpec
  tool_runingconfig
  emit helper
  GenericTool
```

这样可以把工具声明、默认配置、state 和执行逻辑收口到一个文件中。
外部 JSON 只负责覆盖运行配置，不保存运行时 state 或工具执行结果。

## Naming

类名保持统一后缀：

- `ToolInputSm`
- `ToolDescriptionSc`
- `ToolReturnSc`
- `ToolStateTydict`
- `ToolRuningConfigSc`
- `GenericTool`

业务工具可以在前面加业务前缀，但后缀不变，例如 `ImagePromptToolInputSm`、`ImagePromptToolStateTydict`、`ImagePromptTool`。

## State

工具 state 使用 `ToolStateTydict(AgentState)`。运行时直接按 dict 读写：

```python
current_state = runtime.state
total_runs = int(current_state.get("totalRuns", 0) or 0)
```

不要再写 `stateAsDict(...)`、`model_dump()` 或属性访问兼容层。工具只声明自己会更新的字段，中间件负责聚合多个工具的 state。

## Runtime

工具函数签名中保留 `runtime: ToolRuntime`，不要隐藏在别的参数对象里。这个细节会影响 LangChain / LangGraph 对注入参数的识别。

工具运行时只从这里取动态信息：

- `runtime.state`
- `runtime.stream_writer`
- `runtime.tool_call_id`

运行配置放在工具实例的 `runingConfig` 上。

## Return And Stream

工具返回 `Command(update=...)`，并在 `update["messages"]` 中放 `ToolMessage(tool_call_id=runtime.tool_call_id)`。

stream 输出使用结构化 custom event。事件不用复杂，但要有调试价值：

```python
{"type": "tool", "tool": "generic_tool", "event": "start", "preview": "..."}
{"type": "tool", "tool": "generic_tool", "event": "success", "preview": "..."}
{"type": "tool", "tool": "generic_tool", "event": "error", "error": "..."}
```

真实模型测试时要流式观察这些事件，不能只看最终回答。

## Single File Code Example

```python
"""Generic Prefix: 单文件工具，集中声明输入、state、运行配置和工具执行逻辑。"""

# 不在工具文件默认开启 from __future__ import annotations。
# ToolRuntime 注解需要保持运行时可见，便于框架识别 injected args。

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field


class ToolInputSm(BaseModel):
    queryText: str = Field(description="The short input text to process.")
    reason: str | None = Field(default=None, description="Optional reason for calling this tool.")


class ToolDescriptionSc(BaseModel):
    toolName: str = Field(default="generic_lookup", description="Tool name exposed to the model.")
    toolDescription: str = Field(
        default="Look up or derive a compact result for a short query.",
        description="Tool description exposed to the model.",
    )


class ToolReturnSc(BaseModel):
    successText: str = Field(default="Lookup finished.")
    failureText: str = Field(default="Lookup failed.")


class ToolStateTydict(AgentState, total=False):
    totalRuns: int
    lastInput: str | None
    lastResult: str | None
    lastError: str | None


class ToolRuningConfigSc(BaseModel):
    maxReturnChars: int = Field(default=600, ge=1)
    maxPreviewChars: int = Field(default=240, ge=1)
    progressPrefix: str = Field(default="[GenericTool]")
    hideToolMessageContent: bool = Field(default=True)


ToolSpec = {
    "description": ToolDescriptionSc(),
    "inputSm": ToolInputSm,
    "stateTydict": ToolStateTydict,
    "returnsSc": ToolReturnSc(),
}

tool_runingconfig = ToolRuningConfigSc()


def emit_tool_event(writer, payload: dict[str, object]) -> None:
    if writer is None:
        return
    try:
        writer(payload)
    except TypeError:
        write = getattr(writer, "write", None)
        if callable(write):
            write(str(payload))


class GenericTool:
    def __init__(
        self,
        toolSpec: dict[str, object] = ToolSpec,
        runingConfig: ToolRuningConfigSc = tool_runingconfig,
    ) -> None:
        self.toolSpec = toolSpec
        self.runingConfig = runingConfig
        self.stateTydict = toolSpec["stateTydict"]
        self.tool = self.buildTool()

    def buildTool(self):
        tool_spec = self.toolSpec
        description = tool_spec["description"]
        returns = tool_spec["returnsSc"]

        @tool(
            description.toolName,
            args_schema=tool_spec["inputSm"],
            description=description.toolDescription,
        )
        def runGenericTool(
            runtime: ToolRuntime,
            queryText: str,
            reason: str | None = None,
        ) -> Command:
            current_state = runtime.state
            writer = runtime.stream_writer
            preview = queryText[: self.runingConfig.maxPreviewChars]

            try:
                emit_tool_event(
                    writer,
                    {
                        "type": "tool",
                        "tool": description.toolName,
                        "event": "start",
                        "reason": reason,
                        "preview": preview,
                    },
                )

                result_text = f"result for: {queryText}"[: self.runingConfig.maxReturnChars]
                message_content = "" if self.runingConfig.hideToolMessageContent else result_text

                emit_tool_event(
                    writer,
                    {
                        "type": "tool",
                        "tool": description.toolName,
                        "event": "success",
                        "preview": result_text[: self.runingConfig.maxPreviewChars],
                    },
                )

                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=message_content,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "totalRuns": int(current_state.get("totalRuns", 0) or 0) + 1,
                        "lastInput": queryText,
                        "lastResult": result_text,
                        "lastError": None,
                    }
                )
            except Exception as exc:
                error_text = f"{returns.failureText}: {exc}"
                emit_tool_event(
                    writer,
                    {
                        "type": "tool",
                        "tool": description.toolName,
                        "event": "error",
                        "reason": reason,
                        "preview": preview,
                        "error": error_text,
                    },
                )
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=error_text,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "lastInput": queryText,
                        "lastResult": None,
                        "lastError": error_text,
                    }
                )

        return runGenericTool
```

## Check Sketch

```python
from types import SimpleNamespace

from generic_tool import GenericTool


tool_instance = GenericTool()
events: list[object] = []
runtime = SimpleNamespace(
    state={"totalRuns": 0},
    stream_writer=events.append,
    tool_call_id="tool-call-1",
)

assert tool_instance.tool._injected_args_keys == frozenset({"runtime"})

result = tool_instance.tool.func(
    runtime=runtime,
    queryText="hello",
    reason="smoke test",
)

assert result.update["messages"]
assert result.update["lastResult"]
assert any(item.get("event") == "success" for item in events if isinstance(item, dict))
```
