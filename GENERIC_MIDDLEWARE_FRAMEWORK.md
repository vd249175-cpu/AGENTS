# LangChain Agent 中间件编写框架约束

在当前 LANGVIDEO 项目中，middleware 文件放在：

```text
Deepagents/<XxxAgent>/Agent/middlewares/
```

middleware 配置 JSON 与对应 middleware 放在同一层或同一 middleware 子目录中，例如：

```text
Deepagents/SeedAgent/Agent/middlewares/send_messages.py
Deepagents/SeedAgent/Agent/middlewares/send_messages_config.json
```

当前项目不再使用旧的 `Capabilities/` 作为目标目录。能力边界由
`middlewares/` + `tools/` + 配置 JSON 表达；middleware 自动挂载自己的
tools，创建 agent 时只挂 middleware。

所有内容放到一个文件中，不分散摆放

开头不要使用：

```python
from __future__ import annotations
```

原因是 middleware 的 hook、state schema、运行时类型需要保持真实类型可见。开启延迟注解后，部分类型可能变成字符串，容易影响 langchain / langgraph 对类型和注入参数的识别。

声明部分，所有声明部分统一为以下名称，并且必须在同一个文件中完成定义。

## MiddlewareCapabilityPrompt: pydantic BaseModel

用于声明 middleware 要注入给模型的能力提示词。

推荐字段：
- `name`: 命名 system prompt slot
- `prompt`: 注入给模型的提示词内容

## middleware_capability_prompts: list[MiddlewareCapabilityPrompt]

用于集中保存 middleware 的能力提示词声明。

推荐用途：
- 注入工具使用原则，动态提示词时根据name来修改对应的提示词的对应内容

## MiddlewareRuningConfig: pydantic BaseModel

用于声明 middleware 运行时可调参数。

约束：
- 必须继承 `pydantic.BaseModel`
- 从外部 json 文件读取参数并实例化
- 配置项必须有默认值，保证没有外部 json 时 middleware 仍可运行
- 配置项应该使用 `Field(...)` 标明默认值、范围和说明

## middleware_runingconfig: MiddlewareRuningConfig

用于保存当前 middleware 的运行配置实例。

约束：
- 必须由 `MiddlewareRuningConfig` 实例化得到
- middleware 运行时只从这里读取配置
- 外部 json 没有传入时，使用默认配置

## MiddlewareToolConfig: Dict

用于声明 middleware 要自动注入给 agent 的工具集合。

约束：
- 工具集合由抽象层统一构建
- middleware 只持有工具集合，不在 middleware 内重新实现工具逻辑
- 创建 agent 时只挂 middleware，不重复传同一批 tools
- 工具实例、标准 langchain tool、工具 state 声明要集中放在这个配置里

推荐字段：

- `tools`: 标准 langchain tool 字典，键为工具名
- `toolStateTydicts`: 工具 state 声明字典，键为工具名

## MiddlewareStateTydict: TypedDict[继承自 AgentState，和工具state（如果有的话）]

用于声明 middleware 自己会读取或修改的 agent state 字段。

约束：
- 必须继承 `AgentState`
- 只声明 middleware 自己关心的 state 字段
- 如果挂有工具需要把工具state也继承

## before / after 类 middleware 如何获取运行时参数

```python
def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    ...

def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    ...
```

因此 before / after 类 middleware 获取运行时参数的方法是：
- 从 `state` 获取 agent 当前状态
- 从 `runtime` 获取运行时能力

可以从 `state` 中取：
- `state.get("messages")`: 当前消息列表
- `state.get("structured_response")`: 结构化输出，如果有
- 自己在 `MiddlewareStateTydict` 中声明的字段
- 工具回写到 state 中的字段

可以从 `runtime` 中取：
- `runtime.stream_writer`: 自定义流式输出 writer
- `runtime`:几乎可以取到运行时的任何信息，所以不要绕远路获取信息

注意：
- `Runtime` 不包含 `config`
- 如果需要读取 `RunnableConfig`，使用 `langgraph.config.get_config()`
- before / after 返回普通 dict 作为 state update
- 返回的 key 必须在 `MiddlewareStateTydict` 中声明
- 不要直接修改 `state` 本身，应该 return 一个新的更新 dict

推荐写法：

```python
def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    writer = runtime.stream_writer
    beforeCount = int(state.get("beforeCount", 0) or 0) + 1

    writer({
        "type": "middleware",
        "stage": "before_model",
        "beforeCount": beforeCount,
    })

    return {
        "beforeCount": beforeCount,
    }
```

```python
def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    writer = runtime.stream_writer
    afterCount = int(state.get("afterCount", 0) or 0) + 1
    messages = state.get("messages", [])

    writer({
        "type": "middleware",
        "stage": "after_model",
        "afterCount": afterCount,
    })

    return {
        "afterCount": afterCount,
        "lastModelText": str(messages[-1].content) if messages else None,
    }
```

## wrap_model_call 类 middleware 如何获取运行时参数

`wrap_model_call` 的源码签名是：

```python
def wrap_model_call(
    self,
    request: ModelRequest[ContextT],
    handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
    ...
```

因此 wrap 类 middleware 获取运行时参数的方法是：
- 从 `request.state` 获取 agent 当前状态
- 从 `request.runtime` 获取运行时能力
- 从 `request.messages` 获取本次模型调用消息
- 从 `request.model` 获取本次要调用的模型
- 从 `request.tools` 获取本次模型可见的工具
- 从 `request.model_settings` 获取模型调用参数
- 从 `self.runingConfig` 获取 middleware 自己的可调参数

可以从 `request` 中取：
- `request.model`: 当前 chat model
- `request.messages`: 不包含 system message 的消息列表
- `request.system_message`: 当前 system message，如果有
- `request.tool_choice`: 工具选择设置
- `request.tools`: 当前模型可见工具
- `request.response_format`: 结构化输出格式
- `request.state`: 当前 agent state
- `request.runtime`: 当前 `Runtime`
- `request.model_settings`: 模型调用参数

可以从 `request.runtime` 中取的内容，和 before / after 的 `runtime` 一致：
- `request.runtime.context`
- `request.runtime.store`
- `request.runtime.stream_writer`
- `request.runtime.previous`
- `request.runtime.execution_info`
- `request.runtime.server_info`

修改模型请求时，不要直接改 `request.messages`、`request.state` 等属性。

应该使用：

```python
newRequest = request.override(
    messages=newMessages,
    state=newState,
    model_settings=newModelSettings,
)
```

然后调用：

```python
response = handler(newRequest)
```

`wrap_model_call` 可以返回三类结果：
- `ModelResponse`: 只返回模型响应
- `AIMessage`: 简单短路返回一条 AI 消息
- `ExtendedModelResponse`: 返回模型响应，并额外携带 `Command(update=...)` 更新 state

如果需要在 wrap 中更新 state，推荐返回 `ExtendedModelResponse`：

```python
from langchain.agents.middleware import ExtendedModelResponse
from langgraph.types import Command


def wrap_model_call(self, request: ModelRequest, handler) -> ExtendedModelResponse:
    writer = request.runtime.stream_writer
    requestState = request.state
    wrapCount = int(requestState.get("wrapCount", 0) or 0) + 1

    writer({
        "type": "middleware",
        "stage": "wrap_model_call",
        "wrapCount": wrapCount,
    })

    newRequest = request.override(
        state={
            **requestState,
            "wrapCount": wrapCount,
        }
    )

    response = handler(newRequest)

    return ExtendedModelResponse(
        model_response=response,
        command=Command(
            update={
                "wrapCount": wrapCount,
            }
        ),
    )
```

注意：
- `handler(request)` 才是真正执行模型调用
- 可以在调用 `handler` 前修改 request
- 可以在调用 `handler` 后读取 response
- 可以不调用 `handler`，直接短路返回 `AIMessage` 或 `ModelResponse`
- 如果调用多次 `handler`，要明确这是 retry 或 fallback 行为
- 需要更新 state 时，用 `ExtendedModelResponse(..., command=Command(update=...))`
- `Command(update=...)` 中的字段必须在 `MiddlewareStateTydict` 中声明

## wrap_tool_call 类 middleware 如何获取运行时参数

如果 middleware 需要包裹工具调用，可以实现 `wrap_tool_call`。

源码签名是：

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    ...
```

可以从 `request` 中取：
- `request.tool_call`: 模型发起的工具调用 dict，包含 name、args、id
- `request.tool`: 即将被调用的 `BaseTool`
- `request.state`: 当前 agent state
- `request.runtime`: 工具运行时 runtime

修改工具调用时，不要直接改 `request.tool_call`。

应该使用：

```python
newRequest = request.override(
    tool_call={
        **request.tool_call,
        "args": newArgs,
    }
)
result = handler(newRequest)
```

注意：
- `handler(request)` 才是真正执行工具
- `wrap_tool_call` 最终返回 `ToolMessage` 或 `Command`
- 如果工具结果需要更新 state，返回 `Command`
- 如果只改工具返回文本，返回 `ToolMessage`
- 工具调用 id 从 `request.tool_call["id"]` 获取
- 工具参数从 `request.tool_call["args"]` 获取

## 执行后直接结束的 hook

用于在 hook 中让 agent 直接进入结束节点,如果需要直接结束使用这个工具。

关键规则：
- 使用 `@hook_config(can_jump_to=["end"])`
- 在 `before_model` 或 `after_model` 中返回 `{"jump_to": "end"}`
- `jump_to` 已经由 `AgentState` 提供，不需要重复声明
- `wrap_model_call` 中不要使用 `Command.goto`
- 结束时建议同时返回一条 `AIMessage` 说明原因

```python
class SomeMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if should_stop_after_model(state):
            return {
                "jump_to": "end",
                "messages": [
                    AIMessage(content="执行完成，直接结束。")
                ],
            }

        return None
```

## 如何注册工具

本地 langchain 源码确认：

```python
middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]
available_tools = middleware_tools + regular_tools
```

所以 middleware 注册工具的方法是：在 middleware 实例或类上暴露 `tools` 属性。

如果 `MiddlewareToolConfig["tools"]` 是以工具名为 key 的字典，注册到 middleware 时必须转成 list：

```python
class SomeMiddleware(AgentMiddleware):
    toolConfig = MiddlewareToolConfig
    tools = list(MiddlewareToolConfig["tools"].values())
```

创建 agent 时只挂 middleware：

```python
agent = create_agent(
    model=model,
    middleware=[SomeMiddleware()],
)
```

约束：
- `tools` 必须是标准 langchain tool 列表
- 如果配置中用 dict 管理工具，最终暴露给 `AgentMiddleware.tools` 时要转成 list
- 不要在 `create_agent(...)` 里重复传同一批工具
- 如果 `wrap_model_call` 动态加入了新工具，这个工具也必须能被 `ToolNode` 执行
- 动态工具如果没有提前注册，必须在 `wrap_tool_call` 中自己处理或把 `request.tool` override 成真实工具

## 如何注册声明

本地 langchain 源码确认：

```python
state_schemas = {m.state_schema for m in middleware}
```

所以 middleware 注册 state 声明的方法是：在 middleware 类上暴露 `state_schema`。

推荐写法：

```python
class SomeMiddleware(AgentMiddleware[MiddlewareStateTydict, None, Any]):
    middlewareConfig = MiddlewareConfig

    capabilityPromptConfigs = MiddlewareConfig["capabilityPromptConfigs"]
    runingConfig = MiddlewareConfig["runingConfig"]
    toolConfig = MiddlewareConfig["toolConfig"]

    stateTydict = MiddlewareConfig["stateTydict"]
    state_schema = stateTydict

    tools = list(toolConfig["tools"].values())
```

约束：
- `state_schema` 必须指向 `MiddlewareStateTydict`
- `tools` 必须从 `MiddlewareToolConfig` 注册
- `runingConfig` 必须从 `middleware_runingconfig` 注册
- `capabilityPromptConfigs` 必须从 `middleware_capability_prompts` 注册
- middleware 类不重新写散落的声明，统一从 `MiddlewareConfig` 取

如果工具有自己的 state，需要在 `MiddlewareStateTydict` 中继承或合并工具 state：

```python
class MiddlewareStateTydict(AgentState, ToolStateTydict, total=False):
    beforeCount: int
    afterCount: int
```

## 如何取到能力对应的 SystemMessage

能力 prompt 使用 `MiddlewareCapabilityPrompt.name` 作为 `SystemMessage.name`。

注册能力 prompt 时：

```python
middleware_capability_prompts = [
    MiddlewareCapabilityPrompt(
        name="generic.guidance",
        prompt="使用工具时必须先观察工具返回和流式 trace。",
    )
]
```

注入成 `SystemMessage` 时：

```python
from langchain_core.messages import SystemMessage


systemMessage = SystemMessage(
    name=capabilityPrompt.name,
    content=capabilityPrompt.prompt,
)
```

按能力名查找对应 `SystemMessage`：

```python
from langchain_core.messages import SystemMessage


def get_capability_system_message(messages: list, capabilityName: str) -> SystemMessage | None:
    for message in messages:
        if isinstance(message, SystemMessage) and message.name == capabilityName:
            return message
    return None
```

在 `wrap_model_call` 中读取：

```python
def wrap_model_call(self, request: ModelRequest, handler):
    guidanceMessage = get_capability_system_message(
        request.messages,
        "generic.guidance",
    )

    if guidanceMessage is not None:
        guidanceText = guidanceMessage.content

    return handler(request)
```

如果要根据能力名更新对应的 `SystemMessage`，不要直接修改原 list 中的对象，应该生成新的 messages：

```python
def upsert_capability_system_message(
    messages: list,
    capabilityName: str,
    content: str,
) -> list:
    newMessage = SystemMessage(name=capabilityName, content=content)
    replaced = False
    newMessages = []

    for message in messages:
        if isinstance(message, SystemMessage) and message.name == capabilityName:
            newMessages.append(newMessage)
            replaced = True
        else:
            newMessages.append(message)

    if not replaced:
        newMessages.insert(0, newMessage)

    return newMessages
```

在 `wrap_model_call` 中更新并注册到本次模型请求：

```python
def wrap_model_call(self, request: ModelRequest, handler):
    messages = request.messages

    for capabilityPrompt in self.capabilityPromptConfigs:
        messages = upsert_capability_system_message(
            messages,
            capabilityPrompt.name,
            capabilityPrompt.prompt,
        )

    newRequest = request.override(messages=messages)
    return handler(newRequest)
```

注意：
- `SystemMessage.name` 就是能力 slot 的稳定标识
- 能力 prompt 的 name 必须和 `SystemMessage.name` 一一对应
- 多个能力 prompt 用多个带 name 的 `SystemMessage`
- 不要使用已经废弃的 `system_prompt` 字符串字段做多 slot 管理
- 如果只需要一个全局 system message，可以使用 `request.system_message`
- 如果需要多个能力 slot，推荐用 `request.messages` 中的命名 `SystemMessage`

## 自定义流式输出打印关键信息

自定义流式输出统一使用 `stream_writer`，只打印关键过程摘要。

```python
# before_model / after_model
writer = runtime.stream_writer

# wrap_model_call / wrap_tool_call
writer = request.runtime.stream_writer
```

推荐统一输出 dict：

```python
writer({
    "type": "middleware",
    "stage": "after_model",
    "middleware": self.name,
    "messageCount": len(state.get("messages", [])),
})
```

关键字段：
- `type`: 固定写 `"middleware"`
- `stage`: 当前 hook 名称，例如 `before_model`、`wrap_model_call`
- `middleware`: `self.name`
- `messageCount`: 当前消息数量，可选
- `toolNames` 或 `toolCalls`: 工具名称或调用摘要，可选
- `message`: 简短过程说明

约束：
- 只输出过程摘要，不输出大段文本或敏感信息
- writer 不替代 state update
- 每条事件至少包含 `type` 和 `stage`
- 真实模型测试必须逐块消费 stream，不能只看最终 answer
