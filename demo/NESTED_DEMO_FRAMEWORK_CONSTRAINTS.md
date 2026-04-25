# 新版嵌套 Demo 框架约束

这份文档用于约束当前这套 demo 风格的 agent / middleware / tool 组织方式。

它继承 `README.md`、`GENERIC_TOOL_FRAMEWORK.md`、`GENERIC_MIDDLEWARE_FRAMEWORK.md` 中已经确认的基础规则，同时补充这次嵌套配置演示得到的新约束。

当前目标不是做插件市场式的任意拓扑系统，而是让一棵固定但可以继续加深的能力树保持清晰：

```text
outer tool -> agent -> middleware -> inner tool
```

后续可以继续嵌套成：

```text
tool -> agent -> middleware -> tool -> agent -> middleware -> tool
```

只要每一层都遵守本文的配置、context、state、prompt 和真实模型测试规则。

## 总原则

- 组件拓扑由 Python 类结构表达，外部配置只覆盖参数，不临时改写拓扑。
- 每个 agent、middleware、tool 都应该是独特组件；当前不优先支持同一个组件被重复挂载多次。
- 如果未来真的出现重复挂载、动态编排、插件注册表，再引入实例 path / namespace / registry，不提前把 demo 做复杂。
- 标准 LangChain 对象保持标准：tool 是 `@tool` 返回的对象，middleware 是 `AgentMiddleware` 实例，agent 是 `create_agent(...)` 返回的图。
- 我们自己的类只做外层声明和装配，不把自定义字段硬塞进 LangChain 标准对象。

## 文件结构

单个 demo 组件优先放在一个文件中。

工具文件推荐顺序：

```text
Config
SubState
Input
ToolFeedback
ToolSchema
本地 helper 函数
工具包裹类
默认标准工具实例
```

中间件文件推荐顺序：

```text
Config
SubState
AffectedPrompt / AffectedPrompts
MiddlewareSchema
本地 helper 函数
标准 Middleware 类
中间件包裹类
默认中间件入口
```

Agent 文件推荐顺序：

```text
Config
SubState
AgentSchema
本地 helper 函数
agent 包裹类
默认 agent 入口
```

公共 helper 放到 `demo_server.py`。凡是会被工具、中间件、agent 重复使用的逻辑，都不要散落在业务文件中。

## 注解和类型

- 文件开头不要使用 `from __future__ import annotations`。
- `runtime: ToolRuntime[...]`、`Runtime[...]`、`ModelRequest[...]`、`ToolCallRequest` 等真实类型要保持可见。
- 工具函数签名里必须保留 `runtime: ToolRuntime[Config, SubState]`。
- middleware hook 里必须保留真实参数形态，例如 `before_model(self, state: SubState, runtime: Runtime[Config])`。
- `ContextT` 是 LangChain 类型标注层面的类型变量，不是运行时配置父类。实际运行 context 使用当前组件的 `Config` 类型表达。

## Config 和 Context

新版 demo 使用一份可继承的扁平 `Config` 作为运行配置和 context。

约束：

- 最底层组件声明自己的 `Config`。
- 上层组件的 `Config` 继承下层组件的 `Config`，再追加自己需要的字段。
- 每个配置项必须有默认值。
- 每个配置项尽量使用 `Field(...)` 写清说明、范围和默认值。
- 配置模型必须开启默认值校验和额外字段拒绝。
- 不再使用动态创建 Config 子类的方式承载外部配置。
- 不再使用 `ClassVar[type[BaseModel]]` 保存子配置。
- 不再通过修改类属性来注入运行时配置。

推荐形态：

```python
class Config(BaseModel):
    model_config = {
        "extra": "forbid",
        "validate_default": True,
    }

    uppercase: bool = Field(default=False, description="是否把返回片段转成大写。")

    @classmethod
    def load_config_xxx(cls, source=None):
        return config_from_external(cls, source)
```

如果项目为了减少重复而保留公共配置基类，也只能承载通用 Pydantic 行为，不能隐藏业务配置逻辑。

## 外部配置导入

每层组件都保留一个语义明确的类方法，用来从外界导入当前层可用的配置：

```python
Config.load_config_split_text_segments_tool(source)
Config.load_config_demo_split_text_middleware(source)
Config.load_config_demo_nested_agent(source)
```

这些方法只做很薄的一层命名包装，内部统一调用公共服务：

```python
config_from_external(cls, source)
```

公共服务负责：

- 读取 dict / json path / 空配置。
- 调用 Pydantic `model_validate(...)`。
- 返回已经校验过的配置实例。

外部 json 对当前这版扁平继承配置来说应该是扁平结构：

```json
{
  "uppercase": true,
  "maxToolUseCount": 2,
  "temperature": 0.0,
  "numPredict": 300
}
```

使用某一层能力时，只需要传那一层完整的 context；不要求所有调用都知道最顶层。

## 装配方式

用户侧不直接调用 `build()`。

每个外层包裹类在实例化时完成装配，并把标准对象暴露成固定属性：

```python
tool = SplitTextSegmentsTool(config).tool
middleware = DemoSplitTextMiddleware(config).middleware
agent = DemoNestedAgent(config).agent
outer_tool = DemoNestedAgentTool(config).tool
```

约束：

- 类属性只放声明，例如 `config = Config`、`substate = SubState`、`toolschema = ToolSchema`。
- 实例属性保存本次运行配置，例如 `self.config = config or self.config()`。
- 构造标准对象时捕获实例配置，不读取会被全局共享的可变类配置。
- 不通过修改类属性影响某一次构建，避免污染后续实例。
- middleware 可以保留包裹类形式，即 `DemoSplitTextMiddleware(config).middleware`，以保持工具、agent、中间件三层形态一致。

## 子组件传递

子组件配置通过同一个 context 实例向下传。wrapper 类上不再额外声明 `children`，因为挂载关系已经写在对应的 Schema 中。

公共 helper 可以保持这种形式：

```python
build_configured_children(config, MiddlewareSchema.tools, "tool")
build_configured_children(config, AgentSchema.middlewares, "middleware")
```

它的语义是：

- 父组件持有当前完整 context。
- Schema 声明当前层挂载哪些子组件。
- 子 wrapper 用同一份 context 实例化，因此参数会自动向下传。
- 子 wrapper 暴露自己的标准对象。

这适合当前“每个组件独特、配置扁平继承”的 demo 结构。

约束：

- 子组件挂载关系只写在 Schema 中。
- wrapper 不再额外写 `children = ...` 这层转译。
- `children` 只作为“子组件”的概念存在，不作为固定类属性名。

## Runtime Context

agent 必须注册 context schema：

```python
create_agent(
    ...,
    context_schema=type(current_config),
)
```

真实运行时必须把 context 继续传入：

```python
agent.stream(
    {"messages": [...]},
    context=current_config,
    stream_mode=["custom", "updates"],
    version="v2",
)
```

工具读取配置时：

```python
context = runtime.context or current_config
```

middleware 读取配置时：

```python
context = request.runtime.context or self.config
```

嵌套 agent 时，外层 tool 必须把自己的 context 继续传给内层 agent：

```python
agent = DemoNestedAgent(context).agent
agent.stream(..., context=context, stream_mode=["custom", "updates"], version="v2")
```

这样顶层一次性配置可以传到所有下层；如果只单独使用中间某层，也可以只传该层完整 context。

## Tool 约束

- `@tool` 返回对象就是标准工具，不给它硬注入 `config`、`substate`、`toolschema` 等字段。
- 这些声明放在工具包裹类上。
- 工具只做一类清晰任务。
- 工具运行时只从 `runtime` 读取动态值。
- 工具必须返回 `Command(update=...)`。
- `Command.update["messages"]` 中必须包含 `ToolMessage(tool_call_id=runtime.tool_call_id)`。
- 工具必须输出结构化 custom event，至少有 `start`、`success`、`error`。
- 工具更新的 state 字段必须在自己的 `SubState` 中声明。

## Middleware 约束

- 标准 middleware 类继承 `AgentMiddleware[SubState, Config]`。
- middleware 类上必须暴露 `state_schema = SubState`。
- middleware 实例上必须暴露 `tools`，由中间件负责挂载工具。
- 创建 agent 时只挂 middleware，不重复把同一批工具传给 `create_agent(...)`。
- `before_model` / `after_model` 从 `state` 和 `runtime` 取值。
- `wrap_model_call` 从 `request` 取模型调用相关信息。
- `wrap_tool_call` 从 `request.tool_call`、`request.tool`、`request.runtime`、`request.state` 取工具调用相关信息。
- 修改 request 时使用 `request.override(...)`，不要直接改 request 内部对象。
- 需要在 wrap 中更新 state 时，返回 `ExtendedModelResponse(..., command=Command(update=...))` 或 `Command(update=...)`。

## Agent 约束

- agent 使用 `create_agent(...)` 构造。
- 模型使用 `init_chat_model(...)` 构造。
- agent 配置中放模型相关字段，例如 `modelName`、`temperature`、`numPredict`。
- agent 负责把 middleware、state schema、context schema、checkpointer、name 组装在一起。
- agent 不直接实现工具业务逻辑。
- agent 的 `name` 要稳定，便于 trace 和 metadata 读取。

## Agent 封装成工具

当 agent 被封装成 tool 时，仍然遵守工具规则。

约束：

- 外层工具的 `Config` 继承被封装 agent 的 `Config`。
- 外层工具的 `SubState` 继承被封装 agent 的 `SubState`。
- 外层工具执行时创建内层 agent 实例。
- 调用内层 agent 必须使用 stream。
- 内层 agent 的 stream chunk 要转成外层 tool 的结构化 custom event。
- 外层工具不能只等待内层最终结果。

推荐链路：

```text
outer tool custom event
  -> nested agent stream custom / updates
  -> middleware custom event
  -> inner tool custom event
```

## State 约束

State 的核心约束是“谁写回，谁声明，并且按组件隔离”。计数只是这个 demo 为了观察执行过程使用的一种调试统计，不是每一层组件都必须拥有计数。

如果组件需要记录计数，不要把计数字段裸放在顶层，例如不要直接写：

```python
completedInputCount: int
toolUseCount: int
```

当前 demo 的统计信息使用组件级命名空间 dict：

```python
splitTextToolStats: dict[str, int]
demoSplitTextMiddlewareStats: dict[str, int]
nestedAgentToolStats: dict[str, int]
```

约束：

- 每个组件只声明自己负责写回的 state 字段。
- 上层 `SubState` 继承下层 `SubState`。
- 有状态写回时，优先写到组件自己的命名空间中，不直接覆盖其他组件状态。
- 公共计数 helper 只是 demo 调试辅助；组件没有计数需求时不需要使用。
- 如果未来同一组件会重复挂载多次，必须先升级为实例 path namespace，再继续扩展。

## Prompt 约束

middleware 影响提示词时使用稳定的 `SystemMessage.name`。

约束：

- 一个 `AffectedPrompt` 对应一个 `SystemMessage.name` 和一段 prompt。
- 多个 prompt 使用 `AffectedPrompts(Prompts=[...])`。
- name 要带组件前缀，例如 `demo_split_text_middleware.split_text`，不要只写 `split_text`。
- 动态更新时只替换约定标签内部内容，例如 `<SplitCapability>...</SplitCapability>`。
- prompt 渲染可以读取 state，例如把已处理数量写入 prompt。
- 当前 demo 可以在 `before_model` 中插入或更新命名 `SystemMessage`。


## 公共服务

以下逻辑应放在 `demo_server.py` 或后续公共服务层：

- `read_external_json`
- `config_from_external`
- `build_configured_children`
- `emit`
- prompt tag 读取和替换
- 按 `SystemMessage.name` 更新或插入系统消息
- 可选的组件状态统计 helper

这里的两个说法含义如下：

- “按 `SystemMessage.name` 更新或插入系统消息”指的是：如果消息列表里已经有同名 `SystemMessage`，就替换它的内容；如果没有，就插入一条新的。同名 slot 用来稳定定位 middleware 影响的提示词片段。
- “可选的组件状态统计 helper”指的是：如果某个 demo 组件需要记录调用次数、处理数量等调试统计，可以写入类似 `demoSplitTextMiddlewareStats["toolUseCount"]` 的组件字典里；没有计数需求的组件不需要为了符合框架而添加计数字段。

业务文件只保留声明、轻量 helper 和当前组件主体。

## 真实模型测试

真实模型测试必须流式监控，不可以只看最终结果。

固定要求：

- 使用 `agent.stream(...)`。
- 使用 `stream_mode=["custom", "updates"]`。
- 使用 `version="v2"`。
- 测试代码必须逐块消费 stream。
- 必须观察模型是否真的发起 tool call。
- 必须观察 middleware hook 是否触发。
- 必须观察 tool writer 是否输出 start / success / error。
- 必须观察 state update 是否出现。
- 必须观察 context 配置是否真的影响下层行为。
- 测试打印要降噪，不要给每个 chunk 重复打印相同前缀。
- 测试打印要压低内容量，不要重复输出完整 messages、完整 prompt 或大段模型文本。
- 对 `updates` 只打印节点名、更新 key、工具名、状态计数等摘要。
- 对 `custom` 只打印结构化事件的关键字段。

尤其是 agent 封装成 tool 后，要确认外层 tool 的 stream 中能看到内层 agent 的 stream 摘要。

最终回答正确不代表链路正确。下面任何一项坏掉，都算测试失败：

- 模型没有真的调用工具。
- middleware 没有挂上工具。
- middleware hook 没有触发。
- tool 没有返回 `ToolMessage`。
- writer 没有输出 custom event。
- context 没有透传到下层。
- state 没有按 `SubState` 声明更新。

## 当前不做的事

当前 demo 不引入这些复杂机制：

- 任意运行期拓扑描述。
- 组件注册表和插件市场。
- 同一组件多实例挂载。
- 动态 Config 子类。
- ClassVar 子配置树。
- 往 LangChain 标准 tool 上硬塞自定义属性。
- 通过改类属性影响某次构建。
- 只看最终 answer 的真实模型测试。

如果未来需求真的走向动态编排，再单独设计：

- `ComponentSpec`
- `ComponentNodeConfig`
- registry
- instance path
- per-instance state namespace
- per-instance prompt namespace

在那之前，这版 demo 以可读、可继承、可流式验证为优先。
