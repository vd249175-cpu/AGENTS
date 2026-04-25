"""单文件 agent 模板。

这个文件的声明顺序固定为：
1. Config
2. SubState
3. AgentSchema
4. 本地 helper 函数
5. 包裹类
6. 默认 agent 入口

这里同步补上 agent 框架里的关键约束：
- 不要在这个文件中开启 ``from __future__ import annotations``。
- agent 通过 ``create_agent(...)`` 构造，模型通过 ``init_chat_model(...)`` 构造。
- 工具由 middleware 自动注册，不要在 ``create_agent(...)`` 里重复传同一批 tools。
- agent 的默认配置在实例化时用于构造模型和 middleware；
  同一份配置也作为 ``context_schema`` 运行期透传给 middleware / tool。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState
from langchain.chat_models import init_chat_model
from pydantic import Field

from demo_nested_middleware import Config as DemoSplitTextMiddlewareConfig
from demo_nested_middleware import DemoSplitTextMiddleware
from demo_server import build_configured_children, config_from_external
from server.models.ollama import DEFAULT_MODEL_NAME, load_chat_model_config


class Config(DemoSplitTextMiddlewareConfig):
    """agent 运行配置。"""

    modelName: str = Field(default=DEFAULT_MODEL_NAME, description="agent 使用的聊天模型名。")
    temperature: float = Field(default=0.2, ge=0, le=2, description="agent 模型温度。")
    numPredict: int = Field(default=500, ge=1, description="agent 单轮最大输出 token 数。")

    @classmethod
    def load_config_demo_nested_agent(cls, source=None):
        """从外部配置加载当前 agent 可用的 context。"""

        return config_from_external(cls, source)


class SubState(AgentState, DemoSplitTextMiddleware.substate, total=False):
    """当前 agent 对外声明的状态字段。"""

    pass


class AgentSchema:
    """agent 对外声明集中放在这里。"""

    name = "demo_nested_agent"
    systemPrompt = (
        "你是一个 demo nested agent。\n"
        "当任务需要把一段文本切分成多段时，优先调用 split_text_segments。\n"
        "不要手工模拟工具结果。"
    )
    middlewares = {
        DemoSplitTextMiddleware.name: DemoSplitTextMiddleware,
    }


def build_agent_model(config: Config):
    """根据 agent Config 构造聊天模型。"""

    chat_config = load_chat_model_config()
    provider = chat_config.get("provider")
    base_url = chat_config.get("base_url")
    api_key = chat_config.get("api_key")

    return init_chat_model(
        model=config.modelName,
        model_provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.numPredict,
    )


class DemoNestedAgent:
    """当前 agent 的外层包裹声明。"""

    name = AgentSchema.name
    config = Config
    substate = SubState
    agentschema = AgentSchema

    def __init__(self, config: Config | None = None, checkpointer=None):
        self.config = config or self.config()
        self.checkpointer = checkpointer
        self.agent = self.create_agent()

    def create_agent(self):
        """根据当前实例配置构造标准 agent。"""

        current_config = self.config
        return create_agent(
            model=build_agent_model(current_config),
            system_prompt=self.agentschema.systemPrompt,
            middleware=build_configured_children(current_config, self.agentschema.middlewares, "middleware"),
            state_schema=self.substate,
            context_schema=type(current_config),
            checkpointer=self.checkpointer,
            name=self.agentschema.name,
        )


def create_demo_nested_agent(config: Config | None = None, checkpointer=None):
    """标准 agent 构造入口。"""

    return DemoNestedAgent(config, checkpointer=checkpointer).agent


create_demo_nested_agent.config = Config
create_demo_nested_agent.substate = SubState
create_demo_nested_agent.agentschema = AgentSchema
demo_nested_agent = DemoNestedAgent
