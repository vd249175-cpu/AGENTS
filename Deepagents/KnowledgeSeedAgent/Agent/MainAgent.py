import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware import AgentState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from MainServer.comm import AgentComm
from MainServer.state import MessageType
from pydantic import Field

from .Models import build_main_agent_model_from_config
from .middlewares import (
    AgentStepTraceMiddleware,
    DebugTraceMiddleware,
    KnowledgeIngestMiddleware,
    ReceiveMessagesMiddleware,
    SendMessagesMiddleware,
)
from .middlewares.agent_step_trace import Config as AgentStepTraceConfig
from .middlewares.debug_trace import Config as DebugTraceConfig
from .middlewares.knowledge_ingest import Config as KnowledgeIngestConfig
from .middlewares.receive_messages import Config as ReceiveMessagesConfig
from .middlewares.send_messages import Config as SendMessagesConfig
from .server.demo_server import StrictConfig, config_from_external
from .server.memory_bridge import build_knowledge_manager_middleware
from .server.path_resolver import WORKSPACE_ROOT, ensure_seed_workspace, resolve_store_path
from .sandbox import get_knowledge_seed_agent_sandbox


AGENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = AGENT_ROOT / "KnowledgeSeedAgentConfig.local.json"
EXAMPLE_CONFIG_PATH = AGENT_ROOT / "KnowledgeSeedAgentConfig.example.json"


def default_config_path() -> Path:
    override = os.getenv("LANGVIDEO_KNOWLEDGE_SEED_AGENT_CONFIG")
    if override:
        return Path(override).expanduser()
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return EXAMPLE_CONFIG_PATH


class Config(StrictConfig):
    agentName: str = Field(default="KnowledgeSeedAgent")
    agentRole: str = Field(default="knowledge seed agent")
    agentDescription: str = Field(
        default="负责把 knowledge 中的长文档切分进入记忆图，并通过记忆管理工具维护知识库。"
    )
    agentResponsibilities: list[str] = Field(
        default_factory=lambda: [
            "组织当前会话的任务步骤",
            "通过 MainServer 与其他 Agent 通讯",
            "将 workspace/knowledge 中的长文档切分并写入记忆",
            "使用 manage_knowledge 查询、整理、修正和关联记忆图内容",
            "以流式事件汇报关键执行过程",
            "使用 middleware 自动挂载的工具，不在 agent 装配层重复挂 tools",
        ]
    )
    systemPromptExtra: str = Field(default="")

    chatModelProvider: str = Field(default="openai")
    chatModel: str = Field(default="gpt-5-nano")
    chatBaseUrl: str = Field(default="")
    chatApiKey: str = Field(default="")
    chatTemperature: float = Field(default=0.0, ge=0, le=2)
    ollamaBaseUrl: str = Field(default="http://127.0.0.1:11434")

    embeddingProvider: str = Field(default="openai")
    embeddingModel: str = Field(default="text-embedding-3-small")
    embeddingBaseUrl: str = Field(default="")
    embeddingApiKey: str = Field(default="")
    embeddingDimensions: int = Field(default=1536, ge=1)

    neo4jUri: str = Field(default="neo4j://localhost:7687")
    neo4jUsername: str = Field(default="neo4j")
    neo4jPassword: str = Field(default="")
    neo4jDatabase: str | None = Field(default=None)

    enableMemoryMiddleware: bool = Field(default=True)
    enableSkillsMiddleware: bool = Field(default=True)
    enableKnowledgeManagerMiddleware: bool = Field(default=True)
    enableKnowledgeIngestMiddleware: bool = Field(default=True)
    enableReceiveMessagesMiddleware: bool = Field(default=True)
    enableSendMessagesMiddleware: bool = Field(default=True)
    enableAgentStepTraceMiddleware: bool = Field(default=True)
    enableDebugTraceMiddleware: bool = Field(default=False)

    defaultRunId: str = Field(default="knowledgeseedagent-run")
    defaultThreadId: str = Field(default="default")

    defaultDestination: str | None = Field(default=None)
    defaultMessageType: MessageType = Field(default="message")
    hideToolMessageContent: bool = Field(default=False)
    currentAgentName: str | None = Field(default=None)
    blockSelfTarget: bool = Field(default=True)
    maxInboxItems: int = Field(default=20, ge=1)
    peekPeersInGuidance: bool = Field(default=True)

    resume: bool = Field(default=True)
    chunkApplyDeriveDocumentRunId: bool = Field(default=False)
    chunkApplyCheckpointPath: str | None = Field(default=None)
    chunkApplyCachePath: str | None = Field(default=None)
    chunkApplyStagingPath: str | None = Field(default=None)
    chunkApplyRecursionLimit: int | None = Field(default=None, ge=1)
    shardCount: int = Field(default=4, ge=1)
    maxWorkers: int = Field(default=2, ge=1)
    referenceBytes: int = Field(default=6000, ge=1)
    chunkApplyMaxRetries: int = Field(default=3, ge=1)
    chunkHistoryLineCount: int = Field(default=4, ge=0)
    chunkActiveLineCount: int = Field(default=8, ge=1)
    chunkPreviewLineCount: int = Field(default=4, ge=0)
    chunkLineWrapWidth: int = Field(default=30, ge=1)
    chunkWindowBackBytes: int | None = Field(default=1200, ge=1)
    chunkWindowForwardBytes: int | None = Field(default=2400, ge=1)
    chunkTraceLimit: int = Field(default=16, ge=1)
    chunkMaxRetries: int = Field(default=3, ge=1)
    documentEdgeDistance: float = Field(default=0.3, ge=0.0)
    persistKeywordEmbeddings: bool = Field(default=True)

    knowledgeRunId: str | None = Field(default=None)
    knowledgeTraceLimit: int = Field(default=16, ge=1)
    knowledgeManagerSystemPrompt: str | None = Field(default=None)
    knowledgeManagerTemperature: float = Field(default=0.0, ge=0, le=2)
    knowledgeManagerDebug: bool = Field(default=False)
    streamInnerKnowledgeAgent: bool = Field(default=True)
    innerKnowledgeRecursionLimit: int = Field(default=64, ge=1)
    managementDiscoveryMaxItems: int | None = Field(default=None, ge=1)
    managementDiscoveryMaxTotalChars: int | None = Field(default=None, ge=1)
    managementDiscoveryMaxSummaryChars: int | None = Field(default=None, ge=32)
    managementDiscoveryScanMessageLimit: int | None = Field(default=None, ge=1)
    documentQueryTraceLimit: int = Field(default=16, ge=1)
    documentWriteTraceLimit: int = Field(default=16, ge=1)
    graphQueryTraceLimit: int = Field(default=16, ge=1)
    graphWriteTraceLimit: int = Field(default=16, ge=1)
    graphQueryKeywordTopK: int = Field(default=6, ge=1)
    graphQueryKeywordTopKLimit: int = Field(default=10, ge=1)
    graphQueryDistanceTopK: int = Field(default=6, ge=1)
    graphQueryDistanceTopKLimit: int = Field(default=10, ge=1)
    graphQueryDistanceMaxDistance: float = Field(default=1.5, ge=0.0)
    graphQueryUsefulMaxItems: int = Field(default=12, ge=1)
    graphQueryUsefulMaxTotalChars: int = Field(default=3000, ge=1)
    graphQueryBlockedMaxItems: int = Field(default=12, ge=1)
    graphQueryBlockedMaxTotalChars: int = Field(default=3000, ge=1)

    agentStepTraceEventType: str = Field(default="agent_step_timing")
    agentStepTraceEmitFullState: bool = Field(default=True)
    debugTraceEventType: str = Field(default="agent_debug_trace")
    debugTraceTruncateLimit: int = Field(default=220, ge=20)

    @classmethod
    def load_config_knowledge_seed_agent(cls, source=None):
        return config_from_external(cls, source or default_config_path())


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    role: str
    description: str
    responsibilities: tuple[str, ...]


AGENT_SPEC = AgentSpec(
    name="KnowledgeSeedAgent",
    role="knowledge seed agent",
    description="负责把 knowledge 中的长文档切分进入记忆图，并通过记忆管理工具维护知识库。",
    responsibilities=(
        "组织当前会话的任务步骤",
        "通过 MainServer 与其他 Agent 通讯",
        "将 workspace/knowledge 中的长文档切分并写入记忆",
        "使用 manage_knowledge 查询、整理、修正和关联记忆图内容",
        "以流式事件汇报关键执行过程",
        "使用 middleware 自动挂载的工具，不在 agent 装配层重复挂 tools",
    ),
)


class SubState(
    AgentState,
    KnowledgeIngestMiddleware.substate,
    ReceiveMessagesMiddleware.substate,
    SendMessagesMiddleware.substate,
    AgentStepTraceMiddleware.substate,
    DebugTraceMiddleware.substate,
    total=False,
):
    pass


class AgentSchema:
    name = "knowledge_seed_agent"
    state_schema = SubState


def runtime_agent_name(config: Config | None = None) -> str:
    return os.getenv("AGENT_NAME") or (config.agentName if config is not None else AGENT_SPEC.name)


class SeedMainAgent:
    name = AgentSchema.name
    config = Config
    substate = SubState
    agentschema = AgentSchema

    def __init__(
        self,
        *,
        deep_agent: Any,
        config: Config,
        checkpoint_stack: AsyncExitStack | None = None,
    ) -> None:
        self.deep_agent = deep_agent
        self.config = config
        self._checkpoint_stack = checkpoint_stack

    @staticmethod
    def _run_config(
        *,
        session_id: str | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        thread_id = (session_id or "default").strip() or "default"
        resolved_run_id = (run_id or f"{thread_id}-run").strip() or f"{thread_id}-run"
        checkpoint_thread_id = f"{resolved_run_id}:{thread_id}"
        return {
            "configurable": {
                "thread_id": checkpoint_thread_id,
                "langvideo_run_id": resolved_run_id,
                "langvideo_thread_id": thread_id,
            },
            "metadata": {
                "langvideo_run_id": resolved_run_id,
                "langvideo_thread_id": thread_id,
                "langvideo_checkpoint_thread_id": checkpoint_thread_id,
            },
        }

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = list(payload.get("messages") or [])
        session_id = str(payload.get("session_id") or "default")
        run_id = str(payload.get("run_id") or f"{session_id}-run")
        return await self.deep_agent.ainvoke(
            {"messages": messages},
            config=self._run_config(session_id=session_id, run_id=run_id),
            context=self.config,
        )

    async def astream(
        self,
        *,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        run_id: str | None = None,
        stream_mode: list[str] | str = "custom",
        version: str = "v2",
    ):
        async for chunk in self.deep_agent.astream(
            {"messages": messages},
            config=self._run_config(session_id=session_id, run_id=run_id),
            context=self.config,
            stream_mode=stream_mode,
            version=version,
        ):
            yield chunk

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self.ainvoke(payload))

    async def aclose(self) -> None:
        if self._checkpoint_stack is not None:
            await self._checkpoint_stack.aclose()
            self._checkpoint_stack = None

    def close(self) -> None:
        asyncio.run(self.aclose())


def render_system_prompt(config: Config) -> None:
    del config
    return None


def build_context_backend() -> FilesystemBackend:
    return FilesystemBackend(root_dir=WORKSPACE_ROOT, virtual_mode=True)


def build_middlewares(*, config: Config, comm: AgentComm | None) -> list[Any]:
    context_backend = build_context_backend()
    agent_name = runtime_agent_name(config)
    middlewares: list[Any] = []
    if config.enableMemoryMiddleware:
        middlewares.append(MemoryMiddleware(backend=context_backend, sources=["/brain/AGENTS.md"]))
    if config.enableSkillsMiddleware:
        middlewares.append(SkillsMiddleware(backend=context_backend, sources=["/skills"]))
    if config.enableKnowledgeManagerMiddleware:
        middlewares.append(build_knowledge_manager_middleware(agent_name=agent_name, config=config).middleware)
    if config.enableKnowledgeIngestMiddleware:
        middlewares.append(
            KnowledgeIngestMiddleware(
                runingConfig=KnowledgeIngestConfig(
                    enabled=True,
                    defaultRunId=config.defaultRunId,
                    defaultThreadId=config.defaultThreadId,
                    resume=config.resume,
                    shardCount=config.shardCount,
                    maxWorkers=config.maxWorkers,
                    referenceBytes=config.referenceBytes,
                    agentName=agent_name,
                )
            ).middleware
        )
    if config.enableReceiveMessagesMiddleware:
        middlewares.append(
            ReceiveMessagesMiddleware(
                comm=comm,
                runingConfig=ReceiveMessagesConfig(
                    enabled=True,
                    defaultRunId=config.defaultRunId,
                    defaultThreadId=config.defaultThreadId,
                    maxInboxItems=config.maxInboxItems,
                    peekPeersInGuidance=config.peekPeersInGuidance,
                ),
            ).middleware
        )
    if config.enableSendMessagesMiddleware:
        middlewares.append(
            SendMessagesMiddleware(
                comm=comm,
                runingConfig=SendMessagesConfig(
                    enabled=True,
                    defaultRunId=config.defaultRunId,
                    defaultThreadId=config.defaultThreadId,
                    defaultDestination=config.defaultDestination,
                    defaultMessageType=config.defaultMessageType,
                    peekPeersInGuidance=config.peekPeersInGuidance,
                    blockSelfTarget=config.blockSelfTarget,
                ),
            ).middleware
        )
    if config.enableAgentStepTraceMiddleware:
        middlewares.append(
            AgentStepTraceMiddleware(
                runingConfig=AgentStepTraceConfig(
                    enabled=True,
                    defaultRunId=config.defaultRunId,
                    defaultThreadId=config.defaultThreadId,
                    eventType=config.agentStepTraceEventType,
                    emitFullState=config.agentStepTraceEmitFullState,
                )
            ).middleware
        )
    if config.enableDebugTraceMiddleware:
        middlewares.append(
            DebugTraceMiddleware(
                runingConfig=DebugTraceConfig(
                    enabled=True,
                    defaultRunId=config.defaultRunId,
                    defaultThreadId=config.defaultThreadId,
                    eventType=config.debugTraceEventType,
                    truncateLimit=config.debugTraceTruncateLimit,
                )
            ).middleware
        )
    return middlewares


async def _build_main_agent_async(
    *,
    comm: AgentComm | None = None,
    config_path: Path | None = None,
    config: Config | None = None,
    provider: str | None = None,
    debug: bool = False,
) -> SeedMainAgent:
    ensure_seed_workspace()
    current_config = config or Config.load_config_knowledge_seed_agent(config_path)
    updates: dict[str, Any] = {}
    if provider is not None:
        updates["chatModelProvider"] = provider
    if debug:
        updates["enableDebugTraceMiddleware"] = True
    if current_config.currentAgentName is None:
        updates["currentAgentName"] = runtime_agent_name(current_config)
    if current_config.knowledgeRunId is None:
        updates["knowledgeRunId"] = f"{runtime_agent_name(current_config)}-knowledge"
    if updates:
        current_config = current_config.model_copy(update=updates)

    model = build_main_agent_model_from_config(current_config)

    from deepagents import create_deep_agent

    agent_name = runtime_agent_name(current_config)
    backend = get_knowledge_seed_agent_sandbox(WORKSPACE_ROOT, agent_name=agent_name)
    checkpoint_stack = AsyncExitStack()
    checkpoint_path = resolve_store_path(None, default_relative="checkpoints/langgraph.sqlite3")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = await checkpoint_stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string(str(checkpoint_path))
    )
    await checkpointer.setup()
    deep_agent = create_deep_agent(
        model=model,
        backend=backend,
        tools=[],
        middleware=build_middlewares(config=current_config, comm=comm),
        system_prompt=render_system_prompt(current_config),
        context_schema=type(current_config),
        checkpointer=checkpointer,
        name=agent_name,
    )
    return SeedMainAgent(deep_agent=deep_agent, config=current_config, checkpoint_stack=checkpoint_stack)


def build_main_agent(
    *,
    comm: AgentComm | None = None,
    config_path: Path | None = None,
    config: Config | None = None,
    provider: str | None = None,
    debug: bool = False,
    session_id: str = "default",
    run_id: str | None = None,
) -> SeedMainAgent:
    del session_id, run_id
    return asyncio.run(
        _build_main_agent_async(
            comm=comm,
            config_path=config_path,
            config=config,
            provider=provider,
            debug=debug,
        )
    )


async def abuild_main_agent(
    *,
    comm: AgentComm | None = None,
    config_path: Path | None = None,
    config: Config | None = None,
    provider: str | None = None,
    debug: bool = False,
    session_id: str = "default",
    run_id: str | None = None,
) -> SeedMainAgent:
    del session_id, run_id
    return await _build_main_agent_async(
        comm=comm,
        config_path=config_path,
        config=config,
        provider=provider,
        debug=debug,
    )


def build_agent(**kwargs: Any) -> SeedMainAgent:
    return build_main_agent(**kwargs)


async def abuild_agent(**kwargs: Any) -> SeedMainAgent:
    return await abuild_main_agent(**kwargs)
