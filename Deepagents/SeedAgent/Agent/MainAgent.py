import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
from MainServer.comm import AgentComm

from .Models import build_main_agent_model
from .middlewares import (
    AgentStepTraceMiddleware,
    DebugTraceMiddleware,
    ReceiveMessagesMiddleware,
    SendMessagesMiddleware,
)
from .server.memory_bridge import build_knowledge_manager_middleware
from .server.path_resolver import STORE_ROOT, WORKSPACE_ROOT, ensure_seed_workspace
from .sandbox import get_seed_agent_sandbox
from .tools.ingest_knowledge_tool import Config as IngestKnowledgeToolConfig
from .tools.ingest_knowledge_tool import IngestKnowledgeTool


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    role: str
    description: str
    responsibilities: tuple[str, ...]


AGENT_SPEC = AgentSpec(
    name="SeedAgent",
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


def runtime_agent_name() -> str:
    return os.getenv("AGENT_NAME") or AGENT_SPEC.name


class SeedMainAgent:
    def __init__(
        self,
        *,
        deep_agent: Any,
        checkpoint_stack: AsyncExitStack | None = None,
    ) -> None:
        self.deep_agent = deep_agent
        self._checkpoint_stack = checkpoint_stack

    @staticmethod
    def _run_config(
        *,
        session_id: str | None,
        run_id: str | None,
    ) -> dict[str, Any]:
        thread_id = (session_id or "default").strip() or "default"
        resolved_run_id = (run_id or f"{thread_id}-run").strip() or f"{thread_id}-run"
        return {"configurable": {"thread_id": thread_id, "run_id": resolved_run_id}}

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = list(payload.get("messages") or [])
        session_id = str(payload.get("session_id") or "default")
        run_id = str(payload.get("run_id") or f"{session_id}-run")
        return await asyncio.to_thread(
            self.deep_agent.invoke,
            {"messages": messages},
            config=self._run_config(session_id=session_id, run_id=run_id),
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


def render_system_prompt() -> str:
    responsibility_lines = "\n".join(f"- {item}" for item in AGENT_SPEC.responsibilities)
    return (
        f"你是 {runtime_agent_name()}，角色是 {AGENT_SPEC.role}。\n"
        f"{AGENT_SPEC.description}\n\n"
        f"你的主要职责：\n{responsibility_lines}\n\n"
        "运行要求：\n"
        "- 优先使用 middleware 自动挂载的工具。\n"
        "- 工具返回错误时，先阅读原因和建议，再决定是否重试。\n"
        "- 真实模型测试必须观察 stream 过程，不能只看最终回答。"
    )


def build_context_backend() -> FilesystemBackend:
    return FilesystemBackend(root_dir=WORKSPACE_ROOT, virtual_mode=True)


def build_middlewares(*, comm: AgentComm | None, debug: bool) -> list[Any]:
    context_backend = build_context_backend()
    agent_name = runtime_agent_name()
    middlewares: list[Any] = [
        MemoryMiddleware(
            backend=context_backend,
            sources=["/brain/AGENTS.md"],
        ),
        SkillsMiddleware(
            backend=context_backend,
            sources=["/skills"],
        ),
        build_knowledge_manager_middleware(agent_name=agent_name).middleware,
        ReceiveMessagesMiddleware(comm=comm).middleware,
        SendMessagesMiddleware(comm=comm).middleware,
        AgentStepTraceMiddleware().middleware,
    ]
    if debug:
        middlewares.append(DebugTraceMiddleware().middleware)
    return middlewares


def build_tools() -> list[Any]:
    return [
        IngestKnowledgeTool(
            config=IngestKnowledgeToolConfig(agentName=runtime_agent_name())
        ).tool
    ]


async def _build_main_agent_async(
    *,
    comm: AgentComm | None = None,
    config_path: Path | None = None,
    provider: str | None = None,
    debug: bool = False,
) -> SeedMainAgent:
    ensure_seed_workspace()
    model = build_main_agent_model(config_path=config_path, provider=provider)

    from deepagents import create_deep_agent

    agent_name = runtime_agent_name()
    backend = get_seed_agent_sandbox(WORKSPACE_ROOT, agent_name=agent_name)
    deep_agent = create_deep_agent(
        model=model,
        backend=backend,
        tools=build_tools(),
        middleware=build_middlewares(comm=comm, debug=debug),
        system_prompt=render_system_prompt(),
        name=agent_name,
    )
    return SeedMainAgent(deep_agent=deep_agent)


def build_main_agent(
    *,
    comm: AgentComm | None = None,
    config_path: Path | None = None,
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
            provider=provider,
            debug=debug,
        )
    )


async def abuild_main_agent(
    *,
    comm: AgentComm | None = None,
    config_path: Path | None = None,
    provider: str | None = None,
    debug: bool = False,
    session_id: str = "default",
    run_id: str | None = None,
) -> SeedMainAgent:
    del session_id, run_id
    return await _build_main_agent_async(
        comm=comm,
        config_path=config_path,
        provider=provider,
        debug=debug,
    )


def build_agent(**kwargs: Any) -> SeedMainAgent:
    return build_main_agent(**kwargs)


async def abuild_agent(**kwargs: Any) -> SeedMainAgent:
    return await abuild_main_agent(**kwargs)
