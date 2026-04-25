"""Unified database agent for document, graph, and memory capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.chat_models import BaseChatModel, init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from middleware import (
    DocumentQueryMiddlewareConfig,
    DocumentQueryCapabilityMiddleware,
    DocumentWriteMiddlewareConfig,
    DocumentWriteCapabilityMiddleware,
    GraphQueryMiddlewareConfig,
    GraphQueryCapabilityMiddleware,
    GraphWriteMiddlewareConfig,
    GraphWriteCapabilityMiddleware,
    MemoryCapabilityMiddleware,
    MemoryMiddlewareConfig,
    MemoryCapabilityPreset,
)
from models import get_chat_model_config
from server.neo4j.database_config import DEFAULT_DATABASE_CONFIG_PATH


DEFAULT_UNIFIED_AGENT_SYSTEM_PROMPT = """\
你是一个知识库联合管理 agent，负责文档、图和记忆的复杂维护。
你所有有意义的操作都必须通过工具完成；如果没有调用工具，就没有完成任务。
当任务需要整理、比较、关联或修正材料时，优先使用文档工具，再用图工具补充关系。
当任务需要在对话线程中保留稳定偏好或长期事实时，使用 manage_memory。
输出要可追溯、可维护，并尽量保留必要的来源信息。
"""


def _resolve_database_config_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_DATABASE_CONFIG_PATH
    return Path(path).expanduser().resolve()


def _build_chat_model(model: str | BaseChatModel | None, *, temperature: float) -> str | BaseChatModel:
    if isinstance(model, BaseChatModel):
        return model
    chat_config = get_chat_model_config()
    resolved_model = str(model).strip() if isinstance(model, str) and model.strip() else str(chat_config["model"])
    return init_chat_model(
        model=resolved_model,
        model_provider=str(chat_config["provider"]),
        base_url=str(chat_config["base_url"]),
        api_key=str(chat_config["api_key"]),
        temperature=temperature,
    )


def create_unified_agent(
    model: str | BaseChatModel | None = None,
    *,
    run_id: str,
    neo4j_config_path: str | Path | None = None,
    neo4j_uri: str | None = None,
    neo4j_username: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
    memory_preset: MemoryCapabilityPreset | None = None,
    include_memory: bool = True,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    debug: bool = False,
    checkpointer: Any | None = None,
):
    resolved_config_path = _resolve_database_config_path(neo4j_config_path)
    unified_middlewares: list[Any] = []
    if include_memory:
        active_memory_preset = memory_preset or MemoryCapabilityPreset()
        unified_middlewares.append(ToolRetryMiddleware(max_retries=2, tools=["manage_memory"]))
        unified_middlewares.append(
            MemoryCapabilityMiddleware(
                config=MemoryMiddlewareConfig.load(),
                preset=active_memory_preset,
            )
        )

    unified_middlewares.append(
        DocumentQueryCapabilityMiddleware(
            config=DocumentQueryMiddlewareConfig(
                neo4j_config_path=resolved_config_path,
                run_id=run_id,
                trace_limit=16,
            )
        )
    )
    unified_middlewares.append(
        DocumentWriteCapabilityMiddleware(
            config=DocumentWriteMiddlewareConfig(
                neo4j_config_path=resolved_config_path,
                run_id=run_id,
                trace_limit=16,
            )
        )
    )
    unified_middlewares.append(
        GraphQueryCapabilityMiddleware(
            config=GraphQueryMiddlewareConfig(
                neo4j_config_path=resolved_config_path,
                run_id=run_id,
                trace_limit=16,
            )
        )
    )
    unified_middlewares.append(
        GraphWriteCapabilityMiddleware(
            config=GraphWriteMiddlewareConfig(
                neo4j_config_path=resolved_config_path,
                run_id=run_id,
                trace_limit=16,
            )
        )
    )

    chat_model = _build_chat_model(model, temperature=temperature)
    resolved_checkpointer = checkpointer or InMemorySaver()
    return create_agent(
        model=chat_model,
        system_prompt=system_prompt or DEFAULT_UNIFIED_AGENT_SYSTEM_PROMPT,
        middleware=unified_middlewares,
        checkpointer=resolved_checkpointer,
        debug=debug,
        name="unified-agent",
    )


__all__ = ["DEFAULT_UNIFIED_AGENT_SYSTEM_PROMPT", "create_unified_agent"]
