"""Shared demo helpers for the nested component spec."""

import json
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.config import get_config
from pydantic import BaseModel, ConfigDict


ConfigT = TypeVar("ConfigT", bound=BaseModel)


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


def read_external_json(source: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return dict(source)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def config_from_external(
    model_cls: type[ConfigT],
    source: dict[str, Any] | str | Path | None = None,
) -> ConfigT:
    return model_cls.model_validate(read_external_json(source))


def emit(writer: Any, payload: dict[str, Any]) -> None:
    if writer is None:
        return
    try:
        writer(payload)
    except TypeError:
        write = getattr(writer, "write", None)
        if callable(write):
            write(str(payload))
    except Exception:
        return


def get_nested_count(state: dict[str, Any], namespace: str, key: str) -> int:
    namespace_state = state.get(namespace) or {}
    if not isinstance(namespace_state, dict):
        return 0
    return int(namespace_state.get(key, 0) or 0)


def update_nested_count(
    state: dict[str, Any],
    namespace: str,
    key: str,
    value: int,
) -> dict[str, dict[str, int]]:
    namespace_state = state.get(namespace) or {}
    if not isinstance(namespace_state, dict):
        namespace_state = {}
    return {namespace: {**namespace_state, key: value}}


def make_named_system_message(name: str, text: str) -> SystemMessage:
    slot_name = name.strip()
    if not slot_name:
        raise ValueError("SystemMessage name must not be empty")
    return SystemMessage(content=text, name=slot_name)


def remove_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
) -> list[AnyMessage]:
    output: list[AnyMessage] = []
    for message in messages or []:
        if isinstance(message, SystemMessage) and message.name == name:
            continue
        output.append(message)
    return output


def upsert_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
    text: str,
) -> list[AnyMessage]:
    replacement = make_named_system_message(name, text)
    output: list[AnyMessage] = []
    replaced = False
    inserted = False

    for message in messages or []:
        if isinstance(message, SystemMessage):
            if message.name == replacement.name:
                if not replaced:
                    output.append(replacement)
                    replaced = True
                continue
            output.append(message)
            continue

        if not inserted:
            if not replaced:
                output.append(replacement)
                replaced = True
            inserted = True
        output.append(message)

    if not replaced:
        output.append(replacement)
    return output


def set_named_system_message(
    messages: list[AnyMessage] | tuple[AnyMessage, ...] | None,
    *,
    name: str,
    text: str | None,
) -> list[AnyMessage]:
    cleaned = (text or "").strip()
    if not cleaned:
        return remove_named_system_message(messages, name=name)
    return upsert_named_system_message(messages, name=name, text=cleaned)


def safe_current_config() -> dict[str, Any]:
    try:
        return get_config()
    except RuntimeError:
        return {}


class RuntimeIdentity(BaseModel):
    runId: str
    threadId: str


def current_runtime_identity(
    *,
    defaultRunId: str,
    defaultThreadId: str,
) -> RuntimeIdentity:
    config = safe_current_config()
    configurable = config.get("configurable", {}) or {}
    run_id = str(
        configurable.get("run_id")
        or configurable.get("runId")
        or config.get("run_id")
        or defaultRunId
    ).strip() or defaultRunId
    thread_id = str(
        configurable.get("thread_id")
        or configurable.get("threadId")
        or config.get("thread_id")
        or defaultThreadId
    ).strip() or defaultThreadId
    return RuntimeIdentity(runId=run_id, threadId=thread_id)


def build_configured_children(
    config: StrictConfig,
    child_wrappers: dict[str, type[Any]],
    output_attr: str,
) -> list[Any]:
    configured_children = []
    for child_wrapper in child_wrappers.values():
        configured_children.append(getattr(child_wrapper(config), output_attr))
    return configured_children
