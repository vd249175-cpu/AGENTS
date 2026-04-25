"""Shared demo helpers."""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict


class StrictConfig(BaseModel):
    """所有 demo 配置的公共基类。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


def read_external_json(source: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return dict(source)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def config_from_external(
    model_cls: type[StrictConfig],
    source: dict[str, Any] | str | Path | None = None,
) -> StrictConfig:
    """读取外部配置并返回已经由 Pydantic 校验过的配置实例。"""

    return model_cls.model_validate(read_external_json(source))


def build_configured_children(
    config: StrictConfig,
    child_wrappers: dict[str, type[Any]],
    output_attr: str,
) -> list[Any]:
    """按 Schema 中的组件声明映射构建子组件，并取出它们暴露的标准对象。"""

    configured_children = []
    for child_wrapper in child_wrappers.values():
        configured_children.append(getattr(child_wrapper(config), output_attr))
    return configured_children


def get_prompt_tag_content(prompt: str, tag_name: str) -> str | None:
    """读取指定标签之间的内容，不改动原 prompt。"""

    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"

    if open_tag not in prompt or close_tag not in prompt:
        return None

    _, remainder = prompt.split(open_tag, 1)
    content, _ = remainder.split(close_tag, 1)
    return content.strip()


def update_prompt_tag_content(prompt: str, tag_name: str, updated_content: str) -> str:
    """只替换指定标签之间的内容，保留标签本身以及外层其他文本。"""

    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"

    if open_tag not in prompt or close_tag not in prompt:
        return f"{open_tag}\n{updated_content}\n{close_tag}"

    prefix, remainder = prompt.split(open_tag, 1)
    _, suffix = remainder.split(close_tag, 1)
    return f"{prefix}{open_tag}\n{updated_content}\n{close_tag}{suffix}"


def upsert_system_message(messages: list, name: str, content: str) -> list:
    """按固定 slot 名更新或插入 SystemMessage。"""

    new_message = SystemMessage(name=name, content=content)
    replaced = False
    new_messages = []

    for message in messages:
        if isinstance(message, SystemMessage) and message.name == name:
            new_messages.append(new_message)
            replaced = True
        else:
            new_messages.append(message)

    if not replaced:
        new_messages.insert(0, new_message)

    return new_messages


def get_nested_count(state: dict[str, Any], namespace: str, key: str) -> int:
    """从命名空间状态字典中读取计数。

    这是 demo 的可选调试统计 helper，不是每一层组件都必须使用的框架协议。
    """

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
    """基于已有命名空间状态生成合并后的计数更新。

    只有组件确实需要记录调用次数、处理数量等统计值时才使用。
    """

    namespace_state = state.get(namespace) or {}
    if not isinstance(namespace_state, dict):
        namespace_state = {}
    return {namespace: {**namespace_state, key: value}}


def emit(writer: Any, payload: dict[str, object]) -> None:
    if writer is None:
        return
    try:
        writer(payload)
    except TypeError:
        write = getattr(writer, "write", None)
        if callable(write):
            write(str(payload))
