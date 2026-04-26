"""User chat request helpers for direct agent calls and mail delivery."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Literal

from pydantic import BaseModel, Field

from MainServer.state import MessageType


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
IMAGE_MIME_PREFIX = "image/"


class UserChatRequest(BaseModel):
    agent_name: str
    mode: Literal["direct", "mail"] = "direct"
    from_: str = Field(default="user", alias="from")
    text: str | None = None
    content: str | dict[str, Any] | list[dict[str, Any]] | None = None
    parts: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    thread_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    stream_mode: list[str] | str | None = None
    version: str | None = None
    type: MessageType = "message"
    task_info: dict[str, Any] | None = None
    timeout: float = Field(default=600.0, ge=1.0, le=3600.0)

    model_config = {"populate_by_name": True}


class UserChatRuntimeConfig(BaseModel):
    thread_id: str = "default"
    run_id: str | None = None
    stream_mode: list[str] | str = Field(default_factory=lambda: ["messages", "updates", "custom"])
    version: str = "v2"

    model_config = {"extra": "allow"}


def _attachment_link(attachment: dict[str, Any]) -> str:
    return str(attachment.get("link") or attachment.get("url") or attachment.get("value") or "").strip()


def _attachment_summary(attachment: dict[str, Any]) -> str:
    return str(attachment.get("summary") or attachment.get("name") or attachment.get("label") or "").strip()


def _is_image_attachment(attachment: dict[str, Any]) -> bool:
    link = _attachment_link(attachment).lower()
    mime_type = str(attachment.get("mime_type") or attachment.get("mimeType") or "").lower()
    kind = str(attachment.get("kind") or attachment.get("type") or "").lower()
    if kind in {"image", "image_url"}:
        return True
    if mime_type.startswith(IMAGE_MIME_PREFIX):
        return True
    if link.startswith("data:image/"):
        return True
    if link.split("?", 1)[0].endswith(IMAGE_EXTENSIONS):
        return True
    return False


def _text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _image_block(link: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": link}}


def _normalize_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for part in parts:
        item = dict(part)
        if item.get("type") == "image_url" and isinstance(item.get("image_url"), str):
            item["image_url"] = {"url": item["image_url"]}
        normalized.append(item)
    return normalized


def build_user_content(payload: UserChatRequest) -> str | list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    explicit_multimodal = False

    if isinstance(payload.content, list):
        blocks.extend(_normalize_parts(payload.content))
        explicit_multimodal = True
    elif isinstance(payload.content, dict):
        blocks.extend(_normalize_parts([payload.content]))
        explicit_multimodal = True
    elif isinstance(payload.content, str) and payload.content.strip():
        blocks.append(_text_block(payload.content.strip()))

    if payload.text and payload.text.strip():
        blocks.append(_text_block(payload.text.strip()))

    if payload.parts:
        blocks.extend(_normalize_parts(payload.parts))
        explicit_multimodal = True

    for attachment in payload.attachments:
        link = _attachment_link(attachment)
        if not link:
            continue
        summary = _attachment_summary(attachment)
        if _is_image_attachment(attachment):
            if summary:
                blocks.append(_text_block(f"Image attachment: {summary}"))
            blocks.append(_image_block(link))
            explicit_multimodal = True
        else:
            label = summary or os.path.basename(link) or "attachment"
            blocks.append(_text_block(f"Attachment: {label}\nlink: {link}"))

    if not blocks:
        raise ValueError("text, content, parts, or attachments must be provided.")

    if (
        not explicit_multimodal
        and len(blocks) == 1
        and blocks[0].get("type") == "text"
        and isinstance(blocks[0].get("text"), str)
    ):
        return str(blocks[0]["text"])
    return blocks


def build_user_message(payload: UserChatRequest) -> dict[str, Any]:
    return {"role": "user", "content": build_user_content(payload)}


def build_invoke_payload(
    payload: UserChatRequest,
    runtime_config: UserChatRuntimeConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = (
        runtime_config
        if isinstance(runtime_config, UserChatRuntimeConfig)
        else UserChatRuntimeConfig.model_validate(runtime_config or {})
    )
    thread_id = payload.thread_id or payload.session_id or config.thread_id or "default"
    run_id = payload.run_id or config.run_id or f"{thread_id}-run"
    return {
        "messages": [*payload.history, build_user_message(payload)],
        "session_id": thread_id,
        "run_id": run_id,
        "stream_mode": payload.stream_mode or config.stream_mode,
        "version": payload.version or config.version,
    }


def build_mail_content(payload: UserChatRequest) -> str | dict[str, Any]:
    if payload.type == "task":
        if payload.task_info is not None:
            return payload.task_info
        if isinstance(payload.content, dict):
            return payload.content
        raise ValueError("task_info or object content is required when type='task'.")

    content_payload = payload.model_copy(update={"attachments": []})
    try:
        user_content = build_user_content(content_payload)
    except ValueError:
        if not payload.attachments:
            raise
        lines = ["User sent attachments:"]
        for attachment in payload.attachments:
            link = _attachment_link(attachment)
            if not link:
                continue
            summary = _attachment_summary(attachment) or os.path.basename(link) or "attachment"
            lines.append(f"- {summary}: {link}")
        return "\n".join(lines)
    if isinstance(user_content, str):
        return user_content
    return json.dumps(user_content, ensure_ascii=False)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}
