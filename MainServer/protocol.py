"""Message construction helpers for AgentMail."""

from __future__ import annotations

import secrets
import string

from MainServer.state import AgentMail, Link, MessageType, TaskInfo

_ID_ALPHABET = string.ascii_lowercase + string.digits


def short_id(length: int = 4) -> str:
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(length))


def make_message(
    *,
    src: str,
    dst: str,
    content: str | TaskInfo,
    msg_type: MessageType = "message",
    attachments: list[Link] | None = None,
) -> AgentMail:
    return {
        "message_id": short_id(),
        "from": src,
        "to": dst,
        "type": msg_type,
        "content": content,
        "attachments": attachments or [],
    }
