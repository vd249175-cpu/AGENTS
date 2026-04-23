"""Message construction helpers for AgentMail."""

from __future__ import annotations

import uuid

from MainServer.state import AgentMail, Link, MessageType, TaskInfo


def make_message(
    *,
    src: str,
    dst: str,
    content: str | TaskInfo,
    msg_type: MessageType = "message",
    attachments: list[Link] | None = None,
) -> AgentMail:
    return {
        "message_id": str(uuid.uuid4()),
        "from": src,
        "to": dst,
        "type": msg_type,
        "content": content,
        "attachments": attachments or [],
    }
