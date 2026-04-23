"""Agent-side communication SDK based on urllib.request."""

import json
import urllib.request
from typing import Any

from MainServer.protocol import make_message
from MainServer.state import AgentMail, Link, MessageType, TaskInfo


def _post(url: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
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


def _get(url: str, timeout: float = 5.0) -> Any:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


class AgentComm:
    """Small zero-dependency SDK for Agent-to-MainServer communication."""

    def __init__(self, base_url: str, agent_name: str, *, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_name = agent_name
        self.timeout = min(float(timeout), 5.0)

    def send(
        self,
        dst: str,
        content: str | TaskInfo,
        msg_type: MessageType = "message",
        attachments: list[Link] | None = None,
    ) -> dict[str, Any]:
        if not dst:
            raise ValueError("dst must be a non-empty agent name.")
        message = make_message(
            src=self.agent_name,
            dst=dst,
            msg_type=msg_type,
            content=content,
            attachments=attachments,
        )
        return _post(f"{self.base_url}/send", message, timeout=self.timeout)

    def recv(self) -> list[AgentMail]:
        result = _get(f"{self.base_url}/recv/{self.agent_name}", timeout=self.timeout)
        return result.get("messages", [])

    def peers(self) -> list[str]:
        result = _get(f"{self.base_url}/agents/peers/{self.agent_name}", timeout=self.timeout)
        return result.get("peers", [])
