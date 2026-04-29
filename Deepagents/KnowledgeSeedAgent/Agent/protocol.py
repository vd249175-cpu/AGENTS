"""KnowledgeSeedAgent shared task and communication protocol aliases."""

from MainServer.protocol import make_message
from MainServer.state import (
    AgentMail,
    AgentCard,
    AgentMessage,
    AgentMessageBox,
    CapabilityCard,
    CollaborationState,
    Link,
    MessageType,
    TaskInfo,
)

__all__ = [
    "AgentMail",
    "AgentCard",
    "AgentMessage",
    "AgentMessageBox",
    "CapabilityCard",
    "CollaborationState",
    "Link",
    "MessageType",
    "TaskInfo",
    "make_message",
]
