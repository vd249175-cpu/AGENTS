from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


MessageType = Literal["task", "message"]


class Link(TypedDict):
    link: str
    summary: str


class CapabilityCard(TypedDict):
    title: str
    content: str


class AgentCard(TypedDict):
    agent_name: str
    capabilities: list[CapabilityCard]


class Deliverable(Link):
    pass


class TaskInfo(TypedDict):
    title: str
    goal: str
    description: str
    owner: str
    deliverables: list[Deliverable]


AgentMail = TypedDict(
    "AgentMail",
    {
        "message_id": str,
        "from": str,
        "to": str,
        "type": MessageType,
        "content": str | TaskInfo,
        "attachments": list[Link],
    },
)


AgentMessage = AgentMail


class AgentMessageBox(TypedDict):
    inbox: list[AgentMail]
    outbox: list[AgentMail]


class CollaborationState(TypedDict):
    agentcard: AgentCard
    agent_messages: AgentMessageBox


class CollaborationStateWithMessages(CollaborationState):
    messages: NotRequired[list[dict]]


def build_initial_state(
    *,
    agent_name: str | None = None,
    capabilities: list[CapabilityCard] | None = None,
) -> CollaborationState:
    return {
        "agentcard": {
            "agent_name": agent_name or "unknown-agent",
            "capabilities": capabilities or [],
        },
        "agent_messages": {
            "inbox": [],
            "outbox": [],
        },
    }
