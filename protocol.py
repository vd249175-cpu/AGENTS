from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


TaskStatus = Literal["running", "failed"]
MessageType = Literal["task", "message",]


# 传递网址/本地图片/本地文件等
class Link(TypedDict):
    link: str
    summary: str

# 模型能力，用来说明模型可以干什么
class CapabilityCard(TypedDict):
    title: str
    content: str

# 模型能力卡
class AgentCard(TypedDict):
    agent_name: str
    capabilities: list[CapabilityCard]

# 任务信息
class TaskInfo(TypedDict):
    title: str
    goal: str
    description: str
    status: TaskStatus
    owner: str
    deliverables: list[Deliverable]

# agent直接的通讯协议
AgenMtail= TypedDict(
    "AgentMail",
    {
        "message_id": str,
        "from": str,
        "to": str,
        "type": MessageType,
        "content": str ｜ TaskInfo
        "attachments": list[Link],
    },
)
