from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from MainServer.mail_router import route_message_assets
from MainServer.state import AgentMail, MessageType


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat()


def _jsonable(data: Any) -> Any:
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, dict):
        return {str(key): _jsonable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [_jsonable(value) for value in data]
    if hasattr(data, "model_dump"):
        try:
            return _jsonable(data.model_dump())
        except Exception:
            pass
    if isinstance(data, datetime):
        return _iso(data)
    return repr(data)


def _trim_events(events: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    if len(events) <= limit:
        return events
    return events[-limit:]


class AgentRegistration(BaseModel):
    agent_name: str
    host: str | None = None
    pid: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: list[str] | None = None


class AgentStatusUpdate(BaseModel):
    status: str
    phase: str | None = None
    step: str | None = None
    state: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEvent(BaseModel):
    phase: str | None = None
    event: str
    started_at: str | None = None
    finished_at: str | None = None
    request: dict[str, Any] | None = None
    state: Any = None
    tool_call: dict[str, Any] | None = None
    result_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentErrorReport(BaseModel):
    error_type: str | None = None
    error_message: str
    traceback: str | None = None
    phase: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class AgentRecord:
    agent_name: str
    host: str | None = None
    pid: int | None = None
    status: str = "unknown"
    phase: str | None = None
    step: str | None = None
    state: Any = None
    last_error: dict[str, Any] | None = None
    last_event: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    events: list[dict[str, Any]] = field(default_factory=list)
    scope: list[str] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "host": self.host,
            "pid": self.pid,
            "status": self.status,
            "phase": self.phase,
            "step": self.step,
            "state": _jsonable(self.state),
            "last_error": _jsonable(self.last_error),
            "last_event": _jsonable(self.last_event),
            "metadata": _jsonable(self.metadata),
            "registered_at": _iso(self.registered_at),
            "updated_at": _iso(self.updated_at),
            "events": [_jsonable(event) for event in self.events],
            "scope": self.scope,
            "workspace": str(agent_workspace(self.agent_name)),
        }

    def touch(self) -> None:
        self.updated_at = _now()


app = FastAPI(title="LANGVIDEO MainServer", version="1.0.0")

_registry: dict[str, AgentRecord] = {}
_mailboxes: dict[str, list[AgentMail]] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def agent_workspace(agent_name: str) -> Path:
    return PROJECT_ROOT / "Deepagents" / agent_name / "workspace"


def _get_or_create(agent_name: str) -> AgentRecord:
    record = _registry.get(agent_name)
    if record is None:
        record = AgentRecord(agent_name=agent_name)
        _registry[agent_name] = record
    return record


def _append_event(record: AgentRecord, event: dict[str, Any]) -> None:
    record.events.append(event)
    record.events = _trim_events(record.events)
    record.last_event = event
    record.touch()


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "time": _iso()}


@app.get("/agents")
async def list_agents() -> dict[str, Any]:
    return {"count": len(_registry), "agents": [record.snapshot() for record in _registry.values()]}


@app.get("/agents/online")
async def agents_online() -> dict[str, Any]:
    return {"agents": list(_registry.keys())}


@app.get("/agents/{agent_name}")
async def get_agent(agent_name: str) -> dict[str, Any]:
    record = _registry.get(agent_name)
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return record.snapshot()


@app.post("/agents/register")
async def register_agent(payload: AgentRegistration) -> dict[str, Any]:
    record = _get_or_create(payload.agent_name)
    record.host = payload.host or record.host
    record.pid = payload.pid or record.pid
    record.metadata.update(_jsonable(payload.metadata))
    record.scope = payload.scope
    record.status = "registered"
    _append_event(
        record,
        {
            "event": "register",
            "agent_name": record.agent_name,
            "scope": payload.scope,
            "workspace": str(agent_workspace(payload.agent_name)),
            "metadata": _jsonable(payload.metadata),
            "at": _iso(),
        },
    )
    return {"ok": True, "agent": record.snapshot()}


@app.post("/agents/{agent_name}/status")
async def update_status(agent_name: str, payload: AgentStatusUpdate) -> dict[str, Any]:
    record = _get_or_create(agent_name)
    record.status = payload.status
    if payload.phase is not None:
        record.phase = payload.phase
    if payload.step is not None:
        record.step = payload.step
    if payload.state is not None:
        record.state = payload.state
    record.metadata.update(_jsonable(payload.metadata))
    _append_event(
        record,
        {
            "event": "status",
            "status": payload.status,
            "phase": payload.phase,
            "step": payload.step,
            "state": _jsonable(payload.state),
            "at": _iso(),
        },
    )
    return {"ok": True, "agent": record.snapshot()}


@app.post("/agents/{agent_name}/event")
async def agent_event(agent_name: str, payload: AgentEvent) -> dict[str, Any]:
    record = _get_or_create(agent_name)
    if payload.phase is not None:
        record.phase = payload.phase
    if payload.state is not None:
        record.state = payload.state
    record.metadata.update(_jsonable(payload.metadata))
    event = payload.model_dump()
    event["state"] = _jsonable(payload.state)
    event["request"] = _jsonable(payload.request)
    event["tool_call"] = _jsonable(payload.tool_call)
    event["at"] = _iso()
    _append_event(record, event)
    return {"ok": True, "agent": record.snapshot()}


@app.post("/agents/{agent_name}/error")
async def agent_error(agent_name: str, payload: AgentErrorReport) -> dict[str, Any]:
    record = _get_or_create(agent_name)
    record.status = "error"
    if payload.phase is not None:
        record.phase = payload.phase
    record.last_error = payload.model_dump()
    record.metadata.update(_jsonable(payload.metadata))
    _append_event(
        record,
        {
            "event": "error",
            "error_type": payload.error_type,
            "error_message": payload.error_message,
            "traceback": payload.traceback,
            "phase": payload.phase,
            "at": _iso(),
        },
    )
    return {"ok": True, "agent": record.snapshot()}


@app.post("/agents/{agent_name}/stop")
async def stop_agent(agent_name: str) -> dict[str, Any]:
    record = _registry.get(agent_name)
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    record.status = "stopped"
    _append_event(record, {"event": "stop", "at": _iso()})
    return {"ok": True, "agent": record.snapshot()}


@app.get("/agents/peers/{agent_name}")
async def agent_peers(agent_name: str) -> dict[str, Any]:
    record = _registry.get(agent_name)
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    if record.scope is None:
        peers = [registered_name for registered_name in _registry if registered_name != agent_name]
    else:
        peers = [
            registered_name
            for registered_name in record.scope
            if registered_name in _registry and registered_name != agent_name
        ]
    return {"agent_name": agent_name, "peers": peers}


class SendRequest(BaseModel):
    message_id: str
    from_: str = Field(alias="from")
    to: str
    type: MessageType
    content: str | dict[str, Any]
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


@app.post("/send")
async def send_message(payload: SendRequest) -> dict[str, Any]:
    sender = _registry.get(payload.from_)
    sender_scope = sender.scope if sender else None

    message: AgentMail = {
        "message_id": payload.message_id,
        "from": payload.from_,
        "to": payload.to,
        "type": payload.type,
        "content": payload.content,
        "attachments": payload.attachments,
    }

    if sender_scope is not None and payload.to not in sender_scope:
        raise HTTPException(status_code=403, detail=f"agent '{payload.to}' is not in sender scope")
    routed = route_message_assets(message, str(agent_workspace(payload.to)))
    _mailboxes.setdefault(payload.to, []).append(routed)

    return {"ok": True, "message_id": payload.message_id}


@app.get("/recv/{agent_name}")
async def recv_messages(agent_name: str) -> dict[str, Any]:
    record = _registry.get(agent_name)
    receiver_scope = record.scope if record else None
    messages = _mailboxes.pop(agent_name, [])
    if receiver_scope is not None:
        messages = [message for message in messages if message.get("from") in receiver_scope]
    return {"agent_name": agent_name, "messages": messages}
