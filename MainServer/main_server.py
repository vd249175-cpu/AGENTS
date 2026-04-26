import asyncio
import os
import socket
import subprocess
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from MainServer.admin_config import (
    agent_spaces,
    clear_agent_chat_session,
    communication_allows,
    communication_edges,
    communication_peers,
    configured_scope,
    get_communication_config,
    get_agent_config,
    get_ui_state,
    load_config,
    remove_agent_config,
    replace_agent_config,
    replace_communication_config,
    replace_ui_state,
    resolve_scope,
    save_config,
    update_agent_config,
)
from MainServer.agent_templates import clear_agent_runtime, create_agent_from_template, list_agent_directories
from MainServer.agent_templates import delete_agent_directory, read_agent_brain_prompt, read_agent_runtime_config
from MainServer.agent_templates import write_agent_brain_prompt, write_agent_runtime_config
from MainServer.mail_router import route_message_assets
from MainServer.protocol import make_message
from MainServer.state import AgentMail, MessageType
from MainServer.user_chat import (
    UserChatRequest,
    UserChatRuntimeConfig,
    build_invoke_payload,
    build_mail_content,
    post_json,
)


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
    scope: Any = None


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

    model_config = {"extra": "allow"}


class AgentErrorReport(BaseModel):
    error_type: str | None = None
    error_message: str
    traceback: str | None = None
    phase: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentScopeUpdate(BaseModel):
    scope: Any = None


class CommunicationConfigUpdate(BaseModel):
    spaces: list[dict[str, Any]] = Field(default_factory=list)


class UiStateUpdate(BaseModel):
    agent_positions: dict[str, Any] = Field(default_factory=dict)
    chat_sessions: dict[str, Any] = Field(default_factory=dict)


class AgentChatConfigUpdate(BaseModel):
    thread_id: str = Field(default="default")
    run_id: str | None = None
    stream_mode: list[str] | str = Field(default_factory=lambda: ["messages", "updates", "custom"])
    version: str = "v2"


class AgentRuntimeClearRequest(BaseModel):
    include_store: bool = True
    include_mail: bool = True
    include_knowledge: bool = False
    include_checkpoints: bool = False


class AgentBrainPromptUpdate(BaseModel):
    content: str


class AgentServiceStartRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int | None = Field(default=None, ge=1, le=65535)
    main_server_url: str = "http://127.0.0.1:8000"


class CreateAgentRequest(BaseModel):
    agent_name: str
    source_agent: str = "SeedAgent"
    overwrite: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    scope: Any = None


def _record_service_url(record: "AgentRecord") -> str | None:
    metadata = record.metadata or {}
    for key in ("service_url", "base_url", "url"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.rstrip("/")
    port = metadata.get("service_port") or metadata.get("port")
    host = metadata.get("service_host") or record.host
    if host and port:
        return f"http://{host}:{port}".rstrip("/")
    return None


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
    scope: Any = None

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
            "resolved_scope": resolve_scope(self.scope),
            "communication_peers": communication_peers(self.agent_name),
            "communication_spaces": [space["id"] for space in agent_spaces(self.agent_name)],
            "workspace": str(agent_workspace(self.agent_name)),
        }

    def touch(self) -> None:
        self.updated_at = _now()


app = FastAPI(title="LANGVIDEO MainServer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_registry: dict[str, AgentRecord] = {}
_mailboxes: dict[str, list[AgentMail]] = {}
_mail_log: list[dict[str, Any]] = []
_managed_agent_processes: dict[str, subprocess.Popen] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "tests" / "logs"


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


def _append_mail_log(message: AgentMail, *, mode: str) -> None:
    _mail_log.append({"mode": mode, "message": _jsonable(message), "at": _iso()})
    if len(_mail_log) > 300:
        del _mail_log[:-300]


def _clear_agent_activity_history(agent_name: str) -> dict[str, Any]:
    record = _registry.get(agent_name)
    cleared_events = 0
    if record is not None:
        cleared_events = len(record.events)
        record.events = []
        record.last_event = None
        record.touch()
    before_mail = len(_mail_log)
    _mail_log[:] = [
        item
        for item in _mail_log
        if item.get("message", {}).get("from") != agent_name and item.get("message", {}).get("to") != agent_name
    ]
    ui = clear_agent_chat_session(agent_name)
    return {
        "agent_events": cleared_events,
        "mail_events": before_mail - len(_mail_log),
        "ui": ui,
    }


def _agent_is_busy(record: AgentRecord) -> bool:
    if record.status != "running":
        return False
    ready_phases = {None, "startup", "ready", "after_agent"}
    if record.phase in ready_phases:
        return False
    return True


def _mail_wake_payload(agent_name: str, message: AgentMail) -> dict[str, Any]:
    message_id = str(message.get("message_id") or "mail")
    source = str(message.get("from") or "unknown")
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "You have new mail in your MainServer inbox. "
                    "Run your receive_messages middleware, read the inbox, and handle the mail. "
                    f"message_id={message_id}; from={source}."
                ),
            }
        ],
        "session_id": f"mail-{agent_name}",
        "run_id": f"mail-{message_id}",
        "stream_mode": ["updates", "custom", "messages"],
        "version": "v2",
    }


def _queue_mail_wake(background_tasks: BackgroundTasks, agent_name: str, message: AgentMail, *, mode: str) -> bool:
    record = _registry.get(agent_name)
    if record is None:
        return False
    service_url = _record_service_url(record)
    if service_url is None:
        _append_event(
            record,
            {
                "event": "mail_wake_skipped",
                "reason": "missing_service_url",
                "message_id": message.get("message_id"),
                "mode": mode,
                "at": _iso(),
            },
        )
        return False
    _append_event(
        record,
        {
            "event": "mail_wake_scheduled",
            "message_id": message.get("message_id"),
            "mode": mode,
            "service_url": service_url,
            "busy_at_schedule": _agent_is_busy(record),
            "at": _iso(),
        },
    )
    background_tasks.add_task(_wake_agent_for_mail, agent_name, service_url, message, mode)
    return True


async def _wake_agent_for_mail(agent_name: str, service_url: str, message: AgentMail, mode: str) -> None:
    record = _get_or_create(agent_name)
    payload = _mail_wake_payload(agent_name, message)
    _append_event(
        record,
        {
            "event": "mail_wake_start",
            "message_id": message.get("message_id"),
            "mode": mode,
            "run_id": payload["run_id"],
            "at": _iso(),
        },
    )
    response: dict[str, Any] | None = None
    for attempt in range(1, 7):
        try:
            response = await asyncio.to_thread(post_json, f"{service_url}/invoke", payload, 600.0)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 409 and attempt < 6:
                _append_event(
                    record,
                    {
                        "event": "mail_wake_retry",
                        "message_id": message.get("message_id"),
                        "mode": mode,
                        "attempt": attempt,
                        "reason": body or exc.reason,
                        "at": _iso(),
                    },
                )
                await asyncio.sleep(1.0)
                continue
            _append_event(
                record,
                {
                    "event": "mail_wake_error",
                    "message_id": message.get("message_id"),
                    "mode": mode,
                    "status_code": exc.code,
                    "error": body or exc.reason,
                    "at": _iso(),
                },
            )
            return
        except urllib.error.URLError as exc:
            _append_event(
                record,
                {
                    "event": "mail_wake_error",
                    "message_id": message.get("message_id"),
                    "mode": mode,
                    "error": f"agent service unreachable: {exc.reason}",
                    "at": _iso(),
                },
            )
            return
        except Exception as exc:
            _append_event(
                record,
                {
                    "event": "mail_wake_error",
                    "message_id": message.get("message_id"),
                    "mode": mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": _iso(),
                },
            )
            return
    if response is None:
        _append_event(
            record,
            {
                "event": "mail_wake_error",
                "message_id": message.get("message_id"),
                "mode": mode,
                "error": "wake retry exhausted",
                "at": _iso(),
            },
        )
        return
    _append_event(
        record,
        {
            "event": "mail_wake_success",
            "message_id": message.get("message_id"),
            "mode": mode,
            "reply": str(response.get("reply") or "")[:500],
            "at": _iso(),
        },
    )


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def _free_agent_port(host: str, preferred: int | None = None) -> int:
    start = preferred or 8010
    for port in range(start, min(start + 200, 65536)):
        if _is_port_available(host, port):
            return port
    raise RuntimeError("no free agent service port found")


def _agent_module_exists(agent_name: str) -> bool:
    return (PROJECT_ROOT / "Deepagents" / agent_name / "AgentServer" / "__main__.py").exists()


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "time": _iso()}


@app.get("/agents")
async def list_agents() -> dict[str, Any]:
    return {"count": len(_registry), "agents": [record.snapshot() for record in _registry.values()]}


@app.get("/admin/agents/available")
async def admin_available_agents() -> dict[str, Any]:
    available: dict[str, dict[str, Any]] = {}
    for item in list_agent_directories():
        agent_name = item["agent_name"]
        agent_config = get_agent_config(agent_name)
        available[agent_name] = {
            **item,
            "status": "available",
            "phase": None,
            "step": None,
            "metadata": {},
            "scope": agent_config.get("scope"),
            "resolved_scope": resolve_scope(agent_config.get("scope")),
            "communication_peers": communication_peers(agent_name),
            "communication_spaces": [space["id"] for space in agent_spaces(agent_name)],
            "config": agent_config,
            "registered": False,
            "source": "filesystem",
        }
    for agent_name, record in _registry.items():
        snapshot = record.snapshot()
        existing = available.get(agent_name, {})
        available[agent_name] = {
            **existing,
            **snapshot,
            "registered": True,
            "source": "registry",
            "config": get_agent_config(agent_name),
        }
    agents = list(available.values())
    agents.sort(key=lambda item: str(item.get("agent_name") or "").lower())
    return {
        "count": len(agents),
        "agents": agents,
        "communication": get_communication_config(),
        "edges": communication_edges([str(agent.get("agent_name")) for agent in agents]),
        "ui": get_ui_state(),
    }


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
            "scope": record.scope,
            "scope_source": "registration_legacy",
            "resolved_scope": resolve_scope(record.scope),
            "communication_peers": communication_peers(record.agent_name),
            "communication_spaces": [space["id"] for space in agent_spaces(record.agent_name)],
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
    communication_peer_names = communication_peers(agent_name)
    if communication_peer_names is None:
        peers = [registered_name for registered_name in _registry if registered_name != agent_name]
    else:
        peers = [
            registered_name
            for registered_name in communication_peer_names
            if registered_name in _registry and registered_name != agent_name
        ]
    return {
        "agent_name": agent_name,
        "peers": peers,
        "communication_peers": communication_peer_names,
        "communication_spaces": [space["id"] for space in agent_spaces(agent_name)],
        "scope": record.scope,
        "resolved_scope": resolve_scope(record.scope),
    }


class SendRequest(BaseModel):
    message_id: str
    from_: str = Field(alias="from")
    to: str
    type: MessageType
    content: str | dict[str, Any]
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


@app.post("/send")
async def send_message(payload: SendRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    message: AgentMail = {
        "message_id": payload.message_id,
        "from": payload.from_,
        "to": payload.to,
        "type": payload.type,
        "content": payload.content,
        "attachments": payload.attachments,
    }

    if payload.from_ != "user" and not communication_allows(payload.from_, payload.to):
        raise HTTPException(status_code=403, detail=f"agent '{payload.to}' is not reachable from '{payload.from_}'")
    routed = route_message_assets(message, str(agent_workspace(payload.to)))
    _mailboxes.setdefault(payload.to, []).append(routed)
    _append_mail_log(routed, mode="agent")
    wake_scheduled = _queue_mail_wake(background_tasks, payload.to, routed, mode="agent")

    return {"ok": True, "message_id": payload.message_id, "wake_scheduled": wake_scheduled}


@app.get("/recv/{agent_name}")
async def recv_messages(agent_name: str) -> dict[str, Any]:
    messages = _mailboxes.pop(agent_name, [])
    messages = [
        message
        for message in messages
        if str(message.get("from") or "") == "user"
        or communication_allows(str(message.get("from") or ""), agent_name)
    ]
    return {"agent_name": agent_name, "messages": messages}


def _agent_chat_config(agent_name: str) -> UserChatRuntimeConfig:
    config = get_agent_config(agent_name)
    raw_chat = config.get("chat", {})
    if not isinstance(raw_chat, dict):
        raw_chat = {}
    return UserChatRuntimeConfig.model_validate(raw_chat)


@app.get("/user/chat/config/{agent_name}")
async def get_user_chat_config(agent_name: str) -> dict[str, Any]:
    chat_config = _agent_chat_config(agent_name)
    return {"agent_name": agent_name, "chat": chat_config.model_dump()}


@app.put("/user/chat/config/{agent_name}")
async def set_user_chat_config(agent_name: str, payload: AgentChatConfigUpdate) -> dict[str, Any]:
    chat_config = UserChatRuntimeConfig.model_validate(payload.model_dump())
    config = update_agent_config(agent_name, {"chat": chat_config.model_dump()})
    return {"ok": True, "agent_name": agent_name, "chat": chat_config.model_dump(), "config": config}


@app.post("/user/chat")
async def user_chat(payload: UserChatRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    if payload.mode == "mail":
        try:
            mail_content = build_mail_content(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        message = make_message(
            src=payload.from_,
            dst=payload.agent_name,
            msg_type=payload.type,
            content=mail_content,
            attachments=payload.attachments,
        )
        routed = route_message_assets(message, str(agent_workspace(payload.agent_name)))
        _mailboxes.setdefault(payload.agent_name, []).append(routed)
        _append_mail_log(routed, mode="user")
        wake_scheduled = _queue_mail_wake(background_tasks, payload.agent_name, routed, mode="user")
        return {
            "ok": True,
            "mode": "mail",
            "agent_name": payload.agent_name,
            "message_id": message["message_id"],
            "message": routed,
            "wake_scheduled": wake_scheduled,
        }

    record = _registry.get(payload.agent_name)
    if record is None:
        raise HTTPException(status_code=404, detail="agent not found")
    service_url = _record_service_url(record)
    if service_url is None:
        raise HTTPException(
            status_code=409,
            detail="agent has no service_url metadata; restart the AgentServer or register service_url manually",
        )

    chat_config = _agent_chat_config(payload.agent_name)
    try:
        invoke_payload = build_invoke_payload(payload, chat_config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        response = await asyncio.to_thread(
            post_json,
            f"{service_url}/invoke",
            invoke_payload,
            payload.timeout,
        )
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=error_body or exc.reason) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"agent service unreachable: {exc.reason}") from exc
    return {
        "ok": True,
        "mode": "direct",
        "agent_name": payload.agent_name,
        "service_url": service_url,
        "chat": {
            "thread_id": invoke_payload["session_id"],
            "run_id": invoke_payload["run_id"],
            "stream_mode": invoke_payload["stream_mode"],
            "version": invoke_payload["version"],
        },
        "user_message": invoke_payload["messages"][-1],
        "response": response,
    }


@app.get("/admin/agents/config")
async def admin_get_agents_config() -> dict[str, Any]:
    return load_config()


@app.post("/admin/agents/create")
async def admin_create_agent(payload: CreateAgentRequest) -> dict[str, Any]:
    try:
        result = create_agent_from_template(
            agent_name=payload.agent_name,
            source_agent=payload.source_agent,
            overwrite=payload.overwrite,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    config_patch = dict(payload.config)
    config_patch.setdefault("template", payload.source_agent)
    if payload.scope is not None:
        config_patch["scope"] = payload.scope
    config = update_agent_config(payload.agent_name, config_patch)
    return {"ok": True, "agent": result, "config": config}


@app.delete("/admin/agents/{agent_name}")
async def admin_delete_agent(agent_name: str) -> dict[str, Any]:
    try:
        result = delete_agent_directory(agent_name=agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    removed_config = remove_agent_config(agent_name)
    _registry.pop(agent_name, None)
    _mailboxes.pop(agent_name, None)
    return {"ok": True, "agent": result, "removed_config": removed_config}


@app.put("/admin/agents/config")
async def admin_replace_agents_config(payload: dict[str, Any]) -> dict[str, Any]:
    return save_config(payload)


@app.get("/admin/communication")
async def admin_get_communication() -> dict[str, Any]:
    agents = [item["agent_name"] for item in list_agent_directories()]
    return {
        "communication": get_communication_config(),
        "edges": communication_edges(agents),
        "ui": get_ui_state(),
    }


@app.put("/admin/communication")
async def admin_replace_communication(payload: CommunicationConfigUpdate) -> dict[str, Any]:
    communication = replace_communication_config(payload.model_dump())
    records = list(_registry.values())
    for record in records:
        _append_event(
            record,
            {
                "event": "communication_update",
                "communication_peers": communication_peers(record.agent_name),
                "communication_spaces": [space["id"] for space in agent_spaces(record.agent_name)],
                "at": _iso(),
            },
        )
    return {"ok": True, "communication": communication, "edges": communication_edges()}


@app.get("/admin/ui-state")
async def admin_get_ui_state() -> dict[str, Any]:
    return {"ok": True, "ui": get_ui_state()}


@app.put("/admin/ui-state")
async def admin_replace_ui_state(payload: UiStateUpdate) -> dict[str, Any]:
    return {"ok": True, "ui": replace_ui_state(payload.model_dump())}


@app.get("/admin/monitor")
async def admin_monitor() -> dict[str, Any]:
    agents = [record.snapshot() for record in _registry.values()]
    agents.sort(key=lambda item: str(item.get("agent_name") or "").lower())
    mailbox_counts = {agent_name: len(messages) for agent_name, messages in _mailboxes.items()}
    return {
        "agents": agents,
        "mailbox_counts": mailbox_counts,
        "recent_mail": _mail_log[-80:],
        "communication": get_communication_config(),
        "edges": communication_edges([str(agent.get("agent_name")) for agent in agents]),
    }


@app.get("/admin/agents/{agent_name}/config")
async def admin_get_agent_config(agent_name: str) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "config": get_agent_config(agent_name),
        "registered": _registry.get(agent_name).snapshot() if agent_name in _registry else None,
    }


@app.get("/admin/agents/{agent_name}/runtime-config")
async def admin_get_agent_runtime_config(agent_name: str) -> dict[str, Any]:
    try:
        result = read_agent_runtime_config(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "agent_name": agent_name, **result}


@app.patch("/admin/agents/{agent_name}/runtime-config")
async def admin_patch_agent_runtime_config(agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = write_agent_runtime_config(agent_name, payload, merge=True)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "agent_name": agent_name, **result}


@app.put("/admin/agents/{agent_name}/runtime-config")
async def admin_replace_agent_runtime_config(agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = write_agent_runtime_config(agent_name, payload, merge=False)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "agent_name": agent_name, **result}


@app.get("/admin/agents/{agent_name}/brain")
async def admin_get_agent_brain_prompt(agent_name: str) -> dict[str, Any]:
    try:
        result = read_agent_brain_prompt(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "agent_name": agent_name, **result}


@app.put("/admin/agents/{agent_name}/brain")
async def admin_replace_agent_brain_prompt(agent_name: str, payload: AgentBrainPromptUpdate) -> dict[str, Any]:
    try:
        result = write_agent_brain_prompt(agent_name, payload.content)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "agent_name": agent_name, **result}


@app.patch("/admin/agents/{agent_name}/config")
async def admin_patch_agent_config(agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    config = update_agent_config(agent_name, payload)
    record = _registry.get(agent_name)
    if record is not None and "scope" in payload:
        record.scope = payload.get("scope")
        _append_event(
            record,
            {
                "event": "scope_update",
                "scope": record.scope,
                "resolved_scope": resolve_scope(record.scope),
                "source": "mainserver_config",
                "at": _iso(),
            },
        )
    return {"ok": True, "agent_name": agent_name, "config": config}


@app.put("/admin/agents/{agent_name}/config")
async def admin_replace_agent_config(agent_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    config = replace_agent_config(agent_name, payload)
    record = _registry.get(agent_name)
    if record is not None and "scope" in config:
        record.scope = config.get("scope")
    return {"ok": True, "agent_name": agent_name, "config": config}


@app.get("/admin/agents/{agent_name}/scope")
async def admin_get_agent_scope(agent_name: str) -> dict[str, Any]:
    record = _registry.get(agent_name)
    if record is not None:
        scope = record.scope
        source = "registry"
    else:
        has_configured_scope, scope = configured_scope(agent_name)
        source = "mainserver_config" if has_configured_scope else "default_all"
    return {
        "agent_name": agent_name,
        "scope": scope,
        "resolved_scope": resolve_scope(scope),
        "source": source,
    }


@app.put("/admin/agents/{agent_name}/scope")
async def admin_set_agent_scope(agent_name: str, payload: AgentScopeUpdate) -> dict[str, Any]:
    config = update_agent_config(agent_name, {"scope": payload.scope})
    record = _registry.get(agent_name)
    if record is not None:
        record.scope = payload.scope
        _append_event(
            record,
            {
                "event": "scope_update",
                "scope": payload.scope,
                "resolved_scope": resolve_scope(payload.scope),
                "source": "mainserver_config",
                "at": _iso(),
            },
        )
    return {
        "ok": True,
        "agent_name": agent_name,
        "config": config,
        "scope": payload.scope,
        "resolved_scope": resolve_scope(payload.scope),
    }


@app.post("/admin/agents/{agent_name}/runtime/clear")
async def admin_clear_agent_runtime(agent_name: str, payload: AgentRuntimeClearRequest) -> dict[str, Any]:
    try:
        result = clear_agent_runtime(
            agent_name=agent_name,
            include_store=payload.include_store,
            include_mail=payload.include_mail,
            include_knowledge=payload.include_knowledge,
            include_checkpoints=payload.include_checkpoints,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    history = _clear_agent_activity_history(agent_name)
    return {"ok": True, "agent_name": agent_name, "runtime": result, "history": history}


@app.post("/admin/agents/{agent_name}/service/start")
async def admin_start_agent_service(agent_name: str, payload: AgentServiceStartRequest) -> dict[str, Any]:
    if not _agent_module_exists(agent_name):
        raise HTTPException(status_code=404, detail=f"agent service module not found: {agent_name}")

    process = _managed_agent_processes.get(agent_name)
    if process is not None and process.poll() is None:
        record = _registry.get(agent_name)
        return {
            "ok": True,
            "agent_name": agent_name,
            "already_running": True,
            "pid": process.pid,
            "agent": record.snapshot() if record else None,
        }

    try:
        port = _free_agent_port(payload.host, payload.port)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"agent_service_{agent_name}.log"
    log_handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "AGENT_NAME": agent_name,
            "AGENT_HOST": payload.host,
            "AGENT_PORT": str(port),
            "MAIN_SERVER_URL": payload.main_server_url.rstrip("/"),
        }
    )
    try:
        process = subprocess.Popen(
            ["uv", "run", "python", "-m", f"Deepagents.{agent_name}.AgentServer"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_handle.close()
        raise HTTPException(status_code=500, detail=f"failed to start agent service: {exc}") from exc

    _managed_agent_processes[agent_name] = process
    record = _get_or_create(agent_name)
    record.status = "starting"
    record.metadata.update(
        {
            "service_url": f"http://{payload.host}:{port}",
            "service_host": payload.host,
            "service_port": port,
            "managed_pid": process.pid,
            "managed_log": str(log_path),
        }
    )
    _append_event(
        record,
        {
            "event": "service_start_requested",
            "pid": process.pid,
            "service_url": f"http://{payload.host}:{port}",
            "log": str(log_path),
            "at": _iso(),
        },
    )
    return {
        "ok": True,
        "agent_name": agent_name,
        "pid": process.pid,
        "service_url": f"http://{payload.host}:{port}",
        "log": str(log_path),
        "agent": record.snapshot(),
    }


@app.post("/admin/agents/{agent_name}/service/stop")
async def admin_stop_agent_service(agent_name: str) -> dict[str, Any]:
    process = _managed_agent_processes.pop(agent_name, None)
    stopped = False
    if process is not None and process.poll() is None:
        process.terminate()
        stopped = True
    record = _registry.get(agent_name)
    if record is not None:
        record.status = "stopped"
        _append_event(record, {"event": "service_stop_requested", "managed": stopped, "at": _iso()})
    return {
        "ok": True,
        "agent_name": agent_name,
        "stopped": stopped,
        "agent": record.snapshot() if record else None,
    }
