import asyncio
import json
import os
import socket
import sys
import traceback
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Deepagents.SeedAgent.Agent.MainAgent import (
    AGENT_SPEC,
    Config as SeedAgentConfig,
    WORKSPACE_ROOT,
    SeedMainAgent,
    abuild_main_agent,
)
from MainServer.comm import AgentComm


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        return data.astimezone(timezone.utc).isoformat()
    return repr(data)


def _post_json(url: str, payload: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


STREAM_TYPES = {"messages", "updates", "custom"}
AGENT_SERVER_ROOT = Path(__file__).resolve().parent
DEFAULT_SERVICE_CONFIG = AGENT_SERVER_ROOT / "ServiceConfig.json"


def _env_prefix(agent_name: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(agent_name):
        if char.isupper() and index > 0:
            chars.append("_")
        chars.append(char.upper())
    return "".join(chars)


class ServiceConfig(BaseModel):
    agent_name: str = Field(default=AGENT_SPEC.name)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8010, ge=1, le=65535)
    main_server_url: str = Field(default="http://127.0.0.1:8000")


def load_service_config(source: str | Path | None = None) -> ServiceConfig:
    raw_source = source or os.getenv("LANGVIDEO_AGENT_SERVICE_CONFIG") or DEFAULT_SERVICE_CONFIG
    path = Path(raw_source).expanduser()
    data: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid service config: {path}")
        data.update(loaded)

    prefix = _env_prefix(str(data.get("agent_name", AGENT_SPEC.name)))
    env_host_key = f"{prefix}_HOST"
    env_port_key = f"{prefix}_PORT"
    data["agent_name"] = os.getenv("AGENT_NAME") or data.get("agent_name", AGENT_SPEC.name)
    data["host"] = os.getenv(env_host_key) or os.getenv("AGENT_HOST") or data.get("host", "127.0.0.1")
    data["port"] = int(os.getenv(env_port_key) or os.getenv("AGENT_PORT") or data.get("port", 8010))
    data["main_server_url"] = os.getenv("MAIN_SERVER_URL") or data.get(
        "main_server_url", "http://127.0.0.1:8000"
    )
    return ServiceConfig.model_validate(data)


class InvokeRequest(BaseModel):
    messages: list[dict[str, Any]]
    session_id: str = Field(default="default")
    run_id: str = Field(default="default-run")
    stream_mode: list[str] | str = Field(default_factory=lambda: ["messages", "updates", "custom"])
    version: str = "v2"


class AgentErrorPayload(BaseModel):
    error_type: str | None = None
    error_message: str
    traceback: str | None = None
    phase: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEventPayload(BaseModel):
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


@dataclass
class LocalAgentState:
    agent_name: str
    host: str | None = None
    pid: int | None = None
    status: str = "starting"
    phase: str | None = None
    step: str | None = None
    state: Any = None
    last_error: dict[str, Any] | None = None
    last_sync_error: str | None = None
    started_at: str = field(default_factory=_now)
    last_synced_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    running: bool = False
    last_status_signature: str | None = None

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
            "last_sync_error": self.last_sync_error,
            "started_at": self.started_at,
            "last_synced_at": self.last_synced_at,
            "metadata": _jsonable(self.metadata),
            "running": self.running,
            "time": _now(),
        }


class MainServerClient:
    def __init__(self, base_url: str, agent_name: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_name = agent_name
        self.timeout = min(timeout, 5.0)

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(f"{self.base_url}/agents/register", payload, timeout=self.timeout)

    def status(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(f"{self.base_url}/agents/{self.agent_name}/status", payload, timeout=self.timeout)

    def event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(f"{self.base_url}/agents/{self.agent_name}/event", payload, timeout=self.timeout)

    def error(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _post_json(f"{self.base_url}/agents/{self.agent_name}/error", payload, timeout=self.timeout)

    def stop(self) -> dict[str, Any]:
        return _post_json(f"{self.base_url}/agents/{self.agent_name}/stop", {}, timeout=self.timeout)


class AgentObserver:
    def __init__(
        self,
        main_server_url: str,
        agent_name: str | None = None,
        scope: Any = None,
    ) -> None:
        self.agent_name = agent_name or os.getenv("AGENT_NAME") or AGENT_SPEC.name
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.scope = scope
        self.client = MainServerClient(main_server_url, self.agent_name)
        self.state = LocalAgentState(
            agent_name=self.agent_name,
            host=self.host,
            pid=self.pid,
        )

    def _base_payload(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "host": self.host,
            "pid": self.pid,
            "metadata": self.state.metadata,
            "scope": self.scope,
        }

    async def register(self) -> None:
        try:
            await asyncio.to_thread(self.client.register, self._base_payload())
            self.state.last_synced_at = _now()
        except Exception as exc:
            self.state.last_sync_error = f"{type(exc).__name__}: {exc}"

    async def set_status(
        self,
        status: str,
        *,
        phase: str | None = None,
        step: str | None = None,
        state: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.state.status = status
        self.state.phase = phase
        self.state.step = step
        self.state.state = state
        if metadata:
            self.state.metadata.update(metadata)
        signature = _status_signature(
            status=status,
            phase=phase,
            step=step,
            state=state,
            metadata=metadata,
        )
        if self.state.last_status_signature == signature:
            return
        payload = self._base_payload() | {
            "status": status,
            "phase": phase,
            "step": step,
            "state": _jsonable(state),
            "metadata": _jsonable(metadata or {}),
        }
        try:
            await asyncio.to_thread(self.client.status, payload)
            self.state.last_synced_at = _now()
            self.state.last_sync_error = None
            self.state.last_status_signature = signature
        except Exception as exc:
            self.state.last_sync_error = f"{type(exc).__name__}: {exc}"

    async def report_event(self, payload: dict[str, Any]) -> None:
        event = _jsonable(payload)
        try:
            await asyncio.to_thread(self.client.event, event)
            self.state.last_synced_at = _now()
        except Exception as exc:
            self.state.last_sync_error = f"{type(exc).__name__}: {exc}"

    async def report_error(self, payload: dict[str, Any]) -> None:
        error = _jsonable(payload)
        self.state.status = "error"
        self.state.last_error = error
        try:
            await asyncio.to_thread(self.client.error, error)
            self.state.last_synced_at = _now()
        except Exception as exc:
            self.state.last_sync_error = f"{type(exc).__name__}: {exc}"

    async def stop(self) -> None:
        self.state.running = False
        try:
            await asyncio.to_thread(self.client.stop)
        except Exception as exc:
            self.state.last_sync_error = f"{type(exc).__name__}: {exc}"


class SeedAgentRuntime:
    def __init__(self) -> None:
        self.main_agent: SeedMainAgent | None = None

    async def astream(
        self,
        *,
        messages: list[dict[str, Any]],
        session_id: str = "default",
        run_id: str = "default-run",
        stream_mode: list[str] | str = "custom",
        version: str = "v2",
    ):
        if self.main_agent is None:
            raise RuntimeError("main_agent is not initialized")
        async for chunk in self.main_agent.astream(
            messages=messages,
            session_id=session_id,
            run_id=run_id,
            stream_mode=stream_mode,
            version=version,
        ):
            yield chunk


def _normalize_stream_chunk(chunk: Any) -> tuple[str, Any] | None:
    if isinstance(chunk, dict):
        chunk_type = chunk.get("type")
        if isinstance(chunk_type, str):
            return chunk_type, chunk.get("data")
        return None
    if isinstance(chunk, (tuple, list)) and len(chunk) == 2:
        first, second = chunk
        if isinstance(first, str) and first in STREAM_TYPES:
            return first, second
        if isinstance(second, str) and second in STREAM_TYPES:
            return second, first
    return None


def _status_signature(
    *,
    status: str,
    phase: str | None,
    step: str | None,
    state: Any,
    metadata: dict[str, Any] | None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "phase": phase,
            "step": step,
            "state": _jsonable(state),
            "metadata": _jsonable(metadata or {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _guess_step(chunk_type: str, data: Any) -> str | None:
    if chunk_type != "updates":
        return None
    if isinstance(data, dict) and data:
        keys = [str(key) for key in data.keys()]
        if len(keys) == 1:
            return keys[0]
        if "messages" in data and len(keys) > 1:
            keys = [key for key in keys if key != "messages"]
            if len(keys) == 1:
                return keys[0]
    return None


def _parse_scope() -> Any:
    raw = os.getenv("AGENT_SCOPE")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in raw.split(",") if item.strip()]


def create_app(scope: Any = None) -> FastAPI:
    service_config = load_service_config()
    resolved_scope = scope if scope is not None else _parse_scope()
    main_server_url = service_config.main_server_url
    observer = AgentObserver(main_server_url, agent_name=service_config.agent_name, scope=resolved_scope)

    def _sync_exception_hook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        payload = {
            "error_type": exc_type.__name__,
            "error_message": str(exc),
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            "phase": observer.state.phase,
            "metadata": observer.state.metadata,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(observer.report_error(payload))
        except RuntimeError:
            pass

    def _loop_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception") or RuntimeError(context.get("message", "asyncio loop error"))
        loop.create_task(
            observer.report_error(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    "phase": observer.state.phase,
                    "metadata": observer.state.metadata,
                }
            )
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sys.excepthook = _sync_exception_hook
        asyncio.get_event_loop().set_exception_handler(_loop_exception_handler)
        app.state.runtime = SeedAgentRuntime()
        app.state.comm = AgentComm(main_server_url, observer.agent_name)
        agent_config = SeedAgentConfig.load_config_seed_agent()
        if agent_config.agentName != observer.agent_name:
            agent_config = agent_config.model_copy(update={"agentName": observer.agent_name})
        app.state.runtime.main_agent = await abuild_main_agent(comm=app.state.comm, config=agent_config)
        await observer.register()
        await observer.set_status("running", phase="startup", step="ready")
        try:
            yield
        finally:
            if app.state.runtime.main_agent is not None:
                await app.state.runtime.main_agent.aclose()
            await observer.set_status("stopped", phase="shutdown", step="done")
            await observer.stop()

    app = FastAPI(title=f"{observer.agent_name} Service", version="1.0.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "time": _now(), "agent": observer.agent_name}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return observer.state.snapshot()

    @app.post("/event")
    async def publish_event(payload: AgentEventPayload) -> dict[str, Any]:
        event = payload.model_dump()
        event["state"] = _jsonable(payload.state)
        await observer.report_event(event)
        return {"ok": True, "agent": observer.state.snapshot(), "event": _jsonable(event)}

    @app.post("/error")
    async def publish_error(payload: AgentErrorPayload) -> dict[str, Any]:
        error = payload.model_dump()
        await observer.report_error(error)
        return {"ok": True, "agent": observer.state.snapshot(), "error": _jsonable(error)}

    @app.post("/invoke")
    async def invoke(payload: InvokeRequest) -> dict[str, Any]:
        if not payload.messages:
            raise HTTPException(status_code=400, detail="messages is required")
        if observer.state.running:
            raise HTTPException(status_code=409, detail="agent is already running")

        observer.state.running = True
        await observer.set_status("running", phase="before_agent", state={"messages": payload.messages})
        collected_chunks: list[Any] = []
        try:
            async for chunk in app.state.runtime.astream(
                messages=payload.messages,
                session_id=payload.session_id,
                run_id=payload.run_id,
                stream_mode=payload.stream_mode,
                version=payload.version,
            ):
                collected_chunks.append(_jsonable(chunk))
                normalized = _normalize_stream_chunk(chunk)
                if normalized is None:
                    continue
                chunk_type, chunk_data = normalized
                if chunk_type == "custom":
                    if isinstance(chunk_data, dict):
                        await observer.report_event(chunk_data)
                    continue
                if chunk_type in {"messages", "updates"}:
                    await observer.set_status(
                        "running",
                        phase=chunk_type,
                        step=_guess_step(chunk_type, chunk_data),
                        state=chunk_data,
                        metadata={"stream_type": chunk_type},
                    )
            await observer.set_status("idle", phase="after_agent", state={"messages": payload.messages})
            return {"ok": True, "chunks": collected_chunks, "agent": observer.state.snapshot()}
        except Exception as exc:
            await observer.report_error(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    "phase": observer.state.phase,
                    "metadata": observer.state.metadata,
                }
            )
            raise
        finally:
            observer.state.running = False

    class CommSendRequest(BaseModel):
        dst: str
        content: str
        type: str = "message"

    @app.post("/comm/send")
    async def comm_send(payload: CommSendRequest) -> dict[str, Any]:
        return await asyncio.to_thread(app.state.comm.send, payload.dst, payload.content, payload.type)

    @app.get("/comm/recv")
    async def comm_recv() -> dict[str, Any]:
        messages = await asyncio.to_thread(app.state.comm.recv)
        return {"messages": messages}

    @app.get("/comm/peers")
    async def comm_peers() -> dict[str, Any]:
        peers = await asyncio.to_thread(app.state.comm.peers)
        return {"peers": peers}

    return app


app = create_app()


def main() -> int:
    import uvicorn

    service_config = load_service_config()
    uvicorn.run(
        "Deepagents.SeedAgent.AgentServer.service:app",
        host=service_config.host,
        port=service_config.port,
        reload=False,
        access_log=False,
    )
    return 0
