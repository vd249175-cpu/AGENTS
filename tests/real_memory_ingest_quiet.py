import json
import secrets
import string
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


ROOT = Path("/Users/apexwave/Desktop/Agents")
MAIN_SERVER = "http://127.0.0.1:8000"
SEED_URL = "http://127.0.0.1:8010"
SEED_WORKSPACE = ROOT / "Deepagents" / "SeedAgent" / "workspace"
LOG_DIR = ROOT / "tests" / "logs"
ALPHABET = string.ascii_lowercase + string.digits
INNER_TOOL_NAMES = (
    "create_chunk_document",
    "insert_chunks",
    "update_chunks",
    "delete_chunks",
    "list_chunk_documents",
    "query_chunk_positions",
    "graph_create_nodes",
    "graph_update_node",
    "graph_delete_nodes",
    "read_nodes",
    "keyword_recall",
    "graph_distance_recall",
    "graph_mark_useful",
    "graph_mark_blocked",
    "graph_clear_blocked",
)


def short_id(length: int = 4) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def get_json(url: str, timeout: float = 10.0) -> Any:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def post_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def compact_value(value: Any, *, max_len: int = 500) -> Any:
    if isinstance(value, dict):
        return {
            key: compact_value(item, max_len=max_len)
            for key, item in value.items()
            if key not in {"embedding", "keyword_vectors"}
        }
    if isinstance(value, list):
        if len(value) > 20:
            return [compact_value(item, max_len=max_len) for item in value[:20]] + [{"truncated": len(value) - 20}]
        return [compact_value(item, max_len=max_len) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if len(text) > max_len:
            return f"{text[: max_len - 1].rstrip()}…"
        return text
    return value


class JsonlLog:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = LOG_DIR / f"real_memory_ingest_{stamp}_{short_id()}.jsonl"
        self.full_path = LOG_DIR / f"real_memory_ingest_{stamp}_{short_id()}_full.jsonl"

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        full_record = {
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **payload,
        }
        with self.full_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(full_record, ensure_ascii=False, default=str) + "\n")
        compact_record = self.compact_record(full_record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(compact_record, ensure_ascii=False, default=str) + "\n")

    def compact_record(self, record: dict[str, Any]) -> dict[str, Any]:
        kind = record.get("kind")
        base = {"at": record.get("at"), "kind": kind}
        if kind == "event":
            event = record.get("event")
            return {
                **base,
                "label": record.get("label"),
                "event": compact_tool_event(event) or compact_step_event(event) or compact_inner_event(event) or compact_generic_event(event),
            }
        if kind == "stage_end":
            return {
                **base,
                "label": record.get("label"),
                "ok": record.get("ok"),
                "error": record.get("error"),
                "chunks": record.get("chunks"),
                "milestones": record.get("milestones"),
            }
        if kind == "neo4j_check":
            return {
                **base,
                "ok": record.get("ok"),
                "document_name": record.get("document_name"),
                "expected_run_id": record.get("expected_run_id"),
                "chunk_count": record.get("chunk_count"),
                "sample": compact_value(record.get("sample")),
                "error": record.get("error"),
            }
        if kind == "verification":
            return {
                **base,
                "ok": record.get("ok"),
                "checks": record.get("checks"),
                "reason": record.get("reason"),
            }
        if kind == "summary":
            return {**base, "lines": record.get("lines"), "final_ok": record.get("final_ok")}
        return compact_value(record)


class Summary:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, label: str, text: str) -> None:
        line = f"{label}: {text}"
        self.lines.append(line)
        print(line, flush=True)


def compact_tool_event(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict) or event.get("type") != "tool":
        return None
    return {
        "tool": event.get("tool"),
        "event": event.get("event"),
        "status": event.get("status"),
        "success_count": event.get("success_count"),
        "failure_count": event.get("failure_count"),
        "error": compact_value(event.get("error"), max_len=220),
    }


def compact_step_event(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict) or event.get("type") != "agent_step_timing":
        return None
    payload: dict[str, Any] = {
        "middleware": event.get("middleware"),
        "phase": event.get("phase"),
        "event": event.get("event"),
        "result_type": event.get("result_type"),
    }
    tool_call = event.get("tool_call")
    if isinstance(tool_call, dict):
        payload["tool_name"] = tool_call.get("tool_name")
    request = event.get("request")
    if isinstance(request, dict):
        payload["tool_count"] = request.get("tool_count")
    return payload


def compact_inner_event(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict) or event.get("type") != "inner_agent_update":
        return None
    inner = event.get("event")
    tools: list[str] = []
    if isinstance(inner, dict):
        serialized = json.dumps(compact_value(inner), ensure_ascii=False, default=str)
        for tool_name in INNER_TOOL_NAMES:
            if tool_name in serialized:
                tools.append(tool_name)
    return {"type": "inner_agent_update", "tool": event.get("tool"), "inner_tools": sorted(set(tools))}


def compact_generic_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {"value": compact_value(event)}
    return {
        "type": event.get("type"),
        "event": event.get("event"),
        "status": event.get("status"),
        "phase": event.get("phase"),
        "step": event.get("step"),
        "error_type": event.get("error_type"),
        "error_message": compact_value(event.get("error_message"), max_len=220),
    }


def extract_response_tool_milestones(response: Any) -> list[str]:
    serialized = json.dumps(compact_value(response, max_len=2000), ensure_ascii=False, default=str)
    milestones: list[str] = []
    for tool_name in INNER_TOOL_NAMES:
        if tool_name in serialized:
            milestones.append(f"inner_tool {tool_name}")
    return milestones


class Monitor:
    def __init__(self, label: str, log: JsonlLog) -> None:
        self.label = label
        self.log = log
        self.last_status: tuple[str | None, str | None, str | None] | None = None
        self.seen_events: set[tuple[Any, ...]] = set()
        self.milestones: set[str] = set()
        self.seed_existing_events()

    def seed_existing_events(self) -> None:
        try:
            snapshot = get_json(f"{MAIN_SERVER}/agents/SeedAgent", timeout=5.0)
        except Exception:
            return
        self.last_status = (snapshot.get("status"), snapshot.get("phase"), snapshot.get("step"))
        for event in snapshot.get("events", []):
            if isinstance(event, dict):
                self.seen_events.add(self.event_signature(event))

    @staticmethod
    def event_signature(event: dict[str, Any]) -> tuple[Any, ...]:
        return (
            event.get("at"),
            event.get("type"),
            event.get("middleware"),
            event.get("tool"),
            event.get("phase"),
            event.get("event"),
            event.get("error"),
        )

    def poll_once(self) -> list[str]:
        visible: list[str] = []
        try:
            snapshot = get_json(f"{MAIN_SERVER}/agents/SeedAgent", timeout=5.0)
        except Exception as exc:
            self.log.write("monitor_error", {"label": self.label, "error": f"{type(exc).__name__}: {exc}"})
            return visible

        status_sig = (snapshot.get("status"), snapshot.get("phase"), snapshot.get("step"))
        if status_sig != self.last_status:
            self.last_status = status_sig
            self.log.write(
                "status",
                {
                    "label": self.label,
                    "status": snapshot.get("status"),
                    "phase": snapshot.get("phase"),
                    "step": snapshot.get("step"),
                },
            )

        for event in snapshot.get("events", []):
            if not isinstance(event, dict):
                continue
            signature = self.event_signature(event)
            if signature in self.seen_events:
                continue
            self.seen_events.add(signature)
            self.log.write("event", {"label": self.label, "event": event})
            for milestone in self.event_milestones(event):
                if milestone not in self.milestones:
                    self.milestones.add(milestone)
                    visible.append(milestone)
        return visible

    def event_milestones(self, event: dict[str, Any]) -> list[str]:
        tool_event = compact_tool_event(event)
        if tool_event:
            tool = tool_event.get("tool")
            action = tool_event.get("event")
            if tool in {"ingest_knowledge_document", "manage_knowledge"} and action in {"start", "success", "error"}:
                text = f"{tool} {action}"
                if tool_event.get("success_count") is not None:
                    text += f" success_count={tool_event.get('success_count')}"
                if tool_event.get("error"):
                    text += f" error={tool_event.get('error')}"
                return [text]
        step_event = compact_step_event(event)
        if step_event and step_event.get("phase") == "wrap_tool_call" and step_event.get("event") == "start":
            tool_name = step_event.get("tool_name")
            if tool_name in {"ingest_knowledge_document", "manage_knowledge"}:
                return [f"tool_call {tool_name}"]
        inner_event = compact_inner_event(event)
        if inner_event:
            tools = inner_event.get("inner_tools") or []
            return [f"inner_tools {','.join(tools)}"] if tools else []
        return []


def wait_health(url: str, *, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            data = get_json(f"{url}/healthz", timeout=3.0)
            if data.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def ensure_services(log: JsonlLog) -> list[subprocess.Popen[str]]:
    started: list[subprocess.Popen[str]] = []
    if not wait_health(MAIN_SERVER, timeout_seconds=2.0):
        main_log = LOG_DIR / f"real_memory_mainserver_{short_id()}.log"
        handle = main_log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            ["uv", "run", "python", "main.py", "mainserver"],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started.append(process)
        log.write("service_start", {"name": "mainserver", "pid": process.pid, "log": str(main_log)})
    if not wait_health(MAIN_SERVER, timeout_seconds=30.0):
        log.write("service_error", {"name": "mainserver", "error": "health timeout"})
        return started

    if not wait_health(SEED_URL, timeout_seconds=2.0):
        seed_log = LOG_DIR / f"real_memory_seedagent_{short_id()}.log"
        handle = seed_log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            ["uv", "run", "python", "main.py", "seedagent"],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started.append(process)
        log.write("service_start", {"name": "seedagent", "pid": process.pid, "log": str(seed_log)})
    if not wait_health(SEED_URL, timeout_seconds=120.0):
        log.write("service_error", {"name": "seedagent", "error": "health timeout"})
    return started


def stop_started(processes: list[subprocess.Popen[str]], log: JsonlLog) -> None:
    for process in reversed(processes):
        if process.poll() is not None:
            log.write("service_stop", {"pid": process.pid, "returncode": process.returncode})
            continue
        process.terminate()
        try:
            process.wait(timeout=12.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        log.write("service_stop", {"pid": process.pid, "returncode": process.returncode})


def healthcheck(log: JsonlLog) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": True, "errors": []}
    for name, url in (("main", MAIN_SERVER), ("seed", SEED_URL)):
        try:
            data = get_json(f"{url}/healthz", timeout=5.0)
            ok = data.get("status") == "ok"
            result["ok"] = result["ok"] and ok
            log.write("health", {"name": name, "url": url, "ok": ok, "data": data})
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
            log.write("health_error", {"name": name, "url": url, "error": f"{type(exc).__name__}: {exc}"})
    return result


def write_probe_document(token: str, log: JsonlLog) -> tuple[str, str, Path]:
    knowledge_dir = SEED_WORKSPACE / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    document_name = f"memory_probe_{token}"
    path = knowledge_dir / f"{document_name}.md"
    marker = f"MEMORY_PROBE_{token}"
    path.write_text(
        "\n".join(
            [
                f"# Memory Probe {token}",
                "",
                f"唯一标记：{marker}",
                "项目：LangVideo memory ingestion quiet test.",
                "事实一：蓝线计划把文档切分成知识 chunk，并写入 Neo4j。",
                "事实二：青石索引负责把唯一标记、摘要和关键词保存到记忆库。",
                "事实三：查询阶段必须能通过 manage_knowledge 找回这份文档。",
                "",
                "## Delivery",
                f"最终回答必须包含 {marker} 和 青石索引。",
            ]
        ),
        encoding="utf-8",
    )
    log.write("probe_document", {"path": str(path), "workspace_path": f"/workspace/knowledge/{path.name}", "marker": marker})
    return document_name, marker, path


def run_stage(label: str, prompt: str, log: JsonlLog) -> dict[str, Any]:
    session_id = short_id()
    run_id = short_id()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "session_id": session_id,
        "run_id": run_id,
        "stream_mode": ["updates", "custom"],
        "version": "v2",
    }
    log.write("stage_start", {"label": label, "session_id": session_id, "run_id": run_id})
    monitor = Monitor(label, log)
    outcome: dict[str, Any] = {"ok": False, "error": None, "response": None, "milestones": []}

    def target() -> None:
        try:
            outcome["response"] = post_json(f"{SEED_URL}/invoke", payload)
            outcome["ok"] = bool(outcome["response"].get("ok"))
            outcome["milestones"].extend(extract_response_tool_milestones(outcome["response"]))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            outcome["error"] = f"HTTPError {exc.code}: {body}"
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    while worker.is_alive():
        outcome["milestones"].extend(monitor.poll_once())
        time.sleep(0.8)
    worker.join()
    for _ in range(3):
        outcome["milestones"].extend(monitor.poll_once())
        time.sleep(0.2)

    chunks = len((outcome["response"] or {}).get("chunks", []))
    log.write(
        "stage_end",
        {
            "label": label,
            "ok": outcome["ok"],
            "error": outcome["error"],
            "chunks": chunks,
            "milestones": outcome["milestones"],
            "response": compact_value(outcome["response"], max_len=1200),
        },
    )
    outcome["chunks"] = chunks
    return outcome


def expected_run_id(document_name: str) -> str:
    del document_name
    return "SeedAgent-knowledge"


def neo4j_payload() -> dict[str, Any]:
    config_path = ROOT / "Deepagents" / "SeedAgent" / "Agent" / "Models" / "model_config.json"
    del config_path
    return {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "1575338771",
        "database": None,
    }


def check_neo4j_document(document_name: str, marker: str, log: JsonlLog) -> dict[str, Any]:
    run_id = expected_run_id(document_name)
    config = neo4j_payload()
    try:
        with GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"])) as driver:
            with driver.session(database=config["database"]) as session:
                records = list(
                    session.run(
                        """
                        MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
                        RETURN chunk.chunk_index AS chunk_index,
                               chunk.summary AS summary,
                               coalesce(chunk.body, chunk.text, '') AS body,
                               chunk.keywords AS keywords
                        ORDER BY chunk.chunk_index
                        """,
                        run_id=run_id,
                        document_name=document_name,
                    )
                )
        chunks = [
            {
                "chunk_index": int(record["chunk_index"]),
                "summary": str(record["summary"] or ""),
                "body": str(record["body"] or ""),
                "keywords": list(record["keywords"] or []),
            }
            for record in records
        ]
        marker_hit = any(marker in f"{chunk['summary']}\n{chunk['body']}\n{chunk['keywords']}" for chunk in chunks)
        result = {
            "ok": bool(chunks and marker_hit),
            "document_name": document_name,
            "expected_run_id": run_id,
            "chunk_count": len(chunks),
            "marker_hit": marker_hit,
            "sample": chunks[:3],
        }
    except Exception as exc:
        result = {
            "ok": False,
            "document_name": document_name,
            "expected_run_id": run_id,
            "chunk_count": 0,
            "marker_hit": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    log.write("neo4j_check", result)
    return result


def verify_memory_flow(
    *,
    ingest: dict[str, Any],
    manage: dict[str, Any],
    neo4j_check: dict[str, Any],
    marker: str,
    log: JsonlLog,
) -> dict[str, Any]:
    ingest_milestones = set(ingest.get("milestones") or [])
    manage_milestones = set(manage.get("milestones") or [])
    manage_response_text = json.dumps(compact_value(manage.get("response")), ensure_ascii=False, default=str)
    checks = [
        {"name": "ingest_stage_ok", "ok": bool(ingest.get("ok"))},
        {
            "name": "ingest_tool_called",
            "ok": any("ingest_knowledge_document" in item for item in ingest_milestones),
        },
        {
            "name": "ingest_neo4j_written",
            "ok": bool(neo4j_check.get("ok")),
        },
        {"name": "manage_stage_ok", "ok": bool(manage.get("ok"))},
        {
            "name": "manage_tool_called",
            "ok": any("manage_knowledge" in item for item in manage_milestones),
        },
        {
            "name": "manager_used_document_query",
            "ok": any("query_chunk_positions" in item or "list_chunk_documents" in item for item in manage_milestones)
            or "query_chunk_positions" in manage_response_text
            or "list_chunk_documents" in manage_response_text,
        },
        {
            "name": "marker_returned",
            "ok": marker in manage_response_text,
        },
    ]
    failed = [check for check in checks if not check["ok"]]
    result = {"ok": not failed, "checks": checks, "reason": None if not failed else failed[0]["name"]}
    log.write("verification", result)
    return result


def main() -> int:
    log = JsonlLog()
    summary = Summary()
    started: list[subprocess.Popen[str]] = []
    final_ok = False
    summary.add("log", str(log.path))
    summary.add("log_tail", f"tail -n 80 {log.path}")
    summary.add("full_log", str(log.full_path))
    try:
        started = ensure_services(log)
        health = healthcheck(log)
        summary.add("setup", f"health_ok={health['ok']}")

        token = short_id()
        document_name, marker, probe_path = write_probe_document(token, log)
        summary.add("setup", f"document=/workspace/knowledge/{probe_path.name} marker={marker}")

        ingest_prompt = (
            "必须调用 ingest_knowledge_document 工具一次。"
            f"把 /workspace/knowledge/{probe_path.name} 切分并写入记忆库。"
            "chunkingRequirement：保留唯一标记、青石索引、蓝线计划这些关键词。"
            "完成后只用一句话说明入库是否成功，不要调用 manage_knowledge。"
        )
        ingest = run_stage("ingest", ingest_prompt, log)
        summary.add("ingest", f"ok={ingest['ok']} chunks={ingest['chunks']} milestones={', '.join(ingest['milestones'][:5])}")

        neo4j_check = check_neo4j_document(document_name, marker, log)
        summary.add(
            "ingest",
            f"neo4j_ok={neo4j_check['ok']} chunk_count={neo4j_check.get('chunk_count')} run_id={neo4j_check.get('expected_run_id')}",
        )

        manage_prompt = (
            "必须调用 manage_knowledge 工具一次。"
            "target 请写成："
            f"请先列出知识库文档，确认存在文档 {document_name}；"
            f"然后用 query_chunk_positions 读取 {document_name} 的 [0, 3] 范围，"
            f"确认内容里出现唯一标记 {marker} 和 青石索引；"
            "如果找到了，把有用 chunk 放入 useful 桶，并在最终摘要里原样写出唯一标记。"
            "主 agent 不要自己回答检索结果，必须等 manage_knowledge 返回后再总结。"
        )
        manage = run_stage("manage", manage_prompt, log)
        summary.add("manage", f"ok={manage['ok']} chunks={manage['chunks']} milestones={', '.join(manage['milestones'][:8])}")

        verification = verify_memory_flow(
            ingest=ingest,
            manage=manage,
            neo4j_check=neo4j_check,
            marker=marker,
            log=log,
        )
        summary.add("verify", f"ok={verification['ok']} reason={verification.get('reason')}")
        final_ok = bool(health["ok"] and verification["ok"])
        summary.add("summary", f"final_ok={final_ok}")
        log.write("summary", {"lines": summary.lines, "final_ok": final_ok})
        return 0 if final_ok else 1
    finally:
        stop_started(started, log)


if __name__ == "__main__":
    raise SystemExit(main())
