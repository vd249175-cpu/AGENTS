import json
import secrets
import string
import threading
import time
import urllib.error
import urllib.request
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/apexwave/Desktop/Agents")
MAIN_SERVER = "http://127.0.0.1:8000"
SEED_URL = "http://127.0.0.1:8011"
KNOWLEDGE_URL = "http://127.0.0.1:8010"
LOG_DIR = ROOT / "tests" / "logs"
SEED_WORKSPACE_FILE = ROOT / "Deepagents" / "SeedAgent" / "workspace" / "notes" / "mail_file_probe.md"
KNOWLEDGE = "KnowledgeSeedAgent"
SEED = "SeedAgent"
SECRET = "MAIL_FILE_PROBE_GREEN_LANTERN_20260429"
CHINESE_MARKER = "雾港码头的蓝色台灯在午夜亮起，收信人必须按普通文本读取这一行。"
ALPHABET = string.ascii_lowercase + string.digits


def short_id(length: int = 6) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def get_json(url: str, timeout: float = 15.0) -> Any:
    return request_json("GET", url, timeout=timeout)


def post_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0) -> Any:
    return request_json("POST", url, payload or {}, timeout=timeout)


class JsonlLog:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = LOG_DIR / f"real_mail_file_attachment_{stamp}_{short_id()}.jsonl"

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class Summary:
    def add(self, label: str, text: str) -> None:
        print(f"{label}: {text}", flush=True)


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event.get("type"),
        "event": event.get("event"),
        "phase": event.get("phase"),
        "tool": event.get("tool"),
        "middleware": event.get("middleware"),
        "target": event.get("target"),
        "status": event.get("status"),
        "step": event.get("step"),
    }
    tool_call = event.get("tool_call")
    if isinstance(tool_call, dict):
        payload["tool_name"] = tool_call.get("tool_name")
    return {key: value for key, value in payload.items() if value not in (None, "")}


class Monitor:
    def __init__(self, log: JsonlLog) -> None:
        self.log = log
        self.seen_events: set[tuple[Any, ...]] = set()
        self.last_status: dict[str, tuple[Any, Any, Any]] = {}
        self.milestones: list[str] = []

    def poll_agent(self, agent_name: str) -> None:
        try:
            snapshot = get_json(f"{MAIN_SERVER}/agents/{agent_name}", timeout=5.0)
        except Exception as exc:
            self.log.write("monitor_error", {"agent": agent_name, "error": f"{type(exc).__name__}: {exc}"})
            return
        status_sig = (snapshot.get("status"), snapshot.get("phase"), snapshot.get("step"))
        if self.last_status.get(agent_name) != status_sig:
            self.last_status[agent_name] = status_sig
            self.log.write(
                "status",
                {
                    "agent": agent_name,
                    "status": snapshot.get("status"),
                    "phase": snapshot.get("phase"),
                    "step": snapshot.get("step"),
                },
            )
        for event in snapshot.get("events", [])[-30:]:
            if not isinstance(event, dict):
                continue
            signature = (
                agent_name,
                event.get("at"),
                event.get("type"),
                event.get("event"),
                event.get("phase"),
                event.get("tool"),
                event.get("target"),
            )
            if signature in self.seen_events:
                continue
            self.seen_events.add(signature)
            compact = compact_event(event)
            if compact.get("event") == "status" and compact.get("phase") == "messages":
                continue
            self.log.write("event", {"agent": agent_name, "event": compact})
            milestone = self._milestone(agent_name, compact)
            if milestone and milestone not in self.milestones:
                self.milestones.append(milestone)

    @staticmethod
    def _milestone(agent_name: str, event: dict[str, Any]) -> str | None:
        if event.get("type") == "tool" and event.get("tool") == "send_message_to_agent":
            target = event.get("target") or "?"
            return f"{agent_name} send_message_to_agent {event.get('event')}->{target}"
        if event.get("phase") == "wrap_tool_call" and event.get("event") == "start":
            tool_name = event.get("tool_name")
            if tool_name in {"send_message_to_agent", "read_file"}:
                return f"{agent_name} tool_call {tool_name}"
        if event.get("event") in {"mail_wake_start", "mail_wake_success", "mail_wake_error"}:
            return f"{agent_name} {event.get('event')}"
        return None

    def poll(self) -> None:
        self.poll_agent(SEED)
        self.poll_agent(KNOWLEDGE)


def clear_inboxes(log: JsonlLog) -> None:
    for agent in (SEED, KNOWLEDGE):
        try:
            data = get_json(f"{MAIN_SERVER}/recv/{agent}", timeout=10.0)
            log.write("clear_inbox", {"agent": agent, "count": len(data.get("messages", []))})
        except Exception as exc:
            log.write("clear_inbox_error", {"agent": agent, "error": f"{type(exc).__name__}: {exc}"})


def clear_runtime_for_checkpoint_test(log: JsonlLog) -> None:
    payload = {
        "include_store": False,
        "include_mail": True,
        "include_knowledge": False,
        "include_checkpoints": True,
    }
    for agent in (SEED, KNOWLEDGE):
        try:
            data = post_json(f"{MAIN_SERVER}/admin/agents/{agent}/runtime/clear", payload, timeout=60.0)
            log.write(
                "clear_runtime",
                {
                    "agent": agent,
                    "ok": data.get("ok"),
                    "removed": data.get("runtime", {}).get("removed"),
                    "reload_ok": data.get("reload", {}).get("ok"),
                },
            )
        except Exception as exc:
            log.write("clear_runtime_error", {"agent": agent, "error": f"{type(exc).__name__}: {exc}"})


def health(log: JsonlLog) -> bool:
    ok = True
    for name, url in (("main", MAIN_SERVER), (SEED, SEED_URL), (KNOWLEDGE, KNOWLEDGE_URL)):
        try:
            data = get_json(f"{url}/healthz", timeout=8.0)
            item_ok = data.get("status") == "ok"
            ok = ok and item_ok
            log.write("health", {"name": name, "ok": item_ok})
        except Exception as exc:
            ok = False
            log.write("health", {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return ok


def invoke_seed(log: JsonlLog, monitor: Monitor) -> dict[str, Any]:
    prompt = (
        "执行一次邮件附件链路测试。"
        "请向 KnowledgeSeedAgent 发送一封 message 邮件。"
        "邮件 content 必须要求对方：只读取这封邮件 `<Inbox>` attachments 中列出的附件；"
        "按 UTF-8 普通文本读取；不要扫描旧 /workspace/mail；不要做 base64 或二进制解码；"
        "读取后用 send_message_to_agent 给 SeedAgent 回一封 message，"
        "回复 content 必须以 FILE_ATTACHMENT_READ_OK 开头，并包含附件正文里的 secret 和包含“蓝色台灯”的完整中文句子。"
        "你发送邮件时必须把 /workspace/notes/mail_file_probe.md 作为 attachments 发送。"
        "不要在你自己的邮件正文或最终回答里写出附件 secret。"
    )
    payload = {
        "agent_name": SEED,
        "mode": "direct",
        "text": prompt,
        "thread_id": "default",
        "run_id": "default-run",
        "stream_mode": ["custom"],
        "version": "v2",
        "timeout": 600,
    }
    outcome: dict[str, Any] = {"ok": False, "error": None, "reply_len": 0, "chunk_count": None}

    def target() -> None:
        try:
            response = post_json(f"{MAIN_SERVER}/user/chat", payload, timeout=650.0)
            outcome["ok"] = bool(response.get("ok"))
            agent_response = response.get("response") if isinstance(response.get("response"), dict) else {}
            outcome["reply_len"] = len(str(agent_response.get("reply") or ""))
            outcome["chunk_count"] = len(agent_response.get("chunks") or [])
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        monitor.poll()
        time.sleep(0.8)
    thread.join()
    for _ in range(3):
        monitor.poll()
        time.sleep(0.5)
    log.write("seed_invoke", outcome)
    return outcome


def wait_for_reply(log: JsonlLog, monitor: Monitor, timeout: float = 180.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_recent: list[dict[str, Any]] = []
    while time.time() < deadline:
        monitor.poll()
        try:
            data = get_json(f"{MAIN_SERVER}/admin/monitor", timeout=10.0)
            last_recent = list(data.get("recent_mail") or [])
        except Exception as exc:
            log.write("monitor_recent_error", {"error": f"{type(exc).__name__}: {exc}"})
            last_recent = []
        reply = find_reply(last_recent)
        if reply is not None:
            return {"ok": True, "reply": reply, "recent_mail_count": len(last_recent)}
        time.sleep(1.2)
    return {"ok": False, "reply": None, "recent_mail_count": len(last_recent)}


def find_reply(recent_mail: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(recent_mail):
        message = item.get("message") if isinstance(item, dict) else None
        if not isinstance(message, dict):
            continue
        if message.get("from") == KNOWLEDGE and message.get("to") == SEED:
            content = str(message.get("content") or "")
            if "FILE_ATTACHMENT_READ_OK" in content:
                return message
    return None


def find_sent_attachment(recent_mail: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(recent_mail):
        message = item.get("message") if isinstance(item, dict) else None
        if not isinstance(message, dict):
            continue
        if message.get("from") == SEED and message.get("to") == KNOWLEDGE:
            attachments = message.get("attachments") if isinstance(message.get("attachments"), list) else []
            if attachments:
                return message
    return None


def verify(log: JsonlLog, reply_result: dict[str, Any]) -> dict[str, Any]:
    data = get_json(f"{MAIN_SERVER}/admin/monitor", timeout=15.0)
    recent_mail = list(data.get("recent_mail") or [])
    sent = find_sent_attachment(recent_mail)
    reply = reply_result.get("reply")
    checks: list[dict[str, Any]] = []
    checks.append({"name": "seed_sent_attachment_mail", "ok": sent is not None})
    copied_content = ""
    copied_path = None
    if sent is not None:
        attachments = sent.get("attachments") or []
        attachment = attachments[0] if attachments else {}
        copied_path = attachment.get("link")
        checks.append({"name": "attachment_routed_copied", "ok": attachment.get("_routed") == "copied"})
        path = Path(str(copied_path or ""))
        checks.append({"name": "routed_file_exists", "ok": path.exists(), "path": str(path)})
        if path.exists():
            copied_content = path.read_text(encoding="utf-8")
    checks.append({"name": "routed_file_has_secret", "ok": SECRET in copied_content})
    checks.append({"name": "routed_file_has_utf8_chinese", "ok": CHINESE_MARKER in copied_content})
    checks.append({"name": "knowledge_replied", "ok": isinstance(reply, dict)})
    reply_content = str(reply.get("content") if isinstance(reply, dict) else "")
    checks.append({"name": "reply_has_ok_prefix", "ok": reply_content.startswith("FILE_ATTACHMENT_READ_OK")})
    checks.append({"name": "reply_has_secret_from_file", "ok": SECRET in reply_content})
    checks.append({"name": "reply_has_utf8_chinese_from_file", "ok": CHINESE_MARKER in reply_content})
    checkpoint_info = verify_checkpoints()
    checks.extend(checkpoint_info["checks"])
    failed = [check for check in checks if not check.get("ok")]
    result = {
        "ok": not failed,
        "reason": None if not failed else failed[0]["name"],
        "checks": checks,
        "copied_path": copied_path,
        "checkpoint_threads": checkpoint_info["threads"],
    }
    log.write("verification", result)
    return result


def sqlite_threads(path: Path) -> list[str]:
    if not path.exists():
        return []
    with sqlite3.connect(path) as connection:
        rows = connection.execute("select distinct thread_id from checkpoints order by thread_id").fetchall()
    return [str(row[0]) for row in rows]


def verify_checkpoints() -> dict[str, Any]:
    expected = "default-run:default"
    seed_threads = sqlite_threads(ROOT / "Deepagents" / "SeedAgent" / "Agent" / "store" / "checkpoints" / "langgraph.sqlite3")
    knowledge_threads = sqlite_threads(
        ROOT / "Deepagents" / "KnowledgeSeedAgent" / "Agent" / "store" / "checkpoints" / "langgraph.sqlite3"
    )
    return {
        "threads": {SEED: seed_threads, KNOWLEDGE: knowledge_threads},
        "checks": [
            {"name": "seed_checkpoint_single_thread", "ok": seed_threads == [expected], "threads": seed_threads},
            {
                "name": "knowledge_checkpoint_single_thread",
                "ok": knowledge_threads == [expected],
                "threads": knowledge_threads,
            },
        ],
    }


def main() -> int:
    log = JsonlLog()
    summary = Summary()
    monitor = Monitor(log)
    summary.add("log", str(log.path))
    summary.add("tail", f"tail -n 80 {log.path}")
    summary.add("setup", f"fixture_exists={SEED_WORKSPACE_FILE.exists()}")
    if not SEED_WORKSPACE_FILE.exists():
        log.write("error", {"reason": "missing workspace fixture", "path": str(SEED_WORKSPACE_FILE)})
        return 2
    health_ok = health(log)
    summary.add("setup", f"health_ok={health_ok}")
    clear_runtime_for_checkpoint_test(log)
    clear_inboxes(log)
    seed = invoke_seed(log, monitor)
    summary.add(
        "seed",
        f"ok={seed['ok']} chunks={seed['chunk_count']} milestones={', '.join(monitor.milestones[:6])}",
    )
    reply_result = wait_for_reply(log, monitor)
    summary.add("reply", f"ok={reply_result['ok']} recent_mail_count={reply_result['recent_mail_count']}")
    result = verify(log, reply_result)
    summary.add(
        "verify",
        f"ok={result['ok']} reason={result['reason']} copied_path={result['copied_path']} checkpoints={result['checkpoint_threads']}",
    )
    summary.add("summary", f"final_ok={result['ok']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
