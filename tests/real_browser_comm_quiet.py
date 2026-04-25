import json
import secrets
import string
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/apexwave/Desktop/Agents")
MAIN_SERVER = "http://127.0.0.1:8000"
SEED_URL = "http://127.0.0.1:8010"
WORKER_URL = "http://127.0.0.1:8011"
SEED_WORKSPACE = ROOT / "Deepagents" / "SeedAgent" / "workspace"
WORKER_WORKSPACE = ROOT / "Deepagents" / "WorkerAgent" / "workspace"
LOG_DIR = ROOT / "tests" / "logs"
ALPHABET = string.ascii_lowercase + string.digits


def short_id(length: int = 4) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def get_json(url: str, timeout: float = 10.0) -> Any:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def post_json(url: str, payload: dict[str, Any], timeout: float = 300.0) -> Any:
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


class JsonlLog:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = LOG_DIR / f"real_browser_comm_{stamp}_{short_id()}.jsonl"
        self.full_path = LOG_DIR / f"real_browser_comm_{stamp}_{short_id()}_full.jsonl"

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        full_record = {
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **payload,
        }
        with self.full_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(full_record, ensure_ascii=False, default=str) + "\n")

        record = self.compact_record(full_record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def compact_record(self, record: dict[str, Any]) -> dict[str, Any]:
        kind = record.get("kind")
        base = {"at": record.get("at"), "kind": kind}
        if kind == "event":
            event = record.get("event")
            if isinstance(event, dict):
                return {
                    **base,
                    "label": record.get("label"),
                    "agent": record.get("agent"),
                    "event": compact_tool_event(event) or compact_step_event(event) or compact_generic_event(event),
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
        if kind == "recv_seed":
            messages = record.get("messages") if isinstance(record.get("messages"), list) else []
            return {
                **base,
                "count": record.get("count"),
                "messages": [
                    {
                        "message_id": message.get("message_id"),
                        "from": message.get("from"),
                        "to": message.get("to"),
                        "type": message.get("type"),
                        "content": message.get("content"),
                        "attachments": [
                            {
                                "link": item.get("link"),
                                "summary": item.get("summary"),
                                "_routed": item.get("_routed"),
                            }
                            for item in message.get("attachments", [])
                            if isinstance(item, dict)
                        ],
                    }
                    for message in messages
                    if isinstance(message, dict)
                ],
            }
        if kind == "verification":
            return {
                **base,
                "ok": record.get("ok"),
                "reason": record.get("reason"),
                "checks": record.get("checks"),
                "copied_file": record.get("copied_file"),
            }
        if kind == "summary":
            repair = record.get("repair")
            repair_summary = None
            if isinstance(repair, dict):
                repair_summary = {
                    "ok": repair.get("ok"),
                    "error": repair.get("error"),
                    "chunks": repair.get("chunks"),
                    "milestones": repair.get("milestones"),
                }
            return {
                **base,
                "lines": record.get("lines"),
                "final_ok": record.get("final_ok"),
                "repair": repair_summary,
            }
        return record


class Summary:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, label: str, text: str) -> None:
        line = f"{label}: {text}"
        self.lines.append(line)
        print(line, flush=True)


def compact_tool_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") != "tool":
        return None
    return {
        "tool": event.get("tool"),
        "event": event.get("event"),
        "target": event.get("target"),
        "messageType": event.get("messageType"),
        "error": event.get("error"),
    }


def compact_step_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") != "agent_step_timing":
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
        payload["message_count"] = request.get("message_count")
        payload["tool_count"] = request.get("tool_count")
    return payload


def compact_generic_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": event.get("type"),
        "event": event.get("event"),
        "status": event.get("status"),
        "phase": event.get("phase"),
        "step": event.get("step"),
        "error_type": event.get("error_type"),
        "error_message": event.get("error_message"),
    }


class Monitor:
    def __init__(self, agent_name: str, label: str, log: JsonlLog) -> None:
        self.agent_name = agent_name
        self.label = label
        self.log = log
        self.last_status: tuple[str | None, str | None, str | None] | None = None
        self.seen_events: set[tuple[Any, ...]] = set()
        self.milestones: set[str] = set()

    @staticmethod
    def event_signature(event: dict[str, Any]) -> tuple[Any, ...]:
        return (
            event.get("at"),
            event.get("type"),
            event.get("middleware"),
            event.get("tool"),
            event.get("phase"),
            event.get("event"),
            event.get("target"),
            event.get("error"),
        )

    def poll_once(self) -> list[str]:
        visible: list[str] = []
        try:
            snapshot = get_json(f"{MAIN_SERVER}/agents/{self.agent_name}", timeout=5.0)
        except urllib.error.HTTPError as exc:
            self.log.write("monitor_error", {"label": self.label, "error": str(exc)})
            return visible

        status_sig = (snapshot.get("status"), snapshot.get("phase"), snapshot.get("step"))
        if status_sig != self.last_status:
            self.last_status = status_sig
            self.log.write(
                "status",
                {
                    "label": self.label,
                    "agent": self.agent_name,
                    "status": snapshot.get("status"),
                    "phase": snapshot.get("phase"),
                    "step": snapshot.get("step"),
                },
            )

        for event in snapshot.get("events", []):
            signature = self.event_signature(event)
            if signature in self.seen_events:
                continue
            self.seen_events.add(signature)
            self.log.write("event", {"label": self.label, "agent": self.agent_name, "event": event})
            tool_event = compact_tool_event(event)
            if tool_event:
                visible.extend(self.tool_milestones(tool_event))
                continue
            step_event = compact_step_event(event)
            if step_event:
                visible.extend(self.step_milestones(step_event))
        return visible

    def tool_milestones(self, event: dict[str, Any]) -> list[str]:
        tool = event.get("tool")
        action = event.get("event")
        target = event.get("target")
        key = f"tool:{tool}:{action}:{target}"
        if key in self.milestones:
            return []
        self.milestones.add(key)
        if tool == "send_message_to_agent" and action in {"start", "success", "error"}:
            text = f"{tool} {action}"
            if target:
                text += f" -> {target}"
            if event.get("error"):
                text += f" error={str(event['error']).splitlines()[0]}"
            return [text]
        return []

    def step_milestones(self, event: dict[str, Any]) -> list[str]:
        phase = event.get("phase")
        action = event.get("event")
        tool_name = event.get("tool_name")
        if phase == "wrap_tool_call" and action == "start" and tool_name:
            key = f"step:tool:{tool_name}"
            if key in self.milestones:
                return []
            self.milestones.add(key)
            if tool_name in {"execute", "send_message_to_agent"}:
                return [f"tool_call {tool_name}"]
        return []


def healthcheck(log: JsonlLog) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": True, "errors": []}
    for name, url in (("main", MAIN_SERVER), ("seed", SEED_URL), ("worker", WORKER_URL)):
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


def clear_inbox(agent_name: str, log: JsonlLog) -> int:
    data = get_json(f"{MAIN_SERVER}/recv/{agent_name}")
    count = len(data.get("messages", []))
    log.write("clear_inbox", {"agent": agent_name, "count": count, "data": data})
    return count


def reset_workspace(log: JsonlLog) -> None:
    for workspace in (SEED_WORKSPACE, WORKER_WORKSPACE):
        notes_dir = workspace / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for item in (notes_dir / "browser_snapshot.txt", notes_dir / "worker_result.txt"):
            if item.is_file():
                item.unlink()
    log.write("reset_workspace", {"seed": str(SEED_WORKSPACE), "worker": str(WORKER_WORKSPACE)})


def run_stage(label: str, agent_name: str, base_url: str, prompt: str, log: JsonlLog) -> dict[str, Any]:
    session_id = short_id()
    run_id = short_id()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "session_id": session_id,
        "run_id": run_id,
        "stream_mode": ["updates", "custom"],
        "version": "v2",
    }
    log.write("stage_start", {"label": label, "agent": agent_name, "session_id": session_id, "run_id": run_id})
    monitor = Monitor(agent_name, label, log)
    outcome: dict[str, Any] = {"ok": False, "error": None, "response": None, "milestones": []}

    def target() -> None:
        try:
            outcome["response"] = post_json(f"{base_url}/invoke", payload)
            outcome["ok"] = bool(outcome["response"].get("ok"))
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    while worker.is_alive():
        outcome["milestones"].extend(monitor.poll_once())
        time.sleep(0.7)
    worker.join()
    for _ in range(2):
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
            "response": outcome["response"],
        },
    )
    outcome["chunks"] = chunks
    return outcome


def receive_seed_messages(log: JsonlLog) -> list[dict[str, Any]]:
    data = get_json(f"{MAIN_SERVER}/recv/SeedAgent")
    messages = list(data.get("messages", []))
    log.write("recv_seed", {"count": len(messages), "messages": messages})
    return messages


def verify_browser_delivery(messages: list[dict[str, Any]], log: JsonlLog) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    target = None
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if message.get("from") == "WorkerAgent" and content.startswith("BROWSER_DONE Example Domain"):
            target = message
            break
    checks.append({"name": "completion_message", "ok": target is not None})
    if target is None:
        result = {"ok": False, "reason": "missing browser completion message", "checks": checks}
        log.write("verification", result)
        return result

    attachments = target.get("attachments", [])
    checks.append({"name": "has_attachment", "ok": bool(attachments)})
    if not attachments:
        result = {"ok": False, "reason": "missing attachment", "checks": checks, "message": target}
        log.write("verification", result)
        return result

    attachment = attachments[0]
    copied_file = Path(str(attachment.get("link") or ""))
    checks.append({"name": "attachment_copied", "ok": attachment.get("_routed") == "copied"})
    checks.append({"name": "copied_file_exists", "ok": copied_file.exists(), "path": str(copied_file)})

    content = copied_file.read_text(encoding="utf-8").strip() if copied_file.exists() else ""
    checks.append({"name": "snapshot_has_example_domain", "ok": "Example Domain" in content})
    checks.append({"name": "snapshot_has_learn_more", "ok": "Learn more" in content})

    failed = [check for check in checks if not check.get("ok")]
    result = {
        "ok": not failed,
        "reason": None if not failed else failed[0]["name"],
        "checks": checks,
        "message": target,
        "copied_file": str(copied_file) if copied_file.exists() else None,
    }
    log.write("verification", result)
    return result


def main() -> int:
    log = JsonlLog()
    summary = Summary()
    summary.add("log", str(log.path))
    summary.add("log_tail", f"tail -n 80 {log.path}")
    summary.add("full_log", str(log.full_path))

    health = healthcheck(log)
    summary.add("setup", f"health_ok={health['ok']}")
    reset_workspace(log)
    cleared_seed = clear_inbox("SeedAgent", log)
    cleared_worker = clear_inbox("WorkerAgent", log)
    summary.add("setup", f"cleared_inbox SeedAgent={cleared_seed} WorkerAgent={cleared_worker}")

    planner_prompt = (
        "向 WorkerAgent 发送一个 task。任务要求："
        "1. 在 WorkerAgent 的 Docker sandbox 中使用 agent-browser 打开 https://example.com。"
        "2. 执行 snapshot -i，并把输出保存到 /workspace/notes/browser_snapshot.txt。"
        "3. 完成后向 SeedAgent 发送一条 message，content 必须精确为 BROWSER_DONE Example Domain。"
        "4. 同时把 /workspace/notes/browser_snapshot.txt 作为附件发送。"
        "不要自己执行浏览器任务，不要发送其他多余消息。完成后简短说明。"
    )
    worker_prompt = (
        "读取收件箱中的最新任务并执行。"
        "先在 Docker sandbox 中使用 agent-browser 打开 https://example.com，"
        "然后执行 snapshot -i，并把输出保存到 /workspace/notes/browser_snapshot.txt。"
        "确认文件存在且内容里包含 Example Domain 和 Learn more。"
        "最后向 SeedAgent 发送一条 message，content 必须精确为 BROWSER_DONE Example Domain，"
        "并把 /workspace/notes/browser_snapshot.txt 作为附件发送。"
        "如果工具调用失败，修正后继续，不要放弃。完成后简短说明。"
    )
    repair_prompt = (
        "不要依赖收件箱。直接执行以下交付修复任务："
        "1. 使用 execute 在 Docker sandbox 中运行 agent-browser 打开 https://example.com。"
        "2. 使用 execute 运行 agent-browser snapshot -i，并把输出写入 /workspace/notes/browser_snapshot.txt。"
        "3. 使用工具确认 /workspace/notes/browser_snapshot.txt 存在，内容包含 Example Domain 和 Learn more。"
        "4. 向 SeedAgent 发送一条 message，content 必须精确为 BROWSER_DONE Example Domain，"
        "并把 /workspace/notes/browser_snapshot.txt 作为附件发送。"
        "如果第一次命令失败，修正命令后继续。"
    )

    planner = run_stage("planner", "SeedAgent", SEED_URL, planner_prompt, log)
    summary.add("planner", f"ok={planner['ok']} chunks={planner['chunks']} milestones={', '.join(planner['milestones'][:4])}")

    worker = run_stage("worker", "WorkerAgent", WORKER_URL, worker_prompt, log)
    summary.add("worker", f"ok={worker['ok']} chunks={worker['chunks']} milestones={', '.join(worker['milestones'][:6])}")

    messages = receive_seed_messages(log)
    verification = verify_browser_delivery(messages, log)
    repair: dict[str, Any] | None = None
    if not verification["ok"]:
        summary.add("repair", f"triggered reason={verification.get('reason')}")
        repair = run_stage("repair", "WorkerAgent", WORKER_URL, repair_prompt, log)
        summary.add(
            "repair",
            f"ok={repair['ok']} chunks={repair['chunks']} milestones={', '.join(repair['milestones'][:6])}",
        )
        messages = receive_seed_messages(log)
        verification = verify_browser_delivery(messages, log)

    summary.add("verify", f"delivery_ok={verification['ok']} reason={verification.get('reason')}")
    if verification.get("copied_file"):
        summary.add("verify", f"copied_file={verification['copied_file']}")

    final_ok = bool(health["ok"] and planner["ok"] and worker["ok"] and verification["ok"])
    summary.add(
        "summary",
        (
            f"planner_ok={planner['ok']} worker_ok={worker['ok']} "
            f"repair_ok={None if repair is None else repair['ok']} "
            f"delivery_ok={verification['ok']} final_ok={final_ok}"
        ),
    )
    log.write("summary", {"lines": summary.lines, "final_ok": final_ok, "repair": repair})
    return 0 if final_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
