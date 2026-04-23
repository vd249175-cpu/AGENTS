import json
import os
import secrets
import string
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path("/Users/apexwave/Desktop/Agents")
MAIN_SERVER = "http://127.0.0.1:8000"
SEED_URL = "http://127.0.0.1:8010"
WORKER_URL = "http://127.0.0.1:8011"
SEED_WORKSPACE = ROOT / "Deepagents" / "SeedAgent" / "workspace"
WORKER_WORKSPACE = ROOT / "Deepagents" / "WorkerAgent" / "workspace"
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


def healthcheck() -> None:
    for url in (MAIN_SERVER, SEED_URL, WORKER_URL):
        data = get_json(f"{url}/healthz", timeout=5.0)
        print(f"[health] {url} ok={data.get('status') == 'ok'}")


def clear_inbox(agent_name: str) -> None:
    data = get_json(f"{MAIN_SERVER}/recv/{agent_name}")
    print(f"[setup] cleared inbox {agent_name}: {len(data.get('messages', []))}")


def reset_workspace() -> None:
    result_file = WORKER_WORKSPACE / "notes" / "worker_result.txt"
    if result_file.exists():
        result_file.unlink()
    for workspace in (SEED_WORKSPACE, WORKER_WORKSPACE):
        mail_dir = workspace / "mail"
        if mail_dir.exists():
            for child in sorted(mail_dir.iterdir()):
                if child.is_dir():
                    for sub in sorted(child.rglob("*"), reverse=True):
                        if sub.is_file():
                            sub.unlink()
                        elif sub.is_dir():
                            sub.rmdir()
                    child.rmdir()
                elif child.is_file():
                    child.unlink()
    print("[setup] workspaces reset")


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


def format_event(label: str, event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    if event_type == "tool":
        parts = [f"[{label}] tool {event.get('tool')} {event.get('event')}"]
        if event.get("target"):
            parts.append(f"target={event.get('target')}")
        if event.get("messageType"):
            parts.append(f"type={event.get('messageType')}")
        if event.get("error"):
            parts.append(f"error={event.get('error')}")
        return " ".join(parts)
    if event_type == "agent_step_timing":
        phase = event.get("phase")
        lifecycle = event.get("event")
        if phase and lifecycle:
            return f"[{label}] middleware {event.get('middleware')} {phase} {lifecycle}"
    if event_type == "agent_debug_trace":
        phase = event.get("phase")
        lifecycle = event.get("event")
        if phase and lifecycle:
            return f"[{label}] debug {phase} {lifecycle}"
    if event.get("event") == "error":
        return f"[{label}] error {event.get('error_type')}: {event.get('error_message')}"
    return None


class Monitor:
    def __init__(self, agent_name: str, label: str) -> None:
        self.agent_name = agent_name
        self.label = label
        self.last_status: tuple[str | None, str | None, str | None] | None = None
        self.seen_events: set[tuple[Any, ...]] = set()

    def poll_once(self) -> None:
        try:
            snapshot = get_json(f"{MAIN_SERVER}/agents/{self.agent_name}", timeout=5.0)
        except urllib.error.HTTPError:
            return
        status_sig = (snapshot.get("status"), snapshot.get("phase"), snapshot.get("step"))
        if status_sig != self.last_status:
            self.last_status = status_sig
            print(
                f"[{self.label}] status {snapshot.get('status')} "
                f"phase={snapshot.get('phase')} step={snapshot.get('step')}"
            )
        for event in snapshot.get("events", []):
            signature = event_signature(event)
            if signature in self.seen_events:
                continue
            self.seen_events.add(signature)
            line = format_event(self.label, event)
            if line:
                print(line)


def run_stage(label: str, agent_name: str, base_url: str, prompt: str) -> dict[str, Any]:
    session_id = short_id()
    run_id = short_id()
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "session_id": session_id,
        "run_id": run_id,
        "stream_mode": ["updates", "custom"],
        "version": "v2",
    }
    print(f"[{label}] setup agent={agent_name} session={session_id} run={run_id}")
    monitor = Monitor(agent_name, label)
    outcome: dict[str, Any] = {"ok": False, "error": None, "response": None}

    def target() -> None:
        try:
            outcome["response"] = post_json(f"{base_url}/invoke", payload)
            outcome["ok"] = bool(outcome["response"].get("ok"))
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    while worker.is_alive():
        monitor.poll_once()
        time.sleep(0.6)
    worker.join()
    for _ in range(3):
        monitor.poll_once()
        time.sleep(0.2)
    if outcome["error"]:
        print(f"[{label}] result error={outcome['error']}")
    else:
        chunks = len((outcome["response"] or {}).get("chunks", []))
        print(f"[{label}] result ok={outcome['ok']} chunks={chunks}")
    return outcome


def receive_seed_messages() -> list[dict[str, Any]]:
    data = get_json(f"{MAIN_SERVER}/recv/SeedAgent")
    messages = list(data.get("messages", []))
    print(f"[verify] SeedAgent inbox received={len(messages)}")
    return messages


def verify_delivery(messages: list[dict[str, Any]], expected_prefix: str) -> dict[str, Any]:
    target = None
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if message.get("from") == "WorkerAgent" and content.startswith(expected_prefix):
            target = message
            break
    if target is None:
        return {"ok": False, "reason": f"missing message prefix {expected_prefix}"}
    attachments = target.get("attachments", [])
    if not attachments:
        return {"ok": False, "reason": "missing attachment", "message": target}
    attachment = attachments[0]
    routed = attachment.get("_routed")
    copied_path = attachment.get("link")
    if routed != "copied":
        return {"ok": False, "reason": f"attachment routed={routed}", "message": target}
    copied_file = Path(str(copied_path))
    if not copied_file.exists():
        return {"ok": False, "reason": f"copied file missing: {copied_file}", "message": target}
    content = copied_file.read_text(encoding="utf-8").strip()
    if content != "RESULT:42":
        return {"ok": False, "reason": f"copied file content={content!r}", "message": target}
    return {"ok": True, "message": target, "copied_file": str(copied_file)}


def main() -> int:
    healthcheck()
    reset_workspace()
    clear_inbox("SeedAgent")
    clear_inbox("WorkerAgent")

    planner_prompt = (
        "向 WorkerAgent 发送一个 task。任务要求："
        "1. 在 /workspace/notes/worker_result.txt 写入精确文本 RESULT:42。"
        "2. 完成后向 SeedAgent 发送一条 message，content 必须精确为 WORKER_DONE RESULT:42。"
        "3. 同时把 /workspace/notes/worker_result.txt 作为附件发送。"
        "不要自己完成任务，不要发送其他多余消息。完成后简短说明。"
    )
    worker_prompt = (
        "读取收件箱中的最新任务并执行。"
        "先确保 /workspace/notes/worker_result.txt 已存在且内容精确为 RESULT:42，"
        "再向 SeedAgent 发送一条 message，content 必须精确为 WORKER_DONE RESULT:42，"
        "并把 /workspace/notes/worker_result.txt 作为附件发送。"
        "如果工具调用失败，修正后继续，不要放弃。完成后简短说明。"
    )
    repair_prompt = (
        "不要重做任务。确认 /workspace/notes/worker_result.txt 已存在且内容为 RESULT:42。"
        "然后向 SeedAgent 发送一条 message，content 精确为 WORKER_REDELIVER RESULT:42，"
        "并把 /workspace/notes/worker_result.txt 作为附件发送。"
        "如果第一次发送失败，修正后再试一次。完成后简短说明。"
    )

    planner = run_stage("planner-send", "SeedAgent", SEED_URL, planner_prompt)
    worker = run_stage("worker-handle", "WorkerAgent", WORKER_URL, worker_prompt)

    messages = receive_seed_messages()
    verification = verify_delivery(messages, "WORKER_DONE RESULT:42")
    if not verification["ok"]:
        print(f"[repair] trigger reason={verification['reason']}")
        repair = run_stage("worker-repair", "WorkerAgent", WORKER_URL, repair_prompt)
        messages = receive_seed_messages()
        verification = verify_delivery(messages, "WORKER_REDELIVER RESULT:42")
        print(f"[repair] result ok={repair['ok']} error={repair['error']}")

    print(f"[summary] planner_ok={planner['ok']} planner_error={planner['error']}")
    print(f"[summary] worker_ok={worker['ok']} worker_error={worker['error']}")
    print(f"[summary] delivery_ok={verification['ok']} reason={verification.get('reason')}")
    if verification.get("copied_file"):
        print(f"[summary] copied_file={verification['copied_file']}")

    return 0 if planner["ok"] and worker["ok"] and verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
