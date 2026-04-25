import json
import secrets
import string
import threading
import time
import sys
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

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


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
        print(f"[health] {url} ok={data.get('status') == 'ok'}", flush=True)


def clear_inbox(agent_name: str) -> None:
    data = get_json(f"{MAIN_SERVER}/recv/{agent_name}")
    print(f"[setup] cleared inbox {agent_name}: {len(data.get('messages', []))}", flush=True)


def reset_workspace() -> None:
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
        notes_dir = workspace / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        for item in notes_dir.glob("browser_*"):
            if item.is_file():
                item.unlink()
    print("[setup] workspaces reset", flush=True)


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
            parts = [f"[{label}] middleware {event.get('middleware')} {phase} {lifecycle}"]
            if phase == "before_agent" and isinstance(event.get("state"), dict):
                parts.append(f"messages={len(event['state'].get('messages', []))}")
            if phase == "wrap_model_call" and isinstance(event.get("request"), dict):
                request = event["request"]
                parts.append(f"request_messages={request.get('message_count')}")
                parts.append(f"tools={request.get('tool_count')}")
                if request.get("tool_choice") is not None:
                    parts.append(f"tool_choice={request.get('tool_choice')}")
            if phase == "wrap_tool_call" and isinstance(event.get("tool_call"), dict):
                tool_call = event["tool_call"]
                parts.append(f"tool={tool_call.get('tool_name')}")
                if tool_call.get("tool_args_keys") is not None:
                    parts.append(f"args={tool_call.get('tool_args_keys')}")
            if phase in {"wrap_model_call", "wrap_tool_call", "after_agent"} and event.get("result_type"):
                parts.append(f"result={event.get('result_type')}")
            return " ".join(parts)
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
                f"phase={snapshot.get('phase')} step={snapshot.get('step')}",
                flush=True,
            )
        for event in snapshot.get("events", []):
            signature = event_signature(event)
            if signature in self.seen_events:
                continue
            self.seen_events.add(signature)
            line = format_event(self.label, event)
            if line:
                print(line, flush=True)


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
    print(f"[{label}] setup agent={agent_name} session={session_id} run={run_id}", flush=True)
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
        print(f"[{label}] result error={outcome['error']}", flush=True)
    else:
        chunks = len((outcome["response"] or {}).get("chunks", []))
        print(f"[{label}] result ok={outcome['ok']} chunks={chunks}", flush=True)
    return outcome


def receive_seed_messages() -> list[dict[str, Any]]:
    data = get_json(f"{MAIN_SERVER}/recv/SeedAgent")
    messages = list(data.get("messages", []))
    print(f"[verify] SeedAgent inbox received={len(messages)}", flush=True)
    return messages


def verify_browser_delivery(messages: list[dict[str, Any]]) -> dict[str, Any]:
    target = None
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if message.get("from") == "WorkerAgent" and content.startswith("BROWSER_DONE Example Domain"):
            target = message
            break
    if target is None:
        return {"ok": False, "reason": "missing browser completion message"}
    attachments = target.get("attachments", [])
    if not attachments:
        return {"ok": False, "reason": "missing attachment", "message": target}
    attachment = attachments[0]
    if attachment.get("_routed") != "copied":
        return {"ok": False, "reason": f"attachment routed={attachment.get('_routed')}", "message": target}
    copied_file = Path(str(attachment.get("link") or ""))
    if not copied_file.exists():
        return {"ok": False, "reason": f"copied file missing: {copied_file}", "message": target}
    content = copied_file.read_text(encoding="utf-8").strip()
    if "Example Domain" not in content or "Learn more" not in content:
        return {"ok": False, "reason": f"snapshot content mismatch: {content!r}", "message": target}
    return {"ok": True, "message": target, "copied_file": str(copied_file)}


def main() -> int:
    healthcheck()
    reset_workspace()
    clear_inbox("SeedAgent")
    clear_inbox("WorkerAgent")

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

    planner = run_stage("planner-send", "SeedAgent", SEED_URL, planner_prompt)
    worker = run_stage("worker-handle", "WorkerAgent", WORKER_URL, worker_prompt)

    messages = receive_seed_messages()
    verification = verify_browser_delivery(messages)

    print(f"[summary] planner_ok={planner['ok']} planner_error={planner['error']}", flush=True)
    print(f"[summary] worker_ok={worker['ok']} worker_error={worker['error']}", flush=True)
    print(f"[summary] delivery_ok={verification['ok']} reason={verification.get('reason')}", flush=True)
    if verification.get("copied_file"):
        print(f"[summary] copied_file={verification['copied_file']}", flush=True)

    return 0 if planner["ok"] and worker["ok"] and verification["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
