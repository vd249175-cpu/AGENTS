#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${LANGVIDEO_HOST:-127.0.0.1}"
MAIN_PORT="${LANGVIDEO_MAIN_PORT:-8000}"
FRONTEND_PORT="${LANGVIDEO_FRONTEND_PORT:-5173}"
MAIN_URL="http://${HOST}:${MAIN_PORT}"
FRONTEND_URL="http://${HOST}:${FRONTEND_PORT}"
LOG_DIR="${LANGVIDEO_LOG_DIR:-${ROOT}/tests/logs/dev-start}"
START_AGENTS="${LANGVIDEO_START_AGENTS:-1}"
OPEN_BROWSER="${LANGVIDEO_OPEN_BROWSER:-1}"
AGENTS="${LANGVIDEO_AGENTS:-KnowledgeSeedAgent SeedAgent}"
KILL_OLD="${LANGVIDEO_KILL_OLD:-1}"
STARTED_AGENTS_FILE="${LOG_DIR}/started_agents.txt"

MAIN_PID=""
FRONTEND_PID=""
PYTHON_CMD=()
CLEANED_UP=0

mkdir -p "${LOG_DIR}"
: > "${STARTED_AGENTS_FILE}"

if [ -n "${LANGVIDEO_PYTHON:-}" ]; then
  PYTHON_CMD=("${LANGVIDEO_PYTHON}")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=("python3")
else
  PYTHON_CMD=("uv" "run" "python")
fi

is_http_ok() {
  local url="$1"
  "${PYTHON_CMD[@]}" - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=0.5) as response:
        raise SystemExit(0 if response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_url() {
  local url="$1"
  local label="$2"
  for _ in $(seq 1 80); do
    if is_http_ok "$url"; then
      echo "${label}: ready ${url}"
      return 0
    fi
    sleep 0.25
  done
  echo "${label}: failed to become ready at ${url}" >&2
  return 1
}

agent_list_args() {
  # shellcheck disable=SC2086
  printf "%s\n" ${AGENTS}
}

registered_agent_ports() {
  if ! is_http_ok "${MAIN_URL}/healthz"; then
    return
  fi
  "${PYTHON_CMD[@]}" - "${MAIN_URL}" <<'PY' 2>/dev/null || true
import json
import sys
import urllib.parse
import urllib.request

main_url = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(f"{main_url}/admin/monitor", timeout=2) as response:
        data = json.loads(response.read().decode("utf-8") or "{}")
except Exception:
    raise SystemExit(0)

ports = set()
for agent in data.get("agents", []):
    metadata = agent.get("metadata") or {}
    port = metadata.get("service_port")
    if port:
        ports.add(str(port))
        continue
    service_url = metadata.get("service_url") or ""
    parsed = urllib.parse.urlparse(service_url)
    if parsed.port:
        ports.add(str(parsed.port))

for port in sorted(ports, key=int):
    print(port)
PY
}

stop_registered_agents() {
  if [ "${START_AGENTS}" != "1" ] || ! is_http_ok "${MAIN_URL}/healthz"; then
    return
  fi
  "${PYTHON_CMD[@]}" - "${MAIN_URL}" ${AGENTS} <<'PY' || true
import json
import sys
import urllib.request

main_url = sys.argv[1].rstrip("/")
configured = set(sys.argv[2:])
names = set(configured)
try:
    with urllib.request.urlopen(f"{main_url}/admin/monitor", timeout=2) as response:
        data = json.loads(response.read().decode("utf-8") or "{}")
    names.update(agent.get("agent_name") for agent in data.get("agents", []) if agent.get("agent_name"))
except Exception:
    pass

for agent_name in sorted(names):
    request = urllib.request.Request(
        f"{main_url}/admin/agents/{agent_name}/service/stop",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3):
            pass
        print(f"agent: {agent_name} old service stop requested")
    except Exception as exc:
        print(f"agent: {agent_name} old service stop skipped: {type(exc).__name__}: {exc}")
PY
}

kill_pids() {
  local label="$1"
  shift
  local pids=("$@")
  if [ "${#pids[@]}" -eq 0 ]; then
    return
  fi
  echo "${label}: stopping pids ${pids[*]}"
  kill "${pids[@]}" >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    local alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        alive+=("${pid}")
      fi
    done
    if [ "${#alive[@]}" -eq 0 ]; then
      return
    fi
    sleep 0.15
  done
  kill -9 "${pids[@]}" >/dev/null 2>&1 || true
}

kill_listening_port() {
  local port="$1"
  local label="$2"
  if ! command -v lsof >/dev/null 2>&1; then
    echo "${label}: lsof not found, cannot kill listeners on port ${port}"
    return
  fi
  local pids
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true)"
  if [ -z "${pids}" ]; then
    return
  fi
  # shellcheck disable=SC2206
  local pid_array=(${pids})
  kill_pids "${label}:${port}" "${pid_array[@]}"
}

kill_agent_processes_by_name() {
  if ! command -v pgrep >/dev/null 2>&1; then
    return
  fi
  local agent_name
  while IFS= read -r agent_name; do
    [ -n "${agent_name}" ] || continue
    local pids
    pids="$(pgrep -f "Deepagents\\.${agent_name}\\.AgentServer" 2>/dev/null || true)"
    if [ -z "${pids}" ]; then
      continue
    fi
    # shellcheck disable=SC2206
    local pid_array=(${pids})
    kill_pids "agent:${agent_name}" "${pid_array[@]}"
  done < <(agent_list_args)
}

stop_old_stack() {
  if [ "${KILL_OLD}" != "1" ]; then
    echo "restart: preserving old processes because LANGVIDEO_KILL_OLD=${KILL_OLD}"
    return
  fi

  echo "restart: stopping old Long River Agent processes"
  local ports=()
  while IFS= read -r port; do
    [ -n "${port}" ] && ports+=("${port}")
  done < <(registered_agent_ports)

  stop_registered_agents
  kill_agent_processes_by_name
  kill_listening_port "${MAIN_PORT}" "mainserver"
  kill_listening_port "${FRONTEND_PORT}" "frontend"
  if [ "${#ports[@]}" -gt 0 ]; then
    local port
    for port in "${ports[@]}"; do
      kill_listening_port "${port}" "agent"
    done
  fi
}

start_mainserver() {
  if is_http_ok "${MAIN_URL}/healthz"; then
    echo "mainserver: already running ${MAIN_URL}"
    return
  fi
  echo "mainserver: starting ${MAIN_URL}"
  (
    cd "${ROOT}"
    uv run uvicorn MainServer.main_server:app --host "${HOST}" --port "${MAIN_PORT}" --log-level info
  ) >"${LOG_DIR}/mainserver.log" 2>&1 &
  MAIN_PID="$!"
  wait_for_url "${MAIN_URL}/healthz" "mainserver"
}

start_frontend() {
  if is_http_ok "${FRONTEND_URL}"; then
    echo "frontend: already running ${FRONTEND_URL}"
    return
  fi
  if [ ! -d "${ROOT}/frontend/node_modules" ]; then
    echo "frontend: installing npm dependencies"
    npm --prefix "${ROOT}/frontend" install
  fi
  echo "frontend: starting ${FRONTEND_URL}"
  npm --prefix "${ROOT}/frontend" run dev -- --host "${HOST}" --port "${FRONTEND_PORT}" \
    >"${LOG_DIR}/frontend.log" 2>&1 &
  FRONTEND_PID="$!"
  wait_for_url "${FRONTEND_URL}" "frontend"
}

start_agents() {
  if [ "${START_AGENTS}" != "1" ]; then
    echo "agents: skipped by LANGVIDEO_START_AGENTS=${START_AGENTS}"
    return
  fi
  "${PYTHON_CMD[@]}" - "${MAIN_URL}" "${STARTED_AGENTS_FILE}" "${HOST}" ${AGENTS} <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

main_url = sys.argv[1].rstrip("/")
started_file = sys.argv[2]
host = sys.argv[3]
agents = sys.argv[4:]


def wait_for_agent(url: str | None, timeout: float = 30.0) -> bool:
    if not url:
        return False
    deadline = time.time() + timeout
    health_url = url.rstrip("/") + "/healthz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status < 500:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    return False


def service_url_from(data: dict) -> str | None:
    if data.get("service_url"):
        return data["service_url"]
    agent = data.get("agent") or {}
    metadata = agent.get("metadata") or {}
    return metadata.get("service_url")

started = []
for agent_name in agents:
    payload = json.dumps(
        {
            "host": host,
            "port": None,
            "main_server_url": main_url,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{main_url}/admin/agents/{agent_name}/service/start",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8") or "{}")
    except Exception as exc:
        print(f"agent: {agent_name} start failed: {type(exc).__name__}: {exc}")
        continue
    status = "already running" if data.get("already_running") else "started"
    service_url = service_url_from(data)
    ready = wait_for_agent(service_url)
    ready_text = "ready" if ready else "not-ready"
    print(f"agent: {agent_name} {status} {ready_text} {service_url or ''}")
    if data.get("ok") and not data.get("already_running"):
        started.append(agent_name)

with open(started_file, "w", encoding="utf-8") as handle:
    handle.write("\n".join(started))
PY
}

stop_started_agents() {
  if [ ! -s "${STARTED_AGENTS_FILE}" ]; then
    return
  fi
  "${PYTHON_CMD[@]}" - "${MAIN_URL}" "${STARTED_AGENTS_FILE}" <<'PY' || true
import sys
import urllib.request

main_url = sys.argv[1].rstrip("/")
with open(sys.argv[2], encoding="utf-8") as handle:
    agents = [line.strip() for line in handle if line.strip()]
for agent_name in reversed(agents):
    request = urllib.request.Request(
        f"{main_url}/admin/agents/{agent_name}/service/stop",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
        print(f"agent: {agent_name} stopped")
    except Exception as exc:
        print(f"agent: {agent_name} stop skipped: {type(exc).__name__}: {exc}")
PY
}

cleanup() {
  if [ "${CLEANED_UP}" = "1" ]; then
    return
  fi
  CLEANED_UP=1
  echo
  echo "shutdown: stopping services started by this script"
  stop_started_agents
  if [ -n "${FRONTEND_PID}" ]; then
    kill "${FRONTEND_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${MAIN_PID}" ]; then
    kill "${MAIN_PID}" >/dev/null 2>&1 || true
  fi
}

handle_signal() {
  cleanup
  exit 0
}

trap cleanup EXIT
trap handle_signal INT TERM

stop_old_stack
start_mainserver
start_frontend
start_agents

if [ "${OPEN_BROWSER}" = "1" ]; then
  if command -v open >/dev/null 2>&1; then
    open "${FRONTEND_URL}"
  else
    "${PYTHON_CMD[@]}" -m webbrowser "${FRONTEND_URL}" >/dev/null 2>&1 || true
  fi
fi

echo "app: ${FRONTEND_URL}"
echo "api: ${MAIN_URL}"
echo "logs: ${LOG_DIR}"
echo "press Ctrl-C to stop services started by this script"

while true; do
  sleep 3600
done
