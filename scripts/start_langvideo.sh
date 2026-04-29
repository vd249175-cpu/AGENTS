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
  "${PYTHON_CMD[@]}" - "${MAIN_URL}" "${STARTED_AGENTS_FILE}" ${AGENTS} <<'PY'
import json
import sys
import urllib.request

main_url = sys.argv[1].rstrip("/")
started_file = sys.argv[2]
agents = sys.argv[3:]

started = []
for agent_name in agents:
    payload = json.dumps(
        {
            "host": "127.0.0.1",
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
    print(f"agent: {agent_name} {status} {data.get('service_url') or ''}")
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
