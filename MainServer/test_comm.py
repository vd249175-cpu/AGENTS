"""End-to-end communication smoke test.

Run MainServer first:
    uvicorn MainServer.main_server:app --port 8000

Then run:
    python -m MainServer.test_comm
"""

import json
import sys
import urllib.request
from typing import Any


BASE = "http://127.0.0.1:8000"


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str) -> Any:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    from MainServer.protocol import make_message

    _post(f"{BASE}/agents/register", {"agent_name": "AgentA"})
    _post(f"{BASE}/agents/register", {"agent_name": "AgentB"})

    online = _get(f"{BASE}/agents/online")["agents"]
    assert "AgentA" in online and "AgentB" in online

    message = make_message(src="AgentA", dst="AgentB", msg_type="message", content="hello")
    sent = _post(f"{BASE}/send", message)
    assert sent["ok"]

    received = _get(f"{BASE}/recv/AgentB")["messages"]
    assert len(received) == 1
    assert received[0]["from"] == "AgentA"
    assert received[0]["content"] == "hello"

    assert _get(f"{BASE}/recv/AgentB")["messages"] == []
    print("MainServer communication smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
