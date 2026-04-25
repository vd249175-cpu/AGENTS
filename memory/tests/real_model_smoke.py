"""Real model smoke test.

Run with:
    uv run python tests/real_model_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from urllib import request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_chat_model_config, get_embedding_model_config


def _post_json(url: str, *, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_chat_completion(url: str, *, api_key: str, payload: dict[str, Any]) -> str:
    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    chunks: list[str] = []
    print(json.dumps({"type": "model_stream", "event": "start", "url": url}, ensure_ascii=False), flush=True)
    with request.urlopen(req, timeout=60) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                print(json.dumps({"type": "model_stream", "event": "done"}, ensure_ascii=False), flush=True)
                break
            event = json.loads(data)
            delta = event.get("choices", [{}])[0].get("delta", {})
            content = str(delta.get("content") or "")
            if content:
                chunks.append(content)
                print(json.dumps({"type": "model_stream", "event": "delta", "content": content}, ensure_ascii=False), flush=True)
    return "".join(chunks)


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


def main() -> None:
    chat = get_chat_model_config()
    embedding = get_embedding_model_config()

    base_url = str(chat["base_url"]).rstrip("/")
    chat_reply = _stream_chat_completion(
        f"{base_url}/chat/completions",
        api_key=str(chat["api_key"]),
        payload={
            "model": str(chat["model"]),
            "messages": [
                {"role": "system", "content": "Reply with a single short line."},
                {"role": "user", "content": "Say OK from real model test."},
            ],
            "temperature": 0,
        },
    )
    embedding_response = _post_json(
        f"{str(embedding['base_url']).rstrip('/')}/embeddings",
        api_key=str(embedding["api_key"]),
        payload={
            "model": str(embedding["model"]),
            "input": "embedding smoke test",
            "dimensions": int(embedding["dimensions"]),
        },
    )
    embedding_dimensions_returned = len(embedding_response["data"][0]["embedding"])
    result = {
        "chat_model": {
            "provider": chat["provider"],
            "model": chat["model"],
            "base_url": chat["base_url"],
            "api_key_masked": _mask_secret(str(chat["api_key"])),
            "reply": chat_reply,
        },
        "embedding_model": {
            "provider": embedding["provider"],
            "model": embedding["model"],
            "base_url": embedding["base_url"],
            "api_key_masked": _mask_secret(str(embedding["api_key"])),
            "dimensions_requested": int(embedding["dimensions"]),
            "dimensions_returned": embedding_dimensions_returned,
        },
    }
    log_path = PROJECT_ROOT / "workspace" / "logs" / "real_model_smoke.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
