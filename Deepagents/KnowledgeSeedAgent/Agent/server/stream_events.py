"""Shared stream event helpers."""

from typing import Any


def emit_event(writer: Any, payload: dict[str, Any]) -> None:
    if writer is None:
        return
    try:
        writer(payload)
    except TypeError:
        write = getattr(writer, "write", None)
        if callable(write):
            write(str(payload))
    except Exception:
        return

