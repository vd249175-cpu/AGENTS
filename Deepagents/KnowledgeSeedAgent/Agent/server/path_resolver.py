"""Agent-local path helpers."""

import os
from pathlib import Path, PurePosixPath


AGENT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_SEED_AGENT_ROOT = AGENT_ROOT.parent
DEEPAGENTS_ROOT = KNOWLEDGE_SEED_AGENT_ROOT.parent
RUNTIME_AGENT_NAME = os.getenv("AGENT_NAME") or KNOWLEDGE_SEED_AGENT_ROOT.name
WORKSPACE_ROOT = DEEPAGENTS_ROOT / RUNTIME_AGENT_NAME / "workspace"
STORE_ROOT = AGENT_ROOT / "store"
NETWORK_SCHEMES = ("http://", "https://", "ftp://", "s3://", "file://")


def resolve_store_path(value: str | Path | None, *, default_relative: str = ".") -> Path:
    raw = str(value or default_relative).strip() or default_relative
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return STORE_ROOT / candidate


def is_network_link(value: str) -> bool:
    return value.startswith(NETWORK_SCHEMES)


def workspace_visible_path(value: str | Path) -> str:
    """Return the path form an agent should see for local workspace files."""
    raw = str(value or "").strip()
    if not raw or is_network_link(raw):
        return raw
    if raw == "/workspace" or raw.startswith("/workspace/"):
        return raw

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(WORKSPACE_ROOT.resolve())
        except ValueError:
            return raw
        if not relative.parts:
            return "/workspace"
        return "/workspace/" + PurePosixPath(*relative.parts).as_posix()

    cleaned = PurePosixPath(raw)
    if cleaned.parts and cleaned.parts[0] == "workspace":
        cleaned = PurePosixPath(*cleaned.parts[1:]) if len(cleaned.parts) > 1 else PurePosixPath()
    if any(part == ".." for part in cleaned.parts):
        return raw
    return "/workspace" if not cleaned.parts else "/workspace/" + cleaned.as_posix().lstrip("/")


def ensure_seed_workspace() -> None:
    for path in (
        WORKSPACE_ROOT,
        WORKSPACE_ROOT / "brain",
        WORKSPACE_ROOT / "knowledge",
        WORKSPACE_ROOT / "notes",
        WORKSPACE_ROOT / "skills",
        STORE_ROOT,
        STORE_ROOT / "checkpoints",
        STORE_ROOT / "sqlite",
        STORE_ROOT / "state",
    ):
        path.mkdir(parents=True, exist_ok=True)
