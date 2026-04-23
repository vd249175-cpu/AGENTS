"""Agent-local path helpers."""

import os
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
SEED_AGENT_ROOT = AGENT_ROOT.parent
DEEPAGENTS_ROOT = SEED_AGENT_ROOT.parent
RUNTIME_AGENT_NAME = os.getenv("AGENT_NAME") or SEED_AGENT_ROOT.name
WORKSPACE_ROOT = DEEPAGENTS_ROOT / RUNTIME_AGENT_NAME / "workspace"
STORE_ROOT = AGENT_ROOT / "store"


def resolve_store_path(value: str | Path | None, *, default_relative: str = ".") -> Path:
    raw = str(value or default_relative).strip() or default_relative
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return STORE_ROOT / candidate


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
