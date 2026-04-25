import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_CONFIG = PROJECT_ROOT / "MainServer" / "config" / "agents.local.json"


def config_path() -> Path:
    override = os.getenv("MAIN_SERVER_AGENT_CONFIG")
    if override:
        return Path(override).expanduser()
    return DEFAULT_AGENT_CONFIG


def empty_config() -> dict[str, Any]:
    return {"agents": {}}


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return empty_config()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("agent config root must be an object")
    agents = data.setdefault("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("agent config 'agents' must be an object")
    return data


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    agents = config.setdefault("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("agent config 'agents' must be an object")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config


def get_agent_config(agent_name: str) -> dict[str, Any]:
    config = load_config()
    value = config.get("agents", {}).get(agent_name, {})
    if not isinstance(value, dict):
        raise ValueError(f"agent config for {agent_name!r} must be an object")
    return dict(value)


def update_agent_config(agent_name: str, patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("agent config patch must be an object")
    config = load_config()
    agents = config.setdefault("agents", {})
    current = agents.get(agent_name, {})
    if not isinstance(current, dict):
        current = {}
    merged = {**current, **patch}
    agents[agent_name] = merged
    save_config(config)
    return dict(merged)


def replace_agent_config(agent_name: str, value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("agent config value must be an object")
    config = load_config()
    config.setdefault("agents", {})[agent_name] = dict(value)
    save_config(config)
    return dict(value)


def configured_scope(agent_name: str) -> tuple[bool, Any]:
    agent_config = get_agent_config(agent_name)
    if "scope" not in agent_config:
        return False, None
    return True, agent_config.get("scope")


def resolve_scope(scope: Any) -> list[str] | None:
    if scope is None:
        return None

    resolved: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        name = value.strip()
        if not name or name in seen:
            return
        seen.add(name)
        resolved.append(name)

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            add(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for key in ("agent_name", "agent", "name", "to", "dst"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    add(candidate)
            for key in ("scope", "peers", "include", "children", "items"):
                if key in value:
                    visit(value[key])

    visit(scope)
    return resolved


def scope_allows(scope: Any, agent_name: str) -> bool:
    resolved = resolve_scope(scope)
    if resolved is None:
        return True
    return agent_name in resolved
