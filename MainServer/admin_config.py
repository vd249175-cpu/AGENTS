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
    return {
        "agents": {},
        "communication": {"spaces": []},
        "ui": {"agent_positions": {}, "chat_sessions": {}},
    }


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
    communication = data.setdefault("communication", {"spaces": []})
    if not isinstance(communication, dict):
        raise ValueError("agent config 'communication' must be an object")
    spaces = communication.setdefault("spaces", [])
    if not isinstance(spaces, list):
        raise ValueError("agent config 'communication.spaces' must be a list")
    ui = data.setdefault("ui", {"agent_positions": {}, "chat_sessions": {}})
    if not isinstance(ui, dict):
        raise ValueError("agent config 'ui' must be an object")
    return data


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    agents = config.setdefault("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("agent config 'agents' must be an object")
    communication = config.setdefault("communication", {"spaces": []})
    if not isinstance(communication, dict):
        raise ValueError("agent config 'communication' must be an object")
    spaces = communication.setdefault("spaces", [])
    if not isinstance(spaces, list):
        raise ValueError("agent config 'communication.spaces' must be a list")
    ui = config.setdefault("ui", {"agent_positions": {}, "chat_sessions": {}})
    if not isinstance(ui, dict):
        raise ValueError("agent config 'ui' must be an object")
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


def remove_agent_config(agent_name: str) -> dict[str, Any] | None:
    config = load_config()
    agents = config.setdefault("agents", {})
    removed = agents.pop(agent_name, None)
    communication = config.setdefault("communication", {"spaces": []})
    for space in communication.get("spaces", []):
        if isinstance(space, dict):
            members = space.get("members", [])
            if isinstance(members, list):
                space["members"] = [member for member in members if member != agent_name]
    ui = config.setdefault("ui", {"agent_positions": {}, "chat_sessions": {}})
    positions = ui.get("agent_positions", {})
    if isinstance(positions, dict):
        positions.pop(agent_name, None)
    chat_sessions = ui.get("chat_sessions", {})
    if isinstance(chat_sessions, dict):
        chat_sessions.pop(agent_name, None)
    save_config(config)
    return dict(removed) if isinstance(removed, dict) else None


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


def _clean_members(value: Any) -> list[str]:
    members: list[str] = []
    seen: set[str] = set()
    if not isinstance(value, list):
        return []
    for item in value:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        members.append(name)
    return members


def normalize_communication(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    spaces: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw.get("spaces", [])):
        if not isinstance(item, dict):
            continue
        members = _clean_members(item.get("members", []))
        if not members:
            continue
        raw_id = str(item.get("id") or item.get("name") or f"space-{index + 1}").strip()
        space_id = raw_id or f"space-{index + 1}"
        if space_id in seen_ids:
            space_id = f"{space_id}-{index + 1}"
        seen_ids.add(space_id)
        spaces.append(
            {
                "id": space_id,
                "name": str(item.get("name") or space_id).strip() or space_id,
                "members": members,
                "color": str(item.get("color") or "").strip() or None,
            }
        )
    return {"spaces": spaces}


def get_communication_config() -> dict[str, Any]:
    return normalize_communication(load_config().get("communication"))


def replace_communication_config(value: dict[str, Any]) -> dict[str, Any]:
    communication = normalize_communication(value)
    config = load_config()
    config["communication"] = communication
    save_config(config)
    return communication


def agent_spaces(agent_name: str) -> list[dict[str, Any]]:
    return [
        space
        for space in get_communication_config().get("spaces", [])
        if agent_name in space.get("members", [])
    ]


def communication_peers(agent_name: str) -> list[str] | None:
    spaces = get_communication_config().get("spaces", [])
    if not spaces:
        return None
    peers: list[str] = []
    seen: set[str] = set()
    for space in spaces:
        members = space.get("members", [])
        if agent_name not in members:
            continue
        for member in members:
            if member == agent_name or member in seen:
                continue
            seen.add(member)
            peers.append(member)
    return peers


def communication_allows(sender: str, receiver: str) -> bool:
    if sender == receiver:
        return True
    peers = communication_peers(sender)
    if peers is None:
        return True
    return receiver in peers


def communication_edges(agent_names: list[str] | None = None) -> list[dict[str, Any]]:
    allowed = set(agent_names or [])
    edges: dict[tuple[str, str], set[str]] = {}
    spaces = get_communication_config().get("spaces", [])
    for space in spaces:
        members = [name for name in space.get("members", []) if not allowed or name in allowed]
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                key = tuple(sorted((left, right)))
                edges.setdefault(key, set()).add(space["id"])
    return [
        {"from": left, "to": right, "spaces": sorted(space_ids)}
        for (left, right), space_ids in sorted(edges.items())
    ]


def get_ui_state() -> dict[str, Any]:
    ui = load_config().setdefault("ui", {"agent_positions": {}, "chat_sessions": {}})
    positions = ui.get("agent_positions", {})
    chat_sessions = ui.get("chat_sessions", {})
    return {
        "agent_positions": positions if isinstance(positions, dict) else {},
        "chat_sessions": chat_sessions if isinstance(chat_sessions, dict) else {},
    }


def clear_agent_chat_session(agent_name: str) -> dict[str, Any]:
    config = load_config()
    ui = config.setdefault("ui", {"agent_positions": {}, "chat_sessions": {}})
    chat_sessions = ui.setdefault("chat_sessions", {})
    if isinstance(chat_sessions, dict):
        chat_sessions.pop(agent_name, None)
    save_config(config)
    return get_ui_state()


def replace_ui_state(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("ui state must be an object")
    config = load_config()
    current = get_ui_state()
    next_ui = {
        "agent_positions": value.get("agent_positions", current["agent_positions"]),
        "chat_sessions": value.get("chat_sessions", current["chat_sessions"]),
    }
    config["ui"] = next_ui
    save_config(config)
    return get_ui_state()
