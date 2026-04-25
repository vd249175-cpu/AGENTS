import json
import re
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPAGENTS_ROOT = PROJECT_ROOT / "Deepagents"


RUNTIME_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "cache",
    "checkpoint",
    "checkpoints",
    "logs",
    "mail",
    "sqlite",
    "staging",
    "state",
    "store",
}

RUNTIME_FILE_NAMES = {
    ".DS_Store",
    "chunk_apply.sqlite3",
    "chunk_cache.sqlite3",
    "chunk_staging.sqlite3",
}


def validate_agent_name(agent_name: str) -> str:
    name = agent_name.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("agent_name must start with a letter and contain only letters, numbers, and underscores")
    return name


def agent_root(agent_name: str) -> Path:
    return DEEPAGENTS_ROOT / validate_agent_name(agent_name)


def agent_runtime_config_paths(agent_name: str) -> dict[str, Path]:
    target_name = validate_agent_name(agent_name)
    config_dir = agent_root(target_name) / "Agent"
    return {
        "local": config_dir / f"{target_name}Config.local.json",
        "example": config_dir / f"{target_name}Config.example.json",
    }


def read_agent_runtime_config(agent_name: str) -> dict[str, Any]:
    paths = agent_runtime_config_paths(agent_name)
    source = paths["local"] if paths["local"].exists() else paths["example"]
    if not source.exists():
        raise FileNotFoundError(f"agent runtime config not found for {agent_name}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"agent runtime config must be an object: {source}")
    return {"path": str(source), "uses_local": source == paths["local"], "config": data}


def write_agent_runtime_config(
    agent_name: str,
    value: dict[str, Any],
    *,
    merge: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("agent runtime config must be an object")
    paths = agent_runtime_config_paths(agent_name)
    if merge:
        try:
            current = read_agent_runtime_config(agent_name)["config"]
        except FileNotFoundError:
            current = {}
        config = {**current, **value}
    else:
        config = dict(value)
    config.setdefault("agentName", validate_agent_name(agent_name))
    paths["local"].parent.mkdir(parents=True, exist_ok=True)
    paths["local"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"path": str(paths["local"]), "uses_local": True, "config": config}


def _snake_name(value: str) -> str:
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", value)
    if not words:
        return value.lower()
    return "_".join(word.lower() for word in words)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in RUNTIME_FILE_NAMES or name in RUNTIME_DIR_NAMES:
            ignored.add(name)
    return ignored


def _rewrite_text_file(path: Path, replacements: dict[str, str]) -> None:
    if path.suffix not in {".py", ".json", ".md", ".txt"}:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    updated = text
    for source, target in replacements.items():
        updated = updated.replace(source, target)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def _rewrite_path_names(root: Path, replacements: dict[str, str]) -> None:
    paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        rewritten_name = path.name
        for source, target in replacements.items():
            rewritten_name = rewritten_name.replace(source, target)
        if rewritten_name == path.name:
            continue
        target = path.with_name(rewritten_name)
        if target.exists():
            raise FileExistsError(f"cannot rename {path} to {target}: target exists")
        path.rename(target)


def clear_agent_runtime(
    *,
    agent_name: str,
    include_store: bool = True,
    include_mail: bool = True,
    include_knowledge: bool = False,
) -> dict[str, Any]:
    target_name = validate_agent_name(agent_name)
    agent_root_path = agent_root(target_name)
    if not agent_root_path.exists():
        raise FileNotFoundError(f"agent does not exist: {agent_root_path}")

    removed: list[str] = []
    targets: list[Path] = []
    if include_store:
        targets.append(agent_root_path / "Agent" / "store")
    if include_mail:
        targets.append(agent_root_path / "workspace" / "mail")
    if include_knowledge:
        targets.append(agent_root_path / "workspace" / "knowledge")

    for target in targets:
        if not target.exists():
            continue
        shutil.rmtree(target)
        removed.append(str(target))

    for required in (
        agent_root_path / "workspace" / "knowledge",
        agent_root_path / "Agent" / "store",
        agent_root_path / "Agent" / "store" / "memory" / "staging",
        agent_root_path / "Agent" / "store" / "memory" / "checkpoint",
        agent_root_path / "Agent" / "store" / "memory" / "cache",
        agent_root_path / "Agent" / "store" / "checkpoints",
        agent_root_path / "Agent" / "store" / "sqlite",
        agent_root_path / "Agent" / "store" / "state",
    ):
        required.mkdir(parents=True, exist_ok=True)

    return {
        "agent_name": target_name,
        "removed": removed,
        "include_store": include_store,
        "include_mail": include_mail,
        "include_knowledge": include_knowledge,
    }


def create_agent_from_template(
    *,
    agent_name: str,
    source_agent: str = "SeedAgent",
    overwrite: bool = False,
) -> dict[str, Any]:
    target_name = validate_agent_name(agent_name)
    source_name = validate_agent_name(source_agent)
    source_root = DEEPAGENTS_ROOT / source_name
    target_root = agent_root(target_name)
    if not source_root.exists():
        raise FileNotFoundError(f"source agent template does not exist: {source_root}")
    if target_root.exists() and not overwrite:
        raise FileExistsError(f"target agent already exists: {target_root}")
    if target_root.exists():
        shutil.rmtree(target_root)

    shutil.copytree(source_root, target_root, ignore=_copy_ignore)

    source_snake = _snake_name(source_name)
    target_snake = _snake_name(target_name)
    replacements = {
        source_name: target_name,
        source_name.lower(): target_name.lower(),
        source_name.upper(): target_name.upper(),
        source_snake: target_snake,
        source_snake.upper(): target_snake.upper(),
    }
    for path in target_root.rglob("*"):
        if path.is_file():
            _rewrite_text_file(path, replacements)
    _rewrite_path_names(target_root, replacements)

    for required in (
        target_root / "workspace" / "brain",
        target_root / "workspace" / "knowledge",
        target_root / "workspace" / "notes",
        target_root / "workspace" / "skills",
    ):
        required.mkdir(parents=True, exist_ok=True)
    cleanup = clear_agent_runtime(agent_name=target_name, include_store=True, include_mail=True)

    return {
        "agent_name": target_name,
        "source_agent": source_name,
        "path": str(target_root),
        "runtime_cleanup": cleanup,
    }


def delete_agent_directory(*, agent_name: str) -> dict[str, Any]:
    target_name = validate_agent_name(agent_name)
    target_root = agent_root(target_name)
    if not target_root.exists():
        raise FileNotFoundError(f"agent does not exist: {target_root}")
    shutil.rmtree(target_root)
    return {"agent_name": target_name, "path": str(target_root)}
