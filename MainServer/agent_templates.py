import re
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPAGENTS_ROOT = PROJECT_ROOT / "Deepagents"


RUNTIME_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "mail",
    "store",
}

RUNTIME_FILE_NAMES = {
    ".DS_Store",
}


def validate_agent_name(agent_name: str) -> str:
    name = agent_name.strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("agent_name must start with a letter and contain only letters, numbers, and underscores")
    return name


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


def create_agent_from_template(
    *,
    agent_name: str,
    source_agent: str = "SeedAgent",
    overwrite: bool = False,
) -> dict[str, Any]:
    target_name = validate_agent_name(agent_name)
    source_name = validate_agent_name(source_agent)
    source_root = DEEPAGENTS_ROOT / source_name
    target_root = DEEPAGENTS_ROOT / target_name
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

    for required in (
        target_root / "workspace" / "brain",
        target_root / "workspace" / "knowledge",
        target_root / "workspace" / "notes",
        target_root / "workspace" / "skills",
        target_root / "Agent" / "store",
    ):
        required.mkdir(parents=True, exist_ok=True)

    return {
        "agent_name": target_name,
        "source_agent": source_name,
        "path": str(target_root),
    }
