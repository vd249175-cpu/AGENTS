"""Small JSON config loader shared by Agent internals."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def load_pydantic_config(model: type[ConfigT], path: Path) -> ConfigT:
    data = load_json_object(path)
    return model.model_validate(data)

