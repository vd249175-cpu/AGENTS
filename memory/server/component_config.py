"""Small helpers for demo-style component wrappers."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict


class StrictConfig(BaseModel):
    """Base config behavior for wrapper-facing configuration models."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_default=True,
    )


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def read_external_config(source: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return dict(source)
    return json.loads(Path(source).expanduser().read_text(encoding="utf-8"))


def config_from_external(model_cls: type[ConfigT], source: dict[str, Any] | str | Path | None = None) -> ConfigT:
    """Load a dict/json source into a validated Pydantic config instance."""

    return model_cls.model_validate(read_external_config(source))


def emit(writer: Any, payload: dict[str, object]) -> None:
    if writer is None:
        return
    try:
        writer(payload)
    except TypeError:
        write = getattr(writer, "write", None)
        if callable(write):
            write(str(payload))
