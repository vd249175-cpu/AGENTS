import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODELS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG = MODELS_DIR / "model_config.json"
LEGACY_MAIN_AGENT_API_CONFIG = MODELS_DIR / "AgentApi.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model config format: {path}")
    return data


def _normalize_provider(value: Any, *, default: str) -> str:
    raw = str(value or default).strip().lower()
    if raw in {"openai", "openai-compatible"}:
        return "openai"
    if raw in {"openai_like", "openai-like"}:
        return "openai"
    if raw == "ollama":
        return "ollama"
    return default


def _chat_section(data: dict[str, Any], section_name: str) -> dict[str, Any]:
    if any(key in data for key in ("url", "base_url", "apikey", "api_key", "model", "ollama_model")):
        return data
    chat_model = data.get("chat_model")
    if isinstance(chat_model, dict):
        return chat_model
    nested = data.get(section_name)
    if isinstance(nested, dict):
        return nested
    for value in data.values():
        if isinstance(value, dict) and any(
            key in value for key in ("url", "base_url", "apikey", "api_key", "model", "ollama_model")
        ):
            return value
    raise ValueError(f"Cannot find a model section named '{section_name}'")


@dataclass(frozen=True, slots=True)
class MainAgentModelConfig:
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    model_name: str = "gpt-5-nano"
    model_candidates: tuple[str, ...] = ()
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "granite4:3b"
    temperature: float = 0.0


def load_main_agent_model_config(config_path: Path | None = None) -> MainAgentModelConfig:
    source = Path(config_path or os.getenv("LANGVIDEO_AGENT_MODEL_CONFIG") or DEFAULT_MODEL_CONFIG)
    if not source.exists() and source == DEFAULT_MODEL_CONFIG:
        source = LEGACY_MAIN_AGENT_API_CONFIG
    data = _chat_section(_load_json(source), "MainAgent")
    model_list = data.get("model") or []
    if isinstance(model_list, str):
        candidates = (model_list,)
    elif isinstance(model_list, list):
        candidates = tuple(str(item) for item in model_list if str(item).strip())
    else:
        candidates = ()

    provider = _normalize_provider(
        data.get("provider") or data.get("APIFormat") or os.getenv("LANGVIDEO_MODEL_PROVIDER"),
        default="openai",
    )
    model_name = os.getenv("LANGVIDEO_OPENAI_MODEL") or (
        candidates[0] if candidates else "gpt-5-nano"
    )
    return MainAgentModelConfig(
        provider=provider,
        base_url=str(
            data.get("base_url")
            or data.get("url")
            or os.getenv("LANGVIDEO_OPENAI_BASE_URL")
            or ""
        ).strip() or None,
        api_key=str(
            data.get("api_key")
            or data.get("apikey")
            or os.getenv("LANGVIDEO_OPENAI_API_KEY")
            or ""
        ).strip() or None,
        model_name=str(model_name),
        model_candidates=candidates,
        ollama_base_url=str(data.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
        ollama_model=str(data.get("ollama_model") or os.getenv("LANGVIDEO_OLLAMA_MODEL", "granite4:3b")),
        temperature=float(data.get("temperature", 0.0)),
    )


def build_main_agent_model(
    *,
    config_path: Path | None = None,
    provider: str | None = None,
) -> Any:
    config = load_main_agent_model_config(config_path=config_path)
    provider_name = (provider or config.provider).lower()

    if provider_name == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
        )

    from langchain_openai import ChatOpenAI

    if not config.base_url or not config.api_key:
        raise ValueError(
            "OpenAI-compatible configs must provide url and apikey, or set "
            "LANGVIDEO_OPENAI_BASE_URL and LANGVIDEO_OPENAI_API_KEY."
        )
    return ChatOpenAI(
        model=config.model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
    )


def build_main_agent_model_from_config(config: Any) -> Any:
    provider_name = _normalize_provider(
        getattr(config, "chatModelProvider", None),
        default="openai",
    )

    if provider_name == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=str(getattr(config, "chatModel", None) or "granite4:3b"),
            base_url=str(getattr(config, "ollamaBaseUrl", None) or "http://127.0.0.1:11434"),
            temperature=float(getattr(config, "chatTemperature", 0.0)),
        )

    from langchain_openai import ChatOpenAI

    base_url = str(getattr(config, "chatBaseUrl", "") or "").strip()
    api_key = str(getattr(config, "chatApiKey", "") or "").strip()
    if not base_url or not api_key:
        raise ValueError("KnowledgeSeedAgentConfig must provide chatBaseUrl and chatApiKey for OpenAI-compatible models.")
    return ChatOpenAI(
        model=str(getattr(config, "chatModel", None) or "gpt-5-nano"),
        base_url=base_url,
        api_key=api_key,
        temperature=float(getattr(config, "chatTemperature", 0.0)),
    )


def build_model(
    *,
    config_path: Path | None = None,
    provider: str | None = None,
) -> Any:
    return build_main_agent_model(config_path=config_path, provider=provider)
