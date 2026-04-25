"""Ephemeral discovery middleware for knowledge-base management flows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, TypedDict, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from server.component_config import config_from_external


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("management_discovery.json")

DEFAULT_MANAGEMENT_DISCOVERY_SYSTEM_PROMPT = """<SystemPrompt>
<CapabilityGuidance name="management_discovery">
你是知识库管理者。
管理过程中，要持续阅读 <ManagementDiscoveries> 区块，把已经确认的发现、风险和待跟进事项纳入后续判断。
这里记录的是本次知识库管理过程中的工作发现，不是长期记忆，也不要把它当成用户偏好。
如果工具结果暴露出冲突、缺口、异常或可复用结论，后续操作时应参考这些发现。
</CapabilityGuidance>
</SystemPrompt>"""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class ManagementDiscoveryItem(TypedDict):
    id: str
    source: str
    summary: str
    refs: list[str]


class ManagementDiscoveryMiddlewareConfig(BaseModel):
    max_items: int = Field(default=24, ge=1, description="发现区块允许保留的最大条目数。")
    max_total_chars: int = Field(default=4096, ge=1, description="发现区块允许保留的最大字符数。")
    max_summary_chars: int = Field(default=160, ge=32, description="单条发现摘要的最大字符数。")
    scan_message_limit: int = Field(default=128, ge=1, description="每次从最近多少条消息中提取发现。")

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "ManagementDiscoveryMiddlewareConfig":
        payload = _load_json(path)
        return cls(
            max_items=max(1, int(payload.get("max_items", 24))),
            max_total_chars=max(1, int(payload.get("max_total_chars", 4096))),
            max_summary_chars=max(32, int(payload.get("max_summary_chars", 160))),
            scan_message_limit=max(1, int(payload.get("scan_message_limit", 128))),
        )

    @classmethod
    def load_config_management_discovery_middleware(cls, source=None) -> "ManagementDiscoveryMiddlewareConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class ManagementDiscoveryPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


def render_management_discoveries_block(discoveries: list[ManagementDiscoveryItem] | None) -> str:
    normalized = normalize_management_discoveries(discoveries)
    if not normalized:
        return '<CapabilityState name="management_discovery">\n<ManagementDiscoveries>\n</ManagementDiscoveries>\n</CapabilityState>'
    lines = []
    for item in normalized:
        refs = ", ".join(item["refs"])
        ref_attr = f' refs="{refs}"' if refs else ""
        lines.append(
            f'<DiscoveryItem id="{item["id"]}" source="{item["source"]}"{ref_attr}>{item["summary"]}</DiscoveryItem>'
        )
    body = "\n".join(lines)
    return (
        '<CapabilityState name="management_discovery">\n'
        "<ManagementDiscoveries>\n"
        f"{body}\n"
        "</ManagementDiscoveries>\n"
        "</CapabilityState>"
    )


class ManagementDiscoveryStateTydict(AgentState, total=False):
    management_discoveries: list[ManagementDiscoveryItem]


middleware_runingconfig = ManagementDiscoveryMiddlewareConfig.load()
MiddlewareToolConfig = {"tools": {}, "toolStateTydicts": {}}


def normalize_management_discoveries(
    discoveries: list[ManagementDiscoveryItem] | list[dict[str, Any]] | None,
) -> list[ManagementDiscoveryItem]:
    normalized: list[ManagementDiscoveryItem] = []
    for item in discoveries or []:
        if not isinstance(item, Mapping):
            continue
        source = str(item.get("source") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not source or not summary:
            continue
        refs = [str(ref).strip() for ref in item.get("refs") or [] if str(ref).strip()]
        item_id = str(item.get("id") or _make_discovery_id(source=source, summary=summary, refs=refs)).strip()
        normalized.append(
            {
                "id": item_id,
                "source": source,
                "summary": summary,
                "refs": refs,
            }
        )
    return normalized


def _make_discovery_id(*, source: str, summary: str, refs: list[str]) -> str:
    digest = hashlib.sha1(f"{source}|{summary}|{'|'.join(refs)}".encode("utf-8")).hexdigest()
    return digest[:12]


def _trim_text(value: str, *, limit: int) -> str:
    cleaned = " ".join(str(value).split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 1)].rstrip()}…"


def _discovery_chars(item: ManagementDiscoveryItem) -> int:
    return len(item["summary"]) + sum(len(ref) for ref in item["refs"])


def _limit_discoveries(
    discoveries: list[ManagementDiscoveryItem],
    *,
    config: ManagementDiscoveryMiddlewareConfig,
) -> list[ManagementDiscoveryItem]:
    limited: list[ManagementDiscoveryItem] = []
    total_chars = 0
    for item in discoveries:
        if len(limited) >= config.max_items:
            break
        item_chars = _discovery_chars(item)
        if limited and total_chars + item_chars > config.max_total_chars:
            break
        limited.append(item)
        total_chars += item_chars
    return limited


def _merge_discoveries(
    base: list[ManagementDiscoveryItem],
    extra: list[ManagementDiscoveryItem],
    *,
    config: ManagementDiscoveryMiddlewareConfig,
) -> list[ManagementDiscoveryItem]:
    merged_by_id: dict[str, ManagementDiscoveryItem] = {}
    for item in [*base, *extra]:
        merged_by_id[item["id"]] = item
    ordered: list[ManagementDiscoveryItem] = []
    for item in [*reversed(base), *reversed(extra)]:
        if merged_by_id.get(item["id"]) == item and item not in ordered:
            ordered.append(item)
    ordered.reverse()
    return _limit_discoveries(ordered, config=config)


def _parse_tool_payload(message: BaseMessage) -> dict[str, Any] | None:
    if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
        return None
    content = message.content.strip()
    if not content or content[0] not in "{[":
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_management_discoveries(
    messages: list[Any] | None,
    *,
    config: ManagementDiscoveryMiddlewareConfig,
) -> list[ManagementDiscoveryItem]:
    extracted: list[ManagementDiscoveryItem] = []
    recent_messages = list(messages or [])[-config.scan_message_limit :]
    for message in recent_messages:
        payload = _parse_tool_payload(message)
        if payload is None:
            continue
        discovery = _payload_to_discovery(payload, config=config)
        if discovery is not None:
            extracted.append(discovery)
    return _limit_discoveries(extracted, config=config)


def _payload_to_discovery(
    payload: Mapping[str, Any],
    *,
    config: ManagementDiscoveryMiddlewareConfig,
) -> ManagementDiscoveryItem | None:
    source = str(payload.get("operation") or "tool").strip()
    status = str(payload.get("status") or "").strip().lower()
    summary = _summarize_payload(payload, source=source, status=status)
    if not summary:
        return None
    refs = _collect_refs(payload)
    clipped_summary = _trim_text(summary, limit=config.max_summary_chars)
    return {
        "id": _make_discovery_id(source=source, summary=clipped_summary, refs=refs),
        "source": source,
        "summary": clipped_summary,
        "refs": refs,
    }


def _summarize_payload(payload: Mapping[str, Any], *, source: str, status: str) -> str:
    message = str(payload.get("message") or "").strip()
    if status == "error":
        return f"{source} 失败：{message or '请复核本次操作结果。'}"

    if source == "create_chunk_document":
        document_name = str(payload.get("document_name") or "").strip()
        chunk_count = payload.get("chunk_count_after") or payload.get("chunk_count") or 1
        return f"文档 {document_name} 已建立，当前共有 {chunk_count} 个 chunk。"
    if source in {"manage_chunks", "insert_chunks", "update_chunks", "delete_chunks"}:
        document_name = str(payload.get("document_name") or "").strip()
        chunk_count = payload.get("chunk_count_after") or payload.get("chunk_count") or 0
        success_count = int(payload.get("success_count") or 0)
        failure_count = int(payload.get("failure_count") or 0)
        if failure_count:
            return f"文档 {document_name} 的 chunk 管理部分失败，成功 {success_count} 项，失败 {failure_count} 项。"
        return f"文档 {document_name} 已完成 chunk 调整，当前共有 {chunk_count} 个 chunk。"
    if source == "list_chunk_documents":
        document_count = int(payload.get("document_count") or len(payload.get("documents") or []))
        return f"当前 run_id 下可见 {document_count} 篇 chunk 文档。"
    if source == "query_chunk_positions":
        document_name = str(payload.get("document_name") or "").strip()
        chunk_count = int(payload.get("chunk_count") or len(payload.get("chunks") or []))
        return f"已读取文档 {document_name} 的 {chunk_count} 个目标 chunk。"
    if source in {"graph_manage_nodes", "graph_create_nodes", "graph_update_node", "graph_delete_nodes"}:
        success_count = int(payload.get("success_count") or 0)
        failure_count = int(payload.get("failure_count") or 0)
        action_count = int(payload.get("action_count") or len(payload.get("results") or []))
        if failure_count:
            return f"图侧共处理 {action_count} 个动作，成功 {success_count} 个，失败 {failure_count} 个。"
        return f"图侧共完成 {action_count} 个管理动作。"
    if source == "read_nodes":
        result_count = int(payload.get("result_count") or len(payload.get("results") or []))
        missing_count = int(payload.get("missing_id_count") or len(payload.get("missing_ids") or []))
        if missing_count:
            return f"已读取 {result_count} 个节点，另有 {missing_count} 个目标节点缺失。"
        return f"已读取 {result_count} 个图节点或文档 chunk。"
    if source in {"recall_nodes_by_keywords", "keyword_recall"}:
        result_count = int(payload.get("result_count") or len(payload.get("results") or []))
        return f"关键词召回返回了 {result_count} 个相关节点。"
    if source == "graph_distance_recall":
        result_count = int(payload.get("result_count") or len(payload.get("results") or []))
        anchor = str(payload.get("anchor_id") or payload.get("start_node_id") or "").strip()
        if anchor:
            return f"从节点 {anchor} 出发的距离召回返回了 {result_count} 个节点。"
        return f"距离召回返回了 {result_count} 个节点。"
    if source == "graph_mark_useful":
        node_ids = payload.get("node_ids") or payload.get("marked_ids") or []
        return f"useful 状态桶新增了 {len(node_ids)} 个节点。"
    if source == "graph_mark_blocked":
        node_ids = payload.get("node_ids") or payload.get("blocked_ids") or []
        return f"blocked 状态桶新增了 {len(node_ids)} 个节点。"
    if source == "graph_clear_blocked":
        cleared_ids = payload.get("cleared_ids") or []
        return f"blocked 状态桶清除了 {len(cleared_ids)} 个节点。"
    if message:
        return message
    return f"{source} 已返回可用结果。"


def _collect_refs(payload: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    document_name = str(payload.get("document_name") or "").strip()
    if document_name:
        refs.append(f"document:{document_name}")
    chunk_index = payload.get("chunk_index")
    if isinstance(chunk_index, int):
        refs.append(f"chunk_index:{chunk_index}")
    for field_name in ("ids", "missing_ids", "cleared_ids", "node_ids"):
        raw_ids = payload.get(field_name)
        if isinstance(raw_ids, list):
            for raw_id in raw_ids[:4]:
                cleaned = str(raw_id).strip()
                if cleaned:
                    refs.append(f"node:{cleaned}")
    results = payload.get("results")
    if isinstance(results, list):
        for item in results[:4]:
            if not isinstance(item, Mapping):
                continue
            node_id = str(item.get("node_id") or "").strip()
            if node_id:
                refs.append(f"node:{node_id}")
                continue
            result_document_name = str(item.get("document_name") or "").strip()
            result_chunk_index = item.get("chunk_index")
            if result_document_name and isinstance(result_chunk_index, int):
                refs.append(f"chunk:{result_document_name}#{result_chunk_index}")
    deduped: list[str] = []
    for ref in refs:
        if ref not in deduped:
            deduped.append(ref)
    return deduped[:6]


middleware_capability_prompts = [
    ManagementDiscoveryPrompt(name="management_discovery.guidance", prompt=DEFAULT_MANAGEMENT_DISCOVERY_SYSTEM_PROMPT),
    ManagementDiscoveryPrompt(name="management_discovery.state", prompt=render_management_discoveries_block([])),
]


class ManagementDiscoveryMiddleware(AgentMiddleware):
    name = "management_discovery"
    capability_name = "management_discovery"
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = middleware_runingconfig
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = ManagementDiscoveryStateTydict  # type: ignore[assignment]

    def __init__(self, *, config: ManagementDiscoveryMiddlewareConfig | None = None) -> None:
        super().__init__()
        self.config = config or ManagementDiscoveryMiddlewareConfig.load()
        self.system_prompt = DEFAULT_MANAGEMENT_DISCOVERY_SYSTEM_PROMPT
        self.capability_prompts = self.capabilityPromptConfigs
        self.capability_prompt_configs = self.capabilityPromptConfigs
        self.middleware_runingconfig = self.config
        self.runing_config = self.config
        self.middleware = self

    def before_agent(self, state: ManagementDiscoveryStateTydict, runtime: Runtime[Any]) -> dict[str, Any] | None:
        normalized = normalize_management_discoveries(state.get("management_discoveries"))
        if normalized == list(state.get("management_discoveries") or []):
            return None
        return {"management_discoveries": normalized}

    def before_model(self, state: ManagementDiscoveryStateTydict, runtime: Runtime[Any]) -> dict[str, Any] | None:
        existing = normalize_management_discoveries(state.get("management_discoveries"))
        derived = extract_management_discoveries(state.get("messages"), config=self.config)
        merged = _merge_discoveries(existing, derived, config=self.config)
        if merged == existing:
            return None
        return {"management_discoveries": merged}

    def _upsert_named_system_message(self, messages: list[Any], *, name: str, text: str) -> list[Any]:
        new_message = SystemMessage(name=name, content=text)
        replaced = False
        new_messages = []
        for message in messages:
            if isinstance(message, SystemMessage) and message.name == name:
                new_messages.append(new_message)
                replaced = True
            else:
                new_messages.append(message)
        if not replaced:
            new_messages.insert(0, new_message)
        return new_messages

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler,
    ) -> ModelResponse[Any] | AIMessage:
        state = cast(ManagementDiscoveryStateTydict, request.state)
        existing = normalize_management_discoveries(state.get("management_discoveries"))
        derived = extract_management_discoveries(list(request.messages), config=self.config)
        discoveries = _merge_discoveries(existing, derived, config=self.config)
        messages = self._upsert_named_system_message(
            list(request.messages),
            name="management_discovery.guidance",
            text=self.system_prompt,
        )
        messages = self._upsert_named_system_message(
            messages,
            name="management_discovery.state",
            text=render_management_discoveries_block(discoveries),
        )
        return handler(request.override(messages=messages))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        state = cast(ManagementDiscoveryStateTydict, request.state)
        existing = normalize_management_discoveries(state.get("management_discoveries"))
        derived = extract_management_discoveries(list(request.messages), config=self.config)
        discoveries = _merge_discoveries(existing, derived, config=self.config)
        messages = self._upsert_named_system_message(
            list(request.messages),
            name="management_discovery.guidance",
            text=self.system_prompt,
        )
        messages = self._upsert_named_system_message(
            messages,
            name="management_discovery.state",
            text=render_management_discoveries_block(discoveries),
        )
        return await handler(request.override(messages=messages))

    def render_state_prompt(self, state: Mapping[str, Any] | None = None) -> str:
        discoveries = normalize_management_discoveries((state or {}).get("management_discoveries"))
        return render_management_discoveries_block(discoveries)


__all__ = [
    "DEFAULT_MANAGEMENT_DISCOVERY_SYSTEM_PROMPT",
    "ManagementDiscoveryItem",
    "ManagementDiscoveryMiddleware",
    "ManagementDiscoveryMiddlewareConfig",
    "ManagementDiscoveryStateTydict",
    "extract_management_discoveries",
    "render_management_discoveries_block",
]
