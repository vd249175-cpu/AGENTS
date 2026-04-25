"""Line-number-based split tool for the migrated chunk module."""

import json
import re
from html import escape
from pathlib import Path
from secrets import token_hex
from typing import Any, Callable
from dataclasses import dataclass

from langchain.agents.middleware import AgentState
from pydantic import BaseModel, Field, model_validator

from server.component_config import config_from_external


TOOL_CONFIG_PATH = Path(__file__).with_name("split_chunk.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class DisplayLine:
    text: str
    start: int
    end: int
    select_end: int


@dataclass(frozen=True)
class NumberedWindowLine:
    line_number: int
    text: str
    start: int
    end: int
    select_end: int


@dataclass(frozen=True)
class ChunkWindowView:
    history_lines: list[DisplayLine]
    active_lines: list[NumberedWindowLine]
    preview_lines: list[DisplayLine]

    @property
    def has_remaining_text(self) -> bool:
        return bool(self.active_lines)


class SplitChunkStateTydict(AgentState, total=False):
    document_body: str
    document_name: str | None
    messages: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    cursor: int
    retry_count: int
    max_retries: int
    history_line_count: int
    active_line_count: int
    preview_line_count: int
    line_wrap_width: int
    window_back_bytes: int | None
    window_forward_bytes: int | None
    process_trace: list[dict[str, Any]]


class SplitChunkAction(BaseModel):
    summary: str = Field(description="当前 chunk 的摘要。")
    keywords: list[str] = Field(default_factory=list, description="当前 chunk 的关键词列表。")
    line_end: int = Field(description="在当前 Window 内选择的结束行号，从 0 开始。")
    max_retries: int | None = Field(default=None, gt=0, description="该动作允许的最大重试次数；空值表示使用工具默认值。")

    @model_validator(mode="after")
    def validate_action(self) -> "SplitChunkAction":
        self.summary = " ".join(self.summary.split()).strip()
        self.keywords = [keyword.strip() for keyword in self.keywords if keyword.strip()]
        if not self.summary:
            raise ValueError("summary is required")
        if not self.keywords:
            raise ValueError("keywords is required")
        if self.line_end < 0:
            raise ValueError("line_end must be greater than or equal to 0")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitChunkAction":
        summary = " ".join(str(payload.get("summary", "")).split()).strip()
        keywords = [str(item).strip() for item in payload.get("keywords", []) if str(item).strip()]
        line_end = int(payload.get("line_end"))
        max_retries = payload.get("max_retries")
        if not summary:
            raise ValueError("summary is required")
        if not keywords:
            raise ValueError("keywords is required")
        if line_end < 0:
            raise ValueError("line_end must be greater than or equal to 0")
        if max_retries is not None and int(max_retries) <= 0:
            raise ValueError("max_retries must be greater than 0")
        return cls(summary=summary, keywords=keywords, line_end=line_end, max_retries=int(max_retries) if max_retries is not None else None)


class SplitChunkBatchInput(BaseModel):
    items: list[SplitChunkAction] = Field(description="按顺序执行的切分动作列表。")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitChunkBatchInput":
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("items must be a non-empty list")
        return cls(items=[SplitChunkAction.from_dict(item) for item in raw_items])


class SplitChunkToolConfig(BaseModel):
    line_wrap_width: int = Field(default=30, ge=1, description="Window 视图自动换行宽度。")
    emit_tool_message_content: bool = Field(default=True, description="是否把工具结果写入 ToolMessage content。")
    default_max_retries: int = Field(default=3, ge=1, description="默认允许的最大重试次数。")
    max_messages: int = Field(default=16, ge=1, description="state 里最多保留的工具消息条数。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "SplitChunkToolConfig":
        payload = _load_json(path)
        return cls(
            line_wrap_width=max(1, int(payload.get("line_wrap_width", 30))),
            emit_tool_message_content=bool(payload.get("emit_tool_message_content", True)),
            default_max_retries=max(1, int(payload.get("default_max_retries", 3))),
            max_messages=max(1, int(payload.get("max_messages", 16))),
        )

    @classmethod
    def load_config_split_chunk_tool(cls, source=None) -> "SplitChunkToolConfig":
        if source is None:
            return cls.load()
        return config_from_external(cls, source)


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split()).strip()


def extract_keywords(text: str, *, limit: int = 4) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,12}", text)
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords or ["chunk"]


def build_summary(text: str, *, max_chars: int = 24) -> str:
    cleaned = _collapse_whitespace(text)
    if not cleaned:
        return "空白片段"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1] + "…"


def wrap_lines(text: str, *, width: int, base_offset: int = 0) -> list[DisplayLine]:
    if not text:
        return []
    lines: list[DisplayLine] = []
    position = 0
    while position < len(text):
        newline_index = text.find("\n", position)
        if newline_index == -1:
            logical_end = len(text)
            has_newline = False
        else:
            logical_end = newline_index
            has_newline = True
        absolute_logical_end = base_offset + logical_end + (1 if has_newline else 0)
        logical_text = text[position:logical_end]
        if not logical_text:
            lines.append(
                DisplayLine(
                    text="",
                    start=base_offset + position,
                    end=absolute_logical_end,
                    select_end=absolute_logical_end,
                )
            )
        else:
            segment_start = 0
            while segment_start < len(logical_text):
                segment_end = min(len(logical_text), segment_start + width)
                absolute_start = base_offset + position + segment_start
                absolute_end = base_offset + position + segment_end
                if segment_end >= len(logical_text):
                    absolute_end = base_offset + logical_end + (1 if has_newline else 0)
                lines.append(
                    DisplayLine(
                        text=logical_text[segment_start:segment_end],
                        start=absolute_start,
                        end=absolute_end,
                        select_end=absolute_logical_end,
                    )
                )
                segment_start = segment_end
        position = logical_end + (1 if has_newline else 0)
    return lines


def _tail_by_utf8_bytes(text: str, byte_limit: int | None) -> tuple[str, int]:
    if byte_limit is None or byte_limit <= 0:
        return text, 0
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text, 0
    decoded = encoded[-byte_limit:].decode("utf-8", errors="ignore")
    return decoded, len(text) - len(decoded)


def _head_by_utf8_bytes(text: str, byte_limit: int | None) -> str:
    if byte_limit is None or byte_limit <= 0:
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def build_window_view(
    *,
    document_body: str,
    cursor: int,
    history_line_count: int,
    active_line_count: int,
    preview_line_count: int,
    line_wrap_width: int,
    window_back_bytes: int | None = None,
    window_forward_bytes: int | None = None,
) -> ChunkWindowView:
    safe_cursor = max(0, min(cursor, len(document_body)))
    history_text, history_offset = _tail_by_utf8_bytes(document_body[:safe_cursor], window_back_bytes)
    forward_text = _head_by_utf8_bytes(document_body[safe_cursor:], window_forward_bytes)
    history_lines = wrap_lines(history_text, width=line_wrap_width, base_offset=history_offset)
    forward_lines = wrap_lines(forward_text, width=line_wrap_width, base_offset=safe_cursor)
    active_source = forward_lines[: max(0, active_line_count)]
    preview_lines = forward_lines[max(0, active_line_count) : max(0, active_line_count) + max(0, preview_line_count)]
    return ChunkWindowView(
        history_lines=history_lines[-max(0, history_line_count) :],
        active_lines=[
            NumberedWindowLine(
                line_number=index,
                text=line.text,
                start=line.start,
                end=line.end,
                select_end=line.select_end,
            )
            for index, line in enumerate(active_source)
        ],
        preview_lines=preview_lines,
    )


def render_window_view(view: ChunkWindowView) -> str:
    parts = ['<CapabilityState name="chunking">']
    if view.history_lines:
        parts.append("<Passed>")
        parts.extend(line.text for line in view.history_lines)
        parts.append("</Passed>")
    parts.append("<Window>")
    if view.active_lines:
        parts.extend(f"{line.line_number} | {line.text}" for line in view.active_lines)
    else:
        parts.append("(no remaining lines)")
    parts.append("</Window>")
    if view.preview_lines:
        parts.append("<Preview>")
        parts.extend(line.text for line in view.preview_lines)
        parts.append("</Preview>")
    parts.append("</CapabilityState>")
    return "\n".join(parts)


def render_chunked_document(document_body: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return document_body
    parts: list[str] = []
    cursor = 0
    for chunk in sorted(chunks, key=lambda item: int(item["char_start"])):
        start = int(chunk["char_start"])
        end = int(chunk["char_end"])
        if start > cursor:
            parts.append(document_body[cursor:start])
        segment = document_body[start:end]
        parts.append(
            f'<chunk id="{escape(str(chunk["id"]), quote=True)}" '
            f'summary="{escape(str(chunk["summary"]), quote=True)}">'
            f"{segment}</chunk>"
        )
        cursor = end
    if cursor < len(document_body):
        parts.append(document_body[cursor:])
    return "".join(parts)


class SplitChunkToolFeedback(BaseModel):
    successText: str = Field(default="已创建 chunk。")
    failureText: str = Field(default="split_chunk 失败：{error}")


class SplitChunkToolSchema:
    name = "split_chunk"
    args_schema = SplitChunkBatchInput
    description = "按当前带行号 Window 的行号切出一个或多个 chunk。"
    toolfeedback = SplitChunkToolFeedback


class SplitChunkTool:
    name = SplitChunkToolSchema.name
    config = SplitChunkToolConfig
    substate = SplitChunkStateTydict
    toolschema = SplitChunkToolSchema

    def __init__(self, config: SplitChunkToolConfig | None = None) -> None:
        self.config = config or SplitChunkToolConfig.load()
        self.tool = self

    def _emit(self, writer: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
        if writer is not None:
            writer(payload)

    def _error_payload(
        self,
        *,
        state: SplitChunkStateTydict,
        message: str,
        retry_count: int,
    ) -> tuple[SplitChunkStateTydict, dict[str, Any]]:
        payload = {
            "operation": "split",
            "status": "error",
            "message": message,
            "document_name": state.get("document_name"),
            "cursor": state.get("cursor", 0),
            "retry_count": retry_count,
            "chunk_count": len(state.get("chunks", [])),
        }
        next_state: SplitChunkStateTydict = dict(state)
        next_state["retry_count"] = retry_count
        self._append_message(next_state, payload=payload, status="error")
        return next_state, payload

    def run(
        self,
        *,
        batch_input: SplitChunkBatchInput,
        state: SplitChunkStateTydict,
        history_line_count: int,
        active_line_count: int,
        preview_line_count: int,
        stream_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[SplitChunkStateTydict, dict[str, Any]]:
        next_state: SplitChunkStateTydict = dict(state)
        next_state["messages"] = list(state.get("messages", []))
        next_state["chunks"] = list(state.get("chunks", []))
        body = str(next_state.get("document_body", ""))
        max_retries = int(next_state.get("max_retries", self.config.default_max_retries))
        retry_count = int(next_state.get("retry_count", 0))

        self._emit(
            stream_writer,
            {
                "type": "tool",
                "tool": self.name,
                "event": "start",
                "cursor": next_state.get("cursor", 0),
                "item_count": len(batch_input.items),
            },
        )

        for action in batch_input.items:
            if retry_count >= min(max_retries, action.max_retries or max_retries):
                next_state, payload = self._error_payload(
                    state=next_state,
                    message=f"Retry limit reached ({retry_count}/{min(max_retries, action.max_retries or max_retries)}).",
                    retry_count=retry_count,
                )
                self._emit(stream_writer, {"type": "tool", "tool": self.name, "event": "error", "error": payload["message"]})
                return next_state, payload

            view = build_window_view(
                document_body=body,
                cursor=int(next_state.get("cursor", 0)),
                history_line_count=history_line_count,
                active_line_count=active_line_count,
                preview_line_count=preview_line_count,
                line_wrap_width=self.config.line_wrap_width,
                window_back_bytes=next_state.get("window_back_bytes"),
                window_forward_bytes=next_state.get("window_forward_bytes"),
            )
            if not view.active_lines:
                payload = {
                    "operation": "split",
                    "status": "success",
                    "message": "No remaining lines to split.",
                    "document_name": next_state.get("document_name"),
                    "cursor": next_state.get("cursor", 0),
                    "retry_count": retry_count,
                    "chunk_count": len(next_state.get("chunks", [])),
                }
                self._emit(stream_writer, {"type": "tool", "tool": self.name, "event": "success", "preview": payload["message"]})
                return next_state, payload

            if action.line_end >= len(view.active_lines):
                retry_count += 1
                next_state, payload = self._error_payload(
                    state=next_state,
                    message=(
                        f"line_end={action.line_end} 超出当前 Window 的可用范围。"
                        f"当前最大可选行号是 {len(view.active_lines) - 1}。"
                    ),
                    retry_count=retry_count,
                )
                self._emit(stream_writer, {"type": "tool", "tool": self.name, "event": "error", "error": payload["message"]})
                return next_state, payload

            selected_line = view.active_lines[action.line_end]
            char_start = int(next_state.get("cursor", 0))
            char_end = selected_line.select_end
            if char_end <= char_start:
                retry_count += 1
                next_state, payload = self._error_payload(
                    state=next_state,
                    message="line_end 命中了空范围，未能生成可包裹片段。",
                    retry_count=retry_count,
                )
                self._emit(stream_writer, {"type": "tool", "tool": self.name, "event": "error", "error": payload["message"]})
                return next_state, payload

            chunk_text = body[char_start:char_end]
            chunk_id = token_hex(4)
            chunk_record = {
                "id": chunk_id,
                "summary": action.summary,
                "keywords": action.keywords,
                "line_end": action.line_end,
                "char_start": char_start,
                "char_end": char_end,
                "text": chunk_text,
            }
            next_state["chunks"].append(chunk_record)
            next_state["cursor"] = char_end
            retry_count = 0
            next_state["retry_count"] = retry_count
            payload = {
                "operation": "split",
                "status": "success",
                "message": f"Created 1 chunk(s): {chunk_id}.",
                "document_name": next_state.get("document_name"),
                "cursor": char_end,
                "retry_count": retry_count,
                "chunk_count": len(next_state["chunks"]),
                "line_end": action.line_end,
                "chunk_id": chunk_id,
            }
            self._append_message(next_state, payload=payload, status="success")
            self._emit(
                stream_writer,
                {
                    "type": "tool",
                    "tool": self.name,
                    "event": "success",
                    "preview": payload["message"],
                    "line_end": action.line_end,
                    "chunk_id": chunk_id,
                },
            )
        return next_state, payload

    def _append_message(
        self,
        state: SplitChunkStateTydict,
        *,
        payload: dict[str, Any],
        status: str,
    ) -> None:
        messages = list(state.get("messages", []))
        messages.append(
            {
                "kind": "tool",
                "name": self.name,
                "status": status,
                "content": payload if self.config.emit_tool_message_content else "",
            }
        )
        state["messages"] = messages[-self.config.max_messages :]


Config = SplitChunkToolConfig
SubState = SplitChunkStateTydict
Input = SplitChunkBatchInput
ToolFeedback = SplitChunkToolFeedback
ToolSchema = SplitChunkToolSchema
Tool = SplitChunkTool
