"""Chunking middleware and state rendering for the migrated chunk module."""

import json
from pathlib import Path
from typing import Any, Callable

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from pydantic import BaseModel, Field
from tools.split_chunk import (
    SplitChunkTool,
    SplitChunkStateTydict,
    build_window_view,
    render_window_view,
)


MIDDLEWARE_CONFIG_PATH = Path(__file__).with_name("chunking.json")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class MiddlewareCapabilityPrompt(BaseModel):
    name: str = Field(description="system prompt slot 名称。")
    prompt: str = Field(description="注入给模型的提示词内容。")


class MiddlewareRuningConfig(BaseModel):
    history_line_count: int = Field(default=4, ge=0, description="窗口前置历史区域行数。")
    active_line_count: int = Field(default=8, ge=1, description="当前可切分区域行数。")
    preview_line_count: int = Field(default=4, ge=0, description="窗口预览区域行数。")
    line_wrap_width: int = Field(default=30, ge=1, description="Window 视图自动换行宽度。")
    window_back_bytes: int | None = Field(default=None, ge=1, description="Passed 区域按 UTF-8 字节裁剪的上限。")
    window_forward_bytes: int | None = Field(default=None, ge=1, description="Window + Preview 区域按 UTF-8 字节裁剪的上限。")
    trace_limit: int = Field(default=16, ge=1, description="过程 trace 的最大保留条数。")
    max_retries: int = Field(default=3, ge=1, description="默认允许的最大重试次数。")

    @classmethod
    def load(cls, path: Path = MIDDLEWARE_CONFIG_PATH) -> "MiddlewareRuningConfig":
        payload = _load_json(path)
        return cls(
            history_line_count=max(0, int(payload.get("history_line_count", 4))),
            active_line_count=max(1, int(payload.get("active_line_count", 8))),
            preview_line_count=max(0, int(payload.get("preview_line_count", 4))),
            line_wrap_width=max(1, int(payload.get("line_wrap_width", 30))),
            window_back_bytes=int(payload["window_back_bytes"]) if payload.get("window_back_bytes") is not None else None,
            window_forward_bytes=int(payload["window_forward_bytes"]) if payload.get("window_forward_bytes") is not None else None,
            trace_limit=max(1, int(payload.get("trace_limit", 16))),
            max_retries=max(1, int(payload.get("max_retries", 3))),
        )

    @classmethod
    def load_config_chunking_middleware(cls, source=None) -> "MiddlewareRuningConfig":
        if source is None:
            return cls.load()
        from server.component_config import config_from_external

        return config_from_external(cls, source)


class ChunkingStateTydict(AgentState, SplitChunkStateTydict, total=False):
    pass


class AffectedPrompt(MiddlewareCapabilityPrompt):
    pass


class AffectedPrompts(BaseModel):
    Prompts: list[AffectedPrompt]


middleware_capability_prompts = [
    AffectedPrompt(
        name="chunking.guidance",
        prompt=(
            "从左到右阅读文档，把可见 Window 切成连贯片段。"
            "Window 中只有当前可切分部分带行号，行号从 0 开始。"
            "每一行最多 30 个字，超出后会自动换行。"
            "Passed 和 Preview 不带行号，不允许把它们当成切分目标。"
            "split_chunk 必须使用 line_end 选择当前 Window 内的结束行。"
            "每个 chunk 都要给出简短摘要和精炼关键词。"
        ),
    )
]


default_affected_prompts = AffectedPrompts(
    Prompts=middleware_capability_prompts,
)

MiddlewareToolConfig = {
    "tools": {},
    "toolStateTydicts": {
        "split_chunk": SplitChunkStateTydict,
    },
}


class ChunkingCapabilityMiddleware(AgentMiddleware):
    name = "chunking"
    capabilityPromptConfigs = middleware_capability_prompts
    runingConfig = MiddlewareRuningConfig.load()
    toolConfig = MiddlewareToolConfig
    toolStateTydicts = MiddlewareToolConfig["toolStateTydicts"]
    tools = list(MiddlewareToolConfig["tools"].values())
    state_schema = ChunkingStateTydict  # type: ignore[assignment]

    def __init__(
        self,
        *,
        runing_config: MiddlewareRuningConfig | None = None,
        split_tool: SplitChunkTool | None = None,
    ) -> None:
        super().__init__()
        self.runing_config = runing_config or self.runingConfig
        self.split_tool = split_tool or SplitChunkTool()
        self.capability_prompts = self.capabilityPromptConfigs
        self.config = self.runing_config
        self.middleware = self

    def _emit(self, writer: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
        if writer is not None:
            writer(payload)

    def build_state(
        self,
        *,
        document_body: str,
        document_name: str | None = None,
    ) -> ChunkingStateTydict:
        return {
            "document_body": document_body,
            "document_name": document_name,
            "messages": [],
            "chunks": [],
            "cursor": 0,
            "retry_count": 0,
            "max_retries": self.runing_config.max_retries,
            "history_line_count": self.runing_config.history_line_count,
            "active_line_count": self.runing_config.active_line_count,
            "preview_line_count": self.runing_config.preview_line_count,
            "line_wrap_width": self.runing_config.line_wrap_width,
            "window_back_bytes": self.runing_config.window_back_bytes,
            "window_forward_bytes": self.runing_config.window_forward_bytes,
            "process_trace": [],
        }

    def build_window(self, state: ChunkingStateTydict) -> str:
        view = build_window_view(
            document_body=str(state.get("document_body", "")),
            cursor=int(state.get("cursor", 0)),
            history_line_count=int(state.get("history_line_count", self.runing_config.history_line_count)),
            active_line_count=int(state.get("active_line_count", self.runing_config.active_line_count)),
            preview_line_count=int(state.get("preview_line_count", self.runing_config.preview_line_count)),
            line_wrap_width=int(state.get("line_wrap_width", self.runing_config.line_wrap_width)),
            window_back_bytes=state.get("window_back_bytes", self.runing_config.window_back_bytes),
            window_forward_bytes=state.get("window_forward_bytes", self.runing_config.window_forward_bytes),
        )
        return render_window_view(view)

    def append_trace(
        self,
        state: ChunkingStateTydict,
        *,
        step: str,
        detail: str,
        tool_name: str | None = None,
        tool_status: str | None = None,
    ) -> ChunkingStateTydict:
        trace = list(state.get("process_trace", []))
        entry = {
            "step": step,
            "detail": detail,
            "cursor": int(state.get("cursor", 0)),
            "chunk_count": len(state.get("chunks", [])),
        }
        if tool_name is not None:
            entry["tool_name"] = tool_name
        if tool_status is not None:
            entry["tool_status"] = tool_status
        trace.append(entry)
        state["process_trace"] = trace[-self.runing_config.trace_limit :]
        return state

    def before_step(self, state: ChunkingStateTydict, *, writer: Callable[[dict[str, Any]], None] | None = None) -> ChunkingStateTydict:
        self.append_trace(state, step="before_step", detail="Prepared numbered chunk window.")
        self._emit(
            writer,
            {
                "type": "middleware",
                "stage": "before_step",
                "cursor": int(state.get("cursor", 0)),
                "chunk_count": len(state.get("chunks", [])),
                "window": self.build_window(state),
            },
        )
        return state

    def after_tool(
        self,
        state: ChunkingStateTydict,
        *,
        tool_status: str,
        writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> ChunkingStateTydict:
        self.append_trace(
            state,
            step="after_tool",
            detail="Applied split_chunk result.",
            tool_name=self.split_tool.name,
            tool_status=tool_status,
        )
        self._emit(
            writer,
            {
                "type": "middleware",
                "stage": "after_tool",
                "cursor": int(state.get("cursor", 0)),
                "chunk_count": len(state.get("chunks", [])),
                "tool_status": tool_status,
            },
        )
        return state


Config = MiddlewareRuningConfig
SubState = ChunkingStateTydict
class MiddlewareSchema:
    name = "chunking"
    affectedPrompts = default_affected_prompts
    tools = {"split_chunk": SplitChunkTool}


Middleware = ChunkingCapabilityMiddleware
