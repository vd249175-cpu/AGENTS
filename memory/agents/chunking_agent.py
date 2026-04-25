"""Chunking agent for line-number-based chunk processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from middleware import ChunkingCapabilityMiddleware
from tools.split_chunk import (
    SplitChunkAction,
    SplitChunkBatchInput,
    SplitChunkStateTydict,
    build_summary,
    build_window_view,
    extract_keywords,
)


Planner = Callable[[SplitChunkStateTydict, str | None], SplitChunkAction]


class ChunkingAgentSchema:
    name = "chunking_agent"
    systemPrompt = (
        "你是一个负责按行号窗口规划 split_chunk 动作的内部 agent。"
        "只负责将 Window 切分成合适大小的片段，并且并给出摘要和关键词。"
    )
    middlewares = {
        ChunkingCapabilityMiddleware.name: ChunkingCapabilityMiddleware,
    }


@dataclass
class ChunkingAgent:
    middleware: ChunkingCapabilityMiddleware
    planner: Planner | None = None
    name: str = ChunkingAgentSchema.name
    agentschema: type = ChunkingAgentSchema

    def __post_init__(self) -> None:
        self.agent = self

    def _default_planner(self, state: SplitChunkStateTydict, requirement: str | None) -> SplitChunkAction:
        view = build_window_view(
            document_body=str(state.get("document_body", "")),
            cursor=int(state.get("cursor", 0)),
            history_line_count=int(state.get("history_line_count", self.middleware.runing_config.history_line_count)),
            active_line_count=int(state.get("active_line_count", self.middleware.runing_config.active_line_count)),
            preview_line_count=int(state.get("preview_line_count", self.middleware.runing_config.preview_line_count)),
            line_wrap_width=int(state.get("line_wrap_width", self.middleware.runing_config.line_wrap_width)),
            window_back_bytes=state.get("window_back_bytes", self.middleware.runing_config.window_back_bytes),
            window_forward_bytes=state.get("window_forward_bytes", self.middleware.runing_config.window_forward_bytes),
        )
        if not view.active_lines:
            return SplitChunkAction(summary="完成切分", keywords=["完成"], line_end=0)

        preferred_chunk_lines = 4
        if isinstance(requirement, str) and requirement.strip():
            lowered = requirement.strip()
            if "粗" in lowered or "大 chunk" in lowered or "大段" in lowered:
                preferred_chunk_lines = len(view.active_lines)
            elif "细" in lowered or "小 chunk" in lowered:
                preferred_chunk_lines = 2

        chosen_index = min(len(view.active_lines) - 1, max(0, preferred_chunk_lines - 1))
        for line in view.active_lines:
            if not line.text.strip():
                chosen_index = line.line_number
                break

        selected_line = view.active_lines[chosen_index]
        segment = str(state.get("document_body", ""))[int(state.get("cursor", 0)) : selected_line.select_end]
        return SplitChunkAction(
            summary=build_summary(segment),
            keywords=extract_keywords(segment),
            line_end=chosen_index,
        )

    def run(
        self,
        *,
        initial_state: SplitChunkStateTydict,
        requirement: str | None = None,
        stream_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> SplitChunkStateTydict:
        state = dict(initial_state)
        while int(state.get("cursor", 0)) < len(str(state.get("document_body", ""))):
            self.middleware.before_step(state, writer=stream_writer)
            action = (self.planner or self._default_planner)(state, requirement)
            if stream_writer is not None:
                stream_writer(
                    {
                        "step": "model",
                        "messages": [
                            {
                                "kind": "ai",
                                "tool_calls": [
                                    {
                                        "name": self.middleware.split_tool.name,
                                        "args": {
                                            "items": [
                                                {
                                                    "summary": action.summary,
                                                    "keywords": action.keywords,
                                                    "line_end": action.line_end,
                                                }
                                            ]
                                        },
                                    }
                                ],
                            }
                        ],
                        "cursor": int(state.get("cursor", 0)),
                        "chunk_count": len(state.get("chunks", [])),
                        "retry_count": int(state.get("retry_count", 0)),
                    }
                )
            next_state, payload = self.middleware.split_tool.run(
                batch_input=SplitChunkBatchInput(items=[action]),
                state=state,
                history_line_count=int(state.get("history_line_count", self.middleware.runing_config.history_line_count)),
                active_line_count=int(state.get("active_line_count", self.middleware.runing_config.active_line_count)),
                preview_line_count=int(state.get("preview_line_count", self.middleware.runing_config.preview_line_count)),
                stream_writer=stream_writer,
            )
            state = next_state
            self.middleware.after_tool(state, tool_status=str(payload.get("status", "success")), writer=stream_writer)
            if stream_writer is not None:
                stream_writer(
                    {
                        "step": "tools",
                        "messages": [
                            {
                                "kind": "tool",
                                "name": self.middleware.split_tool.name,
                                "status": payload.get("status", "success"),
                                "content": payload,
                            }
                        ],
                        "cursor": int(state.get("cursor", 0)),
                        "chunk_count": len(state.get("chunks", [])),
                        "retry_count": int(state.get("retry_count", 0)),
                    }
                )
            if payload.get("status") == "error":
                raise RuntimeError(str(payload.get("message", "split_chunk failed")))
        return state


def build_chunking_agent(
    *,
    middleware: ChunkingCapabilityMiddleware | None = None,
    planner: Planner | None = None,
) -> ChunkingAgent:
    return ChunkingAgent(
        middleware=middleware or ChunkingCapabilityMiddleware(),
        planner=planner,
    )
