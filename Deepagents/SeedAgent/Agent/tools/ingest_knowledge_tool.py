"""Ingest files from workspace/knowledge into the memory graph."""

import json
from pathlib import Path, PurePosixPath
from typing import Any

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..server.demo_server import StrictConfig, emit
from ..server.memory_bridge import build_chunk_apply_tool
from ..server.path_resolver import WORKSPACE_ROOT


class Config(StrictConfig):
    resume: bool = Field(default=True, description="Whether chunk_apply may resume an interrupted import.")
    shardCount: int = Field(default=4, ge=1, description="Default shard count for long documents.")
    maxWorkers: int = Field(default=2, ge=1, description="Default worker count for long documents.")
    referenceBytes: int = Field(default=6000, ge=1, description="Reference byte window for long documents.")
    agentName: str = Field(default="SeedAgent", description="Agent name used for memory run identity.")


def _context_value(context: Any, fallback: Config, key: str) -> Any:
    return getattr(context, key, getattr(fallback, key))


class SubState(AgentState, total=False):
    ingestKnowledgeStats: dict[str, int]
    ingestKnowledgeLastPath: str | None
    ingestKnowledgeLastResult: str | None
    ingestKnowledgeLastError: str | None


class Input(BaseModel):
    path: str = Field(
        description=(
            "Path to a file under /workspace/knowledge, for example "
            "/workspace/knowledge/research.md. Relative paths are resolved inside /workspace/knowledge."
        )
    )
    chunkingRequirement: str | None = Field(
        default=None,
        description="Optional instruction for semantic chunking.",
    )
    resume: bool | None = Field(default=None, description="Override default resume behavior.")
    shardCount: int | None = Field(default=None, ge=1, description="Override shard count.")
    maxWorkers: int | None = Field(default=None, ge=1, description="Override max workers.")
    referenceBytes: int | None = Field(default=None, ge=1, description="Override reference bytes.")


class ToolFeedback(BaseModel):
    successText: str = Field(default="Knowledge document ingested into memory.")
    failureText: str = Field(default="Knowledge document ingest failed: {error}")


class ToolSchema:
    name = "ingest_knowledge_document"
    args_schema = Input
    description = (
        "将 /workspace/knowledge 中的长文档读取、语义切分并写入记忆图。"
        "只接受 /workspace/knowledge 下的文件；完成后可继续用 manage_knowledge 管理记忆。"
    )
    toolfeedback = ToolFeedback


def _resolve_knowledge_path(path: str) -> Path:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("path must not be empty.")
    if raw.startswith("/workspace"):
        relative = PurePosixPath(raw).relative_to("/workspace")
    else:
        relative = PurePosixPath("knowledge") / raw
    if any(part == ".." for part in relative.parts):
        raise ValueError("path must stay inside /workspace/knowledge.")
    if not relative.parts or relative.parts[0] != "knowledge":
        raise ValueError("path must point to a file under /workspace/knowledge.")
    host_path = (WORKSPACE_ROOT / Path(*relative.parts)).resolve()
    knowledge_root = (WORKSPACE_ROOT / "knowledge").resolve()
    if host_path != knowledge_root and knowledge_root not in host_path.parents:
        raise ValueError("path must stay inside /workspace/knowledge.")
    if not host_path.exists():
        raise FileNotFoundError(f"knowledge file not found: {path}")
    if not host_path.is_file():
        raise ValueError(f"knowledge path must be a file: {path}")
    return host_path


class IngestKnowledgeTool:
    name = ToolSchema.name
    config = Config
    substate = SubState
    toolschema = ToolSchema

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._chunk_apply = None
        self.tool = self.create_tool()

    def _get_chunk_apply(self):
        if self._chunk_apply is None:
            self._chunk_apply = build_chunk_apply_tool(agent_name=self.config.agentName, config=self.config)
        return self._chunk_apply

    def close(self) -> None:
        close = getattr(self._chunk_apply, "close", None) if self._chunk_apply is not None else None
        if callable(close):
            close()

    def create_tool(self):
        current_config = self.config
        current_toolschema = self.toolschema
        current_feedback_cls = current_toolschema.toolfeedback
        tool_owner = self

        @tool(
            current_toolschema.name,
            args_schema=current_toolschema.args_schema,
            description=current_toolschema.description,
        )
        def ingest_knowledge_document(
            runtime: ToolRuntime[Config, SubState],
            path: str,
            chunkingRequirement: str | None = None,
            resume: bool | None = None,
            shardCount: int | None = None,
            maxWorkers: int | None = None,
            referenceBytes: int | None = None,
        ) -> Command:
            feedback = current_feedback_cls()
            context = runtime.context or current_config
            try:
                host_path = _resolve_knowledge_path(path)
                emit(
                    runtime.stream_writer,
                    {
                        "type": "tool",
                        "tool": current_toolschema.name,
                        "event": "start",
                        "path": str(host_path),
                    },
                )
                result = tool_owner._get_chunk_apply().invoke(
                    {
                        "path": str(host_path),
                        "resume": _context_value(context, current_config, "resume") if resume is None else resume,
                        "chunking_requirement": chunkingRequirement,
                        "shard_count": shardCount or _context_value(context, current_config, "shardCount"),
                        "max_workers": maxWorkers or _context_value(context, current_config, "maxWorkers"),
                        "reference_bytes": referenceBytes or _context_value(context, current_config, "referenceBytes"),
                    },
                    stream_writer=runtime.stream_writer,
                )
                message_text = json.dumps(result, ensure_ascii=False, default=str)
                emit(
                    runtime.stream_writer,
                    {
                        "type": "tool",
                        "tool": current_toolschema.name,
                        "event": "success",
                        "status": result.get("status"),
                        "success_count": int(result.get("success_count", 0) or 0),
                        "failure_count": int(result.get("failure_count", 0) or 0),
                    },
                )
                stats = runtime.state.get("ingestKnowledgeStats") or {}
                return Command(
                    update={
                        "messages": [ToolMessage(content=message_text, tool_call_id=runtime.tool_call_id)],
                        "ingestKnowledgeStats": {
                            **stats,
                            "totalRuns": int(stats.get("totalRuns", 0) or 0) + 1,
                            "lastSuccessCount": int(result.get("success_count", 0) or 0),
                            "lastFailureCount": int(result.get("failure_count", 0) or 0),
                        },
                        "ingestKnowledgeLastPath": str(host_path),
                        "ingestKnowledgeLastResult": message_text,
                        "ingestKnowledgeLastError": None,
                    }
                )
            except Exception as exc:
                error_text = feedback.failureText.format(error=f"{type(exc).__name__}: {exc}")
                emit(
                    runtime.stream_writer,
                    {
                        "type": "tool",
                        "tool": current_toolschema.name,
                        "event": "error",
                        "error": error_text,
                    },
                )
                return Command(
                    update={
                        "messages": [ToolMessage(content=error_text, tool_call_id=runtime.tool_call_id)],
                        "ingestKnowledgeLastPath": path,
                        "ingestKnowledgeLastResult": None,
                        "ingestKnowledgeLastError": error_text,
                    }
                )

        return ingest_knowledge_document


def build_ingest_knowledge_document(config: Config | None = None):
    return IngestKnowledgeTool(config=config).tool
