"""Manage thread memory entries in runtime state."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from server.memory_service import MemoryCapabilityPreset, MemoryService
from server.memory_state import MemoryToolStateTydict


TOOL_CONFIG_PATH = Path(__file__).with_name("manage_memory.json")
MemoryOperation = Literal["create", "delete", "update"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class ManageMemoryToolConfig(BaseModel):
    max_items: int = Field(default=32, ge=1, description="记忆条目允许的最大数量。")
    max_total_chars: int = Field(default=4096, ge=1, description="记忆条目允许的最大字符总数。")

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "ManageMemoryToolConfig":
        payload = _load_json(path)
        return cls(max_items=max(1, int(payload.get("max_items", 32))), max_total_chars=max(1, int(payload.get("max_total_chars", 4096))))


class MemoryAction(BaseModel):
    operation: MemoryOperation = Field(description="操作：create、delete 或 update。")
    id: str | None = Field(default=None, description="4 位 id。create 时可省略并自动生成。")
    label: str | None = Field(default=None, description="记忆标签。create 时必填；update 时可选；delete 时省略。")
    content: str | None = Field(default=None, description="记忆内容。create 和 update 时必填；delete 时省略。")

    @model_validator(mode="after")
    def validate_action(self) -> "MemoryAction":
        if self.operation == "create":
            if not self.label or not self.label.strip():
                raise ValueError("label is required for create")
            if self.content is None or not self.content.strip():
                raise ValueError("content is required for create")
        if self.operation == "update":
            if self.id is None or not self.id.strip():
                raise ValueError("id is required for update")
            if self.content is None or not self.content.strip():
                raise ValueError("content is required for update")
        if self.operation == "delete":
            if self.id is None or not self.id.strip():
                raise ValueError("id is required for delete")
            if self.label is not None:
                raise ValueError("label must be omitted for delete")
            if self.content is not None:
                raise ValueError("content must be omitted for delete")

        if self.id is not None:
            resolved_id = self.id.strip()
            if len(resolved_id) != 4 or not resolved_id.isdigit():
                raise ValueError("id must be a 4-digit numeric string")
            self.id = resolved_id
        if self.label is not None:
            self.label = self.label.strip()
        if self.content is not None:
            self.content = self.content.strip()
        return self


class MemoryBatchInput(BaseModel):
    items: list[MemoryAction] = Field(description="按顺序排列的记忆动作列表。")

    @model_validator(mode="after")
    def validate_items(self) -> "MemoryBatchInput":
        if not self.items:
            raise ValueError("items must not be empty")
        return self


MEMORY_TOOL_DESCRIPTION = """用于管理当前对话线程中的持久记忆。

当你需要记录稳定偏好、长期事实、重要修正或线程级持久信息时使用。
这把工具会按顺序批量执行一组记忆动作，并把结果合并成一条结果返回。
动作类型和每个字段的必填规则由参数声明负责。
"""


class MemoryStateTydict(AgentState, MemoryToolStateTydict, total=False):
    pass


def build_manage_memory_tool(
    *,
    config: ManageMemoryToolConfig | None = None,
    service: MemoryService | None = None,
):
    active_config = config or ManageMemoryToolConfig.load()
    active_service = service or MemoryService(
        preset=MemoryCapabilityPreset(
            max_items=active_config.max_items,
            max_total_chars=active_config.max_total_chars,
        )
    )

    @tool("manage_memory", args_schema=MemoryBatchInput, description=MEMORY_TOOL_DESCRIPTION)
    def manage_memory(runtime: ToolRuntime | None = None, items: list[MemoryAction] | None = None) -> Command[None]:
        """Manage thread memory entries in runtime state."""
        active_items = list(items or [])
        active_state: MemoryStateTydict = runtime.state if runtime is not None else {}
        active_tool_call_id = runtime.tool_call_id if runtime is not None else "manual"
        if os.getenv("RESTRUCTURE_TRACE_MEMORY") == "1":
            print(f"[MEMORY_TRACE] tool_run enter tool_call_id={active_tool_call_id} item_count={len(active_items)}", flush=True)
        payload = active_service.apply_operations(
            state=active_state,
            items=active_items,
            tool_call_id=active_tool_call_id,
        )
        if os.getenv("RESTRUCTURE_TRACE_MEMORY") == "1":
            print(f"[MEMORY_TRACE] tool_run exit status={payload.get('status')}", flush=True)
        message = ToolMessage(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            tool_call_id=active_tool_call_id,
            status=str(payload.get("status", "success")),
        )
        update: dict[str, Any] = {"messages": [message]}
        state_update = payload.get("state_update")
        if isinstance(state_update, dict):
            update.update(state_update)
        return Command(update=update)

    return manage_memory


tool_runingconfig = ManageMemoryToolConfig.load()
tools = {}
toolStateTydicts = {
    "manage_memory": MemoryStateTydict,
}
ToolConfig = {
    "inputSm": MemoryBatchInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}
