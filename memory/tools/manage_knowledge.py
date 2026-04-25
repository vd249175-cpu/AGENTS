"""Delegate a knowledge-base management subtask to the internal manager agent."""

from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
from typing import Any

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from server.component_config import config_from_external, emit
from server.config_overrides import merge_model
from server.knowledge_manager_runtime import KnowledgeManagerAgentOverrides, create_knowledge_manager_agent
from server.neo4j.database_config import DEFAULT_DATABASE_CONFIG_PATH, Neo4jConnectionConfig


TOOL_CONFIG_PATH = Path(__file__).with_name("manage_knowledge.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


class ManageKnowledgeToolConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_default=True)

    neo4j: Neo4jConnectionConfig | None = Field(default=None, description="显式的 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="兼容旧链路的 Neo4j 配置文件路径。")
    run_id: str | None = Field(default=None, description="运行隔离 id。")
    temperature: float = Field(default=0.0, description="内部 manager agent 的模型温度。")
    debug: bool = Field(default=False, description="是否开启内部 manager agent 调试。")
    stream_inner_agent: bool = Field(default=False, description="是否在工具调用时流式跑内部 manager agent。")
    inner_recursion_limit: int = Field(default=64, ge=1, description="内部 manager agent 的递归上限。")
    agent_overrides: KnowledgeManagerAgentOverrides = Field(
        default_factory=KnowledgeManagerAgentOverrides,
        description="覆盖内部 manager agent 的中间键、工具和模型构造参数。",
    )

    @classmethod
    def default(cls) -> "ManageKnowledgeToolConfig":
        return cls(
            neo4j=None,
            neo4j_config_path=DEFAULT_DATABASE_CONFIG_PATH,
            run_id=None,
            temperature=0.0,
            debug=False,
            stream_inner_agent=False,
            inner_recursion_limit=64,
            agent_overrides=KnowledgeManagerAgentOverrides(),
        )

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "ManageKnowledgeToolConfig":
        payload = _load_json(path)
        return cls(
            neo4j=Neo4jConnectionConfig.model_validate(payload["neo4j"]) if isinstance(payload.get("neo4j"), dict) else None,
            neo4j_config_path=(PROJECT_ROOT / str(payload.get("neo4j_config_path", "workspace/config/database_config.json"))).resolve(),
            run_id=payload.get("run_id"),
            temperature=float(payload.get("temperature", 0.0)),
            debug=bool(payload.get("debug", False)),
            stream_inner_agent=bool(payload.get("stream_inner_agent", False)),
            inner_recursion_limit=max(1, int(payload.get("inner_recursion_limit", 64))),
            agent_overrides=KnowledgeManagerAgentOverrides.model_validate(payload.get("agent_overrides") or {}),
        )

    @classmethod
    def load_config_manage_knowledge_tool(cls, source=None) -> "ManageKnowledgeToolConfig":
        """Load the demo-style top-level manage_knowledge context."""

        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class ManageKnowledgeToolOverride(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    neo4j: Neo4jConnectionConfig | dict[str, Any] | None = Field(default=None, description="覆盖显式 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="覆盖 Neo4j 连接配置文件路径。")
    run_id: str | None = Field(default=None, description="覆盖运行隔离 id。")
    temperature: float | None = Field(default=None, description="覆盖内部 manager agent 的模型温度。")
    debug: bool | None = Field(default=None, description="覆盖内部 manager agent 调试开关。")
    stream_inner_agent: bool | None = Field(default=None, description="覆盖是否流式跑内部 manager agent。")
    inner_recursion_limit: int | None = Field(default=None, ge=1, description="覆盖内部 manager agent 的递归上限。")
    agent_overrides: KnowledgeManagerAgentOverrides = Field(
        default_factory=KnowledgeManagerAgentOverrides,
        description="覆盖内部 manager agent 的中间键、工具和模型构造参数。",
    )


class ManageKnowledgeInput(BaseModel):
    target: str = Field(description="交给知识管理者处理的目标。应写清楚要查询、整理、关联、修正或复核什么。")

    @model_validator(mode="after")
    def validate_input(self) -> "ManageKnowledgeInput":
        self.target = self.target.strip()
        if not self.target:
            raise ValueError("target is required")
        return self


class ManageKnowledgeToolStateTydict(AgentState, total=False):
    pass


class ManageKnowledgeToolFeedback(BaseModel):
    successText: str = Field(default="知识管理已完成。")
    failureText: str = Field(default="知识管理失败：{error}")


class ManageKnowledgeToolSchema:
    name = "manage_knowledge"
    args_schema = ManageKnowledgeInput
    description = (
        "把一个完整的知识库管理目标交给内部知识管理者处理。"
        "只填写 target，说明要查询、整理、关联、修正或复核什么。"
        "返回管理摘要，以及对主 agent 后续判断有用的节点信息。"
    )
    toolfeedback = ManageKnowledgeToolFeedback


class ManageKnowledgeTool:
    name = ManageKnowledgeToolSchema.name
    config = ManageKnowledgeToolConfig
    substate = ManageKnowledgeToolStateTydict
    toolschema = ManageKnowledgeToolSchema

    def __init__(
        self,
        *,
        config: ManageKnowledgeToolConfig | None = None,
        overrides: ManageKnowledgeToolOverride | dict[str, Any] | None = None,
    ) -> None:
        self.config = config or ManageKnowledgeToolConfig.default()
        self.overrides = overrides
        self.tool = self.create_tool()

    def create_tool(self):
        """根据当前实例配置构造标准 LangChain tool。"""

        return _build_manage_knowledge_tool(
            config=self.config,
            overrides=self.overrides,
        )

    def close(self) -> None:
        close_method = getattr(self.tool, "close", None)
        if callable(close_method):
            close_method()


def _build_manage_knowledge_tool(
    *,
    config: ManageKnowledgeToolConfig | None = None,
    overrides: ManageKnowledgeToolOverride | dict[str, Any] | None = None,
):
    active_config = config or ManageKnowledgeToolConfig.default()
    if isinstance(overrides, dict):
        overrides = ManageKnowledgeToolOverride.model_validate(overrides)
    if overrides is not None:
        active_config = merge_model(active_config, overrides)
    manager_agent_cache: dict[str, Any] = {}

    def _get_manager_agent() -> Any:
        cached = manager_agent_cache.get("agent")
        if cached is not None:
            return cached
        agent = create_knowledge_manager_agent(
            run_id=active_config.run_id or "knowledge-manager",
            neo4j=active_config.neo4j,
            neo4j_config_path=active_config.neo4j_config_path,
            temperature=active_config.temperature,
            debug=active_config.debug,
            overrides=active_config.agent_overrides,
        )
        manager_agent_cache["agent"] = agent
        return agent

    @tool(
        "manage_knowledge",
        args_schema=ManageKnowledgeInput,
        description=(
            "把一个完整的知识库管理目标交给内部知识管理者处理。"
            "只填写 target，说明要查询、整理、关联、修正或复核什么。"
            "返回管理摘要，以及对主 agent 后续判断有用的节点信息。"
        ),
    )
    def manage_knowledge(target: str, runtime: ToolRuntime | None = None) -> dict[str, Any]:
        """Delegate a knowledge-base management subtask to the internal manager agent."""
        manager_agent = _get_manager_agent()
        inherited_messages = _inherit_parent_messages((runtime.state if runtime is not None else {}).get("messages"))
        handoff_prompt = _build_handoff_prompt(target=target)
        child_messages = [*inherited_messages, HumanMessage(content=handoff_prompt)]
        thread_id = f"knowledge-manager-{token_hex(8)}"
        config_payload = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": active_config.inner_recursion_limit,
        }
        final_state: dict[str, Any]
        if active_config.stream_inner_agent:
            for event in manager_agent.stream({"messages": child_messages}, config=config_payload, stream_mode="updates"):
                _forward_inner_stream_event(runtime, event)
            final_state = dict(manager_agent.get_state(config_payload).values)
        else:
            final_state = dict(manager_agent.invoke({"messages": child_messages}, config=config_payload))
        return _build_handoff_payload(target=target, final_state=final_state)

    def close() -> None:
        cached = manager_agent_cache.get("agent")
        close_method = getattr(cached, "close", None)
        if callable(close_method):
            close_method()

    try:
        object.__setattr__(manage_knowledge, "close", close)
    except Exception:
        pass
    return manage_knowledge


def build_manage_knowledge_tool(
    *,
    config: ManageKnowledgeToolConfig | None = None,
    overrides: ManageKnowledgeToolOverride | dict[str, Any] | None = None,
):
    return ManageKnowledgeTool(config=config, overrides=overrides).tool


def _inherit_parent_messages(messages: object) -> list[BaseMessage]:
    inherited: list[BaseMessage] = []
    if not isinstance(messages, list):
        return inherited
    for message in messages:
        if isinstance(message, SystemMessage | ToolMessage):
            continue
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            continue
        if isinstance(message, HumanMessage | AIMessage):
            inherited.append(message)
    return inherited


def _forward_inner_stream_event(runtime: ToolRuntime | None, event: object) -> None:
    payload = {
        "type": "inner_agent_update",
        "tool": "manage_knowledge",
        "event": event,
    }
    runtime_writer = getattr(runtime, "stream_writer", None) if runtime is not None else None
    emitted = False
    if callable(runtime_writer):
        runtime_writer(payload)
        emitted = True
    try:
        graph_writer = get_stream_writer()
    except Exception:
        graph_writer = None
    if callable(graph_writer):
        if graph_writer is not runtime_writer or not emitted:
            graph_writer(payload)


def _build_handoff_prompt(*, target: str) -> str:
    return (
        "这是主 agent 委派给知识管理者的子任务。\n"
        f"目标：{target}\n"
        "要求：\n"
        "1. 必须通过工具完成管理动作，不要空谈。\n"
        "2. 如果发现对主 agent 有帮助的节点或 chunk，请把它们放进 useful 桶。\n"
        "3. useful 桶应尽量保留真正关键的节点，方便主 agent 继续处理。\n"
        "4. 结束前请总结完成了什么、有哪些关键发现、还建议主 agent 做什么。"
    )


def _build_handoff_payload(*, target: str, final_state: dict[str, Any]) -> dict[str, Any]:
    tool_payloads = _collect_tool_payloads(final_state.get("messages"))
    useful_items = _normalize_state_items(final_state.get("useful_items"))
    if not useful_items:
        useful_items = _extract_useful_items_from_tool_payloads(tool_payloads)
    operation_summary = _summarize_operations(tool_payloads)
    manager_summary = _find_last_ai_message(final_state.get("messages"))
    final_message = _build_parent_message(
        target=target,
        tool_payloads=tool_payloads,
        operation_summary=operation_summary,
        manager_summary=manager_summary,
    )
    return {
        "operation": operation_summary,
        "message": final_message,
        "useful_items": useful_items,
    }


def _normalize_state_items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    normalized: list[dict[str, Any]] = []
    for node_id, payload in value.items():
        if not isinstance(payload, dict):
            continue
        normalized_item = _sanitize_useful_item(node_id=node_id, payload=payload)
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized


def _collect_tool_payloads(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        content = message.content.strip()
        if not content or content[0] not in "{[":
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _find_last_ai_message(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return " ".join(str(item) for item in content).strip()
    return ""


def _build_parent_message(
    *,
    target: str,
    tool_payloads: list[dict[str, Any]],
    operation_summary: dict[str, Any],
    manager_summary: str,
) -> str:
    manager_summary = _compact_text(manager_summary, 160)
    if manager_summary:
        return f"知识管理已完成。{manager_summary}"

    operations: list[str] = []
    for payload in tool_payloads:
        operation = str(payload.get("operation") or "").strip()
        if not operation or operation == "graph_mark_useful":
            continue
        if operation not in operations:
            operations.append(operation)
    operation_text = str(operation_summary.get("summary") or "").strip()
    if not operation_text:
        operation_text = "、".join(operations[:3]) if operations else _compact_text(target, 40) or "管理子任务"
    return f"知识管理已完成。{operation_text}"


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(str(value).split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _summarize_operations(tool_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "create_count": 0,
        "update_count": 0,
        "delete_count": 0,
        "read_count": 0,
        "mark_useful_count": 0,
    }
    for payload in tool_payloads:
        operation = str(payload.get("operation") or "").strip()
        if not operation:
            continue
        if operation == "create_chunk_document":
            if _payload_success(payload):
                counts["create_count"] += 1
            continue
        if operation == "manage_chunks":
            _accumulate_result_operations(payload.get("results"), counts, {"insert": "create_count", "update": "update_count", "delete": "delete_count"})
            continue
        if operation == "insert_chunks":
            _accumulate_result_operations(payload.get("results"), counts, {"insert": "create_count"})
            continue
        if operation == "update_chunks":
            _accumulate_result_operations(payload.get("results"), counts, {"update": "update_count"})
            continue
        if operation == "delete_chunks":
            _accumulate_result_operations(payload.get("results"), counts, {"delete": "delete_count"})
            continue
        if operation == "graph_manage_nodes":
            _accumulate_result_operations(payload.get("results"), counts, {"create": "create_count", "update": "update_count", "delete": "delete_count"})
            continue
        if operation == "graph_create_nodes":
            _accumulate_result_operations(payload.get("results"), counts, {"create": "create_count"})
            continue
        if operation == "graph_update_node":
            _accumulate_result_operations(payload.get("results"), counts, {"update": "update_count"})
            continue
        if operation == "graph_delete_nodes":
            _accumulate_result_operations(payload.get("results"), counts, {"delete": "delete_count"})
            continue
        if operation == "graph_mark_useful":
            counts["mark_useful_count"] += _count_useful_marks(payload)
            continue
        if operation in {"read_nodes", "list_chunk_documents", "query_chunk_positions", "recall_nodes_by_keywords", "keyword_recall", "graph_distance_recall"}:
            counts["read_count"] += _count_successful_reads(payload)
            continue

    summary_parts = [
        f"新增 {counts['create_count']}",
        f"修改 {counts['update_count']}",
        f"删除 {counts['delete_count']}",
    ]
    if counts["read_count"] > 0:
        summary_parts.append(f"读取 {counts['read_count']}")
    if counts["mark_useful_count"] > 0:
        summary_parts.append(f"useful 标记 {counts['mark_useful_count']}")
    counts["summary"] = "，".join(summary_parts)
    return counts


def _payload_success(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "").strip().lower() == "success"


def _accumulate_result_operations(
    results: object,
    counts: dict[str, int],
    mapping: dict[str, str],
) -> None:
    if not isinstance(results, list):
        return
    for result in results:
        if not isinstance(result, dict):
            continue
        if str(result.get("status") or "").strip().lower() != "success":
            continue
        operation = str(result.get("operation") or result.get("op") or "").strip().lower()
        target_key = mapping.get(operation)
        if target_key is not None:
            counts[target_key] += 1


def _count_successful_reads(payload: dict[str, Any]) -> int:
    results = payload.get("results")
    if isinstance(results, list):
        success_count = sum(
            1
            for result in results
            if isinstance(result, dict) and str(result.get("status") or "").strip().lower() == "success"
        )
        if success_count > 0:
            return success_count
    if _payload_success(payload):
        count = payload.get("success_count") or payload.get("document_count") or payload.get("chunk_count") or payload.get("result_count")
        if isinstance(count, int) and count > 0:
            return count
        return 1
    return 0


def _count_useful_marks(payload: dict[str, Any]) -> int:
    items = payload.get("items")
    if isinstance(items, list):
        item_count = sum(1 for item in items if isinstance(item, dict))
        if item_count > 0:
            return item_count
    return 1 if _payload_success(payload) else 0


def _extract_useful_items_from_tool_payloads(tool_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for payload in tool_payloads:
        if payload.get("operation") != "graph_mark_useful":
            continue
        items = payload.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                normalized_item = _sanitize_useful_item(node_id=item.get("node_id"), payload=item)
                if normalized_item is not None:
                    extracted.append(normalized_item)
    return extracted


def _sanitize_useful_item(*, node_id: object, payload: dict[str, Any]) -> dict[str, Any] | None:
    cleaned_node_id = str(payload.get("node_id") or node_id or "").strip()
    if not cleaned_node_id:
        return None
    normalized: dict[str, Any] = {
        "node_id": cleaned_node_id,
        "body": str(payload.get("body") or ""),
        "edges": _sanitize_edges(payload.get("edges")),
    }
    if str(payload.get("node_label") or "").strip() == "Chunk":
        document_name = payload.get("document_name")
        chunk_index = payload.get("chunk_index")
        if document_name is not None:
            normalized["document_name"] = str(document_name)
        if chunk_index is not None:
            normalized["chunk_index"] = int(chunk_index)
    return normalized


def _sanitize_edges(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        edge_payload: dict[str, Any] = {}
        for key in ("neighbor_node_id", "dist", "distance", "edge_kind", "edge_type", "edge_key", "u", "v"):
            if key in item and item.get(key) is not None:
                edge_payload[key] = item.get(key)
        if edge_payload:
            normalized.append(edge_payload)
    return normalized


tool_runingconfig = ManageKnowledgeToolConfig.default()
tools = {"manage_knowledge": build_manage_knowledge_tool(config=tool_runingconfig)}
toolStateTydicts = {
    "manage_knowledge": ManageKnowledgeToolStateTydict,
}
ToolConfig = {
    "inputSm": ManageKnowledgeInput,
    "runingConfig": tool_runingconfig,
    "tools": tools,
    "toolStateTydicts": toolStateTydicts,
}

Config = ManageKnowledgeToolConfig
SubState = ManageKnowledgeToolStateTydict
Input = ManageKnowledgeInput
ToolFeedback = ManageKnowledgeToolFeedback
ToolSchema = ManageKnowledgeToolSchema
Tool = ManageKnowledgeTool
