"""Single-file communication tool following the demo wrapper pattern."""

from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from MainServer.comm import AgentComm
from MainServer.state import MessageType
from ..server.demo_server import (
    StrictConfig,
    config_from_external,
    current_runtime_identity,
    emit,
    get_nested_count,
    update_nested_count,
)
from ..server.path_resolver import WORKSPACE_ROOT, is_network_link, workspace_visible_path


class Config(StrictConfig):
    defaultDestination: str | None = Field(default=None, description="Default target agent name.")
    defaultMessageType: MessageType = Field(default="message", description="Default message type.")
    hideToolMessageContent: bool = Field(default=False, description="Whether to hide ToolMessage content.")
    currentAgentName: str | None = Field(default=None, description="Current agent name for self-send checks.")
    blockSelfTarget: bool = Field(default=True, description="Block messages sent to the current agent.")
    defaultRunId: str = Field(default="seedagent-run", description="Fallback LangVideo run id.")
    defaultThreadId: str = Field(default="default", description="Fallback LangVideo thread id.")

    @classmethod
    def load_config_send_message_tool(cls, source=None):
        return config_from_external(cls, source)


ToolRuningConfigSc = Config


def _sum_runs(left: int | None, right: int | None) -> int:
    return int(left or 0) + int(right or 0)


def _last_value(left: Any, right: Any) -> Any:
    return right if right is not None else left


class SubState(AgentState, total=False):
    sendMessageTotalRuns: Annotated[int, _sum_runs]
    sendMessageLastTarget: Annotated[str | None, _last_value]
    sendMessageLastResult: Annotated[str | None, _last_value]
    sendMessageLastError: Annotated[str | None, _last_value]


ToolStateTydict = SubState


class ToolLinkSc(BaseModel):
    link: str = Field(
        description=(
            "Attachment URL or sandbox file path. For a local file produced, read, or referenced by this agent, "
            "pass its /workspace/... path here. The receiver only gets local files listed in attachments; "
            "a /workspace/... path mentioned only in content is plain text and will not copy the file."
        )
    )
    summary: str = Field(
        default="",
        description="Short human-readable description of this attachment, for example 'draft report to review'.",
    )


class ToolTaskInfoSc(BaseModel):
    title: str = Field(description="Task title.")
    goal: str = Field(description="Task goal.")
    description: str = Field(description="Task details.")
    owner: str | None = Field(default=None, description="Target agent name. Defaults to dst.")


class ToolInputSm(BaseModel):
    content: str | None = Field(
        default=None,
        description=(
            "Message body. For msgType=task this can be used to auto-build taskInfo. "
            "Do not rely on content to transfer a file; if the receiver should inspect a document, image, "
            "or generated artifact, also include that file in attachments."
        ),
    )
    dst: str | None = Field(default=None, description="Target agent name. Empty value uses the default target.")
    msgType: MessageType = Field(default="message", description="Message type.")
    taskInfo: ToolTaskInfoSc | None = Field(default=None, description="Structured task content when msgType=task.")
    attachments: list[ToolLinkSc] = Field(
        default_factory=list,
        description=(
            "Files or links to physically send with this mail. Use this whenever the receiver must read, review, "
            "check, compare, ingest, or continue work from a document/file/image/artifact. Local files must use "
            "/workspace/... paths, for example attachments=[{'link':'/workspace/notes/report.md','summary':'report to review'}]. "
            "Network URLs may be passed directly. Mentioning a /workspace/... path in content is not enough. "
            "This same attachments field is used for both msgType='message' and msgType='task'."
        ),
    )


class ToolDescriptionSc(BaseModel):
    toolName: str = Field(default="send_message_to_agent")
    toolDescription: str = Field(
        default=(
            "Send an AgentMail as either a normal message or a task. Use msgType='message' for plain text "
            "and msgType='task' for taskInfo. If the receiver needs to inspect or use any local document, "
            "file, image, or generated artifact, put it in attachments; do not only mention the path in content. "
            "File transfer always uses the top-level attachments field for both message and task mails; "
            "taskInfo only describes the task and has no file-transfer field. "
            "Attachments must be network URLs or sandbox paths starting with /workspace; local files outside "
            "/workspace are not accepted."
        )
    )


class ToolReturnSc(BaseModel):
    successText: str = Field(default="Message sent.")
    failureText: str = Field(default="Message send failed.")


class ToolSchema:
    toolName = "send_message_to_agent"
    name = toolName
    args_schema = ToolInputSm
    description = ToolDescriptionSc().toolDescription
    toolfeedback = ToolReturnSc


ToolSpec = {
    "description": ToolDescriptionSc(),
    "inputSm": ToolInputSm,
    "stateTydict": ToolStateTydict,
    "returnsSc": ToolReturnSc(),
}


tool_runingconfig = Config.load_config_send_message_tool(Path(__file__).with_name("send_message_tool_config.json"))


def _is_network_url(link: str) -> bool:
    return is_network_link(link)


def _workspace_link_to_host_path(link: str) -> str:
    if _is_network_url(link):
        return link
    visible_link = workspace_visible_path(link)
    raw_path = PurePosixPath(visible_link)
    try:
        relative = raw_path.relative_to("/workspace")
    except ValueError as exc:
        raise ValueError("attachment links must resolve to /workspace unless they are network URLs.") from exc
    if any(part == ".." for part in relative.parts):
        raise ValueError("attachment link must not escape /workspace.")
    host_path = (WORKSPACE_ROOT / Path(*relative.parts)).resolve()
    workspace_root = WORKSPACE_ROOT.resolve()
    if host_path != workspace_root and workspace_root not in host_path.parents:
        raise ValueError("attachment link must stay inside /workspace.")
    return str(host_path)


def _normalize_attachments(raw_attachments: list[Any] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw in raw_attachments or []:
        if isinstance(raw, BaseModel):
            item = raw.model_dump()
        else:
            item = dict(raw)
        link = str(item.get("link") or "").strip()
        if not link:
            raise ValueError("attachment link must not be empty.")
        normalized.append(
            {
                "link": _workspace_link_to_host_path(link),
                "summary": str(item.get("summary") or "").strip(),
            }
        )
    return normalized


def _normalize_visible_links(raw_links: list[Any] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for raw in raw_links or []:
        if isinstance(raw, BaseModel):
            item = raw.model_dump()
        else:
            item = dict(raw)
        link = str(item.get("link") or "").strip()
        if not link:
            raise ValueError("deliverable link must not be empty.")
        normalized.append(
            {
                "link": workspace_visible_path(link),
                "summary": str(item.get("summary") or "").strip(),
            }
        )
    return normalized


def _task_payload(
    *,
    task_info: Any,
    content: str | None,
    owner: str,
) -> dict[str, Any]:
    if task_info is not None:
        payload = task_info.model_dump() if isinstance(task_info, BaseModel) else dict(task_info)
        payload["owner"] = str(payload.get("owner") or owner)
        payload.pop("deliverables", None)
        return payload
    body = str(content or "").strip()
    if not body:
        raise ValueError("content or taskInfo must be provided for task messages.")
    return {
        "title": body.splitlines()[0][:80] or "Task",
        "goal": body,
        "description": body,
        "owner": owner,
    }


class SendMessageTool:
    name = ToolSchema.name
    config = Config
    substate = SubState
    toolschema = ToolSchema

    def __init__(
        self,
        *,
        comm: AgentComm | None,
        toolSpec: dict[str, object] = ToolSpec,
        runingConfig: ToolRuningConfigSc = tool_runingconfig,
    ) -> None:
        self.comm = comm
        self.toolSpec = toolSpec
        self.runingConfig = runingConfig
        self.stateTydict = toolSpec["stateTydict"]
        self.tool = self.build_tool()

    def build_tool(self):
        tool_spec = self.toolSpec
        description: ToolDescriptionSc = tool_spec["description"]  # type: ignore[assignment]
        returns: ToolReturnSc = tool_spec["returnsSc"]  # type: ignore[assignment]

        @tool(
            description.toolName,
            args_schema=tool_spec["inputSm"],
            description=description.toolDescription,
        )
        def send_message_to_agent(
            runtime: ToolRuntime[Config, SubState],
            content: str | None = None,
            dst: str | None = None,
            msgType: MessageType = "message",
            taskInfo: dict[str, Any] | None = None,
            attachments: list[dict[str, str]] | None = None,
        ) -> Command:
            writer = runtime.stream_writer
            context = runtime.context or self.runingConfig
            target = (dst or context.defaultDestination or "").strip() or None
            message_type = msgType or context.defaultMessageType

            try:
                emit(
                    writer,
                    {
                        "type": "tool",
                        "tool": description.toolName,
                        "event": "start",
                        "target": target,
                        "messageType": message_type,
                    },
                )
                if self.comm is None:
                    raise RuntimeError("AgentComm is not injected; cannot send cross-agent messages.")
                if not target:
                    raise ValueError("dst or defaultDestination must provide a target agent name.")
                current_agent_name = str(context.currentAgentName or getattr(self.comm, "agent_name", "") or "").strip()
                if context.blockSelfTarget and current_agent_name and target == current_agent_name:
                    raise ValueError(
                        f"target agent '{target}' is the current agent; self-send is blocked by default."
                    )
                normalized_attachments = _normalize_attachments(attachments)
                if message_type == "task":
                    message_content: str | dict[str, Any] = _task_payload(
                        task_info=taskInfo,
                        content=content,
                        owner=target,
                    )
                else:
                    message_text = str(content or "").strip()
                    if not message_text:
                        raise ValueError("content must not be empty for message messages.")
                    message_content = message_text

                identity = current_runtime_identity(
                    defaultRunId=context.defaultRunId,
                    defaultThreadId=context.defaultThreadId,
                )
                result = self.comm.send(
                    dst=target,
                    content=message_content,
                    msg_type=message_type,
                    attachments=normalized_attachments,
                    metadata={
                        "thread_id": identity.threadId,
                        "run_id": identity.runId,
                        "conversation_thread_id": identity.threadId,
                        "conversation_run_id": identity.runId,
                    },
                )
                result_text = (
                    f"{returns.successText}\n"
                    f"dst={target}\n"
                    f"type={message_type}\n"
                    f"attachments={len(normalized_attachments)}\n"
                    f"server_result={result}"
                )
                emit(
                    writer,
                    {
                        "type": "tool",
                        "tool": description.toolName,
                        "event": "success",
                        "target": target,
                    },
                )
                message_content_text = "" if context.hideToolMessageContent else result_text
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=message_content_text,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "sendMessageTotalRuns": 1,
                        "sendMessageLastTarget": target,
                        "sendMessageLastResult": result_text,
                        "sendMessageLastError": None,
                    }
                )
            except Exception as exc:
                error_text = (
                    f"{returns.failureText}\n"
                    f"reason={type(exc).__name__}: {exc}\n"
                    "suggestion=Check whether AgentComm is injected, target agent is online, scope allows it, "
                    "and content is not empty."
                )
                emit(
                    writer,
                    {
                        "type": "tool",
                        "tool": description.toolName,
                        "event": "error",
                        "target": target,
                        "error": error_text,
                    },
                )
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=error_text,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "sendMessageTotalRuns": 1,
                        "sendMessageLastTarget": target,
                        "sendMessageLastResult": None,
                        "sendMessageLastError": error_text,
                    }
                )

        return send_message_to_agent


send_message_to_agent = SendMessageTool(comm=None).tool
send_message_tool = SendMessageTool
ToolDescription = ToolDescriptionSc
ToolInput = ToolInputSm
ToolReturn = ToolReturnSc
