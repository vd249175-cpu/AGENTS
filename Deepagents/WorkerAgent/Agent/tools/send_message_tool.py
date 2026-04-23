from pathlib import Path, PurePosixPath
from typing import Any

from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from MainServer.comm import AgentComm
from MainServer.state import MessageType
from Deepagents.WorkerAgent.Agent.server.path_resolver import WORKSPACE_ROOT
from Deepagents.WorkerAgent.Agent.server.stream_events import emit_event


NETWORK_SCHEMES = ("http://", "https://", "ftp://", "s3://", "file://")


class ToolLinkSc(BaseModel):
    link: str = Field(description="Network URL or a sandbox file path starting with /workspace.")
    summary: str = Field(default="", description="Short description of this attachment.")


class ToolTaskInfoSc(BaseModel):
    title: str = Field(description="Task title.")
    goal: str = Field(description="Task goal.")
    description: str = Field(description="Task details.")
    owner: str | None = Field(default=None, description="Target agent name. Defaults to dst.")
    deliverables: list[ToolLinkSc] = Field(default_factory=list)


class ToolInputSm(BaseModel):
    content: str | None = Field(default=None, description="Message body. For msgType=task this can be used to auto-build taskInfo.")
    dst: str | None = Field(default=None, description="Target agent name. Empty value uses the default target.")
    msgType: MessageType = Field(default="message", description="Message type.")
    taskInfo: ToolTaskInfoSc | None = Field(default=None, description="Structured task content when msgType=task.")
    attachments: list[ToolLinkSc] = Field(
        default_factory=list,
        description="Files or links to send. Use /workspace/... for local sandbox files, unless the value is a network URL.",
    )


class ToolDescriptionSc(BaseModel):
    toolName: str = Field(default="send_message_to_agent")
    toolDescription: str = Field(
        default=(
            "Send an AgentMail as either a normal message or a task. Use msgType='message' for plain text "
            "and msgType='task' for taskInfo. Attachments must be network URLs or sandbox paths starting "
            "with /workspace; local files outside /workspace are not accepted."
        )
    )


class ToolReturnSc(BaseModel):
    successText: str = Field(default="Message sent.")
    failureText: str = Field(default="Message send failed.")


class ToolStateTydict(AgentState, total=False):
    sendMessageTotalRuns: int
    sendMessageLastTarget: str | None
    sendMessageLastResult: str | None
    sendMessageLastError: str | None


class ToolRuningConfigSc(BaseModel):
    defaultDestination: str | None = Field(default=None)
    defaultMessageType: MessageType = Field(default="message")
    hideToolMessageContent: bool = Field(default=False)
    currentAgentName: str | None = Field(default=None)
    blockSelfTarget: bool = Field(default=True)


ToolSpec = {
    "description": ToolDescriptionSc(),
    "inputSm": ToolInputSm,
    "stateTydict": ToolStateTydict,
    "returnsSc": ToolReturnSc(),
}

tool_runingconfig = ToolRuningConfigSc()


def _is_network_url(link: str) -> bool:
    return link.startswith(NETWORK_SCHEMES)


def _workspace_link_to_host_path(link: str) -> str:
    if _is_network_url(link):
        return link
    raw_path = PurePosixPath(link)
    try:
        relative = raw_path.relative_to("/workspace")
    except ValueError as exc:
        raise ValueError("attachment links must start with /workspace unless they are network URLs.") from exc
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


def _task_payload(
    *,
    task_info: Any,
    content: str | None,
    owner: str,
) -> dict[str, Any]:
    if task_info is not None:
        payload = task_info.model_dump() if isinstance(task_info, BaseModel) else dict(task_info)
        payload["owner"] = str(payload.get("owner") or owner)
        payload["deliverables"] = _normalize_attachments(payload.get("deliverables") or [])
        return payload
    body = str(content or "").strip()
    if not body:
        raise ValueError("content or taskInfo must be provided for task messages.")
    return {
        "title": body.splitlines()[0][:80] or "Task",
        "goal": body,
        "description": body,
        "owner": owner,
        "deliverables": [],
    }


class SendMessageTool:
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
        self.tool = self.buildTool()

    def buildTool(self):
        tool_spec = self.toolSpec
        description = tool_spec["description"]
        returns = tool_spec["returnsSc"]

        @tool(
            description.toolName,
            args_schema=tool_spec["inputSm"],
            description=description.toolDescription,
        )
        def runSendMessageTool(
            runtime: ToolRuntime,
            content: str | None = None,
            dst: str | None = None,
            msgType: MessageType = "message",
            taskInfo: dict[str, Any] | None = None,
            attachments: list[dict[str, str]] | None = None,
        ) -> Command:
            current_state = runtime.state
            writer = runtime.stream_writer
            target = dst or self.runingConfig.defaultDestination
            message_type = msgType or self.runingConfig.defaultMessageType

            try:
                emit_event(
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
                current_agent_name = str(
                    self.runingConfig.currentAgentName or getattr(self.comm, "agent_name", "") or ""
                ).strip()
                if self.runingConfig.blockSelfTarget and current_agent_name and target == current_agent_name:
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

                result = self.comm.send(
                    dst=target,
                    content=message_content,
                    msg_type=message_type,
                    attachments=normalized_attachments,
                )
                result_text = (
                    f"{returns.successText}\n"
                    f"dst={target}\n"
                    f"type={message_type}\n"
                    f"attachments={len(normalized_attachments)}\n"
                    f"server_result={result}"
                )
                emit_event(
                    writer,
                    {
                        "type": "tool",
                        "tool": description.toolName,
                        "event": "success",
                        "target": target,
                    },
                )
                message_content = "" if self.runingConfig.hideToolMessageContent else result_text
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=message_content,
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                        "sendMessageTotalRuns": int(current_state.get("sendMessageTotalRuns", 0) or 0) + 1,
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
                emit_event(
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
                        "sendMessageLastTarget": target,
                        "sendMessageLastResult": None,
                        "sendMessageLastError": error_text,
                    }
                )

        return runSendMessageTool
