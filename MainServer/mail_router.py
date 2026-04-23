"""Route local file attachments into a recipient agent workspace."""

import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger(__name__)

_NETWORK_SCHEMES = ("http://", "https://", "ftp://", "s3://", "file://")


def _is_url(value: str) -> bool:
    return any(value.startswith(scheme) for scheme in _NETWORK_SCHEMES)


def _is_local_file(value: str) -> bool:
    return not _is_url(value) and os.path.isfile(value)


def _safe_ts(timestamp: str) -> str:
    return timestamp.replace(":", "-").replace(".", "_")


def _mail_folder_name(sender_id: str, timestamp: str) -> str:
    safe_sender = sender_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{safe_sender}__{_safe_ts(timestamp)}"


def _render_message_md(
    message: dict[str, Any],
    processed_attachments: list[dict[str, Any]],
) -> str:
    sender = message.get("from", "unknown")
    to = message.get("to", "-")
    msg_type = message.get("type", "-")
    message_id = message.get("message_id", "-")
    content = message.get("content", "")

    lines = [
        "# Mail Message",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| From | `{sender}` |",
        f"| To | `{to}` |",
        f"| Type | `{msg_type}` |",
        f"| Message ID | `{message_id}` |",
        "",
        "## Content",
        "",
        str(content),
        "",
        "## Attachments",
        "",
    ]

    if not processed_attachments:
        lines.append("_No attachments._")
    else:
        for index, attachment in enumerate(processed_attachments, 1):
            summary = attachment.get("summary", "-")
            link = attachment.get("link", "")
            routed = attachment.get("_routed", "-")
            lines.extend(
                [
                    f"### {index}. {summary}",
                    "",
                    f"- link: `{link}`",
                    f"- routed: `{routed}`",
                    "",
                ]
            )

    return "\n".join(lines)


def route_message_assets(
    message: dict[str, Any],
    recipient_workspace: str,
) -> dict[str, Any]:
    attachments: list[dict[str, Any]] = message.get("attachments", [])
    if not attachments:
        return message

    sender_id = str(message.get("from", "unknown"))
    timestamp = str(message.get("message_id") or datetime.now(timezone.utc).isoformat())
    mail_folder = os.path.join(
        recipient_workspace,
        "mail",
        _mail_folder_name(sender_id, timestamp),
    )
    os.makedirs(mail_folder, exist_ok=True)

    new_attachments: list[dict[str, Any]] = []
    for raw_attachment in attachments:
        attachment = dict(raw_attachment)
        link = str(attachment.get("link") or attachment.get("value", ""))
        attachment["link"] = link

        if _is_url(link):
            attachment["_routed"] = "url"
            attachment["_mail_folder"] = mail_folder
        elif _is_local_file(link):
            destination = os.path.join(mail_folder, os.path.basename(link))
            try:
                shutil.copy2(link, destination)
                attachment["_original_link"] = link
                attachment["link"] = destination
                attachment["_routed"] = "copied"
                attachment["_mail_folder"] = mail_folder
            except Exception as exc:
                attachment["_routed"] = f"error: {exc}"
                attachment["_mail_folder"] = mail_folder
                logger.warning("Failed to route attachment %s: %s", link, exc)
        else:
            attachment["_routed"] = "not_found"
            attachment["_mail_folder"] = mail_folder

        new_attachments.append(attachment)

    message_md = os.path.join(mail_folder, "message.md")
    try:
        with open(message_md, "w", encoding="utf-8") as handle:
            handle.write(_render_message_md(message, new_attachments))
    except Exception as exc:
        logger.warning("Failed to write routed message summary %s: %s", message_md, exc)

    return {**message, "attachments": new_attachments}
