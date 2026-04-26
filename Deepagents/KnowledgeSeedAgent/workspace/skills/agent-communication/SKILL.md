---
name: agent-communication
description: Use this skill when you need to send a task or message to another agent, especially when attachments or exact delivery requirements matter.
allowed-tools: send_message_to_agent read_file write_file
---

# Agent Communication Skill

## When to Use

- You need to send a `message` or `task` to another agent.
- The user gives exact delivery text or exact artifact requirements.
- You need to attach a local file from `/workspace/...`.

## Rules

1. Confirm the target agent name before sending.
2. Use `msgType="task"` when assigning work to another agent.
3. Use `msgType="message"` when reporting progress or final delivery.
4. Local attachments must use `/workspace/...` paths.
5. If the task requires an artifact, verify the file exists before sending it.
6. When exact text is required, keep the content byte-accurate.

## Recommended Workflow

1. Read the request and extract:
   - target agent
   - message type
   - exact content
   - required attachments
2. If an attachment is local, verify it exists.
3. Call `send_message_to_agent`.
4. If the tool reports an error, fix the cause and retry with corrected parameters.

## Output Style

- Keep the final acknowledgement short.
- Do not claim delivery succeeded unless the tool call succeeded.
