---
name: agent-communication
description: Use this skill when you need to receive a task, produce an artifact, and report completion back to another agent with exact content or attachments.
allowed-tools: send_message_to_agent read_file write_file
---

# Agent Communication Skill

## When to Use

- You receive a task from another agent.
- You must generate or verify a file under `/workspace/...`.
- You need to send a completion message with exact text or attachments.

## Rules

1. Read the latest task carefully before acting.
2. Produce the requested artifact first when the task depends on a file.
3. Verify the artifact exists and matches the requested content.
4. Use `msgType="message"` for final delivery unless the caller explicitly asks for another task.
5. Local attachments must use `/workspace/...` paths.
6. If the first send fails, correct the issue and try again.

## Recommended Workflow

1. Inspect the task requirements.
2. Create or update the requested artifact.
3. Validate the file content.
4. Call `send_message_to_agent` with the exact required content and attachments.
5. Summarize completion briefly.

## Output Style

- Prefer compact completion messages.
- Keep delivery text exact when the task says "must be exactly".
