---
name: agent-browser
description: Browser automation entrypoint for agent sandbox runs. Use when the agent needs to interact with websites, automate browser workflows, inspect page state, or collect screenshots and extracted data.
allowed-tools: execute read_file write_file
hidden: true
---

# agent-browser

This skill is only the discovery entrypoint.

Before using browser commands, load the live workflow from the installed CLI:

- `agent-browser skills get core`
- `agent-browser skills get core --full`

Use the CLI guidance for up-to-date commands, patterns, and troubleshooting.
If you need a specialized workflow, load it from the CLI as well:

- `agent-browser skills get electron`
- `agent-browser skills get slack`
- `agent-browser skills get dogfood`
- `agent-browser skills get vercel-sandbox`
- `agent-browser skills get agentcore`

When Chromium needs to be selected explicitly, use the sandbox path:

- `/usr/bin/chromium`

