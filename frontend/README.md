# LANGVIDEO Agents Frontend

React + Vite control panel for the local MainServer.

One-click local startup from the repository root:

```bash
./scripts/start_langvideo.sh
```

The script starts MainServer, the Vite frontend, enabled seed agent services,
and opens `http://127.0.0.1:5173` in the browser.

Run MainServer first:

```bash
uv run python main.py mainserver
```

Then run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000`.

The create form can copy from any available agent directory. `SeedAgent` and
`KnowledgeSeedAgent` are both knowledge-capable seed templates with document
ingest and memory management tools.
The prompt tab separates public `AgentServer/AgentCard.json` from the actual
agent prompt in `workspace/brain/AGENTS.md`. The config tab shows a merged
MainServer/runtime view; model service fields save back to the agent runtime
config, while communication/UI fields stay in MainServer config.
The graph tab edits global communication spaces. Every member in a space can
communicate with every other member, and overlapping spaces behave like a Venn
diagram: shared agents can talk into both spaces while non-shared members remain
separate unless another space connects them.
The chat tab stores conversation history in MainServer UI state and shows an
agent activity stream beside the conversation. The chat lane only renders user
and agent messages; the detail lane renders filtered message and tool-call cards
instead of raw status/update payloads.
Saving runtime config asks the running AgentServer to reload its main agent when
the service is registered.
Clearing checkpoints also asks the AgentServer to reload, so the main agent's
LangGraph sqlite checkpointer reconnects to `Agent/store/checkpoints/langgraph.sqlite3`
instead of keeping an old deleted file handle.
The clear-cache button only clears agent checkpoint directories and the saved
chat/activity history for that agent; it does not remove mail, knowledge, or
memory cache.
