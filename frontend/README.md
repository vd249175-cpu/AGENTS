# LANGVIDEO Agents Frontend

React + Vite control panel for the local MainServer.

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

The create form can copy from either `SeedAgent` for a general agent or
`KnowledgeSeedAgent` for an agent with document ingest and memory management tools.
The prompt tab edits both runtime prompt fields and `workspace/brain/AGENTS.md`.
The graph tab edits global communication spaces. Every member in a space can
communicate with every other member, and overlapping spaces behave like a Venn
diagram: shared agents can talk into both spaces while non-shared members remain
separate unless another space connects them.
The chat tab stores conversation history in MainServer UI state and shows an
agent activity stream in the same conversation surface. The activity stream is
filtered to rendered message and tool-call cards instead of raw status/update
payloads.
The clear-cache button only clears agent checkpoint directories and the saved
chat/activity history for that agent; it does not remove mail, knowledge, or
memory cache.
