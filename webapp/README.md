# indx webapp

The Next.js (App Router) frontend for **INDX** (`indx app`) — a local app for turning a
pile of unorganized files into an organized, AI-ready knowledge base you can talk to, and
watching that transformation happen. Think Obsidian or Notion, not a developer console: you
point it at your data and it comes back **typed, linked, summarized, graphed, and queryable**,
with every step shown on screen.

The UI is shaped around the user's journey, not the engine's verbs:

- **Library** — open, manage, and switch between knowledge spaces.
- **Ingest** — dump data in (a folder, files, a zip); see what was found and how it will be
  processed via simple presets, before any model runs.
- **Organize** — watch the build narrate the transformation live over SSE, then browse the
  result like a workspace: the relationship graph, document-type breakdown, topic/tag clusters,
  summaries, chunks, and neighbors.
- **Ask** — query the space with ranked, grounded hits and full lineage; the space is now an
  agent over your data, ready to hand to LangChain / LlamaIndex / MCP via the `/api/agent/*`
  connector.

A knowledge space is a portable, self-contained `.indx` artifact, so **import/export is a
first-class capability**: open an existing `.indx` space (or `jsonl` output dir) and land
straight in the organized/browse view with no rebuild, and export the built archive to share,
version, back up, or migrate elsewhere.

The webapp is a frontend that orchestrates and visualizes the existing INDX engine — it does not
introduce its own parser, vector DB, or retrieval runtime. It is **build-time only**:
`next build` with `output: 'export'` produces static HTML/CSS/JS that FastAPI serves directly,
so there is **no Node at runtime**.

See `docs/app-spec.md` for the authoritative product spec and the `/api` contract this app targets.

## Layout

```
app/            # App Router entry: layout.tsx, page.tsx (the app shell), globals.css
components/     # Library / Ingest / Organize / Ask views + shared ui.tsx (cards, browse modal)
lib/api.ts      # typed /api client + streamBuild() SSE helper (fetch + ReadableStream reader)
lib/types.ts    # TS interfaces mirroring the Pydantic models in src/indx/app/models.py
next.config.mjs # output:'export'; dev rewrites /api/* -> http://127.0.0.1:8000
```

## Dev

Run the backend and the frontend in two terminals. Next serves on `:3000` and proxies
`/api/*` to the FastAPI backend on `:8000` via `rewrites()` in `next.config.mjs`
(rewrites are dev-only — they are a no-op under `output: 'export'`).

```bash
# terminal 1 — backend (from the repo root)
cd /path/to/indx && . .venv/bin/activate
pip install fastapi 'uvicorn[standard]'   # the indx[app] extra
indx app --no-open                         # FastAPI on :8000

# terminal 2 — frontend (hot reload, proxies /api -> :8000)
npm install            # from webapp/
npm run dev            # Next on :3000  ->  open http://127.0.0.1:3000
```

The header shows the live `/api/health` version and whether the bundled SPA is present.

## Build (shippable static bundle)

```bash
npm run build          # -> webapp/out/  (static export)
```

Use the repo-root helper to build **and** sync the bundle into the wheel-packaged location
(`src/indx/app/static/`, gitignored except `.gitkeep` + the fallback `index.html`):

```bash
bash scripts/build_webapp.sh
```

After that, `indx app` serves the real SPA same-origin alongside `/api`.

## Scripts

| script          | purpose                                   |
|-----------------|-------------------------------------------|
| `npm run dev`   | Next dev server (`:3000`, proxies `/api`) |
| `npm run build` | static export to `out/`                   |
| `npm run start` | serve a non-export build (not used in prod)|
| `npm run lint`  | `next lint`                               |
