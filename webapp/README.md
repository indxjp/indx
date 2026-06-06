# indx webapp

The Next.js (App Router) frontend for `indx app` — a single-page tabbed tester
(**Config · Build · Inspect · Query**) for an INDX knowledge space. It is **build-time only**:
`next build` with `output: 'export'` produces static HTML/CSS/JS that FastAPI serves directly,
so there is **no Node at runtime**.

See `docs/app-spec.md` for the authoritative `/api` contract this app targets.

## Layout

```
app/            # App Router entry: layout.tsx, page.tsx (the SPA shell + tabs), globals.css
components/     # ConfigTab, BuildTab, InspectTab, QueryTab, shared ui.tsx (cards, browse modal)
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
