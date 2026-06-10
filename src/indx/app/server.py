"""FastAPI app factory + uvicorn launcher for ``indx app`` (docs/app-spec.md §2).

``fastapi`` / ``starlette`` / ``uvicorn`` are imported **lazily inside** these functions (never
at module top), so importing this module is safe on a core-only install. :func:`create_app`
mounts the ``/api`` router and, when the SPA bundle is packaged, serves it at ``/`` with a
catch-all that returns ``index.html`` for any non-``/api`` path. :func:`serve` runs uvicorn.
"""

from __future__ import annotations

import importlib.resources as resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI


def _static_dir() -> resources.abc.Traversable:
    """The packaged static bundle directory (may not contain index.html in dev)."""
    return resources.files("indx.app") / "static"


# Shown at '/' when the SPA bundle has not been built (dev checkout / core install). The built
# bundle is build-time only (scripts/build_webapp.sh); kept inline so there is no tracked file
# for ``next build`` to clobber. The tiny script confirms the API is live.
_FALLBACK_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>indx app</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font:15px/1.6 system-ui,sans-serif;max-width:42rem;margin:4rem auto;padding:0 1.5rem;
color:#1f2937}code{background:#f3f4f6;padding:.15em .4em;border-radius:4px}
.ok{color:#059669}.muted{color:#6b7280}</style></head><body>
<h1>indx app</h1>
<p>The web UI bundle has not been built yet. The JSON API is live at <code>/api</code>.</p>
<p>To build the UI, run from the repo root:</p>
<pre><code>bash scripts/build_webapp.sh</code></pre>
<p class="muted">API status: <span id="s">checking…</span></p>
<script>fetch('/api/health').then(r=>r.json()).then(d=>{
document.getElementById('s').innerHTML='<span class="ok">ok</span> · indx '+d.version;
}).catch(()=>{document.getElementById('s').textContent='unreachable';});</script>
</body></html>"""


def create_app() -> FastAPI:
    """Build the FastAPI app: ``/api`` router plus the SPA catch-all when bundled."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from indx import __version__
    from indx.app.api import _cleanup_app_temp_dirs, build_router

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        # Remove the app-owned temp dirs (build outputs, demo spaces, import work dir) on
        # shutdown. The ASGI lifespan fires reliably under uvicorn's signal handling, whereas a
        # bare ``atexit`` hook does not (uvicorn's SIGTERM path can skip interpreter atexit) — so
        # this, not the atexit fallback in api.py, is what actually cleans up on a normal Ctrl-C.
        yield
        _cleanup_app_temp_dirs()

    app = FastAPI(title="indx app", version=__version__, lifespan=lifespan)
    app.include_router(build_router(), prefix="/api")

    # Mount the exported Next.js SPA at '/' only when the bundle is present (index.html). In a
    # core/dev checkout the bundle is gitignored, so the API is still fully usable on its own —
    # and '/' serves an inline "not built yet" page (below) instead of a bare 404.
    static = _static_dir()
    index = static / "index.html"
    if not index.is_file():

        @app.get("/", response_class=HTMLResponse)
        def root_fallback() -> HTMLResponse:
            return HTMLResponse(_FALLBACK_HTML)

    if index.is_file():
        with resources.as_file(static) as static_path:
            app.mount(
                "/assets",
                StaticFiles(directory=str(static_path)),
                name="assets",
            )

            index_path = static_path / "index.html"

            root = static_path.resolve()

            @app.get("/{full_path:path}")
            def spa(full_path: str) -> JSONResponse | FileResponse:
                # Never shadow the API; anything else falls back to the SPA entrypoint so
                # client-side routing works (output:'export' single-page catch-all).
                if full_path == "api" or full_path.startswith("api/"):
                    return JSONResponse({"detail": "Not Found"}, status_code=404)
                if full_path:
                    candidate = (root / full_path).resolve()
                    # Defense-in-depth: only ever serve files that stay inside the bundle, so a
                    # crafted ``..`` path can never read outside static/ (the ASGI layer usually
                    # normalizes these, but we never rely on that).
                    if candidate.is_file() and candidate.is_relative_to(root):
                        return FileResponse(str(candidate))
                return FileResponse(str(index_path))

    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    open_browser: bool = True,
) -> None:
    """Run the app under uvicorn (no reload), optionally opening a browser first."""
    import uvicorn

    if open_browser:
        import webbrowser

        webbrowser.open(f"http://{host}:{port}")

    uvicorn.run(create_app(), host=host, port=port)
