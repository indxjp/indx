#!/usr/bin/env bash
# Build the Next.js web UI and sync the static export into the packaged bundle dir.
#
# Produces the shippable SPA for `indx app` (docs/app-spec.md §5):
#   webapp/  --(next build, output:'export')-->  webapp/out/  -->  src/indx/app/static/
#
# The bundle dir is gitignored except for a tracked `.gitkeep` and a fallback `index.html`
# (the export overwrites index.html, which is fine). Node is a BUILD-TIME dependency only;
# nothing here runs at `indx app` runtime.
#
# Make this executable once with: chmod +x scripts/build_webapp.sh
# (it is also invoked as `bash scripts/build_webapp.sh`, which works without the +x bit).
set -euo pipefail

# --- Resolve the repo root from this script's location (cwd-independent). ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WEBAPP_DIR="webapp"
EXPORT_DIR="${WEBAPP_DIR}/out"
STATIC_DIR="src/indx/app/static"

# --- Preconditions. ----------------------------------------------------------------------
command -v npm >/dev/null 2>&1 || { echo "error: npm is required (Node build-time dep)." >&2; exit 1; }
[ -d "${WEBAPP_DIR}" ] || { echo "error: ${WEBAPP_DIR}/ not found." >&2; exit 1; }

# --- Install dependencies (reproducible `ci` when a lockfile exists, else `install`). -----
if [ -f "${WEBAPP_DIR}/package-lock.json" ]; then
  echo ">> npm ci (${WEBAPP_DIR})"
  npm --prefix "${WEBAPP_DIR}" ci
else
  echo ">> npm install (${WEBAPP_DIR}; no lockfile)"
  npm --prefix "${WEBAPP_DIR}" install
fi

# --- Build the static export (output:'export' writes ${WEBAPP_DIR}/out/). -----------------
echo ">> npm run build (${WEBAPP_DIR})"
npm --prefix "${WEBAPP_DIR}" run build

[ -f "${EXPORT_DIR}/index.html" ] || {
  echo "error: ${EXPORT_DIR}/index.html missing after build (is output:'export' set?)." >&2
  exit 1
}

# --- Sync the export into the packaged bundle dir, preserving the tracked .gitkeep. -------
mkdir -p "${STATIC_DIR}"
echo ">> syncing ${EXPORT_DIR}/ -> ${STATIC_DIR}/"
rsync -a --delete --exclude='.gitkeep' "${EXPORT_DIR}/" "${STATIC_DIR}/"

# If for any reason the export was empty, restore a minimal fallback index.html so the
# server still serves *something* (and `git` keeps a tracked entry).
if [ ! -f "${STATIC_DIR}/index.html" ]; then
  echo ">> export produced no index.html; writing a fallback" >&2
  printf '<!doctype html><meta charset="utf-8"><title>indx app</title>%s\n' \
    '<p>The web UI bundle is empty. Run scripts/build_webapp.sh.</p>' \
    > "${STATIC_DIR}/index.html"
fi

echo "OK: built webapp and synced bundle into ${STATIC_DIR}/"
