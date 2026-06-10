#!/usr/bin/env bash
# Stage the PUBLIC subset of indx-meta into a destination tree (a checkout of the
# public repo, indxjp/indx) so a release PR can be opened against it.
#
# The public repo is a curated mirror: it ships the buildable / publishable project
# but EXCLUDES internal-only material (design docs, the docs site, .claude state, and
# the meta-only release plumbing). This script is the single source of truth for that
# split — keep the EXCLUDES list below in sync with how the public repo was first
# carved out (see the indx-pypi-release notes / two-repo split).
#
# Usage: scripts/sync_public.sh <dest-dir>
#   <dest-dir> is a working copy of indxjp/indx; its .git/ is preserved.
set -euo pipefail

DEST="${1:?usage: sync_public.sh <dest-dir>}"
SRC="$(git rev-parse --show-toplevel)"

# Internal-only paths that must never reach the public repo, anchored at the repo
# root. A trailing slash means "this directory and everything under it".
EXCLUDES=(
  ".claude/"
  ".docsite-style-guide.md"
  "docs/"
  "docsite/"
  ".github/workflows/docsite-deploy.yml"
  ".github/workflows/release-pr.yml"
)

# git archive emits exactly the tracked files at the checked-out commit (the tag, in
# CI). This automatically excludes gitignored build junk — node_modules/, the Next.js
# static bundle under src/indx/app/static/, .coverage, .claude/worktrees/ full-repo
# copies — that a plain rsync of the working tree would otherwise leak.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git -C "$SRC" archive --format=tar HEAD | tar -x -C "$STAGE"

for e in "${EXCLUDES[@]}"; do
  rm -rf "${STAGE:?}/$e"
done

# Mirror the staged subset into DEST, deleting anything that is no longer part of the
# public set (e.g. a source file removed since the last release) while preserving the
# destination's git metadata.
rsync -a --delete --exclude='.git/' "$STAGE"/ "$DEST"/

echo "Synced public subset into $DEST"
