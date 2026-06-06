#!/usr/bin/env bash
# Install Carnot's standalone git hooks into .git/hooks.
#
# Currently: pre-push (the blog publish guard — see scripts/git-hooks/pre-push).
# Run once per clone. Safe to re-run (idempotent overwrite). If you ever run
# `pre-commit install --hook-type pre-push`, it will clobber the pre-push hook
# with pre-commit's stashing dispatcher — re-run THIS script to restore the
# standalone (no-stash) version.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
install -m 0755 "${ROOT}/scripts/git-hooks/pre-push" "${ROOT}/.git/hooks/pre-push"
echo "installed .git/hooks/pre-push (standalone guards: blog-publish + gitlink)"
