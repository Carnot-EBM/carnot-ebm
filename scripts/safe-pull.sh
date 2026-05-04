#!/bin/bash
# safe-pull.sh — commit-first-then-pull, never stash.
#
# Use this instead of `git pull` when the working tree may be dirty
# from in-flight conductor work or transient outer-loop edits.
#
# Why: `git stash` + `git pull` + `git stash pop` creates a 1-2 second
# window where the working tree is reverted to HEAD. If the conductor's
# codex/claude subprocesses hold open file handles during that window,
# they can write to files that no longer exist in the working tree,
# causing silent data loss or conflicts on stash pop.
#
# Commit-first never has this risk because the working tree is never
# reverted. The cost is identical to the conductor's natural auto-commit
# pattern; it just lands a few seconds earlier.
#
# Codified in CLAUDE.md "Never Stash — Always Commit-First (MANDATORY)"
# and memory feedback_outer_loop_role.md / feedback_never_stash.md.
#
# Origin: 2026-05-04 14:30Z incident where outer-loop reflexively
# stashed during a pull, lucking out that no codex subprocess was
# mid-write but exposing the procedural gap.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "${REPO_ROOT}" ]; then
  echo "safe-pull: not in a git repository" >&2
  exit 1
fi

cd "${REPO_ROOT}"

# Refuse to run if `git stash list` shows pending stashes — those
# indicate an earlier failed flow that needs operator review before
# safe-pull layers more state on top.
if [ -n "$(git stash list 2>/dev/null)" ]; then
  echo "safe-pull: existing stashes detected. Resolve them first:" >&2
  git stash list >&2
  echo "  git stash pop      # or git stash drop after review" >&2
  exit 2
fi

# Detect dirty state.
if [ -n "$(git status --porcelain)" ]; then
  echo "safe-pull: dirty working tree — committing transient state before pull"

  # Add all (tracked + untracked). The conductor's auto-commit pattern
  # uses `git add -A` so this matches existing precedent.
  git add -A

  # Commit only if the index is non-empty after add (avoid empty
  # commits on edge cases).
  if ! git diff --cached --quiet; then
    git commit --no-verify -m "$(cat <<EOF
[safe-pull] preserve transient state before pull

Auto-commit by scripts/safe-pull.sh to prevent stash-window data loss
from in-flight conductor subprocesses. See CLAUDE.md "Never Stash —
Always Commit-First (MANDATORY)" for rationale.
EOF
)"
  fi
fi

# Now pull cleanly. --rebase keeps history linear; --autostash is
# explicitly NOT used because that's the exact failure mode this
# script exists to prevent.
echo "safe-pull: pulling with --rebase"
git pull --rebase --no-autostash "$@"

echo "safe-pull: complete"
