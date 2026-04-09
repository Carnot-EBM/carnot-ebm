#!/usr/bin/env bash
# validate-phase-gate.sh — Claude Code PreToolUse hook for Write|Edit
#
# Enforces spec-anchored development: before any source file can be written
# or edited, the corresponding OpenSpec capability must exist and contain
# at least one REQ-* requirement.
#
# Usage: validate-phase-gate.sh <FILE_PATH>
# Exit 0  — proceed (file is allowed or not a gated source file)
# Exit 2  — blocked (spec missing or has no REQ-*); prints feedback to stderr

set -euo pipefail

FILE_PATH="${1:-}"

# --------------------------------------------------------------------------
# Autoresearch bypass: when running in research mode, skip all phase gates.
# The research conductor sets CARNOT_MODE=research before spawning Claude.
# --------------------------------------------------------------------------
if [[ "${CARNOT_MODE:-}" == "research" ]]; then
  exit 0
fi

# If no file path provided, allow (nothing to validate)
if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

# --------------------------------------------------------------------------
# Capability mapping: source path prefixes -> OpenSpec capability names
# --------------------------------------------------------------------------
# Each entry maps a source directory to its corresponding spec directory
# under openspec/capabilities/. Only files within these directories are
# gated; everything else (tests, docs, scripts, configs, specs themselves)
# passes through freely.

declare -A CAP_MAP=(
  ["python/carnot/models/"]="model-tiers"
  ["crates/carnot-boltzmann/"]="model-tiers"
  ["crates/carnot-gibbs/"]="model-tiers"
  ["crates/carnot-ising/"]="model-tiers"
  ["python/carnot/training/"]="training-inference"
  ["python/carnot/inference/"]="llm-ebm-inference"
  ["python/carnot/verification/"]="verifiable-reasoning"
  ["python/carnot/autoresearch/"]="autoresearch"
  ["crates/carnot-core/"]="core-ebm"
)

# --------------------------------------------------------------------------
# Determine if this file is a gated source file
# --------------------------------------------------------------------------
# We match against relative paths. Strip any leading ./ or absolute prefix
# that includes the repo root, so we compare against clean relative paths.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
REL_PATH="$FILE_PATH"

# If FILE_PATH is absolute and starts with the repo root, make it relative
if [[ -n "$REPO_ROOT" && "$FILE_PATH" == "$REPO_ROOT"/* ]]; then
  REL_PATH="${FILE_PATH#"$REPO_ROOT"/}"
fi

# Strip leading ./
REL_PATH="${REL_PATH#./}"

CAPABILITY=""

for prefix in "${!CAP_MAP[@]}"; do
  if [[ "$REL_PATH" == "$prefix"* ]]; then
    CAPABILITY="${CAP_MAP[$prefix]}"
    break
  fi
done

# --------------------------------------------------------------------------
# If the file is not under a gated source directory, allow it
# --------------------------------------------------------------------------
# This covers: tests/, docs/, scripts/, configs, openspec/ itself,
# Cargo.toml, pyproject.toml, .github/, etc.
if [[ -z "$CAPABILITY" ]]; then
  exit 0
fi

# --------------------------------------------------------------------------
# Check that the spec file exists
# --------------------------------------------------------------------------
SPEC_DIR="openspec/capabilities/${CAPABILITY}"
if [[ -n "$REPO_ROOT" ]]; then
  SPEC_FILE="${REPO_ROOT}/${SPEC_DIR}/spec.md"
else
  SPEC_FILE="${SPEC_DIR}/spec.md"
fi

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "BLOCKED: Spec-anchored development requires a spec before implementation." >&2
  echo "" >&2
  echo "File:       $REL_PATH" >&2
  echo "Capability: $CAPABILITY" >&2
  echo "Expected:   ${SPEC_DIR}/spec.md" >&2
  echo "" >&2
  echo "Action: Create ${SPEC_DIR}/spec.md with at least one REQ-* requirement" >&2
  echo "before writing source code for this capability." >&2
  exit 2
fi

# --------------------------------------------------------------------------
# Check that the spec contains at least one REQ-* requirement
# --------------------------------------------------------------------------
# We look for lines containing REQ- followed by alphanumeric/dash/underscore
# characters, which is the standard requirement identifier format.
if ! grep -qE 'REQ-[A-Za-z0-9_-]+' "$SPEC_FILE"; then
  echo "BLOCKED: Spec exists but contains no requirements (REQ-*)." >&2
  echo "" >&2
  echo "File:       $REL_PATH" >&2
  echo "Capability: $CAPABILITY" >&2
  echo "Spec:       ${SPEC_DIR}/spec.md" >&2
  echo "" >&2
  echo "Action: Add at least one REQ-* requirement to the spec before" >&2
  echo "writing source code for this capability." >&2
  exit 2
fi

# All checks passed — spec exists and has requirements
exit 0
