#!/usr/bin/env python3
"""Standalone linter: validates a research-roadmap YAML file for schema
correctness and prior-failure field completeness.

Why this exists: the conductor DOOMED_RERUN_BLOCK fired 10 times across
milestones .77/.78/.79 because the planner kept proposing tasks that
re-attempted previously-failed scope without a prior_failures YAML field.
This linter catches that omission at planner-output time, before the
conductor can activate the roadmap.

Usage:
    python scripts/validate_prior_failures.py research-roadmap.yaml
    python scripts/validate_prior_failures.py research-roadmap-next.yaml

Exit codes:
    0  clean (no schema errors, no missing prior_failures)
    1  one or more violations found
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

# Add scripts/ to path so roadmap_schema is importable when run as a script
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from roadmap_schema import Roadmap, ResearchTask  # noqa: E402

_PROJECT_ROOT = Path(__file__).parent.parent
_COMPLETE_FILE = _PROJECT_ROOT / "research-complete.yaml"

# Minimum number of overlapping title tokens to consider two experiments
# the same scope. 3 tokens balances recall (catch reruns) vs precision
# (avoid false positives on generic words like "exp" or "v2").
_MIN_TOKEN_OVERLAP = 3

# Tokens that appear in almost every experiment title and should not count
# toward scope matching. The "category keywords" group captures recurring
# task types (preflight every milestone, retrospective every milestone,
# manifest hygiene, SOTA-model verification) — these are not "rerun"
# signals; they are routine plumbing each milestone repeats by design.
# Without these in stopwords, the linter false-positives every recurring
# task as a doomed-rerun candidate, which makes the .80 roadmap fail
# even though every recurring task is, by definition, expected to recur.
_STOPWORDS = {
    "exp",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",
    "v29",
    "v30",
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "on",
    "with",
    "of",
    "in",
    "to",
    "from",
    "via",
    "using",
    "based",
    "new",
    "part",
    "phase",
    "milestone",
    "run",
    "test",
    "final",
    "live",
    "gpu",
    "required",
    "fix",
    "repair",
    "update",
    "add",
    "build",
    "create",
    # Recurring-task category keywords (see comment above)
    "preflight",
    "retrospective",
    "retro",
    "manifest",
    "sota",
    "verify",
    "verification",
    "audit",
    "check",
    "model",
    "models",
    "schema",
    "infrastructure",
    "operational",
    "closure",
    "expansion",
    "first",
    "light",
    "rocm",
    "cuda",
    "production",
    "wiring",
    "next",
    "session",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase-split title into meaningful tokens, dropping stopwords."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if t not in _STOPWORDS and len(t) > 2}


def _build_failure_index(complete_path: Path) -> list[tuple[str, set[str]]]:
    """Load research-complete.yaml and return (exp_id, token_set) pairs for
    every task that appears in the failure record (any completed/archived task).

    We use *all* archived tasks as the scope-match corpus, not just failed ones,
    because the planner cannot easily distinguish archived-successful from
    archived-failed when re-proposing. The validator only flags tasks that
    match AND lack prior_failures — that is the actionable signal.
    """
    if not complete_path.exists():
        return []

    try:
        with open(complete_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []

    index: list[tuple[str, set[str]]] = []
    for milestone in data.get("milestones", []):
        for task in milestone.get("tasks", []):
            task_id = task.get("id", "")
            title = task.get("title", "")
            tokens = _tokenize(title)
            if tokens:
                index.append((task_id, tokens))
    return index


def _scope_matches(
    task_title: str,
    failure_index: list[tuple[str, set[str]]],
) -> list[str]:
    """Return list of archived experiment IDs whose title overlaps >= MIN_TOKEN_OVERLAP
    tokens with task_title."""
    task_tokens = _tokenize(task_title)
    matches = []
    for exp_id, prior_tokens in failure_index:
        if len(task_tokens & prior_tokens) >= _MIN_TOKEN_OVERLAP:
            matches.append(exp_id)
    return matches


def validate_roadmap(
    roadmap_path: Path,
    complete_path: Path = _COMPLETE_FILE,
) -> tuple[list[str], list[str]]:
    """Validate roadmap_path for schema correctness and prior-failure completeness.

    Returns:
        (schema_errors, prior_failure_violations)
        Each element is a human-readable string describing one violation.
        Both lists empty means the roadmap is clean.
    """
    schema_errors: list[str] = []
    prior_violations: list[str] = []

    if not roadmap_path.exists():
        schema_errors.append(f"File not found: {roadmap_path}")
        return schema_errors, prior_violations

    try:
        with open(roadmap_path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        schema_errors.append(f"YAML parse error: {e}")
        return schema_errors, prior_violations

    if not isinstance(raw, dict):
        schema_errors.append("Top-level YAML value must be a mapping")
        return schema_errors, prior_violations

    # Schema validation via Pydantic
    try:
        roadmap = Roadmap.model_validate(raw)
    except ValidationError as e:
        for err in e.errors():
            loc = " -> ".join(str(l) for l in err["loc"])
            schema_errors.append(f"Schema error at {loc}: {err['msg']}")
        # Even with schema errors, try to check prior_failures on individual tasks
        tasks_raw: list[Any] = raw.get("tasks", [])
        validated_tasks: list[ResearchTask] = []
        for t in tasks_raw:
            if not isinstance(t, dict):
                continue
            try:
                validated_tasks.append(ResearchTask.model_validate(t))
            except ValidationError:
                pass
    else:
        validated_tasks = roadmap.tasks

    # Prior-failures completeness check
    failure_index = _build_failure_index(complete_path)
    for task in validated_tasks:
        matches = _scope_matches(task.title, failure_index)
        if matches and not task.prior_failures:
            prior_violations.append(
                f"Task {task.id!r} ({task.title!r}) matches archived experiments "
                f"{matches[:5]} but has no prior_failures field — "
                f"add prior_failures or confirm this is a genuinely new scope"
            )

    return schema_errors, prior_violations


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: validate_prior_failures.py <roadmap.yaml>", file=sys.stderr)
        return 1

    roadmap_path = Path(sys.argv[1])
    schema_errors, prior_violations = validate_roadmap(roadmap_path)

    ok = True

    if schema_errors:
        ok = False
        print(f"\n[SCHEMA ERRORS] {roadmap_path}")
        for e in schema_errors:
            print(f"  ERROR: {e}")

    if prior_violations:
        ok = False
        print(f"\n[PRIOR FAILURE VIOLATIONS] {roadmap_path}")
        for v in prior_violations:
            print(f"  VIOLATION: {v}")

    if ok:
        print(f"[OK] {roadmap_path} — no schema errors, no prior_failures violations")
        return 0
    else:
        total = len(schema_errors) + len(prior_violations)
        print(f"\nTotal violations: {total}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
