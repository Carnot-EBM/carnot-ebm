#!/usr/bin/env python3
"""Refuse a consumer that reads an eval-run field nothing emits (REQ-ARC-WMTE-6642).

WHY THIS EXISTS (2026-09-04). Three times in three days a consumer of
`results/arc_leaderboard_eval_runs/` artifacts ran, failed, and only then did
anyone learn the field it needed was never recorded: `trajectory_supervisor`
(a 13.8h run the refinement ledger could not read), `generator_channels`
(11 artifacts that could not answer the n_ctx question), and
`induction_attempts` round provenance (exp6968 blocked with zero engine
candidates). The runtime object HAD the data each time. The live path recorded
nothing, and the gap surfaced only when a consumer came back empty — after the
GPU-hours were spent.

WHAT THIS CHECK IS. A mechanical join, no model involved:

1. Every tracked .py file under scripts/ or python/ that names the eval-runs
   directory is a CONSUMER. A consumer MUST declare a module-level
   `EVAL_RUN_FIELDS_READ` tuple naming the fields it requires eval-run
   artifacts to contain. A new consumer without the declaration FAILS — that
   is the moment the contract is cheap to state, not months later.
2. Every declared field must appear as a dict key somewhere in some artifact
   in the runs directory (`observed`), or at least as a string literal in the
   producer surface (`wired_unobserved` — wired but no run has landed since).
   A field found in NEITHER place fails: that is a consumer written against a
   field nobody emits, which is the incident class verbatim.
WHAT IT DELIBERATELY DOES NOT DO. It does not auto-extract every `.get()` key
from consumer source: consumers read many dicts that are not eval-run rows,
and the noise would train people to bypass the check. The declaration is the
scoping filter; its price, stated plainly, is that an UNDECLARED read is
invisible here. The declaration-presence rule bounds that gap to fields, not
to whole consumers.

Fail direction: FAIL CLOSED on a missing runs directory or an unparseable
consumer — a check that returns clean when it could not look is the state
this project's guard incidents keep re-teaching (see the QA-Layer discipline).
Population counts are always printed so "0 findings" is distinguishable from
"never looked" — the guarded-hasattr no-op of 2026-09-04 is the incident.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "arc_leaderboard_eval_runs"
FLAT_EVAL = REPO_ROOT / "results" / "arc_leaderboard_eval.json"
DECLARATION_NAME = "EVAL_RUN_FIELDS_READ"
DIR_MARKER = "arc_leaderboard_eval_runs"

# The producer surface: the eval harness itself plus the agentic runtime whose
# objects the harness serializes into rows. A directory, not a hand list of
# files, so a new row-field helper is covered without editing this lint
# (a hand list narrower than the concept is the class-B guard failure).
PRODUCER_SURFACE = (
    REPO_ROOT / "scripts" / "arc_leaderboard_eval.py",
    REPO_ROOT / "python" / "carnot" / "agentic",
)

# Not consumers for this contract, with reasons:
# - the producer emits the fields rather than requiring them;
# - this lint names the directory in its own source;
# - tests construct their own fixture rows, so key-containment over real
#   artifacts says nothing about what a test needs.
EXCLUDED_CONSUMERS = frozenset(
    {
        "scripts/arc_leaderboard_eval.py",
        "scripts/eval_run_consumer_field_lint.py",
    }
)


def find_consumers(repo_root: Path) -> list[Path]:
    """Tracked .py files that name the runs directory and are not excluded."""
    out: list[Path] = []
    for base in ("scripts", "python"):
        root = repo_root / base
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(repo_root).as_posix()
            if rel in EXCLUDED_CONSUMERS or rel.startswith("tests/"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if DIR_MARKER in text:
                out.append(path)
    return out


def declared_fields(path: Path) -> list[str] | None:
    """The module-level declaration. None when absent — the caller treats that
    as a failure, not as an empty declaration."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    fields: list[str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == DECLARATION_NAME for t in node.targets
        ):
            value = node.value
            if isinstance(value, (ast.Tuple, ast.List)):
                fields = [
                    e.value
                    for e in value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
    return fields


def artifact_keys(runs_dir: Path, extra_files: tuple[Path, ...] = ()) -> tuple[set[str], int]:
    """Every dict key, at any depth, across every artifact. Plus the file count."""
    keys: set[str] = set()
    n = 0

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                keys.add(str(k))
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    files = sorted(runs_dir.glob("*.json")) if runs_dir.is_dir() else []
    files += [p for p in extra_files if p.is_file()]
    for path in files:
        try:
            walk(json.loads(path.read_text(encoding="utf-8", errors="replace")))
            n += 1
        except (OSError, json.JSONDecodeError):
            continue
    return keys, n


def producer_literals(surface: tuple[Path, ...]) -> tuple[set[str], int]:
    """Every string literal in the producer surface. Plus the file count."""
    literals: set[str] = set()
    files: list[Path] = []
    for entry in surface:
        if entry.is_dir():
            files.extend(sorted(entry.rglob("*.py")))
        elif entry.is_file():
            files.append(entry)
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
    return literals, len(files)


def run_lint(
    repo_root: Path = REPO_ROOT,
    runs_dir: Path = RUNS_DIR,
    flat_eval: Path = FLAT_EVAL,
    producer_surface: tuple[Path, ...] = PRODUCER_SURFACE,
) -> tuple[list[str], list[str]]:
    """Returns (failures, notices). Empty failures means the contract holds."""
    failures: list[str] = []
    notices: list[str] = []
    consumers = find_consumers(repo_root)
    if not runs_dir.is_dir():
        # Fail closed: with no artifacts the join cannot run, and a clean exit
        # here would be indistinguishable from a real pass.
        failures.append(f"runs directory missing, join cannot run: {runs_dir}")
        return failures, notices
    keys, n_artifacts = artifact_keys(runs_dir, (flat_eval,))
    emitted, n_producer_files = producer_literals(producer_surface)
    notices.append(
        f"population: {len(consumers)} consumer(s), {n_artifacts} artifact(s), "
        f"{len(keys)} distinct artifact keys, {n_producer_files} producer file(s)"
    )
    if n_artifacts == 0:
        failures.append(f"no readable artifacts under {runs_dir}; join cannot run")
        return failures, notices
    for consumer in consumers:
        rel = consumer.relative_to(repo_root).as_posix()
        try:
            fields = declared_fields(consumer)
        except SyntaxError as exc:
            failures.append(f"{rel}: unparseable ({exc})")
            continue
        if fields is None:
            failures.append(
                f"{rel}: reads {DIR_MARKER} but declares no {DECLARATION_NAME}. "
                f"Add a module-level tuple naming the fields it requires eval-run "
                f"artifacts to contain (REQ-ARC-WMTE-6642)."
            )
            continue
        for field in fields:
            if field in keys:
                continue
            if field in emitted:
                notices.append(
                    f"{rel}: {field!r} is wired in the producer surface but observed "
                    f"in no artifact yet — expected only between wiring and the next run."
                )
                continue
            failures.append(
                f"{rel}: requires field {field!r}, which appears in NO eval-run "
                f"artifact and NO producer source. A run cannot satisfy this "
                f"consumer; wire the field before the GPU-hours are spent "
                f"(REQ-ARC-WMTE-6642)."
            )
    return failures, notices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    runs_dir = args.runs_dir or (repo_root / "results" / "arc_leaderboard_eval_runs")
    flat = repo_root / "results" / "arc_leaderboard_eval.json"
    surface = (
        repo_root / "scripts" / "arc_leaderboard_eval.py",
        repo_root / "python" / "carnot" / "agentic",
    )
    failures, notices = run_lint(repo_root, runs_dir, flat, surface)
    for line in notices:
        print(f"NOTE: {line}")
    for line in failures:
        print(f"FAIL: {line}")
    if failures:
        return 1
    print("OK: every declared eval-run consumer field is emitted somewhere real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
