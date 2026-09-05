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

FAIL DIRECTION (revised 2026-09-05, see the worktree incident below).
FAIL CLOSED on anything that means the lint could not look: no `scripts/` or
`python/` tree under the repo root, an empty producer surface, an unparseable
consumer, or an explicit `--runs-dir` that is missing or holds no artifact.

The artifact corpus is different. It is untracked evidence that exists once
per machine, in the main checkout. A git worktree or a fresh clone has none.
When no `--runs-dir` is given, the corpus is resolved in this order:

1. this checkout's `results/arc_leaderboard_eval_runs/`;
2. the main checkout's copy, found through `git rev-parse --git-common-dir`;
3. none: the artifact half of the join is SKIPPED. The skip is printed on its
   own NOTE line and again on the final OK line.

Skipping the artifact half is NOT fail-open. An artifact key can only turn a
failure into a pass (`field in keys` is the first branch). With the corpus
absent, every declared field must be wired in producer source, so the failures
are a superset of what the full join reports
(`test_missing_corpus_never_admits_more`). The skip costs precision, never
safety: a field observed in a real artifact but absent from producer source
FAILS without the corpus. That is the loud direction.

WHY THE REVISION. Before it, every commit that touched `scripts/*.py` from a
git worktree was refused with "runs directory missing". The corpus is
gitignored, so a worktree has none. Every agent this project spawns works in
a worktree. A refused commit leaves the work staged in a shared index, where
the next `git add -A` sweeps it under an unrelated message. That cost work
more than once on 2026-09-05.

Population counts are always printed so "0 findings" is distinguishable from
"never looked" — the guarded-hasattr no-op of 2026-09-04 is the incident.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR_PARTS = ("results", "arc_leaderboard_eval_runs")
RUNS_DIR = REPO_ROOT.joinpath(*RUNS_DIR_PARTS)
#: Prefix of the notice that says the artifact half did not run. `main()` keys
#: the final OK line on it, so a skip is never visible only in the middle of the output.
SKIP_MARKER = "SKIPPED the artifact half of the join"
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

    files = _artifact_files(runs_dir) + [p for p in extra_files if p.is_file()]
    for path in files:
        try:
            walk(json.loads(path.read_text(encoding="utf-8", errors="replace")))
            n += 1
        except (OSError, json.JSONDecodeError):
            continue
    return keys, n


def _artifact_files(runs_dir: Path) -> list[Path]:
    """Candidate artifact files under a runs directory; empty when it is not one.

    The REQ-ARC-WMTE-7010 heartbeat (`*.progress.json`) shares this directory and is
    not a record: a key that exists only there must not satisfy the "observed" join.
    """
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.glob("*.json") if not p.name.endswith(".progress.json"))


def main_checkout_root(repo_root: Path) -> Path | None:
    """The main working tree of the repository that contains `repo_root`, or None.

    A git worktree shares one `.git` directory with the main checkout. Git calls
    it the common dir, and its parent is the main working tree in the normal
    layout. None when git cannot answer, when the layout is not the normal one
    (a bare repository has no working tree), or when `repo_root` already is the
    main checkout.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    common = Path(proc.stdout.strip())
    if not common.is_absolute():
        common = Path(repo_root) / common
    common = common.resolve()
    if common.name != ".git":
        return None
    main_root = common.parent
    if main_root == Path(repo_root).resolve():
        return None
    return main_root


def resolve_runs_dir(repo_root: Path) -> tuple[Path | None, str]:
    """Find the eval-run corpus for this checkout. Returns (directory, how).

    The directory is None when no corpus exists anywhere the lint knows to look.
    Order: this checkout, then the main checkout (a worktree or a second clone
    has no corpus of its own; the corpus is per-machine evidence), then none.
    A directory counts only when it holds at least one artifact file.
    """
    local = Path(repo_root).joinpath(*RUNS_DIR_PARTS)
    if _artifact_files(local):
        return local, "this checkout"
    main_root = main_checkout_root(repo_root)
    if main_root is not None:
        shared = main_root.joinpath(*RUNS_DIR_PARTS)
        if _artifact_files(shared):
            return shared, (
                f"the main checkout at {main_root}, because this checkout has no corpus at {local}"
            )
    where = f"absent from this checkout ({local})"
    if main_root is not None:
        where += f" and from the main checkout ({main_root})"
    else:
        where += ", and git names no main checkout to fall back to"
    return None, where


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
    runs_dir: Path | None = None,
    flat_eval: Path | None = None,
    producer_surface: tuple[Path, ...] | None = None,
) -> tuple[list[str], list[str]]:
    """Returns (failures, notices). Empty failures means the contract holds.

    `runs_dir=None` (the hook's case) resolves the corpus per `resolve_runs_dir`
    and skips the artifact half, loudly, when there is none. An explicit
    `runs_dir` is a claim about where the corpus is: if it is missing or empty
    the lint FAILS, because the caller cannot have meant "no corpus".
    """
    repo_root = Path(repo_root).resolve()
    if flat_eval is None:
        flat_eval = repo_root / "results" / "arc_leaderboard_eval.json"
    if producer_surface is None:
        producer_surface = (
            repo_root / "scripts" / "arc_leaderboard_eval.py",
            repo_root / "python" / "carnot" / "agentic",
        )
    failures: list[str] = []
    notices: list[str] = []
    # FAIL CLOSED: no consumer tree means nothing was scanned, which is not a pass.
    # Without this, a wrong --repo-root would find zero consumers and print OK.
    if not any((repo_root / base).is_dir() for base in ("scripts", "python")):
        failures.append(
            f"neither {repo_root / 'scripts'} nor {repo_root / 'python'} exists; "
            f"no consumer was scanned, which is not a pass"
        )
        return failures, notices
    consumers = find_consumers(repo_root)
    if runs_dir is not None:
        runs_dir = Path(runs_dir)
        if not runs_dir.is_dir():
            # FAIL CLOSED: the caller named a corpus that is not there.
            failures.append(f"runs directory missing, join cannot run: {runs_dir}")
            return failures, notices
        runs_source = "the --runs-dir argument"
    else:
        runs_dir, runs_source = resolve_runs_dir(repo_root)
    artifact_half = runs_dir is not None
    if artifact_half:
        keys, n_artifacts = artifact_keys(runs_dir, (flat_eval,))
        notices.append(f"runs directory: {runs_dir} ({runs_source})")
    else:
        # SKIP, LOUDLY. Not fail-open: with no keys, every field must be wired in
        # producer source, so nothing passes here that the full join would refuse.
        keys, n_artifacts = set(), 0
        notices.append(
            f"{SKIP_MARKER}: the runs directory is {runs_source}. Every declared "
            f"field is checked against producer source only; a field the corpus "
            f"would have shown as observed can only FAIL here, never pass. "
            f"Pass --runs-dir to supply a corpus."
        )
    emitted, n_producer_files = producer_literals(producer_surface)
    notices.append(
        f"population: {len(consumers)} consumer(s), {n_artifacts} artifact(s), "
        f"{len(keys)} distinct artifact keys, {n_producer_files} producer file(s)"
    )
    # FAIL CLOSED: an empty producer surface means the lint looked in the wrong place.
    if n_producer_files == 0:
        surface = ", ".join(str(p) for p in producer_surface)
        failures.append(
            f"producer surface empty ({surface}); the lint could not look, which is not a pass"
        )
        return failures, notices
    if artifact_half and n_artifacts == 0:
        failures.append(f"no readable artifacts under {runs_dir}; join cannot run")
        return failures, notices
    wired_only = 0
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
                if artifact_half:
                    notices.append(
                        f"{rel}: {field!r} is wired in the producer surface but observed "
                        f"in no artifact yet — expected only between wiring and the next run."
                    )
                else:
                    # In skip mode every field lands here; one summary line, not
                    # twenty "expected between wiring and the next run" lines that
                    # would describe a corpus nobody looked at.
                    wired_only += 1
                continue
            failures.append(
                f"{rel}: requires field {field!r}, which appears in NO eval-run "
                f"artifact and NO producer source. A run cannot satisfy this "
                f"consumer; wire the field before the GPU-hours are spent "
                f"(REQ-ARC-WMTE-6642)."
            )
    if not artifact_half:
        notices.append(f"{wired_only} declared field(s) verified against producer source only")
    return failures, notices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Corpus to join against. Default: this checkout's, else the main "
            "checkout's, else the artifact half is skipped and the output says so."
        ),
    )
    args = parser.parse_args(argv)
    failures, notices = run_lint(args.repo_root.resolve(), args.runs_dir)
    for line in notices:
        print(f"NOTE: {line}")
    for line in failures:
        print(f"FAIL: {line}")
    if failures:
        return 1
    if any(n.startswith(SKIP_MARKER) for n in notices):
        print(
            "OK (producer source only): every declared eval-run consumer field is "
            "wired in producer source; the artifact half of the join was SKIPPED, "
            "see the NOTE above."
        )
    else:
        print("OK: every declared eval-run consumer field is emitted somewhere real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
