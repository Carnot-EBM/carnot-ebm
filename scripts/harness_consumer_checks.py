#!/usr/bin/env python3
"""Three deterministic checks for the defect classes that keep reaching production.

Spec: REQ-HARNESS-CONSUMER-1, SCENARIO-HARNESS-CONSUMER-1..4

Nineteen harness defects were filed in roughly thirteen hours on 2026-09-07/08. About eight
were catchable without an LLM, and each reduced to one of three questions:

  who READS this field?      `estimated_wall_time_min` has no consumer that affects execution,
                             yet acceptance criteria were written in terms of it for a day.
  does anything CALL this?   `audit_orphan_test_imports.py` was spec'd "Implemented", called by
                             nothing, and blind to the idiom it was named for.
  do these paths EXIST?      a milestone's prompts named five modules that do not exist.

No LLM call. Three adversarial-review layers already exist; a fourth would inspect more code
without answering any of these.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEARCH_ROOTS = ("scripts", "python")
#: The files that decide what actually happens at run time. A field read only OUTSIDE these is
#: read by analysis and ignored by the thing that runs -- which is what `estimated_wall_time_min`
#: turned out to be after a day of acceptance criteria were written in terms of it.
RUNTIME_FILES = ("scripts/research_conductor.py",)
RUNTIME_DIRS = ("python/carnot/agentic/",)
#: Known not to run in production. conductor_supervisor.py carries a header saying so.
DEAD_BY_DEFAULT = ("scripts/conductor_supervisor.py",)
GUARD_NAME_HINTS = ("lint", "guard", "audit", "check")
#: A path presented as EXISTING whose parent directory is missing is an invented location.
#: A path whose parent exists is an ordinary forward reference to what the task will write.
PROMPT_PATH_RE = re.compile(r"\{project_root\}/([A-Za-z0-9_./\-]+)")


def _py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for name in SEARCH_ROOTS:
        base = root / name
        if base.is_dir():
            out.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return out


def classify_reader(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Runtime, analysis, test or schema. Only a RUNTIME reader can change what happens."""

    rel = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
    if "/tests/" in f"/{rel}" or path.name.startswith("test_"):
        return "test"
    if path.name in ("roadmap_schema.py", "schema.py"):
        return "schema"
    if rel in RUNTIME_FILES or any(rel.startswith(d) for d in RUNTIME_DIRS):
        return "runtime"
    return "analysis"


def field_reads(tree: ast.AST, field: str) -> list[int]:
    """Line numbers where `field` is READ, distinguished from written, by AST context.

    A dict-literal key and an assignment target are WRITES. Counting them as readers is why a
    first version of this check reported sixteen production readers for a field that in fact has
    none: the sixteen were the planner writing it, a dead supervisor, and this file's own
    docstring.
    """

    reads: list[int] = []
    for node in ast.walk(tree):
        # d.get("field") / d.get("field", default)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "setdefault", "pop"):
                for a in node.args[:1]:
                    if isinstance(a, ast.Constant) and a.value == field:
                        reads.append(node.lineno)
        # d["field"] in a LOAD context is a read; a STORE target is a write
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            sl = node.slice
            if isinstance(sl, ast.Constant) and sl.value == field:
                reads.append(node.lineno)
        # obj.field
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if node.attr == field:
                reads.append(node.lineno)
    return sorted(set(reads))


def unread_field(field: str, root: Path = PROJECT_ROOT, dead: tuple[str, ...] = ()) -> dict:
    """Every READER of a field, split by kind. No production reader means nothing acts on it.

    `dead` names files known not to run in production. They are reported separately rather than
    silently dropped, because "the only reader is dead code" is the finding, not a detail.
    """

    dead = tuple(dead) or DEAD_BY_DEFAULT
    readers: dict[str, list[str]] = {
        "runtime": [],
        "analysis": [],
        "test": [],
        "schema": [],
        "dead": [],
    }
    for path in _py_files(root):
        if path.resolve() == Path(__file__).resolve():
            continue  # never let this file's own prose count as a consumer
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        lines = field_reads(tree, field)
        if not lines:
            continue
        rel = path.relative_to(root).as_posix()
        kind = "dead" if rel in dead else classify_reader(path, root)
        readers[kind].extend(f"{rel}:{n}" for n in lines)
    return {
        "field": field,
        "readers": readers,
        "has_runtime_reader": bool(readers["runtime"]),
    }


def spec_claimed_guards(root: Path = PROJECT_ROOT) -> set[str]:
    """Scripts a spec advertises as Implemented. Those are promises, not one-off audits."""

    claimed: set[str] = set()
    specs = root / "openspec"
    if not specs.is_dir():
        return claimed
    for spec in specs.rglob("spec.md"):
        try:
            text = spec.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "Implemented" not in line:
                continue
            for m in re.finditer(r"scripts/([A-Za-z0-9_]+)\.py", line):
                claimed.add(m.group(1))
    return claimed


def uncalled_guards(root: Path = PROJECT_ROOT) -> dict:
    """Guards a SPEC calls Implemented that nothing calls.

    Scoped to spec-claimed guards on purpose. Every guard-shaped script with no caller is 37 of
    136 -- 27 percent -- and most are one-off historical audits that were never meant to run
    again. Flagging those is the crying-wolf failure CLAUDE.md warns about. The defect worth
    catching is narrower and was real: `audit_orphan_test_imports.py` was advertised as
    Implemented in a spec table, called by nothing, and blind to the idiom it was named for.
    """

    scripts_dir = root / "scripts"
    claimed = spec_claimed_guards(root)
    guards = sorted(
        p
        for p in scripts_dir.glob("*.py")
        if p.stem in claimed
        and any(h in p.stem for h in GUARD_NAME_HINTS)
        and not p.stem.startswith("test_")
    )
    haystack: list[tuple[str, str]] = []
    precommit = root / ".pre-commit-config.yaml"
    if precommit.is_file():
        haystack.append((precommit.name, precommit.read_text(encoding="utf-8", errors="replace")))
    for path in _py_files(root):
        try:
            haystack.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    orphans = []
    for guard in guards:
        stem = guard.stem
        called = any(
            stem in text for name, text in haystack if name != guard.name and name != "test_" + stem
        )
        if not called:
            orphans.append(guard.relative_to(root).as_posix())
    return {"guards_scanned": len(guards), "uncalled": orphans}


def invented_prompt_paths(roadmap_text: str, root: Path = PROJECT_ROOT) -> list[str]:
    """Paths a prompt presents as existing whose PARENT DIRECTORY is absent.

    Keyed on the parent rather than the file: a missing file inside an existing directory is the
    task's own deliverable, and flagging those turns 19 real hits into 117.
    """

    bad: set[str] = set()
    for m in PROMPT_PATH_RE.finditer(roadmap_text):
        rel = m.group(1).rstrip(".,;")
        if "." not in Path(rel).name:
            continue
        candidate = root / rel
        if candidate.exists() or candidate.parent.is_dir():
            continue
        bad.add(rel)
    return sorted(bad)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("unread-field")
    f.add_argument("field")
    sub.add_parser("uncalled-guards")
    p = sub.add_parser("prompt-paths")
    p.add_argument("roadmap", nargs="?", default=str(PROJECT_ROOT / "research-roadmap.yaml"))
    args = parser.parse_args(argv)

    if args.cmd == "unread-field":
        r = unread_field(args.field)
        for kind, hits in r["readers"].items():
            print(f"  {kind:11s} {len(hits)}  {', '.join(hits[:4])}")
        if not r["has_runtime_reader"]:
            print(
                f"NO RUNTIME READER: {args.field} is declared and read only by analysis code. "
                "Nothing that decides execution consumes it."
            )
            return 1
        return 0
    if args.cmd == "uncalled-guards":
        r = uncalled_guards()
        print(f"  guards scanned: {r['guards_scanned']}, uncalled: {len(r['uncalled'])}")
        for g in r["uncalled"]:
            print(f"    UNCALLED  {g}")
        return 1 if r["uncalled"] else 0
    bad = invented_prompt_paths(Path(args.roadmap).read_text(encoding="utf-8", errors="replace"))
    for b in bad:
        print(f"    INVENTED PATH  {b}  (parent directory does not exist)")
    print(f"  invented paths: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
