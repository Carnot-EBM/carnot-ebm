"""Autofill roadmap prior_failures entries from the local failure ledger.

This helper is intentionally standalone: the conductor can run it before
dispatch without changing scripts/research_conductor.py. It preserves the
roadmap text and inserts only missing prior_failures blocks for tasks that
have ledger matches and no non-empty prior_failures list already present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from failure_ledger import FailureLedger  # type: ignore[import-not-found]  # noqa: E402
from in_process_doc_reconcile import (  # type: ignore[import-not-found]  # noqa: E402
    _FAILED_TOKENS,
    _PARTIAL_TOKENS,
)

AUTOFILL_SCRIPT = "scripts/conductor_priors_autofill.py"
SUCCESS_VERDICTS = ("success", "complete", "ok")
FAILURE_VERDICTS = tuple(tok.lower() for tok in (*_PARTIAL_TOKENS, *_FAILED_TOKENS))
PRIOR_FIELD_ORDER = (
    "experiment_id",
    "verdict",
    "classification",
    "addressed_by",
    "retire_if_same_verdict",
)


@dataclass(frozen=True)
class AutofillSummary:
    """Counts emitted by an autofill pass."""

    tasks_scanned: int
    stubs_generated: int
    already_populated: int
    roadmap_path: Path


def default_roadmap_path(repo_root: Path | None = None) -> Path:
    """Return the default roadmap, preferring the next-roadmap handoff file."""
    root = repo_root or PROJECT_ROOT
    next_path = root / "research-roadmap-next.yaml"
    if next_path.exists():
        return next_path
    return root / "research-roadmap.yaml"


def classify_verdict(verdict: str) -> str:
    """Classify a ledger verdict as a true failure or successful upstream."""
    verdict_l = (verdict or "").lower()
    if any(token in verdict_l for token in FAILURE_VERDICTS):
        return "true_failure"
    return "successful_upstream"


def _entry_value(entry: object, field: str, default: str = "") -> Any:
    if isinstance(entry, dict):
        return entry.get(field, default)
    return getattr(entry, field, default)


def _prior_stub(prior: object) -> dict[str, object]:
    experiment_id = str(_entry_value(prior, "experiment_id", "unknown_prior"))
    verdict = str(_entry_value(prior, "verdict", "unknown"))
    classification = classify_verdict(verdict)
    if classification == "true_failure":
        addressed_by = (
            f"REVIEW NEEDED: Autofilled from {experiment_id}; confirm the prior root "
            "cause is addressed before dispatch."
        )
        retire_if_same_verdict = True
    else:
        addressed_by = (
            f"Autofilled by {AUTOFILL_SCRIPT}: {experiment_id} is treated as a "
            "successful upstream, not a failed rerun."
        )
        retire_if_same_verdict = False
    return {
        "experiment_id": experiment_id,
        "verdict": verdict,
        "classification": classification,
        "addressed_by": addressed_by,
        "retire_if_same_verdict": retire_if_same_verdict,
    }


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return json.dumps(str(value))


def _render_prior_failures(stubs: list[dict[str, object]], indent: str) -> list[str]:
    lines = [f"{indent}prior_failures:"]
    entry_indent = f"{indent}  "
    field_indent = f"{indent}    "
    for stub in stubs:
        first = True
        for field in PRIOR_FIELD_ORDER:
            line_indent = entry_indent if first else field_indent
            bullet = "- " if first else ""
            lines.append(f"{line_indent}{bullet}{field}: {_yaml_scalar(stub.get(field))}")
            first = False
    return lines


def _task_spans(lines: list[str]) -> list[tuple[int, int, str]]:
    starts: list[tuple[int, str]] = []
    task_indent: str | None = None
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)-\s+id\s*:", line)
        if not match:
            continue
        indent = match.group(1)
        if task_indent is None:
            task_indent = indent
        if indent == task_indent:
            starts.append((index, indent))
    return [
        (start, starts[pos + 1][0] if pos + 1 < len(starts) else len(lines), indent)
        for pos, (start, indent) in enumerate(starts)
    ]


def _find_child_field(
    lines: list[str], start: int, end: int, child_indent: str, field: str
) -> int | None:
    pattern = re.compile(rf"^{re.escape(child_indent)}{re.escape(field)}\s*:")
    for index in range(start + 1, end):
        if pattern.match(lines[index]):
            return index
    return None


def _apply_insertions(
    original_text: str, task_stubs: list[tuple[int, list[dict[str, object]]]]
) -> str:
    lines = original_text.splitlines()
    spans = _task_spans(lines)
    replacements: list[tuple[int, int, list[str]]] = []

    for task_index, stubs in task_stubs:
        if task_index >= len(spans):
            raise ValueError(f"could not locate roadmap text for task index {task_index}")
        start, end, task_indent = spans[task_index]
        child_indent = f"{task_indent}  "
        rendered = _render_prior_failures(stubs, child_indent)
        prior_line = _find_child_field(lines, start, end, child_indent, "prior_failures")
        if prior_line is not None:
            replacements.append((prior_line, prior_line + 1, rendered))
            continue
        prompt_line = _find_child_field(lines, start, end, child_indent, "prompt")
        insert_at = prompt_line if prompt_line is not None else end
        replacements.append((insert_at, insert_at, rendered))

    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = replacement

    new_text = "\n".join(lines)
    if original_text.endswith("\n"):
        new_text += "\n"
    return new_text


def autofill_roadmap(
    roadmap_path: Path | str | None = None,
    *,
    dry_run: bool = False,
    ledger: object | None = None,
) -> AutofillSummary:
    """Scan a roadmap and optionally write missing prior_failures stubs."""
    path = Path(roadmap_path) if roadmap_path is not None else default_roadmap_path()
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    tasks = data.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError(f"{path} has no list-valued tasks field")

    active_ledger = ledger or FailureLedger.load_from_artifacts(PROJECT_ROOT)
    task_stubs: list[tuple[int, list[dict[str, object]]]] = []
    already_populated = 0
    stubs_generated = 0

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        prior_failures = task.get("prior_failures")
        if isinstance(prior_failures, list) and prior_failures:
            already_populated += 1
            continue
        priors = active_ledger.matching_priors(task)  # type: ignore[attr-defined]
        if not priors:
            continue
        stubs = [_prior_stub(prior) for prior in priors]
        task_stubs.append((index, stubs))
        stubs_generated += len(stubs)

    if task_stubs and not dry_run:
        path.write_text(_apply_insertions(text, task_stubs), encoding="utf-8")

    return AutofillSummary(
        tasks_scanned=len(tasks),
        stubs_generated=stubs_generated,
        already_populated=already_populated,
        roadmap_path=path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Populate missing roadmap prior_failures entries from FailureLedger."
    )
    parser.add_argument(
        "roadmap",
        nargs="?",
        type=Path,
        default=None,
        help="Roadmap YAML path. Defaults to research-roadmap-next.yaml, then research-roadmap.yaml.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = autofill_roadmap(args.roadmap, dry_run=args.dry_run)
    print(
        f"{summary.tasks_scanned} tasks scanned, {summary.stubs_generated} stubs generated, "
        f"{summary.already_populated} already populated"
    )
    if args.dry_run:
        print(f"dry-run: no file changes written to {summary.roadmap_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
