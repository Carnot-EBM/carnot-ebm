#!/usr/bin/env python3
"""Lint ARC roadmap preconditions for pytest commands missing ``--no-cov``.

Spec refs: REQ-REPORT-4482, SCENARIO-REPORT-4482-ROADMAP-LINT.

ARC solve launch checks are meant to be quick smoke tests. If a roadmap
precondition runs pytest without ``--no-cov``, repository coverage settings can
turn that smoke test into a long global coverage gate before the solve starts.
This lint catches that mistake while the roadmap is still easy to repair.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROADMAP = REPO_ROOT / "research-roadmap.yaml"
PYTEST_MISSING_NO_COV = "PYTEST_PRECONDITION_MISSING_NO_COV"


@dataclass(frozen=True)
class ArcNoCovPreconditionIssue:
    """One roadmap task precondition that would re-enable coverage gating."""

    path: Path
    task_id: str
    track: str
    line: int
    command: str
    kind: str
    detail: str
    severity: str = "error"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "path": str(self.path),
            "task_id": self.task_id,
            "track": self.track,
            "line": self.line,
            "command": self.command,
            "kind": self.kind,
            "detail": self.detail,
            "severity": self.severity,
        }


def lint_roadmap(path: Path | str) -> list[ArcNoCovPreconditionIssue]:
    """REQ-REPORT-4482: lint one roadmap YAML file for ARC no-cov violations."""

    roadmap_path = Path(path)
    text = roadmap_path.read_text(encoding="utf-8")
    return lint_roadmap_text(roadmap_path, text)


def lint_roadmap_text(path: Path | str, text: str) -> list[ArcNoCovPreconditionIssue]:
    """Lint already-loaded roadmap YAML text.

    The parser uses YAML rather than regular expressions so the check follows
    the task boundaries the conductor actually activates. The line numbers are
    prompt-local because roadmap prompts are YAML block strings whose physical
    file line positions are not preserved by ``safe_load``.
    """

    data = yaml.safe_load(text) or {}
    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    issues: list[ArcNoCovPreconditionIssue] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        track = str(task.get("track", "") or "")
        if not track.startswith("arc-"):
            continue
        task_id = str(task.get("id", "<no-id>") or "<no-id>")
        prompt = str(task.get("prompt", "") or "")
        for line_number, command in _precondition_pytest_commands(prompt):
            if _command_has_no_cov(command):
                continue
            issues.append(
                ArcNoCovPreconditionIssue(
                    path=Path(path),
                    task_id=task_id,
                    track=track,
                    line=line_number,
                    command=command,
                    kind=PYTEST_MISSING_NO_COV,
                    detail=(
                        "ARC roadmap PRECONDITIONS pytest command is missing --no-cov; "
                        "focused smoke gates must not inherit repository coverage fail-under."
                    ),
                )
            )
    return issues


def lint_roadmaps(paths: Iterable[Path | str]) -> list[ArcNoCovPreconditionIssue]:
    """Lint multiple roadmap YAML files and concatenate the issues."""

    issues: list[ArcNoCovPreconditionIssue] = []
    for path in paths:
        issues.extend(lint_roadmap(path))
    return issues


def install_research_conductor_activation_guard(conductor: ModuleType) -> ModuleType:
    """Attach the ARC no-cov activation guard to an imported conductor module."""

    conductor._arc_nocov_precondition_activation_guard = (  # type: ignore[attr-defined]
        _arc_nocov_precondition_activation_guard
    )
    activate = getattr(conductor, "_activate_next_roadmap", None)
    if callable(activate) and not getattr(activate, "_arc_nocov_guard_wrapped", False):

        @wraps(activate)
        def _activate_next_roadmap_with_arc_nocov_guard(*args: Any, **kwargs: Any) -> bool:
            roadmap = Path(getattr(conductor, "NEXT_ROADMAP_FILE", DEFAULT_ROADMAP))
            if roadmap.exists() and not _arc_nocov_precondition_activation_guard(
                roadmap,
                _roadmap_milestone(roadmap),
            ):
                return False
            return bool(activate(*args, **kwargs))

        _activate_next_roadmap_with_arc_nocov_guard._arc_nocov_guard_wrapped = (  # type: ignore[attr-defined]
            True
        )
        conductor._activate_next_roadmap = (  # type: ignore[attr-defined]
            _activate_next_roadmap_with_arc_nocov_guard
        )
    return conductor


def _roadmap_milestone(path: Path) -> str:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:  # pragma: no cover - the guard logs unreadable-roadmap failures.
        return "unknown"
    if not isinstance(data, dict):
        return "unknown"
    return str(data.get("milestone", "unknown") or "unknown")


def _arc_nocov_precondition_activation_guard(
    roadmap_path: Path | str,
    milestone: str,
) -> bool:
    """SCENARIO-REPORT-4482-ACTIVATION-GUARD: block bad ARC next roadmaps."""

    try:
        issues = lint_roadmap(roadmap_path)
    except Exception as exc:
        _log_activation_guard_block(
            milestone,
            f"ARC no-cov precondition lint failed while reading {roadmap_path}: {exc}",
        )
        return False

    if not issues:
        return True

    first = issues[0]
    _log_activation_guard_block(
        milestone,
        (
            f"ARC no-cov precondition lint: {len(issues)} issue(s); "
            f"first: {first.kind} on {first.task_id} prompt-line-{first.line}. "
            "NEXT_ROADMAP_FILE left in place for operator inspection."
        ),
    )
    return False


def _log_activation_guard_block(milestone: str, detail: str) -> None:
    conductor = sys.modules.get("scripts.research_conductor") or sys.modules.get(
        "research_conductor"
    )
    log_step = getattr(conductor, "log_step", None)
    if callable(log_step):
        log_step(f"Activation REFUSED: milestone {milestone}", "BLOCK", detail)


def main(argv: list[str] | None = None) -> int:
    """Run the roadmap no-cov lint CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roadmaps", nargs="*", help="Roadmap YAML paths to lint.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.roadmaps] or [DEFAULT_ROADMAP]
    issues = lint_roadmaps(paths)
    report: dict[str, Any] = {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": [issue.to_dict() for issue in issues],
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:  # pragma: no cover - JSON mode is the conductor-facing path.
        for issue in issues:
            print(
                f"{issue.path}:{issue.task_id}:prompt-line-{issue.line}: "
                f"{issue.kind}: {issue.detail}"
            )
    return 1 if issues else 0


def _precondition_pytest_commands(prompt: str) -> list[tuple[int, str]]:
    commands: list[tuple[int, str]] = []
    in_preconditions = False
    for line_number, line in enumerate(prompt.splitlines(), start=1):
        if _starts_later_numbered_step(line):
            in_preconditions = False
        if "preconditions" in line.lower():
            in_preconditions = True
        if not in_preconditions:
            continue
        command = _extract_pytest_command(line)
        if command:
            commands.append((line_number, command))
    return commands


def _starts_later_numbered_step(line: str) -> bool:
    match = re.match(r"^\s*(\d+)\.\s+", line)
    return bool(match and int(match.group(1)) > 0)


def _extract_pytest_command(line: str) -> str | None:
    for segment in re.findall(r"`([^`]*pytest[^`]*)`", line):
        if _is_pytest_command(_command_tokens(segment)):
            return segment.strip()
    cleaned = re.sub(r"^\s*(?:[-*]\s+|[a-zA-Z]\.\s+|\d+\.\s+)", "", line).strip()
    if _is_pytest_command(_command_tokens(cleaned)):
        return cleaned
    return None


def _command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _is_pytest_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    first = tokens[0]
    if Path(first).name == "pytest" or first.endswith("/pytest"):
        return True
    return (
        len(tokens) >= 3
        and Path(first).name.startswith("python")
        and tokens[1:3]
        == [
            "-m",
            "pytest",
        ]
    )


def _command_has_no_cov(command: str) -> bool:
    return "--no-cov" in _command_tokens(command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
