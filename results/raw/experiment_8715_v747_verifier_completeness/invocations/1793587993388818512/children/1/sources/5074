"""Standalone observability timing detector repair for Exp 4161.

Spec refs: REQ-REPORT-4161, SCENARIO-REPORT-4161-FALLBACK,
SCENARIO-REPORT-4161-ROOT-CAUSE.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any


SCHEMA = "carnot.observability_timing_detector_fix.v1"
OUTPUT_REL_PATH = Path("results/experiment_4161_observability_timing_detector_fix.json")
KNOWN_GOOD_MILESTONE = "2026.06.384"
SOURCE_GIT_ATTRIBUTION = "git_added_terminal_artifacts"
SOURCE_CHANGELOG_FALLBACK = "ops_changelog_window_fallback"
STANDALONE_MODULE_PATH = "python/carnot/reporting/observability_timing_detector_4161.py"
FALSE_ZERO_ROOT_CAUSE = (
    "commit-message predicate mismatch: scripts/research_conductor.py "
    "_run_operational_retrospective still appends timing rows only when the "
    "commit subject contains literal 'Exp ', but recent conductor commits use "
    "task-title subjects such as '[conductor] ACCUMULATE pass 1 ...' and record "
    "experiment identity through added results/experiment_<digits>_*.json "
    "artifact paths; the activation-bounded git range contains terminal "
    "artifacts, but the subject predicate skips every per-experiment commit."
)
REQUIRED_FIELDS = (
    "honest_verdict",
    "false_zero_root_cause",
    "fix_applied",
    "fallback_added",
)

_TERMINAL_ARTIFACT_RE = re.compile(r"results/experiment_\d+_[A-Za-z0-9_.+/-]+\.json")


class DetectorPreconditionError(RuntimeError):
    """Raised when the required source artifacts cannot be read."""


@dataclass(frozen=True)
class TimingDetection:
    """Milestone-scoped detector evidence."""

    milestone: str
    git_range: str
    source: str
    artifact_paths: tuple[str, ...]
    git_terminal_count: int
    changelog_terminal_count: int
    fallback_used: bool

    @property
    def experiment_count(self) -> int:
        return len(self.artifact_paths)

    def as_dict(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone,
            "git_range": self.git_range,
            "source": self.source,
            "experiment_count": self.experiment_count,
            "git_terminal_count": self.git_terminal_count,
            "changelog_terminal_count": self.changelog_terminal_count,
            "fallback_used": self.fallback_used,
            "artifact_paths": list(self.artifact_paths),
        }


RunGit = Callable[[list[str], Path], str]


def _run_git(args: list[str], cwd: Path) -> str:  # pragma: no cover - thin subprocess wrapper
    completed = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout


def _dedupe(paths: list[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for path in paths:
        seen.setdefault(path, None)
    return tuple(seen)


def is_terminal_experiment_artifact(path: str) -> bool:
    normalized = path.strip()
    return bool(_TERMINAL_ARTIFACT_RE.fullmatch(normalized)) and not normalized.endswith(
        "_state.json"
    )


def legacy_retro_subject_matches(subject: str) -> bool:
    """Return true when the conductor's legacy timing predicate would count it."""

    return "Exp " in subject


def extract_git_terminal_artifacts(git_log_text: str) -> tuple[str, ...]:
    paths: list[str] = []
    for line in git_log_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("COMMIT\t"):
            continue
        if "\t" in stripped:
            status, path = stripped.split("\t", 1)
            if status != "A":
                continue
        else:
            path = stripped
        if is_terminal_experiment_artifact(path):
            paths.append(path)
    return _dedupe(paths)


def _milestone_suffix(milestone: str) -> str:
    return milestone.rsplit(".", 1)[1]


def _previous_suffix(milestone: str) -> str:
    return f"{int(_milestone_suffix(milestone)) - 1:03d}"


def _activation_text(milestone: str) -> str:
    return f"Archive .{_previous_suffix(milestone)} -> activate .{_milestone_suffix(milestone)}"


def _retro_line_for_milestone(line: str, milestone: str) -> bool:
    suffix = _milestone_suffix(milestone)
    return "Operational retrospective" in line and (
        milestone in line or f".{suffix}" in line
    )


def extract_changelog_window_artifacts(changelog_text: str, milestone: str) -> tuple[str, ...]:
    """Extract terminal artifacts after activation and before the retro line."""

    active = False
    selected_lines: list[str] = []
    for line in changelog_text.splitlines():
        if not active and _activation_text(milestone) in line:
            active = True
            continue
        if active and _retro_line_for_milestone(line, milestone):
            break
        if active:
            selected_lines.append(line)

    paths: list[str] = []
    for line in selected_lines:
        for match in _TERMINAL_ARTIFACT_RE.findall(line):
            if is_terminal_experiment_artifact(match):
                paths.append(match)
    return _dedupe(paths)


def detect_from_sources(
    milestone: str,
    *,
    git_log_text: str,
    changelog_text: str,
    git_range: str,
) -> TimingDetection:
    git_paths = extract_git_terminal_artifacts(git_log_text)
    if git_paths:
        return TimingDetection(
            milestone=milestone,
            git_range=git_range,
            source=SOURCE_GIT_ATTRIBUTION,
            artifact_paths=git_paths,
            git_terminal_count=len(git_paths),
            changelog_terminal_count=0,
            fallback_used=False,
        )

    changelog_paths = extract_changelog_window_artifacts(changelog_text, milestone)
    return TimingDetection(
        milestone=milestone,
        git_range=git_range,
        source=SOURCE_CHANGELOG_FALLBACK,
        artifact_paths=changelog_paths,
        git_terminal_count=0,
        changelog_terminal_count=len(changelog_paths),
        fallback_used=True,
    )


def next_milestone(milestone: str) -> str:
    prefix, suffix = milestone.rsplit(".", 1)
    return f"{prefix}.{int(suffix) + 1:03d}"


def activation_commit(root: Path, milestone: str, *, run_git: RunGit = _run_git) -> str | None:
    output = run_git(
        [
            "git",
            "log",
            "--format=%H%x09%aI%x09%s",
            f"--grep=\\[conductor\\] Activate milestone {milestone}",
            "-n",
            "1",
        ],
        root,
    ).strip()
    return output.split("\t", 1)[0] if output else None


def milestone_git_range(root: Path, milestone: str, *, run_git: RunGit = _run_git) -> str:
    start = activation_commit(root, milestone, run_git=run_git)
    if start is None:
        raise DetectorPreconditionError(f"blocked_activation_commit_{milestone}")
    end = activation_commit(root, next_milestone(milestone), run_git=run_git)
    return f"{start}..{end or 'HEAD'}"


def git_log_name_status(root: Path, git_range: str, *, run_git: RunGit = _run_git) -> str:
    return run_git(
        [
            "git",
            "log",
            "--format=COMMIT%x09%H%x09%aI%x09%s",
            "--name-status",
            "--diff-filter=A",
            git_range,
            "--",
            "results/experiment_*.json",
        ],
        root,
    )


def _duration_s(started_s: float | None) -> float:
    if started_s is None:
        return 0.0001
    return round(max(0.0001, time.perf_counter() - started_s), 6)


def terminal_verdict(detection: TimingDetection) -> str:
    if detection.experiment_count > 0:
        return "complete: observability_timing_detector_fallback_landed"
    return "blocked_no_terminal_experiments_found"


def build_payload_from_text(
    *,
    milestone: str,
    git_log_text: str,
    changelog_text: str,
    git_range: str,
    duration_s: float,
) -> dict[str, Any]:
    detection = detect_from_sources(
        milestone,
        git_log_text=git_log_text,
        changelog_text=changelog_text,
        git_range=git_range,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "experiment_4161_observability_timing_detector_fix",
        "milestone": "2026.06.385",
        "diagnosed_milestone": milestone,
        "honest_verdict": terminal_verdict(detection),
        "false_zero_root_cause": FALSE_ZERO_ROOT_CAUSE,
        "fix_applied": True,
        "fallback_added": True,
        "fallback_used": detection.fallback_used,
        "detector_source": detection.source,
        "experiment_count": detection.experiment_count,
        "git_experiment_count": detection.git_terminal_count,
        "fallback_experiment_count": detection.changelog_terminal_count,
        "artifact_paths": list(detection.artifact_paths),
        "git_range": git_range,
        "standalone_detector_module": STANDALONE_MODULE_PATH,
        "research_conductor_touched": False,
        "production_wiring_note": (
            "The standalone detector is fixed without touching scripts/research_conductor.py; "
            "the conductor-internal timing block remains the observed false-zero source."
        ),
        "duration_s": duration_s,
        "inference_substrate": "git_added_artifact_scan_with_changelog_window_fallback",
        "detection": detection.as_dict(),
    }
    validate_payload(payload)
    return payload


def _read_required_text(path: Path, blocked_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DetectorPreconditionError(f"blocked_{blocked_name}") from exc


def _assert_preconditions(root: Path) -> None:
    retro_paths = sorted((root / "results").glob("operational_retro_2026_06_38*.json"))
    if not retro_paths:
        raise DetectorPreconditionError("blocked_retro_artifacts")
    detector_path = root / STANDALONE_MODULE_PATH
    if not detector_path.exists():
        raise DetectorPreconditionError("blocked_timing_detector_script")


def blocked_payload(reason: str, *, duration_s: float) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "experiment_4161_observability_timing_detector_fix",
        "milestone": "2026.06.385",
        "diagnosed_milestone": KNOWN_GOOD_MILESTONE,
        "honest_verdict": reason,
        "false_zero_root_cause": "blocked before root-cause confirmation",
        "fix_applied": False,
        "fallback_added": False,
        "fallback_used": False,
        "detector_source": "",
        "experiment_count": 0,
        "git_experiment_count": 0,
        "fallback_experiment_count": 0,
        "artifact_paths": [],
        "git_range": "",
        "standalone_detector_module": STANDALONE_MODULE_PATH,
        "research_conductor_touched": False,
        "duration_s": duration_s,
        "inference_substrate": "blocked_precondition",
    }
    validate_payload(payload)
    return payload


def build_payload(
    root: Path,
    *,
    milestone: str = KNOWN_GOOD_MILESTONE,
    run_git: RunGit = _run_git,
    started_s: float | None = None,
) -> dict[str, Any]:
    start = time.perf_counter() if started_s is None else started_s
    try:
        _assert_preconditions(root)
        git_range = milestone_git_range(root, milestone, run_git=run_git)
        git_log_text = git_log_name_status(root, git_range, run_git=run_git)
        changelog_text = _read_required_text(root / "ops" / "changelog.md", "ops_changelog")
        return build_payload_from_text(
            milestone=milestone,
            git_log_text=git_log_text,
            changelog_text=changelog_text,
            git_range=git_range,
            duration_s=_duration_s(start),
        )
    except DetectorPreconditionError as exc:
        return blocked_payload(str(exc), duration_s=_duration_s(start))


def validate_payload(payload: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(payload["honest_verdict"]).startswith(("complete:", "blocked_")):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(payload["fix_applied"], bool):
        raise ValueError("fix_applied must be a bare bool")
    if not isinstance(payload["fallback_added"], bool):
        raise ValueError("fallback_added must be a bare bool")
    if not isinstance(payload["false_zero_root_cause"], str):
        raise ValueError("false_zero_root_cause must be a string")


def write_payload(root: Path, payload: Mapping[str, Any]) -> Path:
    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run(root: Path) -> Path:
    payload = build_payload(root)
    validate_payload(payload)
    return write_payload(root, payload)
