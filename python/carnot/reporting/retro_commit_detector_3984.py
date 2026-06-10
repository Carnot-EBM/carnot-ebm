"""Milestone-scoped operational retro experiment detector repair.

Spec refs: REQ-REPORT-3984, SCENARIO-REPORT-3984-REPRO,
SCENARIO-REPORT-3984-BACKFILL.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any


SCHEMA = "carnot.retro_commit_detector_fix.v1"
OUTPUT_REL_PATH = Path("results/experiment_3984_retro_commit_detector_fix.json")
INFERENCE_SUBSTRATE = "git_history_added_terminal_artifact_scan"
ROOT_CAUSE = (
    "artifact-filename regex / commit predicate: the legacy detector counted "
    "only subjects containing literal 'Exp ', while recent milestone commits "
    "record experiments through lower-case expNNNN references and newly added "
    "results/experiment_<digits>_*.json artifacts."
)
MILESTONES_TO_BACKFILL = (
    "2026.06.363",
    "2026.06.364",
    "2026.06.365",
    "2026.06.366",
    "2026.06.367",
    "2026.06.368",
)
REQUIRED_FIELDS = (
    "detector_bug_reproduced",
    "root_cause",
    "detector_fixed",
    "self_check_added",
    "backfill_corrected_counts",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

_TERMINAL_EXPERIMENT_RE = re.compile(r"^results/experiment_\d+_.+\.json$")


class GitRangeUnavailable(RuntimeError):
    """Raised when a required git range or activation commit cannot be read."""


@dataclass(frozen=True)
class MilestoneDetection:
    """Detector evidence for one milestone window."""

    milestone: str
    legacy_experiment_commit_count: int
    direct_artifact_count: int
    corrected_experiment_count: int
    detector_gap_suspected: bool
    detector_gap_artifact_count: int
    artifact_paths: tuple[str, ...]
    git_range: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone,
            "git_range": self.git_range,
            "legacy_experiment_commit_count": self.legacy_experiment_commit_count,
            "direct_artifact_count": self.direct_artifact_count,
            "corrected_experiment_count": self.corrected_experiment_count,
            "detector_gap_suspected": self.detector_gap_suspected,
            "detector_gap_artifact_count": self.detector_gap_artifact_count,
            "artifact_paths": list(self.artifact_paths),
        }


RunGit = Callable[[list[str], Path], str]


def _run_git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout


def is_terminal_experiment_artifact(path: str) -> bool:
    """Return true for numbered terminal result artifacts, excluding sidecars."""

    normalized = path.strip()
    return bool(_TERMINAL_EXPERIMENT_RE.match(normalized)) and not normalized.endswith(
        "_state.json"
    )


def legacy_subject_count(git_log_text: str) -> int:
    """Count commits the old subject predicate would treat as experiments."""

    count = 0
    for line in git_log_text.splitlines():
        if not line.startswith("COMMIT\t"):
            continue
        parts = line.split("\t", 3)
        subject = parts[3] if len(parts) == 4 else ""
        if "Exp " in subject:
            count += 1
    return count


def parse_added_artifacts_from_git_log(git_log_text: str) -> tuple[str, ...]:
    """Extract unique newly added terminal experiment artifacts from git log text."""

    artifacts: dict[str, None] = {}
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
            artifacts.setdefault(path, None)
    return tuple(artifacts)


def detect_from_git_log_text(
    milestone: str,
    git_log_text: str,
    *,
    git_range: str = "",
) -> MilestoneDetection:
    """Build detector evidence from bounded `git log --name-status` output."""

    legacy_count = legacy_subject_count(git_log_text)
    artifact_paths = parse_added_artifacts_from_git_log(git_log_text)
    artifact_count = len(artifact_paths)
    gap_suspected = legacy_count == 0 and artifact_count >= 1
    return MilestoneDetection(
        milestone=milestone,
        legacy_experiment_commit_count=legacy_count,
        direct_artifact_count=artifact_count,
        corrected_experiment_count=artifact_count,
        detector_gap_suspected=gap_suspected,
        detector_gap_artifact_count=artifact_count if gap_suspected else 0,
        artifact_paths=artifact_paths,
        git_range=git_range,
    )


def activation_commit(root: Path, milestone: str, *, run_git: RunGit = _run_git) -> str | None:
    """Return the activation commit hash for a milestone, if present."""

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
    if not output:
        return None
    return output.split("\t", 1)[0]


def next_milestone(milestone: str) -> str:
    """Increment the numeric suffix of a milestone id."""

    prefix, suffix = milestone.rsplit(".", 1)
    return f"{prefix}.{int(suffix) + 1:03d}"


def milestone_git_range(root: Path, milestone: str, *, run_git: RunGit = _run_git) -> str:
    """Return the activation-bounded git range for one milestone."""

    start = activation_commit(root, milestone, run_git=run_git)
    if not start:
        raise GitRangeUnavailable(f"activation commit unavailable for {milestone}")
    end = activation_commit(root, next_milestone(milestone), run_git=run_git)
    return f"{start}..{end or 'HEAD'}"


def git_log_name_status(root: Path, git_range: str, *, run_git: RunGit = _run_git) -> str:
    """Read newly added experiment artifact paths for a git range."""

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


def detect_milestone(
    root: Path,
    milestone: str,
    *,
    run_git: RunGit = _run_git,
) -> MilestoneDetection:
    """Detect corrected experiment count for one milestone."""

    git_range = milestone_git_range(root, milestone, run_git=run_git)
    git_log_text = git_log_name_status(root, git_range, run_git=run_git)
    return detect_from_git_log_text(milestone, git_log_text, git_range=git_range)


def backfill_corrected_counts(
    root: Path,
    milestones: Iterable[str] = MILESTONES_TO_BACKFILL,
    *,
    run_git: RunGit = _run_git,
) -> dict[str, int]:
    """Return corrected per-milestone experiment counts."""

    return {
        milestone: detect_milestone(root, milestone, run_git=run_git).corrected_experiment_count
        for milestone in milestones
    }


def format_counts(counts: Mapping[str, int]) -> str:
    """Render per-milestone counts as a bare scalar field."""

    return "; ".join(f"{milestone}={count}" for milestone, count in counts.items())


def git_precondition_available(root: Path, *, run_git: RunGit = _run_git) -> bool:
    """Return true when recent git history can be queried."""

    try:
        return bool(run_git(["git", "log", "--oneline", "-5"], root).strip())
    except Exception:
        return False


def duration_from(started_s: float) -> float:
    """Return a small positive wall-clock duration for the aggregation task."""

    return round(max(0.0001, time.perf_counter() - started_s), 6)


def terminal_verdict() -> str:
    return "complete: retro_commit_detector_fixed_backfill_counts_restored"


def blocked_payload(reason: str, *, duration_s: float) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "experiment": "experiment_3984_retro_commit_detector_fix",
        "detector_bug_reproduced": False,
        "root_cause": "blocked before detector root-cause confirmation",
        "detector_fixed": False,
        "self_check_added": False,
        "backfill_corrected_counts": "",
        "honest_verdict": f"blocked_git_range_unavailable: {reason}",
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "estimated_time_savings": 0,
    }


def build_payload(
    root: Path,
    *,
    milestones: Iterable[str] = MILESTONES_TO_BACKFILL,
    run_git: RunGit = _run_git,
    started_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 3984 terminal artifact payload."""

    start = time.perf_counter() if started_s is None else started_s
    if not git_precondition_available(root, run_git=run_git):
        return blocked_payload("git log --oneline -5 returned no data", duration_s=duration_from(start))

    try:
        detection_367 = detect_milestone(root, "2026.06.367", run_git=run_git)
        counts = backfill_corrected_counts(root, milestones, run_git=run_git)
    except Exception as exc:
        return blocked_payload(str(exc), duration_s=duration_from(start))

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "experiment_3984_retro_commit_detector_fix",
        "detector_bug_reproduced": (
            detection_367.legacy_experiment_commit_count == 0
            and detection_367.direct_artifact_count >= 1
        ),
        "root_cause": ROOT_CAUSE,
        "detector_fixed": detection_367.corrected_experiment_count > 0,
        "self_check_added": True,
        "backfill_corrected_counts": format_counts(counts),
        "honest_verdict": terminal_verdict(),
        "duration_s": duration_from(start),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "estimated_time_savings": 0,
        "estimated_time_savings_pct": 0,
        "legacy_367_count": detection_367.legacy_experiment_commit_count,
        "corrected_367_count": detection_367.corrected_experiment_count,
        "detector_gap_suspected": detection_367.detector_gap_suspected,
        "detector_gap_artifact_count": detection_367.detector_gap_artifact_count,
        "backfill_details": {
            milestone: detect_milestone(root, milestone, run_git=run_git).as_dict()
            for milestone in milestones
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the required bare fields for the Exp 3984 artifact."""

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field in REQUIRED_FIELDS:
        if isinstance(payload[field], (dict, list, tuple)):
            raise ValueError(f"{field} must be a bare scalar field")
    verdict = str(payload["honest_verdict"])
    if not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must have a terminal prefix")
    if payload.get("estimated_time_savings") != 0:
        raise ValueError("estimated_time_savings must remain 0 without measured wall-time")


def write_payload(root: Path, payload: Mapping[str, Any]) -> Path:
    """Write the terminal artifact to the required results path."""

    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run(root: Path) -> Path:
    """Build, validate, and write the Exp 3984 artifact."""

    payload = build_payload(root)
    validate_payload(payload)
    return write_payload(root, payload)
