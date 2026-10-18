"""Exp 4126 nano-trm LR resume correctness artifact helpers.

Spec refs: REQ-LEARN-4126, SCENARIO-LEARN-4126.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4126_lr_resume_correctness_fix.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_STABLE_CHECKPOINT = (
    REPO_ROOT / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
)
FRESH_WARMUP_FIRST_LR = 2.4500000108673703e-06
ROOT_CAUSE_LOCAL_MANUAL_STEP = (
    "local_manual_step_not_checkpointed_global_step_zero_legacy_checkpoint_restarted_warmup"
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "lr_rewarm_root_cause",
    "lr_continuous_across_resume",
    "validation_first_lr",
    "val_exact_accuracy",
    "stable_checkpoint_path",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'cannot fix bounded-resume -> contiguous run needed' "
        "is a COMPLETE verdict."
    ),
    "lr_rewarm_root_cause": (
        "The diagnosed reason the LR resets on resume (scheduler-not-restored / "
        "local-step-warmup / etc.); the audit trail for the fix."
    ),
    "lr_continuous_across_resume": (
        "Bare bool: after the fix, does a resumed pass CONTINUE the schedule "
        "(start-LR != fresh-warmup value)? THE gate -- the prerequisite for "
        "exp4127's accumulation to work."
    ),
    "validation_first_lr": (
        "The observed train/lr at the start of the validation pass; compared to "
        "the prior pass's last-LR proves continuation (not re-warm)."
    ),
    "val_exact_accuracy": (
        "Val after the validation pass; should improve faster than .381's +1pp/pass "
        "if the schedule now continues."
    ),
    "stable_checkpoint_path": "The path exp4127 resumes from with the corrected schedule.",
    "duration_s": "Bounded GPU run < 4800s (stopped before the cap).",
}


@dataclass(frozen=True)
class MetricPoint:
    """One non-empty scalar metric row from a Lightning CSV file."""

    epoch: int | None
    step: int | None
    value: float
    metrics_path: Path

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["metrics_path"] = str(self.metrics_path)
        return row


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _metrics_paths(run_dir_or_file: str | Path) -> list[Path]:
    path = Path(run_dir_or_file)
    if path.is_file():
        return [path]
    return sorted(path.rglob("metrics.csv"))


def extract_train_lr_points(run_dir_or_file: str | Path) -> list[MetricPoint]:
    """REQ-LEARN-4126: collect non-empty train/lr rows from Lightning CSV metrics."""

    points: list[MetricPoint] = []
    for metrics_path in _metrics_paths(run_dir_or_file):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = _float_or_none(row.get("train/lr"))
                if value is None:
                    continue
                points.append(
                    MetricPoint(
                        epoch=_int_or_none(row.get("epoch")),
                        step=_int_or_none(row.get("step")),
                        value=value,
                        metrics_path=metrics_path,
                    )
                )
    return points


def extract_latest_val_exact_accuracy(run_dir_or_file: str | Path) -> float | None:
    """REQ-LEARN-4126: read the last non-empty val/exact_accuracy from CSV metrics."""

    latest: float | None = None
    for metrics_path in _metrics_paths(run_dir_or_file):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = _float_or_none(row.get("val/exact_accuracy"))
                if value is not None:
                    latest = value
    return latest


def _lr_continues(first_lr: float | None) -> bool:
    if first_lr is None:
        return False
    return not math.isclose(first_lr, FRESH_WARMUP_FIRST_LR, rel_tol=0.0, abs_tol=1e-12)


def build_result_artifact(
    *,
    root_cause: str,
    lr_points: Sequence[MetricPoint],
    val_exact_accuracy: float | None,
    stable_checkpoint_path: str | Path,
    duration_s: float,
    prior_last_lr: float | None,
    command: Sequence[str],
    stdout_tail: Sequence[str] = (),
) -> dict[str, Any]:
    """SCENARIO-LEARN-4126: build the terminal LR-continuity artifact."""

    first_lr = lr_points[0].value if lr_points else None
    continuous = _lr_continues(first_lr)
    if continuous:
        verdict = f"complete: lr_resume_continuous_first_lr={first_lr:.8g}"
    else:
        verdict = "complete: bounded_resume_lr_still_rewarmed_contiguous_run_needed"

    return {
        "schema": "carnot.experiment_4126_lr_resume_correctness_fix.v1",
        "spec_refs": ["REQ-LEARN-4126", "SCENARIO-LEARN-4126"],
        "honest_verdict": verdict,
        "lr_rewarm_root_cause": root_cause,
        "lr_continuous_across_resume": continuous,
        "validation_first_lr": first_lr,
        "val_exact_accuracy": val_exact_accuracy,
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "duration_s": float(duration_s),
        "fresh_warmup_lr": FRESH_WARMUP_FIRST_LR,
        "prior_pass_last_lr": prior_last_lr,
        "train_lr_points": [point.to_dict() for point in lr_points],
        "command": list(command),
        "stdout_tail": list(stdout_tail),
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-4126: fail closed on missing fields or a non-terminal verdict."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")

    if not isinstance(artifact.get("lr_continuous_across_resume"), bool):
        raise ValueError("lr_continuous_across_resume must be a bare bool")

    duration = _float_or_none(artifact.get("duration_s"))
    if duration is None or duration < 0 or duration >= 4_800:
        raise ValueError("duration_s must be a bounded nonnegative GPU runtime below 4800")

    if artifact["lr_continuous_across_resume"]:
        first_lr = _float_or_none(artifact.get("validation_first_lr"))
        if first_lr is None or not _lr_continues(first_lr):
            raise ValueError("continuous artifacts must report a non-rewarm validation_first_lr")


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)
