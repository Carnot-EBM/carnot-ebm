"""Exp 4167 read-only status report for outer-loop-owned TRM training.

The conductor no longer owns the contiguous Sudoku Extreme baseline run. This
module only reads the shared checkpoint scalar state, the outer-loop PID, and
CSV validation logs so the graft can decide whether the checkpoint is both
faithful and stable. It deliberately contains no trainer launch path and no
process termination path.

Spec refs: REQ-LEARN-4167, SCENARIO-LEARN-4167-READONLY-MONITOR,
SCENARIO-LEARN-4167-FAITHFUL-STABLE.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import csv
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4167_outerloop_training_monitor.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_TRM_RUNS = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_CHECKPOINT = DEFAULT_TRM_RUNS / "sudoku_extreme_baseline" / "last.ckpt"
DEFAULT_CONTIGUOUS_RUN_DIR = DEFAULT_TRM_RUNS / "contiguous_run_hydra"
DEFAULT_PID_PATH = DEFAULT_TRM_RUNS / "contiguous_run.pid"
SCHEMA = "carnot.experiment_4167_outerloop_training_monitor.v1"
EXPERIMENT_ID = 4167
FAITHFUL_VAL_THRESHOLD = 0.85
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "outerloop_train_alive",
    "current_val_exact_accuracy",
    "baseline_faithful",
    "checkpoint_mtime",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A status report is COMPLETE; this task never fails "
        "on the science (it does not run any)."
    ),
    "outerloop_train_alive": (
        "Bare bool: is the outer-loop training process running? Lets the graft "
        "know whether the checkpoint is stable (safe to read) or being written."
    ),
    "current_val_exact_accuracy": (
        "Latest val from the outer-loop run; the convergence signal toward 0.87."
    ),
    "baseline_faithful": (
        "Bare bool: val >= 0.85 AND train process NOT running -> the checkpoint "
        "is faithful + stable -> the graft may run."
    ),
    "checkpoint_mtime": (
        "Stable-checkpoint mtime; advancing past Jun-13-00:41 confirms the "
        "outer-loop run is saving real progress."
    ),
}


@dataclass(frozen=True)
class MonitorConfig:
    """Filesystem contract for the read-only outer-loop monitor."""

    repo_root: Path | str = REPO_ROOT
    checkpoint_path: Path | str | None = None
    contiguous_run_dir: Path | str | None = None
    pid_path: Path | str | None = None

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        trm_runs = root / "results" / "trm_runs"
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(
            self,
            "checkpoint_path",
            Path(self.checkpoint_path)
            if self.checkpoint_path is not None
            else trm_runs / "sudoku_extreme_baseline" / "last.ckpt",
        )
        object.__setattr__(
            self,
            "contiguous_run_dir",
            Path(self.contiguous_run_dir)
            if self.contiguous_run_dir is not None
            else trm_runs / "contiguous_run_hydra",
        )
        object.__setattr__(
            self,
            "pid_path",
            Path(self.pid_path) if self.pid_path is not None else trm_runs / "contiguous_run.pid",
        )


@dataclass(frozen=True)
class CheckpointStatus:
    """CPU-only checkpoint metadata used to judge checkpoint freshness."""

    path: str
    exists: bool
    load_ok: bool
    detail: str
    epoch: int | None
    mtime_epoch_s: float | None
    mtime_iso: str | None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ProcessStatus:
    """Result of the required `ps -o etime= -p <pid>` liveness probe."""

    pid: int | None
    alive: bool
    etime: str | None
    detail: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class MetricRow:
    """One validation exact-accuracy row from a contiguous-run CSV file."""

    metrics_path: Path
    version: int
    row_number: int
    epoch: int | None
    step: int | None
    val_exact_accuracy: float

    @property
    def signature(self) -> tuple[str, int, int | None, int | None, float]:
        return (
            str(self.metrics_path),
            self.row_number,
            self.epoch,
            self.step,
            self.val_exact_accuracy,
        )

    def to_dict(self, *, delta_vs_previous: float | None = None) -> JsonDict:
        return {
            "metrics_path": str(self.metrics_path),
            "csv_version": self.version,
            "row_number": self.row_number,
            "epoch": self.epoch,
            "step": self.step,
            "val_exact_accuracy": _rounded(self.val_exact_accuracy),
            "delta_vs_previous": _rounded(delta_vs_previous),
        }


@dataclass(frozen=True)
class MetricsSnapshot:
    """Parsed validation trajectory from all discovered contiguous-run CSV files."""

    rows: list[MetricRow]

    @property
    def latest_row(self) -> MetricRow | None:
        return self.rows[-1] if self.rows else None

    @property
    def current_val(self) -> float | None:
        latest = self.latest_row
        return None if latest is None else latest.val_exact_accuracy

    @property
    def metrics_paths(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            path = str(row.metrics_path)
            if path not in seen:
                seen.append(path)
        return seen

    def to_trajectory(self) -> list[JsonDict]:
        trajectory: list[JsonDict] = []
        previous: float | None = None
        for row in self.rows:
            delta = None if previous is None else row.val_exact_accuracy - previous
            trajectory.append(row.to_dict(delta_vs_previous=delta))
            previous = row.val_exact_accuracy
        return trajectory


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return None if number is None else int(number)


def _rounded(value: float | None, digits: int = 12) -> float | None:
    return None if value is None else round(float(value), digits)


def _mtime_iso(mtime: float | None) -> str | None:
    if mtime is None:
        return None
    return datetime.fromtimestamp(float(mtime), tz=UTC).isoformat().replace("+00:00", "Z")


def _version_number(metrics_path: Path) -> int:
    for parent in metrics_path.parents:
        name = parent.name
        if name.startswith("version_"):
            try:
                return int(name.split("_", maxsplit=1)[1])
            except ValueError:
                return -1
    return -1


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def read_checkpoint_status(path: str | Path) -> CheckpointStatus:
    """REQ-LEARN-4167: read checkpoint mtime and top-level epoch on CPU only."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return CheckpointStatus(
            str(checkpoint_path), False, False, "missing checkpoint", None, None, None
        )
    mtime = checkpoint_path.stat().st_mtime
    try:
        import torch

        payload = safe_torch_load(checkpoint_path, map_location="cpu", allow_unsafe_pickle=True)
    except Exception as exc:
        return CheckpointStatus(
            str(checkpoint_path),
            True,
            False,
            f"{type(exc).__name__}: {exc}",
            None,
            mtime,
            _mtime_iso(mtime),
        )
    if not isinstance(payload, Mapping):
        return CheckpointStatus(
            str(checkpoint_path),
            True,
            False,
            f"unexpected checkpoint payload: {type(payload).__name__}",
            None,
            mtime,
            _mtime_iso(mtime),
        )
    return CheckpointStatus(
        str(checkpoint_path),
        True,
        True,
        "loaded checkpoint scalar view",
        _int_or_none(payload.get("epoch")),
        mtime,
        _mtime_iso(mtime),
    )


def read_pid(path: str | Path) -> int | None:
    """Return the PID from the outer-loop PID file, tolerating small labels."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def check_pid_alive(
    pid: int,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> ProcessStatus:
    """REQ-LEARN-4167: use `ps -o etime= -p <pid>` for liveness only."""

    result = runner(
        ["ps", "-o", "etime=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = str(getattr(result, "stdout", "") or "").strip()
    stderr = str(getattr(result, "stderr", "") or "").strip()
    alive = int(getattr(result, "returncode", 1)) == 0 and bool(stdout)
    return ProcessStatus(
        pid=pid,
        alive=alive,
        etime=stdout or None,
        detail=stdout if alive else stderr or stdout or "process not found",
    )


def _metric_value(row: Mapping[str, Any]) -> float | None:
    return _float_or_none(row.get("val_exact_accuracy")) or _float_or_none(
        row.get("val/exact_accuracy")
    )


def read_metrics(root: str | Path) -> MetricsSnapshot:
    """REQ-LEARN-4167: parse validation exact accuracy from contiguous CSV logs."""

    base = Path(root)
    metrics_paths = sorted(
        base.rglob("metrics.csv"),
        key=lambda path: (_version_number(path), str(path)),
    )
    rows: list[MetricRow] = []
    for metrics_path in metrics_paths:
        try:
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, start=2):
                    val = _metric_value(row)
                    if val is None:
                        continue
                    rows.append(
                        MetricRow(
                            metrics_path=metrics_path,
                            version=_version_number(metrics_path),
                            row_number=row_number,
                            epoch=_int_or_none(row.get("epoch")),
                            step=_int_or_none(row.get("step")),
                            val_exact_accuracy=val,
                        )
                    )
        except FileNotFoundError:
            continue
    return MetricsSnapshot(rows)


def _verdict(current_val: float | None, *, alive: bool, crossed: bool, faithful: bool) -> str:
    if current_val is None:
        return "complete: outerloop_status_missing_val"
    if faithful:
        return f"complete: outerloop_stable_faithful_val_{current_val:.4f}"
    if alive and crossed:
        return f"complete: outerloop_val_crossed_0.85_but_checkpoint_live_val_{current_val:.4f}"
    if alive:
        return f"complete: outerloop_training_alive_val_{current_val:.4f}_below_0.85"
    return f"complete: outerloop_training_stopped_unfaithful_val_{current_val:.4f}"


def build_artifact(
    config: MonitorConfig,
    *,
    process_checker: Callable[[int], ProcessStatus] = check_pid_alive,
) -> JsonDict:
    """Build the Exp 4167 artifact without mutating any TRM training path."""

    checkpoint = read_checkpoint_status(config.checkpoint_path)
    pid = read_pid(config.pid_path)
    process = ProcessStatus(None, False, None, "missing pid")
    if pid is not None:
        process = process_checker(pid)
    metrics = read_metrics(config.contiguous_run_dir)
    latest = metrics.latest_row
    current_val = metrics.current_val
    crossed = bool(current_val is not None and current_val >= FAITHFUL_VAL_THRESHOLD)
    faithful = bool(crossed and not process.alive)
    payload: JsonDict = {
        "experiment": "experiment_4167_outerloop_training_monitor",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-LEARN-4167",
            "SCENARIO-LEARN-4167-READONLY-MONITOR",
            "SCENARIO-LEARN-4167-FAITHFUL-STABLE",
        ],
        "honest_verdict": _verdict(
            current_val, alive=process.alive, crossed=crossed, faithful=faithful
        ),
        "outerloop_train_alive": bool(process.alive),
        "current_val_exact_accuracy": _rounded(current_val),
        "baseline_faithful": faithful,
        "checkpoint_mtime": checkpoint.mtime_iso,
        "checkpoint_mtime_epoch_s": checkpoint.mtime_epoch_s,
        "checkpoint_epoch": checkpoint.epoch,
        "checkpoint_path": checkpoint.path,
        "checkpoint_read": checkpoint.to_dict(),
        "outerloop_pid": pid,
        "outerloop_pid_etime": process.etime,
        "outerloop_process": process.to_dict(),
        "faithful_threshold": FAITHFUL_VAL_THRESHOLD,
        "val_crossed_085": crossed,
        "val_trajectory": metrics.to_trajectory(),
        "latest_metrics_path": None if latest is None else str(latest.metrics_path),
        "metrics_files_read": metrics.metrics_paths,
        "read_only_actions": {
            "torch_load_cpu_only": True,
            "ps_etime_probe": pid is not None,
            "training_launched": False,
            "train_process_stop_attempted": False,
            "stable_checkpoint_written": False,
        },
        "inference_substrate": "read_only_outerloop_status_report",
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    payload["reproducibility_checksum"] = _payload_checksum(payload)
    validate_artifact(payload)
    return payload


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4167 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("outerloop_train_alive"), bool):
        errors.append("outerloop_train_alive must be a bare bool")
    if not isinstance(artifact.get("baseline_faithful"), bool):
        errors.append("baseline_faithful must be a bare bool")
    value = artifact.get("current_val_exact_accuracy")
    if value is not None:
        numeric = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        if numeric is None or not math.isfinite(float(numeric)) or not 0.0 <= float(numeric) <= 1.0:
            errors.append("current_val_exact_accuracy must be numeric between 0 and 1 or null")
    checkpoint_mtime = artifact.get("checkpoint_mtime")
    if checkpoint_mtime is not None and not isinstance(checkpoint_mtime, str):
        errors.append("checkpoint_mtime must be an ISO string or null")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != FIELD_PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles must include the required operator principles")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> JsonDict:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)
    return dict(artifact)


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    process_checker: Callable[[int], ProcessStatus] = check_pid_alive,
) -> JsonDict:
    """Run the read-only monitor and write the Exp 4167 result artifact."""

    artifact = build_artifact(MonitorConfig(repo_root=repo_root), process_checker=process_checker)
    return write_result_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
