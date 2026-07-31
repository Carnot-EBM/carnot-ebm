"""Exp 4157 contiguous Sudoku baseline harvest and continue.

This module answers the `.385` baseline-readiness question without repeating
the `.384` bounded-pass mistake. It reads the contiguous CSV logs first, checks
whether the operator run is still alive, and only launches one long contiguous
resume when the run is dead and the baseline is still below the faithful gate.

Spec refs: REQ-LEARN-4157, SCENARIO-LEARN-4157-LIVE,
SCENARIO-LEARN-4157-FAITHFUL, SCENARIO-LEARN-4157-CONTINUE.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import csv
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4127_sudoku_extreme_accumulate_fixed as exp4127


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4157_baseline_harvest_contiguous_continue.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_CONTIGUOUS_RUN_DIR = DEFAULT_SAVE_PARENT / "contiguous_run_hydra"
DEFAULT_PID_PATH = DEFAULT_SAVE_PARENT / "contiguous_run.pid"
RANDOM_SEED = 4108
MAX_TIME = "00:11:30:00"
FAITHFUL_VAL_THRESHOLD = 0.85
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "current_val",
    "max_val",
    "baseline_faithful",
    "run_alive",
    "manual_lr_step",
    "val_trajectory",
    "stable_checkpoint_path",
    "estimated_passes_to_085",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'baseline advancing via the live contiguous run, "
        "val=0.NN, .385 rerank carries the moat signal' is COMPLETE and valuable."
    ),
    "current_val": (
        "Latest val_exact_accuracy from the contiguous-run CSV; the load-bearing "
        "baseline-readiness number."
    ),
    "max_val": "Best val ever recorded; the checkpoint the rerank/graft tasks should snapshot.",
    "baseline_faithful": (
        "Bare bool: val>=0.85. Tells exp4159 whether the full de-confounded RFT graft can run or must defer."
    ),
    "run_alive": (
        "Bare bool: is the operator's contiguous run still training? If true this task RECORDS ONLY "
        "(no competing launch)."
    ),
    "manual_lr_step": (
        "The persisted manual LR step; STEP-based progress evidence that does not misfire on the "
        "stale-Timer state that broke .384."
    ),
    "val_trajectory": "Val across the contiguous run; shows the convergence rate toward 0.87.",
    "stable_checkpoint_path": "The shared checkpoint the rerank (exp4158) + graft (exp4159) snapshot and build on.",
    "estimated_passes_to_085": "Estimated additional contiguous validation intervals needed to reach val>=0.85.",
}


@dataclass(frozen=True)
class Exp4157Config:
    """Filesystem and trainer settings for the contiguous baseline harvest."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    contiguous_run_dir: Path | str | None = None
    dataset_dir: Path | str | None = None
    pid_path: Path | str | None = None
    random_seed: int = RANDOM_SEED
    max_time: str = MAX_TIME
    liveness_probe_s: float = 0.0
    launch_poll_s: float = 5.0

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        parent = Path(self.save_parent)
        if parent == DEFAULT_SAVE_PARENT and root != REPO_ROOT:
            parent = root / "results" / "trm_runs"
        nano_root = Path(self.nano_trm_root) if self.nano_trm_root else root / "nano-trm"
        stable = (
            Path(self.stable_dir)
            if self.stable_dir is not None
            else parent / "sudoku_extreme_baseline"
        )
        run_dir = (
            Path(self.contiguous_run_dir)
            if self.contiguous_run_dir is not None
            else parent / DEFAULT_CONTIGUOUS_RUN_DIR.name
        )
        dataset = (
            Path(self.dataset_dir)
            if self.dataset_dir is not None
            else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        )
        pid = Path(self.pid_path) if self.pid_path is not None else parent / DEFAULT_PID_PATH.name
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "contiguous_run_dir", run_dir)
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(self, "pid_path", pid)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def stable_checkpoint_path(self) -> Path:
        return Path(self.stable_dir) / "last.ckpt"

    def to_4127_config(self) -> exp4127.Exp4127Config:
        return exp4127.Exp4127Config(
            repo_root=self.repo_root,
            nano_trm_root=self.nano_trm_root,
            save_parent=self.save_parent,
            stable_dir=self.stable_dir,
            hydra_run_root=self.contiguous_run_dir,
            dataset_dir=self.dataset_dir,
            random_seed=self.random_seed,
            max_time=self.max_time,
        )


@dataclass(frozen=True)
class MetricsRow:
    """One real `val/exact_accuracy` row from a contiguous CSV logger."""

    metrics_path: Path
    version: int
    row_number: int
    epoch: int | None
    step: int | None
    val_exact_accuracy: float

    def to_dict(self, *, delta_vs_previous: float | None = None) -> dict[str, Any]:
        return {
            "metrics_path": str(self.metrics_path),
            "csv_version": self.version,
            "row_number": self.row_number,
            "epoch": self.epoch,
            "step": self.step,
            "val_exact_accuracy": _rounded(self.val_exact_accuracy),
            "delta_vs_previous": _rounded(delta_vs_previous),
        }

    @property
    def signature(self) -> tuple[str, int, int | None, int | None, float]:
        return (
            str(self.metrics_path),
            self.row_number,
            self.epoch,
            self.step,
            self.val_exact_accuracy,
        )


@dataclass(frozen=True)
class MetricsSnapshot:
    """Parsed validation trajectory from the contiguous Hydra CSV tree."""

    rows: list[MetricsRow]

    @property
    def latest_row(self) -> MetricsRow | None:
        return self.rows[-1] if self.rows else None

    @property
    def max_row(self) -> MetricsRow | None:
        if not self.rows:
            return None
        return max(enumerate(self.rows), key=lambda item: (item[1].val_exact_accuracy, item[0]))[1]

    @property
    def current_val(self) -> float | None:
        row = self.latest_row
        return None if row is None else row.val_exact_accuracy

    @property
    def max_val(self) -> float | None:
        row = self.max_row
        return None if row is None else row.val_exact_accuracy

    @property
    def latest_signature(self) -> tuple[str, int, int | None, int | None, float] | None:
        row = self.latest_row
        return None if row is None else row.signature

    @property
    def val_row_count(self) -> int:
        return len(self.rows)

    def to_trajectory(self) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        previous: float | None = None
        for row in self.rows:
            delta = None if previous is None else row.val_exact_accuracy - previous
            trajectory.append(row.to_dict(delta_vs_previous=delta))
            previous = row.val_exact_accuracy
        return trajectory


@dataclass(frozen=True)
class CheckpointScalars:
    """CPU-only scalar view of the shared Lightning checkpoint."""

    load_ok: bool
    detail: str
    epoch: int | None
    manual_lr_step: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessStatus:
    """`ps` status for the PID recorded by the operator run."""

    alive: bool
    detail: str


@dataclass(frozen=True)
class RunLiveness:
    """Combined process and CSV-advancement liveness decision."""

    pid: int | None
    process_alive: bool
    csv_advancing: bool
    run_alive: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LaunchResult:
    """Result from a task-launched contiguous resume attempt."""

    process_pid: int | None
    return_code: int | None
    stdout_tail: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _checks_to_dicts(
    checks: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        check.to_dict() if isinstance(check, exp4107.PreconditionCheck) else dict(check)
        for check in checks
    ]


def _version_number(metrics_path: Path) -> int:
    name = metrics_path.parent.name
    if name.startswith("version_"):
        try:
            return int(name.split("_", maxsplit=1)[1])
        except ValueError:
            return -1
    return -1


def read_contiguous_metrics(root: str | Path = DEFAULT_CONTIGUOUS_RUN_DIR) -> MetricsSnapshot:
    """REQ-LEARN-4157: parse `val/exact_accuracy` from contiguous CSV logs."""

    base = Path(root)
    csv_root = base if base.name == "csv" else base / "csv"
    metrics_paths = sorted(
        csv_root.glob("version_*/metrics.csv"),
        key=lambda path: (_version_number(path), str(path)),
    )
    rows: list[MetricsRow] = []
    for metrics_path in metrics_paths:
        try:
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, start=2):
                    val = _float_or_none(row.get("val/exact_accuracy"))
                    if val is None:
                        continue
                    rows.append(
                        MetricsRow(
                            metrics_path=metrics_path,
                            version=_version_number(metrics_path),
                            row_number=row_number,
                            epoch=_int_or_none(row.get("epoch")),
                            step=_int_or_none(row.get("step")),
                            val_exact_accuracy=val,
                        )
                    )
        except FileNotFoundError:  # pragma: no cover - defensive race guard.
            continue
    return MetricsSnapshot(rows)


def read_checkpoint_scalars(path: str | Path) -> CheckpointScalars:
    """REQ-LEARN-4157: read only epoch and manual LR step from a checkpoint on CPU."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return CheckpointScalars(False, f"missing: {checkpoint_path}", None, None)
    try:
        import torch

        payload = safe_torch_load(checkpoint_path, map_location="cpu", allow_unsafe_pickle=True)
    except Exception as exc:
        return CheckpointScalars(False, f"{type(exc).__name__}: {exc}", None, None)
    if not isinstance(payload, Mapping):
        return CheckpointScalars(
            False, f"unexpected checkpoint payload: {type(payload).__name__}", None, None
        )
    return CheckpointScalars(
        load_ok=True,
        detail=f"loaded scalar checkpoint view: {checkpoint_path}",
        epoch=_int_or_none(payload.get("epoch")),
        manual_lr_step=_int_or_none(payload.get("nano_trm_manual_lr_step")),
    )


def estimate_passes_to_085(
    snapshot: MetricsSnapshot,
    *,
    target: float = FAITHFUL_VAL_THRESHOLD,
) -> dict[str, Any]:
    """Estimate additional validation intervals from the latest positive delta."""

    current = snapshot.current_val
    if current is None:
        return {
            "target_val": target,
            "estimated_additional_val_intervals": None,
            "basis": "missing_current_val",
            "latest_positive_delta": None,
        }
    if current >= target:
        return {
            "target_val": target,
            "estimated_additional_val_intervals": 0,
            "basis": "already_at_or_above_target",
            "latest_positive_delta": None,
        }
    latest_delta: float | None = None
    previous: float | None = None
    for row in snapshot.rows:
        if previous is not None:
            delta = row.val_exact_accuracy - previous
            if delta > 0:
                latest_delta = delta
        previous = row.val_exact_accuracy
    if latest_delta is None:
        return {
            "target_val": target,
            "estimated_additional_val_intervals": None,
            "basis": "no_positive_val_delta",
            "latest_positive_delta": None,
        }
    return {
        "target_val": target,
        "estimated_additional_val_intervals": int(math.ceil((target - current) / latest_delta)),
        "basis": "latest_positive_val_delta",
        "latest_positive_delta": _rounded(latest_delta),
    }


def _pid_from_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def _default_process_checker(pid: int) -> ProcessStatus:  # pragma: no cover - shell integration.
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid,etime"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or result.stderr or "").strip()
    lines = [line for line in output.splitlines() if line.strip()]
    alive = result.returncode == 0 and len(lines) > 1
    return ProcessStatus(alive=alive, detail=output)


def detect_liveness(
    config: Exp4157Config,
    before: MetricsSnapshot,
    *,
    process_checker: Callable[[int], ProcessStatus] = _default_process_checker,
    metrics_reader: Callable[[Path], MetricsSnapshot] = read_contiguous_metrics,
    sleeper: Callable[[float], None] = time.sleep,
    liveness_probe_s: float | None = None,
) -> RunLiveness:
    """SCENARIO-LEARN-4157-LIVE: require a live PID and advancing CSV metric."""

    pid = _pid_from_file(Path(config.pid_path))
    if pid is None:
        return RunLiveness(None, False, False, False, f"missing pid: {config.pid_path}")
    status = process_checker(pid)
    if not status.alive:
        return RunLiveness(pid, False, False, False, status.detail)
    probe_s = config.liveness_probe_s if liveness_probe_s is None else liveness_probe_s
    if probe_s > 0:
        sleeper(float(probe_s))
    after = metrics_reader(Path(config.contiguous_run_dir))
    csv_advancing = bool(
        before.latest_signature is not None
        and after.latest_signature is not None
        and (
            after.val_row_count > before.val_row_count
            or after.latest_signature != before.latest_signature
            or (
                after.latest_row is not None
                and before.latest_row is not None
                and after.latest_row.epoch is not None
                and before.latest_row.epoch is not None
                and after.latest_row.epoch > before.latest_row.epoch
            )
        )
    )
    latest = after.latest_row
    latest_detail = "missing latest val row" if latest is None else str(latest.metrics_path)
    return RunLiveness(
        pid=pid,
        process_alive=True,
        csv_advancing=csv_advancing,
        run_alive=bool(csv_advancing),
        detail=f"{status.detail}; latest_csv={latest_detail}; csv_advancing={csv_advancing}",
    )


def build_train_command(config: Exp4157Config) -> list[str]:
    """REQ-LEARN-4157: build the one long contiguous native resume command."""

    return [
        "uv",
        "run",
        "python",
        "src/nn/train.py",
        "experiment=trm_sudoku_extreme_1k_aug_1k",
        "logger=csv",
        f"hydra.run.dir={Path(config.contiguous_run_dir)}",
        "save_dir=null",
        "append_wandb_name_to_save_dir=false",
        f"seed={int(config.random_seed)}",
        "data.data_dir=./data/sudoku_extreme_1k_aug_1k",
        f"ckpt_path={config.stable_checkpoint_path}",
        f"+trainer.max_time={config.max_time}",
        "callbacks.model_checkpoint.monitor=val/exact_accuracy",
        "callbacks.model_checkpoint.mode=max",
        f"callbacks.model_checkpoint.dirpath={Path(config.stable_dir)}",
        "callbacks.model_checkpoint.save_last=true",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.auto_insert_metric_name=false",
    ]


def build_train_env(config: Exp4157Config) -> dict[str, str]:
    """REQ-LEARN-4157: mirror Exp 4127's no-compile CUDA-safe environment."""

    env = exp4127.build_train_env(config.to_4127_config())
    env["DISABLE_COMPILE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    return env


def check_preconditions(
    config: Exp4157Config,
    *,
    uv_resolver: Callable[[str], str | None] = shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """REQ-LEARN-4157: verify uv, trainer, CUDA, and the stable checkpoint."""

    checks = [
        exp4107._check_uv(uv_resolver),
        exp4107._check_trainer(config.repo_root),
        exp4107._check_cuda(cuda_checker),
    ]
    checkpoint_ok = Path(config.stable_checkpoint_path).exists()
    checks.append(
        exp4107.PreconditionCheck(
            "stable_checkpoint",
            checkpoint_ok,
            f"exists: {config.stable_checkpoint_path}"
            if checkpoint_ok
            else f"missing: {config.stable_checkpoint_path}",
        )
    )
    if not checks[0].available:
        return checks, "blocked_uv"
    if not checks[1].available:
        return checks, "blocked_trainer"
    if not checks[2].available:
        return checks, "blocked_cuda"
    if not checks[3].available:
        return checks, "blocked_stable_checkpoint"
    return checks, None


def launch_contiguous_run(  # pragma: no cover - launches native trainer.
    config: Exp4157Config,
    seed_checkpoint: CheckpointScalars,
    seed_metrics: MetricsSnapshot,
) -> LaunchResult:
    """Launch the single long resume and return once step+val progress is proven."""

    command = build_train_command(config)
    env = build_train_env(config)
    log_path = Path(config.contiguous_run_dir) / "exp4157_contiguous_resume.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[exp4157] launching contiguous run: {' '.join(command)}", flush=True)
    log_handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(config.nano_trm_root),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    Path(config.pid_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.pid_path).write_text(f"{proc.pid}\n", encoding="utf-8")
    stdout_tail = [f"log_path={log_path}", f"pid={proc.pid}"]
    latest_signature = seed_metrics.latest_signature
    while True:
        time.sleep(max(float(config.launch_poll_s), 1.0))
        snapshot = read_contiguous_metrics(config.contiguous_run_dir)
        checkpoint = read_checkpoint_scalars(config.stable_checkpoint_path)
        if snapshot.latest_signature is not None and snapshot.latest_signature != latest_signature:
            latest_signature = snapshot.latest_signature
            latest = snapshot.latest_row
            if latest is not None:
                line = (
                    f"[exp4157:nano-trm-progress] epoch={latest.epoch} step={latest.step} "
                    f"val_exact_accuracy={latest.val_exact_accuracy:.6f}"
                )
                print(line, flush=True)
                stdout_tail.append(line)
        step_advanced = (
            seed_checkpoint.manual_lr_step is not None
            and checkpoint.manual_lr_step is not None
            and checkpoint.manual_lr_step > seed_checkpoint.manual_lr_step
        )
        new_val = snapshot.val_row_count > seed_metrics.val_row_count
        if step_advanced and new_val:
            stdout_tail.append(
                f"progress_proven manual_lr_step={checkpoint.manual_lr_step} val_rows={snapshot.val_row_count}"
            )
            return LaunchResult(
                process_pid=proc.pid, return_code=None, stdout_tail=stdout_tail[-80:]
            )
        return_code = proc.poll()
        if return_code is not None:
            log_handle.close()
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            except OSError:
                log_tail = []
            stdout_tail.extend(log_tail)
            stdout_tail.append(f"return_code={return_code}")
            return LaunchResult(
                process_pid=proc.pid, return_code=return_code, stdout_tail=stdout_tail[-80:]
            )


def _artifact_base(
    *,
    honest_verdict: str,
    config: Exp4157Config,
    metrics: MetricsSnapshot,
    checkpoint: CheckpointScalars,
    liveness: RunLiveness,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    native_trainer_launched: bool,
    duration_s: float,
) -> dict[str, Any]:
    current_val = metrics.current_val
    artifact = {
        "experiment": "experiment_4157_baseline_harvest_contiguous_continue",
        "schema": "carnot.experiment_4157_baseline_harvest_contiguous_continue.v1",
        "spec_refs": ["REQ-LEARN-4157", "SCENARIO-LEARN-4157-LIVE", "SCENARIO-LEARN-4157-CONTINUE"],
        "honest_verdict": honest_verdict,
        "current_val": _rounded(current_val),
        "max_val": _rounded(metrics.max_val),
        "baseline_faithful": bool(
            current_val is not None and current_val >= FAITHFUL_VAL_THRESHOLD
        ),
        "run_alive": bool(liveness.run_alive),
        "manual_lr_step": checkpoint.manual_lr_step,
        "val_trajectory": metrics.to_trajectory(),
        "stable_checkpoint_path": str(config.stable_checkpoint_path),
        "estimated_passes_to_085": estimate_passes_to_085(metrics),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "checkpoint_scalars": checkpoint.to_dict(),
        "liveness": liveness.to_dict(),
        "native_trainer_launched": bool(native_trainer_launched),
        "task_launched_run": None,
        "blocked_cause": None,
        "command": build_train_command(config),
        "duration_s": round(float(duration_s), 3),
        "random_seed": int(config.random_seed),
    }
    return artifact


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    verdict = str(artifact.get("honest_verdict", ""))
    current_val = _float_or_none(artifact.get("current_val"))
    if current_val is None or not isinstance(artifact.get("run_alive"), bool):
        return False
    if not isinstance(artifact.get("baseline_faithful"), bool):
        return False
    if verdict.startswith("blocked_noop_step_unchanged"):
        return True
    if not verdict.startswith(("complete:", "success:", "passed:", "shipped:")):
        return False
    if artifact.get("native_trainer_launched") is True:
        launched = artifact.get("task_launched_run")
        return bool(
            isinstance(launched, Mapping)
            and launched.get("manual_lr_step_advanced") is True
            and launched.get("new_val_row_written") is True
        )
    return True


def _blocked_artifact(
    reason: str,
    *,
    config: Exp4157Config,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    metrics = read_contiguous_metrics(config.contiguous_run_dir)
    checkpoint = read_checkpoint_scalars(config.stable_checkpoint_path)
    liveness = RunLiveness(None, False, False, False, reason)
    artifact = _artifact_base(
        honest_verdict=reason,
        config=config,
        metrics=metrics,
        checkpoint=checkpoint,
        liveness=liveness,
        preconditions_checked=preconditions_checked,
        native_trainer_launched=False,
        duration_s=duration_s,
    )
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_record_artifact(
    *,
    honest_verdict: str,
    config: Exp4157Config,
    metrics: MetricsSnapshot,
    checkpoint: CheckpointScalars,
    liveness: RunLiveness,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4157-LIVE: build a no-launch harvest artifact."""

    artifact = _artifact_base(
        honest_verdict=honest_verdict,
        config=config,
        metrics=metrics,
        checkpoint=checkpoint,
        liveness=liveness,
        preconditions_checked=preconditions_checked,
        native_trainer_launched=False,
        duration_s=duration_s,
    )
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_launched_artifact(
    *,
    config: Exp4157Config,
    seed_checkpoint: CheckpointScalars,
    post_checkpoint: CheckpointScalars,
    seed_metrics: MetricsSnapshot,
    post_metrics: MetricsSnapshot,
    launch_result: LaunchResult,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4157-CONTINUE: grade a task-launched contiguous resume."""

    step_advanced = bool(
        seed_checkpoint.manual_lr_step is not None
        and post_checkpoint.manual_lr_step is not None
        and post_checkpoint.manual_lr_step > seed_checkpoint.manual_lr_step
    )
    new_val_written = post_metrics.val_row_count > seed_metrics.val_row_count
    liveness = RunLiveness(
        pid=launch_result.process_pid,
        process_alive=launch_result.return_code is None and launch_result.process_pid is not None,
        csv_advancing=new_val_written,
        run_alive=launch_result.return_code is None and launch_result.process_pid is not None,
        detail="task-launched contiguous run",
    )
    current_val = post_metrics.current_val
    if not step_advanced or not new_val_written:
        verdict = "blocked_noop_step_unchanged"
    elif current_val is not None and current_val >= FAITHFUL_VAL_THRESHOLD:
        verdict = f"complete: baseline_faithful_val_{current_val:.4f}"
    else:
        verdict = f"complete: baseline_contiguous_run_relaunched_val_{0.0 if current_val is None else current_val:.4f}"
    artifact = _artifact_base(
        honest_verdict=verdict,
        config=config,
        metrics=post_metrics,
        checkpoint=post_checkpoint,
        liveness=liveness,
        preconditions_checked=preconditions_checked,
        native_trainer_launched=True,
        duration_s=duration_s,
    )
    artifact["task_launched_run"] = {
        "process_pid": launch_result.process_pid,
        "return_code": launch_result.return_code,
        "manual_lr_step_before": seed_checkpoint.manual_lr_step,
        "manual_lr_step_after": post_checkpoint.manual_lr_step,
        "manual_lr_step_advanced": step_advanced,
        "val_rows_before": seed_metrics.val_row_count,
        "val_rows_after": post_metrics.val_row_count,
        "new_val_row_written": new_val_written,
        "stdout_tail": list(launch_result.stdout_tail[-80:]),
    }
    if not step_advanced:
        tail = " | ".join(str(line) for line in launch_result.stdout_tail[-8:])
        artifact["blocked_cause"] = (
            f"manual_lr_step did not advance: before={seed_checkpoint.manual_lr_step} "
            f"after={post_checkpoint.manual_lr_step}; trainer_return_code={launch_result.return_code}; log_tail={tail}"
        )
    elif not new_val_written:
        tail = " | ".join(str(line) for line in launch_result.stdout_tail[-8:])
        artifact["blocked_cause"] = (
            f"manual_lr_step advanced to {post_checkpoint.manual_lr_step} but no new val row was written; "
            f"trainer_return_code={launch_result.return_code}; log_tail={tail}"
        )
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def _verdict_for_live(current_val: float | None) -> str:
    return f"complete: baseline_advancing_contiguous_run_live_val_{0.0 if current_val is None else current_val:.4f}"


def _verdict_for_faithful(current_val: float | None) -> str:
    return f"complete: baseline_faithful_val_{0.0 if current_val is None else current_val:.4f}"


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    uv_resolver: Callable[[str], str | None] = shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    liveness_checker: Callable[[Exp4157Config, MetricsSnapshot], RunLiveness] | None = None,
    trainer_runner: Callable[
        [Exp4157Config, CheckpointScalars, MetricsSnapshot], LaunchResult
    ] = launch_contiguous_run,
) -> dict[str, Any]:
    """Run Exp 4157 and write the required harvest artifact."""

    started = time.time()
    config = Exp4157Config(repo_root=repo_root)
    checks, blocker = check_preconditions(
        config, uv_resolver=uv_resolver, cuda_checker=cuda_checker
    )
    if blocker is not None:
        artifact = _blocked_artifact(
            blocker,
            config=config,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        write_result_artifact(output_path, artifact)
        return artifact

    seed_metrics = read_contiguous_metrics(config.contiguous_run_dir)
    seed_checkpoint = read_checkpoint_scalars(config.stable_checkpoint_path)
    if not seed_checkpoint.load_ok:
        artifact = _blocked_artifact(
            "blocked_stable_checkpoint_load",
            config=config,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        write_result_artifact(output_path, artifact)
        return artifact
    liveness = (
        liveness_checker(config, seed_metrics)
        if liveness_checker is not None
        else detect_liveness(config, seed_metrics)
    )
    current_val = seed_metrics.current_val
    if current_val is None:
        artifact = build_record_artifact(
            honest_verdict="blocked_contiguous_metrics_missing",
            config=config,
            metrics=seed_metrics,
            checkpoint=seed_checkpoint,
            liveness=liveness,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        write_result_artifact(output_path, artifact)
        return artifact
    if liveness.run_alive:
        artifact = build_record_artifact(
            honest_verdict=_verdict_for_live(current_val),
            config=config,
            metrics=seed_metrics,
            checkpoint=seed_checkpoint,
            liveness=liveness,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        write_result_artifact(output_path, artifact)
        return artifact
    if current_val >= FAITHFUL_VAL_THRESHOLD:
        artifact = build_record_artifact(
            honest_verdict=_verdict_for_faithful(current_val),
            config=config,
            metrics=seed_metrics,
            checkpoint=seed_checkpoint,
            liveness=liveness,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        write_result_artifact(output_path, artifact)
        return artifact

    launch_result = trainer_runner(config, seed_checkpoint, seed_metrics)
    post_metrics = read_contiguous_metrics(config.contiguous_run_dir)
    post_checkpoint = read_checkpoint_scalars(config.stable_checkpoint_path)
    artifact = build_launched_artifact(
        config=config,
        seed_checkpoint=seed_checkpoint,
        post_checkpoint=post_checkpoint,
        seed_metrics=seed_metrics,
        post_metrics=post_metrics,
        launch_result=launch_result,
        preconditions_checked=checks,
        duration_s=time.time() - started,
    )
    write_result_artifact(output_path, artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4157 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    for key in ("current_val", "max_val"):
        value = artifact.get(key)
        parsed = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        if value is not None and (
            parsed is None
            or not math.isfinite(float(parsed))
            or float(parsed) < 0.0
            or float(parsed) > 1.0
        ):
            errors.append(f"{key} must be numeric between 0 and 1 or null")
    if not isinstance(artifact.get("baseline_faithful"), bool):
        errors.append("baseline_faithful must be a bare bool")
    if not isinstance(artifact.get("run_alive"), bool):
        errors.append("run_alive must be a bare bool")
    manual = artifact.get("manual_lr_step")
    if manual is not None and (isinstance(manual, bool) or not isinstance(manual, int)):
        errors.append("manual_lr_step must be an integer or null")
    if not isinstance(artifact.get("val_trajectory"), list):
        errors.append("val_trajectory must be a list")
    if not isinstance(artifact.get("stable_checkpoint_path"), str) or not artifact.get(
        "stable_checkpoint_path"
    ):
        errors.append("stable_checkpoint_path must be a non-empty string")
    if not isinstance(artifact.get("estimated_passes_to_085"), Mapping):
        errors.append("estimated_passes_to_085 must be an object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != FIELD_PRINCIPLES[field] for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles must include the required operator principles")
    if artifact.get("native_trainer_launched") is True:
        launched = artifact.get("task_launched_run")
        if not isinstance(launched, Mapping):
            errors.append("task_launched_run must describe launched-run progress")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)
    return dict(artifact)


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
