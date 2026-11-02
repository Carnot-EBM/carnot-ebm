"""Exp 4146 Sudoku Extreme pass-1 epoch-ceiling no-op guard.

This module exists to keep a resumed nano-trm pass honest. A fast clean exit is
only progress when the checkpoint epoch advances and a real validation metric is
written; otherwise the artifact must say which stop condition prevented useful
training.

Spec refs: REQ-LEARN-4146, SCENARIO-LEARN-4146,
SCENARIO-LEARN-4146-BLOCKED-NOOP.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4108_nanotrm_sudoku_extreme_baseline as exp4108
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4127_sudoku_extreme_accumulate_fixed as exp4127
from carnot import experiment_4135_sudoku_accumulate_pass1_fixed_lr as exp4135


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4146_sudoku_accumulate_pass1_epochfix.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_HYDRA_RUN_ROOT = DEFAULT_SAVE_PARENT / "experiment_4146_sudoku_accumulate_pass1_epochfix"
RANDOM_SEED = exp4108.RANDOM_SEED
MAX_TIME = "00:01:00:00"
EPOCH_CEILING_RAISE = 3000
LOCAL_SAFE_BATCH_SIZE = exp4135.LOCAL_SAFE_BATCH_SIZE
MIN_REAL_TRAINING_DURATION_S = 120.0
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_", "blocked_noop_")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "max_epochs_cap_confirmed",
    "seed_epoch",
    "post_epoch",
    "val_exact_accuracy",
    "stable_checkpoint_path",
    "duration_s",
    "random_seed",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A `blocked_noop_*` (real training did not happen) is honest; "
        "a fake `complete` on a 7s no-op is the .383 anti-pattern."
    ),
    "max_epochs_cap_confirmed": (
        "Bare bool: was the .383 no-op caused by the checkpoint hitting max_epochs? "
        "Confirms the root cause + that the ceiling-raise addresses it."
    ),
    "seed_epoch": (
        "The checkpoint's epoch BEFORE this pass; paired with the post-pass epoch it "
        "proves training actually advanced (the anti-no-op evidence)."
    ),
    "post_epoch": "The checkpoint's epoch AFTER this pass; must exceed seed_epoch or the pass no-op'd.",
    "val_exact_accuracy": "Real val after the pass (should climb from 0.278); a null val = the pass did not eval = no-op.",
    "stable_checkpoint_path": "The shared path pass 2 resumes from.",
    "duration_s": "A real bounded pass is minutes of GPU; <120s = the no-op signal that stalled .383.",
    "random_seed": "Determinism precondition.",
}


@dataclass(frozen=True)
class Exp4146Config:
    """Filesystem and trainer settings for the epoch-ceiling repair pass."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_root: Path | str | None = None
    dataset_dir: Path | str | None = None
    random_seed: int = RANDOM_SEED
    max_time: str = MAX_TIME
    timeout_s: int = 86_700
    progress_every_n_steps: int = 100
    batch_size: int = LOCAL_SAFE_BATCH_SIZE

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
        hydra_root = (
            Path(self.hydra_run_root)
            if self.hydra_run_root is not None
            else parent / DEFAULT_HYDRA_RUN_ROOT.name
        )
        dataset = (
            Path(self.dataset_dir)
            if self.dataset_dir is not None
            else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        )
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "hydra_run_root", hydra_root)
        object.__setattr__(self, "dataset_dir", dataset)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def stable_checkpoint_path(self) -> Path:
        return Path(self.stable_dir) / "last.ckpt"

    @property
    def experiment_config_path(self) -> Path:
        return (
            Path(self.nano_trm_root)
            / "src"
            / "nn"
            / "configs"
            / "experiment"
            / "trm_sudoku_extreme_1k_aug_1k.yaml"
        )

    @property
    def data_config_path(self) -> Path:
        return (
            Path(self.nano_trm_root)
            / "src"
            / "nn"
            / "configs"
            / "data"
            / "sudoku_extreme_1k_aug1k.yaml"
        )

    def pass_run_dir(self) -> Path:
        return Path(self.hydra_run_root) / "pass_1_epochfix_hydra"

    def to_4127_config(self) -> exp4127.Exp4127Config:
        return exp4127.Exp4127Config(
            repo_root=self.repo_root,
            nano_trm_root=self.nano_trm_root,
            save_parent=self.save_parent,
            stable_dir=self.stable_dir,
            hydra_run_root=self.hydra_run_root,
            dataset_dir=self.dataset_dir,
            random_seed=self.random_seed,
            max_time=self.max_time,
            timeout_s=self.timeout_s,
            progress_every_n_steps=self.progress_every_n_steps,
            batch_size=self.batch_size,
        )


@dataclass(frozen=True)
class CheckpointState:
    """Small, serializable view of the Lightning checkpoint stop state."""

    load_ok: bool
    detail: str
    epoch: int | None
    global_step: int | None
    manual_lr_step: int | None
    timer_train_elapsed_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpochCapDiagnosis:
    """The root-cause decision that gates whether training is allowed."""

    checkpoint_epoch: int | None
    checkpoint_global_step: int | None
    config_max_epochs: int | None
    target_max_epochs: int | None
    max_epochs_cap_confirmed: bool
    timer_train_elapsed_s: float | None
    max_time_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
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


def _checks_to_dicts(
    checks: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        check.to_dict() if isinstance(check, exp4107.PreconditionCheck) else dict(check)
        for check in checks
    ]


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _timer_train_elapsed(callbacks: Any) -> float | None:
    if not isinstance(callbacks, Mapping):
        return None
    for key, callback_state in callbacks.items():
        if key == "Timer" or (
            isinstance(callback_state, Mapping) and "time_elapsed" in callback_state
        ):
            elapsed = (
                callback_state.get("time_elapsed") if isinstance(callback_state, Mapping) else None
            )
            return _float_or_none(elapsed.get("train") if isinstance(elapsed, Mapping) else None)
    return None


def read_checkpoint_state(path: str | Path) -> CheckpointState:
    """REQ-LEARN-4146: load the checkpoint epoch/timer state used for no-op diagnosis."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return CheckpointState(False, f"missing: {checkpoint_path}", None, None, None, None)
    try:
        import torch  # pylint: disable=import-outside-toplevel

        try:
            payload = safe_torch_load(checkpoint_path, map_location="cpu", allow_unsafe_pickle=True)
        except TypeError:  # pragma: no cover - older torch compatibility.
            payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - corrupt local checkpoint.
        return CheckpointState(False, f"{type(exc).__name__}: {exc}", None, None, None, None)
    if not isinstance(
        payload, Mapping
    ):  # pragma: no cover - defensive against unexpected torch payloads.
        return CheckpointState(
            False,
            f"unexpected checkpoint payload: {type(payload).__name__}",
            None,
            None,
            None,
            None,
        )

    epoch = _int_or_none(payload.get("epoch"))
    if epoch is None:
        epoch = _int_or_none(
            _nested_get(payload, ("loops", "fit_loop", "epoch_progress", "total", "completed"))
        )
    return CheckpointState(
        load_ok=True,
        detail="torch.load ok",
        epoch=epoch,
        global_step=_int_or_none(payload.get("global_step")),
        manual_lr_step=_int_or_none(payload.get("nano_trm_manual_lr_step")),
        timer_train_elapsed_s=_timer_train_elapsed(payload.get("callbacks")),
    )


def read_config_max_epochs(path: str | Path) -> int | None:
    """REQ-LEARN-4146: read the resolved epoch ceiling from the experiment YAML."""

    config_path = Path(path)
    if not config_path.exists():
        return None
    try:
        import yaml  # pylint: disable=import-outside-toplevel

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    trainer_value = _nested_get(payload, ("trainer", "max_epochs"))
    if _int_or_none(trainer_value) is not None:
        return _int_or_none(trainer_value)
    return _int_or_none(_nested_get(payload, ("timekeeping", "max_epochs")))


def _max_time_seconds(value: str) -> float | None:
    parts = value.split(":")
    if len(parts) != 4:
        return None
    numbers = [_float_or_none(part) for part in parts]
    if any(number is None for number in numbers):
        return None
    days, hours, minutes, seconds = [float(number) for number in numbers if number is not None]
    return days * 86_400 + hours * 3_600 + minutes * 60 + seconds


def diagnose_epoch_cap(
    config: Exp4146Config,
    *,
    checkpoint_state: CheckpointState | None = None,
) -> EpochCapDiagnosis:
    """REQ-LEARN-4146: decide whether the requested max-epochs root cause is real."""

    state = (
        checkpoint_state
        if checkpoint_state is not None
        else read_checkpoint_state(config.stable_checkpoint_path)
    )
    config_max_epochs = read_config_max_epochs(config.experiment_config_path)
    epoch = state.epoch
    cap_confirmed = (
        epoch is not None and config_max_epochs is not None and epoch >= config_max_epochs
    )
    return EpochCapDiagnosis(
        checkpoint_epoch=epoch,
        checkpoint_global_step=state.global_step,
        config_max_epochs=config_max_epochs,
        target_max_epochs=None if epoch is None else epoch + EPOCH_CEILING_RAISE,
        max_epochs_cap_confirmed=bool(cap_confirmed),
        timer_train_elapsed_s=state.timer_train_elapsed_s,
        max_time_s=_max_time_seconds(config.max_time),
    )


def build_train_command(config: Exp4146Config, *, seed_epoch: int) -> list[str]:
    """SCENARIO-LEARN-4146: build the native resume command with a raised epoch ceiling."""

    target_max_epochs = int(seed_epoch) + EPOCH_CEILING_RAISE
    return [
        "uv",
        "run",
        "python",
        "src/nn/train.py",
        "experiment=trm_sudoku_extreme_1k_aug_1k",
        "logger=csv",
        f"hydra.run.dir={config.pass_run_dir()}",
        "save_dir=null",
        "append_wandb_name_to_save_dir=false",
        f"seed={int(config.random_seed)}",
        "data.data_dir=./data/sudoku_extreme_1k_aug_1k",
        f"timekeeping.batch_size={int(config.batch_size)}",
        f"ckpt_path={config.stable_checkpoint_path}",
        f"trainer.max_epochs={target_max_epochs}",
        f"+trainer.max_time={config.max_time}",
        "callbacks.model_checkpoint.monitor=val/exact_accuracy",
        "callbacks.model_checkpoint.mode=max",
        f"callbacks.model_checkpoint.dirpath={Path(config.stable_dir)}",
        "callbacks.model_checkpoint.save_last=true",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.auto_insert_metric_name=false",
        (
            "+callbacks.exp4146_progress._target_="
            "carnot.experiment_4127_sudoku_extreme_accumulate_fixed."
            "NanoTrmAccumulateFixedProgressPrinter"
        ),
        f"+callbacks.exp4146_progress.every_n_steps={int(config.progress_every_n_steps)}",
        f"+callbacks.exp4146_progress.checkpoint_dir={Path(config.stable_dir)}",
    ]


def build_train_env(config: Exp4146Config) -> dict[str, str]:
    """REQ-LEARN-4146: preserve the fixed-LR resume environment from Exp 4127."""

    return exp4127.build_train_env(config.to_4127_config())


def check_preconditions(
    config: Exp4146Config,
    *,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """REQ-LEARN-4146: verify resources before any native trainer launch."""

    return exp4135.check_preconditions(
        exp4135.Exp4135Config(
            repo_root=config.repo_root,
            nano_trm_root=config.nano_trm_root,
            save_parent=config.save_parent,
            stable_dir=config.stable_dir,
            hydra_run_root=config.hydra_run_root,
            dataset_dir=config.dataset_dir,
            random_seed=config.random_seed,
            max_time=config.max_time,
            timeout_s=config.timeout_s,
            progress_every_n_steps=config.progress_every_n_steps,
            batch_size=config.batch_size,
        ),
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )


def _noop_reason_for_diagnosis(diagnosis: EpochCapDiagnosis) -> str | None:
    if diagnosis.max_epochs_cap_confirmed:
        return None
    if diagnosis.checkpoint_epoch is None or diagnosis.config_max_epochs is None:
        return "blocked_noop_cap_diagnosis_incomplete"
    if (
        diagnosis.timer_train_elapsed_s is not None
        and diagnosis.max_time_s is not None
        and diagnosis.timer_train_elapsed_s >= diagnosis.max_time_s
    ):
        return "blocked_noop_cap_not_confirmed_timer_elapsed"
    return "blocked_noop_cap_not_confirmed"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict.startswith("blocked_noop_"):
        return True
    return bool(
        verdict.startswith(("complete:", "success:", "passed:", "shipped:"))
        and (_float_or_none(artifact.get("duration_s")) or 0.0) > MIN_REAL_TRAINING_DURATION_S
        and artifact.get("post_epoch") is not None
        and artifact.get("seed_epoch") is not None
        and int(artifact["post_epoch"]) > int(artifact["seed_epoch"])
        and _float_or_none(artifact.get("val_exact_accuracy")) is not None
    )


def _common_fields(
    *,
    honest_verdict: str,
    run_config: Exp4146Config,
    diagnosis: EpochCapDiagnosis,
    duration_s: float,
    post_epoch: int | None,
    val_exact_accuracy: float | None,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4146_sudoku_accumulate_pass1_epochfix",
        "schema": "carnot.experiment_4146_sudoku_accumulate_pass1_epochfix.v1",
        "spec_refs": ["REQ-LEARN-4146", "SCENARIO-LEARN-4146"],
        "honest_verdict": honest_verdict,
        "max_epochs_cap_confirmed": bool(diagnosis.max_epochs_cap_confirmed),
        "seed_epoch": diagnosis.checkpoint_epoch,
        "post_epoch": post_epoch,
        "val_exact_accuracy": None
        if val_exact_accuracy is None
        else round(float(val_exact_accuracy), 12),
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "duration_s": round(float(duration_s), 3),
        "random_seed": int(run_config.random_seed),
        "field_principles": dict(FIELD_PRINCIPLES),
        "diagnosis": diagnosis.to_dict(),
    }


def build_blocked_artifact(
    reason: str,
    *,
    run_config: Exp4146Config,
    diagnosis: EpochCapDiagnosis,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """REQ-LEARN-4146: build a no-training artifact with the measured blocker."""

    seed_epoch = diagnosis.checkpoint_epoch
    artifact = _common_fields(
        honest_verdict=reason,
        run_config=run_config,
        diagnosis=diagnosis,
        duration_s=duration_s,
        post_epoch=seed_epoch,
        val_exact_accuracy=None,
    )
    artifact.update(
        {
            "acceptance_gate_passed": reason.startswith("blocked_noop_"),
            "preconditions_checked": _checks_to_dicts(preconditions_checked),
            "command": []
            if seed_epoch is None
            else build_train_command(run_config, seed_epoch=seed_epoch),
        }
    )
    validate_artifact(artifact)
    return artifact


def _verdict_for_result(
    seed_epoch: int | None, post_epoch: int | None, duration_s: float, val: float | None
) -> str:
    if duration_s <= MIN_REAL_TRAINING_DURATION_S:
        return "blocked_noop_duration_too_short"
    if seed_epoch is None or post_epoch is None or post_epoch <= seed_epoch:
        return "blocked_noop_epoch_not_advanced"
    if val is None:
        return "blocked_noop_missing_val_exact_accuracy"
    return (
        f"complete: epochfix_trained_seed_epoch={seed_epoch}_post_epoch={post_epoch}_val={val:.4f}"
    )


def build_result_artifact(
    *,
    run_config: Exp4146Config,
    diagnosis: EpochCapDiagnosis,
    seed_state: CheckpointState,
    post_state: CheckpointState,
    run_result: exp4116.ResumeRunResult,
    val_exact_accuracy: float | None,
    val_metrics_path: Path | None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4146: build the artifact and apply the anti-no-op guard."""

    verdict = _verdict_for_result(
        seed_state.epoch, post_state.epoch, run_result.duration_s, val_exact_accuracy
    )
    artifact = _common_fields(
        honest_verdict=verdict,
        run_config=run_config,
        diagnosis=diagnosis,
        duration_s=run_result.duration_s,
        post_epoch=post_state.epoch,
        val_exact_accuracy=val_exact_accuracy,
    )
    artifact.update(
        {
            "acceptance_gate_passed": _acceptance_gate(artifact),
            "seed_checkpoint_state": seed_state.to_dict(),
            "post_checkpoint_state": post_state.to_dict(),
            "return_code": int(run_result.return_code),
            "checkpoint_reload_ok": bool(run_result.checkpoint_reload_ok),
            "checkpoint_reload_detail": run_result.checkpoint_reload_detail,
            "exact_accuracy_metric": "val/exact_accuracy"
            if val_exact_accuracy is not None
            else None,
            "exact_accuracy_metrics_path": None
            if val_metrics_path is None
            else str(val_metrics_path),
            "run_dir": str(run_result.run_dir),
            "command": list(run_result.command),
            "stdout_tail": list(run_result.stdout_tail[-60:]),
        }
    )
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4146 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed or blocked_noop")
    if not isinstance(artifact.get("max_epochs_cap_confirmed"), bool):
        errors.append("max_epochs_cap_confirmed must be a bare bool")
    for field in ("seed_epoch", "post_epoch"):
        value = artifact.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            errors.append(f"{field} must be an int or null")
    val = artifact.get("val_exact_accuracy")
    if val is not None:
        number = (
            _float_or_none(val)
            if isinstance(val, (int, float)) and not isinstance(val, bool)
            else None
        )
        if number is None or not 0.0 <= number <= 1.0:
            errors.append("val_exact_accuracy must be numeric between 0 and 1 or null")
    stable_checkpoint_path = artifact.get("stable_checkpoint_path")
    if not isinstance(stable_checkpoint_path, str) or not stable_checkpoint_path.endswith(
        "results/trm_runs/sudoku_extreme_baseline/last.ckpt"
    ):
        errors.append(
            "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path"
        )
    duration = _float_or_none(artifact.get("duration_s"))
    if duration is None or duration < 0 or duration >= 86_400:
        errors.append("duration_s must be a scalar bounded number below 86400")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(
        artifact.get("random_seed"), bool
    ):
        errors.append("random_seed must be a bare int")
    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if isinstance(verdict, str) and verdict.startswith(
        ("complete:", "success:", "passed:", "shipped:")
    ):
        if not _acceptance_gate(artifact):
            errors.append(
                "complete verdict requires duration>120, epoch advance, and real val_exact_accuracy"
            )
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)


def run_native_epochfixed_pass(
    config: Exp4146Config,
    *,
    seed_epoch: int,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches native trainer.
    """Run the real nano-trm trainer with the raised epoch ceiling."""

    started = time.time()
    command = build_train_command(config, seed_epoch=seed_epoch)
    stdout_lines: list[str] = []
    print(
        f"[exp4146] launching epoch-fixed resume stable={config.stable_checkpoint_path}", flush=True
    )
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(config.nano_trm_root),
            env=build_train_env(config),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )  # noqa: S603
    except Exception as exc:
        return exp4116.ResumeRunResult(
            return_code=1,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=False,
            checkpoint_reload_detail=f"{type(exc).__name__}: {exc}",
            val_exact_accuracy=None,
            cumulative_epochs=None,
            duration_s=time.time() - started,
            command=command,
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            run_dir=config.pass_run_dir(),
        )

    assert proc.stdout is not None
    timed_out = False
    for line in proc.stdout:
        clean = line.rstrip()
        stdout_lines.append(clean)
        print(f"[exp4146:nano-trm] {clean}", flush=True)
        if time.time() - started > config.timeout_s:
            proc.kill()
            stdout_lines.append(f"timeout_s exceeded: {config.timeout_s}")
            timed_out = True
            break
    return_code = proc.wait()
    if timed_out and return_code == 0:
        return_code = 124
    return exp4127.verify_completed_resume_pass(
        config.to_4127_config(),
        1,
        duration_s=time.time() - started,
        return_code=return_code,
        command=command,
        stdout_tail=stdout_lines,
        checkpoint_loader=checkpoint_loader,
    )


def generate_sudoku_extreme_dataset_if_missing(
    config: Exp4146Config,
) -> bool:  # pragma: no cover - native helper.
    """Generate the nano-trm Sudoku Extreme dataset only when it is absent."""

    return exp4116.generate_sudoku_extreme_dataset_if_missing(
        exp4116.Exp4116Config(
            repo_root=config.repo_root,
            nano_trm_root=config.nano_trm_root,
            save_parent=config.save_parent,
            stable_dir=config.stable_dir,
            hydra_run_dir=config.pass_run_dir(),
            dataset_dir=config.dataset_dir,
            random_seed=config.random_seed,
            max_time=config.max_time,
            timeout_s=config.timeout_s,
            progress_every_n_steps=config.progress_every_n_steps,
        )
    )


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    save_parent: str | Path = DEFAULT_SAVE_PARENT,
    stable_dir: str | Path | None = None,
    hydra_run_root: str | Path | None = None,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    dataset_builder: Callable[[Exp4146Config], object] = generate_sudoku_extreme_dataset_if_missing,
    timer_reset: Callable[
        [Path], exp4135.TimerResetResult | Mapping[str, Any]
    ] = exp4135.reset_checkpoint_timer_state,
    trainer_runner: Callable[[Exp4146Config, int], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4146, or write an honest blocker artifact before any no-op claim."""

    started = time.time()
    config = Exp4146Config(
        repo_root=repo_root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_root=hydra_run_root,
        random_seed=random_seed,
    )
    out = Path(output_path)
    checks, blocker = check_preconditions(
        config,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )
    seed_state = read_checkpoint_state(config.stable_checkpoint_path)
    diagnosis = diagnose_epoch_cap(config, checkpoint_state=seed_state)
    if blocker is not None:
        artifact = build_blocked_artifact(
            blocker,
            run_config=config,
            diagnosis=diagnosis,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    noop_reason = _noop_reason_for_diagnosis(diagnosis)
    if noop_reason is not None:
        artifact = build_blocked_artifact(
            noop_reason,
            run_config=config,
            diagnosis=diagnosis,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    if not exp4108.dataset_is_complete(config.dataset_dir):
        dataset_builder(config)
    if not exp4108.dataset_is_complete(config.dataset_dir):
        artifact = build_blocked_artifact(
            "blocked_dataset_missing",
            run_config=config,
            diagnosis=diagnosis,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    timer_reset(config.stable_checkpoint_path)
    seed_epoch = seed_state.epoch if seed_state.epoch is not None else 0
    try:
        run_result = (
            run_native_epochfixed_pass(
                config, seed_epoch=seed_epoch, checkpoint_loader=checkpoint_loader
            )
            if trainer_runner is None
            else trainer_runner(config, seed_epoch)
        )
    except Exception as exc:  # pragma: no cover - defensive native failure path.
        run_result = exp4116.ResumeRunResult(
            return_code=1,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=False,
            checkpoint_reload_detail=f"{type(exc).__name__}: {exc}",
            val_exact_accuracy=None,
            cumulative_epochs=None,
            duration_s=time.time() - started,
            command=build_train_command(config, seed_epoch=seed_epoch),
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            run_dir=config.pass_run_dir(),
        )
    post_state = read_checkpoint_state(config.stable_checkpoint_path)
    metrics = exp4135.summarize_pass_metrics(config.pass_run_dir())
    artifact = build_result_artifact(
        run_config=config,
        diagnosis=diagnosis,
        seed_state=seed_state,
        post_state=post_state,
        run_result=run_result,
        val_exact_accuracy=metrics.val_exact_accuracy,
        val_metrics_path=metrics.val_metrics_path,
    )
    _write_json(out, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(
        json.dumps(
            {field: artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
