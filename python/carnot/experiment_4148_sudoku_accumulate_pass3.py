"""Exp 4148 Sudoku Extreme pass3 continuation from the pass2 checkpoint.

This runner exists to keep the accumulate lineage honest. It trains only when
the pass2 artifact proves a real bounded pass and has not already reached the
0.87 target. If pass2 was blocked or no-op, pass3 records that upstream blocker
instead of spending GPU time on an ungrounded retry.

Spec refs: REQ-LEARN-4148, SCENARIO-LEARN-4148,
SCENARIO-LEARN-4148-BLOCKED-PASS2, SCENARIO-LEARN-4148-EARLY-CONVERGED.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4108_nanotrm_sudoku_extreme_baseline as exp4108
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4127_sudoku_extreme_accumulate_fixed as exp4127
from carnot import experiment_4135_sudoku_accumulate_pass1_fixed_lr as exp4135
from carnot import experiment_4146_sudoku_accumulate_pass1_epochfix as exp4146
from carnot import experiment_4147_sudoku_accumulate_pass2 as exp4147


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4148_sudoku_accumulate_pass3.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_HYDRA_RUN_ROOT = DEFAULT_SAVE_PARENT / "experiment_4148_sudoku_accumulate_pass3"
DEFAULT_PASS2_ARTIFACT = REPO_ROOT / "results" / exp4147.RESULT_FILENAME
RANDOM_SEED = exp4108.RANDOM_SEED
PASS_INDEX = 3
MAX_TIME = exp4146.MAX_TIME
EPOCH_CEILING_RAISE = exp4146.EPOCH_CEILING_RAISE
LOCAL_SAFE_BATCH_SIZE = exp4146.LOCAL_SAFE_BATCH_SIZE
MIN_REAL_TRAINING_DURATION_S = exp4146.MIN_REAL_TRAINING_DURATION_S
EARLY_CONVERGENCE_VAL = 0.87
TERMINAL_PREFIXES = exp4147.TERMINAL_PREFIXES

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_exact_accuracy",
    "delta_vs_pass2",
    "post_epoch",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed; honest val/no-op.",
    "val_exact_accuracy": "Solve metric after pass 3.",
    "delta_vs_pass2": "Per-pass improvement; convergence signal.",
    "post_epoch": "Anti-no-op: must advance.",
    "duration_s": "Real bounded pass; <120s = no-op.",
}

_float_or_none = exp4147._float_or_none
_int_or_none = exp4147._int_or_none
_rounded = exp4147._rounded
_checks_to_dicts = exp4147._checks_to_dicts
_write_json = exp4147._write_json


@dataclass(frozen=True)
class Exp4148Config:
    """Filesystem and trainer settings for the third epoch-fixed resume pass."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_root: Path | str | None = None
    dataset_dir: Path | str | None = None
    pass2_artifact_path: Path | str = DEFAULT_PASS2_ARTIFACT
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
        stable = Path(self.stable_dir) if self.stable_dir is not None else parent / "sudoku_extreme_baseline"
        hydra_root = (
            Path(self.hydra_run_root) if self.hydra_run_root is not None else parent / DEFAULT_HYDRA_RUN_ROOT.name
        )
        dataset = (
            Path(self.dataset_dir) if self.dataset_dir is not None else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        )
        pass2_path = Path(self.pass2_artifact_path)
        if pass2_path == DEFAULT_PASS2_ARTIFACT and root != REPO_ROOT:
            pass2_path = root / "results" / exp4147.RESULT_FILENAME
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "hydra_run_root", hydra_root)
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(self, "pass2_artifact_path", pass2_path)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def stable_checkpoint_path(self) -> Path:
        return Path(self.stable_dir) / "last.ckpt"

    def pass_run_dir(self) -> Path:
        return Path(self.hydra_run_root) / "pass_3_epochfix_hydra"

    def to_4146_config(self) -> exp4146.Exp4146Config:
        return exp4146.Exp4146Config(
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


def load_pass2_artifact(path: str | Path) -> dict[str, Any]:
    """REQ-LEARN-4148: read pass2 evidence before any pass3 training decision."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        return {"load_error": f"missing pass2 artifact: {artifact_path}"}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": f"{type(exc).__name__}: {exc}", "artifact_path": str(artifact_path)}
    return payload if isinstance(payload, dict) else {"load_error": f"unexpected pass2 payload: {type(payload).__name__}"}


def pass2_is_early_converged(pass2_artifact: Mapping[str, Any]) -> bool:
    """SCENARIO-LEARN-4148-EARLY-CONVERGED: pass2 at 0.87 needs no pass3."""

    val = _float_or_none(pass2_artifact.get("val_exact_accuracy"))
    return val is not None and val >= EARLY_CONVERGENCE_VAL


def pass2_has_real_training(pass2_artifact: Mapping[str, Any]) -> bool:
    """REQ-LEARN-4148: pass2 must prove duration, epoch advance, and real val."""

    verdict = str(pass2_artifact.get("honest_verdict", ""))
    if verdict.startswith(("blocked_", "blocked_noop_")):
        return False
    duration = _float_or_none(pass2_artifact.get("duration_s"))
    val = _float_or_none(pass2_artifact.get("val_exact_accuracy"))
    pass1_epoch = _int_or_none(pass2_artifact.get("pass1_post_epoch"))
    post_epoch = _int_or_none(pass2_artifact.get("post_epoch"))
    return bool(
        verdict.startswith(("complete:", "success:", "passed:", "shipped:"))
        and duration is not None
        and duration > MIN_REAL_TRAINING_DURATION_S
        and val is not None
        and pass1_epoch is not None
        and post_epoch is not None
        and post_epoch > pass1_epoch
    )


def summarize_pass2_blocker(pass2_artifact: Mapping[str, Any]) -> str:
    """SCENARIO-LEARN-4148-BLOCKED-PASS2: restate why pass2 is unresolved."""

    if "load_error" in pass2_artifact:
        return str(pass2_artifact["load_error"])
    verdict = str(pass2_artifact.get("honest_verdict", "missing_honest_verdict"))
    blocked_cause = pass2_artifact.get("blocked_cause")
    parts = [
        f"pass2 verdict={verdict}",
        f"pass1_honest_verdict={pass2_artifact.get('pass1_honest_verdict')}",
        f"post_epoch={pass2_artifact.get('post_epoch')}",
        f"duration_s={pass2_artifact.get('duration_s')}",
        f"val_exact_accuracy={pass2_artifact.get('val_exact_accuracy')}",
    ]
    if isinstance(blocked_cause, str) and blocked_cause:
        parts.append(f"blocked_cause={blocked_cause}")
    parts.append("The pass2 artifact did not prove real pass2 training, so pass3 is stopped before retraining.")
    return "; ".join(parts)


def build_train_command(config: Exp4148Config, *, current_epoch: int) -> list[str]:
    """SCENARIO-LEARN-4148: build the native pass3 command with a raised ceiling."""

    target_max_epochs = int(current_epoch) + EPOCH_CEILING_RAISE
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
        f"+trainer.max_epochs={target_max_epochs}",
        f"+trainer.max_time={config.max_time}",
        "callbacks.model_checkpoint.monitor=val/exact_accuracy",
        "callbacks.model_checkpoint.mode=max",
        f"callbacks.model_checkpoint.dirpath={Path(config.stable_dir)}",
        "callbacks.model_checkpoint.save_last=true",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.auto_insert_metric_name=false",
        (
            "+callbacks.exp4148_progress._target_="
            "carnot.experiment_4127_sudoku_extreme_accumulate_fixed."
            "NanoTrmAccumulateFixedProgressPrinter"
        ),
        f"+callbacks.exp4148_progress.every_n_steps={int(config.progress_every_n_steps)}",
        f"+callbacks.exp4148_progress.checkpoint_dir={Path(config.stable_dir)}",
    ]


def build_train_env(config: Exp4148Config) -> dict[str, str]:
    """REQ-LEARN-4148: use the same fixed-LR resume environment as pass2."""

    return exp4146.build_train_env(config.to_4146_config())


def check_preconditions(
    config: Exp4148Config,
    *,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """REQ-LEARN-4148: verify uv, nano-trm, CUDA, and the stable checkpoint."""

    return exp4146.check_preconditions(
        config.to_4146_config(),
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict == "blocked_pass2_noop_unresolved":
        return True
    val = _float_or_none(artifact.get("val_exact_accuracy"))
    delta = _float_or_none(artifact.get("delta_vs_pass2"))
    post_epoch = _int_or_none(artifact.get("post_epoch"))
    if verdict.startswith("complete: early_converged"):
        return bool(val is not None and val >= EARLY_CONVERGENCE_VAL and delta == 0.0 and post_epoch is not None)
    duration = _float_or_none(artifact.get("duration_s"))
    pass2_epoch = _int_or_none(artifact.get("pass2_post_epoch"))
    return bool(
        verdict.startswith(("complete:", "success:", "passed:", "shipped:"))
        and duration is not None
        and duration > MIN_REAL_TRAINING_DURATION_S
        and val is not None
        and post_epoch is not None
        and pass2_epoch is not None
        and post_epoch > pass2_epoch
        and (delta is not None and (delta > 0.0 or artifact.get("honest_plateau") is True))
    )


def _common_artifact_fields(
    *,
    honest_verdict: str,
    run_config: Exp4148Config,
    val_exact_accuracy: float | None,
    delta_vs_pass2: float | None,
    post_epoch: int | None,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4148_sudoku_accumulate_pass3",
        "schema": "carnot.experiment_4148_sudoku_accumulate_pass3.v1",
        "spec_refs": ["REQ-LEARN-4148", "SCENARIO-LEARN-4148"],
        "honest_verdict": honest_verdict,
        "val_exact_accuracy": _rounded(val_exact_accuracy),
        "delta_vs_pass2": _rounded(delta_vs_pass2),
        "post_epoch": post_epoch,
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
    }


def build_blocked_pass2_artifact(
    *,
    run_config: Exp4148Config,
    pass2_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4148-BLOCKED-PASS2: build the mandated no-train artifact."""

    artifact = _common_artifact_fields(
        honest_verdict="blocked_pass2_noop_unresolved",
        run_config=run_config,
        val_exact_accuracy=None,
        delta_vs_pass2=None,
        post_epoch=_int_or_none(pass2_artifact.get("post_epoch")),
        duration_s=duration_s,
    )
    artifact.update(
        {
            "spec_refs": ["REQ-LEARN-4148", "SCENARIO-LEARN-4148-BLOCKED-PASS2"],
            "acceptance_gate_passed": True,
            "native_trainer_launched": False,
            "honest_plateau": False,
            "blocked_cause": summarize_pass2_blocker(pass2_artifact),
            "pass2_artifact_path": str(run_config.pass2_artifact_path),
            "pass2_honest_verdict": pass2_artifact.get("honest_verdict"),
            "pass2_val_exact_accuracy": _rounded(_float_or_none(pass2_artifact.get("val_exact_accuracy"))),
            "pass2_delta_vs_pass1": _rounded(_float_or_none(pass2_artifact.get("delta_vs_pass1"))),
            "pass2_post_epoch": _int_or_none(pass2_artifact.get("post_epoch")),
            "pass2_duration_s": _rounded(_float_or_none(pass2_artifact.get("duration_s")), 3),
            "preconditions_checked": _checks_to_dicts(preconditions_checked),
            "command": [],
        }
    )
    validate_artifact(artifact)
    return artifact


def build_early_converged_artifact(
    *,
    run_config: Exp4148Config,
    pass2_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4148-EARLY-CONVERGED: confirm pass2 already solved enough."""

    val = _float_or_none(pass2_artifact.get("val_exact_accuracy"))
    post_epoch = _int_or_none(pass2_artifact.get("post_epoch"))
    verdict_val = 0.0 if val is None else val
    artifact = _common_artifact_fields(
        honest_verdict=f"complete: early_converged_pass2_val={verdict_val:.4f}_no_pass3_training",
        run_config=run_config,
        val_exact_accuracy=val,
        delta_vs_pass2=0.0,
        post_epoch=post_epoch,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "spec_refs": ["REQ-LEARN-4148", "SCENARIO-LEARN-4148-EARLY-CONVERGED"],
            "acceptance_gate_passed": True,
            "native_trainer_launched": False,
            "honest_plateau": False,
            "pass2_artifact_path": str(run_config.pass2_artifact_path),
            "pass2_honest_verdict": pass2_artifact.get("honest_verdict"),
            "pass2_val_exact_accuracy": _rounded(val),
            "pass2_delta_vs_pass1": _rounded(_float_or_none(pass2_artifact.get("delta_vs_pass1"))),
            "pass2_post_epoch": post_epoch,
            "pass2_duration_s": _rounded(_float_or_none(pass2_artifact.get("duration_s")), 3),
            "preconditions_checked": _checks_to_dicts(preconditions_checked),
            "command": [],
        }
    )
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_runtime_blocked_artifact(
    reason: str,
    *,
    run_config: Exp4148Config,
    pass2_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """REQ-LEARN-4148: preserve schema fields when runtime preconditions fail."""

    pass2_epoch = _int_or_none(pass2_artifact.get("post_epoch"))
    artifact = _common_artifact_fields(
        honest_verdict=reason,
        run_config=run_config,
        val_exact_accuracy=None,
        delta_vs_pass2=None,
        post_epoch=pass2_epoch,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "native_trainer_launched": False,
            "honest_plateau": False,
            "blocked_cause": reason,
            "pass2_artifact_path": str(run_config.pass2_artifact_path),
            "pass2_honest_verdict": pass2_artifact.get("honest_verdict"),
            "pass2_val_exact_accuracy": _rounded(_float_or_none(pass2_artifact.get("val_exact_accuracy"))),
            "pass2_delta_vs_pass1": _rounded(_float_or_none(pass2_artifact.get("delta_vs_pass1"))),
            "pass2_post_epoch": pass2_epoch,
            "preconditions_checked": _checks_to_dicts(preconditions_checked),
            "command": [] if pass2_epoch is None else build_train_command(run_config, current_epoch=pass2_epoch),
        }
    )
    validate_artifact(artifact)
    return artifact


def _verdict_for_result(
    *,
    duration_s: float,
    pass2_epoch: int | None,
    post_epoch: int | None,
    val_exact_accuracy: float | None,
    delta_vs_pass2: float | None,
    honest_plateau: bool,
) -> str:
    if duration_s <= MIN_REAL_TRAINING_DURATION_S:
        return "blocked_noop_duration_too_short"
    if pass2_epoch is None or post_epoch is None or post_epoch <= pass2_epoch:
        return "blocked_noop_epoch_not_advanced"
    if val_exact_accuracy is None:
        return "blocked_noop_missing_val_exact_accuracy"
    if delta_vs_pass2 is None:
        return "blocked_noop_missing_delta_vs_pass2"
    if delta_vs_pass2 > 0.0:
        return f"complete: pass3_trained_post_epoch={post_epoch}_val={val_exact_accuracy:.4f}_delta={delta_vs_pass2:.4f}"
    if honest_plateau:
        return f"complete: honest_plateau_pass3_post_epoch={post_epoch}_val={val_exact_accuracy:.4f}_delta={delta_vs_pass2:.4f}"
    return "blocked_noop_nonpositive_delta_without_plateau"


def build_result_artifact(
    *,
    run_config: Exp4148Config,
    pass2_artifact: Mapping[str, Any],
    seed_state: exp4146.CheckpointState,
    post_state: exp4146.CheckpointState,
    run_result: exp4116.ResumeRunResult,
    val_exact_accuracy: float | None,
    val_metrics_path: Path | None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4148: report pass3 val, delta, and anti-no-op proof."""

    pass2_val = _float_or_none(pass2_artifact.get("val_exact_accuracy"))
    delta = None if pass2_val is None or val_exact_accuracy is None else float(val_exact_accuracy) - pass2_val
    honest_plateau = delta is not None and delta <= 0.0
    pass2_epoch = _int_or_none(pass2_artifact.get("post_epoch"))
    verdict = _verdict_for_result(
        duration_s=run_result.duration_s,
        pass2_epoch=pass2_epoch,
        post_epoch=post_state.epoch,
        val_exact_accuracy=val_exact_accuracy,
        delta_vs_pass2=delta,
        honest_plateau=honest_plateau,
    )
    artifact = _common_artifact_fields(
        honest_verdict=verdict,
        run_config=run_config,
        val_exact_accuracy=val_exact_accuracy,
        delta_vs_pass2=delta,
        post_epoch=post_state.epoch,
        duration_s=run_result.duration_s,
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "native_trainer_launched": True,
            "honest_plateau": bool(honest_plateau),
            "pass2_artifact_path": str(run_config.pass2_artifact_path),
            "pass2_honest_verdict": pass2_artifact.get("honest_verdict"),
            "pass2_val_exact_accuracy": _rounded(pass2_val),
            "pass2_delta_vs_pass1": _rounded(_float_or_none(pass2_artifact.get("delta_vs_pass1"))),
            "pass2_post_epoch": pass2_epoch,
            "seed_checkpoint_state": seed_state.to_dict(),
            "post_checkpoint_state": post_state.to_dict(),
            "return_code": int(run_result.return_code),
            "checkpoint_reload_ok": bool(run_result.checkpoint_reload_ok),
            "checkpoint_reload_detail": run_result.checkpoint_reload_detail,
            "exact_accuracy_metric": "val/exact_accuracy" if val_exact_accuracy is not None else None,
            "exact_accuracy_metrics_path": None if val_metrics_path is None else str(val_metrics_path),
            "run_dir": str(run_result.run_dir),
            "command": list(run_result.command),
            "stdout_tail": list(run_result.stdout_tail[-60:]),
        }
    )
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4148 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    val = artifact.get("val_exact_accuracy")
    if val is not None:
        number = _float_or_none(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None
        if number is None or not 0.0 <= number <= 1.0:
            errors.append("val_exact_accuracy must be numeric between 0 and 1 or null")
    delta = artifact.get("delta_vs_pass2")
    if delta is not None and (isinstance(delta, bool) or _float_or_none(delta) is None):
        errors.append("delta_vs_pass2 must be numeric or null")
    post_epoch = artifact.get("post_epoch")
    if post_epoch is not None and (not isinstance(post_epoch, int) or isinstance(post_epoch, bool)):
        errors.append("post_epoch must be an int or null")
    duration = _float_or_none(artifact.get("duration_s"))
    if duration is None or duration < 0 or duration >= 86_400:
        errors.append("duration_s must be a scalar bounded number below 86400")
    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(principles.get(field) != text for field, text in FIELD_PRINCIPLES.items()):
        errors.append("field_principles must include the required operator principles")
    if isinstance(verdict, str) and verdict.startswith(("complete:", "success:", "passed:", "shipped:")):
        if not _acceptance_gate(artifact):
            errors.append("complete/plateau verdict requires duration>120, epoch advance, real val, or early convergence")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)


def run_native_pass3(
    config: Exp4148Config,
    *,
    current_epoch: int,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches native trainer.
    """Run the real nano-trm pass3 trainer with progress prints and checkpointing."""

    started = time.time()
    command = build_train_command(config, current_epoch=current_epoch)
    stdout_lines: list[str] = []
    print(f"[exp4148] launching pass3 resume stable={config.stable_checkpoint_path}", flush=True)
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
        print(f"[exp4148:nano-trm] {clean}", flush=True)
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
        PASS_INDEX,
        duration_s=time.time() - started,
        return_code=return_code,
        command=command,
        stdout_tail=stdout_lines,
        checkpoint_loader=checkpoint_loader,
    )


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    save_parent: str | Path = DEFAULT_SAVE_PARENT,
    stable_dir: str | Path | None = None,
    hydra_run_root: str | Path | None = None,
    pass2_artifact_path: str | Path = DEFAULT_PASS2_ARTIFACT,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    trainer_runner: Callable[[Exp4148Config, int], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4148, or stop honestly when pass2 converged or was a no-op."""

    started = time.time()
    config = Exp4148Config(
        repo_root=repo_root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_root=hydra_run_root,
        pass2_artifact_path=pass2_artifact_path,
        random_seed=random_seed,
    )
    out = Path(output_path)
    checks, blocker = check_preconditions(
        config,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )
    pass2_artifact = load_pass2_artifact(config.pass2_artifact_path)
    if pass2_is_early_converged(pass2_artifact):
        artifact = build_early_converged_artifact(
            run_config=config,
            pass2_artifact=pass2_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact
    if not pass2_has_real_training(pass2_artifact):
        artifact = build_blocked_pass2_artifact(
            run_config=config,
            pass2_artifact=pass2_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact
    if blocker is not None:
        artifact = build_runtime_blocked_artifact(
            blocker,
            run_config=config,
            pass2_artifact=pass2_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    seed_state = exp4146.read_checkpoint_state(config.stable_checkpoint_path)
    current_epoch = seed_state.epoch
    if current_epoch is None:
        current_epoch = _int_or_none(pass2_artifact.get("post_epoch")) or 0
    try:
        run_result = (
            run_native_pass3(config, current_epoch=current_epoch, checkpoint_loader=checkpoint_loader)
            if trainer_runner is None
            else trainer_runner(config, current_epoch)
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
            command=build_train_command(config, current_epoch=current_epoch),
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            run_dir=config.pass_run_dir(),
        )
    post_state = exp4146.read_checkpoint_state(config.stable_checkpoint_path)
    metrics = exp4135.summarize_pass_metrics(config.pass_run_dir())
    artifact = build_result_artifact(
        run_config=config,
        pass2_artifact=pass2_artifact,
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
    print(json.dumps({field: artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS}, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
