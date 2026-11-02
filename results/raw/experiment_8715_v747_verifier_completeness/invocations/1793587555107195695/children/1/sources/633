"""Exp 4149 final `.384` Sudoku Extreme accumulation decision.

This runner exists to make the last `.384` pass useful even when the upstream
lineage is blocked. A downstream graft needs a simple answer: what validation
accuracy is the persisted baseline actually at, does it match the published
0.87 target, and which checkpoint should the next milestone build on? When
Exp 4148 is a no-op, retraining from the same unresolved state would only hide
the blocker, so this module records the blocker and the full trajectory instead.

Spec refs: REQ-LEARN-4149, SCENARIO-LEARN-4149,
SCENARIO-LEARN-4149-BLOCKED-PASS3, SCENARIO-LEARN-4149-EARLY-CONVERGED.
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
from carnot import experiment_4148_sudoku_accumulate_pass3 as exp4148


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4149_sudoku_accumulate_pass4_convergence.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_HYDRA_RUN_ROOT = DEFAULT_SAVE_PARENT / "experiment_4149_sudoku_accumulate_pass4_convergence"
BASELINE_RESULT_FILENAME = "experiment_4145_archive_v383_activate_v384.json"
DEFAULT_BASELINE_ARTIFACT = REPO_ROOT / "results" / BASELINE_RESULT_FILENAME
DEFAULT_PASS1_ARTIFACT = REPO_ROOT / "results" / exp4146.RESULT_FILENAME
DEFAULT_PASS2_ARTIFACT = REPO_ROOT / "results" / exp4147.RESULT_FILENAME
DEFAULT_PASS3_ARTIFACT = REPO_ROOT / "results" / exp4148.RESULT_FILENAME
RANDOM_SEED = exp4108.RANDOM_SEED
PASS_INDEX = 4
PUBLISHED_EXACT_ACCURACY = 0.87
PUBLISHED_TOLERANCE = 0.02
MAX_TIME = exp4146.MAX_TIME
EPOCH_CEILING_RAISE = exp4146.EPOCH_CEILING_RAISE
LOCAL_SAFE_BATCH_SIZE = exp4146.LOCAL_SAFE_BATCH_SIZE
MIN_REAL_TRAINING_DURATION_S = exp4146.MIN_REAL_TRAINING_DURATION_S
TERMINAL_PREFIXES = exp4147.TERMINAL_PREFIXES

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_exact_accuracy",
    "matches_published_087",
    "val_trajectory_v384",
    "stable_checkpoint_path",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. An honest 'val=0.NN, .385 continues' is COMPLETE.",
    "val_exact_accuracy": "Final accumulated solve metric for .384.",
    "matches_published_087": (
        "Bare bool: within 0.02 of 0.87. Tells exp4150 whether a faithful baseline exists to graft onto."
    ),
    "val_trajectory_v384": "Val across the .384 passes (from 0.278); shows the convergence rate post-fix.",
    "stable_checkpoint_path": "The persisted baseline checkpoint exp4150 + .385 build on.",
    "duration_s": "Real bounded pass; <120s = no-op.",
}

_float_or_none = exp4147._float_or_none
_int_or_none = exp4147._int_or_none
_rounded = exp4147._rounded
_checks_to_dicts = exp4147._checks_to_dicts
_write_json = exp4147._write_json


@dataclass(frozen=True)
class Exp4149Config:
    """Filesystem and trainer settings for the final `.384` resume pass."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_root: Path | str | None = None
    dataset_dir: Path | str | None = None
    baseline_artifact_path: Path | str = DEFAULT_BASELINE_ARTIFACT
    pass1_artifact_path: Path | str = DEFAULT_PASS1_ARTIFACT
    pass2_artifact_path: Path | str = DEFAULT_PASS2_ARTIFACT
    pass3_artifact_path: Path | str = DEFAULT_PASS3_ARTIFACT
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
        baseline_path = Path(self.baseline_artifact_path)
        pass1_path = Path(self.pass1_artifact_path)
        pass2_path = Path(self.pass2_artifact_path)
        pass3_path = Path(self.pass3_artifact_path)
        if baseline_path == DEFAULT_BASELINE_ARTIFACT and root != REPO_ROOT:
            baseline_path = root / "results" / BASELINE_RESULT_FILENAME
        if pass1_path == DEFAULT_PASS1_ARTIFACT and root != REPO_ROOT:
            pass1_path = root / "results" / exp4146.RESULT_FILENAME
        if pass2_path == DEFAULT_PASS2_ARTIFACT and root != REPO_ROOT:
            pass2_path = root / "results" / exp4147.RESULT_FILENAME
        if pass3_path == DEFAULT_PASS3_ARTIFACT and root != REPO_ROOT:
            pass3_path = root / "results" / exp4148.RESULT_FILENAME
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "hydra_run_root", hydra_root)
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(self, "baseline_artifact_path", baseline_path)
        object.__setattr__(self, "pass1_artifact_path", pass1_path)
        object.__setattr__(self, "pass2_artifact_path", pass2_path)
        object.__setattr__(self, "pass3_artifact_path", pass3_path)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def stable_checkpoint_path(self) -> Path:
        return Path(self.stable_dir) / "last.ckpt"

    def pass_run_dir(self) -> Path:
        return Path(self.hydra_run_root) / "pass_4_epochfix_hydra"

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


def load_json_artifact(path: str | Path, *, label: str) -> dict[str, Any]:
    """REQ-LEARN-4149: load a lineage artifact without fabricating defaults."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        return {"load_error": f"missing {label} artifact: {artifact_path}", "artifact_path": str(artifact_path)}
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": f"{type(exc).__name__}: {exc}", "artifact_path": str(artifact_path)}
    if not isinstance(payload, dict):
        return {"load_error": f"unexpected {label} payload: {type(payload).__name__}", "artifact_path": str(artifact_path)}
    payload.setdefault("artifact_path", str(artifact_path))
    return payload


def matches_published_087(value: float | None) -> bool:
    """REQ-LEARN-4149: the published-baseline gate is a bare tolerance bool."""

    val = _float_or_none(value)
    return bool(val is not None and abs(val - PUBLISHED_EXACT_ACCURACY) <= PUBLISHED_TOLERANCE + 1e-12)


def _baseline_val(artifact: Mapping[str, Any]) -> float | None:
    close_state = artifact.get("v383_close_state")
    if isinstance(close_state, Mapping):
        for key in ("baseline_val_exact_accuracy", "checkpoint_val_exact_accuracy", "val_exact_accuracy"):
            val = _float_or_none(close_state.get(key))
            if val is not None:
                return val
    for key in ("baseline_val_exact_accuracy", "checkpoint_val_exact_accuracy", "val_exact_accuracy"):
        val = _float_or_none(artifact.get(key))
        if val is not None:
            return val
    return None


def _trajectory_entry(
    *,
    pass_label: str,
    experiment: str,
    artifact_path: Path,
    artifact: Mapping[str, Any],
    val_exact_accuracy: float | None,
    effective_val_exact_accuracy: float | None,
) -> dict[str, Any]:
    return {
        "pass_label": pass_label,
        "experiment": experiment,
        "artifact_path": str(artifact_path),
        "honest_verdict": artifact.get("honest_verdict") or artifact.get("load_error"),
        "val_exact_accuracy": _rounded(val_exact_accuracy),
        "effective_val_exact_accuracy": _rounded(effective_val_exact_accuracy),
        "post_epoch": _int_or_none(artifact.get("post_epoch")),
        "duration_s": _rounded(_float_or_none(artifact.get("duration_s")), 3),
    }


def build_val_trajectory_v384(
    config: Exp4149Config,
    *,
    pass3_artifact: Mapping[str, Any] | None = None,
    pass4_entry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """REQ-LEARN-4149: preserve the `.384` validation path from the 0.278 start."""

    baseline = load_json_artifact(config.baseline_artifact_path, label="baseline")
    pass1 = load_json_artifact(config.pass1_artifact_path, label="pass1")
    pass2 = load_json_artifact(config.pass2_artifact_path, label="pass2")
    pass3 = dict(pass3_artifact) if pass3_artifact is not None else load_json_artifact(config.pass3_artifact_path, label="pass3")
    effective = _baseline_val(baseline)
    entries = [
        _trajectory_entry(
            pass_label="v384_start",
            experiment="experiment_4145_archive_v383_activate_v384",
            artifact_path=Path(config.baseline_artifact_path),
            artifact=baseline,
            val_exact_accuracy=effective,
            effective_val_exact_accuracy=effective,
        )
    ]
    for pass_label, experiment, artifact_path, artifact in (
        ("pass1", "experiment_4146_sudoku_accumulate_pass1_epochfix", config.pass1_artifact_path, pass1),
        ("pass2", "experiment_4147_sudoku_accumulate_pass2", config.pass2_artifact_path, pass2),
        ("pass3", "experiment_4148_sudoku_accumulate_pass3", config.pass3_artifact_path, pass3),
    ):
        val = _float_or_none(artifact.get("val_exact_accuracy"))
        if val is not None:
            effective = val
        entries.append(
            _trajectory_entry(
                pass_label=pass_label,
                experiment=experiment,
                artifact_path=Path(artifact_path),
                artifact=artifact,
                val_exact_accuracy=val,
                effective_val_exact_accuracy=effective,
            )
        )
    if pass4_entry is not None:
        val = _float_or_none(pass4_entry.get("val_exact_accuracy"))
        if val is not None:
            effective = val
        entries.append(
            _trajectory_entry(
                pass_label="pass4",
                experiment="experiment_4149_sudoku_accumulate_pass4_convergence",
                artifact_path=DEFAULT_OUTPUT if Path(config.repo_root) == REPO_ROOT else Path(config.repo_root) / "results" / RESULT_FILENAME,
                artifact=pass4_entry,
                val_exact_accuracy=val,
                effective_val_exact_accuracy=effective,
            )
        )
    return entries


def final_effective_val(trajectory: Sequence[Mapping[str, Any]]) -> float | None:
    """REQ-LEARN-4149: choose the latest real metric, carrying the baseline through no-ops."""

    for row in reversed(trajectory):
        val = _float_or_none(row.get("effective_val_exact_accuracy"))
        if val is not None:
            return val
    return None


def pass3_is_early_converged(pass3_artifact: Mapping[str, Any]) -> bool:
    """SCENARIO-LEARN-4149-EARLY-CONVERGED: pass3 at 0.87 needs no pass4."""

    val = _float_or_none(pass3_artifact.get("val_exact_accuracy"))
    return val is not None and val >= PUBLISHED_EXACT_ACCURACY


def pass3_has_real_training(pass3_artifact: Mapping[str, Any]) -> bool:
    """REQ-LEARN-4149: pass3 must prove duration, epoch advance, and real val."""

    verdict = str(pass3_artifact.get("honest_verdict", ""))
    if verdict.startswith(("blocked_", "blocked_noop_")):
        return False
    duration = _float_or_none(pass3_artifact.get("duration_s"))
    val = _float_or_none(pass3_artifact.get("val_exact_accuracy"))
    pass2_epoch = _int_or_none(pass3_artifact.get("pass2_post_epoch"))
    post_epoch = _int_or_none(pass3_artifact.get("post_epoch"))
    return bool(
        verdict.startswith(("complete:", "success:", "passed:", "shipped:"))
        and duration is not None
        and duration > MIN_REAL_TRAINING_DURATION_S
        and val is not None
        and pass2_epoch is not None
        and post_epoch is not None
        and post_epoch > pass2_epoch
    )


def summarize_pass3_blocker(pass3_artifact: Mapping[str, Any]) -> str:
    """SCENARIO-LEARN-4149-BLOCKED-PASS3: restate why pass3 is unresolved."""

    if "load_error" in pass3_artifact:
        return str(pass3_artifact["load_error"])
    verdict = str(pass3_artifact.get("honest_verdict", "missing_honest_verdict"))
    parts = [
        f"pass3 verdict={verdict}",
        f"post_epoch={pass3_artifact.get('post_epoch')}",
        f"duration_s={pass3_artifact.get('duration_s')}",
        f"val_exact_accuracy={pass3_artifact.get('val_exact_accuracy')}",
    ]
    blocked_cause = pass3_artifact.get("blocked_cause")
    if isinstance(blocked_cause, str) and blocked_cause:
        parts.append(f"blocked_cause={blocked_cause}")
    parts.append("The pass3 artifact did not prove real pass3 training, so pass4 is stopped before retraining.")
    return "; ".join(parts)


def build_train_command(config: Exp4149Config, *, current_epoch: int) -> list[str]:
    """SCENARIO-LEARN-4149: build the native pass4 command with a raised ceiling."""

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
            "+callbacks.exp4149_progress._target_="
            "carnot.experiment_4127_sudoku_extreme_accumulate_fixed."
            "NanoTrmAccumulateFixedProgressPrinter"
        ),
        f"+callbacks.exp4149_progress.every_n_steps={int(config.progress_every_n_steps)}",
        f"+callbacks.exp4149_progress.checkpoint_dir={Path(config.stable_dir)}",
    ]


def build_train_env(config: Exp4149Config) -> dict[str, str]:
    """REQ-LEARN-4149: use the same fixed-LR resume environment as pass3."""

    return exp4146.build_train_env(config.to_4146_config())


def check_preconditions(
    config: Exp4149Config,
    *,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """REQ-LEARN-4149: verify uv, nano-trm, CUDA, and the stable checkpoint."""

    return exp4146.check_preconditions(
        config.to_4146_config(),
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    verdict = str(artifact.get("honest_verdict", ""))
    val = _float_or_none(artifact.get("val_exact_accuracy"))
    has_match_bool = isinstance(artifact.get("matches_published_087"), bool)
    if verdict == "blocked_pass3_noop_unresolved":
        return bool(val is not None and has_match_bool)
    if verdict.startswith("complete: early_converged"):
        return bool(val is not None and val >= PUBLISHED_EXACT_ACCURACY and has_match_bool)
    duration = _float_or_none(artifact.get("duration_s"))
    pass3_epoch = _int_or_none(artifact.get("pass3_post_epoch"))
    post_epoch = _int_or_none(artifact.get("post_epoch"))
    checkpoint = artifact.get("stable_checkpoint_path")
    return bool(
        verdict.startswith(("complete:", "success:", "passed:", "shipped:"))
        and duration is not None
        and duration > MIN_REAL_TRAINING_DURATION_S
        and val is not None
        and has_match_bool
        and pass3_epoch is not None
        and post_epoch is not None
        and post_epoch > pass3_epoch
        and isinstance(checkpoint, str)
        and checkpoint
    )


def _common_artifact_fields(
    *,
    honest_verdict: str,
    run_config: Exp4149Config,
    val_exact_accuracy: float | None,
    duration_s: float,
    trajectory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4149_sudoku_accumulate_pass4_convergence",
        "schema": "carnot.experiment_4149_sudoku_accumulate_pass4_convergence.v1",
        "spec_refs": ["REQ-LEARN-4149", "SCENARIO-LEARN-4149"],
        "honest_verdict": honest_verdict,
        "val_exact_accuracy": _rounded(val_exact_accuracy),
        "matches_published_087": matches_published_087(val_exact_accuracy),
        "val_trajectory_v384": [dict(row) for row in trajectory],
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": int(run_config.random_seed),
        "published_exact_accuracy_target": PUBLISHED_EXACT_ACCURACY,
        "published_match_tolerance": PUBLISHED_TOLERANCE,
    }


def build_blocked_pass3_artifact(
    *,
    run_config: Exp4149Config,
    pass3_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4149-BLOCKED-PASS3: build the mandated no-train artifact."""

    preliminary = {"honest_verdict": "blocked_pass3_noop_unresolved", "val_exact_accuracy": None, "duration_s": duration_s}
    trajectory = build_val_trajectory_v384(run_config, pass3_artifact=pass3_artifact, pass4_entry=preliminary)
    final_val = final_effective_val(trajectory)
    trajectory[-1]["val_exact_accuracy"] = _rounded(final_val)
    trajectory[-1]["effective_val_exact_accuracy"] = _rounded(final_val)
    artifact = _common_artifact_fields(
        honest_verdict="blocked_pass3_noop_unresolved",
        run_config=run_config,
        val_exact_accuracy=final_val,
        duration_s=duration_s,
        trajectory=trajectory,
    )
    artifact.update(
        {
            "spec_refs": ["REQ-LEARN-4149", "SCENARIO-LEARN-4149-BLOCKED-PASS3"],
            "acceptance_gate_passed": True,
            "native_trainer_launched": False,
            "blocked_cause": summarize_pass3_blocker(pass3_artifact),
            "pass3_artifact_path": str(run_config.pass3_artifact_path),
            "pass3_honest_verdict": pass3_artifact.get("honest_verdict"),
            "pass3_val_exact_accuracy": _rounded(_float_or_none(pass3_artifact.get("val_exact_accuracy"))),
            "pass3_post_epoch": _int_or_none(pass3_artifact.get("post_epoch")),
            "pass3_duration_s": _rounded(_float_or_none(pass3_artifact.get("duration_s")), 3),
            "preconditions_checked": _checks_to_dicts(preconditions_checked),
            "command": [],
        }
    )
    validate_artifact(artifact)
    return artifact


def build_early_converged_artifact(
    *,
    run_config: Exp4149Config,
    pass3_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4149-EARLY-CONVERGED: confirm pass3 already solved enough."""

    val = _float_or_none(pass3_artifact.get("val_exact_accuracy"))
    verdict_val = 0.0 if val is None else val
    preliminary = {
        "honest_verdict": f"complete: early_converged_pass3_val={verdict_val:.4f}_no_pass4_training",
        "val_exact_accuracy": val,
        "duration_s": duration_s,
        "post_epoch": _int_or_none(pass3_artifact.get("post_epoch")),
    }
    trajectory = build_val_trajectory_v384(run_config, pass3_artifact=pass3_artifact, pass4_entry=preliminary)
    artifact = _common_artifact_fields(
        honest_verdict=str(preliminary["honest_verdict"]),
        run_config=run_config,
        val_exact_accuracy=val,
        duration_s=duration_s,
        trajectory=trajectory,
    )
    artifact.update(
        {
            "spec_refs": ["REQ-LEARN-4149", "SCENARIO-LEARN-4149-EARLY-CONVERGED"],
            "acceptance_gate_passed": True,
            "native_trainer_launched": False,
            "pass3_artifact_path": str(run_config.pass3_artifact_path),
            "pass3_honest_verdict": pass3_artifact.get("honest_verdict"),
            "pass3_val_exact_accuracy": _rounded(val),
            "pass3_post_epoch": _int_or_none(pass3_artifact.get("post_epoch")),
            "pass3_duration_s": _rounded(_float_or_none(pass3_artifact.get("duration_s")), 3),
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
    run_config: Exp4149Config,
    pass3_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """REQ-LEARN-4149: preserve final metric and schema when preconditions fail."""

    pass3_epoch = _int_or_none(pass3_artifact.get("post_epoch"))
    final_val = _float_or_none(pass3_artifact.get("val_exact_accuracy"))
    preliminary = {
        "honest_verdict": reason,
        "val_exact_accuracy": final_val,
        "duration_s": duration_s,
        "post_epoch": pass3_epoch,
    }
    trajectory = build_val_trajectory_v384(run_config, pass3_artifact=pass3_artifact, pass4_entry=preliminary)
    artifact = _common_artifact_fields(
        honest_verdict=reason,
        run_config=run_config,
        val_exact_accuracy=final_effective_val(trajectory),
        duration_s=duration_s,
        trajectory=trajectory,
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "native_trainer_launched": False,
            "blocked_cause": reason,
            "pass3_artifact_path": str(run_config.pass3_artifact_path),
            "pass3_honest_verdict": pass3_artifact.get("honest_verdict"),
            "pass3_val_exact_accuracy": _rounded(_float_or_none(pass3_artifact.get("val_exact_accuracy"))),
            "pass3_post_epoch": pass3_epoch,
            "preconditions_checked": _checks_to_dicts(preconditions_checked),
            "command": [] if pass3_epoch is None else build_train_command(run_config, current_epoch=pass3_epoch),
        }
    )
    validate_artifact(artifact)
    return artifact


def _verdict_for_result(
    *,
    duration_s: float,
    pass3_epoch: int | None,
    post_epoch: int | None,
    val_exact_accuracy: float | None,
) -> str:
    if duration_s <= MIN_REAL_TRAINING_DURATION_S:
        return "blocked_noop_duration_too_short"
    if pass3_epoch is None or post_epoch is None or post_epoch <= pass3_epoch:
        return "blocked_noop_epoch_not_advanced"
    if val_exact_accuracy is None:
        return "blocked_noop_missing_val_exact_accuracy"
    if matches_published_087(val_exact_accuracy):
        return f"complete: pass4_trained_post_epoch={post_epoch}_val={val_exact_accuracy:.4f}_matches_published_087"
    return f"complete: pass4_trained_post_epoch={post_epoch}_val={val_exact_accuracy:.4f}_.385_continues"


def build_result_artifact(
    *,
    run_config: Exp4149Config,
    pass3_artifact: Mapping[str, Any],
    seed_state: exp4146.CheckpointState,
    post_state: exp4146.CheckpointState,
    run_result: exp4116.ResumeRunResult,
    val_exact_accuracy: float | None,
    val_metrics_path: Path | None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4149: report final `.384` val and anti-no-op proof."""

    pass3_epoch = _int_or_none(pass3_artifact.get("post_epoch"))
    verdict = _verdict_for_result(
        duration_s=run_result.duration_s,
        pass3_epoch=pass3_epoch,
        post_epoch=post_state.epoch,
        val_exact_accuracy=val_exact_accuracy,
    )
    preliminary = {
        "honest_verdict": verdict,
        "val_exact_accuracy": val_exact_accuracy,
        "duration_s": run_result.duration_s,
        "post_epoch": post_state.epoch,
    }
    trajectory = build_val_trajectory_v384(run_config, pass3_artifact=pass3_artifact, pass4_entry=preliminary)
    artifact = _common_artifact_fields(
        honest_verdict=verdict,
        run_config=run_config,
        val_exact_accuracy=val_exact_accuracy,
        duration_s=run_result.duration_s,
        trajectory=trajectory,
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "native_trainer_launched": True,
            "pass3_artifact_path": str(run_config.pass3_artifact_path),
            "pass3_honest_verdict": pass3_artifact.get("honest_verdict"),
            "pass3_val_exact_accuracy": _rounded(_float_or_none(pass3_artifact.get("val_exact_accuracy"))),
            "pass3_post_epoch": pass3_epoch,
            "seed_checkpoint_state": seed_state.to_dict(),
            "post_checkpoint_state": post_state.to_dict(),
            "post_epoch": post_state.epoch,
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
    """Return explicit schema errors for the Exp 4149 deliverable."""

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
    number = _float_or_none(val)
    if number is None or not 0.0 <= number <= 1.0:
        errors.append("val_exact_accuracy must be numeric between 0 and 1")
    if not isinstance(artifact.get("matches_published_087"), bool):
        errors.append("matches_published_087 must be a bare bool")
    trajectory = artifact.get("val_trajectory_v384")
    if (
        not isinstance(trajectory, list)
        or not trajectory
        or not isinstance(trajectory[0], Mapping)
        or trajectory[0].get("pass_label") != "v384_start"
    ):
        errors.append("val_trajectory_v384 must be a non-empty list with the v384_start row")
    stable_checkpoint_path = artifact.get("stable_checkpoint_path")
    if not isinstance(stable_checkpoint_path, str) or not stable_checkpoint_path:
        errors.append("stable_checkpoint_path must be a non-empty string")
    duration = _float_or_none(artifact.get("duration_s"))
    if duration is None or duration < 0 or duration >= 86_400:
        errors.append("duration_s must be a scalar bounded number below 86400")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(principles.get(field) != text for field, text in FIELD_PRINCIPLES.items()):
        errors.append("field_principles must include the required operator principles")
    if isinstance(verdict, str) and verdict.startswith(("complete:", "success:", "passed:", "shipped:")):
        if not _acceptance_gate(artifact):
            errors.append("complete trained verdict requires duration>120, epoch advance, real val, and a checkpoint")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)


def run_native_pass4(
    config: Exp4149Config,
    *,
    current_epoch: int,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches native trainer.
    """Run the real nano-trm pass4 trainer with progress prints and checkpointing."""

    started = time.time()
    command = build_train_command(config, current_epoch=current_epoch)
    stdout_lines: list[str] = []
    print(f"[exp4149] launching pass4 resume stable={config.stable_checkpoint_path}", flush=True)
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
        print(f"[exp4149:nano-trm] {clean}", flush=True)
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
    pass3_artifact_path: str | Path = DEFAULT_PASS3_ARTIFACT,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    trainer_runner: Callable[[Exp4149Config, int], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4149, or stop honestly when pass3 converged or was a no-op."""

    started = time.time()
    config = Exp4149Config(
        repo_root=repo_root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_root=hydra_run_root,
        pass3_artifact_path=pass3_artifact_path,
        random_seed=random_seed,
    )
    out = Path(output_path)
    checks, blocker = check_preconditions(
        config,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )
    pass3_artifact = load_json_artifact(config.pass3_artifact_path, label="pass3")
    if pass3_is_early_converged(pass3_artifact):
        artifact = build_early_converged_artifact(
            run_config=config,
            pass3_artifact=pass3_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact
    if not pass3_has_real_training(pass3_artifact):
        artifact = build_blocked_pass3_artifact(
            run_config=config,
            pass3_artifact=pass3_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact
    if blocker is not None:
        artifact = build_runtime_blocked_artifact(
            blocker,
            run_config=config,
            pass3_artifact=pass3_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    seed_state = exp4146.read_checkpoint_state(config.stable_checkpoint_path)
    current_epoch = seed_state.epoch
    if current_epoch is None:
        current_epoch = _int_or_none(pass3_artifact.get("post_epoch")) or 0
    try:
        run_result = (
            run_native_pass4(config, current_epoch=current_epoch, checkpoint_loader=checkpoint_loader)
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
        pass3_artifact=pass3_artifact,
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
