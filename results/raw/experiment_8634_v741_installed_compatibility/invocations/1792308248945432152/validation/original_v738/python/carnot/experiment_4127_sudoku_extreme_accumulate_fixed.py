"""Exp 4127 fixed-LR nano-trm Sudoku Extreme accumulation.

Spec refs: REQ-LEARN-4127, SCENARIO-LEARN-4127,
SCENARIO-LEARN-4127-BLOCKED.
"""

from __future__ import annotations

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
from carnot import experiment_4126_lr_resume_correctness_fix as exp4126


try:  # pragma: no cover - used only inside the native nano-trm subprocess.
    from lightning import Callback
except Exception:  # pragma: no cover - keeps unit imports robust without lightning.
    Callback = object  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4127_sudoku_extreme_accumulate_fixed.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_LR_FIX_ARTIFACT = REPO_ROOT / "results" / exp4126.RESULT_FILENAME
RANDOM_SEED = exp4108.RANDOM_SEED
MAX_TIME = "00:01:00:00"
MAX_PASSES = 2
LOCAL_SAFE_BATCH_SIZE = 128
PUBLISHED_EXACT_ACCURACY = 0.87
PUBLISHED_TOLERANCE = 0.02
V381_REFERENCE_DELTA = 0.01
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
CONTIGUOUS_RUN_RECOMMENDATION = (
    "Use one contiguous nano-trm Sudoku Extreme baseline run from a clean stable "
    "checkpoint instead of burning more bounded resumes that restart LR warmup."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_trajectory",
    "matches_published_087",
    "per_pass_delta_vs_v381",
    "stable_checkpoint_path",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'val=0.NN, faster than .381 but not yet "
        "0.87 -> .383 continues' is COMPLETE."
    ),
    "val_trajectory": (
        "Val across this milestone's passes; the load-bearing evidence the fixed "
        "schedule accumulates."
    ),
    "matches_published_087": (
        "Bare bool: within 0.02 of 0.87. Tells exp4128 whether a faithful "
        "baseline exists to graft onto."
    ),
    "per_pass_delta_vs_v381": (
        "Did the corrected schedule beat .381's ~+1pp/pass? Confirms the LR fix "
        "mattered (vs the convergence being intrinsically slow)."
    ),
    "stable_checkpoint_path": "The persisted baseline checkpoint exp4128 and .383 build on.",
    "duration_s": "Bounded GPU runs < 4800s each.",
}


@dataclass(frozen=True)
class Exp4127Config:
    """Filesystem and Hydra settings for the fixed-LR accumulation passes."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_root: Path | str | None = None
    dataset_dir: Path | str | None = None
    lr_artifact_path: Path | str = DEFAULT_LR_FIX_ARTIFACT
    random_seed: int = RANDOM_SEED
    max_time: str = MAX_TIME
    timeout_s: int = 3_700
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
            Path(self.hydra_run_root)
            if self.hydra_run_root is not None
            else parent / "experiment_4127_sudoku_extreme_accumulate_fixed"
        )
        dataset = (
            Path(self.dataset_dir)
            if self.dataset_dir is not None
            else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        )
        lr_artifact = Path(self.lr_artifact_path)
        if lr_artifact == DEFAULT_LR_FIX_ARTIFACT and root != REPO_ROOT:
            lr_artifact = root / "results" / exp4126.RESULT_FILENAME
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "hydra_run_root", hydra_root)
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(self, "lr_artifact_path", lr_artifact)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def stable_checkpoint_path(self) -> Path:
        return Path(self.stable_dir) / "last.ckpt"

    def pass_run_dir(self, pass_index: int) -> Path:
        return Path(self.hydra_run_root) / f"pass_{int(pass_index)}_hydra"


@dataclass(frozen=True)
class StartingVal:
    """Validation exact accuracy known before Exp 4127 starts training."""

    val_exact_accuracy: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NanoTrmAccumulateFixedProgressPrinter(Callback):  # pragma: no cover
    """Lightning callback that prints progress and refreshes the stable checkpoint."""

    def __init__(self, every_n_steps: int = 100, checkpoint_dir: str | None = None) -> None:
        self.every_n_steps = max(int(every_n_steps), 1)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def on_train_batch_end(
        self,
        trainer: Any,
        _pl_module: Any,
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(getattr(trainer, "global_step", 0))
        if step > 0 and step % self.every_n_steps == 0:
            exp4116._safe_progress_print(
                f"[exp4127:nano-trm-progress] step={step} "
                f"epoch={getattr(trainer, 'current_epoch', 0)} batch_idx={batch_idx}"
            )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = getattr(trainer, "callback_metrics", {})
        exact = metrics.get("val/exact_accuracy") if isinstance(metrics, Mapping) else None
        exp4116._safe_progress_print(
            f"[exp4127:nano-trm-progress] validation_end "
            f"epoch={getattr(trainer, 'current_epoch', 0)} "
            f"step={getattr(trainer, 'global_step', 0)} "
            f"val_exact_accuracy={exact}"
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.checkpoint_dir / "last.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            exp4116._safe_progress_print(f"[exp4127:nano-trm-progress] checkpoint_saved={checkpoint_path}")
        del pl_module


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _rounded(value: float | None, digits: int = 12) -> float | None:
    return None if value is None else round(float(value), digits)


def _checks_to_dicts(checks: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [check.to_dict() if isinstance(check, exp4107.PreconditionCheck) else dict(check) for check in checks]


def find_lr_fix_artifact(repo_root: str | Path = REPO_ROOT) -> Path:
    """REQ-LEARN-4127: find the latest Exp 4126 LR-continuity artifact."""

    root = Path(repo_root)
    matches = sorted((root / "results").glob("experiment_4126_*.json"))
    if matches:
        return matches[-1]
    return root / "results" / exp4126.RESULT_FILENAME


def load_lr_fix_artifact(path: str | Path) -> dict[str, Any]:
    """REQ-LEARN-4127: read the LR-fix gate and fail closed if absent."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def lr_fix_landed(artifact: Mapping[str, Any]) -> bool:
    """REQ-LEARN-4127: only a bare true LR-continuity gate permits training."""

    return artifact.get("lr_continuous_across_resume") is True


def load_starting_val(repo_root: str | Path = REPO_ROOT) -> StartingVal | None:
    """SCENARIO-LEARN-4127: use the latest comparable pre-4127 validation value."""

    root = Path(repo_root)
    candidates = (
        "experiment_4118_sudoku_extreme_resume_pass3.json",
        "experiment_4117_sudoku_extreme_resume_pass2.json",
        "experiment_4116_sudoku_extreme_resume_pass1.json",
    )
    for name in candidates:
        path = root / "results" / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        value = _float_or_none(payload.get("val_exact_accuracy"))
        if value is not None:
            return StartingVal(val_exact_accuracy=value, source=str(path))
    return None


def matches_published_087(value: float | None) -> bool:
    """REQ-LEARN-4127: compare final validation accuracy to 0.87 within 0.02."""

    return value is not None and abs(float(value) - PUBLISHED_EXACT_ACCURACY) <= PUBLISHED_TOLERANCE + 1e-12


def build_train_command(config: Exp4127Config, pass_index: int) -> list[str]:
    """REQ-LEARN-4127: build one fixed-LR bounded resume command."""

    return [
        "uv",
        "run",
        "python",
        "src/nn/train.py",
        "experiment=trm_sudoku_extreme_1k_aug_1k",
        "logger=csv",
        f"hydra.run.dir={config.pass_run_dir(pass_index)}",
        "save_dir=null",
        "append_wandb_name_to_save_dir=false",
        f"seed={int(config.random_seed)}",
        "data.data_dir=./data/sudoku_extreme_1k_aug_1k",
        f"timekeeping.batch_size={int(config.batch_size)}",
        f"ckpt_path={config.stable_checkpoint_path}",
        f"+trainer.max_time={config.max_time}",
        "callbacks.model_checkpoint.monitor=val/exact_accuracy",
        "callbacks.model_checkpoint.mode=max",
        f"callbacks.model_checkpoint.dirpath={Path(config.stable_dir)}",
        "callbacks.model_checkpoint.save_last=true",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.auto_insert_metric_name=false",
        (
            "+callbacks.exp4127_progress._target_="
            "carnot.experiment_4127_sudoku_extreme_accumulate_fixed."
            "NanoTrmAccumulateFixedProgressPrinter"
        ),
        f"+callbacks.exp4127_progress.every_n_steps={int(config.progress_every_n_steps)}",
        f"+callbacks.exp4127_progress.checkpoint_dir={Path(config.stable_dir)}",
    ]


def build_train_env(config: Exp4127Config) -> dict[str, str]:
    """REQ-LEARN-4127: disable compile/CUDAGraph resume while keeping CUDA enabled."""

    env = exp4116.build_train_env(config)
    env["DISABLE_COMPILE"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    return env


def check_stable_checkpoint(
    config: Exp4127Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4107.PreconditionCheck:
    """SCENARIO-LEARN-4127: verify the shared stable checkpoint before training."""

    checkpoint_path = config.stable_checkpoint_path
    if not checkpoint_path.exists():
        return exp4107.PreconditionCheck("stable_checkpoint", False, f"missing: {checkpoint_path}")
    ok, detail = checkpoint_loader(checkpoint_path)
    return exp4107.PreconditionCheck("stable_checkpoint", ok, detail)


def verify_completed_resume_pass(
    config: Exp4127Config,
    pass_index: int,
    *,
    duration_s: float,
    return_code: int = 0,
    command: Sequence[str] | None = None,
    stdout_tail: Sequence[str] | None = None,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:
    """SCENARIO-LEARN-4127: verify one pass's checkpoint and exact-accuracy metric."""

    checkpoint_path = config.stable_checkpoint_path
    if checkpoint_path.exists():
        reload_ok, reload_detail = checkpoint_loader(checkpoint_path)
    else:
        reload_ok = False
        reload_detail = f"missing stable checkpoint: {checkpoint_path}"
    run_dir = config.pass_run_dir(pass_index)
    try:
        exact = exp4116.extract_latest_val_exact_accuracy(run_dir)
        metric_detail = f"{exact.metric_name}={exact.value}"
    except ValueError as exc:
        exact = None
        metric_detail = str(exc)
    cumulative_epochs = exp4116.extract_cumulative_epochs(run_dir)
    lines = list(stdout_tail or [])
    lines.extend(
        [
            f"return_code={return_code}",
            f"pass_index={pass_index}",
            f"run_dir={run_dir}",
            f"stable_checkpoint={checkpoint_path}",
            f"checkpoint_reload={reload_detail}",
            f"val_exact_accuracy={metric_detail}",
            f"cumulative_epochs={cumulative_epochs}",
        ]
    )
    return exp4116.ResumeRunResult(
        return_code=int(return_code),
        stable_checkpoint_path=checkpoint_path,
        checkpoint_reload_ok=reload_ok,
        checkpoint_reload_detail=reload_detail,
        val_exact_accuracy=exact,
        cumulative_epochs=cumulative_epochs,
        duration_s=float(duration_s),
        command=list(command or build_train_command(config, pass_index)),
        stdout_tail=lines[-60:],
        run_dir=run_dir,
    )


def run_native_resume_pass(
    config: Exp4127Config,
    pass_index: int,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches native trainer.
    """Run one native nano-trm fixed-LR resume pass with a one-hour trainer cap."""

    started = time.time()
    command = build_train_command(config, pass_index)
    stdout_lines: list[str] = []
    print(
        f"[exp4127] launching bounded native resume pass={pass_index} "
        f"stable={config.stable_checkpoint_path}",
        flush=True,
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
            run_dir=config.pass_run_dir(pass_index),
        )

    assert proc.stdout is not None
    timed_out = False
    for line in proc.stdout:
        clean = line.rstrip()
        stdout_lines.append(clean)
        print(f"[exp4127:nano-trm] {clean}", flush=True)
        if time.time() - started > config.timeout_s:
            proc.kill()
            stdout_lines.append(f"timeout_s exceeded: {config.timeout_s}")
            timed_out = True
            break
    return_code = proc.wait()
    if timed_out and return_code == 0:
        return_code = 124
    return verify_completed_resume_pass(
        config,
        pass_index,
        duration_s=time.time() - started,
        return_code=return_code,
        command=command,
        stdout_tail=stdout_lines,
        checkpoint_loader=checkpoint_loader,
    )


def _trajectory(
    starting_val: StartingVal | None,
    run_results: Sequence[exp4116.ResumeRunResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = None
    if starting_val is not None:
        previous = starting_val.val_exact_accuracy
        rows.append(
            {
                "pass_index": 0,
                "kind": "starting_baseline",
                "source": starting_val.source,
                "val_exact_accuracy": _rounded(starting_val.val_exact_accuracy),
                "delta_vs_previous": None,
            }
        )
    for pass_index, result in enumerate(run_results, start=1):
        exact = result.val_exact_accuracy
        value = None if exact is None else float(exact.value)
        delta = None if value is None or previous is None else value - previous
        rows.append(
            {
                "pass_index": pass_index,
                "kind": "fixed_lr_resume_pass",
                "source": None if exact is None else str(exact.metrics_path),
                "val_exact_accuracy": _rounded(value),
                "delta_vs_previous": _rounded(delta),
                "duration_s": round(float(result.duration_s), 3),
                "return_code": int(result.return_code),
                "checkpoint_reload_ok": bool(result.checkpoint_reload_ok),
            }
        )
        if value is not None:
            previous = value
    return rows


def _delta_comparison(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    reference_delta: float = V381_REFERENCE_DELTA,
) -> dict[str, Any]:
    deltas = [
        float(row["delta_vs_previous"])
        for row in trajectory
        if row.get("pass_index") != 0 and _float_or_none(row.get("delta_vs_previous")) is not None
    ]
    mean_delta = None if not deltas else round(sum(deltas) / len(deltas), 12)
    beats = mean_delta is not None and mean_delta > reference_delta
    comparison = "no_measured_delta"
    if mean_delta is not None:
        comparison = "faster_than_v381" if beats else "not_faster_than_v381"
    return {
        "reference_delta": reference_delta,
        "deltas": [_rounded(delta) for delta in deltas],
        "mean_delta": mean_delta,
        "beats_v381": bool(beats),
        "comparison": comparison,
    }


def _final_val(trajectory: Sequence[Mapping[str, Any]]) -> float | None:
    for row in reversed(trajectory):
        if row.get("pass_index") == 0:
            continue
        value = _float_or_none(row.get("val_exact_accuracy"))
        if value is not None:
            return value
    return None


def _verdict_for_final(final_val: float | None, comparison: Mapping[str, Any]) -> str:
    if final_val is None:
        return "complete: missing_real_val_trajectory"
    if matches_published_087(final_val):
        return f"complete: val={final_val:.4f} reproduced_within_0.02_of_0.87"
    if comparison.get("beats_v381") is True:
        return f"complete: val={final_val:.4f} faster_than_.381_but_not_yet_0.87 -> .383 continues"
    return f"complete: val={final_val:.4f} not_faster_than_.381_and_not_yet_0.87 -> .383 continues"


def _result_to_dict(result: exp4116.ResumeRunResult, pass_index: int) -> dict[str, Any]:
    exact = result.val_exact_accuracy
    return {
        "pass_index": pass_index,
        "return_code": int(result.return_code),
        "stable_checkpoint_path": str(result.stable_checkpoint_path),
        "checkpoint_reload_ok": bool(result.checkpoint_reload_ok),
        "checkpoint_reload_detail": result.checkpoint_reload_detail,
        "val_exact_accuracy": None if exact is None else float(exact.value),
        "exact_accuracy_metric": None if exact is None else exact.metric_name,
        "exact_accuracy_metrics_path": None if exact is None else str(exact.metrics_path),
        "cumulative_epochs": result.cumulative_epochs,
        "duration_s": round(float(result.duration_s), 3),
        "command": list(result.command),
        "stdout_tail": list(result.stdout_tail[-60:]),
        "run_dir": str(result.run_dir),
    }


def build_result_artifact(
    *,
    run_config: Exp4127Config,
    lr_fix_artifact: Mapping[str, Any],
    starting_val: StartingVal | None,
    run_results: Sequence[exp4116.ResumeRunResult],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    dataset_generated: bool,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4127: build the measured fixed-LR accumulation artifact."""

    trajectory = _trajectory(starting_val, run_results)
    comparison = _delta_comparison(trajectory)
    final_val = _final_val(trajectory)
    matches = matches_published_087(final_val)
    durations = [round(float(result.duration_s), 3) for result in run_results]
    artifact = {
        "experiment": "experiment_4127_sudoku_extreme_accumulate_fixed",
        "schema": "carnot.experiment_4127_sudoku_extreme_accumulate_fixed.v1",
        "spec_refs": ["REQ-LEARN-4127", "SCENARIO-LEARN-4127"],
        "honest_verdict": _verdict_for_final(final_val, comparison),
        "val_trajectory": trajectory,
        "matches_published_087": bool(matches),
        "per_pass_delta_vs_v381": comparison,
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "duration_s": durations,
        "total_duration_s": round(sum(durations), 3),
        "acceptance_gate_passed": bool(final_val is not None),
        "field_principles": dict(FIELD_PRINCIPLES),
        "lr_fix_artifact": dict(lr_fix_artifact),
        "starting_val": None if starting_val is None else starting_val.to_dict(),
        "pass_results": [_result_to_dict(result, index) for index, result in enumerate(run_results, start=1)],
        "dataset_dir": str(run_config.dataset_dir),
        "dataset_generated": bool(dataset_generated),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "contiguous_run_recommendation": None,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_DIR / "last.ckpt",
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]] = (),
    duration_s: Sequence[float] | float = (),
) -> dict[str, Any]:
    """REQ-LEARN-4127: build a no-training blocked artifact."""

    durations = [round(float(duration_s), 3)] if isinstance(duration_s, (int, float)) else list(duration_s)
    artifact = {
        "experiment": "experiment_4127_sudoku_extreme_accumulate_fixed",
        "schema": "carnot.experiment_4127_sudoku_extreme_accumulate_fixed.v1",
        "spec_refs": ["REQ-LEARN-4127", "SCENARIO-LEARN-4127-BLOCKED"],
        "honest_verdict": reason,
        "val_trajectory": [],
        "matches_published_087": False,
        "per_pass_delta_vs_v381": _delta_comparison([]),
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "duration_s": durations,
        "total_duration_s": round(sum(float(item) for item in durations), 3),
        "acceptance_gate_passed": reason == "blocked_lr_fix_not_landed",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "contiguous_run_recommendation": CONTIGUOUS_RUN_RECOMMENDATION
        if reason == "blocked_lr_fix_not_landed"
        else None,
    }
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4127 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    if not isinstance(artifact.get("val_trajectory"), list):
        errors.append("val_trajectory must be a list")

    if not isinstance(artifact.get("matches_published_087"), bool):
        errors.append("matches_published_087 must be a bare bool")

    comparison = artifact.get("per_pass_delta_vs_v381")
    if not isinstance(comparison, Mapping) or not isinstance(comparison.get("beats_v381"), bool):
        errors.append("per_pass_delta_vs_v381.beats_v381 must be a bare bool")

    stable_checkpoint_path = artifact.get("stable_checkpoint_path")
    if not isinstance(stable_checkpoint_path, str) or not stable_checkpoint_path.endswith(
        "results/trm_runs/sudoku_extreme_baseline/last.ckpt"
    ):
        errors.append("stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path")

    durations = artifact.get("duration_s")
    duration_values: list[Any]
    if isinstance(durations, list):
        duration_values = durations
    elif isinstance(durations, (int, float)) and not isinstance(durations, bool):
        duration_values = [durations]
    else:
        duration_values = [None]
    for value in duration_values:
        number = _float_or_none(value)
        if number is None or number < 0 or number >= 4_800:
            errors.append("each duration_s entry must be a bounded number below 4800")
            break

    if artifact.get("matches_published_087") is True:
        final_val = _final_val(artifact.get("val_trajectory", []))
        if not matches_published_087(final_val):
            errors.append("matches_published_087=true requires a final val within 0.02 of 0.87")

    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    save_parent: str | Path = DEFAULT_SAVE_PARENT,
    stable_dir: str | Path | None = None,
    hydra_run_root: str | Path | None = None,
    lr_artifact_path: str | Path | None = None,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    dataset_builder: Callable[[Exp4127Config], object] = exp4116.generate_sudoku_extreme_dataset_if_missing,
    trainer_runner: Callable[[Exp4127Config, int], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
    max_passes: int = MAX_PASSES,
) -> dict[str, Any]:
    """Run Exp 4127 or write the required honest blocked artifact."""

    started = time.time()
    root = Path(repo_root)
    lr_path = Path(lr_artifact_path) if lr_artifact_path is not None else find_lr_fix_artifact(root)
    config = Exp4127Config(
        repo_root=root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_root=hydra_run_root,
        lr_artifact_path=lr_path,
        random_seed=random_seed,
    )
    out = Path(output_path)
    lr_artifact = load_lr_fix_artifact(lr_path)
    if not lr_fix_landed(lr_artifact):
        stable_value = lr_artifact.get("stable_checkpoint_path")
        stable_path = Path(stable_value) if isinstance(stable_value, str) else config.stable_checkpoint_path
        artifact = build_blocked_artifact(
            "blocked_lr_fix_not_landed",
            stable_checkpoint_path=stable_path,
            preconditions_checked=[],
            duration_s=[],
        )
        artifact["lr_fix_artifact"] = lr_artifact
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    checks, blocker = exp4116.check_preconditions(
        repo_root=config.repo_root,
        stable_dir=config.stable_dir,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            blocker,
            stable_checkpoint_path=config.stable_checkpoint_path,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        artifact["lr_fix_artifact"] = lr_artifact
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    stable_check = check_stable_checkpoint(config, checkpoint_loader=checkpoint_loader)
    checks.append(stable_check)
    if not stable_check.available:
        artifact = build_blocked_artifact(
            "blocked_stable_checkpoint_missing",
            stable_checkpoint_path=config.stable_checkpoint_path,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        artifact["lr_fix_artifact"] = lr_artifact
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    dataset_generated = False
    if not exp4108.dataset_is_complete(config.dataset_dir):
        dataset_builder(config)
        dataset_generated = True
    if not exp4108.dataset_is_complete(config.dataset_dir):
        artifact = build_blocked_artifact(
            "blocked_dataset_missing",
            stable_checkpoint_path=config.stable_checkpoint_path,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        artifact["lr_fix_artifact"] = lr_artifact
        artifact["dataset_dir"] = str(config.dataset_dir)
        artifact["dataset_generated"] = dataset_generated
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    starting_val = load_starting_val(root)
    run_results: list[exp4116.ResumeRunResult] = []
    passes_to_run = max(0, min(int(max_passes), MAX_PASSES))
    for pass_index in range(1, passes_to_run + 1):
        try:
            if trainer_runner is None:  # pragma: no cover - launches native trainer.
                result = run_native_resume_pass(config, pass_index, checkpoint_loader=checkpoint_loader)
            else:
                result = trainer_runner(config, pass_index)
        except Exception as exc:
            result = exp4116.ResumeRunResult(
                return_code=1,
                stable_checkpoint_path=config.stable_checkpoint_path,
                checkpoint_reload_ok=False,
                checkpoint_reload_detail=f"{type(exc).__name__}: {exc}",
                val_exact_accuracy=None,
                cumulative_epochs=None,
                duration_s=time.time() - started,
                command=build_train_command(config, pass_index),
                stdout_tail=[f"{type(exc).__name__}: {exc}"],
                run_dir=config.pass_run_dir(pass_index),
            )
        run_results.append(result)
        exact_value = None if result.val_exact_accuracy is None else float(result.val_exact_accuracy.value)
        if exact_value is None or matches_published_087(exact_value):
            break

    artifact = build_result_artifact(
        run_config=config,
        lr_fix_artifact=lr_artifact,
        starting_val=starting_val,
        run_results=run_results,
        preconditions_checked=checks,
        dataset_generated=dataset_generated,
    )
    _write_json(out, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps({field: artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS}, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
