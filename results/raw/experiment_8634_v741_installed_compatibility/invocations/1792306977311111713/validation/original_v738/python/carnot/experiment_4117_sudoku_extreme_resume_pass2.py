"""Exp 4117 resumable nano-trm Sudoku Extreme pass 2.

Spec refs: REQ-LEARN-4117, SCENARIO-LEARN-4117,
SCENARIO-LEARN-4117-BLOCKED.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4108_nanotrm_sudoku_extreme_baseline as exp4108
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116


try:  # pragma: no cover - used only inside the native nano-trm subprocess.
    from lightning import Callback
except Exception:  # pragma: no cover - keeps unit imports robust without lightning.
    Callback = object  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4117_sudoku_extreme_resume_pass2.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_EXP4116_ARTIFACT = REPO_ROOT / "results" / exp4116.RESULT_FILENAME
RANDOM_SEED = exp4116.RANDOM_SEED
MAX_TIME = "00:01:00:00"
TERMINAL_PREFIXES = exp4116.TERMINAL_PREFIXES
BLOCKED_PREFIX = exp4116.BLOCKED_PREFIX

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_exact_accuracy",
    "val_delta_vs_pass1",
    "accumulation_stalled",
    "stable_checkpoint_path",
    "duration_s",
    "cumulative_epochs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Honest val report is COMPLETE.",
    "val_exact_accuracy": "Solve metric after pass 2; tracks convergence vs pass 1.",
    "val_delta_vs_pass1": (
        "The improvement from one more pass; the load-bearing signal for whether "
        "continued training will converge or has stalled."
    ),
    "accumulation_stalled": (
        "Bare bool: True if val did not improve -- triggers a config audit instead "
        "of burning more GPU (the accumulate-floor)."
    ),
    "stable_checkpoint_path": "The path pass 3 resumes from.",
    "duration_s": "Cleanly-bounded GPU run < 4800s; > 4800s means the bound failed.",
    "cumulative_epochs": "Total epochs trained across the resume lineage.",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class Exp4117Config:
    """Filesystem and Hydra settings for the pass2 stable resume run."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_dir: Path | str | None = None
    dataset_dir: Path | str | None = None
    pass1_artifact_path: Path | str = DEFAULT_EXP4116_ARTIFACT
    random_seed: int = RANDOM_SEED
    max_time: str = MAX_TIME
    timeout_s: int = 4_700
    progress_every_n_steps: int = 100

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        parent = Path(self.save_parent)
        if parent == DEFAULT_SAVE_PARENT and root != REPO_ROOT:
            parent = root / "results" / "trm_runs"
        nano_root = Path(self.nano_trm_root) if self.nano_trm_root else root / "nano-trm"
        stable = Path(self.stable_dir) if self.stable_dir is not None else parent / "sudoku_extreme_baseline"
        hydra_dir = (
            Path(self.hydra_run_dir)
            if self.hydra_run_dir is not None
            else parent / "experiment_4117_sudoku_extreme_resume_pass2_hydra"
        )
        dataset = (
            Path(self.dataset_dir)
            if self.dataset_dir is not None
            else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        )
        pass1_artifact = Path(self.pass1_artifact_path)
        if pass1_artifact == DEFAULT_EXP4116_ARTIFACT and root != REPO_ROOT:
            pass1_artifact = root / "results" / exp4116.RESULT_FILENAME
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "hydra_run_dir", hydra_dir)
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(self, "pass1_artifact_path", pass1_artifact)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def dataset_builder_path(self) -> Path:
        return Path(self.nano_trm_root) / "scripts" / "data" / "build_sudoku_extreme_dataset.py"

    @property
    def stable_checkpoint_path(self) -> Path:
        return Path(self.stable_dir) / "last.ckpt"


@dataclass(frozen=True)
class Pass1Context:
    """The Exp 4116 baseline used to compute pass2 improvement."""

    artifact_path: Path
    stable_checkpoint_path: Path
    val_exact_accuracy: float | None
    val_source: str | None
    run_dir: Path | None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["artifact_path"] = str(self.artifact_path)
        row["stable_checkpoint_path"] = str(self.stable_checkpoint_path)
        row["run_dir"] = None if self.run_dir is None else str(self.run_dir)
        return row


class NanoTrmResumePass2ProgressPrinter(Callback):  # pragma: no cover - native subprocess only.
    """Lightning callback that prints pass2 progress and refreshes stable last.ckpt."""

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
                f"[exp4117:nano-trm-progress] step={step} "
                f"epoch={getattr(trainer, 'current_epoch', 0)} batch_idx={batch_idx}"
            )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = getattr(trainer, "callback_metrics", {})
        exact = metrics.get("val/exact_accuracy") if isinstance(metrics, Mapping) else None
        exp4116._safe_progress_print(
            f"[exp4117:nano-trm-progress] validation_end "
            f"epoch={getattr(trainer, 'current_epoch', 0)} "
            f"step={getattr(trainer, 'global_step', 0)} "
            f"val_exact_accuracy={exact}"
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.checkpoint_dir / "last.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            exp4116._safe_progress_print(f"[exp4117:nano-trm-progress] checkpoint_saved={checkpoint_path}")
        del pl_module


def find_pass1_artifact(repo_root: str | Path = REPO_ROOT) -> Path:
    """REQ-LEARN-4117: find the Exp 4116 JSON artifact under results/."""

    root = Path(repo_root)
    matches = sorted((root / "results").glob("experiment_4116_*.json"))
    if matches:
        return matches[-1]
    return root / "results" / exp4116.RESULT_FILENAME


def load_pass1_context(path: str | Path) -> Pass1Context:
    """REQ-LEARN-4117: read Exp 4116 stable path and validation baseline."""

    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    stable_value = artifact.get("stable_checkpoint_path")
    stable_path = Path(stable_value) if isinstance(stable_value, str) else DEFAULT_STABLE_DIR / "last.ckpt"
    run_dir_value = artifact.get("run_dir")
    run_dir = Path(run_dir_value) if isinstance(run_dir_value, str) else None

    val = _numeric_or_none(artifact.get("val_exact_accuracy"))
    val_source = str(artifact_path) if val is not None else None
    if val is None and run_dir is not None:
        try:
            exact = exp4116.extract_latest_val_exact_accuracy(run_dir)
        except ValueError:
            exact = None
        if exact is not None:
            val = float(exact.value)
            val_source = str(exact.metrics_path)

    return Pass1Context(
        artifact_path=artifact_path,
        stable_checkpoint_path=stable_path,
        val_exact_accuracy=val,
        val_source=val_source,
        run_dir=run_dir,
    )


def check_preconditions(
    *,
    repo_root: str | Path = REPO_ROOT,
    stable_dir: str | Path = DEFAULT_STABLE_DIR,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """REQ-LEARN-4117: verify uv, trainer, CUDA, and stable directory."""

    return exp4116.check_preconditions(
        repo_root=repo_root,
        stable_dir=stable_dir,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
    )


def check_stable_checkpoint(
    config: Exp4117Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4107.PreconditionCheck:
    """SCENARIO-LEARN-4117-BLOCKED: verify the pass1 stable checkpoint."""

    checkpoint_path = config.stable_checkpoint_path
    if not checkpoint_path.exists():
        return exp4107.PreconditionCheck("stable_checkpoint", False, f"missing: {checkpoint_path}")
    ok, detail = checkpoint_loader(checkpoint_path)
    return exp4107.PreconditionCheck("stable_checkpoint", ok, detail)


def build_train_command(config: Exp4117Config) -> list[str]:
    """REQ-LEARN-4117: build the bounded native pass2 resume command."""

    return [
        "uv",
        "run",
        "python",
        "src/nn/train.py",
        "experiment=trm_sudoku_extreme_1k_aug_1k",
        "logger=csv",
        f"hydra.run.dir={Path(config.hydra_run_dir)}",
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
        (
            "+callbacks.exp4117_progress._target_="
            "carnot.experiment_4117_sudoku_extreme_resume_pass2."
            "NanoTrmResumePass2ProgressPrinter"
        ),
        f"+callbacks.exp4117_progress.every_n_steps={int(config.progress_every_n_steps)}",
        f"+callbacks.exp4117_progress.checkpoint_dir={Path(config.stable_dir)}",
    ]


def run_native_resume_training(
    config: Exp4117Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches trainer.
    """Run the real pass2 native nano-trm trainer with a Lightning time bound."""

    started = time.time()
    command = build_train_command(config)
    stdout_lines: list[str] = []
    print(f"[exp4117] launching bounded native resume stable={config.stable_checkpoint_path}", flush=True)
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(config.nano_trm_root),
            env=exp4116.build_train_env(config),
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
            run_dir=Path(config.hydra_run_dir),
        )

    assert proc.stdout is not None
    timed_out = False
    for line in proc.stdout:
        clean = line.rstrip()
        stdout_lines.append(clean)
        print(f"[exp4117:nano-trm] {clean}", flush=True)
        if time.time() - started > config.timeout_s:
            proc.kill()
            stdout_lines.append(f"timeout_s exceeded: {config.timeout_s}")
            timed_out = True
            break
    return_code = proc.wait()
    if timed_out and return_code == 0:
        return_code = 124
    return exp4116.verify_completed_resume_run(
        config,
        duration_s=time.time() - started,
        return_code=return_code,
        command=command,
        stdout_tail=stdout_lines,
        checkpoint_loader=checkpoint_loader,
    )


def _artifact_common(
    *,
    honest_verdict: str,
    val_exact_accuracy: float | None,
    val_delta_vs_pass1: float | None,
    accumulation_stalled: bool,
    cumulative_epochs: int | None,
    stable_checkpoint_path: Path,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4117_sudoku_extreme_resume_pass2",
        "schema": "carnot.experiment_4117_sudoku_extreme_resume_pass2.v1",
        "honest_verdict": honest_verdict,
        "val_exact_accuracy": val_exact_accuracy,
        "val_delta_vs_pass1": val_delta_vs_pass1,
        "accumulation_stalled": bool(accumulation_stalled),
        "cumulative_epochs": cumulative_epochs,
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": ["REQ-LEARN-4117", "SCENARIO-LEARN-4117"],
    }


def build_result_artifact(
    *,
    run_config: Exp4117Config,
    run_result: exp4116.ResumeRunResult,
    pass1_context: Pass1Context,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    dataset_generated: bool,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4117: build the measured pass2 accumulation artifact."""

    exact = run_result.val_exact_accuracy
    exact_value = None if exact is None else float(exact.value)
    pass1_value = pass1_context.val_exact_accuracy
    delta = None if exact_value is None or pass1_value is None else exact_value - pass1_value
    stalled = bool(delta is not None and delta <= 0.0)
    if exact_value is None:
        verdict = "complete: missing_real_val_exact_accuracy"
        if run_result.return_code != 0:
            verdict = f"complete: nanotrm_resume_pass2_failed_return_code_{run_result.return_code}"
    elif delta is None:
        verdict = f"complete: val={exact_value:.4f} pass1_delta_unavailable_config_audit_recommended"
    elif stalled:
        verdict = (
            f"complete: val={exact_value:.4f} delta={delta:.4f} "
            "accumulation_stalled_config_audit_recommended"
        )
    else:
        verdict = f"complete: val={exact_value:.4f} delta={delta:.4f} improved"

    acceptance_gate_passed = bool(
        exact_value is not None
        and run_result.duration_s < 4_800
        and delta is not None
        and (delta > 0.0 or stalled)
    )
    artifact = _artifact_common(
        honest_verdict=verdict,
        val_exact_accuracy=exact_value,
        val_delta_vs_pass1=delta,
        accumulation_stalled=stalled,
        cumulative_epochs=run_result.cumulative_epochs,
        stable_checkpoint_path=run_result.stable_checkpoint_path,
        duration_s=run_result.duration_s,
    )
    artifact.update(
        {
            "acceptance_gate_passed": acceptance_gate_passed,
            "checkpoint_reload_ok": bool(run_result.checkpoint_reload_ok),
            "checkpoint_reload_detail": run_result.checkpoint_reload_detail,
            "exact_accuracy_metric": None if exact is None else exact.metric_name,
            "exact_accuracy_metrics_path": None if exact is None else str(exact.metrics_path),
            "pass1": pass1_context.to_dict(),
            "pass1_val_exact_accuracy": pass1_value,
            "pass1_val_source": pass1_context.val_source,
            "return_code": int(run_result.return_code),
            "run_dir": str(run_result.run_dir),
            "dataset_dir": str(run_config.dataset_dir),
            "dataset_generated": bool(dataset_generated),
            "preconditions_checked": [
                check.to_dict() if isinstance(check, exp4107.PreconditionCheck) else dict(check)
                for check in preconditions_checked
            ],
            "command": list(run_result.command),
            "stdout_tail": list(run_result.stdout_tail[-60:]),
        }
    )
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    stable_checkpoint_path: str | Path = DEFAULT_STABLE_DIR / "last.ckpt",
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4117-BLOCKED: build a no-fabrication blocked artifact."""

    artifact = _artifact_common(
        honest_verdict=reason,
        val_exact_accuracy=None,
        val_delta_vs_pass1=None,
        accumulation_stalled=False,
        cumulative_epochs=None,
        stable_checkpoint_path=Path(stable_checkpoint_path),
        duration_s=duration_s,
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "checkpoint_reload_ok": False,
            "checkpoint_reload_detail": "not attempted",
            "exact_accuracy_metric": None,
            "exact_accuracy_metrics_path": None,
            "pass1": None,
            "pass1_val_exact_accuracy": None,
            "pass1_val_source": None,
            "return_code": None,
            "run_dir": None,
            "dataset_dir": None,
            "dataset_generated": False,
            "preconditions_checked": [
                check.to_dict() if isinstance(check, exp4107.PreconditionCheck) else dict(check)
                for check in preconditions_checked
            ],
            "command": [],
            "stdout_tail": [],
        }
    )
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4117 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith((*TERMINAL_PREFIXES, BLOCKED_PREFIX)):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    exact = artifact.get("val_exact_accuracy")
    if exact is not None:
        if not isinstance(exact, (int, float)) or isinstance(exact, bool):
            errors.append("val_exact_accuracy must be numeric or null")
        elif not 0.0 <= float(exact) <= 1.0:
            errors.append("val_exact_accuracy must be between 0 and 1")

    delta = artifact.get("val_delta_vs_pass1")
    if delta is not None and (not isinstance(delta, (int, float)) or isinstance(delta, bool)):
        errors.append("val_delta_vs_pass1 must be numeric or null")

    if not isinstance(artifact.get("accumulation_stalled"), bool):
        errors.append("accumulation_stalled must be a bare bool")

    cumulative_epochs = artifact.get("cumulative_epochs")
    if cumulative_epochs is not None:
        if (
            not isinstance(cumulative_epochs, int)
            or isinstance(cumulative_epochs, bool)
            or cumulative_epochs < 0
        ):
            errors.append("cumulative_epochs must be a non-negative int or null")

    stable_checkpoint_path = artifact.get("stable_checkpoint_path")
    if not isinstance(stable_checkpoint_path, str) or not stable_checkpoint_path.endswith(
        "results/trm_runs/sudoku_extreme_baseline/last.ckpt"
    ):
        errors.append("stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path")

    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("duration_s must be numeric")

    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if gate is True:
        exact_is_number = isinstance(exact, (int, float)) and not isinstance(exact, bool)
        if not exact_is_number:
            errors.append("accepted artifact requires val_exact_accuracy")
        stalled = artifact.get("accumulation_stalled")
        has_positive_delta = isinstance(delta, (int, float)) and not isinstance(delta, bool) and delta > 0.0
        if not (has_positive_delta or stalled is True):
            errors.append("accepted artifact requires positive delta or accumulation_stalled true")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 4_800:
            errors.append("accepted artifact requires duration_s < 4800")
        if not (
            isinstance(stable_checkpoint_path, str)
            and stable_checkpoint_path.endswith(".ckpt")
            and Path(stable_checkpoint_path).exists()
        ):
            errors.append("accepted artifact requires existing stable checkpoint")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    save_parent: str | Path = DEFAULT_SAVE_PARENT,
    stable_dir: str | Path | None = None,
    hydra_run_dir: str | Path | None = None,
    pass1_artifact_path: str | Path | None = None,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    dataset_builder: Callable[[Exp4117Config], object] = exp4116.generate_sudoku_extreme_dataset_if_missing,
    trainer_runner: Callable[[Exp4117Config], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4117 or write an honest blocked artifact."""

    started = time.time()
    root = Path(repo_root)
    pass1_path = Path(pass1_artifact_path) if pass1_artifact_path is not None else find_pass1_artifact(root)
    try:
        pass1_context = load_pass1_context(pass1_path)
    except (FileNotFoundError, json.JSONDecodeError):
        pass1_context = Pass1Context(
            artifact_path=pass1_path,
            stable_checkpoint_path=(root / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"),
            val_exact_accuracy=None,
            val_source=None,
            run_dir=None,
        )
    stable_parent = stable_dir if stable_dir is not None else pass1_context.stable_checkpoint_path.parent
    config = Exp4117Config(
        repo_root=root,
        save_parent=save_parent,
        stable_dir=stable_parent,
        hydra_run_dir=hydra_run_dir,
        pass1_artifact_path=pass1_path,
        random_seed=random_seed,
    )
    checks, blocker = check_preconditions(
        repo_root=config.repo_root,
        stable_dir=config.stable_dir,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
    )
    out = Path(output_path)
    if blocker is not None:
        artifact = build_blocked_artifact(
            blocker,
            preconditions_checked=checks,
            stable_checkpoint_path=config.stable_checkpoint_path,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    stable_check = check_stable_checkpoint(config, checkpoint_loader=checkpoint_loader)
    checks.append(stable_check)
    if not stable_check.available:
        artifact = build_blocked_artifact(
            "blocked_stable_checkpoint_missing",
            preconditions_checked=checks,
            stable_checkpoint_path=config.stable_checkpoint_path,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    dataset_generated = False
    if not exp4108.dataset_is_complete(config.dataset_dir):
        dataset_builder(config)
        dataset_generated = True
    if not exp4108.dataset_is_complete(config.dataset_dir):
        artifact = build_blocked_artifact(
            "blocked_dataset_missing",
            preconditions_checked=checks,
            stable_checkpoint_path=config.stable_checkpoint_path,
            duration_s=time.time() - started,
        )
        artifact["dataset_dir"] = str(config.dataset_dir)
        artifact["dataset_generated"] = dataset_generated
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    if trainer_runner is None:  # pragma: no cover - launches the native trainer.
        run_result = run_native_resume_training(config, checkpoint_loader=checkpoint_loader)
    else:
        run_result = trainer_runner(config)

    artifact = build_result_artifact(
        run_config=config,
        run_result=run_result,
        pass1_context=pass1_context,
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
