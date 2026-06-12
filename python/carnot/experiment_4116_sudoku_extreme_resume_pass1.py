"""Exp 4116 resumable nano-trm Sudoku Extreme pass 1.

Spec refs: REQ-LEARN-4116, SCENARIO-LEARN-4116,
SCENARIO-LEARN-4116-BLOCKED.
"""

from __future__ import annotations

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
from carnot import experiment_4108_nanotrm_sudoku_extreme_baseline as exp4108


try:  # pragma: no cover - used only inside the native nano-trm subprocess.
    from lightning import Callback
except Exception:  # pragma: no cover - keeps unit imports robust without lightning.
    Callback = object  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4116_sudoku_extreme_resume_pass1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_EXP4108_ARTIFACT = REPO_ROOT / "results" / exp4108.RESULT_FILENAME
RANDOM_SEED = exp4108.RANDOM_SEED
PUBLISHED_EXACT_ACCURACY = 0.87
MAX_TIME = "00:01:00:00"
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
BLOCKED_PREFIX = "blocked_"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_exact_accuracy",
    "cumulative_epochs",
    "stable_checkpoint_path",
    "checkpoint_reload_ok",
    "duration_s",
    "random_seed",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'val=0.NN still below 0.87' is a COMPLETE "
        "verdict; report the real number."
    ),
    "val_exact_accuracy": (
        "The REAL solve metric after this pass; the load-bearing number tracking "
        "convergence toward 0.87."
    ),
    "cumulative_epochs": (
        "Total epochs trained across the resume lineage; lets a reader judge "
        "whether more passes will converge."
    ),
    "stable_checkpoint_path": (
        "The shared path the NEXT pass resumes from; the lineage breaks if this is "
        "a per-run dir."
    ),
    "checkpoint_reload_ok": (
        "Bare bool: the saved checkpoint reloads -- the precondition for the next "
        "pass to resume from it."
    ),
    "duration_s": (
        "Bounded training is a multi-minute GPU run that stops itself before the "
        "80-min cap; capped or overlong duration means the bound failed."
    ),
    "random_seed": "Determinism precondition for reproducing the run.",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_progress_print(message: str, *, printer: Callable[..., None] = print) -> None:
    try:
        printer(message, flush=True)
    except BrokenPipeError:
        return


def _parse_metric_value(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class Exp4116Config:
    """Filesystem and Hydra settings for the stable resume run."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_dir: Path | str | None = None
    dataset_dir: Path | str | None = None
    exp4108_artifact_path: Path | str = DEFAULT_EXP4108_ARTIFACT
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
            else parent / "experiment_4116_sudoku_extreme_resume_pass1_hydra"
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
        object.__setattr__(self, "hydra_run_dir", hydra_dir)
        object.__setattr__(self, "dataset_dir", dataset)
        exp4108_artifact = Path(self.exp4108_artifact_path)
        if exp4108_artifact == DEFAULT_EXP4108_ARTIFACT and root != REPO_ROOT:
            exp4108_artifact = root / "results" / exp4108.RESULT_FILENAME
        object.__setattr__(self, "exp4108_artifact_path", exp4108_artifact)

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
class StableSeedResult:
    """How the stable checkpoint path was prepared before training."""

    seed_status: str
    source_checkpoint_path: Path | None
    stable_checkpoint_path: Path
    checkpoint_reload_ok: bool
    checkpoint_reload_detail: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_checkpoint_path"] = (
            None if self.source_checkpoint_path is None else str(self.source_checkpoint_path)
        )
        row["stable_checkpoint_path"] = str(self.stable_checkpoint_path)
        return row


@dataclass(frozen=True)
class ResumeRunResult:
    """Measured result from one bounded native resume invocation."""

    return_code: int
    stable_checkpoint_path: Path
    checkpoint_reload_ok: bool
    checkpoint_reload_detail: str
    val_exact_accuracy: exp4107.ExactAccuracy | None
    cumulative_epochs: int | None
    duration_s: float
    command: list[str]
    stdout_tail: list[str]
    run_dir: Path


class NanoTrmResumeProgressPrinter(Callback):  # pragma: no cover - native subprocess only.
    """Lightning callback that prints progress and refreshes stable last.ckpt."""

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
            _safe_progress_print(
                f"[exp4116:nano-trm-progress] step={step} "
                f"epoch={getattr(trainer, 'current_epoch', 0)} batch_idx={batch_idx}"
            )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = getattr(trainer, "callback_metrics", {})
        exact = metrics.get("val/exact_accuracy") if isinstance(metrics, Mapping) else None
        _safe_progress_print(
            f"[exp4116:nano-trm-progress] validation_end "
            f"epoch={getattr(trainer, 'current_epoch', 0)} "
            f"step={getattr(trainer, 'global_step', 0)} "
            f"val_exact_accuracy={exact}"
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.checkpoint_dir / "last.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            _safe_progress_print(f"[exp4116:nano-trm-progress] checkpoint_saved={checkpoint_path}")
        del pl_module


def check_preconditions(
    *,
    repo_root: str | Path = REPO_ROOT,
    stable_dir: str | Path = DEFAULT_STABLE_DIR,
    uv_resolver: Callable[[str], str | None] = shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """REQ-LEARN-4116: verify uv, native trainer, CUDA, and stable directory."""

    root = Path(repo_root)
    checks = [
        exp4107._check_uv(uv_resolver),
        exp4107._check_trainer(root),
        exp4107._check_cuda(cuda_checker),
    ]
    stable = Path(stable_dir)
    try:
        stable.mkdir(parents=True, exist_ok=True)
        stable.resolve().relative_to(root.resolve())
        stable_ok = os.access(stable, os.W_OK)
        detail = f"writable stable checkpoint dir: {stable}" if stable_ok else f"not writable: {stable}"
    except Exception as exc:
        stable_ok = False
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(exp4107.PreconditionCheck("stable_checkpoint_dir", stable_ok, detail))

    if not checks[0].available or not checks[1].available:
        return checks, "blocked_nanotrm_or_uv_missing"
    if not checks[2].available:
        return checks, "blocked_cuda_unavailable"
    if not checks[3].available:
        return checks, "blocked_save_dir_unwritable"
    return checks, None


def build_train_command(config: Exp4116Config) -> list[str]:
    """REQ-LEARN-4116: build the bounded native resume command."""

    ckpt_value = str(config.stable_checkpoint_path) if config.stable_checkpoint_path.exists() else "null"
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
        f"ckpt_path={ckpt_value}",
        f"+trainer.max_time={config.max_time}",
        "callbacks.model_checkpoint.monitor=val/exact_accuracy",
        "callbacks.model_checkpoint.mode=max",
        f"callbacks.model_checkpoint.dirpath={Path(config.stable_dir)}",
        "callbacks.model_checkpoint.save_last=true",
        "callbacks.model_checkpoint.save_top_k=1",
        "callbacks.model_checkpoint.auto_insert_metric_name=false",
        (
            "+callbacks.exp4116_progress._target_="
            "carnot.experiment_4116_sudoku_extreme_resume_pass1."
            "NanoTrmResumeProgressPrinter"
        ),
        f"+callbacks.exp4116_progress.every_n_steps={int(config.progress_every_n_steps)}",
        f"+callbacks.exp4116_progress.checkpoint_dir={Path(config.stable_dir)}",
    ]


def build_train_env(config: Exp4116Config) -> dict[str, str]:  # pragma: no cover - subprocess only.
    paths = [
        str(Path(config.repo_root) / "python"),
        str(Path(config.repo_root)),
        str(Path(config.nano_trm_root) / "src"),
        str(Path(config.nano_trm_root)),
    ]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    return env


def generate_sudoku_extreme_dataset_if_missing(
    config: Exp4116Config,
) -> bool:  # pragma: no cover - launches builder.
    if exp4108.dataset_is_complete(config.dataset_dir):
        return False
    command = exp4108.build_dataset_command(
        exp4108.NanoTrmExtremeRunConfig(
            repo_root=config.repo_root,
            nano_trm_root=config.nano_trm_root,
            save_parent=config.save_parent,
            dataset_dir=config.dataset_dir,
        )
    )
    print("[exp4116] generating missing Sudoku Extreme dataset", flush=True)
    subprocess.run(
        command,
        cwd=Path(config.nano_trm_root),
        env=build_train_env(config),
        check=True,
    )  # noqa: S603
    return True


def ensure_stable_checkpoint_seed(
    config: Exp4116Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> StableSeedResult:
    """SCENARIO-LEARN-4116: seed the stable path from a loadable Exp 4108 checkpoint."""

    stable_path = config.stable_checkpoint_path
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    if stable_path.exists():
        stable_ok, stable_detail = checkpoint_loader(stable_path)
        if stable_ok:
            return StableSeedResult(
                "existing_stable_checkpoint",
                stable_path,
                stable_path,
                True,
                stable_detail,
            )

    artifact_path = Path(config.exp4108_artifact_path)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return StableSeedResult(
            "no_exp4108_artifact",
            None,
            stable_path,
            False,
            f"missing: {artifact_path}",
        )
    except json.JSONDecodeError as exc:
        return StableSeedResult(
            "invalid_exp4108_artifact",
            None,
            stable_path,
            False,
            f"JSONDecodeError: {exc}",
        )

    checkpoint_value = artifact.get("checkpoint_path")
    source = Path(checkpoint_value) if isinstance(checkpoint_value, str) else None
    if source is None or not source.exists():
        return StableSeedResult(
            "no_loadable_exp4108_checkpoint",
            source,
            stable_path,
            False,
            f"missing checkpoint: {checkpoint_value}",
        )
    source_ok, source_detail = checkpoint_loader(source)
    if not source_ok:
        return StableSeedResult(
            "no_loadable_exp4108_checkpoint",
            source,
            stable_path,
            False,
            source_detail,
        )
    shutil.copy2(source, stable_path)
    stable_ok, stable_detail = checkpoint_loader(stable_path)
    return StableSeedResult(
        "seeded_from_exp4108" if stable_ok else "exp4108_copy_unloadable",
        source,
        stable_path,
        stable_ok,
        stable_detail,
    )


def _metrics_files(root: str | Path) -> list[Path]:
    path = Path(root)
    if path.is_file() and path.name == "metrics.csv":
        return [path]
    return sorted(path.rglob("metrics.csv"), key=lambda item: (item.stat().st_mtime, str(item)))


def extract_latest_val_exact_accuracy(root: str | Path) -> exp4107.ExactAccuracy:
    """SCENARIO-LEARN-4116: parse latest val/exact_accuracy, never q_halt."""

    latest: exp4107.ExactAccuracy | None = None
    for metrics_path in _metrics_files(root):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = _parse_metric_value(row.get("val/exact_accuracy"))
                if value is not None:
                    latest = exp4107.ExactAccuracy("val/exact_accuracy", value, metrics_path)
    if latest is None:
        raise ValueError(f"val/exact_accuracy metric missing under {root}; q_halt_accuracy is not a solve metric")
    return latest


def extract_cumulative_epochs(root: str | Path) -> int | None:
    """SCENARIO-LEARN-4116: infer cumulative epochs from Lightning CSV logs."""

    max_epoch: int | None = None
    for metrics_path in _metrics_files(root):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw_epoch = row.get("epoch")
                value = _parse_metric_value(raw_epoch)
                if value is not None:
                    epoch = int(value)
                    max_epoch = epoch if max_epoch is None else max(max_epoch, epoch)
    return None if max_epoch is None else max_epoch + 1


def verify_completed_resume_run(
    config: Exp4116Config,
    *,
    duration_s: float,
    return_code: int = 0,
    command: Sequence[str] | None = None,
    stdout_tail: Sequence[str] | None = None,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> ResumeRunResult:
    """Verify the stable checkpoint and CSV metrics after native training."""

    checkpoint_path = config.stable_checkpoint_path
    if checkpoint_path.exists():
        reload_ok, reload_detail = checkpoint_loader(checkpoint_path)
    else:
        reload_ok = False
        reload_detail = f"missing stable checkpoint: {checkpoint_path}"
    try:
        exact = extract_latest_val_exact_accuracy(config.hydra_run_dir)
        metric_detail = f"{exact.metric_name}={exact.value}"
    except ValueError as exc:
        exact = None
        metric_detail = str(exc)
    cumulative_epochs = extract_cumulative_epochs(config.hydra_run_dir)
    lines = list(stdout_tail or [])
    lines.extend(
        [
            f"return_code={return_code}",
            f"run_dir={config.hydra_run_dir}",
            f"stable_checkpoint={checkpoint_path}",
            f"checkpoint_reload={reload_detail}",
            f"val_exact_accuracy={metric_detail}",
            f"cumulative_epochs={cumulative_epochs}",
        ]
    )
    return ResumeRunResult(
        return_code=int(return_code),
        stable_checkpoint_path=checkpoint_path,
        checkpoint_reload_ok=reload_ok,
        checkpoint_reload_detail=reload_detail,
        val_exact_accuracy=exact,
        cumulative_epochs=cumulative_epochs,
        duration_s=float(duration_s),
        command=list(command or build_train_command(config)),
        stdout_tail=lines[-60:],
        run_dir=Path(config.hydra_run_dir),
    )


def run_native_resume_training(
    config: Exp4116Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> ResumeRunResult:  # pragma: no cover - launches trainer.
    """Run the real native nano-trm trainer with a wall-clock Lightning bound."""

    started = time.time()
    command = build_train_command(config)
    stdout_lines: list[str] = []
    print(f"[exp4116] launching bounded native resume stable={config.stable_checkpoint_path}", flush=True)
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
        return ResumeRunResult(
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
        print(f"[exp4116:nano-trm] {clean}", flush=True)
        if time.time() - started > config.timeout_s:
            proc.kill()
            stdout_lines.append(f"timeout_s exceeded: {config.timeout_s}")
            timed_out = True
            break
    return_code = proc.wait()
    if timed_out and return_code == 0:
        return_code = 124
    return verify_completed_resume_run(
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
    val_exact_accuracy: exp4107.ExactAccuracy | None,
    cumulative_epochs: int | None,
    stable_checkpoint_path: Path,
    checkpoint_reload_ok: bool,
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4116_sudoku_extreme_resume_pass1",
        "schema": "carnot.experiment_4116_sudoku_extreme_resume_pass1.v1",
        "honest_verdict": honest_verdict,
        "val_exact_accuracy": None if val_exact_accuracy is None else float(val_exact_accuracy.value),
        "cumulative_epochs": cumulative_epochs,
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "checkpoint_reload_ok": bool(checkpoint_reload_ok),
        "duration_s": round(float(duration_s), 3),
        "random_seed": int(random_seed),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": ["REQ-LEARN-4116", "SCENARIO-LEARN-4116"],
    }


def build_result_artifact(
    *,
    run_config: Exp4116Config,
    run_result: ResumeRunResult,
    seed_result: StableSeedResult,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    dataset_generated: bool,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4116: build the measured resume-pass artifact."""

    exact = run_result.val_exact_accuracy
    exact_value = None if exact is None else float(exact.value)
    if run_result.checkpoint_reload_ok and exact_value is not None:
        if exact_value >= PUBLISHED_EXACT_ACCURACY:
            verdict = f"complete: val={exact_value:.4f} reached_0.87"
        else:
            verdict = f"complete: val={exact_value:.4f} still_below_0.87"
        if run_result.return_code != 0:
            verdict = f"complete: return_code_{run_result.return_code}_val={exact_value:.4f}"
    elif run_result.return_code != 0:
        verdict = f"complete: nanotrm_resume_failed_return_code_{run_result.return_code}"
    elif not run_result.checkpoint_reload_ok:
        verdict = "complete: stable_checkpoint_missing_or_reload_failed"
    else:
        verdict = "complete: missing_real_val_exact_accuracy"

    acceptance_gate_passed = bool(
        run_result.checkpoint_reload_ok
        and run_result.duration_s < 4_800
        and exact_value is not None
    )
    artifact = _artifact_common(
        honest_verdict=verdict,
        val_exact_accuracy=exact,
        cumulative_epochs=run_result.cumulative_epochs,
        stable_checkpoint_path=run_result.stable_checkpoint_path,
        checkpoint_reload_ok=run_result.checkpoint_reload_ok,
        duration_s=run_result.duration_s,
        random_seed=run_config.random_seed,
    )
    artifact.update(
        {
            "acceptance_gate_passed": acceptance_gate_passed,
            "exact_accuracy_metric": None if exact is None else exact.metric_name,
            "exact_accuracy_metrics_path": None if exact is None else str(exact.metrics_path),
            "return_code": int(run_result.return_code),
            "run_dir": str(run_result.run_dir),
            "dataset_dir": str(run_config.dataset_dir),
            "dataset_generated": bool(dataset_generated),
            "seed_checkpoint": seed_result.to_dict(),
            "checkpoint_reload_detail": run_result.checkpoint_reload_detail,
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
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4116-BLOCKED: build a no-fabrication blocked artifact."""

    artifact = _artifact_common(
        honest_verdict=reason,
        val_exact_accuracy=None,
        cumulative_epochs=None,
        stable_checkpoint_path=Path(stable_checkpoint_path),
        checkpoint_reload_ok=False,
        duration_s=duration_s,
        random_seed=random_seed,
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "exact_accuracy_metric": None,
            "exact_accuracy_metrics_path": None,
            "return_code": None,
            "run_dir": None,
            "dataset_dir": None,
            "dataset_generated": False,
            "seed_checkpoint": None,
            "checkpoint_reload_detail": "not attempted",
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
    """Return explicit schema errors for the Exp 4116 deliverable."""

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

    if not isinstance(artifact.get("checkpoint_reload_ok"), bool):
        errors.append("checkpoint_reload_ok must be a bare bool")

    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("duration_s must be numeric")

    if not isinstance(artifact.get("random_seed"), int) or isinstance(artifact.get("random_seed"), bool):
        errors.append("random_seed must be a bare int")

    metric_name = artifact.get("exact_accuracy_metric")
    if isinstance(metric_name, str) and "q_halt" in metric_name:
        errors.append("exact_accuracy_metric must not be q_halt_accuracy")

    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if gate is True:
        if exact is None:
            errors.append("accepted artifact requires val_exact_accuracy")
        if artifact.get("checkpoint_reload_ok") is not True:
            errors.append("accepted artifact requires checkpoint_reload_ok true")
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
    exp4108_artifact_path: str | Path = DEFAULT_EXP4108_ARTIFACT,
    uv_resolver: Callable[[str], str | None] = shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    dataset_builder: Callable[[Exp4116Config], object] = generate_sudoku_extreme_dataset_if_missing,
    trainer_runner: Callable[[Exp4116Config], ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4116 or write an honest blocked artifact."""

    started = time.time()
    config = Exp4116Config(
        repo_root=repo_root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_dir=hydra_run_dir,
        exp4108_artifact_path=exp4108_artifact_path,
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
            random_seed=random_seed,
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
            random_seed=random_seed,
        )
        artifact["dataset_dir"] = str(config.dataset_dir)
        artifact["dataset_generated"] = dataset_generated
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    seed_result = ensure_stable_checkpoint_seed(config, checkpoint_loader=checkpoint_loader)
    try:
        if trainer_runner is None:  # pragma: no cover - launches the native trainer.
            run_result = run_native_resume_training(config, checkpoint_loader=checkpoint_loader)
        else:
            run_result = trainer_runner(config)
    except Exception as exc:
        run_result = ResumeRunResult(
            return_code=1,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=False,
            checkpoint_reload_detail=f"{type(exc).__name__}: {exc}",
            val_exact_accuracy=None,
            cumulative_epochs=None,
            duration_s=time.time() - started,
            command=build_train_command(config),
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            run_dir=Path(config.hydra_run_dir),
        )

    artifact = build_result_artifact(
        run_config=config,
        run_result=run_result,
        seed_result=seed_result,
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
