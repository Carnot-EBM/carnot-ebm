"""Exp 4108 native nano-trm Sudoku Extreme baseline reproduction.

Spec refs: REQ-LEARN-4108, SCENARIO-LEARN-4108,
SCENARIO-LEARN-4108-SHORT.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107


try:  # pragma: no cover - used only inside the native nano-trm subprocess.
    from lightning import Callback
except Exception:  # pragma: no cover - keeps unit imports robust without lightning.
    Callback = object  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4108_nanotrm_sudoku_extreme_baseline.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_EXP4107_ARTIFACT = REPO_ROOT / "results" / "experiment_4107_nanotrm_mechanism_smoke.json"
RANDOM_SEED = 4108
PUBLISHED_EXACT_ACCURACY = 0.87
PUBLISHED_MATCH_TOLERANCE = 0.02
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
BLOCKED_PREFIX = "blocked_"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "reproduced_exact_accuracy",
    "matches_published_087",
    "checkpoint_path",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'reproduced 0.NN below target' is a "
        "COMPLETE verdict; report the real number, do not claim 0.87 if not reached."
    ),
    "reproduced_exact_accuracy": (
        "The measured val exact-accuracy; the load-bearing number this baseline contributes."
    ),
    "matches_published_087": (
        "Bare bool: within tolerance of the published ~0.87. Tells exp4109 whether "
        "it is building on a faithful baseline or a partial one."
    ),
    "checkpoint_path": (
        "Persistent path to the reproducing checkpoint exp4109 will graft the verifier onto."
    ),
    "duration_s": (
        "A faithful Sudoku-Extreme reproduction is a multi-minute-to-hour GPU run; "
        "an implausibly short duration is the fabrication signal."
    ),
    "random_seed": "Determinism precondition for reproducing the baseline.",
    "reproducibility_checksum": (
        "Hash of the dataset + config; catches silent drift vs a future rerun."
    ),
}


def _safe_progress_print(
    message: str,
    *,
    printer: Callable[..., None] = print,
) -> None:
    """REQ-LEARN-4108: keep progress logging from aborting native training."""

    try:
        printer(message, flush=True)
    except BrokenPipeError:
        return


@dataclass(frozen=True)
class Exp4107Status:
    """The prior mechanism-smoke status that controls full vs short mode."""

    artifact_path: Path
    checkpoint_ok: bool
    honest_verdict: str

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["artifact_path"] = str(self.artifact_path)
        return row


@dataclass(frozen=True)
class NanoTrmExtremeRunConfig:
    """Hydra and filesystem settings for the native Sudoku Extreme run."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    save_dir: Path | str | None = None
    hydra_run_dir: Path | str | None = None
    dataset_dir: Path | str | None = None
    random_seed: int = RANDOM_SEED
    timeout_s: int = 21_600
    progress_every_n_steps: int = 100
    shorter_attempt: bool = False

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        parent = Path(self.save_parent)
        nano_root = Path(self.nano_trm_root) if self.nano_trm_root else root / "nano-trm"
        save_stem = "experiment_4108_nanotrm_sudoku_extreme_baseline"
        if self.shorter_attempt:
            save_stem += "_short"
        save_dir = Path(self.save_dir) if self.save_dir is not None else parent / save_stem
        hydra_dir = (
            Path(self.hydra_run_dir)
            if self.hydra_run_dir is not None
            else save_dir.with_name(f"{save_dir.name}_hydra")
        )
        dataset_dir = (
            Path(self.dataset_dir)
            if self.dataset_dir is not None
            else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        )
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "save_dir", save_dir)
        object.__setattr__(self, "hydra_run_dir", hydra_dir)
        object.__setattr__(self, "dataset_dir", dataset_dir)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def dataset_builder_path(self) -> Path:
        return Path(self.nano_trm_root) / "scripts" / "data" / "build_sudoku_extreme_dataset.py"

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


class NanoTrmExtremeProgressPrinter(Callback):  # pragma: no cover - native subprocess only.
    """Lightning callback that prints periodic progress for long Exp 4108 runs."""

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
                f"[exp4108:nano-trm-progress] step={step} "
                f"epoch={getattr(trainer, 'current_epoch', 0)} batch_idx={batch_idx}"
            )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = getattr(trainer, "callback_metrics", {})
        exact = metrics.get("val/exact_accuracy") if isinstance(metrics, Mapping) else None
        _safe_progress_print(
            f"[exp4108:nano-trm-progress] validation_end "
            f"epoch={getattr(trainer, 'current_epoch', 0)} "
            f"step={getattr(trainer, 'global_step', 0)} "
            f"val_exact_accuracy={exact}"
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.checkpoint_dir / "last.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            _safe_progress_print(
                f"[exp4108:nano-trm-progress] checkpoint_saved={checkpoint_path}"
            )
        del pl_module


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fresh_save_dir(save_parent: Path, *, shorter_attempt: bool) -> Path:
    stem = "experiment_4108_nanotrm_sudoku_extreme_baseline"
    if shorter_attempt:
        stem += "_short"
    candidate = save_parent / stem
    hydra_candidate = candidate.with_name(f"{candidate.name}_hydra")
    if not candidate.exists() and not hydra_candidate.exists():
        return candidate
    for index in range(1, 10_000):
        candidate = save_parent / f"{stem}_{index:03d}"
        hydra_candidate = candidate.with_name(f"{candidate.name}_hydra")
        if not candidate.exists() and not hydra_candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a fresh Exp 4108 save_dir")  # pragma: no cover


def load_exp4107_status(path: str | Path = DEFAULT_EXP4107_ARTIFACT) -> Exp4107Status:
    """REQ-LEARN-4108: read the mechanism-smoke artifact before training."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        return Exp4107Status(artifact_path, False, "missing_exp4107_artifact")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return Exp4107Status(artifact_path, False, f"invalid_exp4107_artifact: {exc}")
    return Exp4107Status(
        artifact_path=artifact_path,
        checkpoint_ok=payload.get("nanotrm_trainer_checkpoint_ok") is True,
        honest_verdict=str(payload.get("honest_verdict", "missing_honest_verdict")),
    )


def dataset_is_complete(dataset_dir: str | Path) -> bool:
    """REQ-LEARN-4108: require the real train/val/test Sudoku Extreme files."""

    root = Path(dataset_dir)
    required = ["dataset.json", "all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"]
    if not (root / "metadata.json").is_file():
        return False
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            return False
        if any(not (split_dir / name).is_file() for name in required):
            return False
    return True


def build_dataset_command(_config: NanoTrmExtremeRunConfig) -> list[str]:
    """Build the native dataset command requested by REQ-LEARN-4108."""

    return [
        "uv",
        "run",
        "python",
        "scripts/data/build_sudoku_extreme_dataset.py",
        "--output-dir",
        "./data/sudoku_extreme_1k_aug_1k",
        "--subsample-size",
        "1000",
        "--num-aug",
        "1000",
        "--eval-ratio",
        "0.01",
    ]


def build_train_command(config: NanoTrmExtremeRunConfig) -> list[str]:
    """Build the native Sudoku Extreme trainer command."""

    command = [
        "uv",
        "run",
        "python",
        "src/nn/train.py",
        "experiment=trm_sudoku_extreme_1k_aug_1k",
        "logger=csv",
        f"hydra.run.dir={Path(config.hydra_run_dir)}",
        f"save_dir={Path(config.save_dir)}",
        "append_wandb_name_to_save_dir=false",
        f"seed={int(config.random_seed)}",
        "data.data_dir=./data/sudoku_extreme_1k_aug_1k",
        "callbacks.model_checkpoint.monitor=val/exact_accuracy",
        "callbacks.model_checkpoint.mode=max",
        (
            "+callbacks.exp4108_progress._target_="
            "carnot.experiment_4108_nanotrm_sudoku_extreme_baseline."
            "NanoTrmExtremeProgressPrinter"
        ),
        f"+callbacks.exp4108_progress.every_n_steps={int(config.progress_every_n_steps)}",
        f"+callbacks.exp4108_progress.checkpoint_dir={Path(config.hydra_run_dir) / 'checkpoints'}",
    ]
    if config.shorter_attempt:
        command.extend(["timekeeping.max_epochs=100", "trainer.check_val_every_n_epoch=10"])
    return command


def build_train_env(config: NanoTrmExtremeRunConfig) -> dict[str, str]:  # pragma: no cover - subprocess only.
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
    config: NanoTrmExtremeRunConfig,
) -> bool:  # pragma: no cover - launches builder.
    if dataset_is_complete(config.dataset_dir):
        return False
    command = build_dataset_command(config)
    print("[exp4108] generating missing Sudoku Extreme dataset", flush=True)
    subprocess.run(  # noqa: S603
        command,
        cwd=Path(config.nano_trm_root),
        env=build_train_env(config),
        check=True,
    )
    return True


def run_native_trm_training(
    config: NanoTrmExtremeRunConfig,
) -> exp4107.NanoTrmRunResult:  # pragma: no cover - launches trainer.
    """Run the real native nano-trm trainer and reload its checkpoint."""

    started = time.time()
    command = build_train_command(config)
    stdout_lines: list[str] = []
    print(f"[exp4108] launching native nano-trm trainer save_dir={config.save_dir}", flush=True)
    try:
        proc = subprocess.Popen(  # noqa: S603 - local command assembled above.
            command,
            cwd=str(config.nano_trm_root),
            env=build_train_env(config),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        return exp4107.NanoTrmRunResult(
            return_code=1,
            checkpoint_path=None,
            checkpoint_reload_ok=False,
            exact_accuracy=None,
            duration_s=time.time() - started,
            command=command,
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            save_dir=Path(config.save_dir),
        )

    assert proc.stdout is not None
    timed_out = False
    for line in proc.stdout:
        clean = line.rstrip()
        stdout_lines.append(clean)
        print(f"[exp4108:nano-trm] {clean}", flush=True)
        if time.time() - started > config.timeout_s:
            proc.kill()
            stdout_lines.append(f"timeout_s exceeded: {config.timeout_s}")
            timed_out = True
            break
    return_code = proc.wait()
    if timed_out and return_code == 0:
        return_code = 124

    return exp4107.verify_completed_native_run(
        config,  # type: ignore[arg-type]
        duration_s=time.time() - started,
        return_code=return_code,
        command=command,
        stdout_tail=stdout_lines,
    )


def _hash_file(hasher: Any, root: Path, path: Path) -> None:
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    hasher.update(str(rel).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(path.read_bytes())
    hasher.update(b"\0")


def _hash_path(hasher: Any, label: str, path: Path) -> None:
    hasher.update(label.encode("utf-8"))
    hasher.update(b"\0")
    if path.is_file():
        _hash_file(hasher, path.parent, path)
        return
    if path.is_dir():
        for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item)):
            _hash_file(hasher, path, child)
        return
    hasher.update(f"missing:{path}".encode("utf-8"))
    hasher.update(b"\0")


def compute_reproducibility_checksum(config: NanoTrmExtremeRunConfig) -> str:
    """SCENARIO-LEARN-4108: hash dataset and native configs to catch drift."""

    hasher = hashlib.sha256()
    hasher.update(b"carnot.exp4108.nanotrm_sudoku_extreme_baseline.v1\0")
    hasher.update(json.dumps(build_train_command(config), sort_keys=True).encode("utf-8"))
    hasher.update(b"\0")
    for label, path in (
        ("dataset", Path(config.dataset_dir)),
        ("experiment_config", Path(config.experiment_config_path)),
        ("data_config", Path(config.data_config_path)),
    ):
        _hash_path(hasher, label, path)
    return f"sha256:{hasher.hexdigest()}"


def matches_published_accuracy(value: float | None, *, shorter_attempt: bool = False) -> bool:
    if value is None or shorter_attempt:
        return False
    return abs(float(value) - PUBLISHED_EXACT_ACCURACY) <= PUBLISHED_MATCH_TOLERANCE


def _artifact_common(
    *,
    honest_verdict: str,
    reproduced_exact_accuracy: float | None,
    matches_published_087: bool,
    checkpoint_path: Path | None,
    duration_s: float,
    random_seed: int,
    reproducibility_checksum: str,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4108_nanotrm_sudoku_extreme_baseline",
        "schema": "carnot.experiment_4108_nanotrm_sudoku_extreme_baseline.v1",
        "honest_verdict": honest_verdict,
        "reproduced_exact_accuracy": reproduced_exact_accuracy,
        "matches_published_087": bool(matches_published_087),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "duration_s": round(float(duration_s), 3),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum,
        "published_exact_accuracy_target": PUBLISHED_EXACT_ACCURACY,
        "published_match_tolerance": PUBLISHED_MATCH_TOLERANCE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": ["REQ-LEARN-4108", "SCENARIO-LEARN-4108"],
    }


def build_result_artifact(
    *,
    run_config: NanoTrmExtremeRunConfig,
    run_result: exp4107.NanoTrmRunResult,
    mechanism_status: Exp4107Status,
    dataset_generated: bool,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4108: build the measured baseline artifact."""

    checkpoint_ok = bool(run_result.checkpoint_path is not None and run_result.checkpoint_reload_ok)
    exact = run_result.exact_accuracy
    reproduced = None if exact is None else float(exact.value)
    matches = matches_published_accuracy(reproduced, shorter_attempt=run_config.shorter_attempt)
    if checkpoint_ok and reproduced is not None:
        if run_config.shorter_attempt:
            verdict = f"complete: short_attempt_reproduced_{reproduced:.4f}_mechanism_unproven"
        elif run_result.return_code != 0:
            verdict = (
                f"complete: interrupted_return_code_{run_result.return_code}_"
                f"reproduced_{reproduced:.4f}"
            )
        elif matches:
            verdict = f"complete: reproduced_{reproduced:.4f}_matches_published_0.87"
        else:
            verdict = f"complete: reproduced_{reproduced:.4f}_below_published_0.87"
    elif run_result.return_code != 0:
        verdict = f"complete: nanotrm_sudoku_extreme_training_failed_return_code_{run_result.return_code}"
    elif not checkpoint_ok:
        verdict = "complete: nanotrm_sudoku_extreme_checkpoint_missing_or_reload_failed"
    else:
        verdict = "complete: nanotrm_sudoku_extreme_missing_real_val_exact_accuracy"

    artifact = _artifact_common(
        honest_verdict=verdict,
        reproduced_exact_accuracy=reproduced,
        matches_published_087=matches,
        checkpoint_path=run_result.checkpoint_path,
        duration_s=run_result.duration_s,
        random_seed=run_config.random_seed,
        reproducibility_checksum=compute_reproducibility_checksum(run_config),
    )
    artifact.update(
        {
            "acceptance_gate_passed": bool(checkpoint_ok and reproduced is not None),
            "checkpoint_reload_ok": bool(run_result.checkpoint_reload_ok),
            "exact_accuracy_metric": None if exact is None else exact.metric_name,
            "exact_accuracy_metrics_path": None if exact is None else str(exact.metrics_path),
            "return_code": int(run_result.return_code),
            "save_dir": str(run_result.save_dir),
            "dataset_dir": str(run_config.dataset_dir),
            "dataset_generated": bool(dataset_generated),
            "shorter_attempt": bool(run_config.shorter_attempt),
            "mechanism_checkpoint_ok": bool(mechanism_status.checkpoint_ok),
            "mechanism_status": mechanism_status.to_dict(),
            "command": list(run_result.command),
            "stdout_tail": list(run_result.stdout_tail[-40:]),
        }
    )
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    mechanism_status: Exp4107Status,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Build a required-field artifact when runtime resources are unavailable."""

    artifact = _artifact_common(
        honest_verdict=reason,
        reproduced_exact_accuracy=None,
        matches_published_087=False,
        checkpoint_path=None,
        duration_s=duration_s,
        random_seed=random_seed,
        reproducibility_checksum="sha256:" + hashlib.sha256(reason.encode("utf-8")).hexdigest(),
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "checkpoint_reload_ok": False,
            "exact_accuracy_metric": None,
            "exact_accuracy_metrics_path": None,
            "return_code": None,
            "save_dir": None,
            "dataset_dir": None,
            "dataset_generated": False,
            "shorter_attempt": not mechanism_status.checkpoint_ok,
            "mechanism_checkpoint_ok": bool(mechanism_status.checkpoint_ok),
            "mechanism_status": mechanism_status.to_dict(),
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
    """Return explicit schema errors for the Exp 4108 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith((*TERMINAL_PREFIXES, BLOCKED_PREFIX)):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    exact = artifact.get("reproduced_exact_accuracy")
    if exact is not None:
        if not isinstance(exact, (int, float)) or isinstance(exact, bool):
            errors.append("reproduced_exact_accuracy must be numeric or null")
        elif not 0.0 <= float(exact) <= 1.0:
            errors.append("reproduced_exact_accuracy must be between 0 and 1")

    if not isinstance(artifact.get("matches_published_087"), bool):
        errors.append("matches_published_087 must be a bare bool")

    metric_name = artifact.get("exact_accuracy_metric")
    if isinstance(metric_name, str) and "q_halt" in metric_name:
        errors.append("exact_accuracy_metric must not be q_halt_accuracy")

    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("duration_s must be numeric")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(artifact.get("random_seed"), bool):
        errors.append("random_seed must be a bare int")
    checksum = artifact.get("reproducibility_checksum")
    if not (isinstance(checksum, str) and checksum.startswith("sha256:") and len(checksum) == 71):
        errors.append("reproducibility_checksum must be sha256-prefixed")

    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if gate is True:
        if exact is None:
            errors.append("accepted artifact requires reproduced_exact_accuracy")
        checkpoint_path = artifact.get("checkpoint_path")
        if not (
            isinstance(checkpoint_path, str)
            and checkpoint_path.endswith(".ckpt")
            and Path(checkpoint_path).exists()
        ):
            errors.append("accepted artifact requires an existing .ckpt checkpoint_path")
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
    save_dir: str | Path | None = None,
    exp4107_artifact_path: str | Path = DEFAULT_EXP4107_ARTIFACT,
    uv_resolver: Callable[[str], str | None] = shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    dataset_builder: Callable[[NanoTrmExtremeRunConfig], object] = generate_sudoku_extreme_dataset_if_missing,
    trainer_runner: Callable[[NanoTrmExtremeRunConfig], exp4107.NanoTrmRunResult] = run_native_trm_training,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4108 or write an honest blocked artifact."""

    started = time.time()
    root = Path(repo_root)
    parent = Path(save_parent)
    out = Path(output_path)
    mechanism_status = load_exp4107_status(exp4107_artifact_path)
    checks, blocker = exp4107.check_preconditions(
        repo_root=root,
        save_parent=parent,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            blocker,
            mechanism_status=mechanism_status,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        _write_json(out, artifact)
        return artifact

    shorter_attempt = not mechanism_status.checkpoint_ok
    chosen_save_dir = (
        Path(save_dir)
        if save_dir is not None
        else _fresh_save_dir(parent, shorter_attempt=shorter_attempt)
    )
    config = NanoTrmExtremeRunConfig(
        repo_root=root,
        save_parent=parent,
        save_dir=chosen_save_dir,
        random_seed=random_seed,
        shorter_attempt=shorter_attempt,
    )
    dataset_generated = False
    if not dataset_is_complete(config.dataset_dir):
        dataset_builder(config)
        dataset_generated = True
    try:
        run_result = trainer_runner(config)
    except Exception as exc:
        run_result = exp4107.NanoTrmRunResult(
            return_code=1,
            checkpoint_path=None,
            checkpoint_reload_ok=False,
            exact_accuracy=None,
            duration_s=time.time() - started,
            command=build_train_command(config),
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            save_dir=Path(config.save_dir),
        )
    artifact = build_result_artifact(
        run_config=config,
        run_result=run_result,
        mechanism_status=mechanism_status,
        dataset_generated=dataset_generated,
    )
    _write_json(out, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps({field: artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS}, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
