"""Exp 4107 native nano-trm Sudoku mechanism smoke.

Spec refs: REQ-LEARN-4107, SCENARIO-LEARN-4107,
SCENARIO-LEARN-4107-BLOCKED.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


try:  # pragma: no cover - used only inside the native nano-trm subprocess.
    from lightning import Callback
except Exception:  # pragma: no cover - keeps unit imports robust without lightning.
    Callback = object  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4107_nanotrm_mechanism_smoke.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
RANDOM_SEED = 4107
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
BLOCKED_PREFIX = "blocked_"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "nanotrm_trainer_checkpoint_ok",
    "exact_accuracy",
    "checkpoint_path",
    "duration_s",
    "preconditions_checked",
    "random_seed",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest blocked_<resource> or a real "
        "failure-to-train is a COMPLETE verdict; do not fabricate a checkpoint."
    ),
    "nanotrm_trainer_checkpoint_ok": (
        "Bare bool: did nano-trm's native trainer produce and reload a checkpoint."
    ),
    "exact_accuracy": ("The real solve metric from exact_accuracy, never q_halt_accuracy."),
    "checkpoint_path": ("Persistent path to the saved checkpoint under results/trm_runs."),
    "duration_s": ("Wall-clock native training plus checkpoint/metric verification seconds."),
    "preconditions_checked": (
        "Records uv, CUDA, native trainer, and persistent save parent checks."
    ),
    "random_seed": "Determinism precondition for reproducing the run.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before Exp 4107 can launch training."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactAccuracy:
    """A parsed real exact solve metric from nano-trm CSV logs."""

    metric_name: str
    value: float
    metrics_path: Path


@dataclass(frozen=True)
class NanoTrmRunConfig:
    """Hydra and filesystem settings for the native nano-trm Sudoku run."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    save_dir: Path | str | None = None
    hydra_run_dir: Path | str | None = None
    random_seed: int = RANDOM_SEED
    timeout_s: int = 3600
    progress_every_n_steps: int = 25

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        parent = Path(self.save_parent)
        nano_root = Path(self.nano_trm_root) if self.nano_trm_root else root / "nano-trm"
        save_dir = (
            Path(self.save_dir)
            if self.save_dir is not None
            else parent / "experiment_4107_nanotrm_mechanism_smoke"
        )
        hydra_dir = (
            Path(self.hydra_run_dir)
            if self.hydra_run_dir is not None
            else save_dir.with_name(f"{save_dir.name}_hydra")
        )
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "save_dir", save_dir)
        object.__setattr__(self, "hydra_run_dir", hydra_dir)

    @property
    def trainer_path(self) -> Path:
        return Path(self.nano_trm_root) / "src" / "nn" / "train.py"

    @property
    def dataset_dir(self) -> Path:
        return Path(self.nano_trm_root) / "data" / "sudoku_4x4_small"


@dataclass(frozen=True)
class NanoTrmRunResult:
    """Measured result from one native nano-trm trainer invocation."""

    return_code: int
    checkpoint_path: Path | None
    checkpoint_reload_ok: bool
    exact_accuracy: ExactAccuracy | None
    duration_s: float
    command: list[str]
    stdout_tail: list[str]
    save_dir: Path


class NanoTrmProgressPrinter(Callback):  # pragma: no cover - native subprocess only.
    """Lightning callback that prints periodic progress for long Codex runs."""

    def __init__(self, every_n_steps: int = 25) -> None:
        self.every_n_steps = max(int(every_n_steps), 1)

    def on_train_batch_end(
        self,
        trainer: Any,
        _pl_module: Any,
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(getattr(trainer, "global_step", 0))
        progress_index = step if step > 0 else int(batch_idx)
        if progress_index % self.every_n_steps == 0:
            print(
                f"[exp4107:nano-trm-progress] step={step} "
                f"epoch={getattr(trainer, 'current_epoch', 0)} batch_idx={batch_idx}",
                flush=True,
            )

    def on_validation_epoch_end(self, trainer: Any, _pl_module: Any) -> None:
        print(
            f"[exp4107:nano-trm-progress] validation_end "
            f"epoch={getattr(trainer, 'current_epoch', 0)} "
            f"step={getattr(trainer, 'global_step', 0)}",
            flush=True,
        )


def _default_cuda_checker() -> tuple[bool, str]:  # pragma: no cover - environment dependent.
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    available = bool(torch.cuda.is_available())
    detail = f"torch.cuda.is_available()={available}"
    if available:
        try:
            detail += f"; device={torch.cuda.get_device_name(0)}"
        except Exception:
            pass
    return available, detail


def _check_uv(uv_resolver: Callable[[str], str | None]) -> PreconditionCheck:
    path = uv_resolver("uv")
    if not path:
        return PreconditionCheck("uv", False, "uv not found on PATH")
    return PreconditionCheck("uv", True, str(path))


def _check_trainer(root: Path) -> PreconditionCheck:
    trainer = root / "nano-trm" / "src" / "nn" / "train.py"
    if not trainer.exists():
        return PreconditionCheck("nanotrm_trainer", False, f"missing: {trainer}")
    return PreconditionCheck("nanotrm_trainer", True, f"found: {trainer}")


def _check_cuda(cuda_checker: Callable[[], tuple[bool, str]]) -> PreconditionCheck:
    try:
        available, detail = cuda_checker()
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, f"{type(exc).__name__}: {exc}")
    return PreconditionCheck("cuda_available", bool(available), str(detail))


def _check_save_parent(root: Path, save_parent: Path) -> PreconditionCheck:
    if not save_parent.exists() or not save_parent.is_dir():
        return PreconditionCheck(
            "persistent_save_parent", False, f"missing directory: {save_parent}"
        )
    if not os.access(save_parent, os.W_OK):
        return PreconditionCheck("persistent_save_parent", False, f"not writable: {save_parent}")
    try:
        save_parent.resolve().relative_to(root.resolve())
    except ValueError:
        return PreconditionCheck("persistent_save_parent", False, f"not under repo: {save_parent}")
    return PreconditionCheck(
        "persistent_save_parent", True, f"writable persistent dir: {save_parent}"
    )


def check_preconditions(
    *,
    repo_root: str | Path = REPO_ROOT,
    save_parent: str | Path = DEFAULT_SAVE_PARENT,
    uv_resolver: Callable[[str], str | None] = shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
) -> tuple[list[PreconditionCheck], str | None]:
    """REQ-LEARN-4107: verify uv, nano-trm, CUDA, and persistent save parent."""

    root = Path(repo_root)
    checks = [
        _check_uv(uv_resolver),
        _check_trainer(root),
        _check_cuda(cuda_checker),
        _check_save_parent(root, Path(save_parent)),
    ]
    if not checks[0].available or not checks[1].available:
        return checks, "blocked_nanotrm_or_uv_missing"
    if not checks[2].available:
        return checks, "blocked_cuda_unavailable"
    if not checks[3].available:
        return checks, "blocked_save_dir_unwritable"
    return checks, None


def _precondition_dicts(
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for check in preconditions_checked:
        rows.append(check.to_dict() if isinstance(check, PreconditionCheck) else dict(check))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fresh_save_dir(save_parent: Path) -> Path:
    stem = "experiment_4107_nanotrm_mechanism_smoke"
    candidate = save_parent / stem
    if not candidate.exists():
        return candidate
    for index in range(1, 10_000):
        candidate = save_parent / f"{stem}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate a fresh Exp 4107 save_dir")


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


def extract_latest_exact_accuracy(save_dir: str | Path) -> ExactAccuracy:
    """SCENARIO-LEARN-4107: parse latest real exact_accuracy from CSV logs."""

    root = Path(save_dir)
    metrics_files = sorted(
        root.rglob("metrics.csv"), key=lambda path: (path.stat().st_mtime, str(path))
    )
    if not metrics_files:
        raise ValueError(f"exact_accuracy metrics.csv not found under {root}")

    for metric_name in ("val/exact_accuracy", "train/exact_accuracy"):
        latest: ExactAccuracy | None = None
        for metrics_path in metrics_files:
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    value = _parse_metric_value(row.get(metric_name))
                    if value is not None:
                        latest = ExactAccuracy(metric_name, value, metrics_path)
        if latest is not None:
            return latest
    raise ValueError("exact_accuracy metric missing; q_halt_accuracy is not a solve metric")


def _latest_checkpoint(save_dir: Path) -> Path | None:
    checkpoints = sorted(
        save_dir.rglob("*.ckpt"), key=lambda path: (path.stat().st_mtime, str(path))
    )
    if not checkpoints:
        return None
    last = [path for path in checkpoints if path.name == "last.ckpt"]
    return last[-1] if last else checkpoints[-1]


def _candidate_output_roots(config: NanoTrmRunConfig) -> tuple[Path, ...]:
    roots = [Path(config.save_dir), Path(config.hydra_run_dir)]
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return tuple(unique)


def _select_artifact_root(config: NanoTrmRunConfig) -> Path:
    candidates = _candidate_output_roots(config)
    for root in candidates:
        if _latest_checkpoint(root) is not None and list(root.rglob("metrics.csv")):
            return root
    for root in candidates:
        if root.exists():
            return root
    return Path(config.save_dir)


def _load_torch_checkpoint(
    path: Path,
) -> tuple[bool, str]:  # pragma: no cover - environment dependent.
    try:
        import torch  # pylint: disable=import-outside-toplevel

        try:
            payload = safe_torch_load(path, map_location="cpu", allow_unsafe_pickle=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return False, f"unexpected checkpoint payload: {type(payload).__name__}"
    return True, "torch.load ok"


def build_train_command(config: NanoTrmRunConfig) -> list[str]:
    """Build the native command requested by REQ-LEARN-4107."""

    return [
        "uv",
        "run",
        "python",
        "src/nn/train.py",
        "experiment=trm_sudoku_4x4",
        "logger=csv",
        f"hydra.run.dir={Path(config.hydra_run_dir)}",
        f"save_dir={Path(config.save_dir)}",
        "append_wandb_name_to_save_dir=false",
        f"seed={int(config.random_seed)}",
        (
            "+callbacks.exp4107_progress._target_="
            "carnot.experiment_4107_nanotrm_mechanism_smoke.NanoTrmProgressPrinter"
        ),
        f"+callbacks.exp4107_progress.every_n_steps={int(config.progress_every_n_steps)}",
    ]


def build_train_env(
    config: NanoTrmRunConfig,
) -> dict[str, str]:  # pragma: no cover - subprocess only.
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


def generate_sudoku_data_if_missing(
    config: NanoTrmRunConfig,
) -> None:  # pragma: no cover - subprocess only.
    if Path(config.dataset_dir).exists():
        return
    command = [
        "uv",
        "run",
        "python",
        "scripts/data/generate_sudoku_data.py",
        "--grid-size",
        "4",
        "--num-train",
        "10000",
        "--num-val",
        "1000",
        "--num-test",
        "1000",
        "--output-dir",
        "./data/sudoku_4x4_small",
    ]
    print("[exp4107] generating missing nano-trm 4x4 Sudoku data", flush=True)
    subprocess.run(command, cwd=Path(config.nano_trm_root), check=True)  # noqa: S603


def run_native_trm_training(
    config: NanoTrmRunConfig,
) -> NanoTrmRunResult:  # pragma: no cover - launches trainer.
    """Run the real native nano-trm trainer and reload its checkpoint."""

    generate_sudoku_data_if_missing(config)
    started = time.time()
    command = build_train_command(config)
    stdout_lines: list[str] = []
    print(f"[exp4107] launching native nano-trm trainer save_dir={config.save_dir}", flush=True)
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
        return NanoTrmRunResult(
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
        print(f"[exp4107:nano-trm] {clean}", flush=True)
        if time.time() - started > config.timeout_s:
            proc.kill()
            stdout_lines.append(f"timeout_s exceeded: {config.timeout_s}")
            timed_out = True
            break
    return_code = proc.wait()
    if timed_out and return_code == 0:
        return_code = 124

    return verify_completed_native_run(
        config,
        duration_s=time.time() - started,
        return_code=return_code,
        command=command,
        stdout_tail=stdout_lines,
    )


def verify_completed_native_run(
    config: NanoTrmRunConfig,
    *,
    duration_s: float,
    return_code: int = 0,
    command: Sequence[str] | None = None,
    stdout_tail: Sequence[str] | None = None,
) -> NanoTrmRunResult:
    """Verify a completed native nano-trm run without retraining it."""

    artifact_root = _select_artifact_root(config)
    checkpoint_path = _latest_checkpoint(artifact_root)
    reload_ok = False
    if checkpoint_path is not None:
        reload_ok, reload_detail = _load_torch_checkpoint(checkpoint_path)
    else:
        roots = ", ".join(str(root) for root in _candidate_output_roots(config))
        reload_detail = f"checkpoint missing under persistent output roots: {roots}"
    try:
        exact = extract_latest_exact_accuracy(artifact_root)
        metric_detail = f"{exact.metric_name}={exact.value}"
    except ValueError as exc:
        exact = None
        metric_detail = str(exc)
    lines = list(stdout_tail or [])
    lines.extend(
        [
            f"return_code={return_code}",
            f"artifact_root={artifact_root}",
            f"checkpoint={checkpoint_path}",
            f"checkpoint_reload={reload_detail}",
            f"exact_accuracy={metric_detail}",
        ]
    )
    return NanoTrmRunResult(
        return_code=return_code,
        checkpoint_path=checkpoint_path,
        checkpoint_reload_ok=reload_ok,
        exact_accuracy=exact,
        duration_s=float(duration_s),
        command=list(command or build_train_command(config)),
        stdout_tail=lines[-40:],
        save_dir=artifact_root,
    )


def _artifact_common(
    *,
    honest_verdict: str,
    checkpoint_ok: bool,
    exact_accuracy: ExactAccuracy | None,
    checkpoint_path: Path | None,
    duration_s: float,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    random_seed: int,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4107_nanotrm_mechanism_smoke",
        "schema": "carnot.experiment_4107_nanotrm_mechanism_smoke.v1",
        "honest_verdict": honest_verdict,
        "nanotrm_trainer_checkpoint_ok": bool(checkpoint_ok),
        "exact_accuracy": None if exact_accuracy is None else float(exact_accuracy.value),
        "exact_accuracy_metric": None if exact_accuracy is None else exact_accuracy.metric_name,
        "exact_accuracy_metrics_path": None
        if exact_accuracy is None
        else str(exact_accuracy.metrics_path),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "duration_s": round(float(duration_s), 3),
        "preconditions_checked": _precondition_dicts(preconditions_checked),
        "random_seed": int(random_seed),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": ["REQ-LEARN-4107", "SCENARIO-LEARN-4107"],
    }


def build_blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4107-BLOCKED: build a no-fabrication blocked artifact."""

    artifact = _artifact_common(
        honest_verdict=reason,
        checkpoint_ok=False,
        exact_accuracy=None,
        checkpoint_path=None,
        duration_s=duration_s,
        preconditions_checked=preconditions_checked,
        random_seed=random_seed,
    )
    artifact.update(
        {
            "acceptance_gate_passed": False,
            "return_code": None,
            "save_dir": None,
            "command": [],
            "stdout_tail": [],
        }
    )
    validate_artifact(artifact)
    return artifact


def build_success_artifact(
    *,
    run_result: NanoTrmRunResult,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4107: build the measured native trainer artifact."""

    checkpoint_ok = bool(
        run_result.return_code == 0
        and run_result.checkpoint_path is not None
        and run_result.checkpoint_reload_ok
    )
    exact = run_result.exact_accuracy
    duration = float(run_result.duration_s)
    acceptance = bool(checkpoint_ok and exact is not None and duration > 60.0)
    if acceptance:
        verdict = f"complete: nanotrm_trainer_checkpoint_ok_exact_accuracy_{exact.value:.4f}"
    elif run_result.return_code != 0:
        verdict = f"complete: nanotrm_native_training_failed_return_code_{run_result.return_code}"
    elif not checkpoint_ok:
        verdict = "complete: nanotrm_native_trainer_checkpoint_missing_or_reload_failed"
    elif exact is None:
        verdict = "complete: nanotrm_native_trainer_missing_real_exact_accuracy"
    else:
        verdict = f"complete: nanotrm_native_training_under_60s_duration_{duration:.3f}"

    artifact = _artifact_common(
        honest_verdict=verdict,
        checkpoint_ok=checkpoint_ok,
        exact_accuracy=exact,
        checkpoint_path=run_result.checkpoint_path,
        duration_s=duration,
        preconditions_checked=preconditions_checked,
        random_seed=random_seed,
    )
    artifact.update(
        {
            "acceptance_gate_passed": acceptance,
            "return_code": int(run_result.return_code),
            "save_dir": str(run_result.save_dir),
            "command": list(run_result.command),
            "stdout_tail": list(run_result.stdout_tail[-40:]),
        }
    )
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4107 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith((*TERMINAL_PREFIXES, BLOCKED_PREFIX)):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    checkpoint_ok = artifact.get("nanotrm_trainer_checkpoint_ok")
    if not isinstance(checkpoint_ok, bool):
        errors.append("nanotrm_trainer_checkpoint_ok must be a bare bool")

    exact = artifact.get("exact_accuracy")
    if exact is not None:
        if not isinstance(exact, (int, float)) or isinstance(exact, bool):
            errors.append("exact_accuracy must be numeric or null")
        elif not 0.0 <= float(exact) <= 1.0:
            errors.append("exact_accuracy must be between 0 and 1")

    metric_name = artifact.get("exact_accuracy_metric")
    if isinstance(metric_name, str) and "q_halt" in metric_name:
        errors.append("exact_accuracy_metric must not be q_halt_accuracy")

    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("duration_s must be numeric")

    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked must be a list")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(
        artifact.get("random_seed"), bool
    ):
        errors.append("random_seed must be a bare int")

    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if gate is True:
        if checkpoint_ok is not True:
            errors.append("successful gate requires nanotrm_trainer_checkpoint_ok true")
        if exact is None:
            errors.append("successful gate requires exact_accuracy")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or float(duration) <= 60.0
        ):
            errors.append("successful gate requires duration_s > 60")
        checkpoint_path = artifact.get("checkpoint_path")
        if not (
            isinstance(checkpoint_path, str)
            and checkpoint_path.endswith(".ckpt")
            and Path(checkpoint_path).exists()
        ):
            errors.append("successful gate requires an existing .ckpt checkpoint_path")
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
    uv_resolver: Callable[[str], str | None] = shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
    trainer_runner: Callable[[NanoTrmRunConfig], NanoTrmRunResult] = run_native_trm_training,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4107 or write an honest blocked artifact."""

    started = time.time()
    root = Path(repo_root)
    parent = Path(save_parent)
    out = Path(output_path)
    checks, blocker = check_preconditions(
        repo_root=root,
        save_parent=parent,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            blocker,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        _write_json(out, artifact)
        return artifact

    chosen_save_dir = Path(save_dir) if save_dir is not None else _fresh_save_dir(parent)
    config = NanoTrmRunConfig(
        repo_root=root,
        save_parent=parent,
        save_dir=chosen_save_dir,
        random_seed=random_seed,
    )
    try:
        run_result = trainer_runner(config)
    except Exception as exc:
        run_result = NanoTrmRunResult(
            return_code=1,
            checkpoint_path=None,
            checkpoint_reload_ok=False,
            exact_accuracy=None,
            duration_s=time.time() - started,
            command=build_train_command(config),
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            save_dir=Path(config.save_dir),
        )
    artifact = build_success_artifact(
        run_result=run_result,
        preconditions_checked=checks,
        random_seed=random_seed,
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
