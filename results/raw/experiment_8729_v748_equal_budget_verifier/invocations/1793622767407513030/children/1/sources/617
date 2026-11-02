"""Exp 4135 fixed-LR nano-trm Sudoku Extreme accumulation pass 1.

Spec refs: REQ-LEARN-4135, SCENARIO-LEARN-4135,
SCENARIO-LEARN-4135-BLOCKED.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4108_nanotrm_sudoku_extreme_baseline as exp4108
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4126_lr_resume_correctness_fix as exp4126
from carnot import experiment_4127_sudoku_extreme_accumulate_fixed as exp4127


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4135_sudoku_accumulate_pass1_fixed_lr.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_HYDRA_RUN_ROOT = DEFAULT_SAVE_PARENT / "experiment_4135_sudoku_accumulate_pass1_fixed_lr"
DEFAULT_LR_FIX_ARTIFACT = REPO_ROOT / "results" / exp4126.RESULT_FILENAME
RANDOM_SEED = exp4108.RANDOM_SEED
STARTING_VAL_EXACT_ACCURACY = 0.278
PASS_INDEX = 1
MAX_TIME = "00:01:00:00"
LOCAL_SAFE_BATCH_SIZE = 128
PUBLISHED_EXACT_ACCURACY = exp4127.PUBLISHED_EXACT_ACCURACY
PUBLISHED_TOLERANCE = exp4127.PUBLISHED_TOLERANCE
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_exact_accuracy",
    "delta_vs_previous",
    "lr_continued_not_rewarmed",
    "matches_published_087",
    "stable_checkpoint_path",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'val=0.NN improved, still <0.87 -> "
        "pass2 continues' is COMPLETE."
    ),
    "val_exact_accuracy": "Val after this pass; the load-bearing accumulation evidence.",
    "delta_vs_previous": (
        "Improvement vs the 0.278 start; confirms the fixed schedule is still "
        "accumulating (vs a stall)."
    ),
    "lr_continued_not_rewarmed": (
        "Bare bool: train/lr at pass start CONTINUED the schedule (not a "
        "2.45e-6 reset) -- a regression guard on the exp4126 fix."
    ),
    "matches_published_087": (
        "Bare bool: within 0.02 of 0.87. Tells the next pass / graft whether a "
        "faithful baseline exists."
    ),
    "stable_checkpoint_path": "The persisted baseline path pass2 (exp4136) resumes from.",
    "random_seed": (
        "Determinism precondition for reproducibility; the nano-trm seed (4108) "
        "recorded as a first-class field."
    ),
    "reproducibility_checksum": (
        "Content hash of config, stable checkpoint, and data dir; catches silent "
        "corpus/model drift between this pass and any replication."
    ),
    "model_specs": (
        "Names nano-trm plus the trm_sudoku_extreme_1k_aug_1k config so the "
        "artifact declares what was actually trained."
    ),
    "duration_s": "Bounded GPU run < 4800s; a single scalar duration.",
}


@dataclass(frozen=True)
class Exp4135Config:
    """Filesystem and Hydra settings for the single fixed-LR accumulation pass."""

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
        stable = (
            Path(self.stable_dir)
            if self.stable_dir is not None
            else parent / "sudoku_extreme_baseline"
        )
        hydra_root = (
            Path(self.hydra_run_root)
            if self.hydra_run_root is not None
            else parent / Path(DEFAULT_HYDRA_RUN_ROOT).name
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
        return Path(self.hydra_run_root) / f"pass_{PASS_INDEX}_hydra"

    def to_4127_config(self) -> exp4127.Exp4127Config:
        return exp4127.Exp4127Config(
            repo_root=self.repo_root,
            nano_trm_root=self.nano_trm_root,
            save_parent=self.save_parent,
            stable_dir=self.stable_dir,
            hydra_run_root=self.hydra_run_root,
            dataset_dir=self.dataset_dir,
            lr_artifact_path=self.lr_artifact_path,
            random_seed=self.random_seed,
            max_time=self.max_time,
            timeout_s=self.timeout_s,
            progress_every_n_steps=self.progress_every_n_steps,
            batch_size=self.batch_size,
        )

    def to_4116_config(self) -> exp4116.Exp4116Config:
        return exp4116.Exp4116Config(
            repo_root=self.repo_root,
            nano_trm_root=self.nano_trm_root,
            save_parent=self.save_parent,
            stable_dir=self.stable_dir,
            hydra_run_dir=self.pass_run_dir(),
            dataset_dir=self.dataset_dir,
            random_seed=self.random_seed,
            max_time=self.max_time,
            timeout_s=self.timeout_s,
            progress_every_n_steps=self.progress_every_n_steps,
        )


@dataclass(frozen=True)
class PassMetricSummary:
    """The two CSV observations Exp 4135 needs after the native pass."""

    val_exact_accuracy: float | None
    first_train_lr: float | None
    train_lr_point_count: int
    val_metrics_path: Path | None
    first_train_lr_metrics_path: Path | None

    @property
    def lr_continued_not_rewarmed(self) -> bool:
        """REQ-LEARN-4135: true only when train/lr did not restart fresh warmup."""

        return lr_continued_not_rewarmed(self.first_train_lr)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["val_metrics_path"] = (
            None if self.val_metrics_path is None else str(self.val_metrics_path)
        )
        row["first_train_lr_metrics_path"] = (
            None
            if self.first_train_lr_metrics_path is None
            else str(self.first_train_lr_metrics_path)
        )
        row["lr_continued_not_rewarmed"] = self.lr_continued_not_rewarmed
        return row


@dataclass(frozen=True)
class TimerResetResult:
    """How Exp 4135 handled Lightning's checkpointed wall-clock timer."""

    changed: bool
    detail: str
    manual_lr_step: int | None

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


def _rounded(value: float | None, digits: int = 12) -> float | None:
    return None if value is None else round(float(value), digits)


def _checks_to_dicts(
    checks: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        check.to_dict() if isinstance(check, exp4107.PreconditionCheck) else dict(check)
        for check in checks
    ]


def lr_continued_not_rewarmed(first_lr: float | None) -> bool:
    """REQ-LEARN-4135: reject the known fresh-warmup LR reset."""

    if first_lr is None:
        return False
    return not math.isclose(first_lr, exp4126.FRESH_WARMUP_FIRST_LR, rel_tol=0.0, abs_tol=1e-12)


def matches_published_087(value: float | None) -> bool:
    """REQ-LEARN-4135: compare validation exact accuracy to 0.87 within 0.02."""

    return (
        value is not None
        and abs(float(value) - PUBLISHED_EXACT_ACCURACY) <= PUBLISHED_TOLERANCE + 1e-12
    )


def build_train_command(config: Exp4135Config) -> list[str]:
    """REQ-LEARN-4135: build the single fixed-LR bounded resume command."""

    return exp4127.build_train_command(config.to_4127_config(), PASS_INDEX)


def build_train_env(config: Exp4135Config) -> dict[str, str]:
    """REQ-LEARN-4135: disable compile/CUDAGraph resume while keeping CUDA enabled."""

    return exp4127.build_train_env(config.to_4127_config())


def model_specs(config: Exp4135Config) -> dict[str, Any]:
    """REQ-LEARN-4135: name the native model/config inputs used for training."""

    return {
        "model": "nano-trm",
        "experiment_config": "trm_sudoku_extreme_1k_aug_1k",
        "data_config": "sudoku_extreme_1k_aug1k",
        "trainer_path": str(config.trainer_path),
        "experiment_config_path": str(config.experiment_config_path),
        "data_config_path": str(config.data_config_path),
        "data_dir": str(config.dataset_dir),
        "seed": int(config.random_seed),
        "batch_size": int(config.batch_size),
        "max_time": str(config.max_time),
    }


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
        children = sorted(
            (item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item)
        )
        for child in children:
            _hash_file(hasher, path, child)
        return
    hasher.update(f"missing:{path}".encode())
    hasher.update(b"\0")


def compute_reproducibility_checksum(config: Exp4135Config) -> str:
    """REQ-LEARN-4135: hash config, stable checkpoint, and Sudoku data inputs."""

    hasher = hashlib.sha256()
    hasher.update(b"carnot.exp4135.sudoku_accumulate_pass1_fixed_lr.v1\0")
    hasher.update(json.dumps(build_train_command(config), sort_keys=True).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(json.dumps(model_specs(config), sort_keys=True).encode("utf-8"))
    hasher.update(b"\0")
    for label, path in (
        ("experiment_config", config.experiment_config_path),
        ("data_config", config.data_config_path),
        ("stable_checkpoint", config.stable_checkpoint_path),
        ("dataset", Path(config.dataset_dir)),
    ):
        _hash_path(hasher, label, Path(path))
    return f"sha256:{hasher.hexdigest()}"


def summarize_pass_metrics(run_dir: str | Path) -> PassMetricSummary:
    """SCENARIO-LEARN-4135: read validation exact accuracy and the first train/lr."""

    try:
        exact = exp4116.extract_latest_val_exact_accuracy(run_dir)
    except ValueError:
        exact = None
    lr_points = exp4126.extract_train_lr_points(run_dir)
    first_lr = lr_points[0] if lr_points else None
    return PassMetricSummary(
        val_exact_accuracy=None if exact is None else float(exact.value),
        first_train_lr=None if first_lr is None else float(first_lr.value),
        train_lr_point_count=len(lr_points),
        val_metrics_path=None if exact is None else exact.metrics_path,
        first_train_lr_metrics_path=None if first_lr is None else first_lr.metrics_path,
    )


def reset_checkpoint_timer_state(checkpoint_path: str | Path) -> TimerResetResult:
    """REQ-LEARN-4135: reset only Lightning Timer elapsed state before resume."""

    path = Path(checkpoint_path)
    if not path.exists():
        return TimerResetResult(False, f"missing checkpoint: {path}", None)
    try:
        import torch  # pylint: disable=import-outside-toplevel

        try:
            payload = safe_torch_load(path, map_location="cpu", allow_unsafe_pickle=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return TimerResetResult(False, f"{type(exc).__name__}: {exc}", None)
    if not isinstance(payload, dict):
        return TimerResetResult(
            False, f"unexpected checkpoint payload: {type(payload).__name__}", None
        )

    manual_step = _float_or_none(payload.get("nano_trm_manual_lr_step"))
    callbacks = payload.get("callbacks")
    timer = callbacks.get("Timer") if isinstance(callbacks, dict) else None
    elapsed = timer.get("time_elapsed") if isinstance(timer, dict) else None
    if not isinstance(elapsed, dict):
        return TimerResetResult(
            False,
            "no Lightning Timer time_elapsed state found",
            None if manual_step is None else int(manual_step),
        )

    changed = False
    for key in ("train", "sanity_check", "validate", "test", "predict"):
        if elapsed.get(key) != 0.0:
            elapsed[key] = 0.0
            changed = True
    if changed:
        torch.save(payload, path)
        detail = "reset Lightning Timer time_elapsed to zero"
    else:
        detail = "Lightning Timer time_elapsed already zero"
    return TimerResetResult(changed, detail, None if manual_step is None else int(manual_step))


def check_preconditions(
    config: Exp4135Config,
    *,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """SCENARIO-LEARN-4135-BLOCKED: check all no-fabrication prerequisites."""

    checks = [
        exp4107._check_uv(uv_resolver),
        exp4107._check_trainer(Path(config.repo_root)),
        exp4107._check_cuda(cuda_checker),
    ]
    stable_check = exp4127.check_stable_checkpoint(
        config.to_4127_config(),
        checkpoint_loader=checkpoint_loader,
    )
    checks.append(stable_check)
    if not checks[0].available:
        return checks, "blocked_uv_missing"
    if not checks[1].available:
        return checks, "blocked_nanotrm_train_missing"
    if not checks[2].available:
        return checks, "blocked_cuda_unavailable"
    if not stable_check.available:
        return checks, "blocked_stable_checkpoint_missing"
    return checks, None


def verify_completed_resume_pass(
    config: Exp4135Config,
    *,
    duration_s: float,
    return_code: int = 0,
    command: Sequence[str] | None = None,
    stdout_tail: Sequence[str] | None = None,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:
    """SCENARIO-LEARN-4135: verify the stable checkpoint and pass metrics."""

    return exp4127.verify_completed_resume_pass(
        config.to_4127_config(),
        PASS_INDEX,
        duration_s=duration_s,
        return_code=return_code,
        command=command,
        stdout_tail=stdout_tail,
        checkpoint_loader=checkpoint_loader,
    )


def run_native_resume_pass(
    config: Exp4135Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches native trainer.
    """Run one native nano-trm fixed-LR resume pass with the one-hour cap."""

    print(
        f"[exp4135] launching pass1 fixed-LR resume stable={config.stable_checkpoint_path}",
        flush=True,
    )
    return exp4127.run_native_resume_pass(
        config.to_4127_config(),
        PASS_INDEX,
        checkpoint_loader=checkpoint_loader,
    )


def generate_sudoku_extreme_dataset_if_missing(config: Exp4135Config) -> bool:  # pragma: no cover
    """Generate the nano-trm Sudoku Extreme dataset only when it is absent."""

    return exp4116.generate_sudoku_extreme_dataset_if_missing(config.to_4116_config())


def _verdict_for_pass(val_exact_accuracy: float | None, delta_vs_previous: float | None) -> str:
    if val_exact_accuracy is None:
        return "complete: missing_real_val_exact_accuracy"
    if matches_published_087(val_exact_accuracy):
        return f"complete: val={val_exact_accuracy:.4f} reproduced_within_0.02_of_0.87"
    if delta_vs_previous is not None and delta_vs_previous > 0:
        return (
            f"complete: val={val_exact_accuracy:.4f} improved_delta={delta_vs_previous:.4f} "
            "still_below_0.87 -> pass2 continues"
        )
    delta = 0.0 if delta_vs_previous is None else delta_vs_previous
    return (
        f"complete: val={val_exact_accuracy:.4f} stalled_delta={delta:.4f} "
        "still_below_0.87 -> stall_flagged"
    )


def _acceptance_gate(
    *,
    val_exact_accuracy: float | None,
    delta_vs_previous: float | None,
    lr_continued: bool,
    honest_verdict: str,
) -> bool:
    return bool(
        val_exact_accuracy is not None
        and lr_continued
        and (
            (delta_vs_previous is not None and delta_vs_previous > 0)
            or "stall_flagged" in honest_verdict
        )
    )


def build_result_artifact(
    *,
    run_config: Exp4135Config,
    run_result: exp4116.ResumeRunResult,
    pass_metrics: PassMetricSummary,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    lr_fix_artifact: Mapping[str, Any],
    dataset_generated: bool,
    checkpoint_timer_reset: TimerResetResult | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4135: build the measured single-pass artifact."""

    val = pass_metrics.val_exact_accuracy
    delta = None if val is None else val - STARTING_VAL_EXACT_ACCURACY
    rounded_delta = _rounded(delta)
    lr_continued = pass_metrics.lr_continued_not_rewarmed
    verdict = _verdict_for_pass(val, rounded_delta)
    artifact = {
        "experiment": "experiment_4135_sudoku_accumulate_pass1_fixed_lr",
        "schema": "carnot.experiment_4135_sudoku_accumulate_pass1_fixed_lr.v1",
        "spec_refs": ["REQ-LEARN-4135", "SCENARIO-LEARN-4135"],
        "honest_verdict": verdict,
        "val_exact_accuracy": _rounded(val),
        "delta_vs_previous": rounded_delta,
        "lr_continued_not_rewarmed": bool(lr_continued),
        "matches_published_087": matches_published_087(val),
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "model_specs": model_specs(run_config),
        "duration_s": round(float(run_result.duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": _acceptance_gate(
            val_exact_accuracy=val,
            delta_vs_previous=rounded_delta,
            lr_continued=lr_continued,
            honest_verdict=verdict,
        ),
        "starting_val_exact_accuracy": STARTING_VAL_EXACT_ACCURACY,
        "pass_index": PASS_INDEX,
        "return_code": int(run_result.return_code),
        "checkpoint_reload_ok": bool(run_result.checkpoint_reload_ok),
        "checkpoint_reload_detail": run_result.checkpoint_reload_detail,
        "exact_accuracy_metric": "val/exact_accuracy" if val is not None else None,
        "exact_accuracy_metrics_path": (
            None if pass_metrics.val_metrics_path is None else str(pass_metrics.val_metrics_path)
        ),
        "validation_first_lr": pass_metrics.first_train_lr,
        "fresh_warmup_lr": exp4126.FRESH_WARMUP_FIRST_LR,
        "train_lr_point_count": int(pass_metrics.train_lr_point_count),
        "first_train_lr_metrics_path": (
            None
            if pass_metrics.first_train_lr_metrics_path is None
            else str(pass_metrics.first_train_lr_metrics_path)
        ),
        "pass_metrics": pass_metrics.to_dict(),
        "run_dir": str(run_result.run_dir),
        "command": list(run_result.command),
        "stdout_tail": list(run_result.stdout_tail[-60:]),
        "dataset_dir": str(run_config.dataset_dir),
        "dataset_generated": bool(dataset_generated),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "lr_fix_artifact": dict(lr_fix_artifact),
        "checkpoint_timer_reset": (
            None
            if checkpoint_timer_reset is None
            else (
                checkpoint_timer_reset.to_dict()
                if isinstance(checkpoint_timer_reset, TimerResetResult)
                else dict(checkpoint_timer_reset)
            )
        ),
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    run_config: Exp4135Config,
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float = 0.0,
    lr_fix_artifact: Mapping[str, Any] | None = None,
    checkpoint_timer_reset: TimerResetResult | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4135-BLOCKED: build a no-training artifact."""

    artifact = {
        "experiment": "experiment_4135_sudoku_accumulate_pass1_fixed_lr",
        "schema": "carnot.experiment_4135_sudoku_accumulate_pass1_fixed_lr.v1",
        "spec_refs": ["REQ-LEARN-4135", "SCENARIO-LEARN-4135-BLOCKED"],
        "honest_verdict": reason,
        "val_exact_accuracy": None,
        "delta_vs_previous": None,
        "lr_continued_not_rewarmed": False,
        "matches_published_087": False,
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "model_specs": model_specs(run_config),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
        "starting_val_exact_accuracy": STARTING_VAL_EXACT_ACCURACY,
        "pass_index": PASS_INDEX,
        "return_code": None,
        "checkpoint_reload_ok": False,
        "checkpoint_reload_detail": reason,
        "exact_accuracy_metric": None,
        "exact_accuracy_metrics_path": None,
        "validation_first_lr": None,
        "fresh_warmup_lr": exp4126.FRESH_WARMUP_FIRST_LR,
        "train_lr_point_count": 0,
        "first_train_lr_metrics_path": None,
        "pass_metrics": PassMetricSummary(None, None, 0, None, None).to_dict(),
        "run_dir": str(run_config.pass_run_dir()),
        "command": [],
        "stdout_tail": [],
        "dataset_dir": str(run_config.dataset_dir),
        "dataset_generated": False,
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "lr_fix_artifact": dict(lr_fix_artifact or {}),
        "checkpoint_timer_reset": (
            None
            if checkpoint_timer_reset is None
            else (
                checkpoint_timer_reset.to_dict()
                if isinstance(checkpoint_timer_reset, TimerResetResult)
                else dict(checkpoint_timer_reset)
            )
        ),
    }
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4135 deliverable."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    val = artifact.get("val_exact_accuracy")
    if val is not None:
        number = (
            _float_or_none(val)
            if isinstance(val, (int, float)) and not isinstance(val, bool)
            else None
        )
        if number is None or not 0.0 <= number <= 1.0:
            errors.append("val_exact_accuracy must be numeric between 0 and 1 or null")

    delta = artifact.get("delta_vs_previous")
    if delta is not None and (
        not isinstance(delta, (int, float))
        or isinstance(delta, bool)
        or _float_or_none(delta) is None
    ):
        errors.append("delta_vs_previous must be numeric or null")

    if not isinstance(artifact.get("lr_continued_not_rewarmed"), bool):
        errors.append("lr_continued_not_rewarmed must be a bare bool")

    if not isinstance(artifact.get("matches_published_087"), bool):
        errors.append("matches_published_087 must be a bare bool")

    stable_checkpoint_path = artifact.get("stable_checkpoint_path")
    if not isinstance(stable_checkpoint_path, str) or not stable_checkpoint_path.endswith(
        "results/trm_runs/sudoku_extreme_baseline/last.ckpt"
    ):
        errors.append(
            "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path"
        )

    if not isinstance(artifact.get("random_seed"), int) or isinstance(
        artifact.get("random_seed"), bool
    ):
        errors.append("random_seed must be a bare int")

    checksum = artifact.get("reproducibility_checksum")
    if not (isinstance(checksum, str) and checksum.startswith("sha256:") and len(checksum) == 71):
        errors.append("reproducibility_checksum must be sha256-prefixed")

    specs = artifact.get("model_specs")
    if not (
        isinstance(specs, Mapping)
        and specs.get("model") == "nano-trm"
        and specs.get("experiment_config") == "trm_sudoku_extreme_1k_aug_1k"
    ):
        errors.append("model_specs must name nano-trm and trm_sudoku_extreme_1k_aug_1k")

    duration = _float_or_none(artifact.get("duration_s"))
    if duration is None or duration < 0 or duration >= 4_800:
        errors.append("duration_s must be a scalar bounded number below 4800")

    if artifact.get("matches_published_087") is True and not matches_published_087(
        _float_or_none(val)
    ):
        errors.append("matches_published_087=true requires val within 0.02 of 0.87")

    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if gate is True:
        if not _acceptance_gate(
            val_exact_accuracy=_float_or_none(val),
            delta_vs_previous=_float_or_none(delta),
            lr_continued=artifact.get("lr_continued_not_rewarmed") is True,
            honest_verdict=str(verdict),
        ):
            errors.append(
                "acceptance_gate_passed=true requires val, LR continuity, and improvement or stall verdict"
            )

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
    dataset_builder: Callable[[Exp4135Config], object] = generate_sudoku_extreme_dataset_if_missing,
    trainer_runner: Callable[[Exp4135Config], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4135 or write the required honest blocked artifact."""

    started = time.time()
    root = Path(repo_root)
    lr_path = (
        Path(lr_artifact_path)
        if lr_artifact_path is not None
        else root / "results" / exp4126.RESULT_FILENAME
    )
    config = Exp4135Config(
        repo_root=root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_root=hydra_run_root,
        lr_artifact_path=lr_path,
        random_seed=random_seed,
    )
    out = Path(output_path)

    checks, blocker = check_preconditions(
        config,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )
    lr_fix_artifact = exp4127.load_lr_fix_artifact(lr_path)
    if blocker is not None:
        artifact = build_blocked_artifact(
            blocker,
            run_config=config,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            lr_fix_artifact=lr_fix_artifact,
        )
        _write_json(out, artifact)
        return artifact

    if not exp4127.lr_fix_landed(lr_fix_artifact):
        artifact = build_blocked_artifact(
            "blocked_lr_fix_not_landed",
            run_config=config,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            lr_fix_artifact=lr_fix_artifact,
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
            run_config=config,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            lr_fix_artifact=lr_fix_artifact,
        )
        artifact["dataset_generated"] = dataset_generated
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    checkpoint_timer_reset = reset_checkpoint_timer_state(config.stable_checkpoint_path)
    try:
        if trainer_runner is None:  # pragma: no cover - launches native trainer.
            run_result = run_native_resume_pass(config, checkpoint_loader=checkpoint_loader)
        else:
            run_result = trainer_runner(config)
    except Exception as exc:
        run_result = exp4116.ResumeRunResult(
            return_code=1,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=False,
            checkpoint_reload_detail=f"{type(exc).__name__}: {exc}",
            val_exact_accuracy=None,
            cumulative_epochs=None,
            duration_s=time.time() - started,
            command=build_train_command(config),
            stdout_tail=[f"{type(exc).__name__}: {exc}"],
            run_dir=config.pass_run_dir(),
        )

    pass_metrics = (
        summarize_pass_metrics(config.pass_run_dir())
        if run_result.return_code == 0 or run_result.val_exact_accuracy is not None
        else PassMetricSummary(None, None, 0, None, None)
    )
    artifact = build_result_artifact(
        run_config=config,
        run_result=run_result,
        pass_metrics=pass_metrics,
        preconditions_checked=checks,
        lr_fix_artifact=lr_fix_artifact,
        dataset_generated=dataset_generated,
        checkpoint_timer_reset=checkpoint_timer_reset,
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
