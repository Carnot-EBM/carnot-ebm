"""Exp 4136 fixed-LR nano-trm Sudoku Extreme accumulation pass 2.

Spec refs: REQ-LEARN-4136, SCENARIO-LEARN-4136,
SCENARIO-LEARN-4136-AUDIT.
"""

from __future__ import annotations

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
from carnot import experiment_4135_sudoku_accumulate_pass1_fixed_lr as exp4135


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4136_sudoku_accumulate_pass2_fixed_lr.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_HYDRA_RUN_ROOT = DEFAULT_SAVE_PARENT / "experiment_4136_sudoku_accumulate_pass2_fixed_lr"
DEFAULT_PASS1_ARTIFACT = REPO_ROOT / "results" / exp4135.RESULT_FILENAME
DEFAULT_LR_FIX_ARTIFACT = REPO_ROOT / "results" / exp4126.RESULT_FILENAME
RANDOM_SEED = exp4108.RANDOM_SEED
PASS_INDEX = 2
MAX_TIME = exp4135.MAX_TIME
LOCAL_SAFE_BATCH_SIZE = exp4135.LOCAL_SAFE_BATCH_SIZE
PUBLISHED_EXACT_ACCURACY = exp4127.PUBLISHED_EXACT_ACCURACY
PUBLISHED_TOLERANCE = exp4127.PUBLISHED_TOLERANCE
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
PassMetricSummary = exp4135.PassMetricSummary
TimerResetResult = exp4135.TimerResetResult

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_exact_accuracy",
    "delta_vs_previous",
    "plateau_audit_done",
    "matches_published_087",
    "stable_checkpoint_path",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'val=0.NN improving' OR "
        "'plateau -> config audit: <cause>' is COMPLETE."
    ),
    "val_exact_accuracy": "Val after pass2; the accumulation trajectory.",
    "delta_vs_previous": (
        "Improvement vs pass1; <=0 triggers the config-audit branch, not "
        "another blind pass."
    ),
    "plateau_audit_done": (
        "Bare bool: True if pass1 stalled and this task audited config instead "
        "of blind-training (the Failed-Rerun-Discipline floor)."
    ),
    "matches_published_087": "Bare bool: within 0.02 of 0.87.",
    "stable_checkpoint_path": "The persisted baseline path pass3 (exp4137) resumes from.",
    "random_seed": "Determinism precondition recorded as a first-class field (silences METHODOLOGY_MISSING).",
    "reproducibility_checksum": "Content hash of (config + resumed checkpoint + data dir); catches silent drift.",
    "duration_s": "Bounded GPU run < 4800s (single float), or a short audit (< 600s) if the plateau branch fired.",
}

TIMER_METRIC_CAUSE = (
    "pass1 did not report positive accumulation; its stdout shows Lightning "
    "stopped immediately on an already-elapsed Timer before train/lr or "
    "val/exact_accuracy metrics were written"
)
CORRECTED_CONFIG_RECOMMENDATION = (
    "rerun Exp 4135 from the stable checkpoint after resetting or removing the "
    "Lightning Timer elapsed callback state before Trainer.fit; keep "
    "scheduler horizon max_epochs=50000, peak_lr=1e-4, "
    "data.data_dir=./data/sudoku_extreme_1k_aug_1k, and local safe "
    "timekeeping.batch_size=128 unless memory allows the published batch 768"
)


@dataclass(frozen=True)
class Exp4136Config:
    """Filesystem and Hydra settings for the second fixed-LR accumulation pass."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_root: Path | str | None = None
    dataset_dir: Path | str | None = None
    pass1_artifact_path: Path | str = DEFAULT_PASS1_ARTIFACT
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
            Path(self.hydra_run_root) if self.hydra_run_root is not None else parent / Path(DEFAULT_HYDRA_RUN_ROOT).name
        )
        dataset = (
            Path(self.dataset_dir)
            if self.dataset_dir is not None
            else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        )
        pass1_path = Path(self.pass1_artifact_path)
        if pass1_path == DEFAULT_PASS1_ARTIFACT and root != REPO_ROOT:
            pass1_path = root / "results" / exp4135.RESULT_FILENAME
        lr_artifact = Path(self.lr_artifact_path)
        if lr_artifact == DEFAULT_LR_FIX_ARTIFACT and root != REPO_ROOT:
            lr_artifact = root / "results" / exp4126.RESULT_FILENAME
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "hydra_run_root", hydra_root)
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(self, "pass1_artifact_path", pass1_path)
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
        return Path(self.nano_trm_root) / "src" / "nn" / "configs" / "data" / "sudoku_extreme_1k_aug1k.yaml"

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
class ConfigAuditSummary:
    """Short pass1 audit used when pass2 must not burn another blind resume."""

    suspected_cause: str
    corrected_config_recommendation: str
    scheduler_horizon_vs_published_recipe: dict[str, Any]
    peak_lr_vs_published_recipe: dict[str, Any]
    data_path_vs_published_recipe: dict[str, Any]
    batch_size_vs_published_recipe: dict[str, Any]
    pass1_evidence: dict[str, Any]

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


def _checks_to_dicts(checks: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [check.to_dict() if isinstance(check, exp4107.PreconditionCheck) else dict(check) for check in checks]


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml  # pylint: disable=import-outside-toplevel

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def load_pass1_artifact(path: str | Path) -> dict[str, Any]:
    """REQ-LEARN-4136: defensively read Exp 4135 before any training."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def pass1_allows_training(artifact: Mapping[str, Any]) -> bool:
    """SCENARIO-LEARN-4136-AUDIT: require numeric pass1 val and positive delta."""

    val = _float_or_none(artifact.get("val_exact_accuracy"))
    delta = _float_or_none(artifact.get("delta_vs_previous"))
    return val is not None and delta is not None and delta > 0


def lr_fix_landed(artifact: Mapping[str, Any]) -> bool:
    """REQ-LEARN-4136: only a bare true LR-continuity gate permits training."""

    return exp4127.lr_fix_landed(artifact)


def matches_published_087(value: float | None) -> bool:
    """REQ-LEARN-4136: compare validation exact accuracy to 0.87 within 0.02."""

    return value is not None and abs(float(value) - PUBLISHED_EXACT_ACCURACY) <= PUBLISHED_TOLERANCE + 1e-12


def build_train_command(config: Exp4136Config) -> list[str]:
    """REQ-LEARN-4136: build the single fixed-LR bounded pass2 resume command."""

    return exp4127.build_train_command(config.to_4127_config(), PASS_INDEX)


def build_train_env(config: Exp4136Config) -> dict[str, str]:
    """REQ-LEARN-4136: disable compile/CUDAGraph resume while keeping CUDA enabled."""

    return exp4127.build_train_env(config.to_4127_config())


def model_specs(config: Exp4136Config) -> dict[str, Any]:
    """REQ-LEARN-4136: name the native model/config inputs used for training."""

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
        "pass_index": PASS_INDEX,
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
        children = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item))
        for child in children:
            _hash_file(hasher, path, child)
        return
    hasher.update(f"missing:{path}".encode("utf-8"))
    hasher.update(b"\0")


def compute_reproducibility_checksum(config: Exp4136Config) -> str:
    """REQ-LEARN-4136: hash config, stable checkpoint, and Sudoku data inputs."""

    hasher = hashlib.sha256()
    hasher.update(b"carnot.exp4136.sudoku_accumulate_pass2_fixed_lr.v1\0")
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


def summarize_pass_metrics(run_dir: str | Path) -> exp4135.PassMetricSummary:
    """SCENARIO-LEARN-4136: read validation exact accuracy and first train/lr."""

    return exp4135.summarize_pass_metrics(run_dir)


def _suspected_cause(pass1_artifact: Mapping[str, Any]) -> str:
    val = _float_or_none(pass1_artifact.get("val_exact_accuracy"))
    delta = _float_or_none(pass1_artifact.get("delta_vs_previous"))
    tail = "\n".join(str(line) for line in pass1_artifact.get("stdout_tail", []) if line is not None)
    short_run = (_float_or_none(pass1_artifact.get("duration_s")) or 0.0) < 600
    if val is None or delta is None:
        if short_run and "Time limit reached" in tail and "metric missing" in tail:
            return TIMER_METRIC_CAUSE
        return "pass1 did not report positive accumulation; val_exact_accuracy or delta_vs_previous was missing"
    if delta <= 0:
        return f"pass1 did not report positive accumulation; delta_vs_previous={delta:.6g}"
    return "pass1 reported positive accumulation"


def audit_pass1_config(config: Exp4136Config, pass1_artifact: Mapping[str, Any]) -> ConfigAuditSummary:
    """SCENARIO-LEARN-4136-AUDIT: audit config instead of blind-training."""

    experiment_config = _read_yaml_mapping(config.experiment_config_path)
    data_config = _read_yaml_mapping(config.data_config_path)
    command = build_train_command(config)
    command_data_path = next((item.split("=", 1)[1] for item in command if item.startswith("data.data_dir=")), None)
    max_epochs = _float_or_none(_nested(experiment_config, "timekeeping", "max_epochs"))
    warmup_steps = _float_or_none(_nested(experiment_config, "model_tuning", "warmup_steps"))
    peak_lr = _float_or_none(_nested(experiment_config, "model_tuning", "learning_rate"))
    published_batch = _float_or_none(_nested(experiment_config, "timekeeping", "batch_size"))
    configured_data_path = _nested(data_config, "data_dir")
    return ConfigAuditSummary(
        suspected_cause=_suspected_cause(pass1_artifact),
        corrected_config_recommendation=CORRECTED_CONFIG_RECOMMENDATION,
        scheduler_horizon_vs_published_recipe={
            "actual_max_epochs": None if max_epochs is None else int(max_epochs),
            "expected_max_epochs": 50000,
            "warmup_steps": None if warmup_steps is None else int(warmup_steps),
            "trainer_max_time": str(config.max_time),
            "matches_expected": max_epochs == 50000 and warmup_steps == 2000,
        },
        peak_lr_vs_published_recipe={
            "actual": peak_lr,
            "expected": 1e-4,
            "matches_expected": peak_lr == 1e-4,
        },
        data_path_vs_published_recipe={
            "actual": command_data_path,
            "config_default": configured_data_path,
            "expected": "./data/sudoku_extreme_1k_aug_1k",
            "matches_expected": command_data_path == "./data/sudoku_extreme_1k_aug_1k",
        },
        batch_size_vs_published_recipe={
            "actual": int(config.batch_size),
            "published_config": None if published_batch is None else int(published_batch),
            "expected_local_safe": LOCAL_SAFE_BATCH_SIZE,
            "matches_local_safe": int(config.batch_size) == LOCAL_SAFE_BATCH_SIZE,
        },
        pass1_evidence={
            "honest_verdict": pass1_artifact.get("honest_verdict"),
            "val_exact_accuracy": pass1_artifact.get("val_exact_accuracy"),
            "delta_vs_previous": pass1_artifact.get("delta_vs_previous"),
            "duration_s": pass1_artifact.get("duration_s"),
            "train_lr_point_count": pass1_artifact.get("train_lr_point_count"),
            "exact_accuracy_metrics_path": pass1_artifact.get("exact_accuracy_metrics_path"),
        },
    )


def check_preconditions(
    config: Exp4136Config,
    *,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """SCENARIO-LEARN-4136: check all no-fabrication runtime prerequisites."""

    return exp4135.check_preconditions(
        exp4135.Exp4135Config(
            repo_root=config.repo_root,
            nano_trm_root=config.nano_trm_root,
            save_parent=config.save_parent,
            stable_dir=config.stable_dir,
            hydra_run_root=config.hydra_run_root,
            dataset_dir=config.dataset_dir,
            lr_artifact_path=config.lr_artifact_path,
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


def verify_completed_resume_pass(
    config: Exp4136Config,
    *,
    duration_s: float,
    return_code: int = 0,
    command: Sequence[str] | None = None,
    stdout_tail: Sequence[str] | None = None,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:
    """SCENARIO-LEARN-4136: verify the stable checkpoint and pass2 metrics."""

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
    config: Exp4136Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches native trainer.
    """Run one native nano-trm fixed-LR resume pass with a one-hour cap."""

    print(
        f"[exp4136] launching pass2 fixed-LR resume stable={config.stable_checkpoint_path}",
        flush=True,
    )
    return exp4127.run_native_resume_pass(
        config.to_4127_config(),
        PASS_INDEX,
        checkpoint_loader=checkpoint_loader,
    )


def generate_sudoku_extreme_dataset_if_missing(config: Exp4136Config) -> bool:  # pragma: no cover
    """Generate the nano-trm Sudoku Extreme dataset only when it is absent."""

    return exp4116.generate_sudoku_extreme_dataset_if_missing(config.to_4116_config())


def _verdict_for_pass(val_exact_accuracy: float | None, delta_vs_previous: float | None) -> str:
    if val_exact_accuracy is None:
        return "complete: missing_real_val_exact_accuracy"
    if matches_published_087(val_exact_accuracy):
        return f"complete: val={val_exact_accuracy:.4f} reproduced_within_0.02_of_0.87"
    if delta_vs_previous is not None and delta_vs_previous > 0:
        return (
            f"complete: val={val_exact_accuracy:.4f} improving_delta={delta_vs_previous:.4f} "
            "still_below_0.87 -> pass3 continues"
        )
    delta = 0.0 if delta_vs_previous is None else delta_vs_previous
    return f"complete: val={val_exact_accuracy:.4f} plateau_delta={delta:.4f} -> config audit before pass3"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("plateau_audit_done") is True:
        duration = _float_or_none(artifact.get("duration_s"))
        return bool(
            duration is not None
            and duration < 600
            and artifact.get("suspected_cause")
            and artifact.get("corrected_config_recommendation")
        )
    val = _float_or_none(artifact.get("val_exact_accuracy"))
    delta = _float_or_none(artifact.get("delta_vs_previous"))
    return bool(val is not None and delta is not None and artifact.get("lr_continued_not_rewarmed") is True)


def build_audit_artifact(
    *,
    run_config: Exp4136Config,
    pass1_artifact: Mapping[str, Any],
    config_audit: ConfigAuditSummary | Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4136-AUDIT: build a no-training config-audit artifact."""

    audit = config_audit.to_dict() if isinstance(config_audit, ConfigAuditSummary) else dict(config_audit)
    artifact = {
        "experiment": "experiment_4136_sudoku_accumulate_pass2_fixed_lr",
        "schema": "carnot.experiment_4136_sudoku_accumulate_pass2_fixed_lr.v1",
        "spec_refs": ["REQ-LEARN-4136", "SCENARIO-LEARN-4136-AUDIT"],
        "honest_verdict": f"complete: plateau -> config audit: {audit.get('suspected_cause')}",
        "val_exact_accuracy": None,
        "delta_vs_previous": None,
        "plateau_audit_done": True,
        "matches_published_087": False,
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
        "suspected_cause": str(audit.get("suspected_cause")),
        "corrected_config_recommendation": str(audit.get("corrected_config_recommendation")),
        "config_audit": audit,
        "pass1_artifact_path": str(run_config.pass1_artifact_path),
        "pass1_artifact": dict(pass1_artifact),
        "model_specs": model_specs(run_config),
        "pass_index": PASS_INDEX,
        "command": build_train_command(run_config),
    }
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    run_config: Exp4136Config,
    pass1_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    lr_fix_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-4136: build a no-training blocked artifact after pass1 passed."""

    artifact = {
        "experiment": "experiment_4136_sudoku_accumulate_pass2_fixed_lr",
        "schema": "carnot.experiment_4136_sudoku_accumulate_pass2_fixed_lr.v1",
        "spec_refs": ["REQ-LEARN-4136"],
        "honest_verdict": reason,
        "val_exact_accuracy": None,
        "delta_vs_previous": None,
        "plateau_audit_done": False,
        "matches_published_087": False,
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
        "pass1_artifact_path": str(run_config.pass1_artifact_path),
        "pass1_artifact": dict(pass1_artifact),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "lr_fix_artifact": dict(lr_fix_artifact or {}),
        "model_specs": model_specs(run_config),
        "pass_index": PASS_INDEX,
        "command": build_train_command(run_config),
    }
    validate_artifact(artifact)
    return artifact


def build_result_artifact(
    *,
    run_config: Exp4136Config,
    run_result: exp4116.ResumeRunResult,
    pass_metrics: exp4135.PassMetricSummary,
    pass1_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    lr_fix_artifact: Mapping[str, Any],
    dataset_generated: bool,
    checkpoint_timer_reset: exp4135.TimerResetResult | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4136: build the measured pass2 artifact."""

    val = pass_metrics.val_exact_accuracy
    previous = _float_or_none(pass1_artifact.get("val_exact_accuracy"))
    delta = None if val is None or previous is None else val - previous
    rounded_delta = _rounded(delta)
    artifact = {
        "experiment": "experiment_4136_sudoku_accumulate_pass2_fixed_lr",
        "schema": "carnot.experiment_4136_sudoku_accumulate_pass2_fixed_lr.v1",
        "spec_refs": ["REQ-LEARN-4136", "SCENARIO-LEARN-4136"],
        "honest_verdict": _verdict_for_pass(val, rounded_delta),
        "val_exact_accuracy": _rounded(val),
        "delta_vs_previous": rounded_delta,
        "plateau_audit_done": False,
        "matches_published_087": matches_published_087(val),
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "duration_s": round(float(run_result.duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
        "lr_continued_not_rewarmed": bool(pass_metrics.lr_continued_not_rewarmed),
        "pass1_val_exact_accuracy": _rounded(previous),
        "pass1_artifact_path": str(run_config.pass1_artifact_path),
        "pass1_artifact": dict(pass1_artifact),
        "model_specs": model_specs(run_config),
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
                if isinstance(checkpoint_timer_reset, exp4135.TimerResetResult)
                else dict(checkpoint_timer_reset)
            )
        ),
    }
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4136 deliverable."""

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
        number = _float_or_none(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None
        if number is None or not 0.0 <= number <= 1.0:
            errors.append("val_exact_accuracy must be numeric between 0 and 1 or null")

    delta = artifact.get("delta_vs_previous")
    if delta is not None and (
        not isinstance(delta, (int, float)) or isinstance(delta, bool) or _float_or_none(delta) is None
    ):
        errors.append("delta_vs_previous must be numeric or null")

    if not isinstance(artifact.get("plateau_audit_done"), bool):
        errors.append("plateau_audit_done must be a bare bool")

    if not isinstance(artifact.get("matches_published_087"), bool):
        errors.append("matches_published_087 must be a bare bool")

    stable_checkpoint_path = artifact.get("stable_checkpoint_path")
    if not isinstance(stable_checkpoint_path, str) or not stable_checkpoint_path.endswith(
        "results/trm_runs/sudoku_extreme_baseline/last.ckpt"
    ):
        errors.append("stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path")

    if not isinstance(artifact.get("random_seed"), int) or isinstance(artifact.get("random_seed"), bool):
        errors.append("random_seed must be a bare int")

    checksum = artifact.get("reproducibility_checksum")
    if not (isinstance(checksum, str) and checksum.startswith("sha256:") and len(checksum) == 71):
        errors.append("reproducibility_checksum must be sha256-prefixed")

    duration = _float_or_none(artifact.get("duration_s"))
    if duration is None or duration < 0 or duration >= 4_800:
        errors.append("duration_s must be a scalar bounded number below 4800")
    if artifact.get("plateau_audit_done") is True and (duration is None or duration >= 600):
        errors.append("plateau audit duration_s must be below 600")

    if artifact.get("matches_published_087") is True and not matches_published_087(_float_or_none(val)):
        errors.append("matches_published_087=true requires val within 0.02 of 0.87")

    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if gate is True:
        if artifact.get("plateau_audit_done") is True:
            if not artifact.get("suspected_cause") or not artifact.get("corrected_config_recommendation"):
                errors.append("audit acceptance requires suspected_cause and corrected_config_recommendation")
        elif not (_float_or_none(val) is not None and _float_or_none(delta) is not None):
            errors.append("training acceptance requires val_exact_accuracy and delta_vs_previous")

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
    pass1_artifact_path: str | Path | None = None,
    lr_artifact_path: str | Path | None = None,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    dataset_builder: Callable[[Exp4136Config], object] = generate_sudoku_extreme_dataset_if_missing,
    trainer_runner: Callable[[Exp4136Config], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4136 or write the required pass1-stall audit artifact."""

    started = time.time()
    root = Path(repo_root)
    pass1_path = Path(pass1_artifact_path) if pass1_artifact_path is not None else root / "results" / exp4135.RESULT_FILENAME
    lr_path = Path(lr_artifact_path) if lr_artifact_path is not None else root / "results" / exp4126.RESULT_FILENAME
    config = Exp4136Config(
        repo_root=root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_root=hydra_run_root,
        pass1_artifact_path=pass1_path,
        lr_artifact_path=lr_path,
        random_seed=random_seed,
    )
    out = Path(output_path)
    pass1_artifact = load_pass1_artifact(config.pass1_artifact_path)
    if not pass1_allows_training(pass1_artifact):
        config_audit = audit_pass1_config(config, pass1_artifact)
        artifact = build_audit_artifact(
            run_config=config,
            pass1_artifact=pass1_artifact,
            config_audit=config_audit,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact

    lr_fix_artifact = exp4127.load_lr_fix_artifact(lr_path)
    if not lr_fix_landed(lr_fix_artifact):
        artifact = build_blocked_artifact(
            "blocked_lr_fix_not_landed",
            run_config=config,
            pass1_artifact=pass1_artifact,
            preconditions_checked=[],
            duration_s=time.time() - started,
            lr_fix_artifact=lr_fix_artifact,
        )
        _write_json(out, artifact)
        return artifact

    checks, blocker = check_preconditions(
        config,
        uv_resolver=uv_resolver,
        cuda_checker=cuda_checker,
        checkpoint_loader=checkpoint_loader,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            blocker,
            run_config=config,
            pass1_artifact=pass1_artifact,
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
            pass1_artifact=pass1_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            lr_fix_artifact=lr_fix_artifact,
        )
        artifact["dataset_dir"] = str(config.dataset_dir)
        artifact["dataset_generated"] = dataset_generated
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact

    checkpoint_timer_reset = exp4135.reset_checkpoint_timer_state(config.stable_checkpoint_path)
    try:
        if trainer_runner is None:  # pragma: no cover - launches native trainer.
            run_result = run_native_resume_pass(config, checkpoint_loader=checkpoint_loader)
        else:
            run_result = trainer_runner(config)
    except Exception as exc:  # pragma: no cover - defensive native failure path.
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
        else exp4135.PassMetricSummary(None, None, 0, None, None)
    )
    artifact = build_result_artifact(
        run_config=config,
        run_result=run_result,
        pass_metrics=pass_metrics,
        pass1_artifact=pass1_artifact,
        preconditions_checked=checks,
        lr_fix_artifact=lr_fix_artifact,
        dataset_generated=dataset_generated,
        checkpoint_timer_reset=checkpoint_timer_reset,
    )
    _write_json(out, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps({field: artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS}, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
