"""Exp 4138 fixed-LR nano-trm Sudoku Extreme accumulation pass 4.

Spec refs: REQ-LEARN-4138, SCENARIO-LEARN-4138,
SCENARIO-LEARN-4138-AUDIT.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4108_nanotrm_sudoku_extreme_baseline as exp4108
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4126_lr_resume_correctness_fix as exp4126
from carnot import experiment_4127_sudoku_extreme_accumulate_fixed as exp4127
from carnot import experiment_4135_sudoku_accumulate_pass1_fixed_lr as exp4135
from carnot import experiment_4136_sudoku_accumulate_pass2_fixed_lr as exp4136
from carnot import experiment_4137_sudoku_accumulate_pass3_fixed_lr as exp4137


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4138_sudoku_accumulate_pass4_convergence_check.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_SAVE_PARENT = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_DIR = DEFAULT_SAVE_PARENT / "sudoku_extreme_baseline"
DEFAULT_HYDRA_RUN_ROOT = DEFAULT_SAVE_PARENT / "experiment_4138_sudoku_accumulate_pass4_convergence_check"
DEFAULT_PASS3_ARTIFACT = REPO_ROOT / "results" / exp4137.RESULT_FILENAME
DEFAULT_EXP4127_ARTIFACT = REPO_ROOT / "results" / exp4127.RESULT_FILENAME
DEFAULT_LR_FIX_ARTIFACT = REPO_ROOT / "results" / exp4126.RESULT_FILENAME
RANDOM_SEED = exp4137.RANDOM_SEED
PASS_INDEX = 4
MAX_TIME = exp4137.MAX_TIME
LOCAL_SAFE_BATCH_SIZE = exp4137.LOCAL_SAFE_BATCH_SIZE
PUBLISHED_EXACT_ACCURACY = exp4127.PUBLISHED_EXACT_ACCURACY
PUBLISHED_TOLERANCE = exp4127.PUBLISHED_TOLERANCE
NEAR_FAITHFUL_THRESHOLD = 0.80
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
PassMetricSummary = exp4135.PassMetricSummary
TimerResetResult = exp4135.TimerResetResult

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "val_exact_accuracy",
    "val_trajectory_383",
    "matches_published_087",
    "near_faithful_080",
    "estimated_passes_to_converge",
    "stable_checkpoint_path",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest 'val=0.NN faithful, graft runs full' OR "
        "'val=0.NN near-faithful, RFT de-confound runs' OR "
        "'val=0.NN, .384 finishes convergence' is COMPLETE."
    ),
    "val_exact_accuracy": "Val after pass4; the baseline the graft (exp4139) reads.",
    "val_trajectory_383": (
        "Per-pass val across .382+.383; the load-bearing evidence the fixed schedule "
        "converges across 4 passes."
    ),
    "matches_published_087": (
        "Bare bool: within 0.02 of 0.87. THE gate exp4139 reads for the FULL graft "
        "(rerank + RFT)."
    ),
    "near_faithful_080": (
        "Bare bool: val>=0.80. The gate exp4139 reads to run the RFT LABEL de-confound "
        "on a near-faithful baseline (the de-confound is valid below 0.87 as long as "
        "there is headroom)."
    ),
    "estimated_passes_to_converge": (
        "If still <0.80, the passes-to-0.87 estimate for .384 (so the next planner "
        "sizes the remaining work)."
    ),
    "stable_checkpoint_path": "The faithful/near-faithful baseline checkpoint exp4139 grafts onto.",
    "random_seed": "Determinism precondition recorded as a first-class field (silences METHODOLOGY_MISSING).",
    "reproducibility_checksum": (
        "Content hash of (config + resumed checkpoint + data dir); catches silent drift."
    ),
    "duration_s": "Bounded GPU run < 4800s (single float).",
}

FALLBACK_CORRECTED_CONFIG_RECOMMENDATION = exp4137.FALLBACK_CORRECTED_CONFIG_RECOMMENDATION


@dataclass(frozen=True)
class Exp4138Config:
    """Filesystem and Hydra settings for the fourth fixed-LR accumulation pass."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    save_parent: Path | str = DEFAULT_SAVE_PARENT
    stable_dir: Path | str | None = None
    hydra_run_root: Path | str | None = None
    dataset_dir: Path | str | None = None
    pass3_artifact_path: Path | str = DEFAULT_PASS3_ARTIFACT
    exp4127_artifact_path: Path | str = DEFAULT_EXP4127_ARTIFACT
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
        hydra_root = Path(self.hydra_run_root) if self.hydra_run_root is not None else parent / DEFAULT_HYDRA_RUN_ROOT.name
        dataset = Path(self.dataset_dir) if self.dataset_dir is not None else nano_root / "data" / "sudoku_extreme_1k_aug_1k"
        pass3_path = Path(self.pass3_artifact_path)
        if pass3_path == DEFAULT_PASS3_ARTIFACT and root != REPO_ROOT:
            pass3_path = root / "results" / exp4137.RESULT_FILENAME
        exp4127_path = Path(self.exp4127_artifact_path)
        if exp4127_path == DEFAULT_EXP4127_ARTIFACT and root != REPO_ROOT:
            exp4127_path = root / "results" / exp4127.RESULT_FILENAME
        lr_artifact = Path(self.lr_artifact_path)
        if lr_artifact == DEFAULT_LR_FIX_ARTIFACT and root != REPO_ROOT:
            lr_artifact = root / "results" / exp4126.RESULT_FILENAME
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "save_parent", parent)
        object.__setattr__(self, "nano_trm_root", nano_root)
        object.__setattr__(self, "stable_dir", stable)
        object.__setattr__(self, "hydra_run_root", hydra_root)
        object.__setattr__(self, "dataset_dir", dataset)
        object.__setattr__(self, "pass3_artifact_path", pass3_path)
        object.__setattr__(self, "exp4127_artifact_path", exp4127_path)
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


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_pass3_artifact(path: str | Path) -> dict[str, Any]:
    """REQ-LEARN-4138: defensively read Exp 4137 before any training."""

    return _load_json_mapping(path)


def load_exp4127_artifact(path: str | Path) -> dict[str, Any]:
    """REQ-LEARN-4138: defensively read the .382 anchor trajectory."""

    return _load_json_mapping(path)


def pass3_allows_training(artifact: Mapping[str, Any]) -> bool:
    """SCENARIO-LEARN-4138-AUDIT: require pass3 positive accumulation."""

    if artifact.get("plateau_audit_done") is True or artifact.get("baseline_status") == "config-blocked":
        return False
    val = _float_or_none(artifact.get("val_exact_accuracy"))
    delta = _float_or_none(artifact.get("delta_vs_previous"))
    return val is not None and delta is not None and delta > 0


def matches_published_087(value: float | None) -> bool:
    """REQ-LEARN-4138: compare validation exact accuracy to 0.87 within 0.02."""

    return value is not None and abs(float(value) - PUBLISHED_EXACT_ACCURACY) <= PUBLISHED_TOLERANCE + 1e-12


def near_faithful_080(value: float | None) -> bool:
    """REQ-LEARN-4138: report whether Exp 4139 can run the near-faithful RFT de-confound."""

    return value is not None and float(value) >= NEAR_FAITHFUL_THRESHOLD


def build_train_command(config: Exp4138Config) -> list[str]:
    """REQ-LEARN-4138: build the single fixed-LR bounded pass4 resume command."""

    return exp4127.build_train_command(config.to_4127_config(), PASS_INDEX)


def build_train_env(config: Exp4138Config) -> dict[str, str]:
    """REQ-LEARN-4138: disable compile/CUDAGraph resume while keeping CUDA enabled."""

    return exp4127.build_train_env(config.to_4127_config())


def model_specs(config: Exp4138Config) -> dict[str, Any]:
    """REQ-LEARN-4138: name the native model/config inputs used for training."""

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


def compute_reproducibility_checksum(config: Exp4138Config) -> str:
    """REQ-LEARN-4138: hash config, stable checkpoint, and Sudoku data inputs."""

    hasher = hashlib.sha256()
    hasher.update(b"carnot.exp4138.sudoku_accumulate_pass4_convergence_check.v1\0")
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
        exp4136._hash_path(hasher, label, Path(path))
    return f"sha256:{hasher.hexdigest()}"


def summarize_pass_metrics(run_dir: str | Path) -> exp4135.PassMetricSummary:
    """SCENARIO-LEARN-4138: read validation exact accuracy and first train/lr."""

    return exp4135.summarize_pass_metrics(run_dir)


def _recommendation_for_384(pass3_artifact: Mapping[str, Any]) -> str:
    recommendation = str(pass3_artifact.get("corrected_config_recommendation") or FALLBACK_CORRECTED_CONFIG_RECOMMENDATION)
    if "for the .384 baseline" in recommendation:
        return recommendation
    return f"for the .384 baseline: {recommendation}"


def _last_numeric_4127_val(exp4127_artifact: Mapping[str, Any]) -> float:
    trajectory = exp4127_artifact.get("val_trajectory")
    if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
        for entry in reversed(trajectory):
            if isinstance(entry, Mapping):
                value = _float_or_none(entry.get("val_exact_accuracy"))
                if value is not None:
                    return _rounded(value) or exp4135.STARTING_VAL_EXACT_ACCURACY
    return exp4135.STARTING_VAL_EXACT_ACCURACY


def _entry(
    *,
    label: str,
    pass_index: int,
    experiment: str,
    val_exact_accuracy: Any,
    delta_vs_previous: Any,
    status: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "pass_index": pass_index,
        "experiment": experiment,
        "val_exact_accuracy": _rounded(_float_or_none(val_exact_accuracy)),
        "delta_vs_previous": _rounded(_float_or_none(delta_vs_previous)),
        "status": status,
    }


def _nested_mapping(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _status_for_artifact(artifact: Mapping[str, Any]) -> str:
    if artifact.get("baseline_status") == "config-blocked" or artifact.get("plateau_audit_done") is True:
        return "config-blocked"
    if _float_or_none(artifact.get("val_exact_accuracy")) is not None:
        return "measured"
    return "missing"


def build_val_trajectory_383(
    *,
    pass3_artifact: Mapping[str, Any],
    exp4127_artifact: Mapping[str, Any],
    pass4_val: float | None,
    pass4_delta: float | None,
    pass4_status: str,
) -> list[dict[str, Any]]:
    """REQ-LEARN-4138: report the .382 anchor and .383 passes through pass4."""

    pass2_artifact = _nested_mapping(pass3_artifact, "pass2_artifact")
    pass1_artifact = _nested_mapping(pass2_artifact, "pass1_artifact")
    return [
        _entry(
            label=".382_anchor",
            pass_index=0,
            experiment="experiment_4127_sudoku_extreme_accumulate_fixed",
            val_exact_accuracy=_last_numeric_4127_val(exp4127_artifact),
            delta_vs_previous=None,
            status="measured",
        ),
        _entry(
            label=".383_pass1",
            pass_index=1,
            experiment="experiment_4135_sudoku_accumulate_pass1_fixed_lr",
            val_exact_accuracy=pass1_artifact.get("val_exact_accuracy"),
            delta_vs_previous=pass1_artifact.get("delta_vs_previous"),
            status=_status_for_artifact(pass1_artifact),
        ),
        _entry(
            label=".383_pass2",
            pass_index=2,
            experiment="experiment_4136_sudoku_accumulate_pass2_fixed_lr",
            val_exact_accuracy=pass2_artifact.get("val_exact_accuracy"),
            delta_vs_previous=pass2_artifact.get("delta_vs_previous"),
            status=_status_for_artifact(pass2_artifact),
        ),
        _entry(
            label=".383_pass3",
            pass_index=3,
            experiment="experiment_4137_sudoku_accumulate_pass3_fixed_lr",
            val_exact_accuracy=pass3_artifact.get("val_exact_accuracy"),
            delta_vs_previous=pass3_artifact.get("delta_vs_previous"),
            status=_status_for_artifact(pass3_artifact),
        ),
        _entry(
            label=".383_pass4",
            pass_index=4,
            experiment="experiment_4138_sudoku_accumulate_pass4_convergence_check",
            val_exact_accuracy=pass4_val,
            delta_vs_previous=pass4_delta,
            status=pass4_status,
        ),
    ]


def estimate_passes_to_converge(values: Sequence[float | None], target: float = PUBLISHED_EXACT_ACCURACY) -> int | None:
    """REQ-LEARN-4138: estimate remaining passes from observed positive validation deltas."""

    numeric = [_float_or_none(value) for value in values]
    observed = [value for value in numeric if value is not None]
    if not observed:
        return None
    current = observed[-1]
    if current >= target:
        return 0
    positive_deltas = [
        later - earlier
        for earlier, later in zip(observed, observed[1:], strict=False)
        if later > earlier
    ]
    if not positive_deltas:
        return None
    mean_delta = sum(positive_deltas) / len(positive_deltas)
    if mean_delta <= 0:
        return None
    return max(1, math.ceil((target - current) / mean_delta))


def _verdict_for_pass(val_exact_accuracy: float | None) -> str:
    if val_exact_accuracy is None:
        return "complete: missing_real_val_exact_accuracy"
    if matches_published_087(val_exact_accuracy):
        return f"complete: val={val_exact_accuracy:.4f} faithful, graft runs full"
    if near_faithful_080(val_exact_accuracy):
        return f"complete: val={val_exact_accuracy:.4f} near-faithful, RFT de-confound runs"
    return f"complete: val={val_exact_accuracy:.4f}, .384 finishes convergence"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    duration = _float_or_none(artifact.get("duration_s"))
    if artifact.get("baseline_status") == "config-blocked":
        return bool(duration is not None and duration < 600 and artifact.get("corrected_config_recommendation"))
    return (
        isinstance(artifact.get("val_trajectory_383"), list)
        and isinstance(artifact.get("matches_published_087"), bool)
        and isinstance(artifact.get("near_faithful_080"), bool)
        and _float_or_none(artifact.get("val_exact_accuracy")) is not None
    )


def build_config_blocked_artifact(
    *,
    run_config: Exp4138Config,
    pass3_artifact: Mapping[str, Any],
    exp4127_artifact: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4138-AUDIT: build a no-training pass4 config-blocked artifact."""

    recommendation = _recommendation_for_384(pass3_artifact)
    suspected_cause = str(pass3_artifact.get("suspected_cause") or "pass3 did not report positive accumulation")
    if "pass3 did not report positive accumulation" not in suspected_cause:
        suspected_cause = f"pass3 did not report positive accumulation: {suspected_cause}"
    artifact = {
        "experiment": "experiment_4138_sudoku_accumulate_pass4_convergence_check",
        "schema": "carnot.experiment_4138_sudoku_accumulate_pass4_convergence_check.v1",
        "spec_refs": ["REQ-LEARN-4138", "SCENARIO-LEARN-4138-AUDIT"],
        "honest_verdict": f"complete: baseline config-blocked before pass4 -> {suspected_cause}",
        "val_exact_accuracy": None,
        "val_trajectory_383": build_val_trajectory_383(
            pass3_artifact=pass3_artifact,
            exp4127_artifact=exp4127_artifact,
            pass4_val=None,
            pass4_delta=None,
            pass4_status="config-blocked",
        ),
        "matches_published_087": False,
        "near_faithful_080": False,
        "estimated_passes_to_converge": None,
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
        "baseline_status": "config-blocked",
        "suspected_cause": suspected_cause,
        "corrected_config_recommendation": recommendation,
        "stable_checkpoint_from_pass3": pass3_artifact.get("stable_checkpoint_path"),
        "pass3_artifact_path": str(run_config.pass3_artifact_path),
        "pass3_artifact": dict(pass3_artifact),
        "exp4127_artifact_path": str(run_config.exp4127_artifact_path),
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
    run_config: Exp4138Config,
    pass3_artifact: Mapping[str, Any],
    exp4127_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    lr_fix_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-4138: build a no-training runtime-blocked artifact after pass3 passed."""

    artifact = {
        "experiment": "experiment_4138_sudoku_accumulate_pass4_convergence_check",
        "schema": "carnot.experiment_4138_sudoku_accumulate_pass4_convergence_check.v1",
        "spec_refs": ["REQ-LEARN-4138"],
        "honest_verdict": reason,
        "val_exact_accuracy": None,
        "val_trajectory_383": build_val_trajectory_383(
            pass3_artifact=pass3_artifact,
            exp4127_artifact=exp4127_artifact,
            pass4_val=None,
            pass4_delta=None,
            pass4_status="runtime-blocked",
        ),
        "matches_published_087": False,
        "near_faithful_080": False,
        "estimated_passes_to_converge": None,
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
        "pass3_artifact_path": str(run_config.pass3_artifact_path),
        "pass3_artifact": dict(pass3_artifact),
        "exp4127_artifact_path": str(run_config.exp4127_artifact_path),
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
    run_config: Exp4138Config,
    run_result: exp4116.ResumeRunResult,
    pass_metrics: exp4135.PassMetricSummary,
    pass3_artifact: Mapping[str, Any],
    exp4127_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[exp4107.PreconditionCheck | Mapping[str, Any]],
    lr_fix_artifact: Mapping[str, Any],
    dataset_generated: bool,
    checkpoint_timer_reset: exp4135.TimerResetResult | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4138: build the measured pass4 artifact."""

    val = pass_metrics.val_exact_accuracy
    previous = _float_or_none(pass3_artifact.get("val_exact_accuracy"))
    delta = None if val is None or previous is None else val - previous
    rounded_delta = _rounded(delta)
    trajectory = build_val_trajectory_383(
        pass3_artifact=pass3_artifact,
        exp4127_artifact=exp4127_artifact,
        pass4_val=val,
        pass4_delta=rounded_delta,
        pass4_status="measured" if val is not None else "missing",
    )
    trajectory_values = [entry["val_exact_accuracy"] for entry in trajectory]
    estimate = estimate_passes_to_converge(trajectory_values) if val is not None and val < NEAR_FAITHFUL_THRESHOLD else None
    artifact = {
        "experiment": "experiment_4138_sudoku_accumulate_pass4_convergence_check",
        "schema": "carnot.experiment_4138_sudoku_accumulate_pass4_convergence_check.v1",
        "spec_refs": ["REQ-LEARN-4138", "SCENARIO-LEARN-4138"],
        "honest_verdict": _verdict_for_pass(val),
        "val_exact_accuracy": _rounded(val),
        "val_trajectory_383": trajectory,
        "matches_published_087": matches_published_087(val),
        "near_faithful_080": near_faithful_080(val),
        "estimated_passes_to_converge": estimate,
        "stable_checkpoint_path": str(run_config.stable_checkpoint_path),
        "random_seed": int(run_config.random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(run_config),
        "duration_s": round(float(run_result.duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
        "delta_vs_previous": rounded_delta,
        "lr_continued_not_rewarmed": bool(pass_metrics.lr_continued_not_rewarmed),
        "pass3_val_exact_accuracy": _rounded(previous),
        "pass3_artifact_path": str(run_config.pass3_artifact_path),
        "pass3_artifact": dict(pass3_artifact),
        "exp4127_artifact_path": str(run_config.exp4127_artifact_path),
        "model_specs": model_specs(run_config),
        "pass_index": PASS_INDEX,
        "return_code": int(run_result.return_code),
        "checkpoint_reload_ok": bool(run_result.checkpoint_reload_ok),
        "checkpoint_reload_detail": run_result.checkpoint_reload_detail,
        "exact_accuracy_metric": "val/exact_accuracy" if val is not None else None,
        "exact_accuracy_metrics_path": None if pass_metrics.val_metrics_path is None else str(pass_metrics.val_metrics_path),
        "validation_first_lr": pass_metrics.first_train_lr,
        "fresh_warmup_lr": exp4126.FRESH_WARMUP_FIRST_LR,
        "train_lr_point_count": int(pass_metrics.train_lr_point_count),
        "first_train_lr_metrics_path": (
            None if pass_metrics.first_train_lr_metrics_path is None else str(pass_metrics.first_train_lr_metrics_path)
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
    """Return explicit schema errors for the Exp 4138 deliverable."""

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
    if not isinstance(artifact.get("val_trajectory_383"), list):
        errors.append("val_trajectory_383 must be a list")
    if not isinstance(artifact.get("matches_published_087"), bool):
        errors.append("matches_published_087 must be a bare bool")
    if not isinstance(artifact.get("near_faithful_080"), bool):
        errors.append("near_faithful_080 must be a bare bool")
    estimate = artifact.get("estimated_passes_to_converge")
    if estimate is not None and (not isinstance(estimate, int) or isinstance(estimate, bool) or estimate < 0):
        errors.append("estimated_passes_to_converge must be a non-negative int or null")
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
    if artifact.get("baseline_status") == "config-blocked" and (duration is None or duration >= 600):
        errors.append("config-blocked duration_s must be below 600")
    if artifact.get("matches_published_087") is True and not matches_published_087(_float_or_none(val)):
        errors.append("matches_published_087=true requires val within 0.02 of 0.87")
    if artifact.get("near_faithful_080") is True and not near_faithful_080(_float_or_none(val)):
        errors.append("near_faithful_080=true requires val >= 0.80")
    gate = artifact.get("acceptance_gate_passed")
    if gate is not None and not isinstance(gate, bool):
        errors.append("acceptance_gate_passed must be a bare bool")
    if gate is True:
        if artifact.get("baseline_status") == "config-blocked":
            if not artifact.get("corrected_config_recommendation"):
                errors.append("config-blocked acceptance requires corrected_config_recommendation")
        elif not isinstance(artifact.get("val_trajectory_383"), list):
            errors.append("training acceptance requires val_trajectory_383")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_result_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    _write_json(Path(path), artifact)


def check_preconditions(
    config: Exp4138Config,
    *,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> tuple[list[exp4107.PreconditionCheck], str | None]:
    """SCENARIO-LEARN-4138: check all no-fabrication runtime prerequisites."""

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
    config: Exp4138Config,
    *,
    duration_s: float,
    return_code: int = 0,
    command: Sequence[str] | None = None,
    stdout_tail: Sequence[str] | None = None,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:
    """SCENARIO-LEARN-4138: verify the stable checkpoint and pass4 metrics."""

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
    config: Exp4138Config,
    *,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> exp4116.ResumeRunResult:  # pragma: no cover - launches native trainer.
    """Run one native nano-trm fixed-LR resume pass with a one-hour cap."""

    print(
        f"[exp4138] launching pass4 fixed-LR resume stable={config.stable_checkpoint_path}",
        flush=True,
    )
    return exp4127.run_native_resume_pass(
        config.to_4127_config(),
        PASS_INDEX,
        checkpoint_loader=checkpoint_loader,
    )


def generate_sudoku_extreme_dataset_if_missing(config: Exp4138Config) -> bool:  # pragma: no cover
    """Generate the nano-trm Sudoku Extreme dataset only when it is absent."""

    return exp4116.generate_sudoku_extreme_dataset_if_missing(config.to_4116_config())


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    save_parent: str | Path = DEFAULT_SAVE_PARENT,
    stable_dir: str | Path | None = None,
    hydra_run_root: str | Path | None = None,
    pass3_artifact_path: str | Path | None = None,
    exp4127_artifact_path: str | Path | None = None,
    lr_artifact_path: str | Path | None = None,
    uv_resolver: Callable[[str], str | None] = exp4116.shutil.which,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    dataset_builder: Callable[[Exp4138Config], object] = generate_sudoku_extreme_dataset_if_missing,
    timer_reset: Callable[[Path], exp4135.TimerResetResult | Mapping[str, Any]] = exp4135.reset_checkpoint_timer_state,
    trainer_runner: Callable[[Exp4138Config], exp4116.ResumeRunResult] | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run Exp 4138 or write the required pass3-stall artifact."""

    started = time.time()
    root = Path(repo_root)
    pass3_path = Path(pass3_artifact_path) if pass3_artifact_path is not None else root / "results" / exp4137.RESULT_FILENAME
    exp4127_path = (
        Path(exp4127_artifact_path) if exp4127_artifact_path is not None else root / "results" / exp4127.RESULT_FILENAME
    )
    lr_path = Path(lr_artifact_path) if lr_artifact_path is not None else root / "results" / exp4126.RESULT_FILENAME
    config = Exp4138Config(
        repo_root=root,
        save_parent=save_parent,
        stable_dir=stable_dir,
        hydra_run_root=hydra_run_root,
        pass3_artifact_path=pass3_path,
        exp4127_artifact_path=exp4127_path,
        lr_artifact_path=lr_path,
        random_seed=random_seed,
    )
    out = Path(output_path)
    pass3_artifact = load_pass3_artifact(config.pass3_artifact_path)
    exp4127_artifact = load_exp4127_artifact(config.exp4127_artifact_path)
    if not pass3_allows_training(pass3_artifact):
        print("[exp4138] pass3 stalled or audited; writing config-blocked artifact and skipping training", flush=True)
        artifact = build_config_blocked_artifact(
            run_config=config,
            pass3_artifact=pass3_artifact,
            exp4127_artifact=exp4127_artifact,
            duration_s=time.time() - started,
        )
        _write_json(out, artifact)
        return artifact
    lr_fix_artifact = exp4127.load_lr_fix_artifact(lr_path)
    if not exp4137.lr_fix_landed(lr_fix_artifact):
        artifact = build_blocked_artifact(
            "blocked_lr_fix_not_landed",
            run_config=config,
            pass3_artifact=pass3_artifact,
            exp4127_artifact=exp4127_artifact,
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
            pass3_artifact=pass3_artifact,
            exp4127_artifact=exp4127_artifact,
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
            pass3_artifact=pass3_artifact,
            exp4127_artifact=exp4127_artifact,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            lr_fix_artifact=lr_fix_artifact,
        )
        artifact["dataset_dir"] = str(config.dataset_dir)
        artifact["dataset_generated"] = dataset_generated
        validate_artifact(artifact)
        _write_json(out, artifact)
        return artifact
    checkpoint_timer_reset = timer_reset(config.stable_checkpoint_path)
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
        pass3_artifact=pass3_artifact,
        exp4127_artifact=exp4127_artifact,
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
