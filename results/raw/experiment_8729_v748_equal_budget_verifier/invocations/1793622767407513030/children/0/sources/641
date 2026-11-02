"""Exp 4159 decisive verifier-as-reward graft for Sudoku.

This experiment is defensive by design. It reads the contiguous Exp 4157
baseline and refuses to launch reward-training work while the TRM validation
accuracy is below the faithful threshold. If the baseline is faithful, it first
checks that verifier-certified labels are actually test-gold before attributing
any A-vs-B training delta to the verifier reward.

Spec refs: REQ-LEARN-4159, SCENARIO-LEARN-4159-DEFER,
SCENARIO-LEARN-4159-PHASE0, SCENARIO-LEARN-4159-RFT.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109


CandidatePool = exp4109.CandidatePool

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4159_decisive_verifier_reward_graft.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_EXP4157_ARTIFACT = REPO_ROOT / "results" / "experiment_4157_baseline_harvest_contiguous_continue.json"
DEFAULT_DATA_DIR = REPO_ROOT / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
DEFAULT_HELDOUT_SPLIT = "_valsmall"
RANDOM_SEED = 4159
DEFAULT_MAX_PUZZLES = 64
DEFAULT_K_CANDIDATES = 8
FAITHFUL_VAL_THRESHOLD = 0.85
PHASE0_PRECISION_THRESHOLD = 0.85
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
SPEC_REFS = [
    "REQ-LEARN-4159",
    "SCENARIO-LEARN-4159-DEFER",
    "SCENARIO-LEARN-4159-PHASE0",
    "SCENARIO-LEARN-4159-RFT",
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "graft_deferred",
    "phase0_precision",
    "rft_vs_ablation_delta",
    "verifier_value_added",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest deferral, an A>B win, or an A~=B null are ALL COMPLETE; "
        "an uninformative graft on a non-faithful baseline is the .383 anti-pattern."
    ),
    "graft_deferred": (
        "Bare bool: True if val<0.85 -> deferred. Prevents the uninformative no-headroom "
        "false-negative .383/.384 produced."
    ),
    "phase0_precision": (
        "P(test-gold | demo-perfect) on the verifier-certified corpus; the precondition that "
        "a verifier label means correctness, not just demo-fit (per the verifier-as-reward discipline)."
    ),
    "rft_vs_ablation_delta": (
        "The de-confounded A-vs-B held-out delta with CI95 (if grafted): isolates the verifier "
        "LABEL's training contribution -- THE moat-at-training measurement."
    ),
    "verifier_value_added": (
        "Bare bool: did verifier-cert RFT beat vote-cert RFT (CI95 excl 0)? The headline answer, "
        "ONLY meaningful when graft_deferred is False. Also the DiffusionGemma gate input."
    ),
    "preconditions_checked": "Records the baseline checkpoint + CUDA verified.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One runtime resource check required before Exp 4159 can measure a graft."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineContext:
    """Exp 4157 baseline evidence that decides whether Exp 4159 may graft."""

    artifact_path: Path
    stable_checkpoint_path: Path
    current_val: float | None
    max_val: float | None
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "stable_checkpoint_path": str(self.stable_checkpoint_path),
            "current_val": self.current_val,
            "max_val": self.max_val,
        }


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _checks_to_dicts(checks: Sequence[PreconditionCheck | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [check.to_dict() if isinstance(check, PreconditionCheck) else dict(check) for check in checks]


def _all_preconditions_available(checks: Sequence[PreconditionCheck | Mapping[str, Any]]) -> bool:
    return all(bool(check.available) if isinstance(check, PreconditionCheck) else check.get("available") is True for check in checks)


def _metric_has_ci(metric: Mapping[str, Any]) -> bool:
    ci95 = metric.get("ci95")
    return (
        isinstance(ci95, Sequence)
        and not isinstance(ci95, (str, bytes))
        and len(ci95) == 2
        and _float_or_none(ci95[0]) is not None
        and _float_or_none(ci95[1]) is not None
    )


def _is_json_number(value: Any) -> bool:
    return _float_or_none(value) is not None


def load_baseline_context(path: str | Path) -> BaselineContext:
    """REQ-LEARN-4159: read Exp 4157 without inventing a faithful baseline."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Exp 4157 artifact must be a JSON object")
    stable_value = payload.get("stable_checkpoint_path")
    stable = Path(stable_value) if isinstance(stable_value, str) and stable_value else Path("")
    return BaselineContext(
        artifact_path=artifact_path,
        stable_checkpoint_path=stable,
        current_val=_float_or_none(payload.get("current_val")),
        max_val=_float_or_none(payload.get("max_val")),
        raw=dict(payload),
    )


def baseline_is_faithful(baseline: BaselineContext) -> bool:
    """REQ-LEARN-4159: reward grafts require validation accuracy at least 0.85."""

    return baseline.current_val is not None and baseline.current_val >= FAITHFUL_VAL_THRESHOLD


def estimate_passes_to_converge_for_386(baseline: BaselineContext | None) -> dict[str, Any]:
    """SCENARIO-LEARN-4159-DEFER: estimate `.386` intervals from observed validation movement."""

    current = None if baseline is None else baseline.current_val
    base = {
        "destination": ".386",
        "target_val_exact_accuracy": FAITHFUL_VAL_THRESHOLD,
        "current_val_exact_accuracy": current,
    }
    if current is None:
        return {**base, "estimated_additional_val_intervals": None, "basis": "missing_exp4157_current_val"}
    if current >= FAITHFUL_VAL_THRESHOLD:
        return {**base, "estimated_additional_val_intervals": 0, "basis": "already_faithful"}

    upstream = baseline.raw.get("estimated_passes_to_085") if baseline is not None else None
    if isinstance(upstream, Mapping):
        estimate = upstream.get("estimated_additional_val_intervals")
        if isinstance(estimate, int) and not isinstance(estimate, bool):
            return {**base, "estimated_additional_val_intervals": estimate, "basis": str(upstream.get("basis", "exp4157_estimate"))}

    vals: list[float] = []
    trajectory = [] if baseline is None else baseline.raw.get("val_trajectory", [])
    if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
        for row in trajectory:
            if isinstance(row, Mapping):
                val = _float_or_none(row.get("val_exact_accuracy"))
                if val is not None:
                    vals.append(val)
    positive_deltas = [
        vals[index] - vals[index - 1]
        for index in range(1, len(vals))
        if vals[index] > vals[index - 1]
    ]
    if not positive_deltas:
        return {**base, "estimated_additional_val_intervals": None, "basis": "no_positive_exp4157_convergence_rate"}
    rate = sum(positive_deltas) / len(positive_deltas)
    needed = max(FAITHFUL_VAL_THRESHOLD - current, 0.0)
    return {
        **base,
        "estimated_additional_val_intervals": int(math.ceil(needed / rate)),
        "basis": "mean_positive_exp4157_delta",
        "mean_positive_delta_per_interval": round(float(rate), 12),
    }


def check_preconditions(
    baseline: BaselineContext,
    *,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4109._default_cuda_checker,
) -> list[PreconditionCheck]:
    """REQ-LEARN-4159: record Exp 4157, checkpoint, and CUDA checks."""

    checks = [
        PreconditionCheck("exp4157_artifact", baseline.artifact_path.is_file(), str(baseline.artifact_path)),
        PreconditionCheck(
            "stable_checkpoint_path",
            bool(str(baseline.stable_checkpoint_path)),
            str(baseline.stable_checkpoint_path),
        ),
        PreconditionCheck(
            "stable_checkpoint",
            baseline.stable_checkpoint_path.is_file(),
            str(baseline.stable_checkpoint_path),
        ),
    ]
    try:
        cuda_ok, cuda_detail = cuda_checker()
    except Exception as exc:
        cuda_ok, cuda_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("cuda_available", bool(cuda_ok), str(cuda_detail)))
    return checks


def snapshot_checkpoint(source_path: str | Path, snapshot_path: str | Path | None = None) -> Path:
    """REQ-LEARN-4159: freeze the checkpoint before any faithful-branch model load."""

    source = Path(source_path)
    snapshot = Path(snapshot_path) if snapshot_path is not None else source.with_name(f"{source.stem}-reward-snapshot{source.suffix}")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot)
    return snapshot


def deferred_phase0_precision(status: str, baseline: BaselineContext | None) -> dict[str, Any]:
    """SCENARIO-LEARN-4159-DEFER: keep phase-0 present without fake precision."""

    return {
        "metric": "P(test-gold|demo-perfect)",
        "numerator_test_gold": 0,
        "denominator_demo_perfect": 0,
        "precision": 0.0,
        "threshold": PHASE0_PRECISION_THRESHOLD,
        "passes": False,
        "status": status,
        "current_val_exact_accuracy": None if baseline is None else baseline.current_val,
    }


def deferred_rft_delta(status: str, baseline: BaselineContext | None) -> dict[str, Any]:
    """SCENARIO-LEARN-4159-DEFER: mark the reward-training contrast as not run."""

    return {
        "metric": "heldout_exact_accuracy",
        "n_matched": 0,
        "a_exact_accuracy": 0.0,
        "b_exact_accuracy": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
        "current_val_exact_accuracy": None if baseline is None else baseline.current_val,
    }


def estimate_phase0_precision(corpora: Mapping[str, Any]) -> dict[str, Any]:
    """SCENARIO-LEARN-4159-PHASE0: estimate P(test-gold | demo-perfect)."""

    rows = [row for row in corpora.get("rows", []) if isinstance(row, Mapping)]
    denominator = len(rows)
    numerator = sum(bool(row.get("a_exact")) for row in rows)
    precision = (numerator / denominator) if denominator else 0.0
    if denominator == 0:
        status = "no_verifier_certified_labels"
    elif precision >= PHASE0_PRECISION_THRESHOLD:
        status = "precision_gate_passed"
    else:
        status = "precision_gate_failed_label_noise"
    return {
        "metric": "P(test-gold|demo-perfect)",
        "numerator_test_gold": int(numerator),
        "denominator_demo_perfect": int(denominator),
        "precision": round(float(precision), 6),
        "threshold": PHASE0_PRECISION_THRESHOLD,
        "passes": bool(denominator > 0 and precision >= PHASE0_PRECISION_THRESHOLD),
        "status": status,
    }


def phase0_precision_passed(phase0_precision: Mapping[str, Any]) -> bool:
    precision = _float_or_none(phase0_precision.get("precision"))
    denominator = phase0_precision.get("denominator_demo_perfect")
    return (
        precision is not None
        and precision >= PHASE0_PRECISION_THRESHOLD
        and isinstance(denominator, int)
        and not isinstance(denominator, bool)
        and denominator > 0
    )


def verifier_value_added(rft_vs_ablation_delta: Mapping[str, Any], *, graft_deferred: bool) -> bool:
    """REQ-LEARN-4159: headline bool comes only from the RFT A-vs-B delta."""

    if graft_deferred:
        return False
    ci95 = rft_vs_ablation_delta.get("ci95")
    try:
        return (
            isinstance(ci95, Sequence)
            and not isinstance(ci95, (str, bytes))
            and len(ci95) == 2
            and float(rft_vs_ablation_delta.get("delta", 0.0)) > 0.0
            and float(ci95[0]) > 0.0
        )
    except (TypeError, ValueError):
        return False


def _artifact_verdict(*, graft_deferred: bool, value_added: bool, defer_reason: str | None) -> str:
    if graft_deferred and defer_reason == "baseline_below_0.85":
        return "complete: graft_deferred_baseline_below_0.85"
    if graft_deferred and defer_reason == "phase0_precision_below_0.85":
        return "complete: graft_deferred_phase0_precision_below_0.85"
    if value_added:
        return "success: verifier_value_added_rft_A_gt_B"
    return "complete: A~=B null"


def _baseline_status(baseline: BaselineContext | None) -> dict[str, Any] | None:
    if baseline is None:
        return None
    return {
        "artifact_path": str(baseline.artifact_path),
        "stable_checkpoint_path": str(baseline.stable_checkpoint_path),
        "current_val": baseline.current_val,
        "max_val": baseline.max_val,
        "faithful_threshold": FAITHFUL_VAL_THRESHOLD,
        "faithful": baseline_is_faithful(baseline),
    }


def compute_reproducibility_checksum(
    *,
    baseline: BaselineContext | None,
    heldout_ids: Sequence[str],
    corpora: Mapping[str, Any] | None = None,
    phase0_precision: Mapping[str, Any] | None = None,
) -> str:
    """REQ-LEARN-4159: hash baseline identity plus reward-label evidence."""

    payload = {
        "schema": "carnot.experiment_4159.reward_graft.v1",
        "baseline": None if baseline is None else baseline.to_dict(),
        "heldout_ids": list(heldout_ids),
        "corpora": _jsonable(corpora or {}),
        "phase0_precision": _jsonable(phase0_precision or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or verdict.startswith("blocked_"):
        return False
    if type(artifact.get("graft_deferred")) is not bool:
        return False
    if type(artifact.get("verifier_value_added")) is not bool:
        return False
    baseline_status = artifact.get("baseline_status")
    if not isinstance(baseline_status, Mapping):
        return False

    phase0 = artifact.get("phase0_precision")
    rft = artifact.get("rft_vs_ablation_delta")
    if artifact["graft_deferred"] is True:
        val = _float_or_none(baseline_status.get("current_val"))
        if val is not None and val < FAITHFUL_VAL_THRESHOLD:
            return True
        return isinstance(phase0, Mapping) and phase0.get("status") == "precision_gate_failed_label_noise"
    return isinstance(phase0, Mapping) and phase0_precision_passed(phase0) and isinstance(rft, Mapping) and _metric_has_ci(rft)


def build_result_artifact(
    *,
    baseline: BaselineContext,
    phase0_precision: Mapping[str, Any],
    rft_vs_ablation_delta: Mapping[str, Any],
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    candidate_source: str,
    n_candidate_pools: int,
    corpus_summary: Mapping[str, Any] | None = None,
    snapshot_path: str | Path | None = None,
    random_seed: int = RANDOM_SEED,
    reproducibility_checksum: str | None = None,
    defer_reason: str | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-4159: build an honest deferral, A>B win, or A~=B null artifact."""

    graft_deferred = defer_reason is not None
    value_added = verifier_value_added(rft_vs_ablation_delta, graft_deferred=graft_deferred)
    rft_mode = str(rft_vs_ablation_delta.get("training_mode", "not_run" if graft_deferred else "bounded_contiguous_resume_same_init"))
    artifact: dict[str, Any] = {
        "experiment": "experiment_4159_decisive_verifier_reward_graft",
        "schema": "carnot.experiment_4159_decisive_verifier_reward_graft.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(
            graft_deferred=graft_deferred,
            value_added=value_added,
            defer_reason=defer_reason,
        ),
        "graft_deferred": graft_deferred,
        "phase0_precision": _jsonable(phase0_precision),
        "rft_vs_ablation_delta": _jsonable(rft_vs_ablation_delta),
        "verifier_value_added": bool(value_added),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "baseline_status": _baseline_status(baseline),
        "current_val": baseline.current_val,
        "estimated_passes_to_converge_for_386": estimate_passes_to_converge_for_386(baseline),
        "stable_checkpoint_path": str(baseline.stable_checkpoint_path),
        "snapshot_checkpoint_path": None if snapshot_path is None else str(snapshot_path),
        "candidate_source": candidate_source,
        "n_candidate_pools": int(n_candidate_pools),
        "corpus_summary": _jsonable(corpus_summary or {}),
        "rft_training_mode": rft_mode,
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum
        or compute_reproducibility_checksum(
            baseline=baseline,
            heldout_ids=[],
            phase0_precision=phase0_precision,
        ),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    honest_verdict: str,
    *,
    baseline: BaselineContext | None,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    detail: str | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-4159: fail closed when required resources or native RFT are missing."""

    artifact: dict[str, Any] = {
        "experiment": "experiment_4159_decisive_verifier_reward_graft",
        "schema": "carnot.experiment_4159_decisive_verifier_reward_graft.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "graft_deferred": True,
        "phase0_precision": deferred_phase0_precision(honest_verdict, baseline),
        "rft_vs_ablation_delta": deferred_rft_delta(honest_verdict, baseline),
        "verifier_value_added": False,
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "baseline_status": _baseline_status(baseline),
        "current_val": None if baseline is None else baseline.current_val,
        "estimated_passes_to_converge_for_386": estimate_passes_to_converge_for_386(baseline),
        "stable_checkpoint_path": None if baseline is None else str(baseline.stable_checkpoint_path),
        "snapshot_checkpoint_path": None,
        "candidate_source": "none_blocked",
        "n_candidate_pools": 0,
        "corpus_summary": {},
        "rft_training_mode": "not_run",
        "random_seed": int(random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(baseline=baseline, heldout_ids=[]),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    if detail is not None:
        artifact["blocked_detail"] = detail
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4159 deliverable."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    if type(artifact.get("graft_deferred")) is not bool:
        errors.append("graft_deferred must be a bare bool")
    if type(artifact.get("verifier_value_added")) is not bool:
        errors.append("verifier_value_added must be a bare bool")
    if artifact.get("graft_deferred") is True and artifact.get("verifier_value_added") is True:
        errors.append("verifier_value_added is only meaningful when graft_deferred is false")

    phase0 = artifact.get("phase0_precision")
    if not isinstance(phase0, Mapping):
        errors.append("phase0_precision must be an object")
    else:
        if "precision" not in phase0:
            errors.append("phase0_precision.precision is required")
        elif not _is_json_number(phase0.get("precision")):
            errors.append("phase0_precision.precision must be numeric")

    rft = artifact.get("rft_vs_ablation_delta")
    if not isinstance(rft, Mapping):
        errors.append("rft_vs_ablation_delta must be an object")
    else:
        if "delta" not in rft:
            errors.append("rft_vs_ablation_delta.delta is required")
        if not _metric_has_ci(rft):
            errors.append("rft_vs_ablation_delta.ci95 must have two numeric bounds")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list):
        errors.append("preconditions_checked must be a list")
    elif any(
        not isinstance(item, Mapping) or "resource" not in item or "available" not in item
        for item in preconditions
    ):
        errors.append("preconditions_checked entries must include resource and available")

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field_name, principle in FIELD_PRINCIPLES.items():
            if principles.get(field_name) != principle:
                errors.append(f"field_principles.{field_name} mismatch")

    baseline_status = artifact.get("baseline_status")
    if baseline_status is not None and not isinstance(baseline_status, Mapping):
        errors.append("baseline_status must be an object or null")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("duration_s must be numeric")
    if "acceptance_gate_passed" in artifact and type(artifact.get("acceptance_gate_passed")) is not bool:
        errors.append("acceptance_gate_passed must be a bare bool")
    checksum = artifact.get("reproducibility_checksum")
    if checksum is not None and not (
        isinstance(checksum, str) and checksum.startswith("sha256:") and len(checksum) == 71
    ):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    if "random_seed" in artifact and type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Write the stable Exp 4159 JSON artifact."""

    validate_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(artifact)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def sample_checkpoint_candidate_pools(  # pragma: no cover - live CUDA/checkpoint path.
    *,
    snapshot_path: str | Path,
    repo_root: str | Path = REPO_ROOT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    random_seed: int = RANDOM_SEED,
) -> list[CandidatePool]:
    """REQ-LEARN-4159: sample held-out candidates from the frozen TRM checkpoint."""

    return exp4109.sample_checkpoint_candidate_pools(
        checkpoint_path=snapshot_path,
        repo_root=repo_root,
        data_dir=data_dir,
        split=split,
        max_puzzles=max_puzzles,
        k_candidates=k_candidates,
        random_seed=random_seed,
    )


def _summarize_corpora(corpora: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in corpora.get("rows", []) if isinstance(row, Mapping)]
    return {
        "arm_a": corpora.get("arm_a"),
        "arm_b": corpora.get("arm_b"),
        "n_matched": int(corpora.get("n_matched", 0)),
        "skipped_no_verifier_valid": len(corpora.get("skipped_no_verifier_valid", [])),
        "a_exact_count": sum(bool(row.get("a_exact")) for row in rows),
        "b_exact_count": sum(bool(row.get("b_exact")) for row in rows),
    }


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    exp4157_artifact_path: str | Path = DEFAULT_EXP4157_ARTIFACT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    heldout_split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4109._default_cuda_checker,
    snapshotter: Callable[[Path], Path] = snapshot_checkpoint,
    candidate_pool_provider: Callable[[Path], Sequence[CandidatePool]] | None = None,
    rft_runner: Callable[[BaselineContext, Path, dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4159 and write the decisive reward-graft artifact."""

    started = time.time()
    root = Path(repo_root)
    exp4157_path = Path(exp4157_artifact_path)
    if not exp4157_path.is_file():
        artifact = build_blocked_artifact(
            "blocked_exp4157_artifact",
            baseline=None,
            preconditions_checked=[PreconditionCheck("exp4157_artifact", False, str(exp4157_path))],
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    try:
        baseline = load_baseline_context(exp4157_path)
    except Exception as exc:  # pragma: no cover - covered through direct loader tests.
        artifact = build_blocked_artifact(
            "blocked_exp4157_artifact",
            baseline=None,
            preconditions_checked=[PreconditionCheck("exp4157_artifact", False, str(exp4157_path))],
            duration_s=time.time() - started,
            random_seed=random_seed,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return write_artifact(output_path, artifact)

    checks = check_preconditions(baseline, cuda_checker=cuda_checker)
    if not _all_preconditions_available(checks):
        first_missing = next(check for check in checks if not check.available)
        artifact = build_blocked_artifact(
            f"blocked_{first_missing.resource}",
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
            detail=first_missing.detail,
        )
        return write_artifact(output_path, artifact)

    if not baseline_is_faithful(baseline):
        artifact = build_result_artifact(
            baseline=baseline,
            phase0_precision=deferred_phase0_precision("deferred_baseline_below_0.85", baseline),
            rft_vs_ablation_delta=deferred_rft_delta("deferred_baseline_below_0.85", baseline),
            preconditions_checked=checks,
            duration_s=time.time() - started,
            candidate_source="none_baseline_below_0.85",
            n_candidate_pools=0,
            random_seed=random_seed,
            defer_reason="baseline_below_0.85",
        )
        return write_artifact(output_path, artifact)

    try:
        snapshot_path = snapshotter(baseline.stable_checkpoint_path)
    except Exception as exc:
        artifact = build_blocked_artifact(
            "blocked_checkpoint_snapshot",
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return write_artifact(output_path, artifact)

    try:
        if candidate_pool_provider is not None:
            pools = list(candidate_pool_provider(snapshot_path))
            candidate_source = "provided_candidate_pool"
        else:  # pragma: no cover - live CUDA/checkpoint sampling path.
            pools = sample_checkpoint_candidate_pools(
                snapshot_path=snapshot_path,
                repo_root=root,
                data_dir=data_dir,
                split=heldout_split,
                max_puzzles=max_puzzles,
                k_candidates=k_candidates,
                random_seed=random_seed,
            )
            candidate_source = "snapshot_checkpoint_final_logits_k_sampling"
    except Exception as exc:
        artifact = build_blocked_artifact(
            "blocked_candidate_sampling_failed",
            baseline=baseline,
            preconditions_checked=[
                *checks,
                PreconditionCheck("candidate_sampling", False, f"{type(exc).__name__}: {exc}"),
            ],
            duration_s=time.time() - started,
            random_seed=random_seed,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return write_artifact(output_path, artifact)

    corpora = exp4109.build_matched_corpora(pools)
    phase0 = estimate_phase0_precision(corpora)
    heldout_ids = [pool.puzzle_id for pool in pools]
    checksum = compute_reproducibility_checksum(
        baseline=baseline,
        heldout_ids=heldout_ids,
        corpora=corpora,
        phase0_precision=phase0,
    )
    if not phase0_precision_passed(phase0):
        artifact = build_result_artifact(
            baseline=baseline,
            phase0_precision=phase0,
            rft_vs_ablation_delta=deferred_rft_delta("deferred_phase0_precision_below_0.85", baseline),
            preconditions_checked=checks,
            duration_s=time.time() - started,
            candidate_source=candidate_source,
            n_candidate_pools=len(pools),
            corpus_summary=_summarize_corpora(corpora),
            snapshot_path=snapshot_path,
            random_seed=random_seed,
            reproducibility_checksum=checksum,
            defer_reason="phase0_precision_below_0.85",
        )
        return write_artifact(output_path, artifact)

    if rft_runner is None:
        artifact = build_blocked_artifact(
            "blocked_native_rft_runner_missing",
            baseline=baseline,
            preconditions_checked=[
                *checks,
                PreconditionCheck("native_rft_runner", False, "bounded contiguous-style RFT runner was not provided"),
            ],
            duration_s=time.time() - started,
            random_seed=random_seed,
            detail="bounded contiguous-style RFT runner was not provided",
        )
        return write_artifact(output_path, artifact)

    try:
        rft_delta = dict(rft_runner(baseline, snapshot_path, corpora))
    except Exception as exc:  # pragma: no cover - defensive native-training failure path.
        artifact = build_blocked_artifact(
            "blocked_rft_training_failed",
            baseline=baseline,
            preconditions_checked=[
                *checks,
                PreconditionCheck("native_rft_training", False, f"{type(exc).__name__}: {exc}"),
            ],
            duration_s=time.time() - started,
            random_seed=random_seed,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return write_artifact(output_path, artifact)

    artifact = build_result_artifact(
        baseline=baseline,
        phase0_precision=phase0,
        rft_vs_ablation_delta=rft_delta,
        preconditions_checked=checks,
        duration_s=time.time() - started,
        candidate_source=candidate_source,
        n_candidate_pools=len(pools),
        corpus_summary=_summarize_corpora(corpora),
        snapshot_path=snapshot_path,
        random_seed=random_seed,
        reproducibility_checksum=checksum,
    )
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
