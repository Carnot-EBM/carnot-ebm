"""Exp 4150 decisive verifier-as-reward gate for Sudoku.

This module is deliberately conservative: it reads the final Exp 4149 baseline
first, and it refuses to run verifier graft measurements unless the TRM baseline
has enough validation accuracy to make the contrast meaningful. That matters
because a verifier cannot prove training value on a baseline that still has no
headroom for the verifier-labelled data to improve the model; running the graft
there would repeat the `.383` false-negative pattern.

Spec refs: REQ-LEARN-4150, SCENARIO-LEARN-4150-DEFER,
SCENARIO-LEARN-4150-GRAFT.
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
from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4149_sudoku_accumulate_pass4_convergence as exp4149


CandidatePool = exp4109.CandidatePool

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4150_decisive_verifier_graft_sudoku.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_EXP4149_ARTIFACT = REPO_ROOT / "results" / exp4149.RESULT_FILENAME
DEFAULT_DATA_DIR = REPO_ROOT / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
DEFAULT_HELDOUT_SPLIT = "_valsmall"
RANDOM_SEED = 4150
DEFAULT_MAX_PUZZLES = 64
DEFAULT_K_CANDIDATES = 8
FAITHFUL_VAL_THRESHOLD = 0.85
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
SPEC_REFS = [
    "REQ-LEARN-4150",
    "SCENARIO-LEARN-4150-DEFER",
    "SCENARIO-LEARN-4150-GRAFT",
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "graft_deferred",
    "rerank_lift_vs_vote",
    "rft_vs_ablation_delta",
    "verifier_value_added",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. An honest deferral, an A>B win, or an A~=B null are all COMPLETE; "
        "an uninformative graft on a non-faithful baseline is the .383 anti-pattern."
    ),
    "graft_deferred": (
        "Bare bool: True if val<0.85 -> deferred. Prevents the uninformative "
        "no-headroom false-negative .383 produced."
    ),
    "rerank_lift_vs_vote": (
        "pass@1 lift from verifier-reranking (if grafted); confirms the executable "
        "Sudoku verifier discriminates (contrast to .379 ARC-grid anti-discrimination)."
    ),
    "rft_vs_ablation_delta": (
        "The de-confounded A-vs-B held-out delta with CI (if grafted): isolates the "
        "verifier LABEL's training contribution -- THE moat measurement."
    ),
    "verifier_value_added": (
        "Bare bool: did the graft beat the vote ablation? The headline answer, ONLY "
        "meaningful when graft_deferred is False. Also resolves the DiffusionGemma gate."
    ),
    "preconditions_checked": "Records the baseline checkpoint + CUDA verified.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One runtime resource check required before Exp 4150 can measure a graft."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineContext:
    """Exp 4149 baseline evidence that decides whether Exp 4150 may graft."""

    artifact_path: Path
    stable_checkpoint_path: Path
    val_exact_accuracy: float | None
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "stable_checkpoint_path": str(self.stable_checkpoint_path),
            "val_exact_accuracy": self.val_exact_accuracy,
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


def _latest_trajectory_val(payload: Mapping[str, Any]) -> float | None:
    trajectory = payload.get("val_trajectory_v384")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        return None
    for entry in reversed(trajectory):
        if isinstance(entry, Mapping):
            val = _float_or_none(entry.get("effective_val_exact_accuracy"))
            if val is None:
                val = _float_or_none(entry.get("val_exact_accuracy"))
            if val is not None:
                return val
    return None


def load_baseline_context(path: str | Path) -> BaselineContext:
    """REQ-LEARN-4150: load Exp 4149 without inventing a faithful baseline."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Exp 4149 artifact must be a JSON object")
    stable_value = payload.get("stable_checkpoint_path")
    stable = Path(stable_value) if isinstance(stable_value, str) and stable_value else Path("")
    val = _float_or_none(payload.get("val_exact_accuracy"))
    if val is None:
        val = _latest_trajectory_val(payload)
    return BaselineContext(
        artifact_path=artifact_path,
        stable_checkpoint_path=stable,
        val_exact_accuracy=val,
        raw=dict(payload),
    )


def baseline_is_faithful(baseline: BaselineContext) -> bool:
    """REQ-LEARN-4150: the graft branch requires validation accuracy at least 0.85."""

    return baseline.val_exact_accuracy is not None and baseline.val_exact_accuracy >= FAITHFUL_VAL_THRESHOLD


def estimate_passes_to_converge_for_385(baseline: BaselineContext | None) -> dict[str, Any]:
    """SCENARIO-LEARN-4150-DEFER: estimate `.385` passes from observed validation movement."""

    current = None if baseline is None else baseline.val_exact_accuracy
    if current is None:
        return {
            "destination": ".385",
            "target_val_exact_accuracy": FAITHFUL_VAL_THRESHOLD,
            "current_val_exact_accuracy": None,
            "estimated_additional_passes": None,
            "basis": "missing_exp4149_val_exact_accuracy",
        }
    if current >= FAITHFUL_VAL_THRESHOLD:
        return {
            "destination": ".385",
            "target_val_exact_accuracy": FAITHFUL_VAL_THRESHOLD,
            "current_val_exact_accuracy": current,
            "estimated_additional_passes": 0,
            "basis": "already_faithful",
        }

    trajectory = [] if baseline is None else baseline.raw.get("val_trajectory_v384", [])
    vals: list[float] = []
    if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
        for row in trajectory:
            if isinstance(row, Mapping):
                val = _float_or_none(row.get("effective_val_exact_accuracy"))
                if val is None:
                    val = _float_or_none(row.get("val_exact_accuracy"))
                if val is not None:
                    vals.append(val)
    positive_deltas = [
        vals[index] - vals[index - 1]
        for index in range(1, len(vals))
        if vals[index] > vals[index - 1]
    ]
    if not positive_deltas:
        return {
            "destination": ".385",
            "target_val_exact_accuracy": FAITHFUL_VAL_THRESHOLD,
            "current_val_exact_accuracy": current,
            "estimated_additional_passes": None,
            "basis": "no_positive_v384_convergence_rate",
        }
    rate = sum(positive_deltas) / len(positive_deltas)
    needed = max(FAITHFUL_VAL_THRESHOLD - current, 0.0)
    return {
        "destination": ".385",
        "target_val_exact_accuracy": FAITHFUL_VAL_THRESHOLD,
        "current_val_exact_accuracy": current,
        "estimated_additional_passes": int(math.ceil(needed / rate)),
        "basis": "mean_positive_v384_delta",
        "mean_positive_delta_per_pass": round(float(rate), 12),
    }


def check_preconditions(
    baseline: BaselineContext,
    *,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> list[PreconditionCheck]:
    """REQ-LEARN-4150: record Exp 4149, checkpoint, and CUDA checks."""

    checks = [
        PreconditionCheck("exp4149_artifact", baseline.artifact_path.is_file(), str(baseline.artifact_path)),
        PreconditionCheck(
            "stable_checkpoint_path",
            bool(str(baseline.stable_checkpoint_path)),
            str(baseline.stable_checkpoint_path),
        ),
    ]
    if baseline.stable_checkpoint_path.is_file():
        try:
            checkpoint_ok, checkpoint_detail = checkpoint_loader(baseline.stable_checkpoint_path)
        except Exception as exc:
            checkpoint_ok, checkpoint_detail = False, f"{type(exc).__name__}: {exc}"
    else:
        checkpoint_ok = False
        checkpoint_detail = f"missing: {baseline.stable_checkpoint_path}"
    checks.append(PreconditionCheck("baseline_checkpoint", bool(checkpoint_ok), str(checkpoint_detail)))

    try:
        cuda_ok, cuda_detail = cuda_checker()
    except Exception as exc:
        cuda_ok, cuda_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("cuda_available", bool(cuda_ok), str(cuda_detail)))
    return checks


def deferred_metric(status: str) -> dict[str, Any]:
    """SCENARIO-LEARN-4150-DEFER: keep metric fields present without fake results."""

    return {
        "metric": "pass@1_exact_accuracy",
        "n_puzzles": 0,
        "vote_pass_at_1": 0.0,
        "verifier_pass_at_1": 0.0,
        "oracle_pass_at_k": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
    }


def deferred_rft_delta(status: str, baseline: BaselineContext | None) -> dict[str, Any]:
    """SCENARIO-LEARN-4150-DEFER: mark the label contrast as honestly deferred."""

    return {
        "metric": "heldout_exact_accuracy",
        "n_matched": 0,
        "a_exact_accuracy": 0.0,
        "b_exact_accuracy": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
        "current_val_exact_accuracy": None if baseline is None else baseline.val_exact_accuracy,
    }


def evaluate_rerank_lift(
    pools: Sequence[CandidatePool],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4150-GRAFT: measure executable-verifier pass@1 lift."""

    metric = exp4109.evaluate_rerank(
        pools,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    verifier_pass = _float_or_none(metric.get("verifier_pass_at_1")) or 0.0
    oracle_pass = _float_or_none(metric.get("oracle_ceiling_pass_at_1")) or 0.0
    return {
        "metric": metric.get("metric", "pass@1_exact_accuracy"),
        "n_puzzles": int(metric.get("n_puzzles", 0)),
        "vote_pass_at_1": metric.get("vote_pass_at_1", 0.0),
        "verifier_pass_at_1": metric.get("verifier_pass_at_1", 0.0),
        "oracle_pass_at_k": metric.get("oracle_ceiling_pass_at_1", 0.0),
        "delta": metric.get("delta", 0.0),
        "ci95": metric.get("ci95", [0.0, 0.0]),
        "lift_vs_oracle": round(float(verifier_pass - oracle_pass), 6),
        "status": "measured",
        "per_puzzle": metric.get("per_puzzle", []),
    }


def verifier_value_added(rft_vs_ablation_delta: Mapping[str, Any], *, graft_deferred: bool) -> bool:
    """REQ-LEARN-4150: compute the headline bool only from the RFT A-vs-B delta."""

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


def _artifact_verdict(*, graft_deferred: bool, value_added: bool) -> str:
    if graft_deferred:
        return "complete: graft_deferred_baseline_below_0.85"
    if value_added:
        return "success: verifier_value_added_rft_A_gt_B"
    return "complete: A~=B null"


def _baseline_status(baseline: BaselineContext | None) -> dict[str, Any]:
    return {
        "artifact_path": None if baseline is None else str(baseline.artifact_path),
        "stable_checkpoint_path": None if baseline is None else str(baseline.stable_checkpoint_path),
        "val_exact_accuracy": None if baseline is None else baseline.val_exact_accuracy,
        "faithful_threshold": FAITHFUL_VAL_THRESHOLD,
        "faithful": False if baseline is None else baseline_is_faithful(baseline),
    }


def compute_reproducibility_checksum(
    *,
    baseline: BaselineContext | None,
    heldout_ids: Sequence[str],
    corpora: Mapping[str, Any] | None = None,
) -> str:
    """REQ-LEARN-4150: hash baseline identity plus held-out labels used by the graft."""

    payload = {
        "schema": "carnot.experiment_4150.decisive_graft.v1",
        "baseline": None if baseline is None else baseline.to_dict(),
        "heldout_ids": list(heldout_ids),
        "corpora": _jsonable(corpora or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    if str(artifact.get("honest_verdict", "")).startswith("blocked_"):
        return False
    if type(artifact.get("graft_deferred")) is not bool:
        return False
    if type(artifact.get("verifier_value_added")) is not bool:
        return False
    baseline_status = artifact.get("baseline_status")
    if not isinstance(baseline_status, Mapping):
        return False
    if artifact.get("graft_deferred") is True:
        val = _float_or_none(baseline_status.get("val_exact_accuracy"))
        return val is not None and val < FAITHFUL_VAL_THRESHOLD
    rerank = artifact.get("rerank_lift_vs_vote")
    rft = artifact.get("rft_vs_ablation_delta")
    return isinstance(rerank, Mapping) and isinstance(rft, Mapping) and _metric_has_ci(rerank) and _metric_has_ci(rft)


def build_result_artifact(
    *,
    baseline: BaselineContext,
    rerank_lift_vs_vote: Mapping[str, Any],
    rft_vs_ablation_delta: Mapping[str, Any],
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    candidate_source: str,
    k_candidates: int,
    n_candidate_pools: int,
    corpora_summary: Mapping[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
    reproducibility_checksum: str | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-4150: build either the honest deferral or real graft artifact."""

    graft_deferred = not baseline_is_faithful(baseline)
    value_added = verifier_value_added(rft_vs_ablation_delta, graft_deferred=graft_deferred)
    heldout_ids = [
        str(row.get("puzzle_id"))
        for row in rerank_lift_vs_vote.get("per_puzzle", [])
        if isinstance(row, Mapping)
    ]
    artifact: dict[str, Any] = {
        "experiment": "experiment_4150_decisive_verifier_graft_sudoku",
        "schema": "carnot.experiment_4150_decisive_verifier_graft_sudoku.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(graft_deferred=graft_deferred, value_added=value_added),
        "graft_deferred": graft_deferred,
        "rerank_lift_vs_vote": _jsonable(rerank_lift_vs_vote),
        "rft_vs_ablation_delta": _jsonable(rft_vs_ablation_delta),
        "verifier_value_added": bool(value_added),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "baseline_status": _baseline_status(baseline),
        "estimated_passes_to_converge_for_385": estimate_passes_to_converge_for_385(baseline),
        "stable_checkpoint_path": str(baseline.stable_checkpoint_path),
        "baseline_artifact_path": str(baseline.artifact_path),
        "candidate_source": candidate_source,
        "k_candidates_per_puzzle": int(k_candidates),
        "n_candidate_pools": int(n_candidate_pools),
        "corpus_summary": _jsonable(corpora_summary or {}),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum
        or compute_reproducibility_checksum(baseline=baseline, heldout_ids=heldout_ids),
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
) -> dict[str, Any]:
    """REQ-LEARN-4150: fail closed when required baseline resources are missing."""

    artifact = {
        "experiment": "experiment_4150_decisive_verifier_graft_sudoku",
        "schema": "carnot.experiment_4150_decisive_verifier_graft_sudoku.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "graft_deferred": True,
        "rerank_lift_vs_vote": deferred_metric("not_run_preconditions_failed"),
        "rft_vs_ablation_delta": deferred_rft_delta("not_run_preconditions_failed", baseline),
        "verifier_value_added": False,
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "baseline_status": _baseline_status(baseline),
        "estimated_passes_to_converge_for_385": estimate_passes_to_converge_for_385(baseline),
        "stable_checkpoint_path": None if baseline is None else str(baseline.stable_checkpoint_path),
        "baseline_artifact_path": None if baseline is None else str(baseline.artifact_path),
        "candidate_source": "none_preconditions_failed",
        "k_candidates_per_puzzle": 0,
        "n_candidate_pools": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": compute_reproducibility_checksum(baseline=baseline, heldout_ids=[]),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4150 deliverable."""

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

    for field_name in ("rerank_lift_vs_vote", "rft_vs_ablation_delta"):
        metric = artifact.get(field_name)
        if not isinstance(metric, Mapping):
            errors.append(f"{field_name} must be an object")
            continue
        if "delta" not in metric:
            errors.append(f"{field_name}.delta is required")
        if not _metric_has_ci(metric):
            errors.append(f"{field_name}.ci95 must have two numeric bounds")

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

    if not isinstance(artifact.get("baseline_status"), Mapping):
        errors.append("baseline_status must be an object")
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
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Write the stable Exp 4150 JSON artifact."""

    validate_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(artifact)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def sample_checkpoint_candidate_pools(  # pragma: no cover - live GPU/checkpoint path.
    *,
    baseline: BaselineContext,
    repo_root: str | Path = REPO_ROOT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    random_seed: int = RANDOM_SEED,
) -> list[CandidatePool]:
    """REQ-LEARN-4150: sample K held-out candidate grids from the stable TRM."""

    return exp4109.sample_checkpoint_candidate_pools(
        checkpoint_path=baseline.stable_checkpoint_path,
        repo_root=repo_root,
        data_dir=data_dir,
        split=split,
        max_puzzles=max_puzzles,
        k_candidates=k_candidates,
        random_seed=random_seed,
    )


def _summarize_corpora(corpora: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(corpora.get("rows", []))
    return {
        "arm_a": corpora.get("arm_a"),
        "arm_b": corpora.get("arm_b"),
        "n_matched": int(corpora.get("n_matched", 0)),
        "skipped_no_verifier_valid": len(corpora.get("skipped_no_verifier_valid", [])),
        "a_exact_count": sum(bool(row.get("a_exact")) for row in rows if isinstance(row, Mapping)),
        "b_exact_count": sum(bool(row.get("b_exact")) for row in rows if isinstance(row, Mapping)),
    }


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    exp4149_artifact_path: str | Path | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    heldout_split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4107._default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    candidate_pool_provider: Callable[[BaselineContext], Sequence[CandidatePool]] | None = None,
    rft_runner: Callable[[BaselineContext, dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4150 and write the decisive verifier-graft gate artifact."""

    started = time.time()
    root = Path(repo_root)
    baseline_path = Path(exp4149_artifact_path) if exp4149_artifact_path is not None else root / "results" / exp4149.RESULT_FILENAME
    try:
        baseline = load_baseline_context(baseline_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        artifact = build_blocked_artifact(
            "blocked_exp4149_baseline_missing",
            baseline=None,
            preconditions_checked=[PreconditionCheck("exp4149_artifact", False, str(baseline_path))],
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    checks = check_preconditions(baseline, cuda_checker=cuda_checker, checkpoint_loader=checkpoint_loader)
    if not _all_preconditions_available(checks):
        artifact = build_blocked_artifact(
            "blocked_exp4150_preconditions_missing",
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    if not baseline_is_faithful(baseline):
        artifact = build_result_artifact(
            baseline=baseline,
            rerank_lift_vs_vote=deferred_metric("deferred_baseline_below_0.85"),
            rft_vs_ablation_delta=deferred_rft_delta("deferred_baseline_below_0.85", baseline),
            preconditions_checked=checks,
            duration_s=time.time() - started,
            candidate_source="none_baseline_below_0.85",
            k_candidates=0,
            n_candidate_pools=0,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    try:
        if candidate_pool_provider is not None:
            pools = list(candidate_pool_provider(baseline))
            candidate_source = "provided_candidate_pool"
        else:  # pragma: no cover - live checkpoint sampling path.
            pools = sample_checkpoint_candidate_pools(
                baseline=baseline,
                repo_root=root,
                data_dir=data_dir,
                split=heldout_split,
                max_puzzles=max_puzzles,
                k_candidates=k_candidates,
                random_seed=random_seed,
            )
            candidate_source = "trm_checkpoint_final_logits_k_sampling"
    except Exception as exc:  # pragma: no cover - defensive live failure path.
        artifact = build_blocked_artifact(
            "blocked_candidate_sampling_failed",
            baseline=baseline,
            preconditions_checked=[
                *checks,
                PreconditionCheck("candidate_sampling", False, f"{type(exc).__name__}: {exc}"),
            ],
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    rerank = evaluate_rerank_lift(pools, random_seed=random_seed, bootstrap_resamples=bootstrap_resamples)
    corpora = exp4109.build_matched_corpora(pools)
    if rft_runner is not None:
        rft_delta = dict(rft_runner(baseline, corpora))
    else:
        rft_delta = exp4109.evaluate_label_arms(
            corpora,
            random_seed=random_seed + 1,
            bootstrap_resamples=bootstrap_resamples,
        )
    heldout_ids = [pool.puzzle_id for pool in pools]
    checksum = compute_reproducibility_checksum(
        baseline=baseline,
        heldout_ids=heldout_ids,
        corpora=corpora,
    )
    artifact = build_result_artifact(
        baseline=baseline,
        rerank_lift_vs_vote=rerank,
        rft_vs_ablation_delta=rft_delta,
        preconditions_checked=checks,
        duration_s=time.time() - started,
        candidate_source=candidate_source,
        k_candidates=k_candidates,
        n_candidate_pools=len(pools),
        corpora_summary=_summarize_corpora(corpora),
        random_seed=random_seed,
        reproducibility_checksum=checksum,
    )
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
