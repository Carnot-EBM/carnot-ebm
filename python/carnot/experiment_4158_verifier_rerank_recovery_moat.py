"""Exp 4158 decision-grade Sudoku verifier rerank recovery moat.

This experiment keeps the `.385` verifier-rerank claim separate from the
faithful-baseline training graft. A low TRM baseline is still useful when
oracle@K proves headroom: the executable Sudoku verifier can then be judged as
an external signal that recovers present-but-out-voted candidates.

Spec refs: REQ-LEARN-4158, SCENARIO-LEARN-4158-NO-HEADROOM,
SCENARIO-LEARN-4158-RERANK-RECOVERY.
"""

from __future__ import annotations

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
RESULT_FILENAME = "experiment_4158_verifier_rerank_recovery_moat.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_EXP4157_ARTIFACT = REPO_ROOT / "results" / "experiment_4157_baseline_harvest_contiguous_continue.json"
DEFAULT_DATA_DIR = REPO_ROOT / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
DEFAULT_HELDOUT_SPLIT = "_valsmall"
RANDOM_SEED = 4158
MIN_MAX_PUZZLES = 64
MIN_K_CANDIDATES = 8
LLM_JUDGE_ESTIMATE_US = 1_000_000.0
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
SPEC_REFS = [
    "REQ-LEARN-4158",
    "SCENARIO-LEARN-4158-NO-HEADROOM",
    "SCENARIO-LEARN-4158-RERANK-RECOVERY",
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "headroom_present",
    "oracle_at_k",
    "vote_at_1",
    "rerank_lift_vs_vote",
    "verifier_recovers_outvoted",
    "cost_ratio_vs_llm_judge",
    "random_seed",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A real CI-excl-0 lift, an honest A~=vote null, OR an "
        "honest no-headroom report are ALL COMPLETE and decision-grade."
    ),
    "headroom_present": (
        "Bare bool: oracle@K > vote@1. The positive control -- a null is only a "
        "moat finding when headroom exists (per the FALSE_NEGATIVE_RISK discipline)."
    ),
    "oracle_at_k": (
        "Fraction of puzzles with >=1 exact-valid candidate; the ceiling the verifier "
        "could recover toward."
    ),
    "vote_at_1": (
        "Majority-vote exact-accuracy; the self-consistency baseline the external "
        "verifier must beat."
    ),
    "rerank_lift_vs_vote": (
        "pass@1(verifier-rerank) - vote@1, with bootstrap CI95. CI95 excluding 0 = "
        "the verifier moat at the rerank level (the .385 decision-grade headline)."
    ),
    "verifier_recovers_outvoted": (
        "Count of present-but-out-voted puzzles the executable verifier recovers "
        "(ARBITER framing); the mechanistic evidence the external signal is "
        "orthogonal to the vote."
    ),
    "cost_ratio_vs_llm_judge": (
        "The EFFICIENCY axis: the executable verifier is a microsecond forward "
        "constraint check vs an LLM-judge call; quantifies the 'parity at 10-100x "
        "cheaper' north-star claim cheaply."
    ),
    "random_seed": "Determinism precondition for the bootstrap CI.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One runtime resource check required before Exp 4158 can sample."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineContext:
    """Exp 4157 baseline evidence used by the rerank moat."""

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


def _format_val_tag(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{float(value):.2f}"


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
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def load_baseline_context(path: str | Path) -> BaselineContext:
    """REQ-LEARN-4158: read Exp 4157 baseline and checkpoint evidence."""

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


def check_preconditions(
    baseline: BaselineContext,
    *,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4109._default_cuda_checker,
) -> list[PreconditionCheck]:
    """REQ-LEARN-4158: verify Exp 4157, checkpoint, and CUDA resources."""

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
    except Exception as exc:  # pragma: no cover - defensive host probe wrapper.
        cuda_ok, cuda_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("cuda_available", bool(cuda_ok), str(cuda_detail)))
    return checks


def snapshot_checkpoint(source_path: str | Path, snapshot_path: str | Path | None = None) -> Path:
    """REQ-LEARN-4158: freeze the live checkpoint before any model load."""

    source = Path(source_path)
    snapshot = Path(snapshot_path) if snapshot_path is not None else source.with_name(f"{source.stem}-rerank-snapshot{source.suffix}")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, snapshot)
    return snapshot


def _empty_rerank(status: str) -> dict[str, Any]:
    return {
        "metric": "pass@1_exact_accuracy",
        "n_puzzles": 0,
        "vote_at_1": 0.0,
        "verifier_pass_at_1": 0.0,
        "oracle_at_k": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
        "per_puzzle": [],
    }


def _empty_cost(status: str) -> dict[str, Any]:
    return {
        "verifier_mean_per_candidate_us": 0.0,
        "llm_judge_estimate_per_candidate_us": LLM_JUDGE_ESTIMATE_US,
        "ratio": 0.0,
        "estimate": "Assumes 1.0s per remote LLM judge call.",
        "status": status,
    }


def evaluate_recovery_moat(
    pools: Sequence[CandidatePool],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4158-RERANK-RECOVERY: evaluate headroom and rerank lift."""

    rerank = exp4109.evaluate_rerank(
        pools,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    vote_at_1 = round(float(rerank.get("vote_pass_at_1", 0.0)), 6)
    oracle_at_k = round(float(rerank.get("oracle_ceiling_pass_at_1", 0.0)), 6)
    headroom_present = bool(oracle_at_k > vote_at_1)
    ci95 = list(rerank.get("ci95", [0.0, 0.0]))
    delta = round(float(rerank.get("delta", 0.0)), 6)
    if not headroom_present:
        status = "no_headroom_oracle_at_k_lte_vote_at_1"
    elif delta > 0.0 and len(ci95) == 2 and float(ci95[0]) > 0.0:
        status = "ci95_excludes_zero_positive"
    else:
        status = "headroom_backed_null_ci95_includes_zero"

    per_puzzle = list(rerank.get("per_puzzle", []))
    recovered = sum(
        1
        for row in per_puzzle
        if row.get("oracle_correct") is True
        and row.get("vote_correct") is False
        and row.get("verifier_correct") is True
    )
    return {
        "headroom_present": headroom_present,
        "oracle_at_k": oracle_at_k,
        "vote_at_1": vote_at_1,
        "rerank_lift_vs_vote": {
            "metric": "pass@1_exact_accuracy",
            "n_puzzles": int(rerank.get("n_puzzles", 0)),
            "vote_at_1": vote_at_1,
            "verifier_pass_at_1": round(float(rerank.get("verifier_pass_at_1", 0.0)), 6),
            "oracle_at_k": oracle_at_k,
            "delta": delta,
            "ci95": ci95,
            "status": status,
            "per_puzzle": per_puzzle,
        },
        "verifier_recovers_outvoted": int(recovered),
    }


def measure_verifier_cost_us(
    pools: Sequence[CandidatePool],
    *,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-LEARN-4158: time the local executable constraint check per candidate."""

    total_candidates = sum(len(pool.candidates) for pool in pools)
    if total_candidates <= 0:
        return _empty_cost("no_candidates")
    started = perf_counter()
    for pool in pools:
        for candidate in pool.candidates:
            exp4109.score_sudoku_candidate(pool.puzzle_tokens, candidate.tokens)
    elapsed_s = max(float(perf_counter()) - float(started), 0.0)
    mean_us = (elapsed_s / total_candidates) * 1_000_000.0
    ratio = round(LLM_JUDGE_ESTIMATE_US / mean_us, 3) if mean_us > 0.0 else None
    return {
        "verifier_mean_per_candidate_us": round(float(mean_us), 3),
        "llm_judge_estimate_per_candidate_us": LLM_JUDGE_ESTIMATE_US,
        "ratio": ratio,
        "estimate": "Assumes 1.0s per remote LLM judge call.",
        "status": "measured_local_constraint_check",
    }


def _verdict(metrics: Mapping[str, Any], current_val: float | None) -> str:
    val_tag = _format_val_tag(current_val)
    if metrics.get("headroom_present") is not True:
        return f"complete: no_headroom_rerank_uninformative_at_val_{val_tag}"
    rerank = metrics.get("rerank_lift_vs_vote", {})
    ci95 = rerank.get("ci95") if isinstance(rerank, Mapping) else None
    delta = _float_or_none(rerank.get("delta")) if isinstance(rerank, Mapping) else None
    if delta is not None and delta > 0.0 and isinstance(ci95, Sequence) and len(ci95) == 2 and float(ci95[0]) > 0.0:
        return f"complete: verifier_rerank_moat_ci95_excludes_zero_at_val_{val_tag}"
    return f"complete: headroom_backed_A_approx_vote_null_at_val_{val_tag}"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or verdict.startswith("blocked_"):
        return False
    if type(artifact.get("headroom_present")) is not bool:
        return False
    if artifact["headroom_present"] is True:
        return (
            _metric_has_ci(artifact.get("rerank_lift_vs_vote", {}))
            and type(artifact.get("verifier_recovers_outvoted")) is int
            and isinstance(artifact.get("cost_ratio_vs_llm_judge"), Mapping)
        )
    return _float_or_none(artifact.get("oracle_at_k")) is not None and _float_or_none(artifact.get("vote_at_1")) is not None


def build_result_artifact(
    *,
    baseline: BaselineContext,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    snapshot_path: str | Path,
    metrics: Mapping[str, Any],
    cost_ratio_vs_llm_judge: Mapping[str, Any],
    random_seed: int,
    duration_s: float,
    candidate_source: str,
    max_puzzles: int,
    k_candidates: int,
    heldout_split: str,
) -> dict[str, Any]:
    """Build the Exp 4158 artifact from measured rerank metrics."""

    artifact: dict[str, Any] = {
        "experiment": "experiment_4158_verifier_rerank_recovery_moat",
        "schema": "carnot.experiment_4158_verifier_rerank_recovery_moat.v1",
        "honest_verdict": _verdict(metrics, baseline.current_val),
        "headroom_present": bool(metrics.get("headroom_present")),
        "oracle_at_k": float(metrics.get("oracle_at_k", 0.0)),
        "vote_at_1": float(metrics.get("vote_at_1", 0.0)),
        "rerank_lift_vs_vote": _jsonable(metrics.get("rerank_lift_vs_vote", _empty_rerank("missing"))),
        "verifier_recovers_outvoted": int(metrics.get("verifier_recovers_outvoted", 0)),
        "cost_ratio_vs_llm_judge": _jsonable(cost_ratio_vs_llm_judge),
        "random_seed": int(random_seed),
        "baseline_status": baseline.to_dict(),
        "snapshot_checkpoint_path": str(snapshot_path),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "candidate_source": candidate_source,
        "heldout_split": heldout_split,
        "max_puzzles": int(max_puzzles),
        "k_candidates": int(k_candidates),
        "n_candidate_pools": int(metrics.get("rerank_lift_vs_vote", {}).get("n_puzzles", 0)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 3),
        "spec_refs": list(SPEC_REFS),
    }
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    baseline: BaselineContext | None,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    random_seed: int,
    duration_s: float,
    detail: str | None = None,
) -> dict[str, Any]:
    """Build a terminal blocked artifact without fabricating rerank evidence."""

    artifact: dict[str, Any] = {
        "experiment": "experiment_4158_verifier_rerank_recovery_moat",
        "schema": "carnot.experiment_4158_verifier_rerank_recovery_moat.v1",
        "honest_verdict": reason,
        "headroom_present": False,
        "oracle_at_k": 0.0,
        "vote_at_1": 0.0,
        "rerank_lift_vs_vote": _empty_rerank(reason),
        "verifier_recovers_outvoted": 0,
        "cost_ratio_vs_llm_judge": _empty_cost(reason),
        "random_seed": int(random_seed),
        "baseline_status": None if baseline is None else baseline.to_dict(),
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 3),
        "spec_refs": list(SPEC_REFS),
        "acceptance_gate_passed": False,
    }
    if detail is not None:
        artifact["blocked_detail"] = detail
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4158 deliverable."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")

    if type(artifact.get("headroom_present")) is not bool:
        errors.append("headroom_present must be a bare bool")
    if not _is_json_number(artifact.get("oracle_at_k")):
        errors.append("oracle_at_k must be a number")
    if not _is_json_number(artifact.get("vote_at_1")):
        errors.append("vote_at_1 must be a number")

    metric = artifact.get("rerank_lift_vs_vote")
    if not isinstance(metric, Mapping):
        errors.append("rerank_lift_vs_vote must be an object")
    else:
        if "delta" not in metric:
            errors.append("rerank_lift_vs_vote.delta is required")
        if not _metric_has_ci(metric):
            errors.append("rerank_lift_vs_vote.ci95 must have two bounds")

    if type(artifact.get("verifier_recovers_outvoted")) is not int:
        errors.append("verifier_recovers_outvoted must be a bare int")
    if not isinstance(artifact.get("cost_ratio_vs_llm_judge"), Mapping):
        errors.append("cost_ratio_vs_llm_judge must be an object")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field_name, principle in FIELD_PRINCIPLES.items():
            if principles.get(field_name) != principle:
                errors.append(f"field_principles.{field_name} mismatch")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Write the stable Exp 4158 JSON artifact."""

    validate_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(artifact)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    exp4157_artifact_path: str | Path = DEFAULT_EXP4157_ARTIFACT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    heldout_split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = MIN_MAX_PUZZLES,
    k_candidates: int = MIN_K_CANDIDATES,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
    cuda_checker: Callable[[], tuple[bool, str]] = exp4109._default_cuda_checker,
    candidate_pool_provider: Callable[[Path], Sequence[CandidatePool]] | None = None,
    verifier_cost_provider: Callable[[Sequence[CandidatePool]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4158 and write the rerank recovery moat artifact."""

    started = time.time()
    baseline: BaselineContext | None = None
    exp4157_path = Path(exp4157_artifact_path)
    if not exp4157_path.is_file():
        artifact = build_blocked_artifact(
            "blocked_exp4157_artifact",
            baseline=None,
            preconditions_checked=[
                PreconditionCheck("exp4157_artifact", False, str(exp4157_path)),
            ],
            random_seed=random_seed,
            duration_s=time.time() - started,
        )
        return write_artifact(output_path, artifact)

    try:
        baseline = load_baseline_context(exp4157_path)
    except Exception as exc:
        artifact = build_blocked_artifact(
            "blocked_exp4157_artifact",
            baseline=None,
            preconditions_checked=[PreconditionCheck("exp4157_artifact", False, str(exp4157_path))],
            random_seed=random_seed,
            duration_s=time.time() - started,
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
            random_seed=random_seed,
            duration_s=time.time() - started,
            detail=first_missing.detail,
        )
        return write_artifact(output_path, artifact)

    try:
        snapshot_path = snapshot_checkpoint(baseline.stable_checkpoint_path)
    except Exception as exc:
        artifact = build_blocked_artifact(
            "blocked_checkpoint_snapshot",
            baseline=baseline,
            preconditions_checked=checks,
            random_seed=random_seed,
            duration_s=time.time() - started,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return write_artifact(output_path, artifact)

    live_max_puzzles = max(int(max_puzzles), MIN_MAX_PUZZLES)
    live_k_candidates = max(int(k_candidates), MIN_K_CANDIDATES)
    if candidate_pool_provider is not None:
        pools = list(candidate_pool_provider(snapshot_path))
        candidate_source = "provided_candidate_pool"
    else:  # pragma: no cover - live CUDA checkpoint sampling path.
        pools = exp4109.sample_checkpoint_candidate_pools(
            checkpoint_path=snapshot_path,
            repo_root=repo_root,
            data_dir=data_dir,
            split=heldout_split,
            max_puzzles=live_max_puzzles,
            k_candidates=live_k_candidates,
            random_seed=random_seed,
        )
        candidate_source = "snapshot_checkpoint_final_logits_k_sampling"

    metrics = evaluate_recovery_moat(
        pools,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    cost = (
        dict(verifier_cost_provider(pools))
        if verifier_cost_provider is not None
        else measure_verifier_cost_us(pools)
    )
    artifact = build_result_artifact(
        baseline=baseline,
        preconditions_checked=checks,
        snapshot_path=snapshot_path,
        metrics=metrics,
        cost_ratio_vs_llm_judge=cost,
        random_seed=random_seed,
        duration_s=time.time() - started,
        candidate_source=candidate_source,
        max_puzzles=live_max_puzzles,
        k_candidates=live_k_candidates,
        heldout_split=heldout_split,
    )
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
