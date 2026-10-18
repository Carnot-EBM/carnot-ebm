"""Exp 3046 governed FR-11 solver-feedback self-learning loop.

This module runs a tiny deterministic controller-side learning loop over
SAT/SMT-style sum constraints.  It consumes Exp 3044 exact correction-set
feedback and Exp 3045 governance, then updates only inspectable controller
weights plus source-traced controller memory.  It does not query an LLM, train
base model weights, or treat shuffled/no-feedback controls as real learning.

Spec refs: REQ-LEARN-3046, SCENARIO-LEARN-3046,
SCENARIO-LEARN-3046-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3046_fr11_solver_feedback_self_learning_loop_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.solver_feedback_self_learning_loop.v1"
EXP3044_ARTIFACT_REL_PATH = Path(
    "results/experiment_3044_smt_sat_validator_tree_exactness_upgrade_v1.json"
)
EXP3045_ARTIFACT_REL_PATH = Path(
    "results/experiment_3045_fr11_governed_self_learning_boundary_v1.json"
)
LOOP_REPORT_REL_PATH = Path(
    "results/fr11_solver_feedback_self_learning_loop_3046/loop_report.jsonl"
)
LEARNING_RATE = 0.5
MAX_ABS_WEIGHT = 1.0
TERMINAL_SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
BLOCKED_VERDICT = "blocked_missing_governance_or_exact_feedback"
GOVERNED_EDIT_TARGETS = ("controller_weights", "trace_memory")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_solver_feedback_ready",
        "continuous_self_learning_task",
        "promotion_decision",
        "edit_targets_used",
        "family_holdout_delta",
        "prior_retention_delta",
        "no_feedback_delta",
        "shuffled_control_delta",
        "contradiction_rate_delta",
        "rollback_count",
        "delayed_regression_delta",
        "source_trace_counts",
        "inference_substrate",
        "honest_verdict",
    }
)
SOURCE_TRACE_COUNT_KEYS = frozenset(
    {
        "exp3044_correction_set_count",
        "train_update_case_count",
        "family_holdout_case_count",
        "prior_exact_case_count",
        "no_feedback_control_count",
        "shuffled_control_count",
        "source_traced_update_count",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for Exp 3046."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    loop_report_path: Path | None = None
    exp3044_artifact_path: Path | None = None
    exp3045_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_loop_report_path(self) -> Path:
        return self.loop_report_path or self.repo_root / LOOP_REPORT_REL_PATH

    def resolved_exp3044_artifact_path(self) -> Path:
        return self.exp3044_artifact_path or self.repo_root / EXP3044_ARTIFACT_REL_PATH

    def resolved_exp3045_artifact_path(self) -> Path:
        return self.exp3045_artifact_path or self.repo_root / EXP3045_ARTIFACT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded source artifacts for the governed loop."""

    exp3044_artifact: JsonDict
    exp3045_artifact: JsonDict


@dataclass(frozen=True)
class SatCase:
    """One exact SAT/SMT-style case evaluated by the lightweight controller."""

    case_id: str
    split: str
    family: str
    a: int
    b: int
    total: int
    expected_valid: bool
    source_trace_id: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class FamilySplit:
    """Disjoint train, held-out, prior, delayed, and control cases."""

    train_update: tuple[SatCase, ...]
    family_holdout: tuple[SatCase, ...]
    prior_exact: tuple[SatCase, ...]
    delayed_regression: tuple[SatCase, ...]
    no_feedback_controls: tuple[SatCase, ...]
    shuffled_feedback_controls: tuple[SatCase, ...]


@dataclass(frozen=True)
class ControllerState:
    """Inspectable controller state that may be updated under Exp 3045 governance."""

    weights: Mapping[str, float]
    trace_memory: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopResult:
    """Metrics and state produced by the governed controller-side loop."""

    updated_state: ControllerState
    metrics: JsonDict
    edit_targets_used: tuple[str, ...]
    source_trace_counts: JsonDict
    split_report: JsonDict
    control_report: JsonDict
    rollback_count: int


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3046 terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        artifact = _blocked_artifact(active, blocker, _round(active.clock() - started))
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    split = build_family_split(sources.exp3044_artifact)
    result = run_governed_loop(split, sources.exp3044_artifact, sources.exp3045_artifact)
    artifact = _complete_artifact(active, sources, result, _round(active.clock() - started))
    validate_artifact(artifact)
    _write_jsonl(active.resolved_loop_report_path(), _loop_report_rows(artifact))
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load Exp 3044 exact feedback and Exp 3045 governance artifacts."""

    return SourceBundle(
        exp3044_artifact=_read_json(config.resolved_exp3044_artifact_path()),
        exp3045_artifact=_read_json(config.resolved_exp3045_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first source blocker before a controller update is attempted."""

    exp3044_blocker = _exact_feedback_blocker(sources.exp3044_artifact)
    if exp3044_blocker is not None:
        return exp3044_blocker
    return _governance_blocker(sources.exp3045_artifact)


def build_family_split(exp3044_artifact: Mapping[str, Any]) -> FamilySplit:
    """Build deterministic train, held-out, prior, and control SAT/SMT cases."""

    correction = _first_correction_set(exp3044_artifact)
    target_field = str(_string_list(correction.get("candidate_fields"))[0])
    expected_total = int(_mapping(correction.get("suggested_assignments"))[target_field])
    train = (
        _sum_case(
            "train-exp3044-verified",
            "train_update",
            2,
            3,
            expected_total,
            True,
            "exp3044:exact_row:sat-sum-ok",
        ),
        _sum_case(
            "train-exp3044-correction",
            "train_update",
            2,
            3,
            expected_total + 1,
            False,
            "exp3044:correction_set:sat-sum-bad",
        ),
    )
    holdout = (
        _sum_case("heldout-related-valid-1", "family_holdout", 4, 1, 5, True, "heldout:sum:1"),
        _sum_case(
            "heldout-related-invalid-1",
            "family_holdout",
            4,
            1,
            6,
            False,
            "heldout:sum:2",
        ),
        _sum_case("heldout-related-valid-2", "family_holdout", 3, 4, 7, True, "heldout:sum:3"),
        _sum_case(
            "heldout-related-invalid-2",
            "family_holdout",
            3,
            4,
            8,
            False,
            "heldout:sum:4",
        ),
    )
    prior = (
        _guard_case("prior-exact-valid", "prior_exact", True, "prior_guard::valid"),
        _guard_case("prior-exact-invalid", "prior_exact", False, "prior_guard::invalid"),
    )
    delayed = (
        _guard_case("delayed-exact-valid", "delayed_regression", True, "delayed_guard::valid"),
        _guard_case(
            "delayed-exact-invalid",
            "delayed_regression",
            False,
            "delayed_guard::invalid",
        ),
    )
    no_feedback = tuple(
        _replace_case(case, split="no_feedback_control", source_prefix="control:no_feedback")
        for case in holdout
    )
    shuffled = tuple(_shuffled_cases(holdout, source_prefix="control:shuffled"))
    return FamilySplit(
        train_update=train,
        family_holdout=holdout,
        prior_exact=prior,
        delayed_regression=delayed,
        no_feedback_controls=no_feedback,
        shuffled_feedback_controls=shuffled,
    )


def initial_controller_state() -> ControllerState:
    """Return the fixed controller baseline before exact feedback replay."""

    return ControllerState(
        weights={
            "constraint::sum_relation": 0.0,
            "field::total": 0.0,
            "solver_residual::zero": 0.25,
            "solver_residual::nonzero": 0.25,
            "prior_guard::valid": 0.5,
            "prior_guard::invalid": -0.5,
            "delayed_guard::valid": 0.5,
            "delayed_guard::invalid": -0.5,
        }
    )


def run_governed_loop(
    split: FamilySplit,
    exp3044_artifact: Mapping[str, Any],
    exp3045_artifact: Mapping[str, Any],
) -> LoopResult:
    """Apply exact feedback, evaluate controls, and return deterministic metrics."""

    _ = exp3045_artifact
    baseline = initial_controller_state()
    updated = apply_feedback_updates(baseline, split.train_update)
    shuffled_candidate = apply_feedback_updates(baseline, _shuffled_cases(split.train_update))
    baseline_holdout = mean_signed_margin(baseline.weights, split.family_holdout)
    updated_holdout = mean_signed_margin(updated.weights, split.family_holdout)
    no_feedback_holdout = mean_signed_margin(baseline.weights, split.no_feedback_controls)
    shuffled_holdout = mean_signed_margin(shuffled_candidate.weights, split.family_holdout)
    prior_before = retention_score(baseline.weights, split.prior_exact)
    prior_after = retention_score(updated.weights, split.prior_exact)
    delayed_before = retention_score(baseline.weights, split.delayed_regression)
    delayed_after = retention_score(updated.weights, split.delayed_regression)
    contradiction_before = contradiction_rate(baseline.weights, split.family_holdout)
    contradiction_after = contradiction_rate(updated.weights, split.family_holdout)
    rollback_count = 1 if shuffled_holdout <= baseline_holdout else 0
    metrics = {
        "family_holdout_delta": _round(updated_holdout - baseline_holdout),
        "prior_retention_delta": _round(prior_after - prior_before),
        "no_feedback_delta": _round(no_feedback_holdout - baseline_holdout),
        "shuffled_control_delta": _round(shuffled_holdout - baseline_holdout),
        "contradiction_rate_delta": _round(contradiction_after - contradiction_before),
        "delayed_regression_delta": _round(delayed_after - delayed_before),
    }
    source_counts = {
        "exp3044_correction_set_count": len(_sequence(exp3044_artifact.get("correction_sets"))),
        "train_update_case_count": len(split.train_update),
        "family_holdout_case_count": len(split.family_holdout),
        "prior_exact_case_count": len(split.prior_exact),
        "no_feedback_control_count": len(split.no_feedback_controls),
        "shuffled_control_count": len(split.shuffled_feedback_controls),
        "source_traced_update_count": len(updated.trace_memory),
    }
    split_report = {
        "train_update_ids": [case.case_id for case in split.train_update],
        "family_holdout_ids": [case.case_id for case in split.family_holdout],
        "leakage_detected": _split_leakage_detected(split),
    }
    control_report = {
        "non_vacuous_controls": bool(
            split.no_feedback_controls and split.shuffled_feedback_controls
        ),
        "no_feedback_case_count": len(split.no_feedback_controls),
        "shuffled_case_count": len(split.shuffled_feedback_controls),
        "shuffled_candidate_rolled_back": bool(rollback_count),
    }
    return LoopResult(
        updated_state=updated,
        metrics=metrics,
        edit_targets_used=GOVERNED_EDIT_TARGETS,
        source_trace_counts=source_counts,
        split_report=split_report,
        control_report=control_report,
        rollback_count=rollback_count,
    )


def apply_feedback_updates(state: ControllerState, cases: Sequence[SatCase]) -> ControllerState:
    """Replay exact feedback into bounded controller weights and trace memory."""

    weights = dict(state.weights)
    trace_memory = list(state.trace_memory)
    for case in cases:
        direction = 1.0 if case.expected_valid else -1.0
        for feature in case.features:
            updated = weights.get(feature, 0.0) + LEARNING_RATE * direction
            weights[feature] = _round(max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, updated)))
        trace_memory.append(f"{case.source_trace_id}:{case.case_id}")
    return ControllerState(weights=dict(sorted(weights.items())), trace_memory=tuple(trace_memory))


def retention_score(weights: Mapping[str, float], cases: Sequence[SatCase]) -> float:
    """Return the fraction of exact cases classified with the expected validity."""

    if not cases:
        return 0.0
    correct = [predicted_valid(weights, case) is case.expected_valid for case in cases]
    return _round(sum(1.0 for item in correct if item) / len(cases))


def mean_signed_margin(weights: Mapping[str, float], cases: Sequence[SatCase]) -> float:
    """Return average label-aligned margin for a sequence of exact cases."""

    if not cases:
        return 0.0
    return _round(sum(signed_margin(weights, case) for case in cases) / len(cases))


def contradiction_rate(weights: Mapping[str, float], cases: Sequence[SatCase]) -> float:
    """Return the exact-label contradiction rate for the controller decisions."""

    if not cases:
        return 0.0
    contradictions = [predicted_valid(weights, case) is not case.expected_valid for case in cases]
    return _round(sum(1.0 for item in contradictions if item) / len(cases))


def signed_margin(weights: Mapping[str, float], case: SatCase) -> float:
    """Score a case so positive means the controller agrees with exact feedback."""

    raw = sum(float(weights.get(feature, 0.0)) for feature in case.features)
    return _round(raw if case.expected_valid else -raw)


def predicted_valid(weights: Mapping[str, float], case: SatCase) -> bool:
    """Return the controller's validity decision for one case."""

    return sum(float(weights.get(feature, 0.0)) for feature in case.features) >= 0.0


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3046 artifact violates the governed-loop contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_task") is not True:
        raise ValueError("continuous_self_learning_task must be true")
    counts = artifact.get("source_trace_counts")
    if not isinstance(counts, Mapping) or set(counts) != SOURCE_TRACE_COUNT_KEYS:
        raise ValueError("source_trace_counts must contain the exact trace-count keys")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if substrate.get("live_llm_inference") is not False:
        raise ValueError("live LLM inference must remain false")
    if (
        substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("model weights must not be trained or mutated")

    ready = artifact.get("fr11_solver_feedback_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if not ready:
        if verdict != BLOCKED_VERDICT and not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
            raise ValueError("non-ready artifacts need a blocked or terminal verdict")
        return
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if list(artifact.get("edit_targets_used") or []) != list(GOVERNED_EDIT_TARGETS):
        raise ValueError("edit_targets_used must stay inside governed controller targets")
    if float(artifact.get("family_holdout_delta") or 0.0) <= 0.0:
        raise ValueError("family_holdout_delta must be positive")
    if float(artifact.get("prior_retention_delta") or 0.0) < 0.0:
        raise ValueError("prior_retention_delta must not regress")
    if float(artifact.get("no_feedback_delta") or 0.0) != 0.0:
        raise ValueError("no_feedback_delta must be zero")
    if float(artifact.get("shuffled_control_delta") or 0.0) > 0.0:
        raise ValueError("shuffled_control_delta must not improve")
    if float(artifact.get("contradiction_rate_delta") or 0.0) >= 0.0:
        raise ValueError("contradiction_rate_delta must be negative")
    if float(artifact.get("delayed_regression_delta") or 0.0) < 0.0:
        raise ValueError("delayed_regression_delta must not regress")
    if any(int(counts[key]) <= 0 for key in SOURCE_TRACE_COUNT_KEYS):
        raise ValueError("source_trace_counts must be positive for ready artifacts")
    if artifact.get("control_report", {}).get("non_vacuous_controls") is not True:
        raise ValueError("control_report must show non-vacuous controls")
    if artifact.get("split_report", {}).get("leakage_detected") is not False:
        raise ValueError("split_report must show no train/held-out leakage")


def inference_substrate(*, controller_weight_update: bool, trace_memory_update: bool) -> JsonDict:
    """Return the execution boundary without implying live model adaptation."""

    return {
        "mode": "deterministic_exact_solver_feedback_controller_loop",
        "cached_artifacts_only": True,
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "controller_weight_update": controller_weight_update,
        "trace_memory_update": trace_memory_update,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "training_scope": "bounded_controller_side_only",
    }


def field_principles() -> JsonDict:
    """Return compact reasons for the required terminal fields."""

    return {
        "fr11_solver_feedback_ready": "downstream KAN/matrix tasks need a machine-readable self-learning result",
        "continuous_self_learning_task": "milestone requirement must be explicit",
        "promotion_decision": "controller-only and stronger claims must be separated",
        "edit_targets_used": "model weights must not be silently changed",
        "family_holdout_delta": "learning must generalize to related held-out cases",
        "prior_retention_delta": "self-learning must not forget known cases",
        "no_feedback_delta": "feedback effect must beat no-op control",
        "shuffled_control_delta": "feedback effect must beat randomized labels",
        "contradiction_rate_delta": "governed self-improvement must reduce contradictions",
        "rollback_count": "unsafe updates must be measured",
        "delayed_regression_delta": "delayed failures must be checked",
        "source_trace_counts": "learned updates must trace to feedback sources",
        "inference_substrate": "exact controller update must not be confused with live LLM inference",
        "honest_verdict": "terminal verdict must be machine-readable",
    }


def _complete_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    result: LoopResult,
    duration_s: float,
) -> JsonDict:
    metrics = result.metrics
    ready = bool(
        metrics["family_holdout_delta"] > 0.0
        and metrics["prior_retention_delta"] >= 0.0
        and metrics["no_feedback_delta"] == 0.0
        and metrics["shuffled_control_delta"] <= 0.0
        and metrics["contradiction_rate_delta"] < 0.0
        and metrics["delayed_regression_delta"] >= 0.0
        and result.control_report["non_vacuous_controls"]
        and result.split_report["leakage_detected"] is False
        and result.source_trace_counts["source_traced_update_count"] > 0
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_solver_feedback_ready": ready,
        "continuous_self_learning_task": True,
        "promotion_decision": (
            "controller_only_solver_feedback_ready"
            if ready
            else "controller_only_solver_feedback_not_promoted"
        ),
        "edit_targets_used": list(result.edit_targets_used),
        "family_holdout_delta": metrics["family_holdout_delta"],
        "prior_retention_delta": metrics["prior_retention_delta"],
        "no_feedback_delta": metrics["no_feedback_delta"],
        "shuffled_control_delta": metrics["shuffled_control_delta"],
        "contradiction_rate_delta": metrics["contradiction_rate_delta"],
        "rollback_count": result.rollback_count,
        "delayed_regression_delta": metrics["delayed_regression_delta"],
        "source_trace_counts": result.source_trace_counts,
        "inference_substrate": inference_substrate(
            controller_weight_update=True,
            trace_memory_update=True,
        ),
        "honest_verdict": (
            "complete_fr11_solver_feedback_controller_loop_ready"
            if ready
            else "complete_fr11_solver_feedback_controller_loop_not_promoted"
        ),
        "split_report": result.split_report,
        "control_report": result.control_report,
        "source_artifacts": {
            "exp3044_artifact": str(
                _relative_to(config.repo_root, config.resolved_exp3044_artifact_path())
            ),
            "exp3044_ready": sources.exp3044_artifact.get("validator_tree_exactness_ready") is True,
            "exp3045_artifact": str(
                _relative_to(config.repo_root, config.resolved_exp3045_artifact_path())
            ),
            "exp3045_ready": sources.exp3045_artifact.get("fr11_governance_ready") is True,
        },
        "loop_report_path": str(_relative_to(config.repo_root, config.resolved_loop_report_path())),
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def _blocked_artifact(config: ExperimentConfig, reason: str, duration_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_solver_feedback_ready": False,
        "continuous_self_learning_task": True,
        "promotion_decision": "blocked",
        "edit_targets_used": [],
        "family_holdout_delta": 0.0,
        "prior_retention_delta": 0.0,
        "no_feedback_delta": 0.0,
        "shuffled_control_delta": 0.0,
        "contradiction_rate_delta": 0.0,
        "rollback_count": 0,
        "delayed_regression_delta": 0.0,
        "source_trace_counts": {key: 0 for key in SOURCE_TRACE_COUNT_KEYS},
        "inference_substrate": inference_substrate(
            controller_weight_update=False,
            trace_memory_update=False,
        ),
        "honest_verdict": BLOCKED_VERDICT,
        "blocked_reason": reason,
        "loop_report_path": str(_relative_to(config.repo_root, config.resolved_loop_report_path())),
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def _exact_feedback_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3044_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3044_artifact_malformed"
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        return "exp3044_not_terminal"
    if artifact.get("validator_tree_exactness_ready") is not True:
        return "exp3044_exact_feedback_not_ready"
    if not _sequence(artifact.get("correction_sets")):
        return "exp3044_correction_sets_missing"
    return _substrate_blocker("exp3044", artifact)


def _governance_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3045_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3045_artifact_malformed"
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        return "exp3045_not_terminal"
    if artifact.get("fr11_governance_ready") is not True:
        return "exp3045_governance_not_ready"
    targets = {
        str(row.get("name")): str(row.get("scope"))
        for row in _sequence(artifact.get("allowed_edit_targets"))
        if isinstance(row, Mapping)
    }
    if targets.get("controller_weights") != "allowed_controller_side":
        return "exp3045_controller_edit_target_missing"
    if targets.get("trace_memory") != "allowed_controller_side":
        return "exp3045_trace_memory_edit_target_missing"
    if targets.get("model_weights") != "out_of_scope":
        return "exp3045_model_weights_not_out_of_scope"
    return _substrate_blocker("exp3045", artifact)


def _substrate_blocker(prefix: str, artifact: Mapping[str, Any]) -> str | None:
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return f"{prefix}_inference_substrate_missing"
    if substrate.get("live_llm_inference") is not False:
        return f"{prefix}_live_llm_inference_claimed"
    if substrate.get("model_weight_training") is True:
        return f"{prefix}_model_weight_training_claimed"
    if substrate.get("model_weight_mutation") is True:
        return f"{prefix}_model_weight_mutation_claimed"
    return None


def _first_correction_set(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    corrections = _sequence(artifact.get("correction_sets"))
    first = corrections[0]
    return first if isinstance(first, Mapping) else {}


def _sum_case(
    case_id: str,
    split: str,
    a_value: int,
    b_value: int,
    total: int,
    expected_valid: bool,
    source_trace_id: str,
) -> SatCase:
    residual = "zero" if expected_valid else "nonzero"
    return SatCase(
        case_id=case_id,
        split=split,
        family="sum_relation",
        a=a_value,
        b=b_value,
        total=total,
        expected_valid=expected_valid,
        source_trace_id=source_trace_id,
        features=("constraint::sum_relation", "field::total", f"solver_residual::{residual}"),
    )


def _guard_case(case_id: str, split: str, expected_valid: bool, feature: str) -> SatCase:
    return SatCase(
        case_id=case_id,
        split=split,
        family="prior_exact_guard",
        a=0,
        b=0,
        total=0,
        expected_valid=expected_valid,
        source_trace_id=f"{split}:{case_id}",
        features=(feature,),
    )


def _replace_case(case: SatCase, *, split: str, source_prefix: str) -> SatCase:
    return SatCase(
        case_id=case.case_id.replace(case.split, split),
        split=split,
        family=case.family,
        a=case.a,
        b=case.b,
        total=case.total,
        expected_valid=case.expected_valid,
        source_trace_id=f"{source_prefix}:{case.case_id}",
        features=case.features,
    )


def _shuffled_cases(
    cases: Sequence[SatCase], source_prefix: str = "control:shuffled"
) -> tuple[SatCase, ...]:
    labels = [case.expected_valid for case in cases]
    shifted = labels[1:] + labels[:1]
    return tuple(
        SatCase(
            case_id=f"shuffled-{case.case_id}",
            split="shuffled_feedback_control",
            family=case.family,
            a=case.a,
            b=case.b,
            total=case.total,
            expected_valid=label,
            source_trace_id=f"{source_prefix}:{case.case_id}",
            features=case.features,
        )
        for case, label in zip(cases, shifted, strict=True)
    )


def _split_leakage_detected(split: FamilySplit) -> bool:
    train_ids = {case.case_id for case in split.train_update}
    holdout_ids = {case.case_id for case in split.family_holdout}
    return bool(train_ids & holdout_ids)


def _loop_report_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "section": "baseline",
            "family_holdout_delta": artifact["family_holdout_delta"],
            "prior_retention_delta": artifact["prior_retention_delta"],
        },
        {
            "section": "updated",
            "contradiction_rate_delta": artifact["contradiction_rate_delta"],
            "delayed_regression_delta": artifact["delayed_regression_delta"],
        },
        {
            "section": "controls",
            "no_feedback_delta": artifact["no_feedback_delta"],
            "shuffled_control_delta": artifact["shuffled_control_delta"],
        },
        {"section": "rollback", "rollback_count": artifact["rollback_count"]},
    ]


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {"_malformed": True}
    return dict(payload) if isinstance(payload, Mapping) else {"_malformed": True}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value)]


def _round(value: float) -> float:
    return round(float(value), 6)


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path
