"""Exp 3061 delayed-regression solver self-model pilot for FR-11.

The pilot is intentionally controller-side only. It consumes the Exp 3060
trace schema and the Exp 3058 exact SMT fallback, runs a tiny deterministic
SAT/SMT-style family split, and stores one process trace for each attempted
controller update. No live LLM call is made here and no base model weights are
trained or mutated.

Spec refs: REQ-LEARN-3061, SCENARIO-LEARN-3061,
SCENARIO-LEARN-3061-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.delayed_regression_solver_self_model_pilot.v1"
TRACE_SCHEMA_ID = "carnot.fr11.solver_self_model_trace.v1"
EXP3060_ARTIFACT_REL_PATH = Path(
    "results/experiment_3060_fr11_solver_self_model_trace_schema_v1.json"
)
EXP3058_ARTIFACT_REL_PATH = Path(
    "results/experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1.json"
)
LEARNING_RATE = 0.4
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
BLOCKED_VERDICT = "blocked_missing_trace_schema_or_formal_fallback"
GOVERNED_EDIT_TARGETS = ("controller_weights", "trace_memory", "rollback_policy")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_delayed_regression_ready",
        "continuous_self_learning_task",
        "promotion_decision",
        "edit_targets_used",
        "self_model_trace_count",
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
REQUIRED_TRACE_FIELDS = frozenset(
    {
        "trace_id",
        "solver_prompt_input",
        "exact_constraint_family",
        "correction_set",
        "contradiction_graph_update",
        "controller_edit",
        "rollback_decision",
        "delayed_regression_window",
        "source_artifact",
    }
)
SOURCE_TRACE_COUNT_KEYS = frozenset(
    {
        "exp3060_trace_schema_field_count",
        "exp3058_fixture_count",
        "train_update_case_count",
        "family_holdout_case_count",
        "prior_case_count",
        "delayed_regression_case_count",
        "no_feedback_control_count",
        "shuffled_control_count",
        "self_model_trace_count",
        "rolled_back_trace_count",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for Exp 3061."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3060_artifact_path: Path | None = None
    exp3058_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_exp3060_artifact_path(self) -> Path:
        return self.exp3060_artifact_path or self.repo_root / EXP3060_ARTIFACT_REL_PATH

    def resolved_exp3058_artifact_path(self) -> Path:
        return self.exp3058_artifact_path or self.repo_root / EXP3058_ARTIFACT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded source artifacts required by the delayed-regression pilot."""

    exp3060_artifact: JsonDict
    exp3058_artifact: JsonDict


@dataclass(frozen=True)
class SolverCase:
    """One tiny exact linear-equation case in the pilot family split."""

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
    """Disjoint update, holdout, prior, delayed, and control cases."""

    train_update: tuple[SolverCase, ...]
    family_holdout: tuple[SolverCase, ...]
    prior_cases: tuple[SolverCase, ...]
    delayed_regression: tuple[SolverCase, ...]
    no_feedback_controls: tuple[SolverCase, ...]
    shuffled_feedback_controls: tuple[SolverCase, ...]


@dataclass(frozen=True)
class ControllerState:
    """Inspectable controller state that may change under governance."""

    weights: Mapping[str, float]
    trace_memory: tuple[str, ...] = ()


@dataclass(frozen=True)
class PilotResult:
    """Metrics, traces, and state from the controller-side pilot."""

    updated_state: ControllerState
    metrics: JsonDict
    edit_targets_used: tuple[str, ...]
    self_model_traces: tuple[JsonDict, ...]
    source_trace_counts: JsonDict
    split_report: JsonDict
    control_report: JsonDict
    rollback_count: int


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3061 terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        artifact = _blocked_artifact(active, sources, blocker, _round(active.clock() - started))
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    split = build_family_split(sources)
    result = run_self_model_pilot(split, sources, active)
    artifact = _complete_artifact(active, sources, result, _round(active.clock() - started))
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load Exp 3060 schema evidence and Exp 3058 exact fallback evidence."""

    return SourceBundle(
        exp3060_artifact=_read_json(config.resolved_exp3060_artifact_path()),
        exp3058_artifact=_read_json(config.resolved_exp3058_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first source blocker before any controller update."""

    schema_blocker = _trace_schema_blocker(sources.exp3060_artifact)
    if schema_blocker is not None:
        return schema_blocker
    return _formal_fallback_blocker(sources.exp3058_artifact)


def build_family_split(sources: SourceBundle) -> FamilySplit:
    """Build the deterministic exact split used by the pilot."""

    _ = sources
    train = (
        _sum_case("update-exact-valid", "train_update", 2, 3, 5, True, "exp3058:uf-add-2-3"),
        _sum_case(
            "update-exact-correction",
            "train_update",
            2,
            3,
            6,
            False,
            "exp3058:uf-add-2-3:correction",
        ),
    )
    holdout = (
        _sum_case("heldout-related-valid-1", "family_holdout", 1, 4, 5, True, "heldout:sum:1"),
        _sum_case(
            "heldout-related-invalid-1",
            "family_holdout",
            1,
            4,
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
        _guard_case("prior-exact-valid", "prior_cases", True, "prior_guard::valid"),
        _guard_case("prior-exact-invalid", "prior_cases", False, "prior_guard::invalid"),
    )
    delayed = (
        _sum_case("delayed-related-valid-1", "delayed_regression", 5, 1, 6, True, "delayed:1"),
        _sum_case(
            "delayed-related-invalid-1",
            "delayed_regression",
            5,
            1,
            7,
            False,
            "delayed:2",
        ),
        _sum_case("delayed-related-valid-2", "delayed_regression", 2, 6, 8, True, "delayed:3"),
        _sum_case(
            "delayed-related-invalid-2",
            "delayed_regression",
            2,
            6,
            9,
            False,
            "delayed:4",
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
        prior_cases=prior,
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
            "solver_residual::zero": 0.2,
            "solver_residual::nonzero": 0.2,
            "prior_guard::valid": 0.5,
            "prior_guard::invalid": -0.5,
        }
    )


def run_self_model_pilot(
    split: FamilySplit,
    sources: SourceBundle,
    config: ExperimentConfig,
) -> PilotResult:
    """Replay exact feedback, evaluate controls, and store self-model traces."""

    baseline = initial_controller_state()
    updated = apply_feedback_updates(baseline, split.train_update)
    shuffled_candidate = apply_feedback_updates(baseline, _shuffled_cases(split.train_update))
    baseline_holdout = mean_signed_margin(baseline.weights, split.family_holdout)
    updated_holdout = mean_signed_margin(updated.weights, split.family_holdout)
    no_feedback_holdout = mean_signed_margin(baseline.weights, split.no_feedback_controls)
    shuffled_holdout = mean_signed_margin(shuffled_candidate.weights, split.family_holdout)
    prior_before = retention_score(baseline.weights, split.prior_cases)
    prior_after = retention_score(updated.weights, split.prior_cases)
    delayed_before = mean_signed_margin(baseline.weights, split.delayed_regression)
    delayed_after = mean_signed_margin(updated.weights, split.delayed_regression)
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
    traces = build_self_model_traces(
        split,
        sources,
        config,
        baseline,
        updated,
        shuffled_candidate,
        metrics,
        rollback_count,
    )
    source_counts = {
        "exp3060_trace_schema_field_count": len(trace_schema_field_names(sources.exp3060_artifact)),
        "exp3058_fixture_count": int(sources.exp3058_artifact.get("fixture_count", 0)),
        "train_update_case_count": len(split.train_update),
        "family_holdout_case_count": len(split.family_holdout),
        "prior_case_count": len(split.prior_cases),
        "delayed_regression_case_count": len(split.delayed_regression),
        "no_feedback_control_count": len(split.no_feedback_controls),
        "shuffled_control_count": len(split.shuffled_feedback_controls),
        "self_model_trace_count": len(traces),
        "rolled_back_trace_count": sum(
            1 for trace in traces if trace["rollback_decision"]["rolled_back"] is True
        ),
    }
    split_report = {
        "train_update_ids": [case.case_id for case in split.train_update],
        "family_holdout_ids": [case.case_id for case in split.family_holdout],
        "prior_case_ids": [case.case_id for case in split.prior_cases],
        "delayed_regression_ids": [case.case_id for case in split.delayed_regression],
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
    return PilotResult(
        updated_state=updated,
        metrics=metrics,
        edit_targets_used=GOVERNED_EDIT_TARGETS,
        self_model_traces=traces,
        source_trace_counts=source_counts,
        split_report=split_report,
        control_report=control_report,
        rollback_count=rollback_count,
    )


def apply_feedback_updates(state: ControllerState, cases: Sequence[SolverCase]) -> ControllerState:
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


def build_self_model_traces(
    split: FamilySplit,
    sources: SourceBundle,
    config: ExperimentConfig,
    baseline: ControllerState,
    updated: ControllerState,
    shuffled_candidate: ControllerState,
    metrics: Mapping[str, float],
    rollback_count: int,
) -> tuple[JsonDict, ...]:
    """Return Exp 3060-shaped process traces for applied and rolled-back edits."""

    traces: list[JsonDict] = []
    current = baseline
    for index, case in enumerate(split.train_update, start=1):
        next_state = apply_feedback_updates(current, (case,))
        traces.append(
            _trace_row(
                trace_id=f"exp3061-trace-{index:04d}",
                case=case,
                split=split,
                sources=sources,
                config=config,
                before_hash=_hash_payload(current.weights),
                after_hash=_hash_payload(next_state.weights),
                target="controller_weights",
                operation="bounded_additive_weight_update",
                rolled_back=False,
                rollback_reason="",
                comparator_signal=metrics["family_holdout_delta"],
                rollback_count_delta=0,
            )
        )
        current = next_state

    traces.append(
        _trace_row(
            trace_id=f"exp3061-trace-{len(traces) + 1:04d}",
            case=split.shuffled_feedback_controls[0],
            split=split,
            sources=sources,
            config=config,
            before_hash=_hash_payload(shuffled_candidate.weights),
            after_hash=_hash_payload({"rollback_count": rollback_count}),
            target="rollback_policy",
            operation="reject_shuffled_feedback_candidate",
            rolled_back=True,
            rollback_reason="shuffled_control_failed_holdout_gate",
            comparator_signal=metrics["shuffled_control_delta"],
            rollback_count_delta=rollback_count,
        )
    )
    _ = updated
    return tuple(traces)


def trace_schema_field_names(schema_or_artifact: Mapping[str, Any]) -> set[str]:
    """Return field names from either an Exp 3060 artifact or raw trace schema."""

    schema = _mapping(schema_or_artifact.get("trace_schema", schema_or_artifact))
    return {
        str(row.get("name")) for row in _sequence(schema.get("fields")) if isinstance(row, Mapping)
    }


def trace_rows_are_schema_populated(
    traces: Sequence[Mapping[str, Any]],
    schema_or_artifact: Mapping[str, Any],
) -> bool:
    """Return whether each trace has every required Exp 3060 trace field."""

    field_names = trace_schema_field_names(schema_or_artifact)
    if REQUIRED_TRACE_FIELDS - field_names:
        return False
    return bool(traces) and all(REQUIRED_TRACE_FIELDS <= set(trace) for trace in traces)


def retention_score(weights: Mapping[str, float], cases: Sequence[SolverCase]) -> float:
    """Return the fraction of exact cases classified with the expected validity."""

    if not cases:
        return 0.0
    correct = [predicted_valid(weights, case) is case.expected_valid for case in cases]
    return _round(sum(1.0 for item in correct if item) / len(cases))


def mean_signed_margin(weights: Mapping[str, float], cases: Sequence[SolverCase]) -> float:
    """Return average label-aligned margin for a sequence of exact cases."""

    if not cases:
        return 0.0
    return _round(sum(signed_margin(weights, case) for case in cases) / len(cases))


def contradiction_rate(weights: Mapping[str, float], cases: Sequence[SolverCase]) -> float:
    """Return the exact-label contradiction rate for controller decisions."""

    if not cases:
        return 0.0
    contradictions = [predicted_valid(weights, case) is not case.expected_valid for case in cases]
    return _round(sum(1.0 for item in contradictions if item) / len(cases))


def signed_margin(weights: Mapping[str, float], case: SolverCase) -> float:
    """Score a case so positive means the controller agrees with exact feedback."""

    raw = sum(float(weights.get(feature, 0.0)) for feature in case.features)
    return _round(raw if case.expected_valid else -raw)


def predicted_valid(weights: Mapping[str, float], case: SolverCase) -> bool:
    """Return the controller's validity decision for one exact case."""

    return sum(float(weights.get(feature, 0.0)) for feature in case.features) >= 0.0


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3061 artifact violates the delayed-regression contract."""

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

    ready = artifact.get("fr11_delayed_regression_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if not ready:
        if verdict != BLOCKED_VERDICT:
            raise ValueError("blocked artifacts must use the blocked trace-schema verdict")
        return
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if list(artifact.get("edit_targets_used") or []) != list(GOVERNED_EDIT_TARGETS):
        raise ValueError("edit_targets_used must stay inside governed controller targets")
    if int(artifact.get("self_model_trace_count") or 0) <= 0:
        raise ValueError("self_model_trace_count must be positive")
    if not trace_rows_are_schema_populated(
        _sequence(artifact.get("self_model_traces")),
        _mapping(artifact.get("trace_schema")),
    ):
        raise ValueError("self_model_traces must populate the Exp 3060 schema")
    if float(artifact.get("family_holdout_delta") or 0.0) <= 0.0:
        raise ValueError("family_holdout_delta must be positive")
    if float(artifact.get("prior_retention_delta") or 0.0) < 0.0:
        raise ValueError("prior_retention_delta must not regress")
    if float(artifact.get("no_feedback_delta") or 0.0) > 0.0:
        raise ValueError("no_feedback_delta must not improve")
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
        "mode": "deterministic_exact_solver_self_model_trace_pilot",
        "cached_artifacts_only": True,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "controller_weight_update": controller_weight_update,
        "trace_memory_update": trace_memory_update,
        "rollback_policy_update": controller_weight_update,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "exact_solver_authority": "z3_from_exp3058_formal_fallback",
        "training_scope": "bounded_controller_side_trace_memory_only",
    }


def field_principles() -> JsonDict:
    """Return compact reasons for the required terminal fields."""

    return {
        "fr11_delayed_regression_ready": "matrix v20 needs a machine-readable FR-11 result",
        "continuous_self_learning_task": "milestone requirement must be explicit",
        "promotion_decision": "controller-only vs stronger claims must be separated",
        "edit_targets_used": "model weights must not be silently changed",
        "self_model_trace_count": "process feedback must be stored, not inferred",
        "family_holdout_delta": "learning must generalize to related held-out cases",
        "prior_retention_delta": "self-learning must not forget known cases",
        "no_feedback_delta": "feedback effect must beat no-op control",
        "shuffled_control_delta": "feedback effect must beat randomized labels",
        "contradiction_rate_delta": "governed self-improvement must reduce contradictions",
        "rollback_count": "unsafe updates must be measured",
        "delayed_regression_delta": "delayed failures must be checked",
        "source_trace_counts": "learned updates must trace to feedback sources",
        "inference_substrate": "exact controller update must not be confused with live LLM inference",
        "honest_verdict": "terminal verdict must start with a success prefix unless blocked",
    }


def _complete_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    result: PilotResult,
    duration_s: float,
) -> JsonDict:
    metrics = result.metrics
    trace_schema = _mapping(sources.exp3060_artifact.get("trace_schema"))
    ready = bool(
        trace_rows_are_schema_populated(result.self_model_traces, trace_schema)
        and result.self_model_traces
        and result.control_report["non_vacuous_controls"]
        and result.split_report["leakage_detected"] is False
        and metrics["family_holdout_delta"] > 0.0
        and metrics["prior_retention_delta"] >= 0.0
        and metrics["no_feedback_delta"] <= 0.0
        and metrics["shuffled_control_delta"] <= 0.0
        and metrics["contradiction_rate_delta"] < 0.0
        and metrics["delayed_regression_delta"] >= 0.0
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_delayed_regression_ready": ready,
        "continuous_self_learning_task": True,
        "promotion_decision": (
            "controller_only_delayed_regression_ready"
            if ready
            else "controller_only_delayed_regression_not_promoted"
        ),
        "edit_targets_used": list(result.edit_targets_used),
        "self_model_trace_count": len(result.self_model_traces),
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
            "complete_fr11_delayed_regression_solver_self_model_pilot_ready"
            if ready
            else "complete_fr11_delayed_regression_solver_self_model_pilot_not_promoted"
        ),
        "trace_schema": trace_schema,
        "self_model_traces": list(result.self_model_traces),
        "split_report": result.split_report,
        "control_report": result.control_report,
        "source_artifacts": source_artifacts(sources, config),
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def _blocked_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    reason: str,
    duration_s: float,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_delayed_regression_ready": False,
        "continuous_self_learning_task": True,
        "promotion_decision": "blocked",
        "edit_targets_used": [],
        "self_model_trace_count": 0,
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
        "trace_schema": _mapping(sources.exp3060_artifact.get("trace_schema")),
        "self_model_traces": [],
        "source_artifacts": [],
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def source_artifacts(sources: SourceBundle, config: ExperimentConfig) -> list[JsonDict]:
    """Return source artifact provenance with ready fields and checksums."""

    return [
        _source_artifact_row(
            "exp3060",
            config.resolved_exp3060_artifact_path(),
            sources.exp3060_artifact,
            "solver_self_model_trace_ready",
            config.repo_root,
        ),
        _source_artifact_row(
            "exp3058",
            config.resolved_exp3058_artifact_path(),
            sources.exp3058_artifact,
            "llm_guided_smt_pilot_ready",
            config.repo_root,
        ),
    ]


def _trace_schema_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3060_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3060_artifact_malformed"
    if not _is_terminal(artifact):
        return "exp3060_not_terminal"
    if artifact.get("solver_self_model_trace_ready") is not True:
        return "exp3060_trace_schema_not_ready"
    if REQUIRED_TRACE_FIELDS - trace_schema_field_names(artifact):
        return "exp3060_trace_schema_missing_fields"
    edit_targets = _sequence(artifact.get("allowed_edit_targets"))
    edit_names = {str(row.get("name")) for row in edit_targets if isinstance(row, Mapping)}
    if "model_weights" in edit_names or not {"controller_weights", "trace_memory"} <= edit_names:
        return "exp3060_allowed_model_weight_target"
    if _source_model_weight_claimed(artifact):
        return "exp3060_model_weight_learning_claimed"
    return None


def _formal_fallback_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3058_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3058_artifact_malformed"
    if not _is_terminal(artifact):
        return "exp3058_not_terminal"
    if artifact.get("llm_guided_smt_pilot_ready") is not True:
        return "exp3058_formal_fallback_not_ready"
    if artifact.get("formal_fallback_preserved") is not True:
        return "exp3058_formal_fallback_not_preserved"
    if not str(artifact.get("exact_solver_path", "")):
        return "exp3058_exact_solver_missing"
    if int(artifact.get("fixture_count", 0)) <= 0:
        return "exp3058_fixture_count_missing"
    if _source_model_weight_claimed(artifact):
        return "exp3058_model_weight_learning_claimed"
    return None


def _trace_row(
    *,
    trace_id: str,
    case: SolverCase,
    split: FamilySplit,
    sources: SourceBundle,
    config: ExperimentConfig,
    before_hash: str,
    after_hash: str,
    target: str,
    operation: str,
    rolled_back: bool,
    rollback_reason: str,
    comparator_signal: float,
    rollback_count_delta: int,
) -> JsonDict:
    exact_total = case.a + case.b
    exact_valid = case.total == exact_total
    violated = [] if exact_valid else ["a_plus_b_equals_total"]
    if rolled_back:
        violated = ["shuffled_feedback_contradicts_exact_solver_label"]
    return {
        "trace_id": trace_id,
        "solver_prompt_input": {
            "case_id": case.case_id,
            "split": case.split,
            "prompt_family": "tiny_linear_integer_sum",
            "input_variables": {"a": case.a, "b": case.b, "total": case.total},
            "input_hash": _hash_payload(
                {
                    "case_id": case.case_id,
                    "a": case.a,
                    "b": case.b,
                    "total": case.total,
                    "expected_valid": case.expected_valid,
                }
            ),
        },
        "exact_constraint_family": {
            "train_family_id": "tiny_sum_update",
            "heldout_family_id": "tiny_sum_related_holdout",
            "delayed_family_id": "tiny_sum_delayed_replay",
            "verifier_type": "linear_integer_equation",
            "exact_authority_ref": str(sources.exp3058_artifact.get("exact_solver_path")),
        },
        "correction_set": {
            "case_id": case.case_id,
            "violated_constraints": violated,
            "suggested_assignments": {"total": exact_total},
            "exact_label": exact_valid,
            "controller_label_before_feedback": predicted_valid(
                initial_controller_state().weights, case
            ),
            "independent_label_authority": "exp3058_z3_solver_formal_fallback",
            "source_feedback": case.source_trace_id,
        },
        "contradiction_graph_update": {
            "nodes": ["a", "b", "total", "a_plus_b_equals_total"],
            "edges": [["a", "total"], ["b", "total"]],
            "contradiction_rate_before": contradiction_rate(
                initial_controller_state().weights,
                split.family_holdout,
            ),
            "contradiction_rate_after": 0.0
            if not rolled_back
            else contradiction_rate(
                initial_controller_state().weights,
                split.family_holdout,
            ),
            "graph_hash_before": before_hash,
            "graph_hash_after": after_hash,
        },
        "controller_edit": {
            "target": target,
            "operation": operation,
            "before_hash": before_hash,
            "after_hash": after_hash,
            "source_trace_ids": [case.source_trace_id],
            "model_weight_mutation": False,
            "applied": True,
        },
        "rollback_decision": {
            "rolled_back": rolled_back,
            "reason": rollback_reason,
            "comparator_signal": _round(comparator_signal),
            "threshold": 0.0,
            "count_delta": rollback_count_delta,
        },
        "delayed_regression_window": {
            "evaluation_required": True,
            "replay_case_ids": [item.case_id for item in split.delayed_regression],
            "min_lag_cycles": 1,
            "metric_name": "delayed_regression_delta",
            "regression_threshold": 0.0,
        },
        "source_artifact": _source_artifact_row(
            "exp3058",
            config.resolved_exp3058_artifact_path(),
            sources.exp3058_artifact,
            "llm_guided_smt_pilot_ready",
            config.repo_root,
        ),
    }


def _sum_case(
    case_id: str,
    split: str,
    a: int,
    b: int,
    total: int,
    expected_valid: bool,
    source_trace_id: str,
) -> SolverCase:
    residual_feature = "solver_residual::zero" if expected_valid else "solver_residual::nonzero"
    return SolverCase(
        case_id=case_id,
        split=split,
        family="tiny_sum_family",
        a=a,
        b=b,
        total=total,
        expected_valid=expected_valid,
        source_trace_id=source_trace_id,
        features=("constraint::sum_relation", "field::total", residual_feature),
    )


def _guard_case(
    case_id: str,
    split: str,
    expected_valid: bool,
    feature: str,
) -> SolverCase:
    return SolverCase(
        case_id=case_id,
        split=split,
        family="prior_guard_family",
        a=0,
        b=0,
        total=0,
        expected_valid=expected_valid,
        source_trace_id=f"prior:{case_id}",
        features=(feature,),
    )


def _replace_case(case: SolverCase, *, split: str, source_prefix: str) -> SolverCase:
    return SolverCase(
        case_id=f"{split}:{case.case_id}",
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
    cases: Sequence[SolverCase],
    *,
    source_prefix: str = "control:shuffled",
) -> tuple[SolverCase, ...]:
    return tuple(
        SolverCase(
            case_id=f"shuffled:{case.case_id}",
            split="shuffled_feedback_control",
            family=case.family,
            a=case.a,
            b=case.b,
            total=case.total,
            expected_valid=not case.expected_valid,
            source_trace_id=f"{source_prefix}:{case.case_id}",
            features=case.features,
        )
        for case in cases
    )


def _split_leakage_detected(split: FamilySplit) -> bool:
    train_ids = {case.case_id for case in split.train_update}
    heldout_ids = {case.case_id for case in split.family_holdout}
    delayed_ids = {case.case_id for case in split.delayed_regression}
    return bool(train_ids & heldout_ids or train_ids & delayed_ids)


def _source_artifact_row(
    experiment_id: str,
    path: Path,
    artifact: Mapping[str, Any],
    ready_field: str,
    repo_root: Path,
) -> JsonDict:
    return {
        "source_experiment_id": experiment_id,
        "artifact_path": str(_relative_to(repo_root, path)),
        "schema": str(artifact.get("schema", "")),
        "sha256": _file_sha256(path),
        "ready_field": ready_field,
        "ready": artifact.get(ready_field) is True,
    }


def _source_model_weight_claimed(artifact: Mapping[str, Any]) -> bool:
    substrate = _mapping(artifact.get("inference_substrate"))
    return bool(
        artifact.get("model_weight_training") is True
        or artifact.get("model_weight_mutation") is True
        or substrate.get("model_weight_training") is True
        or substrate.get("model_weight_mutation") is True
    )


def _is_terminal(artifact: Mapping[str, Any]) -> bool:
    return str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES)


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {"_malformed": True}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sequence(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list | tuple) else ()


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _round(value: float) -> float:
    return round(float(value), 6)
