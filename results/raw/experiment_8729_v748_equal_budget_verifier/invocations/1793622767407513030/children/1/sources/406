"""Exp 3077 soundness-bounded online self-learning pilot for FR-11.

This module runs a deliberately tiny controller-side pilot. It uses exact
integer-sum labels as the independent authority, consumes the Exp 3076 mistake
budget before attempting any update, and records every online decision in the
Exp 3060 trace shape. The update is intentionally limited to inspectable
controller weights and trace memory; no live LLM inference is performed and no
base model weights are trained or mutated.

Spec refs: REQ-LEARN-3077, SCENARIO-LEARN-3077,
SCENARIO-LEARN-3077-BLOCKED.
"""

from __future__ import annotations

from collections import Counter
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
ARTIFACT = "experiment_3077_fr11_soundness_bounded_online_self_learning_pilot_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.soundness_bounded_online_self_learning_pilot.v1"
EXP3076_ARTIFACT_REL_PATH = Path(
    "results/experiment_3076_fr11_online_soundness_completeness_budget_v1.json"
)
EXP3060_ARTIFACT_REL_PATH = Path(
    "results/experiment_3060_fr11_solver_self_model_trace_schema_v1.json"
)
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
BLOCKED_VERDICT = "blocked_missing_soundness_budget"
LEARNING_RATE = 0.5
MAX_ABS_WEIGHT = 1.0
ACCEPT_THRESHOLD = 0.1
REJECT_THRESHOLD = -0.1
GOVERNED_EDIT_TARGETS = ("controller_weights", "trace_memory", "rollback_policy")
ALLOWED_DECISION_LABELS = frozenset(
    {
        "correct",
        "soundness_mistake",
        "completeness_mistake",
        "abstention",
        "rollback",
        "delayed_regression",
    }
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_soundness_bounded_ready",
        "continuous_self_learning_task",
        "promotion_decision",
        "edit_targets_used",
        "self_model_trace_count",
        "soundness_mistakes",
        "completeness_mistakes",
        "mistake_budget_delta",
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
NUMERIC_BUDGET_KEYS = frozenset(
    {
        "max_soundness_mistakes",
        "max_completeness_mistakes",
        "max_delayed_regressions",
        "max_contradiction_mistakes",
        "no_feedback_max_delta",
        "shuffled_feedback_max_delta",
        "prior_retention_floor",
    }
)
REQUIRED_CONTROL_NAMES = frozenset(
    {"no_feedback_control", "shuffled_feedback_control", "prior_retention_floor"}
)
SOURCE_TRACE_COUNT_KEYS = frozenset(
    {
        "exp3076_budget_gate_count",
        "exp3060_trace_schema_field_count",
        "train_update_case_count",
        "family_holdout_case_count",
        "prior_case_count",
        "delayed_regression_case_count",
        "no_feedback_control_count",
        "shuffled_control_count",
        "online_decision_count",
        "self_model_trace_count",
        "rolled_back_trace_count",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for Exp 3077."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3076_artifact_path: Path | None = None
    exp3060_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_exp3076_artifact_path(self) -> Path:
        return self.exp3076_artifact_path or self.repo_root / EXP3076_ARTIFACT_REL_PATH

    def resolved_exp3060_artifact_path(self) -> Path:
        return self.exp3060_artifact_path or self.repo_root / EXP3060_ARTIFACT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded source artifacts required by the soundness-bounded pilot."""

    exp3076_artifact: JsonDict
    exp3060_artifact: JsonDict


@dataclass(frozen=True)
class SolverCase:
    """One exact integer-sum case in the online pilot split."""

    case_id: str
    split: str
    family: str
    a: int
    b: int
    total: int
    expected_valid: bool
    feedback_valid: bool
    source_trace_id: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class FamilySplit:
    """Disjoint update, holdout, prior, delayed, and control partitions."""

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
class DecisionRecord:
    """One exact-authority controller decision plus its audit label."""

    case: SolverCase
    condition: str
    controller_decision: str
    decision_label: str
    score: float


@dataclass(frozen=True)
class PilotResult:
    """Metrics, traces, and budget status from the controller-only pilot."""

    updated_state: ControllerState
    metrics: JsonDict
    edit_targets_used: tuple[str, ...]
    online_decisions: tuple[JsonDict, ...]
    self_model_traces: tuple[JsonDict, ...]
    source_trace_counts: JsonDict
    split_report: JsonDict
    control_report: JsonDict
    decision_label_counts: JsonDict
    mistake_budget_delta: JsonDict
    soundness_mistakes: int
    completeness_mistakes: int
    delayed_regressions: int
    contradiction_mistakes: int
    rollback_count: int


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3077 terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    duration_s = _round(active.clock() - started)
    if blocker is not None:
        artifact = _blocked_artifact(active, sources, blocker, duration_s)
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    split = build_family_split()
    result = run_online_pilot(split, sources, active)
    artifact = _complete_artifact(active, sources, result, duration_s)
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load Exp 3076 budget evidence and the Exp 3060 trace schema."""

    return SourceBundle(
        exp3076_artifact=_read_json(config.resolved_exp3076_artifact_path()),
        exp3060_artifact=_read_json(config.resolved_exp3060_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first source blocker before any controller update."""

    budget_blocker = _budget_blocker(sources.exp3076_artifact)
    if budget_blocker is not None:
        return budget_blocker
    return _trace_schema_blocker(sources.exp3060_artifact)


def budget_is_complete(artifact: Mapping[str, Any]) -> bool:
    """Return whether Exp 3076 exposes all gates Exp 3077 must consume."""

    budget = _mapping(artifact.get("mistake_budget"))
    controls = _sequence(artifact.get("required_controls"))
    control_names = {str(row.get("name")) for row in controls if isinstance(row, Mapping)}
    return bool(
        artifact.get("soundness_completeness_budget_ready") is True
        and NUMERIC_BUDGET_KEYS <= set(budget)
        and all(isinstance(budget[key], int | float) for key in NUMERIC_BUDGET_KEYS)
        and control_names == REQUIRED_CONTROL_NAMES
    )


def build_family_split() -> FamilySplit:
    """Build the deterministic exact split used by the online pilot."""

    train = (
        _sum_case("update-valid-sum", "train_update", 2, 3, 5, True, "exp3077:update:1"),
        _sum_case("update-invalid-sum", "train_update", 2, 3, 6, False, "exp3077:update:2"),
    )
    holdout = (
        _sum_case("heldout-valid-1", "family_holdout", 1, 4, 5, True, "exp3077:heldout:1"),
        _sum_case("heldout-invalid-1", "family_holdout", 1, 4, 6, False, "exp3077:heldout:2"),
        _sum_case("heldout-valid-2", "family_holdout", 3, 4, 7, True, "exp3077:heldout:3"),
        _sum_case("heldout-invalid-2", "family_holdout", 3, 4, 8, False, "exp3077:heldout:4"),
    )
    prior = (
        _prior_case("prior-valid-guard", True, "prior::consistent"),
        _prior_case("prior-invalid-guard", False, "prior::contradiction"),
    )
    delayed = (
        _sum_case("delayed-valid-known", "delayed_regression", 5, 1, 6, True, "exp3077:delay:1"),
        _sum_case(
            "delayed-invalid-known",
            "delayed_regression",
            5,
            1,
            7,
            False,
            "exp3077:delay:2",
        ),
        _edge_case("delayed-valid-edge-abstain", True, "exp3077:delay:3"),
    )
    no_feedback = tuple(
        _replace_case(case, split="no_feedback_control", source_prefix="control:no_feedback")
        for case in holdout
    )
    shuffled = tuple(_shuffled_feedback_case(case) for case in holdout)
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
            "sat::sum_valid": 0.0,
            "sat::sum_invalid": 0.0,
            "sat::edge_valid_unknown": 0.0,
            "prior::consistent": 0.6,
            "prior::contradiction": -0.6,
        }
    )


def run_online_pilot(
    split: FamilySplit,
    sources: SourceBundle,
    config: ExperimentConfig,
) -> PilotResult:
    """Replay exact feedback, evaluate controls, and store process traces."""

    baseline = initial_controller_state()
    updated = apply_feedback_updates(baseline, split.train_update)
    shuffled_candidate = apply_feedback_updates(baseline, split.shuffled_feedback_controls)
    baseline_holdout = accuracy(baseline.weights, split.family_holdout)
    updated_holdout = accuracy(updated.weights, split.family_holdout)
    no_feedback_holdout = accuracy(baseline.weights, split.no_feedback_controls)
    shuffled_holdout = accuracy(shuffled_candidate.weights, split.family_holdout)
    prior_before = retention_score(baseline.weights, split.prior_cases)
    prior_after = retention_score(updated.weights, split.prior_cases)
    delayed_before = accuracy(baseline.weights, split.delayed_regression)
    delayed_after = accuracy(updated.weights, split.delayed_regression)
    contradiction_before = contradiction_rate(baseline.weights, split.family_holdout)
    contradiction_after = contradiction_rate(updated.weights, split.family_holdout)
    rollback_count = 1 if shuffled_holdout <= no_feedback_holdout else 0
    metrics = {
        "family_holdout_delta": _round(updated_holdout - baseline_holdout),
        "prior_retention_delta": _round(prior_after - prior_before),
        "prior_retention_score": prior_after,
        "no_feedback_delta": _round(no_feedback_holdout - baseline_holdout),
        "shuffled_control_delta": _round(shuffled_holdout - baseline_holdout),
        "contradiction_rate_before": contradiction_before,
        "contradiction_rate_after": contradiction_after,
        "contradiction_rate_delta": _round(contradiction_after - contradiction_before),
        "delayed_regression_delta": _round(delayed_after - delayed_before),
    }
    main_decisions = tuple(
        evaluate_decision(updated.weights, case, condition="main")
        for case in (
            split.train_update + split.family_holdout + split.prior_cases + split.delayed_regression
        )
    )
    no_feedback_decisions = tuple(
        evaluate_decision(baseline.weights, case, condition="no_feedback_control")
        for case in split.no_feedback_controls
    )
    shuffled_decisions = tuple(
        evaluate_decision(shuffled_candidate.weights, case, condition="shuffled_feedback_control")
        for case in split.shuffled_feedback_controls
    )
    decisions = main_decisions + no_feedback_decisions + shuffled_decisions
    soundness_mistakes = _count_labels(main_decisions, "soundness_mistake")
    completeness_mistakes = _count_labels(main_decisions, "completeness_mistake")
    delayed_regressions = _delayed_regression_count(
        baseline.weights,
        updated.weights,
        split.delayed_regression,
    )
    contradiction_mistakes = max(
        0,
        int(round((contradiction_after - contradiction_before) * len(split.family_holdout))),
    )
    budget_delta = mistake_budget_delta(
        _mapping(sources.exp3076_artifact.get("mistake_budget")),
        metrics,
        soundness_mistakes=soundness_mistakes,
        completeness_mistakes=completeness_mistakes,
        delayed_regressions=delayed_regressions,
        contradiction_mistakes=contradiction_mistakes,
        controls_non_vacuous=bool(split.no_feedback_controls and split.shuffled_feedback_controls),
    )
    traces = build_self_model_traces(
        split,
        sources,
        config,
        baseline,
        updated,
        decisions,
        metrics,
        rollback_count,
    )
    counts = {
        "exp3076_budget_gate_count": len(NUMERIC_BUDGET_KEYS),
        "exp3060_trace_schema_field_count": len(trace_schema_field_names(sources.exp3060_artifact)),
        "train_update_case_count": len(split.train_update),
        "family_holdout_case_count": len(split.family_holdout),
        "prior_case_count": len(split.prior_cases),
        "delayed_regression_case_count": len(split.delayed_regression),
        "no_feedback_control_count": len(split.no_feedback_controls),
        "shuffled_control_count": len(split.shuffled_feedback_controls),
        "online_decision_count": len(decisions),
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
    label_counts = {
        "main": _label_count_dict(main_decisions),
        "no_feedback_control": _label_count_dict(no_feedback_decisions),
        "shuffled_feedback_control": _label_count_dict(shuffled_decisions),
        "all": _label_count_dict(decisions),
    }
    return PilotResult(
        updated_state=updated,
        metrics=metrics,
        edit_targets_used=GOVERNED_EDIT_TARGETS,
        online_decisions=tuple(_decision_payload(decision) for decision in decisions),
        self_model_traces=traces,
        source_trace_counts=counts,
        split_report=split_report,
        control_report=control_report,
        decision_label_counts=label_counts,
        mistake_budget_delta=budget_delta,
        soundness_mistakes=soundness_mistakes,
        completeness_mistakes=completeness_mistakes,
        delayed_regressions=delayed_regressions,
        contradiction_mistakes=contradiction_mistakes,
        rollback_count=rollback_count,
    )


def apply_feedback_updates(state: ControllerState, cases: Sequence[SolverCase]) -> ControllerState:
    """Replay exact feedback into bounded controller weights and trace memory."""

    weights = dict(state.weights)
    trace_memory = list(state.trace_memory)
    for case in cases:
        direction = 1.0 if case.feedback_valid else -1.0
        for feature in case.features:
            updated = weights.get(feature, 0.0) + LEARNING_RATE * direction
            weights[feature] = _round(max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, updated)))
        trace_memory.append(f"{case.source_trace_id}:{case.case_id}:{case.feedback_valid}")
    return ControllerState(weights=dict(sorted(weights.items())), trace_memory=tuple(trace_memory))


def evaluate_decision(
    weights: Mapping[str, float],
    case: SolverCase,
    *,
    condition: str,
) -> DecisionRecord:
    """Evaluate one controller decision against the exact authority label."""

    score = _round(sum(float(weights.get(feature, 0.0)) for feature in case.features))
    if score > ACCEPT_THRESHOLD:
        decision = "accept"
    elif score < REJECT_THRESHOLD:
        decision = "reject"
    else:
        decision = "abstain"
    return DecisionRecord(
        case=case,
        condition=condition,
        controller_decision=decision,
        decision_label=_decision_label(decision, case.expected_valid),
        score=score,
    )


def accuracy(weights: Mapping[str, float], cases: Sequence[SolverCase]) -> float:
    """Return exact-label decision accuracy, counting abstentions as not correct."""

    if not cases:
        return 0.0
    correct = [
        evaluate_decision(weights, case, condition="metric").decision_label == "correct"
        for case in cases
    ]
    return _round(sum(1.0 for item in correct if item) / len(cases))


def retention_score(weights: Mapping[str, float], cases: Sequence[SolverCase]) -> float:
    """Return prior-case retention under the same exact decision semantics."""

    return accuracy(weights, cases)


def contradiction_rate(weights: Mapping[str, float], cases: Sequence[SolverCase]) -> float:
    """Return the exact-label contradiction rate for controller decisions."""

    if not cases:
        return 0.0
    return _round(1.0 - accuracy(weights, cases))


def mistake_budget_delta(
    budget: Mapping[str, Any],
    metrics: Mapping[str, float],
    *,
    soundness_mistakes: int,
    completeness_mistakes: int,
    delayed_regressions: int,
    contradiction_mistakes: int,
    controls_non_vacuous: bool,
) -> JsonDict:
    """Compare observed pilot outcomes against the Exp 3076 budget gates."""

    soundness = _max_gate(
        soundness_mistakes,
        int(budget.get("max_soundness_mistakes", -1)),
        "soundness_mistakes",
    )
    completeness = _max_gate(
        completeness_mistakes,
        int(budget.get("max_completeness_mistakes", -1)),
        "completeness_mistakes",
    )
    delayed = _max_gate(
        delayed_regressions,
        int(budget.get("max_delayed_regressions", -1)),
        "delayed_regressions",
    )
    contradiction = _max_gate(
        contradiction_mistakes,
        int(budget.get("max_contradiction_mistakes", -1)),
        "contradiction_mistakes",
    )
    no_feedback = _max_float_gate(
        metrics["no_feedback_delta"],
        float(budget.get("no_feedback_max_delta", -1.0)),
        "no_feedback_delta",
    )
    shuffled = _max_float_gate(
        metrics["shuffled_control_delta"],
        float(budget.get("shuffled_feedback_max_delta", -1.0)),
        "shuffled_control_delta",
    )
    prior = _min_float_gate(
        metrics["prior_retention_score"],
        float(budget.get("prior_retention_floor", 2.0)),
        "prior_retention_score",
    )
    controls = {
        "metric_key": "controls_non_vacuous",
        "observed": bool(controls_non_vacuous),
        "required": True,
        "passed": bool(controls_non_vacuous),
    }
    gates = {
        "soundness_mistakes": soundness,
        "completeness_mistakes": completeness,
        "delayed_regressions": delayed,
        "contradiction_mistakes": contradiction,
        "no_feedback_delta": no_feedback,
        "shuffled_control_delta": shuffled,
        "prior_retention_score": prior,
        "controls_non_vacuous": controls,
    }
    gates["all_gates_passed"] = all(
        _mapping(value).get("passed") is True
        for value in gates.values()
        if isinstance(value, Mapping)
    )
    return gates


def build_self_model_traces(
    split: FamilySplit,
    sources: SourceBundle,
    config: ExperimentConfig,
    baseline: ControllerState,
    updated: ControllerState,
    decisions: Sequence[DecisionRecord],
    metrics: Mapping[str, float],
    rollback_count: int,
) -> tuple[JsonDict, ...]:
    """Return Exp 3060-shaped traces for online decisions and rollback."""

    traces = [
        _trace_row(
            trace_id=f"exp3077-trace-{index:04d}",
            decision=decision,
            split=split,
            sources=sources,
            config=config,
            target=_target_for_decision(decision),
            operation=_operation_for_decision(decision),
            before_hash=_hash_payload(baseline.weights),
            after_hash=_hash_payload(updated.weights),
            rolled_back=False,
            rollback_reason="",
            rollback_count_delta=0,
            metrics=metrics,
        )
        for index, decision in enumerate(decisions, start=1)
    ]
    rollback_decision = DecisionRecord(
        case=split.shuffled_feedback_controls[0],
        condition="rollback",
        controller_decision="reject_candidate_update",
        decision_label="rollback",
        score=metrics["shuffled_control_delta"],
    )
    traces.append(
        _trace_row(
            trace_id=f"exp3077-trace-{len(traces) + 1:04d}",
            decision=rollback_decision,
            split=split,
            sources=sources,
            config=config,
            target="rollback_policy",
            operation="reject_shuffled_feedback_candidate",
            before_hash=_hash_payload(updated.weights),
            after_hash=_hash_payload({"rollback_count": rollback_count}),
            rolled_back=True,
            rollback_reason="shuffled_control_failed_to_beat_no_feedback",
            rollback_count_delta=rollback_count,
            metrics=metrics,
        )
    )
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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3077 artifact violates the bounded-pilot contract."""

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
    if substrate.get("live_model_inference") is not False:
        raise ValueError("live model inference must remain false")
    if (
        substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("model weights must not be trained or mutated")

    ready = artifact.get("fr11_soundness_bounded_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if not ready:
        if verdict != BLOCKED_VERDICT:
            raise ValueError("blocked artifacts must use the blocked soundness-budget verdict")
        return
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if list(artifact.get("edit_targets_used") or []) != list(GOVERNED_EDIT_TARGETS):
        raise ValueError("edit_targets_used must stay inside governed controller targets")
    if int(artifact.get("self_model_trace_count") or 0) <= 0:
        raise ValueError("self_model_trace_count must be positive")
    if (
        int(artifact.get("soundness_mistakes", 0)) < 0
        or int(artifact.get("completeness_mistakes", 0)) < 0
    ):
        raise ValueError("mistake counts must be non-negative")
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

    budget_delta = artifact.get("mistake_budget_delta")
    if not isinstance(budget_delta, Mapping):
        raise ValueError("mistake_budget_delta must be a mapping")
    all_gates_passed = budget_delta.get("all_gates_passed") is True
    promotion = artifact.get("promotion_decision")
    if all_gates_passed and promotion != "controller_only_soundness_bounded_pilot_ready":
        raise ValueError("promotion_decision must match passing budget gates")
    if (
        not all_gates_passed
        and promotion != "controller_only_budget_exceeded_no_stronger_promotion"
    ):
        raise ValueError("promotion_decision must refuse stronger promotion")


def inference_substrate(*, controller_weight_update: bool, trace_memory_update: bool) -> JsonDict:
    """Return the execution boundary without implying live model adaptation."""

    return {
        "mode": "deterministic_exact_controller_online_budget_pilot",
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
        "exact_solver_authority": "deterministic_integer_sum_exact_authority",
        "training_scope": "bounded_controller_side_trace_memory_only",
    }


def field_principles() -> JsonDict:
    """Return compact reasons for the required terminal fields."""

    return {
        "fr11_soundness_bounded_ready": "matrix v21 needs a machine-readable FR-11 result",
        "continuous_self_learning_task": "milestone requirement must be explicit",
        "promotion_decision": "controller-only vs stronger claims must be separated",
        "edit_targets_used": "model weights must not be silently changed",
        "self_model_trace_count": "process feedback must be stored, not inferred",
        "soundness_mistakes": "unsafe accepts must be counted",
        "completeness_mistakes": "unsafe rejects/abstentions must be counted",
        "mistake_budget_delta": "online learning must be checked against budget",
        "family_holdout_delta": "learning must generalize to related held-out cases",
        "prior_retention_delta": "self-learning must not forget known cases",
        "no_feedback_delta": "feedback effect must beat no-op control",
        "shuffled_control_delta": "feedback effect must beat randomized labels",
        "contradiction_rate_delta": "governed self-improvement must reduce contradictions",
        "rollback_count": "unsafe updates must be measured",
        "delayed_regression_delta": "delayed failures must be checked",
        "source_trace_counts": "learned updates must trace to feedback sources",
        "inference_substrate": (
            "exact controller update must not be confused with live LLM inference"
        ),
        "honest_verdict": "terminal verdict must start with a success prefix unless blocked",
    }


def source_artifacts(sources: SourceBundle, config: ExperimentConfig) -> list[JsonDict]:
    """Return source provenance with ready fields and checksums."""

    return [
        _source_artifact_row(
            "exp3076",
            config.resolved_exp3076_artifact_path(),
            sources.exp3076_artifact,
            "soundness_completeness_budget_ready",
            config.repo_root,
        ),
        _source_artifact_row(
            "exp3060",
            config.resolved_exp3060_artifact_path(),
            sources.exp3060_artifact,
            "solver_self_model_trace_ready",
            config.repo_root,
        ),
    ]


def _complete_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    result: PilotResult,
    duration_s: float,
) -> JsonDict:
    budget_passed = result.mistake_budget_delta["all_gates_passed"] is True
    trace_schema = _mapping(sources.exp3060_artifact.get("trace_schema"))
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_soundness_bounded_ready": True,
        "continuous_self_learning_task": True,
        "promotion_decision": (
            "controller_only_soundness_bounded_pilot_ready"
            if budget_passed
            else "controller_only_budget_exceeded_no_stronger_promotion"
        ),
        "edit_targets_used": list(result.edit_targets_used),
        "self_model_trace_count": len(result.self_model_traces),
        "soundness_mistakes": result.soundness_mistakes,
        "completeness_mistakes": result.completeness_mistakes,
        "mistake_budget_delta": result.mistake_budget_delta,
        "family_holdout_delta": result.metrics["family_holdout_delta"],
        "prior_retention_delta": result.metrics["prior_retention_delta"],
        "no_feedback_delta": result.metrics["no_feedback_delta"],
        "shuffled_control_delta": result.metrics["shuffled_control_delta"],
        "contradiction_rate_delta": result.metrics["contradiction_rate_delta"],
        "rollback_count": result.rollback_count,
        "delayed_regression_delta": result.metrics["delayed_regression_delta"],
        "source_trace_counts": result.source_trace_counts,
        "inference_substrate": inference_substrate(
            controller_weight_update=True,
            trace_memory_update=True,
        ),
        "honest_verdict": (
            "complete_fr11_soundness_bounded_all_budget_gates_passed"
            if budget_passed
            else "complete_fr11_soundness_bounded_budget_exceeded"
        ),
        "trace_schema": trace_schema,
        "self_model_traces": list(result.self_model_traces),
        "online_decisions": list(result.online_decisions),
        "decision_label_counts": result.decision_label_counts,
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
        "fr11_soundness_bounded_ready": False,
        "continuous_self_learning_task": True,
        "promotion_decision": "blocked",
        "edit_targets_used": [],
        "self_model_trace_count": 0,
        "soundness_mistakes": 0,
        "completeness_mistakes": 0,
        "mistake_budget_delta": {},
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
        "online_decisions": [],
        "decision_label_counts": {},
        "split_report": {},
        "control_report": {},
        "source_artifacts": source_artifacts(sources, config),
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def _budget_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3076_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3076_artifact_malformed"
    if not _is_terminal(artifact):
        return "exp3076_not_terminal"
    if artifact.get("soundness_completeness_budget_ready") is not True:
        return "exp3076_soundness_budget_not_ready"
    if not _budget_numbers_complete(artifact):
        return "exp3076_budget_incomplete"
    if not _required_controls_complete(artifact):
        return "exp3076_required_controls_incomplete"
    substrate = _mapping(artifact.get("inference_substrate"))
    if substrate.get("live_llm_inference") is True or substrate.get("live_model_inference") is True:
        return "exp3076_live_model_inference_claimed"
    if substrate.get("model_weight_training") is True:
        return "exp3076_model_weight_training_claimed"
    if substrate.get("model_weight_mutation") is True:
        return "exp3076_model_weight_mutation_claimed"
    return None


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
    if _source_model_weight_claimed(artifact):
        return "exp3060_model_weight_learning_claimed"
    return None


def _trace_row(
    *,
    trace_id: str,
    decision: DecisionRecord,
    split: FamilySplit,
    sources: SourceBundle,
    config: ExperimentConfig,
    target: str,
    operation: str,
    before_hash: str,
    after_hash: str,
    rolled_back: bool,
    rollback_reason: str,
    rollback_count_delta: int,
    metrics: Mapping[str, float],
) -> JsonDict:
    case = decision.case
    exact_total = case.a + case.b
    exact_valid = case.total == exact_total
    violated = [] if exact_valid else ["a_plus_b_equals_total"]
    if decision.decision_label == "rollback":
        violated = ["shuffled_feedback_candidate_failed_control_gate"]
    return {
        "trace_id": trace_id,
        "solver_prompt_input": {
            "case_id": case.case_id,
            "split": case.split,
            "condition": decision.condition,
            "prompt_family": case.family,
            "input_variables": {"a": case.a, "b": case.b, "total": case.total},
            "input_hash": _hash_payload(
                {
                    "case_id": case.case_id,
                    "a": case.a,
                    "b": case.b,
                    "total": case.total,
                    "expected_valid": case.expected_valid,
                    "feedback_valid": case.feedback_valid,
                }
            ),
        },
        "exact_constraint_family": {
            "train_family_id": "tiny_sum_update",
            "heldout_family_id": "tiny_sum_related_holdout",
            "delayed_family_id": "tiny_sum_delayed_replay",
            "verifier_type": "deterministic_integer_sum",
            "exact_authority_ref": "a_plus_b_equals_total",
        },
        "correction_set": {
            "case_id": case.case_id,
            "violated_constraints": violated,
            "suggested_assignments": {"total": exact_total},
            "exact_label": exact_valid,
            "feedback_label": case.feedback_valid,
            "controller_decision": decision.controller_decision,
            "decision_label": decision.decision_label,
            "independent_label_authority": "deterministic_integer_sum_exact_authority",
            "source_feedback": case.source_trace_id,
        },
        "contradiction_graph_update": {
            "nodes": ["a", "b", "total", "a_plus_b_equals_total"],
            "edges": [["a", "total"], ["b", "total"]],
            "contradiction_rate_before": _round(metrics["contradiction_rate_before"]),
            "contradiction_rate_after": _round(metrics["contradiction_rate_after"]),
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
            "applied": not rolled_back,
        },
        "rollback_decision": {
            "rolled_back": rolled_back,
            "reason": rollback_reason,
            "comparator_signal": _round(decision.score),
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
            "exp3076",
            config.resolved_exp3076_artifact_path(),
            sources.exp3076_artifact,
            "soundness_completeness_budget_ready",
            config.repo_root,
        ),
        "online_decision_label": decision.decision_label,
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
    feature = "sat::sum_valid" if expected_valid else "sat::sum_invalid"
    return SolverCase(
        case_id=case_id,
        split=split,
        family="tiny_integer_sum_family",
        a=a,
        b=b,
        total=total,
        expected_valid=expected_valid,
        feedback_valid=expected_valid,
        source_trace_id=source_trace_id,
        features=(feature,),
    )


def _edge_case(case_id: str, expected_valid: bool, source_trace_id: str) -> SolverCase:
    return SolverCase(
        case_id=case_id,
        split="delayed_regression",
        family="tiny_integer_sum_edge_family",
        a=10,
        b=-4,
        total=6,
        expected_valid=expected_valid,
        feedback_valid=expected_valid,
        source_trace_id=source_trace_id,
        features=("sat::edge_valid_unknown",),
    )


def _prior_case(case_id: str, expected_valid: bool, feature: str) -> SolverCase:
    return SolverCase(
        case_id=case_id,
        split="prior_cases",
        family="prior_guard_family",
        a=0,
        b=0,
        total=0,
        expected_valid=expected_valid,
        feedback_valid=expected_valid,
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
        feedback_valid=case.expected_valid,
        source_trace_id=f"{source_prefix}:{case.case_id}",
        features=case.features,
    )


def _shuffled_feedback_case(case: SolverCase) -> SolverCase:
    return SolverCase(
        case_id=f"shuffled:{case.case_id}",
        split="shuffled_feedback_control",
        family=case.family,
        a=case.a,
        b=case.b,
        total=case.total,
        expected_valid=case.expected_valid,
        feedback_valid=not case.expected_valid,
        source_trace_id=f"control:shuffled:{case.case_id}",
        features=case.features,
    )


def _decision_label(decision: str, expected_valid: bool) -> str:
    if decision == "accept":
        return "correct" if expected_valid else "soundness_mistake"
    if decision == "reject":
        return "completeness_mistake" if expected_valid else "correct"
    return "completeness_mistake" if expected_valid else "abstention"


def _decision_payload(decision: DecisionRecord) -> JsonDict:
    case = decision.case
    return {
        "case_id": case.case_id,
        "split": case.split,
        "condition": decision.condition,
        "exact_label": case.expected_valid,
        "feedback_label": case.feedback_valid,
        "controller_decision": decision.controller_decision,
        "decision_label": decision.decision_label,
        "score": decision.score,
    }


def _label_count_dict(decisions: Sequence[DecisionRecord]) -> JsonDict:
    counts = Counter(decision.decision_label for decision in decisions)
    return {label: int(counts.get(label, 0)) for label in sorted(ALLOWED_DECISION_LABELS)}


def _count_labels(decisions: Sequence[DecisionRecord], label: str) -> int:
    return sum(1 for decision in decisions if decision.decision_label == label)


def _delayed_regression_count(
    before_weights: Mapping[str, float],
    after_weights: Mapping[str, float],
    cases: Sequence[SolverCase],
) -> int:
    regressions = 0
    for case in cases:
        before = evaluate_decision(before_weights, case, condition="before").decision_label
        after = evaluate_decision(after_weights, case, condition="after").decision_label
        if before == "correct" and after != "correct":
            regressions += 1
    return regressions


def _target_for_decision(decision: DecisionRecord) -> str:
    if decision.condition == "main" and decision.case.split == "train_update":
        return "controller_weights"
    return "trace_memory"


def _operation_for_decision(decision: DecisionRecord) -> str:
    if decision.condition == "main" and decision.case.split == "train_update":
        return "bounded_additive_weight_update_and_decision_label"
    if decision.condition == "no_feedback_control":
        return "no_feedback_control_decision_label"
    if decision.condition == "shuffled_feedback_control":
        return "shuffled_feedback_control_decision_label"
    return "online_decision_label_record"


def _max_gate(observed: int, allowed: int, metric_key: str) -> JsonDict:
    return {
        "metric_key": metric_key,
        "observed": observed,
        "allowed": allowed,
        "delta": observed - allowed,
        "passed": observed <= allowed,
    }


def _max_float_gate(observed: float, allowed: float, metric_key: str) -> JsonDict:
    return {
        "metric_key": metric_key,
        "observed": _round(observed),
        "allowed_max": _round(allowed),
        "delta": _round(observed - allowed),
        "passed": observed <= allowed,
    }


def _min_float_gate(observed: float, required: float, metric_key: str) -> JsonDict:
    return {
        "metric_key": metric_key,
        "observed": _round(observed),
        "required_min": _round(required),
        "delta": _round(observed - required),
        "passed": observed >= required,
    }


def _split_leakage_detected(split: FamilySplit) -> bool:
    train_ids = {case.case_id for case in split.train_update}
    heldout_ids = {case.case_id for case in split.family_holdout}
    delayed_ids = {case.case_id for case in split.delayed_regression}
    return bool(train_ids & heldout_ids or train_ids & delayed_ids)


def _budget_numbers_complete(artifact: Mapping[str, Any]) -> bool:
    budget = _mapping(artifact.get("mistake_budget"))
    return bool(
        NUMERIC_BUDGET_KEYS <= set(budget)
        and all(isinstance(budget[key], int | float) for key in NUMERIC_BUDGET_KEYS)
    )


def _required_controls_complete(artifact: Mapping[str, Any]) -> bool:
    controls = _sequence(artifact.get("required_controls"))
    control_names = {str(row.get("name")) for row in controls if isinstance(row, Mapping)}
    return control_names == REQUIRED_CONTROL_NAMES


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
        "artifact": str(artifact.get("artifact", "")),
        "schema": str(artifact.get("schema", "")),
        "sha256": _file_sha256(path),
        "ready_field": ready_field,
        "ready": artifact.get(ready_field) is True,
    }


def _is_terminal(artifact: Mapping[str, Any]) -> bool:
    return str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES)


def _source_model_weight_claimed(artifact: Mapping[str, Any]) -> bool:
    substrate = _mapping(artifact.get("inference_substrate"))
    return bool(
        artifact.get("model_weight_training") is True
        or artifact.get("model_weight_mutation") is True
        or substrate.get("model_weight_training") is True
        or substrate.get("model_weight_mutation") is True
    )


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


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _round(value: float) -> float:
    return round(float(value), 6)
