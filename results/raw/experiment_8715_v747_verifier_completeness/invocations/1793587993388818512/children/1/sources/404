"""Exp 3090 ReSyn KAN-CL completeness repair for FR-11.

This pilot is a controller-side learning experiment over the exact Exp 3084
fixture bank. The controller starts conservative: it rejects known invalid
families but abstains on valid and repairable rows, which creates completeness
mistakes without soundness mistakes. Online exact feedback updates only small,
inspectable controller anchors. The KAN-CL inspiration is deliberately scoped to
per-family knots and local basis weights; no KAN model weights, LLM weights, or
other learned model parameters are trained or mutated.

Spec refs: REQ-LEARN-3090, SCENARIO-LEARN-3090,
SCENARIO-LEARN-3090-BLOCKED.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3090_fr11_resyn_kancl_completeness_repair_v1"
SCHEMA = "carnot.fr11.resyn_kancl_completeness_repair.v1"
OUTPUT_REL_PATH = Path(f"results/{ARTIFACT}.json")
EXP3084_ARTIFACT_REL_PATH = Path(
    "results/experiment_3084_resyn_exact_fixture_bank_generator_v1.json"
)
EXP3084_MANIFEST_REL_PATH = Path("results/resyn_exact_fixture_bank_3084/fixture_manifest.jsonl")
BLOCKED_VERDICT = "blocked_fixture_precondition_failed"
SUCCESS_VERDICT = "complete_fr11_resyn_kancl_controller_only_ready"
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
LEARNING_RATE = 0.45
MAX_ABS_WEIGHT = 1.0
ACCEPT_THRESHOLD = 0.25
REJECT_THRESHOLD = -0.25
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_resyn_kancl_ready",
        "continuous_self_learning_task",
        "promotion_decision",
        "soundness_mistakes",
        "completeness_mistakes",
        "family_holdout_delta",
        "prior_retention_delta",
        "no_feedback_control_delta",
        "shuffled_feedback_control_delta",
        "kancl_anchor_count",
        "rollback_count",
        "delayed_regression_delta",
        "preconditions_checked",
        "inference_substrate",
        "honest_verdict",
    }
)
REQUIRED_PRECONDITION_KEYS = frozenset(
    {
        "exp3084_artifact_ready",
        "fixture_manifest_exists",
        "fixture_count",
        "exact_labels_available",
        "delayed_regression_labels_available",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock injection for the deterministic pilot."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3084_artifact_path: Path | None = None
    manifest_path: Path | None = None
    started_at: float | None = None
    clock: ClockFn = time.perf_counter
    tests_run: Sequence[str] = ()

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else float(self.started_at)

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def resolved_exp3084_artifact_path(self) -> Path:
        return self.exp3084_artifact_path or self.repo_root / EXP3084_ARTIFACT_REL_PATH

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / EXP3084_MANIFEST_REL_PATH


@dataclass(frozen=True)
class PreconditionResult:
    """Fixture-bank preconditions and loaded rows for the pilot."""

    ok: bool
    checks: JsonDict
    rows: tuple[JsonDict, ...]
    source_artifact: JsonDict
    blocked_reason: str


@dataclass(frozen=True)
class FixtureSplit:
    """Online, control, holdout, prior, and delayed-regression rows."""

    train_update: tuple[JsonDict, ...]
    family_holdout: tuple[JsonDict, ...]
    prior_cases: tuple[JsonDict, ...]
    delayed_regression: tuple[JsonDict, ...]
    no_feedback_controls: tuple[JsonDict, ...]
    shuffled_feedback_controls: tuple[JsonDict, ...]


@dataclass(frozen=True)
class ControllerState:
    """All mutable learning state is controller-side and inspectable."""

    weights: Mapping[str, float]
    trace_memory: tuple[str, ...] = ()
    model_weight_mutation: bool = False
    kan_model_weight_training: bool = False


@dataclass(frozen=True)
class DecisionRecord:
    """One controller decision compared against exact fixture authority."""

    fixture_id: str
    condition: str
    controller_decision: str
    decision_label: str
    score: float
    exact_accept: bool


@dataclass(frozen=True)
class PolicyResult:
    """Metrics and audit state from the bounded controller update."""

    updated_state: ControllerState
    anchors: tuple[JsonDict, ...]
    decisions: tuple[DecisionRecord, ...]
    metrics: JsonDict
    control_report: JsonDict
    split_report: JsonDict
    source_trace_counts: JsonDict
    budget_gates: JsonDict
    soundness_mistakes: int
    completeness_mistakes: int
    baseline_soundness_mistakes: int
    baseline_completeness_mistakes: int
    rollback_count: int
    kancl_anchor_count: int


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Run the preconditioned Exp 3090 pilot and write its terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    preconditions = check_preconditions(active)
    duration_s = _round(active.clock() - started)
    if not preconditions.ok:
        artifact = blocked_artifact(active, preconditions, duration_s)
        validate_artifact(artifact)
        _write_json(active.resolved_output_path(), artifact)
        return artifact
    split = build_fixture_split(preconditions.rows)
    result = run_online_policy(split)
    artifact = complete_artifact(active, preconditions, result, duration_s)
    validate_artifact(artifact)
    _write_json(active.resolved_output_path(), artifact)
    return artifact


def check_preconditions(config: ExperimentConfig) -> PreconditionResult:
    """Confirm Exp 3084 and exact delayed-regression labels before any update."""

    artifact_path = config.resolved_exp3084_artifact_path()
    manifest_path = config.resolved_manifest_path()
    source_artifact = _read_json(artifact_path)
    rows = _read_jsonl(manifest_path)
    artifact_ready = bool(
        source_artifact
        and source_artifact.get("resyn_fixture_bank_ready") is True
        and source_artifact.get("fixture_manifest_path")
    )
    manifest_exists = manifest_path.is_file()
    expected_count = int(source_artifact.get("exact_fixture_count", 0) or 0)
    fixture_count_ok = bool(rows and len(rows) == expected_count)
    exact_labels_ok = bool(rows and all(_row_has_supported_exact_label(row) for row in rows))
    delayed_rows = _delayed_regression_candidates(rows)
    delayed_labels_ok = bool(
        len(delayed_rows) >= 4 and all(_row_has_supported_exact_label(row) for row in delayed_rows)
    )
    checks = {
        "exp3084_artifact_ready": {
            "ok": artifact_ready,
            "path": _relative_path(config.repo_root, artifact_path),
        },
        "fixture_manifest_exists": {
            "ok": manifest_exists,
            "path": _relative_path(config.repo_root, manifest_path),
        },
        "fixture_count": {
            "ok": fixture_count_ok,
            "observed": len(rows),
            "expected": expected_count,
        },
        "exact_labels_available": {
            "ok": exact_labels_ok,
            "observed": sum(1 for row in rows if _row_has_supported_exact_label(row)),
        },
        "delayed_regression_labels_available": {
            "ok": delayed_labels_ok,
            "observed": len(delayed_rows),
        },
    }
    ok = all(_mapping(item).get("ok") is True for item in checks.values())
    return PreconditionResult(
        ok=ok,
        checks=checks,
        rows=rows if ok else (),
        source_artifact=source_artifact,
        blocked_reason="" if ok else _first_failed_precondition(checks),
    )


def build_fixture_split(rows: Sequence[Mapping[str, Any]]) -> FixtureSplit:
    """Build a deterministic split over Exp 3084 perturbation families."""

    by_perturbation: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        by_perturbation[str(row["perturbation_family"])].append(dict(row))
    train = (
        *_take(by_perturbation, "smt_sat_solving", 2, 0),
        *_take(by_perturbation, "smt_unsat_abstention", 2, 0),
        *_take(by_perturbation, "arithmetic_true_verification", 2, 0),
        *_take(by_perturbation, "arithmetic_false_verification", 2, 0),
        *_take(by_perturbation, "json_syntax_repair", 2, 0),
        *_take(by_perturbation, "numeric_bound_repair", 2, 0),
    )
    holdout = tuple(by_perturbation.get("python_assertion_repair", ()))
    prior = (
        *_take(by_perturbation, "smt_unsat_abstention", 4, 2),
        *_take(by_perturbation, "arithmetic_false_verification", 4, 2),
    )
    delayed = (
        *_take(by_perturbation, "smt_sat_solving", 2, 2),
        *_take(by_perturbation, "smt_unsat_abstention", 2, 6),
        *_take(by_perturbation, "arithmetic_true_verification", 2, 2),
        *_take(by_perturbation, "arithmetic_false_verification", 2, 6),
    )
    no_feedback = tuple(dict(row) for row in holdout)
    shuffled = tuple(_with_feedback_override(row, not exact_accept_label(row)) for row in train)
    split = FixtureSplit(
        train_update=tuple(train),
        family_holdout=holdout,
        prior_cases=tuple(prior),
        delayed_regression=tuple(delayed),
        no_feedback_controls=no_feedback,
        shuffled_feedback_controls=shuffled,
    )
    if not all(
        (
            split.train_update,
            split.family_holdout,
            split.prior_cases,
            split.delayed_regression,
            split.no_feedback_controls,
            split.shuffled_feedback_controls,
        )
    ):
        raise ValueError("fixture split missing required partition")
    return split


def run_online_policy(split: FixtureSplit) -> PolicyResult:
    """Apply exact online feedback, controls, rollback, and retention checks."""

    baseline = initial_controller_state()
    updated = apply_online_feedback(baseline, split.train_update)
    shuffled_candidate = apply_online_feedback(baseline, split.shuffled_feedback_controls)
    baseline_main = evaluate_cases(
        baseline,
        split.train_update + split.family_holdout + split.prior_cases + split.delayed_regression,
        condition="baseline",
    )
    online_main = evaluate_cases(
        updated,
        split.train_update + split.family_holdout + split.prior_cases + split.delayed_regression,
        condition="online_update",
    )
    no_feedback = evaluate_cases(baseline, split.no_feedback_controls, condition="no_feedback")
    shuffled = evaluate_cases(
        shuffled_candidate,
        split.family_holdout,
        condition="shuffled_feedback_control",
    )
    prior_before = accuracy(baseline, split.prior_cases)
    prior_after = accuracy(updated, split.prior_cases)
    delayed_before = accuracy(baseline, split.delayed_regression)
    delayed_after = accuracy(updated, split.delayed_regression)
    baseline_holdout = accuracy(baseline, split.family_holdout)
    online_holdout = accuracy(updated, split.family_holdout)
    no_feedback_holdout = accuracy(baseline, split.no_feedback_controls)
    shuffled_holdout = accuracy(shuffled_candidate, split.family_holdout)
    rollback_count = 1 if shuffled_holdout <= no_feedback_holdout else 0
    metrics = {
        "family_holdout_delta": _round(online_holdout - baseline_holdout),
        "prior_retention_delta": _round(prior_after - prior_before),
        "no_feedback_control_delta": _round(no_feedback_holdout - baseline_holdout),
        "shuffled_feedback_control_delta": _round(shuffled_holdout - baseline_holdout),
        "delayed_regression_delta": _round(delayed_after - delayed_before),
        "contradiction_rate_delta": _round(
            contradiction_rate(updated, split.train_update + split.family_holdout)
            - contradiction_rate(baseline, split.train_update + split.family_holdout)
        ),
    }
    soundness_mistakes = _count_label(online_main, "soundness_mistake")
    completeness_mistakes = _count_label(online_main, "completeness_mistake")
    anchors = build_kancl_anchors(updated, split.train_update)
    budget_gates = {
        "soundness_mistakes": {
            "observed": soundness_mistakes,
            "allowed": 0,
            "passed": soundness_mistakes == 0,
        },
        "completeness_mistakes": {
            "observed": completeness_mistakes,
            "allowed": 0,
            "passed": completeness_mistakes == 0,
        },
        "prior_retention_delta": {
            "observed": metrics["prior_retention_delta"],
            "required_min": 0.0,
            "passed": metrics["prior_retention_delta"] >= 0.0,
        },
        "controls_non_vacuous": {
            "observed": bool(split.no_feedback_controls and split.shuffled_feedback_controls),
            "required": True,
            "passed": bool(split.no_feedback_controls and split.shuffled_feedback_controls),
        },
    }
    budget_gates["all_gates_passed"] = all(
        _mapping(row).get("passed") is True
        for row in budget_gates.values()
        if isinstance(row, Mapping)
    )
    decisions = baseline_main + online_main + no_feedback + shuffled
    return PolicyResult(
        updated_state=updated,
        anchors=anchors,
        decisions=decisions,
        metrics=metrics,
        control_report={
            "non_vacuous_controls": bool(
                split.no_feedback_controls and split.shuffled_feedback_controls
            ),
            "no_feedback_case_count": len(split.no_feedback_controls),
            "shuffled_case_count": len(split.shuffled_feedback_controls),
            "shuffled_candidate_rolled_back": rollback_count > 0,
        },
        split_report={
            "train_update_count": len(split.train_update),
            "family_holdout_count": len(split.family_holdout),
            "prior_case_count": len(split.prior_cases),
            "delayed_regression_count": len(split.delayed_regression),
            "family_holdout_perturbation_family": "python_assertion_repair",
        },
        source_trace_counts={
            "train_update_count": len(split.train_update),
            "family_holdout_count": len(split.family_holdout),
            "prior_case_count": len(split.prior_cases),
            "delayed_regression_count": len(split.delayed_regression),
            "no_feedback_control_count": len(split.no_feedback_controls),
            "shuffled_feedback_control_count": len(split.shuffled_feedback_controls),
            "online_decision_count": len(decisions),
        },
        budget_gates=budget_gates,
        soundness_mistakes=soundness_mistakes,
        completeness_mistakes=completeness_mistakes,
        baseline_soundness_mistakes=_count_label(baseline_main, "soundness_mistake"),
        baseline_completeness_mistakes=_count_label(baseline_main, "completeness_mistake"),
        rollback_count=rollback_count,
        kancl_anchor_count=len(anchors),
    )


def initial_controller_state() -> ControllerState:
    """Return the conservative exact-negative baseline controller."""

    return ControllerState(
        weights={
            "perturbation:arithmetic_false_verification": -0.75,
            "perturbation:smt_unsat_abstention": -0.75,
        }
    )


def apply_online_feedback(
    state: ControllerState,
    rows: Sequence[Mapping[str, Any]],
) -> ControllerState:
    """Update bounded controller anchors from exact or control feedback."""

    weights = dict(state.weights)
    trace_memory = list(state.trace_memory)
    for row in rows:
        direction = 1.0 if feedback_accept_label(row) else -1.0
        for feature in anchor_features(row):
            weights[feature] = _clamp(weights.get(feature, 0.0) + LEARNING_RATE * direction)
        trace_memory.append(
            f"{row['fixture_id']}:{row['perturbation_family']}:{feedback_accept_label(row)}"
        )
    return ControllerState(weights=dict(sorted(weights.items())), trace_memory=tuple(trace_memory))


def evaluate_cases(
    state: ControllerState,
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str = "metric",
) -> tuple[DecisionRecord, ...]:
    """Evaluate rows with the current controller and exact authority."""

    return tuple(evaluate_case(state, row, condition=condition) for row in rows)


def evaluate_case(
    state: ControllerState,
    row: Mapping[str, Any],
    *,
    condition: str,
) -> DecisionRecord:
    """Evaluate one row and classify the controller mistake type."""

    row_score = score(state, row)
    if row_score > ACCEPT_THRESHOLD:
        decision = "accept"
    elif row_score < REJECT_THRESHOLD:
        decision = "reject"
    else:
        decision = "abstain"
    exact_accept = exact_accept_label(row)
    return DecisionRecord(
        fixture_id=str(row["fixture_id"]),
        condition=condition,
        controller_decision=decision,
        decision_label=decision_label(decision, exact_accept),
        score=row_score,
        exact_accept=exact_accept,
    )


def score(state: ControllerState, row: Mapping[str, Any]) -> float:
    """Return the controller score from local anchor basis features."""

    return _round(sum(float(state.weights.get(feature, 0.0)) for feature in anchor_features(row)))


def accuracy(state: ControllerState, rows: Sequence[Mapping[str, Any]]) -> float:
    """Return exact decision accuracy, counting abstention as incorrect."""

    if not rows:
        return 0.0
    correct = _count_label(evaluate_cases(state, rows), "correct")
    return _round(correct / len(rows))


def contradiction_rate(state: ControllerState, rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the soundness-mistake rate over exact-rejected rows."""

    exact_rejected = tuple(row for row in rows if not exact_accept_label(row))
    if not exact_rejected:
        return 0.0
    mistakes = _count_label(evaluate_cases(state, exact_rejected), "soundness_mistake")
    return _round(mistakes / len(exact_rejected))


def exact_accept_label(row: Mapping[str, Any]) -> bool:
    """Map each exact fixture label schema to the controller accept target."""

    label = _mapping(row.get("exact_label"))
    kind = str(label.get("kind"))
    if kind == "smt_satisfiability":
        return bool(label.get("is_satisfiable"))
    if kind == "arithmetic_assertion":
        return bool(label.get("assertion_passes"))
    if kind == "repairability":
        return bool(label.get("repairable"))
    raise ValueError(f"unsupported exact label kind: {kind}")


def feedback_accept_label(row: Mapping[str, Any]) -> bool:
    """Return exact feedback unless a control row intentionally shuffles it."""

    if "_feedback_accept_override" in row:
        return bool(row["_feedback_accept_override"])
    return exact_accept_label(row)


def decision_label(decision: str, exact_accept: bool) -> str:
    """Classify a controller decision against exact authority."""

    if decision == "accept":
        return "correct" if exact_accept else "soundness_mistake"
    if exact_accept:
        return "completeness_mistake"
    return "correct" if decision == "reject" else "abstention"


def anchor_features(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return local basis features for controller-side KAN-CL-style anchors."""

    return (
        f"family:{row['family']}",
        f"task_axis:{row['task_axis']}",
        f"perturbation:{row['perturbation_family']}",
        f"label_source:{row['label_source']}",
    )


def build_kancl_anchors(
    state: ControllerState,
    train_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, ...]:
    """Materialize concrete per-family knots and local basis weights."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[(str(row["family"]), str(row["perturbation_family"]))].append(row)
    anchors = []
    max_index = max((_fixture_numeric_index(row) for row in train_rows), default=1)
    for (family, perturbation), rows in sorted(grouped.items()):
        knot = _round(
            sum(_fixture_numeric_index(row) for row in rows) / max(1, len(rows)) / max(1, max_index)
        )
        features = sorted({feature for row in rows for feature in anchor_features(row)})
        basis_weights = {
            feature: _round(float(state.weights.get(feature, 0.0)))
            for feature in features
            if abs(float(state.weights.get(feature, 0.0))) > 0.0
        }
        anchors.append(
            {
                "anchor_id": f"kancl:{family}:{perturbation}",
                "family": family,
                "perturbation_family": perturbation,
                "family_knot": knot,
                "constraint_local_basis_weights": basis_weights,
                "source_fixture_ids": [str(row["fixture_id"]) for row in rows],
            }
        )
    return tuple(anchors)


def controller_has_model_weight_mutation(state: ControllerState) -> bool:
    """Return whether a state claims any forbidden model-weight mutation."""

    return bool(state.model_weight_mutation or state.kan_model_weight_training)


def count_decision_labels(records: Sequence[DecisionRecord]) -> dict[str, int]:
    """Return a deterministic label-count dictionary."""

    return dict(sorted(Counter(record.decision_label for record in records).items()))


def complete_artifact(
    config: ExperimentConfig,
    preconditions: PreconditionResult,
    result: PolicyResult,
    duration_s: float,
) -> JsonDict:
    """Build the successful Exp 3090 terminal artifact."""

    source_counts = dict(result.source_trace_counts)
    source_counts["fixture_count"] = len(preconditions.rows)
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_resyn_kancl_ready": True,
        "continuous_self_learning_task": True,
        "promotion_decision": "controller_only_resyn_kancl_budget_passed",
        "soundness_mistakes": result.soundness_mistakes,
        "completeness_mistakes": result.completeness_mistakes,
        "baseline_soundness_mistakes": result.baseline_soundness_mistakes,
        "baseline_completeness_mistakes": result.baseline_completeness_mistakes,
        "family_holdout_delta": result.metrics["family_holdout_delta"],
        "prior_retention_delta": result.metrics["prior_retention_delta"],
        "no_feedback_control_delta": result.metrics["no_feedback_control_delta"],
        "shuffled_feedback_control_delta": result.metrics["shuffled_feedback_control_delta"],
        "kancl_anchor_count": result.kancl_anchor_count,
        "rollback_count": result.rollback_count,
        "delayed_regression_delta": result.metrics["delayed_regression_delta"],
        "contradiction_rate_delta": result.metrics["contradiction_rate_delta"],
        "preconditions_checked": preconditions.checks,
        "inference_substrate": inference_substrate(controller_update=True),
        "kancl_anchors": list(result.anchors),
        "online_decisions": [_decision_payload(record) for record in result.decisions],
        "control_report": result.control_report,
        "split_report": result.split_report,
        "source_trace_counts": source_counts,
        "budget_gates": result.budget_gates,
        "source_artifacts": [source_artifact_row(config)],
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
        "honest_verdict": SUCCESS_VERDICT,
    }


def blocked_artifact(
    config: ExperimentConfig,
    preconditions: PreconditionResult,
    duration_s: float,
) -> JsonDict:
    """Build the fail-closed artifact when fixture evidence is unavailable."""

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_resyn_kancl_ready": False,
        "continuous_self_learning_task": True,
        "promotion_decision": BLOCKED_VERDICT,
        "soundness_mistakes": 0,
        "completeness_mistakes": 0,
        "baseline_soundness_mistakes": 0,
        "baseline_completeness_mistakes": 0,
        "family_holdout_delta": 0.0,
        "prior_retention_delta": 0.0,
        "no_feedback_control_delta": 0.0,
        "shuffled_feedback_control_delta": 0.0,
        "kancl_anchor_count": 0,
        "rollback_count": 0,
        "delayed_regression_delta": 0.0,
        "contradiction_rate_delta": 0.0,
        "preconditions_checked": preconditions.checks,
        "inference_substrate": inference_substrate(controller_update=False),
        "kancl_anchors": [],
        "online_decisions": [],
        "control_report": {"non_vacuous_controls": False, "shuffled_candidate_rolled_back": False},
        "split_report": {},
        "source_trace_counts": {},
        "budget_gates": {"all_gates_passed": False},
        "source_artifacts": [source_artifact_row(config)],
        "tests_run": list(config.tests_run),
        "blocked_reason": preconditions.blocked_reason,
        "duration_s": duration_s,
        "honest_verdict": BLOCKED_VERDICT,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate readiness, safety, and controller-only claim boundaries."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_task") is not True:
        raise ValueError("continuous_self_learning_task must be true")
    substrate = _mapping(artifact.get("inference_substrate"))
    if (
        substrate.get("live_llm_inference") is not False
        or substrate.get("live_model_inference") is not False
        or substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("model weights and live inference must remain untouched")
    ready = artifact.get("fr11_resyn_kancl_ready") is True
    if not ready:
        if artifact.get("honest_verdict") != BLOCKED_VERDICT:
            raise ValueError("blocked artifacts must use the blocked verdict")
        return
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if int(artifact.get("soundness_mistakes", -1)) != 0:
        raise ValueError("soundness_mistakes must be zero")
    if int(artifact.get("completeness_mistakes", -1)) != 0:
        raise ValueError("completeness_mistakes must be zero")
    if float(artifact.get("family_holdout_delta", 0.0)) <= 0.0:
        raise ValueError("family_holdout_delta must be positive")
    if float(artifact.get("prior_retention_delta", -1.0)) < 0.0:
        raise ValueError("prior_retention_delta must be nonnegative")
    if float(artifact.get("no_feedback_control_delta", 1.0)) > 0.0:
        raise ValueError("no_feedback_control_delta must not explain the gain")
    if float(artifact.get("shuffled_feedback_control_delta", 1.0)) > 0.0:
        raise ValueError("shuffled_feedback_control_delta must not explain the gain")
    anchors = _sequence(artifact.get("kancl_anchors"))
    if int(artifact.get("kancl_anchor_count", 0)) <= 0 or len(anchors) != artifact.get(
        "kancl_anchor_count"
    ):
        raise ValueError("kancl_anchor_count must match concrete anchors")
    if int(artifact.get("rollback_count", 0)) <= 0:
        raise ValueError("rollback_count must be positive")
    if float(artifact.get("delayed_regression_delta", -1.0)) < 0.0:
        raise ValueError("delayed_regression_delta must be measured")
    preconditions = _mapping(artifact.get("preconditions_checked"))
    if REQUIRED_PRECONDITION_KEYS - set(preconditions) or not all(
        _mapping(row).get("ok") is True for row in preconditions.values()
    ):
        raise ValueError("preconditions_checked must all pass")
    gates = _mapping(artifact.get("budget_gates"))
    if (
        gates.get("all_gates_passed") is not True
        and artifact.get("promotion_decision") == "controller_only_resyn_kancl_budget_passed"
    ):
        raise ValueError("promotion_decision requires all budget gates to pass")


def inference_substrate(*, controller_update: bool) -> JsonDict:
    """Declare the exact CPU fixture replay substrate and forbidden claims."""

    return {
        "mode": "deterministic_resyn_fixture_controller_replay",
        "cached_artifacts_only": True,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "controller_weight_update": bool(controller_update),
        "trace_memory_update": bool(controller_update),
        "training_scope": "bounded_controller_side_anchor_memory_only",
    }


def source_artifact_row(config: ExperimentConfig) -> JsonDict:
    """Return source provenance for Exp 3084 fixture evidence."""

    path = config.resolved_exp3084_artifact_path()
    return {
        "id": "exp3084",
        "path": _relative_path(config.repo_root, path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "role": "ReSyn exact fixture bank with delayed-regression labels",
    }


def sha256_file(path: Path) -> str:
    """Return a stable file digest for artifact provenance."""

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _decision_payload(record: DecisionRecord) -> JsonDict:
    return {
        "fixture_id": record.fixture_id,
        "condition": record.condition,
        "controller_decision": record.controller_decision,
        "decision_label": record.decision_label,
        "score": record.score,
        "exact_accept": record.exact_accept,
    }


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive corruption path.
        return {"_malformed": str(exc)}


def _read_jsonl(path: Path) -> tuple[JsonDict, ...]:
    if not path.is_file():
        return ()
    try:
        return tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive corruption path.
        return ({"_malformed": str(exc)},)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_has_supported_exact_label(row: Mapping[str, Any]) -> bool:
    try:
        exact_accept_label(row)
        return True
    except (TypeError, ValueError):  # pragma: no cover - malformed fixture path.
        return False


def _delayed_regression_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    delayed_families = {
        "smt_sat_solving",
        "smt_unsat_abstention",
        "arithmetic_true_verification",
        "arithmetic_false_verification",
    }
    return tuple(row for row in rows if row.get("perturbation_family") in delayed_families)


def _first_failed_precondition(checks: Mapping[str, Any]) -> str:
    for key, value in checks.items():
        if _mapping(value).get("ok") is not True:
            return key
    return "unknown_precondition_failure"


def _take(
    by_perturbation: Mapping[str, Sequence[JsonDict]],
    name: str,
    count: int,
    offset: int,
) -> tuple[JsonDict, ...]:
    return tuple(dict(row) for row in by_perturbation.get(name, ())[offset : offset + count])


def _with_feedback_override(row: Mapping[str, Any], feedback_accept: bool) -> JsonDict:
    return dict(row) | {"_feedback_accept_override": bool(feedback_accept)}


def _fixture_numeric_index(row: Mapping[str, Any]) -> int:
    suffix = str(row.get("fixture_id", "0")).rsplit("-", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _count_label(records: Sequence[DecisionRecord], label: str) -> int:
    return sum(1 for record in records if record.decision_label == label)


def _clamp(value: float) -> float:
    return _round(max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, value)))


def _round(value: float) -> float:
    return round(float(value), 6)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
