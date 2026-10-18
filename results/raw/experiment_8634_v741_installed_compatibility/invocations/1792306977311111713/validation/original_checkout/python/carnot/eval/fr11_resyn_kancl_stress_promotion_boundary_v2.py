"""Exp 3103 FR-11 ReSyn/KAN-CL stress promotion boundary.

This stress runner starts from the controller-only evidence that Exp 3090
reported as successful, then evaluates it against the stricter Exp 3097 exact
protocol. The point is not to train a stronger model. It is to make the
promotion boundary explicit: a controller update can be useful on the new
fixture protocol while still being unsafe to promote when it forgets prior
controller semantics or creates exact-protocol mistakes.

Spec refs: REQ-LEARN-3103, SCENARIO-LEARN-3103,
SCENARIO-LEARN-3103-BLOCKED.
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
RUN_DATE = "20260526"
ARTIFACT = "experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2"
SCHEMA = "carnot.fr11.resyn_kancl_stress_promotion_boundary.v2"
OUTPUT_REL_PATH = Path("results/experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2.json")
EXP3090_REL_PATH = Path("results/experiment_3090_fr11_resyn_kancl_completeness_repair_v1.json")
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
STRATIFIED_MANIFEST_REL_PATH = Path(
    "results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl"
)
BLOCKED_VERDICT = "blocked_precondition_failed"
BOUNDARY_VERDICT = "complete_fr11_stress_boundary_blocks_promotion"
CONTROLLER_VERDICT = "complete_fr11_stress_controller_only"
LEARNING_RATE = 0.45
MAX_ABS_WEIGHT = 1.0
ACCEPT_THRESHOLD = 0.25
REJECT_THRESHOLD = -0.25
PROMOTION_DECISIONS = {"controller_only", "blocked", "broader_promotion_candidate"}
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_stress_ready",
        "continuous_self_learning_task",
        "promotion_decision",
        "soundness_mistakes",
        "completeness_mistakes",
        "family_holdout_delta",
        "prior_retention_delta",
        "delayed_regression_delta",
        "rollback_count",
        "negative_control_results",
        "source_artifacts",
        "inference_substrate",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class StressConfig:
    """Paths and deterministic hooks for the offline stress replay."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3090_artifact_path: Path | None = None
    exp3097_artifact_path: Path | None = None
    protocol_manifest_path: Path | None = None
    started_s: float | None = None
    clock: ClockFn = time.perf_counter
    tests_run: Sequence[str] = ()

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def prior_artifact_path(self) -> Path:
        return self.exp3090_artifact_path or self.repo_root / EXP3090_REL_PATH

    def protocol_artifact_path(self) -> Path:
        return self.exp3097_artifact_path or self.repo_root / EXP3097_REL_PATH

    def manifest_path(self) -> Path:
        return self.protocol_manifest_path or self.repo_root / STRATIFIED_MANIFEST_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_s is None else float(self.started_s)


@dataclass(frozen=True)
class StressRow:
    """One protocol or prior-retention row with the target action to test."""

    fixture_id: str
    task_family: str
    task_axis: str
    perturbation_type: str
    label_source: str
    target_action: str
    target_source: str


@dataclass(frozen=True)
class ControllerState:
    """Inspectable controller state; no learned model weights live here."""

    weights: Mapping[str, float]
    trace_memory: tuple[str, ...] = ()
    model_weight_mutation: bool = False
    kan_model_weight_training: bool = False


@dataclass(frozen=True)
class PreconditionResult:
    """Loaded prior/protocol evidence and precondition diagnostics."""

    ok: bool
    checks: JsonDict
    exp3090_artifact: JsonDict
    exp3097_artifact: JsonDict
    protocol_rows: tuple[StressRow, ...]
    blocked_reason: str


@dataclass(frozen=True)
class StressSplit:
    """Disjoint stress partitions plus prior and negative-control probes."""

    train_update: tuple[StressRow, ...]
    family_holdout: tuple[StressRow, ...]
    prior_retention: tuple[StressRow, ...]
    delayed_regression: tuple[StressRow, ...]
    no_feedback_controls: tuple[StressRow, ...]
    shuffled_label_controls: tuple[StressRow, ...]


@dataclass(frozen=True)
class DecisionRecord:
    """One controller decision compared against an exact target action."""

    fixture_id: str
    target_source: str
    perturbation_type: str
    controller_decision: str
    target_action: str
    decision_label: str
    score: float


@dataclass(frozen=True)
class StressResult:
    """Metrics, controls, and promotion boundary from the stress replay."""

    prior_state: ControllerState
    candidate_state: ControllerState
    metrics: JsonDict
    stress_decisions: tuple[DecisionRecord, ...]
    negative_control_results: JsonDict
    split_report: JsonDict
    label_counts: JsonDict
    promotion_decision: str
    rollback_count: int
    soundness_mistakes: int
    completeness_mistakes: int


def write_artifact(config: StressConfig | None = None) -> JsonDict:
    """Run the stress replay, validate the result, and persist the artifact."""

    active = config or StressConfig()
    started = active.start_time()
    preconditions = load_preconditions(active)
    duration_s = _round(active.clock() - started)
    if not preconditions.ok:
        artifact = blocked_artifact(active, preconditions, duration_s)
        validate_artifact(artifact)
        write_json(active.artifact_path(), artifact)
        return artifact
    split = build_stress_split(preconditions)
    result = run_stress_replay(preconditions, split)
    artifact = complete_artifact(active, preconditions, result, duration_s)
    validate_artifact(artifact)
    write_json(active.artifact_path(), artifact)
    return artifact


def load_preconditions(config: StressConfig) -> PreconditionResult:
    """Load Exp 3090 prior evidence and the Exp 3097 stress protocol."""

    exp3090 = safe_load_json(config.prior_artifact_path())
    exp3097 = safe_load_json(config.protocol_artifact_path())
    raw_rows = safe_load_jsonl(config.manifest_path())
    protocol_rows = tuple(protocol_row(row) for row in raw_rows if row_is_fr11_stress_ready(row))
    checks = {
        "exp3090_artifact_ready": {
            "ok": exp3090.get("fr11_resyn_kancl_ready") is True,
            "path": relative_path(config.repo_root, config.prior_artifact_path()),
        },
        "exp3097_artifact_ready": {
            "ok": exp3097.get("eval_protocol_ready") is True,
            "path": relative_path(config.repo_root, config.protocol_artifact_path()),
        },
        "protocol_manifest_exists": {
            "ok": config.manifest_path().is_file(),
            "path": relative_path(config.repo_root, config.manifest_path()),
        },
        "fr11_stress_rows_available": {
            "ok": len(protocol_rows) >= 48,
            "observed": len(protocol_rows),
            "required_min": 48,
        },
        "protocol_targets_supported": {
            "ok": bool(protocol_rows)
            and {row.target_action for row in protocol_rows} <= {"accept", "reject"},
            "observed": sorted({row.target_action for row in protocol_rows}),
        },
    }
    ok = all(item["ok"] is True for item in checks.values())
    blocked_reason = "" if ok else first_failed_precondition(checks)
    return PreconditionResult(
        ok=ok,
        checks=checks,
        exp3090_artifact=exp3090,
        exp3097_artifact=exp3097,
        protocol_rows=protocol_rows if ok else (),
        blocked_reason=blocked_reason,
    )


def row_is_fr11_stress_ready(row: Mapping[str, Any]) -> bool:
    """Return whether an Exp 3097 row is assigned to the FR-11 stress task."""

    return "fr11_stress_v2" in set(row.get("evaluation_tasks", ()))


def protocol_row(row: Mapping[str, Any]) -> StressRow:
    """Convert one Exp 3097 manifest row into a stress replay row."""

    target = str(row.get("verifier_target", {}).get("expected_action"))
    return StressRow(
        fixture_id=str(row["source_fixture_id"]),
        task_family=str(row["task_family"]),
        task_axis=str(row["task_axis"]),
        perturbation_type=str(row["perturbation_type"]),
        label_source=str(row["label_source"]),
        target_action=target,
        target_source="exp3097_protocol",
    )


def first_failed_precondition(checks: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the first failed precondition key, or a stable fallback."""

    failed_key = next(
        (key for key, value in checks.items() if value.get("ok") is not True),
        "unknown_precondition_failure",
    )
    return {
        "exp3090_artifact_ready": "exp3090_artifact_missing_or_empty",
        "exp3097_artifact_ready": "exp3097_artifact_missing_or_empty",
        "protocol_manifest_exists": "exp3097_manifest_missing",
    }.get(failed_key, failed_key)


def build_stress_split(preconditions: PreconditionResult) -> StressSplit:
    """Build update, harder holdout, prior, delayed, and control partitions."""

    by_perturbation: dict[str, list[StressRow]] = defaultdict(list)
    for row in preconditions.protocol_rows:
        by_perturbation[row.perturbation_type].append(row)
    train = []
    holdout = []
    delayed = []
    for perturbation in sorted(by_perturbation):
        rows = by_perturbation[perturbation]
        train.extend(rows[:2])
        delayed.extend(rows[-1:])
        holdout.extend(rows[2:-1])
    prior = build_prior_retention_rows(preconditions.exp3090_artifact, preconditions.protocol_rows)
    split = StressSplit(
        train_update=tuple(train),
        family_holdout=tuple(holdout),
        prior_retention=prior,
        delayed_regression=tuple(delayed),
        no_feedback_controls=tuple(holdout),
        shuffled_label_controls=shuffled_label_rows(train),
    )
    if not all(
        (
            split.train_update,
            split.family_holdout,
            split.prior_retention,
            split.delayed_regression,
            split.no_feedback_controls,
            split.shuffled_label_controls,
        )
    ):
        raise ValueError("stress split missing required partition")
    return split


def build_prior_retention_rows(
    exp3090_artifact: Mapping[str, Any],
    protocol_rows: Sequence[StressRow],
) -> tuple[StressRow, ...]:
    """Map Exp 3090 online decisions onto Exp 3097 rows with prior targets."""

    protocol_by_id = {row.fixture_id: row for row in protocol_rows}
    prior_rows = []
    seen = set()
    for decision in exp3090_artifact.get("online_decisions", ()):
        fixture_id = str(decision.get("fixture_id"))
        if (
            decision.get("condition") == "online_update"
            and fixture_id in protocol_by_id
            and fixture_id not in seen
        ):
            target = "accept" if bool(decision.get("exact_accept")) else "reject"
            prior_rows.append(with_target(protocol_by_id[fixture_id], target, "exp3090_prior"))
            seen.add(fixture_id)
    return tuple(prior_rows)


def with_target(row: StressRow, target_action: str, target_source: str) -> StressRow:
    """Return the same row under a different exact target semantics."""

    return StressRow(
        fixture_id=row.fixture_id,
        task_family=row.task_family,
        task_axis=row.task_axis,
        perturbation_type=row.perturbation_type,
        label_source=row.label_source,
        target_action=target_action,
        target_source=target_source,
    )


def reconstruct_prior_controller(exp3090_artifact: Mapping[str, Any]) -> ControllerState:
    """Rebuild the controller weights materialized by Exp 3090 anchors."""

    weights: dict[str, float] = {}
    trace_memory = []
    for anchor in exp3090_artifact.get("kancl_anchors", ()):
        trace_memory.append(str(anchor.get("anchor_id", "")))
        for feature, weight in anchor.get("constraint_local_basis_weights", {}).items():
            weights[str(feature)] = float(weight)
    return ControllerState(weights=dict(sorted(weights.items())), trace_memory=tuple(trace_memory))


def apply_protocol_feedback(
    state: ControllerState,
    rows: Sequence[StressRow],
) -> ControllerState:
    """Apply bounded controller-side feedback for protocol target actions."""

    weights = dict(state.weights)
    trace_memory = list(state.trace_memory)
    for row in rows:
        direction = 1.0 if row.target_action == "accept" else -1.0
        for feature in anchor_features(row):
            weights[feature] = _clamp(float(weights.get(feature, 0.0)) + LEARNING_RATE * direction)
        trace_memory.append(f"{row.fixture_id}:{row.target_source}:{row.target_action}")
    return ControllerState(weights=dict(sorted(weights.items())), trace_memory=tuple(trace_memory))


def shuffled_label_rows(rows: Sequence[StressRow]) -> tuple[StressRow, ...]:
    """Return negative-control rows with exact labels intentionally flipped."""

    return tuple(
        with_target(
            row,
            "reject" if row.target_action == "accept" else "accept",
            "shuffled_label_negative_control",
        )
        for row in rows
    )


def anchor_features(row: StressRow) -> tuple[str, ...]:
    """Return the same local controller basis used by the Exp 3090 anchors."""

    return (
        f"family:{row.task_family}",
        f"task_axis:{row.task_axis}",
        f"perturbation:{row.perturbation_type}",
        f"label_source:{row.label_source}",
    )


def run_stress_replay(preconditions: PreconditionResult, split: StressSplit) -> StressResult:
    """Replay the controller update and measure the promotion boundary."""

    prior_state = reconstruct_prior_controller(preconditions.exp3090_artifact)
    candidate_state = apply_protocol_feedback(prior_state, split.train_update)
    shuffled_state = apply_protocol_feedback(prior_state, split.shuffled_label_controls)
    stress_decisions = evaluate_rows(candidate_state, preconditions.protocol_rows)
    stress_counts = count_labels(stress_decisions)
    baseline_counts = count_labels(evaluate_rows(prior_state, preconditions.protocol_rows))
    shuffled_counts = count_labels(evaluate_rows(shuffled_state, split.family_holdout))
    no_feedback_counts = count_labels(evaluate_rows(prior_state, split.no_feedback_controls))
    holdout_before = accuracy(prior_state, split.family_holdout)
    holdout_after = accuracy(candidate_state, split.family_holdout)
    no_feedback_holdout = accuracy(prior_state, split.no_feedback_controls)
    shuffled_holdout = accuracy(shuffled_state, split.family_holdout)
    prior_before = accuracy(prior_state, split.prior_retention)
    prior_after = accuracy(candidate_state, split.prior_retention)
    delayed_before = accuracy(prior_state, split.delayed_regression)
    delayed_after = accuracy(candidate_state, split.delayed_regression)
    metrics = {
        "family_holdout_delta": _round(holdout_after - holdout_before),
        "prior_retention_delta": _round(prior_after - prior_before),
        "delayed_regression_delta": _round(delayed_after - delayed_before),
        "no_feedback_family_holdout_delta": _round(no_feedback_holdout - holdout_before),
        "shuffled_label_family_holdout_delta": _round(shuffled_holdout - holdout_before),
    }
    negative_controls = {
        "no_feedback": {
            "case_count": len(split.no_feedback_controls),
            "family_holdout_delta": metrics["no_feedback_family_holdout_delta"],
            "soundness_mistakes": int(no_feedback_counts.get("soundness_mistake", 0)),
            "completeness_mistakes": int(no_feedback_counts.get("completeness_mistake", 0)),
            "rolled_back": False,
        },
        "shuffled_label": {
            "case_count": len(split.shuffled_label_controls),
            "family_holdout_delta": metrics["shuffled_label_family_holdout_delta"],
            "soundness_mistakes": int(shuffled_counts.get("soundness_mistake", 0)),
            "completeness_mistakes": int(shuffled_counts.get("completeness_mistake", 0)),
            "rolled_back": bool(
                shuffled_counts.get("soundness_mistake", 0)
                or shuffled_counts.get("completeness_mistake", 0)
                or metrics["shuffled_label_family_holdout_delta"] > 0.0
            ),
        },
    }
    gates_passed = promotion_gates_passed(stress_counts, metrics, negative_controls)
    main_rolled_back = not gates_passed
    rollback_count = int(main_rolled_back) + int(negative_controls["shuffled_label"]["rolled_back"])
    split_report = {
        "train_update_count": len(split.train_update),
        "family_holdout_count": len(split.family_holdout),
        "prior_retention_count": len(split.prior_retention),
        "delayed_regression_count": len(split.delayed_regression),
        "no_feedback_control_count": len(split.no_feedback_controls),
        "shuffled_label_control_count": len(split.shuffled_label_controls),
        "harder_holdout_perturbation_count": len(
            {row.perturbation_type for row in split.family_holdout}
        ),
        "candidate_update_rolled_back": main_rolled_back,
    }
    return StressResult(
        prior_state=prior_state,
        candidate_state=candidate_state,
        metrics=metrics,
        stress_decisions=stress_decisions,
        negative_control_results=negative_controls,
        split_report=split_report,
        label_counts={
            "baseline_protocol": dict(sorted(baseline_counts.items())),
            "candidate_protocol": dict(sorted(stress_counts.items())),
            "shuffled_holdout": dict(sorted(shuffled_counts.items())),
            "no_feedback_holdout": dict(sorted(no_feedback_counts.items())),
        },
        promotion_decision="controller_only" if gates_passed else "blocked",
        rollback_count=rollback_count,
        soundness_mistakes=int(stress_counts.get("soundness_mistake", 0)),
        completeness_mistakes=int(stress_counts.get("completeness_mistake", 0)),
    )


def promotion_gates_passed(
    stress_counts: Mapping[str, int],
    metrics: Mapping[str, float],
    negative_controls: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether the measured stress boundaries allow controller-only promotion."""

    return (
        int(stress_counts.get("soundness_mistake", 0)) == 0
        and int(stress_counts.get("completeness_mistake", 0)) == 0
        and float(metrics["family_holdout_delta"]) > 0.0
        and float(metrics["prior_retention_delta"]) >= 0.0
        and float(metrics["delayed_regression_delta"]) >= 0.0
        and float(negative_controls["no_feedback"]["family_holdout_delta"]) <= 0.0
        and float(negative_controls["shuffled_label"]["family_holdout_delta"]) <= 0.0
    )


def evaluate_rows(
    state: ControllerState,
    rows: Sequence[StressRow],
) -> tuple[DecisionRecord, ...]:
    """Evaluate a batch of stress rows against exact target actions."""

    return tuple(evaluate_row(state, row) for row in rows)


def evaluate_row(state: ControllerState, row: StressRow) -> DecisionRecord:
    """Score one row and classify the controller decision."""

    row_score = score(state, row)
    decision = decision_from_score(row_score)
    return DecisionRecord(
        fixture_id=row.fixture_id,
        target_source=row.target_source,
        perturbation_type=row.perturbation_type,
        controller_decision=decision,
        target_action=row.target_action,
        decision_label=decision_label(decision, row.target_action),
        score=row_score,
    )


def score(state: ControllerState, row: StressRow) -> float:
    """Return the controller score from local basis weights."""

    return _round(sum(float(state.weights.get(feature, 0.0)) for feature in anchor_features(row)))


def decision_from_score(row_score: float) -> str:
    """Map a controller score into accept, reject, or abstain."""

    if row_score > ACCEPT_THRESHOLD:
        return "accept"
    if row_score < REJECT_THRESHOLD:
        return "reject"
    return "abstain"


def decision_label(decision: str, target_action: str) -> str:
    """Classify a controller decision as a soundness or completeness outcome."""

    if target_action not in {"accept", "reject"}:
        raise ValueError(f"unsupported target action: {target_action}")
    if decision == target_action:
        return "correct"
    if decision == "accept" and target_action == "reject":
        return "soundness_mistake"
    if target_action == "accept":
        return "completeness_mistake"
    return "abstention"


def accuracy(state: ControllerState, rows: Sequence[StressRow]) -> float:
    """Return exact target-action accuracy, counting abstention as incorrect."""

    if not rows:
        return 0.0
    labels = count_labels(evaluate_rows(state, rows))
    return _round(int(labels.get("correct", 0)) / len(rows))


def count_labels(records: Sequence[DecisionRecord]) -> dict[str, int]:
    """Return deterministic decision-label counts."""

    return dict(sorted(Counter(record.decision_label for record in records).items()))


def complete_artifact(
    config: StressConfig,
    preconditions: PreconditionResult,
    result: StressResult,
    duration_s: float,
) -> JsonDict:
    """Build the complete terminal stress artifact."""

    promotion_decision = result.promotion_decision
    honest_verdict = CONTROLLER_VERDICT if promotion_decision == "controller_only" else BOUNDARY_VERDICT
    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_stress_ready": True,
        "continuous_self_learning_task": True,
        "promotion_decision": promotion_decision,
        "soundness_mistakes": result.soundness_mistakes,
        "completeness_mistakes": result.completeness_mistakes,
        "family_holdout_delta": result.metrics["family_holdout_delta"],
        "prior_retention_delta": result.metrics["prior_retention_delta"],
        "delayed_regression_delta": result.metrics["delayed_regression_delta"],
        "rollback_count": result.rollback_count,
        "negative_control_results": result.negative_control_results,
        "stress_split_report": result.split_report,
        "stress_decision_label_counts": result.label_counts,
        "online_decisions": [_decision_payload(record) for record in result.stress_decisions],
        "preconditions_checked": preconditions.checks,
        "source_artifacts": source_artifacts(config),
        "inference_substrate": inference_substrate(controller_update=True),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
        "honest_verdict": honest_verdict,
    }


def blocked_artifact(
    config: StressConfig,
    preconditions: PreconditionResult,
    duration_s: float,
) -> JsonDict:
    """Build the fail-closed artifact when prior/protocol evidence is missing."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_stress_ready": False,
        "continuous_self_learning_task": True,
        "promotion_decision": "blocked",
        "soundness_mistakes": 0,
        "completeness_mistakes": 0,
        "family_holdout_delta": 0.0,
        "prior_retention_delta": 0.0,
        "delayed_regression_delta": 0.0,
        "rollback_count": 0,
        "negative_control_results": {},
        "stress_split_report": {},
        "stress_decision_label_counts": {},
        "online_decisions": [],
        "preconditions_checked": preconditions.checks,
        "source_artifacts": source_artifacts(config),
        "inference_substrate": inference_substrate(controller_update=False),
        "tests_run": list(config.tests_run),
        "blocked_reason": preconditions.blocked_reason,
        "duration_s": duration_s,
        "honest_verdict": BLOCKED_VERDICT,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact violates the stress-boundary contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_task") is not True:
        raise ValueError("continuous_self_learning_task must be true")
    if artifact.get("promotion_decision") not in PROMOTION_DECISIONS:
        raise ValueError("promotion_decision must be explicit")
    if not artifact.get("source_artifacts"):
        raise ValueError("source_artifacts must trace prior controller evidence")
    substrate = dict(artifact.get("inference_substrate", {}))
    if (
        substrate.get("live_llm_inference") is not False
        or substrate.get("live_model_inference") is not False
        or substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
        or substrate.get("base_model_weights_updated") is not False
        or substrate.get("kan_model_weight_training") is not False
    ):
        raise ValueError("model weights and live inference must remain untouched")
    ready = artifact.get("fr11_stress_ready") is True
    if not ready:
        if artifact.get("honest_verdict") != BLOCKED_VERDICT:
            raise ValueError("blocked stress artifacts must use the blocked verdict")
        return
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if artifact.get("promotion_decision") == "controller_only" and (
        int(artifact.get("soundness_mistakes", -1)) != 0
        or int(artifact.get("completeness_mistakes", -1)) != 0
        or float(artifact.get("family_holdout_delta", 0.0)) <= 0.0
        or float(artifact.get("prior_retention_delta", -1.0)) < 0.0
        or float(artifact.get("delayed_regression_delta", -1.0)) < 0.0
    ):
        raise ValueError("controller_only promotion requires all stress gates to pass")


def inference_substrate(*, controller_update: bool) -> JsonDict:
    """Declare the deterministic controller-only replay substrate."""

    return {
        "mode": "deterministic_fr11_stress_controller_replay",
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


def source_artifacts(config: StressConfig) -> list[JsonDict]:
    """Return source provenance for the prior controller and protocol evidence."""

    return [
        source_artifact_row(
            "exp3090_prior_fr11",
            config.prior_artifact_path(),
            config.repo_root,
            ".288 FR-11 controller-only ReSyn/KAN-CL evidence",
        ),
        source_artifact_row(
            "exp3097_protocol",
            config.protocol_artifact_path(),
            config.repo_root,
            ".289 exact fixture evaluation protocol artifact",
        ),
        source_artifact_row(
            "exp3097_manifest",
            config.manifest_path(),
            config.repo_root,
            ".289 stratified exact fixture manifest for FR-11 stress",
        ),
    ]


def source_artifact_row(source_id: str, path: Path, repo_root: Path, role: str) -> JsonDict:
    """Return one source-artifact row with stable relative path and hash."""

    exists = path.is_file()
    return {
        "id": source_id,
        "path": relative_path(repo_root, path),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "role": role,
    }


def _decision_payload(record: DecisionRecord) -> JsonDict:
    """Return a JSON-serializable decision record."""

    return {
        "fixture_id": record.fixture_id,
        "target_source": record.target_source,
        "perturbation_type": record.perturbation_type,
        "controller_decision": record.controller_decision,
        "target_action": record.target_action,
        "decision_label": record.decision_label,
        "score": record.score,
    }


def safe_load_json(path: Path) -> JsonDict:
    """Load a JSON object, returning an empty object on missing or malformed input."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def safe_load_jsonl(path: Path) -> tuple[JsonDict, ...]:
    """Load JSONL rows, returning an empty tuple on missing or malformed input."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return tuple(json.loads(line) for line in lines if line.strip())
    except (OSError, json.JSONDecodeError):
        return ()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable pretty JSON for the terminal artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for source-artifact provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return a readable repository-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _clamp(value: float) -> float:
    """Clamp a controller feature weight to the bounded online range."""

    return _round(max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, value)))


def _round(value: float) -> float:
    """Round metrics so artifacts remain stable across Python versions."""

    return round(float(value), 6)
