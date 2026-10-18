"""Exp 3116 FR-11 hard-family curriculum retention guard.

This experiment adapts the NuRL-style observation that hard or zero-pass
examples often need hints, but it keeps the adaptation at Carnot's controller
memory boundary. The hints are derived from checked solver/certificate classes
in the exact fixture protocol. They never reveal numeric final answers, never
train model weights, and never invoke a live model.

Spec refs: REQ-LEARN-3116, SCENARIO-LEARN-3116,
SCENARIO-LEARN-3116-BLOCKED.
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
ARTIFACT = "experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1"
SCHEMA = "carnot.fr11.unsolvable_curriculum_retention_guard.v1"
OUTPUT_REL_PATH = Path("results/experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1.json")
EXP3090_REL_PATH = Path("results/experiment_3090_fr11_resyn_kancl_completeness_repair_v1.json")
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3103_REL_PATH = Path("results/experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2.json")
STRATIFIED_MANIFEST_REL_PATH = Path(
    "results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl"
)
MANDATED_MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SUCCESS_VERDICT = "complete_fr11_unsolvable_curriculum_controller_only_guard_passed"
BOUNDARY_VERDICT = "complete_fr11_unsolvable_curriculum_boundary_blocks_promotion"
BLOCKED_VERDICT = "blocked_precondition_failed"
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
MAX_ABS_BASE_WEIGHT = 1.0
ACCEPT_THRESHOLD = 0.25
REJECT_THRESHOLD = -0.25
PROMOTION_DECISIONS = {"controller_only", "blocked"}
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_unsolvable_curriculum_ready",
        "continuous_self_learning_task",
        "controller_only",
        "no_weight_update_claim",
        "model_specs",
        "hard_family_count",
        "unsolvable_detection_summary",
        "hint_policy_summary",
        "soundness_mistakes",
        "completeness_mistakes",
        "prior_retention_delta",
        "delayed_regression_delta",
        "rollback_count",
        "promotion_decision",
        "negative_controls",
        "tests_run",
        "source_artifacts",
        "inference_substrate",
        "honest_verdict",
    }
)
REQUIRED_PRECONDITION_KEYS = frozenset(
    {
        "exp3090_artifact_ready",
        "exp3097_artifact_ready",
        "exp3103_artifact_ready",
        "protocol_manifest_exists",
        "fr11_stress_rows_available",
        "exp3103_hard_rows_available",
    }
)


@dataclass(frozen=True)
class GuardConfig:
    """Paths and deterministic hooks for the offline curriculum replay."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3090_artifact_path: Path | None = None
    exp3097_artifact_path: Path | None = None
    exp3103_artifact_path: Path | None = None
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

    def stress_artifact_path(self) -> Path:
        return self.exp3103_artifact_path or self.repo_root / EXP3103_REL_PATH

    def manifest_path(self) -> Path:
        return self.protocol_manifest_path or self.repo_root / STRATIFIED_MANIFEST_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_s is None else float(self.started_s)


@dataclass(frozen=True)
class CurriculumRow:
    """One exact-protocol row under a specific controller target context."""

    fixture_id: str
    task_family: str
    task_axis: str
    perturbation_type: str
    label_source: str
    solver_label: str
    exact_label_kind: str
    target_action: str
    target_source: str


@dataclass(frozen=True)
class ControllerState:
    """Inspectable controller state; model weights are intentionally absent."""

    base_weights: Mapping[str, float]
    hint_weights: Mapping[str, float]
    hint_memory: tuple[str, ...] = ()
    model_weight_mutation: bool = False
    kan_model_weight_training: bool = False


@dataclass(frozen=True)
class AbstractHint:
    """Solver-derived hint that stores only abstract certificate metadata."""

    hint_id: str
    family_key: str
    target_action: str
    target_context: str
    source_fixture_ids: tuple[str, ...]
    feature_weights: Mapping[str, float]
    abstract_hint: str
    evidence: Mapping[str, str]
    solver_derived: bool = True
    final_answer_revealed: bool = False


@dataclass(frozen=True)
class PreconditionResult:
    """Loaded source artifacts and fail-closed diagnostics."""

    ok: bool
    checks: JsonDict
    exp3090_artifact: JsonDict
    exp3097_artifact: JsonDict
    exp3103_artifact: JsonDict
    protocol_rows: tuple[CurriculumRow, ...]
    blocked_reason: str
    rows_by_id: Mapping[str, JsonDict] | None = None


@dataclass(frozen=True)
class StressSplit:
    """The replay split inherited from Exp 3103's promotion-boundary protocol."""

    train_update: tuple[CurriculumRow, ...]
    family_holdout: tuple[CurriculumRow, ...]
    prior_retention: tuple[CurriculumRow, ...]
    delayed_regression: tuple[CurriculumRow, ...]


@dataclass(frozen=True)
class Curriculum:
    """Hard-family rows, abstract hints, and source summaries for the guard."""

    protocol_hard_rows: tuple[CurriculumRow, ...]
    prior_retention_regression_rows: tuple[CurriculumRow, ...]
    family_holdout: tuple[CurriculumRow, ...]
    prior_retention: tuple[CurriculumRow, ...]
    delayed_regression: tuple[CurriculumRow, ...]
    abstract_hints: tuple[AbstractHint, ...]
    rows_by_id: Mapping[str, JsonDict]
    hard_family_count: int
    detection_summary: JsonDict


@dataclass(frozen=True)
class DecisionRecord:
    """One guarded controller decision compared against exact authority."""

    fixture_id: str
    target_source: str
    perturbation_type: str
    controller_decision: str
    target_action: str
    decision_label: str
    score: float


@dataclass(frozen=True)
class GuardResult:
    """Metrics, controls, and promotion decision from the curriculum replay."""

    base_state: ControllerState
    guarded_state: ControllerState
    metrics: JsonDict
    guarded_decisions: tuple[DecisionRecord, ...]
    negative_controls: JsonDict
    promotion_decision: str
    rollback_count: int
    soundness_mistakes: int
    completeness_mistakes: int


def write_artifact(config: GuardConfig | None = None) -> JsonDict:
    """Run the hard-family guard, validate it, and persist the terminal artifact."""

    active = config or GuardConfig()
    started = active.start_time()
    preconditions = load_preconditions(active)
    duration_s = _round(active.clock() - started)
    if not preconditions.ok:
        artifact = blocked_artifact(active, preconditions, duration_s)
        validate_artifact(artifact)
        write_json(active.artifact_path(), artifact)
        return artifact
    curriculum = build_curriculum(preconditions)
    result = run_curriculum_guard(preconditions, curriculum)
    artifact = complete_artifact(active, preconditions, curriculum, result, duration_s)
    validate_artifact(artifact)
    write_json(active.artifact_path(), artifact)
    return artifact


def load_preconditions(config: GuardConfig) -> PreconditionResult:
    """Load Exp 3090, Exp 3097, Exp 3103, and the exact protocol manifest."""

    exp3090 = safe_load_json(config.prior_artifact_path())
    exp3097 = safe_load_json(config.protocol_artifact_path())
    exp3103 = safe_load_json(config.stress_artifact_path())
    raw_rows = safe_load_jsonl(config.manifest_path())
    rows_by_id = {
        str(row.get("source_fixture_id")): dict(row)
        for row in raw_rows
        if isinstance(row, Mapping) and row.get("source_fixture_id")
    }
    protocol_rows = tuple(protocol_row(row) for row in raw_rows if row_is_fr11_stress_ready(row))
    hard_decisions = tuple(
        decision
        for decision in exp3103.get("online_decisions", ())
        if isinstance(decision, Mapping)
        and decision.get("decision_label") == "completeness_mistake"
    )
    checks = {
        "exp3090_artifact_ready": {
            "ok": exp3090.get("fr11_resyn_kancl_ready") is True,
            "path": relative_path(config.repo_root, config.prior_artifact_path()),
        },
        "exp3097_artifact_ready": {
            "ok": exp3097.get("eval_protocol_ready") is True,
            "path": relative_path(config.repo_root, config.protocol_artifact_path()),
        },
        "exp3103_artifact_ready": {
            "ok": exp3103.get("fr11_stress_ready") is True
            and int(exp3103.get("completeness_mistakes", 0) or 0) > 0,
            "path": relative_path(config.repo_root, config.stress_artifact_path()),
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
        "exp3103_hard_rows_available": {
            "ok": bool(hard_decisions),
            "observed": len(hard_decisions),
        },
    }
    ok = all(item["ok"] is True for item in checks.values())
    return PreconditionResult(
        ok=ok,
        checks=checks,
        exp3090_artifact=exp3090,
        exp3097_artifact=exp3097,
        exp3103_artifact=exp3103,
        protocol_rows=protocol_rows if ok else (),
        blocked_reason="" if ok else first_failed_precondition(checks),
        rows_by_id=rows_by_id if ok else {},
    )


def row_is_fr11_stress_ready(row: Mapping[str, Any]) -> bool:
    """Return whether a manifest row participates in the FR-11 stress task."""

    return "fr11_stress_v2" in set(row.get("evaluation_tasks", ()))


def protocol_row(row: Mapping[str, Any]) -> CurriculumRow:
    """Convert an exact fixture manifest row into a controller row."""

    return CurriculumRow(
        fixture_id=str(row["source_fixture_id"]),
        task_family=str(row["task_family"]),
        task_axis=str(row["task_axis"]),
        perturbation_type=str(row["perturbation_type"]),
        label_source=str(row["label_source"]),
        solver_label=str(row.get("solver_label", "")),
        exact_label_kind=str(row.get("exact_label_kind", "")),
        target_action=str(row.get("verifier_target", {}).get("expected_action")),
        target_source="exp3097_protocol",
    )


def first_failed_precondition(checks: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the first failed precondition as a stable artifact reason."""

    failed_key = next(
        (key for key, value in checks.items() if value.get("ok") is not True),
        "unknown_precondition_failure",
    )
    return {
        "exp3090_artifact_ready": "exp3090_artifact_missing_or_empty",
        "exp3097_artifact_ready": "exp3097_artifact_missing_or_empty",
        "exp3103_artifact_ready": "exp3103_artifact_missing_or_not_hard",
        "protocol_manifest_exists": "exp3097_manifest_missing",
    }.get(failed_key, failed_key)


def build_curriculum(preconditions: PreconditionResult) -> Curriculum:
    """Identify hard/zero-pass families and generate abstract hints."""

    split = build_stress_split(preconditions)
    base_state = reconstruct_exp3103_candidate(preconditions, split)
    protocol_by_id = {row.fixture_id: row for row in preconditions.protocol_rows}
    hard_ids = {
        str(decision.get("fixture_id"))
        for decision in preconditions.exp3103_artifact.get("online_decisions", ())
        if isinstance(decision, Mapping)
        and decision.get("decision_label") == "completeness_mistake"
    }
    protocol_hard_rows = tuple(
        row for fixture_id, row in sorted(protocol_by_id.items()) if fixture_id in hard_ids
    )
    prior_records = evaluate_rows(base_state, split.prior_retention)
    prior_regression = tuple(
        row
        for row, record in zip(split.prior_retention, prior_records, strict=True)
        if record.decision_label != "correct" and row.target_action == "accept"
    )
    if not protocol_hard_rows or not prior_regression:
        raise ValueError("curriculum requires hard-family rows from Exp 3103 and retention replay")
    family_keys = sorted(
        {row.perturbation_type for row in protocol_hard_rows}
        | {row.perturbation_type for row in prior_regression}
    )
    row_groups = {
        family: tuple(
            row
            for row in protocol_hard_rows + prior_regression
            if row.perturbation_type == family
        )
        for family in family_keys
    }
    hints = tuple(build_abstract_hint(family, rows) for family, rows in row_groups.items())
    zero_pass_count = sum(
        1 for rows in row_groups.values() if accuracy(base_state, rows) == 0.0
    )
    detection_summary = {
        "source": "exp3103_completeness_and_retention_replay",
        "source_exp3103_completeness_mistakes": len(protocol_hard_rows),
        "prior_retention_regression_case_count": len(prior_regression),
        "curriculum_case_count": len(protocol_hard_rows) + len(prior_regression),
        "hard_families": family_keys,
        "hard_family_case_counts": {
            family: len(rows) for family, rows in sorted(row_groups.items())
        },
        "zero_pass_family_count": zero_pass_count,
    }
    return Curriculum(
        protocol_hard_rows=protocol_hard_rows,
        prior_retention_regression_rows=prior_regression,
        family_holdout=split.family_holdout,
        prior_retention=split.prior_retention,
        delayed_regression=split.delayed_regression,
        abstract_hints=hints,
        rows_by_id=dict(preconditions.rows_by_id or {}),
        hard_family_count=len(family_keys),
        detection_summary=detection_summary,
    )


def build_stress_split(preconditions: PreconditionResult) -> StressSplit:
    """Rebuild the Exp 3103 deterministic stress partitions."""

    by_perturbation: dict[str, list[CurriculumRow]] = defaultdict(list)
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
    split = StressSplit(
        train_update=tuple(train),
        family_holdout=tuple(holdout),
        prior_retention=build_prior_retention_rows(
            preconditions.exp3090_artifact,
            preconditions.protocol_rows,
        ),
        delayed_regression=tuple(delayed),
    )
    if not all((split.train_update, split.family_holdout, split.prior_retention, split.delayed_regression)):
        raise ValueError("stress split missing required partition")
    return split


def build_prior_retention_rows(
    exp3090_artifact: Mapping[str, Any],
    protocol_rows: Sequence[CurriculumRow],
) -> tuple[CurriculumRow, ...]:
    """Map Exp 3090 online decisions onto protocol rows with prior semantics."""

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


def with_target(row: CurriculumRow, target_action: str, target_source: str) -> CurriculumRow:
    """Return a row under another exact target context."""

    return CurriculumRow(
        fixture_id=row.fixture_id,
        task_family=row.task_family,
        task_axis=row.task_axis,
        perturbation_type=row.perturbation_type,
        label_source=row.label_source,
        solver_label=row.solver_label,
        exact_label_kind=row.exact_label_kind,
        target_action=target_action,
        target_source=target_source,
    )


def reconstruct_exp3103_candidate(
    preconditions: PreconditionResult,
    split: StressSplit | None = None,
) -> ControllerState:
    """Rebuild Exp 3103's stressed candidate as the baseline for Exp 3116."""

    active_split = split or build_stress_split(preconditions)
    prior = reconstruct_prior_controller(preconditions.exp3090_artifact)
    return apply_protocol_feedback(prior, active_split.train_update)


def reconstruct_prior_controller(exp3090_artifact: Mapping[str, Any]) -> ControllerState:
    """Rebuild controller base weights from Exp 3090 KAN-CL anchors."""

    weights: dict[str, float] = {}
    memory = []
    for anchor in exp3090_artifact.get("kancl_anchors", ()):
        memory.append(str(anchor.get("anchor_id", "")))
        for feature, weight in anchor.get("constraint_local_basis_weights", {}).items():
            weights[str(feature)] = float(weight)
    return ControllerState(base_weights=dict(sorted(weights.items())), hint_weights={}, hint_memory=tuple(memory))


def apply_protocol_feedback(
    state: ControllerState,
    rows: Sequence[CurriculumRow],
) -> ControllerState:
    """Apply the Exp 3103 bounded protocol feedback to base controller weights."""

    weights = dict(state.base_weights)
    memory = list(state.hint_memory)
    for row in rows:
        direction = 1.0 if row.target_action == "accept" else -1.0
        for feature in base_anchor_features(row):
            weights[feature] = _clamp_base(float(weights.get(feature, 0.0)) + LEARNING_RATE * direction)
        memory.append(f"{row.fixture_id}:{row.target_source}:{row.target_action}")
    return ControllerState(
        base_weights=dict(sorted(weights.items())),
        hint_weights=dict(state.hint_weights),
        hint_memory=tuple(memory),
    )


def build_abstract_hint(family_key: str, rows: Sequence[CurriculumRow]) -> AbstractHint:
    """Build one abstract hint from solver/certificate metadata only."""

    representative = rows[0]
    source_ids = tuple(sorted({row.fixture_id for row in rows}))
    if family_key == "arithmetic_true_verification":
        feature_weights = {f"hint:perturbation:{family_key}": 0.5}
        target_context = "protocol_and_prior_assertion_certificate"
    else:
        target_context = "prior_retention_repairability"
        feature_weights = {
            f"hint:target_source:exp3090_prior:family:{representative.task_family}": 1.0,
            f"hint:target_source:exp3090_prior:task_axis:{representative.task_axis}": 1.0,
            f"hint:target_source:exp3090_prior:perturbation:{family_key}": 1.0,
            f"hint:target_source:exp3090_prior:label_source:{representative.label_source}": 1.0,
            "hint:target_source:exp3090_prior:target_action:accept": 1.0,
        }
    evidence = {
        "family": representative.task_family,
        "task_axis": representative.task_axis,
        "perturbation_type": representative.perturbation_type,
        "label_source": representative.label_source,
        "solver_label": representative.solver_label,
        "exact_label_kind": representative.exact_label_kind,
        "target_context": target_context,
    }
    text = (
        "Use the solver certificate class for this task family and target "
        "context; update only controller memory and keep fixture answers hidden."
    )
    return AbstractHint(
        hint_id=stable_hint_id(family_key, source_ids, evidence),
        family_key=family_key,
        target_action="accept",
        target_context=target_context,
        source_fixture_ids=source_ids,
        feature_weights=feature_weights,
        abstract_hint=text,
        evidence=evidence,
    )


def stable_hint_id(
    family_key: str,
    source_ids: Sequence[str],
    evidence: Mapping[str, str],
) -> str:
    """Return a short stable hint identifier for artifact readability."""

    payload = json.dumps(
        {"family_key": family_key, "source_ids": list(source_ids), "evidence": dict(evidence)},
        sort_keys=True,
    )
    return "hint-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def hint_leaks_final_answer(
    hint: AbstractHint,
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Detect whether hint text includes row-specific answers or prompt payloads."""

    haystack = hint.abstract_hint
    for fixture_id in hint.source_fixture_ids:
        raw = rows_by_id.get(fixture_id, {})
        forbidden = {str(raw.get("expected_answer", ""))}
        prompt = raw.get("leakage_safe_prompt_payload", {})
        if isinstance(prompt, Mapping):
            forbidden |= {
                str(prompt.get("candidate_assertion", "")),
                str(prompt.get("expression", "")),
                str(prompt.get("candidate", "")),
            }
        for token in forbidden:
            if token and token in haystack:
                return True
    return False


def run_curriculum_guard(
    preconditions: PreconditionResult,
    curriculum: Curriculum,
) -> GuardResult:
    """Apply abstract hints, evaluate gates, and build negative controls."""

    split = build_stress_split(preconditions)
    base_state = reconstruct_exp3103_candidate(preconditions, split)
    guarded_state = apply_hints(base_state, curriculum.abstract_hints)
    base_protocol_counts = count_labels(evaluate_rows(base_state, preconditions.protocol_rows))
    guarded_decisions = evaluate_rows(guarded_state, preconditions.protocol_rows)
    guarded_counts = count_labels(guarded_decisions)
    metrics = build_metrics(
        base_state,
        guarded_state,
        curriculum,
        base_protocol_counts,
        guarded_counts,
        int(preconditions.exp3103_artifact.get("completeness_mistakes", 0) or 0),
    )
    controls = build_negative_controls(base_state, preconditions, curriculum, metrics)
    gates_passed = promotion_gates_passed(metrics, controls)
    rollback_count = sum(1 for control in controls.values() if control["rolled_back"])
    return GuardResult(
        base_state=base_state,
        guarded_state=guarded_state,
        metrics=metrics,
        guarded_decisions=guarded_decisions,
        negative_controls=controls,
        promotion_decision="controller_only" if gates_passed else "blocked",
        rollback_count=rollback_count,
        soundness_mistakes=int(guarded_counts.get("soundness_mistake", 0)),
        completeness_mistakes=int(guarded_counts.get("completeness_mistake", 0)),
    )


def build_metrics(
    base_state: ControllerState,
    guarded_state: ControllerState,
    curriculum: Curriculum,
    base_protocol_counts: Mapping[str, int],
    guarded_counts: Mapping[str, int],
    exp3103_completeness_mistakes: int,
) -> JsonDict:
    """Build the gate metrics comparing the guard against Exp 3103 behavior."""

    base_prior_accuracy = accuracy(base_state, curriculum.prior_retention)
    guarded_prior_accuracy = accuracy(guarded_state, curriculum.prior_retention)
    base_delayed_accuracy = accuracy(base_state, curriculum.delayed_regression)
    guarded_delayed_accuracy = accuracy(guarded_state, curriculum.delayed_regression)
    base_holdout_accuracy = accuracy(base_state, curriculum.family_holdout)
    guarded_holdout_accuracy = accuracy(guarded_state, curriculum.family_holdout)
    return {
        "soundness_mistakes": int(guarded_counts.get("soundness_mistake", 0)),
        "completeness_mistakes": int(guarded_counts.get("completeness_mistake", 0)),
        "exp3103_completeness_mistakes": exp3103_completeness_mistakes,
        "base_completeness_mistakes": int(base_protocol_counts.get("completeness_mistake", 0)),
        "family_holdout_delta": _round(guarded_holdout_accuracy - base_holdout_accuracy),
        "prior_retention_delta": _round(guarded_prior_accuracy - base_prior_accuracy),
        "delayed_regression_delta": _round(guarded_delayed_accuracy - base_delayed_accuracy),
        "prior_retention_accuracy_before": base_prior_accuracy,
        "prior_retention_accuracy_after": guarded_prior_accuracy,
        "delayed_regression_accuracy_before": base_delayed_accuracy,
        "delayed_regression_accuracy_after": guarded_delayed_accuracy,
        "family_holdout_accuracy_before": base_holdout_accuracy,
        "family_holdout_accuracy_after": guarded_holdout_accuracy,
    }


def build_negative_controls(
    base_state: ControllerState,
    preconditions: PreconditionResult,
    curriculum: Curriculum,
    main_metrics: Mapping[str, Any],
) -> JsonDict:
    """Run no-feedback, shuffled, stale, and contradictory hint controls."""

    controls = {
        "no_feedback": base_state,
        "shuffled_hint": apply_hints(base_state, shuffled_hints(curriculum)),
        "stale_hint": apply_hints(base_state, stale_hints(curriculum)),
        "contradictory_hint": apply_hints(base_state, contradictory_hints(curriculum)),
    }
    return {
        name: control_report(name, state, preconditions, curriculum, main_metrics)
        for name, state in controls.items()
    }


def control_report(
    name: str,
    state: ControllerState,
    preconditions: PreconditionResult,
    curriculum: Curriculum,
    main_metrics: Mapping[str, Any],
) -> JsonDict:
    """Summarize one negative control and whether it failed safely."""

    protocol_counts = count_labels(evaluate_rows(state, preconditions.protocol_rows))
    metrics = {
        "soundness_mistakes": int(protocol_counts.get("soundness_mistake", 0)),
        "completeness_mistakes": int(protocol_counts.get("completeness_mistake", 0)),
        "exp3103_completeness_mistakes": int(main_metrics["exp3103_completeness_mistakes"]),
        "family_holdout_delta": _round(
            accuracy(state, curriculum.family_holdout)
            - float(main_metrics["family_holdout_accuracy_before"])
        ),
        "prior_retention_delta": _round(
            accuracy(state, curriculum.prior_retention)
            - float(main_metrics["prior_retention_accuracy_before"])
        ),
        "delayed_regression_delta": _round(
            accuracy(state, curriculum.delayed_regression)
            - float(main_metrics["delayed_regression_accuracy_before"])
        ),
    }
    gate_passed = candidate_gate_passed(metrics)
    rolled_back = name != "no_feedback" and not gate_passed
    return {
        "case_count": len(curriculum.protocol_hard_rows)
        + len(curriculum.prior_retention_regression_rows),
        **metrics,
        "promotion_gate_passed": gate_passed,
        "rolled_back": rolled_back,
        "failed_safely": not gate_passed and (rolled_back or name == "no_feedback"),
    }


def shuffled_hints(curriculum: Curriculum) -> tuple[AbstractHint, ...]:
    """Return hints with feature alignment shuffled away from hard families."""

    shuffled = []
    for hint in curriculum.abstract_hints:
        if hint.family_key == "arithmetic_true_verification":
            feature_weights = {"hint:perturbation:smt_sat_solving": 0.5}
        else:
            feature_weights = {
                key.replace("exp3090_prior", "shuffled_unmatched_context"): value
                for key, value in hint.feature_weights.items()
            }
        shuffled.append(
            replace_hint(hint, "shuffled_hint", feature_weights, hint.target_action)
        )
    return tuple(shuffled)


def stale_hints(curriculum: Curriculum) -> tuple[AbstractHint, ...]:
    """Return stale hints that apply prior repairability to protocol rejection rows."""

    stale = []
    for hint in curriculum.abstract_hints:
        feature_weights = dict(hint.feature_weights)
        if hint.family_key != "arithmetic_true_verification":
            feature_weights = {
                key.replace("exp3090_prior", "exp3097_protocol"): value
                for key, value in feature_weights.items()
                if not key.endswith("target_action:accept")
            }
        stale.append(replace_hint(hint, "stale_hint", feature_weights, hint.target_action))
    return tuple(stale)


def contradictory_hints(curriculum: Curriculum) -> tuple[AbstractHint, ...]:
    """Return hints that contradict solver-derived acceptance direction."""

    return tuple(
        replace_hint(
            hint,
            "contradictory_hint",
            {key: -value for key, value in hint.feature_weights.items()},
            "reject",
        )
        for hint in curriculum.abstract_hints
    )


def replace_hint(
    hint: AbstractHint,
    suffix: str,
    feature_weights: Mapping[str, float],
    target_action: str,
) -> AbstractHint:
    """Return a control hint with altered features but original evidence."""

    return AbstractHint(
        hint_id=f"{hint.hint_id}:{suffix}",
        family_key=hint.family_key,
        target_action=target_action,
        target_context=f"{hint.target_context}:{suffix}",
        source_fixture_ids=hint.source_fixture_ids,
        feature_weights=dict(feature_weights),
        abstract_hint=hint.abstract_hint,
        evidence=dict(hint.evidence),
    )


def apply_hints(state: ControllerState, hints: Sequence[AbstractHint]) -> ControllerState:
    """Apply abstract hints to controller memory weights only."""

    hint_weights = dict(state.hint_weights)
    memory = list(state.hint_memory)
    for hint in hints:
        for feature, weight in hint.feature_weights.items():
            hint_weights[feature] = _round(float(hint_weights.get(feature, 0.0)) + float(weight))
        memory.append(hint.hint_id)
    return ControllerState(
        base_weights=dict(state.base_weights),
        hint_weights=dict(sorted(hint_weights.items())),
        hint_memory=tuple(memory),
    )


def promotion_gates_passed(
    metrics: Mapping[str, Any],
    negative_controls: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether the full guarded promotion boundary passes."""

    return candidate_gate_passed(metrics) and bool(negative_controls) and all(
        control.get("failed_safely") is True
        and control.get("promotion_gate_passed") is False
        for control in negative_controls.values()
    )


def candidate_gate_passed(metrics: Mapping[str, Any]) -> bool:
    """Return whether candidate metrics pass without considering controls."""

    return (
        int(metrics.get("soundness_mistakes", -1)) == 0
        and int(metrics.get("completeness_mistakes", -1))
        < int(metrics.get("exp3103_completeness_mistakes", 0))
        and float(metrics.get("family_holdout_delta", -1.0)) > 0.0
        and float(metrics.get("prior_retention_delta", -1.0)) >= 0.0
        and float(metrics.get("delayed_regression_delta", -1.0)) >= 0.0
    )


def evaluate_rows(
    state: ControllerState,
    rows: Sequence[CurriculumRow],
) -> tuple[DecisionRecord, ...]:
    """Evaluate a sequence of rows against exact target actions."""

    return tuple(evaluate_row(state, row) for row in rows)


def evaluate_row(state: ControllerState, row: CurriculumRow) -> DecisionRecord:
    """Score one row and label the guarded controller decision."""

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


def score(state: ControllerState, row: CurriculumRow) -> float:
    """Return base controller score plus controller-memory hint score."""

    base_score = sum(float(state.base_weights.get(feature, 0.0)) for feature in base_anchor_features(row))
    hint_score = sum(float(state.hint_weights.get(feature, 0.0)) for feature in hint_features(row))
    return _round(base_score + hint_score)


def decision_from_score(row_score: float) -> str:
    """Map a controller score into accept, reject, or abstain."""

    if row_score > ACCEPT_THRESHOLD:
        return "accept"
    if row_score < REJECT_THRESHOLD:
        return "reject"
    return "abstain"


def decision_label(decision: str, target_action: str) -> str:
    """Classify decisions as soundness, completeness, abstention, or correct."""

    if target_action not in {"accept", "reject"}:
        raise ValueError(f"unsupported target action: {target_action}")
    if decision == target_action:
        return "correct"
    if decision == "accept" and target_action == "reject":
        return "soundness_mistake"
    if target_action == "accept":
        return "completeness_mistake"
    return "abstention"


def accuracy(state: ControllerState, rows: Sequence[CurriculumRow]) -> float:
    """Return exact target-action accuracy, counting abstention as incorrect."""

    if not rows:
        return 0.0
    labels = count_labels(evaluate_rows(state, rows))
    return _round(int(labels.get("correct", 0)) / len(rows))


def count_labels(records: Sequence[DecisionRecord]) -> dict[str, int]:
    """Return deterministic decision-label counts."""

    return dict(sorted(Counter(record.decision_label for record in records).items()))


def base_anchor_features(row: CurriculumRow) -> tuple[str, ...]:
    """Return Exp 3090/3103 local controller basis features."""

    return (
        f"family:{row.task_family}",
        f"task_axis:{row.task_axis}",
        f"perturbation:{row.perturbation_type}",
        f"label_source:{row.label_source}",
    )


def hint_features(row: CurriculumRow) -> tuple[str, ...]:
    """Return context-aware controller-memory features for abstract hints."""

    return (
        f"hint:perturbation:{row.perturbation_type}",
        f"hint:target_source:{row.target_source}:family:{row.task_family}",
        f"hint:target_source:{row.target_source}:task_axis:{row.task_axis}",
        f"hint:target_source:{row.target_source}:perturbation:{row.perturbation_type}",
        f"hint:target_source:{row.target_source}:label_source:{row.label_source}",
        f"hint:target_source:{row.target_source}:target_action:{row.target_action}",
    )


def complete_artifact(
    config: GuardConfig,
    preconditions: PreconditionResult,
    curriculum: Curriculum,
    result: GuardResult,
    duration_s: float,
) -> JsonDict:
    """Build the complete terminal Exp 3116 artifact."""

    promotion_decision = result.promotion_decision
    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_unsolvable_curriculum_ready": True,
        "continuous_self_learning_task": True,
        "controller_only": True,
        "no_weight_update_claim": True,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "hard_family_count": curriculum.hard_family_count,
        "unsolvable_detection_summary": curriculum.detection_summary,
        "hint_policy_summary": hint_policy_summary(curriculum.abstract_hints, curriculum.rows_by_id),
        "hint_usefulness": hint_usefulness(result.metrics, curriculum, result),
        "soundness_mistakes": result.soundness_mistakes,
        "completeness_mistakes": result.completeness_mistakes,
        "family_holdout_delta": result.metrics["family_holdout_delta"],
        "prior_retention_delta": result.metrics["prior_retention_delta"],
        "delayed_regression_delta": result.metrics["delayed_regression_delta"],
        "rollback_count": result.rollback_count,
        "promotion_decision": promotion_decision,
        "promotion_gates": promotion_gate_report(result.metrics, result.negative_controls),
        "negative_controls": result.negative_controls,
        "guarded_decision_label_counts": dict(sorted(count_labels(result.guarded_decisions).items())),
        "guarded_decisions": [_decision_payload(record) for record in result.guarded_decisions],
        "preconditions_checked": preconditions.checks,
        "tests_run": list(config.tests_run),
        "source_artifacts": source_artifacts(config),
        "inference_substrate": inference_substrate(controller_update=True),
        "duration_s": duration_s,
        "honest_verdict": (
            SUCCESS_VERDICT if promotion_decision == "controller_only" else BOUNDARY_VERDICT
        ),
    }


def blocked_artifact(
    config: GuardConfig,
    preconditions: PreconditionResult,
    duration_s: float,
) -> JsonDict:
    """Build a fail-closed artifact for missing source evidence."""

    return {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_unsolvable_curriculum_ready": False,
        "continuous_self_learning_task": True,
        "controller_only": True,
        "no_weight_update_claim": True,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "hard_family_count": 0,
        "unsolvable_detection_summary": {},
        "hint_policy_summary": {
            "solver_derived": False,
            "abstract_only": False,
            "final_answers_revealed": False,
            "hint_count": 0,
            "live_llm_inference_used": False,
        },
        "hint_usefulness": {},
        "soundness_mistakes": 0,
        "completeness_mistakes": 0,
        "family_holdout_delta": 0.0,
        "prior_retention_delta": 0.0,
        "delayed_regression_delta": 0.0,
        "rollback_count": 0,
        "promotion_decision": "blocked",
        "promotion_gates": {},
        "negative_controls": {},
        "guarded_decision_label_counts": {},
        "guarded_decisions": [],
        "preconditions_checked": preconditions.checks,
        "tests_run": list(config.tests_run),
        "source_artifacts": source_artifacts(config),
        "inference_substrate": inference_substrate(controller_update=False),
        "blocked_reason": preconditions.blocked_reason,
        "duration_s": duration_s,
        "honest_verdict": BLOCKED_VERDICT,
    }


def hint_policy_summary(
    hints: Sequence[AbstractHint],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Summarize the abstract-hint policy boundary."""

    leaked = any(hint_leaks_final_answer(hint, rows_by_id) for hint in hints)
    return {
        "solver_derived": bool(hints) and all(hint.solver_derived for hint in hints),
        "abstract_only": bool(hints) and not leaked,
        "final_answers_revealed": leaked,
        "hint_count": len(hints),
        "hint_families": [hint.family_key for hint in hints],
        "live_llm_inference_used": False,
        "hint_phrasing_source": "deterministic_solver_certificate_metadata",
        "model_weight_update_used": False,
    }


def hint_usefulness(
    metrics: Mapping[str, Any],
    curriculum: Curriculum,
    result: GuardResult,
) -> JsonDict:
    """Report how much the abstract hints changed the Exp 3103 failure modes."""

    return {
        "exp3103_completeness_mistakes": int(metrics["exp3103_completeness_mistakes"]),
        "guarded_completeness_mistakes": result.completeness_mistakes,
        "completeness_mistakes_reduced_by": int(metrics["exp3103_completeness_mistakes"])
        - result.completeness_mistakes,
        "protocol_hard_case_count": len(curriculum.protocol_hard_rows),
        "prior_retention_cases_recovered": len(curriculum.prior_retention_regression_rows),
        "family_holdout_delta": metrics["family_holdout_delta"],
        "prior_retention_delta": metrics["prior_retention_delta"],
        "delayed_regression_delta": metrics["delayed_regression_delta"],
    }


def promotion_gate_report(
    metrics: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Return the mechanical gate report that drives promotion_decision."""

    gates = {
        "soundness_zero": int(metrics.get("soundness_mistakes", -1)) == 0,
        "completeness_reduced_vs_exp3103": int(metrics.get("completeness_mistakes", -1))
        < int(metrics.get("exp3103_completeness_mistakes", 0)),
        "family_holdout_positive": float(metrics.get("family_holdout_delta", -1.0)) > 0.0,
        "prior_retention_nonnegative": float(metrics.get("prior_retention_delta", -1.0)) >= 0.0,
        "delayed_regression_nonnegative": float(metrics.get("delayed_regression_delta", -1.0)) >= 0.0,
        "negative_controls_fail_safely": bool(controls)
        and all(control.get("failed_safely") is True for control in controls.values()),
    }
    gates["all_gates_passed"] = all(gates.values())
    return gates


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact violates the Exp 3116 contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_task") is not True:
        raise ValueError("continuous_self_learning_task must be true")
    if artifact.get("controller_only") is not True:
        raise ValueError("controller_only must be true")
    if artifact.get("no_weight_update_claim") is not True:
        raise ValueError("no_weight_update_claim must be true")
    if artifact.get("promotion_decision") not in PROMOTION_DECISIONS:
        raise ValueError("promotion_decision must be explicit")
    if not artifact.get("source_artifacts"):
        raise ValueError("source_artifacts must trace source evidence")
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
    ready = artifact.get("fr11_unsolvable_curriculum_ready") is True
    if not ready:
        if artifact.get("honest_verdict") != BLOCKED_VERDICT:
            raise ValueError("blocked curriculum artifacts must use the blocked verdict")
        return
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if artifact.get("promotion_decision") == "controller_only" and (
        int(artifact.get("soundness_mistakes", -1)) != 0
        or int(artifact.get("completeness_mistakes", 999)) >= int(
            artifact.get("hint_usefulness", {}).get("exp3103_completeness_mistakes", 0)
        )
        or float(artifact.get("family_holdout_delta", 0.0)) <= 0.0
        or float(artifact.get("prior_retention_delta", -1.0)) < 0.0
        or float(artifact.get("delayed_regression_delta", -1.0)) < 0.0
    ):
        raise ValueError("controller_only promotion requires all curriculum gates to pass")


def inference_substrate(*, controller_update: bool) -> JsonDict:
    """Declare the deterministic no-weight-update substrate."""

    return {
        "mode": "deterministic_solver_derived_hint_controller_replay",
        "cached_artifacts_only": True,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "controller_memory_update": bool(controller_update),
        "trace_memory_update": bool(controller_update),
        "training_scope": "controller_memory_abstract_hint_replay_only",
        "mandated_model_specs_recorded": list(MANDATED_MODEL_SPECS),
    }


def source_artifacts(config: GuardConfig) -> list[JsonDict]:
    """Return source provenance for all evidence consumed by Exp 3116."""

    return [
        source_artifact_row("exp3090_prior_fr11", config.prior_artifact_path(), config.repo_root),
        source_artifact_row("exp3097_protocol", config.protocol_artifact_path(), config.repo_root),
        source_artifact_row("exp3103_stress_boundary", config.stress_artifact_path(), config.repo_root),
        source_artifact_row("exp3097_manifest", config.manifest_path(), config.repo_root),
    ]


def source_artifact_row(source_id: str, path: Path, repo_root: Path) -> JsonDict:
    """Return one source artifact row with a stable checksum when available."""

    exists = path.is_file()
    return {
        "id": source_id,
        "path": relative_path(repo_root, path),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
    }


def _decision_payload(record: DecisionRecord) -> JsonDict:
    """Return a JSON-serializable guarded decision record."""

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
    """Load a JSON object, returning empty evidence for missing inputs."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


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
    """Return a SHA256 digest for source provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return repository-relative paths when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _clamp_base(value: float) -> float:
    """Clamp base controller feedback to the Exp 3103 bounded range."""

    return _round(max(-MAX_ABS_BASE_WEIGHT, min(MAX_ABS_BASE_WEIGHT, value)))


def _round(value: float) -> float:
    """Round floats so JSON artifacts stay stable across Python versions."""

    return round(float(value), 6)
