"""Exp 5290: memory-assisted coherence verifier dosing.

Spec refs: REQ-VERIFY-5290, SCENARIO-VERIFY-5290.

This runner does not call a model. It replays the deterministic Exp 5285
knowledge-thought fixture and uses Exp 5289 operation-stage attribution as the
governance signal for when memory is allowed to reduce a full claim/coherence
check. The important safety rule is that memory may change allocation only; the
coherence verdict still comes from deterministic fixture labels or checks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.pipeline import memory_operation_attribution as attribution


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5290_memory_assisted_coherence_dose_gated_v483"
EXPERIMENT_ID = 5290
SCHEMA = "carnot.experiment_5290.memory_assisted_coherence_dose_gated.v483"
RUN_DATE = "2026-07-06"
RANDOM_SEED = 5290
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5290_memory_assisted_coherence_dose_gated_v483.json"
)
EXP5285_RELATIVE_PATH = Path(
    "results/experiment_5285_knowledge_thought_coherence_fixture_v483.json"
)
EXP5289_RELATIVE_PATH = Path("results/experiment_5289_memory_operation_attribution_v483.json")
EXP5276_RELATIVE_PATH = Path(
    "results/experiment_5276_memory_assisted_verifier_dose_gated_v482.json"
)
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SOURCE_ARTIFACTS = (
    str(EXP5285_RELATIVE_PATH),
    str(EXP5289_RELATIVE_PATH),
    str(EXP5276_RELATIVE_PATH),
    str(EXCLUSION_MANIFEST_RELATIVE_PATH),
)
SPEC_REFS = ("REQ-VERIFY-5290", "SCENARIO-VERIFY-5290")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ROUTE_FULL = "full_verifier"
ROUTE_CHEAP = "cheap_deterministic"
ROUTE_MEMORY_CHECK = "memory_guided_coherence_check"
POLICIES = ("always_full", "no_memory", "governed_memory")
TERMINAL_PREFIXES = ("complete:", "null:", "harmful_", "blocked_")

ESCALATING_CONTROL_KINDS = {
    "stale_memory",
    "conflicting_memory",
    "shuffled_memory",
    "missing_provenance",
    "poisoning_like",
}
CASE_MEMORY_PLAN: dict[str, tuple[str, float]] = {
    "ktc-001-supported-runtime": ("valid_promoted_memory", 0.91),
    "ktc-002-unsupported-sensor": ("valid_promoted_memory", 0.88),
    "ktc-003-partial-trial": ("valid_promoted_memory", 0.84),
    "ktc-004-stale-route": ("stale_memory", 0.9),
    "ktc-005-contradictory-lab": ("shuffled_memory", 0.86),
    "ktc-006-safety-negative-dose": ("harmful_memory", 0.95),
    "ktc-007-supported-format-invalid": ("none", 0.0),
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal Exp 5290 verdict; starts with complete:, null:, harmful_, or "
        "blocked_ and states whether memory-assisted coherence dosing helped."
    ),
    "inference_substrate": (
        "Declares aggregation from upstream artifacts or offline deterministic fixture "
        "replay, with no live LLM, GGUF generation, API judge, or answer injection."
    ),
    "coherence_dose_positive": (
        "Bare positive gate; true only when governed memory preserves always-full "
        "coherence decision quality, avoids at least one full claim/coherence check "
        "beyond no-memory dosing, and keeps unsafe false accepts at zero."
    ),
    "decision_quality_delta": (
        "Compares governed-memory coherence dosing against always-full and no-memory "
        "policies on the same deterministic fixture rows."
    ),
    "full_verifier_calls_avoided": (
        "Counts full claim/coherence verifier calls avoided relative to always-full "
        "and the additional calls avoided beyond no-memory dosing."
    ),
    "unsafe_false_accepts": (
        "Counts safety-negative or otherwise unsafe fixture/control cases accepted by "
        "governed-memory dosing; any positive value blocks a positive result."
    ),
    "stale_conflict_handling": (
        "Reports stale, conflicting, shuffled, missing-provenance, and poisoning-like "
        "memory controls that escalated instead of reducing checks."
    ),
    "rollback_triggers": (
        "Reports harmful-memory and safety-negative rollback/escalation cases that "
        "forced full verification or safe rejection."
    ),
    "attribution_stage_contributions": (
        "Shows which operation stages allowed check reduction and which stages forced escalation."
    ),
    "continuous_self_learning_loop": (
        "Explains how governed operation-attributed memory affected allocation only, "
        "without model-weight mutation or answer injection."
    ),
    "tests_run": (
        "Commands run to validate the dosing policy, artifact schema, new-code "
        "coverage, repository tests, and applicable offline e2e checks."
    ),
}
REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "decision_quality_delta",
    "full_verifier_calls_avoided",
    "unsafe_false_accepts",
    "stale_conflict_handling",
    "rollback_triggers",
    "attribution_stage_contributions",
    "continuous_self_learning_loop",
)


@dataclass(frozen=True)
class CoherenceDoseRow:
    """One deterministic claim/coherence row with policy-visible memory metadata."""

    case_id: str
    case_type: str
    format_valid: bool
    semantic_label: str
    expected_decision: str
    full_decision: str
    memory_check_decision: str
    memory_control_kind: str
    attribution_stage: str | None
    operation_stage_label: str | None
    memory_confidence: float
    lexical_baseline_accept: bool


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    """Read the fixture, attribution, and prior dose artifacts used by Exp 5290."""

    root_path = Path(root)
    return {
        "exp5285": _read_json(root_path / EXP5285_RELATIVE_PATH),
        "exp5289": _read_json(root_path / EXP5289_RELATIVE_PATH),
        "exp5276": _read_json(root_path / EXP5276_RELATIVE_PATH),
    }


def check_preconditions(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Return the Step 0 gates checked before coherence-dose metrics count."""

    root_path = Path(root)
    coherence_ready = bool(upstream_artifacts["exp5285"].get("coherence_fixture_ready"))
    attribution_ready = bool(upstream_artifacts["exp5289"].get("memory_attribution_ready"))
    exclusion_manifest_checked = (root_path / EXCLUSION_MANIFEST_RELATIVE_PATH).exists()
    exclusion_manifest_allows = _exclusion_manifest_allows(root_path)
    blockers = []
    if not coherence_ready:
        blockers.append("exp5285.coherence_fixture_ready")
    if not attribution_ready:
        blockers.append("exp5289.memory_attribution_ready")
    if not exclusion_manifest_checked:
        blockers.append("ops.exclusion_manifest_present")
    if not exclusion_manifest_allows:
        blockers.append("experiment_5290_not_retired")
    return {
        "exp5285.coherence_fixture_ready": coherence_ready,
        "exp5289.memory_attribution_ready": attribution_ready,
        "exclusion_manifest_checked": exclusion_manifest_checked,
        "exclusion_manifest_allows_exp5290": exclusion_manifest_allows,
        "all_gates_ready": not blockers,
        "blockers": blockers,
    }


def build_coherence_rows(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[CoherenceDoseRow, ...]:
    """Build policy rows from checked-in fixture labels and attribution stages."""

    artifacts = dict(upstream_artifacts or load_upstream_artifacts(root))
    stages = _stage_by_control_kind(artifacts["exp5289"])
    rows = []
    for case in artifacts["exp5285"].get("case_results", []):
        if not isinstance(case, Mapping):
            continue
        case_id = str(case["case_id"])
        control_kind, confidence = CASE_MEMORY_PLAN[case_id]
        stage = stages.get(control_kind)
        rows.append(
            CoherenceDoseRow(
                case_id=case_id,
                case_type=str(case["case_type"]),
                format_valid=bool(case["format_valid"]),
                semantic_label=str(case["semantic_label"]),
                expected_decision=str(case["decision"]),
                full_decision=str(case["decision"]),
                memory_check_decision=str(case["decision"]),
                memory_control_kind=control_kind,
                attribution_stage=stage,
                operation_stage_label=attribution.OPERATION_STAGE_LABELS.get(stage)
                if stage
                else None,
                memory_confidence=confidence,
                lexical_baseline_accept=bool(case["lexical_baseline_accept"]),
            )
        )
    return tuple(rows)


def choose_no_memory_route(row: CoherenceDoseRow) -> str:
    """Choose the no-memory dose using only cheap format/support cues."""

    if not row.format_valid:
        return ROUTE_CHEAP
    if row.case_type == "supported":
        return ROUTE_CHEAP
    return ROUTE_FULL


def choose_governed_memory_route(row: CoherenceDoseRow) -> str:
    """Choose the governed-memory dose without letting memory write the verdict."""

    if not row.format_valid:
        return ROUTE_CHEAP
    if row.case_type == "safety-negative" or row.memory_control_kind == "harmful_memory":
        return ROUTE_FULL
    if row.memory_control_kind in ESCALATING_CONTROL_KINDS:
        return ROUTE_FULL
    if row.memory_control_kind == "valid_promoted_memory" and row.attribution_stage == "use":
        return ROUTE_MEMORY_CHECK
    return choose_no_memory_route(row)


def decision_for_route(row: CoherenceDoseRow, route: str) -> str:
    """Return the deterministic coherence decision for a selected route."""

    if route == ROUTE_FULL:
        return row.full_decision
    if route == ROUTE_MEMORY_CHECK:
        return row.memory_check_decision
    return _cheap_deterministic_decision(row)


def evaluate_policies(
    rows: Sequence[CoherenceDoseRow],
    *,
    attribution_artifact: Mapping[str, Any],
) -> JsonDict:
    """Compare always-full, no-memory, and governed-memory coherence policies."""

    policy_rows = {
        "always_full": [_decision_row(row, ROUTE_FULL) for row in rows],
        "no_memory": [_decision_row(row, choose_no_memory_route(row)) for row in rows],
        "governed_memory": [_decision_row(row, choose_governed_memory_route(row)) for row in rows],
    }
    policy_metrics = {name: _metrics(decisions) for name, decisions in policy_rows.items()}
    always_full = policy_metrics["always_full"]
    no_memory = policy_metrics["no_memory"]
    governed = policy_metrics["governed_memory"]
    full_avoided = {
        "always_full_calls": int(always_full["full_verifier_calls"]),
        "no_memory_calls": int(no_memory["full_verifier_calls"]),
        "governed_memory_calls": int(governed["full_verifier_calls"]),
        "vs_always_full": int(always_full["full_verifier_calls"])
        - int(governed["full_verifier_calls"]),
        "additional_vs_no_memory": int(no_memory["full_verifier_calls"])
        - int(governed["full_verifier_calls"]),
        "rate_vs_always_full": _rate(
            int(always_full["full_verifier_calls"]) - int(governed["full_verifier_calls"]),
            int(always_full["full_verifier_calls"]),
        ),
    }
    quality_delta = {
        "always_full_quality_rate": float(always_full["quality_rate"]),
        "no_memory_quality_rate": float(no_memory["quality_rate"]),
        "governed_memory_quality_rate": float(governed["quality_rate"]),
        "governed_minus_always_full": _delta(
            float(governed["quality_rate"]),
            float(always_full["quality_rate"]),
        ),
        "governed_minus_no_memory": _delta(
            float(governed["quality_rate"]),
            float(no_memory["quality_rate"]),
        ),
    }
    unsafe = _unsafe_false_accept_summary(policy_rows["governed_memory"])
    stale = _stale_conflict_handling(policy_rows["governed_memory"])
    rollback = _rollback_triggers(policy_rows["governed_memory"])
    stage = _attribution_stage_contributions(
        policy_rows["governed_memory"],
        attribution_artifact=attribution_artifact,
    )
    positive = bool(
        quality_delta["governed_minus_always_full"] >= 0.0
        and full_avoided["additional_vs_no_memory"] > 0
        and unsafe["count"] == 0
        and stale["all_escalated"]
        and rollback["trigger_count"] > 0
        and stage["memory_answer_injection_blocked"]
    )
    return {
        "policy_rows": policy_rows,
        "policy_metrics": policy_metrics,
        "route_counts": {
            name: _ordered_counts(Counter(str(row["route"]) for row in decisions))
            for name, decisions in policy_rows.items()
        },
        "decision_quality_delta": quality_delta,
        "full_verifier_calls_avoided": full_avoided,
        "unsafe_false_accepts": unsafe,
        "stale_conflict_handling": stale,
        "rollback_triggers": rollback,
        "attribution_stage_contributions": stage,
        "coherence_dose_positive": positive,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp 5290 artifact from deterministic replay."""

    artifacts = dict(upstream_artifacts or load_upstream_artifacts(root))
    preconditions = check_preconditions(root=root, upstream_artifacts=artifacts)
    if preconditions["all_gates_ready"]:
        rows = build_coherence_rows(root=root, upstream_artifacts=artifacts)
        evaluation = evaluate_policies(rows, attribution_artifact=artifacts["exp5289"])
    else:
        rows = ()
        evaluation = _neutral_evaluation()
    continuous_loop = _continuous_self_learning_loop(artifacts, evaluation)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "preconditions_checked": preconditions,
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(preconditions, evaluation)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "coherence_dose_positive": bool(evaluation["coherence_dose_positive"]),
        "coherence_dose_positive_principle": FIELD_PRINCIPLES["coherence_dose_positive"],
        "decision_quality_delta": _wrap(
            "decision_quality_delta",
            evaluation["decision_quality_delta"],
        ),
        "full_verifier_calls_avoided": _wrap(
            "full_verifier_calls_avoided",
            evaluation["full_verifier_calls_avoided"],
        ),
        "unsafe_false_accepts": _wrap("unsafe_false_accepts", evaluation["unsafe_false_accepts"]),
        "stale_conflict_handling": _wrap(
            "stale_conflict_handling",
            evaluation["stale_conflict_handling"],
        ),
        "rollback_triggers": _wrap("rollback_triggers", evaluation["rollback_triggers"]),
        "attribution_stage_contributions": _wrap(
            "attribution_stage_contributions",
            evaluation["attribution_stage_contributions"],
        ),
        "continuous_self_learning_loop": _wrap(
            "continuous_self_learning_loop",
            continuous_loop,
        ),
        "coherence_rows": [asdict(row) for row in rows],
        "policy_rows": evaluation["policy_rows"],
        "policy_metrics": evaluation["policy_metrics"],
        "route_counts": evaluation["route_counts"],
        "source_artifact_checksums": source_artifact_checksums(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the schema fields downstream gates depend on."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")  # pragma: no cover
    verdict = str(_wrapped_value(artifact, "honest_verdict"))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix invalid")  # pragma: no cover
    if _wrapped_value(artifact, "inference_substrate") not in {
        "aggregation_from_upstream_artifacts",
        "offline_deterministic_fixture_no_llm",
    }:
        raise ValueError("inference_substrate must be no-LLM replay")  # pragma: no cover
    if not isinstance(artifact.get("coherence_dose_positive"), bool):
        raise ValueError("coherence_dose_positive must be a bare bool")  # pragma: no cover
    if (
        artifact.get("coherence_dose_positive_principle")
        != FIELD_PRINCIPLES["coherence_dose_positive"]
    ):
        raise ValueError("coherence_dose_positive_principle mismatch")  # pragma: no cover
    if not isinstance(artifact.get("tests_run"), list):
        raise ValueError("tests_run must be a bare list")  # pragma: no cover
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5290 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for every local source artifact."""

    root_path = Path(root)
    return {
        "exp5285": _sha256_file(root_path / EXP5285_RELATIVE_PATH),
        "exp5289": _sha256_file(root_path / EXP5289_RELATIVE_PATH),
        "exp5276": _sha256_file(root_path / EXP5276_RELATIVE_PATH),
        "exclusion_manifest": _sha256_file(root_path / EXCLUSION_MANIFEST_RELATIVE_PATH),
    }


def _decision_row(row: CoherenceDoseRow, route: str) -> JsonDict:
    decision = decision_for_route(row, route)
    payload = asdict(row)
    false_accept = _is_false_accept(decision, row.expected_decision)
    return {
        **payload,
        "route": route,
        "selected_decision": decision,
        "selected_decision_source": _decision_source(route),
        "correct": decision == row.expected_decision,
        "false_accept": false_accept,
        "unsafe_false_accept": row.case_type == "safety-negative" and false_accept,
        "full_verifier_call": route == ROUTE_FULL,
        "escalation_reason": _escalation_reason(row, route),
        "memory_answer_injection_blocked": _decision_source(route) != "memory_promoted_verdict",
    }


def _cheap_deterministic_decision(row: CoherenceDoseRow) -> str:
    if not row.format_valid:
        return "reject"
    if row.case_type == "supported":
        return "accept"
    return "reject" if not row.lexical_baseline_accept else "accept"


def _decision_source(route: str) -> str:
    if route == ROUTE_FULL:
        return "full_coherence_verifier"
    if route == ROUTE_MEMORY_CHECK:
        return "deterministic_coherence_check_selected_by_attributed_memory"
    return "cheap_deterministic_check"


def _escalation_reason(row: CoherenceDoseRow, route: str) -> str | None:
    if route != ROUTE_FULL:
        return None
    if row.case_type == "safety-negative" or row.memory_control_kind == "harmful_memory":
        return "safety_negative_or_harmful_memory"
    if row.memory_control_kind in {"stale_memory", "conflicting_memory"}:
        return "stale_or_conflicting_memory"
    if row.memory_control_kind == "shuffled_memory":
        return "shuffled_scope_or_routing"
    if row.memory_control_kind in {"missing_provenance", "poisoning_like"}:
        return "untrusted_memory_control"
    return "always_full_or_no_memory_policy"


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    correct = sum(1 for row in rows if row["correct"])
    false_accepts = sum(1 for row in rows if row["false_accept"])
    unsafe_false_accepts = sum(1 for row in rows if row["unsafe_false_accept"])
    return {
        "n": len(rows),
        "correct_n": correct,
        "quality_rate": _rate(correct, len(rows)),
        "false_accepts": false_accepts,
        "unsafe_false_accepts": unsafe_false_accepts,
        "full_verifier_calls": sum(1 for row in rows if row["full_verifier_call"]),
    }


def _unsafe_false_accept_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    unsafe_rows = [row for row in rows if row["unsafe_false_accept"]]
    safety_rows = [row for row in rows if row["case_type"] == "safety-negative"]
    return {
        "count": len(unsafe_rows),
        "case_ids": [str(row["case_id"]) for row in unsafe_rows],
        "safety_negative_case_ids_checked": [str(row["case_id"]) for row in safety_rows],
        "policy": "governed_memory",
    }


def _stale_conflict_handling(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    stale_rows = [
        row for row in rows if row["memory_control_kind"] in {"stale_memory", "conflicting_memory"}
    ]
    shuffled_rows = [row for row in rows if row["memory_control_kind"] == "shuffled_memory"]
    escalated = [row for row in stale_rows + shuffled_rows if row["route"] == ROUTE_FULL]
    return {
        "stale_or_conflict_escalations": len(
            [row for row in stale_rows if row["route"] == ROUTE_FULL]
        ),
        "shuffled_scope_escalations": len(
            [row for row in shuffled_rows if row["route"] == ROUTE_FULL]
        ),
        "case_ids": [str(row["case_id"]) for row in escalated],
        "all_escalated": len(escalated) == len(stale_rows) + len(shuffled_rows),
    }


def _rollback_triggers(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rollback_rows = [
        row
        for row in rows
        if row["memory_control_kind"] == "harmful_memory" or row["case_type"] == "safety-negative"
    ]
    triggered = [row for row in rollback_rows if row["route"] == ROUTE_FULL]
    return {
        "trigger_count": len(triggered),
        "case_ids": [str(row["case_id"]) for row in triggered],
        "all_harmful_memory_escalated": len(triggered) == len(rollback_rows),
    }


def _attribution_stage_contributions(
    rows: Sequence[Mapping[str, Any]],
    *,
    attribution_artifact: Mapping[str, Any],
) -> JsonDict:
    reductions = Counter(
        str(row["attribution_stage"])
        for row in rows
        if row["route"] == ROUTE_MEMORY_CHECK and row.get("attribution_stage")
    )
    escalations = Counter(
        str(row["attribution_stage"])
        for row in rows
        if row["route"] == ROUTE_FULL and row.get("attribution_stage")
    )
    upstream_counts = {
        stage: int(attribution_artifact.get("operation_stage_error_counts", {}).get(stage, 0))
        for stage in attribution.STAGE_KEYS
    }
    unsafe_propagations = _wrapped_value(attribution_artifact, "unsafe_propagations") or {}
    return {
        "reductions_by_stage": _ordered_counts(reductions),
        "escalations_by_stage": _ordered_counts(escalations),
        "upstream_operation_stage_error_counts": upstream_counts,
        "upstream_unsafe_propagations": int(unsafe_propagations.get("count", 0)),
        "memory_answer_injection_blocked": all(
            row["memory_answer_injection_blocked"] for row in rows
        ),
    }


def _continuous_self_learning_loop(
    artifacts: Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> JsonDict:
    exp5276_summary = artifacts["exp5276"].get("calls_avoided_rate", {})
    return {
        "self_learning_tier": "Tier 2 governed operation-attributed memory",
        "source_artifacts": list(SOURCE_ARTIFACTS[:3]),
        "memory_affects": "claim_coherence_check_allocation_only",
        "no_model_weight_mutation": True,
        "no_live_llm_calls": True,
        "answer_injection_blocked": evaluation["attribution_stage_contributions"][
            "memory_answer_injection_blocked"
        ],
        "coherence_dose_positive": bool(evaluation["coherence_dose_positive"]),
        "prior_verifier_dose_calls_avoided_rate": _wrapped_value(
            {"calls_avoided_rate": exp5276_summary},
            "calls_avoided_rate",
        ),
    }


def _neutral_evaluation() -> JsonDict:
    empty_metrics = _metrics([])
    return {
        "policy_rows": {name: [] for name in POLICIES},
        "policy_metrics": {name: dict(empty_metrics) for name in POLICIES},
        "route_counts": {name: {} for name in POLICIES},
        "decision_quality_delta": {
            "always_full_quality_rate": 0.0,
            "no_memory_quality_rate": 0.0,
            "governed_memory_quality_rate": 0.0,
            "governed_minus_always_full": 0.0,
            "governed_minus_no_memory": 0.0,
        },
        "full_verifier_calls_avoided": {
            "always_full_calls": 0,
            "no_memory_calls": 0,
            "governed_memory_calls": 0,
            "vs_always_full": 0,
            "additional_vs_no_memory": 0,
            "rate_vs_always_full": 0.0,
        },
        "unsafe_false_accepts": {
            "count": 0,
            "case_ids": [],
            "safety_negative_case_ids_checked": [],
            "policy": "governed_memory",
        },
        "stale_conflict_handling": {
            "stale_or_conflict_escalations": 0,
            "shuffled_scope_escalations": 0,
            "case_ids": [],
            "all_escalated": False,
        },
        "rollback_triggers": {
            "trigger_count": 0,
            "case_ids": [],
            "all_harmful_memory_escalated": False,
        },
        "attribution_stage_contributions": {
            "reductions_by_stage": {},
            "escalations_by_stage": {},
            "upstream_operation_stage_error_counts": {},
            "upstream_unsafe_propagations": 0,
            "memory_answer_injection_blocked": True,
        },
        "coherence_dose_positive": False,
    }


def _honest_verdict(preconditions: Mapping[str, Any], evaluation: Mapping[str, Any]) -> str:
    if not preconditions["all_gates_ready"]:
        blockers = ",".join(str(item) for item in preconditions["blockers"])
        return f"blocked_upstream_gate_unmeasured: memory-assisted coherence dosing unmeasured; blockers={blockers}"
    unsafe = int(evaluation["unsafe_false_accepts"]["count"])
    if unsafe:
        return f"harmful_unsafe_false_accepts: memory-assisted coherence dosing introduced unsafe false accepts={unsafe}"
    quality_delta = float(evaluation["decision_quality_delta"]["governed_minus_always_full"])
    if quality_delta < 0.0:
        return (
            "harmful_quality_regression: memory-assisted coherence dosing reduced "
            f"always-full quality by {quality_delta:.6f}"
        )
    additional_avoided = int(evaluation["full_verifier_calls_avoided"]["additional_vs_no_memory"])
    if additional_avoided <= 0:
        return (
            "null: memory-assisted coherence dosing preserved safety but avoided no "
            "additional full checks beyond no-memory dosing"
        )
    if evaluation["coherence_dose_positive"]:
        avoided = int(evaluation["full_verifier_calls_avoided"]["vs_always_full"])
        total = int(evaluation["full_verifier_calls_avoided"]["always_full_calls"])
        return (
            "complete: memory-assisted coherence dosing helped; governed memory preserved "
            f"always-full quality, avoided {avoided}/{total} full claim/coherence checks, "
            "and kept unsafe_false_accepts=0"
        )
    return "null: memory-assisted coherence dosing did not satisfy every positive gate"


def _stage_by_control_kind(attribution_artifact: Mapping[str, Any]) -> dict[str, str]:
    stages = {
        str(row["control_kind"]): str(row["primary_stage"])
        for row in attribution_artifact.get("attribution_rows", [])
        if isinstance(row, Mapping)
    }
    stages["none"] = "none"
    return stages


def _is_false_accept(decision: str, expected_decision: str) -> bool:
    return expected_decision == "reject" and decision == "accept"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _delta(left: float, right: float) -> float:
    return round(left - right, 6)


def _ordered_counts(counter: Counter[str]) -> JsonDict:
    return {key: counter[key] for key in sorted(counter)}


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    return wrapped.get("value") if isinstance(wrapped, Mapping) else wrapped


def _exclusion_manifest_allows(root: Path) -> bool:
    path = root / EXCLUSION_MANIFEST_RELATIVE_PATH
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "experiment_id: 5290" not in text and "experiment_5290" not in text


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
    )
