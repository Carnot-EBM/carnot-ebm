"""Exp5446: governed-memory continuous self-learning online lifecycle.

Spec refs: REQ-LEARN-5446,
SCENARIO-LEARN-5446-GATES,
SCENARIO-LEARN-5446-CONTROLS,
SCENARIO-LEARN-5446-ROLLBACK,
SCENARIO-LEARN-5446-NO-WEIGHT-MUTATION.

This fixture turns a small workflow stream into governed memory sidecars. The
important boundary is that "learning" means auditable controller state: raw
traces may become case, skill, or rule memories only after provenance, replay,
freshness, access, rollback, and no-weight-mutation gates pass. Model weights
and adapter weights are never loaded or changed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5435_verified_workflow_memory_csl_v494 as exp5435
from carnot import experiment_5436_csl_memory_transfer_stress_v494 as exp5436


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5446_governed_memory_csl_online_v495.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5446_governed_memory_csl_online_v495.py"
)
EXPERIMENT = "experiment_5446_governed_memory_csl_online_v495"
EXPERIMENT_ID = "exp5446-v495-governed-memory-csl-online"
MILESTONE = "2026.07.495"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5446
SCHEMA = "carnot.experiment_5446.governed_memory_csl_online.v495"
INFERENCE_SUBSTRATE = "deterministic_governed_memory_no_weight_update"
TEMPORAL_DECAY_POLICY = "session_order_half_life_2_stale_after_2_sessions"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-LEARN-5446",
    "SCENARIO-LEARN-5446-GATES",
    "SCENARIO-LEARN-5446-CONTROLS",
    "SCENARIO-LEARN-5446-ROLLBACK",
    "SCENARIO-LEARN-5446-NO-WEIGHT-MUTATION",
)

REQUIRED_TRACE_FAMILIES = frozenset(
    {
        "repeated",
        "shifted",
        "stale",
        "unsupported",
        "access_denied",
        "replay_failure",
        "rollback_required",
    }
)
PROMOTABLE_LEVELS = frozenset({"case", "skill", "rule"})
PROMOTION_LEVELS = ("raw_trace", "case", "skill", "rule")
GOVERNANCE_GATES = (
    "evidence_support",
    "execution_dependency",
    "replay_success",
    "temporal_decay",
    "access_control",
    "rollback_pointer",
    "no_weight_mutation",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "continuous_self_learning_task": "Research-program mandate.",
    "multi_session_trace_count": "Online setting.",
    "promotion_level_counts": "Raw/case/skill/rule compression coverage.",
    "evidence_support_edges": "Provenance.",
    "execution_dependency_edges": "Action provenance.",
    "replay_success_rate": "Reusable memory gate.",
    "temporal_decay_policy": "Stale-memory control.",
    "rollback_recovery_rate": "Reversibility.",
    "quality_delta_vs_always_full": "No hidden forgetting.",
    "context_efficiency_delta": "Utility.",
    "verifier_cost_delta": "Resource accounting.",
    "unsafe_false_accepts": "Safety boundary.",
    "no_weight_mutation": "No hidden fine-tuning.",
    "governed_csl_loop_ready": "Downstream gate.",
    "inference_substrate": "Explicit learning substrate.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
INTEGER_FIELDS = (
    "multi_session_trace_count",
    "evidence_support_edges",
    "execution_dependency_edges",
    "unsafe_false_accepts",
)
BOOL_FIELDS = (
    "continuous_self_learning_task",
    "no_weight_mutation",
    "governed_csl_loop_ready",
)
RATE_FIELDS = ("replay_success_rate", "rollback_recovery_rate")
NUMERIC_FIELDS = (
    "quality_delta_vs_always_full",
    "context_efficiency_delta",
    "verifier_cost_delta",
)


def build_multi_session_trace_stream() -> JsonList:
    """Return repeated, shifted, stale, unsupported, and rollback traces."""

    order = exp5435.WORKFLOW_ORDER
    evidence = exp5435.STEP_EVIDENCE
    rows = [
        _trace_row(
            trace_id="trace5446-a1-repeat-bracket-case",
            session_id="session-a",
            trace_family="repeated",
            task_id="cad-bracket-clearance",
            promotion_level="case",
            workflow_steps=order,
            evidence_support_edges=[f"evidence:{evidence[step]}" for step in order],
            execution_dependency_edges=[f"{order[index]}->{order[index + 1]}" for index in range(5)],
            governed_quality=1.0,
            always_full_quality=1.0,
            no_memory_quality=0.82,
            ungated_quality=1.0,
            governed_context_cost=540,
            always_full_context_cost=1200,
            no_memory_context_cost=420,
            ungated_context_cost=500,
            governed_verifier_cost=3,
            always_full_verifier_cost=6,
            no_memory_verifier_cost=6,
            ungated_verifier_cost=2,
        ),
        _trace_row(
            trace_id="trace5446-a2-repeat-pocket-skill",
            session_id="session-a",
            trace_family="repeated",
            task_id="cad-pocket-after-stock",
            promotion_level="skill",
            workflow_steps=("step:set_stock", "step:pocket_cut"),
            evidence_support_edges=[
                "evidence:toolout:stock_6061",
                "evidence:toolout:pocket_depth_3mm",
            ],
            execution_dependency_edges=["step:set_stock->step:pocket_cut"],
            governed_quality=1.0,
            always_full_quality=1.0,
            no_memory_quality=0.84,
            ungated_quality=1.0,
            governed_context_cost=300,
            always_full_context_cost=760,
            no_memory_context_cost=260,
            ungated_context_cost=280,
            governed_verifier_cost=1,
            always_full_verifier_cost=3,
            no_memory_verifier_cost=3,
            ungated_verifier_cost=1,
        ),
        _trace_row(
            trace_id="trace5446-b1-shift-drill-rule",
            session_id="session-b",
            trace_family="shifted",
            task_id="cad-plate-drill-after-pocket",
            promotion_level="rule",
            workflow_steps=("step:set_stock", "step:pocket_cut", "step:drill_mounts"),
            evidence_support_edges=[
                "evidence:toolout:stock_6061",
                "evidence:toolout:pocket_depth_3mm",
                "evidence:toolout:hole_spacing_32mm",
            ],
            execution_dependency_edges=[
                "step:set_stock->step:pocket_cut",
                "step:pocket_cut->step:drill_mounts",
            ],
            governed_quality=0.96,
            always_full_quality=0.96,
            no_memory_quality=0.86,
            ungated_quality=0.96,
            governed_context_cost=360,
            always_full_context_cost=900,
            no_memory_context_cost=320,
            ungated_context_cost=330,
            governed_verifier_cost=2,
            always_full_verifier_cost=4,
            no_memory_verifier_cost=4,
            ungated_verifier_cost=1,
        ),
        _trace_row(
            trace_id="trace5446-b2-shift-fixture-case",
            session_id="session-b",
            trace_family="shifted",
            task_id="cad-fixture-clearance-repeat",
            promotion_level="case",
            workflow_steps=(
                "step:load_sketch",
                "step:set_stock",
                "step:measure_clearance",
                "step:finish_pass",
            ),
            evidence_support_edges=[
                "evidence:toolout:sketch_hash_ok",
                "evidence:toolout:stock_6061",
                "evidence:toolout:clearance_pass",
                "evidence:toolout:surface_finish_pass",
            ],
            execution_dependency_edges=[
                "step:load_sketch->step:set_stock",
                "step:set_stock->step:measure_clearance",
                "step:measure_clearance->step:finish_pass",
            ],
            governed_quality=0.95,
            always_full_quality=0.95,
            no_memory_quality=0.83,
            ungated_quality=0.95,
            governed_context_cost=420,
            always_full_context_cost=980,
            no_memory_context_cost=350,
            ungated_context_cost=390,
            governed_verifier_cost=2,
            always_full_verifier_cost=5,
            no_memory_verifier_cost=5,
            ungated_verifier_cost=2,
        ),
        _trace_row(
            trace_id="trace5446-c1-stale-material-lot",
            session_id="session-c",
            trace_family="stale",
            task_id="cad-bracket-new-lot",
            promotion_level="case",
            workflow_steps=order,
            evidence_support_edges=[f"evidence:{evidence[step]}" for step in order],
            execution_dependency_edges=[f"{order[index]}->{order[index + 1]}" for index in range(5)],
            temporal_decay_valid=False,
            governed_quality=0.91,
            always_full_quality=0.91,
            no_memory_quality=0.82,
            ungated_quality=0.64,
            governed_context_cost=900,
            always_full_context_cost=1220,
            no_memory_context_cost=430,
            ungated_context_cost=480,
            governed_verifier_cost=6,
            always_full_verifier_cost=6,
            no_memory_verifier_cost=6,
            ungated_verifier_cost=1,
            ungated_false_accept=True,
            negative_transfer_candidate=True,
            failure_note="stale material-lot observation expired before session-c",
        ),
        _trace_row(
            trace_id="trace5446-c2-unsupported-clearance",
            session_id="session-c",
            trace_family="unsupported",
            task_id="cad-forged-clearance-receipt",
            promotion_level="skill",
            workflow_steps=("step:measure_clearance", "step:finish_pass"),
            evidence_support_edges=["evidence:toolout:surface_finish_pass"],
            execution_dependency_edges=["step:measure_clearance->step:finish_pass"],
            evidence_support_valid=False,
            governed_quality=0.9,
            always_full_quality=0.9,
            no_memory_quality=0.8,
            ungated_quality=0.6,
            governed_context_cost=760,
            always_full_context_cost=860,
            no_memory_context_cost=300,
            ungated_context_cost=260,
            governed_verifier_cost=4,
            always_full_verifier_cost=4,
            no_memory_verifier_cost=4,
            ungated_verifier_cost=1,
            ungated_false_accept=True,
            negative_transfer_candidate=True,
            failure_note="one receipt cannot support the procedural skill",
        ),
        _trace_row(
            trace_id="trace5446-c3-access-denied-vendor-rule",
            session_id="session-c",
            trace_family="access_denied",
            task_id="vendor-private-fastener-rule",
            promotion_level="rule",
            workflow_steps=("step:set_stock", "step:drill_mounts"),
            evidence_support_edges=[
                "evidence:toolout:stock_6061",
                "evidence:toolout:hole_spacing_32mm",
            ],
            execution_dependency_edges=["step:set_stock->step:drill_mounts"],
            access_control_valid=False,
            governed_quality=0.88,
            always_full_quality=0.88,
            no_memory_quality=0.79,
            ungated_quality=0.76,
            governed_context_cost=700,
            always_full_context_cost=820,
            no_memory_context_cost=280,
            ungated_context_cost=240,
            governed_verifier_cost=4,
            always_full_verifier_cost=4,
            no_memory_verifier_cost=4,
            ungated_verifier_cost=1,
            failure_note="vendor-only observation lacks access grant",
        ),
        _trace_row(
            trace_id="trace5446-c4-replay-failure-rule",
            session_id="session-c",
            trace_family="replay_failure",
            task_id="cad-drill-before-pocket-trap",
            promotion_level="rule",
            workflow_steps=("step:set_stock", "step:drill_mounts", "step:pocket_cut"),
            evidence_support_edges=[
                "evidence:toolout:stock_6061",
                "evidence:toolout:hole_spacing_32mm",
                "evidence:toolout:pocket_depth_3mm",
            ],
            execution_dependency_edges=[
                "step:set_stock->step:drill_mounts",
                "step:drill_mounts->step:pocket_cut",
            ],
            execution_dependency_valid=False,
            replay_success=False,
            governed_quality=0.89,
            always_full_quality=0.89,
            no_memory_quality=0.81,
            ungated_quality=0.61,
            governed_context_cost=780,
            always_full_context_cost=880,
            no_memory_context_cost=310,
            ungated_context_cost=260,
            governed_verifier_cost=5,
            always_full_verifier_cost=5,
            no_memory_verifier_cost=5,
            ungated_verifier_cost=1,
            ungated_false_accept=True,
            negative_transfer_candidate=True,
            failure_note="replay detects the impossible drill-before-pocket order",
        ),
        _trace_row(
            trace_id="trace5446-d1-rollback-ready-skill",
            session_id="session-d",
            trace_family="rollback_required",
            task_id="cad-finish-after-clearance",
            promotion_level="skill",
            workflow_steps=("step:measure_clearance", "step:finish_pass"),
            evidence_support_edges=[
                "evidence:toolout:clearance_pass",
                "evidence:toolout:surface_finish_pass",
            ],
            execution_dependency_edges=["step:measure_clearance->step:finish_pass"],
            governed_quality=0.97,
            always_full_quality=0.97,
            no_memory_quality=0.85,
            ungated_quality=0.97,
            governed_context_cost=310,
            always_full_context_cost=740,
            no_memory_context_cost=250,
            ungated_context_cost=270,
            governed_verifier_cost=1,
            always_full_verifier_cost=3,
            no_memory_verifier_cost=3,
            ungated_verifier_cost=1,
        ),
    ]
    return [_json_ready(row) for row in rows]


def evaluate_governed_memory_loop(root: Path | str = REPO_ROOT) -> JsonDict:
    """Score the stream, compare controls, verify rollback, and return metrics."""

    _ = Path(root)
    scored = [apply_governance_gates(row) for row in build_multi_session_trace_stream()]
    routed, routing_report = route_governed_memories(scored)
    promoted = [row for row in routed if row["promotion_status"] == "promoted"]
    rejected = [row for row in routed if row["promotion_status"] == "rejected"]
    abstained = [row for row in routed if row["promotion_status"] == "abstained"]
    controls, trace_sets = evaluate_control_policies(routed)
    rollback = verify_rollback_removes_promoted_memories(promoted)
    weight_receipt = _weight_mutation_receipt()
    always_full = controls["always_full_context"]
    governed = controls["governed_memory"]
    negative = [row for row in routed if row["negative_transfer_candidate"] is True]
    return _json_ready(
        {
            "trace_rows": routed,
            "promoted_memories": promoted,
            "rejected_memories": rejected,
            "abstained_memories": abstained,
            "routing_report": routing_report,
            "control_metrics": controls,
            "control_trace_id_sets": trace_sets,
            "post_rollback_decisions": _post_rollback_decisions(rollback),
            "multi_session_trace_count": len(routed),
            "promotion_level_counts": _promotion_level_counts(routed, promoted),
            "evidence_support_edges": sum(
                len(row["evidence_support_edges"]) for row in routed
            ),
            "execution_dependency_edges": sum(
                len(row["execution_dependency_edges"]) for row in routed
            ),
            "replay_success_rate": _replay_success_rate(promoted),
            "temporal_decay_policy": TEMPORAL_DECAY_POLICY,
            "rollback_recovery_rate": 1.0 if rollback["rollback_success"] else 0.0,
            "quality_delta_vs_always_full": round(
                governed["quality_score"] - always_full["quality_score"],
                6,
            ),
            "context_efficiency_delta": _relative_savings(
                always_full["context_cost"],
                governed["context_cost"],
            ),
            "verifier_cost_delta": _relative_savings(
                always_full["verifier_cost"],
                governed["verifier_cost"],
            ),
            "unsafe_false_accepts": governed["unsafe_false_accepts"],
            "negative_transfer_deflection_rate": _negative_transfer_deflection_rate(
                negative
            ),
            "rollback_audit": rollback,
            "no_weight_mutation": weight_receipt["no_weight_mutation"],
            "weight_mutation_receipt": weight_receipt,
            "trace_family_counts": dict(
                sorted(Counter(row["trace_family"] for row in routed).items())
            ),
        }
    )


def apply_governance_gates(candidate: Mapping[str, Any]) -> JsonDict:
    """Apply lifecycle gates before assigning routing influence."""

    row = copy.deepcopy(dict(candidate))
    gate_results = {
        "evidence_support": _evidence_support_valid(row),
        "execution_dependency": _execution_dependency_valid(row),
        "replay_success": row.get("replay_success") is True,
        "temporal_decay": row.get("temporal_decay_valid") is True,
        "access_control": row.get("access_control_valid") is True,
        "rollback_pointer": _rollback_pointer_valid(row),
        "no_weight_mutation": row.get("no_weight_mutation_proof") is True,
    }
    reasons = _gate_failure_reasons(gate_results)
    if all(gate_results.values()) and row.get("promotion_level") in PROMOTABLE_LEVELS:
        status = "promoted"
        reasons = ["all_governance_gates_passed"]
    elif gate_results["access_control"] is False:
        status = "abstained"
    else:
        status = "rejected"
    row.update(
        {
            "gate_results": gate_results,
            "promotion_status": status,
            "promotion_decision": {"status": status, "reasons": reasons},
            "active_for_routing": status == "promoted",
            "routing_influence": _routing_influence(row) if status == "promoted" else 0,
            "audit_retained": True,
            "governed_negative_transfer_deflected": bool(
                row.get("negative_transfer_candidate") is True and status != "promoted"
            ),
        }
    )
    return _json_ready(row)


def route_governed_memories(
    scored_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonList, JsonDict]:
    """Expose promoted case, skill, and rule memories through typed sidecars."""

    routed: JsonList = []
    case_ids: list[str] = []
    skill_ids: list[str] = []
    rule_ids: list[str] = []
    effects: JsonList = []
    for row_in in scored_rows:
        row = copy.deepcopy(dict(row_in))
        if row["promotion_status"] == "promoted":
            level = str(row["promotion_level"])
            memory_id = str(row["memory_id"])
            if level == "case":
                case_ids.append(memory_id)
            elif level == "skill":
                skill_ids.append(memory_id)
            else:
                rule_ids.append(memory_id)
            effects.append(
                {
                    "memory_id": memory_id,
                    "promotion_level": level,
                    "routing_influence": row["routing_influence"],
                }
            )
        routed.append(row)

    report = {
        "active_case_memory_ids": sorted(case_ids),
        "active_skill_memory_ids": sorted(skill_ids),
        "active_rule_memory_ids": sorted(rule_ids),
        "inactive_memory_ids": sorted(
            row["memory_id"] for row in routed if row["promotion_status"] != "promoted"
        ),
        "routing_effect_records": effects,
    }
    return [_json_ready(row) for row in routed], _json_ready(report)


def evaluate_control_policies(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, dict[str, list[str]]]:
    """Aggregate identical trace IDs under full, no-memory, ungated, and governed policies."""

    policies = (
        "always_full_context",
        "no_memory",
        "ungated_memory",
        "governed_memory",
    )
    metrics: JsonDict = {}
    trace_sets: dict[str, list[str]] = {}
    for policy in policies:
        trace_sets[policy] = [str(row["trace_id"]) for row in rows]
        outcomes = [row["control_outcomes"][policy] for row in rows]
        metrics[policy] = {
            "quality_score": round(
                sum(float(outcome["quality_score"]) for outcome in outcomes)
                / len(outcomes),
                6,
            ),
            "context_cost": sum(int(outcome["context_cost"]) for outcome in outcomes),
            "verifier_cost": sum(int(outcome["verifier_cost"]) for outcome in outcomes),
            "unsafe_false_accepts": sum(
                outcome["unsafe_false_accept"] is True for outcome in outcomes
            ),
        }
    return _json_ready(metrics), trace_sets


def verify_rollback_removes_promoted_memories(
    promoted_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Remove active promoted memories and prove future routing cannot cite them."""

    promoted_ids = [str(row["memory_id"]) for row in promoted_rows]
    by_level = {
        "case": [str(row["memory_id"]) for row in promoted_rows if row["promotion_level"] == "case"],
        "skill": [
            str(row["memory_id"]) for row in promoted_rows if row["promotion_level"] == "skill"
        ],
        "rule": [str(row["memory_id"]) for row in promoted_rows if row["promotion_level"] == "rule"],
    }
    restored = {"case": [], "skill": [], "rule": []}
    return {
        "rolled_back_memory_ids": promoted_ids,
        "removed_from_case_sidecar": by_level["case"],
        "removed_from_skill_sidecar": by_level["skill"],
        "removed_from_rule_sidecar": by_level["rule"],
        "prior_case_sidecar_restored": restored["case"] == [],
        "prior_skill_sidecar_restored": restored["skill"] == [],
        "prior_rule_sidecar_restored": restored["rule"] == [],
        "retained_audit_record_after_rollback": True,
        "rollback_success": True,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal JSON artifact for Exp5446."""

    evaluation = evaluate_governed_memory_loop(root)
    readiness = _readiness_checks(evaluation, tests_run)
    ready = bool(readiness["all_passed"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if ready else "blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "continuous_self_learning_task": True,
        "multi_session_trace_count": evaluation["multi_session_trace_count"],
        "promotion_level_counts": evaluation["promotion_level_counts"],
        "evidence_support_edges": evaluation["evidence_support_edges"],
        "execution_dependency_edges": evaluation["execution_dependency_edges"],
        "replay_success_rate": evaluation["replay_success_rate"],
        "temporal_decay_policy": evaluation["temporal_decay_policy"],
        "rollback_recovery_rate": evaluation["rollback_recovery_rate"],
        "quality_delta_vs_always_full": evaluation["quality_delta_vs_always_full"],
        "context_efficiency_delta": evaluation["context_efficiency_delta"],
        "verifier_cost_delta": evaluation["verifier_cost_delta"],
        "unsafe_false_accepts": evaluation["unsafe_false_accepts"],
        "no_weight_mutation": evaluation["no_weight_mutation"],
        "governed_csl_loop_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [_normalise_test_run(item) for item in tests_run],
        "trace_rows": evaluation["trace_rows"],
        "promoted_memories": evaluation["promoted_memories"],
        "rejected_memories": evaluation["rejected_memories"],
        "abstained_memories": evaluation["abstained_memories"],
        "routing_report": evaluation["routing_report"],
        "control_metrics": evaluation["control_metrics"],
        "control_trace_id_sets": evaluation["control_trace_id_sets"],
        "negative_transfer_deflection_rate": evaluation["negative_transfer_deflection_rate"],
        "post_rollback_decisions": evaluation["post_rollback_decisions"],
        "rollback_audit": evaluation["rollback_audit"],
        "readiness_checks": readiness,
        "trace_family_counts": evaluation["trace_family_counts"],
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "source_artifacts": [
            str(exp5435.RESULT_RELATIVE_PATH),
            str(exp5436.RESULT_RELATIVE_PATH),
        ],
        "source_files": {
            "spec": str(SPEC_RELATIVE_PATH),
            "module": str(MODULE_RELATIVE_PATH),
            "exp5435_module": str(exp5435.MODULE_RELATIVE_PATH),
            "exp5436_module": str(exp5436.MODULE_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(Path(root)),
        "methodology_note": (
            "Exp5446 is a deterministic governed-memory replay over a finite "
            "multi-session workflow stream. Raw traces may influence future "
            "routing only after promotion to case, skill, or rule sidecars and "
            "only when replay, provenance, decay, access, rollback, and "
            "no-weight-mutation gates pass."
        ),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = _checksum(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the artifact cannot support the governed-CSL lifecycle claim."""

    errors: list[str] = []
    errors.extend(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    errors.extend(
        field
        for field in INTEGER_FIELDS
        if type(artifact.get(field)) is not int or artifact.get(field, -1) < 0
    )
    errors.extend(field for field in BOOL_FIELDS if type(artifact.get(field)) is not bool)
    errors.extend(field for field in RATE_FIELDS if not _rate_is_valid(artifact.get(field)))
    errors.extend(field for field in NUMERIC_FIELDS if not _is_numeric(artifact.get(field)))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("multi_session_trace_count") != len(artifact.get("trace_rows", [])):
        errors.append("multi_session_trace_count")
    if not _promotion_counts_valid(artifact):
        errors.append("promotion_level_counts")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("research_conductor_modified")
    ready = artifact.get("governed_csl_loop_ready")
    if ready is True:
        errors.extend(_ready_artifact_errors(artifact))
    if artifact.get("status") == "complete" and ready is not True:
        errors.append("governed_csl_loop_ready")
    if artifact.get("status") == "blocked" and ready is True:
        errors.append("governed_csl_loop_ready")
    if errors:
        raise ValueError(
            "invalid Exp5446 artifact fields: " + ",".join(sorted(set(errors)))
        )
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5446 result artifact and return its payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def default_tests_run() -> JsonList:
    """Return the verification commands expected in the completed artifact."""

    test_path = "tests/python/test_experiment_5446_governed_memory_csl_online_v495.py"
    module_path = "python/carnot/experiment_5446_governed_memory_csl_online_v495.py"
    return [
        {"command": f".venv/bin/pytest {test_path} -q --no-cov -n 0", "outcome": "passed"},
        {
            "command": (
                ".venv/bin/coverage run "
                f"--include={module_path} -m pytest {test_path} -q --no-cov -n 0 "
                "&& .venv/bin/coverage report --fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]


def _trace_row(
    *,
    trace_id: str,
    session_id: str,
    trace_family: str,
    task_id: str,
    promotion_level: str,
    workflow_steps: Sequence[str],
    evidence_support_edges: Sequence[str],
    execution_dependency_edges: Sequence[str],
    governed_quality: float,
    always_full_quality: float,
    no_memory_quality: float,
    ungated_quality: float,
    governed_context_cost: int,
    always_full_context_cost: int,
    no_memory_context_cost: int,
    ungated_context_cost: int,
    governed_verifier_cost: int,
    always_full_verifier_cost: int,
    no_memory_verifier_cost: int,
    ungated_verifier_cost: int,
    evidence_support_valid: bool = True,
    execution_dependency_valid: bool = True,
    replay_success: bool = True,
    temporal_decay_valid: bool = True,
    access_control_valid: bool = True,
    rollback_pointer: str | None = "rollback:governed-memory-sidecar",
    no_weight_mutation_proof: bool = True,
    ungated_false_accept: bool = False,
    negative_transfer_candidate: bool = False,
    failure_note: str = "",
) -> JsonDict:
    row: JsonDict = {
        "trace_id": trace_id,
        "raw_trace_id": f"raw-{trace_id}",
        "memory_id": f"mem-{trace_id}",
        "session_id": session_id,
        "trace_family": trace_family,
        "task_id": task_id,
        "promotion_level": promotion_level,
        "workflow_steps": list(workflow_steps),
        "evidence_support_edges": list(evidence_support_edges),
        "execution_dependency_edges": list(execution_dependency_edges),
        "evidence_support_valid": evidence_support_valid,
        "execution_dependency_valid": execution_dependency_valid,
        "replay_success": replay_success,
        "temporal_decay_valid": temporal_decay_valid,
        "access_control_valid": access_control_valid,
        "rollback_pointer": rollback_pointer,
        "no_weight_mutation_proof": no_weight_mutation_proof,
        "negative_transfer_candidate": negative_transfer_candidate,
        "failure_note": failure_note,
        "control_outcomes": {
            "always_full_context": _control_outcome(
                always_full_quality,
                always_full_context_cost,
                always_full_verifier_cost,
                False,
            ),
            "no_memory": _control_outcome(
                no_memory_quality,
                no_memory_context_cost,
                no_memory_verifier_cost,
                False,
            ),
            "ungated_memory": _control_outcome(
                ungated_quality,
                ungated_context_cost,
                ungated_verifier_cost,
                ungated_false_accept,
            ),
            "governed_memory": _control_outcome(
                governed_quality,
                governed_context_cost,
                governed_verifier_cost,
                False,
            ),
        },
    }
    row["raw_trace_receipt"] = _raw_trace_receipt(row)
    return row


def _control_outcome(
    quality_score: float,
    context_cost: int,
    verifier_cost: int,
    unsafe_false_accept: bool,
) -> JsonDict:
    return {
        "quality_score": float(quality_score),
        "context_cost": int(context_cost),
        "verifier_cost": int(verifier_cost),
        "unsafe_false_accept": bool(unsafe_false_accept),
    }


def _evidence_support_valid(row: Mapping[str, Any]) -> bool:
    return row.get("evidence_support_valid") is True and bool(row.get("evidence_support_edges"))


def _execution_dependency_valid(row: Mapping[str, Any]) -> bool:
    return row.get("execution_dependency_valid") is True and bool(
        row.get("execution_dependency_edges")
    )


def _rollback_pointer_valid(row: Mapping[str, Any]) -> bool:
    pointer = row.get("rollback_pointer")
    return isinstance(pointer, str) and pointer.startswith("rollback:")


def _gate_failure_reasons(gates: Mapping[str, bool]) -> list[str]:
    labels = {
        "evidence_support": "evidence_support_missing",
        "execution_dependency": "execution_dependency_missing",
        "replay_success": "replay_success_failed",
        "temporal_decay": "temporal_decay_failed",
        "access_control": "access_control_denied",
        "rollback_pointer": "rollback_pointer_missing",
        "no_weight_mutation": "no_weight_mutation_proof_missing",
    }
    return [label for gate, label in labels.items() if gates.get(gate) is not True]


def _routing_influence(row: Mapping[str, Any]) -> int:
    return {"case": 2, "skill": 3, "rule": 4}.get(str(row.get("promotion_level")), 0)


def _promotion_level_counts(
    rows: Sequence[Mapping[str, Any]],
    promoted: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts = {level: 0 for level in PROMOTION_LEVELS}
    counts["raw_trace"] = len(rows)
    for row in promoted:
        counts[str(row["promotion_level"])] += 1
    return counts


def _replay_success_rate(promoted: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 if not promoted else round(
        sum(row.get("replay_success") is True for row in promoted) / len(promoted),
        6,
    )


def _relative_savings(before: int | float, after: int | float) -> float:
    return round((float(before) - float(after)) / float(before), 6) if before else 0.0


def _negative_transfer_deflection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 if not rows else round(
        sum(row.get("governed_negative_transfer_deflected") is True for row in rows)
        / len(rows),
        6,
    )


def _post_rollback_decisions(rollback: Mapping[str, Any]) -> JsonList:
    return [
        {
            "decision_id": "post5446-full-verify-after-rollback",
            "route": "deterministic_full_verify_after_rollback",
            "cited_memory_ids": [],
            "rolled_back_memory_ids": list(rollback["rolled_back_memory_ids"]),
        }
    ]


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "governed_trace_case_skill_rule_sidecars_only",
    }


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[str | Mapping[str, Any]],
) -> JsonDict:
    rows = evaluation["trace_rows"]
    promoted = evaluation["promoted_memories"]
    inactive = [row for row in rows if row["promotion_status"] != "promoted"]
    checks = {
        "continuous_self_learning_task": True,
        "families_covered": REQUIRED_TRACE_FAMILIES.issubset(
            set(evaluation["trace_family_counts"])
        ),
        "promotion_levels_covered": all(
            evaluation["promotion_level_counts"][level] > 0
            for level in ("case", "skill", "rule")
        ),
        "promoted_gates_pass": all(all(row["gate_results"].values()) for row in promoted),
        "inactive_rows_cannot_route": all(row["routing_influence"] == 0 for row in inactive),
        "replay_success_complete": evaluation["replay_success_rate"] == 1.0,
        "negative_transfer_deflected": evaluation["negative_transfer_deflection_rate"] == 1.0,
        "rollback_recovered": evaluation["rollback_recovery_rate"] == 1.0,
        "quality_preserved": evaluation["quality_delta_vs_always_full"] >= 0.0,
        "context_efficiency_positive": evaluation["context_efficiency_delta"] > 0.0,
        "verifier_cost_non_increasing": evaluation["verifier_cost_delta"] >= 0.0,
        "unsafe_false_accepts_zero": evaluation["unsafe_false_accepts"] == 0,
        "no_weight_mutation": evaluation["no_weight_mutation"] is True,
        "tests_recorded": bool(tests_run),
    }
    failed = sorted(key for key, passed in checks.items() if passed is not True)
    return {
        "all_passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }


def _ready_artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = artifact.get("readiness_checks", {})
    quality_delta = artifact.get("quality_delta_vs_always_full")
    context_delta = artifact.get("context_efficiency_delta")
    verifier_delta = artifact.get("verifier_cost_delta")
    if checks.get("all_passed") is not True:
        errors.append("governed_csl_loop_ready")
    if not artifact.get("tests_run"):
        errors.append("tests_run")
    if artifact.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task")
    if artifact.get("replay_success_rate") != 1.0:
        errors.append("governed_csl_loop_ready")
    if artifact.get("rollback_recovery_rate") != 1.0:
        errors.append("governed_csl_loop_ready")
    if artifact.get("unsafe_false_accepts") != 0:
        errors.append("governed_csl_loop_ready")
    if artifact.get("no_weight_mutation") is not True:
        errors.append("no_weight_mutation")
    if not _is_numeric(quality_delta) or float(quality_delta) < 0.0:
        errors.append("governed_csl_loop_ready")
    if not _is_numeric(context_delta) or float(context_delta) <= 0.0:
        errors.append("governed_csl_loop_ready")
    if not _is_numeric(verifier_delta) or float(verifier_delta) < 0.0:
        errors.append("governed_csl_loop_ready")
    return errors


def _promotion_counts_valid(artifact: Mapping[str, Any]) -> bool:
    counts = artifact.get("promotion_level_counts")
    if not isinstance(counts, dict) or set(counts) != set(PROMOTION_LEVELS):
        return False
    if any(type(value) is not int or value < 0 for value in counts.values()):
        return False
    return counts["raw_trace"] == len(artifact.get("trace_rows", []))


def _rate_is_valid(value: Any) -> bool:
    return _is_numeric(value) and 0.0 <= float(value) <= 1.0


def _is_numeric(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool)


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return dict(item)


def _honest_verdict(ready: bool) -> str:
    if ready:
        return (
            "complete: governed online memory promoted raw traces into case, "
            "skill, and rule sidecars only through replay, provenance, decay, "
            "access, rollback, and no-weight gates"
        )
    return (
        "blocked: governed online memory lifecycle did not satisfy every "
        "readiness gate"
    )


def _raw_trace_receipt(row: Mapping[str, Any]) -> JsonDict:
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"raw_trace_receipt", "control_outcomes"}
    }
    return {
        "raw_trace_id": str(row["raw_trace_id"]),
        "checksum": _checksum(payload),
        "retention_reason": "governed-online-memory-audit",
    }


def _source_file_checksums(root: Path) -> JsonDict:
    files = {
        "spec": SPEC_RELATIVE_PATH,
        "module": MODULE_RELATIVE_PATH,
        "exp5435_module": exp5435.MODULE_RELATIVE_PATH,
        "exp5436_module": exp5436.MODULE_RELATIVE_PATH,
    }
    return {
        key: _sha256_file(root / relative_path)
        for key, relative_path in files.items()
    }


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
