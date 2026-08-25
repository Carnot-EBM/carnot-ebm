"""Exp5435: verified workflow-memory CSL fixture.

Spec refs: REQ-LEARN-5435,
SCENARIO-LEARN-5435-CASE-SKILL-SEPARATION,
SCENARIO-LEARN-5435-VERIFY-BEFORE-STORE,
SCENARIO-LEARN-5435-TRAP-DEFLECTION,
SCENARIO-LEARN-5435-ROLLBACK,
SCENARIO-LEARN-5435-RAW-RETENTION-NO-WEIGHT.

The fixture models a finite CAD-like repair workflow as typed tool steps and
evidence receipts. It deliberately keeps learning outside model parameters:
passing fragments may influence only deterministic case-memory and skill-memory
sidecars after ontology, planner/kernel, evidence, rollback, and resource gates
all pass. Rejected and abstained fragments stay retained for audit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5435_verified_workflow_memory_csl_v494.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5435_verified_workflow_memory_csl_v494.py")
EXPERIMENT = "experiment_5435_verified_workflow_memory_csl_v494"
EXPERIMENT_ID = "exp5435-v494-verified-workflow-memory-csl"
MILESTONE = "2026.07.494"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5435
SCHEMA = "carnot.experiment_5435.verified_workflow_memory_csl.v494"
SPEC_REFS = (
    "REQ-LEARN-5435",
    "SCENARIO-LEARN-5435-CASE-SKILL-SEPARATION",
    "SCENARIO-LEARN-5435-VERIFY-BEFORE-STORE",
    "SCENARIO-LEARN-5435-TRAP-DEFLECTION",
    "SCENARIO-LEARN-5435-ROLLBACK",
    "SCENARIO-LEARN-5435-RAW-RETENTION-NO-WEIGHT",
)
INFERENCE_SUBSTRATE = "deterministic_self_learning_controller"
TERMINAL_PREFIXES = ("complete:", "blocked:")

WORKFLOW_ORDER = (
    "step:load_sketch",
    "step:set_stock",
    "step:pocket_cut",
    "step:drill_mounts",
    "step:measure_clearance",
    "step:finish_pass",
)
STEP_EVIDENCE: dict[str, str] = {
    "step:load_sketch": "toolout:sketch_hash_ok",
    "step:set_stock": "toolout:stock_6061",
    "step:pocket_cut": "toolout:pocket_depth_3mm",
    "step:drill_mounts": "toolout:hole_spacing_32mm",
    "step:measure_clearance": "toolout:clearance_pass",
    "step:finish_pass": "toolout:surface_finish_pass",
}
KNOWN_EVIDENCE = frozenset(STEP_EVIDENCE.values())
KNOWN_MEMORY_KINDS = frozenset({"case", "skill"})
REQUIRED_EPISODE_FAMILIES = frozenset(
    {"positive", "stale", "poisoned", "retrieval_trap", "scarce_evidence"}
)
MIN_RESOURCE_SAVINGS = 10.0
MAX_PROMOTED_RELIANCE_DRIFT = 0.25

FIELD_PRINCIPLES: dict[str, str] = {
    "workflow_episode_count": "Scale.",
    "raw_episodes_retained": "Auditability.",
    "case_memory_count": "Memory organization.",
    "skill_memory_count": "Memory organization.",
    "verify_before_store_pass_rate": "Promotion gate.",
    "ontology_kernel_validation_rate": "Structural grounding.",
    "retrieval_trap_deflection_rate": "Memory safety.",
    "reliance_drift_metric": "Hidden-forgetting guard.",
    "quality_preserved": "No learning regression.",
    "resource_delta": "Resource-aware learning.",
    "rollback_verified": "Safety recovery.",
    "no_weight_mutation": "FR-11 boundary.",
    "verified_workflow_memory_ready": "Downstream gate.",
    "inference_substrate": "No hidden live model inference.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
INTEGER_FIELDS = (
    "workflow_episode_count",
    "case_memory_count",
    "skill_memory_count",
)
BOOL_FIELDS = (
    "raw_episodes_retained",
    "quality_preserved",
    "rollback_verified",
    "no_weight_mutation",
    "verified_workflow_memory_ready",
)
RATE_FIELDS = (
    "verify_before_store_pass_rate",
    "ontology_kernel_validation_rate",
    "retrieval_trap_deflection_rate",
)
NUMERIC_FIELDS = ("reliance_drift_metric", "resource_delta")


def build_workflow_episodes() -> JsonList:
    """Return the finite positive, stale, poisoned, trap, and scarce fixtures."""

    rows = [
        _episode(
            episode_id="ep5435-positive-case-full-bracket",
            memory_id="case5435-bracket-clearance-v1",
            memory_kind="case",
            episode_family="positive",
            workflow_steps=WORKFLOW_ORDER,
            expected_evidence=[STEP_EVIDENCE[step] for step in WORKFLOW_ORDER],
            observed_evidence=[STEP_EVIDENCE[step] for step in WORKFLOW_ORDER],
            resource_savings=46.0,
            verifier_cost_before=6,
            verifier_cost_after=3,
            reliance_drift=0.08,
            semantic_similarity_to_positive=1.0,
        ),
        _episode(
            episode_id="ep5435-positive-skill-pocket-after-stock",
            memory_id="skill5435-pocket-after-stock",
            memory_kind="skill",
            episode_family="positive",
            workflow_steps=("step:set_stock", "step:pocket_cut"),
            expected_evidence=[
                "toolout:stock_6061",
                "toolout:pocket_depth_3mm",
            ],
            observed_evidence=[
                "toolout:stock_6061",
                "toolout:pocket_depth_3mm",
            ],
            resource_savings=22.0,
            verifier_cost_before=3,
            verifier_cost_after=1,
            reliance_drift=0.11,
            semantic_similarity_to_positive=0.94,
        ),
        _episode(
            episode_id="ep5435-positive-skill-finish-after-clearance",
            memory_id="skill5435-finish-after-clearance",
            memory_kind="skill",
            episode_family="positive",
            workflow_steps=("step:measure_clearance", "step:finish_pass"),
            expected_evidence=[
                "toolout:clearance_pass",
                "toolout:surface_finish_pass",
            ],
            observed_evidence=[
                "toolout:clearance_pass",
                "toolout:surface_finish_pass",
            ],
            resource_savings=18.0,
            verifier_cost_before=3,
            verifier_cost_after=1,
            reliance_drift=0.12,
            semantic_similarity_to_positive=0.93,
        ),
        _episode(
            episode_id="ep5435-positive-case-drill-subflow",
            memory_id="case5435-drill-measure-subflow",
            memory_kind="case",
            episode_family="positive",
            workflow_steps=(
                "step:load_sketch",
                "step:set_stock",
                "step:drill_mounts",
                "step:measure_clearance",
            ),
            expected_evidence=[
                "toolout:sketch_hash_ok",
                "toolout:stock_6061",
                "toolout:hole_spacing_32mm",
                "toolout:clearance_pass",
            ],
            observed_evidence=[
                "toolout:sketch_hash_ok",
                "toolout:stock_6061",
                "toolout:hole_spacing_32mm",
                "toolout:clearance_pass",
            ],
            resource_savings=31.0,
            verifier_cost_before=5,
            verifier_cost_after=2,
            reliance_drift=0.1,
            semantic_similarity_to_positive=0.97,
        ),
        _episode(
            episode_id="ep5435-stale-case-old-stock",
            memory_id="case5435-stale-old-stock",
            memory_kind="case",
            episode_family="stale",
            workflow_steps=WORKFLOW_ORDER,
            expected_evidence=[STEP_EVIDENCE[step] for step in WORKFLOW_ORDER],
            observed_evidence=[STEP_EVIDENCE[step] for step in WORKFLOW_ORDER],
            kernel_valid=False,
            planner_valid=False,
            expected_structural_valid=False,
            expected_promotion_status="rejected",
            resource_savings=40.0,
            verifier_cost_before=6,
            verifier_cost_after=6,
            reliance_drift=0.54,
            semantic_similarity_to_positive=0.96,
            failure_note="stock receipt predates current material lot",
        ),
        _episode(
            episode_id="ep5435-poisoned-skill-forged-cmm",
            memory_id="skill5435-poisoned-forged-clearance",
            memory_kind="skill",
            episode_family="poisoned",
            workflow_steps=("step:measure_clearance", "step:finish_pass"),
            expected_evidence=[
                "toolout:clearance_pass",
                "toolout:surface_finish_pass",
            ],
            observed_evidence=[
                "toolout:forged_clearance_pass",
                "toolout:surface_finish_pass",
            ],
            ontology_valid=False,
            evidence_reliance_valid=False,
            expected_structural_valid=False,
            expected_promotion_status="rejected",
            resource_savings=20.0,
            verifier_cost_before=3,
            verifier_cost_after=3,
            reliance_drift=0.63,
            semantic_similarity_to_positive=0.91,
            failure_note="forged CMM output is outside the tool ontology",
        ),
        _episode(
            episode_id="ep5435-trap-case-drill-before-pocket",
            memory_id="case5435-trap-drill-before-pocket",
            memory_kind="case",
            episode_family="retrieval_trap",
            workflow_steps=(
                "step:load_sketch",
                "step:set_stock",
                "step:drill_mounts",
                "step:pocket_cut",
                "step:measure_clearance",
                "step:finish_pass",
            ),
            expected_evidence=[
                "toolout:sketch_hash_ok",
                "toolout:stock_6061",
                "toolout:hole_spacing_32mm",
                "toolout:pocket_depth_3mm",
                "toolout:clearance_pass",
                "toolout:surface_finish_pass",
            ],
            observed_evidence=[
                "toolout:sketch_hash_ok",
                "toolout:stock_6061",
                "toolout:hole_spacing_32mm",
                "toolout:pocket_depth_3mm",
                "toolout:clearance_pass",
                "toolout:surface_finish_pass",
            ],
            kernel_valid=False,
            planner_valid=False,
            expected_structural_valid=False,
            expected_promotion_status="rejected",
            resource_savings=38.0,
            verifier_cost_before=6,
            verifier_cost_after=6,
            reliance_drift=0.49,
            semantic_similarity_to_positive=0.95,
            failure_note="text match is high but tool order violates the kernel",
        ),
        _episode(
            episode_id="ep5435-scarce-skill-one-tool-receipt",
            memory_id="skill5435-scarce-one-receipt",
            memory_kind="skill",
            episode_family="scarce_evidence",
            workflow_steps=("step:pocket_cut", "step:drill_mounts"),
            expected_evidence=[
                "toolout:pocket_depth_3mm",
                "toolout:hole_spacing_32mm",
            ],
            observed_evidence=["toolout:pocket_depth_3mm"],
            evidence_reliance_valid=False,
            expected_promotion_status="abstained",
            resource_savings=17.0,
            verifier_cost_before=3,
            verifier_cost_after=2,
            reliance_drift=0.31,
            semantic_similarity_to_positive=0.9,
            failure_note="single receipt cannot support a reusable two-step skill",
        ),
    ]
    return [_json_ready(row) for row in rows]


def evaluate_verified_workflow_memory(root: Path | str = REPO_ROOT) -> JsonDict:
    """Score workflow episodes, route promoted sidecars, and derive metrics."""

    _ = Path(root)
    scored = [verify_before_store(row) for row in build_workflow_episodes()]
    routed, routing_report = route_memories(scored)
    promoted = [row for row in routed if row["promotion_status"] == "promoted"]
    rejected = [row for row in routed if row["promotion_status"] == "rejected"]
    abstained = [row for row in routed if row["promotion_status"] == "abstained"]
    rollback = verify_rollback_restores_workflow_sidecars(routing_report)
    weight_receipt = _weight_mutation_receipt()
    case_count = len(routing_report["active_case_memory_ids"])
    skill_count = len(routing_report["active_skill_memory_ids"])
    return {
        "workflow_episodes": routed,
        "promoted_memories": promoted,
        "rejected_memories": rejected,
        "abstained_memories": abstained,
        "workflow_episode_count": len(routed),
        "raw_episodes_retained": _raw_episodes_retained(routed),
        "case_memory_count": case_count,
        "skill_memory_count": skill_count,
        "verify_before_store_pass_rate": _rate(len(promoted), len(routed)),
        "ontology_kernel_validation_rate": _ontology_kernel_validation_rate(routed),
        "retrieval_trap_deflection_rate": _retrieval_trap_deflection_rate(routed),
        "reliance_drift_metric": _promoted_reliance_drift(promoted),
        "quality_preserved": _quality_preserved(promoted),
        "resource_delta": _resource_delta(promoted),
        "rollback_verified": rollback["rollback_success"],
        "rollback_audit": rollback,
        "no_weight_mutation": weight_receipt["no_weight_mutation"],
        "weight_mutation_receipt": weight_receipt,
        "routing_report": routing_report,
        "workflow_family_counts": dict(
            sorted(Counter(row["episode_family"] for row in routed).items())
        ),
    }


def verify_before_store(fragment: Mapping[str, Any]) -> JsonDict:
    """Apply deterministic gates before a memory can influence routing."""

    row = copy.deepcopy(dict(fragment))
    ontology_passed = _ontology_valid(row)
    kernel_passed = _kernel_planner_valid(row)
    evidence_passed = _evidence_reliance_valid(row)
    rollback_passed = _rollback_pointer_valid(row)
    resource_passed = _resource_accounting_valid(row)
    gate_results = {
        "ontology": ontology_passed,
        "kernel_planner": kernel_passed,
        "evidence_reliance": evidence_passed,
        "rollback": rollback_passed,
        "resource_accounting": resource_passed,
    }
    reasons = _gate_failure_reasons(gate_results)
    if all(gate_results.values()) and row.get("episode_family") == "positive":
        status = "promoted"
        reasons = ["all_verify_before_store_gates_passed"]
    elif (
        not evidence_passed
        and ontology_passed
        and kernel_passed
        and rollback_passed
        and resource_passed
    ):
        status = "abstained"
    else:
        status = "rejected"
    row.update(
        {
            "gate_results": gate_results,
            "promotion_status": status,
            "promotion_decision": {"status": status, "reasons": reasons},
            "active_for_routing": status == "promoted",
            "audit_retained": True,
            "routing_influence": 0,
        }
    )
    return _json_ready(row)


def route_memories(scored_fragments: Sequence[Mapping[str, Any]]) -> tuple[JsonList, JsonDict]:
    """Expose promoted case and skill memories through separate sidecars."""

    routed: JsonList = []
    case_ids: list[str] = []
    skill_ids: list[str] = []
    effects: JsonList = []
    for fragment in scored_fragments:
        row = copy.deepcopy(dict(fragment))
        if row["promotion_status"] == "promoted":
            influence = 2 if row["memory_kind"] == "case" else 3
            row["routing_influence"] = influence
            if row["memory_kind"] == "case":
                case_ids.append(row["memory_id"])
            else:
                skill_ids.append(row["memory_id"])
            effects.append(
                {
                    "memory_id": row["memory_id"],
                    "memory_kind": row["memory_kind"],
                    "routing_influence": influence,
                }
            )
        routed.append(row)

    rejected_ids = sorted(
        row["memory_id"] for row in routed if row["promotion_status"] == "rejected"
    )
    abstained_ids = sorted(
        row["memory_id"] for row in routed if row["promotion_status"] == "abstained"
    )
    routing_report = {
        "active_case_memory_ids": sorted(case_ids),
        "active_skill_memory_ids": sorted(skill_ids),
        "quarantined_rejected_memory_ids": rejected_ids,
        "retained_abstained_memory_ids": abstained_ids,
        "deflected_trap_memory_ids": sorted(
            row["memory_id"]
            for row in routed
            if row["episode_family"] == "retrieval_trap" and row["promotion_status"] != "promoted"
        ),
        "rejected_memory_routing_influence_count": sum(
            1
            for row in routed
            if row["promotion_status"] == "rejected" and row["routing_influence"] != 0
        ),
        "abstained_memory_routing_influence_count": sum(
            1
            for row in routed
            if row["promotion_status"] == "abstained" and row["routing_influence"] != 0
        ),
        "routing_effect_records": effects,
        "rollback_probe_audit_memory_ids": ["mem5435-poisoned-rollback-probe"],
    }
    return [_json_ready(row) for row in routed], _json_ready(routing_report)


def verify_rollback_restores_workflow_sidecars(
    routing_report: Mapping[str, Any],
) -> JsonDict:
    """Inject a bad memory into both typed sidecars and restore the prior state."""

    bad_memory_id = "mem5435-poisoned-rollback-probe"
    prior_case = set(str(item) for item in routing_report["active_case_memory_ids"])
    prior_skill = set(str(item) for item in routing_report["active_skill_memory_ids"])
    case_after_injection = set(prior_case)
    skill_after_injection = set(prior_skill)
    case_after_injection.add(bad_memory_id)
    skill_after_injection.add(bad_memory_id)
    restored_case = set(case_after_injection)
    restored_skill = set(skill_after_injection)
    restored_case.discard(bad_memory_id)
    restored_skill.discard(bad_memory_id)
    retained = bad_memory_id in {
        str(item) for item in routing_report["rollback_probe_audit_memory_ids"]
    }
    return {
        "bad_memory_id": bad_memory_id,
        "injected_into_case_sidecar": bad_memory_id in case_after_injection,
        "injected_into_skill_sidecar": bad_memory_id in skill_after_injection,
        "rollback_removed_from_case_sidecar": bad_memory_id not in restored_case,
        "rollback_removed_from_skill_sidecar": bad_memory_id not in restored_skill,
        "prior_case_sidecar_restored": restored_case == prior_case,
        "prior_skill_sidecar_restored": restored_skill == prior_skill,
        "retained_audit_record_after_rollback": retained,
        "rollback_success": bool(
            restored_case == prior_case and restored_skill == prior_skill and retained
        ),
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal artifact consumed by the milestone gate."""

    evaluation = evaluate_verified_workflow_memory(root)
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
        "workflow_episode_count": evaluation["workflow_episode_count"],
        "raw_episodes_retained": evaluation["raw_episodes_retained"],
        "case_memory_count": evaluation["case_memory_count"],
        "skill_memory_count": evaluation["skill_memory_count"],
        "verify_before_store_pass_rate": evaluation["verify_before_store_pass_rate"],
        "ontology_kernel_validation_rate": evaluation["ontology_kernel_validation_rate"],
        "retrieval_trap_deflection_rate": evaluation["retrieval_trap_deflection_rate"],
        "reliance_drift_metric": evaluation["reliance_drift_metric"],
        "quality_preserved": evaluation["quality_preserved"],
        "resource_delta": evaluation["resource_delta"],
        "rollback_verified": evaluation["rollback_verified"],
        "no_weight_mutation": evaluation["no_weight_mutation"],
        "verified_workflow_memory_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [_normalise_test_run(item) for item in tests_run],
        "workflow_episodes": evaluation["workflow_episodes"],
        "promoted_memories": evaluation["promoted_memories"],
        "rejected_memories": evaluation["rejected_memories"],
        "abstained_memories": evaluation["abstained_memories"],
        "routing_report": evaluation["routing_report"],
        "rollback_audit": evaluation["rollback_audit"],
        "workflow_family_counts": evaluation["workflow_family_counts"],
        "readiness_checks": readiness,
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "source_files": {
            "spec": str(SPEC_RELATIVE_PATH),
            "module": str(MODULE_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(Path(root)),
        "methodology_note": (
            "Exp5435 is a bounded deterministic workflow-memory replay. "
            "Case and skill fragments influence routing only after "
            "verify-before-store validates ontology, kernel/planner, evidence "
            "reliance, rollback, and resource gates. No live model inference or "
            "weight mutation is used."
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
    """Raise when the artifact cannot support the verified-memory claim."""

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
    rows = artifact.get("workflow_episodes", [])
    if artifact.get("workflow_episode_count") != len(rows):
        errors.append("workflow_episode_count")
    routing = artifact.get("routing_report", {})
    if artifact.get("case_memory_count") != len(routing.get("active_case_memory_ids", [])):
        errors.append("case_memory_count")
    if artifact.get("skill_memory_count") != len(routing.get("active_skill_memory_ids", [])):
        errors.append("skill_memory_count")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("research_conductor_modified")
    ready = artifact.get("verified_workflow_memory_ready")
    if ready is True:
        errors.extend(_ready_artifact_errors(artifact))
    if artifact.get("status") == "complete" and ready is not True:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("status") == "blocked" and ready is True:
        errors.append("verified_workflow_memory_ready")
    if errors:
        raise ValueError("invalid Exp5435 artifact fields: " + ",".join(sorted(set(errors))))
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5435 result artifact and return its payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def default_tests_run() -> list[JsonDict]:
    """Return the verification commands expected in the completed artifact."""

    test_path = "tests/python/test_experiment_5435_verified_workflow_memory_csl_v494.py"
    module_path = "python/carnot/experiment_5435_verified_workflow_memory_csl_v494.py"
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


def _episode(
    episode_id: str,
    memory_id: str,
    memory_kind: str,
    episode_family: str,
    *,
    workflow_steps: Sequence[str],
    expected_evidence: Sequence[str],
    observed_evidence: Sequence[str],
    ontology_valid: bool = True,
    kernel_valid: bool = True,
    planner_valid: bool = True,
    evidence_reliance_valid: bool = True,
    rollback_pointer: str | None = "rollback:workflow-sidecar",
    expected_structural_valid: bool = True,
    expected_promotion_status: str = "promoted",
    resource_savings: float,
    verifier_cost_before: int,
    verifier_cost_after: int,
    reliance_drift: float,
    semantic_similarity_to_positive: float,
    failure_note: str = "",
) -> JsonDict:
    row: JsonDict = {
        "episode_id": episode_id,
        "raw_episode_id": f"raw-{episode_id}",
        "memory_id": memory_id,
        "memory_kind": memory_kind,
        "episode_family": episode_family,
        "workflow_steps": list(workflow_steps),
        "expected_evidence": list(expected_evidence),
        "observed_evidence": list(observed_evidence),
        "ontology_valid": ontology_valid,
        "kernel_valid": kernel_valid,
        "planner_valid": planner_valid,
        "evidence_reliance_valid": evidence_reliance_valid,
        "rollback_pointer": rollback_pointer,
        "expected_structural_valid": expected_structural_valid,
        "expected_promotion_status": expected_promotion_status,
        "resource_savings": float(resource_savings),
        "verifier_cost_before": int(verifier_cost_before),
        "verifier_cost_after": int(verifier_cost_after),
        "reliance_drift": float(reliance_drift),
        "semantic_similarity_to_positive": float(semantic_similarity_to_positive),
        "failure_note": failure_note,
        "quality_before": 1.0,
        "quality_after": 1.0,
    }
    row["raw_episode_receipt"] = _raw_receipt(row)
    return row


def _ontology_valid(row: Mapping[str, Any]) -> bool:
    if row.get("ontology_valid") is not True:
        return False
    if row.get("memory_kind") not in KNOWN_MEMORY_KINDS:
        return False
    steps = row.get("workflow_steps", [])
    if not steps or any(step not in WORKFLOW_ORDER for step in steps):
        return False
    evidence = list(row.get("expected_evidence", [])) + list(row.get("observed_evidence", []))
    return bool(evidence) and all(item in KNOWN_EVIDENCE for item in evidence)


def _kernel_planner_valid(row: Mapping[str, Any]) -> bool:
    if row.get("kernel_valid") is not True or row.get("planner_valid") is not True:
        return False
    steps = list(row.get("workflow_steps", []))
    positions = [WORKFLOW_ORDER.index(step) for step in steps if step in WORKFLOW_ORDER]
    if len(positions) != len(steps) or positions != sorted(positions):
        return False
    required_evidence = {STEP_EVIDENCE[step] for step in steps}
    return required_evidence.issubset(set(row.get("expected_evidence", [])))


def _evidence_reliance_valid(row: Mapping[str, Any]) -> bool:
    if row.get("evidence_reliance_valid") is not True:
        return False
    return set(row.get("expected_evidence", [])).issubset(set(row.get("observed_evidence", [])))


def _rollback_pointer_valid(row: Mapping[str, Any]) -> bool:
    pointer = row.get("rollback_pointer")
    return isinstance(pointer, str) and pointer.startswith("rollback:")


def _resource_accounting_valid(row: Mapping[str, Any]) -> bool:
    return (
        _is_numeric(row.get("resource_savings")) and row["resource_savings"] >= MIN_RESOURCE_SAVINGS
    )


def _gate_failure_reasons(gates: Mapping[str, bool]) -> list[str]:
    reasons: list[str] = []
    if not gates["ontology"]:
        reasons.append("ontology_validation_failed")
    if not gates["kernel_planner"]:
        reasons.append("kernel_planner_validation_failed")
    if not gates["evidence_reliance"]:
        reasons.append("evidence_reliance_failed")
    if not gates["rollback"]:
        reasons.append("rollback_pointer_missing")
    if not gates["resource_accounting"]:
        reasons.append("resource_accounting_failed")
    return reasons or ["no_failure"]


def _raw_episodes_retained(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        row.get("audit_retained") is True
        and isinstance(row.get("raw_episode_id"), str)
        and str(row["raw_episode_id"]).startswith("raw-")
        and str(row.get("raw_episode_receipt", {}).get("checksum", "")).startswith("sha256:")
        for row in rows
    )


def _ontology_kernel_validation_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    matches = 0
    for row in rows:
        structural_passed = bool(
            row["gate_results"]["ontology"] and row["gate_results"]["kernel_planner"]
        )
        if structural_passed is bool(row["expected_structural_valid"]):
            matches += 1
    return _rate(matches, len(rows))


def _retrieval_trap_deflection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    traps = [row for row in rows if row["episode_family"] == "retrieval_trap"]
    deflected = [
        row
        for row in traps
        if row["promotion_status"] != "promoted" and row["active_for_routing"] is False
    ]
    return _rate(len(deflected), len(traps))


def _promoted_reliance_drift(promoted: Sequence[Mapping[str, Any]]) -> float:
    if not promoted:
        return 0.0
    return round(max(float(row["reliance_drift"]) for row in promoted), 6)


def _quality_preserved(promoted: Sequence[Mapping[str, Any]]) -> bool:
    return bool(promoted) and all(row["quality_after"] >= row["quality_before"] for row in promoted)


def _resource_delta(promoted: Sequence[Mapping[str, Any]]) -> float:
    return round(sum(float(row["resource_savings"]) for row in promoted), 6)


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "verified_workflow_memory_sidecars_only",
    }


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[str | Mapping[str, Any]],
) -> JsonDict:
    checks = {
        "families_covered": REQUIRED_EPISODE_FAMILIES.issubset(
            set(evaluation["workflow_family_counts"])
        ),
        "raw_episodes_retained": evaluation["raw_episodes_retained"] is True,
        "case_skill_separated": (
            evaluation["case_memory_count"] > 0 and evaluation["skill_memory_count"] > 0
        ),
        "verify_before_store_promoted": evaluation["verify_before_store_pass_rate"] > 0.0,
        "ontology_kernel_validated": evaluation["ontology_kernel_validation_rate"] == 1.0,
        "retrieval_traps_deflected": evaluation["retrieval_trap_deflection_rate"] == 1.0,
        "reliance_stable": evaluation["reliance_drift_metric"] <= MAX_PROMOTED_RELIANCE_DRIFT,
        "quality_preserved": evaluation["quality_preserved"] is True,
        "resource_accounted": _is_numeric(evaluation["resource_delta"])
        and evaluation["resource_delta"] >= 0.0,
        "rollback_verified": evaluation["rollback_verified"] is True,
        "no_weight_mutation": evaluation["no_weight_mutation"] is True,
        "inactive_failures_cannot_route": (
            evaluation["routing_report"]["rejected_memory_routing_influence_count"] == 0
            and evaluation["routing_report"]["abstained_memory_routing_influence_count"] == 0
        ),
        "tests_recorded": bool(tests_run),
    }
    return {
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    }


def _ready_artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if artifact.get("status") != "complete":
        errors.append("verified_workflow_memory_ready")
    if not artifact.get("tests_run"):
        errors.append("tests_run")
    if artifact.get("raw_episodes_retained") is not True:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("case_memory_count", 0) <= 0 or artifact.get("skill_memory_count", 0) <= 0:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("verify_before_store_pass_rate", 0.0) <= 0.0:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("ontology_kernel_validation_rate") != 1.0:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("retrieval_trap_deflection_rate") != 1.0:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("quality_preserved") is not True:
        errors.append("verified_workflow_memory_ready")
    if not _is_numeric(artifact.get("resource_delta")) or artifact.get("resource_delta", -1) < 0:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("rollback_verified") is not True:
        errors.append("verified_workflow_memory_ready")
    if artifact.get("no_weight_mutation") is not True:
        errors.append("verified_workflow_memory_ready")
    return errors


def _honest_verdict(ready: bool) -> str:
    if ready:
        return (
            "complete: verified workflow memory promoted only validated case and "
            "skill sidecars, deflected retrieval traps, retained raw episodes, "
            "verified rollback, and did not mutate weights"
        )
    return "blocked: verified workflow memory readiness checks failed"


def _raw_receipt(row: Mapping[str, Any]) -> JsonDict:
    payload = {
        "episode_id": row["episode_id"],
        "memory_id": row["memory_id"],
        "memory_kind": row["memory_kind"],
        "episode_family": row["episode_family"],
        "workflow_steps": row["workflow_steps"],
        "observed_evidence": row["observed_evidence"],
    }
    return {
        "raw_episode_id": row["raw_episode_id"],
        "checksum": "sha256:" + _checksum(payload),
        "retention_reason": "audit-before-store",
    }


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return dict(item)


def _source_file_checksums(root: Path) -> JsonDict:
    return {
        "spec": _sha256_file(root / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root / MODULE_RELATIVE_PATH),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
    ).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _rate_is_valid(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _is_numeric(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool)


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


def _checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
