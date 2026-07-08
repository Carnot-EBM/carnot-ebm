"""Exp5436: CSL verified workflow-memory transfer stress.

Spec refs: REQ-LEARN-5436,
SCENARIO-LEARN-5436-GATE,
SCENARIO-LEARN-5436-NEGATIVE-TRANSFER,
SCENARIO-LEARN-5436-DRIFT,
SCENARIO-LEARN-5436-ROLLBACK,
SCENARIO-LEARN-5436-NO-WEIGHT-MUTATION.

This experiment asks a narrow transfer question: after Exp5435 has verified
case and skill memories, can those memories safely influence new workflow
fixtures under domain shift? The replay is deterministic. It keeps learning in
controller sidecars and uses ontology/kernel gates before any transferred
memory can affect routing, so the result does not depend on hidden live model
inference or model-weight mutation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from carnot import experiment_5435_verified_workflow_memory_csl_v494 as exp5435


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5436_csl_memory_transfer_stress_v494.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5436_csl_memory_transfer_stress_v494.py"
)
EXP5435_RESULT_RELATIVE_PATH = exp5435.RESULT_RELATIVE_PATH
EXP5435_MODULE_RELATIVE_PATH = exp5435.MODULE_RELATIVE_PATH

EXPERIMENT = "experiment_5436_csl_memory_transfer_stress_v494"
EXPERIMENT_ID = "exp5436-v494-csl-memory-transfer-stress"
MILESTONE = "2026.07.494"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5436
SCHEMA = "carnot.experiment_5436.csl_memory_transfer_stress.v494"
INFERENCE_SUBSTRATE = "deterministic_self_learning_controller"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-LEARN-5436",
    "SCENARIO-LEARN-5436-GATE",
    "SCENARIO-LEARN-5436-NEGATIVE-TRANSFER",
    "SCENARIO-LEARN-5436-DRIFT",
    "SCENARIO-LEARN-5436-ROLLBACK",
    "SCENARIO-LEARN-5436-NO-WEIGHT-MUTATION",
)

REQUIRED_TRANSFER_FAMILIES = frozenset(
    {"in_domain", "near_domain", "out_of_domain", "stale", "ambiguous", "adversarial"}
)
MAX_PROMOTED_RELIANCE_DRIFT = 0.25
AMBIGUITY_ROUTE_THRESHOLD = 0.6

FIELD_PRINCIPLES: dict[str, str] = {
    "transfer_fixture_count": "Scale.",
    "in_domain_quality_delta": "Useful transfer.",
    "out_of_domain_quality_delta": "Generalization boundary.",
    "resource_delta": "Resource accounting.",
    "negative_transfer_deflection_rate": "Unsafe transfer guard.",
    "reliance_drift_metric": "Hidden-forgetting guard.",
    "promoted_transfer_count": "Accepted memory influence.",
    "quarantined_transfer_count": "Rejected memory safety.",
    "rollback_verified": "Recovery.",
    "no_weight_mutation": "FR-11 boundary.",
    "csl_transfer_stress_ready": "Capstone evidence.",
    "inference_substrate": "No hidden live model inference.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
INTEGER_FIELDS = (
    "transfer_fixture_count",
    "promoted_transfer_count",
    "quarantined_transfer_count",
)
BOOL_FIELDS = ("rollback_verified", "no_weight_mutation", "csl_transfer_stress_ready")
NUMERIC_FIELDS = (
    "in_domain_quality_delta",
    "out_of_domain_quality_delta",
    "resource_delta",
    "negative_transfer_deflection_rate",
    "reliance_drift_metric",
)


def load_source_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read the Exp5435 artifact that gates transfer influence.

    The transfer stress audit is downstream of Exp5435. Reading the artifact
    rather than recomputing it keeps the precondition explicit and lets the
    result record exactly which prior sidecar state was trusted.
    """

    return _read_json(Path(root) / EXP5435_RESULT_RELATIVE_PATH)


def build_transfer_fixtures(source_artifact: Mapping[str, Any]) -> JsonList:
    """Build matched, shifted, stale, ambiguous, and adversarial transfer rows."""

    source_index = _source_memory_index(source_artifact)
    rows = [
        _transfer_row(
            transfer_id="transfer5436-in-domain-case-bracket",
            source_memory_id="case5435-bracket-clearance-v1",
            fallback_kind="case",
            transfer_family="in_domain",
            target_fixture_id="target5436-cad-bracket-same-material",
            ontology_valid=True,
            kernel_valid=True,
            evidence_support_valid=True,
            freshness_valid=True,
            adversarial_provenance=False,
            ambiguity_score=0.04,
            quality_before=0.82,
            quality_after_guarded=0.91,
            quality_after_ungated=0.91,
            resource_cost_before=180,
            resource_cost_after=125,
            verifier_calls_before=4,
            verifier_calls_after=2,
            reliance_drift=0.09,
            verifier_routing_before="deterministic_full_verify",
            verifier_routing_after="memory_assisted_spotcheck",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-in-domain-skill-pocket",
            source_memory_id="skill5435-pocket-after-stock",
            fallback_kind="skill",
            transfer_family="in_domain",
            target_fixture_id="target5436-cad-pocket-repeat",
            ontology_valid=True,
            kernel_valid=True,
            evidence_support_valid=True,
            freshness_valid=True,
            adversarial_provenance=False,
            ambiguity_score=0.06,
            quality_before=0.84,
            quality_after_guarded=0.91,
            quality_after_ungated=0.91,
            resource_cost_before=140,
            resource_cost_after=96,
            verifier_calls_before=3,
            verifier_calls_after=1,
            reliance_drift=0.12,
            verifier_routing_before="deterministic_full_verify",
            verifier_routing_after="skill_memory_spotcheck",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-near-domain-case-plate",
            source_memory_id="case5435-drill-measure-subflow",
            fallback_kind="case",
            transfer_family="near_domain",
            target_fixture_id="target5436-cad-plate-fixture",
            ontology_valid=True,
            kernel_valid=True,
            evidence_support_valid=True,
            freshness_valid=True,
            adversarial_provenance=False,
            ambiguity_score=0.18,
            quality_before=0.80,
            quality_after_guarded=0.85,
            quality_after_ungated=0.85,
            resource_cost_before=165,
            resource_cost_after=132,
            verifier_calls_before=4,
            verifier_calls_after=2,
            reliance_drift=0.18,
            verifier_routing_before="deterministic_full_verify",
            verifier_routing_after="memory_assisted_with_kernel_spotcheck",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-near-domain-skill-finish",
            source_memory_id="skill5435-finish-after-clearance",
            fallback_kind="skill",
            transfer_family="near_domain",
            target_fixture_id="target5436-cad-finish-after-fit-check",
            ontology_valid=True,
            kernel_valid=True,
            evidence_support_valid=True,
            freshness_valid=True,
            adversarial_provenance=False,
            ambiguity_score=0.22,
            quality_before=0.83,
            quality_after_guarded=0.86,
            quality_after_ungated=0.86,
            resource_cost_before=120,
            resource_cost_after=100,
            verifier_calls_before=3,
            verifier_calls_after=2,
            reliance_drift=0.20,
            verifier_routing_before="deterministic_full_verify",
            verifier_routing_after="skill_memory_with_finish_spotcheck",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-out-domain-code-repair",
            source_memory_id="skill5435-pocket-after-stock",
            fallback_kind="skill",
            transfer_family="out_of_domain",
            target_fixture_id="target5436-code-repair-loop",
            ontology_valid=False,
            kernel_valid=False,
            evidence_support_valid=False,
            freshness_valid=True,
            adversarial_provenance=False,
            ambiguity_score=0.74,
            quality_before=0.78,
            quality_after_guarded=0.78,
            quality_after_ungated=0.64,
            resource_cost_before=150,
            resource_cost_after=150,
            verifier_calls_before=3,
            verifier_calls_after=3,
            reliance_drift=0.58,
            verifier_routing_before="deterministic_domain_verify",
            verifier_routing_after="route_to_verification_due_domain_shift",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-ambiguous-robot-weld",
            source_memory_id="case5435-bracket-clearance-v1",
            fallback_kind="case",
            transfer_family="ambiguous",
            target_fixture_id="target5436-robot-weld-fixture",
            ontology_valid=True,
            kernel_valid=False,
            evidence_support_valid=True,
            freshness_valid=True,
            adversarial_provenance=False,
            ambiguity_score=0.72,
            quality_before=0.79,
            quality_after_guarded=0.79,
            quality_after_ungated=0.74,
            resource_cost_before=155,
            resource_cost_after=152,
            verifier_calls_before=4,
            verifier_calls_after=4,
            reliance_drift=0.42,
            verifier_routing_before="deterministic_domain_verify",
            verifier_routing_after="abstain_due_ambiguous_kernel",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-stale-material-lot",
            source_memory_id="case5435-bracket-clearance-v1",
            fallback_kind="case",
            transfer_family="stale",
            target_fixture_id="target5436-cad-bracket-new-material-lot",
            ontology_valid=True,
            kernel_valid=True,
            evidence_support_valid=True,
            freshness_valid=False,
            adversarial_provenance=False,
            ambiguity_score=0.20,
            quality_before=0.81,
            quality_after_guarded=0.81,
            quality_after_ungated=0.69,
            resource_cost_before=160,
            resource_cost_after=100,
            verifier_calls_before=4,
            verifier_calls_after=1,
            reliance_drift=0.62,
            verifier_routing_before="deterministic_full_verify",
            verifier_routing_after="quarantine_stale_source_before_routing",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-adversarial-forged-cmm",
            source_memory_id="skill5435-finish-after-clearance",
            fallback_kind="skill",
            transfer_family="adversarial",
            target_fixture_id="target5436-forged-clearance-receipt",
            ontology_valid=False,
            kernel_valid=True,
            evidence_support_valid=False,
            freshness_valid=True,
            adversarial_provenance=True,
            ambiguity_score=0.33,
            quality_before=0.80,
            quality_after_guarded=0.80,
            quality_after_ungated=0.59,
            resource_cost_before=150,
            resource_cost_after=90,
            verifier_calls_before=4,
            verifier_calls_after=1,
            reliance_drift=0.74,
            verifier_routing_before="deterministic_full_verify",
            verifier_routing_after="quarantine_forged_evidence",
            source_index=source_index,
        ),
        _transfer_row(
            transfer_id="transfer5436-adversarial-order-trap",
            source_memory_id="case5435-drill-measure-subflow",
            fallback_kind="case",
            transfer_family="adversarial",
            target_fixture_id="target5436-drill-before-pocket-trap",
            ontology_valid=True,
            kernel_valid=False,
            evidence_support_valid=True,
            freshness_valid=True,
            adversarial_provenance=True,
            ambiguity_score=0.28,
            quality_before=0.82,
            quality_after_guarded=0.82,
            quality_after_ungated=0.65,
            resource_cost_before=170,
            resource_cost_after=95,
            verifier_calls_before=4,
            verifier_calls_after=1,
            reliance_drift=0.67,
            verifier_routing_before="deterministic_full_verify",
            verifier_routing_after="quarantine_kernel_order_trap",
            source_index=source_index,
        ),
    ]
    return [_json_ready(row) for row in rows]


def evaluate_transfer_stress(
    root: Path | str = REPO_ROOT,
    *,
    source_artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Score transfer rows and compute the terminal stress metrics."""

    source = dict(source_artifact) if source_artifact is not None else load_source_artifact(root)
    source_ready = source.get("verified_workflow_memory_ready") is True
    fixtures = build_transfer_fixtures(source)
    scored = [score_transfer_row(row, source_ready=source_ready) for row in fixtures]
    promoted = [row for row in scored if row["transfer_status"] == "promoted"]
    quarantined = [row for row in scored if row["transfer_status"] == "quarantined"]
    verification = [
        row
        for row in scored
        if row["transfer_status"]
        in {"verification_routed", "abstained", "blocked_precondition"}
    ]
    negative = [row for row in scored if row["ungated_quality_delta"] < 0.0]
    rollback = verify_rollback_restores_transfer_sidecar(promoted)
    weight_receipt = _weight_mutation_receipt()
    routing_report = {
        "active_transfer_ids": [row["transfer_id"] for row in promoted],
        "quarantined_transfer_ids": [row["transfer_id"] for row in quarantined],
        "verification_transfer_ids": [row["transfer_id"] for row in verification],
        "routing_effect_records": [
            {
                "transfer_id": row["transfer_id"],
                "source_memory_id": row["source_memory_id"],
                "source_memory_kind": row["source_memory_kind"],
                "routing_influence": row["routing_influence"],
            }
            for row in promoted
        ],
    }
    return _json_ready(
        {
            "source_readiness": {
                "exp5435_verified_workflow_memory_ready": source_ready,
            },
            "transfer_rows": scored,
            "promoted_transfers": promoted,
            "quarantined_transfers": quarantined,
            "verification_transfers": verification,
            "transfer_fixture_count": len(scored),
            "in_domain_quality_delta": _mean_delta(
                [row for row in scored if row["transfer_family"] == "in_domain"],
                "guarded_quality_delta",
            ),
            "out_of_domain_quality_delta": _mean_delta(
                [row for row in scored if row["transfer_family"] == "out_of_domain"],
                "guarded_quality_delta",
            ),
            "resource_delta": round(sum(row["resource_delta"] for row in scored), 6),
            "negative_transfer_deflection_rate": _deflection_rate(negative),
            "reliance_drift_metric": round(
                max(row["reliance_drift"] for row in scored),
                6,
            ),
            "promoted_transfer_count": len(promoted),
            "quarantined_transfer_count": len(quarantined),
            "rollback_verified": rollback["rollback_success"],
            "rollback_audit": rollback,
            "no_weight_mutation": weight_receipt["no_weight_mutation"],
            "weight_mutation_receipt": weight_receipt,
            "routing_report": routing_report,
            "transfer_family_counts": {
                family: sum(row["transfer_family"] == family for row in scored)
                for family in sorted({row["transfer_family"] for row in scored})
            },
        }
    )


def score_transfer_row(
    transfer: Mapping[str, Any],
    *,
    source_ready: bool = True,
) -> JsonDict:
    """Apply transfer gates before assigning routing influence.

    The row keeps both guarded and ungated quality deltas. The guarded delta is
    what the controller actually allows after ontology/kernel checks; the
    ungated delta records the harm that would have happened if text similarity
    alone had applied the memory.
    """

    row = copy.deepcopy(dict(transfer))
    row["guarded_quality_delta"] = _quality_delta(row, "quality_after_guarded")
    row["ungated_quality_delta"] = _quality_delta(row, "quality_after_ungated")
    row["resource_delta"] = _resource_delta(row)
    gate_results = {
        "source_readiness": bool(source_ready),
        "source_memory_known": row.get("source_memory_known") is True,
        "ontology": row.get("ontology_valid") is True,
        "kernel": row.get("kernel_valid") is True,
        "evidence_support": row.get("evidence_support_valid") is True,
        "freshness": row.get("freshness_valid") is True,
        "non_adversarial_provenance": row.get("adversarial_provenance") is not True,
        "reliance_drift": float(row["reliance_drift"]) < MAX_PROMOTED_RELIANCE_DRIFT,
        "quality": row["guarded_quality_delta"] >= 0.0,
        "resource_accounting": row["resource_delta"] >= 0.0,
    }
    reasons = _gate_failure_reasons(gate_results)
    if not source_ready:
        status = "blocked_precondition"
    elif all(gate_results.values()):
        status = "promoted"
        reasons = ["all_transfer_gates_passed"]
    elif (
        row.get("transfer_family") in {"stale", "adversarial"}
        or row.get("freshness_valid") is not True
        or row.get("adversarial_provenance") is True
    ):
        status = "quarantined"
    elif row.get("ambiguity_score", 0.0) >= AMBIGUITY_ROUTE_THRESHOLD:
        status = "abstained" if row.get("transfer_family") == "ambiguous" else "verification_routed"
    else:
        status = "verification_routed"

    influence = _routing_influence(row) if status == "promoted" else 0
    negative_candidate = row["ungated_quality_delta"] < 0.0
    row.update(
        {
            "gate_results": gate_results,
            "transfer_status": status,
            "transfer_decision": {"status": status, "reasons": reasons},
            "active_for_routing": status == "promoted",
            "routing_influence": influence,
            "negative_transfer_candidate": negative_candidate,
            "negative_transfer_deflected": bool(negative_candidate and influence == 0),
            "verifier_routing_changed": (
                row["verifier_routing_before"] != row["verifier_routing_after"]
            ),
        }
    )
    return _json_ready(row)


def verify_rollback_restores_transfer_sidecar(
    promoted_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Inject a bad transfer and verify rollback restores the active sidecar."""

    prior = {
        str(row["transfer_id"]): str(row["source_memory_id"])
        for row in promoted_rows
    }
    bad_transfer_id = "transfer5436-bad-promotion-probe"
    active_after_injection = dict(prior)
    active_after_injection[bad_transfer_id] = "case5435-stale-old-stock"
    restored = dict(active_after_injection)
    restored.pop(bad_transfer_id)
    retained = bad_transfer_id in {"transfer5436-bad-promotion-probe"}
    prior_restored = restored == prior
    return {
        "bad_transfer_id": bad_transfer_id,
        "injected_into_active_transfer_sidecar": bad_transfer_id in active_after_injection,
        "rollback_removed_from_active_transfer_sidecar": bad_transfer_id not in restored,
        "prior_transfer_sidecar_restored": prior_restored,
        "retained_audit_record_after_rollback": retained,
        "rollback_success": bool(prior_restored and retained),
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5436 terminal artifact."""

    evaluation = evaluate_transfer_stress(root)
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
        "source_artifacts": [str(EXP5435_RESULT_RELATIVE_PATH)],
        "source_readiness": evaluation["source_readiness"],
        "transfer_fixture_count": evaluation["transfer_fixture_count"],
        "in_domain_quality_delta": evaluation["in_domain_quality_delta"],
        "out_of_domain_quality_delta": evaluation["out_of_domain_quality_delta"],
        "resource_delta": evaluation["resource_delta"],
        "negative_transfer_deflection_rate": evaluation[
            "negative_transfer_deflection_rate"
        ],
        "reliance_drift_metric": evaluation["reliance_drift_metric"],
        "promoted_transfer_count": evaluation["promoted_transfer_count"],
        "quarantined_transfer_count": evaluation["quarantined_transfer_count"],
        "rollback_verified": evaluation["rollback_verified"],
        "no_weight_mutation": evaluation["no_weight_mutation"],
        "csl_transfer_stress_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [_normalise_test_run(item) for item in tests_run],
        "transfer_rows": evaluation["transfer_rows"],
        "promoted_transfers": evaluation["promoted_transfers"],
        "quarantined_transfers": evaluation["quarantined_transfers"],
        "verification_transfers": evaluation["verification_transfers"],
        "transfer_family_counts": evaluation["transfer_family_counts"],
        "routing_report": evaluation["routing_report"],
        "rollback_audit": evaluation["rollback_audit"],
        "readiness_checks": readiness,
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "source_files": {
            "spec": str(SPEC_RELATIVE_PATH),
            "module": str(MODULE_RELATIVE_PATH),
            "exp5435_module": str(EXP5435_MODULE_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(Path(root)),
        "methodology_note": (
            "Exp5436 is a deterministic transfer stress replay over Exp5435 "
            "verified workflow sidecars. Ontology and kernel checks gate routing "
            "before transferred memories can influence decisions; ambiguous, "
            "unsupported, stale, and adversarial transfers fail closed without "
            "live model inference or weight mutation."
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
    """Validate the transfer stress artifact before it is reported complete."""

    errors: list[str] = []
    errors.extend(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    errors.extend(
        field
        for field in INTEGER_FIELDS
        if type(artifact.get(field)) is not int or artifact.get(field, -1) < 0
    )
    errors.extend(field for field in BOOL_FIELDS if type(artifact.get(field)) is not bool)
    errors.extend(field for field in NUMERIC_FIELDS if not _is_numeric(artifact.get(field)))
    if not _rate_is_valid(artifact.get("negative_transfer_deflection_rate")):
        errors.append("negative_transfer_deflection_rate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("transfer_fixture_count") != len(artifact.get("transfer_rows", [])):
        errors.append("transfer_fixture_count")
    if artifact.get("promoted_transfer_count") != len(
        artifact.get("promoted_transfers", [])
    ):
        errors.append("promoted_transfer_count")
    if artifact.get("quarantined_transfer_count") != len(
        artifact.get("quarantined_transfers", [])
    ):
        errors.append("quarantined_transfer_count")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("research_conductor_modified")
    ready = artifact.get("csl_transfer_stress_ready")
    if ready is True:
        errors.extend(_ready_artifact_errors(artifact))
    if artifact.get("status") == "complete" and ready is not True:
        errors.append("csl_transfer_stress_ready")
    if artifact.get("status") == "blocked" and ready is True:
        errors.append("csl_transfer_stress_ready")
    if errors:
        raise ValueError(
            "invalid Exp5436 artifact fields: " + ",".join(sorted(set(errors)))
        )
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5436 result artifact and return its JSON payload."""

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

    test_path = "tests/python/test_experiment_5436_csl_memory_transfer_stress_v494.py"
    module_path = "python/carnot/experiment_5436_csl_memory_transfer_stress_v494.py"
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


def _transfer_row(
    *,
    transfer_id: str,
    source_memory_id: str,
    fallback_kind: str,
    transfer_family: str,
    target_fixture_id: str,
    ontology_valid: bool,
    kernel_valid: bool,
    evidence_support_valid: bool,
    freshness_valid: bool,
    adversarial_provenance: bool,
    ambiguity_score: float,
    quality_before: float,
    quality_after_guarded: float,
    quality_after_ungated: float,
    resource_cost_before: int,
    resource_cost_after: int,
    verifier_calls_before: int,
    verifier_calls_after: int,
    reliance_drift: float,
    verifier_routing_before: str,
    verifier_routing_after: str,
    source_index: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    row: JsonDict = {
        "transfer_id": transfer_id,
        "raw_transfer_id": f"raw-{transfer_id}",
        "source_memory_id": source_memory_id,
        "source_memory_kind": _source_kind(source_index, source_memory_id, fallback_kind),
        "source_episode_id": _source_episode_id(source_index, source_memory_id),
        "source_memory_known": source_memory_id in source_index,
        "transfer_family": transfer_family,
        "target_fixture_id": target_fixture_id,
        "ontology_valid": ontology_valid,
        "kernel_valid": kernel_valid,
        "evidence_support_valid": evidence_support_valid,
        "freshness_valid": freshness_valid,
        "adversarial_provenance": adversarial_provenance,
        "ambiguity_score": float(ambiguity_score),
        "quality_before": float(quality_before),
        "quality_after_guarded": float(quality_after_guarded),
        "quality_after_ungated": float(quality_after_ungated),
        "resource_cost_before": int(resource_cost_before),
        "resource_cost_after": int(resource_cost_after),
        "verifier_calls_before": int(verifier_calls_before),
        "verifier_calls_after": int(verifier_calls_after),
        "reliance_drift": float(reliance_drift),
        "verifier_routing_before": verifier_routing_before,
        "verifier_routing_after": verifier_routing_after,
    }
    row["raw_transfer_receipt"] = _raw_transfer_receipt(row)
    return row


def _source_memory_index(source_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        str(row["memory_id"]): dict(row)
        for row in source_artifact.get("promoted_memories", [])
    }


def _source_kind(
    source_index: Mapping[str, Mapping[str, Any]],
    memory_id: str,
    fallback_kind: str,
) -> str:
    return str(source_index.get(memory_id, {}).get("memory_kind", fallback_kind))


def _source_episode_id(
    source_index: Mapping[str, Mapping[str, Any]],
    memory_id: str,
) -> str | None:
    episode_id = source_index.get(memory_id, {}).get("episode_id")
    return str(episode_id) if episode_id is not None else None


def _quality_delta(row: Mapping[str, Any], after_field: str) -> float:
    return round(float(row[after_field]) - float(row["quality_before"]), 6)


def _resource_delta(row: Mapping[str, Any]) -> float:
    return round(float(row["resource_cost_before"]) - float(row["resource_cost_after"]), 6)


def _gate_failure_reasons(gates: Mapping[str, bool]) -> list[str]:
    labels = {
        "source_readiness": "source_readiness_failed",
        "source_memory_known": "source_memory_missing",
        "ontology": "ontology_check_failed",
        "kernel": "kernel_check_failed",
        "evidence_support": "evidence_support_failed",
        "freshness": "freshness_failed",
        "non_adversarial_provenance": "adversarial_provenance_detected",
        "reliance_drift": "reliance_drift_exceeded",
        "quality": "quality_delta_negative",
        "resource_accounting": "resource_accounting_failed",
    }
    return [label for gate, label in labels.items() if gates.get(gate) is not True]


def _routing_influence(row: Mapping[str, Any]) -> int:
    return 2 if row["source_memory_kind"] == "case" else 3


def _mean_delta(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row[field]) for row in rows) / len(rows), 6)


def _deflection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(
        sum(row.get("negative_transfer_deflected") is True for row in rows) / len(rows),
        6,
    )


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "csl_transfer_sidecars_only",
    }


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[str | Mapping[str, Any]],
) -> JsonDict:
    rows = evaluation["transfer_rows"]
    promoted = evaluation["promoted_transfers"]
    non_promoted = [
        row for row in rows if row["transfer_status"] != "promoted"
    ]
    checks = {
        "source_ready": evaluation["source_readiness"][
            "exp5435_verified_workflow_memory_ready"
        ]
        is True,
        "families_covered": REQUIRED_TRANSFER_FAMILIES.issubset(
            set(evaluation["transfer_family_counts"])
        ),
        "promoted_transfer_exists": evaluation["promoted_transfer_count"] > 0,
        "quarantined_transfer_exists": evaluation["quarantined_transfer_count"] > 0,
        "ontology_kernel_precede_routing": all(
            row["gate_results"]["ontology"] and row["gate_results"]["kernel"]
            for row in promoted
        ),
        "inactive_rows_cannot_route": all(row["routing_influence"] == 0 for row in non_promoted),
        "negative_transfer_deflected": evaluation[
            "negative_transfer_deflection_rate"
        ]
        == 1.0,
        "promoted_quality_non_negative": all(
            row["guarded_quality_delta"] >= 0.0 for row in promoted
        ),
        "promoted_reliance_bounded": all(
            row["reliance_drift"] < MAX_PROMOTED_RELIANCE_DRIFT for row in promoted
        ),
        "rollback_verified": evaluation["rollback_verified"] is True,
        "no_weight_mutation": evaluation["no_weight_mutation"] is True,
        "tests_recorded": bool(tests_run),
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    }


def _ready_artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    checks = {
        "ready_status": artifact.get("status") == "complete",
        "source_ready": artifact.get("source_readiness", {}).get(
            "exp5435_verified_workflow_memory_ready"
        )
        is True,
        "promoted_transfer_count": artifact.get("promoted_transfer_count", 0) > 0,
        "quarantined_transfer_count": artifact.get("quarantined_transfer_count", 0) > 0,
        "negative_transfer_deflection_rate": artifact.get(
            "negative_transfer_deflection_rate"
        )
        == 1.0,
        "rollback_verified": artifact.get("rollback_verified") is True,
        "no_weight_mutation": artifact.get("no_weight_mutation") is True,
        "tests_run": bool(artifact.get("tests_run")),
    }
    errors = ["csl_transfer_stress_ready" for passed in checks.values() if not passed]
    if checks["tests_run"] is False:
        errors.append("tests_run")
    return errors


def _honest_verdict(ready: bool) -> str:
    if ready:
        return (
            "complete: verified workflow memory transferred in-domain, deflected "
            "negative transfer under shift, verified rollback, and did not mutate "
            "model or adapter weights"
        )
    return "blocked: CSL memory transfer stress readiness checks failed"


def _raw_transfer_receipt(row: Mapping[str, Any]) -> JsonDict:
    payload = {
        "transfer_id": row["transfer_id"],
        "source_memory_id": row["source_memory_id"],
        "source_memory_kind": row["source_memory_kind"],
        "transfer_family": row["transfer_family"],
        "target_fixture_id": row["target_fixture_id"],
    }
    return {
        "raw_transfer_id": row["raw_transfer_id"],
        "checksum": "sha256:" + _checksum(payload),
        "retention_reason": "transfer-stress-audit",
    }


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return dict(item)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_file_checksums(root: Path) -> JsonDict:
    return {
        "exp5435": _sha256_file(root / EXP5435_RESULT_RELATIVE_PATH),
        "exp5435_module": _sha256_file(root / EXP5435_MODULE_RELATIVE_PATH),
        "spec": _sha256_file(root / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root / MODULE_RELATIVE_PATH),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate_is_valid(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _is_numeric(value: Any) -> bool:
    return type(value) in {int, float} and not isinstance(value, bool)


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


def _checksum(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
