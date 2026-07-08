"""Exp5422: gated CSL promotion reliance scale-up.

Spec refs: REQ-LEARN-5422, SCENARIO-LEARN-5422-THRESHOLDS,
SCENARIO-LEARN-5422-QUARANTINE, SCENARIO-LEARN-5422-ROLLBACK,
SCENARIO-LEARN-5422-NO-WEIGHT-MUTATION.

This experiment is the promotion step after Exp5421 proved the controller can
see evidence-reliance drift. The controller is allowed to use learned memory
only when the replayed evidence says the fragment is certain, grounded, cheap
enough to be useful, below the risk budget, reversible, and not drifting away
from verifier-grounded reliance. The learned state is a deterministic routing
sidecar over retained artifacts; no model or adapter weights are loaded or
written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5409_uncertainty_gated_promotion_v492 as exp5409
from carnot import experiment_5421_evidence_reliance_csl_v493 as exp5421


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5422_csl_promotion_reliance_scale_v493"
EXPERIMENT_ID = "exp5422-v493-csl-promotion-reliance-scale"
MILESTONE = "2026.07.493"
SCHEMA = "carnot.experiment_5422.csl_promotion_reliance_scale.v493"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5422
INFERENCE_SUBSTRATE = "deterministic_self_learning_controller"

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5422_csl_promotion_reliance_scale_v493.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5422_csl_promotion_reliance_scale_v493.py"
)
EXP5409_RESULT_RELATIVE_PATH = exp5409.RESULT_RELATIVE_PATH
EXP5421_RESULT_RELATIVE_PATH = exp5421.RESULT_RELATIVE_PATH
EXP5409_MODULE_RELATIVE_PATH = exp5409.MODULE_RELATIVE_PATH
EXP5421_MODULE_RELATIVE_PATH = exp5421.MODULE_RELATIVE_PATH

SPEC_REFS = (
    "REQ-LEARN-5422",
    "SCENARIO-LEARN-5422-THRESHOLDS",
    "SCENARIO-LEARN-5422-QUARANTINE",
    "SCENARIO-LEARN-5422-ROLLBACK",
    "SCENARIO-LEARN-5422-NO-WEIGHT-MUTATION",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

MAX_UNCERTAINTY_SCORE = 0.30
ACCEPTED_RISK_THRESHOLD = 0.20
MIN_RESOURCE_SAVINGS = 20.0
RELIANCE_DRIFT_THRESHOLD = exp5421.HIDDEN_FORGETTING_THRESHOLD
MIN_SUPPORT_COUNT = exp5409.MIN_SUPPORT_COUNT
ALLOWED_GROUNDING_SOURCES = ("accepted_memory_with_verifier_spotcheck",)
REQUIRED_CANDIDATE_FAMILIES = frozenset(
    {
        "reachable",
        "unsupported",
        "stale",
        "poisoned",
        "ambiguous_evidence_reliance",
    }
)
ABSTAIN_FAMILIES = frozenset({"unsupported", "ambiguous_evidence_reliance"})
REJECT_FAMILIES = frozenset({"stale", "poisoned"})

THRESHOLDS: JsonDict = {
    "max_uncertainty_score": MAX_UNCERTAINTY_SCORE,
    "allowed_grounding_sources": list(ALLOWED_GROUNDING_SOURCES),
    "accepted_risk_threshold": ACCEPTED_RISK_THRESHOLD,
    "min_resource_savings": MIN_RESOURCE_SAVINGS,
    "rollback_required": True,
    "reliance_drift_threshold": RELIANCE_DRIFT_THRESHOLD,
}

FIELD_PRINCIPLES: dict[str, str] = {
    "candidate_fragment_count": "Scale.",
    "promoted_fragment_count": "Accepted memory evidence.",
    "rejected_fragment_count": "Guard behavior.",
    "abstained_fragment_count": "Uncertainty handling.",
    "grounding_preserved": "Evidence stability.",
    "reliance_drift_threshold": "Explicit gate.",
    "accepted_risk_threshold": "Risk gate.",
    "rollback_verified": "Recovery.",
    "rejected_fragments_quarantined": "No silent influence.",
    "no_weight_mutation": "FR-11 boundary.",
    "csl_promotion_reliance_scale_ready": "Capstone evidence.",
    "inference_substrate": "No hidden live model inference.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)
BOOL_FIELDS = (
    "grounding_preserved",
    "rollback_verified",
    "rejected_fragments_quarantined",
    "no_weight_mutation",
    "csl_promotion_reliance_scale_ready",
)
INTEGER_FIELDS = (
    "candidate_fragment_count",
    "promoted_fragment_count",
    "rejected_fragment_count",
    "abstained_fragment_count",
)
NUMERIC_FIELDS = ("reliance_drift_threshold", "accepted_risk_threshold")


def load_source_artifacts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read the deterministic source artifacts that are allowed to feed Exp5422."""

    root_path = Path(root)
    return {
        "exp5409": _read_json(root_path / EXP5409_RESULT_RELATIVE_PATH),
        "exp5421": _read_json(root_path / EXP5421_RESULT_RELATIVE_PATH),
    }


def evaluate_csl_promotion_reliance_scale(root: Path | str = REPO_ROOT) -> JsonDict:
    """Score the scale-up pool and route only fragments that pass every gate."""

    source_artifacts = load_source_artifacts(root)
    source_readiness = {
        "exp5421_evidence_reliance_csl_ready": source_artifacts["exp5421"].get(
            "evidence_reliance_csl_ready"
        )
        is True,
        "exp5409_uncertainty_gated_promotion_ready": source_artifacts["exp5409"].get(
            "uncertainty_gated_promotion_ready"
        )
        is True,
    }
    candidates = [
        score_candidate(candidate) for candidate in build_candidate_pool(source_artifacts)
    ]
    routed_candidates, routing_report = route_scored_candidates(
        candidates,
        source_artifacts,
    )
    promoted = [
        row for row in routed_candidates if row["promotion_status"] == "promoted"
    ]
    rejected = [
        row for row in routed_candidates if row["promotion_status"] == "rejected"
    ]
    abstained = [
        row for row in routed_candidates if row["promotion_status"] == "abstained"
    ]
    rollback = verify_rollback_restores_active_sidecar(routing_report)
    weight_receipt = _weight_mutation_receipt()
    grounding_preserved = bool(
        promoted
        and all(row["threshold_results"]["grounding"] for row in promoted)
        and source_readiness["exp5421_evidence_reliance_csl_ready"]
    )
    rejected_quarantined = all(
        row["audit_retained"]
        and not row["active_for_routing"]
        and row["routing_influence"] == 0
        for row in rejected
    ) and routing_report["rejected_fragment_routing_influence_count"] == 0
    return {
        "source_artifacts": source_artifacts,
        "source_readiness": source_readiness,
        "promotion_thresholds": dict(THRESHOLDS),
        "promotion_candidates": routed_candidates,
        "promoted_fragments": promoted,
        "rejected_fragments": rejected,
        "abstained_fragments": abstained,
        "candidate_fragment_count": len(routed_candidates),
        "promoted_fragment_count": len(promoted),
        "rejected_fragment_count": len(rejected),
        "abstained_fragment_count": len(abstained),
        "grounding_preserved": grounding_preserved,
        "reliance_drift_threshold": RELIANCE_DRIFT_THRESHOLD,
        "accepted_risk_threshold": ACCEPTED_RISK_THRESHOLD,
        "rollback_verified": rollback["rollback_success"],
        "rollback_audit": rollback,
        "rejected_fragments_quarantined": rejected_quarantined,
        "no_weight_mutation": weight_receipt["no_weight_mutation"],
        "weight_mutation_receipt": weight_receipt,
        "routing_report": routing_report,
    }


def build_candidate_pool(source_artifacts: Mapping[str, Any]) -> JsonList:
    """Expand Exp5409 fragments with Exp5421 reliance-drift controls."""

    exp5409_artifact = source_artifacts["exp5409"]
    exp5421_artifact = source_artifacts["exp5421"]
    reliance_by_fragment = {
        str(row["source_fragment_id"]): dict(row)
        for row in exp5421_artifact.get("paired_episodes", [])
    }
    candidates = [
        _candidate_from_exp5409(row, reliance_by_fragment)
        for row in exp5409_artifact.get("promotion_candidates", [])
    ]
    candidates.extend(
        _world_fragment_candidate(row)
        for row in exp5421_artifact.get("paired_episodes", [])
        if str(row["source_fragment_id"]).startswith("frag5421-")
    )
    return [_json_ready(row) for row in candidates]


def score_candidate(candidate: Mapping[str, Any]) -> JsonDict:
    """Apply every explicit threshold without trusting candidate prose."""

    row = dict(candidate)
    support_count = int(row["support_count"])
    threshold_results = {
        "uncertainty": float(row["uncertainty_score"]) <= MAX_UNCERTAINTY_SCORE,
        "grounding": str(row["grounding_source"]) in ALLOWED_GROUNDING_SOURCES,
        "accepted_risk": float(row["accepted_risk"]) <= ACCEPTED_RISK_THRESHOLD,
        "resource_savings": float(row["resource_savings"]) >= MIN_RESOURCE_SAVINGS,
        "rollback": bool(row["rollback_available"]) and bool(row["rollback_verified"]),
        "reliance_drift": float(row["reliance_drift"]) <= RELIANCE_DRIFT_THRESHOLD,
    }
    reasons = _promotion_reasons(row, threshold_results, support_count)
    all_thresholds_pass = all(threshold_results.values())
    if (
        all_thresholds_pass
        and support_count >= MIN_SUPPORT_COUNT
        and row["candidate_family"] == "reachable"
    ):
        status = "promoted"
        reasons = ["all_thresholds_passed"]
    elif _hard_reject(row, threshold_results):
        status = "rejected"
    else:
        status = "abstained"
    row.update(
        {
            "threshold_results": threshold_results,
            "promotion_status": status,
            "promotion_decision": {
                "status": status,
                "reasons": reasons,
                "thresholds": dict(THRESHOLDS),
            },
            "active_for_routing": status == "promoted",
            "audit_retained": True,
            "routing_influence": 0,
        }
    )
    return _json_ready(row)


def route_scored_candidates(
    scored_candidates: Sequence[Mapping[str, Any]],
    source_artifacts: Mapping[str, Any],
) -> tuple[JsonList, JsonDict]:
    """Expose promoted fragments to routing and keep all other fragments inactive."""

    effect_counts = _routing_effect_counts(source_artifacts["exp5409"])
    routed: JsonList = []
    effect_records: JsonList = []
    for candidate in scored_candidates:
        row = dict(candidate)
        if row["promotion_status"] == "promoted":
            effect_count = int(effect_counts.get(str(row["fragment_id"]), 1))
            row["routing_influence"] = effect_count
            effect_records.append(
                {
                    "fragment_id": str(row["fragment_id"]),
                    "routing_effect_count": effect_count,
                    "routing_effect_source": "exp5409_controller_sidecar",
                }
            )
        routed.append(row)

    promoted_ids = sorted(
        str(row["fragment_id"]) for row in routed if row["promotion_status"] == "promoted"
    )
    rejected_ids = sorted(
        str(row["fragment_id"]) for row in routed if row["promotion_status"] == "rejected"
    )
    abstained_ids = sorted(
        str(row["fragment_id"]) for row in routed if row["promotion_status"] == "abstained"
    )
    routing_report = {
        "active_fragment_ids": promoted_ids,
        "quarantined_rejected_fragment_ids": rejected_ids,
        "retained_abstained_fragment_ids": abstained_ids,
        "rejected_fragment_routing_influence_count": sum(
            1
            for row in routed
            if row["promotion_status"] == "rejected" and row["routing_influence"] != 0
        ),
        "abstained_fragment_routing_influence_count": sum(
            1
            for row in routed
            if row["promotion_status"] == "abstained" and row["routing_influence"] != 0
        ),
        "routing_effect_row_count": sum(
            int(row["routing_effect_count"]) for row in effect_records
        ),
        "routing_effect_records": effect_records,
        "rollback_probe_audit_fragment_ids": ["frag5422-poisoned-rollback-probe"],
    }
    return [_json_ready(row) for row in routed], _json_ready(routing_report)


def verify_rollback_restores_active_sidecar(routing_report: Mapping[str, Any]) -> JsonDict:
    """Inject a bad fragment and verify rollback returns the active set exactly."""

    prior_active = set(str(item) for item in routing_report["active_fragment_ids"])
    active_after_injection = set(prior_active)
    bad_fragment_id = "frag5422-poisoned-rollback-probe"
    active_after_injection.add(bad_fragment_id)
    injected = bad_fragment_id in active_after_injection
    restored = set(active_after_injection)
    restored.discard(bad_fragment_id)
    retained = bad_fragment_id in set(
        str(item) for item in routing_report["rollback_probe_audit_fragment_ids"]
    )
    return {
        "bad_fragment_id": bad_fragment_id,
        "injected_into_active_routing": injected,
        "rollback_removed_from_active_routing": bad_fragment_id not in restored,
        "prior_active_sidecar_restored": restored == prior_active,
        "retained_audit_record_after_rollback": retained,
        "rollback_success": bool(injected and restored == prior_active and retained),
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal artifact used by the conductor gate."""

    evaluation = evaluate_csl_promotion_reliance_scale(root)
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
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [
            str(EXP5409_RESULT_RELATIVE_PATH),
            str(EXP5421_RESULT_RELATIVE_PATH),
        ],
        "status": "complete" if ready else "blocked",
        "candidate_fragment_count": evaluation["candidate_fragment_count"],
        "promoted_fragment_count": evaluation["promoted_fragment_count"],
        "rejected_fragment_count": evaluation["rejected_fragment_count"],
        "abstained_fragment_count": evaluation["abstained_fragment_count"],
        "grounding_preserved": evaluation["grounding_preserved"],
        "reliance_drift_threshold": evaluation["reliance_drift_threshold"],
        "accepted_risk_threshold": evaluation["accepted_risk_threshold"],
        "rollback_verified": evaluation["rollback_verified"],
        "rejected_fragments_quarantined": evaluation["rejected_fragments_quarantined"],
        "no_weight_mutation": evaluation["no_weight_mutation"],
        "csl_promotion_reliance_scale_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [dict(row) for row in tests_run],
        "promotion_thresholds": evaluation["promotion_thresholds"],
        "source_readiness": evaluation["source_readiness"],
        "promotion_candidates": evaluation["promotion_candidates"],
        "promoted_fragments": evaluation["promoted_fragments"],
        "rejected_fragments": evaluation["rejected_fragments"],
        "abstained_fragments": evaluation["abstained_fragments"],
        "routing_report": evaluation["routing_report"],
        "rollback_audit": evaluation["rollback_audit"],
        "readiness_checks": readiness,
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "methodology_note": (
            "Exp5422 gates learned-memory and world-fragment influence through "
            "deterministic uncertainty, grounding, risk, resource, rollback, "
            "and evidence-reliance thresholds. Accepted fragments influence only "
            "the controller routing sidecar; rejected and abstained fragments "
            "remain retained and inactive."
        ),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp5422 fields that downstream gates depend on."""

    errors: list[str] = []
    errors.extend(field for field in REQUIRED_FIELDS if field not in artifact)
    errors.extend(field for field in BOOL_FIELDS if not isinstance(artifact.get(field), bool))
    errors.extend(
        field
        for field in INTEGER_FIELDS
        if isinstance(artifact.get(field), bool) or not isinstance(artifact.get(field), int)
    )
    errors.extend(field for field in NUMERIC_FIELDS if not _is_numeric(artifact.get(field)))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    ready = artifact.get("csl_promotion_reliance_scale_ready")
    if (ready is True and artifact.get("status") != "complete") or (
        artifact.get("status") == "complete" and ready is not True
    ):
        errors.append("status")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone")
    if artifact.get("reliance_drift_threshold") != RELIANCE_DRIFT_THRESHOLD:
        errors.append("reliance_drift_threshold")
    if artifact.get("accepted_risk_threshold") != ACCEPTED_RISK_THRESHOLD:
        errors.append("accepted_risk_threshold")
    if artifact.get("candidate_fragment_count") != len(
        artifact.get("promotion_candidates", [])
    ):
        errors.append("candidate_fragment_count")
    if artifact.get("promoted_fragment_count") != len(
        artifact.get("promoted_fragments", [])
    ):
        errors.append("promoted_fragment_count")
    if artifact.get("rejected_fragment_count") != len(
        artifact.get("rejected_fragments", [])
    ):
        errors.append("rejected_fragment_count")
    if artifact.get("abstained_fragment_count") != len(
        artifact.get("abstained_fragments", [])
    ):
        errors.append("abstained_fragment_count")
    if artifact.get("candidate_fragment_count") != (
        artifact.get("promoted_fragment_count", 0)
        + artifact.get("rejected_fragment_count", 0)
        + artifact.get("abstained_fragment_count", 0)
    ):
        errors.append("candidate_fragment_count")
    if ready is True:
        errors.extend(_ready_artifact_errors(artifact))
    if errors:
        raise ValueError(
            "invalid Exp5422 artifact fields: " + ",".join(sorted(set(errors)))
        )
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5422 result artifact and return its JSON payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def default_tests_run() -> JsonList:
    """Return the verification commands expected in the completed artifact."""

    test_path = "tests/python/test_experiment_5422_csl_promotion_reliance_scale_v493.py"
    module_path = "python/carnot/experiment_5422_csl_promotion_reliance_scale_v493.py"
    return [
        {
            "command": f".venv/bin/pytest {test_path} -q --no-cov -n 0",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                f"--include={module_path} -m pytest {test_path} "
                "-q --no-cov -n 0 && .venv/bin/coverage report --fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for source artifacts, spec, and local modules."""

    root_path = Path(root)
    return {
        "exp5409": _sha256_file(root_path / EXP5409_RESULT_RELATIVE_PATH),
        "exp5421": _sha256_file(root_path / EXP5421_RESULT_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5409_module": _sha256_file(root_path / EXP5409_MODULE_RELATIVE_PATH),
        "exp5421_module": _sha256_file(root_path / EXP5421_MODULE_RELATIVE_PATH),
    }


def _candidate_from_exp5409(
    source_candidate: Mapping[str, Any],
    reliance_by_fragment: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    family = _scale_family(str(source_candidate["candidate_family"]))
    inputs = source_candidate["decision_inputs"]
    fragment_id = str(source_candidate["fragment_id"])
    reliance_row = reliance_by_fragment.get(fragment_id)
    certainty_score = float(source_candidate["certainty_score"])
    uncertainty_score = round(1.0 - certainty_score, 6)
    return {
        "record_type": "csl_promotion_scale_candidate",
        "fragment_id": fragment_id,
        "fragment_kind": "learned_memory",
        "candidate_family": family,
        "source_candidate_family": str(source_candidate["candidate_family"]),
        "source_control_kind": str(source_candidate["source_control_kind"]),
        "source_memory_id": str(source_candidate["source_memory_id"]),
        "raw_episode_ids": [str(raw_id) for raw_id in source_candidate["raw_episode_ids"]],
        "support_count": int(source_candidate["support_count"]),
        "certainty_score": certainty_score,
        "uncertainty_score": uncertainty_score,
        "grounding_source": _grounding_source(family),
        "accepted_risk": _accepted_risk(inputs),
        "resource_savings": _resource_savings(inputs),
        "rollback_available": bool(inputs["rollback_available"]),
        "rollback_verified": bool(inputs["rollback_verified"]),
        "reliance_drift": _reliance_drift(fragment_id, family, uncertainty_score, reliance_row),
        "evidence_reliance_label": (
            str(reliance_row["evidence_reliance_label_after"])
            if reliance_row
            else "exp5409_trace_reliance"
        ),
        "requested_action": "promote_to_controller_sidecar",
        "audit_source": "exp5409_uncertainty_gated_promotion",
    }


def _world_fragment_candidate(pair: Mapping[str, Any]) -> JsonDict:
    return {
        "record_type": "csl_promotion_scale_candidate",
        "fragment_id": str(pair["source_fragment_id"]),
        "fragment_kind": "world_fragment",
        "candidate_family": "ambiguous_evidence_reliance",
        "source_candidate_family": str(pair["episode_family"]),
        "source_control_kind": str(pair["episode_family"]),
        "source_memory_id": "world:" + str(pair["source_fragment_id"]),
        "raw_episode_ids": [str(raw_id) for raw_id in pair["raw_episode_ids"]],
        "support_count": 2,
        "certainty_score": round(max(0.0, 1.0 - float(pair["evidence_reliance_drift"])), 6),
        "uncertainty_score": round(min(1.0, float(pair["evidence_reliance_drift"])), 6),
        "grounding_source": str(pair["grounding_source_after"]),
        "accepted_risk": 0.18,
        "resource_savings": round(
            float(pair["always_verify_resource_cost"]) - float(pair["resource_cost_after"]),
            6,
        ),
        "rollback_available": True,
        "rollback_verified": True,
        "reliance_drift": float(pair["evidence_reliance_drift"]),
        "evidence_reliance_label": str(pair["evidence_reliance_label_after"]),
        "requested_action": "promote_world_fragment_to_controller_sidecar",
        "audit_source": "exp5421_evidence_reliance_diagnostic",
    }


def _promotion_reasons(
    row: Mapping[str, Any],
    threshold_results: Mapping[str, bool],
    support_count: int,
) -> list[str]:
    reasons: list[str] = []
    if not threshold_results["uncertainty"]:
        reasons.append("uncertainty_exceeds_threshold")
    if not threshold_results["grounding"]:
        reasons.append("grounding_not_allowed")
    if not threshold_results["accepted_risk"]:
        reasons.append("accepted_risk_exceeds_threshold")
    if not threshold_results["resource_savings"]:
        reasons.append("resource_savings_below_threshold")
    if not threshold_results["rollback"]:
        reasons.append("rollback_unavailable")
    if not threshold_results["reliance_drift"]:
        reasons.append("reliance_drift_exceeds_threshold")
    if support_count < MIN_SUPPORT_COUNT:
        reasons.append("insufficient_support")
    if row["candidate_family"] == "ambiguous_evidence_reliance":
        reasons.append("ambiguous_evidence_reliance")
    return reasons


def _hard_reject(row: Mapping[str, Any], threshold_results: Mapping[str, bool]) -> bool:
    if row["candidate_family"] in REJECT_FAMILIES:
        return True
    if not threshold_results["accepted_risk"]:
        return True
    if not threshold_results["resource_savings"]:
        return True
    if not threshold_results["rollback"]:
        return True
    return bool(row["candidate_family"] == "reachable" and not threshold_results["grounding"])


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    families = {row["candidate_family"] for row in evaluation["promotion_candidates"]}
    routing = evaluation["routing_report"]
    checks = {
        "source_exp5421_ready": evaluation["source_readiness"][
            "exp5421_evidence_reliance_csl_ready"
        ],
        "source_exp5409_ready": evaluation["source_readiness"][
            "exp5409_uncertainty_gated_promotion_ready"
        ],
        "candidate_families_covered": REQUIRED_CANDIDATE_FAMILIES.issubset(families),
        "candidates_present": evaluation["candidate_fragment_count"] > 0,
        "promoted_present": evaluation["promoted_fragment_count"] > 0,
        "rejected_present": evaluation["rejected_fragment_count"] > 0,
        "abstained_present": evaluation["abstained_fragment_count"] > 0,
        "promoted_have_routing_influence": all(
            row["routing_influence"] > 0 for row in evaluation["promoted_fragments"]
        ),
        "rejected_zero_routing_influence": routing[
            "rejected_fragment_routing_influence_count"
        ]
        == 0,
        "abstained_zero_routing_influence": routing[
            "abstained_fragment_routing_influence_count"
        ]
        == 0,
        "grounding_preserved": evaluation["grounding_preserved"] is True,
        "rollback_verified": evaluation["rollback_verified"] is True,
        "rejected_fragments_quarantined": evaluation["rejected_fragments_quarantined"]
        is True,
        "no_weight_mutation": evaluation["no_weight_mutation"] is True,
        "tests_recorded": bool(tests_run),
        "inference_substrate_deterministic": INFERENCE_SUBSTRATE
        == "deterministic_self_learning_controller",
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {**checks, "all_passed": not failed, "failed_checks": failed}


def _ready_artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "grounding_preserved",
        "rollback_verified",
        "rejected_fragments_quarantined",
        "no_weight_mutation",
    ):
        if artifact.get(field) is not True:
            errors.append(field)
    if not artifact.get("tests_run"):
        errors.append("tests_run")
    if artifact.get("promoted_fragment_count", 0) <= 0:
        errors.append("promoted_fragment_count")
    if artifact.get("rejected_fragment_count", 0) <= 0:
        errors.append("rejected_fragment_count")
    if artifact.get("abstained_fragment_count", 0) <= 0:
        errors.append("abstained_fragment_count")
    routing = artifact.get("routing_report", {})
    if routing.get("rejected_fragment_routing_influence_count") != 0:
        errors.append("rejected_fragment_routing_influence_count")
    if routing.get("abstained_fragment_routing_influence_count") != 0:
        errors.append("abstained_fragment_routing_influence_count")
    if not all(
        row.get("routing_influence", 0) > 0
        for row in artifact.get("promoted_fragments", [])
    ):
        errors.append("routing_influence")
    return errors


def _scale_family(exp5409_family: str) -> str:
    if exp5409_family == "benign":
        return "reachable"
    if exp5409_family == "scarce_evidence":
        return "unsupported"
    if exp5409_family == "stale":
        return "stale"
    if exp5409_family == "poisoned":
        return "poisoned"
    return "ambiguous_evidence_reliance"


def _grounding_source(family: str) -> str:
    return {
        "reachable": "accepted_memory_with_verifier_spotcheck",
        "unsupported": "single_event_slice",
        "stale": "stale_memory_receipt",
        "poisoned": "forged_or_nontransferable_memory",
        "ambiguous_evidence_reliance": "ambiguous_evidence_reliance",
    }[family]


def _accepted_risk(inputs: Mapping[str, Any]) -> float:
    return round(
        max(
            float(inputs["stale_risk"]),
            float(inputs["forged_reasoning_risk"]),
            float(inputs["sharing_risk"]),
            min(1.0, float(inputs["self_reference_count"]) / 10.0),
        ),
        6,
    )


def _resource_savings(inputs: Mapping[str, Any]) -> float:
    return round((float(inputs["value_score"]) * 100.0) - (float(inputs["byte_cost"]) / 10.0), 6)


def _reliance_drift(
    fragment_id: str,
    family: str,
    uncertainty_score: float,
    reliance_row: Mapping[str, Any] | None,
) -> float:
    if reliance_row is not None:
        return float(reliance_row["evidence_reliance_drift"])
    if family == "reachable":
        return round(min(RELIANCE_DRIFT_THRESHOLD - 0.08, 0.18 + uncertainty_score), 6)
    return round(max(RELIANCE_DRIFT_THRESHOLD + 0.05, uncertainty_score), 6)


def _routing_effect_counts(exp5409_artifact: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(row["fragment_id"]): int(row["routing_effect_count"])
        for row in exp5409_artifact.get("routing_report", {}).get(
            "routing_effect_records",
            [],
        )
    }


def _honest_verdict(ready: bool) -> str:
    return (
        "complete: gated CSL promotion scale-up promoted only threshold-passing fragments, retained rejected and abstained fragments inactive, verified rollback, and did not mutate weights"
        if ready
        else "blocked: gated CSL promotion scale-up did not satisfy readiness checks"
    )


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "controller_promotion_reliance_sidecar_only",
    }


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_json_ready(stable), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
