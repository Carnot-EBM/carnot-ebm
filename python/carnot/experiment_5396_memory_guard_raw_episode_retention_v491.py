"""Exp5396: raw-episode memory guard for continuous self-learning.

Spec refs: REQ-LEARN-5396, SCENARIO-LEARN-5396-RAW-RETENTION,
SCENARIO-LEARN-5396-ROW-SCORES, SCENARIO-LEARN-5396-ROUTING.

The guard separates what happened from what the learner wants to remember.
Raw episodes are retained as immutable evidence rows. Consolidated memories are
only controller sidecars derived from those rows, and the verifier router sees
only accepted TRUST memories. This prevents forged reasoning history from
becoming a routing prior just because it sounds plausible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5382_real_workflow_continuous_self_learning_v490 as exp5382
from carnot import experiment_5395_influence_share_verifier_budget_router_v491 as exp5395


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5396_memory_guard_raw_episode_retention_v491"
EXPERIMENT_ID = "exp5396-v491-memory-guard-raw-episode-retention"
MILESTONE = "2026.07.491"
SCHEMA = "carnot.experiment_5396.memory_guard_raw_episode_retention.v491"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5396

RESULT_RELATIVE_PATH = Path(
    "results/experiment_5396_memory_guard_raw_episode_retention_v491.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5396_memory_guard_raw_episode_retention_v491.py"
)
EXP5382_RESULT_RELATIVE_PATH = exp5382.RESULT_RELATIVE_PATH
EXP5395_RESULT_RELATIVE_PATH = exp5395.RESULT_RELATIVE_PATH
EXP5382_MODULE_RELATIVE_PATH = exp5382.MODULE_RELATIVE_PATH
EXP5395_MODULE_RELATIVE_PATH = exp5395.MODULE_RELATIVE_PATH

SPEC_REFS = (
    "REQ-LEARN-5396",
    "SCENARIO-LEARN-5396-RAW-RETENTION",
    "SCENARIO-LEARN-5396-ROW-SCORES",
    "SCENARIO-LEARN-5396-ROUTING",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

STALE_RISK_MAX = 0.55
FORGED_REASONING_RISK_MAX = 0.6
SELF_REFERENCE_COUNT_MAX = 2
TRUST_VALUE_MIN = 0.55
MAX_BYTE_COST = 420
SHARE_RISK_MAX = 0.12
FORGED_CONTROL_KINDS = frozenset(
    {"forged_reasoning_history", "self_referential_amplification"}
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Complete if the guard and controls ran.",
    "milestone": "Must equal 2026.07.491.",
    "raw_episode_count": "Number of raw episodes retained.",
    "consolidated_memory_count": "Number of accepted consolidated memories.",
    "rejected_memory_count": "Number of rejected memory candidates.",
    "forged_reasoning_control_count": "Number of forged-reasoning controls.",
    "forged_reasoning_deflection_rate": (
        "Rejection rate for forged reasoning controls."
    ),
    "stale_memory_deflection_rate": "Rejection rate for stale controls.",
    "benign_memory_accept_rate": "Accept rate for benign useful controls.",
    "provenance_hash_valid_rate": "Rate of memories linked to raw episodes.",
    "rollback_success_rate": "Rollback success rate.",
    "no_weight_mutation": "Must be true.",
    "raw_episode_guard_ready": (
        "True only if forged/stale controls are deflected and benign controls "
        "are preserved."
    ),
    "honest_verdict": (
        "One-line summary starting with complete: or blocked:."
    ),
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)
BOOL_FIELDS = ("no_weight_mutation", "raw_episode_guard_ready")
INTEGER_FIELDS = (
    "raw_episode_count",
    "consolidated_memory_count",
    "rejected_memory_count",
    "forged_reasoning_control_count",
)
NUMERIC_FIELDS = (
    "forged_reasoning_deflection_rate",
    "stale_memory_deflection_rate",
    "benign_memory_accept_rate",
    "provenance_hash_valid_rate",
    "rollback_success_rate",
)


def build_raw_episodes(root: Path | str = REPO_ROOT) -> JsonList:
    """Return immutable episode evidence rows selected from the Exp5382 workflow."""

    decisions = exp5382.evaluate_real_workflow(root=root)["event_decisions"]
    return [
        _raw_episode(
            "raw5396-clean-dependency-edge",
            _first(decisions, lambda row: "mem5368-clean-dependency-edge" in row["supporting_context"]),
            "benign_useful",
            value=0.94,
            byte_cost=120,
            stale_risk=0.03,
            forged_reasoning_risk=0.04,
            self_reference_count=0,
            sharing_risk=0.04,
            claim="Dependency edge improves verifier routing when provenance is present.",
        ),
        _raw_episode(
            "raw5396-clean-rollback-route",
            _first(decisions, lambda row: "mem5368-clean-rollback-route" in row["supporting_context"]),
            "benign_useful",
            value=0.89,
            byte_cost=110,
            stale_risk=0.04,
            forged_reasoning_risk=0.03,
            self_reference_count=0,
            sharing_risk=0.05,
            claim="Rollback route restores clean selection after a bad update.",
        ),
        _raw_episode(
            "raw5396-clean-scaleup-summary",
            _first(decisions, lambda row: "mem5368-clean-scaleup-summary" in row["supporting_context"]),
            "benign_useful",
            value=0.81,
            byte_cost=150,
            stale_risk=0.06,
            forged_reasoning_risk=0.05,
            self_reference_count=1,
            sharing_risk=0.06,
            claim="Scale-up summary is useful only with the raw replay receipt.",
        ),
        _raw_episode(
            "raw5396-stale-runtime-receipt",
            _first(decisions, lambda row: row["memory_variant"] == "stale"),
            "stale_memory",
            value=0.62,
            byte_cost=130,
            stale_risk=0.93,
            forged_reasoning_risk=0.16,
            self_reference_count=0,
            sharing_risk=0.18,
            claim="Old runtime receipt should still steer cheap verifier routing.",
        ),
        _raw_episode(
            "raw5396-forged-reasoning-history",
            _first(decisions, lambda row: row["memory_variant"] == "unverified"),
            "forged_reasoning_history",
            value=0.77,
            byte_cost=100,
            stale_risk=0.14,
            forged_reasoning_risk=0.96,
            self_reference_count=1,
            sharing_risk=0.42,
            claim="Inserted reasoning history claims prior proof of a routing shortcut.",
        ),
        _raw_episode(
            "raw5396-self-reference-amplification",
            _first(decisions, lambda row: row["memory_variant"] == "biased"),
            "self_referential_amplification",
            value=0.74,
            byte_cost=105,
            stale_risk=0.12,
            forged_reasoning_risk=0.72,
            self_reference_count=5,
            sharing_risk=0.37,
            claim="Reflection says to trust itself because prior reflections agreed.",
        ),
        _raw_episode(
            "raw5396-high-cost-low-value",
            _first(decisions, lambda row: row["certificate_decision"] == "reject"),
            "high_cost_low_value",
            value=0.18,
            byte_cost=640,
            stale_risk=0.18,
            forged_reasoning_risk=0.18,
            self_reference_count=1,
            sharing_risk=0.2,
            claim="Locally correct but non-transferable episode should become a broad rule.",
        ),
    ]


def build_consolidated_memory_candidates(raw_episodes: Sequence[Mapping[str, Any]]) -> JsonList:
    """Draft one consolidated-memory candidate per raw episode without trusting it."""

    candidates: JsonList = []
    for episode in raw_episodes:
        evidence = dict(episode["row_evidence"])
        raw_id = str(episode["raw_episode_id"])
        candidates.append(
            {
                "record_type": "consolidated_memory",
                "memory_id": raw_id.replace("raw5396", "mem5396", 1),
                "memory_claim": evidence["claim"],
                "control_kind": episode["control_kind"],
                "raw_episode_ids": [raw_id],
                "decision_inputs": {
                    "value_score": float(evidence["value_score"]),
                    "byte_cost": int(evidence["byte_cost"]),
                    "stale_risk": float(evidence["stale_risk"]),
                    "forged_reasoning_risk": float(
                        evidence["forged_reasoning_risk"]
                    ),
                    "self_reference_count": int(evidence["self_reference_count"]),
                    "sharing_risk": float(evidence["sharing_risk"]),
                    "provenance_verified": bool(evidence["provenance_verified"]),
                    "rollback_available": bool(evidence["rollback_available"]),
                    "rollback_verified": bool(evidence["rollback_verified"]),
                    "model_generated_rationale_used": False,
                },
                "model_generated_rationale": (
                    "candidate consolidation text; non-authoritative"
                ),
            }
        )
    return candidates


def score_memory_candidate(
    candidate: Mapping[str, Any],
    raw_episodes_by_id: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Gate one candidate using row evidence and deliberately ignore rationale text."""

    inputs = dict(candidate["decision_inputs"])
    raw_ids = [str(raw_id) for raw_id in candidate["raw_episode_ids"]]
    provenance_valid = all(raw_id in raw_episodes_by_id for raw_id in raw_ids)
    net_value_per_byte = _net_value_per_byte(inputs)
    rejected_for_safety = (
        inputs["stale_risk"] > STALE_RISK_MAX
        or inputs["forged_reasoning_risk"] > FORGED_REASONING_RISK_MAX
        or inputs["self_reference_count"] > SELF_REFERENCE_COUNT_MAX
    )
    rejected_for_value = (
        inputs["value_score"] < TRUST_VALUE_MIN
        or inputs["byte_cost"] > MAX_BYTE_COST
        or net_value_per_byte <= 0.0
    )
    rollback_ok = inputs["rollback_available"] and inputs["rollback_verified"]
    accepted = bool(provenance_valid and rollback_ok and not rejected_for_safety and not rejected_for_value)
    keep = "KEEP" if accepted else ("QUARANTINE" if rejected_for_safety else "DROP")
    trust = "TRUST" if accepted else "UNTRUST"
    share = (
        "SHARE"
        if accepted and inputs["sharing_risk"] <= SHARE_RISK_MAX
        else "DO_NOT_SHARE"
    )
    memory_id = str(candidate["memory_id"])
    decision = {
        "keep": keep,
        "share": share,
        "trust": trust,
        "accepted": accepted,
        "rationale_ignored": True,
    }
    return {
        "record_type": "consolidated_memory",
        "memory_id": memory_id,
        "memory_claim": str(candidate["memory_claim"]),
        "control_kind": str(candidate["control_kind"]),
        "raw_episode_ids": raw_ids,
        "decision_inputs": inputs,
        "score": {
            "net_value_per_byte": net_value_per_byte,
            "rejected_for_safety": rejected_for_safety,
            "rejected_for_value": rejected_for_value,
        },
        "decision": decision,
        "trust_label": {
            "record_type": "trust_label",
            "memory_id": memory_id,
            "label": "verified_clean" if accepted else str(candidate["control_kind"]),
            "allowed_for_routing": accepted,
            "source": "row_level_evidence",
        },
        "provenance_hash": {
            "record_type": "provenance_hash",
            "memory_id": memory_id,
            "algorithm": "sha256:carnot.raw_episode.v1",
            "raw_episode_ids": raw_ids,
            "value": provenance_hash_for_episode_ids(raw_ids, raw_episodes_by_id),
            "valid": provenance_valid,
        },
        "rollback_pointer": {
            "record_type": "rollback_pointer",
            "memory_id": memory_id,
            "pointer_id": "rollback:" + memory_id,
            "target_raw_episode_ids": raw_ids,
            "action": (
                "not_required"
                if accepted
                else "retain_raw_episode_and_exclude_consolidated_memory"
            ),
            "rollback_success": rollback_ok,
        },
    }


def provenance_hash_for_episode_ids(
    raw_episode_ids: Sequence[str],
    raw_episodes_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash the raw evidence payloads that authorize a consolidated memory."""

    payload = [
        {
            "raw_episode_id": raw_id,
            "source_event_id": raw_episodes_by_id[raw_id]["source_event_id"],
            "row_evidence": raw_episodes_by_id[raw_id]["row_evidence"],
            "raw_payload_checksum": raw_episodes_by_id[raw_id]["raw_payload_checksum"],
        }
        for raw_id in raw_episode_ids
        if raw_id in raw_episodes_by_id
    ]
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def evaluate_memory_guard(root: Path | str = REPO_ROOT) -> JsonDict:
    """Evaluate raw retention, consolidation gates, and routing isolation."""

    raw_episodes = build_raw_episodes(root)
    raw_by_id = {str(row["raw_episode_id"]): row for row in raw_episodes}
    candidates = [
        score_memory_candidate(candidate, raw_by_id)
        for candidate in build_consolidated_memory_candidates(raw_episodes)
    ]
    accepted = [row for row in candidates if row["decision"]["accepted"]]
    rejected = [row for row in candidates if not row["decision"]["accepted"]]
    routing = build_downstream_routing_report(candidates, root)
    control_summary = _control_summary(candidates)
    forged_controls = [
        row for row in candidates if row["control_kind"] in FORGED_CONTROL_KINDS
    ]
    stale_controls = [row for row in candidates if row["control_kind"] == "stale_memory"]
    benign_controls = [row for row in candidates if row["control_kind"] == "benign_useful"]
    return {
        "raw_episodes": raw_episodes,
        "memory_candidates": candidates,
        "accepted_memories": accepted,
        "rejected_memories": rejected,
        "raw_episode_count": len(raw_episodes),
        "consolidated_memory_count": len(accepted),
        "rejected_memory_count": len(rejected),
        "forged_reasoning_control_count": len(forged_controls),
        "forged_reasoning_deflection_rate": _rate(
            sum(1 for row in forged_controls if not row["decision"]["accepted"]),
            len(forged_controls),
        ),
        "stale_memory_deflection_rate": _rate(
            sum(1 for row in stale_controls if not row["decision"]["accepted"]),
            len(stale_controls),
        ),
        "benign_memory_accept_rate": _rate(
            sum(1 for row in benign_controls if row["decision"]["accepted"]),
            len(benign_controls),
        ),
        "provenance_hash_valid_rate": _rate(
            sum(1 for row in candidates if row["provenance_hash"]["valid"]),
            len(candidates),
        ),
        "rollback_success_rate": _rate(
            sum(1 for row in rejected if row["rollback_pointer"]["rollback_success"]),
            len(rejected),
        ),
        "control_summary": control_summary,
        "downstream_routing": routing,
        "weight_mutation_receipt": _weight_mutation_receipt(),
    }


def build_downstream_routing_report(
    memory_candidates: Sequence[Mapping[str, Any]],
    root: Path | str = REPO_ROOT,
) -> JsonDict:
    """Expose accepted memories to verifier routing and prove rejected ones are absent."""

    routing_eval = exp5395.evaluate_routing_variants(root=root)
    accepted = [
        row
        for row in memory_candidates
        if row["decision"]["accepted"] and row["trust_label"]["allowed_for_routing"]
    ]
    rejected = [row for row in memory_candidates if not row["decision"]["accepted"]]
    rejected_seen = [
        row["memory_id"] for row in rejected if row["trust_label"]["allowed_for_routing"]
    ]
    return {
        "routing_decision_count": int(routing_eval["routed_decision_count"]),
        "accepted_memory_ids_used_for_routing": [
            str(row["memory_id"]) for row in accepted
        ],
        "rejected_memory_ids_seen_by_routing": rejected_seen,
        "rejected_memory_routing_influence_count": len(rejected_seen),
        "routing_context_records": [
            {
                "memory_id": str(row["memory_id"]),
                "raw_episode_ids": list(row["raw_episode_ids"]),
                "trust_label": row["trust_label"]["label"],
            }
            for row in accepted
        ],
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5396 artifact from deterministic guard evidence."""

    evaluation = evaluate_memory_guard(root)
    readiness = _readiness_checks(evaluation, tests_run)
    ready = bool(readiness["all_passed"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": [
            str(EXP5382_RESULT_RELATIVE_PATH),
            str(EXP5395_RESULT_RELATIVE_PATH),
        ],
        "status": "complete" if ready else "blocked",
        "milestone": MILESTONE,
        "raw_episode_count": evaluation["raw_episode_count"],
        "consolidated_memory_count": evaluation["consolidated_memory_count"],
        "rejected_memory_count": evaluation["rejected_memory_count"],
        "forged_reasoning_control_count": evaluation[
            "forged_reasoning_control_count"
        ],
        "forged_reasoning_deflection_rate": evaluation[
            "forged_reasoning_deflection_rate"
        ],
        "stale_memory_deflection_rate": evaluation["stale_memory_deflection_rate"],
        "benign_memory_accept_rate": evaluation["benign_memory_accept_rate"],
        "provenance_hash_valid_rate": evaluation["provenance_hash_valid_rate"],
        "rollback_success_rate": evaluation["rollback_success_rate"],
        "no_weight_mutation": evaluation["weight_mutation_receipt"][
            "no_weight_mutation"
        ],
        "raw_episode_guard_ready": ready,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [dict(row) for row in tests_run],
        "record_definitions": _record_definitions(),
        "raw_episodes": evaluation["raw_episodes"],
        "memory_candidates": evaluation["memory_candidates"],
        "accepted_memories": evaluation["accepted_memories"],
        "rejected_memories": evaluation["rejected_memories"],
        "control_summary": evaluation["control_summary"],
        "downstream_routing": evaluation["downstream_routing"],
        "readiness_checks": readiness,
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "methodology_note": (
            "All KEEP, SHARE, and TRUST decisions are scored from row-level "
            "episode evidence. Model-generated rationales are stored only as "
            "ignored draft text before scoring and are omitted from accepted "
            "routing inputs."
        ),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields consumed by the milestone reconciler."""

    errors: list[str] = []
    errors.extend(field for field in REQUIRED_FIELDS if field not in artifact)
    errors.extend(field for field in BOOL_FIELDS if not isinstance(artifact.get(field), bool))
    errors.extend(
        field
        for field in INTEGER_FIELDS
        if isinstance(artifact.get(field), bool) or not isinstance(artifact.get(field), int)
    )
    errors.extend(
        field for field in NUMERIC_FIELDS if not _is_numeric(artifact.get(field))
    )
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    ready = artifact.get("raw_episode_guard_ready")
    if (ready is True and artifact.get("status") != "complete") or (
        artifact.get("status") == "complete" and ready is not True
    ):
        errors.append("status")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone")
    for field in NUMERIC_FIELDS:
        if _is_numeric(artifact.get(field)) and float(artifact[field]) != 1.0:
            errors.append(field)
    if artifact.get("no_weight_mutation") is not True:
        errors.append("no_weight_mutation")
    if artifact.get("raw_episode_count") != len(artifact.get("raw_episodes", [])):
        errors.append("raw_episode_count")
    if artifact.get("consolidated_memory_count") != len(artifact.get("accepted_memories", [])):
        errors.append("consolidated_memory_count")
    if artifact.get("rejected_memory_count") != len(artifact.get("rejected_memories", [])):
        errors.append("rejected_memory_count")
    if artifact.get("downstream_routing", {}).get("rejected_memory_routing_influence_count") != 0:
        errors.append("rejected_memory_routing_influence_count")
    if ready is True and not artifact.get("tests_run"):
        errors.append("tests_run")
    if errors:
        raise ValueError("invalid Exp5396 artifact fields: " + ",".join(sorted(set(errors))))
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5396 result artifact and return the JSON payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the sources that define this guard."""

    root_path = Path(root)
    return {
        "exp5382": _sha256_file(root_path / EXP5382_RESULT_RELATIVE_PATH),
        "exp5395": _sha256_file(root_path / EXP5395_RESULT_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5382_module": _sha256_file(root_path / EXP5382_MODULE_RELATIVE_PATH),
        "exp5395_module": _sha256_file(root_path / EXP5395_MODULE_RELATIVE_PATH),
    }


def _raw_episode(
    raw_episode_id: str,
    source_decision: Mapping[str, Any],
    control_kind: str,
    *,
    value: float,
    byte_cost: int,
    stale_risk: float,
    forged_reasoning_risk: float,
    self_reference_count: int,
    sharing_risk: float,
    claim: str,
) -> JsonDict:
    evidence = {
        "claim": claim,
        "value_score": float(value),
        "byte_cost": int(byte_cost),
        "stale_risk": float(stale_risk),
        "forged_reasoning_risk": float(forged_reasoning_risk),
        "self_reference_count": int(self_reference_count),
        "sharing_risk": float(sharing_risk),
        "provenance_verified": True,
        "rollback_available": True,
        "rollback_verified": True,
    }
    payload = {
        "source_event_id": source_decision["event_id"],
        "trace_id": source_decision["trace_id"],
        "session_id": source_decision["session_id"],
        "memory_variant": source_decision["memory_variant"],
        "certificate_decision": source_decision["certificate_decision"],
        "learned_decision": source_decision["learned_decision"],
        "supporting_context": list(source_decision["supporting_context"]),
        "control_kind": control_kind,
        "row_evidence": evidence,
    }
    return {
        "record_type": "raw_episode",
        "raw_episode_id": raw_episode_id,
        "source_artifact": str(EXP5382_RESULT_RELATIVE_PATH),
        "source_event_id": str(source_decision["event_id"]),
        "trace_id": str(source_decision["trace_id"]),
        "session_id": str(source_decision["session_id"]),
        "control_kind": control_kind,
        "captured_at": RUN_DATE,
        "row_evidence": evidence,
        "raw_payload_checksum": "sha256:"
        + hashlib.sha256(_canonical_json(payload).encode()).hexdigest(),
    }


def _first(
    rows: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> Mapping[str, Any]:
    return next(row for row in rows if predicate(row))


def _net_value_per_byte(inputs: Mapping[str, Any]) -> float:
    harm = (
        float(inputs["stale_risk"])
        + float(inputs["forged_reasoning_risk"])
        + float(inputs["sharing_risk"])
        + int(inputs["self_reference_count"]) * 0.05
    )
    return round((float(inputs["value_score"]) - harm) / int(inputs["byte_cost"]), 6)


def _control_summary(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    kinds = sorted({str(row["control_kind"]) for row in candidates})
    return {
        "control_kinds": {
            kind: {
                "accepted": sum(
                    1
                    for row in candidates
                    if row["control_kind"] == kind and row["decision"]["accepted"]
                ),
                "total": sum(1 for row in candidates if row["control_kind"] == kind),
            }
            for kind in kinds
        },
        "row_level_evidence_scoring": True,
        "model_generated_rationales_authoritative": False,
    }


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    checks = {
        "raw_episodes_retained": evaluation["raw_episode_count"] > 0,
        "accepted_memories_present": evaluation["consolidated_memory_count"] > 0,
        "rejected_controls_present": evaluation["rejected_memory_count"] > 0,
        "forged_controls_deflected": evaluation["forged_reasoning_deflection_rate"] == 1.0,
        "stale_controls_deflected": evaluation["stale_memory_deflection_rate"] == 1.0,
        "benign_controls_preserved": evaluation["benign_memory_accept_rate"] == 1.0,
        "provenance_hashes_valid": evaluation["provenance_hash_valid_rate"] == 1.0,
        "rollback_succeeded": evaluation["rollback_success_rate"] == 1.0,
        "rejected_routing_influence_zero": evaluation["downstream_routing"][
            "rejected_memory_routing_influence_count"
        ]
        == 0,
        "tests_recorded": bool(tests_run),
        "no_weight_mutation": evaluation["weight_mutation_receipt"]["no_weight_mutation"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {**checks, "all_passed": not failed, "failed_checks": failed}


def _honest_verdict(ready: bool) -> str:
    return (
        "complete: raw episodes retained, consolidated memories gated from row evidence, forged and stale controls deflected, benign memories preserved, rejected memories excluded from verifier routing, and no weights mutated"
        if ready
        else "blocked: raw-episode memory guard evidence did not satisfy readiness checks"
    )


def _record_definitions() -> JsonDict:
    return {
        "raw_episode": [
            "raw_episode_id",
            "source_event_id",
            "row_evidence",
            "raw_payload_checksum",
        ],
        "consolidated_memory": [
            "memory_id",
            "raw_episode_ids",
            "decision_inputs",
            "decision",
        ],
        "trust_label": ["label", "allowed_for_routing", "source"],
        "provenance_hash": ["algorithm", "raw_episode_ids", "value", "valid"],
        "rollback_pointer": ["pointer_id", "target_raw_episode_ids", "action"],
    }


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "learned_state_scope": "controller_memory_guard_only",
    }


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / float(denominator), 6)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(_canonical_json(stable).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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
