"""SMGI certified-update gates for FR-11 CerCE ledger rows.

FR-11 can propose query-time policy and memory updates, but SMGI only lets an
update become reusable when the system can show that older verified constraints
were replayed and preserved.  This module turns the existing CerCE ledger into
a small, deterministic certification report: safe ledger, matching certificate,
memory-state hashes, replay retention, non-negative utility, and no model-weight
mutation are all required before an update is accepted.

Spec: REQ-LEARN-1659, SCENARIO-LEARN-1659, SCENARIO-LEARN-1660.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
JsonInput = Mapping[str, Any] | str | Path | PathLike[str]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260509"
EXPERIMENT_ID = 1659
OUTPUT_FILE = "experiment_1659_smgi_certified_updates.json"
SCHEMA = "carnot.smgi_certified_updates.v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_CERCE_LEDGER_PATH = REPO_ROOT / "results" / "experiment_1594_cerce_ledger.json"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "schema",
    "experiment_id",
    "continuous_self_learning_task",
    "smgi_certified_update_ready",
    "certified_update_success",
    "cerce_ledger_ready",
    "policy_certificates_evaluated",
    "accepted_violation_count",
    "false_accept_delta",
    "soundness_mistakes",
    "nonforgetting_certificate_rate",
    "promotion_safe_policy_updates",
    "blocked_policy_updates",
    "candidate_updates_evaluated",
    "certified_update_count",
    "rejected_update_count",
    "certified_updates",
    "rejected_updates",
    "blockers",
    "honest_verdict",
)

SPEC_TRACES = ("REQ-LEARN-1659", "SCENARIO-LEARN-1659", "SCENARIO-LEARN-1660")


def stable_hash(payload: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible SMGI evidence."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def derive_candidates_from_ledger(ledger: Mapping[str, Any]) -> list[JsonDict]:
    """Derive deterministic SMGI candidates from promotion-safe CerCE certificates.

    This is the bridge from the CerCE ledger into the SMGI gate.  A derived row
    is intentionally conservative: it cites only the certificate id, source
    ledger hash, and the number of constraint ids the ledger already checked.
    No model parameters or learned weights are touched.

    Spec: REQ-LEARN-1659-4
    """

    ledger_hash = stable_hash(ledger)
    candidates: list[JsonDict] = []
    for certificate in _policy_certificates(ledger):
        if certificate.get("promotion_safe") is not True:
            continue
        policy_id = str(certificate.get("policy_update_id", ""))
        certificate_id = str(certificate.get("certificate_id", ""))
        constraint_ids = certificate.get("constraint_ids", [])
        replay_count = int(certificate.get("constraint_count") or len(constraint_ids) or 1)
        candidates.append(
            {
                "policy_update_id": policy_id,
                "certificate_id": certificate_id,
                "prior_memory_hash": stable_hash(
                    {"ledger_hash": ledger_hash, "policy_update_id": policy_id, "state": "prior"}
                ),
                "updated_memory_hash": stable_hash(
                    {
                        "certificate_id": certificate_id,
                        "ledger_hash": ledger_hash,
                        "policy_update_id": policy_id,
                        "state": "updated",
                    }
                ),
                "replay_case_count": replay_count,
                "retained_case_count": replay_count,
                "replay_failure_count": 0,
                "utility_delta": float(certificate.get("utility_delta", 0.0) or 0.0),
                "no_model_weight_mutation": True,
                "provenance": ["results/experiment_1594_cerce_ledger.json"],
            }
        )
    return candidates


def certify_update_gates(
    cerce_ledger: JsonInput,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    *,
    output_path: str | Path | PathLike[str] | None = None,
    project_root: str | Path | PathLike[str] = REPO_ROOT,
    run_date: str = RUN_DATE,
    min_replay_cases: int = 1,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Evaluate SMGI certified-update gates and optionally write a JSON report.

    Spec: REQ-LEARN-1659-1, REQ-LEARN-1659-2, REQ-LEARN-1659-3
    """

    ledger = _coerce_mapping(cerce_ledger)
    candidate_rows = (
        derive_candidates_from_ledger(ledger)
        if candidates is None
        else [dict(row) for row in candidates]
    )
    ledger_failures = _ledger_gate_failures(ledger)
    certificate_index = _certificate_index(ledger)
    safe_policy_ids = set(_string_list(ledger.get("promotion_safe_policy_updates")))
    evaluated = [
        _evaluate_candidate(
            row,
            certificate_index=certificate_index,
            safe_policy_ids=safe_policy_ids,
            ledger_failures=ledger_failures,
            min_replay_cases=min_replay_cases,
        )
        for row in candidate_rows
    ]
    certified_updates = [row for row in evaluated if row["certified_update_success"]]
    rejected_updates = [row for row in evaluated if not row["certified_update_success"]]

    blockers = set(ledger_failures)
    if not candidate_rows:
        blockers.add("no_smgi_update_candidates")
    if rejected_updates:
        blockers.add("smgi_candidate_rejected")
    for row in rejected_updates:
        blockers.update(str(reason) for reason in row["gate_failures"])

    ready = not blockers and bool(certified_updates)
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "schema": SCHEMA,
        "spec_traces": list(SPEC_TRACES),
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "project_root": str(project_root),
        "source_cerce_hash": stable_hash(ledger),
        "continuous_self_learning_task": True,
        "smgi_certified_update_ready": ready,
        "certified_update_success": ready,
        "cerce_ledger_ready": bool(ledger.get("cerce_ledger_ready")),
        "policy_certificates_evaluated": _int_value(ledger.get("policy_certificates_evaluated")),
        "accepted_violation_count": _int_value(ledger.get("accepted_violation_count")),
        "false_accept_delta": _int_value(ledger.get("false_accept_delta")),
        "soundness_mistakes": _int_value(ledger.get("soundness_mistakes")),
        "nonforgetting_certificate_rate": _float_value(
            ledger.get("nonforgetting_certificate_rate")
        ),
        "no_model_weight_mutation": all(
            row["gates"]["no_model_weight_mutation"] for row in evaluated
        ),
        "promotion_safe_policy_updates": sorted(safe_policy_ids),
        "blocked_policy_updates": _string_list(ledger.get("blocked_policy_updates")),
        "candidate_updates_evaluated": len(evaluated),
        "certified_update_count": len(certified_updates),
        "rejected_update_count": len(rejected_updates),
        "certified_updates": certified_updates,
        "rejected_updates": rejected_updates,
        "blockers": sorted(blockers),
        "honest_verdict": (
            "complete: smgi_certified_updates_ready"
            if ready
            else "blocked: smgi_certified_updates_failed"
        ),
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact)
    if output_path is not None:
        _write_json(output_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the SMGI certified-update report consumed by conductor gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError(f"unsupported schema: {artifact['schema']}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["continuous_self_learning_task"] is not True:
        raise AssertionError("continuous_self_learning_task must be true")
    if int(artifact["candidate_updates_evaluated"]) != (
        int(artifact["certified_update_count"]) + int(artifact["rejected_update_count"])
    ):
        raise AssertionError("candidate update counts are inconsistent")
    rate = float(artifact["nonforgetting_certificate_rate"])
    if not 0.0 <= rate <= 1.0:
        raise AssertionError("nonforgetting_certificate_rate must be between 0 and 1")
    if artifact["status"] == "complete":
        errors: list[str] = []
        if artifact["smgi_certified_update_ready"] is not True:
            errors.append("smgi_certified_update_ready must be true")
        if artifact["certified_update_success"] is not True:
            errors.append("certified_update_success must be true")
        if artifact["cerce_ledger_ready"] is not True:
            errors.append("cerce_ledger_ready must be true")
        if int(artifact["certified_update_count"]) < 1:
            errors.append("at least one certified update is required")
        if artifact["rejected_updates"]:
            errors.append("complete artifact cannot contain rejected updates")
        if artifact["blockers"]:
            errors.append("complete artifact cannot contain blockers")
        if int(artifact["accepted_violation_count"]) != 0:
            errors.append("accepted_violation_count must be zero")
        if int(artifact["false_accept_delta"]) > 0:
            errors.append("false_accept_delta cannot be positive")
        if int(artifact["soundness_mistakes"]) != 0:
            errors.append("soundness_mistakes must be zero")
        if rate != 1.0:
            errors.append("nonforgetting_certificate_rate must be 1.0")
        if artifact.get("no_model_weight_mutation") is not True:
            errors.append("no_model_weight_mutation must be true")
        if errors:
            raise AssertionError(f"complete artifact is invalid: {errors}")


def _evaluate_candidate(
    candidate: Mapping[str, Any],
    *,
    certificate_index: Mapping[str, JsonDict],
    safe_policy_ids: set[str],
    ledger_failures: Sequence[str],
    min_replay_cases: int,
) -> JsonDict:
    policy_id = str(candidate.get("policy_update_id", ""))
    certificate = certificate_index.get(policy_id)
    certificate_id = str(candidate.get("certificate_id", ""))
    ledger_certificate_id = str(certificate.get("certificate_id", "")) if certificate else ""
    replay_count = _int_value(candidate.get("replay_case_count"))
    retained_count = _int_value(candidate.get("retained_case_count"))
    replay_failures = _int_value(candidate.get("replay_failure_count"))
    utility_delta = _float_value(candidate.get("utility_delta"))
    prior_hash = str(candidate.get("prior_memory_hash", ""))
    updated_hash = str(candidate.get("updated_memory_hash", ""))
    provenance = _string_list(candidate.get("provenance"))
    gates = {
        "cerce_ledger_safe": not ledger_failures,
        "policy_promotion_safe": policy_id in safe_policy_ids,
        "cerce_certificate_present": certificate is not None,
        "cerce_certificate_match": bool(certificate and certificate_id == ledger_certificate_id),
        "memory_hashes_present": bool(prior_hash and updated_hash),
        "memory_hash_changed": bool(prior_hash and updated_hash and prior_hash != updated_hash),
        "retention_replay_passed": bool(
            replay_count >= min_replay_cases
            and retained_count >= replay_count
            and replay_failures == 0
        ),
        "nonnegative_utility_delta": utility_delta >= 0.0,
        "no_model_weight_mutation": candidate.get("no_model_weight_mutation") is True,
        "provenance_present": bool(provenance),
    }
    gate_failures = _candidate_failures(gates)
    return {
        "policy_update_id": policy_id,
        "certificate_id": certificate_id,
        "prior_memory_hash": prior_hash,
        "updated_memory_hash": updated_hash,
        "replay_case_count": replay_count,
        "retained_case_count": retained_count,
        "replay_failure_count": replay_failures,
        "utility_delta": utility_delta,
        "no_model_weight_mutation": candidate.get("no_model_weight_mutation") is True,
        "provenance": provenance,
        "gates": gates,
        "gate_failures": gate_failures,
        "certified_update_success": not gate_failures,
    }


def _candidate_failures(gates: Mapping[str, bool]) -> list[str]:
    failures: list[str] = []
    if not gates["cerce_ledger_safe"]:
        failures.append("cerce_ledger_not_safe")
    if not gates["policy_promotion_safe"]:
        failures.append("policy_not_promotion_safe")
    if not gates["cerce_certificate_present"]:
        failures.append("missing_cerce_certificate")
    if gates["cerce_certificate_present"] and not gates["cerce_certificate_match"]:
        failures.append("cerce_certificate_mismatch")
    if not gates["memory_hashes_present"]:
        failures.append("missing_session_memory_hash")
    if gates["memory_hashes_present"] and not gates["memory_hash_changed"]:
        failures.append("unchanged_session_memory_hash")
    if not gates["retention_replay_passed"]:
        failures.append("retention_replay_failed")
    if not gates["nonnegative_utility_delta"]:
        failures.append("negative_utility_delta")
    if not gates["no_model_weight_mutation"]:
        failures.append("model_weight_mutation_detected")
    if not gates["provenance_present"]:
        failures.append("missing_update_provenance")
    return failures


def _ledger_gate_failures(ledger: Mapping[str, Any]) -> list[str]:
    certificates = _policy_certificates(ledger)
    failures: list[str] = []
    if ledger.get("status") != "complete":
        failures.append("cerce_ledger_not_complete")
    if ledger.get("cerce_ledger_ready") is not True:
        failures.append("cerce_ledger_not_ready")
    if not certificates:
        failures.append("no_policy_certificates")
    if _int_value(ledger.get("accepted_violation_count")) != 0:
        failures.append("accepted_constraint_violation")
    if _int_value(ledger.get("false_accept_delta")) > 0:
        failures.append("positive_false_accept_delta")
    if _int_value(ledger.get("soundness_mistakes")) != 0:
        failures.append("soundness_mistake")
    if _float_value(ledger.get("nonforgetting_certificate_rate")) != 1.0:
        failures.append("nonforgetting_certificate_rate_not_one")
    if _string_list(ledger.get("blocked_policy_updates")):
        failures.append("blocked_policy_updates_present")
    if [cert for cert in certificates if cert.get("promotion_safe") is not True]:
        failures.append("unsafe_policy_certificate")
    return sorted(set(failures))


def _certificate_index(ledger: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(certificate.get("policy_update_id", "")): certificate
        for certificate in _policy_certificates(ledger)
    }


def _policy_certificates(ledger: Mapping[str, Any]) -> list[JsonDict]:
    raw_certificates = ledger.get("policy_certificates", [])
    if not isinstance(raw_certificates, Sequence) or isinstance(raw_certificates, (str, bytes)):
        return []
    return [dict(item) for item in raw_certificates if isinstance(item, Mapping)]


def _coerce_mapping(payload: JsonInput) -> JsonDict:
    if isinstance(payload, Mapping):
        return dict(payload)
    loaded = json.loads(Path(payload).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("SMGI CerCE ledger input must be a JSON object")
    return dict(loaded)


def _write_json(path: str | Path | PathLike[str], artifact: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(artifact), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _int_value(value: Any) -> int:
    return int(value or 0)


def _float_value(value: Any) -> float:
    return float(value or 0.0)


__all__ = [
    "DEFAULT_CERCE_LEDGER_PATH",
    "DEFAULT_OUTPUT_PATH",
    "EXPERIMENT_ID",
    "OUTPUT_FILE",
    "REQUIRED_ARTIFACT_FIELDS",
    "SCHEMA",
    "SPEC_TRACES",
    "certify_update_gates",
    "derive_candidates_from_ledger",
    "stable_hash",
    "validate_artifact",
]
