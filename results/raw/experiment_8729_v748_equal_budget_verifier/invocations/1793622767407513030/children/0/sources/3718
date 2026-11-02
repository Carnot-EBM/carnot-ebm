"""CerCE-style ledger gate for FR-11 policy promotion.

FR-11 can propose query-time policy and memory updates, but promotion is only
safe when old replay cases remain retained and the post-update constraint
violation bound does not get worse.  This module keeps that check as explicit
bookkeeping: it compares pre/post violation bounds, records replay retention,
and emits deterministic certificates that downstream gates can audit.

Spec: REQ-LEARN-1668, SCENARIO-LEARN-1668, SCENARIO-LEARN-1669.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260510"
EXPERIMENT_ID = 1668
OUTPUT_FILE = "experiment_1668_cerce.json"
SCHEMA = "carnot.pipeline_cerce_ledger.v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
SPEC_TRACES = ("REQ-LEARN-1668", "SCENARIO-LEARN-1668", "SCENARIO-LEARN-1669")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "schema",
    "experiment_id",
    "continuous_self_learning_task",
    "cerce_ledger_ready",
    "promotion_gate_passed",
    "policy_certificates_evaluated",
    "promotion_safe_policy_updates",
    "blocked_policy_updates",
    "accepted_violation_count",
    "pre_violation_bound",
    "post_violation_bound",
    "violation_bound_delta",
    "replay_retention_rate",
    "nonforgetting_rate",
    "nonforgetting_certificate_rate",
    "certificates",
    "blockers",
    "honest_verdict",
)


@dataclass(frozen=True)
class ReplayCase:
    """One replayed constraint case used to test a proposed memory update."""

    case_id: str
    pre_violation_bound: float
    post_violation_bound: float
    retained: bool = True
    replay_failed: bool = False
    source: str = "simulated_memory_update"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReplayCase:
        """Build a replay row from a JSON-compatible payload."""

        return cls(
            case_id=str(payload.get("case_id", "")),
            pre_violation_bound=float(payload.get("pre_violation_bound", 0.0) or 0.0),
            post_violation_bound=float(payload.get("post_violation_bound", 0.0) or 0.0),
            retained=payload.get("retained", True) is True,
            replay_failed=payload.get("replay_failed", False) is True,
            source=str(payload.get("source", "simulated_memory_update")),
        )

    @property
    def bound_delta(self) -> float:
        """Return post minus pre, where positive means the update forgot a case."""

        return self.post_violation_bound - self.pre_violation_bound

    @property
    def bound_worsened(self) -> bool:
        """Whether the post-update bound is strictly worse for this replay case."""

        return self.bound_delta > 0.0

    def to_dict(self) -> JsonDict:
        """Return deterministic JSON-compatible replay evidence."""

        return {
            "case_id": self.case_id,
            "pre_violation_bound": self.pre_violation_bound,
            "post_violation_bound": self.post_violation_bound,
            "bound_delta": self.bound_delta,
            "bound_worsened": self.bound_worsened,
            "retained": self.retained,
            "replay_failed": self.replay_failed,
            "source": self.source,
        }


@dataclass(frozen=True)
class MemoryPolicyUpdate:
    """A proposed FR-11 policy/memory update plus replay evidence."""

    policy_update_id: str
    prior_memory_hash: str
    updated_memory_hash: str
    replay_cases: tuple[ReplayCase, ...]
    utility_delta: float = 0.0
    no_model_weight_mutation: bool = True
    provenance: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MemoryPolicyUpdate:
        """Build an update candidate from a JSON-compatible payload."""

        raw_cases = payload.get("replay_cases", ())
        replay_cases = tuple(
            ReplayCase.from_mapping(item)
            for item in raw_cases
            if isinstance(item, Mapping)
        )
        raw_provenance = payload.get("provenance", ())
        provenance = tuple(str(item) for item in raw_provenance)
        return cls(
            policy_update_id=str(payload.get("policy_update_id", "")),
            prior_memory_hash=str(payload.get("prior_memory_hash", "")),
            updated_memory_hash=str(payload.get("updated_memory_hash", "")),
            replay_cases=replay_cases,
            utility_delta=float(payload.get("utility_delta", 0.0) or 0.0),
            no_model_weight_mutation=payload.get("no_model_weight_mutation") is True,
            provenance=provenance,
        )


UpdateInput = Mapping[str, Any] | MemoryPolicyUpdate


def stable_hash(payload: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible certificate evidence."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def certificate_for_update(candidate: UpdateInput) -> JsonDict:
    """Evaluate one proposed update and return its CerCE promotion certificate."""

    update = (
        candidate
        if isinstance(candidate, MemoryPolicyUpdate)
        else MemoryPolicyUpdate.from_mapping(candidate)
    )
    replay_rows = [case.to_dict() for case in update.replay_cases]
    replay_count = len(update.replay_cases)
    retained_count = sum(1 for case in update.replay_cases if case.retained)
    replay_failure_count = sum(1 for case in update.replay_cases if case.replay_failed)
    case_worsened_count = sum(1 for case in update.replay_cases if case.bound_worsened)
    pre_bound = sum(case.pre_violation_bound for case in update.replay_cases)
    post_bound = sum(case.post_violation_bound for case in update.replay_cases)
    bound_delta = post_bound - pre_bound
    retention_rate = retained_count / replay_count if replay_count else 0.0
    replay_retention_passed = bool(
        replay_count and retained_count == replay_count and replay_failure_count == 0
    )

    gate_failures: list[str] = []
    if replay_count == 0:
        gate_failures.append("no_replay_cases")
    if case_worsened_count:
        gate_failures.append("case_bound_worsened")
    if bound_delta > 0.0:
        gate_failures.append("aggregate_bound_worsened")
    if replay_count and retained_count != replay_count:
        gate_failures.append("replay_retention_failed")
    if replay_failure_count:
        gate_failures.append("replay_failed")
    if not update.prior_memory_hash or not update.updated_memory_hash:
        gate_failures.append("missing_memory_hash")
    elif update.prior_memory_hash == update.updated_memory_hash:
        gate_failures.append("unchanged_memory_hash")
    if update.no_model_weight_mutation is not True:
        gate_failures.append("model_weight_mutation_detected")

    payload = {
        "policy_update_id": update.policy_update_id,
        "prior_memory_hash": update.prior_memory_hash,
        "updated_memory_hash": update.updated_memory_hash,
        "replay_cases": replay_rows,
    }
    return {
        "policy_update_id": update.policy_update_id,
        "certificate_id": stable_hash(payload),
        "prior_memory_hash": update.prior_memory_hash,
        "updated_memory_hash": update.updated_memory_hash,
        "pre_violation_bound": pre_bound,
        "post_violation_bound": post_bound,
        "violation_bound_delta": bound_delta,
        "accepted_violation_count": case_worsened_count,
        "replay_case_count": replay_count,
        "retained_case_count": retained_count,
        "replay_failure_count": replay_failure_count,
        "replay_retention_rate": retention_rate,
        "replay_retention_passed": replay_retention_passed,
        "utility_delta": update.utility_delta,
        "no_model_weight_mutation": update.no_model_weight_mutation,
        "provenance": list(update.provenance),
        "replay_cases": replay_rows,
        "gate_failures": gate_failures,
        "promotion_safe": not gate_failures,
    }


def evaluate_promotion_gate(
    candidates: Sequence[UpdateInput],
    *,
    output_path: str | Path | PathLike[str] | None = None,
    project_root: str | Path | PathLike[str] = REPO_ROOT,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Evaluate the FR-11 CerCE promotion gate and optionally write JSON."""

    certificates = [certificate_for_update(candidate) for candidate in candidates]
    safe_ids = sorted(
        str(certificate["policy_update_id"])
        for certificate in certificates
        if certificate["promotion_safe"]
    )
    blocked_ids = sorted(
        str(certificate["policy_update_id"])
        for certificate in certificates
        if not certificate["promotion_safe"]
    )
    blockers = sorted(
        {
            "bound_worsened"
            if reason in {"case_bound_worsened", "aggregate_bound_worsened"}
            else str(reason)
            for certificate in certificates
            for reason in certificate["gate_failures"]
        }
    )
    if not certificates:
        blockers = ["no_policy_updates"]

    pre_bound = sum(float(certificate["pre_violation_bound"]) for certificate in certificates)
    post_bound = sum(float(certificate["post_violation_bound"]) for certificate in certificates)
    replay_cases = sum(int(certificate["replay_case_count"]) for certificate in certificates)
    retained_cases = sum(int(certificate["retained_case_count"]) for certificate in certificates)
    nonforgetting_rate = round(len(safe_ids) / len(certificates), 6) if certificates else 0.0
    replay_retention_rate = round(retained_cases / replay_cases, 6) if replay_cases else 0.0
    accepted_violation_count = sum(
        int(certificate["accepted_violation_count"]) for certificate in certificates
    )
    ready = bool(certificates and not blockers and nonforgetting_rate == 1.0)
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "schema": SCHEMA,
        "spec_traces": list(SPEC_TRACES),
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "project_root": str(project_root),
        "continuous_self_learning_task": True,
        "cerce_ledger_ready": ready,
        "promotion_gate_passed": ready,
        "policy_certificates_evaluated": len(certificates),
        "promotion_safe_policy_updates": safe_ids,
        "blocked_policy_updates": blocked_ids,
        "accepted_violation_count": accepted_violation_count,
        "pre_violation_bound": pre_bound,
        "post_violation_bound": post_bound,
        "violation_bound_delta": post_bound - pre_bound,
        "replay_retention_rate": replay_retention_rate,
        "nonforgetting_rate": nonforgetting_rate,
        "nonforgetting_certificate_rate": nonforgetting_rate,
        "certificates": certificates,
        "blockers": blockers,
        "honest_verdict": (
            "complete: cerce_promotion_ledger_ready"
            if ready
            else "blocked: cerce_promotion_ledger_failed"
        ),
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact)
    if output_path is not None:
        _write_json(output_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the fields consumed by the FR-11 promotion gate."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError(f"unsupported schema: {artifact['schema']}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    rate = float(artifact["nonforgetting_rate"])
    if not 0.0 <= rate <= 1.0:
        raise AssertionError("nonforgetting_rate must be between 0 and 1")
    if float(artifact["nonforgetting_certificate_rate"]) != rate:
        raise AssertionError("nonforgetting_certificate_rate must equal nonforgetting_rate")
    if int(artifact["policy_certificates_evaluated"]) != len(artifact["certificates"]):
        raise AssertionError("policy certificate counts are inconsistent")
    if artifact["status"] == "complete":
        errors: list[str] = []
        if artifact["cerce_ledger_ready"] is not True:
            errors.append("cerce_ledger_ready must be true")
        if artifact["promotion_gate_passed"] is not True:
            errors.append("promotion_gate_passed must be true")
        if artifact["blockers"]:
            errors.append("complete artifact cannot contain blockers")
        if artifact["blocked_policy_updates"]:
            errors.append("complete artifact cannot contain blocked policy updates")
        if int(artifact["accepted_violation_count"]) != 0:
            errors.append("accepted_violation_count must be zero")
        if rate != 1.0:
            errors.append("nonforgetting_rate must be 1.0")
        if not artifact["certificates"]:
            errors.append("at least one certificate is required")
        if errors:
            raise AssertionError(f"complete artifact is invalid: {errors}")


def _write_json(path: str | Path | PathLike[str], artifact: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(artifact), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "EXPERIMENT_ID",
    "OUTPUT_FILE",
    "REQUIRED_ARTIFACT_FIELDS",
    "SCHEMA",
    "SPEC_TRACES",
    "MemoryPolicyUpdate",
    "ReplayCase",
    "certificate_for_update",
    "evaluate_promotion_gate",
    "stable_hash",
    "validate_artifact",
]
