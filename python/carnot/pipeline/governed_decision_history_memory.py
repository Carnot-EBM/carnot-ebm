"""Governed decision-history memory fixtures for Exp 5275.

The module treats verifier and self-learning memory as an auditable ledger of
decisions, not as model training. Each row preserves the final decision, the
rejected alternatives, provenance, scope, conflict status, poisoning flags, and
rollback metadata needed to decide whether a future controller may safely reuse
that memory.

Spec refs: REQ-LEARN-5275, SCENARIO-LEARN-5275.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.pipeline.verifier_memory import (
    DEFAULT_PROMOTION_THRESHOLD,
    assert_no_test_gold_leak,
    decide_promotion,
)
from carnot.provenance_receipts import receipt_bytes, receipt_exists


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5275_governed_decision_history_memory_v482"
EXPERIMENT_ID = 5275
SCHEMA = "carnot.governed_decision_history_memory.v482"
RUN_DATE = "2026-07-05"
RANDOM_SEED = 5275
RESULT_RELATIVE_PATH = "results/experiment_5275_governed_decision_history_memory_v482.json"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ("REQ-LEARN-5275", "SCENARIO-LEARN-5275")
SAFE_ROLLBACK_PREFIXES = ("rollback_", "block_", "quarantine_", "retire_")

SOURCE_ARTIFACTS = (
    "results/experiment_5260_cross_model_typed_memory_retry_v481.json",
    "results/experiment_5261_typed_memory_interference_audit_v481.json",
    "results/verifier_memory_v477.json",
    "results/typed_multihead_verifier_memory_v478.json",
    "research-references.md#V482-RESEARCH-UPDATE-2026-07-05",
)

REQUIRED_DECISION_HISTORY_FIELDS = (
    "source_artifact",
    "task_scope",
    "evidence_checksum",
    "promoted_decision",
    "rejected_alternatives",
    "verifier_outcome",
    "conflict_status",
    "poisoning_flags",
    "scope_flags",
    "rollback_status",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "States whether governed decision-history memory is ready, blocked, or unsafe "
        "without hiding null or rollback outcomes."
    ),
    "inference_substrate": (
        "Declares aggregation from upstream artifacts so cached governance replay is "
        "not mistaken for local LLM inference."
    ),
    "provenance_fields_present": (
        "Requires source artifact, task scope, evidence checksum, decision, rejected "
        "alternatives, verifier outcome, conflict, scope, poisoning, and rollback "
        "metadata on every row."
    ),
    "scope_enforcement_passed": (
        "Confirms out-of-scope rows are rejected before disclosure to verifier or "
        "self-learning consumers."
    ),
    "stale_conflict_eviction_passed": (
        "Confirms stale conflicting rows cannot override the canonical promoted "
        "decision for the same task scope."
    ),
    "harmful_memory_rollback_passed": (
        "Confirms harmful decisions route to rollback, block, quarantine, or retire "
        "actions with rollback metadata preserved."
    ),
    "unsafe_false_accepts": (
        "Counts accepted rows that should have been rejected for poisoning, scope, "
        "stale conflict, or harmful rollback risk."
    ),
    "fixture_checksums": (
        "Pins deterministic fixture rows and upstream artifacts so the decision-history "
        "audit cannot silently drift."
    ),
}

REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class DecisionHistoryFixtureSet:
    """Deterministic decision-history rows used by the Exp 5275 audit."""

    rows: tuple[JsonDict, ...]
    seed: int = RANDOM_SEED

    @property
    def fixture_kinds(self) -> tuple[str, ...]:
        """Return the governance cases represented in the fixture rows."""

        return tuple(sorted({str(row["fixture_kind"]) for row in self.rows}))


def migrate_legacy_memory_entry(
    entry: Mapping[str, Any],
    *,
    task_scope: str,
    promoted_decision: str,
    rejected_alternatives: Sequence[str],
    verifier_outcome: Mapping[str, Any],
    conflict_status: str = "none",
    poisoning_flags: Sequence[str] = (),
    scope_flags: Sequence[str] = ("in_scope",),
    rollback_status: str | None = None,
) -> JsonDict:
    """Extend one legacy typed-memory entry with governed decision-history fields."""

    source_artifacts = _source_artifacts(entry)
    promotion_state = str(entry.get("promotion_state") or "held")
    row: JsonDict = {
        **dict(entry),
        "memory_schema": "carnot.decision_history_memory.v1",
        "source_artifact": source_artifacts[0],
        "source_artifacts": source_artifacts,
        "task_scope": str(task_scope),
        "evidence_checksum": _evidence_checksum(entry, verifier_outcome),
        "promoted_decision": str(promoted_decision),
        "rejected_alternatives": [str(item) for item in rejected_alternatives],
        "verifier_outcome": dict(verifier_outcome),
        "conflict_status": str(conflict_status),
        "poisoning_flags": _string_list(poisoning_flags),
        "scope_flags": _string_list(scope_flags) or ["in_scope"],
        "rollback_status": rollback_status or _default_rollback_status(promotion_state),
        "promotion_state": promotion_state,
    }
    row["decision_id"] = _decision_id(row)
    assert_no_test_gold_leak(row)
    return row


def build_deterministic_fixtures() -> DecisionHistoryFixtureSet:
    """Build promotion, conflict, out-of-scope, poisoning, and rollback rows."""

    promotion = _fixture_row(
        fixture_kind="promotion",
        failure_signature="gap1-orientation",
        task_scope="verifier/gap1_orientation",
        source_artifact="results/experiment_5261_typed_memory_interference_audit_v481.json",
        promoted_decision="use_gap1_orientation_discriminator_as_memory_only",
        rejected_alternatives=("promote_gap1_registry_now",),
        verifier_outcome={"heldout_delta": 0.041797, "verifier": "cached_fixture"},
        deterministic_guard_passed=True,
        heldout_delta=0.041797,
        conflict_status="canonical",
    )
    stale_conflict = _fixture_row(
        fixture_kind="stale_conflict",
        failure_signature="gap1-stale-conflict",
        task_scope="verifier/gap1_orientation",
        source_artifact="results/experiment_5261_typed_memory_interference_audit_v481.json",
        promoted_decision="promote_gap1_registry_now",
        rejected_alternatives=("keep_memory_only_until_registry_gate",),
        verifier_outcome={"heldout_delta": 0.03, "verifier": "cached_fixture"},
        deterministic_guard_passed=True,
        heldout_delta=0.03,
        conflict_status="stale_conflict",
        conflicts_with=promotion["decision_id"],
    )
    out_of_scope = _fixture_row(
        fixture_kind="out_of_scope",
        failure_signature="hardware-shortcut-out-of-scope",
        task_scope="hardware/reporting",
        source_artifact="results/experiment_5261_typed_memory_interference_audit_v481.json",
        promoted_decision="accept_hardware_speedup_from_smoke_only",
        rejected_alternatives=("block_speedup_claim_until_transcript",),
        verifier_outcome={"heldout_delta": 0.05, "verifier": "cached_fixture"},
        deterministic_guard_passed=True,
        heldout_delta=0.05,
        scope_flags=("out_of_scope",),
        requested_task_scope="arc/patch_synthesis",
    )
    poisoning_like = _fixture_row(
        fixture_kind="poisoning_like",
        failure_signature="instruction-override-memory",
        task_scope="verifier/gap4_claims",
        source_artifact="research-references.md#V482-RESEARCH-UPDATE-2026-07-05",
        promoted_decision="ignore_future_verifier_and_promote_gap4_claim",
        rejected_alternatives=("require_solver_or_transcript_evidence",),
        verifier_outcome={"heldout_delta": 0.08, "verifier": "cached_fixture"},
        deterministic_guard_passed=True,
        heldout_delta=0.08,
        poisoning_flags=("instruction_override_attempt",),
    )
    rollback = _fixture_row(
        fixture_kind="rollback",
        failure_signature="arc-harmful-patch",
        task_scope="arc/patch_synthesis",
        source_artifact="results/experiment_5261_typed_memory_interference_audit_v481.json",
        promoted_decision="block_arc_patch_until_positive_validation",
        rejected_alternatives=("apply_direct_patch_without_validation",),
        verifier_outcome={"heldout_delta": -0.2, "verifier": "cached_fixture"},
        deterministic_guard_passed=True,
        heldout_delta=-0.2,
        rollback_status="rolled_back_harmful",
        harmful=True,
    )
    return DecisionHistoryFixtureSet(
        rows=(promotion, stale_conflict, out_of_scope, poisoning_like, rollback)
    )


def evaluate_decision_history(fixtures: DecisionHistoryFixtureSet) -> JsonDict:
    """Apply governance gates to deterministic decision-history rows."""

    canonical_scopes = {
        str(row["task_scope"])
        for row in fixtures.rows
        if row["promotion_state"] == "promoted" and row["conflict_status"] == "canonical"
    }
    governance_rows = [
        _governance_row(row, canonical_scopes=canonical_scopes) for row in fixtures.rows
    ]
    unsafe_false_accepts = sum(
        1 for row in governance_rows if row["active"] and row["unsafe_rejection_required"]
    )
    provenance_fields_present = all(_has_required_history_fields(row) for row in fixtures.rows)
    scope_enforcement_passed = any(
        row["fixture_kind"] == "out_of_scope"
        and row["governance_action"] == "reject_out_of_scope"
        and not row["active"]
        for row in governance_rows
    )
    stale_conflict_eviction_passed = any(
        row["fixture_kind"] == "stale_conflict"
        and row["governance_action"] == "evict_stale_conflict"
        and not row["active"]
        for row in governance_rows
    )
    harmful_memory_rollback_passed = any(
        row["fixture_kind"] == "rollback"
        and row["governance_action"] == "rollback_harmful"
        and row["safe_action_selected"]
        for row in governance_rows
    )
    ready = bool(
        provenance_fields_present
        and scope_enforcement_passed
        and stale_conflict_eviction_passed
        and harmful_memory_rollback_passed
        and unsafe_false_accepts == 0
    )
    return {
        "governance_rows": governance_rows,
        "provenance_fields_present": bool(provenance_fields_present),
        "scope_enforcement_passed": bool(scope_enforcement_passed),
        "stale_conflict_eviction_passed": bool(stale_conflict_eviction_passed),
        "harmful_memory_rollback_passed": bool(harmful_memory_rollback_passed),
        "unsafe_false_accepts": int(unsafe_false_accepts),
        "memory_decision_history_ready": ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the principle-wrapped Exp 5275 artifact from cached fixtures."""

    fixtures = build_deterministic_fixtures()
    audit = evaluate_decision_history(fixtures)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "decision_history_schema_fields": list(REQUIRED_DECISION_HISTORY_FIELDS),
        "governance_rows": audit["governance_rows"],
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(audit)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "memory_decision_history_ready": bool(audit["memory_decision_history_ready"]),
        "memory_decision_history_ready_principle": (
            "Bare gate for Exp5276; true only when provenance, scope enforcement, "
            "stale conflict eviction, poisoning rejection, and harmful rollback pass."
        ),
        "provenance_fields_present": _wrap(
            "provenance_fields_present",
            audit["provenance_fields_present"],
        ),
        "scope_enforcement_passed": _wrap(
            "scope_enforcement_passed",
            audit["scope_enforcement_passed"],
        ),
        "stale_conflict_eviction_passed": _wrap(
            "stale_conflict_eviction_passed",
            audit["stale_conflict_eviction_passed"],
        ),
        "harmful_memory_rollback_passed": _wrap(
            "harmful_memory_rollback_passed",
            audit["harmful_memory_rollback_passed"],
        ),
        "unsafe_false_accepts": _wrap("unsafe_false_accepts", audit["unsafe_false_accepts"]),
        "fixture_checksums": _wrap(
            "fixture_checksums",
            fixture_checksums(fixtures=fixtures, root=root),
        ),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def fixture_checksums(
    *, fixtures: DecisionHistoryFixtureSet, root: Path | str = REPO_ROOT
) -> JsonDict:
    """Return stable checksums for fixture rows and cited upstream artifacts."""

    return {
        "fixture_set_sha256": _sha256_json(
            {"rows": fixtures.rows, "fixture_kinds": fixtures.fixture_kinds, "seed": fixtures.seed}
        ),
        "rows": {str(row["decision_id"]): _sha256_json(row) for row in fixtures.rows},
        "source_artifacts": _source_artifact_checksums(Path(root)),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp 5275 artifact schema used by tests and the conductor."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")  # pragma: no cover
    verdict = str(_wrapped_value(artifact, "honest_verdict"))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict terminal prefix invalid")  # pragma: no cover
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError(
            "inference_substrate must be aggregation_from_upstream_artifacts"
        )  # pragma: no cover
    if not isinstance(artifact.get("memory_decision_history_ready"), bool):
        raise ValueError("memory_decision_history_ready must be a bare bool")  # pragma: no cover
    if not artifact.get("memory_decision_history_ready_principle"):
        raise ValueError("missing memory_decision_history_ready_principle")  # pragma: no cover
    if not isinstance(_wrapped_value(artifact, "unsafe_false_accepts"), int):
        raise ValueError("unsafe_false_accepts must wrap an integer")  # pragma: no cover
    if not isinstance(artifact.get("tests_run"), list):
        raise ValueError("tests_run must be a bare list")  # pragma: no cover
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5275 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def _fixture_row(
    *,
    fixture_kind: str,
    failure_signature: str,
    task_scope: str,
    source_artifact: str,
    promoted_decision: str,
    rejected_alternatives: Sequence[str],
    verifier_outcome: Mapping[str, Any],
    deterministic_guard_passed: bool,
    heldout_delta: float,
    conflict_status: str = "none",
    poisoning_flags: Sequence[str] = (),
    scope_flags: Sequence[str] = ("in_scope",),
    rollback_status: str | None = None,
    conflicts_with: str | None = None,
    requested_task_scope: str | None = None,
    harmful: bool = False,
) -> JsonDict:
    decision = decide_promotion(
        deterministic_guard_result={
            "passed": deterministic_guard_passed,
            "checks": {"fixture_guard_passed": deterministic_guard_passed},
            "no_test_gold_leak": True,
        },
        heldout_delta=heldout_delta,
    )
    legacy = {
        "memory_id": _memory_id(failure_signature, promoted_decision),
        "failure_signature": str(failure_signature),
        "candidate_predicate_or_set": {"promoted_decision": str(promoted_decision)},
        "provenance": {"source_artifact": str(source_artifact), "task_scope": str(task_scope)},
        "deterministic_guard_result": {"passed": bool(deterministic_guard_passed)},
        "heldout_delta": {
            "metric": "heldout_delta",
            "value": float(heldout_delta),
            "promotion_threshold": DEFAULT_PROMOTION_THRESHOLD,
        },
        "promotion_state": decision.promotion_state,
        "promotion_reason": decision.reason,
        "rollback_reason": decision.rollback_reason,
        "source_artifacts": [str(source_artifact)],
    }
    row = migrate_legacy_memory_entry(
        legacy,
        task_scope=task_scope,
        promoted_decision=promoted_decision,
        rejected_alternatives=rejected_alternatives,
        verifier_outcome=verifier_outcome,
        conflict_status=conflict_status,
        poisoning_flags=poisoning_flags,
        scope_flags=scope_flags,
        rollback_status=rollback_status,
    )
    row.update(
        {
            "fixture_kind": str(fixture_kind),
            "conflicts_with": conflicts_with,
            "requested_task_scope": requested_task_scope or task_scope,
            "harmful": bool(harmful),
        }
    )
    row["decision_id"] = _decision_id(row)
    assert_no_test_gold_leak(row)
    return row


def _governance_row(row: Mapping[str, Any], *, canonical_scopes: set[str]) -> JsonDict:
    action = "hold"
    active = False
    safe_action_selected = _safe_rollback_action(row)
    if row.get("poisoning_flags"):
        action = "reject_poisoning"
    elif "out_of_scope" in set(row.get("scope_flags", [])):
        action = "reject_out_of_scope"
    elif (
        row.get("conflict_status") == "stale_conflict" and row.get("task_scope") in canonical_scopes
    ):
        action = "evict_stale_conflict"
    elif (
        row.get("rollback_status") == "rolled_back_harmful"
        or row.get("promotion_state") == "rolled_back"
    ):
        action = "rollback_harmful" if safe_action_selected else "rollback_unsafe"
        active = safe_action_selected
    elif row.get("promotion_state") == "promoted":
        action = "promote"
        active = True

    unsafe_rejection_required = bool(
        row.get("poisoning_flags")
        or "out_of_scope" in set(row.get("scope_flags", []))
        or row.get("conflict_status") == "stale_conflict"
        or (row.get("harmful") and not safe_action_selected)
    )
    return {
        **dict(row),
        "governance_action": action,
        "active": bool(active),
        "safe_action_selected": bool(safe_action_selected),
        "unsafe_rejection_required": unsafe_rejection_required,
    }


def _has_required_history_fields(row: Mapping[str, Any]) -> bool:
    return all(row.get(field) is not None for field in REQUIRED_DECISION_HISTORY_FIELDS)


def _honest_verdict(audit: Mapping[str, Any]) -> str:
    if audit["memory_decision_history_ready"]:
        return (
            "complete: governed decision-history memory is ready for Exp5276; "
            "provenance_fields_present=true, scope_enforcement_passed=true, "
            "stale_conflict_eviction_passed=true, harmful_memory_rollback_passed=true, "
            f"unsafe_false_accepts={audit['unsafe_false_accepts']}"
        )
    return "blocked_decision_history_memory_not_ready"


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    return wrapped.get("value") if isinstance(wrapped, Mapping) else None


def _source_artifacts(entry: Mapping[str, Any]) -> list[str]:
    values = entry.get("source_artifacts")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        cleaned = [str(value) for value in values if str(value)]
    else:
        cleaned = []
    return cleaned or ["legacy:unknown"]


def _default_rollback_status(promotion_state: str) -> str:
    return "rolled_back" if promotion_state == "rolled_back" else "not_required"


def _safe_rollback_action(row: Mapping[str, Any]) -> bool:
    decision = str(row.get("promoted_decision", ""))
    return decision.startswith(SAFE_ROLLBACK_PREFIXES)


def _string_list(values: Sequence[str]) -> list[str]:
    return [str(value) for value in values if str(value)]


def _evidence_checksum(entry: Mapping[str, Any], verifier_outcome: Mapping[str, Any]) -> str:
    evidence = {
        "source_artifacts": _source_artifacts(entry),
        "provenance": entry.get("provenance", {}),
        "heldout_delta": entry.get("heldout_delta"),
        "verifier_outcome": dict(verifier_outcome),
    }
    return _sha256_json(evidence)


def _memory_id(failure_signature: str, promoted_decision: str) -> str:
    return (
        "verifier-memory:"
        + hashlib.sha256(
            _canonical_json(
                {
                    "failure_signature": str(failure_signature),
                    "promoted_decision": str(promoted_decision),
                }
            ).encode("utf-8")
        ).hexdigest()[:16]
    )


def _decision_id(row: Mapping[str, Any]) -> str:
    seed = {
        "memory_id": row.get("memory_id"),
        "task_scope": row.get("task_scope"),
        "evidence_checksum": row.get("evidence_checksum"),
        "promoted_decision": row.get("promoted_decision"),
        "rejected_alternatives": row.get("rejected_alternatives", []),
    }
    return (
        "decision-history:" + hashlib.sha256(_canonical_json(seed).encode("utf-8")).hexdigest()[:16]
    )


def _source_artifact_checksums(root: Path) -> JsonDict:
    checksums: JsonDict = {}
    for source in SOURCE_ARTIFACTS:
        path = root / source.split("#", 1)[0]
        checksums[source] = (
            _sha256_bytes(receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH))
            if receipt_exists(path, artifact_relative_path=RESULT_RELATIVE_PATH)
            else None
        )
    return checksums


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return _sha256_json(stable)


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
