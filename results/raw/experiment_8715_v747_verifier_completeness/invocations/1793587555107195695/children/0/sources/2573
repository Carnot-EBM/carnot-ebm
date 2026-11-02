"""Build a prospective reliability stream from row-level exact receipts.

The reducer does not import earlier experiment producers. It reads checked-in
rows, binds transaction state to later outcomes, and freezes a controller input
contract. It never applies a memory or reliability update.

Spec ref: REQ-LEARN-6871.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6871_observable_reliability_opportunity_stream.json"
)
INFERENCE_SUBSTRATE = "deterministic CPU primary-receipt opportunity reconstruction"
RANDOM_SEED = 6_871_001
ORDER_REPLICATE_SEEDS = (6_871_011, 6_871_012, 6_871_013, 6_871_014, 6_871_015)
BLOCKED_VERDICT = "complete_blocked_observable_reliability_opportunity_stream"
READY_VERDICT = "complete_positive_observable_reliability_opportunity_stream_ready"

PRIMARY_SOURCE_PATHS = {
    "evidence_contract": "results/experiment_6865_v601_evidence_method_change_contract.json",
    "transactions": "results/experiment_6827_chronological_causal_edge_memory_stream.json",
    "outcomes_a": "results/experiment_6840_residual_memory_chronological_shard_a.json",
    "outcomes_b": "results/experiment_6841_residual_memory_delayed_correction_shard_b.json",
}
V599_AGGREGATE_PATHS = {
    "exp6853": "results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json",
    "exp6854": "results/experiment_6854_risk_sensitive_abstention_memory_controller.json",
    "exp6855": "results/experiment_6855_counterfactual_memory_credit_audit.json",
    "exp6856": "results/experiment_6856_sealed_risk_sensitive_learning_audit.json",
}
OWN_SOURCE_PATHS = {
    "module": "python/carnot/experiment_6871_observable_reliability_opportunity_stream.py",
    "wrapper": "scripts/experiments/experiment_6871_observable_reliability_opportunity_stream.py",
    "focused_tests": "tests/python/test_experiment_6871_observable_reliability_opportunity_stream.py",
    "spec": "openspec/capabilities/continuous-learning/spec.md",
    "adversarial_verifier": "scripts/adversarial_verify.py",
}

ACTION_NAMES = (
    "no_memory",
    "read_only_retrieval",
    "bounded_update",
    "quarantine",
    "v599_unsafe_reference",
)
OBSERVABLE_FEATURES = {
    "source_identity",
    "source_content_sha256",
    "family",
    "chronological_position",
    "operation_kind",
    "candidate_action_identity",
    "pre_action_state_sha256",
    "decision_snapshot_sha256",
    "counterfactual_kind",
    "counterfactual_available",
    "evidence_status_at_decision",
}
OFFLINE_SUPERVISION_FIELDS = {
    "later_exact_outcome",
    "signed_direction",
    "exact_outcome_hash",
    "outcome_identity",
    "delayed_correction",
    "future_correction",
    "audit_label",
    "held_split_label",
    "task_order_label",
    "order_id",
    "source_split",
}
REQUIRED_ATTACK_IDS = {
    "omission",
    "corruption",
    "unsupported_insertion",
    "provenance_loss",
    "delayed_invalidation",
    "tombstone_reappearance",
    "restart_loss",
    "rollback_mismatch",
    "poison_nonfinite",
    "poison_over_bound",
}
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

STANDARD_FIELDS = {
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "openspec_requirement_ids",
    "replay_commands",
}
TASK_FIELDS = {
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "primary_receipt_manifest",
    "rows",
    "rejected_opportunity_rows",
    "observable_feature_manifest",
    "offline_supervision_manifest",
    "leakage_witnesses",
    "action_manifest",
    "counterfactual_support_rows",
    "reliability_state_schema",
    "bounded_update_contract",
    "chronological_order_manifest",
    "order_replicate_manifest",
    "old_family_anchor_rows",
    "transition_attack_manifest",
    "random_seed",
    "reproducibility_checksum",
    "observable_reliability_stream_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}
REQUIRED_ARTIFACT_FIELDS = STANDARD_FIELDS | TASK_FIELDS


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes so all identities can be replayed."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Hash one JSON value with the project identity prefix."""

    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def _legacy_receipt_sha256(value: Any) -> str:
    """Reproduce the newline-terminated identity used by sealed shard B."""

    encoded = (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash exact file bytes, or return an empty identity when absent."""

    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _is_sha256(value: Any) -> bool:
    """Accept only a complete prefixed SHA-256 identity."""

    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Return one uniform gate row with exact evidence."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and expose the first exact failure."""

    copied = [dict(row) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": copied,
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def _read_json(path: Path) -> tuple[JsonDict, str | None]:
    """Read one JSON object and keep parse failures auditable."""

    if not path.is_file():
        return {}, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {}, f"unreadable:{type(error).__name__}"
    if not isinstance(value, dict):
        return {}, "json_object_required"
    return value, None


def _source_hashes(root: Path) -> JsonDict:
    """Bind every input and new contract file to exact bytes."""

    paths = {**PRIMARY_SOURCE_PATHS, **V599_AGGREGATE_PATHS, **OWN_SOURCE_PATHS}
    return {
        source_id: {"path": relative, "sha256": sha256_file(root / relative)}
        for source_id, relative in paths.items()
    }


def _load_validator(root: Path):
    """Load the current verifier without importing another experiment."""

    path = root / "scripts/adversarial_verify.py"
    name = "_exp6871_adversarial_verify"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("adversarial verifier import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.verify_artifact


def collect_live_adversarial_reports(root: Path) -> dict[str, JsonDict]:
    """Run the current read-only verifier against all excluded V599 aggregates."""

    verify_artifact = _load_validator(root)
    return {
        source_id: dict(verify_artifact(root / relative))
        for source_id, relative in V599_AGGREGATE_PATHS.items()
    }


def _live_status_rows(reports: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize live verifier output without treating it as row authority."""

    rows: list[JsonDict] = []
    for source_id, path in V599_AGGREGATE_PATHS.items():
        report = reports.get(source_id, {})
        flags = [dict(row) for row in report.get("flags", []) if isinstance(row, Mapping)]
        severities = {str(row.get("severity", "")).lower() for row in flags}
        status = "critical" if "critical" in severities else "warn" if flags else "clean"
        rows.append(
            {
                "source_id": source_id,
                "path": path,
                "live_status": status,
                "live_flag_count": len(flags),
                "live_flags": flags,
                "live_gate_version": report.get("gate_version"),
                "current_status_available": bool(report.get("gate_version")),
            }
        )
    return rows


def _order_number(order_id: Any) -> int:
    """Read the numeric suffix that defines the frozen base chronology."""

    match = re.fullmatch(r"order_(\d+)", str(order_id))
    return int(match.group(1)) if match else 10**9


def chronological_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the full stable key used for base-order reconstruction."""

    if "chronological_sort_key" in row:
        return tuple(row["chronological_sort_key"])
    return (
        _order_number(row.get("order_id")),
        str(row.get("source_family", "")),
        int(row.get("chronological_position", 0)),
        str(row.get("counterfactual_kind", "")),
        str(row.get("row_id", "")),
    )


def _transaction_rejection_reasons(row: Mapping[str, Any]) -> list[str]:
    """Reject any transaction that cannot support a prospective decision."""

    reasons: list[str] = []
    required_text = (
        "row_id",
        "action_identity",
        "causal_edge_id",
        "outcome_identity",
        "write_operation_id",
        "read_operation_id",
        "parent_state_sha256",
        "decision_snapshot_sha256",
        "source_proposal_sha256",
        "receipt_sha256",
    )
    missing = [field for field in required_text if not row.get(field)]
    if missing:
        reasons.append(f"missing_primary_fields:{','.join(missing)}")
    if row.get("operation_admitted") is not True:
        reasons.append("transaction_not_admitted")
    if (
        row.get("counterfactual_applicable") is not True
        or int(row.get("legal_alternative_count") or 0) < 1
    ):
        reasons.append("absent_pre_action_counterfactual")
    identities = (
        "action_identity",
        "causal_edge_id",
        "outcome_identity",
        "parent_state_sha256",
        "decision_snapshot_sha256",
        "source_proposal_sha256",
        "receipt_sha256",
    )
    invalid = [field for field in identities if row.get(field) and not _is_sha256(row.get(field))]
    if invalid:
        reasons.append(f"invalid_content_identity:{','.join(invalid)}")
    return reasons


def _outcome_receipt(row: Mapping[str, Any], source_artifact: str) -> JsonDict | None:
    """Reduce one row-level later receipt to exact authority fields."""

    source_identity = str(row.get("source_event_row_id") or "")
    exact = row.get("exact_outcome")
    if not source_identity or not isinstance(exact, Mapping):
        return None
    if not _is_sha256(exact.get("outcome_identity")) or not _is_sha256(
        exact.get("exact_outcome_hash")
    ):
        return None
    direction = exact.get("signed_direction")
    if not isinstance(direction, int) or isinstance(direction, bool):
        return None
    if row.get("decision_frozen_before_outcome_reveal") is not True:
        return None
    if row.get("outcome_revealed_after_decision") is not True:
        return None
    delayed = {
        "correction_family": row.get("correction_family"),
        "reveal_delay_events": row.get("reveal_delay_events", 0),
        "correction_latency_events": row.get("correction_latency_events", 0),
    }
    return {
        "source_event_identity": source_identity,
        "source_artifact": source_artifact,
        "source_row_id": row.get("row_id"),
        "source_row_sha256": row.get("row_sha256"),
        "outcome_identity": exact.get("outcome_identity"),
        "exact_outcome_hash": exact.get("exact_outcome_hash"),
        "signed_direction": direction,
        "decision_frozen_before_outcome_reveal": True,
        "revealed_after_decision": True,
        "delayed_correction": delayed,
        "arm": row.get("arm"),
        "seed": row.get("seed"),
    }


def _outcome_signature(receipt: Mapping[str, Any]) -> str:
    """Compare repeated arm and seed receipts on their exact outcome only."""

    return sha256_json(
        {
            "outcome_identity": receipt.get("outcome_identity"),
            "exact_outcome_hash": receipt.get("exact_outcome_hash"),
            "signed_direction": receipt.get("signed_direction"),
            "delayed_correction": receipt.get("delayed_correction"),
        }
    )


def build_outcome_index(
    outcomes_a: Mapping[str, Any], outcomes_b: Mapping[str, Any]
) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Index canonical later receipts and reject exact disagreements."""

    grouped: dict[str, list[JsonDict]] = {}
    for source_name, payload in (("outcomes_a", outcomes_a), ("outcomes_b", outcomes_b)):
        source_rows = payload.get("rows", [])
        if not isinstance(source_rows, list):
            continue
        for source_row in source_rows:
            if not isinstance(source_row, Mapping):
                continue
            receipt = _outcome_receipt(source_row, source_name)
            if receipt is not None:
                grouped.setdefault(receipt["source_event_identity"], []).append(receipt)

    index: dict[str, JsonDict] = {}
    conflicts: list[JsonDict] = []
    for source_identity, receipts in grouped.items():
        signatures = {_outcome_signature(row) for row in receipts}
        if len(signatures) != 1:
            conflicts.append(
                {
                    "source_event_identity": source_identity,
                    "exact_outcome_signatures": sorted(signatures),
                }
            )
            continue
        canonical = min(
            receipts,
            key=lambda row: (
                0 if row.get("arm") == "no_memory" else 1,
                int(row.get("seed") or 0),
                str(row.get("source_row_id") or ""),
            ),
        )
        canonical = deepcopy(canonical)
        canonical["support_receipt_count"] = len(receipts)
        canonical["support_receipt_set_sha256"] = sha256_json(
            sorted(str(row.get("source_row_sha256") or "") for row in receipts)
        )
        index[source_identity] = canonical
    return index, conflicts


def _evidence_status(event_id: Any) -> str:
    """Use only event text that was present in the primary decision receipt."""

    text = str(event_id)
    if "stale_prerequisites" in text:
        return "stale"
    if "competing_authorities" in text or "soft_conflict" in text:
        return "conflicting"
    return "current"


def _lookup_outcome(
    event_identity: str, outcome_index: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Join plain shard-A keys or sealed shard-B content identities."""

    direct = outcome_index.get(event_identity)
    if direct is not None:
        return direct
    current = outcome_index.get(sha256_json(event_identity))
    if current is not None:
        return current
    return outcome_index.get(_legacy_receipt_sha256(event_identity))


def _content_binding(row: Mapping[str, Any]) -> JsonDict:
    """Select the fields that form one immutable opportunity identity."""

    primary = row.get("primary_source", {})
    candidate = row.get("candidate_action", {})
    pre_action = row.get("pre_action_state", {})
    counterfactual = row.get("counterfactual", {})
    outcome = row.get("later_exact_outcome", {})
    return {
        "event_identity": row.get("event_identity"),
        "transaction_receipt_sha256": primary.get("transaction_receipt_sha256"),
        "source_proposal_sha256": primary.get("source_proposal_sha256"),
        "candidate_action_identity": candidate.get("action_identity"),
        "parent_state_sha256": pre_action.get("parent_state_sha256"),
        "counterfactual_kind": counterfactual.get("kind"),
        "exact_outcome_hash": outcome.get("exact_outcome_hash"),
        "exact_outcome_source_row_sha256": outcome.get("source_row_sha256"),
    }


def reconstruct_opportunities(
    transaction_rows: Sequence[Mapping[str, Any]],
    outcome_index: Mapping[str, Mapping[str, Any]],
    *,
    sort_rows: bool = True,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Bind each valid primary transaction to one later exact receipt."""

    ordered = (
        sorted(transaction_rows, key=chronological_sort_key)
        if sort_rows
        else list(transaction_rows)
    )
    rows: list[JsonDict] = []
    rejected: list[JsonDict] = []
    for transaction in ordered:
        event_identity = str(transaction.get("row_id") or "")
        reasons = _transaction_rejection_reasons(transaction)
        if reasons:
            rejected.append(
                {
                    "row_kind": "primary_transaction",
                    "event_identity": event_identity,
                    "transaction_receipt_sha256": transaction.get("receipt_sha256"),
                    "reasons": reasons,
                }
            )
            continue
        outcome = _lookup_outcome(event_identity, outcome_index)
        if outcome is None:
            rejected.append(
                {
                    "row_kind": "primary_transaction",
                    "event_identity": event_identity,
                    "transaction_receipt_sha256": transaction.get("receipt_sha256"),
                    "reasons": ["missing_later_exact_outcome"],
                }
            )
            continue
        status = _evidence_status(transaction.get("event_id"))
        row: JsonDict = {
            "event_identity": event_identity,
            "decision_sequence_index": len(rows) + 1,
            "chronological_sort_key": list(chronological_sort_key(transaction)),
            "primary_source": {
                "transaction_artifact": "exp6827",
                "transaction_row_id": event_identity,
                "transaction_receipt_sha256": transaction.get("receipt_sha256"),
                "source_proposal_sha256": transaction.get("source_proposal_sha256"),
                "causal_edge_id": transaction.get("causal_edge_id"),
            },
            "candidate_action": {
                "action_identity": transaction.get("action_identity"),
                "operation_kind": transaction.get("operation_kind"),
                "operation_id": transaction.get("operation_id"),
                "write_operation_id": transaction.get("write_operation_id"),
                "read_operation_id": transaction.get("read_operation_id"),
            },
            "pre_action_state": {
                "parent_state_sha256": transaction.get("parent_state_sha256"),
                "decision_snapshot_sha256": transaction.get("decision_snapshot_sha256"),
            },
            "decision_features": {
                "source_identity": "exp6827",
                "source_content_sha256": transaction.get("receipt_sha256"),
                "family": str(transaction.get("source_family") or ""),
                "chronological_position": int(transaction.get("chronological_position") or 0),
                "operation_kind": transaction.get("operation_kind"),
                "candidate_action_identity": transaction.get("action_identity"),
                "pre_action_state_sha256": transaction.get("parent_state_sha256"),
                "decision_snapshot_sha256": transaction.get("decision_snapshot_sha256"),
                "counterfactual_kind": transaction.get("counterfactual_kind"),
                "counterfactual_available": True,
                "evidence_status_at_decision": status,
            },
            "counterfactual": {
                "kind": transaction.get("counterfactual_kind"),
                "transformation": deepcopy(transaction.get("counterfactual_transformation")),
                "legal_alternative_count": transaction.get("legal_alternative_count"),
                "valid_pre_action": True,
                "action_specific_outcome_fabricated": False,
            },
            "later_exact_outcome": {
                key: deepcopy(value)
                for key, value in outcome.items()
                if key
                not in {
                    "source_event_identity",
                    "decision_frozen_before_outcome_reveal",
                    "arm",
                    "seed",
                }
            },
            "offline_labels": {
                "order_id": transaction.get("order_id"),
                "source_split": transaction.get("split"),
            },
        }
        row["primary_content_sha256"] = sha256_json(_content_binding(row))
        row["action_support"] = action_support(row)
        rows.append(row)
    return rows, rejected


def action_support(row: Mapping[str, Any]) -> JsonDict:
    """Freeze action eligibility from pre-action evidence only."""

    features = row.get("decision_features", {})
    counterfactual = row.get("counterfactual", {})
    valid_alternative = counterfactual.get("valid_pre_action") is True
    stale = features.get("evidence_status_at_decision") == "stale"
    retrievable = features.get("operation_kind") in {"retrieve", "filter"}
    return {
        "no_memory": {
            "supported": valid_alternative,
            "reason": "primary_pre_action_counterfactual",
        },
        "read_only_retrieval": {
            "supported": retrievable,
            "reason": "primary_read_or_filter_receipt"
            if retrievable
            else "no_primary_read_receipt",
        },
        "bounded_update": {
            "supported": valid_alternative and not stale,
            "reason": "stale_evidence_at_decision" if stale else "eligible_for_future_controller",
        },
        "quarantine": {"supported": True, "reason": "fail_closed_safety_action"},
        "v599_unsafe_reference": {
            "supported": True,
            "selectable": False,
            "reason": "frozen_non_authoritative_reference",
        },
    }


def find_leakage_witnesses(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report every offline or unknown field inside a decision context."""

    witnesses: list[JsonDict] = []
    for row in rows:
        features = row.get("decision_features", {})
        if not isinstance(features, Mapping):
            witnesses.append(
                {
                    "event_identity": row.get("event_identity"),
                    "feature": "decision_features",
                    "reason": "decision_context_must_be_mapping",
                }
            )
            continue
        for feature in features:
            if feature not in OBSERVABLE_FEATURES:
                witnesses.append(
                    {
                        "event_identity": row.get("event_identity"),
                        "feature": feature,
                        "reason": "offline_or_unknown_feature_in_decision_context",
                    }
                )
    return witnesses


def validate_stream(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check uniqueness, order, provenance, outcomes, and leakage."""

    errors: list[str] = []
    identities = [str(row.get("event_identity") or "") for row in rows]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        errors.append(f"duplicate_event_identity:{','.join(duplicates)}")
    expected_order = sorted(rows, key=chronological_sort_key)
    if identities != [str(row.get("event_identity") or "") for row in expected_order]:
        errors.append("chronological_order_mismatch")
    if [row.get("decision_sequence_index") for row in rows] != list(range(1, len(rows) + 1)):
        errors.append("decision_sequence_mismatch")
    for row in rows:
        primary = row.get("primary_source", {})
        candidate = row.get("candidate_action", {})
        state = row.get("pre_action_state", {})
        counterfactual = row.get("counterfactual", {})
        outcome = row.get("later_exact_outcome", {})
        provenance = (
            primary.get("transaction_artifact") == "exp6827"
            and _is_sha256(primary.get("transaction_receipt_sha256"))
            and _is_sha256(primary.get("source_proposal_sha256"))
            and _is_sha256(candidate.get("action_identity"))
            and _is_sha256(state.get("parent_state_sha256"))
            and _is_sha256(row.get("primary_content_sha256"))
        )
        if not provenance:
            errors.append(f"invalid_primary_provenance:{row.get('event_identity')}")
        if counterfactual.get("valid_pre_action") is not True:
            errors.append(f"invalid_counterfactual:{row.get('event_identity')}")
        complete_outcome = (
            outcome.get("revealed_after_decision") is True
            and _is_sha256(outcome.get("outcome_identity"))
            and _is_sha256(outcome.get("exact_outcome_hash"))
            and outcome.get("source_artifact") in {"outcomes_a", "outcomes_b"}
        )
        if not complete_outcome:
            errors.append(f"invalid_later_exact_outcome:{row.get('event_identity')}")
        if row.get("primary_content_sha256") != sha256_json(_content_binding(row)):
            errors.append(f"primary_content_identity_mismatch:{row.get('event_identity')}")
    if find_leakage_witnesses(rows):
        errors.append("decision_feature_leakage")
    return errors


def _action_manifest() -> JsonDict:
    """Describe all actions while keeping the unsafe arm non-authoritative."""

    return {
        "no_memory": {
            "mutates_memory": False,
            "description": "Continue without retrieval or write exposure.",
        },
        "read_only_retrieval": {
            "mutates_memory": False,
            "description": "Read supported evidence without changing state.",
        },
        "bounded_update": {
            "mutates_memory": True,
            "description": "Future controller action under the frozen bounds.",
        },
        "quarantine": {
            "mutates_memory": False,
            "description": "Hide a transition that fails an exact check.",
        },
        "v599_unsafe_reference": {
            "mutates_memory": True,
            "selectable": False,
            "authoritative": False,
            "description": "Preserved reference for the retired unsafe behavior.",
        },
    }


def initial_reliability_state_schema() -> JsonDict:
    """Freeze a small symmetric zero state without applying an update."""

    source_nodes = ["exp6827_transactions", "exp6840_outcomes", "exp6841_outcomes"]
    action_nodes = list(ACTION_NAMES)
    nodes = source_nodes + action_nodes
    matrix = [[0.0 for _ in nodes] for _ in nodes]
    return {
        "schema": "carnot.observable_reliability_state.v1",
        "evidence_source_nodes": source_nodes,
        "memory_action_nodes": action_nodes,
        "node_order": nodes,
        "dimension": len(nodes),
        "initial_matrix": matrix,
        "initial_matrix_sha256": sha256_json(matrix),
        "finite": True,
        "symmetric": True,
        "initial_state_unchanged_by_fixture": True,
    }


def _bounded_update_contract() -> JsonDict:
    """Freeze update limits for the downstream controller."""

    return {
        "updates_applied": False,
        "update_timing": "after_action_freeze_and_later_exact_outcome_reveal",
        "same_event_outcome_may_select_action": False,
        "max_abs_entry_delta_per_event": 0.05,
        "max_spectral_norm_delta_per_event": 0.1,
        "nonfinite_feedback_policy": "reject",
        "over_bound_feedback_policy": "reject",
        "state_before_and_after_hash_required": True,
    }


def _base_transition_case() -> JsonDict:
    """Build one valid transition-shaped case for frozen attack mutations."""

    schema = initial_reliability_state_schema()
    state_sha = schema["initial_matrix_sha256"]
    return {
        "affected_keys": ["source:exp6827/action:bounded_update"],
        "content_sha256": sha256_json("content"),
        "source_identity": "exp6827",
        "event_identity": "event-attack-fixture",
        "action": "bounded_update",
        "provenance_complete": True,
        "correction_valid": True,
        "tombstone_reappeared": False,
        "node_order": schema["node_order"],
        "parent_state_sha256": state_sha,
        "rollback_state_sha256": state_sha,
        "proposed_entry_delta": 0.01,
        "proposed_spectral_delta": 0.02,
    }


def _transition_failures(case: Mapping[str, Any]) -> list[str]:
    """Evaluate exact transition checks without changing any state."""

    contract = _bounded_update_contract()
    failures: list[str] = []
    if not case.get("affected_keys"):
        failures.append("omission")
    if not _is_sha256(case.get("content_sha256")):
        failures.append("corruption")
    if case.get("action") not in ACTION_NAMES:
        failures.append("unsupported_insertion")
    if not case.get("source_identity") or case.get("provenance_complete") is not True:
        failures.append("provenance_loss")
    if case.get("correction_valid") is not True:
        failures.append("delayed_invalidation")
    if case.get("tombstone_reappeared") is True:
        failures.append("tombstone_reappearance")
    if case.get("node_order") != initial_reliability_state_schema()["node_order"]:
        failures.append("restart_loss")
    if case.get("rollback_state_sha256") != case.get("parent_state_sha256"):
        failures.append("rollback_mismatch")
    entry_delta = case.get("proposed_entry_delta")
    spectral_delta = case.get("proposed_spectral_delta")
    deltas = (entry_delta, spectral_delta)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in deltas):
        failures.append("poison_nonfinite")
    elif (
        abs(float(entry_delta)) > contract["max_abs_entry_delta_per_event"]
        or abs(float(spectral_delta)) > contract["max_spectral_norm_delta_per_event"]
    ):
        failures.append("poison_over_bound")
    return failures


def transition_attack_manifest() -> list[JsonDict]:
    """Freeze all required attacks and prove their quarantine route."""

    attacks: dict[str, dict[str, Any]] = {
        "omission": {"affected_keys": []},
        "corruption": {"content_sha256": "corrupt"},
        "unsupported_insertion": {"action": "unsupported_insert"},
        "provenance_loss": {"source_identity": "", "provenance_complete": False},
        "delayed_invalidation": {"correction_valid": False},
        "tombstone_reappearance": {"tombstone_reappeared": True},
        "restart_loss": {"node_order": initial_reliability_state_schema()["node_order"][:-1]},
        "rollback_mismatch": {"rollback_state_sha256": sha256_json("wrong-parent")},
        "poison_nonfinite": {"proposed_entry_delta": float("nan")},
        "poison_over_bound": {"proposed_spectral_delta": 1.0},
    }
    rows: list[JsonDict] = []
    for attack_id, mutation in attacks.items():
        case = _base_transition_case()
        case.update(mutation)
        reasons = _transition_failures(case)
        rows.append(
            {
                "attack_id": attack_id,
                "mutation": mutation,
                "detected": attack_id in reasons,
                "detection_reasons": reasons,
                "failure_action": "quarantine",
                "state_update_applied": False,
            }
        )
    return rows


def select_old_family_anchors(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Select the first exact event for each established source family."""

    anchors: list[JsonDict] = []
    seen: set[str] = set()
    for row in sorted(rows, key=chronological_sort_key):
        family = str(row.get("decision_features", {}).get("family") or "")
        if family and family not in seen:
            anchors.append(
                {
                    "family": family,
                    "event_identity": row.get("event_identity"),
                    "primary_content_sha256": row.get("primary_content_sha256"),
                    "later_exact_outcome_hash": row.get("later_exact_outcome", {}).get(
                        "exact_outcome_hash"
                    ),
                }
            )
            seen.add(family)
    return anchors


def deterministic_order_replicates(
    rows: Sequence[Mapping[str, Any]], anchors: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze seeded permutations while retaining every event and anchor."""

    identities = [str(row.get("event_identity") or "") for row in rows]
    families = sorted(
        {str(row.get("decision_features", {}).get("family") or "") for row in rows} - {""}
    )
    anchor_ids = [str(row.get("event_identity") or "") for row in anchors]
    replicates: list[JsonDict] = []
    for index, seed in enumerate(ORDER_REPLICATE_SEEDS):
        order = identities.copy()
        random.Random(seed).shuffle(order)
        held_out_family = families[index % len(families)] if families else ""
        held_out = [
            str(row.get("event_identity") or "")
            for row in rows
            if row.get("decision_features", {}).get("family") == held_out_family
        ]
        replicates.append(
            {
                "replicate_id": f"order_replicate_{index + 1}",
                "seed": seed,
                "event_identities": order,
                "order_sha256": sha256_json(order),
                "held_out_family": held_out_family,
                "family_held_out_event_identities": held_out,
                "old_family_anchor_event_identities": anchor_ids,
                "all_events_preserved": len(order) == len(identities)
                and set(order) == set(identities),
            }
        )
    return replicates


def _observable_manifest() -> JsonDict:
    """Label every controller feature that exists at decision time."""

    return {
        "label": "decision_time_observable",
        "decision_feature_fields": sorted(OBSERVABLE_FEATURES),
        "row_field_labels": {
            "event_identity": "decision_time_observable",
            "decision_sequence_index": "decision_time_observable",
            "primary_source": "decision_time_observable",
            "candidate_action": "decision_time_observable",
            "pre_action_state": "decision_time_observable",
            "decision_features": "decision_time_observable",
            "counterfactual": "decision_time_observable",
            "action_support": "decision_time_observable",
            "primary_content_sha256": "decision_time_observable",
        },
        "unknown_decision_features_fail_closed": True,
    }


def _offline_manifest() -> JsonDict:
    """Label supervision that can arrive only after the action freezes."""

    return {
        "label": "offline_supervision_only",
        "fields": sorted(OFFLINE_SUPERVISION_FIELDS),
        "row_field_labels": {
            "later_exact_outcome": "offline_supervision_only",
            "offline_labels": "offline_supervision_only",
            "chronological_sort_key": "offline_supervision_only",
        },
        "same_event_supervision_may_select_action": False,
    }


def _leakage_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run fixed leakage probes in addition to checking the clean rows."""

    clean = find_leakage_witnesses(rows)
    probes: list[JsonDict] = []
    if rows:
        for feature in (
            "later_exact_outcome",
            "future_correction",
            "audit_label",
            "task_order_label",
        ):
            attacked = deepcopy(rows[0])
            attacked["decision_features"][feature] = "injected"
            detected = find_leakage_witnesses([attacked])
            probes.append(
                {
                    "feature": feature,
                    "detected": any(row.get("feature") == feature for row in detected),
                }
            )
    return {
        "clean_stream_witnesses": clean,
        "attack_probes": probes,
        "passed": not clean and len(probes) == 4 and all(row["detected"] for row in probes),
    }


def _field_principles(fields: Sequence[str]) -> JsonDict:
    """Explain why each top-level artifact field exists."""

    specific = {
        "field_principles": "Each field records why its evidence matters.",
        "preconditions_checked": "Exact gates prevent reconstruction from missing evidence.",
        "inference_substrate": "The substrate limits the artifact to deterministic receipt reduction.",
        "duration_s": "Measured wall time exposes the actual local work.",
        "source_artifact_hashes": "File hashes bind all inputs to exact bytes.",
        "primary_receipt_manifest": "The manifest separates row authority from excluded aggregates.",
        "rows": "Per-event rows keep decisions and later outcomes independently auditable.",
        "rejected_opportunity_rows": "Rejections prevent aggregate prose or incomplete receipts from disappearing.",
        "observable_feature_manifest": "The allowlist prevents future supervision from selecting its own action.",
        "offline_supervision_manifest": "The offline label keeps later authority outside the decision context.",
        "leakage_witnesses": "Leakage probes prove the observable boundary fails closed.",
        "action_manifest": "Frozen actions prevent the controller from changing its comparison after outcomes.",
        "counterfactual_support_rows": "Per-event support prevents invented alternatives.",
        "reliability_state_schema": "A fixed symmetric state gives the next experiment a bounded substrate.",
        "bounded_update_contract": "Frozen bounds limit the effect of any later event.",
        "chronological_order_manifest": "The base order prevents future outcomes from moving earlier.",
        "order_replicate_manifest": "Seeded orders expose dependence on one chronology.",
        "old_family_anchor_rows": "Anchors preserve established families in every replicate.",
        "transition_attack_manifest": "Attacks freeze exact quarantine failures before controller work.",
        "random_seed": "A fixed seed makes every ordering choice reproducible.",
        "reproducibility_checksum": "The checksum detects drift in deterministic artifact content.",
        "observable_reliability_stream_ready_score": "The exact downstream gate is fully conjunctive.",
        "gate_check_summary": "Expected and observed values explain every blocked result.",
        "verifier_is_oracle": "False keeps the reducer from authorizing its own evidence.",
        "verdict_class": "The closed class states the evidence boundary.",
        "honest_verdict": "A complete prefix gives the conductor a terminal result.",
    }
    defaults = {
        "schema": "The versioned schema prevents silent contract drift.",
        "experiment_id": "The numeric identity links code, tests, and result.",
        "run_date": "The requested date records when reconstruction ran.",
        "status": "The status distinguishes a terminal artifact from partial output.",
        "openspec_requirement_ids": "Requirement IDs link each result to its contract.",
        "replay_commands": "Replay commands make verification repeatable.",
    }
    return {field: specific[field] if field in specific else defaults[field] for field in fields}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic content while excluding wall time and this checksum."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def build_artifact(
    root: Path,
    run_date: str,
    *,
    live_reports: Mapping[str, Mapping[str, Any]] | None = None,
    primary_source_paths: Mapping[str, str] = PRIMARY_SOURCE_PATHS,
) -> JsonDict:
    """Reconstruct and freeze the complete downstream controller input."""

    started = time.perf_counter()
    documents: dict[str, JsonDict] = {}
    source_errors: dict[str, str] = {}
    source_rows: list[JsonDict] = []
    for source_id, relative in primary_source_paths.items():
        document, error = _read_json(root / relative)
        documents[source_id] = document
        if error:
            source_errors[source_id] = error
        source_rows.append(
            {
                "source_id": source_id,
                "path": relative,
                "readable": error is None,
                "error": error,
                "sha256": sha256_file(root / relative),
            }
        )

    basic_sources_ready = not source_errors
    if live_reports is None:
        live_reports = collect_live_adversarial_reports(root) if basic_sources_ready else {}
    live_status = _live_status_rows(live_reports)
    evidence = documents.get("evidence_contract", {})
    transactions_value = documents.get("transactions", {}).get("rows", [])
    transactions = transactions_value if isinstance(transactions_value, list) else []
    outcomes_a = documents.get("outcomes_a", {})
    outcomes_b = documents.get("outcomes_b", {})
    outcome_index, outcome_conflicts = build_outcome_index(outcomes_a, outcomes_b)
    rows, rejected = reconstruct_opportunities(
        [row for row in transactions if isinstance(row, Mapping)], outcome_index
    )
    rejected.extend(
        {
            "row_kind": "aggregate_source",
            "event_identity": None,
            "source_id": source_id,
            "reasons": ["aggregate_not_primary_authority"],
        }
        for source_id in V599_AGGREGATE_PATHS
    )

    leakage = _leakage_manifest(rows)
    anchors = select_old_family_anchors(rows)
    replicates = deterministic_order_replicates(rows, anchors)
    attacks = transition_attack_manifest()
    stream_errors = validate_stream(rows)
    counterfactual_rows = [
        {
            "event_identity": row["event_identity"],
            "kind": row["counterfactual"]["kind"],
            "transformation": row["counterfactual"]["transformation"],
            "legal_alternative_count": row["counterfactual"]["legal_alternative_count"],
            "valid_pre_action": row["counterfactual"]["valid_pre_action"],
            "action_specific_outcome_fabricated": False,
            "supported_actions": row["action_support"],
        }
        for row in rows
    ]
    families = sorted(
        {str(row.get("decision_features", {}).get("family") or "") for row in rows} - {""}
    )
    anchor_ids = {row["event_identity"] for row in anchors}
    attacks_by_id = {row["attack_id"]: row for row in attacks}

    checks = [
        _check(
            "required_source_readability",
            {source_id: "readable" for source_id in primary_source_paths},
            source_errors,
            basic_sources_ready,
        ),
        _check(
            "v601_evidence_contract_ready_score",
            1,
            evidence.get("v601_evidence_contract_ready_score"),
            evidence.get("v601_evidence_contract_ready_score") == 1,
        ),
        _check(
            "primary_transaction_rows",
            ">0",
            len(transactions),
            len(transactions) > 0,
        ),
        _check(
            "primary_exact_outcome_rows",
            {"outcomes_a": ">0", "outcomes_b": ">0"},
            {
                "outcomes_a": len(outcomes_a.get("rows", []))
                if isinstance(outcomes_a.get("rows"), list)
                else 0,
                "outcomes_b": len(outcomes_b.get("rows", []))
                if isinstance(outcomes_b.get("rows"), list)
                else 0,
            },
            isinstance(outcomes_a.get("rows"), list)
            and bool(outcomes_a.get("rows"))
            and isinstance(outcomes_b.get("rows"), list)
            and bool(outcomes_b.get("rows")),
        ),
        _check(
            "current_live_v599_adversarial_status",
            sorted(V599_AGGREGATE_PATHS),
            sorted(row["source_id"] for row in live_status if row["current_status_available"]),
            len(live_status) == len(V599_AGGREGATE_PATHS)
            and all(row["current_status_available"] for row in live_status),
        ),
        _check("exact_outcome_conflicts", [], outcome_conflicts, not outcome_conflicts),
        _check("accepted_primary_events", ">0", len(rows), len(rows) > 0),
        _check("stream_validation", [], stream_errors, not stream_errors),
        _check(
            "counterfactual_support",
            {"invalid_count": 0},
            {
                "invalid_count": len(
                    [row for row in counterfactual_rows if row["valid_pre_action"] is not True]
                )
            },
            bool(counterfactual_rows)
            and all(row["valid_pre_action"] is True for row in counterfactual_rows),
        ),
        _check("decision_time_leakage", True, leakage["passed"], leakage["passed"] is True),
        _check(
            "old_family_anchors",
            families,
            sorted(str(row["family"]) for row in anchors),
            bool(families) and sorted(str(row["family"]) for row in anchors) == families,
        ),
        _check(
            "order_replicates_preserve_events_and_anchors",
            {"minimum_replicates": 5, "all_anchors": sorted(anchor_ids)},
            {
                "replicate_count": len(replicates),
                "all_preserved": all(
                    row["all_events_preserved"]
                    and set(row["old_family_anchor_event_identities"]) == anchor_ids
                    for row in replicates
                ),
            },
            len(replicates) >= 5
            and all(
                row["all_events_preserved"]
                and set(row["old_family_anchor_event_identities"]) == anchor_ids
                for row in replicates
            ),
        ),
        _check(
            "transition_attacks",
            sorted(REQUIRED_ATTACK_IDS),
            sorted(attack_id for attack_id, row in attacks_by_id.items() if row["detected"]),
            set(attacks_by_id) == REQUIRED_ATTACK_IDS
            and all(row["detected"] and row["failure_action"] == "quarantine" for row in attacks),
        ),
    ]
    gate_summary = _gate_summary(checks)
    ready = int(gate_summary["passed"])
    source_hashes = _source_hashes(root)
    excluded_aggregates = []
    for status_row in live_status:
        source_id = status_row["source_id"]
        excluded_aggregates.append(
            {
                **status_row,
                "sha256": source_hashes[source_id]["sha256"],
                "authoritative": False,
                "exclusion_reason": "aggregate_not_primary_authority",
            }
        )

    artifact: JsonDict = {
        "schema": "carnot.experiment_6871.observable_reliability_opportunity_stream.v1",
        "experiment_id": 6871,
        "run_date": run_date,
        "status": "complete",
        "openspec_requirement_ids": [
            "REQ-LEARN-6871",
            "SCENARIO-LEARN-6871-MISSING-PRIMARY",
            "SCENARIO-LEARN-6871-DUPLICATE",
            "SCENARIO-LEARN-6871-LEAKAGE",
            "SCENARIO-LEARN-6871-ORDER",
            "SCENARIO-LEARN-6871-COUNTERFACTUAL",
            "SCENARIO-LEARN-6871-STALE",
            "SCENARIO-LEARN-6871-DELAYED-CORRECTION",
            "SCENARIO-LEARN-6871-POISON",
            "SCENARIO-LEARN-6871-RESTART",
            "SCENARIO-LEARN-6871-ROLLBACK",
        ],
        "replay_commands": [
            ".venv/bin/python scripts/experiments/experiment_6871_observable_reliability_opportunity_stream.py --date 20260902",
            ".venv/bin/pytest tests/python/test_experiment_6871_observable_reliability_opportunity_stream.py -q",
        ],
        "field_principles": {},
        "preconditions_checked": {
            "sources": source_rows,
            "v599_live_adversarial_status": live_status,
            "passed": checks[0]["passed"] and checks[1]["passed"] and checks[4]["passed"],
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.perf_counter() - started, 6),
        "source_artifact_hashes": source_hashes,
        "primary_receipt_manifest": {
            "transaction_source": {
                "source_id": "exp6827",
                "path": primary_source_paths.get("transactions"),
                "row_count": len(transactions),
                "accepted_event_count": len(rows),
                "authoritative": True,
            },
            "exact_outcome_sources": [
                {
                    "source_id": "exp6840",
                    "path": primary_source_paths.get("outcomes_a"),
                    "row_count": len(outcomes_a.get("rows", []))
                    if isinstance(outcomes_a.get("rows"), list)
                    else 0,
                    "authoritative": True,
                },
                {
                    "source_id": "exp6841",
                    "path": primary_source_paths.get("outcomes_b"),
                    "row_count": len(outcomes_b.get("rows", []))
                    if isinstance(outcomes_b.get("rows"), list)
                    else 0,
                    "authoritative": True,
                },
            ],
            "excluded_v599_aggregates": excluded_aggregates,
            "aggregate_headlines_used_as_authority": False,
            "producer_modules_imported": [],
        },
        "rows": rows,
        "rejected_opportunity_rows": rejected,
        "observable_feature_manifest": _observable_manifest(),
        "offline_supervision_manifest": _offline_manifest(),
        "leakage_witnesses": leakage,
        "action_manifest": _action_manifest(),
        "counterfactual_support_rows": counterfactual_rows,
        "reliability_state_schema": initial_reliability_state_schema(),
        "bounded_update_contract": _bounded_update_contract(),
        "chronological_order_manifest": {
            "ordering_rule": "order number, family, chronological position, counterfactual kind, event identity",
            "event_identities": [row["event_identity"] for row in rows],
            "base_order_sha256": sha256_json([row["event_identity"] for row in rows]),
            "base_order_is_chronological": "chronological_order_mismatch" not in stream_errors,
            "event_count": len(rows),
        },
        "order_replicate_manifest": replicates,
        "old_family_anchor_rows": anchors,
        "transition_attack_manifest": attacks,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "observable_reliability_stream_ready_score": ready,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "blocked",
        "honest_verdict": READY_VERDICT if ready else BLOCKED_VERDICT,
    }
    artifact["field_principles"] = _field_principles(list(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay the output contract before the result file is replaced."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        errors.append(f"missing_required_fields:{','.join(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_must_cover_every_field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    ready = artifact.get("observable_reliability_stream_ready_score")
    if ready not in {0, 1}:
        errors.append("invalid_ready_score")
    if ready == 1:
        gate = artifact.get("gate_check_summary", {})
        rows = artifact.get("rows", [])
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append("ready_artifact_has_failed_gate")
        if isinstance(rows, list) and find_leakage_witnesses(rows):
            errors.append("ready_artifact_has_leakage")
        if artifact.get("verdict_class") != "positive":
            errors.append("ready_artifact_must_be_positive")
    if ready == 0:
        gate = artifact.get("gate_check_summary", {})
        if not isinstance(gate, Mapping) or not gate.get("failed_check"):
            errors.append("blocked_artifact_requires_failed_check")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_artifact_must_use_blocked_class")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the target only after a complete temporary JSON file exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the requested date, validate the artifact, and write it once."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", default=RESULT_RELATIVE_PATH.as_posix())
    args = parser.parse_args(argv)
    artifact = build_artifact(REPO_ROOT, args.date)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    output = Path(args.output)
    write_atomic(output, artifact)
    print(
        json.dumps(
            {
                "result": output.as_posix(),
                "ready_score": artifact["observable_reliability_stream_ready_score"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        )
    )
    return 0
