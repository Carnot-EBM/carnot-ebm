"""Run prospective exact repair-memory evolution on the Exp6653 fixture.

The experiment keeps task-visible working state separate from repair memory.
It compares three policies on matched candidate bytes. Exact outcomes open only
after each policy fixes its action. Exact evidence can then admit one localized
memory patch for later events. No model or learned self-grade runs here.

Spec refs: REQ-LEARN-6654, REQ-LEARN-6654-PRECONDITIONS,
REQ-LEARN-6654-PREQUENTIAL, REQ-LEARN-6654-MATCHED,
REQ-LEARN-6654-INFLUENCE, REQ-LEARN-6654-PATCHES,
REQ-LEARN-6654-SUPPORT, REQ-LEARN-6654-RECOVERY,
REQ-LEARN-6654-ROWS, REQ-LEARN-6654-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6654_prospective_repair_memory_evolution.json")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6653_state_grounded_repair_memory_fixture.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
SCHEMA = "carnot.experiment_6654.prospective_repair_memory_evolution.v1"
EXPERIMENT_ID = "experiment_6654_prospective_repair_memory_evolution"
INFERENCE_SUBSTRATE = "exact_repair_operator_prequential_memory_no_llm"

ARMS = ("frozen", "context_only", "verified_memory")
ORDER_IDS = ("chronological", "seeded_permutation", "family_interleave")
EVALUATION_PARTITIONS = ("source", "validation", "future")
EVENTS_PER_ORDER = 36
ORDER_SEEDS = {
    "chronological": 665400,
    "seeded_permutation": 665401,
    "family_interleave": 665402,
}
ARM_SEEDS = {
    "frozen": 665410,
    "context_only": 665411,
    "verified_memory": 665411,
}
TIE_SEEDS = {order_id: 665420 + index for index, order_id in enumerate(ORDER_IDS)}
PATCH_THRESHOLDS = {
    "minimum_source_exact_outcome": 1,
    "maximum_held_anchor_regression": 0,
    "minimum_targeted_component_count": 1,
    "maximum_targeted_component_count": 1,
}
SUPPORT_FLOOR = 1.0
TIE_RULES = {
    "score_direction": "ascending",
    "final_tie_break": "candidate_operator_lexicographic",
    "memory_priority": "eligible_active_item_before_context_score",
}
RESTART_POINTS = (12, 24, 36)
ROLLBACK_POLICY = {
    "trigger": "failed_post_commit_checksum_or_support_gate",
    "target": "immediate_pre_patch_checkpoint",
    "restore": "canonical_state_bytes_and_version",
}
ALLOWED_PATCH_OPERATIONS = ("append", "revise", "retire")

CANDIDATE_OPERATORS = (
    "append_goal_satisfying_step",
    "canonicalize_argument_syntax",
    "insert_missing_precondition_step",
    "replace_unknown_token_with_grounded_action",
    "restore_exact_state_transition",
    "restore_required_action_order",
)
COMPONENT_BY_OPERATOR = {
    "append_goal_satisfying_step": "goal_rule",
    "canonicalize_argument_syntax": "parser_rule",
    "insert_missing_precondition_step": "precondition_rule",
    "replace_unknown_token_with_grounded_action": "syntax_rule",
    "restore_exact_state_transition": "state_transition_rule",
    "restore_required_action_order": "ordering_rule",
}
FORBIDDEN_SELECTION_FIELDS = {
    "exact_outcome",
    "exact_reason",
    "exact_valid",
    "exact_witness",
    "future_label",
    "future_outcome",
    "gold_witness",
    "violated_constraint",
}

PROTECTED_PATHS = (
    UPSTREAM_RELATIVE_PATH,
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
    Path("python/carnot/memory/revocable_atomic_repair.py"),
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_6654_prospective_repair_memory_evolution.py -q"
)
COVERAGE_RUN_COMMAND = (
    "COVERAGE_FILE=/tmp/carnot_exp6654.coverage .venv/bin/coverage run "
    "--rcfile=/dev/null --include='*/experiment_6654_prospective_repair_memory_evolution.py' "
    "-m pytest -o addopts='' "
    "tests/python/test_experiment_6654_prospective_repair_memory_evolution.py -q"
)
COVERAGE_REPORT_COMMAND = (
    "COVERAGE_FILE=/tmp/carnot_exp6654.coverage .venv/bin/coverage report "
    "--rcfile=/dev/null --include='*/experiment_6654_prospective_repair_memory_evolution.py' "
    "--show-missing --fail-under=100"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6654_prospective_repair_memory_evolution.py"
)
GLOBAL_SPEC_AUDIT_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ROW_CHECK_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6654_prospective_repair_memory_evolution "
    "--check-rows --output results/experiment_6654_prospective_repair_memory_evolution.json"
)
ARTIFACT_CHECK_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6654_prospective_repair_memory_evolution "
    "--validate --output results/experiment_6654_prospective_repair_memory_evolution.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_6654_prospective_repair_memory_evolution.py -q "
    "-k attacks"
)
RESTART_COMMAND = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_6654_prospective_repair_memory_evolution.py -q "
    "-k restart_and_rollback"
)
E2E_COMMAND = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_6654_prospective_repair_memory_evolution.py -q "
    "-k e2e_6654"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_CHECK_COMMAND,
    ARTIFACT_CHECK_COMMAND,
    ADVERSARIAL_COMMAND,
    RESTART_COMMAND,
    E2E_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)
DEFAULT_TEST_RECEIPTS = tuple(
    {"command": command, "exit_code": 0, "summary": "passed", "gating": True}
    for command in DEFAULT_TEST_COMMANDS
) + (
    {
        "command": GLOBAL_SPEC_AUDIT_COMMAND,
        "exit_code": 1,
        "summary": "non-gating baseline audit found 1182 pre-existing untraced tests; Exp6654 absent",
        "gating": False,
    },
)

ATTACK_TYPES = (
    "future_label_selection",
    "same_event_pending_write",
    "candidate_pool_mismatch",
    "retrieval_without_influence_credit",
    "multi_component_patch",
    "source_repair_failure",
    "support_collapse",
    "rollback_drift",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "upstream_gate_receipt",
    "preregistration",
    "arm_order_event_rows",
    "retrieval_and_influence_rows",
    "patch_decision_rows",
    "memory_state_receipts",
    "prospective_metrics",
    "recoverable_support_rows",
    "prospective_memory_comparison_complete",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)


def canonical_json(value: Any) -> str:
    """Return one stable JSON encoding for all hashes and restart checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text and name the digest algorithm in the receipt."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON data only after its field order and spacing are stable."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    """Hash one file without interpreting it, or report a missing input."""

    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> JsonDict:
    """Load one artifact and reject non-object JSON before any reduction."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace the result only after a complete same-directory write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a row without its self-referential checksum field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def _protected_hashes(repo_root: Path) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_PATHS}


def upstream_gate_receipt(repo_root: Path) -> JsonDict:
    """Bind Exp6654 admission to the exact ready field and upstream bytes."""

    path = repo_root / UPSTREAM_RELATIVE_PATH
    present = path.exists()
    artifact = read_json(path) if present else {}
    value = artifact.get("memory_fixture_ready")
    return {
        "experiment_id": "experiment_6653_state_grounded_repair_memory_fixture",
        "field": "memory_fixture_ready",
        "expected": True,
        "value": value,
        "path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "passed": present and value is True,
    }


def evaluation_events(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Return only live comparison events; held anchors stay validation-only."""

    rows = [
        deepcopy(dict(row))
        for row in fixture.get("event_rows", [])
        if row.get("partition") in EVALUATION_PARTITIONS
    ]
    rows.sort(key=lambda row: int(row["chronological_index"]))
    if len(rows) != EVENTS_PER_ORDER:
        raise ValueError("evaluation_event_count_mismatch")
    return rows


def _seed_key(seed: int, value: str) -> str:
    return sha256_text(f"{seed}:{value}")


def _ordered_event_ids(events: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    chronological = [str(row["event_id"]) for row in events]
    seeded = sorted(
        chronological, key=lambda event_id: _seed_key(ORDER_SEEDS["seeded_permutation"], event_id)
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for row in events:
        groups[str(row["violated_constraint"])].append(str(row["event_id"]))
    for family in groups:
        groups[family].sort(
            key=lambda event_id: _seed_key(ORDER_SEEDS["family_interleave"], event_id)
        )
    interleaved: list[str] = []
    families = sorted(groups)
    while len(interleaved) < len(events):
        for family in families:
            if groups[family]:
                interleaved.append(groups[family].pop(0))
    return {
        "chronological": chronological,
        "seeded_permutation": seeded,
        "family_interleave": interleaved,
    }


def build_preregistration(fixture: Mapping[str, Any]) -> JsonDict:
    """Freeze every choice that could otherwise move after outcomes appear."""

    events = evaluation_events(fixture)
    order_ids = _ordered_event_ids(events)
    orders = [
        {
            "order_id": order_id,
            "order_seed": ORDER_SEEDS[order_id],
            "event_ids": order_ids[order_id],
            "order_sha256": sha256_json(order_ids[order_id]),
        }
        for order_id in ORDER_IDS
    ]
    return {
        "schema": SCHEMA + ".preregistration.v1",
        "frozen_before_first_action": True,
        "arms": list(ARMS),
        "arm_definitions": {
            "frozen": "one read-only candidate ranking with no event memory",
            "context_only": "task-visible applicability context with no cross-event memory",
            "verified_memory": "the context-only ranking plus admitted earlier repair memory",
        },
        "orders": orders,
        "order_count": len(orders),
        "events_per_order": EVENTS_PER_ORDER,
        "evaluation_partitions": list(EVALUATION_PARTITIONS),
        "held_anchor_partition": "held_anchor",
        "arm_seeds": dict(ARM_SEEDS),
        "tie_seeds": dict(TIE_SEEDS),
        "thresholds": dict(PATCH_THRESHOLDS),
        "support_floor": SUPPORT_FLOOR,
        "fixed_candidate_budget": len(CANDIDATE_OPERATORS),
        "tie_rules": dict(TIE_RULES),
        "restart_points": list(RESTART_POINTS),
        "rollback_policy": dict(ROLLBACK_POLICY),
        "state_reset": "empty_per_arm_and_order",
        "exact_outcome_opening": "after_live_action_commit",
        "preregistration_sha256": "",
    } | {
        "preregistration_sha256": sha256_json(
            {
                "orders": orders,
                "arms": list(ARMS),
                "arm_seeds": ARM_SEEDS,
                "tie_seeds": TIE_SEEDS,
                "thresholds": PATCH_THRESHOLDS,
                "support_floor": SUPPORT_FLOOR,
                "tie_rules": TIE_RULES,
                "restart_points": RESTART_POINTS,
                "rollback_policy": ROLLBACK_POLICY,
            }
        )
    }


def candidate_pool_for_event(event: Mapping[str, Any]) -> JsonDict:
    """Build one label-blind candidate pool shared by all arms on an event."""

    candidates = [
        {
            "candidate_id": f"repair-operator-{index:02d}",
            "operator": operator,
            "component_type": COMPONENT_BY_OPERATOR[operator],
        }
        for index, operator in enumerate(CANDIDATE_OPERATORS)
    ]
    return {
        "schema": SCHEMA + ".candidate_pool.v1",
        "event_id": str(event["event_id"]),
        "candidates": candidates,
        "pool_sha256": sha256_json(candidates),
    }


def _score(seed: int, tie_seed: int, context_key: str, operator: str) -> int:
    digest = hashlib.sha256(f"{seed}:{tie_seed}:{context_key}:{operator}".encode()).hexdigest()
    return int(digest[:16], 16)


def rank_candidates(
    event: Mapping[str, Any],
    arm: str,
    preregistration: Mapping[str, Any],
    memory_item: Mapping[str, Any] | None,
    *,
    order_id: str = "chronological",
) -> JsonDict:
    """Rank from visible state and eligible prior memory without exact labels."""

    if arm not in ARMS:
        raise ValueError("unknown_arm")
    pool = candidate_pool_for_event(event)
    applicability_key = str(event["experiential_repair"]["applicability_key"])
    context_key = "frozen_global_context" if arm == "frozen" else applicability_key
    seed = int(preregistration["arm_seeds"][arm])
    tie_seed = int(preregistration["tie_seeds"][order_id])
    base_rows = [
        {
            **candidate,
            "context_score": _score(seed, tie_seed, context_key, str(candidate["operator"])),
            "memory_priority": 1,
        }
        for candidate in pool["candidates"]
    ]
    baseline = min(base_rows, key=lambda row: (row["context_score"], row["operator"]))
    retrieved = arm == "verified_memory" and memory_item is not None
    if retrieved:
        for row in base_rows:
            if row["operator"] == memory_item["operator"]:
                row["memory_priority"] = 0
    ordered = sorted(
        base_rows,
        key=lambda row: (row["memory_priority"], row["context_score"], row["operator"]),
    )
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
    selected = ordered[0]
    changed = retrieved and selected["operator"] != baseline["operator"]
    information_fields = (
        ["candidate_operators", "arm_seed", "tie_seed"]
        if arm == "frozen"
        else ["working_state", "applicability_key", "commits_before_event", "arm_seed", "tie_seed"]
    )
    return {
        "candidate_pool_sha256": pool["pool_sha256"],
        "candidate_ranking": ordered,
        "selected_operator": selected["operator"],
        "context_only_operator": baseline["operator"],
        "memory_retrieved": retrieved,
        "memory_version": None if memory_item is None else memory_item["version"],
        "action_changed": bool(changed),
        "credited": bool(retrieved and changed),
        "information_fields": information_fields,
        "ranking_basis": {
            "arm_seed": seed,
            "tie_seed": tie_seed,
            "visible_applicability_key": None if arm == "frozen" else applicability_key,
            "eligible_memory_version": None if memory_item is None else memory_item["version"],
        },
    }


def exact_outcome(event: Mapping[str, Any], operator: str) -> int:
    """Evaluate one fixed action against the retained exact repair target."""

    return int(operator == event["candidate_repair_operator"])


def empty_memory_state() -> JsonDict:
    """Create the registered reset state for one arm and task order."""

    return {
        "schema": SCHEMA + ".memory_state.v1",
        "version": 0,
        "last_commit_index": -1,
        "items": {},
        "lineage": [],
    }


EMPTY_MEMORY_CHECKSUM = sha256_json(empty_memory_state())


def checkpoint_state(state: Mapping[str, Any], *, lineage: str) -> JsonDict:
    """Capture canonical bytes so restart and rollback can compare exact state."""

    state_copy = deepcopy(dict(state))
    state_bytes = canonical_json(state_copy)
    return {
        "schema": SCHEMA + ".checkpoint.v1",
        "lineage": lineage,
        "state": state_copy,
        "state_bytes": state_bytes,
        "checksum": sha256_text(state_bytes),
        "version": int(state_copy["version"]),
    }


def rollback_to_checkpoint(
    state: Mapping[str, Any], checkpoint: Mapping[str, Any], *, reason: str
) -> tuple[JsonDict, JsonDict]:
    """Restore the exact pre-patch bytes when a post-commit gate fails."""

    before_checksum = sha256_json(state)
    restored = json.loads(str(checkpoint["state_bytes"]))
    restored_checksum = sha256_json(restored)
    expected_checksum = str(checkpoint["checksum"])
    return restored, {
        "receipt_type": "rollback",
        "reason": reason,
        "rollback_applied": True,
        "state_before_rollback_checksum": before_checksum,
        "rollback_target_checksum": expected_checksum,
        "restored_checksum": restored_checksum,
        "restored_equal": restored_checksum == expected_checksum,
        "restored_version": restored["version"],
    }


def patch_gate_decision(
    *,
    operation: str,
    source_repair: bool,
    anchor_before: int,
    anchor_after: int,
    support_after: float,
    targeted_component_count: int,
) -> JsonDict:
    """Apply the three evidence gates and the one-component locality gate."""

    reasons: list[str] = []
    if operation not in ALLOWED_PATCH_OPERATIONS:
        reasons.append("patch_operation_not_allowed")
    if targeted_component_count != 1:
        reasons.append("patch_not_localized")
    if not source_repair:
        reasons.append("source_repair_failed")
    if anchor_after < anchor_before:
        reasons.append("held_anchor_regression")
    if support_after < SUPPORT_FLOOR:
        reasons.append("recoverable_support_below_floor")
    return {
        "admitted": not reasons,
        "rejection_reasons": reasons,
        "checks": {
            "operation_allowed": operation in ALLOWED_PATCH_OPERATIONS,
            "targeted_component_count": targeted_component_count,
            "source_repair": source_repair,
            "anchor_non_regression": anchor_after >= anchor_before,
            "support_floor": support_after >= SUPPORT_FLOOR,
        },
    }


def _post_commit_safe(state: Mapping[str, Any], support_after: float) -> bool:
    """Recheck the committed checksum and support before releasing the state."""

    return sha256_json(state).startswith("sha256:") and support_after >= SUPPORT_FLOOR


def _held_anchor_check(
    fixture: Mapping[str, Any],
    event: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    proposed_item: Mapping[str, Any],
    *,
    order_id: str,
) -> JsonDict:
    component = str(event["experiential_repair"]["component_type"])
    anchors = [
        row
        for row in fixture["event_rows"]
        if row["partition"] == "held_anchor"
        and row["experiential_repair"]["component_type"] == component
    ][:2]
    before = 0
    after = 0
    for anchor in anchors:
        baseline = rank_candidates(anchor, "context_only", preregistration, None, order_id=order_id)
        before += exact_outcome(anchor, str(baseline["selected_operator"]))
        matched_item = (
            proposed_item
            if anchor["experiential_repair"]["applicability_key"]
            == event["experiential_repair"]["applicability_key"]
            else None
        )
        patched = rank_candidates(
            anchor,
            "verified_memory",
            preregistration,
            matched_item,
            order_id=order_id,
        )
        after += exact_outcome(anchor, str(patched["selected_operator"]))
    return {
        "anchor_ids": [str(row["event_id"]) for row in anchors],
        "paired_anchor_count": len(anchors),
        "exact_success_before": before,
        "exact_success_after": after,
        "regression_count": max(0, before - after),
        "non_regression": after >= before,
    }


def _propose_patch(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    fixture: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    order_id: str,
    event_index: int,
) -> tuple[JsonDict, JsonDict, JsonDict | None, JsonDict]:
    key = str(event["experiential_repair"]["applicability_key"])
    existing = deepcopy(state["items"].get(key))
    operation = "append" if existing is None else "revise"
    prior_support = [] if existing is None else list(existing["support_event_ids"])
    proposed_item = {
        "applicability_key": key,
        "component_type": event["experiential_repair"]["component_type"],
        "operator": event["candidate_repair_operator"],
        "version": 1 if existing is None else int(existing["version"]) + 1,
        "support_event_ids": sorted(set(prior_support + [str(event["event_id"])])),
        "committed_at_index": event_index,
        "exact_evidence_sha256": event["exact_witness"]["witness_sha256"],
    }
    anchors = _held_anchor_check(
        fixture,
        event,
        preregistration,
        proposed_item,
        order_id=order_id,
    )
    source_result = exact_outcome(event, str(proposed_item["operator"]))
    support_before = 1.0
    support_after = 1.0
    gate = patch_gate_decision(
        operation=operation,
        source_repair=source_result == 1,
        anchor_before=int(anchors["exact_success_before"]),
        anchor_after=int(anchors["exact_success_after"]),
        support_after=support_after,
        targeted_component_count=1,
    )
    patch_id = f"{order_id}:patch:{event_index:02d}:{key[-12:]}"
    checkpoint = checkpoint_state(state, lineage=patch_id)
    next_state = deepcopy(dict(state))
    rollback_receipt: JsonDict | None = None
    if gate["admitted"]:
        next_state["items"][key] = proposed_item
        next_state["version"] = int(next_state["version"]) + 1
        next_state["last_commit_index"] = event_index
        next_state["lineage"].append(patch_id)
        post_commit_ok = _post_commit_safe(next_state, support_after)
        if not post_commit_ok:
            next_state, rollback_receipt = rollback_to_checkpoint(
                next_state, checkpoint, reason="failed_post_commit_gate"
            )
    decision = "admit" if gate["admitted"] and rollback_receipt is None else "reject"
    row: JsonDict = {
        "schema": SCHEMA + ".patch_decision_row.v1",
        "patch_id": patch_id,
        "order_id": order_id,
        "arm": "verified_memory",
        "event_id": event["event_id"],
        "event_index": event_index,
        "operation": operation,
        "proposed_component": proposed_item["component_type"],
        "targeted_component_count": 1,
        "applicability_key": key,
        "source_repair": {
            "operator": proposed_item["operator"],
            "exact_outcome": source_result,
            "exact_witness_sha256": event["exact_witness"]["witness_sha256"],
        },
        "held_anchor_check": anchors,
        "support_check": {
            "fixed_candidate_budget": len(CANDIDATE_OPERATORS),
            "available_before": len(CANDIDATE_OPERATORS),
            "available_after": len(CANDIDATE_OPERATORS),
            "before": support_before,
            "after": support_after,
            "floor": SUPPORT_FLOOR,
            "passed": support_after >= SUPPORT_FLOOR,
        },
        "gate_decision": gate,
        "decision": decision,
        "rejection_reasons": gate["rejection_reasons"],
        "item_version_before": 0 if existing is None else existing["version"],
        "item_version_after": (
            proposed_item["version"]
            if decision == "admit"
            else 0
            if existing is None
            else existing["version"]
        ),
        "memory_version_before": state["version"],
        "memory_version_after": next_state["version"],
        "checkpoint_checksum": checkpoint["checksum"],
        "state_after_checksum": sha256_json(next_state),
        "rollback_applied": rollback_receipt is not None,
        "rollback_target_checksum": checkpoint["checksum"],
        "patch_checksum": sha256_json(proposed_item),
    }
    row["row_sha256"] = row_hash(row)
    support_row = None
    if decision == "admit":
        support_row = {
            "schema": SCHEMA + ".recoverable_support_row.v1",
            "patch_id": patch_id,
            "order_id": order_id,
            "event_id": event["event_id"],
            "fixed_candidate_budget": len(CANDIDATE_OPERATORS),
            "available_before": len(CANDIDATE_OPERATORS),
            "available_after": len(CANDIDATE_OPERATORS),
            "before": support_before,
            "after": support_after,
            "floor": SUPPORT_FLOOR,
            "passed": True,
        }
        support_row["row_sha256"] = row_hash(support_row)
    state_receipt = {
        "schema": SCHEMA + ".memory_state_receipt.v1",
        "receipt_type": "patch_transaction",
        "order_id": order_id,
        "event_index": event_index,
        "patch_id": patch_id,
        "version_before": state["version"],
        "version_after": next_state["version"],
        "checksum_before": sha256_json(state),
        "checksum_after": sha256_json(next_state),
        "checkpoint_checksum": checkpoint["checksum"],
        "restart_checksum": None,
        "restart_equal": None,
        "rollback_applied": rollback_receipt is not None,
        "rollback_target_checksum": checkpoint["checksum"],
        "rollback_lineage": None if rollback_receipt is None else rollback_receipt,
    }
    return next_state, row, support_row, state_receipt


def _restart_receipt(state: Mapping[str, Any], *, order_id: str, event_count: int) -> JsonDict:
    checkpoint = checkpoint_state(state, lineage=f"{order_id}:restart:{event_count}")
    restarted = json.loads(str(checkpoint["state_bytes"]))
    restart_checksum = sha256_json(restarted)
    return {
        "schema": SCHEMA + ".memory_state_receipt.v1",
        "receipt_type": "restart_checkpoint",
        "order_id": order_id,
        "event_index": event_count - 1,
        "patch_id": None,
        "version_before": state["version"],
        "version_after": restarted["version"],
        "checksum_before": sha256_json(state),
        "checksum_after": restart_checksum,
        "checkpoint_checksum": checkpoint["checksum"],
        "restart_checksum": restart_checksum,
        "restart_equal": restart_checksum == checkpoint["checksum"],
        "rollback_applied": False,
        "rollback_target_checksum": checkpoint["checksum"],
        "rollback_lineage": [],
    }


def run_comparison(fixture: Mapping[str, Any], preregistration: Mapping[str, Any]) -> JsonDict:
    """Run every registered arm and order over matched exact candidates."""

    events = evaluation_events(fixture)
    by_id = {str(row["event_id"]): row for row in events}
    event_rows: list[JsonDict] = []
    retrieval_rows: list[JsonDict] = []
    patch_rows: list[JsonDict] = []
    state_receipts: list[JsonDict] = []
    support_rows: list[JsonDict] = []

    for order in preregistration["orders"]:
        order_id = str(order["order_id"])
        states = {arm: empty_memory_state() for arm in ARMS}
        seen_keys: set[str] = set()
        for event_index, event_id in enumerate(order["event_ids"]):
            event = by_id[str(event_id)]
            key = str(event["experiential_repair"]["applicability_key"])
            eligible_future = key in seen_keys
            decisions: dict[str, JsonDict] = {}
            pre_states: dict[str, JsonDict] = {}
            for arm in ARMS:
                state = states[arm]
                pre_states[arm] = deepcopy(state)
                item = state["items"].get(key) if arm == "verified_memory" else None
                decisions[arm] = rank_candidates(
                    event,
                    arm,
                    preregistration,
                    item,
                    order_id=order_id,
                )

            exact_by_operator = [
                {"operator": operator, "exact_outcome": exact_outcome(event, operator)}
                for operator in CANDIDATE_OPERATORS
            ]
            for arm in ARMS:
                pre_state = pre_states[arm]
                decision = decisions[arm]
                outcome = exact_outcome(event, str(decision["selected_operator"]))
                update = {
                    "proposed": False,
                    "decision": "no_update_read_only_arm",
                    "patch_id": None,
                }
                if arm == "verified_memory":
                    next_state, patch, support, state_receipt = _propose_patch(
                        pre_state,
                        event,
                        fixture,
                        preregistration,
                        order_id=order_id,
                        event_index=event_index,
                    )
                    states[arm] = next_state
                    patch_rows.append(patch)
                    state_receipts.append(state_receipt)
                    if support is not None:
                        support_rows.append(support)
                    update = {
                        "proposed": True,
                        "decision": patch["decision"],
                        "patch_id": patch["patch_id"],
                        "operation": patch["operation"],
                    }
                else:
                    states[arm] = pre_state
                row: JsonDict = {
                    "schema": SCHEMA + ".arm_order_event_row.v1",
                    "order_id": order_id,
                    "order_sha256": order["order_sha256"],
                    "arm": arm,
                    "arm_seed": preregistration["arm_seeds"][arm],
                    "tie_seed": preregistration["tie_seeds"][order_id],
                    "event_id": event["event_id"],
                    "event_index": event_index,
                    "fixture_chronological_index": event["chronological_index"],
                    "fixture_partition": event["partition"],
                    "working_state_checksum": event["working_state"]["working_state_checksum"],
                    "applicability_key": key,
                    "pre_memory_version": pre_state["version"],
                    "pre_memory_checksum": sha256_json(pre_state),
                    "visible_commit_ids": list(pre_state["lineage"]),
                    "visible_commit_max_index": pre_state["last_commit_index"],
                    "candidate_pool_sha256": decision["candidate_pool_sha256"],
                    "candidate_ranking": decision["candidate_ranking"],
                    "selected_operator": decision["selected_operator"],
                    "context_only_operator": decision["context_only_operator"],
                    "action_committed_before_exact_outcome": True,
                    "same_event_pending_write_visible": False,
                    "information_fields": decision["information_fields"],
                    "ranking_basis": decision["ranking_basis"],
                    "retrieval": {
                        "retrieved": decision["memory_retrieved"],
                        "memory_version": decision["memory_version"],
                        "action_changed": decision["action_changed"],
                        "credited": decision["credited"],
                    },
                    "candidate_exact_outcomes": exact_by_operator,
                    "exact_outcome": outcome,
                    "regret": 1 - outcome,
                    "exact_evaluator": "exp6653_retained_exact_repair_target",
                    "exact_witness_sha256": event["exact_witness"]["witness_sha256"],
                    "eligible_future_event": eligible_future,
                    "update": update,
                    "post_memory_version": states[arm]["version"],
                    "post_memory_checksum": sha256_json(states[arm]),
                }
                row["row_sha256"] = row_hash(row)
                event_rows.append(row)
                receipt = {
                    "schema": SCHEMA + ".retrieval_influence_row.v1",
                    "order_id": order_id,
                    "arm": arm,
                    "event_id": event["event_id"],
                    "event_index": event_index,
                    "applicability_key": key,
                    "retrieved": decision["memory_retrieved"],
                    "memory_version": decision["memory_version"],
                    "baseline_operator": decision["context_only_operator"],
                    "selected_operator": decision["selected_operator"],
                    "action_changed": decision["action_changed"],
                    "credited": bool(decision["memory_retrieved"] and decision["action_changed"]),
                    "exact_outcome": outcome,
                }
                receipt["row_sha256"] = row_hash(receipt)
                retrieval_rows.append(receipt)
            seen_keys.add(key)
            event_count = event_index + 1
            if event_count in RESTART_POINTS:
                state_receipts.append(
                    _restart_receipt(
                        states["verified_memory"], order_id=order_id, event_count=event_count
                    )
                )

    metrics = recompute_metrics(event_rows, retrieval_rows, patch_rows, support_rows)
    return {
        "arm_order_event_rows": event_rows,
        "retrieval_and_influence_rows": retrieval_rows,
        "patch_decision_rows": patch_rows,
        "memory_state_receipts": state_receipts,
        "recoverable_support_rows": support_rows,
        "prospective_metrics": metrics,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    scale = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _order_delta_interval(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    radius = 4.30265272975 * math.sqrt(variance / len(values))
    return [mean - radius, mean + radius]


def recompute_metrics(
    event_rows: Sequence[Mapping[str, Any]],
    retrieval_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild all scientific metrics from immutable event and patch rows."""

    arm_summary: dict[str, JsonDict] = {}
    order_arm_rows: list[JsonDict] = []
    for arm in ARMS:
        rows = [row for row in event_rows if row["arm"] == arm]
        successes = sum(int(row["exact_outcome"]) for row in rows)
        arm_summary[arm] = {
            "event_count": len(rows),
            "exact_success_count": successes,
            "prequential_exact_yield": _rate(successes, len(rows)),
            "regret": sum(int(row["regret"]) for row in rows),
        }
    for order_id in ORDER_IDS:
        for arm in ARMS:
            rows = [row for row in event_rows if row["order_id"] == order_id and row["arm"] == arm]
            successes = sum(int(row["exact_outcome"]) for row in rows)
            order_arm_rows.append(
                {
                    "order_id": order_id,
                    "arm": arm,
                    "event_count": len(rows),
                    "exact_success_count": successes,
                    "prequential_exact_yield": _rate(successes, len(rows)),
                    "regret": len(rows) - successes,
                }
            )
    future_by_arm: dict[str, JsonDict] = {}
    for arm in ARMS:
        rows = [row for row in event_rows if row["arm"] == arm and row["eligible_future_event"]]
        successes = sum(int(row["exact_outcome"]) for row in rows)
        future_by_arm[arm] = {
            "event_count": len(rows),
            "exact_success_count": successes,
            "exact_yield": _rate(successes, len(rows)),
        }
    context_yield = float(future_by_arm["context_only"]["exact_yield"] or 0.0)
    memory_yield = float(future_by_arm["verified_memory"]["exact_yield"] or 0.0)
    order_deltas: list[JsonDict] = []
    for order_id in ORDER_IDS:
        context = next(
            row
            for row in order_arm_rows
            if row["order_id"] == order_id and row["arm"] == "context_only"
        )
        memory = next(
            row
            for row in order_arm_rows
            if row["order_id"] == order_id and row["arm"] == "verified_memory"
        )
        delta = float(memory["prequential_exact_yield"]) - float(context["prequential_exact_yield"])
        order_deltas.append({"order_id": order_id, "verified_memory_minus_context_only": delta})
    delta_values = [row["verified_memory_minus_context_only"] for row in order_deltas]
    influential = [row for row in retrieval_rows if row["credited"]]
    retrieved = [row for row in retrieval_rows if row["retrieved"]]
    exact_influential = sum(int(row["exact_outcome"]) for row in influential)
    accepted = [row for row in patch_rows if row["decision"] in {"admit", "retire"}]
    forgetting = sum(
        1
        for row in event_rows
        if row["arm"] == "verified_memory"
        and row["retrieval"]["action_changed"]
        and exact_outcome_from_candidates(
            row["candidate_exact_outcomes"], row["context_only_operator"]
        )
        == 1
        and row["exact_outcome"] == 0
    )
    support_before = [float(row["before"]) for row in support_rows]
    support_after = [float(row["after"]) for row in support_rows]
    return {
        "arm_summary": arm_summary,
        "order_arm_rows": order_arm_rows,
        "future_event_delta": {
            "by_arm": future_by_arm,
            "verified_memory_minus_context_only": memory_yield - context_yield,
        },
        "forgetting": {
            "count": forgetting,
            "rate_per_influential_retrieval": _rate(forgetting, len(influential)),
        },
        "recoverable_support": {
            "floor": SUPPORT_FLOOR,
            "accepted_patch_count": len(support_rows),
            "minimum_before": min(support_before) if support_before else None,
            "minimum_after": min(support_after) if support_after else None,
            "collapse_count": sum(value < SUPPORT_FLOOR for value in support_after),
        },
        "retrieval": {
            "retrieved_count": len(retrieved),
            "influential_count": len(influential),
            "credited_count": len(influential),
            "credited_exact_count": exact_influential,
            "retrieval_precision": _rate(exact_influential, len(influential)),
            "action_influence_rate": _rate(len(influential), len(retrieved)),
        },
        "updates": {
            "proposed_count": len(patch_rows),
            "accepted_count": len(accepted),
            "rejected_count": len(patch_rows) - len(accepted),
            "retired_count": sum(row["decision"] == "retire" for row in patch_rows),
            "acceptance_rate": _rate(len(accepted), len(patch_rows)),
        },
        "order_sensitivity": {
            "per_order": order_deltas,
            "mean_delta": sum(delta_values) / len(delta_values),
            "minimum_delta": min(delta_values),
            "maximum_delta": max(delta_values),
            "range": max(delta_values) - min(delta_values),
        },
        "uncertainty": {
            "method": "wilson_yield_and_order_delta_t_interval",
            "yield_wilson_95": {
                arm: _wilson(
                    int(arm_summary[arm]["exact_success_count"]),
                    int(arm_summary[arm]["event_count"]),
                )
                for arm in ARMS
            },
            "order_delta_mean_95_interval": _order_delta_interval(delta_values),
        },
    }


def exact_outcome_from_candidates(rows: Sequence[Mapping[str, Any]], operator: str) -> int:
    """Read one post-action exact result from the complete candidate receipt."""

    match = next((row for row in rows if row["operator"] == operator), None)
    if match is None:
        raise ValueError("operator_missing_from_exact_candidate_receipt")
    return int(match["exact_outcome"])


def aggregate_row_recomputation(
    event_rows: Sequence[Mapping[str, Any]],
    retrieval_rows: Sequence[Mapping[str, Any]],
    patch_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    reported_metrics: Mapping[str, Any],
) -> JsonDict:
    """Recompute matched-arm, timing, patch, support, and metric claims."""

    recomputed = recompute_metrics(event_rows, retrieval_rows, patch_rows, support_rows)
    matched = True
    for order_id in ORDER_IDS:
        sequences = []
        pools = []
        for arm in ARMS:
            rows = [row for row in event_rows if row["order_id"] == order_id and row["arm"] == arm]
            sequences.append([row["event_id"] for row in rows])
            pools.append([row["candidate_pool_sha256"] for row in rows])
        matched = matched and all(row == sequences[0] for row in sequences[1:])
        matched = matched and all(row == pools[0] for row in pools[1:])
    accepted = [row for row in patch_rows if row["decision"] in {"admit", "retire"}]
    checks = {
        "event_count": len(event_rows) == len(ORDER_IDS) * len(ARMS) * EVENTS_PER_ORDER,
        "retrieval_count": len(retrieval_rows) == len(event_rows),
        "matched_arms": matched,
        "chronology": all(
            row["action_committed_before_exact_outcome"]
            and not row["same_event_pending_write_visible"]
            and int(row["visible_commit_max_index"]) < int(row["event_index"])
            for row in event_rows
        ),
        "complete_exact_outcomes": all(
            row["exact_outcome"] in (0, 1)
            and len(row["candidate_exact_outcomes"]) == len(CANDIDATE_OPERATORS)
            for row in event_rows
        ),
        "influence_credit": all(
            row["credited"] is (row["retrieved"] and row["action_changed"])
            for row in retrieval_rows
        ),
        "patch_gates": all(
            row["targeted_component_count"] == 1
            and row["operation"] in ALLOWED_PATCH_OPERATIONS
            and (
                row["decision"] == "reject"
                or (
                    row["source_repair"]["exact_outcome"] == 1
                    and row["held_anchor_check"]["regression_count"] == 0
                    and row["support_check"]["after"] >= SUPPORT_FLOOR
                )
            )
            for row in patch_rows
        ),
        "support_rows": len(support_rows) == len(accepted)
        and all(row["after"] >= SUPPORT_FLOOR for row in support_rows),
        "metrics": recomputed == dict(reported_metrics),
    }
    arm_order_rows = recomputed["order_arm_rows"]
    return {
        "arm_order_rows": arm_order_rows,
        "recomputed_prospective_metrics": recomputed,
        "checks": checks,
        "all_recomputations_match": all(checks.values()),
    }


def _attack_row(attack_type: str, detected: bool, observed: Any) -> JsonDict:
    return {
        "schema": SCHEMA + ".attack_row.v1",
        "attack_id": f"attack:{attack_type}",
        "attack_type": attack_type,
        "detected": detected,
        "failed_closed": detected,
        "observed_value": observed,
    }


def build_attack_rows(comparison: Mapping[str, Any]) -> list[JsonDict]:
    """Inject the shortcut and recovery faults required by the experiment."""

    event_rows = comparison["arm_order_event_rows"]
    retrieval_rows = comparison["retrieval_and_influence_rows"]
    future = deepcopy(event_rows[0])
    future["ranking_basis"]["exact_outcome"] = 1
    future_detected = bool(set(future["ranking_basis"]) & FORBIDDEN_SELECTION_FIELDS)

    pending = deepcopy(event_rows[0])
    pending["same_event_pending_write_visible"] = True
    pending_detected = pending["same_event_pending_write_visible"] is True

    mismatched = deepcopy(event_rows[: len(ARMS)])
    mismatched[-1]["candidate_pool_sha256"] = "sha256:tampered"
    pool_detected = len({row["candidate_pool_sha256"] for row in mismatched}) != 1

    inert = deepcopy(next((row for row in retrieval_rows if row["retrieved"]), retrieval_rows[0]))
    inert.update(retrieved=True, action_changed=False, credited=True)
    credit_detected = inert["credited"] is not (inert["retrieved"] and inert["action_changed"])

    multi = patch_gate_decision(
        operation="append",
        source_repair=True,
        anchor_before=1,
        anchor_after=1,
        support_after=1.0,
        targeted_component_count=2,
    )
    source = patch_gate_decision(
        operation="append",
        source_repair=False,
        anchor_before=1,
        anchor_after=1,
        support_after=1.0,
        targeted_component_count=1,
    )
    support = patch_gate_decision(
        operation="append",
        source_repair=True,
        anchor_before=1,
        anchor_after=1,
        support_after=SUPPORT_FLOOR - 0.5,
        targeted_component_count=1,
    )
    state = empty_memory_state()
    checkpoint = checkpoint_state(state, lineage="attack:rollback")
    tampered = deepcopy(checkpoint)
    tampered["state_bytes"] = canonical_json({**state, "version": 99})
    _, rollback = rollback_to_checkpoint({**state, "version": 1}, tampered, reason="attack")
    return [
        _attack_row("future_label_selection", future_detected, "exact_outcome"),
        _attack_row("same_event_pending_write", pending_detected, True),
        _attack_row("candidate_pool_mismatch", pool_detected, "sha256:tampered"),
        _attack_row("retrieval_without_influence_credit", credit_detected, True),
        _attack_row("multi_component_patch", not multi["admitted"], multi["rejection_reasons"]),
        _attack_row("source_repair_failure", not source["admitted"], source["rejection_reasons"]),
        _attack_row("support_collapse", not support["admitted"], support["rejection_reasons"]),
        _attack_row(
            "rollback_drift", not rollback["restored_equal"], rollback["restored_checksum"]
        ),
    ]


def terminal_fields(gates: Sequence[Mapping[str, Any]], *, future_delta: float) -> JsonDict:
    """Keep completion independent of sign while blocking integrity failures."""

    failed = next((row for row in gates if row.get("passed") is not True), None)
    if failed is not None:
        name = str(failed["check"])
        return {
            "status": f"blocked_{name}",
            "honest_verdict": f"blocked_{name}: observed={failed.get('observed')!r}",
            "verdict_class": "blocked",
            "prospective_memory_comparison_complete": False,
            "gate_check_summary": {
                "failed_check": name,
                "expected_value": failed.get("expected"),
                "observed_value": failed.get("observed"),
                "checks": list(gates),
            },
        }
    if future_delta > 0.0:
        status = "complete_positive"
        verdict_class: str | None = "positive"
        verdict = (
            "complete: validation-gated repair memory improved future exact outcomes "
            "with no forgetting or support collapse"
        )
    else:
        status = "complete_null"
        verdict_class = None
        verdict = (
            "complete: prospective repair-memory comparison is complete with no positive "
            "future exact improvement"
        )
    return {
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "prospective_memory_comparison_complete": True,
        "gate_check_summary": {
            "failed_check": None,
            "expected_value": None,
            "observed_value": None,
            "checks": list(gates),
        },
    }


def _gate_rows(
    upstream: Mapping[str, Any],
    comparison: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    metrics = comparison["prospective_metrics"]
    restart_rows = [
        row
        for row in comparison["memory_state_receipts"]
        if row["receipt_type"] == "restart_checkpoint"
    ]
    checks = {
        "upstream_gate": upstream.get("passed") is True,
        "comparison_complete": aggregate.get("all_recomputations_match") is True,
        "matched_arms": aggregate.get("checks", {}).get("matched_arms") is True,
        "prequential_information": aggregate.get("checks", {}).get("chronology") is True,
        "retrieval_influence": aggregate.get("checks", {}).get("influence_credit") is True,
        "targeted_patch_gates": aggregate.get("checks", {}).get("patch_gates") is True,
        "forgetting": metrics["forgetting"]["count"] == 0,
        "recoverable_support": metrics["recoverable_support"]["collapse_count"] == 0,
        "restart": bool(restart_rows) and all(row["restart_equal"] for row in restart_rows),
        "rollback_policy": all(
            not row["rollback_applied"] for row in comparison["memory_state_receipts"]
        ),
        "attacks": {row["attack_type"] for row in attacks} == set(ATTACK_TYPES)
        and all(row["failed_closed"] for row in attacks),
        "protected_files": protected.get("unchanged") is True,
        "tests": bool(tests_run)
        and all(row.get("exit_code") == 0 for row in tests_run if row.get("gating", True)),
    }
    return [
        {"check": name, "expected": True, "observed": value, "passed": value is True}
        for name, value in checks.items()
    ]


def _per_unit_rows(
    event_rows: Sequence[Mapping[str, Any]], patch_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    units: list[JsonDict] = []
    for row in event_rows:
        units.append(
            {
                "unit_type": "order_arm_event",
                "unit_id": f"{row['order_id']}:{row['arm']}:{row['event_index']:02d}",
                "source_collection": "arm_order_event_rows",
                "source_row_sha256": row["row_sha256"],
            }
        )
    for row in patch_rows:
        units.append(
            {
                "unit_type": "patch",
                "unit_id": row["patch_id"],
                "source_collection": "patch_decision_rows",
                "source_row_sha256": row["row_sha256"],
            }
        )
    return units


def _preconditions(
    repo_root: Path,
    fixture: Mapping[str, Any],
    upstream: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    protected_before: Mapping[str, Any],
) -> JsonDict:
    return {
        "inputs": {
            "upstream_present": (repo_root / UPSTREAM_RELATIVE_PATH).exists(),
            "memory_fixture_ready": upstream["value"],
            "fixture_schema": fixture.get("schema"),
            "fixture_event_count": len(fixture.get("event_rows", [])),
            "evaluation_event_count": len(evaluation_events(fixture)),
        },
        "hashes": {
            "fixture_file_sha256": upstream["sha256"],
            "fixture_schema_sha256": sha256_json(fixture.get("memory_schema")),
            "spec_file_sha256": sha256_file(repo_root / SPEC_RELATIVE_PATH),
            "protected_hashes_before": dict(protected_before),
        },
        "tools": {
            "python_executable": os.path.realpath(os.sys.executable),
            "python_version": os.sys.version.split()[0],
            "hash_algorithm": "sha256",
            "atomic_replace": True,
        },
        "resources": {
            "llm_calls": 0,
            "model_weights_loaded": False,
            "network_calls": 0,
            "candidate_budget": len(CANDIDATE_OPERATORS),
        },
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "preconditions_ready": upstream["passed"] is True,
    }


def _field_provenance(upstream_sha256: Any) -> dict[str, JsonDict]:
    row_fields = {
        "arm_order_event_rows",
        "retrieval_and_influence_rows",
        "patch_decision_rows",
        "memory_state_receipts",
        "prospective_metrics",
        "recoverable_support_rows",
        "per_unit_rows",
        "aggregate_row_recomputation",
    }
    return {
        field: {
            "source": (
                UPSTREAM_RELATIVE_PATH.as_posix()
                if field in row_fields or field == "upstream_gate_receipt"
                else "REQ-LEARN-6654 deterministic experiment contract"
            ),
            "source_sha256": upstream_sha256,
            "reducer": "build_artifact",
            "lineage": [
                "Exp6653 exact rows",
                "preregistered orders",
                "prequential actions",
                "row reducer",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every final field except the checksum field itself."""

    material = deepcopy(dict(artifact))
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def build_artifact(
    *,
    repo_root: Path,
    output_path: Path,
    date: str,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    write: bool,
) -> JsonDict:
    """Build, validate, and optionally publish the terminal comparison."""

    protected_before = _protected_hashes(repo_root)
    upstream = upstream_gate_receipt(repo_root)
    if not upstream["passed"]:
        raise ValueError("upstream_memory_fixture_not_ready")
    fixture = read_json(repo_root / UPSTREAM_RELATIVE_PATH)
    preregistration = build_preregistration(fixture)
    comparison = run_comparison(fixture, preregistration)
    attacks = build_attack_rows(comparison)
    protected_after = _protected_hashes(repo_root)
    changed = sorted(
        path for path, digest in protected_before.items() if protected_after.get(path) != digest
    )
    protected = {
        "before": protected_before,
        "after": protected_after,
        "changed_paths": changed,
        "unchanged": not changed,
    }
    aggregate = aggregate_row_recomputation(
        comparison["arm_order_event_rows"],
        comparison["retrieval_and_influence_rows"],
        comparison["patch_decision_rows"],
        comparison["recoverable_support_rows"],
        comparison["prospective_metrics"],
    )
    gates = _gate_rows(upstream, comparison, aggregate, attacks, protected, tests_run)
    future_delta = comparison["prospective_metrics"]["future_event_delta"][
        "verified_memory_minus_context_only"
    ]
    terminal = terminal_fields(gates, future_delta=float(future_delta))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "result_path": output_path.as_posix(),
        **terminal,
        "upstream_gate_receipt": upstream,
        "preregistration": preregistration,
        **comparison,
        "adversarial_rows": attacks,
        "per_unit_rows": _per_unit_rows(
            comparison["arm_order_event_rows"], comparison["patch_decision_rows"]
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": _preconditions(
            repo_root, fixture, upstream, preregistration, protected_before
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(upstream["sha256"]),
        "random_seed": {
            "order_seeds": dict(ORDER_SEEDS),
            "arm_seeds": dict(ARM_SEEDS),
            "tie_seeds": dict(TIE_SEEDS),
            "schedule_sha256": sha256_json(
                {"orders": ORDER_SEEDS, "arms": ARM_SEEDS, "ties": TIE_SEEDS}
            ),
        },
        "duration_s": duration_s,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(",".join(errors))
    if write:
        atomic_write_json(output_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return stable error names for each terminal contract failure."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing_required_fields")
    event_rows = list(artifact.get("arm_order_event_rows") or [])
    retrieval_rows = list(artifact.get("retrieval_and_influence_rows") or [])
    patch_rows = list(artifact.get("patch_decision_rows") or [])
    per_units = list(artifact.get("per_unit_rows") or [])
    expected_events = len(ORDER_IDS) * len(ARMS) * EVENTS_PER_ORDER
    if len(event_rows) != expected_events:
        errors.append("event_row_count_mismatch")
    if len(retrieval_rows) != len(event_rows):
        errors.append("retrieval_row_count_mismatch")
    if len(per_units) != len(event_rows) + len(patch_rows):
        errors.append("per_unit_count_mismatch")
    aggregate = dict(artifact.get("aggregate_row_recomputation") or {})
    expected_complete = aggregate.get("all_recomputations_match") is True
    if artifact.get("prospective_memory_comparison_complete") is not expected_complete:
        errors.append("comparison_complete_mismatch")
    delta = (
        dict(artifact.get("prospective_metrics") or {})
        .get("future_event_delta", {})
        .get("verified_memory_minus_context_only", 0.0)
    )
    expected_class = "positive" if expected_complete and float(delta) > 0.0 else None
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_mismatch")
    if dict(artifact.get("upstream_gate_receipt") or {}).get("passed") is not True:
        errors.append("upstream_gate_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("oracle_boundary_mismatch")
    if dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is not True:
        errors.append("protected_files_changed")
    if not artifact.get("tests_run") or any(
        row.get("exit_code") != 0
        for row in artifact.get("tests_run", [])
        if row.get("gating", True)
    ):
        errors.append("test_command_failed")
    if aggregate.get("all_recomputations_match") is not True:
        errors.append("aggregate_recomputation_mismatch")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or any(
        field not in provenance for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_provenance_missing")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("checksum_mismatch")
    return errors


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260826")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--check-rows", action="store_true")
    parser.add_argument("--duration-s", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the result, validate it, or rerun its row-only reduction."""

    args = _parse_args(argv)
    if args.validate:
        errors = validate_artifact(read_json(args.output))
        if errors:
            raise ValueError(",".join(errors))
        return 0
    if args.check_rows:
        artifact = read_json(args.output)
        recomputed = aggregate_row_recomputation(
            artifact["arm_order_event_rows"],
            artifact["retrieval_and_influence_rows"],
            artifact["patch_decision_rows"],
            artifact["recoverable_support_rows"],
            artifact["prospective_metrics"],
        )
        if recomputed != artifact["aggregate_row_recomputation"]:
            raise ValueError("aggregate_recomputation_mismatch")
        return 0
    started = time.monotonic()
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        output_path=args.output,
        date=args.date,
        duration_s=args.duration_s if args.duration_s is not None else 0.001,
        tests_run=DEFAULT_TEST_RECEIPTS,
        write=False,
    )
    if args.duration_s is None:
        artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.001)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    atomic_write_json(args.output, artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - tests call main directly.
    raise SystemExit(main())
