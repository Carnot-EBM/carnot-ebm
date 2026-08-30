"""Build a frozen CPU stream for bounded constraint-route decisions.

Spec refs: REQ-CL-6790 and SCENARIO-CL-6790-*.

The stream shows a controller only the current candidate graph and safe history.
Exact results stay in a separate receipt until after each route is selected.
This fixture measures opportunity and headroom. It does not claim online gain.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6790_chronological_constraint_routing_stream"
SCHEMA = "carnot.experiment_6790.chronological_constraint_routing_stream.v1"
RUN_DATE = "20260830"
RANDOM_SEED = 6_790_000
INFERENCE_SUBSTRATE = "CPU exact chronological decision fixture, no LLM"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6790_chronological_constraint_routing_stream.py"
)
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6790_chronological_constraint_routing_stream.py"
)
RESULT_RELATIVE_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
SOURCE_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6786_constraint_dependency_hard_negative_fixture.json"
)
SOURCE_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6786_constraint_dependency_hard_negative_fixture.py"
)
EXPECTED_SOURCE_ARTIFACT_SHA256 = (
    "sha256:f3780c85e29cda8dbd897b6c43a0ce3c938252625e823e54107f918d2052514a"
)
EXPECTED_SOURCE_MODULE_SHA256 = (
    "sha256:62985bac04b0af4527e2117c89fce023e5ff19f6aa77747ef19f998da0d53ef8"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "sha256:99c6ea4200db1d22092951d9e46d1f8d1b598e3a8815d74449c828d427bbd7b9"
)
EXPECTED_NESTED_SOURCE_HASHES: JsonDict = {
    "python/carnot/experiment_5299_constraint_lns_solver_repair_fixture_v484.py": (
        "sha256:e171ae2fa21674871e765e7b84f2fe72464daf8348b8246f700f1b1f7d4ad737"
    ),
    "python/carnot/experiment_5304_kan_dynamic_abstraction_spotcheck_v484.py": (
        "sha256:ea6b9cd7d4d7aecb88d15762ba5612a6172585479ce584855f40163da0d7d01c"
    ),
    "results/experiment_6768_targetable_proof_panel_expansion.json": (
        "sha256:5e44fd4db55e11c99f0f5720aa5815bf26135f5b0801e3687ebf1095a1ee180d"
    ),
}

EVENT_COUNT = 240
ORDER_COUNT = 5
NON_HELD_EVENT_COUNT = 160
LIVE_ROUTE_BUDGET = 3
EXHAUSTIVE_ROUTE_ID = "exhaustive_oracle_diagnostic"
HELD_FUTURE_FAMILY = "directed_implication_cycle"
TOPOLOGY_FAMILIES = (
    "directed_implication_chain",
    "directed_implication_star",
    HELD_FUTURE_FAMILY,
)
DIFFICULTIES = ("easy", "medium", "hard", "challenge")
DIFFICULTY_RANK = {name: rank for rank, name in enumerate(DIFFICULTIES)}
ROUTE_DEFINITIONS: JsonDict = {
    "local_prefix": {
        "description": "Inspect the first three local constraint groups.",
        "selector": "first_three_local_groups",
        "check_budget": LIVE_ROUTE_BUDGET,
        "cost_per_check": 1,
        "live_action": True,
    },
    "local_suffix": {
        "description": "Inspect the last three local constraint groups.",
        "selector": "last_three_local_groups",
        "check_budget": LIVE_ROUTE_BUDGET,
        "cost_per_check": 1,
        "live_action": True,
    },
    "dependency_prefix": {
        "description": "Inspect the first three cross-group dependencies.",
        "selector": "first_three_dependencies",
        "check_budget": LIVE_ROUTE_BUDGET,
        "cost_per_check": 1,
        "live_action": True,
    },
    "dependency_suffix": {
        "description": "Inspect the last three cross-group dependencies.",
        "selector": "last_three_dependencies",
        "check_budget": LIVE_ROUTE_BUDGET,
        "cost_per_check": 1,
        "live_action": True,
    },
    "mixed_boundary": {
        "description": "Inspect the first local group and both dependency boundaries.",
        "selector": "first_local_first_dependency_last_dependency",
        "check_budget": LIVE_ROUTE_BUDGET,
        "cost_per_check": 1,
        "live_action": True,
    },
    EXHAUSTIVE_ROUTE_ID: {
        "description": "Inspect every factor after action for diagnostic headroom only.",
        "selector": "all_factors",
        "check_budget": None,
        "cost_per_check": 1,
        "live_action": False,
    },
}
LIVE_ROUTE_IDS = tuple(
    route_id
    for route_id, definition in ROUTE_DEFINITIONS.items()
    if definition["live_action"] is True
)
FEATURE_ALLOWLIST = (
    "schema",
    "candidate_graph",
    "reusable_motif_id",
    "topology_family",
    "difficulty",
    "provenance_class",
    "retention_rule",
    "candidate_assignment_summary",
)
FEATURE_DENYLIST = (
    "exact_label",
    "exact_valid",
    "exact_receipt",
    "revealed_post_action_receipt",
    "hidden_receipt_hash",
    "future_outcome",
    "future_receipts",
    "order_id",
    "order_index",
    "chronology_position",
    "planted_failure_factor_ids",
    "failed_dependency_ids",
    "poison_status",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
STANDARD_ARTIFACT_FIELDS = ("schema", "experiment_id", "run_date", "status")
REQUIRED_ARTIFACT_FIELDS = STANDARD_ARTIFACT_FIELDS + (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hash",
    "frozen_manifest",
    "order_definitions",
    "feature_allowlist",
    "feature_denylist",
    "route_definitions",
    "rows",
    "event_count_by_order",
    "topology_count_by_order",
    "held_future_counts",
    "poison_counts",
    "reusable_motif_counts",
    "future_feature_violations",
    "frozen_policy_metrics_by_order",
    "random_route_metrics_by_order",
    "diagnostic_headroom_by_order",
    "cold_replay_hashes",
    "constraint_routing_stream_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned schema makes incompatible stream readers fail closed.",
    "experiment_id": "A stable ID binds the artifact to its owned producer.",
    "run_date": "The fixed execution date prevents silent protocol drift.",
    "status": "A terminal status separates readiness from a complete block.",
    "field_principles": "Each required field states the evidence boundary it protects.",
    "inference_substrate": "The CPU declaration prevents fixture evidence from becoming an LLM claim.",
    "duration_s": "Measured wall time shows that row generation and replay executed.",
    "random_seed": "One seed fixes order shuffles and random-route actions.",
    "reproducibility_checksum": "A stable hash detects stream or metric drift.",
    "source_artifact_hash": "The exact Exp6786 bytes bind every derived event.",
    "frozen_manifest": "The manifest freezes events and hidden receipts before replay.",
    "order_definitions": "Five hashes freeze chronology before any action receipt opens.",
    "feature_allowlist": "Only these current fields may select a live route.",
    "feature_denylist": "Exact, future, and order fields stay outside action inputs.",
    "route_definitions": "Named costs prove live checking is bounded below exhaustive work.",
    "rows": "One event-order row preserves pre-action and post-action separation.",
    "event_count_by_order": "Counts prove every order contains the complete event set.",
    "topology_count_by_order": "Counts expose topology coverage in every replicate.",
    "held_future_counts": "Counts prove each order reaches novel held topology.",
    "poison_counts": "Counts prove unsafe reward conflicts are present in every order.",
    "reusable_motif_counts": "Counts prove prior visible motifs can inform later actions.",
    "future_feature_violations": "An empty list proves action observations passed the denylist audit.",
    "frozen_policy_metrics_by_order": "Frozen row outcomes establish a nonzero live floor.",
    "random_route_metrics_by_order": "Seeded random actions provide a weaker route baseline.",
    "diagnostic_headroom_by_order": "Exhaustive-only outcomes measure unused opportunity.",
    "cold_replay_hashes": "A fresh process must reproduce each order and aggregate row hash.",
    "constraint_routing_stream_ready": "All opportunity, isolation, replay, and headroom gates must pass.",
    "gate_check_summary": "Every gate keeps its expected and observed value.",
    "verifier_is_oracle": "False separates the route policy from exact labeling authority.",
    "verdict_class": "A closed class keeps readiness distinct from a learning-gain claim.",
    "honest_verdict": "A terminal prefix reports completion and the no-gain boundary.",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Use one JSON byte form so hashes do not depend on output formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Hash canonical JSON and name the digest algorithm in stored receipts."""

    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def sha256_file(path: Path) -> str | None:
    """Hash one required file, or return no value when the file is absent."""

    if not path.is_file():
        return None
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON object and reject roots that cannot hold named receipts."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record one gate without discarding the value that caused a block."""

    return {
        "check": check,
        "expected": expected,
        "observed": deepcopy(observed),
        "passed": observed == expected if passed is None else passed,
    }


def _summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name all failed gates and retain the first failure for terminal prose."""

    copied = [deepcopy(dict(check)) for check in checks]
    failures = [str(check["check"]) for check in copied if check.get("passed") is not True]
    return {
        "all_passed": not failures,
        "checks": copied,
        "failed_checks": failures,
        "first_failure": next((check for check in copied if check.get("passed") is not True), None),
    }


def first_failed_check(summary: Mapping[str, Any]) -> JsonDict:
    """Return the first failed gate or an explicit all-passed receipt."""

    first = summary.get("first_failure")
    if isinstance(first, Mapping):
        return deepcopy(dict(first))
    return _gate("all_preconditions", True, True)


def _bounded_route_model(source: Mapping[str, Any]) -> bool:
    """Prove each live action has one equal budget below every source graph."""

    units = source.get("frozen_manifest", {}).get("units", [])
    factor_counts = [
        len(unit.get("graph", {}).get("local_groups", []))
        + len(unit.get("graph", {}).get("dependency_edges", []))
        for unit in units
        if isinstance(unit, Mapping)
    ]
    live = [definition for definition in ROUTE_DEFINITIONS.values() if definition["live_action"]]
    return bool(
        len(live) >= 2
        and factor_counts
        and all(definition["check_budget"] == LIVE_ROUTE_BUDGET for definition in live)
        and LIVE_ROUTE_BUDGET < min(factor_counts)
        and ROUTE_DEFINITIONS[EXHAUSTIVE_ROUTE_ID]["live_action"] is False
    )


def evaluate_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    source_artifact_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Require the exact Exp6786 bytes, failure classes, topology, and route cost."""

    source_path = source_artifact_path or repo_root / SOURCE_ARTIFACT_RELATIVE_PATH
    exists = source_path.is_file()
    source: JsonDict = {}
    parsed = False
    if exists:
        try:
            source = load_json_object(source_path)
            parsed = True
        except (OSError, ValueError, json.JSONDecodeError):
            parsed = False
    observed: JsonDict = {
        "exp6786_artifact_exists": exists,
        "exp6786_artifact_parses": parsed,
        "constraint_group_fixture_ready": source.get("constraint_group_fixture_ready") is True,
        "source_artifact_sha256": sha256_file(source_path),
        "source_module_sha256": sha256_file(repo_root / SOURCE_MODULE_RELATIVE_PATH),
        "source_manifest_sha256": source.get("frozen_manifest", {}).get("manifest_sha256"),
        "nested_source_hashes": source.get("source_artifact_hashes"),
        "minimum_topology_families": len(
            {
                row.get("family_id")
                for row in source.get("topology_families", [])
                if isinstance(row, Mapping)
            }
        ),
        "local_failure_rows": sum(source.get("easy_negative_counts", {}).values()),
        "cross_dependency_failure_rows": sum(
            source.get("local_pass_cross_dependency_fail_counts", {}).values()
        ),
        "bounded_route_cost_model": _bounded_route_model(source),
    }
    observed.update(dict(overrides or {}))
    checks = [
        _gate("exp6786_artifact_exists", True, observed["exp6786_artifact_exists"]),
        _gate("exp6786_artifact_parses", True, observed["exp6786_artifact_parses"]),
        _gate(
            "constraint_group_fixture_ready",
            True,
            observed["constraint_group_fixture_ready"],
        ),
        _gate(
            "source_artifact_sha256",
            EXPECTED_SOURCE_ARTIFACT_SHA256,
            observed["source_artifact_sha256"],
        ),
        _gate(
            "source_module_sha256",
            EXPECTED_SOURCE_MODULE_SHA256,
            observed["source_module_sha256"],
        ),
        _gate(
            "source_manifest_sha256",
            EXPECTED_SOURCE_MANIFEST_SHA256,
            observed["source_manifest_sha256"],
        ),
        _gate(
            "nested_source_hashes",
            EXPECTED_NESTED_SOURCE_HASHES,
            observed["nested_source_hashes"],
        ),
        _gate(
            "minimum_topology_families",
            ">=3",
            observed["minimum_topology_families"],
            isinstance(observed["minimum_topology_families"], int)
            and observed["minimum_topology_families"] >= 3,
        ),
        _gate(
            "local_failure_rows",
            ">0",
            observed["local_failure_rows"],
            isinstance(observed["local_failure_rows"], int) and observed["local_failure_rows"] > 0,
        ),
        _gate(
            "cross_dependency_failure_rows",
            ">0",
            observed["cross_dependency_failure_rows"],
            isinstance(observed["cross_dependency_failure_rows"], int)
            and observed["cross_dependency_failure_rows"] > 0,
        ),
        _gate(
            "bounded_route_cost_model",
            True,
            observed["bounded_route_cost_model"],
        ),
    ]
    return _summary(checks)


def _assignment_summary(graph: Mapping[str, Any], assignment: Mapping[str, Any]) -> JsonDict:
    """Expose small current assignment facts without attaching an exact label."""

    counts = [
        sum(int(assignment.get(variable, 0)) for variable in group["variables"])
        for group in graph["local_groups"]
    ]
    selected_states = [
        next(
            (
                index
                for index, variable in enumerate(group["variables"])
                if assignment.get(variable)
            ),
            None,
        )
        if counts[group_index] == 1
        else None
        for group_index, group in enumerate(graph["local_groups"])
    ]
    relations = Counter(edge["relation_type"] for edge in graph["dependency_edges"])
    return {
        "selected_count_by_group": counts,
        "selected_state_by_group": selected_states,
        "relation_type_counts": dict(sorted(relations.items())),
    }


def _motif_id(difficulty: str, summary: Mapping[str, Any], group_count: int) -> str:
    """Create a repeated family-blind motif from legal current observations."""

    counts = summary["selected_count_by_group"]
    if any(count != 1 for count in counts):
        surface = "local_imbalance"
    else:
        selected = [value for value in summary["selected_state_by_group"] if value is not None]
        surface = "balanced_even" if sum(selected) % 2 == 0 else "balanced_odd"
    return f"motif:{difficulty}:{surface}:groups_{group_count % 2}"


def _event_from_candidate(
    unit: Mapping[str, Any],
    source_row: Mapping[str, Any] | None,
    event_class: str,
    family_index: int,
) -> JsonDict:
    """Bind one source candidate to legal observations and a hidden exact receipt."""

    graph = deepcopy(unit["graph"])
    if source_row is None:
        assignment = deepcopy(unit["exact_assignments"][0])
        exact_receipt: JsonDict = {
            "exact_valid": True,
            "local_checks_passed": True,
            "failed_local_group_ids": [],
            "failed_dependency_ids": [],
        }
        planted: list[str] = []
        suffix = "clean"
    else:
        assignment = deepcopy(source_row["candidate_assignment"])
        exact_receipt = deepcopy(source_row["exact_receipt"])
        if event_class == "hard_cross_dependency_failure":
            planted = [f"dependency:{source_row['named_broken_dependency']}"]
            suffix = "cross"
        else:
            planted = [f"local:{source_row['named_broken_local_group']}"]
            suffix = "local"
    summary = _assignment_summary(graph, assignment)
    difficulty = str(unit["difficulty_stratum"])
    family = str(unit["topology_family"])
    local_ids = [f"local:{group['group_id']}" for group in graph["local_groups"]]
    dependency_ids = [f"dependency:{edge['dependency_id']}" for edge in graph["dependency_edges"]]
    poison_status = "none"
    if event_class == "easy_local_failure" and family_index % 8 == 0:
        poison_status = "provenance_conflict"
    elif event_class == "clean_valid_case" and family_index % 8 == 2:
        poison_status = "retention_conflict"
    event_id = f"route-{family.split('_')[-1]}-{family_index:02d}-{suffix}"
    observation = {
        "schema": "carnot.experiment_6790.legal_observation.v1",
        "candidate_graph": {
            "graph_id": unit["graph_id"],
            "local_groups": deepcopy(graph["local_groups"]),
            "dependency_edges": deepcopy(graph["dependency_edges"]),
            "candidate_assignment": assignment,
        },
        "reusable_motif_id": _motif_id(difficulty, summary, len(local_ids)),
        "topology_family": family,
        "difficulty": difficulty,
        "provenance_class": "unattributed"
        if poison_status == "provenance_conflict"
        else "exact_source",
        "retention_rule": "preserve_anchor"
        if poison_status == "retention_conflict"
        else "standard",
        "candidate_assignment_summary": summary,
    }
    return {
        "schema": "carnot.experiment_6790.frozen_event.v1",
        "event_id": event_id,
        "source_unit_id": unit["unit_id"],
        "source_graph_id": unit["graph_id"],
        "event_class": event_class,
        "topology_family": family,
        "topology": family.rsplit("_", 1)[-1],
        "difficulty": difficulty,
        "held_future": family == HELD_FUTURE_FAMILY,
        "poison_status": poison_status,
        "reusable_motif_id": observation["reusable_motif_id"],
        "legal_observation": observation,
        "available_actions": list(LIVE_ROUTE_IDS),
        "all_factor_ids": [*local_ids, *dependency_ids],
        "exhaustive_route_cost": len(local_ids) + len(dependency_ids),
        "planted_failure_factor_ids": planted,
        "exact_failed_factor_ids": [
            *(f"local:{group_id}" for group_id in exact_receipt.get("failed_local_group_ids", [])),
            *(
                f"dependency:{dependency_id}"
                for dependency_id in exact_receipt.get("failed_dependency_ids", [])
            ),
        ],
        "exact_failed_dependency_ids": list(exact_receipt.get("failed_dependency_ids", [])),
        "exact_receipt": exact_receipt,
    }


def build_events(source: Mapping[str, Any]) -> list[JsonDict]:
    """Compose 240 source-bound events with 80 events per topology family."""

    units = source.get("frozen_manifest", {}).get("units", [])
    rows = source.get("rows", [])
    row_by_key = {
        (row["unit_id"], row["negative_class"]): row for row in rows if isinstance(row, Mapping)
    }
    events: list[JsonDict] = []
    for family in TOPOLOGY_FAMILIES:
        family_units = sorted(
            (unit for unit in units if unit.get("topology_family") == family),
            key=lambda unit: unit["unit_id"],
        )
        if len(family_units) != 32:
            raise ValueError(f"source family must contain 32 units: {family}")
        for family_index, unit in enumerate(family_units):
            for event_class in (
                "hard_cross_dependency_failure",
                "easy_local_failure",
            ):
                source_row = row_by_key.get((unit["unit_id"], event_class))
                if source_row is None:
                    raise ValueError(f"missing source row: {unit['unit_id']}:{event_class}")
                events.append(_event_from_candidate(unit, source_row, event_class, family_index))
            if family_index % 2 == 0:
                events.append(_event_from_candidate(unit, None, "clean_valid_case", family_index))
    if len(events) != EVENT_COUNT or len({event["event_id"] for event in events}) != EVENT_COUNT:
        raise ValueError("event composition must produce 240 unique events")
    return events


def freeze_orders(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze five difficulty-drifting orders with the held family always last."""

    orders: list[JsonDict] = []
    for order_index in range(ORDER_COUNT):
        event_ids: list[str] = []
        for held in (False, True):
            for difficulty in DIFFICULTIES:
                block = [
                    str(event["event_id"])
                    for event in events
                    if event["held_future"] is held and event["difficulty"] == difficulty
                ]
                random.Random(
                    RANDOM_SEED + order_index * 101 + DIFFICULTY_RANK[difficulty]
                ).shuffle(block)
                event_ids.extend(block)
        order: JsonDict = {
            "order_id": f"order_{order_index + 1}",
            "event_ids": event_ids,
            "held_future_first_position": next(
                index
                for index, event_id in enumerate(event_ids)
                if next(event for event in events if event["event_id"] == event_id)["held_future"]
            ),
            "frozen_before_replay": True,
        }
        order["order_hash"] = sha256_json(order)
        orders.append(order)
    return orders


def single_event_order(event: Mapping[str, Any]) -> JsonDict:
    """Build one tiny order for defensive replay tests without changing the main floor."""

    order: JsonDict = {
        "order_id": "order_1",
        "event_ids": [event["event_id"]],
        "held_future_first_position": 0 if event["held_future"] else None,
        "frozen_before_replay": True,
    }
    order["order_hash"] = sha256_json(order)
    return order


def route_factor_ids(event: Mapping[str, Any], route_id: str) -> list[str]:
    """Resolve one named route without reading any exact failure fields."""

    factors = list(event["all_factor_ids"])
    local = [factor for factor in factors if factor.startswith("local:")]
    dependency = [factor for factor in factors if factor.startswith("dependency:")]
    if route_id == "local_prefix":
        return local[:LIVE_ROUTE_BUDGET]
    if route_id == "local_suffix":
        return local[-LIVE_ROUTE_BUDGET:]
    if route_id == "dependency_prefix":
        return dependency[:LIVE_ROUTE_BUDGET]
    if route_id == "dependency_suffix":
        return dependency[-LIVE_ROUTE_BUDGET:]
    if route_id == "mixed_boundary":
        return [local[0], dependency[0], dependency[-1]]
    if route_id == EXHAUSTIVE_ROUTE_ID:
        return factors
    raise ValueError(f"unknown route: {route_id}")


def route_cost(event: Mapping[str, Any], route_id: str) -> int:
    """Charge one unit for each factor a route actually inspects."""

    return len(route_factor_ids(event, route_id))


def choose_frozen_action(observation: Mapping[str, Any]) -> str:
    """Apply a fixed untuned rule that leaves both useful signal and headroom."""

    counts = observation["candidate_assignment_summary"]["selected_count_by_group"]
    return "local_prefix" if any(count != 1 for count in counts) else "dependency_prefix"


def _random_action(order_id: str, event_id: str) -> str:
    """Select a seeded uniform route without reading current or future receipts."""

    seed_material = f"{RANDOM_SEED}:{order_id}:{event_id}".encode()
    index = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % len(LIVE_ROUTE_IDS)
    return LIVE_ROUTE_IDS[index]


def evaluate_route(event: Mapping[str, Any], route_id: str) -> JsonDict:
    """Reveal route outcome only after the caller has fixed the named action."""

    checked = route_factor_ids(event, route_id)
    planted = set(event["planted_failure_factor_ids"])
    failed = set(event["exact_failed_factor_ids"])
    found_planted = bool(planted.intersection(checked))
    correct = found_planted if planted else True
    apparent = int(correct)
    credited = apparent if event["poison_status"] == "none" else 0
    return {
        "route_id": route_id,
        "route_cost": len(checked),
        "checked_factors": checked,
        "found_planted_failure": found_planted,
        "correct_decision": correct,
        "missed_dependencies": [
            dependency_id
            for dependency_id in event["exact_failed_dependency_ids"]
            if f"dependency:{dependency_id}" not in checked
        ],
        "safe_factors": [factor for factor in checked if factor not in failed],
        "apparent_reward": apparent,
        "credited_reward": credited,
    }


def _history_snapshot(
    family_statistics: Mapping[str, Any],
    motif_statistics: Mapping[str, int],
    retrieval_memory: Sequence[Mapping[str, Any]],
    tuning_data: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Copy only evidence from events that already closed in the same order."""

    return {
        "factor_statistics": deepcopy(dict(family_statistics)),
        "motif_statistics": deepcopy(dict(motif_statistics)),
        "retrieval_memory": deepcopy(list(retrieval_memory[-4:])),
        "tuning_data": deepcopy(list(tuning_data[-4:])),
    }


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a row without its self-referential checksum field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def build_rows(
    events: Sequence[Mapping[str, Any]], orders: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Replay each order while revealing exact receipts only after both actions."""

    event_by_id = {str(event["event_id"]): event for event in events}
    rows: list[JsonDict] = []
    for order in orders:
        family_statistics: JsonDict = {}
        motif_statistics: dict[str, int] = {}
        retrieval_memory: list[JsonDict] = []
        tuning_data: list[JsonDict] = []
        for position, event_id in enumerate(order["event_ids"]):
            event = event_by_id[str(event_id)]
            history = _history_snapshot(
                family_statistics, motif_statistics, retrieval_memory, tuning_data
            )
            observation = deepcopy(event["legal_observation"])
            frozen_action = choose_frozen_action(observation)
            random_action = _random_action(str(order["order_id"]), str(event_id))
            receipt = {
                "schema": "carnot.experiment_6790.post_action_receipt.v1",
                "event_id": event_id,
                "poison_status": event["poison_status"],
                "frozen_policy": evaluate_route(event, frozen_action),
                "random_route": evaluate_route(event, random_action),
                "exhaustive_diagnostic": {
                    **evaluate_route(event, EXHAUSTIVE_ROUTE_ID),
                    "live_action": False,
                },
                "exact_event_valid": event["exact_receipt"]["exact_valid"],
            }
            row: JsonDict = {
                "schema": "carnot.experiment_6790.event_order_row.v1",
                "row_id": f"{order['order_id']}:{event_id}",
                "order_id": order["order_id"],
                "event_id": event_id,
                "chronology": {
                    "position": position,
                    "prior_closed_event_count": position,
                    "receipt_visible_at_action": False,
                    "actions_fixed_before_receipt": True,
                },
                "pre_action": {
                    "legal_observation": observation,
                    "history_snapshot": history,
                },
                "available_actions": list(event["available_actions"]),
                "chosen_baseline_actions": {
                    "frozen_policy": frozen_action,
                    "random_route": random_action,
                },
                "hidden_receipt_hash": sha256_json(receipt),
                "revealed_post_action_receipt": receipt,
                "reusable_motif_id": event["reusable_motif_id"],
                "reusable_signal": motif_statistics.get(event["reusable_motif_id"], 0) > 0,
                "topology": event["topology"],
                "topology_family": event["topology_family"],
                "difficulty": event["difficulty"],
                "held_future": event["held_future"],
                "poison_status": event["poison_status"],
                "route_cost": {
                    "frozen_policy": receipt["frozen_policy"]["route_cost"],
                    "random_route": receipt["random_route"]["route_cost"],
                },
            }
            row["row_sha256"] = row_checksum(row)
            rows.append(row)

            family = str(event["topology_family"])
            stats = family_statistics.setdefault(
                family, {"closed_event_count": 0, "found_failure_count": 0}
            )
            stats["closed_event_count"] += 1
            stats["found_failure_count"] += int(receipt["frozen_policy"]["found_planted_failure"])
            motif = str(event["reusable_motif_id"])
            motif_statistics[motif] = motif_statistics.get(motif, 0) + 1
            memory_record = {
                "event_id": event_id,
                "topology_family": family,
                "reusable_motif_id": motif,
            }
            if event["poison_status"] == "none":
                retrieval_memory.append(memory_record)
                tuning_data.append(memory_record)
    return rows


def _walk_keys(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Return nested object keys so denied fields cannot hide inside graph data."""

    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            found.append((str(key), nested_path))
            found.extend(_walk_keys(nested, nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_walk_keys(nested, f"{path}[{index}]"))
    return found


def audit_feature_contract(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Find any exact, future, or order field inside a legal observation."""

    violations: list[str] = []
    denied = set(FEATURE_DENYLIST)
    allowed = set(FEATURE_ALLOWLIST)
    for row in rows:
        observation = row["pre_action"]["legal_observation"]
        row_id = str(row["row_id"])
        for key in observation:
            if key not in allowed or key in denied:
                violations.append(f"{row_id}.{key}")
        for key, path in _walk_keys(observation):
            if key in denied:
                violation = f"{row_id}.{path}"
                if violation not in violations:
                    violations.append(violation)
    return sorted(violations)


def audit_row_consistency(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    orders: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Recheck coverage, chronology, actions, hashes, receipts, and fixed costs."""

    violations: list[str] = []
    event_by_id = {event["event_id"]: event for event in events}
    row_by_key = {(row["order_id"], row["event_id"]): row for row in rows}
    if len(row_by_key) != len(rows):
        violations.append("duplicate_row_keys")
    for order in orders:
        for position, event_id in enumerate(order["event_ids"]):
            row = row_by_key.get((order["order_id"], event_id))
            if row is None:
                violations.append(f"missing:{order['order_id']}:{event_id}")
                continue
            event = event_by_id[event_id]
            if row["chronology"]["position"] != position:
                violations.append(f"position:{row['row_id']}")
            if set(row["available_actions"]) != set(event["available_actions"]):
                violations.append(f"actions:{row['row_id']}")
            if any(
                action not in row["available_actions"]
                for action in row["chosen_baseline_actions"].values()
            ):
                violations.append(f"choice:{row['row_id']}")
            if row["hidden_receipt_hash"] != sha256_json(row["revealed_post_action_receipt"]):
                violations.append(f"receipt:{row['row_id']}")
            if row["row_sha256"] != row_checksum(row):
                violations.append(f"hash:{row['row_id']}")
            if any(cost != LIVE_ROUTE_BUDGET for cost in row["route_cost"].values()):
                violations.append(f"cost:{row['row_id']}")
    return sorted([*violations, *audit_feature_contract(rows)])


def _policy_metrics(rows: Sequence[Mapping[str, Any]], policy: str) -> JsonDict:
    """Reduce one baseline directly from its post-action row receipts."""

    correct = sum(row["revealed_post_action_receipt"][policy]["correct_decision"] for row in rows)
    credited = sum(row["revealed_post_action_receipt"][policy]["credited_reward"] for row in rows)
    total_cost = sum(row["revealed_post_action_receipt"][policy]["route_cost"] for row in rows)
    return {
        "event_count": len(rows),
        "correct_decisions": correct,
        "decision_accuracy": correct / len(rows) if rows else 0.0,
        "credited_reward": credited,
        "total_route_cost": total_cost,
        "actual_alternative_count": sum(len(row["available_actions"]) > 1 for row in rows),
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], orders: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Derive every count, baseline, and headroom field from event-order rows."""

    summary: JsonDict = {
        "event_count_by_order": {},
        "topology_count_by_order": {},
        "held_future_counts": {},
        "poison_counts": {},
        "reusable_motif_counts": {},
        "frozen_policy_metrics_by_order": {},
        "random_route_metrics_by_order": {},
        "diagnostic_headroom_by_order": {},
    }
    for order in orders:
        order_id = str(order["order_id"])
        order_rows = [row for row in rows if row["order_id"] == order_id]
        frozen = _policy_metrics(order_rows, "frozen_policy")
        random_metrics = _policy_metrics(order_rows, "random_route")
        exhaustive = _policy_metrics(order_rows, "exhaustive_diagnostic")
        summary["event_count_by_order"][order_id] = len(order_rows)
        summary["topology_count_by_order"][order_id] = dict(
            sorted(Counter(row["topology_family"] for row in order_rows).items())
        )
        summary["held_future_counts"][order_id] = sum(row["held_future"] for row in order_rows)
        summary["poison_counts"][order_id] = sum(
            row["poison_status"] != "none" for row in order_rows
        )
        summary["reusable_motif_counts"][order_id] = sum(
            row["reusable_signal"] for row in order_rows
        )
        summary["frozen_policy_metrics_by_order"][order_id] = frozen
        summary["random_route_metrics_by_order"][order_id] = random_metrics
        summary["diagnostic_headroom_by_order"][order_id] = {
            "frozen_decision_accuracy": frozen["decision_accuracy"],
            "exhaustive_decision_accuracy": exhaustive["decision_accuracy"],
            "accuracy_gap": exhaustive["decision_accuracy"] - frozen["decision_accuracy"],
            "exhaustive_route_is_live": False,
        }
    return summary


def replay_hashes(
    rows: Sequence[Mapping[str, Any]], orders: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Hash each chronological row list and their ordered aggregate."""

    order_hashes = {
        order["order_id"]: sha256_json(
            [
                row["row_sha256"]
                for row in sorted(
                    (row for row in rows if row["order_id"] == order["order_id"]),
                    key=lambda row: row["chronology"]["position"],
                )
            ]
        )
        for order in orders
    }
    return {
        "order_row_hashes": order_hashes,
        "aggregate_rows_sha256": sha256_json(order_hashes),
        "row_count": len(rows),
    }


def run_cold_replay(
    events: Sequence[Mapping[str, Any]],
    orders: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Start a fresh process and require it to rebuild all row hashes."""

    expected = replay_hashes(build_rows(events, orders), orders)
    environment = os.environ.copy()
    python_path = str(repo_root / "python")
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    process = subprocess.run(
        [sys.executable, "-m", __name__, "--cold-replay-worker"],
        input=json.dumps({"events": events, "orders": orders}, sort_keys=True),
        text=True,
        capture_output=True,
        cwd=repo_root,
        env=environment,
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"cold replay failed: {process.stderr.strip()}")
    receipt = json.loads(process.stdout)
    receipt.update(
        {
            "agreement": receipt["aggregate_rows_sha256"] == expected["aggregate_rows_sha256"]
            and receipt["order_row_hashes"] == expected["order_row_hashes"],
            "fresh_process": receipt["cold_pid"] != os.getpid(),
            "producer_pid": os.getpid(),
            "worker_exit_code": process.returncode,
        }
    )
    return receipt


def _frozen_manifest(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Seal complete events while keeping their exact fields outside action inputs."""

    manifest: JsonDict = {
        "schema": "carnot.experiment_6790.frozen_manifest.v1",
        "random_seed": RANDOM_SEED,
        "event_count": len(events),
        "held_future_family": HELD_FUTURE_FAMILY,
        "live_route_budget": LIVE_ROUTE_BUDGET,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "events": deepcopy(list(events)),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding measured time and process identities."""

    cold = artifact.get("cold_replay_hashes", {})
    stable_cold = {
        key: cold.get(key)
        for key in (
            "agreement",
            "fresh_process",
            "worker_exit_code",
            "order_row_hashes",
            "aggregate_rows_sha256",
            "row_count",
        )
    }
    material = {
        key: artifact.get(key)
        for key in REQUIRED_ARTIFACT_FIELDS
        if key not in {"duration_s", "reproducibility_checksum", "cold_replay_hashes"}
    }
    material["cold_replay_hashes"] = stable_cold
    return sha256_json(material)


def _empty_summary() -> JsonDict:
    """Return empty row-derived fields for a precondition-blocked artifact."""

    return {
        "event_count_by_order": {},
        "topology_count_by_order": {},
        "held_future_counts": {},
        "poison_counts": {},
        "reusable_motif_counts": {},
        "frozen_policy_metrics_by_order": {},
        "random_route_metrics_by_order": {},
        "diagnostic_headroom_by_order": {},
    }


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_hash: str | None,
    gate_summary: Mapping[str, Any],
) -> JsonDict:
    """Build the complete no-row artifact required for any failed authority gate."""

    summary = _empty_summary()
    first = first_failed_check(gate_summary)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_constraint_routing_stream",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hash": source_hash,
        "frozen_manifest": {},
        "order_definitions": [],
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "route_definitions": deepcopy(ROUTE_DEFINITIONS),
        "rows": [],
        **summary,
        "future_feature_violations": [],
        "cold_replay_hashes": {
            "agreement": False,
            "fresh_process": False,
            "worker_exit_code": None,
            "order_row_hashes": {},
            "aggregate_rows_sha256": sha256_json({}),
            "row_count": 0,
        },
        "constraint_routing_stream_ready": False,
        "gate_check_summary": deepcopy(dict(gate_summary)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": (
            "complete_blocked_constraint_routing_stream: "
            f"{first['check']} observed {first['observed']}"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _validate_run_date(run_date: str) -> None:
    """Reject ambiguous dates before any source or output work begins."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    source_artifact_path: Path | None = None,
    duration_s: float | None = None,
    precondition_overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a ready opportunity stream, or a complete blocked result."""

    _validate_run_date(run_date)
    started = time.monotonic()
    source_path = source_artifact_path or repo_root / SOURCE_ARTIFACT_RELATIVE_PATH
    source_hash = sha256_file(source_path)
    preconditions = evaluate_preconditions(
        repo_root=repo_root,
        source_artifact_path=source_path,
        overrides=precondition_overrides,
    )
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    if not preconditions["all_passed"]:
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=measured,
            source_hash=source_hash,
            gate_summary=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return artifact

    source = load_json_object(source_path)
    events = build_events(source)
    orders = freeze_orders(events)
    rows = build_rows(events, orders)
    summary = summarize_rows(rows, orders)
    feature_violations = audit_feature_contract(rows)
    row_violations = audit_row_consistency(rows, events, orders)
    cold = run_cold_replay(events, orders, repo_root=repo_root)
    opportunity_checks = [
        *preconditions["checks"],
        _gate("event_count", EVENT_COUNT, len(events)),
        _gate("order_count", ORDER_COUNT, len(orders)),
        _gate("row_count", EVENT_COUNT * ORDER_COUNT, len(rows)),
        _gate(
            "actual_alternatives_each_order",
            True,
            all(
                metrics["actual_alternative_count"] == EVENT_COUNT
                for metrics in summary["frozen_policy_metrics_by_order"].values()
            ),
        ),
        _gate(
            "reusable_signal_each_order",
            True,
            all(count > 0 for count in summary["reusable_motif_counts"].values()),
        ),
        _gate(
            "held_future_each_order",
            True,
            all(count > 0 for count in summary["held_future_counts"].values()),
        ),
        _gate(
            "poison_each_order",
            True,
            all(count > 0 for count in summary["poison_counts"].values()),
        ),
        _gate(
            "floor_and_headroom_each_order",
            True,
            all(
                summary["random_route_metrics_by_order"][order_id]["decision_accuracy"]
                < summary["frozen_policy_metrics_by_order"][order_id]["decision_accuracy"]
                < headroom["exhaustive_decision_accuracy"]
                and headroom["accuracy_gap"] > 0
                for order_id, headroom in summary["diagnostic_headroom_by_order"].items()
            ),
        ),
        _gate("future_feature_violations", [], feature_violations),
        _gate("row_consistency_violations", [], row_violations),
        _gate("cold_replay_agreement", True, cold["agreement"]),
        _gate("cold_replay_fresh_process", True, cold["fresh_process"]),
    ]
    gate_summary = _summary(opportunity_checks)
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    if not gate_summary["all_passed"]:
        return _blocked_artifact(
            run_date=run_date,
            duration_s=measured,
            source_hash=source_hash,
            gate_summary=gate_summary,
        )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": measured,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hash": source_hash,
        "frozen_manifest": _frozen_manifest(events),
        "order_definitions": orders,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "route_definitions": deepcopy(ROUTE_DEFINITIONS),
        "rows": rows,
        **summary,
        "future_feature_violations": feature_violations,
        "cold_replay_hashes": cold,
        "constraint_routing_stream_ready": True,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": (
            "complete: chronological constraint-routing stream ready with 240 events, "
            "five orders, held-future isolation, poison, and non-saturated headroom; "
            "this is not an online-learning gain claim"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all closed-schema errors without changing stored evidence."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field principle coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be non-negative")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random seed mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed enum")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks a terminal prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must remain false")
    ready = artifact.get("constraint_routing_stream_ready") is True
    if ready and artifact.get("status") != "complete":
        errors.append("ready artifact status mismatch")
    if ready and len(artifact.get("rows", [])) != EVENT_COUNT * ORDER_COUNT:
        errors.append("ready artifact row count mismatch")
    if ready and len(artifact.get("order_definitions", [])) != ORDER_COUNT:
        errors.append("ready artifact order count mismatch")
    if ready and artifact.get("future_feature_violations") != []:
        errors.append("ready artifact contains future feature violations")
    if ready and artifact.get("gate_check_summary", {}).get("all_passed") is not True:
        errors.append("ready artifact has failed gates")
    if ready and artifact.get("cold_replay_hashes", {}).get("agreement") is not True:
        errors.append("ready artifact lacks cold replay agreement")
    if ready:
        for order_id, headroom in artifact.get("diagnostic_headroom_by_order", {}).items():
            frozen = artifact.get("frozen_policy_metrics_by_order", {}).get(order_id, {})
            random_metrics = artifact.get("random_route_metrics_by_order", {}).get(order_id, {})
            if not (
                headroom.get("accuracy_gap", 0) > 0
                and random_metrics.get("decision_accuracy", 1)
                < frozen.get("decision_accuracy", 0)
                < headroom.get("exhaustive_decision_accuracy", 0)
            ):
                errors.append(f"ready artifact lacks headroom: {order_id}")
    if not ready and artifact.get("status") != "complete_blocked_constraint_routing_stream":
        errors.append("not-ready artifact must use complete blocked status")
    if not ready and artifact.get("rows") != []:
        errors.append("blocked artifact must not contain rows")
    if not ready and artifact.get("verdict_class") != "blocked":
        errors.append("blocked artifact must use blocked verdict class")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def _atomic_write(path: Path, data: bytes) -> None:
    """Publish one complete file and remove a temporary file after any failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_artifact(
    *,
    artifact_path: Path | str = RESULT_RELATIVE_PATH,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    source_artifact_path: Path | None = None,
    duration_s: float | None = None,
    precondition_overrides: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Validate and atomically publish the task-owned terminal JSON object."""

    artifact = build_artifact(
        run_date=run_date,
        repo_root=repo_root,
        source_artifact_path=source_artifact_path,
        duration_s=duration_s,
        precondition_overrides=precondition_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(artifact_path)
    if not path.is_absolute():
        path = repo_root / path
    data = (
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    _atomic_write(path, data)
    return artifact


def _cold_replay_worker() -> int:
    """Read frozen inputs from standard input and emit fresh-process row hashes."""

    payload = json.load(sys.stdin)
    rows = build_rows(payload["events"], payload["orders"])
    receipt = replay_hashes(rows, payload["orders"])
    receipt["cold_pid"] = os.getpid()
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the frozen date and publish the chronological stream artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--cold-replay-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cold_replay_worker:
        return _cold_replay_worker()
    artifact = write_artifact(run_date=args.date, artifact_path=args.output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper calls main.
    raise SystemExit(main())
