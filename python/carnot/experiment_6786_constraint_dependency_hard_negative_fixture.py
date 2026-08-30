"""Build an exact graph fixture without proposal-time answer access.

Spec refs: REQ-VERIFY-6786 and SCENARIO-VERIFY-6786-*.

The fixture separates local one-hot checks from dependencies between groups.
This separation creates hard negatives that look locally valid. A later model
can use the graph structure, but only the CPU enumerator can attach labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5299_constraint_lns_solver_repair_fixture_v484 as repair_fixture
from carnot import experiment_5304_kan_dynamic_abstraction_spotcheck_v484 as abstraction_fixture
from carnot import experiment_6768_targetable_proof_panel_expansion as source_fixture


JsonDict = dict[str, Any]
GroupBuilder = Callable[[], Sequence[Mapping[str, Any]]]
ExactEnumerator = Callable[[Mapping[str, Any]], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_6786_constraint_dependency_hard_negative_fixture"
SCHEMA = "carnot.experiment_6786.constraint_dependency_hard_negative_fixture.v1"
RUN_DATE = "20260830"
RANDOM_SEED = 6_786_000
INFERENCE_SUBSTRATE = "cpu_exact_enumeration_no_llm"
SOURCE_PANEL_RELATIVE_PATH = Path("results/experiment_6768_targetable_proof_panel_expansion.json")
RESULT_RELATIVE_PATH = Path("results") / f"{EXPERIMENT_ID}.json"
SOURCE_BUILDER_RELATIVE_PATHS = (
    Path("python/carnot/experiment_5299_constraint_lns_solver_repair_fixture_v484.py"),
    Path("python/carnot/experiment_5304_kan_dynamic_abstraction_spotcheck_v484.py"),
)
UNIT_COUNT_PER_FAMILY = 32
UNIT_COUNT = 96
ROW_COUNT = 192
NEGATIVE_CLASSES = (
    "hard_cross_dependency_failure",
    "easy_local_failure",
)
TOPOLOGY_SPECS: tuple[JsonDict, ...] = (
    {
        "family_id": "directed_implication_chain",
        "shape": "chain",
        "split": "train",
    },
    {
        "family_id": "directed_implication_star",
        "shape": "star",
        "split": "development",
    },
    {
        "family_id": "directed_implication_cycle",
        "shape": "cycle",
        "split": "held_topology_test",
    },
)
DIFFICULTY_STRATA = ("easy", "medium", "hard", "challenge")
FEATURE_ALLOWLIST = (
    "schema",
    "graph_id",
    "topology_family",
    "difficulty_stratum",
    "variables",
    "local_groups",
    "dependency_edges",
)
REQUIRED_DENIED_FEATURES = (
    "exact_assignments",
    "exact_valid",
    "future_receipts",
    "solver_conflicts",
    "oracle_residuals",
    "post_action_outcomes",
)
FEATURE_DENYLIST = REQUIRED_DENIED_FEATURES + (
    "exact_certificate",
    "exact_label",
    "ground_truth_assignment",
    "cold_replay_receipt",
    "group_receipts",
    "dependency_receipts",
    "failed_cross_dependencies",
    "post_action_outcome",
    "oracle_residual",
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
STANDARD_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
)
REQUIRED_ARTIFACT_FIELDS = STANDARD_ARTIFACT_FIELDS + (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "frozen_manifest",
    "topology_families",
    "split_by_topology",
    "feature_allowlist",
    "feature_denylist",
    "rows",
    "exact_label_counts_by_split",
    "local_pass_cross_dependency_fail_counts",
    "easy_negative_counts",
    "duplicate_rows",
    "future_feature_violations",
    "cold_replay_agreement",
    "constraint_group_fixture_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible fixture readers fail closed.",
    "experiment_id": "A stable ID binds the artifact to the planned exact fixture.",
    "run_date": "The fixed date identifies this frozen fixture execution.",
    "status": "A terminal status separates readiness from a complete blocked result.",
    "field_principles": "A one-line purpose makes every required field auditable.",
    "inference_substrate": "The substrate prevents CPU labels from becoming an LLM claim.",
    "duration_s": "Measured wall time shows that generation and replay executed.",
    "random_seed": "One seed fixes graph relations, difficulty, and row identities.",
    "reproducibility_checksum": "A stable hash detects fixture or label drift.",
    "source_artifact_hashes": "Source hashes bind the panel and group builders to this run.",
    "frozen_manifest": "The manifest freezes every unit graph and exact certificate.",
    "topology_families": "Family metadata proves coverage of three graph shapes.",
    "split_by_topology": "Whole-family splits prevent random-row topology leakage.",
    "feature_allowlist": "The allowlist defines the only legal proposal inputs.",
    "feature_denylist": "The denylist names answer and future fields that proposals cannot read.",
    "rows": "Unit-negative rows preserve candidate, group, and dependency receipts.",
    "exact_label_counts_by_split": "Split counts expose the exact-label denominator.",
    "local_pass_cross_dependency_fail_counts": "Counts prove hard negatives exist in every split.",
    "easy_negative_counts": "Counts prove local-failure controls exist in every split.",
    "duplicate_rows": "Zero duplicates protects the planned denominator.",
    "future_feature_violations": "An empty list proves proposal features passed the denylist audit.",
    "cold_replay_agreement": "A fresh process must recompute every label and failure class.",
    "constraint_group_fixture_ready": "This exact gate authorizes the two downstream V592 branches.",
    "gate_check_summary": "Each gate records its expected and observed value.",
    "verifier_is_oracle": "False records that the exact checker labels but never proposes.",
    "verdict_class": "A closed class keeps the terminal result machine-readable.",
    "honest_verdict": "A terminal prefix reports completion without a model-quality claim.",
}


def canonical_json(value: Any) -> str:
    """Serialize graph data once so identity is independent of JSON formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    """Hash canonical JSON and mark the digest algorithm in the stored value."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def sha256_file(path: Path) -> str | None:
    """Hash a required source, or return no value for a blocked precondition."""

    if not path.is_file():
        return None
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def load_json_object(path: Path) -> JsonDict:
    """Load one required JSON object and reject non-object roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record one precondition or fixture gate with its real observation."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else passed,
    }


def _summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Copy gate rows and name the first failure for blocked artifacts."""

    copied = [deepcopy(dict(check)) for check in checks]
    failures = [str(check["check"]) for check in copied if check.get("passed") is not True]
    first = next((check for check in copied if check.get("passed") is not True), None)
    return {
        "all_passed": not failures,
        "checks": copied,
        "failed_checks": failures,
        "first_failure": first,
    }


def first_failed_check(summary: Mapping[str, Any]) -> JsonDict:
    """Return the first failed check or an explicit all-passed receipt."""

    first = summary.get("first_failure")
    if isinstance(first, Mapping):
        return deepcopy(dict(first))
    return _gate("all_preconditions", True, True)


def _smoke_unit() -> JsonDict:
    """Provide a two-group case that proves the enumerator executes semantics."""

    return {
        "graph": {
            "local_groups": [
                {
                    "group_id": "g00",
                    "group_type": "one_hot_domain",
                    "variables": ["g00_v0", "g00_v1"],
                },
                {
                    "group_id": "g01",
                    "group_type": "one_hot_domain",
                    "variables": ["g01_v0", "g01_v1"],
                },
            ],
            "dependency_edges": [
                {
                    "dependency_id": "d00",
                    "source_group": "g00",
                    "target_group": "g01",
                    "relation_type": "implies_selected_one",
                }
            ],
        }
    }


def evaluate_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    source_panel_path: Path | None = None,
    group_builders: Sequence[GroupBuilder] | None = None,
    exact_enumerator: ExactEnumerator | None = None,
    topology_specs: Sequence[Mapping[str, Any]] = TOPOLOGY_SPECS,
) -> JsonDict:
    """Require the panel, row hashes, group builders, enumerator, and families."""

    panel_path = source_panel_path or repo_root / SOURCE_PANEL_RELATIVE_PATH
    panel_exists = panel_path.is_file()
    panel: JsonDict = load_json_object(panel_path) if panel_exists else {}
    rows = panel.get("rows", []) if isinstance(panel.get("rows", []), list) else []
    hash_count = sum(
        isinstance(row, Mapping) and row.get("row_sha256") == source_fixture.row_checksum(row)
        for row in rows
    )

    active_builders = group_builders or (
        repair_fixture.constraint_group_metadata,
        abstraction_fixture.declarative_constraint_groups,
    )
    try:
        builder_groups = [list(builder()) for builder in active_builders]
        builder_observation = {
            "builder_count": len(active_builders),
            "nonempty_builder_count": sum(bool(groups) for groups in builder_groups),
        }
        builders_ready = len(active_builders) == 2 and all(builder_groups)
    except Exception as exc:  # pragma: no cover - defensive import failure becomes a gate row.
        builder_observation = {"error": f"{type(exc).__name__}:{exc}"}
        builders_ready = False

    active_enumerator = exact_enumerator or enumerate_exact_semantics
    try:
        smoke = active_enumerator(_smoke_unit())
        enumeration_observation = {
            "enumerated_state_count": smoke.get("enumerated_state_count"),
            "exact_assignment_count": len(smoke.get("exact_assignments", [])),
        }
        enumeration_ready = enumeration_observation == {
            "enumerated_state_count": 4,
            "exact_assignment_count": 3,
        }
    except Exception as exc:  # pragma: no cover - defensive solver failure becomes a gate row.
        enumeration_observation = {"error": f"{type(exc).__name__}:{exc}"}
        enumeration_ready = False

    family_ids = [str(spec.get("family_id")) for spec in topology_specs]
    checks = [
        _gate("exp6768_artifact_exists", True, panel_exists),
        _gate(
            "exp6768_panel_ready",
            True,
            panel.get("targetable_panel_ready") is True,
        ),
        _gate(
            "exp6768_row_count",
            126,
            len(rows),
            len(rows) == 126 and panel.get("targetable_row_count") == 126,
        ),
        _gate("exp6768_row_hashes", 126, hash_count),
        _gate(
            "declarative_constraint_group_builders",
            {"builder_count": 2, "nonempty_builder_count": 2},
            builder_observation,
            builders_ready,
        ),
        _gate(
            "cpu_exact_enumeration_path",
            {"enumerated_state_count": 4, "exact_assignment_count": 3},
            enumeration_observation,
            enumeration_ready,
        ),
        _gate(
            "minimum_topology_families",
            ">=3 unique families",
            {"count": len(set(family_ids)), "family_ids": family_ids},
            len(family_ids) >= 3 and len(set(family_ids)) == len(family_ids),
        ),
    ]
    return _summary(checks)


def _topology_edge_pairs(shape: str, group_count: int) -> list[tuple[int, int]]:
    """Return ordered directed edges for one declared graph family."""

    if shape == "chain":
        return [(index, index + 1) for index in range(group_count - 1)]
    if shape == "star":
        return [(0, index) for index in range(1, group_count)]
    if shape == "cycle":
        return [(index, (index + 1) % group_count) for index in range(group_count)]
    raise ValueError(f"unknown topology shape: {shape}")


def _expand_states(groups: Sequence[Mapping[str, Any]], states: Sequence[int]) -> JsonDict:
    """Convert one state per group into the two indicator variables it owns."""

    assignment: JsonDict = {}
    for group, state in zip(groups, states, strict=True):
        variables = group["variables"]
        assignment[str(variables[0])] = int(state == 0)
        assignment[str(variables[1])] = int(state == 1)
    return assignment


def _dependency_passes(relation_type: str, source_state: int, target_state: int) -> bool:
    """Apply one named implication without using any exact assignment."""

    if relation_type == "implies_selected_one":
        return not (source_state == 1 and target_state == 0)
    if relation_type == "implies_selected_zero":
        return not (source_state == 0 and target_state == 1)
    raise ValueError(f"unknown dependency relation: {relation_type}")


def evaluate_candidate(unit: Mapping[str, Any], assignment: Mapping[str, Any]) -> JsonDict:
    """Recompute local and cross-group semantics for one candidate assignment."""

    graph = unit["graph"]
    states: dict[str, int | None] = {}
    group_receipts: list[JsonDict] = []
    failed_groups: list[str] = []
    for group in graph["local_groups"]:
        group_id = str(group["group_id"])
        variables = [str(variable) for variable in group["variables"]]
        values = [assignment.get(variable) for variable in variables]
        selected = [index for index, value in enumerate(values) if value == 1]
        passed = values.count(0) + values.count(1) == len(values) and len(selected) == 1
        states[group_id] = selected[0] if passed else None
        if not passed:
            failed_groups.append(group_id)
        group_receipts.append(
            {
                "group_id": group_id,
                "group_type": group["group_type"],
                "variables": variables,
                "values": values,
                "selected_count": len(selected),
                "passed": passed,
            }
        )

    dependency_receipts: list[JsonDict] = []
    failed_dependencies: list[str] = []
    for edge in graph["dependency_edges"]:
        source_state = states[str(edge["source_group"])]
        target_state = states[str(edge["target_group"])]
        evaluated = source_state is not None and target_state is not None
        passed = (
            _dependency_passes(str(edge["relation_type"]), source_state, target_state)
            if evaluated
            else None
        )
        dependency_id = str(edge["dependency_id"])
        if passed is False:
            failed_dependencies.append(dependency_id)
        dependency_receipts.append(
            {
                "dependency_id": dependency_id,
                "source_group": edge["source_group"],
                "target_group": edge["target_group"],
                "relation_type": edge["relation_type"],
                "source_state": source_state,
                "target_state": target_state,
                "evaluated": evaluated,
                "passed": passed,
            }
        )

    local_passed = not failed_groups
    exact_valid = (
        local_passed
        and not failed_dependencies
        and all(receipt["evaluated"] for receipt in dependency_receipts)
    )
    return {
        "local_checks_passed": local_passed,
        "failed_local_group_ids": failed_groups,
        "failed_dependency_ids": failed_dependencies,
        "exact_valid": exact_valid,
        "group_receipts": group_receipts,
        "dependency_receipts": dependency_receipts,
    }


def enumerate_exact_semantics(unit: Mapping[str, Any]) -> JsonDict:
    """Enumerate every bounded group state and retain only exact assignments."""

    groups = unit["graph"]["local_groups"]
    exact_assignments: list[JsonDict] = []
    enumerated_state_count = 0
    for states in product((0, 1), repeat=len(groups)):
        enumerated_state_count += 1
        assignment = _expand_states(groups, states)
        if evaluate_candidate(unit, assignment)["exact_valid"]:
            exact_assignments.append(assignment)
    return {
        "enumerated_state_count": enumerated_state_count,
        "exact_assignments": exact_assignments,
        "contradiction_certificate": (
            None
            if exact_assignments
            else {
                "certificate_type": "exhaustive_no_assignment",
                "enumerated_state_count": enumerated_state_count,
            }
        ),
    }


def _build_graph(spec: Mapping[str, Any], family_index: int, local_index: int) -> JsonDict:
    """Build one ordered graph with a deterministic relation mask."""

    group_count = 5 + local_index // 8
    groups = [
        {
            "group_id": f"g{index:02d}",
            "group_type": "one_hot_domain",
            "variables": [f"g{index:02d}_v0", f"g{index:02d}_v1"],
            "source_group_types": [
                "one_hot_domain",
                "bounded_pwa_upper_envelope",
                "ising_style_factor_boundary",
            ],
        }
        for index in range(group_count)
    ]
    edge_pairs = _topology_edge_pairs(str(spec["shape"]), group_count)
    relation_mask = (local_index % 8) ^ (family_index * 3)
    edges = [
        {
            "dependency_id": f"d{index:02d}",
            "source_group": f"g{source:02d}",
            "target_group": f"g{target:02d}",
            "relation_type": (
                "implies_selected_zero"
                if relation_mask & (1 << (index % 3))
                else "implies_selected_one"
            ),
        }
        for index, (source, target) in enumerate(edge_pairs)
    ]
    variables = [variable for group in groups for variable in group["variables"]]
    return {
        "serialization_version": "carnot.constraint_dependency_graph.v1",
        "topology_family": spec["family_id"],
        "topology_shape": spec["shape"],
        "variables": variables,
        "local_groups": groups,
        "dependency_edges": edges,
    }


def build_units(
    panel: Mapping[str, Any],
    *,
    topology_specs: Sequence[Mapping[str, Any]] = TOPOLOGY_SPECS,
) -> list[JsonDict]:
    """Build 96 source-linked graphs and attach their complete exact labels."""

    source_rows = sorted(panel["rows"], key=lambda row: str(row["row_id"]))
    if len(source_rows) < UNIT_COUNT:
        raise ValueError("Exp6768 panel has fewer than 96 source rows")
    units: list[JsonDict] = []
    for family_index, spec in enumerate(topology_specs):
        for local_index in range(UNIT_COUNT_PER_FAMILY):
            global_index = family_index * UNIT_COUNT_PER_FAMILY + local_index
            source_row = source_rows[global_index]
            graph = _build_graph(spec, family_index, local_index)
            graph_serialization = canonical_json(graph)
            unit: JsonDict = {
                "unit_id": f"u{global_index:03d}-{spec['family_id']}",
                "graph_id": sha256_json(graph),
                "graph_serialization": graph_serialization,
                "split": spec["split"],
                "topology_family": spec["family_id"],
                "difficulty_stratum": DIFFICULTY_STRATA[local_index // 8],
                "seed": RANDOM_SEED + global_index,
                "unit_role": "satisfiable",
                "graph": graph,
                "provenance": {
                    "source_fixture": str(SOURCE_PANEL_RELATIVE_PATH),
                    "source_panel_row_id": source_row["row_id"],
                    "source_panel_row_sha256": source_row["row_sha256"],
                    "source_panel_family": source_row["family"],
                    "group_builders": [
                        "experiment_5299.constraint_group_metadata",
                        "experiment_5304.declarative_constraint_groups",
                    ],
                },
            }
            exact = enumerate_exact_semantics(unit)
            unit["enumerated_state_count"] = exact["enumerated_state_count"]
            unit["exact_assignments"] = exact["exact_assignments"]
            unit["contradiction_certificate"] = exact["contradiction_certificate"]
            units.append(unit)
    return units


def _hard_negative(unit: Mapping[str, Any]) -> tuple[JsonDict, JsonDict]:
    """Find the first locally valid assignment that breaks one dependency."""

    groups = unit["graph"]["local_groups"]
    for states in product((0, 1), repeat=len(groups)):
        assignment = _expand_states(groups, states)
        receipt = evaluate_candidate(unit, assignment)
        if receipt["local_checks_passed"] and len(receipt["failed_dependency_ids"]) == 1:
            return assignment, receipt
    raise ValueError(f"no single-dependency hard negative for {unit['unit_id']}")


def proposal_features(unit: Mapping[str, Any]) -> JsonDict:
    """Project graph structure only, without exact assignments or receipts."""

    graph = unit["graph"]
    return {
        "schema": "carnot.experiment_6786.proposal_features.v1",
        "graph_id": unit["graph_id"],
        "topology_family": unit["topology_family"],
        "difficulty_stratum": unit["difficulty_stratum"],
        "variables": deepcopy(graph["variables"]),
        "local_groups": [
            {
                "group_id": group["group_id"],
                "group_type": group["group_type"],
                "variables": deepcopy(group["variables"]),
            }
            for group in graph["local_groups"]
        ],
        "dependency_edges": deepcopy(graph["dependency_edges"]),
    }


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one row without its self-referential checksum field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def build_rows(units: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create one hard and one easy negative for every frozen unit."""

    rows: list[JsonDict] = []
    for unit in units:
        hard_assignment, hard_receipt = _hard_negative(unit)
        first_group = unit["graph"]["local_groups"][0]
        easy_assignment = deepcopy(hard_assignment)
        for variable in first_group["variables"]:
            easy_assignment[str(variable)] = 1
        easy_receipt = evaluate_candidate(unit, easy_assignment)
        row_data = (
            (
                "hard_cross_dependency_failure",
                hard_assignment,
                hard_receipt,
                hard_receipt["failed_dependency_ids"][0],
                None,
            ),
            (
                "easy_local_failure",
                easy_assignment,
                easy_receipt,
                None,
                easy_receipt["failed_local_group_ids"][0],
            ),
        )
        for negative_class, assignment, receipt, broken_dependency, broken_group in row_data:
            row: JsonDict = {
                "schema": "carnot.experiment_6786.constraint_negative_row.v1",
                "row_id": f"{unit['unit_id']}|{negative_class}",
                "unit_id": unit["unit_id"],
                "graph_id": unit["graph_id"],
                "split": unit["split"],
                "topology_family": unit["topology_family"],
                "difficulty_stratum": unit["difficulty_stratum"],
                "negative_class": negative_class,
                "candidate_assignment": assignment,
                "proposal_features": proposal_features(unit),
                "named_broken_dependency": broken_dependency,
                "named_broken_local_group": broken_group,
                "exact_valid": receipt["exact_valid"],
                "exact_receipt": receipt,
                "source_panel_row_id": unit["provenance"]["source_panel_row_id"],
                "source_panel_row_sha256": unit["provenance"]["source_panel_row_sha256"],
                "row_sha256": "",
            }
            row["row_sha256"] = row_checksum(row)
            rows.append(row)
    return rows


def _walk_feature_keys(value: Any, prefix: str = "") -> list[str]:
    """Return each nested key path so denylist checks include child objects."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(path)
            paths.extend(_walk_feature_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_walk_feature_keys(child, f"{prefix}[{index}]"))
    return paths


def audit_feature_contract(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Report any extra top-level feature or denied key at any nesting depth."""

    violations: list[str] = []
    allowed = set(FEATURE_ALLOWLIST)
    denied = set(FEATURE_DENYLIST)
    for row in rows:
        row_id = str(row["row_id"])
        features = row["proposal_features"]
        for key in features:
            if key not in allowed or key in denied:
                violations.append(f"{row_id}.{key}")
        for path in _walk_feature_keys(features):
            leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
            violation = f"{row_id}.{path}"
            if leaf in denied and violation not in violations:
                violations.append(violation)
    return sorted(violations)


def summarize_splits(
    units: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Show that each whole topology family belongs to only one split."""

    summary: JsonDict = {}
    for split in ("train", "development", "held_topology_test"):
        split_units = [unit for unit in units if unit["split"] == split]
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = {
            "topology_families": sorted({unit["topology_family"] for unit in split_units}),
            "unit_ids": [unit["unit_id"] for unit in split_units],
            "unit_count": len(split_units),
            "negative_row_count": len(split_rows),
        }
    return summary


def exact_label_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count exact valid and invalid rows separately for every split."""

    return {
        split: {
            "valid": sum(row["exact_valid"] is True for row in rows if row["split"] == split),
            "invalid": sum(row["exact_valid"] is False for row in rows if row["split"] == split),
        }
        for split in ("train", "development", "held_topology_test")
    }


def _class_counts(rows: Sequence[Mapping[str, Any]], negative_class: str) -> JsonDict:
    """Count one negative class by its topology-held split."""

    return {
        split: sum(
            row["negative_class"] == negative_class and row["split"] == split for row in rows
        )
        for split in ("train", "development", "held_topology_test")
    }


def _row_label_signature(row: Mapping[str, Any]) -> JsonDict:
    """Keep only fields that cold replay must independently reproduce."""

    receipt = row["exact_receipt"]
    return {
        "exact_valid": row["exact_valid"],
        "local_checks_passed": receipt["local_checks_passed"],
        "failed_local_group_ids": receipt["failed_local_group_ids"],
        "failed_dependency_ids": receipt["failed_dependency_ids"],
    }


def replay_payload(
    units: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Recompute exact labels and failure classifications from candidate bytes."""

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    mismatches: list[str] = []
    for row in rows:
        replay_receipt = evaluate_candidate(
            unit_by_id[str(row["unit_id"])], row["candidate_assignment"]
        )
        replay_row = {
            "exact_valid": replay_receipt["exact_valid"],
            "exact_receipt": replay_receipt,
        }
        if _row_label_signature(replay_row) != _row_label_signature(row):
            mismatches.append(str(row["row_id"]))
    return {
        "agreement": not mismatches,
        "replayed_row_count": len(rows),
        "mismatches": mismatches,
        "rows_sha256": sha256_json([row["row_sha256"] for row in rows]),
    }


def run_cold_replay(
    units: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Start a fresh Python process and require a complete replay receipt."""

    payload = json.dumps({"units": units, "rows": rows}, sort_keys=True)
    environment = os.environ.copy()
    python_path = str(repo_root / "python")
    environment["PYTHONPATH"] = (
        python_path
        if not environment.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    process = subprocess.run(
        [sys.executable, "-m", __name__, "--cold-replay-worker"],
        input=payload,
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
            "fresh_process": receipt.get("cold_pid") != os.getpid(),
            "producer_pid": os.getpid(),
            "worker_exit_code": process.returncode,
        }
    )
    return receipt


def _manifest(units: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze serialization rules and every exact-enumerable unit."""

    manifest: JsonDict = {
        "schema": "carnot.experiment_6786.frozen_manifest.v1",
        "random_seed": RANDOM_SEED,
        "unit_count": len(units),
        "negative_classes": list(NEGATIVE_CLASSES),
        "graph_serialization": "canonical JSON with sorted object keys",
        "group_ordering": "ascending zero-padded group_id",
        "row_id_rule": "{unit_id}|{negative_class}",
        "difficulty_strata": list(DIFFICULTY_STRATA),
        "units": [deepcopy(dict(unit)) for unit in units],
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def _topology_family_metadata() -> list[JsonDict]:
    """Publish the three preregistered shape and split assignments."""

    return [
        {
            "family_id": spec["family_id"],
            "shape": spec["shape"],
            "split": spec["split"],
            "unit_count": UNIT_COUNT_PER_FAMILY,
        }
        for spec in TOPOLOGY_SPECS
    ]


def source_artifact_hashes(repo_root: Path, panel_path: Path) -> JsonDict:
    """Hash the frozen panel and both declarative group-builder modules."""

    paths = [panel_path, *(repo_root / path for path in SOURCE_BUILDER_RELATIVE_PATHS)]
    return {
        str(path.relative_to(repo_root) if path.is_relative_to(repo_root) else path): sha256_file(
            path
        )
        for path in paths
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable fixture facts while excluding duration and process IDs."""

    replay = artifact.get("cold_replay_agreement", {})
    stable_replay = {
        key: replay.get(key)
        for key in ("agreement", "replayed_row_count", "mismatches", "rows_sha256")
    }
    material = {
        "schema": artifact.get("schema"),
        "random_seed": artifact.get("random_seed"),
        "source_artifact_hashes": artifact.get("source_artifact_hashes"),
        "frozen_manifest": artifact.get("frozen_manifest"),
        "topology_families": artifact.get("topology_families"),
        "split_by_topology": artifact.get("split_by_topology"),
        "feature_allowlist": artifact.get("feature_allowlist"),
        "feature_denylist": artifact.get("feature_denylist"),
        "row_hashes": [row.get("row_sha256") for row in artifact.get("rows", [])],
        "exact_label_counts_by_split": artifact.get("exact_label_counts_by_split"),
        "local_pass_cross_dependency_fail_counts": artifact.get(
            "local_pass_cross_dependency_fail_counts"
        ),
        "easy_negative_counts": artifact.get("easy_negative_counts"),
        "duplicate_rows": artifact.get("duplicate_rows"),
        "future_feature_violations": artifact.get("future_feature_violations"),
        "cold_replay_agreement": stable_replay,
        "constraint_group_fixture_ready": artifact.get("constraint_group_fixture_ready"),
        "failed_checks": artifact.get("gate_check_summary", {}).get("failed_checks"),
    }
    return sha256_json(material)


def _empty_split_counts() -> JsonDict:
    """Return stable zero counts for a precondition-blocked artifact."""

    return {
        split: {"valid": 0, "invalid": 0}
        for split in ("train", "development", "held_topology_test")
    }


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    hashes: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
) -> JsonDict:
    """Build the complete terminal artifact required when any gate fails."""

    zero_class_counts = {split: 0 for split in ("train", "development", "held_topology_test")}
    first = first_failed_check(gate_summary)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_constraint_graph_fixture",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "frozen_manifest": {},
        "topology_families": _topology_family_metadata(),
        "split_by_topology": {},
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "rows": [],
        "exact_label_counts_by_split": _empty_split_counts(),
        "local_pass_cross_dependency_fail_counts": deepcopy(zero_class_counts),
        "easy_negative_counts": deepcopy(zero_class_counts),
        "duplicate_rows": 0,
        "future_feature_violations": [],
        "cold_replay_agreement": {
            "agreement": False,
            "replayed_row_count": 0,
            "mismatches": [],
            "rows_sha256": sha256_json([]),
            "fresh_process": False,
        },
        "constraint_group_fixture_ready": False,
        "gate_check_summary": deepcopy(dict(gate_summary)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": (
            "complete_blocked_constraint_graph_fixture: "
            f"{first['check']} observed {first['observed']}"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _validate_run_date(run_date: str) -> None:
    """Reject ambiguous dates before any artifact work starts."""

    if len(run_date) != 8 or not run_date.isdigit():
        raise ValueError("run date must use YYYYMMDD")


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    source_panel_path: Path | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Build a ready fixture, or a complete blocked artifact on failed authority."""

    _validate_run_date(run_date)
    started = time.monotonic()
    panel_path = source_panel_path or repo_root / SOURCE_PANEL_RELATIVE_PATH
    hashes = source_artifact_hashes(repo_root, panel_path)
    preconditions = evaluate_preconditions(
        repo_root=repo_root,
        source_panel_path=panel_path,
    )
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    if not preconditions["all_passed"]:
        artifact = _blocked_artifact(
            run_date=run_date,
            duration_s=measured,
            hashes=hashes,
            gate_summary=preconditions,
        )
        errors = validate_artifact(artifact)
        if errors:  # pragma: no cover - construction and validation share one contract.
            raise ValueError("; ".join(errors))
        return artifact

    panel = load_json_object(panel_path)
    units = build_units(panel)
    rows = build_rows(units)
    split_summary = summarize_splits(units, rows)
    label_counts = exact_label_counts(rows)
    hard_counts = _class_counts(rows, "hard_cross_dependency_failure")
    easy_counts = _class_counts(rows, "easy_local_failure")
    duplicates = len(rows) - len({row["row_id"] for row in rows})
    feature_violations = audit_feature_contract(rows)
    cold_replay = run_cold_replay(units, rows, repo_root=repo_root)
    family_sets = [set(value["topology_families"]) for value in split_summary.values()]
    split_disjoint = all(
        family_sets[left].isdisjoint(family_sets[right])
        for left in range(len(family_sets))
        for right in range(left + 1, len(family_sets))
    )
    fixture_checks = [
        *preconditions["checks"],
        _gate(
            "minimum_unique_units",
            ">=96",
            len({unit["unit_id"] for unit in units}),
            len(units) >= 96,
        ),
        _gate("unique_graph_identities", len(units), len({unit["graph_id"] for unit in units})),
        _gate("planned_negative_rows", ROW_COUNT, len(rows)),
        _gate("duplicate_rows", 0, duplicates),
        _gate("topology_split_disjoint", True, split_disjoint),
        _gate("hard_negative_count_each_split", [32, 32, 32], list(hard_counts.values())),
        _gate("easy_negative_count_each_split", [32, 32, 32], list(easy_counts.values())),
        _gate("future_feature_violations", [], feature_violations),
        _gate("cold_replay_agreement", True, cold_replay["agreement"]),
        _gate("cold_replay_row_count", ROW_COUNT, cold_replay["replayed_row_count"]),
    ]
    gate_summary = _summary(fixture_checks)
    measured = round(time.monotonic() - started, 6) if duration_s is None else duration_s
    if not gate_summary["all_passed"]:
        return _blocked_artifact(
            run_date=run_date,
            duration_s=measured,
            hashes=hashes,
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
        "source_artifact_hashes": hashes,
        "frozen_manifest": _manifest(units),
        "topology_families": _topology_family_metadata(),
        "split_by_topology": split_summary,
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "rows": rows,
        "exact_label_counts_by_split": label_counts,
        "local_pass_cross_dependency_fail_counts": hard_counts,
        "easy_negative_counts": easy_counts,
        "duplicate_rows": duplicates,
        "future_feature_violations": feature_violations,
        "cold_replay_agreement": cold_replay,
        "constraint_group_fixture_ready": True,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": (
            "complete: constraint dependency graph fixture ready with 96 units "
            "and 192 cold-replayed negative rows; this is not a model-quality claim"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - construction and validation share one contract.
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return every final contract error without changing artifact evidence."""

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
    ready = artifact.get("constraint_group_fixture_ready") is True
    if artifact.get("status") == "complete" and not ready:
        errors.append("ready artifact must set readiness true")
    if ready and artifact.get("status") != "complete":
        errors.append("ready artifact status mismatch")
    if ready and len(artifact.get("frozen_manifest", {}).get("units", [])) != UNIT_COUNT:
        errors.append("ready manifest must contain 96 units")
    if ready and len(artifact.get("rows", [])) != ROW_COUNT:
        errors.append("ready artifact must contain 192 rows")
    if ready and artifact.get("duplicate_rows") != 0:
        errors.append("ready artifact contains duplicate rows")
    if ready and artifact.get("future_feature_violations") != []:
        errors.append("ready artifact contains forbidden proposal features")
    if ready and artifact.get("cold_replay_agreement", {}).get("agreement") is not True:
        errors.append("ready artifact lacks cold replay agreement")
    if ready and artifact.get("gate_check_summary", {}).get("all_passed") is not True:
        errors.append("ready artifact has failed gates")
    if not ready and artifact.get("status") != "complete_blocked_constraint_graph_fixture":
        errors.append("not-ready artifact must use the complete blocked status")
    if not ready and artifact.get("rows") != []:
        errors.append("blocked artifact must not contain rows")
    if not ready and artifact.get("verdict_class") != "blocked":
        errors.append("blocked artifact must use blocked verdict class")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must remain false")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    return errors


def write_outputs(
    *,
    run_date: str = RUN_DATE,
    artifact_path: Path | str = RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
    source_panel_path: Path | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Write the validated terminal artifact with a same-directory replace."""

    artifact = build_artifact(
        run_date=run_date,
        repo_root=repo_root,
        source_panel_path=source_panel_path,
        duration_s=duration_s,
    )
    path = Path(artifact_path)
    if not path.is_absolute():
        path = repo_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return artifact


def _cold_replay_worker() -> int:
    """Read frozen data from standard input and emit one replay receipt."""

    payload = json.load(sys.stdin)
    receipt = replay_payload(payload["units"], payload["rows"])
    receipt["cold_pid"] = os.getpid()
    print(json.dumps(receipt, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by run command.
    """Parse the public run date and write the task-owned result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--cold-replay-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cold_replay_worker:
        return _cold_replay_worker()
    artifact = write_outputs(run_date=args.date, artifact_path=args.output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - module and repository entry points share main.
    raise SystemExit(main())
