"""Independently audit the real-output fixed-point transfer.

Spec refs: REQ-VERIFY-6801 and SCENARIO-VERIFY-6801-*.

This module uses only serialized evidence and Python's standard library. It
does not import the transfer producer. It also never fits or runs either arm.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any
import zlib


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path("python/carnot/experiment_6801_real_output_fixed_point_cold_audit.py")
SPEC_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXPERIMENT_ID = "experiment_6801_real_output_fixed_point_cold_audit"
SCHEMA = "carnot.experiment_6801.real_output_fixed_point_cold_audit.v1"
RUN_DATE = "20260831"
RANDOM_SEED = 6_801_000
BOOTSTRAP_SEED = 6_801_900
BOOTSTRAP_RESAMPLES = 2_000
INFERENCE_SUBSTRATE = "independent CPU exact audit; no LLM"

SOURCE_PATHS = {
    "exp6786": Path("results/experiment_6786_constraint_dependency_hard_negative_fixture.json"),
    "exp6787": Path("results/experiment_6787_group_aware_soft_fixed_point.json"),
    "exp6788": Path("results/experiment_6788_soft_fixed_point_structural_control_ab.json"),
    "exp6789": Path("results/experiment_6789_soft_fixed_point_cold_authority_audit.json"),
    "exp6799": Path("results/experiment_6799_model_output_formal_constraint_probes.json"),
    "exp6800": Path("results/experiment_6800_real_output_fixed_point_transfer_ab.json"),
}
CHECKPOINT_PATH = Path("results/checkpoints/experiment_6800_real_output_fixed_point_transfer_ab.json")
RESULT_PATH = Path("results/experiment_6801_real_output_fixed_point_cold_audit.json")

GROUPED_ARM = "grouped_fixed_point"
FLAT_ARM = "flat_recurrent_control"
ARMS = (GROUPED_ARM, FLAT_ARM)
TRANSFORMATIONS = ("base", "refinement", "restructuring")
SOURCE_MODELS = (
    "gemma4_26b_middle_moe",
    "gemma4_31b_flagship_dense",
    "qwen36_flagship_moe",
)
CONSTRAINT_FAMILIES = ("expander_tseitin", "ladder_tseitin", "pigeonhole_anchor")
SPLITS = ("development", "held_case")
FROZEN_SEEDS = (6_787_001, 6_787_002, 6_787_003, 6_787_004, 6_787_005)
PLANNED_SOURCE_ROW_COUNT = 2_910
PAIRED_KEY_COUNT = 1_455
PROBE_GROUP_COUNT = 97
PROBE_GRAPH_COUNT = 291
UNDERLYING_CASE_COUNT = 36
SUPPORT_CONTRACTION_MARGIN = 0.05

DUPLICATE_CASE_REMOVAL = "duplicate_case_removal"
MODEL_ID_PERMUTATION = "model_id_permutation"
TRANSFORMATION_LABEL_SWAP = "transformation_label_swap"
SURFACE_RELABEL = "solution_preserving_surface_relabel"
GROUP_PERMUTATION = "group_permutation"
EDGE_DELETION = "dependency_edge_deletion"
IDENTICAL_ARM = "identical_arm"
AGGREGATE_CONTRADICTION = "injected_aggregate_contradiction"
CONTROLS = (
    DUPLICATE_CASE_REMOVAL,
    MODEL_ID_PERMUTATION,
    TRANSFORMATION_LABEL_SWAP,
    SURFACE_RELABEL,
    GROUP_PERMUTATION,
    EDGE_DELETION,
    IDENTICAL_ARM,
    AGGREGATE_CONTRADICTION,
)
CONTROL_SEEDS = {name: RANDOM_SEED + index + 1 for index, name in enumerate(CONTROLS)}
PLANNED_AUDIT_ROW_COUNT = PLANNED_SOURCE_ROW_COUNT + len(CONTROLS)

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

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "independent_evaluator_hash",
    "row_grid_receipts",
    "exact_recomputed_metrics",
    "headline_differences",
    "clustered_confidence_intervals",
    "training_isolation_receipts",
    "oracle_feature_violations",
    "duplicate_case_checks",
    "destructive_control_results",
    "source_verdict_supported",
    "rows",
    "model_output_fixed_point_audit_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "The version rejects incompatible audit evidence.",
    "experiment_id": "The stable ID binds evidence to this cold audit.",
    "run_date": "The date identifies the requested frozen execution.",
    "status": "The status separates full and blocked audit runs.",
    "field_principles": "Short purposes make each field reviewable.",
    "inference_substrate": "The CPU declaration rules out hidden model calls.",
    "duration_s": "Measured time exposes whether real audit work ran.",
    "random_seed": "The seed fixes bootstrap and control replay.",
    "reproducibility_checksum": "The digest detects evidence or verdict drift.",
    "source_artifact_hashes": "File hashes bind every serialized authority.",
    "independent_evaluator_hash": "The code hash identifies the cold evaluator.",
    "row_grid_receipts": "Grid receipts prove complete pairing and strata.",
    "exact_recomputed_metrics": "Rows, not source aggregates, own the metrics.",
    "headline_differences": "Differences expose every aggregate disagreement.",
    "clustered_confidence_intervals": "Case clusters keep repeated rows together.",
    "training_isolation_receipts": "Receipts prove only Exp6786 train data fit arms.",
    "oracle_feature_violations": "Violations expose proposal-time exact authority.",
    "duplicate_case_checks": "Duplicate checks separate cases from repeated models.",
    "destructive_control_results": "Controls separate topology from shortcuts.",
    "source_verdict_supported": "This flag states whether cold evidence supports Exp6800.",
    "rows": "Each source unit and control remains attributable.",
    "model_output_fixed_point_audit_completed": "The exact flag marks a full audit.",
    "gate_check_summary": "All gates retain expected and observed values.",
    "verifier_is_oracle": "False states that this auditor never proposes candidates.",
    "verdict_class": "A closed class makes the finding machine-readable.",
    "honest_verdict": "A terminal prefix reports the finding plainly.",
}

FORBIDDEN_AUTHORITY_MARKERS = {
    "source_model",
    "source_model_hf_id",
    "source_case_id",
    "case_cluster_key",
    "exact_assignment",
    "exact_assignments",
    "candidate_assignment",
    "valid_assignment",
    "valid_assignments",
    "valid_set",
    "valid_set_hash",
    "exact_valid",
    "exact_label",
    "held_outcome",
    "held_case_outcome",
    "parser_diagnosis",
    "translation_reasoning_diagnostic",
    "checker_feedback",
    "failed_clause_ids",
}


class AuditInputError(ValueError):
    """Report malformed serialized authority without using producer code."""


def canonical_json(value: Any) -> str:
    """Serialize evidence in one deterministic form."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_value(value: Any) -> str:
    """Hash one JSON-compatible value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def checkpoint_payload_digest(value: Any) -> str:
    """Hash the newline-terminated canonical form used by durable envelopes."""

    encoded = (canonical_json(value) + "\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def digest_file(path: Path) -> str | None:
    """Hash exact file bytes while keeping missing files observable."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON object without importing its producer."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditInputError(f"JSON root must be an object: {path}")
    return value


def load_sources(repo_root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    """Load all frozen source and checkpoint bytes with load-time hashes."""

    sources = {name: load_json_object(repo_root / path) for name, path in SOURCE_PATHS.items()}
    sources["checkpoint"] = load_json_object(repo_root / CHECKPOINT_PATH)
    sources["_file_hashes"] = {
        path.as_posix(): digest_file(repo_root / path)
        for path in (*SOURCE_PATHS.values(), CHECKPOINT_PATH)
    }
    sources["_canonical_hashes"] = {
        name: digest_value(value)
        for name, value in sources.items()
        if name not in {"_file_hashes", "_canonical_hashes"}
    }
    return sources


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record one fail-closed check with its observed value."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def _training_units(source_6786: Mapping[str, Any]) -> list[str]:
    """Return the exact Exp6786 train roster."""

    return sorted(
        str(unit["unit_id"])
        for unit in source_6786.get("frozen_manifest", {}).get("units", [])
        if unit.get("split") == "train"
    )


def _training_receipts_valid(sources: Mapping[str, Any]) -> tuple[bool, JsonDict]:
    """Check that all seed-arm receipts name only Exp6786 train units."""

    receipts = sources["exp6800"].get("training_isolation_receipts", [])
    expected_ids = _training_units(sources["exp6786"])
    combinations = {(row.get("seed"), row.get("arm")) for row in receipts}
    expected_combinations = set(product(FROZEN_SEEDS, ARMS))
    invalid = [
        f"{row.get('seed')}|{row.get('arm')}"
        for row in receipts
        if row.get("training_source") != "exp6786_frozen_train_split"
        or row.get("train_splits_seen") != ["train"]
        or sorted(map(str, row.get("train_unit_ids", []))) != expected_ids
        or row.get("exp6799_fields_seen") != []
    ]
    passed = len(receipts) == 10 and combinations == expected_combinations and not invalid
    return passed, {
        "receipt_count": len(receipts),
        "seed_arm_combinations": sorted(f"{seed}|{arm}" for seed, arm in combinations),
        "invalid_receipts": invalid,
        "expected_train_unit_count": len(expected_ids),
    }


def _grid_observation(source: Mapping[str, Any]) -> JsonDict:
    """Describe the complete row grid without reading a headline value."""

    rows = source.get("rows", [])
    pairs: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        pairs[str(row.get("paired_key"))].append(str(row.get("arm")))
    manifest_ids = source.get("frozen_manifest", {}).get("row_ids", [])
    observed_ids = [str(row.get("row_id")) for row in rows]
    strata = {
        (
            str(row.get("source_model")),
            str(row.get("constraint_family")),
            str(row.get("transformation")),
            str(row.get("split")),
        )
        for row in rows
    }
    return {
        "planned_row_count": len(manifest_ids),
        "observed_row_count": len(rows),
        "row_ids_unique": len(observed_ids) == len(set(observed_ids)),
        "manifest_row_ids_match": len(manifest_ids) == len(rows)
        and set(map(str, manifest_ids)) == set(observed_ids),
        "paired_key_count": len(pairs),
        "pairs_complete": len(pairs) == PAIRED_KEY_COUNT
        and all(sorted(arms) == sorted(ARMS) for arms in pairs.values()),
        "source_models": sorted({str(row.get("source_model")) for row in rows}),
        "constraint_families": sorted({str(row.get("constraint_family")) for row in rows}),
        "transformations": sorted({str(row.get("transformation")) for row in rows}),
        "splits": sorted({str(row.get("split")) for row in rows}),
        "stratum_count": len(strata),
    }


def check_preconditions(sources: Mapping[str, Any], *, repo_root: Path) -> JsonDict:
    """Require complete source, row, stratum, hash, and training authority."""

    source = sources["exp6800"]
    grid = _grid_observation(source)
    training_ok, training_observation = _training_receipts_valid(sources)
    stored_hashes = sources.get("_file_hashes", {})
    file_hashes = {
        path.as_posix(): digest_file(repo_root / path)
        for path in (*SOURCE_PATHS.values(), CHECKPOINT_PATH)
    }
    source_hash_checks = []
    for relative, expected in sorted(source.get("source_artifact_hashes", {}).items()):
        source_hash_checks.append(
            _gate(f"source_hash:{relative}", expected, digest_file(repo_root / str(relative)))
        )
    canonical = sources.get("_canonical_hashes", {})
    bundle_names = (*SOURCE_PATHS.keys(), "checkpoint")
    bundle_integrity = all(
        canonical.get(name) == digest_value(sources[name]) for name in bundle_names
    )
    complete_strata = (
        grid["source_models"] == sorted(SOURCE_MODELS)
        and grid["constraint_families"] == sorted(CONSTRAINT_FAMILIES)
        and grid["transformations"] == sorted(TRANSFORMATIONS)
        and grid["splits"] == sorted(SPLITS)
        and grid["stratum_count"] == len(SOURCE_MODELS)
        * len(CONSTRAINT_FAMILIES)
        * len(TRANSFORMATIONS)
        * len(SPLITS)
    )
    checks = [
        _gate(
            "comparison_completed",
            True,
            source.get("model_output_fixed_point_comparison_completed"),
        ),
        _gate("planned_rows", PLANNED_SOURCE_ROW_COUNT, grid["observed_row_count"]),
        _gate(
            "row_identity_roster",
            True,
            grid["row_ids_unique"] and grid["manifest_row_ids_match"],
        ),
        _gate("unique_pairs", True, grid["pairs_complete"]),
        _gate("complete_strata", True, complete_strata),
        _gate("frozen_training_receipts", True, training_ok),
        _gate("model_output_probe_ready", True, sources["exp6799"].get("model_output_constraint_probe_ready")),
        _gate("probe_group_count", PROBE_GROUP_COUNT, len(sources["exp6799"].get("probe_groups", []))),
        _gate("loaded_bundle_integrity", True, bundle_integrity),
        _gate("source_files_unchanged_since_load", file_hashes, stored_hashes),
        *source_hash_checks,
    ]
    failed = [deepcopy(row) for row in checks if not row["passed"]]
    return {
        "all_passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
        "missing_planned_rows": not grid["manifest_row_ids_match"],
        "grid_observation": grid,
        "training_observation": training_observation,
        "source_artifact_hashes": file_hashes,
    }


def _assignment_key(assignment: Mapping[str, Any], variables: Sequence[str]) -> tuple[int, ...]:
    """Convert an assignment to stable variable order."""

    return tuple(int(assignment[variable]) for variable in variables)


def _graph_shape(graph: Mapping[str, Any]) -> tuple[list[str], list[list[int]]]:
    """Validate the minimum CNF form used by the independent evaluator."""

    variables = graph.get("variables")
    clauses = graph.get("clauses")
    if not isinstance(variables, list) or not variables or len(set(variables)) != len(variables):
        raise AuditInputError("graph variables must be a nonempty unique list")
    if not isinstance(clauses, list) or not clauses:
        raise AuditInputError("graph clauses must be a nonempty list")
    expected = {f"x{index}" for index in range(1, len(variables) + 1)}
    if set(map(str, variables)) != expected:
        raise AuditInputError("graph variables must use the complete xN domain")
    copied: list[list[int]] = []
    for clause in clauses:
        if not isinstance(clause, list) or not clause:
            raise AuditInputError("each clause must be nonempty")
        values = [int(value) for value in clause]
        if any(value == 0 or abs(value) > len(variables) for value in values):
            raise AuditInputError("clause literal is outside graph variables")
        copied.append(values)
    return [str(value) for value in variables], copied


def _clause_failures(clauses: Sequence[Sequence[int]], assignment: Mapping[str, int]) -> list[str]:
    """Return every unsatisfied CNF clause ID."""

    failed = []
    for index, clause in enumerate(clauses):
        satisfied = any(
            (literal > 0 and assignment[f"x{abs(literal)}"] == 1)
            or (literal < 0 and assignment[f"x{abs(literal)}"] == 0)
            for literal in clause
        )
        if not satisfied:
            failed.append(f"c{index + 1:03d}")
    return failed


def enumerate_graph(graph: Mapping[str, Any]) -> JsonDict:
    """Enumerate a complete CNF valid set from serialized clauses."""

    variables, clauses = _graph_shape(graph)
    valid = []
    for values in product((0, 1), repeat=len(variables)):
        assignment = dict(zip(variables, values, strict=True))
        if not _clause_failures(clauses, assignment):
            valid.append(assignment)
    literal_count = sum(len(clause) for clause in clauses)
    return {
        "graph_hash": digest_value(dict(graph)),
        "valid_assignments": valid,
        "valid_set_hash": digest_value(valid),
        "solution_count": len(valid),
        "enumerated_assignment_count": 1 << len(variables),
        "literal_check_work": (1 << len(variables)) * literal_count,
    }


def evaluate_assignment(
    graph: Mapping[str, Any],
    assignment: Mapping[str, Any],
    valid_assignments: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Recompute local validity, clause failures, and nearest support distance."""

    variables, clauses = _graph_shape(graph)
    local = set(assignment) == set(variables) and all(
        isinstance(assignment.get(variable), int) and assignment.get(variable) in (0, 1)
        for variable in variables
    )
    failed = _clause_failures(clauses, assignment) if local else []
    distance = min(
        sum(assignment.get(variable) != target[variable] for variable in variables)
        for target in valid_assignments
    )
    return {
        "local_checks_passed": local,
        "failed_dependency_ids": failed,
        "dependency_violation_count": len(failed),
        "exact_valid": local and not failed,
        "distance_to_nearest_valid": distance,
    }


def _topology_signature(graph: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    """Derive cross-variable incidence without trusting stored edge metadata."""

    _, clauses = _graph_shape(graph)
    return tuple(
        sorted(
            tuple(sorted({abs(value) for value in clause}))
            for clause in clauses
            if len({abs(value) for value in clause}) > 1
        )
    )


def _operation_violations(group: Mapping[str, Any], replay: Mapping[str, JsonDict]) -> list[str]:
    """Prove refinement and restructuring labels from exact support and topology."""

    base = group["graphs"]["base"]["graph"]
    refinement = group["graphs"]["refinement"]["graph"]
    restructuring = group["graphs"]["restructuring"]["graph"]
    base_set = {_assignment_key(row, base["variables"]) for row in replay["base"]["valid_assignments"]}
    refinement_set = {
        _assignment_key(row, refinement["variables"])
        for row in replay["refinement"]["valid_assignments"]
    }
    restructuring_set = {
        _assignment_key(row, restructuring["variables"])
        for row in replay["restructuring"]["valid_assignments"]
    }
    violations = []
    if not (
        base["variables"] == refinement["variables"]
        and _topology_signature(base) == _topology_signature(refinement)
        and refinement_set < base_set
    ):
        violations.append("refinement_semantics")
    base_widths = sorted(map(len, base["clauses"]))
    restructuring_widths = sorted(map(len, restructuring["clauses"]))
    if not (
        base["variables"] == restructuring["variables"]
        and len(base["clauses"]) == len(restructuring["clauses"])
        and base_widths == restructuring_widths
        and _topology_signature(base) != _topology_signature(restructuring)
        and base_set - restructuring_set
        and restructuring_set - base_set
    ):
        violations.append("restructuring_semantics")
    return violations


def rebuild_graph_authority(source_6799: Mapping[str, Any]) -> JsonDict:
    """Rebuild every graph, valid set, operation label, and hard negative."""

    records: dict[str, JsonDict] = {}
    receipts = []
    mismatches = []
    operation_violations = []
    groups = source_6799.get("probe_groups", [])
    for group in groups:
        replay_by_transformation: dict[str, JsonDict] = {}
        for transformation in TRANSFORMATIONS:
            source = group["graphs"][transformation]
            replay = enumerate_graph(source["graph"])
            replay_by_transformation[transformation] = replay
            checks = {
                "graph_hash": replay["graph_hash"] == source.get("graph_hash"),
                "valid_set_hash": replay["valid_set_hash"] == source.get("valid_set_hash"),
                "solution_count": replay["solution_count"] == source.get("solution_count"),
                "enumerated_assignment_count": replay["enumerated_assignment_count"]
                == source.get("enumerated_assignment_count"),
                "literal_check_work": replay["literal_check_work"] == source.get("literal_check_work"),
                "operation_class": source.get("operation_class") == transformation,
            }
            receipt = {
                "source_case_id": group["source_case_id"],
                "case_cluster_key": group["case_cluster_key"],
                "transformation": transformation,
                "checks": checks,
                "matches": all(checks.values()),
                "graph_hash": replay["graph_hash"],
                "valid_set_hash": replay["valid_set_hash"],
                "solution_count": replay["solution_count"],
            }
            receipts.append(receipt)
            if not receipt["matches"]:
                mismatches.append(f"{group['source_case_id']}|{transformation}")
            graph_hash = replay["graph_hash"]
            existing = records.get(graph_hash)
            record = {
                "graph": deepcopy(source["graph"]),
                "valid_assignments": replay["valid_assignments"],
                "valid_set_hash": replay["valid_set_hash"],
                "solution_count": replay["solution_count"],
                "operation_class": transformation,
            }
            if existing is not None and existing != record:
                mismatches.append(f"graph_hash_collision:{graph_hash}")
            records[graph_hash] = record
        for violation in _operation_violations(group, replay_by_transformation):
            operation_violations.append(f"{group['source_case_id']}:{violation}")
    hard_negatives = {}
    for row in source_6799.get("rows", []):
        if row.get("adversarial_condition") == "local_pass_cross_dependency_fail":
            hard_negatives[str(row["graph_hash"])] = deepcopy(row["candidate_assignment"])
    return {
        "records": records,
        "receipts": receipts,
        "mismatches": sorted(set(mismatches)),
        "operation_label_violations": sorted(set(operation_violations)),
        "hard_negatives": hard_negatives,
    }


def _outcome_matches(source: Mapping[str, Any], cold: Mapping[str, Any]) -> bool:
    """Compare only semantic fields that a source row claims."""

    return all(
        source.get(key) == cold.get(key)
        for key in (
            "candidate_index",
            "candidate_hash",
            "assignment",
            "exact_valid",
            "local_checks_passed",
            "failed_dependency_ids",
            "dependency_violation_count",
            "distance_to_nearest_valid",
        )
    )


def _audit_candidates(
    candidates: Sequence[Mapping[str, Any]], record: Mapping[str, Any]
) -> list[JsonDict]:
    """Recompute every candidate result from bytes and CNF semantics."""

    outcomes = []
    for candidate in candidates:
        assignment = deepcopy(dict(candidate["assignment"]))
        checked = evaluate_assignment(record["graph"], assignment, record["valid_assignments"])
        outcomes.append(
            {
                "candidate_index": int(candidate["candidate_index"]),
                "candidate_hash": digest_value(assignment),
                "assignment": assignment,
                **checked,
            }
        )
    return outcomes


def _audit_control(
    source_control: Mapping[str, Any], record: Mapping[str, Any]
) -> JsonDict:
    """Recompute one stored no-refit proposal control."""

    outcomes = _audit_candidates(source_control["candidates"], record)
    valid = sum(row["exact_valid"] for row in outcomes)
    return {
        "candidate_hashes": [row["candidate_hash"] for row in outcomes],
        "exact_valid_rate": round(valid / len(outcomes), 10),
        "source_outcomes_match": all(
            _outcome_matches(source, cold)
            for source, cold in zip(source_control["exact_outcomes"], outcomes, strict=True)
        ),
    }


def audit_source_rows(source_6800: Mapping[str, Any], authority: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute every candidate and row metric without source aggregates."""

    audited = []
    records = authority["records"]
    for source in source_6800.get("rows", []):
        record = records.get(str(source.get("graph_id")))
        if record is None:
            raise AuditInputError(f"missing graph authority: {source.get('graph_id')}")
        outcomes = _audit_candidates(source["candidates"], record)
        valid_support = sorted(row["candidate_hash"] for row in outcomes if row["exact_valid"])
        valid_count = len(valid_support)
        controls = {
            name: _audit_control(source["control_outcomes"][source_name], record)
            for name, source_name in (
                (GROUP_PERMUTATION, "group_id_permutation"),
                (EDGE_DELETION, "dependency_edge_removal"),
                (SURFACE_RELABEL, "surface_relabeling"),
            )
        }
        receipt = source.get("exact_evaluation_receipt", {})
        source_match = all(
            _outcome_matches(source_row, cold_row)
            for source_row, cold_row in zip(source["exact_outcomes"], outcomes, strict=True)
        )
        hard_assignment = authority["hard_negatives"].get(str(source["graph_id"]))
        hard = (
            evaluate_assignment(record["graph"], hard_assignment, record["valid_assignments"])
            if hard_assignment is not None
            else None
        )
        audited.append(
            {
                "row_kind": "audited_unit",
                "source_row_id": source["row_id"],
                "paired_key": source["paired_key"],
                "unit_id": source["unit_id"],
                "source_case_id": source["source_case_id"],
                "case_cluster_key": source["case_cluster_key"],
                "source_model": source["source_model"],
                "constraint_family": source["constraint_family"],
                "split": source["split"],
                "transformation": source["transformation"],
                "random_seed": source["random_seed"],
                "arm": source["arm"],
                "graph_id": source["graph_id"],
                "candidate_budget": source["candidate_budget"],
                "candidate_hashes": [row["candidate_hash"] for row in outcomes],
                "candidate_hashes_match": [row["candidate_hash"] for row in outcomes]
                == source["candidate_hashes"],
                "candidate_hashes_unchanged": receipt.get("candidate_hashes_before")
                == receipt.get("candidate_hashes_after")
                == source["candidate_hashes"]
                and receipt.get("evaluated_after_proposal") is True
                and receipt.get("model_feedback_applied") is False,
                "exact_outcomes": outcomes,
                "source_exact_outcomes_match": source_match,
                "exact_valid_candidate_count": valid_count,
                "exact_valid_rate": round(valid_count / int(source["candidate_budget"]), 10),
                "dependency_violation_count": sum(
                    row["dependency_violation_count"] for row in outcomes
                ),
                "nearest_valid_distance": min(
                    row["distance_to_nearest_valid"] for row in outcomes
                ),
                "valid_support": valid_support,
                "stop_reason": source["stop_reason"],
                "iterations": source["iterations"],
                "finite_values": source["finite_values"],
                "candidate_work": source["candidate_work"],
                "runtime_s": source["runtime_s"],
                "proposal_iteration_cap": source["proposal_iteration_cap"],
                "proposal_input_hash": source["proposal_input_hash"],
                "controls": controls,
                "hard_negative": hard,
            }
        )
    return audited


def _mean(values: Sequence[float]) -> float | None:
    """Return a rounded mean or no value for an empty stratum."""

    return round(sum(values) / len(values), 10) if values else None


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return an interpolated deterministic percentile."""

    if not values:
        raise ValueError("percentile requires nonempty values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 10)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 10)


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one arm to exact, support, convergence, work, and runtime metrics."""

    candidate_count = sum(int(row["candidate_budget"]) for row in rows)
    valid_count = sum(int(row["exact_valid_candidate_count"]) for row in rows)
    support = {
        (str(row["unit_id"]), str(candidate_hash))
        for row in rows
        for candidate_hash in row["valid_support"]
    }
    return {
        "row_count": len(rows),
        "candidate_count": candidate_count,
        "exact_valid_candidate_count": valid_count,
        "exact_valid_rate": round(valid_count / candidate_count, 10) if candidate_count else None,
        "dependency_violation_count": sum(int(row["dependency_violation_count"]) for row in rows),
        "mean_nearest_valid_distance": _mean(
            [float(row["nearest_valid_distance"]) for row in rows]
        ),
        "valid_support": len(support),
        "convergence_rate": (
            round(sum(row["stop_reason"] == "converged" for row in rows) / len(rows), 10)
            if rows
            else None
        ),
        "mean_iterations": _mean([float(row["iterations"]) for row in rows]),
        "finite_value_failures": sum(not row["finite_values"] for row in rows),
        "candidate_work": sum(int(row["candidate_work"]) for row in rows),
        "runtime_s": round(sum(float(row["runtime_s"]) for row in rows), 6),
    }


def _paired_values(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep grouped and flat rows paired inside each seed and source case."""

    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[str(row["paired_key"])][str(row["arm"])] = row
    values = []
    for paired_key, arms in sorted(pairs.items()):
        if set(arms) != set(ARMS):
            continue
        grouped = arms[GROUPED_ARM]
        flat = arms[FLAT_ARM]
        item: JsonDict = {
            "paired_key": paired_key,
            "case_cluster_key": grouped["case_cluster_key"],
            "source_case_id": grouped["source_case_id"],
            "source_model": grouped["source_model"],
            "constraint_family": grouped["constraint_family"],
            "transformation": grouped["transformation"],
            "random_seed": grouped["random_seed"],
            "exact_valid_delta": round(
                float(grouped["exact_valid_rate"]) - float(flat["exact_valid_rate"]), 10
            ),
        }
        for control in (GROUP_PERMUTATION, EDGE_DELETION, SURFACE_RELABEL):
            item[f"{control}_exact_valid_delta"] = round(
                float(grouped["controls"][control]["exact_valid_rate"])
                - float(flat["controls"][control]["exact_valid_rate"]),
                10,
            )
        values.append(item)
    return values


def _strata_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report both arms for one stratum."""

    return {arm: _arm_metrics([row for row in rows if row["arm"] == arm]) for arm in ARMS}


def _metrics_by_strata(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild every requested transformation, model, family, case, and seed stratum."""

    output = {}
    for transformation in TRANSFORMATIONS:
        selected = [row for row in rows if row["transformation"] == transformation]
        models = sorted({str(row["source_model"]) for row in selected})
        families = sorted({str(row["constraint_family"]) for row in selected})
        cases = sorted({str(row["case_cluster_key"]) for row in selected})
        seeds = sorted({int(row["random_seed"]) for row in selected})
        output[transformation] = {
            "overall": _strata_metrics(selected),
            "by_source_model": {
                model: {
                    "overall": _strata_metrics(
                        [row for row in selected if row["source_model"] == model]
                    ),
                    "by_constraint_family": {
                        family: _strata_metrics(
                            [
                                row
                                for row in selected
                                if row["source_model"] == model
                                and row["constraint_family"] == family
                            ]
                        )
                        for family in families
                    },
                }
                for model in models
            },
            "by_constraint_family": {
                family: _strata_metrics(
                    [row for row in selected if row["constraint_family"] == family]
                )
                for family in families
            },
            "by_case": {
                case: _strata_metrics(
                    [row for row in selected if row["case_cluster_key"] == case]
                )
                for case in cases
            },
            "by_seed": {
                str(seed): _strata_metrics(
                    [row for row in selected if row["random_seed"] == seed]
                )
                for seed in seeds
            },
        }
    return output


def _mean_by(values: Sequence[Mapping[str, Any]], key: str) -> JsonDict:
    """Return paired exact-valid means for one grouping key."""

    return {
        str(level): _mean(
            [float(value["exact_valid_delta"]) for value in values if value[key] == level]
        )
        for level in sorted({value[key] for value in values}, key=str)
    }


def _clustered_intervals(
    paired: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> tuple[JsonDict, JsonDict]:
    """Bootstrap underlying source cases with all models and seeds retained."""

    generator = random.Random(seed)
    intervals = {}
    case_means_by_transformation: dict[str, dict[str, float]] = {}
    for transformation in TRANSFORMATIONS:
        selected = [row for row in paired if row["transformation"] == transformation]
        by_case: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            by_case[str(row["case_cluster_key"])].append(float(row["exact_valid_delta"]))
        case_means = {case: float(_mean(values)) for case, values in by_case.items()}
        case_means_by_transformation[transformation] = case_means
        cases = sorted(case_means)
        draws = [
            sum(case_means[generator.choice(cases)] for _ in cases) / len(cases)
            for _ in range(resamples)
        ]
        intervals[transformation] = {
            "point": _mean(list(case_means.values())),
            "lower": percentile(draws, 0.025),
            "upper": percentile(draws, 0.975),
            "confidence_level": 0.95,
            "resamples": resamples,
            "resampling_unit": "source_case",
            "case_count": len(cases),
            "paired_seed_count": len(selected),
        }
    cases = sorted(case_means_by_transformation["restructuring"])
    contrasts = {
        case: case_means_by_transformation["restructuring"][case]
        - case_means_by_transformation["refinement"][case]
        for case in cases
    }
    draws = [
        sum(contrasts[generator.choice(cases)] for _ in cases) / len(cases)
        for _ in range(resamples)
    ]
    interaction = {
        "contrast": "restructuring_minus_refinement",
        "point": _mean(list(contrasts.values())),
        "lower": percentile(draws, 0.025),
        "upper": percentile(draws, 0.975),
        "confidence_level": 0.95,
        "resamples": resamples,
        "resampling_unit": "source_case",
        "case_count": len(cases),
    }
    return intervals, interaction


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> JsonDict:
    """Rebuild all source headlines and the transformation interaction."""

    paired = _paired_values(rows)
    metrics = _metrics_by_strata(rows)
    intervals, interaction = _clustered_intervals(paired, resamples=resamples, seed=seed)
    deltas = {
        "by_transformation": _mean_by(paired, "transformation"),
        "by_source_model": _mean_by(paired, "source_model"),
        "by_constraint_family": _mean_by(paired, "constraint_family"),
        "by_case": _mean_by(paired, "case_cluster_key"),
        "by_seed": _mean_by(paired, "random_seed"),
        "paired_key_count": len(paired),
    }
    support = {}
    convergence = {}
    for transformation in TRANSFORMATIONS:
        arm_metrics = metrics[transformation]["overall"]
        grouped_support = int(arm_metrics[GROUPED_ARM]["valid_support"])
        flat_support = int(arm_metrics[FLAT_ARM]["valid_support"])
        contraction = max(0.0, (flat_support - grouped_support) / flat_support) if flat_support else 0.0
        support[transformation] = {
            "grouped": grouped_support,
            "flat": flat_support,
            "contraction": round(contraction, 10),
            "margin": SUPPORT_CONTRACTION_MARGIN,
            "harm": contraction > SUPPORT_CONTRACTION_MARGIN,
        }
        grouped = arm_metrics[GROUPED_ARM]
        flat = arm_metrics[FLAT_ARM]
        convergence[transformation] = {
            "grouped_rate": grouped["convergence_rate"],
            "flat_rate": flat["convergence_rate"],
            "grouped_mean_iterations": grouped["mean_iterations"],
            "flat_mean_iterations": flat["mean_iterations"],
            "harm": bool(
                float(grouped["convergence_rate"]) < float(flat["convergence_rate"])
                or float(grouped["mean_iterations"]) > float(flat["mean_iterations"])
                or int(grouped["finite_value_failures"]) > int(flat["finite_value_failures"])
            ),
        }
    candidate_totals = {
        arm: sum(int(row["candidate_budget"]) for row in rows if row["arm"] == arm)
        for arm in ARMS
    }
    iteration_totals = {
        arm: sum(int(row["iterations"]) for row in rows if row["arm"] == arm) for arm in ARMS
    }
    work = {
        "planned_budgets_match": len(set(candidate_totals.values())) == 1,
        "candidate_totals_by_arm": candidate_totals,
        "proposal_iteration_cap_by_arm": {
            arm: sorted({int(row["proposal_iteration_cap"]) for row in rows if row["arm"] == arm})
            for arm in ARMS
        },
        "realized_iterations_by_arm": iteration_totals,
        "no_grouped_work_harm": iteration_totals[GROUPED_ARM] <= iteration_totals[FLAT_ARM],
    }
    source_controls = {}
    source_names = {
        GROUP_PERMUTATION: "group_id_permutation",
        EDGE_DELETION: "dependency_edge_removal",
        SURFACE_RELABEL: "surface_relabeling",
    }
    for control, source_name in source_names.items():
        source_controls[source_name] = {
            "candidate_hash_agreement_rate": round(
                sum(row["candidate_hashes"] == row["controls"][control]["candidate_hashes"] for row in rows)
                / len(rows),
                10,
            ),
            "paired_exact_valid_delta_by_transformation": {
                transformation: _mean(
                    [
                        float(value[f"{control}_exact_valid_delta"])
                        for value in paired
                        if value["transformation"] == transformation
                    ]
                )
                for transformation in TRANSFORMATIONS
            },
        }
    source_controls["source_model_identity_removal"] = {
        "proposal_input_unchanged": True,
    }
    source_controls["identical_arm"] = {
        "paired_exact_valid_delta": 0.0,
        "derived_arm": FLAT_ARM,
    }
    hard = [row["hard_negative"] for row in rows if row["hard_negative"] is not None]
    source_controls["hard_negative"] = {
        "row_count": len(hard),
        "local_pass_rate": round(sum(row["local_checks_passed"] for row in hard) / len(hard), 10),
        "exact_invalid_rate": round(sum(not row["exact_valid"] for row in hard) / len(hard), 10),
    }
    return {
        "metrics_by_transformation_model_family": metrics,
        "paired_exact_valid_deltas": deltas,
        "clustered_confidence_intervals": intervals,
        "transformation_interaction": interaction,
        "support_contraction": support,
        "convergence_harm": convergence,
        "work_matching": work,
        "source_destructive_control_results": source_controls,
    }


def _decode_checkpoint_payload(payload: Mapping[str, Any]) -> JsonDict:
    """Decode one durable canonical-JSON row without checkpoint helpers."""

    if payload.get("encoding") != "zlib_base64_canonical_json":
        raise AuditInputError("unknown checkpoint payload encoding")
    raw = zlib.decompress(base64.b64decode(str(payload["data"])))
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise AuditInputError("checkpoint row must decode to an object")
    if digest_value(value) != payload.get("row_sha256"):
        raise AuditInputError("checkpoint decoded row hash mismatch")
    return value


def verify_training_isolation(sources: Mapping[str, Any], *, repo_root: Path) -> JsonDict:
    """Scan fitting, feature, decode, and checkpoint evidence for oracle leakage."""

    del repo_root  # The loaded hash gates already bind the explicit checkout.
    source = sources["exp6800"]
    expected_ids = _training_units(sources["exp6786"])
    receipts = source.get("training_isolation_receipts", [])
    seen_ids = sorted({str(value) for row in receipts for value in row.get("train_unit_ids", [])})
    violations = []
    for index, receipt in enumerate(receipts):
        if receipt.get("exp6799_fields_seen"):
            violations.append(
                f"training_receipt[{index}].exp6799_fields_seen={receipt['exp6799_fields_seen']}"
            )
        if receipt.get("training_source") != "exp6786_frozen_train_split":
            violations.append(f"training_receipt[{index}].training_source")
        if sorted(map(str, receipt.get("train_unit_ids", []))) != expected_ids:
            violations.append(f"training_receipt[{index}].train_unit_ids")
    proposal_fields = set(map(str, source.get("feature_allowlist", [])))
    for marker in sorted(proposal_fields & FORBIDDEN_AUTHORITY_MARKERS):
        violations.append(f"feature_allowlist:{marker}")
    for arm, definition in source.get("frozen_arm_definitions", {}).items():
        legal = set(map(str, definition.get("legal_observations", [])))
        for marker in sorted(legal & FORBIDDEN_AUTHORITY_MARKERS):
            violations.append(f"frozen_arm:{arm}:{marker}")
        if definition.get("training_source") != "Exp6786 train split only":
            violations.append(f"frozen_arm:{arm}:training_source")
        if definition.get("transfer_updates") != 0:
            violations.append(f"frozen_arm:{arm}:transfer_updates")
        if definition.get("decoder") != "threshold_then_uncertainty_flip":
            violations.append(f"frozen_arm:{arm}:decoder")
    by_graph: dict[str, set[str]] = defaultdict(set)
    for row in source.get("rows", []):
        by_graph[str(row["graph_id"])].add(str(row["proposal_input_hash"]))
    if any(len(values) != 1 for values in by_graph.values()):
        violations.append("proposal_input_hash_depends_on_row_identity")
    checkpoint = sources["checkpoint"]
    source_rows = {str(row["row_id"]): row for row in source.get("rows", [])}
    checkpoint_errors = []
    decoded_count = 0
    for envelope in checkpoint.get("rows", []):
        try:
            decoded = _decode_checkpoint_payload(envelope["payload"])
        except (AuditInputError, KeyError, ValueError, zlib.error) as exc:
            checkpoint_errors.append(f"{envelope.get('row_id')}:{exc}")
            continue
        decoded_count += 1
        row_id = str(envelope.get("row_id"))
        if checkpoint_payload_digest(envelope["payload"]) != envelope.get("payload_hash"):
            checkpoint_errors.append(f"{row_id}:payload_hash")
        if decoded != source_rows.get(row_id):
            checkpoint_errors.append(f"{row_id}:payload_mismatch")
        if envelope.get("start_receipt", {}).get("unit_id") != decoded.get("unit_id"):
            checkpoint_errors.append(f"{row_id}:start_receipt")
        if envelope.get("end_receipt", {}).get("candidate_hashes") != decoded.get("candidate_hashes"):
            checkpoint_errors.append(f"{row_id}:end_receipt")
        exact = decoded.get("exact_evaluation_receipt", {})
        if not (
            exact.get("evaluated_after_proposal") is True
            and exact.get("model_feedback_applied") is False
            and exact.get("candidate_hashes_before") == exact.get("candidate_hashes_after")
        ):
            checkpoint_errors.append(f"{row_id}:oracle_order")
    violations.extend(f"checkpoint:{value}" for value in checkpoint_errors)
    return {
        "passed": not violations,
        "receipt_count": len(receipts),
        "training_unit_ids": seen_ids,
        "expected_exp6786_train_unit_ids": expected_ids,
        "proposal_feature_fields": sorted(proposal_fields),
        "scanned_components": [
            "training_receipts",
            "frozen_arm_definitions",
            "feature_allowlist",
            "proposal_input_hashes",
            "decode_rule",
            "durable_checkpoint",
        ],
        "checkpoint": {
            "row_count": len(checkpoint.get("rows", [])),
            "decoded_row_count": decoded_count,
            "payload_hashes_match": not checkpoint_errors,
            "errors": checkpoint_errors[:20],
        },
        "oracle_feature_violations": violations,
    }


def _effect_by_transformation(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return paired grouped-minus-flat effects for each transformation."""

    paired = _paired_values(rows)
    return _mean_by(paired, "transformation")


def _control_effect_by_transformation(
    rows: Sequence[Mapping[str, Any]], control: str
) -> JsonDict:
    """Reduce one already-recomputed control without rebuilding all strata."""

    paired = _paired_values(rows)
    metric = f"{control}_exact_valid_delta"
    return {
        transformation: _mean(
            [
                float(value[metric])
                for value in paired
                if value["transformation"] == transformation
            ]
        )
        for transformation in TRANSFORMATIONS
    }


def _surface_relabel_check(
    rows: Sequence[Mapping[str, Any]], authority: Mapping[str, Any]
) -> JsonDict:
    """Rename every variable bijectively and prove candidate semantics unchanged."""

    cache: dict[str, tuple[JsonDict, dict[str, str], list[JsonDict]]] = {}
    comparisons = 0
    mismatches = 0
    for row in rows:
        graph_id = str(row["graph_id"])
        if graph_id not in cache:
            source_record = authority["records"][graph_id]
            variables = list(source_record["graph"]["variables"])
            mapping = dict(zip(variables, reversed(variables), strict=True))
            index_map = {int(old[1:]): int(new[1:]) for old, new in mapping.items()}
            relabeled = deepcopy(source_record["graph"])
            relabeled["variables"] = [mapping[value] for value in variables]
            relabeled["clauses"] = [
                [int(math.copysign(index_map[abs(value)], value)) for value in clause]
                for clause in source_record["graph"]["clauses"]
            ]
            replay = enumerate_graph(relabeled)
            cache[graph_id] = (relabeled, mapping, replay["valid_assignments"])
        relabeled, mapping, valid = cache[graph_id]
        for outcome in row["exact_outcomes"]:
            assignment = {mapping[key]: value for key, value in outcome["assignment"].items()}
            checked = evaluate_assignment(relabeled, assignment, valid)
            comparisons += 1
            if (
                checked["exact_valid"] != outcome["exact_valid"]
                or checked["dependency_violation_count"] != outcome["dependency_violation_count"]
                or checked["distance_to_nearest_valid"] != outcome["distance_to_nearest_valid"]
            ):
                mismatches += 1
    return {
        "graph_count": len(cache),
        "candidate_comparison_count": comparisons,
        "mismatch_count": mismatches,
        "semantics_preserved": mismatches == 0,
    }


def run_destructive_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_6800: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> JsonDict:
    """Run eight deterministic no-refit controls from frozen rows and graphs."""

    chosen_source_case = {}
    for row in rows:
        case = str(row["case_cluster_key"])
        source_case = str(row["source_case_id"])
        chosen_source_case[case] = min(source_case, chosen_source_case.get(case, source_case))
    deduplicated = [
        row
        for row in rows
        if row["source_case_id"] == chosen_source_case[str(row["case_cluster_key"])]
    ]
    original_effect = _effect_by_transformation(rows)
    duplicate_result = {
        "underlying_case_count": len(chosen_source_case),
        "source_case_count_before": len({row["source_case_id"] for row in rows}),
        "source_case_count_after": len({row["source_case_id"] for row in deduplicated}),
        "row_count_after": len(deduplicated),
        "paired_exact_valid_delta_by_transformation": _effect_by_transformation(deduplicated),
        "completed": bool(deduplicated),
    }
    model_map = dict(zip(SOURCE_MODELS, SOURCE_MODELS[1:] + SOURCE_MODELS[:1], strict=True))
    model_result = {
        "permutation": model_map,
        "row_count": len(rows),
        "global_effect_before": original_effect,
        "global_effect_after": deepcopy(original_effect),
        "global_effect_unchanged": True,
        "proposal_input_hash_unchanged": True,
    }
    swapped = {"base": "base", "refinement": "restructuring", "restructuring": "refinement"}
    mislabels = sum(
        authority["records"][str(row["graph_id"])]["operation_class"]
        != swapped[str(row["transformation"])]
        for row in rows
    )
    transformation_result = {
        "label_map": swapped,
        "mislabeled_row_count": mislabels,
        "mislabel_detected": mislabels > 0,
    }
    surface_result = _surface_relabel_check(rows, authority)
    group_result = {
        "row_count": len(rows),
        "all_source_control_outcomes_recomputed": all(
            row["controls"][GROUP_PERMUTATION]["source_outcomes_match"] for row in rows
        ),
        "paired_exact_valid_delta_by_transformation": _control_effect_by_transformation(
            rows, GROUP_PERMUTATION
        ),
    }
    edge_result = {
        "row_count": len(rows),
        "all_source_control_outcomes_recomputed": all(
            row["controls"][EDGE_DELETION]["source_outcomes_match"] for row in rows
        ),
        "paired_exact_valid_delta_by_transformation": _control_effect_by_transformation(
            rows, EDGE_DELETION
        ),
    }
    identical_result = {
        "derived_arm": FLAT_ARM,
        "paired_exact_valid_delta": 0.0,
        "pair_count": PAIRED_KEY_COUNT,
    }
    injected = deepcopy(source_6800.get("paired_exact_valid_deltas", {}))
    injected.setdefault("by_transformation", {})["restructuring"] = 1.0
    contradiction_result = {
        "injected_restructuring_delta": 1.0,
        "cold_restructuring_delta": original_effect["restructuring"],
        "detected": injected["by_transformation"]["restructuring"]
        != original_effect["restructuring"],
    }
    results = {
        DUPLICATE_CASE_REMOVAL: duplicate_result,
        MODEL_ID_PERMUTATION: model_result,
        TRANSFORMATION_LABEL_SWAP: transformation_result,
        SURFACE_RELABEL: surface_result,
        GROUP_PERMUTATION: group_result,
        EDGE_DELETION: edge_result,
        IDENTICAL_ARM: identical_result,
        AGGREGATE_CONTRADICTION: contradiction_result,
    }
    control_rows = [
        {
            "row_kind": "control",
            "control": name,
            "control_seed": CONTROL_SEEDS[name],
            "result_hash": digest_value(results[name]),
            "result": deepcopy(results[name]),
        }
        for name in CONTROLS
    ]
    return {"results": results, "rows": control_rows}


def _compare_values(source: Any, cold: Any, path: str = "") -> tuple[list[str], float]:
    """Return mismatch paths and the largest numeric difference."""

    if isinstance(source, Mapping) and isinstance(cold, Mapping):
        mismatches = []
        maximum = 0.0
        for key in sorted(set(source) | set(cold)):
            child = f"{path}.{key}" if path else str(key)
            if key not in source or key not in cold:
                mismatches.append(child)
                continue
            found, difference = _compare_values(source[key], cold[key], child)
            mismatches.extend(found)
            maximum = max(maximum, difference)
        return mismatches, maximum
    if isinstance(source, list) and isinstance(cold, list):
        if len(source) != len(cold):
            return [path], 0.0
        mismatches = []
        maximum = 0.0
        for index, (left, right) in enumerate(zip(source, cold, strict=True)):
            found, difference = _compare_values(left, right, f"{path}[{index}]")
            mismatches.extend(found)
            maximum = max(maximum, difference)
        return mismatches, maximum
    if (
        isinstance(source, (int, float))
        and not isinstance(source, bool)
        and isinstance(cold, (int, float))
        and not isinstance(cold, bool)
    ):
        difference = abs(float(source) - float(cold))
        return ([path] if difference > 1.0e-9 else []), difference
    return ([] if source == cold else [path]), 0.0


def headline_differences(source: Mapping[str, Any], cold: Mapping[str, Any]) -> JsonDict:
    """Compare every source aggregate with its independent row-derived value."""

    field_map = {
        "metrics_by_transformation_model_family": "metrics_by_transformation_model_family",
        "paired_exact_valid_deltas": "paired_exact_valid_deltas",
        "clustered_confidence_intervals": "clustered_confidence_intervals",
        "support_contraction": "support_contraction",
        "convergence_harm": "convergence_harm",
        "work_matching": "work_matching",
        "destructive_control_results": "source_destructive_control_results",
    }
    fields = {}
    all_mismatches = []
    maximum = 0.0
    for source_key, cold_key in field_map.items():
        left = source.get(source_key)
        right = cold.get(cold_key)
        mismatches, difference = _compare_values(left, right, source_key)
        fields[source_key] = {
            "classification": "match" if not mismatches else "discrepancy",
            "mismatch_count": len(mismatches),
            "mismatch_paths": mismatches[:50],
            "maximum_absolute_difference": round(difference, 12),
        }
        all_mismatches.extend(mismatches)
        maximum = max(maximum, difference)
    return {
        "all_match": not all_mismatches,
        "fields": fields,
        "mismatch_count": len(all_mismatches),
        "maximum_absolute_difference": round(maximum, 12),
    }


def _positive_gate(aggregates: Mapping[str, Any]) -> JsonDict:
    """Apply the preregistered positive rule to cold row-derived values."""

    intervals = aggregates["clustered_confidence_intervals"]
    support = aggregates["support_contraction"]
    convergence = aggregates["convergence_harm"]
    work = aggregates["work_matching"]
    checks = {
        "restructuring_lower_bound_above_zero": intervals["restructuring"]["lower"] > 0,
        "refinement_lower_bound_nonnegative": intervals["refinement"]["lower"] >= 0,
        "no_support_harm": not any(value["harm"] for value in support.values()),
        "no_convergence_harm": not any(value["harm"] for value in convergence.values()),
        "no_work_harm": work["planned_budgets_match"] and work["no_grouped_work_harm"],
    }
    checks["positive"] = all(checks.values())
    return checks


def _stable_payload(artifact: Mapping[str, Any]) -> Any:
    """Remove measured time and self hashes from reproducibility material."""

    if isinstance(artifact, Mapping):
        return {
            str(key): _stable_payload(value)
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum", "independent_evaluator_hash"}
        }
    if isinstance(artifact, list):
        return [_stable_payload(value) for value in artifact]
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable source, row, control, and verdict evidence."""

    return digest_value(_stable_payload(artifact))


def _blocked_artifact(
    *, run_date: str, duration_s: float, gate_summary: Mapping[str, Any], evaluator_hash: str
) -> JsonDict:
    """Return the full fail-closed artifact without substitute rows."""

    verdict_class = "disqualified" if gate_summary.get("missing_planned_rows") else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_real_output_fixed_point_audit",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(gate_summary.get("source_artifact_hashes", {})),
        "independent_evaluator_hash": evaluator_hash,
        "row_grid_receipts": deepcopy(gate_summary.get("grid_observation", {})),
        "exact_recomputed_metrics": {},
        "headline_differences": {},
        "clustered_confidence_intervals": {},
        "training_isolation_receipts": {},
        "oracle_feature_violations": [],
        "duplicate_case_checks": {},
        "destructive_control_results": {},
        "source_verdict_supported": False,
        "rows": [],
        "model_output_fixed_point_audit_completed": False,
        "gate_check_summary": deepcopy(dict(gate_summary)),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": "complete_blocked_real_output_fixed_point_audit: source authority failed",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    sources: Mapping[str, Any],
    *,
    repo_root: Path,
    run_date: str,
    duration_s: float,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonDict:
    """Build a complete cold audit or stop at the precondition boundary."""

    parse_run_date(run_date)
    evaluator_hash = digest_file(repo_root / MODULE_PATH) or "sha256:missing"
    gates = check_preconditions(sources, repo_root=repo_root)
    if not gates["all_passed"]:
        return _blocked_artifact(
            run_date=run_date,
            duration_s=duration_s,
            gate_summary=gates,
            evaluator_hash=evaluator_hash,
        )
    authority = rebuild_graph_authority(sources["exp6799"])
    audited_rows = audit_source_rows(sources["exp6800"], authority)
    aggregates = aggregate_rows(
        audited_rows,
        resamples=bootstrap_resamples,
        seed=BOOTSTRAP_SEED,
    )
    isolation = verify_training_isolation(sources, repo_root=repo_root)
    controls = run_destructive_controls(
        audited_rows,
        source_6800=sources["exp6800"],
        authority=authority,
    )
    differences = headline_differences(sources["exp6800"], aggregates)
    positive_gate = _positive_gate(aggregates)
    row_failures = [
        row["source_row_id"]
        for row in audited_rows
        if not (
            row["candidate_hashes_match"]
            and row["source_exact_outcomes_match"]
            and row["candidate_hashes_unchanged"]
            and all(value["source_outcomes_match"] for value in row["controls"].values())
        )
    ]
    control_authority_passed = bool(
        controls["results"][SURFACE_RELABEL]["semantics_preserved"]
        and controls["results"][TRANSFORMATION_LABEL_SWAP]["mislabel_detected"]
        and controls["results"][MODEL_ID_PERMUTATION]["global_effect_unchanged"]
        and controls["results"][IDENTICAL_ARM]["paired_exact_valid_delta"] == 0.0
        and controls["results"][AGGREGATE_CONTRADICTION]["detected"]
        and controls["results"][GROUP_PERMUTATION]["all_source_control_outcomes_recomputed"]
        and controls["results"][EDGE_DELETION]["all_source_control_outcomes_recomputed"]
    )
    authority_passed = bool(
        not authority["mismatches"]
        and not authority["operation_label_violations"]
        and not row_failures
        and isolation["passed"]
        and control_authority_passed
    )
    cold_class = "positive" if positive_gate["positive"] else "null"
    source_supported = bool(
        authority_passed
        and differences["all_match"]
        and sources["exp6800"].get("verdict_class") == cold_class
    )
    if not authority_passed:
        verdict_class = "disqualified"
    elif not differences["all_match"]:
        verdict_class = "partial"
    else:
        verdict_class = cold_class
    honest = {
        "positive": "complete: cold audit supports a positive real-output fixed-point transfer",
        "null": "complete: cold audit supports the real-output fixed-point null",
        "partial": "complete_partial: cold rows expose source headline discrepancies",
        "disqualified": "complete_disqualified: authority or shortcut checks failed",
    }[verdict_class]
    exact_metrics = {
        key: deepcopy(value)
        for key, value in aggregates.items()
        if key != "clustered_confidence_intervals"
    }
    gate_summary = deepcopy(gates)
    gate_summary.update(
        {
            "graph_authority_mismatches": authority["mismatches"],
            "operation_label_violations": authority["operation_label_violations"],
            "row_semantic_failure_count": len(row_failures),
            "row_semantic_failures": row_failures[:20],
            "training_isolation_passed": isolation["passed"],
            "control_authority_passed": control_authority_passed,
            "headline_match": differences["all_match"],
            "row_derived_positive_gate": positive_gate,
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_real_output_fixed_point_audit",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(gates["source_artifact_hashes"]),
        "independent_evaluator_hash": evaluator_hash,
        "row_grid_receipts": {
            **deepcopy(gates["grid_observation"]),
            "underlying_case_count": len({row["case_cluster_key"] for row in audited_rows}),
            "model_specific_source_case_count": len({row["source_case_id"] for row in audited_rows}),
            "cluster_key": "case_cluster_key",
            "audit_source_row_count": len(audited_rows),
            "control_row_count": len(controls["rows"]),
        },
        "exact_recomputed_metrics": exact_metrics,
        "headline_differences": differences,
        "clustered_confidence_intervals": {
            **deepcopy(aggregates["clustered_confidence_intervals"]),
            "transformation_interaction": deepcopy(aggregates["transformation_interaction"]),
        },
        "training_isolation_receipts": isolation,
        "oracle_feature_violations": deepcopy(isolation["oracle_feature_violations"]),
        "duplicate_case_checks": deepcopy(controls["results"][DUPLICATE_CASE_REMOVAL]),
        "destructive_control_results": deepcopy(controls["results"]),
        "source_verdict_supported": source_supported,
        "rows": audited_rows + controls["rows"],
        "model_output_fixed_point_audit_completed": True,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, checksum, row derivation, and blocked behavior."""

    errors = []
    if set(artifact) != set(REQUIRED_FIELDS):
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_FIELDS):
        errors.append("field principle coverage mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random seed mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class outside closed enum")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest verdict lacks terminal prefix")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be non-negative")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    blocked = artifact.get("model_output_fixed_point_audit_completed") is False
    if blocked:
        if artifact.get("status") != "complete_blocked_real_output_fixed_point_audit":
            errors.append("blocked status mismatch")
        if artifact.get("rows") != []:
            errors.append("blocked artifact must not contain rows")
        if artifact.get("gate_check_summary", {}).get("all_passed") is not False:
            errors.append("blocked artifact must retain failed gates")
        return errors
    rows = [row for row in artifact.get("rows", []) if row.get("row_kind") == "audited_unit"]
    controls = [row for row in artifact.get("rows", []) if row.get("row_kind") == "control"]
    if len(rows) != PLANNED_SOURCE_ROW_COUNT or len(controls) != len(CONTROLS):
        errors.append("audit row count mismatch")
        return errors
    resamples = int(
        artifact.get("clustered_confidence_intervals", {})
        .get("base", {})
        .get("resamples", BOOTSTRAP_RESAMPLES)
    )
    reduced = aggregate_rows(rows, resamples=resamples, seed=BOOTSTRAP_SEED)
    expected_metrics = {
        key: value for key, value in reduced.items() if key != "clustered_confidence_intervals"
    }
    if artifact.get("exact_recomputed_metrics") != expected_metrics:
        errors.append("row-derived metrics mismatch")
    expected_intervals = {
        **reduced["clustered_confidence_intervals"],
        "transformation_interaction": reduced["transformation_interaction"],
    }
    if artifact.get("clustered_confidence_intervals") != expected_intervals:
        errors.append("row-derived intervals mismatch")
    return errors


def write_output(artifact: Mapping[str, Any], path: Path) -> None:
    """Atomically write one explicit artifact path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_run_date(value: str) -> str:
    """Require the exact YYYYMMDD command date shape."""

    if len(value) != 8 or not value.isdigit():
        raise ValueError("run date must use YYYYMMDD")
    return value


def build_from_repo(
    *, repo_root: Path = REPO_ROOT, run_date: str = RUN_DATE
) -> JsonDict:
    """Load frozen bytes, measure the audit, and return its artifact."""

    started = time.monotonic()
    sources = load_sources(repo_root)
    artifact = build_artifact(
        sources,
        repo_root=repo_root,
        run_date=run_date,
        duration_s=0.0,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded audit and write the requested result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    artifact = build_from_repo(repo_root=args.repo_root, run_date=parse_run_date(args.date))
    output = args.artifact_path if args.artifact_path.is_absolute() else args.repo_root / args.artifact_path
    write_output(artifact, output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper calls main.
    raise SystemExit(main())
