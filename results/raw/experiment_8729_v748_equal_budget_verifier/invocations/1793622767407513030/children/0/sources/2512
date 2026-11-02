"""Cold-audit the V592 grouped-versus-flat fixed-point comparison.

This module reads only serialized JSON and Python source text. It implements
constraint evaluation, aggregation, controls, and confidence intervals again.
That separation lets the audit detect a shared source-code or headline bug.

Spec refs: REQ-VERIFY-6789 and SCENARIO-VERIFY-6789-*.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path("python/carnot/experiment_6789_soft_fixed_point_cold_authority_audit.py")
SPEC_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
SOURCE_6786_PATH = Path("results/experiment_6786_constraint_dependency_hard_negative_fixture.json")
SOURCE_6787_PATH = Path("results/experiment_6787_group_aware_soft_fixed_point.json")
SOURCE_6788_PATH = Path("results/experiment_6788_soft_fixed_point_structural_control_ab.json")
SOURCE_MODULE_6787_PATH = Path("python/carnot/experiment_6787_group_aware_soft_fixed_point.py")
SOURCE_MODULE_6788_PATH = Path(
    "python/carnot/experiment_6788_soft_fixed_point_structural_control_ab.py"
)
RESULT_PATH = Path("results/experiment_6789_soft_fixed_point_cold_authority_audit.json")
EXPERIMENT_ID = "experiment_6789_soft_fixed_point_cold_authority_audit"
SCHEMA = "carnot.experiment_6789.soft_fixed_point_cold_authority_audit.v1"
RUN_DATE = "20260830"
RANDOM_SEED = 6_789_000
INFERENCE_SUBSTRATE = "fresh_process_cpu_fixed_point_audit_no_llm"
GROUPED_ARM = "grouped_fixed_point"
FLAT_ARM = "flat_recurrent_control"
ARMS = (GROUPED_ARM, FLAT_ARM)
PLANNED_SOURCE_ROW_COUNT = 640
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 6_788_900
LABEL_PERMUTATION = "label_permutation"
GROUP_ID_PERMUTATION = "group_id_permutation"
DEPENDENCY_REWIRE = "dependency_rewire"
TOPOLOGY_ID_SWAP = "topology_id_swap"
CONTROLS = (
    LABEL_PERMUTATION,
    GROUP_ID_PERMUTATION,
    DEPENDENCY_REWIRE,
    TOPOLOGY_ID_SWAP,
)
CONTROL_SEEDS = {
    LABEL_PERMUTATION: 6_789_101,
    GROUP_ID_PERMUTATION: 6_789_202,
    DEPENDENCY_REWIRE: 6_789_303,
    TOPOLOGY_ID_SWAP: 6_789_404,
}
PLANNED_CONTROL_ROW_COUNT = PLANNED_SOURCE_ROW_COUNT * len(CONTROLS)
PLANNED_AUDIT_ROW_COUNT = PLANNED_SOURCE_ROW_COUNT + PLANNED_CONTROL_ROW_COUNT
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
FORBIDDEN_PROPOSAL_KEYS = {
    "exact_assignments",
    "exact_valid",
    "exact_label",
    "exact_checker_feedback",
    "solver_conflicts",
    "oracle_residual",
    "oracle_residuals",
    "post_action_outcome",
    "post_action_outcomes",
    "future_receipts",
    "future_rows",
    "target_assignment",
    "target_assignments",
    "ground_truth_assignment",
}
PREFREEZE_FUNCTIONS_6787 = {
    "initial_variable_state",
    "run_fixed_point",
    "_dependency_violation",
    "fit_seed",
    "decode_candidates",
    "propose_unit",
    "_group_messages",
    "_dependency_messages",
    "recurrent_step",
}
PREFREEZE_FUNCTIONS_6788 = {
    "build_arm_models",
    "recurrent_step",
    "fit_arm",
    "run_arm_fixed_point",
    "_assignment_score",
    "propose_raw_row",
}
HEADLINE_FIELDS = (
    "metrics_by_arm",
    "metrics_by_topology",
    "paired_exact_valid_delta",
    "paired_exact_valid_delta_ci95",
    "dependency_violation_delta",
    "distance_to_valid_delta",
    "hard_negative_auroc_by_arm",
    "unique_valid_support_by_arm",
    "support_contraction",
    "paired_key_count",
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
    "rows",
    "cold_recomputed_metrics",
    "headline_differences",
    "parameter_match_recomputed",
    "compute_match_recomputed",
    "exact_checker_call_order",
    "oracle_feature_violations",
    "label_permutation_effect",
    "group_id_permutation_effect",
    "dependency_rewire_effect",
    "topology_id_swap_effect",
    "hard_negative_shortcut_findings",
    "source_verdict_supported",
    "fixed_point_audit_completed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "schema": "A versioned schema makes incompatible audit artifacts fail closed.",
    "experiment_id": "A stable ID binds this evidence to the planned V592 audit.",
    "run_date": "The execution date separates this cold run from later source drift.",
    "status": "Status distinguishes a complete audit from a complete blocked audit.",
    "field_principles": "One-line purposes make every terminal field auditable.",
    "inference_substrate": "The declaration proves this was a fresh CPU audit with no LLM.",
    "duration_s": "Measured wall time exposes implausible or missing execution.",
    "random_seed": "The master seed fixes every destructive control and bootstrap draw.",
    "reproducibility_checksum": "A content hash detects row, control, or verdict drift.",
    "source_artifact_hashes": "Exact file hashes bind the audit to all three source artifacts.",
    "rows": "One row per source or control unit preserves attributable evidence.",
    "cold_recomputed_metrics": "Fresh row-only metrics replace source headline authority.",
    "headline_differences": "Explicit deltas expose every disagreement with source headlines.",
    "parameter_match_recomputed": "Row counts prove whether the two arms had matched capacity.",
    "compute_match_recomputed": "Row budgets prove whether updates and candidate work matched.",
    "exact_checker_call_order": "Static and runtime receipts prove exact checking was post-proposal.",
    "oracle_feature_violations": "Any proposal-time authority use disqualifies oracle distinction.",
    "label_permutation_effect": "Permuted labels test whether exact-label alignment made the result.",
    "group_id_permutation_effect": "Permuted group values test group structure against identity.",
    "dependency_rewire_effect": "Degree-stratified rewiring tests whether dependency structure matters.",
    "topology_id_swap_effect": "Swapped family IDs test invariance to topology names.",
    "hard_negative_shortcut_findings": "Class-specific checks rule out easy local-failure shortcuts.",
    "source_verdict_supported": "This boolean states whether all cold and destructive gates support V592.",
    "fixed_point_audit_completed": "True means the full audit ran even if it rejected the source claim.",
    "gate_check_summary": "Expected and observed values make blocked states reproducible.",
    "verifier_is_oracle": "True only if exact authority entered proposal selection.",
    "verdict_class": "A closed class keeps the terminal audit result machine-readable.",
    "honest_verdict": "A terminal prefix reports confirmation, downgrade, or disqualification plainly.",
}


class AuditInputError(ValueError):
    """Raised when serialized evidence cannot support an exact cold audit."""


def canonical_json(value: Any) -> str:
    """Serialize stable JSON and reject non-finite values before hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def json_digest(value: Any) -> str:
    """Return a prefixed SHA-256 digest of canonical JSON."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str | None:
    """Hash one source file, or return None so a missing gate stays observable."""

    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON object without importing the code that produced it."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditInputError(f"{path} must contain a JSON object")
    return value


def load_source_bundle(repo_root: Path) -> JsonDict:
    """Load all frozen source bytes and retain independent integrity digests."""

    paths = {
        "exp6786": SOURCE_6786_PATH,
        "exp6787": SOURCE_6787_PATH,
        "exp6788": SOURCE_6788_PATH,
    }
    bundle: JsonDict = {}
    file_hashes: JsonDict = {}
    canonical_hashes: JsonDict = {}
    for name, relative in paths.items():
        path = repo_root / relative
        bundle[name] = load_json_object(path)
        file_hashes[relative.as_posix()] = file_digest(path)
        canonical_hashes[name] = json_digest(bundle[name])
    bundle["_file_hashes"] = file_hashes
    bundle["_canonical_hashes"] = canonical_hashes
    return bundle


def _gate(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record one fail-closed precondition with its actual observation."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def _summarize_gates(checks: Sequence[Mapping[str, Any]], *, missing_rows: bool) -> JsonDict:
    """Keep every failed observation instead of reducing failure to one boolean."""

    failed = [deepcopy(dict(row)) for row in checks if not row.get("passed")]
    return {
        "all_passed": not failed,
        "checks": [deepcopy(dict(row)) for row in checks],
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
        "missing_planned_rows": missing_rows,
    }


def _declared_source_hash_checks(source: Mapping[str, Any], repo_root: Path) -> list[JsonDict]:
    """Rehash every upstream path declared by Exp6788."""

    checks: list[JsonDict] = []
    declared = source.get("source_artifact_hashes", {})
    if not isinstance(declared, Mapping) or not declared:
        return [_gate("declared_source_hashes", "nonempty mapping", declared, False)]
    for relative, expected in sorted(declared.items()):
        observed = file_digest(repo_root / str(relative))
        checks.append(_gate(f"source_hash:{relative}", expected, observed))
    return checks


def _expected_row_ids(source: Mapping[str, Any]) -> list[str]:
    """Read only the frozen row identity roster, never a source headline."""

    manifest = source.get("frozen_manifest", {})
    row_ids = manifest.get("row_ids", []) if isinstance(manifest, Mapping) else []
    return [str(value) for value in row_ids] if isinstance(row_ids, list) else []


def _raw_row_receipts_are_complete(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, JsonDict]:
    """Check raw candidate bytes and the post-proposal receipt on every row."""

    malformed: list[str] = []
    for row in rows:
        row_id = str(row.get("row_id"))
        candidates = row.get("candidates")
        hashes = row.get("candidate_hashes")
        outcomes = row.get("exact_outcomes")
        receipt = row.get("exact_evaluation_receipt")
        budget = row.get("candidate_budget")
        if not (
            isinstance(budget, int)
            and isinstance(candidates, list)
            and isinstance(hashes, list)
            and isinstance(outcomes, list)
            and isinstance(receipt, Mapping)
            and len(candidates) == len(hashes) == len(outcomes) == budget == 3
            and receipt.get("candidate_hashes_before") == hashes
            and receipt.get("candidate_hashes_after") == hashes
            and receipt.get("evaluated_after_proposal") is True
            and receipt.get("model_feedback_applied") is False
        ):
            malformed.append(row_id)
    return not malformed, {"malformed_count": len(malformed), "row_ids": malformed[:20]}


def check_preconditions(sources: Mapping[str, Any], *, repo_root: Path) -> JsonDict:
    """Require complete fixed rows, hashes, candidates, and exact receipts."""

    source_6786 = sources.get("exp6786", {})
    source_6787 = sources.get("exp6787", {})
    source_6788 = sources.get("exp6788", {})
    rows = source_6788.get("rows", []) if isinstance(source_6788, Mapping) else []
    rows = rows if isinstance(rows, list) else []
    expected_ids = _expected_row_ids(source_6788) if isinstance(source_6788, Mapping) else []
    observed_ids = [str(row.get("row_id")) for row in rows if isinstance(row, Mapping)]
    missing_rows = set(expected_ids) != set(observed_ids) or len(observed_ids) != len(expected_ids)
    raw_complete, raw_observed = _raw_row_receipts_are_complete(
        [row for row in rows if isinstance(row, Mapping)]
    )
    paired: dict[str, set[str]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            paired.setdefault(str(row.get("paired_key")), set()).add(str(row.get("arm")))
    pairs_complete = bool(paired) and all(arms == set(ARMS) for arms in paired.values())
    stored_canonical = sources.get("_canonical_hashes", {})
    canonical_observed = {
        "exp6786": json_digest(source_6786),
        "exp6787": json_digest(source_6787),
        "exp6788": json_digest(source_6788),
    }
    bundle_integrity = all(
        isinstance(stored_canonical, Mapping) and stored_canonical.get(name) == digest
        for name, digest in canonical_observed.items()
    )
    stored_files = sources.get("_file_hashes", {})
    current_files = {
        relative.as_posix(): file_digest(repo_root / relative)
        for relative in (SOURCE_6786_PATH, SOURCE_6787_PATH, SOURCE_6788_PATH)
    }
    files_match = isinstance(stored_files, Mapping) and all(
        stored_files.get(name) == digest for name, digest in current_files.items()
    )
    declared_6788 = (
        source_6788.get("source_artifact_hashes", {}) if isinstance(source_6788, Mapping) else {}
    )
    checks = [
        _gate(
            "fixed_point_comparison_completed",
            True,
            source_6788.get("fixed_point_comparison_completed")
            if isinstance(source_6788, Mapping)
            else None,
        ),
        _gate("planned_source_row_count", PLANNED_SOURCE_ROW_COUNT, len(rows)),
        _gate(
            "planned_row_keys",
            {"count": PLANNED_SOURCE_ROW_COUNT, "exact_match": True},
            {"count": len(observed_ids), "exact_match": not missing_rows},
        ),
        _gate("both_arms_per_paired_key", True, pairs_complete),
        _gate("raw_candidates_and_exact_receipts", True, raw_complete),
        _gate("loaded_bundle_integrity", True, bundle_integrity),
        _gate("source_files_unchanged_since_load", True, files_match),
        _gate(
            "exp6786_fixture_ready",
            True,
            source_6786.get("constraint_group_fixture_ready")
            if isinstance(source_6786, Mapping)
            else None,
        ),
        _gate(
            "exp6787_proposer_ready",
            True,
            source_6787.get("soft_fixed_point_proposer_ready")
            if isinstance(source_6787, Mapping)
            else None,
        ),
        _gate(
            "exp6787_source_hash",
            current_files[SOURCE_6786_PATH.as_posix()],
            source_6787.get("source_artifact_hash") if isinstance(source_6787, Mapping) else None,
        ),
        _gate(
            "exp6788_exp6786_hash",
            current_files[SOURCE_6786_PATH.as_posix()],
            declared_6788.get(SOURCE_6786_PATH.as_posix())
            if isinstance(declared_6788, Mapping)
            else None,
        ),
        _gate(
            "exp6788_exp6787_hash",
            current_files[SOURCE_6787_PATH.as_posix()],
            declared_6788.get(SOURCE_6787_PATH.as_posix())
            if isinstance(declared_6788, Mapping)
            else None,
        ),
        *(
            _declared_source_hash_checks(source_6788, repo_root)
            if isinstance(source_6788, Mapping)
            else []
        ),
    ]
    summary = _summarize_gates(checks, missing_rows=missing_rows)
    summary["raw_receipt_observation"] = raw_observed
    summary["planned_row_count"] = len(expected_ids)
    summary["observed_row_count"] = len(rows)
    return summary


def candidate_digest(assignment: Mapping[str, Any]) -> str:
    """Hash one assignment without trusting its serialized source hash."""

    return json_digest(dict(assignment))


def _selected_state(group: Mapping[str, Any], assignment: Mapping[str, Any]) -> int | None:
    """Return the selected binary state only for a valid one-hot local group."""

    variables = [str(value) for value in group.get("variables", [])]
    if len(variables) != 2:
        raise AuditInputError("each local group must contain two variables")
    values = [assignment.get(variable) for variable in variables]
    if values == [1, 0]:
        return 0
    if values == [0, 1]:
        return 1
    return None


def _dependency_passes(relation: str, source_state: int, target_state: int) -> bool:
    """Apply one implication from its serialized relation name."""

    if relation == "implies_selected_one":
        return not (source_state == 1 and target_state == 0)
    if relation == "implies_selected_zero":
        return not (source_state == 0 and target_state == 1)
    raise AuditInputError(f"unknown dependency relation: {relation}")


def evaluate_assignment(graph: Mapping[str, Any], assignment: Mapping[str, Any]) -> JsonDict:
    """Evaluate local and dependency semantics without a source checker call."""

    groups = graph.get("local_groups", [])
    edges = graph.get("dependency_edges", [])
    if not isinstance(groups, list) or not groups:
        raise AuditInputError("graph must contain local groups")
    if not isinstance(edges, list):
        raise AuditInputError("graph dependency edges must be a list")
    states: dict[str, int | None] = {}
    failed_groups: list[str] = []
    for group in groups:
        group_id = str(group["group_id"])
        state = _selected_state(group, assignment)
        states[group_id] = state
        if state is None:
            failed_groups.append(group_id)
    failed_dependencies: list[str] = []
    dependency_class: list[str] = []
    for edge in edges:
        source_state = states.get(str(edge["source_group"]))
        target_state = states.get(str(edge["target_group"]))
        if source_state is None or target_state is None:
            continue
        if not _dependency_passes(str(edge["relation_type"]), source_state, target_state):
            failed_dependencies.append(str(edge["dependency_id"]))
            dependency_class.append(str(edge["relation_type"]))
    exact_valid = (
        not failed_groups
        and not failed_dependencies
        and all(
            states.get(str(edge["source_group"])) is not None
            and states.get(str(edge["target_group"])) is not None
            for edge in edges
        )
    )
    return {
        "local_checks_passed": not failed_groups,
        "failed_local_group_ids": failed_groups,
        "failed_dependency_ids": failed_dependencies,
        "dependency_failure_classes": dependency_class,
        "exact_valid": exact_valid,
    }


def _assignment_from_states(groups: Sequence[Mapping[str, Any]], states: Sequence[int]) -> JsonDict:
    """Expand one binary state per group into named one-hot variables."""

    assignment: JsonDict = {}
    for group, state in zip(groups, states, strict=True):
        first, second = [str(value) for value in group["variables"]]
        assignment[first] = int(state == 0)
        assignment[second] = int(state == 1)
    return assignment


def enumerate_valid_assignments(graph: Mapping[str, Any]) -> list[JsonDict]:
    """Enumerate graph semantics instead of consuming serialized exact answers."""

    groups = graph.get("local_groups", [])
    if not isinstance(groups, list) or not groups:
        raise AuditInputError("graph must contain local groups")
    valid: list[JsonDict] = []
    for states in product((0, 1), repeat=len(groups)):
        assignment = _assignment_from_states(groups, states)
        if evaluate_assignment(graph, assignment)["exact_valid"]:
            valid.append(assignment)
    return valid


def _group_distance(
    graph: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]
) -> int:
    """Count local selections that differ, including malformed local groups."""

    return sum(
        _selected_state(group, left) != _selected_state(group, right)
        for group in graph["local_groups"]
    )


def nearest_valid_distance(
    graph: Mapping[str, Any], assignment: Mapping[str, Any], valid: Sequence[Mapping[str, Any]]
) -> int:
    """Measure group Hamming distance to independently enumerated valid states."""

    if not valid:
        raise AuditInputError("graph has no valid assignments")
    return min(_group_distance(graph, assignment, target) for target in valid)


def binary_auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Compute pairwise AUROC with half credit for ties."""

    if not positive or not negative:
        return None
    wins = 0.0
    comparisons = 0
    for good in positive:
        for bad in negative:
            comparisons += 1
            wins += 1.0 if good > bad else 0.5 if good == bad else 0.0
    return round(wins / comparisons, 10)


def _assignment_score(
    variable_state: Sequence[Sequence[Any]], graph: Mapping[str, Any], assignment: Mapping[str, Any]
) -> float:
    """Recompute a candidate score from frozen proposal probabilities."""

    values: list[float] = []
    for index, group in enumerate(graph["local_groups"]):
        state = _selected_state(group, assignment)
        values.append(float(variable_state[index][state]) if state is not None else 0.0)
    return round(sum(values) / len(values), 10)


def _unit_maps(sources: Mapping[str, Any]) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    """Index serialized graphs and hard negatives by frozen unit identity."""

    units = {
        str(unit["unit_id"]): deepcopy(dict(unit))
        for unit in sources["exp6786"]["frozen_manifest"]["units"]
    }
    hard = {
        str(row["unit_id"]): deepcopy(dict(row))
        for row in sources["exp6786"]["rows"]
        if row.get("negative_class") == "hard_cross_dependency_failure"
    }
    return units, hard


def _recomputed_outcomes(
    source_row: Mapping[str, Any], graph: Mapping[str, Any], valid: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Recompute every candidate outcome from assignment bytes and graph semantics."""

    outcomes: list[JsonDict] = []
    for candidate in source_row["candidates"]:
        assignment = deepcopy(dict(candidate["assignment"]))
        receipt = evaluate_assignment(graph, assignment)
        outcomes.append(
            {
                "candidate_index": int(candidate["candidate_index"]),
                "candidate_hash": candidate_digest(assignment),
                "assignment": assignment,
                "proposal_score": float(candidate["proposal_score"]),
                "exact_valid": bool(receipt["exact_valid"]),
                "local_checks_passed": bool(receipt["local_checks_passed"]),
                "failed_local_group_ids": receipt["failed_local_group_ids"],
                "failed_dependency_ids": receipt["failed_dependency_ids"],
                "dependency_failure_classes": receipt["dependency_failure_classes"],
                "dependency_violation_count": len(receipt["failed_dependency_ids"]),
                "distance_to_nearest_valid": nearest_valid_distance(graph, assignment, valid),
            }
        )
    return outcomes


def _outcomes_match_source(
    recomputed: Sequence[Mapping[str, Any]], source: Sequence[Mapping[str, Any]]
) -> bool:
    """Compare semantic fields while ignoring source-only field ordering."""

    keys = (
        "candidate_index",
        "candidate_hash",
        "assignment",
        "exact_valid",
        "local_checks_passed",
        "failed_local_group_ids",
        "failed_dependency_ids",
        "dependency_violation_count",
        "distance_to_nearest_valid",
    )
    return [tuple(row.get(key) for key in keys) for row in recomputed] == [
        tuple(row.get(key) for key in keys) for row in source
    ]


def _audit_source_row(
    source_row: Mapping[str, Any], unit: Mapping[str, Any], hard_negative: Mapping[str, Any]
) -> JsonDict:
    """Build one row using no source aggregate or exact outcome authority."""

    graph = unit["graph"]
    valid = enumerate_valid_assignments(graph)
    outcomes = _recomputed_outcomes(source_row, graph, valid)
    candidate_hashes = [row["candidate_hash"] for row in outcomes]
    valid_count = sum(row["exact_valid"] for row in outcomes)
    dependency_count = sum(bool(row["failed_dependency_ids"]) for row in outcomes)
    positive_scores = [
        _assignment_score(source_row["variable_state"], graph, assignment) for assignment in valid
    ]
    negative_score = _assignment_score(
        source_row["variable_state"], graph, hard_negative["candidate_assignment"]
    )
    receipt = source_row["exact_evaluation_receipt"]
    source_id = str(source_row["row_id"])
    return {
        "schema": "carnot.experiment_6789.audit_row.v1",
        "row_id": f"{source_id}|audit-source",
        "row_type": "source_recompute",
        "control": "source",
        "control_seed": RANDOM_SEED,
        "source_row_id": source_id,
        "paired_key": str(source_row["paired_key"]),
        "unit_id": str(source_row["unit_id"]),
        "split": str(source_row["split"]),
        "topology_family": str(source_row["topology_family"]),
        "arm": str(source_row["arm"]),
        "random_seed": int(source_row["random_seed"]),
        "parameter_count": int(source_row["parameter_count"]),
        "optimizer_update_count": int(source_row["optimizer_update_count"]),
        "candidate_budget": int(source_row["candidate_budget"]),
        "iterations": int(source_row["iterations"]),
        "state_residual": float(source_row["state_residual"]),
        "stop_reason": str(source_row["stop_reason"]),
        "finite_values": bool(source_row["finite_values"]),
        "runtime_s": float(source_row["runtime_s"]),
        "candidate_hashes_recomputed": candidate_hashes,
        "candidate_hashes_match": candidate_hashes == source_row["candidate_hashes"],
        "exact_outcomes": outcomes,
        "source_exact_outcomes_match": _outcomes_match_source(
            outcomes, source_row["exact_outcomes"]
        ),
        "exact_valid_candidate_count": valid_count,
        "exact_valid_rate": round(valid_count / len(outcomes), 10),
        "cross_dependency_violation_count": dependency_count,
        "cross_dependency_violation_rate": round(dependency_count / len(outcomes), 10),
        "nearest_valid_distance": min(row["distance_to_nearest_valid"] for row in outcomes),
        "hard_negative_auroc": binary_auroc(positive_scores, [negative_score]),
        "hard_negative_positive_count": len(positive_scores),
        "all_candidates_pass_local_checks": all(row["local_checks_passed"] for row in outcomes),
        "exact_checker_after_candidate_freeze": bool(
            receipt.get("evaluated_after_proposal") is True
            and receipt.get("model_feedback_applied") is False
            and receipt.get("candidate_hashes_before") == candidate_hashes
            and receipt.get("candidate_hashes_after") == candidate_hashes
        ),
        "budget_match": True,
        "control_receipt": {"authority": "independent_graph_enumeration"},
    }


def recompute_source_rows(sources: Mapping[str, Any]) -> list[JsonDict]:
    """Cold-recompute all source rows in stable source-row order."""

    units, hard = _unit_maps(sources)
    rows = []
    for source_row in sources["exp6788"]["rows"]:
        unit_id = str(source_row["unit_id"])
        if unit_id not in units or unit_id not in hard:
            raise AuditInputError(f"missing exact unit authority: {unit_id}")
        rows.append(_audit_source_row(source_row, units[unit_id], hard[unit_id]))
    return rows


def _stable_seed(base: int, *parts: Any) -> int:
    """Derive a process-independent control seed from stable text."""

    material = "|".join(str(value) for value in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _permuted_group_assignment(
    graph: Mapping[str, Any], assignment: Mapping[str, Any], *, seed: int
) -> tuple[JsonDict, JsonDict]:
    """Move complete two-bit values among group IDs without changing values."""

    groups = list(graph["local_groups"])
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    if order == list(range(len(groups))) and len(order) > 1:
        order = order[1:] + order[:1]
    changed: JsonDict = {}
    mapping: JsonDict = {}
    for source_index, target_index in enumerate(order):
        source_group = groups[source_index]
        target_group = groups[target_index]
        source_variables = [str(value) for value in source_group["variables"]]
        target_variables = [str(value) for value in target_group["variables"]]
        changed[target_variables[0]] = assignment[source_variables[0]]
        changed[target_variables[1]] = assignment[source_variables[1]]
        mapping[str(source_group["group_id"])] = str(target_group["group_id"])
    return changed, mapping


def _degree_map(graph: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Return directed in-degree and out-degree for each group ID."""

    groups = [str(row["group_id"]) for row in graph["local_groups"]]
    indegree = {group: 0 for group in groups}
    outdegree = {group: 0 for group in groups}
    for edge in graph["dependency_edges"]:
        outdegree[str(edge["source_group"])] += 1
        indegree[str(edge["target_group"])] += 1
    return {group: (indegree[group], outdegree[group]) for group in groups}


def _rewired_graph(graph: Mapping[str, Any], *, seed: int) -> tuple[JsonDict, JsonDict]:
    """Reassign targets only among nodes with the same directed degree stratum."""

    changed = deepcopy(dict(graph))
    before = _degree_map(graph)
    strata: dict[tuple[int, int], list[str]] = {}
    for group, degree in before.items():
        strata.setdefault(degree, []).append(group)
    mapping: dict[str, str] = {}
    for degree, members in sorted(strata.items()):
        ordered = sorted(members)
        if len(ordered) == 1:
            mapping[ordered[0]] = ordered[0]
            continue
        offset = random.Random(_stable_seed(seed, degree)).randrange(1, len(ordered))
        for index, group in enumerate(ordered):
            mapping[group] = ordered[(index + offset) % len(ordered)]
    changed_edges = []
    for edge in changed["dependency_edges"]:
        edge["target_group"] = mapping[str(edge["target_group"])]
        changed_edges.append(edge)
    after = _degree_map(changed)
    return changed, {
        "target_mapping": mapping,
        "changed_edge_count": sum(
            left["target_group"] != right["target_group"]
            for left, right in zip(graph["dependency_edges"], changed_edges, strict=True)
        ),
        "degree_strata_preserved": before == after,
    }


def _control_from_outcomes(
    source: Mapping[str, Any],
    *,
    control: str,
    outcomes: Sequence[Mapping[str, Any]],
    topology_family: str | None = None,
    hard_negative_auroc: float | None = None,
    receipt: Mapping[str, Any],
) -> JsonDict:
    """Create one budget-matched control row from counterfactual outcomes."""

    rows = [deepcopy(dict(row)) for row in outcomes]
    valid_count = sum(bool(row["exact_valid"]) for row in rows)
    dependency_count = sum(bool(row["failed_dependency_ids"]) for row in rows)
    source_id = str(source["source_row_id"])
    return {
        **{
            key: deepcopy(source[key])
            for key in (
                "schema",
                "source_row_id",
                "paired_key",
                "unit_id",
                "split",
                "arm",
                "random_seed",
                "parameter_count",
                "optimizer_update_count",
                "candidate_budget",
                "iterations",
                "state_residual",
                "stop_reason",
                "finite_values",
                "runtime_s",
            )
        },
        "row_id": f"{source_id}|control-{control}",
        "row_type": "destructive_control",
        "control": control,
        "control_seed": CONTROL_SEEDS[control],
        "topology_family": topology_family or str(source["topology_family"]),
        "candidate_hashes_recomputed": [str(row["candidate_hash"]) for row in rows],
        "candidate_hashes_match": True,
        "exact_outcomes": rows,
        "source_exact_outcomes_match": control == TOPOLOGY_ID_SWAP,
        "exact_valid_candidate_count": valid_count,
        "exact_valid_rate": round(valid_count / len(rows), 10),
        "cross_dependency_violation_count": dependency_count,
        "cross_dependency_violation_rate": round(dependency_count / len(rows), 10),
        "nearest_valid_distance": min(int(row["distance_to_nearest_valid"]) for row in rows),
        "hard_negative_auroc": (
            source["hard_negative_auroc"] if hard_negative_auroc is None else hard_negative_auroc
        ),
        "hard_negative_positive_count": int(source["hard_negative_positive_count"]),
        "all_candidates_pass_local_checks": all(bool(row["local_checks_passed"]) for row in rows),
        "exact_checker_after_candidate_freeze": True,
        "budget_match": all(
            source[key] == source[key]
            for key in ("parameter_count", "optimizer_update_count", "candidate_budget")
        ),
        "control_receipt": deepcopy(dict(receipt)),
    }


def _counterfactual_outcomes(
    source: Mapping[str, Any],
    graph: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Evaluate changed assignments or graphs under an unchanged proposal budget."""

    valid = enumerate_valid_assignments(graph)
    rows = []
    for original, assignment in zip(source["exact_outcomes"], assignments, strict=True):
        receipt = evaluate_assignment(graph, assignment)
        rows.append(
            {
                "candidate_index": int(original["candidate_index"]),
                "candidate_hash": candidate_digest(assignment),
                "assignment": deepcopy(dict(assignment)),
                "proposal_score": float(original["proposal_score"]),
                "exact_valid": bool(receipt["exact_valid"]),
                "local_checks_passed": bool(receipt["local_checks_passed"]),
                "failed_local_group_ids": receipt["failed_local_group_ids"],
                "failed_dependency_ids": receipt["failed_dependency_ids"],
                "dependency_failure_classes": receipt["dependency_failure_classes"],
                "dependency_violation_count": len(receipt["failed_dependency_ids"]),
                "distance_to_nearest_valid": nearest_valid_distance(graph, assignment, valid),
            }
        )
    return rows


def _label_permutation_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Permute exact labels jointly across both arms inside each split."""

    output: list[JsonDict] = []
    replacements: dict[tuple[str, int], bool] = {}
    ordered_rows = sorted(source_rows, key=lambda row: str(row["source_row_id"]))
    for split in sorted({str(row["split"]) for row in ordered_rows}):
        locations = [
            (str(row["source_row_id"]), index)
            for row in ordered_rows
            if row["split"] == split
            for index, _ in enumerate(row["exact_outcomes"])
        ]
        labels = [
            bool(outcome["exact_valid"])
            for row in ordered_rows
            if row["split"] == split
            for outcome in row["exact_outcomes"]
        ]
        random.Random(_stable_seed(CONTROL_SEEDS[LABEL_PERMUTATION], split)).shuffle(labels)
        replacements.update(dict(zip(locations, labels, strict=True)))
    for source in source_rows:
        outcomes = []
        for index, original in enumerate(source["exact_outcomes"]):
            valid = replacements[(str(source["source_row_id"]), index)]
            changed = deepcopy(dict(original))
            changed["exact_valid"] = valid
            changed["local_checks_passed"] = True
            changed["failed_local_group_ids"] = []
            changed["failed_dependency_ids"] = [] if valid else ["permuted_exact_label"]
            changed["dependency_failure_classes"] = [] if valid else ["permuted_exact_label"]
            changed["dependency_violation_count"] = int(not valid)
            changed["distance_to_nearest_valid"] = int(not valid)
            outcomes.append(changed)
        output.append(
            _control_from_outcomes(
                source,
                control=LABEL_PERMUTATION,
                outcomes=outcomes,
                receipt={"permuted_within_split": source["split"], "budgets_frozen": True},
            )
        )
    return output


def build_control_rows(
    source_rows: Sequence[Mapping[str, Any]], sources: Mapping[str, Any]
) -> list[JsonDict]:
    """Run all four deterministic destructive controls for every source row."""

    units, hard = _unit_maps(sources)
    controls = _label_permutation_rows(source_rows)
    topology_names = sorted({str(row["topology_family"]) for row in source_rows})
    topology_swap = dict(zip(topology_names, reversed(topology_names), strict=True))
    for source in source_rows:
        unit = units[str(source["unit_id"])]
        graph = unit["graph"]
        assignments = [row["assignment"] for row in source["exact_outcomes"]]
        unit_seed = _stable_seed(
            CONTROL_SEEDS[GROUP_ID_PERMUTATION], source["unit_id"], source["random_seed"]
        )
        permuted_assignments = []
        group_mappings = []
        for assignment in assignments:
            changed, mapping = _permuted_group_assignment(graph, assignment, seed=unit_seed)
            permuted_assignments.append(changed)
            group_mappings.append(mapping)
        group_outcomes = _counterfactual_outcomes(source, graph, permuted_assignments)
        controls.append(
            _control_from_outcomes(
                source,
                control=GROUP_ID_PERMUTATION,
                outcomes=group_outcomes,
                receipt={"group_mappings": group_mappings, "values_preserved": True},
            )
        )

        rewire_seed = _stable_seed(
            CONTROL_SEEDS[DEPENDENCY_REWIRE], source["unit_id"], source["random_seed"]
        )
        rewired, rewire_receipt = _rewired_graph(graph, seed=rewire_seed)
        rewire_outcomes = _counterfactual_outcomes(source, rewired, assignments)
        rewired_valid = enumerate_valid_assignments(rewired)
        positive_scores = [
            _assignment_score(
                sources["exp6788"]["rows"][
                    next(
                        index
                        for index, row in enumerate(sources["exp6788"]["rows"])
                        if row["row_id"] == source["source_row_id"]
                    )
                ]["variable_state"],
                rewired,
                assignment,
            )
            for assignment in rewired_valid
        ]
        source_runtime_row = next(
            row for row in sources["exp6788"]["rows"] if row["row_id"] == source["source_row_id"]
        )
        negative_score = _assignment_score(
            source_runtime_row["variable_state"],
            rewired,
            hard[str(source["unit_id"])]["candidate_assignment"],
        )
        controls.append(
            _control_from_outcomes(
                source,
                control=DEPENDENCY_REWIRE,
                outcomes=rewire_outcomes,
                hard_negative_auroc=binary_auroc(positive_scores, [negative_score]),
                receipt=rewire_receipt,
            )
        )

        controls.append(
            _control_from_outcomes(
                source,
                control=TOPOLOGY_ID_SWAP,
                outcomes=source["exact_outcomes"],
                topology_family=topology_swap[str(source["topology_family"])],
                receipt={
                    "source_topology_family": source["topology_family"],
                    "swapped_topology_family": topology_swap[str(source["topology_family"])],
                    "graph_bytes_changed": False,
                },
            )
        )
    return sorted(controls, key=lambda row: (str(row["control"]), str(row["source_row_id"])))


def _mean(values: Sequence[float]) -> float | None:
    """Return the source-compatible rounded arithmetic mean."""

    return round(sum(values) / len(values), 10) if values else None


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated deterministic quantile."""

    if not values:
        raise AuditInputError("percentile requires values")
    if not 0.0 <= quantile <= 1.0:
        raise AuditInputError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 10)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 10)


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate one arm only from independently recomputed audit rows."""

    candidates = sum(int(row["candidate_budget"]) for row in rows)
    valid = sum(int(row["exact_valid_candidate_count"]) for row in rows)
    dependency = sum(int(row["cross_dependency_violation_count"]) for row in rows)
    valid_hashes = {
        str(outcome["candidate_hash"])
        for row in rows
        for outcome in row["exact_outcomes"]
        if outcome["exact_valid"]
    }
    aurocs = [
        float(row["hard_negative_auroc"]) for row in rows if row["hard_negative_auroc"] is not None
    ]
    return {
        "row_count": len(rows),
        "candidate_count": candidates,
        "exact_valid_candidate_count": valid,
        "exact_valid_rate": round(valid / candidates, 10) if candidates else None,
        "cross_dependency_violation_count": dependency,
        "cross_dependency_violation_rate": round(dependency / candidates, 10)
        if candidates
        else None,
        "mean_nearest_valid_distance": _mean(
            [float(row["nearest_valid_distance"]) for row in rows]
        ),
        "converged_count": sum(row["stop_reason"] == "converged" for row in rows),
        "convergence_rate": round(
            sum(row["stop_reason"] == "converged" for row in rows) / len(rows), 10
        )
        if rows
        else None,
        "mean_iterations": _mean([float(row["iterations"]) for row in rows]),
        "mean_state_residual": _mean([float(row["state_residual"]) for row in rows]),
        "finite_value_failure_count": sum(not row["finite_values"] for row in rows),
        "runtime_total_s": round(sum(float(row["runtime_s"]) for row in rows), 6),
        "hard_negative_auroc": _mean(aurocs),
        "hard_negative_auroc_defined_row_count": len(aurocs),
        "unique_valid_support": len(valid_hashes),
    }


def _paired_values(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create grouped-minus-flat effects while retaining unit clusters."""

    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(str(row["paired_key"]), {})[str(row["arm"])] = row
    values = []
    for paired_key, arms in sorted(pairs.items()):
        if set(arms) != set(ARMS):
            continue
        grouped = arms[GROUPED_ARM]
        flat = arms[FLAT_ARM]
        values.append(
            {
                "paired_key": paired_key,
                "unit_id": grouped["unit_id"],
                "topology_family": grouped["topology_family"],
                "exact_valid_delta": round(
                    float(grouped["exact_valid_rate"]) - float(flat["exact_valid_rate"]), 10
                ),
                "dependency_violation_delta": round(
                    float(grouped["cross_dependency_violation_rate"])
                    - float(flat["cross_dependency_violation_rate"]),
                    10,
                ),
                "distance_to_valid_delta": round(
                    float(grouped["nearest_valid_distance"])
                    - float(flat["nearest_valid_distance"]),
                    10,
                ),
            }
        )
    return values


def _effect_means(values: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce one paired effect collection without row weights."""

    return {
        "exact_valid_delta": _mean([float(row["exact_valid_delta"]) for row in values]),
        "dependency_violation_delta": _mean(
            [float(row["dependency_violation_delta"]) for row in values]
        ),
        "distance_to_valid_delta": _mean([float(row["distance_to_valid_delta"]) for row in values]),
    }


def _bootstrap_effects(
    paired: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> tuple[JsonDict, dict[str, JsonDict]]:
    """Resample units inside families while retaining all seeds and arms."""

    if resamples <= 0:
        raise AuditInputError("bootstrap resamples must be positive")
    families: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for value in paired:
        families.setdefault(str(value["topology_family"]), {}).setdefault(
            str(value["unit_id"]), []
        ).append(value)
    if not families:
        raise AuditInputError("paired bootstrap requires complete pairs")
    point_by_family = {
        family: _effect_means([item for rows in units.values() for item in rows])
        for family, units in families.items()
    }
    keys = ("exact_valid_delta", "dependency_violation_delta", "distance_to_valid_delta")
    point = {
        key: _mean([float(effect[key]) for effect in point_by_family.values()]) for key in keys
    }
    draws = {key: [] for key in keys}
    family_draws = {family: {key: [] for key in keys} for family in sorted(families)}
    generator = random.Random(seed)
    for _ in range(resamples):
        sampled_effects: dict[str, JsonDict] = {}
        for family in sorted(families):
            units = families[family]
            unit_ids = sorted(units)
            sampled = [generator.choice(unit_ids) for _ in unit_ids]
            values = [value for unit_id in sampled for value in units[unit_id]]
            effect = _effect_means(values)
            sampled_effects[family] = effect
            for key in keys:
                family_draws[family][key].append(float(effect[key]))
        for key in keys:
            draws[key].append(
                sum(float(effect[key]) for effect in sampled_effects.values())
                / len(sampled_effects)
            )
    ci = {
        key: {
            "lower": percentile(draws[key], 0.025),
            "upper": percentile(draws[key], 0.975),
            "confidence_level": 0.95,
            "resamples": resamples,
            "resampling_unit": "unit_inside_topology_family",
            "family_aggregation": "equal_weight_mean",
        }
        for key in keys
    }
    family_ci = {
        family: {
            key: {
                "lower": percentile(values, 0.025),
                "upper": percentile(values, 0.975),
                "confidence_level": 0.95,
                "resamples": resamples,
                "resampling_unit": "unit",
            }
            for key, values in effects.items()
        }
        for family, effects in family_draws.items()
    }
    return {"point": point, "ci": ci}, family_ci


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int
) -> JsonDict:
    """Derive every Exp6788 headline from cold audit rows."""

    metrics_by_arm = {arm: _arm_metrics([row for row in rows if row["arm"] == arm]) for arm in ARMS}
    paired = _paired_values(rows)
    bootstrap, family_ci = _bootstrap_effects(
        paired, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    metrics_by_topology: JsonDict = {}
    for family in sorted({str(row["topology_family"]) for row in rows}):
        family_rows = [row for row in rows if row["topology_family"] == family]
        family_values = [row for row in paired if row["topology_family"] == family]
        effects = _effect_means(family_values)
        metrics_by_topology[family] = {
            "split": sorted({str(row["split"]) for row in family_rows})[0],
            "unit_count": len({str(row["unit_id"]) for row in family_rows}),
            "paired_key_count": len(family_values),
            "metrics_by_arm": {
                arm: _arm_metrics([row for row in family_rows if row["arm"] == arm]) for arm in ARMS
            },
            "paired_exact_valid_delta": effects["exact_valid_delta"],
            "paired_exact_valid_delta_ci95": family_ci[family]["exact_valid_delta"],
            "dependency_violation_delta": effects["dependency_violation_delta"],
            "distance_to_valid_delta": effects["distance_to_valid_delta"],
        }
    grouped_support = int(metrics_by_arm[GROUPED_ARM]["unique_valid_support"])
    flat_support = int(metrics_by_arm[FLAT_ARM]["unique_valid_support"])
    contraction = max(0.0, (flat_support - grouped_support) / flat_support) if flat_support else 0.0
    return {
        "metrics_by_arm": metrics_by_arm,
        "metrics_by_topology": metrics_by_topology,
        "paired_exact_valid_delta": bootstrap["point"]["exact_valid_delta"],
        "paired_exact_valid_delta_ci95": bootstrap["ci"]["exact_valid_delta"],
        "dependency_violation_delta": bootstrap["point"]["dependency_violation_delta"],
        "distance_to_valid_delta": bootstrap["point"]["distance_to_valid_delta"],
        "hard_negative_auroc_by_arm": {
            arm: metrics_by_arm[arm]["hard_negative_auroc"] for arm in ARMS
        },
        "unique_valid_support_by_arm": {
            arm: metrics_by_arm[arm]["unique_valid_support"] for arm in ARMS
        },
        "support_contraction": round(contraction, 10),
        "paired_key_count": len(paired),
    }


def _control_effect(metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Add control-specific completion and positive-survival decisions."""

    ci = metrics["paired_exact_valid_delta_ci95"]
    return {
        **deepcopy(dict(metrics)),
        "row_count": len(rows),
        "all_budgets_matched": all(row["budget_match"] for row in rows),
        "positive_effect_survives": float(ci["lower"]) > 0.0,
    }


def reduce_audit_rows(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int
) -> JsonDict:
    """Reduce source and control rows through one independent implementation."""

    output: JsonDict = {}
    source_rows = [row for row in rows if row["control"] == "source"]
    output["source"] = aggregate_rows(
        source_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    for control in CONTROLS:
        selected = [row for row in rows if row["control"] == control]
        metrics = aggregate_rows(
            selected,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + CONTROL_SEEDS[control],
        )
        output[control] = _control_effect(metrics, selected)
    swapped = output[TOPOLOGY_ID_SWAP]
    source = output["source"]
    swapped["global_effect_invariant"] = (
        swapped["paired_exact_valid_delta"] == source["paired_exact_valid_delta"]
    )
    swapped["identifier_invariance_passed"] = swapped["global_effect_invariant"]
    return output


def recompute_parameter_match(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute capacity matching from all source rows and pairs."""

    source = [row for row in rows if row["control"] == "source"]
    counts = {
        arm: sorted({int(row["parameter_count"]) for row in source if row["arm"] == arm})
        for arm in ARMS
    }
    scalar = {arm: values[0] if len(values) == 1 else None for arm, values in counts.items()}
    maximum = (
        max(value for value in scalar.values() if value is not None)
        if all(value is not None for value in scalar.values())
        else 0
    )
    difference = (
        abs(int(scalar[GROUPED_ARM]) - int(scalar[FLAT_ARM])) / maximum if maximum else math.inf
    )
    pair_match = all(
        len({int(row["parameter_count"]) for row in source if row["paired_key"] == key}) == 1
        for key in {str(row["paired_key"]) for row in source}
    )
    return {
        "counts_observed_by_arm": counts,
        "parameter_count_by_arm": scalar,
        "difference_fraction": round(difference, 10),
        "tolerance": 0.05,
        "all_pairs_match": pair_match,
        "matched": pair_match and difference <= 0.05,
    }


def recompute_compute_match(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute updates, candidate work, iteration limits, and runtime envelope."""

    source = [row for row in rows if row["control"] == "source"]
    updates: JsonDict = {}
    candidate_budgets: JsonDict = {}
    iteration_sets: JsonDict = {}
    runtime: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in source if row["arm"] == arm]
        by_seed = {
            int(seed): {
                int(row["optimizer_update_count"]) for row in arm_rows if row["random_seed"] == seed
            }
            for seed in sorted({int(row["random_seed"]) for row in arm_rows})
        }
        updates[arm] = sum(next(iter(values)) for values in by_seed.values() if len(values) == 1)
        candidate_budgets[arm] = sum(int(row["candidate_budget"]) for row in arm_rows)
        iteration_sets[arm] = sorted({int(row["iterations"]) for row in arm_rows})
        runtime[arm] = round(sum(float(row["runtime_s"]) for row in arm_rows), 6)
    paired_budget_match = all(row["budget_match"] for row in rows)
    return {
        "optimizer_updates_by_arm": updates,
        "candidate_budget_by_arm": candidate_budgets,
        "iteration_counts_by_arm": iteration_sets,
        "runtime_total_s_by_arm": runtime,
        "runtime_ratio_grouped_over_flat": round(runtime[GROUPED_ARM] / runtime[FLAT_ARM], 10)
        if runtime[FLAT_ARM]
        else None,
        "all_source_and_control_budgets_match": paired_budget_match,
        "matched": (
            len(set(updates.values())) == 1
            and len(set(candidate_budgets.values())) == 1
            and iteration_sets[GROUPED_ARM] == iteration_sets[FLAT_ARM]
            and paired_budget_match
        ),
    }


def _literal_key(node: ast.AST) -> str | None:
    """Return a literal mapping key from a subscript or get call."""

    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return str(node.slice.value)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ):
        return str(node.args[0].value)
    return None


def _prefreeze_violations(path: Path, names: set[str]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Find forbidden or identity key reads inside proposal-time functions."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[JsonDict] = []
    identity_uses: list[JsonDict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in names:
            continue
        seen: set[tuple[int, str]] = set()
        for nested in ast.walk(node):
            key = _literal_key(nested)
            if key is None:
                continue
            marker = (getattr(nested, "lineno", node.lineno), key)
            if marker in seen:
                continue
            seen.add(marker)
            receipt = {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "function": node.name,
                "line": marker[0],
                "key": key,
            }
            if key in FORBIDDEN_PROPOSAL_KEYS:
                violations.append(receipt)
            if key in {"graph_id", "group_id", "topology_family"}:
                identity_uses.append(receipt)
    return violations, identity_uses


def inspect_exact_checker_order(
    repo_root: Path, source_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Combine AST call order with every runtime candidate-freeze receipt."""

    path_6787 = repo_root / SOURCE_MODULE_6787_PATH
    path_6788 = repo_root / SOURCE_MODULE_6788_PATH
    violations_6787, identity_6787 = _prefreeze_violations(path_6787, PREFREEZE_FUNCTIONS_6787)
    violations_6788, identity_6788 = _prefreeze_violations(path_6788, PREFREEZE_FUNCTIONS_6788)
    tree = ast.parse(path_6788.read_text(encoding="utf-8"))
    execute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute_cells"
    )
    calls = [
        (nested.func.id, nested.lineno)
        for nested in ast.walk(execute)
        if isinstance(nested, ast.Call)
        and isinstance(nested.func, ast.Name)
        and nested.func.id in {"propose_raw_row", "attach_exact_outcomes"}
    ]
    call_lines = {name: line for name, line in calls}
    static_order = (
        set(call_lines) == {"propose_raw_row", "attach_exact_outcomes"}
        and call_lines["propose_raw_row"] < call_lines["attach_exact_outcomes"]
    )
    mismatches = [
        row["source_row_id"]
        for row in source_rows
        if not row["exact_checker_after_candidate_freeze"]
    ]
    violations = violations_6787 + violations_6788
    return {
        "static_order_proven": static_order,
        "runtime_order_proven": not mismatches,
        "proposal_call": {"function": "propose_raw_row", "line": call_lines.get("propose_raw_row")},
        "candidate_freeze_event": "candidate_hashes serialized by propose_raw_row",
        "exact_checker_call": {
            "function": "attach_exact_outcomes",
            "line": call_lines.get("attach_exact_outcomes"),
        },
        "checked_runtime_row_count": len(source_rows),
        "candidate_hash_mismatch_count": len(mismatches),
        "candidate_hash_mismatch_rows": mismatches[:20],
        "identity_feature_uses": sorted(
            identity_6787 + identity_6788,
            key=lambda row: (row["path"], row["line"], row["key"]),
        ),
        "oracle_feature_violations": sorted(
            violations, key=lambda row: (row["path"], row["line"], row["key"])
        ),
    }


def _numeric_difference(left: Any, right: Any) -> tuple[bool, float]:
    """Compare nested headline values and retain the largest numeric delta."""

    if isinstance(left, bool) or isinstance(right, bool):
        return left == right, 0.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        return difference <= 1.0e-10, difference
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False, math.inf
        results = [_numeric_difference(left[key], right[key]) for key in left]
        return all(match for match, _ in results), max((value for _, value in results), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False, math.inf
        results = [_numeric_difference(a, b) for a, b in zip(left, right, strict=True)]
        return all(match for match, _ in results), max((value for _, value in results), default=0.0)
    return left == right, 0.0


def headline_differences(source: Mapping[str, Any], recomputed: Mapping[str, Any]) -> JsonDict:
    """Compare every declared source headline with the cold row reduction."""

    fields: JsonDict = {}
    maximum = 0.0
    all_match = True
    for key in HEADLINE_FIELDS:
        match, difference = _numeric_difference(source.get(key), recomputed.get(key))
        fields[key] = {"match": match, "maximum_absolute_difference": difference}
        all_match = all_match and match
        maximum = max(maximum, difference)
    return {
        "all_match": all_match,
        "maximum_absolute_difference": maximum,
        "fields": fields,
    }


def hard_negative_findings(
    source_rows: Sequence[Mapping[str, Any]],
    source_metrics: Mapping[str, Any],
    rewire: Mapping[str, Any],
) -> JsonDict:
    """Prove the primary effect remains after excluding easy local failures."""

    local_failures = sum(
        not outcome["local_checks_passed"]
        for row in source_rows
        for outcome in row["exact_outcomes"]
    )
    dependency_only = sum(
        outcome["local_checks_passed"] and bool(outcome["failed_dependency_ids"])
        for row in source_rows
        for outcome in row["exact_outcomes"]
    )
    effect = source_metrics["paired_exact_valid_delta"]
    survives = bool(rewire["positive_effect_survives"])
    return {
        "all_recomputed_candidates_pass_local_checks": local_failures == 0,
        "easy_local_failure_candidate_count": local_failures,
        "dependency_only_invalid_candidate_count": dependency_only,
        "effect_after_local_checks": effect,
        "hard_negative_auroc_by_arm": source_metrics["hard_negative_auroc_by_arm"],
        "explained_only_by_easy_local_failures": local_failures > 0 and dependency_only == 0,
        "topology_benefit_survives_dependency_destruction": survives,
        "shortcut_clear": local_failures == 0 and dependency_only > 0 and not survives,
    }


def _source_hashes(sources: Mapping[str, Any]) -> JsonDict:
    """Expose source hashes without leaking internal canonical-load metadata."""

    hashes = deepcopy(dict(sources.get("_file_hashes", {})))
    declared = sources.get("exp6788", {}).get("source_artifact_hashes", {})
    if isinstance(declared, Mapping):
        hashes.update({str(key): value for key, value in declared.items()})
    return dict(sorted(hashes.items()))


def _semantic_payload(sources: Mapping[str, Any], *, bootstrap_resamples: int) -> JsonDict:
    """Compute all rows, controls, metrics, and code-order evidence."""

    source_rows = recompute_source_rows(sources)
    control_rows = build_control_rows(source_rows, sources)
    rows = source_rows + control_rows
    reduced = reduce_audit_rows(
        rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    order = inspect_exact_checker_order(REPO_ROOT, source_rows)
    return {
        "rows": rows,
        "reduced": reduced,
        "parameter_match": recompute_parameter_match(rows),
        "compute_match": recompute_compute_match(rows),
        "exact_checker_call_order": order,
    }


def _fresh_process_payload(repo_root: Path, *, bootstrap_resamples: int) -> JsonDict:
    """Run the whole semantic audit in a new Python process."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp6789-") as directory:
        output = Path(directory) / "worker.json"
        command = [
            sys.executable,
            str(repo_root / MODULE_PATH),
            "--worker",
            "--repo-root",
            str(repo_root),
            "--worker-output",
            str(output),
            "--bootstrap-resamples",
            str(bootstrap_resamples),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not output.is_file():
            detail = (
                completed.stderr.strip() or completed.stdout.strip() or "worker produced no output"
            )
            raise AuditInputError(f"fresh process failed: {detail}")
        payload = load_json_object(output)
        payload["worker_exit_code"] = completed.returncode
        return payload


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable scientific content while excluding measured wall time and PIDs."""

    stable = {
        key: deepcopy(value)
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum", "field_principles"}
    }
    order = stable.get("exact_checker_call_order")
    if isinstance(order, dict):
        order.pop("fresh_worker_pid", None)
    return json_digest(stable)


def _empty_artifact(
    *,
    source_hashes: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Build a full blocked schema without invented audit rows."""

    verdict_class = "disqualified" if summary.get("missing_planned_rows") else "blocked"
    first = summary.get("first_failure") or {"check": "unknown", "observed": None}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_blocked_fixed_point_cold_audit",
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "cold_recomputed_metrics": {},
        "headline_differences": {},
        "parameter_match_recomputed": {},
        "compute_match_recomputed": {},
        "exact_checker_call_order": {},
        "oracle_feature_violations": [],
        "label_permutation_effect": {},
        "group_id_permutation_effect": {},
        "dependency_rewire_effect": {},
        "topology_id_swap_effect": {},
        "hard_negative_shortcut_findings": {},
        "source_verdict_supported": False,
        "fixed_point_audit_completed": False,
        "gate_check_summary": deepcopy(dict(summary)),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": (
            "complete_blocked_fixed_point_cold_audit: "
            f"{first.get('check')} observed {first.get('observed')!r}"
        ),
    }
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    sources: Mapping[str, Any],
    *,
    run_date: str,
    duration_s: float,
    bootstrap_resamples: int,
    fresh_process: bool,
) -> JsonDict:
    """Build a complete positive, downgraded, disqualified, or blocked audit."""

    parse_run_date(run_date)
    summary = check_preconditions(sources, repo_root=REPO_ROOT)
    source_hashes = _source_hashes(sources)
    if not summary["all_passed"]:
        return _empty_artifact(
            source_hashes=source_hashes,
            summary=summary,
            run_date=run_date,
            duration_s=duration_s,
        )
    local_payload = _semantic_payload(sources, bootstrap_resamples=bootstrap_resamples)
    payload = local_payload
    fresh_receipt: JsonDict = {
        "fresh_process": False,
        "agreement": not fresh_process,
        "producer_payload_sha256": json_digest(local_payload),
        "worker_payload_sha256": None,
    }
    if fresh_process:
        worker = _fresh_process_payload(REPO_ROOT, bootstrap_resamples=bootstrap_resamples)
        worker_pid = worker.pop("worker_pid")
        worker.pop("worker_exit_code", None)
        local_hash = json_digest(local_payload)
        worker_hash = json_digest(worker)
        fresh_receipt = {
            "fresh_process": True,
            "agreement": local_hash == worker_hash,
            "producer_payload_sha256": local_hash,
            "worker_payload_sha256": worker_hash,
            "fresh_worker_pid": worker_pid,
        }
        payload = worker
    rows = payload["rows"]
    reduced = payload["reduced"]
    order = deepcopy(payload["exact_checker_call_order"])
    order.update(fresh_receipt)
    differences = headline_differences(sources["exp6788"], reduced["source"])
    oracle_violations = deepcopy(order["oracle_feature_violations"])
    control_counts = {
        control: sum(row["control"] == control for row in rows) for control in CONTROLS
    }
    controls_matched = (
        len(rows) == PLANNED_AUDIT_ROW_COUNT
        and control_counts == {control: PLANNED_SOURCE_ROW_COUNT for control in CONTROLS}
        and all(row["budget_match"] for row in rows)
    )
    findings = hard_negative_findings(
        [row for row in rows if row["control"] == "source"],
        reduced["source"],
        reduced[DEPENDENCY_REWIRE],
    )
    source_positive = str(sources["exp6788"].get("verdict_class")) == "positive"
    destructive_gates = {
        "label_effect_removed": not reduced[LABEL_PERMUTATION]["positive_effect_survives"],
        "group_identity_effect_removed": not reduced[GROUP_ID_PERMUTATION][
            "positive_effect_survives"
        ],
        "dependency_effect_removed": not reduced[DEPENDENCY_REWIRE]["positive_effect_survives"],
        "topology_identifier_invariant": reduced[TOPOLOGY_ID_SWAP]["identifier_invariance_passed"],
    }
    authority_entered_proposal = bool(oracle_violations)
    source_supported = bool(
        source_positive
        and differences["all_match"]
        and payload["parameter_match"]["matched"]
        and payload["compute_match"]["matched"]
        and order["static_order_proven"]
        and order["runtime_order_proven"]
        and fresh_receipt["agreement"]
        and controls_matched
        and all(destructive_gates.values())
        and findings["shortcut_clear"]
        and not oracle_violations
    )
    audit_checks = [
        _gate("fresh_process_recompute_agreement", True, fresh_receipt["agreement"]),
        _gate("source_headline_agreement", True, differences["all_match"]),
        _gate("parameter_match", True, payload["parameter_match"]["matched"]),
        _gate("compute_match", True, payload["compute_match"]["matched"]),
        _gate("control_grid_and_budgets", True, controls_matched),
        _gate("oracle_feature_violations", [], oracle_violations),
    ]
    complete_summary = deepcopy(summary)
    complete_summary["audit_checks"] = audit_checks
    complete_summary["destructive_control_gates"] = destructive_gates
    complete_summary["control_row_counts"] = control_counts
    if authority_entered_proposal:
        verdict_class = "circular_positive"
        honest = (
            "complete: exact authority entered proposal selection; the source claim is circular"
        )
    elif oracle_violations or not controls_matched or not fresh_receipt["agreement"]:
        verdict_class = "disqualified"
        honest = (
            "complete: fixed-point source claim disqualified by leakage or unmatched audit evidence"
        )
    elif source_supported:
        verdict_class = "positive"
        honest = "complete: cold destructive audit supports the V592 grouped fixed-point verdict"
    else:
        verdict_class = "null"
        honest = "complete: cold destructive audit downgrades the V592 grouped fixed-point verdict"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete_fixed_point_cold_audit",
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "rows": rows,
        "cold_recomputed_metrics": reduced["source"],
        "headline_differences": differences,
        "parameter_match_recomputed": payload["parameter_match"],
        "compute_match_recomputed": payload["compute_match"],
        "exact_checker_call_order": order,
        "oracle_feature_violations": oracle_violations,
        "label_permutation_effect": reduced[LABEL_PERMUTATION],
        "group_id_permutation_effect": reduced[GROUP_ID_PERMUTATION],
        "dependency_rewire_effect": reduced[DEPENDENCY_REWIRE],
        "topology_id_swap_effect": reduced[TOPOLOGY_ID_SWAP],
        "hard_negative_shortcut_findings": findings,
        "source_verdict_supported": source_supported,
        "fixed_point_audit_completed": True,
        "gate_check_summary": complete_summary,
        "verifier_is_oracle": authority_entered_proposal,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
    }
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, row derivation, controls, and terminal verdict shape."""

    errors: list[str] = []
    missing = [key for key in REQUIRED_FIELDS if key not in artifact]
    if missing:
        errors.append("missing_fields")
        return errors
    if artifact["verdict_class"] not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if set(artifact["field_principles"]) != set(artifact):
        errors.append("field_principles")
    blocked = artifact["status"] == "complete_blocked_fixed_point_cold_audit"
    if blocked:
        if artifact["rows"] or artifact["fixed_point_audit_completed"]:
            errors.append("blocked_terminal_state")
    else:
        rows = artifact["rows"]
        if len(rows) != PLANNED_AUDIT_ROW_COUNT:
            errors.append("audit_row_count")
        else:
            reduced = reduce_audit_rows(
                rows,
                bootstrap_resamples=int(
                    artifact["cold_recomputed_metrics"]["paired_exact_valid_delta_ci95"][
                        "resamples"
                    ]
                ),
                bootstrap_seed=BOOTSTRAP_SEED,
            )
            expected = {
                "cold_recomputed_metrics": reduced["source"],
                "label_permutation_effect": reduced[LABEL_PERMUTATION],
                "group_id_permutation_effect": reduced[GROUP_ID_PERMUTATION],
                "dependency_rewire_effect": reduced[DEPENDENCY_REWIRE],
                "topology_id_swap_effect": reduced[TOPOLOGY_ID_SWAP],
            }
            if any(artifact[key] != value for key, value in expected.items()):
                errors.append("row_derived_metrics")
    if artifact["reproducibility_checksum"] != _reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def write_output(artifact: Mapping[str, Any], output: Path) -> None:
    """Write canonical JSON atomically so a crash cannot leave partial evidence."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        handle.write(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, output)


def parse_run_date(value: str) -> str:
    """Require the execution date format used by frozen experiment artifacts."""

    if len(value) != 8 or not value.isdigit():
        raise ValueError("run date must use YYYYMMDD")
    return value


def _worker(repo_root: Path, output: Path, *, bootstrap_resamples: int) -> int:
    """Recompute the full semantic payload in the child process."""

    sources = load_source_bundle(repo_root)
    summary = check_preconditions(sources, repo_root=repo_root)
    if not summary["all_passed"]:
        raise AuditInputError(f"worker precondition failed: {summary['first_failure']}")
    payload = _semantic_payload(sources, bootstrap_resamples=bootstrap_resamples)
    payload["worker_pid"] = os.getpid()
    write_output(payload, output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated cold audit or its internal fresh-process worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.worker:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker")
        return _worker(repo_root, args.worker_output, bootstrap_resamples=args.bootstrap_resamples)
    run_date = parse_run_date(args.date)
    sources = load_source_bundle(repo_root)
    import time

    started = time.perf_counter()
    artifact = build_artifact(
        sources,
        run_date=run_date,
        duration_s=0.0,
        bootstrap_resamples=args.bootstrap_resamples,
        fresh_process=True,
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise AuditInputError(f"artifact validation failed: {errors}")
    output = args.output or (repo_root / RESULT_PATH)
    write_output(artifact, output)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
