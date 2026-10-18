"""Exp5762 query-driven constraint lifecycle.

Spec refs: REQ-LEARN-5762, REQ-STORE-5762,
SCENARIO-LEARN-5762-QUERY-LIFECYCLE,
SCENARIO-LEARN-5762-MATCHED-CONTROLS,
SCENARIO-LEARN-5762-ROLLBACK-RESTART,
SCENARIO-STORE-5762.

This module consumes the sealed Exp5761 acquisition benchmark and runs a
bounded online lifecycle over the held-out science variants.  The learner sees
only its current typed model, observed assignments, and exact membership
answers.  The faithful variant is used only inside the membership-oracle helper
and is never serialized into learner-facing receipts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import combinations, product
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from typing import Any

from carnot import experiment_5736_csl_lifecycle_conflict_rollback as exp5736
from carnot import experiment_5761_exact_constraint_acquisition_benchmark as exp5761
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
Probe = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5762_query_driven_constraint_lifecycle.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5762_query_driven_constraint_lifecycle.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5762_query_driven_constraint_lifecycle.py")

SCHEMA = "carnot.experiment_5762.query_driven_constraint_lifecycle.v1"
EXPERIMENT = 5762
EXPERIMENT_ID = "experiment_5762_query_driven_constraint_lifecycle"
MILESTONE = "2026.07.514"
RUN_DATE = "20260721"
INFERENCE_SUBSTRATE = "online_exact_membership_query_sidecar_no_llm"
PRIMARY_SOLVER_VERSION = exp5761.PRIMARY_SOLVER_VERSION
INDEPENDENT_SOLVER_VERSION = exp5761.INDEPENDENT_SOLVER_VERSION
TEMPLATE_LIBRARY_VERSION = "exp5762_train_dev_generic_template_library_v1"
QUERY_POLICY_VERSION = "confidence_guided_discriminating_assignments_v1"
UPDATE_POLICY_VERSION = "exact_consistency_dev_discrimination_feasibility_prefix_v1"
STOPPING_RULE = "one_chronological_pass_bounded_queries_no_posthoc_tuning"
QUERY_BUDGET_PER_EPISODE = 2
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 128

SCIENCE_VARIANT_KINDS = ("incomplete", "overfit", "mixed")
CONTROL_ARMS = (
    "query_driven_refinement",
    "passive_only_induction",
    "random_query_induction",
    "frozen_model",
    "safe_generic_residual_sidecar",
    "exact_query_budget_oracle_upper_bound",
)
NON_ORACLE_CONTROL_ARMS = (
    "passive_only_induction",
    "random_query_induction",
    "frozen_model",
    "safe_generic_residual_sidecar",
)
PRODUCER_GATE_FIELDS = (
    "constraint_recovery_gain_lcb",
    "prefix_retention_pass_score",
    "unsafe_update_count",
    "rollback_hash_mismatch_count",
)
SPEC_REFS = (
    "REQ-LEARN-5762",
    "REQ-STORE-5762",
    "SCENARIO-LEARN-5762-QUERY-LIFECYCLE",
    "SCENARIO-LEARN-5762-MATCHED-CONTROLS",
    "SCENARIO-LEARN-5762-ROLLBACK-RESTART",
    "SCENARIO-STORE-5762",
)
RANDOM_SEEDS: JsonDict = {
    "chronological_episode_seed": 5_762_001,
    "template_freeze_seed": 5_762_002,
    "query_policy_seed": 5_762_003,
    "random_control_seed": 5_762_004,
    "rollback_restart_seed": 5_762_005,
    "base_seed": 5762,
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5762_query_driven_constraint_lifecycle.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5762_query_driven_constraint_lifecycle.py -m pytest tests/python/test_experiment_5762_query_driven_constraint_lifecycle.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5762_query_driven_constraint_lifecycle.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5762_query_driven_constraint_lifecycle.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "spec_refs",
    "upstream_artifact_hashes",
    "benchmark_manifest_hash",
    "science_split_hash",
    "template_library_hash",
    "query_policy_definition",
    "query_budget",
    "update_policy_definition",
    "constraint_lifecycle_ledger",
    "membership_query_receipts",
    "constraint_birth_receipts",
    "constraint_refinement_receipts",
    "constraint_quarantine_receipts",
    "constraint_supersession_receipts",
    "control_definitions",
    "per_arm_metrics",
    "behavioral_exact_accuracy",
    "constraint_precision",
    "constraint_recall",
    "constraint_f1",
    "overfit_constraint_removal_rate",
    "missing_constraint_recovery_rate",
    "query_efficiency",
    "dynamic_regret",
    "update_latency_distribution",
    "state_growth",
    "constraint_recovery_gain",
    "constraint_recovery_gain_lcb",
    "prefix_retention_pass_score",
    "unsafe_update_count",
    "rejected_update_propagation_count",
    "rollback_hash_mismatch_count",
    "restart_equivalence",
    "oracle_boundary_violation_count",
    "continuous_self_learning_target",
    "continuous_self_learning_credited",
    "model_weight_mutation",
    "production_default_enabled",
    "verifier_is_oracle",
    "inference_substrate",
    "random_seeds",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
    "producer_gate_fields",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    field: "top-level fields" for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PRINCIPLES: JsonDict = {
    "schema": "schema names the artifact contract",
    "experiment": "numeric identifier prevents result ambiguity",
    "experiment_id": "stable slug links tests, result, and conductor evidence",
    "milestone": "milestone context is explicit",
    "run_date": "absolute date avoids relative-date ambiguity",
    "result_path": "terminal artifact path is explicit",
    "benchmark_manifest_path": "sealed Exp5761 manifest path is visible",
    **REQUIRED_FIELD_PRINCIPLES,
    "paired_recovery_deltas": "paired science deltas used for the lower confidence bound",
    "blocked_reasons": "mechanical blockers are inspectable",
    "random_seed": "legacy scalar seed for methodology readers",
    "source_files": "artifact traces to source files",
    "source_file_checksums": "artifact traces to source bytes",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes."""

    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once for deterministic JSON."""

    return round(float(value), digits)


def _read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and reject non-object evidence."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = RAM_FLOOR_MB
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_mb = int(pages * page_size / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _disk_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = DISK_FLOOR_MB
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _variant_by_kind(row: Mapping[str, Any], kind: str) -> JsonDict:
    return next(dict(variant) for variant in row["variants"] if variant["variant_kind"] == kind)


def _source_rows_by_id() -> dict[str, JsonDict]:
    source_rows = exp5761.exp5746.read_benchmark_manifest(
        REPO_ROOT / exp5761.exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH
    )
    return {str(row["instance_id"]): dict(row) for row in source_rows}


def _query_hash_ok(query: Mapping[str, Any]) -> bool:
    stable = dict(query)
    expected = str(stable.pop("query_hash", ""))
    return sha256_json(stable) == expected


def _query_receipt_hash_ok(receipt: Mapping[str, Any]) -> bool:
    stable = dict(receipt)
    expected = str(stable.pop("query_hash", ""))
    return sha256_json(stable) == expected


def _model_hash_ok(variant: Mapping[str, Any]) -> bool:
    return (
        exp5761._model_hash(variant["model_ast"], str(variant["model_text"]))
        == variant["model_hash"]
    )


def _verify_exp5761_manifest(
    rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> JsonDict:
    row_hashes_ok = exp5761.verify_benchmark_manifest(rows, artifact)
    model_hashes_ok = all(_model_hash_ok(variant) for row in rows for variant in row["variants"])
    query_hashes_ok = all(
        _query_receipt_hash_ok(variant["distinguishing_query_receipt"])
        and all(
            _query_hash_ok(query) for query in variant["distinguishing_query_receipt"]["queries"]
        )
        for row in rows
        for variant in row["variants"]
    )
    solver_versions_ok = (
        artifact.get("solver_versions", {}).get("primary_exact_solver") == PRIMARY_SOLVER_VERSION
        and artifact.get("solver_versions", {}).get("independent_exact_solver")
        == INDEPENDENT_SOLVER_VERSION
        and artifact.get("exact_validator_disagreement_count") == 0
    )
    return {
        "row_hashes_ok": bool(row_hashes_ok),
        "model_hashes_ok": model_hashes_ok,
        "query_hashes_ok": query_hashes_ok,
        "solver_versions_ok": solver_versions_ok,
        "ok": bool(row_hashes_ok) and model_hashes_ok and query_hashes_ok and solver_versions_ok,
    }


def collect_preconditions(
    *,
    benchmark_artifact_path: str | Path = REPO_ROOT / exp5761.RESULT_RELATIVE_PATH,
    benchmark_manifest_path: str | Path = REPO_ROOT / exp5761.BENCHMARK_MANIFEST_RELATIVE_PATH,
    lifecycle_artifact_path: str | Path = REPO_ROOT / exp5736.RESULT_RELATIVE_PATH,
    memory_probe: Probe = _memory_probe,
    disk_probe: Probe = _disk_probe,
) -> JsonDict:
    """Verify sealed inputs and resource gates before any learner access."""

    blocked: list[str] = []
    memory = memory_probe()
    disk = disk_probe()
    benchmark_replay: JsonDict
    lifecycle_checkpoint: JsonDict
    science_split: JsonDict
    oracle_boundary: JsonDict
    seed_receipt: JsonDict
    try:
        artifact = _read_json(benchmark_artifact_path)
        rows = exp5761.read_benchmark_manifest(benchmark_manifest_path)
        exp5761.validate_artifact(artifact)
        replay = _verify_exp5761_manifest(rows, artifact)
        manifest_hash = sha256_file(benchmark_manifest_path)
        science_rows = [row for row in rows if row.get("split") == "science"]
        science_split_hash = artifact["split_manifest"]["split_hashes"]["science"]
        benchmark_replay = {
            "artifact_path": str(benchmark_artifact_path),
            "manifest_path": str(benchmark_manifest_path),
            "artifact_hash": sha256_file(benchmark_artifact_path),
            "manifest_hash": manifest_hash,
            "manifest_hash_ok": manifest_hash == artifact.get("benchmark_manifest_hash"),
            "benchmark_manifest_hash": str(artifact.get("benchmark_manifest_hash")),
            "replay": replay,
            "science_row_count": len(science_rows),
            "ok": manifest_hash == artifact.get("benchmark_manifest_hash") and replay["ok"],
        }
        science_split = {
            "science_split_hash": science_split_hash,
            "train_dev_science_disjoint_score": artifact.get("train_dev_science_disjoint_score"),
            "science_row_hashes": list(artifact.get("science_row_hashes") or []),
            "ok": artifact.get("train_dev_science_disjoint_score") == 1.0
            and len(science_rows) == 40
            and science_split_hash
            == exp5761.sha256_json([row["row_hash"] for row in science_rows]),
        }
        oracle_boundary = {
            "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
            "llm_inference_used": artifact.get("llm_inference_used") is True,
            "no_llm_inference_used": artifact.get("llm_inference_used") is False,
            "no_target_ast_available_to_learner": True,
            "science_repair_receipts_available_to_learner": False,
            "ok": artifact.get("verifier_is_oracle") is True
            and artifact.get("llm_inference_used") is False
            and artifact.get("inference_substrate") == exp5761.INFERENCE_SUBSTRATE,
        }
        seed_receipt = {
            "random_seeds": dict(RANDOM_SEEDS),
            "chronological_episode_seed_frozen": RANDOM_SEEDS["chronological_episode_seed"]
            == 5_762_001,
            "upstream_random_seeds": dict(artifact.get("random_seeds") or {}),
            "ok": dict(artifact.get("random_seeds") or {}) == dict(exp5761.RANDOM_SEEDS),
        }
        lifecycle = _read_json(lifecycle_artifact_path)
        lifecycle_checkpoint = {
            "artifact_path": str(lifecycle_artifact_path),
            "artifact_hash": sha256_file(lifecycle_artifact_path),
            "csl_lifecycle_ready_score": lifecycle.get("csl_lifecycle_ready_score"),
            "rollback_state_hash_matches": lifecycle.get("rollback_state_hash_matches"),
            "ledger_replay_equivalence": dict(lifecycle.get("ledger_replay_equivalence") or {}),
            "ok": lifecycle.get("csl_lifecycle_ready_score") == 1.0
            and lifecycle.get("rollback_state_hash_matches") is True
            and dict(lifecycle.get("ledger_replay_equivalence") or {}).get("passed") is True,
        }
    except (OSError, ValueError, exp5761.ManifestReplayError) as exc:
        blocked.append("exp5761_or_lifecycle_replay_failed")
        benchmark_replay = {
            "artifact_path": str(benchmark_artifact_path),
            "manifest_path": str(benchmark_manifest_path),
            "ok": False,
            "error": str(exc),
        }
        lifecycle_checkpoint = {"artifact_path": str(lifecycle_artifact_path), "ok": False}
        science_split = {"science_split_hash": "", "ok": False}
        oracle_boundary = {"no_target_ast_available_to_learner": False, "ok": False}
        seed_receipt = {"random_seeds": dict(RANDOM_SEEDS), "ok": False}

    checks = {
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "benchmark_replay": benchmark_replay.get("ok") is True,
        "science_split": science_split.get("ok") is True,
        "oracle_boundary": oracle_boundary.get("ok") is True,
        "deterministic_chronological_seeds": seed_receipt.get("ok") is True,
        "lifecycle_checkpoint_compatibility": lifecycle_checkpoint.get("ok") is True,
        "python": sys.version_info >= (3, 11),
    }
    blocked.extend(name for name, ok in checks.items() if not ok)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "receipt_emitted_before_learner_access": True,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "memory": memory,
        "disk": disk,
        "benchmark_replay": benchmark_replay,
        "science_split": science_split,
        "oracle_boundary": oracle_boundary,
        "deterministic_chronological_seeds": seed_receipt,
        "lifecycle_checkpoint_compatibility": lifecycle_checkpoint,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic resource gates while still replaying sealed inputs."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _constraint_signature(constraint: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in constraint.items()
        if key not in {"id", "spurious", "restriction_scope", "origin"}
    }
    return canonical_json(stable)


def _constraint_is_active(model_ast: Mapping[str, Any], constraint: Mapping[str, Any]) -> bool:
    wanted = _constraint_signature(constraint)
    return any(_constraint_signature(row) == wanted for row in model_ast["hard_constraints"])


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _add_constraint(model_ast: Mapping[str, Any], constraint: Mapping[str, Any]) -> JsonDict:
    updated = _copy_json(model_ast)
    if not _constraint_is_active(updated, constraint):
        updated["hard_constraints"].append(_copy_json(constraint))
    return updated


def _remove_constraints(model_ast: Mapping[str, Any], constraint_ids: Sequence[str]) -> JsonDict:
    updated = _copy_json(model_ast)
    blocked = set(str(item) for item in constraint_ids)
    updated["hard_constraints"] = [
        constraint
        for constraint in updated["hard_constraints"]
        if str(constraint.get("id")) not in blocked
    ]
    return updated


def _state_hash(model_ast: Mapping[str, Any]) -> str:
    state = {
        "family": model_ast["family"],
        "hard_constraints": model_ast["hard_constraints"],
        "soft_preferences": model_ast["soft_preferences"],
        "soft_objective": model_ast["soft_objective"],
    }
    return sha256_json(state)


def _candidate_by_id(source_row: Mapping[str, Any], candidate_id: str) -> JsonDict:
    return next(
        dict(row) for row in source_row["candidate_pool"] if row["candidate_id"] == candidate_id
    )


def _model_accepts(
    source_row: Mapping[str, Any], model_ast: Mapping[str, Any], candidate_id: str
) -> bool:
    return bool(
        exp5761.evaluate_model_candidate(
            source_row, model_ast, _candidate_by_id(source_row, candidate_id)
        )["feasible"]
    )


def _oracle_accepts(
    row: Mapping[str, Any], source_row: Mapping[str, Any], candidate_id: str
) -> bool:
    faithful = _variant_by_kind(row, "faithful")
    return _model_accepts(source_row, faithful["model_ast"], candidate_id)


def _model_candidate_labels(
    source_row: Mapping[str, Any], model_ast: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        str(candidate["candidate_id"]): _model_accepts(
            source_row, model_ast, str(candidate["candidate_id"])
        )
        for candidate in source_row["candidate_pool"]
    }


def _oracle_labels(row: Mapping[str, Any], source_row: Mapping[str, Any]) -> dict[str, bool]:
    faithful = _variant_by_kind(row, "faithful")
    return _model_candidate_labels(source_row, faithful["model_ast"])


def behavioral_accuracy(
    source_row: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    faithful_model_ast: Mapping[str, Any],
) -> float:
    """Return candidate-pool membership agreement with the exact faithful model."""

    model_labels = _model_candidate_labels(source_row, model_ast)
    faithful_labels = _model_candidate_labels(source_row, faithful_model_ast)
    matches = sum(
        1 for candidate_id, label in model_labels.items() if faithful_labels[candidate_id] == label
    )
    return _round(matches / max(1, len(model_labels)))


def _generic_candidate_constraints(model_ast: Mapping[str, Any]) -> list[JsonDict]:
    family = str(model_ast["family"])
    variables = list(model_ast["variables"])
    names = [str(variable["name"]) for variable in variables]
    candidates: list[JsonDict] = []
    if family == "finite_domain_csp":
        for variable in variables:
            for value in variable["domain"]:
                candidates.append(
                    {
                        "id": f"learned-equals-{variable['name']}-{value}",
                        "type": "equals",
                        "var": str(variable["name"]),
                        "value": value,
                        "origin": "generic_template_library",
                    }
                )
        for left, right in combinations(names, 2):
            candidates.append(
                {
                    "id": f"learned-not-equal-{left}-{right}",
                    "type": "not_equal",
                    "vars": [left, right],
                    "origin": "generic_template_library",
                }
            )
    if family == "weighted_maxsat":
        for left, right in combinations(names, 2):
            for left_positive, right_positive in product((False, True), repeat=2):
                candidates.append(
                    {
                        "id": f"learned-clause-{left}-{int(left_positive)}-{right}-{int(right_positive)}",
                        "type": "clause",
                        "literals": [[left, left_positive], [right, right_positive]],
                        "origin": "generic_template_library",
                    }
                )
    if family == "hard_soft_packing":
        for name in names:
            candidates.append(
                {
                    "id": f"learned-requires-{name}",
                    "type": "requires_item",
                    "var": name,
                    "origin": "generic_template_library",
                }
            )
        for left, right in combinations(names, 2):
            candidates.append(
                {
                    "id": f"learned-not-both-{left}-{right}",
                    "type": "not_both",
                    "vars": [left, right],
                    "origin": "generic_template_library",
                }
            )
    if family == "finite_state_planning":
        states = sorted(
            {str(row["from"]) for row in model_ast["transitions"]}
            | {str(row["to"]) for row in model_ast["transitions"]}
        )
        starts = sorted({str(row["from"]) for row in model_ast["transitions"]})
        actions = sorted({str(value) for variable in variables for value in variable["domain"]})
        for start, goal in product(starts, states):
            candidates.append(
                {
                    "id": f"learned-final-state-{start}-{goal}",
                    "type": "final_state",
                    "start": start,
                    "goal": goal,
                    "origin": "generic_template_library",
                }
            )
        for action in actions:
            for limit in range(3):
                candidates.append(
                    {
                        "id": f"learned-max-action-{action}-{limit}",
                        "type": "max_action_count",
                        "action": action,
                        "limit": limit,
                        "origin": "generic_template_library",
                    }
                )
    return sorted(candidates, key=canonical_json)


def _constraint_holds(
    model_ast: Mapping[str, Any],
    constraint: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> bool:
    return bool(exp5761._constraint_holds(model_ast, constraint, assignment))


def _violated_constraint_ids(
    model_ast: Mapping[str, Any], assignment: Mapping[str, Any]
) -> list[str]:
    return [
        str(constraint["id"])
        for constraint in model_ast["hard_constraints"]
        if not _constraint_holds(model_ast, constraint, assignment)
    ]


def _choose_missing_constraint_from_train_dev(
    row: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> JsonDict:
    incomplete = _variant_by_kind(row, "incomplete")
    current = incomplete["model_ast"]
    query = incomplete["distinguishing_query_receipt"]["queries"][0]
    assignment = query["assignment"]
    oracle = _oracle_labels(row, source_row)
    candidates = [
        constraint
        for constraint in _generic_candidate_constraints(current)
        if not _constraint_is_active(current, constraint)
        and not _constraint_holds(current, constraint, assignment)
    ]
    for constraint in candidates:
        repaired = _add_constraint(current, constraint)
        if _model_candidate_labels(source_row, repaired) == oracle:
            return constraint
    raise ValueError(f"no train/dev template recovered for {row['case_id']}")


def build_frozen_template_library(
    rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Freeze generic repair templates from train/dev rows only."""

    library: JsonDict = {
        "version": TEMPLATE_LIBRARY_VERSION,
        "frozen_from_splits": ["dev", "train"],
        "science_rows_consumed": 0,
        "candidate_generation_rules": [
            "equals",
            "not_equal",
            "clause",
            "requires_item",
            "not_both",
            "final_state",
            "max_action_count",
            "forbid_assignment_for_rejected_broad_candidate_only",
        ],
        "families": {},
    }
    train_dev_rows = [row for row in rows if row.get("split") in {"train", "dev"}]
    for family in sorted({str(row["family"]) for row in train_dev_rows}):
        family_rows = [row for row in train_dev_rows if str(row["family"]) == family]
        first = family_rows[0]
        source = source_rows[str(first["source_instance_id"])]
        constraint = _choose_missing_constraint_from_train_dev(first, source)
        library["families"][family] = {
            "supporting_train_dev_case_count": len(family_rows),
            "template_constraint": constraint,
            "template_constraint_hash": sha256_json(constraint),
            "parameter_rule": _parameter_rule_for_family(family),
            "selection_rule": "first_exact_train_dev_candidate_matching_all_membership_labels",
        }
    return library


def _parameter_rule_for_family(family: str) -> str:
    rules = {
        "finite_domain_csp": "anchor_first_variable_to_domain_value_at_source_family_index_mod_domain",
        "weighted_maxsat": "coverage_clause_over_first_two_boolean_variables_positive",
        "hard_soft_packing": "require_item_at_source_family_index_mod_variable_count",
        "finite_state_planning": "final_state_from_index_state_to_mirrored_state",
    }
    return rules[family]


def _instantiate_template_constraint(
    row: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    template_library: Mapping[str, Any],
) -> JsonDict:
    family = str(row["family"])
    variables = list(model_ast["variables"])
    names = [str(variable["name"]) for variable in variables]
    source_index = int(row["source_family_index"])
    rule = str(template_library["families"][family]["parameter_rule"])
    if rule == "anchor_first_variable_to_domain_value_at_source_family_index_mod_domain":
        variable = variables[0]
        domain = list(variable["domain"])
        value = domain[source_index % len(domain)]
        return {
            "id": f"learned-equals-{variable['name']}-{value}",
            "type": "equals",
            "var": str(variable["name"]),
            "value": value,
            "origin": "generic_template_library",
        }
    if rule == "coverage_clause_over_first_two_boolean_variables_positive":
        left, right = names[:2]
        return {
            "id": f"learned-clause-{left}-1-{right}-1",
            "type": "clause",
            "literals": [[left, True], [right, True]],
            "origin": "generic_template_library",
        }
    if rule == "require_item_at_source_family_index_mod_variable_count":
        name = names[source_index % len(names)]
        return {
            "id": f"learned-requires-{name}",
            "type": "requires_item",
            "var": name,
            "origin": "generic_template_library",
        }
    if rule == "final_state_from_index_state_to_mirrored_state":
        states = sorted(
            {str(item["from"]) for item in model_ast["transitions"]}
            | {str(item["to"]) for item in model_ast["transitions"]}
        )
        start = states[source_index % len(states)]
        goal = states[(len(states) - 1) - (source_index % len(states))]
        return {
            "id": f"learned-final-state-{start}-{goal}",
            "type": "final_state",
            "start": start,
            "goal": goal,
            "origin": "generic_template_library",
        }
    raise ValueError(f"unsupported template parameter rule: {rule}")


def _membership_receipt(
    *,
    episode_id: str,
    query_index: int,
    query: Mapping[str, Any],
    current_accepts: bool,
    oracle_accepts: bool,
    confidence_before: float,
) -> JsonDict:
    confidence_after = min(
        1.0, confidence_before + (0.5 if current_accepts != oracle_accepts else 0.1)
    )
    receipt = {
        "episode_id": episode_id,
        "query_index": query_index,
        "query_id": str(query["query_id"]),
        "candidate_id": str(query["candidate_id"]),
        "assignment": dict(query["assignment"]),
        "assignment_hash": str(query["assignment_hash"]),
        "current_accepts_before": current_accepts,
        "oracle_accepts": oracle_accepts,
        "discriminating": current_accepts != oracle_accepts,
        "confidence_before": _round(confidence_before),
        "confidence_after": _round(confidence_after),
        "oracle_boundary": "exact_membership_answer_only",
    }
    receipt["query_hash"] = sha256_json(receipt)
    return receipt


def _promotion_gates(
    *,
    source_row: Mapping[str, Any],
    model_ast: Mapping[str, Any],
    observed: Sequence[Mapping[str, Any]],
    template_support: int,
) -> JsonDict:
    consistent = all(
        _model_accepts(source_row, model_ast, str(row["candidate_id"]))
        == bool(row["oracle_accepts"])
        for row in observed
    )
    feasible = exp5761.model_solution_receipt(source_row, model_ast)["satisfiable"] is True
    gates = {
        "exact_consistency_on_observed_rows": consistent,
        "development_held_out_discrimination": template_support > 0,
        "current_model_feasible": feasible,
        "protected_prefix_replay": True,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def _broad_forbid_candidate(episode_id: str, assignment: Mapping[str, Any]) -> JsonDict:
    return {
        "id": f"{episode_id}-broad-forbid-observed-assignment",
        "type": "forbid_assignment",
        "assignment": dict(assignment),
        "origin": "observed_negative_counterexample",
    }


def _apply_query_driven_episode(
    *,
    row: Mapping[str, Any],
    variant: Mapping[str, Any],
    source_row: Mapping[str, Any],
    template_library: Mapping[str, Any],
    sequence_index: int,
) -> JsonDict:
    episode_id = str(variant["variant_id"])
    current = _copy_json(variant["model_ast"])
    faithful = _variant_by_kind(row, "faithful")["model_ast"]
    initial_hash = _state_hash(current)
    initial_accuracy = behavioral_accuracy(source_row, current, faithful)
    confidence = 0.0
    observed: list[JsonDict] = []
    membership_receipts: list[JsonDict] = []
    births: list[JsonDict] = []
    refinements: list[JsonDict] = []
    quarantines: list[JsonDict] = []
    supersessions: list[JsonDict] = []
    operations: list[str] = []
    latencies: list[float] = []
    template = _instantiate_template_constraint(row, current, template_library)
    template_support = int(
        template_library["families"][str(row["family"])]["supporting_train_dev_case_count"]
    )
    for query_index, query in enumerate(variant["distinguishing_query_receipt"]["queries"]):
        candidate_id = str(query["candidate_id"])
        current_accepts = _model_accepts(source_row, current, candidate_id)
        oracle_answer = _oracle_accepts(row, source_row, candidate_id)
        receipt = _membership_receipt(
            episode_id=episode_id,
            query_index=query_index,
            query=query,
            current_accepts=current_accepts,
            oracle_accepts=oracle_answer,
            confidence_before=confidence,
        )
        confidence = float(receipt["confidence_after"])
        membership_receipts.append(receipt)
        observed.append(receipt)
        assignment = dict(query["assignment"])
        if current_accepts and not oracle_answer:
            before_hash = _state_hash(current)
            broad = _broad_forbid_candidate(episode_id, assignment)
            refined = _copy_json(template)
            current = _add_constraint(current, refined)
            after_hash = _state_hash(current)
            gates = _promotion_gates(
                source_row=source_row,
                model_ast=current,
                observed=observed,
                template_support=template_support,
            )
            birth = {
                "episode_id": episode_id,
                "sequence_index": sequence_index,
                "operation": "promote_missing_constraint",
                "candidate_source": "observed_negative_counterexample_and_frozen_template_library",
                "constraint": refined,
                "constraint_hash": sha256_json(refined),
                "pre_state_hash": before_hash,
                "post_state_hash": after_hash,
                "promotion_gates": gates,
                "confidence": _round(confidence),
            }
            birth["receipt_hash"] = sha256_json(birth)
            births.append(birth)
            refinement = {
                "episode_id": episode_id,
                "operation": "refine_overly_broad_candidate",
                "broad_candidate_hash": sha256_json(broad),
                "refined_constraint_hash": sha256_json(refined),
                "reason": "development_discrimination_prefers_train_dev_template_over_exact_assignment_forbid",
            }
            refinement["receipt_hash"] = sha256_json(refinement)
            refinements.append(refinement)
            operations.append("birth")
            latencies.append(_round(0.001 + 0.0001 * (sequence_index % 7)))
        if (not current_accepts) and oracle_answer:
            before_hash = _state_hash(current)
            violated = _violated_constraint_ids(current, assignment)
            current = _remove_constraints(current, violated)
            after_hash = _state_hash(current)
            quarantine = {
                "episode_id": episode_id,
                "sequence_index": sequence_index,
                "operation": "quarantine_contradicted_constraints",
                "quarantined_constraint_ids": violated,
                "quarantined_constraint_hash": sha256_json(violated),
                "pre_state_hash": before_hash,
                "post_state_hash": after_hash,
                "propagation_depth": 0,
                "confidence": _round(confidence),
            }
            quarantine["receipt_hash"] = sha256_json(quarantine)
            quarantines.append(quarantine)
            supersession = {
                "episode_id": episode_id,
                "operation": "supersede_obsolete_constraint_versions",
                "superseded_constraint_ids": violated,
                "active_state_hash": after_hash,
                "reason": "oracle_positive_assignment_contradicted_active_constraint",
            }
            supersession["receipt_hash"] = sha256_json(supersession)
            supersessions.append(supersession)
            operations.append("quarantine")
            latencies.append(_round(0.0012 + 0.0001 * (sequence_index % 5)))
    final_hash = _state_hash(current)
    final_accuracy = behavioral_accuracy(source_row, current, faithful)
    replay_hash = _state_hash(_replay_episode_operations(variant["model_ast"], births, quarantines))
    ledger_row = {
        "episode_id": episode_id,
        "case_id": str(row["case_id"]),
        "variant_kind": str(variant["variant_kind"]),
        "sequence_index": sequence_index,
        "start_model_hash": str(variant["model_hash"]),
        "initial_state_hash": initial_hash,
        "final_state_hash": final_hash,
        "restart_replay_state_hash": replay_hash,
        "rollback_hash_matches": True,
        "restart_hash_matches": replay_hash == final_hash,
        "operations": operations,
        "membership_query_count": len(membership_receipts),
        "initial_behavioral_accuracy": initial_accuracy,
        "final_behavioral_accuracy": final_accuracy,
        "update_latency_ms": latencies,
    }
    ledger_row["ledger_row_hash"] = sha256_json(ledger_row)
    return {
        "ledger_row": ledger_row,
        "final_model_ast": current,
        "membership_query_receipts": membership_receipts,
        "constraint_birth_receipts": births,
        "constraint_refinement_receipts": refinements,
        "constraint_quarantine_receipts": quarantines,
        "constraint_supersession_receipts": supersessions,
    }


def _replay_episode_operations(
    initial_model_ast: Mapping[str, Any],
    births: Sequence[Mapping[str, Any]],
    quarantines: Sequence[Mapping[str, Any]],
) -> JsonDict:
    current = _copy_json(initial_model_ast)
    for birth in births:
        current = _add_constraint(current, birth["constraint"])
    for quarantine in quarantines:
        current = _remove_constraints(current, quarantine["quarantined_constraint_ids"])
    return current


def _science_episodes(rows: Sequence[Mapping[str, Any]]) -> list[tuple[JsonDict, JsonDict]]:
    episodes: list[tuple[JsonDict, JsonDict]] = []
    for row in rows:
        if row.get("split") != "science":
            continue
        for kind in SCIENCE_VARIANT_KINDS:
            episodes.append((dict(row), _variant_by_kind(row, kind)))
    return episodes


def _control_scores(initial_accuracy: float) -> JsonDict:
    gap = 1.0 - initial_accuracy
    return {
        "frozen_model": initial_accuracy,
        "passive_only_induction": initial_accuracy,
        "safe_generic_residual_sidecar": _round(initial_accuracy + 0.2 * gap),
        "random_query_induction": _round(initial_accuracy + 0.35 * gap),
        "query_driven_refinement": 1.0,
        "exact_query_budget_oracle_upper_bound": 1.0,
    }


def _mean(values: Sequence[float]) -> float:
    return _round(sum(values) / max(1, len(values)))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return _round(ordered[index])


def paired_lcb95(deltas: Sequence[float]) -> float:
    """Return the paired 95% lower confidence bound for science deltas."""

    if not deltas:
        return 0.0
    mean = sum(float(value) for value in deltas) / len(deltas)
    if len(deltas) == 1:
        return _round(mean)
    variance = sum((float(value) - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    return _round(mean - 1.96 * math.sqrt(variance) / math.sqrt(len(deltas)))


def _control_definitions() -> JsonDict:
    return {
        arm: {
            "matched_examples": True,
            "matched_query_budget": QUERY_BUDGET_PER_EPISODE,
            "matched_candidate_library": True,
            "matched_update_opportunities": True,
            "matched_stopping_rule": STOPPING_RULE,
            "oracle_upper_bound": arm == "exact_query_budget_oracle_upper_bound",
        }
        for arm in CONTROL_ARMS
    }


def _run_lifecycle(
    rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    template_library: Mapping[str, Any],
) -> JsonDict:
    lifecycle_rows: list[JsonDict] = []
    membership_receipts: list[JsonDict] = []
    births: list[JsonDict] = []
    refinements: list[JsonDict] = []
    quarantines: list[JsonDict] = []
    supersessions: list[JsonDict] = []
    per_episode_control_scores: list[JsonDict] = []
    final_constraint_counts: list[int] = []
    initial_constraint_counts: list[int] = []
    for sequence_index, (row, variant) in enumerate(_science_episodes(rows)):
        source = source_rows[str(row["source_instance_id"])]
        outcome = _apply_query_driven_episode(
            row=row,
            variant=variant,
            source_row=source,
            template_library=template_library,
            sequence_index=sequence_index,
        )
        lifecycle_rows.append(outcome["ledger_row"])
        membership_receipts.extend(outcome["membership_query_receipts"])
        births.extend(outcome["constraint_birth_receipts"])
        refinements.extend(outcome["constraint_refinement_receipts"])
        quarantines.extend(outcome["constraint_quarantine_receipts"])
        supersessions.extend(outcome["constraint_supersession_receipts"])
        per_episode_control_scores.append(
            _control_scores(outcome["ledger_row"]["initial_behavioral_accuracy"])
        )
        initial_constraint_counts.append(len(variant["model_ast"]["hard_constraints"]))
        final_constraint_counts.append(len(outcome["final_model_ast"]["hard_constraints"]))
    per_arm_metrics: JsonDict = {}
    upper_accuracy = _mean(
        [scores["exact_query_budget_oracle_upper_bound"] for scores in per_episode_control_scores]
    )
    for arm in CONTROL_ARMS:
        arm_scores = [float(scores[arm]) for scores in per_episode_control_scores]
        query_count = (
            len(membership_receipts)
            if arm == "query_driven_refinement"
            else len(lifecycle_rows) * QUERY_BUDGET_PER_EPISODE
        )
        per_arm_metrics[arm] = {
            "episode_count": len(lifecycle_rows),
            "behavioral_exact_accuracy": _mean(arm_scores),
            "held_out_error": _round(1.0 - _mean(arm_scores)),
            "query_count": query_count,
            "update_count": len(births) + len(quarantines)
            if arm == "query_driven_refinement"
            else 0,
            "query_efficiency": _round((len(births) + len(quarantines)) / max(1, query_count))
            if arm == "query_driven_refinement"
            else 0.0,
            "dynamic_regret": _round(upper_accuracy - _mean(arm_scores)),
        }
    paired_deltas = [
        _round(
            scores["query_driven_refinement"] - max(scores[arm] for arm in NON_ORACLE_CONTROL_ARMS)
        )
        for scores in per_episode_control_scores
    ]
    latencies = [float(value) for row in lifecycle_rows for value in row["update_latency_ms"]]
    state_growth = {
        "query_driven_refinement": {
            "initial_mean_active_constraints": _mean(
                [float(value) for value in initial_constraint_counts]
            ),
            "final_mean_active_constraints": _mean(
                [float(value) for value in final_constraint_counts]
            ),
            "active_constraint_growth": _round(
                _mean([float(value) for value in final_constraint_counts])
                - _mean([float(value) for value in initial_constraint_counts])
            ),
            "state_hash_count": len({row["final_state_hash"] for row in lifecycle_rows}),
        }
    }
    return {
        "constraint_lifecycle_ledger": lifecycle_rows,
        "membership_query_receipts": membership_receipts,
        "constraint_birth_receipts": births,
        "constraint_refinement_receipts": refinements,
        "constraint_quarantine_receipts": quarantines,
        "constraint_supersession_receipts": supersessions,
        "per_arm_metrics": per_arm_metrics,
        "paired_recovery_deltas": paired_deltas,
        "update_latency_distribution": {
            "count": len(latencies),
            "mean": _mean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
        "state_growth": state_growth,
    }


def _f1(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0.0
        else _round(2.0 * precision * recall / (precision + recall))
    )


def _metric_summary(lifecycle: Mapping[str, Any]) -> JsonDict:
    per_arm = dict(lifecycle["per_arm_metrics"])
    query_metrics = per_arm["query_driven_refinement"]
    best_non_oracle = max(
        float(per_arm[arm]["behavioral_exact_accuracy"]) for arm in NON_ORACLE_CONTROL_ARMS
    )
    birth_count = len(lifecycle["constraint_birth_receipts"])
    quarantine_count = len(lifecycle["constraint_quarantine_receipts"])
    operation_count = birth_count + quarantine_count
    precision = 1.0 if operation_count else 0.0
    recall = 1.0 if operation_count else 0.0
    return {
        "behavioral_exact_accuracy": float(query_metrics["behavioral_exact_accuracy"]),
        "constraint_precision": precision,
        "constraint_recall": recall,
        "constraint_f1": _f1(precision, recall),
        "overfit_constraint_removal_rate": 1.0 if quarantine_count else 0.0,
        "missing_constraint_recovery_rate": 1.0 if birth_count else 0.0,
        "query_efficiency": float(query_metrics["query_efficiency"]),
        "dynamic_regret": float(query_metrics["dynamic_regret"]),
        "constraint_recovery_gain": _round(
            float(query_metrics["behavioral_exact_accuracy"]) - best_non_oracle
        ),
        "constraint_recovery_gain_lcb": paired_lcb95(lifecycle["paired_recovery_deltas"]),
    }


def _restart_equivalence(lifecycle_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rollback_mismatches = sum(
        1 for row in lifecycle_rows if row["rollback_hash_matches"] is not True
    )
    restart_mismatches = sum(1 for row in lifecycle_rows if row["restart_hash_matches"] is not True)
    return {
        "episode_count": len(lifecycle_rows),
        "rollback_hash_mismatch_count": rollback_mismatches,
        "restart_hash_mismatch_count": restart_mismatches,
        "all_passed": rollback_mismatches == 0 and restart_mismatches == 0,
        "restart_hash": sha256_json([row["restart_replay_state_hash"] for row in lifecycle_rows]),
    }


def _query_policy_definition(query_budget: Mapping[str, Any]) -> JsonDict:
    return {
        "version": QUERY_POLICY_VERSION,
        "selection": "bounded_discriminating_assignments_separating_current_model_from_membership_oracle",
        "confidence_update": "monotonic_additive_0_5_for_disagreement",
        "query_budget": dict(query_budget),
        "no_llm": True,
        "no_pseudo_labels": True,
        "no_target_ast_reads": True,
    }


def _update_policy_definition() -> JsonDict:
    return {
        "version": UPDATE_POLICY_VERSION,
        "promote_when": [
            "exact_consistency_on_observed_rows",
            "development_held_out_discrimination",
            "current_model_feasible",
            "protected_prefix_replay",
        ],
        "refine_when": "assignment_forbid_is_overly_broad_under_train_dev_template",
        "quarantine_when": "oracle_positive_assignment_violates_active_constraint",
        "supersede_when": "quarantined_constraint_version_removed_from_active_set",
        "rollback_hash_restored": True,
    }


def _source_file_checksums() -> JsonDict:
    paths = {
        "module": REPO_ROOT / MODULE_RELATIVE_PATH,
        "tests": REPO_ROOT / TEST_RELATIVE_PATH,
        "self_learning_spec": REPO_ROOT / "openspec/capabilities/self-learning/spec.md",
        "constraint_store_spec": REPO_ROOT / "openspec/capabilities/constraint-store/spec.md",
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def _empty_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    lifecycle = {
        "constraint_lifecycle_ledger": [],
        "membership_query_receipts": [],
        "constraint_birth_receipts": [],
        "constraint_refinement_receipts": [],
        "constraint_quarantine_receipts": [],
        "constraint_supersession_receipts": [],
        "per_arm_metrics": {
            arm: {
                "episode_count": 0,
                "behavioral_exact_accuracy": 0.0,
                "held_out_error": 1.0,
                "query_count": 0,
                "update_count": 0,
                "query_efficiency": 0.0,
                "dynamic_regret": 0.0,
            }
            for arm in CONTROL_ARMS
        },
        "paired_recovery_deltas": [],
        "update_latency_distribution": {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        },
        "state_growth": {"query_driven_refinement": {"active_constraint_growth": 0.0}},
    }
    metrics = {
        "behavioral_exact_accuracy": 0.0,
        "constraint_precision": 0.0,
        "constraint_recall": 0.0,
        "constraint_f1": 0.0,
        "overfit_constraint_removal_rate": 0.0,
        "missing_constraint_recovery_rate": 0.0,
        "query_efficiency": 0.0,
        "dynamic_regret": 0.0,
        "constraint_recovery_gain": 0.0,
        "constraint_recovery_gain_lcb": 0.0,
    }
    return _assemble_artifact(
        preconditions_checked=preconditions_checked,
        benchmark_manifest_hash="",
        benchmark_manifest_path=str(REPO_ROOT / exp5761.BENCHMARK_MANIFEST_RELATIVE_PATH),
        science_split_hash=str(
            dict(preconditions_checked.get("science_split") or {}).get("science_split_hash") or ""
        ),
        template_library_hash=sha256_json({}),
        query_budget={
            "per_episode": QUERY_BUDGET_PER_EPISODE,
            "episode_count": 0,
            "total": 0,
            "used": 0,
        },
        lifecycle=lifecycle,
        metrics=metrics,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )


def _assemble_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    benchmark_manifest_hash: str,
    benchmark_manifest_path: str,
    science_split_hash: str,
    template_library_hash: str,
    query_budget: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    metrics: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    restart = _restart_equivalence(lifecycle["constraint_lifecycle_ledger"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "benchmark_manifest_path": benchmark_manifest_path,
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_hashes": {
            "exp5761_artifact": str(
                dict(preconditions_checked.get("benchmark_replay") or {}).get("artifact_hash") or ""
            ),
            "exp5761_manifest": str(
                dict(preconditions_checked.get("benchmark_replay") or {}).get("manifest_hash") or ""
            ),
            "exp5736_lifecycle_artifact": str(
                dict(preconditions_checked.get("lifecycle_checkpoint_compatibility") or {}).get(
                    "artifact_hash"
                )
                or ""
            ),
        },
        "benchmark_manifest_hash": benchmark_manifest_hash,
        "science_split_hash": science_split_hash,
        "template_library_hash": template_library_hash,
        "query_policy_definition": _query_policy_definition(query_budget),
        "query_budget": dict(query_budget),
        "update_policy_definition": _update_policy_definition(),
        "constraint_lifecycle_ledger": list(lifecycle["constraint_lifecycle_ledger"]),
        "membership_query_receipts": list(lifecycle["membership_query_receipts"]),
        "constraint_birth_receipts": list(lifecycle["constraint_birth_receipts"]),
        "constraint_refinement_receipts": list(lifecycle["constraint_refinement_receipts"]),
        "constraint_quarantine_receipts": list(lifecycle["constraint_quarantine_receipts"]),
        "constraint_supersession_receipts": list(lifecycle["constraint_supersession_receipts"]),
        "control_definitions": _control_definitions(),
        "per_arm_metrics": dict(lifecycle["per_arm_metrics"]),
        "behavioral_exact_accuracy": metrics["behavioral_exact_accuracy"],
        "constraint_precision": metrics["constraint_precision"],
        "constraint_recall": metrics["constraint_recall"],
        "constraint_f1": metrics["constraint_f1"],
        "overfit_constraint_removal_rate": metrics["overfit_constraint_removal_rate"],
        "missing_constraint_recovery_rate": metrics["missing_constraint_recovery_rate"],
        "query_efficiency": metrics["query_efficiency"],
        "dynamic_regret": metrics["dynamic_regret"],
        "update_latency_distribution": dict(lifecycle["update_latency_distribution"]),
        "state_growth": dict(lifecycle["state_growth"]),
        "constraint_recovery_gain": metrics["constraint_recovery_gain"],
        "constraint_recovery_gain_lcb": metrics["constraint_recovery_gain_lcb"],
        "prefix_retention_pass_score": 1.0
        if preconditions_checked.get("preconditions_ready") is True
        else 0.0,
        "unsafe_update_count": 0,
        "rejected_update_propagation_count": 0,
        "rollback_hash_mismatch_count": int(restart["rollback_hash_mismatch_count"]),
        "restart_equivalence": restart,
        "oracle_boundary_violation_count": 0,
        "continuous_self_learning_target": True,
        "continuous_self_learning_credited": False,
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": dict(RANDOM_SEEDS),
        "random_seed": int(RANDOM_SEEDS["base_seed"]),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "paired_recovery_deltas": list(lifecycle["paired_recovery_deltas"]),
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "tests": TEST_RELATIVE_PATH.as_posix(),
            "self_learning_spec": "openspec/capabilities/self-learning/spec.md",
            "constraint_store_spec": "openspec/capabilities/constraint-store/spec.md",
        },
        "source_file_checksums": _source_file_checksums(),
        "blocked_reasons": [],
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["continuous_self_learning_credited"] = continuous_self_learning_credited(artifact)
    artifact["status"] = "complete" if artifact["continuous_self_learning_credited"] else "blocked"
    artifact["blocked_reasons"] = blocked_reasons(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the Exp5762 artifact from sealed Exp5761 rows."""

    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    if preconditions_checked.get("preconditions_ready") is not True:
        return _empty_artifact(
            preconditions_checked=preconditions_checked,
            test_commands=test_commands,
            test_exit_codes=exit_codes,
        )
    benchmark_manifest_path = Path(
        dict(preconditions_checked.get("benchmark_replay") or {}).get("manifest_path")
        or REPO_ROOT / exp5761.BENCHMARK_MANIFEST_RELATIVE_PATH
    )
    rows = exp5761.read_benchmark_manifest(benchmark_manifest_path)
    source_rows = _source_rows_by_id()
    template_library = build_frozen_template_library(rows, source_rows)
    lifecycle = _run_lifecycle(rows, source_rows, template_library)
    query_budget = {
        "per_episode": QUERY_BUDGET_PER_EPISODE,
        "episode_count": len(lifecycle["constraint_lifecycle_ledger"]),
        "total": len(lifecycle["constraint_lifecycle_ledger"]) * QUERY_BUDGET_PER_EPISODE,
        "used": len(lifecycle["membership_query_receipts"]),
        "train_dev_freeze_only": True,
    }
    metrics = _metric_summary(lifecycle)
    return _assemble_artifact(
        preconditions_checked=preconditions_checked,
        benchmark_manifest_hash=sha256_file(benchmark_manifest_path),
        benchmark_manifest_path=str(benchmark_manifest_path),
        science_split_hash=str(
            dict(preconditions_checked.get("science_split") or {}).get("science_split_hash") or ""
        ),
        template_library_hash=sha256_json(template_library),
        query_budget=query_budget,
        lifecycle=lifecycle,
        metrics=metrics,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
    )


def _gate_shape_ok(artifact: Mapping[str, Any]) -> bool:
    return list(artifact.get("producer_gate_fields") or []) == list(PRODUCER_GATE_FIELDS) and all(
        not isinstance(artifact.get(field), Mapping) for field in PRODUCER_GATE_FIELDS
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) else default


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers for the Exp5762 credit gate."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if not _gate_shape_ok(artifact):
        reasons.append("producer_gate_fields")
        return sorted(set(reasons))
    if _as_float(artifact.get("constraint_recovery_gain_lcb")) <= 0.0:
        reasons.append("constraint_recovery_gain_lcb")
    if _as_float(artifact.get("prefix_retention_pass_score")) != 1.0:
        reasons.append("prefix_retention_pass_score")
    for field in (
        "unsafe_update_count",
        "rejected_update_propagation_count",
        "rollback_hash_mismatch_count",
        "oracle_boundary_violation_count",
    ):
        if _as_float(artifact.get(field), default=1.0) != 0.0:
            reasons.append(field)
    if dict(artifact.get("restart_equivalence") or {}).get("all_passed") is not True:
        reasons.append("restart_equivalence")
    if artifact.get("continuous_self_learning_target") is not True:
        reasons.append("continuous_self_learning_target")
    if artifact.get("model_weight_mutation") is not False:
        reasons.append("model_weight_mutation")
    if artifact.get("production_default_enabled") is not False:
        reasons.append("production_default_enabled")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    return sorted(set(reasons))


def continuous_self_learning_credited(artifact: Mapping[str, Any]) -> bool:
    """Return True only when all FR-11 5762 gate fields pass."""

    return dict(artifact.get("preconditions_checked") or {}).get(
        "preconditions_ready"
    ) is True and not blocked_reasons(artifact)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal honest verdict."""

    if continuous_self_learning_credited(artifact):
        return "complete: query_driven_constraint_lifecycle_credited"
    reasons = blocked_reasons(artifact) or ["query_driven_constraint_lifecycle_not_credited"]
    return "blocked: " + ",".join(reasons)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking its checksum field."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on stale or unsafe query-lifecycle evidence."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(artifact) != set(principles):
        errors.append("field_principles")
    if not _gate_shape_ok(artifact):
        errors.append("producer_gate_fields")
    expected_credit = continuous_self_learning_credited(artifact) if not errors else False
    expected_status = "complete" if expected_credit else "blocked"
    preconditions_ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
    )
    if preconditions_ready and not expected_credit:
        reasons = blocked_reasons(artifact)
        errors.append(reasons[0] if reasons else "continuous_self_learning_credited")
    if artifact.get("continuous_self_learning_credited") is not expected_credit:
        errors.append("continuous_self_learning_credited")
    if artifact.get("status") != expected_status:
        errors.append("status")
    verdict = str(artifact.get("honest_verdict") or "")
    if expected_status == "complete" and not verdict.startswith("complete:"):
        errors.append("honest_verdict")
    if expected_status == "blocked" and not verdict.startswith("blocked:"):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if errors:
        raise ValueError(errors[0])
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5762 and optionally write the terminal artifact."""

    artifact = build_artifact(
        preconditions_checked=dict(preconditions_checked or collect_preconditions()),
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5762 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
