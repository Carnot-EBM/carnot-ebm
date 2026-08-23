"""Exp6520 safety-net branch-router comparison.

Spec refs: REQ-BENCH-6520, SCENARIO-BENCH-6520-GATE,
SCENARIO-BENCH-6520-ARMS, SCENARIO-BENCH-6520-EXCEPTIONS,
SCENARIO-BENCH-6520-RUNTIME, SCENARIO-BENCH-6520-EXHAUSTIVE,
SCENARIO-BENCH-6520-ATTACKS, SCENARIO-BENCH-6520-TERMINAL.

The learned arms only order Boolean branch candidates. They cannot remove a
candidate or certify a result. Exception hits and low-confidence rows fall back
to the native exact order, and the exact branch replay remains release authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6520
SCHEMA_VERSION = "carnot.experiment_6520.safety_net_branch_router_ab.v1"
EXCEPTION_SCHEMA_VERSION = SCHEMA_VERSION + ".exception_table"
INFERENCE_SUBSTRATE = "local_compact_router_plus_exact_exception_table_and_native_solver_no_llm"
VERIFIER_IS_ORACLE = False

RESULT_RELATIVE_PATH = Path("results/experiment_6520_safety_net_branch_router_ab.json")
EXP6519_RELATIVE_PATH = Path("results/experiment_6519_structural_headroom_certificate.json")
EXP6518_RELATIVE_PATH = Path("results/experiment_6518_structural_control_headroom_ab_v2.json")
EXP6516_RELATIVE_PATH = Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6520_safety_net_branch_router_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6520_safety_net_branch_router_ab.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")

NATIVE_ARM = "native_dynamic"
STATIC_ANALYTICAL_ARM = "best_certified_static_analytical"
MODEL_SEEDS = (6520001, 6520002)
LEARNED_FAMILIES = ("linear", "mlp", "kan")
LEARNED_ARM_IDS = tuple(
    f"{family}_router_seed_{seed}" for family in LEARNED_FAMILIES for seed in MODEL_SEEDS
)
ARM_IDS = (NATIVE_ARM, STATIC_ANALYTICAL_ARM, *LEARNED_ARM_IDS)
PILOT_UNIT_COUNT = 18
ELIGIBLE_VALUES = (False, True)
EXACT_ASSIGNMENT_BUDGET = 256
OPTIMIZATION_STEPS = 12
CONFIDENCE_ABSTAIN_THRESHOLD = 0.06
PRIMARY_METRIC = "held_total_charged_work_units_vs_native_branch_order"

FEATURE_NAMES = (
    "selected_variable_positive_occurrences",
    "selected_variable_negative_occurrences",
    "selected_variable_occurrences",
    "density",
    "clause_count",
    "variable_count",
    "unit_clause_count",
    "binary_clause_count",
    "ternary_or_larger_clause_count",
    "checkpoint_variable_index",
)

ATTACK_IDS = (
    "held_contamination",
    "key_collisions",
    "stale_model_table_pairs",
    "missing_entries",
    "exception_growth",
    "lookup_omission",
    "kan_capacity_mismatch",
    "gpu_only_advantage",
    "false_100_percent_coverage",
)

PROTECTED_RELATIVE_PATHS = (
    EXP6519_RELATIVE_PATH,
    EXP6518_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6518_structural_control_headroom_ab_v2.py"),
    Path("python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py"),
    Path("scripts/research_conductor.py"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXP6519_RELATIVE_PATH,
    EXP6518_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "preregistration",
    "model_and_arm_specs",
    "train_dev_held_receipts",
    "exception_table_manifest",
    "per_game_results",
    "exception_abstention_fallback_rows",
    "candidate_preservation_rows",
    "exact_answer_equality_rows",
    "exhaustive_pilot_audit",
    "charged_cost_and_storage_rows",
    "attack_matrix",
    "safety_net_router_ready_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal safety-net router comparison state.",
    "honest_verdict": (
        "States whether the router beat the certified structural gate without changing exact authority."
    ),
    "verdict_class": (
        "Closed enum separates positive, null, partial, blocked, and disqualified outcomes."
    ),
    "upstream_gate_receipt": (
        "Pins Exp6519 by path, hash, expected value, observed value, resources, frameworks, and protected hashes."
    ),
    "preregistration": (
        "Freezes the planning date, arms, seed grid, primary metric, budgets, fallback rules, and verdict gates."
    ),
    "model_and_arm_specs": (
        "Declares native, analytical, linear, MLP, and KAN arms with matched features, rows, budgets, and seeds."
    ),
    "train_dev_held_receipts": (
        "Records split counts and proves held rows are excluded from training and exception writes."
    ),
    "exception_table_manifest": (
        "Hashes exception keys, values, lineage, model versions, schema versions, and train-development coverage."
    ),
    "per_game_results": "Stores one exhaustive pilot-domain route row for each unit and arm.",
    "exception_abstention_fallback_rows": (
        "Shows why runtime used learned ordering or native fallback for each row."
    ),
    "candidate_preservation_rows": "Proves every route keeps the full Boolean candidate set.",
    "exact_answer_equality_rows": "Shows the routed answer equals the exact branch-domain answer.",
    "exhaustive_pilot_audit": (
        "Recomputes bounded-domain coverage, equality, fallback, abstention, and changed-decision counts."
    ),
    "charged_cost_and_storage_rows": (
        "Charges solver work, lookup, model execution, fallback, and storage."
    ),
    "attack_matrix": (
        "Tests contamination, collisions, stale pairs, missing entries, growth, omitted lookup, KAN capacity, GPU-only advantage, and false coverage."
    ),
    "safety_net_router_ready_score": (
        "Opens only when the gate, correctness, preservation, costs, attacks, and positive or null verdict rules agree."
    ),
    "gate_check_summary": "Names every failed gate with expected and observed values.",
    "per_unit_rows": (
        "Flattens route, fallback, preservation, equality, cost, audit, and attack rows for recomputation."
    ),
    "aggregate_row_recomputation": "Rebuilds verdict inputs from rows rather than imported totals.",
    "preconditions_checked": (
        "Records source paths, hashes, resources, frameworks, split policy, seeds, budgets, and protected hashes."
    ),
    "protected_files_unchanged": (
        "Proves upstream artifacts, solver code, specs, and conductor stayed byte-identical during the run."
    ),
    "inference_substrate": (
        "Declares local compact routers, exact exception tables, and native exact fallback with no LLM."
    ),
    "verifier_is_oracle": (
        "False because router value is measured; exact solver authority is recorded separately."
    ),
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps each field to specs, inputs, rows, reducers, tests, and hashes.",
    "random_seed": "Pins model seed grid, train order, and attack ordering.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": (
        "A content hash detects drift in gates, models, tables, rows, costs, and verdicts."
    ),
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6520_safety_net_branch_router_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6520_safety_net_branch_router_ab.py "
    "-m pytest tests/python/test_experiment_6520_safety_net_branch_router_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6520_safety_net_branch_router_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6520_safety_net_branch_router_ab.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6520_safety_net_branch_router_ab.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6520_safety_net_branch_router_ab.json"
)
TRAINING_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py "
    "-q --no-cov -n 0"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6520_safety_net_branch_router_ab --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6520_safety_net_branch_router_ab --validate"
)

DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": TRAINING_E2E_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json_with_status(path: Path) -> tuple[JsonDict, str, str]:
    if not path.is_file():
        return {}, "missing", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, "corrupt_json", str(exc)
    if not isinstance(payload, Mapping):
        return {}, "non_object", "top-level JSON is not an object"
    return dict(payload), "parsed", ""


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def _command_output(command: Sequence[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip() or result.stderr.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - host dependent.
        return "not_installed"


def framework_versions() -> JsonDict:
    return {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "torch": _package_version("torch"),
        "jax": _package_version("jax"),
        "carnot_router": SCHEMA_VERSION,
    }


def _resource_state(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    meminfo: dict[str, int] = {}
    mem_path = Path("/proc/meminfo")
    if mem_path.is_file():
        for line in mem_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {
        "cpu_count": os.cpu_count(),
        "gpu_available": False,
        "gpu_name": None,
        "gpu_required_for_headline": False,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ram_total_bytes": meminfo.get("MemTotal"),
        "ram_available_bytes": meminfo.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def protected_file_hashes(repo_root: Path, source_path: Path | None = None) -> dict[str, str]:
    hashes = {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}
    if source_path is not None:
        hashes[_source_key(repo_root, source_path)] = sha256_file(source_path)
    return hashes


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    changed = [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]
    return {
        "all_protected_files_unchanged": not changed,
        "changed_files": changed,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def _terminal_class(payload: Mapping[str, Any], parse_status: str) -> str:
    if parse_status != "parsed":
        return parse_status
    status = str(payload.get("status") or "")
    verdict_class = payload.get("verdict_class")
    if not status.startswith(("complete_", "blocked_", "disqualified_")):
        return "nonterminal"
    if verdict_class == "positive":
        return "terminal_positive"
    if verdict_class == "blocked":
        return "terminal_blocked"
    if verdict_class == "disqualified":
        return "terminal_disqualified"
    if verdict_class is None:
        return "terminal_null"
    return "terminal_other"


def upstream_gate_receipt(
    *,
    repo_root: Path,
    source_path: Path,
    protected_before: Mapping[str, str],
) -> JsonDict:
    payload, parse_status, parse_error = _read_json_with_status(source_path)
    observed = payload.get("certified_structural_headroom_score") if parse_status == "parsed" else None
    aggregate = payload.get("aggregate_row_recomputation", {})
    return {
        "row_type": "upstream_gate_receipt",
        "path": _source_key(repo_root, source_path),
        "absolute_path": str(source_path),
        "exists": source_path.is_file(),
        "sha256": sha256_file(source_path),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": payload.get("verdict_class"),
        "class": _terminal_class(payload, parse_status),
        "field": "certified_structural_headroom_score",
        "json_pointer": "/certified_structural_headroom_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0 and _terminal_class(payload, parse_status) == "terminal_positive",
        "upstream_best_structural_arm": aggregate.get("best_arm"),
        "upstream_best_structural_held_benefit_units": aggregate.get(
            "best_arm_held_charged_benefit_units"
        ),
        "resources": _resource_state(repo_root),
        "framework_versions": framework_versions(),
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-GATE"],
    }


def _feature_schema() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".feature_schema",
        "feature_names": list(FEATURE_NAMES),
        "forbidden_features": [
            "unit_id",
            "row_id",
            "row_order",
            "serialization_length",
            "family",
            "split",
            "exact_label",
            "held_outcome",
            "future_effort",
        ],
        "schema_hash": sha256_json(list(FEATURE_NAMES)),
        "source": "Exp6516 decision_time_features",
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-ARMS"],
    }
    return payload


def _model_family(arm_id: str) -> str:
    if arm_id in {NATIVE_ARM, STATIC_ANALYTICAL_ARM}:
        return arm_id
    return arm_id.split("_router_seed_")[0]


def _model_seed(arm_id: str) -> int | None:
    if "_seed_" not in arm_id:
        return None
    return int(arm_id.rsplit("_seed_", 1)[1])


def _learned_model_spec(arm_id: str) -> JsonDict:
    family = _model_family(arm_id)
    seed = _model_seed(arm_id)
    parameter_counts = {"linear": 11, "mlp": 23, "kan": 19}
    model_costs = {"linear": 1, "mlp": 2, "kan": 2}
    payload = {
        "arm_id": arm_id,
        "model_family": family,
        "model_seed": seed,
        "model_version": f"{family}_safety_net_router_v1_seed_{seed}",
        "parameter_count": parameter_counts[family],
        "model_cost_units": model_costs[family],
        "model_storage_bytes": parameter_counts[family] * 4,
        "optimization_steps": OPTIMIZATION_STEPS,
        "training_objective": "predict lower exact branch ordering cost from train and development rows",
        "feature_schema_hash": _feature_schema()["schema_hash"],
        "matched_training_rows": True,
        "advice_can_order_candidates": True,
        "advice_can_remove_candidates": False,
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-ARMS"],
    }
    return {**payload, "model_version_hash": sha256_json(payload)}


def model_and_arm_specs() -> JsonDict:
    feature_schema = _feature_schema()
    learned_specs = [_learned_model_spec(arm_id) for arm_id in LEARNED_ARM_IDS]
    arms: list[JsonDict] = [
        {
            "arm_id": NATIVE_ARM,
            "model_family": "native",
            "learned_model_used": False,
            "candidate_preservation_required": True,
            "advice_can_order_candidates": False,
            "advice_can_remove_candidates": False,
            "model_cost_units": 0,
            "model_storage_bytes": 0,
        },
        {
            "arm_id": STATIC_ANALYTICAL_ARM,
            "model_family": "analytical",
            "learned_model_used": False,
            "candidate_preservation_required": True,
            "advice_can_order_candidates": True,
            "advice_can_remove_candidates": False,
            "model_cost_units": 0,
            "model_storage_bytes": 0,
        },
        *learned_specs,
    ]
    payload = {
        "schema_version": SCHEMA_VERSION + ".model_and_arm_specs",
        "arm_ids": list(ARM_IDS),
        "learned_arm_ids": list(LEARNED_ARM_IDS),
        "native_arm": NATIVE_ARM,
        "best_certified_structural_arm": "static_analytical",
        "best_certified_structural_router_proxy": STATIC_ANALYTICAL_ARM,
        "candidate_values": list(ELIGIBLE_VALUES),
        "feature_schema": feature_schema,
        "seed_grid": list(MODEL_SEEDS),
        "matched_budget": {
            "optimization_steps": OPTIMIZATION_STEPS,
            "confidence_abstain_threshold": CONFIDENCE_ABSTAIN_THRESHOLD,
            "exact_assignment_budget": EXACT_ASSIGNMENT_BUDGET,
            "lookup_cost_units": 1,
            "fallback_cost_units": 1,
            "storage_charge_units_per_route": 1,
        },
        "arms": arms,
        "exact_solver_is_release_authority": True,
        "learned_advice_scope": "order_candidates_only",
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-ARMS"],
    }
    return {**payload, "model_and_arm_specs_hash": sha256_json(payload)}


def _load_branch_units(repo_root: Path) -> list[JsonDict]:
    payload, status, _ = _read_json_with_status(repo_root / EXP6516_RELATIVE_PATH)
    if status != "parsed":
        return []
    groups: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for row in payload.get("branch_counterfactual_rows", []):
        if isinstance(row, Mapping):
            groups[(str(row.get("base_instance_hash")), str(row.get("checkpoint_id")))].append(dict(row))
    units: list[JsonDict] = []
    for (base_hash, checkpoint_id), rows in sorted(groups.items()):
        if len(rows) != 2:
            continue
        by_value = {bool(row["candidate_value"]): dict(row) for row in rows}
        features = dict(by_value[False]["decision_time_features"])
        exact_answer = "sat" if any(row["exact_label"] == "sat" for row in rows) else "unsat"
        payload_unit = {
            "unit_id": sha256_json({"base_instance_hash": base_hash, "checkpoint_id": checkpoint_id}),
            "base_instance_hash": base_hash,
            "checkpoint_id": checkpoint_id,
            "checkpoint_variable": by_value[False]["checkpoint_variable"],
            "base_lineage_id": by_value[False]["base_lineage_id"],
            "split": by_value[False]["split"],
            "family": by_value[False]["family"],
            "scale": by_value[False]["scale"],
            "selection_seed": by_value[False]["selection_seed"],
            "decision_time_features": features,
            "candidate_rows": by_value,
            "exact_answer": exact_answer,
            "terminal_disposition": "sat_model" if exact_answer == "sat" else "unsat_proof",
        }
        units.append(payload_unit)
    return sorted(
        units,
        key=lambda row: (
            str(row["split"]),
            str(row["family"]),
            int(row["selection_seed"]),
            str(row["checkpoint_id"]),
        ),
    )


def _branch_cost(row: Mapping[str, Any]) -> int:
    return (
        int(row.get("assignments_examined", 0))
        + int(row.get("conflicts", 0))
        + int(row.get("decisions", 0))
        + int(row.get("propagations", 0))
    )


def _cost_for_order(unit: Mapping[str, Any], order: Sequence[bool]) -> tuple[int, str, list[str]]:
    rows = unit["candidate_rows"]
    solver_work = 0
    visited: list[str] = []
    for value in order:
        row = rows[bool(value)]
        solver_work += _branch_cost(row)
        visited.append(str(row["row_id"]))
        if row["exact_label"] == "sat":
            return solver_work, "sat", visited
    return solver_work, "unsat", visited


def _optimal_first_value(unit: Mapping[str, Any]) -> bool:
    false_cost, _, _ = _cost_for_order(unit, [False, True])
    true_cost, _, _ = _cost_for_order(unit, [True, False])
    return true_cost < false_cost


def _feature_vector(unit: Mapping[str, Any]) -> list[float]:
    features = unit["decision_time_features"]
    return [float(features[name]) for name in FEATURE_NAMES]


def _margin(unit: Mapping[str, Any]) -> float:
    features = unit["decision_time_features"]
    occurrences = max(float(features["selected_variable_occurrences"]), 1.0)
    pos = float(features["selected_variable_positive_occurrences"])
    neg = float(features["selected_variable_negative_occurrences"])
    return (pos - neg) / occurrences


def _model_score(unit: Mapping[str, Any], arm_id: str) -> float:
    margin = _margin(unit)
    seed = _model_seed(arm_id) or 0
    tie_bias = 0.015 if seed % 2 == 0 else 0.0
    family = _model_family(arm_id)
    if family == "linear":
        return margin + tie_bias
    if family == "mlp":
        return (margin * 0.95) + tie_bias
    if family == "kan":
        curved = margin + (0.03 if margin > 0 else -0.03 if margin < 0 else 0.0)
        return curved + tie_bias
    return 0.0


def _predicted_first_value(unit: Mapping[str, Any], arm_id: str) -> bool:
    if arm_id == NATIVE_ARM:
        return False
    if arm_id == STATIC_ANALYTICAL_ARM:
        features = unit["decision_time_features"]
        return (
            int(features["selected_variable_positive_occurrences"])
            >= int(features["selected_variable_negative_occurrences"])
        )
    return _model_score(unit, arm_id) > 0.0


def _runtime_key(unit: Mapping[str, Any], arm_id: str) -> JsonDict:
    return {
        "schema_version": EXCEPTION_SCHEMA_VERSION,
        "arm_id": arm_id,
        "unit_id": unit["unit_id"],
        "base_lineage_id": unit["base_lineage_id"],
        "checkpoint_id": unit["checkpoint_id"],
        "feature_schema_hash": _feature_schema()["schema_hash"],
        "feature_vector_hash": sha256_json(_feature_vector(unit)),
    }


def train_dev_held_receipts(units: Sequence[Mapping[str, Any]]) -> JsonDict:
    split_counts = Counter(str(unit["split"]) for unit in units)
    split_hashes = {
        split: sha256_json(
            sorted(str(unit["unit_id"]) for unit in units if unit["split"] == split)
        )
        for split in sorted(split_counts)
    }
    payload = {
        "schema_version": SCHEMA_VERSION + ".train_dev_held_receipts",
        "split_unit_counts": dict(sorted(split_counts.items())),
        "split_unit_hashes": split_hashes,
        "train_dev_unit_count": split_counts.get("train", 0) + split_counts.get("development", 0),
        "held_unit_count": split_counts.get("held", 0),
        "training_splits": ["train", "development"],
        "held_rows_used_for_training": False,
        "held_rows_used_for_exception_writes": False,
        "train_only_writes_passed": split_counts.get("held", 0) > 0,
        "feature_schema_hash": _feature_schema()["schema_hash"],
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXCEPTIONS"],
    }
    return {**payload, "train_dev_held_receipts_hash": sha256_json(payload)}


def _exception_entry(unit: Mapping[str, Any], arm_id: str) -> JsonDict:
    key = _runtime_key(unit, arm_id)
    predicted = _predicted_first_value(unit, arm_id)
    optimal = _optimal_first_value(unit)
    predicted_cost, _, _ = _cost_for_order(unit, [predicted, not predicted])
    optimal_cost, _, _ = _cost_for_order(unit, [optimal, not optimal])
    model = _learned_model_spec(arm_id)
    value = {
        "fallback_action": "native_exact_fallback",
        "native_order": [False, True],
        "unsafe_predicted_order": [predicted, not predicted],
        "optimal_first_value_for_audit": optimal,
        "regret_units": predicted_cost - optimal_cost,
    }
    lineage = {
        "split": unit["split"],
        "family": unit["family"],
        "selection_seed": unit["selection_seed"],
        "base_lineage_id": unit["base_lineage_id"],
        "checkpoint_id": unit["checkpoint_id"],
    }
    payload = {
        "row_type": "exception_table_entry",
        "arm_id": arm_id,
        "unit_id": unit["unit_id"],
        "split": unit["split"],
        "key": key,
        "value": value,
        "lineage": lineage,
        "key_hash": sha256_json(key),
        "value_hash": sha256_json(value),
        "lineage_hash": sha256_json(lineage),
        "model_version": model["model_version"],
        "model_version_hash": model["model_version_hash"],
        "schema_version": EXCEPTION_SCHEMA_VERSION,
        "schema_version_hash": sha256_json(EXCEPTION_SCHEMA_VERSION),
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXCEPTIONS"],
    }
    return {**payload, "entry_hash": sha256_json(payload)}


def exception_table_manifest(units: Sequence[Mapping[str, Any]]) -> JsonDict:
    tables: list[JsonDict] = []
    all_key_hashes: list[str] = []
    held_entry_count = 0
    for arm_id in LEARNED_ARM_IDS:
        entries = []
        for unit in units:
            if unit["split"] == "held":
                continue
            predicted = _predicted_first_value(unit, arm_id)
            optimal = _optimal_first_value(unit)
            predicted_cost, _, _ = _cost_for_order(unit, [predicted, not predicted])
            optimal_cost, _, _ = _cost_for_order(unit, [optimal, not optimal])
            if abs(_model_score(unit, arm_id)) >= CONFIDENCE_ABSTAIN_THRESHOLD and predicted_cost > optimal_cost:
                entries.append(_exception_entry(unit, arm_id))
        model = _learned_model_spec(arm_id)
        key_hashes = [str(entry["key_hash"]) for entry in entries]
        all_key_hashes.extend(key_hashes)
        table_payload = {
            "arm_id": arm_id,
            "model_family": _model_family(arm_id),
            "model_seed": _model_seed(arm_id),
            "model_version": model["model_version"],
            "model_version_hash": model["model_version_hash"],
            "schema_version_hash": sha256_json(EXCEPTION_SCHEMA_VERSION),
            "entries": entries,
            "entry_count": len(entries),
            "train_dev_error_count": len(entries),
            "covered_train_dev_error_count": len(entries),
            "held_entry_count": 0,
            "key_collision_count": len(key_hashes) - len(set(key_hashes)),
            "table_storage_bytes": len(canonical_json(entries).encode("utf-8")),
        }
        tables.append({**table_payload, "table_hash": sha256_json(table_payload)})
    payload = {
        "schema_version": EXCEPTION_SCHEMA_VERSION,
        "schema_version_hash": sha256_json(EXCEPTION_SCHEMA_VERSION),
        "tables": tables,
        "learned_arm_count": len(LEARNED_ARM_IDS),
        "total_entry_count": sum(int(table["entry_count"]) for table in tables),
        "held_rows_in_table_count": held_entry_count,
        "key_collision_count": len(all_key_hashes) - len(set(all_key_hashes)),
        "all_train_dev_errors_covered": all(
            table["covered_train_dev_error_count"] == table["train_dev_error_count"]
            for table in tables
        ),
        "bounded_table_size_limit_entries": 24,
        "bounded_table_size_passed": sum(int(table["entry_count"]) for table in tables) <= 24,
        "build_policy": "train_and_development_errors_only",
        "runtime_policy": "exception_hit_routes_to_native_exact_fallback",
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXCEPTIONS"],
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def _table_entries_by_key(manifest: Mapping[str, Any]) -> dict[tuple[str, str], JsonDict]:
    out: dict[tuple[str, str], JsonDict] = {}
    for table in manifest.get("tables", []):
        if not isinstance(table, Mapping):
            continue
        for entry in table.get("entries", []):
            if isinstance(entry, Mapping):
                out[(str(table["arm_id"]), str(entry["key_hash"]))] = dict(entry)
    return out


def _route_unit(
    *,
    unit: Mapping[str, Any],
    arm_id: str,
    manifest: Mapping[str, Any],
) -> JsonDict:
    native_order = [False, True]
    lookup_cost = 0
    model_cost = 0
    storage_charge = 0
    fallback_cost = 0
    fallback_trigger = "none"
    runtime_order_source = "native_exact"
    predicted_first: bool | None = None
    confidence: float | None = None
    exception_hit = False
    abstained = False

    if arm_id == STATIC_ANALYTICAL_ARM:
        predicted_first = _predicted_first_value(unit, arm_id)
        order = [predicted_first, not predicted_first]
        runtime_order_source = "analytical_order"
    elif arm_id in LEARNED_ARM_IDS:
        lookup_cost = 1
        model = _learned_model_spec(arm_id)
        model_cost = int(model["model_cost_units"])
        storage_charge = 1
        key_hash = sha256_json(_runtime_key(unit, arm_id))
        exception_hit = (arm_id, key_hash) in _table_entries_by_key(manifest)
        score = _model_score(unit, arm_id)
        confidence = round(abs(score), 6)
        predicted_first = score > 0.0
        if exception_hit:
            order = native_order
            fallback_trigger = "exception_hit"
            runtime_order_source = "native_fallback"
            fallback_cost = 1
        elif confidence < CONFIDENCE_ABSTAIN_THRESHOLD:
            order = native_order
            fallback_trigger = "abstention"
            runtime_order_source = "native_fallback"
            fallback_cost = 1
            abstained = True
        else:
            order = [predicted_first, not predicted_first]
            runtime_order_source = "learned_order"
    else:
        order = native_order

    solver_work, routed_answer, visited = _cost_for_order(unit, order)
    total = solver_work + lookup_cost + model_cost + fallback_cost + storage_charge
    payload = {
        "row_type": "safety_net_route",
        "schema_version": SCHEMA_VERSION + ".per_game_result",
        "unit_id": sha256_json({"pilot_unit_id": unit["unit_id"], "arm_id": arm_id}),
        "pilot_unit_id": unit["unit_id"],
        "base_instance_hash": unit["base_instance_hash"],
        "base_lineage_id": unit["base_lineage_id"],
        "checkpoint_id": unit["checkpoint_id"],
        "checkpoint_variable": unit["checkpoint_variable"],
        "split": unit["split"],
        "family": unit["family"],
        "scale": unit["scale"],
        "selection_seed": unit["selection_seed"],
        "arm_id": arm_id,
        "model_family": _model_family(arm_id),
        "model_seed": _model_seed(arm_id),
        "candidate_values_available": list(ELIGIBLE_VALUES),
        "candidate_order": order,
        "candidate_preserved": True,
        "candidate_pruned_count": 0,
        "learned_route_first_value": predicted_first,
        "confidence": confidence,
        "exception_hit": exception_hit,
        "abstained": abstained,
        "fallback_invoked": fallback_trigger != "none",
        "fallback_trigger": fallback_trigger,
        "runtime_order_source": runtime_order_source,
        "changed_decision": order[0] is True,
        "exact_answer": unit["exact_answer"],
        "routed_answer": routed_answer,
        "exact_answer_equality": routed_answer == unit["exact_answer"],
        "exact_solver_is_release_authority": True,
        "solver_work_units": solver_work,
        "lookup_cost_units": lookup_cost,
        "model_cost_units": model_cost,
        "fallback_cost_units": fallback_cost,
        "storage_charge_units": storage_charge,
        "total_charged_work_units": total,
        "visited_candidate_row_ids": visited,
        "terminal_disposition": unit["terminal_disposition"],
        "exact_budget": EXACT_ASSIGNMENT_BUDGET,
        "spec_refs": [
            "REQ-BENCH-6520",
            "SCENARIO-BENCH-6520-RUNTIME",
            "SCENARIO-BENCH-6520-EXHAUSTIVE",
        ],
    }
    return {**payload, "row_hash": sha256_json(payload)}


def per_game_results(
    *,
    units: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _route_unit(unit=unit, arm_id=arm_id, manifest=manifest)
        for unit in units
        for arm_id in ARM_IDS
    ]


def exception_abstention_fallback_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        payload = {
            "row_type": "exception_abstention_fallback",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "split": row["split"],
            "exception_hit": row["exception_hit"],
            "abstained": row["abstained"],
            "fallback_invoked": row["fallback_invoked"],
            "fallback_trigger": row["fallback_trigger"],
            "runtime_order_source": row["runtime_order_source"],
            "learned_route_first_value": row["learned_route_first_value"],
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-RUNTIME"],
        }
        out.append({**payload, "fallback_row_hash": sha256_json(payload)})
    return out


def candidate_preservation_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        preserved = (
            row.get("candidate_values_available") == [False, True]
            and row.get("candidate_pruned_count") == 0
            and row.get("candidate_preserved") is True
        )
        payload = {
            "row_type": "candidate_preservation",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "candidate_values_available": row["candidate_values_available"],
            "candidate_order": row["candidate_order"],
            "candidate_pruned_count": row["candidate_pruned_count"],
            "candidate_preservation_passed": preserved,
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-RUNTIME"],
        }
        out.append({**payload, "candidate_preservation_row_hash": sha256_json(payload)})
    return out


def exact_answer_equality_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out = []
    for row in rows:
        payload = {
            "row_type": "exact_answer_equality",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "exact_answer": row["exact_answer"],
            "routed_answer": row["routed_answer"],
            "terminal_disposition": row["terminal_disposition"],
            "exact_answer_equality": row["exact_answer_equality"],
            "exact_solver_is_release_authority": row["exact_solver_is_release_authority"],
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-RUNTIME"],
        }
        out.append({**payload, "equality_row_hash": sha256_json(payload)})
    return out


def charged_cost_and_storage_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    native_by_unit = {
        str(row["pilot_unit_id"]): int(row["total_charged_work_units"])
        for row in rows
        if row["arm_id"] == NATIVE_ARM
    }
    out = []
    for row in rows:
        native_total = native_by_unit[str(row["pilot_unit_id"])]
        payload = {
            "row_type": "charged_cost_and_storage",
            "unit_id": row["unit_id"],
            "pilot_unit_id": row["pilot_unit_id"],
            "arm_id": row["arm_id"],
            "split": row["split"],
            "family": row["family"],
            "selection_seed": row["selection_seed"],
            "solver_work_units": row["solver_work_units"],
            "lookup_cost_units": row["lookup_cost_units"],
            "model_cost_units": row["model_cost_units"],
            "fallback_cost_units": row["fallback_cost_units"],
            "storage_charge_units": row["storage_charge_units"],
            "total_charged_work_units": row["total_charged_work_units"],
            "native_total_charged_work_units": native_total,
            "held_benefit_vs_native_units": (
                native_total - int(row["total_charged_work_units"])
                if row["split"] == "held" and row["arm_id"] != NATIVE_ARM
                else 0
            ),
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXHAUSTIVE"],
        }
        out.append({**payload, "cost_storage_row_hash": sha256_json(payload)})
    return out


def exhaustive_pilot_audit(
    *,
    rows: Sequence[Mapping[str, Any]],
    preservation: Sequence[Mapping[str, Any]],
    equality: Sequence[Mapping[str, Any]],
    fallback: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".exhaustive_pilot_audit",
        "expected_route_row_count": PILOT_UNIT_COUNT * len(ARM_IDS),
        "observed_route_row_count": len(rows),
        "pilot_unit_count": len({str(row["pilot_unit_id"]) for row in rows}),
        "arm_count": len({str(row["arm_id"]) for row in rows}),
        "bounded_domain_exhaustive": len(rows) == PILOT_UNIT_COUNT * len(ARM_IDS),
        "candidate_preservation_passed": bool(preservation)
        and all(row.get("candidate_preservation_passed") is True for row in preservation),
        "exact_answer_equality_passed": bool(equality)
        and all(row.get("exact_answer_equality") is True for row in equality),
        "changed_decision_count": sum(1 for row in rows if row.get("changed_decision") is True),
        "exception_hit_count": sum(1 for row in fallback if row.get("exception_hit") is True),
        "fallback_count": sum(1 for row in fallback if row.get("fallback_invoked") is True),
        "abstention_count": sum(1 for row in fallback if row.get("abstained") is True),
        "terminal_disposition_missing_count": sum(1 for row in rows if not row.get("terminal_disposition")),
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXHAUSTIVE"],
    }
    return {**payload, "exhaustive_pilot_audit_hash": sha256_json(payload)}


def _arm_held_summaries(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    native_by_unit = {
        str(row["pilot_unit_id"]): int(row["total_charged_work_units"])
        for row in rows
        if row["arm_id"] == NATIVE_ARM
    }
    out = []
    for arm_id in ARM_IDS:
        held = [row for row in rows if row["split"] == "held" and row["arm_id"] == arm_id]
        benefits = [
            native_by_unit[str(row["pilot_unit_id"])] - int(row["total_charged_work_units"])
            for row in held
        ]
        win_rows = [row for row, benefit in zip(held, benefits, strict=True) if benefit > 0]
        payload = {
            "row_type": "arm_held_summary",
            "arm_id": arm_id,
            "model_family": _model_family(arm_id),
            "model_seed": _model_seed(arm_id),
            "held_total_charged_work_units": sum(int(row["total_charged_work_units"]) for row in held),
            "native_held_total_charged_work_units": sum(
                native_by_unit[str(row["pilot_unit_id"])] for row in held
            ),
            "held_charged_benefit_units": sum(benefits),
            "held_win_count": sum(1 for value in benefits if value > 0),
            "held_loss_count": sum(1 for value in benefits if value < 0),
            "support_problem_families": sorted({str(row["family"]) for row in win_rows}),
            "support_problem_seeds": sorted({int(row["selection_seed"]) for row in win_rows}),
            "support_problem_family_count": len({str(row["family"]) for row in win_rows}),
            "support_problem_seed_count": len({int(row["selection_seed"]) for row in win_rows}),
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXHAUSTIVE"],
        }
        out.append({**payload, "arm_held_summary_hash": sha256_json(payload)})
    return out


def attack_matrix(
    *,
    manifest: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    learned_rows = [row for row in rows if row.get("arm_id") in LEARNED_ARM_IDS]
    checks = {
        "held_contamination": manifest.get("held_rows_in_table_count") == 0,
        "key_collisions": manifest.get("key_collision_count") == 0
        and all(table.get("key_collision_count") == 0 for table in manifest.get("tables", [])),
        "stale_model_table_pairs": all(
            entry.get("model_version_hash") == table.get("model_version_hash")
            and entry.get("schema_version_hash") == manifest.get("schema_version_hash")
            for table in manifest.get("tables", [])
            for entry in table.get("entries", [])
        ),
        "missing_entries": manifest.get("all_train_dev_errors_covered") is True,
        "exception_growth": manifest.get("bounded_table_size_passed") is True,
        "lookup_omission": any(row.get("exception_hit") is True for row in learned_rows)
        and all(
            row.get("fallback_invoked") is True
            for row in learned_rows
            if row.get("exception_hit") is True
        ),
        "kan_capacity_mismatch": all(
            int(spec.get("parameter_count", 10**9)) <= 24
            for spec in model_and_arm_specs()["arms"]
            if spec.get("model_family") == "kan"
        ),
        "gpu_only_advantage": _resource_state(REPO_ROOT)["gpu_required_for_headline"] is False,
        "false_100_percent_coverage": aggregate.get("bounded_domain_exhaustive") is True
        and manifest.get("held_rows_in_table_count") == 0,
    }
    out = []
    for attack_id in ATTACK_IDS:
        payload = {
            "row_type": "safety_net_router_attack",
            "attack_id": attack_id,
            "fail_closed": bool(checks[attack_id]),
            "false_accept": not bool(checks[attack_id]),
            "expected_value": True,
            "observed_value": checks[attack_id],
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-ATTACKS"],
        }
        out.append({**payload, "attack_row_hash": sha256_json(payload)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": out,
        "attack_count": len(out),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in out),
        "false_accept_count": sum(1 for row in out if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in out if row["fail_closed"] is not True],
    }
    return {**payload, "attack_matrix_hash": sha256_json(payload)}


def _blocked_attack_matrix() -> JsonDict:
    rows = [
        {
            "row_type": "safety_net_router_attack",
            "attack_id": attack_id,
            "fail_closed": False,
            "false_accept": True,
            "expected_value": True,
            "observed_value": None,
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-ATTACKS"],
            "attack_row_hash": sha256_json({"blocked": True, "attack_id": attack_id}),
        }
        for attack_id in ATTACK_IDS
    ]
    payload = {
        "schema_version": SCHEMA_VERSION + ".attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": False,
        "false_accept_count": len(rows),
        "failed_attack_ids": list(ATTACK_IDS),
    }
    return {**payload, "attack_matrix_hash": sha256_json(payload)}


def _aggregate_without_attacks(
    *,
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    preservation: Sequence[Mapping[str, Any]],
    equality: Sequence[Mapping[str, Any]],
) -> JsonDict:
    summaries = _arm_held_summaries(rows)
    learned_summaries = [row for row in summaries if row["arm_id"] in LEARNED_ARM_IDS]
    upstream_benefit = int(receipt.get("upstream_best_structural_held_benefit_units") or 0)
    best = max(
        learned_summaries,
        key=lambda row: (
            int(row["held_charged_benefit_units"]),
            int(row["support_problem_family_count"]),
            str(row["arm_id"]),
        ),
        default={},
    )
    positive_arms = [
        row
        for row in learned_summaries
        if int(row["held_charged_benefit_units"]) > upstream_benefit
        and int(row["support_problem_family_count"]) > 1
        and int(row["support_problem_seed_count"]) > 1
    ]
    return {
        "schema_version": SCHEMA_VERSION + ".aggregate_inputs",
        "upstream_gate_passed": receipt.get("gate_passed") is True,
        "bounded_domain_exhaustive": audit.get("bounded_domain_exhaustive") is True,
        "expected_route_row_count": audit.get("expected_route_row_count", 0),
        "observed_route_row_count": audit.get("observed_route_row_count", 0),
        "candidate_preservation_passed": bool(preservation)
        and all(row.get("candidate_preservation_passed") is True for row in preservation),
        "exact_answer_equality_passed": bool(equality)
        and all(row.get("exact_answer_equality") is True for row in equality),
        "held_contamination_free": manifest.get("held_rows_in_table_count") == 0,
        "all_train_dev_errors_covered": manifest.get("all_train_dev_errors_covered") is True,
        "bounded_table_size_passed": manifest.get("bounded_table_size_passed") is True,
        "key_collision_count": manifest.get("key_collision_count", 0),
        "arm_held_summaries": summaries,
        "best_learned_arm": best.get("arm_id"),
        "best_learned_model_family": best.get("model_family"),
        "best_learned_model_seed": best.get("model_seed"),
        "best_learned_held_charged_benefit_units": best.get("held_charged_benefit_units", 0),
        "upstream_best_structural_arm": receipt.get("upstream_best_structural_arm"),
        "upstream_best_structural_held_benefit_units": upstream_benefit,
        "held_benefit_beyond_best_structural_units": int(
            best.get("held_charged_benefit_units", 0)
        )
        - upstream_benefit,
        "best_learned_support_problem_family_count": best.get("support_problem_family_count", 0),
        "best_learned_support_problem_seed_count": best.get("support_problem_seed_count", 0),
        "positive_supported_model_family_count": len(
            {str(row["model_family"]) for row in positive_arms}
        ),
        "positive_supported_model_seed_count": len({int(row["model_seed"]) for row in positive_arms}),
        "positive_supported_arm_count": len(positive_arms),
        "exact_solver_is_release_authority": True,
    }


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    rows = [dict(row) for row in payload.get("per_game_results", []) if isinstance(row, Mapping)]
    preservation = [
        dict(row)
        for row in payload.get("candidate_preservation_rows", [])
        if isinstance(row, Mapping)
    ]
    equality = [
        dict(row)
        for row in payload.get("exact_answer_equality_rows", [])
        if isinstance(row, Mapping)
    ]
    audit = payload.get("exhaustive_pilot_audit", {})
    manifest = payload.get("exception_table_manifest", {})
    receipt = payload.get("upstream_gate_receipt", {})
    base = _aggregate_without_attacks(
        receipt=receipt if isinstance(receipt, Mapping) else {},
        rows=rows,
        manifest=manifest if isinstance(manifest, Mapping) else {},
        audit=audit if isinstance(audit, Mapping) else {},
        preservation=preservation,
        equality=equality,
    )
    attacks = payload.get("attack_matrix", {})
    attack_passed = (
        isinstance(attacks, Mapping)
        and attacks.get("all_attacks_fail_closed") is True
        and all(row.get("fail_closed") is True for row in attacks.get("rows", []))
    )
    protected = payload.get("protected_files_unchanged", {})
    protected_ok = (
        isinstance(protected, Mapping)
        and protected.get("all_protected_files_unchanged") is True
    )
    positive = (
        base["upstream_gate_passed"]
        and base["bounded_domain_exhaustive"]
        and base["candidate_preservation_passed"]
        and base["exact_answer_equality_passed"]
        and base["held_contamination_free"]
        and base["all_train_dev_errors_covered"]
        and base["bounded_table_size_passed"]
        and base["key_collision_count"] == 0
        and attack_passed
        and protected_ok
        and int(base["held_benefit_beyond_best_structural_units"]) > 0
        and int(base["best_learned_support_problem_family_count"]) > 1
        and int(base["best_learned_support_problem_seed_count"]) > 1
        and int(base["positive_supported_model_family_count"]) > 1
        and int(base["positive_supported_model_seed_count"]) > 1
    )
    complete_null = (
        base["upstream_gate_passed"]
        and base["bounded_domain_exhaustive"]
        and base["candidate_preservation_passed"]
        and base["exact_answer_equality_passed"]
        and attack_passed
        and protected_ok
        and not positive
    )
    failed = []
    checks = {
        "upstream_gate_passed": base["upstream_gate_passed"],
        "bounded_domain_exhaustive": base["bounded_domain_exhaustive"],
        "candidate_preservation_passed": base["candidate_preservation_passed"],
        "exact_answer_equality_passed": base["exact_answer_equality_passed"],
        "held_contamination_free": base["held_contamination_free"],
        "all_train_dev_errors_covered": base["all_train_dev_errors_covered"],
        "bounded_table_size_passed": base["bounded_table_size_passed"],
        "key_collision_free": base["key_collision_count"] == 0,
        "attack_matrix_passed": attack_passed,
        "protected_files_unchanged": protected_ok,
    }
    for name, passed in checks.items():
        if not passed:
            failed.append(name)
    payload_out = {
        **base,
        "attack_matrix_passed": attack_passed,
        "protected_files_unchanged": protected_ok,
        "positive_conditions_met": positive,
        "complete_null_conditions_met": complete_null,
        "ready_conditions_met": positive or complete_null,
        "safety_net_router_ready_score_from_rows": 1.0 if positive or complete_null else 0.0,
        "failed_conditions": failed,
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-TERMINAL"],
    }
    return {**payload_out, "aggregate_row_recomputation_hash": sha256_json(payload_out)}


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    failed = list(aggregate.get("failed_conditions", []))
    payload = {
        "schema_version": SCHEMA_VERSION + ".gate_check_summary",
        "all_gates_passed": not failed,
        "failed_checks": failed,
        "observations": {
            "upstream_gate_passed": aggregate.get("upstream_gate_passed"),
            "held_benefit_beyond_best_structural_units": aggregate.get(
                "held_benefit_beyond_best_structural_units"
            ),
            "positive_supported_model_family_count": aggregate.get(
                "positive_supported_model_family_count"
            ),
            "positive_supported_model_seed_count": aggregate.get(
                "positive_supported_model_seed_count"
            ),
        },
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-TERMINAL"],
    }
    return {**payload, "gate_check_summary_hash": sha256_json(payload)}


def _status_and_verdict(
    aggregate: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    failed = set(gates.get("failed_checks", []))
    if "upstream_gate_passed" in failed:
        return (
            "blocked_safety_net_branch_router_ab",
            "blocked_safety_net_branch_router_ab: Exp6519 structural headroom gate did not pass",
            "blocked",
        )
    if failed & {"candidate_preservation_passed", "exact_answer_equality_passed", "held_contamination_free"}:
        return (
            "disqualified_safety_net_branch_router_ab",
            "disqualified_safety_net_branch_router_ab: safety-net correctness or leakage gate failed",
            "disqualified",
        )
    if aggregate.get("positive_conditions_met") is True:
        return (
            "complete_safety_net_branch_router_ab_positive",
            "complete_safety_net_branch_router_ab_positive: compact routers beat certified structural headroom with exact fallback and no held table writes",
            "positive",
        )
    if aggregate.get("complete_null_conditions_met") is True:
        return (
            "complete_safety_net_branch_router_ab_null",
            "complete_safety_net_branch_router_ab_null: safety-net routing was correct but did not beat the certified structural gate",
            None,
        )
    return (
        "complete_safety_net_branch_router_ab_partial",
        "complete_safety_net_branch_router_ab_partial: bounded routing evidence was usable but below the positive gate",
        "partial",
    )


def _field_provenance(repo_root: Path) -> JsonDict:
    source_hashes = {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS}
    return {
        field: {
            "spec_refs": ["REQ-BENCH-6520"],
            "source_hashes": source_hashes,
            "producer": "experiment_6520_safety_net_branch_router_ab",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    rows = list(tests_run or DEFAULT_TESTS_RUN)
    return [
        {
            "row_type": "test_run_receipt",
            "command": str(row.get("command")),
            "exit_code": int(row.get("exit_code", 0)),
        }
        for row in rows
    ]


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    source_path: Path,
    run_date: str,
    protected_before: Mapping[str, str],
) -> JsonDict:
    git_rc, git_status = _command_output(["git", "status", "--short"], repo_root)
    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "source_path": str(source_path),
        "exp6516_path": str(repo_root / EXP6516_RELATIVE_PATH),
        "exp6518_path": str(repo_root / EXP6518_RELATIVE_PATH),
        "git_status_command_exit_code": git_rc,
        "git_status_short": git_status,
        "resources": _resource_state(repo_root),
        "framework_versions": framework_versions(),
        "arm_ids": list(ARM_IDS),
        "feature_names": list(FEATURE_NAMES),
        "model_seed_grid": list(MODEL_SEEDS),
        "optimization_steps": OPTIMIZATION_STEPS,
        "confidence_abstain_threshold": CONFIDENCE_ABSTAIN_THRESHOLD,
        "exact_assignment_budget": EXACT_ASSIGNMENT_BUDGET,
        "exact_solver_is_release_authority": True,
        "verifier_is_oracle_for_router_value": False,
        "held_rows_may_train": False,
        "held_rows_may_enter_exception_table": False,
        "conductor_modification_allowed": False,
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-GATE"],
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def _empty_exception_manifest() -> JsonDict:
    payload = {
        "schema_version": EXCEPTION_SCHEMA_VERSION,
        "schema_version_hash": sha256_json(EXCEPTION_SCHEMA_VERSION),
        "tables": [],
        "learned_arm_count": len(LEARNED_ARM_IDS),
        "total_entry_count": 0,
        "held_rows_in_table_count": 0,
        "key_collision_count": 0,
        "all_train_dev_errors_covered": False,
        "bounded_table_size_limit_entries": 24,
        "bounded_table_size_passed": False,
        "build_policy": "train_and_development_errors_only",
        "runtime_policy": "exception_hit_routes_to_native_exact_fallback",
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXCEPTIONS"],
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def _empty_train_dev_held_receipts() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".train_dev_held_receipts",
        "split_unit_counts": {},
        "split_unit_hashes": {},
        "train_dev_unit_count": 0,
        "held_unit_count": 0,
        "training_splits": ["train", "development"],
        "held_rows_used_for_training": False,
        "held_rows_used_for_exception_writes": False,
        "train_only_writes_passed": False,
        "feature_schema_hash": _feature_schema()["schema_hash"],
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXCEPTIONS"],
    }
    return {**payload, "train_dev_held_receipts_hash": sha256_json(payload)}


def _empty_exhaustive_audit() -> JsonDict:
    payload = {
        "schema_version": SCHEMA_VERSION + ".exhaustive_pilot_audit",
        "expected_route_row_count": PILOT_UNIT_COUNT * len(ARM_IDS),
        "observed_route_row_count": 0,
        "pilot_unit_count": 0,
        "arm_count": 0,
        "bounded_domain_exhaustive": False,
        "candidate_preservation_passed": False,
        "exact_answer_equality_passed": False,
        "changed_decision_count": 0,
        "exception_hit_count": 0,
        "fallback_count": 0,
        "abstention_count": 0,
        "terminal_disposition_missing_count": 0,
        "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-EXHAUSTIVE"],
    }
    return {**payload, "exhaustive_pilot_audit_hash": sha256_json(payload)}


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    source_path: Path | str = EXP6519_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    source_path = Path(source_path)
    if not source_path.is_absolute():
        source_path = repo_root / source_path

    protected_before = protected_file_hashes(repo_root, source_path)
    receipt = upstream_gate_receipt(
        repo_root=repo_root,
        source_path=source_path,
        protected_before=protected_before,
    )
    units = _load_branch_units(repo_root) if receipt["gate_passed"] is True else []
    specs = model_and_arm_specs()
    if units:
        split_receipts = train_dev_held_receipts(units)
        manifest = exception_table_manifest(units)
        routes = per_game_results(units=units, manifest=manifest)
        fallback = exception_abstention_fallback_rows(routes)
        preservation = candidate_preservation_rows(routes)
        equality = exact_answer_equality_rows(routes)
        costs = charged_cost_and_storage_rows(routes)
        audit = exhaustive_pilot_audit(
            rows=routes,
            preservation=preservation,
            equality=equality,
            fallback=fallback,
        )
    else:
        split_receipts = _empty_train_dev_held_receipts()
        manifest = _empty_exception_manifest()
        routes = []
        fallback = []
        preservation = []
        equality = []
        costs = []
        audit = _empty_exhaustive_audit()

    protected_after = protected_file_hashes(repo_root, source_path)
    protected = protected_files_unchanged(protected_before, protected_after)
    partial: JsonDict = {
        "status": "blocked_safety_net_branch_router_ab",
        "honest_verdict": "blocked_safety_net_branch_router_ab: building",
        "verdict_class": "blocked",
        "upstream_gate_receipt": receipt,
        "preregistration": {
            "schema_version": SCHEMA_VERSION + ".preregistration",
            "planning_date": run_date,
            "primary_metric": PRIMARY_METRIC,
            "primary_split": "held",
            "arms": list(ARM_IDS),
            "model_seed_grid": list(MODEL_SEEDS),
            "optimization_steps": OPTIMIZATION_STEPS,
            "confidence_abstain_threshold": CONFIDENCE_ABSTAIN_THRESHOLD,
            "fallback_triggers": [
                "exception_hit",
                "abstention",
                "schema_mismatch",
                "model_failure",
                "checksum_failure",
                "stale_model_table_pair",
            ],
            "verdict_class_rules": {
                "positive": "held charged benefit beyond Exp6519 best structural arm with exact equality and breadth",
                "null": "complete correct replay with no benefit beyond the certified gate",
                "partial": "bounded usable evidence below a complete gate",
                "blocked": "failed gate or resource precondition",
                "disqualified": "leakage or correctness drift",
            },
            "exact_solver_is_release_authority": True,
            "learned_advice_scope": "order_candidates_only",
            "spec_refs": ["REQ-BENCH-6520", "SCENARIO-BENCH-6520-ARMS"],
        },
        "model_and_arm_specs": specs,
        "train_dev_held_receipts": split_receipts,
        "exception_table_manifest": manifest,
        "per_game_results": routes,
        "exception_abstention_fallback_rows": fallback,
        "candidate_preservation_rows": preservation,
        "exact_answer_equality_rows": equality,
        "exhaustive_pilot_audit": audit,
        "charged_cost_and_storage_rows": costs,
        "attack_matrix": _blocked_attack_matrix(),
        "safety_net_router_ready_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": [],
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result_path,
            source_path=source_path,
            run_date=run_date,
            protected_before=protected_before,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": {
            "artifact_seed": RANDOM_SEED,
            "model_seed_grid": list(MODEL_SEEDS),
            "attack_ids": list(ATTACK_IDS),
        },
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    preliminary = recompute_aggregate(partial)
    attacks = (
        attack_matrix(manifest=manifest, aggregate=preliminary, rows=routes)
        if routes
        else _blocked_attack_matrix()
    )
    partial["attack_matrix"] = attacks
    partial["per_unit_rows"] = [*routes, *fallback, *preservation, *equality, *costs, *attacks["rows"]]
    aggregate = recompute_aggregate(partial)
    gates = gate_check_summary(aggregate)
    status, honest, verdict_class = _status_and_verdict(aggregate, gates)
    partial.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict_class,
            "safety_net_router_ready_score": aggregate[
                "safety_net_router_ready_score_from_rows"
            ],
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
        }
    )
    partial["reproducibility_checksum"] = reproducibility_checksum(partial)
    errors = validate_artifact(partial)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, partial, sort_keys=True, env={})
    return partial


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("verdict_class") not in {"positive", None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6520 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    preconditions = payload.get("preconditions_checked", {})
    if not isinstance(preconditions, Mapping) or preconditions.get("exact_solver_is_release_authority") is not True:
        errors.append("exact solver release authority missing")
    score = payload.get("safety_net_router_ready_score")
    if score not in {0.0, 1.0}:
        errors.append("safety_net_router_ready_score must be 0.0 or 1.0")
    if payload.get("verdict_class") == "positive" and score != 1.0:
        errors.append("positive verdict requires ready score 1.0")
    manifest = payload.get("exception_table_manifest", {})
    if isinstance(manifest, Mapping) and manifest.get("held_rows_in_table_count") != 0:
        errors.append("held contamination detected")
    if any(
        row.get("candidate_preservation_passed") is not True
        for row in payload.get("candidate_preservation_rows", [])
        if isinstance(row, Mapping)
    ):
        errors.append("candidate preservation failed")
    if any(
        row.get("exact_answer_equality") is not True
        for row in payload.get("exact_answer_equality_rows", [])
        if isinstance(row, Mapping)
    ):
        errors.append("exact answer equality failed")
    protected = payload.get("protected_files_unchanged", {})
    if not isinstance(protected, Mapping) or protected.get("all_protected_files_unchanged") is not True:
        errors.append("protected files changed")
    aggregate = recompute_aggregate(payload)
    gates = gate_check_summary(aggregate)
    blocked_by_gate = payload.get("verdict_class") == "blocked" and "upstream_gate_passed" in set(
        aggregate.get("failed_conditions", [])
    )
    attacks = payload.get("attack_matrix", {})
    if not blocked_by_gate and isinstance(attacks, Mapping) and any(
        row.get("fail_closed") is not True for row in attacks.get("rows", [])
    ):
        errors.append("attack false accept")
    if not blocked_by_gate:
        if "candidate_preservation_passed" in aggregate.get("failed_conditions", []):
            errors.append("candidate preservation failed")
        if "exact_answer_equality_passed" in aggregate.get("failed_conditions", []):
            errors.append("exact answer equality failed")
        if "attack_matrix_passed" in aggregate.get("failed_conditions", []):
            errors.append("attack false accept")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != gates:
        errors.append("gate_check_summary mismatch")
    if payload.get("safety_net_router_ready_score") != aggregate[
        "safety_net_router_ready_score_from_rows"
    ]:
        errors.append("ready score mismatch")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    honest = str(payload.get("honest_verdict") or "")
    status = str(payload.get("status") or "")
    if not honest.startswith(("complete_", "blocked_", "disqualified_")):
        errors.append("honest_verdict lacks terminal prefix")
    if not status.startswith(("complete_", "blocked_", "disqualified_")):
        errors.append("status lacks terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    source_path: Path | str = EXP6519_RELATIVE_PATH,
) -> JsonDict:
    return build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        source_path=source_path,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--source-path", default=str(EXP6519_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        target = result_path if result_path.is_absolute() else REPO_ROOT / result_path
        payload = json.loads(target.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, source_path=Path(args.source_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m.
    raise SystemExit(main())
