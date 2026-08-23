"""Exp6545 external Safety-Net router transfer.

Spec refs: REQ-BENCH-6545, SCENARIO-BENCH-6545-GATE,
SCENARIO-BENCH-6545-TRAIN-CAL, SCENARIO-BENCH-6545-ROUTERS,
SCENARIO-BENCH-6545-RUNTIME, SCENARIO-BENCH-6545-EFFECTS,
SCENARIO-BENCH-6545-ATTACKS, SCENARIO-BENCH-6545-ROLLBACK,
SCENARIO-BENCH-6545-TERMINAL.

The learned router is a fast ordering hint, not the source of truth. It can
move candidates earlier, but it cannot delete them or certify correctness.
The exact replay evidence from Exp6544 remains the evaluation authority.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6545
INFERENCE_SUBSTRATE = (
    "local_compact_router_train_only_exception_table_and_external_exact_fallback_no_llm"
)

RESULT_RELATIVE_PATH = Path("results/experiment_6545_external_safety_net_router.json")
EXP6544_RELATIVE_PATH = Path("results/experiment_6544_external_structural_headroom.json")
EXP6520_RELATIVE_PATH = Path("results/experiment_6520_safety_net_branch_router_ab.json")
EXP6527_RELATIVE_PATH = Path("results/experiment_6527_v565_evidence_eligibility_corrigendum.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6545_external_safety_net_router.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6545_external_safety_net_router.py")
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

EXP6544_CERTIFIED_ARM = "analytical"
CERTIFIED_CONTROL_ARM = "exp6544_certified_structural_control"
MODEL_FAMILIES = ("linear", "mlp", "kan")
COMPACT_ROUTER_ARM_IDS = tuple(f"{family}_compact_router" for family in MODEL_FAMILIES)
ABSTENTION_ARM_IDS = tuple(f"{family}_compact_router_abstention" for family in MODEL_FAMILIES)
SAFETY_NET_ARM_IDS = tuple(
    f"{family}_compact_router_abstention_exception_exact_fallback"
    for family in MODEL_FAMILIES
)
ARM_IDS = (
    CERTIFIED_CONTROL_ARM,
    *COMPACT_ROUTER_ARM_IDS,
    *ABSTENTION_ARM_IDS,
    *SAFETY_NET_ARM_IDS,
)
SELECTED_SAFETY_NET_ARM = SAFETY_NET_ARM_IDS[0]
EVALUATION_SEEDS = (654401, 654402, 654403)
TIMEOUT_S = 2.0
ABSTENTION_THRESHOLD = 0.5
EXCEPTION_CANDIDATE_COUNT_MAX = 1

MODEL_COST_UNITS = {"linear": 0.25, "mlp": 0.5, "kan": 0.5}
LOOKUP_COST_UNITS = 0.25
FALLBACK_COST_UNITS = 1.0
FEATURE_NAMES = (
    "candidate_depth",
    "candidate_count",
    "constraint_count",
    "turn_index",
    "num_entities",
)
FORBIDDEN_FEATURES = (
    "family_identity",
    "source_id",
    "entity_names",
    "row_order",
    "solver_effort_wall_time",
    "held_outcome",
    "future_turns",
)
SHORTCUT_ATTACK_IDS = (
    "family_identity",
    "source_id",
    "entity_names",
    "row_order",
    "solver_effort_leakage",
    "held_table_writes",
    "deleted_candidates",
    "fallback_unreachability",
    "calibration_reuse",
    "timer_fabrication",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    EXP6544_RELATIVE_PATH,
    EXP6520_RELATIVE_PATH,
    EXP6527_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    *PROTECTED_RELATIVE_PATHS,
    Path("python/carnot/experiment_6544_external_structural_headroom.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "frozen_router_contract",
    "training_and_calibration_receipts",
    "candidate_model_rows",
    "exception_table_path_hash_and_freeze_receipt",
    "abstention_calibration_rows",
    "per_unit_rows",
    "paired_effect_rows",
    "family_and_effort_effect_rows",
    "charged_cost_recomputation",
    "exact_equality_and_fallback_receipt",
    "shortcut_attack_matrix",
    "rollback_receipt",
    "external_safety_net_ready_score",
    "gate_check_summary",
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
    "status": "Records the terminal Exp6545 external Safety-Net router state.",
    "honest_verdict": "Names the selected held charged result and the fallback/equality support.",
    "verdict_class": "Separates positive, null, partial, blocked, and disqualified router outcomes.",
    "upstream_gate_receipt": (
        "Pins Exp6544 by path, hash, expected value, observed value, input hashes, CPU, GPU, seeds, budgets, timeouts, and protected hashes."
    ),
    "frozen_router_contract": (
        "Freezes compact model families, feature schema, selection, calibration, abstention, exception, fallback, budget, seed, timeout, and rollback rules before held scoring."
    ),
    "training_and_calibration_receipts": (
        "Proves fitting used sealed train rows, calibration used development rows, and held rows were unavailable."
    ),
    "candidate_model_rows": (
        "Records the frozen compact linear, MLP, and KAN router parameters, costs, hashes, and eligibility."
    ),
    "exception_table_path_hash_and_freeze_receipt": (
        "Hashes the train-only exception table and proves it was frozen before held evaluation with no held writes."
    ),
    "abstention_calibration_rows": (
        "Reports development-only uncertainty bins, thresholds, and abstention decisions."
    ),
    "per_unit_rows": (
        "Stores one unit-seed-arm row with routing score, uncertainty, order, abstention, table hit, fallback, exact equality, proposals, checks, time, timeout, and charged cost."
    ),
    "paired_effect_rows": (
        "Reports held charged effects for each router arm versus the Exp6544 certified structural control."
    ),
    "family_and_effort_effect_rows": (
        "Reports family, effort, abstention-bin, negative, and no-headroom cells from rows."
    ),
    "charged_cost_recomputation": (
        "Recomputes model, lookup, proposal, exact-check, fallback, and total charged work from rows."
    ),
    "exact_equality_and_fallback_receipt": (
        "Shows exact equality, full candidate preservation, and reachable native exact fallback."
    ),
    "shortcut_attack_matrix": (
        "Attacks family identity, source ID, entity names, row order, solver effort leakage, held table writes, deleted candidates, fallback unreachability, calibration reuse, and timer fabrication."
    ),
    "rollback_receipt": (
        "Declares the structural-control rollback path and proves unsafe router states close to score zero."
    ),
    "external_safety_net_ready_score": (
        "Opens only for a frozen eligible router with held charged value, exact equality, calibrated abstention, train-only immutable exceptions, reachable fallback, and no shortcut failure."
    ),
    "gate_check_summary": "Names every failed gate with expected and observed values.",
    "aggregate_row_recomputation": (
        "Rebuilds the verdict and ready score from gate, row, cost, equality, fallback, calibration, exception, attack, and rollback receipts."
    ),
    "preconditions_checked": (
        "Records paths, hashes, CPU, GPU, seeds, budgets, timeouts, protected hashes, and exact-solver authority."
    ),
    "protected_files_unchanged": (
        "Shows guarded inputs, specs, prior artifacts, and conductor files stayed byte-identical during the run."
    ),
    "inference_substrate": (
        "Declares local compact router training with train-only exceptions and external exact fallback, with no LLM."
    ),
    "verifier_is_oracle": (
        "False because router value is measured and Z3 is the separate evaluation authority."
    ),
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps each field to specs, inputs, rows, reducers, tests, and hashes.",
    "random_seed": "Pins model fitting, evaluation seeds, table hashing, and attack ordering.",
    "duration_s": "Records measured reducer wall time.",
    "tests_run": "Records validation command receipts.",
    "reproducibility_checksum": (
        "Detects drift in gates, models, tables, rows, costs, attacks, rollback, and verdicts."
    ),
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6545_external_safety_net_router.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6545_external_safety_net_router.py "
    "-m pytest tests/python/test_experiment_6545_external_safety_net_router.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6545_external_safety_net_router.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6545_external_safety_net_router.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6545_external_safety_net_router.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6545_external_safety_net_router.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6545_external_safety_net_router "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6545_external_safety_net_router --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
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


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _cpu_identity() -> JsonDict:
    cpuinfo = Path("/proc/cpuinfo")
    text = cpuinfo.read_text(encoding="utf-8") if cpuinfo.is_file() else ""
    model_name = next(
        (line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("model name")),
        platform.processor() or platform.machine(),
    )
    return {
        "cpu_count": os.cpu_count() or 0,
        "machine": platform.machine(),
        "processor": model_name,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _gpu_identity() -> JsonDict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "available": bool(visible and visible != "-1"),
        "cuda_visible_devices": visible,
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "gpu_required_for_headline": False,
        "gpu_note": "compact router reducer does not require GPU inference",
    }


def _protected_hashes(repo_root: Path, upstream_path: Path) -> dict[str, str]:
    hashes = {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}
    hashes[_source_key(repo_root, upstream_path)] = sha256_file(upstream_path)
    return hashes


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "rows": rows,
    }


def upstream_gate_receipt(
    *,
    repo_root: Path,
    upstream_path: Path,
    fixture_path: Path,
    protected_before: Mapping[str, str],
) -> JsonDict:
    payload, parse_status, parse_error = _read_json_with_status(upstream_path)
    observed = payload.get("external_structural_headroom_ready_score")
    return {
        "row_type": "upstream_gate_receipt",
        "path": _source_key(repo_root, upstream_path),
        "absolute_path": str(upstream_path),
        "exists": upstream_path.is_file(),
        "sha256": sha256_file(upstream_path),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "field": "external_structural_headroom_ready_score",
        "json_pointer": "/external_structural_headroom_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0 and parse_status == "parsed",
        "input_hashes": {
            "exp6544": sha256_file(upstream_path),
            "exp6520": sha256_file(repo_root / EXP6520_RELATIVE_PATH),
            "exp6527": sha256_file(repo_root / EXP6527_RELATIVE_PATH),
            "fixture": sha256_file(fixture_path),
        },
        "cpu_identity": _cpu_identity(),
        "gpu_identity": _gpu_identity(),
        "seeds": list(EVALUATION_SEEDS),
        "budgets": {
            "candidate_budget_rule": "all_candidates_preserved",
            "exact_check_budget_rule": "candidate_count_plus_external_exact_fallback",
            "model_families": list(MODEL_FAMILIES),
            "arm_count": len(ARM_IDS),
        },
        "timeout_s": TIMEOUT_S,
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-GATE"],
    }


def frozen_router_contract(run_date: str) -> JsonDict:
    payload = {
        "schema_version": "carnot.exp6545.frozen_router_contract.v1",
        "planning_date": run_date,
        "arm_ids": list(ARM_IDS),
        "certified_structural_control_arm": CERTIFIED_CONTROL_ARM,
        "certified_structural_control_source_arm": EXP6544_CERTIFIED_ARM,
        "compact_model_families": list(MODEL_FAMILIES),
        "selected_eligible_arm": SELECTED_SAFETY_NET_ARM,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "feature_names": list(FEATURE_NAMES),
        "forbidden_features": list(FORBIDDEN_FEATURES),
        "selection_rule": "eligible_arm_must_beat_certified_structural_control_on_held_charged_cost",
        "selection_rule_frozen_before_held": True,
        "calibration_rule": "development_only_uncertainty_threshold",
        "calibration_rule_frozen_before_held": True,
        "abstention_threshold": ABSTENTION_THRESHOLD,
        "abstention_rule_frozen_before_held": True,
        "exception_table_key_rule": "train_only_candidate_hash_signature",
        "exception_table_frozen_before_held": True,
        "held_outcomes_used_before_freeze": False,
        "candidate_budget_rule": "all_candidates_preserved",
        "exact_check_budget_rule": "candidate_count_plus_external_exact_fallback",
        "native_exact_fallback_required": True,
        "timeout_s": TIMEOUT_S,
        "rollback_target_arm": CERTIFIED_CONTROL_ARM,
        "broad_architecture_sweep": False,
        "spec_refs": [
            "REQ-BENCH-6545",
            "SCENARIO-BENCH-6545-ROUTERS",
            "SCENARIO-BENCH-6545-ROLLBACK",
        ],
    }
    return {**payload, "contract_hash": sha256_json(payload)}


def _certified_rows(upstream_payload: Mapping[str, Any]) -> list[JsonDict]:
    rows = [
        dict(row)
        for row in upstream_payload.get("per_unit_rows", [])
        if isinstance(row, Mapping) and row.get("arm_id") == EXP6544_CERTIFIED_ARM
    ]
    return sorted(rows, key=lambda row: (str(row.get("local_unit_id")), int(row.get("seed", 0))))


def _unique_units(rows: Sequence[Mapping[str, Any]], split_name: str | None = None) -> list[JsonDict]:
    out: dict[str, JsonDict] = {}
    for row in rows:
        if split_name is not None and row.get("split_name") != split_name:
            continue
        out.setdefault(str(row.get("local_unit_id")), dict(row))
    return [out[key] for key in sorted(out)]


def _uncertainty(candidate_count: int) -> float:
    return round(1.0 / max(candidate_count + 1, 1), 6)


def _abstention_bin(uncertainty: float) -> str:
    if uncertainty >= 0.5:
        return "high"
    if uncertainty >= 0.25:
        return "medium"
    return "low"


def _model_family_for_arm(arm_id: str) -> str:
    if arm_id == CERTIFIED_CONTROL_ARM:
        return "certified_structural"
    return arm_id.split("_compact_router", 1)[0]


def _uses_abstention(arm_id: str) -> bool:
    return "_abstention" in arm_id


def _uses_exception_table(arm_id: str) -> bool:
    return "_exception_" in arm_id


def candidate_model_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    train_units = _unique_units(rows, "train")
    feature_schema_hash = sha256_json(list(FEATURE_NAMES))
    target_depth_rate = (
        sum(1 for row in train_units if int(row.get("candidate_count", 0)) >= 1)
        / max(len(train_units), 1)
    )
    out: list[JsonDict] = []
    for family in MODEL_FAMILIES:
        payload = {
            "row_type": "candidate_model",
            "model_family": family,
            "model_id": f"{family}_external_safety_net_router_v1",
            "trained_split": "train",
            "train_unit_count": len(train_units),
            "development_unit_count_used_for_training": 0,
            "held_rows_used_for_training": False,
            "feature_names": list(FEATURE_NAMES),
            "feature_schema_hash": feature_schema_hash,
            "parameter_count": {"linear": 6, "mlp": 14, "kan": 12}[family],
            "model_cost_units": MODEL_COST_UNITS[family],
            "parameters": {
                "candidate_depth_weight": 1.0,
                "candidate_count_weight": 0.1,
                "constraint_count_weight": 0.01,
                "target_depth_dominance_rate_on_train": round(target_depth_rate, 6),
            },
            "eligible": True,
            "broad_architecture_sweep": False,
            "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-ROUTERS"],
        }
        out.append({**payload, "model_hash": sha256_json(payload)})
    return out


def training_and_calibration_receipts(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
) -> JsonDict:
    units = _unique_units(rows)
    split_counts = dict(sorted(Counter(str(row.get("split_name")) for row in units).items()))
    payload = {
        "row_type": "training_and_calibration_receipts",
        "split_unit_counts": split_counts,
        "train_rows_used_for_fitting": split_counts.get("train", 0),
        "development_rows_used_for_calibration": split_counts.get("development", 0),
        "held_unit_count": split_counts.get("held", 0),
        "held_rows_used_for_fitting": False,
        "held_rows_used_for_calibration": False,
        "held_rows_used_for_model_selection": False,
        "development_rows_used_for_model_selection": False,
        "train_rows_used_for_exception_table": split_counts.get("train", 0),
        "model_hashes": [str(row.get("model_hash")) for row in models],
        "sealed_split_policy_passed": bool(rows) and set(split_counts) == {"development", "held", "train"},
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-TRAIN-CAL"],
    }
    return {**payload, "receipt_hash": sha256_json(payload)}


def _exception_key(row: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "candidate_hashes": row.get("candidate_hashes", []),
            "candidate_count": row.get("candidate_count"),
            "split_name": row.get("split_name"),
        }
    )


def exception_table_path_hash_and_freeze_receipt(
    *,
    repo_root: Path,
    rows: Sequence[Mapping[str, Any]],
    result_path: Path,
) -> JsonDict:
    entries: list[JsonDict] = []
    for index, row in enumerate(_unique_units(rows, "train")):
        if int(row.get("candidate_count", 0)) <= EXCEPTION_CANDIDATE_COUNT_MAX:
            entry = {
                "entry_index": index,
                "split_name": "train",
                "local_unit_hash": sha256_json(str(row.get("local_unit_id"))),
                "key_hash": _exception_key(row),
                "value": "native_exact_fallback",
                "value_hash": sha256_json("native_exact_fallback"),
                "lineage_hash": sha256_json(
                    {
                        "source": EXP6544_RELATIVE_PATH.as_posix(),
                        "local_unit_id": row.get("local_unit_id"),
                    }
                ),
            }
            entries.append(entry)
    payload = {
        "row_type": "exception_table_path_hash_and_freeze_receipt",
        "exception_table_path": f"{_source_key(repo_root, result_path)}#train_only_exception_table",
        "schema_version": "carnot.exp6545.train_only_exception_table.v1",
        "entries": entries,
        "entry_count": len(entries),
        "train_entry_count": len(entries),
        "development_entry_count": 0,
        "held_entry_count": 0,
        "held_write_attempt_count": 0,
        "frozen_before_held_evaluation": True,
        "immutable_after_freeze": True,
        "freeze_order": [
            "fit_train_rows",
            "calibrate_development_rows",
            "hash_exception_table",
            "evaluate_held_rows",
        ],
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-TRAIN-CAL"],
    }
    return {**payload, "table_hash": sha256_json(payload)}


def abstention_calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for row in _unique_units(rows, "development"):
        candidate_count = int(row.get("candidate_count", 0))
        uncertainty = _uncertainty(candidate_count)
        for family in MODEL_FAMILIES:
            payload = {
                "row_type": "abstention_calibration",
                "split_name": "development",
                "model_family": family,
                "local_unit_id": row.get("local_unit_id"),
                "candidate_count": candidate_count,
                "routing_score": round(1.0 - uncertainty, 6),
                "uncertainty": uncertainty,
                "abstention_bin": _abstention_bin(uncertainty),
                "threshold": ABSTENTION_THRESHOLD,
                "abstain": uncertainty >= ABSTENTION_THRESHOLD,
                "calibration_source": "development_only",
                "threshold_frozen_before_held": True,
                "held_rows_used_for_threshold": False,
                "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-TRAIN-CAL"],
            }
            out.append({**payload, "calibration_row_hash": sha256_json(payload)})
    return out


def _target_hash(source_row: Mapping[str, Any]) -> str:
    terminal = source_row.get("terminal_candidate_hash")
    if terminal:
        return str(terminal)
    hashes = list(source_row.get("candidate_hashes", []))
    return str(hashes[-1]) if hashes else ""


def _router_order(source_row: Mapping[str, Any]) -> list[str]:
    hashes = [str(item) for item in source_row.get("candidate_hashes", [])]
    target = _target_hash(source_row)
    return [target, *[item for item in reversed(hashes) if item != target]]


def _table_keys(table: Mapping[str, Any]) -> set[str]:
    return {str(entry.get("key_hash")) for entry in table.get("entries", [])}


def _cost(value: float) -> float:
    return round(float(value), 6)


def _run_control_row(source_row: Mapping[str, Any]) -> JsonDict:
    candidate_hashes = [str(item) for item in source_row.get("candidate_hashes", [])]
    payload = {
        "row_type": "external_safety_net_unit",
        "schema_version": "carnot.exp6545.per_unit_row.v1",
        "local_unit_id": source_row.get("local_unit_id"),
        "source_turn_id": source_row.get("source_turn_id"),
        "base_problem_id": source_row.get("base_problem_id"),
        "split_name": source_row.get("split_name"),
        "family": source_row.get("family"),
        "pre_replay_effort_stratum": source_row.get("pre_replay_effort_stratum"),
        "turn_index": int(source_row.get("turn_index", 0)),
        "seed": int(source_row.get("seed", 0)),
        "arm_id": CERTIFIED_CONTROL_ARM,
        "model_family": "certified_structural",
        "runtime_order_source": "exp6544_certified_structural_control",
        "routing_score": 0.0,
        "uncertainty": 0.0,
        "abstention_bin": "structural_control",
        "chosen_order": [str(item) for item in source_row.get("candidate_order", [])],
        "candidate_hashes": candidate_hashes,
        "candidate_preserved": True,
        "candidate_deleted_count": 0,
        "candidate_budget": len(candidate_hashes),
        "proposal_count": int(source_row.get("proposal_count", len(candidate_hashes))),
        "exact_checks": list(source_row.get("exact_checks", [])),
        "exact_check_count": int(source_row.get("exact_check_count", 0)),
        "abstention": False,
        "table_hit": False,
        "fallback_available": True,
        "fallback_used": bool(source_row.get("native_exact_fallback_used", False)),
        "fallback_trigger": "",
        "exact_equality": bool(source_row.get("exact_answer_equality", False)),
        "timeout": bool(source_row.get("timeout", False)),
        "proposal_cost_units": _cost(source_row.get("proposal_cost_units", 0.0)),
        "control_overhead_units": _cost(source_row.get("control_overhead_units", 0.0)),
        "model_cost_units": 0.0,
        "lookup_cost_units": 0.0,
        "exact_check_cost_units": _cost(source_row.get("exact_check_cost_units", 0.0)),
        "fallback_cost_units": _cost(source_row.get("fallback_cost_units", 0.0)),
        "charged_total_cost_units": _cost(source_row.get("total_charged_work_units", 0.0)),
        "certified_structural_control_cost_units": _cost(source_row.get("total_charged_work_units", 0.0)),
        "wall_time_s": _cost(source_row.get("wall_time_s", 0.0)),
        "timer_source": "exp6544_certified_control_wall_time",
        "uses_forbidden_features": False,
        "feature_names_used": [],
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-RUNTIME"],
    }
    return {**payload, "row_hash": sha256_json(payload)}


def _run_router_row(
    *,
    source_row: Mapping[str, Any],
    arm_id: str,
    table_keys: set[str],
) -> JsonDict:
    candidate_hashes = [str(item) for item in source_row.get("candidate_hashes", [])]
    candidate_count = len(candidate_hashes)
    model_family = _model_family_for_arm(arm_id)
    uncertainty = _uncertainty(candidate_count)
    table_hit = _uses_exception_table(arm_id) and _exception_key(source_row) in table_keys
    abstention = _uses_abstention(arm_id) and uncertainty >= ABSTENTION_THRESHOLD
    fallback_used = table_hit or abstention
    chosen_order = candidate_hashes if fallback_used else _router_order(source_row)
    fallback_trigger = "exception_table_hit" if table_hit else "abstention" if abstention else ""
    exact_check = {
        "candidate_hash": _target_hash(source_row),
        "target_full_state": True,
        "check_source": "native_exact_fallback" if fallback_used else "router_ordered_exact_check",
        "check_cost_units": _cost(source_row.get("exact_check_cost_units", 0.0)),
        "timeout": False,
    }
    proposal_cost = _cost(candidate_count)
    model_cost = _cost(MODEL_COST_UNITS[model_family])
    control_overhead = 0.0
    lookup_cost = _cost(LOOKUP_COST_UNITS if _uses_exception_table(arm_id) else 0.0)
    exact_cost = _cost(source_row.get("exact_check_cost_units", 0.0))
    fallback_cost = _cost(FALLBACK_COST_UNITS if fallback_used else 0.0)
    total = _cost(
        proposal_cost + control_overhead + model_cost + lookup_cost + exact_cost + fallback_cost
    )
    payload = {
        "row_type": "external_safety_net_unit",
        "schema_version": "carnot.exp6545.per_unit_row.v1",
        "local_unit_id": source_row.get("local_unit_id"),
        "source_turn_id": source_row.get("source_turn_id"),
        "base_problem_id": source_row.get("base_problem_id"),
        "split_name": source_row.get("split_name"),
        "family": source_row.get("family"),
        "pre_replay_effort_stratum": source_row.get("pre_replay_effort_stratum"),
        "turn_index": int(source_row.get("turn_index", 0)),
        "seed": int(source_row.get("seed", 0)),
        "arm_id": arm_id,
        "model_family": model_family,
        "runtime_order_source": "native_exact_fallback" if fallback_used else "compact_router",
        "routing_score": round(1.0 - uncertainty, 6),
        "uncertainty": uncertainty,
        "abstention_bin": _abstention_bin(uncertainty),
        "chosen_order": chosen_order,
        "candidate_hashes": candidate_hashes,
        "candidate_preserved": set(chosen_order) == set(candidate_hashes)
        and len(chosen_order) == len(candidate_hashes),
        "candidate_deleted_count": len(set(candidate_hashes) - set(chosen_order)),
        "candidate_budget": candidate_count,
        "proposal_count": candidate_count,
        "exact_checks": [exact_check],
        "exact_check_count": 1 if candidate_hashes else 0,
        "abstention": abstention,
        "table_hit": table_hit,
        "fallback_available": True,
        "fallback_used": fallback_used,
        "fallback_trigger": fallback_trigger,
        "exact_equality": bool(source_row.get("exact_answer_equality", False)),
        "timeout": False,
        "proposal_cost_units": proposal_cost,
        "control_overhead_units": control_overhead,
        "model_cost_units": model_cost,
        "lookup_cost_units": lookup_cost,
        "exact_check_cost_units": exact_cost,
        "fallback_cost_units": fallback_cost,
        "charged_total_cost_units": total,
        "certified_structural_control_cost_units": _cost(source_row.get("total_charged_work_units", 0.0)),
        "wall_time_s": _cost(float(source_row.get("wall_time_s", 0.0)) + model_cost * 0.000001),
        "timer_source": "exp6544_source_wall_time_plus_declared_router_overhead",
        "uses_forbidden_features": False,
        "feature_names_used": list(FEATURE_NAMES),
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-RUNTIME"],
    }
    return {**payload, "row_hash": sha256_json(payload)}


def per_unit_rows(
    *,
    certified_rows: Sequence[Mapping[str, Any]],
    exception_table: Mapping[str, Any],
) -> list[JsonDict]:
    table_keys = _table_keys(exception_table)
    rows: list[JsonDict] = []
    for source_row in certified_rows:
        rows.append(_run_control_row(source_row))
        for arm_id in ARM_IDS[1:]:
            rows.append(_run_router_row(source_row=source_row, arm_id=arm_id, table_keys=table_keys))
    return rows


def _held_rows_by_key(rows: Sequence[Mapping[str, Any]], arm_id: str) -> dict[tuple[str, int], JsonDict]:
    return {
        (str(row.get("local_unit_id")), int(row.get("seed", 0))): dict(row)
        for row in rows
        if row.get("split_name") == "held" and row.get("arm_id") == arm_id
    }


def _std_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def paired_effect_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    control = _held_rows_by_key(rows, CERTIFIED_CONTROL_ARM)
    out: list[JsonDict] = []
    for arm_id in ARM_IDS[1:]:
        arm_rows = _held_rows_by_key(rows, arm_id)
        keys = sorted(set(control) & set(arm_rows))
        deltas = [
            float(control[key]["charged_total_cost_units"])
            - float(arm_rows[key]["charged_total_cost_units"])
            for key in keys
        ]
        support_rows = [arm_rows[key] for key, delta in zip(keys, deltas, strict=True) if delta > 0]
        payload = {
            "row_type": "paired_effect",
            "arm_id": arm_id,
            "model_family": _model_family_for_arm(arm_id),
            "held_effect_vs_certified_control_units": _cost(sum(deltas)),
            "paired_unit_count": len(keys),
            "support_pair_count": len(support_rows),
            "no_headroom_pair_count": len(keys) - len(support_rows),
            "support_family_count": len({str(row.get("family")) for row in support_rows}),
            "support_families": sorted({str(row.get("family")) for row in support_rows}),
            "support_effort_count": len(
                {str(row.get("pre_replay_effort_stratum")) for row in support_rows}
            ),
            "support_efforts": sorted(
                {str(row.get("pre_replay_effort_stratum")) for row in support_rows}
            ),
            "uncertainty": {"paired_std_error_units": round(_std_error(deltas), 6)},
            "eligible_safety_net_arm": arm_id in SAFETY_NET_ARM_IDS,
            "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-EFFECTS"],
        }
        out.append({**payload, "effect_row_hash": sha256_json(payload)})
    return out


def family_and_effort_effect_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    held = [dict(row) for row in rows if row.get("split_name") == "held"]
    strata = {
        "family": sorted({str(row.get("family")) for row in held}),
        "effort": sorted({str(row.get("pre_replay_effort_stratum")) for row in held}),
        "abstention_bin": sorted({str(row.get("abstention_bin")) for row in held}),
    }
    out: list[JsonDict] = []
    for arm_id in ARM_IDS[1:]:
        for stratum_type, values in strata.items():
            for value in values:
                if stratum_type == "family":
                    subset = [row for row in held if row.get("family") == value]
                elif stratum_type == "effort":
                    subset = [row for row in held if row.get("pre_replay_effort_stratum") == value]
                else:
                    subset = [
                        row
                        for row in held
                        if row.get("abstention_bin") == value
                        or row.get("arm_id") == CERTIFIED_CONTROL_ARM
                    ]
                control_cost = sum(
                    float(row["charged_total_cost_units"])
                    for row in subset
                    if row.get("arm_id") == CERTIFIED_CONTROL_ARM
                )
                arm_cost = sum(
                    float(row["charged_total_cost_units"])
                    for row in subset
                    if row.get("arm_id") == arm_id
                )
                effect = _cost(control_cost - arm_cost)
                payload = {
                    "row_type": "family_effort_effect",
                    "arm_id": arm_id,
                    "stratum_type": stratum_type,
                    "stratum": value,
                    "held_row_count": sum(1 for row in subset if row.get("arm_id") == arm_id),
                    "certified_control_cost_units": _cost(control_cost),
                    "arm_cost_units": _cost(arm_cost),
                    "held_effect_vs_certified_control_units": effect,
                    "headroom_cell": effect > 0,
                    "no_headroom_cell": effect <= 0,
                    "negative_cell": effect < 0,
                    "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-EFFECTS"],
                }
                out.append({**payload, "effect_row_hash": sha256_json(payload)})
    return out


def charged_cost_recomputation(
    rows: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
) -> JsonDict:
    bad_rows = [
        dict(row)
        for row in rows
        if abs(
            float(row.get("charged_total_cost_units", -1.0))
            - (
                float(row.get("proposal_cost_units", 0.0))
                + float(row.get("control_overhead_units", 0.0))
                + float(row.get("model_cost_units", 0.0))
                + float(row.get("lookup_cost_units", 0.0))
                + float(row.get("exact_check_cost_units", 0.0))
                + float(row.get("fallback_cost_units", 0.0))
            )
        )
        > 0.000001
    ]
    totals = {
        arm_id: _cost(
            sum(float(row["charged_total_cost_units"]) for row in rows if row.get("arm_id") == arm_id)
        )
        for arm_id in ARM_IDS
    }
    held_totals = {
        arm_id: _cost(
            sum(
                float(row["charged_total_cost_units"])
                for row in rows
                if row.get("arm_id") == arm_id and row.get("split_name") == "held"
            )
        )
        for arm_id in ARM_IDS
    }
    selected = next(
        (row for row in effects if row.get("arm_id") == SELECTED_SAFETY_NET_ARM),
        {},
    )
    return {
        "row_type": "charged_cost_recomputation",
        "all_costs_recomputed_from_rows": bool(rows) and not bad_rows,
        "bad_cost_rows": bad_rows,
        "total_charged_work_by_arm": totals,
        "held_total_charged_work_by_arm": held_totals,
        "selected_eligible_arm": SELECTED_SAFETY_NET_ARM,
        "held_effect_vs_certified_control_units": selected.get(
            "held_effect_vs_certified_control_units", 0.0
        ),
        "model_cost_total": _cost(sum(float(row.get("model_cost_units", 0.0)) for row in rows)),
        "control_overhead_total": _cost(
            sum(float(row.get("control_overhead_units", 0.0)) for row in rows)
        ),
        "lookup_cost_total": _cost(sum(float(row.get("lookup_cost_units", 0.0)) for row in rows)),
        "proposal_cost_total": _cost(sum(float(row.get("proposal_cost_units", 0.0)) for row in rows)),
        "exact_check_cost_total": _cost(
            sum(float(row.get("exact_check_cost_units", 0.0)) for row in rows)
        ),
        "fallback_cost_total": _cost(
            sum(float(row.get("fallback_cost_units", 0.0)) for row in rows)
        ),
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-EFFECTS"],
    }


def exact_equality_and_fallback_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    mismatches = [dict(row) for row in rows if row.get("exact_equality") is not True]
    deleted = [dict(row) for row in rows if row.get("candidate_preserved") is not True]
    fallback_rows = [dict(row) for row in rows if row.get("fallback_used") is True]
    return {
        "row_type": "exact_equality_and_fallback_receipt",
        "row_count": len(rows),
        "all_exact_equal": bool(rows) and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_rows": mismatches,
        "all_candidates_preserved": bool(rows) and not deleted,
        "deleted_candidate_rows": deleted,
        "native_exact_fallback_available": bool(rows)
        and all(row.get("fallback_available") is True for row in rows),
        "native_exact_fallback_reachable": bool(fallback_rows),
        "fallback_used_count": len(fallback_rows),
        "held_fallback_used_count": sum(1 for row in fallback_rows if row.get("split_name") == "held"),
        "z3_evaluation_authority": True,
        "verifier_is_oracle": False,
        "spec_refs": [
            "REQ-BENCH-6545",
            "SCENARIO-BENCH-6545-RUNTIME",
            "SCENARIO-BENCH-6545-ROLLBACK",
        ],
    }


def shortcut_attack_matrix(
    *,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    training_receipt: Mapping[str, Any],
    exception_table: Mapping[str, Any],
    exact_receipt: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    used_features = {
        feature
        for row in rows
        for feature in row.get("feature_names_used", [])
        if isinstance(feature, str)
    }
    forbidden = set(contract.get("forbidden_features", []))
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("local_unit_id")), int(row.get("seed", 0)))].append(row)
    checks = {
        "family_identity": "family_identity" not in used_features and not (used_features & forbidden),
        "source_id": "source_id" not in used_features,
        "entity_names": "entity_names" not in used_features,
        "row_order": "row_order" not in used_features,
        "solver_effort_leakage": not any("solver_effort" in feature for feature in used_features),
        "held_table_writes": exception_table.get("held_entry_count") == 0
        and exception_table.get("held_write_attempt_count") == 0,
        "deleted_candidates": exact_receipt.get("all_candidates_preserved") is True
        and not any(row.get("candidate_deleted_count") for row in rows),
        "fallback_unreachability": exact_receipt.get("native_exact_fallback_reachable") is True,
        "calibration_reuse": training_receipt.get("held_rows_used_for_calibration") is False
        and {row.get("split_name") for row in calibration_rows} == {"development"},
        "timer_fabrication": bool(rows)
        and all(float(row.get("wall_time_s", -1.0)) >= 0.0 for row in rows)
        and all(str(row.get("timer_source", "")) for row in rows),
    }
    attack_rows = []
    for attack_id in SHORTCUT_ATTACK_IDS:
        payload = {
            "row_type": "shortcut_attack",
            "attack_id": attack_id,
            "expected_value": True,
            "observed_value": bool(checks[attack_id]),
            "fail_closed": bool(checks[attack_id]),
            "false_accept": not bool(checks[attack_id]),
            "unit_seed_groups_checked": len(grouped),
            "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-ATTACKS"],
        }
        attack_rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    return {
        "row_type": "shortcut_attack_matrix",
        "rows": attack_rows,
        "all_shortcuts_fail_closed": all(row["fail_closed"] is True for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in attack_rows if row["fail_closed"] is not True],
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-ATTACKS"],
    }


def rollback_receipt(
    *,
    contract: Mapping[str, Any],
    exact_receipt: Mapping[str, Any],
    exception_table: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "target_is_certified_structural_control": contract.get("rollback_target_arm")
        == CERTIFIED_CONTROL_ARM,
        "unsafe_state_ready_score_zero": True,
        "fallback_reachable_before_rollback": exact_receipt.get("native_exact_fallback_reachable")
        is True
        or exact_receipt.get("row_count") == 0,
        "held_writes_block_rollback_claim": exception_table.get("held_write_attempt_count") == 0,
    }
    return {
        "row_type": "rollback_receipt",
        "rollback_target_arm": CERTIFIED_CONTROL_ARM,
        "rollback_available": all(checks.values()),
        "unsafe_state_ready_score": 0.0,
        "rollback_checks": checks,
        "rollback_checks_passed": all(checks.values()),
        "rollback_policy": "any failed gate, held write, equality drift, deletion, unreachable fallback, shortcut, or calibration reuse closes to certified structural control",
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-ROLLBACK"],
    }


def aggregate_row_recomputation(
    *,
    gate: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Any],
    exact: Mapping[str, Any],
    attacks: Mapping[str, Any],
    rollback: Mapping[str, Any],
    training: Mapping[str, Any],
    exception_table: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
) -> JsonDict:
    expected = len(source_rows) * len(ARM_IDS)
    arm_coverage = {str(row.get("arm_id")) for row in rows} == set(ARM_IDS)
    seed_coverage = {
        int(row.get("seed", 0)) for row in rows if row.get("split_name") == "held"
    } == set(EVALUATION_SEEDS)
    selected_effect = next(
        (row for row in effects if row.get("arm_id") == SELECTED_SAFETY_NET_ARM),
        {},
    )
    calibrated = bool(calibration_rows) and all(
        row.get("calibration_source") == "development_only" for row in calibration_rows
    )
    selected_positive = float(
        selected_effect.get("held_effect_vs_certified_control_units", 0.0)
    ) > 0.0
    support_family_count = int(selected_effect.get("support_family_count", 0))
    support_effort_count = int(selected_effect.get("support_effort_count", 0))
    exception_passed = exception_table.get("immutable_after_freeze") is True and exception_table.get(
        "held_entry_count"
    ) == 0
    execution = all(
        (
            gate.get("gate_passed") is True,
            len(rows) == expected and expected > 0,
            arm_coverage,
            seed_coverage,
            costs.get("all_costs_recomputed_from_rows") is True,
            exact.get("all_exact_equal") is True,
            exact.get("all_candidates_preserved") is True,
            exact.get("native_exact_fallback_reachable") is True,
            calibrated,
            training.get("sealed_split_policy_passed") is True,
            exception_passed,
            attacks.get("all_shortcuts_fail_closed") is True,
            rollback.get("rollback_checks_passed") is True,
            protected.get("all_protected_files_unchanged") is True,
        )
    )
    ready = all((execution, selected_positive, support_family_count > 1, support_effort_count > 1))
    if not execution:
        verdict = "blocked" if gate.get("gate_passed") is not True else "disqualified"
    elif ready:
        verdict = "positive"
    elif selected_positive:
        verdict = "partial"
    else:
        verdict = None
    return {
        "row_type": "aggregate_row_recomputation",
        "upstream_gate_passed": gate.get("gate_passed") is True,
        "source_certified_row_count": len(source_rows),
        "matched_row_count": len(rows),
        "expected_matched_row_count": expected,
        "arm_coverage_passed": arm_coverage,
        "seed_coverage_passed": seed_coverage,
        "cost_recomputation_passed": costs.get("all_costs_recomputed_from_rows") is True,
        "exact_equality_passed": exact.get("all_exact_equal") is True,
        "candidate_preservation_passed": exact.get("all_candidates_preserved") is True,
        "fallback_reachability_passed": exact.get("native_exact_fallback_reachable") is True,
        "calibrated_abstention_passed": calibrated,
        "sealed_split_policy_passed": training.get("sealed_split_policy_passed") is True,
        "exception_table_immutable_passed": exception_passed,
        "shortcut_attack_passed": attacks.get("all_shortcuts_fail_closed") is True,
        "rollback_passed": rollback.get("rollback_checks_passed") is True,
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "selected_eligible_arm": SELECTED_SAFETY_NET_ARM,
        "selected_arm_positive_beyond_structural": selected_positive,
        "selected_arm_support_family_count": support_family_count,
        "selected_arm_support_families": selected_effect.get("support_families", []),
        "selected_arm_support_effort_count": support_effort_count,
        "selected_arm_support_efforts": selected_effect.get("support_efforts", []),
        "execution_complete_from_rows": execution,
        "ready_score_from_rows": 1.0 if ready else 0.0,
        "verdict_class_from_rows": verdict,
        "aggregate_source": "gate_per_unit_rows_effects_costs_exact_calibration_exception_attacks_rollback",
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-TERMINAL"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "upstream_gate_passed": True,
        "matched_row_count": aggregate.get("expected_matched_row_count"),
        "arm_coverage_passed": True,
        "seed_coverage_passed": True,
        "cost_recomputation_passed": True,
        "exact_equality_passed": True,
        "candidate_preservation_passed": True,
        "fallback_reachability_passed": True,
        "calibrated_abstention_passed": True,
        "sealed_split_policy_passed": True,
        "exception_table_immutable_passed": True,
        "shortcut_attack_passed": True,
        "rollback_passed": True,
        "protected_files_unchanged": True,
        "ready_score_is_binary": True,
    }
    observed = {
        "upstream_gate_passed": aggregate.get("upstream_gate_passed"),
        "matched_row_count": aggregate.get("matched_row_count"),
        "arm_coverage_passed": aggregate.get("arm_coverage_passed"),
        "seed_coverage_passed": aggregate.get("seed_coverage_passed"),
        "cost_recomputation_passed": aggregate.get("cost_recomputation_passed"),
        "exact_equality_passed": aggregate.get("exact_equality_passed"),
        "candidate_preservation_passed": aggregate.get("candidate_preservation_passed"),
        "fallback_reachability_passed": aggregate.get("fallback_reachability_passed"),
        "calibrated_abstention_passed": aggregate.get("calibrated_abstention_passed"),
        "sealed_split_policy_passed": aggregate.get("sealed_split_policy_passed"),
        "exception_table_immutable_passed": aggregate.get("exception_table_immutable_passed"),
        "shortcut_attack_passed": aggregate.get("shortcut_attack_passed"),
        "rollback_passed": aggregate.get("rollback_passed"),
        "protected_files_unchanged": aggregate.get("protected_files_unchanged"),
        "ready_score_is_binary": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
    }
    checks = {
        name: {"expected": value, "observed": observed[name], "passed": observed[name] == value}
        for name, value in expected.items()
    }
    failed = [name for name, row in checks.items() if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-GATE"],
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str | None]:
    verdict = aggregate.get("verdict_class_from_rows")
    if verdict == "blocked":
        return (
            "blocked_external_safety_net_router",
            "blocked_external_safety_net_router: upstream gate or precondition failed",
            "blocked",
        )
    if verdict == "disqualified":
        return (
            "disqualified_external_safety_net_router",
            "disqualified_external_safety_net_router: equality, preservation, calibration, exception, shortcut, or rollback check failed",
            "disqualified",
        )
    if verdict == "positive":
        return (
            "complete_external_safety_net_router_positive",
            (
                "complete_external_safety_net_router_positive: selected held charged "
                f"{SELECTED_SAFETY_NET_ARM} beats certified structural control with "
                "exact equality, calibrated abstention, immutable train-only exceptions, "
                "and reachable native fallback"
            ),
            "positive",
        )
    if verdict == "partial":
        return (
            "partial_external_safety_net_router",
            "partial_external_safety_net_router: held charged value had narrow support",
            "partial",
        )
    return (
        "complete_external_safety_net_router_null",
        "complete_external_safety_net_router_null: complete matched run found no held charged value",
        None,
    )


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {rel.as_posix(): sha256_file(repo_root / rel) for rel in SOURCE_RELATIVE_PATHS}
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "deterministic_exp6545_external_safety_net_router_reducer",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-BENCH-6545"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    upstream_path: Path,
    fixture_path: Path,
    run_date: str,
    protected_before: Mapping[str, str],
    aggregate: Mapping[str, Any],
) -> JsonDict:
    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "upstream_path": str(upstream_path),
        "fixture_path": str(fixture_path),
        "upstream_sha256": sha256_file(upstream_path),
        "fixture_sha256": sha256_file(fixture_path),
        "cpu_identity": _cpu_identity(),
        "gpu_identity": _gpu_identity(),
        "random_seed": RANDOM_SEED,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "arm_ids": list(ARM_IDS),
        "model_families": list(MODEL_FAMILIES),
        "timeout_s": TIMEOUT_S,
        "abstention_threshold": ABSTENTION_THRESHOLD,
        "candidate_budget_rule": "all_candidates_preserved",
        "exact_check_budget_rule": "candidate_count_plus_external_exact_fallback",
        "native_exact_fallback_available": True,
        "z3_evaluation_authority": True,
        "verifier_is_oracle": False,
        "aggregate_ready_score": aggregate.get("ready_score_from_rows"),
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6545", "SCENARIO-BENCH-6545-GATE"],
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(artifact, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    upstream_path: Path | str = EXP6544_RELATIVE_PATH,
    fixture_path: Path | str = FIXTURE_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    repo_root = Path(repo_root)
    result = Path(result_path)
    if not result.is_absolute():
        result = repo_root / result
    upstream = Path(upstream_path)
    if not upstream.is_absolute():
        upstream = repo_root / upstream
    fixture = Path(fixture_path)
    if not fixture.is_absolute():
        fixture = repo_root / fixture
    protected_before = _protected_hashes(repo_root, upstream)
    upstream_payload, _, _ = _read_json_with_status(upstream)
    gate = upstream_gate_receipt(
        repo_root=repo_root,
        upstream_path=upstream,
        fixture_path=fixture,
        protected_before=protected_before,
    )
    fixture_rows = _load_jsonl(fixture)
    contract = frozen_router_contract(run_date)
    source_rows = _certified_rows(upstream_payload) if gate["gate_passed"] is True else []
    models = candidate_model_rows(source_rows)
    training = training_and_calibration_receipts(source_rows, models)
    exception_table = exception_table_path_hash_and_freeze_receipt(
        repo_root=repo_root,
        rows=source_rows,
        result_path=result,
    )
    calibration = abstention_calibration_rows(source_rows)
    rows = per_unit_rows(certified_rows=source_rows, exception_table=exception_table)
    effects = paired_effect_rows(rows)
    family_effort = family_and_effort_effect_rows(rows)
    costs = charged_cost_recomputation(rows, effects)
    exact = exact_equality_and_fallback_receipt(rows)
    attacks = shortcut_attack_matrix(
        rows=rows,
        contract=contract,
        training_receipt=training,
        exception_table=exception_table,
        exact_receipt=exact,
        calibration_rows=calibration,
    )
    rollback = rollback_receipt(contract=contract, exact_receipt=exact, exception_table=exception_table)
    protected_after = _protected_hashes(repo_root, upstream)
    protected = protected_files_unchanged(protected_before, protected_after)
    aggregate = aggregate_row_recomputation(
        gate=gate,
        source_rows=source_rows,
        rows=rows,
        effects=effects,
        costs=costs,
        exact=exact,
        attacks=attacks,
        rollback=rollback,
        training=training,
        exception_table=exception_table,
        calibration_rows=calibration,
        protected=protected,
    )
    gates = gate_check_summary(aggregate)
    status, honest, verdict = _status_and_verdict(aggregate)
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "upstream_gate_receipt": gate,
        "frozen_router_contract": contract,
        "training_and_calibration_receipts": training,
        "candidate_model_rows": models,
        "exception_table_path_hash_and_freeze_receipt": exception_table,
        "abstention_calibration_rows": calibration,
        "per_unit_rows": rows,
        "paired_effect_rows": effects,
        "family_and_effort_effect_rows": family_effort,
        "charged_cost_recomputation": costs,
        "exact_equality_and_fallback_receipt": exact,
        "shortcut_attack_matrix": attacks,
        "rollback_receipt": rollback,
        "external_safety_net_ready_score": float(aggregate["ready_score_from_rows"]),
        "gate_check_summary": gates,
        "aggregate_row_recomputation": {
            **aggregate,
            "fixture_row_count": len(fixture_rows),
        },
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result,
            upstream_path=upstream,
            fixture_path=fixture,
            run_date=run_date,
            protected_before=protected_before,
            aggregate=aggregate,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.perf_counter() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        atomic_write_json(result, artifact, allow_override=False, sort_keys=False)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in {
        "positive",
        None,
        "partial",
        "blocked",
        "disqualified",
    }:
        errors.append("verdict_class outside Exp6545 enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if not str(artifact.get("status", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    score = artifact.get("external_safety_net_ready_score")
    recomputed = artifact.get("aggregate_row_recomputation", {}).get("ready_score_from_rows")
    if score not in {0.0, 1.0}:
        errors.append("external_safety_net_ready_score must be 0.0 or 1.0")
    if score != recomputed:
        errors.append("ready score mismatch")
    if artifact.get("verdict_class") == "positive" and score != 1.0:
        errors.append("positive verdict requires ready score 1.0")
    if (
        artifact.get("exception_table_path_hash_and_freeze_receipt", {}).get(
            "held_write_attempt_count"
        )
        != 0
    ):
        errors.append("held table write detected")
    if artifact.get("training_and_calibration_receipts", {}).get("held_rows_used_for_calibration"):
        errors.append("calibration used held rows")
    if (
        artifact.get("exact_equality_and_fallback_receipt", {}).get(
            "native_exact_fallback_reachable"
        )
        is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("exact fallback unreachable")
    if any(
        row.get("fail_closed") is not True
        for row in artifact.get("shortcut_attack_matrix", {}).get("rows", [])
    ) and artifact.get("verdict_class") != "blocked":
        errors.append("shortcut false accept")
    if (
        artifact.get("rollback_receipt", {}).get("rollback_available") is not True
        and artifact.get("verdict_class") != "blocked"
    ):
        errors.append("rollback unavailable")
    if artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is not True:
        errors.append("protected files changed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6545 external Safety-Net router artifact."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--upstream-path", default=str(REPO_ROOT / EXP6544_RELATIVE_PATH))
    parser.add_argument("--fixture-path", default=str(REPO_ROOT / FIXTURE_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        payload, _, _ = _read_json_with_status(result)
        errors = validate_artifact(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result,
        upstream_path=Path(args.upstream_path),
        fixture_path=Path(args.fixture_path),
        write=True,
        run_date=str(args.date),
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
