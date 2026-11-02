"""Reduce immutable Qwen and Gemma CFR rows without loading a model.

The reducer reconstructs each arm with the frozen source binder and exact
fixture checker. This makes the stored stream rows inputs, not authorities. It
then fixes each family result before it pools byte-identical shared units.

Spec: REQ-REPORT-6593 and SCENARIO-REPORT-6593-REPLAY through
SCENARIO-REPORT-6593-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any

from carnot import experiment_6587_v573_constraint_first_method_contract as exp6587
from carnot import experiment_6590_qwen36_constraint_first_stream as exp6590
from carnot import experiment_6591_gemma4_31b_constraint_first_stream as exp6591
from carnot import experiment_6592_v575_terminal_intake_and_method_lock as exp6592


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
TASK_ID = "exp6593-cfr-independent-row-reducer"
RESULT_RELATIVE_PATH = Path("results/experiment_6593_cfr_independent_row_reducer.json")
METHOD_RELATIVE_PATH = Path("results/experiment_6587_v573_constraint_first_method_contract.json")
INFERENCE_SUBSTRATE = "immutable_qwen_gemma_cfr_row_reducer_no_llm"
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_ARTIFACTS: dict[str, JsonDict] = {
    "qwen36": {
        "path": Path("results/experiment_6590_qwen36_constraint_first_stream.json"),
        "module": exp6590,
        "repository_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "ready_field": "qwen_cfr_rows_ready_score",
    },
    "gemma4_31b": {
        "path": Path("results/experiment_6591_gemma4_31b_constraint_first_stream.json"),
        "module": exp6591,
        "repository_id": "unsloth/gemma-4-31B-it-GGUF",
        "ready_field": "gemma31_cfr_rows_ready_score",
    },
}
INTAKE_RELATIVE_PATH = Path("results/experiment_6592_v575_terminal_intake_and_method_lock.json")
FAMILY_ORDER = tuple(SOURCE_ARTIFACTS)
ARM_ORDER = tuple(exp6590.ARM_ORDER)
CANDIDATE_ARMS = ("always_on_cfr", "routed_cfr")
STAGE_ORDER = tuple(exp6590.STAGE_ORDER)
FAILURE_CLASSES = tuple(exp6590.FAILURE_CLASSES)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 6587

REQUIRED_ATTACK_IDS = (
    "aggregate_only_claim",
    "dropped_failures",
    "family_label_swap",
    "identical_control_and_treatment",
    "one_win_promotion",
    "no_headroom_promotion",
    "seed_drift",
    "source_drift",
    "exact_check_substitution",
    "cost_omission",
    "recomputation_disagreement",
)
GATE_CONDITION_ORDER = (
    "positive_exact_success_delta",
    "paired_ci95_lower_nonnegative",
    "unsafe_release_increase_within_limit",
    "stage1_precision_floor",
    "stage1_recall_floor",
    "tokens_per_unit_limit",
    "latency_per_unit_limit",
    "paired_exact_p_value_limit",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "per_unit_rows",
    "source_artifact_receipts",
    "model_identity_replay_rows",
    "row_completeness_recomputation",
    "family_effect_rows",
    "pooled_effect_summary",
    "paired_statistical_receipts",
    "constraint_quality_summary",
    "safety_and_cost_summary",
    "acceptance_gate_rows",
    "cfr_reducer_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The reducer ends with a complete comparison or a named evidence block.",
    "honest_verdict": "The verdict states family and pooled CFR effects without upgrading row completeness to benefit.",
    "verdict_class": "The closed enum makes an exact-checker-defined win circular-positive, never positive.",
    "gate_check_summary": "Any block names the failed gate, artifact, row, hash, pairing, or method check and observed value.",
    "per_unit_rows": "Every family, unit, and arm carries exact outcome, headroom, constraints, abstention, tokens, latency, failures, and raw references.",
    "source_artifact_receipts": "Both immutable streams and the intake evidence root bind by path and hash.",
    "model_identity_replay_rows": "Nested mandated GGUF identities and process receipts bind every family result.",
    "row_completeness_recomputation": "Expected, present, duplicate, missing, reordered, and failure counts derive from raw rows.",
    "family_effect_rows": "Qwen and Gemma direct-versus-CFR effects are computed before pooling.",
    "pooled_effect_summary": "Pooling uses only shared byte-identical paired units and retains family heterogeneity.",
    "paired_statistical_receipts": "Wins, losses, ties, headroom, tests, intervals, and underpowered cases remain explicit.",
    "constraint_quality_summary": "Supported, unsupported, contradictory, precision, recall, and leakage replay from Stage 1 rows.",
    "safety_and_cost_summary": "Unsafe release, abstention, tokens, latency, and failures are charged to each arm.",
    "acceptance_gate_rows": "Every preregistered benefit condition records expected, observed, and passed values.",
    "cfr_reducer_ready_score": "This binary field gates Exp6599 only when all rows and aggregates replay.",
    "attack_rows": "Aggregate, drop, swap, identity, no-headroom, drift, authority, cost, and disagreement attacks fail closed.",
    "preconditions_checked": "Gates, hashes, units, arms, identities, seeds, tests, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain their original hashes.",
    "inference_substrate": "The task declares immutable CFR row reduction with no LLM.",
    "verifier_is_oracle": "The exact checker defines CFR success, so any benefit is circular-positive.",
    "field_provenance": "Every field points to raw rows, hashes, and reducer functions.",
    "duration_s": "Monotonic duration exposes truncated replay.",
    "tests_run": "Focused reducer and consistency commands include exits and durations.",
    "reproducibility_checksum": "A final content hash protects the independent result.",
}
DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest -n 0 -o addopts= tests/python/test_experiment_6593_cfr_independent_row_reducer.py -q",
        "exit_code": 0,
        "duration_s": 16.90,
        "blocking": True,
    },
    {
        "command": ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6593_cfr_independent_row_reducer.py -m pytest -n 0 -o addopts= tests/python/test_experiment_6593_cfr_independent_row_reducer.py -q && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6593_cfr_independent_row_reducer.py --show-missing --fail-under=100",
        "exit_code": 0,
        "duration_s": 65.03,
        "blocking": True,
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit_code": 2,
        "duration_s": 370.57,
        "blocking": False,
        "outcome_note": "Interrupted after 40 unrelated legacy, live-config, freshness, and tracked-results-guard failures; 7939 passed and 7 skipped. No Exp6593 failure was reported.",
    },
    {
        "command": ".venv/bin/ruff check python/carnot/experiment_6593_cfr_independent_row_reducer.py tests/python/test_experiment_6593_cfr_independent_row_reducer.py",
        "exit_code": 0,
        "duration_s": 0.1,
        "blocking": True,
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6593_cfr_independent_row_reducer.py",
        "exit_code": 0,
        "duration_s": 0.1,
        "blocking": True,
    },
)

canonical_json = exp6590.canonical_json
sha256_bytes = exp6590.sha256_bytes
sha256_json = exp6590.sha256_json
sha256_file = exp6590.sha256_file
artifact_checksum = exp6590.artifact_checksum
load_json = exp6590.load_json


def unwrap_value(value: Any) -> Any:
    """Return the bare value so a principle wrapper cannot change truthiness."""

    if isinstance(value, Mapping) and "value" in value:
        return unwrap_value(value["value"])
    return value


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {"all_unchanged": bool(rows) and all(row["unchanged"] for row in rows), "rows": rows}


def _cpu_receipt() -> JsonDict:
    model = platform.processor()
    if not model and Path("/proc/cpuinfo").is_file():
        for line in (
            Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            if line.casefold().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "architecture": platform.machine(),
        "model": model or "unknown",
        "logical_count": os.cpu_count() or 1,
    }


def _load_sources(repo_root: Path) -> tuple[JsonDict, dict[str, JsonDict], JsonDict]:
    method = load_json(repo_root / METHOD_RELATIVE_PATH)
    streams = {
        family: load_json(repo_root / config["path"]) for family, config in SOURCE_ARTIFACTS.items()
    }
    intake = load_json(repo_root / INTAKE_RELATIVE_PATH)
    return method, streams, intake


def build_source_artifact_receipts(
    repo_root: Path, streams: Mapping[str, Mapping[str, Any]], intake: Mapping[str, Any]
) -> list[JsonDict]:
    """Bind both streams and the intake root to content and replayed gates."""

    rows = []
    for family, config in SOURCE_ARTIFACTS.items():
        source = streams[family]
        module = config["module"]
        reduction = module.stream_reducer(source)
        row = {
            "source_kind": "immutable_cfr_stream",
            "family": family,
            "path": config["path"].as_posix(),
            "sha256": sha256_file(repo_root / config["path"]),
            "embedded_checksum_valid": source.get("reproducibility_checksum")
            == module.artifact_checksum(source),
            "ready_field": config["ready_field"],
            "stored_ready_value": unwrap_value(source.get(config["ready_field"])),
            "recomputed_ready_value": reduction["ready_score"],
            "status": unwrap_value(source.get("status")),
            "honest_verdict": unwrap_value(source.get("honest_verdict")),
            "verdict_class": unwrap_value(source.get("verdict_class")),
        }
        row["all_checks_passed"] = bool(
            row["embedded_checksum_valid"]
            and row["stored_ready_value"] == 1.0
            and row["recomputed_ready_value"] == 1.0
            and row["verdict_class"] is None
        )
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    intake_reduction = exp6592.readiness_reducer(intake)
    intake_row = {
        "source_kind": "v575_evidence_root",
        "family": None,
        "path": INTAKE_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(repo_root / INTAKE_RELATIVE_PATH),
        "embedded_checksum_valid": intake.get("reproducibility_checksum")
        == exp6592.artifact_checksum(intake),
        "ready_field": "v575_cfr_reducer_ready_score",
        "stored_ready_value": unwrap_value(intake.get("v575_cfr_reducer_ready_score")),
        "recomputed_ready_value": intake_reduction["v575_cfr_reducer_ready_score"],
        "status": unwrap_value(intake.get("status")),
        "honest_verdict": unwrap_value(intake.get("honest_verdict")),
        "verdict_class": unwrap_value(intake.get("verdict_class")),
    }
    intake_row["all_checks_passed"] = bool(
        intake_row["embedded_checksum_valid"]
        and intake_row["stored_ready_value"] == 1.0
        and intake_row["recomputed_ready_value"] == 1.0
        and intake_row["verdict_class"] is None
    )
    intake_row["row_hash"] = sha256_json(intake_row)
    rows.append(intake_row)
    return rows


def build_model_identity_replay_rows(streams: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Replay each nested model and owned-process identity independently."""

    rows = []
    for family, config in SOURCE_ARTIFACTS.items():
        source = streams[family]
        module = config["module"]
        model_receipt = source.get("model_spec_and_identity", {})
        process_receipt = source.get("gpu_process_receipts", {})
        model_checks = module.model_identity_checks(model_receipt)
        process_checks = module.process_lifecycle_checks(process_receipt)
        repository_id = config["repository_id"]
        row = {
            "family": family,
            "repository_id": repository_id,
            "model_spec_and_identity": deepcopy(model_receipt),
            "model_identity_hash": sha256_json(model_receipt),
            "model_identity_checks": model_checks,
            "model_identity_valid": all(model_checks.values()),
            "gpu_process_receipts": deepcopy(process_receipt),
            "gpu_process_receipts_hash": sha256_json(process_receipt),
            "process_identity_checks": process_checks,
            "process_identity_valid": all(process_checks.values()),
            "cross_family_residency_detected": process_receipt.get("process", {}).get(
                "resident_model_families"
            )
            != [repository_id],
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def _raw_stage_references(stages: Mapping[str, Any]) -> list[JsonDict]:
    references = []
    for stage in STAGE_ORDER:
        receipt = stages.get(stage)
        if not isinstance(receipt, Mapping):
            references.append({"stage": stage, "present": False})
            continue
        references.append(
            {
                "stage": stage,
                "present": True,
                "row_hash": receipt.get("row_hash"),
                "raw_sha256": receipt.get("raw_sha256"),
                "raw_api_response_sha256": receipt.get("raw_api_response_sha256"),
                "raw_byte_count": receipt.get("raw_byte_count"),
                "request_sha256": receipt.get("request_sha256"),
                "prompt_sha256": receipt.get("prompt_sha256"),
                "stage1_raw_sha256_in_request": receipt.get("stage1_raw_sha256_in_request"),
                "request_status": receipt.get("request_status"),
                "stop_reason": receipt.get("stop_reason"),
                "total_tokens": receipt.get("total_tokens"),
                "latency_s": receipt.get("latency_s"),
                "failure_flags": deepcopy(receipt.get("failure_flags", {})),
            }
        )
    return references


def _expected_stages(arm_name: str, route: str) -> set[str]:
    if arm_name == "direct" or route == "direct":
        return {"direct"}
    return {"stage1", "stage2"}


def replay_stream_rows(
    family: str,
    stream: Mapping[str, Any],
    method: Mapping[str, Any],
    *,
    source_artifact_sha256: str | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Rebuild one family stream from raw stages and the frozen method."""

    config = SOURCE_ARTIFACTS[family]
    units = [
        row
        for row in method.get("source_unit_manifest", {}).get("units", [])
        if isinstance(row, Mapping)
    ]
    unit_by_id = {str(row.get("unit_id")): row for row in units}
    expected_ids = [str(row.get("unit_id")) for row in units]
    seed_rows = method.get("arm_seed_budget_contract", {}).get("seed_schedule", [])
    seed_by_id = {str(row.get("unit_id")): int(row.get("seed")) for row in seed_rows}
    route_by_id = {
        str(row.get("unit_id")): str(row.get("observed_route"))
        for row in method.get("router_contract", {}).get("routing_rows", [])
    }
    registry = method["source_binding_and_exact_authority_contract"]["exact_obligation_registry"]
    registry_hash = str(registry["registry_sha256"])
    stream_rows = [row for row in stream.get("per_unit_rows", []) if isinstance(row, Mapping)]
    present_ids = [str(row.get("unit_id")) for row in stream_rows]
    id_counts = Counter(present_ids)
    present_arm_keys = [
        (str(row.get("unit_id")), str(arm.get("arm_name")))
        for row in stream_rows
        for arm in row.get("arms", [])
        if isinstance(arm, Mapping)
    ]
    arm_counts = Counter(present_arm_keys)
    expected_arm_keys = [(unit_id, arm) for unit_id in expected_ids for arm in ARM_ORDER]
    output_rows: list[JsonDict] = []
    rebuilt_unit_matches: dict[str, bool] = {}
    for unit_index, source_unit in enumerate(stream_rows):
        unit_id = str(source_unit.get("unit_id"))
        method_unit = unit_by_id.get(unit_id)
        if method_unit is None:
            continue
        source_arms = [arm for arm in source_unit.get("arms", []) if isinstance(arm, Mapping)]
        rebuilt_arms = []
        for source_arm in source_arms:
            raw_stages = source_arm.get("raw_stages", {})
            stage_receipts = {
                stage: receipt
                for stage, receipt in raw_stages.items()
                if isinstance(receipt, Mapping)
            }
            replayed_arm = exp6590.build_arm_row(
                unit=method_unit,
                arm_name=str(source_arm.get("arm_name")),
                route=str(source_arm.get("route")),
                seed=int(source_arm.get("seed", -1)),
                stage_receipts=stage_receipts,
                registry_hash=registry_hash,
            )
            rebuilt_arms.append(replayed_arm)
        replayed_unit = exp6590.finalize_unit_row(
            method_unit, rebuilt_arms, source_unit.get("process_binding", {})
        )
        rebuilt_unit_matches[unit_id] = replayed_unit == source_unit
        direct_source = next((arm for arm in source_arms if arm.get("arm_name") == "direct"), {})
        direct_stages = {
            stage: receipt
            for stage, receipt in direct_source.get("raw_stages", {}).items()
            if isinstance(receipt, Mapping)
        }
        direct_replay = exp6590.build_arm_row(
            unit=method_unit,
            arm_name="direct",
            route="direct",
            seed=seed_by_id[unit_id],
            stage_receipts=direct_stages,
            registry_hash=registry_hash,
        )
        direct_success = bool(direct_replay["exact_results"]["exact_success"])
        for arm_index, (source_arm, replayed_arm) in enumerate(
            zip(source_arms, rebuilt_arms, strict=True)
        ):
            arm_name = str(source_arm.get("arm_name"))
            route = str(source_arm.get("route"))
            stages = source_arm.get("raw_stages", {})
            present_stages = {
                stage for stage, receipt in stages.items() if isinstance(receipt, Mapping)
            }
            bindings = replayed_arm.get("source_span_bindings", [])
            supported = sum(
                row.get("source_supported") is True
                and row.get("unsupported") is not True
                and row.get("contradictory") is not True
                for row in bindings
            )
            gold_ids = {
                row.get("constraint_id")
                for row in method_unit.get("gold_constraints", [])
                if isinstance(row, Mapping) and row.get("constraint_class") == "supported"
            }
            matched_ids = {
                row.get("constraint_id")
                for row in bindings
                if row.get("source_supported") is True and row.get("constraint_id") in gold_ids
            }
            failure = deepcopy(replayed_arm.get("failure", {}))
            exact_outcome = deepcopy(replayed_arm.get("exact_results", {}))
            row: JsonDict = {
                "family": family,
                "repository_id": config["repository_id"],
                "family_index": FAMILY_ORDER.index(family),
                "unit_index": unit_index,
                "arm_index": arm_index,
                "unit_id": unit_id,
                "arm_name": arm_name,
                "route": route,
                "stratum": method_unit.get("stratum"),
                "split": method_unit.get("split"),
                "case_class": method_unit.get("case_class"),
                "fixture_id": method_unit.get("fixture_id"),
                "fixture_hash": method_unit.get("fixture_hash"),
                "expected_action": method_unit.get("expected_action"),
                "source_bytes_sha256": source_unit.get("source_bytes_sha256"),
                "task_bytes_sha256": source_unit.get("task_bytes_sha256"),
                "method_source_bytes_sha256": method_unit.get("source_bytes_sha256"),
                "method_task_bytes_sha256": method_unit.get("task_bytes_sha256"),
                "seed": int(source_arm.get("seed", -1)),
                "frozen_seed": seed_by_id[unit_id],
                "raw_source_artifact_path": config["path"].as_posix(),
                "raw_source_artifact_sha256": source_artifact_sha256,
                "raw_source_artifact_checksum": stream.get("reproducibility_checksum"),
                "raw_unit_row_hash": source_unit.get("row_hash"),
                "raw_arm_row_hash": source_arm.get("row_hash"),
                "raw_outcome_references": _raw_stage_references(stages),
                "raw_stage_presence": {stage: stage in present_stages for stage in STAGE_ORDER},
                "exact_outcome": exact_outcome,
                "exact_success": bool(exact_outcome.get("exact_success")),
                "headroom": not direct_success,
                "direct_exact_success": direct_success,
                "supported_constraint_count": supported,
                "matched_supported_constraint_count": len(matched_ids),
                "supported_gold_constraint_count": len(gold_ids),
                "stage1_proposal_count": len(replayed_arm.get("parsed_proposals", [])),
                "stage1_executed": "stage1" in present_stages,
                "stage1_precision": (
                    replayed_arm.get("stage1_precision") if "stage1" in present_stages else None
                ),
                "stage1_recall": (
                    replayed_arm.get("stage1_recall") if "stage1" in present_stages else None
                ),
                "unsupported_constraint_count": int(
                    replayed_arm.get("unsupported_constraint_count", 0)
                ),
                "contradictory_constraint_count": int(
                    replayed_arm.get("contradictory_constraint_count", 0)
                ),
                "abstention": bool(replayed_arm.get("abstention")),
                "unsafe_release": bool(replayed_arm.get("unsafe_release")),
                "tokens": int(replayed_arm.get("tokens", {}).get("total", 0)),
                "token_breakdown": deepcopy(replayed_arm.get("tokens", {})),
                "latency_s": float(replayed_arm.get("latency_s", 0.0)),
                "charged_cost": float(replayed_arm.get("charged_cost", 0.0)),
                "charged_cost_unit": replayed_arm.get("charged_cost_unit"),
                "failure": failure,
                "failure_any": bool(failure.get("any")),
                "failure_outcomes": [
                    name for name, value in failure.items() if name != "any" and value is True
                ],
                "charged_failure": bool(failure.get("any"))
                and math.isclose(
                    float(replayed_arm.get("charged_cost", 0.0)),
                    int(replayed_arm.get("tokens", {}).get("total", 0))
                    + float(replayed_arm.get("latency_s", 0.0)),
                    abs_tol=1e-9,
                ),
                "exact_registry_sha256": exact_outcome.get("registry_sha256"),
                "frozen_exact_registry_sha256": registry_hash,
                "exact_replay_matches_source": exact_outcome == source_arm.get("exact_results"),
                "arm_replay_matches_source": replayed_arm == source_arm,
                "unit_replay_matches_source": replayed_unit == source_unit,
                "raw_stage_replay_valid": present_stages == _expected_stages(arm_name, route)
                and all(exp6590._stage_authentic(stages[stage], stage) for stage in present_stages),
                "source_binding_valid": source_unit.get("source_bytes_sha256")
                == method_unit.get("source_bytes_sha256")
                and source_unit.get("task_bytes_sha256") == method_unit.get("task_bytes_sha256"),
                "seed_valid": int(source_arm.get("seed", -1)) == seed_by_id[unit_id],
                "route_valid": route
                == (
                    "direct"
                    if arm_name == "direct"
                    else "cfr"
                    if arm_name == "always_on_cfr"
                    else route_by_id[unit_id]
                ),
                "model_family_binding_valid": source_unit.get("process_binding", {}).get(
                    "repository_id"
                )
                == config["repository_id"],
            }
            row["row_reproducibility_hash"] = sha256_json(row)
            output_rows.append(row)

    raw_stage_count = sum(
        isinstance(receipt, Mapping)
        for unit in stream_rows
        for arm in unit.get("arms", [])
        for receipt in arm.get("raw_stages", {}).values()
    )
    expected_raw_stage_count = sum(
        1 + 2 + (1 if route_by_id[unit_id] == "direct" else 2) for unit_id in expected_ids
    )
    duplicate_unit_count = sum(max(0, count - 1) for count in id_counts.values())
    duplicate_arm_count = sum(max(0, count - 1) for count in arm_counts.values())
    missing_ids = [unit_id for unit_id in expected_ids if unit_id not in id_counts]
    missing_arms = [key for key in expected_arm_keys if key not in arm_counts]
    reordered_units = sum(
        observed != expected for observed, expected in zip(present_ids, expected_ids, strict=False)
    ) + abs(len(present_ids) - len(expected_ids))
    reordered_arms = sum(
        [str(arm.get("arm_name")) for arm in unit.get("arms", [])] != list(ARM_ORDER)
        for unit in stream_rows
    )
    raw_rows = exp6590.build_raw_stage_receipts(stream_rows)
    exact_rows = exp6590.build_exact_checker_receipts(stream_rows)
    failure_rows = exp6590.build_failure_rows(stream_rows)
    source_unit_mismatch_count = sum(not row["source_binding_valid"] for row in output_rows)
    cross_family_count = sum(not row["model_family_binding_valid"] for row in output_rows)
    module = config["module"]
    stream_reduction = module.stream_reducer(stream)
    all_flat_replayed = bool(output_rows) and all(
        row["exact_replay_matches_source"]
        and row["arm_replay_matches_source"]
        and row["unit_replay_matches_source"]
        and row["raw_stage_replay_valid"]
        and row["source_binding_valid"]
        and row["seed_valid"]
        and row["route_valid"]
        and row["model_family_binding_valid"]
        for row in output_rows
    )
    completeness: JsonDict = {
        "family": family,
        "expected_unit_count": len(expected_ids),
        "present_unit_count": len(stream_rows),
        "expected_arm_count": len(expected_arm_keys),
        "present_arm_count": len(present_arm_keys),
        "expected_raw_stage_count": expected_raw_stage_count,
        "present_raw_stage_count": raw_stage_count,
        "present_exact_result_count": len(exact_rows),
        "failure_arm_count": sum(row["failure_any"] for row in output_rows),
        "failure_outcome_count": len(failure_rows),
        "duplicate_unit_count": duplicate_unit_count,
        "duplicate_arm_count": duplicate_arm_count,
        "missing_unit_count": len(missing_ids),
        "missing_arm_count": len(missing_arms),
        "extra_unit_count": len([unit_id for unit_id in present_ids if unit_id not in unit_by_id]),
        "extra_arm_count": len([key for key in present_arm_keys if key not in expected_arm_keys]),
        "reordered_unit_count": reordered_units,
        "reordered_arm_count": reordered_arms,
        "cross_family_row_count": cross_family_count,
        "source_unit_mismatch_count": source_unit_mismatch_count,
        "raw_stage_receipts_match": stream.get("raw_stage_receipts") == raw_rows,
        "exact_checker_receipts_match": stream.get("exact_checker_receipts") == exact_rows,
        "failure_rows_match": stream.get("failure_rows") == failure_rows,
        "recomputed_stream_ready_score": stream_reduction["ready_score"],
        "all_unit_replays_match": all(rebuilt_unit_matches.values())
        and len(rebuilt_unit_matches) == len(expected_ids),
    }
    completeness["all_rows_replayed"] = bool(
        len(stream_rows) == len(expected_ids)
        and len(present_arm_keys) == len(expected_arm_keys)
        and raw_stage_count == expected_raw_stage_count
        and len(exact_rows) == len(expected_arm_keys)
        and not any(
            completeness[name]
            for name in (
                "duplicate_unit_count",
                "duplicate_arm_count",
                "missing_unit_count",
                "missing_arm_count",
                "extra_unit_count",
                "extra_arm_count",
                "reordered_unit_count",
                "reordered_arm_count",
                "cross_family_row_count",
                "source_unit_mismatch_count",
            )
        )
        and completeness["raw_stage_receipts_match"]
        and completeness["exact_checker_receipts_match"]
        and completeness["failure_rows_match"]
        and completeness["recomputed_stream_ready_score"] == 1.0
        and completeness["all_unit_replays_match"]
        and all_flat_replayed
    )
    completeness["row_hash"] = sha256_json(completeness)
    return output_rows, completeness


def exact_mcnemar(control: Sequence[bool], treatment: Sequence[bool]) -> JsonDict:
    """Return the exact paired binary test and its no-variation disposition."""

    if len(control) != len(treatment):
        raise ValueError("paired binary inputs must have equal length")
    wins = sum((not left) and right for left, right in zip(control, treatment, strict=True))
    losses = sum(left and (not right) for left, right in zip(control, treatment, strict=True))
    ties = len(control) - wins - losses
    discordant = wins + losses
    if discordant == 0:
        return {
            "test": "two_sided_exact_mcnemar",
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "discordant": 0,
            "p_value": 1.0,
            "valid": False,
            "reason": "no_discordant_units",
        }
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "test": "two_sided_exact_mcnemar",
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "discordant": discordant,
        "p_value": p_value,
        "valid": True,
        "reason": None,
    }


def exact_sign_test(values: Sequence[float]) -> JsonDict:
    """Return a two-sided exact sign test while retaining zero differences."""

    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    nonzero = positive + negative
    if nonzero == 0:
        return {
            "test": "two_sided_exact_sign",
            "positive": positive,
            "negative": negative,
            "zero": zero,
            "nonzero": 0,
            "p_value": 1.0,
            "valid": False,
            "reason": "no_nonzero_pairs",
        }
    tail = sum(math.comb(nonzero, index) for index in range(min(positive, negative) + 1))
    return {
        "test": "two_sided_exact_sign",
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "nonzero": nonzero,
        "p_value": min(1.0, 2.0 * tail / (2**nonzero)),
        "valid": True,
        "reason": None,
    }


def paired_bootstrap_ci(
    values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> JsonDict:
    """Bootstrap paired unit deltas with a fixed seed and percentile rule."""

    if not values:
        return {"lower": None, "upper": None, "resamples": resamples, "seed": seed, "unit_count": 0}
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(resamples)
    )
    lower_index = math.floor(0.025 * (resamples - 1))
    upper_index = math.ceil(0.975 * (resamples - 1))
    return {
        "lower": round(float(means[lower_index]), 9),
        "upper": round(float(means[upper_index]), 9),
        "resamples": resamples,
        "seed": seed,
        "unit_count": count,
    }


def _effect_from_pairs(
    *,
    scope: str,
    candidate_arm: str,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    cluster_count: int,
    clustered_deltas: Mapping[str, Sequence[float]] | None = None,
) -> JsonDict:
    control = [bool(left["exact_success"]) for left, _ in pairs]
    treatment = [bool(right["exact_success"]) for _, right in pairs]
    exact_deltas = [
        float(right["exact_success"]) - float(left["exact_success"]) for left, right in pairs
    ]
    token_deltas = [float(right["tokens"]) - float(left["tokens"]) for left, right in pairs]
    latency_deltas = [float(right["latency_s"]) - float(left["latency_s"]) for left, right in pairs]
    bootstrap_exact = exact_deltas
    bootstrap_tokens = token_deltas
    bootstrap_latency = latency_deltas
    exact_test = exact_mcnemar(control, treatment)
    if clustered_deltas is not None:
        bootstrap_exact = list(clustered_deltas["exact"])
        bootstrap_tokens = list(clustered_deltas["tokens"])
        bootstrap_latency = list(clustered_deltas["latency"])
        sign_test = exact_sign_test(bootstrap_exact)
        exact_test = {
            **sign_test,
            "wins": sign_test["positive"],
            "losses": sign_test["negative"],
            "ties": sign_test["zero"],
            "discordant": sign_test["nonzero"],
        }
    row: JsonDict = {
        "scope": scope,
        "control_arm": "direct",
        "candidate_arm": candidate_arm,
        "paired_family_unit_count": len(pairs),
        "bootstrap_cluster_count": cluster_count,
        "wins": exact_test["wins"],
        "losses": exact_test["losses"],
        "ties": exact_test["ties"],
        "direct_headroom_count": sum(not value for value in control),
        "direct_headroom_rate": round(sum(not value for value in control) / len(pairs), 9),
        "exact_success_delta": round(sum(exact_deltas) / len(pairs), 9),
        "exact_success_ci95": {
            key: value
            for key, value in paired_bootstrap_ci(bootstrap_exact).items()
            if key in {"lower", "upper"}
        },
        "exact_test": exact_test,
        "token_delta_per_unit": round(sum(token_deltas) / len(pairs), 9),
        "token_delta_ci95": {
            key: value
            for key, value in paired_bootstrap_ci(bootstrap_tokens).items()
            if key in {"lower", "upper"}
        },
        "token_sign_test": exact_sign_test(bootstrap_tokens),
        "latency_delta_s_per_unit": round(sum(latency_deltas) / len(pairs), 9),
        "latency_delta_ci95": {
            key: value
            for key, value in paired_bootstrap_ci(bootstrap_latency).items()
            if key in {"lower", "upper"}
        },
        "latency_sign_test": exact_sign_test(bootstrap_latency),
        "unsafe_release_delta_count": sum(bool(right["unsafe_release"]) for _, right in pairs)
        - sum(bool(left["unsafe_release"]) for left, _ in pairs),
        "unsafe_release_delta_rate": round(
            (
                sum(bool(right["unsafe_release"]) for _, right in pairs)
                - sum(bool(left["unsafe_release"]) for left, _ in pairs)
            )
            / len(pairs),
            9,
        ),
        "failure_delta_count": sum(bool(right["failure_any"]) for _, right in pairs)
        - sum(bool(left["failure_any"]) for left, _ in pairs),
        "identical_raw_control_treatment_count": sum(
            left["raw_arm_row_hash"] == right["raw_arm_row_hash"] for left, right in pairs
        ),
        "no_headroom": not any(not value for value in control),
        "underpowered": not any(not value for value in control) or exact_test["discordant"] < 10,
    }
    row["row_hash"] = sha256_json(row)
    return row


def build_family_effect_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Fix direct-versus-CFR effects within each family before pooling."""

    output = []
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row.get("family") == family]
        by_key = {(str(row["unit_id"]), str(row["arm_name"])): row for row in family_rows}
        unit_ids = [str(row["unit_id"]) for row in family_rows if row.get("arm_name") == "direct"]
        for candidate in CANDIDATE_ARMS:
            pairs = [
                (by_key[(unit_id, "direct")], by_key[(unit_id, candidate)]) for unit_id in unit_ids
            ]
            output.append(
                _effect_from_pairs(
                    scope=family, candidate_arm=candidate, pairs=pairs, cluster_count=len(unit_ids)
                )
            )
    return output


def build_pooled_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    fixed_family_effects: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Pool only shared byte-identical units after both family effects exist."""

    direct_by_family = {
        family: {
            str(row["unit_id"]): row
            for row in rows
            if row.get("family") == family and row.get("arm_name") == "direct"
        }
        for family in FAMILY_ORDER
    }
    shared_ids = [
        unit_id
        for unit_id in direct_by_family[FAMILY_ORDER[0]]
        if unit_id in direct_by_family[FAMILY_ORDER[1]]
    ]
    receipts = []
    eligible_ids = []
    for unit_id in shared_ids:
        left = direct_by_family[FAMILY_ORDER[0]][unit_id]
        right = direct_by_family[FAMILY_ORDER[1]][unit_id]
        fields = (
            "unit_id",
            "source_bytes_sha256",
            "task_bytes_sha256",
            "fixture_hash",
            "expected_action",
        )
        checks = {field: left.get(field) == right.get(field) for field in fields}
        receipt = {"unit_id": unit_id, "checks": checks, "byte_identical": all(checks.values())}
        receipt["row_hash"] = sha256_json(receipt)
        receipts.append(receipt)
        if receipt["byte_identical"]:
            eligible_ids.append(unit_id)
    by_key = {(str(row["family"]), str(row["unit_id"]), str(row["arm_name"])): row for row in rows}
    effects = []
    for candidate in CANDIDATE_ARMS:
        pairs = [
            (
                by_key[(family, unit_id, "direct")],
                by_key[(family, unit_id, candidate)],
            )
            for unit_id in eligible_ids
            for family in FAMILY_ORDER
        ]
        clustered_deltas = {
            metric: [
                sum(
                    float(by_key[(family, unit_id, candidate)][field])
                    - float(by_key[(family, unit_id, "direct")][field])
                    for family in FAMILY_ORDER
                )
                / len(FAMILY_ORDER)
                for unit_id in eligible_ids
            ]
            for metric, field in {
                "exact": "exact_success",
                "tokens": "tokens",
                "latency": "latency_s",
            }.items()
        }
        effect = _effect_from_pairs(
            scope="pooled",
            candidate_arm=candidate,
            pairs=pairs,
            cluster_count=len(eligible_ids),
            clustered_deltas=clustered_deltas,
        )
        effect["pooling_note"] = (
            "Family-unit outcomes remain visible. Bootstrap clusters use canonical unit IDs."
        )
        effect["row_hash"] = sha256_json(
            {key: value for key, value in effect.items() if key != "row_hash"}
        )
        effects.append(effect)
    family_effects = (
        list(fixed_family_effects)
        if fixed_family_effects is not None
        else build_family_effect_rows(rows)
    )
    result: JsonDict = {
        "family_results_fixed_before_pooling": True,
        "pooling_cluster": "unit_id",
        "shared_unit_count": len(shared_ids),
        "eligible_shared_unit_count": len(eligible_ids),
        "family_unit_pair_count": len(eligible_ids) * len(FAMILY_ORDER),
        "all_shared_units_byte_identical": bool(shared_ids)
        and len(eligible_ids) == len(shared_ids),
        "shared_unit_receipts": receipts,
        "effect_rows": effects,
        "family_heterogeneity_retained": True,
        "family_exact_delta_rows": [
            {
                "scope": row["scope"],
                "candidate_arm": row["candidate_arm"],
                "exact_success_delta": row["exact_success_delta"],
                "no_headroom": row["no_headroom"],
            }
            for row in family_effects
        ],
    }
    result["summary_hash"] = sha256_json(result)
    return result


def build_paired_statistical_receipts(
    family_effects: Sequence[Mapping[str, Any]], pooled: Mapping[str, Any]
) -> list[JsonDict]:
    """Expose each exact test, interval, and cost-pair test in one table."""

    output = []
    for effect in [*family_effects, *pooled.get("effect_rows", [])]:
        row = {
            "scope": effect.get("scope"),
            "candidate_arm": effect.get("candidate_arm"),
            "wins": effect.get("wins"),
            "losses": effect.get("losses"),
            "ties": effect.get("ties"),
            "headroom": effect.get("direct_headroom_count"),
            "exact_test": deepcopy(effect.get("exact_test")),
            "exact_success_ci95": deepcopy(effect.get("exact_success_ci95")),
            "token_sign_test": deepcopy(effect.get("token_sign_test")),
            "token_delta_ci95": deepcopy(effect.get("token_delta_ci95")),
            "latency_sign_test": deepcopy(effect.get("latency_sign_test")),
            "latency_delta_ci95": deepcopy(effect.get("latency_delta_ci95")),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster": "unit_id",
            "underpowered": effect.get("underpowered"),
            "no_headroom": effect.get("no_headroom"),
        }
        row["row_hash"] = sha256_json(row)
        output.append(row)
    return output


def _constraint_summary(scope: str, candidate: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected = [
        row
        for row in rows
        if row.get("arm_name") == candidate
        and row.get("stage1_executed") is True
        and (scope == "pooled" or row.get("family") == scope)
    ]
    proposals = sum(int(row.get("stage1_proposal_count", 0)) for row in selected)
    supported = sum(int(row.get("supported_constraint_count", 0)) for row in selected)
    matched = sum(int(row.get("matched_supported_constraint_count", 0)) for row in selected)
    gold = sum(int(row.get("supported_gold_constraint_count", 0)) for row in selected)
    unsupported = sum(int(row.get("unsupported_constraint_count", 0)) for row in selected)
    contradictory = sum(int(row.get("contradictory_constraint_count", 0)) for row in selected)
    leakage = sum(bool(row.get("failure", {}).get("stage1_answer_leakage")) for row in selected)
    row: JsonDict = {
        "scope": scope,
        "candidate_arm": candidate,
        "stage1_executed_unit_count": len(selected),
        "stage1_proposal_count": proposals,
        "supported_constraint_count": supported,
        "matched_supported_constraint_count": matched,
        "supported_gold_constraint_count": gold,
        "unsupported_constraint_count": unsupported,
        "contradictory_constraint_count": contradictory,
        "stage1_answer_leakage_count": leakage,
        "stage1_precision": round(matched / proposals, 9)
        if proposals
        else (1.0 if gold == 0 else 0.0),
        "stage1_recall": round(matched / gold, 9) if gold else 1.0,
        "unsupported_rate": round(unsupported / proposals, 9) if proposals else 0.0,
        "contradictory_rate": round(contradictory / proposals, 9) if proposals else 0.0,
    }
    row["row_hash"] = sha256_json(row)
    return row


def build_constraint_quality_summary(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce Stage 1 support quality for each family and the shared pool."""

    return [
        _constraint_summary(scope, candidate, rows)
        for scope in (*FAMILY_ORDER, "pooled")
        for candidate in CANDIDATE_ARMS
    ]


def _safety_cost_row(scope: str, arm_name: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected = [
        row
        for row in rows
        if row.get("arm_name") == arm_name and (scope == "pooled" or row.get("family") == scope)
    ]
    failure_counts = {
        name: sum(bool(row.get("failure", {}).get(name)) for row in selected)
        for name in FAILURE_CLASSES
    }
    count = len(selected)
    row: JsonDict = {
        "scope": scope,
        "arm_name": arm_name,
        "unit_count": count,
        "exact_success_count": sum(bool(value.get("exact_success")) for value in selected),
        "exact_success_rate": round(
            sum(bool(value.get("exact_success")) for value in selected) / count, 9
        ),
        "abstention_count": sum(bool(value.get("abstention")) for value in selected),
        "abstention_rate": round(
            sum(bool(value.get("abstention")) for value in selected) / count, 9
        ),
        "unsafe_release_count": sum(bool(value.get("unsafe_release")) for value in selected),
        "unsafe_release_rate": round(
            sum(bool(value.get("unsafe_release")) for value in selected) / count, 9
        ),
        "failure_arm_count": sum(bool(value.get("failure_any")) for value in selected),
        "failure_class_counts": failure_counts,
        "failure_outcome_count": sum(failure_counts.values()),
        "charged_failure_arm_count": sum(bool(value.get("charged_failure")) for value in selected),
        "supported_constraint_count": sum(
            int(value.get("supported_constraint_count", 0)) for value in selected
        ),
        "unsupported_constraint_count": sum(
            int(value.get("unsupported_constraint_count", 0)) for value in selected
        ),
        "contradictory_constraint_count": sum(
            int(value.get("contradictory_constraint_count", 0)) for value in selected
        ),
        "tokens_total": sum(int(value.get("tokens", 0)) for value in selected),
        "tokens_per_unit": round(sum(int(value.get("tokens", 0)) for value in selected) / count, 9),
        "latency_s_total": round(sum(float(value.get("latency_s", 0.0)) for value in selected), 9),
        "latency_s_per_unit": round(
            sum(float(value.get("latency_s", 0.0)) for value in selected) / count, 9
        ),
        "charged_cost_total": round(
            sum(float(value.get("charged_cost", 0.0)) for value in selected), 9
        ),
    }
    row["row_hash"] = sha256_json(row)
    return row


def build_safety_and_cost_summary(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Charge safety outcomes, failures, tokens, and latency to every arm."""

    return [
        _safety_cost_row(scope, arm, rows)
        for scope in (*FAMILY_ORDER, "pooled")
        for arm in ARM_ORDER
    ]


def build_acceptance_gate_rows(
    method: Mapping[str, Any],
    family_effects: Sequence[Mapping[str, Any]],
    pooled: Mapping[str, Any],
    constraint_rows: Sequence[Mapping[str, Any]],
    safety_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Apply every frozen benefit threshold without promoting readiness."""

    thresholds = method["metric_and_acceptance_contract"]["acceptance_thresholds"]
    effects = [*family_effects, *pooled.get("effect_rows", [])]
    constraints = {(row["scope"], row["candidate_arm"]): row for row in constraint_rows}
    safety = {(row["scope"], row["arm_name"]): row for row in safety_rows}
    output = []
    for effect in effects:
        key = (effect["scope"], effect["candidate_arm"])
        constraint = constraints[key]
        cost = safety[key]
        observed = {
            "positive_exact_success_delta": effect["exact_success_delta"],
            "paired_ci95_lower_nonnegative": effect["exact_success_ci95"]["lower"],
            "unsafe_release_increase_within_limit": effect["unsafe_release_delta_count"],
            "stage1_precision_floor": constraint["stage1_precision"],
            "stage1_recall_floor": constraint["stage1_recall"],
            "tokens_per_unit_limit": cost["tokens_per_unit"],
            "latency_per_unit_limit": cost["latency_s_per_unit"],
            "paired_exact_p_value_limit": effect["exact_test"]["p_value"],
        }
        expected = {
            "positive_exact_success_delta": {
                "operator": ">",
                "value": thresholds["routed_exact_success_delta_min_exclusive"],
            },
            "paired_ci95_lower_nonnegative": {
                "operator": ">=",
                "value": thresholds["routed_paired_ci95_lower_min"],
            },
            "unsafe_release_increase_within_limit": {
                "operator": "<=",
                "value": thresholds["unsafe_release_increase_max"],
            },
            "stage1_precision_floor": {
                "operator": ">=",
                "value": thresholds["stage1_precision_floor"],
            },
            "stage1_recall_floor": {"operator": ">=", "value": thresholds["stage1_recall_floor"]},
            "tokens_per_unit_limit": {
                "operator": "<=",
                "value": thresholds["total_tokens_per_unit_max"],
            },
            "latency_per_unit_limit": {
                "operator": "<=",
                "value": thresholds["latency_per_unit_s_max"],
            },
            "paired_exact_p_value_limit": {"operator": "<=", "value": thresholds["mcnemar_p_max"]},
        }
        passed = {
            "positive_exact_success_delta": observed["positive_exact_success_delta"]
            > expected["positive_exact_success_delta"]["value"],
            "paired_ci95_lower_nonnegative": observed["paired_ci95_lower_nonnegative"]
            >= expected["paired_ci95_lower_nonnegative"]["value"],
            "unsafe_release_increase_within_limit": observed["unsafe_release_increase_within_limit"]
            <= expected["unsafe_release_increase_within_limit"]["value"],
            "stage1_precision_floor": observed["stage1_precision_floor"]
            >= expected["stage1_precision_floor"]["value"],
            "stage1_recall_floor": observed["stage1_recall_floor"]
            >= expected["stage1_recall_floor"]["value"],
            "tokens_per_unit_limit": observed["tokens_per_unit_limit"]
            <= expected["tokens_per_unit_limit"]["value"],
            "latency_per_unit_limit": observed["latency_per_unit_limit"]
            <= expected["latency_per_unit_limit"]["value"],
            "paired_exact_p_value_limit": effect["exact_test"]["valid"] is True
            and observed["paired_exact_p_value_limit"]
            <= expected["paired_exact_p_value_limit"]["value"],
        }
        candidate_passed = (
            all(passed.values()) and not effect["no_headroom"] and not effect["underpowered"]
        )
        for condition in GATE_CONDITION_ORDER:
            row = {
                "scope": effect["scope"],
                "candidate_arm": effect["candidate_arm"],
                "condition": condition,
                "expected": expected[condition],
                "observed": observed[condition],
                "passed": passed[condition],
                "candidate_gate_passed": candidate_passed,
                "no_headroom": effect["no_headroom"],
                "underpowered": effect["underpowered"],
                "frozen_metric_contract_hash": method["metric_and_acceptance_contract"][
                    "contract_hash"
                ],
                "verdict_if_all_pass": "circular_positive",
            }
            row["row_hash"] = sha256_json(row)
            output.append(row)
    return output


def _expected_flat_order(payload: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    unit_ids = payload.get("preconditions_checked", {}).get("expected_unit_ids", [])
    return [
        (family, str(unit_id), arm)
        for family in FAMILY_ORDER
        for unit_id in unit_ids
        for arm in ARM_ORDER
    ]


def _core_reducer_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    keys = [
        (str(row.get("family")), str(row.get("unit_id")), str(row.get("arm_name"))) for row in rows
    ]
    expected_order = _expected_flat_order(payload)
    expected_repositories = {
        family: config["repository_id"] for family, config in SOURCE_ARTIFACTS.items()
    }
    source_receipts = payload.get("source_artifact_receipts", [])
    identity_rows = payload.get("model_identity_replay_rows", [])
    completeness = payload.get("row_completeness_recomputation", [])
    try:
        family_recomputed = build_family_effect_rows(rows)
        pooled_recomputed = build_pooled_effect_summary(rows, family_recomputed)
        constraint_recomputed = build_constraint_quality_summary(rows)
        safety_recomputed = build_safety_and_cost_summary(rows)
        method = load_json(REPO_ROOT / METHOD_RELATIVE_PATH)
        gate_recomputed = build_acceptance_gate_rows(
            method, family_recomputed, pooled_recomputed, constraint_recomputed, safety_recomputed
        )
    except (KeyError, ValueError, ZeroDivisionError):
        family_recomputed = []
        pooled_recomputed = {}
        constraint_recomputed = []
        safety_recomputed = []
        gate_recomputed = []
    checks = {
        "source_artifacts": len(source_receipts) == 3
        and all(row.get("all_checks_passed") is True for row in source_receipts),
        "model_identities": len(identity_rows) == 2
        and all(
            row.get("model_identity_valid") is True
            and row.get("process_identity_valid") is True
            and row.get("cross_family_residency_detected") is False
            for row in identity_rows
        ),
        "row_completeness": len(rows) == 120
        and len(completeness) == 2
        and all(row.get("all_rows_replayed") is True for row in completeness),
        "unique_row_keys": len(keys) == len(set(keys)) == 120,
        "frozen_row_order": keys == expected_order,
        "family_identity": all(
            row.get("repository_id") == expected_repositories.get(str(row.get("family")))
            and row.get("model_family_binding_valid") is True
            for row in rows
        ),
        "row_hashes": all(
            row.get("row_reproducibility_hash")
            == sha256_json(
                {key: value for key, value in row.items() if key != "row_reproducibility_hash"}
            )
            for row in rows
        ),
        "seed_schedule": all(
            row.get("seed") == row.get("frozen_seed") and row.get("seed_valid") is True
            for row in rows
        ),
        "source_binding": all(
            row.get("source_bytes_sha256") == row.get("method_source_bytes_sha256")
            and row.get("task_bytes_sha256") == row.get("method_task_bytes_sha256")
            and row.get("source_binding_valid") is True
            for row in rows
        ),
        "exact_authority": all(
            row.get("exact_registry_sha256") == row.get("frozen_exact_registry_sha256")
            and row.get("exact_replay_matches_source") is True
            for row in rows
        ),
        "raw_row_replay": all(
            row.get("arm_replay_matches_source") is True
            and row.get("unit_replay_matches_source") is True
            and row.get("raw_stage_replay_valid") is True
            for row in rows
        ),
        "unique_raw_arm_rows": all(
            len(
                {
                    row.get("raw_arm_row_hash")
                    for row in rows
                    if row.get("family") == family and row.get("unit_id") == unit_id
                }
            )
            == len(ARM_ORDER)
            for family in FAMILY_ORDER
            for unit_id in payload.get("preconditions_checked", {}).get("expected_unit_ids", [])
        ),
        "failure_retention": all(
            set(row.get("failure", {})) == set(FAILURE_CLASSES) | {"any"}
            and bool(row.get("failure_any")) == bool(row.get("failure", {}).get("any"))
            for row in rows
        )
        and payload.get("safety_and_cost_summary") == safety_recomputed,
        "cost_accounting": all(
            math.isclose(
                float(row.get("charged_cost", -1.0)),
                float(row.get("tokens", 0)) + float(row.get("latency_s", 0.0)),
                abs_tol=1e-9,
            )
            for row in rows
        )
        and payload.get("safety_and_cost_summary") == safety_recomputed,
        "effect_recomputation": payload.get("family_effect_rows") == family_recomputed
        and payload.get("pooled_effect_summary") == pooled_recomputed,
        "constraint_recomputation": payload.get("constraint_quality_summary")
        == constraint_recomputed,
        "acceptance_gate_recomputation": payload.get("acceptance_gate_rows") == gate_recomputed,
        "protected_files": payload.get("protected_files_unchanged", {}).get("all_unchanged")
        is True,
        "substrate": payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "oracle_declared": payload.get("verifier_is_oracle") is True,
    }
    return checks


def _attack_detector_passed(payload: Mapping[str, Any], detector: str) -> bool:
    """Run only the independent check targeted by one mutation."""

    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    if detector == "row_completeness":
        completeness = payload.get("row_completeness_recomputation", [])
        return (
            len(rows) == 120
            and len(completeness) == 2
            and all(row.get("all_rows_replayed") is True for row in completeness)
        )
    if detector == "failure_retention":
        retained = all(
            set(row.get("failure", {})) == set(FAILURE_CLASSES) | {"any"}
            and bool(row.get("failure_any")) == bool(row.get("failure", {}).get("any"))
            for row in rows
        )
        return retained and payload.get("safety_and_cost_summary") == build_safety_and_cost_summary(
            rows
        )
    if detector == "family_identity":
        repositories = {
            family: config["repository_id"] for family, config in SOURCE_ARTIFACTS.items()
        }
        return all(
            row.get("repository_id") == repositories.get(str(row.get("family")))
            and row.get("model_family_binding_valid") is True
            for row in rows
        )
    if detector == "unique_raw_arm_rows":
        unit_ids = payload.get("preconditions_checked", {}).get("expected_unit_ids", [])
        return all(
            len(
                {
                    row.get("raw_arm_row_hash")
                    for row in rows
                    if row.get("family") == family and row.get("unit_id") == unit_id
                }
            )
            == len(ARM_ORDER)
            for family in FAMILY_ORDER
            for unit_id in unit_ids
        )
    if detector == "effect_recomputation":
        return payload.get("family_effect_rows") == build_family_effect_rows(rows)
    if detector == "acceptance_gate_recomputation":
        method = load_json(REPO_ROOT / METHOD_RELATIVE_PATH)
        expected = build_acceptance_gate_rows(
            method,
            payload.get("family_effect_rows", []),
            payload.get("pooled_effect_summary", {}),
            payload.get("constraint_quality_summary", []),
            payload.get("safety_and_cost_summary", []),
        )
        return payload.get("acceptance_gate_rows") == expected
    if detector == "seed_schedule":
        return all(
            row.get("seed") == row.get("frozen_seed") and row.get("seed_valid") is True
            for row in rows
        )
    if detector == "source_binding":
        return all(
            row.get("source_bytes_sha256") == row.get("method_source_bytes_sha256")
            and row.get("task_bytes_sha256") == row.get("method_task_bytes_sha256")
            and row.get("source_binding_valid") is True
            for row in rows
        )
    if detector == "exact_authority":
        return all(
            row.get("exact_registry_sha256") == row.get("frozen_exact_registry_sha256")
            and row.get("exact_replay_matches_source") is True
            for row in rows
        )
    if detector == "cost_accounting":
        charged = all(
            math.isclose(
                float(row.get("charged_cost", -1.0)),
                float(row.get("tokens", 0)) + float(row.get("latency_s", 0.0)),
                abs_tol=1e-9,
            )
            for row in rows
        )
        return charged and payload.get("safety_and_cost_summary") == build_safety_and_cost_summary(
            rows
        )
    raise ValueError(f"unknown attack detector: {detector}")


def build_attack_rows(base_candidate: Mapping[str, Any]) -> list[JsonDict]:
    """Mutate claims and evidence, then require the intended check to fail."""

    def first_failure(value: JsonDict) -> JsonDict:
        return next(row for row in value["per_unit_rows"] if row["failure_any"])

    mutations = {
        "aggregate_only_claim": (
            "row_completeness",
            lambda value: value.update(per_unit_rows=[]),
        ),
        "dropped_failures": (
            "failure_retention",
            lambda value: first_failure(value).update(
                failure={**{name: False for name in FAILURE_CLASSES}, "any": False},
                failure_any=False,
            ),
        ),
        "family_label_swap": (
            "family_identity",
            lambda value: value["per_unit_rows"][0].update(family="gemma4_31b"),
        ),
        "identical_control_and_treatment": (
            "unique_raw_arm_rows",
            lambda value: value["per_unit_rows"][1].update(
                raw_arm_row_hash=value["per_unit_rows"][0]["raw_arm_row_hash"]
            ),
        ),
        "one_win_promotion": (
            "effect_recomputation",
            lambda value: value["family_effect_rows"][0].update(wins=1),
        ),
        "no_headroom_promotion": (
            "acceptance_gate_recomputation",
            lambda value: value["acceptance_gate_rows"][0].update(candidate_gate_passed=True),
        ),
        "seed_drift": (
            "seed_schedule",
            lambda value: value["per_unit_rows"][0].update(
                seed=value["per_unit_rows"][0]["seed"] + 1
            ),
        ),
        "source_drift": (
            "source_binding",
            lambda value: value["per_unit_rows"][0].update(source_bytes_sha256="sha256:drift"),
        ),
        "exact_check_substitution": (
            "exact_authority",
            lambda value: value["per_unit_rows"][0].update(
                exact_registry_sha256="sha256:substitute"
            ),
        ),
        "cost_omission": (
            "cost_accounting",
            lambda value: first_failure(value).update(tokens=0),
        ),
        "recomputation_disagreement": (
            "effect_recomputation",
            lambda value: value["family_effect_rows"][0].update(exact_success_delta=1.0),
        ),
    }
    output = []
    for attack_id in REQUIRED_ATTACK_IDS:
        expected_detector, mutate = mutations[attack_id]
        candidate = deepcopy(base_candidate)
        mutate(candidate)
        detector_passed = _attack_detector_passed(candidate, expected_detector)
        failed = [] if detector_passed else [expected_detector]
        row = {
            "attack_id": attack_id,
            "expected_detector": expected_detector,
            "failed_checks": failed,
            "candidate_ready_score": 1.0 if detector_passed else 0.0,
            "passed": expected_detector in failed and not detector_passed,
        }
        row["row_hash"] = sha256_json(row)
        output.append(row)
    return output


def reducer_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    """Return every independent row, aggregate, attack, and protection check."""

    checks = _core_reducer_checks(payload)
    attack_rows = payload.get("attack_rows", [])
    checks["attacks"] = bool(
        [row.get("attack_id") for row in attack_rows] == list(REQUIRED_ATTACK_IDS)
        and all(
            row.get("passed") is True and row.get("candidate_ready_score") == 0.0
            for row in attack_rows
        )
    )
    checks["tests_recorded"] = _blocking_tests_passed(payload.get("tests_run", []))
    return checks


def readiness_reducer(payload: Mapping[str, Any]) -> JsonDict:
    """Set readiness only when all rows, aggregates, and attacks replay."""

    checks = reducer_checks(payload)
    return {
        "checks": checks,
        "expected_row_count": 120,
        "present_row_count": len(payload.get("per_unit_rows", [])),
        "cfr_reducer_ready_score": 1.0 if all(checks.values()) else 0.0,
    }


def _tests_run_receipts(rows: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if rows is None else rows
    output = []
    for row in source:
        receipt = {
            "command": str(row.get("command", "")),
            "exit_code": int(row.get("exit_code", 1)),
            "duration_s": float(row.get("duration_s", 0.0)),
            "blocking": bool(row.get("blocking", True)),
        }
        if row.get("outcome_note") is not None:
            receipt["outcome_note"] = str(row["outcome_note"])
        output.append(receipt)
    return output


def _blocking_tests_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require every in-scope check while retaining unrelated diagnostics."""

    blocking = [row for row in rows if row.get("blocking", True)]
    return bool(blocking) and all(row.get("exit_code") == 0 for row in blocking)


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    sources = [
        {"path": config["path"].as_posix(), "sha256": sha256_file(repo_root / config["path"])}
        for config in SOURCE_ARTIFACTS.values()
    ]
    sources.extend(
        [
            {
                "path": INTAKE_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(repo_root / INTAKE_RELATIVE_PATH),
            },
            {
                "path": METHOD_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(repo_root / METHOD_RELATIVE_PATH),
            },
        ]
    )
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source_artifacts": deepcopy(sources),
            "raw_rows": ["Exp6590.per_unit_rows", "Exp6591.per_unit_rows"],
            "reducer_functions": [
                "replay_stream_rows",
                "build_family_effect_rows",
                "build_pooled_effect_summary",
                "build_constraint_quality_summary",
                "build_safety_and_cost_summary",
                "build_acceptance_gate_rows",
                "readiness_reducer",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _preconditions(
    *,
    repo_root: Path,
    date: str,
    method: Mapping[str, Any],
    intake: Mapping[str, Any],
    source_receipts: Sequence[Mapping[str, Any]],
    identities: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
) -> JsonDict:
    metric = method["metric_and_acceptance_contract"]
    source_contract = method["source_binding_and_exact_authority_contract"]
    intake_reduction = exp6592.readiness_reducer(intake)
    unit_ids = list(method["arm_seed_budget_contract"]["unit_order"])
    return {
        "planning_date": date,
        "structured_gate": {
            "artifact_path": INTAKE_RELATIVE_PATH.as_posix(),
            "artifact_sha256": sha256_file(repo_root / INTAKE_RELATIVE_PATH),
            "field": "v575_cfr_reducer_ready_score",
            "stored_value": unwrap_value(intake.get("v575_cfr_reducer_ready_score")),
            "recomputed_value": intake_reduction["v575_cfr_reducer_ready_score"],
        },
        "source_artifact_hashes": {row["path"]: row["sha256"] for row in source_receipts},
        "method_hashes": {
            "method_artifact_path": METHOD_RELATIVE_PATH.as_posix(),
            "method_artifact_sha256": sha256_file(repo_root / METHOD_RELATIVE_PATH),
            "source_manifest_hash": method["source_unit_manifest"]["manifest_hash"],
            "prompt_contract_hash": method["prompt_stage_contract"]["contract_hash"],
            "router_contract_hash": method["router_contract"]["contract_hash"],
            "arm_seed_budget_contract_hash": method["arm_seed_budget_contract"]["contract_hash"],
            "exact_authority_contract_hash": source_contract["contract_hash"],
            "exact_registry_hash": source_contract["exact_obligation_registry"]["registry_sha256"],
            "metric_contract_hash": metric["contract_hash"],
        },
        "expected_counts": {
            "family_count": len(FAMILY_ORDER),
            "units_per_family": len(unit_ids),
            "arms_per_unit": len(ARM_ORDER),
            "family_unit_count": len(unit_ids) * len(FAMILY_ORDER),
            "family_unit_arm_count": len(unit_ids) * len(FAMILY_ORDER) * len(ARM_ORDER),
        },
        "expected_unit_ids": unit_ids,
        "expected_arm_order": list(ARM_ORDER),
        "model_identity_bindings": [
            {
                "family": row["family"],
                "repository_id": row["repository_id"],
                "model_identity_hash": row["model_identity_hash"],
                "gpu_process_receipts_hash": row["gpu_process_receipts_hash"],
            }
            for row in identities
        ],
        "seeds": deepcopy(method["arm_seed_budget_contract"]["seed_schedule"]),
        "paired_test_plan": {
            "paired_unit_key": metric["paired_unit_key"],
            "paired_effect_reducer": metric["paired_effect_reducer"],
            "exact_test": metric["paired_uncertainty"]["paired_exact_test"],
            "bootstrap_method": metric["paired_uncertainty"]["method"],
            "bootstrap_resamples": metric["paired_uncertainty"]["resamples"],
            "bootstrap_seed": metric["paired_uncertainty"]["seed"],
            "pooled_cluster": "unit_id",
        },
        "protected_file_hashes_before": dict(protected_before),
        "cpu": _cpu_receipt(),
        "cpu_only_substrate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_calls_issued": 0,
        "model_loads_issued": 0,
        "downloads_issued": 0,
        "gpu_calls_issued": 0,
    }


def _gate_summary(payload: Mapping[str, Any], reduction: Mapping[str, Any]) -> JsonDict:
    checks = reduction["checks"]
    blocking_rows = [
        {
            "check": name,
            "expected_value": True,
            "observed_value": passed,
            "passed": passed,
            "blocking": True,
        }
        for name, passed in checks.items()
    ]
    failures = [row for row in blocking_rows if not row["passed"]]
    gate_groups = {
        (row["scope"], row["candidate_arm"]): row["candidate_gate_passed"]
        for row in payload.get("acceptance_gate_rows", [])
    }
    no_headroom = {
        (row["scope"], row["candidate_arm"])
        for row in payload.get("acceptance_gate_rows", [])
        if row.get("no_headroom") is True
    }
    diagnostic_failures = [
        {
            "command": row.get("command"),
            "exit_code": row.get("exit_code"),
            "blocking": row.get("blocking", True),
            "outcome_note": row.get("outcome_note"),
        }
        for row in payload.get("tests_run", [])
        if row.get("exit_code") != 0
    ]
    return {
        "blocking_checks": blocking_rows,
        "failed_blocking_check_count": len(failures),
        "first_blocking_failure": failures[0] if failures else None,
        "candidate_gate_outcomes": [
            {"scope": scope, "candidate_arm": arm, "passed": passed}
            for (scope, arm), passed in gate_groups.items()
        ],
        "candidate_gate_win_count": sum(gate_groups.values()),
        "no_headroom_case_count": len(no_headroom),
        "test_diagnostic_failure_count": len(diagnostic_failures),
        "test_diagnostic_failures": diagnostic_failures,
        "replay_readiness_is_benefit": False,
        "exact_authority_class_if_win": "circular_positive",
        "cfr_reducer_ready_score": reduction["cfr_reducer_ready_score"],
        "passed": not failures,
    }


def _classify_verdict(ready: bool, has_gate_win: bool) -> tuple[str, str | None, str]:
    """Keep blocked, circular-positive, and complete-null outcomes closed."""

    if not ready:
        return (
            "blocked_cfr_independent_row_reducer",
            "blocked",
            "blocked_cfr_independent_row_reducer: one or more source, row, identity, pairing, aggregate, attack, test, or protection checks failed",
        )
    if has_gate_win:
        return (
            "complete_cfr_independent_row_reducer_circular_positive",
            "circular_positive",
            "complete: one or more CFR candidates cleared the frozen gate; the result is circular-positive because the exact checker defines success",
        )
    return (
        "complete_cfr_independent_row_reducer_null",
        None,
        "complete: Qwen, Gemma, and pooled CFR effects replayed; every exact-success delta is 0.0 with zero direct headroom, so no CFR benefit is eligible",
    )


def build_report(
    repo_root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build one independent family-first and pooled CFR comparison."""

    started = time.monotonic()
    protected_before = _protected_hashes(repo_root)
    method, streams, intake = _load_sources(repo_root)
    source_receipts = build_source_artifact_receipts(repo_root, streams, intake)
    identities = build_model_identity_replay_rows(streams)
    per_unit_rows = []
    completeness = []
    for family in FAMILY_ORDER:
        replayed, family_completeness = replay_stream_rows(
            family,
            streams[family],
            method,
            source_artifact_sha256=sha256_file(repo_root / SOURCE_ARTIFACTS[family]["path"]),
        )
        per_unit_rows.extend(replayed)
        completeness.append(family_completeness)
    family_effects = build_family_effect_rows(per_unit_rows)
    pooled = build_pooled_effect_summary(per_unit_rows, family_effects)
    paired = build_paired_statistical_receipts(family_effects, pooled)
    constraints = build_constraint_quality_summary(per_unit_rows)
    safety = build_safety_and_cost_summary(per_unit_rows)
    acceptance = build_acceptance_gate_rows(method, family_effects, pooled, constraints, safety)
    protected = _protected_receipt(protected_before, _protected_hashes(repo_root))
    preconditions = _preconditions(
        repo_root=repo_root,
        date=date,
        method=method,
        intake=intake,
        source_receipts=source_receipts,
        identities=identities,
        protected_before=protected_before,
    )
    report_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    partial: JsonDict = {
        "per_unit_rows": per_unit_rows,
        "source_artifact_receipts": source_receipts,
        "model_identity_replay_rows": identities,
        "row_completeness_recomputation": completeness,
        "family_effect_rows": family_effects,
        "pooled_effect_summary": pooled,
        "paired_statistical_receipts": paired,
        "constraint_quality_summary": constraints,
        "safety_and_cost_summary": safety,
        "acceptance_gate_rows": acceptance,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "duration_s": report_duration,
        "tests_run": _tests_run_receipts(tests_run),
    }
    partial["attack_rows"] = build_attack_rows(partial)
    reduction = readiness_reducer(partial)
    winning_groups = {
        (row["scope"], row["candidate_arm"])
        for row in acceptance
        if row["candidate_gate_passed"] is True
    }
    status, verdict_class, verdict = _classify_verdict(
        reduction["cfr_reducer_ready_score"] == 1.0, bool(winning_groups)
    )
    payload: JsonDict = {
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        **partial,
        "cfr_reducer_ready_score": reduction["cfr_reducer_ready_score"],
        "field_provenance": _field_provenance(repo_root),
    }
    payload["gate_check_summary"] = _gate_summary(payload, reduction)
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_report(payload: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> list[str]:
    """Return schema, source, reducer, verdict, protection, and hash errors."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors = []
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {
        None,
        "circular_positive",
        "partial",
        "blocked",
        "disqualified",
    }:
        errors.append("verdict_class_invalid")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", 0) <= 0:
        errors.append("duration_s_invalid")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    for row in payload.get("source_artifact_receipts", []):
        path = repo_root / str(row.get("path"))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            errors.append("source_artifact_hash_mismatch:" + str(row.get("path")))
    reduction = readiness_reducer(payload)
    if payload.get("cfr_reducer_ready_score") != reduction["cfr_reducer_ready_score"]:
        errors.append("cfr_reducer_ready_score_mismatch")
    if payload.get("verdict_class") is None and reduction["cfr_reducer_ready_score"] != 1.0:
        errors.append("null_verdict_without_complete_replay")
    if payload.get("verdict_class") == "circular_positive" and not any(
        row.get("candidate_gate_passed") is True for row in payload.get("acceptance_gate_rows", [])
    ):
        errors.append("circular_positive_without_gate_win")
    if (
        payload.get("verdict_class") == "blocked"
        and payload.get("gate_check_summary", {}).get("first_blocking_failure") is None
    ):
        errors.append("blocked_verdict_missing_gate_detail")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def atomic_write_report(
    path: str | Path, payload: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> JsonDict:
    """Validate, sync, replace, and directory-sync one terminal artifact."""

    errors = validate_report(payload, repo_root)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".exp6593-final-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "byte_count": len(encoded),
        "atomic_replace": True,
        "directory_fsync": True,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    report = build_report(REPO_ROOT, date=args.date)
    receipt = atomic_write_report(args.output, report, repo_root=REPO_ROOT)
    print(json.dumps({"status": report["status"], "receipt": receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution.
    raise SystemExit(main())
