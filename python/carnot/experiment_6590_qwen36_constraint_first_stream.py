"""Run one immutable Qwen constraint-first stream through llama.cpp.

The module first rechecks every frozen Exp6587 method byte. It then runs direct,
always-on CFR, and routed CFR arms in one fresh local Qwen process. Raw stages
reach durable unit checkpoints before later units run. Exact fixture checks own
release, so this artifact proves row completeness and makes no CFR benefit claim.

Spec refs: REQ-REPORT-6590 and SCENARIO-REPORT-6590-FROZEN through
SCENARIO-REPORT-6590-ATOMIC.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any
import urllib.error

from carnot import experiment_6573_sequential_flagship_gguf_admission_v2 as runtime_helpers
from carnot import experiment_6574_joint_sufficiency_method_contract as exact
from carnot import experiment_6581_qwen36_flagship_source_shard as shard
from carnot import experiment_6587_v573_constraint_first_method_contract as method_builder
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
TASK_ID = "exp6590-qwen36-constraint-first-stream"
QWEN_REPOSITORY_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
QWEN_ARCHITECTURE = "qwen35moe"
INFERENCE_SUBSTRATE = "fresh_local_qwen36_gguf_cfr_inference"
EXACT_CHECKER_NAME = "exp6574.compile_node_plus_joint_sufficiency_reduce"

RESULT_RELATIVE_PATH = Path("results/experiment_6590_qwen36_constraint_first_stream.json")
CHECKPOINT_RELATIVE_PATH = Path("results/experiment_6590_qwen36_constraint_first_stream.raw")
METHOD_RELATIVE_PATH = Path("results/experiment_6587_v573_constraint_first_method_contract.json")
LAUNCH_RELATIVE_PATH = Path("results/experiment_6588_v574_bounded_cfr_launch_root.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6590_qwen36_constraint_first_stream.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6590_qwen36_constraint_first_stream.py")
PROTECTED_RELATIVE_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

ARM_ORDER = ("direct", "always_on_cfr", "routed_cfr")
STAGE_ORDER = ("direct", "stage1", "stage2")
FAILURE_CLASSES = (
    "timeout",
    "malformed_output",
    "process_failure",
    "contradiction",
    "unsupported_constraint",
    "exact_rejection",
    "task_deadline_exhausted",
    "stage1_answer_leakage",
)
REQUIRED_ATTACK_IDS = (
    "prompt_drift",
    "post_outcome_unit_loss",
    "stage_overwrite",
    "answer_leakage",
    "uncharged_stage1",
    "family_label_substitution",
    "legacy_model_substitution",
    "aggregate_only_output",
    "ready_score_with_missing_rows",
)

MODEL_SPECS = [
    {
        "name": "Qwen3.6-35B-A3B",
        "repository_id": QWEN_REPOSITORY_ID,
        "expected_architecture": QWEN_ARCHITECTURE,
        "quantization": "Q4_K_M",
        "headline_eligible": True,
    }
]

LOAD_TIMEOUT_S = 600.0
PER_GENERATION_TIMEOUT_S = 60.0
TASK_TIMEOUT_S = 4200.0
SHUTDOWN_TIMEOUT_S = 30.0
RECOVERY_TIMEOUT_S = 180.0
RECOVERY_TOLERANCE_MB = 256
GPU_LOAD_DELTA_MIN_MB = 128
TELEMETRY_INTERVAL_S = 1.0
CONTEXT_SIZE = 4096
LAUNCH_MAX_OUTPUT_TOKENS = 512

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "per_unit_rows",
    "model_spec_and_identity",
    "prompt_source_router_hashes",
    "raw_stage_receipts",
    "exact_checker_receipts",
    "checkpoint_receipts",
    "gpu_process_receipts",
    "failure_rows",
    "qwen_cfr_rows_ready_score",
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
    "status": "The stream ends as complete rows, bounded partial rows, or a named precondition block.",
    "honest_verdict": "The verdict reports Qwen row completeness without claiming CFR benefit.",
    "verdict_class": "A complete source stream is null evidence infrastructure.",
    "gate_check_summary": "A block names the exact gate, cache, resource, drift, or runtime value.",
    "per_unit_rows": "Every source unit and arm carries raw stages, exact results, failures, tokens, and latency.",
    "model_spec_and_identity": "The mandated Qwen GGUF spec and content-derived local file identity bind inference.",
    "prompt_source_router_hashes": "The Exp6587 byte-frozen method cannot drift by family or outcome.",
    "raw_stage_receipts": "Direct, Stage 1, and Stage 2 bytes remain separate and immutable.",
    "exact_checker_receipts": "Whitelisted executable checks, not the model, decide validity.",
    "checkpoint_receipts": "Each completed unit survives later timeout or process failure.",
    "gpu_process_receipts": "Ownership, offload, memory, utilization, and clean unload bind the local run.",
    "failure_rows": "Timeouts, parse failures, unsupported constraints, contradictions, and exact rejection remain evidence.",
    "qwen_cfr_rows_ready_score": "This exact binary field gates the independent comparison.",
    "attack_rows": "Drift, leakage, substitution, dropped failures, and aggregate-only claims fail closed.",
    "preconditions_checked": "Gates, hashes, cache, resources, ownership, budgets, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain their original hashes.",
    "inference_substrate": "The task declares fresh local Qwen GGUF inference through llama.cpp.",
    "verifier_is_oracle": "Exact checks define row validity, so later exact wins are circular-positive.",
    "field_provenance": "Every field names source rows, raw bytes, process receipts, and reducer code.",
    "duration_s": "Monotonic duration exposes smoke-only or truncated execution.",
    "tests_run": "Named commands, exits, durations, and focused scope make validation reproducible.",
    "reproducibility_checksum": "A final content hash protects the immutable stream.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6590_qwen36_constraint_first_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6590_qwen36_constraint_first_stream.py "
    "-m pytest tests/python/test_experiment_6590_qwen36_constraint_first_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6590_qwen36_constraint_first_stream.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = f".venv/bin/ruff check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
RUFF_FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_RELATIVE_PATH}"

canonical_json = shard.canonical_json
sha256_bytes = shard.sha256_bytes
sha256_text = shard.sha256_text
sha256_json = shard.sha256_json
sha256_file = shard.sha256_file
load_json = shard.load_json
row_hash = shard.row_hash
artifact_checksum = shard.artifact_checksum


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    """Hash one stored contract without trusting its self-hash."""

    return sha256_json({key: item for key, item in value.items() if key != field})


def build_gate_receipt(repo_root: Path) -> JsonDict:
    """Read only the exact Exp6588 launch field owned by this roadmap."""

    path = repo_root / LAUNCH_RELATIVE_PATH
    observed = load_json(path).get("v574_cfr_launch_ready_score")
    return {
        "upstream": "exp6588-v574-bounded-cfr-launch-root",
        "path": LAUNCH_RELATIVE_PATH.as_posix(),
        "absolute_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "field": "v574_cfr_launch_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "passed": observed == 1.0,
    }


def build_frozen_hash_receipt(repo_root: Path, method: Mapping[str, Any]) -> JsonDict:
    """Rebuild every frozen method hash before any model process can start."""

    manifest = method.get("source_unit_manifest", {})
    prompts = method.get("prompt_stage_contract", {})
    router = method.get("router_contract", {})
    arms = method.get("arm_seed_budget_contract", {})
    authority = method.get("source_binding_and_exact_authority_contract", {})
    units = manifest.get("units", []) if isinstance(manifest, Mapping) else []
    units = [dict(row) for row in units if isinstance(row, Mapping)]
    current_manifest = method_builder.build_source_unit_manifest(repo_root)
    current_prompts = method_builder.build_prompt_stage_contract()
    current_router = method_builder.build_router_contract(current_manifest)
    current_arms = method_builder.build_arm_seed_budget_contract(current_manifest)
    current_authority = method_builder.build_source_binding_and_exact_authority_contract(repo_root)

    prompt_rows = prompts.get("prompts", {}) if isinstance(prompts, Mapping) else {}
    prompt_hashes = bool(prompt_rows) and all(
        isinstance(row, Mapping) and row.get("sha256") == sha256_text(str(row.get("text", "")))
        for row in prompt_rows.values()
    )
    source_hashes = bool(units) and all(
        row.get("source_bytes_sha256") == sha256_text(str(row.get("exact_source_bytes", "")))
        and row.get("task_bytes_sha256") == sha256_text(str(row.get("exact_task_bytes", "")))
        and row.get("row_hash") == row_hash(row)
        for row in units
    )
    routing_rows = router.get("routing_rows", []) if isinstance(router, Mapping) else []
    router_hashes = (
        isinstance(router, Mapping)
        and router.get("contract_hash") == _hash_without(router, "contract_hash")
        and router.get("contract_hash") == current_router.get("contract_hash")
        and bool(routing_rows)
        and all(
            isinstance(row, Mapping) and row.get("row_hash") == row_hash(row)
            for row in routing_rows
        )
    )
    seed_rows = arms.get("seed_schedule", []) if isinstance(arms, Mapping) else []
    unit_order = arms.get("unit_order", []) if isinstance(arms, Mapping) else []
    arm_rows = arms.get("arms", {}) if isinstance(arms, Mapping) else {}
    arm_hashes = (
        isinstance(arms, Mapping)
        and arms.get("contract_hash") == _hash_without(arms, "contract_hash")
        and arms.get("contract_hash") == current_arms.get("contract_hash")
        and [row.get("unit_id") for row in units] == unit_order
        and sha256_json(unit_order) == next(iter(arm_rows.values()), {}).get("unit_order_hash")
        and sha256_json(seed_rows) == next(iter(arm_rows.values()), {}).get("seed_schedule_hash")
        and sha256_json(arms.get("decoding"))
        == next(iter(arm_rows.values()), {}).get("decoding_hash")
        and sha256_json(arms.get("stop_rules"))
        == next(iter(arm_rows.values()), {}).get("stop_rules_hash")
    )
    registry = (
        authority.get("exact_obligation_registry", {}) if isinstance(authority, Mapping) else {}
    )
    checker_hash = (
        isinstance(authority, Mapping)
        and authority.get("contract_hash") == _hash_without(authority, "contract_hash")
        and authority.get("contract_hash") == current_authority.get("contract_hash")
        and registry.get("registry_sha256")
        == current_authority.get("exact_obligation_registry", {}).get("registry_sha256")
        and authority.get("exact_obligation_dispatch")
        == current_authority.get("exact_obligation_dispatch")
        and all(row.get("checker") == EXACT_CHECKER_NAME for row in units)
    )
    checks = {
        "method_ready": method.get("v573_constraint_first_method_ready_score") == 1.0,
        "source_manifest_hash": isinstance(manifest, Mapping)
        and manifest.get("manifest_hash") == _hash_without(manifest, "manifest_hash")
        and manifest.get("manifest_hash") == current_manifest.get("manifest_hash"),
        "source_unit_hashes": source_hashes,
        "prompt_hashes": isinstance(prompts, Mapping)
        and prompts.get("contract_hash") == _hash_without(prompts, "contract_hash")
        and prompts.get("contract_hash") == current_prompts.get("contract_hash")
        and prompt_hashes,
        "router_hashes": bool(router_hashes),
        "decoding_seed_order_budget_hashes": bool(arm_hashes),
        "checker_registry_hash": bool(checker_hash),
    }
    return {
        "method_artifact_path": METHOD_RELATIVE_PATH.as_posix(),
        "method_artifact_sha256": sha256_file(repo_root / METHOD_RELATIVE_PATH),
        "source_manifest_hash": manifest.get("manifest_hash")
        if isinstance(manifest, Mapping)
        else None,
        "expected_unit_count": len(units),
        "expected_unit_ids": [row.get("unit_id") for row in units],
        "source_unit_row_hashes": [row.get("row_hash") for row in units],
        "prompt_contract_hash": prompts.get("contract_hash")
        if isinstance(prompts, Mapping)
        else None,
        "router_contract_hash": router.get("contract_hash")
        if isinstance(router, Mapping)
        else None,
        "arm_seed_budget_contract_hash": arms.get("contract_hash")
        if isinstance(arms, Mapping)
        else None,
        "exact_authority_contract_hash": authority.get("contract_hash")
        if isinstance(authority, Mapping)
        else None,
        "exact_registry_hash": registry.get("registry_sha256")
        if isinstance(registry, Mapping)
        else None,
        "checks": checks,
        "all_frozen_hashes_match": all(checks.values()),
    }


def empty_failure_flags() -> dict[str, bool]:
    """Return every required terminal failure field at a false baseline."""

    return {name: False for name in FAILURE_CLASSES}


def make_stage_receipt(
    *,
    stage: str,
    prompt_sha256: str,
    request_sha256: str,
    raw_bytes: bytes,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    stop_reason: str,
    request_status: int,
    recorded_monotonic_ns: int,
    failure_flags: Mapping[str, bool] | None = None,
) -> JsonDict:
    """Seal one raw stage before any parser or exact checker consumes it."""

    failures = empty_failure_flags()
    failures.update(dict(failure_flags or {}))
    receipt: JsonDict = {
        "stage": stage,
        "prompt_sha256": prompt_sha256,
        "request_sha256": request_sha256,
        "raw_bytes_b64": base64.b64encode(raw_bytes).decode("ascii"),
        "raw_byte_count": len(raw_bytes),
        "raw_sha256": sha256_bytes(raw_bytes),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "latency_s": round(float(latency_s), 9),
        "stop_reason": stop_reason,
        "request_status": int(request_status),
        "recorded_monotonic_ns": int(recorded_monotonic_ns),
        "failure_flags": failures,
    }
    receipt["row_hash"] = row_hash(receipt)
    return receipt


def _decode_stage(receipt: Mapping[str, Any] | None) -> bytes | None:
    """Decode exact retained bytes or reject malformed receipt encoding."""

    if not isinstance(receipt, Mapping):
        return None
    try:
        return base64.b64decode(str(receipt.get("raw_bytes_b64", "")), validate=True)
    except (TypeError, ValueError):
        return None


def parse_stage1_proposals(raw: bytes, unit: Mapping[str, Any]) -> list[JsonDict]:
    """Parse only quoted plain text and project exact known semantics deterministically."""

    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return []
    candidates = re.findall(r'["“]([^"”]+)["”]', text)
    source = str(unit.get("exact_source_bytes", ""))
    gold_by_span = {
        str(row.get("quoted_span")): row
        for row in unit.get("gold_constraints", [])
        if isinstance(row, Mapping)
    }
    rows = []
    for candidate in dict.fromkeys(candidates):
        gold = gold_by_span.get(candidate)
        if gold is not None:
            rows.append(
                {
                    "constraint_id": gold.get("constraint_id"),
                    "quoted_span": candidate,
                    "source_start": gold.get("source_start"),
                    "source_end": gold.get("source_end"),
                    "relation": gold.get("relation"),
                    "operands": deepcopy(gold.get("operands", {})),
                    "parser_used_gold_semantics": True,
                    "unsupported": False,
                }
            )
            continue
        start = source.find(candidate)
        rows.append(
            {
                "constraint_id": None,
                "quoted_span": candidate,
                "source_start": start,
                "source_end": start + len(candidate.encode("utf-8")) if start >= 0 else -1,
                "relation": "unparsed_plain_text",
                "operands": {},
                "parser_used_gold_semantics": False,
                "unsupported": True,
            }
        )
    return rows


def cost_from_stages(stages: Mapping[str, Any]) -> float:
    """Charge input tokens, output tokens, and latency for every present stage."""

    return round(
        sum(
            int(row.get("total_tokens", 0) or 0) + float(row.get("latency_s", 0.0) or 0.0)
            for row in stages.values()
            if isinstance(row, Mapping)
        ),
        9,
    )


def _stage1_leaks_answer(raw: bytes | None) -> bool:
    """Detect explicit result transport while allowing the frozen abstention phrase."""

    if raw is None:
        return False
    text = raw.decode("utf-8", "replace").casefold()
    return bool(re.search(r"\b(?:final answer|result is|answer is)\b", text))


def build_arm_row(
    *,
    unit: Mapping[str, Any],
    arm_name: str,
    route: str,
    seed: int,
    stage_receipts: Mapping[str, Mapping[str, Any]],
    registry_hash: str,
) -> JsonDict:
    """Combine raw stages with deterministic source binding and exact authority."""

    raw_stages = {stage: deepcopy(stage_receipts.get(stage)) for stage in STAGE_ORDER}
    stage1_raw = _decode_stage(raw_stages.get("stage1"))
    proposals = parse_stage1_proposals(stage1_raw or b"", unit)
    bindings = []
    for proposal in proposals:
        if proposal.get("unsupported") is True:
            bindings.append(
                {
                    "constraint_id": proposal.get("constraint_id"),
                    "source_supported": False,
                    "unsupported": True,
                    "contradictory": False,
                    "exact_result": "unsupported_plain_text_proposal",
                    "action": "abstain",
                    "release_eligible": False,
                }
            )
        else:
            bindings.append(
                {
                    "constraint_id": proposal.get("constraint_id"),
                    **method_builder.bind_constraint_proposal(
                        str(unit.get("exact_source_bytes", "")), proposal
                    ),
                }
            )
    fixture = exact.build_fixture(str(unit.get("fixture_id")))
    exact_fixture = exact.evaluate_fixture(fixture)
    expected_action = str(unit.get("expected_action"))
    authority_action = str(exact_fixture.get("action"))
    contradiction = (
        any(row.get("contradictory") is True for row in bindings)
        or unit.get("case_class") == "contradictory"
    )
    unsupported = (
        any(row.get("unsupported") is True for row in bindings)
        or unit.get("case_class") == "unsupported"
    )
    failure = empty_failure_flags()
    for receipt in raw_stages.values():
        if isinstance(receipt, Mapping):
            for name in FAILURE_CLASSES:
                failure[name] = failure[name] or bool(receipt.get("failure_flags", {}).get(name))
    failure["contradiction"] = contradiction
    failure["unsupported_constraint"] = unsupported
    failure["exact_rejection"] = authority_action != "release"
    failure["stage1_answer_leakage"] = _stage1_leaks_answer(stage1_raw)
    failure["any"] = any(failure.values())
    supported_gold = {
        row.get("constraint_id")
        for row in unit.get("gold_constraints", [])
        if isinstance(row, Mapping) and row.get("constraint_class") == "supported"
    }
    matched = {
        row.get("constraint_id")
        for row in proposals
        if row.get("unsupported") is False and row.get("constraint_id") in supported_gold
    }
    supported_proposals = sum(row.get("unsupported") is False for row in proposals)
    precision = (
        len(matched) / supported_proposals
        if supported_proposals
        else (1.0 if not supported_gold else 0.0)
    )
    recall = len(matched) / len(supported_gold) if supported_gold else 1.0
    token_counts = {
        stage: int(receipt.get("total_tokens", 0) or 0) if isinstance(receipt, Mapping) else 0
        for stage, receipt in raw_stages.items()
    }
    token_counts.update(
        {
            "stage1_charged": raw_stages["stage1"] is None or token_counts["stage1"] > 0,
            "stage2_charged": raw_stages["stage2"] is None or token_counts["stage2"] > 0,
            "direct_charged": raw_stages["direct"] is None or token_counts["direct"] > 0,
            "total": sum(token_counts.values()),
        }
    )
    row: JsonDict = {
        "arm_name": arm_name,
        "route": route,
        "seed": int(seed),
        "raw_stages": raw_stages,
        "stage1_passed_verbatim_to_stage2": (
            raw_stages["stage1"] is not None and raw_stages["stage2"] is not None
        )
        or (raw_stages["stage1"] is None and raw_stages["stage2"] is None),
        "parsed_proposals": proposals,
        "source_span_bindings": bindings,
        "exact_results": {
            "checker": EXACT_CHECKER_NAME,
            "checker_version": exact.COMPILER_VERSION,
            "registry_sha256": registry_hash,
            "fixture_hash": unit.get("fixture_hash"),
            "expected_action": expected_action,
            "observed_action": authority_action,
            "abstention_reasons": exact_fixture.get("abstention_reasons", []),
            "model_is_release_authority": False,
            "exact_success": authority_action == expected_action,
        },
        "stage1_precision": round(precision, 9),
        "stage1_recall": round(recall, 9),
        "unsupported_constraint_count": sum(row.get("unsupported") is True for row in bindings),
        "contradictory_constraint_count": sum(row.get("contradictory") is True for row in bindings),
        "abstention": authority_action != "release",
        "unsafe_release": False,
        "tokens": token_counts,
        "latency_s": round(
            sum(
                float(receipt.get("latency_s", 0.0) or 0.0)
                for receipt in raw_stages.values()
                if isinstance(receipt, Mapping)
            ),
            9,
        ),
        "charged_cost_unit": "normalized_token_and_second_units",
        "charged_cost": cost_from_stages(raw_stages),
        "failure": failure,
        "attempt_count": 1,
        "retry_count": 0,
    }
    row["row_hash"] = row_hash(row)
    return row


def finalize_unit_row(
    unit: Mapping[str, Any],
    arms: Sequence[Mapping[str, Any]],
    process_binding: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Bind one complete unit to exact source bytes, ordered arms, and process identity."""

    row: JsonDict = {
        "unit_id": unit.get("unit_id"),
        "selection_index": unit.get("selection_index"),
        "fixture_id": unit.get("fixture_id"),
        "case_class": unit.get("case_class"),
        "stratum": unit.get("stratum"),
        "split": unit.get("split"),
        "source_bytes_b64": base64.b64encode(
            str(unit.get("exact_source_bytes", "")).encode("utf-8")
        ).decode("ascii"),
        "source_bytes_sha256": unit.get("source_bytes_sha256"),
        "task_bytes_b64": base64.b64encode(
            str(unit.get("exact_task_bytes", "")).encode("utf-8")
        ).decode("ascii"),
        "task_bytes_sha256": unit.get("task_bytes_sha256"),
        "fixture_hash": unit.get("fixture_hash"),
        "expected_action": unit.get("expected_action"),
        "process_binding": dict(process_binding or {}),
        "arms": [dict(arm) for arm in arms],
    }
    row["row_hash"] = row_hash(row)
    return row


def build_raw_stage_receipts(per_unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Flatten all retained raw stages without discarding their recoverable bytes."""

    rows = []
    for unit in per_unit_rows:
        for arm in unit.get("arms", []):
            for stage in STAGE_ORDER:
                receipt = arm.get("raw_stages", {}).get(stage)
                if isinstance(receipt, Mapping):
                    rows.append(
                        {
                            "unit_id": unit.get("unit_id"),
                            "arm_name": arm.get("arm_name"),
                            "stage": stage,
                            "raw_bytes_b64": receipt.get("raw_bytes_b64"),
                            "raw_byte_count": receipt.get("raw_byte_count"),
                            "raw_sha256": receipt.get("raw_sha256"),
                            "stage_row_hash": receipt.get("row_hash"),
                            "recorded_monotonic_ns": receipt.get("recorded_monotonic_ns"),
                        }
                    )
    return rows


def build_exact_checker_receipts(per_unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Flatten one independently recheckable exact result for every arm row."""

    return [
        {
            "unit_id": unit.get("unit_id"),
            "arm_name": arm.get("arm_name"),
            **dict(arm.get("exact_results", {})),
        }
        for unit in per_unit_rows
        for arm in unit.get("arms", [])
    ]


def build_failure_rows(per_unit_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retain every arm whose runtime or exact path ended in a named failure."""

    rows = []
    for unit in per_unit_rows:
        for arm in unit.get("arms", []):
            failure = arm.get("failure", {})
            classes = [name for name in FAILURE_CLASSES if bool(failure.get(name))]
            if classes:
                rows.append(
                    {
                        "unit_id": unit.get("unit_id"),
                        "arm_name": arm.get("arm_name"),
                        "failure_classes": classes,
                        "failure": dict(failure),
                        "arm_row_hash": arm.get("row_hash"),
                    }
                )
    return rows


def write_unit_checkpoint(
    checkpoint_dir: Path, completed_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Durably replace one completed-prefix checkpoint after a full unit."""

    rows = [dict(row) for row in completed_rows]
    payload = {
        "schema": "carnot.exp6590.completed_unit_prefix.v1",
        "completed_unit_count": len(rows),
        "completed_unit_ids": [row.get("unit_id") for row in rows],
        "completed_unit_rows": rows,
        "prefix_hash": sha256_json(rows),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(rows[-1].get("unit_id", "empty")).removeprefix("sha256:")[:12] if rows else "empty"
    target = checkpoint_dir / f"unit-{len(rows):02d}-{suffix}.json"
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_dir, prefix=".exp6590-unit-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "completed_unit_count": len(rows),
        "completed_unit_ids": payload["completed_unit_ids"],
        "completed_unit_row_hashes": [row.get("row_hash") for row in rows],
        "prefix_hash": payload["prefix_hash"],
        "absolute_path": str(target.resolve()),
        "checkpoint_sha256": sha256_file(target),
        "byte_count": len(encoded),
        "written_monotonic_ns": time.monotonic_ns(),
        "atomic_replace": True,
        "directory_fsync": True,
    }


def model_identity_checks(receipt: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute Qwen family, cache provenance, tokenizer, and CUDA-build identity."""

    specs = receipt.get("model_specs", [])
    identity = receipt.get("identity", {})
    pair = receipt.get("cached_sota_pair", []) or []
    content = identity.get("content_metadata", {}) if isinstance(identity, Mapping) else {}
    bounded = content.get("bounded_read_receipt", {}) if isinstance(content, Mapping) else {}
    provenance = identity.get("provenance", {}) if isinstance(identity, Mapping) else {}
    tokenizer = content.get("tokenizer_metadata", {}) if isinstance(content, Mapping) else {}
    build = receipt.get("llama_cpp_build", {})
    return {
        "headline_model": specs == MODEL_SPECS,
        "cached_sota_pair_pattern": any(
            isinstance(row, Mapping)
            and row.get("hf_id") == QWEN_REPOSITORY_ID
            and bool(row.get("model_path"))
            for row in pair
        ),
        "content_identity": bool(
            identity.get("repository_id") == QWEN_REPOSITORY_ID
            and identity.get("admitted") is True
            and not identity.get("rejection_reasons")
            and content.get("architecture") == QWEN_ARCHITECTURE
            and content.get("quantization") == "Q4_K_M"
            and content.get("is_language_model") is True
            and int(content.get("tensor_count", 0) or 0) > 0
            and bounded.get("tensor_payload_bytes_read") == 0
            and provenance.get("valid") is True
            and provenance.get("repository_id") == QWEN_REPOSITORY_ID
            and provenance.get("trusted_sha256") == identity.get("trusted_sha256")
        ),
        "embedded_tokenizer": receipt.get("embedded_tokenizer_used") is True
        and int(tokenizer.get("token_count", 0) or 0) > 0,
        "no_auto_tokenizer": receipt.get("auto_tokenizer_used") is False
        and identity.get("auto_tokenizer_used", False) is False,
        "no_download": receipt.get("download_performed") is False
        and identity.get("download_performed", False) is False,
        "llama_cpp_cuda": build.get("exists") is True
        and build.get("executable") is True
        and build.get("cuda_linked") is True
        and build.get("version_receipt", {}).get("exit_code") == 0,
    }


def process_lifecycle_checks(receipts: Mapping[str, Any]) -> dict[str, bool]:
    """Recheck one owned process, positive offload, telemetry, and clean unload."""

    process = receipts.get("process", {})
    unload = receipts.get("unload", {})
    pid = int(process.get("pid", 0) or 0)
    samples = [row for row in process.get("gpu_samples", []) if isinstance(row, Mapping)]
    before = [row for row in samples if row.get("stage") == "before"]
    during = [row for row in samples if row.get("stage") == "during"]
    after = [row for row in samples if row.get("stage") == "after"]
    linked = [
        item
        for sample in during
        for item in sample.get("compute_processes", [])
        if int(item.get("pid", 0) or 0) == pid
    ]
    baseline = min(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in before),
        default=0,
    )
    peak = max(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in during),
        default=0,
    )
    command = [str(part) for part in process.get("command", [])]
    model_path = str(process.get("selected_blob_path", ""))
    return {
        "one_fresh_owned_process": pid > 1
        and process.get("fresh_process") is True
        and process.get("owned_child") is True,
        "command_bound": process.get("command_sha256") == sha256_json(command)
        and bool(model_path)
        and model_path in command,
        "runtime_selected_gpu": process.get("selected_gpu") is not None
        and str(process.get("cuda_visible_devices")) == str(process.get("selected_gpu")),
        "positive_offload": int(process.get("offloaded_layers", 0) or 0) > 0,
        "server_healthy": process.get("server_healthy") is True
        and process.get("http_status") == 200,
        "repeated_samples": bool(before) and len(during) >= 2 and bool(after),
        "pid_linked_to_gpu": bool(linked),
        "positive_gpu_memory": peak - baseline >= GPU_LOAD_DELTA_MIN_MB,
        "utilization_sampled": bool(during)
        and all("utilization_pct" in row.get("device", {}) for row in during),
        "one_family_resident": process.get("resident_model_families") == [QWEN_REPOSITORY_ID],
        "clean_process_exit": process.get("shutdown_requested") is True
        and process.get("normal_shutdown") is True
        and process.get("exit_code") in {0, -signal.SIGTERM}
        and process.get("worker_alive_after_exit") is False,
        "clean_unload": unload.get("worker_absent_from_proc") is True
        and unload.get("worker_absent_from_nvidia_smi") is True
        and unload.get("port_closed") is True
        and abs(int(unload.get("memory_delta_from_baseline_mb", 0) or 0)) <= RECOVERY_TOLERANCE_MB
        and unload.get("recovery_tolerance_mb") == RECOVERY_TOLERANCE_MB
        and unload.get("no_task_worker_remains") is True
        and unload.get("recovery_bounded") is True
        and unload.get("recovery_complete") is True,
        "unrelated_processes_preserved": process.get("signals_sent_to_unrelated_pids") == []
        and unload.get("signals_sent_to_unrelated_pids") == [],
    }


def _stage_authentic(receipt: Mapping[str, Any] | None, stage: str) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    raw = _decode_stage(receipt)
    failures = receipt.get("failure_flags", {})
    return bool(
        raw is not None
        and receipt.get("stage") == stage
        and receipt.get("raw_sha256") == sha256_bytes(raw)
        and int(receipt.get("raw_byte_count", -1) or 0) == len(raw)
        and int(receipt.get("total_tokens", -1) or 0)
        == int(receipt.get("prompt_tokens", 0) or 0) + int(receipt.get("completion_tokens", 0) or 0)
        and float(receipt.get("latency_s", -1.0) or 0.0) >= 0.0
        and isinstance(failures, Mapping)
        and set(failures) == set(FAILURE_CLASSES)
        and receipt.get("row_hash") == row_hash(receipt)
    )


def _arm_authentic(arm: Mapping[str, Any], expected_name: str) -> bool:
    stages = arm.get("raw_stages", {})
    if not isinstance(stages, Mapping) or set(stages) != set(STAGE_ORDER):
        return False
    present = {name for name, value in stages.items() if isinstance(value, Mapping)}
    route = arm.get("route")
    expected_stages = {"direct"} if route == "direct" else {"stage1", "stage2"}
    stage_row_hashes = [stages[name].get("row_hash") for name in present]
    tokens = arm.get("tokens", {})
    failure = arm.get("failure", {})
    return bool(
        arm.get("arm_name") == expected_name
        and present == expected_stages
        and all(_stage_authentic(stages[name], name) for name in present)
        and len(stage_row_hashes) == len(set(stage_row_hashes))
        and arm.get("stage1_passed_verbatim_to_stage2") is True
        and tokens.get("stage1_charged") is True
        and tokens.get("stage2_charged") is True
        and tokens.get("direct_charged") is True
        and tokens.get("total")
        == int(tokens.get("direct", 0))
        + int(tokens.get("stage1", 0))
        + int(tokens.get("stage2", 0))
        and arm.get("charged_cost") == cost_from_stages(stages)
        and arm.get("exact_results", {}).get("checker") == EXACT_CHECKER_NAME
        and arm.get("exact_results", {}).get("model_is_release_authority") is False
        and arm.get("unsafe_release") is False
        and isinstance(failure, Mapping)
        and set(failure) == set(FAILURE_CLASSES) | {"any"}
        and failure.get("stage1_answer_leakage") is False
        and arm.get("attempt_count") == 1
        and arm.get("retry_count") == 0
        and arm.get("row_hash") == row_hash(arm)
    )


def _unit_authentic(row: Mapping[str, Any], expected_id: Any) -> bool:
    try:
        source = base64.b64decode(str(row.get("source_bytes_b64", "")), validate=True)
        task = base64.b64decode(str(row.get("task_bytes_b64", "")), validate=True)
    except (TypeError, ValueError):
        return False
    arms = [arm for arm in row.get("arms", []) if isinstance(arm, Mapping)]
    return bool(
        row.get("unit_id") == expected_id
        and row.get("source_bytes_sha256") == sha256_bytes(source)
        and row.get("task_bytes_sha256") == sha256_bytes(task)
        and [arm.get("arm_name") for arm in arms] == list(ARM_ORDER)
        and all(_arm_authentic(arm, name) for arm, name in zip(arms, ARM_ORDER, strict=True))
        and row.get("row_hash") == row_hash(row)
    )


def _checkpoint_prefixes_ready(
    rows: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]]
) -> bool:
    if len(rows) != len(receipts):
        return False
    for index, receipt in enumerate(receipts, start=1):
        prefix = list(rows[:index])
        if not (
            receipt.get("completed_unit_count") == index
            and receipt.get("completed_unit_ids") == [row.get("unit_id") for row in prefix]
            and receipt.get("completed_unit_row_hashes") == [row.get("row_hash") for row in prefix]
            and receipt.get("prefix_hash") == sha256_json(prefix)
            and receipt.get("atomic_replace") is True
            and receipt.get("directory_fsync") is True
            and receipt.get("checkpoint_sha256") == sha256_file(receipt.get("absolute_path"))
        ):
            return False
    return True


def stream_reducer(payload: Mapping[str, Any], *, require_attack_rows: bool = True) -> JsonDict:
    """Recompute binary stream readiness only from independently checkable receipts."""

    frozen = payload.get("prompt_source_router_hashes", {})
    expected_ids = list(frozen.get("expected_unit_ids", [])) if isinstance(frozen, Mapping) else []
    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    raw = build_raw_stage_receipts(rows)
    exact_rows = build_exact_checker_receipts(rows)
    failures = build_failure_rows(rows)
    attack_rows = [row for row in payload.get("attack_rows", []) if isinstance(row, Mapping)]
    checks = {
        "launch_gate": payload.get("gate_check_summary", {}).get("rows", [{}])[0].get("passed")
        is True,
        "frozen_method": frozen.get("all_frozen_hashes_match") is True
        and all(frozen.get("checks", {}).values()),
        "model_identity": all(
            model_identity_checks(payload.get("model_spec_and_identity", {})).values()
        ),
        "complete_unit_order": bool(expected_ids)
        and len(rows) == len(expected_ids)
        and [row.get("unit_id") for row in rows] == expected_ids,
        "authentic_unit_rows": bool(rows)
        and len(rows) == len(expected_ids)
        and all(
            _unit_authentic(row, expected_id)
            for row, expected_id in zip(rows, expected_ids, strict=True)
        ),
        "raw_stage_receipts": payload.get("raw_stage_receipts") == raw and bool(raw),
        "exact_checker_receipts": payload.get("exact_checker_receipts") == exact_rows
        and len(exact_rows) == len(expected_ids) * len(ARM_ORDER),
        "failure_rows_retained": payload.get("failure_rows") == failures,
        "checkpoint_prefixes": _checkpoint_prefixes_ready(
            rows, payload.get("checkpoint_receipts", [])
        ),
        "gpu_lifecycle": all(
            process_lifecycle_checks(payload.get("gpu_process_receipts", {})).values()
        ),
        "attacks": (
            {row.get("attack_id") for row in attack_rows} == set(REQUIRED_ATTACK_IDS)
            and all(
                row.get("passed") is True and row.get("candidate_ready_score") == 0.0
                for row in attack_rows
            )
        )
        if require_attack_rows
        else True,
        "preconditions": payload.get("preconditions_checked", {}).get(
            "all_required_preconditions_available"
        )
        is True
        and payload.get("preconditions_checked", {}).get("repo_wide_suite_is_launch_gate") is False,
        "protected_files": payload.get("protected_files_unchanged", {}).get("all_unchanged")
        is True,
        "tests_recorded": bool(payload.get("tests_run"))
        and all(row.get("exit_code") == 0 for row in payload.get("tests_run", [])),
        "substrate": payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "oracle_declared": payload.get("verifier_is_oracle") is True,
    }
    return {
        "expected_unit_count": len(expected_ids),
        "completed_unit_count": len(rows),
        "arm_row_count": sum(len(row.get("arms", [])) for row in rows),
        "raw_stage_count": len(raw),
        "exact_checker_row_count": len(exact_rows),
        "failure_row_count": len(failures),
        "total_tokens": sum(
            int(arm.get("tokens", {}).get("total", 0) or 0)
            for row in rows
            for arm in row.get("arms", [])
        ),
        "latency_s": round(
            sum(
                float(arm.get("latency_s", 0.0) or 0.0)
                for row in rows
                for arm in row.get("arms", [])
            ),
            9,
        ),
        "charged_cost": round(
            sum(
                float(arm.get("charged_cost", 0.0) or 0.0)
                for row in rows
                for arm in row.get("arms", [])
            ),
            9,
        ),
        "checks": checks,
        "ready_score": 1.0 if all(checks.values()) else 0.0,
        "reducer": "conjunction of frozen hashes, complete rows, raw stages, exact checks, checkpoints, Qwen GPU lifecycle, attacks, and protection receipts",
    }


def build_attack_rows(base_payload: Mapping[str, Any]) -> list[JsonDict]:
    """Apply every required mutation and retain the reducer's zero score."""

    mutations = {
        "prompt_drift": lambda value: value["prompt_source_router_hashes"]["checks"].update(
            prompt_hashes=False
        ),
        "post_outcome_unit_loss": lambda value: value["per_unit_rows"].pop(),
        "stage_overwrite": lambda value: value["per_unit_rows"][0]["arms"][1]["raw_stages"][
            "stage2"
        ].update(
            raw_sha256=value["per_unit_rows"][0]["arms"][1]["raw_stages"]["stage1"]["raw_sha256"]
        ),
        "answer_leakage": lambda value: value["per_unit_rows"][0]["arms"][1]["failure"].update(
            stage1_answer_leakage=True, any=True
        ),
        "uncharged_stage1": lambda value: value["per_unit_rows"][0]["arms"][1]["tokens"].update(
            stage1_charged=False
        ),
        "family_label_substitution": lambda value: value["model_spec_and_identity"][
            "identity"
        ].update(repository_id="unsloth/gemma-4-31B-it-GGUF"),
        "legacy_model_substitution": lambda value: value["model_spec_and_identity"]["model_specs"][
            0
        ].update(repository_id="Qwen/Qwen3.5-0.8B"),
        "aggregate_only_output": lambda value: value.update(per_unit_rows=[]),
        "ready_score_with_missing_rows": lambda value: value["per_unit_rows"].pop(),
    }
    base_ready = stream_reducer(base_payload, require_attack_rows=False)["ready_score"]
    rows = []
    for attack_id in REQUIRED_ATTACK_IDS:
        candidate = deepcopy(base_payload)
        if base_ready == 1.0:
            mutations[attack_id](candidate)
        score = stream_reducer(candidate, require_attack_rows=False)["ready_score"]
        rows.append(
            {
                "attack_id": attack_id,
                "candidate_ready_score": score,
                "expected_ready_score": 0.0,
                "passed": score == 0.0,
                "reducer": "stream_reducer(require_attack_rows=False)",
            }
        )
    return rows


def _field_provenance() -> dict[str, JsonDict]:
    """Name the raw receipts and deterministic reducer for every required field."""

    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "raw_sources": [
                "Exp6587 frozen method bytes",
                "per_unit_rows raw stages",
                "exact checker rows",
                "checkpoint and GPU lifecycle receipts",
            ],
            "reducer": "stream_reducer"
            if field in {"qwen_cfr_rows_ready_score", "failure_rows", "attack_rows"}
            else "direct receipt or deterministic assembly",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _gate_summary(
    gate: Mapping[str, Any],
    frozen: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Name the first exact gate, drift, or resource failure and its value."""

    first_failure = None
    if gate.get("passed") is not True:
        first_failure = {
            "check": gate.get("field"),
            "expected": gate.get("expected_value"),
            "observed": gate.get("observed_value"),
        }
    elif frozen.get("all_frozen_hashes_match") is not True:
        name = next(
            (key for key, value in frozen.get("checks", {}).items() if value is not True),
            "frozen_method",
        )
        first_failure = {"check": name, "expected": True, "observed": False}
    elif preconditions.get("all_required_preconditions_available") is not True:
        names = list(preconditions.get("failed_preconditions", []))
        name = names[0] if names else "preconditions"
        first_failure = {
            "check": name,
            "expected": True,
            "observed": preconditions.get("checks", {}).get(name, False),
        }
    return {
        "rows": [dict(gate)],
        "all_launch_checks_passed": first_failure is None,
        "first_failure": first_failure,
    }


def build_report(
    *,
    run_date: str,
    gate_receipt: Mapping[str, Any],
    frozen_receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    per_unit_rows: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    gpu_receipts: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble one terminal stream and derive its binary completeness score."""

    payload: JsonDict = {
        "status": "assembling",
        "honest_verdict": "partial_qwen_cfr_rows_incomplete_without_benefit_claim",
        "verdict_class": "partial",
        "gate_check_summary": _gate_summary(gate_receipt, frozen_receipt, preconditions),
        "per_unit_rows": [dict(row) for row in per_unit_rows],
        "model_spec_and_identity": dict(model_identity),
        "prompt_source_router_hashes": dict(frozen_receipt),
        "raw_stage_receipts": build_raw_stage_receipts(per_unit_rows),
        "exact_checker_receipts": build_exact_checker_receipts(per_unit_rows),
        "checkpoint_receipts": [dict(row) for row in checkpoint_receipts],
        "gpu_process_receipts": dict(gpu_receipts),
        "failure_rows": build_failure_rows(per_unit_rows),
        "qwen_cfr_rows_ready_score": 0.0,
        "attack_rows": [],
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    payload["attack_rows"] = build_attack_rows(payload)
    reduction = stream_reducer(payload)
    ready = reduction["ready_score"]
    payload["qwen_cfr_rows_ready_score"] = ready
    payload["stream_recomputation"] = reduction
    payload["planning_date"] = run_date
    payload["task_id"] = TASK_ID
    if ready == 1.0:
        payload["status"] = "complete"
        payload["honest_verdict"] = (
            "complete: every frozen Qwen CFR unit, arm, raw stage, exact check, checkpoint, "
            "cost, failure, model, and GPU receipt is complete; no CFR benefit claim is made"
        )
        payload["verdict_class"] = None
    elif protected.get("all_unchanged") is not True:
        payload["status"] = "disqualified"
        payload["honest_verdict"] = "disqualified_protected_file_changed_without_benefit_claim"
        payload["verdict_class"] = "disqualified"
    elif not per_unit_rows and payload["gate_check_summary"]["first_failure"] is not None:
        payload["status"] = "blocked"
        name = str(payload["gate_check_summary"]["first_failure"]["check"])
        payload["honest_verdict"] = f"blocked_{name}_without_benefit_claim"
        payload["verdict_class"] = "blocked"
    else:
        payload["status"] = "partial"
        payload["honest_verdict"] = "partial_qwen_cfr_rows_incomplete_without_benefit_claim"
        payload["verdict_class"] = "partial"
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def build_blocked_report(
    *,
    run_date: str,
    gate_receipt: Mapping[str, Any],
    frozen_receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Close a named precondition block without starting model inference."""

    return build_report(
        run_date=run_date,
        gate_receipt=gate_receipt,
        frozen_receipt=frozen_receipt,
        model_identity=model_identity,
        per_unit_rows=[],
        checkpoint_receipts=[],
        gpu_receipts={},
        preconditions=preconditions,
        protected=protected,
        duration_s=duration_s,
        tests_run=tests_run,
    )


def validate_report(payload: Mapping[str, Any]) -> list[str]:
    """Validate terminal schema, readiness, verdict, and checksum without trust."""

    errors = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class_invalid")
    if payload.get("model_spec_and_identity", {}).get("model_specs") != MODEL_SPECS:
        errors.append("model_specs_mismatch")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    reduction = stream_reducer(payload)
    if payload.get("qwen_cfr_rows_ready_score") != reduction["ready_score"]:
        errors.append("ready_score_mismatch")
    if payload.get("verdict_class") is None and reduction["ready_score"] != 1.0:
        errors.append("null_verdict_without_ready_stream")
    if payload.get("verdict_class") == "blocked":
        if payload.get("per_unit_rows"):
            errors.append("blocked_report_started_rows")
        if payload.get("gate_check_summary", {}).get("first_failure") is None:
            errors.append("blocked_report_missing_gate_value")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def atomic_write_report(path: str | Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate, sync, replace, and directory-sync one terminal artifact."""

    errors = validate_report(payload)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".exp6590-final-", delete=False
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


def _utc_now() -> str:  # pragma: no cover - live receipt.
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_command(
    command: Sequence[str], repo_root: Path, timeout_s: float
) -> JsonDict:  # pragma: no cover
    start = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "exit_code": result.returncode,
            "duration_s": round(time.monotonic() - start, 6),
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
            "stdout_sha256": sha256_text(result.stdout),
            "stderr_sha256": sha256_text(result.stderr),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "exit_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
            "duration_s": round(time.monotonic() - start, 6),
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(str(exc)),
        }


def _checkpoint_tests(repo_root: Path) -> list[JsonDict]:  # pragma: no cover
    commands = (
        (FOCUSED_TEST_COMMAND, 180.0),
        (COVERAGE_RUN_COMMAND, 180.0),
        (COVERAGE_REPORT_COMMAND, 60.0),
        (RUFF_CHECK_COMMAND, 60.0),
        (RUFF_FORMAT_COMMAND, 60.0),
        (SPEC_COVERAGE_COMMAND, 60.0),
    )
    return [_run_command(text.split(), repo_root, timeout) for text, timeout in commands]


def _host_resources(repo_root: Path) -> JsonDict:  # pragma: no cover
    cpu_model = "unknown"
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0])
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(repo_root)
    return {
        "cpu": {"count": os.cpu_count(), "model": cpu_model, "architecture": platform.machine()},
        "ram": {"total_kib": memory.get("MemTotal"), "available_kib": memory.get("MemAvailable")},
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
    }


def _protected_hashes(repo_root: Path) -> dict[str, str]:  # pragma: no cover
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(
    before: Mapping[str, str], after: Mapping[str, str]
) -> JsonDict:  # pragma: no cover
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {"all_unchanged": bool(rows) and all(row["unchanged"] for row in rows), "rows": rows}


def _resolve_model_identity() -> JsonDict:  # pragma: no cover
    identity = shard._resolve_metadata_receipt()  # noqa: SLF001
    pair = cached_sota_pair() or []
    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    build = runtime_helpers._llama_cpp_build_receipt(server)  # noqa: SLF001
    return {
        "model_specs": deepcopy(MODEL_SPECS),
        "identity": identity,
        "cached_sota_pair": pair,
        "llama_cpp_build": build,
        "embedded_tokenizer_used": True,
        "auto_tokenizer_used": False,
        "download_performed": False,
    }


def _model_path(model_identity: Mapping[str, Any]) -> str:  # pragma: no cover
    identity = model_identity.get("identity", {})
    return str(identity.get("selected_blob_path") or identity.get("cache_path") or "")


def _collect_preconditions(
    repo_root: Path,
    gate: Mapping[str, Any],
    frozen: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, Path, JsonDict]:  # pragma: no cover
    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    model_path = _model_path(model_identity)
    initial = runtime_helpers._live_gpu_sample(  # noqa: SLF001
        repository_id=QWEN_REPOSITORY_ID,
        worker_pid=0,
        stage="preconditions",
        sample_index=0,
        selected_gpu=0,
        model_paths=[model_path],
    )
    selection = runtime_helpers.choose_idle_gpu(initial)
    owned_pids = runtime_helpers._task_owned_pids([model_path])  # noqa: SLF001
    checks = {
        "upstream_gate": gate.get("passed") is True,
        "frozen_method": frozen.get("all_frozen_hashes_match") is True,
        "model_identity": all(model_identity_checks(model_identity).values()),
        "llama_cpp_cuda": model_identity_checks(model_identity)["llama_cpp_cuda"],
        "idle_rtx_3090": selection.get("eligible") is True,
        "owned_process_absent": not owned_pids,
        "focused_verification": bool(tests_run)
        and all(row.get("exit_code") == 0 for row in tests_run),
        "atomic_output": os.access((repo_root / RESULT_RELATIVE_PATH).parent, os.W_OK),
    }
    launch = load_json(repo_root / LAUNCH_RELATIVE_PATH)
    budget = next(
        (
            row
            for row in launch.get("execution_budget_contract", [])
            if row.get("task_id") == TASK_ID
        ),
        {},
    )
    return (
        {
            "all_required_preconditions_available": all(checks.values()),
            "checks": checks,
            "failed_preconditions": [name for name, passed in checks.items() if not passed],
            "model_process_started": False,
            "repo_wide_suite_is_launch_gate": False,
            "protected_file_hashes_before": {},
            "upstream_gate": dict(gate),
            "method_and_source_hashes": dict(frozen),
            "exact_registry_hash": frozen.get("exact_registry_hash"),
            "model_identity_checks": model_identity_checks(model_identity),
            "llama_cpp_build": model_identity.get("llama_cpp_build", {}),
            "initial_gpu_state": initial,
            "gpu_selection": selection,
            "selected_gpu": selection.get("selected_gpu"),
            "task_owned_pids_before": owned_pids,
            "seed_schedule": load_json(repo_root / METHOD_RELATIVE_PATH)
            .get("arm_seed_budget_contract", {})
            .get("seed_schedule", []),
            "budgets": budget,
            "resource_receipts": _host_resources(repo_root),
            "embedded_tokenizer_required": True,
            "auto_tokenizer_allowed": False,
            "external_download_allowed": False,
        },
        server,
        initial,
    )


def _server_command(server: Path, model_path: str, port: int) -> list[str]:  # pragma: no cover
    return [
        str(server),
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--n-gpu-layers",
        "all",
        "--device",
        "CUDA0",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-ui",
        "--log-verbosity",
        "4",
    ]


def _free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _port_open(port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.settimeout(0.25)
        return handle.connect_ex(("127.0.0.1", port)) == 0


def _render_request(
    prompt: str, unit: Mapping[str, Any], stage1_raw: bytes | None = None
) -> bytes:  # pragma: no cover
    blocks = [
        prompt,
        "[SOURCE]",
        str(unit.get("exact_source_bytes", "")),
        "[TASK]",
        str(unit.get("exact_task_bytes", "")),
    ]
    if stage1_raw is not None:
        blocks.extend(["[STAGE1_RAW]", stage1_raw.decode("utf-8", "replace")])
    return "\n\n".join(blocks).encode("utf-8")


def _request_stage(
    *,
    port: int,
    unit: Mapping[str, Any],
    stage: str,
    prompt_row: Mapping[str, Any],
    stage1_raw: bytes | None,
    seed: int,
    max_tokens: int,
    deadline: float,
) -> JsonDict:  # pragma: no cover
    request_bytes = _render_request(str(prompt_row.get("text", "")), unit, stage1_raw)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        failure = empty_failure_flags()
        failure["task_deadline_exhausted"] = True
        return make_stage_receipt(
            stage=stage,
            prompt_sha256=str(prompt_row.get("sha256")),
            request_sha256=sha256_bytes(request_bytes),
            raw_bytes=b"",
            prompt_tokens=0,
            completion_tokens=0,
            latency_s=0.0,
            stop_reason="task_deadline_exhausted",
            request_status=124,
            recorded_monotonic_ns=time.monotonic_ns(),
            failure_flags=failure,
        )
    payload = {
        "model": "local-gguf",
        "messages": [{"role": "user", "content": request_bytes.decode("utf-8")}],
        "seed": seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": min(max_tokens, LAUNCH_MAX_OUTPUT_TOKENS),
        "stop": list(method_builder.STOP_RULES),
        "stream": False,
    }
    started = time.monotonic()
    raw_api = b""
    raw_response = b""
    prompt_tokens = 0
    completion_tokens = 0
    stop_reason = "request_failure"
    status = 1
    failure = empty_failure_flags()
    try:
        status, raw_api = shard._http_bytes(  # noqa: SLF001
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload,
            timeout_s=min(PER_GENERATION_TIMEOUT_S, max(0.1, remaining)),
        )
        raw_response, prompt_tokens, completion_tokens, stop_reason, malformed = (
            shard._parse_api_response(raw_api)  # noqa: SLF001
        )
        failure["malformed_output"] = malformed
    except (OSError, TimeoutError, urllib.error.URLError):
        status = 124
        stop_reason = "generation_timeout"
        failure["timeout"] = True
    if not raw_response and not failure["timeout"]:
        failure["malformed_output"] = True
    receipt = make_stage_receipt(
        stage=stage,
        prompt_sha256=str(prompt_row.get("sha256")),
        request_sha256=sha256_bytes(request_bytes),
        raw_bytes=raw_response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_s=time.monotonic() - started,
        stop_reason=stop_reason,
        request_status=status,
        recorded_monotonic_ns=time.monotonic_ns(),
        failure_flags=failure,
    )
    receipt["raw_api_response_sha256"] = sha256_bytes(raw_api)
    receipt["stage1_raw_sha256_in_request"] = (
        sha256_bytes(stage1_raw) if stage1_raw is not None else None
    )
    receipt["row_hash"] = row_hash(receipt)
    return receipt


def _failed_stage(
    stage: str, prompt_row: Mapping[str, Any], reason: str
) -> JsonDict:  # pragma: no cover
    failure = empty_failure_flags()
    failure[reason] = True
    return make_stage_receipt(
        stage=stage,
        prompt_sha256=str(prompt_row.get("sha256")),
        request_sha256=sha256_text("process unavailable"),
        raw_bytes=b"",
        prompt_tokens=0,
        completion_tokens=0,
        latency_s=0.0,
        stop_reason=reason,
        request_status=1,
        recorded_monotonic_ns=time.monotonic_ns(),
        failure_flags=failure,
    )


def _run_live_stream(
    *,
    repo_root: Path,
    method: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    server: Path,
    selected_gpu: int,
    task_deadline: float,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    model_path = _model_path(model_identity)
    port = _free_port()
    command = _server_command(server, model_path, port)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    checkpoint_dir = repo_root / CHECKPOINT_RELATIVE_PATH / f"run-{time.monotonic_ns()}"
    before = runtime_helpers._live_gpu_sample(  # noqa: SLF001
        repository_id=QWEN_REPOSITORY_ID,
        worker_pid=0,
        stage="before",
        sample_index=0,
        selected_gpu=selected_gpu,
        model_paths=[model_path],
    )
    samples = [before]
    baseline = int(before.get("device", {}).get("memory_used_mb", 0) or 0)
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    process: subprocess.Popen[bytes] | None = None
    identity: JsonDict = {}
    healthy = False
    http_status = 0
    offloaded = 0
    shutdown_requested = False
    forced_kill = False
    error = ""
    start_ns = time.monotonic_ns()
    start_utc = _utc_now()
    with tempfile.TemporaryDirectory(prefix="exp6590-llama-") as temporary:
        stdout_path = Path(temporary) / "stdout.bin"
        stderr_path = Path(temporary) / "stderr.bin"
        try:
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
            identity = runtime_helpers._wait_for_process_identity(process.pid, command)  # noqa: SLF001
            sample_index = 1
            load_deadline = min(task_deadline, time.monotonic() + LOAD_TIMEOUT_S)
            while time.monotonic() < load_deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited during load with {process.returncode}")
                samples.append(
                    runtime_helpers._live_gpu_sample(  # noqa: SLF001
                        repository_id=QWEN_REPOSITORY_ID,
                        worker_pid=process.pid,
                        stage="during",
                        sample_index=sample_index,
                        selected_gpu=selected_gpu,
                        model_paths=[model_path],
                    )
                )
                sample_index += 1
                try:
                    status, raw_health = shard._http_bytes(  # noqa: SLF001
                        f"http://127.0.0.1:{port}/health", timeout_s=0.5
                    )
                    health = json.loads(raw_health.decode("utf-8"))
                    healthy = status == 200 and health.get("status") == "ok"
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.URLError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    healthy = False
                if healthy:
                    http_status = 200
                    break
                time.sleep(TELEMETRY_INTERVAL_S)
            if not healthy:
                raise TimeoutError("llama-server did not become healthy before load timeout")
            identity = runtime_helpers.select_process_identity_receipt(
                identity,
                runtime_helpers._proc_identity(process.pid),
                command,  # noqa: SLF001
            )
            offloaded = shard._offloaded_layers(stderr_path.read_bytes())  # noqa: SLF001
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        prompts = method["prompt_stage_contract"]["prompts"]
        registry_hash = method["source_binding_and_exact_authority_contract"][
            "exact_obligation_registry"
        ]["registry_sha256"]
        seeds = method["arm_seed_budget_contract"]["seed_schedule"]
        routing = {
            row["unit_id"]: row["observed_route"]
            for row in method["router_contract"]["routing_rows"]
        }
        sample_index = len(samples)
        for unit, seed_row in zip(method["source_unit_manifest"]["units"], seeds, strict=True):
            seed = int(seed_row["seed"])
            alive = healthy and process is not None and process.poll() is None
            if alive:
                samples.append(
                    runtime_helpers._live_gpu_sample(  # noqa: SLF001
                        repository_id=QWEN_REPOSITORY_ID,
                        worker_pid=process.pid,
                        stage="during",
                        sample_index=sample_index,
                        selected_gpu=selected_gpu,
                        model_paths=[model_path],
                    )
                )
                sample_index += 1
                direct_stage = _request_stage(
                    port=port,
                    unit=unit,
                    stage="direct",
                    prompt_row=prompts["direct"],
                    stage1_raw=None,
                    seed=seed,
                    max_tokens=method_builder.STAGE_TOKEN_LIMITS["direct"],
                    deadline=task_deadline,
                )
            else:
                direct_stage = _failed_stage("direct", prompts["direct"], "process_failure")
            direct_arm = build_arm_row(
                unit=unit,
                arm_name="direct",
                route="direct",
                seed=seed,
                stage_receipts={"direct": direct_stage},
                registry_hash=registry_hash,
            )

            alive = healthy and process is not None and process.poll() is None
            stage1 = (
                _request_stage(
                    port=port,
                    unit=unit,
                    stage="stage1",
                    prompt_row=prompts["stage1"],
                    stage1_raw=None,
                    seed=seed,
                    max_tokens=method_builder.STAGE_TOKEN_LIMITS["stage1"],
                    deadline=task_deadline,
                )
                if alive
                else _failed_stage("stage1", prompts["stage1"], "process_failure")
            )
            stage1_raw = _decode_stage(stage1) or b""
            alive = healthy and process is not None and process.poll() is None
            stage2 = (
                _request_stage(
                    port=port,
                    unit=unit,
                    stage="stage2",
                    prompt_row=prompts["stage2"],
                    stage1_raw=stage1_raw,
                    seed=seed,
                    max_tokens=method_builder.STAGE_TOKEN_LIMITS["stage2"],
                    deadline=task_deadline,
                )
                if alive
                else _failed_stage("stage2", prompts["stage2"], "process_failure")
            )
            always_arm = build_arm_row(
                unit=unit,
                arm_name="always_on_cfr",
                route="cfr",
                seed=seed,
                stage_receipts={"stage1": stage1, "stage2": stage2},
                registry_hash=registry_hash,
            )

            route = routing[unit["unit_id"]]
            alive = healthy and process is not None and process.poll() is None
            if route == "direct":
                routed_stages = {
                    "direct": _request_stage(
                        port=port,
                        unit=unit,
                        stage="direct",
                        prompt_row=prompts["direct"],
                        stage1_raw=None,
                        seed=seed,
                        max_tokens=method_builder.STAGE_TOKEN_LIMITS["direct"],
                        deadline=task_deadline,
                    )
                    if alive
                    else _failed_stage("direct", prompts["direct"], "process_failure")
                }
            else:
                routed_stage1 = (
                    _request_stage(
                        port=port,
                        unit=unit,
                        stage="stage1",
                        prompt_row=prompts["stage1"],
                        stage1_raw=None,
                        seed=seed,
                        max_tokens=method_builder.STAGE_TOKEN_LIMITS["stage1"],
                        deadline=task_deadline,
                    )
                    if alive
                    else _failed_stage("stage1", prompts["stage1"], "process_failure")
                )
                routed_raw = _decode_stage(routed_stage1) or b""
                alive = healthy and process is not None and process.poll() is None
                routed_stage2 = (
                    _request_stage(
                        port=port,
                        unit=unit,
                        stage="stage2",
                        prompt_row=prompts["stage2"],
                        stage1_raw=routed_raw,
                        seed=seed,
                        max_tokens=method_builder.STAGE_TOKEN_LIMITS["stage2"],
                        deadline=task_deadline,
                    )
                    if alive
                    else _failed_stage("stage2", prompts["stage2"], "process_failure")
                )
                routed_stages = {"stage1": routed_stage1, "stage2": routed_stage2}
            routed_arm = build_arm_row(
                unit=unit,
                arm_name="routed_cfr",
                route=route,
                seed=seed,
                stage_receipts=routed_stages,
                registry_hash=registry_hash,
            )
            process_binding = {
                "pid": 0 if process is None else process.pid,
                "repository_id": QWEN_REPOSITORY_ID,
                "selected_gpu": selected_gpu,
                "command_sha256": sha256_json(command),
            }
            rows.append(
                finalize_unit_row(unit, [direct_arm, always_arm, routed_arm], process_binding)
            )
            checkpoints.append(write_unit_checkpoint(checkpoint_dir, rows))

        exit_code: int | None = 127
        if process is not None:
            if process.poll() is None:
                shutdown_requested = True
                process.send_signal(signal.SIGTERM)
                try:
                    exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    forced_kill = True
                    process.kill()
                    exit_code = process.wait(timeout=5)
            else:
                exit_code = process.returncode
        end_ns = time.monotonic_ns()
        stdout_bytes = stdout_path.read_bytes() if stdout_path.is_file() else b""
        stderr_bytes = stderr_path.read_bytes() if stderr_path.is_file() else b""
        offloaded = max(offloaded, shard._offloaded_layers(stderr_bytes))  # noqa: SLF001

    after: JsonDict = {}
    recovery_start = time.monotonic()
    recovery_complete = False
    worker_pid = 0 if process is None else process.pid
    while time.monotonic() - recovery_start <= RECOVERY_TIMEOUT_S:
        after = runtime_helpers._live_gpu_sample(  # noqa: SLF001
            repository_id=QWEN_REPOSITORY_ID,
            worker_pid=worker_pid,
            stage="after",
            sample_index=len(samples),
            selected_gpu=selected_gpu,
            model_paths=[model_path],
        )
        samples.append(after)
        recovered = int(after.get("device", {}).get("memory_used_mb", 0) or 0)
        pids = {int(row.get("pid", 0) or 0) for row in after.get("compute_processes", [])}
        recovery_complete = (
            worker_pid > 1
            and not Path(f"/proc/{worker_pid}").exists()
            and worker_pid not in pids
            and not _port_open(port)
            and abs(recovered - baseline) <= RECOVERY_TOLERANCE_MB
            and not runtime_helpers._task_owned_pids([model_path])  # noqa: SLF001
        )
        if recovery_complete:
            break
        time.sleep(TELEMETRY_INTERVAL_S)
    recovered = int(after.get("device", {}).get("memory_used_mb", 0) or 0)
    pids = {int(row.get("pid", 0) or 0) for row in after.get("compute_processes", [])}
    process_receipt = {
        "pid": worker_pid,
        "parent_pid": identity.get("parent_pid"),
        "fresh_process": True,
        "owned_child": identity.get("parent_pid") == os.getpid(),
        "command": command,
        "command_sha256": sha256_json(command),
        "selected_blob_path": model_path,
        "selected_gpu": selected_gpu,
        "cuda_visible_devices": str(selected_gpu),
        "offloaded_layers": offloaded,
        "server_healthy": healthy,
        "http_status": http_status,
        "gpu_samples": samples,
        "started_utc": start_utc,
        "started_monotonic_ns": start_ns,
        "ended_monotonic_ns": end_ns,
        "shutdown_requested": shutdown_requested,
        "normal_shutdown": shutdown_requested
        and not forced_kill
        and exit_code in {0, -signal.SIGTERM},
        "exit_code": exit_code,
        "worker_alive_after_exit": worker_pid > 1 and Path(f"/proc/{worker_pid}").exists(),
        "resident_model_families": [QWEN_REPOSITORY_ID],
        "signals_sent_to_unrelated_pids": [],
        "stdout_sha256": sha256_bytes(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes),
        "stderr_tail": stderr_bytes.decode("utf-8", "replace")[-8000:],
        "evidence_mode": "measured",
        "error": error,
    }
    unload = {
        "worker_pid": worker_pid,
        "worker_absent_from_proc": worker_pid > 1 and not Path(f"/proc/{worker_pid}").exists(),
        "worker_absent_from_nvidia_smi": worker_pid > 1 and worker_pid not in pids,
        "port_closed": not _port_open(port),
        "baseline_memory_used_mb": baseline,
        "recovered_memory_used_mb": recovered,
        "memory_delta_from_baseline_mb": recovered - baseline,
        "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
        "no_task_worker_remains": not runtime_helpers._task_owned_pids([model_path]),  # noqa: SLF001
        "recovery_bounded": True,
        "recovery_duration_s": round(time.monotonic() - recovery_start, 6),
        "recovery_complete": recovery_complete,
        "signals_sent_to_unrelated_pids": [],
    }
    return rows, checkpoints, {"process": process_receipt, "unload": unload}


def run_experiment(repo_root: Path, run_date: str) -> JsonDict:  # pragma: no cover
    """Run focused gates, one fresh Qwen stream, cleanup, and atomic output."""

    start = time.monotonic()
    protected_before = _protected_hashes(repo_root)
    gate = build_gate_receipt(repo_root)
    method = load_json(repo_root / METHOD_RELATIVE_PATH)
    frozen = build_frozen_hash_receipt(repo_root, method)
    model_identity = _resolve_model_identity()
    tests_run = _checkpoint_tests(repo_root)
    preconditions, server, _initial = _collect_preconditions(
        repo_root, gate, frozen, model_identity, tests_run
    )
    preconditions["protected_file_hashes_before"] = protected_before
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    gpu_receipts: JsonDict = {}
    if preconditions["all_required_preconditions_available"]:
        preconditions["model_process_started"] = True
        rows, checkpoints, gpu_receipts = _run_live_stream(
            repo_root=repo_root,
            method=method,
            model_identity=model_identity,
            server=server,
            selected_gpu=int(preconditions["selected_gpu"]),
            task_deadline=start + TASK_TIMEOUT_S,
        )
    protected_after = _protected_hashes(repo_root)
    protected = _protected_receipt(protected_before, protected_after)
    preconditions["protected_file_hashes_after"] = protected_after
    if not preconditions["all_required_preconditions_available"]:
        artifact = build_blocked_report(
            run_date=run_date,
            gate_receipt=gate,
            frozen_receipt=frozen,
            model_identity=model_identity,
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
        )
    else:
        artifact = build_report(
            run_date=run_date,
            gate_receipt=gate,
            frozen_receipt=frozen,
            model_identity=model_identity,
            per_unit_rows=rows,
            checkpoint_receipts=checkpoints,
            gpu_receipts=gpu_receipts,
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
        )
    atomic_write_report(repo_root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the immutable Qwen CFR stream artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (REPO_ROOT / RESULT_RELATIVE_PATH)
    if args.validate:
        errors = validate_report(load_json(output))
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 1 if errors else 0
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "qwen_cfr_rows_ready_score": artifact["qwen_cfr_rows_ready_score"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
