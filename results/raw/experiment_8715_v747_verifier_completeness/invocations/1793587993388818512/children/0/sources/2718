"""Qualify the repaired source-grounding runtime with one Qwen call.

The run preserves model, CUDA, memory, output, and teardown evidence. It also
freezes the next comparison schedule without measuring verifier value.

Spec refs: REQ-VERIFY-7153 and SCENARIO-VERIFY-7153-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib import request

from carnot import experiment_7150_v628_grounding_preflight as prior
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    resolve_native_llama_server,
)
from carnot.experiment_7139_v627_symbolic_grounding_ab import (
    OUTPUT_TOKEN_LIMIT,
    _prompt_for as v627_prompt_for,
    artifact_checksum,
    canonical_json,
    gate_row,
    sha256_file,
    sha256_text,
)
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    supervisor_contract,
)
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260909"
RANDOM_SEED = 7_153_202_609_09
RESULT_PATH = Path("results/experiment_7153_v629_grounding_runtime.json")
FIXTURE_PATH = prior.FIXTURE_PATH
RAW_DIR = Path("results/raw/experiment_7153_v629_grounding_runtime")
INFERENCE_SUBSTRATE = "live_llm_inference: fixed source-grounding runtime canary"
EXECUTION_VENUE = "host"
QWEN_MODEL_ID = prior.QWEN_MODEL_ID
PREFERRED_QUANT = prior.PREFERRED_QUANT
FROZEN_FIXTURE_IDS = prior.FROZEN_FIXTURE_IDS
SOURCE_FAMILIES = prior.SOURCE_FAMILIES
CLASS_LABELS = prior.CLASS_LABELS
CANARY_PROMPT = "Reply with exactly RUNTIME_OK."
CANARY_EXPECTED_OUTPUT = "RUNTIME_OK"

CALL_PLAN = (
    ("direct", "direct", (1,)),
    ("self_check", "self_verification", (1, 2)),
    ("relational_sql", "relational_sql", (1, 2)),
    ("dual_side", "relational_sql", (1, 2)),
)
MODEL_SPECS: list[JsonDict] = deepcopy(prior.MODEL_SPECS)

READINESS_CHECKS = (
    "run_date",
    "output_paths",
    "fixture_contract",
    "typed_blinding",
    "frozen_schedule",
    "class_strata",
    "cached_qwen_q4",
    "embedded_chat_template",
    "native_cuda_linkage",
    "gpu_available",
    "real_qwen_canary",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "MODEL_SPECS",
    "model_identity_rows",
    "binary_linkage_rows",
    "backend_rows",
    "gpu_rows",
    "model_load_receipts",
    "canary_prompt_rows",
    "canary_raw_output_rows",
    "canary_token_rows",
    "runtime_evidence_rows",
    "blinding_rule_rows",
    "blinding_mutation_rows",
    "schedule_rows",
    "source_family_rows",
    "class_stratum_rows",
    "frozen_fixture_ids",
    "frozen_schedule_hash",
    "label_exposure_count",
    "grounding_runtime_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "status": "A durable state prevents an interrupted run from appearing complete.",
    "field_principles": "A principle for each field makes omitted evidence visible.",
    "preconditions_checked": "Exact gate rows show which resource was checked before startup.",
    "run_date": "The fixed date binds the receipt to the requested execution window.",
    "inference_substrate": "The substrate states that one local generation qualified the runtime.",
    "inference_substrate_class": "The class separates full generation from a blocked no-run.",
    "execution_venue": "The host venue prevents an unsupported remote execution claim.",
    "duration_s": "Wall time exposes interruption and implausibly short model work.",
    "source_artifact_hashes": "Hashes bind the result to its fixture, prior failure, code, tests, and spec.",
    "rows": "Per-fixture rows prevent an aggregate-only schedule claim.",
    "MODEL_SPECS": "The local-only declaration prevents model or quantization substitution.",
    "model_identity_rows": "File and template identities prove which model bytes were used.",
    "binary_linkage_rows": "Dynamic linkage supplies static CUDA capability evidence.",
    "backend_rows": "Backend rows retain the binary path without trusting its banner for CUDA.",
    "gpu_rows": "Snapshots show availability, task-owned memory, and release after teardown.",
    "model_load_receipts": "The load receipt preserves command, log, process, timing, and cleanup facts.",
    "canary_prompt_rows": "Exact request hashes make the bounded generation repeatable.",
    "canary_raw_output_rows": "Raw and parsed output prevent convenient reconstruction.",
    "canary_token_rows": "Positive token counts and latency prove that generation occurred.",
    "runtime_evidence_rows": "One joined row proves that runtime facts came from the same process.",
    "blinding_rule_rows": "Typed rules make the sealed-label boundary inspectable.",
    "blinding_mutation_rows": "Positive and negative controls prove that each rule bites.",
    "schedule_rows": "The full schedule freezes every future model-call opportunity.",
    "source_family_rows": "Family counts prove source balance before sealed labels open.",
    "class_stratum_rows": "Post-freeze label counts confirm each balanced source-class cell.",
    "frozen_fixture_ids": "Exact ordered IDs prevent data-dependent resampling.",
    "frozen_schedule_hash": "One digest binds prompts, limits, passes, arms, and call IDs.",
    "label_exposure_count": "Zero means model-visible schedule data contains no typed outcome.",
    "grounding_runtime_ready_score": "One reports execution readiness, not verifier value.",
    "random_seed": "A fixed seed makes the canary request repeatable.",
    "reproducibility_checksum": "A canonical digest detects later receipt mutation.",
    "gate_check_summary": "The first failed comparison keeps exact diagnostic evidence.",
    "verifier_is_oracle": "False prevents runtime readiness from becoming a correctness oracle.",
    "verdict_class": "A closed class separates readiness from blocked execution.",
    "honest_verdict": "The prefix states runtime readiness without a verifier value claim.",
}

HIDDEN_OUTCOME_FIELDS = frozenset({"hidden_outcome", "outcome"})

resolve_model_specs = prior.resolve_model_specs
model_spec_errors = prior.model_spec_errors
model_identity_receipts = prior.model_identity_receipts
binary_runtime_receipts = prior.binary_runtime_receipts
cuda_linkage_errors = prior.cuda_linkage_errors
cuda_offload_receipt = prior.cuda_offload_receipt
write_artifact = prior.write_artifact
build_source_family_rows = prior.build_source_family_rows
build_class_stratum_rows = prior.build_class_stratum_rows
class_stratum_errors = prior.class_stratum_errors


def base_artifact(run_date: str) -> JsonDict:
    """Create every final field without checking an external resource."""

    artifact: JsonDict = {
        "status": "running",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_identity_rows": [],
        "binary_linkage_rows": [],
        "backend_rows": [],
        "gpu_rows": [],
        "model_load_receipts": [],
        "canary_prompt_rows": [],
        "canary_raw_output_rows": [],
        "canary_token_rows": [],
        "runtime_evidence_rows": [],
        "blinding_rule_rows": [],
        "blinding_mutation_rows": [],
        "schedule_rows": [],
        "source_family_rows": [],
        "class_stratum_rows": [],
        "frozen_fixture_ids": [],
        "frozen_schedule_hash": "",
        "label_exposure_count": 0,
        "grounding_runtime_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "experiment_complete",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_grounding_runtime",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def initialize_artifact(path: Path | str, run_date: str) -> JsonDict:
    """Write the full artifact shape before the first prerequisite check."""

    artifact = base_artifact(run_date)
    write_artifact(path, artifact)
    return artifact


def finish_blocked(
    artifact: Mapping[str, Any],
    path: Path | str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Finish once as blocked while preserving all evidence collected so far."""

    blocked = deepcopy(dict(artifact))
    copied_checks = [deepcopy(dict(row)) for row in checks]
    summary = prior._gate_summary(copied_checks)
    failed = str(summary.get("failed_check") or "unknown_precondition")
    blocked.update(
        {
            "status": "blocked",
            "preconditions_checked": copied_checks,
            "inference_substrate_class": "blocked_no_run",
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "grounding_runtime_ready_score": 0,
            "gate_check_summary": summary,
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failed}",
        }
    )
    blocked["reproducibility_checksum"] = artifact_checksum(blocked)
    write_artifact(path, blocked)
    return blocked


def typed_blinding_errors(value: Any) -> list[str]:
    """Extend the tested typed rules to explicit hidden outcome fields."""

    errors = list(prior.typed_blinding_errors(value))

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key in HIDDEN_OUTCOME_FIELDS:
                    errors.append(f"hidden_outcome_field:{key}@{child_path}")
                walk(child, child_path)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "")
    return list(dict.fromkeys(errors))


def blinding_rule_rows() -> list[JsonDict]:
    """Describe the inherited typed boundaries and the hidden-outcome boundary."""

    return [
        *deepcopy(prior.blinding_rule_rows()),
        {
            "rule_id": "hidden_outcome_fields",
            "typed_target": "mapping_key",
            "forbidden": sorted(HIDDEN_OUTCOME_FIELDS),
        },
    ]


def blinding_mutation_rows() -> list[JsonDict]:
    """Run one allowed control and every required typed negative control."""

    base = {
        "fixture_id": "mutation-unit",
        "source_text": "The bottle had a paper label from Label House.",
        "response_text": "The bottle had a label.",
        "metadata": {"publisher": "Label House"},
    }
    cases = [
        ("ordinary_label_prose", "allowed", base),
        ("hidden_truth_label", "rejected", dict(base, truth_label="clean")),
        ("hidden_response_label", "rejected", dict(base, response_label="hallucinated")),
        ("hidden_outcome", "rejected", dict(base, hidden_outcome="clean")),
        ("scorer_metadata", "rejected", dict(base, metadata={"scorer_outcome": "clean"})),
        ("label_filename", "rejected", dict(base, input_path="/private/response_labels.jsonl")),
        ("explicit_outcome_injection", "rejected", dict(base, source_text="[[outcome=clean]]")),
    ]
    rows = []
    for mutation_id, expected, item in cases:
        found = typed_blinding_errors(item)
        observed = "rejected" if found else "allowed"
        rows.append(
            {
                "mutation_id": mutation_id,
                "expected": expected,
                "observed": observed,
                "errors": found,
                "passed": observed == expected,
            }
        )
    return rows


def build_schedule(
    model_rows: Sequence[Mapping[str, Any]], fixture_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze 24 label-blind rows and 168 matched call opportunities."""

    models = {str(row.get("fixture_id")): row for row in model_rows}
    fixtures = {str(row.get("fixture_id")): row for row in fixture_rows}
    missing = [
        fixture_id
        for fixture_id in FROZEN_FIXTURE_IDS
        if fixture_id not in models or fixture_id not in fixtures
    ]
    if missing:
        raise ValueError(f"frozen fixture IDs missing: {missing}")
    schedule = []
    for fixture_id in FROZEN_FIXTURE_IDS:
        model = models[fixture_id]
        fixture = fixtures[fixture_id]
        opportunities = []
        for arm, prompt_arm, passes in CALL_PLAN:
            calls = []
            for pass_index in passes:
                prompt = v627_prompt_for(model, prompt_arm, pass_index)
                calls.append(
                    {
                        "call_id": f"{QWEN_MODEL_ID}|{fixture_id}|{arm}|pass-{pass_index}",
                        "pass_index": pass_index,
                        "output_token_limit": OUTPUT_TOKEN_LIMIT,
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                    }
                )
            opportunities.append({"arm": arm, "pass_count": len(passes), "calls": calls})
        schedule.append(
            {
                "fixture_id": fixture_id,
                "model_id": QWEN_MODEL_ID,
                "source_family": str(fixture.get("source_family")),
                "source_text_sha256": model.get("source_text_sha256"),
                "response_text_sha256": model.get("response_text_sha256"),
                "call_opportunities": opportunities,
            }
        )
    return schedule


def schedule_errors(schedule: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject ID, arm, prompt, pass, limit, call, or blinding drift."""

    errors: list[str] = []
    observed_ids = [str(row.get("fixture_id")) for row in schedule]
    if observed_ids != list(FROZEN_FIXTURE_IDS):
        errors.append("frozen_fixture_ids_mismatch")
    plan = {arm: tuple(passes) for arm, _prompt_arm, passes in CALL_PLAN}
    for row in schedule:
        fixture_id = str(row.get("fixture_id"))
        opportunities = list(row.get("call_opportunities") or [])
        if [item.get("arm") for item in opportunities] != list(plan):
            errors.append(f"arm_plan_mismatch:{fixture_id}")
        for opportunity in opportunities:
            arm = str(opportunity.get("arm"))
            expected_passes = plan.get(arm, ())
            calls = list(opportunity.get("calls") or [])
            if opportunity.get("pass_count") != len(expected_passes):
                errors.append(f"pass_count_mismatch:{fixture_id}:{arm}")
            if tuple(call.get("pass_index") for call in calls) != expected_passes:
                errors.append(f"pass_index_mismatch:{fixture_id}:{arm}")
            for call in calls:
                pass_index = call.get("pass_index")
                expected_id = f"{QWEN_MODEL_ID}|{fixture_id}|{arm}|pass-{pass_index}"
                if call.get("call_id") != expected_id:
                    errors.append(f"call_id_mismatch:{fixture_id}:{arm}:{pass_index}")
                if call.get("output_token_limit") != OUTPUT_TOKEN_LIMIT:
                    errors.append(f"output_limit_mismatch:{fixture_id}:{arm}:{pass_index}")
                prompt = str(call.get("prompt", ""))
                if call.get("prompt_sha256") != sha256_text(prompt):
                    errors.append(f"prompt_hash_mismatch:{fixture_id}:{arm}:{pass_index}")
        by_arm = {str(item.get("arm")): item for item in opportunities}
        relational = list(by_arm.get("relational_sql", {}).get("calls") or [])
        dual = list(by_arm.get("dual_side", {}).get("calls") or [])
        if [call.get("prompt_sha256") for call in relational] != [
            call.get("prompt_sha256") for call in dual
        ]:
            errors.append(f"dual_side_prompt_mismatch:{fixture_id}")
    errors.extend(typed_blinding_errors(schedule))
    return list(dict.fromkeys(errors))


def schedule_projection(schedule: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project one audit row per fixture without adding sealed outcomes."""

    return [
        {
            "fixture_id": row.get("fixture_id"),
            "source_family": row.get("source_family"),
            "arm_count": len(list(row.get("call_opportunities") or [])),
            "call_count": sum(
                len(list(arm.get("calls") or []))
                for arm in list(row.get("call_opportunities") or [])
            ),
        }
        for row in schedule
    ]


def parse_canary_output(raw_output: str) -> JsonDict:
    """Keep the normalized text separate from the immutable raw output."""

    text = raw_output.strip()
    return {
        "text": text,
        "nonempty": bool(text),
        "matches_expected": text == CANARY_EXPECTED_OUTPUT,
    }


def build_runtime_evidence_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Join the same-process runtime facts into one replayable receipt."""

    specs = list(artifact.get("MODEL_SPECS") or [])
    loads = list(artifact.get("model_load_receipts") or [])
    linkage = list(artifact.get("binary_linkage_rows") or [])
    raw_rows = list(artifact.get("canary_raw_output_rows") or [])
    token_rows = list(artifact.get("canary_token_rows") or [])
    if not all((specs, loads, linkage, raw_rows, token_rows)):
        return []
    spec = specs[0]
    load = loads[0]
    cuda = dict(load.get("cuda_receipt") or {})
    raw = raw_rows[0]
    tokens = token_rows[0]
    server_log = str(load.get("server_log", ""))
    return [
        {
            "model_id": load.get("model_id"),
            "model_sha256": load.get("sha256"),
            "model_spec_sha256": spec.get("sha256"),
            "model_size_bytes": load.get("size_bytes", spec.get("size_bytes")),
            "server_command": deepcopy(load.get("command")),
            "server_pid": load.get("pid"),
            "process_returncode": load.get("process_returncode"),
            "requested_gpu_layers": cuda.get("requested_gpu_layers"),
            "native_cuda_linkage_confirmed": linkage[0].get("cuda_linkage_confirmed"),
            "cuda_runtime_log_confirmed": cuda.get("cuda_runtime_log_confirmed"),
            "cuda_log_markers": deepcopy(cuda.get("cuda_log_markers")),
            "owned_gpu_memory_mb": cuda.get("owned_gpu_memory_mb"),
            "owned_gpu_count": cuda.get("owned_gpu_count"),
            "gpu_offload_confirmed": cuda.get("gpu_offload_confirmed"),
            "health_ok": dict(load.get("health") or {}).get("ok"),
            "server_log_sha256": sha256_text(server_log),
            "raw_output_sha256": raw.get("raw_output_sha256"),
            "parsed_output": deepcopy(raw.get("parsed_output")),
            "completion_tokens": tokens.get("completion_tokens"),
            "generation_duration_s": tokens.get("generation_duration_s"),
            "model_load_duration_s": dict(load.get("health") or {}).get("duration_s"),
            "server_lifetime_duration_s": load.get("duration_s"),
            "teardown_result": deepcopy(load.get("cleanup")),
        }
    ]


def runtime_evidence_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Require one linked, token-producing, GPU-offloaded process receipt."""

    errors: list[str] = []
    loads = list(artifact.get("model_load_receipts") or [])
    raw_rows = list(artifact.get("canary_raw_output_rows") or [])
    token_rows = list(artifact.get("canary_token_rows") or [])
    runtime_rows = list(artifact.get("runtime_evidence_rows") or [])
    if len(loads) != 1:
        errors.append("canary_model_load_receipt_count")
    if len(raw_rows) != 1:
        errors.append("canary_raw_output_receipt_count")
    if len(token_rows) != 1:
        errors.append("canary_token_receipt_count")
    if len(runtime_rows) != 1:
        errors.append("runtime_evidence_row_count")
    if not loads:
        return errors
    load = loads[0]
    server_log = str(load.get("server_log", ""))
    server_log_hash = sha256_text(server_log)
    if load.get("server_log_sha256") != server_log_hash:
        errors.append("server_log_hash_mismatch")
    if load.get("model_id") != QWEN_MODEL_ID:
        errors.append("canary_model_id_mismatch")
    specs = list(artifact.get("MODEL_SPECS") or [])
    if not specs or load.get("sha256") != specs[0].get("sha256"):
        errors.append("model_hash_link_mismatch")
    if dict(load.get("health") or {}).get("ok") is not True:
        errors.append("canary_health_failed")
    cuda = dict(load.get("cuda_receipt") or {})
    if cuda.get("requested_gpu_layers") != "all":
        errors.append("all_layer_request_missing")
    if cuda.get("gpu_offload_confirmed") is not True:
        errors.append("canary_gpu_offload_unconfirmed")
    if load.get("process_returncode") is None:
        errors.append("server_returncode_missing")
    if float(load.get("duration_s", 0.0) or 0.0) <= 0.0:
        errors.append("server_timing_missing")
    if dict(load.get("cleanup") or {}).get("leak_free") is not True:
        errors.append("canary_cleanup_failed")
    gpu_rows = list(artifact.get("gpu_rows") or [])
    loaded_gpu = next((row for row in gpu_rows if row.get("phase") == "model_loaded"), {})
    loaded_apps = [
        app
        for app in list(loaded_gpu.get("compute_apps") or [])
        if app.get("owned_by_task") is True and int(app.get("used_memory_mb", 0) or 0) > 0
    ]
    if len(loaded_apps) != 2:
        errors.append("model_loaded_owned_gpu_memory_missing")
    after = next((row for row in gpu_rows if row.get("phase") == "after_teardown"), {})
    if any(app.get("owned_by_task") is True for app in list(after.get("compute_apps") or [])):
        errors.append("teardown_gpu_process_leak")
    if raw_rows:
        raw = raw_rows[0]
        raw_output = str(raw.get("raw_output", ""))
        if raw.get("raw_output_sha256") != sha256_text(raw_output):
            errors.append("canary_raw_output_hash_mismatch")
        if raw.get("parsed_output") != parse_canary_output(raw_output) or not dict(
            raw.get("parsed_output") or {}
        ).get("matches_expected"):
            errors.append("canary_parsed_output_invalid")
        if raw.get("server_log_sha256") != server_log_hash:
            errors.append("canary_server_log_link_mismatch")
    if token_rows:
        tokens = token_rows[0]
        if int(tokens.get("completion_tokens", 0) or 0) <= 0:
            errors.append("canary_completion_tokens_missing")
        if float(tokens.get("generation_duration_s", 0.0) or 0.0) <= 0.0:
            errors.append("canary_generation_timing_missing")
    expected_runtime = build_runtime_evidence_rows(artifact)
    if runtime_rows != expected_runtime:
        errors.append("runtime_evidence_rows_mismatch")
    return list(dict.fromkeys(errors))


def terminal_evidence_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute readiness from detailed evidence instead of trusting its score."""

    errors: list[str] = []
    schedule = list(artifact.get("schedule_rows") or [])
    errors.extend(schedule_errors(schedule))
    if artifact.get("rows") != schedule_projection(schedule):
        errors.append("row_projection_mismatch")
    if artifact.get("frozen_fixture_ids") != list(FROZEN_FIXTURE_IDS):
        errors.append("frozen_fixture_ids_field_mismatch")
    if artifact.get("frozen_schedule_hash") != sha256_text(canonical_json(schedule)):
        errors.append("frozen_schedule_hash_mismatch")
    if artifact.get("source_family_rows") != build_source_family_rows(schedule):
        errors.append("source_family_rows_mismatch")
    errors.extend(class_stratum_errors(list(artifact.get("class_stratum_rows") or [])))
    exposure_count = len(typed_blinding_errors(schedule))
    if artifact.get("label_exposure_count") != exposure_count or exposure_count != 0:
        errors.append("label_exposure_count_mismatch")
    mutation_rows = list(artifact.get("blinding_mutation_rows") or [])
    expected_mutations = blinding_mutation_rows()
    if {row.get("mutation_id") for row in mutation_rows} != {
        row["mutation_id"] for row in expected_mutations
    } or not all(row.get("passed") is True for row in mutation_rows):
        errors.append("blinding_mutation_controls_failed")
    if artifact.get("blinding_rule_rows") != blinding_rule_rows():
        errors.append("blinding_rule_rows_mismatch")
    errors.extend(model_spec_errors(list(artifact.get("MODEL_SPECS") or [])))
    identities = list(artifact.get("model_identity_rows") or [])
    if len(identities) != 1 or identities[0].get("template_present") is not True:
        errors.append("model_identity_receipt_missing")
    errors.extend(cuda_linkage_errors(list(artifact.get("binary_linkage_rows") or [])))
    backend = list(artifact.get("backend_rows") or [])
    if len(backend) != 1 or backend[0].get("version_text_used_for_cuda_decision") is not False:
        errors.append("backend_decision_receipt_invalid")
    gpu_rows = list(artifact.get("gpu_rows") or [])
    before = next((row for row in gpu_rows if row.get("phase") == "before"), {})
    if before.get("ok") is not True or int(before.get("gpu_count", 0) or 0) < 2:
        errors.append("gpu_preflight_missing")
    errors.extend(runtime_evidence_errors(artifact))
    checks = {
        str(row.get("check")): row.get("passed")
        for row in artifact.get("preconditions_checked", [])
    }
    if any(checks.get(name) is not True for name in READINESS_CHECKS):
        errors.append("readiness_checks_incomplete")
    return list(dict.fromkeys(errors))


def finalize_artifact(
    artifact: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Set readiness from receipts while making no verifier value claim."""

    result = deepcopy(dict(artifact))
    copied_checks = [deepcopy(dict(row)) for row in checks]
    result["preconditions_checked"] = copied_checks
    result["duration_s"] = round(max(0.0, float(duration_s)), 6)
    result["gate_check_summary"] = prior._gate_summary(copied_checks)
    check_map = {str(row.get("check")): row.get("passed") for row in copied_checks}
    checks_pass = all(check_map.get(name) is True for name in READINESS_CHECKS)
    evidence_errors = terminal_evidence_errors(result) if checks_pass else []
    if checks_pass and not evidence_errors:
        result.update(
            {
                "status": "completed",
                "inference_substrate_class": "model_full_generation",
                "grounding_runtime_ready_score": 1,
                "verdict_class": "positive",
                "honest_verdict": "positive_grounding_runtime_ready_no_verifier_value_claim",
            }
        )
    else:
        if result["gate_check_summary"]["passed"] and evidence_errors:
            copied_checks.append(gate_row("terminal_evidence", [], evidence_errors, False))
            result["preconditions_checked"] = copied_checks
            result["gate_check_summary"] = prior._gate_summary(copied_checks)
        failed = str(result["gate_check_summary"].get("failed_check") or "readiness_checks")
        result.update(
            {
                "status": "blocked",
                "inference_substrate_class": "blocked_no_run",
                "grounding_runtime_ready_score": 0,
                "verdict_class": "blocked",
                "honest_verdict": f"blocked_{failed}",
            }
        )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def validate_artifact(value: Mapping[str, Any] | str | Path | object) -> list[str]:
    """Cold-check schema, runtime evidence, schedule, verdict, and checksum."""

    artifact = prior._load_artifact_value(value)
    if artifact is None:
        return ["artifact_missing"]
    if artifact.get("__artifact_unreadable__"):
        return ["artifact_unreadable"]
    if artifact.get("__artifact_not_object__"):
        return ["artifact_not_object"]
    required = set(REQUIRED_ARTIFACT_FIELDS)
    if set(artifact) != required:
        return [f"artifact_fields_mismatch:{sorted(set(artifact) ^ required)}"]
    errors: list[str] = []
    if set(artifact.get("field_principles", {})) != required or any(
        not str(item).strip() for item in artifact.get("field_principles", {}).values()
    ):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(f"{verdict_class}_"):
        errors.append("honest_verdict_prefix_mismatch")
    checks = list(artifact.get("preconditions_checked") or [])
    if artifact.get("gate_check_summary") != prior._gate_summary(checks):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("status") not in {"completed", "blocked"}:
        errors.append("terminal_status_invalid")
    if verdict_class == "blocked":
        if artifact.get("status") != "blocked":
            errors.append("blocked_status_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("grounding_runtime_ready_score") != 0:
            errors.append("blocked_readiness_score_mismatch")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked_gate_summary_mismatch")
    elif verdict_class == "positive":
        if artifact.get("status") != "completed":
            errors.append("positive_status_mismatch")
        if artifact.get("inference_substrate_class") != "model_full_generation":
            errors.append("positive_substrate_class_mismatch")
        if artifact.get("grounding_runtime_ready_score") != 1:
            errors.append("positive_readiness_score_mismatch")
        if artifact.get("gate_check_summary", {}).get("passed") is not True:
            errors.append("positive_gate_summary_mismatch")
        errors.extend(terminal_evidence_errors(artifact))
    else:
        errors.append("terminal_verdict_class_invalid_for_runtime")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Flush one compact line so the outer runner can observe progress."""

    prior._progress(phase, event, **fields)


def _run_subprocess(
    command: list[str], *, timeout_s: float = 15.0, phase: int = 6
) -> JsonDict:  # pragma: no cover
    """Run one bounded command with visible start and end receipts."""

    return prior._run_subprocess(command, timeout_s=timeout_s, phase=phase)


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover
    """Hash the exact inputs, prior receipt, code, tests, and runtime helpers."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("results/experiment_7138_v627_relational_fixture.json"),
        Path("results/experiment_7139_v627_symbolic_grounding_ab.json"),
        Path("results/experiment_7150_v628_grounding_preflight.json"),
        Path("python/carnot/experiment_7150_v628_grounding_preflight.py"),
        Path("python/carnot/experiment_7153_v629_grounding_runtime.py"),
        Path("python/carnot/inference/llama_server_supervisor.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("scripts/experiment_template.py"),
        Path("scripts/experiments/experiment_7153_v629_grounding_runtime.py"),
        Path("tests/python/test_experiment_7153_v629_grounding_runtime.py"),
    )
    return {
        str(path): sha256_file(root / path) if (root / path).is_file() else None for path in paths
    }


def _output_precondition(result_path: Path, raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Require writable result paths and a clean task-specific raw directory."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(str(path) for path in raw_dir.iterdir())
    return {
        "result_parent_writable": os.access(result_path.parent, os.W_OK),
        "raw_dir_writable": os.access(raw_dir, os.W_OK),
        "raw_dir_isolated": not existing,
        "existing_raw_entries": existing,
    }


def _canary_request(port: int) -> tuple[JsonDict, JsonDict]:  # pragma: no cover
    """Send one deterministic chat call and retain the complete HTTP body."""

    payload = {
        "messages": [
            {"role": "system", "content": "Follow the user instruction exactly."},
            {"role": "user", "content": CANARY_PROMPT},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": RANDOM_SEED & 0x7FFFFFFF,
        "cache_prompt": False,
    }
    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(http_request, timeout=240.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = list(body.get("choices") or [{}])
    message = dict(choices[0].get("message") or {})
    usage = dict(body.get("usage") or {})
    output = str(message.get("content") or message.get("reasoning_content") or "")
    return (
        {
            "model_id": QWEN_MODEL_ID,
            "request": payload,
            "request_sha256": sha256_text(canonical_json(payload)),
            "prompt": CANARY_PROMPT,
            "prompt_sha256": sha256_text(CANARY_PROMPT),
        },
        {
            "raw_output": output,
            "parsed_output": parse_canary_output(output),
            "raw_response": body,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "generation_duration_s": time.perf_counter() - started,
        },
    )


def _run_canary(
    spec: Mapping[str, Any], *, server_path: Path, raw_dir: Path
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, list[JsonDict]]:  # pragma: no cover
    """Run one task-owned Qwen server and retain its full lifecycle evidence."""

    port = prior._free_port()
    command = prior._server_command(server_path, Path(str(spec["model_path"])), port)
    contract = supervisor_contract(
        outer_deadline_s=900,
        health_timeout_s=480,
        token_timeout_s=240,
        cleanup_grace_s=30,
        kill_after_cleanup_timeout_s=10,
        retry_budget=0,
        endurance_interval_s=0,
        endurance_sample_count=1,
    )
    supervisor = NativeLlamaServerSupervisor(command, raw_dir, contract)
    started = time.perf_counter()
    identity: JsonDict = {}
    health: JsonDict = {"ok": False, "classification": "not_started"}
    prompt_row: JsonDict = {"model_id": QWEN_MODEL_ID}
    response_row: JsonDict = {}
    gpu_rows: list[JsonDict] = []
    cleanup: JsonDict = {"action": "not_started", "bounded": True, "leak_free": True}
    failure = None
    load_end_printed = False
    generation_end_printed = False
    _progress(8, "subprocess_start", command=command)
    _progress(8, "model_load_receipt_start", model_id=QWEN_MODEL_ID, command=command)
    try:
        identity = supervisor.launch()
        health = prior._wait_for_health(supervisor, port, timeout_s=480.0)
        _progress(
            8,
            "model_load_receipt_end",
            model_id=QWEN_MODEL_ID,
            health=health.get("ok"),
            pid=identity.get("pid"),
        )
        load_end_printed = True
        gpu_rows.append(prior._gpu_snapshot("model_loaded", phase=8))
        if health.get("ok") is not True:
            raise RuntimeError(f"server health failed: {health.get('classification')}")
        _progress(8, "generation_receipt_start", model_id=QWEN_MODEL_ID)
        prompt_row, response_row = _canary_request(port)
        _progress(
            8,
            "generation_receipt_end",
            model_id=QWEN_MODEL_ID,
            completion_tokens=response_row.get("completion_tokens"),
        )
        generation_end_printed = True
        gpu_rows.append(prior._gpu_snapshot("after_generation", phase=8))
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if not load_end_printed:
            _progress(8, "model_load_receipt_end", model_id=QWEN_MODEL_ID, error=failure)
        if not generation_end_printed:
            _progress(8, "generation_receipt_end", model_id=QWEN_MODEL_ID, error=failure)
    finally:
        _progress(8, "subprocess_cleanup_start", pid=identity.get("pid"))
        cleanup = supervisor.cleanup()
        _progress(8, "subprocess_cleanup_end", leak_free=cleanup.get("leak_free"))
    process_returncode = None
    if supervisor.proc is not None:
        try:
            process_returncode = supervisor.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process_returncode = supervisor.proc.poll()
    _progress(8, "subprocess_end", command=command, returncode=process_returncode)
    gpu_rows.append(prior._gpu_snapshot("after_teardown", phase=8))
    server_log = (
        supervisor.log_path.read_text(encoding="utf-8", errors="replace")
        if supervisor.log_path.is_file()
        else ""
    )
    server_log_hash = sha256_text(server_log)
    loaded_gpu = next((row for row in gpu_rows if row.get("phase") == "model_loaded"), {})
    cuda = cuda_offload_receipt(
        server_log,
        loaded_gpu,
        pid=int(identity.get("pid", -1)),
        command=command,
    )
    load_receipt = {
        "model_id": QWEN_MODEL_ID,
        "loaded_path": spec.get("loaded_path", spec.get("model_path")),
        "revision": spec.get("revision"),
        "size_bytes": spec.get("size_bytes"),
        "sha256": spec.get("sha256"),
        "template_source": spec.get("chat_template_source"),
        "template_sha256": spec.get("chat_template_sha256"),
        "backend": "native_llama_server",
        "command": command,
        "pid": identity.get("pid"),
        "health": health,
        "server_log_path": str(supervisor.log_path),
        "server_log": server_log,
        "server_log_sha256": server_log_hash,
        "process_returncode": process_returncode,
        "cleanup": cleanup,
        "cuda_layers_offloaded": cuda.get("logged_offloaded_layers"),
        "total_layers": cuda.get("logged_total_layers"),
        "cuda_placement_confirmed": cuda.get("gpu_offload_confirmed", False),
        "gpu_offload_confirmed": cuda.get("gpu_offload_confirmed", False),
        "cuda_receipt": cuda,
        "duration_s": time.perf_counter() - started,
        "error": failure,
    }
    raw_output = str(response_row.get("raw_output", ""))
    raw_row = {
        "model_id": QWEN_MODEL_ID,
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "parsed_output": response_row.get("parsed_output", parse_canary_output(raw_output)),
        "raw_response": response_row.get("raw_response", {}),
        "raw_response_sha256": sha256_text(canonical_json(response_row.get("raw_response", {}))),
        "server_log_sha256": server_log_hash,
        "error": failure,
    }
    prompt_tokens = int(response_row.get("prompt_tokens", 0) or 0)
    completion_tokens = int(response_row.get("completion_tokens", 0) or 0)
    token_row = {
        "model_id": QWEN_MODEL_ID,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "generation_duration_s": response_row.get("generation_duration_s", 0.0),
    }
    return load_receipt, prompt_row, raw_row, token_row, gpu_rows


def _checkpoint(path: Path, artifact: JsonDict) -> None:  # pragma: no cover
    """Refresh the checksum before each fallible live phase."""

    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    write_artifact(path, artifact)


def run_experiment(  # pragma: no cover
    *, root: Path, run_date: str, result_path: Path, fixture_path: Path, raw_dir: Path
) -> JsonDict:
    """Write first, freeze labels last, run one canary, and reduce readiness."""

    started = time.perf_counter()
    _progress(0, "phase_start", name="schema_first_write")
    artifact = initialize_artifact(result_path, run_date)
    _progress(0, "phase_end", name="schema_first_write", path=str(result_path))

    _progress(1, "phase_start", name="date_sources_and_output")
    artifact["source_artifact_hashes"] = _source_hashes(root)
    output_observed = _output_precondition(result_path, raw_dir)
    output_expected = {
        "result_parent_writable": True,
        "raw_dir_writable": True,
        "raw_dir_isolated": True,
        "existing_raw_entries": [],
    }
    checks = [
        gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE),
        gate_row(
            "output_paths", output_expected, output_observed, output_observed == output_expected
        ),
    ]
    _checkpoint(result_path, artifact)
    _progress(
        1, "phase_end", name="date_sources_and_output", passed=all(row["passed"] for row in checks)
    )
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(2, "phase_start", name="fixture_contract")
    fixture, fixture_checks = prior._load_fixture(fixture_path)
    checks.extend(fixture_checks)
    _progress(
        2,
        "phase_end",
        name="fixture_contract",
        passed=bool(fixture) and fixture_checks[-1]["passed"],
    )
    if fixture is None or any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(3, "phase_start", name="typed_blinding_and_schedule")
    schedule = build_schedule(list(fixture["model_view_rows"]), list(fixture["fixture_rows"]))
    selected = {str(row["fixture_id"]): row for row in fixture["model_view_rows"]}
    blinding_errors = typed_blinding_errors([selected[item] for item in FROZEN_FIXTURE_IDS])
    blinding_errors.extend(typed_blinding_errors(schedule))
    mutation_rows = blinding_mutation_rows()
    if not all(row["passed"] for row in mutation_rows):
        blinding_errors.append("blinding_mutation_control_failed")
    schedule_failures = schedule_errors(schedule)
    artifact.update(
        {
            "rows": schedule_projection(schedule),
            "blinding_rule_rows": blinding_rule_rows(),
            "blinding_mutation_rows": mutation_rows,
            "schedule_rows": schedule,
            "source_family_rows": build_source_family_rows(schedule),
            "frozen_fixture_ids": list(FROZEN_FIXTURE_IDS),
            "frozen_schedule_hash": sha256_text(canonical_json(schedule)),
            "label_exposure_count": len(blinding_errors),
        }
    )
    checks.extend(
        [
            gate_row("typed_blinding", [], blinding_errors, not blinding_errors),
            gate_row("frozen_schedule", [], schedule_failures, not schedule_failures),
        ]
    )
    _checkpoint(result_path, artifact)
    _progress(3, "phase_end", name="typed_blinding_and_schedule", call_count=168)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(4, "phase_start", name="sealed_class_strata")
    class_rows = build_class_stratum_rows(schedule, list(fixture["sealed_scorer_rows"]))
    class_errors = class_stratum_errors(class_rows)
    artifact["class_stratum_rows"] = class_rows
    checks.append(gate_row("class_strata", [], class_errors, not class_errors))
    _checkpoint(result_path, artifact)
    _progress(4, "phase_end", name="sealed_class_strata", passed=not class_errors)
    if class_errors:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(5, "phase_start", name="cached_model_identity")
    specs = resolve_model_specs()
    _progress(5, "benchmark_start", name="model_file_hash_and_template")
    specs, identities, model_errors = model_identity_receipts(specs)
    _progress(5, "benchmark_end", name="model_file_hash_and_template", errors=model_errors)
    artifact["MODEL_SPECS"] = specs
    artifact["model_identity_rows"] = identities
    non_template_errors = [
        error for error in model_errors if error != "embedded_chat_template_missing"
    ]
    checks.extend(
        [
            gate_row("cached_qwen_q4", [], non_template_errors, not non_template_errors),
            gate_row(
                "embedded_chat_template",
                True,
                bool(identities and identities[0].get("template_present")),
                bool(identities and identities[0].get("template_present")),
            ),
        ]
    )
    _checkpoint(result_path, artifact)
    _progress(5, "phase_end", name="cached_model_identity", passed=not model_errors)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(6, "phase_start", name="native_binary_and_cuda_linkage")
    server_path = resolve_native_llama_server()
    linkage, backend = binary_runtime_receipts(server_path, command_runner=_run_subprocess)
    artifact["binary_linkage_rows"] = linkage
    artifact["backend_rows"] = backend
    linkage_errors = cuda_linkage_errors(linkage)
    checks.append(
        gate_row(
            "native_cuda_linkage", [], linkage_errors, not linkage_errors and backend[0]["exists"]
        )
    )
    _checkpoint(result_path, artifact)
    _progress(6, "phase_end", name="native_binary_and_cuda_linkage", passed=not linkage_errors)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(7, "phase_start", name="gpu_precondition_and_checkpoint")
    before_gpu = prior._gpu_snapshot("before", phase=7)
    artifact["gpu_rows"] = [before_gpu]
    external_apps = [
        row for row in before_gpu.get("compute_apps", []) if not row.get("owned_by_task")
    ]
    gpu_observed = {
        "query_ok": before_gpu.get("ok"),
        "gpu_count": before_gpu.get("gpu_count"),
        "external_compute_apps": external_apps,
    }
    gpu_expected = {"query_ok": True, "gpu_count": 2, "external_compute_apps": []}
    gpu_passed = (
        before_gpu.get("ok") is True and before_gpu.get("gpu_count") == 2 and not external_apps
    )
    checks.append(gate_row("gpu_available", gpu_expected, gpu_observed, gpu_passed))
    if not gpu_passed:
        _progress(7, "phase_end", name="gpu_precondition_and_checkpoint", passed=False)
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )
    artifact["status"] = "preconditions_passed"
    artifact["preconditions_checked"] = deepcopy(checks)
    _checkpoint(result_path, artifact)
    _progress(7, "status", status="preconditions_passed")
    _progress(7, "phase_end", name="gpu_precondition_and_checkpoint", passed=True)

    _progress(8, "phase_start", name="real_qwen_canary")
    load, prompt_row, raw_row, token_row, runtime_gpu_rows = _run_canary(
        specs[0], server_path=server_path, raw_dir=raw_dir
    )
    artifact["gpu_rows"].extend(runtime_gpu_rows)
    artifact["model_load_receipts"] = [load]
    artifact["canary_prompt_rows"] = [prompt_row]
    artifact["canary_raw_output_rows"] = [raw_row]
    artifact["canary_token_rows"] = [token_row]
    artifact["runtime_evidence_rows"] = build_runtime_evidence_rows(artifact)
    canary_errors = runtime_evidence_errors(artifact)
    checks.append(gate_row("real_qwen_canary", [], canary_errors, not canary_errors))
    _checkpoint(result_path, artifact)
    _progress(8, "phase_end", name="real_qwen_canary", passed=not canary_errors)
    if canary_errors:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(9, "phase_start", name="terminal_reduction")
    result = finalize_artifact(artifact, checks, duration_s=time.perf_counter() - started)
    write_artifact(result_path, result)
    _progress(
        9,
        "phase_end",
        name="terminal_reduction",
        readiness=result["grounding_runtime_ready_score"],
        status=result["status"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the live qualification or cold-validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(10, "subprocess_start", name="artifact_validation", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(10, "subprocess_end", name="artifact_validation", valid=not errors)
        print(canonical_json({"valid": not errors, "errors": errors}), flush=True)
        return int(bool(errors))
    root = find_repo_root()
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    result = run_experiment(
        root=root,
        run_date=args.date,
        result_path=result_path,
        fixture_path=root / FIXTURE_PATH,
        raw_dir=root / RAW_DIR,
    )
    errors = validate_artifact(result)
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "valid": not errors,
                "errors": errors,
                "verdict_class": result.get("verdict_class"),
                "grounding_runtime_ready_score": result.get("grounding_runtime_ready_score"),
            }
        ),
        flush=True,
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
