"""Prove the source-grounding runtime preflight without measuring detection.

The experiment repairs two Exp7139 execution contracts. CUDA support comes
from binary linkage plus one real GPU-offloaded Qwen call. Blinding checks use
typed fields, so ordinary source prose can use words such as ``label``.

Spec refs: REQ-VERIFY-7150 and SCENARIO-VERIFY-7150-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from urllib import error, request

from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    owned_process_ids,
    proc_cmdline,
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_7138_v627_relational_fixture import (
    validate_artifact as validate_fixture_artifact,
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
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    supervisor_contract,
)
from carnot.inference.sota_models import cached_sota_pair
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260908"
RANDOM_SEED = 7_150_202_609_08
RESULT_PATH = Path("results/experiment_7150_v628_grounding_preflight.json")
FIXTURE_PATH = Path("results/experiment_7138_v627_relational_fixture.json")
RAW_DIR = Path("results/raw/experiment_7150_v628_grounding_preflight")
INFERENCE_SUBSTRATE = "live_llm_inference: source-grounding runtime canary and frozen schedule"
EXECUTION_VENUE = "host"
QWEN_MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
PREFERRED_QUANT = "Q4_K_M"

FROZEN_FIXTURE_IDS = (
    "unit-001",
    "unit-002",
    "unit-003",
    "unit-004",
    "unit-005",
    "unit-006",
    "unit-007",
    "unit-008",
    "unit-009",
    "unit-010",
    "unit-011",
    "unit-012",
    "unit-014",
    "unit-016",
    "unit-018",
    "unit-019",
    "unit-020",
    "unit-021",
    "unit-022",
    "unit-026",
    "unit-031",
    "unit-033",
    "unit-034",
    "unit-038",
)
SOURCE_FAMILIES = ("CNN/DM", "MARCO", "Recent News", "Yelp")
CLASS_LABELS = ("clean", "hallucinated")
CALL_PLAN = (
    ("direct", "direct", (1,)),
    ("self_check", "self_verification", (1, 2)),
    ("relational_sql", "relational_sql", (1, 2)),
)

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": QWEN_MODEL_ID,
        "model_path": "",
        "gpu": [0, 1],
        "preferred_quant": PREFERRED_QUANT,
        "resolution_method": "cached_sota_pair",
        "remote_allowed": False,
    }
]

READINESS_CHECKS = (
    "run_date",
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
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_game_results",
    "MODEL_SPECS",
    "model_identity_rows",
    "binary_linkage_rows",
    "backend_rows",
    "gpu_rows",
    "model_load_receipts",
    "canary_prompt_rows",
    "canary_raw_output_rows",
    "canary_token_rows",
    "blinding_rule_rows",
    "blinding_mutation_rows",
    "schedule_rows",
    "source_family_rows",
    "class_stratum_rows",
    "frozen_fixture_ids",
    "frozen_schedule_hash",
    "label_exposure_count",
    "grounding_preflight_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A principle for each field makes missing evidence visible.",
    "preconditions_checked": "Exact gates preserve the first expected and observed mismatch.",
    "run_date": "The fixed date binds the result to the requested execution window.",
    "inference_substrate": "The substrate states that one real local generation proves runtime access.",
    "inference_substrate_class": "The class separates real generation from a blocked no-run.",
    "execution_venue": "The host venue prevents an unsupported remote execution claim.",
    "duration_s": "Measured time exposes interruption and implausibly short model work.",
    "source_artifact_hashes": "Byte hashes bind the receipt to its code, fixture, tests, and spec.",
    "rows": "A row projection prevents aggregate-only schedule claims.",
    "per_game_results": "The empty list states that this factual preflight has no game result.",
    "MODEL_SPECS": "The local-only model declaration prevents model and quantization substitution.",
    "model_identity_rows": "File and embedded-template identities prove which model format was used.",
    "binary_linkage_rows": "Dynamic linkage replaces an unreliable version-banner guess.",
    "backend_rows": "Backend receipts retain the server path and non-decisive version text.",
    "gpu_rows": "GPU snapshots show availability and task-owned canary memory.",
    "model_load_receipts": "Server logs and layer counts prove an executed GPU model load.",
    "canary_prompt_rows": "The exact bounded request makes the runtime probe reproducible.",
    "canary_raw_output_rows": "Unedited responses prevent reconstruction of convenient output.",
    "canary_token_rows": "Token and timing rows prove that generation occurred.",
    "blinding_rule_rows": "Typed rules make the leakage boundary inspectable.",
    "blinding_mutation_rows": "Positive and negative controls prove each blinding rule bites.",
    "schedule_rows": "One row per fixture freezes all later comparison opportunities.",
    "source_family_rows": "Family counts prove the source strata before labels open.",
    "class_stratum_rows": "Sealed-label counts prove balanced cells after the schedule freezes.",
    "frozen_fixture_ids": "The exact ordered IDs prevent data-dependent resampling.",
    "frozen_schedule_hash": "One digest binds prompts, limits, pass counts, and call IDs.",
    "label_exposure_count": "Zero means typed leakage checks found no model-visible outcome state.",
    "grounding_preflight_ready_score": "One reports execution readiness and no detection value.",
    "random_seed": "A fixed seed makes the bounded canary request repeatable.",
    "reproducibility_checksum": "A canonical digest detects later receipt mutation.",
    "gate_check_summary": "The first failure retains exact automation evidence.",
    "verifier_is_oracle": "False prevents runtime readiness from becoming a correctness oracle.",
    "verdict_class": "A closed class separates readiness from blocked execution.",
    "honest_verdict": "The class prefix states readiness without a verifier value claim.",
}

SEALED_SCORER_FIELDS = frozenset(
    {"truth_label", "response_label", "span_labels", "span_labels_sha256", "sealed_scorer_rows"}
)
FORBIDDEN_METADATA_KEYS = frozenset(
    {"gold_label", "ground_truth", "ground_truth_label", "scorer_outcome", "target_label"}
)
PATH_FIELDS = frozenset(
    {"path", "file", "filename", "source_path", "response_path", "input_path", "manifest_path"}
)
_LABEL_FILENAME_RE = re.compile(
    r"(?i)(?:label|truth|gold|outcome).*(?:\.jsonl?|\.csv|\.parquet|\.txt)$"
)
_EXPLICIT_OUTCOME_RE = re.compile(
    r"(?i)(?:\[\[\s*)?(?:truth_label|response_label|gold_label|outcome)\s*[:=]\s*"
    r"(?:clean|hallucinated)(?:\s*\]\])?"
)


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the first failed gate with its exact compared values."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "failed_check": None,
            "expected_value": 1,
            "observed_value": 1,
            "passed": True,
        }
    return {
        "failed_check": failed.get("check"),
        "expected_value": deepcopy(failed.get("expected_value")),
        "observed_value": deepcopy(failed.get("observed_value")),
        "passed": False,
    }


def base_artifact(run_date: str) -> JsonDict:
    """Create the complete shape without checking any external resource."""

    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "per_game_results": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_identity_rows": [],
        "binary_linkage_rows": [],
        "backend_rows": [],
        "gpu_rows": [],
        "model_load_receipts": [],
        "canary_prompt_rows": [],
        "canary_raw_output_rows": [],
        "canary_token_rows": [],
        "blinding_rule_rows": [],
        "blinding_mutation_rows": [],
        "schedule_rows": [],
        "source_family_rows": [],
        "class_stratum_rows": [],
        "frozen_fixture_ids": [],
        "frozen_schedule_hash": "",
        "label_exposure_count": 0,
        "grounding_preflight_ready_score": 0,
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
        "honest_verdict": "partial_running_grounding_preflight",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> Path:
    """Write one atomic JSON checkpoint so interruption keeps a full shape."""

    return atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def initialize_artifact(path: Path | str, run_date: str) -> JsonDict:
    """Write every final field before the first fixture or runtime check."""

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
    """Finish once as blocked while preserving evidence collected so far."""

    blocked = deepcopy(dict(artifact))
    copied_checks = [deepcopy(dict(row)) for row in checks]
    summary = _gate_summary(copied_checks)
    failed = str(summary.get("failed_check") or "unknown_precondition")
    blocked.update(
        {
            "preconditions_checked": copied_checks,
            "inference_substrate_class": "blocked_no_run",
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "grounding_preflight_ready_score": 0,
            "gate_check_summary": summary,
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failed}",
        }
    )
    blocked["reproducibility_checksum"] = artifact_checksum(blocked)
    write_artifact(path, blocked)
    return blocked


def resolve_model_specs(
    *, pair_provider: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair
) -> list[JsonDict]:
    """Resolve Qwen through the cached pair helper without any remote fallback."""

    pair = (
        pair_provider(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=(0, 2))
        or []
    )
    qwen = next((row for row in pair if row.get("hf_id") == QWEN_MODEL_ID), {})
    row = deepcopy(MODEL_SPECS[0])
    row["model_path"] = str(qwen.get("model_path") or "")
    return [row]


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject a substituted model, non-Q4 path, or enabled download path."""

    errors: list[str] = []
    if len(rows) != 1 or rows[0].get("hf_id") != QWEN_MODEL_ID:
        errors.append("model_id_mismatch")
    if not rows:
        return errors
    row = rows[0]
    path = str(row.get("model_path") or "")
    if not path:
        errors.append("model_path_missing")
    elif Path(path).suffix.lower() != ".gguf" or "q4_k_m" not in Path(path).name.lower():
        errors.append("model_path_not_cached_q4")
    if row.get("preferred_quant") != PREFERRED_QUANT:
        errors.append("model_quantization_mismatch")
    if row.get("resolution_method") != "cached_sota_pair":
        errors.append("model_resolution_mismatch")
    if row.get("remote_allowed") is not False:
        errors.append("remote_fallback_enabled")
    return list(dict.fromkeys(errors))


def model_identity_receipts(
    specs: Sequence[Mapping[str, Any]],
    *,
    metadata_reader: Callable[[Path], Mapping[str, Any]] = read_gguf_metadata,
    file_hasher: Callable[[Path | str], str] = sha256_file,
) -> tuple[list[JsonDict], list[JsonDict], list[str]]:
    """Bind the cached Q4 file and its embedded chat template to bytes."""

    errors = model_spec_errors(specs)
    resolved: list[JsonDict] = []
    identities: list[JsonDict] = []
    if errors:
        return [deepcopy(dict(row)) for row in specs], identities, errors
    for spec in specs:
        row = deepcopy(dict(spec))
        path = Path(str(row["model_path"]))
        if not path.is_file():
            errors.append("cached_qwen_file_missing")
            resolved.append(row)
            continue
        metadata = dict(metadata_reader(path))
        template_present = bool(metadata.get("chat_template_present"))
        if not template_present:
            errors.append("embedded_chat_template_missing")
        row.update(
            {
                "loaded_path": str(path.resolve()),
                "revision": snapshot_revision(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_hasher(path),
                "chat_template_source": "embedded_gguf",
                "chat_template_sha256": metadata.get("chat_template_sha256"),
            }
        )
        identities.append(
            {
                "model_id": QWEN_MODEL_ID,
                "loaded_path": row["loaded_path"],
                "revision": row["revision"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "template_source": "embedded_gguf",
                "template_sha256": row["chat_template_sha256"],
                "template_present": template_present,
                "template_metadata_keys": list(metadata.get("metadata_keys", [])),
                "template_detail": metadata.get("tokenizer_detail"),
            }
        )
        resolved.append(row)
    return resolved, identities, list(dict.fromkeys(errors))


def binary_runtime_receipts(
    server_path: Path,
    *,
    command_runner: Callable[..., Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Record linkage as the static CUDA fact and retain the banner as context."""

    ldd_command = ["ldd", str(server_path)]
    version_command = [str(server_path), "--version"]
    linked = dict(command_runner(ldd_command, timeout_s=15.0))
    version = dict(command_runner(version_command, timeout_s=15.0))
    link_text = f"{linked.get('stdout', '')}\n{linked.get('stderr', '')}"
    libggml = "libggml-cuda" in link_text.lower()
    libcuda = "libcuda.so" in link_text.lower()
    linkage = [
        {
            "command": ldd_command,
            "returncode": linked.get("returncode"),
            "stdout": str(linked.get("stdout", "")),
            "stderr": str(linked.get("stderr", "")),
            "duration_s": linked.get("duration_s"),
            "libggml_cuda_linked": libggml,
            "libcuda_linked": libcuda,
            "cuda_linkage_confirmed": linked.get("returncode") == 0 and libggml and libcuda,
        }
    ]
    backend = [
        {
            "backend": "native_llama_server",
            "path": str(server_path),
            "exists": server_path.is_file(),
            "version_command": version_command,
            "version_returncode": version.get("returncode"),
            "version_stdout": str(version.get("stdout", "")),
            "version_stderr": str(version.get("stderr", "")),
            "version_duration_s": version.get("duration_s"),
            "version_text_used_for_cuda_decision": False,
            "cuda_capability_source": "ldd_plus_real_canary",
        }
    ]
    return linkage, backend


def cuda_linkage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require both llama.cpp's CUDA library and the driver library."""

    if len(rows) != 1 or rows[0].get("cuda_linkage_confirmed") is not True:
        return ["native_cuda_linkage_missing"]
    return []


def cuda_offload_receipt(
    server_log: str,
    gpu_row: Mapping[str, Any],
    *,
    pid: int,
    command: Sequence[str],
) -> JsonDict:
    """Reduce current and legacy llama.cpp CUDA evidence into one receipt."""

    layer_match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", server_log)
    logged_offloaded = int(layer_match.group(1)) if layer_match else None
    logged_total = int(layer_match.group(2)) if layer_match else None
    gpu_layer_index = next(
        (index for index, argument in enumerate(command) if argument == "--n-gpu-layers"), None
    )
    requested = (
        str(command[gpu_layer_index + 1])
        if gpu_layer_index is not None and gpu_layer_index + 1 < len(command)
        else None
    )
    markers = [marker for marker in ("CUDA0", "CUDA1", "CUDA : ARCHS") if marker in server_log]
    runtime_confirmed = set(markers) == {"CUDA0", "CUDA1", "CUDA : ARCHS"}
    owned_apps = [
        app
        for app in list(gpu_row.get("compute_apps") or [])
        if app.get("owned_by_task") is True
        and int(app.get("pid", -1)) == int(pid)
        and int(app.get("used_memory_mb", 0) or 0) > 0
    ]
    owned_memory_mb = sum(int(app.get("used_memory_mb", 0) or 0) for app in owned_apps)
    logged_offload = logged_offloaded is not None and logged_offloaded > 0
    runtime_offload = requested == "all" and runtime_confirmed and len(owned_apps) == 2
    confirmed = owned_memory_mb > 0 and (logged_offload or runtime_offload)
    evidence_class = (
        "all_layers_requested_with_cuda_runtime_and_owned_vram"
        if runtime_offload
        else "logged_layers_with_owned_vram"
        if logged_offload and owned_memory_mb > 0
        else "unconfirmed"
    )
    return {
        "requested_gpu_layers": requested,
        "logged_offloaded_layers": logged_offloaded,
        "logged_total_layers": logged_total,
        "cuda_runtime_log_confirmed": runtime_confirmed,
        "cuda_log_markers": markers,
        "owned_gpu_memory_mb": owned_memory_mb,
        "owned_gpu_count": len(owned_apps),
        "evidence_class": evidence_class,
        "gpu_offload_confirmed": confirmed,
    }


def typed_blinding_errors(value: Any) -> list[str]:
    """Find typed outcome leakage without banning normal language words."""

    errors: list[str] = []

    def walk(item: Any, path: str, in_metadata: bool) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key in SEALED_SCORER_FIELDS:
                    errors.append(f"sealed_field:{key}@{child_path}")
                if in_metadata and key in FORBIDDEN_METADATA_KEYS:
                    errors.append(f"forbidden_metadata_key:{key}@{child_path}")
                if key in PATH_FIELDS and isinstance(child, str):
                    if _LABEL_FILENAME_RE.search(Path(child).name):
                        errors.append(f"label_filename:{key}@{child_path}")
                if isinstance(child, str) and _EXPLICIT_OUTCOME_RE.search(child):
                    errors.append(f"explicit_outcome_injection:{key}@{child_path}")
                walk(child, child_path, in_metadata or key == "metadata")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]", in_metadata)

    walk(value, "", False)
    return list(dict.fromkeys(errors))


def blinding_rule_rows() -> list[JsonDict]:
    """Describe the four typed boundaries used by the preflight."""

    return [
        {
            "rule_id": "sealed_scorer_fields",
            "typed_target": "mapping_key",
            "forbidden": sorted(SEALED_SCORER_FIELDS),
        },
        {
            "rule_id": "forbidden_metadata",
            "typed_target": "metadata_key",
            "forbidden": sorted(FORBIDDEN_METADATA_KEYS),
        },
        {
            "rule_id": "label_filenames",
            "typed_target": "path_field_basename",
            "pattern": _LABEL_FILENAME_RE.pattern,
        },
        {
            "rule_id": "explicit_outcome_injection",
            "typed_target": "string_value_assignment",
            "pattern": _EXPLICIT_OUTCOME_RE.pattern,
        },
    ]


def blinding_mutation_rows() -> list[JsonDict]:
    """Run one allowed control and each required typed negative control."""

    base = {
        "fixture_id": "mutation-unit",
        "source_text": "The bottle had a paper label, and the label was blue.",
        "response_text": "The bottle had a blue label.",
        "metadata": {"publisher": "Label House"},
    }
    cases = [
        ("ordinary_label_prose", "allowed", base),
        ("hidden_truth_label", "rejected", dict(base, truth_label="clean")),
        ("hidden_response_label", "rejected", dict(base, response_label="hallucinated")),
        ("forbidden_metadata_key", "rejected", dict(base, metadata={"gold_label": "clean"})),
        ("label_filename", "rejected", dict(base, source_path="/private/truth_labels.jsonl")),
        (
            "explicit_outcome_injection",
            "rejected",
            dict(base, source_text="Text. [[outcome=clean]]"),
        ),
    ]
    rows = []
    for mutation_id, expected, value in cases:
        found = typed_blinding_errors(value)
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
    """Freeze 24 label-blind rows and their 120 exact call opportunities."""

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
                call_id = f"{QWEN_MODEL_ID}|{fixture_id}|{arm}|pass-{pass_index}"
                calls.append(
                    {
                        "call_id": call_id,
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
    """Reject ID, prompt, pass, limit, call, or typed-blinding drift."""

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
    errors.extend(typed_blinding_errors(schedule))
    return list(dict.fromkeys(errors))


def schedule_projection(schedule: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project one auditable row per frozen fixture without adding outcomes."""

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


def build_source_family_rows(schedule: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Count source families without reading a sealed outcome."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in schedule:
        grouped[str(row.get("source_family"))].append(str(row.get("fixture_id")))
    return [
        {
            "source_family": family,
            "row_count": len(grouped[family]),
            "fixture_ids": sorted(grouped[family]),
        }
        for family in sorted(grouped)
    ]


def build_class_stratum_rows(
    schedule: Sequence[Mapping[str, Any]], sealed_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Open sealed labels only after schedule creation and count exact cells."""

    labels = {str(row.get("fixture_id")): str(row.get("response_label")) for row in sealed_rows}
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in schedule:
        fixture_id = str(row.get("fixture_id"))
        grouped[(str(row.get("source_family")), labels.get(fixture_id, "missing"))].append(
            fixture_id
        )
    return [
        {
            "source_family": family,
            "response_class": response_class,
            "row_count": len(ids),
            "fixture_ids": sorted(ids),
        }
        for (family, response_class), ids in sorted(grouped.items())
    ]


def class_stratum_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require three rows in each of four-family by two-class cells."""

    observed = {
        (str(row.get("source_family")), str(row.get("response_class"))): int(
            row.get("row_count", 0) or 0
        )
        for row in rows
    }
    expected = {(family, label): 3 for family in SOURCE_FAMILIES for label in CLASS_LABELS}
    return [] if observed == expected else ["class_strata_mismatch"]


def canary_evidence_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Require a healthy token-producing Qwen process with observed CUDA use."""

    errors: list[str] = []
    loads = list(artifact.get("model_load_receipts") or [])
    raw_rows = list(artifact.get("canary_raw_output_rows") or [])
    token_rows = list(artifact.get("canary_token_rows") or [])
    prompt_rows = list(artifact.get("canary_prompt_rows") or [])
    if len(loads) != 1:
        errors.append("canary_model_load_receipt_count")
    if len(prompt_rows) != 1:
        errors.append("canary_prompt_receipt_count")
    if len(raw_rows) != 1:
        errors.append("canary_raw_output_receipt_count")
    if len(token_rows) != 1:
        errors.append("canary_token_receipt_count")
    if not loads:
        return errors
    load = loads[0]
    if load.get("model_id") != QWEN_MODEL_ID:
        errors.append("canary_model_id_mismatch")
    if load.get("health", {}).get("ok") is not True:
        errors.append("canary_health_failed")
    numeric_offload = int(load.get("cuda_layers_offloaded", 0) or 0) > 0
    if not numeric_offload and load.get("gpu_offload_confirmed") is not True:
        errors.append("canary_cuda_layers_missing")
    if load.get("cuda_placement_confirmed") is not True:
        errors.append("canary_cuda_placement_unconfirmed")
    if load.get("cleanup", {}).get("leak_free") is not True:
        errors.append("canary_cleanup_failed")
    if raw_rows:
        raw_output = str(raw_rows[0].get("raw_output", ""))
        if not raw_output:
            errors.append("canary_raw_output_missing")
        if raw_rows[0].get("raw_output_sha256") != sha256_text(raw_output):
            errors.append("canary_raw_output_hash_mismatch")
    if token_rows and int(token_rows[0].get("completion_tokens", 0) or 0) <= 0:
        errors.append("canary_completion_tokens_missing")
    return list(dict.fromkeys(errors))


def terminal_evidence_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute readiness from row evidence instead of trusting the score."""

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
    if {row.get("mutation_id") for row in mutation_rows} != {
        row["mutation_id"] for row in blinding_mutation_rows()
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
    during = next((row for row in gpu_rows if row.get("phase") == "canary"), {})
    if before.get("ok") is not True or int(before.get("gpu_count", 0) or 0) < 2:
        errors.append("gpu_preflight_missing")
    if not any(
        app.get("owned_by_task") is True and int(app.get("used_memory_mb", 0) or 0) > 0
        for app in list(during.get("compute_apps") or [])
    ):
        errors.append("canary_owned_gpu_memory_missing")
    errors.extend(canary_evidence_errors(artifact))
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
    """Set readiness from all receipts while making no detection claim."""

    result = deepcopy(dict(artifact))
    copied_checks = [deepcopy(dict(row)) for row in checks]
    result["preconditions_checked"] = copied_checks
    result["duration_s"] = round(max(0.0, float(duration_s)), 6)
    result["gate_check_summary"] = _gate_summary(copied_checks)
    check_map = {str(row.get("check")): row.get("passed") for row in copied_checks}
    checks_pass = all(check_map.get(name) is True for name in READINESS_CHECKS)
    evidence_errors = terminal_evidence_errors(result) if checks_pass else []
    if checks_pass and not evidence_errors:
        result.update(
            {
                "inference_substrate_class": "model_full_generation",
                "grounding_preflight_ready_score": 1,
                "verdict_class": "positive",
                "honest_verdict": "positive_grounding_preflight_ready_no_verifier_value_claim",
            }
        )
    else:
        if result["gate_check_summary"]["passed"] and evidence_errors:
            copied_checks.append(gate_row("terminal_evidence", [], evidence_errors, False))
            result["preconditions_checked"] = copied_checks
            result["gate_check_summary"] = _gate_summary(copied_checks)
        failed = str(result["gate_check_summary"].get("failed_check") or "readiness_checks")
        result.update(
            {
                "inference_substrate_class": "blocked_no_run",
                "grounding_preflight_ready_score": 0,
                "verdict_class": "blocked",
                "honest_verdict": f"blocked_{failed}",
            }
        )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def _load_artifact_value(
    value: Mapping[str, Any] | str | Path | object,
) -> Mapping[str, Any] | None:
    """Load one artifact path or return a supplied mapping."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"__artifact_unreadable__": True}
        return loaded if isinstance(loaded, Mapping) else {"__artifact_not_object__": True}
    return {"__artifact_not_object__": True}


def validate_artifact(value: Mapping[str, Any] | str | Path | object) -> list[str]:
    """Cold-check shape, receipts, schedule, verdict, and checksum."""

    artifact = _load_artifact_value(value)
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
        not str(value).strip() for value in artifact.get("field_principles", {}).values()
    ):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if artifact.get("per_game_results") != []:
        errors.append("per_game_results_not_empty")
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
    if artifact.get("gate_check_summary") != _gate_summary(checks):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if verdict_class == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("grounding_preflight_ready_score") != 0:
            errors.append("blocked_readiness_score_mismatch")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked_gate_summary_mismatch")
    elif verdict_class == "positive":
        if artifact.get("inference_substrate_class") != "model_full_generation":
            errors.append("positive_substrate_class_mismatch")
        if artifact.get("grounding_preflight_ready_score") != 1:
            errors.append("positive_readiness_score_mismatch")
        if artifact.get("gate_check_summary", {}).get("passed") is not True:
            errors.append("positive_gate_summary_mismatch")
        errors.extend(terminal_evidence_errors(artifact))
    else:
        errors.append("terminal_verdict_class_invalid_for_preflight")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Flush one compact row so the outer runner can observe forward motion."""

    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def _run_subprocess(
    command: list[str], *, timeout_s: float = 15.0, phase: int = 3
) -> JsonDict:  # pragma: no cover
    """Run one bounded command with visible start and end receipts."""

    _progress(phase, "subprocess_start", command=command)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        row = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_s": time.perf_counter() - started,
        }
    except Exception as exc:
        row = {
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": time.perf_counter() - started,
        }
    _progress(phase, "subprocess_end", command=command, returncode=row["returncode"])
    return row


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover
    """Hash the exact fixture, prior failure, code, tests, and runtime helpers."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("results/experiment_7138_v627_relational_fixture.json"),
        Path("results/experiment_7139_v627_symbolic_grounding_ab.json"),
        Path("python/carnot/experiment_7138_v627_relational_fixture.py"),
        Path("python/carnot/experiment_7139_v627_symbolic_grounding_ab.py"),
        Path("python/carnot/experiment_7150_v628_grounding_preflight.py"),
        Path("python/carnot/inference/llama_server_supervisor.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("scripts/experiment_template.py"),
        Path("scripts/experiments/experiment_7150_v628_grounding_preflight.py"),
        Path("tests/python/test_experiment_7150_v628_grounding_preflight.py"),
    )
    return {
        str(path): sha256_file(root / path) if (root / path).is_file() else None for path in paths
    }


def _load_fixture(path: Path) -> tuple[JsonDict | None, list[JsonDict]]:  # pragma: no cover
    """Load the existing fixture and preserve its exact readiness checks."""

    if not path.is_file():
        return None, [gate_row("fixture_contract", {"exists": True}, {"exists": False}, False)]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [gate_row("fixture_contract", "valid_json_object", type(exc).__name__, False)]
    errors = validate_fixture_artifact(value) if isinstance(value, Mapping) else ["not_object"]
    observed = {
        "validation_errors": errors,
        "ready_score": value.get("source_grounding_fixture_ready_score")
        if isinstance(value, Mapping)
        else None,
        "fixture_row_count": value.get("fixture_row_count") if isinstance(value, Mapping) else None,
        "label_exposure_count": value.get("label_exposure_count")
        if isinstance(value, Mapping)
        else None,
    }
    expected = {
        "validation_errors": [],
        "ready_score": 1,
        "fixture_row_count": 72,
        "label_exposure_count": 0,
    }
    passed = observed == expected
    return (dict(value) if isinstance(value, Mapping) else None), [
        gate_row("fixture_contract", expected, observed, passed)
    ]


def _gpu_snapshot(phase_name: str, *, phase: int) -> JsonDict:  # pragma: no cover
    """Capture GPU devices and compute owners through two visible subprocesses."""

    gpu = _run_subprocess(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=15.0,
        phase=phase,
    )
    apps = _run_subprocess(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=15.0,
        phase=phase,
    )
    devices = []
    uuid_to_index: dict[str, int] = {}
    for line in str(gpu.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 7:
            index = int(parts[0])
            uuid_to_index[parts[1]] = index
            devices.append(
                {
                    "index": index,
                    "uuid": parts[1],
                    "name": parts[2],
                    "utilization_pct": int(float(parts[3])),
                    "memory_total_mb": int(float(parts[4])),
                    "memory_used_mb": int(float(parts[5])),
                    "memory_free_mb": int(float(parts[6])),
                }
            )
    owned = owned_process_ids()
    compute_apps = []
    for line in str(apps.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and parts[1].isdigit():
            pid = int(parts[1])
            compute_apps.append(
                {
                    "gpu_uuid": parts[0],
                    "gpu_index": uuid_to_index.get(parts[0]),
                    "pid": pid,
                    "process_name": parts[2],
                    "used_memory_mb": int(float(parts[3])),
                    "command": proc_cmdline(pid),
                    "owned_by_task": pid in owned,
                }
            )
    return {
        "phase": phase_name,
        "ok": gpu.get("returncode") == 0 and bool(devices),
        "gpu_count": len(devices),
        "devices": devices,
        "compute_apps": compute_apps,
        "command_receipts": {"gpu": gpu, "compute_apps": apps},
    }


def _free_port() -> int:  # pragma: no cover
    """Reserve one loopback port for the task-owned canary server."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(server: Path, model: Path, port: int) -> list[str]:  # pragma: no cover
    """Use the proven offline two-GPU llama-server command."""

    return [
        str(server),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--fit",
        "off",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--log-verbosity",
        "3",
    ]


def _wait_for_health(
    supervisor: NativeLlamaServerSupervisor, port: int, *, timeout_s: float
) -> JsonDict:  # pragma: no cover
    """Poll health with a visible heartbeat during a long model load."""

    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    next_heartbeat = time.monotonic()
    attempts = 0
    last_error = "not_started"
    while time.monotonic() < deadline:
        attempts += 1
        if supervisor.proc and supervisor.proc.poll() is not None:
            return {
                "ok": False,
                "attempts": attempts,
                "classification": "early_exit",
                "last_error": last_error,
                "duration_s": time.perf_counter() - started,
            }
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as response:
                return {
                    "ok": response.status == 200,
                    "attempts": attempts,
                    "classification": "healthy",
                    "status": response.status,
                    "duration_s": time.perf_counter() - started,
                }
        except (OSError, error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= next_heartbeat:
            _progress(
                7,
                "model_load_heartbeat",
                attempts=attempts,
                elapsed_s=round(time.perf_counter() - started, 3),
            )
            next_heartbeat = time.monotonic() + 60.0
        time.sleep(1.0)
    return {
        "ok": False,
        "attempts": attempts,
        "classification": "deadline_expired",
        "last_error": last_error,
        "duration_s": time.perf_counter() - started,
    }


def _canary_request(port: int) -> tuple[JsonDict, JsonDict]:  # pragma: no cover
    """Send one short deterministic chat call and retain the complete body."""

    prompt = "Reply with exactly PREFLIGHT_OK."
    payload = {
        "messages": [
            {"role": "system", "content": "Follow the user instruction exactly."},
            {"role": "user", "content": prompt},
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
    prompt_row = {
        "model_id": QWEN_MODEL_ID,
        "request": payload,
        "request_sha256": sha256_text(canonical_json(payload)),
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
    }
    response_row = {
        "raw_output": output,
        "raw_response": body,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "generation_duration_s": time.perf_counter() - started,
    }
    return prompt_row, response_row


def _run_canary(
    spec: Mapping[str, Any], *, server_path: Path, raw_dir: Path
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:  # pragma: no cover
    """Run one task-owned Qwen server and preserve all executed evidence."""

    port = _free_port()
    command = _server_command(server_path, Path(str(spec["model_path"])), port)
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
    prompt_row: JsonDict = {}
    response_row: JsonDict = {}
    gpu_row: JsonDict = {
        "phase": "canary",
        "ok": False,
        "gpu_count": 0,
        "devices": [],
        "compute_apps": [],
    }
    failure = None
    _progress(7, "subprocess_start", command=command)
    _progress(7, "model_load_start", model_id=QWEN_MODEL_ID)
    try:
        identity = supervisor.launch()
        health = _wait_for_health(supervisor, port, timeout_s=480.0)
        _progress(7, "model_load_end", model_id=QWEN_MODEL_ID, health=health.get("ok"))
        gpu_row = _gpu_snapshot("canary", phase=7)
        if health.get("ok") is not True:
            raise RuntimeError(f"server health failed: {health.get('classification')}")
        _progress(7, "generation_start", model_id=QWEN_MODEL_ID)
        prompt_row, response_row = _canary_request(port)
        _progress(
            7,
            "generation_end",
            model_id=QWEN_MODEL_ID,
            completion_tokens=response_row.get("completion_tokens"),
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _progress(7, "generation_end", model_id=QWEN_MODEL_ID, error=failure)
    finally:
        _progress(7, "subprocess_cleanup_start", pid=identity.get("pid"))
        cleanup = supervisor.cleanup()
        _progress(7, "subprocess_cleanup_end", leak_free=cleanup.get("leak_free"))
    process_returncode = supervisor.proc.poll() if supervisor.proc is not None else None
    _progress(7, "subprocess_end", command=command, returncode=process_returncode)
    server_log = (
        supervisor.log_path.read_text(encoding="utf-8", errors="replace")
        if supervisor.log_path.is_file()
        else ""
    )
    cuda = cuda_offload_receipt(
        server_log,
        gpu_row,
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
    prompt_row = {"model_id": QWEN_MODEL_ID, **prompt_row}
    raw_output = str(response_row.get("raw_output", ""))
    raw_row = {
        "model_id": QWEN_MODEL_ID,
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "raw_response": response_row.get("raw_response", {}),
        "raw_response_sha256": sha256_text(canonical_json(response_row.get("raw_response", {}))),
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
    return load_receipt, prompt_row, raw_row, token_row, gpu_row


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
    checks = [gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE)]
    artifact["source_artifact_hashes"] = _source_hashes(root)
    if checks[-1]["passed"] is not True:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(1, "phase_start", name="fixture_contract")
    fixture, fixture_checks = _load_fixture(fixture_path)
    checks.extend(fixture_checks)
    _progress(
        1,
        "phase_end",
        name="fixture_contract",
        passed=fixture is not None and fixture_checks[-1]["passed"],
    )
    if fixture is None or any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(2, "phase_start", name="typed_blinding_and_schedule")
    schedule = build_schedule(list(fixture["model_view_rows"]), list(fixture["fixture_rows"]))
    selected_models = {str(row["fixture_id"]): row for row in fixture["model_view_rows"]}
    blinding_errors = typed_blinding_errors(
        [selected_models[fixture_id] for fixture_id in FROZEN_FIXTURE_IDS]
    ) + typed_blinding_errors(schedule)
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
    _progress(2, "phase_end", name="typed_blinding_and_schedule", schedule_rows=len(schedule))
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(3, "phase_start", name="sealed_class_strata")
    class_rows = build_class_stratum_rows(schedule, list(fixture["sealed_scorer_rows"]))
    class_errors = class_stratum_errors(class_rows)
    artifact["class_stratum_rows"] = class_rows
    checks.append(gate_row("class_strata", [], class_errors, not class_errors))
    _checkpoint(result_path, artifact)
    _progress(3, "phase_end", name="sealed_class_strata", passed=not class_errors)
    if class_errors:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(4, "phase_start", name="cached_model_identity")
    specs = resolve_model_specs()
    _progress(4, "benchmark_start", name="model_file_hash_and_template")
    specs, identities, model_errors = model_identity_receipts(specs)
    _progress(4, "benchmark_end", name="model_file_hash_and_template", errors=model_errors)
    artifact["MODEL_SPECS"] = specs
    artifact["model_identity_rows"] = identities
    checks.extend(
        [
            gate_row(
                "cached_qwen_q4",
                [],
                [error for error in model_errors if error != "embedded_chat_template_missing"],
                not any(error != "embedded_chat_template_missing" for error in model_errors),
            ),
            gate_row(
                "embedded_chat_template",
                True,
                bool(identities and identities[0].get("template_present")),
                bool(identities and identities[0].get("template_present")),
            ),
        ]
    )
    _checkpoint(result_path, artifact)
    _progress(4, "phase_end", name="cached_model_identity", passed=not model_errors)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(5, "phase_start", name="native_binary_linkage")
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
    _progress(5, "phase_end", name="native_binary_linkage", passed=not linkage_errors)
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(6, "phase_start", name="gpu_availability")
    before_gpu = _gpu_snapshot("before", phase=6)
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
    _checkpoint(result_path, artifact)
    _progress(6, "phase_end", name="gpu_availability", passed=gpu_passed)
    if not gpu_passed:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(7, "phase_start", name="real_qwen_canary")
    load, prompt_row, raw_row, token_row, during_gpu = _run_canary(
        specs[0], server_path=server_path, raw_dir=raw_dir
    )
    artifact["gpu_rows"].append(during_gpu)
    artifact["model_load_receipts"] = [load]
    artifact["canary_prompt_rows"] = [prompt_row]
    artifact["canary_raw_output_rows"] = [raw_row]
    artifact["canary_token_rows"] = [token_row]
    canary_errors = canary_evidence_errors(artifact)
    checks.append(gate_row("real_qwen_canary", [], canary_errors, not canary_errors))
    _checkpoint(result_path, artifact)
    _progress(7, "phase_end", name="real_qwen_canary", passed=not canary_errors)
    if canary_errors:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(8, "phase_start", name="terminal_reduction")
    result = finalize_artifact(artifact, checks, duration_s=time.perf_counter() - started)
    write_artifact(result_path, result)
    _progress(
        8,
        "phase_end",
        name="terminal_reduction",
        readiness=result["grounding_preflight_ready_score"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the live preflight or cold-validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(9, "subprocess_start", name="artifact_validation", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(9, "subprocess_end", name="artifact_validation", valid=not errors)
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
                "grounding_preflight_ready_score": result.get("grounding_preflight_ready_score"),
            }
        ),
        flush=True,
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
