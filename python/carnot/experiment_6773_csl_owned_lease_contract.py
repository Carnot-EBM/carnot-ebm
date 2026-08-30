"""Validate the frozen memory stream and admit two local CUDA models.

This experiment proves an execution contract only. It does not measure memory
quality and does not claim continuous learning. Each live model runs in a
fresh process under an owner-bound GPU lease.

Spec refs: REQ-CL-6773, SCENARIO-CL-6773-*, REQ-INFRA-6773,
SCENARIO-INFRA-6773-*, REQ-REPORT-6773, and SCENARIO-REPORT-6773-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, TypedDict

from carnot import experiment_6752_arc_code_carrying_tool_preflight as exp6752
from carnot import experiment_6761_procedural_memory_stream as stream_mod
from carnot import experiment_6764_arc_exclusive_load_preflight as infra
from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "results/experiment_6761_procedural_memory_stream.json"
RESULT_PATH = REPO_ROOT / "results/experiment_6773_csl_owned_lease_contract.json"
WORK_DIR = REPO_ROOT / "results/.experiment_6773_csl_owned_lease_contract"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
MODULE_PATH = Path(__file__).resolve()
SCRIPT_PATH = REPO_ROOT / "scripts/experiments/experiment_6773_csl_owned_lease_contract.py"
TEST_PATH = REPO_ROOT / "tests/python/test_experiment_6773_csl_owned_lease_contract.py"

SCHEMA = "carnot.experiment_6773.csl_owned_lease_contract.v1"
EXPERIMENT_ID = "experiment_6773_csl_owned_lease_contract"
RUN_DATE = "20260830"
RANDOM_SEED = 6_773
INFERENCE_SUBSTRATE = "task-owned local llama.cpp CUDA GGUF preflight"
EXPECTED_SOURCE_ARTIFACT_SHA256 = (
    "sha256:e99699342f1edc1504f922498a5c18c21adea5a35241d36b1a1ce248bcca80b8"
)
EXPECTED_STREAM_HASH = "sha256:56dcae39d25dc5671a2c4b74cc44fb88b4386dfb8cff8d4b5f86496b1f7ff84f"
EXPECTED_GPU_UUIDS = infra.EXPECTED_GPU_UUIDS
FROZEN_FREE_VRAM_THRESHOLD_MB = 22_610
VRAM_RECOVERY_TOLERANCE_MB = 512
VRAM_RECOVERY_TIMEOUT_S = 180.0
WORKER_TIMEOUT_S = 1_200.0
RAM_AVAILABLE_FLOOR_BYTES = 64 * 1024**3
DISK_FREE_FLOOR_BYTES = 1024**3
CANARY_CONTEXT_TOKENS = 1_024
CANARY_PROMPT_MAX_BYTES = 4_096
COMPLETE_PHASE_SEQUENCE = lease_api.COMPLETE_PHASE_SEQUENCE


class TokenizerRecord(TypedDict):
    """Describe the embedded tokenizer that llama.cpp loads from one GGUF."""

    source: str
    loadable: bool
    detail: str


class ModelRecord(TypedDict):
    """Carry every model identity field shared by planning and live evidence."""

    model_id: str
    role: str
    family: str
    quantization: str
    revision: str
    filename: str
    model_path: str
    model_sha256: str
    model_size_bytes: int
    tokenizer: TokenizerRecord


PLANNED_MODELS: tuple[JsonDict, ...] = (
    {
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_acquisition_and_within_family",
        "family": "qwen_moe",
        "quantization": "Q4_K_M",
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "expected_sha256": "sha256:ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
    },
    {
        "model_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense_held_family_transfer",
        "family": "gemma_dense",
        "quantization": "Q4_K_M",
        "filename": "gemma-4-31B-it-Q4_K_M.gguf",
        "expected_sha256": "sha256:9fdf3dc8b0384830b4402d151388c140bd8eb2abf8d60588d8224231198254a1",
    },
)
MODEL_RECORD_FIELDS = {
    "model_id",
    "role",
    "family",
    "quantization",
    "revision",
    "filename",
    "model_path",
    "model_sha256",
    "model_size_bytes",
    "tokenizer",
}
TOKENIZER_RECORD_FIELDS = {"source", "loadable", "detail"}
STREAM_CHECK_NAMES = (
    "source_artifact_hash",
    "upstream_validator",
    "stream_ready",
    "stream_hash",
    "order_count",
    "capacity_contract",
    "read_only_episode",
    "transaction_schema",
    "restart_receipts",
    "rollback_receipts",
    "poison_receipts",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_sha256",
    "code_receipts",
    "model_specs",
    "models_used",
    "live_model_invoked",
    "rows",
    "gpu_receipts",
    "lease_receipts",
    "teardown_receipts",
    "stream_manifest",
    "stream_contract_checks",
    "preconditions_checked",
    "csl_live_preflight_ready",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned shape lets cold readers reject incompatible receipts.",
    "experiment_id": "The stable identifier binds the artifact to its owned producer.",
    "run_date": "The fixed planning date prevents silent protocol drift.",
    "status": "The status separates ready, partial, and blocked execution.",
    "field_principles": "Every artifact field states why it is required.",
    "inference_substrate": "The exact substrate excludes CPU, remote, and substituted inference.",
    "duration_s": "Monotonic task wall time makes the live work auditable.",
    "random_seed": "The fixed canary seed makes both bounded requests repeatable.",
    "reproducibility_checksum": "The hash binds stream, models, code, rows, and receipts.",
    "source_artifact_sha256": "The source hash binds the frozen Exp6761 bytes.",
    "code_receipts": "Code hashes identify the producer, wrapper, and contract tests.",
    "model_specs": "Planned typed identities are fixed before any model load.",
    "models_used": "Only exact planned records that produced live tokens appear here.",
    "live_model_invoked": "True requires bounded first-token evidence from both models.",
    "rows": "One row per stream check and lease phase prevents aggregate-only readiness.",
    "gpu_receipts": "Per-model CUDA, offload, token, and recovery evidence stays separate.",
    "lease_receipts": "Owner and release evidence defines who had authority over each GPU.",
    "teardown_receipts": "Process absence and VRAM recovery must precede the next load.",
    "stream_manifest": "Six frozen order identities prove that the stream was not regenerated.",
    "stream_contract_checks": "Independent checks cover capacity, transactions, and safety receipts.",
    "preconditions_checked": "Observed resource gates explain why workers did or did not start.",
    "csl_live_preflight_ready": "The Exp6774 gate opens only after every admission check passes.",
    "gate_check_summary": "Failed checks retain their expected and observed values.",
    "verifier_is_oracle": "False marks this as execution admission, not a correctness oracle.",
    "verdict_class": "A closed enum prevents admission evidence from becoming a learning claim.",
    "honest_verdict": "A terminal prefix makes the final execution state machine-readable.",
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for content-addressed receipts."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes and include the algorithm name in the value."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after stable serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks so large GGUFs are not copied into memory."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gpu_receipt_checksum(receipt: Mapping[str, Any]) -> str:
    """Hash one model receipt without its self-referential field."""

    return sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable stream, model, code, row, lease, and teardown evidence."""

    return sha256_json(
        {
            key: artifact.get(key)
            for key in (
                "schema",
                "experiment_id",
                "run_date",
                "random_seed",
                "source_artifact_sha256",
                "code_receipts",
                "model_specs",
                "models_used",
                "rows",
                "gpu_receipts",
                "lease_receipts",
                "teardown_receipts",
                "stream_manifest",
                "stream_contract_checks",
                "preconditions_checked",
                "csl_live_preflight_ready",
                "gate_check_summary",
                "verdict_class",
                "honest_verdict",
            )
        }
    )


def _revision_from_path(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-unversioned"


def _blank_model_record(planned: Mapping[str, Any]) -> ModelRecord:
    return {
        "model_id": str(planned["model_id"]),
        "role": str(planned["role"]),
        "family": str(planned["family"]),
        "quantization": str(planned["quantization"]),
        "revision": "missing",
        "filename": str(planned["filename"]),
        "model_path": "",
        "model_sha256": "missing",
        "model_size_bytes": 0,
        "tokenizer": {
            "source": "llama.cpp_embedded_gguf",
            "loadable": False,
            "detail": "model path unresolved",
        },
    }


def resolve_model_specs(
    *,
    pair_resolver: Callable[..., list[dict] | None] = cached_sota_pair,
    tokenizer_probe: Callable[[str | None], tuple[bool, str]] = gguf_tokenizer_loadable,
    file_hasher: Callable[[str | Path], str] = sha256_file,
) -> list[ModelRecord]:
    """Resolve the mandated Qwen and Gemma files through the shared cache helper."""

    cached = pair_resolver(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    by_id = {str(row.get("hf_id")): row for row in cached if isinstance(row, Mapping)}
    records: list[ModelRecord] = []
    for planned in PLANNED_MODELS:
        source = by_id.get(str(planned["model_id"]))
        if source is None:
            records.append(_blank_model_record(planned))
            continue
        raw_path = Path(str(source.get("model_path", "")))
        present = raw_path.is_file() and raw_path.name == planned["filename"]
        model_hash = file_hasher(raw_path) if present else "missing"
        tokenizer_ok, tokenizer_detail = tokenizer_probe(str(raw_path) if present else None)
        records.append(
            {
                "model_id": str(planned["model_id"]),
                "role": str(planned["role"]),
                "family": str(planned["family"]),
                "quantization": str(planned["quantization"]),
                "revision": _revision_from_path(raw_path) if present else "missing",
                "filename": str(planned["filename"]),
                "model_path": str(raw_path.resolve()) if present else "",
                "model_sha256": model_hash,
                "model_size_bytes": raw_path.stat().st_size if present else 0,
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": bool(tokenizer_ok),
                    "detail": str(tokenizer_detail),
                },
            }
        )
    return records


def model_record_errors(record: Mapping[str, Any], planned: Mapping[str, Any]) -> list[str]:
    """Reject a missing, renamed, extra, unresolved, or mismatched identity field."""

    errors: list[str] = []
    if set(record) != MODEL_RECORD_FIELDS:
        errors.append("field_set")
    identity = {
        "model_id": planned.get("model_id"),
        "role": planned.get("role"),
        "family": planned.get("family"),
        "quantization": planned.get("quantization"),
        "filename": planned.get("filename"),
        "model_sha256": planned.get("expected_sha256"),
    }
    for field, expected in identity.items():
        if record.get(field) != expected:
            errors.append(field)
    if not isinstance(record.get("revision"), str) or not record.get("revision"):
        errors.append("revision")
    if not isinstance(record.get("model_path"), str) or not record.get("model_path"):
        errors.append("model_path")
    if (
        not isinstance(record.get("model_size_bytes"), int)
        or record.get("model_size_bytes", 0) <= 0
    ):
        errors.append("model_size_bytes")
    tokenizer = record.get("tokenizer")
    tokenizer = tokenizer if isinstance(tokenizer, Mapping) else {}
    if (
        set(tokenizer) != TOKENIZER_RECORD_FIELDS
        or tokenizer.get("source") != "llama.cpp_embedded_gguf"
        or tokenizer.get("loadable") is not True
        or not isinstance(tokenizer.get("detail"), str)
        or not tokenizer.get("detail")
    ):
        errors.append("tokenizer")
    return list(dict.fromkeys(errors))


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
    }


def stream_contract_checks(
    fixture: Mapping[str, Any],
    *,
    source_artifact_sha256: str,
    upstream_validator_errors: Sequence[str],
) -> list[JsonDict]:
    """Revalidate the frozen stream from checked-in evidence without rebuilding it."""

    manifest = fixture.get("stream_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    orders = manifest.get("orders")
    orders = orders if isinstance(orders, list) else []
    capacity = fixture.get("capacity_contract")
    capacity = capacity if isinstance(capacity, Mapping) else {}
    arms = capacity.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    arm_values = list(arms.values())
    max_committed = capacity.get("max_committed_bytes_by_arm")
    max_committed = max_committed if isinstance(max_committed, Mapping) else {}
    ceiling = capacity.get("storage_ceiling_bytes")
    capacity_pass = (
        len(arm_values) == 2
        and arm_values[0] == arm_values[1]
        and isinstance(ceiling, int)
        and bool(max_committed)
        and all(isinstance(value, int) and value < ceiling for value in max_committed.values())
    )
    transaction = fixture.get("transaction_schema")
    transaction = transaction if isinstance(transaction, Mapping) else {}
    transaction_pass = (
        transaction.get("version") == 1
        and set(transaction.get("required_fields", []))
        == set(stream_mod.TRANSACTION_REQUIRED_FIELDS)
        and transaction.get("active_episode_policy") == "read_only"
        and transaction.get("commit_timing") == "after_exact_result_closes_episode"
    )
    restart = fixture.get("restart_receipts")
    restart = restart if isinstance(restart, list) else []
    rollback = fixture.get("rollback_receipts")
    rollback = rollback if isinstance(rollback, list) else []
    poison = fixture.get("poison_fixture_receipts")
    poison = poison if isinstance(poison, list) else []
    return [
        _check(
            "source_artifact_hash",
            EXPECTED_SOURCE_ARTIFACT_SHA256,
            source_artifact_sha256,
            source_artifact_sha256 == EXPECTED_SOURCE_ARTIFACT_SHA256,
        ),
        _check(
            "upstream_validator", [], list(upstream_validator_errors), not upstream_validator_errors
        ),
        _check(
            "stream_ready",
            {"procedural_memory_stream_ready": True, "future_evidence_violations": 0},
            {
                "procedural_memory_stream_ready": fixture.get("procedural_memory_stream_ready"),
                "future_evidence_violations": fixture.get("future_evidence_violations"),
            },
            fixture.get("procedural_memory_stream_ready") is True
            and fixture.get("future_evidence_violations") == 0,
        ),
        _check(
            "stream_hash",
            EXPECTED_STREAM_HASH,
            manifest.get("stream_hash"),
            manifest.get("stream_hash") == EXPECTED_STREAM_HASH,
        ),
        _check(
            "order_count",
            6,
            {"declared": fixture.get("order_count"), "manifest": len(orders)},
            fixture.get("order_count") == 6 and len(orders) == 6,
        ),
        _check("capacity_contract", "equal_and_unsaturated", capacity, capacity_pass),
        _check(
            "read_only_episode",
            True,
            fixture.get("read_only_episode_enforced"),
            fixture.get("read_only_episode_enforced") is True,
        ),
        _check(
            "transaction_schema", "Exp6761 transaction schema v1", transaction, transaction_pass
        ),
        _check(
            "restart_receipts",
            "all byte and hash matches",
            {"count": len(restart)},
            bool(restart)
            and all(
                isinstance(row, Mapping)
                and row.get("bytes_match") is True
                and row.get("hash_match") is True
                for row in restart
            ),
        ),
        _check(
            "rollback_receipts",
            "all inverse patches restore exact bytes",
            {"count": len(rollback)},
            bool(rollback)
            and all(
                isinstance(row, Mapping)
                and row.get("inverse_patch_applied") is True
                and row.get("byte_identical") is True
                for row in rollback
            ),
        ),
        _check(
            "poison_receipts",
            "all unsafe candidates reject for their intended reason",
            {"count": len(poison)},
            bool(poison)
            and all(
                isinstance(row, Mapping)
                and row.get("committed") is False
                and row.get("admission_reason") == row.get("intended_admission_reason")
                and row.get("state_hash") == row.get("parent_hash")
                for row in poison
            ),
        ),
    ]


def compact_stream_manifest(fixture: Mapping[str, Any]) -> JsonDict:
    """Copy only the frozen stream and order identities needed downstream."""

    manifest = fixture.get("stream_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    orders = manifest.get("orders")
    orders = orders if isinstance(orders, list) else []
    return {
        "stream_hash": manifest.get("stream_hash"),
        "frozen_before_dry_replay": manifest.get("frozen_before_dry_replay"),
        "order_count": len(orders),
        "orders": [
            {
                "order_id": row.get("order_id"),
                "order_hash": row.get("order_hash"),
                "event_ids": deepcopy(row.get("event_ids", [])),
            }
            for row in orders
            if isinstance(row, Mapping)
        ],
    }


def build_canary_prompt(fixture: Mapping[str, Any]) -> str:
    """Build one bounded public-fixture prompt with procedural memory context."""

    pairs = fixture.get("representation_pair_receipts")
    pairs = pairs if isinstance(pairs, list) else []
    first = pairs[0] if pairs and isinstance(pairs[0], Mapping) else {}
    representations = first.get("representations")
    representations = representations if isinstance(representations, Mapping) else {}
    lesson = representations.get("procedural_lesson")
    lesson = lesson if isinstance(lesson, Mapping) else {}
    payload = lesson.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    prompt = (
        "Memory branch admission canary. Use this frozen procedural lesson as context.\n"
        f"Abstract constraint: {payload.get('abstract_constraint', '')}\n"
        f"Applicability scope: {payload.get('applicability_scope', '')}\n"
        f"Repair procedure: {payload.get('repair_procedure', '')}\n"
        "Reply with one concise acknowledgement token."
    )
    if len(prompt.encode("utf-8")) > CANARY_PROMPT_MAX_BYTES:
        raise ValueError("canary prompt exceeds frozen byte budget")
    return prompt


def rank_eligible_devices(devices: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Use the established least-used fixed-RTX-3090 ranking contract."""

    return infra.rank_eligible_devices(devices, threshold_mb=FROZEN_FREE_VRAM_THRESHOLD_MB)


def select_device_before_load() -> JsonDict:
    """Refresh both fixed GPU identities and select the least-used eligible card."""

    inventory = infra.nvidia_smi_inventory()
    selection = rank_eligible_devices(inventory.get("devices", []))
    return {
        **selection,
        "inventory_commands": {
            "devices": deepcopy(inventory.get("device_query")),
            "processes": deepcopy(inventory.get("process_query")),
        },
    }


def _load_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def collect_preconditions(
    *,
    source_path: Path = SOURCE_PATH,
    date: str = RUN_DATE,
    model_resolver: Callable[[], list[ModelRecord]] = resolve_model_specs,
    inventory_fn: Callable[[], JsonDict] = infra.nvidia_smi_inventory,
    llama_receipt_fn: Callable[[], JsonDict] = infra._llama_cpp_receipt,
    port_picker: Callable[[int], list[int]] = infra.choose_free_ports,
    port_probe: Callable[[int], bool] = infra.port_is_free,
    resource_fn: Callable[[Path], JsonDict] = infra._host_resources,
    stream_validator: Callable[[Mapping[str, Any]], list[str]] = stream_mod.validate_artifact,
) -> JsonDict:
    """Check source, models, CUDA, leases, devices, ports, RAM, and disk first."""

    fixture = _load_json(source_path)
    source_hash = sha256_file(source_path)
    validator_errors = stream_validator(fixture) if fixture else ["source_artifact_unreadable"]
    stream_checks = stream_contract_checks(
        fixture,
        source_artifact_sha256=source_hash,
        upstream_validator_errors=validator_errors,
    )
    models = model_resolver()
    model_errors = [
        model_record_errors(row, planned)
        for row, planned in zip(models, PLANNED_MODELS, strict=False)
    ]
    models_pass = len(models) == len(PLANNED_MODELS) and all(not row for row in model_errors)
    inventory = inventory_fn()
    devices = inventory.get("devices", []) if isinstance(inventory, Mapping) else []
    selection = rank_eligible_devices(devices)
    identity_rows = {row.get("uuid"): row.get("name") for row in devices}
    expected_identities = {uuid: "NVIDIA GeForce RTX 3090" for uuid in EXPECTED_GPU_UUIDS}
    llama = llama_receipt_fn()
    ports = port_picker(len(PLANNED_MODELS))
    port_status = {str(port): port_probe(port) for port in ports}
    resources = resource_fn(REPO_ROOT)
    checks = [
        _check("planning_date_matches", True, date == RUN_DATE, date == RUN_DATE),
        _check(
            "frozen_stream_contract",
            True,
            {row["check"]: row["passed"] for row in stream_checks},
            all(row["passed"] is True for row in stream_checks),
        ),
        _check("models_resolved", True, model_errors, models_pass),
        _check(
            "two_fixed_rtx3090_identities",
            expected_identities,
            identity_rows,
            identity_rows == expected_identities,
        ),
        _check(
            "llama_cpp_cuda",
            True,
            llama,
            llama.get("exists") is True
            and llama.get("executable", True) is True
            and llama.get("cuda_linked") is True
            and llama.get("python_cuda_offload") is True,
        ),
        _check(
            "gpu_lease_api",
            True,
            {
                name: callable(getattr(lease_api.GpuLease, name, None))
                for name in ("acquire", "transition", "release")
            },
            all(
                callable(getattr(lease_api.GpuLease, name, None))
                for name in ("acquire", "transition", "release")
            ),
        ),
        _check(
            "least_used_eligible_rtx3090",
            {"free_vram_mb_at_least": FROZEN_FREE_VRAM_THRESHOLD_MB},
            selection.get("selected_device"),
            selection.get("selected_device") is not None,
        ),
        _check(
            "ports_available",
            len(PLANNED_MODELS),
            port_status,
            len(ports) == len(PLANNED_MODELS) and all(port_status.values()),
        ),
        _check(
            "ram_and_disk",
            {
                "ram_available_bytes_at_least": RAM_AVAILABLE_FLOOR_BYTES,
                "disk_free_bytes_at_least": DISK_FREE_FLOOR_BYTES,
            },
            resources,
            int(resources.get("ram_available_bytes", 0)) >= RAM_AVAILABLE_FLOOR_BYTES
            and int(resources.get("disk_free_bytes", 0)) >= DISK_FREE_FLOOR_BYTES,
        ),
    ]
    return {
        "all_passed": all(row["passed"] is True for row in checks),
        "checks": checks,
        "models": models,
        "stream_fixture": fixture,
        "stream_contract_checks": stream_checks,
        "source_artifact_sha256": source_hash,
        "device_inventory_before": deepcopy(devices),
        "device_inventory_commands": {
            "devices": deepcopy(inventory.get("device_query")),
            "processes": deepcopy(inventory.get("process_query")),
        },
        "device_selection_receipt": selection,
        "ports": ports,
        "llama_cpp": llama,
        "resources": resources,
    }


def stream_rows(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert each frozen stream check into one auditable row."""

    return [
        {
            "row_kind": "stream_contract",
            "ordinal": ordinal,
            "check": row.get("check"),
            "expected": deepcopy(row.get("expected")),
            "observed": deepcopy(row.get("observed")),
            "passed": row.get("passed") is True,
        }
        for ordinal, row in enumerate(checks)
    ]


def phase_rows_for_receipt(receipt: Mapping[str, Any]) -> list[JsonDict]:
    """Bind every lease phase to its exact model and owner identity."""

    owner = receipt.get("lease_owner")
    owner = owner if isinstance(owner, Mapping) else {}
    rows: list[JsonDict] = []
    for ordinal, event in enumerate(receipt.get("phase_history", [])):
        if not isinstance(event, Mapping):
            continue
        rows.append(
            {
                "row_kind": "model_phase",
                "model_id": receipt.get("model_id"),
                "ordinal": ordinal,
                "phase": event.get("phase"),
                "previous_phase": event.get("previous_phase"),
                "monotonic_ns": event.get("monotonic_ns"),
                "event_checksum": event.get("event_checksum"),
                "owner_pid": owner.get("pid"),
                "owner_pid_start_ticks": owner.get("pid_start_ticks"),
                "device_uuid": owner.get("device_uuid"),
                "passed": bool(event.get("phase") in lease_api.PHASES),
            }
        )
    return rows


def build_vram_recovery_receipt(
    before_used_mb: int, after_used_mb: int, owned_pid_present: bool
) -> JsonDict:
    """Prove the owned PID is absent and total device use returned to baseline."""

    delta = abs(int(after_used_mb) - int(before_used_mb))
    return {
        "before_used_mb": int(before_used_mb),
        "after_used_mb": int(after_used_mb),
        "absolute_delta_mb": delta,
        "tolerance_mb": VRAM_RECOVERY_TOLERANCE_MB,
        "owned_pid_present": bool(owned_pid_present),
        "passed": not owned_pid_present and delta <= VRAM_RECOVERY_TOLERANCE_MB,
    }


def _process_identity(pid: int | None = None) -> JsonDict:
    observed_pid = os.getpid() if pid is None else int(pid)
    try:
        executable = os.readlink(f"/proc/{observed_pid}/exe")
    except OSError:
        executable = ""
    return {
        "pid": observed_pid,
        "pid_start_ticks": lease_api.proc_start_ticks(observed_pid),
        "executable": executable,
        "exit_code": None,
        "absent_after_exit": False,
    }


def _gpu_snapshot(device_uuid: str, owned_pid: int = 0) -> JsonDict:
    inventory = infra.nvidia_smi_inventory()
    device = next(
        (row for row in inventory.get("devices", []) if row.get("uuid") == device_uuid), {}
    )
    active = device.get("active_compute_processes")
    active = active if isinstance(active, list) else []
    owned = next((row for row in active if row.get("pid") == owned_pid), None)
    return {
        **deepcopy(device),
        "owned_pid": int(owned_pid),
        "owned_pid_present": owned is not None,
        "owned_pid_vram_mb": int((owned or {}).get("used_memory_mb", 0) or 0),
        "observed_monotonic_ns": time.monotonic_ns(),
    }


def _terminalize_lease(lease: Any, complete: bool, after: Mapping[str, Any]) -> JsonDict:
    phase = lease.document.get("phase")
    if phase in {"resident", "inferencing"}:
        lease.transition("unloading")
        phase = "unloading"
    if phase == "unloading":
        lease.transition(
            "validating",
            vram_mb=int(after.get("memory_used_mb", 0) or 0),
            exit_code=0 if complete else 1,
            unload_observed=True,
        )
        phase = "validating"
    if phase in {"preflight", "admitted", "loading"}:
        lease.transition("terminal_blocked")
    elif phase == "validating":
        lease.transition("terminal_complete" if complete else "terminal_blocked")
    if lease.document.get("phase") in lease_api.TERMINAL_PHASES:
        return dict(lease.release())
    lease.close()
    return {}


def run_live_model_worker(
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    *,
    prompt: str,
    lease_runtime_dir: Path = LEASE_RUNTIME_DIR,
    llama_factory: Callable[..., Any] | None = None,
    lease_factory: Callable[..., Any] = lease_api.GpuLease.acquire,
    snapshot_fn: Callable[[str, int], JsonDict] = _gpu_snapshot,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    """Load, decode one token, unload, and release one owner-bound lease."""

    started_ns = time.monotonic_ns()
    device_uuid = str(selected_device["uuid"])
    worker = _process_identity()
    owned_pid = int(worker["pid"])
    before = snapshot_fn(device_uuid, owned_pid)
    unrelated = [
        deepcopy(row)
        for row in before.get("active_compute_processes", [])
        if row.get("pid") != owned_pid
    ]
    lease: Any = None
    llm: Any = None
    owner: JsonDict = {}
    release: JsonDict = {}
    errors: list[str] = []
    resident = deepcopy(before)
    peak_vram = 0
    first_token = ""
    completion_tokens = 0
    after = deepcopy(before)
    recovery = build_vram_recovery_receipt(
        int(before.get("memory_used_mb", 0) or 0),
        int(before.get("memory_used_mb", 0) or 0),
        False,
    )
    try:
        device_recheck_passed = (
            before.get("uuid") == selected_device.get("uuid")
            and before.get("uuid") in EXPECTED_GPU_UUIDS
            and before.get("name") == "NVIDIA GeForce RTX 3090"
            and int(before.get("memory_free_mb", 0) or 0) >= FROZEN_FREE_VRAM_THRESHOLD_MB
        )
        if not device_recheck_passed:
            raise RuntimeError("selected_device_recheck_failed")
        lease = lease_factory(
            runtime_dir=lease_runtime_dir,
            task_id=f"exp6773-{model['model_id'].split('/')[-1]}",
            device_uuid=device_uuid,
            expected_model=str(model["model_path"]),
            vram_before_mb=int(before.get("memory_used_mb", 0) or 0),
            ttl_s=WORKER_TIMEOUT_S,
        )
        owner = dict(lease.owner_receipt())
        lease.transition("admitted")
        lease.transition("loading")
        if llama_factory is None:
            from llama_cpp import Llama  # noqa: PLC0415

            llama_factory = Llama
        llm = llama_factory(
            model_path=str(model["model_path"]),
            n_ctx=CANARY_CONTEXT_TOKENS,
            n_batch=512,
            n_gpu_layers=-1,
            main_gpu=0,
            seed=RANDOM_SEED,
            verbose=True,
        )
        resident = snapshot_fn(device_uuid, owned_pid)
        peak_vram = int(resident.get("owned_pid_vram_mb", 0) or 0)
        if resident.get("owned_pid_present") is not True or peak_vram <= 0:
            raise RuntimeError("owner_bound_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0) or 0))
        lease.transition("inferencing")
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            seed=RANDOM_SEED,
            stream=False,
        )
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        first_token = str((choices[0] if choices else {}).get("text", ""))
        usage = result.get("usage", {}) if isinstance(result, Mapping) else {}
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        sampled = snapshot_fn(device_uuid, owned_pid)
        peak_vram = max(peak_vram, int(sampled.get("owned_pid_vram_mb", 0) or 0))
        if completion_tokens < 1:
            raise RuntimeError("first_token_not_observed")
    except Exception as exc:  # noqa: BLE001 - live failures belong in the receipt
        errors.append(f"{type(exc).__name__}: {exc}"[:500])
    finally:
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
        if llm is not None:
            try:
                llm.close()
            except Exception as exc:  # noqa: BLE001 - teardown errors must remain visible
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
            llm = None
            gc.collect()
        deadline = time.monotonic() + VRAM_RECOVERY_TIMEOUT_S
        after = snapshot_fn(device_uuid, owned_pid)
        recovery = build_vram_recovery_receipt(
            int(before.get("memory_used_mb", 0) or 0),
            int(after.get("memory_used_mb", 0) or 0),
            bool(after.get("owned_pid_present")),
        )
        while not recovery["passed"] and time.monotonic() < deadline:
            sleep_fn(1.0)
            after = snapshot_fn(device_uuid, owned_pid)
            recovery = build_vram_recovery_receipt(
                int(before.get("memory_used_mb", 0) or 0),
                int(after.get("memory_used_mb", 0) or 0),
                bool(after.get("owned_pid_present")),
            )
        complete = bool(not errors and completion_tokens >= 1 and recovery["passed"])
        if lease is not None:
            try:
                release = _terminalize_lease(lease, complete, after)
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
                lease.close()
    history = deepcopy(lease.document.get("phase_history", [])) if lease is not None else []
    receipt: JsonDict = {
        "model_record": deepcopy(dict(model)),
        "model_id": model.get("model_id"),
        "device": deepcopy(dict(selected_device)),
        "unrelated_process_inventory": unrelated,
        "lease_owner": owner,
        "phase_history": history,
        "lease_release": release,
        "gpu_layers": {"requested": -1, "offloaded": 0, "total": None},
        "offload_full": False,
        "resident_owned_vram_mb": int(resident.get("owned_pid_vram_mb", 0) or 0),
        "peak_owned_vram_mb": peak_vram,
        "first_token_canary": {
            "fixture_event_id": "a01",
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "first_token_observed": completion_tokens >= 1,
            "completion_tokens": completion_tokens,
            "first_token_sha256": sha256_bytes(first_token.encode("utf-8")),
            "bounded": completion_tokens <= 1,
        },
        "worker_process": worker,
        "vram_recovery": recovery,
        "duration_s": round((time.monotonic_ns() - started_ns) / 1_000_000_000, 6),
        "unrelated_processes_signaled": [],
        "errors": errors,
    }
    receipt["receipt_sha256"] = gpu_receipt_checksum(receipt)
    return receipt


def gpu_receipt_errors(receipt: Mapping[str, Any], model: Mapping[str, Any]) -> list[str]:
    """Return every reason one receipt cannot satisfy live admission."""

    errors: list[str] = []
    if receipt.get("receipt_sha256") != gpu_receipt_checksum(receipt):
        errors.append("receipt_sha256")
    if receipt.get("model_record") != model or receipt.get("model_id") != model.get("model_id"):
        errors.append("model_record")
    planned = next((row for row in PLANNED_MODELS if row["model_id"] == model.get("model_id")), {})
    if model_record_errors(model, planned):
        errors.append("model_identity")
    device = receipt.get("device")
    device = device if isinstance(device, Mapping) else {}
    if device.get("uuid") not in EXPECTED_GPU_UUIDS:
        errors.append("device_uuid")
    owner = receipt.get("lease_owner")
    owner = owner if isinstance(owner, Mapping) else {}
    worker = receipt.get("worker_process")
    worker = worker if isinstance(worker, Mapping) else {}
    if not (
        owner.get("pid") == worker.get("pid")
        and owner.get("pid_start_ticks") == worker.get("pid_start_ticks")
        and owner.get("device_uuid") == device.get("uuid")
        and owner.get("expected_model") == model.get("model_path")
    ):
        errors.append("lease_owner")
    sequence = [row.get("phase") for row in receipt.get("phase_history", [])]
    if sequence != list(COMPLETE_PHASE_SEQUENCE):
        errors.append("phase_sequence")
    release = receipt.get("lease_release")
    release = release if isinstance(release, Mapping) else {}
    if release.get("released") is not True or release.get("phase") != "terminal_complete":
        errors.append("lease_release")
    layers = receipt.get("gpu_layers")
    layers = layers if isinstance(layers, Mapping) else {}
    if (
        receipt.get("offload_full") is not True
        or not isinstance(layers.get("offloaded"), int)
        or layers.get("offloaded", 0) <= 0
        or layers.get("offloaded") != layers.get("total")
    ):
        errors.append("offload_full")
    if int(receipt.get("resident_owned_vram_mb", 0) or 0) <= 0:
        errors.append("resident_owned_vram_mb")
    if int(receipt.get("peak_owned_vram_mb", 0) or 0) <= 0:
        errors.append("peak_owned_vram_mb")
    canary = receipt.get("first_token_canary")
    canary = canary if isinstance(canary, Mapping) else {}
    if not (
        canary.get("first_token_observed") is True
        and canary.get("completion_tokens") == 1
        and canary.get("bounded") is True
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(canary.get("prompt_sha256", "")))
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(canary.get("first_token_sha256", "")))
    ):
        errors.append("first_token_canary")
    if worker.get("exit_code") != 0 or worker.get("absent_after_exit") is not True:
        errors.append("worker_process")
    if (receipt.get("vram_recovery") or {}).get("passed") is not True:
        errors.append("vram_recovery")
    if receipt.get("unrelated_processes_signaled") != []:
        errors.append("unrelated_processes_signaled")
    if receipt.get("errors") != []:
        errors.append("errors")
    return list(dict.fromkeys(errors))


def _wait_parent_recovery(
    device_uuid: str,
    worker_pid: int,
    before_used_mb: int,
    *,
    timeout_s: float = VRAM_RECOVERY_TIMEOUT_S,
) -> JsonDict:
    deadline = time.monotonic() + timeout_s
    snapshot = _gpu_snapshot(device_uuid, worker_pid)
    receipt = build_vram_recovery_receipt(
        before_used_mb,
        int(snapshot.get("memory_used_mb", 0) or 0),
        bool(snapshot.get("owned_pid_present")),
    )
    while not receipt["passed"] and time.monotonic() < deadline:
        time.sleep(1.0)
        snapshot = _gpu_snapshot(device_uuid, worker_pid)
        receipt = build_vram_recovery_receipt(
            before_used_mb,
            int(snapshot.get("memory_used_mb", 0) or 0),
            bool(snapshot.get("owned_pid_present")),
        )
    receipt["observed_monotonic_ns"] = snapshot.get("observed_monotonic_ns")
    return receipt


def worker_environment(
    base: Mapping[str, str], model: Mapping[str, Any], selected_device: Mapping[str, Any]
) -> dict[str, str]:
    """Expose only the selected physical GPU and bind expected worker identity."""

    env = dict(base)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selected_device["index"]),
            "CARNOT_CSL_EXPECTED_GPU_UUID": str(selected_device["uuid"]),
            "CARNOT_CSL_EXPECTED_MODEL": str(model["model_path"]),
            "PYTHONPATH": str(REPO_ROOT / "python")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    return env


def _write_json_unchecked(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    stream_mod.atomic_write(path, data)


def _blocked_worker_receipt(
    model: Mapping[str, Any], device: Mapping[str, Any], error: str
) -> JsonDict:
    row: JsonDict = {
        "model_record": deepcopy(dict(model)),
        "model_id": model.get("model_id"),
        "device": deepcopy(dict(device)),
        "unrelated_process_inventory": [],
        "lease_owner": {},
        "phase_history": [],
        "lease_release": {},
        "gpu_layers": {"requested": -1, "offloaded": 0, "total": None},
        "offload_full": False,
        "resident_owned_vram_mb": 0,
        "peak_owned_vram_mb": 0,
        "first_token_canary": {
            "fixture_event_id": "a01",
            "prompt_sha256": "missing",
            "first_token_observed": False,
            "completion_tokens": 0,
            "first_token_sha256": "missing",
            "bounded": True,
        },
        "worker_process": {
            "pid": 0,
            "pid_start_ticks": None,
            "exit_code": 127,
            "absent_after_exit": True,
        },
        "vram_recovery": build_vram_recovery_receipt(0, 0, False),
        "duration_s": 0.0,
        "unrelated_processes_signaled": [],
        "errors": [error],
    }
    row["receipt_sha256"] = gpu_receipt_checksum(row)
    return row


def run_model_worker(
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    prompt: str,
    runtime_dir: Path,
    *,
    timeout_s: float = WORKER_TIMEOUT_S,
) -> JsonDict:
    """Run one fresh worker, then prove its absence and parent-observed recovery."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(model["model_id"])).strip("-").lower()
    model_path = runtime_dir / f"{slug}.model.json"
    device_path = runtime_dir / f"{slug}.device.json"
    prompt_path = runtime_dir / f"{slug}.prompt.json"
    output_path = runtime_dir / f"{slug}.receipt.json"
    _write_json_unchecked(model_path, model)
    _write_json_unchecked(device_path, selected_device)
    _write_json_unchecked(prompt_path, {"prompt": prompt})
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6773_csl_owned_lease_contract",
        "--worker",
        "--worker-model",
        str(model_path),
        "--worker-device",
        str(device_path),
        "--worker-prompt",
        str(prompt_path),
        "--worker-output",
        str(output_path),
        "--lease-runtime-dir",
        str(LEASE_RUNTIME_DIR),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=worker_environment(os.environ, model, selected_device),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    start_ticks = lease_api.proc_start_ticks(process.pid)
    timeout_cleanup: JsonDict = {}
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timeout_cleanup = infra._terminate_worker_group(process, start_ticks)
        stdout, stderr = process.communicate()
    receipt = _load_json(output_path)
    if not receipt:
        receipt = _blocked_worker_receipt(
            model,
            selected_device,
            f"worker_output_missing:exit={process.returncode}:stderr={sha256_bytes(stderr.encode())}",
        )
    worker = receipt.get("worker_process")
    worker = dict(worker) if isinstance(worker, Mapping) else {}
    worker.update(
        {
            "pid": int(process.pid),
            "pid_start_ticks": start_ticks,
            "exit_code": process.returncode,
            "absent_after_exit": process.poll() is not None,
            "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
            "timeout_cleanup": timeout_cleanup,
        }
    )
    receipt["worker_process"] = worker
    layers = exp6752._gpu_layers_from_log(stderr, -1)
    receipt["gpu_layers"] = layers
    receipt["offload_full"] = bool(
        isinstance(layers.get("offloaded"), int)
        and layers.get("offloaded", 0) > 0
        and layers.get("offloaded") == layers.get("total")
    )
    receipt["llama_cpp_log_sha256"] = sha256_bytes(stderr.encode("utf-8"))
    receipt["vram_recovery"] = _wait_parent_recovery(
        str(selected_device["uuid"]),
        int(process.pid),
        int(selected_device.get("memory_used_mb", 0) or 0),
    )
    receipt["receipt_sha256"] = gpu_receipt_checksum(receipt)
    return receipt


def code_receipts(paths: Sequence[Path] = (MODULE_PATH, SCRIPT_PATH, TEST_PATH)) -> JsonDict:
    """Hash the producer, repository wrapper, and focused contract tests."""

    return {str(path): sha256_file(path) for path in paths}


def _models_used(
    models: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    used_ids = [
        row.get("model_id")
        for row in receipts
        if (row.get("first_token_canary") or {}).get("first_token_observed") is True
    ]
    return [deepcopy(dict(model)) for model in models if model.get("model_id") in used_ids]


def _lease_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "model_id": row.get("model_id"),
            "owner": deepcopy(row.get("lease_owner")),
            "release": deepcopy(row.get("lease_release")),
        }
        for row in receipts
    ]


def _teardown_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "model_id": row.get("model_id"),
            "worker_process": deepcopy(row.get("worker_process")),
            "vram_recovery": deepcopy(row.get("vram_recovery")),
            "unrelated_processes_signaled": deepcopy(row.get("unrelated_processes_signaled")),
        }
        for row in receipts
    ]


def _gate_summary(
    preconditions: Mapping[str, Any],
    stream_checks: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows = [deepcopy(dict(row)) for row in preconditions.get("checks", [])]
    rows.extend(deepcopy(dict(row)) for row in preconditions.get("runtime_device_checks", []))
    rows.extend(deepcopy(dict(row)) for row in stream_checks)
    if preconditions.get("all_passed") is True:
        for model in models:
            receipt = next(
                (row for row in receipts if row.get("model_id") == model.get("model_id")), None
            )
            failures = (
                ["receipt_missing"] if receipt is None else gpu_receipt_errors(receipt, model)
            )
            rows.append(
                _check(
                    f"model_lifecycle:{model.get('model_id')}",
                    "complete full-offload token, teardown, release, and recovery",
                    {"failures": failures},
                    not failures,
                )
            )
    failures = [
        {
            "check": row.get("check"),
            "expected": deepcopy(row.get("expected")),
            "observed": deepcopy(row.get("observed")),
        }
        for row in rows
        if row.get("passed") is not True
    ]
    return {
        "checks": {str(row.get("check")): row.get("passed") is True for row in rows},
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _preconditions_for_artifact(preconditions: Mapping[str, Any]) -> JsonDict:
    return {
        key: deepcopy(value)
        for key, value in preconditions.items()
        if key not in {"stream_fixture", "models", "stream_contract_checks"}
    }


def _expected_state(
    preconditions: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    stream_checks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    models_used = _models_used(models, receipts)
    live = bool(models_used)
    runtime_device_checks = preconditions.get("runtime_device_checks", [])
    runtime_devices_passed = all(
        isinstance(row, Mapping) and row.get("passed") is True for row in runtime_device_checks
    )
    ready = bool(
        preconditions.get("all_passed") is True
        and runtime_devices_passed
        and len(models) == len(PLANNED_MODELS)
        and models_used == list(models)
        and all(row.get("passed") is True for row in stream_checks)
        and len(receipts) == len(models)
        and all(
            not gpu_receipt_errors(receipt, model)
            for receipt, model in zip(receipts, models, strict=True)
        )
    )
    if ready:
        return {
            "ready": True,
            "live": live,
            "models_used": models_used,
            "status": "complete",
            "verdict_class": "positive",
            "honest_verdict": "complete_csl_live_preflight_ready",
        }
    if preconditions.get("all_passed") is not True:
        return {
            "ready": False,
            "live": False,
            "models_used": models_used,
            "status": "blocked",
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_csl_owned_lease_contract",
        }
    return {
        "ready": False,
        "live": live,
        "models_used": models_used,
        "status": "partial",
        "verdict_class": "partial",
        "honest_verdict": "complete_partial_csl_owned_lease_contract",
    }


def build_artifact(
    *,
    date: str,
    preconditions: Mapping[str, Any],
    gpu_receipts: Sequence[Mapping[str, Any]],
    code_receipts: Mapping[str, Any],
    started_ns: int,
    finished_ns: int,
) -> JsonDict:
    """Reduce preconditions and model receipts into one cold-valid admission result."""

    models = [deepcopy(dict(row)) for row in preconditions.get("models", [])]
    receipts = [deepcopy(dict(row)) for row in gpu_receipts]
    fixture = preconditions.get("stream_fixture")
    fixture = fixture if isinstance(fixture, Mapping) else {}
    checks = [deepcopy(dict(row)) for row in preconditions.get("stream_contract_checks", [])]
    state = _expected_state(preconditions, models, receipts, checks)
    rows = stream_rows(checks) + [
        row for receipt in receipts for row in phase_rows_for_receipt(receipt)
    ]
    gates = _gate_summary(preconditions, checks, models, receipts)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(date),
        "status": state["status"],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0, int(finished_ns) - int(started_ns)) / 1_000_000_000, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_sha256": preconditions.get("source_artifact_sha256", "missing"),
        "code_receipts": deepcopy(dict(code_receipts)),
        "model_specs": models,
        "models_used": state["models_used"],
        "live_model_invoked": state["live"],
        "rows": rows,
        "gpu_receipts": receipts,
        "lease_receipts": _lease_receipts(receipts),
        "teardown_receipts": _teardown_receipts(receipts),
        "stream_manifest": compact_stream_manifest(fixture),
        "stream_contract_checks": checks,
        "preconditions_checked": _preconditions_for_artifact(preconditions),
        "csl_live_preflight_ready": state["ready"],
        "gate_check_summary": gates,
        "verifier_is_oracle": False,
        "verdict_class": state["verdict_class"],
        "honest_verdict": state["honest_verdict"],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-recompute every identity, row, readiness, verdict, and checksum field."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_field_set")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        errors.append("duration_s")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    models = artifact.get("model_specs")
    models = models if isinstance(models, list) else []
    model_schema_errors = len(models) != len(PLANNED_MODELS) or any(
        model_record_errors(model, planned)
        for model, planned in zip(models, PLANNED_MODELS, strict=False)
    )
    if model_schema_errors:
        errors.append("model_specs")
    receipts = artifact.get("gpu_receipts")
    receipts = receipts if isinstance(receipts, list) else []
    expected_used = _models_used(models, receipts)
    if artifact.get("models_used") != expected_used:
        errors.append("models_used")
    stream_checks = artifact.get("stream_contract_checks")
    stream_checks = stream_checks if isinstance(stream_checks, list) else []
    expected_rows = stream_rows(stream_checks) + [
        row for receipt in receipts for row in phase_rows_for_receipt(receipt)
    ]
    if artifact.get("rows") != expected_rows:
        errors.append("rows")
    if artifact.get("lease_receipts") != _lease_receipts(receipts):
        errors.append("lease_receipts")
    if artifact.get("teardown_receipts") != _teardown_receipts(receipts):
        errors.append("teardown_receipts")
    preconditions = artifact.get("preconditions_checked")
    preconditions = preconditions if isinstance(preconditions, Mapping) else {}
    rebuilt_preconditions = {
        **deepcopy(dict(preconditions)),
        "models": models,
        "stream_contract_checks": stream_checks,
    }
    state = _expected_state(rebuilt_preconditions, models, receipts, stream_checks)
    if artifact.get("csl_live_preflight_ready") is not state["ready"]:
        errors.append("csl_live_preflight_ready")
    if artifact.get("live_model_invoked") is not state["live"]:
        errors.append("live_model_invoked")
    for field in ("status", "verdict_class", "honest_verdict"):
        if artifact.get(field) != state[field]:
            errors.append(field)
    expected_gates = _gate_summary(rebuilt_preconditions, stream_checks, models, receipts)
    if artifact.get("gate_check_summary") != expected_gates:
        errors.append("gate_check_summary")
    if artifact.get("source_artifact_sha256") != preconditions.get(
        "source_artifact_sha256", artifact.get("source_artifact_sha256")
    ):
        errors.append("source_artifact_sha256")
    if state["ready"]:
        manifest = artifact.get("stream_manifest")
        manifest = manifest if isinstance(manifest, Mapping) else {}
        if (
            manifest.get("stream_hash") != EXPECTED_STREAM_HASH
            or manifest.get("order_count") != 6
            or len(manifest.get("orders", [])) != 6
        ):
            errors.append("stream_manifest")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and publish one complete JSON object through atomic replacement."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp6773 artifact: " + ",".join(errors))
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    receipt = stream_mod.atomic_write(path, data)
    return {"path": str(path), "atomic_rename": receipt["rename"], "sha256": sha256_file(path)}


def run(
    *,
    result_path: Path = RESULT_PATH,
    date: str = RUN_DATE,
    preflight_fn: Callable[[], JsonDict] = collect_preconditions,
    device_selector: Callable[[], JsonDict] = select_device_before_load,
    worker_runner: Callable[
        [Mapping[str, Any], Mapping[str, Any], str, Path], JsonDict
    ] = run_model_worker,
    code_receipt_fn: Callable[[], JsonDict] = code_receipts,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run preconditions, then two leased workers in fixed sequence."""

    started_ns = clock()
    preflight = preflight_fn()
    preflight.setdefault("runtime_device_checks", [])
    receipts: list[JsonDict] = []
    if preflight.get("all_passed") is True:
        fixture = preflight.get("stream_fixture")
        fixture = fixture if isinstance(fixture, Mapping) else {}
        prompt = build_canary_prompt(fixture)
        runtime_dir = result_path.parent / ".experiment_6773_csl_owned_lease_contract"
        for model in preflight.get("models", []):
            selection = device_selector()
            selected = selection.get("selected_device")
            preflight["runtime_device_checks"].append(
                _check(
                    f"device_recheck:{model.get('model_id')}",
                    {"free_vram_mb_at_least": FROZEN_FREE_VRAM_THRESHOLD_MB},
                    selection,
                    isinstance(selected, Mapping),
                )
            )
            if isinstance(selected, Mapping):
                receipt = worker_runner(model, selected, prompt, runtime_dir)
                receipts.append(receipt)
                if gpu_receipt_errors(receipt, model):
                    break
            else:
                break
    artifact = build_artifact(
        date=date,
        preconditions=preflight,
        gpu_receipts=receipts,
        code_receipts=code_receipt_fn(),
        started_ns=started_ns,
        finished_ns=clock(),
    )
    write_artifact(result_path, artifact)
    return artifact


def _worker_entry(
    model_path: Path,
    device_path: Path,
    prompt_path: Path,
    output_path: Path,
    lease_runtime_dir: Path,
) -> int:
    model = _load_json(model_path)
    device = _load_json(device_path)
    prompt = str(_load_json(prompt_path).get("prompt", ""))
    receipt = run_live_model_worker(
        model,
        device,
        prompt=prompt,
        lease_runtime_dir=lease_runtime_dir,
    )
    _write_json_unchecked(output_path, receipt)
    return 0 if not receipt.get("errors") else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent contract, a live worker, or a cold validation pass."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument("--worker-device", type=Path)
    parser.add_argument("--worker-prompt", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--lease-runtime-dir", type=Path, default=LEASE_RUNTIME_DIR)
    args = parser.parse_args(argv)
    if args.worker:
        required = (
            args.worker_model,
            args.worker_device,
            args.worker_prompt,
            args.worker_output,
        )
        if not all(value is not None for value in required):
            parser.error("--worker requires model, device, prompt, and output paths")
        return _worker_entry(
            args.worker_model,
            args.worker_device,
            args.worker_prompt,
            args.worker_output,
            args.lease_runtime_dir,
        )
    if args.validate:
        artifact = _load_json(args.result_path)
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("invalid Exp6773 artifact: " + ",".join(errors))
        return 0
    if args.date != RUN_DATE:
        raise ValueError(f"planning date must be {RUN_DATE}")
    run(result_path=args.result_path, date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper is the CLI surface.
    raise SystemExit(main())
