"""Qualify owned forced-sequence scoring for three local GGUF families.

Spec refs: REQ-INFERENCE-6850 and SCENARIO-INFERENCE-6850-*.

This experiment proves that one owned CUDA scoring process can start, score a
fixed sequence, checkpoint, and stop cleanly for each model. It does not compare
models and does not make a compatibility or quality claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess
from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]
ModelRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6850_three_family_scoring_admission_canary.py"
)
PROCESS_MODULE_RELATIVE_PATH = Path("python/carnot/inference/llama_cpp_process.py")
WRAPPER_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6850_three_family_scoring_admission_canary.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6850_three_family_scoring_admission_canary.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6850_three_family_scoring_admission_canary.json")
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6850_three_family_scoring_admission_canary.checkpoint.json"
)
EXP6848_RELATIVE_PATH = Path("results/experiment_6848_v599_method_change_evidence_contract.json")

SCHEMA = "carnot.experiment_6850.three_family_scoring_admission_canary.v1"
INFERENCE_SUBSTRATE = "live_local_llama_cpp_cuda_forced_sequence_canary"
RUN_DATE = "20260901"
RANDOM_SEED = 6850
DISK_FLOOR_MB = 1024
LEASE_TTL_S = 900.0
HEALTH_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 300.0
BLOCKED_VERDICT = "complete_blocked_three_family_scoring_admission_canary"

MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen3_6_35b_a3b",
    "unsloth/gemma-4-31B-it-GGUF": "gemma4_31b_it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma4_26b_a4b_it",
}
CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

FIXED_PROMPT = "Owned scoring canary. Continue the fixed text exactly:"
FIXED_CANDIDATE = " execution readiness is checked without a scientific label."

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "model_specs",
    "models_used",
    "model_artifact_hashes",
    "tokenizer_receipts",
    "process_receipts",
    "accelerator_samples",
    "rows",
    "canary_token_receipts",
    "lease_receipts",
    "checkpoint_manifest",
    "teardown_receipts",
    "admission_canary_complete_score",
    "three_family_scoring_admission_ready_score",
    "scientific_effect_claimed",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Every top-level field states why it exists.",
    "preconditions_checked": "Admission fails before model load when a required resource is absent.",
    "inference_substrate": "The exact substrate separates forced scoring from generated answers.",
    "duration_s": "Measured wall time exposes skipped or implausibly short live work.",
    "model_specs": "The exact three public GGUF identities cannot be substituted.",
    "models_used": "Only models with an owned canary receipt count as used.",
    "model_artifact_hashes": "Each canary is bound to exact local model bytes.",
    "tokenizer_receipts": "Native GGUF metadata and tokenization replace an invalid HF tokenizer path.",
    "process_receipts": "PID, start time, command, port, GPU, and token digest prove ownership.",
    "accelerator_samples": "VRAM and process samples show real CUDA residency and recovery.",
    "rows": "One unlabeled canary row per model makes admission independently auditable.",
    "canary_token_receipts": "Finite candidate token scores prove the scoring path is usable.",
    "lease_receipts": "A bounded owner lease separates admission from unrelated GPU work.",
    "checkpoint_manifest": "Verified complete canaries are skipped while incomplete work reruns.",
    "teardown_receipts": "Exit and port release prevent one model from contaminating the next.",
    "admission_canary_complete_score": "Only all-three complete token receipts can set this score to one.",
    "three_family_scoring_admission_ready_score": "Exp6851 consumes this receipt-completeness gate.",
    "scientific_effect_claimed": "False prevents an execution canary from becoming a scientific result.",
    "gate_check_summary": "The first failed check retains its exact observed value.",
    "verifier_is_oracle": "The scoring process has no authority over correctness labels.",
    "verdict_class": "A closed class keeps admission separate from scientific benefit.",
    "honest_verdict": "A complete_ prefix gives the conductor a terminal result.",
}


class AdmissionError(RuntimeError):
    """A stable failure for malformed admission evidence."""


def canonical_json(value: Any) -> str:
    """Serialize JSON consistently so receipt hashes are reproducible."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash text with the prefixed repository digest format."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a large GGUF without loading the whole file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Return one exact admission comparison."""

    return {
        "check": str(check),
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and reject other JSON shapes."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AdmissionError(f"json_object_required:{path}")
    return dict(value)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON so a restart never reads a partial receipt."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def native_tokenizer_receipt(model_path: str) -> JsonDict:  # pragma: no cover - live GGUF path.
    """Hash native tokenizer metadata through the same runtime used for scoring."""

    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        token_ids = [int(token) for token in llm.tokenize(FIXED_CANDIDATE.encode("utf-8"))]
        metadata_value = getattr(llm, "metadata", {})
        metadata_value = metadata_value() if callable(metadata_value) else metadata_value
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        tokenizer_metadata = {
            str(key): value for key, value in metadata.items() if "tokenizer" in str(key).lower()
        }
        receipt = {
            "source": "embedded_gguf_llama_cpp_vocab_only",
            "metadata_present": bool(tokenizer_metadata),
            "loadable": bool(token_ids),
            "detail": f"native tokenizer returned {len(token_ids)} canary tokens",
            "probe_token_ids": token_ids,
            "metadata_key_count": len(tokenizer_metadata),
            "tokenizer_sha256": sha256_text(
                canonical_json({"metadata": tokenizer_metadata, "probe_token_ids": token_ids})
            ),
        }
        del llm
        gc.collect()
        return receipt
    except Exception as exc:
        return {
            "source": "embedded_gguf_llama_cpp_vocab_only",
            "metadata_present": False,
            "loadable": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "probe_token_ids": [],
            "metadata_key_count": 0,
            "tokenizer_sha256": "",
        }


def _registry() -> dict[str, JsonDict]:
    return {str(row["hf_id"]): dict(row) for row in SOTA_GGUF_MODELS}


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize exact model records in the required family order."""

    by_id = {str(row.get("hf_id")): dict(row) for row in model_specs}
    registry = _registry()
    normalized: list[JsonDict] = []
    for hf_id in MODEL_SPECS:
        source = by_id.get(hf_id, {})
        model_path = str(source.get("model_path") or "")
        path = Path(model_path).expanduser() if model_path else Path()
        present = bool(model_path and path.is_file())
        tokenizer = source.get("tokenizer_receipt")
        tokenizer = (
            dict(tokenizer)
            if isinstance(tokenizer, Mapping)
            else native_tokenizer_receipt(model_path)
            if present
            else {
                "source": "missing_model",
                "metadata_present": False,
                "loadable": False,
                "detail": "model file missing",
                "tokenizer_sha256": "",
            }
        )
        normalized.append(
            {
                "hf_id": hf_id,
                "family": MODEL_FAMILIES[hf_id],
                "role": registry.get(hf_id, {}).get("role"),
                "quantization": registry.get(hf_id, {}).get("quantization", "Q4_K_M"),
                "model_path": model_path,
                "local_model_present": present,
                "model_size_bytes": path.stat().st_size if present else 0,
                "model_sha256": str(
                    source.get("model_sha256") or (sha256_file(path) if present else "")
                ),
                "tokenizer_receipt": tokenizer,
                "cached_sota_pair_called": bool(source.get("cached_sota_pair_called", False)),
                "cached_sota_pair_hf_ids": list(source.get("cached_sota_pair_hf_ids") or []),
            }
        )
    return normalized


def resolve_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache dependent.
    """Call cached_sota_pair first, then resolve every exact model file."""

    pair = cached_sota_pair() or []
    pair_rows = [dict(row) for row in pair if isinstance(row, Mapping)]
    pair_ids = [str(row.get("hf_id")) for row in pair_rows]
    by_id = {str(row.get("hf_id")): row for row in pair_rows}
    rows: list[JsonDict] = []
    for hf_id in MODEL_SPECS:
        source = dict(by_id.get(hf_id, {}))
        if not source.get("model_path"):
            source["model_path"] = resolve_cached_gguf(hf_id, "Q4_K_M") or ""
        source["hf_id"] = hf_id
        source["cached_sota_pair_called"] = True
        source["cached_sota_pair_hf_ids"] = pair_ids
        rows.append(source)
    return normalize_model_specs(rows)


def _run_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_s": round(time.monotonic() - started, 6),
            "ok": completed.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.monotonic() - started, 6),
            "ok": False,
        }


def _choose_free_ports(count: int) -> list[int]:  # pragma: no cover - host state varies.
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(int(count)):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


def _port_free(port: int) -> bool:  # pragma: no cover - host state varies.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def _gpu_inventory() -> list[JsonDict]:  # pragma: no cover - host state varies.
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[JsonDict] = []
    for line in str(query.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "name": parts[2],
                    "free_vram_mb": int(parts[3]),
                    "total_vram_mb": int(parts[4]),
                    "visible_devices": parts[0],
                }
            )
    return rows


def _compute_apps() -> list[JsonDict]:  # pragma: no cover - host state varies.
    query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[JsonDict] = []
    for line in str(query.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4 and parts[0].isdigit():
            rows.append(
                {
                    "pid": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "used_memory_mb": int(parts[2]),
                    "process_name": parts[3],
                    "observed_only": True,
                    "signals_sent": [],
                }
            )
    return rows


def _cuda_token_scoring() -> JsonDict:  # pragma: no cover - environment dependent.
    try:
        from llama_cpp import Llama, llama_cpp

        offload = bool(llama_cpp.llama_supports_gpu_offload())
        return {
            "ok": offload and hasattr(Llama, "eval"),
            "llama_cpp_gpu_offload": offload,
            "llama_eval_present": hasattr(Llama, "eval"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "llama_cpp_gpu_offload": False,
            "llama_eval_present": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _probe_lease(runtime_dir: Path, gpu: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Prove the kernel-backed lease can be acquired without loading a model."""

    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=runtime_dir,
            task_id="exp6850-preflight",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model="exp6850-preflight-no-model-load",
            vram_before_mb=int(gpu["free_vram_mb"]),
            ttl_s=30.0,
        )
        owner = lease.owner_receipt()
        lease.transition("terminal_blocked")
        release = lease.release()
        return {"ok": True, "owner": owner, "release": release}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "signals_sent": []}


def collect_preconditions(
    *, root: Path, model_specs: Sequence[Mapping[str, Any]], runtime_dir: Path
) -> JsonDict:  # pragma: no cover - host and model dependent.
    """Check artifacts, native metadata, CUDA, storage, ports, GPU, and lease."""

    ports = _choose_free_ports(len(MODEL_SPECS))
    free_ports = len(ports) == len(MODEL_SPECS) and all(_port_free(port) for port in ports)
    cuda = _cuda_token_scoring()
    disk_free_mb = int(shutil.disk_usage(root).free / (1024 * 1024))
    inventory = _gpu_inventory()
    required_mb = (
        max(
            (int(row.get("model_size_bytes", 0)) + 1024 * 1024 - 1) // (1024 * 1024)
            for row in model_specs
        )
        + 1024
    )
    eligible = [
        row
        for row in inventory
        if "RTX 3090" in str(row.get("name")) and int(row.get("free_vram_mb", 0)) >= required_mb
    ]
    gpu = max(eligible, key=lambda row: int(row["free_vram_mb"])) if eligible else {}
    lease_probe = _probe_lease(runtime_dir / "leases", gpu) if gpu else {"ok": False}
    v599 = (
        read_json(root / EXP6848_RELATIVE_PATH) if (root / EXP6848_RELATIVE_PATH).is_file() else {}
    )
    checks = [
        gate_check(
            "cached_sota_pair_called",
            True,
            all(row.get("cached_sota_pair_called") is True for row in model_specs),
        ),
        gate_check(
            "all_three_exact_gguf_artifacts",
            list(MODEL_SPECS),
            [row.get("hf_id") for row in model_specs if row.get("local_model_present") is True],
        ),
        gate_check(
            "model_hashes_present",
            True,
            all(str(row.get("model_sha256", "")).startswith("sha256:") for row in model_specs),
        ),
        gate_check(
            "native_tokenizer_metadata",
            True,
            all(
                dict(row.get("tokenizer_receipt") or {}).get("metadata_present") is True
                and dict(row.get("tokenizer_receipt") or {}).get("loadable") is True
                and str(
                    dict(row.get("tokenizer_receipt") or {}).get("tokenizer_sha256", "")
                ).startswith("sha256:")
                for row in model_specs
            ),
        ),
        gate_check("cuda_token_scoring", True, cuda.get("ok") is True),
        gate_check("sufficient_disk", True, disk_free_mb >= DISK_FLOOR_MB),
        gate_check("free_owned_ports", True, free_ports),
        gate_check("eligible_gpu", True, bool(gpu)),
        gate_check("bounded_exclusive_gpu_lease", True, lease_probe.get("ok") is True),
        gate_check(
            "v599_evidence_contract_ready_score",
            1,
            v599.get("v599_evidence_contract_ready_score"),
        ),
    ]
    failed = [str(row["check"]) for row in checks if row["passed"] is not True]
    return {
        "schema": SCHEMA + ".preconditions",
        "preconditions_ready": not failed,
        "checks": checks,
        "blocked_reasons": failed,
        "ports": ports,
        "eligible_gpu": gpu,
        "required_free_vram_mb": required_mb,
        "accelerator_samples": [{**row, "phase": "preflight"} for row in inventory],
        "unrelated_processes_observed": _compute_apps(),
        "cuda_token_scoring": cuda,
        "disk": {"free_mb": disk_free_mb, "required_mb": DISK_FLOOR_MB},
        "lease_probe": lease_probe,
        "python": {"executable": sys.executable, "version": platform.python_version()},
    }


def canary_hash(row: Mapping[str, Any]) -> str:
    """Hash one canary while excluding its self-referential field."""

    unsigned = {key: value for key, value in row.items() if key != "canary_hash"}
    return sha256_text(canonical_json(unsigned))


def _canary_errors(row: Mapping[str, Any], model: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if row.get("hf_id") != model.get("hf_id"):
        errors.append("hf_id")
    if row.get("model_hash") != model.get("model_sha256"):
        errors.append("model_hash")
    tokenizer_hash = dict(model.get("tokenizer_receipt") or {}).get("tokenizer_sha256")
    if row.get("tokenizer_hash") != tokenizer_hash:
        errors.append("tokenizer_hash")
    token_ids = row.get("candidate_token_ids")
    logprobs = row.get("token_logprobs")
    if not isinstance(token_ids, list) or not token_ids:
        errors.append("candidate_token_ids")
    if (
        not isinstance(logprobs, list)
        or not logprobs
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in logprobs
        )
    ):
        errors.append("finite_token_logprobs")
    if (
        isinstance(token_ids, list)
        and isinstance(logprobs, list)
        and len(token_ids) != len(logprobs)
    ):
        errors.append("token_logprob_alignment")
    if not str(row.get("first_useful_output") or "").strip():
        errors.append("first_useful_output")
    if not str(row.get("final_output") or "").strip():
        errors.append("final_output")
    if not isinstance(row.get("latency_s"), (int, float)) or float(row.get("latency_s", 0)) < 0:
        errors.append("latency_s")
    if row.get("scientific_label", "not-none") is not None:
        errors.append("scientific_label")
    if row.get("supports_margin_claim") is not False:
        errors.append("supports_margin_claim")
    if row.get("canary_hash") != canary_hash(row):
        errors.append("canary_hash")
    return errors


def bundle_errors(bundle: Mapping[str, Any], model: Mapping[str, Any]) -> list[str]:
    """Return every receipt failure that prevents a model checkpoint."""

    row = bundle.get("row")
    row = row if isinstance(row, Mapping) else {}
    errors = _canary_errors(row, model)
    process_receipt = bundle.get("process_receipt")
    process_receipt = process_receipt if isinstance(process_receipt, Mapping) else {}
    required_process = (
        "pid",
        "start_time_ticks",
        "command_hash",
        "process_group_id",
        "owner_pid",
        "owner_start_time_ticks",
        "ownership_token_digest",
        "port",
        "gpu_uuid",
        "visible_devices",
        "model_hash",
        "tokenizer_hash",
    )
    if process_receipt.get("owned_by_task") is not True or any(
        process_receipt.get(field) in {None, ""} for field in required_process
    ):
        errors.append("process_receipt")
    lease = bundle.get("lease_receipt")
    lease = lease if isinstance(lease, Mapping) else {}
    release = lease.get("release")
    release = release if isinstance(release, Mapping) else {}
    if lease.get("lease_valid") is not True:
        errors.append("lease_valid")
    if release.get("released") is not True or release.get("phase") != "terminal_complete":
        errors.append("lease_release")
    teardown = bundle.get("teardown_receipt")
    teardown = teardown if isinstance(teardown, Mapping) else {}
    for field in (
        "ownership_verified",
        "process_exit_confirmed",
        "port_release_confirmed",
        "leak_free",
    ):
        if teardown.get(field) is not True:
            errors.append(f"teardown_{field}")
    if teardown.get("unrelated_process_kill_count_delta") != 0:
        errors.append("unrelated_process_kill_count_delta")
    return list(dict.fromkeys(errors))


def _bundle_hash(bundle: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in bundle.items() if key != "entry_hash"}
    return sha256_text(canonical_json(unsigned))


def build_checkpoint_manifest(
    bundles: Sequence[Mapping[str, Any]], *, model_specs: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Store only complete model bundles whose current hashes still match."""

    by_id = {str(row.get("hf_id")): row for row in model_specs}
    completed: list[JsonDict] = []
    for bundle_value in bundles:
        bundle = deepcopy(dict(bundle_value))
        row = bundle.get("row")
        hf_id = str(row.get("hf_id")) if isinstance(row, Mapping) else ""
        model = by_id.get(hf_id)
        if model is None or bundle_errors(bundle, model):
            continue
        bundle["entry_hash"] = _bundle_hash(bundle)
        completed.append(bundle)
    return {
        "schema": SCHEMA + ".checkpoint",
        "model_order": list(MODEL_SPECS),
        "completed": completed,
        "complete_model_count": len(completed),
        "completed_model_ids": [row["row"]["hf_id"] for row in completed],
        "resumed_model_count": 0,
        "rerun_model_count": 0,
        "checkpoint_hash": sha256_text(
            canonical_json(
                {
                    "model_order": list(MODEL_SPECS),
                    "entry_hashes": [row["entry_hash"] for row in completed],
                }
            )
        ),
    }


def write_checkpoint(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write restart state through the same atomic JSON path as the artifact."""

    write_json_atomic(path, manifest)


def _verified_checkpoint_bundles(
    path: Path, model_specs: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    if not path.is_file():
        return {}
    manifest = read_json(path)
    by_id = {str(row.get("hf_id")): row for row in model_specs}
    verified: dict[str, JsonDict] = {}
    for value in manifest.get("completed", []):
        if not isinstance(value, Mapping):
            continue
        bundle = dict(value)
        row = bundle.get("row")
        hf_id = str(row.get("hf_id")) if isinstance(row, Mapping) else ""
        model = by_id.get(hf_id)
        if (
            model is not None
            and bundle.get("entry_hash") == _bundle_hash(bundle)
            and not bundle_errors(bundle, model)
        ):
            verified[hf_id] = bundle
    return verified


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_check": failed[0].get("check") if failed else None,
        "expected": failed[0].get("expected") if failed else None,
        "observed": failed[0].get("observed") if failed else "all checks pass",
        "failed_checks": failed,
        "checks": [dict(row) for row in checks],
    }


def _source_hashes(root: Path) -> JsonDict:
    paths = {
        "spec": SPEC_RELATIVE_PATH,
        "module": MODULE_RELATIVE_PATH,
        "process_module": PROCESS_MODULE_RELATIVE_PATH,
        "wrapper": WRAPPER_RELATIVE_PATH,
        "focused_tests": TEST_RELATIVE_PATH,
        "exp6848": EXP6848_RELATIVE_PATH,
    }
    return {
        name: {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path) if (root / path).is_file() else "missing",
        }
        for name, path in paths.items()
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum", "field_principles"}
    }
    return sha256_text(canonical_json(unsigned))


def _attach_field_principles(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves Exp6850 provenance.") for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    return artifact


def _build_artifact(
    *,
    root: Path,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    bundles: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    precondition_blocked: bool,
) -> JsonDict:
    bundle_by_id = {
        str(bundle.get("row", {}).get("hf_id")): bundle
        for bundle in bundles
        if isinstance(bundle.get("row"), Mapping)
    }
    canary_complete = len(bundle_by_id) == len(MODEL_SPECS) and all(
        not _canary_errors(bundle_by_id[hf_id]["row"], model)
        for hf_id, model in zip(MODEL_SPECS, model_specs, strict=True)
        if hf_id in bundle_by_id
    )
    receipt_checks: list[JsonDict] = []
    for model in model_specs:
        hf_id = str(model["hf_id"])
        bundle = bundle_by_id.get(hf_id)
        errors = ["missing_bundle"] if bundle is None else bundle_errors(bundle, model)
        receipt_checks.append(gate_check(f"model.{hf_id}.receipts_complete", [], errors))
    receipt_checks.append(
        gate_check(
            "checkpoint_all_three_complete",
            len(MODEL_SPECS),
            checkpoint.get("complete_model_count"),
        )
    )
    ready = not precondition_blocked and all(row["passed"] for row in receipt_checks)
    if precondition_blocked:
        checks = [dict(row) for row in preconditions.get("checks", [])]
    else:
        checks = receipt_checks
    gate_summary = _gate_summary(checks)
    valid_count = sum(
        1
        for model in model_specs
        if (bundle := bundle_by_id.get(str(model["hf_id"]))) is not None
        and not bundle_errors(bundle, model)
    )
    verdict_class = "positive" if ready else "blocked" if valid_count == 0 else "partial"
    honest_verdict = (
        "complete_three_family_scoring_admission_ready"
        if ready
        else BLOCKED_VERDICT
        if verdict_class == "blocked"
        else "complete_partial_three_family_scoring_admission_canary"
    )
    rows = [deepcopy(bundle["row"]) for bundle in bundles if isinstance(bundle.get("row"), Mapping)]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6850,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "blocked" if verdict_class == "blocked" else "partial",
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": ["REQ-INFERENCE-6850", "SCENARIO-INFERENCE-6850-*"],
        "random_seed": RANDOM_SEED,
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "models_used": [hf_id for hf_id in MODEL_SPECS if hf_id in bundle_by_id],
        "model_artifact_hashes": {
            str(row["hf_id"]): {
                "path": row.get("model_path"),
                "sha256": row.get("model_sha256"),
            }
            for row in model_specs
        },
        "tokenizer_receipts": [
            {"hf_id": row["hf_id"], **deepcopy(dict(row.get("tokenizer_receipt") or {}))}
            for row in model_specs
        ],
        "process_receipts": [
            deepcopy(dict(bundle.get("process_receipt") or {})) for bundle in bundles
        ],
        "accelerator_samples": [
            deepcopy(dict(sample))
            for sample in preconditions.get("accelerator_samples", [])
            if isinstance(sample, Mapping)
        ]
        + [
            deepcopy(dict(sample))
            for bundle in bundles
            for sample in bundle.get("accelerator_samples", [])
            if isinstance(sample, Mapping)
        ],
        "rows": rows,
        "canary_token_receipts": deepcopy(rows),
        "lease_receipts": [deepcopy(dict(bundle.get("lease_receipt") or {})) for bundle in bundles],
        "checkpoint_manifest": deepcopy(dict(checkpoint)),
        "teardown_receipts": [
            deepcopy(dict(bundle.get("teardown_receipt") or {})) for bundle in bundles
        ],
        "admission_canary_complete_score": int(canary_complete),
        "three_family_scoring_admission_ready_score": int(ready),
        "scientific_effect_claimed": False,
        "compatibility_effect_claimed": False,
        "model_quality_claimed": False,
        "method_limits": {
            "scientific_label_present": False,
            "margin_claim_supported": False,
            "model_comparison_supported": False,
            "admission_only": True,
        },
        "source_artifact_hashes": _source_hashes(root),
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return _attach_field_principles(artifact)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    model_runner: ModelRunner | None = None,
    runtime_dir: str | Path | None = None,
    write: bool = True,
) -> JsonDict:
    """Run three sequential canaries or emit a complete blocked artifact."""

    started = time.monotonic()
    root = Path(root)
    result = Path(result_path) if result_path is not None else root / RESULT_RELATIVE_PATH
    checkpoint_file = (
        Path(checkpoint_path) if checkpoint_path is not None else root / CHECKPOINT_RELATIVE_PATH
    )
    runtime = (
        Path(runtime_dir)
        if runtime_dir is not None
        else Path(os.environ.get("CARNOT_EXP6850_RUNTIME_DIR", "/tmp/carnot-exp6850"))
    )
    specs = normalize_model_specs(model_specs) if model_specs is not None else resolve_model_specs()
    preconditions = (
        deepcopy(dict(preconditions_checked))
        if preconditions_checked is not None
        else collect_preconditions(root=root, model_specs=specs, runtime_dir=runtime)
    )
    blocked = preconditions.get("preconditions_ready") is not True or bool(
        preconditions.get("blocked_reasons")
    )
    if blocked:
        checkpoint = build_checkpoint_manifest([], model_specs=specs)
        artifact = _build_artifact(
            root=root,
            duration_s=time.monotonic() - started,
            model_specs=specs,
            preconditions=preconditions,
            bundles=[],
            checkpoint=checkpoint,
            precondition_blocked=True,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact

    ports = [int(port) for port in preconditions.get("ports", [])]
    if len(ports) != len(MODEL_SPECS):
        raise AdmissionError("three_ports_required")
    gpu = dict(preconditions.get("eligible_gpu") or {})
    verified = _verified_checkpoint_bundles(checkpoint_file, specs)
    resumed_count = len(verified)
    runner = model_runner or run_live_model_canary
    bundles_by_id: dict[str, JsonDict] = dict(verified)
    rerun_count = 0
    for model, port in zip(specs, ports, strict=True):
        hf_id = str(model["hf_id"])
        if hf_id in verified:
            continue
        rerun_count += 1
        bundle = runner(
            model,
            port,
            gpu=gpu,
            root=root,
            lease_runtime_dir=runtime / "leases",
            runtime_dir=runtime,
        )
        if not isinstance(bundle, Mapping):
            raise AdmissionError(f"model_runner_object_required:{hf_id}")
        bundles_by_id[hf_id] = dict(bundle)
        checkpoint = build_checkpoint_manifest(list(bundles_by_id.values()), model_specs=specs)
        checkpoint["resumed_model_count"] = resumed_count
        checkpoint["rerun_model_count"] = rerun_count
        write_checkpoint(checkpoint_file, checkpoint)
    bundles = [bundles_by_id[hf_id] for hf_id in MODEL_SPECS if hf_id in bundles_by_id]
    checkpoint = build_checkpoint_manifest(bundles, model_specs=specs)
    checkpoint["resumed_model_count"] = resumed_count
    checkpoint["rerun_model_count"] = rerun_count
    write_checkpoint(checkpoint_file, checkpoint)
    artifact = _build_artifact(
        root=root,
        duration_s=time.monotonic() - started,
        model_specs=specs,
        preconditions=preconditions,
        bundles=bundles,
        checkpoint=checkpoint,
        precondition_blocked=False,
    )
    if write:
        write_json_atomic(result, artifact)
    return artifact


def _gpu_snapshot(
    gpu: Mapping[str, Any], *, phase: str, owned_pid: int = 0
) -> JsonDict:  # pragma: no cover - live CUDA state.
    inventory = _gpu_inventory()
    sample = next(
        (row for row in inventory if row.get("gpu_uuid") == gpu.get("gpu_uuid")),
        dict(gpu),
    )
    apps = [row for row in _compute_apps() if row.get("gpu_uuid") == gpu.get("gpu_uuid")]
    owned = [row for row in apps if int(row.get("pid", -1)) == int(owned_pid)]
    return {
        **sample,
        "phase": phase,
        "compute_apps": apps,
        "owned_pid": int(owned_pid),
        "owned_vram_mb": sum(int(row.get("used_memory_mb", 0)) for row in owned),
        "owned_cuda_residency": bool(owned),
    }


def _blank_canary(model: Mapping[str, Any], *, error_text: str = "") -> JsonDict:
    row = {
        "hf_id": model.get("hf_id"),
        "model_hash": model.get("model_sha256"),
        "tokenizer_hash": dict(model.get("tokenizer_receipt") or {}).get("tokenizer_sha256"),
        "prompt_token_ids": [],
        "candidate_token_ids": [],
        "token_logprobs": [],
        "conditional_log_likelihood": None,
        "first_useful_output": "",
        "final_output": "",
        "latency_s": 0.0,
        "scientific_label": None,
        "supports_margin_claim": False,
        "error": error_text,
        "canary_hash": "",
    }
    row["canary_hash"] = canary_hash(row)
    return row


def _finish_lease(
    lease: Any,
    *,
    complete: bool,
    teardown: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[JsonDict, str | None]:  # pragma: no cover - live lease state.
    error_text: str | None = None
    try:
        phase = str(lease.document.get("phase"))
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            phase = "unloading"
        if phase == "unloading":
            lease.transition(
                "validating",
                vram_mb=int(after.get("owned_vram_mb", 0)),
                exit_code=0 if teardown.get("process_exit_confirmed") is True else 1,
                unload_observed=teardown.get("leak_free") is True,
            )
            phase = "validating"
        target = "terminal_complete" if complete and phase == "validating" else "terminal_blocked"
        if phase in {"preflight", "admitted", "loading", "validating"}:
            lease.transition(target)
        release = lease.release()
        return release, None
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        lease.close()
        return {}, error_text


def run_live_model_canary(
    model: Mapping[str, Any],
    port: int,
    *,
    gpu: Mapping[str, Any],
    root: Path,
    lease_runtime_dir: Path,
    runtime_dir: Path,
) -> JsonDict:  # pragma: no cover - live model and GPU path.
    """Run one fixed canary under a lease and an owned subprocess."""

    del root
    before = _gpu_snapshot(gpu, phase="before")
    lease: Any = None
    process: OwnedLlamaCppProcess | None = None
    process_receipt: JsonDict = {}
    row = _blank_canary(model)
    resident: JsonDict = {}
    teardown: JsonDict = {
        "ownership_verified": False,
        "process_exit_confirmed": True,
        "port_release_confirmed": True,
        "leak_free": True,
        "unrelated_process_kill_count_delta": 0,
    }
    lease_owner: JsonDict = {}
    lease_release: JsonDict = {}
    lease_error: str | None = None
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=lease_runtime_dir,
            task_id=f"exp6850-{model['family']}",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=str(model["hf_id"]),
            vram_before_mb=int(before.get("free_vram_mb", 0)),
            ttl_s=LEASE_TTL_S,
        )
        lease_owner = lease.owner_receipt()
        lease.transition("admitted")
        lease.transition("loading")
        command = [
            sys.executable,
            "-m",
            "carnot.experiment_6850_three_family_scoring_admission_canary",
            "--score-worker",
            "--model-path",
            str(model["model_path"]),
            "--port",
            str(port),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        process = OwnedLlamaCppProcess(
            command=command,
            port=port,
            env=env,
            log_path=runtime_dir / f"{model['family']}.log",
            state_path=runtime_dir / f"{model['family']}.owner.json",
        )
        process_receipt = process.launch()
        process_receipt.update(
            {
                "model_hash": model.get("model_sha256"),
                "tokenizer_hash": dict(model.get("tokenizer_receipt") or {}).get(
                    "tokenizer_sha256"
                ),
                "gpu_uuid": gpu.get("gpu_uuid"),
                "visible_devices": str(gpu.get("index")),
                "free_vram_before_mb": before.get("free_vram_mb"),
            }
        )
        health = process.wait_for_health(HEALTH_TIMEOUT_S)
        if health.get("ok") is not True:
            raise AdmissionError(f"worker_health_failed:{health.get('reason')}")
        resident = _gpu_snapshot(gpu, phase="resident", owned_pid=int(process_receipt["pid"]))
        if resident.get("owned_cuda_residency") is not True:
            raise AdmissionError("owned_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("owned_vram_mb", 0)))
        lease.transition("inferencing")
        lease.heartbeat()
        started = time.monotonic()
        response = process.post_json(
            "/score",
            {"prompt_text": FIXED_PROMPT, "candidate_text": FIXED_CANDIDATE},
            REQUEST_TIMEOUT_S,
        )
        response["latency_s"] = round(time.monotonic() - started, 6)
        row = {
            "hf_id": model.get("hf_id"),
            "model_hash": model.get("model_sha256"),
            "tokenizer_hash": dict(model.get("tokenizer_receipt") or {}).get("tokenizer_sha256"),
            "prompt_token_ids": [int(token) for token in response.get("prompt_token_ids", [])],
            "candidate_token_ids": [
                int(token) for token in response.get("candidate_token_ids", [])
            ],
            "token_logprobs": [float(value) for value in response.get("token_logprobs", [])],
            "conditional_log_likelihood": response.get("conditional_log_likelihood"),
            "first_useful_output": str(response.get("first_useful_output") or ""),
            "final_output": str(response.get("final_output") or ""),
            "latency_s": response["latency_s"],
            "scientific_label": None,
            "supports_margin_claim": False,
            "canary_hash": "",
        }
        row["canary_hash"] = canary_hash(row)
        lease.heartbeat()
    except Exception as exc:
        if not row.get("error"):
            row = _blank_canary(model, error_text=f"{type(exc).__name__}: {exc}")
    finally:
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except Exception as exc:
                lease_error = f"{type(exc).__name__}: {exc}"
        if process is not None:
            teardown = process.cleanup()
        after = _gpu_snapshot(
            gpu,
            phase="after",
            owned_pid=int(process_receipt.get("pid", 0) or 0),
        )
        process_receipt["free_vram_after_mb"] = after.get("free_vram_mb")
        canary_ok = not _canary_errors(row, model)
        teardown_ok = all(
            teardown.get(field) is True
            for field in ("process_exit_confirmed", "port_release_confirmed", "leak_free")
        )
        if lease is not None:
            lease_release, finish_error = _finish_lease(
                lease,
                complete=canary_ok and teardown_ok and lease_error is None,
                teardown=teardown,
                after=after,
            )
            lease_error = lease_error or finish_error
    lease_valid = (
        lease_error is None
        and lease_release.get("released") is True
        and lease_release.get("phase") == "terminal_complete"
    )
    return {
        "row": row,
        "process_receipt": process_receipt,
        "accelerator_samples": [before, resident, after],
        "lease_receipt": {
            "owner": lease_owner,
            "lease_valid": lease_valid,
            "phase_history": deepcopy(lease.document.get("phase_history", []))
            if lease is not None
            else [],
            "release": lease_release,
            "error": lease_error,
        },
        "teardown_receipt": teardown,
    }


class _CanaryEngine:  # pragma: no cover - live model path.
    """Load one GGUF and expose only fixed-sequence token scoring."""

    def __init__(self, model_path: str) -> None:
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=512,
            n_batch=128,
            n_ubatch=128,
            seed=RANDOM_SEED,
            logits_all=True,
            verbose=True,
        )

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        return [
            int(token)
            for token in self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=False)
        ]

    @staticmethod
    def _logprob(logits: Any, token_id: int) -> float:
        values = [float(value) for value in logits]
        max_logit = max(values)
        denominator = sum(math.exp(value - max_logit) for value in values)
        return values[token_id] - max_logit - math.log(denominator)

    def score(self, prompt_text: str, candidate_text: str) -> JsonDict:
        prompt = self._tokenize(prompt_text, add_bos=True)
        candidate = self._tokenize(candidate_text, add_bos=False)
        if not prompt or not candidate:
            raise AdmissionError("canary_tokens_missing")
        tokens = prompt + candidate
        self.llm.reset()
        self.llm.eval(tokens)
        scores = self.llm.scores
        logprobs = [
            self._logprob(scores[index - 1], token_id)
            for index, token_id in enumerate(candidate, start=len(prompt))
        ]
        if not all(math.isfinite(value) for value in logprobs):
            raise AdmissionError("nonfinite_canary_logprob")
        final_output = self.llm.detokenize(candidate).decode("utf-8", "replace")
        first_useful = ""
        for end in range(1, len(candidate) + 1):
            text = self.llm.detokenize(candidate[:end]).decode("utf-8", "replace")
            if text.strip():
                first_useful = text
                break
        return {
            "prompt_token_ids": prompt,
            "candidate_token_ids": candidate,
            "token_logprobs": [round(value, 8) for value in logprobs],
            "conditional_log_likelihood": round(sum(logprobs), 8),
            "first_useful_output": first_useful,
            "final_output": final_output,
        }


def _run_score_worker(model_path: str, port: int) -> int:  # pragma: no cover
    engine = _CanaryEngine(model_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            del fmt, args

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/score":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            value = engine.score(str(payload["prompt_text"]), str(payload["candidate_text"]))
            body = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", int(port)), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        del engine
        gc.collect()
    return 0


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and readiness errors without changing the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing_required_fields:{missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(artifact) <= set(principles):
        errors.append("field_principles")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("scientific_effect_claimed") is not False:
        errors.append("scientific_effect_claimed")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    ready = artifact.get("three_family_scoring_admission_ready_score")
    if ready not in {0, 1}:
        errors.append("three_family_scoring_admission_ready_score")
    if ready == 1 and artifact.get("admission_canary_complete_score") != 1:
        errors.append("ready_without_complete_canaries")
    if ready == 1 and artifact.get("gate_check_summary", {}).get("passed") is not True:
        errors.append("ready_with_failed_gate")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6850 or host one private scoring worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=REPO_ROOT / CHECKPOINT_RELATIVE_PATH
    )
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--score-worker", action="store_true")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if args.score_worker:
        return _run_score_worker(args.model_path, args.port)
    if args.date != RUN_DATE:
        raise AdmissionError(f"run_date_mismatch:{args.date}")
    artifact = run(
        root=REPO_ROOT,
        result_path=args.result_path,
        checkpoint_path=args.checkpoint_path,
        runtime_dir=args.runtime_dir,
        write=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        for error_text in errors:
            print(error_text)
        return 1
    print(
        json.dumps(
            {
                "result_path": str(args.result_path),
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
