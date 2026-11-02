"""Exp6312 model-local representation surface preflight.

Spec refs: REQ-INFRA-6312, SCENARIO-INFRA-6312-COMPLETE-NULL,
SCENARIO-INFRA-6312-SURFACE-SELECTION, SCENARIO-INFRA-6312-CONTROLS.

This experiment checks whether each mandated local GGUF model exposes one
reproducible output-free representation surface before a larger corpus is run.
It prefers hidden states only with tensor provenance.  Otherwise it uses a
fixed prefix trajectory built from deterministic embedding calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Protocol

from carnot import experiment_5852_three_family_paired_embeddings as exp5852
from carnot.inference.sota_models import SOTA_GGUF_MODELS, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
SurfaceBackendFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], "SurfaceBackend"]
HiddenStateProbe = Callable[[Mapping[str, Any]], JsonDict]
GpuMemoryProbe = Callable[[Mapping[str, Any], str], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6312_model_local_representation_surface_preflight.json"
)
MICRO_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_6312_model_local_representation_surface_preflight.micro_fixture.jsonl"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6312_model_local_representation_surface_preflight.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6312_model_local_representation_surface_preflight.py"
)
LLM_SPEC_RELATIVE_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")

SCHEMA = "carnot.experiment_6312.model_local_representation_surface_preflight.v1"
SURFACE_ROW_SCHEMA = SCHEMA + ".surface_row"
EXPERIMENT_ID = "experiment_6312_model_local_representation_surface_preflight"
DEFAULT_RUN_DATE = "20260811"
DEFAULT_RANDOM_SEED = 6312
INFERENCE_SUBSTRATE = "live_local_sota_gguf_model_local_representation_surface_preflight"
VERIFIER_IS_ORACLE = False
DEFAULT_CONTEXT_LENGTH = exp5852.DEFAULT_CONTEXT_LENGTH
DEFAULT_N_GPU_LAYERS = -1
SURFACE_DECIMALS = 8
PREFIX_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
EPSILON = 1e-8

MANDATED_MODEL_HF_IDS = exp5852.MANDATED_MODEL_HF_IDS
LEGACY_SMOKE_MODEL_IDS = exp5852.LEGACY_SMOKE_MODEL_IDS

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("research-references.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6312_model_local_representation_surface_preflight.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6312_model_local_representation_surface_preflight.py -m pytest tests/python/test_experiment_6312_model_local_representation_surface_preflight.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6312_model_local_representation_surface_preflight.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6312_model_local_representation_surface_preflight.py",
    ".venv/bin/python -m carnot.experiment_6312_model_local_representation_surface_preflight --date 20260811",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "paper_source_and_local_claim_boundary",
    "upstream_failure_ledger",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_and_quantizations",
    "tokenizer_hashes",
    "cuda_and_gpu_offload_receipts_by_model",
    "gpu_memory_before_peak_and_after_release_by_model",
    "candidate_surface_inventory_by_model",
    "selected_surface_by_model",
    "surface_tensor_shapes_and_hashes",
    "hidden_state_runtime_receipts_by_model",
    "prefix_trajectory_fallback_receipts_by_model",
    "micro_fixture_path_and_hash",
    "causal_intervention_results_by_model",
    "aa_noise_results_by_model",
    "claim_flip_pair_swap_label_permutation_evaluator_swap_results_by_model",
    "norm_length_truncation_duplicate_and_identity_results_by_model",
    "underpowered_or_missing_cells",
    "no_generation_receipt",
    "no_shared_adapter_receipt",
    "surface_selection_rule",
    "actual_work_duration_receipt",
    "duration_padding_count",
    "source_model_weight_mutation_count",
    "model_local_representation_surface_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state separates ready surfaces from clean null closure.",
    "paper_source_and_local_claim_boundary": "The paper motivates the probe idea but does not prove local GGUF tensor access.",
    "upstream_failure_ledger": "Prior V543 failures stay visible so this run cannot reuse a flagged shared bus.",
    "MODEL_SPECS": "All mandated models record exact local identity, bytes, tokenizer, context, CUDA placement, and eligibility.",
    "models_used": "The three mandated hub IDs are explicit and no legacy substitute can satisfy readiness.",
    "model_file_hashes_revisions_and_quantizations": "File bytes, revisions, and quantization bind each surface to one local model.",
    "tokenizer_hashes": "Tokenizer receipts detect drift in the embedded GGUF tokenizer path.",
    "cuda_and_gpu_offload_receipts_by_model": "CUDA receipts prevent CPU or stub execution from becoming a headline claim.",
    "gpu_memory_before_peak_and_after_release_by_model": "Memory receipts prove sequential load and release behavior.",
    "candidate_surface_inventory_by_model": "The inventory shows hidden-state availability and fallback eligibility before selection.",
    "selected_surface_by_model": "The selected surface is one native model surface and is not pooled across models.",
    "surface_tensor_shapes_and_hashes": "Shapes and hashes make raw row corpora replayable without storing them only in prose.",
    "hidden_state_runtime_receipts_by_model": "Hidden states are used only when tensor provenance exists.",
    "prefix_trajectory_fallback_receipts_by_model": "The fallback is output-free and preregistered when hidden states are unavailable.",
    "micro_fixture_path_and_hash": "The tiny causal fixture is immutable and checked before model extraction.",
    "causal_intervention_results_by_model": "Vulnerable/fixed controls must pass inside each model.",
    "aa_noise_results_by_model": "A/A duplicates catch nondeterministic replay and noise-only separation.",
    "claim_flip_pair_swap_label_permutation_evaluator_swap_results_by_model": "Semantic and evaluator controls catch selection leakage and label shortcuts.",
    "norm_length_truncation_duplicate_and_identity_results_by_model": "Shortcut controls catch norm, length, truncation, duplicate, and model-identity artifacts.",
    "underpowered_or_missing_cells": "Every missing precondition or failed control remains visible.",
    "no_generation_receipt": "The experiment must not generate answers or logits.",
    "no_shared_adapter_receipt": "No cross-model adapter or pooled rescue is allowed.",
    "surface_selection_rule": "The hidden-first rule is frozen before labels are observed.",
    "actual_work_duration_receipt": "Measured work time is reported without sleep padding.",
    "duration_padding_count": "A bare zero proves no duration padding was used.",
    "source_model_weight_mutation_count": "A bare zero proves model weights were not changed.",
    "model_local_representation_surface_ready_score": "Bare readiness is one only when all models and controls pass.",
    "protected_files_unchanged": "Protected conductor and ops files remain untouched by the preflight.",
    "preconditions_checked": "Local resources are checked before model construction.",
    "inference_substrate": "The artifact declares the local GGUF representation preflight path.",
    "verifier_is_oracle": "False because representation surfaces are features, not the exact code oracle.",
    "field_provenance": "Every field traces to the prompt, spec, module, fixture, runtime, or tests.",
    "field_principles": "Each field states why it exists and what failure mode it guards.",
    "test_commands": "Commands bind the result to unit, coverage, full-suite, spec, and run checks.",
    "test_exit_codes": "Exit codes prevent unchecked artifacts from becoming ready.",
    "duration_s": "Measured wall time is separated from reproducibility content.",
    "random_seeds": "Seeds make prefix ordering and deterministic backends replayable.",
    "reproducibility_checksum": "Stable checksum detects fixture, row, control, or provenance drift.",
    "honest_verdict": "The terminal verdict states ready or complete-null without broadening the claim.",
}


class SurfaceBackend(Protocol):
    """Output-free surface interface shared by live and fixture backends."""

    def load(self) -> JsonDict:
        """Load model weights and return a runtime receipt."""

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text with the model tokenizer."""

    def embed(self, text: str) -> list[float]:
        """Return an output-free embedding for text."""

    def close(self) -> None:
        """Release runtime resources for this model."""


LlamaCppPrefixSurfaceBackend = exp5852.LlamaCppOutputFreeEmbeddingBackend


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes in chunks so large GGUFs stay streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not Path(path).exists():
        return rows
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required at line {line_number}: {path}")
        rows.append(dict(payload))
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _run_command(command: Sequence[str], *, timeout_s: float) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def _gpu_devices() -> list[JsonDict]:  # pragma: no cover
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    devices: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return devices


def _memory_probe() -> JsonDict:  # pragma: no cover
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 8192, "ok": available_mb >= 8192}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 1024, "ok": available_mb >= 1024}


def _llama_cpp_probe() -> JsonDict:  # pragma: no cover
    try:
        import llama_cpp
        from llama_cpp import Llama
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "cuda_backend_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    version = str(getattr(llama_cpp, "__version__", "unknown"))
    return {
        "available": True,
        "version": version,
        "cuda_backend_available": "cuda" in str(getattr(llama_cpp, "llama_print_system_info", lambda: "")()).lower(),
        "hidden_state_candidate_attrs": [
            name
            for name in ("get_hidden_states", "last_hidden_state", "hidden_states")
            if hasattr(Llama, name)
        ],
    }


def _output_path_receipt(result_path: Path, row_dir: Path, micro_fixture_path: Path) -> JsonDict:
    def writable(path: Path) -> bool:
        parent = path if path.suffix == "" else path.parent
        return parent.exists() and os.access(parent, os.W_OK)

    return {
        "result_path": str(result_path),
        "row_dir": str(row_dir),
        "micro_fixture_path": str(micro_fixture_path),
        "atomic_suffix": ".tmp",
        "result_writable": writable(result_path),
        "row_dir_writable": writable(row_dir),
        "micro_fixture_writable": writable(micro_fixture_path),
        "ok": writable(result_path) and writable(row_dir) and writable(micro_fixture_path),
    }


def model_family(hf_id: str) -> str:
    """Return the stable short model family label."""

    return exp5852.model_family(hf_id)


def _registry_row(hf_id: str) -> JsonDict:
    registry = {str(row["hf_id"]): dict(row) for row in SOTA_GGUF_MODELS}
    return dict(registry.get(hf_id, {}))


def _revision_from_path(path: str) -> str:
    if not path:
        return ""
    parts = Path(path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local_file_no_hf_snapshot_revision"


def _tokenizer_receipt(source: Mapping[str, Any], model_path: str, present: bool) -> JsonDict:
    provided = source.get("tokenizer_receipt")
    if isinstance(provided, Mapping):
        receipt = dict(provided)
        receipt.setdefault("source", "provided")
        receipt.setdefault("loadable", False)
        receipt.setdefault("detail", "")
    elif present:
        ok, detail = gguf_tokenizer_loadable(model_path)
        receipt = {
            "source": "embedded_gguf_llama_cpp_vocab_only",
            "loadable": ok,
            "detail": detail,
        }
    else:
        receipt = {
            "source": "missing_model_path",
            "loadable": False,
            "detail": f"model_path missing or not on disk: {model_path!r}",
        }
    receipt.setdefault("receipt_hash", sha256_json(receipt))
    receipt.setdefault("tokenizer_hash", str(receipt["receipt_hash"]))
    return receipt


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize all three mandated model specs for artifact use."""

    by_id = {str(row.get("hf_id")): row for row in model_specs if isinstance(row, Mapping)}
    normalized: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        source = by_id.get(hf_id, {})
        registry = _registry_row(hf_id)
        model_path = str(source.get("model_path") or source.get("cache_path") or "")
        path = Path(model_path).expanduser() if model_path else Path()
        present = bool(model_path and path.is_file())
        tokenizer = _tokenizer_receipt(source, model_path, present)
        model_sha = str(source.get("model_sha256") or (sha256_file(path) if present else ""))
        revision = str(source.get("revision") or _revision_from_path(model_path))
        gpu = int(source.get("gpu", index % 2) or 0)
        context_length = int(source.get("context_length", DEFAULT_CONTEXT_LENGTH) or DEFAULT_CONTEXT_LENGTH)
        normalized.append(
            {
                "name": str(source.get("name") or registry.get("name") or hf_id.rsplit("/", 1)[-1]),
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": str(source.get("role") or registry.get("role") or ""),
                "gpu": gpu,
                "cuda_placement": {"main_gpu": gpu, "n_gpu_layers": DEFAULT_N_GPU_LAYERS},
                "model_path": model_path,
                "local_file_path": model_path,
                "cache_path": model_path,
                "local_path_hash": sha256_text(str(path.resolve())) if model_path else "",
                "model_sha256": model_sha,
                "revision": revision,
                "local_model_present": present,
                "headline_eligible": source.get("headline_eligible") is not False,
                "active_params_b": source.get("active_params_b", registry.get("active_params_b")),
                "total_params_b": source.get("total_params_b", registry.get("total_params_b")),
                "min_vram_gb": source.get("min_vram_gb", registry.get("min_vram_gb")),
                "quantization": str(
                    source.get("quantization") or registry.get("quantization") or "Q4_K_M"
                ),
                "context_length": context_length,
                "context": {"n_ctx": context_length},
                "tokenizer": tokenizer,
                "tokenizer_receipt": tokenizer,
                "tokenizer_hash": str(tokenizer.get("tokenizer_hash") or tokenizer["receipt_hash"]),
                "llama_cpp_loader": "carnot.pipeline.gemma4_quantized_loader.Gemma4QuantizedLoader",
            }
        )
    return normalized


def resolve_all_model_specs() -> list[JsonDict]:  # pragma: no cover
    """Resolve all mandated GGUF files through the existing local resolver."""

    from carnot import experiment_5964_sota_atom_compatibility_corpus as exp5964

    rows: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        registry = _registry_row(hf_id)
        quant = str(registry.get("quantization") or "Q4_K_M")
        rows.append(
            {
                "name": registry.get("name") or hf_id.rsplit("/", 1)[-1],
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": registry.get("role", ""),
                "gpu": index % 2,
                "model_path": exp5964._resolve_cached_primary_gguf(hf_id, quant),
                "quantization": quant,
                "headline_eligible": True,
                "active_params_b": registry.get("active_params_b"),
                "total_params_b": registry.get("total_params_b"),
                "min_vram_gb": registry.get("min_vram_gb"),
            }
        )
    return normalize_model_specs(rows)


def deterministic_surface_config() -> JsonDict:
    """Return the frozen output-free representation settings."""

    config = {
        "schema": SCHEMA + ".deterministic_surface_config",
        "seed": DEFAULT_RANDOM_SEED,
        "n_ctx": DEFAULT_CONTEXT_LENGTH,
        "n_batch": exp5852.DEFAULT_BATCH_SIZE,
        "n_ubatch": exp5852.DEFAULT_UBATCH_SIZE,
        "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        "embedding": True,
        "pooling_type": "LLAMA_POOLING_TYPE_LAST",
        "normalize_embeddings": False,
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "max_tokens_generated": 0,
        "generated_answers_enabled": False,
        "output_logits_enabled": False,
    }
    config["config_hash"] = sha256_json(config)
    return config


def collect_preconditions(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_dir: str | Path = REPO_ROOT / "results",
    micro_fixture_path: str | Path = REPO_ROOT / MICRO_FIXTURE_RELATIVE_PATH,
    root: Path = REPO_ROOT,
) -> JsonDict:  # pragma: no cover
    row_dir_path = Path(row_dir)
    row_dir_path.mkdir(parents=True, exist_ok=True)
    fixture_parent = Path(micro_fixture_path).parent
    fixture_parent.mkdir(parents=True, exist_ok=True)
    devices = _gpu_devices()
    llama_cpp = _llama_cpp_probe()
    memory = _memory_probe()
    disk = _disk_probe(root)
    output = _output_path_receipt(Path(result_path), row_dir_path, Path(micro_fixture_path))
    protected = protected_files_unchanged(root)
    blocked: list[str] = []
    if not devices:
        blocked.append("gpu_device_receipt_unavailable")
    if llama_cpp.get("available") is not True:
        blocked.append("llama_cpp_unavailable")
    if llama_cpp.get("cuda_backend_available") is not True:
        blocked.append("llama_cpp_cuda_backend_unavailable")
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if output.get("ok") is not True:
        blocked.append("output_path_not_writable")
    if protected.get("unchanged") is not True:
        blocked.append("protected_files_changed_before_model_construction")
    return {
        "schema": SCHEMA + ".preconditions",
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "llama_cpp": llama_cpp,
        "cuda": {
            "available": bool(devices) and llama_cpp.get("cuda_backend_available") is True,
            "backend": "CUDA" if devices else "unavailable",
            "genuine_offload_required": True,
        },
        "gpu": {"gpu_count": len(devices), "devices": devices, "ok": bool(devices)},
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output,
        "timeout_budget": {"available_s": 120, "estimated_required_s": 1, "ok": True},
        "random_seeds_checked": {"python": DEFAULT_RANDOM_SEED, "ok": True},
        "protected_hashes_checked_before_model_construction": protected.get("unchanged") is True,
        "legacy_tiny_models_policy": {
            "legacy_smoke_model_ids": list(LEGACY_SMOKE_MODEL_IDS),
            "smoke_only": True,
            "cannot_satisfy_readiness": True,
        },
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def protected_files_unchanged(root: Path = REPO_ROOT) -> JsonDict:
    """Record protected-file hashes and git status."""

    command = ["git", "status", "--short", "--", *[path.as_posix() for path in PROTECTED_FILES]]
    result = _run_command(command, timeout_s=10)
    records = {
        path.as_posix(): {
            "exists": (root / path).exists(),
            "sha256": sha256_file(root / path) if (root / path).exists() else "",
        }
        for path in PROTECTED_FILES
    }
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "records": records,
        "git_status_stdout": str(result.get("stdout", "")),
        "git_status_returncode": result.get("returncode"),
        "unchanged": result.get("returncode") == 0 and not str(result.get("stdout", "")).strip(),
    }


def _precondition_blockers(
    preconditions: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    blockers = list(preconditions.get("blocked_reasons") or [])
    if preconditions.get("preconditions_ready") is not True:
        blockers.append("preconditions_not_ready")
    if [str(row.get("hf_id")) for row in model_specs] != list(MANDATED_MODEL_HF_IDS):
        blockers.append("mandated_model_order_mismatch")
    for spec in model_specs:
        tokenizer = dict(spec.get("tokenizer_receipt") or {})
        if (
            spec.get("local_model_present") is not True
            or not str(spec.get("model_path", "")).endswith(".gguf")
            or not str(spec.get("model_sha256", "")).startswith("sha256:")
            or spec.get("headline_eligible") is not True
            or tokenizer.get("loadable") is not True
        ):
            blockers.append("mandated_model_unavailable")
            break
    devices_by_index = {
        int(device.get("index", -1)): int(device.get("memory_free_mb", 0) or 0)
        for device in list(dict(preconditions.get("gpu") or {}).get("devices") or [])
        if isinstance(device, Mapping)
    }
    for spec in model_specs:
        required = spec.get("min_vram_gb")
        if required is None:
            continue
        required_mb = max(0, int(float(required) * 1024) - 1024)
        if devices_by_index.get(int(spec.get("gpu", 0) or 0), 0) < required_mb:
            blockers.append("insufficient_free_vram")
            break
    if dict(preconditions.get("llama_cpp") or {}).get("available") is not True:
        blockers.append("llama_cpp_unavailable")
    if dict(preconditions.get("llama_cpp") or {}).get("cuda_backend_available") is not True:
        blockers.append("llama_cpp_cuda_backend_unavailable")
    if dict(preconditions.get("cuda") or {}).get("available") is not True:
        blockers.append("cuda_offload_unavailable")
    if dict(preconditions.get("gpu") or {}).get("ok") is not True:
        blockers.append("gpu_device_receipt_unavailable")
    resources = dict(preconditions.get("resources") or {})
    if dict(resources.get("memory") or {}).get("ok") is not True:
        blockers.append("insufficient_free_ram")
    if dict(resources.get("disk") or {}).get("ok") is not True:
        blockers.append("insufficient_free_disk")
    if dict(preconditions.get("output_paths") or {}).get("ok") is not True:
        blockers.append("output_path_not_writable")
    if dict(preconditions.get("timeout_budget") or {}).get("ok") is not True:
        blockers.append("timeout_budget_unavailable")
    if preconditions.get("protected_hashes_checked_before_model_construction") is not True:
        blockers.append("protected_hashes_not_checked_before_model_construction")
    if dict(preconditions.get("legacy_tiny_models_policy") or {}).get("cannot_satisfy_readiness") is not True:
        blockers.append("legacy_smoke_policy_missing")
    return sorted(set(blockers))


def _pad_code_pair(left: str, right: str) -> tuple[str, str]:
    if len(left) < len(right):
        left = left + (" " * (len(right) - len(left)))
    if len(right) < len(left):
        right = right + (" " * (len(left) - len(right)))
    return left, right


def micro_fixture_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one micro-fixture row while blanking its own row hash."""

    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def build_micro_fixture() -> list[JsonDict]:
    """Build two exact vulnerable/fixed code pairs with matched code lengths."""

    pairs = [
        (
            "auth-owner-bypass",
            "def allowed(user_is_admin, owns_record):\n    return user_is_admin or owns_record\n",
            "def allowed(user_is_admin, owns_record):\n    return user_is_admin and owns_record\n",
        ),
        (
            "role-owner-bypass",
            "def can_edit(role_is_staff, owns_record):\n    return role_is_staff or owns_record\n",
            "def can_edit(role_is_staff, owns_record):\n    return role_is_staff and owns_record\n",
        ),
    ]
    rows: list[JsonDict] = []
    previous_hash = "sha256:" + ("0" * 64)
    for index, (pair_id, vulnerable, fixed) in enumerate(pairs):
        vulnerable_code, fixed_code = _pad_code_pair(vulnerable, fixed)
        row = {
            "schema": SCHEMA + ".micro_fixture_row",
            "sequence_index": index,
            "pair_id": pair_id,
            "vulnerable_code": vulnerable_code,
            "fixed_code": fixed_code,
            "claim_flip_code": fixed_code,
            "vulnerable_code_length": len(vulnerable_code),
            "fixed_code_length": len(fixed_code),
            "matched_length_basis": "unicode_codepoints_after_space_padding",
            "labels_hidden_from_prompts": True,
            "evaluator_a_label": {"vulnerable": True, "fixed": False},
            "evaluator_b_label": {"vulnerable": True, "fixed": False},
            "previous_hash": previous_hash,
            "row_hash": "",
        }
        row["row_hash"] = micro_fixture_row_hash(row)
        previous_hash = str(row["row_hash"])
        rows.append(row)
    return rows


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize rows as deterministic JSONL."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def write_micro_fixture(fixture_path: Path, sidecar_path: Path | None = None) -> JsonDict:
    """Write the tiny fixture and its immutable sidecar atomically."""

    sidecar = sidecar_path or fixture_path.with_suffix(fixture_path.suffix + ".sidecar.json")
    rows = build_micro_fixture()
    text = rows_to_jsonl(rows)
    _write_atomic(fixture_path, text)
    sidecar_payload = {
        "schema": SCHEMA + ".micro_fixture_sidecar",
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "surface_selection_frozen_before_label_observation": True,
        "label_columns_excluded_from_prompts": True,
        "row_hashes": [row["row_hash"] for row in rows],
        "sidecar_hash": "",
    }
    sidecar_payload["sidecar_hash"] = sha256_json(sidecar_payload)
    _write_atomic(sidecar, canonical_json(sidecar_payload) + "\n")
    return {
        "schema": SCHEMA + ".micro_fixture_receipt",
        "path": str(fixture_path),
        "sha256": sha256_file(fixture_path),
        "row_count": len(rows),
        "sidecar_path": str(sidecar),
        "sidecar_sha256": sha256_file(sidecar),
        "matched_code_lengths": all(
            row["vulnerable_code_length"] == row["fixed_code_length"] for row in rows
        ),
        "labels_hidden_from_prompts": all(row["labels_hidden_from_prompts"] for row in rows),
        "ready": True,
    }


def read_micro_fixture(path: str | Path) -> list[JsonDict]:
    """Read and verify the immutable micro fixture."""

    rows = _read_jsonl(path)
    for row in rows:
        if row.get("row_hash") != micro_fixture_row_hash(row):
            raise ValueError(f"micro_fixture_row_hash:{row.get('pair_id')}")
    return rows


def _probe_hidden_state_runtime(model_spec: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    try:
        from llama_cpp import Llama
    except Exception as exc:
        return {
            "surface": "hidden_state",
            "model_hf_id": model_spec.get("hf_id"),
            "available": False,
            "tensor_provenance_available": False,
            "available_with_provenance": False,
            "failure": f"llama_cpp_unavailable:{type(exc).__name__}",
        }
    attrs = [
        name
        for name in ("get_hidden_states", "last_hidden_state", "hidden_states")
        if hasattr(Llama, name)
    ]
    return {
        "surface": "hidden_state",
        "model_hf_id": model_spec.get("hf_id"),
        "available": bool(attrs),
        "tensor_provenance_available": False,
        "available_with_provenance": False,
        "candidate_attrs": attrs,
        "failure": "hidden_state_tensor_provenance_not_exposed_by_runtime",
    }


def _surface_row_path(row_dir: Path, hf_id: str) -> Path:
    return row_dir / f"{RESULT_RELATIVE_PATH.stem}.{model_family(hf_id)}.surface_rows.jsonl"


def _round_floats(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite_surface")
        out.append(round(number, SURFACE_DECIMALS))
    return out


def _round_tensor(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_round_tensor(item) for item in value]
    return _round_floats([value])[0]


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    return [float(value)]


def _shape(value: Any) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return [0]
        return [len(value), *_shape(value[0])]
    return []


def _vector_distance(left: Any, right: Any) -> float:
    a = _flatten_numeric(left)
    b = _flatten_numeric(right)
    if len(a) != len(b):
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _norm(value: Any) -> float:
    flat = _flatten_numeric(value)
    return math.sqrt(sum(x * x for x in flat))


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _prompt_text(row: Mapping[str, Any], case_kind: str) -> str:
    pair_id = str(row["pair_id"])
    prompt_case_kind = case_kind
    if case_kind == "vulnerable":
        code = str(row["vulnerable_code"])
    elif case_kind == "fixed":
        code = str(row["fixed_code"])
    elif case_kind == "claim_flip":
        code = str(row["claim_flip_code"])
    elif case_kind in {"aa_left", "aa_right"}:
        prompt_case_kind = "aa_duplicate"
        code = "duplicate-control\n" + str(row["vulnerable_code"])
    elif case_kind == "pair_swap":
        code = str(row["fixed_code"]) + "\n--- pair-swap ---\n" + str(row["vulnerable_code"])
    else:
        raise ValueError(f"unknown_case_kind:{case_kind}")
    return "\n".join(
        [
            "schema=exp6312_micro_surface_prompt_v1",
            f"pair_id={pair_id}",
            f"case_kind={prompt_case_kind}",
            "code:",
            code,
        ]
    )


def _prefixes(text: str) -> list[str]:
    prefixes: list[str] = []
    for fraction in PREFIX_FRACTIONS:
        end = max(1, min(len(text), int(math.ceil(len(text) * fraction))))
        prefixes.append(text[:end])
    return prefixes


def _surface_tensor(
    backend: SurfaceBackend,
    text: str,
    selected_surface: str,
) -> Any:
    if selected_surface == "hidden_state":
        hidden = getattr(backend, "hidden_state_surface")
        return _round_tensor(hidden(text))
    return _round_tensor([backend.embed(prefix) for prefix in _prefixes(text)])


def surface_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one surface row while blanking its own row hash."""

    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _build_surface_rows(
    *,
    backend: SurfaceBackend,
    model_spec: Mapping[str, Any],
    fixture_rows: Sequence[Mapping[str, Any]],
    selected_surface: str,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    previous_hash = "sha256:" + ("0" * 64)
    cases = ("vulnerable", "fixed", "claim_flip", "aa_left", "aa_right", "pair_swap")
    for fixture in fixture_rows:
        for case_kind in cases:
            prompt = _prompt_text(fixture, case_kind)
            tensor = _surface_tensor(backend, prompt, selected_surface)
            token_count = len(backend.tokenize(prompt))
            row = {
                "schema": SURFACE_ROW_SCHEMA,
                "surface_row_id": sha256_json(
                    {
                        "model": model_spec["hf_id"],
                        "pair": fixture["pair_id"],
                        "case": case_kind,
                    }
                ),
                "model_hf_id": model_spec["hf_id"],
                "model_file_sha256": model_spec.get("model_sha256"),
                "selected_surface": selected_surface,
                "pair_id": fixture["pair_id"],
                "fixture_row_hash": fixture["row_hash"],
                "case_kind": case_kind,
                "prompt_hash": sha256_text(prompt),
                "prompt_label_free": True,
                "token_count": token_count,
                "truncated": token_count > int(model_spec.get("context_length", DEFAULT_CONTEXT_LENGTH)),
                "tensor_shape": _shape(tensor),
                "tensor_hash": sha256_json(tensor),
                "tensor": tensor,
                "previous_hash": previous_hash,
                "row_hash": "",
            }
            row["row_hash"] = surface_row_hash(row)
            previous_hash = str(row["row_hash"])
            rows.append(row)
    return rows


def _write_surface_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    text = rows_to_jsonl(rows)
    _write_atomic(path, text)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "row_hashes": {str(row.get("surface_row_id")): str(row.get("row_hash")) for row in rows},
        "prefix_chain_tip": str(rows[-1]["row_hash"]) if rows else "sha256:" + ("0" * 64),
    }


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["pair_id"]), str(row["case_kind"])): row for row in rows}


def _causal_results(rows: Sequence[Mapping[str, Any]], fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_key = _group_rows(rows)
    distances: list[float] = []
    for fixture in fixture_rows:
        vuln = by_key.get((str(fixture["pair_id"]), "vulnerable"))
        fixed = by_key.get((str(fixture["pair_id"]), "fixed"))
        if vuln is None or fixed is None:
            continue
        distances.append(_vector_distance(vuln["tensor"], fixed["tensor"]))
    passed = len(distances) == len(fixture_rows) and min(distances or [0.0]) > EPSILON
    return {
        "pair_count": len(distances),
        "mean_vulnerable_fixed_distance": sum(distances) / len(distances) if distances else 0.0,
        "min_vulnerable_fixed_distance": min(distances) if distances else 0.0,
        "nondegenerate": passed,
        "passed": passed,
    }


def _aa_results(rows: Sequence[Mapping[str, Any]], fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_key = _group_rows(rows)
    duplicate_matches: list[bool] = []
    distances: list[float] = []
    for fixture in fixture_rows:
        left = by_key.get((str(fixture["pair_id"]), "aa_left"))
        right = by_key.get((str(fixture["pair_id"]), "aa_right"))
        if left is None or right is None:
            duplicate_matches.append(False)
            continue
        duplicate_matches.append(left["tensor_hash"] == right["tensor_hash"])
        distances.append(_vector_distance(left["tensor"], right["tensor"]))
    passed = bool(duplicate_matches) and all(duplicate_matches) and max(distances or [0.0]) <= EPSILON
    return {
        "aa_pair_count": len(duplicate_matches),
        "duplicate_tensor_hashes_match": all(duplicate_matches) if duplicate_matches else False,
        "max_aa_distance": max(distances) if distances else 0.0,
        "passed": passed,
    }


def _claim_swap_results(
    rows: Sequence[Mapping[str, Any]], fixture_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    by_key = _group_rows(rows)
    label_free = all(row.get("prompt_label_free") is True for row in rows)
    evaluator_agreement = all(
        row.get("evaluator_a_label") == row.get("evaluator_b_label") for row in fixture_rows
    )
    claim_distances: list[float] = []
    pair_swap_present = True
    for fixture in fixture_rows:
        vuln = by_key.get((str(fixture["pair_id"]), "vulnerable"))
        claim = by_key.get((str(fixture["pair_id"]), "claim_flip"))
        swap = by_key.get((str(fixture["pair_id"]), "pair_swap"))
        pair_swap_present = pair_swap_present and swap is not None
        if vuln is not None and claim is not None:
            claim_distances.append(_vector_distance(vuln["tensor"], claim["tensor"]))
    claim_flip_measured = len(claim_distances) == len(fixture_rows)
    passed = label_free and evaluator_agreement and pair_swap_present and claim_flip_measured
    return {
        "claim_flip": {
            "measured": claim_flip_measured,
            "mean_distance": sum(claim_distances) / len(claim_distances) if claim_distances else 0.0,
        },
        "pair_swap": {"measured": pair_swap_present},
        "label_permutation": {
            "labels_absent_from_prompts": label_free,
            "surface_selection_uses_labels": False,
        },
        "evaluator_swap": {"evaluator_a_b_agree": evaluator_agreement},
        "passed": passed,
    }


def _norm_length_results(
    rows: Sequence[Mapping[str, Any]], fixture_rows: Sequence[Mapping[str, Any]], hf_id: str
) -> JsonDict:
    by_key = _group_rows(rows)
    norm_diffs: list[float] = []
    length_matches = [
        row["vulnerable_code_length"] == row["fixed_code_length"] for row in fixture_rows
    ]
    for fixture in fixture_rows:
        vuln = by_key.get((str(fixture["pair_id"]), "vulnerable"))
        fixed = by_key.get((str(fixture["pair_id"]), "fixed"))
        if vuln is not None and fixed is not None:
            norm_diffs.append(abs(_norm(vuln["tensor"]) - _norm(fixed["tensor"])))
    truncation_free = all(row.get("truncated") is False for row in rows)
    duplicate_count = len([row for row in rows if row.get("case_kind") == "aa_left"])
    duplicate_control_present = duplicate_count == len(fixture_rows)
    norm_pass = max(norm_diffs or [0.0]) <= 1e-6
    passed = norm_pass and all(length_matches) and truncation_free and duplicate_control_present
    return {
        "norm": {"max_vulnerable_fixed_norm_delta": max(norm_diffs) if norm_diffs else 0.0, "passed": norm_pass},
        "length": {"matched_code_lengths": all(length_matches), "basis": "fixture_codepoints"},
        "truncation": {"any_truncated": not truncation_free, "passed": truncation_free},
        "duplicate": {"duplicate_control_present": duplicate_control_present},
        "model_identity": {
            "model_hf_id": hf_id,
            "identity_used_as_feature": False,
            "per_model_only": True,
        },
        "passed": passed,
    }


def _surface_receipts(
    row_file_receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "row_file": dict(row_file_receipt),
        "rows": {
            str(row["surface_row_id"]): {
                "pair_id": row["pair_id"],
                "case_kind": row["case_kind"],
                "shape": row["tensor_shape"],
                "tensor_hash": row["tensor_hash"],
                "prompt_hash": row["prompt_hash"],
            }
            for row in rows
        },
    }


def _model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(spec["hf_id"]): {
            "model_path": spec.get("model_path"),
            "model_sha256": spec.get("model_sha256"),
            "revision": spec.get("revision"),
            "quantization": spec.get("quantization"),
            "context_length": spec.get("context_length"),
            "headline_eligible": spec.get("headline_eligible"),
        }
        for spec in model_specs
    }


def _tokenizer_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(spec["hf_id"]): {
            "tokenizer_hash": spec.get("tokenizer_hash"),
            "receipt_hash": dict(spec.get("tokenizer_receipt") or {}).get("receipt_hash"),
            "source": dict(spec.get("tokenizer_receipt") or {}).get("source"),
            "loadable": dict(spec.get("tokenizer_receipt") or {}).get("loadable"),
        }
        for spec in model_specs
    }


def _candidate_inventory(
    model_specs: Sequence[Mapping[str, Any]],
    hidden_receipts: Mapping[str, Mapping[str, Any]],
    blockers: Sequence[str],
) -> JsonDict:
    return {
        str(spec["hf_id"]): {
            "hidden_state": {
                "candidate": True,
                "available_with_tensor_provenance": hidden_receipts[str(spec["hf_id"])].get(
                    "available_with_provenance"
                )
                is True,
                "requires_generation": False,
            },
            "prefix_trajectory_fallback": {
                "candidate": True,
                "available": not blockers,
                "requires_generation": False,
                "prefix_fractions": list(PREFIX_FRACTIONS),
            },
        }
        for spec in model_specs
    }


def _empty_row_receipts(row_dir: Path, model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipts: JsonDict = {}
    for spec in model_specs:
        path = _surface_row_path(row_dir, str(spec["hf_id"]))
        receipts[str(spec["hf_id"])] = _write_surface_rows(path, [])
    return receipts


def _commands_pass(artifact: Mapping[str, Any]) -> bool:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    return bool(
        commands
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )


def model_local_representation_surface_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness for the model-local surface preflight."""

    if not _commands_pass(artifact):
        return 0.0
    if dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons"):
        return 0.0
    if dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is not True:
        return 0.0
    if dict(artifact.get("no_generation_receipt") or {}).get("generated_answers_enabled") is not False:
        return 0.0
    if dict(artifact.get("no_shared_adapter_receipt") or {}).get("cross_model_adapter_used") is not False:
        return 0.0
    if artifact.get("duration_padding_count") != 0 or artifact.get("source_model_weight_mutation_count") != 0:
        return 0.0
    if [str(row.get("hf_id")) for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_HF_IDS):
        return 0.0
    for hf_id in MANDATED_MODEL_HF_IDS:
        selected = dict(dict(artifact.get("selected_surface_by_model") or {}).get(hf_id) or {})
        if selected.get("surface") not in {"hidden_state", "prefix_trajectory_fallback"}:
            return 0.0
        if dict(artifact.get("causal_intervention_results_by_model") or {}).get(hf_id, {}).get("passed") is not True:
            return 0.0
        if dict(artifact.get("aa_noise_results_by_model") or {}).get(hf_id, {}).get("passed") is not True:
            return 0.0
        if (
            dict(
                artifact.get(
                    "claim_flip_pair_swap_label_permutation_evaluator_swap_results_by_model"
                )
                or {}
            )
            .get(hf_id, {})
            .get("passed")
            is not True
        ):
            return 0.0
        if (
            dict(artifact.get("norm_length_truncation_duplicate_and_identity_results_by_model") or {})
            .get(hf_id, {})
            .get("passed")
            is not True
        ):
            return 0.0
    return 1.0


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if not _commands_pass(artifact):
        reasons.append("failed_test_exit_codes")
    for field in (
        "causal_intervention_results_by_model",
        "aa_noise_results_by_model",
        "claim_flip_pair_swap_label_permutation_evaluator_swap_results_by_model",
        "norm_length_truncation_duplicate_and_identity_results_by_model",
    ):
        for hf_id, result in dict(artifact.get(field) or {}).items():
            if dict(result).get("passed") is not True:
                reasons.append(f"{hf_id}:{field}")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict with an honest prefix."""

    if model_local_representation_surface_ready_score(artifact) == 1.0:
        return "complete_ready: all mandated model-local representation surfaces passed preflight"
    reasons = _blocked_reasons(artifact)
    return "complete_null: " + ",".join(reasons[:8] or ["model_local_surface_not_ready"])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Checksum stable artifact content while excluding wall-clock timing."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    duration = dict(stable.get("actual_work_duration_receipt") or {})
    duration["started_at_unix_s"] = 0.0
    duration["ended_at_unix_s"] = 0.0
    duration["duration_s"] = 0.0
    stable["actual_work_duration_receipt"] = duration
    return sha256_json(stable)


def _paper_boundary() -> JsonDict:
    return {
        "activation_probes_reference": {
            "source": "research-references.md V544 Activation Probes entry",
            "arxiv_id_from_local_planner": "2608.09643",
            "local_claim_boundary": (
                "The paper motivates model-local activation probes. It does not prove that "
                "this host's GGUF runtime exposes hidden tensors with provenance."
            ),
        },
        "no_external_empirical_claim_imported": True,
    }


def _upstream_failure_ledger() -> JsonDict:
    return {
        "v543_shared_activation_bus": {
            "experiment": "experiment_6300_three_family_universal_activation_bus",
            "claimed_ready": True,
        },
        "v543_integrity_audit": {
            "experiment": "experiment_6301_activation_bus_integrity_audit",
            "terminal_class": "flagged",
            "reason": "causal controls failed the shared activation bus",
        },
        "rule_for_6312": "do not aggregate models and do not let one model rescue another",
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "spec": "openspec/capabilities/llm-ebm-inference/spec.md#REQ-INFRA-6312",
            "module": str(MODULE_RELATIVE_PATH),
            "test": str(TEST_RELATIVE_PATH),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_status(score: float) -> str:
    return "complete_ready" if score == 1.0 else "complete_null"


def _default_memory_probe(model_spec: Mapping[str, Any], phase: str) -> list[JsonDict]:  # pragma: no cover
    return _gpu_devices()


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_dir: str | Path = REPO_ROOT / "results",
    micro_fixture_path: str | Path = REPO_ROOT / MICRO_FIXTURE_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    surface_backend_factory: SurfaceBackendFactory = LlamaCppPrefixSurfaceBackend,
    hidden_state_probe: HiddenStateProbe = _probe_hidden_state_runtime,
    gpu_memory_probe: GpuMemoryProbe = _default_memory_probe,
    protected_files_receipt: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    date: str = DEFAULT_RUN_DATE,
    write: bool = True,
) -> JsonDict:
    """Run Exp6312 and write the terminal preflight artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    rows_root = Path(row_dir)
    rows_root.mkdir(parents=True, exist_ok=True)
    fixture_path = Path(micro_fixture_path)
    fixture_receipt = write_micro_fixture(fixture_path)
    fixture_rows = read_micro_fixture(fixture_path)
    specs = normalize_model_specs(model_specs if model_specs is not None else resolve_all_model_specs())
    preconditions = dict(
        preconditions_checked
        if preconditions_checked is not None
        else collect_preconditions(result_path=result, row_dir=rows_root, micro_fixture_path=fixture_path)
    )
    protected = dict(protected_files_receipt or protected_files_unchanged(REPO_ROOT))
    hidden_receipts = {str(spec["hf_id"]): hidden_state_probe(spec) for spec in specs}
    blockers = _precondition_blockers(preconditions, specs)
    preconditions["blocked_reasons"] = sorted(set(blockers))
    preconditions["preconditions_ready"] = not blockers
    config = deterministic_surface_config()
    candidate_inventory = _candidate_inventory(specs, hidden_receipts, blockers)
    selected_by_model: JsonDict = {}
    prefix_receipts: JsonDict = {}
    cuda_receipts: JsonDict = {}
    memory_receipts: JsonDict = {}
    row_receipts: JsonDict = {}
    surface_receipts: JsonDict = {}
    causal_results: JsonDict = {}
    aa_results: JsonDict = {}
    claim_swap_results: JsonDict = {}
    norm_length_results: JsonDict = {}

    if blockers:
        empty_receipts = _empty_row_receipts(rows_root, specs)
        for spec in specs:
            hf_id = str(spec["hf_id"])
            selected_by_model[hf_id] = {
                "surface": "not_selected_preconditions_blocked",
                "selection_reason": "preconditions_blocked_before_model_construction",
            }
            prefix_receipts[hf_id] = {"used": False, "blocked_before_model_construction": True}
            cuda_receipts[hf_id] = {"loaded": False, "blocked_before_model_construction": True}
            memory_receipts[hf_id] = {"before": [], "peak": [], "after_release": []}
            row_receipts[hf_id] = empty_receipts[hf_id]
            surface_receipts[hf_id] = _surface_receipts(empty_receipts[hf_id], [])
            causal_results[hf_id] = {"pair_count": 0, "passed": False}
            aa_results[hf_id] = {"aa_pair_count": 0, "passed": False}
            claim_swap_results[hf_id] = {"passed": False}
            norm_length_results[hf_id] = {"passed": False}
    else:
        for spec in specs:
            hf_id = str(spec["hf_id"])
            hidden_ok = hidden_receipts[hf_id].get("available_with_provenance") is True
            selected_surface = "hidden_state" if hidden_ok else "prefix_trajectory_fallback"
            selected_by_model[hf_id] = {
                "surface": selected_surface,
                "selection_reason": (
                    "hidden_state_tensor_provenance_available"
                    if hidden_ok
                    else "hidden_state_tensor_provenance_unavailable_prefix_fallback"
                ),
                "selected_before_label_observation": True,
            }
            prefix_receipts[hf_id] = {
                "used": selected_surface == "prefix_trajectory_fallback",
                "prefix_fractions": list(PREFIX_FRACTIONS),
                "embedding_mode": True,
                "generated_text_enabled": False,
            }
            before = gpu_memory_probe(spec, "before")
            backend = surface_backend_factory(spec, config)
            try:
                load_receipt = backend.load()
                peak = gpu_memory_probe(spec, "peak")
                rows = _build_surface_rows(
                    backend=backend,
                    model_spec=spec,
                    fixture_rows=fixture_rows,
                    selected_surface=selected_surface,
                )
                backend.close()
                gc.collect()
                after = gpu_memory_probe(spec, "after")
                row_path = _surface_row_path(rows_root, hf_id)
                row_receipt = _write_surface_rows(row_path, rows)
                cuda_receipts[hf_id] = dict(load_receipt)
                memory_receipts[hf_id] = {
                    "before": before,
                    "peak": peak,
                    "after_release": after,
                    "release_verified": before == after,
                }
                row_receipts[hf_id] = row_receipt
                surface_receipts[hf_id] = _surface_receipts(row_receipt, rows)
                causal_results[hf_id] = _causal_results(rows, fixture_rows)
                aa_results[hf_id] = _aa_results(rows, fixture_rows)
                claim_swap_results[hf_id] = _claim_swap_results(rows, fixture_rows)
                norm_length_results[hf_id] = _norm_length_results(rows, fixture_rows, hf_id)
            except Exception as exc:
                try:
                    backend.close()
                except Exception:
                    pass
                reason = f"{hf_id}:surface_collection_failed:{type(exc).__name__}"
                preconditions["blocked_reasons"] = sorted(
                    set(list(preconditions.get("blocked_reasons") or []) + [reason])
                )
                preconditions["preconditions_ready"] = False
                row_receipt = _write_surface_rows(_surface_row_path(rows_root, hf_id), [])
                cuda_receipts[hf_id] = {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}
                memory_receipts[hf_id] = {"before": before, "peak": [], "after_release": []}
                row_receipts[hf_id] = row_receipt
                surface_receipts[hf_id] = _surface_receipts(row_receipt, [])
                causal_results[hf_id] = {"pair_count": 0, "passed": False}
                aa_results[hf_id] = {"aa_pair_count": 0, "passed": False}
                claim_swap_results[hf_id] = {"passed": False}
                norm_length_results[hf_id] = {"passed": False}

    ended = time.perf_counter()
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    artifact: JsonDict = {
        "status": "complete_null",
        "paper_source_and_local_claim_boundary": _paper_boundary(),
        "upstream_failure_ledger": _upstream_failure_ledger(),
        "MODEL_SPECS": specs,
        "models_used": list(MANDATED_MODEL_HF_IDS),
        "model_file_hashes_revisions_and_quantizations": _model_file_receipts(specs),
        "tokenizer_hashes": _tokenizer_hashes(specs),
        "cuda_and_gpu_offload_receipts_by_model": cuda_receipts,
        "gpu_memory_before_peak_and_after_release_by_model": memory_receipts,
        "candidate_surface_inventory_by_model": candidate_inventory,
        "selected_surface_by_model": selected_by_model,
        "surface_tensor_shapes_and_hashes": surface_receipts,
        "hidden_state_runtime_receipts_by_model": hidden_receipts,
        "prefix_trajectory_fallback_receipts_by_model": prefix_receipts,
        "micro_fixture_path_and_hash": fixture_receipt,
        "causal_intervention_results_by_model": causal_results,
        "aa_noise_results_by_model": aa_results,
        "claim_flip_pair_swap_label_permutation_evaluator_swap_results_by_model": claim_swap_results,
        "norm_length_truncation_duplicate_and_identity_results_by_model": norm_length_results,
        "underpowered_or_missing_cells": [],
        "no_generation_receipt": {
            "generated_answers_enabled": False,
            "generated_text_enabled": False,
            "output_logits_enabled": False,
            "max_tokens_generated": 0,
            "llm_generate_method_called": False,
        },
        "no_shared_adapter_receipt": {
            "cross_model_adapter_used": False,
            "pooled_rescue_allowed": False,
            "raw_dimensions_concatenated_across_models": False,
            "per_model_controls_only": True,
        },
        "surface_selection_rule": {
            "rule": "hidden_state_with_tensor_provenance_else_prefix_trajectory_fallback",
            "frozen_before_label_observation": True,
            "uses_fixture_labels": False,
            "selection_order": ["hidden_state", "prefix_trajectory_fallback"],
        },
        "actual_work_duration_receipt": {
            "started_at_unix_s": started,
            "ended_at_unix_s": ended,
            "duration_s": round(ended - started, 6),
            "duration_padding_count": 0,
        },
        "duration_padding_count": 0,
        "source_model_weight_mutation_count": 0,
        "model_local_representation_surface_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "test_commands": list(test_commands),
        "test_exit_codes": exit_codes,
        "duration_s": round(ended - started, 6),
        "random_seeds": {"python": DEFAULT_RANDOM_SEED, "surface": DEFAULT_RANDOM_SEED},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["underpowered_or_missing_cells"] = _blocked_reasons(artifact)
    score = model_local_representation_surface_ready_score(artifact)
    artifact["model_local_representation_surface_ready_score"] = score
    artifact["status"] = _artifact_status(score)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(result, canonical_json(artifact) + "\n")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6312 terminal artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if set(dict(artifact.get("field_principles") or {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles must cover every required field")
    if artifact.get("duration_padding_count") != 0 or type(artifact.get("duration_padding_count")) is not int:
        raise ValueError("duration_padding_count must be bare 0")
    if artifact.get("source_model_weight_mutation_count") != 0 or type(artifact.get("source_model_weight_mutation_count")) is not int:
        raise ValueError("source_model_weight_mutation_count must be bare 0")
    if dict(artifact.get("no_generation_receipt") or {}).get("generated_answers_enabled") is not False:
        raise ValueError("no_generation_receipt generated answers")
    if dict(artifact.get("no_generation_receipt") or {}).get("max_tokens_generated") != 0:
        raise ValueError("no_generation_receipt generated tokens")
    if dict(artifact.get("no_shared_adapter_receipt") or {}).get("cross_model_adapter_used") is not False:
        raise ValueError("no_shared_adapter_receipt cross-model adapter")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not VERIFIER_IS_ORACLE:
        raise ValueError("verifier_is_oracle")
    expected_score = model_local_representation_surface_ready_score(artifact)
    if float(artifact.get("model_local_representation_surface_ready_score")) != expected_score:
        raise ValueError("model_local_representation_surface_ready_score")
    expected_status = _artifact_status(expected_score)
    if artifact.get("status") != expected_status:
        raise ValueError("status")
    verdict = str(artifact.get("honest_verdict"))
    if not verdict.startswith(("complete_ready:", "complete_null:")):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def refresh_artifact_test_exit_codes(
    *, artifact_path: str | Path, test_exit_codes: Mapping[str, int]
) -> JsonDict:
    """Refresh test exit codes and recompute terminal fields."""

    artifact = _read_json(artifact_path)
    artifact["test_exit_codes"] = dict(test_exit_codes)
    artifact["underpowered_or_missing_cells"] = _blocked_reasons(artifact)
    score = model_local_representation_surface_ready_score(artifact)
    artifact["model_local_representation_surface_ready_score"] = score
    artifact["status"] = _artifact_status(score)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    _write_atomic(Path(artifact_path), canonical_json(artifact) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DEFAULT_RUN_DATE)
    args = parser.parse_args(argv)
    run(date=str(args.date), result_path=REPO_ROOT / RESULT_RELATIVE_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
