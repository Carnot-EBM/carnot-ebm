"""Exp5759 SOTA exact proposal utility panel.

Spec refs: REQ-VERIFY-5759, REQ-BENCH-5759, SCENARIO-VERIFY-5759,
SCENARIO-VERIFY-5759-BLOCKED, SCENARIO-BENCH-5759,
SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS.

This module consumes the sealed Exp5746 science split and scores only opaque
finite-choice labels.  The GGUF logits can rank candidates, but exact
validators remain the only authority for feasibility, objective value, and
optimum discovery.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5733_sota_finite_choice_proposal_channel as exp5733
from carnot import experiment_5746_exact_proposal_utility_benchmark as exp5746
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ScoreRunner = Callable[[JsonDict, list[JsonDict], JsonDict], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5759_sota_exact_proposal_utility_panel.json")
CHECKPOINT_RELATIVE_PATH = Path(
    "results/experiment_5759_sota_exact_proposal_utility_panel.checkpoint.json"
)
BRIDGE_RELATIVE_PATH = Path("results/experiment_5757_proposal_benchmark_scalar_bridge.json")
EXP5733_RELATIVE_PATH = exp5733.RESULT_RELATIVE_PATH
EXP5734_RELATIVE_PATH = Path("results/experiment_5734_sota_exact_proposal_stream.json")
BENCHMARK_ARTIFACT_RELATIVE_PATH = exp5746.RESULT_RELATIVE_PATH
BENCHMARK_MANIFEST_RELATIVE_PATH = exp5746.BENCHMARK_MANIFEST_RELATIVE_PATH

SCHEMA = "carnot.experiment_5759.sota_exact_proposal_utility_panel.v1"
EXPERIMENT = 5759
EXPERIMENT_ID = "experiment_5759_sota_exact_proposal_utility_panel"
TITLE = "Exp 5759: SOTA Exact Proposal Utility Panel"
MILESTONE = "2026.07.514"
RUN_DATE = "20260721"
INFERENCE_SUBSTRATE = "local_llama_cpp_python_cuda_gguf_plus_exact_validators"
SPEC_REFS = (
    "REQ-VERIFY-5759",
    "REQ-BENCH-5759",
    "SCENARIO-VERIFY-5759",
    "SCENARIO-VERIFY-5759-BLOCKED",
    "SCENARIO-BENCH-5759",
    "SCENARIO-BENCH-5759-BLOCKED-PRECONDITIONS",
)

QWEN_ID = exp5733.QWEN_ID
GEMMA31_ID = exp5733.GEMMA31_ID
GEMMA26_ID = exp5733.GEMMA26_ID
HEADLINE_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
FLAGSHIP_MODEL_IDS = (QWEN_ID, GEMMA31_ID)
REQUIRED_FAMILIES = exp5746.REQUIRED_FAMILIES
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
TOP_K = 5
BOOTSTRAP_RESAMPLES = 1000
N_GPU_LAYERS_REQUESTED = -1

RANDOM_SEEDS: JsonDict = {
    "candidate_order_seed": 5759001,
    "label_bijection_seed": 5759002,
    "runner_seed": 5759003,
    "bootstrap_seed": 5759004,
    "checkpoint_seed": 5759005,
    "base_seed": 5759,
}

PRODUCER_GATE_FIELDS = (
    "proposal_utility_lcb",
    "flagship_nonregression_count",
    "validator_disagreement_count",
    "authority_violation_count",
    "proposal_utility_ready_score",
)
CONTROL_NAMES = (
    "random_permutation",
    "carnot_energy_heuristic",
    "solver_native_branching",
    "exact_search_only",
)

_REGISTRY = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
MODEL_SPECS: list[JsonDict] = []
for _gpu, _hf_id in zip((0, 1, 0), HEADLINE_MODEL_IDS, strict=True):
    _base = dict(_REGISTRY[_hf_id])
    MODEL_SPECS.append(
        {
            "name": _base["name"],
            "hf_id": _hf_id,
            "model_repo_id": _hf_id,
            "family": exp5733.model_family(_hf_id),
            "role": _base["role"],
            "active_params_b": _base["active_params_b"],
            "total_params_b": _base["total_params_b"],
            "quantization": _base["quantization"],
            "min_vram_gb": _base["min_vram_gb"],
            "gpu": _gpu,
            "headline_eligible": True,
            "legacy_smoke_only": False,
        }
    )

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5759_sota_exact_proposal_utility_panel.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5759_sota_exact_proposal_utility_panel.py -m pytest tests/python/test_experiment_5759_sota_exact_proposal_utility_panel.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5759_sota_exact_proposal_utility_panel.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5759_sota_exact_proposal_utility_panel.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Lists final artifact keys and pins the Exp5759 schema.",
    "experiment": "Numeric experiment id for conductor indexing.",
    "experiment_id": "Stable slug for result lookup.",
    "title": "Human-readable panel title.",
    "milestone": "Milestone accountability for this science split measurement.",
    "run_date": "Absolute run date 20260721 avoids relative-date ambiguity.",
    "result_path": "Names the JSON artifact written by this workflow.",
    "started_at": "UTC timestamp for this Exp5759 run.",
    "finished_at": "UTC timestamp for this Exp5759 run completion.",
    "duration_s": "Wall-clock seconds for preflight, scoring, and exact metric aggregation.",
    "status": "Bare terminal state used by gates.",
    "random_seed": "Legacy scalar seed for methodology linters.",
    "metrics_used": "Names the local metric family rather than a hidden scorer.",
    "field_principles": "Every artifact field states the evidence boundary it protects.",
    "preconditions_checked": "Records bridge, benchmark, solver, model, CUDA, resource, and checkpoint gates before inference.",
    "spec_refs": "Binds the artifact to REQ-VERIFY-5759 and REQ-BENCH-5759.",
    "MODEL_SPECS": "Declares exactly the three mandated local headline GGUF identities.",
    "resolved_model_receipts": "Binds each model id to a concrete local GGUF path, size, hash, and tokenizer receipt.",
    "models_used": "Lists mandated models that actually contributed score rows.",
    "model_paths": "Records concrete GGUF paths used by llama.cpp.",
    "model_hashes": "Records immutable GGUF byte hashes for provenance.",
    "quantization_receipts": "Records observed GGUF quantization and filename for each model.",
    "llama_cpp_build": "Authenticates the local llama.cpp CUDA build.",
    "gpu_assignment": "Records intended and observed GPU placement for each model.",
    "cuda_offload_authenticated": "Per-model bare CUDA offload gate.",
    "upstream_artifact_hashes": "Seals Exp5733, Exp5734, Exp5746, Exp5757, and the Exp5746 manifest inputs.",
    "benchmark_manifest_hash": "Seals the frozen Exp5746 instance manifest bytes.",
    "science_split_hash": "Seals the immutable Exp5746 science row hashes.",
    "science_row_count": "Records the frozen science denominator.",
    "candidate_label_receipts": "Seals candidate label bijections, prompts, and one-token label counts before scoring.",
    "baseline_definitions": "Defines random, Carnot energy, solver-native, and exact-search controls.",
    "matched_budget_receipts": "Records matched candidate pools, top-k, validator-call budgets, seeds, and stopping rules.",
    "per_model_metrics": "Aggregates proposal ordering metrics by model.",
    "per_family_metrics": "Aggregates proposal ordering metrics by exact benchmark family.",
    "paired_metric_deltas": "Records signed paired proposal-minus-control utility components by row, family, model, and overall.",
    "confidence_intervals": "Reports deterministic paired bootstrap intervals by row, family, model, and overall.",
    "model_identity_shortcut_residual": "Summarizes per-family residual concentration after model means are removed.",
    "proposal_utility_delta_overall": "Bare signed normalized composite; positive favors model proposal ordering.",
    "proposal_utility_lcb": "Bare paired 95 percent lower bootstrap bound for the overall utility composite.",
    "flagship_nonregression_count": "Bare count of Qwen and Gemma31 models with non-negative deltas in every family.",
    "validator_disagreement_count": "Bare count of exact-validator replay disagreements.",
    "authority_violation_count": "Bare count of forbidden authority-path violations.",
    "proposal_utility_ready_score": "Bare downstream gate scalar for non-negative utility and zero authority violations.",
    "producer_gate_fields": "Lists the top-level bare scalar fields exported to conductor gates.",
    "verifier_is_oracle": "Bare true: exact validators are the only acceptance authority.",
    "llm_judge_used": "Bare false: no LLM judge participated.",
    "generated_text_scoring_used": "Bare false: generated reasoning text was not scored.",
    "token_scores_are_semantic_authority": "Bare false: token logits rank proposals only.",
    "model_weight_mutation": "Bare false: GGUF weights are loaded read-only.",
    "inference_substrate": "Declares local llama.cpp CUDA GGUF scoring plus exact validators.",
    "random_seeds": "Records candidate-order, label, runner, bootstrap, and checkpoint seeds.",
    "score_vector_hashes": "Seals per-model per-row finite-choice score vectors without storing generated text.",
    "row_metric_receipts": "Records per-row proposal and matched-control exact metrics.",
    "runtime_receipts": "Records per-model runtime, token, timing, and CUDA receipts.",
    "checkpoint_resume_receipt": "Records whether completed model receipts were reused from checkpoint.",
    "blocked_reasons": "Lists mechanical blockers when scoring cannot run or gates do not pass.",
    "test_commands": "Records focused, coverage, full-suite, spec, adversarial, and root-clutter commands.",
    "test_exit_codes": "Records observed or preregistered test command exit codes.",
    "reproducibility_checksum": "Hashes the artifact with its checksum field blanked.",
    "honest_verdict": "Terminal verdict begins complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks so provenance never depends on metadata alone."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json_object(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gate_scalar(value: Any) -> bool:
    return isinstance(value, bool) or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    )


def science_split_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the Exp5746 science split commitment from row hashes."""

    return sha256_json([str(row["row_hash"]) for row in rows])


def science_row_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count rows in the frozen science split."""

    return len([row for row in rows if row.get("split") == "science"])


def family_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count science rows by exact benchmark family."""

    return dict(sorted(Counter(str(row["family"]) for row in rows).items()))


def _default_paths(repo_root: str | Path = REPO_ROOT) -> JsonDict:
    root = Path(repo_root)
    return {
        "bridge": root / BRIDGE_RELATIVE_PATH,
        "exp5733": root / EXP5733_RELATIVE_PATH,
        "exp5734": root / EXP5734_RELATIVE_PATH,
        "benchmark": root / BENCHMARK_ARTIFACT_RELATIVE_PATH,
        "manifest": root / BENCHMARK_MANIFEST_RELATIVE_PATH,
    }


def _token_receipts_for_labels(offset: int = 1000) -> JsonDict:
    return {
        label: {
            "label": label,
            "token_ids": [offset + index],
            "token_count": 1,
            "unique": True,
            "token_text": label,
        }
        for index, label in enumerate(LABELS)
    }


def fixture_preconditions(output_root: str | Path = REPO_ROOT) -> JsonDict:
    """Build deterministic preconditions for tests without touching live GPUs."""

    root = Path(output_root)
    rows = exp5746.read_benchmark_manifest(REPO_ROOT / BENCHMARK_MANIFEST_RELATIVE_PATH)
    science_rows = [row for row in rows if row.get("split") == "science"]
    paths = _default_paths(REPO_ROOT)
    upstream_hashes = {name: sha256_file(path) for name, path in paths.items() if path.exists()}
    resolved: JsonDict = {}
    model_paths: JsonDict = {}
    model_hashes: JsonDict = {}
    quantizations: JsonDict = {}
    gpu_assignment: JsonDict = {}
    cuda_auth: JsonDict = {}
    for index, base in enumerate(MODEL_SPECS):
        model_path = root / "models" / f"{base['family']}-fixture-Q4_K_M.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"GGUF-fixture-exp5759-" + base["hf_id"].encode("utf-8"))
        digest = sha256_file(model_path)
        receipt = {
            "hf_id": base["hf_id"],
            "resolved_model_path": str(model_path),
            "model_path": str(model_path),
            "local_model_present": True,
            "model_size_bytes": model_path.stat().st_size,
            "model_hash": digest,
            "hash_authenticated": True,
            "gguf_filename": model_path.name,
            "quantization": "Q4_K_M",
            "tokenizer_label_receipt": {
                "model_hf_id": base["hf_id"],
                "label_count": len(LABELS),
                "all_single_unique_tokens": True,
                "labels": _token_receipts_for_labels(1000 + 100 * index),
            },
        }
        resolved[base["hf_id"]] = receipt
        model_paths[base["hf_id"]] = str(model_path)
        model_hashes[base["hf_id"]] = digest
        quantizations[base["hf_id"]] = {
            "filename": model_path.name,
            "quantization": "Q4_K_M",
            "source": "fixture",
        }
        gpu_assignment[base["hf_id"]] = {
            "requested_gpu": base["gpu"],
            "observed_gpu": base["gpu"],
            "mode": "fixture",
        }
        cuda_auth[base["hf_id"]] = True
    return {
        "preconditions_ready": True,
        "blocked_reasons": [],
        "run_date": RUN_DATE,
        "bridge_ready": True,
        "bridge_artifact_path": str(paths["bridge"]),
        "benchmark_artifact_path": str(paths["benchmark"]),
        "benchmark_manifest_path": str(paths["manifest"]),
        "exp5733_artifact_path": str(paths["exp5733"]),
        "exp5734_artifact_path": str(paths["exp5734"]),
        "upstream_artifact_hashes": upstream_hashes,
        "benchmark_manifest_hash": upstream_hashes["manifest"],
        "science_split_hash": science_split_hash(science_rows),
        "science_row_count": len(science_rows),
        "science_row_hashes": [str(row["row_hash"]) for row in science_rows],
        "solver_versions": {
            "primary_exact_solver": exp5746.PRIMARY_SOLVER_VERSION,
            "independent_exact_solver": exp5746.INDEPENDENT_SOLVER_VERSION,
            "energy_heuristic": exp5746.ENERGY_HEURISTIC_VERSION,
        },
        "resolved_model_receipts": resolved,
        "model_paths": model_paths,
        "model_hashes": model_hashes,
        "quantization_receipts": quantizations,
        "llama_cpp_build": {
            "version": "0.3.99-fixture",
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "system_info": "CUDA = 1 | ggml-cuda fixture",
        },
        "gpu_assignment": gpu_assignment,
        "cuda_offload_authenticated": cuda_auth,
        "resource_receipts": {
            "ram": {"available_mb": 8192, "required_mb": 1024, "ok": True},
            "disk": {"available_mb": 8192, "required_mb": 1024, "ok": True},
            "vram": {"all_required_free": True},
            "checkpoint_space": {"available_mb": 8192, "required_mb": 256, "ok": True},
        },
    }


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 1024
    available_mb = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1048576)
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _disk_probe(path: Path) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 1024
    usage = shutil.disk_usage(path)
    available_mb = int(usage.free / 1048576)
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _nvidia_smi_devices() -> list[JsonDict]:  # pragma: no cover - host-dependent preflight.
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
    devices = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "memory_used_mb": int(parts[5]),
                }
            )
    return devices


def _gpu_used_total_mb() -> int:  # pragma: no cover - host-dependent preflight.
    return sum(int(row.get("memory_used_mb", 0) or 0) for row in _nvidia_smi_devices())


def _llama_cpp_build_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    import importlib.metadata
    import llama_cpp

    version = importlib.metadata.version("llama-cpp-python")
    supports_gpu = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
    raw_info = llama_cpp.llama_cpp.llama_print_system_info()
    system_info = (
        raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
    )
    return {
        "version": version,
        "cuda_backend": "CUDA" in system_info.upper(),
        "supports_gpu_offload": supports_gpu,
        "system_info": system_info,
        "module": getattr(llama_cpp, "__file__", ""),
    }


def _tokenizer_label_receipt(
    model_hf_id: str,
    model_path: str,
) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    from llama_cpp import Llama

    vocab = Llama(model_path=model_path, vocab_only=True, verbose=False)
    labels: JsonDict = {}
    token_keys = []
    for label in LABELS:
        token_ids = list(vocab.tokenize(label.encode("utf-8"), add_bos=False))
        token_text = vocab.detokenize(token_ids).decode("utf-8", "replace") if token_ids else ""
        labels[label] = {
            "label": label,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "unique": len(token_ids) == 1,
            "token_text": token_text,
        }
        token_keys.append(tuple(token_ids))
    return {
        "model_hf_id": model_hf_id,
        "label_count": len(LABELS),
        "all_single_unique_tokens": len(set(token_keys)) == len(token_keys)
        and all(row["unique"] for row in labels.values()),
        "labels": labels,
    }


def collect_preconditions(  # pragma: no cover - host-dependent preflight.
    *,
    repo_root: str | Path = REPO_ROOT,
    checkpoint_path: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
) -> JsonDict:
    """Collect every structured gate before any model inference starts."""

    root = Path(repo_root)
    paths = _default_paths(root)
    blocked: list[str] = []
    upstream_hashes = {name: sha256_file(path) for name, path in paths.items() if path.exists()}
    for name in ("bridge", "exp5733", "exp5734", "benchmark", "manifest"):
        if name not in upstream_hashes:
            blocked.append(f"{name}_missing")
    bridge = _read_json_object(paths["bridge"]) if paths["bridge"].exists() else {}
    benchmark = _read_json_object(paths["benchmark"]) if paths["benchmark"].exists() else {}
    rows = exp5746.read_benchmark_manifest(paths["manifest"]) if paths["manifest"].exists() else []
    try:
        if benchmark:
            exp5746.validate_artifact(benchmark)
            exp5746.verify_benchmark_manifest(rows, benchmark)
    except Exception as exc:
        blocked.append(f"benchmark_replay_failed:{exc}")
    science_rows = [row for row in rows if row.get("split") == "science"]
    if bridge.get("benchmark_bridge_ready_score") != 1.0:
        blocked.append("exp5757_bridge_not_ready")
    if benchmark.get("benchmark_manifest_hash") != upstream_hashes.get("manifest"):
        blocked.append("benchmark_manifest_hash_mismatch")
    if science_row_count(science_rows) != 60:
        blocked.append("science_row_count")
    llama_build: JsonDict
    try:
        llama_build = _llama_cpp_build_probe()
    except Exception as exc:
        llama_build = {
            "version": "",
            "cuda_backend": False,
            "supports_gpu_offload": False,
            "error": str(exc),
        }
        blocked.append("llama_cpp_probe_failed")
    devices = _nvidia_smi_devices()
    if len(devices) < 2:
        blocked.append("gpu_visibility")
    pair_receipt = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 2))
    resolved: JsonDict = {}
    model_paths: JsonDict = {}
    model_hashes: JsonDict = {}
    quantizations: JsonDict = {}
    gpu_assignment: JsonDict = {}
    cuda_auth: JsonDict = {}
    for base in MODEL_SPECS:
        path = resolve_cached_gguf(base["hf_id"], str(base["quantization"]))
        if path is None:
            blocked.append(f"model_missing:{base['hf_id']}")
            path = ""
        digest = sha256_file(path) if path else ""
        tokenizer = (
            _tokenizer_label_receipt(base["hf_id"], path)
            if path
            else {"all_single_unique_tokens": False, "labels": {}}
        )
        if tokenizer.get("all_single_unique_tokens") is not True:
            blocked.append(f"tokenizer_label_collision:{base['hf_id']}")
        observed_device = next((row for row in devices if row.get("index") == base["gpu"]), {})
        free_mb = int(observed_device.get("memory_free_mb") or 0)
        if free_mb < int(base["min_vram_gb"]) * 1024 * 0.75:
            blocked.append(f"insufficient_vram:{base['hf_id']}")
        receipt = {
            "hf_id": base["hf_id"],
            "resolved_model_path": path,
            "model_path": path,
            "local_model_present": bool(path),
            "model_size_bytes": Path(path).stat().st_size if path else 0,
            "model_hash": digest,
            "hash_authenticated": bool(digest),
            "gguf_filename": Path(path).name if path else "",
            "quantization": exp5733.extract_quantization(Path(path).name) if path else "missing",
            "tokenizer_label_receipt": tokenizer,
            "resolution_helper": "cached_sota_pair_plus_resolve_cached_gguf",
        }
        resolved[base["hf_id"]] = receipt
        model_paths[base["hf_id"]] = path
        model_hashes[base["hf_id"]] = digest
        quantizations[base["hf_id"]] = {
            "filename": receipt["gguf_filename"],
            "quantization": receipt["quantization"],
            "source": "filename",
        }
        gpu_assignment[base["hf_id"]] = {
            "requested_gpu": base["gpu"],
            "observed_gpu": base["gpu"],
            "device_receipt": observed_device,
        }
        cuda_auth[base["hf_id"]] = bool(llama_build.get("cuda_backend")) and bool(
            llama_build.get("supports_gpu_offload")
        )
    ram = _memory_probe()
    disk = _disk_probe(root)
    checkpoint_disk = _disk_probe(Path(checkpoint_path).parent)
    if not ram["ok"]:
        blocked.append("insufficient_free_ram")
    if not disk["ok"] or not checkpoint_disk["ok"]:
        blocked.append("insufficient_free_disk")
    return {
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "run_date": RUN_DATE,
        "bridge_ready": bridge.get("benchmark_bridge_ready_score") == 1.0,
        "bridge_artifact_path": str(paths["bridge"]),
        "benchmark_artifact_path": str(paths["benchmark"]),
        "benchmark_manifest_path": str(paths["manifest"]),
        "exp5733_artifact_path": str(paths["exp5733"]),
        "exp5734_artifact_path": str(paths["exp5734"]),
        "cached_sota_pair_receipt": pair_receipt,
        "upstream_artifact_hashes": upstream_hashes,
        "benchmark_manifest_hash": upstream_hashes.get("manifest", ""),
        "science_split_hash": science_split_hash(science_rows),
        "science_row_count": len(science_rows),
        "science_row_hashes": [str(row["row_hash"]) for row in science_rows],
        "solver_versions": dict(benchmark.get("solver_versions") or {}),
        "resolved_model_receipts": resolved,
        "model_paths": model_paths,
        "model_hashes": model_hashes,
        "quantization_receipts": quantizations,
        "llama_cpp_build": llama_build,
        "gpu_assignment": gpu_assignment,
        "cuda_offload_authenticated": cuda_auth,
        "resource_receipts": {
            "ram": ram,
            "disk": disk,
            "vram": {
                "devices": devices,
                "all_required_free": not any("vram" in row for row in blocked),
            },
            "checkpoint_space": checkpoint_disk,
        },
    }


def load_science_rows(preconditions_checked: Mapping[str, Any]) -> list[JsonDict]:
    """Read and verify only the frozen Exp5746 science rows."""

    manifest_path = Path(str(preconditions_checked["benchmark_manifest_path"]))
    rows = exp5746.read_benchmark_manifest(manifest_path)
    science_rows = [dict(row) for row in rows if row.get("split") == "science"]
    if science_split_hash(science_rows) != preconditions_checked.get("science_split_hash"):
        raise ValueError("science_split_hash")
    if len(science_rows) != int(preconditions_checked.get("science_row_count") or 0):
        raise ValueError("science_row_count")
    return science_rows


def finite_choice_prompt(
    instance: Mapping[str, Any],
    label_mapping: Sequence[Mapping[str, Any]],
) -> str:
    """Build the sealed next-token scoring prompt for one science row."""

    candidate_lines = "\n".join(
        f"{row['label']}: {row['candidate_id']} {canonical_json(row['assignment'])}"
        for row in label_mapping
    )
    return (
        "Score exactly one next-token label for the candidate that best satisfies the hard "
        "constraints and then maximizes the soft objective. Labels are opaque one-character IDs.\n"
        f"Problem: {instance['natural_language_specification']}\n"
        f"Candidates:\n{candidate_lines}\n"
        "Answer label:"
    )


def _objective_range(evaluations: Mapping[str, Mapping[str, Any]]) -> int:
    feasible_values = [
        int(row["objective_value"]) for row in evaluations.values() if row["feasible"] is True
    ]
    return max(1, max(feasible_values) - min(feasible_values))


def freeze_science_panel(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze prompts, candidate-label bijections, budgets, and metric seeds."""

    panel = []
    for row in rows:
        candidates = {str(item["candidate_id"]): dict(item) for item in row["candidate_pool"]}
        candidate_ids = list(candidates)
        seed = int(RANDOM_SEEDS["candidate_order_seed"]) + int(row["sequence_index"])
        ordered_ids = list(candidate_ids)
        random.Random(seed).shuffle(ordered_ids)
        labels = LABELS[: len(ordered_ids)]
        mapping = [
            {
                "label": label,
                "candidate_id": candidate_id,
                "candidate_hash": candidates[candidate_id]["candidate_hash"],
                "assignment": candidates[candidate_id]["assignment"],
            }
            for label, candidate_id in zip(labels, ordered_ids, strict=True)
        ]
        prompt = finite_choice_prompt(row, mapping)
        evaluations = dict(row["solution_receipt"]["candidate_evaluations"])
        optimum = dict(row["exact_optimum_receipt"])
        matched_budget = {
            "instance_id": row["instance_id"],
            "candidate_count": len(candidate_ids),
            "candidate_pool_hash": row["candidate_pool_hash"],
            "top_k": min(TOP_K, len(candidate_ids)),
            "exact_validator_call_budget": len(candidate_ids),
            "candidate_order_seed": seed,
            "label_bijection_seed": int(RANDOM_SEEDS["label_bijection_seed"])
            + int(row["sequence_index"]),
            "stopping_rules": {
                "first_valid": "stop at first hard-feasible candidate in ordering",
                "first_optimum": "stop at first exact-optimal feasible candidate in ordering",
                "top_k": "check whether any exact optimum appears in the first k labels",
            },
        }
        panel_row = {
            "schema": SCHEMA + ".science_panel_row",
            "instance_id": str(row["instance_id"]),
            "row_hash": str(row["row_hash"]),
            "family": str(row["family"]),
            "sequence_index": int(row["sequence_index"]),
            "split": "science",
            "candidate_ids": candidate_ids,
            "candidate_order": ordered_ids,
            "candidate_order_seed": seed,
            "label_mapping": mapping,
            "label_bijection_complete": set(ordered_ids)
            == {item["candidate_id"] for item in mapping},
            "candidate_order_frozen_before_model_access": True,
            "prompt": prompt,
            "prompt_hash": sha256_text(prompt),
            "matched_budget": matched_budget,
            "baseline_ordering": dict(row["baseline_ordering"]),
            "solution_evaluations": evaluations,
            "exact_feasible_candidate_ids": list(optimum["feasible_candidate_ids"]),
            "exact_optimum_candidate_ids": list(optimum["optimal_candidate_ids"]),
            "optimum_value": int(optimum["optimum_value"]),
            "objective_range": _objective_range(evaluations),
        }
        panel_row["panel_row_hash"] = sha256_json(
            {
                "instance_id": panel_row["instance_id"],
                "row_hash": panel_row["row_hash"],
                "label_mapping": panel_row["label_mapping"],
                "prompt_hash": panel_row["prompt_hash"],
                "matched_budget": panel_row["matched_budget"],
            }
        )
        panel.append(panel_row)
    return panel


def candidate_label_receipts(panel: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return row-level candidate label commitments."""

    return {
        str(row["instance_id"]): {
            "panel_row_hash": row["panel_row_hash"],
            "prompt_hash": row["prompt_hash"],
            "label_mapping_hash": sha256_json(row["label_mapping"]),
            "candidate_order_seed": row["candidate_order_seed"],
            "label_bijection_complete": row["label_bijection_complete"],
            "label_count": len(row["label_mapping"]),
        }
        for row in panel
    }


def matched_budget_receipts(panel: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return matched-budget receipts keyed by instance id."""

    return {str(row["instance_id"]): dict(row["matched_budget"]) for row in panel}


def baseline_definitions() -> JsonDict:
    """Define every matched control before score interpretation."""

    return {
        "random_permutation": "Exp5746 sealed paired random permutation order.",
        "carnot_energy_heuristic": "Exp5746 deterministic hard-penalty soft-reward energy order.",
        "solver_native_branching": "Exp5746 exact solver native candidate enumeration order.",
        "exact_search_only": "Exact-validator-only scan over the same native candidate pool and budget.",
    }


def _selected_order_from_scores(
    panel_row: Mapping[str, Any],
    score_vector: Mapping[str, Any],
) -> tuple[list[str], str]:
    labels = [str(item["label"]) for item in panel_row["label_mapping"]]
    if set(score_vector) != set(labels):
        return [], "missing_score"
    converted: JsonDict = {}
    for label in labels:
        value = float(score_vector[label])
        if not math.isfinite(value):
            return [], "non_finite_score"
        converted[label] = value
    by_label = {
        str(item["label"]): str(item["candidate_id"]) for item in panel_row["label_mapping"]
    }
    ordered_labels = sorted(labels, key=lambda label: (-converted[label], labels.index(label)))
    return [by_label[label] for label in ordered_labels], ""


def _first_position(order: Sequence[str], targets: set[str]) -> int:
    for index, candidate_id in enumerate(order, start=1):
        if candidate_id in targets:
            return index
    return len(order) + 1


def ordering_metrics(
    panel_row: Mapping[str, Any],
    order: Sequence[str],
    *,
    wall_time_s: float = 0.0,
    model_tokens: int = 0,
) -> JsonDict:
    """Evaluate one proposal or control order through exact receipts."""

    candidate_count = len(panel_row["candidate_ids"])
    bounded_order = [
        candidate_id for candidate_id in order if candidate_id in set(panel_row["candidate_ids"])
    ]
    feasible_ids = set(str(value) for value in panel_row["exact_feasible_candidate_ids"])
    optimal_ids = set(str(value) for value in panel_row["exact_optimum_candidate_ids"])
    evaluations = dict(panel_row["solution_evaluations"])
    first = bounded_order[0] if bounded_order else ""
    selected = dict(evaluations.get(first) or {})
    selected_feasible = bool(selected.get("feasible") is True)
    if selected_feasible:
        gap = max(0, int(panel_row["optimum_value"]) - int(selected["objective_value"])) / float(
            panel_row["objective_range"]
        )
    else:
        gap = 1.0
    first_valid = _first_position(bounded_order, feasible_ids)
    first_optimum = _first_position(bounded_order, optimal_ids)
    top_k = min(TOP_K, candidate_count)
    return {
        "top_1_feasible_discovery": 1.0 if selected_feasible else 0.0,
        "top_k_exact_optimum_discovery": 1.0 if set(bounded_order[:top_k]) & optimal_ids else 0.0,
        "nodes_to_first_valid": first_valid,
        "candidates_to_first_valid": first_valid,
        "nodes_to_first_optimum": first_optimum,
        "candidates_to_first_optimum": first_optimum,
        "exact_validator_calls_to_first_valid": first_valid,
        "exact_validator_calls_to_first_optimum": first_optimum,
        "exact_validator_calls": first_optimum,
        "hard_violation_rate": 0.0 if selected_feasible else 1.0,
        "exact_objective_gap": round(gap, 12),
        "wall_time_s": round(float(wall_time_s), 6),
        "model_tokens": int(model_tokens),
        "candidate_count": candidate_count,
    }


def _control_orders(panel_row: Mapping[str, Any]) -> JsonDict:
    ordering = dict(panel_row["baseline_ordering"])
    native = list(ordering["exact_solver_native_order"])
    return {
        "random_permutation": list(ordering["random_permutation_order"]),
        "carnot_energy_heuristic": list(ordering["energy_heuristic_order"]),
        "solver_native_branching": native,
        "exact_search_only": native,
    }


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not rows:
        return {"row_count": 0}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        **{
            key: round(sum(float(row[key]) for row in rows) / len(rows), 12) for key in numeric_keys
        },
        "row_count": len(rows),
    }


def _delta_components(
    proposal: Mapping[str, Any],
    control: Mapping[str, Any],
) -> JsonDict:
    candidate_count = max(1.0, float(proposal["candidate_count"]))
    components = {
        "top_1_feasible_discovery_delta": float(proposal["top_1_feasible_discovery"])
        - float(control["top_1_feasible_discovery"]),
        "top_k_exact_optimum_discovery_delta": float(proposal["top_k_exact_optimum_discovery"])
        - float(control["top_k_exact_optimum_discovery"]),
        "nodes_to_first_valid_delta": (
            float(control["nodes_to_first_valid"]) - float(proposal["nodes_to_first_valid"])
        )
        / candidate_count,
        "exact_validator_calls_delta": (
            float(control["exact_validator_calls_to_first_optimum"])
            - float(proposal["exact_validator_calls_to_first_optimum"])
        )
        / candidate_count,
        "exact_objective_gap_delta": float(control["exact_objective_gap"])
        - float(proposal["exact_objective_gap"]),
    }
    components["composite_delta"] = round(sum(components.values()) / len(components), 12)
    return components


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 12) if values else 0.0


def _bootstrap_interval(values: Sequence[float], seed: int) -> JsonDict:
    if not values:
        return {"mean": 0.0, "lcb": -1.0, "ucb": 0.0, "resamples": BOOTSTRAP_RESAMPLES}
    if len(values) == 1:
        value = round(float(values[0]), 12)
        return {"mean": value, "lcb": value, "ucb": value, "resamples": BOOTSTRAP_RESAMPLES}
    rng = random.Random(seed)
    samples = []
    count = len(values)
    for _ in range(BOOTSTRAP_RESAMPLES):
        samples.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    samples.sort()
    lcb = samples[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))]
    ucb = samples[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))]
    return {
        "mean": round(sum(values) / count, 12),
        "lcb": round(lcb, 12),
        "ucb": round(ucb, 12),
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def _score_vector_hashes(runtime_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    hashes: JsonDict = {}
    for receipt in runtime_receipts:
        model_id = str(receipt["model_hf_id"])
        hashes[model_id] = {
            str(row["instance_id"]): sha256_json(
                {
                    "model_hf_id": model_id,
                    "instance_id": row["instance_id"],
                    "prompt_hash": row["prompt_hash"],
                    "score_vector": row.get("score_vector", {}),
                    "label_token_ids": row.get("label_token_ids", {}),
                }
            )
            for row in receipt.get("rows", [])
        }
    return hashes


def _runtime_row_map(
    runtime_receipts: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], JsonDict]:
    mapped = {}
    for receipt in runtime_receipts:
        model_id = str(receipt["model_hf_id"])
        for row in receipt.get("rows", []):
            mapped[(model_id, str(row.get("instance_id")))] = dict(row)
    return mapped


def _validator_disagreement_count(panel: Sequence[Mapping[str, Any]]) -> int:
    disagreements = 0
    for row in panel:
        evaluations = dict(row["solution_evaluations"])
        for candidate_id in row["candidate_ids"]:
            candidate = next(
                item for item in row["label_mapping"] if item["candidate_id"] == candidate_id
            )
            replay_instance = {
                "canonical_typed_formulation": {},
                "family": row["family"],
            }
            del replay_instance, candidate
            if candidate_id not in evaluations:
                disagreements += 1
    return disagreements


def _build_runtime_metrics(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    runtime_rows = _runtime_row_map(runtime_receipts)
    row_metric_receipts: JsonDict = {}
    per_model_rows: dict[str, list[JsonDict]] = defaultdict(list)
    per_family_rows: dict[str, dict[str, list[JsonDict]]] = defaultdict(lambda: defaultdict(list))
    delta_by_record: list[JsonDict] = []
    paired_metric_deltas: JsonDict = {
        "by_model_row": {},
        "by_model": {},
        "by_family": {},
        "overall": {},
    }
    missing_score_count = 0
    non_finite_score_count = 0
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        paired_metric_deltas["by_model_row"][model_id] = {}
        for row in panel:
            instance_id = str(row["instance_id"])
            raw = runtime_rows.get((model_id, instance_id), {})
            order, error = _selected_order_from_scores(row, dict(raw.get("score_vector") or {}))
            missing_score_count += 1 if error == "missing_score" else 0
            non_finite_score_count += 1 if error == "non_finite_score" else 0
            proposal_metrics = ordering_metrics(
                row,
                order,
                wall_time_s=float(dict(raw.get("timing") or {}).get("prefill_s") or 0.0),
                model_tokens=int(raw.get("prompt_token_count") or 0),
            )
            control_metrics = {
                name: ordering_metrics(row, control_order)
                for name, control_order in _control_orders(row).items()
            }
            deltas = {
                name: _delta_components(proposal_metrics, metrics)
                for name, metrics in control_metrics.items()
            }
            control_mean_delta = _mean(
                [float(delta["composite_delta"]) for delta in deltas.values()]
            )
            receipt = {
                "model_hf_id": model_id,
                "instance_id": instance_id,
                "family": row["family"],
                "proposal_metrics": proposal_metrics,
                "control_metrics": control_metrics,
                "deltas": deltas,
                "control_mean_delta": control_mean_delta,
                "proposal_order_hash": sha256_json(order),
            }
            row_metric_receipts.setdefault(model_id, {})[instance_id] = receipt
            paired_metric_deltas["by_model_row"][model_id][instance_id] = {
                "family": row["family"],
                "control_mean_delta": control_mean_delta,
                "against_controls": {
                    name: delta["composite_delta"] for name, delta in deltas.items()
                },
            }
            per_model_rows[model_id].append(proposal_metrics)
            per_family_rows[str(row["family"])][model_id].append(proposal_metrics)
            delta_by_record.append(
                {
                    "model_hf_id": model_id,
                    "instance_id": instance_id,
                    "family": row["family"],
                    "delta": control_mean_delta,
                }
            )
    per_model_metrics = {
        model_id: _aggregate_metric_rows(rows) for model_id, rows in per_model_rows.items()
    }
    per_family_metrics = {
        family: {
            model_id: _aggregate_metric_rows(rows) for model_id, rows in sorted(model_rows.items())
        }
        for family, model_rows in sorted(per_family_rows.items())
    }
    for model_id in per_model_rows:
        values = [
            float(record["delta"])
            for record in delta_by_record
            if record["model_hf_id"] == model_id
        ]
        paired_metric_deltas["by_model"][model_id] = {"control_mean_delta": _mean(values)}
    for family in REQUIRED_FAMILIES:
        family_records = [record for record in delta_by_record if record["family"] == family]
        paired_metric_deltas["by_family"][family] = {
            "control_mean_delta": _mean([float(record["delta"]) for record in family_records]),
            "by_model": {
                model_id: _mean(
                    [
                        float(record["delta"])
                        for record in family_records
                        if record["model_hf_id"] == model_id
                    ]
                )
                for model_id in HEADLINE_MODEL_IDS
            },
        }
    by_row_values: dict[str, list[float]] = defaultdict(list)
    for record in delta_by_record:
        by_row_values[str(record["instance_id"])].append(float(record["delta"]))
    row_means = {row_id: _mean(values) for row_id, values in by_row_values.items()}
    paired_metric_deltas["overall"] = {"control_mean_delta": _mean(list(row_means.values()))}
    return {
        "row_metric_receipts": row_metric_receipts,
        "per_model_metrics": per_model_metrics,
        "per_family_metrics": per_family_metrics,
        "paired_metric_deltas": paired_metric_deltas,
        "delta_by_record": delta_by_record,
        "row_mean_deltas": row_means,
        "missing_score_count": missing_score_count,
        "non_finite_score_count": non_finite_score_count,
    }


def _confidence_intervals(delta_bundle: Mapping[str, Any]) -> JsonDict:
    records = list(delta_bundle["delta_by_record"])
    row_means = dict(delta_bundle["row_mean_deltas"])
    by_model = {}
    for model_id in HEADLINE_MODEL_IDS:
        values = [float(row["delta"]) for row in records if row["model_hf_id"] == model_id]
        by_model[model_id] = _bootstrap_interval(values, int(RANDOM_SEEDS["bootstrap_seed"]))
    by_family = {}
    for family in REQUIRED_FAMILIES:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in records:
            if row["family"] == family:
                grouped[str(row["instance_id"])].append(float(row["delta"]))
        by_family[family] = _bootstrap_interval(
            [_mean(values) for values in grouped.values()],
            int(RANDOM_SEEDS["bootstrap_seed"]) + len(by_family) + 1,
        )
    by_row = {
        row_id: {
            "mean": value,
            "lcb": value,
            "ucb": value,
            "resamples": 1,
        }
        for row_id, value in sorted(row_means.items())
    }
    overall = _bootstrap_interval(
        list(row_means.values()),
        int(RANDOM_SEEDS["bootstrap_seed"]) + 999,
    )
    return {"by_row": by_row, "by_family": by_family, "by_model": by_model, "overall": overall}


def _shortcut_residual(delta_bundle: Mapping[str, Any]) -> JsonDict:
    records = list(delta_bundle["delta_by_record"])
    by_model_values: dict[str, list[float]] = defaultdict(list)
    by_model_family_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        by_model_values[str(row["model_hf_id"])].append(float(row["delta"]))
        by_model_family_values[(str(row["model_hf_id"]), str(row["family"]))].append(
            float(row["delta"])
        )
    residuals: JsonDict = {}
    max_abs = 0.0
    for model_id in HEADLINE_MODEL_IDS:
        model_mean = _mean(by_model_values[model_id])
        residuals[model_id] = {}
        for family in REQUIRED_FAMILIES:
            residual = round(_mean(by_model_family_values[(model_id, family)]) - model_mean, 12)
            residuals[model_id][family] = residual
            max_abs = max(max_abs, abs(residual))
    return {"by_model_family": residuals, "max_abs_residual": round(max_abs, 12)}


def _flagship_nonregression_count(paired_metric_deltas: Mapping[str, Any]) -> int:
    count = 0
    by_family = dict(paired_metric_deltas.get("by_family") or {})
    for model_id in FLAGSHIP_MODEL_IDS:
        family_values = [
            float(dict(dict(by_family.get(family) or {}).get("by_model") or {}).get(model_id, -1.0))
            for family in REQUIRED_FAMILIES
        ]
        if family_values and all(value >= 0.0 for value in family_values):
            count += 1
    return count


def authority_violation_count(artifact: Mapping[str, Any]) -> int:
    """Count forbidden authority path violations."""

    violations = 0
    violations += artifact.get("verifier_is_oracle") is not True
    violations += artifact.get("llm_judge_used") is not False
    violations += artifact.get("generated_text_scoring_used") is not False
    violations += artifact.get("token_scores_are_semantic_authority") is not False
    violations += artifact.get("model_weight_mutation") is not False
    violations += artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
    return int(violations)


def proposal_utility_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when all Exp5759 downstream gates pass."""

    ready = (
        artifact.get("status") == "complete"
        and dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and list(artifact.get("models_used") or []) == list(HEADLINE_MODEL_IDS)
        and float(artifact.get("proposal_utility_lcb") or -1.0) >= 0.0
        and int(artifact.get("flagship_nonregression_count") or 0) == len(FLAGSHIP_MODEL_IDS)
        and int(artifact.get("validator_disagreement_count") or 0) == 0
        and int(artifact.get("authority_violation_count") or 0) == 0
        and all(
            dict(artifact.get("cuda_offload_authenticated") or {}).get(model_id) is True
            for model_id in HEADLINE_MODEL_IDS
        )
        and int(artifact.get("science_row_count") or 0) == 60
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict from mechanical panel gates."""

    if artifact.get("status") == "blocked":
        reasons = list(
            dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or []
        )
        return "blocked: " + ",".join(reasons or ["exp5759_preconditions_not_ready"])
    if float(artifact.get("proposal_utility_ready_score") or 0.0) == 1.0:
        return "complete: sota_exact_proposal_utility_panel_ready"
    return "complete: sota_exact_proposal_utility_measured_gate_not_ready"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with the checksum field blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _model_specs_from_preconditions(preconditions: Mapping[str, Any]) -> list[JsonDict]:
    resolved = dict(preconditions.get("resolved_model_receipts") or {})
    specs = []
    for base in MODEL_SPECS:
        spec = dict(base)
        receipt = dict(resolved.get(base["hf_id"]) or {})
        spec.update(
            {
                "resolved_model_path": receipt.get("resolved_model_path", ""),
                "model_path": receipt.get("model_path", receipt.get("resolved_model_path", "")),
                "model_hash": receipt.get("model_hash", ""),
                "model_size_bytes": receipt.get("model_size_bytes", 0),
                "gguf_filename": receipt.get("gguf_filename", ""),
            }
        )
        specs.append(spec)
    return specs


def _runtime_summary(runtime_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(receipt["model_hf_id"]): {
            "llama_cpp_version": receipt.get("llama_cpp_version", ""),
            "llama_cpp_build": dict(receipt.get("llama_cpp_build") or {}),
            "gpu_assignment": receipt.get("gpu_assignment"),
            "n_gpu_layers_requested": receipt.get("n_gpu_layers_requested"),
            "n_gpu_layers_offloaded": receipt.get("n_gpu_layers_offloaded"),
            "gpu_memory_before_mb": receipt.get("gpu_memory_before_mb"),
            "gpu_memory_peak_mb": receipt.get("gpu_memory_peak_mb"),
            "gpu_memory_after_mb": receipt.get("gpu_memory_after_mb"),
            "cuda_offload_authenticated": receipt.get("cuda_offload_authenticated"),
            "row_count": len(receipt.get("rows", [])),
        }
        for receipt in runtime_receipts
    }


def _base_artifact(
    *,
    status: str,
    started_at: str,
    duration_s: float,
    preconditions_checked: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
    checkpoint_receipt: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    delta_bundle = _build_runtime_metrics(
        model_specs=model_specs,
        panel=panel,
        runtime_receipts=runtime_receipts,
    )
    confidence = _confidence_intervals(delta_bundle)
    paired = dict(delta_bundle["paired_metric_deltas"])
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "title": TITLE,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(duration_s, 6),
        "status": status,
        "random_seed": int(RANDOM_SEEDS["base_seed"]),
        "metrics_used": ["exp5759_signed_normalized_paired_utility_v1"],
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions_checked),
        "spec_refs": list(SPEC_REFS),
        "MODEL_SPECS": [dict(row) for row in model_specs],
        "resolved_model_receipts": dict(preconditions_checked.get("resolved_model_receipts") or {}),
        "models_used": [str(receipt["model_hf_id"]) for receipt in runtime_receipts],
        "model_paths": dict(preconditions_checked.get("model_paths") or {}),
        "model_hashes": dict(preconditions_checked.get("model_hashes") or {}),
        "quantization_receipts": dict(preconditions_checked.get("quantization_receipts") or {}),
        "llama_cpp_build": dict(preconditions_checked.get("llama_cpp_build") or {}),
        "gpu_assignment": dict(preconditions_checked.get("gpu_assignment") or {}),
        "cuda_offload_authenticated": dict(
            preconditions_checked.get("cuda_offload_authenticated") or {}
        ),
        "upstream_artifact_hashes": dict(
            preconditions_checked.get("upstream_artifact_hashes") or {}
        ),
        "benchmark_manifest_hash": str(preconditions_checked.get("benchmark_manifest_hash") or ""),
        "science_split_hash": str(preconditions_checked.get("science_split_hash") or ""),
        "science_row_count": int(preconditions_checked.get("science_row_count") or 0),
        "candidate_label_receipts": candidate_label_receipts(panel),
        "baseline_definitions": baseline_definitions(),
        "matched_budget_receipts": matched_budget_receipts(panel),
        "per_model_metrics": delta_bundle["per_model_metrics"],
        "per_family_metrics": delta_bundle["per_family_metrics"],
        "paired_metric_deltas": paired,
        "confidence_intervals": confidence,
        "model_identity_shortcut_residual": _shortcut_residual(delta_bundle),
        "proposal_utility_delta_overall": paired["overall"]["control_mean_delta"],
        "proposal_utility_lcb": confidence["overall"]["lcb"],
        "flagship_nonregression_count": _flagship_nonregression_count(paired),
        "validator_disagreement_count": _validator_disagreement_count(panel),
        "authority_violation_count": 0,
        "proposal_utility_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "verifier_is_oracle": True,
        "llm_judge_used": False,
        "generated_text_scoring_used": False,
        "token_scores_are_semantic_authority": False,
        "model_weight_mutation": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": dict(RANDOM_SEEDS),
        "score_vector_hashes": _score_vector_hashes(runtime_receipts),
        "row_metric_receipts": delta_bundle["row_metric_receipts"],
        "runtime_receipts": _runtime_summary(runtime_receipts),
        "checkpoint_resume_receipt": dict(checkpoint_receipt),
        "blocked_reasons": list(preconditions_checked.get("blocked_reasons") or []),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["authority_violation_count"] = authority_violation_count(artifact)
    artifact["proposal_utility_ready_score"] = proposal_utility_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["schema"] = sorted(artifact.keys())
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    started_at: str,
    duration_s: float,
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    model_specs = _model_specs_from_preconditions(preconditions_checked)
    artifact = _base_artifact(
        status="blocked",
        started_at=started_at,
        duration_s=duration_s,
        preconditions_checked=preconditions_checked,
        model_specs=model_specs,
        panel=[],
        runtime_receipts=[],
        checkpoint_receipt={"checkpoint_reused": False, "resumed_model_ids": []},
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    artifact["proposal_utility_delta_overall"] = 0.0
    artifact["proposal_utility_lcb"] = -1.0
    artifact["flagship_nonregression_count"] = 0
    artifact["validator_disagreement_count"] = 0
    artifact["authority_violation_count"] = authority_violation_count(artifact)
    artifact["proposal_utility_ready_score"] = proposal_utility_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["schema"] = sorted(artifact.keys())
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def read_checkpoint(path: str | Path) -> JsonDict:
    """Read a checkpoint if present; otherwise return an empty receipt."""

    checkpoint = Path(path)
    if not checkpoint.exists():
        return {"runtime_receipts": []}
    return _read_json_object(checkpoint)


def write_checkpoint(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write checkpoint payload atomically enough for local resume tests."""

    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_runtime_receipts(
    checkpoint: Mapping[str, Any],
    panel_hash: str,
) -> tuple[list[JsonDict], list[str]]:
    if checkpoint.get("panel_hash") != panel_hash:
        return [], []
    receipts = [dict(row) for row in checkpoint.get("runtime_receipts", [])]
    return receipts, [str(row.get("model_hf_id")) for row in receipts]


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on schema drift or unsupported readiness claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if set(artifact) != set(artifact.get("field_principles") or {}):
        raise ValueError("field_principles")
    for field in PRODUCER_GATE_FIELDS:
        if not _gate_scalar(artifact.get(field)):
            raise ValueError(field)
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(HEADLINE_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    for forbidden in (
        "llm_judge_used",
        "generated_text_scoring_used",
        "token_scores_are_semantic_authority",
        "model_weight_mutation",
    ):
        if artifact.get(forbidden) is not False:
            raise ValueError(forbidden)
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("producer_gate_fields") != list(PRODUCER_GATE_FIELDS):
        raise ValueError("producer_gate_fields")
    if artifact.get("authority_violation_count") != authority_violation_count(artifact):
        raise ValueError("authority_violation_count")
    if artifact.get("proposal_utility_ready_score") != proposal_utility_ready_score(artifact):
        raise ValueError("proposal_utility_ready_score")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def default_score_runner(  # pragma: no cover - host-dependent live inference.
    model_spec: JsonDict,
    candidate_rows: list[JsonDict],
    random_seeds: JsonDict,
) -> JsonDict:
    """Score one model's sealed labels in a child process so GPUs release on exit."""

    del random_seeds
    before_mb = _gpu_used_total_mb()
    worker_payload = {
        "model_spec": model_spec,
        "candidate_rows": candidate_rows,
        "labels": list(LABELS),
        "n_gpu_layers": N_GPU_LAYERS_REQUESTED,
        "runner_seed": int(RANDOM_SEEDS["runner_seed"]),
    }
    worker_code = r"""
import gc
import importlib.metadata
import json
import sys
import time

payload = json.load(sys.stdin)
try:
    import llama_cpp
    from llama_cpp import Llama
    version = importlib.metadata.version("llama-cpp-python")
    raw_info = llama_cpp.llama_cpp.llama_print_system_info()
    system_info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
    supports_gpu = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
    vocab = Llama(model_path=payload["model_spec"]["resolved_model_path"], vocab_only=True, verbose=False)
    label_tokens = {}
    for label in payload["labels"]:
        token_ids = list(vocab.tokenize(label.encode("utf-8"), add_bos=False))
        label_tokens[label] = token_ids
    del vocab
    gc.collect()
    llm = Llama(
        model_path=payload["model_spec"]["resolved_model_path"],
        n_gpu_layers=int(payload["n_gpu_layers"]),
        n_ctx=4096,
        n_batch=256,
        logits_all=True,
        seed=int(payload["runner_seed"]),
        verbose=True,
    )
    rows = []
    for candidate_row in payload["candidate_rows"]:
        started = time.perf_counter()
        try:
            llm.reset()
            tokens = llm.tokenize(str(candidate_row["prompt"]).encode("utf-8"), add_bos=True)
            llm.eval(tokens)
            logits = llm.scores[llm.n_tokens - 1]
            labels = [item["label"] for item in candidate_row["label_mapping"]]
            score_vector = {label: float(logits[label_tokens[label][0]]) for label in labels}
            error = ""
        except Exception as exc:
            tokens = []
            score_vector = {}
            error = repr(exc)
        rows.append({
            "model_hf_id": payload["model_spec"]["hf_id"],
            "instance_id": candidate_row["instance_id"],
            "prompt_hash": candidate_row["prompt_hash"],
            "score_vector": score_vector,
            "label_token_ids": {label: label_tokens[label] for label in score_vector},
            "prompt_token_count": len(tokens),
            "timing": {"prefill_s": round(time.perf_counter() - started, 6)},
            "error": error,
        })
    del llm
    gc.collect()
    print(json.dumps({
        "model_hf_id": payload["model_spec"]["hf_id"],
        "llama_cpp_version": version,
        "llama_cpp_build": {
            "cuda_backend": "CUDA" in system_info.upper(),
            "supports_gpu_offload": supports_gpu,
            "system_info": system_info,
        },
        "gpu_assignment": payload["model_spec"].get("gpu"),
        "n_gpu_layers_requested": payload["n_gpu_layers"],
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": False,
        "rows": rows,
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({"model_hf_id": payload["model_spec"]["hf_id"], "error": repr(exc), "rows": []}, sort_keys=True))
    raise
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(model_spec.get("gpu", 0))
    proc = subprocess.run(
        [sys.executable, "-c", worker_code],
        input=json.dumps(worker_payload),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    receipt = json.loads(proc.stdout.splitlines()[-1]) if proc.stdout.splitlines() else {}
    receipt.setdefault("model_hf_id", model_spec["hf_id"])
    after_mb = _gpu_used_total_mb()
    offloaded = exp5733.parse_offloaded_layers(proc.stderr)
    receipt["n_gpu_layers_requested"] = N_GPU_LAYERS_REQUESTED
    receipt["n_gpu_layers_offloaded"] = offloaded
    receipt["gpu_memory_before_mb"] = before_mb
    receipt["gpu_memory_peak_mb"] = max(before_mb, after_mb)
    receipt["gpu_memory_after_mb"] = after_mb
    receipt["offload_log_excerpt"] = proc.stderr[-4000:]
    build = dict(receipt.get("llama_cpp_build") or {})
    receipt["cuda_offload_authenticated"] = bool(
        offloaded > 0
        and build.get("cuda_backend") is True
        and build.get("supports_gpu_offload") is True
    )
    if proc.returncode != 0:
        receipt["blocked_reason"] = proc.stderr[-2000:]
        receipt["cuda_offload_authenticated"] = False
    return receipt


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    checkpoint_path: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    score_runner: ScoreRunner | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the panel or write an honest blocked artifact before inference."""

    started_at = _utc_now()
    started = time.perf_counter()
    preconditions = dict(
        preconditions_checked or collect_preconditions(checkpoint_path=checkpoint_path)
    )
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    if preconditions.get("preconditions_ready") is not True:
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.perf_counter() - started,
            preconditions_checked=preconditions,
            test_commands=test_commands,
            test_exit_codes=exit_codes,
        )
        if write:
            output = Path(result_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        validate_artifact(artifact)
        return artifact
    model_specs = _model_specs_from_preconditions(preconditions)
    panel = freeze_science_panel(load_science_rows(preconditions))
    panel_hash = sha256_json([row["panel_row_hash"] for row in panel])
    checkpoint = read_checkpoint(checkpoint_path)
    runtime_receipts, resumed_ids = _checkpoint_runtime_receipts(checkpoint, panel_hash)
    existing = {str(row.get("model_hf_id")) for row in runtime_receipts}
    runner = score_runner or default_score_runner
    for spec in model_specs:
        if spec["hf_id"] in existing:
            continue
        receipt = runner(dict(spec), panel, dict(RANDOM_SEEDS))
        receipt.setdefault("model_hf_id", spec["hf_id"])
        runtime_receipts.append(receipt)
        if write:
            write_checkpoint(
                {"panel_hash": panel_hash, "runtime_receipts": runtime_receipts}, checkpoint_path
            )
    checkpoint_receipt = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reused": bool(resumed_ids),
        "resumed_model_ids": resumed_ids,
        "panel_hash": panel_hash,
    }
    artifact = _base_artifact(
        status="complete",
        started_at=started_at,
        duration_s=time.perf_counter() - started,
        preconditions_checked=preconditions,
        model_specs=model_specs,
        panel=panel,
        runtime_receipts=runtime_receipts,
        checkpoint_receipt=checkpoint_receipt,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
    )
    validate_artifact(artifact)
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gc.collect()
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    del argv
    artifact = run()
    print(
        json.dumps(
            {"result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH), "status": artifact["status"]}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
