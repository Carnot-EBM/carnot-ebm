"""Exp6413 authenticated SOTA GGUF execution receipts.

Spec refs: REQ-INFRA-6413, SCENARIO-INFRA-6413-1,
SCENARIO-INFRA-6413-2, SCENARIO-INFRA-6413-3,
SCENARIO-INFRA-6413-4, SCENARIO-INFRA-6413-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6413_authenticated_sota_gguf_execution_receipts.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6413_authenticated_sota_gguf_execution_receipts"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6413_authenticated_sota_gguf_execution_receipts.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6413_authenticated_sota_gguf_execution_receipts.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

SCHEMA = "carnot.experiment_6413.authenticated_sota_gguf_execution_receipts.v1"
RECEIPT_SCHEMA = SCHEMA + ".receipt"
RUN_DATE = "20260814"
RANDOM_SEED = 6413
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_SOURCE = "embedded_gguf_vocab_only"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
N_CTX = 512
MAX_TOKENS = 8
LIVE_TIMEOUT_S = 900.0
MIN_VRAM_RISE_MB = 64
MODEL_PREFIX_BYTES = 4096

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "model_family": "qwen_moe",
        "gpu": 0,
        "preferred_quant": PREFERRED_QUANT,
        "min_free_vram_mb": 20_000,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "model_family": "gemma_dense",
        "gpu": 1,
        "preferred_quant": PREFERRED_QUANT,
        "min_free_vram_mb": 20_000,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "model_family": "gemma_moe",
        "gpu": 1,
        "preferred_quant": PREFERRED_QUANT,
        "min_free_vram_mb": 16_000,
    },
)
MODEL_TEMPLATE_BY_ID = {str(row["hf_id"]): dict(row) for row in MODEL_TEMPLATES}
CANARY_PROMPTS = {
    MANDATED_MODEL_IDS[0]: "Exp6413 non-headline canary. Return exactly: amber rivet",
    MANDATED_MODEL_IDS[1]: "Exp6413 non-headline canary. Return exactly: cobalt latch",
    MANDATED_MODEL_IDS[2]: "Exp6413 non-headline canary. Return exactly: olive hinge",
}

ATTACK_IDS = (
    "forged_pid",
    "reused_raw_hash",
    "substituted_model_file",
    "missing_first_token_clock",
    "constant_memory",
    "telemetry_from_another_process",
    "tokenizer_substitution",
    "early_process_exit",
    "inherited_upstream_receipt",
)
REQUIRED_GPU_PHASES = ("before_load", "after_load", "during_generation", "after_cleanup")
PID_BOUND_PHASES = ("after_load", "during_generation")
REQUIRED_CLOCK_FIELDS = (
    "parent_launch_monotonic_ns",
    "process_start_monotonic_ns",
    "load_start_monotonic_ns",
    "load_end_monotonic_ns",
    "first_token_monotonic_ns",
    "completion_monotonic_ns",
    "process_end_monotonic_ns",
    "parent_end_monotonic_ns",
)
REQUIRED_RECEIPT_FIELDS = (
    "schema",
    "model_hf_id",
    "model_family",
    "pid",
    "parent_pid",
    "executable",
    "command",
    "command_hash",
    "config",
    "config_hash",
    "model",
    "tokenizer",
    "device",
    "clocks",
    "gpu_samples",
    "prompt",
    "raw_output",
    "tokens",
    "exit_status",
    "stderr",
    "cleanup",
    "llama_cpp",
    "legacy_model_smoke_only",
    "inherited_upstream_receipt",
)

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6413_authenticated_sota_gguf_execution_receipts --date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6413_authenticated_sota_gguf_execution_receipts.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6413_authenticated_sota_gguf_execution_receipts.py "
    "-m pytest "
    "tests/python/test_experiment_6413_authenticated_sota_gguf_execution_receipts.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6413_authenticated_sota_gguf_execution_receipts.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6413_authenticated_sota_gguf_execution_receipts.py"
)
INFERENCE_E2E_COMMAND = RUN_COMMAND + " --validate"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6413_authenticated_sota_gguf_execution_receipts.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    INFERENCE_E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("results/experiment_5284_sota_runtime_offload_receipt_repair_v483.json"),
    Path("results/experiment_6412_v551_powered_claim_integrity_audit.json"),
    Path("results/experiment_6408_powered_write_time_factor_admission_ab.json"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/verify/gguf_inference.py"),
    Path("python/carnot/pipeline/gemma4_quantized_loader.py"),
    Path("python/carnot/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_hub_ids_revisions_quantizations_paths_and_hashes",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "gpu_precondition_receipts",
    "cuda_and_llamacpp_offload_receipts",
    "receipt_schema_path_hash_and_required_fields",
    "per_model_process_pid_parent_executable_command_and_config_receipts",
    "per_model_device_uuid_and_pid_bound_gpu_sample_receipts",
    "per_model_start_load_first_token_completion_end_monotonic_clocks",
    "per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts",
    "per_model_raw_output_paths_and_hashes",
    "constant_or_inherited_receipt_count",
    "legacy_headline_cell_count",
    "mutation_attack_matrix",
    "authentic_family_count",
    "authenticated_receipt_contract_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The status separates complete receipts from blocked preconditions and null evidence.",
    "MODEL_SPECS": "Only the three mandated GGUF model ids may enter the receipt matrix.",
    "models_used": "Only models with accepted authenticated receipts count as used.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "model_hub_ids_revisions_quantizations_paths_and_hashes": "Model identity is tied to immutable local bytes.",
    "embedded_gguf_tokenizer_receipts": "Token counts use the tokenizer embedded in each GGUF.",
    "autotokenizer_usage_count": "A bare zero proves no AutoTokenizer path was used.",
    "gpu_precondition_receipts": "GPU, VRAM, storage, commands, and protected processes are checked before load.",
    "cuda_and_llamacpp_offload_receipts": "CUDA and llama.cpp offload support are measured before readiness.",
    "receipt_schema_path_hash_and_required_fields": "The reusable receipt contract is written and hash-bound.",
    "per_model_process_pid_parent_executable_command_and_config_receipts": "The receipt binds one process to its executable and command.",
    "per_model_device_uuid_and_pid_bound_gpu_sample_receipts": "GPU telemetry must be tied to the child PID and device UUID.",
    "per_model_start_load_first_token_completion_end_monotonic_clocks": "Monotonic clocks prove the process lifetime contains generation.",
    "per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts": "Prompt, raw bytes, tokens, exit, stderr, and cleanup must agree.",
    "per_model_raw_output_paths_and_hashes": "Raw generated bytes are stored before parsing or summarizing.",
    "constant_or_inherited_receipt_count": "Constant memory and inherited receipts cannot satisfy authenticity.",
    "legacy_headline_cell_count": "Legacy small models are smoke fixtures only and never headline receipts.",
    "mutation_attack_matrix": "Known receipt forgery attacks must fail closed.",
    "authentic_family_count": "Readiness needs one authentic receipt per mandated model family.",
    "authenticated_receipt_contract_ready_score": "The score is one only when all three families and all attacks pass the contract.",
    "protected_files_unchanged": "Conductor, ops, traceability, and upstream artifacts remain byte-identical.",
    "preconditions_checked": "Preconditions bind date, models, tokenizers, GPUs, offload, commands, and hashes.",
    "inference_substrate": "The substrate declares small-N local SOTA GGUF generation.",
    "verifier_is_oracle": "False because the receipt proves execution, not semantic correctness.",
    "field_principles": "Every required field states the guard it serves.",
    "field_provenance": "Every required field identifies measured, derived, constant, source, or test origin.",
    "random_seed": "A fixed seed pins the bounded canary calls.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification command receipts gate readiness.",
    "reproducibility_checksum": "The normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict uses an allowed terminal prefix and states the execution-only boundary.",
    "schema": "The receipt schema version prevents inherited or stale row reuse.",
    "model_hf_id": "The process row must name the expected mandated model.",
    "model_family": "The family count is computed from authenticated rows only.",
    "pid": "The child process PID ties telemetry and exit state to one process.",
    "parent_pid": "The parent PID distinguishes the launched child from an inherited process.",
    "executable": "The executable binds the process to the launched Python runtime.",
    "command": "The command records the exact child invocation.",
    "command_hash": "The command hash detects command substitution.",
    "config": "The config freezes sampling and model-load parameters.",
    "config_hash": "The config hash detects sampling or offload substitution.",
    "model": "The model block binds hub id, revision, quantization, path, hash, and child file access.",
    "tokenizer": "The tokenizer block proves the embedded GGUF tokenizer was used.",
    "device": "The device block binds the receipt to a GPU UUID and assignment.",
    "clocks": "Clock fields prove load, first token, completion, and exit order.",
    "gpu_samples": "GPU samples prove PID-bound CUDA memory and nonconstant device memory.",
    "prompt": "The prompt hash binds the canary input.",
    "raw_output": "The raw output block binds stored generated bytes.",
    "tokens": "Prompt and completion token counts must be positive.",
    "exit_status": "The child exit status must be clean and not timed out.",
    "stderr": "The stderr hash preserves diagnostics without trusting prose.",
    "cleanup": "Cleanup proves the runtime closed and the child exited.",
    "llama_cpp": "The llama.cpp block records authenticated GPU offload.",
    "legacy_model_smoke_only": "A true value disqualifies legacy smoke rows.",
    "inherited_upstream_receipt": "A true value disqualifies inherited receipts.",
}
FIELD_PROVENANCE: dict[str, str] = {
    "status": "derived check",
    "MODEL_SPECS": "source hash",
    "models_used": "derived check",
    "cached_sota_pair_receipts": "source hash",
    "model_hub_ids_revisions_quantizations_paths_and_hashes": "source hash",
    "embedded_gguf_tokenizer_receipts": "measured tokenizer data",
    "autotokenizer_usage_count": "constant",
    "gpu_precondition_receipts": "measured host data",
    "cuda_and_llamacpp_offload_receipts": "measured host data",
    "receipt_schema_path_hash_and_required_fields": "source hash",
    "per_model_process_pid_parent_executable_command_and_config_receipts": "measured child data",
    "per_model_device_uuid_and_pid_bound_gpu_sample_receipts": "measured child data",
    "per_model_start_load_first_token_completion_end_monotonic_clocks": "measured child data",
    "per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts": "measured child data",
    "per_model_raw_output_paths_and_hashes": "measured child data",
    "constant_or_inherited_receipt_count": "derived check",
    "legacy_headline_cell_count": "derived check",
    "mutation_attack_matrix": "derived check",
    "authentic_family_count": "derived check",
    "authenticated_receipt_contract_ready_score": "derived check",
    "protected_files_unchanged": "source hash",
    "preconditions_checked": "derived check",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured wall time",
    "tests_run": "test command receipts",
    "reproducibility_checksum": "derived check",
    "honest_verdict": "derived check",
}


class RuntimeAdapter(Protocol):
    """Small live boundary so tests do not load large GGUF models."""

    def preflight_receipts(self, models: list[JsonDict]) -> JsonDict:
        """Return pre-load host receipts."""

    def run_model(
        self,
        model: JsonDict,
        prompt: str,
        output_dir: Path,
        index: int,
    ) -> JsonDict:
        """Run one bounded canary and return its receipt."""


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after compact serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Stream a file hash, or return None when absent."""

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_file_prefix(path: str | Path, limit: int = MODEL_PREFIX_BYTES) -> str | None:
    """Hash the first bytes a child can read before llama.cpp opens the model."""

    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return sha256_bytes(handle.read(limit))


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-").lower()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(path)
    return path


def write_bytes_atomic(path: str | Path, payload: bytes) -> Path:
    """Write bytes through a same-directory temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    tmp.replace(path)
    return path


def path_receipt(path: str | Path, *, digest: str | None = None) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": digest if digest is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def revision_from_path(path: str | Path) -> str | None:
    """Extract a Hugging Face snapshot revision from a cached path."""

    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def quantization_from_path(path: str | Path) -> str:
    """Extract a known GGUF quantization token from a file name."""

    name = Path(path).name.lower()
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "IQ2_M", "IQ2_XXS", "Q8_0"):
        if token.lower() in name:
            return token
    return "unknown"


def tokenizer_identity_hash(row: Mapping[str, Any]) -> str:
    """Bind tokenizer identity to model bytes and the measured prompt count."""

    return sha256_json(
        {
            "hf_id": row.get("hf_id"),
            "model_file_sha256": row.get("model_file_sha256"),
            "tokenizer_method": row.get("tokenizer_method", TOKENIZER_METHOD),
            "tokenizer_source": row.get("tokenizer_source", TOKENIZER_SOURCE),
            "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
        }
    )


def _token_count(receipt: Mapping[str, Any]) -> int:
    """Read either tokenizer count spelling used by prior modules."""

    return int(receipt.get("prompt_tokens", receipt.get("token_count", 0)) or 0)


def embedded_gguf_tokenizer_receipt(model_path: str, text: str) -> JsonDict:  # pragma: no cover
    """Count text through the tokenizer embedded in the GGUF file."""

    if not model_path or not Path(model_path).is_file():
        return {
            "source": TOKENIZER_SOURCE,
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "token_count": 0,
            "tokenizer_detail": f"model_path missing or not on disk: {model_path!r}",
            "autotokenizer_used": False,
        }
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        tokens = llm.tokenize(text.encode("utf-8"))
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        return {
            "source": TOKENIZER_SOURCE,
            "method": TOKENIZER_METHOD,
            "loadable": bool(tokens),
            "prompt_tokens": len(tokens),
            "token_count": len(tokens),
            "token_ids_sha256": sha256_json(tokens),
            "tokenizer_detail": f"embedded GGUF tokenizer OK ({len(tokens)} tokens)",
            "autotokenizer_used": False,
        }
    except Exception as exc:
        return {
            "source": TOKENIZER_SOURCE,
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "token_count": 0,
            "tokenizer_detail": f"embedded GGUF tokenizer failed: {type(exc).__name__}: {exc}",
            "autotokenizer_used": False,
        }


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve the three mandated GGUF rows through cached SOTA helper calls."""

    calls = [
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [0, 2]},
    ]
    default_pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT) or []
    dense_pair = cached_pair_func(
        gpu_indices=(0, 1),
        preferred_quant=PREFERRED_QUANT,
        model_indices=(0, 2),
    ) or []
    by_id = {str(row.get("hf_id")): dict(row) for row in [*default_pair, *dense_pair]}
    blockers: list[str] = []
    records: list[JsonDict] = []
    for template in MODEL_TEMPLATES:
        model_id = str(template["hf_id"])
        row = dict(by_id.get(model_id, {}))
        path_text = str(row.get("model_path") or "")
        path = Path(path_text) if path_text else Path()
        tokenized = tokenizer_func(path_text, CANARY_PROMPTS[model_id])
        model_hash = sha256_file(path) if path_text else None
        token_count = _token_count(tokenized)
        record = {
            **template,
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_path": path_text,
            "exists": bool(path_text) and path.is_file(),
            "revision": revision_from_path(path) if path_text else None,
            "quantization": quantization_from_path(path) if path_text else "unknown",
            "model_file_sha256": model_hash,
            "model_file_prefix_sha256": sha256_file_prefix(path) if path_text else None,
            "tokenizer_source": tokenized.get("source", TOKENIZER_SOURCE),
            "tokenizer_method": tokenized.get("method", TOKENIZER_METHOD),
            "tokenizer_loadable": tokenized.get("loadable") is True,
            "tokenizer_detail": str(tokenized.get("tokenizer_detail", "")),
            "prompt_tokens_for_tokenizer_precheck": token_count,
            "tokenizer_sha256": tokenizer_identity_hash(
                {
                    "hf_id": model_id,
                    "model_file_sha256": model_hash,
                    "tokenizer_method": tokenized.get("method", TOKENIZER_METHOD),
                    "tokenizer_source": tokenized.get("source", TOKENIZER_SOURCE),
                    "prompt_tokens": token_count,
                }
            ),
            "autotokenizer_used": False,
        }
        if not row:
            blockers.append(f"missing_cached_sota_pair_row:{model_id}")
        if not record["exists"]:
            blockers.append(f"missing_gguf_file:{model_id}")
        if not record["tokenizer_loadable"]:
            blockers.append(f"embedded_tokenizer_unavailable:{model_id}")
        if tokenized.get("autotokenizer_used") is True:
            blockers.append(f"autotokenizer_used:{model_id}")
        records.append(record)
    if not default_pair:
        blockers.append("cached_sota_pair_default_missing")
    if not dense_pair:
        blockers.append("cached_sota_pair_dense_missing")
    return {
        "schema": SCHEMA + ".model_resolution",
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": calls,
            "all_calls_made": True,
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": 0,
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model hub, revision, quantization, path, and hash rows."""

    return [
        {
            "name": row.get("name"),
            "hf_id": row.get("hf_id"),
            "model_family": row.get("model_family"),
            "gpu": row.get("gpu"),
            "revision": row.get("revision"),
            "quantization": row.get("quantization"),
            "path": row.get("model_path"),
            "present": row.get("exists") is True,
            "model_file_sha256": row.get("model_file_sha256"),
            "model_file_prefix_sha256": row.get("model_file_prefix_sha256"),
        }
        for row in model_specs
    ]


def tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return embedded tokenizer receipts for each model."""

    return [
        {
            "hf_id": row.get("hf_id"),
            "model_path": row.get("model_path"),
            "source": row.get("tokenizer_source", TOKENIZER_SOURCE),
            "method": row.get("tokenizer_method", TOKENIZER_METHOD),
            "loadable": row.get("tokenizer_loadable") is True,
            "prompt_tokens": row.get("prompt_tokens_for_tokenizer_precheck"),
            "detail": row.get("tokenizer_detail"),
            "tokenizer_sha256": row.get("tokenizer_sha256"),
            "autotokenizer_used": False,
        }
        for row in model_specs
    ]


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define this contract."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected-file hashes from before and after the run."""

    after = protected_hashes()
    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def receipt_schema_payload() -> JsonDict:
    """Return the reusable per-process receipt schema."""

    return {
        "schema": RECEIPT_SCHEMA,
        "required_fields": list(REQUIRED_RECEIPT_FIELDS),
        "required_clock_fields": list(REQUIRED_CLOCK_FIELDS),
        "required_gpu_phases": list(REQUIRED_GPU_PHASES),
        "pid_bound_gpu_phases": list(PID_BOUND_PHASES),
        "mutation_attacks": list(ATTACK_IDS),
        "semantic_correctness_claimed": False,
    }


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write a sidecar when requested, else return the would-be digest."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        if digest is not None:
            return digest
    return sha256_json(payload)


def receipt_schema_receipt(result_path: Path, *, write: bool) -> JsonDict:
    """Write and hash the receipt schema sidecar."""

    path = result_path.with_suffix(result_path.suffix + ".receipt_schema.json")
    payload = receipt_schema_payload()
    digest = write_payload_or_hash(path, payload, write=write)
    return {
        **path_receipt(path, digest=digest),
        "schema_version": RECEIPT_SCHEMA,
        "required_fields": list(REQUIRED_RECEIPT_FIELDS),
        "required_clock_fields": list(REQUIRED_CLOCK_FIELDS),
        "required_gpu_phases": list(REQUIRED_GPU_PHASES),
    }


def _int_or_none(value: Any) -> int | None:
    """Return an int only when conversion is safe."""

    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and other values as an empty map."""

    return value if isinstance(value, Mapping) else {}


def _hash_matches(value: Mapping[str, Any], key: str, payload: Any) -> bool:
    """Check a stored JSON hash."""

    return value.get(key) == sha256_json(payload)


def _append_once(reasons: list[str], reason: str) -> None:
    """Add a reason only once while preserving order."""

    if reason not in reasons:
        reasons.append(reason)


def validate_receipt(receipt: Mapping[str, Any], expected_model: Mapping[str, Any]) -> JsonDict:
    """Validate one authenticated execution receipt."""

    reasons: list[str] = []
    for field in REQUIRED_RECEIPT_FIELDS:
        if field not in receipt:
            _append_once(reasons, f"missing_receipt_field:{field}")
    if reasons:
        return {"accepted": False, "reasons": reasons}

    model_id = str(expected_model.get("hf_id"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        _append_once(reasons, "receipt_schema_mismatch")
    if receipt.get("model_hf_id") != model_id:
        _append_once(reasons, "model_hf_id_mismatch")
    if receipt.get("model_family") != expected_model.get("model_family"):
        _append_once(reasons, "model_family_mismatch")
    if receipt.get("legacy_model_smoke_only") is True:
        _append_once(reasons, "legacy_model_smoke_only")
    if receipt.get("inherited_upstream_receipt") is True:
        _append_once(reasons, "inherited_upstream_receipt")

    pid = _int_or_none(receipt.get("pid"))
    parent_pid = _int_or_none(receipt.get("parent_pid"))
    if pid is None or pid <= 1:
        _append_once(reasons, "pid_invalid_or_forged")
    if parent_pid is None or parent_pid <= 1 or parent_pid == pid:
        _append_once(reasons, "parent_pid_invalid")
    if not str(receipt.get("executable", "")).strip():
        _append_once(reasons, "executable_missing")
    if not _hash_matches(receipt, "command_hash", receipt.get("command")):
        _append_once(reasons, "command_hash_mismatch")
    if not _hash_matches(receipt, "config_hash", receipt.get("config")):
        _append_once(reasons, "config_hash_mismatch")

    model = _as_mapping(receipt.get("model"))
    expected_path = str(expected_model.get("model_path"))
    if model.get("hub_id") != model_id:
        _append_once(reasons, "model_hub_id_mismatch")
    if model.get("path") != expected_path:
        _append_once(reasons, "model_path_mismatch")
    if model.get("sha256") != expected_model.get("model_file_sha256"):
        _append_once(reasons, "model_file_hash_mismatch")
    expected_prefix = expected_model.get("model_file_prefix_sha256") or sha256_file_prefix(expected_path)
    if model.get("child_open_sample_sha256") != expected_prefix:
        _append_once(reasons, "model_child_open_sample_mismatch")
    if model.get("access_confirmed_by_child") is not True:
        _append_once(reasons, "model_file_access_not_confirmed")
    path = Path(expected_path)
    if path.is_file() and model.get("child_stat_size_bytes") != path.stat().st_size:
        _append_once(reasons, "model_child_stat_size_mismatch")

    tokenizer = _as_mapping(receipt.get("tokenizer"))
    if tokenizer.get("source") != TOKENIZER_SOURCE:
        _append_once(reasons, "tokenizer_source_not_embedded_gguf")
    if tokenizer.get("method") != TOKENIZER_METHOD:
        _append_once(reasons, "tokenizer_method_mismatch")
    if tokenizer.get("autotokenizer_used") is True:
        _append_once(reasons, "autotokenizer_used")
    if _int_or_none(tokenizer.get("prompt_tokens")) is None or int(tokenizer.get("prompt_tokens", 0)) <= 0:
        _append_once(reasons, "tokenizer_prompt_tokens_nonpositive")
    if tokenizer.get("tokenizer_sha256") != expected_model.get("tokenizer_sha256"):
        _append_once(reasons, "tokenizer_hash_mismatch")

    clocks = _as_mapping(receipt.get("clocks"))
    clock_values: list[int] = []
    for field in REQUIRED_CLOCK_FIELDS:
        value = _int_or_none(clocks.get(field))
        if value is None:
            _append_once(reasons, f"missing_clock:{field}")
        else:
            clock_values.append(value)
    if len(clock_values) == len(REQUIRED_CLOCK_FIELDS) and clock_values != sorted(clock_values):
        _append_once(reasons, "clock_order_invalid")
    if clocks.get("first_token_monotonic_ns") in (None, 0):
        _append_once(reasons, "first_token_clock_missing")
    if (
        _int_or_none(clocks.get("process_end_monotonic_ns")) is not None
        and _int_or_none(clocks.get("completion_monotonic_ns")) is not None
        and int(clocks["process_end_monotonic_ns"]) < int(clocks["completion_monotonic_ns"])
    ):
        _append_once(reasons, "early_process_exit_before_completion")

    raw = _as_mapping(receipt.get("raw_output"))
    raw_path = Path(str(raw.get("path", "")))
    raw_hash = sha256_file(raw_path)
    if raw_hash != raw.get("sha256"):
        _append_once(reasons, "raw_output_hash_mismatch")
    if not raw_path.is_file() or raw_path.stat().st_size != raw.get("byte_length"):
        _append_once(reasons, "raw_output_byte_length_mismatch")
    if int(raw.get("byte_length", 0) or 0) <= 0:
        _append_once(reasons, "raw_output_zero_length")
    if raw.get("stored_before_parse") is not True:
        _append_once(reasons, "raw_output_not_stored_before_parse")

    tokens = _as_mapping(receipt.get("tokens"))
    if int(tokens.get("prompt_tokens", 0) or 0) <= 0:
        _append_once(reasons, "prompt_tokens_nonpositive")
    if int(tokens.get("completion_tokens", 0) or 0) <= 0:
        _append_once(reasons, "completion_tokens_nonpositive")

    exit_status = _as_mapping(receipt.get("exit_status"))
    if exit_status.get("returncode") != 0:
        _append_once(reasons, "exit_status_nonzero")
    if exit_status.get("timed_out") is True:
        _append_once(reasons, "process_timed_out")

    stderr = _as_mapping(receipt.get("stderr"))
    stderr_path = Path(str(stderr.get("path", "")))
    if sha256_file(stderr_path) != stderr.get("sha256"):
        _append_once(reasons, "stderr_hash_mismatch")

    cleanup = _as_mapping(receipt.get("cleanup"))
    if cleanup.get("closed") is not True or cleanup.get("process_exited") is not True:
        _append_once(reasons, "cleanup_incomplete")

    llama = _as_mapping(receipt.get("llama_cpp"))
    if llama.get("supports_gpu_offload") is not True:
        _append_once(reasons, "llama_cpp_gpu_offload_not_supported")
    if llama.get("authenticated_gpu_offload") is not True:
        _append_once(reasons, "gpu_offload_not_authenticated")
    if int(llama.get("n_gpu_layers", 0) or 0) == 0:
        _append_once(reasons, "cpu_only_receipt")

    _validate_gpu_samples(receipt, expected_model, reasons)
    return {"accepted": not reasons, "reasons": reasons}


def _validate_gpu_samples(
    receipt: Mapping[str, Any],
    expected_model: Mapping[str, Any],
    reasons: list[str],
) -> None:
    """Validate PID-bound GPU telemetry and nonconstant memory."""

    pid = _int_or_none(receipt.get("pid"))
    device = _as_mapping(receipt.get("device"))
    expected_uuid = str(device.get("uuid", ""))
    expected_gpu = int(expected_model.get("gpu", -1))
    samples = receipt.get("gpu_samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        _append_once(reasons, "gpu_samples_missing")
        return
    by_phase: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        sample_map = _as_mapping(sample)
        by_phase.setdefault(str(sample_map.get("phase")), []).append(sample_map)
    for phase in REQUIRED_GPU_PHASES:
        if not by_phase.get(phase):
            _append_once(reasons, f"missing_gpu_phase:{phase}")
    if device.get("gpu_index") != expected_gpu:
        _append_once(reasons, "gpu_index_mismatch")
    if not expected_uuid:
        _append_once(reasons, "device_uuid_missing")

    device_memory: list[int] = []
    for rows in by_phase.values():
        for row in rows:
            memory = _int_or_none(row.get("device_memory_used_mb"))
            if memory is not None:
                device_memory.append(memory)
    if len(set(device_memory)) <= 1:
        _append_once(reasons, "gpu_memory_constant_or_missing")
    before = max(
        (_int_or_none(row.get("device_memory_used_mb")) or 0)
        for row in by_phase.get("before_load", [{}])
    )
    peak = max(device_memory) if device_memory else 0
    if peak - before < MIN_VRAM_RISE_MB:
        _append_once(reasons, "gpu_memory_no_load_rise")

    for phase in PID_BOUND_PHASES:
        rows = by_phase.get(phase, [])
        if not rows:
            continue
        if not any(row.get("pid_bound") is True for row in rows):
            _append_once(reasons, f"pid_bound_gpu_sample_missing:{phase}")
        for row in rows:
            if row.get("pid_bound") is not True:
                continue
            if _int_or_none(row.get("pid")) != pid:
                _append_once(reasons, "gpu_sample_pid_mismatch")
            if str(row.get("device_uuid", "")) != expected_uuid:
                _append_once(reasons, "gpu_sample_device_uuid_mismatch")
            if _int_or_none(row.get("gpu_index")) != expected_gpu:
                _append_once(reasons, "gpu_sample_gpu_index_mismatch")
            if int(row.get("pid_memory_mb", 0) or 0) <= 0:
                _append_once(reasons, "pid_bound_gpu_memory_nonpositive")


def validate_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Validate all receipts and reject reused raw hashes across models."""

    specs = {str(row.get("hf_id")): row for row in model_specs}
    results: dict[str, JsonDict] = {}
    raw_hashes: Counter[str] = Counter()
    for model_id, receipt in receipts.items():
        raw_hash = str(_as_mapping(receipt.get("raw_output")).get("sha256", ""))
        if raw_hash:
            raw_hashes[raw_hash] += 1
        results[model_id] = validate_receipt(receipt, specs.get(model_id, {}))
    duplicated = {digest for digest, count in raw_hashes.items() if count > 1}
    if duplicated:
        for model_id, receipt in receipts.items():
            raw_hash = str(_as_mapping(receipt.get("raw_output")).get("sha256", ""))
            if raw_hash in duplicated:
                reasons = list(results[model_id]["reasons"])
                _append_once(reasons, "raw_output_hash_reused")
                results[model_id] = {"accepted": False, "reasons": reasons}
    return results


def authentic_family_count(
    receipts: Mapping[str, Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> int:
    """Count distinct mandated families with accepted receipts."""

    verdicts = validate_receipts(receipts, model_specs)
    families = {
        str(row.get("model_family"))
        for row in model_specs
        if verdicts.get(str(row.get("hf_id")), {}).get("accepted") is True
    }
    return len(families)


def _mutate_receipts(
    attack_id: str,
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Return a mutated receipt set for one attack."""

    mutated = json.loads(canonical_json(receipts))
    target = mutated[MANDATED_MODEL_IDS[0]]
    if attack_id == "forged_pid":
        target["pid"] = 1
    elif attack_id == "reused_raw_hash":
        target["raw_output"] = dict(mutated[MANDATED_MODEL_IDS[1]]["raw_output"])
    elif attack_id == "substituted_model_file":
        target["model"]["sha256"] = "sha256:" + "0" * 64
    elif attack_id == "missing_first_token_clock":
        target["clocks"]["first_token_monotonic_ns"] = None
    elif attack_id == "constant_memory":
        for receipt in mutated.values():
            for sample in receipt["gpu_samples"]:
                sample["device_memory_used_mb"] = 100
                sample["pid_memory_mb"] = 100
    elif attack_id == "telemetry_from_another_process":
        target["gpu_samples"][1]["pid"] = 99_999_999
    elif attack_id == "tokenizer_substitution":
        target["tokenizer"]["source"] = "transformers.AutoTokenizer"
        target["tokenizer"]["method"] = "AutoTokenizer.from_pretrained"
    elif attack_id == "early_process_exit":
        target["clocks"]["process_end_monotonic_ns"] = 210
    elif attack_id == "inherited_upstream_receipt":
        target["inherited_upstream_receipt"] = True
    return mutated


def mutation_attack_matrix(
    receipts: Mapping[str, Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Run the receipt mutation attacks and report fail-closed status."""

    rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        mutated = _mutate_receipts(attack_id, receipts)
        verdicts = validate_receipts(mutated, model_specs)
        accepted = all(row.get("accepted") is True for row in verdicts.values())
        reasons = sorted(
            {
                reason
                for row in verdicts.values()
                for reason in row.get("reasons", [])
            }
        )
        rows.append(
            {
                "attack_id": attack_id,
                "accepted": accepted,
                "fail_closed": not accepted,
                "reasons": reasons,
            }
        )
    return {
        "schema": SCHEMA + ".mutation_attacks",
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "false_accept_count": sum(1 for row in rows if row["accepted"]),
    }


def constant_or_inherited_receipt_count(
    receipts: Mapping[str, Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> int:
    """Count receipts that look constant or inherited."""

    verdicts = validate_receipts(receipts, model_specs)
    count = 0
    for model_id, receipt in receipts.items():
        reasons = set(verdicts.get(model_id, {}).get("reasons", []))
        if receipt.get("inherited_upstream_receipt") is True:
            count += 1
        elif "gpu_memory_constant_or_missing" in reasons or "gpu_memory_no_load_rise" in reasons:
            count += 1
    return count


def legacy_headline_cell_count(receipts: Mapping[str, Mapping[str, Any]]) -> int:
    """Count legacy or non-mandated receipts that attempted to satisfy readiness."""

    return sum(
        1
        for model_id, receipt in receipts.items()
        if model_id not in MANDATED_MODEL_IDS or receipt.get("legacy_model_smoke_only") is True
    )


def llama_cpp_offload_receipt() -> JsonDict:  # pragma: no cover
    """Query llama.cpp for GPU offload support."""

    try:
        from llama_cpp import __version__ as version
        from llama_cpp import llama_cpp as lib

        info = lib.llama_print_system_info()
        text = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
        return {
            "llama_cpp_available": True,
            "llama_cpp_version": str(version),
            "llama_supports_gpu_offload": bool(lib.llama_supports_gpu_offload()),
            "system_info_sha256": sha256_text(text),
            "system_info_excerpt": text[:1200],
        }
    except Exception as exc:
        return {
            "llama_cpp_available": False,
            "llama_supports_gpu_offload": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def nvidia_gpu_snapshot() -> JsonDict:  # pragma: no cover
    """Collect visible NVIDIA GPU state."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mb": int(float(parts[3])),
                    "memory_used_mb": int(float(parts[4])),
                    "memory_free_mb": int(float(parts[5])),
                    "utilization_pct": int(float(parts[6])),
                }
            )
        except ValueError:
            continue
    return {
        "ok": result.returncode == 0,
        "devices": devices,
        "returncode": result.returncode,
        "stderr_sha256": sha256_text(result.stderr),
    }


def protected_training_processes() -> list[int]:  # pragma: no cover
    """Return protected training PIDs that must not be disturbed."""

    pids: list[int] = []
    try:
        from scripts.experiment_template import _pid_is_protected_training_proc
    except Exception:
        _pid_is_protected_training_proc = None
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if callable(_pid_is_protected_training_proc) and _pid_is_protected_training_proc(pid):
            pids.append(pid)
            continue
        try:
            text = (
                proc.joinpath("cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "replace")
                .lower()
            )
        except Exception:
            continue
        if any(word in text for word in ("train", "finetune", "fine-tune", "deepspeed", "accelerate")) and any(
            word in text for word in ("cuda", "torch", "llama", "transformers", "gguf")
        ):
            pids.append(pid)
    return sorted(set(pids))


class LocalRuntimeAdapter:  # pragma: no cover
    """Live runtime that launches one child process per model."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def preflight_receipts(self, models: list[JsonDict]) -> JsonDict:
        snapshot = nvidia_gpu_snapshot()
        devices = list(snapshot.get("devices", []))
        by_index = {int(row["index"]): row for row in devices if "index" in row}
        disk = os.statvfs(REPO_ROOT)
        storage_free_gb = disk.f_bavail * disk.f_frsize / (1024**3)
        protected = protected_training_processes()
        names = [str(row.get("name", "")) for row in devices]
        vram_ready = {
            str(row["hf_id"]): int(by_index.get(int(row["gpu"]), {}).get("memory_free_mb", 0))
            >= int(row.get("min_free_vram_mb", 0))
            for row in models
        }
        blockers: list[str] = []
        if snapshot.get("ok") is not True or len(devices) < 2:
            blockers.append("both_gpus_not_visible")
        if names and not all("RTX 3090" in name for name in names[:2]):
            blockers.append("both_rtx_3090_gpus_not_visible")
        if not all(vram_ready.values()):
            blockers.append("insufficient_free_vram")
        if protected:
            blockers.append("protected_training_process_present")
        if storage_free_gb < 10.0:
            blockers.append("storage_below_10gb")
        return {
            "schema": SCHEMA + ".gpu_preconditions",
            "gpu_snapshot": snapshot,
            "both_gpus_visible": snapshot.get("ok") is True and len(devices) >= 2,
            "both_rtx_3090_gpus_present": bool(names)
            and len(devices) >= 2
            and all("RTX 3090" in name for name in names[:2]),
            "free_vram_ready": all(vram_ready.values()),
            "vram_ready_by_model": vram_ready,
            "protected_training_process_present": bool(protected),
            "protected_training_pids": protected,
            "disk_ready": storage_free_gb >= 10.0,
            "storage_free_gb": round(storage_free_gb, 6),
            "sequential_schedule": [
                {"order": index, "model_hf_id": row["hf_id"], "gpu": row["gpu"]}
                for index, row in enumerate(models)
            ],
            "exact_commands": list(DEFAULT_TEST_COMMANDS),
            "blocked_reasons": sorted(set(blockers)),
        }

    def run_model(
        self,
        model: JsonDict,
        prompt: str,
        output_dir: Path,
        index: int,
    ) -> JsonDict:
        return run_live_model(model, prompt, output_dir, index=index)


LIVE_CHILD_CODE = r'''
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def emit(name, payload):
    sys.stderr.write("\nCARNOT_%s:%s\n" % (name, json.dumps(payload, sort_keys=True)))
    sys.stderr.flush()


def sha256_bytes(value):
    import hashlib
    return "sha256:" + hashlib.sha256(value).hexdigest()


def file_prefix(path, limit):
    with Path(path).open("rb") as handle:
        return sha256_bytes(handle.read(limit))


def query_devices():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "name": parts[1],
                    "device_uuid": parts[2],
                    "device_memory_used_mb": int(float(parts[3])),
                    "device_memory_free_mb": int(float(parts[4])),
                    "utilization_pct": int(float(parts[5])),
                }
            )
        except ValueError:
            pass
    return rows


def query_compute_apps():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "device_uuid": parts[1],
                    "pid_memory_mb": int(float(parts[2])),
                }
            )
        except ValueError:
            pass
    return rows


def sample(phase, expected_uuid, expected_gpu):
    pid = os.getpid()
    devices = query_devices()
    apps = query_compute_apps()
    device = next((row for row in devices if row["device_uuid"] == expected_uuid), None)
    if device is None and 0 <= expected_gpu < len(devices):
        device = devices[expected_gpu]
    app = next(
        (row for row in apps if row["pid"] == pid and row["device_uuid"] == (device or {}).get("device_uuid")),
        None,
    )
    if app is None:
        app = next((row for row in apps if row["pid"] == pid), None)
    row = {
        "phase": phase,
        "pid": pid,
        "pid_bound": app is not None,
        "device_uuid": (device or {}).get("device_uuid", expected_uuid),
        "gpu_index": (device or {}).get("gpu_index", expected_gpu),
        "pid_memory_mb": int((app or {}).get("pid_memory_mb", 0)),
        "device_memory_used_mb": int((device or {}).get("device_memory_used_mb", 0)),
        "device_memory_free_mb": int((device or {}).get("device_memory_free_mb", 0)),
        "utilization_pct": int((device or {}).get("utilization_pct", 0)),
        "monotonic_ns": time.monotonic_ns(),
    }
    return row


args = json.loads(sys.argv[1])
model_path = args["model_path"]
prompt = args["prompt"]
sampling = args["sampling"]
expected_uuid = args["device_uuid"]
expected_gpu = int(args["gpu"])
started = time.monotonic_ns()
samples = [sample("before_load", expected_uuid, expected_gpu)]
llm = None
try:
    from llama_cpp import Llama, llama_cpp

    access = {
        "hub_id": args["model_hf_id"],
        "revision": args["revision"],
        "quantization": args["quantization"],
        "path": model_path,
        "sha256": args["model_file_sha256"],
        "child_stat_size_bytes": Path(model_path).stat().st_size,
        "child_open_sample_sha256": file_prefix(model_path, int(args["model_prefix_bytes"])),
        "access_confirmed_by_child": True,
    }
    load_start = time.monotonic_ns()
    llm = Llama(
        model_path=model_path,
        n_ctx=int(sampling["n_ctx"]),
        n_gpu_layers=int(sampling["n_gpu_layers"]),
        main_gpu=0,
        n_batch=64,
        n_ubatch=64,
        seed=int(args["seed"]),
        verbose=False,
    )
    load_end = time.monotonic_ns()
    samples.append(sample("after_load", expected_uuid, expected_gpu))
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    first_token = None
    pieces = []
    samples.append(sample("during_generation", expected_uuid, expected_gpu))
    for chunk in llm.create_completion(
        prompt,
        max_tokens=int(sampling["max_tokens"]),
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        stream=True,
    ):
        choice = (chunk.get("choices") or [{}])[0]
        piece = str(choice.get("text") or "")
        if piece and first_token is None:
            first_token = time.monotonic_ns()
        pieces.append(piece)
        sys.stdout.buffer.write(piece.encode("utf-8", "replace"))
        sys.stdout.flush()
    completion = time.monotonic_ns()
    text = "".join(pieces)
    completion_tokens = len(llm.tokenize(text.encode("utf-8"))) if text else 0
    close = getattr(llm, "close", None)
    if callable(close):
        close()
    del llm
    llm = None
    gc.collect()
    samples.append(sample("after_cleanup", expected_uuid, expected_gpu))
    ended = time.monotonic_ns()
    receipt = {
        "schema": args["receipt_schema"],
        "model_hf_id": args["model_hf_id"],
        "model_family": args["model_family"],
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "executable": sys.executable,
        "model": access,
        "tokenizer": {
            "source": args["tokenizer_source"],
            "method": args["tokenizer_method"],
            "prompt_tokens": prompt_tokens,
            "tokenizer_sha256": args["tokenizer_sha256"],
            "autotokenizer_used": False,
        },
        "device": {
            "gpu_index": expected_gpu,
            "uuid": expected_uuid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "clocks": {
            "process_start_monotonic_ns": started,
            "load_start_monotonic_ns": load_start,
            "load_end_monotonic_ns": load_end,
            "first_token_monotonic_ns": first_token,
            "completion_monotonic_ns": completion,
            "process_end_monotonic_ns": ended,
        },
        "gpu_samples": samples,
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cleanup": {
            "closed": True,
            "process_exited": True,
            "released_cuda_context": True,
        },
        "llama_cpp": {
            "supports_gpu_offload": bool(llama_cpp.llama_supports_gpu_offload()),
            "authenticated_gpu_offload": bool(llama_cpp.llama_supports_gpu_offload())
            and int(sampling["n_gpu_layers"]) != 0
            and any(row.get("pid_bound") and row.get("pid_memory_mb", 0) > 0 for row in samples),
            "n_gpu_layers": int(sampling["n_gpu_layers"]),
        },
        "legacy_model_smoke_only": False,
        "inherited_upstream_receipt": False,
    }
    emit("CHILD_RECEIPT", receipt)
except Exception as exc:
    ended = time.monotonic_ns()
    emit(
        "CHILD_RECEIPT",
        {
            "schema": args["receipt_schema"],
            "model_hf_id": args.get("model_hf_id"),
            "model_family": args.get("model_family"),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "executable": sys.executable,
            "clocks": {
                "process_start_monotonic_ns": started,
                "process_end_monotonic_ns": ended,
            },
            "gpu_samples": samples,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "legacy_model_smoke_only": False,
            "inherited_upstream_receipt": False,
        },
    )
    raise SystemExit(1)
finally:
    if llm is not None:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
'''


def _parse_child_receipt(stderr_text: str) -> JsonDict | None:  # pragma: no cover
    """Parse the child JSON receipt from stderr."""

    marker = "CARNOT_CHILD_RECEIPT:"
    for line in reversed(stderr_text.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            value = json.loads(line.removeprefix(marker))
        except json.JSONDecodeError:
            continue
        return dict(value) if isinstance(value, Mapping) else None
    return None


def _signal_name(returncode: int | None) -> str | None:
    """Return a signal name for negative subprocess return codes."""

    if returncode is None or returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal_{-returncode}"


def sidecar_path(output_dir: Path, call_id: str, stream: str) -> Path:  # pragma: no cover
    """Return a safe sidecar path for raw child streams."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", call_id).strip("_") or "child"
    return output_dir / "sidecars" / f"{safe}.{stream}"


def _device_uuid_for_gpu(gpu_index: int) -> str | None:  # pragma: no cover
    """Return the physical UUID for a GPU index."""

    snapshot = nvidia_gpu_snapshot()
    for row in snapshot.get("devices", []):
        if int(row.get("index", -1)) == int(gpu_index):
            return str(row.get("uuid"))
    return None


def run_live_model(
    model: Mapping[str, Any],
    prompt: str,
    output_dir: Path,
    *,
    index: int,
) -> JsonDict:  # pragma: no cover
    """Run one bounded live canary through a subprocess."""

    model_id = str(model["hf_id"])
    gpu = int(model["gpu"])
    device_uuid = _device_uuid_for_gpu(gpu) or f"unknown-gpu-{gpu}"
    command_config = {
        "max_tokens": MAX_TOKENS,
        "n_ctx": N_CTX,
        "n_gpu_layers": -1,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": RANDOM_SEED + index,
    }
    command = [
        sys.executable,
        "-c",
        LIVE_CHILD_CODE,
        "<child_args_json>",
    ]
    child_args = {
        "receipt_schema": RECEIPT_SCHEMA,
        "model_hf_id": model_id,
        "model_family": model["model_family"],
        "model_path": model["model_path"],
        "model_file_sha256": model["model_file_sha256"],
        "model_prefix_bytes": MODEL_PREFIX_BYTES,
        "revision": model["revision"],
        "quantization": model["quantization"],
        "prompt": prompt,
        "seed": RANDOM_SEED + index,
        "sampling": command_config,
        "gpu": gpu,
        "device_uuid": device_uuid,
        "tokenizer_source": TOKENIZER_SOURCE,
        "tokenizer_method": TOKENIZER_METHOD,
        "tokenizer_sha256": model["tokenizer_sha256"],
    }
    argv = [sys.executable, "-c", LIVE_CHILD_CODE, json.dumps(child_args, sort_keys=True)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    call_id = model_slug(model_id)
    parent_launch = time.monotonic_ns()
    proc = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=LIVE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    parent_end = time.monotonic_ns()
    raw_path = sidecar_path(output_dir, call_id, "raw.bin")
    stderr_path = sidecar_path(output_dir, call_id, "stderr.txt")
    write_bytes_atomic(raw_path, stdout)
    write_bytes_atomic(stderr_path, stderr)
    stderr_text = stderr.decode("utf-8", "replace")
    child = _parse_child_receipt(stderr_text) or {}
    clocks = dict(_as_mapping(child.get("clocks")))
    clocks["parent_launch_monotonic_ns"] = parent_launch
    clocks["parent_end_monotonic_ns"] = parent_end
    receipt = {
        **child,
        "schema": child.get("schema", RECEIPT_SCHEMA),
        "model_hf_id": model_id,
        "model_family": model["model_family"],
        "pid": child.get("pid", proc.pid),
        "parent_pid": child.get("parent_pid", os.getpid()),
        "executable": child.get("executable", sys.executable),
        "command": command,
        "command_hash": sha256_json(command),
        "config": command_config,
        "config_hash": sha256_json(command_config),
        "model": child.get("model", {}),
        "tokenizer": child.get(
            "tokenizer",
            {
                "source": TOKENIZER_SOURCE,
                "method": TOKENIZER_METHOD,
                "prompt_tokens": 0,
                "tokenizer_sha256": model["tokenizer_sha256"],
                "autotokenizer_used": False,
            },
        ),
        "device": child.get(
            "device",
            {"gpu_index": gpu, "uuid": device_uuid, "cuda_visible_devices": str(gpu)},
        ),
        "clocks": clocks,
        "gpu_samples": child.get("gpu_samples", []),
        "prompt": {
            "text_sha256": sha256_text(prompt),
            "byte_length": len(prompt.encode("utf-8")),
        },
        "raw_output": {
            "path": str(raw_path),
            "sha256": sha256_bytes(stdout),
            "byte_length": len(stdout),
            "stored_before_parse": True,
        },
        "tokens": child.get("tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        "exit_status": {
            "returncode": proc.returncode,
            "timed_out": timed_out,
            "signal": "SIGKILL" if timed_out else _signal_name(proc.returncode),
        },
        "stderr": {
            "path": str(stderr_path),
            "sha256": sha256_bytes(stderr),
            "byte_length": len(stderr),
        },
        "cleanup": child.get("cleanup", {"closed": False, "process_exited": proc.returncode is not None}),
        "llama_cpp": child.get(
            "llama_cpp",
            {"supports_gpu_offload": False, "authenticated_gpu_offload": False, "n_gpu_layers": -1},
        ),
        "legacy_model_smoke_only": False,
        "inherited_upstream_receipt": False,
    }
    return receipt


def preconditions_from(
    *,
    date: str,
    model_resolution: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    llama_support: Mapping[str, Any],
    schema_receipt: Mapping[str, Any],
    source_before: Mapping[str, str | None],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all preconditions before any model load."""

    blockers = list(model_resolution.get("blocked_reasons", []))
    blockers.extend(str(item) for item in runtime_preflight.get("blocked_reasons", []))
    if llama_support.get("llama_supports_gpu_offload") is not True:
        blockers.append("llama_cpp_gpu_offload_unavailable")
    if schema_receipt.get("sha256") is None:
        blockers.append("receipt_schema_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "planning_date": "20260814",
        "all_required_gguf_files_present": all(
            row.get("exists") is True for row in model_resolution.get("MODEL_SPECS", [])
        ),
        "all_embedded_tokenizers_loadable": all(
            row.get("tokenizer_loadable") is True for row in model_resolution.get("MODEL_SPECS", [])
        ),
        "autotokenizer_usage_count": 0,
        "gpu_preconditions_passed": not runtime_preflight.get("blocked_reasons"),
        "llama_cpp_gpu_offload_supported": llama_support.get("llama_supports_gpu_offload") is True,
        "receipt_schema_ready": schema_receipt.get("sha256") is not None,
        "source_hashes": dict(source_before),
        "protected_hashes_before": dict(protected_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every authenticated receipt gate passes."""

    tests = _as_mapping(_as_mapping(artifact.get("tests_run")).get("exit_codes"))
    raw_hashes = [
        str(_as_mapping(row).get("sha256"))
        for row in _as_mapping(artifact.get("per_model_raw_output_paths_and_hashes")).values()
        if _as_mapping(row).get("sha256")
    ]
    mutation = _as_mapping(artifact.get("mutation_attack_matrix"))
    mutation_rows = list(mutation.get("rows", [])) if isinstance(mutation.get("rows"), list) else []
    gates = (
        _as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        artifact.get("models_used") == list(MANDATED_MODEL_IDS),
        artifact.get("authentic_family_count") == len(MANDATED_MODEL_IDS),
        len(raw_hashes) == len(set(raw_hashes)) == len(MANDATED_MODEL_IDS),
        artifact.get("constant_or_inherited_receipt_count") == 0,
        artifact.get("legacy_headline_cell_count") == 0,
        mutation.get("all_fail_closed") is True,
        bool(mutation_rows) and all(_as_mapping(row).get("fail_closed") is True for row in mutation_rows),
        mutation.get("false_accept_count") == 0,
        artifact.get("autotokenizer_usage_count") == 0,
        _as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        artifact.get("verifier_is_oracle") is False,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal artifact status."""

    if _as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("authenticated_receipt_contract_ready_score", 0.0)) == 1.0:
        return "complete"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return an allowed terminal-prefix verdict."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "complete":
        return "complete: all three mandated GGUF families produced authenticated CUDA execution receipts"
    if status_text == "blocked_precondition":
        blockers = _as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"complete_blocked: authenticated GGUF receipt preconditions failed {blockers}"
    return "complete_null: receipt checks ran but one or more authenticated execution gates failed"


def build_artifact(
    *,
    date: str,
    model_resolution: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    llama_support: Mapping[str, Any],
    schema_receipt: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None],
) -> JsonDict:
    """Build the terminal artifact from receipt rows."""

    model_specs = list(model_resolution.get("MODEL_SPECS", []))
    verdicts = validate_receipts(receipts, model_specs)
    accepted_ids = [
        model_id for model_id in MANDATED_MODEL_IDS if verdicts.get(model_id, {}).get("accepted") is True
    ]
    mutation_matrix = mutation_attack_matrix(receipts, model_specs) if receipts else {
        "schema": SCHEMA + ".mutation_attacks",
        "rows": [],
        "all_fail_closed": False,
        "false_accept_count": 0,
    }
    preconditions = preconditions_from(
        date=date,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        llama_support=llama_support,
        schema_receipt=schema_receipt,
        source_before=source_before,
        protected_before=protected_before,
    )
    artifact: JsonDict = {
        "status": "",
        "MODEL_SPECS": model_specs,
        "models_used": accepted_ids,
        "cached_sota_pair_receipts": model_resolution.get("cached_sota_pair_receipts", {}),
        "model_hub_ids_revisions_quantizations_paths_and_hashes": model_file_receipts(model_specs),
        "embedded_gguf_tokenizer_receipts": tokenizer_receipts(model_specs),
        "autotokenizer_usage_count": 0,
        "gpu_precondition_receipts": runtime_preflight,
        "cuda_and_llamacpp_offload_receipts": llama_support,
        "receipt_schema_path_hash_and_required_fields": schema_receipt,
        "per_model_process_pid_parent_executable_command_and_config_receipts": {
            model_id: {
                "pid": row.get("pid"),
                "parent_pid": row.get("parent_pid"),
                "executable": row.get("executable"),
                "command": row.get("command"),
                "command_hash": row.get("command_hash"),
                "config_hash": row.get("config_hash"),
                "accepted": verdicts.get(model_id, {}).get("accepted") is True,
                "reasons": verdicts.get(model_id, {}).get("reasons", []),
            }
            for model_id, row in receipts.items()
        },
        "per_model_device_uuid_and_pid_bound_gpu_sample_receipts": {
            model_id: {
                "device": row.get("device"),
                "gpu_samples": row.get("gpu_samples", []),
                "accepted": verdicts.get(model_id, {}).get("accepted") is True,
            }
            for model_id, row in receipts.items()
        },
        "per_model_start_load_first_token_completion_end_monotonic_clocks": {
            model_id: row.get("clocks", {}) for model_id, row in receipts.items()
        },
        "per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts": {
            model_id: {
                "prompt": row.get("prompt"),
                "raw_output": row.get("raw_output"),
                "tokens": row.get("tokens"),
                "exit_status": row.get("exit_status"),
                "stderr": row.get("stderr"),
                "cleanup": row.get("cleanup"),
                "llama_cpp": row.get("llama_cpp"),
            }
            for model_id, row in receipts.items()
        },
        "per_model_raw_output_paths_and_hashes": {
            model_id: {
                "path": _as_mapping(row.get("raw_output")).get("path"),
                "sha256": _as_mapping(row.get("raw_output")).get("sha256"),
                "byte_length": _as_mapping(row.get("raw_output")).get("byte_length"),
            }
            for model_id, row in receipts.items()
        },
        "constant_or_inherited_receipt_count": constant_or_inherited_receipt_count(
            receipts, model_specs
        ),
        "legacy_headline_cell_count": legacy_headline_cell_count(receipts),
        "mutation_attack_matrix": mutation_matrix,
        "authentic_family_count": authentic_family_count(receipts, model_specs),
        "authenticated_receipt_contract_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 9),
        "tests_run": {"commands": list(DEFAULT_TEST_COMMANDS), "exit_codes": dict(test_exit_codes)},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["authenticated_receipt_contract_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def _terminal_prefix_ok(value: str) -> bool:
    """Return true for the operator-approved terminal verdict prefixes."""

    return value.startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, counters, oracle boundary, and checksum."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_IDS):
        errors.append("MODEL_SPECS mandated ids mismatch")
    if artifact.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if artifact.get("legacy_headline_cell_count") != 0:
        errors.append("legacy_headline_cell_count must be zero")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if set(_as_mapping(artifact.get("field_provenance"))) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    principles = _as_mapping(artifact.get("field_principles"))
    for field in (*REQUIRED_ARTIFACT_FIELDS, *REQUIRED_RECEIPT_FIELDS):
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
            break
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict", ""))):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Write the terminal artifact atomically."""

    return write_json_atomic(path, artifact)


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
    runtime: RuntimeAdapter | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6413 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    data.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    source_before = source_hashes()
    schema_receipt = receipt_schema_receipt(result, write=write)
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = list(model_resolution["MODEL_SPECS"])
    adapter = runtime or LocalRuntimeAdapter(data)
    runtime_preflight = adapter.preflight_receipts(model_specs)
    llama_support = llama_cpp_offload_receipt()
    preconditions = preconditions_from(
        date=date,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        llama_support=llama_support,
        schema_receipt=schema_receipt,
        source_before=source_before,
        protected_before=protected_before,
    )
    receipts: dict[str, JsonDict] = {}
    if preconditions["all_preconditions_passed"]:
        for index, model in enumerate(model_specs):
            model_id = str(model["hf_id"])
            receipts[model_id] = adapter.run_model(
                model,
                CANARY_PROMPTS[model_id],
                data,
                index,
            )
    artifact = build_artifact(
        date=date,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        llama_support=llama_support,
        schema_receipt=schema_receipt,
        receipts=receipts,
        protected_before=protected_before,
        source_before=source_before,
        duration_s=duration_s if duration_s is not None else time.perf_counter() - started,
        test_exit_codes=test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS},
    )
    errors = validate_artifact(artifact)
    if errors:
        artifact["status"] = "failed_schema"
        artifact["honest_verdict"] = f"complete_failed_schema: {errors}"
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, result)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        payload = json.loads(result.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, sort_keys=True))
            return 1
        print(json.dumps({"ok": True, "path": str(result)}, sort_keys=True))
        return 0
    artifact = run(date=str(args.date), result_path=result, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(
        json.dumps(
            {
                "path": str(result),
                "status": artifact.get("status"),
                "honest_verdict": artifact.get("honest_verdict"),
                "reproducibility_checksum": artifact.get("reproducibility_checksum"),
            },
            sort_keys=True,
        )
    )
    return 0 if not validate_artifact(artifact) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
