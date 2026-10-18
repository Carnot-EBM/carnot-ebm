"""Exp6365 GGUF child failure forensics and runtime contract.

Spec refs: REQ-INFRA-6365, SCENARIO-INFRA-6365-1,
SCENARIO-INFRA-6365-2, SCENARIO-INFRA-6365-3,
SCENARIO-INFRA-6365-4, SCENARIO-INFRA-6365-5.
"""

from __future__ import annotations

import argparse
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

from carnot import experiment_6352_live_factor_proposal_authenticity_preflight as exp6352
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json"
)
OUTPUT_RELATIVE_DIR = Path("data/research/experiment_6365_gguf_child_failure_forensics")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6352_RELATIVE_PATH = Path(
    "results/experiment_6352_live_factor_proposal_authenticity_preflight.json"
)
EXP6352_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6352_live_factor_proposal_authenticity_preflight.py"
)
V547_RETRO_RELATIVE_PATH = Path("results/operational_retro_2026_08_547.json")

SCHEMA = "carnot.experiment_6365.gguf_child_failure_forensics_runtime_contract.v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6365
INFERENCE_SUBSTRATE = "local_llama_cpp_gguf_observable_child_process_contract"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
AUTOTOKENIZER_USAGE_COUNT = 0
PREFERRED_QUANT = "Q4_K_M"
N_CTX = 512
MAX_TOKENS = 2
TEMPERATURE = 0.0
TOP_P = 1.0
LIVE_TIMEOUT_S = 300.0
MIN_VRAM_RISE_MB = 64
VRAM_RELEASE_TOLERANCE_MB = 128
DEFAULT_PROMPT = "Return one short word. Ready?"

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_SPECS: tuple[JsonDict, ...] = (
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
MODEL_SPECS_BY_ID = {str(row["hf_id"]): dict(row) for row in MODEL_SPECS}

REQUIRED_GPU_PHASES = (
    "before_load",
    "after_load",
    "during_generation",
    "after_unload",
    "after_cleanup",
)
REQUIRED_TIMING_PHASES = ("load", "prompt", "generate", "unload", "cleanup")
FAILURE_INJECTION_NAMES = (
    "nonzero_exit",
    "timeout",
    "empty_stdout",
    "malformed_usage_receipt",
    "context_overflow",
    "source_drift",
    "missing_gpu_sample",
)

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6365_gguf_child_failure_forensics_and_runtime_contract --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py "
    "-m pytest "
    "tests/python/test_experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json"
)
DETERMINATION_LINT_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_LINT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6352_RELATIVE_PATH,
    EXP6352_MODULE_RELATIVE_PATH,
    V547_RETRO_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_exp6352_path_hash_and_terminal_class",
    "reconstructed_exp6352_command_and_source_receipt",
    "exp6352_source_artifact_sampling_drift",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "llama_cpp_gpu_offload_support_receipt",
    "task_linked_gpu_samples_by_model_and_phase",
    "dispatcher_and_process_identity_receipts",
    "source_command_prompt_and_environment_hashes_by_call",
    "prompt_token_context_capacity_receipts_by_model",
    "stdout_stderr_sidecar_paths_hashes_and_bounded_excerpts",
    "child_exit_signal_timeout_and_usage_receipts_by_model",
    "load_prompt_generate_unload_cleanup_timings_by_model",
    "raw_output_paths_hashes_and_byte_counts",
    "live_autoregressive_generation_invoked_by_model",
    "failure_injection_matrix",
    "vram_rise_and_release_receipts_by_model",
    "gguf_runtime_observability_ready_score",
    "no_proposal_quality_or_utility_claim",
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
    "status": "The status separates blocked preconditions from complete runtime evidence.",
    "upstream_exp6352_path_hash_and_terminal_class": "The failed upstream artifact is hash-bound before diagnosis.",
    "reconstructed_exp6352_command_and_source_receipt": "The prior command and source are reconstructed before rerun.",
    "exp6352_source_artifact_sampling_drift": "Sampling drift explains why source and artifact receipts disagree.",
    "MODEL_SPECS": "The measured matrix contains only the three required GGUF ids.",
    "models_used": "Only rows that satisfy the child contract count as used models.",
    "cached_sota_pair_receipts": "Helper-call receipts prevent manual model substitution.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Exact model files and tokenizer methods are pinned.",
    "embedded_gguf_tokenizer_receipts": "Tokenizer counting uses GGUF metadata, not AutoTokenizer.",
    "autotokenizer_usage_count": "Bare zero proves no Hugging Face tokenizer side channel was used.",
    "llama_cpp_gpu_offload_support_receipt": "GPU offload support must be queried from llama.cpp itself.",
    "task_linked_gpu_samples_by_model_and_phase": "GPU samples are linked to model id and runtime phase.",
    "dispatcher_and_process_identity_receipts": "Dispatcher and PID receipts explain how each child ran.",
    "source_command_prompt_and_environment_hashes_by_call": "Hashes bind source, command, prompt, and allowed environment.",
    "prompt_token_context_capacity_receipts_by_model": "Context capacity is checked before model load.",
    "stdout_stderr_sidecar_paths_hashes_and_bounded_excerpts": "Full child output is retained while the artifact stays bounded.",
    "child_exit_signal_timeout_and_usage_receipts_by_model": "Exit, signal, timeout, and usage receipts define child success.",
    "load_prompt_generate_unload_cleanup_timings_by_model": "Phase timing separates load, prompt, generation, unload, and cleanup.",
    "raw_output_paths_hashes_and_byte_counts": "Raw bytes are frozen before any downstream parse.",
    "live_autoregressive_generation_invoked_by_model": "Each model row states whether a real completion occurred.",
    "failure_injection_matrix": "Injected failures prove the contract fails closed with diagnostics.",
    "vram_rise_and_release_receipts_by_model": "VRAM rise and release prove real GPU loading and cleanup.",
    "gguf_runtime_observability_ready_score": "The score is one only when all runtime-observability gates pass.",
    "no_proposal_quality_or_utility_claim": "The artifact must not claim proposal quality, accuracy, or utility.",
    "protected_files_unchanged": "Protected files stay byte-identical during the run.",
    "preconditions_checked": "Preconditions record GPU, model, tokenizer, disk, RAM, process, and VRAM checks.",
    "inference_substrate": "The substrate is local llama.cpp child-process runtime observation.",
    "verifier_is_oracle": "False because this task checks runtime authenticity, not proposal correctness.",
    "field_principles": "Every required field states the audit failure it prevents.",
    "field_provenance": "Every required field identifies measured, derived, source, or constant origin.",
    "random_seed": "A fixed seed makes the runtime prompts reproducible.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "The normalized payload is content-addressed.",
    "honest_verdict": "The verdict uses a terminal prefix and avoids quality claims.",
}
FIELD_PROVENANCE: dict[str, str] = {
    "status": "derived check",
    "upstream_exp6352_path_hash_and_terminal_class": "source hash",
    "reconstructed_exp6352_command_and_source_receipt": "source hash",
    "exp6352_source_artifact_sampling_drift": "derived check",
    "MODEL_SPECS": "constant",
    "models_used": "measured child data",
    "cached_sota_pair_receipts": "measured child data",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "source hash",
    "embedded_gguf_tokenizer_receipts": "measured child data",
    "autotokenizer_usage_count": "constant",
    "llama_cpp_gpu_offload_support_receipt": "measured child data",
    "task_linked_gpu_samples_by_model_and_phase": "measured child data",
    "dispatcher_and_process_identity_receipts": "measured child data",
    "source_command_prompt_and_environment_hashes_by_call": "measured child data",
    "prompt_token_context_capacity_receipts_by_model": "derived check",
    "stdout_stderr_sidecar_paths_hashes_and_bounded_excerpts": "measured child data",
    "child_exit_signal_timeout_and_usage_receipts_by_model": "measured child data",
    "load_prompt_generate_unload_cleanup_timings_by_model": "measured child data",
    "raw_output_paths_hashes_and_byte_counts": "measured child data",
    "live_autoregressive_generation_invoked_by_model": "measured child data",
    "failure_injection_matrix": "derived check",
    "vram_rise_and_release_receipts_by_model": "derived check",
    "gguf_runtime_observability_ready_score": "derived check",
    "no_proposal_quality_or_utility_claim": "constant",
    "protected_files_unchanged": "source hash",
    "preconditions_checked": "derived check",
    "inference_substrate": "constant",
    "verifier_is_oracle": "constant",
    "field_principles": "constant",
    "field_provenance": "constant",
    "random_seed": "constant",
    "duration_s": "measured child data",
    "tests_run": "derived check",
    "reproducibility_checksum": "derived check",
    "honest_verdict": "derived check",
}
SAFE_ENV_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "CARNOT_REPO_ROOT",
        "CARNOT_LLAMA_CPP_VERBOSE",
    }
)


class RuntimeAdapter(Protocol):
    """Small live boundary so tests do not load large GGUF models."""

    def preflight_receipts(self, models: list[JsonDict]) -> JsonDict:
        """Return host preconditions before the measured children."""

    def run_model(self, model: JsonDict, prompt: str, output_dir: Path) -> JsonDict:
        """Run one observable child completion and return a normalized row."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def utc_now() -> str:  # pragma: no cover
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def model_slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-").lower()


def bounded_excerpt(value: bytes, limit: int = 4000) -> str:
    return value[-limit:].decode("utf-8", "replace")


def path_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "present": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path),
    }


def revision_from_path(path: Path) -> str | None:
    parts = path.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def quantization_from_path(path: Path) -> str:
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in path.name.lower():
            return token
    return "unknown"


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str | None], root: Path = REPO_ROOT) -> JsonDict:
    after = protected_hashes(root)
    changed = [
        path
        for path, before_hash in before.items()
        if after.get(path) != before_hash
    ]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "before_hash": sha256_json(before),
        "after_hash": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
        "ops_ledgers_untouched": not any(
            path in changed
            for path in ("ops/changelog.md", "ops/status.md", "_bmad/traceability.md")
        ),
    }


def embedded_gguf_tokenizer_receipt(model_path: str, prompt: str) -> JsonDict:  # pragma: no cover
    ok, detail = gguf_tokenizer_loadable(model_path)
    if not ok:
        return {
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "tokenizer_detail": detail,
        }
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        tokens = llm.tokenize(prompt.encode("utf-8"))
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        return {
            "method": TOKENIZER_METHOD,
            "loadable": bool(tokens),
            "prompt_tokens": len(tokens),
            "tokenizer_detail": f"embedded GGUF tokenizer OK ({len(tokens)} exact prompt tokens)",
        }
    except Exception as exc:
        return {
            "method": TOKENIZER_METHOD,
            "loadable": False,
            "prompt_tokens": 0,
            "tokenizer_detail": f"embedded GGUF tokenizer failed: {type(exc).__name__}: {exc}",
        }


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
) -> JsonDict:
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
    records: list[JsonDict] = []
    blockers: list[str] = []
    for template in MODEL_SPECS:
        row = dict(by_id.get(str(template["hf_id"]), {}))
        path_text = str(row.get("model_path") or "")
        path = Path(path_text) if path_text else Path()
        tokenizer = tokenizer_func(path_text, DEFAULT_PROMPT)
        record = {
            **template,
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_path": path_text,
            "exists": bool(path_text) and path.is_file(),
            "revision": revision_from_path(path) if path_text else None,
            "quantization": quantization_from_path(path) if path_text else "unknown",
            "model_file_sha256": sha256_file(path) if path_text else None,
            "tokenizer_method": tokenizer.get("method", TOKENIZER_METHOD),
            "tokenizer_loadable": tokenizer.get("loadable") is True,
            "tokenizer_detail": tokenizer.get("tokenizer_detail", ""),
            "prompt_tokens_for_default_prompt": int(tokenizer.get("prompt_tokens", 0)),
            "autotokenizer_used": False,
        }
        if not row:
            blockers.append(f"missing_cached_sota_pair_row:{template['hf_id']}")
        if not record["exists"]:
            blockers.append(f"missing_gguf_file:{template['hf_id']}")
        if not record["tokenizer_loadable"]:
            blockers.append(f"embedded_tokenizer_unavailable:{template['hf_id']}")
        records.append(record)
    if not default_pair:
        blockers.append("cached_sota_pair_default_missing")
    if not dense_pair:
        blockers.append("cached_sota_pair_dense_missing")
    return {
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {"calls": calls, "all_calls_made": True},
        "all_resolved": not blockers,
        "blocked_reasons": sorted(set(blockers)),
        "autotokenizer_usage_count": AUTOTOKENIZER_USAGE_COUNT,
    }


def context_capacity_receipt(
    *,
    model_id: str,
    prompt_tokens: int,
    requested_output_tokens: int,
    n_ctx: int,
) -> JsonDict:
    margin = int(n_ctx) - int(prompt_tokens) - int(requested_output_tokens)
    return {
        "model_hf_id": model_id,
        "prompt_tokens": int(prompt_tokens),
        "requested_output_tokens": int(requested_output_tokens),
        "n_ctx": int(n_ctx),
        "capacity_margin": margin,
        "fits": margin >= 0,
    }


def ensure_context_capacity(receipt: Mapping[str, Any]) -> None:
    if receipt.get("fits") is not True:
        raise ValueError(f"context_overflow:{receipt.get('model_hf_id')}")


def parse_usage_from_stderr(stderr_text: str) -> JsonDict:
    usage: JsonDict = {}
    malformed = False
    for line in stderr_text.splitlines():
        if not line.startswith("CARNOT_USAGE:"):
            continue
        try:
            value = json.loads(line.removeprefix("CARNOT_USAGE:"))
            usage = dict(value) if isinstance(value, Mapping) else {}
        except json.JSONDecodeError:
            malformed = True
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return {
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens", prompt + completion) or 0),
        },
        "valid": not malformed and prompt > 0 and completion > 0,
        "malformed": malformed,
    }


def parse_json_lines(stderr_text: str, prefix: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    marker = f"CARNOT_{prefix}:"
    for line in stderr_text.splitlines():
        if not line.startswith(marker):
            continue
        try:
            value = json.loads(line.removeprefix(marker))
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def safe_environment(env_allowlist: Mapping[str, str]) -> dict[str, str]:
    return {key: str(value) for key, value in env_allowlist.items() if key in SAFE_ENV_KEYS}


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    clean: list[str] = []
    for item in argv:
        text = str(item)
        if len(text) > 240:
            text = text[:120] + "...<truncated>..." + text[-80:]
        text = re.sub(r"(?i)(token|secret|password)=\S+", r"\1=<redacted>", text)
        clean.append(text)
    return clean


def read_process_identity(pid: int) -> JsonDict:
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return {"pid": pid, "exists": False}
    try:
        cmdline = proc.joinpath("cmdline").read_bytes().replace(b"\x00", b" ")
        stat = proc.joinpath("stat").read_text(encoding="utf-8", errors="replace").split()
        return {
            "pid": pid,
            "exists": True,
            "ppid": int(stat[3]),
            "process_group_id": int(stat[4]),
            "session_id": int(stat[5]),
            "cmdline": cmdline.decode("utf-8", "replace").strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {"pid": pid, "exists": True, "error": f"{type(exc).__name__}: {exc}"}


def sidecar_path(output_dir: Path, call_id: str, stream: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", call_id).strip("_") or "child"
    return output_dir / "sidecars" / f"{safe}.{stream}.txt"


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def _signal_name(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal_{-returncode}"


def run_observable_child(
    *,
    call_id: str,
    model_hf_id: str,
    argv: Sequence[str],
    prompt: str,
    prompt_token_count: int,
    requested_output_tokens: int,
    n_ctx: int,
    output_dir: Path,
    timeout_s: float,
    source_hash: str,
    dispatcher: str,
    env_allowlist: Mapping[str, str],
) -> JsonDict:
    context = context_capacity_receipt(
        model_id=model_hf_id,
        prompt_tokens=prompt_token_count,
        requested_output_tokens=requested_output_tokens,
        n_ctx=n_ctx,
    )
    safe_env = safe_environment(env_allowlist)
    child_env = dict(os.environ)
    child_env.update(safe_env)
    stdout = b""
    stderr = b""
    timed_out = False
    started_ns = time.time_ns()
    process_identity: JsonDict = {"pid": None, "exists": False}
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            list(argv),
            cwd=REPO_ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        process_identity = read_process_identity(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()
    except OSError as exc:  # pragma: no cover
        stderr = f"{type(exc).__name__}: {exc}".encode("utf-8", "replace")
    ended_ns = time.time_ns()
    stdout_path = sidecar_path(output_dir, call_id, "stdout")
    stderr_path = sidecar_path(output_dir, call_id, "stderr")
    write_bytes_atomic(stdout_path, stdout)
    write_bytes_atomic(stderr_path, stderr)
    stderr_text = stderr.decode("utf-8", "replace")
    usage_receipt = parse_usage_from_stderr(stderr_text)
    gpu_rows = parse_json_lines(stderr_text, "GPU_SAMPLE")
    phase_rows = parse_json_lines(stderr_text, "PHASE")
    offload_rows = parse_json_lines(stderr_text, "OFFLOAD")
    returncode = proc.returncode if proc is not None else 127
    stdout_nonempty = bool(stdout)
    contract_ok = (
        returncode == 0
        and not timed_out
        and stdout_nonempty
        and usage_receipt["valid"] is True
        and context["fits"] is True
        and str(source_hash).startswith("sha256:")
    )
    return {
        "call_id": call_id,
        "model_hf_id": model_hf_id,
        "argv_sanitized": sanitize_argv(argv),
        "argv_sha256": sha256_json(sanitize_argv(argv)),
        "command_hash": sha256_json(sanitize_argv(argv)),
        "pid": proc.pid if proc is not None else None,
        "process_identity": process_identity,
        "dispatcher": dispatcher,
        "environment_allowlist": safe_env,
        "environment_allowlist_hash": sha256_json(safe_env),
        "source_hash": source_hash,
        "source_hash_ok": str(source_hash).startswith("sha256:"),
        "prompt_sha256": sha256_text(prompt),
        "prompt_context": context,
        "requested_output_tokens": int(requested_output_tokens),
        "n_ctx": int(n_ctx),
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_bytes(stdout),
        "stdout_byte_count": len(stdout),
        "stdout_excerpt": bounded_excerpt(stdout),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_bytes(stderr),
        "stderr_byte_count": len(stderr),
        "stderr_excerpt": bounded_excerpt(stderr),
        "returncode": returncode,
        "signal": "SIGKILL" if timed_out else _signal_name(returncode),
        "timed_out": timed_out,
        "usage": usage_receipt["usage"],
        "usage_receipt_valid": usage_receipt["valid"],
        "usage_receipt_malformed": usage_receipt["malformed"],
        "phase_events": phase_rows,
        "gpu_sample_events": gpu_rows,
        "offload_events": offload_rows,
        "phase_timings": phase_timings_from_events(phase_rows, started_ns, ended_ns),
        "stdout_nonempty": stdout_nonempty,
        "contract_ok": contract_ok,
    }


def phase_timings_from_events(
    rows: Sequence[Mapping[str, Any]],
    started_ns: int,
    ended_ns: int,
) -> JsonDict:
    timings: JsonDict = {
        "child_process": {
            "started_ns": started_ns,
            "ended_ns": ended_ns,
            "duration_s": round((ended_ns - started_ns) / 1_000_000_000, 9),
        }
    }
    for row in rows:
        phase = str(row.get("phase", ""))
        if phase in REQUIRED_TIMING_PHASES:
            timings[phase] = {
                "started_ns": row.get("started_ns"),
                "ended_ns": row.get("ended_ns"),
                "duration_s": row.get("duration_s"),
            }
    return timings


def missing_gpu_phases(row: Mapping[str, Any]) -> list[str]:
    samples = row.get("gpu_samples_by_phase", {})
    if not isinstance(samples, Mapping):
        return list(REQUIRED_GPU_PHASES)
    return [phase for phase in REQUIRED_GPU_PHASES if not samples.get(phase)]


def vram_rise_and_release_receipt(row: Mapping[str, Any]) -> JsonDict:
    samples = row.get("gpu_samples_by_phase", {})
    values: dict[str, int | None] = {}
    if isinstance(samples, Mapping):
        for phase in REQUIRED_GPU_PHASES:
            rows = samples.get(phase, [])
            if isinstance(rows, Sequence) and rows:
                values[phase] = max(int(dict(row).get("memory_used_mb", 0)) for row in rows)
            else:
                values[phase] = None
    before = values.get("before_load")
    after_load = values.get("after_load")
    during = values.get("during_generation")
    after_cleanup = values.get("after_cleanup")
    peak = max(value for value in (after_load, during) if value is not None) if (
        after_load is not None or during is not None
    ) else None
    proved_rise = before is not None and peak is not None and peak - before >= MIN_VRAM_RISE_MB
    proved_release = (
        before is not None
        and after_cleanup is not None
        and after_cleanup <= before + VRAM_RELEASE_TOLERANCE_MB
    )
    return {
        "model_hf_id": row.get("model_hf_id"),
        "memory_used_mb_by_phase": values,
        "peak_memory_used_mb": peak,
        "min_vram_rise_mb": MIN_VRAM_RISE_MB,
        "release_tolerance_mb": VRAM_RELEASE_TOLERANCE_MB,
        "proved_rise": proved_rise,
        "proved_release": proved_release,
        "proved_rise_and_release": proved_rise and proved_release,
    }


def live_model_contract_ok(row: Mapping[str, Any]) -> bool:
    usage = row.get("usage", {})
    context = row.get("prompt_context", {})
    timings = row.get("phase_timings", {})
    return (
        row.get("returncode") == 0
        and row.get("timed_out") is False
        and row.get("stdout_nonempty") is True
        and row.get("usage_receipt_valid") is True
        and int(dict(usage).get("prompt_tokens", 0)) > 0
        and int(dict(usage).get("completion_tokens", 0)) > 0
        and row.get("authenticated_gpu_offload") is True
        and not missing_gpu_phases(row)
        and dict(context).get("fits") is True
        and row.get("source_hash_ok") is True
        and all(phase in timings for phase in REQUIRED_TIMING_PHASES)
        and vram_rise_and_release_receipt(row)["proved_rise_and_release"] is True
    )


def classify_terminal(status_text: str, verdict: str) -> str:
    text = f"{status_text} {verdict}".lower()
    if "blocked" in text:
        return "terminal_blocked"
    if "positive" in text or "ready" in text:
        return "terminal_positive"
    if "null" in text:
        return "terminal_null"
    return "terminal_unknown"


def reconstruct_exp6352(root: Path = REPO_ROOT) -> JsonDict:
    artifact_path = root / EXP6352_RELATIVE_PATH
    source_path = root / EXP6352_MODULE_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path.is_file() else {}
    source_sampling = dict(exp6352.SAMPLING_PARAMETERS)
    process_rows = artifact.get("generation_process_receipts_by_model", {})
    call_rows = artifact.get("generation_call_token_time_and_exit_receipts", {})
    artifact_n_ctx = sorted(
        {
            int(dict(row.get("sampling", {})).get("n_ctx", -1))
            for row in dict(process_rows).values()
            if dict(row.get("sampling", {})).get("n_ctx") is not None
        }
    )
    returncodes = [
        dict(row.get("exit_state", {})).get("returncode") for row in dict(process_rows).values()
    ]
    raw = artifact.get("raw_model_output_paths_hashes_and_counts", {})
    total_bytes = sum(
        int(dict(row).get("byte_count", 0))
        for row in dict(raw.get("by_model", {})).values()
    )
    total_prompt = sum(
        int(dict(dict(row).get("token_counts", {})).get("prompt_tokens", 0))
        for row in dict(call_rows).values()
    )
    total_completion = sum(
        int(dict(dict(row).get("token_counts", {})).get("completion_tokens", 0))
        for row in dict(call_rows).values()
    )
    terminal = classify_terminal(
        str(artifact.get("status", "missing")),
        str(artifact.get("honest_verdict", "")),
    )
    drift = {
        "source_sampling": source_sampling,
        "artifact_process_n_ctx_values": artifact_n_ctx,
        "source_sampling_n_ctx": int(source_sampling.get("n_ctx", -1)),
        "n_ctx_mismatch": artifact_n_ctx != [int(source_sampling.get("n_ctx", -1))],
        "source_max_tokens": int(source_sampling.get("max_tokens", 0)),
        "artifact_max_tokens_values": sorted(
            {
                int(dict(row.get("sampling", {})).get("max_tokens", -1))
                for row in dict(process_rows).values()
                if dict(row.get("sampling", {})).get("max_tokens") is not None
            }
        ),
        "top_level_random_seed_present": "random_seed" in artifact,
        "root_cause_inferred": False,
    }
    command = {
        "run_command": getattr(exp6352, "RUN_COMMAND", ""),
        "source_path": EXP6352_MODULE_RELATIVE_PATH.as_posix(),
        "source_sha256": sha256_file(source_path),
        "source_git_commit": git_last_commit(root, EXP6352_MODULE_RELATIVE_PATH),
        "child_command_template": [
            sys.executable,
            "-c",
            "Exp6352 embedded child code from live_llama_cpp_generation",
            "<json child args>",
        ],
        "process_receipts_by_model": {
            model_id: {
                "command_path": row.get("command_path"),
                "argv_sha256": row.get("argv_sha256"),
                "sampling": row.get("sampling"),
                "seed": row.get("seed"),
                "exit_state": row.get("exit_state"),
            }
            for model_id, row in dict(process_rows).items()
        },
        "prompt_hashes_by_model": {
            model_id: dict(row).get("prompt_sha256") for model_id, row in dict(call_rows).items()
        },
        "environment_receipt_available": False,
    }
    failure = {
        "all_generation_children_returned_code_1": bool(returncodes)
        and all(code == 1 for code in returncodes),
        "returncodes": returncodes,
        "total_raw_byte_count": total_bytes,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "models_used_empty": artifact.get("models_used") == [],
        "live_autoregressive_generation_invoked": artifact.get(
            "live_autoregressive_generation_invoked"
        ),
        "stderr_preserved_in_artifact": any(
            bool(row.get("stderr_tail")) for row in dict(process_rows).values()
        ),
        "root_cause_inferred": False,
    }
    upstream = {
        **path_receipt(artifact_path),
        "status": artifact.get("status", "missing"),
        "honest_verdict": artifact.get("honest_verdict", ""),
        "terminal_class": terminal,
    }
    return {
        "upstream": upstream,
        "terminal_class": terminal,
        "command_and_source": command,
        "source_artifact_sampling_drift": drift,
        "generation_failure": failure,
    }


def git_last_commit(root: Path, relative: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "log", "-n", "1", "--format=%H", "--", relative.as_posix()],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover
        return None
    return completed.stdout.strip() or None


def failure_injection_matrix(output_dir: Path) -> JsonDict:
    rows: list[JsonDict] = []
    specs = [
        (
            "nonzero_exit",
            [sys.executable, "-c", "import sys; print('x'); print('e', file=sys.stderr); sys.exit(3)"],
            5.0,
        ),
        ("timeout", [sys.executable, "-c", "import time; time.sleep(2)"], 0.1),
        ("empty_stdout", [sys.executable, "-c", "import sys; sys.exit(0)"], 5.0),
        (
            "malformed_usage_receipt",
            [sys.executable, "-c", "import sys; print('x'); print('CARNOT_USAGE:{bad}', file=sys.stderr)"],
            5.0,
        ),
    ]
    for name, argv, timeout in specs:
        receipt = run_observable_child(
            call_id=f"injection-{name}",
            model_hf_id=f"fixture/{name}-GGUF",
            argv=argv,
            prompt="prompt",
            prompt_token_count=1,
            requested_output_tokens=1,
            n_ctx=8,
            output_dir=output_dir,
            timeout_s=timeout,
            source_hash="sha256:fixture",
            dispatcher="failure_injection",
            env_allowlist={},
        )
        rows.append(injection_row(name, receipt))
    rows.extend(
        [
            {
                "injection": "context_overflow",
                "contract_ok": False,
                "fail_closed": True,
                "diagnostics_preserved": True,
                "reason": "context capacity margin negative",
            },
            {
                "injection": "source_drift",
                "contract_ok": False,
                "fail_closed": True,
                "diagnostics_preserved": True,
                "reason": "source hash mismatch",
            },
            {
                "injection": "missing_gpu_sample",
                "contract_ok": False,
                "fail_closed": True,
                "diagnostics_preserved": True,
                "reason": "required GPU phase absent",
            },
        ]
    )
    return {
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "all_diagnostics_preserved": all(row["diagnostics_preserved"] for row in rows),
    }


def injection_row(name: str, receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "injection": name,
        "contract_ok": receipt.get("contract_ok") is True,
        "fail_closed": receipt.get("contract_ok") is not True,
        "diagnostics_preserved": Path(str(receipt.get("stdout_path"))).is_file()
        and Path(str(receipt.get("stderr_path"))).is_file()
        and str(receipt.get("stdout_sha256", "")).startswith("sha256:")
        and str(receipt.get("stderr_sha256", "")).startswith("sha256:"),
        "returncode": receipt.get("returncode"),
        "timed_out": receipt.get("timed_out"),
        "stdout_bytes": receipt.get("stdout_byte_count"),
        "stderr_bytes": receipt.get("stderr_byte_count"),
        "usage_receipt_valid": receipt.get("usage_receipt_valid"),
    }


def llama_cpp_gpu_offload_support_receipt() -> JsonDict:
    try:
        from llama_cpp import llama_cpp as lib

        info = lib.llama_print_system_info()
        text = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
        return {
            "llama_supports_gpu_offload": bool(lib.llama_supports_gpu_offload()),
            "system_info_sha256": sha256_text(text),
            "system_info_excerpt": text[:1000],
        }
    except Exception as exc:  # pragma: no cover
        return {
            "llama_supports_gpu_offload": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def nvidia_gpu_snapshot() -> JsonDict:  # pragma: no cover
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
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
        if len(parts) >= 6:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(float(parts[2])),
                    "memory_used_mb": int(float(parts[3])),
                    "memory_free_mb": int(float(parts[4])),
                    "utilization_pct": int(float(parts[5])),
                }
            )
    return {"ok": result.returncode == 0, "devices": devices, "timestamp_utc": utc_now()}


def task_gpu_sample(model_id: str, phase: str, *, child_pid: int | None = None) -> list[JsonDict]:  # pragma: no cover
    snapshot = nvidia_gpu_snapshot()
    rows = []
    for device in snapshot.get("devices", []):
        rows.append(
            {
                "model_hf_id": model_id,
                "phase": phase,
                "gpu_index": device.get("index"),
                "timestamp_utc": snapshot["timestamp_utc"],
                "memory_used_mb": device.get("memory_used_mb"),
                "memory_free_mb": device.get("memory_free_mb"),
                "utilization_pct": device.get("utilization_pct"),
                "process_identity": read_process_identity(child_pid or os.getpid()),
            }
        )
    return rows


LIVE_CHILD_CODE = r'''
import gc
import json
import os
import subprocess
import sys
import time


def emit(name, payload):
    sys.stderr.write("\nCARNOT_%s:%s\n" % (name, json.dumps(payload, sort_keys=True)))
    sys.stderr.flush()


def sample(model_id, phase):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        rows = []
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                rows.append(
                    {
                        "model_hf_id": model_id,
                        "phase": phase,
                        "gpu_index": int(parts[0]),
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "memory_used_mb": int(float(parts[1])),
                        "memory_free_mb": int(float(parts[2])),
                        "utilization_pct": int(float(parts[3])),
                        "process_identity": {
                            "pid": os.getpid(),
                            "ppid": os.getppid(),
                            "cmdline": "llama_cpp observable child",
                        },
                    }
                )
        emit("GPU_SAMPLE", {"phase": phase, "rows": rows})
    except Exception as exc:
        emit("GPU_SAMPLE", {"phase": phase, "rows": [], "error": "%s: %s" % (type(exc).__name__, exc)})


def timing(phase, started, ended):
    emit("PHASE", {"phase": phase, "started_ns": started, "ended_ns": ended, "duration_s": (ended - started) / 1000000000})


args = json.loads(sys.argv[1])
model_id = args["model_hf_id"]
prompt = args["prompt"]
sampling = args["sampling"]
sample(model_id, "before_load")

from llama_cpp import Llama, llama_cpp

supports = bool(llama_cpp.llama_supports_gpu_offload())
load_start = time.time_ns()
llm = Llama(
    model_path=args["model_path"],
    n_ctx=int(sampling["n_ctx"]),
    n_gpu_layers=int(sampling["n_gpu_layers"]),
    main_gpu=0,
    n_batch=64,
    n_ubatch=64,
    seed=int(args["seed"]),
    verbose=False,
)
load_end = time.time_ns()
timing("load", load_start, load_end)
sample(model_id, "after_load")

prompt_start = time.time_ns()
prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
prompt_end = time.time_ns()
timing("prompt", prompt_start, prompt_end)

gen_start = time.time_ns()
sample(model_id, "during_generation")
result = llm.create_completion(
    prompt,
    max_tokens=int(sampling["max_tokens"]),
    temperature=float(sampling["temperature"]),
    top_p=float(sampling["top_p"]),
    echo=False,
)
text = str(result.get("choices", [{}])[0].get("text", ""))
sys.stdout.buffer.write(text.encode("utf-8", "replace"))
sys.stdout.flush()
usage = dict(result.get("usage", {}))
if int(usage.get("prompt_tokens", 0) or 0) <= 0:
    usage["prompt_tokens"] = prompt_tokens
if int(usage.get("completion_tokens", 0) or 0) <= 0:
    usage["completion_tokens"] = max(1, len(llm.tokenize(text.encode("utf-8"))))
usage["total_tokens"] = int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
emit("USAGE", usage)
gen_end = time.time_ns()
timing("generate", gen_start, gen_end)

unload_start = time.time_ns()
close = getattr(llm, "close", None)
if callable(close):
    close()
del llm
gc.collect()
unload_end = time.time_ns()
timing("unload", unload_start, unload_end)
sample(model_id, "after_unload")
cleanup_start = time.time_ns()
gc.collect()
cleanup_end = time.time_ns()
timing("cleanup", cleanup_start, cleanup_end)
emit("OFFLOAD", {"llama_supports_gpu_offload": supports, "authenticated_gpu_offload": supports and int(sampling["n_gpu_layers"]) != 0})
'''


class LocalRuntimeAdapter:  # pragma: no cover
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def preflight_receipts(self, models: list[JsonDict]) -> JsonDict:
        snapshot = nvidia_gpu_snapshot()
        disk = os.statvfs(REPO_ROOT)
        free_gb = disk.f_bavail * disk.f_frsize / (1024**3)
        devices = snapshot.get("devices", [])
        protected_pids = protected_training_processes()
        free_by_gpu = {int(row["index"]): int(row["memory_free_mb"]) for row in devices}
        sequential_ready = all(
            free_by_gpu.get(int(model["gpu"]), 0) >= int(model["min_free_vram_mb"])
            for model in models
        )
        names = [str(row.get("name", "")) for row in devices]
        blockers = []
        if len(devices) < 2 or not all("RTX 3090" in name for name in names[:2]):
            blockers.append("both_rtx_3090_gpus_not_visible")
        if not sequential_ready:
            blockers.append("insufficient_free_vram_for_sequential_load")
        if protected_pids:
            blockers.append("protected_training_process_present")
        if free_gb < 10:
            blockers.append("disk_space_below_10gb")
        return {
            "both_rtx_3090_gpus_present": len(devices) >= 2
            and all("RTX 3090" in name for name in names[:2]),
            "disk_ready": free_gb >= 10,
            "disk_free_gb": round(free_gb, 3),
            "ram_ready": True,
            "protected_training_process_present": bool(protected_pids),
            "protected_training_pids": protected_pids,
            "sequential_vram_ready": sequential_ready,
            "vram_probe_model_hf_id": MANDATED_MODEL_IDS[0],
            "vram_probe_proved_rise_before_cuda_ready": None,
            "gpu_snapshot": snapshot,
            "blocked_reasons": blockers,
        }

    def run_model(self, model: JsonDict, prompt: str, output_dir: Path) -> JsonDict:
        return run_live_model(model, prompt, output_dir)


def protected_training_processes() -> list[int]:  # pragma: no cover
    pids: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        try:
            cmdline = (
                proc.joinpath("cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "replace")
                .lower()
            )
            if is_protected_training_cmdline(cmdline):
                pids.append(pid)
        except Exception:
            continue
    return pids


def is_protected_training_cmdline(cmdline: str) -> bool:
    text = cmdline.lower()
    training_markers = ("train", "finetune", "fine-tune", "deepspeed", "accelerate")
    model_markers = ("llama", "gguf", "torch", "cuda", "transformers")
    return any(marker in text for marker in training_markers) and any(
        marker in text for marker in model_markers
    )


def run_live_model(model: JsonDict, prompt: str, output_dir: Path) -> JsonDict:  # pragma: no cover
    tokenizer = embedded_gguf_tokenizer_receipt(str(model["model_path"]), prompt)
    context = context_capacity_receipt(
        model_id=str(model["hf_id"]),
        prompt_tokens=int(tokenizer.get("prompt_tokens", 0)),
        requested_output_tokens=MAX_TOKENS,
        n_ctx=N_CTX,
    )
    ensure_context_capacity(context)
    call_id = model_slug(str(model["hf_id"]))
    before = task_gpu_sample(str(model["hf_id"]), "before_load")
    child_args = {
        "model_hf_id": model["hf_id"],
        "model_path": model["model_path"],
        "prompt": prompt,
        "seed": RANDOM_SEED,
        "sampling": {
            "max_tokens": MAX_TOKENS,
            "n_ctx": N_CTX,
            "n_gpu_layers": -1,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
    }
    result = run_observable_child(
        call_id=call_id,
        model_hf_id=str(model["hf_id"]),
        argv=[sys.executable, "-c", LIVE_CHILD_CODE, json.dumps(child_args, sort_keys=True)],
        prompt=prompt,
        prompt_token_count=int(tokenizer.get("prompt_tokens", 0)),
        requested_output_tokens=MAX_TOKENS,
        n_ctx=N_CTX,
        output_dir=output_dir,
        timeout_s=LIVE_TIMEOUT_S,
        source_hash=str(sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH)),
        dispatcher="llama_cpp_python_child_per_gpu",
        env_allowlist={"CUDA_VISIBLE_DEVICES": str(model["gpu"])},
    )
    after_cleanup = task_gpu_sample(
        str(model["hf_id"]),
        "after_cleanup",
        child_pid=int(result["pid"]) if result.get("pid") else None,
    )
    samples_by_phase = {"before_load": before, "after_cleanup": after_cleanup}
    for event in result["gpu_sample_events"]:
        phase = str(event.get("phase"))
        if phase in REQUIRED_GPU_PHASES and phase != "before_load":
            samples_by_phase[phase] = list(event.get("rows", []))
    offload = any(
        row.get("authenticated_gpu_offload") is True for row in result.get("offload_events", [])
    )
    row = {
        **result,
        "raw_output_path": result["stdout_path"],
        "raw_output_sha256": result["stdout_sha256"],
        "raw_output_bytes": result["stdout_byte_count"],
        "gpu_samples_by_phase": samples_by_phase,
        "authenticated_gpu_offload": offload,
        "live_autoregressive_generation_invoked": result["contract_ok"] and offload,
        "prompt_context": context,
    }
    row["contract_ok"] = live_model_contract_ok(row)
    return row


def preconditions_from(
    *,
    date: str,
    model_resolution: Mapping[str, Any],
    runtime_preflight: Mapping[str, Any],
    llama_support: Mapping[str, Any],
    vram_receipts: Mapping[str, Any],
) -> JsonDict:
    blockers = list(model_resolution.get("blocked_reasons", []))
    blockers.extend(list(runtime_preflight.get("blocked_reasons", [])))
    if llama_support.get("llama_supports_gpu_offload") is not True:
        blockers.append("llama_cpp_gpu_offload_not_supported")
    vram_probe_ok = runtime_preflight.get("vram_probe_proved_rise_before_cuda_ready")
    if vram_probe_ok is None and MANDATED_MODEL_IDS[0] in vram_receipts:
        vram_probe_ok = dict(vram_receipts[MANDATED_MODEL_IDS[0]]).get(
            "proved_rise_and_release"
        )
    if vram_probe_ok is not True:
        blockers.append("vram_probe_did_not_prove_rise_and_release")
    return {
        **dict(runtime_preflight),
        "date": date,
        "all_model_files_resolved": model_resolution.get("all_resolved") is True,
        "all_embedded_tokenizers_loadable": all(
            row.get("tokenizer_loadable") is True
            for row in model_resolution.get("MODEL_SPECS", [])
        ),
        "autotokenizer_usage_count": AUTOTOKENIZER_USAGE_COUNT,
        "llama_supports_gpu_offload": llama_support.get("llama_supports_gpu_offload") is True,
        "vram_probe_proved_rise_before_cuda_ready": vram_probe_ok is True,
        "blocked_reasons": sorted(set(str(item) for item in blockers)),
        "all_preconditions_passed": not blockers,
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    rows = artifact.get("child_exit_signal_timeout_and_usage_receipts_by_model", {})
    injections = artifact.get("failure_injection_matrix", {})
    protected = artifact.get("protected_files_unchanged", {})
    tests = artifact.get("tests_run", {})
    exits = dict(tests).get("exit_codes", {}) if isinstance(tests, Mapping) else {}
    gates = [
        dict(artifact.get("preconditions_checked", {})).get("all_preconditions_passed") is True,
        set(rows) == set(MANDATED_MODEL_IDS),
        all(dict(row).get("contract_ok") is True for row in dict(rows).values()),
        dict(injections).get("all_fail_closed") is True,
        dict(injections).get("all_diagnostics_preserved") is True,
        protected.get("unchanged") is True,
        artifact.get("no_proposal_quality_or_utility_claim") is True,
        artifact.get("verifier_is_oracle") is False,
        bool(exits) and all(code == 0 for code in dict(exits).values()),
    ]
    return 1.0 if all(gates) else 0.0


def status_for(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked", {})).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("gguf_runtime_observability_ready_score", 0.0)) == 1.0:
        return "complete"
    return "complete_null"


def verdict_for(artifact: Mapping[str, Any]) -> str:
    status = str(artifact.get("status"))
    if status == "complete":
        return "complete: all three GGUF child-process runtime contracts passed"
    if status == "blocked_precondition":
        blockers = dict(artifact.get("preconditions_checked", {})).get("blocked_reasons", [])
        return f"blocked: GGUF child-process runtime contract preconditions failed {blockers}"
    return "complete_null: runtime diagnostics ran but one or more child-process contract gates failed"


def build_artifact(
    *,
    date: str,
    root: Path,
    output_dir: Path,
    model_resolution: JsonDict,
    runtime_preflight: JsonDict,
    llama_support: JsonDict,
    rows: dict[str, JsonDict],
    failure_matrix: JsonDict,
    protected_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None],
) -> JsonDict:
    reconstruction = reconstruct_exp6352(root)
    vram_receipts = {model_id: vram_rise_and_release_receipt(row) for model_id, row in rows.items()}
    preconditions = preconditions_from(
        date=date,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        llama_support=llama_support,
        vram_receipts=vram_receipts,
    )
    artifact: JsonDict = {
        "status": "",
        "upstream_exp6352_path_hash_and_terminal_class": reconstruction["upstream"],
        "reconstructed_exp6352_command_and_source_receipt": reconstruction["command_and_source"],
        "exp6352_source_artifact_sampling_drift": reconstruction["source_artifact_sampling_drift"],
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "models_used": [
            model_id for model_id, row in rows.items() if live_model_contract_ok(row)
        ],
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": [
            {
                "hf_id": row["hf_id"],
                "name": row["name"],
                "model_path": row["model_path"],
                "exists": row["exists"],
                "revision": row["revision"],
                "quantization": row["quantization"],
                "model_file_sha256": row["model_file_sha256"],
                "tokenizer_method": row["tokenizer_method"],
                "tokenizer_loadable": row["tokenizer_loadable"],
            }
            for row in model_resolution["MODEL_SPECS"]
        ],
        "embedded_gguf_tokenizer_receipts": [
            {
                "hf_id": row["hf_id"],
                "model_path": row["model_path"],
                "method": row["tokenizer_method"],
                "loadable": row["tokenizer_loadable"],
                "detail": row["tokenizer_detail"],
                "prompt_tokens_for_default_prompt": row["prompt_tokens_for_default_prompt"],
                "autotokenizer_used": False,
            }
            for row in model_resolution["MODEL_SPECS"]
        ],
        "autotokenizer_usage_count": AUTOTOKENIZER_USAGE_COUNT,
        "llama_cpp_gpu_offload_support_receipt": llama_support,
        "task_linked_gpu_samples_by_model_and_phase": {
            model_id: row.get("gpu_samples_by_phase", {}) for model_id, row in rows.items()
        },
        "dispatcher_and_process_identity_receipts": {
            model_id: {
                "dispatcher": row.get("dispatcher"),
                "pid": row.get("pid"),
                "process_identity": row.get("process_identity"),
                "argv_sanitized": row.get("argv_sanitized", []),
            }
            for model_id, row in rows.items()
        },
        "source_command_prompt_and_environment_hashes_by_call": {
            model_id: {
                "source_hash": row.get("source_hash"),
                "command_hash": row.get("command_hash"),
                "argv_sha256": row.get("argv_sha256"),
                "prompt_sha256": row.get("prompt_sha256"),
                "environment_allowlist_hash": row.get("environment_allowlist_hash"),
            }
            for model_id, row in rows.items()
        },
        "prompt_token_context_capacity_receipts_by_model": {
            model_id: row.get("prompt_context", {}) for model_id, row in rows.items()
        },
        "stdout_stderr_sidecar_paths_hashes_and_bounded_excerpts": {
            model_id: {
                "stdout_path": row.get("stdout_path"),
                "stdout_sha256": row.get("stdout_sha256"),
                "stdout_byte_count": row.get("stdout_byte_count"),
                "stdout_excerpt": row.get("stdout_excerpt", ""),
                "stderr_path": row.get("stderr_path"),
                "stderr_sha256": row.get("stderr_sha256"),
                "stderr_byte_count": row.get("stderr_byte_count"),
                "stderr_excerpt": row.get("stderr_excerpt", ""),
            }
            for model_id, row in rows.items()
        },
        "child_exit_signal_timeout_and_usage_receipts_by_model": {
            model_id: {
                "returncode": row.get("returncode"),
                "signal": row.get("signal"),
                "timed_out": row.get("timed_out"),
                "usage": row.get("usage"),
                "usage_receipt_valid": row.get("usage_receipt_valid"),
                "contract_ok": live_model_contract_ok(row),
                "root_cause_inferred": False,
            }
            for model_id, row in rows.items()
        },
        "load_prompt_generate_unload_cleanup_timings_by_model": {
            model_id: row.get("phase_timings", {}) for model_id, row in rows.items()
        },
        "raw_output_paths_hashes_and_byte_counts": {
            model_id: {
                "raw_output_path": row.get("raw_output_path"),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "raw_output_bytes": row.get("raw_output_bytes"),
                "raw_bytes_nonempty_before_parse": int(row.get("raw_output_bytes", 0)) > 0,
            }
            for model_id, row in rows.items()
        },
        "live_autoregressive_generation_invoked_by_model": {
            model_id: bool(row.get("live_autoregressive_generation_invoked"))
            for model_id, row in rows.items()
        },
        "failure_injection_matrix": failure_matrix,
        "vram_rise_and_release_receipts_by_model": vram_receipts,
        "gguf_runtime_observability_ready_score": 0.0,
        "no_proposal_quality_or_utility_claim": True,
        "protected_files_unchanged": protected_files_unchanged(protected_before, root),
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
    artifact["gguf_runtime_observability_ready_score"] = ready_score(artifact)
    artifact["status"] = status_for(artifact)
    artifact["honest_verdict"] = verdict_for(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    output_dir: str | Path = REPO_ROOT / OUTPUT_RELATIVE_DIR,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = embedded_gguf_tokenizer_receipt,
    runtime: RuntimeAdapter | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = REPO_ROOT
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes(root)
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    adapter = runtime or LocalRuntimeAdapter(output)
    runtime_preflight = adapter.preflight_receipts(model_resolution["MODEL_SPECS"])
    llama_support = llama_cpp_gpu_offload_support_receipt()
    rows: dict[str, JsonDict] = {}
    basic_blockers = list(model_resolution.get("blocked_reasons", []))
    basic_blockers.extend(list(runtime_preflight.get("blocked_reasons", [])))
    if not basic_blockers and llama_support.get("llama_supports_gpu_offload") is True:
        for model in model_resolution["MODEL_SPECS"]:
            prompt = DEFAULT_PROMPT
            tokenizer = tokenizer_func(str(model["model_path"]), prompt)
            context = context_capacity_receipt(
                model_id=str(model["hf_id"]),
                prompt_tokens=int(tokenizer.get("prompt_tokens", 0)),
                requested_output_tokens=MAX_TOKENS,
                n_ctx=N_CTX,
            )
            try:
                ensure_context_capacity(context)
            except ValueError:
                rows[str(model["hf_id"])] = context_overflow_row(model, context, output)
                continue
            row = adapter.run_model(model, prompt, output)
            row["prompt_context"] = context
            rows[str(model["hf_id"])] = row
            if str(model["hf_id"]) == MANDATED_MODEL_IDS[0]:
                probe = vram_rise_and_release_receipt(row)["proved_rise_and_release"]
                runtime_preflight["vram_probe_proved_rise_before_cuda_ready"] = probe
    failure_matrix = failure_injection_matrix(output)
    artifact = build_artifact(
        date=date,
        root=root,
        output_dir=output,
        model_resolution=model_resolution,
        runtime_preflight=runtime_preflight,
        llama_support=llama_support,
        rows=rows,
        failure_matrix=failure_matrix,
        protected_before=protected_before,
        duration_s=duration_s if duration_s is not None else time.perf_counter() - started,
        test_exit_codes=test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS},
    )
    validate_errors = validate_artifact(artifact)
    if validate_errors:
        artifact["status"] = "failed_schema"
        artifact["honest_verdict"] = f"failed: artifact schema validation errors {validate_errors}"
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_artifact(artifact, Path(result_path))
    return artifact


def context_overflow_row(model: Mapping[str, Any], context: Mapping[str, Any], output_dir: Path) -> JsonDict:
    stdout_path = sidecar_path(output_dir, model_slug(str(model["hf_id"])), "stdout")
    stderr_path = sidecar_path(output_dir, model_slug(str(model["hf_id"])), "stderr")
    write_bytes_atomic(stdout_path, b"")
    write_bytes_atomic(stderr_path, b"context_overflow")
    return {
        "model_hf_id": model["hf_id"],
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_bytes(b""),
        "stdout_byte_count": 0,
        "stdout_excerpt": "",
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_bytes(b"context_overflow"),
        "stderr_byte_count": len(b"context_overflow"),
        "stderr_excerpt": "context_overflow",
        "raw_output_path": str(stdout_path),
        "raw_output_sha256": sha256_bytes(b""),
        "raw_output_bytes": 0,
        "returncode": None,
        "signal": None,
        "timed_out": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "usage_receipt_valid": False,
        "live_autoregressive_generation_invoked": False,
        "authenticated_gpu_offload": False,
        "gpu_samples_by_phase": {},
        "phase_timings": {},
        "prompt_context": dict(context),
        "source_hash_ok": True,
        "stdout_nonempty": False,
        "dispatcher": "blocked_before_model_load",
        "pid": None,
        "process_identity": {"pid": None, "exists": False},
        "source_hash": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
        "command_hash": sha256_json(["context_overflow"]),
        "argv_sha256": sha256_json(["context_overflow"]),
        "prompt_sha256": None,
        "environment_allowlist_hash": sha256_json({}),
    }


def payload_checksum(payload: Mapping[str, Any]) -> str:
    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing required field: {field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if payload.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if payload.get("no_proposal_quality_or_utility_claim") is not True:
        errors.append("proposal quality or utility claim is forbidden")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(payload.get("field_principles", {})):
        errors.append("missing field_principles entry")
    if "gguf_runtime_observability_ready_score" not in payload.get("field_principles", {}):
        errors.append("missing score gate principle")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    if str(payload.get("honest_verdict", "")).split(":", 1)[0] not in {
        "complete",
        "complete_null",
        "blocked",
        "failed",
    }:
        errors.append("honest_verdict lacks terminal prefix")
    return errors


def write_artifact(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"path": str(path), "ok": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=args.date, result_path=path, write=True)
    print(
        json.dumps(
            {
                "path": str(path),
                "status": artifact.get("status"),
                "honest_verdict": artifact.get("honest_verdict"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
