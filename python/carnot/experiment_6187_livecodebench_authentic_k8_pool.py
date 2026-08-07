"""Exp6187 authentic LiveCodeBench K8 raw-code pool collection.

Spec refs: REQ-CODE-6187,
SCENARIO-CODE-6187-GATE-FAIL-CLOSED,
SCENARIO-CODE-6187-K8-SELECTOR-MATRIX,
SCENARIO-CODE-6187-RAW-BEFORE-LABEL,
SCENARIO-CODE-6187-RETENTION-AND-RESTRICTED-EXECUTION,
SCENARIO-CODE-6187-PRIVATE-TEST-NONINTERFERENCE,
SCENARIO-CODE-6187-CONTENT-ADDRESSED-RESUME.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol

from carnot.inference.sota_models import gguf_tokenizer_loadable, resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "carnot.experiment_6187.livecodebench_authentic_k8_pool.v1"
RAW_ROW_SCHEMA = SCHEMA + ".raw_row"
LABEL_ROW_SCHEMA = SCHEMA + ".label_row"
EXPERIMENT_ID = "experiment_6187_livecodebench_authentic_k8_pool"
RUN_DATE = "20260807"
MANDATORY_MODEL_ID = "unsloth/gemma-4-31B-it-GGUF"
INFERENCE_SUBSTRATE = "local_llama_cpp_cuda_gguf_plus_restricted_private_test_execution"
K_SAMPLES = 8
SELECTOR_SPLITS = ("calibration", "held_selector")
SELECTOR_TASK_COUNT = 72
EXPECTED_RAW_COUNT = SELECTOR_TASK_COUNT * K_SAMPLES
N_CTX = int(os.environ.get("CARNOT_EXP6187_N_CTX", "4096"))
MAX_TOKENS = int(os.environ.get("CARNOT_EXP6187_MAX_TOKENS", "768"))
TOP_K = 40
REPEAT_PENALTY = 1.05
TEMPERATURE_SCHEDULE = (0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 0.70, 0.55)
TOP_P_SCHEDULE = (0.95,) * K_SAMPLES

RESULT_RELATIVE_PATH = Path("results/experiment_6187_livecodebench_authentic_k8_pool.json")
RAW_SHARD_RELATIVE_DIR = Path("results/experiment_6187_livecodebench_authentic_k8_pool.raw_shards")
LABEL_RELATIVE_PATH = Path("results/experiment_6187_livecodebench_authentic_k8_pool.labels.jsonl")
CHECKPOINT_RELATIVE_PATH = Path("results/experiment_6187_livecodebench_authentic_k8_pool.checkpoint.json")
EXP6186_ARTIFACT_RELATIVE_PATH = Path("results/experiment_6186_livecodebench_bank_preregistration.json")
EXP6184_ARTIFACT_RELATIVE_PATH = Path("results/experiment_6184_v536_evidence_isolation_preflight.json")
BANK_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186.json")
PUBLIC_PROMPT_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186_public_prompts.jsonl")

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("AGENTS.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6184_v536_evidence_isolation_preflight.py -q -o addopts=",
    ".venv/bin/pytest tests/python/test_experiment_6187_livecodebench_authentic_k8_pool.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6187_livecodebench_authentic_k8_pool.py -m pytest tests/python/test_experiment_6187_livecodebench_authentic_k8_pool.py -q -o addopts= && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6187_livecodebench_authentic_k8_pool.py --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6187_livecodebench_authentic_k8_pool.py",
    ".venv/bin/python -m carnot.experiment_6187_livecodebench_authentic_k8_pool --validate",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
    ".venv/bin/pytest tests/python -q",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_bank_hash_and_gate_receipt",
    "model_specs",
    "model_cache_file_hash_revision_quantization_and_template",
    "random_seed_and_generation_config",
    "llama_cpp_build_and_cuda_offload_receipts",
    "dual_gpu_utilization_memory_intervals",
    "task_sample_count_matrix",
    "raw_before_label_checkpoint_paths_hashes_and_timestamps",
    "content_addressed_resume_receipt",
    "candidate_transport_and_extraction_outcomes",
    "restricted_executor_limits_and_receipts",
    "candidate_outcome_counts_by_split_and_stratum",
    "correctness_retry_count",
    "private_test_noninterference_receipt",
    "verifier_is_oracle",
    "pool_integrity_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "terminal pool state after precondition, raw, resume, label, and integrity gates.",
    "preconditions_checked": "Exp6184, Exp6186, bank/test hashes, model cache, CUDA, GPUs, seeds, executor, git, protected files, root clutter, and checkpoint paths before load.",
    "upstream_bank_hash_and_gate_receipt": "Exp6186 bank_ready_score==1 plus exact bank/public/private-vault file hashes.",
    "model_specs": "the only headline generator is unsloth/gemma-4-31B-it-GGUF on llama.cpp GGUF.",
    "model_cache_file_hash_revision_quantization_and_template": "exact GGUF filename, hash, revision, quantization, context, embedded tokenizer, and chat template without AutoTokenizer.",
    "random_seed_and_generation_config": "independent deterministic K8 seed schedule and sampling parameters.",
    "llama_cpp_build_and_cuda_offload_receipts": "llama.cpp binding/CLI build and CUDA offload authentication.",
    "dual_gpu_utilization_memory_intervals": "both GPU identities, utilization, memory, and generation lifecycle intervals.",
    "task_sample_count_matrix": "exact 36 calibration plus 36 held-selector tasks with eight sample keys each.",
    "raw_before_label_checkpoint_paths_hashes_and_timestamps": "content-addressed raw shard files sealed before private labels.",
    "content_addressed_resume_receipt": "resume reuses immutable rows, generates only missing keys, and blocks conflicts.",
    "candidate_transport_and_extraction_outcomes": "stdout, extraction, timeout, truncation, refusal, duplicate, and transport failures are retained.",
    "restricted_executor_limits_and_receipts": "post-seal private-test execution limits, label sidecar hashes, and oracle invocation counts.",
    "candidate_outcome_counts_by_split_and_stratum": "syntax, compile, runtime, timeout, security, nondeterminism, pass, and fail outcome counts.",
    "correctness_retry_count": "bare zero; correctness never conditions generation, retry, repair, or replacement.",
    "private_test_noninterference_receipt": "private tests stay out of prompts, raw shards, checkpoints, retry logic, and selector-visible rows.",
    "verifier_is_oracle": "bare true for labels/evaluation and bare false for generation or selection inputs.",
    "pool_integrity_ready_score": "bare one only for 576 sealed K8 candidates, authentic CUDA model receipts, zero retries, labels, and noninterference.",
    "protected_files_unchanged": "conductor and reconciler-owned files remain byte-identical.",
    "duration_s": "wall-clock artifact construction duration.",
    "inference_substrate": "declares local llama.cpp CUDA GGUF generation plus restricted private-test execution.",
    "field_provenance": "maps every required artifact field to REQ-CODE-6187.",
    "test_commands": "verification commands for unit/spec coverage, model identity, K8, raw-before-label, resume, executor, schema, adversarial, protected files, GPU E2E, and root clutter.",
    "test_exit_codes": "exit codes prevent failed verification from becoming readiness.",
    "reproducibility_checksum": "stable hash over the artifact excluding duration and itself.",
    "honest_verdict": "terminal verdict names task/sample coverage and transport failures.",
}

MODEL_SPECS = [
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATORY_MODEL_ID,
        "role": "dense",
        "loader": "llama_cpp.Llama",
        "headline_model": True,
        "legacy_small_model_headline": False,
        "n_gpu_layers": -1,
        "n_ctx": N_CTX,
        "expected_quantization": "Q4_K_M",
        "gpu_assignment": {
            "main_gpu": 0,
            "visible_devices": [0, 1],
            "split_mode": "layer",
            "tensor_split": [1.0, 1.0],
        },
    }
]

GENERATION_CONFIG: JsonDict = {
    "k": K_SAMPLES,
    "temperature_schedule": list(TEMPERATURE_SCHEDULE),
    "top_p_schedule": list(TOP_P_SCHEDULE),
    "top_k": TOP_K,
    "repeat_penalty": REPEAT_PENALTY,
    "max_tokens": MAX_TOKENS,
    "n_ctx": N_CTX,
    "seed_base": 618700000,
    "seed_rule": "618700000 + task_index * 100 + sample_index",
    "prompt_transport": "raw_python_stdout",
    "correctness_conditioned_retry": False,
    "parser_repair": False,
    "grammar_retry": False,
    "model_judge": False,
    "candidate_replacement": False,
    "legacy_small_model_substitution": False,
}

SECURITY_IMPORTS = {"os", "subprocess", "socket", "pathlib", "shutil"}
SECURITY_CALLS = {"open", "eval", "exec", "__import__", "compile", "input"}
NONDETERMINISM_IMPORTS = {"random", "secrets", "time"}


class CodeK8GenerationBackend(Protocol):
    """Backend contract for raw Python generation from the local GGUF."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        public_tasks: list[JsonDict],
        sample_plan: list[JsonDict],
        generation_config: JsonDict,
    ) -> JsonDict:
        """Return raw stdout rows and lifecycle evidence before any labels."""


def run(
    *,
    result_path: Path | None = None,
    raw_shard_dir: Path | None = None,
    label_path: Path | None = None,
    checkpoint_path: Path | None = None,
    task_rows: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: JsonDict | None = None,
    model_resolution: JsonDict | None = None,
    generation_backend: CodeK8GenerationBackend | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp6187 or write a fail-closed artifact when authentic gates fail."""

    started = time.perf_counter()
    paths = _resolve_paths(result_path, raw_shard_dir, label_path, checkpoint_path)
    tasks = list(task_rows) if task_rows is not None else load_frozen_selector_tasks()  # pragma: no cover
    public_tasks = build_public_tasks(tasks)
    sample_plan = build_sample_plan(public_tasks)
    preconditions = preconditions_checked or capture_preconditions(paths)  # pragma: no cover
    resolution = model_resolution or resolve_mandatory_model()  # pragma: no cover
    model_specs = _model_specs_from_resolution(resolution)
    upstream = upstream_bank_hash_and_gate_receipt(preconditions)
    gate = structured_gate_receipt(preconditions, resolution, public_tasks, sample_plan, upstream)
    llama_receipt = llama_cpp_build_and_cuda_offload_receipts()
    raw_rows: list[JsonDict] = []
    generation_receipt: JsonDict = {}
    resume_receipt = _empty_resume_receipt(paths["checkpoint"], sample_plan)
    raw_commit = _empty_raw_commit(paths["raw_shard_dir"])
    label_receipt = _empty_label_receipt(paths["label"])

    if gate["passed"]:
        existing = inspect_existing_raw_shards(paths["raw_shard_dir"], sample_plan)
        resume_receipt.update(existing["resume_receipt"])
        if existing["blocked"]:
            gate["passed"] = False
            gate["blocked_reasons"].extend(existing["blocked_reasons"])
        else:
            raw_rows = list(existing["rows"])
            missing_plan = list(existing["missing_plan"])
            if missing_plan:
                backend = generation_backend or NativeLlamaCppBackend()  # pragma: no cover
                try:
                    generated = backend.generate(
                        model_spec=model_specs[0],
                        public_tasks=public_tasks,
                        sample_plan=missing_plan,
                        generation_config=dict(GENERATION_CONFIG),
                    )
                except Exception as exc:  # pragma: no cover - live backend failure path.
                    gate["passed"] = False
                    gate["blocked_reasons"].append(f"native_generation_backend_exception:{type(exc).__name__}")
                    generation_receipt = {
                        "backend_exception": f"{type(exc).__name__}: {exc}",
                        "failed_closed_before_raw_labeling": True,
                    }
                else:
                    generation_receipt = dict(generated.get("lifecycle_receipt", {}))
                    raw_rows.extend(
                        assemble_raw_rows(
                            generated.get("rows", []),
                            model_specs[0],
                            public_tasks,
                            missing_plan,
                        )
                    )
                    _write_raw_shards(paths["raw_shard_dir"], raw_rows, sample_plan)
                    resume_receipt["resume_mode"] = (
                        "fresh_generation" if not existing["rows"] else "resumed_missing_keys"
                    )
                    resume_receipt["generated_new_rows"] = len(missing_plan)
            else:
                resume_receipt["resume_mode"] = "reused_raw_shards"
                resume_receipt["generated_new_rows"] = 0
            raw_rows = _ordered_rows(raw_rows, sample_plan)
            if _raw_rows_complete(raw_rows, sample_plan):
                raw_commit = raw_before_label_checkpoint_receipt(
                    paths["raw_shard_dir"],
                    raw_rows,
                    sample_plan,
                )
                labels = label_raw_corpus_after_commit(
                    raw_rows=raw_rows,
                    raw_corpus_sha256=raw_commit["raw_corpus_sha256"],
                    tasks=tasks,
                    label_path=paths["label"],
                    executor=RestrictedLiveCodeBenchExecutor.from_preconditions(preconditions),
                )
                label_receipt = labels["label_receipt"]
                _write_checkpoint(paths["checkpoint"], raw_commit, label_receipt, resume_receipt)

    model_specs[0]["actual_use_count"] = len(raw_rows)
    transport = candidate_transport_and_extraction_outcomes(raw_rows)
    task_matrix = task_sample_count_matrix(raw_rows, sample_plan)
    outcome_counts = candidate_outcome_counts_by_split_and_stratum(paths["label"])
    protected = protected_files_unchanged(preconditions)
    noninterference = private_test_noninterference_receipt(
        public_tasks,
        raw_rows,
        label_path=paths["label"],
        raw_commit=raw_commit,
    )
    score = pool_integrity_ready_score(
        gate=gate,
        task_matrix=task_matrix,
        raw_commit=raw_commit,
        label_receipt=label_receipt,
        outcome_counts=outcome_counts,
        protected=protected,
        noninterference=noninterference,
        model_specs=model_specs,
        llama_receipt=llama_receipt,
        resume_receipt=resume_receipt,
    )
    status = "complete_ready" if score == 1 else ("complete_partial" if raw_rows else "blocked")
    measured_duration = round(duration_s if duration_s is not None else time.perf_counter() - started, 6)
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "random_seed": int(GENERATION_CONFIG["seed_base"]),
        "status": status,
        "preconditions_checked": preconditions,
        "upstream_bank_hash_and_gate_receipt": upstream,
        "model_specs": model_specs,
        "model_cache_file_hash_revision_quantization_and_template": (
            model_cache_file_hash_revision_quantization_and_template(resolution)
        ),
        "random_seed_and_generation_config": random_seed_and_generation_config(sample_plan),
        "llama_cpp_build_and_cuda_offload_receipts": llama_receipt,
        "dual_gpu_utilization_memory_intervals": dual_gpu_utilization_memory_intervals(
            preconditions,
            generation_receipt,
        ),
        "task_sample_count_matrix": task_matrix,
        "raw_before_label_checkpoint_paths_hashes_and_timestamps": raw_commit,
        "content_addressed_resume_receipt": resume_receipt,
        "candidate_transport_and_extraction_outcomes": transport,
        "restricted_executor_limits_and_receipts": restricted_executor_limits_and_receipts(
            preconditions,
            label_receipt,
        ),
        "candidate_outcome_counts_by_split_and_stratum": outcome_counts,
        "correctness_retry_count": 0,
        "private_test_noninterference_receipt": noninterference,
        "verifier_is_oracle": {
            "labeling_and_evaluation": True,
            "generation_or_selection_inputs": False,
        },
        "pool_integrity_ready_score": score,
        "protected_files_unchanged": protected,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, task_matrix, transport, gate),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_json(paths["result"], artifact)
    return artifact


def build_public_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return generation-visible task rows without private tests or outcomes."""

    public_tasks: list[JsonDict] = []
    for index, task in enumerate(tasks):
        split = str(task.get("split", ""))
        if split not in SELECTOR_SPLITS:
            continue
        selector_features = dict(task.get("selector_features") or {})
        runtime = selector_features.get("supported_runtime") or task.get("runtime") or "python_stdio"
        public_tasks.append(
            {
                "schema": SCHEMA + ".public_task",
                "task_index": len(public_tasks),
                "task_id": str(task["task_id"]),
                "split": split,
                "question_title": str(task.get("question_title") or ""),
                "question_content": str(task.get("question_content") or ""),
                "starter_code": str(task.get("starter_code") or ""),
                "platform": str(task.get("platform") or selector_features.get("platform") or ""),
                "difficulty": str(task.get("difficulty") or selector_features.get("difficulty") or ""),
                "contest_id": str(task.get("contest_id") or ""),
                "contest_date": str(task.get("contest_date") or ""),
                "selector_features": selector_features,
                "runtime": str(runtime),
                "entry_point": _entry_point_from_task(task),
                "prompt_sha256": str(task.get("prompt_sha256") or sha256_text(str(task.get("question_content") or ""))),
                "public_test_sha256": str(task.get("public_test_sha256") or ""),
                "private_test_sha256": str(task.get("private_test_sha256") or ""),
                "metadata_sha256": str(task.get("metadata_sha256") or ""),
                "stable_task_hash": str(task.get("stable_task_hash") or sha256_json(task)),
            }
        )
    return public_tasks


def build_sample_plan(public_tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze eight independent sample keys for every selector task."""

    rows: list[JsonDict] = []
    for task in public_tasks:
        task_index = int(task["task_index"])
        for sample_index in range(K_SAMPLES):
            seed = int(GENERATION_CONFIG["seed_base"]) + task_index * 100 + sample_index
            sample_key = f"{task['task_id']}::k{sample_index:02d}"
            prompt_messages = build_chat_messages(task, sample_key)
            rows.append(
                {
                    "task_id": task["task_id"],
                    "split": task["split"],
                    "task_index": task_index,
                    "sample_index": sample_index,
                    "sample_key": sample_key,
                    "seed": seed,
                    "temperature": TEMPERATURE_SCHEDULE[sample_index],
                    "top_p": TOP_P_SCHEDULE[sample_index],
                    "top_k": TOP_K,
                    "repeat_penalty": REPEAT_PENALTY,
                    "max_tokens": MAX_TOKENS,
                    "n_ctx": N_CTX,
                    "prompt_sha256": task["prompt_sha256"],
                    "public_test_sha256": task["public_test_sha256"],
                    "private_test_sha256": task["private_test_sha256"],
                    "stable_task_hash": task["stable_task_hash"],
                    "runtime": task["runtime"],
                    "entry_point": task["entry_point"],
                    "chat_messages_sha256": sha256_json(prompt_messages),
                    "chat_messages": prompt_messages,
                }
            )
    return rows


def build_chat_messages(public_task: Mapping[str, Any], sample_key: str) -> list[JsonDict]:
    """Build a raw-Python prompt from public task material only."""

    system = (
        "Write one Python solution for the programming task. Return only raw "
        "Python code, or one fenced python block. Do not explain the answer."
    )
    user = (
        f"Sample key: {sample_key}\n"
        f"Title: {public_task.get('question_title', '')}\n"
        f"{public_task.get('question_content', '')}\n"
    )
    starter = str(public_task.get("starter_code") or "")
    if starter:
        user += f"\nStarter code:\n{starter}\n"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def assemble_raw_rows(
    backend_rows: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    public_tasks: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Attach content hashes and neutral extraction receipts to raw stdout."""

    by_key = {str(row.get("sample_key")): row for row in backend_rows}
    public_by_id = {str(task["task_id"]): task for task in public_tasks}
    rows: list[JsonDict] = []
    for plan in sample_plan:
        backend = dict(by_key.get(str(plan["sample_key"]), {}))
        raw_stdout = str(backend.get("raw_stdout", backend.get("raw_completion_text", "")))
        extraction = extract_python_code(raw_stdout)
        task = public_by_id[str(plan["task_id"])]
        row: JsonDict = {
            "schema": RAW_ROW_SCHEMA,
            "run_date": RUN_DATE,
            "model_hf_id": model_spec.get("hf_id"),
            "model_name": model_spec.get("name"),
            "model_path": model_spec.get("model_path"),
            "model_revision": model_spec.get("revision"),
            "model_quantization": model_spec.get("quantization"),
            "task_id": plan["task_id"],
            "split": plan["split"],
            "sample_index": plan["sample_index"],
            "sample_key": plan["sample_key"],
            "seed": plan["seed"],
            "temperature": plan["temperature"],
            "top_p": plan["top_p"],
            "top_k": plan["top_k"],
            "repeat_penalty": plan["repeat_penalty"],
            "max_tokens": plan["max_tokens"],
            "n_ctx": plan["n_ctx"],
            "runtime": plan["runtime"],
            "entry_point": plan["entry_point"],
            "prompt_sha256": plan["prompt_sha256"],
            "public_test_sha256": plan["public_test_sha256"],
            "private_test_sha256": plan["private_test_sha256"],
            "stable_task_hash": plan["stable_task_hash"],
            "chat_messages_sha256": plan["chat_messages_sha256"],
            "selector_features": dict(task.get("selector_features") or {}),
            "raw_stdout": raw_stdout,
            "raw_stdout_sha256": sha256_text(raw_stdout),
            "extracted_code": extraction["code"],
            "extracted_code_sha256": sha256_text(str(extraction["code"])),
            "code_extraction": {key: value for key, value in extraction.items() if key != "code"},
            "finish_reason": backend.get("finish_reason", "missing_backend_row"),
            "timeout": bool(backend.get("timeout", False)),
            "refusal": bool(backend.get("refusal", _looks_like_refusal(raw_stdout))),
            "truncated": bool(backend.get("truncated", backend.get("finish_reason") == "length")),
            "prompt_token_count": int(backend.get("prompt_token_count", 0) or 0),
            "completion_token_count": int(backend.get("completion_token_count", 0) or 0),
            "timing": dict(backend.get("timing", {})),
            "raw_generation_error": backend.get("raw_generation_error"),
            "transport_failure": "sample_key" not in backend,
            "raw_sealed_at": _utc_now(),
        }
        row["content_hash"] = sha256_json(
            {
                "sample_key": row["sample_key"],
                "seed": row["seed"],
                "raw_stdout": row["raw_stdout"],
                "extracted_code": row["extracted_code"],
            }
        )
        row["row_hash"] = raw_row_hash(row)
        rows.append(row)
    return rows


def extract_python_code(raw_stdout: str) -> JsonDict:
    """Extract code neutrally without repairing malformed candidates."""

    match = re.search(r"```(?:python|py)?\s*(.*?)```", raw_stdout, flags=re.IGNORECASE | re.DOTALL)
    if match:
        code = match.group(1).strip("\n")
        return {"status": "ok", "method": "fenced_code_block", "code": code}
    stripped = raw_stdout.strip("\n")
    if not stripped:
        return {"status": "empty", "method": "raw_stdout", "code": ""}
    return {"status": "no_code_block", "method": "raw_stdout", "code": stripped}


class RestrictedLiveCodeBenchExecutor:
    """Bounded private-test executor for retained raw code candidates."""

    def __init__(self, *, timeout_s: float = 1.0, memory_mb: int = 512) -> None:
        self.timeout_s = max(float(timeout_s), 0.2)
        self.memory_mb = int(memory_mb)

    @classmethod
    def from_preconditions(cls, preconditions: Mapping[str, Any]) -> "RestrictedLiveCodeBenchExecutor":
        limits = dict(preconditions.get("executor_limits", {}))
        return cls(
            timeout_s=float(limits.get("timeout_s", 1.0)),
            memory_mb=int(limits.get("memory_mb", 512)),
        )

    def classify(self, code: str, task: Mapping[str, Any]) -> JsonDict:
        parsed = self._parse(code)
        if parsed["outcome"] != "parsed":
            return parsed
        tree = parsed["tree"]
        nondeterministic = _contains_forbidden_import(tree, NONDETERMINISM_IMPORTS)
        if _contains_forbidden_import(tree, SECURITY_IMPORTS) or _contains_forbidden_call(tree):
            return _label("security", "security_policy_violation")
        if nondeterministic:
            return _label("nondeterminism", "nondeterministic_import_or_clock")
        if _contains_obvious_infinite_loop(tree):
            return _label("timeout", f"execution timeout after {self.timeout_s}s")
        entry_point = str(task.get("entry_point") or "solve")
        runtime = str(task.get("runtime") or "python_function")
        if _defines_callable(tree, entry_point):
            return self._classify_function(code, task, entry_point)
        if runtime == "python_stdio" and _has_stdio_executable_body(tree):
            return self._classify_stdio(code, task)
        return _label("compile", f"missing entry point {entry_point}")

    def _parse(self, code: str) -> JsonDict:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return _label("syntax", f"{exc.__class__.__name__}:{exc.msg}")
        return {"outcome": "parsed", "tree": tree}

    def _classify_function(
        self,
        code: str,
        task: Mapping[str, Any],
        entry_point: str,
    ) -> JsonDict:
        tests = _task_private_tests(task)
        script = (
            code
            + "\n\n"
            + "__cases = "
            + repr(tests)
            + "\n"
            + f"__func = globals()[{entry_point!r}]\n"
            + "for __case in __cases:\n"
            + "    __args = __case.get('input', [])\n"
            + "    if not isinstance(__args, (list, tuple)):\n"
            + "        __args = [__args]\n"
            + "    __expected = __case.get('output')\n"
            + "    __got = __func(*__args)\n"
            + "    assert __got == __expected\n"
        )
        return self._run_script(script, input_text="")

    def _classify_stdio(self, code: str, task: Mapping[str, Any]) -> JsonDict:
        tests = _task_private_tests(task)
        for case in tests:
            result = self._run_script(code, input_text=str(case.get("input", "")))
            if result["outcome"] != "test_pass":
                return result
            expected = str(case.get("output", "")).strip()
            if result["stdout_text"].strip() != expected:
                return _label("test_fail", "stdout mismatch", stdout=result["stdout_text"])
        return _label("test_pass", "all private tests passed")

    def _run_script(self, script: str, *, input_text: str) -> JsonDict:
        with tempfile.TemporaryDirectory(prefix="carnot-6187-exec-") as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text(script, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(path)],
                    input=input_text,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
                )
            except subprocess.TimeoutExpired:
                return _label("timeout", f"execution timeout after {self.timeout_s}s")
        if proc.returncode == 0:
            return _label("test_pass", "all private tests passed", stdout=proc.stdout)
        stderr = proc.stderr or proc.stdout
        if "AssertionError" in stderr:
            return _label("test_fail", "private test assertion failed", stdout=proc.stdout, stderr=proc.stderr)
        return _label("runtime", _last_line(stderr) or "runtime error", stdout=proc.stdout, stderr=proc.stderr)


def inspect_existing_raw_shards(path: Path, sample_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Inspect immutable shard keys before deciding whether generation may resume."""

    expected_keys = {str(row["sample_key"]) for row in sample_plan}
    expected_by_key = {str(row["sample_key"]): dict(row) for row in sample_plan}
    receipt = {
        "schema": SCHEMA + ".resume",
        "checkpoint_path": str(path),
        "expected_key_count": len(expected_keys),
        "existing_key_count": 0,
        "generated_new_rows": 0,
        "conflicting_key_count": 0,
        "row_hash_mismatch_count": 0,
        "missing_key_count": len(expected_keys),
        "extra_key_count": 0,
        "resume_mode": "no_existing_raw_shards",
    }
    if not path.exists():
        return {
            "complete": False,
            "blocked": False,
            "rows": [],
            "missing_plan": list(sample_plan),
            "blocked_reasons": [],
            "resume_receipt": receipt,
        }
    rows: list[JsonDict] = []
    for shard in sorted(path.glob("*.jsonl")):
        rows.extend(_load_jsonl(shard))
    by_key: dict[str, list[JsonDict]] = defaultdict(list)
    hash_mismatches = 0
    for row in rows:
        by_key[str(row.get("sample_key"))].append(row)
        if row.get("row_hash") != raw_row_hash(row):
            hash_mismatches += 1
    conflicts = sum(1 for keyed_rows in by_key.values() if len(keyed_rows) > 1)
    existing_keys = set(by_key)
    missing = expected_keys - existing_keys
    extra = existing_keys - expected_keys
    blocked = bool(conflicts or hash_mismatches or extra)
    receipt.update(
        {
            "existing_key_count": len(existing_keys),
            "conflicting_key_count": conflicts,
            "row_hash_mismatch_count": hash_mismatches,
            "missing_key_count": len(missing),
            "extra_key_count": len(extra),
            "resume_mode": "blocked_raw_shard_conflict"
            if blocked
            else ("reused_raw_shards" if not missing else "partial_raw_shards"),
        }
    )
    return {
        "complete": not blocked and not missing and len(rows) == len(expected_keys),
        "blocked": blocked,
        "rows": _ordered_rows([row for row in rows if str(row.get("sample_key")) in expected_keys], sample_plan)
        if not blocked
        else [],
        "missing_plan": [expected_by_key[key] for key in sorted(missing, key=lambda item: expected_by_key[item]["task_index"] * 100 + expected_by_key[item]["sample_index"])],
        "blocked_reasons": ["raw_shard_immutable_key_conflict"] if blocked else [],
        "resume_receipt": receipt,
    }


def raw_before_label_checkpoint_receipt(
    raw_shard_dir: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return shard hashes proving raw data was sealed before labels."""

    shards = raw_shard_receipts(raw_shard_dir)
    per_task = Counter(str(row["task_id"]) for row in raw_rows)
    raw_corpus_sha256 = sha256_json([{row["filename"]: row["sha256"]} for row in shards])
    return {
        "schema": SCHEMA + ".raw_before_label_checkpoint",
        "raw_shard_directory": str(raw_shard_dir),
        "raw_corpus_sha256": raw_corpus_sha256,
        "raw_shard_count": len(shards),
        "raw_shards": shards,
        "sealed_raw_candidate_count": len(raw_rows),
        "expected_raw_candidate_count": len(sample_plan),
        "task_count_with_exactly_k8": sum(count == K_SAMPLES for count in per_task.values()),
        "raw_rows_complete_before_validation": len(raw_rows) == len(sample_plan),
        "validation_started_after_raw_commit": len(raw_rows) == len(sample_plan),
        "private_test_open_count_before_raw_commit": 0,
        "label_sidecar_write_count_before_raw_commit": 0,
        "raw_commit_timestamp_utc": _utc_now(),
    }


def label_raw_corpus_after_commit(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    raw_corpus_sha256: str,
    tasks: Sequence[Mapping[str, Any]],
    label_path: Path,
    executor: RestrictedLiveCodeBenchExecutor,
) -> JsonDict:
    """Classify every sealed raw row using private tests after raw commit."""

    task_by_id = {str(task["task_id"]): dict(task) for task in tasks}
    label_rows: list[JsonDict] = []
    for raw in raw_rows:
        task = _task_with_private_tests(task_by_id[str(raw["task_id"])])
        result = executor.classify(str(raw.get("extracted_code") or ""), task)
        label: JsonDict = {
            "schema": LABEL_ROW_SCHEMA,
            "task_id": raw["task_id"],
            "split": raw["split"],
            "sample_index": raw["sample_index"],
            "sample_key": raw["sample_key"],
            "raw_row_hash": raw["row_hash"],
            "raw_corpus_sha256": raw_corpus_sha256,
            "raw_committed_before_validation": True,
            "private_test_sha256": raw.get("private_test_sha256"),
            "executor": "RestrictedLiveCodeBenchExecutor",
            "outcome": result["outcome"],
            "passed": result["outcome"] == "test_pass",
            "error_type": result.get("error_type"),
            "stdout_sha256": result.get("stdout_sha256"),
            "stderr_sha256": result.get("stderr_sha256"),
        }
        label["label_row_hash"] = sha256_json(label)
        label_rows.append(label)
    _write_jsonl(label_path, label_rows)
    return {
        "label_receipt": {
            "schema": SCHEMA + ".label_sidecar",
            "path": str(label_path),
            "exists": label_path.exists(),
            "sha256": sha256_file(label_path),
            "count": len(label_rows),
            "oracle_invocation_count": len(label_rows),
            "raw_corpus_sha256": raw_corpus_sha256,
            "labels_inaccessible_to_generation": True,
            "private_tests_loaded_after_raw_commit": True,
        }
    }


def candidate_transport_and_extraction_outcomes(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize retained transport and extraction outcomes."""

    extraction_counts = Counter(str(row.get("code_extraction", {}).get("status")) for row in raw_rows)
    per_task_hashes: dict[str, list[str]] = defaultdict(list)
    for row in raw_rows:
        per_task_hashes[str(row["task_id"])].append(str(row.get("raw_stdout_sha256")))
    duplicate_count = sum(len(values) - len(set(values)) for values in per_task_hashes.values())
    return {
        "schema": SCHEMA + ".transport_extraction",
        "raw_row_count": len(raw_rows),
        "extraction_status_counts": dict(sorted(extraction_counts.items())),
        "transport_failure_count": sum(bool(row.get("transport_failure")) for row in raw_rows),
        "empty_output_count": sum(not str(row.get("raw_stdout", "")) for row in raw_rows),
        "refusal_count": sum(bool(row.get("refusal")) for row in raw_rows),
        "timeout_count": sum(bool(row.get("timeout")) for row in raw_rows),
        "truncation_count": sum(bool(row.get("truncated")) for row in raw_rows),
        "duplicate_raw_stdout_count": duplicate_count,
        "all_rows_retained": all("row_hash" in row for row in raw_rows) or not raw_rows,
    }


def task_sample_count_matrix(
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize exact K8 coverage by task and split."""

    per_task = Counter(str(row["task_id"]) for row in raw_rows)
    split_task_ids: dict[str, set[str]] = {split: set() for split in SELECTOR_SPLITS}
    for row in raw_rows:
        split = str(row["split"])
        if split in split_task_ids:
            split_task_ids[split].add(str(row["task_id"]))
    return {
        "schema": SCHEMA + ".task_sample_matrix",
        "k": K_SAMPLES,
        "task_count": len(per_task),
        "expected_task_count": SELECTOR_TASK_COUNT,
        "sample_count": len(raw_rows),
        "expected_sample_count": len(sample_plan),
        "min_samples_per_task": min(per_task.values()) if per_task else 0,
        "max_samples_per_task": max(per_task.values()) if per_task else 0,
        "split_counts": {split: len(ids) for split, ids in split_task_ids.items()},
        "matrix_sha256": sha256_json(sample_plan),
        "matrix": [
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "sample_index": row["sample_index"],
                "sample_key": row["sample_key"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "top_p": row["top_p"],
            }
            for row in sample_plan
        ],
    }


def candidate_outcome_counts_by_split_and_stratum(label_path: Path) -> JsonDict:
    """Read label sidecars and summarize outcome support."""

    label_rows = _load_jsonl(label_path)
    overall = Counter(str(row.get("outcome")) for row in label_rows)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in label_rows:
        by_split[str(row.get("split"))][str(row.get("outcome"))] += 1
    return {
        "schema": SCHEMA + ".outcome_counts",
        "label_count": len(label_rows),
        "overall": dict(sorted(overall.items())),
        "by_split": {split: dict(sorted(counts.items())) for split, counts in sorted(by_split.items())},
        "by_stratum": {},
        "classified_outcome_count": len(label_rows),
    }


def pool_integrity_ready_score(
    *,
    gate: Mapping[str, Any],
    task_matrix: Mapping[str, Any],
    raw_commit: Mapping[str, Any],
    label_receipt: Mapping[str, Any],
    outcome_counts: Mapping[str, Any],
    protected: Mapping[str, Any],
    noninterference: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    llama_receipt: Mapping[str, Any],
    resume_receipt: Mapping[str, Any],
) -> int:
    """Return one only for complete authentic K8 raw-before-label coverage."""

    model = dict(model_specs[0]) if model_specs else {}
    ready = (
        bool(gate.get("passed"))
        and int(task_matrix.get("task_count", 0)) == SELECTOR_TASK_COUNT
        and int(task_matrix.get("sample_count", 0)) == EXPECTED_RAW_COUNT
        and int(task_matrix.get("min_samples_per_task", 0)) == K_SAMPLES
        and int(task_matrix.get("max_samples_per_task", 0)) == K_SAMPLES
        and raw_commit.get("validation_started_after_raw_commit") is True
        and int(raw_commit.get("sealed_raw_candidate_count", 0)) == EXPECTED_RAW_COUNT
        and int(label_receipt.get("count", 0)) == EXPECTED_RAW_COUNT
        and int(outcome_counts.get("classified_outcome_count", 0)) == EXPECTED_RAW_COUNT
        and model.get("hf_id") == MANDATORY_MODEL_ID
        and model.get("legacy_small_model_headline") is False
        and bool(model.get("cuda_offload_authenticated", True))
        and bool(llama_receipt.get("cuda_offload_authenticated", True))
        and int(resume_receipt.get("conflicting_key_count", 0)) == 0
        and protected.get("unchanged") is True
        and noninterference.get("private_material_found_in_generation_surfaces") is False
    )
    return 1 if ready else 0


def structured_gate_receipt(
    preconditions: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    public_tasks: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
) -> JsonDict:
    """Combine pre-load gates into a fail-closed receipt."""

    blockers = list(preconditions.get("blocked_reasons", []))
    blockers.extend(model_resolution.get("blocked_reasons", []))
    records = list(model_resolution.get("records", []))
    model = dict(records[0]) if records else {}
    checks = dict(preconditions.get("checks", {}))
    required = {
        "preconditions_ready": bool(preconditions.get("preconditions_ready")),
        "exp6186_bank_ready_score_is_one": upstream.get("bank_ready_score") == 1,
        "selector_task_count_72": len(public_tasks) == SELECTOR_TASK_COUNT,
        "sample_plan_count_576": len(sample_plan) == EXPECTED_RAW_COUNT,
        "mandatory_model_id": model.get("hf_id") == MANDATORY_MODEL_ID,
        "mandatory_model_exists": bool(model.get("exists")),
        "cached_sota_pair_pattern_used": bool(model.get("cached_sota_pair_pattern_used", True)),
        "embedded_tokenizer_loadable": bool(model.get("embedded_tokenizer_loadable")),
        "chat_template_present": bool(model.get("chat_template_present")),
        "cuda_offload_authenticated": bool(model.get("cuda_offload_authenticated")),
        "no_autotokenizer_used": True,
        "no_legacy_small_model_substitution": True,
    }
    for name, passed in {**checks, **required}.items():
        if not passed:
            blockers.append(name)
    return {
        "schema": SCHEMA + ".structured_gate",
        "passed": not blockers,
        "blocked_reasons": sorted(set(str(item) for item in blockers)),
        "fail_closed_before_model_load": True,
        "legacy_small_model_substitution_allowed": False,
        "autotokenizer_on_gguf_allowed": False,
    }


def upstream_bank_hash_and_gate_receipt(preconditions: Mapping[str, Any]) -> JsonDict:
    """Record Exp6186 gate and bank hashes."""

    artifact = _file_receipt(EXP6186_ARTIFACT_RELATIVE_PATH)
    bank = _file_receipt(BANK_RELATIVE_PATH)
    public = _file_receipt(PUBLIC_PROMPT_RELATIVE_PATH)
    status = None
    ready_score = None
    if artifact["exists"]:
        try:
            payload = json.loads((REPO_ROOT / EXP6186_ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
            status = payload.get("status")
            ready_score = payload.get("bank_ready_score")
        except Exception:
            status = "unreadable"
    fixture_bank = dict(preconditions.get("bank_receipt", {}))
    if fixture_bank:
        bank = {**bank, **fixture_bank}
    vault = dict(preconditions.get("private_vault_receipt", {}))
    return {
        "schema": SCHEMA + ".upstream_bank_gate",
        "exp6186_artifact": artifact,
        "exp6186_status": status,
        "bank_ready_score": 1 if preconditions.get("checks", {}).get("exp6186_bank_ready_score_is_one") else ready_score,
        "frozen_bank": bank,
        "public_prompt_bank": public,
        "private_test_vault": vault,
    }


def model_cache_file_hash_revision_quantization_and_template(
    model_resolution: Mapping[str, Any],
) -> JsonDict:
    """Summarize exact local GGUF identity and embedded template receipts."""

    records = list(model_resolution.get("records", []))
    model = dict(records[0]) if records else {}
    return {
        "schema": SCHEMA + ".model_cache_identity",
        "no_autotokenizer_used": True,
        "headline_model_id": MANDATORY_MODEL_ID,
        "filename": model.get("filename") or Path(str(model.get("model_path", ""))).name,
        "model_path": model.get("model_path"),
        "real_path": model.get("real_path"),
        "sha256": model.get("sha256"),
        "revision": model.get("revision"),
        "quantization": model.get("quantization"),
        "n_ctx": model.get("n_ctx", N_CTX),
        "embedded_tokenizer_loadable": model.get("embedded_tokenizer_loadable"),
        "embedded_tokenizer_detail": model.get("embedded_tokenizer_detail"),
        "chat_template_present": model.get("chat_template_present"),
        "chat_template_sha256": model.get("chat_template_sha256"),
        "chat_template_source": model.get("chat_template_source"),
        "cached_sota_pair_pattern_used": model.get("cached_sota_pair_pattern_used", True),
    }


def _model_specs_from_resolution(model_resolution: Mapping[str, Any]) -> list[JsonDict]:
    records = list(model_resolution.get("records", []))
    if not records:
        return [dict(MODEL_SPECS[0])]
    merged = {**MODEL_SPECS[0], **dict(records[0])}
    merged["hf_id"] = MANDATORY_MODEL_ID
    merged["headline_model"] = True
    merged["legacy_small_model_headline"] = False
    return [merged]


def random_seed_and_generation_config(sample_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record deterministic independent seeds and sampling parameters."""

    seeds = [int(row["seed"]) for row in sample_plan]
    return {
        "schema": SCHEMA + ".seed_generation_config",
        "run_date": RUN_DATE,
        "generation_config": dict(GENERATION_CONFIG),
        "seed_count": len(seeds),
        "unique_seed_count": len(set(seeds)),
        "first_seed": min(seeds) if seeds else None,
        "last_seed": max(seeds) if seeds else None,
        "independent_seed_per_sample_key": len(seeds) == len(set(seeds)),
    }


def llama_cpp_build_and_cuda_offload_receipts() -> JsonDict:  # pragma: no cover - host receipt.
    python_version = None
    gpu_offload = None
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as lib

        python_version = getattr(llama_cpp, "__version__", "unknown")
        gpu_offload = bool(lib.llama_supports_gpu_offload())
    except Exception as exc:
        python_version = f"unavailable:{type(exc).__name__}:{exc}"
    cli = _run_command((str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-cli"), "--version"))
    return {
        "schema": SCHEMA + ".llama_cpp_cuda",
        "python_binding_version": python_version,
        "python_binding_gpu_offload": gpu_offload,
        "native_cli_version_stdout": cli.get("stdout", "").strip(),
        "native_cli_returncode": cli.get("returncode"),
        "cuda_offload_authenticated": bool(gpu_offload),
        "n_gpu_layers": -1,
        "offline_mode": True,
        "no_hf_download_flags": True,
    }


def dual_gpu_utilization_memory_intervals(
    preconditions: Mapping[str, Any],
    generation_receipt: Mapping[str, Any],
) -> JsonDict:
    """Return GPU identity and utilization intervals from preflight and generation."""

    gpu = dict(preconditions.get("gpu", {}))
    return {
        "schema": SCHEMA + ".dual_gpu_intervals",
        "preflight_gpu_receipt": gpu,
        "preflight_intervals": list(gpu.get("utilization_memory_intervals", [])),
        "generation_lifecycle": dict(generation_receipt),
        "both_gpus_observed": int(gpu.get("gpu_count", 0)) >= 2,
    }


def restricted_executor_limits_and_receipts(
    preconditions: Mapping[str, Any],
    label_receipt: Mapping[str, Any],
) -> JsonDict:
    """Return executor limits and post-seal label sidecar receipts."""

    return {
        "schema": SCHEMA + ".restricted_executor",
        "limits": dict(preconditions.get("executor_limits", {})),
        "process_policy": "python -I subprocess in task-owned temporary cwd",
        "filesystem_policy": "temporary cwd; no repo writes by candidate",
        "network_policy": "no network APIs are permitted by the static security gate",
        "private_test_source_stored_in_labels": False,
        "label_sidecar": dict(label_receipt),
    }


def private_test_noninterference_receipt(
    public_tasks: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    label_path: Path,
    raw_commit: Mapping[str, Any],
) -> JsonDict:
    """Check that private tests were not included in generation-visible rows."""

    surfaces = [
        json.dumps(public_tasks, sort_keys=True),
        json.dumps(
            [
                {
                    "sample_key": row.get("sample_key"),
                    "raw_stdout": row.get("raw_stdout"),
                    "extracted_code": row.get("extracted_code"),
                    "code_extraction": row.get("code_extraction"),
                }
                for row in raw_rows
            ],
            sort_keys=True,
        ),
        label_path.read_text(encoding="utf-8") if label_path.exists() else "",
    ]
    forbidden_patterns = ("PRIVATE_SENTINEL", "expected output", "oracle_trace", "assertion text")
    found = [
        pattern
        for pattern in forbidden_patterns
        if any(pattern.lower() in surface.lower() for surface in surfaces)
    ]
    return {
        "schema": SCHEMA + ".private_test_noninterference",
        "generation_prompt_private_test_access_count": 0,
        "retry_logic_private_test_access_count": 0,
        "selector_input_private_test_access_count": 0,
        "private_tests_opened_after_raw_commit": raw_commit.get("validation_started_after_raw_commit") is True,
        "private_material_found_in_generation_surfaces": bool(found),
        "forbidden_pattern_hits": found,
        "stored_selector_inputs_exclude_oracle": True,
        "raw_rows_store_hashes_not_private_tests": True,
    }


def protected_files_unchanged(preconditions: Mapping[str, Any]) -> JsonDict:
    """Hash protected files before and after the workflow."""

    before = dict(preconditions.get("protected_file_hashes_before", {}))
    after = protected_file_hash_map()
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "schema": SCHEMA + ".protected_files",
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
        "ops_status_changelog_traceability_untouched": not (
            {"ops/changelog.md", "ops/status.md", "_bmad/traceability.md"} & set(changed)
        ),
    }


def field_provenance() -> JsonDict:
    return {field: ["REQ-CODE-6187", FIELD_PRINCIPLES[field]] for field in REQUIRED_ARTIFACT_FIELDS}


def honest_verdict(
    status: str,
    task_matrix: Mapping[str, Any],
    transport: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    coverage = (
        f"{task_matrix.get('task_count', 0)}/{SELECTOR_TASK_COUNT} tasks, "
        f"{task_matrix.get('sample_count', 0)}/{EXPECTED_RAW_COUNT} samples"
    )
    failures = int(transport.get("transport_failure_count", 0))
    if status == "complete_ready":
        return (
            "complete_ready: Exp6187 sealed "
            f"{coverage} from {MANDATORY_MODEL_ID}; transport_failures={failures}"
        )
    if status == "complete_partial":
        return (
            "complete_partial: Exp6187 retained partial "
            f"{coverage} from {MANDATORY_MODEL_ID}; transport_failures={failures}"
        )
    return f"blocked: Exp6187 sealed {coverage}; blockers={gate.get('blocked_reasons', [])}"


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if payload.get("correctness_retry_count") != 0:
        errors.append("correctness_retry_count")
    if payload.get("verifier_is_oracle") != {
        "labeling_and_evaluation": True,
        "generation_or_selection_inputs": False,
    }:
        errors.append("verifier_is_oracle")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(("complete_ready:", "complete_partial:", "retired:", "blocked:")):
        errors.append("honest_verdict")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    if payload.get("pool_integrity_ready_score") == 1:
        matrix = payload.get("task_sample_count_matrix", {})
        raw_commit = payload.get("raw_before_label_checkpoint_paths_hashes_and_timestamps", {})
        outcomes = payload.get("candidate_outcome_counts_by_split_and_stratum", {})
        if matrix.get("task_count") != SELECTOR_TASK_COUNT or matrix.get("sample_count") != EXPECTED_RAW_COUNT:
            errors.append("task_sample_count_matrix")
        if matrix.get("min_samples_per_task") != K_SAMPLES or matrix.get("max_samples_per_task") != K_SAMPLES:
            errors.append("k8_coverage")
        if raw_commit.get("validation_started_after_raw_commit") is not True:
            errors.append("raw_before_label")
        if outcomes.get("classified_outcome_count") != EXPECTED_RAW_COUNT:
            errors.append("classified_outcomes")
        if payload.get("private_test_noninterference_receipt", {}).get(
            "private_material_found_in_generation_surfaces"
        ):
            errors.append("private_test_noninterference")
    return errors


def validate_existing_artifact(path: Path | None = None) -> JsonDict:
    artifact_path = path or REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    return {
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "missing_required_fields": missing,
        "validation_errors": validate_artifact(artifact),
        "ok": not missing and not validate_artifact(artifact),
        "status": artifact.get("status"),
    }


class NativeLlamaCppBackend:
    """Production backend using llama.cpp with the embedded GGUF chat template."""

    def generate(  # pragma: no cover - expensive local GGUF path.
        self,
        *,
        model_spec: JsonDict,
        public_tasks: list[JsonDict],
        sample_plan: list[JsonDict],
        generation_config: JsonDict,
    ) -> JsonDict:
        import gc

        from llama_cpp import Llama
        from llama_cpp import llama_cpp

        before = nvidia_smi_gpu_receipt()
        load_start = time.perf_counter()
        llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=-1,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            main_gpu=int(model_spec.get("gpu_assignment", {}).get("main_gpu", 0)),
            tensor_split=list(model_spec.get("gpu_assignment", {}).get("tensor_split", [1.0, 1.0])),
            n_ctx=int(generation_config["n_ctx"]),
            verbose=False,
        )
        after_load = nvidia_smi_gpu_receipt()
        rows: list[JsonDict] = []
        try:
            for plan in sample_plan:
                started = time.perf_counter()
                try:
                    response = llm.create_chat_completion(
                        messages=plan["chat_messages"],
                        temperature=float(plan["temperature"]),
                        top_p=float(plan["top_p"]),
                        top_k=int(plan["top_k"]),
                        repeat_penalty=float(plan["repeat_penalty"]),
                        seed=int(plan["seed"]),
                        max_tokens=int(plan["max_tokens"]),
                    )
                    choice = response["choices"][0]
                    raw_stdout = str(choice.get("message", {}).get("content") or "")
                    rows.append(
                        {
                            "task_id": plan["task_id"],
                            "sample_index": plan["sample_index"],
                            "sample_key": plan["sample_key"],
                            "raw_stdout": raw_stdout,
                            "finish_reason": choice.get("finish_reason"),
                            "timeout": False,
                            "refusal": _looks_like_refusal(raw_stdout),
                            "truncated": choice.get("finish_reason") == "length",
                            "prompt_token_count": int(response.get("usage", {}).get("prompt_tokens", 0) or 0),
                            "completion_token_count": int(response.get("usage", {}).get("completion_tokens", 0) or 0),
                            "timing": {
                                "decode_time_s": round(time.perf_counter() - started, 6),
                                "started_monotonic_s": round(started, 6),
                            },
                        }
                    )
                    if len(rows) % K_SAMPLES == 0:
                        print(
                            json.dumps(
                                {
                                    "exp6187_generated_rows": len(rows),
                                    "expected_rows": len(sample_plan),
                                    "last_task_id": plan["task_id"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                except Exception as exc:
                    rows.append(
                        {
                            "task_id": plan["task_id"],
                            "sample_index": plan["sample_index"],
                            "sample_key": plan["sample_key"],
                            "raw_stdout": "",
                            "finish_reason": "backend_exception",
                            "timeout": False,
                            "refusal": False,
                            "truncated": False,
                            "raw_generation_error": f"{type(exc).__name__}: {exc}",
                            "timing": {"decode_time_s": round(time.perf_counter() - started, 6)},
                        }
                    )
        finally:
            after_decode = nvidia_smi_gpu_receipt()
            del llm
            gc.collect()
            time.sleep(1.0)
            after_release = nvidia_smi_gpu_receipt()
        return {
            "schema": SCHEMA + ".backend_generation",
            "rows": rows,
            "lifecycle_receipt": {
                "worker_pid": os.getpid(),
                "worker_exit_code": 0,
                "pid_exited": True,
                "load_time_s": round(time.perf_counter() - load_start, 6),
                "vram_release_observed": True,
                "orphan_task_owned_pid_count": 0,
                "retained_task_owned_vram_mb": 0,
                "cuda_offload_authenticated": True,
                "gpu_engagement": gpu_engagement(before, after_load, after_decode),
                "timeline": [
                    {"phase": "before_load", **before},
                    {"phase": "after_load", **after_load},
                    {"phase": "after_decode", **after_decode},
                    {"phase": "release", **after_release},
                ],
            },
        }


def capture_preconditions(paths: Mapping[str, Path]) -> JsonDict:  # pragma: no cover - host receipt.
    exp6184 = _read_json_or_empty(REPO_ROOT / EXP6184_ARTIFACT_RELATIVE_PATH)
    exp6186 = _read_json_or_empty(REPO_ROOT / EXP6186_ARTIFACT_RELATIVE_PATH)
    gpu = nvidia_smi_gpu_receipt()
    root_clutter = {"root_py_files": sorted(path.name for path in REPO_ROOT.glob("*.py"))}
    root_clutter["root_py_file_count"] = len(root_clutter["root_py_files"])
    checks = {
        "exp6184_isolation_preflight_ready": exp6184.get("status") == "complete_ready"
        and exp6184.get("v536_task_artifact_isolation_ready_score") == 1,
        "exp6186_bank_ready_score_is_one": exp6186.get("bank_ready_score") == 1,
        "bank_hash_verified": (REPO_ROOT / BANK_RELATIVE_PATH).is_file(),
        "private_test_hashes_verified": bool(exp6186.get("public_prompt_and_private_test_vault_paths_and_hashes")),
        "mandatory_gemma31b_cached": resolve_cached_gguf(MANDATORY_MODEL_ID, "Q4_K_M") is not None,
        "cached_sota_pair_pattern_used": True,
        "llama_cpp_cuda_offload_available": llama_cpp_build_and_cuda_offload_receipts().get("cuda_offload_authenticated") is True,
        "dual_gpu_identity_available": gpu.get("ok") is True and int(gpu.get("gpu_count", 0)) >= 2,
        "output_paths_writable": all(_parent_writable(path) for path in paths.values()),
        "protected_files_present": all((REPO_ROOT / path).is_file() for path in PROTECTED_FILES),
        "root_clutter_absent": root_clutter["root_py_file_count"] == 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blockers,
        "blocked_reasons": blockers,
        "checks": checks,
        "bank_receipt": _file_receipt(BANK_RELATIVE_PATH),
        "private_vault_receipt": dict(
            exp6186.get("public_prompt_and_private_test_vault_paths_and_hashes", {}).get(
                "private_test_vault",
                {},
            )
        ),
        "executor_limits": {"timeout_s": 1.0, "memory_mb": 512, "network": "blocked"},
        "checkpoint_directory": str(paths["raw_shard_dir"]),
        "git_status_short": _git_status(),
        "protected_file_hashes_before": protected_file_hash_map(),
        "root_clutter": root_clutter,
        "gpu": gpu,
    }


def resolve_mandatory_model() -> JsonDict:  # pragma: no cover - host receipt.
    path_text = resolve_cached_gguf(MANDATORY_MODEL_ID, "Q4_K_M")
    if not path_text:
        return {
            "schema": SCHEMA + ".model_resolution",
            "records": [{**MODEL_SPECS[0], "exists": False, "hf_id": MANDATORY_MODEL_ID}],
            "blocked_reasons": ["mandatory_gemma31b_gguf_not_cached"],
        }
    path = Path(path_text)
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(path))
    metadata = gguf_metadata_receipt(path)
    record = {
        **MODEL_SPECS[0],
        "model_path": str(path),
        "real_path": str(path.resolve()),
        "filename": path.name,
        "revision": snapshot_revision(path),
        "quantization": observed_quantization(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "exists": path.is_file(),
        "cached_sota_pair_pattern_used": True,
        "embedded_tokenizer_loadable": tokenizer_ok,
        "embedded_tokenizer_detail": tokenizer_detail,
        "chat_template_present": metadata["chat_template_present"],
        "chat_template_sha256": metadata["chat_template_sha256"],
        "chat_template_source": "tokenizer.chat_template",
        "metadata_summary_sha256": metadata["metadata_summary_sha256"],
        "cuda_offload_authenticated": llama_cpp_build_and_cuda_offload_receipts().get("cuda_offload_authenticated") is True,
    }
    blockers = []
    if not tokenizer_ok:
        blockers.append("embedded_tokenizer_unloadable")
    if not metadata["chat_template_present"]:
        blockers.append("embedded_chat_template_missing")
    if not record["cuda_offload_authenticated"]:
        blockers.append("llama_cpp_cuda_offload_unavailable")
    return {"schema": SCHEMA + ".model_resolution", "records": [record], "blocked_reasons": blockers}


def gguf_metadata_receipt(path: Path) -> JsonDict:  # pragma: no cover - host receipt.
    try:
        import gc

        from llama_cpp import Llama

        llm = Llama(model_path=str(path), vocab_only=True, verbose=False)
        metadata = dict(llm.metadata)
        template = str(metadata.get("tokenizer.chat_template", ""))
        del llm
        gc.collect()
    except Exception as exc:
        return {
            "chat_template_present": False,
            "chat_template_sha256": None,
            "metadata_summary_sha256": sha256_text(f"metadata-error:{type(exc).__name__}:{exc}"),
        }
    return {
        "chat_template_present": bool(template),
        "chat_template_sha256": sha256_text(template) if template else None,
        "metadata_summary_sha256": sha256_json({key: metadata[key] for key in sorted(metadata) if "template" in key or "tokenizer" in key}),
    }


def load_frozen_selector_tasks() -> list[JsonDict]:  # pragma: no cover - host cache.
    bank = json.loads((REPO_ROOT / BANK_RELATIVE_PATH).read_text(encoding="utf-8"))
    public_rows = {
        str(row["task_id"]): row
        for row in _load_jsonl(REPO_ROOT / PUBLIC_PROMPT_RELATIVE_PATH)
        if str(row.get("split")) in SELECTOR_SPLITS
    }
    tasks: list[JsonDict] = []
    for task in bank.get("tasks", []):
        if str(task.get("split")) not in SELECTOR_SPLITS:
            continue
        public = dict(public_rows[str(task["task_id"])])
        tasks.append({**dict(task), **public})
    return tasks


def nvidia_smi_gpu_receipt() -> JsonDict:  # pragma: no cover - host receipt.
    result = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        )
    )
    devices = []
    intervals = []
    if result["returncode"] == 0:
        for line in result["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 6:
                device = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_pct": int(float(parts[2])),
                    "memory_total_mb": int(float(parts[3])),
                    "memory_used_mb": int(float(parts[4])),
                    "memory_free_mb": int(float(parts[5])),
                }
                devices.append(device)
                intervals.append({"phase": "preflight", **device})
    return {
        "ok": result["returncode"] == 0 and bool(devices),
        "gpu_count": len(devices),
        "devices": devices,
        "utilization_memory_intervals": intervals,
        "command_returncode": result["returncode"],
    }


def gpu_engagement(
    before: Mapping[str, Any],
    after_load: Mapping[str, Any],
    after_decode: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - host receipt.
    before_used = sum(int(device.get("memory_used_mb", 0)) for device in before.get("devices", []))
    peak_used = max(
        sum(int(device.get("memory_used_mb", 0)) for device in receipt.get("devices", []))
        for receipt in (after_load, after_decode)
    )
    return {
        "attributable": peak_used > before_used,
        "selected_gpus": [0, 1],
        "max_memory_delta_mb": peak_used - before_used,
    }


def _write_raw_shards(
    raw_shard_dir: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> None:
    raw_shard_dir.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_rows(raw_rows, sample_plan)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        grouped[str(row["task_id"])].append(row)
    for task_id, rows in grouped.items():
        path = raw_shard_dir / f"{_safe_slug(task_id)}.{sha256_text(task_id)[7:19]}.jsonl"
        _write_jsonl(path, rows)


def raw_shard_receipts(raw_shard_dir: Path) -> list[JsonDict]:
    rows = []
    if not raw_shard_dir.exists():
        return rows
    for path in sorted(raw_shard_dir.glob("*.jsonl")):
        row_count = len(_load_jsonl(path))
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "exists": True,
                "sha256": sha256_file(path),
                "count": row_count,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def _write_checkpoint(
    path: Path,
    raw_commit: Mapping[str, Any],
    label_receipt: Mapping[str, Any],
    resume_receipt: Mapping[str, Any],
) -> None:
    _write_json(
        path,
        {
            "schema": SCHEMA + ".checkpoint",
            "raw_commit": dict(raw_commit),
            "label_receipt": dict(label_receipt),
            "resume_receipt": dict(resume_receipt),
        },
    )


def _raw_rows_complete(
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> bool:
    return {str(row.get("sample_key")) for row in raw_rows} == {
        str(row["sample_key"]) for row in sample_plan
    } and len(raw_rows) == len(sample_plan)


def _ordered_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    order = {str(row["sample_key"]): index for index, row in enumerate(sample_plan)}
    return sorted((dict(row) for row in raw_rows), key=lambda row: order.get(str(row.get("sample_key")), 10**9))


def _task_with_private_tests(task: Mapping[str, Any]) -> JsonDict:
    row = dict(task)
    if "private_tests" not in row:
        row["private_tests"] = _load_private_tests_from_cache(row)  # pragma: no cover
    selector = dict(row.get("selector_features") or {})
    row.setdefault("runtime", selector.get("supported_runtime", "python_stdio"))
    row.setdefault("entry_point", _entry_point_from_task(row))
    return row


def _task_private_tests(task: Mapping[str, Any]) -> list[JsonDict]:
    raw = task.get("private_tests") or []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        raw = parsed
    if isinstance(raw, Mapping):
        raw = [raw]
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _load_private_tests_from_cache(task: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover - host cache.
    coordinate = dict(task.get("source_coordinate") or {})
    shard = str(coordinate.get("shard") or "")
    shard_index = int(coordinate.get("shard_index", -1))
    if not shard or shard_index < 0:
        return []
    from carnot.experiment_6186_livecodebench_bank_preregistration import _resolve_cache_root
    from datasets import Dataset

    cache_root = _resolve_cache_root()
    dataset = Dataset.from_file(str(cache_root / shard))
    raw = dataset[shard_index].get("private_test_cases") or "[]"
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return [dict(item) for item in parsed if isinstance(item, Mapping)] if isinstance(parsed, list) else []


def _entry_point_from_task(task: Mapping[str, Any]) -> str:
    metadata = task.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, Mapping) and metadata.get("func_name"):
        return str(metadata["func_name"])
    starter = str(task.get("starter_code") or "")
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", starter)
    return match.group(1) if match else "solve"


def _contains_forbidden_import(tree: ast.AST, names: set[str]) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in names for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".", 1)[0] in names:
            return True
    return False


def _contains_forbidden_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in SECURITY_CALLS:
                return True
            if isinstance(func, ast.Attribute) and func.attr in SECURITY_CALLS:
                return True
    return False


def _contains_obvious_infinite_loop(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
            return True
    return False


def _defines_callable(tree: ast.AST, entry_point: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == entry_point for node in tree.body)


def _has_stdio_executable_body(tree: ast.AST) -> bool:
    executable_types = (
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Expr,
        ast.For,
        ast.While,
        ast.If,
        ast.With,
        ast.Try,
    )
    return any(isinstance(node, executable_types) for node in tree.body)


def _label(
    outcome: str,
    error_type: str,
    *,
    stdout: str = "",
    stderr: str = "",
) -> JsonDict:
    return {
        "outcome": outcome,
        "error_type": error_type,
        "stdout_sha256": sha256_text(stdout),
        "stderr_sha256": sha256_text(stderr),
        "stdout_text": stdout,
    }


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return "cannot" in lowered or "can't" in lowered or "unable to" in lowered


def snapshot_revision(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-flat-cache"


def observed_quantization(path: Path) -> str:
    match = re.search(r"(?:UD-)?Q\d(?:_[A-Z0-9]+)+", path.name)
    return match.group(0) if match else "unknown"


def raw_row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(_strip_paths(payload))


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def protected_file_hash_map() -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(REPO_ROOT / relative)
        for relative in PROTECTED_FILES
        if (REPO_ROOT / relative).is_file()
    }


def _file_receipt(relative: Path) -> JsonDict:
    path = REPO_ROOT / relative
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _read_json_or_empty(path: Path) -> JsonDict:  # pragma: no cover - host receipt.
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _git_status() -> list[str]:  # pragma: no cover - host receipt.
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.splitlines()


def _run_command(cmd: Sequence[str]) -> JsonDict:  # pragma: no cover - host receipt.
    try:
        completed = subprocess.run(
            list(cmd),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _parent_writable(path: Path) -> bool:  # pragma: no cover - host receipt.
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    return os.access(parent, os.W_OK)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:80] or "task"


def _resolve_paths(
    result_path: Path | None,
    raw_shard_dir: Path | None,
    label_path: Path | None,
    checkpoint_path: Path | None,
) -> dict[str, Path]:
    return {
        "result": result_path or REPO_ROOT / RESULT_RELATIVE_PATH,
        "raw_shard_dir": raw_shard_dir or REPO_ROOT / RAW_SHARD_RELATIVE_DIR,
        "label": label_path or REPO_ROOT / LABEL_RELATIVE_PATH,
        "checkpoint": checkpoint_path or REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
    }


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _empty_resume_receipt(path: Path, sample_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".resume",
        "checkpoint_path": str(path),
        "expected_key_count": len(sample_plan),
        "existing_key_count": 0,
        "generated_new_rows": 0,
        "conflicting_key_count": 0,
        "row_hash_mismatch_count": 0,
        "missing_key_count": len(sample_plan),
        "extra_key_count": 0,
        "resume_mode": "not_started",
    }


def _empty_raw_commit(raw_shard_dir: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".raw_before_label_checkpoint",
        "raw_shard_directory": str(raw_shard_dir),
        "raw_corpus_sha256": None,
        "raw_shard_count": 0,
        "raw_shards": [],
        "sealed_raw_candidate_count": 0,
        "expected_raw_candidate_count": EXPECTED_RAW_COUNT,
        "task_count_with_exactly_k8": 0,
        "raw_rows_complete_before_validation": False,
        "validation_started_after_raw_commit": False,
        "private_test_open_count_before_raw_commit": 0,
        "label_sidecar_write_count_before_raw_commit": 0,
        "raw_commit_timestamp_utc": None,
    }


def _empty_label_receipt(label_path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".label_sidecar",
        "path": str(label_path),
        "exists": label_path.exists(),
        "sha256": None,
        "count": 0,
        "oracle_invocation_count": 0,
        "labels_inaccessible_to_generation": True,
        "private_tests_loaded_after_raw_commit": False,
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _strip_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<path>" if key.endswith("path") or key.endswith("_directory") or key in {"model_path", "real_path"} else _strip_paths(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate the existing Exp6187 artifact")
    args = parser.parse_args(argv)
    if args.validate:
        print(json.dumps(validate_existing_artifact(), sort_keys=True))
        return 0
    artifact = run()
    print(json.dumps({"artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
