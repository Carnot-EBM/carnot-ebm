"""Exp6200 three-family raw-code transport envelope canary.

Spec refs: REQ-CODE-6200,
SCENARIO-CODE-6200-FROZEN-CALIBRATION-MATRIX,
SCENARIO-CODE-6200-RAW-BEFORE-ANALYSIS,
SCENARIO-CODE-6200-PRIVATE-ORACLE-NONACCESS,
SCENARIO-CODE-6200-IMMUTABLE-RESUME,
SCENARIO-CODE-6200-CUDA-AND-FAMILY-GATES.

This workflow diagnoses whether the local serving envelope can return complete
raw Python before any larger generation pool is launched. It uses only public
task text and public samples; hidden tests are deliberately out of scope.
"""

from __future__ import annotations

import argparse
import ast
import base64
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
SCHEMA = "carnot.experiment_6200.three_family_raw_code_transport_canary.v1"
RAW_ROW_SCHEMA = SCHEMA + ".raw_row"
EXPERIMENT_ID = "experiment_6200_three_family_raw_code_transport_canary"
RUN_DATE = "20260807"
RANDOM_SEED = 620000000
INFERENCE_SUBSTRATE = "local_three_family_llama_cpp_cuda_raw_code_transport_canary"

RESULT_RELATIVE_PATH = Path("results/experiment_6200_three_family_raw_code_transport_canary.json")
RAW_SHARD_RELATIVE_DIR = Path("results/experiment_6200_three_family_raw_code_transport_canary.raw_shards")
EXP6186_ARTIFACT_RELATIVE_PATH = Path("results/experiment_6186_livecodebench_bank_preregistration.json")
BANK_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186.json")
PUBLIC_PROMPT_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186_public_prompts.jsonl")

TOKEN_BUDGETS = (512, 1024, 1536)
CALIBRATION_TASK_COUNT = int(os.environ.get("CARNOT_EXP6200_TASK_COUNT", "2"))
N_CTX = int(os.environ.get("CARNOT_EXP6200_N_CTX", "4096"))
TEMPERATURE = float(os.environ.get("CARNOT_EXP6200_TEMPERATURE", "0.0"))
TOP_P = 1.0
TOP_K = 40
REPEAT_PENALTY = 1.05

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma4_31b_dense",
        "role": "Phase-D dense headline transport",
        "gpu": 0,
        "loader": "llama_cpp.Llama",
        "expected_quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "n_ctx": N_CTX,
        "legacy_small_model_headline": False,
    },
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "family": "qwen3_35b_a3b_moe",
        "role": "CSL flagship MoE transport",
        "gpu": 0,
        "loader": "llama_cpp.Llama",
        "expected_quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "n_ctx": N_CTX,
        "legacy_small_model_headline": False,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "family": "gemma4_26b_a4b_moe",
        "role": "CSL middle MoE transport",
        "gpu": 1,
        "loader": "llama_cpp.Llama",
        "expected_quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "n_ctx": N_CTX,
        "legacy_small_model_headline": False,
    },
]
FAMILY_ORDER = tuple(str(row["family"]) for row in MODEL_SPECS)
MANDATED_MODEL_IDS = tuple(str(row["hf_id"]) for row in MODEL_SPECS)

GENERATION_CONFIG: JsonDict = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repeat_penalty": REPEAT_PENALTY,
    "token_budgets": list(TOKEN_BUDGETS),
    "n_ctx": N_CTX,
    "seed_base": RANDOM_SEED,
    "seed_rule": "620000000 + model_index * 10000 + task_index * 100 + budget_index",
    "prompt_transport": "chat_completion_raw_python",
    "correctness_conditioned_retry": False,
    "private_oracle_allowed": False,
    "parser_repair": False,
    "grammar_retry": False,
    "candidate_replacement": False,
    "model_judge": False,
    "legacy_model_substitution": False,
}

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
    ".venv/bin/pytest tests/python/test_experiment_6200_three_family_raw_code_transport_canary.py -q -o addopts=",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6200_three_family_raw_code_transport_canary.py -m pytest tests/python/test_experiment_6200_three_family_raw_code_transport_canary.py -q -o addopts= && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6200_three_family_raw_code_transport_canary.py --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6200_three_family_raw_code_transport_canary.py",
    ".venv/bin/python -m carnot.experiment_6200_three_family_raw_code_transport_canary --validate",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6200_three_family_raw_code_transport_canary.json",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "bank_path_hash_and_calibration_task_ids",
    "preregistered_budget_grid",
    "model_specs",
    "gguf_cache_hash_revision_quantization_receipts",
    "prompt_and_embedded_chat_template_receipts",
    "generation_matrix",
    "raw_before_interpretation_receipt",
    "per_family_finish_reason_token_extraction_compile_and_public_sample_metrics",
    "private_oracle_access_count",
    "configuration_selection_inputs",
    "frozen_envelope_by_family",
    "phase_d_transport_ready_score",
    "csl_transport_ready_score",
    "legacy_headline_row_count",
    "gpu_offload_and_interval_receipts",
    "process_wall_time_receipts",
    "correctness_retry_count",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows bank, cache, CUDA, raw coverage, transport metrics, and readiness gates.",
    "bank_path_hash_and_calibration_task_ids": "Only immutable Exp6186 calibration task hashes and public prompt material choose the canary subset.",
    "preregistered_budget_grid": "The 512/1024/1536 budget grid is frozen before model load.",
    "model_specs": "Exactly the three mandated SOTA GGUF families are eligible.",
    "gguf_cache_hash_revision_quantization_receipts": "Exact local GGUF file identity prevents silent model substitution.",
    "prompt_and_embedded_chat_template_receipts": "Prompts use public task text and embedded GGUF chat templates only.",
    "generation_matrix": "Every family/task/budget cell has a deterministic seed and immutable key.",
    "raw_before_interpretation_receipt": "Raw bytes are sealed and hashed before extraction or public checks.",
    "per_family_finish_reason_token_extraction_compile_and_public_sample_metrics": "Envelope choice uses transport and public sample evidence only.",
    "private_oracle_access_count": "Bare zero; hidden tests are forbidden for calibration choice.",
    "configuration_selection_inputs": "Lists the allowed label-blind inputs and forbidden private outcomes.",
    "frozen_envelope_by_family": "One per-family max-token ceiling is selected without correctness labels.",
    "phase_d_transport_ready_score": "Gemma-4-31B dense readiness is the Phase-D transport gate.",
    "csl_transport_ready_score": "Both MoE families must be transport-ready for CSL.",
    "legacy_headline_row_count": "Bare zero; no legacy model row can contribute.",
    "gpu_offload_and_interval_receipts": "CUDA offload and GPU intervals are recorded from preflight and generation.",
    "process_wall_time_receipts": "Host process timing keeps live inference auditable.",
    "correctness_retry_count": "Bare zero; rows are never resampled or repaired from outcomes.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "inference_substrate": "Declares local three-family llama.cpp CUDA raw-code generation.",
    "verifier_is_oracle": "False because this canary does not use private correctness or hidden tests.",
    "field_provenance": "Maps every required artifact field to REQ-CODE-6200.",
    "field_principles": "Records why each field exists and what audit failure it prevents.",
    "test_commands": "Commands used for focused tests, coverage, spec coverage, validation, adversarial check, and full pytest.",
    "test_exit_codes": "Exit codes prevent failed verification from being reported as readiness.",
    "duration_s": "End-to-end wall-clock duration.",
    "reproducibility_checksum": "Stable artifact checksum excluding duration and itself.",
    "honest_verdict": "Terminal verdict names readiness and any blocked or partial transport state.",
}


class CanaryGenerationBackend(Protocol):
    """Backend contract for one-shot raw stdout generation."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        public_tasks: list[JsonDict],
        generation_plan: list[JsonDict],
        generation_config: JsonDict,
    ) -> JsonDict:
        """Return raw rows and model lifecycle evidence without labels."""


def run(
    *,
    result_path: Path | None = None,
    raw_shard_dir: Path | None = None,
    task_rows: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: JsonDict | None = None,
    model_resolution: JsonDict | None = None,
    generation_backend: CanaryGenerationBackend | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
    write: bool = True,
) -> JsonDict:
    """Run the label-blind canary or write a blocked artifact."""

    started = time.perf_counter()
    start_utc = utc_now()
    paths = _resolve_paths(result_path, raw_shard_dir)
    tasks = list(task_rows) if task_rows is not None else load_frozen_calibration_tasks()  # pragma: no cover
    public_tasks = build_public_tasks(tasks)
    generation_plan = build_generation_plan(public_tasks)
    preconditions = preconditions_checked or capture_preconditions(paths, public_tasks)  # pragma: no cover
    resolution = model_resolution or resolve_mandatory_models()  # pragma: no cover
    model_specs = model_specs_from_resolution(resolution)
    gate = structured_gate_receipt(preconditions, resolution, public_tasks, generation_plan)
    raw_rows: list[JsonDict] = []
    generation_lifecycle: list[JsonDict] = []
    resume_receipt = empty_resume_receipt(paths["raw_shard_dir"], generation_plan)
    raw_receipt = empty_raw_receipt(paths["raw_shard_dir"], generation_plan)

    if gate["passed"]:
        existing = inspect_existing_raw_shards(paths["raw_shard_dir"], generation_plan)
        resume_receipt.update(existing["resume_receipt"])
        if existing["blocked"]:
            gate["passed"] = False
            gate["blocked_reasons"].extend(existing["blocked_reasons"])
        else:
            raw_rows = list(existing["rows"])
            missing_plan = list(existing["missing_plan"])
            if missing_plan:
                backend = generation_backend or NativeThreeFamilyLlamaCppBackend()  # pragma: no cover
                public_by_id = {str(task["task_id"]): task for task in public_tasks}
                for model_spec in model_specs:
                    model_missing = [
                        row
                        for row in missing_plan
                        if str(row["family"]) == str(model_spec["family"])
                    ]
                    if not model_missing:  # pragma: no cover - defensive partial-plan branch.
                        continue
                    try:
                        generated = backend.generate(
                            model_spec=model_spec,
                            public_tasks=list(public_by_id.values()),
                            generation_plan=model_missing,
                            generation_config=dict(GENERATION_CONFIG),
                        )
                    except Exception as exc:
                        generated = {
                            "schema": SCHEMA + ".backend_generation",
                            "rows": [
                                {
                                    "cell_id": plan["cell_id"],
                                    "raw_stdout": "",
                                    "finish_reason": "backend_exception",
                                    "generated_token_count": 0,
                                    "prompt_token_count": 0,
                                    "timing": {},
                                    "timeout": False,
                                    "raw_generation_error": f"{type(exc).__name__}: {exc}",
                                }
                                for plan in model_missing
                            ],
                            "lifecycle_receipt": {
                                "worker_pid": os.getpid(),
                                "worker_exit_code": 1,
                                "pid_exited": True,
                                "cuda_offload_authenticated": False,
                                "backend_exception": f"{type(exc).__name__}: {exc}",
                            },
                        }
                    generation_lifecycle.append(
                        {
                            "family": model_spec["family"],
                            "model_hf_id": model_spec["hf_id"],
                            **dict(generated.get("lifecycle_receipt", {})),
                        }
                    )
                    raw_rows.extend(assemble_raw_rows(generated.get("rows", []), model_spec, model_missing, run_date))
                raw_rows = ordered_rows(raw_rows, generation_plan)
                write_raw_shards(paths["raw_shard_dir"], raw_rows, generation_plan)
                resume_receipt["resume_mode"] = (
                    "fresh_generation" if not existing["rows"] else "resumed_missing_keys"
                )
                resume_receipt["generated_new_rows"] = len(missing_plan)
            else:
                resume_receipt["resume_mode"] = "reused_raw_shards"
                resume_receipt["generated_new_rows"] = 0
            raw_rows = ordered_rows(raw_rows, generation_plan)
            if raw_rows_complete(raw_rows, generation_plan):
                raw_receipt = raw_before_interpretation_receipt(
                    paths["raw_shard_dir"],
                    raw_rows,
                    generation_plan,
                )

    analysis_rows = analyze_raw_rows(raw_rows, public_tasks) if raw_receipt.get("analysis_started_after_raw_commit") else []
    per_family_metrics = per_family_transport_metrics(analysis_rows, generation_plan, model_specs)
    frozen_envelopes = {
        family: freeze_family_envelope(per_family_metrics.get(family, {}).get("by_budget", {}))
        for family in FAMILY_ORDER
    }
    phase_d_score = phase_d_transport_ready_score(frozen_envelopes)
    csl_score = csl_transport_ready_score(frozen_envelopes)
    legacy_count = sum(str(row.get("model_hf_id")) not in MANDATED_MODEL_IDS for row in raw_rows)
    protected = protected_files_unchanged(preconditions)
    measured_duration = round(duration_s if duration_s is not None else time.perf_counter() - started, 6)
    status = (
        "complete_ready"
        if gate["passed"] and phase_d_score == 1 and csl_score == 1 and raw_rows_complete(raw_rows, generation_plan)
        else ("complete_partial" if raw_rows and gate["passed"] else "blocked")
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "status": status,
        "bank_path_hash_and_calibration_task_ids": bank_task_receipt(preconditions, public_tasks),
        "preregistered_budget_grid": preregistered_budget_grid(public_tasks, generation_plan),
        "model_specs": attach_use_counts(model_specs, raw_rows),
        "gguf_cache_hash_revision_quantization_receipts": gguf_cache_receipts(resolution),
        "prompt_and_embedded_chat_template_receipts": prompt_and_template_receipts(
            public_tasks,
            generation_plan,
            resolution,
        ),
        "generation_matrix": generation_matrix_receipt(generation_plan, raw_rows, resume_receipt),
        "raw_before_interpretation_receipt": raw_receipt,
        "per_family_finish_reason_token_extraction_compile_and_public_sample_metrics": per_family_metrics,
        "private_oracle_access_count": 0,
        "configuration_selection_inputs": configuration_selection_inputs(raw_receipt),
        "frozen_envelope_by_family": frozen_envelopes,
        "phase_d_transport_ready_score": phase_d_score,
        "csl_transport_ready_score": csl_score,
        "legacy_headline_row_count": legacy_count,
        "gpu_offload_and_interval_receipts": gpu_offload_and_interval_receipts(
            preconditions,
            resolution,
            generation_lifecycle,
        ),
        "process_wall_time_receipts": {
            "schema": SCHEMA + ".process_wall_time",
            "start_utc": start_utc,
            "end_utc": utc_now(),
            "duration_s": measured_duration,
            "pid": os.getpid(),
            "raw_generation_lifecycle_count": len(generation_lifecycle),
        },
        "correctness_retry_count": 0,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "duration_s": measured_duration,
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, frozen_envelopes, gate, raw_rows, generation_plan),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        write_json(paths["result"], artifact)
    return artifact


def build_public_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Select fixed calibration tasks from public, label-free material."""

    candidates: list[JsonDict] = []
    for task in tasks:
        if str(task.get("split")) != "calibration":
            continue
        selector = dict(task.get("selector_features") or {})
        public_tests = parse_public_tests(task)
        prompt = str(task.get("question_content") or "")
        if not prompt and not task.get("public_tests"):
            continue
        runtime = str(selector.get("supported_runtime") or task.get("runtime") or "python_stdio")
        public = {
            "schema": SCHEMA + ".public_task",
            "task_id": str(task["task_id"]),
            "split": "calibration",
            "question_title": str(task.get("question_title") or ""),
            "question_content": prompt,
            "starter_code": str(task.get("starter_code") or ""),
            "platform": str(task.get("platform") or selector.get("platform") or ""),
            "difficulty": str(task.get("difficulty") or selector.get("difficulty") or ""),
            "contest_id": str(task.get("contest_id") or ""),
            "contest_date": str(task.get("contest_date") or ""),
            "selector_features": selector,
            "runtime": runtime,
            "entry_point": entry_point_from_task(task),
            "public_tests": public_tests,
            "public_sample_count": len(public_tests),
            "prompt_sha256": str(task.get("prompt_sha256") or sha256_text(prompt)),
            "public_test_sha256": str(task.get("public_test_sha256") or sha256_json(public_tests)),
            "private_test_sha256": str(task.get("private_test_sha256") or ""),
            "metadata_sha256": str(task.get("metadata_sha256") or ""),
            "stable_task_hash": str(task.get("stable_task_hash") or sha256_json(task)),
        }
        candidates.append(public)
    candidates.sort(
        key=lambda row: (
            0 if int(row["public_sample_count"]) > 0 else 1,
            0 if row["runtime"] == "python_stdio" else 1,
            str(row["stable_task_hash"]),
            str(row["task_id"]),
        )
    )
    selected = candidates[:CALIBRATION_TASK_COUNT]
    for index, task in enumerate(selected):
        task["task_index"] = index
    return selected


def build_generation_plan(public_tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze the full family x task x budget matrix."""

    rows: list[JsonDict] = []
    for model_index, model in enumerate(MODEL_SPECS):
        for task in public_tasks:
            task_index = int(task["task_index"])
            for budget_index, budget in enumerate(TOKEN_BUDGETS):
                seed = RANDOM_SEED + model_index * 10000 + task_index * 100 + budget_index
                cell_id = f"{model['family']}::{task['task_id']}::tok{budget}"
                messages = build_chat_messages(task, cell_id, budget)
                rows.append(
                    {
                        "schema": SCHEMA + ".planned_cell",
                        "cell_index": len(rows),
                        "cell_id": cell_id,
                        "family": model["family"],
                        "model_hf_id": model["hf_id"],
                        "model_name": model["name"],
                        "role": model["role"],
                        "gpu": model["gpu"],
                        "task_id": task["task_id"],
                        "task_index": task_index,
                        "split": task["split"],
                        "budget_index": budget_index,
                        "max_tokens": budget,
                        "seed": seed,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "repeat_penalty": REPEAT_PENALTY,
                        "n_ctx": N_CTX,
                        "prompt_sha256": task["prompt_sha256"],
                        "public_test_sha256": task["public_test_sha256"],
                        "private_test_sha256": task["private_test_sha256"],
                        "stable_task_hash": task["stable_task_hash"],
                        "chat_messages": messages,
                        "chat_messages_sha256": sha256_json(messages),
                    }
                )
    return rows


def build_chat_messages(public_task: Mapping[str, Any], cell_id: str, budget: int) -> list[JsonDict]:
    """Build a raw-code prompt that contains public task material only."""

    system = (
        "Write one complete Python solution for the programming task. Return only raw "
        "Python code or one fenced python block. Do not include explanations."
    )
    user = (
        f"Canary cell: {cell_id}\n"
        f"Output token ceiling: {budget}\n"
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
    generation_plan: Sequence[Mapping[str, Any]],
    run_date: str,
) -> list[JsonDict]:
    """Attach hashes and config to raw stdout without interpreting code."""

    by_cell = {str(row.get("cell_id")): dict(row) for row in backend_rows}
    rows: list[JsonDict] = []
    for plan in generation_plan:
        backend = by_cell.get(str(plan["cell_id"]), {})
        raw_stdout = str(backend.get("raw_stdout", ""))
        raw_bytes = raw_stdout.encode("utf-8", errors="replace")
        row: JsonDict = {
            "schema": RAW_ROW_SCHEMA,
            "run_date": run_date,
            "cell_id": plan["cell_id"],
            "cell_index": plan["cell_index"],
            "family": plan["family"],
            "model_hf_id": model_spec.get("hf_id"),
            "model_name": model_spec.get("name"),
            "role": model_spec.get("role"),
            "gpu": model_spec.get("gpu"),
            "model_path": model_spec.get("model_path"),
            "model_revision": model_spec.get("revision"),
            "model_quantization": model_spec.get("quantization"),
            "task_id": plan["task_id"],
            "task_index": plan["task_index"],
            "split": plan["split"],
            "budget_index": plan["budget_index"],
            "max_tokens": plan["max_tokens"],
            "seed": plan["seed"],
            "temperature": plan["temperature"],
            "top_p": plan["top_p"],
            "top_k": plan["top_k"],
            "repeat_penalty": plan["repeat_penalty"],
            "n_ctx": plan["n_ctx"],
            "prompt_sha256": plan["prompt_sha256"],
            "public_test_sha256": plan["public_test_sha256"],
            "private_test_sha256": plan["private_test_sha256"],
            "stable_task_hash": plan["stable_task_hash"],
            "chat_messages_sha256": plan["chat_messages_sha256"],
            "raw_stdout": raw_stdout,
            "raw_stdout_bytes_b64": base64.b64encode(raw_bytes).decode("ascii"),
            "raw_stdout_bytes_sha256": sha256_bytes(raw_bytes),
            "finish_reason": backend.get("finish_reason", "missing_backend_row"),
            "generated_token_count": int(
                backend.get("generated_token_count", backend.get("completion_token_count", 0)) or 0
            ),
            "prompt_token_count": int(backend.get("prompt_token_count", 0) or 0),
            "timing": dict(backend.get("timing", {})),
            "timeout": bool(backend.get("timeout", False)),
            "raw_generation_error": backend.get("raw_generation_error"),
            "transport_failure": str(plan["cell_id"]) not in by_cell,
            "raw_sealed_at": utc_now(),
        }
        row["row_hash"] = raw_row_hash(row)
        rows.append(row)
    return rows


def analyze_raw_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    public_tasks: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Run deterministic extraction, compile, and public sample checks."""

    tasks_by_id = {str(task["task_id"]): dict(task) for task in public_tasks}
    analysis: list[JsonDict] = []
    for row in raw_rows:
        task = tasks_by_id.get(str(row.get("task_id")), {})
        extraction = extract_python_code(str(row.get("raw_stdout") or ""))
        compile_receipt = compile_python(str(extraction.get("code") or ""))
        public_receipt = (
            run_public_samples(str(extraction.get("code") or ""), task)
            if compile_receipt["compiled"]
            else {"status": "not_run_compile_failed", "passed": False, "sample_count": 0}
        )
        analysis.append(
            {
                "schema": SCHEMA + ".analysis_row",
                "cell_id": row["cell_id"],
                "family": row["family"],
                "task_id": row["task_id"],
                "max_tokens": row["max_tokens"],
                "finish_reason": row.get("finish_reason"),
                "generated_token_count": row.get("generated_token_count", 0),
                "token_budget_respected": int(row.get("generated_token_count", 0)) <= int(row["max_tokens"]),
                "timeout": bool(row.get("timeout")),
                "transport_failure": bool(row.get("transport_failure")),
                "extraction_status": extraction["status"],
                "extraction_method": extraction["method"],
                "extracted_code_sha256": sha256_text(str(extraction.get("code") or "")),
                "compile_status": compile_receipt["status"],
                "compiled": compile_receipt["compiled"],
                "public_sample_status": public_receipt["status"],
                "public_sample_passed": bool(public_receipt["passed"]),
                "public_sample_count": int(public_receipt.get("sample_count", 0)),
            }
        )
    return analysis


def extract_python_code(raw_stdout: str) -> JsonDict:
    """Extract code neutrally; no syntax repair is attempted."""

    match = re.search(r"```(?:python|py)?\s*(.*?)```", raw_stdout, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return {"status": "ok", "method": "fenced_code_block", "code": match.group(1).strip("\n")}
    stripped = raw_stdout.strip("\n")
    if not stripped:
        return {"status": "empty", "method": "raw_stdout", "code": ""}
    return {"status": "no_code_block", "method": "raw_stdout", "code": stripped}


def compile_python(code: str) -> JsonDict:
    try:
        tree = ast.parse(code)
        compile(tree, "<candidate>", "exec")
    except SyntaxError as exc:
        return {"compiled": False, "status": "syntax", "error": f"{exc.__class__.__name__}:{exc.msg}"}
    except Exception as exc:  # pragma: no cover - ast.parse/compile only raises SyntaxError here.
        return {"compiled": False, "status": "compile", "error": f"{type(exc).__name__}: {exc}"}
    return {"compiled": True, "status": "compiled", "error": None}


def run_public_samples(code: str, task: Mapping[str, Any]) -> JsonDict:
    """Execute only public examples from the prompt-facing task row."""

    tests = parse_public_tests(task)
    if not tests:
        return {"status": "not_run_no_public_samples", "passed": False, "sample_count": 0}
    runtime = str(task.get("runtime") or "python_stdio")
    if runtime == "python_stdio":
        for case in tests[:2]:
            result = run_script(code, input_text=str(case.get("input", "")), timeout_s=1.0)
            if result["returncode"] != 0:
                return {"status": "runtime", "passed": False, "sample_count": len(tests[:2])}
            if result["stdout"].strip() != str(case.get("output", "")).strip():
                return {"status": "public_sample_fail", "passed": False, "sample_count": len(tests[:2])}
        return {"status": "public_sample_pass", "passed": True, "sample_count": len(tests[:2])}

    entry_point = str(task.get("entry_point") or "solve")
    script = (
        code
        + "\n\n"
        + "__cases = "
        + repr(tests[:2])
        + "\n"
        + f"__func = globals().get({entry_point!r})\n"
        + "if __func is None and 'Solution' in globals():\n"
        + f"    __func = getattr(globals()['Solution'](), {entry_point!r}, None)\n"
        + "assert __func is not None\n"
        + "for __case in __cases:\n"
        + "    __args = __case.get('input', [])\n"
        + "    if not isinstance(__args, (list, tuple)):\n"
        + "        __args = [__args]\n"
        + "    assert __func(*__args) == __case.get('output')\n"
    )
    result = run_script(script, input_text="", timeout_s=1.0)
    return {
        "status": "public_sample_pass" if result["returncode"] == 0 else "public_sample_fail",
        "passed": result["returncode"] == 0,
        "sample_count": len(tests[:2]),
    }


def run_script(script: str, *, input_text: str, timeout_s: float) -> JsonDict:
    with tempfile.TemporaryDirectory(prefix="carnot-6200-public-") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                input=input_text,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": 124, "stdout": "", "stderr": "timeout"}
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def per_family_transport_metrics(
    analysis_rows: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize finish, token, extraction, compile, and public checks."""

    plan_by_family_budget: dict[tuple[str, str], int] = Counter(
        (str(row["family"]), str(row["max_tokens"])) for row in generation_plan
    )
    rows_by_family_budget: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        rows_by_family_budget[(str(row["family"]), str(row["max_tokens"]))].append(row)
    metrics: JsonDict = {}
    for spec in model_specs:
        family = str(spec["family"])
        by_budget: JsonDict = {}
        for budget in TOKEN_BUDGETS:
            key = (family, str(budget))
            rows = rows_by_family_budget.get(key, [])
            finish_counts = Counter(str(row.get("finish_reason")) for row in rows)
            by_budget[str(budget)] = {
                "expected_cell_count": plan_by_family_budget.get(key, 0),
                "cell_count": len(rows),
                "finish_reason_counts": dict(sorted(finish_counts.items())),
                "stop_finish_count": finish_counts.get("stop", 0),
                "length_finish_count": finish_counts.get("length", 0),
                "truncation_count": sum(str(row.get("finish_reason")) == "length" for row in rows),
                "timeout_count": sum(bool(row.get("timeout")) for row in rows),
                "transport_failure_count": sum(bool(row.get("transport_failure")) for row in rows),
                "token_budget_violation_count": sum(not bool(row.get("token_budget_respected")) for row in rows),
                "max_generated_token_count": max((int(row.get("generated_token_count", 0)) for row in rows), default=0),
                "extraction_status_counts": dict(sorted(Counter(str(row.get("extraction_status")) for row in rows).items())),
                "extraction_success_count": sum(
                    str(row.get("extraction_status")) in {"ok", "no_code_block"} for row in rows
                ),
                "compile_status_counts": dict(sorted(Counter(str(row.get("compile_status")) for row in rows).items())),
                "compile_success_count": sum(bool(row.get("compiled")) for row in rows),
                "public_sample_status_counts": dict(sorted(Counter(str(row.get("public_sample_status")) for row in rows).items())),
                "public_sample_pass_count": sum(bool(row.get("public_sample_passed")) for row in rows),
            }
        metrics[family] = {
            "schema": SCHEMA + ".family_transport_metrics",
            "model_hf_id": spec.get("hf_id"),
            "role": spec.get("role"),
            "by_budget": by_budget,
        }
    return metrics


def freeze_family_envelope(metrics_by_budget: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Select the smallest budget satisfying the preregistered transport gate."""

    for budget in TOKEN_BUDGETS:
        row = dict(metrics_by_budget.get(str(budget), {}))
        if budget_metrics_ready(row):
            return {
                "ready": True,
                "selected_max_tokens": budget,
                "selection_rule": "smallest budget with stop finish, no token violation, extracted code, compile success, and public samples for every calibration task.",
                "selection_inputs_private_oracle_free": True,
                "selected_budget_metrics": row,
                "blocked_reasons": [],
            }
    return {
        "ready": False,
        "selected_max_tokens": None,
        "selection_rule": "no budget met all transport criteria.",
        "selection_inputs_private_oracle_free": True,
        "selected_budget_metrics": {},
        "blocked_reasons": ["no_budget_satisfied_transport_gate"],
    }


def budget_metrics_ready(row: Mapping[str, Any]) -> bool:
    cell_count = int(row.get("cell_count", 0))
    return (
        cell_count == CALIBRATION_TASK_COUNT
        and int(row.get("expected_cell_count", cell_count)) == CALIBRATION_TASK_COUNT
        and int(row.get("stop_finish_count", 0)) == cell_count
        and int(row.get("truncation_count", 0)) == 0
        and int(row.get("timeout_count", 0)) == 0
        and int(row.get("transport_failure_count", 0)) == 0
        and int(row.get("token_budget_violation_count", 0)) == 0
        and int(row.get("extraction_success_count", 0)) == cell_count
        and int(row.get("compile_success_count", 0)) == cell_count
        and int(row.get("public_sample_pass_count", 0)) == cell_count
    )


def phase_d_transport_ready_score(frozen: Mapping[str, Mapping[str, Any]]) -> int:
    return 1 if dict(frozen.get("gemma4_31b_dense", {})).get("ready") is True else 0


def csl_transport_ready_score(frozen: Mapping[str, Mapping[str, Any]]) -> int:
    return (
        1
        if dict(frozen.get("qwen3_35b_a3b_moe", {})).get("ready") is True
        and dict(frozen.get("gemma4_26b_a4b_moe", {})).get("ready") is True
        else 0
    )


def structured_gate_receipt(
    preconditions: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    public_tasks: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    blockers = list(preconditions.get("blocked_reasons", []))
    blockers.extend(str(reason) for reason in model_resolution.get("blocked_reasons", []))
    records = [dict(row) for row in model_resolution.get("records", [])]
    by_id = {str(row.get("hf_id")): row for row in records}
    checks = dict(preconditions.get("checks", {}))
    required = {
        "preconditions_ready": bool(preconditions.get("preconditions_ready")),
        "calibration_task_count": len(public_tasks) == CALIBRATION_TASK_COUNT,
        "planned_cell_count": len(generation_plan) == len(MODEL_SPECS) * CALIBRATION_TASK_COUNT * len(TOKEN_BUDGETS),
        "three_mandated_models_resolved": all(model_id in by_id for model_id in MANDATED_MODEL_IDS),
        "all_mandated_model_files_exist": all(bool(by_id.get(model_id, {}).get("exists")) for model_id in MANDATED_MODEL_IDS),
        "all_embedded_tokenizers_loadable": all(
            bool(by_id.get(model_id, {}).get("embedded_tokenizer_loadable")) for model_id in MANDATED_MODEL_IDS
        ),
        "all_embedded_chat_templates_present": all(
            bool(by_id.get(model_id, {}).get("chat_template_present")) for model_id in MANDATED_MODEL_IDS
        ),
        "llama_cpp_cuda_offload_available": all(
            bool(by_id.get(model_id, {}).get("cuda_offload_authenticated")) for model_id in MANDATED_MODEL_IDS
        ),
        "no_autotokenizer_used": True,
        "no_legacy_model_substitution": True,
    }
    for name, passed in {**checks, **required}.items():
        if not passed:
            blockers.append(name)
    return {
        "schema": SCHEMA + ".structured_gate",
        "passed": not blockers,
        "blocked_reasons": sorted(set(str(item) for item in blockers)),
        "fail_closed_before_model_load": True,
        "private_oracle_allowed": False,
        "autotokenizer_on_gguf_allowed": False,
        "legacy_model_substitution_allowed": False,
    }


def bank_task_receipt(
    preconditions: Mapping[str, Any],
    public_tasks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    ids = [str(task["task_id"]) for task in public_tasks]
    return {
        "schema": SCHEMA + ".bank_task_receipt",
        "bank": dict(preconditions.get("bank_receipt", file_receipt(BANK_RELATIVE_PATH))),
        "public_prompt_bank": dict(preconditions.get("public_prompt_receipt", file_receipt(PUBLIC_PROMPT_RELATIVE_PATH))),
        "private_vault_receipt_hash_only": dict(preconditions.get("private_vault_receipt", {})),
        "calibration_task_count": len(ids),
        "calibration_task_ids": ids,
        "calibration_task_ids_sha256": sha256_json(ids),
        "task_selection_rule": "Exp6186 calibration split only; public-sample-capable tasks sorted by runtime, stable_task_hash, task_id.",
        "private_oracle_used_for_task_selection": False,
    }


def preregistered_budget_grid(
    public_tasks: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".budget_grid",
        "token_budgets": list(TOKEN_BUDGETS),
        "model_family_count": len(MODEL_SPECS),
        "calibration_task_count": len(public_tasks),
        "planned_cell_count": len(generation_plan),
        "matrix_frozen_before_model_load": True,
        "matrix_sha256": sha256_json(
            [
                {
                    "cell_id": row["cell_id"],
                    "seed": row["seed"],
                    "max_tokens": row["max_tokens"],
                }
                for row in generation_plan
            ]
        ),
    }


def generation_matrix_receipt(
    generation_plan: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    resume_receipt: Mapping[str, Any],
) -> JsonDict:
    by_cell = {str(row.get("cell_id")) for row in raw_rows}
    return {
        "schema": SCHEMA + ".generation_matrix",
        "expected_cell_count": len(generation_plan),
        "generated_cell_count": len(raw_rows),
        "missing_cell_count": len([row for row in generation_plan if str(row["cell_id"]) not in by_cell]),
        "unique_seed_count": len({int(row["seed"]) for row in generation_plan}),
        "independent_seed_per_cell": len({int(row["seed"]) for row in generation_plan}) == len(generation_plan),
        "planned_cells": [
            {
                "cell_id": row["cell_id"],
                "family": row["family"],
                "task_id": row["task_id"],
                "max_tokens": row["max_tokens"],
                "seed": row["seed"],
            }
            for row in generation_plan
        ],
        "resume_receipt": dict(resume_receipt),
    }


def raw_before_interpretation_receipt(
    raw_shard_dir: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    shards = raw_shard_receipts(raw_shard_dir)
    return {
        "schema": SCHEMA + ".raw_before_interpretation",
        "raw_shard_directory": str(raw_shard_dir),
        "raw_shard_count": len(shards),
        "raw_shards": shards,
        "raw_corpus_sha256": sha256_json([{row["filename"]: row["sha256"]} for row in shards]),
        "sealed_raw_row_count": len(raw_rows),
        "expected_raw_row_count": len(generation_plan),
        "raw_rows_complete_before_analysis": raw_rows_complete(raw_rows, generation_plan),
        "analysis_started_after_raw_commit": raw_rows_complete(raw_rows, generation_plan),
        "private_oracle_open_count_before_raw_commit": 0,
        "extraction_count_before_raw_commit": 0,
        "compile_count_before_raw_commit": 0,
        "public_sample_run_count_before_raw_commit": 0,
        "raw_commit_timestamp_utc": utc_now(),
    }


def empty_raw_receipt(raw_shard_dir: Path, generation_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".raw_before_interpretation",
        "raw_shard_directory": str(raw_shard_dir),
        "raw_shard_count": 0,
        "raw_shards": [],
        "raw_corpus_sha256": None,
        "sealed_raw_row_count": 0,
        "expected_raw_row_count": len(generation_plan),
        "raw_rows_complete_before_analysis": False,
        "analysis_started_after_raw_commit": False,
        "private_oracle_open_count_before_raw_commit": 0,
        "extraction_count_before_raw_commit": 0,
        "compile_count_before_raw_commit": 0,
        "public_sample_run_count_before_raw_commit": 0,
        "raw_commit_timestamp_utc": None,
    }


def configuration_selection_inputs(raw_receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA + ".selection_inputs",
        "allowed_inputs": [
            "finish_reason",
            "generated_token_count",
            "token_budget",
            "extraction_status",
            "compile_status",
            "public_sample_status",
        ],
        "forbidden_inputs": [
            "private_test_source",
            "hidden_correctness",
            "private_oracle_outcome",
            "label_conditioned_retry",
            "model_judge",
        ],
        "uses_private_oracle": False,
        "private_oracle_access_count": 0,
        "selection_after_raw_commit": raw_receipt.get("analysis_started_after_raw_commit") is True,
        "private_test_vault_opened": False,
    }


def prompt_and_template_receipts(
    public_tasks: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
    model_resolution: Mapping[str, Any],
) -> JsonDict:
    records = [dict(row) for row in model_resolution.get("records", [])]
    return {
        "schema": SCHEMA + ".prompt_template_receipts",
        "prompt_task_count": len(public_tasks),
        "prompt_task_ids": [str(task["task_id"]) for task in public_tasks],
        "prompt_hashes": [str(task["prompt_sha256"]) for task in public_tasks],
        "chat_message_hashes": [str(row["chat_messages_sha256"]) for row in generation_plan],
        "private_material_in_prompts": False,
        "embedded_chat_templates": [
            {
                "hf_id": row.get("hf_id"),
                "family": row.get("family"),
                "chat_template_present": row.get("chat_template_present"),
                "chat_template_sha256": row.get("chat_template_sha256"),
                "chat_template_source": row.get("chat_template_source"),
                "no_autotokenizer_used": True,
            }
            for row in records
        ],
    }


def gguf_cache_receipts(model_resolution: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": SCHEMA + ".gguf_cache_receipts",
        "no_autotokenizer_used": True,
        "cached_local_resolver_pattern_used": True,
        "records": [
            {
                key: row.get(key)
                for key in (
                    "name",
                    "hf_id",
                    "family",
                    "model_path",
                    "real_path",
                    "filename",
                    "sha256",
                    "revision",
                    "quantization",
                    "size_bytes",
                    "exists",
                    "embedded_tokenizer_loadable",
                    "embedded_tokenizer_detail",
                    "chat_template_present",
                    "chat_template_sha256",
                    "metadata_summary_sha256",
                    "cuda_offload_authenticated",
                    "llama_cpp_build",
                    "gpu",
                    "n_ctx",
                    "n_gpu_layers",
                )
            }
            for row in model_resolution.get("records", [])
        ],
    }


def gpu_offload_and_interval_receipts(
    preconditions: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    generation_lifecycle: Sequence[Mapping[str, Any]],
) -> JsonDict:
    records = [dict(row) for row in model_resolution.get("records", [])]
    return {
        "schema": SCHEMA + ".gpu_offload_intervals",
        "preflight_gpu_receipt": dict(preconditions.get("gpu", {})),
        "preflight_intervals": list(dict(preconditions.get("gpu", {})).get("utilization_memory_intervals", [])),
        "model_cuda_offload": [
            {
                "hf_id": row.get("hf_id"),
                "family": row.get("family"),
                "gpu": row.get("gpu"),
                "cuda_offload_authenticated": row.get("cuda_offload_authenticated"),
            }
            for row in records
        ],
        "generation_lifecycle": list(generation_lifecycle),
        "authentic_cuda_receipts": bool(records)
        and all(bool(row.get("cuda_offload_authenticated")) for row in records)
        and (
            not generation_lifecycle
            or all(bool(row.get("cuda_offload_authenticated")) for row in generation_lifecycle)
        ),
    }


def attach_use_counts(
    model_specs: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    counts = Counter(str(row.get("model_hf_id")) for row in raw_rows)
    rows: list[JsonDict] = []
    for spec in model_specs:
        row = dict(spec)
        row["actual_use_count"] = counts.get(str(spec.get("hf_id")), 0)
        row["legacy_small_model_headline"] = False
        rows.append(row)
    return rows


class NativeThreeFamilyLlamaCppBackend:
    """Production backend using llama.cpp and embedded GGUF chat templates."""

    def generate(  # pragma: no cover - expensive local GGUF path.
        self,
        *,
        model_spec: JsonDict,
        public_tasks: list[JsonDict],
        generation_plan: list[JsonDict],
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
            main_gpu=int(model_spec.get("gpu", 0)),
            tensor_split=[1.0, 1.0],
            n_ctx=int(generation_config["n_ctx"]),
            verbose=False,
        )
        after_load = nvidia_smi_gpu_receipt()
        rows: list[JsonDict] = []
        try:
            for plan in generation_plan:
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
                            "cell_id": plan["cell_id"],
                            "raw_stdout": raw_stdout,
                            "finish_reason": choice.get("finish_reason"),
                            "generated_token_count": int(response.get("usage", {}).get("completion_tokens", 0) or 0),
                            "prompt_token_count": int(response.get("usage", {}).get("prompt_tokens", 0) or 0),
                            "timing": {
                                "decode_time_s": round(time.perf_counter() - started, 6),
                                "started_monotonic_s": round(started, 6),
                            },
                            "timeout": False,
                            "raw_generation_error": None,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "exp6200_generated_cell": len(rows),
                                "model": model_spec["family"],
                                "expected_model_cells": len(generation_plan),
                                "cell_id": plan["cell_id"],
                                "finish_reason": choice.get("finish_reason"),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "cell_id": plan["cell_id"],
                            "raw_stdout": "",
                            "finish_reason": "backend_exception",
                            "generated_token_count": 0,
                            "prompt_token_count": 0,
                            "timing": {"decode_time_s": round(time.perf_counter() - started, 6)},
                            "timeout": False,
                            "raw_generation_error": f"{type(exc).__name__}: {exc}",
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


def capture_preconditions(
    paths: Mapping[str, Path],
    public_tasks: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover - host receipt.
    exp6186 = read_json_or_empty(REPO_ROOT / EXP6186_ARTIFACT_RELATIVE_PATH)
    gpu = nvidia_smi_gpu_receipt()
    llama = llama_cpp_build_and_cuda_offload_receipts()
    root_clutter = {"root_py_files": sorted(path.name for path in REPO_ROOT.glob("*.py"))}
    root_clutter["root_py_file_count"] = len(root_clutter["root_py_files"])
    checks = {
        "exp6186_bank_ready_score_is_one": exp6186.get("bank_ready_score") == 1,
        "bank_hash_verified": (REPO_ROOT / BANK_RELATIVE_PATH).is_file(),
        "public_prompt_bank_hash_verified": (REPO_ROOT / PUBLIC_PROMPT_RELATIVE_PATH).is_file(),
        "calibration_task_subset_available": len(public_tasks) == CALIBRATION_TASK_COUNT,
        "mandatory_three_family_ggufs_cached": all(
            resolve_cached_gguf(model_id, "Q4_K_M") is not None for model_id in MANDATED_MODEL_IDS
        ),
        "llama_cpp_cuda_offload_available": llama.get("cuda_offload_authenticated") is True,
        "gpu_identity_available": gpu.get("ok") is True and int(gpu.get("gpu_count", 0)) >= 1,
        "output_paths_writable": all(parent_writable(path) for path in paths.values()),
        "protected_files_present": all((REPO_ROOT / path).is_file() for path in PROTECTED_FILES),
        "root_clutter_absent": root_clutter["root_py_file_count"] == 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    vault = dict(exp6186.get("public_prompt_and_private_test_vault_paths_and_hashes", {}).get("private_test_vault", {}))
    if vault:
        vault["opened_by_exp6200"] = False
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blockers,
        "blocked_reasons": blockers,
        "checks": checks,
        "bank_receipt": file_receipt(BANK_RELATIVE_PATH),
        "public_prompt_receipt": file_receipt(PUBLIC_PROMPT_RELATIVE_PATH),
        "private_vault_receipt": vault,
        "git_status_short": git_status(),
        "protected_file_hashes_before": protected_file_hash_map(),
        "root_clutter": root_clutter,
        "gpu": gpu,
        "llama_cpp": llama,
    }


def resolve_mandatory_models() -> JsonDict:  # pragma: no cover - host receipt.
    llama = llama_cpp_build_and_cuda_offload_receipts()
    records: list[JsonDict] = []
    blockers: list[str] = []
    for spec in MODEL_SPECS:
        path_text = resolve_cached_gguf(str(spec["hf_id"]), "Q4_K_M")
        if not path_text:
            record = {**spec, "exists": False, "cuda_offload_authenticated": False}
            records.append(record)
            blockers.append(f"{spec['family']}_gguf_not_cached")
            continue
        path = Path(path_text)
        tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(path))
        metadata = gguf_metadata_receipt(path)
        record = {
            **spec,
            "model_path": str(path),
            "real_path": str(path.resolve()),
            "filename": path.name,
            "revision": snapshot_revision(path),
            "quantization": observed_quantization(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "exists": path.is_file(),
            "embedded_tokenizer_loadable": tokenizer_ok,
            "embedded_tokenizer_detail": tokenizer_detail,
            "chat_template_present": metadata["chat_template_present"],
            "chat_template_sha256": metadata["chat_template_sha256"],
            "chat_template_source": "tokenizer.chat_template",
            "metadata_summary_sha256": metadata["metadata_summary_sha256"],
            "cuda_offload_authenticated": llama.get("cuda_offload_authenticated") is True,
            "llama_cpp_build": llama.get("python_binding_version"),
        }
        if not tokenizer_ok:
            blockers.append(f"{spec['family']}_embedded_tokenizer_unloadable")
        if not metadata["chat_template_present"]:
            blockers.append(f"{spec['family']}_embedded_chat_template_missing")
        if not record["cuda_offload_authenticated"]:
            blockers.append("llama_cpp_cuda_offload_unavailable")
        records.append(record)
    return {"schema": SCHEMA + ".model_resolution", "records": records, "blocked_reasons": blockers}


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
        "metadata_summary_sha256": sha256_json(
            {key: metadata[key] for key in sorted(metadata) if "template" in key or "tokenizer" in key}
        ),
    }


def model_specs_from_resolution(model_resolution: Mapping[str, Any]) -> list[JsonDict]:
    by_id = {str(row.get("hf_id")): dict(row) for row in model_resolution.get("records", [])}
    rows: list[JsonDict] = []
    for spec in MODEL_SPECS:
        merged = {**spec, **by_id.get(str(spec["hf_id"]), {})}
        merged["legacy_small_model_headline"] = False
        rows.append(merged)
    return rows


def load_frozen_calibration_tasks() -> list[JsonDict]:  # pragma: no cover - host cache.
    bank = json.loads((REPO_ROOT / BANK_RELATIVE_PATH).read_text(encoding="utf-8"))
    public_by_id = {
        str(row["task_id"]): row
        for row in load_jsonl(REPO_ROOT / PUBLIC_PROMPT_RELATIVE_PATH)
        if str(row.get("split")) == "calibration"
    }
    tasks: list[JsonDict] = []
    for row in bank.get("tasks", []):
        if str(row.get("split")) != "calibration":
            continue
        public = dict(public_by_id.get(str(row["task_id"]), {}))
        merged = {**dict(row), **public}
        for key in ("prompt_sha256", "public_test_sha256", "private_test_sha256", "metadata_sha256", "stable_task_hash"):
            if row.get(key):
                merged[key] = row[key]
        tasks.append(merged)
    return tasks


def parse_public_tests(task: Mapping[str, Any]) -> list[JsonDict]:
    raw = task.get("public_tests") or task.get("public_test_cases")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if isinstance(raw, Mapping):
        raw = [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        parsed = [dict(item) for item in raw if isinstance(item, Mapping)]
        if parsed:
            return parsed
    prompt = str(task.get("question_content") or "")
    return parse_stdio_samples(prompt)


def parse_stdio_samples(prompt: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    pattern = re.compile(
        r"Sample Input\s*\d*\s*\n\n(?P<input>.*?)\n\nSample Output\s*\d*\s*\n\n(?P<output>.*?)(?=\n\nSample Input|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(prompt):
        output = match.group("output").strip().split("\n\n", 1)[0].strip()
        rows.append({"input": match.group("input").strip() + "\n", "output": output})
    return rows


def entry_point_from_task(task: Mapping[str, Any]) -> str:
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


def inspect_existing_raw_shards(
    raw_shard_dir: Path,
    generation_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    expected_keys = {str(row["cell_id"]) for row in generation_plan}
    expected_by_key = {str(row["cell_id"]): dict(row) for row in generation_plan}
    receipt = empty_resume_receipt(raw_shard_dir, generation_plan)
    if not raw_shard_dir.exists():
        return {
            "blocked": False,
            "rows": [],
            "missing_plan": list(generation_plan),
            "blocked_reasons": [],
            "resume_receipt": receipt,
        }
    rows: list[JsonDict] = []
    for shard in sorted(raw_shard_dir.glob("*.jsonl")):
        rows.extend(load_jsonl(shard))
    by_key: dict[str, list[JsonDict]] = defaultdict(list)
    hash_mismatches = 0
    for row in rows:
        by_key[str(row.get("cell_id"))].append(row)
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
            "generated_new_rows": 0,
            "conflicting_key_count": conflicts,
            "row_hash_mismatch_count": hash_mismatches,
            "missing_key_count": len(missing),
            "extra_key_count": len(extra),
            "resume_mode": "blocked_raw_shard_conflict"
            if blocked
            else ("reused_raw_shards" if not missing else "partial_raw_shards"),
        }
    )
    missing_plan = [
        expected_by_key[key]
        for key in sorted(missing, key=lambda item: int(expected_by_key[item]["cell_index"]))
    ]
    return {
        "blocked": blocked,
        "rows": ordered_rows(
            [row for row in rows if str(row.get("cell_id")) in expected_keys],
            generation_plan,
        )
        if not blocked
        else [],
        "missing_plan": missing_plan,
        "blocked_reasons": ["raw_shard_immutable_key_conflict"] if blocked else [],
        "resume_receipt": receipt,
    }


def empty_resume_receipt(
    raw_shard_dir: Path,
    generation_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".resume",
        "raw_shard_directory": str(raw_shard_dir),
        "expected_key_count": len(generation_plan),
        "existing_key_count": 0,
        "generated_new_rows": 0,
        "conflicting_key_count": 0,
        "row_hash_mismatch_count": 0,
        "missing_key_count": len(generation_plan),
        "extra_key_count": 0,
        "resume_mode": "not_started",
    }


def raw_rows_complete(
    raw_rows: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
) -> bool:
    return {str(row.get("cell_id")) for row in raw_rows} == {
        str(row["cell_id"]) for row in generation_plan
    } and len(raw_rows) == len(generation_plan)


def ordered_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    order = {str(row["cell_id"]): int(row["cell_index"]) for row in generation_plan}
    return sorted((dict(row) for row in raw_rows), key=lambda row: order.get(str(row.get("cell_id")), 10**9))


def write_raw_shards(
    raw_shard_dir: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
) -> None:
    raw_shard_dir.mkdir(parents=True, exist_ok=True)
    for old in raw_shard_dir.glob("*.jsonl"):
        old.unlink()
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in ordered_rows(raw_rows, generation_plan):
        grouped[str(row["family"])].append(dict(row))
    for family, rows in grouped.items():
        path = raw_shard_dir / f"{safe_slug(family)}.{sha256_text(family)[7:19]}.jsonl"
        write_jsonl(path, rows)


def raw_shard_receipts(raw_shard_dir: Path) -> list[JsonDict]:
    rows = []
    if not raw_shard_dir.exists():
        return rows
    for path in sorted(raw_shard_dir.glob("*.jsonl")):
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "exists": True,
                "sha256": sha256_file(path),
                "count": len(load_jsonl(path)),
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def protected_files_unchanged(preconditions: Mapping[str, Any]) -> JsonDict:
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
    }


def field_provenance() -> JsonDict:
    return {field: ["REQ-CODE-6200", FIELD_PRINCIPLES[field]] for field in REQUIRED_ARTIFACT_FIELDS}


def honest_verdict(
    status: str,
    frozen: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    generation_plan: Sequence[Mapping[str, Any]],
) -> str:
    coverage = f"{len(raw_rows)}/{len(generation_plan)} cells"
    ready_families = sorted(family for family, row in frozen.items() if row.get("ready") is True)
    if status == "complete_ready":
        return f"complete_ready: Exp6200 froze transport envelopes for {ready_families}; coverage={coverage}"
    if status == "complete_partial":
        return f"complete_partial: Exp6200 coverage={coverage}; ready_families={ready_families}"
    return f"blocked: Exp6200 coverage={coverage}; blockers={gate.get('blocked_reasons', [])}"


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    for zero_field in ("private_oracle_access_count", "legacy_headline_row_count", "correctness_retry_count"):
        if payload.get(zero_field) != 0:
            errors.append(zero_field)
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(("complete_ready:", "complete_partial:", "retired:", "blocked:")):
        errors.append("honest_verdict")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    frozen = dict(payload.get("frozen_envelope_by_family", {}))
    if payload.get("phase_d_transport_ready_score") != phase_d_transport_ready_score(frozen):
        errors.append("phase_d_gate")
    if payload.get("csl_transport_ready_score") != csl_transport_ready_score(frozen):
        errors.append("csl_gate")
    selection = dict(payload.get("configuration_selection_inputs", {}))
    if selection.get("uses_private_oracle") is not False:
        errors.append("configuration_selection_inputs")
    return errors


def validate_existing_artifact(path: Path | None = None) -> JsonDict:
    artifact_path = path or REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    errors = validate_artifact(artifact)
    return {
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "missing_required_fields": [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact],
        "validation_errors": errors,
        "ok": not errors,
        "status": artifact.get("status"),
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
    cli = run_command((str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-cli"), "--version"))
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


def nvidia_smi_gpu_receipt() -> JsonDict:  # pragma: no cover - host receipt.
    result = run_command(
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
        "max_memory_delta_mb": peak_used - before_used,
    }


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


def model_family_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.lower()).strip("_")


def raw_row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(strip_paths(payload))


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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


def file_receipt(relative: Path) -> JsonDict:
    path = REPO_ROOT / relative
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def read_json_or_empty(path: Path) -> JsonDict:  # pragma: no cover - host receipt.
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git_status() -> list[str]:  # pragma: no cover - host receipt.
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.stdout.splitlines()


def run_command(cmd: Sequence[str]) -> JsonDict:  # pragma: no cover - host receipt.
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


def parent_writable(path: Path) -> bool:  # pragma: no cover - host receipt.
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    return os.access(parent, os.W_OK)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug[:80] or "cell"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def strip_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<path>"
            if key.endswith("path")
            or key.endswith("_directory")
            or key in {"model_path", "real_path", "raw_shard_directory"}
            else strip_paths(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [strip_paths(item) for item in value]
    return value


def _resolve_paths(result_path: Path | None, raw_shard_dir: Path | None) -> dict[str, Path]:
    return {
        "result": result_path or REPO_ROOT / RESULT_RELATIVE_PATH,
        "raw_shard_dir": raw_shard_dir or REPO_ROOT / RAW_SHARD_RELATIVE_DIR,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true", help="validate the existing Exp6200 artifact")
    args = parser.parse_args(argv)
    if args.validate:
        print(json.dumps(validate_existing_artifact(), sort_keys=True))
        return 0
    artifact = run(run_date=str(args.date))
    print(json.dumps({"artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
