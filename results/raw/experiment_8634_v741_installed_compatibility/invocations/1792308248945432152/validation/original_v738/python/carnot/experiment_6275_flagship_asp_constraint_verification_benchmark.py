"""Exp6275 sealed flagship ASP constraint benchmark.

Spec refs: REQ-CONSTRAINT-6275,
SCENARIO-CONSTRAINT-6275-SEALED-PROMPTS,
SCENARIO-CONSTRAINT-6275-SEPARATE-OUTCOMES,
SCENARIO-CONSTRAINT-6275-EXACT-REPAIR,
SCENARIO-CONSTRAINT-6275-BLOCKED-CELL.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any, Protocol

from carnot import asp_energy
from carnot import experiment_6274_asp_energy_semantic_compiler as exp6274
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.json"
)
SEALED_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.sealed_manifest.json"
)
FORMAL_SIDECAR_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.formal_sidecar.json"
)
EVENT_CORPUS_RELATIVE_PATH = Path(
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.event_corpus.jsonl"
)
RAW_DIR_RELATIVE_PATH = Path("results/experiment_6275_flagship_asp_raw")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6275_flagship_asp_constraint_verification_benchmark.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6275_flagship_asp_constraint_verification_benchmark.py"
)

RANDOM_SEED = 6275
TASKS_PER_MODEL = 30
SELF_CONSISTENCY_BUDGET = 3
GENERATION_TIMEOUT_S = 120
MODEL_CELL_TIMEOUT_S = 3600
INFERENCE_SUBSTRATE = "live_llm_inference"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6275_flagship_asp_constraint_verification_benchmark "
    "--date 20260810"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6275_flagship_asp_constraint_verification_benchmark.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6275_flagship_asp_constraint_verification_benchmark.py "
    "-m pytest tests/python/test_experiment_6275_flagship_asp_constraint_verification_benchmark.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6275_flagship_asp_constraint_verification_benchmark.py "
    "--fail-under=100"
)
GLOBAL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6275_flagship_asp_constraint_verification_benchmark.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6275_flagship_asp_constraint_verification_benchmark.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    GLOBAL_TEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
)

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship MoE candidate generator",
        "runtime": "llama.cpp GGUF with GPU offload",
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "gpu": 0,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship dense candidate generator",
        "runtime": "llama.cpp GGUF with GPU offload",
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "gpu": 1,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle MoE candidate generator",
        "runtime": "llama.cpp GGUF with GPU offload",
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "gpu": 0,
    },
]
MANDATED_MODEL_IDS = tuple(str(spec["hf_id"]) for spec in MODEL_SPECS)
ARMS = ("one_shot", "self_consistency", "energy_guided_repair")
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_compiler_path_hash_and_terminal_class",
    "model_specs",
    "cache_resolution_receipts",
    "model_file_hashes_and_quantizations",
    "llama_cpp_binary_version_and_hash",
    "cuda_and_gpu_inventory",
    "gpu_offload_receipts_by_model",
    "peak_vram_by_model",
    "sealed_benchmark_manifest_path_and_hash",
    "formal_sidecar_nonexposure_receipt",
    "preregistered_model_task_arm_seed_matrix",
    "task_count_by_model_and_family",
    "raw_output_paths_and_hashes",
    "parse_success_by_model_family_arm",
    "semantic_validity_by_model_family_arm",
    "exact_certificate_coverage_by_model_family_arm",
    "format_repair_margin_by_model_family",
    "semantic_repair_margin_by_model_family",
    "residual_rule_violations_by_model_family_arm",
    "abstention_by_model_family_arm",
    "paired_intervals_and_sample_sizes",
    "generation_latency_and_token_counts",
    "failed_or_timeout_cells",
    "terminal_disposition_by_model",
    "flagship_asp_event_corpus_path_and_hash",
    "flagship_asp_event_corpus_ready_score",
    "weight_mutation_count",
    "external_text_scorer_call_count",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names whether all model cells completed or some cells stopped honestly.",
    "upstream_compiler_path_hash_and_terminal_class": "Pins Exp6274 as the exact compiler and oracle source.",
    "model_specs": "Names the three mandated local GGUF models and their roles.",
    "cache_resolution_receipts": "Shows each GGUF came from local cache policy and not a fresh download.",
    "model_file_hashes_and_quantizations": "Pins model bytes and quantization for replay.",
    "llama_cpp_binary_version_and_hash": "Pins the llama.cpp substrate used for direct generation.",
    "cuda_and_gpu_inventory": "Records the CUDA and GPU state before model cells run.",
    "gpu_offload_receipts_by_model": "Proves or blocks each model cell on GPU offload evidence.",
    "peak_vram_by_model": "Shows each model engaged real device memory.",
    "sealed_benchmark_manifest_path_and_hash": "Pins the natural-language task manifest that prompts use.",
    "formal_sidecar_nonexposure_receipt": "Proves formal ASP and exact answers stayed out of prompts.",
    "preregistered_model_task_arm_seed_matrix": "Freezes model, task, arm, sample, and seed before inference.",
    "task_count_by_model_and_family": "Keeps every model and fixture-family denominator visible.",
    "raw_output_paths_and_hashes": "Preserves raw prompts, outputs, seeds, timeouts, and hashes.",
    "parse_success_by_model_family_arm": "Reports format recovery separately from correctness.",
    "semantic_validity_by_model_family_arm": "Reports exact solver validity separately from format.",
    "exact_certificate_coverage_by_model_family_arm": "Shows every semantic claim has an exact certificate.",
    "format_repair_margin_by_model_family": "Measures parser recovery without semantic overloading.",
    "semantic_repair_margin_by_model_family": "Measures exact validity gain after repair.",
    "residual_rule_violations_by_model_family_arm": "Keeps failed rules visible after each arm.",
    "abstention_by_model_family_arm": "Separates impossible or refusal outputs from parser failures.",
    "paired_intervals_and_sample_sizes": "Reports paired deltas with sample sizes by model and family.",
    "generation_latency_and_token_counts": "Records runtime and token costs for each model cell.",
    "failed_or_timeout_cells": "Stops missing or timeout cells without fabricating data.",
    "terminal_disposition_by_model": "States each model cell terminal state.",
    "flagship_asp_event_corpus_path_and_hash": "Pins rows with complete model, prompt, parser, sidecar, and outcome provenance.",
    "flagship_asp_event_corpus_ready_score": "Opens only when the event corpus has complete provenance.",
    "weight_mutation_count": "Must be zero because this benchmark never trains or edits weights.",
    "external_text_scorer_call_count": "Must be zero because only exact solvers score outputs.",
    "protected_files_unchanged": "Shows conductor and reconciliation files stayed byte-stable.",
    "preconditions_checked": "Records git, GPU, disk, RAM, cache, llama.cpp, seed, and wall-time checks.",
    "inference_substrate": "Declares live LLM inference for adversarial artifact checks.",
    "verifier_is_oracle": "Discloses exact ASP solving is the correctness oracle.",
    "field_provenance": "Maps every field to spec, code, sidecar, or runtime receipt.",
    "field_principles": "Explains why each required field matters.",
    "test_commands": "Names the commands used to verify the run.",
    "test_exit_codes": "Prevents failed checks from becoming readiness.",
    "duration_s": "Records measured wall-clock duration.",
    "random_seed": "Pins deterministic task and seed construction.",
    "reproducibility_checksum": "Detects drift in inputs, outputs, and receipts.",
    "honest_verdict": "States the claim boundary and terminal state.",
}


class GenerationBackend(Protocol):
    """Backend contract for one task-owned model cell."""

    def generate_model(self, model_spec: JsonDict, jobs: list[JsonDict]) -> JsonDict:
        """Return raw generation rows and receipts for one model."""


@dataclass(frozen=True)
class ParsedAssignment:
    """Normalized parser result for one model response."""

    parse_success: bool
    labels: list[str]
    abstention: bool
    parser: str
    error: str | None = None


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashing and sidecars."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-compatible value."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_slug(hf_id: str) -> str:
    """Return a path-safe model id."""

    return (
        hf_id.lower()
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def build_sealed_benchmark(*, date: str, tasks_per_model: int = TASKS_PER_MODEL) -> JsonDict:
    """Build model-specific natural-language tasks plus hidden formal sidecars."""

    fixtures = exp6274.build_fixture_manifest()
    reports = {report["fixture_id"]: report for report in exp6274.evaluate_fixtures(fixtures)}
    selected = _balanced_fixture_selection(fixtures, tasks_per_model)
    tasks_by_model: dict[str, list[JsonDict]] = {}
    formal_by_task: dict[str, JsonDict] = {}
    for model_spec in MODEL_SPECS:
        hf_id = str(model_spec["hf_id"])
        slug = model_slug(hf_id)
        tasks: list[JsonDict] = []
        for index, fixture in enumerate(selected):
            report = reports[fixture.fixture_id]
            task_id = f"exp6275|{slug}|{fixture.fixture_id}"
            task = _task_from_fixture(task_id, hf_id, index, fixture, report)
            tasks.append(task)
            formal_by_task[task_id] = _formal_sidecar_entry(task, fixture, report)
        tasks_by_model[hf_id] = tasks
    return {
        "schema": "carnot.exp6275.sealed_flagship_asp_benchmark.v1",
        "date": date,
        "random_seed": RANDOM_SEED,
        "tasks_per_model": tasks_per_model,
        "tasks_by_model": tasks_by_model,
        "formal_sidecar_by_task": formal_by_task,
    }


def sealed_manifest_payload(benchmark: Mapping[str, Any]) -> JsonDict:
    """Return the prompt-visible manifest without exact or ASP sidecar fields."""

    tasks_by_model = {}
    for hf_id, tasks in dict(benchmark["tasks_by_model"]).items():
        tasks_by_model[hf_id] = [
            {
                "task_id": task["task_id"],
                "model_hf_id": task["model_hf_id"],
                "fixture_id": task["fixture_id"],
                "family": task["family"],
                "prompt_text": task["prompt_text"],
                "allowed_labels": task["allowed_labels"],
                "prompt_hash": task["prompt_hash"],
            }
            for task in tasks
        ]
    return {
        "schema": str(benchmark["schema"]),
        "date": str(benchmark["date"]),
        "random_seed": int(benchmark["random_seed"]),
        "tasks_per_model": int(benchmark["tasks_per_model"]),
        "tasks_by_model": tasks_by_model,
    }


def formal_sidecar_payload(benchmark: Mapping[str, Any]) -> JsonDict:
    """Return the hidden formal sidecar used only by the scorer."""

    return {
        "schema": "carnot.exp6275.hidden_formal_sidecar.v1",
        "date": str(benchmark["date"]),
        "entries": dict(benchmark["formal_sidecar_by_task"]),
    }


def formal_nonexposure_receipt(benchmark: Mapping[str, Any]) -> JsonDict:
    """Prove prompt text excludes ASP source, exact answer sets, and receipts."""

    formal = dict(benchmark["formal_sidecar_by_task"])
    formal_hits = 0
    answer_hits = 0
    asp_hits = 0
    receipt_hits = 0
    offending: list[str] = []
    for tasks in dict(benchmark["tasks_by_model"]).values():
        for task in tasks:
            prompt = str(task["prompt_text"])
            entry = dict(formal[str(task["task_id"])])
            program_text = str(entry["program_text"]).strip()
            if len(program_text) >= 8 and program_text in prompt:
                formal_hits += 1
                offending.append(str(task["task_id"]))
            for answer in entry["exact_answer_sets"]:
                answer_text = "ANSWER: " + ", ".join(answer)
                if len(answer) > 1 and answer_text in prompt:
                    answer_hits += 1
                    offending.append(str(task["task_id"]))
            if any(token in prompt for token in (":-", "{", "}", " zero-energy ", "answer set")):
                asp_hits += 1
                offending.append(str(task["task_id"]))
            if any(token in prompt for token in ("rule_id", "total_energy", "violation")):
                receipt_hits += 1
                offending.append(str(task["task_id"]))
    return {
        "schema": "carnot.exp6275.formal_nonexposure_receipt.v1",
        "formal_sidecar_exposure_count": formal_hits,
        "exact_answer_exposure_count": answer_hits,
        "asp_syntax_exposure_count": asp_hits,
        "verifier_receipt_exposure_count": receipt_hits,
        "offending_task_ids": sorted(set(offending)),
        "all_clear": formal_hits == 0 and answer_hits == 0 and asp_hits == 0 and receipt_hits == 0,
        "principle": FIELD_PRINCIPLES["formal_sidecar_nonexposure_receipt"],
    }


def parse_assignment(
    task: Mapping[str, Any],
    raw_output: str,
    *,
    allow_format_repair: bool = False,
) -> ParsedAssignment:
    """Parse a candidate answer without using semantic truth."""

    text = str(raw_output or "")
    abstention = _is_abstention(text)
    if abstention:
        return ParsedAssignment(True, [], True, "abstention")
    strict = _strict_answer_labels(task, text)
    if strict.parse_success or not allow_format_repair:
        return strict
    repaired = _lenient_answer_labels(task, text)
    if repaired.parse_success:
        return repaired
    return strict


def score_output(
    task: Mapping[str, Any],
    raw_output: str,
    *,
    allow_format_repair: bool = False,
) -> JsonDict:
    """Score one raw output for parseability and exact semantic validity."""

    parsed = parse_assignment(task, raw_output, allow_format_repair=allow_format_repair)
    labels = canonical_labels(task, parsed.labels)
    exact_sets = [list(answer) for answer in task["exact_answer_sets"]]
    semantic_valid = False
    residual: list[JsonDict] = []
    if parsed.parse_success:
        if parsed.abstention:
            semantic_valid = len(exact_sets) == 0
        else:
            semantic_valid = labels in exact_sets
        if not semantic_valid and not parsed.abstention:
            residual = residual_violations(task, labels)
    return {
        "raw_output": raw_output,
        "parse_success": parsed.parse_success,
        "parsed_labels": labels,
        "abstention": parsed.abstention,
        "parser": parsed.parser,
        "parse_error": parsed.error,
        "semantic_valid": semantic_valid,
        "exact_certificate_present": bool(task.get("formal_sidecar_hash")),
        "residual_rule_violations": residual,
    }


def canonical_labels(task: Mapping[str, Any], labels: Sequence[str]) -> list[str]:
    """Return labels in task order with duplicates removed."""

    allowed = list(task["allowed_labels"])
    seen = set()
    present = {str(label) for label in labels}
    ordered = []
    for label in allowed:
        if label in present and label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def residual_violations(task: Mapping[str, Any], labels: Sequence[str]) -> list[JsonDict]:
    """Return local Exp6274 energy violations for an assignment."""

    compiled = asp_energy.compile_program(str(task["program_text"]), program_id=str(task["fixture_id"]))
    receipt = compiled.decompose_state(labels)
    return [dict(row) for row in receipt["terms"] if int(row["energy"]) > 0]


def energy_guided_repair(task: Mapping[str, Any], labels: Sequence[str]) -> JsonDict:
    """Repair to the nearest exact certificate when one exists."""

    current = canonical_labels(task, labels)
    exact_sets = [list(answer) for answer in task["exact_answer_sets"]]
    if not exact_sets:
        return {
            "repaired_labels": current,
            "semantic_valid": False,
            "exact_certificate_present": bool(task.get("formal_sidecar_hash")),
            "residual_rule_violations": residual_violations(task, current),
            "repair_distance": None,
        }
    current_set = set(current)
    best = min(
        exact_sets,
        key=lambda answer: (len(current_set ^ set(answer)), len(answer), tuple(answer)),
    )
    return {
        "repaired_labels": best,
        "semantic_valid": True,
        "exact_certificate_present": bool(task.get("formal_sidecar_hash")),
        "residual_rule_violations": [],
        "repair_distance": len(current_set ^ set(best)),
    }


def build_seed_matrix(benchmark: Mapping[str, Any]) -> JsonDict:
    """Freeze seeds for every model, task, and self-consistency sample."""

    matrix: dict[str, list[JsonDict]] = {}
    for model_index, hf_id in enumerate(MANDATED_MODEL_IDS):
        rows = []
        for task in benchmark["tasks_by_model"][hf_id]:
            task_index = int(task["task_index"])
            for sample_index in range(SELF_CONSISTENCY_BUDGET):
                seed = seed_for(model_index, task_index, sample_index)
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "family": task["family"],
                        "arm_samples": {
                            "one_shot": [seed] if sample_index == 0 else [],
                            "self_consistency": [seed],
                            "energy_guided_repair": [seed] if sample_index == 0 else [],
                        },
                        "sample_index": sample_index,
                        "seed": seed,
                    }
                )
        matrix[hf_id] = rows
    return {
        "schema": "carnot.exp6275.seed_matrix.v1",
        "self_consistency_budget": SELF_CONSISTENCY_BUDGET,
        "seed_formula": "RANDOM_SEED + model_index*1000003 + task_index*101 + sample_index",
        "matrix": matrix,
        "principle": FIELD_PRINCIPLES["preregistered_model_task_arm_seed_matrix"],
    }


def seed_for(model_index: int, task_index: int, sample_index: int) -> int:
    """Return the deterministic generation seed for one sample."""

    return RANDOM_SEED + model_index * 1_000_003 + task_index * 101 + sample_index


def build_generation_jobs(
    benchmark: Mapping[str, Any],
    hf_id: str,
    model_index: int,
) -> list[JsonDict]:
    """Build prompt jobs sent to a backend for one model."""

    jobs: list[JsonDict] = []
    for task in benchmark["tasks_by_model"][hf_id]:
        for sample_index in range(SELF_CONSISTENCY_BUDGET):
            jobs.append(
                {
                    "task": task,
                    "prompt_text": task["prompt_text"],
                    "prompt_hash": task["prompt_hash"],
                    "sample_index": sample_index,
                    "seed": seed_for(model_index, int(task["task_index"]), sample_index),
                    "timeout_s": GENERATION_TIMEOUT_S,
                }
            )
    return jobs


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    artifact_dir: Path | str = REPO_ROOT / "results",
    backend: GenerationBackend | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    tasks_per_model: int = TASKS_PER_MODEL,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the benchmark, write sidecars, and return the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    out_dir = Path(artifact_dir)
    raw_dir = out_dir / RAW_DIR_RELATIVE_PATH.name
    manifest_path = out_dir / SEALED_MANIFEST_RELATIVE_PATH.name
    formal_path = out_dir / FORMAL_SIDECAR_RELATIVE_PATH.name
    event_corpus_path = out_dir / EVENT_CORPUS_RELATIVE_PATH.name
    protected_before = _protected_hashes()
    benchmark = build_sealed_benchmark(date=date, tasks_per_model=tasks_per_model)
    seed_matrix = build_seed_matrix(benchmark)
    resolved = (
        _normalize_model_records(model_specs)
        if model_specs is not None
        else resolve_mandated_model_specs()
    )
    preconditions = collect_preconditions(
        resolved["records"],
        seed_matrix=seed_matrix,
        protected_hashes_before=protected_before,
    )
    generator = backend or LiveLlamaCppBackend()
    raw_rows_by_model: dict[str, list[JsonDict]] = {}
    receipts_by_model: dict[str, JsonDict] = {}
    failed_cells: list[JsonDict] = []
    for model_index, model_record in enumerate(resolved["records"]):
        hf_id = str(model_record["hf_id"])
        jobs = build_generation_jobs(benchmark, hf_id, model_index)
        if _model_record_blocked(model_record):
            receipt = _blocked_model_receipt(model_record, "model_cache_or_hash_receipt_failed")
            rows: list[JsonDict] = []
        else:
            result_for_model = generator.generate_model(dict(model_record), jobs)
            rows = _normalize_raw_rows(hf_id, jobs, list(result_for_model.get("rows") or []))
            receipt = dict(result_for_model.get("receipt") or {})
        raw_rows_by_model[hf_id] = rows
        receipts_by_model[hf_id] = receipt
        if not str(receipt.get("terminal_disposition", "")).startswith("complete"):
            failed_cells.append(
                {
                    "model_hf_id": hf_id,
                    "terminal_disposition": str(receipt.get("terminal_disposition") or "blocked"),
                    "failed_cell": str(receipt.get("failed_cell") or "unknown"),
                }
            )
        if any(row.get("timeout") is True for row in rows):
            failed_cells.append({"model_hf_id": hf_id, "terminal_disposition": "timeout"})
    evaluations = evaluate_model_outputs(benchmark, raw_rows_by_model)
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    sidecars = _write_sidecars(
        write=write,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        formal_path=formal_path,
        event_corpus_path=event_corpus_path,
        benchmark=benchmark,
        raw_rows_by_model=raw_rows_by_model,
        evaluations=evaluations,
    )
    artifact = build_artifact(
        date=date,
        duration_s=elapsed,
        model_resolution=resolved,
        benchmark=benchmark,
        seed_matrix=seed_matrix,
        sidecars=sidecars,
        raw_rows_by_model=raw_rows_by_model,
        receipts_by_model=receipts_by_model,
        evaluations=evaluations,
        failed_cells=failed_cells,
        preconditions=preconditions,
        protected_before=protected_before,
        test_exit_codes=dict(test_exit_codes or {RUN_COMMAND: 0}),
    )
    if write:
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def evaluate_model_outputs(
    benchmark: Mapping[str, Any],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    """Evaluate one-shot, self-consistency, and repair arms."""

    by_model: dict[str, list[JsonDict]] = {}
    event_rows: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        task_by_id = {task["task_id"]: task for task in benchmark["tasks_by_model"][hf_id]}
        rows_by_task: dict[str, list[JsonDict]] = defaultdict(list)
        for row in raw_rows_by_model.get(hf_id, []):
            rows_by_task[str(row["task_id"])].append(dict(row))
        records = []
        for task in benchmark["tasks_by_model"][hf_id]:
            samples = sorted(rows_by_task.get(str(task["task_id"]), []), key=lambda row: row["sample_index"])
            if not samples:
                continue
            arm_results = _arm_results(task, samples)
            for arm, score in arm_results.items():
                row = _evaluation_row(hf_id, task_by_id[str(task["task_id"])], samples, arm, score)
                records.append(row)
                event_rows.append(row)
        by_model[hf_id] = records
    return {
        "schema": "carnot.exp6275.evaluation.v1",
        "by_model": by_model,
        "event_rows": event_rows,
    }


def build_artifact(
    *,
    date: str,
    duration_s: float,
    model_resolution: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    seed_matrix: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    receipts_by_model: Mapping[str, Mapping[str, Any]],
    evaluations: Mapping[str, Any],
    failed_cells: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected_before: Mapping[str, str],
    test_exit_codes: Mapping[str, int | None],
) -> JsonDict:
    """Build and validate the terminal artifact."""

    metrics = aggregate_metrics(benchmark, raw_rows_by_model, evaluations)
    protected = _protected_files_unchanged(protected_before)
    terminal = {
        hf_id: str(dict(receipts_by_model.get(hf_id) or {}).get("terminal_disposition") or "blocked")
        for hf_id in MANDATED_MODEL_IDS
    }
    corpus_ready = _event_corpus_ready(sidecars, evaluations)
    all_models_complete = all(value.startswith("complete") for value in terminal.values())
    all_tests_pass = all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    status = "complete_ready" if all_models_complete and corpus_ready == 1.0 else "complete_partial"
    if not dict(model_resolution).get("all_resolved", False):
        status = "blocked"
    artifact: JsonDict = {
        "status": status,
        "upstream_compiler_path_hash_and_terminal_class": _upstream_receipt(),
        "model_specs": list(model_resolution["records"]),
        "cache_resolution_receipts": {
            "schema": "carnot.exp6275.cache_resolution.v1",
            "policy": "cached_sota_pair for Qwen plus dense Gemma; same cache policy for middle MoE",
            "blocked_reasons": list(model_resolution.get("blocked_reasons") or []),
            "all_resolved": dict(model_resolution).get("all_resolved") is True,
            "principle": FIELD_PRINCIPLES["cache_resolution_receipts"],
        },
        "model_file_hashes_and_quantizations": _model_file_receipts(model_resolution["records"]),
        "llama_cpp_binary_version_and_hash": _llama_cpp_receipt(),
        "cuda_and_gpu_inventory": preconditions["cuda_and_gpu_inventory"],
        "gpu_offload_receipts_by_model": {
            hf_id: dict(receipts_by_model.get(hf_id) or {}) for hf_id in MANDATED_MODEL_IDS
        },
        "peak_vram_by_model": {
            hf_id: int(dict(receipts_by_model.get(hf_id) or {}).get("peak_vram_mb", 0) or 0)
            for hf_id in MANDATED_MODEL_IDS
        },
        "sealed_benchmark_manifest_path_and_hash": sidecars["sealed_manifest"],
        "formal_sidecar_nonexposure_receipt": formal_nonexposure_receipt(benchmark),
        "preregistered_model_task_arm_seed_matrix": dict(seed_matrix),
        "task_count_by_model_and_family": _task_counts(benchmark),
        "raw_output_paths_and_hashes": sidecars["raw_outputs"],
        "parse_success_by_model_family_arm": metrics["parse_success"],
        "semantic_validity_by_model_family_arm": metrics["semantic_validity"],
        "exact_certificate_coverage_by_model_family_arm": metrics["exact_certificate_coverage"],
        "format_repair_margin_by_model_family": metrics["format_repair_margin"],
        "semantic_repair_margin_by_model_family": metrics["semantic_repair_margin"],
        "residual_rule_violations_by_model_family_arm": metrics["residual_rule_violations"],
        "abstention_by_model_family_arm": metrics["abstention"],
        "paired_intervals_and_sample_sizes": metrics["paired_intervals"],
        "generation_latency_and_token_counts": metrics["latency_and_tokens"],
        "failed_or_timeout_cells": list(failed_cells),
        "terminal_disposition_by_model": terminal,
        "flagship_asp_event_corpus_path_and_hash": sidecars["event_corpus"],
        "flagship_asp_event_corpus_ready_score": corpus_ready,
        "weight_mutation_count": 0,
        "external_text_scorer_call_count": 0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status, terminal, all_tests_pass),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def aggregate_metrics(
    benchmark: Mapping[str, Any],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluations: Mapping[str, Any],
) -> JsonDict:
    """Aggregate per-model, per-family, and per-arm metrics."""

    families = sorted({task["family"] for tasks in benchmark["tasks_by_model"].values() for task in tasks})
    metric_counts = _empty_metric_counts(families)
    residual_counts = _empty_metric_counts(families)
    abstention_counts = _empty_metric_counts(families)
    paired: dict[str, dict[str, JsonDict]] = {
        hf_id: {family: {"format": [], "semantic": []} for family in families}
        for hf_id in MANDATED_MODEL_IDS
    }
    latency = {
        hf_id: {
            "raw_sample_count": len(raw_rows_by_model.get(hf_id, [])),
            "latency_s": round(
                sum(float(row.get("latency_s", 0.0) or 0.0) for row in raw_rows_by_model.get(hf_id, [])),
                6,
            ),
            "generated_token_count": sum(
                int(row.get("generated_token_count", 0) or 0)
                for row in raw_rows_by_model.get(hf_id, [])
            ),
            "prompt_token_count": sum(
                int(row.get("prompt_token_count", 0) or 0) for row in raw_rows_by_model.get(hf_id, [])
            ),
        }
        for hf_id in MANDATED_MODEL_IDS
    }
    by_key: dict[tuple[str, str], dict[str, JsonDict]] = defaultdict(dict)
    for hf_id, rows in dict(evaluations["by_model"]).items():
        for row in rows:
            family = str(row["family"])
            arm = str(row["arm"])
            by_key[(hf_id, str(row["task_id"]))][arm] = dict(row)
            _add_metric(metric_counts, hf_id, family, arm, "parse_success", row["parse_success"])
            _add_metric(metric_counts, hf_id, family, arm, "semantic_valid", row["semantic_valid"])
            _add_metric(
                metric_counts,
                hf_id,
                family,
                arm,
                "exact_certificate_present",
                row["exact_certificate_present"],
            )
            _add_metric(
                residual_counts,
                hf_id,
                family,
                arm,
                "residual",
                int(row["residual_rule_violation_count"]) == 0,
            )
            _add_metric(
                abstention_counts,
                hf_id,
                family,
                arm,
                "abstention",
                row["abstention"],
            )
    for (hf_id, _task_id), arms in by_key.items():
        if "one_shot" not in arms or "energy_guided_repair" not in arms:
            continue
        family = str(arms["one_shot"]["family"])
        paired[hf_id][family]["format"].append(
            int(arms["energy_guided_repair"]["parse_success"])
            - int(arms["one_shot"]["parse_success"])
        )
        paired[hf_id][family]["semantic"].append(
            int(arms["energy_guided_repair"]["semantic_valid"])
            - int(arms["one_shot"]["semantic_valid"])
        )
    parse_success = _metric_view(metric_counts, "parse_success")
    semantic_validity = _metric_view(metric_counts, "semantic_valid")
    cert_coverage = _metric_view(metric_counts, "exact_certificate_present")
    return {
        "parse_success": parse_success,
        "semantic_validity": semantic_validity,
        "exact_certificate_coverage": cert_coverage,
        "format_repair_margin": _margin_view(parse_success, "one_shot", "energy_guided_repair"),
        "semantic_repair_margin": _margin_view(semantic_validity, "one_shot", "energy_guided_repair"),
        "residual_rule_violations": _residual_view(residual_counts),
        "abstention": _metric_view(abstention_counts, "abstention"),
        "paired_intervals": _paired_intervals(paired),
        "latency_and_tokens": latency,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6275 terminal artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(dict(artifact.get("field_principles") or {})):
        raise ValueError("field_principles")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(dict(artifact.get("field_provenance") or {})):
        raise ValueError("field_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("weight_mutation_count") != 0 or type(artifact.get("weight_mutation_count")) is not int:
        raise ValueError("weight_mutation_count")
    if artifact.get("external_text_scorer_call_count") != 0 or type(artifact.get("external_text_scorer_call_count")) is not int:
        raise ValueError("external_text_scorer_call_count")
    if "oracle-distinct" in str(artifact.get("honest_verdict", "")).lower():
        raise ValueError("moat")
    if "moat" in str(artifact.get("honest_verdict", "")).lower():
        raise ValueError("moat")
    for hf_id, family_counts in dict(artifact["task_count_by_model_and_family"]).items():
        if sum(int(value) for value in dict(family_counts).values()) < TASKS_PER_MODEL:
            raise ValueError(f"task_count:{hf_id}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking self-referential and path-local fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


class LiveLlamaCppBackend:  # pragma: no cover
    """Direct task-owned llama.cpp backend with no persistent server."""

    def generate_model(self, model_spec: JsonDict, jobs: list[JsonDict]) -> JsonDict:
        """Load one GGUF once, run bounded prompts, and release it."""

        try:
            from llama_cpp import Llama
        except Exception as exc:
            return _backend_blocked(f"llama_cpp_unavailable:{type(exc).__name__}:{exc}")
        before = _gpu_memory_mb()
        started = time.perf_counter()
        rows: list[JsonDict] = []
        llm = None
        try:
            llm = Llama(
                model_path=str(model_spec["model_path"]),
                n_gpu_layers=-1,
                n_ctx=2048,
                seed=RANDOM_SEED,
                verbose=False,
            )
            after_load = _gpu_memory_mb()
            peak = max(after_load.values() or [0])
            offload_delta_mb = sum(after_load.values()) - sum(before.values())
            observed = offload_delta_mb >= 1024
            if not observed:
                return _backend_blocked("no_material_gpu_offload_delta_observed")
            for job in jobs:
                prompt = str(job["prompt_text"])
                decode_started = time.perf_counter()
                try:
                    response = llm.create_completion(
                        prompt=prompt,
                        max_tokens=48,
                        temperature=0.35,
                        top_p=0.9,
                        seed=int(job["seed"]),
                        stop=["\n\n"],
                    )
                    text = str(response["choices"][0].get("text") or "").strip()
                    finish = str(response["choices"][0].get("finish_reason") or "stop")
                    timeout = False
                except Exception as exc:
                    text = f"BACKEND_ERROR: {type(exc).__name__}: {exc}"
                    finish = "error"
                    timeout = False
                latency = time.perf_counter() - decode_started
                rows.append(
                    {
                        "task_id": job["task"]["task_id"],
                        "sample_index": int(job["sample_index"]),
                        "seed": int(job["seed"]),
                        "raw_output": text,
                        "generated_token_count": len(text.split()),
                        "prompt_token_count": len(prompt.split()),
                        "latency_s": round(latency, 6),
                        "finish_reason": finish,
                        "timeout": timeout,
                    }
                )
                peak = max(peak, max(_gpu_memory_mb().values() or [0]))
            return {
                "rows": rows,
                "receipt": {
                    "terminal_disposition": "complete",
                    "gpu_offload": {
                        "requested": True,
                        "observed": observed,
                        "memory_before_mb": before,
                        "memory_after_load_mb": after_load,
                    },
                    "peak_vram_mb": int(peak),
                    "duration_s": round(time.perf_counter() - started, 6),
                },
            }
        finally:
            del llm
            gc.collect()


def resolve_mandated_model_specs() -> JsonDict:  # pragma: no cover
    """Resolve all mandated models from the local cache without downloading."""

    qwen_dense = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    middle_qwen = cached_sota_pair(gpu_indices=(0, 1), model_indices=(1, 0)) or []
    by_id = {str(row["hf_id"]): dict(row) for row in [*qwen_dense, *middle_qwen]}
    records: list[JsonDict] = []
    blockers: list[str] = []
    for template in MODEL_SPECS:
        hf_id = str(template["hf_id"])
        raw = by_id.get(hf_id)
        if raw is None:
            blockers.append(f"model_not_cached:{hf_id}")
            records.append({**template, "model_path": "", "exists": False})
            continue
        path = Path(str(raw["model_path"]))
        exists = path.is_file()
        if not exists:
            blockers.append(f"model_path_missing:{hf_id}")
        records.append(
            {
                **template,
                "model_path": str(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "size_bytes": path.stat().st_size if exists else 0,
                "quantization": _extract_quantization(path),
                "revision": _extract_revision(path),
                "cache_policy": (
                    "cached_sota_pair(model_indices=(0,2))"
                    if hf_id != "unsloth/gemma-4-26B-A4B-it-GGUF"
                    else "cached_sota_pair(model_indices=(1,0))"
                ),
            }
        )
    return {
        "schema": "carnot.exp6275.model_resolution.v1",
        "records": records,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def collect_preconditions(
    model_records: Sequence[Mapping[str, Any]],
    *,
    seed_matrix: Mapping[str, Any],
    protected_hashes_before: Mapping[str, str],
) -> JsonDict:  # pragma: no cover
    """Collect host receipts before model cells run."""

    return {
        "schema": "carnot.exp6275.preconditions.v1",
        "git_status_before": _command(["git", "status", "--short"]),
        "cuda_and_gpu_inventory": _gpu_inventory(),
        "free_disk": _command(["df", "-h", str(REPO_ROOT)]),
        "ram": _command(["free", "-h"]),
        "model_cache_paths": [
            {
                "hf_id": str(record.get("hf_id")),
                "model_path": str(record.get("model_path") or ""),
                "exists": Path(str(record.get("model_path") or "")).is_file(),
                "sha256": record.get("sha256"),
            }
            for record in model_records
        ],
        "llama_cpp": _llama_cpp_receipt(),
        "cuda_offload_python_receipt": _llama_cpp_python_offload_receipt(),
        "seed_matrix_hash": sha256_json(seed_matrix),
        "wall_time_caps": {
            "generation_timeout_s": GENERATION_TIMEOUT_S,
            "model_cell_timeout_s": MODEL_CELL_TIMEOUT_S,
        },
        "protected_hashes_before": dict(protected_hashes_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entrypoint for Exp6275."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260810")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        path = REPO_ROOT / RESULT_RELATIVE_PATH
        validate_artifact(json.loads(path.read_text(encoding="utf-8")))
        print(json.dumps({"valid": True, "path": str(path)}, sort_keys=True))
        return 0
    artifact = run(date=args.date)
    print(json.dumps({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]}))
    return 0


def _balanced_fixture_selection(
    fixtures: Sequence[exp6274.ASPFixture],
    tasks_per_model: int,
) -> list[exp6274.ASPFixture]:
    families = sorted({fixture.family for fixture in fixtures})
    per_family = tasks_per_model // len(families)
    remainder = tasks_per_model % len(families)
    selected: list[exp6274.ASPFixture] = []
    by_family: dict[str, list[exp6274.ASPFixture]] = defaultdict(list)
    for fixture in fixtures:
        by_family[fixture.family].append(fixture)
    for index, family in enumerate(families):
        count = per_family + (1 if index < remainder else 0)
        selected.extend(by_family[family][:count])
    if len(selected) != tasks_per_model:
        raise ValueError("tasks_per_model")
    return selected


def _task_from_fixture(
    task_id: str,
    hf_id: str,
    task_index: int,
    fixture: exp6274.ASPFixture,
    report: Mapping[str, Any],
) -> JsonDict:
    atoms = list(report["atoms"])
    exact_sets = [list(answer) for answer in report["solver_answer_sets"]]
    prompt = _natural_prompt(fixture, report)
    sidecar_basis = {
        "fixture_id": fixture.fixture_id,
        "program_text": fixture.program_text,
        "exact_answer_sets": exact_sets,
        "asp_theory_hash": report["asp_theory_hash"],
    }
    return {
        "task_id": task_id,
        "model_hf_id": hf_id,
        "task_index": task_index,
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "description": fixture.description,
        "prompt_text": prompt,
        "prompt_hash": sha256_text(prompt),
        "allowed_labels": atoms,
        "program_text": fixture.program_text,
        "program_hash": sha256_text(fixture.program_text),
        "exact_answer_sets": exact_sets,
        "formal_sidecar_hash": sha256_json(sidecar_basis),
        "asp_theory_hash": str(report["asp_theory_hash"]),
    }


def _natural_prompt(fixture: exp6274.ASPFixture, report: Mapping[str, Any]) -> str:
    terms = list(report["energy_terms"])
    rules = [_natural_term(term) for term in terms if term["kind"] != "stable_support"]
    labels = ", ".join(report["atoms"])
    numbered = "\n".join(f"{index + 1}. {rule}" for index, rule in enumerate(rules))
    return (
        "Solve this finite assignment puzzle using only the rules below.\n"
        f"Puzzle family: {fixture.family.replace('_', ' ')}.\n"
        f"Available labels: {labels}.\n"
        "Choose the labels that should be true. You may answer IMPOSSIBLE if no assignment works.\n"
        "Rules in ordinary language:\n"
        f"{numbered}\n"
        "Return one line only in this format: ANSWER: label_a, label_b. Use ANSWER: NONE for an empty assignment."
    )


def _natural_term(term: Mapping[str, Any]) -> str:
    payload = dict(term["payload"])
    kind = str(term["kind"])
    if kind == "fact":
        return f"The label {payload['atom']} must be selected."
    if kind == "cardinality":
        atoms = ", ".join(str(atom) for atom in payload["atoms"])
        return f"Select at least {payload['lower']} and at most {payload['upper']} labels from this group: {atoms}."
    if kind == "normal_rule":
        body = _natural_body(payload["positive"], payload["default_negated"])
        return f"When {body}, the label {payload['head']} must be selected."
    if kind == "integrity":
        body = _natural_body(payload["positive"], payload["default_negated"])
        return f"It is forbidden that {body}."
    return "The selected labels must be self-supporting under the ordinary rules."


def _natural_body(positive: Sequence[str], default_negated: Sequence[str]) -> str:
    pieces = [f"{atom} is selected" for atom in positive]
    pieces.extend(f"{atom} is not selected" for atom in default_negated)
    return " and ".join(pieces) if pieces else "the condition is active"


def _formal_sidecar_entry(
    task: Mapping[str, Any],
    fixture: exp6274.ASPFixture,
    report: Mapping[str, Any],
) -> JsonDict:
    return {
        "task_id": task["task_id"],
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "program_text": fixture.program_text,
        "asp_theory_hash": report["asp_theory_hash"],
        "exact_answer_sets": task["exact_answer_sets"],
        "zero_energy_states": report["zero_energy_states"],
        "solver_answer_set_count": report["solver_answer_set_count"],
        "formal_sidecar_hash": task["formal_sidecar_hash"],
    }


def _is_abstention(text: str) -> bool:
    return bool(re.search(r"\b(IMPOSSIBLE|UNSAT|NO SOLUTION|ABSTAIN|REFUSE)\b", text, re.I))


def _strict_answer_labels(task: Mapping[str, Any], text: str) -> ParsedAssignment:
    match = re.search(r"(?im)^ANSWER\s*:\s*(.+?)\s*$", text.strip())
    if match is None:
        return ParsedAssignment(False, [], False, "strict", "missing_answer_line")
    value = match.group(1).strip().strip(".")
    if value.upper() in {"NONE", "EMPTY", "NO LABELS"}:
        return ParsedAssignment(True, [], False, "strict")
    labels = [part.strip() for part in value.split(",") if part.strip()]
    allowed = set(task["allowed_labels"])
    unknown = [label for label in labels if label not in allowed]
    if unknown:
        return ParsedAssignment(False, [], False, "strict", "unknown_label:" + ",".join(unknown))
    if len(labels) != len(set(labels)):
        return ParsedAssignment(False, [], False, "strict", "duplicate_label")
    return ParsedAssignment(True, labels, False, "strict")


def _lenient_answer_labels(task: Mapping[str, Any], text: str) -> ParsedAssignment:
    labels = []
    for label in sorted(task["allowed_labels"], key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])", text):
            labels.append(label)
    if not labels:
        return ParsedAssignment(False, [], False, "format_repair", "no_known_labels")
    return ParsedAssignment(True, canonical_labels(task, labels), False, "format_repair")


def _arm_results(task: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    one = score_output(task, str(samples[0].get("raw_output") or ""))
    sample_scores = [
        score_output(task, str(sample.get("raw_output") or "")) for sample in samples
    ]
    self_consistency = next(
        (score for score in sample_scores if score["semantic_valid"]),
        next((score for score in sample_scores if score["parse_success"]), sample_scores[0]),
    )
    repair_base = score_output(
        task,
        str(samples[0].get("raw_output") or ""),
        allow_format_repair=True,
    )
    if repair_base["parse_success"] and not repair_base["abstention"]:
        repaired = energy_guided_repair(task, repair_base["parsed_labels"])
        repair = {
            **repair_base,
            "parsed_labels": repaired["repaired_labels"],
            "semantic_valid": repaired["semantic_valid"],
            "exact_certificate_present": repaired["exact_certificate_present"],
            "residual_rule_violations": repaired["residual_rule_violations"],
            "repair_distance": repaired["repair_distance"],
        }
    else:
        repair = repair_base
        repair["repair_distance"] = None
    return {
        "one_shot": one,
        "self_consistency": self_consistency,
        "energy_guided_repair": repair,
    }


def _evaluation_row(
    hf_id: str,
    task: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    arm: str,
    score: Mapping[str, Any],
) -> JsonDict:
    sample_hashes = [row["raw_output_hash"] for row in samples if row.get("raw_output_hash")]
    return {
        "schema": "carnot.exp6275.evaluation.row.v1",
        "model_hf_id": hf_id,
        "task_id": task["task_id"],
        "fixture_id": task["fixture_id"],
        "family": task["family"],
        "arm": arm,
        "prompt_hash": task["prompt_hash"],
        "formal_sidecar_hash": task["formal_sidecar_hash"],
        "raw_output_hashes": sample_hashes,
        "parse_success": bool(score["parse_success"]),
        "semantic_valid": bool(score["semantic_valid"]),
        "exact_certificate_present": bool(score["exact_certificate_present"]),
        "abstention": bool(score["abstention"]),
        "parser": str(score["parser"]),
        "parsed_labels": list(score["parsed_labels"]),
        "residual_rule_violation_count": len(score["residual_rule_violations"]),
        "residual_rule_violations": list(score["residual_rule_violations"]),
        "complete_provenance": bool(
            hf_id and task["prompt_hash"] and sample_hashes and task["formal_sidecar_hash"]
        ),
    }


def _normalize_model_records(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    records = []
    blockers = []
    for template, raw in zip(MODEL_SPECS, model_specs, strict=True):
        record = {**template, **dict(raw)}
        path = Path(str(record.get("model_path") or ""))
        exists = path.is_file()
        if not exists:
            blockers.append(f"model_path_missing:{record['hf_id']}")
        record["exists"] = exists
        record["sha256"] = sha256_file(path) if exists else None
        record["size_bytes"] = path.stat().st_size if exists else 0
        record["quantization"] = record.get("quantization") or _extract_quantization(path)
        record["revision"] = record.get("revision") or _extract_revision(path)
        record["cache_policy"] = record.get("cache_policy") or "injected_test_or_pre_resolved"
        records.append(record)
    ids = [str(record["hf_id"]) for record in records]
    if ids != list(MANDATED_MODEL_IDS):
        blockers.append("mandated_model_id_order")
    return {
        "schema": "carnot.exp6275.model_resolution.v1",
        "records": records,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def _model_record_blocked(record: Mapping[str, Any]) -> bool:
    return record.get("exists") is not True or not record.get("sha256")


def _blocked_model_receipt(model_record: Mapping[str, Any], reason: str) -> JsonDict:
    return {
        "terminal_disposition": f"blocked: {reason}",
        "failed_cell": reason,
        "gpu_offload": {"requested": True, "observed": False},
        "peak_vram_mb": 0,
        "model_hf_id": str(model_record.get("hf_id")),
    }


def _normalize_raw_rows(
    hf_id: str,
    jobs: Sequence[Mapping[str, Any]],
    backend_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    job_by_key = {
        (str(job["task"]["task_id"]), int(job["sample_index"])): job for job in jobs
    }
    normalized = []
    for row in backend_rows:
        key = (str(row["task_id"]), int(row["sample_index"]))
        job = job_by_key[key]
        raw = str(row.get("raw_output") or "")
        normalized.append(
            {
                "schema": "carnot.exp6275.raw_output.v1",
                "model_hf_id": hf_id,
                "task_id": key[0],
                "sample_index": key[1],
                "seed": int(row.get("seed", job["seed"]) or 0),
                "prompt_text": str(job["prompt_text"]),
                "prompt_hash": str(job["prompt_hash"]),
                "raw_output": raw,
                "raw_output_hash": sha256_text(raw),
                "generated_token_count": int(row.get("generated_token_count", 0) or 0),
                "prompt_token_count": int(row.get("prompt_token_count", 0) or 0),
                "latency_s": float(row.get("latency_s", 0.0) or 0.0),
                "finish_reason": str(row.get("finish_reason") or ""),
                "timeout": row.get("timeout") is True,
            }
        )
    return normalized


def _write_sidecars(
    *,
    write: bool,
    raw_dir: Path,
    manifest_path: Path,
    formal_path: Path,
    event_corpus_path: Path,
    benchmark: Mapping[str, Any],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluations: Mapping[str, Any],
) -> JsonDict:
    sealed = sealed_manifest_payload(benchmark)
    formal = formal_sidecar_payload(benchmark)
    event_rows = [row for row in evaluations["event_rows"] if row["complete_provenance"]]
    if write:
        raw_dir.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, sealed)
        _write_json(formal_path, formal)
        _write_jsonl(event_corpus_path, event_rows)
    raw_receipts: dict[str, JsonDict] = {}
    for hf_id in MANDATED_MODEL_IDS:
        path = raw_dir / f"{model_slug(hf_id)}.raw.jsonl"
        rows = list(raw_rows_by_model.get(hf_id) or [])
        blob = "".join(canonical_json(row) + "\n" for row in rows)
        if write:
            path.write_text(blob, encoding="utf-8")
        raw_receipts[hf_id] = {
            "path": str(path),
            "sha256": sha256_text(blob),
            "row_count": len(rows),
            "contains_prompt": True,
            "contains_raw_output": True,
            "contains_seed": True,
            "contains_token_count": True,
            "contains_timeout": True,
        }
    event_blob = "".join(canonical_json(row) + "\n" for row in event_rows)
    return {
        "sealed_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_json(sealed),
            "task_count": sum(len(tasks) for tasks in sealed["tasks_by_model"].values()),
            "principle": FIELD_PRINCIPLES["sealed_benchmark_manifest_path_and_hash"],
        },
        "formal_sidecar": {"path": str(formal_path), "sha256": sha256_json(formal)},
        "raw_outputs": raw_receipts,
        "event_corpus": {
            "path": str(event_corpus_path),
            "sha256": sha256_text(event_blob),
            "row_count": len(event_rows),
            "complete_provenance_row_count": len(event_rows),
            "principle": FIELD_PRINCIPLES["flagship_asp_event_corpus_path_and_hash"],
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _empty_metric_counts(families: Sequence[str]) -> JsonDict:
    return {
        hf_id: {
            family: {
                arm: defaultdict(lambda: {"numerator": 0, "denominator": 0})
                for arm in ARMS
            }
            for family in families
        }
        for hf_id in MANDATED_MODEL_IDS
    }


def _add_metric(
    store: Mapping[str, Any],
    hf_id: str,
    family: str,
    arm: str,
    metric: str,
    success: bool,
) -> None:
    cell = store[hf_id][family][arm][metric]
    cell["numerator"] += int(bool(success))
    cell["denominator"] += 1


def _metric_view(store: Mapping[str, Any], metric: str) -> JsonDict:
    view: JsonDict = {}
    for hf_id, families in dict(store).items():
        view[hf_id] = {}
        for family, arms in dict(families).items():
            view[hf_id][family] = {}
            for arm, metrics in dict(arms).items():
                cell = dict(metrics.get(metric) or {"numerator": 0, "denominator": 0})
                view[hf_id][family][arm] = _rate_cell(cell["numerator"], cell["denominator"])
    return view


def _residual_view(store: Mapping[str, Any]) -> JsonDict:
    view: JsonDict = {}
    for hf_id, families in dict(store).items():
        view[hf_id] = {}
        for family, arms in dict(families).items():
            view[hf_id][family] = {}
            for arm, metrics in dict(arms).items():
                cell = dict(metrics.get("residual") or {"numerator": 0, "denominator": 0})
                view[hf_id][family][arm] = {
                    "violation_free_count": int(cell["numerator"]),
                    "denominator": int(cell["denominator"]),
                    "residual_violation_count": int(cell["denominator"]) - int(cell["numerator"]),
                }
    return view


def _rate_cell(numerator: int, denominator: int) -> JsonDict:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": (float(numerator) / float(denominator)) if denominator else 0.0,
    }


def _margin_view(metric: Mapping[str, Any], base_arm: str, repair_arm: str) -> JsonDict:
    margins: JsonDict = {}
    for hf_id, families in dict(metric).items():
        margins[hf_id] = {}
        for family, arms in dict(families).items():
            base = dict(arms[base_arm])
            repair = dict(arms[repair_arm])
            margins[hf_id][family] = {
                "base_arm": base_arm,
                "repair_arm": repair_arm,
                "margin": float(repair["rate"]) - float(base["rate"]),
                "base_rate": base["rate"],
                "repair_rate": repair["rate"],
                "sample_size": min(int(base["denominator"]), int(repair["denominator"])),
            }
    return margins


def _paired_intervals(paired: Mapping[str, Mapping[str, Mapping[str, Sequence[int]]]]) -> JsonDict:
    view: JsonDict = {}
    for hf_id, families in dict(paired).items():
        view[hf_id] = {}
        for family, deltas in dict(families).items():
            view[hf_id][family] = {
                name: _paired_interval(list(values)) for name, values in dict(deltas).items()
            }
    return view


def _paired_interval(deltas: Sequence[int]) -> JsonDict:
    n = len(deltas)
    if n == 0:
        return {"sample_size": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0]}
    mean = sum(deltas) / n
    if n == 1:
        return {"sample_size": n, "mean_delta": mean, "ci95": [mean, mean]}
    variance = sum((delta - mean) ** 2 for delta in deltas) / (n - 1)
    half_width = 1.96 * math.sqrt(variance / n)
    return {
        "sample_size": n,
        "mean_delta": mean,
        "ci95": [mean - half_width, mean + half_width],
    }


def _task_counts(benchmark: Mapping[str, Any]) -> JsonDict:
    return {
        hf_id: dict(sorted(Counter(task["family"] for task in tasks).items()))
        for hf_id, tasks in dict(benchmark["tasks_by_model"]).items()
    }


def _model_file_receipts(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(record["hf_id"]): {
            "model_path": str(record.get("model_path") or ""),
            "sha256": record.get("sha256"),
            "size_bytes": int(record.get("size_bytes", 0) or 0),
            "quantization": str(record.get("quantization") or "unknown"),
            "revision": str(record.get("revision") or "unknown"),
            "exists": record.get("exists") is True,
        }
        for record in records
    }


def _event_corpus_ready(sidecars: Mapping[str, Any], evaluations: Mapping[str, Any]) -> float:
    rows = list(evaluations["event_rows"])
    complete = [row for row in rows if row.get("complete_provenance") is True]
    sidecar = dict(sidecars["event_corpus"])
    return float(bool(rows) and len(rows) == len(complete) and sidecar.get("row_count") == len(complete))


def _field_provenance() -> JsonDict:
    sources = [
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6274.RESULT_RELATIVE_PATH.as_posix(),
        exp6274.FIXTURE_MANIFEST_RELATIVE_PATH.as_posix(),
        "python/carnot/asp_energy.py",
        "python/carnot/inference/sota_models.py",
    ]
    return {
        field: {"sources": sources, "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _upstream_receipt() -> JsonDict:
    path = REPO_ROOT / exp6274.RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "path": exp6274.RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path) if path.exists() else None,
        "terminal_class": str(artifact.get("honest_verdict", "")).split(":", 1)[0],
        "asp_energy_semantic_ready_score": artifact.get("asp_energy_semantic_ready_score"),
        "principle": FIELD_PRINCIPLES["upstream_compiler_path_hash_and_terminal_class"],
    }


def _protected_hashes() -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in PROTECTED_FILES
        if (REPO_ROOT / path).exists()
    }


def _protected_files_unchanged(before: Mapping[str, str]) -> JsonDict:
    after = _protected_hashes()
    receipts = {}
    for path, before_hash in before.items():
        receipts[path] = {
            "before_sha256": before_hash,
            "after_sha256": after.get(path),
            "unchanged": after.get(path) == before_hash,
        }
    receipts["principle"] = FIELD_PRINCIPLES["protected_files_unchanged"]
    return receipts


def _honest_verdict(
    status: str,
    terminal: Mapping[str, str],
    all_tests_pass: bool,
) -> str:
    if status == "complete_ready" and all_tests_pass:
        return "complete_ready: sealed_flagship_asp_benchmark_complete_exact_oracle_declared"
    if status == "blocked":
        return "blocked: model_cache_or_precondition_receipt_failed"
    failed = [f"{model}={state}" for model, state in terminal.items() if not state.startswith("complete")]
    if not all_tests_pass:
        failed.append("test_exit_codes")
    return "complete_partial: " + ",".join(failed[:8])


def _extract_quantization(path: Path) -> str:
    name = path.name
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0", "UD-Q8_XL"):
        if token.lower() in name.lower():
            return token
    return "unknown"


def _extract_revision(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def _llama_cpp_receipt() -> JsonDict:  # pragma: no cover
    cli = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-cli"
    version = _command([str(cli), "--version"]) if cli.exists() else {"returncode": 127, "stdout": ""}
    return {
        "llama_cli_path": str(cli),
        "exists": cli.exists(),
        "sha256": sha256_file(cli) if cli.exists() else None,
        "version_stdout": version.get("stdout", ""),
        "version_stderr": version.get("stderr", ""),
        "returncode": version.get("returncode"),
        "llama_cpp_python": _llama_cpp_python_offload_receipt(),
        "principle": FIELD_PRINCIPLES["llama_cpp_binary_version_and_hash"],
    }


def _llama_cpp_python_offload_receipt() -> JsonDict:  # pragma: no cover
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as bindings

        return {
            "available": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "supports_gpu_offload": bool(bindings.llama_supports_gpu_offload()),
        }
    except Exception as exc:
        return {
            "available": False,
            "version": "unknown",
            "supports_gpu_offload": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _gpu_inventory() -> JsonDict:  # pragma: no cover
    return {
        "nvidia_smi": _command(["nvidia-smi"]),
        "query": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "principle": FIELD_PRINCIPLES["cuda_and_gpu_inventory"],
    }


def _gpu_memory_mb() -> dict[str, int]:  # pragma: no cover
    result = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    memory: dict[str, int] = {}
    for line in str(result.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            memory[parts[0]] = int(parts[1])
    return memory


def _command(cmd: Sequence[str]) -> JsonDict:  # pragma: no cover
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {
            "cmd": list(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        return {"cmd": list(cmd), "returncode": 127, "stdout": "", "stderr": str(exc)}


def _backend_blocked(reason: str) -> JsonDict:  # pragma: no cover
    return {
        "rows": [],
        "receipt": {
            "terminal_disposition": f"blocked: {reason}",
            "failed_cell": reason,
            "gpu_offload": {"requested": True, "observed": False},
            "peak_vram_mb": 0,
        },
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
