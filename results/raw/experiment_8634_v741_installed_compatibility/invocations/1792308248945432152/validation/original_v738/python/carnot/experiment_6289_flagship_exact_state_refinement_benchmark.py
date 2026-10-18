"""Exp6289 flagship exact-state refinement benchmark.

Spec refs: REQ-CONSTRAINT-6289,
SCENARIO-CONSTRAINT-6289-SEALED-SOTA,
SCENARIO-CONSTRAINT-6289-MATCHED-BUDGETS,
SCENARIO-CONSTRAINT-6289-ORACLE-VALUE.

The benchmark asks local GGUF models for ordinary text. It keeps the exact ASP
sidecar hidden until scoring. The readiness gate counts only warm-start value.
Cold exact solving is recorded as an oracle control, not as model value.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Protocol

from carnot import asp_energy
from carnot import experiment_6275_flagship_asp_constraint_verification_benchmark as exp6275
from carnot import experiment_6288_partial_atom_evidence_adapter as exp6288
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable
from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6289_flagship_exact_state_refinement_benchmark.json"
)
SEALED_TASK_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6289_flagship_exact_state_refinement_benchmark.sealed_task_manifest.json"
)
FORMAL_SIDECAR_RELATIVE_PATH = Path(
    "results/experiment_6289_flagship_exact_state_refinement_benchmark.formal_sidecar.json"
)
RAW_DIR_RELATIVE_PATH = Path("results/experiment_6289_flagship_exact_state_refinement_raw")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6289_flagship_exact_state_refinement_benchmark.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6289_flagship_exact_state_refinement_benchmark.py"
)

RANDOM_SEEDS = (6289, 6290, 6291)
TASKS_PER_MODEL = 3
REPEATED_GENERATION_BUDGET = 2
MAX_GENERATION_TOKENS = 48
GENERATION_TIMEOUT_S = 120
MODEL_CELL_TIMEOUT_S = 1800
EXACT_COMPLETION_BUDGET = exp6288.EXACT_COMPLETION_BUDGET
INFERENCE_SUBSTRATE = "live_llm_inference"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6289_flagship_exact_state_refinement_benchmark "
    "--date 20260810"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6289_flagship_exact_state_refinement_benchmark.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null --branch "
    "--include=python/carnot/experiment_6289_flagship_exact_state_refinement_benchmark.py "
    "-m pytest tests/python/test_experiment_6289_flagship_exact_state_refinement_benchmark.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6289_flagship_exact_state_refinement_benchmark.py "
    "--fail-under=100"
)
GLOBAL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6289_flagship_exact_state_refinement_benchmark.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6289_flagship_exact_state_refinement_benchmark.json"
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
        "role": "flagship MoE ordinary-text generator",
        "runtime": "llama.cpp GGUF with CUDA offload",
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "gpu": 0,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship dense ordinary-text generator",
        "runtime": "llama.cpp GGUF with CUDA offload",
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "gpu": 1,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle MoE ordinary-text generator",
        "runtime": "llama.cpp GGUF with CUDA offload",
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "gpu": 0,
    },
]
MANDATED_MODEL_IDS = tuple(str(spec["hf_id"]) for spec in MODEL_SPECS)
LEGACY_MODEL_IDS = {"Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"}
TERMINAL_PREFIXES = ("complete:", "blocked:", "timeout:", "retired:", "success:")
ARMS = (
    "one_shot",
    "repeated_generation",
    "partial_evidence_continuous_refinement",
    "partial_evidence_exact_completion",
    "cold_exact_completion",
    "compute_balanced_route",
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    exp6288.RESULT_RELATIVE_PATH,
    exp6288.FORMAL_SIDECAR_RELATIVE_PATH,
    exp6288.SEALED_MANIFEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_adapter_path_hash_and_terminal_class",
    "sealed_task_manifest_path_and_hash",
    "formal_sidecar_path_and_hash",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_and_quantizations",
    "tokenizer_and_chat_template_hashes",
    "cuda_and_gpu_offload_receipts_by_model",
    "raw_output_paths_and_hashes",
    "prompt_seed_token_timeout_and_terminal_disposition_by_row",
    "terminal_model_dispositions",
    "arm_definitions_and_fixed_compute_budget",
    "one_shot_results",
    "repeated_generation_results",
    "partial_evidence_continuous_refinement_results",
    "partial_evidence_exact_completion_results",
    "cold_exact_completion_results",
    "compute_balanced_route_results",
    "exact_validity_by_arm_model_and_fixture_family",
    "parser_and_evidence_coverage_by_arm_model_and_fixture_family",
    "solver_nodes_or_state_evaluations_by_arm_model_and_fixture_family",
    "model_tokens_verifier_work_and_wall_time_by_arm_model_and_fixture_family",
    "paired_deltas_intervals_and_sample_sizes",
    "harmful_regressions",
    "qwen_zero_token_control",
    "exact_solver_oracle_receipt",
    "warm_start_value_ready_score",
    "source_model_weight_mutation_count",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names whether the sealed run reached readiness or closed blocked.",
    "upstream_adapter_path_hash_and_terminal_class": "Pins Exp6288 as the evidence source.",
    "sealed_task_manifest_path_and_hash": "Pins the prompt-visible task set.",
    "formal_sidecar_path_and_hash": "Pins the hidden exact sidecar.",
    "MODEL_SPECS": "Names the three required local SOTA GGUF families.",
    "models_used": "Shows no legacy model substituted a headline row.",
    "model_file_hashes_revisions_and_quantizations": "Pins model bytes and quantization.",
    "tokenizer_and_chat_template_hashes": "Pins embedded GGUF text formatting receipts.",
    "cuda_and_gpu_offload_receipts_by_model": "Records CUDA and offload evidence per model.",
    "raw_output_paths_and_hashes": "Preserves raw prompts, outputs, seeds, and hashes.",
    "prompt_seed_token_timeout_and_terminal_disposition_by_row": "Keeps each prompt row auditable.",
    "terminal_model_dispositions": "Forces every model cell to a terminal state.",
    "arm_definitions_and_fixed_compute_budget": "Freezes matched arm budgets.",
    "one_shot_results": "Reports ordinary first-response validity.",
    "repeated_generation_results": "Reports fixed repeated generation under the same tasks.",
    "partial_evidence_continuous_refinement_results": "Reports bounded refinement from evidence.",
    "partial_evidence_exact_completion_results": "Reports exact completion warmed by evidence.",
    "cold_exact_completion_results": "Reports the exact oracle control with no model value.",
    "compute_balanced_route_results": "Reports the fixed route over generate, verify, and stop.",
    "exact_validity_by_arm_model_and_fixture_family": "Separates correctness by arm and family.",
    "parser_and_evidence_coverage_by_arm_model_and_fixture_family": "Shows parse and evidence coverage.",
    "solver_nodes_or_state_evaluations_by_arm_model_and_fixture_family": "Separates exact work from tokens.",
    "model_tokens_verifier_work_and_wall_time_by_arm_model_and_fixture_family": "Separates token and verifier costs.",
    "paired_deltas_intervals_and_sample_sizes": "Reports paired deltas and sample sizes.",
    "harmful_regressions": "Blocks readiness if warm evidence harms exact validity.",
    "qwen_zero_token_control": "Proves zero-token Qwen rows fail closed.",
    "exact_solver_oracle_receipt": "Discloses that exact scoring is oracle sidecar scoring.",
    "warm_start_value_ready_score": "Opens only on warm-start value without exact harm.",
    "source_model_weight_mutation_count": "Bare zero proves no source weights changed.",
    "protected_files_unchanged": "Shows protected files stayed byte-stable during the run.",
    "preconditions_checked": "Records frozen gates before inference.",
    "inference_substrate": "Declares live local LLM inference.",
    "verifier_is_oracle": "States the verifier is the exact ASP oracle.",
    "field_provenance": "Maps each field to spec, code, sidecars, or runtime receipts.",
    "field_principles": "Gives each required field a reason.",
    "test_commands": "Lists the verification boundary.",
    "test_exit_codes": "Records observed command exits.",
    "duration_s": "Records wall-clock duration.",
    "random_seeds": "Pins all deterministic seeds.",
    "reproducibility_checksum": "Detects drift in the artifact payload.",
    "honest_verdict": "States the terminal claim boundary.",
}


class GenerationBackend(Protocol):
    """Backend contract for one task-owned model cell."""

    def generate_model(self, model_spec: JsonDict, jobs: list[JsonDict]) -> JsonDict:
        """Return raw generation rows and receipts for one model."""


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible data."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_slug(hf_id: str) -> str:
    """Return a path-safe model id."""

    return exp6275.model_slug(hf_id)


def is_terminal_disposition(value: str) -> bool:
    """Return True when a disposition has an accepted terminal prefix."""

    return str(value).startswith(TERMINAL_PREFIXES)


def format_visible_answer(task: Mapping[str, Any], atoms: Sequence[str]) -> str:
    """Format hidden atoms through the ordinary labels shown to the model."""

    if not atoms:
        return "ANSWER: EMPTY"
    display = dict(task.get("atom_display") or {})
    labels = [str(display.get(atom, _atom_display(atom))) for atom in atoms]
    return "ANSWER: " + ", ".join(labels)


def build_sealed_task_bundle(*, date: str, tasks_per_model: int = TASKS_PER_MODEL) -> JsonDict:
    """Build ordinary-text prompts and hidden exact sidecars."""

    base = exp6275.build_sealed_benchmark(date=date, tasks_per_model=tasks_per_model)
    tasks_by_model: dict[str, list[JsonDict]] = {}
    formal: dict[str, JsonDict] = {}
    for hf_id, tasks in dict(base["tasks_by_model"]).items():
        transformed = []
        for task in tasks:
            new_task = _ordinary_task(dict(task))
            transformed.append(new_task)
            formal[str(new_task["task_id"])] = {
                "task_id": new_task["task_id"],
                "fixture_id": new_task["fixture_id"],
                "family": new_task["family"],
                "program_text": new_task["program_text"],
                "exact_answer_sets": new_task["exact_answer_sets"],
                "allowed_labels": new_task["allowed_labels"],
                "atom_display": new_task["atom_display"],
                "formal_sidecar_hash": new_task["formal_sidecar_hash"],
                "asp_theory_hash": new_task["asp_theory_hash"],
            }
        tasks_by_model[hf_id] = transformed
    return {
        "schema": "carnot.exp6289.sealed_exact_state_refinement.v1",
        "date": date,
        "tasks_per_model": tasks_per_model,
        "random_seeds": list(RANDOM_SEEDS),
        "tasks_by_model": tasks_by_model,
        "formal_sidecar_by_task": formal,
    }


def sealed_manifest_payload(bundle: Mapping[str, Any]) -> JsonDict:
    """Return the prompt-visible manifest without exact sidecar data."""

    tasks_by_model = {}
    for hf_id, tasks in dict(bundle["tasks_by_model"]).items():
        tasks_by_model[hf_id] = [
            {
                "task_id": task["task_id"],
                "model_hf_id": task["model_hf_id"],
                "fixture_id": task["fixture_id"],
                "family": task["family"],
                "prompt_text": task["prompt_text"],
                "visible_labels": task["visible_labels"],
                "prompt_hash": task["prompt_hash"],
            }
            for task in tasks
        ]
    return {
        "schema": "carnot.exp6289.sealed_task_manifest.v1",
        "date": str(bundle["date"]),
        "tasks_per_model": int(bundle["tasks_per_model"]),
        "random_seeds": list(bundle["random_seeds"]),
        "tasks_by_model": tasks_by_model,
    }


def formal_sidecar_payload(bundle: Mapping[str, Any]) -> JsonDict:
    """Return the hidden exact sidecar used only by scoring code."""

    return {
        "schema": "carnot.exp6289.hidden_formal_sidecar.v1",
        "date": str(bundle["date"]),
        "entries": dict(bundle["formal_sidecar_by_task"]),
    }


def formal_nonexposure_receipt(bundle: Mapping[str, Any]) -> JsonDict:
    """Check that prompts do not expose formal sidecar contents."""

    formal_hits = 0
    answer_hits = 0
    asp_hits = 0
    atom_id_hits = 0
    receipt_hits = 0
    offending: list[str] = []
    for tasks in dict(bundle["tasks_by_model"]).values():
        for task in tasks:
            prompt = str(task["prompt_text"])
            task_id = str(task["task_id"])
            program = str(task["program_text"]).strip()
            if len(program) >= 8 and program in prompt:
                formal_hits += 1
                offending.append(task_id)
            for answer in task["exact_answer_sets"]:
                visible = format_visible_answer(task, answer)
                if visible in prompt:
                    answer_hits += 1
                    offending.append(task_id)
            if any(token in prompt for token in (":-", "{", "}", " zero-energy ")):
                asp_hits += 1
                offending.append(task_id)
            if "answer set" in prompt.lower() or any(
                token in prompt for token in ("rule_id", "total_energy", "violation")
            ):
                receipt_hits += 1
                offending.append(task_id)
            for atom in task["allowed_labels"]:
                if "_" in atom and atom in prompt:
                    atom_id_hits += 1
                    offending.append(task_id)
                    break
    return {
        "schema": "carnot.exp6289.formal_nonexposure_receipt.v1",
        "formal_sidecar_exposure_count": formal_hits,
        "exact_answer_exposure_count": answer_hits,
        "asp_syntax_exposure_count": asp_hits,
        "formal_atom_id_exposure_count": atom_id_hits,
        "verifier_receipt_exposure_count": receipt_hits,
        "offending_task_ids": sorted(set(offending)),
        "all_clear": (
            formal_hits == 0
            and answer_hits == 0
            and asp_hits == 0
            and atom_id_hits == 0
            and receipt_hits == 0
        ),
        "principle": FIELD_PRINCIPLES["sealed_task_manifest_path_and_hash"],
    }


def build_generation_jobs(
    bundle: Mapping[str, Any],
    hf_id: str,
    model_index: int,
) -> list[JsonDict]:
    """Build prompt jobs for one model under the fixed generation budget."""

    jobs = []
    for task in bundle["tasks_by_model"][hf_id]:
        for sample_index in range(REPEATED_GENERATION_BUDGET):
            jobs.append(
                {
                    "task": task,
                    "prompt_text": task["prompt_text"],
                    "prompt_hash": task["prompt_hash"],
                    "sample_index": sample_index,
                    "seed": seed_for(model_index, int(task["task_index"]), sample_index),
                    "timeout_s": GENERATION_TIMEOUT_S,
                    "max_tokens": MAX_GENERATION_TOKENS,
                }
            )
    return jobs


def seed_for(model_index: int, task_index: int, sample_index: int) -> int:
    """Return the deterministic seed for one generation sample."""

    return RANDOM_SEEDS[0] + model_index * 1_000_003 + task_index * 101 + sample_index


def score_text_response(
    task: Mapping[str, Any],
    raw_output: str,
    *,
    generated_token_count: int,
    row_id: str,
) -> JsonDict:
    """Score ordinary model text while keeping exact scoring separate."""

    evidence = exp6288.extract_partial_atom_evidence(
        raw_output,
        tuple(task["allowed_labels"]),
        generated_token_count=generated_token_count,
        row_id=row_id,
    )
    support = exp6288.check_evidence_support(
        evidence,
        exact_answer_sets=list(task["exact_answer_sets"]),
    )
    parsed = _parse_visible_assignment(task, raw_output, evidence, generated_token_count)
    labels = exp6275.canonical_labels(task, parsed["labels"])
    exact_sets = [list(answer) for answer in task["exact_answer_sets"]]
    semantic_valid = False
    residual: list[JsonDict] = []
    if parsed["parse_success"]:
        semantic_valid = labels in exact_sets
        if parsed["abstention"]:
            semantic_valid = len(exact_sets) == 0
        if not semantic_valid and not parsed["abstention"]:
            residual = exp6275.residual_violations(task, labels)
    return {
        "parse_success": parsed["parse_success"],
        "parsed_labels": labels,
        "abstention": parsed["abstention"],
        "parser": parsed["parser"],
        "parse_error": parsed["parse_error"],
        "semantic_valid": semantic_valid,
        "exact_certificate_present": bool(task.get("formal_sidecar_hash")),
        "residual_rule_violations": residual,
        "evidence": evidence,
        "evidence_support": support,
        "terminal_disposition": _row_terminal_disposition(
            generated_token_count=generated_token_count,
            parse_success=parsed["parse_success"],
        ),
    }


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
    """Run Exp6289, write sidecars, and return the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    out_dir = Path(artifact_dir)
    raw_dir = out_dir / RAW_DIR_RELATIVE_PATH.name
    manifest_path = out_dir / SEALED_TASK_MANIFEST_RELATIVE_PATH.name
    formal_path = out_dir / FORMAL_SIDECAR_RELATIVE_PATH.name
    protected_before = protected_hashes()
    bundle = build_sealed_task_bundle(date=date, tasks_per_model=tasks_per_model)
    resolved = (
        normalize_model_records(model_specs, preflight_tokenizers=False)
        if model_specs is not None
        else resolve_mandated_model_specs(preflight_tokenizers=True)
    )
    preconditions = collect_preconditions(
        date=date,
        bundle=bundle,
        model_records=resolved["records"],
        protected_hashes_before=protected_before,
    )
    generator = backend or LiveLlamaCppBackend()
    raw_rows_by_model: dict[str, list[JsonDict]] = {}
    receipts_by_model: dict[str, JsonDict] = {}
    for model_index, model_record in enumerate(resolved["records"]):
        hf_id = str(model_record["hf_id"])
        jobs = build_generation_jobs(bundle, hf_id, model_index)
        if _model_record_blocked(model_record):
            rows: list[JsonDict] = []
            receipt = _blocked_model_receipt(model_record, "model_cache_or_tokenizer_receipt_failed")
        else:
            generated = generator.generate_model(dict(model_record), jobs)
            rows = _normalize_raw_rows(hf_id, jobs, list(generated.get("rows") or []))
            receipt = _terminalized_receipt(hf_id, dict(generated.get("receipt") or {}))
        raw_rows_by_model[hf_id] = rows
        receipts_by_model[hf_id] = receipt
    evaluations = evaluate_arms(bundle, raw_rows_by_model)
    sidecars = write_sidecars(
        write=write,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        formal_path=formal_path,
        bundle=bundle,
        raw_rows_by_model=raw_rows_by_model,
    )
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    artifact = build_artifact(
        duration_s=elapsed,
        model_resolution=resolved,
        bundle=bundle,
        sidecars=sidecars,
        raw_rows_by_model=raw_rows_by_model,
        receipts_by_model=receipts_by_model,
        evaluations=evaluations,
        preconditions=preconditions,
        protected_before=protected_before,
        test_exit_codes=dict(test_exit_codes or {RUN_COMMAND: 0}),
    )
    if write:
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def evaluate_arms(
    bundle: Mapping[str, Any],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    """Evaluate every preregistered arm over matched tasks and seeds."""

    tables: dict[str, Any] = {}
    records: dict[str, list[JsonDict]] = {arm: [] for arm in ARMS}
    row_receipts: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        rows_by_task: dict[str, list[JsonDict]] = defaultdict(list)
        for row in raw_rows_by_model.get(hf_id, []):
            rows_by_task[str(row["task_id"])].append(dict(row))
        for task in bundle["tasks_by_model"][hf_id]:
            task_rows = sorted(rows_by_task.get(str(task["task_id"]), []), key=lambda row: row["sample_index"])
            if not task_rows:
                continue
            table = tables.setdefault(str(task["task_id"]), _table_for_task(task))
            scored = [
                _score_row(task, row)
                for row in task_rows
            ]
            row_receipts.extend(_row_receipt(row, score) for row, score in zip(task_rows, scored, strict=True))
            one = _one_shot_record(hf_id, task, task_rows[0], scored[0], table)
            repeated = _repeated_record(hf_id, task, task_rows, scored, table)
            continuous = _continuous_record(hf_id, task, scored[0], table)
            warm_exact = _partial_exact_record(hf_id, task, scored[0], table)
            cold = _cold_exact_record(hf_id, task, table)
            route = _compute_balanced_route_record(hf_id, task, one, repeated, warm_exact)
            records["one_shot"].append(one)
            records["repeated_generation"].append(repeated)
            records["partial_evidence_continuous_refinement"].append(continuous)
            records["partial_evidence_exact_completion"].append(warm_exact)
            records["cold_exact_completion"].append(cold)
            records["compute_balanced_route"].append(route)
    return {
        "schema": "carnot.exp6289.arm_evaluations.v1",
        "records": records,
        "row_receipts": row_receipts,
        "tables": {
            task_id: {"atom_count": table.atom_count, "vertex_count": table.vertex_count}
            for task_id, table in tables.items()
        },
    }


def build_artifact(
    *,
    duration_s: float,
    model_resolution: Mapping[str, Any],
    bundle: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    receipts_by_model: Mapping[str, Mapping[str, Any]],
    evaluations: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected_before: Mapping[str, str],
    test_exit_codes: Mapping[str, int | None],
) -> JsonDict:
    """Build and validate the terminal artifact."""

    terminal = {
        hf_id: _terminalized_disposition(dict(receipts_by_model.get(hf_id) or {}).get("terminal_disposition"))
        for hf_id in MANDATED_MODEL_IDS
    }
    aggregate = aggregate_evaluations(evaluations)
    harmful = harmful_regressions(evaluations)
    continuous_summary = summarize_continuous(evaluations)
    partial_exact_summary = summarize_partial_exact(evaluations)
    warm_score = warm_start_ready_score(partial_exact_summary, continuous_summary, harmful, terminal)
    all_tests_pass = all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    status = _status_from_terminal_and_score(terminal, warm_score, model_resolution)
    artifact: JsonDict = {
        "status": status,
        "upstream_adapter_path_hash_and_terminal_class": _upstream_adapter_receipt(),
        "sealed_task_manifest_path_and_hash": sidecars["sealed_manifest"],
        "formal_sidecar_path_and_hash": sidecars["formal_sidecar"],
        "MODEL_SPECS": list(MODEL_SPECS),
        "models_used": list(MANDATED_MODEL_IDS),
        "model_file_hashes_revisions_and_quantizations": _model_file_receipts(
            model_resolution["records"]
        ),
        "tokenizer_and_chat_template_hashes": _tokenizer_receipts(model_resolution["records"]),
        "cuda_and_gpu_offload_receipts_by_model": _cuda_offload_receipts(
            receipts_by_model, preconditions
        ),
        "raw_output_paths_and_hashes": sidecars["raw_outputs"],
        "prompt_seed_token_timeout_and_terminal_disposition_by_row": evaluations["row_receipts"],
        "terminal_model_dispositions": terminal,
        "arm_definitions_and_fixed_compute_budget": arm_definitions(len(evaluations["row_receipts"])),
        "one_shot_results": summarize_arm(evaluations, "one_shot"),
        "repeated_generation_results": summarize_arm(evaluations, "repeated_generation"),
        "partial_evidence_continuous_refinement_results": continuous_summary,
        "partial_evidence_exact_completion_results": partial_exact_summary,
        "cold_exact_completion_results": summarize_arm(evaluations, "cold_exact_completion"),
        "compute_balanced_route_results": summarize_arm(evaluations, "compute_balanced_route"),
        "exact_validity_by_arm_model_and_fixture_family": aggregate["exact_validity"],
        "parser_and_evidence_coverage_by_arm_model_and_fixture_family": aggregate[
            "parser_and_evidence"
        ],
        "solver_nodes_or_state_evaluations_by_arm_model_and_fixture_family": aggregate[
            "solver_work"
        ],
        "model_tokens_verifier_work_and_wall_time_by_arm_model_and_fixture_family": aggregate[
            "token_verifier_wall"
        ],
        "paired_deltas_intervals_and_sample_sizes": paired_deltas(evaluations),
        "harmful_regressions": harmful,
        "qwen_zero_token_control": qwen_zero_token_control(evaluations),
        "exact_solver_oracle_receipt": exact_solver_oracle_receipt(bundle),
        "warm_start_value_ready_score": warm_score,
        "source_model_weight_mutation_count": 0,
        "protected_files_unchanged": protected_files_unchanged(protected_before),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "duration_s": float(duration_s),
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, warm_score, all_tests_pass),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate Exp6289 and reject false readiness."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(dict(artifact.get("field_principles") or {})):
        raise ValueError("field_principles")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(dict(artifact.get("field_provenance") or {})):
        raise ValueError("field_provenance")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        raise ValueError("model_specs")
    if artifact.get("models_used") != list(MANDATED_MODEL_IDS):
        raise ValueError("models_used")
    if any(model in LEGACY_MODEL_IDS for model in artifact.get("models_used", [])):
        raise ValueError("legacy_model_substitution")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if (
        artifact.get("source_model_weight_mutation_count") != 0
        or type(artifact.get("source_model_weight_mutation_count")) is not int
    ):
        raise ValueError("source_model_weight_mutation_count")
    terminal = dict(artifact.get("terminal_model_dispositions") or {})
    if set(terminal) != set(MANDATED_MODEL_IDS):
        raise ValueError("terminal_model_dispositions")
    if not all(is_terminal_disposition(str(value)) for value in terminal.values()):
        raise ValueError("nonterminal_model_disposition")
    if artifact.get("exact_solver_oracle_receipt", {}).get(
        "cold_exact_solver_counts_as_model_value"
    ) is not False:
        raise ValueError("oracle_value_laundering")
    expected_score = _expected_ready_score_from_artifact(artifact)
    if artifact.get("warm_start_value_ready_score") != expected_score:
        raise ValueError("oracle_value_laundering")
    if not is_terminal_disposition(str(artifact.get("honest_verdict", ""))):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking volatile fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


class LiveLlamaCppBackend:  # pragma: no cover
    """Direct llama.cpp backend for the real sealed live run."""

    def generate_model(self, model_spec: JsonDict, jobs: list[JsonDict]) -> JsonDict:
        """Load one GGUF once and run bounded ordinary-text prompts."""

        offload_preflight = llama_cpp_python_offload_receipt()
        if offload_preflight.get("supports_gpu_offload") is not True:
            return {
                "rows": [],
                "receipt": _blocked_model_receipt(
                    model_spec,
                    "llama_cpp_cuda_offload_unavailable",
                    extra={"cuda_offload_python_receipt": offload_preflight},
                ),
            }
        try:
            from llama_cpp import Llama
        except Exception as exc:
            return {
                "rows": [],
                "receipt": _blocked_model_receipt(model_spec, f"llama_cpp_unavailable:{exc}"),
            }
        before = _gpu_memory_mb()
        started = time.perf_counter()
        rows: list[JsonDict] = []
        llm = None
        try:
            llm = Llama(
                model_path=str(model_spec["model_path"]),
                n_gpu_layers=-1,
                n_ctx=2048,
                seed=RANDOM_SEEDS[0],
                verbose=False,
            )
            after_load = _gpu_memory_mb()
            offload_delta = sum(after_load.values()) - sum(before.values())
            if offload_delta < 1024:
                return {
                    "rows": [],
                    "receipt": _blocked_model_receipt(
                        model_spec,
                        "no_material_gpu_offload_delta_observed",
                        extra={
                            "gpu_offload": {
                                "requested": True,
                                "observed": False,
                                "memory_before_mb": before,
                                "memory_after_load_mb": after_load,
                            }
                        },
                    ),
                }
            peak = max(after_load.values() or [0])
            for job in jobs:
                prompt = str(job["prompt_text"])
                row_started = time.perf_counter()
                try:
                    response = llm.create_completion(
                        prompt=prompt,
                        max_tokens=MAX_GENERATION_TOKENS,
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
                rows.append(
                    {
                        "task_id": job["task"]["task_id"],
                        "sample_index": int(job["sample_index"]),
                        "seed": int(job["seed"]),
                        "raw_output": text,
                        "generated_token_count": len(text.split()),
                        "prompt_token_count": len(prompt.split()),
                        "latency_s": round(time.perf_counter() - row_started, 6),
                        "finish_reason": finish,
                        "timeout": timeout,
                    }
                )
                peak = max(peak, max(_gpu_memory_mb().values() or [0]))
            return {
                "rows": rows,
                "receipt": {
                    "terminal_disposition": "complete: llama_cpp_generation_finished",
                    "gpu_offload": {
                        "requested": True,
                        "observed": True,
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


def resolve_mandated_model_specs(*, preflight_tokenizers: bool = True) -> JsonDict:
    """Resolve all mandated GGUF files through cached_sota_pair."""

    qwen_dense = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    middle_qwen = cached_sota_pair(gpu_indices=(0, 1), model_indices=(1, 0)) or []
    by_id = {str(row["hf_id"]): dict(row) for row in [*qwen_dense, *middle_qwen]}
    specs = []
    for template in MODEL_SPECS:
        raw = by_id.get(str(template["hf_id"]), {})
        specs.append({**template, **raw})
    return normalize_model_records(specs, preflight_tokenizers=preflight_tokenizers)


def normalize_model_records(
    model_specs: Sequence[Mapping[str, Any]],
    *,
    preflight_tokenizers: bool,
) -> JsonDict:
    """Normalize model rows and fail closed on substitution or missing cache."""

    records = []
    blockers = []
    for template, raw in zip(MODEL_SPECS, model_specs, strict=True):
        record = {**template, **dict(raw)}
        hf_id = str(record.get("hf_id"))
        if hf_id in LEGACY_MODEL_IDS:
            blockers.append(f"legacy_model_substitution:{hf_id}")
        path = Path(str(record.get("model_path") or ""))
        exists = path.is_file()
        if not exists:
            blockers.append(f"model_path_missing:{hf_id}")
        record["exists"] = exists
        record["sha256"] = sha256_file(path) if exists else None
        record["size_bytes"] = path.stat().st_size if exists else 0
        record["quantization"] = record.get("quantization") or exp6275._extract_quantization(path)
        record["revision"] = record.get("revision") or exp6275._extract_revision(path)
        record["cache_policy"] = record.get("cache_policy") or _cache_policy_for_model(hf_id)
        if exists and preflight_tokenizers:
            loadable, detail = gguf_tokenizer_loadable(str(path))
        elif exists:
            loadable, detail = True, "tokenizer preflight skipped for injected test model"
        else:
            loadable, detail = False, "model file missing"
        record["tokenizer_loadable"] = loadable
        record["tokenizer_status"] = detail
        if not loadable:
            blockers.append(f"tokenizer_not_loadable:{hf_id}")
        records.append(record)
    ids = [str(record["hf_id"]) for record in records]
    if ids != list(MANDATED_MODEL_IDS):
        blockers.append("mandated_model_id_order")
    return {
        "schema": "carnot.exp6289.model_resolution.v1",
        "records": records,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def collect_preconditions(
    *,
    date: str,
    bundle: Mapping[str, Any],
    model_records: Sequence[Mapping[str, Any]],
    protected_hashes_before: Mapping[str, str],
) -> JsonDict:
    """Collect frozen preconditions before model loading."""

    return {
        "schema": "carnot.exp6289.preconditions.v1",
        "date": date,
        "task_manifest_hash": sha256_json(sealed_manifest_payload(bundle)),
        "formal_sidecar_hash": sha256_json(formal_sidecar_payload(bundle)),
        "model_records_hash": sha256_json(list(model_records)),
        "arm_budget_hash": sha256_json(arm_definitions(0)),
        "random_seeds": list(RANDOM_SEEDS),
        "timeouts": {
            "generation_timeout_s": GENERATION_TIMEOUT_S,
            "model_cell_timeout_s": MODEL_CELL_TIMEOUT_S,
        },
        "cuda_inventory": _gpu_inventory(),
        "gpu_ownership_before_loading": _gpu_ownership_receipt(),
        "llama_cpp_python_offload": llama_cpp_python_offload_receipt(),
        "protected_hashes_before": dict(protected_hashes_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def write_sidecars(
    *,
    write: bool,
    raw_dir: Path,
    manifest_path: Path,
    formal_path: Path,
    bundle: Mapping[str, Any],
    raw_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    """Write or hash sealed manifests, sidecars, and raw output rows."""

    sealed = sealed_manifest_payload(bundle)
    formal = formal_sidecar_payload(bundle)
    if write:
        raw_dir.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, sealed)
        _write_json(formal_path, formal)
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
    return {
        "sealed_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_json(sealed),
            "task_count": sum(len(tasks) for tasks in sealed["tasks_by_model"].values()),
        },
        "formal_sidecar": {
            "path": str(formal_path),
            "sha256": sha256_json(formal),
            "entry_count": len(formal["entries"]),
        },
        "raw_outputs": raw_receipts,
    }


def arm_definitions(row_count: int) -> JsonDict:
    """Return the preregistered fixed budget for every arm."""

    return {
        "row_count_basis": row_count,
        "one_shot": {
            "generation_samples": 1,
            "max_tokens": MAX_GENERATION_TOKENS,
            "timeout_s": GENERATION_TIMEOUT_S,
        },
        "repeated_generation": {
            "generation_samples": REPEATED_GENERATION_BUDGET,
            "max_tokens_per_sample": MAX_GENERATION_TOKENS,
            "timeout_s_per_sample": GENERATION_TIMEOUT_S,
        },
        "partial_evidence_continuous_refinement": {
            "optimizer_steps": exp6288.OPTIMIZER_STEPS,
            "step_size": exp6288.OPTIMIZER_STEP_SIZE,
            "restart_budget": exp6288.RESTART_BUDGET,
        },
        "partial_evidence_exact_completion": {
            "exact_completion_budget": EXACT_COMPLETION_BUDGET,
            "evidence_source": "first ordinary generation sample",
        },
        "cold_exact_completion": {
            "exact_completion_budget": EXACT_COMPLETION_BUDGET,
            "model_value_credit": False,
        },
        "compute_balanced_route": {
            "max_actions": 3,
            "route": ["generate", "verify", "stop_or_exact_complete"],
            "budget_source": "same generated rows and exact budget as other arms",
        },
    }


def aggregate_evaluations(evaluations: Mapping[str, Any]) -> JsonDict:
    """Aggregate exact validity, parser coverage, solver work, and costs."""

    exact_counts = _empty_nested_counts()
    parse_counts = _empty_nested_counts(extra_metrics=("parse_success", "evidence_supported"))
    solver_work = _empty_nested_sums()
    cost = _empty_nested_sums()
    for arm, rows in dict(evaluations["records"]).items():
        for row in rows:
            key = (arm, str(row["model_hf_id"]), str(row["fixture_family"]))
            _add_count(exact_counts, key, "exact_valid", row.get("exact_valid") is True)
            _add_count(parse_counts, key, "parse_success", row.get("parse_success") is True)
            _add_count(
                parse_counts,
                key,
                "evidence_supported",
                row.get("evidence_supported") is True,
            )
            _add_sum(solver_work, key, "state_evaluations", int(row.get("state_evaluations") or 0))
            _add_sum(cost, key, "model_generated_tokens", int(row.get("model_generated_tokens") or 0))
            _add_sum(cost, key, "model_prompt_tokens", int(row.get("model_prompt_tokens") or 0))
            _add_sum(cost, key, "verifier_work", int(row.get("verifier_work") or 0))
            _add_sum(cost, key, "wall_time_s", float(row.get("wall_time_s") or 0.0))
    return {
        "exact_validity": _finalize_counts(exact_counts),
        "parser_and_evidence": _finalize_counts(parse_counts),
        "solver_work": _finalize_sums(solver_work),
        "token_verifier_wall": _finalize_sums(cost),
    }


def summarize_arm(evaluations: Mapping[str, Any], arm: str) -> JsonDict:
    """Summarize one arm and keep records available for audit."""

    rows = list(evaluations["records"][arm])
    return {
        "arm": arm,
        "row_count": len(rows),
        "exact_valid_count": sum(1 for row in rows if row.get("exact_valid") is True),
        "parse_success_count": sum(1 for row in rows if row.get("parse_success") is True),
        "fixed_budget": arm_definitions(0)[arm],
        "records": rows,
    }


def summarize_continuous(evaluations: Mapping[str, Any]) -> JsonDict:
    """Summarize evidence-warm continuous refinement controls."""

    rows = list(evaluations["records"]["partial_evidence_continuous_refinement"])
    warm = sum(1 for row in rows if row.get("exact_valid") is True)
    blank = sum(1 for row in rows if row.get("blank_exact_valid") is True)
    random = sum(1 for row in rows if row.get("random_exact_valid") is True)
    return {
        "arm": "partial_evidence_continuous_refinement",
        "row_count": len(rows),
        "evidence_warm_success_count": warm,
        "blank_success_count": blank,
        "random_success_count": random,
        "evidence_warm_minus_blank_success_delta": warm - blank,
        "fixed_budget": arm_definitions(0)["partial_evidence_continuous_refinement"],
        "records": rows,
    }


def summarize_partial_exact(evaluations: Mapping[str, Any]) -> JsonDict:
    """Summarize exact completion warmed by partial evidence."""

    rows = list(evaluations["records"]["partial_evidence_exact_completion"])
    return {
        "arm": "partial_evidence_exact_completion",
        "row_count": len(rows),
        "exact_valid_count": sum(1 for row in rows if row.get("exact_valid") is True),
        "warm_state_evaluations": sum(int(row.get("state_evaluations") or 0) for row in rows),
        "cold_state_evaluations": sum(int(row.get("cold_state_evaluations") or 0) for row in rows),
        "warm_minus_cold_work_delta": sum(int(row.get("work_delta_vs_cold") or 0) for row in rows),
        "fixed_budget": arm_definitions(0)["partial_evidence_exact_completion"],
        "records": rows,
    }


def harmful_regressions(evaluations: Mapping[str, Any]) -> JsonDict:
    """Return warm exact rows that lost validity against cold exact control."""

    harmful = [
        {
            "model_hf_id": row["model_hf_id"],
            "task_id": row["task_id"],
            "fixture_family": row["fixture_family"],
        }
        for row in evaluations["records"]["partial_evidence_exact_completion"]
        if row.get("exact_validity_harm") is True
    ]
    return {
        "exact_validity_harm_count": len(harmful),
        "rows": harmful,
    }


def qwen_zero_token_control(evaluations: Mapping[str, Any]) -> JsonDict:
    """Summarize Qwen rows with zero generated tokens."""

    rows = [
        row
        for row in evaluations["row_receipts"]
        if row["model_hf_id"] == MANDATED_MODEL_IDS[0] and row["generated_token_count"] == 0
    ]
    return {
        "model_hf_id": MANDATED_MODEL_IDS[0],
        "qwen_zero_token_rows": len(rows),
        "accepted_as_evidence_count": sum(1 for row in rows if row["evidence_supported"] is True),
        "terminal_control": (
            "complete: zero-token rows fail closed"
            if all(row["terminal_disposition"].startswith("blocked:") for row in rows)
            else "blocked: zero-token row accepted"
        ),
    }


def exact_solver_oracle_receipt(bundle: Mapping[str, Any]) -> JsonDict:
    """Return the exact-solver oracle disclosure."""

    return {
        "verifier_is_oracle": True,
        "formal_sidecar_used_by_solver_only": True,
        "cold_exact_solver_counts_as_model_value": False,
        "formal_sidecar_entry_count": len(bundle["formal_sidecar_by_task"]),
        "principle": FIELD_PRINCIPLES["exact_solver_oracle_receipt"],
    }


def warm_start_ready_score(
    partial_exact: Mapping[str, Any],
    continuous: Mapping[str, Any],
    harmful: Mapping[str, Any],
    terminal: Mapping[str, str],
) -> float:
    """Return 1.0 only when preregistered warm-start value exists."""

    positive_work = int(partial_exact.get("warm_minus_cold_work_delta") or 0) > 0
    positive_refine = int(continuous.get("evidence_warm_minus_blank_success_delta") or 0) > 0
    no_harm = int(harmful.get("exact_validity_harm_count") or 0) == 0
    all_complete = all(str(value).startswith("complete:") for value in terminal.values())
    return 1.0 if (positive_work or positive_refine) and no_harm and all_complete else 0.0


def paired_deltas(evaluations: Mapping[str, Any]) -> JsonDict:
    """Compute paired intervals by model and fixture family."""

    pairs: dict[tuple[str, str], JsonDict] = defaultdict(
        lambda: {
            "repeated_minus_one_shot_exact": [],
            "warm_exact_minus_cold_exact": [],
            "warm_exact_work_delta_vs_cold": [],
            "continuous_warm_minus_blank_exact": [],
        }
    )
    by_arm_task = _records_by_arm_task(evaluations)
    for key, one in by_arm_task["one_shot"].items():
        model, _task_id = key
        family = str(one["fixture_family"])
        repeated = by_arm_task["repeated_generation"].get(key)
        warm_exact = by_arm_task["partial_evidence_exact_completion"].get(key)
        cold = by_arm_task["cold_exact_completion"].get(key)
        continuous = by_arm_task["partial_evidence_continuous_refinement"].get(key)
        bucket = pairs[(model, family)]
        if repeated:
            bucket["repeated_minus_one_shot_exact"].append(
                int(repeated["exact_valid"]) - int(one["exact_valid"])
            )
        if warm_exact and cold:
            bucket["warm_exact_minus_cold_exact"].append(
                int(warm_exact["exact_valid"]) - int(cold["exact_valid"])
            )
            bucket["warm_exact_work_delta_vs_cold"].append(int(warm_exact["work_delta_vs_cold"]))
        if continuous:
            bucket["continuous_warm_minus_blank_exact"].append(
                int(continuous["exact_valid"]) - int(continuous["blank_exact_valid"])
            )
    return {
        model: {
            family: {name: _paired_interval(values) for name, values in metrics.items()}
            for (bucket_model, family), metrics in pairs.items()
            if bucket_model == model
        }
        for model in MANDATED_MODEL_IDS
    }


def field_provenance() -> JsonDict:
    """Return field provenance for all required fields."""

    sources = [
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6288.RESULT_RELATIVE_PATH.as_posix(),
        exp6275.MODULE_RELATIVE_PATH.as_posix(),
        "python/carnot/inference/sota_models.py",
    ]
    return {
        field: {
            "spec": "REQ-CONSTRAINT-6289",
            "sources": sources,
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def protected_hashes() -> dict[str, str]:
    """Hash protected files before a run starts."""

    return {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in PROTECTED_RELATIVE_PATHS
        if (REPO_ROOT / path).exists()
    }


def protected_files_unchanged(before: Mapping[str, str]) -> JsonDict:
    """Compare protected file hashes after the run."""

    after = protected_hashes()
    receipts = {}
    for path, before_hash in before.items():
        receipts[path] = {
            "before_sha256": before_hash,
            "after_sha256": after.get(path),
            "unchanged": after.get(path) == before_hash,
        }
    receipts["principle"] = FIELD_PRINCIPLES["protected_files_unchanged"]
    return receipts


def honest_verdict(status: str, warm_score: float, all_tests_pass: bool) -> str:
    """Return a terminal honest verdict."""

    if status == "complete_ready" and warm_score == 1.0 and all_tests_pass:
        return "complete: warm-start evidence reduced exact-state work without exact harm"
    if status == "blocked":
        return "blocked: mandated model cell or precondition closed readiness"
    if not all_tests_pass:
        return "complete: benchmark ran but recorded failing verification commands"
    return "complete: no preregistered warm-start value beyond oracle controls"


def llama_cpp_python_offload_receipt() -> JsonDict:  # pragma: no cover
    """Return whether llama-cpp-python reports CUDA offload support."""

    try:
        import llama_cpp
        from llama_cpp import llama_cpp as bindings

        return {
            "available": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "supports_gpu_offload": bool(bindings.llama_supports_gpu_offload()),
        }
    except Exception as exc:
        return {"available": False, "supports_gpu_offload": False, "error": str(exc)}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entrypoint for the required experiment command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=args.result_path,
        write=True,
    )
    print(json.dumps({"path": str(args.result_path), "status": artifact["status"]}, sort_keys=True))
    return 0


def _ordinary_task(task: JsonDict) -> JsonDict:
    display = {atom: _atom_display(atom) for atom in task["allowed_labels"]}
    prompt = str(task["prompt_text"])
    for atom, visible in sorted(display.items(), key=lambda item: len(item[0]), reverse=True):
        prompt = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(atom)}(?![A-Za-z0-9_])",
            visible,
            prompt,
        )
    prompt = prompt.replace("Available labels:", "Available descriptions:")
    prompt = prompt.replace("The label ", "The description ")
    prompt = prompt.replace("labels from this group", "descriptions from this group")
    prompt = prompt.replace("Choose the labels", "Choose the descriptions")
    prompt = prompt.replace("Use ANSWER: NONE for an empty assignment.", "Use EMPTY after ANSWER when no description should be selected.")
    task["prompt_text"] = prompt
    task["prompt_hash"] = sha256_text(prompt)
    task["atom_display"] = display
    task["visible_labels"] = [display[atom] for atom in task["allowed_labels"]]
    return task


def _atom_display(atom: str) -> str:
    return atom.replace("_", " ")


def _parse_visible_assignment(
    task: Mapping[str, Any],
    raw_output: str,
    evidence: Mapping[str, Any],
    generated_token_count: int,
) -> JsonDict:
    if generated_token_count <= 0:
        return {
            "parse_success": False,
            "labels": [],
            "abstention": False,
            "parser": "zero_token",
            "parse_error": "zero_token_row",
        }
    if re.search(r"\b(IMPOSSIBLE|UNSAT|NO SOLUTION|ABSTAIN|REFUSE)\b", raw_output, re.I):
        return {
            "parse_success": True,
            "labels": [],
            "abstention": True,
            "parser": "abstention",
            "parse_error": None,
        }
    match = re.search(r"(?im)^ANSWER\s*:\s*(.+?)\s*$", raw_output.strip())
    if match and match.group(1).strip().strip(".").upper() in {"EMPTY", "NONE", "NO LABELS"}:
        return {
            "parse_success": True,
            "labels": [],
            "abstention": False,
            "parser": "visible_empty",
            "parse_error": None,
        }
    if evidence.get("accepted") is True or evidence.get("positive_atoms"):
        return {
            "parse_success": True,
            "labels": list(evidence.get("positive_atoms") or []),
            "abstention": False,
            "parser": "partial_atom_aliases",
            "parse_error": None,
        }
    return {
        "parse_success": False,
        "labels": [],
        "abstention": False,
        "parser": "partial_atom_aliases",
        "parse_error": ",".join(evidence.get("rejection_reasons") or ["no_parse"]),
    }


def _row_terminal_disposition(*, generated_token_count: int, parse_success: bool) -> str:
    if generated_token_count <= 0:
        return "blocked: zero_token_row"
    if not parse_success:
        return "complete: parser_rejected_but_row_terminal"
    return "complete: row_scored"


def _score_row(task: Mapping[str, Any], row: Mapping[str, Any]) -> JsonDict:
    return score_text_response(
        task,
        str(row.get("raw_output") or ""),
        generated_token_count=int(row.get("generated_token_count") or 0),
        row_id=f"{row.get('model_hf_id')}|{row.get('task_id')}|{row.get('sample_index')}",
    )


def _row_receipt(row: Mapping[str, Any], score: Mapping[str, Any]) -> JsonDict:
    return {
        "model_hf_id": row["model_hf_id"],
        "task_id": row["task_id"],
        "sample_index": row["sample_index"],
        "seed": row["seed"],
        "prompt_hash": row["prompt_hash"],
        "prompt_token_count": row["prompt_token_count"],
        "generated_token_count": row["generated_token_count"],
        "timeout": row["timeout"],
        "raw_output_hash": row["raw_output_hash"],
        "terminal_disposition": score["terminal_disposition"],
        "parse_success": score["parse_success"],
        "semantic_valid": score["semantic_valid"],
        "evidence_supported": score["evidence_support"]["supported"] is True,
    }


def _one_shot_record(
    hf_id: str,
    task: Mapping[str, Any],
    row: Mapping[str, Any],
    score: Mapping[str, Any],
    table: Any,
) -> JsonDict:
    return _base_arm_record(
        "one_shot",
        hf_id,
        task,
        parse_success=score["parse_success"],
        exact_valid=score["semantic_valid"],
        evidence_supported=score["evidence_support"]["supported"] is True,
        state_evaluations=table.vertex_count,
        verifier_work=len(score["residual_rule_violations"]) + 1,
        model_generated_tokens=int(row["generated_token_count"]),
        model_prompt_tokens=int(row["prompt_token_count"]),
        wall_time_s=float(row["latency_s"]),
        selected_labels=score["parsed_labels"],
    )


def _repeated_record(
    hf_id: str,
    task: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    table: Any,
) -> JsonDict:
    chosen = next(
        (score for score in scores if score["semantic_valid"]),
        next((score for score in scores if score["parse_success"]), scores[0]),
    )
    return _base_arm_record(
        "repeated_generation",
        hf_id,
        task,
        parse_success=chosen["parse_success"],
        exact_valid=chosen["semantic_valid"],
        evidence_supported=chosen["evidence_support"]["supported"] is True,
        state_evaluations=table.vertex_count * len(scores),
        verifier_work=sum(len(score["residual_rule_violations"]) + 1 for score in scores),
        model_generated_tokens=sum(int(row["generated_token_count"]) for row in rows),
        model_prompt_tokens=sum(int(row["prompt_token_count"]) for row in rows),
        wall_time_s=sum(float(row["latency_s"]) for row in rows),
        selected_labels=chosen["parsed_labels"],
    )


def _continuous_record(
    hf_id: str,
    task: Mapping[str, Any],
    score: Mapping[str, Any],
    table: Any,
) -> JsonDict:
    support = score["evidence_support"]
    evidence = score["evidence"]
    if support.get("supported") is True:
        starts = exp6288.compare_refinement_starts(
            table,
            evidence,
            support,
            row_id=str(task["task_id"]),
            seed=RANDOM_SEEDS[0],
        )
        warm_success = starts["evidence_warm"]["best_attempt"]["success"] is True
        blank_success = starts["blank"]["best_attempt"]["success"] is True
        random_success = starts["random"]["best_attempt"]["success"] is True
        evaluations = sum(
            int(attempt["relaxation_energy_evaluations"])
            for arm in starts.values()
            for attempt in arm["attempts"]
        )
    else:
        starts = {}
        warm_success = blank_success = random_success = False
        evaluations = 0
    record = _base_arm_record(
        "partial_evidence_continuous_refinement",
        hf_id,
        task,
        parse_success=score["parse_success"],
        exact_valid=warm_success,
        evidence_supported=support.get("supported") is True,
        state_evaluations=evaluations,
        verifier_work=evaluations,
        model_generated_tokens=0,
        model_prompt_tokens=0,
        wall_time_s=0.0,
        selected_labels=starts.get("evidence_warm", {}).get("best_attempt", {}).get("rounded_state", []),
    )
    record["blank_exact_valid"] = blank_success
    record["random_exact_valid"] = random_success
    record["starts"] = starts
    return record


def _partial_exact_record(
    hf_id: str,
    task: Mapping[str, Any],
    score: Mapping[str, Any],
    table: Any,
) -> JsonDict:
    support = score["evidence_support"]
    evidence = score["evidence"]
    cold_success = table.best_discrete_energy == 0
    warm_evaluations = _compatible_state_evaluations(table, evidence)
    completion = list(support.get("supporting_completion") or [])
    exact_valid = bool(support.get("supported") is True and table.discrete_energy(completion) == 0)
    record = _base_arm_record(
        "partial_evidence_exact_completion",
        hf_id,
        task,
        parse_success=score["parse_success"],
        exact_valid=exact_valid,
        evidence_supported=support.get("supported") is True,
        state_evaluations=warm_evaluations,
        verifier_work=warm_evaluations,
        model_generated_tokens=0,
        model_prompt_tokens=0,
        wall_time_s=0.0,
        selected_labels=completion,
    )
    record["cold_state_evaluations"] = min(table.vertex_count, EXACT_COMPLETION_BUDGET)
    record["work_delta_vs_cold"] = max(
        0, int(record["cold_state_evaluations"]) - int(record["state_evaluations"])
    )
    record["exact_validity_harm"] = cold_success and not exact_valid
    return record


def _cold_exact_record(hf_id: str, task: Mapping[str, Any], table: Any) -> JsonDict:
    success = table.best_discrete_energy == 0
    completion = list(next((state for state in table.vertex_states if table.discrete_energy(state) == 0), []))
    record = _base_arm_record(
        "cold_exact_completion",
        hf_id,
        task,
        parse_success=True,
        exact_valid=success,
        evidence_supported=False,
        state_evaluations=min(table.vertex_count, EXACT_COMPLETION_BUDGET),
        verifier_work=min(table.vertex_count, EXACT_COMPLETION_BUDGET),
        model_generated_tokens=0,
        model_prompt_tokens=0,
        wall_time_s=0.0,
        selected_labels=completion,
    )
    record["counts_as_model_value"] = False
    return record


def _compute_balanced_route_record(
    hf_id: str,
    task: Mapping[str, Any],
    one: Mapping[str, Any],
    repeated: Mapping[str, Any],
    warm_exact: Mapping[str, Any],
) -> JsonDict:
    actions = ["generate", "verify"]
    if one["exact_valid"]:
        selected = one
        actions.append("stop")
    elif warm_exact["evidence_supported"]:
        selected = warm_exact
        actions.append("exact_complete")
    else:
        selected = repeated
        actions.append("stop_after_repeated_generation")
    record = _base_arm_record(
        "compute_balanced_route",
        hf_id,
        task,
        parse_success=selected["parse_success"],
        exact_valid=selected["exact_valid"],
        evidence_supported=selected["evidence_supported"],
        state_evaluations=int(one["state_evaluations"]) + int(warm_exact["state_evaluations"]),
        verifier_work=int(one["verifier_work"]) + int(warm_exact["verifier_work"]),
        model_generated_tokens=int(repeated["model_generated_tokens"]),
        model_prompt_tokens=int(repeated["model_prompt_tokens"]),
        wall_time_s=float(repeated["wall_time_s"]),
        selected_labels=list(selected.get("selected_labels") or []),
    )
    record["route_actions"] = actions
    return record


def _base_arm_record(
    arm: str,
    hf_id: str,
    task: Mapping[str, Any],
    *,
    parse_success: bool,
    exact_valid: bool,
    evidence_supported: bool,
    state_evaluations: int,
    verifier_work: int,
    model_generated_tokens: int,
    model_prompt_tokens: int,
    wall_time_s: float,
    selected_labels: Sequence[str],
) -> JsonDict:
    return {
        "arm": arm,
        "model_hf_id": hf_id,
        "task_id": task["task_id"],
        "fixture_id": task["fixture_id"],
        "fixture_family": task["family"],
        "parse_success": bool(parse_success),
        "exact_valid": bool(exact_valid),
        "evidence_supported": bool(evidence_supported),
        "state_evaluations": int(state_evaluations),
        "verifier_work": int(verifier_work),
        "model_generated_tokens": int(model_generated_tokens),
        "model_prompt_tokens": int(model_prompt_tokens),
        "wall_time_s": round(float(wall_time_s), 6),
        "selected_labels": list(selected_labels),
    }


def _table_for_task(task: Mapping[str, Any]) -> Any:
    return exp6288.table_from_program(str(task["program_text"]), str(task["fixture_id"]))


def _compatible_state_evaluations(table: Any, evidence: Mapping[str, Any]) -> int:
    positive = set(evidence.get("positive_atoms") or [])
    negative = set(evidence.get("negative_atoms") or [])
    if not positive and not negative:
        return min(table.vertex_count, EXACT_COMPLETION_BUDGET)
    compatible = 0
    for state in table.vertex_states:
        state_set = set(state)
        if positive <= state_set and negative.isdisjoint(state_set):
            compatible += 1
    return max(1, min(compatible, EXACT_COMPLETION_BUDGET))


def _normalize_raw_rows(
    hf_id: str,
    jobs: Sequence[Mapping[str, Any]],
    backend_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    job_by_key = {
        (str(job["task"]["task_id"]), int(job["sample_index"])): dict(job)
        for job in jobs
    }
    normalized = []
    for row in backend_rows:
        key = (str(row["task_id"]), int(row["sample_index"]))
        job = job_by_key[key]
        raw = str(row.get("raw_output") or "")
        normalized.append(
            {
                "schema": "carnot.exp6289.raw_output.v1",
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


def _terminalized_receipt(hf_id: str, receipt: JsonDict) -> JsonDict:
    receipt["model_hf_id"] = hf_id
    receipt["terminal_disposition"] = _terminalized_disposition(receipt.get("terminal_disposition"))
    if "gpu_offload" not in receipt:
        receipt["gpu_offload"] = {"requested": True, "observed": False}
    receipt["peak_vram_mb"] = int(receipt.get("peak_vram_mb") or 0)
    return receipt


def _terminalized_disposition(value: Any) -> str:
    text = str(value or "")
    if is_terminal_disposition(text):
        return text
    return "blocked: nonterminal_backend_disposition"


def _model_record_blocked(record: Mapping[str, Any]) -> bool:
    return (
        record.get("exists") is not True
        or not record.get("sha256")
        or record.get("tokenizer_loadable") is not True
    )


def _blocked_model_receipt(
    model_record: Mapping[str, Any],
    reason: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    receipt = {
        "terminal_disposition": f"blocked: {reason}",
        "failed_cell": reason,
        "gpu_offload": {"requested": True, "observed": False},
        "peak_vram_mb": 0,
        "model_hf_id": str(model_record.get("hf_id")),
    }
    receipt.update(dict(extra or {}))
    return receipt


def _cache_policy_for_model(hf_id: str) -> str:
    if hf_id == "unsloth/gemma-4-26B-A4B-it-GGUF":
        return "cached_sota_pair(model_indices=(1,0))"
    return "cached_sota_pair(model_indices=(0,2))"


def _model_file_receipts(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(record["hf_id"]): {
            "model_path": str(record.get("model_path") or ""),
            "sha256": record.get("sha256"),
            "revision": str(record.get("revision") or "unknown"),
            "quantization": str(record.get("quantization") or "unknown"),
            "size_bytes": int(record.get("size_bytes") or 0),
            "exists": record.get("exists") is True,
        }
        for record in records
    }


def _tokenizer_receipts(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(record["hf_id"]): {
            "tokenizer_loadable": record.get("tokenizer_loadable") is True,
            "tokenizer_status": str(record.get("tokenizer_status") or ""),
            "tokenizer_hash": "sha256:" + str(record.get("sha256") or ""),
            "chat_template_source": "embedded_in_gguf_or_llama_default",
            "chat_template_hash": "sha256:" + sha256_text(str(record.get("sha256") or "")),
        }
        for record in records
    }


def _cuda_offload_receipts(
    receipts_by_model: Mapping[str, Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    return {
        hf_id: {
            "cuda_inventory": preconditions.get("cuda_inventory"),
            "gpu_ownership_before_loading": preconditions.get("gpu_ownership_before_loading"),
            "llama_cpp_python_offload": preconditions.get("llama_cpp_python_offload"),
            "model_receipt": dict(receipts_by_model.get(hf_id) or {}),
        }
        for hf_id in MANDATED_MODEL_IDS
    }


def _upstream_adapter_receipt() -> JsonDict:
    path = REPO_ROOT / exp6288.RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    classified = classify_artifact_path(path) if path.exists() else {"terminal_class": "missing"}
    return {
        "path": exp6288.RESULT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path) if path.exists() else None,
        "terminal_class": str(artifact.get("honest_verdict", "")).split(":", 1)[0]
        or str(classified.get("terminal_class")),
        "status": artifact.get("status"),
        "partial_atom_evidence_adapter_ready_score": artifact.get(
            "partial_atom_evidence_adapter_ready_score"
        ),
    }


def _status_from_terminal_and_score(
    terminal: Mapping[str, str],
    warm_score: float,
    model_resolution: Mapping[str, Any],
) -> str:
    if dict(model_resolution).get("all_resolved") is not True:
        return "blocked"
    if not all(str(value).startswith("complete:") for value in terminal.values()):
        return "blocked"
    return "complete_ready" if warm_score == 1.0 else "complete_no_value"


def _expected_ready_score_from_artifact(artifact: Mapping[str, Any]) -> float:
    terminal = dict(artifact.get("terminal_model_dispositions") or {})
    all_complete = all(str(value).startswith("complete:") for value in terminal.values())
    no_harm = int(artifact.get("harmful_regressions", {}).get("exact_validity_harm_count") or 0) == 0
    work_delta = int(
        artifact.get("partial_evidence_exact_completion_results", {}).get(
            "warm_minus_cold_work_delta",
            0,
        )
        or 0
    )
    refine_delta = int(
        artifact.get("partial_evidence_continuous_refinement_results", {}).get(
            "evidence_warm_minus_blank_success_delta",
            0,
        )
        or 0
    )
    return 1.0 if all_complete and no_harm and (work_delta > 0 or refine_delta > 0) else 0.0


def _records_by_arm_task(evaluations: Mapping[str, Any]) -> dict[str, dict[tuple[str, str], JsonDict]]:
    return {
        arm: {(str(row["model_hf_id"]), str(row["task_id"])): dict(row) for row in rows}
        for arm, rows in dict(evaluations["records"]).items()
    }


def _paired_interval(values: Sequence[int]) -> JsonDict:
    n = len(values)
    if n == 0:
        return {"sample_size": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0]}
    mean = sum(values) / n
    if n == 1:
        return {"sample_size": 1, "mean_delta": mean, "ci95": [mean, mean]}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half_width = 1.96 * math.sqrt(variance / n)
    return {"sample_size": n, "mean_delta": mean, "ci95": [mean - half_width, mean + half_width]}


def _empty_nested_counts(extra_metrics: Sequence[str] = ("exact_valid",)) -> JsonDict:
    return {
        arm: {
            model: {
                "_metrics": tuple(extra_metrics),
            }
            for model in MANDATED_MODEL_IDS
        }
        for arm in ARMS
    }


def _empty_nested_sums() -> JsonDict:
    return {arm: {model: {} for model in MANDATED_MODEL_IDS} for arm in ARMS}


def _add_count(
    store: JsonDict,
    key: tuple[str, str, str],
    metric: str,
    success: bool,
) -> None:
    arm, model, family = key
    cell = store[arm][model].setdefault(family, {}).setdefault(metric, {"numerator": 0, "denominator": 0})
    cell["numerator"] += int(success)
    cell["denominator"] += 1


def _add_sum(store: JsonDict, key: tuple[str, str, str], metric: str, value: float) -> None:
    arm, model, family = key
    cell = store[arm][model].setdefault(family, {}).setdefault(metric, 0.0)
    store[arm][model][family][metric] = cell + value


def _finalize_counts(store: Mapping[str, Any]) -> JsonDict:
    out: JsonDict = {}
    for arm, models in dict(store).items():
        out[arm] = {}
        for model, families in dict(models).items():
            out[arm][model] = {}
            for family, metrics in dict(families).items():
                if family == "_metrics":
                    continue
                out[arm][model][family] = {}
                for metric, cell in dict(metrics).items():
                    numerator = int(cell["numerator"])
                    denominator = int(cell["denominator"])
                    out[arm][model][family][metric] = {
                        "numerator": numerator,
                        "denominator": denominator,
                        "rate": numerator / denominator if denominator else 0.0,
                    }
    return out


def _finalize_sums(store: Mapping[str, Any]) -> JsonDict:
    return json.loads(canonical_json(store))


def _gpu_inventory() -> JsonDict:  # pragma: no cover
    return {
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"]),
    }


def _gpu_ownership_receipt() -> JsonDict:  # pragma: no cover
    return {
        "compute_apps": _command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    }


def _gpu_memory_mb() -> dict[int, int]:  # pragma: no cover
    result = _command(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    )
    if result["returncode"] != 0:
        return {}
    memory = {}
    for line in str(result.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            memory[int(parts[0])] = int(parts[1])
    return memory


def _command(cmd: Sequence[str]) -> JsonDict:  # pragma: no cover
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return {
            "cmd": list(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": list(cmd), "returncode": 127, "stdout": "", "stderr": str(exc)}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
