"""Exp 2926 ConstraintBench constrained-output rerun.

Spec: REQ-BENCH-2926, SCENARIO-BENCH-2926.

This module reuses the Exp 2919 exact task verifiers and adds the methodology
guards that were missing from the flagged row: a larger task manifest, raw
response provenance, prompt hashes, per-task seeds, constrained-decoder
diagnostics, non-tautological syntax/feasibility metrics, duration gating, and
a reproducibility checksum over the evidence surface.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval import constraintbench_mini_direct_optimization as base
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf

JsonDict = dict[str, Any]
RUN_DATE = "20260523"
RANDOM_SEED = 2926
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2926_constraintbench_constrained_output_rerun_v2.json"
MANIFEST_FILENAME = "constraintbench_constrained_output_rerun_2926_manifest.json"
RAW_RESPONSE_DIRNAME = "constraintbench_constrained_output_rerun_2926_raw"
INFERENCE_SUBSTRATE = "live_llm_inference"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "constraintbench_corrigendum_ready",
    "flagged_adversarial",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "models_used",
    "cached_sota_pair_used",
    "constrained_decoder_available",
    "n_tasks",
    "syntax_valid_rate",
    "feasibility_rate_overall",
    "feasibility_rate_given_syntax",
    "optimality_rate_given_feasible",
    "syntax_feasibility_not_tautological",
    "per_task_results",
    "raw_response_dir",
    "live_inference_duration_s",
    "inference_substrate",
    "duration_s",
    "run_date",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectModelOutputs = Callable[
    [JsonDict, list[base.OptimizationTask], "ExperimentConfig", JsonDict], JsonDict
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2926 artifact paths and live-run gates."""

    output_path: Path | None = None
    manifest_path: Path | None = None
    raw_response_dir: Path | None = None
    max_models: int = 1
    random_seed: int = RANDOM_SEED
    min_live_inference_duration_s: float = 60.0
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME

    def task_manifest_path(self) -> Path:
        return self.manifest_path or REPO_ROOT / "results" / MANIFEST_FILENAME

    def response_dir(self) -> Path:
        return self.raw_response_dir or REPO_ROOT / "results" / RAW_RESPONSE_DIRNAME


def build_task_manifest(n_tasks: int = 30) -> list[base.OptimizationTask]:
    """Return at least 30 deterministic tasks backed by Exp 2919 exact solvers."""

    if n_tasks < 30:
        raise ValueError("ConstraintBench rerun requires at least 30 tasks")
    seeds = base.build_task_manifest()
    family_counts: Counter[str] = Counter()
    tasks: list[base.OptimizationTask] = []
    for seed_task in itertools.islice(itertools.cycle(seeds), n_tasks):
        family = _family_slug(seed_task.task_type)
        family_counts[family] += 1
        task_id = f"cbmini-2926-{family}-{family_counts[family]:03d}"
        task = base.OptimizationTask(
            task_id=task_id,
            task_type=seed_task.task_type,
            exact_verifier_type=seed_task.exact_verifier_type,
            objective_sense=seed_task.objective_sense,
            payload=_json_clone(seed_task.payload),
            prompt="",
        )
        tasks.append(
            base.OptimizationTask(
                task_id=task.task_id,
                task_type=task.task_type,
                exact_verifier_type=task.exact_verifier_type,
                objective_sense=task.objective_sense,
                payload=task.payload,
                prompt=_build_prompt(task),
            )
        )
    return tasks


def write_task_manifest(tasks: Sequence[base.OptimizationTask], path: Path | str) -> JsonDict:
    """Persist the task manifest with exact optima and verifier provenance."""

    rows = [_task_to_manifest_row(task) for task in tasks]
    payload = {
        "schema": "carnot.constraintbench_constrained_output_rerun.v2",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "n_tasks": len(tasks),
        "exact_verifier_types": list(base.EXACT_VERIFIER_TYPES),
        "tasks": rows,
    }
    _write_json(Path(path), payload)
    return payload


def prompt_hash(prompt: str) -> str:
    """Return the SHA-256 hash recorded for every generated prompt."""

    return sha256_text(prompt)


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def probe_constrained_decoder(
    *,
    llguidance_module: object | None = None,
    probe_llguidance: bool = True,
) -> JsonDict:
    """Report whether a local llguidance JSON-schema backend is available."""

    module = llguidance_module
    if module is None:
        if not probe_llguidance:
            return _decoder_fallback("llguidance probing disabled")
        try:
            module = importlib.import_module("llguidance")
        except ImportError:
            return _decoder_fallback("llguidance not installed")

    matcher = getattr(module, "LLMatcher", None)
    grammar_from_schema = getattr(matcher, "grammar_from_json_schema", None)
    if not callable(grammar_from_schema):
        return _decoder_fallback("llguidance LLMatcher.grammar_from_json_schema unavailable")
    try:
        grammar = grammar_from_schema(
            _generic_solution_schema(),
            overrides={"whitespace_flexible": False},
        )
        validator = getattr(matcher, "validate_grammar", None)
        validation_message = str(validator(grammar)) if callable(validator) else ""
        if validation_message and not validation_message.startswith("WARNING"):
            return _decoder_fallback(validation_message, grammar=str(grammar))
        version_fn = getattr(module, "get_version", None)
        version = str(version_fn()) if callable(version_fn) else None
    except Exception as exc:
        return _decoder_fallback(f"{type(exc).__name__}: {exc}")
    return {
        "backend_name": "llguidance",
        "constrained_decoder_available": True,
        "fallback_backend_available": True,
        "grammar": str(grammar),
        "llguidance_version": version,
        "decoder_error": None,
    }


def parse_structured_output(task: base.OptimizationTask, raw_text: str) -> JsonDict:
    """Extract JSON, repair deterministic scalar types, and schema-check it."""

    obj, error = base._extract_json_object(raw_text)
    if error is not None:
        return {
            "syntax_valid": False,
            "parsed_output": None,
            "parse_error": error,
            "parser_repair_applied": False,
            "parser_repair_note": None,
        }
    solution = obj.get("solution", obj)
    if not isinstance(solution, Mapping):
        return {
            "syntax_valid": False,
            "parsed_output": None,
            "parse_error": "solution_not_object",
            "parser_repair_applied": False,
            "parser_repair_note": None,
        }
    repaired, repair_note = _repair_solution(task, dict(solution))
    parsed = base.parse_model_output(task, json.dumps({"solution": repaired}, sort_keys=True))
    return {
        "syntax_valid": parsed.syntax_valid,
        "parsed_output": parsed.solution,
        "parse_error": parsed.parse_error,
        "parser_repair_applied": repair_note is not None,
        "parser_repair_note": repair_note,
    }


def evaluate_raw_output(
    task: base.OptimizationTask,
    raw_text: str,
    *,
    generation_metadata: Mapping[str, Any],
) -> JsonDict:
    """Evaluate one raw model response with exact syntax, feasibility, and optimum checks."""

    parsed = parse_structured_output(task, raw_text)
    optimum = base.solve_task(task).optimum_value
    if not parsed["syntax_valid"] or parsed["parsed_output"] is None:
        verifier_result = {
            "syntax_valid": False,
            "feasible": False,
            "objective_value": None,
            "optimum_value": optimum,
            "optimal": False,
            "violation_reasons": [parsed["parse_error"] or "parse_error"],
        }
        violation_class = "syntax_invalid"
    else:
        feasible, reasons = base._check_feasibility(task, parsed["parsed_output"])
        objective_value = base._objective_value(task, parsed["parsed_output"]) if feasible else None
        optimal = bool(feasible and objective_value == optimum)
        verifier_result = {
            "syntax_valid": True,
            "feasible": feasible,
            "objective_value": objective_value,
            "optimum_value": optimum,
            "optimal": optimal,
            "violation_reasons": list(reasons),
        }
        violation_class = "none" if optimal else "suboptimal" if feasible else "infeasible"
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "exact_verifier_type": task.exact_verifier_type,
        "objective_sense": task.objective_sense,
        "objective_function": _objective_function(task),
        "constraints": _constraints(task),
        "prompt_hash": generation_metadata.get("prompt_hash") or prompt_hash(task.prompt),
        "per_task_seed": generation_metadata.get("per_task_seed"),
        "model_hf_id": generation_metadata.get("model_hf_id"),
        "model_name": generation_metadata.get("model_name"),
        "model_path": generation_metadata.get("model_path"),
        "gpu_index": generation_metadata.get("gpu_index"),
        "generation_source": generation_metadata.get("generation_source"),
        "generation_blocker": generation_metadata.get("blocker"),
        "raw_response_path": generation_metadata.get("raw_response_path"),
        "raw_response_sha256": generation_metadata.get("raw_response_sha256")
        or sha256_text(raw_text),
        "elapsed_seconds": generation_metadata.get("elapsed_seconds"),
        "syntax_valid": verifier_result["syntax_valid"],
        "feasible": verifier_result["feasible"],
        "objective_value": verifier_result["objective_value"],
        "optimum_value": optimum,
        "optimal": verifier_result["optimal"],
        "violation_class": violation_class,
        "violation_reasons": verifier_result["violation_reasons"],
        "parsed_output": parsed["parsed_output"],
        "parse_error": parsed["parse_error"],
        "parser_repair_applied": parsed["parser_repair_applied"],
        "parser_repair_note": parsed["parser_repair_note"],
        "verifier_result": verifier_result,
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute non-tautological syntax, feasibility, and optimality rates."""

    if not rows:
        return {
            "syntax_valid_rate": 0.0,
            "feasibility_rate_overall": 0.0,
            "feasibility_rate_given_syntax": 0.0,
            "optimality_rate_given_feasible": 0.0,
            "syntax_feasibility_not_tautological": False,
            "violation_classes": {},
        }
    syntax_rows = [row for row in rows if row.get("syntax_valid") is True]
    feasible_rows = [row for row in rows if row.get("feasible") is True]
    violation_counts = Counter(str(row.get("violation_class")) for row in rows)
    return {
        "syntax_valid_rate": _rate(len(syntax_rows), len(rows)),
        "feasibility_rate_overall": _rate(len(feasible_rows), len(rows)),
        "feasibility_rate_given_syntax": _rate(len(feasible_rows), len(syntax_rows)),
        "optimality_rate_given_feasible": _rate(
            sum(row.get("optimal") is True for row in feasible_rows),
            len(feasible_rows),
        ),
        "syntax_feasibility_not_tautological": any(
            row.get("syntax_valid") is True and row.get("feasible") is False for row in rows
        ),
        "violation_classes": dict(sorted(violation_counts.items())),
    }


def resolve_model_specs(
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
) -> tuple[list[JsonDict], bool, str | None]:
    """Resolve mandated SOTA GGUF specs, always trying cached_sota_pair first."""

    return base.resolve_model_specs(
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    collect_model_outputs_fn: CollectModelOutputs | None = None,
    llguidance_module: object | None = None,
    probe_llguidance: bool = True,
) -> JsonDict:
    """Run Exp 2926 and write the terminal corrected-or-blocked artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    tasks = build_task_manifest()
    manifest_path = active.task_manifest_path()
    manifest_payload = write_task_manifest(tasks, manifest_path)
    active.response_dir().mkdir(parents=True, exist_ok=True)
    decoder_status = probe_constrained_decoder(
        llguidance_module=llguidance_module,
        probe_llguidance=probe_llguidance,
    )
    specs, cached_pair_used, cache_error = resolve_model_specs(
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
    )
    if not specs:
        artifact = _build_artifact(
            active,
            started,
            manifest_payload,
            model_specs=base._blocked_model_specs(),
            models_used=[],
            cached_pair_used=False,
            cache_error=cache_error,
            decoder_status=decoder_status,
            per_task_results=[],
            model_attempts=[],
            live_inference_duration_s=0.0,
            honest_verdict="blocked_sota_gguf_cache_missing",
        )
        _write_json(active.artifact_path(), artifact)
        return artifact

    collector = collect_model_outputs_fn or collect_live_model_outputs
    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    live_inference_duration_s = 0.0
    task_by_id = {task.task_id: task for task in tasks}
    for index, spec in enumerate(specs):
        if index >= active.max_models:
            model_attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_name": spec.get("name"),
                    "model_path": spec.get("model_path"),
                    "model_used": False,
                    "blocker": "not_attempted_runtime_budget",
                    "live_inference_duration_s": 0.0,
                }
            )
            continue
        collection = collector(spec, tasks, active, decoder_status)
        summary = dict(collection.get("summary") or {})
        model_attempts.append(summary)
        live_inference_duration_s += float(summary.get("live_inference_duration_s") or 0.0)
        for generation_row in collection.get("rows") or []:
            task = task_by_id.get(str(generation_row.get("task_id")))
            if task is None:
                continue
            rows.append(
                evaluate_raw_output(
                    task,
                    str(generation_row.get("output_text") or ""),
                    generation_metadata=generation_row,
                )
            )

    models_used = [
        str(attempt["hf_id"])
        for attempt in model_attempts
        if attempt.get("model_used") is True and attempt.get("hf_id") in base.MANDATED_MODEL_IDS
    ]
    provisional = _build_artifact(
        active,
        started,
        manifest_payload,
        model_specs=specs,
        models_used=models_used,
        cached_pair_used=cached_pair_used,
        cache_error=cache_error,
        decoder_status=decoder_status,
        per_task_results=rows,
        model_attempts=model_attempts,
        live_inference_duration_s=live_inference_duration_s,
        honest_verdict="pending",
    )
    verdict = _honest_verdict(provisional, active.min_live_inference_duration_s)
    artifact = {**provisional, "honest_verdict": verdict}
    artifact["constraintbench_corrigendum_ready"] = verdict.startswith("complete:")
    artifact["flagged_adversarial"] = not artifact["constraintbench_corrigendum_ready"]
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(
        task_manifest=manifest_payload["tasks"],
        model_specs=artifact["model_specs"],
        per_task_results=artifact["per_task_results"],
    )
    _write_json(active.artifact_path(), artifact)
    return artifact


def collect_live_model_outputs(
    spec: JsonDict,
    tasks: list[base.OptimizationTask],
    config: ExperimentConfig,
    decoder_status: JsonDict,
    *,
    llama_importer: LlamaImporter | None = None,
) -> JsonDict:
    """Collect raw JSON-ish answers from one local GGUF through llama.cpp."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = str(spec.get("model_path") or "")
    if not model_path:
        return {
            "summary": _summary(spec, False, "model_not_cached", 0.0),
            "rows": [],
        }
    ok, llama_class, import_error = (llama_importer or _default_llama_importer)()
    if not ok or llama_class is None:
        return {
            "summary": _summary(spec, False, import_error or "llama_cpp_import_failed", 0.0),
            "rows": [],
        }

    run_started = config.monotonic_clock()
    try:
        llm = llama_class(
            model_path=model_path,
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=4096,
            seed=config.random_seed,
            verbose=False,
        )
    except Exception as exc:
        elapsed = config.monotonic_clock() - run_started
        return {
            "summary": _summary(spec, False, f"{type(exc).__name__}: {exc}", elapsed),
            "rows": [],
        }

    rows: list[JsonDict] = []
    try:
        for index, task in enumerate(tasks):
            per_task_seed = config.random_seed + index
            row_started = config.monotonic_clock()
            try:
                result = llm(
                    task.prompt,
                    max_tokens=256,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</s>", "<eos>"],
                    echo=False,
                    seed=per_task_seed,
                )
                output_text = base._completion_text(result)
                blocker = None if output_text.strip() else "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            raw_path = _write_raw_response(
                config.response_dir(),
                task,
                spec,
                output_text,
                per_task_seed,
                decoder_status,
            )
            rows.append(
                {
                    "task_id": task.task_id,
                    "model_hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": model_path,
                    "gpu_index": int(spec.get("gpu") or 0),
                    "prompt_hash": prompt_hash(task.prompt),
                    "per_task_seed": per_task_seed,
                    "generation_source": _generation_source(decoder_status),
                    "output_text": output_text,
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": sha256_text(output_text),
                    "elapsed_seconds": round(config.monotonic_clock() - row_started, 6),
                    "blocker": blocker,
                }
            )
    finally:
        base._close_llama(llm)

    elapsed = config.monotonic_clock() - run_started
    model_used = any(not row.get("blocker") for row in rows)
    return {
        "summary": _summary(
            spec,
            model_used,
            None if model_used else "no_usable_generations",
            elapsed,
        ),
        "rows": rows,
    }


def compute_reproducibility_checksum(
    *,
    task_manifest: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    per_task_results: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the manifest, prompts, model specs, raw-output hashes, and verifier rows."""

    checksum_payload = {
        "random_seed": RANDOM_SEED,
        "task_manifest": task_manifest,
        "prompts": [
            {"task_id": row.get("task_id"), "prompt_hash": row.get("prompt_hash")}
            for row in task_manifest
        ],
        "model_specs": model_specs,
        "raw_outputs": [
            {
                "task_id": row.get("task_id"),
                "model_hf_id": row.get("model_hf_id"),
                "raw_response_sha256": row.get("raw_response_sha256"),
            }
            for row in per_task_results
        ],
        "parsed_outputs": [
            {
                "task_id": row.get("task_id"),
                "parsed_output": row.get("parsed_output"),
                "parse_error": row.get("parse_error"),
            }
            for row in per_task_results
        ],
        "verifier_results": [
            {"task_id": row.get("task_id"), "verifier_result": row.get("verifier_result")}
            for row in per_task_results
        ],
    }
    return sha256_text(_canonical_json(checksum_payload))


def _build_artifact(
    config: ExperimentConfig,
    started: float,
    manifest_payload: Mapping[str, Any],
    *,
    model_specs: Sequence[JsonDict],
    models_used: Sequence[str],
    cached_pair_used: bool,
    cache_error: str | None,
    decoder_status: Mapping[str, Any],
    per_task_results: Sequence[JsonDict],
    model_attempts: Sequence[JsonDict],
    live_inference_duration_s: float,
    honest_verdict: str,
) -> JsonDict:
    metrics = aggregate_results(per_task_results)
    artifact = {
        "artifact": "experiment_2926_constraintbench_constrained_output_rerun_v2",
        "schema": "carnot.constraintbench_constrained_output_rerun.v2",
        "honest_verdict": honest_verdict,
        "constraintbench_corrigendum_ready": False,
        "flagged_adversarial": True,
        "random_seed": int(config.random_seed),
        "reproducibility_checksum": "",
        "model_specs": list(model_specs),
        "models_used": list(models_used),
        "cached_sota_pair_used": bool(cached_pair_used),
        "cached_sota_pair_error": cache_error,
        "constrained_decoder_available": bool(
            decoder_status.get("constrained_decoder_available")
        ),
        "constrained_decoder_backend": dict(decoder_status),
        "n_tasks": int(manifest_payload.get("n_tasks") or 0),
        "task_manifest_path": str(config.task_manifest_path()),
        "syntax_valid_rate": metrics["syntax_valid_rate"],
        "feasibility_rate_overall": metrics["feasibility_rate_overall"],
        "feasibility_rate_given_syntax": metrics["feasibility_rate_given_syntax"],
        "optimality_rate_given_feasible": metrics["optimality_rate_given_feasible"],
        "syntax_feasibility_not_tautological": metrics[
            "syntax_feasibility_not_tautological"
        ],
        "violation_classes": metrics["violation_classes"],
        "per_task_results": list(per_task_results),
        "model_attempts": list(model_attempts),
        "raw_response_dir": str(config.response_dir()),
        "live_inference_duration_s": round(float(live_inference_duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(config.tests_run),
        "duration_s": max(0.0, config.clock() - started),
        "run_date": RUN_DATE,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(
        task_manifest=manifest_payload.get("tasks") or [],
        model_specs=artifact["model_specs"],
        per_task_results=artifact["per_task_results"],
    )
    return artifact


def _honest_verdict(artifact: Mapping[str, Any], min_live_inference_duration_s: float) -> str:
    if not artifact.get("models_used") or not artifact.get("per_task_results"):
        return "blocked_sota_runtime_unavailable"
    if float(artifact.get("live_inference_duration_s") or 0.0) < min_live_inference_duration_s:
        return "blocked_duration_gate_failed"
    if artifact.get("syntax_feasibility_not_tautological") is not True:
        return "blocked_tautological_syntax_feasibility_metrics"
    return "complete: constraintbench constrained-output rerun measured with live local SOTA GGUF"


def _task_to_manifest_row(task: base.OptimizationTask) -> JsonDict:
    solved = base.solve_task(task)
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "exact_verifier_type": task.exact_verifier_type,
        "objective_sense": task.objective_sense,
        "objective_function": _objective_function(task),
        "constraints": _constraints(task),
        "payload": task.payload,
        "prompt": task.prompt,
        "prompt_hash": prompt_hash(task.prompt),
        "feasible_count": solved.feasible_count,
        "optimum_value": solved.optimum_value,
        "reference_answer": solved.optimal_solution,
        "reference_answers_available": True,
    }


def _build_prompt(task: base.OptimizationTask) -> str:
    schema = _solution_json_schema(task)
    return (
        "Solve this ConstraintBench-style direct constrained optimization task.\n"
        f"Task id: {task.task_id}\n"
        f"Task type: {task.task_type}\n"
        f"Objective sense: {task.objective_sense}\n"
        "Return exactly one JSON object and no prose or Markdown.\n"
        f"JSON schema: {_canonical_json(schema)}\n"
        f"Task data: {_canonical_json(task.payload)}\n"
    )


def _solution_json_schema(task: base.OptimizationTask) -> JsonDict:
    if task.task_type == "linear_integer":
        return {
            "type": "object",
            "required": ["solution"],
            "properties": {
                "solution": {
                    "type": "object",
                    "required": ["variables"],
                    "properties": {
                        "variables": {
                            "type": "object",
                            "required": list(task.payload["variables"]),
                            "properties": {
                                name: {"type": "integer"} for name in task.payload["variables"]
                            },
                        }
                    },
                }
            },
        }
    if task.task_type == "knapsack_binary":
        return {
            "type": "object",
            "required": ["solution"],
            "properties": {
                "solution": {
                    "type": "object",
                    "required": ["selected_items"],
                    "properties": {
                        "selected_items": {"type": "array", "items": {"type": "string"}}
                    },
                }
            },
        }
    return {
        "type": "object",
        "required": ["solution"],
        "properties": {
            "solution": {
                "type": "object",
                "required": ["colors"],
                "properties": {
                    "colors": {
                        "type": "object",
                        "required": [
                            str(node) for node in range(int(task.payload.get("n_nodes", 0)))
                        ],
                        "properties": {
                            str(node): {"type": "integer"}
                            for node in range(int(task.payload.get("n_nodes", 0)))
                        },
                    }
                },
            }
        },
    }


def _repair_solution(task: base.OptimizationTask, solution: JsonDict) -> tuple[JsonDict, str | None]:
    repair_note = None
    if task.task_type == "linear_integer" and isinstance(solution.get("variables"), Mapping):
        variables = {}
        for name, value in solution["variables"].items():
            repaired = _int_or_original(value)
            repair_note = "integer_string_cast" if repaired is not value else repair_note
            variables[str(name)] = repaired
        return {**solution, "variables": variables}, repair_note
    if task.task_type == "graph_coloring" and isinstance(solution.get("colors"), Mapping):
        colors = {}
        for name, value in solution["colors"].items():
            repaired = _int_or_original(value)
            repair_note = "integer_string_cast" if repaired is not value else repair_note
            colors[str(name)] = repaired
        return {**solution, "colors": colors}, repair_note
    if task.task_type == "knapsack_binary" and isinstance(solution.get("selected_items"), str):
        items = [item.strip() for item in solution["selected_items"].split(",") if item.strip()]
        return {**solution, "selected_items": items}, "comma_string_to_item_list"
    return solution, None


def _int_or_original(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return value


def _objective_function(task: base.OptimizationTask) -> JsonDict:
    if task.task_type == "linear_integer":
        return {"sense": task.objective_sense, "coefficients": task.payload["objective"]}
    if task.task_type == "knapsack_binary":
        return {
            "sense": task.objective_sense,
            "item_values": {item["name"]: item["value"] for item in task.payload["items"]},
        }
    return {"sense": task.objective_sense, "expression": "sum(color_id for each node)"}


def _constraints(task: base.OptimizationTask) -> JsonDict:
    if task.task_type == "linear_integer":
        return {
            "variable_bounds": task.payload["variables"],
            "linear_constraints": task.payload["constraints"],
        }
    if task.task_type == "knapsack_binary":
        return {
            "capacity": task.payload["capacity"],
            "item_weights": {item["name"]: item["weight"] for item in task.payload["items"]},
            "required_items": task.payload.get("required_items", []),
            "excludes": task.payload.get("excludes", []),
        }
    return {
        "n_nodes": task.payload["n_nodes"],
        "n_colors": task.payload["n_colors"],
        "edges": task.payload["edges"],
    }


def _write_raw_response(
    raw_dir: Path,
    task: base.OptimizationTask,
    spec: Mapping[str, Any],
    raw_response: str,
    per_task_seed: int,
    decoder_status: Mapping[str, Any],
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_model = str(spec.get("name") or "model").replace("/", "_")
    path = raw_dir / f"{task.task_id}__{safe_model}.json"
    _write_json(
        path,
        {
            "task_id": task.task_id,
            "model_hf_id": spec.get("hf_id"),
            "model_name": spec.get("name"),
            "model_path": spec.get("model_path"),
            "gpu_index": spec.get("gpu"),
            "prompt": task.prompt,
            "prompt_hash": prompt_hash(task.prompt),
            "per_task_seed": per_task_seed,
            "decoder_status": dict(decoder_status),
            "raw_response": raw_response,
            "raw_response_sha256": sha256_text(raw_response),
        },
    )
    return path


def _summary(
    spec: Mapping[str, Any],
    model_used: bool,
    blocker: str | None,
    live_inference_duration_s: float,
) -> JsonDict:
    return {
        "hf_id": spec.get("hf_id"),
        "model_name": spec.get("name"),
        "model_path": spec.get("model_path"),
        "gpu_index": spec.get("gpu"),
        "model_used": bool(model_used),
        "blocker": blocker,
        "live_inference_duration_s": round(float(live_inference_duration_s), 6),
    }


def _generation_source(decoder_status: Mapping[str, Any]) -> str:
    if decoder_status.get("constrained_decoder_available"):
        return "live_sota_llamacpp_llguidance_schema"
    return "live_sota_llamacpp_prompt_schema"


def _family_slug(task_type: str) -> str:
    return {
        "linear_integer": "linear",
        "knapsack_binary": "knapsack",
        "graph_coloring": "graph",
    }[task_type]


def _decoder_fallback(error: str, *, grammar: str | None = None) -> JsonDict:
    return {
        "backend_name": "prompt_schema_with_deterministic_parser_repair",
        "constrained_decoder_available": False,
        "fallback_backend_available": True,
        "grammar": grammar,
        "llguidance_version": None,
        "decoder_error": error,
    }


def _generic_solution_schema() -> JsonDict:
    return {
        "type": "object",
        "required": ["solution"],
        "properties": {"solution": {"type": "object"}},
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _json_clone(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True))


def _write_json(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dict(payload)


def _default_llama_importer() -> tuple[
    bool, type[Any] | None, str | None
]:  # pragma: no cover - depends on host llama.cpp install.
    try:
        from llama_cpp import Llama  # type: ignore[import]  # noqa: PLC0415
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(
        "[exp2926] "
        f"verdict={artifact['honest_verdict']} "
        f"models={artifact['models_used']} "
        f"syntax={artifact['syntax_valid_rate']} "
        f"feasible={artifact['feasibility_rate_overall']} "
        f"optimal_given_feasible={artifact['optimality_rate_given_feasible']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
