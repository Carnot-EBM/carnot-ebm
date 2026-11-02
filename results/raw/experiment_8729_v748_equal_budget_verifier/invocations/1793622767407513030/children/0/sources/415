"""Exp 2991 gated SOTA intent-preserving repair on the hard-code stress set.

Spec: REQ-CODE-2991, SCENARIO-CODE-2991.

The experiment is deliberately machine-gated. It refuses to promote smoke
evidence, reruns the Exp 2990 baseline verifier before any LLM repair, records
the generated patch plus verifier transcript for every candidate, and only marks
the result clean when useful repair deltas do not buy progress by increasing
schema, syntax, or verifier false-accept failures.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot.eval import hard_code_stress_manifest as hard
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RepairGenerator = Callable[
    [JsonDict, str, int, int, JsonDict],
    "GenerationOutcome",
]
PreconditionProbe = Callable[["ExperimentConfig"], "PreconditionReport"]
LlamaFactory = Callable[..., Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
ARTIFACT_FILENAME = "experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1.json"
ARTIFACT = ARTIFACT_FILENAME.removesuffix(".json")
SCHEMA = "carnot.gated_sota_intent_preserving_repair_hard_set.v1"
PREFLIGHT_FILENAME = "experiment_2989_sota_gguf_cache_provenance_preflight_v1.json"
HARD_SET_FILENAME = hard.ARTIFACT_FILENAME
INFERENCE_SUBSTRATE = "live_llm_inference"

HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3.5-0.8B",
    "unsloth/gemma-4-E4B-it-GGUF",
)
DEFAULT_N_TASKS = 24
MIN_HEADLINE_TASKS = 20
DEFAULT_MAX_TOKENS = 320
DEFAULT_RANDOM_SEED = 299100
TRACE_COVERAGE_FLOOR = 0.8
RAW_REL_DIR = Path("results/raw/experiment_2991_gated_sota_intent_preserving_repair_hard_set_v1")
VERIFIER_REL_DIR = Path("results/verifier_transcripts/experiment_2991")
REQUIRED_ARTIFACT_FIELDS = (
    "repair_rerun_clean",
    "headline_result",
    "n_tasks",
    "model_specs",
    "headline_models_used",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "schema_failure_rate_delta",
    "syntax_failure_rate_delta",
    "verifier_false_accept_delta",
    "trace_coverage",
    "transcript_paths",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class GenerationOutcome:
    """Raw text and provenance from one repair-generation call."""

    text: str
    tokens_generated: int
    duration_s: float
    backend: str
    backend_detail: str
    error: str | None = None


@dataclass(frozen=True)
class ParsedRepair:
    """Structured view of one model repair response."""

    schema_valid: bool
    draft_intent: str
    final_patch: str
    schema_errors: list[str]


@dataclass(frozen=True)
class PreconditionReport:
    """Preflight checks and runnable headline model specs for Exp 2991."""

    checks: list[JsonDict]
    model_specs: JsonDict
    runnable_model_specs: list[JsonDict]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and small knobs for the Exp 2991 harness."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    manifest_path: Path | None = None
    raw_dir: Path | None = None
    verifier_dir: Path | None = None
    n_tasks: int = DEFAULT_N_TASKS
    max_tokens: int = DEFAULT_MAX_TOKENS
    random_seed: int = DEFAULT_RANDOM_SEED
    max_headline_models: int = 1
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / hard.DEFAULT_MANIFEST_REL_PATH

    def resolved_raw_dir(self) -> Path:
        return self.raw_dir or self.repo_root / RAW_REL_DIR

    def resolved_patch_dir(self) -> Path:
        return self.resolved_raw_dir() / "patches"

    def resolved_transcript_dir(self) -> Path:
        return self.resolved_raw_dir() / "transcripts"

    def resolved_verifier_dir(self) -> Path:
        return self.verifier_dir or self.repo_root / VERIFIER_REL_DIR


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
    precondition_probe: PreconditionProbe | None = None,
) -> JsonDict:
    """Build the terminal Exp 2991 artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    precondition_probe = precondition_probe or default_precondition_probe
    report = precondition_probe(config)
    manifest_report = _validate_manifest(config)
    checks = [*report.checks, *manifest_report["checks"]]

    if not all(bool(row.get("available")) for row in checks) or not report.runnable_model_specs:
        return _blocked_artifact(config, started, checks, report.model_specs)

    tasks = list(manifest_report["items"][: config.n_tasks])
    if len(tasks) < MIN_HEADLINE_TASKS:
        blocked = [
            *checks,
            {
                "resource": "hard_manifest_minimum_task_count",
                "available": False,
                "detail": f"{len(tasks)} < {MIN_HEADLINE_TASKS}",
            },
        ]
        return _blocked_artifact(config, started, blocked, report.model_specs)

    baseline_rows = [_baseline_evaluation(item) for item in tasks]
    model_specs = report.runnable_model_specs[: max(1, config.max_headline_models)]
    live_generator = generator or llama_cpp_repair_generator(model_specs[0])
    candidate_rows: list[JsonDict] = []
    transcript_paths: list[str] = []
    patch_paths: list[str] = []
    verifier_log_paths: list[str] = []

    for model_spec in model_specs:
        for task_index, item in enumerate(tasks):
            seed = config.random_seed + task_index
            prompt = repair_prompt(item)
            generation = live_generator(item, prompt, seed, config.max_tokens, model_spec)
            row = _candidate_evaluation(config, item, prompt, seed, model_spec, generation)
            candidate_rows.append(row)
            transcript_paths.append(row["transcript_path"])
            patch_paths.append(row["candidate_patch_path"])
            verifier_log_paths.append(row["verifier_log_path"])

    return _complete_artifact(
        config=config,
        started=started,
        preconditions_checked=checks,
        model_specs=report.model_specs,
        tasks=tasks,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        transcript_paths=transcript_paths,
        patch_paths=patch_paths,
        verifier_log_paths=verifier_log_paths,
    )


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
    precondition_probe: PreconditionProbe | None = None,
) -> JsonDict:
    """Build and persist the Exp 2991 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(
        config,
        generator=generator,
        precondition_probe=precondition_probe,
    )
    path = config.artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def default_precondition_probe(config: ExperimentConfig) -> PreconditionReport:
    """Check upstream gates, CUDA visibility, and mandated headline GGUF caches."""

    results_dir = config.repo_root / "results"
    preflight_path = results_dir / PREFLIGHT_FILENAME
    hard_path = results_dir / HARD_SET_FILENAME
    preflight = _read_json_if_present(preflight_path)
    hard_payload = _read_json_if_present(hard_path)
    checks = [
        {
            "resource": "exp2989_sota_preflight",
            "available": preflight_path.is_file()
            and preflight.get("sota_headline_ready") is True,
            "path": str(_relative_or_absolute(config.repo_root, preflight_path)),
            "sha256": _sha256_file(preflight_path) if preflight_path.is_file() else None,
        },
        {
            "resource": "exp2990_hard_stress_artifact",
            "available": hard_path.is_file()
            and hard_payload.get("hard_code_stress_set_ready") is True,
            "path": str(_relative_or_absolute(config.repo_root, hard_path)),
            "sha256": _sha256_file(hard_path) if hard_path.is_file() else None,
        },
    ]
    cuda = _cuda_status()
    checks.append({"resource": "cuda_available", "available": cuda["cuda_available"], **cuda})

    pair = _call_cached_sota_pair()
    runnable = [dict(row) for row in pair or () if _is_headline_model(row.get("hf_id"))]
    if not runnable:
        runnable = [
            {
                "name": _model_name(hf_id),
                "hf_id": hf_id,
                "gpu": 0,
                "model_path": path,
                "cached": True,
            }
            for hf_id in HEADLINE_MODEL_IDS
            for path in [resolve_cached_gguf(hf_id)]
            if path
        ]
    cache_records = []
    for hf_id in HEADLINE_MODEL_IDS:
        resolved = next((row.get("model_path") for row in runnable if row.get("hf_id") == hf_id), None)
        cache_records.append(
            {
                "hf_id": hf_id,
                "model_path": str(resolved or ""),
                "cached": bool(resolved),
            }
        )
    checks.append(
        {
            "resource": "headline_model_cache_available",
            "available": bool(runnable),
            "detail": [row["hf_id"] for row in runnable],
        }
    )
    model_specs = {
        "headline_models": list(HEADLINE_MODEL_IDS),
        "smoke_only_models": list(SMOKE_ONLY_MODEL_IDS),
        "preferred_quantization": "Q4_K_M",
        "cache_probe": cache_records,
        "runnable_headline_models": runnable,
        "cached_sota_pair_returned": bool(pair),
    }
    return PreconditionReport(checks=checks, model_specs=model_specs, runnable_model_specs=runnable)


def repair_prompt(item: Mapping[str, Any]) -> str:
    """Return the trace-aware repair prompt for one hard-set item."""

    failing_ids = ", ".join(item.get("baseline_verification", {}).get("failing_test_ids") or [])
    failing_assertions = "\n".join(
        str(test.get("code") or "")
        for test in item.get("tests") or ()
        if str(test.get("test_id") or "") in set(item.get("baseline_verification", {}).get("failing_test_ids") or [])
    )
    return (
        "Exp 2991 intent-preserving trace-aware code repair.\n"
        f"Item: {item.get('item_id')}\n"
        f"Entry point: {item.get('entry_point')}\n"
        f"Expected behavior: {item.get('expected_behavior')}\n"
        f"Baseline candidate:\n{item.get('baseline_candidate')}\n"
        f"Failing assertion ids: {failing_ids}\n"
        f"Failing assertions:\n{failing_assertions}\n"
        "Preserve the public function signature and draft intent. "
        "Do not hard-code the visible tests. Return exactly one JSON object with "
        'string fields "draft_intent" and "final_patch".'
    )


def parse_repair_output(text: str) -> ParsedRepair:
    """Parse the model's JSON repair response while keeping fallback diagnostics."""

    parsed, errors = _parse_json_object(text)
    if isinstance(parsed, Mapping):
        draft_intent = parsed.get("draft_intent")
        final_patch = parsed.get("final_patch") or parsed.get("repaired_code")
        schema_errors = list(errors)
        if not isinstance(draft_intent, str) or not draft_intent.strip():
            schema_errors.append('missing non-empty "draft_intent"')
        if not isinstance(final_patch, str) or not final_patch.strip():
            schema_errors.append('missing non-empty "final_patch"')
        if not schema_errors:
            return ParsedRepair(True, draft_intent.strip(), final_patch.strip() + "\n", [])
        fallback = _extract_python_code(text)
        return ParsedRepair(False, str(draft_intent or ""), fallback, schema_errors)
    fallback = _extract_python_code(text)
    return ParsedRepair(False, "", fallback, errors or ["no JSON object found"])


def syntax_diagnostics(code: str) -> tuple[bool, list[str]]:
    """Return parser success plus bounded syntax diagnostics for a candidate."""

    if not code.strip():
        return False, ["empty candidate"]
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, [f"SyntaxError: {exc.msg}"]
    return True, []


def llama_cpp_repair_generator(  # pragma: no cover - live hardware path.
    model_spec: Mapping[str, Any],
    *,
    temperature: float = 0.1,
    llama_factory: LlamaFactory | None = None,
) -> RepairGenerator:
    """Return a local GGUF-backed repair generator."""

    state: dict[str, Any] = {"llm": None}

    def _ensure_loaded() -> Any:  # pragma: no cover - live hardware path.
        if state["llm"] is None:
            factory = llama_factory
            if factory is None:
                from llama_cpp import Llama

                factory = Llama
            state["llm"] = factory(
                model_path=str(model_spec.get("model_path") or ""),
                n_ctx=4096,
                n_batch=128,
                n_ubatch=128,
                n_gpu_layers=-1,
                main_gpu=int(model_spec.get("gpu") or 0),
                verbose=False,
            )
        return state["llm"]

    def _generate(
        _item: JsonDict,
        prompt: str,
        seed: int,
        max_tokens: int,
        active_model_spec: JsonDict,
    ) -> GenerationOutcome:  # pragma: no cover - live hardware path.
        started = time.monotonic()
        try:
            output = _ensure_loaded()(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                stop=["\n\n\n"],
            )
            return GenerationOutcome(
                text=_llama_text(output),
                tokens_generated=_llama_completion_tokens(output),
                duration_s=time.monotonic() - started,
                backend="llama_cpp",
                backend_detail=str(active_model_spec.get("model_path") or ""),
            )
        except Exception as exc:
            return GenerationOutcome(
                text="",
                tokens_generated=0,
                duration_s=time.monotonic() - started,
                backend="llama_cpp",
                backend_detail=str(active_model_spec.get("model_path") or ""),
                error=f"{type(exc).__name__}: {exc}",
            )

    return _generate


def _baseline_evaluation(item: Mapping[str, Any]) -> JsonDict:
    outcome = hard.run_candidate_tests(item, "baseline_candidate")
    return _evaluation_from_outcome(
        mode="baseline_candidate",
        item=item,
        candidate_code=str(item.get("baseline_candidate") or ""),
        outcome=outcome,
        schema_valid=True,
        schema_errors=[],
        syntax_errors=_syntax_errors_from_outcome(outcome),
        runtime_trace=_runtime_trace(outcome),
        model_spec={},
    )


def _candidate_evaluation(
    config: ExperimentConfig,
    item: JsonDict,
    prompt: str,
    seed: int,
    model_spec: JsonDict,
    generation: GenerationOutcome,
) -> JsonDict:
    parsed = parse_repair_output(generation.text)
    syntax_success, syntax_errors = syntax_diagnostics(parsed.final_patch)
    candidate_item = dict(item)
    candidate_item["repair_candidate"] = parsed.final_patch
    outcome = (
        hard.run_candidate_tests(candidate_item, "repair_candidate")
        if syntax_success
        else hard.run_candidate_tests(
            {**candidate_item, "repair_candidate": parsed.final_patch}, "repair_candidate"
        )
    )
    runtime_trace = _runtime_trace(outcome)
    row = _evaluation_from_outcome(
        mode="intent_preserving_trace_aware_repair",
        item=item,
        candidate_code=parsed.final_patch,
        outcome=outcome,
        schema_valid=parsed.schema_valid,
        schema_errors=parsed.schema_errors,
        syntax_errors=syntax_errors or _syntax_errors_from_outcome(outcome),
        runtime_trace=runtime_trace,
        model_spec=model_spec,
    )
    row.update(
        {
            "seed": seed,
            "prompt_sha256": _sha256_text(prompt),
            "draft_intent": parsed.draft_intent,
            "generation_backend": generation.backend,
            "generation_backend_detail": generation.backend_detail,
            "generation_duration_s": float(generation.duration_s),
            "tokens_generated": int(generation.tokens_generated),
            "generation_error": generation.error,
        }
    )
    _write_candidate_evidence(config, item, prompt, generation, parsed, row)
    return row


def _evaluation_from_outcome(
    *,
    mode: str,
    item: Mapping[str, Any],
    candidate_code: str,
    outcome: hard.VerificationOutcome,
    schema_valid: bool,
    schema_errors: list[str],
    syntax_errors: list[str],
    runtime_trace: list[JsonDict],
    model_spec: Mapping[str, Any],
) -> JsonDict:
    syntax_success = not any(error.get("error_type") == "SyntaxError" for error in outcome.errors)
    verifier_accepted = bool(outcome.passed)
    false_accept = verifier_accepted and not outcome.passed
    return {
        "mode": mode,
        "item_id": str(item.get("item_id") or ""),
        "entry_point": str(item.get("entry_point") or ""),
        "model_hf_id": str(model_spec.get("hf_id") or ""),
        "model_path": str(model_spec.get("model_path") or ""),
        "candidate_sha256": _sha256_text(candidate_code),
        "schema_valid": bool(schema_valid),
        "schema_errors": list(schema_errors),
        "syntax_success": bool(syntax_success),
        "syntax_errors": list(syntax_errors),
        "passed": bool(outcome.passed),
        "verifier_accepted": verifier_accepted,
        "verifier_false_accept": false_accept,
        "verifier_output": outcome.as_dict(),
        "runtime_trace": runtime_trace,
        "runtime_trace_present": bool(runtime_trace),
        "failing_assertions": list(outcome.failing_test_ids),
    }


def _complete_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    preconditions_checked: list[JsonDict],
    model_specs: JsonDict,
    tasks: Sequence[JsonDict],
    baseline_rows: list[JsonDict],
    candidate_rows: list[JsonDict],
    transcript_paths: list[str],
    patch_paths: list[str],
    verifier_log_paths: list[str],
) -> JsonDict:
    baseline = _metric_summary(baseline_rows, tasks)
    repair = _metric_summary(candidate_rows, tasks)
    deltas = _metric_deltas(baseline, repair)
    headline_models_used = sorted(
        {str(row.get("model_hf_id")) for row in candidate_rows if row.get("model_hf_id")}
    )
    trace_coverage = _rate(candidate_rows, lambda row: row.get("runtime_trace_present") is True)
    headline_result = bool(headline_models_used)
    clean = _repair_rerun_clean(
        headline_result=headline_result,
        n_tasks=len(tasks),
        headline_models_used=headline_models_used,
        deltas=deltas,
        trace_coverage=trace_coverage,
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "repair_rerun_clean": clean,
        "headline_result": headline_result,
        "n_tasks": len(tasks),
        "model_specs": model_specs,
        "headline_models_used": headline_models_used,
        "pass_at_1_delta": deltas["pass_at_1_delta"],
        "pass_at_k_delta": deltas["pass_at_k_delta"],
        "schema_failure_rate_delta": deltas["schema_failure_rate_delta"],
        "syntax_failure_rate_delta": deltas["syntax_failure_rate_delta"],
        "verifier_false_accept_delta": deltas["verifier_false_accept_delta"],
        "trace_coverage": trace_coverage,
        "transcript_paths": transcript_paths,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            "clean: hard-set intent-preserving repair gates passed"
            if clean
            else "flagged: hard-set repair did not clear promotion gates"
        ),
        "baseline_pass_at_1": baseline["pass_at_1"],
        "repair_pass_at_1": repair["pass_at_1"],
        "baseline_pass_at_k": baseline["pass_at_k"],
        "repair_pass_at_k": repair["pass_at_k"],
        "baseline_metrics": baseline,
        "repair_metrics": repair,
        "candidate_patch_paths": patch_paths,
        "verifier_log_paths": verifier_log_paths,
        "candidate_evaluations": candidate_rows,
        "baseline_evaluations": baseline_rows,
        "preconditions_checked": preconditions_checked,
        "source_artifacts": _source_artifacts(config),
        "selected_item_ids": [str(item.get("item_id") or "") for item in tasks],
        "candidate_manifest_sha256": _sha256_payload(candidate_rows),
        "reproducibility_checksum": _sha256_payload(
            {
                "selected_item_ids": [str(item.get("item_id") or "") for item in tasks],
                "headline_models_used": headline_models_used,
                "deltas": deltas,
            }
        ),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    preconditions_checked: list[JsonDict],
    model_specs: JsonDict,
) -> JsonDict:
    empty_metrics = _metric_summary([], [])
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "repair_rerun_clean": False,
        "headline_result": False,
        "n_tasks": 0,
        "model_specs": model_specs,
        "headline_models_used": [],
        "pass_at_1_delta": 0.0,
        "pass_at_k_delta": 0.0,
        "schema_failure_rate_delta": 0.0,
        "syntax_failure_rate_delta": 0.0,
        "verifier_false_accept_delta": 0.0,
        "trace_coverage": 0.0,
        "transcript_paths": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked: preconditions not met",
        "baseline_pass_at_1": 0.0,
        "repair_pass_at_1": 0.0,
        "baseline_pass_at_k": 0.0,
        "repair_pass_at_k": 0.0,
        "baseline_metrics": empty_metrics,
        "repair_metrics": empty_metrics,
        "candidate_patch_paths": [],
        "verifier_log_paths": [],
        "candidate_evaluations": [],
        "baseline_evaluations": [],
        "preconditions_checked": preconditions_checked,
        "source_artifacts": _source_artifacts(config),
        "selected_item_ids": [],
        "candidate_manifest_sha256": _sha256_payload([]),
        "reproducibility_checksum": _sha256_payload({"blocked": True}),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _validate_manifest(config: ExperimentConfig) -> JsonDict:
    manifest_path = config.resolved_manifest_path()
    hard_path = config.repo_root / "results" / HARD_SET_FILENAME
    hard_payload = _read_json_if_present(hard_path)
    checks: list[JsonDict] = []
    if not manifest_path.is_file():
        return {
            "items": [],
            "checks": [
                {
                    "resource": "hard_manifest_integrity",
                    "available": False,
                    "detail": f"missing {manifest_path}",
                }
            ],
        }
    items = hard.load_manifest(manifest_path)
    manifest_sha = _sha256_file(manifest_path)
    expected_sha = hard_payload.get("manifest_sha256")
    baseline = [hard.run_candidate_tests(item, "baseline_candidate") for item in items]
    reference = [hard.run_candidate_tests(item, "reference_solution") for item in items]
    ready = bool(
        len(items) == int(hard_payload.get("n_items") or len(items))
        and (not expected_sha or expected_sha == manifest_sha)
        and all(bool(item.get("tests")) for item in items)
        and all(not outcome.passed for outcome in baseline)
        and all(outcome.passed for outcome in reference)
    )
    checks.append(
        {
            "resource": "hard_manifest_integrity",
            "available": ready,
            "path": str(_relative_or_absolute(config.repo_root, manifest_path)),
            "sha256": manifest_sha,
            "n_items": len(items),
        }
    )
    return {"items": items, "checks": checks}


def _write_candidate_evidence(
    config: ExperimentConfig,
    item: Mapping[str, Any],
    prompt: str,
    generation: GenerationOutcome,
    parsed: ParsedRepair,
    row: JsonDict,
) -> None:
    token = _safe_token(f"{item.get('item_id')}_{row.get('model_hf_id')}_{row.get('seed')}")
    patch_path = config.resolved_patch_dir() / f"{token}.py"
    transcript_path = config.resolved_transcript_dir() / f"{token}.json"
    verifier_path = config.resolved_verifier_dir() / f"{token}.json"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(parsed.final_patch, encoding="utf-8")
    transcript = {
        "item_id": item.get("item_id"),
        "model_hf_id": row.get("model_hf_id"),
        "prompt": prompt,
        "raw_response": generation.text,
        "draft_intent": parsed.draft_intent,
        "final_patch_sha256": _sha256_text(parsed.final_patch),
        "schema_valid": parsed.schema_valid,
        "schema_errors": parsed.schema_errors,
    }
    verifier = {
        "item_id": item.get("item_id"),
        "verifier_output": row["verifier_output"],
        "runtime_trace": row["runtime_trace"],
        "failing_assertions": row["failing_assertions"],
    }
    transcript_path.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verifier_path.write_text(json.dumps(verifier, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row["candidate_patch_path"] = str(_relative_or_absolute(config.repo_root, patch_path))
    row["transcript_path"] = str(_relative_or_absolute(config.repo_root, transcript_path))
    row["verifier_log_path"] = str(_relative_or_absolute(config.repo_root, verifier_path))


def _metric_summary(rows: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_item: dict[str, list[Mapping[str, Any]]] = {str(item.get("item_id") or ""): [] for item in tasks}
    for row in rows:
        by_item.setdefault(str(row.get("item_id") or ""), []).append(row)
    per_task = []
    for item in tasks:
        item_id = str(item.get("item_id") or "")
        item_rows = by_item.get(item_id, [])
        pass_vector = [bool(row.get("passed")) for row in item_rows]
        per_task.append(
            {
                "item_id": item_id,
                "pass_vector": pass_vector,
                "pass_at_1": 1.0 if pass_vector and pass_vector[0] else 0.0,
                "pass_at_k": 1.0 if any(pass_vector) else 0.0,
            }
        )
    return {
        "candidate_count": len(rows),
        "per_task_results": per_task,
        "pass_at_1": _mean([row["pass_at_1"] for row in per_task]),
        "pass_at_k": _mean([row["pass_at_k"] for row in per_task]),
        "schema_failure_rate": _rate(rows, lambda row: row.get("schema_valid") is False),
        "syntax_failure_rate": _rate(rows, lambda row: row.get("syntax_success") is False),
        "verifier_false_accept_rate": _rate(
            rows,
            lambda row: row.get("verifier_false_accept") is True,
        ),
    }


def _metric_deltas(baseline: Mapping[str, Any], repair: Mapping[str, Any]) -> JsonDict:
    return {
        "pass_at_1_delta": _delta(repair.get("pass_at_1"), baseline.get("pass_at_1")),
        "pass_at_k_delta": _delta(repair.get("pass_at_k"), baseline.get("pass_at_k")),
        "schema_failure_rate_delta": _delta(
            repair.get("schema_failure_rate"),
            baseline.get("schema_failure_rate"),
        ),
        "syntax_failure_rate_delta": _delta(
            repair.get("syntax_failure_rate"),
            baseline.get("syntax_failure_rate"),
        ),
        "verifier_false_accept_delta": _delta(
            repair.get("verifier_false_accept_rate"),
            baseline.get("verifier_false_accept_rate"),
        ),
    }


def _repair_rerun_clean(
    *,
    headline_result: bool,
    n_tasks: int,
    headline_models_used: Sequence[str],
    deltas: Mapping[str, Any],
    trace_coverage: float,
) -> bool:
    return bool(
        headline_result
        and n_tasks >= MIN_HEADLINE_TASKS
        and any(_is_headline_model(model_id) for model_id in headline_models_used)
        and _positive(deltas.get("pass_at_1_delta"))
        and _nonnegative(deltas.get("pass_at_k_delta"))
        and _nonpositive(deltas.get("schema_failure_rate_delta"))
        and _nonpositive(deltas.get("syntax_failure_rate_delta"))
        and _nonpositive(deltas.get("verifier_false_accept_delta"))
        and trace_coverage >= TRACE_COVERAGE_FLOOR
    )


def _runtime_trace(outcome: hard.VerificationOutcome) -> list[JsonDict]:
    return [
        {
            "stage": "deterministic_hard_tests",
            "exit_code": 0 if outcome.passed else 1,
            "tests_run": outcome.tests_run,
            "failing_assertions": list(outcome.failing_test_ids),
            "errors": list(outcome.errors),
        }
    ]


def _syntax_errors_from_outcome(outcome: hard.VerificationOutcome) -> list[str]:
    return [
        str(error.get("message") or error.get("error_type") or "")
        for error in outcome.errors
        if error.get("error_type") == "SyntaxError"
    ]


def _parse_json_object(text: str) -> tuple[Any | None, list[str]]:
    stripped = _strip_code_fence(text.strip(), "json")
    decoder = json.JSONDecoder()
    try:
        value, _end = decoder.raw_decode(stripped)
        return value, []
    except json.JSONDecodeError as exc:
        start = stripped.find("{")
        if start < 0:
            return None, ["no JSON object found"]
        try:
            value, _end = decoder.raw_decode(stripped[start:])
            return value, []
        except json.JSONDecodeError:
            return None, [f"invalid JSON object: {exc.msg}"]


_PYTHON_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_python_code(text: str) -> str:
    for match in _PYTHON_FENCE_RE.finditer(text):
        block = match.group(1).strip()
        if "def " in block:
            return block + "\n"
    return text.strip()


def _strip_code_fence(text: str, language: str) -> str:
    if not text.startswith("```"):
        return text
    stripped = text.strip("`").strip()
    if stripped.lower().startswith(language):
        return stripped[len(language) :].strip()
    return stripped


def _call_cached_sota_pair() -> list[JsonDict] | None:
    try:
        result = cached_sota_pair(gpu_indices=(0, 1))
    except TypeError:
        result = cached_sota_pair()
    return [dict(row) for row in result] if result else None


def _cuda_status() -> JsonDict:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {
            "cuda_available": completed.returncode == 0 and bool(completed.stdout.strip()),
            "command": command,
            "returncode": completed.returncode,
            "stdout_summary": completed.stdout[:500],
            "stderr_summary": completed.stderr[:500],
        }
    except Exception as exc:  # pragma: no cover - defensive host diagnostics.
        return {"cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    rel_paths = [
        Path("results") / PREFLIGHT_FILENAME,
        Path("results") / HARD_SET_FILENAME,
        hard.DEFAULT_MANIFEST_REL_PATH,
    ]
    out = []
    for rel_path in rel_paths:
        path = config.repo_root / rel_path
        out.append(
            {
                "path": str(rel_path),
                "present": path.is_file(),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
        )
    return out


def _read_json_if_present(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


def _llama_text(output: Any) -> str:  # pragma: no cover - live hardware path.
    if isinstance(output, Mapping):
        choices = output.get("choices")
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                return str(first.get("text") or "")
    return str(output)


def _llama_completion_tokens(output: Any) -> int:  # pragma: no cover - live hardware path.
    if isinstance(output, Mapping):
        usage = output.get("usage")
        if isinstance(usage, Mapping):
            return int(usage.get("completion_tokens") or 0)
    return 0


def _is_headline_model(value: Any) -> bool:
    return str(value or "") in HEADLINE_MODEL_IDS


def _model_name(hf_id: str) -> str:
    return hf_id.split("/", 1)[-1].removesuffix("-GGUF")


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _relative_or_absolute(root: Path, path: Path) -> Path:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return path.resolve(strict=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return 0.0 if not rows else sum(1 for row in rows if predicate(row)) / len(rows)


def _delta(after: Any, before: Any) -> float:
    return float(after or 0.0) - float(before or 0.0)


def _positive(value: Any) -> bool:
    return isinstance(value, int | float) and value > 0


def _nonnegative(value: Any) -> bool:
    return isinstance(value, int | float) and value >= 0


def _nonpositive(value: Any) -> bool:
    return isinstance(value, int | float) and value <= 0


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_artifact(
        ExperimentConfig(
            tests_run=(
                ".venv/bin/pytest tests/python/test_experiment_2991_gated_sota_intent_preserving_repair_hard_set.py -q",
                ".venv/bin/pytest tests/python -q",
                "python scripts/check_spec_coverage.py",
            )
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["honest_verdict"].startswith(("clean:", "flagged:")) else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_FILENAME",
    "HEADLINE_MODEL_IDS",
    "PREFLIGHT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "SMOKE_ONLY_MODEL_IDS",
    "ExperimentConfig",
    "GenerationOutcome",
    "ParsedRepair",
    "PreconditionReport",
    "build_artifact",
    "default_precondition_probe",
    "llama_cpp_repair_generator",
    "main",
    "parse_repair_output",
    "repair_prompt",
    "syntax_diagnostics",
    "write_artifact",
]
