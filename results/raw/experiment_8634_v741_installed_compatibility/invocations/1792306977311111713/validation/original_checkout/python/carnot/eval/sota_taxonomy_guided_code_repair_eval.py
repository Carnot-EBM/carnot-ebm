"""Exp 2952 live SOTA taxonomy-guided code-repair evaluation.

The evaluator compares two repair prompts on the same failed Exp 2946 code
candidates: a baseline prompt that receives only the task and failure evidence,
and a taxonomy-guided prompt that also receives the Exp 2950 failure label and
repair focus. Every generated repair is converted into the Exp 2951 structured
candidate manifest before parser, static, sandbox-test, and verifier-threshold
metrics are reported.

Spec: REQ-CODE-2952, SCENARIO-CODE-2952.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot.eval import code_taxonomy_repair_prompt_manifest as exp2950
from carnot.eval import mbpp_humaneval_generated_code_clean_row as exp2889
from carnot.eval import sota_code_generation_corrigendum as exp2910
from carnot.eval import structured_candidate_manifest_adapter as exp2951
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf
from carnot.reporting.verifier_ensemble_auprc_code_corpora_2940 import (
    approval_score_from_energy,
    candidate_status_energy,
)
from carnot.verify.sandbox import get_sandbox_status


JsonDict = dict[str, Any]
GenerationOutcome = exp2910.GenerationOutcome
ExecutionOutcome = exp2910.ExecutionOutcome
RepairGenerator = Callable[[str, int, int, Mapping[str, Any]], GenerationOutcome]
Executor = Callable[[str, float], ExecutionOutcome]
PreconditionProbe = Callable[["ExperimentConfig"], "PreconditionReport"]
TaskRowProvider = Callable[["ExperimentConfig"], dict[tuple[str, str], JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json"
ARTIFACT = "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1"
SCHEMA = "carnot.sota_taxonomy_guided_code_repair_eval.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"

EXP2940_REL_PATH = Path("results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json")
EXP2946_REL_PATH = Path("results/experiment_2946_sota_code_generation_continuation_v1.json")
NESTED_EXP2946_REL_PATH = Path("results/experiment_2946_nested_exp2910_protocol.json")
EXP2950_REL_PATH = Path("results/experiment_2950_code_taxonomy_repair_prompt_manifest_v1.json")
EXP2951_REL_PATH = Path("results/experiment_2951_structured_candidate_manifest_adapter_v1.json")
RAW_RESPONSE_REL_DIR = Path("results/raw/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1")

DEFAULT_RANDOM_SEED = 2952
DEFAULT_N_TASKS = 4
DEFAULT_SAMPLES_PER_MODE = 4
DEFAULT_MAX_TOKENS = 192
DEFAULT_TEMPERATURE = 0.2
DEFAULT_SANDBOX_TIMEOUT_S = 10.0

BASELINE_MODE = "baseline_no_taxonomy"
REPAIR_MODE = "taxonomy_guided"
MODES = (BASELINE_MODE, REPAIR_MODE)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "headline_models_used",
    "legacy_models_only_for_smoke",
    "n_tasks",
    "baseline_pass_at_1",
    "repair_pass_at_1",
    "pass_at_1_delta",
    "baseline_pass_at_k",
    "repair_pass_at_k",
    "pass_at_k_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "false_accept_delta",
    "taxonomy_repair_delta_pass",
    "candidate_manifest_sha256",
    "reproducibility_checksum",
    "duration_s",
)

UNSAFE_IMPORTS = getattr(exp2950, "UNSAFE_IMPORTS", frozenset({"os", "subprocess", "sys"}))
UNSUPPORTED_ATTRS = {
    ("json", "parse"),
    ("json", "parsefast"),
    ("math", "avg"),
}


@dataclass(frozen=True)
class PreconditionReport:
    """System precondition evidence collected before live repair generation."""

    checks: list[JsonDict]
    model_specs: list[JsonDict]
    runnable_model_specs: list[JsonDict]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2952."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_response_dir: Path | None = None
    exp2940_path: Path = EXP2940_REL_PATH
    exp2946_path: Path = EXP2946_REL_PATH
    exp2950_path: Path = EXP2950_REL_PATH
    exp2951_path: Path = EXP2951_REL_PATH
    nested_exp2946_path: Path = NESTED_EXP2946_REL_PATH
    n_tasks: int = DEFAULT_N_TASKS
    samples_per_mode: int = DEFAULT_SAMPLES_PER_MODE
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    random_seed: int = DEFAULT_RANDOM_SEED
    sandbox_timeout_s: float = DEFAULT_SANDBOX_TIMEOUT_S
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def raw_dir(self) -> Path:
        return self.raw_response_dir or self.repo_root / RAW_RESPONSE_REL_DIR


@dataclass(frozen=True)
class RepairSource:
    """One failed upstream candidate selected for equal-budget repair."""

    task_index: int
    corpus: str
    stable_id: str
    sample_id: str
    candidate_row: JsonDict
    task_row: JsonDict
    original_failure_categories: tuple[str, ...]

    @property
    def task_key(self) -> str:
        return f"{self.corpus}:{self.stable_id}"


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
    executor: Executor = exp2910.execute_script_in_sandbox,
    precondition_probe: PreconditionProbe = None,
    task_row_provider: TaskRowProvider = None,
) -> JsonDict:
    """Build the Exp 2952 repair-evaluation artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    precondition_probe = precondition_probe or default_precondition_probe
    task_row_provider = task_row_provider or default_task_row_provider

    source_checks = _source_precondition_checks(config)
    if not all(row["available"] for row in source_checks):
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_upstream_artifact_missing",
            preconditions_checked=source_checks,
            model_specs=[],
        )

    report = precondition_probe(config)
    preconditions_checked = source_checks + [dict(row) for row in report.checks]
    if not report.runnable_model_specs or any(row.get("available") is False for row in report.checks):
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_sota_gguf_precondition",
            preconditions_checked=preconditions_checked,
            model_specs=report.model_specs,
        )

    exp2940 = _read_json(_repo_path(config.repo_root, config.exp2940_path))
    exp2946 = _read_json(_repo_path(config.repo_root, config.exp2946_path))
    exp2950_payload = _read_json(_repo_path(config.repo_root, config.exp2950_path))
    nested = _read_json(_repo_path(config.repo_root, _nested_protocol_path(config, exp2946)))
    task_rows = task_row_provider(config)
    selected = select_repair_set(nested, task_rows, config.n_tasks)
    if not selected:
        return _blocked_artifact(
            config=config,
            started=started,
            verdict="blocked_no_failed_exp2946_candidates",
            preconditions_checked=preconditions_checked,
            model_specs=report.model_specs,
        )

    model_spec = dict(report.runnable_model_specs[0])
    live_generator = generator or llama_cpp_repair_generator(
        model_path=str(model_spec["model_path"]),
        main_gpu=int(model_spec.get("gpu") or 0),
        temperature=config.temperature,
    )
    threshold = _verifier_threshold(exp2940)
    evaluations: list[JsonDict] = []
    manifests: list[JsonDict] = []
    templates = dict(exp2950_payload.get("repair_prompt_templates") or {})

    for source in selected:
        for mode in MODES:
            for sample_index in range(config.samples_per_mode):
                seed = _candidate_seed(config, mode, source.task_index, sample_index)
                prompt = _repair_prompt(source, mode, templates)
                generation = live_generator(prompt, seed, config.max_tokens, model_spec)
                evaluation, manifest = evaluate_repair_candidate(
                    config=config,
                    source=source,
                    mode=mode,
                    sample_index=sample_index,
                    seed=seed,
                    prompt=prompt,
                    generation=generation,
                    model_spec=model_spec,
                    threshold=threshold,
                    executor=executor,
                )
                evaluations.append(evaluation)
                manifests.append(manifest)

    baseline = _mode_metrics(evaluations, BASELINE_MODE, selected)
    repair = _mode_metrics(evaluations, REPAIR_MODE, selected)
    duration_s = _elapsed(config, started)
    candidate_manifest_sha = _sha256_payload(manifests)
    deltas = _metric_deltas(baseline, repair)
    delta_pass = _taxonomy_delta_pass(deltas)
    false_accept_notes = _false_accept_notes(baseline, repair, deltas)
    selected_task_ids = [source.task_key for source in selected]

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _complete_verdict(deltas, delta_pass),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "source_artifacts": _source_artifacts(config),
        "model_specs": report.model_specs,
        "headline_models_used": sorted(
            {
                str(row.get("model_hf_id"))
                for row in evaluations
                if row.get("model_hf_id")
            }
        ),
        "legacy_models_only_for_smoke": False,
        "n_tasks": len(selected),
        "selected_task_ids": selected_task_ids,
        "selected_repair_set": [_selected_source_row(source) for source in selected],
        "samples_per_mode": config.samples_per_mode,
        "sample_budget_per_mode": len(selected) * config.samples_per_mode,
        "baseline_pass_at_1": baseline["pass_at_1"],
        "repair_pass_at_1": repair["pass_at_1"],
        "pass_at_1_delta": deltas["pass_at_1_delta"],
        "baseline_pass_at_k": baseline["pass_at_k"],
        "repair_pass_at_k": repair["pass_at_k"],
        "pass_at_k_delta": deltas["pass_at_k_delta"],
        "baseline_syntax_failure_rate": baseline["syntax_failure_rate"],
        "repair_syntax_failure_rate": repair["syntax_failure_rate"],
        "syntax_failure_rate_delta": deltas["syntax_failure_rate_delta"],
        "baseline_schema_failure_rate": baseline["schema_failure_rate"],
        "repair_schema_failure_rate": repair["schema_failure_rate"],
        "schema_failure_rate_delta": deltas["schema_failure_rate_delta"],
        "baseline_verifier_acceptance_rate": baseline["verifier_acceptance_rate"],
        "repair_verifier_acceptance_rate": repair["verifier_acceptance_rate"],
        "verifier_acceptance_rate_delta": deltas["verifier_acceptance_rate_delta"],
        "baseline_false_accept_rate": baseline["false_accept_rate"],
        "repair_false_accept_rate": repair["false_accept_rate"],
        "false_accept_delta": deltas["false_accept_delta"],
        "false_accept_audit_notes": false_accept_notes,
        "taxonomy_repair_delta_pass": delta_pass,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "candidate_manifests": manifests,
        "candidate_evaluations": evaluations,
        "reproducibility_checksum": _reproducibility_checksum(
            selected_task_ids=selected_task_ids,
            candidate_manifest_sha256=candidate_manifest_sha,
            model_specs=report.model_specs,
            deltas=deltas,
        ),
        "duration_s": duration_s,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }
    return artifact


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    generator: RepairGenerator | None = None,
    executor: Executor = exp2910.execute_script_in_sandbox,
    precondition_probe: PreconditionProbe = None,
    task_row_provider: TaskRowProvider = None,
) -> JsonDict:
    """Build and persist the Exp 2952 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(
        config,
        generator=generator,
        executor=executor,
        precondition_probe=precondition_probe or default_precondition_probe,
        task_row_provider=task_row_provider or default_task_row_provider,
    )
    out_path = config.artifact_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def select_repair_set(
    nested_protocol: Mapping[str, Any],
    task_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    n_tasks: int,
) -> list[RepairSource]:
    """Select the first failed Exp 2946 tasks with local test context."""

    failed_task_ids = {
        str(row.get("stable_id"))
        for row in nested_protocol.get("per_task_results", [])
        if isinstance(row, Mapping) and _number(row.get("pass_at_k"), 1.0) == 0.0
    }
    selected: list[RepairSource] = []
    seen: set[str] = set()
    for row in _candidate_rows(nested_protocol):
        stable_id = str(row.get("stable_id") or "")
        corpus = _display_corpus(str(row.get("corpus") or ""))
        if not stable_id or stable_id not in failed_task_ids or stable_id in seen:
            continue
        if row.get("passed") is True:
            continue
        task_row = task_rows.get((corpus, stable_id))
        if task_row is None:
            continue
        labels = _original_failure_categories(row)
        selected.append(
            RepairSource(
                task_index=len(selected),
                corpus=corpus,
                stable_id=stable_id,
                sample_id=_sample_id(row),
                candidate_row=dict(row),
                task_row=dict(task_row),
                original_failure_categories=labels,
            )
        )
        seen.add(stable_id)
        if len(selected) >= n_tasks:
            break
    return selected


def evaluate_repair_candidate(
    *,
    config: ExperimentConfig,
    source: RepairSource,
    mode: str,
    sample_index: int,
    seed: int,
    prompt: str,
    generation: GenerationOutcome,
    model_spec: Mapping[str, Any],
    threshold: float,
    executor: Executor,
) -> tuple[JsonDict, JsonDict]:
    """Validate one generated repair through schema, parser, static checks, and tests."""

    raw_response_ref = _write_raw_response(
        config,
        source=source,
        mode=mode,
        sample_index=sample_index,
        seed=seed,
        text=generation.text,
    )
    extraction = exp2910.extract_python_candidate(generation.text)
    repaired_code = extraction.code or generation.text.strip()
    parser_status = "parsed" if extraction.syntax_success else "syntax_error"
    static_checks = _static_checks(repaired_code) if extraction.syntax_success else _syntax_static_checks()
    outcome = _execute_candidate(config, source, repaired_code, extraction.syntax_success, static_checks, executor)
    test_status = "passed" if outcome.passed else "failed" if extraction.syntax_success else "not_run"
    runtime_success = _runtime_success(extraction.syntax_success, outcome)
    verifier_score = approval_score_from_energy(
        candidate_status_energy(
            {
                "extraction_success": extraction.extraction_success,
                "syntax_success": extraction.syntax_success,
                "runtime_success": runtime_success,
                "passed": outcome.passed,
            }
        )
    )
    failure_taxonomy = _post_repair_taxonomy(
        extraction.syntax_success,
        static_checks,
        outcome.passed,
    )
    manifest = _candidate_manifest(
        source=source,
        mode=mode,
        sample_index=sample_index,
        seed=seed,
        model_id=str(model_spec.get("hf_id") or ""),
        raw_response_ref=raw_response_ref,
        raw_response_text=generation.text,
        repaired_code=repaired_code,
        failure_taxonomy=failure_taxonomy,
        parser_status=parser_status,
        test_status=test_status,
        verifier_score=verifier_score,
    )
    validation = exp2951.StructuredCandidateManifestAdapter().validate_record(manifest)
    verifier_accepted = validation.ok and verifier_score >= threshold
    false_accept = verifier_accepted and not outcome.passed
    evaluation = {
        "mode": mode,
        "task_id": source.task_key,
        "stable_id": source.stable_id,
        "corpus": source.corpus,
        "sample_id": source.sample_id,
        "sample_index": sample_index,
        "seed": seed,
        "model_hf_id": str(model_spec.get("hf_id") or ""),
        "model_path": str(model_spec.get("model_path") or ""),
        "prompt_sha256": _sha256_text(prompt),
        "raw_response_ref": raw_response_ref,
        "raw_response_sha256": _sha256_text(generation.text),
        "generation_backend": generation.backend,
        "generation_backend_detail": generation.backend_detail,
        "generation_duration_s": float(generation.duration_s),
        "tokens_generated": int(generation.tokens_generated),
        "generation_error": generation.error,
        "original_failure_categories": list(source.original_failure_categories),
        "parser_status": parser_status,
        "syntax_success": extraction.syntax_success,
        "static_checks": static_checks,
        "test_status": test_status,
        "passed": bool(outcome.passed),
        "execution_error_type": outcome.error_type,
        "execution_error_message": outcome.error_message,
        "verifier_score": verifier_score,
        "verifier_threshold": threshold,
        "verifier_accepted": verifier_accepted,
        "false_accept": false_accept,
        "schema_valid": validation.ok,
        "schema_errors": validation.errors,
        "candidate_manifest_sha256": _sha256_payload(manifest),
    }
    return evaluation, manifest


def default_precondition_probe(config: ExperimentConfig) -> PreconditionReport:  # pragma: no cover - hardware path.
    """Probe local hardware/runtime readiness without mutating the environment."""

    checks: list[JsonDict] = []
    gpu_detail = _nvidia_smi_summary()
    checks.append(
        {
            "resource": "dual_rtx_3090_host",
            "available": gpu_detail.count("RTX 3090") >= 2,
            "detail": gpu_detail,
        }
    )
    try:
        llama_cpp = importlib.import_module("llama_cpp")
        runtime_ok = hasattr(llama_cpp, "Llama")
        runtime_detail = f"llama_cpp imported; Llama={runtime_ok}"
    except Exception as exc:
        runtime_ok = False
        runtime_detail = f"{type(exc).__name__}: {exc}"
    checks.append({"resource": "llama_cpp_runtime", "available": runtime_ok, "detail": runtime_detail})
    sandbox = get_sandbox_status()
    checks.append(
        {
            "resource": "runsc_sandbox",
            "available": bool(sandbox.get("available") and sandbox.get("runtime") == "runsc"),
            "detail": str(sandbox),
        }
    )
    pair = cached_sota_pair(gpu_indices=(0, 1))
    model_specs: list[JsonDict] = []
    runnable: list[JsonDict] = []
    pair_by_hf = {str(row.get("hf_id")): dict(row) for row in pair or []}
    for index, model in enumerate(SOTA_GGUF_MODELS):
        pair_row = pair_by_hf.get(model["hf_id"])
        model_path = str(pair_row.get("model_path")) if pair_row else resolve_cached_gguf(model["hf_id"])
        cached = bool(model_path)
        row = {
            "name": model["name"],
            "hf_id": model["hf_id"],
            "role": model["role"],
            "gpu": int(pair_row.get("gpu")) if pair_row else index % 2,
            "model_path": model_path,
            "cached": cached,
            "selected_for_live_repair": False,
        }
        if cached:
            runnable.append({k: row[k] for k in ("name", "hf_id", "gpu", "model_path")})
        model_specs.append(row)
    if runnable:
        selected_hf = runnable[0]["hf_id"]
        for row in model_specs:
            row["selected_for_live_repair"] = row["hf_id"] == selected_hf
    checks.append(
        {
            "resource": "cached_sota_pair_or_single_headline_gguf",
            "available": bool(pair or runnable),
            "detail": (
                "cached_sota_pair resolved two-model pair"
                if pair
                else "cached_sota_pair returned None; single mandated GGUF resolved"
                if runnable
                else "cached_sota_pair returned None and no mandated GGUF resolved"
            ),
        }
    )
    checks.append(
        {
            "resource": "headline_gguf_cache",
            "available": bool(runnable),
            "detail": ",".join(str(row["hf_id"]) for row in runnable) or "none",
        }
    )
    return PreconditionReport(checks=checks, model_specs=model_specs, runnable_model_specs=runnable)


def default_task_row_provider(config: ExperimentConfig) -> dict[tuple[str, str], JsonDict]:  # pragma: no cover - filesystem path.
    """Load MBPP/HumanEval manifest rows so selected failed candidates can run tests."""

    manifest_config = exp2889.ExperimentConfig(repo_root=config.repo_root)
    resolved, ready = exp2889._resolve_code_manifests(manifest_config)
    if not ready:
        return {}
    rows: dict[tuple[str, str], JsonDict] = {}
    for corpus_key, display in (("mbpp", "MBPP"), ("humaneval", "HumanEval")):
        for row in _read_jsonl(resolved[corpus_key].path):
            stable_id = str(row.get("stable_id") or "")
            if stable_id:
                rows[(display, stable_id)] = dict(row)
    return rows


def llama_cpp_repair_generator(
    *,
    model_path: str,
    main_gpu: int = 0,
    n_ctx: int = 4096,
    n_batch: int = 128,
    n_gpu_layers: int = -1,
    temperature: float = DEFAULT_TEMPERATURE,
) -> RepairGenerator:  # pragma: no cover - live GGUF path.
    """Return a prompt-to-code repair generator backed by llama.cpp."""

    from llama_cpp import Llama

    state: dict[str, Any] = {"llm": None}

    def _ensure_loaded() -> Any:
        if state["llm"] is None:
            state["llm"] = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_ubatch=n_batch,
                n_gpu_layers=n_gpu_layers,
                main_gpu=main_gpu,
                verbose=False,
            )
        return state["llm"]

    def _generate(
        prompt: str,
        seed: int,
        max_tokens: int,
        _model_spec: Mapping[str, Any],
    ) -> GenerationOutcome:
        started = time.monotonic()
        try:
            out = _ensure_loaded()(
                prompt,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                seed=int(seed),
                stop=["\n\n\n"],
            )
        except Exception as exc:
            return GenerationOutcome(
                text="",
                tokens_generated=0,
                duration_s=time.monotonic() - started,
                backend="llama_cpp",
                backend_detail=model_path,
                error=f"{type(exc).__name__}: {exc}",
            )
        return GenerationOutcome(
            text=str(out.get("choices", [{}])[0].get("text", "")),
            tokens_generated=int(out.get("usage", {}).get("completion_tokens") or 0),
            duration_s=time.monotonic() - started,
            backend="llama_cpp",
            backend_detail=model_path,
        )

    return _generate


def _source_precondition_checks(config: ExperimentConfig) -> list[JsonDict]:
    sources = [
        ("exp2940_verifier_threshold", config.exp2940_path, lambda payload: bool(payload)),
        ("exp2946_failed_candidates", config.exp2946_path, lambda payload: bool(payload)),
        (
            "exp2950_repair_prompt_manifest",
            config.exp2950_path,
            lambda payload: payload.get("repair_prompt_manifest_ready") is True,
        ),
        (
            "exp2951_structured_candidate_manifest_adapter",
            config.exp2951_path,
            lambda payload: payload.get("structured_decode_manifest_ready") is True,
        ),
    ]
    checks: list[JsonDict] = []
    for resource, rel_path, ready_fn in sources:
        path = _repo_path(config.repo_root, rel_path)
        payload = _read_json(path) if path.is_file() else {}
        checks.append(
            {
                "resource": resource,
                "available": path.is_file() and ready_fn(payload),
                "detail": str(rel_path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
        )
    return checks


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    verdict: str,
    preconditions_checked: list[JsonDict],
    model_specs: list[JsonDict],
) -> JsonDict:
    empty_sha = _sha256_payload([])
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions_checked,
        "source_artifacts": _source_artifacts(config),
        "model_specs": model_specs,
        "headline_models_used": [],
        "legacy_models_only_for_smoke": False,
        "n_tasks": 0,
        "selected_task_ids": [],
        "selected_repair_set": [],
        "samples_per_mode": config.samples_per_mode,
        "sample_budget_per_mode": 0,
        "baseline_pass_at_1": None,
        "repair_pass_at_1": None,
        "pass_at_1_delta": None,
        "baseline_pass_at_k": None,
        "repair_pass_at_k": None,
        "pass_at_k_delta": None,
        "baseline_syntax_failure_rate": None,
        "repair_syntax_failure_rate": None,
        "syntax_failure_rate_delta": None,
        "baseline_schema_failure_rate": None,
        "repair_schema_failure_rate": None,
        "schema_failure_rate_delta": None,
        "baseline_verifier_acceptance_rate": None,
        "repair_verifier_acceptance_rate": None,
        "verifier_acceptance_rate_delta": None,
        "baseline_false_accept_rate": None,
        "repair_false_accept_rate": None,
        "false_accept_delta": None,
        "false_accept_audit_notes": [verdict],
        "taxonomy_repair_delta_pass": False,
        "candidate_manifest_sha256": empty_sha,
        "candidate_manifests": [],
        "candidate_evaluations": [],
        "reproducibility_checksum": _reproducibility_checksum(
            selected_task_ids=[],
            candidate_manifest_sha256=empty_sha,
            model_specs=model_specs,
            deltas={},
        ),
        "duration_s": _elapsed(config, started),
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _repair_prompt(source: RepairSource, mode: str, templates: Mapping[str, Any]) -> str:
    task_prompt = exp2889._build_prompt(_corpus_key(source.corpus), source.task_row)
    failure_evidence = str(source.candidate_row.get("error_message") or "")
    candidate_code = str(source.candidate_row.get("extracted_code") or source.candidate_row.get("raw_response") or "")
    if mode == BASELINE_MODE:
        return (
            "Repair this Python coding candidate.\n"
            f"Sample: {source.sample_id}\n"
            f"Failure evidence: {failure_evidence}\n"
            f"Task context: {task_prompt}\n"
            f"Candidate code:\n{candidate_code}\n"
            "Return only corrected Python code. Preserve the public function signature."
        )
    label = source.original_failure_categories[0]
    template = templates.get(label) if isinstance(templates.get(label), Mapping) else {}
    raw_template = str(template.get("template") or "")
    if raw_template:
        return raw_template.format(
            sample_id=source.sample_id,
            failure_evidence=failure_evidence,
            task_prompt=task_prompt,
            candidate_code=candidate_code,
            deterministic_checks=", ".join(exp2950.LABEL_CHECKS.get(label, ())),
        )
    return (
        f"Taxonomy label: {label}\n"
        f"Sample: {source.sample_id}\n"
        f"Failure evidence: {failure_evidence}\n"
        f"Task context: {task_prompt}\n"
        f"Candidate code:\n{candidate_code}\n"
        "Return only corrected Python code."
    )


def _execute_candidate(
    config: ExperimentConfig,
    source: RepairSource,
    repaired_code: str,
    syntax_success: bool,
    static_checks: JsonDict,
    executor: Executor,
) -> ExecutionOutcome:
    if not syntax_success or static_checks["status"] != "passed":
        return ExecutionOutcome(passed=False, error_type=parser_or_static_error(static_checks))
    sandbox_script, _n_tests = exp2889._build_sandbox_script(
        _corpus_key(source.corpus),
        source.task_row,
        repaired_code,
    )
    return executor(sandbox_script, config.sandbox_timeout_s)


def parser_or_static_error(static_checks: Mapping[str, Any]) -> str:
    if static_checks.get("status") == "syntax_error":
        return "SyntaxError"
    return "StaticCheckFailed"


def _candidate_manifest(
    *,
    source: RepairSource,
    mode: str,
    sample_index: int,
    seed: int,
    model_id: str,
    raw_response_ref: str,
    raw_response_text: str,
    repaired_code: str,
    failure_taxonomy: list[str],
    parser_status: str,
    test_status: str,
    verifier_score: float,
) -> JsonDict:
    return {
        "task_id": f"{source.task_key}:{mode}:{sample_index}",
        "prompt_id": f"exp2952:{source.sample_id}:{mode}:s{seed}",
        "model_id": model_id,
        "raw_completion_ref": raw_response_ref,
        "repaired_code": repaired_code,
        "failure_taxonomy": failure_taxonomy,
        "parser_status": parser_status,
        "test_status": test_status,
        "verifier_score": verifier_score,
        "provenance_checksums": {
            "raw_completion_sha256": _sha256_text(raw_response_text),
            "repaired_code_sha256": _sha256_text(repaired_code),
            "manifest_schema_sha256": exp2951.schema_checksum(),
        },
    }


def _mode_metrics(
    evaluations: Sequence[Mapping[str, Any]],
    mode: str,
    selected: Sequence[RepairSource],
) -> JsonDict:
    rows = [row for row in evaluations if row.get("mode") == mode]
    by_task: dict[str, list[Mapping[str, Any]]] = {source.task_key: [] for source in selected}
    for row in rows:
        by_task[str(row.get("task_id"))].append(row)
    per_task: list[JsonDict] = []
    for source in selected:
        task_rows = sorted(by_task[source.task_key], key=lambda row: int(row.get("sample_index") or 0))
        pass_vector = [bool(row.get("passed")) for row in task_rows]
        per_task.append(
            {
                "task_id": source.task_key,
                "pass_vector": pass_vector,
                "pass_at_1": 1.0 if pass_vector and pass_vector[0] else 0.0,
                "pass_at_k": 1.0 if any(pass_vector) else 0.0,
            }
        )
    denominator = len(rows)
    return {
        "mode": mode,
        "candidate_count": denominator,
        "per_task_results": per_task,
        "pass_at_1": _mean([row["pass_at_1"] for row in per_task]),
        "pass_at_k": _mean([row["pass_at_k"] for row in per_task]),
        "syntax_failure_rate": _rate(rows, lambda row: row.get("syntax_success") is False),
        "schema_failure_rate": _rate(rows, lambda row: row.get("schema_valid") is False),
        "verifier_acceptance_rate": _rate(rows, lambda row: row.get("verifier_accepted") is True),
        "false_accept_rate": _rate(rows, lambda row: row.get("false_accept") is True),
    }


def _metric_deltas(baseline: Mapping[str, Any], repair: Mapping[str, Any]) -> JsonDict:
    return {
        "pass_at_1_delta": _delta(repair.get("pass_at_1"), baseline.get("pass_at_1")),
        "pass_at_k_delta": _delta(repair.get("pass_at_k"), baseline.get("pass_at_k")),
        "syntax_failure_rate_delta": _delta(
            repair.get("syntax_failure_rate"),
            baseline.get("syntax_failure_rate"),
        ),
        "schema_failure_rate_delta": _delta(
            repair.get("schema_failure_rate"),
            baseline.get("schema_failure_rate"),
        ),
        "verifier_acceptance_rate_delta": _delta(
            repair.get("verifier_acceptance_rate"),
            baseline.get("verifier_acceptance_rate"),
        ),
        "false_accept_delta": _delta(repair.get("false_accept_rate"), baseline.get("false_accept_rate")),
    }


def _taxonomy_delta_pass(deltas: Mapping[str, Any]) -> bool:
    false_accept_delta = deltas.get("false_accept_delta")
    return bool(
        isinstance(false_accept_delta, int | float)
        and false_accept_delta <= 0
        and (
            _positive(deltas.get("pass_at_1_delta"))
            or _negative(deltas.get("syntax_failure_rate_delta"))
        )
    )


def _static_checks(source: str) -> JsonDict:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _syntax_static_checks()
    imported_roots = _imported_roots(tree)
    unsafe = sorted(root for root in imported_roots if root in UNSAFE_IMPORTS)
    unsupported = sorted(_unsupported_attrs(tree, imported_roots))
    status = "passed" if not unsafe and not unsupported else "failed"
    return {
        "status": status,
        "unsafe_imports": unsafe,
        "unsupported_api_calls": unsupported,
    }


def _syntax_static_checks() -> JsonDict:
    return {"status": "syntax_error", "unsafe_imports": [], "unsupported_api_calls": []}


def _post_repair_taxonomy(
    syntax_success: bool,
    static_checks: Mapping[str, Any],
    passed: bool,
) -> list[str]:
    if not syntax_success:
        return ["syntax_error"]
    if static_checks.get("unsafe_imports"):
        return ["unsupported_import"]
    if static_checks.get("unsupported_api_calls"):
        return ["unsupported_api_hallucination"]
    if passed:
        return ["none"]
    return ["failed_tests"]


def _imported_roots(tree: ast.AST) -> set[str]:
    roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    return roots


def _unsupported_attrs(tree: ast.AST, imported_roots: set[str]) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            root = _attribute_root(node.func.value)
            pair = (root, node.func.attr)
            if root in imported_roots and pair in UNSUPPORTED_ATTRS:
                calls.add(".".join(pair))
    return calls


def _attribute_root(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _attribute_root(node.value)
    return ""


def _runtime_success(syntax_success: bool, outcome: ExecutionOutcome) -> bool:
    return bool(syntax_success and not outcome.timed_out and outcome.error_type in (None, "AssertionError"))


def _candidate_rows(nested_protocol: Mapping[str, Any]) -> list[JsonDict]:
    rows = nested_protocol.get("candidate_results")
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _original_failure_categories(row: Mapping[str, Any]) -> tuple[str, ...]:
    labels = exp2950._candidate_labels(row)
    if labels:
        return tuple(labels)
    if row.get("syntax_success") is False:
        return ("syntax_error",)
    return ("failed_tests",)


def _selected_source_row(source: RepairSource) -> JsonDict:
    return {
        "task_id": source.task_key,
        "corpus": source.corpus,
        "stable_id": source.stable_id,
        "sample_id": source.sample_id,
        "candidate_seed": source.candidate_row.get("random_seed"),
        "candidate_index": source.candidate_row.get("candidate_index"),
        "original_failure_categories": list(source.original_failure_categories),
    }


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    rel_paths = [
        config.exp2940_path,
        config.exp2946_path,
        config.nested_exp2946_path,
        config.exp2950_path,
        config.exp2951_path,
    ]
    rows: list[JsonDict] = []
    for rel_path in rel_paths:
        path = _repo_path(config.repo_root, rel_path)
        rows.append(
            {
                "path": str(rel_path),
                "present": path.is_file(),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def _nested_protocol_path(config: ExperimentConfig, exp2946: Mapping[str, Any]) -> Path:
    return Path(str(exp2946.get("protocol_artifact_path") or config.nested_exp2946_path))


def _sample_id(row: Mapping[str, Any]) -> str:
    corpus = _display_corpus(str(row.get("corpus") or "unknown"))
    stable_id = str(row.get("stable_id") or "unknown")
    candidate_index = row.get("candidate_index")
    seed = row.get("random_seed")
    return f"{corpus}:{stable_id}:c{candidate_index}:s{seed}"


def _candidate_seed(config: ExperimentConfig, mode: str, task_index: int, sample_index: int) -> int:
    mode_offset = 100 if mode == REPAIR_MODE else 0
    return config.random_seed + mode_offset + task_index * config.samples_per_mode + sample_index


def _write_raw_response(
    config: ExperimentConfig,
    *,
    source: RepairSource,
    mode: str,
    sample_index: int,
    seed: int,
    text: str,
) -> str:
    filename = f"{mode}_{_safe_name(source.stable_id)}_r{sample_index}_seed_{seed}.txt"
    path = config.raw_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path.resolve().relative_to(config.repo_root.resolve()))


def _verifier_threshold(exp2940: Mapping[str, Any]) -> float:
    return _number((exp2940.get("max_f1_operating_point") or {}).get("threshold"), 1.0)


def _false_accept_notes(
    baseline: Mapping[str, Any],
    repair: Mapping[str, Any],
    deltas: Mapping[str, Any],
) -> list[str]:
    delta = deltas.get("false_accept_delta")
    if isinstance(delta, int | float) and delta < 0:
        return [
            "false accepts decreased under taxonomy-guided repair",
            f"baseline={baseline.get('false_accept_rate')}, repair={repair.get('false_accept_rate')}",
        ]
    if delta == 0:
        return ["false accepts unchanged under taxonomy-guided repair"]
    return ["false accepts increased or unavailable; taxonomy delta gate remains closed"]


def _complete_verdict(deltas: Mapping[str, Any], delta_pass: bool) -> str:
    return (
        "complete: taxonomy-guided repair delta passed"
        if delta_pass
        else "complete: taxonomy-guided repair did not clear the guarded delta gate"
    ) + (
        f"; pass@1_delta={deltas.get('pass_at_1_delta')}, "
        f"syntax_failure_rate_delta={deltas.get('syntax_failure_rate_delta')}, "
        f"false_accept_delta={deltas.get('false_accept_delta')}"
    )


def _reproducibility_checksum(
    *,
    selected_task_ids: Sequence[str],
    candidate_manifest_sha256: str,
    model_specs: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, Any],
) -> str:
    return _sha256_payload(
        {
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "deltas": dict(deltas),
            "model_specs": [dict(row) for row in model_specs],
            "selected_task_ids": list(selected_task_ids),
        }
    )


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "task"


def _display_corpus(value: str) -> str:
    lowered = value.lower()
    if lowered == "mbpp":
        return "MBPP"
    if lowered == "humaneval":
        return "HumanEval"
    return value


def _corpus_key(value: str) -> str:
    return "mbpp" if value == "MBPP" else "humaneval"


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _rate(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else 0.0


def _delta(new: Any, old: Any) -> float | None:
    if isinstance(new, int | float) and isinstance(old, int | float):
        return float(new) - float(old)
    return None


def _positive(value: Any) -> bool:
    return isinstance(value, int | float) and value > 0


def _negative(value: Any) -> bool:
    return isinstance(value, int | float) and value < 0


def _nvidia_smi_summary() -> str:  # pragma: no cover - hardware path.
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return completed.stdout.strip() or completed.stderr.strip()


def main() -> int:  # pragma: no cover - script entrypoint.
    artifact = write_artifact(
        ExperimentConfig(
            tests_run=(
                ".venv/bin/pytest tests/python/test_experiment_2952_sota_taxonomy_guided_code_repair_eval.py -q",
                ".venv/bin/pytest tests/python -q",
            )
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["honest_verdict"].startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover - script entrypoint.
    raise SystemExit(main())


__all__ = [
    "ARTIFACT",
    "BASELINE_MODE",
    "EXP2940_REL_PATH",
    "EXP2946_REL_PATH",
    "EXP2950_REL_PATH",
    "EXP2951_REL_PATH",
    "ExecutionOutcome",
    "ExperimentConfig",
    "GenerationOutcome",
    "INFERENCE_SUBSTRATE",
    "NESTED_EXP2946_REL_PATH",
    "OUTPUT_FILENAME",
    "PreconditionReport",
    "REPAIR_MODE",
    "REQUIRED_ARTIFACT_FIELDS",
    "REPO_ROOT",
    "_sha256_payload",
    "build_artifact",
    "default_precondition_probe",
    "evaluate_repair_candidate",
    "select_repair_set",
    "write_artifact",
]
