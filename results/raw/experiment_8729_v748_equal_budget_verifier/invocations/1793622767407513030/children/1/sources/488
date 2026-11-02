"""Exp 2910 corrected SOTA code-generation artifact.

This runner exists because Exp 2905 used only two tasks per corpus and omitted
top-level seed provenance. Exp 2910 reruns the same MBPP/HumanEval-style
surface with a defensible bounded sample, explicit SOTA GGUF cache checks,
raw-response capture, and pass@1/pass@k methodology checks.

Spec: REQ-CODE-2910, SCENARIO-CODE-2910.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carnot.eval import mbpp_humaneval_generated_code_clean_row as exp2889
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf
from carnot.verify.sandbox import get_sandbox_status

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2910_sota_code_generation_corrigendum_v2.json"
RAW_RESPONSE_REL_DIR = Path("results/raw/experiment_2910_sota_code_generation_corrigendum_v2")
MANIFEST_CONTRACT_REL_PATH = exp2889.MANIFEST_CONTRACT_REL_PATH
CROSS_CORPUS_MATRIX_REL_PATH = exp2889.CROSS_CORPUS_MATRIX_REL_PATH
CODE_CORPORA = exp2889.CODE_CORPORA
DEFAULT_RANDOM_SEED = 2910
DEFAULT_N_TASKS_PER_CORPUS = 20
DEFAULT_K_CANDIDATES_PER_TASK = 8
DEFAULT_MAX_TOKENS = 192
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SANDBOX_TIMEOUT_S = 10.0
DEFAULT_DURATION_FLOOR_S = 60.0
INFERENCE_SUBSTRATE = "live_llm_inference"

GenerationOutcome = exp2889.GenerationOutcome
ExecutionOutcome = exp2889.ExecutionOutcome
Executor = exp2889.Executor
execute_script_in_sandbox = exp2889.execute_script_in_sandbox

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "codegen_corrigendum_ready",
    "flagged_adversarial",
    "methodology_note",
    "random_seed",
    "random_seeds_used",
    "model_specs",
    "models_used",
    "cached_sota_pair_used",
    "legacy_smoke_only",
    "n_tasks_per_corpus",
    "k_candidates_per_task",
    "aggregate_pass_at_1",
    "aggregate_pass_at_k",
    "pass_at_k_exceeds_pass_at_1",
    "per_task_results",
    "raw_response_dir",
    "inference_substrate",
    "duration_s",
    "run_date",
)

METHODOLOGY_NOTE = (
    "pass@1 and pass@k are distinct metrics: pass@1 counts only the first "
    "candidate for a task, while pass@k counts whether any of the k bounded "
    "candidates for the same task passes the executable tests. Equal aggregate "
    "values are not accepted silently; when they occur, every task must explain "
    "whether candidate 0 already passed or no candidate in the bounded set passed."
)

SELECTION_RULE = (
    "Select the first N eligible rows in manifest order for each of MBPP and "
    "HumanEval, then generate exactly k candidates per task using consecutive "
    "candidate seeds from top-level random_seed=2910."
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal verdict; blocked verdicts do not report headline rates.",
    "codegen_corrigendum_ready": "True only after SOTA generation, sandbox execution, and methodology gates complete.",
    "flagged_adversarial": "True for blocked or methodology-unsafe artifacts; false for corrected rows.",
    "random_seed": "Top-level experiment seed required by the corrigendum.",
    "random_seeds_used": "Every per-candidate seed sent to the generation backend.",
    "model_specs": "Mandated local GGUF model specs considered for headline generation.",
    "models_used": "HF IDs actually used for candidate generation.",
    "cached_sota_pair_used": "Whether cached_sota_pair(gpu_indices=(0, 1)) supplied usable specs.",
    "legacy_smoke_only": "True only when no mandated SOTA GGUF is cached, so headline metrics stay null.",
    "per_task_results": "One row per selected task, including the k-length pass vector.",
    "raw_response_dir": "Directory containing raw model responses, one file per candidate.",
}

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ExtractionResult:
    """Classified Python-code extraction result for one raw LLM response."""

    code: str
    extraction_status: str
    extraction_success: bool
    syntax_success: bool
    error_message: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2910."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    raw_response_dir: Path | None = None
    manifest_contract_path: Path = MANIFEST_CONTRACT_REL_PATH
    cross_corpus_matrix_path: Path = CROSS_CORPUS_MATRIX_REL_PATH
    n_tasks_per_corpus: int = DEFAULT_N_TASKS_PER_CORPUS
    k_candidates_per_task: int = DEFAULT_K_CANDIDATES_PER_TASK
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    random_seed: int = DEFAULT_RANDOM_SEED
    sandbox_timeout_s: float = DEFAULT_SANDBOX_TIMEOUT_S
    duration_floor_s: float = DEFAULT_DURATION_FLOOR_S
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def raw_dir(self) -> Path:
        return self.raw_response_dir or self.repo_root / RAW_RESPONSE_REL_DIR


Generator = Callable[[str, dict[str, Any], int, int, dict[str, Any]], GenerationOutcome]
CachedPairProvider = Callable[..., list[dict[str, Any]] | None]
MandatedModelResolver = Callable[[], list[dict[str, Any]]]


def _repo_relative(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _safe_name(value: str) -> str:
    return _SAFE_FILENAME_RE.sub("_", value).strip("_") or "task"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_payload_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_mandated_model_resolver() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, model in enumerate(SOTA_GGUF_MODELS):
        model_path = resolve_cached_gguf(model["hf_id"])
        if model_path is not None and Path(model_path).is_file():
            specs.append(
                {
                    "name": model["name"],
                    "hf_id": model["hf_id"],
                    "gpu": index,
                    "model_path": model_path,
                }
            )
    return specs


def _usable_specs(specs: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for spec in specs or ():
        hf_id = str(spec.get("hf_id") or "")
        model_path = str(spec.get("model_path") or "")
        if hf_id and model_path and Path(model_path).is_file():
            usable.append(dict(spec))
    return usable


def _resolve_generation_specs(
    *,
    cached_pair_provider: CachedPairProvider,
    mandated_model_resolver: MandatedModelResolver,
) -> tuple[list[dict[str, Any]], bool]:
    pair = _usable_specs(cached_pair_provider(gpu_indices=(0, 1)))
    if pair:
        return pair, True
    return _usable_specs(mandated_model_resolver()), False


def extract_python_candidate(text: str) -> ExtractionResult:
    """Return the best Python candidate and classify extraction failures."""

    blocks = [match.group(1).strip() for match in _CODE_BLOCK_RE.finditer(text)]
    status = "python_fence"
    candidate = ""
    for block in blocks:
        if "def " in block:
            candidate = block.rstrip()
            break
    if not candidate and blocks:
        status = "fence_without_function"
        candidate = blocks[0].rstrip()
    if not candidate and "def " in text:
        status = "raw_function"
        candidate = text.strip()
    if "def " not in candidate:
        return ExtractionResult(
            code="",
            extraction_status="no_function_found",
            extraction_success=False,
            syntax_success=False,
            error_message="No Python function definition found in response.",
        )
    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        return ExtractionResult(
            code=candidate,
            extraction_status="syntax_error",
            extraction_success=True,
            syntax_success=False,
            error_message=f"SyntaxError: {exc.msg}",
        )
    return ExtractionResult(
        code=candidate,
        extraction_status=status,
        extraction_success=True,
        syntax_success=True,
    )


def _raw_response_file(config: ExperimentConfig, corpus: str, stable_id: str, seed: int) -> Path:
    filename = f"{corpus}_{_safe_name(stable_id)}_seed_{seed}.txt"
    return config.raw_dir() / filename


def _write_raw_response(config: ExperimentConfig, path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return _repo_relative(config.repo_root, path)


def _runtime_success(executed: bool, extraction: ExtractionResult, outcome: ExecutionOutcome) -> bool:
    if not executed or not extraction.syntax_success or outcome.timed_out:
        return False
    return outcome.error_type in (None, "AssertionError")


def _candidate_result(
    *,
    corpus: str,
    row: dict[str, Any],
    generator: Generator,
    executor: Executor,
    config: ExperimentConfig,
    seed: int,
    candidate_index: int,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    prompt = exp2889._build_prompt(corpus, row)
    generation = generator(corpus, row, seed, config.max_tokens, model_spec)
    raw_path = _raw_response_file(config, corpus, str(row["stable_id"]), seed)
    raw_response_path = _write_raw_response(config, raw_path, generation.text)
    base = {
        "corpus": "MBPP" if corpus == "mbpp" else "HumanEval",
        "stable_id": str(row["stable_id"]),
        "candidate_index": int(candidate_index),
        "random_seed": int(seed),
        "model_hf_id": str(model_spec.get("hf_id") or ""),
        "model_path": str(model_spec.get("model_path") or ""),
        "gpu_index": int(model_spec.get("gpu") or 0),
        "row_sha256": exp2889._stable_json_sha256(row),
        "prompt_sha256": _sha256_text(prompt),
        "raw_response": generation.text,
        "raw_response_path": raw_response_path,
        "raw_response_sha256": _sha256_text(generation.text),
        "tokens_generated": int(generation.tokens_generated),
        "generation_duration_s": float(generation.duration_s),
        "generation_backend": generation.backend,
        "generation_backend_detail": generation.backend_detail,
    }
    if generation.error is not None or not generation.text.strip():
        error = generation.error or "empty_generation"
        return {
            **base,
            "extracted_code": "",
            "extraction_status": "generation_failed",
            "extraction_success": False,
            "syntax_success": False,
            "runtime_success": False,
            "execution_payload_sha256": "",
            "generation_error": error,
            "n_tests": 0,
            "executed": False,
            "passed": False,
            "error_type": "GenerationFailed",
            "error_message": error,
            "timed_out": False,
            "row_status": "candidate_generation_failed",
        }

    extraction = extract_python_candidate(generation.text)
    if not extraction.extraction_success:
        return {
            **base,
            "extracted_code": extraction.code,
            "extraction_status": extraction.extraction_status,
            "extraction_success": False,
            "syntax_success": False,
            "runtime_success": False,
            "execution_payload_sha256": "",
            "generation_error": None,
            "n_tests": 0,
            "executed": False,
            "passed": False,
            "error_type": "ExtractionFailed",
            "error_message": extraction.error_message,
            "timed_out": False,
            "row_status": "candidate_extraction_failed",
        }

    sandbox_script, n_tests = exp2889._build_sandbox_script(corpus, row, extraction.code)
    outcome = executor(sandbox_script, config.sandbox_timeout_s)
    runtime_success = _runtime_success(True, extraction, outcome)
    if outcome.passed:
        row_status = "candidate_passed"
    elif not extraction.syntax_success:
        row_status = "candidate_syntax_failed"
    else:
        row_status = "candidate_failed"
    return {
        **base,
        "extracted_code": extraction.code,
        "extraction_status": extraction.extraction_status,
        "extraction_success": True,
        "syntax_success": extraction.syntax_success,
        "runtime_success": runtime_success,
        "execution_payload_sha256": _sha256_text(sandbox_script),
        "generation_error": None,
        "n_tests": n_tests,
        "executed": True,
        "passed": bool(outcome.passed),
        "error_type": outcome.error_type,
        "error_message": outcome.error_message or extraction.error_message,
        "timed_out": bool(outcome.timed_out),
        "row_status": row_status,
    }


def _pass_metric_explanation(pass_vector: Sequence[bool]) -> str:
    if pass_vector and pass_vector[0] and any(pass_vector):
        return "candidate_0_passed_so_pass_at_1_and_pass_at_k_are_equal_for_this_task"
    if any(pass_vector):
        first_passing = next(index for index, passed in enumerate(pass_vector) if passed)
        return f"candidate_{first_passing}_passed_after_candidate_0_failed"
    return "no_candidate_in_the_bounded_k_set_passed_for_this_task"


def _reproducibility_checksum(
    *,
    config: ExperimentConfig,
    model_specs: Sequence[dict[str, Any]],
    selected_task_ids: Sequence[str],
    random_seeds_used: Sequence[int],
) -> str:
    payload = {
        "k_candidates_per_task": int(config.k_candidates_per_task),
        "max_tokens": int(config.max_tokens),
        "model_specs": [
            {
                "hf_id": str(spec.get("hf_id") or ""),
                "model_path": str(spec.get("model_path") or ""),
                "gpu": int(spec.get("gpu") or 0),
            }
            for spec in model_specs
        ],
        "n_tasks_per_corpus": int(config.n_tasks_per_corpus),
        "random_seed": int(config.random_seed),
        "random_seeds_used": list(random_seeds_used),
        "selected_task_ids": list(selected_task_ids),
        "selection_rule": SELECTION_RULE,
        "temperature": float(config.temperature),
    }
    return _stable_payload_sha(payload)[:16]


def _base_artifact(
    config: ExperimentConfig,
    started: float,
    model_specs: Sequence[dict[str, Any]],
    cached_pair_used: bool,
) -> dict[str, Any]:
    raw_dir = config.raw_dir()
    return {
        "artifact": "experiment_2910_sota_code_generation_corrigendum_v2",
        "schema": "carnot.sota_code_generation_corrigendum.v2",
        "honest_verdict": "blocked_preconditions",
        "codegen_corrigendum_ready": False,
        "flagged_adversarial": True,
        "methodology_note": METHODOLOGY_NOTE,
        "methodology_check": "",
        "random_seed": int(config.random_seed),
        "random_seeds_used": [],
        "model_specs": [dict(spec) for spec in model_specs],
        "models_used": [],
        "cached_sota_pair_used": bool(cached_pair_used),
        "legacy_smoke_only": not bool(model_specs),
        "n_tasks_per_corpus": int(config.n_tasks_per_corpus),
        "k_candidates_per_task": int(config.k_candidates_per_task),
        "max_tokens_per_candidate": int(config.max_tokens),
        "temperature": float(config.temperature),
        "aggregate_pass_at_1": None,
        "aggregate_pass_at_k": None,
        "pass_at_k_exceeds_pass_at_1": False,
        "per_task_results": [],
        "candidate_results": [],
        "raw_response_dir": _repo_relative(config.repo_root, raw_dir),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "selection_rule": SELECTION_RULE,
        "selected_task_ids": [],
        "manifest_contract_ready": False,
        "sandbox_status": "",
        "blocked_reason": "",
        "candidate_generation_clean": False,
        "deterministic_execution_used": False,
        "reproducibility_checksum": _reproducibility_checksum(
            config=config,
            model_specs=model_specs,
            selected_task_ids=(),
            random_seeds_used=(),
        ),
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_floor_s": float(config.duration_floor_s),
        "duration_s": max(0.0, config.clock() - started),
        "run_date": RUN_DATE,
    }


def _blocked(artifact: dict[str, Any], verdict: str, reason: str) -> dict[str, Any]:
    artifact["honest_verdict"] = verdict
    artifact["blocked_reason"] = reason
    artifact["codegen_corrigendum_ready"] = False
    artifact["flagged_adversarial"] = True
    return artifact


def build_experiment_artifact(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    generator: Generator,
    executor: Executor = execute_script_in_sandbox,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    mandated_model_resolver: MandatedModelResolver = _default_mandated_model_resolver,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
) -> dict[str, Any]:
    """Build the Exp 2910 artifact from live generation and sandbox outcomes."""

    started = config.start_time()
    model_specs, cached_pair_used = _resolve_generation_specs(
        cached_pair_provider=cached_pair_provider,
        mandated_model_resolver=mandated_model_resolver,
    )
    artifact = _base_artifact(config, started, model_specs, cached_pair_used)
    if not model_specs:
        return _blocked(
            artifact,
            "blocked_sota_gguf_cache_missing",
            "cached_sota_pair returned no usable mandated GGUFs and no individual mandated GGUF path is cached.",
        )

    resolved, manifest_contract_ready = exp2889._resolve_code_manifests(config)
    artifact["manifest_contract_ready"] = manifest_contract_ready
    artifact["manifest_paths"] = {
        corpus: str(resolved[corpus].path) for corpus in CODE_CORPORA
    }
    artifact["manifest_declared_sha256"] = {
        corpus: resolved[corpus].declared_sha256 for corpus in CODE_CORPORA
    }
    if not manifest_contract_ready:
        return _blocked(
            artifact,
            "blocked_manifest_contract",
            "Eval manifest contract checksums failed verification.",
        )

    selected_by_corpus = exp2889._select_rows(resolved, config.n_tasks_per_corpus)
    if any(len(selected_by_corpus[corpus]) < config.n_tasks_per_corpus for corpus in CODE_CORPORA):
        return _blocked(
            artifact,
            "blocked_insufficient_local_code_rows",
            "Local manifests do not contain the requested eligible MBPP and HumanEval task counts.",
        )

    sandbox_status = sandbox_status_provider()
    sandbox_ready = bool(
        sandbox_status.get("available") and sandbox_status.get("runtime") == "runsc"
    )
    artifact["sandbox_status"] = (
        "available: runsc" if sandbox_ready else "blocked_sandbox: runsc unavailable"
    )
    if not sandbox_ready:
        return _blocked(
            artifact,
            "blocked_sandbox",
            "runsc sandbox is not available; in-process fallback is forbidden.",
        )

    selected_tasks = [(corpus, row) for corpus in CODE_CORPORA for row in selected_by_corpus[corpus]]
    selected_task_ids = [str(row["stable_id"]) for _corpus, row in selected_tasks]
    artifact["selected_task_ids"] = selected_task_ids

    model_spec = dict(model_specs[0])
    random_seeds_used: list[int] = []
    candidate_results: list[dict[str, Any]] = []
    per_task_results: list[dict[str, Any]] = []
    per_task_pass_at_1: list[float] = []
    per_task_pass_at_k: list[float] = []

    for task_index, (corpus, row) in enumerate(selected_tasks):
        task_candidates: list[dict[str, Any]] = []
        for candidate_index in range(config.k_candidates_per_task):
            seed = config.random_seed + task_index * config.k_candidates_per_task + candidate_index
            random_seeds_used.append(seed)
            result = _candidate_result(
                corpus=corpus,
                row=row,
                generator=generator,
                executor=executor,
                config=config,
                seed=seed,
                candidate_index=candidate_index,
                model_spec=model_spec,
            )
            candidate_results.append(result)
            task_candidates.append(result)
        pass_vector = [bool(candidate["passed"]) for candidate in task_candidates]
        pass_at_1 = 1.0 if pass_vector and pass_vector[0] else 0.0
        pass_at_k = 1.0 if any(pass_vector) else 0.0
        per_task_pass_at_1.append(pass_at_1)
        per_task_pass_at_k.append(pass_at_k)
        first_passing = next(
            (index for index, passed in enumerate(pass_vector) if passed),
            None,
        )
        per_task_results.append(
            {
                "corpus": "MBPP" if corpus == "mbpp" else "HumanEval",
                "stable_id": str(row["stable_id"]),
                "candidate_count": len(task_candidates),
                "candidate_seeds": [candidate["random_seed"] for candidate in task_candidates],
                "pass_vector": pass_vector,
                "pass_at_1": pass_at_1,
                "pass_at_k": pass_at_k,
                "first_passing_candidate_index": first_passing,
                "pass_metric_explanation": _pass_metric_explanation(pass_vector),
            }
        )

    aggregate_pass_at_1 = _mean(per_task_pass_at_1)
    aggregate_pass_at_k = _mean(per_task_pass_at_k)
    pass_at_k_exceeds = any(
        pass_k > pass_1
        for pass_1, pass_k in zip(per_task_pass_at_1, per_task_pass_at_k, strict=True)
    )
    equal_metrics_explained = bool(per_task_results) and all(
        task["pass_metric_explanation"] for task in per_task_results
    )
    methodology_check = (
        "pass_at_k_exceeds_pass_at_1"
        if pass_at_k_exceeds
        else "equal_metrics_explained_per_task"
        if aggregate_pass_at_1 == aggregate_pass_at_k and equal_metrics_explained
        else "blocked_equal_metrics_without_per_task_explanation"
    )
    artifact.update(
        {
            "random_seeds_used": random_seeds_used,
            "models_used": [str(model_spec.get("hf_id") or "")],
            "legacy_smoke_only": False,
            "aggregate_pass_at_1": aggregate_pass_at_1,
            "aggregate_pass_at_k": aggregate_pass_at_k,
            "pass_at_k_exceeds_pass_at_1": pass_at_k_exceeds,
            "per_task_results": per_task_results,
            "candidate_results": candidate_results,
            "candidate_generation_clean": all(
                candidate["generation_error"] is None for candidate in candidate_results
            ),
            "deterministic_execution_used": any(
                candidate["executed"] for candidate in candidate_results
            ),
            "methodology_check": methodology_check,
            "reproducibility_checksum": _reproducibility_checksum(
                config=config,
                model_specs=model_specs,
                selected_task_ids=selected_task_ids,
                random_seeds_used=random_seeds_used,
            ),
            "duration_s": max(0.0, config.clock() - started),
        }
    )

    if methodology_check == "blocked_equal_metrics_without_per_task_explanation":
        return _blocked(
            artifact,
            methodology_check,
            "Aggregate pass@1 and pass@k matched without complete per-task explanations.",
        )
    if artifact["duration_s"] < config.duration_floor_s:
        return _blocked(
            artifact,
            "blocked_duration_floor_not_met",
            "Live inference duration was shorter than the declared 60s floor.",
        )

    artifact["honest_verdict"] = (
        "complete: SOTA code-generation corrigendum executed with "
        f"pass@1={aggregate_pass_at_1:.4f} and pass@k={aggregate_pass_at_k:.4f}"
    )
    artifact["codegen_corrigendum_ready"] = True
    artifact["flagged_adversarial"] = False
    artifact["blocked_reason"] = ""
    return artifact


def write_experiment_artifact(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    generator: Generator,
    executor: Executor = execute_script_in_sandbox,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    mandated_model_resolver: MandatedModelResolver = _default_mandated_model_resolver,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
) -> dict[str, Any]:
    """Build and persist the Exp 2910 artifact under ``results/``."""

    artifact = build_experiment_artifact(
        config,
        generator=generator,
        executor=executor,
        cached_pair_provider=cached_pair_provider,
        mandated_model_resolver=mandated_model_resolver,
        sandbox_status_provider=sandbox_status_provider,
    )
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def llama_cpp_model_generator(*, model_path: str, temperature: float = DEFAULT_TEMPERATURE) -> Generator:
    """Wrap the existing llama.cpp generator with Exp 2910's model-spec argument."""

    base_generator = exp2889.llama_cpp_generator(model_path=model_path, temperature=temperature)

    def _generate(
        corpus: str,
        row: dict[str, Any],
        seed: int,
        max_tokens: int,
        _model_spec: dict[str, Any],
    ) -> GenerationOutcome:
        return base_generator(corpus, row, seed, max_tokens)

    return _generate


def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    generator: Generator | None = None,
    executor: Executor = execute_script_in_sandbox,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    mandated_model_resolver: MandatedModelResolver = _default_mandated_model_resolver,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
) -> dict[str, Any]:
    """Run Exp 2910 with the selected mandated GGUF unless a generator is injected."""

    if generator is None:
        model_specs, _cached_pair_used = _resolve_generation_specs(
            cached_pair_provider=cached_pair_provider,
            mandated_model_resolver=mandated_model_resolver,
        )
        if model_specs:
            generator = llama_cpp_model_generator(
                model_path=str(model_specs[0]["model_path"]),
                temperature=config.temperature,
            )
        else:  # pragma: no cover - build_experiment_artifact blocks before calling it.
            generator = lambda *_args, **_kwargs: GenerationOutcome(
                text="",
                tokens_generated=0,
                duration_s=0.0,
                backend="unavailable",
                error="missing_sota_gguf_cache",
            )
    return write_experiment_artifact(
        config,
        generator=generator,
        executor=executor,
        cached_pair_provider=cached_pair_provider,
        mandated_model_resolver=mandated_model_resolver,
        sandbox_status_provider=sandbox_status_provider,
    )


__all__ = [
    "CROSS_CORPUS_MATRIX_REL_PATH",
    "DEFAULT_K_CANDIDATES_PER_TASK",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_N_TASKS_PER_CORPUS",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TEMPERATURE",
    "ExecutionOutcome",
    "ExperimentConfig",
    "ExtractionResult",
    "GenerationOutcome",
    "INFERENCE_SUBSTRATE",
    "MANIFEST_CONTRACT_REL_PATH",
    "METHODOLOGY_NOTE",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_experiment_artifact",
    "execute_script_in_sandbox",
    "extract_python_candidate",
    "llama_cpp_model_generator",
    "run_experiment",
    "write_experiment_artifact",
]
