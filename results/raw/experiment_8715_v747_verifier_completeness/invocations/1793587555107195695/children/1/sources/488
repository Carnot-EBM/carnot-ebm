"""Exp 2905 live SOTA code generation with expanded budget and k=8 sampling.

The runner keeps Exp 2889's anti-fabrication gates: clean Exp 2874 runtime
evidence, verified MBPP/HumanEval manifests, GPU/offload availability, and
runsc-only sandbox execution. It then raises the per-prompt token budget and
samples a bounded candidate set per task so pass@1 and pass@k are computed from
actual sandbox outcomes rather than inferred from generated text.

Spec: REQ-CODE-2905, SCENARIO-CODE-2905.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carnot.eval import mbpp_humaneval_generated_code_clean_row as exp2889
from carnot.verify.sandbox import get_sandbox_status

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2905_sota_code_generation_bounded_budget_expansion_v1.json"
EXP2874_REL_PATH = exp2889.EXP2874_REL_PATH
MANIFEST_CONTRACT_REL_PATH = exp2889.MANIFEST_CONTRACT_REL_PATH
CROSS_CORPUS_MATRIX_REL_PATH = exp2889.CROSS_CORPUS_MATRIX_REL_PATH
CODE_CORPORA = exp2889.CODE_CORPORA
DEFAULT_RANDOM_SEED = 2905
DEFAULT_N_TASKS_PER_CORPUS = 2
DEFAULT_K_CANDIDATES_PER_TASK = 8
DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SANDBOX_TIMEOUT_S = 10.0
DEFAULT_DURATION_FLOOR_S = 60.0
LIVE_MODEL_PRINCIPLE = (
    "Forward-only declaration that this task actually invokes the live SOTA "
    "model (60s duration floor applies)."
)

GenerationOutcome = exp2889.GenerationOutcome
ExecutionOutcome = exp2889.ExecutionOutcome
Generator = exp2889.Generator
Executor = exp2889.Executor
execute_script_in_sandbox = exp2889.execute_script_in_sandbox
llama_cpp_generator = exp2889.llama_cpp_generator

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "n_tasks_per_corpus",
    "k_candidates_per_task",
    "per_task_pass_at_1",
    "per_task_pass_at_k",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict grounded in live inference and sandbox results.",
    "inference_substrate": "Always live_llm_inference for Exp 2905.",
    "model_specs": "Carries the live-model principle and Exp 2874 selected model evidence.",
    "n_tasks_per_corpus": "Bounded deterministic sample size per code corpus.",
    "k_candidates_per_task": "Exactly eight by default so pass@k has a candidate set.",
    "per_task_pass_at_1": "One float per selected task; 1.0 only when candidate 0 passes.",
    "per_task_pass_at_k": "One float per selected task; 1.0 when any sampled candidate passes.",
    "random_seeds_used": "Every seed sent to the live generation backend, in call order.",
    "reproducibility_checksum": "Hash over manifests, selected rows, seeds, model, and budget.",
    "duration_s": "Measured wall time; complete runs must meet the live-inference floor.",
}

SELECTION_RULE = (
    "Select the first N eligible rows in manifest order for each of MBPP and "
    "HumanEval, then sample k candidates per selected task with consecutive "
    "seeds from the experiment seed."
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2905.

    Unit tests inject fake generation and execution. Production uses the live
    llama.cpp-backed generator selected from Exp 2874 evidence.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    manifest_contract_path: Path = MANIFEST_CONTRACT_REL_PATH
    cross_corpus_matrix_path: Path = CROSS_CORPUS_MATRIX_REL_PATH
    exp2874_path: Path = EXP2874_REL_PATH
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


def extract_python_candidate(text: str) -> str:
    """Return the most code-like fenced block, falling back to raw text."""

    blocks = [match.group(1).strip() for match in _CODE_BLOCK_RE.finditer(text)]
    for block in blocks:
        if "def " in block:
            return block.rstrip()
    if blocks:
        return blocks[0].rstrip()
    return text.strip()


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _model_specs(exp2874: exp2889.Exp2874Evidence | None) -> dict[str, Any]:
    return {
        "principle": LIVE_MODEL_PRINCIPLE,
        "selected_model_hf_id": exp2874.selected_model_hf_id if exp2874 else "",
        "selected_model_path": exp2874.selected_model_path if exp2874 else "",
        "selected_model_fingerprint": exp2874.selected_model_fingerprint if exp2874 else "",
        "upstream_exp2874_model_specs": list(exp2874.model_specs) if exp2874 else [],
    }


def _source_artifacts(
    config: ExperimentConfig,
    resolved: dict[str, exp2889.ManifestResolution] | None,
) -> tuple[list[str], dict[str, str]]:
    paths = [
        exp2889._repo_path(config.repo_root, config.exp2874_path),
        exp2889._repo_path(config.repo_root, config.manifest_contract_path),
        exp2889._repo_path(config.repo_root, config.cross_corpus_matrix_path),
    ]
    if resolved is not None:
        paths.extend([resolved["mbpp"].path, resolved["humaneval"].path])
    existing = [path for path in paths if path.is_file()]
    names = [exp2889._source_name(config.repo_root, path) for path in existing]
    return names, {name: exp2889._sha256(path) for name, path in zip(names, existing, strict=True)}


def _reproducibility_checksum(
    *,
    config: ExperimentConfig,
    exp2874: exp2889.Exp2874Evidence | None,
    resolved: dict[str, exp2889.ManifestResolution] | None,
    selected_task_ids: Sequence[str],
    random_seeds_used: Sequence[int],
) -> str:
    manifest_sha256 = (
        {corpus: resolved[corpus].declared_sha256 for corpus in CODE_CORPORA}
        if resolved is not None
        else {}
    )
    payload = {
        "fingerprint": exp2874.selected_model_fingerprint if exp2874 else "",
        "k_candidates_per_task": int(config.k_candidates_per_task),
        "manifest_sha256": manifest_sha256,
        "max_tokens": int(config.max_tokens),
        "n_tasks_per_corpus": int(config.n_tasks_per_corpus),
        "random_seeds_used": list(random_seeds_used),
        "selected_task_ids": list(selected_task_ids),
        "selection_rule": SELECTION_RULE,
        "temperature": float(config.temperature),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _base_artifact(
    config: ExperimentConfig,
    started: float,
    resolved: dict[str, exp2889.ManifestResolution] | None,
    exp2874: exp2889.Exp2874Evidence | None,
) -> dict[str, Any]:
    source_artifacts, source_sha = _source_artifacts(config, resolved)
    manifest_paths = (
        {corpus: str(resolved[corpus].path) for corpus in CODE_CORPORA}
        if resolved is not None
        else {}
    )
    manifest_declared = (
        {corpus: resolved[corpus].declared_sha256 for corpus in CODE_CORPORA}
        if resolved is not None
        else {}
    )
    manifest_verified = (
        {corpus: resolved[corpus].ready for corpus in CODE_CORPORA}
        if resolved is not None
        else {}
    )
    return {
        "artifact": "experiment_2905_sota_code_generation_bounded_budget_expansion_v1",
        "schema": "carnot.sota_code_generation_bounded_budget_expansion.v1",
        "honest_verdict": "blocked_preconditions",
        "inference_substrate": "live_llm_inference",
        "model_specs": _model_specs(exp2874),
        "source_artifacts": source_artifacts,
        "source_artifact_sha256": source_sha,
        "manifest_paths": manifest_paths,
        "manifest_declared_sha256": manifest_declared,
        "manifest_checksum_verified": manifest_verified,
        "manifest_contract_ready": False,
        "selection_rule": SELECTION_RULE,
        "selected_task_ids": [],
        "n_tasks_per_corpus": int(config.n_tasks_per_corpus),
        "k_candidates_per_task": int(config.k_candidates_per_task),
        "max_tokens_per_candidate": int(config.max_tokens),
        "temperature": float(config.temperature),
        "per_task_pass_at_1": [],
        "per_task_pass_at_k": [],
        "aggregate_pass_at_1": None,
        "aggregate_pass_at_k": None,
        "pass_at_k_exceeds_pass_at_1": False,
        "random_seeds_used": [],
        "task_results": [],
        "candidate_results": [],
        "candidate_generation_clean": False,
        "deterministic_execution_used": False,
        "sandbox_status": "",
        "blocked_reason": "",
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_floor_s": float(config.duration_floor_s),
        "duration_s": max(0.0, config.clock() - started),
        "reproducibility_checksum": _reproducibility_checksum(
            config=config,
            exp2874=exp2874,
            resolved=resolved,
            selected_task_ids=(),
            random_seeds_used=(),
        ),
    }


def _blocked(artifact: dict[str, Any], verdict: str, reason: str) -> dict[str, Any]:
    artifact["honest_verdict"] = verdict
    artifact["blocked_reason"] = reason
    artifact["candidate_generation_clean"] = False
    return artifact


def _candidate_result(
    *,
    corpus: str,
    row: dict[str, Any],
    generator: Generator,
    executor: Executor,
    config: ExperimentConfig,
    seed: int,
    candidate_index: int,
) -> dict[str, Any]:
    prompt = exp2889._build_prompt(corpus, row)
    generation = generator(corpus, row, seed, config.max_tokens)
    base = {
        "corpus": "MBPP" if corpus == "mbpp" else "HumanEval",
        "stable_id": str(row["stable_id"]),
        "candidate_index": int(candidate_index),
        "random_seed": int(seed),
        "row_sha256": exp2889._stable_json_sha256(row),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "generated_text": generation.text,
        "generated_text_sha256": hashlib.sha256(generation.text.encode("utf-8")).hexdigest(),
        "tokens_generated": int(generation.tokens_generated),
        "generation_duration_s": float(generation.duration_s),
        "generation_backend": generation.backend,
        "generation_backend_detail": generation.backend_detail,
    }
    if generation.error is not None or not generation.text.strip():
        return {
            **base,
            "extracted_code": "",
            "execution_payload_sha256": "",
            "generation_error": generation.error or "empty_generation",
            "n_tests": 0,
            "executed": False,
            "passed": False,
            "error_type": "GenerationFailed",
            "error_message": generation.error or "empty_generation",
            "timed_out": False,
            "row_status": "blocked_generation",
        }

    extracted_code = extract_python_candidate(generation.text)
    sandbox_script, n_tests = exp2889._build_sandbox_script(corpus, row, extracted_code)
    outcome = executor(sandbox_script, config.sandbox_timeout_s)
    return {
        **base,
        "extracted_code": extracted_code,
        "execution_payload_sha256": hashlib.sha256(sandbox_script.encode("utf-8")).hexdigest(),
        "generation_error": None,
        "n_tests": n_tests,
        "executed": True,
        "passed": bool(outcome.passed),
        "error_type": outcome.error_type,
        "error_message": outcome.error_message,
        "timed_out": bool(outcome.timed_out),
        "row_status": "candidate_passed" if outcome.passed else "candidate_failed",
    }


def _selected_tasks(
    resolved: dict[str, exp2889.ManifestResolution],
    n_tasks_per_corpus: int,
) -> list[tuple[str, dict[str, Any]]]:
    selected = exp2889._select_rows(resolved, n_tasks_per_corpus)
    return [(corpus, row) for corpus in CODE_CORPORA for row in selected[corpus]]


def build_experiment_artifact(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    generator: Generator,
    executor: Executor = execute_script_in_sandbox,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
) -> dict[str, Any]:
    """Build the Exp 2905 artifact from live-generation and sandbox outcomes."""

    started = config.start_time()
    exp2874 = exp2889._load_exp2874_evidence(config)
    artifact = _base_artifact(config, started, None, exp2874)
    if exp2874 is None or not exp2874.sota_runtime_clean:
        return _blocked(
            artifact,
            "blocked_exp2874_sota_runtime_not_clean",
            "Exp 2874 sota_runtime_clean is missing or False; refusing live generation.",
        )
    if not exp2874.selected_model_path or not Path(exp2874.selected_model_path).is_file():
        return _blocked(
            artifact,
            "blocked_selected_model_path_missing",
            "Exp 2874 selected_model_path does not resolve to an on-disk GGUF file.",
        )
    if not exp2874.llama_cpp_supports_gpu_offload or not exp2874.gpu_available:
        return _blocked(
            artifact,
            "blocked_gpu_offload_unavailable",
            "llama.cpp GPU offload or nvidia-smi inventory is missing on this host.",
        )

    resolved, manifest_contract_ready = exp2889._resolve_code_manifests(config)
    artifact = _base_artifact(config, started, resolved, exp2874)
    artifact["manifest_contract_ready"] = manifest_contract_ready
    if not manifest_contract_ready:
        return _blocked(
            artifact,
            "blocked_manifest_contract",
            "Eval manifest contract checksums failed verification.",
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

    selected_tasks = _selected_tasks(resolved, config.n_tasks_per_corpus)
    selected_task_ids = [str(row["stable_id"]) for _corpus, row in selected_tasks]
    artifact["selected_task_ids"] = selected_task_ids
    if not selected_tasks:
        return _blocked(
            artifact,
            "blocked_no_eligible_code_rows",
            "No eligible MBPP or HumanEval rows with canonical code + tests.",
        )

    random_seeds_used: list[int] = []
    candidate_results: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    per_task_pass_at_1: list[float] = []
    per_task_pass_at_k: list[float] = []

    for task_index, (corpus, row) in enumerate(selected_tasks):
        start = len(candidate_results)
        for candidate_index in range(config.k_candidates_per_task):
            seed = config.random_seed + task_index * config.k_candidates_per_task + candidate_index
            random_seeds_used.append(seed)
            candidate_results.append(
                _candidate_result(
                    corpus=corpus,
                    row=row,
                    generator=generator,
                    executor=executor,
                    config=config,
                    seed=seed,
                    candidate_index=candidate_index,
                )
            )
        task_candidates = candidate_results[start:]
        pass_at_1 = 1.0 if task_candidates and task_candidates[0]["passed"] else 0.0
        pass_at_k = 1.0 if any(candidate["passed"] for candidate in task_candidates) else 0.0
        first_passing = next(
            (candidate["candidate_index"] for candidate in task_candidates if candidate["passed"]),
            None,
        )
        per_task_pass_at_1.append(pass_at_1)
        per_task_pass_at_k.append(pass_at_k)
        task_results.append(
            {
                "corpus": "MBPP" if corpus == "mbpp" else "HumanEval",
                "stable_id": str(row["stable_id"]),
                "candidate_count": len(task_candidates),
                "pass_at_1": pass_at_1,
                "pass_at_k": pass_at_k,
                "first_passing_candidate_index": first_passing,
            }
        )

    artifact["duration_s"] = max(0.0, config.clock() - started)
    candidate_generation_clean = bool(candidate_results) and all(
        candidate["generation_error"] is None and candidate["executed"]
        for candidate in candidate_results
    )
    deterministic_execution_used = any(candidate["executed"] for candidate in candidate_results)
    artifact.update(
        {
            "per_task_pass_at_1": per_task_pass_at_1,
            "per_task_pass_at_k": per_task_pass_at_k,
            "aggregate_pass_at_1": _mean(per_task_pass_at_1),
            "aggregate_pass_at_k": _mean(per_task_pass_at_k),
            "pass_at_k_exceeds_pass_at_1": any(
                k_value > one_value
                for one_value, k_value in zip(per_task_pass_at_1, per_task_pass_at_k, strict=True)
            ),
            "random_seeds_used": random_seeds_used,
            "task_results": task_results,
            "candidate_results": candidate_results,
            "candidate_generation_clean": candidate_generation_clean,
            "deterministic_execution_used": deterministic_execution_used,
            "reproducibility_checksum": _reproducibility_checksum(
                config=config,
                exp2874=exp2874,
                resolved=resolved,
                selected_task_ids=selected_task_ids,
                random_seeds_used=random_seeds_used,
            ),
        }
    )

    if artifact["duration_s"] < config.duration_floor_s:
        return _blocked(
            artifact,
            "blocked_duration_floor_not_met",
            (
                "Live inference duration was shorter than the declared 60s floor; "
                "artifact remains non-complete even though candidate outcomes are recorded."
            ),
        )
    if candidate_generation_clean:
        artifact["honest_verdict"] = (
            "complete: bounded-budget k=8 live SOTA code generation executed with "
            f"pass@1={artifact['aggregate_pass_at_1']:.4f} and "
            f"pass@k={artifact['aggregate_pass_at_k']:.4f}"
        )
        artifact["blocked_reason"] = ""
        return artifact
    return _blocked(
        artifact,
        "blocked_generation_or_execution_unclean",
        "At least one candidate failed generation or did not reach sandbox execution.",
    )


def write_experiment_artifact(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    generator: Generator,
    executor: Executor = execute_script_in_sandbox,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
) -> dict[str, Any]:
    """Build and persist the Exp 2905 artifact under ``results/``."""

    artifact = build_experiment_artifact(
        config,
        generator=generator,
        executor=executor,
        sandbox_status_provider=sandbox_status_provider,
    )
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def run_experiment(
    config: ExperimentConfig = ExperimentConfig(),
    *,
    generator: Generator | None = None,
    executor: Executor = execute_script_in_sandbox,
    sandbox_status_provider: Callable[[], dict[str, Any]] = get_sandbox_status,
) -> dict[str, Any]:
    """Run Exp 2905 with the selected live SOTA GGUF unless a generator is injected."""

    if generator is None:
        exp2874 = exp2889._load_exp2874_evidence(config)
        if exp2874 is not None and exp2874.selected_model_path:
            generator = llama_cpp_generator(
                model_path=exp2874.selected_model_path,
                temperature=config.temperature,
            )
        else:  # pragma: no cover - build_experiment_artifact blocks before calling it.
            generator = lambda *_args, **_kwargs: GenerationOutcome(
                text="",
                tokens_generated=0,
                duration_s=0.0,
                backend="unavailable",
                error="missing_exp2874_model_path",
            )
    return write_experiment_artifact(
        config,
        generator=generator,
        executor=executor,
        sandbox_status_provider=sandbox_status_provider,
    )


__all__ = [
    "CROSS_CORPUS_MATRIX_REL_PATH",
    "DEFAULT_K_CANDIDATES_PER_TASK",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_N_TASKS_PER_CORPUS",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TEMPERATURE",
    "EXP2874_REL_PATH",
    "ExecutionOutcome",
    "ExperimentConfig",
    "Generator",
    "GenerationOutcome",
    "LIVE_MODEL_PRINCIPLE",
    "MANIFEST_CONTRACT_REL_PATH",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_experiment_artifact",
    "extract_python_candidate",
    "llama_cpp_generator",
    "run_experiment",
    "write_experiment_artifact",
]
