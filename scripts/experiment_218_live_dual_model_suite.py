#!/usr/bin/env python3
"""Experiment 218: shared dual-model live benchmark harness.

Builds one checkpointed CLI for the next live benchmark milestone:

- ``gsm8k_semantic``
- ``humaneval_property``
- ``constraint_ir``

The harness keeps paired comparisons honest by:

- restricting the model set to the two target small instruction-tuned models
- sampling cohorts deterministically
- recording one shared prompt seed per case and reusing it across
  ``baseline``, ``verify_only``, and ``verify_repair``
- checkpointing every benchmark/model/mode cell independently so long runs
  can resume without scrambling case order or duplicating results

Usage:
    .venv/bin/python scripts/experiment_218_live_dual_model_suite.py --help

Spec: REQ-VERIFY-025, REQ-VERIFY-026, REQ-VERIFY-027,
SCENARIO-VERIFY-025, SCENARIO-VERIFY-026, SCENARIO-VERIFY-027
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

RUN_DATE = "20260412"
EXPERIMENT = 218
SCHEMA_ARTIFACT = "carnot.live_dual_model_suite.v1"
DEFAULT_SAMPLE_SEED = 218
DEFAULT_MAX_REPAIRS = 3
MODE_ORDER = ("baseline", "verify_only", "verify_repair")
_EXPERIMENT_OUTPUT_RE = re.compile(r"experiment_(\d+)_results\.json$")

MODEL_SPECS: list[dict[str, str]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]

BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "gsm8k_semantic": {
        "title": "Live GSM8K semantic benchmark",
        "case_schema": "gsm8k_semantic.v1",
        "default_sample_size": 200,
        "source_artifacts": [
            "results/experiment_206_results.json",
            "results/experiment_207_results.json",
            "results/monitorability_policy_213.json",
        ],
    },
    "humaneval_property": {
        "title": "Live HumanEval property benchmark",
        "case_schema": "humaneval_property.v1",
        "default_sample_size": 50,
        "source_artifacts": [
            "results/experiment_208_results.json",
            "results/monitorability_policy_213.json",
        ],
    },
    "constraint_ir": {
        "title": "Live prompt-side constraint benchmark",
        "case_schema": "constraint_ir.v1",
        "default_sample_size": 100,
        "source_artifacts": [
            "data/research/constraint_ir_benchmark_211.jsonl",
            "results/experiment_213_results.json",
            "results/monitorability_policy_213.json",
        ],
    },
}


def get_repo_root() -> Path:
    """Resolve the repository root, honoring the usual test override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def safe_slug(text: str) -> str:
    """Convert a label into a filesystem-safe slug."""
    cleaned = text.strip().lower().replace("/", "_").replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in cleaned)


def default_output_path(benchmark: str) -> Path:
    """Return the default artifact path for a benchmark."""
    return get_repo_root() / "results" / f"experiment_218_{benchmark}_results.json"


def default_checkpoint_dir() -> Path:
    """Return the default checkpoint directory."""
    return get_repo_root() / "results" / "checkpoints" / "experiment_218"


def artifact_experiment_id(output_path: Path) -> int:
    """Infer the deliverable experiment id from the output filename when present."""
    match = _EXPERIMENT_OUTPUT_RE.search(output_path.name)
    if match is None:
        return EXPERIMENT
    return int(match.group(1))


def live_inference_mode() -> str:
    """Describe the current run's intended inference mode from env defaults."""
    if os.environ.get("CARNOT_FORCE_LIVE") == "1":
        return "live_cpu" if os.environ.get("CARNOT_FORCE_CPU") == "1" else "live_gpu"
    return "simulated"


def _token_count(tokenizer: Any, text: str) -> int:
    """Count tokens conservatively, falling back to whitespace if needed."""
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, list):
            return len(input_ids)
    except Exception:
        pass
    return len(text.split())


def _round_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _typed_reasoning_parse_status(typed_reasoning: Any) -> str:
    """Return a stable parse-status label for typed reasoning artifacts."""
    if typed_reasoning is None:
        return "unavailable"
    provenance = getattr(typed_reasoning, "provenance", None)
    extraction_method = getattr(provenance, "extraction_method", None)
    if isinstance(extraction_method, str) and extraction_method:
        return extraction_method
    if hasattr(typed_reasoning, "to_dict"):
        payload = typed_reasoning.to_dict()
        provenance_payload = payload.get("provenance", {})
        if isinstance(provenance_payload, dict):
            method = provenance_payload.get("extraction_method")
            if isinstance(method, str) and method:
                return method
    return "parsed"


def _serialize_jsonable(value: Any) -> Any:
    """Best-effort conversion into JSON-friendly structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _serialize_jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_jsonable(asdict(value))
    return str(value)


def _serialize_constraint_result(constraint: Any) -> dict[str, Any]:
    """Serialize one pipeline constraint result without leaking opaque objects."""
    if isinstance(constraint, dict):
        return dict(constraint)
    return {
        "constraint_type": getattr(constraint, "constraint_type", None),
        "description": getattr(constraint, "description", ""),
        "metadata": _serialize_jsonable(getattr(constraint, "metadata", {})),
    }


def _serialize_verification_result(verification: Any) -> dict[str, Any]:
    """Serialize verification output for later trace-learning and audit work."""
    typed_reasoning = getattr(verification, "typed_reasoning", None)
    semantic_grounding = getattr(verification, "semantic_grounding", None)
    constraints = [
        _serialize_constraint_result(constraint)
        for constraint in list(getattr(verification, "constraints", []))
    ]
    violations = [
        _serialize_constraint_result(violation)
        for violation in list(getattr(verification, "violations", []))
    ]
    return {
        "verified": bool(getattr(verification, "verified", False)),
        "energy": round(float(getattr(verification, "energy", 0.0)), 6),
        "certificate": _serialize_jsonable(getattr(verification, "certificate", {})),
        "constraints": constraints,
        "violations": violations,
        "n_constraints": len(constraints),
        "n_violations": len(violations),
        "typed_reasoning_parse_status": _typed_reasoning_parse_status(typed_reasoning),
        "typed_reasoning": _serialize_jsonable(typed_reasoning),
        "semantic_grounding": _serialize_jsonable(semantic_grounding),
    }


def _serialize_generation_attempt(
    *,
    prompt: str,
    response: str,
    tokenizer: Any,
    valid: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Serialize one generation attempt with prompt/response token counts."""
    prompt_tokens = _token_count(tokenizer, prompt)
    response_tokens = _token_count(tokenizer, response)
    payload = {
        "prompt": prompt,
        "response": response,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
    }
    if valid is not None:
        payload["valid"] = valid
    if error is not None:
        payload["error"] = error
    return payload


def _build_generation_trace(
    *,
    tokenizer: Any,
    attempts: list[dict[str, Any]],
    fallback_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect prompt/response attempts and aggregate token usage."""
    serialized_attempts = list(attempts)
    if fallback_record:
        serialized_attempts.append(
            _serialize_generation_attempt(
                prompt=str(fallback_record.get("prompt", "")),
                response=str(fallback_record.get("response", "")),
                tokenizer=tokenizer,
            )
        )
    prompt_tokens = sum(int(attempt.get("prompt_tokens", 0)) for attempt in serialized_attempts)
    response_tokens = sum(int(attempt.get("response_tokens", 0)) for attempt in serialized_attempts)
    return {
        "attempts": serialized_attempts,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "fallback_used": bool(fallback_record),
    }


def load_monitorability_policy(path: Path | None = None) -> dict[str, Any]:
    """Load the Exp 213 monitorability policy when it is present."""
    policy_path = path or (get_repo_root() / "results" / "monitorability_policy_213.json")
    if not policy_path.exists():
        return {}
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def recommended_response_mode(
    task_slice: str,
    policy: dict[str, Any] | None = None,
) -> str:
    """Return the policy-recommended response mode for a task slice."""
    payload = policy if policy is not None else load_monitorability_policy()
    per_task_slice = payload.get("per_task_slice", {})
    if isinstance(per_task_slice, dict):
        entry = per_task_slice.get(task_slice, {})
        if isinstance(entry, dict):
            mode = entry.get("recommended_mode")
            if isinstance(mode, str):
                return mode
    return "answer_only_terse"


def sample_records(
    records: list[dict[str, Any]],
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic seeded shuffle followed by first-N selection."""
    shuffled = list(records)
    random.Random(sample_seed).shuffle(shuffled)
    return shuffled[:sample_size]


def shared_prompt_seed(sample_seed: int, case_id: str) -> int:
    """Derive one stable prompt seed for a case from the sample seed."""
    digest = hashlib.sha256(f"{sample_seed}:{case_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def build_cohort_manifest(
    records: list[dict[str, Any]],
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Build the sampled cohort manifest with shared prompt seeds."""
    sampled = sample_records(records, sample_size=sample_size, sample_seed=sample_seed)
    cohort: list[dict[str, Any]] = []
    for sample_position, record in enumerate(sampled, start=1):
        enriched = dict(record)
        case_id = str(enriched["case_id"])
        prompt_seed = shared_prompt_seed(sample_seed, case_id)
        enriched["sample_position"] = sample_position
        enriched["prompt_seeds"] = {
            "baseline": prompt_seed,
            "verify_only": prompt_seed,
            "verify_repair": prompt_seed,
        }
        cohort.append(enriched)
    return cohort


def checkpoint_path(
    checkpoint_dir: Path,
    *,
    benchmark: str,
    model_name: str,
    mode: str,
) -> Path:
    """Return the per-benchmark/model/mode checkpoint path."""
    return checkpoint_dir / (
        f"{safe_slug(benchmark)}__{safe_slug(model_name)}__{safe_slug(mode)}.json"
    )


def load_checkpoint(path: Path, expected_case_ids: list[str]) -> dict[str, Any]:
    """Load a checkpoint when the cohort metadata still matches."""
    fresh = {
        "case_ids": list(expected_case_ids),
        "results_by_case": {},
    }
    if not path.exists():
        return fresh

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_ids") != expected_case_ids:
        return fresh
    results_by_case = payload.get("results_by_case", {})
    if not isinstance(results_by_case, dict):
        return fresh
    return {
        **payload,
        "case_ids": list(expected_case_ids),
        "results_by_case": dict(results_by_case),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_mode(
    *,
    benchmark: str,
    model_name: str,
    mode: str,
    cases: list[dict[str, Any]],
    checkpoint_dir: Path,
    execute_case: Any,
) -> list[dict[str, Any]]:
    """Resume-aware execution helper for one benchmark/model/mode cell."""
    case_ids = [str(case["case_id"]) for case in cases]
    ckpt_path = checkpoint_path(
        checkpoint_dir,
        benchmark=benchmark,
        model_name=model_name,
        mode=mode,
    )
    checkpoint = load_checkpoint(ckpt_path, case_ids)
    results_by_case: dict[str, Any] = dict(checkpoint["results_by_case"])

    for case in cases:
        case_id = str(case["case_id"])
        if case_id in results_by_case:
            continue
        result = dict(execute_case(case))
        result.setdefault("case_id", case_id)
        result.setdefault("mode", mode)
        results_by_case[case_id] = result
        save_checkpoint(
            ckpt_path,
            {
                "benchmark": benchmark,
                "model_name": model_name,
                "mode": mode,
                "case_ids": case_ids,
                "results_by_case": results_by_case,
            },
        )

    return [dict(results_by_case[case_id]) for case_id in case_ids]


def build_artifact_payload(
    *,
    benchmark: str,
    output_path: Path,
    cohort: list[dict[str, Any]],
    paired_runs: list[dict[str, Any]],
    statistics: dict[str, Any],
    sample_seed: int,
    sample_size: int,
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    checkpoint_dir: Path,
    max_repairs: int,
    policy_path: Path,
    inference_mode: str,
) -> dict[str, Any]:
    """Build the stable top-level Exp 218 artifact payload."""
    spec = BENCHMARK_SPECS[benchmark]
    return {
        "experiment": artifact_experiment_id(output_path),
        "benchmark": benchmark,
        "title": spec["title"],
        "run_date": RUN_DATE,
        "schema": {
            "artifact": SCHEMA_ARTIFACT,
            "benchmark_case_schema": spec["case_schema"],
        },
        "metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 3),
            "sample_seed": sample_seed,
            "sample_size": sample_size,
            "sample_strategy": "seeded_shuffle_first_n",
            "modes": list(MODE_ORDER),
            "models": [dict(model) for model in MODEL_SPECS],
            "source_artifacts": list(spec["source_artifacts"]),
            "output_path": str(output_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_pattern": (
                "results/checkpoints/experiment_218/<benchmark>__<model>__<mode>.json"
            ),
            "max_repairs": max_repairs,
            "policy_source": str(policy_path),
            "inference_mode": inference_mode,
            "force_live": os.environ.get("CARNOT_FORCE_LIVE") == "1",
            "force_cpu": os.environ.get("CARNOT_FORCE_CPU") == "1",
        },
        "cohort": {
            "case_count": len(cohort),
            "case_ids": [str(case["case_id"]) for case in cohort],
            "cases": [dict(case) for case in cohort],
        },
        "paired_runs": list(paired_runs),
        "statistics": dict(statistics),
    }


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write an artifact with parent directory creation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class HarnessArgumentParser(argparse.ArgumentParser):
    """Parser that fills benchmark-specific defaults after parsing."""

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.sample_size is None:
            parsed.sample_size = BENCHMARK_SPECS[parsed.benchmark]["default_sample_size"]
        if parsed.output is None:
            parsed.output = default_output_path(parsed.benchmark)
        if parsed.checkpoint_dir is None:
            parsed.checkpoint_dir = default_checkpoint_dir()
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the unified Exp 218 CLI parser."""
    parser = HarnessArgumentParser(
        description="Checkpointed shared live benchmark harness for Exp 219-221.",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_SPECS),
        required=True,
        help="Benchmark to run.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override the benchmark-specific default sample size.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Seed for deterministic cohort sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Artifact path. Defaults to results/experiment_218_<benchmark>_results.json.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-benchmark/model/mode checkpoints.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=DEFAULT_MAX_REPAIRS,
        help="Maximum verify-repair iterations per case.",
    )
    return parser


def _load_script_module(module_name: str, relative_path: str) -> Any:  # pragma: no cover
    module_path = get_repo_root() / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_runtime(seed: int) -> None:  # pragma: no cover
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_live_model(model_spec: dict[str, str]) -> tuple[Any, Any]:  # pragma: no cover
    os.environ.setdefault("CARNOT_FORCE_LIVE", "1")
    os.environ.setdefault("CARNOT_FORCE_CPU", "0")

    from carnot.inference.model_loader import load_model  # type: ignore[import-untyped]

    model, tokenizer = load_model(model_spec["hf_id"], device="cuda")
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load live model: {model_spec['hf_id']}")
    return model, tokenizer


def _unload_live_model(model: Any, tokenizer: Any) -> None:  # pragma: no cover
    del model, tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_text(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_seed: int,
    max_new_tokens: int,
) -> str:  # pragma: no cover
    _seed_runtime(prompt_seed)
    from carnot.inference.model_loader import generate

    return str(generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens))


def _extract_final_number(text: str) -> int | None:  # pragma: no cover
    numeric_token = r"-?(?:\d[\d,]*)"

    match = re.search(rf"####\s*({numeric_token})", text)
    if match:
        return int(match.group(1).replace(",", ""))

    match = re.search(rf"[Aa]nswer[:\s]+({numeric_token})", text)
    if match:
        return int(match.group(1).replace(",", ""))

    numbers = re.findall(numeric_token, text)
    if numbers:
        return int(numbers[-1].replace(",", ""))
    return None


def _load_benchmark_records(benchmark: str) -> list[dict[str, Any]]:  # pragma: no cover
    if benchmark == "gsm8k_semantic":
        from datasets import load_dataset  # type: ignore[import-untyped]

        dataset = None
        for dataset_name in ("openai/gsm8k", "gsm8k"):
            try:
                dataset = load_dataset(dataset_name, "main", split="test")
                break
            except Exception:
                continue
        if dataset is None:
            raise RuntimeError("Could not load GSM8K test split.")

        records: list[dict[str, Any]] = []
        for dataset_idx in range(len(dataset)):
            row = dataset[dataset_idx]
            ground_truth = _extract_final_number(str(row["answer"]))
            if ground_truth is None:
                continue
            records.append(
                {
                    "case_id": f"gsm8k-{dataset_idx}",
                    "dataset_idx": dataset_idx,
                    "question": row["question"],
                    "ground_truth": ground_truth,
                    "task_slice": "live_gsm8k_semantic_failure",
                }
            )
        return records

    if benchmark == "humaneval_property":
        from datasets import load_dataset

        dataset = load_dataset("openai_humaneval", split="test")
        return [
            {
                "case_id": f"humaneval-{idx}",
                "dataset_idx": idx,
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "task_slice": "code_typed_properties",
            }
            for idx, row in enumerate(dataset)
        ]

    if benchmark == "constraint_ir":
        path = get_repo_root() / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _build_constraint_ir_prompt(
    case: dict[str, Any],
    response_mode: str,
) -> tuple[str, int]:  # pragma: no cover
    module = _load_script_module(
        "experiment_213_monitorability_audit",
        "scripts/experiment_213_monitorability_audit.py",
    )
    return (
        str(module.build_mode_prompt(case, response_mode)),
        int(module.max_new_tokens_for(case, response_mode)),
    )


def _evaluate_constraint_ir_response(
    case: dict[str, Any],
    response_mode: str,
    response: str,
) -> dict[str, Any]:  # pragma: no cover
    module = _load_script_module(
        "experiment_213_monitorability_audit",
        "scripts/experiment_213_monitorability_audit.py",
    )
    return dict(module.evaluate_response(case, response_mode, response))


def _run_gsm8k_baseline(
    case: dict[str, Any],
    *,
    model_spec: dict[str, str],
    model: Any,
    tokenizer: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.verify_repair import VerifyRepairPipeline  # type: ignore[import-untyped]

    typed_reasoning = None
    response_mode = recommended_response_mode(str(case["task_slice"]), policy)
    prompt_seed = int(case["prompt_seeds"]["baseline"])
    task = f"Question: {case['question']}\nSolve the problem and give the final answer as a number."
    prompt = (
        task + "\nReturn only the final numeric answer. No explanation."
        if response_mode != "structured_json"
        else task
    )
    started = time.perf_counter()
    if response_mode == "structured_json":
        from carnot.pipeline.structured_reasoning import (  # type: ignore[import-untyped]
            StructuredReasoningController,
        )

        fallback_record: dict[str, Any] = {}

        def fallback_generate(generated_prompt: str, max_new_tokens: int) -> str:
            response = _generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=generated_prompt,
                prompt_seed=prompt_seed,
                max_new_tokens=max_new_tokens,
            )
            fallback_record.update(
                {
                    "prompt": generated_prompt,
                    "response": response,
                }
            )
            return response

        controller = StructuredReasoningController(policy=policy)
        emission = controller.emit(
            question=task,
            task_slice=str(case["task_slice"]),
            model_name=model_spec["hf_id"],
            model=model,
            tokenizer=tokenizer,
            fallback_generate=fallback_generate,
        )
        response = emission.response
        typed_reasoning = emission.typed_reasoning
        effective_mode = emission.response_mode
        generation_trace = _build_generation_trace(
            tokenizer=tokenizer,
            attempts=[
                _serialize_generation_attempt(
                    prompt=str(attempt.prompt),
                    response=str(attempt.raw_response),
                    tokenizer=tokenizer,
                    valid=bool(attempt.valid),
                    error=attempt.error,
                )
                for attempt in emission.attempts
            ],
            fallback_record=fallback_record if emission.fallback_used else None,
        )
    else:
        response = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_seed=prompt_seed,
            max_new_tokens=96,
        )
        typed_reasoning = VerifyRepairPipeline(model=None).extract_typed_reasoning(task, response)
        effective_mode = response_mode
        generation_trace = _build_generation_trace(
            tokenizer=tokenizer,
            attempts=[
                _serialize_generation_attempt(
                    prompt=prompt,
                    response=response,
                    tokenizer=tokenizer,
                )
            ],
        )

    extracted_answer = _extract_final_number(response)
    correct = extracted_answer == int(case["ground_truth"])
    return {
        "case_id": str(case["case_id"]),
        "mode": "baseline",
        "prompt_seed": prompt_seed,
        "response_mode": effective_mode,
        "response": response,
        "extracted_answer": extracted_answer,
        "ground_truth": int(case["ground_truth"]),
        "correct": correct,
        "typed_reasoning_available": typed_reasoning is not None,
        "typed_reasoning_parse_status": _typed_reasoning_parse_status(typed_reasoning),
        "typed_reasoning": _serialize_jsonable(typed_reasoning),
        "generation_trace": generation_trace,
        "prompt_tokens": int(generation_trace["prompt_tokens"]),
        "response_tokens": int(generation_trace["response_tokens"]),
        "total_tokens": int(generation_trace["total_tokens"]),
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _run_gsm8k_verify_only(
    case: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"], timeout_seconds=30.0)
    started = time.perf_counter()
    verification = pipeline.verify(
        str(case["question"]),
        str(baseline["response"]),
        domain="arithmetic",
    )
    verification_trace = _serialize_verification_result(verification)
    semantic_grounding = verification_trace.get("semantic_grounding", {})
    semantic_violations = []
    if isinstance(semantic_grounding, dict):
        violations = semantic_grounding.get("violations", [])
        if isinstance(violations, list):
            semantic_violations = violations
    semantic_violation_count = len(semantic_violations)
    flagged = (not verification.verified) or bool(verification.violations)
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_only",
        "prompt_seed": int(case["prompt_seeds"]["verify_only"]),
        "response_mode": baseline["response_mode"],
        "response": baseline["response"],
        "verified": verification.verified,
        "flagged": flagged,
        "n_constraints": len(verification.constraints),
        "n_violations": len(verification.violations),
        "semantic_violation_count": semantic_violation_count,
        "typed_reasoning_available": bool(verification.typed_reasoning),
        "typed_reasoning_parse_status": str(verification_trace["typed_reasoning_parse_status"]),
        "parseable": verification_trace["typed_reasoning_parse_status"] != "unavailable",
        "verification": verification_trace,
        "correct": bool(baseline["correct"]),
        "accepted_correct": bool(baseline["correct"]) and not flagged,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _run_gsm8k_verify_repair(
    case: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model_spec: dict[str, str],
    model: Any,
    tokenizer: Any,
    policy: dict[str, Any],
    max_repairs: int,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"], timeout_seconds=30.0)
    current_response = str(baseline["response"])
    started = time.perf_counter()
    verification = pipeline.verify(str(case["question"]), current_response, domain="arithmetic")
    initial_trace = _serialize_verification_result(verification)
    history = [
        {
            "iteration": 0,
            "response_mode": baseline["response_mode"],
            "response": current_response,
            "verification": initial_trace,
        }
    ]
    if verification.verified:
        return {
            "case_id": str(case["case_id"]),
            "mode": "verify_repair",
            "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
            "response_mode": baseline["response_mode"],
            "initial_response": baseline["response"],
            "final_response": current_response,
            "initial_correct": bool(baseline["correct"]),
            "correct": bool(baseline["correct"]),
            "verified": True,
            "repaired": False,
            "n_repairs": 0,
            "typed_reasoning_parse_status": str(initial_trace["typed_reasoning_parse_status"]),
            "initial_verification": initial_trace,
            "final_verification": initial_trace,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "history": history,
        }

    task_slice = str(case["task_slice"])
    task_prefix = (
        f"Question: {case['question']}\n\n"
        f"Your previous answer was:\n{current_response}\n\n"
        f"The following issues were found:\n"
        f"{VerifyRepairPipeline._format_violations(verification.violations)}\n\n"
        "Please provide a corrected answer."
    )
    n_repairs = 0
    total_prompt_tokens = 0
    total_response_tokens = 0
    for repair_idx in range(1, max_repairs + 1):
        response_mode = str(baseline["response_mode"])
        if response_mode == "structured_json":
            from carnot.pipeline.structured_reasoning import StructuredReasoningController

            controller = StructuredReasoningController(policy=policy)
            repair_seed = int(case["prompt_seeds"]["verify_repair"]) + repair_idx
            fallback_record: dict[str, Any] = {}

            def fallback_generate(
                generated_prompt: str,
                max_new_tokens: int,
                *,
                repair_seed: int = repair_seed,
                fallback_record: dict[str, Any] = fallback_record,
            ) -> str:
                response = _generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=generated_prompt,
                    prompt_seed=repair_seed,
                    max_new_tokens=max_new_tokens,
                )
                fallback_record.update(
                    {
                        "prompt": generated_prompt,
                        "response": response,
                    }
                )
                return response

            emission = controller.emit(
                question=task_prefix,
                task_slice=task_slice,
                model_name=model_spec["hf_id"],
                model=model,
                tokenizer=tokenizer,
                fallback_generate=fallback_generate,
            )
            current_response = emission.response
            generation_trace = _build_generation_trace(
                tokenizer=tokenizer,
                attempts=[
                    _serialize_generation_attempt(
                        prompt=str(attempt.prompt),
                        response=str(attempt.raw_response),
                        tokenizer=tokenizer,
                        valid=bool(attempt.valid),
                        error=attempt.error,
                    )
                    for attempt in emission.attempts
                ],
                fallback_record=fallback_record if emission.fallback_used else None,
            )
            current_response_mode = emission.response_mode
        else:
            repair_prompt = task_prefix + "\nReturn only the final numeric answer."
            current_response = _generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=repair_prompt,
                prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx,
                max_new_tokens=96,
            )
            generation_trace = _build_generation_trace(
                tokenizer=tokenizer,
                attempts=[
                    _serialize_generation_attempt(
                        prompt=repair_prompt,
                        response=current_response,
                        tokenizer=tokenizer,
                    )
                ],
            )
            current_response_mode = response_mode
        total_prompt_tokens += int(generation_trace["prompt_tokens"])
        total_response_tokens += int(generation_trace["response_tokens"])
        verification = pipeline.verify(str(case["question"]), current_response, domain="arithmetic")
        verification_trace = _serialize_verification_result(verification)
        history.append(
            {
                "iteration": repair_idx,
                "response_mode": current_response_mode,
                "response": current_response,
                "generation_trace": generation_trace,
                "verification": verification_trace,
            }
        )
        n_repairs = repair_idx
        if verification.verified:
            break

    final_answer = _extract_final_number(current_response)
    final_correct = final_answer == int(case["ground_truth"])
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_repair",
        "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
        "response_mode": baseline["response_mode"],
        "initial_response": baseline["response"],
        "final_response": current_response,
        "initial_correct": bool(baseline["correct"]),
        "correct": final_correct,
        "verified": verification.verified,
        "repaired": (not bool(baseline["correct"])) and final_correct,
        "n_repairs": n_repairs,
        "typed_reasoning_parse_status": str(
            history[-1]["verification"]["typed_reasoning_parse_status"]
        ),
        "initial_verification": initial_trace,
        "final_verification": history[-1]["verification"],
        "prompt_tokens": total_prompt_tokens,
        "response_tokens": total_response_tokens,
        "total_tokens": total_prompt_tokens + total_response_tokens,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }


def _run_humaneval_baseline(
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.humaneval_live_benchmark import (  # type: ignore[import-untyped]
        build_candidate_code,
        execute_humaneval,
        run_instrumentation,
    )

    prompt_seed = int(case["prompt_seeds"]["baseline"])
    generation_prompt = (
        "You are an expert Python programmer.\n"
        "Complete the following function.\n"
        "Return ONLY the function body lines. No def line. No markdown fences.\n"
        "Indent with 4 spaces.\n\n"
        f"{case['prompt']}"
    )
    started = time.perf_counter()
    body = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=generation_prompt,
        prompt_seed=prompt_seed,
        max_new_tokens=220,
    )
    candidate_code = build_candidate_code(str(case["prompt"]), body)
    harness = execute_humaneval(candidate_code, case, timeout=5.0)
    instrumentation = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=str(case["test"]),
    )
    return {
        "case_id": str(case["case_id"]),
        "mode": "baseline",
        "prompt_seed": prompt_seed,
        "response_mode": "answer_only_terse",
        "body": body,
        "candidate_code": candidate_code,
        "passed": harness.passed,
        "error_type": harness.error_type,
        "error_message": harness.error_message,
        "instrumentation": instrumentation,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _run_humaneval_verify_only(
    case: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.humaneval_live_benchmark import (
        execute_humaneval,
        run_instrumentation,
    )

    candidate_code = str(baseline["candidate_code"])
    execution_only = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=None,
    )
    execution_plus_property = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=str(case["test"]),
    )
    harness = execute_humaneval(candidate_code, case, timeout=5.0)
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_only",
        "prompt_seed": int(case["prompt_seeds"]["verify_only"]),
        "response_mode": "answer_only_terse",
        "passed": harness.passed,
        "execution_only": execution_only,
        "execution_plus_property": execution_plus_property,
    }


def _run_humaneval_verify_repair(
    case: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    max_repairs: int,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.humaneval_live_benchmark import (
        HarnessResult,
        build_candidate_code,
        build_repair_prompt,
        execute_humaneval,
        run_instrumentation,
    )

    if bool(baseline["passed"]):
        return {
            "case_id": str(case["case_id"]),
            "mode": "verify_repair",
            "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
            "response_mode": "answer_only_terse",
            "initial_passed": True,
            "passed": True,
            "repaired": False,
            "n_repairs": 0,
            "final_body": baseline["body"],
        }

    current_body = str(baseline["body"])
    current_code = str(baseline["candidate_code"])
    harness = HarnessResult(
        passed=False,
        error_type=str(baseline["error_type"]),
        error_message=str(baseline["error_message"]),
        stdout="",
    )
    instrumentation = dict(baseline["instrumentation"])
    n_repairs = 0

    for repair_idx in range(max_repairs):
        repair_prompt = build_repair_prompt(
            str(case["prompt"]),
            current_body,
            harness,
            instrumentation,
            repair_idx=repair_idx,
        )
        current_body = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx + 1,
            max_new_tokens=220,
        )
        current_code = build_candidate_code(str(case["prompt"]), current_body)
        harness = execute_humaneval(current_code, case, timeout=5.0)
        instrumentation = run_instrumentation(
            current_code,
            str(case["prompt"]),
            str(case["entry_point"]),
            official_tests=str(case["test"]),
        )
        n_repairs = repair_idx + 1
        if harness.passed:
            break

    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_repair",
        "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
        "response_mode": "answer_only_terse",
        "initial_passed": bool(baseline["passed"]),
        "passed": harness.passed,
        "repaired": (not bool(baseline["passed"])) and harness.passed,
        "n_repairs": n_repairs,
        "final_body": current_body,
        "final_code": current_code,
        "error_type": harness.error_type,
        "error_message": harness.error_message,
    }


def _run_constraint_ir_baseline(
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    response_mode = recommended_response_mode(str(case["task_slice"]), policy)
    prompt, max_new_tokens = _build_constraint_ir_prompt(case, response_mode)
    started = time.perf_counter()
    response = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_seed=int(case["prompt_seeds"]["baseline"]),
        max_new_tokens=max_new_tokens,
    )
    evaluation = _evaluate_constraint_ir_response(case, response_mode, response)
    return {
        "case_id": str(case["case_id"]),
        "mode": "baseline",
        "prompt_seed": int(case["prompt_seeds"]["baseline"]),
        "response_mode": response_mode,
        "response": response,
        "evaluation": evaluation,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _constraint_ir_verified(evaluation: dict[str, Any]) -> bool:  # pragma: no cover
    parseable = bool(evaluation.get("parseable"))
    answer_quality = float(evaluation.get("answer_quality", 0.0))
    coverage = float(evaluation.get("constraint_coverage", 0.0))
    return parseable and answer_quality >= 1.0 and coverage >= 1.0


def _run_constraint_ir_verify_only(
    case: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    evaluation = dict(baseline["evaluation"])
    verified = _constraint_ir_verified(evaluation)
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_only",
        "prompt_seed": int(case["prompt_seeds"]["verify_only"]),
        "response_mode": str(baseline["response_mode"]),
        "response": baseline["response"],
        "verified": verified,
        "flagged": not verified,
        "evaluation": evaluation,
    }


def _run_constraint_ir_verify_repair(
    case: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    max_repairs: int,
) -> dict[str, Any]:  # pragma: no cover
    evaluation = dict(baseline["evaluation"])
    if _constraint_ir_verified(evaluation):
        return {
            "case_id": str(case["case_id"]),
            "mode": "verify_repair",
            "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
            "response_mode": str(baseline["response_mode"]),
            "initial_verified": True,
            "verified": True,
            "repaired": False,
            "n_repairs": 0,
            "final_response": baseline["response"],
            "evaluation": evaluation,
        }

    current_response = str(baseline["response"])
    response_mode = str(baseline["response_mode"])
    prompt, max_new_tokens = _build_constraint_ir_prompt(case, response_mode)
    issue_lines = [
        f"- parseable={evaluation.get('parseable')}",
        f"- answer_quality={evaluation.get('answer_quality')}",
        f"- constraint_coverage={evaluation.get('constraint_coverage')}",
    ]
    n_repairs = 0
    for repair_idx in range(max_repairs):
        repair_prompt = (
            f"{prompt}\n\n"
            "Your previous response did not satisfy the required contract.\n"
            f"Previous response:\n{current_response}\n\n"
            "Issues:\n"
            + "\n".join(issue_lines)
            + "\n\nAnswer again using the same response contract.\n"
        )
        current_response = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx + 1,
            max_new_tokens=max_new_tokens,
        )
        evaluation = _evaluate_constraint_ir_response(case, response_mode, current_response)
        issue_lines = [
            f"- parseable={evaluation.get('parseable')}",
            f"- answer_quality={evaluation.get('answer_quality')}",
            f"- constraint_coverage={evaluation.get('constraint_coverage')}",
        ]
        n_repairs = repair_idx + 1
        if _constraint_ir_verified(evaluation):
            break

    verified = _constraint_ir_verified(evaluation)
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_repair",
        "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
        "response_mode": response_mode,
        "initial_verified": _constraint_ir_verified(dict(baseline["evaluation"])),
        "verified": verified,
        "repaired": (not _constraint_ir_verified(dict(baseline["evaluation"]))) and verified,
        "n_repairs": n_repairs,
        "final_response": current_response,
        "evaluation": evaluation,
    }


def _summarize_runs(
    benchmark: str,
    baseline_runs: list[dict[str, Any]],
    verify_only_runs: list[dict[str, Any]],
    verify_repair_runs: list[dict[str, Any]],
) -> dict[str, Any]:  # pragma: no cover
    n_cases = len(baseline_runs)
    if benchmark == "gsm8k_semantic":
        baseline_accuracy = sum(1 for run in baseline_runs if run["correct"]) / n_cases
        verify_accuracy = sum(1 for run in verify_only_runs if run["accepted_correct"]) / n_cases
        repair_accuracy = sum(1 for run in verify_repair_runs if run["correct"]) / n_cases
        n_wrong_answers = sum(1 for run in baseline_runs if not run["correct"])
        n_wrong_detected = sum(
            1 for run in verify_only_runs if run["flagged"] and not run["correct"]
        )
        false_positives = sum(1 for run in verify_only_runs if run["flagged"] and run["correct"])
        n_repaired = sum(1 for run in verify_repair_runs if run["repaired"])
        semantic_violation_count = sum(
            int(run.get("semantic_violation_count", 0)) for run in verify_only_runs
        )
        parse_coverage = (
            sum(
                1
                for run in verify_only_runs
                if str(run.get("typed_reasoning_parse_status", "unavailable")) != "unavailable"
            )
            / n_cases
        )
        return {
            "baseline": {
                "n_cases": n_cases,
                "accuracy": baseline_accuracy,
                "n_correct": sum(1 for run in baseline_runs if run["correct"]),
                "mean_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in baseline_runs]
                ),
                "mean_prompt_tokens": _round_mean(
                    [float(run.get("prompt_tokens", 0.0)) for run in baseline_runs]
                ),
                "mean_response_tokens": _round_mean(
                    [float(run.get("response_tokens", 0.0)) for run in baseline_runs]
                ),
                "mean_total_tokens": _round_mean(
                    [float(run.get("total_tokens", 0.0)) for run in baseline_runs]
                ),
            },
            "verify_only": {
                "n_cases": n_cases,
                "accuracy": verify_accuracy,
                "n_flagged": sum(1 for run in verify_only_runs if run["flagged"]),
                "n_wrong_answers": n_wrong_answers,
                "n_wrong_detected": n_wrong_detected,
                "wrong_detection_rate": round(
                    n_wrong_detected / n_wrong_answers if n_wrong_answers else 0.0,
                    6,
                ),
                "semantic_violation_count": semantic_violation_count,
                "parse_coverage": round(parse_coverage, 6),
                "false_positives": false_positives,
                "false_positive_rate": round(
                    false_positives / max(1, sum(1 for run in baseline_runs if run["correct"])),
                    6,
                ),
                "mean_additional_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in verify_only_runs]
                ),
                "mean_additional_tokens": _round_mean(
                    [float(run.get("total_tokens", 0.0)) for run in verify_only_runs]
                ),
            },
            "verify_repair": {
                "n_cases": n_cases,
                "accuracy": repair_accuracy,
                "n_repaired": n_repaired,
                "repair_yield": round(n_repaired / n_wrong_answers if n_wrong_answers else 0.0, 6),
                "avg_repairs": round(
                    sum(int(run["n_repairs"]) for run in verify_repair_runs) / n_cases,
                    3,
                ),
                "mean_additional_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in verify_repair_runs]
                ),
                "mean_additional_tokens": _round_mean(
                    [float(run.get("total_tokens", 0.0)) for run in verify_repair_runs]
                ),
            },
            "paired_deltas": {
                "verify_only_minus_baseline": verify_accuracy - baseline_accuracy,
                "repair_minus_baseline": repair_accuracy - baseline_accuracy,
            },
        }

    if benchmark == "humaneval_property":
        baseline_pass = sum(1 for run in baseline_runs if run["passed"]) / n_cases
        repair_pass = sum(1 for run in verify_repair_runs if run["passed"]) / n_cases
        return {
            "baseline": {
                "n_cases": n_cases,
                "pass_at_1": baseline_pass,
            },
            "verify_only": {
                "n_cases": n_cases,
                "execution_only_detected": sum(
                    1 for run in verify_only_runs if run["execution_only"]["detected"]
                ),
                "execution_plus_property_detected": sum(
                    1 for run in verify_only_runs if run["execution_plus_property"]["detected"]
                ),
                "property_violation_total": sum(
                    int(run["execution_plus_property"]["n_property_violations"])
                    for run in verify_only_runs
                ),
            },
            "verify_repair": {
                "n_cases": n_cases,
                "pass_at_1": repair_pass,
                "n_repaired": sum(1 for run in verify_repair_runs if run["repaired"]),
                "avg_repairs": (sum(int(run["n_repairs"]) for run in verify_repair_runs) / n_cases),
            },
            "paired_deltas": {
                "repair_minus_baseline": repair_pass - baseline_pass,
            },
        }

    if benchmark == "constraint_ir":
        baseline_quality = (
            sum(float(run["evaluation"]["answer_quality"]) for run in baseline_runs) / n_cases
        )
        verify_rate = sum(1 for run in verify_only_runs if run["verified"]) / n_cases
        repair_rate = sum(1 for run in verify_repair_runs if run["verified"]) / n_cases
        return {
            "baseline": {
                "n_cases": n_cases,
                "mean_answer_quality": baseline_quality,
                "parseable_rate": (
                    sum(1 for run in baseline_runs if run["evaluation"]["parseable"]) / n_cases
                ),
            },
            "verify_only": {
                "n_cases": n_cases,
                "verified_rate": verify_rate,
                "mean_constraint_coverage": (
                    sum(float(run["evaluation"]["constraint_coverage"]) for run in verify_only_runs)
                    / n_cases
                ),
            },
            "verify_repair": {
                "n_cases": n_cases,
                "verified_rate": repair_rate,
                "n_repaired": sum(1 for run in verify_repair_runs if run["repaired"]),
            },
            "paired_deltas": {
                "repair_minus_verify_only": repair_rate - verify_rate,
            },
        }

    raise ValueError(f"Unsupported benchmark for summary: {benchmark}")


def _run_live_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover
    benchmark = str(args.benchmark)
    started_at = utc_now()
    started = time.perf_counter()
    policy_path = get_repo_root() / "results" / "monitorability_policy_213.json"
    policy = load_monitorability_policy(policy_path)
    records = _load_benchmark_records(benchmark)
    cohort = build_cohort_manifest(
        records,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    paired_runs: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {}

    for model_spec in MODEL_SPECS:
        print(f"Running {benchmark} on {model_spec['name']} ({model_spec['hf_id']})")
        model, tokenizer = _load_live_model(model_spec)
        try:

            def execute_baseline_case(
                case: dict[str, Any],
                *,
                benchmark: str = benchmark,
                model_spec: dict[str, str] = model_spec,
                model: Any = model,
                tokenizer: Any = tokenizer,
                policy: dict[str, Any] = policy,
            ) -> dict[str, Any]:
                if benchmark == "gsm8k_semantic":
                    return _run_gsm8k_baseline(
                        case,
                        model_spec=model_spec,
                        model=model,
                        tokenizer=tokenizer,
                        policy=policy,
                    )
                if benchmark == "humaneval_property":
                    return _run_humaneval_baseline(case, model=model, tokenizer=tokenizer)
                return _run_constraint_ir_baseline(
                    case,
                    model=model,
                    tokenizer=tokenizer,
                    policy=policy,
                )

            baseline_runs = run_mode(
                benchmark=benchmark,
                model_name=model_spec["name"],
                mode="baseline",
                cases=cohort,
                checkpoint_dir=checkpoint_dir,
                execute_case=execute_baseline_case,
            )
            baseline_by_case = {run["case_id"]: run for run in baseline_runs}

            def execute_verify_only_case(
                case: dict[str, Any],
                *,
                benchmark: str = benchmark,
                baseline_by_case: dict[str, dict[str, Any]] = baseline_by_case,
            ) -> dict[str, Any]:
                baseline = baseline_by_case[str(case["case_id"])]
                if benchmark == "gsm8k_semantic":
                    return _run_gsm8k_verify_only(case, baseline)
                if benchmark == "humaneval_property":
                    return _run_humaneval_verify_only(case, baseline)
                return _run_constraint_ir_verify_only(case, baseline)

            verify_only_runs = run_mode(
                benchmark=benchmark,
                model_name=model_spec["name"],
                mode="verify_only",
                cases=cohort,
                checkpoint_dir=checkpoint_dir,
                execute_case=execute_verify_only_case,
            )

            def execute_verify_repair_case(
                case: dict[str, Any],
                *,
                benchmark: str = benchmark,
                baseline_by_case: dict[str, dict[str, Any]] = baseline_by_case,
                model_spec: dict[str, str] = model_spec,
                model: Any = model,
                tokenizer: Any = tokenizer,
                policy: dict[str, Any] = policy,
                max_repairs: int = args.max_repairs,
            ) -> dict[str, Any]:
                baseline = baseline_by_case[str(case["case_id"])]
                if benchmark == "gsm8k_semantic":
                    return _run_gsm8k_verify_repair(
                        case,
                        baseline,
                        model_spec=model_spec,
                        model=model,
                        tokenizer=tokenizer,
                        policy=policy,
                        max_repairs=max_repairs,
                    )
                if benchmark == "humaneval_property":
                    return _run_humaneval_verify_repair(
                        case,
                        baseline,
                        model=model,
                        tokenizer=tokenizer,
                        max_repairs=max_repairs,
                    )
                return _run_constraint_ir_verify_repair(
                    case,
                    baseline,
                    model=model,
                    tokenizer=tokenizer,
                    max_repairs=max_repairs,
                )

            verify_repair_runs = run_mode(
                benchmark=benchmark,
                model_name=model_spec["name"],
                mode="verify_repair",
                cases=cohort,
                checkpoint_dir=checkpoint_dir,
                execute_case=execute_verify_repair_case,
            )
        finally:
            _unload_live_model(model, tokenizer)

        model_summary = _summarize_runs(
            benchmark,
            baseline_runs,
            verify_only_runs,
            verify_repair_runs,
        )
        statistics[model_spec["name"]] = model_summary
        paired_runs.extend(
            [
                {
                    "benchmark": benchmark,
                    "mode": "baseline",
                    "model_name": model_spec["name"],
                    "model_hf_id": model_spec["hf_id"],
                    "summary": model_summary["baseline"],
                    "cases": baseline_runs,
                },
                {
                    "benchmark": benchmark,
                    "mode": "verify_only",
                    "model_name": model_spec["name"],
                    "model_hf_id": model_spec["hf_id"],
                    "summary": model_summary["verify_only"],
                    "cases": verify_only_runs,
                },
                {
                    "benchmark": benchmark,
                    "mode": "verify_repair",
                    "model_name": model_spec["name"],
                    "model_hf_id": model_spec["hf_id"],
                    "summary": model_summary["verify_repair"],
                    "cases": verify_repair_runs,
                },
            ]
        )

    finished_at = utc_now()
    return build_artifact_payload(
        benchmark=benchmark,
        output_path=Path(args.output),
        cohort=cohort,
        paired_runs=paired_runs,
        statistics=statistics,
        sample_seed=args.sample_seed,
        sample_size=args.sample_size,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=time.perf_counter() - started,
        checkpoint_dir=Path(args.checkpoint_dir),
        max_repairs=int(args.max_repairs),
        policy_path=policy_path,
        inference_mode=live_inference_mode(),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = _run_live_benchmark(args)
    write_artifact(Path(args.output), payload)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
