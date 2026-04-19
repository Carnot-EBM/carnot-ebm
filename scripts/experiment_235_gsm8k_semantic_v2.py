#!/usr/bin/env python3
"""Experiment 235: live GSM8K semantic benchmark with semantic verifier v2.

Spec: REQ-VERIFY-048, REQ-VERIFY-049, SCENARIO-VERIFY-050,
SCENARIO-VERIFY-051
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

RUN_DATE = "20260413"
EXPERIMENT = 235
BENCHMARK = "gsm8k_semantic"
TITLE = "Live GSM8K semantic benchmark v2"
SCHEMA_ARTIFACT = "carnot.live_dual_model_suite.v1"
DEFAULT_MAX_REPAIRS = 3
MODEL_SPECS: list[dict[str, str]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]


def get_repo_root() -> Path:
    """Return the repository root, honoring `CARNOT_REPO_ROOT` when set."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def default_output_path() -> Path:
    """Return the default Exp 235 artifact path."""
    return get_repo_root() / "results" / "experiment_235_results.json"


def default_checkpoint_dir() -> Path:
    """Return the default Exp 235 checkpoint directory."""
    return get_repo_root() / "results" / "checkpoints" / "experiment_235"


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _display_path(path: Path) -> str:
    """Prefer repo-relative paths in artifacts when possible."""
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path)


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write an artifact with parent directory creation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_harness_module() -> Any:  # pragma: no cover
    """Load the shared Exp 218 live harness module."""
    module_path = get_repo_root() / "scripts" / "experiment_218_live_dual_model_suite.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_218_live_dual_model_suite",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared live harness from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exp219_reference(path: Path | None = None) -> dict[str, Any]:
    """Load the checked-in Exp 219 artifact used for the paired comparison."""
    artifact_path = path or (get_repo_root() / "results" / "experiment_219_results.json")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if str(payload.get("benchmark")) != BENCHMARK:
        raise ValueError(f"Expected {artifact_path} to contain {BENCHMARK}")
    return payload if isinstance(payload, dict) else {}


def load_shared_cohort(path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse the checked-in Exp 219 cohort verbatim for a direct paired rerun."""
    artifact_path = path or (get_repo_root() / "results" / "experiment_219_results.json")
    reference = load_exp219_reference(artifact_path)
    cohort_payload = reference.get("cohort", {})
    if not isinstance(cohort_payload, dict):
        raise ValueError("Exp 219 artifact is missing a cohort block")
    cases = cohort_payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Exp 219 artifact cohort.cases must be a list")
    metadata = reference.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    cohort = [dict(case) for case in cases]
    return cohort, {
        "source_artifact": _display_path(artifact_path),
        "sample_seed": int(metadata.get("sample_seed", 218)),
        "sample_size": int(metadata.get("sample_size", len(cohort))),
        "case_count": int(cohort_payload.get("case_count", len(cohort))),
        "same_as_exp219": True,
    }


def _round_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _round_delta(value: float) -> float:
    return round(value, 6)


def _verdict_counts(verdicts: list[str]) -> dict[str, int]:
    counts = {
        "abstain": 0,
        "supported": 0,
        "unavailable": 0,
        "violated": 0,
    }
    for verdict in verdicts:
        normalized = verdict if verdict in counts else "unavailable"
        counts[normalized] += 1
    return counts


def summarize_gsm8k_v2_runs(
    *,
    baseline_runs: list[dict[str, Any]],
    verify_only_runs: list[dict[str, Any]],
    verify_repair_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the Exp 235 GSM8K runs with calibrated semantic-v2 fields."""
    n_cases = len(baseline_runs)
    baseline_accuracy = sum(1 for run in baseline_runs if run["correct"]) / n_cases
    verify_accuracy = sum(1 for run in verify_only_runs if run["accepted_correct"]) / n_cases
    repair_accuracy = sum(1 for run in verify_repair_runs if run["correct"]) / n_cases
    n_wrong_answers = sum(1 for run in baseline_runs if not run["correct"])
    n_wrong_detected = sum(1 for run in verify_only_runs if run["flagged"] and not run["correct"])
    false_positives = sum(1 for run in verify_only_runs if run["flagged"] and run["correct"])
    verify_only_verdicts = [
        str(run.get("semantic_verifier_v2_verdict", "unavailable")) for run in verify_only_runs
    ]
    repair_initial_verdicts = [
        str(run.get("initial_semantic_verifier_v2_verdict", "unavailable"))
        for run in verify_repair_runs
    ]
    repair_final_verdicts = [
        str(run.get("final_semantic_verifier_v2_verdict", "unavailable"))
        for run in verify_repair_runs
    ]
    n_repaired = sum(1 for run in verify_repair_runs if run["repaired"])
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
            "false_positives": false_positives,
            "false_positive_rate": round(
                false_positives / max(1, sum(1 for run in baseline_runs if run["correct"])),
                6,
            ),
            "semantic_verifier_v2_violated_cases": sum(
                1
                for run in verify_only_runs
                if run.get("semantic_verifier_v2_verdict") == "violated"
            ),
            "semantic_verifier_v2_detected_wrong_answers": sum(
                1
                for run in verify_only_runs
                if run.get("semantic_verifier_v2_detected_wrong_answer")
            ),
            "semantic_verifier_v2_false_positives": sum(
                1 for run in verify_only_runs if run.get("semantic_verifier_v2_false_positive")
            ),
            "parse_coverage": round(
                sum(
                    1
                    for run in verify_only_runs
                    if str(run.get("typed_reasoning_parse_status", "unavailable")) != "unavailable"
                )
                / n_cases,
                6,
            ),
            "confidence_summary": {
                "mean_error_probability": _round_mean(
                    [
                        float(run.get("semantic_verifier_v2_error_probability", 0.0))
                        for run in verify_only_runs
                    ]
                ),
                "mean_monitorability_confidence": _round_mean(
                    [
                        float(run.get("semantic_verifier_v2_monitorability_confidence", 0.0))
                        for run in verify_only_runs
                    ]
                ),
                "verdict_counts": _verdict_counts(verify_only_verdicts),
            },
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
                sum(int(run.get("n_repairs", 0)) for run in verify_repair_runs) / n_cases,
                3,
            ),
            "unnecessary_repairs": sum(
                1 for run in verify_repair_runs if run.get("unnecessary_repair", False)
            ),
            "confidence_summary": {
                "initial_mean_error_probability": _round_mean(
                    [
                        float(run.get("initial_semantic_verifier_v2_error_probability", 0.0))
                        for run in verify_repair_runs
                    ]
                ),
                "final_mean_error_probability": _round_mean(
                    [
                        float(run.get("final_semantic_verifier_v2_error_probability", 0.0))
                        for run in verify_repair_runs
                    ]
                ),
                "initial_mean_monitorability_confidence": _round_mean(
                    [
                        float(
                            run.get("initial_semantic_verifier_v2_monitorability_confidence", 0.0)
                        )
                        for run in verify_repair_runs
                    ]
                ),
                "final_mean_monitorability_confidence": _round_mean(
                    [
                        float(run.get("final_semantic_verifier_v2_monitorability_confidence", 0.0))
                        for run in verify_repair_runs
                    ]
                ),
                "initial_verdict_counts": _verdict_counts(repair_initial_verdicts),
                "final_verdict_counts": _verdict_counts(repair_final_verdicts),
            },
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


def build_exp219_comparison(
    *,
    current_statistics: dict[str, Any],
    exp219_statistics: dict[str, Any],
    same_cohort: bool,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the direct per-model comparison block against Exp 219."""
    comparisons: dict[str, Any] = {}
    justified: list[str] = []
    not_justified: list[str] = []
    for model_name, current in current_statistics.items():
        reference = exp219_statistics.get(model_name)
        if not isinstance(reference, dict):
            comparisons[model_name] = {"status": "missing_exp219_reference"}
            not_justified.append(model_name)
            continue
        current_verify = dict(current.get("verify_only", {}))
        reference_verify = dict(reference.get("verify_only", {}))
        current_repair = dict(current.get("verify_repair", {}))
        reference_repair = dict(reference.get("verify_repair", {}))
        current_conf = dict(current_verify.get("confidence_summary", {}))
        reference_conf = dict(reference_verify.get("confidence_summary", {}))
        verify_only_justified = float(
            current.get("paired_deltas", {}).get("verify_only_minus_baseline", 0.0)
        ) >= 0.0 and int(current_verify.get("false_positives", 0)) < int(
            reference_verify.get("false_positives", 0)
        )
        if verify_only_justified:
            justified.append(model_name)
        else:
            not_justified.append(model_name)
        comparisons[model_name] = {
            "verify_only_accuracy_delta": _round_delta(
                float(current_verify.get("accuracy", 0.0))
                - float(reference_verify.get("accuracy", 0.0))
            ),
            "verify_repair_accuracy_delta": _round_delta(
                float(current_repair.get("accuracy", 0.0))
                - float(reference_repair.get("accuracy", 0.0))
            ),
            "wrong_detection_delta": int(current_verify.get("n_wrong_detected", 0))
            - int(reference_verify.get("n_wrong_detected", 0)),
            "false_positive_delta": int(current_verify.get("false_positives", 0))
            - int(reference_verify.get("false_positives", 0)),
            "wrong_detection_rate_delta": _round_delta(
                float(current_verify.get("wrong_detection_rate", 0.0))
                - float(reference_verify.get("wrong_detection_rate", 0.0))
            ),
            "false_positive_rate_delta": _round_delta(
                float(current_verify.get("false_positive_rate", 0.0))
                - float(reference_verify.get("false_positive_rate", 0.0))
            ),
            "repair_yield_delta": _round_delta(
                float(current_repair.get("repair_yield", 0.0))
                - float(reference_repair.get("repair_yield", 0.0))
            ),
            "mean_error_probability_delta": _round_delta(
                float(current_conf.get("mean_error_probability", 0.0))
                - float(reference_conf.get("mean_error_probability", 0.0))
            ),
            "mean_monitorability_confidence_delta": _round_delta(
                float(current_conf.get("mean_monitorability_confidence", 0.0))
                - float(reference_conf.get("mean_monitorability_confidence", 0.0))
            ),
            "verify_only_justified": verify_only_justified,
        }
    return {
        "source_artifact": "results/experiment_219_results.json",
        "same_cohort_as_exp219": same_cohort,
        "models": comparisons,
        "blockers": [dict(blocker) for blocker in blockers],
        "overall": {
            "verify_only_models_justified": justified,
            "verify_only_models_not_justified": not_justified,
        },
    }


def build_artifact_payload(
    *,
    output_path: Path,
    cohort: list[dict[str, Any]],
    paired_runs: list[dict[str, Any]],
    statistics: dict[str, Any],
    comparison_to_exp219: dict[str, Any],
    blockers: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    checkpoint_dir: Path,
    max_repairs: int,
    policy_path: Path,
    cohort_meta: dict[str, Any],
    run_status: str,
) -> dict[str, Any]:
    """Build the Exp 235 artifact while preserving the Exp 218-221 top-level schema."""
    return {
        "experiment": EXPERIMENT,
        "benchmark": BENCHMARK,
        "title": TITLE,
        "run_date": RUN_DATE,
        "schema": {
            "artifact": SCHEMA_ARTIFACT,
            "benchmark_case_schema": "gsm8k_semantic.v1",
        },
        "metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 3),
            "sample_seed": int(cohort_meta["sample_seed"]),
            "sample_size": int(cohort_meta["sample_size"]),
            "sample_strategy": "reused_exp219_cohort",
            "modes": ["baseline", "verify_only", "verify_repair"],
            "models": [dict(model) for model in MODEL_SPECS],
            "source_artifacts": [
                "results/experiment_219_results.json",
                _display_path(policy_path),
                "python/carnot/pipeline/semantic_verifier_v2.py",
            ],
            "output_path": _display_path(output_path),
            "checkpoint_dir": _display_path(checkpoint_dir),
            "checkpoint_pattern": (
                "results/checkpoints/experiment_235/<benchmark>__<model>__<mode>.json"
            ),
            "max_repairs": max_repairs,
            "policy_source": _display_path(policy_path),
            "semantic_verifier": {
                "name": "semantic_verifier_v2",
                "run_date": RUN_DATE,
                "policy_source": _display_path(policy_path),
            },
            "inference_mode": "live_gpu" if os.environ.get("CARNOT_FORCE_LIVE") == "1" else "live",
            "force_live": os.environ.get("CARNOT_FORCE_LIVE") == "1",
            "force_cpu": os.environ.get("CARNOT_FORCE_CPU") == "1",
        },
        "cohort": {
            "case_count": len(cohort),
            "case_ids": [str(case["case_id"]) for case in cohort],
            "cases": [dict(case) for case in cohort],
            "shared_with_experiment_219": bool(cohort_meta["same_as_exp219"]),
        },
        "paired_runs": list(paired_runs),
        "statistics": dict(statistics),
        "comparison_to_experiment_219": dict(comparison_to_exp219),
        "blockers": [dict(blocker) for blocker in blockers],
        "run_status": run_status,
    }


def _serialize_verification_result_v2(
    harness: Any,
    verification: Any,
) -> dict[str, Any]:  # pragma: no cover
    """Serialize verification output with semantic-verifier-v2 detail preserved."""
    payload = harness._serialize_verification_result(verification)
    payload["semantic_verifier_v2"] = harness._serialize_jsonable(
        getattr(verification, "semantic_verifier_v2", None)
    )
    return payload


def _semantic_v2_violation_count(result: Any) -> int:  # pragma: no cover
    semantic_v2 = getattr(result, "semantic_verifier_v2", None)
    if semantic_v2 is None:
        return 0
    claim_results = getattr(semantic_v2, "claim_results", [])
    return sum(1 for claim in claim_results if getattr(claim, "status", None) == "violated")


def _run_gsm8k_verify_only_v2(
    harness: Any,
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
    verification_trace = _serialize_verification_result_v2(harness, verification)
    semantic_v2 = getattr(verification, "semantic_verifier_v2", None)
    verdict = getattr(semantic_v2, "verdict", "unavailable")
    error_probability = float(getattr(semantic_v2, "semantic_error_probability", 0.0) or 0.0)
    monitorability = float(getattr(semantic_v2, "monitorability_confidence", 0.0) or 0.0)
    flagged = (not verification.verified) or bool(verification.violations)
    semantic_v2_detected_wrong_answer = (not bool(baseline["correct"])) and verdict == "violated"
    semantic_v2_false_positive = bool(baseline["correct"]) and verdict == "violated"
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
        "typed_reasoning_available": bool(verification.typed_reasoning),
        "typed_reasoning_parse_status": str(verification_trace["typed_reasoning_parse_status"]),
        "parseable": verification_trace["typed_reasoning_parse_status"] != "unavailable",
        "verification": verification_trace,
        "semantic_verifier_v2_verdict": verdict,
        "semantic_verifier_v2_error_probability": round(error_probability, 6),
        "semantic_verifier_v2_monitorability_confidence": round(monitorability, 6),
        "semantic_verifier_v2_violation_count": _semantic_v2_violation_count(verification),
        "semantic_verifier_v2_detected_wrong_answer": semantic_v2_detected_wrong_answer,
        "semantic_verifier_v2_false_positive": semantic_v2_false_positive,
        "correct": bool(baseline["correct"]),
        "accepted_correct": bool(baseline["correct"]) and not flagged,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _run_gsm8k_verify_repair_v2(  # pragma: no cover
    harness: Any,
    case: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model_spec: dict[str, str],
    model: Any,
    tokenizer: Any,
    policy: dict[str, Any],
    max_repairs: int,
) -> dict[str, Any]:
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"], timeout_seconds=30.0)
    current_response = str(baseline["response"])
    started = time.perf_counter()
    verification = pipeline.verify(str(case["question"]), current_response, domain="arithmetic")
    initial_trace = _serialize_verification_result_v2(harness, verification)
    initial_semantic_v2 = getattr(verification, "semantic_verifier_v2", None)
    history = [
        {
            "iteration": 0,
            "response_mode": baseline["response_mode"],
            "response": current_response,
            "verification": initial_trace,
        }
    ]
    if verification.verified:
        verdict = getattr(initial_semantic_v2, "verdict", "unavailable")
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
            "initial_semantic_verifier_v2_verdict": verdict,
            "final_semantic_verifier_v2_verdict": verdict,
            "initial_semantic_verifier_v2_error_probability": round(
                float(getattr(initial_semantic_v2, "semantic_error_probability", 0.0) or 0.0),
                6,
            ),
            "final_semantic_verifier_v2_error_probability": round(
                float(getattr(initial_semantic_v2, "semantic_error_probability", 0.0) or 0.0),
                6,
            ),
            "initial_semantic_verifier_v2_monitorability_confidence": round(
                float(getattr(initial_semantic_v2, "monitorability_confidence", 0.0) or 0.0),
                6,
            ),
            "final_semantic_verifier_v2_monitorability_confidence": round(
                float(getattr(initial_semantic_v2, "monitorability_confidence", 0.0) or 0.0),
                6,
            ),
            "unnecessary_repair": False,
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
        if response_mode in harness._STRUCTURED_RESPONSE_MODES:
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
                response = harness._generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=generated_prompt,
                    prompt_seed=repair_seed,
                    max_new_tokens=max_new_tokens,
                )
                fallback_record.update({"prompt": generated_prompt, "response": response})
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
            generation_trace = harness._build_generation_trace(
                tokenizer=tokenizer,
                attempts=[
                    harness._serialize_generation_attempt(
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
            current_response = harness._generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=repair_prompt,
                prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx,
                max_new_tokens=96,
            )
            generation_trace = harness._build_generation_trace(
                tokenizer=tokenizer,
                attempts=[
                    harness._serialize_generation_attempt(
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
        verification_trace = _serialize_verification_result_v2(harness, verification)
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

    final_semantic_v2 = getattr(verification, "semantic_verifier_v2", None)
    final_answer = harness._extract_final_number(current_response)
    final_correct = final_answer == int(case["ground_truth"])
    initial_correct = bool(baseline["correct"])
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_repair",
        "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
        "response_mode": baseline["response_mode"],
        "initial_response": baseline["response"],
        "final_response": current_response,
        "initial_correct": initial_correct,
        "correct": final_correct,
        "verified": verification.verified,
        "repaired": (not initial_correct) and final_correct,
        "n_repairs": n_repairs,
        "typed_reasoning_parse_status": str(
            history[-1]["verification"]["typed_reasoning_parse_status"]
        ),
        "initial_verification": initial_trace,
        "final_verification": history[-1]["verification"],
        "initial_semantic_verifier_v2_verdict": getattr(
            initial_semantic_v2,
            "verdict",
            "unavailable",
        ),
        "final_semantic_verifier_v2_verdict": getattr(final_semantic_v2, "verdict", "unavailable"),
        "initial_semantic_verifier_v2_error_probability": round(
            float(getattr(initial_semantic_v2, "semantic_error_probability", 0.0) or 0.0),
            6,
        ),
        "final_semantic_verifier_v2_error_probability": round(
            float(getattr(final_semantic_v2, "semantic_error_probability", 0.0) or 0.0),
            6,
        ),
        "initial_semantic_verifier_v2_monitorability_confidence": round(
            float(getattr(initial_semantic_v2, "monitorability_confidence", 0.0) or 0.0),
            6,
        ),
        "final_semantic_verifier_v2_monitorability_confidence": round(
            float(getattr(final_semantic_v2, "monitorability_confidence", 0.0) or 0.0),
            6,
        ),
        "unnecessary_repair": initial_correct and n_repairs > 0,
        "prompt_tokens": total_prompt_tokens,
        "response_tokens": total_response_tokens,
        "total_tokens": total_prompt_tokens + total_response_tokens,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }


def _run_model_suite_v2(  # pragma: no cover
    *,
    harness: Any,
    model_spec: dict[str, str],
    policy: dict[str, Any],
    cohort: list[dict[str, Any]],
    checkpoint_dir: Path,
    max_repairs: int,
) -> dict[str, Any]:
    """Run Exp 235 baseline, verify-only, and verify-repair for one live model."""
    model, tokenizer = harness._load_live_model(model_spec)
    try:
        baseline_runs = harness.run_mode(
            benchmark=BENCHMARK,
            model_name=model_spec["name"],
            mode="baseline",
            cases=cohort,
            checkpoint_dir=checkpoint_dir,
            execute_case=lambda case: harness._run_gsm8k_baseline(
                case,
                model_spec=model_spec,
                model=model,
                tokenizer=tokenizer,
                policy=policy,
            ),
        )
        baseline_by_case = {run["case_id"]: run for run in baseline_runs}
        verify_only_runs = harness.run_mode(
            benchmark=BENCHMARK,
            model_name=model_spec["name"],
            mode="verify_only",
            cases=cohort,
            checkpoint_dir=checkpoint_dir,
            execute_case=lambda case: _run_gsm8k_verify_only_v2(
                harness,
                case,
                baseline_by_case[str(case["case_id"])],
            ),
        )
        verify_repair_runs = harness.run_mode(
            benchmark=BENCHMARK,
            model_name=model_spec["name"],
            mode="verify_repair",
            cases=cohort,
            checkpoint_dir=checkpoint_dir,
            execute_case=lambda case: _run_gsm8k_verify_repair_v2(
                harness,
                case,
                baseline_by_case[str(case["case_id"])],
                model_spec=model_spec,
                model=model,
                tokenizer=tokenizer,
                policy=policy,
                max_repairs=max_repairs,
            ),
        )
    finally:
        harness._unload_live_model(model, tokenizer)
    model_summary = summarize_gsm8k_v2_runs(
        baseline_runs=baseline_runs,
        verify_only_runs=verify_only_runs,
        verify_repair_runs=verify_repair_runs,
    )
    return {
        "model_summary": model_summary,
        "paired_runs": [
            {
                "benchmark": BENCHMARK,
                "mode": "baseline",
                "model_name": model_spec["name"],
                "model_hf_id": model_spec["hf_id"],
                "summary": model_summary["baseline"],
                "cases": baseline_runs,
            },
            {
                "benchmark": BENCHMARK,
                "mode": "verify_only",
                "model_name": model_spec["name"],
                "model_hf_id": model_spec["hf_id"],
                "summary": model_summary["verify_only"],
                "cases": verify_only_runs,
            },
            {
                "benchmark": BENCHMARK,
                "mode": "verify_repair",
                "model_name": model_spec["name"],
                "model_hf_id": model_spec["hf_id"],
                "summary": model_summary["verify_repair"],
                "cases": verify_repair_runs,
            },
        ],
    }


def _run_live_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover
    harness = load_harness_module()
    started_at = utc_now()
    started = time.perf_counter()
    exp219_reference = load_exp219_reference()
    cohort, cohort_meta = load_shared_cohort()
    results_dir = get_repo_root() / "results"
    refreshed_policy = results_dir / "output_policy_233.json"
    policy_path = (
        refreshed_policy
        if refreshed_policy.exists()
        else (results_dir / "monitorability_policy_213.json")
    )
    policy = harness.load_monitorability_policy(policy_path)
    paired_runs: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {}
    blockers: list[dict[str, Any]] = []
    checkpoint_dir = Path(args.checkpoint_dir)

    for model_spec in MODEL_SPECS:
        try:
            suite = _run_model_suite_v2(
                harness=harness,
                model_spec=model_spec,
                policy=policy,
                cohort=cohort,
                checkpoint_dir=checkpoint_dir,
                max_repairs=int(args.max_repairs),
            )
        except Exception as exc:
            blockers.append(
                {
                    "model_name": model_spec["name"],
                    "stage": "model_suite",
                    "error": str(exc),
                }
            )
            continue
        statistics[model_spec["name"]] = suite["model_summary"]
        paired_runs.extend(suite["paired_runs"])

    comparison = build_exp219_comparison(
        current_statistics=statistics,
        exp219_statistics=dict(exp219_reference.get("statistics", {})),
        same_cohort=bool(cohort_meta["same_as_exp219"]),
        blockers=blockers,
    )
    run_status = "complete" if not blockers and len(statistics) == len(MODEL_SPECS) else "partial"
    return build_artifact_payload(
        output_path=Path(args.output),
        cohort=cohort,
        paired_runs=paired_runs,
        statistics=statistics,
        comparison_to_exp219=comparison,
        blockers=blockers,
        started_at=started_at,
        finished_at=utc_now(),
        runtime_seconds=time.perf_counter() - started,
        checkpoint_dir=checkpoint_dir,
        max_repairs=int(args.max_repairs),
        policy_path=policy_path,
        cohort_meta=cohort_meta,
        run_status=run_status,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Exp 235 CLI parser."""
    parser = argparse.ArgumentParser(
        description="Exp 235: live GSM8K semantic benchmark v2 on the Exp 219 cohort.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Artifact path. Defaults to results/experiment_235_results.json.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir(),
        help="Directory for per-model/mode checkpoints.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=DEFAULT_MAX_REPAIRS,
        help="Maximum verify-repair iterations per case.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = _run_live_benchmark(args)
    write_artifact(Path(args.output), payload)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
