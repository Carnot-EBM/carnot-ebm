#!/usr/bin/env python3
"""Experiment 226: full HumanEval PBT benchmark on Gemma4-E4B-it.

Runs the official 164-problem HumanEval split with live Gemma4-E4B-it
generation, additive static/runtime instrumentation, Hypothesis-backed
property-based verification, and up to 3 repair attempts for baseline failures.

Spec: REQ-CODE-012, REQ-CODE-013, REQ-CODE-014,
SCENARIO-CODE-011, SCENARIO-CODE-012
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.pipeline.humaneval_live_benchmark import (  # noqa: E402
    HarnessResult,
    bootstrap_ci,
    bootstrap_delta_ci,
    build_candidate_code,
    execute_humaneval,
    run_instrumentation,
)
from carnot.pipeline.pbt_code_verifier import PBTCodeVerifier  # noqa: E402

__all__ = ["HarnessResult"]

EXPERIMENT_ID = 226
MODEL_NAME = "Gemma4-E4B-it"
MODEL_HF_ID = "google/gemma-4-E4B-it"
DEFAULT_SAMPLE_SEED = 226
DEFAULT_CHECKPOINT_INTERVAL = 10
DEFAULT_MAX_REPAIRS = 3
DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_PBT_MAX_EXAMPLES = 64
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
EXEC_TIMEOUT_SECONDS = 5.0
RESULTS_DIR = REPO_ROOT / "results"

PUBLISHED_BASELINES: list[dict[str, Any]] = [
    {
        "label": "Gemma 4 E4B model card",
        "metric": "LiveCodeBench v6 pass@1",
        "value": 0.52,
        "source_title": "Gemma 4 model card",
        "source_url": "https://ai.google.dev/gemma/docs/core/model_card_4",
        "notes": (
            "Official Google-published coding baseline for Gemma 4 E4B. "
            "This is the closest published code-generation metric found, but it "
            "is not the same benchmark as HumanEval."
        ),
    }
]


def default_output_path() -> Path:
    """Return the default Exp 226 artifact path."""
    return RESULTS_DIR / "experiment_226_results.json"


def default_checkpoint_path() -> Path:
    """Return the default Exp 226 checkpoint path."""
    return RESULTS_DIR / "checkpoints" / "experiment_226" / "gemma4_e4b_it_full.json"


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_parser() -> argparse.ArgumentParser:
    """Build the Exp 226 CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the full HumanEval PBT benchmark on live Gemma4-E4B-it.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional first-N cohort limit for local smoke tests. Default runs all 164 cases.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Deterministic seed used for per-case prompt seeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Artifact path for results/experiment_226_results.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="Checkpoint file path for resume support.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Save a resume checkpoint every N completed cases.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=DEFAULT_MAX_REPAIRS,
        help="Maximum repair attempts for failing baselines.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens for baseline and repair generations.",
    )
    parser.add_argument(
        "--pbt-max-examples",
        type=int,
        default=DEFAULT_PBT_MAX_EXAMPLES,
        help="Hypothesis max_examples for the PBT verifier.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap sample count for 95%% confidence intervals.",
    )
    return parser


def _stable_prompt_seed(sample_seed: int, case_id: str) -> int:
    """Derive a deterministic prompt seed from the case ID."""
    digest = hashlib.sha256(f"{sample_seed}:{case_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def load_humaneval_cases(
    *,
    sample_seed: int,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Load the official HumanEval split and enrich rows for Exp 226."""
    from datasets import load_dataset

    dataset = load_dataset("openai_humaneval", split="test")
    cases: list[dict[str, Any]] = []
    limit = len(dataset) if sample_size is None else min(sample_size, len(dataset))
    for dataset_idx in range(limit):
        row = dataset[dataset_idx]
        case_id = f"humaneval-{dataset_idx}"
        prompt_seed = _stable_prompt_seed(sample_seed, case_id)
        cases.append(
            {
                "case_id": case_id,
                "dataset_idx": dataset_idx,
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "sample_position": dataset_idx + 1,
                "prompt_seeds": {
                    "baseline": prompt_seed,
                    "verify_only": prompt_seed,
                    "verify_repair": prompt_seed,
                },
            }
        )
    return cases


def load_checkpoint(path: Path, expected_case_ids: list[str]) -> dict[str, Any]:
    """Load a checkpoint only when the cohort metadata still matches."""
    fresh = {
        "case_ids": list(expected_case_ids),
        "results_by_case": {},
    }
    if not path.exists():
        return fresh

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_ids") != expected_case_ids:
        return fresh
    results_by_case = payload.get("results_by_case")
    if not isinstance(results_by_case, dict):
        return fresh
    return {
        "case_ids": list(expected_case_ids),
        "results_by_case": dict(results_by_case),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint atomically with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _seed_runtime(seed: int) -> None:
    """Seed available RNGs for deterministic generation."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _best_cuda_device() -> str:
    """Pick the CUDA device with the most immediately free memory."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Exp 226 requires CUDA for live inference.")

    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = 0
        if free_bytes > best_free:
            best_index = index
            best_free = free_bytes
    return f"cuda:{best_index}"


def _load_live_model() -> tuple[Any, Any, str]:
    """Load Gemma4-E4B-it on the most suitable CUDA device."""
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    os.environ["CARNOT_FORCE_CPU"] = "0"

    from carnot.inference.model_loader import load_model

    device_str = _best_cuda_device()
    model, tokenizer = load_model(MODEL_HF_ID, device=device_str)
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load live model: {MODEL_HF_ID}")
    return model, tokenizer, device_str


def _unload_live_model(model: Any, tokenizer: Any) -> None:
    """Release live model resources after the run."""
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
) -> str:
    """Generate text deterministically from the loaded live model."""
    _seed_runtime(prompt_seed)
    from carnot.inference.model_loader import generate

    return str(generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens))


def build_generation_prompt(case: dict[str, Any]) -> str:
    """Build the baseline code-generation prompt for one HumanEval case."""
    return (
        "You are an expert Python programmer.\n"
        "Complete the following function.\n"
        "Return ONLY the function body lines. No def line. No markdown fences.\n"
        "Indent with 4 spaces.\n\n"
        f"{case['prompt']}"
    )


def _serialize_pbt_properties(properties: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(getattr(prop, "name", "")),
            "source": str(getattr(prop, "source", "")),
            "description": str(getattr(prop, "description", "")),
        }
        for prop in properties
    ]


def _serialize_pbt_failures(failures: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "property_name": str(getattr(failure, "property_name", "")),
            "source": str(getattr(failure, "source", "")),
            "description": str(getattr(failure, "description", "")),
            "input_args": list(getattr(failure, "input_args", ()) or ()),
            "actual": str(getattr(failure, "actual", "")),
            "expected": str(getattr(failure, "expected", "")),
            "error": getattr(failure, "error", None),
        }
        for failure in failures
    ]


def evaluate_candidate(
    case: dict[str, Any],
    candidate_code: str,
    *,
    pbt_max_examples: int,
) -> dict[str, Any]:
    """Run official tests, static/runtime checks, and the PBT verifier."""
    started = time.perf_counter()
    harness = execute_humaneval(candidate_code, case, timeout=EXEC_TIMEOUT_SECONDS)
    instrumentation = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=None,
    )
    pbt_result = PBTCodeVerifier(max_examples=pbt_max_examples).verify(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        str(case["test"]),
    )
    pbt_constraints = pbt_result.to_constraint_results()
    pbt_violations = [constraint.description for constraint in pbt_constraints][:5]
    detected = bool(instrumentation.get("detected") or pbt_result.failures)
    return {
        "harness": {
            "passed": harness.passed,
            "error_type": harness.error_type,
            "error_message": harness.error_message,
            "stdout": harness.stdout,
        },
        "instrumentation": dict(instrumentation),
        "pbt": {
            "verified": pbt_result.verified,
            "derived_properties": _serialize_pbt_properties(pbt_result.derived_properties),
            "failure_records": _serialize_pbt_failures(pbt_result.failures),
            "n_failures": len(pbt_result.failures),
            "violations": pbt_violations,
            "repair_feedback": pbt_result.repair_feedback(),
            "wall_clock_seconds": round(float(pbt_result.wall_clock_seconds), 6),
            "max_examples": int(getattr(pbt_result, "max_examples", pbt_max_examples)),
        },
        "detected": detected,
        "accepted": bool(harness.passed and not detected),
        "official_test_miss_caught_by_pbt": bool(harness.passed and len(pbt_result.failures) > 0),
        "latency_seconds": round(time.perf_counter() - started, 6),
    }


def build_pbt_repair_prompt(
    case: dict[str, Any],
    *,
    previous_body: str,
    evaluation: dict[str, Any],
    repair_idx: int,
) -> str:
    """Build a repair prompt that includes official and PBT-specific failures."""
    harness = evaluation["harness"]
    instrumentation = evaluation["instrumentation"]
    pbt = evaluation["pbt"]

    lines = [
        f"You are fixing a Python function (repair attempt {repair_idx + 1}).",
        "",
        "Function prompt:",
        str(case["prompt"]).rstrip(),
        "",
        "Previous function body:",
        "    " + (previous_body.strip() or "pass").replace("\n", "\n    "),
        "",
        "HumanEval test failure:",
        f"  - {harness.get('error_message') or harness.get('error_type')}",
    ]

    for heading, key in (
        ("Static constraint findings:", "constraint_feedback"),
        ("Runtime instrumentation findings:", "dynamic_violations"),
    ):
        findings = list(instrumentation.get(key, []))
        if findings:
            lines.extend(["", heading])
            lines.extend(f"  - {finding}" for finding in findings[:5])

    pbt_lines = list(pbt.get("violations", []))
    if pbt_lines:
        lines.extend(["", "Hypothesis-backed PBT counterexamples:"])
        lines.extend(f"  - {line}" for line in pbt_lines[:5])
    repair_feedback = str(pbt.get("repair_feedback", "")).strip()
    if repair_feedback:
        lines.extend(["", "PBT repair feedback:", repair_feedback])

    lines.extend(
        [
            "",
            "Write ONLY the corrected function body. No markdown fences.",
            "Indent with 4 spaces.",
        ]
    )
    return "\n".join(lines)


def _baseline_record(
    body: str,
    candidate_code: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    harness = evaluation["harness"]
    instrumentation = evaluation["instrumentation"]
    pbt = evaluation["pbt"]
    return {
        "passed": bool(harness["passed"]),
        "error_type": str(harness["error_type"]),
        "error_message": str(harness["error_message"]),
        "body": body,
        "candidate_code": candidate_code,
        "detected": bool(evaluation["detected"]),
        "accepted": bool(evaluation["accepted"]),
        "official_test_miss_caught_by_pbt": bool(evaluation["official_test_miss_caught_by_pbt"]),
        "n_static_violations": int(instrumentation.get("n_static_violations", 0)),
        "n_dynamic_violations": int(instrumentation.get("n_dynamic_violations", 0)),
        "constraint_feedback": list(instrumentation.get("constraint_feedback", [])),
        "dynamic_violations": list(instrumentation.get("dynamic_violations", [])),
        "probe_inputs": list(instrumentation.get("probe_inputs", [])),
        "n_pbt_failures": int(pbt.get("n_failures", 0)),
        "pbt_violations": list(pbt.get("violations", [])),
        "pbt_derived_properties": list(pbt.get("derived_properties", [])),
        "pbt_failure_records": list(pbt.get("failure_records", [])),
        "pbt_verified": bool(pbt.get("verified", False)),
        "latency_seconds": float(evaluation.get("latency_seconds", 0.0)),
    }


def _history_entry(
    *,
    iteration: int,
    body: str,
    candidate_code: str,
    evaluation: dict[str, Any],
    repair_prompt: str | None = None,
) -> dict[str, Any]:
    entry = {
        "iteration": iteration,
        "body": body,
        "candidate_code": candidate_code,
        "harness": dict(evaluation["harness"]),
        "instrumentation": dict(evaluation["instrumentation"]),
        "pbt": dict(evaluation["pbt"]),
        "detected": bool(evaluation["detected"]),
        "accepted": bool(evaluation["accepted"]),
    }
    if repair_prompt is not None:
        entry["repair_prompt"] = repair_prompt
    return entry


def run_case(
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    device_str: str,
    max_repairs: int,
    pbt_max_examples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run baseline generation plus verify-repair for one HumanEval case."""
    del device_str
    baseline_prompt = build_generation_prompt(case)
    baseline_body = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=baseline_prompt,
        prompt_seed=int(case["prompt_seeds"]["baseline"]),
        max_new_tokens=max_new_tokens,
    )
    baseline_code = build_candidate_code(str(case["prompt"]), baseline_body)
    baseline_eval = evaluate_candidate(case, baseline_code, pbt_max_examples=pbt_max_examples)

    result = {
        "case_id": str(case["case_id"]),
        "dataset_idx": int(case["dataset_idx"]),
        "task_id": str(case["task_id"]),
        "entry_point": str(case["entry_point"]),
        "baseline": _baseline_record(baseline_body, baseline_code, baseline_eval),
        "verify_only": {
            "detected": bool(baseline_eval["detected"]),
            "accepted": bool(baseline_eval["accepted"]),
            "official_test_miss_caught_by_pbt": bool(
                baseline_eval["official_test_miss_caught_by_pbt"]
            ),
            "n_pbt_failures": int(baseline_eval["pbt"]["n_failures"]),
            "pbt_violations": list(baseline_eval["pbt"]["violations"]),
            "latency_seconds": float(baseline_eval.get("latency_seconds", 0.0)),
        },
        "verify_repair": {
            "passed": bool(baseline_eval["harness"]["passed"]),
            "repaired": False,
            "n_repairs": 0,
            "final_body": baseline_body,
            "final_code": baseline_code,
            "final_detected": bool(baseline_eval["detected"]),
            "final_accepted": bool(baseline_eval["accepted"]),
            "final_error_type": str(baseline_eval["harness"]["error_type"]),
            "final_error_message": str(baseline_eval["harness"]["error_message"]),
        },
        "history": [
            _history_entry(
                iteration=0,
                body=baseline_body,
                candidate_code=baseline_code,
                evaluation=baseline_eval,
            )
        ],
    }

    if bool(baseline_eval["harness"]["passed"]):
        return result

    current_body = baseline_body
    current_code = baseline_code
    current_eval = baseline_eval

    for repair_idx in range(max_repairs):
        repair_prompt = build_pbt_repair_prompt(
            case,
            previous_body=current_body,
            evaluation=current_eval,
            repair_idx=repair_idx,
        )
        current_body = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx + 1,
            max_new_tokens=max_new_tokens,
        )
        current_code = build_candidate_code(str(case["prompt"]), current_body)
        current_eval = evaluate_candidate(case, current_code, pbt_max_examples=pbt_max_examples)
        result["history"].append(
            _history_entry(
                iteration=repair_idx + 1,
                body=current_body,
                candidate_code=current_code,
                evaluation=current_eval,
                repair_prompt=repair_prompt,
            )
        )
        result["verify_repair"].update(
            {
                "passed": bool(current_eval["harness"]["passed"]),
                "repaired": (not bool(baseline_eval["harness"]["passed"]))
                and bool(current_eval["harness"]["passed"]),
                "n_repairs": repair_idx + 1,
                "final_body": current_body,
                "final_code": current_code,
                "final_detected": bool(current_eval["detected"]),
                "final_accepted": bool(current_eval["accepted"]),
                "final_error_type": str(current_eval["harness"]["error_type"]),
                "final_error_message": str(current_eval["harness"]["error_message"]),
            }
        )
        if bool(current_eval["harness"]["passed"]):
            break

    return result


def run_benchmark(
    cases: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device_str: str,
    checkpoint_path: Path,
    checkpoint_interval: int,
    max_repairs: int,
    pbt_max_examples: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run the full benchmark with periodic checkpointing and resume support."""
    case_ids = [str(case["case_id"]) for case in cases]
    checkpoint = load_checkpoint(checkpoint_path, case_ids)
    results_by_case = dict(checkpoint["results_by_case"])
    since_last_checkpoint = 0

    for case in cases:
        case_id = str(case["case_id"])
        if case_id in results_by_case:
            continue
        result = run_case(
            case,
            model=model,
            tokenizer=tokenizer,
            device_str=device_str,
            max_repairs=max_repairs,
            pbt_max_examples=pbt_max_examples,
            max_new_tokens=max_new_tokens,
        )
        results_by_case[case_id] = result
        since_last_checkpoint += 1
        if since_last_checkpoint >= checkpoint_interval:
            save_checkpoint(
                checkpoint_path,
                {
                    "case_ids": case_ids,
                    "results_by_case": results_by_case,
                },
            )
            since_last_checkpoint = 0

    if since_last_checkpoint > 0 or not checkpoint_path.exists():
        save_checkpoint(
            checkpoint_path,
            {
                "case_ids": case_ids,
                "results_by_case": results_by_case,
            },
        )

    return [dict(results_by_case[case_id]) for case_id in case_ids]


def _published_comparison(
    *,
    baseline_pass_at_1: float,
    verify_repair_pass_at_1: float,
    published_baselines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for entry in published_baselines:
        value = float(entry["value"])
        comparisons.append(
            {
                "label": str(entry["label"]),
                "metric": str(entry["metric"]),
                "value": value,
                "source_title": str(entry["source_title"]),
                "source_url": str(entry["source_url"]),
                "notes": str(entry.get("notes", "")),
                "baseline_delta": baseline_pass_at_1 - value,
                "verify_repair_delta": verify_repair_pass_at_1 - value,
            }
        )
    return comparisons


def _technical_report_summary(
    *,
    n_cases: int,
    baseline: dict[str, Any],
    verify_repair: dict[str, Any],
    improvement: dict[str, Any],
    repair_stats: dict[str, Any],
    published_comparison: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_percent = baseline["pass_at_1"] * 100.0
    repair_percent = verify_repair["pass_at_1"] * 100.0
    delta_points = improvement["delta"] * 100.0
    first_published = published_comparison[0] if published_comparison else None
    if first_published is None:
        publication_line = (
            "No published Gemma4-E4B-it HumanEval baseline is attached to this artifact."
        )
    else:
        publication_line = (
            f"Against the published {first_published['metric']} reference from "
            f"{first_published['label']} ({first_published['value'] * 100.0:.1f}%), "
            f"the local baseline delta is {first_published['baseline_delta'] * 100.0:+.1f}pp and "
            f"the verify-repair delta is {first_published['verify_repair_delta'] * 100.0:+.1f}pp. "
            f"{first_published.get('notes', '')}".strip()
        )
    paragraph = (
        f"On the full 164-problem HumanEval benchmark contract (n={n_cases} run cases), "
        f"live {MODEL_NAME} reached baseline pass@1 {baseline_percent:.1f}% "
        f"[{baseline['ci_lower'] * 100.0:.1f}%, {baseline['ci_upper'] * 100.0:.1f}%] and "
        f"verify-repair pass@1 {repair_percent:.1f}% "
        f"[{verify_repair['ci_lower'] * 100.0:.1f}%, {verify_repair['ci_upper'] * 100.0:.1f}%], "
        f"for a paired improvement of {delta_points:+.1f}pp "
        f"[{improvement['ci_lower'] * 100.0:+.1f}pp, {improvement['ci_upper'] * 100.0:+.1f}pp]. "
        f"PBT-guided repair fixed {repair_stats['n_repaired']}/"
        f"{repair_stats['n_problems_needing_repair']} failing baselines "
        f"({repair_stats['repair_success_rate'] * 100.0:.1f}%). {publication_line}"
    )
    bullets = [
        f"Baseline pass@1: {baseline_percent:.1f}% across {n_cases} cases.",
        f"Verify-repair pass@1: {repair_percent:.1f}% ({delta_points:+.1f}pp vs baseline).",
        f"Repair yield: {repair_stats['n_repaired']}/{repair_stats['n_problems_needing_repair']} "
        f"failing baselines fixed.",
    ]
    if first_published is not None:
        bullets.append(
            f"Published comparison: {first_published['label']} reports "
            f"{first_published['value'] * 100.0:.1f}%."
        )
    return {
        "paragraph": paragraph,
        "bullets": bullets,
    }


def summarize_results(
    cases: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    published_baselines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the full benchmark with bootstrap CIs and report text."""
    if not cases:
        raise ValueError("Cannot summarize an empty benchmark run.")

    baseline_flags = [bool(case["baseline"]["passed"]) for case in cases]
    verify_only_flags = [
        bool(case["baseline"]["passed"]) and not bool(case["verify_only"]["detected"])
        for case in cases
    ]
    repair_flags = [bool(case["verify_repair"]["passed"]) for case in cases]

    base_acc, base_lo, base_hi = bootstrap_ci(
        baseline_flags,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    repair_acc, repair_lo, repair_hi = bootstrap_ci(
        repair_flags,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    delta, delta_lo, delta_hi = bootstrap_delta_ci(
        baseline_flags,
        repair_flags,
        n_bootstrap=n_bootstrap,
        seed=seed + 2,
    )

    n_wrong_answers = sum(1 for flag in baseline_flags if not flag)
    n_wrong_detected = sum(
        1
        for case in cases
        if not bool(case["baseline"]["passed"]) and bool(case["verify_only"]["detected"])
    )
    false_positives = sum(
        1
        for case in cases
        if bool(case["baseline"]["passed"]) and bool(case["verify_only"]["detected"])
    )
    n_repaired = sum(1 for case in cases if bool(case["verify_repair"]["repaired"]))
    repair_iterations = [int(case["verify_repair"].get("n_repairs", 0)) for case in cases]

    published_comparison = _published_comparison(
        baseline_pass_at_1=base_acc,
        verify_repair_pass_at_1=repair_acc,
        published_baselines=published_baselines,
    )
    summary = {
        "baseline": {
            "pass_at_1": base_acc,
            "ci_lower": base_lo,
            "ci_upper": base_hi,
            "n_correct": int(sum(baseline_flags)),
        },
        "verify_only": {
            "accepted_pass_at_1": sum(1 for flag in verify_only_flags if flag) / len(cases),
            "n_wrong_answers": n_wrong_answers,
            "n_wrong_detected": n_wrong_detected,
            "false_positives": false_positives,
            "official_test_misses_caught_by_pbt": sum(
                1 for case in cases if bool(case["verify_only"]["official_test_miss_caught_by_pbt"])
            ),
            "problems_with_pbt_failures": sum(
                1 for case in cases if int(case["verify_only"].get("n_pbt_failures", 0)) > 0
            ),
            "total_pbt_failures": sum(
                int(case["verify_only"].get("n_pbt_failures", 0)) for case in cases
            ),
        },
        "verify_repair": {
            "pass_at_1": repair_acc,
            "ci_lower": repair_lo,
            "ci_upper": repair_hi,
            "n_correct": int(sum(repair_flags)),
            "n_repaired": n_repaired,
        },
        "improvement": {
            "delta": delta,
            "ci_lower": delta_lo,
            "ci_upper": delta_hi,
        },
        "repair_stats": {
            "n_problems_needing_repair": n_wrong_answers,
            "n_repaired": n_repaired,
            "repair_success_rate": (n_repaired / n_wrong_answers if n_wrong_answers else 1.0),
            "avg_repair_iterations": (
                sum(repair_iterations) / len(cases) if repair_iterations else 0.0
            ),
        },
        "published_comparison": published_comparison,
    }
    summary["technical_report_summary"] = _technical_report_summary(
        n_cases=len(cases),
        baseline=summary["baseline"],
        verify_repair=summary["verify_repair"],
        improvement=summary["improvement"],
        repair_stats=summary["repair_stats"],
        published_comparison=published_comparison,
    )
    return summary


def build_results_payload(
    *,
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    output_path: Path,
    checkpoint_path: Path,
    device_str: str,
    sample_seed: int,
    checkpoint_interval: int,
    max_repairs: int,
    pbt_max_examples: int,
    n_bootstrap: int,
    cohort: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable Exp 226 artifact payload."""
    return {
        "experiment": EXPERIMENT_ID,
        "benchmark": "humaneval_pbt_full",
        "title": "Full HumanEval PBT benchmark on Gemma4-E4B-it",
        "run_date": started_at[:10].replace("-", ""),
        "schema": {
            "artifact": "carnot.humaneval_pbt_full.v1",
            "benchmark_case_schema": "humaneval_pbt_full.v1",
        },
        "metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 3),
            "model_name": MODEL_NAME,
            "model_hf_id": MODEL_HF_ID,
            "device": device_str,
            "dataset_source": "HumanEval (openai_humaneval)",
            "sample_seed": sample_seed,
            "sample_size": len(cohort),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_interval": checkpoint_interval,
            "max_repairs": max_repairs,
            "pbt_max_examples": pbt_max_examples,
            "bootstrap_samples": n_bootstrap,
            "confidence_level": 0.95,
            "inference_mode": "live_gpu",
            "output_path": str(output_path),
            "force_live": os.environ.get("CARNOT_FORCE_LIVE") == "1",
            "force_cpu": os.environ.get("CARNOT_FORCE_CPU") == "1",
        },
        "cohort": {
            "case_count": len(cohort),
            "case_ids": [str(case["case_id"]) for case in cohort],
            "cases": [dict(case) for case in cohort],
        },
        "statistics": dict(statistics),
        "per_problem_results": list(case_results),
    }


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write the final Exp 226 artifact with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run Exp 226 end to end."""
    args = build_parser().parse_args(argv)
    started_at = utc_now()
    start = time.perf_counter()
    cohort = load_humaneval_cases(sample_seed=args.sample_seed, sample_size=args.sample_size)

    model, tokenizer, device_str = _load_live_model()
    try:
        case_results = run_benchmark(
            cohort,
            model=model,
            tokenizer=tokenizer,
            device_str=device_str,
            checkpoint_path=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval,
            max_repairs=args.max_repairs,
            pbt_max_examples=args.pbt_max_examples,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        _unload_live_model(model, tokenizer)

    finished_at = utc_now()
    statistics = summarize_results(
        case_results,
        n_bootstrap=args.bootstrap_samples,
        seed=args.sample_seed,
        published_baselines=PUBLISHED_BASELINES,
    )
    payload = build_results_payload(
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=time.perf_counter() - start,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        device_str=device_str,
        sample_seed=args.sample_seed,
        checkpoint_interval=args.checkpoint_interval,
        max_repairs=args.max_repairs,
        pbt_max_examples=args.pbt_max_examples,
        n_bootstrap=args.bootstrap_samples,
        cohort=cohort,
        case_results=case_results,
        statistics=statistics,
    )
    write_artifact(args.output, payload)

    baseline = statistics["baseline"]
    verify_repair = statistics["verify_repair"]
    improvement = statistics["improvement"]
    print(f"Saved results to {args.output}")
    print(
        f"Baseline pass@1: {baseline['n_correct']}/{len(case_results)} "
        f"({baseline['pass_at_1']:.1%})"
    )
    print(
        f"Verify-repair pass@1: {verify_repair['n_correct']}/{len(case_results)} "
        f"({verify_repair['pass_at_1']:.1%})"
    )
    print(
        f"Improvement delta: {improvement['delta']:+.1%} "
        f"[{improvement['ci_lower']:+.1%}, {improvement['ci_upper']:+.1%}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
