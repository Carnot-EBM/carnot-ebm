#!/usr/bin/env python3
"""Experiment 206: Z3 live GSM8K benchmark on Gemma4-E4B-it.

Usage:
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_206_z3_live.py

Spec: REQ-VERIFY-009, SCENARIO-VERIFY-009
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CARNOT_FORCE_CPU"] = "0"
os.environ.setdefault("CARNOT_FORCE_LIVE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

import experiment_181_gsm8k_live_gpu as exp181  # noqa: E402
from carnot.pipeline.extract import ArithmeticExtractor  # noqa: E402
from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: E402
from carnot.pipeline.z3_extractor import Z3ArithmeticExtractor  # noqa: E402
from carnot.pipeline.z3_live_benchmark import (  # noqa: E402
    build_results_payload,
    compare_extractors,
    run_repair_loop,
    run_verify_only,
    sample_questions,
    summarize_cases,
)

SAMPLE_SIZE = 100
SAMPLE_SEED = 5
MAX_NEW_TOKENS = 768
MAX_REPAIRS = 3


def _baseline_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Solve step by step, showing all arithmetic. "
        "Give the final answer as a number.\n"
        "Format:\nAnswer: <number>"
    )


def main() -> int:
    start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    all_questions = exp181.load_gsm8k_questions()
    questions = sample_questions(all_questions, SAMPLE_SIZE, SAMPLE_SEED)
    gemma_config = next(
        config for config in exp181.MODEL_CONFIGS if config["name"] == "Gemma4-E4B-it"
    )

    print("=" * 78)
    print("EXPERIMENT 206: Z3 extractor on 100 live GSM8K (Gemma4-E4B-it)")
    print(f"  Sample size: {SAMPLE_SIZE} questions (seed={SAMPLE_SEED})")
    print("  Modes: baseline | verify-only | verify-repair")
    print("  Comparison: Z3 SMT extractor vs legacy regex extractor")
    print("=" * 78)

    z3_pipeline = VerifyRepairPipeline(
        model=None,
        domains=["arithmetic"],
        extractor=Z3ArithmeticExtractor(),
        timeout_seconds=30.0,
    )
    regex_pipeline = VerifyRepairPipeline(
        model=None,
        domains=["arithmetic"],
        extractor=ArithmeticExtractor(),
        timeout_seconds=30.0,
    )

    tokenizer = None
    model = None
    cases: list[dict[str, Any]] = []

    try:
        tokenizer, model, device_str = exp181.load_model_on_gpu(gemma_config)

        def generate_response(prompt: str) -> str:
            return exp181.generate_response(
                prompt,
                tokenizer,
                model,
                device_str,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        for sample_position, question in enumerate(questions, start=1):
            question_text = str(question["question"])
            ground_truth = int(question["ground_truth"])

            baseline_prompt = _baseline_prompt(question_text)
            baseline_response = generate_response(baseline_prompt)
            baseline_answer = exp181.extract_final_number(baseline_response)
            baseline_correct = baseline_answer == ground_truth

            z3_verify = run_verify_only(
                question=question_text,
                response=baseline_response,
                pipeline=z3_pipeline,
            )
            regex_verify = run_verify_only(
                question=question_text,
                response=baseline_response,
                pipeline=regex_pipeline,
            )
            z3_repair = run_repair_loop(
                question=question_text,
                ground_truth=ground_truth,
                initial_response=baseline_response,
                pipeline=z3_pipeline,
                generate_response=generate_response,
                extract_answer=exp181.extract_final_number,
                max_repairs=MAX_REPAIRS,
            )
            regex_repair = run_repair_loop(
                question=question_text,
                ground_truth=ground_truth,
                initial_response=baseline_response,
                pipeline=regex_pipeline,
                generate_response=generate_response,
                extract_answer=exp181.extract_final_number,
                max_repairs=MAX_REPAIRS,
            )

            case = {
                "sample_position": sample_position,
                "dataset_idx": int(question["idx"]),
                "question": question_text,
                "ground_truth": ground_truth,
                "baseline": {
                    "response": baseline_response,
                    "extracted_answer": baseline_answer,
                    "correct": baseline_correct,
                },
                "z3": {
                    "verify_only": z3_verify,
                    "verify_repair": z3_repair,
                },
                "regex": {
                    "verify_only": regex_verify,
                    "verify_repair": regex_repair,
                },
            }
            cases.append(case)

            print(
                f"[{sample_position:03d}/{SAMPLE_SIZE}] "
                f"dataset_idx={case['dataset_idx']} "
                f"baseline={baseline_correct} "
                f"z3(flagged={z3_verify['flagged']}, repair={z3_repair['correct']}) "
                f"regex(flagged={regex_verify['flagged']}, repair={regex_repair['correct']})"
            )

    finally:
        if tokenizer is not None and model is not None:
            exp181.unload_model(model, tokenizer, gemma_config["device_index"])

    z3_summary = summarize_cases(cases, extractor_key="z3", seed=206)
    regex_summary = summarize_cases(cases, extractor_key="regex", seed=306)
    comparison = compare_extractors(z3_summary, regex_summary)

    payload = build_results_payload(
        timestamp=timestamp,
        sample_seed=SAMPLE_SEED,
        sample_size=SAMPLE_SIZE,
        sample_dataset_indices=[int(question["idx"]) for question in questions],
        max_new_tokens=MAX_NEW_TOKENS,
        max_repairs=MAX_REPAIRS,
        runtime_seconds=time.perf_counter() - start,
        model_name=gemma_config["name"],
        hf_id=gemma_config["hf_id"],
        inference_mode="live_gpu",
        z3_summary=z3_summary,
        regex_summary=regex_summary,
        comparison=comparison,
        cases=cases,
    )

    out_path = RESULTS_DIR / "experiment_206_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\nZ3 summary:")
    print(json.dumps(z3_summary, indent=2))
    print("\nRegex summary:")
    print(json.dumps(regex_summary, indent=2))
    print("\nComparison:")
    print(json.dumps(comparison, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
