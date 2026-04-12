#!/usr/bin/env python3
"""Experiment 203: live extraction autopsy for Gemma4-E4B-it on GSM8K.

Usage:
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_203_extraction_autopsy.py

Spec: REQ-VERIFY-008, SCENARIO-VERIFY-008
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Force live GPU inference before importing the Exp 181 helpers.
os.environ["CARNOT_FORCE_CPU"] = "0"
os.environ.setdefault("CARNOT_FORCE_LIVE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

from carnot.pipeline.extraction_autopsy import (  # noqa: E402
    build_case_record,
    select_showcase_cases,
    summarize_cases,
)

SAMPLE_SIZE = 20
SAMPLE_SEED = 5
N_CORRECT_SHOWCASE = 3
MAX_NEW_TOKENS = 768
PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Solve step by step, showing all arithmetic. "
    "Give the final answer as a number.\n"
    "Format:\nAnswer: <number>"
)

# Hand-written overrides for wrong live cases after inspection against GSM8K gold
# answers. These preserve the raw model response but replace the generic
# heuristic diagnosis with a question-specific autopsy.
MANUAL_OVERRIDES: dict[int, dict[str, str]] = {
    923: {
        "failure_category": "missing_intermediate_step",
        "actual_error": (
            "After correctly deriving 27 - 15 = 12 cups for chamomile+mint, "
            "the response never completes the remaining chain 12 / 2 = 6 mint "
            "cups total, then 6 / 3 = 2 mint cups per row. It stops at a row "
            "total instead of the mint-per-row quantity."
        ),
        "proposed_extraction": (
            "Build a quantity graph from the response and verify the full chain "
            "27-15 -> 12/2 -> 6/3 instead of only isolated explicit equations."
        ),
    },
    814: {
        "failure_category": "semantic_modeling_error",
        "actual_error": (
            "The response models cost as a continuous $0.0125/minute rate and "
            "treats the weaker/stronger friends as playing 2, 2, and 6 hours. "
            "The gold solution scales quarter duration instead: half-as-long "
            "and 1.5x-as-long describe minutes per quarter, not hours played."
        ),
        "proposed_extraction": (
            "Use unit-aware semantic parsing that binds 'half as long' to play "
            "duration per quarter and verifies quarter-insertion counts as "
            "integer quantities."
        ),
    },
    943: {
        "failure_category": "reading_comprehension_error",
        "actual_error": (
            "The response interprets 'sells them for 40' as $40 per CD and "
            "therefore computes revenue 5 * 40 = 200. The gold answer treats "
            "40 as the total resale amount for all 5 CDs, so the correct net "
            "loss is 90 - 40 = 50."
        ),
        "proposed_extraction": (
            "Add question-grounded entity/value alignment that distinguishes a "
            "total sale amount from a per-item price before checking the "
            "downstream arithmetic."
        ),
    },
}


def _load_exp181_helpers() -> Any:
    """Import Exp 181 lazily so tests do not need GPU dependencies."""
    import experiment_181_gsm8k_live_gpu as exp181

    return exp181


def _render_regex_matches(case: dict[str, Any]) -> str:
    matches = case["regex_matches"]
    if not matches:
        return "  [no regex matches]"

    lines = []
    for idx, match in enumerate(matches, start=1):
        verdict = "OK" if match["satisfied"] else "VIOLATION"
        lines.append(
            f"  {idx}. {verdict} | {match['matched_text']} | correct={match['correct_result']}"
        )
    return "\n".join(lines)


def _print_case(case: dict[str, Any], label: str) -> None:
    print("=" * 78)
    print(
        f"{label} | sample={case['sample_position']} | dataset_idx={case['dataset_idx']} | "
        f"correct={case['correct']}"
    )
    print(f"Question: {case['question']}")
    print(f"Ground truth: {case['ground_truth']}")
    print(f"Extracted answer: {case['extracted_answer']}")
    print(f"Failure category: {case['failure_category']}")
    pipeline = case.get("pipeline_verification")
    if pipeline is not None:
        print(
            "Pipeline: "
            f"verified={pipeline['verified']} "
            f"constraints={pipeline['n_constraints']} "
            f"violations={pipeline['n_violations']} "
            f"energy={pipeline['energy']}"
        )
    print("Response:")
    print(case["response"])
    print("Regex matches:")
    print(_render_regex_matches(case))
    print(f"Actual error: {case['actual_error']}")
    print(f"Would have caught it: {case['proposed_extraction']}")


def main() -> int:
    start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    exp181 = _load_exp181_helpers()
    all_questions = exp181.load_gsm8k_questions()
    sample_indices = list(range(len(all_questions)))
    random.Random(SAMPLE_SEED).shuffle(sample_indices)
    questions = [all_questions[idx] for idx in sample_indices[:SAMPLE_SIZE]]
    gemma_config = next(
        config for config in exp181.MODEL_CONFIGS if config["name"] == "Gemma4-E4B-it"
    )

    print("=" * 78)
    print("EXPERIMENT 203: Extraction autopsy for Gemma4-E4B-it")
    print(f"  Sample size: {SAMPLE_SIZE} GSM8K questions (seeded shuffle, seed={SAMPLE_SEED})")
    print("  Goal: explain why ArithmeticExtractor finds zero violations on wrong answers")
    print("=" * 78)

    tokenizer = None
    model = None
    cases: list[dict[str, Any]] = []
    try:
        tokenizer, model, device_str = exp181.load_model_on_gpu(gemma_config)
        for sample_position, question in enumerate(questions):
            prompt = PROMPT_TEMPLATE.format(question=question["question"])
            response = exp181.generate_response(
                prompt,
                tokenizer,
                model,
                device_str,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            case = build_case_record(
                sample_position=sample_position,
                dataset_idx=int(question["idx"]),
                question=str(question["question"]),
                ground_truth=int(question["ground_truth"]),
                response=response,
                prompt=prompt,
                manual_override=MANUAL_OVERRIDES.get(int(question["idx"])),
            )
            case["pipeline_verification"] = exp181.verify_with_pipeline(
                question["question"],
                response,
            )
            cases.append(case)
            print(
                f"[{sample_position + 1:02d}/{SAMPLE_SIZE}] "
                f"dataset_idx={case['dataset_idx']} "
                f"answer={case['extracted_answer']} "
                f"correct={case['correct']} "
                f"regex_matches={case['n_regex_matches']} "
                f"regex_violations={case['n_regex_violations']}"
            )
    finally:
        if tokenizer is not None and model is not None:
            exp181.unload_model(model, tokenizer, gemma_config["device_index"])

    showcase = select_showcase_cases(cases, n_correct_examples=N_CORRECT_SHOWCASE)
    summary = summarize_cases(cases)
    summary["n_wrong_with_pipeline_violations"] = sum(
        1
        for case in cases
        if (not case["correct"]) and case["pipeline_verification"]["n_violations"] > 0
    )

    output = {
        "experiment": 203,
        "title": "Extraction autopsy — why ArithmeticExtractor finds 0 violations on IT models",
        "metadata": {
            "timestamp": timestamp,
            "model_name": gemma_config["name"],
            "hf_id": gemma_config["hf_id"],
            "inference_mode": "live_gpu",
            "sample_size": SAMPLE_SIZE,
            "sample_seed": SAMPLE_SEED,
            "sample_strategy": "seeded_shuffle_first_20",
            "sample_dataset_indices": [int(question["idx"]) for question in questions],
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_style": "verify_only_show_all_arithmetic",
            "manual_override_count": len(MANUAL_OVERRIDES),
            "runtime_seconds": round(time.perf_counter() - start, 3),
        },
        "summary": summary,
        "wrong_answer_autopsies": showcase["wrong_answer_autopsies"],
        "correct_answer_examples": showcase["correct_answer_examples"],
        "all_cases": cases,
    }

    out_path = RESULTS_DIR / "experiment_203_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")

    print("\nWrong-answer autopsies:")
    for case in showcase["wrong_answer_autopsies"]:
        _print_case(case, label="WRONG")

    print("\nCorrect-answer contrast cases:")
    for case in showcase["correct_answer_examples"]:
        _print_case(case, label="CORRECT")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
