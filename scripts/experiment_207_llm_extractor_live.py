#!/usr/bin/env python3
"""Experiment 207: LLM extractor live GSM8K benchmark on Gemma4-E4B-it.

Usage:
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_207_llm_extractor_live.py

Compares the Exp 205 LLMConstraintExtractor against the Exp 204 Z3 extractor on
the exact same 100-question live Gemma4-E4B-it cohort already captured in
`results/experiment_206_results.json`. Baseline responses are reused from
Exp 206 so the comparison stays perfectly paired; this run performs the LLM
verify-only / verify-repair passes live on GPU and records the head-to-head
comparison artifact in `results/experiment_207_results.json`.

Spec: REQ-VERIFY-009, REQ-VERIFY-010, SCENARIO-VERIFY-009, SCENARIO-VERIFY-010
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
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
from carnot.pipeline.llm_extractor import LLMConstraintExtractor  # noqa: E402
from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: E402
from carnot.pipeline.z3_live_benchmark import (  # noqa: E402
    build_comparison_payload,
    compare_named_extractors,
    run_repair_loop,
    run_verify_only,
    summarize_cases,
)

SOURCE_RESULTS_PATH = RESULTS_DIR / "experiment_206_results.json"
SAMPLE_SIZE = 100
SAMPLE_SEED = 5
MAX_NEW_TOKENS = 768
EXTRACTOR_MAX_NEW_TOKENS = 256
MAX_REPAIRS = 3
PIPELINE_TIMEOUT_SECONDS = 180.0


def _load_exp206_payload() -> dict[str, Any]:
    payload = json.loads(SOURCE_RESULTS_PATH.read_text())

    if int(payload.get("experiment", -1)) != 206:
        raise RuntimeError(f"{SOURCE_RESULTS_PATH} is not an Exp 206 artifact.")

    metadata = dict(payload.get("metadata", {}))
    if int(metadata.get("sample_size", -1)) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Expected Exp 206 sample_size={SAMPLE_SIZE}, got {metadata.get('sample_size')}."
        )
    if int(metadata.get("sample_seed", -1)) != SAMPLE_SEED:
        raise RuntimeError(
            f"Expected Exp 206 sample_seed={SAMPLE_SEED}, got {metadata.get('sample_seed')}."
        )
    if metadata.get("inference_mode") != "live_gpu":
        raise RuntimeError("Exp 207 requires the live_gpu Exp 206 baseline artifact.")

    results = list(payload.get("results", []))
    if len(results) != SAMPLE_SIZE:
        raise RuntimeError(f"Expected {SAMPLE_SIZE} Exp 206 cases, got {len(results)}.")

    return payload


def _select_device_index(preferred_index: int) -> int:
    """Pick the GPU with the most free VRAM, preferring the requested device on ties."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    candidates: list[tuple[int, int]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        index_text, free_text = [part.strip() for part in line.split(",", maxsplit=1)]
        candidates.append((int(index_text), int(free_text)))

    if not candidates:
        raise RuntimeError("nvidia-smi returned no GPU memory data.")

    return max(
        candidates,
        key=lambda item: (item[1], item[0] == preferred_index),
    )[0]


def main() -> int:
    start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    source_payload = _load_exp206_payload()
    source_cases = list(source_payload["results"])
    source_z3_summary = dict(source_payload["statistics"]["z3"])
    sample_dataset_indices = list(source_payload["metadata"]["sample_dataset_indices"])

    base_gemma_config = next(
        config for config in exp181.MODEL_CONFIGS if config["name"] == "Gemma4-E4B-it"
    )
    runtime_device_index = _select_device_index(int(base_gemma_config["device_index"]))
    gemma_config = dict(base_gemma_config)
    gemma_config["device_index"] = runtime_device_index

    print("=" * 78)
    print("EXPERIMENT 207: LLM extractor on 100 live GSM8K (Gemma4-E4B-it)")
    print(f"  Source cohort: Exp 206 shared baseline responses ({SAMPLE_SIZE} questions)")
    print("  Modes: baseline | verify-only | verify-repair")
    print("  Comparison: LLMConstraintExtractor vs Z3ArithmeticExtractor")
    print("  Pairing: baseline responses reused from results/experiment_206_results.json")
    print(
        f"  Runtime device: cuda:{runtime_device_index} "
        f"(preferred cuda:{base_gemma_config['device_index']})"
    )
    print("=" * 78)

    tokenizer = None
    model = None
    cases: list[dict[str, Any]] = []

    try:
        tokenizer, model, device_str = exp181.load_model_on_gpu(gemma_config)

        def extractor_generate(
            model: Any,
            tokenizer: Any,
            prompt: str,
            requested_tokens: int,
        ) -> str:
            return exp181.generate_response(
                prompt,
                tokenizer,
                model,
                device_str,
                max_new_tokens=requested_tokens,
            )

        llm_pipeline = VerifyRepairPipeline(
            model=None,
            domains=["arithmetic"],
            extractor=LLMConstraintExtractor(
                model=model,
                tokenizer=tokenizer,
                generate_fn=extractor_generate,
                max_new_tokens=EXTRACTOR_MAX_NEW_TOKENS,
            ),
            timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
        )

        def generate_response(prompt: str) -> str:
            return exp181.generate_response(
                prompt,
                tokenizer,
                model,
                device_str,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        for source_case in source_cases:
            question_text = str(source_case["question"])
            ground_truth = int(source_case["ground_truth"])
            baseline = dict(source_case["baseline"])
            baseline_response = str(baseline["response"])

            llm_verify = run_verify_only(
                question=question_text,
                response=baseline_response,
                pipeline=llm_pipeline,
            )
            llm_repair = run_repair_loop(
                question=question_text,
                ground_truth=ground_truth,
                initial_response=baseline_response,
                pipeline=llm_pipeline,
                generate_response=generate_response,
                extract_answer=exp181.extract_final_number,
                max_repairs=MAX_REPAIRS,
            )

            case = {
                "sample_position": int(source_case["sample_position"]),
                "dataset_idx": int(source_case["dataset_idx"]),
                "question": question_text,
                "ground_truth": ground_truth,
                "baseline": baseline,
                "z3": copy.deepcopy(source_case["z3"]),
                "llm": {
                    "verify_only": llm_verify,
                    "verify_repair": llm_repair,
                },
            }
            cases.append(case)

            print(
                f"[{case['sample_position']:03d}/{SAMPLE_SIZE}] "
                f"dataset_idx={case['dataset_idx']} "
                f"baseline={baseline['correct']} "
                f"z3(flagged={case['z3']['verify_only']['flagged']}, "
                f"repair={case['z3']['verify_repair']['correct']}) "
                f"llm(flagged={llm_verify['flagged']}, repair={llm_repair['correct']})"
            )

    finally:
        if tokenizer is not None and model is not None:
            exp181.unload_model(model, tokenizer, gemma_config["device_index"])

    z3_summary = summarize_cases(cases, extractor_key="z3", seed=206)
    if z3_summary != source_z3_summary:
        raise RuntimeError(
            "Recomputed Z3 summary does not match the Exp 206 source artifact. "
            "The paired comparison is no longer trustworthy."
        )

    llm_summary = summarize_cases(cases, extractor_key="llm", seed=207)
    comparison = compare_named_extractors(
        llm_summary,
        z3_summary,
        primary_label="llm",
        secondary_label="z3",
    )

    payload = build_comparison_payload(
        experiment=207,
        title="LLM extractor on 100 live GSM8K (Gemma4-E4B-it) — head-to-head vs Z3",
        timestamp=timestamp,
        sample_seed=SAMPLE_SEED,
        sample_size=SAMPLE_SIZE,
        sample_dataset_indices=sample_dataset_indices,
        max_new_tokens=MAX_NEW_TOKENS,
        max_repairs=MAX_REPAIRS,
        runtime_seconds=time.perf_counter() - start,
        model_name=gemma_config["name"],
        hf_id=gemma_config["hf_id"],
        inference_mode="live_gpu",
        statistics={
            "llm": llm_summary,
            "z3": z3_summary,
        },
        comparison=comparison,
        cases=cases,
        extra_metadata={
            "baseline_source_experiment": 206,
            "baseline_source_path": "results/experiment_206_results.json",
            "baseline_source_timestamp": source_payload["metadata"]["timestamp"],
            "shared_baseline_responses": True,
            "extractor_max_new_tokens": EXTRACTOR_MAX_NEW_TOKENS,
            "pipeline_timeout_seconds": PIPELINE_TIMEOUT_SECONDS,
            "runtime_device_index": runtime_device_index,
            "preferred_device_index": int(base_gemma_config["device_index"]),
            "z3_summary_verified_from_source": True,
        },
    )

    out_path = RESULTS_DIR / "experiment_207_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\nLLM summary:")
    print(json.dumps(llm_summary, indent=2))
    print("\nZ3 summary:")
    print(json.dumps(z3_summary, indent=2))
    print("\nComparison:")
    print(json.dumps(comparison, indent=2))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
