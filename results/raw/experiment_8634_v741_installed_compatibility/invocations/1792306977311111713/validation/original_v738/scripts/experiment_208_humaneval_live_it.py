#!/usr/bin/env python3
"""Experiment 208: 30 HumanEval live benchmark on Gemma4-E4B-it.

Runs a deterministic 30-problem HumanEval sample through the code-focused
Carnot pipeline:
1. Generate code with live Gemma4-E4B-it inference on GPU.
2. Extract static code constraints via `CodeExtractor`.
3. Run Exp 53-style runtime instrumentation on deterministic probe inputs.
4. Execute the official HumanEval `check()` test harness.
5. If tests fail, run a verify-repair loop using the test failure plus the
   extracted/static/runtime feedback.

Usage:
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_208_humaneval_live_it.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003,
      REQ-CODE-008, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import gc
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

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "experiment_208_results.json"
CHECKPOINT_PATH = RESULTS_DIR / "exp208_ckpt.json"

try:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - script-only path
    print(f"FATAL: missing dependency: {exc}")
    raise SystemExit(1) from exc

from carnot.pipeline.humaneval_live_benchmark import (  # noqa: E402
    build_candidate_code,
    build_repair_prompt,
    build_results_payload,
    execute_humaneval,
    run_instrumentation,
    sample_problems,
    summarize_cases,
)

MODEL_NAME = "Gemma4-E4B-it"
MODEL_HF_ID = "google/gemma-4-E4B-it"
SAMPLE_SIZE = 30
SAMPLE_SEED = 208
MAX_NEW_TOKENS = 512
MAX_REPAIRS = 3
EXEC_TIMEOUT_SECONDS = 5.0
N_BOOTSTRAP = 10_000


def load_humaneval_problems() -> list[dict[str, Any]]:
    """Load HumanEval and annotate each row with its dataset index."""
    dataset = load_dataset("openai_humaneval", split="test")
    problems: list[dict[str, Any]] = []
    for dataset_idx in range(len(dataset)):
        row = dataset[dataset_idx]
        problems.append(
            {
                "dataset_idx": dataset_idx,
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        )
    return problems


def select_device_index() -> int:
    """Pick the CUDA device with the most free memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Exp 208.")

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
    return best_index


def load_model() -> tuple[Any, Any, str]:
    """Load Gemma4-E4B-it onto the selected GPU."""
    device_index = select_device_index()
    device_str = f"cuda:{device_index}"

    print(f"  Loading {MODEL_HF_ID} on {device_str}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": device_index},
    )
    model.eval()

    smoke = generate_raw(model, tokenizer, "Write Python: return 1", device_str, 16)
    print(f"  Smoke test OK: {smoke[:60].strip()!r}")
    return tokenizer, model, device_str


def unload_model(model: Any, tokenizer: Any, device_str: str) -> None:
    """Free GPU memory after the run."""
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Released model from {device_str}.")


def generate_raw(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device_str: str,
    max_new_tokens: int,
) -> str:
    """Generate text from a chat prompt on a specific CUDA device."""
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {key: value.to(device_str) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response


def generate_body(
    prompt: str,
    model: Any,
    tokenizer: Any,
    device_str: str,
) -> str:
    """Generate only the function body for a HumanEval prompt."""
    request = (
        "You are an expert Python programmer.\n"
        "Complete the following function.\n"
        "Return ONLY the function body lines. No def line. No markdown fences.\n"
        "Indent with 4 spaces.\n\n"
        f"{prompt}"
    )
    return generate_raw(
        model,
        tokenizer,
        request,
        device_str,
        MAX_NEW_TOKENS,
    )


def load_checkpoint(expected_task_ids: list[str]) -> dict[str, Any]:
    """Resume a prior Exp 208 run if the checkpoint matches the same sample."""
    if not CHECKPOINT_PATH.exists():
        return {"sample_task_ids": expected_task_ids, "cases": {}}

    payload = json.loads(CHECKPOINT_PATH.read_text())
    if payload.get("sample_task_ids") != expected_task_ids:
        return {"sample_task_ids": expected_task_ids, "cases": {}}
    return payload


def save_checkpoint(sample_task_ids: list[str], cases_by_task: dict[str, Any]) -> None:
    """Persist the partial Exp 208 run after each problem."""
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "sample_task_ids": sample_task_ids,
                "cases": cases_by_task,
            },
            indent=2,
        )
    )


def run_problem(
    problem: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device_str: str,
) -> dict[str, Any]:
    """Run baseline plus verify-repair for one HumanEval problem."""
    baseline_body = generate_body(problem["prompt"], model, tokenizer, device_str)
    baseline_code = build_candidate_code(problem["prompt"], baseline_body)
    instrumentation = run_instrumentation(
        baseline_code,
        problem["prompt"],
        problem["entry_point"],
        official_tests=problem["test"],
    )
    baseline_result = execute_humaneval(
        baseline_code,
        problem,
        timeout=EXEC_TIMEOUT_SECONDS,
    )

    case = {
        "task_id": problem["task_id"],
        "dataset_idx": problem["dataset_idx"],
        "entry_point": problem["entry_point"],
        "baseline": {
            "passed": baseline_result.passed,
            "error_type": baseline_result.error_type,
            "error_message": baseline_result.error_message,
            "n_static_violations": instrumentation["n_static_violations"],
            "n_dynamic_violations": instrumentation["n_dynamic_violations"],
            "n_property_violations": instrumentation["n_property_violations"],
            "constraint_feedback": instrumentation["constraint_feedback"],
            "static_violations": instrumentation["static_violations"],
            "dynamic_violations": instrumentation["dynamic_violations"],
            "property_violations": instrumentation["property_violations"],
            "probe_inputs": instrumentation["probe_inputs"],
            "body": baseline_body,
        },
        "verify_repair": {
            "passed": baseline_result.passed,
            "repaired": False,
            "n_repairs": 0,
            "final_error_type": baseline_result.error_type,
            "final_error_message": baseline_result.error_message,
            "final_body": baseline_body,
        },
        "iterations": [
            {
                "iteration": 0,
                "passed": baseline_result.passed,
                "error_type": baseline_result.error_type,
                "error_message": baseline_result.error_message,
                "n_static_violations": instrumentation["n_static_violations"],
                "n_dynamic_violations": instrumentation["n_dynamic_violations"],
                "n_property_violations": instrumentation["n_property_violations"],
            }
        ],
    }

    if baseline_result.passed:
        return case

    current_body = baseline_body
    current_result = baseline_result

    for repair_idx in range(MAX_REPAIRS):
        repair_prompt = build_repair_prompt(
            problem["prompt"],
            current_body,
            current_result,
            instrumentation,
            repair_idx=repair_idx,
        )
        current_body = generate_raw(
            model,
            tokenizer,
            repair_prompt,
            device_str,
            MAX_NEW_TOKENS,
        )
        current_code = build_candidate_code(problem["prompt"], current_body)
        instrumentation = run_instrumentation(
            current_code,
            problem["prompt"],
            problem["entry_point"],
            official_tests=problem["test"],
        )
        current_result = execute_humaneval(
            current_code,
            problem,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
        case["iterations"].append(
            {
                "iteration": repair_idx + 1,
                "passed": current_result.passed,
                "error_type": current_result.error_type,
                "error_message": current_result.error_message,
                "n_static_violations": instrumentation["n_static_violations"],
                "n_dynamic_violations": instrumentation["n_dynamic_violations"],
                "n_property_violations": instrumentation["n_property_violations"],
            }
        )
        case["verify_repair"]["n_repairs"] = repair_idx + 1
        case["verify_repair"]["passed"] = current_result.passed
        case["verify_repair"]["final_error_type"] = current_result.error_type
        case["verify_repair"]["final_error_message"] = current_result.error_message
        case["verify_repair"]["final_body"] = current_body
        if current_result.passed:
            case["verify_repair"]["repaired"] = True
            break

    return case


def main() -> int:
    start = time.perf_counter()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print("=" * 78)
    print("EXPERIMENT 208: HumanEval live verify-repair on Gemma4-E4B-it")
    print(f"  Cohort: {SAMPLE_SIZE} problems | sample_seed={SAMPLE_SEED}")
    print(
        "  Pipeline: generate -> CodeExtractor -> runtime instrumentation -> "
        "property verifier -> tests -> repair"
    )
    print("=" * 78)

    print("\n[1/4] Loading HumanEval...")
    problems = load_humaneval_problems()
    sampled = sample_problems(problems, SAMPLE_SIZE, SAMPLE_SEED)
    sample_task_ids = [problem["task_id"] for problem in sampled]
    sample_dataset_indices = [int(problem["dataset_idx"]) for problem in sampled]
    print(f"  Loaded {len(problems)} total problems; sampled {len(sampled)}.")

    print("\n[2/4] Loading live Gemma model...")
    tokenizer, model, device_str = load_model()

    print("\n[3/4] Running benchmark...")
    checkpoint = load_checkpoint(sample_task_ids)
    cases_by_task = dict(checkpoint.get("cases", {}))

    try:
        for problem in sampled:
            if problem["task_id"] in cases_by_task:
                continue

            problem_start = time.perf_counter()
            case = run_problem(problem, model, tokenizer, device_str)
            cases_by_task[problem["task_id"]] = case
            save_checkpoint(sample_task_ids, cases_by_task)

            completed = len(cases_by_task)
            baseline_correct = sum(
                1 for item in cases_by_task.values() if item["baseline"]["passed"]
            )
            repair_correct = sum(
                1 for item in cases_by_task.values() if item["verify_repair"]["passed"]
            )
            elapsed = time.perf_counter() - problem_start
            print(
                f"  [{completed:02d}/{SAMPLE_SIZE}] {problem['task_id']} "
                f"baseline={case['baseline']['passed']} "
                f"repair={case['verify_repair']['passed']} "
                f"repairs={case['verify_repair']['n_repairs']} "
                f"running baseline={baseline_correct / completed:.1%} "
                f"repair={repair_correct / completed:.1%} "
                f"[{elapsed:.1f}s]"
            )
    finally:
        unload_model(model, tokenizer, device_str)

    print("\n[4/4] Summarizing and saving artifact...")
    ordered_cases = [cases_by_task[task_id] for task_id in sample_task_ids]
    statistics = summarize_cases(
        ordered_cases,
        n_bootstrap=N_BOOTSTRAP,
        seed=SAMPLE_SEED,
    )
    statistics["inference_mode"] = "live_gpu"

    payload = build_results_payload(
        experiment=208,
        title="30 HumanEval live verify-repair benchmark on Gemma4-E4B-it",
        timestamp=timestamp,
        model_name=MODEL_NAME,
        hf_id=MODEL_HF_ID,
        device=device_str,
        inference_mode="live_gpu",
        sample_seed=SAMPLE_SEED,
        sample_size=SAMPLE_SIZE,
        sample_dataset_indices=sample_dataset_indices,
        sample_task_ids=sample_task_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        max_repairs=MAX_REPAIRS,
        runtime_seconds=time.perf_counter() - start,
        statistics=statistics,
        cases=ordered_cases,
    )
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"  Saved results to {OUTPUT_PATH}")
    print()
    print(
        f"  Baseline pass@1:      {statistics['baseline']['n_correct']}/{SAMPLE_SIZE} "
        f"({statistics['baseline']['pass_at_1']:.1%})"
    )
    print(
        f"  Verify-repair pass@1: {statistics['verify_repair']['n_correct']}/{SAMPLE_SIZE} "
        f"({statistics['verify_repair']['pass_at_1']:.1%})"
    )
    print(
        f"  Improvement:          {statistics['improvement']['delta']:+.1%} "
        f"[{statistics['improvement']['ci_lower']:+.1%}, "
        f"{statistics['improvement']['ci_upper']:+.1%}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
