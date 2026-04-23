#!/usr/bin/env python3
"""Exp 759: Live GPU 2-Round Code Repair — HumanEval signed improvement.

**Researcher summary (arXiv 2604.10508):**
    "How Many Tries Does It Take?" shows self-repair universally improves
    HumanEval pass@1 by +4.9 to +17.1pp with most gains in the FIRST TWO
    ROUNDS.  This experiment executes the Exp 744 harness on a live RTX 3090
    with Qwen3.5-0.8B and reports the signed_improvement:
        pass_at_1_round2 - pass_at_1_round1

    Where:
    - pass_at_1_round1: fraction of 50 problems passing in the INITIAL generation
    - pass_at_1_round2: cumulative fraction passing after at most ONE repair round

**Why signed_improvement instead of Exp 744's total_improvement?**
    Exp 744 tracks three cumulative rounds (round0, round1, round2).  Exp 759
    focuses on the single most-impactful gap: single-shot generation vs one-repair.
    This maps directly to arXiv 2604.10508's main finding (round 1 >> round 2+).

**Why execution-based (not regex)?**
    The paper's gains rely on feeding real execution errors back to the model.
    Carnot's CodeExtractor already exercises actual Python execution, so we
    extend that path rather than adding a fragile regex extraction layer.

**honest_verdict logic:**
    - "code_repair_positive"   if signed_improvement > 0 AND inference_mode="live_gpu"
    - "code_repair_zero"       if signed_improvement = 0 AND inference_mode="live_gpu"
    - "code_repair_negative"   if signed_improvement < 0 (unexpected)
    - "blocked_no_live_gpu"    if CARNOT_FORCE_LIVE not set

**Gate:** CARNOT_FORCE_LIVE=1 required (GPU needed for live Qwen inference).

Spec: REQ-REPAIR-020, REQ-REPAIR-021, SCENARIO-REPAIR-040, SCENARIO-REPAIR-041,
      REQ-CODE-031, REQ-CODE-032
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU JAX — we use JAX only for EBM ops; inference runs on CUDA via torch.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.two_round_repair import TwoRoundCodeRepairPipeline, TwoRoundResult  # noqa: E402

# Reuse the validated 50-problem HumanEval subset from Exp 744 — no need to duplicate
# 462 lines of problem definitions that have already been checked for correctness.
from scripts.experiment_744_iterative_2round_repair import _HUMANEVAL_SUBSET  # noqa: E402

EXP_ID = 759
TITLE = "Live GPU 2-Round Code Repair — HumanEval Signed Improvement (arXiv 2604.10508)"
DELIVERABLE = "results/experiment_759_iterative_code_repair_live.json"
N_PROBLEMS = 50
FORCE_LIVE = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"


# ---------------------------------------------------------------------------
# Pure helper functions (unit-testable; no GPU imports)
# ---------------------------------------------------------------------------


def build_repair_prompt_759(
    original_problem: str,
    failed_code: str,
    traceback_str: str,
    test_case_call: str,
    expected_output: str,
    actual_output: str,
) -> str:
    """Build a repair prompt satisfying REQ-REPAIR-020: includes full traceback AND test case input.

    The prompt differs from the initial generation prompt by adding:
    1. The failing code — the model needs to see what it wrote.
    2. The FULL traceback — the primary error signal (REQ-REPAIR-020).
    3. The failing test case call expression — the explicit input that failed (REQ-REPAIR-020).
    4. Expected vs actual output — the correctness gap.

    This directly implements arXiv 2604.10508's finding that error message quality
    is the primary driver of self-repair gains.  A prompt that omits the traceback
    or test input gives the model insufficient signal to fix the bug.

    Args:
        original_problem: The original HumanEval problem docstring + signature.
        failed_code: The Python code that failed execution.
        traceback_str: Full traceback from the execution failure.
        test_case_call: The exact Python expression that returned wrong output.
        expected_output: String representation of the expected return value.
        actual_output: String representation of the actual return value.

    Returns:
        Formatted repair prompt string ready for the LLM caller.

    Spec: REQ-REPAIR-020, SCENARIO-REPAIR-040
    """
    parts = [
        "You are an expert Python programmer.  The code below has a bug.",
        "",
        "## Original Problem",
        original_problem.strip(),
        "",
        "## Failing Code",
        "```python",
        failed_code.strip(),
        "```",
        "",
        "## Execution Error",
        traceback_str.strip() if traceback_str.strip() else "(no traceback — wrong output)",
        "",
        "## Failing Test Case",
        f"Call: {test_case_call}",
        "",
        "## Expected Output",
        expected_output if expected_output else "<unknown>",
        "",
        "## Actual Output",
        actual_output if actual_output else "<no return value>",
        "",
        "Fix the bug in the code above.  Return ONLY the corrected function "
        "definition with no extra explanation.",
    ]
    return "\n".join(parts)


def compute_pass_at_1(results: list[TwoRoundResult]) -> tuple[float, float]:
    """Compute pass_at_1 for round1 (initial) and round2 (cumulative after one repair).

    pass_at_1_round1: fraction of problems where the INITIAL generation passed.
    pass_at_1_round2: fraction where initial OR one repair pass (cumulative).

    Calling these "round1" and "round2" aligns with the experiment task spec,
    which uses 1-indexed rounds (round1 = first attempt, round2 = after first repair).
    In TwoRoundResult terminology: round0_pass → "round1", round1_pass → "round2".

    Args:
        results: List of TwoRoundResult, one per problem.

    Returns:
        Tuple (pass_at_1_round1, pass_at_1_round2) as fractions in [0.0, 1.0].

    Spec: REQ-CODE-032
    """
    n = len(results)
    if n == 0:
        return (0.0, 0.0)
    # round1 = initial generation only (TwoRoundResult.round0_pass)
    r1 = sum(1 for r in results if r.round0_pass) / n
    # round2 = cumulative: passed in round0 OR round1 (one repair round)
    r2 = sum(1 for r in results if r.round0_pass or r.round1_pass) / n
    return (round(r1, 4), round(r2, 4))


def compute_signed_improvement(pass_at_1_round1: float, pass_at_1_round2: float) -> float:
    """Compute signed improvement from single-round to 2-round cumulative pass@1.

    signed_improvement = pass_at_1_round2 - pass_at_1_round1

    A positive value confirms the arXiv 2604.10508 finding: one repair round
    improves pass rate.  A zero result means no problems were repaired (the
    model either passed all on the first try, or repair never helped).

    Args:
        pass_at_1_round1: Fraction of problems passing in initial generation.
        pass_at_1_round2: Cumulative fraction passing after at most one repair.

    Returns:
        Signed difference (float), negative if repair made things worse (unexpected).

    Spec: REQ-CODE-032
    """
    return round(pass_at_1_round2 - pass_at_1_round1, 4)


def classify_honest_verdict(signed_improvement: float, inference_mode: str) -> str:
    """Map signed_improvement + inference_mode to an honest_verdict label.

    Verdict ladder (REQ-REPAIR-020, REQ-REPAIR-021):
    - "blocked_no_live_gpu"  — CARNOT_FORCE_LIVE not set; no inference ran.
    - "code_repair_positive" — signed_improvement > 0, live GPU confirmed.
    - "code_repair_zero"     — signed_improvement = 0, live GPU ran but no gain.
    - "code_repair_negative" — signed_improvement < 0 (unexpected; repair hurt).

    Args:
        signed_improvement: pass_at_1_round2 - pass_at_1_round1.
        inference_mode: "live_gpu" when CARNOT_FORCE_LIVE=1 ran, "blocked" otherwise.

    Returns:
        One of the four verdict strings above.

    Spec: REQ-REPAIR-020, REQ-REPAIR-021, SCENARIO-REPAIR-041
    """
    if inference_mode == "blocked":
        return "blocked_no_live_gpu"
    if signed_improvement > 0:
        return "code_repair_positive"
    if signed_improvement < 0:
        return "code_repair_negative"
    return "code_repair_zero"


# ---------------------------------------------------------------------------
# LLM caller (live mode)
# ---------------------------------------------------------------------------


def _build_qwen_caller(model_name: str = "Qwen/Qwen3.5-0.8B", gpu_id: int = 0):
    """Build a Qwen3.5-0.8B callable for live GPU inference on the given device.

    Loads the model once and returns a stateful closure so that
    BatchedInferenceRunner can reuse the same loaded weights across all batches.
    Kept at float16 to fit in RTX 3090 VRAM alongside EBM overhead.

    Args:
        model_name: HuggingFace model ID.
        gpu_id: CUDA device index (0 or 1 on dual-3090 host).

    Returns:
        Callable(prompt: str) -> str.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: PLC0415

    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=False,
    )
    model.eval()

    def _call(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    return _call


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 759: Live 2-round code repair benchmark on 50 HumanEval problems."""
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()
    tmpl.check_exclusion_manifest()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=90, result_path=DELIVERABLE):

        # Guard: CARNOT_FORCE_LIVE=1 required — REQ-REPAIR-021
        if not FORCE_LIVE:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_no_live_gpu",
                    "blocked_reason": (
                        "CARNOT_FORCE_LIVE=1 not set — live GPU required for Qwen inference"
                    ),
                    "inference_mode": "blocked",
                    "model_id": "Qwen/Qwen3.5-0.8B",
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # GPU setup
        MODEL_SPECS = [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0}]
        gpu_status = tmpl.setup_gpu(MODEL_SPECS)
        if not gpu_status["all_healthy"]:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_no_live_gpu",
                    "blocked_reason": "GPU pre-warm failed — cannot run live inference",
                    "inference_mode": "blocked",
                    "model_id": "Qwen/Qwen3.5-0.8B",
                    "n_problems": 0,
                    "pass_at_1_round1": 0.0,
                    "pass_at_1_round2": 0.0,
                    "signed_improvement": 0.0,
                    "n_repaired": 0,
                },
                status="blocked",
            )
            out = Path(_REPO) / DELIVERABLE
            out.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        llm_caller = _build_qwen_caller("Qwen/Qwen3.5-0.8B", gpu_id=0)
        pipeline = TwoRoundCodeRepairPipeline()
        all_results: list[TwoRoundResult] = []

        # Resume from checkpoint if available.
        checkpoint = tmpl.checkpoint
        start_idx = 0
        if checkpoint and "results" in checkpoint:
            raw = checkpoint["results"].get("results", [])
            for r in raw:
                all_results.append(TwoRoundResult(**r))
            start_idx = len(all_results)

        problems_to_run = _HUMANEVAL_SUBSET[start_idx:]
        batch_log: list[dict] = []

        # Run problems sequentially in the main thread.
        # Why not BatchedInferenceRunner: TwoRoundCodeRepairPipeline.execute() uses
        # SIGALRM for timeouts, which only works in the main thread.  Threading through
        # BatchedInferenceRunner raises ValueError("signal only works in main thread").
        import time as _time  # noqa: PLC0415
        for i, problem_dict in enumerate(problems_to_run):
            t0 = _time.perf_counter()
            try:
                result = pipeline.run(
                    problem=problem_dict["prompt"],
                    test_cases=problem_dict["test_cases"],
                    llm_caller=llm_caller,
                )
            except Exception:
                result = TwoRoundResult(
                    round0_pass=False, round1_pass=False, round2_pass=False,
                    round0_code="", round1_code="", round2_code="",
                    error_types=["other"],
                )
            elapsed = round(_time.perf_counter() - t0, 3)
            batch_log.append({"problem_idx": start_idx + i, "time_s": elapsed})
            all_results.append(result)

            if len(all_results) % 10 == 0:
                tmpl.checkpoint_save(
                    {"results": [vars(r) for r in all_results]},
                    step=len(all_results),
                )

        pass_at_1_round1, pass_at_1_round2 = compute_pass_at_1(all_results)
        signed_improvement = compute_signed_improvement(pass_at_1_round1, pass_at_1_round2)
        n_repaired = sum(
            1 for r in all_results if not r.round0_pass and r.round1_pass
        )
        verdict = classify_honest_verdict(signed_improvement, "live_gpu")

        artifact = tmpl.build_result(
            {
                "honest_verdict": verdict,
                "model_id": "Qwen/Qwen3.5-0.8B",
                "n_problems": len(all_results),
                "pass_at_1_round1": pass_at_1_round1,
                "pass_at_1_round2": pass_at_1_round2,
                "signed_improvement": signed_improvement,
                "n_repaired": n_repaired,
                "inference_mode": "live_gpu",
                "batch_log": batch_log,
                "arxiv_ref": "2604.10508",
            },
            status="success",
            decision_class="repair",
        )
        out = Path(_REPO) / DELIVERABLE
        out.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
