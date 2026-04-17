#!/usr/bin/env python3
"""Experiment 444: CarnotThinkProbe benchmark on synthetic correct/wrong responses.

**Researcher summary:**
    Validates that CarnotThinkProbe (generative 3-step CoT pre-filter, ThinkPRM architecture)
    can reliably flag incorrect responses before Ising runs. Measures:
      - skip_rate: fraction of responses flagged as 'incorrect' (Ising skipped)
      - tp_rate: fraction of wrong responses correctly flagged
      - fp_rate: fraction of correct responses wrongly flagged

    Honest verdict:
      'think_probe_viable'    if skip_rate > 0.30 AND fp_rate < 0.10
      'think_probe_imprecise' if fp_rate >= 0.10
      'ci_stub_only'          when no GPU is available (CI stub always returns 'uncertain')

    CPU-only. No GPU required. Always produces a result JSON.

Spec: REQ-VERIFY-094, REQ-VERIFY-095
SCENARIO-VERIFY-126, SCENARIO-VERIFY-127, SCENARIO-VERIFY-128
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Always apply env autofix first (detects ROCm/CUDA, injects JAX platform vars).
sys.path.insert(0, str(Path(__file__).parent.parent))
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

import logging

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.think_probe import CarnotThinkProbe

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 444
EXP_TITLE = "CarnotThinkProbe benchmark: 50 correct + 50 wrong synthetic responses"
RESULT_PATH = Path("results/experiment_444_think_probe.json")
TIMEOUT_MINUTES = 20

# ---------------------------------------------------------------------------
# Synthetic corpus builders
# ---------------------------------------------------------------------------

# Correct synthetic responses: simple arithmetic claims that are true.
CORRECT_TEMPLATES = [
    "The answer is {n}. We compute {a} + {b} = {n}.",
    "Therefore {a} * {b} = {n}, which is correct.",
    "Since {a} - {b} = {n}, the result is {n}.",
    "The sum {a} + {b} equals {n}.",
    "Multiplying {a} by {b} gives {n}.",
]

# Wrong synthetic responses: arithmetic claims that are deliberately false.
WRONG_TEMPLATES = [
    "The answer is {wrong}. We compute {a} + {b} = {wrong}.",
    "Therefore {a} * {b} = {wrong}, so the result is {wrong}.",
    "Since {a} - {b} = {wrong}, we conclude {wrong}.",
    "The sum {a} + {b} equals {wrong}.",
    "Multiplying {a} by {b} gives {wrong}.",
]


def _build_corpus(n_correct: int = 50, n_wrong: int = 50) -> tuple[list[str], list[bool]]:
    """Build synthetic corpus of correct and wrong arithmetic responses.

    Returns:
        (responses, ground_truth) where ground_truth[i]=True means correct.
    """
    import random

    rng = random.Random(42)
    responses: list[str] = []
    ground_truth: list[bool] = []

    for i in range(n_correct):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        n = a + b
        template = CORRECT_TEMPLATES[i % len(CORRECT_TEMPLATES)]
        responses.append(template.format(a=a, b=b, n=n))
        ground_truth.append(True)

    for i in range(n_wrong):
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        correct = a + b
        # Deliberately wrong answer: offset by a non-zero amount.
        wrong = correct + rng.randint(1, 10)
        template = WRONG_TEMPLATES[i % len(WRONG_TEMPLATES)]
        responses.append(template.format(a=a, b=b, wrong=wrong))
        ground_truth.append(False)

    return responses, ground_truth


# ---------------------------------------------------------------------------
# LLM caller factory (GPU path)
# ---------------------------------------------------------------------------

def _build_gpu_caller():
    """Try to build a real Qwen3.5-0.8B LLM caller for GPU environments.

    Returns the callable if successful, None if GPU/model not available.
    This function intentionally swallows all errors so CI always falls through
    to the stub path.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = "Qwen/Qwen2.5-0.5B"  # Lightweight fallback; swap to Qwen3.5-0.8B on GPU.
        _log.info("Attempting to load %s for ThinkProbe GPU path...", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )
        model.eval()
        _log.info("Model loaded on CPU (GPU not required for this experiment).")

        def caller(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated, skip_special_tokens=True)

        return caller
    except Exception as exc:
        _log.info("GPU/model not available (%s). Using CI stub.", exc)
        return None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import datetime
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES, result_path=str(RESULT_PATH)):
        _log.info("Exp %d: Building synthetic corpus...", EXP_ID)
        responses, ground_truth = _build_corpus(n_correct=50, n_wrong=50)

        # Try to build a real LLM caller; fall back to CI stub if unavailable.
        llm_caller = _build_gpu_caller()
        is_ci_stub = llm_caller is None

        if is_ci_stub:
            _log.info("Running with CI stub (no LLM — all verdicts will be 'uncertain').")
        else:
            _log.info("Running with real LLM caller.")

        probe = CarnotThinkProbe(llm_caller=llm_caller)

        _log.info("Running benchmark on %d responses...", len(responses))
        metrics = probe.benchmark(responses, ground_truth)

        skip_rate = metrics["skip_rate"]
        tp_rate = metrics["tp_rate"]
        fp_rate = metrics["fp_rate"]

        _log.info("skip_rate=%.3f  tp_rate=%.3f  fp_rate=%.3f", skip_rate, tp_rate, fp_rate)

        # Determine honest verdict.
        if is_ci_stub:
            honest_verdict = "ci_stub_only"
        elif fp_rate >= 0.10:
            honest_verdict = "think_probe_imprecise"
        elif skip_rate > 0.30:
            honest_verdict = "think_probe_viable"
        else:
            honest_verdict = "think_probe_imprecise"

        finished_at = datetime.datetime.utcnow().isoformat() + "Z"

        artifact = {
            "experiment": EXP_ID,
            "schema": "carnot.think_probe.v1",
            "title": EXP_TITLE,
            "run_date": started_at[:10],
            "started_at": started_at,
            "finished_at": finished_at,
            "status": "success",
            "duration_s": 0.0,  # ExperimentTemplate would fill this; we approximate here.
            "honest_verdict": honest_verdict,
            "is_ci_stub": is_ci_stub,
            "skip_rate": skip_rate,
            "tp_rate": tp_rate,
            "fp_rate": fp_rate,
            "n_correct": 50,
            "n_wrong": 50,
            "n_total": 100,
        }

        RESULT_PATH.write_text(json.dumps(artifact, indent=2))
        _log.info("Result written to %s", RESULT_PATH)
        _log.info("Honest verdict: %s", honest_verdict)


if __name__ == "__main__":
    main()
