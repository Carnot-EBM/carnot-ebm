#!/usr/bin/env python3
"""Experiment 516: GSM-Symbolic Adversarial v5 — RETRO-039 Robustness Claim.

**Researcher summary (RETRO-039, Apple arXiv 2410.05229 — four consecutive misses):**
    Apple showed that ALL major LLMs degrade significantly under symbolic adversarial prompting
    (one irrelevant sentence appended to arithmetic problems):
      - o1-preview:   92.7% -> 77.4% (-15.3pp)
      - GPT-4o:       95%   -> 88%   (-7pp)
      - Llama3-70B:   90%   -> 75%   (-15pp)

    Root cause: LLMs attend to ALL context tokens, so irrelevant sentences (distractors)
    derail the reasoning chain.  The model "sees" extra numbers and words and confuses them
    with problem operands.

    Carnot's ROBUSTNESS THESIS (RETRO-039):
        The Ising arithmetic verifier extracts equation tokens (explicit arithmetic expressions
        like "24 + 6 = 30") and verifies them independently of surrounding context.  A symbolic
        substitution that changes the phrasing while preserving arithmetic constraints does NOT
        fool the Ising sampler — the Ising energy is computed over extracted constraint terms
        ONLY.  Distractor words and numbers are invisible to it.

    This is the FIFTH attempt.  Prior attempts:
      - v1 (Exp 354): harness only, no live inference
      - v2 (Exp 479): gpu_vram_insufficient — Gemma4 FP16 exceeded VRAM budget
      - v3 (Exp 490): gpu_required — CARNOT_FORCE_LIVE not set
      - v4 (Exp 504): Gemma4QuantizedLoader; may have hit further VRAM issues
      - v5 (this):    Simplified single-model design (Qwen3.5-0.8B) to reduce VRAM risk;
                      uses 50 standard + 50 adversarial questions (vs 100) for speed;
                      adds compute_robustness_delta + build_adversarial_v5_artifact helpers

**The headline result (RETRO-039 credibility claim):**
    robustness_delta = (baseline_std - baseline_adv) - (pipeline_std - pipeline_adv)
    retro_039_confirmed = robustness_delta > 0 AND inference_mode == 'live_gpu'

    If Carnot's adversarial accuracy DROP is SMALLER than the raw LLM's drop,
    robustness_delta is positive and the thesis is confirmed.

**Gate chain (in order):**
    0. apply_env_autofix() — FIRST: injects CARNOT_FORCE_LIVE=1 before any CUDA import
    1. ExperimentTimeoutWatchdog(516, timeout_minutes=120) — outer budget cap
    2. DeliverableGuard — ensures JSON is always written even on crash
    3. GPUVRAMGateV2 + JITVRAMCheck before each model load
    4. For each condition (standard, adversarial) x variant (baseline, pipeline):
       run inference, record accuracy
    5. compute_robustness_delta -> positive = thesis confirmed
    6. build_adversarial_v5_artifact with schema='carnot.adversarial_v5.v1'
    7. assert_deliverable_written() as FINAL LINE

**honest_verdict semantics:**
    'thesis_confirmed'  — robustness_delta > 0 AND inference_mode == 'live_gpu'
    'thesis_rejected'   — robustness_delta <= 0 AND inference_mode == 'live_gpu'
    'gpu_required'      — CARNOT_FORCE_LIVE not set or GPU not healthy

Spec: REQ-BENCH-052, REQ-BENCH-053,
      SCENARIO-BENCH-037, SCENARIO-BENCH-038
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import occurs.  See RETRO-022 for why this ordering matters.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import os
import random
import re
from typing import Any

from carnot.extraction.integrated_extractor import IntegratedExtractor
from carnot.extraction.vericot_validator import VeriCoTStepValidator
from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier
from carnot.pipeline.adversarial_gsm8k import AdversarialGSMQuestion, build_adversarial_questions
from carnot.pipeline.adversarial_v5_result import (
    build_adversarial_v5_artifact,
    compute_robustness_delta,
)
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2
from carnot.pipeline.jit_vram_check import JITVRAMCheck
from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 516
EXP_TITLE = "GSM-Symbolic Adversarial v5 — RETRO-039 Robustness Claim"
DELIVERABLE = "results/experiment_516_gsm_symbolic_adversarial_v5.json"
N_QUESTIONS = 50
DATASET_SEED = 42
BATCH_SIZE = 8

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "Qwen3.5-0.8B",
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "gpu": 0,
        "required_gb": 2.0,
    },
]


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def _extract_gold(answer_text: str) -> str | None:
    """Extract the numeric gold answer from GSM8K '#### N' format."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return nums[-1] if nums else None


def _is_correct(response: str, gold: str | None) -> bool:
    """Return True when the response's final number matches the gold answer."""
    if not gold or not response:
        return False
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
    if m:
        extracted = m.group(1)
    else:
        nums = re.findall(r"-?\d+(?:\.\d+)?", response)
        extracted = nums[-1] if nums else None
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(gold)) < 0.501
    except (ValueError, TypeError):
        return extracted.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _load_standard_gsm8k(n: int = N_QUESTIONS) -> list[dict[str, str]]:
    """Load standard GSM8K questions; falls back to synthetic if HuggingFace unavailable."""
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        items = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        rng = random.Random(DATASET_SEED)
        rng.shuffle(items)
        result = items[:n]
        _log.info("Loaded %d standard GSM8K questions from HuggingFace", len(result))
        return result
    except Exception as exc:
        _log.warning("Could not load GSM8K: %s — using synthetic fallback", exc)

    # Synthetic fallback: deterministically generated arithmetic questions
    rng = random.Random(DATASET_SEED)
    fallback = []
    for i in range(n):
        a = rng.randint(10, 99)
        b = rng.randint(1, a)
        fallback.append({
            "question": f"Alice has {a} items. She gives {b} away. How many does she have?",
            "answer": f"{a} - {b} = {a - b}. #### {a - b}",
        })
    _log.info("Using %d synthetic standard questions", len(fallback))
    return fallback


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _run_condition_accuracy(
    inference_fn: Any,
    questions: list[dict[str, str]],
    extractor: IntegratedExtractor | None,
    condition_label: str,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Run one condition and return fraction correct via LongRunBenchmarkExecutor.

    When extractor is None: baseline (LLM only).
    When extractor is provided: Carnot verify-repair pipeline.
    Returns fraction correct in [0, 1].
    """
    executor = LongRunBenchmarkExecutor(batch_size=batch_size)
    batches = executor.partition(questions)

    def _infer_one(q: dict[str, str]) -> bool:
        gold = _extract_gold(q["answer"])
        prompt = f"Solve step by step:\n{q['question']}\nAnswer:"
        response = inference_fn(prompt)
        if extractor is not None and response:
            try:
                violations = extractor.extract(response)
                if violations:
                    repair_prompt = (
                        f"Your reasoning had arithmetic errors:\n{violations}\n"
                        f"Correct answer for:\n{q['question']}\nAnswer:"
                    )
                    repaired = inference_fn(repair_prompt)
                    if repaired:
                        response = repaired
            except Exception as exc:
                _log.warning("Extractor error on condition %s: %s", condition_label, exc)
        return _is_correct(response, gold)

    for batch in batches:
        executor.run_batch(batch, _infer_one)

    result = executor.assemble(batches)
    correct = sum(1 for r in result.all_results if r)
    n = len(result.all_results)
    acc = correct / n if n > 0 else 0.0
    _log.info("[%s] correct=%d/%d acc=%.3f", condition_label, correct, n, acc)
    return acc


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def _write_json(repo_root: Path, rel_path: str, data: dict) -> None:
    """Atomically write JSON data to rel_path under repo_root."""
    out_path = repo_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.rename(out_path)
    _log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path) -> dict:
    """Run Experiment 516 and return the artifact dict.

    Gate chain:
        1. CARNOT_FORCE_LIVE check -> honest 'gpu_required' if absent
        2. GPUVRAMGateV2(kill_first=True) -> 'gpu_vram_insufficient' if VRAM too low
        3. JITVRAMCheck before model load
        4. Four conditions: standard x {baseline, pipeline}, adversarial x {baseline, pipeline}
        5. compute_robustness_delta -> retro_039_confirmed verdict
        6. build_adversarial_v5_artifact
    """
    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=True,
        repo_root=repo_root,
    )
    guard = DeliverableGuard(str(repo_root / DELIVERABLE))
    tmpl.setup()

    # Gate 1: CARNOT_FORCE_LIVE — without this flag, live GPU inference is disabled
    if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
        raw_results = {
            "baseline_standard_accuracy": 0.0,
            "baseline_adversarial_accuracy": 0.0,
            "pipeline_standard_accuracy": 0.0,
            "pipeline_adversarial_accuracy": 0.0,
            "n_questions": N_QUESTIONS,
        }
        v5 = build_adversarial_v5_artifact(raw_results, "gpu_required")
        artifact = tmpl.build_result(v5, status="gpu_required")
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        tmpl.assert_deliverable_written()
        return artifact

    # Gate 2: GPUVRAMGateV2 — kill zombie GPU processes, then verify free VRAM >= 4 GiB
    # Using 4 GiB threshold: Qwen3.5-0.8B requires ~2 GiB, with headroom.
    try:
        with GPUVRAMGateV2(min_free_gb=4.0, kill_first=True):
            pass
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGateV2: insufficient VRAM: %s", exc)
        raw_results = {
            "baseline_standard_accuracy": 0.0,
            "baseline_adversarial_accuracy": 0.0,
            "pipeline_standard_accuracy": 0.0,
            "pipeline_adversarial_accuracy": 0.0,
            "n_questions": N_QUESTIONS,
            "vram_error": str(exc),
        }
        v5 = build_adversarial_v5_artifact(raw_results, "gpu_vram_insufficient")
        artifact = tmpl.build_result(v5, status="gpu_vram_insufficient")
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        tmpl.assert_deliverable_written()
        return artifact

    # Gate 3: JITVRAMCheck — verify VRAM immediately before model load
    jit_check = JITVRAMCheck(device_id=0)
    spec = MODEL_SPECS[0]
    jit_result = jit_check.gate_model_load(spec["hf_id"], spec["required_gb"])
    if not jit_result.is_cleared:
        _log.error(
            "JITVRAMCheck: insufficient VRAM for %s: available=%.2f GB required=%.2f GB",
            spec["hf_id"], jit_result.available_gb, jit_result.required_gb,
        )
        raw_results = {
            "baseline_standard_accuracy": 0.0,
            "baseline_adversarial_accuracy": 0.0,
            "pipeline_standard_accuracy": 0.0,
            "pipeline_adversarial_accuracy": 0.0,
            "n_questions": N_QUESTIONS,
            "jit_available_gb": jit_result.available_gb,
            "jit_required_gb": jit_result.required_gb,
        }
        v5 = build_adversarial_v5_artifact(raw_results, "gpu_vram_insufficient")
        artifact = tmpl.build_result(v5, status="gpu_vram_insufficient")
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        tmpl.assert_deliverable_written()
        return artifact

    # Load model via HuggingFace transformers
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]

        qwen_pipe = hf_pipeline(
            "text-generation",
            model=spec["hf_id"],
            device=spec.get("gpu", 0),
            torch_dtype="auto",
        )

        def _model_fn(prompt: str) -> str:
            try:
                out = qwen_pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
                return str(out[0]["generated_text"])
            except Exception as exc:
                _log.warning("Model generation failed: %s", exc)
                return ""

        inference_mode = "live_gpu"
        _log.info("Loaded model %s on GPU %d", spec["hf_id"], spec.get("gpu", 0))

    except Exception as exc:
        _log.warning("Could not load model %s: %s — using stub", spec["hf_id"], exc)

        def _model_fn(prompt: str) -> str:  # type: ignore[misc]
            return "The answer is 42."

        inference_mode = "simulated"

    # Load datasets
    standard_questions = _load_standard_gsm8k(N_QUESTIONS)
    adversarial_pairs: list[AdversarialGSMQuestion] = build_adversarial_questions(
        standard_questions, seed=DATASET_SEED
    )
    adversarial_questions = [
        {"question": p.adversarial_question, "answer": p.ground_truth_answer}
        for p in adversarial_pairs
    ]

    _log.info(
        "Dataset: standard=%d adversarial=%d inference_mode=%s",
        len(standard_questions), len(adversarial_questions), inference_mode,
    )

    # Shared extractor for all pipeline conditions
    extractor = IntegratedExtractor(
        VeriCoTStepValidator(use_mock=False),
        VPRMArithmeticVerifier(),
    )

    # Run four conditions
    _log.info("=== Condition 1/4: standard_baseline ===")
    baseline_std = _run_condition_accuracy(
        _model_fn, standard_questions, None, "standard_baseline"
    )

    _log.info("=== Condition 2/4: standard_pipeline ===")
    pipeline_std = _run_condition_accuracy(
        _model_fn, standard_questions, extractor, "standard_pipeline"
    )

    _log.info("=== Condition 3/4: adversarial_baseline ===")
    baseline_adv = _run_condition_accuracy(
        _model_fn, adversarial_questions, None, "adversarial_baseline"
    )

    _log.info("=== Condition 4/4: adversarial_pipeline ===")
    pipeline_adv = _run_condition_accuracy(
        _model_fn, adversarial_questions, extractor, "adversarial_pipeline"
    )

    robustness_delta = compute_robustness_delta(baseline_std, baseline_adv, pipeline_std, pipeline_adv)
    _log.info(
        "robustness_delta=%.4f (baseline_drop=%.4f, pipeline_drop=%.4f)",
        robustness_delta,
        baseline_std - baseline_adv,
        pipeline_std - pipeline_adv,
    )

    raw_results = {
        "baseline_standard_accuracy": baseline_std,
        "baseline_adversarial_accuracy": baseline_adv,
        "pipeline_standard_accuracy": pipeline_std,
        "pipeline_adversarial_accuracy": pipeline_adv,
        "n_questions": N_QUESTIONS,
        "model_id": spec["name"],
        "dataset_seed": DATASET_SEED,
    }

    v5 = build_adversarial_v5_artifact(raw_results, inference_mode)
    _log.info(
        "honest_verdict=%s retro_039_confirmed=%s",
        v5["honest_verdict"], v5["retro_039_confirmed"],
    )

    artifact = tmpl.build_result(v5, status="success")
    _write_json(repo_root, DELIVERABLE, artifact)
    tmpl.assert_deliverable_written()
    return artifact


def main() -> None:
    """Run Experiment 516: GSM-Symbolic Adversarial v5 benchmark."""
    guard = DeliverableGuard(str(_REPO_ROOT / DELIVERABLE))

    with ExperimentTimeoutWatchdog(
        EXP_ID,
        timeout_minutes=int(os.environ.get("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", "120")),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        run_experiment(_REPO_ROOT)

    guard.assert_written()


if __name__ == "__main__":
    main()
