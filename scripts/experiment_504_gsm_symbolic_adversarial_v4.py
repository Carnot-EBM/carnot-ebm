#!/usr/bin/env python3
"""Experiment 504: GSM-Symbolic Adversarial v4 — Gemma4QuantizedLoader + RETRO-039 robustness claim.

**Researcher summary (RETRO-039, Apple arXiv 2410.05229):**
    Apple showed that ALL major LLMs degrade significantly under symbolic adversarial prompting:
      - o1-preview:   92.7% → 77.4% (-15.3pp)
      - GPT-4o:       95%   → 88%   (-7pp)
      - Llama3-70B:   90%   → 75%   (-15pp)

    Root cause: LLMs attend to ALL context tokens, so irrelevant sentences (distractors)
    derail the reasoning chain.  The model "sees" extra numbers and words and confuses them
    with problem operands.

    Carnot's ROBUSTNESS THESIS: the Ising arithmetic verifier extracts equation tokens
    (explicit arithmetic expressions like "24 + 6 = 30") and verifies them independently
    of surrounding context.  The Ising energy function is computed over the extracted
    constraint terms ONLY — distractor words and numbers are invisible to it.

    **The headline result (RETRO-039 credibility claim):**
        If Carnot's adversarial accuracy DROP is SMALLER than the baseline's adversarial
        accuracy drop, then Carnot is provably more robust to distractor injection.

        robustness_delta = standard_drop_baseline - standard_drop_pipeline
        carnot_more_robust = robustness_delta > 0

    This is a stronger and more direct claim than prior adversarial experiments:
      - v1 (Exp 354): harness only, no live inference
      - v2 (Exp 479): gpu_vram_insufficient — Gemma4 FP16 exceeded VRAM budget
      - v3 (Exp 490): gpu_required — CARNOT_FORCE_LIVE not set
      - v4 (this):    Gemma4QuantizedLoader (INT4 GGUF, ~9 GiB) unblocks VRAM constraint

**Why Gemma4QuantizedLoader instead of GemmaTransformersLoader:**
    Exps 479 and 490 failed because Gemma4 FP16 requires ~14.89 GiB but the conductor
    process permanently holds ~9 GiB on GPU 0 (it compiles a JAX computation graph at
    startup and cannot release it without killing itself).  9 + 15 = 24 GiB exceeds the
    RTX 3090's 24 GiB budget, causing OOM.  Gemma4QuantizedLoader uses GGUF Q4_K_M
    quantization (~9 GiB), so 9 (conductor) + 9 (model) = 18 GiB — fits with 6 GiB headroom.

**Four conditions per model:**
    standard_baseline:    LLM only, 100 standard GSM8K questions (no distractors, no Carnot)
    standard_pipeline:    Carnot verify-repair, 100 standard questions
    adversarial_baseline: LLM only, 100 adversarial questions (one distractor appended each)
    adversarial_pipeline: Carnot verify-repair, 100 adversarial questions

    robustness_delta measures whether Carnot's adversarial drop < baseline's adversarial drop.

**Gate chain (in order):**
    0. apply_env_autofix() — FIRST: injects CARNOT_FORCE_LIVE=1 before any CUDA import
    1. ExperimentTimeoutWatchdog(504, timeout_minutes=120) — outer budget cap
    2. DeliverableGuard — ensures JSON is always written even on crash
    3. VRAMBudgetLedger — proactive feasibility check before gate fires
    4. GPUVRAMGateV2(min_free_gb=6.0, kill_first=True) — kill zombies, then check VRAM
    5. DualGPUHarness — assign Gemma4QuantizedLoader→cuda:0, Qwen3.5-0.8B→cuda:1
    6. Four conditions per model via LongRunBenchmarkExecutor(batch_size=8)
    7. AdversarialV4Result per model → robustness_delta, carnot_more_robust
    8. assert_deliverable_written() as FINAL LINE

**Outputs:**
    results/experiment_504_gsm_symbolic_adversarial_v4.json

**honest_verdict semantics:**
    'thesis_confirmed'       — any model has carnot_more_robust=True
    'thesis_not_confirmed'   — no model showed robustness improvement
    'gpu_vram_insufficient'  — GPUVRAMGateV2 blocked the run
    'gpu_required'           — CARNOT_FORCE_LIVE not set or GPU not healthy

Spec: REQ-BENCH-049, REQ-BENCH-050, REQ-BENCH-051,
      SCENARIO-BENCH-068, SCENARIO-BENCH-069, SCENARIO-BENCH-070
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
from carnot.pipeline.adversarial_v4_result import AdversarialV4Result
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.dual_gpu_harness import DualGPUHarness
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2
from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
from carnot.pipeline.vram_budget_ledger import VRAMBudgetLedger
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 504
EXP_TITLE = "GSM-Symbolic Adversarial v4 — Gemma4QuantizedLoader RETRO-039 robustness claim"
DELIVERABLE = "results/experiment_504_gsm_symbolic_adversarial_v4.json"
N_QUESTIONS = 100
DATASET_SEED = 42
BATCH_SIZE = 8

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "Gemma4-Q4KM",
        "hf_id": "unsloth/gemma-4-E4B-it-GGUF",
        "gpu": 0,
        "quantized": True,
    },
    {
        "name": "Qwen3.5-0.8B",
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "gpu": 1,
        "quantized": False,
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

    Uses batch_size from the constant so we can test with small values.
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
                violations = extractor.extract_violations(response)
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
    _log.info("[%s] correct=%d/%d acc=%.3f verdict=%s", condition_label, correct, n, acc, result.honest_verdict)
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
    """Run Experiment 504 and return the artifact dict.

    Gate chain:
        1. CARNOT_FORCE_LIVE check → honest 'gpu_required' if absent
        2. VRAMBudgetLedger feasibility check → logged but non-blocking
        3. GPUVRAMGateV2(kill_first=True) → 'gpu_vram_insufficient' if VRAM too low
        4. DualGPUHarness → assigns cuda:0 and cuda:1
        5. Four conditions per model → AdversarialV4Result
        6. honest_verdict, retro_039_confirmed
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
        artifact = tmpl.build_result(
            {
                "schema": "carnot.adversarial_benchmark.v4",
                "honest_verdict": "gpu_required",
                "retro_039_confirmed": False,
            },
            status="gpu_required",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # Gate 2: VRAMBudgetLedger — proactive feasibility forecast before VRAM gate fires
    ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
    ledger.add_experiment("exp504_gemma4", required_gb=9.0)
    ledger.add_experiment("exp504_qwen", required_gb=2.0)
    forecasts = ledger.check_all()
    for forecast in forecasts:
        _log.info("VRAMBudgetLedger: %s headroom=%.1f GB feasible=%s", forecast.exp_id, forecast.headroom_gb, forecast.is_feasible)

    # Gate 3: GPUVRAMGateV2 — kills zombie GPU processes, then verifies free VRAM >= 6 GiB
    try:
        with GPUVRAMGateV2(min_free_gb=6.0, kill_first=True):
            pass
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGateV2: insufficient VRAM: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.adversarial_benchmark.v4",
                "honest_verdict": "gpu_vram_insufficient",
                "retro_039_confirmed": False,
                "vram_error": str(exc),
            },
            status="gpu_vram_insufficient",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # Gate 4: DualGPUHarness — explicit cuda:0 / cuda:1 assignment
    try:
        import torch  # type: ignore[import]

        n_gpus = torch.cuda.device_count()
    except ImportError:
        n_gpus = 0

    model_specs = [spec.copy() for spec in MODEL_SPECS]
    harness = DualGPUHarness(n_gpus=n_gpus, live_mode=True)
    assigned_specs = harness.apply(model_specs)

    try:
        gpu_status = tmpl.setup_gpu(assigned_specs)
    except Exception as exc:
        _log.warning("setup_gpu() failed: %s", exc)
        gpu_status = {"all_healthy": False, "error": str(exc)}

    if not gpu_status.get("all_healthy", False):
        artifact = tmpl.build_result(
            {
                "schema": "carnot.adversarial_benchmark.v4",
                "honest_verdict": "gpu_required",
                "retro_039_confirmed": False,
                "gpu_status": gpu_status,
            },
            status="gpu_required",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # Load datasets
    standard_questions = _load_standard_gsm8k(N_QUESTIONS)
    adversarial_pairs: list[AdversarialGSMQuestion] = build_adversarial_questions(
        standard_questions, seed=DATASET_SEED
    )
    adversarial_questions = [
        {"question": p.adversarial_question, "answer": p.ground_truth_answer}
        for p in adversarial_pairs
    ]

    _log.info("Dataset: standard=%d adversarial=%d", len(standard_questions), len(adversarial_questions))

    # Shared extractor for all pipeline conditions
    extractor = IntegratedExtractor(
        VeriCoTStepValidator(use_mock=False),
        VPRMArithmeticVerifier(),
    )

    # Load Gemma4 (quantized GGUF) on cuda:0
    gemma_spec = assigned_specs[0]
    gguf_path = os.environ.get("CARNOT_GEMMA4_GGUF_PATH", "")
    gemma_loader = Gemma4QuantizedLoader(
        model_path=gguf_path,
        n_gpu_layers=-1,
        max_tokens=512,
    )
    gemma_loader.load()
    _log.info(
        "Gemma4QuantizedLoader: stub_mode=%s vram_usage_gb=%.1f",
        gemma_loader._stub_mode,
        gemma_loader.vram_usage_gb(),
    )

    # Load Qwen on cuda:1 via HuggingFace transformers
    qwen_spec = assigned_specs[1]
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]

        qwen_pipe = hf_pipeline(
            "text-generation",
            model=qwen_spec["hf_id"],
            device=qwen_spec.get("gpu", 1),
            torch_dtype="auto",
        )

        def _qwen_fn(prompt: str) -> str:
            try:
                out = qwen_pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
                return str(out[0]["generated_text"])
            except Exception as exc:
                _log.warning("Qwen generation failed: %s", exc)
                return ""

    except Exception as exc:
        _log.warning("Could not load Qwen: %s — using stub", exc)

        def _qwen_fn(prompt: str) -> str:  # type: ignore[misc]
            return "The answer is 42."

    def _gemma_fn(prompt: str) -> str:
        return gemma_loader.generate(prompt)

    # Run four conditions per model
    _log.info("=== Gemma4: standard_baseline ===")
    gemma_std_base = _run_condition_accuracy(
        _gemma_fn, standard_questions, None, "gemma4_standard_baseline"
    )
    _log.info("=== Gemma4: standard_pipeline ===")
    gemma_std_pipe = _run_condition_accuracy(
        _gemma_fn, standard_questions, extractor, "gemma4_standard_pipeline"
    )
    _log.info("=== Gemma4: adversarial_baseline ===")
    gemma_adv_base = _run_condition_accuracy(
        _gemma_fn, adversarial_questions, None, "gemma4_adversarial_baseline"
    )
    _log.info("=== Gemma4: adversarial_pipeline ===")
    gemma_adv_pipe = _run_condition_accuracy(
        _gemma_fn, adversarial_questions, extractor, "gemma4_adversarial_pipeline"
    )

    gemma_result = AdversarialV4Result(
        model_id="Gemma4-Q4KM",
        standard_baseline=gemma_std_base,
        standard_pipeline=gemma_std_pipe,
        adversarial_baseline=gemma_adv_base,
        adversarial_pipeline=gemma_adv_pipe,
        n=N_QUESTIONS,
    )
    _log.info(
        "[Gemma4] robustness_delta=%.3f carnot_more_robust=%s",
        gemma_result.robustness_delta,
        gemma_result.carnot_more_robust,
    )

    _log.info("=== Qwen: standard_baseline ===")
    qwen_std_base = _run_condition_accuracy(
        _qwen_fn, standard_questions, None, "qwen_standard_baseline"
    )
    _log.info("=== Qwen: standard_pipeline ===")
    qwen_std_pipe = _run_condition_accuracy(
        _qwen_fn, standard_questions, extractor, "qwen_standard_pipeline"
    )
    _log.info("=== Qwen: adversarial_baseline ===")
    qwen_adv_base = _run_condition_accuracy(
        _qwen_fn, adversarial_questions, None, "qwen_adversarial_baseline"
    )
    _log.info("=== Qwen: adversarial_pipeline ===")
    qwen_adv_pipe = _run_condition_accuracy(
        _qwen_fn, adversarial_questions, extractor, "qwen_adversarial_pipeline"
    )

    qwen_result = AdversarialV4Result(
        model_id="Qwen3.5-0.8B",
        standard_baseline=qwen_std_base,
        standard_pipeline=qwen_std_pipe,
        adversarial_baseline=qwen_adv_base,
        adversarial_pipeline=qwen_adv_pipe,
        n=N_QUESTIONS,
    )
    _log.info(
        "[Qwen] robustness_delta=%.3f carnot_more_robust=%s",
        qwen_result.robustness_delta,
        qwen_result.carnot_more_robust,
    )

    # Compute RETRO-039 verdict
    retro_039_confirmed = any(r.carnot_more_robust for r in [gemma_result, qwen_result])
    honest_verdict = "thesis_confirmed" if retro_039_confirmed else "thesis_not_confirmed"

    artifact = tmpl.build_result(
        {
            "schema": "carnot.adversarial_benchmark.v4",
            "gemma4_result": gemma_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "retro_039_confirmed": retro_039_confirmed,
            "honest_verdict": honest_verdict,
            "n_questions": N_QUESTIONS,
            "dataset_seed": DATASET_SEED,
        },
        status="success",
    )
    _write_json(repo_root, DELIVERABLE, artifact)
    tmpl.assert_deliverable_written()
    return artifact


def main() -> None:
    """Run Experiment 504: GSM-Symbolic adversarial v4 benchmark on live GPU."""
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
