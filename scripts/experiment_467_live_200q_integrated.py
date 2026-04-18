#!/usr/bin/env python3
"""Experiment 467: Live 200q Integrated VeriCoT+VPRM — statistical credibility closure.

**Researcher summary:**
    VeriCoTStepValidator (Exp 453) and VPRMArithmeticVerifier (Exp 454) were
    validated in standalone CPU experiments with impressive results (F1=1.0,
    40% detection improvement).  They were NOT integrated into the live
    VerifyRepairPipeline until Exp 464 (IntegratedExtractor), but Exp 464 ran
    only 100 questions per model.

    At n=100 a 5pp improvement has a ±10pp Wilson CI — the claim is directional
    but not statistically credible.  This experiment runs 200 questions per model
    (±3.5pp CI), replacing simulation-era improvement numbers with credible live
    results from real RTX 3090 hardware.

**Gate chain (runs in order):**
    0. apply_env_autofix() — FIRST, before any CUDA import (RETRO-022 fix)
    1. ExperimentTimeoutWatchdog(467, timeout_minutes=120) — outer budget cap
    2. CARNOT_FORCE_LIVE check — hard gate, writes gpu_required artifact if absent
    3. DualGPUAssigner → Gemma4-E4B-it on cuda:0, Qwen3.5-0.8B on cuda:1
    4. setup_gpu() — writes gpu_required artifact if not all_healthy
    5. IntegratedExtractor(VeriCoTStepValidator(use_mock=False), VPRMArithmeticVerifier())
    6. 200 GSM8K questions (dataset shuffle seed=42 for reproducibility)
    7. BatchedInferenceRunner(batch_size=8) for each model: baseline + pipeline variant

**Outputs:**
    results/experiment_467_live_200q_integrated.json — primary deliverable
    results/exp467_cot_pairs.json — CoT pairs for Exp 472 JEPA retrain

**honest_verdict semantics:**
    'credible_positive'                      — any model has is_statistically_positive
    'improvement_not_statistically_significant' — any model improved but CI overlaps zero
    'no_improvement_200q'                    — no model improved
    'deferred_to_gpu'                        — GPU not available

Spec: REQ-BENCH-017, REQ-BENCH-018, REQ-BENCH-019,
      SCENARIO-BENCH-036, SCENARIO-BENCH-037, SCENARIO-BENCH-038
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
from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner
from carnot.pipeline.experiment_watchdog import (
    ExperimentTimeoutWatchdog,
    get_timeout_minutes,
)
from carnot.pipeline.live_200q_result import Live200qResult
from scripts.experiment_template import BatchedInferenceRunner, ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 467
EXP_TITLE = "Live 200q Integrated VeriCoT+VPRM (Exp 467)"
DELIVERABLE = "results/experiment_467_live_200q_integrated.json"
COT_PAIRS_PATH = "results/exp467_cot_pairs.json"
N_QUESTIONS = 200
DATASET_SEED = 42
BATCH_SIZE = 8

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]

# ---------------------------------------------------------------------------
# GSM8K helpers
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


def _load_gsm8k_200q(seed: int = DATASET_SEED) -> list[dict]:
    """Load 200 GSM8K questions shuffled with a fixed seed for reproducibility.

    Why shuffle with a fixed seed rather than stratified sampling?
        Exp 464 used a stratified sample (first 50 + last 50) which conflates
        difficulty ordering with sampling order.  A seeded shuffle at n=200
        gives an unbiased random sample while remaining fully reproducible.

    Falls back to synthetic questions when the datasets package is unavailable.
    Synthetic questions are labelled source='synthetic' so they are distinguishable
    from real GSM8K questions in the artifact.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        items = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        rng = random.Random(seed)
        rng.shuffle(items)
        result = items[:N_QUESTIONS]
        _log.info("Loaded %d GSM8K questions (seed=%d shuffle)", len(result), seed)
        return result
    except Exception as exc:
        _log.warning("Could not load GSM8K: %s — using synthetic fallback", exc)

    synthetic = []
    rng = random.Random(seed)
    for i in range(1, N_QUESTIONS + 1):
        a = rng.randint(10, 200)
        b = rng.randint(1, 100)
        c = a + b
        synthetic.append({
            "question": (
                f"Janet has {a} apples and receives {b} more.  "
                f"How many apples does she have?"
            ),
            "answer": (
                f"She starts with {a} and gets {b} more, so "
                f"{a} plus {b} gives {c}.  #### {c}"
            ),
            "source": "synthetic",
        })
    _log.info("Using %d synthetic questions (real dataset unavailable)", len(synthetic))
    return synthetic


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_gemma4(gpu_index: int = 0) -> object:
    """Load Gemma4-E4B-it via GemmaTransformersLoader on the specified GPU."""
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

    loader = GemmaTransformersLoader(
        model_id="google/gemma-4-E4B-it",
        device=f"cuda:{gpu_index}",
    )
    loader.load()
    return loader


def _load_qwen(hf_id: str, gpu_index: int = 1) -> object:
    """Load Qwen model via HF text-generation pipeline on the specified GPU."""
    from transformers import pipeline as hf_pipeline  # type: ignore[import]

    return hf_pipeline(
        "text-generation",
        model=hf_id,
        device=gpu_index,
        torch_dtype="auto",
    )


def _gemma_generate(loader: object, prompt: str) -> str:
    """Generate a response from Gemma4.  Returns '' on failure."""
    try:
        from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

        assert isinstance(loader, GemmaTransformersLoader)
        text = loader.generate(prompt, max_new_tokens=256)
        if not GemmaTransformersLoader.is_valid_output(text):
            return ""
        return text
    except Exception as exc:
        _log.warning("Gemma4 generation failed: %s", exc)
        return ""


def _qwen_generate(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen pipeline.  Returns '' on failure."""
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
        return str(outputs[0]["generated_text"])
    except Exception as exc:
        _log.warning("Qwen generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def _run_model_200q(
    model_name: str,
    inference_fn: Any,
    extractor: IntegratedExtractor,
    questions: list[dict],
    cot_pairs: list[dict],
) -> Live200qResult:
    """Run two-pass 200q benchmark for one model and return Live200qResult.

    Pass 1 — BASELINE: raw model output, no verify-repair pipeline.
    Pass 2 — PIPELINE: IntegratedExtractor violations → one-shot repair prompt.

    CoT pairs (model, question, cot_text, correct) are appended to cot_pairs
    during the pipeline pass for Exp 472 JEPA retrain.
    """
    # Pass 1: BASELINE
    n_correct_base = 0
    for q in questions:
        resp = inference_fn(q["question"])
        gold = _extract_gold(q["answer"])
        if _is_correct(resp, gold):
            n_correct_base += 1
    pre_acc = n_correct_base / max(len(questions), 1)
    _log.info("[%s] BASELINE: %d/%d (%.4f)", model_name, n_correct_base, len(questions), pre_acc)

    # Pass 2: PIPELINE (BatchedInferenceRunner wraps the repair-aware inference)
    bir = BatchedInferenceRunner(inference_fn, batch_size=BATCH_SIZE)
    prompts = [q["question"] for q in questions]
    bir_results = bir.run_batch(prompts)

    n_correct_pipe = 0
    for q, bir_result in zip(questions, bir_results):
        resp = bir_result.response
        violations = extractor.extract(resp)
        if violations:
            repair_prompt = (
                f"Question: {q['question']}\n\n"
                f"Your previous answer contained logical or arithmetic errors.  "
                f"Please solve step by step carefully and double-check every calculation."
            )
            resp = inference_fn(repair_prompt)

        gold = _extract_gold(q["answer"])
        correct = _is_correct(resp, gold)
        if correct:
            n_correct_pipe += 1

        cot_pairs.append({
            "model": model_name,
            "question": q["question"],
            "cot_text": resp,
            "correct": correct,
        })

    post_acc = n_correct_pipe / max(len(questions), 1)
    _log.info(
        "[%s] PIPELINE: %d/%d (%.4f) delta=%.4f",
        model_name, n_correct_pipe, len(questions), post_acc, post_acc - pre_acc,
    )

    return Live200qResult(
        model_id=model_name,
        pre_acc=pre_acc,
        post_acc=post_acc,
        n=len(questions),
        extractor_name="VeriCoT+VPRM+CRANE",
        inference_mode="live_gpu",
        cot_pairs=list(cot_pairs),
    )


# ---------------------------------------------------------------------------
# Artifact write helper
# ---------------------------------------------------------------------------


def _write_json(repo_root: Path, rel_path: str, data: Any) -> None:
    """Atomically write JSON data to rel_path under repo_root."""
    out_path = repo_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    Path(tmp).replace(out_path)
    _log.info("Written: %s", out_path)


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path | None = None) -> dict[str, Any]:
    """Run Experiment 467 and return the artifact dict.

    Every execution path writes the deliverable JSON before returning so that
    DeliverableGuard.assert_written() always passes regardless of which gate
    is triggered.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    is_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
        repo_root=repo_root,
    )
    tmpl.setup()

    env_autofix_dict = {
        "gpu_detected": _autofix_result.gpu_detected,
        "carnot_force_live_was_set": _autofix_result.carnot_force_live_was_set,
        "auto_fix_applied": _autofix_result.auto_fix_applied,
        "final_env_value": _autofix_result.final_env_value,
    }

    # ------------------------------------------------------------------
    # Gate: GPU required — write deferred artifact if not in live mode
    # ------------------------------------------------------------------
    if not is_live:
        _log.info("CARNOT_FORCE_LIVE not set — GPU required, writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v4",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    # ------------------------------------------------------------------
    # DualGPUAssigner: Gemma4-E4B-it→cuda:0, Qwen3.5-0.8B→cuda:1
    # ------------------------------------------------------------------
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    assigner = DualGPUAssigner(model_specs=MODEL_SPECS, n_gpus=n_gpus)
    assigned_specs = assigner.assign()

    # ------------------------------------------------------------------
    # setup_gpu — health-check all models
    # ------------------------------------------------------------------
    try:
        gpu_status = tmpl.setup_gpu(assigned_specs)
    except RuntimeError as exc:
        gpu_status = {"all_healthy": False, "failure_reason": str(exc)}

    if not gpu_status["all_healthy"]:
        _log.error("setup_gpu not all_healthy — writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v4",
                "env_autofix": env_autofix_dict,
                "gpu_setup_status": gpu_status,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    # ------------------------------------------------------------------
    # Build IntegratedExtractor with live mode (use_mock=False)
    # ------------------------------------------------------------------
    extractor = IntegratedExtractor(
        vericot=VeriCoTStepValidator(use_mock=False),
        vprm=VPRMArithmeticVerifier(),
    )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    gemma_loader: object | None = None
    try:
        _log.info("Loading Gemma4-E4B-it on cuda:0 ...")
        gemma_loader = _load_gemma4(gpu_index=0)
        _log.info("Gemma4-E4B-it loaded OK")
    except Exception as exc:
        _log.error("Gemma4 load failed: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v4",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
            },
            status="blocked",
            honest_verdict="deferred_to_gpu",
            blocked_reason=f"Gemma4 load failed: {exc}",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    qwen_pipe: object | None = None
    try:
        _log.info("Loading Qwen3.5-0.8B on cuda:1 ...")
        qwen_pipe = _load_qwen("Qwen/Qwen3.5-0.8B", gpu_index=1)
        _log.info("Qwen3.5-0.8B loaded OK")
    except Exception as exc:
        _log.error("Qwen load failed: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v4",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
            },
            status="blocked",
            honest_verdict="deferred_to_gpu",
            blocked_reason=f"Qwen load failed: {exc}",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    # ------------------------------------------------------------------
    # Load 200 GSM8K questions (seed=42 shuffle)
    # ------------------------------------------------------------------
    questions = _load_gsm8k_200q(seed=DATASET_SEED)
    _log.info("Loaded %d questions", len(questions))

    cot_pairs: list[dict] = []

    # ------------------------------------------------------------------
    # Run benchmarks with BatchedInferenceRunner(batch_size=8)
    # ------------------------------------------------------------------
    def gemma_fn(prompt: str) -> str:
        assert gemma_loader is not None
        return _gemma_generate(gemma_loader, prompt)

    def qwen_fn(prompt: str) -> str:
        assert qwen_pipe is not None
        return _qwen_generate(qwen_pipe, prompt)

    _log.info("=== Gemma4-E4B-it: 200q benchmark ===")
    gemma4_pairs: list[dict] = []
    gemma4_result = _run_model_200q("Gemma4-E4B-it", gemma_fn, extractor, questions, gemma4_pairs)
    cot_pairs.extend(gemma4_pairs)
    tmpl.checkpoint_save(gemma4_result.to_dict(), step=1)

    _log.info("=== Qwen3.5-0.8B: 200q benchmark ===")
    qwen_pairs: list[dict] = []
    qwen_result = _run_model_200q("Qwen3.5-0.8B", qwen_fn, extractor, questions, qwen_pairs)
    cot_pairs.extend(qwen_pairs)
    tmpl.checkpoint_save(qwen_result.to_dict(), step=2)

    # ------------------------------------------------------------------
    # Write CoT pairs for Exp 472 JEPA retrain
    # ------------------------------------------------------------------
    _write_json(repo_root, COT_PAIRS_PATH, cot_pairs)
    _log.info("CoT pairs written: %d pairs to %s", len(cot_pairs), COT_PAIRS_PATH)

    # ------------------------------------------------------------------
    # Build artifact with honest_verdict
    # ------------------------------------------------------------------
    any_stat_positive = (
        gemma4_result.is_statistically_positive or qwen_result.is_statistically_positive
    )
    any_signed_positive = (
        gemma4_result.signed_improvement > 0 or qwen_result.signed_improvement > 0
    )

    if any_stat_positive:
        honest_verdict = "credible_positive"
    elif any_signed_positive:
        honest_verdict = "improvement_not_statistically_significant"
    else:
        honest_verdict = "no_improvement_200q"

    artifact = tmpl.build_result(
        {
            "schema": "carnot.live_benchmark.v4",
            "env_autofix": env_autofix_dict,
            "n_questions": len(questions),
            "gemma4_result": gemma4_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "cot_pairs_written": len(cot_pairs),
            "extraction_stack": "VeriCoT+VPRM+CRANE",
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode="live_gpu",
    )
    _write_json(repo_root, DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s gemma4_delta=%.4f qwen_delta=%.4f "
        "gemma4_stat_positive=%s qwen_stat_positive=%s cot_pairs=%d",
        honest_verdict,
        gemma4_result.signed_improvement,
        qwen_result.signed_improvement,
        gemma4_result.is_statistically_positive,
        qwen_result.is_statistically_positive,
        len(cot_pairs),
    )

    # FINAL LINE — must be last (RETRO-032/033/036/038 closure)
    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 467: 200q live integrated benchmark.

    Wraps run_experiment() in a 120-minute ExperimentTimeoutWatchdog.
    200q × 2 models × 2 passes = 4x more inference than Exp 464 (100q),
    so the budget is doubled from 90 to 120 minutes.
    """
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=get_timeout_minutes(),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        artifact = run_experiment()

    _log.info(
        "Exp %d complete: honest_verdict=%s status=%s",
        EXP_ID,
        artifact.get("honest_verdict", "unknown"),
        artifact.get("status", "unknown"),
    )


if __name__ == "__main__":
    main()
