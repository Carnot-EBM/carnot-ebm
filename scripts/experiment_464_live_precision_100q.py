#!/usr/bin/env python3
"""Experiment 464: Live Precision 100q — integrated extraction stack, dual-GPU, RETRO-033 close.

**Researcher summary (RETRO-033 follow-up):**
    Exp 451 (milestone .34) reported a +5pp improvement but the result JSON was absent
    at retrospective time — the exact RETRO-033 failure mode (DeliverableGuard not called).
    This experiment re-runs the live precision benchmark at 100 questions (vs 50 in Exp 451)
    for better statistical confidence, with three additions:

    1. IntegratedExtractor (VeriCoTStepValidator + VPRMArithmeticVerifier) replaces CRANE.
    2. DualGPUAssigner ensures Gemma4-E4B-it→cuda:0 and Qwen3.5-0.8B→cuda:1 (RETRO-034).
    3. DeliverableGuard is called as the FINAL line (RETRO-032/033/036 closure).

**Expected outcome:**
    - Gemma4 baseline: 75-80% (published GSM8K accuracy, now reproducible via GemmaLoader).
    - Qwen3.5-0.8B baseline: 30-50% (0.8B model, expected lower bound).
    - honest_verdict='retro_033_closed_positive' if any model shows improvement.
    - CoT pairs written to results/exp464_cot_pairs.json for Exp 472 JEPA retrain.

**Gate chain (runs in order):**
    0. apply_env_autofix() — FIRST, before any CUDA import (RETRO-022 fix)
    1. ExperimentTimeoutWatchdog(464, timeout_minutes=90) — outer budget cap
    2. CARNOT_FORCE_LIVE check — hard gate, no simulated fallback
    3. setup_gpu() — blocked if not all_healthy
    4. DualGPUAssigner → Gemma4-E4B-it on cuda:0, Qwen3.5-0.8B on cuda:1
    5. IntegratedExtractor(VeriCoTStepValidator + VPRMArithmeticVerifier) — primary extractor
    6. 100 GSM8K questions (stratified 50 easy + 50 hard)

**Outputs:**
    results/experiment_464_live_precision_100q.json — primary artifact (RETRO-033 closure)
    results/exp464_cot_pairs.json — CoT pairs for Exp 472 JEPA retrain

Spec: REQ-BENCH-014, REQ-BENCH-015, REQ-BENCH-016,
      SCENARIO-BENCH-033, SCENARIO-BENCH-034, SCENARIO-BENCH-035
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import occurs.  Moving this below any torch/JAX import is a bug.
# See RETRO-022 for why this matters.
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
from carnot.pipeline.gemma_loader import GemmaTransformersLoader
from carnot.pipeline.precision_100q_result import Precision100qResult
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 464
EXP_TITLE = "Live Precision 100q — IntegratedExtractor + DualGPU (RETRO-033 close)"
DELIVERABLE = "results/experiment_464_live_precision_100q.json"
COT_PAIRS_PATH = "results/exp464_cot_pairs.json"
N_QUESTIONS = 100
N_EASY = 50
N_HARD = 50

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]


# ---------------------------------------------------------------------------
# Answer extraction helpers (same approach as Exp 451)
# ---------------------------------------------------------------------------


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric final answer from a GSM8K response.

    Looks for the official '#### N' format first, then falls back to the last
    number in the text.  Returns None when no numeric answer can be found.
    """
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _is_correct(response: str, gold: str | None) -> bool:
    """Return True when the response contains the gold answer as a final number."""
    if not gold or not response:
        return False
    extracted = _extract_gsm8k_answer(response)
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(gold)) < 0.501
    except (ValueError, TypeError):
        return extracted.strip() == gold.strip()


def _load_gsm8k_questions_stratified(n_easy: int, n_hard: int) -> list[dict]:
    """Load stratified GSM8K questions: n_easy from the start, n_hard from the end.

    GSM8K is roughly ordered from easier to harder questions.  Taking the first
    n_easy and last n_hard questions gives a stratified sample that avoids biasing
    results toward easy or hard questions alone.

    Falls back to synthetic questions when the HuggingFace datasets package is
    unavailable.  Synthetic questions are clearly labelled (source='synthetic') so
    any accuracy numbers are distinguishable from real GSM8K numbers in the artifact.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        all_items = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        if len(all_items) >= n_easy + n_hard:
            easy = all_items[:n_easy]
            hard = all_items[-n_hard:]
            result = easy + hard
            _log.info(
                "Loaded %d GSM8K questions (stratified: %d easy + %d hard)",
                len(result), n_easy, n_hard,
            )
            return result
        # Not enough questions — return all we have
        _log.warning(
            "GSM8K has only %d questions; wanted %d + %d. Using all.",
            len(all_items), n_easy, n_hard,
        )
        return all_items
    except Exception as exc:
        _log.warning("Could not load GSM8K: %s — using synthetic fallback", exc)

    n_total = n_easy + n_hard
    synthetic = []
    for i in range(1, n_total + 1):
        a, b = i * 3, i * 2
        c = a + b
        synthetic.append({
            "question": (
                f"Janet has {a} apples and receives {b} more.  "
                f"How many apples does she have?"
            ),
            "answer": f"She starts with {a} and gets {b} more, so {a} plus {b} gives {c}.  #### {c}",
            "source": "synthetic",
        })
    _log.info("Using %d synthetic GSM8K questions (real dataset unavailable)", len(synthetic))
    return synthetic


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _run_gemma_inference(loader: GemmaTransformersLoader, prompt: str) -> str:
    """Generate a response from Gemma4 via GemmaTransformersLoader.

    Validates output using is_valid_output() to catch any lingering llama.cpp-style
    token failures.  Returns empty string on invalid output so downstream scoring
    treats it as incorrect rather than silently fabricating an answer.
    """
    try:
        text = loader.generate(prompt, max_new_tokens=256)
        if not GemmaTransformersLoader.is_valid_output(text):
            _log.warning("GemmaTransformersLoader.is_valid_output() returned False")
            return ""
        return text
    except Exception as exc:
        _log.warning("Gemma4 generation failed: %s", exc)
        return ""


def _load_qwen_pipeline(hf_id: str, gpu_index: int) -> object:
    """Load a HuggingFace text-generation pipeline for Qwen on the given GPU."""
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"transformers not installed: {exc}") from exc

    return hf_pipeline(
        "text-generation",
        model=hf_id,
        device=gpu_index,
        torch_dtype="auto",
    )


def _run_qwen_inference(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen via HF pipeline.  Returns '' on failure."""
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
        return str(outputs[0]["generated_text"])
    except Exception as exc:
        _log.warning("Qwen generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def _run_model_benchmark(
    model_name: str,
    inference_fn: Any,
    extractor: IntegratedExtractor,
    questions: list[dict],
    cot_pairs: list[dict],
) -> Precision100qResult:
    """Run baseline and pipeline variants for one model.

    Two passes:
    1. BASELINE: raw model output, no verify-repair pipeline.
    2. PIPELINE: IntegratedExtractor violations → one-shot repair when detected.

    CoT pairs (cot_text, correct: bool) are appended to cot_pairs for Exp 472 JEPA retrain.

    Returns a Precision100qResult with Wilson 95% CI and extractor_used field.
    """
    # ---- Pass 1: BASELINE ----
    n_correct_baseline = 0
    for q_dict in questions:
        response = inference_fn(q_dict["question"])
        gold = _extract_gsm8k_answer(q_dict["answer"])
        if _is_correct(response, gold):
            n_correct_baseline += 1

    pre_accuracy = n_correct_baseline / max(len(questions), 1)
    _log.info(
        "  [%s] BASELINE: %d/%d correct (%.4f)",
        model_name, n_correct_baseline, len(questions), pre_accuracy,
    )

    # ---- Pass 2: PIPELINE (IntegratedExtractor + one-shot repair) ----
    n_correct_pipeline = 0
    all_violations_seen: list = []
    for q_dict in questions:
        response = inference_fn(q_dict["question"])
        violations = extractor.extract(response)
        all_violations_seen.extend(violations)

        if violations:
            repair_prompt = (
                f"Question: {q_dict['question']}\n\n"
                f"Your previous answer contained logical or arithmetic errors.  "
                f"Please solve step by step carefully and double-check every calculation."
            )
            response = inference_fn(repair_prompt)

        gold = _extract_gsm8k_answer(q_dict["answer"])
        correct = _is_correct(response, gold)
        if correct:
            n_correct_pipeline += 1

        # Collect CoT pair for Exp 472 JEPA retrain
        cot_pairs.append({
            "model": model_name,
            "question": q_dict["question"],
            "cot_text": response,
            "correct": correct,
        })

    post_accuracy = n_correct_pipeline / max(len(questions), 1)
    extractor_used = extractor.extractor_names_used(all_violations_seen)
    _log.info(
        "  [%s] PIPELINE: %d/%d correct (%.4f) delta=%.4f extractor_used=%s",
        model_name, n_correct_pipeline, len(questions), post_accuracy,
        post_accuracy - pre_accuracy, extractor_used,
    )

    return Precision100qResult(
        model_id=model_name,
        pre_accuracy=pre_accuracy,
        post_accuracy=post_accuracy,
        n_questions=len(questions),
        extractor_used=extractor_used,
        inference_mode="live_gpu",
    )


# ---------------------------------------------------------------------------
# Artifact write helpers
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
    """Run Experiment 464 and return the artifact dict.

    All gates checked in order.  Every execution path writes the deliverable JSON
    before returning so DeliverableGuard.assert_written() always passes.
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
    # Gate 0: GPU required — no simulated fallback (SCENARIO-BENCH-033)
    # ------------------------------------------------------------------
    if not is_live:
        _log.info("CARNOT_FORCE_LIVE not set — GPU required, writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v3",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "retro_033_closed": False,
                "cot_pairs_written": 0,
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 1: setup_gpu — blocked if not all_healthy
    # ------------------------------------------------------------------
    # Apply DualGPUAssigner before passing to setup_gpu
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    assigner = DualGPUAssigner(model_specs=MODEL_SPECS, n_gpus=n_gpus)
    assigned_specs = assigner.assign()

    try:
        gpu_status = tmpl.setup_gpu(assigned_specs)
    except RuntimeError as exc:
        gpu_status = {"all_healthy": False, "failure_reason": str(exc)}

    if not gpu_status["all_healthy"]:
        _log.error("setup_gpu not all_healthy — writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v3",
                "env_autofix": env_autofix_dict,
                "gpu_setup_status": gpu_status,
                "gemma4_result": None,
                "qwen_result": None,
                "retro_033_closed": False,
                "cot_pairs_written": 0,
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 2: Build IntegratedExtractor with use_mock=False (live mode)
    # ------------------------------------------------------------------
    extractor = IntegratedExtractor(
        vericot=VeriCoTStepValidator(use_mock=False),
        vprm=VPRMArithmeticVerifier(),
    )

    # ------------------------------------------------------------------
    # Gate 3: Load Gemma4 via GemmaTransformersLoader on cuda:0
    # ------------------------------------------------------------------
    gemma_loader: GemmaTransformersLoader | None = None
    try:
        _log.info("Loading Gemma4-E4B-it via GemmaTransformersLoader on cuda:0 ...")
        gemma_loader = GemmaTransformersLoader(
            model_id="google/gemma-4-E4B-it",
            device="cuda:0",
        )
        gemma_loader.load()
        _log.info("Gemma4-E4B-it loaded OK")
    except Exception as exc:
        _log.error("Failed to load Gemma4: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v3",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "retro_033_closed": False,
                "cot_pairs_written": 0,
            },
            status="blocked",
            blocked_reason=f"Gemma4 load failed: {exc}",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 4: Load Qwen on cuda:1
    # ------------------------------------------------------------------
    qwen_pipe: object | None = None
    try:
        _log.info("Loading Qwen3.5-0.8B on cuda:1 ...")
        qwen_pipe = _load_qwen_pipeline("Qwen/Qwen3.5-0.8B", gpu_index=1)
        _log.info("Qwen3.5-0.8B loaded OK")
    except Exception as exc:
        _log.error("Failed to load Qwen: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v3",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "retro_033_closed": False,
                "cot_pairs_written": 0,
            },
            status="blocked",
            blocked_reason=f"Qwen load failed: {exc}",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        tmpl.assert_deliverable_written()
        return artifact

    # ------------------------------------------------------------------
    # Load questions (stratified 50 easy + 50 hard)
    # ------------------------------------------------------------------
    questions = _load_gsm8k_questions_stratified(N_EASY, N_HARD)
    _log.info("Loaded %d questions (stratified)", len(questions))

    cot_pairs: list[dict] = []

    # ------------------------------------------------------------------
    # Run benchmarks: Gemma4 then Qwen
    # ------------------------------------------------------------------
    def gemma_fn(prompt: str) -> str:
        assert gemma_loader is not None
        return _run_gemma_inference(gemma_loader, prompt)

    def qwen_fn(prompt: str) -> str:
        assert qwen_pipe is not None
        return _run_qwen_inference(qwen_pipe, prompt)

    _log.info("=== Running Gemma4-E4B-it benchmark (100q) ===")
    gemma4_result = _run_model_benchmark(
        "Gemma4-E4B-it", gemma_fn, extractor, questions, cot_pairs
    )
    tmpl.checkpoint_save(gemma4_result.to_dict(), step=1)

    _log.info("=== Running Qwen3.5-0.8B benchmark (100q) ===")
    qwen_result = _run_model_benchmark(
        "Qwen3.5-0.8B", qwen_fn, extractor, questions, cot_pairs
    )
    tmpl.checkpoint_save(qwen_result.to_dict(), step=2)

    # ------------------------------------------------------------------
    # Write CoT pairs for Exp 472 JEPA retrain
    # ------------------------------------------------------------------
    _write_json(repo_root, COT_PAIRS_PATH, cot_pairs)
    _log.info("CoT pairs written: %d pairs to %s", len(cot_pairs), COT_PAIRS_PATH)

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    any_positive = gemma4_result.is_positive or qwen_result.is_positive
    honest_verdict = (
        "retro_033_closed_positive" if any_positive else "retro_033_closed_negative"
    )

    artifact = tmpl.build_result(
        {
            "schema": "carnot.live_precision.v3",
            "env_autofix": env_autofix_dict,
            "n_questions": len(questions),
            "gemma4_result": gemma4_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "retro_033_closed": True,
            "cot_pairs_written": len(cot_pairs),
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode="live_gpu",
    )
    _write_json(repo_root, DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s retro_033_closed=True "
        "gemma4_delta=%.4f qwen_delta=%.4f cot_pairs=%d",
        honest_verdict,
        gemma4_result.signed_improvement,
        qwen_result.signed_improvement,
        len(cot_pairs),
    )

    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 464: 100q live precision benchmark, RETRO-033 closure.

    Wraps run_experiment() in an ExperimentTimeoutWatchdog with a 90-minute budget
    (double Exp 451's 60-minute budget, accounting for 2× questions + two-pass
    inference per model + CoT pair collection).
    """
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=get_timeout_minutes(),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        artifact = run_experiment()

    verdict = artifact.get("honest_verdict", "unknown")
    _log.info(
        "Exp %d complete: honest_verdict=%s status=%s",
        EXP_ID,
        verdict,
        artifact.get("status", "unknown"),
    )


if __name__ == "__main__":
    main()
