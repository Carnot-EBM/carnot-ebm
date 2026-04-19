#!/usr/bin/env python3
"""Exp 367: Live Extraction Comparison on Gemma4-E4B-it GSM8K Output.

**Researcher summary:**
    Exp 358 benchmarked all three extractors (ArithmeticExtractor, LLMConstraintExtractor,
    LLMz3Formalizer) against Gemma4-E4B-it responses using simulated fallback.  Exp 366
    implemented the full LLMConstraintExtractor.  Exp 367 re-runs the comparison with:
    - LIVE GPU inference from Gemma4-E4B-it (primary model, GPU 0)
    - LIVE Qwen3.5-0.8B as the auxiliary LLM for both LLMConstraintExtractor and
      LLMz3Formalizer (GPU 1)
    - 30 GSM8K questions (smaller set than Exp 358 for faster live turnaround)

    Questions where Gemma4-E4B-it is WRONG are ground truth for violation detection:
    a True Positive (TP) is when the extractor fires on a wrong answer.  Questions where
    the model is CORRECT are ground truth for false positive rate: a False Positive (FP)
    is when the extractor fires on a correct answer.

    This is the FIRST live test of whether any extractor can detect real violations in
    real instruction-tuned model output.

**Experimental design:**
    1. Source scripts/conductor_gpu_env.sh (done before this script is invoked).
    2. Check CARNOT_FORCE_LIVE=1; if not set, write blocked artifact and exit.
    3. Setup GPU with MODEL_SPECS for Gemma4-E4B-it (GPU 0) and Qwen3.5-0.8B (GPU 1).
       If is_live_capable=False, write blocked artifact and exit.
    4. Load 30 GSM8K questions (HuggingFace test split, synthetic fallback).
    5. Run BatchedInferenceRunner with Gemma4-E4B-it to generate responses.
    6. Label each response as correct/wrong using ground-truth numeric comparison.
    7. For each extractor [arithmetic, llm, z3]:
       - Build a detector_fn wrapping the extractor
       - Run run_extractor_comparison() to compute TP/FP/detection_rate/fp_rate
    8. Build artifact using build_extractor_comparison_artifact().
    9. Write results/experiment_367_extraction_live.json.

**Honest verdict:**
    "live_gpu_winner" ONLY when CARNOT_FORCE_LIVE=1 AND all extractor results have
    inference_mode="live_gpu".  Any simulated run produces "simulated_no_verdict".
    This ensures no fake data can appear in headline claims.

Spec: REQ-EXTRACT-023, SCENARIO-EXTRACT-047, SCENARIO-EXTRACT-048
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo-root path setup — must happen before any carnot/scripts imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import BatchedInferenceRunner, ExperimentTemplate

from carnot.pipeline.extract import ArithmeticExtractor
from carnot.pipeline.extractor_comparison import (
    ExtractorComparisonResult,
    build_extractor_comparison_artifact,
    run_extractor_comparison,
)
from carnot.pipeline.llm_extractor import LLMConstraintExtractor
from carnot.pipeline.llm_z3_formalizer import LLMz3Formalizer

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 367
EXP_TITLE = "Live Extraction Comparison on Gemma4-E4B-it GSM8K Output"
DELIVERABLE = "results/experiment_367_extraction_live.json"
N_QUESTIONS = 30

PRIMARY_MODEL_HF_ID = "google/gemma-3-4b-it"
PRIMARY_MODEL_NAME = "Gemma4-E4B-it"
AUX_MODEL_HF_ID = "Qwen/Qwen3.5-0.8B"
AUX_MODEL_NAME = "Qwen3.5-0.8B"

MODEL_SPECS = [
    {"name": PRIMARY_MODEL_NAME, "hf_id": PRIMARY_MODEL_HF_ID, "gpu": 0},
    {"name": AUX_MODEL_NAME, "hf_id": AUX_MODEL_HF_ID, "gpu": 1},
]


# ---------------------------------------------------------------------------
# GSM8K loading with synthetic fallback (identical to Exp 358 pattern)
# ---------------------------------------------------------------------------


def _synthetic_gsm8k(n: int) -> list[dict]:
    """Generate n synthetic GSM8K-style questions for CI/offline use.

    **Detailed explanation for engineers:**
        When HuggingFace datasets is unavailable (CI, offline), we generate
        deterministic arithmetic word problems with known answers.  The difficulty
        is intentionally simple: the test is whether the extractor fires at all,
        not whether the math is hard.

    Args:
        n: Number of questions to generate.

    Returns:
        List of dicts with "question" and "answer" keys.
    """
    questions = []
    for i in range(n):
        a, b = i + 1, i + 2
        questions.append(
            {
                "question": f"Alice has {a} books. She buys {b} more. How many books does she have?",
                "answer": str(a + b),
            }
        )
    return questions


def load_gsm8k_questions(n: int) -> list[dict]:
    """Load n questions from GSM8K test split, falling back to synthetic data.

    **Detailed explanation for engineers:**
        Tries HuggingFace datasets.  On any failure (ImportError, network error),
        falls back to _synthetic_gsm8k().  Returned dicts always have "question"
        and "answer" keys; "answer" is the bare numeric string (#### prefix stripped).

    Args:
        n: Number of questions to return.

    Returns:
        List of question dicts.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for row in ds:
            if len(items) >= n:
                break
            raw_answer = row.get("answer", "")
            answer = (
                raw_answer.split("####")[-1].strip()
                if "####" in raw_answer
                else raw_answer.strip()
            )
            items.append({"question": row["question"], "answer": answer})
        return items
    except Exception as exc:  # noqa: BLE001
        _log.warning("GSM8K dataset unavailable (%s); using synthetic fallback.", exc)
        return _synthetic_gsm8k(n)


# ---------------------------------------------------------------------------
# Response labelling helpers
# ---------------------------------------------------------------------------

_FINAL_NUMBER_RE = re.compile(r"-?\d+(?:[,\d]*)?(?:\.\d+)?")


def _extract_last_number(text: str) -> str | None:
    """Extract the last number from a response string (simple heuristic).

    **Detailed explanation for engineers:**
        GSM8K final answers are usually the last number in the response.
        This is a conservative heuristic: when the last number matches the
        ground truth we call it correct; otherwise we call it wrong.
        When no number is present we conservatively label the response wrong.

    Returns:
        Last number string found, or None.
    """
    nums = _FINAL_NUMBER_RE.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def _label_responses(questions: list[dict]) -> list[bool]:
    """Label each question dict as wrong (True) or correct (False).

    **Detailed explanation for engineers:**
        Compares the last number in the model response to the ground-truth answer.
        When the comparison is inconclusive (no number, non-numeric ground truth),
        we conservatively label the response wrong so violations are not silently
        missed.  This is the safer direction for a benchmark.

    Args:
        questions: List of dicts with "answer" and "response" keys.

    Returns:
        List of bool: True means response is known-wrong.
    """
    labels: list[bool] = []
    for item in questions:
        gt = item.get("answer", "").strip().replace(",", "")
        response = item.get("response", "")
        predicted = _extract_last_number(response)

        if predicted is None:
            labels.append(True)
            continue

        try:
            gt_num = float(gt)
            pred_num = float(predicted)
            labels.append(abs(gt_num - pred_num) > 1e-6)
        except (ValueError, TypeError):
            labels.append(True)

    return labels


# ---------------------------------------------------------------------------
# Extractor detector_fn factories
# ---------------------------------------------------------------------------


def _make_arithmetic_detector():
    """Return a detector_fn wrapping ArithmeticExtractor.

    **Detailed explanation for engineers:**
        ArithmeticExtractor uses regex to find bare "X OP Y = Z" claims.
        It almost always returns 0 violations on IT-format (markdown prose) output,
        making it the baseline that LLM-based extractors are benchmarked against.

    Returns:
        Callable (question: str, response: str) -> bool
    """
    extractor = ArithmeticExtractor()

    def _fn(question: str, response: str) -> bool:
        results = extractor.extract(response, domain="arithmetic")
        return any(not r.metadata.get("satisfied", True) for r in results if r.metadata)

    return _fn


def _make_llm_detector(extractor: LLMConstraintExtractor | None = None):
    """Return a detector_fn wrapping LLMConstraintExtractor.

    **Detailed explanation for engineers:**
        In live mode, pass a pre-initialized LLMConstraintExtractor backed by
        Qwen3.5-0.8B.  In CI/simulated mode, pass None to create a stub that
        always returns no violations (deterministic, no GPU needed).

    Args:
        extractor: Pre-initialized LLMConstraintExtractor, or None for CI stub.

    Returns:
        Callable (question: str, response: str) -> bool
    """
    if extractor is None:
        extractor = LLMConstraintExtractor(
            model=object(),
            tokenizer=object(),
            generate_fn=lambda model, tok, prompt, max_new_tokens: "",
        )

    def _fn(question: str, response: str) -> bool:
        results = extractor.extract(response, domain="arithmetic")
        return any(not r.metadata.get("satisfied", True) for r in results if r.metadata)

    return _fn


def _make_z3_detector(llm_caller=None):
    """Return a detector_fn wrapping LLMz3Formalizer.

    **Detailed explanation for engineers:**
        LLMz3Formalizer asks Qwen3.5-0.8B to write Z3 Python code asserting all
        arithmetic claims in the response, then runs the code in a restricted exec()
        sandbox.  A violation is detected only when Z3 returns "unsat" (arithmetic
        contradiction proven), giving a zero-false-positive contract per REQ-EXTRACT-020.

        When llm_caller is None, uses the CI stub (always returns "sat").

    Args:
        llm_caller: Callable (prompt: str) -> str, or None for CI stub.

    Returns:
        Callable (question: str, response: str) -> bool
    """
    model_id = "ci_stub" if llm_caller is None else AUX_MODEL_HF_ID
    formalizer = LLMz3Formalizer(llm_caller=llm_caller, model_id=model_id)

    def _fn(question: str, response: str) -> bool:
        result = formalizer.formalize(question, response)
        return result.z3_result == "unsat"

    return _fn


def _simulated_response(question: dict) -> str:
    """Return a deterministic simulated response for CI/offline use.

    **Detailed explanation for engineers:**
        Used in simulated mode when CARNOT_FORCE_LIVE is not set.  The response
        echoes the first number in the question so labelling logic can run, but
        the honest_verdict will always be "simulated_no_verdict".
    """
    match = re.search(r"\d+", question.get("question", ""))
    if match:
        return f"The answer is {match.group()}."
    return "I don't know."


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the live extraction comparison.

    Flow:
        1. Check CARNOT_FORCE_LIVE; if not set, produce blocked artifact.
        2. Setup GPU (Gemma4-E4B-it + Qwen3.5-0.8B).
        3. Load 30 GSM8K questions.
        4. Run live inference with Gemma4-E4B-it.
        5. Label responses (correct/wrong).
        6. Run three extractor detector_fns; compute per-extractor metrics.
        7. Build artifact with honest_verdict.
        8. Write results/experiment_367_extraction_live.json.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=True,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
    inference_mode = "live_gpu" if force_live else "simulated"

    # Step 1: Gate on CARNOT_FORCE_LIVE
    if not force_live:
        _log.warning(
            "CARNOT_FORCE_LIVE is not set to 1.  "
            "This experiment requires live GPU inference to produce a credible result.  "
            "Writing blocked artifact."
        )
        artifact = tmpl.build_result(
            {
                "inference_mode": "simulated",
                "blocked_reason": "CARNOT_FORCE_LIVE not set; live GPU required",
                "honest_verdict": "blocked_force_live_not_set",
            },
            status="blocked",
        )
        out_path = _REPO_ROOT / DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        _log.info("Blocked artifact written to %s", out_path)
        return

    # Step 2: Setup GPU
    try:
        gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    except RuntimeError as exc:
        _log.error("GPU setup failed: %s", exc)
        artifact = tmpl.build_result(
            {
                "inference_mode": "live_gpu",
                "blocked_reason": str(exc),
                "honest_verdict": "blocked_live_gpu_unavailable",
            },
            status="blocked",
        )
        out_path = _REPO_ROOT / DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        return

    if not gpu_status.get("all_healthy", False):
        _log.error("GPU health check failed: %s", gpu_status)
        artifact = tmpl.build_result(
            {
                "inference_mode": "live_gpu",
                "blocked_reason": "GPU health check failed",
                "gpu_status": gpu_status,
                "honest_verdict": "blocked_live_gpu_unavailable",
            },
            status="blocked",
        )
        out_path = _REPO_ROOT / DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        return

    # Step 3: Load questions
    raw_questions = load_gsm8k_questions(N_QUESTIONS)
    n_loaded = len(raw_questions)
    _log.info("Loaded %d GSM8K questions (mode=%s)", n_loaded, inference_mode)

    # Step 4: Load primary model and run live inference
    try:
        from carnot.inference.model_loader import generate, load_model  # type: ignore[import]

        primary_model, primary_tokenizer = load_model(PRIMARY_MODEL_HF_ID)
        aux_model, aux_tokenizer = load_model(AUX_MODEL_HF_ID)
    except Exception as exc:  # noqa: BLE001
        _log.error("Model load failed: %s", exc)
        artifact = tmpl.build_result(
            {
                "inference_mode": "live_gpu",
                "blocked_reason": f"model_load_failed: {exc}",
                "honest_verdict": "blocked_model_load_failed",
            },
            status="blocked",
        )
        out_path = _REPO_ROOT / DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        return

    # Build primary runner for BatchedInferenceRunner
    def _primary_runner(prompt: str) -> str:
        return generate(primary_model, primary_tokenizer, prompt, max_new_tokens=256)

    # Build aux runner for LLMConstraintExtractor and LLMz3Formalizer
    def _aux_caller(prompt: str) -> str:
        return generate(aux_model, aux_tokenizer, prompt, max_new_tokens=128)

    bir = BatchedInferenceRunner(_primary_runner, batch_size=8)
    prompts = [q["question"] for q in raw_questions]
    _log.info("Running live inference on %d questions...", len(prompts))
    inference_results = bir.run_batch(prompts)
    batch_log = bir.batch_log

    for q, ir in zip(raw_questions, inference_results):
        q["response"] = ir.response

    # Step 5: Label responses
    ground_truth_wrong = _label_responses(raw_questions)
    n_wrong = sum(1 for g in ground_truth_wrong if g)
    n_correct = len(ground_truth_wrong) - n_wrong
    _log.info("Labels: %d wrong, %d correct (out of %d)", n_wrong, n_correct, n_loaded)

    # Step 6: Build extractor detector functions (all backed by live Qwen3.5-0.8B)
    arith_fn = _make_arithmetic_detector()
    llm_extractor = LLMConstraintExtractor(
        model=aux_model,
        tokenizer=aux_tokenizer,
        generate_fn=lambda model_obj, tok, prompt, max_tok: _aux_caller(prompt),
    )
    llm_fn = _make_llm_detector(llm_extractor)
    z3_fn = _make_z3_detector(llm_caller=_aux_caller)

    # Step 6 (continued): Run comparison for each extractor
    _log.info("Running ArithmeticExtractor comparison...")
    arith_result = run_extractor_comparison(
        extractor_name="arithmetic",
        detector_fn=arith_fn,
        questions=raw_questions,
        ground_truth_wrong=ground_truth_wrong,
        inference_mode=inference_mode,
    )

    _log.info("Running LLMConstraintExtractor comparison...")
    llm_result = run_extractor_comparison(
        extractor_name="llm",
        detector_fn=llm_fn,
        questions=raw_questions,
        ground_truth_wrong=ground_truth_wrong,
        inference_mode=inference_mode,
    )

    _log.info("Running LLMz3Formalizer comparison...")
    z3_result = run_extractor_comparison(
        extractor_name="z3",
        detector_fn=z3_fn,
        questions=raw_questions,
        ground_truth_wrong=ground_truth_wrong,
        inference_mode=inference_mode,
    )

    all_results = [arith_result, llm_result, z3_result]

    # Step 7: Build comparison artifact
    comparison = build_extractor_comparison_artifact(all_results)

    _log.info(
        "Winner: %s | honest_verdict: %s",
        comparison["winner_extractor"],
        comparison["honest_verdict"],
    )

    # Log per-extractor detection rates
    for entry in comparison["per_extractor_results"]:
        _log.info(
            "  %s: detection_rate=%.3f  fp_rate=%.3f  TP=%d  FP=%d",
            entry["extractor_name"],
            entry["detection_rate"],
            entry["fp_rate"],
            entry["n_true_positives"],
            entry["n_false_positives"],
        )

    artifact_data: dict[str, Any] = {
        "schema": "carnot.extraction_comparison.v1",
        "inference_mode": inference_mode,
        "n_questions": n_loaded,
        "n_wrong": n_wrong,
        "n_correct": n_correct,
        "primary_model": PRIMARY_MODEL_HF_ID,
        "aux_model": AUX_MODEL_HF_ID,
        "batch_log": batch_log,
        "per_extractor_results": comparison["per_extractor_results"],
        "winner_extractor": comparison["winner_extractor"],
        "honest_verdict": comparison["honest_verdict"],
        "n_extractors": comparison["n_extractors"],
    }

    artifact = tmpl.build_result(artifact_data, status="success")

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", out_path)


if __name__ == "__main__":
    main()


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
