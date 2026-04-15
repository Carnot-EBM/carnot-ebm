#!/usr/bin/env python3
"""Exp 358: Comparative Extraction Benchmark on Live IT Model Output (Gemma4-E4B-it).

**Researcher summary:**
    ArithmeticExtractor found 0/20 violations on Gemma4-E4B-it responses (Exp 353/355)
    because IT-format responses are markdown prose, not bare "X + Y = Z" lines.
    This experiment benchmarks all three extraction approaches on 50 live Gemma4-E4B-it
    responses to GSM8K questions.  Primary metric: violation detection rate on known-wrong
    answers.

**Experimental design:**
    1. Load 50 GSM8K questions from the test split (or synthetic fallback in CI).
    2. Get Gemma4-E4B-it answers via BatchedInferenceRunner (live GPU if CARNOT_FORCE_LIVE=1).
    3. Label each answer as correct/wrong using ground-truth numeric comparison.
    4. Run three extractors on all 50 responses:
       - ArithmeticExtractor ("arithmetic") — regex-based baseline
       - LLMConstraintExtractor ("llm") — second LLM call for structured CLAIM: extraction
       - LLMz3Formalizer ("z3") — LLM rewrites arithmetic as Z3 assertions
    5. For each wrong answer: record whether the extractor detected a violation (TP).
    6. For each correct answer: record whether the extractor raised a false alarm (FP).
    7. Compute detection_rate = TP/(TP+FN), false_positive_rate = FP/(FP+TN).
    8. Build artifact with winner and honest_verdict.

**Honest verdict:**
    "live_gpu_llm_extractor_wins" ONLY when CARNOT_FORCE_LIVE=1 AND LLMConstraintExtractor
    detection_rate > ArithmeticExtractor detection_rate.  Simulated runs always produce
    "simulated_no_verdict" so fake data can never appear in headline claims.

Spec: REQ-EXTRACT-021, SCENARIO-EXTRACT-042, SCENARIO-EXTRACT-043
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
from carnot.pipeline.extraction_benchmark import (
    ExtractionBenchmarkResult,
    build_extraction_comparison_artifact,
    run_extraction_benchmark,
)
from carnot.pipeline.llm_extractor import LLMConstraintExtractor
from carnot.pipeline.llm_z3_formalizer import LLMz3Formalizer

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 358
EXP_TITLE = "Comparative Extraction Benchmark on Live IT Model Output (Gemma4-E4B-it)"
DELIVERABLE = "results/experiment_358_extraction_benchmark.json"
N_QUESTIONS = 50

# Primary inference model for generating responses to benchmark
PRIMARY_MODEL_HF_ID = "google/gemma-3-4b-it"
PRIMARY_MODEL_NAME = "Gemma4-E4B-it"

MODEL_SPECS = [
    {"name": PRIMARY_MODEL_NAME, "hf_id": PRIMARY_MODEL_HF_ID, "gpu": 0},
]


# ---------------------------------------------------------------------------
# GSM8K loading with synthetic fallback
# ---------------------------------------------------------------------------


def _synthetic_gsm8k(n: int) -> list[dict]:
    """Generate n synthetic GSM8K-style questions for CI/offline use.

    **Detailed explanation for engineers:**
        When the HuggingFace datasets library is not available (CI, offline),
        we generate deterministic arithmetic word problems with known answers.
        These have known correct numeric answers so _label_responses() works.
        The difficulty is intentionally simple: the extractor baseline tests
        whether violation detection works at all, not whether math is hard.

    Args:
        n: Number of synthetic questions to generate.

    Returns:
        List of dicts with "question" and "answer" keys.
    """
    questions = []
    for i in range(n):
        a, b = i + 1, i + 2
        questions.append(
            {
                "question": f"John has {a} apples. He buys {b} more. How many apples does he have?",
                "answer": str(a + b),
            }
        )
    return questions


def load_gsm8k_questions(n: int) -> list[dict]:
    """Load n questions from the GSM8K test split, falling back to synthetic data.

    **Detailed explanation for engineers:**
        Tries to load via HuggingFace datasets.  On failure (ImportError or any
        datasets error), falls back to _synthetic_gsm8k() which produces
        deterministic questions valid for CI runs.

        Each returned dict has at least "question" (str) and "answer" (str, the
        final numeric answer string — may include "####" prefix per GSM8K format
        which _label_responses strips).

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
            # GSM8K answer format: "... #### 42"
            raw_answer = row.get("answer", "")
            answer = raw_answer.split("####")[-1].strip() if "####" in raw_answer else raw_answer.strip()
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
        GSM8K answers are usually the last numeric value in a response.
        This is a naive heuristic — it works well enough for detecting
        gross errors but is not a substitute for semantic understanding.
        For the benchmark we only need to know if the final answer matches
        the ground truth; we don't need to parse every step.

    Returns:
        The last number string found, or None if no number present.
    """
    nums = _FINAL_NUMBER_RE.findall(text)
    if not nums:
        return None
    # Remove comma separators for comparison
    return nums[-1].replace(",", "")


def _label_responses(questions: list[dict]) -> list[bool]:
    """Label each question dict as wrong (True) or correct (False).

    **Detailed explanation for engineers:**
        Each dict must have "answer" (ground truth) and "response" (model output).
        We extract the last number from the model response and compare it to the
        ground truth answer string numerically.

        When comparison fails (non-numeric ground truth, no number in response),
        we conservatively label the response as wrong (True) to avoid missing
        real violations.  This is the safer direction for a benchmark: we prefer
        false negatives in labelling over silent label errors.

    Args:
        questions: List of dicts, each with "answer" and "response" keys.

    Returns:
        List of bool: True means the response is known-wrong.
    """
    labels: list[bool] = []
    for item in questions:
        gt = item.get("answer", "").strip().replace(",", "")
        response = item.get("response", "")
        predicted = _extract_last_number(response)

        if predicted is None:
            labels.append(True)  # no number found → treat as wrong
            continue

        try:
            gt_num = float(gt)
            pred_num = float(predicted)
            labels.append(abs(gt_num - pred_num) > 1e-6)
        except (ValueError, TypeError):
            labels.append(True)  # non-numeric ground truth → conservative wrong

    return labels


# ---------------------------------------------------------------------------
# Extractor inference function factories
# ---------------------------------------------------------------------------


def _make_arithmetic_inference_fn():
    """Return a ViolationDetector for ArithmeticExtractor.

    **Detailed explanation for engineers:**
        ArithmeticExtractor.extract() returns ConstraintResult objects.
        A violation is detected when any result has metadata["satisfied"] == False.
        This is the baseline — it only catches "X + Y = Z" claims written in
        bare numeric form without markdown formatting.

    Returns:
        Callable (question: str, response: str) -> bool
    """
    extractor = ArithmeticExtractor()

    def _fn(question: str, response: str) -> bool:
        results = extractor.extract(response, domain="arithmetic")
        return any(
            not r.metadata.get("satisfied", True)
            for r in results
            if r.metadata
        )

    return _fn


def _make_llm_inference_fn(extractor: LLMConstraintExtractor | None = None):
    """Return a ViolationDetector for LLMConstraintExtractor.

    **Detailed explanation for engineers:**
        LLMConstraintExtractor issues a second LLM call to canonicalize arithmetic
        claims into "CLAIM: a OP b = c" format, then verifies each claim.
        A violation is detected when any extracted claim has satisfied == False.

        In CI mode (extractor=None), a fresh LLMConstraintExtractor is created
        without a model_name — callers in live mode should pass a pre-loaded extractor.

    Args:
        extractor: Pre-initialized LLMConstraintExtractor; or None to create a stub.

    Returns:
        Callable (question: str, response: str) -> bool
    """
    if extractor is None:
        # CI stub: no model loaded, extract() will return [] on empty responses
        extractor = LLMConstraintExtractor(
            model=object(),  # Dummy — generate_fn will be stubbed below
            tokenizer=object(),
            generate_fn=lambda model, tok, prompt, max_new_tokens: "",
        )

    def _fn(question: str, response: str) -> bool:
        results = extractor.extract(response, domain="arithmetic")
        return any(
            not r.metadata.get("satisfied", True)
            for r in results
            if r.metadata
        )

    return _fn


def _make_z3_inference_fn(llm_caller=None):
    """Return a ViolationDetector for LLMz3Formalizer.

    **Detailed explanation for engineers:**
        LLMz3Formalizer.formalize() runs the LLM to produce Z3 assertions, then
        executes them in a restricted exec() sandbox.  A violation is detected
        only when Z3 returns "unsat" (arithmetic contradiction proven) — per
        REQ-EXTRACT-020's zero-false-positive contract.

        When llm_caller=None, the CI stub is used (always returns "sat").

    Args:
        llm_caller: Callable (prompt: str) -> str; or None for CI stub.

    Returns:
        Callable (question: str, response: str) -> bool
    """
    formalizer = LLMz3Formalizer(llm_caller=llm_caller, model_id="ci_stub" if llm_caller is None else "live")

    def _fn(question: str, response: str) -> bool:
        result = formalizer.formalize(question, response)
        # Only "unsat" means a verified arithmetic contradiction (REQ-EXTRACT-020)
        return result.z3_result == "unsat"

    return _fn


# ---------------------------------------------------------------------------
# Live inference runner
# ---------------------------------------------------------------------------


def _build_live_runner(model, tokenizer):
    """Build a runner function for BatchedInferenceRunner using a loaded model.

    Args:
        model:     HuggingFace model object.
        tokenizer: HuggingFace tokenizer object.

    Returns:
        Callable (prompt: str) -> str
    """
    from carnot.inference.model_loader import generate  # type: ignore[import]

    def _runner(prompt: str) -> str:
        return generate(model, tokenizer, prompt, max_new_tokens=256)

    return _runner


def _simulated_response(question: dict) -> str:
    """Return a simulated response for CI/offline benchmarks.

    **Detailed explanation for engineers:**
        In simulated mode we cannot claim live_gpu results, so the response
        is minimal and deterministic.  The benchmark will still run all three
        extractors but the honest_verdict will be "simulated_no_verdict".
    """
    a_match = re.search(r"\d+", question.get("question", ""))
    if a_match:
        n = int(a_match.group())
        return f"The answer is {n}."
    return "I don't know."


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the comparative extraction benchmark.

    Flow:
        1. Check CARNOT_FORCE_LIVE to determine inference mode.
        2. Read experiment_353 smoke test result for live GPU provenance note.
        3. Load N_QUESTIONS from GSM8K (or synthetic fallback).
        4. In live mode: setup_gpu() + load model + run BatchedInferenceRunner.
           In simulated mode: generate synthetic responses deterministically.
        5. Label responses using ground-truth numeric comparison.
        6. Run three extractor inference functions over all responses.
        7. Build comparison artifact with winner + honest_verdict.
        8. Write to DELIVERABLE.
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

    # Step 2: Read experiment_353 smoke test for provenance note
    exp353_path = _REPO_ROOT / "results" / "experiment_353_live_gpu_smoke_test.json"
    exp353_note = "not_found"
    if exp353_path.exists():
        try:
            exp353_data = json.loads(exp353_path.read_text())
            exp353_note = exp353_data.get("finding", "unknown")
        except (json.JSONDecodeError, OSError):
            exp353_note = "unreadable"

    # Exp 353 result is "partial" — not a confirmed live_gpu; we honour CARNOT_FORCE_LIVE
    # as the authoritative gate rather than requiring a confirmed smoke test status.

    # Step 3: Load questions
    raw_questions = load_gsm8k_questions(N_QUESTIONS)
    n_loaded = len(raw_questions)
    _log.info("Loaded %d questions (mode=%s)", n_loaded, inference_mode)

    # Step 4: Gather model responses
    if force_live:
        # Try to set up GPU; build responses via live inference
        try:
            gpu_status = tmpl.setup_gpu(MODEL_SPECS)
        except RuntimeError as exc:
            _log.error("GPU setup failed: %s", exc)
            artifact = tmpl.build_result(
                {
                    "inference_mode": "live_gpu",
                    "exp353_smoke_test_note": exp353_note,
                    "blocked_reason": str(exc),
                    "honest_verdict": "blocked_live_gpu_unavailable",
                },
                status="blocked",
            )
            out_path = _REPO_ROOT / DELIVERABLE
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(artifact, indent=2))
            return

        # Load model for inference
        try:
            from carnot.inference.model_loader import load_model  # type: ignore[import]

            model, tokenizer = load_model(PRIMARY_MODEL_HF_ID)
        except Exception as exc:  # noqa: BLE001
            _log.error("Model load failed: %s", exc)
            artifact = tmpl.build_result(
                {
                    "inference_mode": "live_gpu",
                    "exp353_smoke_test_note": exp353_note,
                    "blocked_reason": f"model_load_failed: {exc}",
                    "honest_verdict": "blocked_model_load_failed",
                },
                status="blocked",
            )
            out_path = _REPO_ROOT / DELIVERABLE
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(artifact, indent=2))
            return

        runner = _build_live_runner(model, tokenizer)
        bir = BatchedInferenceRunner(runner, batch_size=8)
        prompts = [q["question"] for q in raw_questions]
        inference_results = bir.run_batch(prompts)
        batch_log = bir.batch_log

        for q, ir in zip(raw_questions, inference_results):
            q["response"] = ir.response

    else:
        # Simulated mode: generate deterministic responses
        batch_log = []
        for q in raw_questions:
            q["response"] = _simulated_response(q)
        gpu_status = None

    # Step 5: Label responses
    ground_truth_wrong = _label_responses(raw_questions)
    n_wrong = sum(1 for g in ground_truth_wrong if g)
    n_correct = len(ground_truth_wrong) - n_wrong
    _log.info("Labels: %d wrong, %d correct", n_wrong, n_correct)

    # Step 6: Build extractor inference functions
    # In live mode we create a second model caller for LLMConstraintExtractor.
    # In simulated mode we use stubs that return empty / no-violation results.
    if force_live:
        llm_generate_fn = lambda model_obj, tok, prompt, max_tok: runner(prompt)  # noqa: E731
        llm_extractor = LLMConstraintExtractor(
            model=model,
            tokenizer=tokenizer,
            generate_fn=llm_generate_fn,
        )
        z3_llm_caller = lambda prompt: runner(prompt)  # noqa: E731
    else:
        llm_extractor = None  # will use stub path in _make_llm_inference_fn
        z3_llm_caller = None  # will use CI stub in LLMz3Formalizer

    arith_fn = _make_arithmetic_inference_fn()
    llm_fn = _make_llm_inference_fn(llm_extractor)
    z3_fn = _make_z3_inference_fn(z3_llm_caller)

    # Step 6 (continued): Run extractors and collect results
    _log.info("Running ArithmeticExtractor benchmark...")
    arith_result = run_extraction_benchmark(
        extractor_name="arithmetic",
        inference_fn=arith_fn,
        questions=raw_questions,
        ground_truth_wrong=ground_truth_wrong,
        inference_mode=inference_mode,
    )

    _log.info("Running LLMConstraintExtractor benchmark...")
    llm_result = run_extraction_benchmark(
        extractor_name="llm",
        inference_fn=llm_fn,
        questions=raw_questions,
        ground_truth_wrong=ground_truth_wrong,
        inference_mode=inference_mode,
    )

    _log.info("Running LLMz3Formalizer benchmark...")
    z3_result = run_extraction_benchmark(
        extractor_name="z3",
        inference_fn=z3_fn,
        questions=raw_questions,
        ground_truth_wrong=ground_truth_wrong,
        inference_mode=inference_mode,
    )

    # Step 7: Build comparison artifact
    comparison = build_extraction_comparison_artifact([arith_result, llm_result, z3_result])

    _log.info(
        "Winner: %s | detection_rate: %.3f | honest_verdict: %s",
        comparison["winner"],
        next(
            r["detection_rate"]
            for r in comparison["per_extractor_results"]
            if r["extractor_name"] == comparison["winner"]
        ),
        comparison["honest_verdict"],
    )

    artifact_data: dict[str, Any] = {
        "schema": "carnot.extraction_benchmark.v1",
        "inference_mode": inference_mode,
        "n_questions": n_loaded,
        "n_wrong": n_wrong,
        "n_correct": n_correct,
        "primary_model": PRIMARY_MODEL_HF_ID,
        "exp353_smoke_test_note": exp353_note,
        "batch_log": batch_log,
        "per_extractor_results": comparison["per_extractor_results"],
        "winner": comparison["winner"],
        "improvement_over_arithmetic_extractor": comparison["improvement_over_arithmetic_extractor"],
        "honest_verdict": comparison["honest_verdict"],
        "n_extractors": comparison["n_extractors"],
    }

    artifact = tmpl.build_result(artifact_data, status="success")

    # Step 8: Write artifact
    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", out_path)


if __name__ == "__main__":
    main()
