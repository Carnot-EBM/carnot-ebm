#!/usr/bin/env python3
"""Experiment 354: Adversarial GSM8K benchmark harness (script-generation phase).

**Researcher summary:**
    Apple researchers (arXiv 2410.05229) showed that appending one irrelevant sentence
    to math problems causes frontier LLMs to drop up to 65% accuracy.  Carnot's
    ArithmeticExtractor parses explicit equation tokens — not context words — so the
    Ising energy should be invariant to irrelevant-sentence injection.

    This experiment is the HARNESS ONLY (script-generation phase).  It:
    1. Loads 50 GSM8K questions (from HuggingFace or a deterministic synthetic fallback).
    2. Calls build_adversarial_questions() to produce distractor-appended variants.
    3. Verifies the harness round-trips correctly (adversarial_question contains original).
    4. Emits a harness-ready artifact with schema="carnot.adversarial_harness.v1".

    Live inference (running the pipeline on both standard and adversarial questions
    with a real GPU model) is Exp 355.  This separation follows the
    research-program.md "Large Benchmark Experiments" rule: write the harness first,
    validate it offline, then execute the live benchmark separately.

**CI-safe mode:**
    When CARNOT_FORCE_LIVE is not set or is "0", the experiment runs entirely offline:
    it loads synthetic GSM8K questions (no network required) and validates the harness
    round-trip without running any model inference.  The artifact has status="success"
    and harness_ready=True regardless of live GPU availability.

**Output:** results/experiment_354_adversarial_gsm8k_harness.json

Spec: REQ-BENCH-006, REQ-BENCH-007,
      SCENARIO-BENCH-014, SCENARIO-BENCH-015, SCENARIO-BENCH-016
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path so scripts.* and carnot.* resolve.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.adversarial_gsm8k import (  # noqa: E402
    build_adversarial_questions,
)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 354
EXP_TITLE = "Adversarial GSM8K benchmark harness (script-generation phase)"
DELIVERABLE = "results/experiment_354_adversarial_gsm8k_harness.json"
N_QUESTIONS = 50
SEED = 42


# ---------------------------------------------------------------------------
# GSM8K question loading
# ---------------------------------------------------------------------------


def load_gsm8k_questions(n: int = N_QUESTIONS) -> list[dict]:
    """Load up to ``n`` GSM8K questions from HuggingFace or fall back to synthetic.

    **Detailed explanation for engineers:**
        Tries the HuggingFace ``datasets`` library first (requires network + cache).
        Falls back to a deterministic synthetic set for CI / offline use.  The
        synthetic set produces simple multi-step arithmetic questions whose answers
        are derivable without model inference — this is sufficient for harness
        validation (the harness only needs to confirm round-trip correctness, not
        measure accuracy).

    Args:
        n: Maximum number of questions to return.

    Returns:
        List of dicts with keys ``"question"`` (str) and ``"answer"`` (str).
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        items = list(ds)[:n]
        _log.info("Loaded %d GSM8K questions from HuggingFace datasets.", len(items))
        return [{"question": item["question"], "answer": item["answer"]} for item in items]
    except Exception as exc:
        _log.warning(
            "GSM8K HuggingFace load failed (%s) — using synthetic questions.", exc
        )

    return _synthetic_gsm8k(n)


def _synthetic_gsm8k(n: int) -> list[dict]:
    """Generate ``n`` deterministic synthetic arithmetic questions for CI / offline use.

    **Detailed explanation for engineers:**
        Uses the same pattern as Exp 340 to keep synthetic questions comparable
        across experiments.  Each question contains an explicit equation that the
        ArithmeticExtractor can find (a + b = c).  Answers use the #### format
        that the real GSM8K dataset uses.

    Args:
        n: Number of questions to generate.

    Returns:
        List of dicts with keys ``"question"`` and ``"answer"``.
    """
    questions = []
    for i in range(n):
        a = i + 1
        b = i + 2
        answer = (a + b) * 2
        questions.append(
            {
                "question": (
                    f"A store has {a} red apples and {b} green apples. "
                    f"Each apple is sold in pairs. How many apples are sold in total? "
                    f"So {a} + {b} = {a + b} apples total, and {a + b} * 2 = {answer}."
                ),
                "answer": f"#### {answer}",
            }
        )
    return questions


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------


def _validate_round_trip(adversarial_questions: list) -> dict:
    """Verify that every adversarial_question contains its original_question.

    **Detailed explanation for engineers:**
        This is the core harness correctness check.  build_adversarial_questions()
        must always produce an adversarial variant that strictly extends the original
        (the distractor is appended; the original text is never modified).

        Any failure here would indicate a bug in build_adversarial_questions() that
        could silently corrupt the benchmark — original and adversarial conditions
        would differ in ways beyond a simple appended sentence.

    Args:
        adversarial_questions: Output of build_adversarial_questions().

    Returns:
        Dict with keys ``n_checked``, ``n_passed``, ``n_failed``, ``all_ok`` (bool),
        and ``failed_ids`` (list of question_id strings where the check failed).
    """
    n_failed = 0
    failed_ids = []
    for aq in adversarial_questions:
        if aq.original_question not in aq.adversarial_question:
            n_failed += 1
            failed_ids.append(aq.question_id)
        if not aq.adversarial_question.endswith(aq.irrelevant_sentence):
            n_failed += 1
            failed_ids.append(f"{aq.question_id}:suffix_mismatch")

    n_passed = len(adversarial_questions) - n_failed
    return {
        "n_checked": len(adversarial_questions),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "all_ok": n_failed == 0,
        "failed_ids": failed_ids,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 354: adversarial GSM8K harness (script-generation phase)."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # --- Step 1: Load questions ---
    _log.info("Loading %d GSM8K questions...", N_QUESTIONS)
    original_questions = load_gsm8k_questions(N_QUESTIONS)
    n_loaded = len(original_questions)
    _log.info("Loaded %d questions.", n_loaded)

    # --- Step 2: Build adversarial variants ---
    _log.info("Building adversarial variants (seed=%d)...", SEED)
    adversarial_questions = build_adversarial_questions(original_questions, seed=SEED)
    n_adversarial = len(adversarial_questions)
    _log.info("Built %d adversarial variants.", n_adversarial)

    # --- Step 3: Validate round-trip ---
    _log.info("Validating harness round-trip...")
    validation = _validate_round_trip(adversarial_questions)
    if not validation["all_ok"]:
        _log.error(
            "Harness round-trip validation FAILED: %d questions failed check.",
            validation["n_failed"],
        )
    else:
        _log.info("Harness round-trip validation passed (%d/%d ok).",
                  validation["n_passed"], validation["n_checked"])

    # --- Step 4: Sample adversarial question for artifact ---
    sample_aq = adversarial_questions[0] if adversarial_questions else None
    sample_adversarial_question = None
    if sample_aq is not None:
        sample_adversarial_question = {
            "question_id": sample_aq.question_id,
            "original_question": sample_aq.original_question[:200],  # truncate for readability
            "adversarial_question": sample_aq.adversarial_question[:250],
            "ground_truth_answer": sample_aq.ground_truth_answer,
            "irrelevant_sentence": sample_aq.irrelevant_sentence,
        }

    # --- Step 5: Build artifact ---
    artifact = tmpl.build_result(
        {
            "schema": "carnot.adversarial_harness.v1",
            "n_questions_prepared": n_loaded,
            "n_adversarial_prepared": n_adversarial,
            "seed": SEED,
            "sample_adversarial_question": sample_adversarial_question,
            "validation": validation,
            "harness_ready": validation["all_ok"],
            "next_step": (
                "Exp 355: Run live inference on standard and adversarial question sets "
                "with CARNOT_FORCE_LIVE=1 and a real GPU model."
            ),
        },
        status="success" if validation["all_ok"] else "error",
    )

    # --- Write artifact ---
    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", output_path)
    _log.info(
        "Harness ready: %s | %d original + %d adversarial questions prepared.",
        validation["all_ok"],
        n_loaded,
        n_adversarial,
    )


if __name__ == "__main__":
    main()
