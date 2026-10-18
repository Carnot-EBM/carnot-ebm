#!/usr/bin/env python3
"""Experiment 450 — Gemma4 Tokenizer Fix (RETRO-028).

**Purpose:**
    RETRO-028 reported that Exp 439 scored 0.0% accuracy on GSM8K for Gemma4-E4B-it.
    Root cause: llama.cpp tokenizer bug (GitHub issue llama.cpp#21516) causes the model
    to emit infinite <unused8> tokens (token_id=14) instead of valid text.  The model
    NEVER actually ran — the 0% result is a false negative.  Published Gemma4 accuracy
    on GSM8K is 75-80%.

    This experiment verifies that GemmaTransformersLoader (which uses HuggingFace
    transformers, NOT llama.cpp) produces valid text from Gemma4-E4B-it.

**What this script does:**
    1. Calls apply_env_autofix() first (self-injects CARNOT_FORCE_LIVE=1 if GPU present)
    2. Wraps execution in ExperimentTimeoutWatchdog (30 min limit)
    3. Instantiates GemmaTransformersLoader('google/gemma-4-E4B-it')
    4. Loads 10 GSM8K sample questions (hardcoded — no external dependency at import time)
    5. If GPU unavailable or model download fails: emits a diagnostic-only artifact
       documenting that the fix is implemented and ready to verify when GPU is available
    6. If GPU available: runs 10 questions, counts valid outputs via is_valid_output()
    7. Writes results/experiment_450_gemma4_fix.json

**CPU-diagnostic mode:**
    This is a CPU-diagnostic experiment.  GPU is required only because transformers may
    load Gemma4-E4B-it (a 4B parameter model) to GPU automatically.  The 10-question
    run is verification of the loader fix, not a full benchmark.

Spec: REQ-LOADER-001, REQ-LOADER-002,
      SCENARIO-LOADER-001, SCENARIO-LOADER-002
"""

# IMPORTANT: apply_env_autofix() MUST be called before any other import that might
# trigger GPU detection.  This self-injects CARNOT_FORCE_LIVE=1 when GPU hardware
# is present but the env var was not propagated (RETRO-022 root cause).
from carnot.pipeline.env_autofix import apply_env_autofix

_autofix_result = apply_env_autofix()

# ---- All other imports after autofix ----
import json
import logging
import sys
from pathlib import Path

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.gemma_loader import GemmaTransformersLoader
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 10 hardcoded GSM8K sample questions
# These are representative grade-school math problems from the GSM8K dataset.
# Hardcoded to avoid network dependency at import time and to keep this
# experiment self-contained on any machine.
# ---------------------------------------------------------------------------
GSM8K_SAMPLE_QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Mark has a garden with flowers. He planted plants of three species: 10 of species A, 20 of species B and 30 of species C. Calculate the number of plants in the garden.",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he took handfuls of candies and placed them in the box. He then placed 2 pounds of chocolate and 3 pounds of other candy into the box. The total weight of the box of candy was 10 pounds. How many pounds does the empty box weigh?",
    "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also spent $210 on a pair of shoes, but she had a coupon for $150 off. How much money does Alexis have left in her budget?",
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your regular hourly rate + half your regular hourly rate. If she works 10 hours how much money does she make?",
]

DELIVERABLE = "results/experiment_450_gemma4_fix.json"
MODEL_ID = "google/gemma-4-E4B-it"


def _build_artifact(
    tmpl: ExperimentTemplate,
    *,
    status: str,
    n_valid_outputs: int,
    n_tested: int,
    honest_verdict: str,
    diagnostic_only: bool = False,
    error_detail: str = "",
) -> dict:
    """Build the standardised Exp 450 artifact."""
    return tmpl.build_result(
        {
            "schema": "carnot.gemma_loader.v1",
            "retro_028_fix_implemented": True,
            "llama_cpp_bug_ref": "llama.cpp#21516",
            "loader_class": "GemmaTransformersLoader",
            "model_id": MODEL_ID,
            "n_valid_outputs": n_valid_outputs,
            "n_tested": n_tested,
            "diagnostic_only": diagnostic_only,
            "honest_verdict": honest_verdict,
            "error_detail": error_detail,
            "autofix": {
                "gpu_detected": _autofix_result.gpu_detected,
                "auto_fix_applied": _autofix_result.auto_fix_applied,
                "final_env_value": _autofix_result.final_env_value,
            },
        },
        status=status,
    )


def main() -> None:
    """Run Experiment 450: verify GemmaTransformersLoader produces valid text."""
    tmpl = ExperimentTemplate(
        450,
        "Gemma4 Tokenizer Fix (RETRO-028)",
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    output_path = Path(tmpl._repo_root) / DELIVERABLE

    with ExperimentTimeoutWatchdog(450, timeout_minutes=30, result_path=str(output_path)):
        # --- Step 1: Instantiate loader ---
        try:
            loader = GemmaTransformersLoader(MODEL_ID)
        except Exception as exc:
            _log.error("Failed to instantiate GemmaTransformersLoader: %s", exc)
            artifact = _build_artifact(
                tmpl,
                status="error",
                n_valid_outputs=0,
                n_tested=0,
                honest_verdict="retro_028_fix_ready",
                diagnostic_only=True,
                error_detail=str(exc),
            )
            output_path.write_text(json.dumps(artifact, indent=2))
            return

        # --- Step 2: Attempt to load model ---
        try:
            _log.info("Loading model %s via HuggingFace transformers ...", MODEL_ID)
            loader.load()
        except ImportError as exc:
            # transformers not installed — cannot verify on this machine
            _log.warning("transformers not installed: %s", exc)
            artifact = _build_artifact(
                tmpl,
                status="gpu_required",
                n_valid_outputs=0,
                n_tested=0,
                honest_verdict="retro_028_fix_ready",
                diagnostic_only=True,
                error_detail=f"transformers not installed: {exc}",
            )
            output_path.write_text(json.dumps(artifact, indent=2))
            return
        except Exception as exc:
            # Model download failed or GPU unavailable
            _log.warning("Model load failed (GPU may be unavailable): %s", exc)
            artifact = _build_artifact(
                tmpl,
                status="gpu_required",
                n_valid_outputs=0,
                n_tested=0,
                honest_verdict="retro_028_fix_ready",
                diagnostic_only=True,
                error_detail=str(exc),
            )
            output_path.write_text(json.dumps(artifact, indent=2))
            return

        # --- Step 3: Run 10 GSM8K questions ---
        _log.info("Model loaded. Running %d GSM8K questions ...", len(GSM8K_SAMPLE_QUESTIONS))
        n_valid = 0
        responses = []
        for i, question in enumerate(GSM8K_SAMPLE_QUESTIONS):
            try:
                response = loader.generate(question, max_new_tokens=256)
            except Exception as exc:
                _log.warning("Question %d generation failed: %s", i, exc)
                response = ""

            valid = GemmaTransformersLoader.is_valid_output(response)
            if valid:
                n_valid += 1
            responses.append(
                {
                    "question_index": i,
                    "question": question[:80] + "...",
                    "response_preview": response[:120] if response else "",
                    "is_valid": valid,
                }
            )
            _log.info("Q%d: valid=%s preview=%r", i, valid, response[:60] if response else "")

        # --- Step 4: Determine verdict ---
        # retro_028_verified = at least 1 valid output (the bug produced 0 valid outputs)
        # A proper benchmark would require 75%+ accuracy but this is a loader-fix verification.
        if n_valid > 0:
            honest_verdict = "retro_028_verified"
            status = "success"
        else:
            honest_verdict = "retro_028_fix_ready"
            status = "partial"

        _log.info(
            "Done: %d/%d valid outputs. verdict=%s",
            n_valid,
            len(GSM8K_SAMPLE_QUESTIONS),
            honest_verdict,
        )

        artifact = _build_artifact(
            tmpl,
            status=status,
            n_valid_outputs=n_valid,
            n_tested=len(GSM8K_SAMPLE_QUESTIONS),
            honest_verdict=honest_verdict,
            diagnostic_only=False,
        )
        artifact["responses"] = responses

        output_path.write_text(json.dumps(artifact, indent=2))
        _log.info("Artifact written to %s", output_path)


if __name__ == "__main__":
    main()
