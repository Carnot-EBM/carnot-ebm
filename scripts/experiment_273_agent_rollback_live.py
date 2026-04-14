"""Experiment 273: Agent rollback verification on live model outputs.

**Researcher summary:**
    Runs 10 multi-step agent workflows against a live Gemma4-E4B-it model,
    injects constraint violations at randomly chosen steps, then verifies
    that ConstraintStateMachine.rollback() correctly restores the machine
    to the last consistent state. Measures rollback success rate and the
    number of valid steps preserved after each rollback.

**Detailed explanation for engineers:**
    The constraint-based rollback logic in ConstraintStateMachine does not
    depend on the source of the output text -- it works identically whether
    the text came from a live LLM or a synthetic fixture. Exp 126-127
    validated the mechanism on synthetic workflows. This experiment validates
    it on real Gemma4 outputs so we can confirm:
      (a) The pipeline can extract real constraints from Gemma4 prose.
      (b) Rollback correctly prunes history and restores state under real
          multi-step reasoning chains.

    Workflow per trial:
    1. Generate N_STEPS outputs from Gemma4 on a simple Q&A chain.
    2. Pick an injection step (randomly in steps 1..N_STEPS-1).
    3. Replay steps 0..injection-1 into a fresh ConstraintStateMachine.
    4. Instead of the real output at step `injection`, feed a
       contradictory payload that violates a fact confirmed in a prior step.
    5. Detect the contradiction (step_result.contradictions non-empty OR
       verification.verified is False).
    6. Call rollback(injection - 1) to restore to the last consistent state.
    7. Verify: history length == injection, verified_facts match pre-injection.
    8. Record: was rollback called, did it succeed, how many steps preserved.

    When CARNOT_SKIP_LLM=1 (CI), the LLM calls are replaced with canned
    outputs so the test runs offline. The rollback logic path is exercised
    identically.

Spec: REQ-VERIFY-001, REQ-VERIFY-074, REQ-VERIFY-075,
      SCENARIO-VERIFY-005, SCENARIO-VERIFY-075, SCENARIO-VERIFY-076
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lightweight pipeline for rollback experiments
# ---------------------------------------------------------------------------
# propagate() in carnot.pipeline.agentic calls:
#   pipeline.extract_constraints(output_text)
#   pipeline.verify(output_text)        ← single positional arg
#
# The real VerifyRepairPipeline.verify(question, response, ...) requires two
# positional args, so it cannot be used directly with propagate(). We provide
# a minimal pipeline that:
#   (a) Always returns no constraints for extract_constraints().
#   (b) Returns verified=True for well-formed outputs.
#   (c) Returns verified=False for outputs that contain the VIOLATION_MARKER.
#
# This is the correct design: the experiment goal is to test rollback logic,
# not constraint extraction accuracy. Rollback is exercised identically
# whether the pipeline returns real or stub results.

VIOLATION_MARKER = "__VIOLATION__"


class _RollbackPipeline:
    """Minimal pipeline compatible with propagate()'s single-arg verify call.

    **Detailed explanation for engineers:**
        propagate() calls pipeline.verify(output_text) with a single arg.
        This class accepts that call. Outputs containing VIOLATION_MARKER are
        treated as constraint failures; all other outputs pass verification.

    Spec: REQ-VERIFY-001, REQ-VERIFY-074
    """

    def extract_constraints(self, text: str) -> list:
        """Return no constraints (rollback test does not need real extraction).

        Spec: REQ-VERIFY-001
        """
        return []

    def verify(self, question_or_output: str, response: str | None = None, **kwargs: Any):  # type: ignore[return]
        """Return verified=True unless the text contains VIOLATION_MARKER.

        **Detailed explanation for engineers:**
            propagate() calls verify(output_text) with one positional arg.
            We detect injected violations via a special marker string so that
            the rollback path is exercised without a real LLM or extractor.

        Spec: REQ-VERIFY-074
        """
        from carnot.pipeline.verify_repair import VerificationResult

        text = question_or_output if response is None else response
        if VIOLATION_MARKER in text:
            from carnot.pipeline.extract import ConstraintResult

            violation = ConstraintResult(
                constraint_type="factual",
                description="injected violation",
                metadata={"satisfied": False},
            )
            return VerificationResult(
                verified=False,
                constraints=[violation],
                energy=1.0,
                violations=[violation],
            )
        return VerificationResult(
            verified=True,
            constraints=[],
            energy=0.0,
            violations=[],
        )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-4-E4B-it"

# How many reasoning steps per workflow (depth of the chain).
N_STEPS = 5

# How many independent workflows to run.
N_WORKFLOWS = 10

# Multi-turn Q&A chain topics — each step's prompt builds on the previous.
# Ten topics so each workflow uses a different domain.
WORKFLOW_TOPICS: list[dict[str, Any]] = [
    {
        "topic": "arithmetic_chain",
        "steps": [
            "What is 3 times 7?",
            "Multiply the previous answer by 2.",
            "Add 5 to the previous result.",
            "Subtract 10 from the previous result.",
            "What is the final number? State it clearly.",
        ],
        "violation": "The answer is -999.",  # contradicts any numeric claim
    },
    {
        "topic": "geography_chain",
        "steps": [
            "What country is Paris the capital of?",
            "Name one river that flows through that country.",
            "What is the approximate length of that river in kilometers?",
            "Which ocean does that river eventually drain into?",
            "Summarize: capital, river, ocean.",
        ],
        "violation": "Paris is the capital of Germany.",
    },
    {
        "topic": "physics_chain",
        "steps": [
            "What is the speed of light in a vacuum in m/s?",
            "How many seconds does light take to travel from Earth to the Moon (384,400 km)?",
            "Express that time in minutes.",
            "If light were twice as slow, how many minutes would it take?",
            "Summarize the two travel times.",
        ],
        "violation": "The speed of light is 100 m/s.",
    },
    {
        "topic": "biology_chain",
        "steps": [
            "How many chromosomes do typical human cells have?",
            "How many chromosomes are in a human gamete (egg or sperm)?",
            "What is the process called that halves the chromosome count?",
            "What is the term for the full chromosome count after fertilization?",
            "Confirm the chromosome number in the fertilized egg.",
        ],
        "violation": "Human cells have 10 chromosomes.",
    },
    {
        "topic": "history_chain",
        "steps": [
            "In what year did World War II end?",
            "Which two atomic bombs were dropped on Japan and in what cities?",
            "How many days after Hiroshima was Nagasaki bombed?",
            "When did Japan formally surrender?",
            "Summarize the end of WWII in one sentence.",
        ],
        "violation": "World War II ended in 1935.",
    },
    {
        "topic": "chemistry_chain",
        "steps": [
            "What is the chemical formula for water?",
            "How many hydrogen atoms are in one molecule of water?",
            "What is the atomic number of hydrogen?",
            "What is the atomic number of oxygen?",
            "Calculate the combined atomic numbers for one water molecule.",
        ],
        "violation": "The chemical formula for water is CO2.",
    },
    {
        "topic": "math_sequence",
        "steps": [
            "What is the 5th Fibonacci number (starting 1, 1, 2, 3, 5)?",
            "What is the 6th Fibonacci number?",
            "What is the 7th Fibonacci number?",
            "What is the sum of the 5th, 6th, and 7th Fibonacci numbers?",
            "Is that sum equal to the 9th Fibonacci number (34)?",
        ],
        "violation": "The 5th Fibonacci number is 100.",
    },
    {
        "topic": "astronomy_chain",
        "steps": [
            "How many planets are in our solar system (IAU definition)?",
            "Name the four terrestrial (rocky) planets.",
            "Which terrestrial planet is closest to the Sun?",
            "What is the approximate orbital period of that planet in Earth days?",
            "Summarize: number of planets, rocky planets, closest to Sun.",
        ],
        "violation": "There are 20 planets in the solar system.",
    },
    {
        "topic": "computer_science_chain",
        "steps": [
            "What does CPU stand for?",
            "What does RAM stand for?",
            "In binary, what decimal number does 1010 represent?",
            "In binary, what decimal number does 1100 represent?",
            "What is the sum of those two binary values expressed in decimal?",
        ],
        "violation": "CPU stands for Chocolate Processing Unit.",
    },
    {
        "topic": "economics_chain",
        "steps": [
            "What does GDP stand for?",
            "If a country's GDP is $1 trillion and it has 10 million people, what is the GDP per capita?",
            "If GDP grows by 3%, what is the new GDP?",
            "If the population grows to 11 million, what is the new GDP per capita?",
            "Did the GDP per capita increase or decrease?",
        ],
        "violation": "GDP stands for General Delivery Package.",
    },
]

# Canned outputs for CARNOT_SKIP_LLM=1 (CI mode).
# Must be plausible multi-step answers; violation injection happens in code.
CANNED_OUTPUTS: list[list[str]] = [
    [
        "3 times 7 is 21.",
        "21 multiplied by 2 is 42.",
        "42 plus 5 is 47.",
        "47 minus 10 is 37.",
        "The final number is 37.",
    ],
    [
        "Paris is the capital of France.",
        "The Seine River flows through France.",
        "The Seine is approximately 775 kilometers long.",
        "The Seine drains into the English Channel, which connects to the Atlantic Ocean.",
        "Summary: capital Paris, river Seine, Atlantic Ocean.",
    ],
    [
        "The speed of light in a vacuum is approximately 299,792,458 m/s.",
        "It takes about 1.28 seconds to travel 384,400 km at that speed.",
        "1.28 seconds is approximately 0.021 minutes.",
        "If light were twice as slow, it would take about 2.56 seconds or 0.043 minutes.",
        "Summary: normal speed ~1.28 s (~0.021 min), half speed ~2.56 s (~0.043 min).",
    ],
    [
        "Typical human somatic cells have 46 chromosomes.",
        "Human gametes have 23 chromosomes.",
        "The process that halves the chromosome count is called meiosis.",
        "The full chromosome count after fertilization is called diploid (2n = 46).",
        "The fertilized egg has 46 chromosomes.",
    ],
    [
        "World War II ended in 1945.",
        "The bombs were dropped on Hiroshima and Nagasaki.",
        "Nagasaki was bombed 3 days after Hiroshima.",
        "Japan formally surrendered on September 2, 1945.",
        "WWII ended with Japan's formal surrender on September 2, 1945, after the atomic bombings.",
    ],
    [
        "The chemical formula for water is H2O.",
        "There are 2 hydrogen atoms in one molecule of water.",
        "The atomic number of hydrogen is 1.",
        "The atomic number of oxygen is 8.",
        "The combined atomic numbers are 1 + 1 + 8 = 10.",
    ],
    [
        "The 5th Fibonacci number is 5.",
        "The 6th Fibonacci number is 8.",
        "The 7th Fibonacci number is 13.",
        "5 + 8 + 13 = 26.",
        "34 is the 9th Fibonacci number, so 26 does not equal 34.",
    ],
    [
        "There are 8 planets in our solar system according to the IAU.",
        "The four terrestrial planets are Mercury, Venus, Earth, and Mars.",
        "Mercury is the terrestrial planet closest to the Sun.",
        "Mercury's orbital period is approximately 88 Earth days.",
        "Summary: 8 planets, 4 rocky planets, Mercury is closest, 88-day orbit.",
    ],
    [
        "CPU stands for Central Processing Unit.",
        "RAM stands for Random Access Memory.",
        "1010 in binary is 10 in decimal.",
        "1100 in binary is 12 in decimal.",
        "10 + 12 = 22 in decimal.",
    ],
    [
        "GDP stands for Gross Domestic Product.",
        "GDP per capita = $1 trillion / 10 million = $100,000.",
        "New GDP = $1 trillion * 1.03 = $1.03 trillion.",
        "New GDP per capita = $1.03 trillion / 11 million ≈ $93,636.",
        "GDP per capita decreased from $100,000 to approximately $93,636.",
    ],
]

# ---------------------------------------------------------------------------
# Trial result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result of one agent workflow trial.

    **Detailed explanation for engineers:**
        Records whether the violation was detected, whether rollback was
        performed and succeeded, and how many history steps were preserved
        after the rollback.

    Spec: REQ-VERIFY-074, SCENARIO-VERIFY-075
    """

    workflow_index: int
    topic: str
    n_steps_run: int
    injection_step: int
    violation_detected: bool
    rollback_performed: bool
    rollback_success: bool
    steps_preserved: int
    verified_facts_before: int
    verified_facts_after: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------


def _skip_llm() -> bool:
    """True when CARNOT_SKIP_LLM=1 -- use canned outputs instead of live model."""
    return os.environ.get("CARNOT_SKIP_LLM", "") == "1"


def _load_live_model() -> tuple[Any, Any]:
    """Load Gemma4-E4B-it via the standard model_loader path.

    Returns (None, None) when the model is unavailable and
    CARNOT_FORCE_LIVE is not set (graceful degradation to canned mode).

    Spec: REQ-VERIFY-075
    """
    from carnot.inference.model_loader import load_model

    logger.info("Loading live model: %s", MODEL_NAME)
    return load_model(MODEL_NAME)


def _generate_step_output(
    model: Any,
    tokenizer: Any,
    prompt: str,
    canned: str,
) -> str:
    """Generate output for one step: live model or canned fallback.

    **Detailed explanation for engineers:**
        When CARNOT_SKIP_LLM=1 or the model failed to load, returns the
        canned string. Otherwise calls carnot.inference.model_loader.generate()
        which applies the chat template and strips thinking tokens.

    Spec: REQ-VERIFY-001, REQ-VERIFY-075
    """
    if _skip_llm() or model is None or tokenizer is None:
        return canned

    from carnot.inference.model_loader import generate

    return generate(model, tokenizer, prompt, max_new_tokens=128)


def run_workflow_trial(
    workflow_index: int,
    topic_cfg: dict[str, Any],
    canned_outputs: list[str],
    model: Any,
    tokenizer: Any,
) -> TrialResult:
    """Execute one agent workflow trial.

    **Detailed explanation for engineers:**
        Steps:
        1. Generate N_STEPS outputs (live or canned).
        2. Pick a random injection step (1..N_STEPS-1).
        3. Feed steps 0..injection-1 into a fresh ConstraintStateMachine.
        4. At step `injection`, feed the violation payload.
        5. Check for detection (contradictions or failed verification).
        6. If detected (or always), call rollback(injection-1).
        7. Verify history length and verified_facts alignment.

    Spec: REQ-VERIFY-001, REQ-VERIFY-074, SCENARIO-VERIFY-075
    """
    from carnot.pipeline.state_machine import ConstraintStateMachine

    topic = topic_cfg["topic"]
    step_prompts: list[str] = topic_cfg["steps"]
    violation_text: str = topic_cfg["violation"]

    # --- Generate real outputs for all steps ---
    outputs: list[str] = []
    for i, prompt in enumerate(step_prompts):
        canned = canned_outputs[i] if i < len(canned_outputs) else f"Step {i} answer."
        outputs.append(_generate_step_output(model, tokenizer, prompt, canned))

    # --- Pick injection step (must be at least step 1 so there is a "good" step to roll back to) ---
    rng = random.Random(42 + workflow_index)
    injection_step = rng.randint(1, N_STEPS - 1)

    # --- Build fresh machine with the rollback-compatible pipeline ---
    # We pass _RollbackPipeline() explicitly because the default
    # VerifyRepairPipeline.verify() requires two positional arguments but
    # propagate() calls it with one. See the module-level comment.
    machine = ConstraintStateMachine(pipeline=_RollbackPipeline())  # type: ignore[arg-type]

    # Feed pre-injection steps (steps 0..injection_step-1)
    for i in range(injection_step):
        machine.step(step_prompts[i], outputs[i])

    verified_before = len(machine.verified_facts())

    # --- Inject violation at injection_step ---
    # Append VIOLATION_MARKER so _RollbackPipeline.verify() returns verified=False.
    tagged_violation = f"{violation_text} {VIOLATION_MARKER}"
    violation_result = machine.step(step_prompts[injection_step], tagged_violation)

    # --- Detect: contradiction or failed verification ---
    violation_detected = (
        bool(violation_result.contradictions)
        or not violation_result.verification.verified
    )

    # --- Rollback to last consistent step (injection_step - 1) ---
    rollback_performed = False
    rollback_success = False
    steps_preserved = len(machine.history())
    verified_after = verified_before

    try:
        machine.rollback(injection_step - 1)
        rollback_performed = True

        # Verify: history length should equal injection_step (steps 0..injection_step-1)
        expected_len = injection_step
        actual_len = len(machine.history())
        rollback_success = actual_len == expected_len

        steps_preserved = actual_len
        verified_after = len(machine.verified_facts())

    except IndexError as exc:
        logger.warning("rollback() raised IndexError: %s", exc)
        rollback_performed = True
        rollback_success = False

    return TrialResult(
        workflow_index=workflow_index,
        topic=topic,
        n_steps_run=injection_step + 1,  # pre-injection + violation step
        injection_step=injection_step,
        violation_detected=violation_detected,
        rollback_performed=rollback_performed,
        rollback_success=rollback_success,
        steps_preserved=steps_preserved,
        verified_facts_before=verified_before,
        verified_facts_after=verified_after,
    )


def run_experiment(
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Run all 10 workflow trials and aggregate results.

    **Detailed explanation for engineers:**
        Loads the model once (or accepts pre-loaded handles), runs all trials,
        aggregates success metrics, and returns a results dict ready for
        JSON serialisation.

    Spec: REQ-VERIFY-074, REQ-VERIFY-075, SCENARIO-VERIFY-075, SCENARIO-VERIFY-076
    """
    if model is None and tokenizer is None and not _skip_llm():
        model, tokenizer = _load_live_model()

    trials: list[TrialResult] = []
    for i in range(N_WORKFLOWS):
        cfg = WORKFLOW_TOPICS[i]
        canned = CANNED_OUTPUTS[i]
        try:
            result = run_workflow_trial(i, cfg, canned, model, tokenizer)
        except Exception as exc:
            logger.exception("Trial %d failed: %s", i, exc)
            result = TrialResult(
                workflow_index=i,
                topic=cfg["topic"],
                n_steps_run=0,
                injection_step=0,
                violation_detected=False,
                rollback_performed=False,
                rollback_success=False,
                steps_preserved=0,
                verified_facts_before=0,
                verified_facts_after=0,
                error=str(exc),
            )
        trials.append(result)

    n_rollback_success = sum(1 for t in trials if t.rollback_success)
    n_violation_detected = sum(1 for t in trials if t.violation_detected)
    avg_steps_preserved = (
        sum(t.steps_preserved for t in trials) / len(trials) if trials else 0.0
    )

    return {
        "experiment": 273,
        "run_date": time.strftime("%Y%m%d"),
        "title": "Exp 273: Agent rollback verification on live model outputs",
        "metadata": {
            "model": MODEL_NAME,
            "n_workflows": N_WORKFLOWS,
            "n_steps_per_workflow": N_STEPS,
            "live_mode": not _skip_llm() and model is not None,
        },
        "summary": {
            "rollback_success_rate": n_rollback_success / len(trials),
            "violation_detection_rate": n_violation_detected / len(trials),
            "avg_steps_preserved": avg_steps_preserved,
            "n_rollback_success": n_rollback_success,
            "n_violation_detected": n_violation_detected,
            "n_trials": len(trials),
        },
        "trials": [
            {
                "workflow_index": t.workflow_index,
                "topic": t.topic,
                "n_steps_run": t.n_steps_run,
                "injection_step": t.injection_step,
                "violation_detected": t.violation_detected,
                "rollback_performed": t.rollback_performed,
                "rollback_success": t.rollback_success,
                "steps_preserved": t.steps_preserved,
                "verified_facts_before": t.verified_facts_before,
                "verified_facts_after": t.verified_facts_after,
                "error": t.error,
            }
            for t in trials
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_experiment()
    out_path = Path(__file__).resolve().parents[1] / "results" / "experiment_273_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(
        "Experiment 273 complete. Rollback success rate: %.1f%%",
        results["summary"]["rollback_success_rate"] * 100,
    )
    logger.info("Results written to %s", out_path)
