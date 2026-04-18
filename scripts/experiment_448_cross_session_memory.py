#!/usr/bin/env python3
"""Experiment 448: Cross-session Tier 2 constraint memory relay validation.

**Research question:**
    Do constraint templates learned in Session N persist to Session N+1 and
    reduce false positive (FP) rate on the same error domain?

**Why this matters:**
    Within-session learning (Exp 361, SelfLearningRelay) resets to zero when the
    process exits.  Tier 2 of the self-learning architecture promises "constraint
    memory across sessions" — but until Exp 448, this was never validated end-to-end
    across simulated process boundaries.  This experiment is the first to confirm
    that carry_check templates activated in Session 1 are PRESENT (pre-loaded) at
    the start of Session 2.

**Design:**
    Three simulated sessions, each processing 50 synthetic arithmetic questions
    with known carry-error patterns.

    Session 1 (no prior memory): starts with no accumulated template observations.
    After processing 50 questions, carry_check has been observed 50 times (well above
    min_frequency=5).  Template library is saved to disk.

    Session 2 (loads Session 1 memory): carry_check is ALREADY ACTIVE when the first
    question arrives.  This means carry-error questions are flagged immediately rather
    than needing 5 warm-up questions.  FP rate may differ from Session 1.

    Session 3 (loads Session 2 memory): further accumulation.  May plateau if the
    template is already maximally active.

    Honest verdict: compute_relay_verdict(sessions[0:2]) tells us whether Session 2
    actually improved over Session 1.  CPU-only; always produces a result.

**Spec:** REQ-LEARN-037, REQ-LEARN-038,
          SCENARIO-LEARN-066, SCENARIO-LEARN-067, SCENARIO-LEARN-068
"""

from __future__ import annotations

import json
import pathlib
import sys

# Add project root to path so imports work when run directly.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts.experiment_template import ExperimentTemplate

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.cross_session_relay import (
    CrossSessionResult,
    compute_relay_verdict,
    simulate_session,
)

# ---------------------------------------------------------------------------
# Step 1: apply env autofix (MUST be first)
# ---------------------------------------------------------------------------
_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP_ID = 448
RESULT_PATH = "results/experiment_448_cross_session_memory.json"
N_QUESTIONS_PER_SESSION = 50
N_SESSIONS = 3


# ---------------------------------------------------------------------------
# Synthetic question generation
# ---------------------------------------------------------------------------


def _generate_arithmetic_questions(n: int) -> list[str]:
    """Generate n synthetic GSM8K-style arithmetic questions with carry errors.

    **Why we embed carry errors:**
        The carry_check template only fires when it finds "A × B = C" where
        at least one factor > 9 AND the claimed product is wrong.  We inject
        intentional carry errors in half the questions so there is a measurable
        FP rate to track across sessions.

        Questions 0..n//2-1: CORRECT products — pipeline should NOT flag these
        (if it does, that's a real FP from the base extractor).
        Questions n//2..n-1: WRONG products (carry error) — pipeline WITH
        carry_check active SHOULD flag these.

    Returns:
        List of n question strings alternating correct / carry-error arithmetic.
    """
    questions = []
    for i in range(n):
        a = 10 + (i % 20)  # 10–29 (always >9, ensures carry check fires)
        b = 3 + (i % 7)    # 3–9
        correct = a * b

        if i % 2 == 0:
            # Correct arithmetic — carry check should NOT flag this.
            q = (
                f"Problem {i}: A worker completes {a} tasks per day for {b} days. "
                f"Total tasks: {a} × {b} = {correct}"
            )
        else:
            # Carry error: drop the tens digit of the units column.
            # E.g., 14 × 3 = 42: units digit 4×3=12, we report 2 instead of 42.
            carry_error = (correct % 10) + (b * (a // 10)) * 10
            if carry_error == correct:
                carry_error = correct - 10  # Ensure it's actually wrong
            q = (
                f"Problem {i}: A worker completes {a} tasks per day for {b} days. "
                f"Total tasks: {a} × {b} = {carry_error}"
            )
        questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the three-session cross-session memory relay experiment."""
    # ------------------------------------------------------------------
    # Step 2: watchdog
    # ------------------------------------------------------------------
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=25)
    watchdog.start()

    try:
        _run_experiment()
    finally:
        watchdog.stop()


def _run_experiment() -> None:
    # ------------------------------------------------------------------
    # Step 3: ExperimentTemplate (CPU mode)
    # ------------------------------------------------------------------
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title="Cross-Session Tier 2 Constraint Memory Relay",
        deliverable=RESULT_PATH,
        requires_gpu=False,  # CPU-only; always produces a result
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 4: Generate synthetic questions
    # ------------------------------------------------------------------
    print(f"[Exp {EXP_ID}] Generating {N_QUESTIONS_PER_SESSION} synthetic arithmetic questions...")
    all_questions = _generate_arithmetic_questions(N_QUESTIONS_PER_SESSION * N_SESSIONS)
    session_questions = [
        all_questions[i * N_QUESTIONS_PER_SESSION : (i + 1) * N_QUESTIONS_PER_SESSION]
        for i in range(N_SESSIONS)
    ]
    print(f"[Exp {EXP_ID}] {N_SESSIONS} sessions × {N_QUESTIONS_PER_SESSION} questions = {len(all_questions)} total")

    # ------------------------------------------------------------------
    # Step 5: Session 1 (no prior memory)
    # ------------------------------------------------------------------
    memory_root = str(pathlib.Path(RESULT_PATH).parent / f"exp_{EXP_ID}_session_memory")
    print(f"\n[Exp {EXP_ID}] === Session 1: no prior memory ===")
    r0 = simulate_session(
        session_id=0,
        questions=session_questions[0],
        prior_memory_path=None,
        memory_dir=memory_root,
    )
    print(
        f"  session=0, n_q={r0.n_questions}, fp_rate={r0.fp_rate:.3f}, "
        f"n_templates_active={r0.n_templates_active}, "
        f"n_loaded_from_prior={r0.n_templates_loaded_from_prior}"
    )
    print(f"  carry_check note: after {N_QUESTIONS_PER_SESSION} questions, "
          f"carry_check should be active (obs={N_QUESTIONS_PER_SESSION} > min_freq=5)")

    # ------------------------------------------------------------------
    # Step 6: Session 2 (load Session 1 memory)
    # ------------------------------------------------------------------
    prior_path_0 = str(pathlib.Path(memory_root) / "session_0")
    print(f"\n[Exp {EXP_ID}] === Session 2: load Session 1 memory ===")
    r1 = simulate_session(
        session_id=1,
        questions=session_questions[1],
        prior_memory_path=prior_path_0,
        memory_dir=memory_root,
    )
    print(
        f"  session=1, n_q={r1.n_questions}, fp_rate={r1.fp_rate:.3f}, "
        f"n_templates_active={r1.n_templates_active}, "
        f"n_loaded_from_prior={r1.n_templates_loaded_from_prior}"
    )
    if r1.n_templates_loaded_from_prior > 0:
        print(f"  CONFIRMED: {r1.n_templates_loaded_from_prior} template(s) pre-loaded from Session 1")
    else:
        print("  WARNING: no templates loaded from Session 1 (carry_check may not have activated)")

    # ------------------------------------------------------------------
    # Step 7: Session 3 (load Session 2 memory)
    # ------------------------------------------------------------------
    prior_path_1 = str(pathlib.Path(memory_root) / "session_1")
    print(f"\n[Exp {EXP_ID}] === Session 3: load Session 2 memory ===")
    r2 = simulate_session(
        session_id=2,
        questions=session_questions[2],
        prior_memory_path=prior_path_1,
        memory_dir=memory_root,
    )
    print(
        f"  session=2, n_q={r2.n_questions}, fp_rate={r2.fp_rate:.3f}, "
        f"n_templates_active={r2.n_templates_active}, "
        f"n_loaded_from_prior={r2.n_templates_loaded_from_prior}"
    )

    # ------------------------------------------------------------------
    # Step 8: compute relay verdict
    # ------------------------------------------------------------------
    sessions = [r0, r1, r2]
    honest_verdict = compute_relay_verdict(sessions)
    print(f"\n[Exp {EXP_ID}] honest_verdict = {honest_verdict}")
    print(f"  Session 0 fp_rate={r0.fp_rate:.3f} → Session 1 fp_rate={r1.fp_rate:.3f}")
    if honest_verdict == "cross_session_improvement":
        print("  Result: cross-session relay reduced FP rate — Tier 2 memory confirmed working")
    elif honest_verdict == "no_improvement":
        print("  Result: FP rate did not decrease — templates may not generate violations on this data")
    else:
        print("  Result: insufficient_data (should not happen with 3 sessions)")

    # ------------------------------------------------------------------
    # Step 9: build and save artifact
    # ------------------------------------------------------------------
    session_results = [
        {
            "session_id": r.session_id,
            "n_questions": r.n_questions,
            "fp_rate": r.fp_rate,
            "n_templates_active": r.n_templates_active,
            "n_loaded": r.n_templates_loaded_from_prior,
        }
        for r in sessions
    ]

    artifact = tmpl.build_result(
        {
            "session_results": session_results,
            "honest_verdict": honest_verdict,
            "memory_root": memory_root,
        },
        status="success",
    )
    # Override schema to cross_session_relay.v1
    artifact["schema"] = "carnot.cross_session_relay.v1"
    artifact["session_results"] = session_results
    artifact["honest_verdict"] = honest_verdict

    result_path = pathlib.Path(RESULT_PATH)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n[Exp {EXP_ID}] Result written to {RESULT_PATH}")
    print(f"[Exp {EXP_ID}] Schema: {artifact['schema']}")
    print(f"[Exp {EXP_ID}] Honest verdict: {honest_verdict}")


if __name__ == "__main__":
    main()
