"""Cross-session Tier 2 relay: constraint templates persist across process restarts.

**Researcher summary (Exp 448):**
    Within-session learning (Exp 361, SelfLearningRelay) is washed out the moment the
    process restarts — the in-memory ConstraintTemplateLibrary resets to empty.
    This is the fundamental gap in Tier 2 (Constraint Memory / Trace2Skill): the
    templates WERE accumulating signal, but they had no place to go.

    This module closes that gap by wiring SessionMemory (Exp 345) to the simulated
    session boundary: templates learned in Session N are serialised to disk, then
    deserialised and injected into Session N+1 BEFORE the first question is processed.
    This means carry_check, sign_check, unit_consistency, and comparison_direction
    templates that activated in Session N start already-active in Session N+1 — they
    do not need to re-accumulate evidence from scratch.

    The expected observable effect: false positive rate in Session N+1 should be
    LOWER than Session N because the active templates provide denser constraint
    coverage, catching real arithmetic errors faster and with fewer spurious FP
    constraints from the base pipeline.

**Why cross-session memory matters (alignment with research-program.md Tier 2):**
    research-program.md Tier 2 goal: "cache verified facts across sessions, learn
    per-user error patterns, consolidate into reusable constraint templates."
    The KEY word is "across sessions." Single-session learning gives you at most
    the improvement from the last N questions in the current run. Cross-session
    memory gives you compound improvement: each new session starts from the
    accumulated state of all prior sessions on the same error domain.

    This is the mechanism by which Carnot can become progressively better at
    verifying arithmetic responses for a given model — not by retraining the
    model itself, but by improving the VERIFIER's coverage of known error patterns.

**Hardware path:**
    Current: CPU + system memory for SessionMemory JSON storage. Fast enough for
    up to ~10,000 template observations before JSON size becomes a bottleneck.

    FPGA path (KV260, arrives 2026-04-20): fast template matching at production
    scale. The template library's observation dict can be mapped onto a hash-
    addressable BRAM structure so that template activation decisions at inference
    time become O(1) lookups rather than Python dict scans.

Spec: REQ-LEARN-037, REQ-LEARN-038,
      SCENARIO-LEARN-066, SCENARIO-LEARN-067, SCENARIO-LEARN-068
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any

from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.case_memory import CaseMemory
from carnot.pipeline.constraint_template_library import (
    CaseMemoryTemplateWiring,
    ConstraintTemplateLibrary,
)
from carnot.pipeline.session_memory import SessionMemory
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# CrossSessionResult
# ---------------------------------------------------------------------------


@dataclass
class CrossSessionResult:
    """Per-session metrics from one round of the cross-session relay.

    **Detailed explanation for engineers:**
        Each call to ``simulate_session()`` produces one of these.
        Together, a list of results forms the evidence base for
        ``compute_relay_verdict()`` — did templates from an earlier session
        actually improve verification in a later session?

        ``session_id``:
            Zero-based index. Session 0 has no prior templates; Session 1
            loaded Session 0's templates; Session 2 loaded Session 1's, etc.

        ``n_questions``:
            How many questions were run in this session.

        ``fp_rate``:
            Fraction of questions where the pipeline flagged a violation
            but the question was syntactically correct (i.e., a false positive).
            In our synthetic setup, all questions with correct arithmetic
            are "true answers" — any violation flagged is an FP.
            Range [0.0, 1.0]. Lower is better.

        ``n_templates_active``:
            Number of constraint templates that were active (above their
            min_frequency threshold for the model) by the END of this session.
            Should increase across sessions as patterns accumulate.

        ``n_templates_loaded_from_prior``:
            Number of templates that were already active at the START of this
            session (loaded from the prior session's saved state). Zero for
            Session 0 (no prior session). Non-zero for Session 1+ confirms
            that cross-session relay is working.

    Spec: REQ-LEARN-037-1
    """

    session_id: int
    n_questions: int
    fp_rate: float
    n_templates_active: int
    n_templates_loaded_from_prior: int


# ---------------------------------------------------------------------------
# simulate_session
# ---------------------------------------------------------------------------

# Synthetic model ID used across all simulated sessions so observation counts
# accumulate correctly across the session boundary.
_RELAY_MODEL_ID = "carnot_relay_synthetic"

# Synthetic violation type that maps to "carry_check" via CaseMemoryTemplateWiring.
# We inject this for every question so that carry_check activates quickly in tests.
_RELAY_VIOLATION_TYPE = "carry_error"


def simulate_session(
    session_id: int,
    questions: list[str],
    prior_memory_path: str | None,
    memory_dir: str,
) -> CrossSessionResult:
    """Run one simulated session of the cross-session Tier 2 relay.

    **Detailed explanation for engineers:**
        Steps performed inside this function:

        1. Restore prior session's ConstraintTemplateLibrary from disk (if
           ``prior_memory_path`` is set).  The restored library has the
           observation counts from the prior session already populated —
           this is what gives Session N+1 a head start.

        2. Count how many templates are already active (above threshold)
           BEFORE processing any new questions.  This is
           ``n_templates_loaded_from_prior``.

        3. Create a VerifyRepairPipeline with the loaded template library
           (or a fresh one if no prior session).

        4. Process each question:
           a. Run pipeline.verify(question, response).
           b. If there are violations, call the CaseMemoryTemplateWiring so
              that observation counts accumulate for future template activation.
           c. Classify the question as FP (false positive) when the pipeline
              flags a violation on an arithmetic-correct response.

        5. Compute FP rate = n_fp / n_questions.

        6. Save the updated ConstraintTemplateLibrary to a session-specific
           SessionMemory path so Session N+1 can load it.

        7. Return CrossSessionResult.

    **What counts as a false positive here:**
        In the synthetic setup, question strings contain valid arithmetic that
        the pipeline may or may not correctly verify.  We run the pipeline in
        verify-only mode (no model loaded) so ``pipeline.verify()`` returns
        based purely on constraint extraction.  An FP is counted when:
        - The pipeline returned ``verified=False`` (found a violation)
        - The question itself contains no intentional arithmetic error.
        In session simulations the "no intentional error" check is implicit:
        all questions generated by the experiment script are arithmetically
        correct, so every violation flagged is an FP.

    Args:
        session_id:         Zero-based session index.
        questions:          List of question strings to process.
        prior_memory_path:  Path to the directory of the previous session's
                            SessionMemory.  If None (Session 0), no prior state
                            is loaded and templates start fresh.
        memory_dir:         Root directory under which this session's state is
                            saved.  A per-session subdirectory is created.

    Returns:
        CrossSessionResult with all per-session metrics populated.

    Spec: REQ-LEARN-037-2, SCENARIO-LEARN-066
    """
    # ------------------------------------------------------------------
    # Step 1: load prior session template library (if available)
    # ------------------------------------------------------------------
    template_library = ConstraintTemplateLibrary()
    template_library.register_builtin_templates()
    n_templates_loaded_from_prior = 0

    if prior_memory_path is not None:
        prior_sm = SessionMemory(storage_dir=prior_memory_path, model_id=_RELAY_MODEL_ID)
        restored = prior_sm.load()
        if restored is not None:
            _prior_cm, prior_lib, _prior_tracker = restored
            # Graft the prior observation counts onto our fresh library
            # (which has callable template functions registered).
            # from_dict gives us the raw observation counts; we then call
            # register_builtin_templates() on that instance to attach callables.
            prior_lib.register_builtin_templates()
            template_library = prior_lib
            n_templates_loaded_from_prior = len(
                template_library.get_active_templates(_RELAY_MODEL_ID)
            )

    # ------------------------------------------------------------------
    # Step 2: count initially-active templates (head start from prior session)
    # ------------------------------------------------------------------
    # Already computed above as n_templates_loaded_from_prior.

    # ------------------------------------------------------------------
    # Step 3: build pipeline with loaded template library
    # ------------------------------------------------------------------
    pipeline = VerifyRepairPipeline(
        model=None,  # verify-only mode — no LLM needed for relay simulation
        template_library=template_library,
    )
    wiring = CaseMemoryTemplateWiring(template_library)

    # ------------------------------------------------------------------
    # Step 4: process questions
    # ------------------------------------------------------------------
    n_fp = 0
    for question in questions:
        # In verify-only mode, use the question itself as the "response" —
        # the extractor will parse any arithmetic expressions it finds.
        # This is identical to the SelfLearningRelay CI pattern (Exp 361).
        result = pipeline.verify(question=question, response=question)

        # Apply active templates directly: VerifyRepairPipeline only invokes
        # template_library.apply_active_templates() when model_name is not None
        # (a real model is loaded).  In our verify-only simulation we have no
        # loaded model, so we call apply_active_templates() directly here.
        # This is the mechanism that makes Session N+1 benefit from Session N's
        # accumulated template observations — the templates fire on the response
        # text and may produce additional constraint violations.
        template_constraints = template_library.apply_active_templates(
            question, _RELAY_MODEL_ID
        )
        template_violated = any(
            c.metadata.get("satisfied") is False for c in template_constraints
        )

        # Accumulate pattern observations: every question gets a carry_error
        # observation so that the carry_check template activates quickly.
        # In a real system this would come from actual violation types.
        wiring.on_violation_recorded(_RELAY_VIOLATION_TYPE, _RELAY_MODEL_ID)

        # Count FP: pipeline OR template flagged a violation.
        if not result.verified or template_violated:
            n_fp += 1

    # ------------------------------------------------------------------
    # Step 5: compute FP rate
    # ------------------------------------------------------------------
    n_questions = len(questions)
    fp_rate = n_fp / n_questions if n_questions > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 6: count active templates at end of session
    # ------------------------------------------------------------------
    n_templates_active = len(template_library.get_active_templates(_RELAY_MODEL_ID))

    # ------------------------------------------------------------------
    # Step 7: save updated template library to disk for next session
    # ------------------------------------------------------------------
    session_memory_dir = str(pathlib.Path(memory_dir) / f"session_{session_id}")
    sm = SessionMemory(storage_dir=session_memory_dir, model_id=_RELAY_MODEL_ID)
    case_memory = CaseMemory()
    fp_tracker = PerModelFPTracker()
    sm.save(case_memory, template_library, fp_tracker)

    return CrossSessionResult(
        session_id=session_id,
        n_questions=n_questions,
        fp_rate=fp_rate,
        n_templates_active=n_templates_active,
        n_templates_loaded_from_prior=n_templates_loaded_from_prior,
    )


# ---------------------------------------------------------------------------
# compute_relay_verdict
# ---------------------------------------------------------------------------


def compute_relay_verdict(sessions: list[CrossSessionResult]) -> str:
    """Compute whether cross-session template relay reduced the FP rate.

    **Detailed explanation for engineers:**
        The relay verdict answers: did the templates carried forward from
        Session N actually help Session N+1?

        We compare sessions[0].fp_rate (Session 0 — no prior templates) to
        sessions[1].fp_rate (Session 1 — loaded Session 0 templates).
        If Session 1's FP rate is strictly lower, the relay is working.

        Verdict meanings:
        - ``"cross_session_improvement"``: Session 1 FP rate < Session 0 FP rate.
          This is the primary success criterion for REQ-LEARN-037.
        - ``"no_improvement"``: Session 1 FP rate >= Session 0 FP rate.
          Templates were loaded but did not reduce FP rate — could indicate
          the test questions don't trigger the templates, or that the templates
          increase rather than decrease violations on this data.
        - ``"insufficient_data"``: Fewer than 2 sessions provided.
          Cannot compare without a before/after pair.

        **Why STRICT less-than:**
            An equal FP rate means no observable benefit from template loading.
            We only claim improvement when the number is measurably lower.

    Args:
        sessions: List of CrossSessionResult from simulate_session() calls,
                  ordered by session_id (sessions[0] is the first session,
                  sessions[1] is the second, etc.).

    Returns:
        One of "cross_session_improvement", "no_improvement", or
        "insufficient_data".

    Spec: REQ-LEARN-037-3, SCENARIO-LEARN-068
    """
    if len(sessions) < 2:
        # Cannot determine improvement without at least two sessions.
        return "insufficient_data"

    if sessions[1].fp_rate < sessions[0].fp_rate:
        return "cross_session_improvement"
    return "no_improvement"


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "CrossSessionResult",
    "simulate_session",
    "compute_relay_verdict",
]
