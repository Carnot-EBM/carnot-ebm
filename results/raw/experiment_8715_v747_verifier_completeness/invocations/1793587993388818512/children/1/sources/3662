"""V2 resume-and-extend logic for the P0.1 generation corpus (exp3459).

**Why this module exists (plain-language summary):**
    The single most important test in the project — P0.1, "does energy-based
    selection/voting BEAT plain token-sampling self-consistency at equal
    compute?" — needs a HEADLINE-eligible sample of generated GSM8K solutions
    before its downstream crux (exp3460) can report a real (not merely
    preliminary) trained-energy verdict.

    Exp 3448 built the first, RESUMABLE half of that corpus: it generated
    candidate solutions from the SOTA GGUF and appended them to
    ``data/p01_gsm8k_generations.jsonl`` one completed problem at a time, then
    exited clean on its ~18-minute wall-time budget with ``n=47/120`` problems.
    That clean exit was by design — a partial corpus is progress, not failure —
    so the builder is meant to be RE-INVOKED across milestones until the corpus
    reaches the target.

    Exp 3459 is that re-invocation (v2). The heavy, GPU-bound generation +
    resume machinery is already implemented and unit-tested in
    :mod:`carnot.phase3.p01_generation_corpus` (the ``completed_problem_ids``
    resume contract, the per-problem row shape, the warm-up self-consistency
    self-check). This module holds ONLY the three pure pieces that differ for
    the v2 extend run, so a reviewer (or CI) can exercise every scientific
    decision without loading a 26B model:

      * the v2 terminal-verdict band — sharpened so ``>=80`` problems is
        explicitly labelled HEADLINE-eligible (the threshold exp3460 needs),
        ``>=120`` is complete, and below 80 is an extended-partial that resumes
        again next milestone;
      * the v2 acceptance gates — G1 CORPUS-NOT-REGRESSED (the corpus did not
        shrink and still carries the logprobs the scorer needs) and G2
        HEADLINE-ELIGIBLE (``>=80`` problems); and
      * ``added_this_run`` — how many problems THIS invocation newly generated,
        distinct from the running total, so the artifact proves the resume
        actually did new work rather than re-counting finished problems.

    Keeping this logic GPU-free is exactly the reproducibility property the
    adversarial-verify discipline asks for: the verdict the live run reports is
    computed by code a test can drive deterministically.

Spec: REQ-KONA-3459 (resume-and-extend the P0.1 corpus toward n=120, v2),
SCENARIO-KONA-3459, SCENARIO-KONA-3459-RESUME-MONOTONE.
"""

from __future__ import annotations

# The full-corpus target the extend run accumulates toward, and the threshold at
# which the corpus is large enough for exp3460 to report a HEADLINE (not merely
# preliminary) trained-energy verdict. Both are module-level so a test can read
# the same constants the experiment script uses.
DEFAULT_N_TARGET = 120
HEADLINE_ELIGIBLE_FLOOR = 80
# The corpus that exp3448 already landed. G1 (corpus-not-regressed) asserts the
# extend run never shrinks the corpus below what exp3448 produced — the resume
# contract is monotone.
EXP3448_CORPUS_FLOOR = 47


def derive_extend_verdict(n_completed: int, n_target: int = DEFAULT_N_TARGET) -> str:
    """Map (completed, target) to exactly one terminal `complete:`-prefixed verdict.

    A partial corpus is a terminal SUCCESS, not a failure — the whole point of
    the resumable builder is that it accumulates across milestones. The v2 bands
    are sharpened relative to exp3448's so the downstream crux can tell whether
    the sample is headline-eligible:

      * ``n_completed >= n_target``        -> corpus complete.
      * ``HEADLINE_ELIGIBLE_FLOOR <= n < target`` -> headline-eligible: exp3460
        can report a headline trained-energy verdict, not just a preliminary one.
      * ``n_completed < HEADLINE_ELIGIBLE_FLOOR`` -> extended-partial; resume
        again next milestone.

    Every branch is `complete:`-prefixed so the conductor reconciler classifies
    the run as terminal regardless of how far the corpus got (Verdict
    Terminal-Prefix Discipline).
    """
    if n_completed >= n_target:
        return f"complete: p01_generation_corpus_complete_n={n_completed}"
    if n_completed >= HEADLINE_ELIGIBLE_FLOOR:
        return f"complete: p01_generation_corpus_headline_eligible_n={n_completed}"
    return (
        f"complete: p01_generation_corpus_extended_partial_n={n_completed}"
        f"_resume_next_milestone"
    )


def extend_acceptance_gates(
    n_completed: int,
    per_sample_logprobs_captured: bool,
    *,
    corpus_floor: int = EXP3448_CORPUS_FLOOR,
    headline_floor: int = HEADLINE_ELIGIBLE_FLOOR,
) -> dict[str, bool]:
    """Compute the two v2 acceptance gates as named booleans.

    * **G1 CORPUS-NOT-REGRESSED** — ``n_completed >= corpus_floor AND
      per_sample_logprobs_captured``. The corpus did not shrink below what
      exp3448 produced and still carries the mean-token logprobs the scoring task
      (exp3460) consumes; resume is monotone. Both halves must hold: a corpus
      that kept the row count but dropped the logprobs is just as broken for the
      scorer as one that shrank.
    * **G2 HEADLINE-ELIGIBLE** — ``n_completed >= headline_floor``. At least 80
      problems lets exp3460 report a headline (not preliminary) verdict.

    Returned as a dict so the experiment artifact can record each gate by name
    with its own principle annotation.
    """
    g1 = (n_completed >= corpus_floor) and bool(per_sample_logprobs_captured)
    g2 = n_completed >= headline_floor
    return {"g1_corpus_not_regressed": g1, "g2_headline_eligible": g2}


def added_this_run(n_total: int, n_prior: int) -> int:
    """Problems newly generated THIS invocation: ``max(0, n_total - n_prior)``.

    ``n_total`` is the completed-problem count after this run; ``n_prior`` is the
    count that was already on disk when the run started. Clamped at zero so a
    re-count anomaly (e.g. a stale row filtered out of the post-run total) can
    never report a negative "added" count, which would falsely read as the corpus
    having regressed. This is the field that proves the resume actually produced
    new work, distinct from the running total.
    """
    return max(0, n_total - n_prior)
