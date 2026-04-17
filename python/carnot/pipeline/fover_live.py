"""LiveFOVERResult — result type and artifact builder for Exp 442 live CoT annotation.

**Researcher summary:**
    FR-11 (autonomous self-learning) has been unconfirmed for 8 consecutive milestones
    because all EORM/JEPA retrains used SYNTHETIC data only (honest_verdict=synthetic_only).
    The missing piece is real (step, correct/incorrect) labels from live LLM inference.

    Exp 442 runs FOVERAnnotator on live CoT data collected by Exp 439 (live_gpu inference).
    This module holds the structured result type and the honest-verdict builder for that
    experiment.  It is deliberately separate from the core FOVERAnnotator so the annotation
    engine stays reusable and the verdict logic is testable in isolation.

**Honest verdict semantics (REQ-LEARN-035):**
    - 'real_data_labeled'      — source='live' AND n_labeled >= 20
      (enough real labeled pairs for EORM/JEPA training; FR-11 relay can proceed)
    - 'real_data_insufficient' — source='live' AND n_labeled < 20
      (live data present but too sparse to train; investigate why Z3 labels so few steps)
    - 'synthetic_fallback'     — source='synthetic'
      (Exp 439 data absent or not live_gpu; synthetic CoT used — same as prior milestones)

**Why 20 pairs is the threshold:**
    REQ-LEARN-032 requires n_real_pairs >= 10 for EORM retrain.  We set the FOVER threshold
    at 20 to give the contrastive pair builder room to filter: if 20 labeled steps span
    multiple questions, at least 10 contrastive (correct, incorrect) pairs can be formed
    for training even after the inter-question grouping step.

Spec: REQ-LEARN-035, SCENARIO-LEARN-062, SCENARIO-LEARN-063
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# LiveFOVERResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class LiveFOVERResult:
    """Structured summary of a live FOVER annotation run.

    **Detailed explanation for engineers:**
        All count fields are non-negative integers derived from calling
        FOVERAnnotator on a corpus of CoT responses.

        ``source`` records whether the input came from real GPU inference
        ('live') or synthetic GSM8K generation ('synthetic').  This is
        the primary discriminant for the honest verdict.

        ``labeling_rate`` is the fraction of found steps that received
        a verifiable label (correct OR incorrect).  A low labeling_rate
        signals that most steps lacked arithmetic equations, which is
        expected for natural-language preambles but surprising if it
        approaches 0 on math CoT data.

        ``honest_verdict`` is set by ``build_live_fover_artifact`` — it
        is NOT stored on the result itself so that the artifact builder
        can be the single source of truth for verdict logic.

    Attributes:
        n_responses:       Number of CoT responses processed.
        n_steps_found:     Total steps extracted across all responses.
        n_labeled:         Steps with z3_label in ('correct', 'incorrect')
                           AND z3_confidence >= 0.3 (the training-pair filter).
        n_correct:         Subset of n_labeled with z3_label='correct'.
        n_incorrect:       Subset of n_labeled with z3_label='incorrect'.
        n_not_verifiable:  Steps with z3_label='not_verifiable'.
        labeling_rate:     n_labeled / n_steps_found (0.0 when n_steps_found==0).
        source:            'live' or 'synthetic'.
        honest_verdict:    Filled in by build_live_fover_artifact.

    Spec: REQ-LEARN-035, SCENARIO-LEARN-062, SCENARIO-LEARN-063
    """

    n_responses: int
    n_steps_found: int
    n_labeled: int
    n_correct: int
    n_incorrect: int
    n_not_verifiable: int
    labeling_rate: float
    source: Literal["live", "synthetic"]
    honest_verdict: str


# ---------------------------------------------------------------------------
# build_live_fover_artifact
# ---------------------------------------------------------------------------


def build_live_fover_artifact(result: LiveFOVERResult) -> dict:
    """Build a JSON-serializable artifact for Exp 442 with honest verdict.

    **Detailed explanation for engineers:**
        The honest verdict is the critical output for FR-11.  It answers the question:
        "did we actually produce real labeled training data this run?"

        Verdict logic (single source of truth — do NOT duplicate in the script):
        - source='live' AND n_labeled >= 20  → 'real_data_labeled'
          The EORM/JEPA retrain can now proceed on real data.  FR-11 relay condition met.
        - source='live' AND n_labeled < 20   → 'real_data_insufficient'
          We have live data but too few verifiable arithmetic steps.  Possible causes:
          the model's CoT used prose rather than inline equations, or the regex missed
          a new equation format.  Investigate before claiming FR-11 is resolved.
        - source='synthetic'                 → 'synthetic_fallback'
          Exp 439 data absent or not live_gpu.  Same situation as Exps 430-441.

        The artifact schema is 'carnot.fover_live.v1' to distinguish it from Exp 430's
        'carnot.fover_labels.v1' (which covers synthetic-only annotation output).

    Args:
        result: A completed LiveFOVERResult.

    Returns:
        Dict with keys: schema, honest_verdict, n_responses, n_steps_found,
        n_labeled, n_correct, n_incorrect, n_not_verifiable, labeling_rate, source.

    Spec: REQ-LEARN-035, SCENARIO-LEARN-062, SCENARIO-LEARN-063
    """
    if result.source == "live":
        if result.n_labeled >= 20:
            verdict = "real_data_labeled"
        else:
            verdict = "real_data_insufficient"
    else:
        verdict = "synthetic_fallback"

    return {
        "schema": "carnot.fover_live.v1",
        "honest_verdict": verdict,
        "n_responses": result.n_responses,
        "n_steps_found": result.n_steps_found,
        "n_labeled": result.n_labeled,
        "n_correct": result.n_correct,
        "n_incorrect": result.n_incorrect,
        "n_not_verifiable": result.n_not_verifiable,
        "labeling_rate": result.labeling_rate,
        "source": result.source,
    }
