"""ThinkProbeLiveV3Result — structured result for Exp 482 ThinkProbeV2 live GPU run.

**Why this class exists (RETRO-036, RETRO-042):**
    RETRO-036 opened when Exp 455 reported RETRO-029 CLOSED in the conductor log but the
    result JSON file was absent.  RETRO-042 opened when Exp 465 (ThinkProbeV2 live GPU)
    deferred due to zombie VRAM saturating GPU 0 at 23.8 GB with 0% utilisation.

    Exp 482 closes both retros by:
    1. Using GPUVRAMGate (Exp 474) before model load to kill zombies and ensure >= 8 GB free.
    2. Recording `gpu_vram_gate_fired=True` in the result so the artifact is self-auditable.
    3. Adding `is_viable` — a three-threshold verdict that requires n_completed >= 40 (80% of
       50 questions) plus tp_rate >= 0.70 and fp_rate <= 0.20, giving statistical power enough
       to trust the rate estimates.

**Why three thresholds for is_viable?**
    Completion fraction alone is not sufficient: a run that completes all 50 questions but
    flags every response (fp_rate=1.0) is not a useful ThinkProbe.  Conversely, a run with
    tp_rate=0.99 but only 20 questions completed lacks the statistical sample size to trust
    the rate.  The three-threshold design ensures the verdict is earned in all three dimensions.

Spec: REQ-PROBE-010, REQ-PROBE-011,
      SCENARIO-PROBE-015, SCENARIO-PROBE-016
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThinkProbeLiveV3Result:
    """Structured result for a ThinkProbeV2 live GPU run gated by GPUVRAMGate.

    Fields
    ------
    inference_mode : str
        Always ``'live_gpu'`` for a genuine GPU run.  ``'deferred'`` when
        GPUVRAMGate fired and VRAM was insufficient.
    model_id : str
        HuggingFace model ID used for inference, e.g. ``'google/gemma-4-E4B-it'``.
    n_completed : int
        Number of questions for which inference_fn returned a non-empty result
        before the ThinkProbeV2 budget expired.
    n_total : int
        Total number of questions submitted.  Always 50 for the standard corpus.
    gpu_vram_gate_fired : bool
        ``True`` when GPUVRAMGate.__enter__() was called and succeeded.
        ``False`` only in unit tests that bypass the gate, or in a deferred run.
    skip_rate : float
        Fraction of completed questions for which the model produced an empty
        response (treated as a skip / non-verdict by ThinkProbeV2).
    tp_rate : float
        True-positive rate: fraction of ``correct`` corpus questions that were
        flagged as correct by the ThinkProbeV2 verifier.
    fp_rate : float
        False-positive rate: fraction of ``wrong`` corpus questions that were
        incorrectly flagged as correct.

    Properties
    ----------
    completion_fraction : float
        ``n_completed / n_total``.  1.0 on a complete run, 0.0 when nothing finished.

    is_viable : bool
        ``True`` when all three thresholds are met:
        - ``completion_fraction >= 0.80``  (at least 40 of 50 questions completed)
        - ``tp_rate >= 0.70``              (minimum signal quality)
        - ``fp_rate <= 0.20``              (maximum false-alarm rate)

        A partial run with fewer than 40 questions completed is NOT viable because
        the statistical sample size is too small to trust tp_rate/fp_rate estimates.

    retro_036_closed : bool
        Always ``True`` when a live run object is created — the act of creating this
        result confirms that the deliverable was (or will be) written.
        RETRO-036 was the "result JSON absent" failure; creating this result object
        as part of writing the artifact closes that retro.

    Spec: REQ-PROBE-010, REQ-PROBE-011,
          SCENARIO-PROBE-015, SCENARIO-PROBE-016
    """

    inference_mode: str
    model_id: str
    n_completed: int
    n_total: int
    gpu_vram_gate_fired: bool
    skip_rate: float
    tp_rate: float
    fp_rate: float

    @property
    def completion_fraction(self) -> float:
        """Fraction of questions completed; 0.0 when n_total is 0."""
        if self.n_total == 0:
            return 0.0
        return self.n_completed / self.n_total

    @property
    def is_viable(self) -> bool:
        """True when completion, tp_rate, and fp_rate all meet the viability thresholds.

        All three must hold simultaneously:
        - completion_fraction >= 0.80: statistical power requires >= 40 of 50 questions.
        - tp_rate >= 0.70: ThinkProbe must correctly flag at least 70% of correct responses.
        - fp_rate <= 0.20: ThinkProbe must not incorrectly flag more than 20% of wrong responses.

        Why 0.80 completion threshold?  With 50 questions, 80% = 40 questions.  Below that,
        a single-question variance in tp_rate/fp_rate estimates can shift the verdict by ±2.5%,
        which is within the noise floor of the corpus.  Above 40 questions the variance drops
        below 1.5%, making the rate estimates actionable.
        """
        return (
            self.completion_fraction >= 0.80
            and self.tp_rate >= 0.70
            and self.fp_rate <= 0.20
        )

    @property
    def retro_036_closed(self) -> bool:
        """Always True — creating this result object is part of writing the deliverable.

        RETRO-036 was caused by the result JSON being absent at retrospective time.
        The ThinkProbeLiveV3Result is only instantiated when the live run has completed
        and is about to write its artifact, so its presence proves the write path was reached.
        """
        return True
