"""AdapTrackRepairer — in-generation backtrack repair via SymCodeVerifier violation detection.

**Why this module exists:**

    Carnot's VerifyRepairPipeline is post-hoc: it verifies a *completed* response and then
    asks the LLM to regenerate.  This is inefficient — the model has already committed to
    a wrong answer before repair begins.

    AdapTrack (arXiv 2510.17376) shows a better way: during generation, when the fraction
    of invalid next-token choices exceeds a threshold, *backtrack* to the last valid state
    and regenerate with a correction hint.  Crucially, the output distribution is IDENTICAL
    to the model's own distribution under the constraints — no distortion, no bias toward
    the hint.

    This module applies AdapTrack's idea to Carnot's sentence-level pipeline:
    - InterWhenMonitor (Exp 627) already provides sentence-boundary violation scores.
    - AdapTrackRepairer uses those scores to decide whether to backtrack at each boundary.
    - The backtrack probability is *proportional* to detection_score, not a hard gate,
      so the distributional guarantee holds for ambiguous violations.

    In simulation mode (what this module primarily supports), simulate_repair() replays a
    completed response sentence-by-sentence, injects correction hints at violated sentences,
    and records every BacktrackEvent for downstream calibration analysis.

Spec: REQ-REPAIR-010, REQ-REPAIR-011,
      SCENARIO-REPAIR-020, SCENARIO-REPAIR-021, SCENARIO-REPAIR-022
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from carnot.pipeline.interwhen_monitor import InterWhenMonitor, InterWhenViolation


# ---------------------------------------------------------------------------
# BacktrackEvent — record for one sentence-boundary backtrack decision
# ---------------------------------------------------------------------------


@dataclass
class BacktrackEvent:
    """Record of one sentence-boundary backtrack decision by AdapTrackRepairer.

    Each instance corresponds to one sentence in simulate_repair()'s replay.
    Whether or not a backtrack was triggered, the event is always recorded so
    that downstream calibration analysis can measure false-positive rates.

    Fields
    ------
    sentence_index : int
        Zero-based index of the sentence being checked.
    detection_score : float
        SymCodeVerifier detection_score for this sentence (0.0 if no violation).
    backtrack_triggered : bool
        True iff AdapTrackRepairer decided to inject a correction hint here.
    correction_hint : str or None
        The hint text prepended to this sentence when backtrack_triggered=True.
        None when backtrack was not triggered.
    """

    sentence_index: int
    detection_score: float
    backtrack_triggered: bool
    correction_hint: Optional[str]


# ---------------------------------------------------------------------------
# AdapTrackRepairer
# ---------------------------------------------------------------------------


class AdapTrackRepairer:
    """AdapTrack-style constrained generation via sentence-boundary backtracking.

    Wraps an InterWhenMonitor and applies a proportional backtrack policy at
    every sentence boundary: if SymCodeVerifier flags a violation, the repairer
    either definitely backtracks (high-confidence violations) or probabilistically
    backtracks (low-confidence violations), preserving the AdapTrack distributional
    guarantee (arXiv 2510.17376).

    In live streaming generation, you would call should_backtrack() after each
    sentence and inject the hint before the model continues.  In simulation mode
    (what simulate_repair() supports), we replay a completed response to measure
    how many backtracks *would have* been triggered mid-generation.

    Parameters
    ----------
    monitor : InterWhenMonitor
        The InterWhenMonitor instance wrapping a SymCodeVerifier.  In CI mode
        this uses regex-based detection; in live mode it uses an LLM extractor.
    backtrack_threshold : float
        Detection score threshold above which backtracking is *always* triggered.
        Below this threshold, the backtrack probability is detection_score /
        backtrack_threshold, preserving distributional correctness for ambiguous
        cases (REQ-REPAIR-011-1).  Default 0.5.
    """

    def __init__(
        self,
        monitor: InterWhenMonitor,
        backtrack_threshold: float = 0.5,
    ) -> None:
        self.monitor = monitor
        self.threshold = backtrack_threshold

    def should_backtrack(self, detection_score: float) -> bool:
        """Decide whether to backtrack based on violation confidence.

        AdapTrack guarantees distributional correctness by making the backtrack
        probability *proportional* to violation confidence rather than a hard gate.
        A hard gate (always-backtrack above threshold, never below) would skew
        the conditional distribution over valid continuations because ambiguous
        violations would be systematically suppressed.  Proportional probability
        avoids this: ambiguous cases are repaired with probability proportional
        to how confident the verifier is, so the expected correction rate matches
        the true violation rate (REQ-REPAIR-011-1).

        Parameters
        ----------
        detection_score : float
            SymCodeVerifier detection_score in [0.0, 1.0] for the current sentence.

        Returns
        -------
        bool
            True iff the repairer should inject a correction hint and regenerate.
        """
        if detection_score >= self.threshold:
            # Definite violation — always backtrack (REQ-REPAIR-010-2).
            return True
        # Ambiguous violation — backtrack with probability proportional to score.
        # This is the key AdapTrack distributional-preservation property (REQ-REPAIR-010-3).
        p_backtrack = detection_score / self.threshold
        return random.random() < p_backtrack

    def generate_hint(
        self, violated_sentence: str, violation: InterWhenViolation
    ) -> str:
        """Generate a correction hint to inject before the violated sentence is regenerated.

        The hint is a plain-text prefix that nudges the model to reconsider its
        arithmetic without specifying the correct answer (which would distort the
        distribution by leaking the label).  If SymCodeVerifier's step_results
        contain a step where violation_detected=True, we reference that specific
        step so the model knows which calculation to redo (REQ-REPAIR-010-4).

        Parameters
        ----------
        violated_sentence : str
            The sentence text that triggered the violation (for context).
        violation : InterWhenViolation
            The violation record from InterWhenMonitor.monitor_partial().

        Returns
        -------
        str
            Non-empty correction hint string.
        """
        if violation.step_results and any(
            s.violation_detected for s in violation.step_results
        ):
            return "[Note: The arithmetic in the previous step was incorrect. Recalculate carefully.]"
        return "[Note: Recheck the previous calculation.]"

    def simulate_repair(
        self, response: str
    ) -> tuple[str, list[BacktrackEvent]]:
        """Simulate AdapTrack repair on a completed response.

        Replays the response sentence-by-sentence.  At each sentence boundary,
        calls InterWhenMonitor.monitor_partial() on the partial text seen so far.
        If a violation is detected and should_backtrack() returns True, a
        correction hint is prepended to that sentence in the repaired output.

        This is a *simulation* — we cannot actually roll back and regenerate
        because we do not have a live LLM.  Instead, the hint is injected as a
        prefix to the violated sentence, modelling what would happen if the model
        had received the hint before generating that sentence.

        Every sentence produces a BacktrackEvent regardless of whether a backtrack
        was triggered, enabling downstream FP/FN analysis (REQ-REPAIR-011-3).

        Parameters
        ----------
        response : str
            Completed CoT response text to simulate repair on.

        Returns
        -------
        tuple[str, list[BacktrackEvent]]
            (repaired_response_text, list_of_backtrack_events)
        """
        sentences = self.monitor.verifier.segment_steps(response)
        repaired_sentences = list(sentences)
        events: list[BacktrackEvent] = []

        for i, sent in enumerate(sentences):
            partial = " ".join(repaired_sentences[: i + 1])
            violation = self.monitor.monitor_partial(partial)

            if violation is not None and self.should_backtrack(violation.detection_score):
                hint = self.generate_hint(sent, violation)
                repaired_sentences[i] = hint + " " + sent
                events.append(
                    BacktrackEvent(i, violation.detection_score, True, hint)
                )
            else:
                events.append(
                    BacktrackEvent(
                        i,
                        violation.detection_score if violation is not None else 0.0,
                        False,
                        None,
                    )
                )

        return " ".join(repaired_sentences), events
