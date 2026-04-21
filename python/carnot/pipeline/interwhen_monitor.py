"""InterWhenMonitor — mid-generation violation detection via sentence-boundary interception.

**Why this module exists (RETRO-070):**

    15 consecutive Verify-Repair (VR) attempts achieved 0% improvement because
    post-hoc extraction of completed responses finds only 0–4% of arithmetic
    violations.  The root cause: IT-tuned models bury arithmetic in natural
    language prose, so sentence-by-sentence scanning of the *finished* response
    finds almost nothing.

    arXiv 2602.11202 (Interwhen) demonstrated +15 pp accuracy by checking
    intermediate solutions DURING generation.  This module implements that idea:
    instead of scanning a completed response once, we simulate mid-generation
    monitoring by iterating sentence-by-sentence and invoking SymCodeVerifier at
    each boundary.  This mirrors what a real streaming monitor would do: at every
    sentence-end token, run the verifier on the text seen so far.

    SymCodeVerifier (Exp 619, AUC=0.804) is the executable verifier we call at
    each boundary.  It is distribution-invariant: code execution is always correct
    regardless of how the model phrased the step.

**How this module fits into the Carnot pipeline:**

    - InterWhenMonitor wraps any SymCodeVerifier instance.
    - monitor_full_response() simulates mid-generation monitoring on a completed
      response by re-playing it sentence by sentence.
    - monitor_partial() processes one partial generation window (text ending at
      a sentence boundary) and returns a violation if detected.
    - any_violation() is the simple boolean gate for downstream routing.

**The early_detection_rate metric:**

    A violation detected before the *last* sentence is an "early" detection.
    Early detection is the key advantage of mid-generation monitoring: the verifier
    can flag an error and trigger repair *before* the model commits to the wrong
    answer in its final sentence.  The early_detection_rate is the fraction of
    detected violations that were caught before the last sentence boundary.

Spec: REQ-VERIFY-130, REQ-VERIFY-131,
      SCENARIO-VERIFY-168, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from carnot.pipeline.symcode_verifier import CoTStep, SymCodeVerifier

# ---------------------------------------------------------------------------
# InterWhenViolation — result for one sentence-boundary monitoring event
# ---------------------------------------------------------------------------


@dataclass
class InterWhenViolation:
    """Record of a violation detected at a sentence boundary during mid-generation monitoring.

    Each instance corresponds to one sentence boundary where SymCodeVerifier
    detected an arithmetic violation in the partial response text seen so far.

    Fields
    ------
    sentence_index : int
        Zero-based index of the sentence that triggered the violation.  Index 0
        is the first sentence in the response.  An early detection is one where
        sentence_index < (total_sentences - 1).
    sentence_text : str
        The text of the sentence that triggered the violation (last sentence of
        the partial window that was checked).
    violation_detected : bool
        Always True for an InterWhenViolation (violations with score==0.0 are not
        stored; this field exists for clarity and symmetry with CoTStep).
    detection_score : float
        The SymCodeVerifier.detection_score() result for the partial text window
        that triggered the violation.  Value in (0.0, 1.0].
    step_results : list[CoTStep]
        Full per-step CoTStep list from SymCodeVerifier.verify_response() for the
        sentence that triggered this violation.  Useful for debugging which
        arithmetic expression inside the sentence caused the flag.
    """

    sentence_index: int
    sentence_text: str
    violation_detected: bool
    detection_score: float
    step_results: list[CoTStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# InterWhenMonitor
# ---------------------------------------------------------------------------


class InterWhenMonitor:
    """Simulate mid-generation violation monitoring on completed CoT responses.

    The Interwhen approach (arXiv 2602.11202) checks intermediate solutions
    DURING generation rather than only post-hoc.  This class implements that
    idea as a replay simulator: given a completed response, it iterates
    sentence-by-sentence and calls SymCodeVerifier at each sentence boundary,
    exactly as a real streaming monitor would.

    In production streaming, you would call monitor_partial() each time the
    model emits a sentence-ending token.  In benchmark mode (what this class
    primarily supports), you call monitor_full_response() on a completed
    response to measure how many violations would have been detected mid-stream.

    Parameters
    ----------
    verifier : SymCodeVerifier
        The underlying SymCodeVerifier instance to call at each boundary.
        In live mode, this has an llm_caller.  In CI mode, it uses regex fallback.
    sentence_boundary_chars : str
        Characters that delimit sentence boundaries.  Default covers full stop,
        exclamation mark, question mark, and newline.  Split is per-character:
        any character in this string acts as a boundary.
    """

    def __init__(
        self,
        verifier: SymCodeVerifier,
        sentence_boundary_chars: str = ".!?\n",
    ) -> None:
        self.verifier = verifier
        self.boundary_chars = sentence_boundary_chars
        # Accumulates all violations detected across monitor_full_response() calls.
        # Reset between experiments by constructing a fresh InterWhenMonitor.
        self.violations_detected: list[InterWhenViolation] = []

    # ------------------------------------------------------------------
    # split_at_boundaries
    # ------------------------------------------------------------------

    def split_at_boundaries(self, text: str) -> list[str]:
        """Split text at sentence boundaries and return non-empty sentences.

        Splits on every character in self.boundary_chars (full stop, exclamation
        mark, question mark, newline by default).  Empty strings and
        whitespace-only fragments are discarded so that double-punctuation
        ('!?') or trailing newlines do not produce empty "sentences".

        Parameters
        ----------
        text : str
            Arbitrary response text to split.

        Returns
        -------
        list[str]
            Ordered list of non-empty, stripped sentence fragments.
        """
        # Build a single split pattern from all boundary characters.
        # We split on each character individually rather than substrings so that
        # ".!?" doesn't need to appear as a unit — any one of those chars splits.
        parts = [text]
        for ch in self.boundary_chars:
            new_parts: list[str] = []
            for part in parts:
                new_parts.extend(part.split(ch))
            parts = new_parts
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # monitor_partial
    # ------------------------------------------------------------------

    def monitor_partial(self, partial_text: str) -> Optional[InterWhenViolation]:
        """Check the last sentence of partial_text for an arithmetic violation.

        This is the core "mid-generation" check.  In a streaming setting it is
        called every time the model emits a sentence-ending token.  partial_text
        is the full response seen so far (up to and including the new sentence).

        Algorithm:
        1. Split partial_text into sentences.
        2. Take the last sentence as the "new" content to verify (the sentence
           the model just finished).
        3. Run SymCodeVerifier.verify_response() and detection_score() on the
           last sentence only.
        4. If score > 0.0, construct an InterWhenViolation and append it to
           self.violations_detected.

        Parameters
        ----------
        partial_text : str
            The generation text seen so far, ending at a sentence boundary.

        Returns
        -------
        InterWhenViolation or None
            Violation record if an arithmetic error was detected in the last
            sentence; None if no violation was found or text was empty.
        """
        sentences = self.split_at_boundaries(partial_text)
        if not sentences:
            return None
        last = sentences[-1]
        step_results = self.verifier.verify_response(last)
        score = self.verifier.detection_score(last)
        violated = score > 0.0
        v = InterWhenViolation(
            sentence_index=len(sentences) - 1,
            sentence_text=last,
            violation_detected=violated,
            detection_score=score,
            step_results=step_results,
        )
        if violated:
            self.violations_detected.append(v)
        return v if violated else None

    # ------------------------------------------------------------------
    # monitor_full_response
    # ------------------------------------------------------------------

    def monitor_full_response(self, response: str) -> list[InterWhenViolation]:
        """Simulate mid-generation monitoring on a completed response.

        Iterates sentence-by-sentence through the response, calling
        monitor_partial() at each boundary.  This replays the streaming
        scenario: at sentence 0 we check the first sentence; at sentence 1 we
        check the second sentence given the first is already known; etc.

        Why sentence-by-sentence instead of word-by-word: the Interwhen paper
        (arXiv 2602.11202) checks at "step" boundaries, which in CoT corresponds
        to sentence boundaries.  SymCodeVerifier is also sentence-level, so the
        granularity matches.

        Parameters
        ----------
        response : str
            Completed CoT response text.

        Returns
        -------
        list[InterWhenViolation]
            All violations detected during the sentence-by-sentence replay.
            Empty list if no violations found.
        """
        sentences = self.split_at_boundaries(response)
        violations: list[InterWhenViolation] = []
        for i in range(len(sentences)):
            # Reconstruct the partial text up to and including sentence i.
            # Space-join is approximate but sufficient for SymCodeVerifier which
            # operates at sentence granularity, not character-exact positions.
            partial = " ".join(sentences[: i + 1])
            v = self.monitor_partial(partial)
            if v is not None:
                violations.append(v)
        return violations

    # ------------------------------------------------------------------
    # any_violation
    # ------------------------------------------------------------------

    def any_violation(self, response: str) -> bool:
        """Return True iff mid-generation monitoring detects at least one violation.

        Convenience wrapper around monitor_full_response() for binary classification.
        A True result means the response contains at least one sentence boundary
        where SymCodeVerifier flagged an arithmetic inconsistency.

        Parameters
        ----------
        response : str
            Completed CoT response text.

        Returns
        -------
        bool
            True if any violation was detected during sentence-by-sentence replay.
        """
        return len(self.monitor_full_response(response)) > 0
