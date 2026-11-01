"""MARSMarginGate — logit-margin oracle skip gate from arXiv 2601.15498.

**Why a margin gate? (MARS paper insight)**
    Running every generated code snippet through a test oracle is expensive:
    each oracle call may spawn a subprocess, execute untrusted code, and wait
    for a timeout.  The MARS paper (arXiv 2601.15498) observes that when a
    model assigns a large margin between its top-1 and top-2 token logits,
    the output is likely correct without oracle verification.  Skipping the
    oracle for these high-confidence outputs reduces wall time significantly
    without meaningfully degrading measured accuracy.

**Logit margin definition:**
    margin = logit_top1 - logit_top2

    A high margin means the model was "sure" about its best prediction.
    A low margin means the model was uncertain and the oracle should run.

**CI mode:**
    When logits are unavailable (e.g. llama.cpp does not expose per-token
    logits in all configurations), the gate falls back to ci_logits_unavailable
    — it does NOT skip the oracle.  This is the safe default: uncertain input
    → keep verification.

Spec: REQ-BENCH-060, REQ-BENCH-061, SCENARIO-BENCH-084, SCENARIO-BENCH-085
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MARSMarginResult:
    """Outcome of a single MARS margin-gate decision.

    Fields
    ------
    logit_margin : float
        Computed margin (top1 - top2).  0.0 when logits were unavailable.
    margin_threshold : float
        The threshold configured on this gate instance.
    skip_oracle : bool
        True when the margin exceeds the threshold and oracle can be skipped.
    honest_verdict : str
        Human-readable gate decision for audit logging:
        - ``"margin_skip"`` — margin above threshold; oracle skipped.
        - ``"margin_run_oracle"`` — margin at or below threshold; run oracle.
        - ``"ci_logits_unavailable"`` — no logits provided; oracle runs by default.
    """

    logit_margin: float
    margin_threshold: float
    skip_oracle: bool
    honest_verdict: str


def compute_logit_margin(logits: list[float]) -> float:
    """Return the margin between the top-1 and top-2 logits.

    Why top1 - top2: this is the MARS paper formulation.  A large gap means
    the model strongly preferred one token over all alternatives; a small gap
    means it was nearly indifferent.

    Parameters
    ----------
    logits : list[float]
        Raw logit values for each token in the model's vocabulary (or a
        representative subset).  Must contain at least two values.

    Returns
    -------
    float
        top1_logit - top2_logit.  Always >= 0.0 when len(logits) >= 2.
        Returns 0.0 when fewer than two logits are provided (single-token
        vocabulary edge case — treated as zero-margin / uncertain).

    Spec: REQ-BENCH-061
    """
    if len(logits) < 2:
        return 0.0
    top2 = sorted(logits, reverse=True)[:2]
    return top2[0] - top2[1]


class MARSMarginGate:
    """Oracle-skip gate based on the MARS logit-margin criterion.

    Instantiate once per experiment run and call ``decide()`` for each
    generated code snippet.  Pass ``logits=None`` to signal that the
    inference backend did not expose token logits (CI / no-GPU mode).

    Parameters
    ----------
    threshold : float
        Minimum logit margin required to skip the oracle.  Default 2.0
        follows the MARS paper recommendation; lower values skip more
        aggressively (higher oracle_calls_saved, lower reliability).

    Spec: REQ-BENCH-060, REQ-BENCH-061
    """

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold

    def decide(self, logits: list[float] | None) -> MARSMarginResult:
        """Decide whether to skip the test oracle for a generated output.

        Parameters
        ----------
        logits : list[float] | None
            Per-token logits from the last generated token (or a sampled
            subset).  Pass ``None`` when the backend does not expose logits.

        Returns
        -------
        MARSMarginResult
            ``skip_oracle=True`` iff margin > threshold AND logits were
            provided.  See ``MARSMarginResult`` docstring for verdict codes.

        Spec: REQ-BENCH-060, SCENARIO-BENCH-084, SCENARIO-BENCH-085
        """
        if logits is None:
            # CI mode: logits unavailable — default to running the oracle.
            return MARSMarginResult(
                logit_margin=0.0,
                margin_threshold=self.threshold,
                skip_oracle=False,
                honest_verdict="ci_logits_unavailable",
            )

        margin = compute_logit_margin(logits)
        if margin > self.threshold:
            return MARSMarginResult(
                logit_margin=margin,
                margin_threshold=self.threshold,
                skip_oracle=True,
                honest_verdict="margin_skip",
            )
        return MARSMarginResult(
            logit_margin=margin,
            margin_threshold=self.threshold,
            skip_oracle=False,
            honest_verdict="margin_run_oracle",
        )
