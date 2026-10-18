"""RPRMStepReward — Tier 2.9 reasoning-driven step-level process reward model.

**Researcher summary:**
    arXiv 2503.21295 (R-PRM: Reasoning-Driven Process Reward Modeling) shows that
    having the model first generate a brief explanation of WHY a reasoning step may
    be wrong — before assigning a numerical score — yields +11.9 F1 points on
    ProcessBench versus direct-scoring PRMs.

    The key insight: the reasoning explanation acts as a regularizer that forces
    the scorer to engage with the step's logical content rather than surface patterns
    (e.g. a step that just "looks right" because it uses the right vocabulary).
    The explanation also doubles as a repair hint for downstream IterativeSelfRepair.

    This module implements Tier 2.9 — inserted between Tier 2.7 (CausalReasoningVerifier)
    and Tier 3 (Ising) in the ThreeTierPipeline cascade.

    Two operating modes:
        heuristic — no LLM required; uses regex-based pattern matching to flag
                    suspicious arithmetic steps.  Fully CI-safe (no GPU needed).
        llm       — uses a provided llm_runner callable to generate a VERDICT-tagged
                    explanation, then maps VERDICT token to a score.

    WHY split steps at sentence/newline boundaries?
        LLM-generated math reasoning typically places one logical move per sentence
        or one per numbered step.  Finer granularity would mostly split tokens
        mid-arithmetic; coarser (paragraph) would obscure which step triggered the flag.

    WHY 0.7 / 0.1 as heuristic scores?
        0.7 is deliberately below the 0.9 used for confirmed LLM "VERDICT: wrong" so
        that heuristic detections don't dominate a mixed-mode ensemble.  0.1 keeps
        clean steps as background noise rather than zero (avoids AUC artifacts from
        clipping at exactly 0.0).

Spec: REQ-VERIFY-148, SCENARIO-VERIFY-148
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StepReasoningResult:
    """Per-step output from score_step_with_reasoning().

    Attributes:
        step_text: The raw step string that was evaluated.
        reasoning: A brief explanation of why the step is correct or suspicious.
                   In heuristic mode this is a short canned message; in LLM mode
                   it is the model's verbatim explanation up to n_reasoning_tokens.
        step_score: Violation probability in [0, 1].  Higher = more likely wrong.
        reasoning_mode: "heuristic" or "llm", so callers can weight accordingly.
    """

    step_text: str
    reasoning: str
    step_score: float
    reasoning_mode: str  # "heuristic" | "llm"


@dataclass
class RPRMResult:
    """Aggregate result from verify_response().

    Attributes:
        steps: Per-step results in the order they appear in the response.
        overall_violation_prob: max step_score across all steps (worst step dominates).
        n_flagged: Number of steps with step_score > 0.5.
        repair_hints: reasoning strings for flagged steps; feed to IterativeSelfRepair.
    """

    steps: list[StepReasoningResult]
    overall_violation_prob: float
    n_flagged: int
    repair_hints: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RPRMStepReward:
    """Tier 2.9 reasoning-driven step-level process reward model.

    Operates in one of two modes:
        - heuristic (llm_runner=None): pattern-match suspicious arithmetic;
          fully deterministic and CI-safe.
        - llm (llm_runner provided): call the LLM once per step, ask it to
          explain the error, then parse a VERDICT token.

    Args:
        llm_runner: Optional callable with signature
            ``(prompt: str, max_tokens: int) -> str``.
            If None, heuristic mode is used.
        n_reasoning_tokens: Max tokens to request from the LLM per step.
            Ignored in heuristic mode.
    """

    def __init__(
        self,
        llm_runner: Callable[[str, int], str] | None = None,
        n_reasoning_tokens: int = 50,
    ) -> None:
        self.llm = llm_runner
        self.n_tokens = n_reasoning_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_step_with_reasoning(self, step: str, context: str) -> StepReasoningResult:
        """Score a single reasoning step, generating an explanation first.

        Args:
            step: One sentence / logical step extracted from a model response.
            context: The original question, used as grounding context for the LLM.

        Returns:
            StepReasoningResult with reasoning explanation and violation probability.
        """
        if self.llm is None:
            return self._heuristic_score(step)
        return self._llm_score(step, context)

    def verify_response(self, question: str, response: str) -> RPRMResult:
        """Verify a full model response by scoring each step.

        Splits the response into steps, scores each one with reasoning,
        and aggregates into an RPRMResult.

        Args:
            question: The original question (grounding context for the LLM).
            response: The model's full response string.

        Returns:
            RPRMResult with per-step details and aggregate violation probability.
        """
        steps = self._split_steps(response)
        if not steps:
            return RPRMResult(
                steps=[],
                overall_violation_prob=0.0,
                n_flagged=0,
                repair_hints=[],
            )

        step_results = [self.score_step_with_reasoning(s, question) for s in steps]
        overall_score = max(r.step_score for r in step_results)
        flagged = [r for r in step_results if r.step_score > 0.5]

        return RPRMResult(
            steps=step_results,
            overall_violation_prob=overall_score,
            n_flagged=len(flagged),
            repair_hints=[r.reasoning for r in flagged],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_score(self, step: str) -> StepReasoningResult:
        """Flag a step using deterministic regex patterns without an LLM.

        Patterns that indicate a suspicious arithmetic step:
        - "= 0" in a long step (final result zero is unusual in word problems)
        - More than two equals signs (often indicates contradictory rewrites)
        - Division-by-zero pattern: a literal "0" on the left of an equals
        """
        suspicious = any(
            [
                "= 0" in step and len(step) > 20,
                step.count("=") > 2,
                bool(re.search(r"\b0\b.*=", step)),
            ]
        )
        if suspicious:
            reasoning = "heuristic: suspicious pattern detected"
            score = 0.7
        else:
            reasoning = "ok"
            score = 0.1
        return StepReasoningResult(
            step_text=step,
            reasoning=reasoning,
            step_score=score,
            reasoning_mode="heuristic",
        )

    def _llm_score(self, step: str, context: str) -> StepReasoningResult:
        """Score a step using the LLM to generate an explanation first.

        The prompt asks the LLM to reason about correctness before issuing a
        VERDICT token.  This matches the R-PRM paper's chain-of-thought-before-score
        design that delivers the +11.9 F1 improvement.
        """
        reasoning_prompt = (
            f"Context: {context}\n"
            f"Step: {step}\n"
            f"Is this step correct? Explain briefly in {self.n_tokens} tokens max.\n"
            f"End with: VERDICT: [correct|suspicious|wrong]"
        )
        reasoning_text = self.llm(reasoning_prompt, self.n_tokens)

        if "VERDICT: wrong" in reasoning_text:
            score = 0.9
        elif "VERDICT: suspicious" in reasoning_text:
            score = 0.5
        else:
            score = 0.1

        return StepReasoningResult(
            step_text=step,
            reasoning=reasoning_text,
            step_score=score,
            reasoning_mode="llm",
        )

    def _split_steps(self, response: str) -> list[str]:
        """Split a response string into individual reasoning steps.

        Splits on sentence-ending punctuation or newlines.  Discards fragments
        shorter than 10 characters (too short to be a meaningful logical step).

        Args:
            response: Full model response string.

        Returns:
            List of non-trivial step strings, order preserved.
        """
        raw = re.split(r"[.\n]", response)
        return [s.strip() for s in raw if len(s.strip()) > 10]
