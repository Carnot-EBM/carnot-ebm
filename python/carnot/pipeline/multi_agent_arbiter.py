"""Multi-Agent Arbiter: rank competing agent responses by EBM energy.

**Researcher summary:**
    Multi-agent systems need a neutral judge to pick the best answer when
    multiple agents disagree.  Energy is a natural arbitration signal — the
    agent whose response satisfies the most constraints has the lowest energy
    and should be preferred.  This module implements that idea in three
    dataclasses plus one orchestrating class.

**Detailed explanation for engineers:**
    The arbiter wraps ``VerifyRepairPipeline.verify()`` in a loop, calling it
    once per agent response.  Each call returns a ``VerificationResult`` whose
    ``.energy`` field is a float (0.0 = perfectly constraint-satisfied, higher
    = more violations).  The arbiter sorts the results by ascending energy,
    assigns rank 1 to the winner, and packages everything into an
    ``ArbiterResult``.

    Why energy and not a binary pass/fail vote?  Binary voting requires a
    threshold, which requires calibration.  Energy is continuous and always
    available without tuning.  The agent with the smallest energy value is
    the one that violates the fewest constraints by the smallest magnitude —
    which is exactly what "most correct" means in an EBM sense.

    Pipeline instantiation: the arbiter creates a fresh ``VerifyRepairPipeline``
    with ``model=None`` (verify-only, no LLM repair) by default.  This keeps
    the arbiter stateless across calls and avoids any risk of one agent's
    context leaking into another's score.

Spec: REQ-AGENT-003, REQ-AGENT-004, SCENARIO-AGENT-004
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentScore:
    """Score assigned to a single agent response.

    **Detailed explanation for engineers:**
        ``rank`` is 1-based (rank=1 is the winner with lowest energy).
        ``agent_index`` is the position of this response in the original
        ``responses`` list passed to ``rank_agents()``.  This lets the caller
        map scores back to agents without knowing the sorted order.

    Attributes:
        agent_index: Original list index of this agent (0-based).
        response: The agent's raw response text.
        energy: EBM energy — lower is better / more constraint-consistent.
        rank: Final rank (1 = winner, 2 = runner-up, …).
    """

    agent_index: int
    response: str
    energy: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return asdict(self)


@dataclass
class ArbiterResult:
    """Full result of a multi-agent arbitration run.

    **Detailed explanation for engineers:**
        ``winner_index`` indexes into the original ``responses`` list so the
        caller can recover the winning agent without unpacking ``all_scores``.

        ``honest_verdict`` is a short string that downstream tools (conductor,
        retro scripts) can use to classify the run without parsing numbers.
        Possible values are determined by the calling experiment script, not
        by this class.

        ``inference_mode`` is always ``"cpu_ebm"`` when instantiated via the
        default pipeline; it changes only if a caller passes a custom pipeline
        that runs on GPU or a different backend.

    Attributes:
        n_agents: Number of agents whose responses were scored.
        winner_index: Original list index of the winning (lowest-energy) agent.
        winner_response: Response text of the winning agent.
        winner_energy: Energy score of the winning agent.
        all_scores: Full ranked list of AgentScore objects (rank 1 first).
        inference_mode: Short string describing how energy was computed.
        honest_verdict: Caller-supplied verdict string ("arbiter_correct", etc.).
    """

    n_agents: int
    winner_index: int
    winner_response: str
    winner_energy: float
    all_scores: list[AgentScore] = field(default_factory=list)
    inference_mode: str = "cpu_ebm"
    honest_verdict: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "n_agents": self.n_agents,
            "winner_index": self.winner_index,
            "winner_response": self.winner_response,
            "winner_energy": self.winner_energy,
            "all_scores": [s.to_dict() for s in self.all_scores],
            "inference_mode": self.inference_mode,
            "honest_verdict": self.honest_verdict,
        }


# ---------------------------------------------------------------------------
# Main arbiter class
# ---------------------------------------------------------------------------


class MultiAgentArbiter:
    """Rank competing agent responses by EBM energy via VerifyRepairPipeline.

    **Detailed explanation for engineers:**
        The key invariant: ``rank_agents()`` never mutates the input list and
        is safe to call multiple times on different response sets.  A fresh
        ``VerifyRepairPipeline.verify()`` call is made for each response so
        there is no cross-contamination of constraint context between agents.

        Energy score semantics: the pipeline's verify() method extracts
        constraints from the (question, response) pair and returns their total
        energy.  A response that satisfies all arithmetic constraints will have
        energy 0.0; a response with one violated constraint will have energy > 0.
        The arbiter uses these raw floats for ranking — no normalisation or
        threshold is applied.

    Args:
        pipeline: An already-constructed VerifyRepairPipeline.  Pass
            ``VerifyRepairPipeline(model=None)`` for verify-only mode.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def score_response(self, question: str, response: str) -> float:
        """Return the EBM energy for one (question, response) pair.

        **Detailed explanation for engineers:**
            Energy is extracted by running the full constraint pipeline.
            Lower energy = fewer / smaller constraint violations = better response.
            The domain is left as None so the AutoExtractor picks the most
            appropriate extractors automatically.

        Args:
            question: The original question posed to the agent.
            response: The agent's response text.

        Returns:
            Float energy value (0.0 = perfect, higher = more violations).

        Spec: REQ-AGENT-003
        """
        result = self._pipeline.verify(question, response, domain=None)
        return float(result.energy)

    def rank_agents(self, question: str, responses: list[str]) -> ArbiterResult:
        """Score all agent responses and return a ranked ArbiterResult.

        **Detailed explanation for engineers:**
            1. Score every response by calling score_response().
            2. Sort ascending by energy (lowest = rank 1 = winner).
            3. Build AgentScore objects preserving original indices.
            4. Return ArbiterResult with winner fields pre-populated.

        Args:
            question: The shared question all agents responded to.
            responses: List of response strings, one per agent.

        Returns:
            ArbiterResult with winner info and full ranked AgentScore list.

        Spec: REQ-AGENT-003, REQ-AGENT-004, SCENARIO-AGENT-004
        """
        if not responses:
            return ArbiterResult(
                n_agents=0,
                winner_index=0,
                winner_response="",
                winner_energy=0.0,
                all_scores=[],
                inference_mode="cpu_ebm",
                honest_verdict="no_agents",
            )

        # Score each response, preserving original index.
        scored: list[tuple[int, str, float]] = [
            (i, r, self.score_response(question, r)) for i, r in enumerate(responses)
        ]

        # Sort ascending by energy so rank 1 has the lowest energy.
        sorted_scored = sorted(scored, key=lambda x: x[2])

        all_scores: list[AgentScore] = [
            AgentScore(
                agent_index=orig_idx,
                response=resp,
                energy=energy,
                rank=rank + 1,  # 1-based
            )
            for rank, (orig_idx, resp, energy) in enumerate(sorted_scored)
        ]

        winner_index, winner_response, winner_energy = sorted_scored[0]

        return ArbiterResult(
            n_agents=len(responses),
            winner_index=winner_index,
            winner_response=winner_response,
            winner_energy=winner_energy,
            all_scores=all_scores,
            inference_mode="cpu_ebm",
            honest_verdict="unknown",  # caller sets this after the fact
        )
