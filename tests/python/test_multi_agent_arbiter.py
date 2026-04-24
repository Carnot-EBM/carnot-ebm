"""Tests for MultiAgentArbiter — EBM-based ranking of competing agent responses.

**Detailed explanation for engineers:**
    These tests verify:
    - rank_agents() returns the agent with the lowest energy as the winner
    - AgentScore.rank field is assigned correctly (1-based, 1 = winner)
    - ArbiterResult.to_dict() contains all fields required by REQ-AGENT-004
    - Empty responses list produces a graceful no-op result
    - score_response() delegates to pipeline.verify() correctly

Spec: REQ-AGENT-003, REQ-AGENT-004, SCENARIO-AGENT-004
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from carnot.pipeline.multi_agent_arbiter import AgentScore, ArbiterResult, MultiAgentArbiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(*energies: float) -> MagicMock:
    """Build a mock VerifyRepairPipeline whose verify() returns successive energies.

    Each call to pipeline.verify() pops the next energy from the sequence.
    This lets tests deterministically control which agent wins without running
    the real constraint extraction pipeline.

    Spec: REQ-AGENT-003
    """
    pipeline = MagicMock()
    side_effects = []
    for e in energies:
        result = MagicMock()
        result.energy = e
        side_effects.append(result)
    pipeline.verify.side_effect = side_effects
    return pipeline


# ---------------------------------------------------------------------------
# Tests: rank_agents() winner selection
# ---------------------------------------------------------------------------


def test_rank_agents_winner_has_lowest_energy() -> None:
    """Winner must be the agent with the smallest energy value.

    Spec: REQ-AGENT-003, SCENARIO-AGENT-004
    """
    # Agent 0 has energy 0.1 (correct), agents 1 and 2 are higher.
    pipeline = _make_pipeline(0.1, 0.5, 0.9)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("What is 47 + 28?", ["75", "76", "70"])
    assert result.winner_index == 0
    assert result.winner_energy == pytest.approx(0.1)
    assert result.winner_response == "75"


def test_rank_agents_middle_agent_wins() -> None:
    """When the middle agent has lowest energy, winner_index should be 1.

    Spec: REQ-AGENT-003
    """
    pipeline = _make_pipeline(0.8, 0.1, 0.5)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", ["wrong", "correct", "also wrong"])
    assert result.winner_index == 1
    assert result.winner_response == "correct"


def test_rank_agents_last_agent_wins() -> None:
    """When the last agent has lowest energy, winner_index should be 2.

    Spec: REQ-AGENT-003
    """
    pipeline = _make_pipeline(0.9, 0.5, 0.0)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", ["bad", "ok", "best"])
    assert result.winner_index == 2
    assert result.winner_energy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: AgentScore.rank assignment
# ---------------------------------------------------------------------------


def test_rank_field_is_one_based() -> None:
    """Rank 1 must be the winner; ranks must be consecutive from 1.

    Spec: REQ-AGENT-004
    """
    pipeline = _make_pipeline(0.3, 0.1, 0.6)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", ["a", "b", "c"])

    ranks = [s.rank for s in result.all_scores]
    assert ranks == [1, 2, 3]


def test_rank_1_corresponds_to_winner_index() -> None:
    """The AgentScore with rank=1 must have agent_index matching winner_index.

    Spec: REQ-AGENT-004
    """
    pipeline = _make_pipeline(0.5, 0.2, 0.8)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", ["a", "b", "c"])
    rank1 = next(s for s in result.all_scores if s.rank == 1)
    assert rank1.agent_index == result.winner_index


# ---------------------------------------------------------------------------
# Tests: ArbiterResult.to_dict() schema
# ---------------------------------------------------------------------------


def test_to_dict_contains_all_required_fields() -> None:
    """to_dict() must include all fields required by REQ-AGENT-004.

    Spec: REQ-AGENT-004
    """
    pipeline = _make_pipeline(0.1, 0.9)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", ["good", "bad"])
    d = result.to_dict()

    required_fields = {
        "n_agents",
        "winner_index",
        "winner_response",
        "winner_energy",
        "all_scores",
        "inference_mode",
        "honest_verdict",
    }
    assert required_fields.issubset(d.keys())


def test_to_dict_all_scores_are_serializable() -> None:
    """Every entry in all_scores must be a plain dict with the four required keys.

    Spec: REQ-AGENT-004
    """
    pipeline = _make_pipeline(0.3, 0.7, 0.1)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", ["x", "y", "z"])
    d = result.to_dict()
    for score in d["all_scores"]:
        assert "agent_index" in score
        assert "response" in score
        assert "energy" in score
        assert "rank" in score


# ---------------------------------------------------------------------------
# Tests: n_agents field
# ---------------------------------------------------------------------------


def test_n_agents_matches_input_length() -> None:
    """n_agents must equal len(responses).

    Spec: REQ-AGENT-004
    """
    pipeline = _make_pipeline(0.5, 0.3, 0.8)
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", ["a", "b", "c"])
    assert result.n_agents == 3


# ---------------------------------------------------------------------------
# Tests: Empty responses
# ---------------------------------------------------------------------------


def test_empty_responses_returns_graceful_result() -> None:
    """Empty responses list must return a valid ArbiterResult without crashing.

    Spec: REQ-AGENT-003
    """
    pipeline = MagicMock()
    arbiter = MultiAgentArbiter(pipeline)
    result = arbiter.rank_agents("Q?", [])
    assert result.n_agents == 0
    assert result.all_scores == []
    pipeline.verify.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: score_response delegates to pipeline
# ---------------------------------------------------------------------------


def test_score_response_calls_pipeline_verify() -> None:
    """score_response() must call pipeline.verify() exactly once with the right args.

    Spec: REQ-AGENT-003
    """
    pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.energy = 0.42
    pipeline.verify.return_value = mock_result
    arbiter = MultiAgentArbiter(pipeline)
    energy = arbiter.score_response("What is 1 + 1?", "2")
    pipeline.verify.assert_called_once_with("What is 1 + 1?", "2", domain=None)
    assert energy == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Tests: AgentScore dataclass
# ---------------------------------------------------------------------------


def test_agent_score_to_dict() -> None:
    """AgentScore.to_dict() must round-trip all four fields.

    Spec: REQ-AGENT-004
    """
    score = AgentScore(agent_index=2, response="hello", energy=0.25, rank=1)
    d = score.to_dict()
    assert d["agent_index"] == 2
    assert d["response"] == "hello"
    assert d["energy"] == pytest.approx(0.25)
    assert d["rank"] == 1
