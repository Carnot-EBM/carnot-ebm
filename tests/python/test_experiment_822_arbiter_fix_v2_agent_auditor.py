"""Tests for Exp 822: Arbiter Fix v2 + AgentAuditor Consensus Penalty.

Covers:
- detect_consensus returns True when energy variance < threshold     (REQ-VERIFY-144)
- apply_consensus_penalty adds penalty to majority cluster agents    (REQ-VERIFY-144)
- arbitrate selects min-energy agent after consensus adjustment      (REQ-VERIFY-143)
- score_agents calls external field scoring (not legacy diagonal)    (REQ-VERIFY-143)
- Gate check blocks when Exp 819 verdict is not injection_field_fixed (REQ-VERIFY-143)

Spec: REQ-VERIFY-143, REQ-VERIFY-144, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.inference.multi_agent_arbiter import MultiAgentArbiter
from scripts.experiment_822_arbiter_fix_v2_agent_auditor import (
    _check_exp819_gate,
    map_honest_verdict,
    _build_standard_scenarios,
    _build_adversarial_scenarios,
    _make_constraint_embeddings,
    EXP_819_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def arbiter() -> MultiAgentArbiter:
    """A MultiAgentArbiter with small spin count for fast tests."""
    return MultiAgentArbiter(
        n_spins=16,
        embedding_dim=384,
        consensus_threshold=0.01,
        consensus_penalty=0.1,
    )


# ---------------------------------------------------------------------------
# REQ-VERIFY-144: detect_consensus returns True when variance < threshold
# ---------------------------------------------------------------------------


class TestDetectConsensus:
    """REQ-VERIFY-144: detect_consensus must detect near-zero-variance energy arrays."""

    def test_returns_true_when_all_energies_equal(self, arbiter: MultiAgentArbiter) -> None:
        """All identical energies → variance = 0 < 0.01 → consensus detected.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([1.5, 1.5, 1.5])
        assert arbiter.detect_consensus(energies) is True

    def test_returns_true_when_range_below_threshold(self, arbiter: MultiAgentArbiter) -> None:
        """Max - min = 0.005 < 0.01 → consensus detected.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([1.0, 1.003, 1.005])
        assert arbiter.detect_consensus(energies) is True

    def test_returns_false_when_range_above_threshold(self, arbiter: MultiAgentArbiter) -> None:
        """Max - min = 0.05 > 0.01 → consensus NOT detected.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([1.0, 1.02, 1.05])
        assert arbiter.detect_consensus(energies) is False

    def test_custom_threshold_override(self, arbiter: MultiAgentArbiter) -> None:
        """threshold parameter overrides instance default.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([1.0, 1.005, 1.008])
        # With default threshold=0.01: True (range 0.008 < 0.01).
        assert arbiter.detect_consensus(energies, threshold=0.01) is True
        # With threshold=0.005: False (range 0.008 >= 0.005).
        assert arbiter.detect_consensus(energies, threshold=0.005) is False

    def test_single_agent_always_consensus(self, arbiter: MultiAgentArbiter) -> None:
        """Single agent: range = 0 < any positive threshold → always True.

        Spec: REQ-VERIFY-144
        """
        assert arbiter.detect_consensus(np.array([2.5])) is True


# ---------------------------------------------------------------------------
# REQ-VERIFY-144: apply_consensus_penalty adds penalty to majority cluster
# ---------------------------------------------------------------------------


class TestApplyConsensusPenalty:
    """REQ-VERIFY-144: apply_consensus_penalty must penalise majority-cluster agents."""

    def test_penalty_added_to_majority_two_of_three(self, arbiter: MultiAgentArbiter) -> None:
        """Two agents share the majority response → both get +0.1 penalty.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([1.0, 0.8, 0.8])
        responses = ["correct", "wrong", "wrong"]
        adjusted = arbiter.apply_consensus_penalty(energies, responses)

        # "wrong" appears twice — majority cluster.
        assert adjusted[0] == pytest.approx(1.0)   # "correct" not penalised
        assert adjusted[1] == pytest.approx(0.9)   # 0.8 + 0.1
        assert adjusted[2] == pytest.approx(0.9)   # 0.8 + 0.1

    def test_penalty_only_on_majority_not_minority(self, arbiter: MultiAgentArbiter) -> None:
        """Minority agent energy is unchanged after penalty.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([0.5, 1.2, 1.2])
        responses = ["minority_correct", "majority_wrong", "majority_wrong"]
        adjusted = arbiter.apply_consensus_penalty(energies, responses)

        assert adjusted[0] == pytest.approx(0.5)   # minority unchanged
        assert adjusted[1] == pytest.approx(1.3)
        assert adjusted[2] == pytest.approx(1.3)

    def test_penalty_does_not_mutate_input(self, arbiter: MultiAgentArbiter) -> None:
        """apply_consensus_penalty must return a new array, not mutate input.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([1.0, 0.5, 0.5])
        original = energies.copy()
        arbiter.apply_consensus_penalty(energies, ["a", "b", "b"])
        np.testing.assert_array_equal(energies, original)

    def test_custom_penalty_parameter(self, arbiter: MultiAgentArbiter) -> None:
        """penalty kwarg overrides instance default.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([0.0, 0.0, 0.0])
        responses = ["a", "b", "b"]
        adjusted = arbiter.apply_consensus_penalty(energies, responses, penalty=0.5)
        assert adjusted[1] == pytest.approx(0.5)
        assert adjusted[2] == pytest.approx(0.5)
        assert adjusted[0] == pytest.approx(0.0)

    def test_all_same_response_all_penalised(self, arbiter: MultiAgentArbiter) -> None:
        """When all agents agree, all are penalised equally — no change in ranking.

        Spec: REQ-VERIFY-144
        """
        energies = np.array([0.3, 0.5, 0.7])
        responses = ["same", "same", "same"]
        adjusted = arbiter.apply_consensus_penalty(energies, responses, penalty=0.2)
        np.testing.assert_allclose(adjusted, [0.5, 0.7, 0.9])


# ---------------------------------------------------------------------------
# REQ-VERIFY-143: arbitrate selects min-energy agent (with consensus adjustment)
# ---------------------------------------------------------------------------


class TestArbitrate:
    """REQ-VERIFY-143: arbitrate must select the minimum-energy agent post-adjustment."""

    def test_selects_lowest_energy_without_consensus(self, arbiter: MultiAgentArbiter) -> None:
        """Non-consensus case: arbiter picks the agent with the lowest raw energy.

        We use distinct responses so their spin encodings differ and energy variance
        should exceed the consensus threshold.

        Spec: REQ-VERIFY-143
        """
        rng = np.random.default_rng(42)
        constraint_embeddings = _make_constraint_embeddings(rng, n=3)

        # Three distinct responses — spin encodings will differ.
        responses = ["response_alpha_unique_123", "response_beta_unique_456", "response_gamma_unique_789"]
        result = arbiter.arbitrate(responses, constraint_embeddings)

        # The arbiter must have picked one of the 3 agents.
        assert 0 <= result["arbiter_index"] < 3
        assert result["arbiter_response"] in responses
        assert "energies_raw" in result
        assert "energies_adjusted" in result
        assert "used_consensus_penalty" in result

        # Whichever agent was picked must have the lowest adjusted energy.
        adj = np.array(result["energies_adjusted"])
        assert result["arbiter_index"] == int(np.argmin(adj))

    def test_consensus_penalty_triggers_for_identical_responses(
        self, arbiter: MultiAgentArbiter
    ) -> None:
        """When two agents give identical responses, consensus penalty is applied.

        Two identical response strings → identical spin configs → identical energies for
        those two agents → range = 0 < threshold → penalty triggered.

        Spec: REQ-VERIFY-143, REQ-VERIFY-144
        """
        rng = np.random.default_rng(7)
        constraint_embeddings = _make_constraint_embeddings(rng, n=3)

        responses = ["unique_correct_xyz", "same_wrong_abc", "same_wrong_abc"]
        result = arbiter.arbitrate(responses, constraint_embeddings)

        assert result["used_consensus_penalty"] is True

    def test_consensus_penalty_favors_minority(self, arbiter: MultiAgentArbiter) -> None:
        """Minority agent wins after consensus penalty is applied to the majority cluster.

        Scenario: 2 agents share the wrong answer (majority cluster), 1 has correct answer.
        The wrong agents have slightly lower raw energy than the correct agent (adversarial:
        majority-wrong would win without penalty).  After +0.1 penalty to the majority:
            raw:      [0.35, 0.30, 0.30]
            adjusted: [0.35, 0.40, 0.40]  → min at index 0 → correct minority wins.

        Spec: REQ-VERIFY-143, REQ-VERIFY-144, SCENARIO-VERIFY-173
        """
        responses = ["correct_minority", "wrong_majority", "wrong_majority"]
        # Majority (wrong) has slightly lower raw energy — wins WITHOUT penalty.
        raw_energies = np.array([0.35, 0.3, 0.3])

        with patch.object(arbiter, "score_agents", return_value=raw_energies):
            result = arbiter.arbitrate(responses, [])

        # After penalty: [0.35, 0.4, 0.4] → min is 0 → correct minority wins.
        assert result["arbiter_index"] == 0
        assert result["used_consensus_penalty"] is True

    def test_arbitrate_returns_all_required_fields(self, arbiter: MultiAgentArbiter) -> None:
        """arbitrate result contains all required output keys.

        Spec: REQ-VERIFY-143
        """
        rng = np.random.default_rng(1)
        result = arbiter.arbitrate(
            ["resp_a", "resp_b"],
            _make_constraint_embeddings(rng, n=2),
        )
        assert "arbiter_index" in result
        assert "arbiter_response" in result
        assert "energies_raw" in result
        assert "energies_adjusted" in result
        assert "used_consensus_penalty" in result
        assert isinstance(result["arbiter_index"], int)
        assert result["arbiter_response"] in ["resp_a", "resp_b"]

    def test_no_constraint_embeddings_still_returns_result(
        self, arbiter: MultiAgentArbiter
    ) -> None:
        """Empty constraint list is valid (zero external field, fallback to Ising energy).

        Spec: REQ-VERIFY-143
        """
        result = arbiter.arbitrate(["a", "b", "c"], [])
        assert 0 <= result["arbiter_index"] < 3


# ---------------------------------------------------------------------------
# REQ-VERIFY-143: score_agents uses external field (not legacy constant shift)
# ---------------------------------------------------------------------------


class TestScoreAgents:
    """REQ-VERIFY-143: score_agents must use external field scoring, not diagonal injection."""

    def test_different_responses_give_different_energies_with_embeddings(
        self, arbiter: MultiAgentArbiter
    ) -> None:
        """With non-zero constraint embeddings, different responses get different energies.

        This confirms the external field is actually discriminating — if it were a
        constant shift (legacy diagonal injection), all energies would be equal.

        Spec: REQ-VERIFY-143
        """
        rng = np.random.default_rng(99)
        embeddings = _make_constraint_embeddings(rng, n=5)

        # Use responses with maximally different text so their spin configs differ.
        responses = [
            "the correct answer is forty two percent",
            "the wrong answer is ninety nine percent",
            "a completely different wrong response here",
        ]
        energies = arbiter.score_agents(responses, embeddings)

        # At least two energies must differ — if all equal, field is not discriminating.
        assert not np.allclose(energies[0], energies[1]) or not np.allclose(
            energies[0], energies[2]
        ), "All agent energies are equal — external field is not discriminating."

    def test_same_response_same_energy(self, arbiter: MultiAgentArbiter) -> None:
        """Identical responses must produce identical energies (deterministic encoding).

        Spec: REQ-VERIFY-143
        """
        rng = np.random.default_rng(5)
        embeddings = _make_constraint_embeddings(rng, n=3)
        responses = ["identical_response_abc", "identical_response_abc"]
        energies = arbiter.score_agents(responses, embeddings)
        assert energies[0] == pytest.approx(energies[1])

    def test_empty_embeddings_gives_ising_only_energy(self, arbiter: MultiAgentArbiter) -> None:
        """With no constraint embeddings, h=0 so E_total = E_ising only.

        Spec: REQ-VERIFY-143
        """
        energies = arbiter.score_agents(["resp_x", "resp_y"], [])
        # Both should be finite numbers (not NaN or inf).
        assert np.all(np.isfinite(energies))


# ---------------------------------------------------------------------------
# Gate check tests
# ---------------------------------------------------------------------------


class TestExp819Gate:
    """Gate check must block when Exp 819 verdict is not 'injection_field_fixed'."""

    def test_gate_blocks_when_file_missing(self, tmp_path: Path) -> None:
        """Returns blocked artifact when Exp 819 result file does not exist.

        Spec: REQ-VERIFY-143
        """
        missing = tmp_path / "no_file.json"
        tmpl = MagicMock()
        tmpl.build_result.side_effect = lambda *a, **kw: {"status": "blocked", **kw}
        with patch(
            "scripts.experiment_822_arbiter_fix_v2_agent_auditor.EXP_819_PATH", missing
        ):
            result = _check_exp819_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "blocked_gate"

    def test_gate_blocks_when_wrong_verdict(self, tmp_path: Path) -> None:
        """Returns blocked artifact when Exp 819 verdict is not 'injection_field_fixed'.

        Spec: REQ-VERIFY-143
        """
        f = tmp_path / "exp819.json"
        f.write_text(json.dumps({"honest_verdict": "injection_partial"}))
        tmpl = MagicMock()
        tmpl.build_result.side_effect = lambda *a, **kw: {"status": "blocked", **kw}
        with patch(
            "scripts.experiment_822_arbiter_fix_v2_agent_auditor.EXP_819_PATH", f
        ):
            result = _check_exp819_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "blocked_gate"

    def test_gate_passes_when_correct_verdict(self, tmp_path: Path) -> None:
        """Returns None (proceed) when Exp 819 verdict == 'injection_field_fixed'.

        Spec: REQ-VERIFY-143
        """
        f = tmp_path / "exp819.json"
        f.write_text(json.dumps({"honest_verdict": "injection_field_fixed"}))
        tmpl = MagicMock()
        with patch(
            "scripts.experiment_822_arbiter_fix_v2_agent_auditor.EXP_819_PATH", f
        ):
            result = _check_exp819_gate(tmpl)
        assert result is None


# ---------------------------------------------------------------------------
# honest_verdict mapping
# ---------------------------------------------------------------------------


class TestMapHonestVerdict:
    """map_honest_verdict maps accuracy correctly.

    Spec: SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
    """

    def test_blocked_gate(self) -> None:
        assert map_honest_verdict(0.5, gate_blocked=True) == "blocked_gate"

    def test_arbiter_correct_at_threshold(self) -> None:
        assert map_honest_verdict(0.80) == "arbiter_correct"

    def test_arbiter_correct_above_threshold(self) -> None:
        assert map_honest_verdict(1.0) == "arbiter_correct"

    def test_arbiter_partial_at_lower_bound(self) -> None:
        assert map_honest_verdict(0.60) == "arbiter_partial"

    def test_arbiter_partial_just_below_correct(self) -> None:
        assert map_honest_verdict(0.75) == "arbiter_partial"

    def test_arbiter_still_wrong_below_partial(self) -> None:
        assert map_honest_verdict(0.50) == "arbiter_still_wrong"

    def test_arbiter_still_wrong_at_zero(self) -> None:
        assert map_honest_verdict(0.0) == "arbiter_still_wrong"


# ---------------------------------------------------------------------------
# Scenario-level: standard and adversarial batches
# ---------------------------------------------------------------------------


class TestScenarioBatches:
    """SCENARIO-VERIFY-172/173: standard and adversarial scenarios return expected shapes."""

    def test_standard_scenarios_return_six(self, arbiter: MultiAgentArbiter) -> None:
        """_build_standard_scenarios returns exactly 6 results.

        Spec: SCENARIO-VERIFY-172
        """
        rng = np.random.default_rng(42)
        results = _build_standard_scenarios(arbiter, rng)
        assert len(results) == 6

    def test_adversarial_scenarios_return_six(self, arbiter: MultiAgentArbiter) -> None:
        """_build_adversarial_scenarios returns exactly 6 results.

        Spec: SCENARIO-VERIFY-173
        """
        rng = np.random.default_rng(42)
        results = _build_adversarial_scenarios(arbiter, rng)
        assert len(results) == 6

    def test_standard_results_have_required_fields(self, arbiter: MultiAgentArbiter) -> None:
        """Each standard scenario result has all required keys.

        Spec: SCENARIO-VERIFY-172
        """
        rng = np.random.default_rng(42)
        results = _build_standard_scenarios(arbiter, rng)
        for r in results:
            assert "scenario_id" in r
            assert "type" in r
            assert "arbiter_index" in r
            assert "is_correct" in r
            assert "used_consensus_penalty" in r
            assert "energies_raw" in r
            assert "energies_adjusted" in r
            assert r["type"] == "standard"

    def test_adversarial_results_have_required_fields(self, arbiter: MultiAgentArbiter) -> None:
        """Each adversarial scenario result has all required keys.

        Spec: SCENARIO-VERIFY-173
        """
        rng = np.random.default_rng(42)
        results = _build_adversarial_scenarios(arbiter, rng)
        for r in results:
            assert r["type"] == "adversarial"
            assert "used_consensus_penalty" in r
            assert "energies_raw" in r
            assert "energies_adjusted" in r

    def test_adversarial_always_triggers_consensus_penalty(
        self, arbiter: MultiAgentArbiter
    ) -> None:
        """All adversarial scenarios trigger consensus penalty (two identical wrong agents).

        Two identical response strings → majority cluster ≥ 2 → detect_consensus returns True.

        Spec: SCENARIO-VERIFY-173, REQ-VERIFY-144
        """
        rng = np.random.default_rng(42)
        results = _build_adversarial_scenarios(arbiter, rng)
        for r in results:
            assert r["used_consensus_penalty"] is True, (
                f"Consensus penalty not triggered for {r['scenario_id']}"
            )

    def test_standard_correct_agent_wins(self, arbiter: MultiAgentArbiter) -> None:
        """Standard scenarios: correct agent (index 0) wins because it has lowest energy.

        Synthetic energy assignment guarantees correct has lowest energy.
        Spec: SCENARIO-VERIFY-172
        """
        rng = np.random.default_rng(42)
        results = _build_standard_scenarios(arbiter, rng)
        n_correct = sum(r["is_correct"] for r in results)
        assert n_correct == 6, f"Expected all 6 correct, got {n_correct}"

    def test_adversarial_correct_agent_wins_with_penalty(
        self, arbiter: MultiAgentArbiter
    ) -> None:
        """Adversarial scenarios: correct agent wins after consensus penalty flips the ranking.

        Synthetic setup: wrong majority has lower raw energy, but gap < penalty → flip.
        Spec: SCENARIO-VERIFY-173
        """
        rng = np.random.default_rng(42)
        results = _build_adversarial_scenarios(arbiter, rng)
        n_correct = sum(r["is_correct"] for r in results)
        assert n_correct == 6, f"Expected all 6 correct, got {n_correct}"
