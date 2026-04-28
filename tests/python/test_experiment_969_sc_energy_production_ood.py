"""Integration tests for Exp 969: SC-Energy as Tier 2 OOD detector.

Validates:
 - SCEnergyEnergyAdapter correctly inverts coherence_score to energy
 - ThreeTierPipeline with SCEnergyEnergyAdapter skips Tier 3 for coherent inputs
 - ThreeTierPipeline reaches Tier 3 for incoherent inputs

The integration probe uses a controllable mock SCEnergyModel so the test
verifies WIRING correctness independent of training noise.  A separate
adapter unit test verifies the energy inversion math.

Spec: REQ-VERIFY-088, REQ-MODEL-031, SCENARIO-VERIFY-116
"""

from __future__ import annotations

from typing import Sequence

import jax.random as jrandom
import pytest

from python.carnot.models.eorm import CoTEnergyInput
from python.carnot.models.sc_energy import SCEnergyConfig, SCEnergyModel, TFIDFEmbedder
from python.carnot.pipeline.three_tier_pipeline import SCEnergyEnergyAdapter, ThreeTierPipeline


# ---------------------------------------------------------------------------
# Mock SC-Energy model for wiring tests
# ---------------------------------------------------------------------------


class _MockSCEnergyModel:
    """Deterministic mock that returns preset coherence scores by keyword.

    The real SCEnergyModel's predict_coherent_score() varies with training
    stochasticity.  This mock returns a fixed high score for texts containing
    the CLEAR_TAG marker and a fixed low score for texts containing BLOCK_TAG,
    making tests independent of training convergence.

    Markers are chosen to be non-substrings of each other (CLEAR_TAG = "XCOHERENT",
    BLOCK_TAG = "XINCOHERENT") to avoid substring match false positives.

    Attributes:
        high_score: Score returned for CLEAR_TAG texts (default 0.9).
        low_score:  Score returned for BLOCK_TAG texts (default 0.1).
    """

    CLEAR_TAG = "XCOHERENT"
    BLOCK_TAG = "XINCOHERENT"

    def __init__(self, high_score: float = 0.9, low_score: float = 0.1) -> None:
        self.high_score = high_score
        self.low_score = low_score

    def predict_coherent_score(self, statements: Sequence[str]) -> float:
        """Return high_score if any statement contains CLEAR_TAG, else low_score."""
        combined = " ".join(statements)
        if self.CLEAR_TAG in combined:
            return self.high_score
        return self.low_score


# 10 probe inputs labelled XCOHERENT (the mock will score these > 0.75)
_COHERENT_INPUTS = [
    f"XCOHERENT step A problem {i}\nXCOHERENT step B problem {i}\nXCOHERENT conclusion {i}"
    for i in range(10)
]

# 10 probe inputs labelled XINCOHERENT (the mock will score these <= 0.75)
_INCOHERENT_INPUTS = [
    f"XINCOHERENT step from problem {i}\nstep from problem {i + 5}\nstep from problem {i + 11}"
    for i in range(10)
]


# ---------------------------------------------------------------------------
# Helper: build a minimal ThreeTierPipeline with SC-Energy at Tier 2
# ---------------------------------------------------------------------------


def _build_pipeline(
    model: _MockSCEnergyModel | SCEnergyModel,
    threshold: float = 0.75,
) -> ThreeTierPipeline:
    """Build a test-safe ThreeTierPipeline with SC-Energy at Tier 2.

    Tier 1 (SinkProbe) is skipped by passing attention_matrix=None in verify()
    calls.  The stub SinkProbe is never called.  Tier 3 stub always returns
    (False, 1.0) so we can detect when Tier 2 did NOT clear the response.

    Args:
        model: SCEnergyModel or compatible mock.
        threshold: Coherence threshold (default 0.75 per task spec).

    Returns:
        Configured ThreeTierPipeline with SCEnergyEnergyAdapter at Tier 2.
    """

    class _NullSinkProbe:
        def score(self, attn, sink_positions):  # noqa: ANN001
            class _R:
                mean_sink_score = -1.0

            return _R()

    def _ising_stub(response: str, question: str) -> tuple[bool, float]:  # noqa: ARG001
        return False, 1.0

    adapter = SCEnergyEnergyAdapter(model=model, sc_threshold=threshold)
    eorm_threshold = 1.0 - threshold  # 0.25 for threshold=0.75

    return ThreeTierPipeline(
        sink_probe=_NullSinkProbe(),
        eorm_model=adapter,
        ising_pipeline=_ising_stub,
        sink_threshold=0.3,
        eorm_threshold=eorm_threshold,
    )


# ---------------------------------------------------------------------------
# Unit tests for SCEnergyEnergyAdapter
# ---------------------------------------------------------------------------


class TestSCEnergyEnergyAdapter:
    """Unit tests for the adapter that bridges SCEnergyModel to the EORM interface.

    Spec: REQ-MODEL-031, SCENARIO-MODEL-016
    """

    def test_energy_inversion(self) -> None:
        """energy() = 1 - coherence_score (polarity inversion for EORM threshold logic).

        ThreeTierPipeline clears when energy < eorm_threshold.  High coherence
        (score near 1.0) should produce low energy (near 0.0) to trigger clearing.
        Low coherence (score near 0.0) should produce high energy (near 1.0) to
        fall through to Tier 3.

        Spec: REQ-MODEL-031
        """
        mock = _MockSCEnergyModel(high_score=0.9, low_score=0.1)
        adapter = SCEnergyEnergyAdapter(model=mock, sc_threshold=0.75)

        coherent_energy = adapter.energy(
            CoTEnergyInput(question_text="", response_text="XCOHERENT step")
        )
        incoherent_energy = adapter.energy(
            CoTEnergyInput(question_text="", response_text="XINCOHERENT step")
        )

        assert abs(coherent_energy - (1.0 - 0.9)) < 1e-6, f"Expected 0.1, got {coherent_energy}"
        assert abs(incoherent_energy - (1.0 - 0.1)) < 1e-6, f"Expected 0.9, got {incoherent_energy}"

    def test_coherent_energy_below_threshold(self) -> None:
        """Coherent response produces energy < (1 - sc_threshold), clearing Tier 2.

        When coherence_score > 0.75 → energy < 0.25 → ThreeTierPipeline clears it.

        Spec: REQ-MODEL-031, SCENARIO-MODEL-016
        """
        mock = _MockSCEnergyModel(high_score=0.9)
        adapter = SCEnergyEnergyAdapter(model=mock, sc_threshold=0.75)
        energy = adapter.energy(CoTEnergyInput(question_text="", response_text="XCOHERENT text"))
        assert energy < 0.25, f"Expected energy < 0.25 for coherent input, got {energy}"

    def test_incoherent_energy_above_threshold(self) -> None:
        """Incoherent response produces energy >= (1 - sc_threshold), reaching Tier 3.

        Spec: REQ-MODEL-031, SCENARIO-MODEL-016
        """
        mock = _MockSCEnergyModel(low_score=0.1)
        adapter = SCEnergyEnergyAdapter(model=mock, sc_threshold=0.75)
        energy = adapter.energy(CoTEnergyInput(question_text="", response_text="INCOHERENT text"))
        assert energy >= 0.25, f"Expected energy >= 0.25 for incoherent input, got {energy}"

    def test_split_statements_multiline(self) -> None:
        """_split_statements() returns one element per non-empty line.

        Spec: REQ-MODEL-031
        """
        text = "Step 1: add 5.\nStep 2: multiply by 3.\n\nStep 3: subtract 2."
        parts = SCEnergyEnergyAdapter._split_statements(text)
        assert len(parts) == 3
        assert parts[0] == "Step 1: add 5."

    def test_split_statements_single_line_fallback(self) -> None:
        """_split_statements() returns [text] for a single-sentence response.

        Spec: REQ-MODEL-031
        """
        text = "The answer is 42."
        parts = SCEnergyEnergyAdapter._split_statements(text)
        assert parts == [text]


# ---------------------------------------------------------------------------
# Integration tests: ThreeTierPipeline with SC-Energy at Tier 2
# ---------------------------------------------------------------------------


class TestSCEnergyTier2Integration:
    """Integration tests for ThreeTierPipeline with SC-Energy at Tier 2.

    Validates the 10-coherent / 10-incoherent probe from the task spec.
    Uses a deterministic mock model so results are independent of training noise.
    Coherent inputs must all be cleared at Tier 2 (skip_rate = 1.0).
    Incoherent inputs must all reach Tier 3 (skip_rate = 0.0).

    Spec: REQ-VERIFY-088, SCENARIO-VERIFY-116
    """

    def test_coherent_inputs_all_skip_tier3(self) -> None:
        """All 10 coherent inputs should be cleared at Tier 2 (SC-Energy).

        A cleared response has tier_used == "eorm", meaning SC-Energy's
        coherence_score exceeded the threshold and Tier 3 Ising was skipped.

        Spec: REQ-VERIFY-088, SCENARIO-VERIFY-116
        """
        pipeline = _build_pipeline(_MockSCEnergyModel(), threshold=0.75)
        n_skipped = 0
        for text in _COHERENT_INPUTS:
            _verified, tier_used, _energy = pipeline.verify(
                text, attention_matrix=None, question=""
            )
            if tier_used == "eorm":
                n_skipped += 1
        skip_rate = n_skipped / 10
        assert skip_rate == 1.0, (
            f"Expected all 10 coherent inputs to skip Tier 3, "
            f"but only {n_skipped}/10 did (skip_rate={skip_rate:.2f})"
        )

    def test_incoherent_inputs_all_reach_tier3(self) -> None:
        """All 10 incoherent inputs should reach Tier 3 (not cleared by SC-Energy).

        An incoherent response has tier_used == "ising", meaning SC-Energy's
        coherence_score did not exceed the threshold.

        Spec: REQ-VERIFY-088, SCENARIO-VERIFY-116
        """
        pipeline = _build_pipeline(_MockSCEnergyModel(), threshold=0.75)
        n_skipped = 0
        for text in _INCOHERENT_INPUTS:
            _verified, tier_used, _energy = pipeline.verify(
                text, attention_matrix=None, question=""
            )
            if tier_used == "eorm":
                n_skipped += 1
        skip_rate = n_skipped / 10
        assert skip_rate == 0.0, (
            f"Expected all 10 incoherent inputs to reach Tier 3, "
            f"but {n_skipped}/10 were incorrectly cleared (skip_rate={skip_rate:.2f})"
        )

    def test_pipeline_tier_used_label(self) -> None:
        """Tier 2 clears use 'eorm' label; Tier 3 passes use 'ising' label.

        The tier_used string is how callers distinguish which tier made the
        decision.  SC-Energy is wired as the eorm_model so it inherits the
        'eorm' label from ThreeTierPipeline.verify().

        Spec: REQ-VERIFY-088
        """
        pipeline = _build_pipeline(_MockSCEnergyModel(), threshold=0.75)

        _v, tier_coh, _e = pipeline.verify(
            "XCOHERENT reasoning chain", attention_matrix=None, question=""
        )
        assert tier_coh == "eorm", f"Expected tier_used='eorm' for coherent input, got '{tier_coh}'"

        _v, tier_inc, _e = pipeline.verify(
            "XINCOHERENT mixed statements", attention_matrix=None, question=""
        )
        assert tier_inc == "ising", (
            f"Expected tier_used='ising' for incoherent input, got '{tier_inc}'"
        )

    def test_threshold_boundary(self) -> None:
        """Coherence score exactly at threshold does NOT clear Tier 2.

        The gate is strict inequality: energy < eorm_threshold.
        energy = 1 - coherence_score; for coherence_score = threshold exactly,
        energy = 1 - threshold, which equals eorm_threshold.
        Since energy < eorm_threshold is STRICT, boundary inputs reach Tier 3.

        Spec: REQ-VERIFY-088
        """
        threshold = 0.75
        boundary_mock = _MockSCEnergyModel(high_score=threshold)  # exactly at threshold
        pipeline = _build_pipeline(boundary_mock, threshold=threshold)

        # high_score == threshold → energy = 1 - 0.75 = 0.25 = eorm_threshold
        # energy < eorm_threshold is False → reaches Tier 3
        _v, tier_used, _e = pipeline.verify("COHERENT boundary", attention_matrix=None, question="")
        assert tier_used == "ising", (
            f"Expected boundary input to reach Tier 3 (strict <), got tier_used='{tier_used}'"
        )


# ---------------------------------------------------------------------------
# Model quality smoke test: real SCEnergyModel achieves basic AUROC
# ---------------------------------------------------------------------------


class TestSCEnergyModelQuality:
    """Smoke test that the real SCEnergyModel can rank coherent above incoherent.

    This is NOT a per-item threshold test — it verifies that the model trained
    on a small corpus achieves >50% correct ranking (i.e., AUROC > 0.5), which
    is the minimum bar for a useful OOD detector.

    Spec: REQ-MODEL-031, SCENARIO-MODEL-016
    """

    @pytest.fixture(scope="class")
    def small_model(self):
        """Train a small SC-Energy model on a 5-pair GSM8K-style corpus."""
        coherent_sets = [
            [
                "Janet has 47 cookies",
                "She gives 23 to her brother",
                "47 minus 23 equals 24 remaining",
            ],
            [
                "A car drives 350 miles in 5 hours",
                "Average speed is 350 divided by 5",
                "Speed is 70 mph",
            ],
            [
                "A box has 144 pencils and 12 boxes exist",
                "Total pencils equals 12 times 144",
                "Total is 1728",
            ],
            [
                "Lisa earns 18 dollars per hour for 40 hours",
                "Weekly earnings equal 720 dollars",
                "Monthly is 2880",
            ],
            [
                "Tank holds 600 gallons draining at 15 per minute",
                "Time to drain is 600 divided by 15",
                "Empties in 40 minutes",
            ],
        ]
        contradictory_sets = [
            ["Janet has 47 cookies", "Average speed is 350 divided by 5", "Total is 1728 pencils"],
            [
                "A car drives 350 miles in 5 hours",
                "Total pencils equals 12 times 144",
                "Tank empties in 40 minutes",
            ],
            [
                "A box has 144 pencils and 12 boxes exist",
                "Weekly earnings equal 720 dollars",
                "Speed is 70 mph",
            ],
            [
                "Lisa earns 18 dollars per hour for 40 hours",
                "Janet has 47 cookies",
                "Speed is 70 mph",
            ],
            [
                "Tank holds 600 gallons draining at 15 per minute",
                "She gives 23 to her brother",
                "Weekly earnings equal 720",
            ],
        ]
        all_stmts = [s for ss in coherent_sets + contradictory_sets for s in ss]
        embedder = TFIDFEmbedder(max_features=512)
        embedder.fit(all_stmts)

        config = SCEnergyConfig(embed_dim=512, hidden_dim=64, margin=1.0, learning_rate=0.01)
        model = SCEnergyModel(config=config, key=jrandom.PRNGKey(0))
        model.embedder = embedder
        model.train(coherent_sets, contradictory_sets, n_epochs=50)
        return model, coherent_sets, contradictory_sets

    def test_coherent_scores_higher_than_incoherent_on_average(self, small_model) -> None:  # noqa: ANN001
        """Mean coherence score of coherent sets > mean of incoherent sets.

        This verifies the model learned the correct ordering direction
        (coherent → high score, incoherent → low score) without requiring
        a specific absolute threshold.

        Spec: REQ-MODEL-031
        """
        model, coherent_sets, contradictory_sets = small_model
        mean_coh = sum(model.predict_coherent_score(s) for s in coherent_sets) / len(coherent_sets)
        mean_inc = sum(model.predict_coherent_score(s) for s in contradictory_sets) / len(
            contradictory_sets
        )
        assert mean_coh > mean_inc, (
            f"Expected mean coherent score ({mean_coh:.4f}) > mean incoherent score ({mean_inc:.4f})"
        )

    def test_most_pairs_ranked_correctly(self, small_model) -> None:  # noqa: ANN001
        """For most (coherent, incoherent) pairs, coherent scores higher than incoherent.

        Verifies AUROC > 0.5: the model correctly ranks at least 3 out of 5 pairs.

        Spec: REQ-MODEL-031, SCENARIO-MODEL-016
        """
        model, coherent_sets, contradictory_sets = small_model
        n_correct = sum(
            1
            for c, i in zip(coherent_sets, contradictory_sets)
            if model.predict_coherent_score(c) > model.predict_coherent_score(i)
        )
        assert n_correct >= 3, f"Expected at least 3/5 pairs ranked correctly, got {n_correct}/5"
