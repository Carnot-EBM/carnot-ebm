"""Tests for SemanticEnergyProbe — Tier 0f advisory hallucination detector.

Spec traces: REQ-VERIFY-155, SCENARIO-VERIFY-180, SCENARIO-VERIFY-181
"""

from __future__ import annotations

import pytest

from carnot.pipeline.semantic_energy_probe import (
    SemanticEnergyProbe,
    SemanticEnergyResult,
    _embed_sentences,
    _extract_sentences,
    _l2_normalize,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Coherent: 4 sentences about the same topic sharing many words
# ("dog", "barks", "tree", "yard" appear in every sentence)
COHERENT_RESPONSE = (
    "The dog barks at the tree in the yard. "
    "In the yard the dog barks at the tree loudly. "
    "The tree stands in the yard where the dog barks. "
    "At the tree in the yard the barking dog stands."
)

# Hallucinated: 3 coherent sentences + 1 rogue with zero vocabulary overlap
HALLUCINATED_RESPONSE = (
    "The dog barks at the tree in the yard. "
    "In the yard the dog barks at the tree loudly. "
    "The tree stands in the yard where the dog barks. "
    "Supernovae eject stellar material into interstellar space across galaxies."
)

# Strongly hallucinated: 2 sentences with completely disjoint vocabulary
# Guaranteed is_unstable=True because pairwise kernel ≈ exp(-2) ≈ 0.135 > -0.5
STRONGLY_HALLUCINATED = (
    "The dog barks loudly at the yard tree. "
    "Supernovae eject stellar material across vast interstellar space."
)


# ---------------------------------------------------------------------------
# REQ-VERIFY-155: core probe behaviour
# ---------------------------------------------------------------------------


class TestCoherentResponseLowEnergy:
    """SCENARIO-VERIFY-180: coherent response → energy < threshold → not unstable."""

    def test_coherent_response_low_energy(self):
        """Coherent multi-sentence response produces energy below the threshold.

        The COHERENT_RESPONSE shares dog/barks/tree/yard across all 4 sentences,
        producing high TF cosine similarity → kernel values → energy < -0.5.

        Spec: REQ-VERIFY-155, SCENARIO-VERIFY-180
        """
        probe = SemanticEnergyProbe(sigma=1.0, threshold=-0.5, embedding_dim=64)
        result = probe.score(COHERENT_RESPONSE)
        assert isinstance(result, SemanticEnergyResult)
        assert result.energy < result.threshold, (
            f"Expected energy={result.energy:.4f} < threshold={result.threshold:.4f}"
        )
        assert result.is_unstable is False
        assert result.sentence_count >= 4


class TestHallucinatedResponseHighEnergy:
    """SCENARIO-VERIFY-181: hallucinated response → higher energy than coherent."""

    def test_hallucinated_response_high_energy(self):
        """Response with rogue sentence has higher (less negative) energy than coherent.

        HALLUCINATED_RESPONSE = 3 coherent sentences + 1 rogue with zero vocabulary
        overlap.  The rogue sentence pulls the pairwise mean kernel down, raising energy.

        Spec: REQ-VERIFY-155, SCENARIO-VERIFY-181
        """
        probe = SemanticEnergyProbe(sigma=1.0, threshold=-0.5, embedding_dim=64)
        coherent_result = probe.score(COHERENT_RESPONSE)
        hallucinated_result = probe.score(HALLUCINATED_RESPONSE)
        assert hallucinated_result.energy > coherent_result.energy, (
            f"Hallucinated energy={hallucinated_result.energy:.4f} should exceed "
            f"coherent energy={coherent_result.energy:.4f}"
        )

    def test_hallucinated_result_is_unstable(self):
        """2-sentence response with zero vocabulary overlap is flagged is_unstable=True.

        With only 2 sentences and no shared words, the single pair has
        kernel ≈ exp(-2) ≈ 0.135, so energy ≈ -0.135 > threshold=-0.5.

        Spec: REQ-VERIFY-155, SCENARIO-VERIFY-181
        """
        probe = SemanticEnergyProbe(sigma=1.0, threshold=-0.5, embedding_dim=64)
        result = probe.score(STRONGLY_HALLUCINATED)
        assert result.is_unstable is True, (
            f"Expected is_unstable=True but energy={result.energy:.4f}, "
            f"threshold={result.threshold:.4f}"
        )

    def test_hallucinated_cluster_entropy_positive(self):
        """Multi-sentence hallucinated response has non-trivial cluster entropy.

        Spec: SCENARIO-VERIFY-181
        """
        probe = SemanticEnergyProbe()
        result = probe.score(HALLUCINATED_RESPONSE)
        assert result.cluster_entropy > 0.0


class TestSingleSentenceReturnsZeroEnergy:
    """Single-sentence response: no pairs → energy=0, not unstable."""

    def test_single_sentence_returns_zero_energy(self):
        """Score a single sentence: energy=0.0, is_unstable=False, sentence_count=1.

        No pairwise comparison is possible with one sentence.

        Spec: REQ-VERIFY-155
        """
        probe = SemanticEnergyProbe()
        result = probe.score("The sky is blue.")
        assert result.energy == pytest.approx(0.0)
        assert result.is_unstable is False
        assert result.sentence_count <= 1

    def test_empty_response_returns_zero_energy(self):
        """Empty string: energy=0.0, no crash.

        Spec: REQ-VERIFY-155
        """
        probe = SemanticEnergyProbe()
        result = probe.score("")
        assert result.energy == pytest.approx(0.0)
        assert result.is_unstable is False


class TestL2NormalizationApplied:
    """Verify that _l2_normalize produces unit-norm vectors."""

    def test_l2_normalization_applied(self):
        """L2-normalised vector has unit norm.

        Spec: REQ-VERIFY-155
        """
        import math

        vec = [3.0, 4.0]
        normed = _l2_normalize(vec)
        norm = math.sqrt(sum(x * x for x in normed))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_l2_normalize_zero_vector(self):
        """Zero vector normalises to zero vector (no division by zero).

        Spec: REQ-VERIFY-155
        """
        normed = _l2_normalize([0.0, 0.0, 0.0])
        assert all(v == pytest.approx(0.0) for v in normed)

    def test_embed_sentences_unit_norm(self):
        """_embed_sentences returns L2-normalised embeddings.

        Non-zero embeddings must have unit L2 norm.

        Spec: REQ-VERIFY-155
        """
        import math

        sentences = ["The cat sat on the mat.", "The dog ran in the park."]
        embeddings = _embed_sentences(sentences, embedding_dim=32)
        for emb in embeddings:
            norm = math.sqrt(sum(x * x for x in emb))
            # Zero embeddings are acceptable (empty sentences); non-zero must be unit
            if norm > 1e-9:
                assert norm == pytest.approx(1.0, abs=1e-6)


class TestSemanticEnergyResultFields:
    """SemanticEnergyResult must carry all required fields."""

    def test_result_fields_present(self):
        """score() returns SemanticEnergyResult with all required fields populated.

        Spec: REQ-VERIFY-155
        """
        probe = SemanticEnergyProbe(sigma=0.8, threshold=-0.3, embedding_dim=32)
        result = probe.score(COHERENT_RESPONSE)
        assert hasattr(result, "energy")
        assert hasattr(result, "is_unstable")
        assert hasattr(result, "sentence_count")
        assert hasattr(result, "cluster_entropy")
        assert hasattr(result, "threshold")
        assert result.threshold == pytest.approx(-0.3)
        assert isinstance(result.sentence_count, int)
        assert isinstance(result.cluster_entropy, float)

    def test_result_is_frozen(self):
        """SemanticEnergyResult is a frozen dataclass (immutable).

        Spec: REQ-VERIFY-155
        """
        probe = SemanticEnergyProbe()
        result = probe.score("One sentence here.")
        with pytest.raises((AttributeError, TypeError)):
            result.energy = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pipeline wiring tests
# ---------------------------------------------------------------------------


class TestTier0fWiredInPipeline:
    """test_tier0f_wired_in_pipeline: SemanticEnergyProbe wired as advisory in verify()."""

    def test_tier0f_wired_in_pipeline(self):
        """verify() records tier_0f_semantic_energy in certificate when probe is supplied.

        Spec: REQ-VERIFY-155
        """
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"])
        probe = SemanticEnergyProbe()
        result = pipeline.verify(
            question="What is photosynthesis?",
            response=COHERENT_RESPONSE,
            semantic_energy_probe=probe,
        )
        assert "tier_0f_semantic_energy" in result.certificate
        cert = result.certificate["tier_0f_semantic_energy"]
        assert "energy" in cert
        assert "is_unstable" in cert
        assert "sentence_count" in cert
        assert "cluster_entropy" in cert
        assert "threshold" in cert


class TestAdvisoryNoShortCircuit:
    """test_advisory_no_short_circuit: is_unstable=True does not short-circuit the pipeline."""

    def test_advisory_no_short_circuit(self):
        """Pipeline runs to completion even when probe flags is_unstable=True.

        Advisory signal only — tier 1-3 continue running regardless of is_unstable.
        We use threshold=-2.0 to guarantee is_unstable=True (energy is always > -2.0).

        Spec: REQ-VERIFY-155
        """
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"])
        # threshold=-2.0: any energy > -2.0 is flagged unstable.
        # All probe energies are in (-1, 0], so is_unstable=True for any response.
        probe_force_unstable = SemanticEnergyProbe(threshold=-2.0)
        result = pipeline.verify(
            question="What is 2+2?",
            response="The answer is 4. Mathematics confirms this. Addition works.",
            semantic_energy_probe=probe_force_unstable,
        )
        # Pipeline must complete (not raise, not return fast-path skip due to probe)
        assert result is not None
        assert isinstance(result.verified, bool)
        # Certificate must have tier_0f entry
        assert "tier_0f_semantic_energy" in result.certificate
        assert result.certificate["tier_0f_semantic_energy"]["is_unstable"] is True
        # The pipeline still computed constraints (not short-circuited by the probe)
        # mode should be FULL or another valid tier mode — not a probe-short-circuit mode
        assert result.mode in ("FULL", "NUP_PROBE_FAST_PATH", "THINK_PROBE_FAST_PATH",
                               "FAST_PATH", "RUST")


# ---------------------------------------------------------------------------
# Sentence extraction edge cases
# ---------------------------------------------------------------------------


class TestExtractSentences:
    """Edge cases for _extract_sentences."""

    def test_question_mark_split(self):
        """Sentences split on '? ' as well as '. '.

        Spec: REQ-VERIFY-155
        """
        text = "Is this correct? Yes it is. Definitely true."
        sentences = _extract_sentences(text)
        assert len(sentences) >= 2

    def test_single_fragment(self):
        """Single fragment (no split point) returns one-element list.

        Spec: REQ-VERIFY-155
        """
        text = "This is a single sentence without a split"
        sentences = _extract_sentences(text)
        assert len(sentences) == 1
