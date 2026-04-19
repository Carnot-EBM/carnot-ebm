"""Tests for EnergyGuidedDecoder and EnergyGuidedConfig.

Spec: REQ-VERIFY-113, REQ-VERIFY-114, SCENARIO-VERIFY-149, SCENARIO-VERIFY-150
"""

from __future__ import annotations

import random

import pytest

from carnot.pipeline.energy_guided_decoder import EnergyGuidedConfig, EnergyGuidedDecoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _length_energy(text: str) -> float:
    """Toy energy: shorter text has lower energy.

    Used to make 'short' the minimum-energy candidate without any ML model.
    Energy = number of characters in the full text.
    """
    return float(len(text))


def _fixed_energy(mapping: dict[str, float]):
    """Return an energy function that maps text -> fixed score via substring lookup."""

    def _fn(text: str) -> float:
        for key, val in mapping.items():
            if text.endswith(key):
                return val
        return 0.0

    return _fn


# ---------------------------------------------------------------------------
# EnergyGuidedConfig
# ---------------------------------------------------------------------------


class TestEnergyGuidedConfig:
    def test_defaults(self):
        """REQ-VERIFY-113: default K=5, energy_weight=1.0."""
        cfg = EnergyGuidedConfig()
        assert cfg.k_candidates == 5
        assert cfg.energy_weight == 1.0

    def test_custom_values(self):
        """REQ-VERIFY-113: custom parameters are stored correctly."""
        cfg = EnergyGuidedConfig(k_candidates=3, energy_weight=0.5)
        assert cfg.k_candidates == 3
        assert cfg.energy_weight == 0.5


# ---------------------------------------------------------------------------
# EnergyGuidedDecoder.score_candidates
# ---------------------------------------------------------------------------


class TestScoreCandidates:
    def test_returns_one_score_per_candidate(self):
        """REQ-VERIFY-113: score_candidates returns list of same length as candidates."""
        decoder = EnergyGuidedDecoder(_length_energy)
        scores = decoder.score_candidates("hello", ["a", "bb", "ccc"])
        assert len(scores) == 3

    def test_scores_are_floats(self):
        """REQ-VERIFY-113: each score is a Python float."""
        decoder = EnergyGuidedDecoder(_length_energy)
        scores = decoder.score_candidates("x", ["a", "b"])
        assert all(isinstance(s, float) for s in scores)

    def test_longer_continuation_has_higher_energy(self):
        """REQ-VERIFY-113: length_energy increases with total text length."""
        decoder = EnergyGuidedDecoder(_length_energy)
        scores = decoder.score_candidates("prefix ", ["a", "longer_word"])
        # "prefix a" is shorter than "prefix longer_word" -> lower energy
        assert scores[0] < scores[1]


# ---------------------------------------------------------------------------
# EnergyGuidedDecoder.select_next
# ---------------------------------------------------------------------------


class TestSelectNext:
    def test_returns_minimum_energy_candidate(self):
        """SCENARIO-VERIFY-149: select_next returns the lowest-energy candidate."""
        # Energy map: 'good' continuation has lowest energy
        energy_map = {"good": -10.0, "bad": 0.0, "ugly": 5.0}
        decoder = EnergyGuidedDecoder(_fixed_energy(energy_map))
        result = decoder.select_next("prefix ", ["bad", "good", "ugly"])
        assert result == "good"

    def test_empty_candidates_raises(self):
        """select_next must raise ValueError when candidates is empty."""
        decoder = EnergyGuidedDecoder(_length_energy)
        with pytest.raises(ValueError):
            decoder.select_next("prefix", [])

    def test_single_candidate_returned(self):
        """If only one candidate, it must be returned regardless of energy."""
        decoder = EnergyGuidedDecoder(_length_energy)
        result = decoder.select_next("prefix", ["only"])
        assert result == "only"

    def test_energy_weight_zero_ignores_energy(self):
        """REQ-VERIFY-114: energy_weight=0.0 degenerates to random selection.

        With energy_weight=0.0 the EBM score has no influence.  Over many
        trials the decoder should return different candidates (i.e., is not
        always returning the same one regardless of energy ranking).
        """
        # Make candidate 'bad' have by far the worst (highest) energy
        energy_map = {"good": -100.0, "bad": 9999.0}
        cfg = EnergyGuidedConfig(energy_weight=0.0)
        decoder = EnergyGuidedDecoder(_fixed_energy(energy_map), config=cfg)

        random.seed(42)
        results = {decoder.select_next("p ", ["good", "bad"]) for _ in range(30)}
        # Without energy guidance both candidates should appear
        assert "bad" in results, "energy_weight=0.0 should sometimes return high-energy candidate"

    def test_default_config_used_when_none(self):
        """When no config is passed, EnergyGuidedConfig() defaults are used."""
        decoder = EnergyGuidedDecoder(_length_energy, config=None)
        assert decoder.config.k_candidates == 5
        assert decoder.config.energy_weight == 1.0


# ---------------------------------------------------------------------------
# EnergyGuidedDecoder.generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_appends_max_steps_words(self):
        """SCENARIO-VERIFY-150: generate produces exactly max_steps words appended."""
        decoder = EnergyGuidedDecoder(_length_energy)
        vocab = ["alpha", "beta", "gamma"]
        result = decoder.generate("start", vocab, max_steps=5)
        # "start" + 5 words separated by spaces = 6 tokens
        tokens = result.split()
        assert len(tokens) == 6  # "start" + 5 generated words
        assert result.startswith("start")

    def test_generate_words_from_vocab(self):
        """SCENARIO-VERIFY-150: every generated word must come from vocab."""
        decoder = EnergyGuidedDecoder(_length_energy)
        vocab = ["apple", "banana", "cherry"]
        result = decoder.generate("seed", vocab, max_steps=10)
        generated_words = result.split()[1:]  # drop "seed"
        for word in generated_words:
            assert word in vocab

    def test_generate_zero_steps_returns_prompt(self):
        """generate with max_steps=0 returns the original prompt unchanged."""
        decoder = EnergyGuidedDecoder(_length_energy)
        result = decoder.generate("hello", ["a", "b"], max_steps=0)
        assert result == "hello"

    def test_generate_energy_steers_toward_low_energy(self):
        """REQ-VERIFY-114: energy-guided generation always picks 'short' words."""
        # 'a' is 1 char, 'zzzzz' is 5 chars — energy = total text length
        # The decoder should always pick 'a'
        decoder = EnergyGuidedDecoder(_length_energy)
        result = decoder.generate("x", ["a", "zzzzz"], max_steps=5)
        # All generated words should be 'a' since it minimises total text length
        generated = result.split()[1:]
        assert all(w == "a" for w in generated)
