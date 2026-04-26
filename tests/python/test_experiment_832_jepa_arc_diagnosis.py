"""Tests for scripts/experiment_832_jepa_arc_diagnosis.py.

Traces to: REQ-LEARN-832-001, SCENARIO-LEARN-832-001

**What we test:**
    - diagnose_domain() correctly identifies is_anti_correlated when wrong direction.
    - diagnose_domain() correctly identifies is_uncertain when scores are equal.
    - diagnose_domain() correctly marks is_working when model discriminates correctly.
    - _compute_feature_vector() returns 8-dim list.
    - _variance() and _mean() are numerically correct.
    - build_recommendation() produces meaningful strings for each failure mode.
    - compute_honest_verdict() maps findings to correct verdict labels.
    - The deliverable JSON exists and has all required schema fields.

All tests run on CPU, no GPU or live model needed — a fake model stub is used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test — all public symbols plus private helpers.
from scripts.experiment_832_jepa_arc_diagnosis import (
    N_ARC_TRAINING_PAIRS,
    SYNTHETIC_STEPS,
    _compute_feature_vector,
    _mean,
    _variance,
    build_recommendation,
    compute_honest_verdict,
    diagnose_domain,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal fake JEPA v23 model stubs
# ---------------------------------------------------------------------------


def _make_model(
    energy_correct: float = 0.3,
    energy_incorrect: float = 0.7,
    vocab: dict[str, int] | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics JEPAv23Predictor.

    Args:
        energy_correct:  What predict_energy() returns for "correct" prefix/step pairs.
        energy_incorrect: What predict_energy() returns for "incorrect" prefix/step pairs.
        vocab:           Vocabulary dict for the _vectoriser.

    The mock alternates: every other call is treated as correct/incorrect by recording
    call count — but since diagnose_domain() calls separately for correct and incorrect
    lists, the easiest approach is to patch predict_energy directly via side_effect.
    """
    if vocab is None:
        vocab = {"step": 0, "the": 1, "is": 2, "a": 3}

    mock = MagicMock()

    # _vectoriser.transform — returns a short non-zero vector.
    mock._vectoriser.transform.return_value = [0.1, 0.2, 0.0, 0.0]
    mock._vectoriser._tokenise.side_effect = lambda text: text.lower().split()
    mock._vectoriser._vocab = vocab

    # encode — return a fixed 4-D unit vector.
    mock.encode.return_value = [0.5, 0.5, 0.5, 0.5]

    return mock


def _make_working_model() -> MagicMock:
    """Model that correctly scores incorrect steps higher than correct steps."""
    model = _make_model()

    call_count = [0]

    def side_effect(prefix: str, step: str) -> float:
        # The diagnose_domain function calls predict_energy once per step in
        # correct_steps, then once per step in incorrect_steps.  We exploit
        # that the correct steps list contains "Step 1" strings and incorrect
        # have different text — but it's simpler to use a toggle.
        call_count[0] += 1
        # First 10 calls → correct steps → low energy.
        # Next 10 calls → incorrect steps → high energy.
        if call_count[0] <= 10:
            return 0.3
        return 0.7

    model.predict_energy.side_effect = side_effect
    return model


def _make_anti_correlated_model() -> MagicMock:
    """Model that INVERTS scores (correct > incorrect — anti-correlated)."""
    model = _make_model()
    call_count = [0]

    def side_effect(prefix: str, step: str) -> float:
        call_count[0] += 1
        if call_count[0] <= 10:
            return 0.9  # correct steps get HIGH energy (bad)
        return 0.5  # incorrect steps get lower energy

    model.predict_energy.side_effect = side_effect
    return model


def _make_uncertain_model() -> MagicMock:
    """Model that returns the same energy for correct and incorrect (uncertain)."""
    model = _make_model()
    model.predict_energy.return_value = 0.5  # identical for all calls
    return model


# ---------------------------------------------------------------------------
# _variance and _mean
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for _variance() and _mean() utility functions.

    Traces to: REQ-LEARN-832-001
    """

    def test_variance_uniform(self) -> None:
        """Variance of all-equal values must be zero."""
        assert _variance([3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_variance_known(self) -> None:
        """Variance of [0, 1, 2] = 2/3."""
        assert _variance([0.0, 1.0, 2.0]) == pytest.approx(2.0 / 3.0)

    def test_variance_empty(self) -> None:
        """Variance of empty list must return 0.0 without raising."""
        assert _variance([]) == 0.0

    def test_mean_known(self) -> None:
        """Mean of [1, 2, 3] = 2.0."""
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_mean_empty(self) -> None:
        """Mean of empty list must return 0.0 without raising."""
        assert _mean([]) == 0.0

    def test_mean_single(self) -> None:
        """Mean of a singleton is the element itself."""
        assert _mean([42.0]) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# _compute_feature_vector
# ---------------------------------------------------------------------------


class TestComputeFeatureVector:
    """Tests for _compute_feature_vector() producing an 8-dim list.

    Traces to: REQ-LEARN-832-001
    """

    def test_returns_eight_elements(self) -> None:
        """Feature vector must have exactly 8 elements."""
        model = _make_model()
        model.predict_energy.return_value = 0.4
        fv = _compute_feature_vector(model, "prefix text", "step text")
        assert len(fv) == 8

    def test_all_floats(self) -> None:
        """Every element must be a float (not None, not a string)."""
        model = _make_model()
        model.predict_energy.return_value = 0.4
        fv = _compute_feature_vector(model, "prefix text", "step text")
        assert all(isinstance(v, float) for v in fv)

    def test_cosine_dist_propagated(self) -> None:
        """The 4th element (index 3) must equal the mock predict_energy return value."""
        model = _make_model()
        model.predict_energy.return_value = 0.777
        fv = _compute_feature_vector(model, "p", "s")
        assert fv[3] == pytest.approx(0.777)

    def test_coverage_between_zero_and_one(self) -> None:
        """Vocabulary coverage (index 2) must be in [0, 1]."""
        model = _make_model()
        model.predict_energy.return_value = 0.5
        fv = _compute_feature_vector(model, "known step words", "another token test")
        assert 0.0 <= fv[2] <= 1.0


# ---------------------------------------------------------------------------
# diagnose_domain
# ---------------------------------------------------------------------------


class TestDiagnoseDomain:
    """Tests for diagnose_domain() classifying model behaviour per domain.

    Traces to: REQ-LEARN-832-001, SCENARIO-LEARN-832-001
    """

    def test_working_model_is_working(self) -> None:
        """A model that scores incorrect higher than correct must be marked is_working."""
        model = _make_working_model()
        result = diagnose_domain(
            model=model,
            domain="gsm8k",
            correct_steps=["correct " * 5] * 10,
            incorrect_steps=["wrong " * 5] * 10,
            prefix="prefix",
        )
        assert result["is_working"] is True
        assert result["is_anti_correlated"] is False
        assert result["is_uncertain"] is False

    def test_anti_correlated_detection(self) -> None:
        """Anti-correlated model (correct > incorrect by >0.02) must set is_anti_correlated."""
        model = _make_anti_correlated_model()
        result = diagnose_domain(
            model=model,
            domain="arc",
            correct_steps=["correct " * 5] * 10,
            incorrect_steps=["wrong " * 5] * 10,
            prefix="prefix",
        )
        assert result["is_anti_correlated"] is True
        assert result["is_working"] is False

    def test_uncertain_detection(self) -> None:
        """Uncertain model (identical scores) must set is_uncertain and not is_working."""
        model = _make_uncertain_model()
        result = diagnose_domain(
            model=model,
            domain="arc",
            correct_steps=["step " * 5] * 10,
            incorrect_steps=["step " * 5] * 10,
            prefix="prefix",
        )
        assert result["is_uncertain"] is True
        assert result["is_working"] is False

    def test_arc_domain_has_n_arc_training_pairs(self) -> None:
        """ARC domain result must include n_arc_training_pairs = N_ARC_TRAINING_PAIRS."""
        model = _make_uncertain_model()
        result = diagnose_domain(
            model=model,
            domain="arc",
            correct_steps=["x"] * 10,
            incorrect_steps=["y"] * 10,
            prefix="q",
        )
        assert result["n_arc_training_pairs"] == N_ARC_TRAINING_PAIRS
        assert result["n_arc_training_pairs"] == 0

    def test_non_arc_domain_n_arc_is_none(self) -> None:
        """Non-ARC domains must have n_arc_training_pairs=None."""
        model = _make_working_model()
        result = diagnose_domain(
            model=model,
            domain="gsm8k",
            correct_steps=["c"] * 10,
            incorrect_steps=["i"] * 10,
            prefix="p",
        )
        assert result["n_arc_training_pairs"] is None

    def test_correct_scores_length(self) -> None:
        """Diagnosis must record exactly 10 correct scores and 10 incorrect scores."""
        model = _make_uncertain_model()
        result = diagnose_domain(
            model=model,
            domain="humaneval",
            correct_steps=["c"] * 10,
            incorrect_steps=["i"] * 10,
            prefix="p",
        )
        assert len(result["correct_scores"]) == 10
        assert len(result["incorrect_scores"]) == 10

    def test_score_delta_sign_working(self) -> None:
        """score_delta = mean_incorrect - mean_correct must be positive for a working model."""
        model = _make_working_model()
        result = diagnose_domain(
            model=model,
            domain="humaneval",
            correct_steps=["c " * 5] * 10,
            incorrect_steps=["i " * 5] * 10,
            prefix="p",
        )
        assert result["score_delta"] > 0


# ---------------------------------------------------------------------------
# build_recommendation
# ---------------------------------------------------------------------------


class TestBuildRecommendation:
    """Tests for build_recommendation() producing actionable text.

    Traces to: REQ-LEARN-832-001
    """

    def _arc_finding(self, **overrides: Any) -> dict[str, Any]:
        """Build a minimal ARC finding dict with sensible defaults."""
        base: dict[str, Any] = {
            "mean_score_correct": 0.5,
            "mean_score_incorrect": 0.5,
            "score_delta": 0.0,
            "variance": 0.01,
            "mean_feature_norm": 1.0,
            "mean_vocab_coverage": 0.4,
            "is_anti_correlated": False,
            "is_uncertain": False,
            "is_working": True,
            "n_arc_training_pairs": 0,
        }
        base.update(overrides)
        return {"arc": base}

    def test_near_zero_norm_recommends_50_pairs(self) -> None:
        """Near-zero feature norm → recommend 50 ARC training pairs."""
        findings = self._arc_finding(mean_feature_norm=0.001)
        rec = build_recommendation(findings)
        assert "50" in rec
        assert "ARC" in rec

    def test_anti_correlated_recommends_50_pairs(self) -> None:
        """Anti-correlated ARC → recommend 50 training pairs."""
        findings = self._arc_finding(is_anti_correlated=True, mean_feature_norm=0.5)
        rec = build_recommendation(findings)
        assert "50" in rec

    def test_uncertain_recommends_30_pairs(self) -> None:
        """Uncertain ARC (signal = noise) → recommend 30 training pairs."""
        findings = self._arc_finding(is_uncertain=True, mean_feature_norm=0.5)
        rec = build_recommendation(findings)
        assert "30" in rec

    def test_working_returns_investigate_message(self) -> None:
        """Working ARC → 'investigate' rather than 'add training pairs'."""
        findings = self._arc_finding(is_working=True, mean_feature_norm=0.5)
        rec = build_recommendation(findings)
        assert "investigate" in rec.lower()

    def test_missing_arc_returns_insufficient_data(self) -> None:
        """Empty findings dict → fallback message about insufficient data."""
        rec = build_recommendation({})
        assert "insufficient" in rec.lower() or "diagnos" in rec.lower()


# ---------------------------------------------------------------------------
# compute_honest_verdict
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """Tests for compute_honest_verdict() mapping findings to verdict labels.

    Traces to: SCENARIO-LEARN-832-001
    """

    def _arc(self, **kw: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "is_anti_correlated": False,
            "is_uncertain": False,
            "is_working": True,
            "mean_feature_norm": 1.0,
        }
        base.update(kw)
        return {"arc": base}

    def test_anti_correlated_is_arc_diagnosis_found(self) -> None:
        """Anti-correlated ARC → arc_diagnosis_found."""
        assert compute_honest_verdict(self._arc(is_anti_correlated=True)) == "arc_diagnosis_found"

    def test_near_zero_norm_is_arc_diagnosis_found(self) -> None:
        """Near-zero feature norm → arc_diagnosis_found."""
        assert compute_honest_verdict(self._arc(mean_feature_norm=0.005)) == "arc_diagnosis_found"

    def test_uncertain_is_arc_diagnosis_found(self) -> None:
        """Uncertain (no signal) → arc_diagnosis_found."""
        assert (
            compute_honest_verdict(self._arc(is_uncertain=True, is_working=False))
            == "arc_diagnosis_found"
        )

    def test_working_is_arc_unexpected_viable(self) -> None:
        """Apparently working model → arc_unexpected_viable."""
        assert (
            compute_honest_verdict(self._arc(is_working=True, mean_feature_norm=1.0))
            == "arc_unexpected_viable"
        )

    def test_empty_findings_is_arc_diagnosis_uncertain(self) -> None:
        """Empty findings → arc_diagnosis_uncertain (cannot determine root cause)."""
        assert compute_honest_verdict({}) == "arc_diagnosis_uncertain"


# ---------------------------------------------------------------------------
# Deliverable artifact integration tests
# ---------------------------------------------------------------------------


class TestDeliverableArtifact:
    """Integration test: the written JSON must have all required schema fields.

    Traces to: REQ-LEARN-832-001, SCENARIO-LEARN-832-001
    """

    _artifact_path = Path("results/experiment_832_jepa_arc_collapse_diagnosis.json")

    def test_deliverable_exists(self) -> None:
        """The deliverable JSON must exist on disk after the experiment runs."""
        assert self._artifact_path.exists(), (
            f"Deliverable not found at {self._artifact_path}. "
            "Run scripts/experiment_832_jepa_arc_diagnosis.py first."
        )

    def _load(self) -> dict[str, Any]:
        with open(self._artifact_path) as fh:
            return json.load(fh)

    def test_required_schema_fields_present(self) -> None:
        """All REQUIRED_RESULT_FIELDS must be present in the artifact."""
        d = self._load()
        required = [
            "experiment",
            "schema",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "title",
        ]
        for field in required:
            assert field in d, f"Missing required field: {field}"

    def test_experiment_id(self) -> None:
        """experiment field must be 832."""
        assert self._load()["experiment"] == 832

    def test_status_success(self) -> None:
        """status must be 'success'."""
        assert self._load()["status"] == "success"

    def test_honest_verdict_valid(self) -> None:
        """honest_verdict must be one of the three recognised values."""
        verdict = self._load()["honest_verdict"]
        assert verdict in {
            "arc_diagnosis_found",
            "arc_diagnosis_uncertain",
            "arc_unexpected_viable",
        }

    def test_diagnosis_finding_has_three_domains(self) -> None:
        """diagnosis_finding must have exactly three domain keys."""
        d = self._load()
        assert set(d["diagnosis_finding"].keys()) == {"gsm8k", "humaneval", "arc"}

    def test_arc_finding_has_n_arc_training_pairs(self) -> None:
        """ARC finding must include n_arc_training_pairs = 0 (from Exp 824)."""
        d = self._load()
        assert d["diagnosis_finding"]["arc"]["n_arc_training_pairs"] == 0

    def test_recommendation_nonempty_string(self) -> None:
        """recommendation must be a non-empty string."""
        rec = self._load()["recommendation"]
        assert isinstance(rec, str) and len(rec) > 0

    def test_decision_class_detect(self) -> None:
        """decision_class must be 'detect' (diagnosis experiment)."""
        assert self._load()["decision_class"] == "detect"

    def test_schema_lists_all_keys(self) -> None:
        """schema field must match the actual top-level keys sorted."""
        d = self._load()
        schema_keys = set(d["schema"])
        actual_keys = set(d.keys())
        assert schema_keys == actual_keys

    def test_correct_and_incorrect_scores_per_domain(self) -> None:
        """Each domain finding must have 10 correct and 10 incorrect scores."""
        d = self._load()
        for domain_name, finding in d["diagnosis_finding"].items():
            assert len(finding["correct_scores"]) == 10, domain_name
            assert len(finding["incorrect_scores"]) == 10, domain_name


# ---------------------------------------------------------------------------
# Synthetic data sanity
# ---------------------------------------------------------------------------


class TestSyntheticSteps:
    """Sanity checks for the embedded test data.

    Traces to: REQ-LEARN-832-001
    """

    def test_all_three_domains_present(self) -> None:
        """SYNTHETIC_STEPS must contain gsm8k, humaneval, arc."""
        assert set(SYNTHETIC_STEPS.keys()) == {"gsm8k", "humaneval", "arc"}

    def test_each_domain_has_ten_correct_and_ten_incorrect(self) -> None:
        """Each domain must have exactly 10 correct and 10 incorrect examples."""
        for domain, steps in SYNTHETIC_STEPS.items():
            assert len(steps["correct"]) == 10, f"{domain} correct count wrong"
            assert len(steps["incorrect"]) == 10, f"{domain} incorrect count wrong"

    def test_n_arc_training_pairs_is_zero(self) -> None:
        """N_ARC_TRAINING_PAIRS must be 0, matching the Exp 824 artifact."""
        assert N_ARC_TRAINING_PAIRS == 0
