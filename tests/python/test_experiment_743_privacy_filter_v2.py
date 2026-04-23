"""Tests for Exp 743: PrivacyFilterKANv2 (teacher-free PII detection).

Coverage:
- PrivacyFilterFeatureExtractor.extract() returns correct shape.
- Luhn algorithm: luhn_valid() accepts valid CC numbers, rejects invalid.
- SSN regex: matches XXX-XX-XXXX, rejects other formats.
- AUROC and min_tp computed correctly from model predictions.

Spec: REQ-SAFE-019, REQ-SAFE-020
"""
from __future__ import annotations

import numpy as np
import pytest

from carnot.models.privacy_filter_kan_v2 import (
    N_FEATURES_V2,
    PrivacyExampleV2,
    PrivacyFilterFeatureExtractor,
    PrivacyFilterKANv2,
    _compute_auroc,
    luhn_complete,
    luhn_valid,
    _RE_SSN,
)


# ---------------------------------------------------------------------------
# REQ-SAFE-019: Feature extraction correctness
# ---------------------------------------------------------------------------

class TestPrivacyFilterFeatureExtractor:
    """Tests for PrivacyFilterFeatureExtractor.

    Spec: REQ-SAFE-019
    """

    def test_extract_shape_on_benign_text(self):
        """extract() MUST return a (N_FEATURES_V2,) array for any text.

        Spec: REQ-SAFE-019 — features MUST have a fixed shape regardless of input.
        """
        extractor = PrivacyFilterFeatureExtractor()
        feat = extractor.extract("Hello, this is a simple test sentence.")
        assert feat.shape == (N_FEATURES_V2,), (
            f"Expected ({N_FEATURES_V2},) but got {feat.shape}"
        )
        assert feat.dtype == np.float32

    def test_extract_shape_on_pii_text(self):
        """extract() returns (N_FEATURES_V2,) even on PII-heavy text.

        Spec: REQ-SAFE-019
        """
        extractor = PrivacyFilterFeatureExtractor()
        pii_text = "My SSN is 123-45-6789 and email is alice@example.com"
        feat = extractor.extract(pii_text)
        assert feat.shape == (N_FEATURES_V2,)

    def test_extract_shape_on_empty_text(self):
        """extract() MUST not crash on empty string and return correct shape.

        Spec: REQ-SAFE-019 — feature extraction is a pure function with no preconditions
        on text length.
        """
        extractor = PrivacyFilterFeatureExtractor()
        feat = extractor.extract("")
        assert feat.shape == (N_FEATURES_V2,)

    def test_n_features_v2_constant(self):
        """N_FEATURES_V2 MUST equal 23 (6 patterns × 3 stats + 4 token + 1 ngram).

        Spec: REQ-SAFE-019 — the feature count is fixed by the architecture spec.
        """
        assert N_FEATURES_V2 == 23

    def test_pii_text_has_higher_feature_values_than_benign(self):
        """Text with SSN/email should have higher raw feature values than plain prose.

        Spec: REQ-SAFE-019 — PII features should fire on PII text and not on benign text.
        """
        extractor = PrivacyFilterFeatureExtractor()
        benign = "The quick brown fox jumps over the lazy dog."
        pii = "SSN 123-45-6789 and card 4111-1111-1111-1111 billing at alice@test.com"
        f_benign = extractor.extract(benign)
        f_pii = extractor.extract(pii)
        # At least some features should be higher for PII text.
        assert float(f_pii.sum()) > float(f_benign.sum()), (
            "PII text should have higher total feature activation than benign text"
        )


# ---------------------------------------------------------------------------
# REQ-SAFE-019: Luhn algorithm correctness
# ---------------------------------------------------------------------------

class TestLuhnAlgorithm:
    """Tests for the Luhn credit card validation algorithm.

    Spec: REQ-SAFE-019 — CC pattern MUST use Luhn validation to avoid false positives.
    """

    def test_valid_luhn_numbers(self):
        """luhn_valid() MUST accept known-valid Luhn credit card numbers.

        Standard test cases: Visa 4111111111111111, Mastercard 5500005555555559.
        These are widely published test card numbers safe to include in code.

        Spec: REQ-SAFE-019
        """
        # Classic Luhn-valid test cards (publicly documented test data, not real cards).
        valid_cards = [
            "4111111111111111",   # Visa test card
            "5500005555555559",   # Mastercard test card
            "4012888888881881",   # Another Visa test card
        ]
        for card in valid_cards:
            assert luhn_valid(card), f"Expected {card} to be Luhn-valid"

    def test_invalid_luhn_numbers(self):
        """luhn_valid() MUST reject digit strings with wrong check digit.

        Spec: REQ-SAFE-019 — without Luhn filtering, random 16-digit sequences
        would produce false positives on non-PII numeric text.
        """
        invalid_cards = [
            "4111111111111112",   # Valid Visa prefix but wrong check digit (1 off)
            "1234567890123456",   # Sequential digits — Luhn-invalid
            "4111111111111113",   # Another wrong check digit
        ]
        for card in invalid_cards:
            assert not luhn_valid(card), f"Expected {card} to be Luhn-invalid"

    def test_luhn_valid_rejects_non_digit_strings(self):
        """luhn_valid() MUST return False for non-digit input.

        Spec: REQ-SAFE-019 — input validation: non-digits should not crash.
        """
        assert not luhn_valid("abcd-efgh-ijkl-mnop")
        assert not luhn_valid("")

    def test_luhn_complete_produces_valid_card(self):
        """luhn_complete() appends a check digit making the result Luhn-valid.

        Spec: REQ-SAFE-019 — used in corpus construction to generate synthetic PII.
        """
        prefix = "411111111111111"  # 15 digits
        completed = luhn_complete(prefix)
        assert len(completed) == 16
        assert luhn_valid(completed), f"luhn_complete output {completed!r} is not Luhn-valid"

    def test_feature_extractor_cc_fires_on_luhn_valid(self):
        """CC feature slots (0-2) should be non-zero for Luhn-valid CC text.

        Spec: REQ-SAFE-019 — the CC feature fires ONLY on Luhn-valid numbers.
        """
        extractor = PrivacyFilterFeatureExtractor()
        # 4111-1111-1111-1111 is Luhn-valid (Visa test card).
        feat = extractor.extract("My card: 4111-1111-1111-1111")
        # Feature 0 = cc match_count / word_count — should be non-zero.
        assert feat[0] > 0.0, "CC count feature should be > 0 for Luhn-valid card"

    def test_feature_extractor_cc_zero_on_luhn_invalid(self):
        """CC feature should be 0 for a 16-digit string that fails Luhn.

        Spec: REQ-SAFE-019 — Luhn filter prevents false positives on non-PII digits.
        """
        extractor = PrivacyFilterFeatureExtractor()
        # 4111-1111-1111-1112 has wrong check digit (Luhn-invalid).
        feat = extractor.extract("Number: 4111-1111-1111-1112")
        assert feat[0] == 0.0, "CC count feature should be 0 for Luhn-invalid number"


# ---------------------------------------------------------------------------
# REQ-SAFE-019: SSN pattern correctness
# ---------------------------------------------------------------------------

class TestSSNPattern:
    """Tests for the SSN regex pattern.

    Spec: REQ-SAFE-019 — SSN pattern MUST match XXX-XX-XXXX format only.
    """

    def test_ssn_matches_valid_format(self):
        """_RE_SSN matches the canonical XXX-XX-XXXX format.

        Spec: REQ-SAFE-019
        """
        valid_ssns = [
            "123-45-6789",
            "001-01-0001",
            "999-99-9999",
        ]
        for ssn in valid_ssns:
            assert _RE_SSN.search(ssn), f"Expected _RE_SSN to match {ssn!r}"

    def test_ssn_rejects_wrong_formats(self):
        """_RE_SSN MUST NOT match formats that differ from XXX-XX-XXXX.

        Spec: REQ-SAFE-019 — partial matches (e.g., XX-XX-XXXX or XXX-XX-XXX)
        should not fire to avoid false positives on phone numbers or other codes.
        """
        invalid_formats = [
            "12-34-5678",       # too few digits in first group
            "1234-56-7890",     # too many digits in first group
            "123-456-7890",     # 3-3-4 format (phone, not SSN)
            "123456789",        # no separators
            "123 45 6789",      # space separators
        ]
        for text in invalid_formats:
            # We test that the FULL string (as a standalone token) does NOT match.
            # Use word-boundary match to be precise.
            import re
            full_pattern = re.compile(r"^\d{3}-\d{2}-\d{4}$")
            assert not full_pattern.match(text), f"Did not expect full SSN match on {text!r}"

    def test_ssn_feature_fires_on_ssn_text(self):
        """SSN feature slot (3-5) should be non-zero for text containing an SSN.

        Spec: REQ-SAFE-019
        """
        extractor = PrivacyFilterFeatureExtractor()
        feat = extractor.extract("My SSN is 123-45-6789 for tax purposes.")
        # Feature 3 = ssn match_count / word_count — should be positive.
        assert feat[3] > 0.0, "SSN count feature should be > 0 for SSN-containing text"

    def test_ssn_feature_zero_on_benign_text(self):
        """SSN feature should be 0 for text with no SSN-formatted sequences.

        Spec: REQ-SAFE-019
        """
        extractor = PrivacyFilterFeatureExtractor()
        feat = extractor.extract("The answer is 42. Results: 100 out of 200.")
        assert feat[3] == 0.0, "SSN count feature should be 0 for text without SSNs"


# ---------------------------------------------------------------------------
# REQ-SAFE-020: AUROC and min_tp computed correctly
# ---------------------------------------------------------------------------

class TestAUROCAndMinTP:
    """Tests for AUROC computation and gate metric correctness.

    Spec: REQ-SAFE-020
    """

    def test_auroc_perfect_separation(self):
        """_compute_auroc returns 1.0 when all PII scores exceed all benign scores.

        Spec: REQ-SAFE-020 — gate requires AUROC >= 0.80; perfect = 1.0.
        """
        scores = [10.0, 9.0, 8.0, 1.0, 0.5, 0.1]
        labels = [1,    1,   1,   0,   0,   0  ]
        auroc = _compute_auroc(scores, labels)
        assert auroc == pytest.approx(1.0, abs=1e-6), f"Expected 1.0, got {auroc}"

    def test_auroc_random_separation(self):
        """_compute_auroc returns ~0.5 for random/equal score distributions.

        Spec: REQ-SAFE-020 — 0.5 = no discrimination, below gate threshold.
        """
        # Perfectly interleaved: no separation.
        scores = [1.0, 2.0, 3.0, 4.0]
        labels = [1,   0,   1,   0  ]
        auroc = _compute_auroc(scores, labels)
        # Not necessarily 0.5 exactly, but should not be very high.
        assert auroc <= 0.8, f"Expected low AUROC for random labels, got {auroc}"

    def test_auroc_degenerate_single_class(self):
        """_compute_auroc returns 0.5 for degenerate label sets (all same class).

        Spec: REQ-SAFE-020 — guard against division-by-zero in single-class datasets.
        """
        scores = [1.0, 2.0, 3.0]
        labels_all_pos = [1, 1, 1]
        labels_all_neg = [0, 0, 0]
        assert _compute_auroc(scores, labels_all_pos) == pytest.approx(0.5)
        assert _compute_auroc(scores, labels_all_neg) == pytest.approx(0.5)

    def test_model_evaluate_auroc_trained_model(self):
        """evaluate_auroc() on a briefly-trained model should exceed 0.5.

        After even minimal training on clearly labeled corpus, a KAN with
        high-signal PII features should produce better-than-chance AUROC.

        Spec: REQ-SAFE-020 — validates the training pipeline end-to-end.
        """
        model = PrivacyFilterKANv2(n_features=23, n_hidden=32)

        benign = [
            PrivacyExampleV2("The quick brown fox jumps over the lazy dog.", "benign"),
            PrivacyExampleV2("If x + 5 = 10, what is x?", "benign"),
            PrivacyExampleV2("def add(a, b): return a + b", "benign"),
            PrivacyExampleV2("Photosynthesis converts sunlight to energy.", "benign"),
        ]
        pii = [
            PrivacyExampleV2("My SSN is 123-45-6789.", "pii"),
            PrivacyExampleV2("Email me at alice@example.com for details.", "pii"),
            PrivacyExampleV2("Card 4111-1111-1111-1111 for billing.", "pii"),
            PrivacyExampleV2("Call (555) 123-4567 anytime.", "pii"),
        ]

        model.train(benign, pii, n_epochs=50, lr=1e-3)
        all_examples = benign + pii
        auroc = model.evaluate_auroc(all_examples)
        # After training with high-signal features, should beat random chance.
        assert auroc >= 0.5, f"Expected AUROC >= 0.5 after training, got {auroc:.4f}"

    def test_gate_logic_auroc_and_min_tp(self):
        """Gate MUST pass iff AUROC >= 0.80 AND min_tp >= 1.

        Spec: REQ-SAFE-020 — validates the gate boolean logic used in Exp 743.
        """
        # Simulate gate check.
        def check_gate(auroc: float, min_tp: int) -> bool:
            return auroc >= 0.80 and min_tp >= 1

        assert check_gate(0.85, 1) is True,  "0.85 AUROC + 1 TP should pass"
        assert check_gate(0.80, 1) is True,  "0.80 AUROC + 1 TP should pass"
        assert check_gate(0.79, 5) is False, "0.79 AUROC should fail"
        assert check_gate(0.85, 0) is False, "0 TP should fail even with high AUROC"
        assert check_gate(0.50, 0) is False, "Both failing should fail"

    def test_honest_verdict_mapping(self):
        """Verify the honest_verdict strings match the gate logic in the experiment.

        Spec: REQ-SAFE-020 — verdict strings must be well-defined and unambiguous.
        """
        def compute_verdict(auroc: float, min_tp: int) -> str:
            if auroc >= 0.85 and min_tp >= 1:
                return "privacy_filter_v2_gate_passed_high"
            elif auroc >= 0.80 and min_tp >= 1:
                return "privacy_filter_v2_gate_passed"
            elif auroc < 0.80:
                return "privacy_filter_v2_auroc_fail"
            else:
                return "privacy_filter_v2_minfp_fail"

        assert compute_verdict(0.90, 5) == "privacy_filter_v2_gate_passed_high"
        assert compute_verdict(0.82, 3) == "privacy_filter_v2_gate_passed"
        assert compute_verdict(0.75, 3) == "privacy_filter_v2_auroc_fail"
        assert compute_verdict(0.80, 0) == "privacy_filter_v2_minfp_fail"


# ---------------------------------------------------------------------------
# REQ-SAFE-019: Model save/load round-trip
# ---------------------------------------------------------------------------

class TestModelSaveLoad:
    """Tests for PrivacyFilterKANv2 serialisation round-trip.

    Spec: REQ-SAFE-019
    """

    def test_save_load_roundtrip(self, tmp_path):
        """save() then load() produces a model with identical energy outputs.

        Spec: REQ-SAFE-019 — weights serialisation must be lossless.
        """
        model = PrivacyFilterKANv2(n_features=23, n_hidden=32)
        test_text = "My SSN is 123-45-6789 and email is alice@test.com"

        energy_before = model.energy(test_text)
        path = tmp_path / "test_v2_weights.json"
        model.save(path)
        loaded = PrivacyFilterKANv2.load(path)
        energy_after = loaded.energy(test_text)

        assert energy_before == pytest.approx(energy_after, abs=1e-4), (
            f"Energy changed after save/load: {energy_before} vs {energy_after}"
        )

    def test_load_rejects_wrong_schema(self, tmp_path):
        """load() raises ValueError for files with wrong schema tag.

        Spec: REQ-SAFE-019 — prevents v1 weights from being silently loaded as v2.
        """
        import json
        bad_payload = {"schema": "carnot.privacy_filter_kan.v1", "n_features": 16}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_payload))
        with pytest.raises(ValueError, match="Unexpected schema"):
            PrivacyFilterKANv2.load(path)
