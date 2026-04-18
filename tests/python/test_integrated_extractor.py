"""Tests for carnot.extraction.integrated_extractor.IntegratedExtractor.

Spec: REQ-BENCH-015, SCENARIO-BENCH-035
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.extraction.integrated_extractor import IntegratedExtractor, Violation
from carnot.extraction.vericot_validator import StepVerdict, VeriCoTStepValidator
from carnot.extraction.vprm_verifier import RuleVerdict, VPRMArithmeticVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vericot_with_violations(violations: list[StepVerdict]) -> VeriCoTStepValidator:
    """Return a mock VeriCoTStepValidator whose detect_violations() returns violations."""
    v = MagicMock(spec=VeriCoTStepValidator)
    v.detect_violations.return_value = violations
    return v


def _make_vprm_with_violations(violations: list[RuleVerdict]) -> VPRMArithmeticVerifier:
    """Return a mock VPRMArithmeticVerifier whose detect_violations() returns violations."""
    p = MagicMock(spec=VPRMArithmeticVerifier)
    p.detect_violations.return_value = violations
    return p


def _step_verdict(step_text: str = "bad step") -> StepVerdict:
    return StepVerdict(step_idx=0, step_text=step_text, status="unsat", fol_premises=[])


def _rule_verdict(rule_name: str = "addition") -> RuleVerdict:
    return RuleVerdict(
        rule_name=rule_name,
        passed=False,
        computed_value=75.0,
        stated_value=76.0,
        error_magnitude=1.0,
    )


# ---------------------------------------------------------------------------
# TestIntegratedExtractor
# ---------------------------------------------------------------------------


class TestIntegratedExtractor:
    """Tests for IntegratedExtractor.extract() — SCENARIO-BENCH-035."""

    def test_extract_returns_vericot_violations_first(self) -> None:
        """SCENARIO-BENCH-035: VeriCoT is called first; its violations appear first."""
        sv = _step_verdict("vericot step")
        vericot = _make_vericot_with_violations([sv])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        violations = extractor.extract("some CoT text")

        assert len(violations) == 1
        assert violations[0].source == "vericot"
        assert violations[0].step_text == "vericot step"
        vericot.detect_violations.assert_called_once_with("some CoT text")

    def test_extract_returns_vprm_violations_after_vericot(self) -> None:
        """SCENARIO-BENCH-035: VPRM is called second; its violations follow VeriCoT."""
        rv = _rule_verdict("addition")
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([rv])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        violations = extractor.extract("some CoT text")

        assert len(violations) == 1
        assert violations[0].source == "vprm"
        assert violations[0].detail["rule_name"] == "addition"
        vprm.detect_violations.assert_called_once_with("some CoT text")

    def test_extract_merges_vericot_and_vprm_violations(self) -> None:
        """SCENARIO-BENCH-035: violations from both extractors are combined."""
        sv = _step_verdict()
        rv = _rule_verdict()
        vericot = _make_vericot_with_violations([sv])
        vprm = _make_vprm_with_violations([rv])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        violations = extractor.extract("cot text")

        assert len(violations) == 2
        assert violations[0].source == "vericot"
        assert violations[1].source == "vprm"

    def test_extract_returns_empty_when_no_violations(self) -> None:
        """No violations from either extractor → empty list."""
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        assert extractor.extract("clean reasoning step") == []

    def test_extract_calls_vericot_before_vprm(self) -> None:
        """SCENARIO-BENCH-035: execution order is VeriCoT then VPRM."""
        call_order: list[str] = []
        vericot = MagicMock(spec=VeriCoTStepValidator)
        vericot.detect_violations.side_effect = lambda t: call_order.append("vericot") or []
        vprm = MagicMock(spec=VPRMArithmeticVerifier)
        vprm.detect_violations.side_effect = lambda t: call_order.append("vprm") or []
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        extractor.extract("text")

        assert call_order == ["vericot", "vprm"]

    def test_extract_with_fallback_extractor(self) -> None:
        """Fallback extractor violations are tagged source='arithmetic'."""
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        fallback = MagicMock()
        fallback.extract.return_value = ["violation_object"]
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm, fallback=fallback)

        violations = extractor.extract("text with arithmetic")

        assert len(violations) == 1
        assert violations[0].source == "arithmetic"
        fallback.extract.assert_called_once_with("text with arithmetic", "arithmetic")

    def test_extract_skips_fallback_when_none(self) -> None:
        """No fallback configured → no arithmetic violations added."""
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm, fallback=None)

        violations = extractor.extract("text")

        assert violations == []

    def test_vericot_violation_detail_contains_step_idx(self) -> None:
        """VeriCoT violation detail includes step_idx from to_dict()."""
        sv = StepVerdict(step_idx=3, step_text="step", status="unsat", fol_premises=[])
        vericot = _make_vericot_with_violations([sv])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        violations = extractor.extract("text")

        assert violations[0].detail["step_idx"] == 3
        assert violations[0].detail["status"] == "unsat"

    def test_vprm_violation_detail_contains_error_magnitude(self) -> None:
        """VPRM violation detail includes error_magnitude and computed/stated values."""
        rv = _rule_verdict("multiplication")
        rv.computed_value = 42.0
        rv.stated_value = 40.0
        rv.error_magnitude = 2.0
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([rv])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        violations = extractor.extract("text")

        d = violations[0].detail
        assert d["rule_name"] == "multiplication"
        assert d["computed_value"] == 42.0
        assert d["stated_value"] == 40.0
        assert d["error_magnitude"] == 2.0

    # -----------------------------------------------------------------------
    # extractor_names_used
    # -----------------------------------------------------------------------

    def test_extractor_names_used_empty(self) -> None:
        """No violations → 'none'."""
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)
        assert extractor.extractor_names_used([]) == "none"

    def test_extractor_names_used_single_source(self) -> None:
        """Single source → just that name."""
        violations = [Violation(source="vericot", step_text="s")]
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)
        assert extractor.extractor_names_used(violations) == "vericot"

    def test_extractor_names_used_multiple_sources_sorted(self) -> None:
        """Multiple sources → sorted comma-separated names."""
        violations = [
            Violation(source="vprm", step_text="s1"),
            Violation(source="vericot", step_text="s2"),
        ]
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)
        result = extractor.extractor_names_used(violations)
        assert result == "vericot,vprm"

    # -----------------------------------------------------------------------
    # detection_rate
    # -----------------------------------------------------------------------

    def test_detection_rate_empty_samples(self) -> None:
        """Empty sample list → 0.0 (no division by zero)."""
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)
        assert extractor.detection_rate([]) == 0.0

    def test_detection_rate_all_detected(self) -> None:
        """All samples have violations → 1.0."""
        sv = _step_verdict()
        vericot = _make_vericot_with_violations([sv])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        samples = [{"cot_text": "bad step"}, {"cot_text": "another bad step"}]
        rate = extractor.detection_rate(samples)
        assert rate == 1.0

    def test_detection_rate_none_detected(self) -> None:
        """No sample has violations → 0.0."""
        vericot = _make_vericot_with_violations([])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        samples = [{"cot_text": "clean step"}, {"cot_text": "another clean step"}]
        rate = extractor.detection_rate(samples)
        assert rate == 0.0

    def test_detection_rate_partial(self) -> None:
        """Half of samples have violations → 0.5."""
        call_count = [0]

        def side_effect(text: str) -> list[StepVerdict]:
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return [_step_verdict()]
            return []

        vericot = MagicMock(spec=VeriCoTStepValidator)
        vericot.detect_violations.side_effect = side_effect
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        samples = [{"cot_text": "s1"}, {"cot_text": "s2"}]
        rate = extractor.detection_rate(samples)
        assert rate == 0.5

    def test_detection_rate_skips_missing_cot_text(self) -> None:
        """Samples without cot_text key are skipped (counted as no detection)."""
        sv = _step_verdict()
        vericot = _make_vericot_with_violations([sv])
        vprm = _make_vprm_with_violations([])
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        samples = [{"cot_text": "bad step"}, {"no_cot": "irrelevant"}]
        rate = extractor.detection_rate(samples)
        # 1 detected out of 2 total
        assert rate == 0.5

    # -----------------------------------------------------------------------
    # Real extractors (integration-style, use_mock=True so no GPU)
    # -----------------------------------------------------------------------

    def test_real_vericot_detects_arithmetic_error(self) -> None:
        """With real VeriCoTStepValidator(use_mock=True), detects 47+28==76 as unsat."""
        vericot = VeriCoTStepValidator(use_mock=True)
        vprm = VPRMArithmeticVerifier()
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        cot = "47 plus 28, which gives 76"
        violations = extractor.extract(cot)

        sources = {v.source for v in violations}
        assert "vericot" in sources

    def test_real_vericot_no_violation_on_correct(self) -> None:
        """47 plus 28 gives 75 is correct — no VeriCoT violation."""
        vericot = VeriCoTStepValidator(use_mock=True)
        vprm = VPRMArithmeticVerifier()
        extractor = IntegratedExtractor(vericot=vericot, vprm=vprm)

        cot = "47 plus 28, which gives 75"
        violations = extractor.extract(cot)

        # VeriCoT should find no violation; VPRM uses different patterns
        vericot_violations = [v for v in violations if v.source == "vericot"]
        assert vericot_violations == []
