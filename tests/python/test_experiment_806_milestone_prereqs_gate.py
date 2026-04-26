"""Tests for Exp 806 — Milestone Prereqs Gate and JEPA Wiring Guard.

Spec: REQ-INFRA-060, REQ-INFRA-061, SCENARIO-INFRA-069, SCENARIO-INFRA-070
"""

import json
import os
import tempfile

import pytest

from carnot.pipeline.jepa_wiring_guard import JepaWiringCheckResult, check_cpmi_wiring


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_triples_file(n: int, n_prefixes: int | None = None) -> str:
    """Write a temp CPMI triples JSON file with ``n`` entries.

    Each entry uses a distinct prefix_text so that n_input_pairs == n_prefixes
    (or n when n_prefixes is None, meaning one prefix per triple).
    """
    if n_prefixes is None:
        n_prefixes = n
    triples = []
    for i in range(n):
        prefix_idx = i % max(n_prefixes, 1)
        triples.append(
            {
                "prefix_text": f"prefix_{prefix_idx}",
                "positive_step": f"pos_{i}",
                "negative_step": f"neg_{i}",
            }
        )
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(triples, fh)
    return path


# ---------------------------------------------------------------------------
# REQ-INFRA-061 — AssertionError when ratio < threshold
# SCENARIO-INFRA-070
# ---------------------------------------------------------------------------


class TestCheckCpmiWiringAssertionFailure:
    """REQ-INFRA-061: check_cpmi_wiring raises AssertionError when ratio < min."""

    def test_assertion_raised_when_ratio_equals_one(self) -> None:
        """SCENARIO-INFRA-070: ratio=1.0 (no augmentation) raises AssertionError."""
        # 3 triples, all with the same prefix → ratio = 3/3 = 1.0, below default 1.5.
        path = _make_triples_file(n=3, n_prefixes=3)
        try:
            with pytest.raises(AssertionError) as exc_info:
                check_cpmi_wiring(path, min_augmentation_ratio=1.5)
            assert "CPMI corpus not wired in" in str(exc_info.value)
        finally:
            os.unlink(path)

    def test_assertion_raised_when_ratio_below_custom_threshold(self) -> None:
        """REQ-INFRA-061: custom min_augmentation_ratio is enforced."""
        # 4 triples, 2 unique prefixes → ratio = 4/2 = 2.0 < 3.0.
        path = _make_triples_file(n=4, n_prefixes=2)
        try:
            with pytest.raises(AssertionError):
                check_cpmi_wiring(path, min_augmentation_ratio=3.0)
        finally:
            os.unlink(path)

    def test_file_not_found_raises_file_not_found_error(self) -> None:
        """REQ-INFRA-061: missing triples file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            check_cpmi_wiring("/nonexistent/path/triples.json")


# ---------------------------------------------------------------------------
# REQ-INFRA-061 — is_wired=True when ratio >= threshold
# SCENARIO-INFRA-069
# ---------------------------------------------------------------------------


class TestCheckCpmiWiringSuccess:
    """REQ-INFRA-061: check_cpmi_wiring returns is_wired=True when ratio is sufficient."""

    def test_is_wired_true_when_ratio_meets_threshold(self) -> None:
        """SCENARIO-INFRA-069: ratio >= threshold → is_wired=True, no AssertionError."""
        # 6 triples, 2 unique prefixes → ratio = 6/2 = 3.0 >= 1.5.
        path = _make_triples_file(n=6, n_prefixes=2)
        try:
            result = check_cpmi_wiring(path, min_augmentation_ratio=1.5)
            assert result.is_wired is True
            assert result.augmentation_ratio == pytest.approx(3.0)
        finally:
            os.unlink(path)

    def test_returns_correct_triple_count(self) -> None:
        """REQ-INFRA-061: n_triples field matches actual file size."""
        path = _make_triples_file(n=10, n_prefixes=2)
        try:
            result = check_cpmi_wiring(path, min_augmentation_ratio=1.5)
            assert result.n_triples == 10
        finally:
            os.unlink(path)

    def test_returns_correct_pair_count(self) -> None:
        """REQ-INFRA-061: n_input_pairs field matches unique prefix_text values."""
        path = _make_triples_file(n=10, n_prefixes=5)
        try:
            result = check_cpmi_wiring(path, min_augmentation_ratio=1.5)
            assert result.n_input_pairs == 5
        finally:
            os.unlink(path)

    def test_honest_verdict_contains_ratio(self) -> None:
        """REQ-INFRA-061: honest_verdict is a non-empty human-readable string."""
        path = _make_triples_file(n=6, n_prefixes=2)
        try:
            result = check_cpmi_wiring(path, min_augmentation_ratio=1.5)
            assert "ratio=" in result.honest_verdict
            assert result.triples_path == path
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# REQ-INFRA-061 — JepaWiringCheckResult dataclass types
# ---------------------------------------------------------------------------


class TestJepaWiringCheckResultFields:
    """REQ-INFRA-061: JepaWiringCheckResult fields are typed correctly."""

    def test_all_fields_present_and_typed(self) -> None:
        """REQ-INFRA-061: dataclass exposes all required fields with correct types."""
        path = _make_triples_file(n=6, n_prefixes=2)
        try:
            result = check_cpmi_wiring(path, min_augmentation_ratio=1.5)
            assert isinstance(result, JepaWiringCheckResult)
            assert isinstance(result.triples_path, str)
            assert isinstance(result.n_triples, int)
            assert isinstance(result.n_input_pairs, int)
            assert isinstance(result.augmentation_ratio, float)
            assert isinstance(result.is_wired, bool)
            assert isinstance(result.honest_verdict, str)
        finally:
            os.unlink(path)
