"""Tests for Exp 625: Tier 1 FR-11 Self-Learning Relay.

100% targeted coverage on functions added in
scripts/experiment_625_tier1_fr11_relay.py:
  - _load_json()
  - _build_synthetic_violations()
  - _compute_fp_rate()

The experiment's top-level run is not re-executed here — the deliverable
JSON produced by running the script is the integration test artifact.  These
unit tests verify the helper logic that determines mode selection and FP-rate
measurement.

Spec: REQ-LEARN-080, SCENARIO-LEARN-124, SCENARIO-LEARN-125
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

os.environ.setdefault("CARNOT_IS_CI", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import scripts.experiment_625_tier1_fr11_relay as exp625  # noqa: E402


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    """REQ-LEARN-080: upstream experiment JSON must load gracefully."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        # SCENARIO-LEARN-124: real-mode path loads Exp 620 violations.
        f = tmp_path / "result.json"
        f.write_text(json.dumps({"signed_improvement": 1.5, "n_violations_found": 3}))
        data = exp625._load_json(f)
        assert data["signed_improvement"] == 1.5

    def test_returns_empty_dict_for_missing_file(self, tmp_path: Path) -> None:
        # SCENARIO-LEARN-125: blocked upstream degrades to synthetic fallback.
        data = exp625._load_json(tmp_path / "nonexistent.json")
        assert data == {}

    def test_returns_empty_dict_for_malformed_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        data = exp625._load_json(f)
        assert data == {}


# ---------------------------------------------------------------------------
# _build_synthetic_violations
# ---------------------------------------------------------------------------


class TestBuildSyntheticViolations:
    """REQ-LEARN-080: synthetic violations must cover all four arithmetic families."""

    def test_returns_25_total_violations(self) -> None:
        # SCENARIO-LEARN-125: 25 violations generated in synthetic fallback mode.
        violations = exp625._build_synthetic_violations(25)
        total = sum(v.count for v in violations)
        assert total == 25

    def test_covers_all_four_families(self) -> None:
        violations = exp625._build_synthetic_violations(25)
        types = {v.type for v in violations}
        assert types == {"carry", "sign", "unit", "comparison"}

    def test_each_violation_has_example_steps(self) -> None:
        violations = exp625._build_synthetic_violations(25)
        for v in violations:
            assert len(v.example_steps) > 0

    def test_small_n_distributes_evenly(self) -> None:
        violations = exp625._build_synthetic_violations(4)
        total = sum(v.count for v in violations)
        assert total == 4
        assert len(violations) == 4

    def test_returns_correct_type_for_large_n(self) -> None:
        violations = exp625._build_synthetic_violations(100)
        total = sum(v.count for v in violations)
        assert total == 100


# ---------------------------------------------------------------------------
# _compute_fp_rate
# ---------------------------------------------------------------------------


class TestComputeFpRate:
    """REQ-LEARN-080: FP-rate proxy must correctly flag/not-flag correct responses."""

    def _make_monitor_with_patterns(self, patterns: list[str], threshold: int = 1):
        """Return a ConstraintAdditionFromMemory pre-seeded above threshold."""
        from carnot.pipeline.constraint_addition import ConstraintAdditionFromMemory

        mon = ConstraintAdditionFromMemory(threshold=threshold, pipeline=None)
        for p in patterns:
            for _ in range(threshold):
                mon.observe(p, "test_step")
        return mon

    def test_zero_fp_rate_when_no_patterns_observed(self) -> None:
        # SCENARIO-LEARN-125: empty monitor → 0.0 FP rate.
        from carnot.pipeline.constraint_addition import ConstraintAdditionFromMemory

        mon = ConstraintAdditionFromMemory(threshold=5, pipeline=None)
        rate = exp625._compute_fp_rate(mon, ["clean answer"], ["bad answer"])
        assert rate == 0.0

    def test_fp_rate_one_when_all_correct_flagged(self) -> None:
        # All correct texts contain a keyword the monitor has observed.
        mon = self._make_monitor_with_patterns(["carry"])
        correct = ["carry error here", "carry propagation issue"]
        rate = exp625._compute_fp_rate(mon, correct, [])
        assert rate == 1.0

    def test_fp_rate_zero_when_no_correct_flagged(self) -> None:
        mon = self._make_monitor_with_patterns(["carry"])
        correct = ["totally clean answer", "sum is 42"]
        rate = exp625._compute_fp_rate(mon, correct, [])
        assert rate == 0.0

    def test_fp_rate_partial(self) -> None:
        mon = self._make_monitor_with_patterns(["sign"])
        correct = ["sign flip happened", "clean response", "another clean one"]
        rate = exp625._compute_fp_rate(mon, correct, [])
        # 1 out of 3 correct texts flagged.
        assert abs(rate - 1 / 3) < 1e-9

    def test_fp_rate_empty_correct_corpus_returns_zero(self) -> None:
        mon = self._make_monitor_with_patterns(["carry"])
        rate = exp625._compute_fp_rate(mon, [], ["bad"])
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Deliverable JSON sanity check
# ---------------------------------------------------------------------------


class TestDeliverableJson:
    """Integration: deliverable JSON must contain all required schema fields."""

    _REQUIRED_FIELDS = {
        "schema",
        "mode",
        "n_violations_used",
        "fr11_real_violations_confirmed",
        "fp_rate_before",
        "fp_rate_after",
        "fp_rate_delta",
        "honest_verdict",
    }

    def test_deliverable_exists(self) -> None:
        # SCENARIO-LEARN-125: synthetic fallback writes deliverable.
        path = _REPO_ROOT / "results/experiment_625_tier1_fr11_relay.json"
        assert path.exists(), f"Deliverable not found: {path}"

    def test_deliverable_has_required_fields(self) -> None:
        path = _REPO_ROOT / "results/experiment_625_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written — run the experiment script first.")
        data = json.loads(path.read_text())
        missing = self._REQUIRED_FIELDS - set(data.keys())
        assert not missing, f"Missing fields in deliverable: {missing}"

    def test_deliverable_schema_is_correct(self) -> None:
        path = _REPO_ROOT / "results/experiment_625_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        assert data.get("schema") == "carnot.tier1_fr11_relay.v1"

    def test_deliverable_mode_is_valid(self) -> None:
        path = _REPO_ROOT / "results/experiment_625_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        assert data.get("mode") in {"real_violations", "synthetic_fallback"}

    def test_deliverable_fp_rate_delta_consistent(self) -> None:
        path = _REPO_ROOT / "results/experiment_625_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        before = data.get("fp_rate_before", 0.0)
        after = data.get("fp_rate_after", 0.0)
        delta = data.get("fp_rate_delta", 0.0)
        assert abs((after - before) - delta) < 1e-9

    def test_deliverable_honest_verdict_matches_mode(self) -> None:
        path = _REPO_ROOT / "results/experiment_625_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        mode = data.get("mode")
        verdict = data.get("honest_verdict")
        if mode == "real_violations":
            assert verdict == "real_violations_relay_complete"
        else:
            assert verdict == "synthetic_fallback_relay_complete"
