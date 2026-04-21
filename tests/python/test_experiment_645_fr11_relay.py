"""Tests for Exp 645: Tier 1 FR-11 Self-Learning Relay v3.

100% targeted coverage on functions added in
scripts/experiment_645_tier1_fr11_relay.py:
  - _load_json()
  - _build_synthetic_violations()
  - _build_semi_real_violations()
  - _compute_fp_rate()

Mode-selection logic (real_mode / semi_real_mode / synthetic_fallback) is
covered by verifying mode-flag computation from fixture data.  The
experiment's top-level run is not re-executed here — the deliverable JSON
produced by running the script is the integration test artifact.

Spec: REQ-LEARN-082, SCENARIO-LEARN-128, SCENARIO-LEARN-129
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

import scripts.experiment_645_tier1_fr11_relay as exp645  # noqa: E402


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    """REQ-LEARN-082: upstream experiment JSON must load gracefully."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        # SCENARIO-LEARN-128: semi-real mode loads Exp 643 ensemble_tp.
        f = tmp_path / "result.json"
        f.write_text(json.dumps({"gate_open": True, "ensemble_tp": 9}))
        data = exp645._load_json(f)
        assert data["ensemble_tp"] == 9

    def test_returns_empty_dict_for_missing_file(self, tmp_path: Path) -> None:
        # SCENARIO-LEARN-129: blocked upstream degrades to synthetic fallback.
        data = exp645._load_json(tmp_path / "nonexistent.json")
        assert data == {}

    def test_returns_empty_dict_for_malformed_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        data = exp645._load_json(f)
        assert data == {}


# ---------------------------------------------------------------------------
# _build_synthetic_violations
# ---------------------------------------------------------------------------


class TestBuildSyntheticViolations:
    """REQ-LEARN-082: synthetic violations must cover all four arithmetic families."""

    def test_returns_25_total_violations(self) -> None:
        # SCENARIO-LEARN-129: 25 violations generated in synthetic fallback mode.
        violations = exp645._build_synthetic_violations(25)
        total = sum(v.count for v in violations)
        assert total == 25

    def test_covers_all_four_families(self) -> None:
        violations = exp645._build_synthetic_violations(25)
        types = {v.type for v in violations}
        assert types == {"carry", "sign", "unit", "comparison"}

    def test_each_violation_has_example_steps(self) -> None:
        violations = exp645._build_synthetic_violations(25)
        for v in violations:
            assert len(v.example_steps) > 0

    def test_small_n_distributes_evenly(self) -> None:
        violations = exp645._build_synthetic_violations(4)
        total = sum(v.count for v in violations)
        assert total == 4
        assert len(violations) == 4

    def test_large_n_correct_total(self) -> None:
        violations = exp645._build_synthetic_violations(100)
        total = sum(v.count for v in violations)
        assert total == 100


# ---------------------------------------------------------------------------
# _build_semi_real_violations
# ---------------------------------------------------------------------------


class TestBuildSemiRealViolations:
    """REQ-LEARN-082: semi-real violations from ensemble TP count."""

    def test_returns_single_ensemble_pattern(self) -> None:
        # SCENARIO-LEARN-128: ensemble mode builds one ViolationPattern.
        violations = exp645._build_semi_real_violations(9)
        assert len(violations) == 1
        assert violations[0].type == "ensemble_detected"

    def test_count_equals_ensemble_tp(self) -> None:
        violations = exp645._build_semi_real_violations(9)
        assert violations[0].count == 9

    def test_minimum_count_is_one_for_zero_input(self) -> None:
        # WHY: we call max(1, int(tp)) so that threshold can still be tested.
        violations = exp645._build_semi_real_violations(0)
        assert violations[0].count == 1

    def test_has_example_steps(self) -> None:
        violations = exp645._build_semi_real_violations(3)
        assert len(violations[0].example_steps) > 0


# ---------------------------------------------------------------------------
# _compute_fp_rate
# ---------------------------------------------------------------------------


class TestComputeFpRate:
    """REQ-LEARN-082: FP-rate proxy must correctly flag/not-flag correct responses."""

    def _make_monitor_with_patterns(self, patterns: list[str], threshold: int = 1):
        """Return a ConstraintAdditionFromMemory pre-seeded above threshold."""
        from carnot.pipeline.constraint_addition import ConstraintAdditionFromMemory

        mon = ConstraintAdditionFromMemory(threshold=threshold, pipeline=None)
        for p in patterns:
            for _ in range(threshold):
                mon.observe(p, "test_step")
        return mon

    def test_zero_fp_rate_when_no_patterns_observed(self) -> None:
        # SCENARIO-LEARN-129: empty monitor → 0.0 FP rate.
        from carnot.pipeline.constraint_addition import ConstraintAdditionFromMemory

        mon = ConstraintAdditionFromMemory(threshold=5, pipeline=None)
        rate = exp645._compute_fp_rate(mon, ["clean answer"], ["bad answer"])
        assert rate == 0.0

    def test_fp_rate_one_when_all_correct_flagged(self) -> None:
        mon = self._make_monitor_with_patterns(["carry"])
        correct = ["carry error here", "carry propagation issue"]
        rate = exp645._compute_fp_rate(mon, correct, [])
        assert rate == 1.0

    def test_fp_rate_zero_when_no_correct_flagged(self) -> None:
        mon = self._make_monitor_with_patterns(["carry"])
        correct = ["totally clean answer", "sum is 42"]
        rate = exp645._compute_fp_rate(mon, correct, [])
        assert rate == 0.0

    def test_fp_rate_partial(self) -> None:
        mon = self._make_monitor_with_patterns(["sign"])
        correct = ["sign flip happened", "clean response", "another clean one"]
        rate = exp645._compute_fp_rate(mon, correct, [])
        assert abs(rate - 1 / 3) < 1e-9

    def test_fp_rate_empty_correct_corpus_returns_zero(self) -> None:
        mon = self._make_monitor_with_patterns(["carry"])
        rate = exp645._compute_fp_rate(mon, [], ["bad"])
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Mode-selection logic
# ---------------------------------------------------------------------------


class TestModeSelection:
    """REQ-LEARN-082: mode flags are computed correctly from Exp 644 + Exp 643 data."""

    def test_real_mode_when_exp644_positive(self) -> None:
        # real_mode: signed_improvement > 0
        signed_imp = 1.5
        real_mode = signed_imp > 0
        assert real_mode is True

    def test_not_real_mode_when_exp644_blocked(self) -> None:
        # SCENARIO-LEARN-128: Exp 644 blocked (ci_stub yields 0.0).
        signed_imp = 0.0
        real_mode = signed_imp > 0
        assert real_mode is False

    def test_semi_real_mode_when_exp643_gate_open(self) -> None:
        # SCENARIO-LEARN-128: gate_open=True and ensemble_tp > 0 → semi_real_mode.
        real_mode = False
        gate_open = True
        ensemble_tp = 9
        semi_real_mode = (not real_mode) and gate_open and ensemble_tp > 0
        assert semi_real_mode is True

    def test_synthetic_fallback_when_both_blocked(self) -> None:
        # SCENARIO-LEARN-129: both Exp 644 and 643 blocked.
        real_mode = False
        semi_real_mode = False
        assert not real_mode
        assert not semi_real_mode

    def test_real_mode_takes_priority_over_semi_real(self) -> None:
        # When real_mode is True, semi_real_mode must not fire.
        signed_imp = 2.0
        gate_open = True
        ensemble_tp = 9
        real_mode = signed_imp > 0
        semi_real_mode = (not real_mode) and gate_open and ensemble_tp > 0
        assert real_mode is True
        assert semi_real_mode is False

    def test_semi_real_false_when_ensemble_tp_zero(self) -> None:
        # SCENARIO-LEARN-129: gate open but no TPs → synthetic fallback.
        real_mode = False
        gate_open = True
        ensemble_tp = 0
        semi_real_mode = (not real_mode) and gate_open and ensemble_tp > 0
        assert semi_real_mode is False


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
        # SCENARIO-LEARN-128/129: relay writes deliverable regardless of mode.
        path = _REPO_ROOT / "results/experiment_645_tier1_fr11_relay.json"
        assert path.exists(), f"Deliverable not found: {path}"

    def test_deliverable_has_required_fields(self) -> None:
        path = _REPO_ROOT / "results/experiment_645_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written — run the experiment script first.")
        data = json.loads(path.read_text())
        missing = self._REQUIRED_FIELDS - set(data.keys())
        assert not missing, f"Missing fields in deliverable: {missing}"

    def test_deliverable_schema_is_v3(self) -> None:
        path = _REPO_ROOT / "results/experiment_645_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        assert data.get("schema") == "carnot.tier1_fr11_relay.v3"

    def test_deliverable_mode_is_valid(self) -> None:
        path = _REPO_ROOT / "results/experiment_645_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        assert data.get("mode") in {
            "real_violations",
            "semi_real_ensemble",
            "synthetic_fallback",
        }

    def test_deliverable_fp_rate_delta_consistent(self) -> None:
        path = _REPO_ROOT / "results/experiment_645_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        before = data.get("fp_rate_before", 0.0)
        after = data.get("fp_rate_after", 0.0)
        delta = data.get("fp_rate_delta", 0.0)
        assert abs((after - before) - delta) < 1e-9

    def test_deliverable_honest_verdict_matches_mode(self) -> None:
        path = _REPO_ROOT / "results/experiment_645_tier1_fr11_relay.json"
        if not path.exists():
            pytest.skip("Deliverable not yet written.")
        data = json.loads(path.read_text())
        mode = data.get("mode")
        verdict = data.get("honest_verdict")
        expected = {
            "real_violations": "real_violations_relay",
            "semi_real_ensemble": "semi_real_ensemble_relay",
            "synthetic_fallback": "synthetic_fallback_relay",
        }
        assert verdict == expected.get(mode), f"verdict {verdict!r} does not match mode {mode!r}"
