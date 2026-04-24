"""Tests for Exp 813: Constraint Addition Live.

Covers:
- Gate blocks when Exp 812 honest_verdict != "injection_works"  (REQ-LEARN-813-001)
- delta_overall computation correctness                          (REQ-LEARN-813-002)
- honest_verdict maps correctly to delta value                   (REQ-LEARN-813-002)

Spec: REQ-LEARN-813-001, REQ-LEARN-813-002
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_813_constraint_addition_live import (
    compute_delta_overall,
    map_honest_verdict,
    _load_exp812_gate,
)


# ---------------------------------------------------------------------------
# REQ-LEARN-813-001: Gate blocks when Exp 812 injection_works is missing
# ---------------------------------------------------------------------------


class TestExp812Gate:
    """REQ-LEARN-813-001: Exp 812 gate must block if honest_verdict != 'injection_works'."""

    def _make_tmpl(self) -> MagicMock:
        """Minimal ExperimentTemplate mock that returns a blocked artifact."""
        tmpl = MagicMock()
        tmpl.build_result.side_effect = lambda *args, **kwargs: {
            "status": "blocked",
            **kwargs,
        }
        return tmpl

    def test_gate_blocks_when_file_missing(self, tmp_path: Path) -> None:
        """Gate returns blocked artifact when Exp 812 result file does not exist.

        Spec: REQ-LEARN-813-001
        """
        missing = tmp_path / "no_such_file.json"
        tmpl = self._make_tmpl()
        with patch(
            "scripts.experiment_813_constraint_addition_live.EXP_812_PATH", missing
        ):
            result = _load_exp812_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "injection_not_wired"
        assert result["status"] == "blocked"

    def test_gate_blocks_when_verdict_is_injection_negative_delta(
        self, tmp_path: Path
    ) -> None:
        """Gate returns blocked artifact when honest_verdict is 'injection_negative_delta'.

        This is the real Exp 812 state — the injector lowers energy instead of raising it,
        so the full live pipeline is not yet ready.

        Spec: REQ-LEARN-813-001
        """
        exp812_file = tmp_path / "experiment_812.json"
        exp812_file.write_text(
            json.dumps({"honest_verdict": "injection_negative_delta"})
        )
        tmpl = self._make_tmpl()
        with patch(
            "scripts.experiment_813_constraint_addition_live.EXP_812_PATH", exp812_file
        ):
            result = _load_exp812_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "injection_not_wired"
        assert "injection_negative_delta" in result["blocked_reason"]

    def test_gate_blocks_when_verdict_is_injection_no_delta(
        self, tmp_path: Path
    ) -> None:
        """Gate blocks when honest_verdict is 'injection_no_delta'.

        Spec: REQ-LEARN-813-001
        """
        exp812_file = tmp_path / "experiment_812.json"
        exp812_file.write_text(json.dumps({"honest_verdict": "injection_no_delta"}))
        tmpl = self._make_tmpl()
        with patch(
            "scripts.experiment_813_constraint_addition_live.EXP_812_PATH", exp812_file
        ):
            result = _load_exp812_gate(tmpl)
        assert result is not None
        assert result["honest_verdict"] == "injection_not_wired"

    def test_gate_passes_when_verdict_is_injection_works(
        self, tmp_path: Path
    ) -> None:
        """Gate returns None (pass) when honest_verdict is 'injection_works'.

        Spec: REQ-LEARN-813-001
        """
        exp812_file = tmp_path / "experiment_812.json"
        exp812_file.write_text(json.dumps({"honest_verdict": "injection_works"}))
        tmpl = self._make_tmpl()
        with patch(
            "scripts.experiment_813_constraint_addition_live.EXP_812_PATH", exp812_file
        ):
            result = _load_exp812_gate(tmpl)
        assert result is None


# ---------------------------------------------------------------------------
# REQ-LEARN-813-002: delta_overall computation
# ---------------------------------------------------------------------------


class TestComputeDeltaOverall:
    """REQ-LEARN-813-002: delta_overall must be the arithmetic mean of per-session deltas."""

    def test_three_equal_deltas(self) -> None:
        """Mean of [2.0, 2.0, 2.0] is 2.0.

        Spec: REQ-LEARN-813-002
        """
        assert compute_delta_overall([2.0, 2.0, 2.0]) == pytest.approx(2.0)

    def test_three_distinct_deltas(self) -> None:
        """Mean of [0.05, 0.08, 0.12] is ~0.0833.

        This matches SCENARIO-LEARN-813-001 expected values.

        Spec: REQ-LEARN-813-002
        """
        result = compute_delta_overall([0.05, 0.08, 0.12])
        assert result == pytest.approx(0.08333333, rel=1e-5)

    def test_negative_deltas_give_negative_mean(self) -> None:
        """Negative per-session deltas yield a negative overall delta.

        Spec: REQ-LEARN-813-002
        """
        assert compute_delta_overall([-1.0, -2.0, -3.0]) == pytest.approx(-2.0)

    def test_empty_list_returns_zero(self) -> None:
        """Empty session list returns 0.0 (no data = no improvement).

        Spec: REQ-LEARN-813-002
        """
        assert compute_delta_overall([]) == 0.0

    def test_mixed_signs_correct_mean(self) -> None:
        """Mixed-sign deltas are averaged correctly.

        Spec: REQ-LEARN-813-002
        """
        assert compute_delta_overall([3.0, -1.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# REQ-LEARN-813-002: honest_verdict mapping
# ---------------------------------------------------------------------------


class TestMapHonestVerdict:
    """REQ-LEARN-813-002: honest_verdict must map correctly to delta and inference_mode."""

    def test_works_live_when_positive_delta_and_live_gpu(self) -> None:
        """Positive delta + live_gpu -> constraint_addition_works_live.

        Spec: REQ-LEARN-813-002
        """
        assert (
            map_honest_verdict(0.05, "live_gpu")
            == "constraint_addition_works_live"
        )

    def test_no_delta_when_zero_delta_and_live_gpu(self) -> None:
        """Zero delta + live_gpu -> constraint_addition_no_delta_live.

        Spec: REQ-LEARN-813-002
        """
        assert (
            map_honest_verdict(0.0, "live_gpu")
            == "constraint_addition_no_delta_live"
        )

    def test_no_delta_when_negative_delta_and_live_gpu(self) -> None:
        """Negative delta + live_gpu -> constraint_addition_no_delta_live.

        Spec: REQ-LEARN-813-002
        """
        assert (
            map_honest_verdict(-0.03, "live_gpu")
            == "constraint_addition_no_delta_live"
        )

    def test_injection_not_wired_when_gate_blocked(self) -> None:
        """gate_blocked=True -> injection_not_wired regardless of delta.

        Spec: REQ-LEARN-813-001
        """
        assert (
            map_honest_verdict(0.5, "live_gpu", gate_blocked=True)
            == "injection_not_wired"
        )

    def test_blocked_no_live_gpu_when_gpu_gate_blocked(self) -> None:
        """live_gpu_blocked=True -> blocked_no_live_gpu.

        Spec: REQ-LEARN-813-001
        """
        assert (
            map_honest_verdict(None, "blocked", live_gpu_blocked=True)
            == "blocked_no_live_gpu"
        )

    def test_gate_blocked_takes_priority_over_live_gpu_blocked(self) -> None:
        """gate_blocked takes priority over live_gpu_blocked.

        Spec: REQ-LEARN-813-001
        """
        assert (
            map_honest_verdict(None, "blocked", gate_blocked=True, live_gpu_blocked=True)
            == "injection_not_wired"
        )

    def test_positive_delta_non_live_mode_gives_no_delta(self) -> None:
        """Positive delta with inference_mode != live_gpu does NOT close the retro.

        Synthetic_cpu delta results do not count toward RETRO-CONSTRAINT-ZERO-DELTA.

        Spec: REQ-LEARN-813-001
        """
        assert (
            map_honest_verdict(0.5, "synthetic_cpu")
            == "constraint_addition_no_delta_live"
        )
