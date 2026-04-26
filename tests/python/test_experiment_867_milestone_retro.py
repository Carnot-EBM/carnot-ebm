"""Tests for Experiment 867 — Milestone 2026.04.66 operational retro.

Traces to:
  REQ-INFRA-080 — operational retrospective MUST be generated at each milestone
                  boundary using schema carnot.operational_retro.v41.
  SCENARIO-INFRA-090 — all 13 criteria for milestone 2026.04.66 evaluated and
                        recorded with evidence strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RETRO_PATH = _REPO_ROOT / "results" / "operational_retro_2026_04_66.json"


@pytest.fixture(scope="module")
def retro() -> dict:
    """Load the written retro artifact once for all tests."""
    assert _RETRO_PATH.exists(), (
        f"Retro artifact missing: {_RETRO_PATH}. Run scripts/experiment_867_milestone_retro.py."
    )
    with _RETRO_PATH.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# REQ-INFRA-080: schema and required top-level fields
# ---------------------------------------------------------------------------


class TestSchemaAndRequiredFields:
    """Validates that the retro artifact has the correct schema and required fields."""

    REQUIRED_FIELDS = {
        "retro_schema",
        "milestone",
        "milestone_title",
        "experiments_completed",
        "experiments_this_milestone",
        "wall_time_minutes",
        "wall_time_delta_vs_65",
        "success_criteria",
        "criteria_met_count",
        "criteria_met_total",
        "retros_closed",
        "retros_opened",
        "open_retros",
        "open_retros_count",
        "honest_verdict",
        "key_wins",
        "key_failures",
        "recommended_focus_next_milestone",
    }

    def test_schema_field(self, retro: dict) -> None:
        # REQ-INFRA-080: retro_schema version must be carnot.operational_retro.v41
        # Note: ExperimentTemplate auto-populates "schema" as a key list; we use "retro_schema"
        assert retro["retro_schema"] == "carnot.operational_retro.v41"

    def test_all_required_fields_present(self, retro: dict) -> None:
        missing = self.REQUIRED_FIELDS - set(retro.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_milestone_label(self, retro: dict) -> None:
        assert retro["milestone"] == "2026.04.66"

    def test_criteria_count_matches_list(self, retro: dict) -> None:
        # criteria_met_total must equal length of success_criteria list
        assert retro["criteria_met_total"] == len(retro["success_criteria"])

    def test_criteria_met_count_matches_list(self, retro: dict) -> None:
        actual_met = sum(1 for c in retro["success_criteria"] if c["met"])
        assert retro["criteria_met_count"] == actual_met

    def test_criteria_total_is_13(self, retro: dict) -> None:
        # Milestone .66 specifies exactly 13 success criteria
        assert retro["criteria_met_total"] == 13

    def test_open_retros_count_matches_list(self, retro: dict) -> None:
        assert retro["open_retros_count"] == len(retro["open_retros"])


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-090: per-criterion evaluation correctness
# ---------------------------------------------------------------------------


class TestCriteriaEvaluation:
    """Validates that each of the 13 criteria has the correct met/not-met outcome."""

    def _get_criterion(self, retro: dict, cid: int) -> dict:
        matches = [c for c in retro["success_criteria"] if c["id"] == cid]
        assert len(matches) == 1, f"Expected exactly 1 criterion with id={cid}"
        return matches[0]

    def test_criterion_1_live_env_fixed(self, retro: dict) -> None:
        # Exp 855: live_env_fixed=True, env_guard_deployed=True → MET
        c = self._get_criterion(retro, 1)
        assert c["met"] is True

    def test_criterion_2_dual_gpu_deployed(self, retro: dict) -> None:
        # Exp 856: dual_gpu_deployed=True, throughput_ratio=1.979 ≥ 1.5 → MET
        c = self._get_criterion(retro, 2)
        assert c["met"] is True

    def test_criterion_3_code_repair_blocked(self, retro: dict) -> None:
        # Exp 857: blocked (404), signed_improvement=null → NOT MET
        c = self._get_criterion(retro, 3)
        assert c["met"] is False
        assert "blocked" in c["evidence"].lower() or "null" in c["evidence"].lower()

    def test_criterion_4_benchmark_simulation_fallback(self, retro: dict) -> None:
        # Exp 858: pipeline_improvement=0.2133 but inference_mode=simulation_fallback → NOT MET
        c = self._get_criterion(retro, 4)
        assert c["met"] is False
        assert "simulation_fallback" in c["evidence"]

    def test_criterion_5_ice40_bitstream(self, retro: dict) -> None:
        # Exp 859: bitstream_generated=True, lut_count=134 < 500 → MET
        c = self._get_criterion(retro, 5)
        assert c["met"] is True

    def test_criterion_6_inertia_sweeps_below_5x(self, retro: dict) -> None:
        # Exp 860: discrimination_delta=71.5 (>0 ✓) but sweeps_reduction=2x (< 5x) → NOT MET
        c = self._get_criterion(retro, 6)
        assert c["met"] is False
        assert "2x" in c["evidence"] or "2.0" in c["evidence"]

    def test_criterion_7_streaming_cot_auc(self, retro: dict) -> None:
        # Exp 861: AUC_streaming=1.0 > 0.65 → MET
        c = self._get_criterion(retro, 7)
        assert c["met"] is True

    def test_criterion_8_lagrange_adaptive(self, retro: dict) -> None:
        # Exp 862: delta_s1_to_s5=0.05 > 0 → MET
        c = self._get_criterion(retro, 8)
        assert c["met"] is True

    def test_criterion_9_hallusae_below_threshold(self, retro: dict) -> None:
        # Exp 863: AUC_geometric=0.6144 < 0.65 → NOT MET
        c = self._get_criterion(retro, 9)
        assert c["met"] is False
        assert "0.61" in c["evidence"]

    def test_criterion_10_fr11_tier2_relay(self, retro: dict) -> None:
        # Exp 864: tier2_relay_confirmed=True → MET
        c = self._get_criterion(retro, 10)
        assert c["met"] is True

    def test_criterion_11_memory_compression(self, retro: dict) -> None:
        # Exp 865: retrieval_auroc_after=1.0 > 0.75 → MET
        c = self._get_criterion(retro, 11)
        assert c["met"] is True

    def test_criterion_12_kan_over_budget(self, retro: dict) -> None:
        # Exp 866: within_ice40_budget=False → NOT MET
        c = self._get_criterion(retro, 12)
        assert c["met"] is False
        assert "14400" in c["evidence"] or "over" in c["evidence"].lower()

    def test_criterion_13_wall_time_improvement(self, retro: dict) -> None:
        # Wall time .66 << .65 (0.86 min vs 78 min) → MET
        c = self._get_criterion(retro, 13)
        assert c["met"] is True
        assert retro["wall_time_delta_vs_65"] < 0

    def test_exactly_8_criteria_met(self, retro: dict) -> None:
        # Criteria 1,2,5,7,8,10,11,13 are met = 8 total
        assert retro["criteria_met_count"] == 8


# ---------------------------------------------------------------------------
# REQ-INFRA-080: honest verdict logic
# ---------------------------------------------------------------------------


class TestHonestVerdict:
    def test_verdict_is_milestone_partial(self, retro: dict) -> None:
        # 8/13 = 61.5% → between 46% and 75% → "milestone_partial"
        assert retro["honest_verdict"] == "milestone_partial"

    def test_verdict_fn_success_boundary(self) -> None:
        from scripts.experiment_867_milestone_retro import honest_verdict

        assert honest_verdict(10, 13) == "milestone_success"
        assert honest_verdict(13, 13) == "milestone_success"

    def test_verdict_fn_partial_boundary(self) -> None:
        from scripts.experiment_867_milestone_retro import honest_verdict

        assert honest_verdict(8, 13) == "milestone_partial"
        assert honest_verdict(6, 13) == "milestone_partial"

    def test_verdict_fn_blocked_boundary(self) -> None:
        from scripts.experiment_867_milestone_retro import honest_verdict

        assert honest_verdict(5, 13) == "milestone_blocked"
        assert honest_verdict(0, 13) == "milestone_blocked"


# ---------------------------------------------------------------------------
# REQ-INFRA-080: RETRO audit correctness
# ---------------------------------------------------------------------------


class TestRetroAudit:
    """Validates the open/closed/opened RETRO lists."""

    EXPECTED_CLOSED = {
        "RETRO-LIVE-ENV-NOT-PROPAGATED",
        "RETRO-ICE40-N16-UNEXPECTED-EXPANSION",
        "RETRO-ICE40-PNR-LUT-OVERFLOW",
        "RETRO-ISING-INJECTION-NO-DISCRIMINATION",
        "RETRO-CONSTRAINT-ZERO-DELTA",
    }

    EXPECTED_STILL_OPEN_FROM_65 = {
        "RETRO-MANIFEST-FULL-SCOPE",
        "RETRO-JEPA-OOD",
        "RETRO-XILINX-TOOLS-UNAVAILABLE",
        "RETRO-SVAMP-ZERO-AUC",
        "RETRO-SOTA-MODEL-DOWNLOAD",
    }

    EXPECTED_OPENED = {
        "RETRO-HALLUSAE-AUC-BELOW-THRESHOLD",
        "RETRO-INERTIA-SWEEPS-TARGET-MISSED",
    }

    def test_closed_retros(self, retro: dict) -> None:
        closed = set(retro["retros_closed"])
        assert self.EXPECTED_CLOSED == closed

    def test_opened_retros(self, retro: dict) -> None:
        opened = set(retro["retros_opened"])
        assert self.EXPECTED_OPENED == opened

    def test_still_open_retros_from_65(self, retro: dict) -> None:
        open_set = set(retro["open_retros"])
        for r in self.EXPECTED_STILL_OPEN_FROM_65:
            assert r in open_set, f"{r} should still be open"

    def test_closed_retros_not_in_open(self, retro: dict) -> None:
        open_set = set(retro["open_retros"])
        for r in retro["retros_closed"]:
            assert r not in open_set, f"Closed RETRO {r} must not appear in open_retros"

    def test_open_retros_count_is_7(self, retro: dict) -> None:
        # 5 still-open from .65 + 2 new = 7
        assert retro["open_retros_count"] == 7


# ---------------------------------------------------------------------------
# REQ-INFRA-080: wall-time delta
# ---------------------------------------------------------------------------


class TestWallTime:
    def test_wall_time_delta_negative(self, retro: dict) -> None:
        # Must be negative (improvement) — .66 ran fast vs .65's 78 min
        assert retro["wall_time_delta_vs_65"] < 0

    def test_wall_time_minutes_positive(self, retro: dict) -> None:
        assert retro["wall_time_minutes"] > 0

    def test_wall_time_66_much_less_than_65(self, retro: dict) -> None:
        # .66 used simulation/fast runs; should be well under 5 minutes
        assert retro["wall_time_minutes"] < 5.0


# ---------------------------------------------------------------------------
# REQ-INFRA-080: compute_wall_time unit test
# ---------------------------------------------------------------------------


class TestComputeWallTime:
    def test_sum_and_conversion(self) -> None:
        from scripts.experiment_867_milestone_retro import compute_wall_time

        arts = {1: {"duration_s": 30.0}, 2: {"duration_s": 90.0}}
        result = compute_wall_time(arts)
        assert abs(result - 2.0) < 0.001  # 120s = 2.0 min

    def test_missing_duration_defaults_to_zero(self) -> None:
        from scripts.experiment_867_milestone_retro import compute_wall_time

        arts = {1: {}, 2: {"duration_s": 60.0}}
        result = compute_wall_time(arts)
        assert abs(result - 1.0) < 0.001
