"""Tests for Experiment 903: Milestone 2026.04.69 Operational Retrospective.

Verifies that the retro script correctly loads artifacts, computes wall-time
statistics, evaluates criteria, determines retro closures, and writes a
valid deliverable JSON.

Spec: REQ-INFRA-073, SCENARIO-INFRA-073

Why these tests:
  The retro script is the governance record for the milestone. If it
  silently produces wrong criteria counts or fails to close resolved retros,
  the next planner starts from incorrect ground truth. These tests catch
  arithmetic errors, field-lookup bugs, and missing deliverable fields before
  the conductor archives the milestone.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import functions under test
from scripts.experiment_903_milestone_retro import (
    compute_wall_time,
    evaluate_criteria,
    evaluate_retro_closures,
    check_slowest5_governance,
    build_honest_verdict,
    assert_deliverable_written,
    PRIOR_EXPERIMENT_COUNT,
    MILESTONE_EXPERIMENTS,
)


# ---------------------------------------------------------------------------
# Minimal stub data sufficient to exercise each function independently.
# We do NOT load real result files — that would tie tests to on-disk state.
# ---------------------------------------------------------------------------


def _make_stub_data(overrides: dict | None = None) -> dict:
    """
    Build a minimal dict that matches what load_artifacts() would return.

    Each experiment entry has the fields required by the corresponding
    evaluate_criteria() check.  Missing optional fields default to None / 0.
    """
    base = {
        892: {
            "status": "success",
            "honest_verdict": "preflight_complete",
            "duration_s": 0.0,
            "enforcement_wired": False,
            "enforcement_note": "Cannot wire without modifying scripts/research_conductor.py.",
            "notes": "RETRO-MANIFEST-FULL-SCOPE cannot be wired; documented in ops/known-issues.md.",
        },
        893: {
            "status": None,
            "honest_verdict": "mismatch_confirmed_gate_open",
            "duration_s": 0.01,
            "labeling_mismatch_confirmed": True,
            "svamp_auc": 0.5,
        },
        894: {
            "status": "blocked",
            "honest_verdict": "streaming_blocked_no_gpu",
            "duration_s": 312.0,
            "signed_improvement": 0,
        },
        895: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_881_not_met",
            "duration_s": 34.0,
            "signed_improvement": None,
        },
        896: {
            "status": None,
            "honest_verdict": "svamp_retro_closed",
            "duration_s": 0.001,
            "svamp_auc": 1.0,
            "retro_closed": True,
        },
        897: {
            "status": "success",
            "honest_verdict": "forgetting_improves_precision",
            "duration_s": None,
            "constraint_precision_with_forget": 0.832117,
            "constraint_precision_no_forget": 0.290816,
            "precision_delta": 0.5413,
        },
        898: {
            "status": "success",
            "honest_verdict": "tier4_viable_seed",
            "duration_s": 3.199,
            "energy_loss_before": 3.093484,
            "energy_loss_after": 2.538113,
            "energy_loss_delta": -0.555371,
            "tier4_viable": True,
        },
        899: {
            "status": "success",
            "honest_verdict": "drift_probe_not_viable",
            "duration_s": 32.128,
            "probe_auc": 0.5,
        },
        900: {
            "status": "success",
            "honest_verdict": "draft_verifier_viable",
            "duration_s": 0.002,
            "signed_improvement": 3,
            "constraint_violations_baseline": 3,
            "constraint_violations_draft_conditioned": 0,
        },
        901: {
            "status": "success",
            "honest_verdict": "pimi_improved_below_5x",
            "duration_s": 452.0,
            "sweeps_reduction": 4.33,
            "retired": True,
        },
        902: {
            "status": "success",
            "honest_verdict": "published_no_ipfs_mirror",
            "duration_s": 2.192,
            "publish_confirmed": True,
            "ipfs_mirror_cid": None,
        },
    }
    if overrides:
        for eid, fields in overrides.items():
            base[eid].update(fields)
    return base


# ---------------------------------------------------------------------------
# REQ-INFRA-073: wall-time statistics are computed correctly
# ---------------------------------------------------------------------------


class TestComputeWallTime:
    """SCENARIO-INFRA-073-A: wall-time computation."""

    def test_total_wall_time_matches_expected(self):
        """Sum of duration_s / 60 should equal total_wall_time_minutes."""
        data = _make_stub_data()
        total_min, per_avg, slowest_5 = compute_wall_time(data)
        # Manual sum: 0+0.01+312+34+0.001+0+3.199+32.128+0.002+452+2.192 = 835.532 s
        expected_total = 835.532 / 60.0
        assert abs(total_min - expected_total) < 0.01

    def test_per_experiment_avg(self):
        """per_experiment_avg_minutes = total / n_experiments (within rounding)."""
        data = _make_stub_data()
        total_min, per_avg, _ = compute_wall_time(data)
        # Both values are rounded to 4 decimal places, so tolerance of 1e-4 is appropriate.
        assert abs(per_avg - total_min / len(data)) < 1e-4

    def test_slowest_5_ordered_descending(self):
        """slowest_5 is sorted by elapsed_seconds descending."""
        data = _make_stub_data()
        _, _, slowest_5 = compute_wall_time(data)
        durations = [s["elapsed_seconds"] for s in slowest_5]
        assert durations == sorted(durations, reverse=True)

    def test_slowest_5_top_two_experiments(self):
        """The two longest experiments (901 at 452s, 894 at 312s) are top-2."""
        data = _make_stub_data()
        _, _, slowest_5 = compute_wall_time(data)
        top_two = [s["experiment"] for s in slowest_5[:2]]
        assert top_two == [901, 894]

    def test_none_duration_treated_as_zero(self):
        """Exp 897 has duration_s=None; should be treated as 0.0."""
        data = _make_stub_data()
        _, _, slowest_5 = compute_wall_time(data)
        exp_ids = [s["experiment"] for s in slowest_5]
        assert 897 not in exp_ids  # 0 seconds cannot be in slowest-5


# ---------------------------------------------------------------------------
# REQ-INFRA-073: criteria evaluation is correct
# ---------------------------------------------------------------------------


class TestEvaluateCriteria:
    """SCENARIO-INFRA-073-B: criteria evaluation."""

    def test_n_criteria_met_baseline(self):
        """Baseline stub data should yield 7 criteria met."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert sum(criteria.values()) == 7

    def test_manifest_enforcement_via_notes(self):
        """ops/known-issues.md in notes field counts as escalated."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["manifest_enforcement_verified"] is True

    def test_manifest_enforcement_via_enforcement_note(self):
        """ops/known-issues.md in enforcement_note also counts as escalated."""
        data = _make_stub_data(
            {892: {"enforcement_note": "Documented in ops/known-issues.md", "notes": ""}}
        )
        criteria = evaluate_criteria(data)
        assert criteria["manifest_enforcement_verified"] is True

    def test_manifest_enforcement_false_when_not_escalated(self):
        """False when neither enforcement_wired nor escalated."""
        data = _make_stub_data(
            {
                892: {
                    "enforcement_wired": False,
                    "enforcement_note": "Cannot wire.",
                    "notes": "No escalation.",
                }
            }
        )
        criteria = evaluate_criteria(data)
        assert criteria["manifest_enforcement_verified"] is False

    def test_svamp_root_cause_true(self):
        """labeling_mismatch_confirmed=True → criterion met."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["svamp_root_cause_confirmed"] is True

    def test_svamp_root_cause_false(self):
        """labeling_mismatch_confirmed=False → criterion not met."""
        data = _make_stub_data({893: {"labeling_mismatch_confirmed": False}})
        criteria = evaluate_criteria(data)
        assert criteria["svamp_root_cause_confirmed"] is False

    def test_vjepa_streaming_positive_false(self):
        """signed_improvement=0 (blocked) → criterion not met."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["vjepa_streaming_positive"] is False

    def test_vjepa_streaming_positive_true(self):
        """signed_improvement=1 → criterion met."""
        data = _make_stub_data({894: {"signed_improvement": 1}})
        criteria = evaluate_criteria(data)
        assert criteria["vjepa_streaming_positive"] is True

    def test_svamp_auc_above_threshold(self):
        """Exp 896 auc=1.0 → above 0.60 threshold."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["svamp_auc_above_threshold"] is True

    def test_svamp_auc_below_threshold(self):
        """auc=0.5 → below threshold."""
        data = _make_stub_data({896: {"svamp_auc": 0.5}})
        criteria = evaluate_criteria(data)
        assert criteria["svamp_auc_above_threshold"] is False

    def test_lagrange_forgetting_improves(self):
        """with_forget > no_forget → criterion met."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["lagrange_forgetting_improves"] is True

    def test_lagrange_forgetting_not_improves(self):
        """with_forget <= no_forget → criterion not met."""
        data = _make_stub_data(
            {
                897: {
                    "constraint_precision_with_forget": 0.2,
                    "constraint_precision_no_forget": 0.3,
                }
            }
        )
        criteria = evaluate_criteria(data)
        assert criteria["lagrange_forgetting_improves"] is False

    def test_kan_tier4_viable(self):
        """energy_loss_after < energy_loss_before → tier4 viable."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["kan_tier4_viable"] is True

    def test_drift_probe_not_viable(self):
        """probe_auc=0.5 < 0.65 threshold → not viable."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["drift_probe_viable"] is False

    def test_draft_verifier_viable(self):
        """signed_improvement=3 > 0 → viable."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["draft_conditioned_verifier_viable"] is True

    def test_pimi_resolved_via_retired_flag(self):
        """retired=True → pimi_resolved even if sweeps_reduction < 5.0."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["pimi_resolved"] is True

    def test_pimi_resolved_via_sweeps(self):
        """sweeps_reduction >= 5.0 → pimi_resolved."""
        data = _make_stub_data({901: {"sweeps_reduction": 5.0, "retired": False}})
        criteria = evaluate_criteria(data)
        assert criteria["pimi_resolved"] is True

    def test_pimi_not_resolved(self):
        """sweeps_reduction < 5.0 AND retired=False → not resolved."""
        data = _make_stub_data({901: {"sweeps_reduction": 4.33, "retired": False}})
        criteria = evaluate_criteria(data)
        assert criteria["pimi_resolved"] is False

    def test_hf_publish_incomplete_missing_ipfs(self):
        """publish_confirmed=True but ipfs_mirror_cid=None → not complete."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        assert criteria["hf_publish_complete"] is False

    def test_hf_publish_complete_with_ipfs(self):
        """Both publish_confirmed=True and ipfs_mirror_cid set → complete."""
        data = _make_stub_data({902: {"publish_confirmed": True, "ipfs_mirror_cid": "Qm123"}})
        criteria = evaluate_criteria(data)
        assert criteria["hf_publish_complete"] is True


# ---------------------------------------------------------------------------
# REQ-INFRA-073: retro closure logic
# ---------------------------------------------------------------------------


class TestEvaluateRetroCLosures:
    """SCENARIO-INFRA-073-C: retro closure determination."""

    def test_three_retros_closed_baseline(self):
        """Baseline stub: MANIFEST, SVAMP, INERTIA retros closed."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        closed, open_retros = evaluate_retro_closures(data, criteria)
        assert "RETRO-SVAMP-ZERO-AUC" in closed
        assert "RETRO-INERTIA-SWEEPS-TARGET-MISSED" in closed
        assert "RETRO-MANIFEST-FULL-SCOPE" in closed

    def test_xilinx_always_open(self):
        """RETRO-XILINX-TOOLS-UNAVAILABLE is always open (no .69 action)."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        _, open_retros = evaluate_retro_closures(data, criteria)
        assert "RETRO-XILINX-TOOLS-UNAVAILABLE" in open_retros

    def test_svamp_retro_stays_open_if_auc_low(self):
        """RETRO-SVAMP-ZERO-AUC stays open when svamp_auc <= 0.60."""
        data = _make_stub_data({896: {"svamp_auc": 0.5}})
        criteria = evaluate_criteria(data)
        _, open_retros = evaluate_retro_closures(data, criteria)
        assert "RETRO-SVAMP-ZERO-AUC" in open_retros

    def test_inertia_retro_stays_open_if_not_retired(self):
        """RETRO-INERTIA-SWEEPS-TARGET-MISSED stays open when retired=False."""
        data = _make_stub_data({901: {"sweeps_reduction": 3.0, "retired": False}})
        criteria = evaluate_criteria(data)
        _, open_retros = evaluate_retro_closures(data, criteria)
        assert "RETRO-INERTIA-SWEEPS-TARGET-MISSED" in open_retros


# ---------------------------------------------------------------------------
# REQ-INFRA-073: slowest-5 governance check
# ---------------------------------------------------------------------------


class TestCheckSlowest5Governance:
    """SCENARIO-INFRA-073-D: governance violation detection."""

    def test_no_violations_clean_experiments(self):
        """New experiments 892-902 are not in the retired set → clean."""
        data = _make_stub_data()
        _, _, slowest_5 = compute_wall_time(data)
        gov = check_slowest5_governance(slowest_5)
        assert gov["clean"] is True
        assert gov["slowest5_governance_violation"] is False

    def test_violation_detected_for_retired_id(self):
        """A retired experiment ID in slowest-5 triggers a violation."""
        # Exp 527 is in the YAML retired set (exclusion_manifest.yaml, completed .57)
        fake_slowest_5 = [
            {
                "experiment": 527,
                "elapsed_minutes": 52.0,
                "elapsed_seconds": 3120.0,
                "status": "success",
                "honest_verdict": "live_100q_precision",
            },
        ]
        gov = check_slowest5_governance(fake_slowest_5)
        assert gov["slowest5_governance_violation"] is True
        assert 527 in gov["violations"]


# ---------------------------------------------------------------------------
# REQ-INFRA-073: honest_verdict content
# ---------------------------------------------------------------------------


class TestBuildHonestVerdict:
    """SCENARIO-INFRA-073-E: honest_verdict string construction."""

    def test_contains_criteria_count(self):
        """honest_verdict must include n_criteria_met/11."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        n = sum(criteria.values())
        closed, open_retros = evaluate_retro_closures(data, criteria)
        verdict = build_honest_verdict(criteria, n, closed, open_retros, data)
        assert f"{n}/11_criteria_met" in verdict

    def test_contains_retros_closed_count(self):
        """honest_verdict must include retros_closed= token."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        closed, open_retros = evaluate_retro_closures(data, criteria)
        verdict = build_honest_verdict(criteria, 7, closed, open_retros, data)
        assert "retros_closed=" in verdict

    def test_contains_pimi_retired_info(self):
        """PIMI retirement status must appear in honest_verdict."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        closed, open_retros = evaluate_retro_closures(data, criteria)
        verdict = build_honest_verdict(criteria, 7, closed, open_retros, data)
        assert "pimi_retired=" in verdict

    def test_contains_svamp_auc(self):
        """SVAMP AUC value must appear in honest_verdict."""
        data = _make_stub_data()
        criteria = evaluate_criteria(data)
        closed, open_retros = evaluate_retro_closures(data, criteria)
        verdict = build_honest_verdict(criteria, 7, closed, open_retros, data)
        assert "svamp_estimation_auc=" in verdict


# ---------------------------------------------------------------------------
# REQ-INFRA-073: assert_deliverable_written validation
# ---------------------------------------------------------------------------


class TestAssertDeliverableWritten:
    """SCENARIO-INFRA-073-F: deliverable validation."""

    def test_valid_artifact_passes(self):
        """A fully-formed artifact with all required fields passes the check."""
        artifact = {
            "schema": "carnot.operational_retro.v45",
            "milestone": "2026.04.69",
            "experiment_count": 830,
            "wall_time_minutes": 13.93,
            "per_experiment_avg_minutes": 1.27,
            "slowest_5": [],
            "n_criteria_met": 7,
            "criteria": {},
            "retros_closed_this_milestone": [],
            "open_retros": [],
            "governance": {},
            "honest_verdict": "7/11_criteria_met",
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(artifact, f)
            tmp_path = Path(f.name)
        assert_deliverable_written(tmp_path)  # must not raise

    def test_missing_required_field_raises(self):
        """An artifact missing a required field raises RuntimeError."""
        artifact = {
            "schema": "carnot.operational_retro.v45",
            "milestone": "2026.04.69",
            # missing experiment_count and others
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(artifact, f)
            tmp_path = Path(f.name)
        with pytest.raises(RuntimeError, match="missing required fields"):
            assert_deliverable_written(tmp_path)

    def test_nonexistent_file_raises(self):
        """A non-existent path raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Deliverable not written"):
            assert_deliverable_written(Path("/tmp/does_not_exist_903_retro.json"))


# ---------------------------------------------------------------------------
# REQ-INFRA-073: experiment count constants
# ---------------------------------------------------------------------------


class TestConstants:
    """SCENARIO-INFRA-073-G: constant values match documented state."""

    def test_prior_experiment_count(self):
        """PRIOR_EXPERIMENT_COUNT matches .68 retro documented value."""
        assert PRIOR_EXPERIMENT_COUNT == 818

    def test_milestone_experiments_count(self):
        """MILESTONE_EXPERIMENTS contains exactly 11 experiments (892-902)."""
        assert len(MILESTONE_EXPERIMENTS) == 11
        assert min(MILESTONE_EXPERIMENTS) == 892
        assert max(MILESTONE_EXPERIMENTS) == 902
