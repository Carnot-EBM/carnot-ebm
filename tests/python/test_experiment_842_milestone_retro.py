"""Tests for Experiment 842: Milestone 2026.04.64 Operational Retrospective.

Traces to:
    REQ-INFRA-064: The milestone retrospective MUST evaluate all 11 experiments (831-841)
        against pre-declared success criteria and produce a JSON artifact with
        schema=carnot.operational_retro.v39 before the next milestone begins.
    SCENARIO-INFRA-072: Exp 836 delta_overall=0.0; success criterion constraint_delta_positive
        must be marked met=False regardless of the write-path fix status.
    SCENARIO-INFRA-073: Exp 841 speedup=1.71; RETRO-SYMCODE-SERIAL must appear in
        retros_closed in the artifact.
    SCENARIO-INFRA-074: Experiments 837 and 838 were both blocked; tier1_relay_works_live
        and jepa_v24_tier35_deployed must both be met=False.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.experiment_842_milestone_retro import (
    DELIVERABLE,
    EXPERIMENT_CAP,
    IMPROVEMENTS,
    MILESTONE_EXPERIMENTS,
    MILESTONE_WALL_TIME_MINUTES,
    PRIOR_EXPERIMENTS_COMPLETED,
    PRIOR_TOTAL_WALL_TIME_MINUTES,
    RETROS_CLOSED,
    RETROS_OPENED,
    RETROS_STILL_OPEN,
    assert_deliverable_written,
    audit_retros,
    compute_honest_verdict,
    compute_metrics,
    eval_criteria,
    load_experiments,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic experiment artifacts mirroring actual results
# ---------------------------------------------------------------------------


@pytest.fixture()
def experiments_all_passing() -> dict:
    """Synthetic artifacts where every success criterion is met.

    Used to verify that eval_criteria correctly returns n_met=11 when all
    thresholds are satisfied.  The field values match the exact threshold checks
    in eval_criteria() so the test is not fragile to internal rounding.
    """
    return {
        831: {"honest_verdict": "governance_ready"},
        832: {"exp_824_auc_arc": 0.04, "status": "success"},
        833: {"hypothesis_confirmed": "H1", "status": "success"},
        834: {"min_domain_auc": 0.60, "auc_svamp": 0.60, "status": "success"},
        835: {"accuracy_standard": 0.80, "status": "success"},
        836: {"delta_overall": 0.15, "status": "success"},
        837: {"honest_verdict": "monotonic_confirmed", "status": "success"},
        838: {"tier35_deployed": True, "status": "success"},
        839: {"bitstream_generated": True, "status": "success"},
        840: {"honest_verdict": "live_gpu_benchmarked", "status": "success"},
        841: {"speedup": 2.0, "retro_symcode_serial_closed": True, "status": "success"},
    }


@pytest.fixture()
def experiments_actual() -> dict:
    """Synthetic artifacts matching actual .64 experiment result values.

    These values are taken directly from the experiment result JSON files
    (verified at retrospective write time) and used to confirm the retro
    correctly marks 4 criteria as met and 7 as not-met.
    """
    return {
        831: {"honest_verdict": "governance_ready"},
        832: {"exp_824_auc_arc": 0.04, "n_domains": 3, "status": "success"},
        833: {
            "hypothesis_confirmed": "H1",
            "root_cause": "write_path_missing",
            "status": "success",
        },
        834: {
            "min_domain_auc": 0.0,
            "auc_svamp": 0.0,
            "auc_arc": 0.71875,
            "overall_ood_auc": 0.4921875,
            "status": "success",
        },
        835: {
            "accuracy_standard": 0.0,
            "accuracy_adversarial": 0.5,
            "accuracy_overall": 0.25,
            "honest_verdict": "arbiter_still_wrong",
            "status": "success",
        },
        836: {
            "delta_overall": 0.0,
            "n_constraints_in_store_after_s3": 15,
            "honest_verdict": "write_path_fixed_no_delta",
            "status": "success",
        },
        837: {
            "honest_verdict": "blocked_gate",
            "status": "blocked",
            "blocked": True,
        },
        838: {
            "tier35_deployed": False,
            "honest_verdict": "jepa_v24_not_deployed_below_gate",
            "status": "blocked",
        },
        839: {
            "bitstream_generated": False,
            "honest_verdict": "pnr_failed",
            "status": "partial",
        },
        840: {
            "honest_verdict": "simulated_no_verdict",
            "status": "blocked",
        },
        841: {
            "speedup": 1.709641263771632,
            "retro_symcode_serial_closed": True,
            "honest_verdict": "batching_marginal",
            "status": "success",
        },
    }


# ---------------------------------------------------------------------------
# Tests for eval_criteria()
# ---------------------------------------------------------------------------


class TestEvalCriteria:
    """REQ-INFRA-064: eval_criteria must score each of the 11 criteria correctly."""

    def test_all_criteria_met_when_all_thresholds_satisfied(self, experiments_all_passing):
        """SCENARIO-INFRA-072: when all thresholds satisfied, n_met=11."""
        criteria, n_met = eval_criteria(experiments_all_passing)
        assert n_met == 11
        assert len(criteria) == 11
        assert all(c["met"] for c in criteria)

    def test_actual_results_yield_4_met(self, experiments_actual):
        """SCENARIO-INFRA-072: actual .64 results yield n_met=4."""
        criteria, n_met = eval_criteria(experiments_actual)
        # governance_ready, arc_diagnosis_found, constraint_root_cause_found, batching_effective
        assert n_met == 4

    def test_governance_ready_met_when_verdict_matches(self, experiments_actual):
        """Exp 831 honest_verdict='governance_ready' → governance_ready criterion met."""
        criteria, _ = eval_criteria(experiments_actual)
        gov = next(c for c in criteria if c["criterion"] == "governance_ready")
        assert gov["met"] is True
        assert gov["actual_value"] == "governance_ready"

    def test_arc_diagnosis_met_when_auc_low(self, experiments_actual):
        """SCENARIO-INFRA-072: exp_824_auc_arc=0.04 ≤ 0.10 → arc_diagnosis_found met."""
        criteria, _ = eval_criteria(experiments_actual)
        arc = next(c for c in criteria if c["criterion"] == "arc_diagnosis_found")
        assert arc["met"] is True
        assert arc["actual_value"] == 0.04

    def test_arc_diagnosis_not_met_when_auc_high(self, experiments_all_passing):
        """arc_diagnosis_found must be not-met when ARC AUC > 0.10 (no collapse detected)."""
        exps = dict(experiments_all_passing)
        exps[832] = {"exp_824_auc_arc": 0.75}
        criteria, _ = eval_criteria(exps)
        arc = next(c for c in criteria if c["criterion"] == "arc_diagnosis_found")
        assert arc["met"] is False

    def test_constraint_root_cause_met_on_h1(self, experiments_actual):
        """Exp 833 hypothesis_confirmed='H1' → constraint_root_cause_found met."""
        criteria, _ = eval_criteria(experiments_actual)
        root = next(c for c in criteria if c["criterion"] == "constraint_root_cause_found")
        assert root["met"] is True

    def test_constraint_root_cause_not_met_on_other_hypothesis(self, experiments_actual):
        """constraint_root_cause_found must be not-met when hypothesis != 'H1'."""
        exps = dict(experiments_actual)
        exps[833] = {"hypothesis_confirmed": "H2"}
        criteria, _ = eval_criteria(exps)
        root = next(c for c in criteria if c["criterion"] == "constraint_root_cause_found")
        assert root["met"] is False

    def test_jepa_domain_balanced_not_met_on_zero_min_auc(self, experiments_actual):
        """SCENARIO-INFRA-072: min_domain_auc=0.0 → jepa_v24_domain_balanced not met."""
        criteria, _ = eval_criteria(experiments_actual)
        jepa = next(c for c in criteria if c["criterion"] == "jepa_v24_domain_balanced")
        assert jepa["met"] is False
        assert jepa["actual_value"] == 0.0

    def test_arbiter_not_met_on_zero_accuracy(self, experiments_actual):
        """SCENARIO-INFRA-072: accuracy_standard=0.0 → arbiter_calibrated not met."""
        criteria, _ = eval_criteria(experiments_actual)
        arb = next(c for c in criteria if c["criterion"] == "arbiter_calibrated")
        assert arb["met"] is False
        assert arb["actual_value"] == 0.0

    def test_constraint_delta_not_met_when_zero(self, experiments_actual):
        """SCENARIO-INFRA-072: delta_overall=0.0 → constraint_delta_positive not met."""
        criteria, _ = eval_criteria(experiments_actual)
        delta = next(c for c in criteria if c["criterion"] == "constraint_delta_positive")
        assert delta["met"] is False
        assert delta["actual_value"] == 0.0

    def test_tier1_relay_not_met_when_blocked(self, experiments_actual):
        """SCENARIO-INFRA-074: Exp 837 blocked_gate → tier1_relay_works_live not met."""
        criteria, _ = eval_criteria(experiments_actual)
        relay = next(c for c in criteria if c["criterion"] == "tier1_relay_works_live")
        assert relay["met"] is False
        assert relay["actual_value"] == "blocked_gate"

    def test_jepa_tier35_not_met_when_not_deployed(self, experiments_actual):
        """SCENARIO-INFRA-074: Exp 838 tier35_deployed=False → jepa_v24_tier35_deployed not met."""
        criteria, _ = eval_criteria(experiments_actual)
        t35 = next(c for c in criteria if c["criterion"] == "jepa_v24_tier35_deployed")
        assert t35["met"] is False
        assert t35["actual_value"] is False

    def test_bitstream_not_met_when_false(self, experiments_actual):
        """Exp 839 bitstream_generated=False → bitstream_generated criterion not met."""
        criteria, _ = eval_criteria(experiments_actual)
        bs = next(c for c in criteria if c["criterion"] == "bitstream_generated")
        assert bs["met"] is False
        assert bs["actual_value"] is False

    def test_pipeline_not_met_when_simulated(self, experiments_actual):
        """Exp 840 honest_verdict='simulated_no_verdict' → pipeline_improvement not met."""
        criteria, _ = eval_criteria(experiments_actual)
        pip = next(c for c in criteria if c["criterion"] == "pipeline_improvement")
        assert pip["met"] is False
        assert pip["actual_value"] == "simulated_no_verdict"

    def test_batching_met_when_speedup_above_one(self, experiments_actual):
        """SCENARIO-INFRA-073: Exp 841 speedup=1.71 → batching_effective met."""
        criteria, _ = eval_criteria(experiments_actual)
        bat = next(c for c in criteria if c["criterion"] == "batching_effective")
        assert bat["met"] is True

    def test_batching_not_met_when_speedup_below_one(self):
        """batching_effective must be not-met when speedup < 1.0."""
        exps = {i: {} for i in range(831, 842)}
        exps[841] = {"speedup": 0.95, "retro_symcode_serial_closed": False}
        criteria, _ = eval_criteria(exps)
        bat = next(c for c in criteria if c["criterion"] == "batching_effective")
        assert bat["met"] is False

    def test_empty_experiments_yield_all_not_met(self):
        """When all experiments return empty dicts (missing files), all criteria are not-met."""
        exps = {i: {} for i in range(831, 842)}
        criteria, n_met = eval_criteria(exps)
        assert n_met == 0
        assert all(not c["met"] for c in criteria)


# ---------------------------------------------------------------------------
# Tests for compute_metrics()
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """REQ-INFRA-064: compute_metrics must correctly accumulate wall-time and experiment counts."""

    def test_total_wall_time_is_cumulative(self):
        """total_wall_time_minutes = PRIOR_TOTAL + MILESTONE."""
        metrics = compute_metrics()
        expected = PRIOR_TOTAL_WALL_TIME_MINUTES + MILESTONE_WALL_TIME_MINUTES
        assert metrics["total_wall_time_minutes"] == expected

    def test_experiments_completed_is_cumulative(self):
        """experiments_completed = PRIOR_EXPERIMENTS + MILESTONE_EXPERIMENTS."""
        metrics = compute_metrics()
        expected = PRIOR_EXPERIMENTS_COMPLETED + MILESTONE_EXPERIMENTS
        assert metrics["experiments_completed"] == expected

    def test_avg_time_is_total_over_experiments(self):
        """avg_time_per_experiment_minutes = total_wall_time / experiments_completed."""
        metrics = compute_metrics()
        total = PRIOR_TOTAL_WALL_TIME_MINUTES + MILESTONE_WALL_TIME_MINUTES
        count = PRIOR_EXPERIMENTS_COMPLETED + MILESTONE_EXPERIMENTS
        expected = round(total / count, 2)
        assert metrics["avg_time_per_experiment_minutes"] == expected

    def test_milestone_wall_time_field_present(self):
        """milestone_wall_time_minutes field must be present and equal MILESTONE_WALL_TIME_MINUTES."""
        metrics = compute_metrics()
        assert "milestone_wall_time_minutes" in metrics
        assert metrics["milestone_wall_time_minutes"] == MILESTONE_WALL_TIME_MINUTES

    def test_delta_vs_63_is_milestone_wall_time(self):
        """wall_time_delta_vs_63_minutes is the .64 milestone contribution (218 min)."""
        metrics = compute_metrics()
        assert metrics["wall_time_delta_vs_63_minutes"] == MILESTONE_WALL_TIME_MINUTES

    def test_regression_direction_when_milestone_longer_than_prior(self):
        """Wall time direction must be 'regression' when .64 took longer than .63 milestone."""
        # .64 milestone = 218 min; .63 contribution = 103 min → regression
        metrics = compute_metrics()
        assert metrics["wall_time_delta_vs_63_direction"] == "regression"

    def test_experiment_count_vs_cap_string_present(self):
        """experiment_count_vs_cap field must be a non-empty string."""
        metrics = compute_metrics()
        assert isinstance(metrics["experiment_count_vs_cap"], str)
        assert len(metrics["experiment_count_vs_cap"]) > 0


# ---------------------------------------------------------------------------
# Tests for audit_retros()
# ---------------------------------------------------------------------------


class TestAuditRetros:
    """REQ-INFRA-064: audit_retros must correctly classify each named RETRO."""

    def test_symcode_serial_closed(self, experiments_actual):
        """SCENARIO-INFRA-073: RETRO-SYMCODE-SERIAL must be status='closed' after .64."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-SYMCODE-SERIAL"]["status"] == "closed"

    def test_tier1_plateau_closed_per_governance(self, experiments_actual):
        """RETRO-TIER1-PLATEAU confirmed closed by governance in Exp 831; audit must reflect this."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-TIER1-PLATEAU"]["status"] == "closed_per_governance"

    def test_jepa_ood_open(self, experiments_actual):
        """RETRO-JEPA-OOD must be open (min_domain_auc=0.0)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-JEPA-OOD"]["status"] == "open"

    def test_arbiter_flat_energy_open(self, experiments_actual):
        """RETRO-ARBITER-FLAT-ENERGY must be open (accuracy_standard=0.0)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-ARBITER-FLAT-ENERGY"]["status"] == "open"

    def test_constraint_zero_delta_partially_mitigated(self, experiments_actual):
        """RETRO-CONSTRAINT-ZERO-DELTA must be partially_mitigated (write path fixed, delta=0)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-CONSTRAINT-ZERO-DELTA"]["status"] == "partially_mitigated"

    def test_manifest_full_scope_open(self, experiments_actual):
        """RETRO-MANIFEST-FULL-SCOPE must be open (requires conductor code change)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-MANIFEST-FULL-SCOPE"]["status"] == "open"

    def test_xilinx_tools_open(self, experiments_actual):
        """RETRO-XILINX-TOOLS-UNAVAILABLE must be open (Exp 839 pnr_failed)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-XILINX-TOOLS-UNAVAILABLE"]["status"] == "open"

    def test_svamp_zero_auc_opened_this_milestone(self, experiments_actual):
        """RETRO-SVAMP-ZERO-AUC must be open (new .64 RETRO)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-SVAMP-ZERO-AUC"]["status"] == "open"

    def test_ice40_pnr_overflow_opened_this_milestone(self, experiments_actual):
        """RETRO-ICE40-PNR-LUT-OVERFLOW must be open (new .64 RETRO)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-ICE40-PNR-LUT-OVERFLOW"]["status"] == "open"

    def test_all_retros_have_evidence_field(self, experiments_actual):
        """Every entry in the audit dict must have a non-empty 'evidence' field."""
        audit = audit_retros(experiments_actual)
        for name, entry in audit.items():
            assert "evidence" in entry, f"{name} missing 'evidence' field"
            assert len(entry["evidence"]) > 0, f"{name} has empty evidence"


# ---------------------------------------------------------------------------
# Tests for compute_honest_verdict()
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """REQ-INFRA-064: honest_verdict must follow the encoding schema."""

    def test_actual_verdict_format(self):
        """actual .64 verdict: regression_partial_close_4of11_5open."""
        verdict = compute_honest_verdict(4, 11, RETROS_STILL_OPEN)
        assert verdict == "regression_partial_close_4of11_5open"

    def test_verdict_contains_open_count(self):
        """Verdict must encode the number of open RETROs."""
        n_open = len(RETROS_STILL_OPEN)
        verdict = compute_honest_verdict(4, 11, RETROS_STILL_OPEN)
        assert f"{n_open}open" in verdict

    def test_verdict_contains_criteria_ratio(self):
        """Verdict must encode n_met and n_total in 'Nof M' format."""
        verdict = compute_honest_verdict(7, 11, RETROS_STILL_OPEN)
        assert "7of11" in verdict

    def test_verdict_starts_with_regression(self):
        """.64 took 218 min vs .63 contribution of 103 min → regression prefix."""
        verdict = compute_honest_verdict(4, 11, RETROS_STILL_OPEN)
        assert verdict.startswith("regression")


# ---------------------------------------------------------------------------
# Tests for RETRO accounting constants
# ---------------------------------------------------------------------------


class TestRetroConstants:
    """Schema contract: RETROS_CLOSED and RETROS_OPENED must list the correct items."""

    def test_symcode_serial_in_retros_closed(self):
        """SCENARIO-INFRA-073: RETRO-SYMCODE-SERIAL must be in RETROS_CLOSED."""
        assert "RETRO-SYMCODE-SERIAL" in RETROS_CLOSED

    def test_tier1_plateau_in_retros_closed(self):
        """RETRO-TIER1-PLATEAU must be in RETROS_CLOSED (governance confirmed)."""
        assert "RETRO-TIER1-PLATEAU" in RETROS_CLOSED

    def test_svamp_zero_auc_in_retros_opened(self):
        """RETRO-SVAMP-ZERO-AUC must be in RETROS_OPENED (new this milestone)."""
        assert "RETRO-SVAMP-ZERO-AUC" in RETROS_OPENED

    def test_ice40_pnr_in_retros_opened(self):
        """RETRO-ICE40-PNR-LUT-OVERFLOW must be in RETROS_OPENED (new this milestone)."""
        assert "RETRO-ICE40-PNR-LUT-OVERFLOW" in RETROS_OPENED

    def test_manifest_full_scope_in_retros_still_open(self):
        """RETRO-MANIFEST-FULL-SCOPE must remain in RETROS_STILL_OPEN."""
        assert "RETRO-MANIFEST-FULL-SCOPE" in RETROS_STILL_OPEN

    def test_retros_closed_and_still_open_disjoint(self):
        """No RETRO can appear in both RETROS_CLOSED and RETROS_STILL_OPEN."""
        overlap = set(RETROS_CLOSED) & set(RETROS_STILL_OPEN)
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_five_retros_still_open(self):
        """Exactly 5 RETROs must remain open after .64."""
        assert len(RETROS_STILL_OPEN) == 5


# ---------------------------------------------------------------------------
# Tests for IMPROVEMENTS constants
# ---------------------------------------------------------------------------


class TestImprovements:
    """REQ-INFRA-064: improvements_suggested must contain required priority levels."""

    def test_at_least_four_immediate_improvements(self):
        """At least 4 IMMEDIATE improvements must be listed (one per critical blocker)."""
        immediate = [i for i in IMPROVEMENTS if i["priority"] == "IMMEDIATE"]
        assert len(immediate) >= 4

    def test_all_improvements_have_action_and_rationale(self):
        """Every improvement must have 'action' and 'rationale' non-empty strings."""
        for imp in IMPROVEMENTS:
            assert "action" in imp and len(imp["action"]) > 0
            assert "rationale" in imp and len(imp["rationale"]) > 0

    def test_all_priorities_are_valid(self):
        """Priority must be one of IMMEDIATE, HIGH, MEDIUM, LOW."""
        valid = {"IMMEDIATE", "HIGH", "MEDIUM", "LOW"}
        for imp in IMPROVEMENTS:
            assert imp["priority"] in valid, f"Invalid priority: {imp['priority']}"

    def test_constraint_retrieval_fix_in_immediate(self):
        """IMMEDIATE improvements must include the constraint retrieval fix."""
        immediate_actions = " ".join(
            i["action"] for i in IMPROVEMENTS if i["priority"] == "IMMEDIATE"
        )
        assert (
            "ConstraintRetriever" in immediate_actions or "retrieval" in immediate_actions.lower()
        )

    def test_ice40_n16_fix_in_immediate(self):
        """IMMEDIATE improvements must include the N=16 iCE40 fix."""
        immediate_actions = " ".join(
            i["action"] for i in IMPROVEMENTS if i["priority"] == "IMMEDIATE"
        )
        assert "N=16" in immediate_actions or "iCE40" in immediate_actions


# ---------------------------------------------------------------------------
# Integration test: run main() and verify the deliverable
# ---------------------------------------------------------------------------


class TestMainIntegration:
    """Integration test: run the full retro script against real result files."""

    def test_deliverable_written_by_main(self, tmp_path, monkeypatch):
        """main() must write operational_retro_2026_04_64.json with correct schema."""
        # Redirect the deliverable to a temp path so we don't overwrite the real one
        # during testing.  We keep RESULTS_DIR pointing at the real results/ directory
        # so load_experiments() can read actual experiment result files.
        temp_deliverable = str(tmp_path / "operational_retro_2026_04_64.json")
        monkeypatch.setattr("scripts.experiment_842_milestone_retro.DELIVERABLE", temp_deliverable)

        from scripts.experiment_842_milestone_retro import main

        main()

        assert os.path.exists(temp_deliverable)
        with open(temp_deliverable) as fh:
            artifact = json.load(fh)

        assert artifact["schema"] == "carnot.operational_retro.v39"
        assert artifact["milestone"] == "2026.04.64"
        assert artifact["experiment"] == 842
        assert isinstance(artifact["n_criteria_met"], int)
        assert len(artifact["success_criteria"]) == 11
        assert artifact["n_criteria_total"] == 11
        assert "honest_verdict" in artifact
        assert "RETRO-SYMCODE-SERIAL" in artifact["retros_closed"]
        assert "RETRO-SVAMP-ZERO-AUC" in artifact["retros_opened"]
        assert len(artifact["retros_still_open"]) == 5

    def test_assert_deliverable_written_passes_on_valid_file(self, tmp_path, monkeypatch):
        """assert_deliverable_written must not raise when the file has all required fields."""
        temp_deliverable = str(tmp_path / "operational_retro_2026_04_64.json")
        monkeypatch.setattr("scripts.experiment_842_milestone_retro.DELIVERABLE", temp_deliverable)

        # Write a minimal valid artifact
        artifact = {
            "schema": "carnot.operational_retro.v39",
            "milestone": "2026.04.64",
            "experiment": 842,
            "honest_verdict": "regression_partial_close_4of11_5open",
            "n_criteria_met": 4,
            "n_criteria_total": 11,
            "success_criteria": [{}] * 11,
            "retros_closed": ["RETRO-SYMCODE-SERIAL"],
            "retros_opened": ["RETRO-SVAMP-ZERO-AUC"],
            "retros_still_open": ["A", "B", "C", "D", "E"],
            "improvements_suggested": [],
            "total_wall_time_minutes": 4122,
            "experiments_completed": 740,
            "avg_time_per_experiment_minutes": 5.57,
        }
        with open(temp_deliverable, "w") as fh:
            json.dump(artifact, fh)

        # Should not raise
        assert_deliverable_written()

    def test_assert_deliverable_written_raises_on_missing_file(self, tmp_path, monkeypatch):
        """assert_deliverable_written must raise AssertionError when file does not exist."""
        temp_deliverable = str(tmp_path / "missing.json")
        monkeypatch.setattr("scripts.experiment_842_milestone_retro.DELIVERABLE", temp_deliverable)

        with pytest.raises(AssertionError, match="Deliverable not written"):
            assert_deliverable_written()

    def test_assert_deliverable_written_raises_on_wrong_schema(self, tmp_path, monkeypatch):
        """assert_deliverable_written must raise when schema version is wrong."""
        temp_deliverable = str(tmp_path / "operational_retro_2026_04_64.json")
        monkeypatch.setattr("scripts.experiment_842_milestone_retro.DELIVERABLE", temp_deliverable)

        artifact = {
            "schema": "carnot.operational_retro.v38",  # wrong version
            "milestone": "2026.04.64",
            "experiment": 842,
            "honest_verdict": "test",
            "n_criteria_met": 4,
            "n_criteria_total": 11,
            "success_criteria": [{}] * 11,
            "retros_closed": [],
            "retros_opened": [],
            "retros_still_open": [],
            "improvements_suggested": [],
            "total_wall_time_minutes": 4122,
            "experiments_completed": 740,
            "avg_time_per_experiment_minutes": 5.57,
        }
        with open(temp_deliverable, "w") as fh:
            json.dump(artifact, fh)

        with pytest.raises(AssertionError, match="Schema version wrong"):
            assert_deliverable_written()

    def test_load_experiments_returns_dict_with_11_keys(self):
        """load_experiments must return a dict with keys 831-841."""
        exps = load_experiments()
        assert set(exps.keys()) == set(range(831, 842))

    def test_load_experiments_returns_dict_for_each_key(self):
        """load_experiments must return a dict (possibly empty) for each experiment ID."""
        exps = load_experiments()
        for eid, data in exps.items():
            assert isinstance(data, dict), f"Exp {eid} returned non-dict: {type(data)}"
