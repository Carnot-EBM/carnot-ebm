"""Tests for Experiment 854: Milestone 2026.04.65 Operational Retrospective.

Traces to:
    REQ-INFRA-064: The milestone retrospective MUST evaluate all 11 experiments (843-853)
        against pre-declared success criteria and produce a JSON artifact with
        schema=carnot.operational_retro.v40 before the next milestone begins.
    SCENARIO-INFRA-075: Exp 846 accuracy_standard=1.0; RETRO-ARBITER-FLAT-ENERGY must
        appear in retros_closed in the artifact.
    SCENARIO-INFRA-076: Exp 849 honest_verdict='gguf_cache_implemented'; RETRO-GGUF-CACHE-IMPORT
        must appear in retros_closed.
    SCENARIO-INFRA-077: Exp 851 bitstream_generated=False; bitstream_generated criterion must
        be met=False regardless of synthesis LUT count.
    SCENARIO-INFRA-078: Exp 844 auc_svamp=0.0 and Exp 845 tier35_deployed=False; both
        svamp_corpus_balanced (corpus was added) and jepa_v24b_tier35_deployed (gated out)
        must be scored correctly — balanced=True, tier35_deployed=False.
    SCENARIO-INFRA-079: Exp 853 honest_verdict='simulated_no_verdict'; pipeline_improvement
        must be met=False.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.experiment_854_milestone_retro import (
    DELIVERABLE,
    EXPERIMENT_CAP,
    IMPROVEMENTS,
    MILESTONE_EXPERIMENTS,
    MILESTONE_WALL_TIME_MINUTES,
    PRIOR_EXPERIMENTS_COMPLETED,
    PRIOR_MILESTONE_WALL_TIME_MINUTES,
    PRIOR_TOTAL_WALL_TIME_MINUTES,
    RETROS_CLOSED,
    RETROS_OPENED,
    RETROS_STILL_OPEN,
    assert_deliverable_written,
    audit_retros,
    compute_honest_verdict,
    compute_metrics,
    compute_slowest_5,
    eval_criteria,
    load_experiments,
    write_milestone_prereqs_section,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def experiments_all_passing() -> dict:
    """Synthetic artifacts where every success criterion is met (n_met=12 expected)."""
    return {
        843: {"honest_verdict": "governance_ready"},
        844: {
            "all_domains_coverage": True,
            "corpus_composition": {
                "svamp": {"correct": 20, "incorrect": 20}
            },
            "auc_svamp": 0.60,
            "min_domain_auc": 0.60,
        },
        845: {"tier35_deployed": True},
        846: {"accuracy_standard": 1.0},
        847: {"retrieval_auroc": 0.85},
        848: {"honest_verdict": "tier1_relay_works_live"},
        849: {"honest_verdict": "gguf_cache_implemented"},
        850: {"n_baseline_pass": 3, "n_repair_pass": 8, "signed_improvement": True},
        851: {"bitstream_generated": True},
        852: {"honest_verdict": "probe_viable"},
        853: {"honest_verdict": "live_gpu_benchmarked"},
    }


@pytest.fixture()
def experiments_actual() -> dict:
    """Synthetic artifacts matching actual .65 experiment result values.

    Values taken directly from the experiment result JSON files written during .65.
    Used to confirm the retro scores 7 criteria as met and 5 as not-met.
    """
    return {
        843: {"honest_verdict": "governance_ready", "status": "success"},
        844: {
            "all_domains_coverage": True,
            "corpus_composition": {
                "svamp": {"correct": 20, "incorrect": 20}
            },
            "auc_svamp": 0.0,
            "min_domain_auc": 0.0,
            "status": "success",
        },
        845: {"tier35_deployed": False, "status": "blocked",
              "blocked_reason": "min_domain_auc=0.0 < 0.50 deployment gate"},
        846: {"accuracy_standard": 1.0, "accuracy_adversarial": 1.0,
              "honest_verdict": "arbiter_calibrated", "status": "success"},
        847: {"retrieval_auroc": 0.72, "honest_verdict": "retrieval_partial",
              "status": "success"},
        848: {"honest_verdict": "tier1_relay_works_live", "is_monotonic": True,
              "status": "success"},
        849: {"honest_verdict": "gguf_cache_implemented", "status": "success"},
        850: {"n_baseline_pass": 0, "n_repair_pass": 0, "signed_improvement": None,
              "status": "blocked", "honest_verdict": "model_not_cached"},
        851: {"bitstream_generated": False, "status": "partial",
              "honest_verdict": "pnr_failed_n16"},
        852: {"honest_verdict": "probe_viable", "auc_synthetic": 1.0, "status": "success"},
        853: {"honest_verdict": "simulated_no_verdict", "status": "blocked"},
    }


@pytest.fixture()
def experiments_all_failing() -> dict:
    """Synthetic artifacts where every criterion fails (n_met=0 expected)."""
    return {
        843: {"honest_verdict": "blocked"},
        844: {
            "all_domains_coverage": False,
            "corpus_composition": {"svamp": {"correct": 0, "incorrect": 0}},
            "auc_svamp": 0.0,
        },
        845: {"tier35_deployed": False},
        846: {"accuracy_standard": 0.0},
        847: {"retrieval_auroc": 0.5},
        848: {"honest_verdict": "blocked_gate"},
        849: {"honest_verdict": "failed"},
        850: {"n_baseline_pass": 0, "n_repair_pass": 0, "signed_improvement": False},
        851: {"bitstream_generated": False},
        852: {"honest_verdict": "probe_not_viable"},
        853: {"honest_verdict": "simulated_no_verdict"},
    }


# ---------------------------------------------------------------------------
# Tests: eval_criteria
# ---------------------------------------------------------------------------

class TestEvalCriteria:
    def test_all_passing_yields_12_met(self, experiments_all_passing):
        """REQ-INFRA-064: When all experiment fields meet thresholds, n_met must equal 12."""
        criteria, n_met = eval_criteria(experiments_all_passing)
        assert n_met == 12
        assert len(criteria) == 12
        assert all(c["met"] for c in criteria)

    def test_all_failing_yields_0_met(self, experiments_all_failing):
        """REQ-INFRA-064: When all fields fail thresholds, n_met must equal 0."""
        criteria, n_met = eval_criteria(experiments_all_failing)
        assert n_met == 0
        assert not any(c["met"] for c in criteria)

    def test_actual_results_yield_7_met(self, experiments_actual):
        """REQ-INFRA-064: Actual .65 data must produce exactly 7 met criteria."""
        criteria, n_met = eval_criteria(experiments_actual)
        assert n_met == 7, f"Expected 7 met, got {n_met}"

    def test_governance_ready_criterion(self, experiments_actual):
        """SCENARIO-INFRA-075: governance_ready is met when Exp 843 honest_verdict is correct."""
        criteria, _ = eval_criteria(experiments_actual)
        gov = next(c for c in criteria if c["criterion"] == "governance_ready")
        assert gov["met"] is True
        assert gov["experiment"] == 843

    def test_svamp_corpus_balanced_met_despite_zero_auc(self, experiments_actual):
        """SCENARIO-INFRA-078: corpus balance and JEPA AUC are independent — balance met, AUC failed."""
        criteria, _ = eval_criteria(experiments_actual)
        bal = next(c for c in criteria if c["criterion"] == "svamp_corpus_balanced")
        auc = next(c for c in criteria if c["criterion"] == "jepa_v24b_all_domains")
        assert bal["met"] is True, "corpus is balanced (20+20 SVAMP pairs)"
        assert auc["met"] is True, "all_domains_coverage=True"

    def test_jepa_tier35_not_deployed(self, experiments_actual):
        """SCENARIO-INFRA-078: tier35_deployed=False means criterion not met."""
        criteria, _ = eval_criteria(experiments_actual)
        dep = next(c for c in criteria if c["criterion"] == "jepa_v24b_tier35_deployed")
        assert dep["met"] is False

    def test_arbiter_calibrated_met(self, experiments_actual):
        """SCENARIO-INFRA-075: accuracy_standard=1.0 must mark arbiter_calibrated as met."""
        criteria, _ = eval_criteria(experiments_actual)
        arb = next(c for c in criteria if c["criterion"] == "arbiter_calibrated")
        assert arb["met"] is True
        assert arb["actual_value"] == 1.0

    def test_retrieval_not_met_below_gate(self, experiments_actual):
        """AUROC=0.72 is below the 0.80 gate; retrieval_fixed must be met=False."""
        criteria, _ = eval_criteria(experiments_actual)
        ret = next(c for c in criteria if c["criterion"] == "retrieval_fixed")
        assert ret["met"] is False
        assert ret["actual_value"] == 0.72

    def test_tier1_relay_works_live(self, experiments_actual):
        """Exp 848 honest_verdict='tier1_relay_works_live' → criterion met."""
        criteria, _ = eval_criteria(experiments_actual)
        relay = next(c for c in criteria if c["criterion"] == "tier1_relay_works_live")
        assert relay["met"] is True

    def test_gguf_cache_implemented(self, experiments_actual):
        """SCENARIO-INFRA-076: Exp 849 honest_verdict='gguf_cache_implemented' → criterion met."""
        criteria, _ = eval_criteria(experiments_actual)
        cache = next(c for c in criteria if c["criterion"] == "gguf_cache_implemented")
        assert cache["met"] is True

    def test_code_repair_blocked_not_met(self, experiments_actual):
        """signed_improvement=None (blocked) scores as met=False."""
        criteria, _ = eval_criteria(experiments_actual)
        repair = next(c for c in criteria if c["criterion"] == "code_repair_positive")
        assert repair["met"] is False
        assert repair["actual_value"]["signed_improvement"] is None

    def test_bitstream_not_generated(self, experiments_actual):
        """SCENARIO-INFRA-077: bitstream_generated=False must score as not-met."""
        criteria, _ = eval_criteria(experiments_actual)
        bits = next(c for c in criteria if c["criterion"] == "bitstream_generated")
        assert bits["met"] is False
        assert bits["actual_value"] is False

    def test_semantic_probe_viable(self, experiments_actual):
        """honest_verdict='probe_viable' → semantic_probe_viable criterion met."""
        criteria, _ = eval_criteria(experiments_actual)
        probe = next(c for c in criteria if c["criterion"] == "semantic_probe_viable")
        assert probe["met"] is True

    def test_pipeline_improvement_not_met(self, experiments_actual):
        """SCENARIO-INFRA-079: simulated_no_verdict must score pipeline_improvement as not-met."""
        criteria, _ = eval_criteria(experiments_actual)
        pip = next(c for c in criteria if c["criterion"] == "pipeline_improvement")
        assert pip["met"] is False
        assert pip["actual_value"] == "simulated_no_verdict"

    def test_criteria_have_required_fields(self, experiments_actual):
        """Every criterion dict must have criterion, experiment, target, met, actual_value."""
        criteria, _ = eval_criteria(experiments_actual)
        required = {"criterion", "experiment", "target", "met", "actual_value"}
        for c in criteria:
            assert required.issubset(c.keys()), f"Missing fields in {c['criterion']}: {required - c.keys()}"

    def test_empty_experiments_yields_0_met(self):
        """Missing experiment files (empty dicts) score all criteria as not-met."""
        empty = {eid: {} for eid in range(843, 854)}
        _, n_met = eval_criteria(empty)
        assert n_met == 0


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_total_wall_time(self):
        """Total wall time must equal prior total plus milestone contribution."""
        metrics = compute_metrics()
        expected = PRIOR_TOTAL_WALL_TIME_MINUTES + MILESTONE_WALL_TIME_MINUTES
        assert metrics["total_wall_time_minutes"] == expected

    def test_total_experiments(self):
        """Total experiments must equal prior count plus milestone experiments."""
        metrics = compute_metrics()
        expected = PRIOR_EXPERIMENTS_COMPLETED + MILESTONE_EXPERIMENTS
        assert metrics["experiments_completed"] == expected

    def test_avg_time_is_positive(self):
        """Average time per experiment must be a positive float."""
        metrics = compute_metrics()
        assert metrics["avg_time_per_experiment_minutes"] > 0

    def test_wall_time_delta_vs_64_is_negative(self):
        """Milestone .65 ran shorter than .64; delta must be negative (improvement)."""
        metrics = compute_metrics()
        assert metrics["wall_time_delta_vs_64_minutes"] < 0

    def test_wall_time_delta_direction_is_improvement(self):
        """Negative delta must produce direction='improvement'."""
        metrics = compute_metrics()
        assert metrics["wall_time_delta_vs_64_direction"] == "improvement"

    def test_experiment_count_vs_cap_string(self):
        """experiment_count_vs_cap must be a string and mention the cap."""
        metrics = compute_metrics()
        assert isinstance(metrics["experiment_count_vs_cap"], str)
        assert str(EXPERIMENT_CAP) in metrics["experiment_count_vs_cap"]

    def test_milestone_experiments_count(self):
        """milestone_experiments field must equal MILESTONE_EXPERIMENTS constant."""
        metrics = compute_metrics()
        assert metrics["milestone_experiments"] == MILESTONE_EXPERIMENTS


# ---------------------------------------------------------------------------
# Tests: audit_retros
# ---------------------------------------------------------------------------

class TestAuditRetros:
    def test_arbiter_closed(self, experiments_actual):
        """SCENARIO-INFRA-075: RETRO-ARBITER-FLAT-ENERGY must be closed when accuracy=1.0."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-ARBITER-FLAT-ENERGY"]["status"] == "closed"

    def test_gguf_cache_closed(self, experiments_actual):
        """SCENARIO-INFRA-076: RETRO-GGUF-CACHE-IMPORT must be closed when implemented."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-GGUF-CACHE-IMPORT"]["status"] == "closed"

    def test_svamp_open(self, experiments_actual):
        """RETRO-SVAMP-ZERO-AUC must remain open when auc_svamp=0.0."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-SVAMP-ZERO-AUC"]["status"] == "open"

    def test_jepa_ood_open(self, experiments_actual):
        """RETRO-JEPA-OOD must remain open when min_domain_auc=0.0."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-JEPA-OOD"]["status"] == "open"

    def test_constraint_delta_partial(self, experiments_actual):
        """RETRO-CONSTRAINT-ZERO-DELTA must be partially_mitigated (relay works, AUROC 0.72 < 0.80)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-CONSTRAINT-ZERO-DELTA"]["status"] == "partially_mitigated"

    def test_ice40_open(self, experiments_actual):
        """RETRO-ICE40-PNR-LUT-OVERFLOW must remain open when bitstream_generated=False."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-ICE40-PNR-LUT-OVERFLOW"]["status"] == "open"

    def test_manifest_open(self, experiments_actual):
        """RETRO-MANIFEST-FULL-SCOPE must remain open (requires human action)."""
        audit = audit_retros(experiments_actual)
        assert audit["RETRO-MANIFEST-FULL-SCOPE"]["status"] == "open"

    def test_new_retros_present(self, experiments_actual):
        """Three new RETROs opened in .65 must appear in the audit dict."""
        audit = audit_retros(experiments_actual)
        assert "RETRO-SOTA-MODEL-DOWNLOAD" in audit
        assert "RETRO-ICE40-N16-UNEXPECTED-EXPANSION" in audit
        assert "RETRO-LIVE-ENV-NOT-PROPAGATED" in audit

    def test_all_retro_entries_have_status_and_evidence(self, experiments_actual):
        """Every RETRO audit entry must have 'status' and 'evidence' fields."""
        audit = audit_retros(experiments_actual)
        for retro_name, entry in audit.items():
            assert "status" in entry, f"Missing 'status' in {retro_name}"
            assert "evidence" in entry, f"Missing 'evidence' in {retro_name}"

    def test_retros_closed_constant_contains_arbiter_and_gguf(self):
        """RETROS_CLOSED constant must list both closed RETROs."""
        assert "RETRO-ARBITER-FLAT-ENERGY" in RETROS_CLOSED
        assert "RETRO-GGUF-CACHE-IMPORT" in RETROS_CLOSED
        assert len(RETROS_CLOSED) == 2

    def test_retros_opened_contains_three_new(self):
        """RETROS_OPENED constant must list all three new .65 RETROs."""
        assert "RETRO-SOTA-MODEL-DOWNLOAD" in RETROS_OPENED
        assert "RETRO-ICE40-N16-UNEXPECTED-EXPANSION" in RETROS_OPENED
        assert "RETRO-LIVE-ENV-NOT-PROPAGATED" in RETROS_OPENED
        assert len(RETROS_OPENED) == 3


# ---------------------------------------------------------------------------
# Tests: compute_slowest_5
# ---------------------------------------------------------------------------

class TestComputeSlowest5:
    def test_returns_5_entries(self):
        """Slowest-5 analysis must return exactly 5 experiment entries."""
        slowest = compute_slowest_5()
        assert len(slowest) == 5

    def test_sorted_descending(self):
        """Experiments must be sorted by elapsed_minutes descending."""
        slowest = compute_slowest_5()
        elapsed = [e["elapsed_minutes"] for e in slowest]
        assert elapsed == sorted(elapsed, reverse=True)

    def test_slowest_is_exp_845(self):
        """Exp 845 (JEPA deployment gate, 26 min) must be the slowest .65 experiment."""
        slowest = compute_slowest_5()
        assert slowest[0]["experiment"] == 845
        assert slowest[0]["elapsed_minutes"] == 26

    def test_all_entries_have_required_fields(self):
        """Each slowest entry must have experiment, elapsed_minutes, status, note fields."""
        slowest = compute_slowest_5()
        required = {"experiment", "elapsed_minutes", "status", "note"}
        for entry in slowest:
            assert required.issubset(entry.keys()), f"Missing fields in {entry}"

    def test_no_legacy_retired_experiments(self):
        """Retired experiments 786, 527, 627 must not appear in .65 slowest-5."""
        slowest = compute_slowest_5()
        exp_ids = {e["experiment"] for e in slowest}
        assert 786 not in exp_ids
        assert 527 not in exp_ids
        assert 627 not in exp_ids


# ---------------------------------------------------------------------------
# Tests: compute_honest_verdict
# ---------------------------------------------------------------------------

class TestComputeHonestVerdict:
    def test_contains_improvement(self):
        """Verdict must contain 'improvement' when milestone ran faster than .64."""
        v = compute_honest_verdict(7, 12, RETROS_STILL_OPEN)
        assert "improvement" in v.lower()

    def test_contains_criteria_count(self):
        """Verdict must encode the n_met/n_total criteria score."""
        v = compute_honest_verdict(7, 12, RETROS_STILL_OPEN)
        assert "7of12" in v

    def test_contains_retro_closed_references(self):
        """Verdict must reference closed RETROs to document the milestone progress."""
        v = compute_honest_verdict(7, 12, RETROS_STILL_OPEN)
        assert "ARBITER" in v
        assert "GGUF" in v

    def test_verdict_is_string(self):
        """Verdict must be a non-empty string."""
        v = compute_honest_verdict(7, 12, RETROS_STILL_OPEN)
        assert isinstance(v, str)
        assert len(v) > 0


# ---------------------------------------------------------------------------
# Tests: write_milestone_prereqs_section
# ---------------------------------------------------------------------------

class TestWriteMilestonePrereqsSection:
    def test_appends_66_section(self, tmp_path):
        """A .66 prerequisites section must be appended to MILESTONE_PREREQS.md."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("# Existing content\n")
        with patch("scripts.experiment_854_milestone_retro.MILESTONE_PREREQS",
                   str(prereqs_file)):
            write_milestone_prereqs_section()
        content = prereqs_file.read_text()
        assert "Milestone 2026.04.66 Prerequisites" in content
        assert "# Existing content" in content  # existing content preserved

    def test_idempotent_no_duplicate(self, tmp_path):
        """Writing the section twice must not duplicate it."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        prereqs_file.write_text("# Existing content\n")
        with patch("scripts.experiment_854_milestone_retro.MILESTONE_PREREQS",
                   str(prereqs_file)):
            write_milestone_prereqs_section()
            write_milestone_prereqs_section()
        content = prereqs_file.read_text()
        assert content.count("Milestone 2026.04.66 Prerequisites") == 1

    def test_creates_file_if_missing(self, tmp_path):
        """If MILESTONE_PREREQS.md does not exist, the function must create it."""
        prereqs_file = tmp_path / "MILESTONE_PREREQS.md"
        with patch("scripts.experiment_854_milestone_retro.MILESTONE_PREREQS",
                   str(prereqs_file)):
            write_milestone_prereqs_section()
        assert prereqs_file.exists()
        assert "2026.04.66" in prereqs_file.read_text()


# ---------------------------------------------------------------------------
# Tests: main + assert_deliverable_written
# ---------------------------------------------------------------------------

class TestMainAndDeliverable:
    def test_main_writes_deliverable(self, tmp_path):
        """main() must write results/operational_retro_2026_04_65.json with all required fields."""
        deliverable = tmp_path / "operational_retro_2026_04_65.json"
        prereqs = tmp_path / "MILESTONE_PREREQS.md"
        prereqs.write_text("# existing\n")
        with patch("scripts.experiment_854_milestone_retro.DELIVERABLE",
                   str(deliverable)), \
             patch("scripts.experiment_854_milestone_retro.MILESTONE_PREREQS",
                   str(prereqs)):
            from scripts.experiment_854_milestone_retro import main
            main()
        assert deliverable.exists()
        artifact = json.loads(deliverable.read_text())
        assert artifact["schema"] == "carnot.operational_retro.v40"
        assert artifact["milestone"] == "2026.04.65"
        assert artifact["experiment"] == 854
        assert len(artifact["success_criteria"]) == 12
        assert artifact["n_criteria_met"] == 7

    def test_assert_deliverable_written_passes_on_valid_file(self, tmp_path):
        """assert_deliverable_written must pass when deliverable has all required fields."""
        deliverable = tmp_path / "operational_retro_2026_04_65.json"
        artifact = {
            "schema": "carnot.operational_retro.v40",
            "milestone": "2026.04.65",
            "experiment": 854,
            "honest_verdict": "test_verdict",
            "n_criteria_met": 7,
            "n_criteria_total": 12,
            "success_criteria": [{}] * 12,
            "retros_closed": [],
            "retros_opened": [],
            "retros_still_open": [],
            "improvements_suggested": [],
            "total_wall_time_minutes": 4174,
            "experiments_completed": 762,
            "avg_time_per_experiment_minutes": 5.48,
            "slowest_5_experiments": [{}] * 5,
        }
        deliverable.write_text(json.dumps(artifact))
        with patch("scripts.experiment_854_milestone_retro.DELIVERABLE",
                   str(deliverable)):
            assert_deliverable_written()  # should not raise

    def test_assert_deliverable_written_fails_on_missing_file(self, tmp_path):
        """assert_deliverable_written must raise AssertionError when file is absent."""
        missing = str(tmp_path / "not_written.json")
        with patch("scripts.experiment_854_milestone_retro.DELIVERABLE", missing):
            with pytest.raises(AssertionError, match="Deliverable not written"):
                assert_deliverable_written()

    def test_assert_deliverable_written_fails_on_wrong_schema(self, tmp_path):
        """assert_deliverable_written must raise AssertionError on wrong schema version."""
        deliverable = tmp_path / "retro.json"
        artifact = {
            "schema": "carnot.operational_retro.v39",  # wrong version
            "milestone": "2026.04.65",
            "experiment": 854,
            "honest_verdict": "test",
            "n_criteria_met": 7,
            "n_criteria_total": 12,
            "success_criteria": [{}] * 12,
            "retros_closed": [],
            "retros_opened": [],
            "retros_still_open": [],
            "improvements_suggested": [],
            "total_wall_time_minutes": 4174,
            "experiments_completed": 762,
            "avg_time_per_experiment_minutes": 5.48,
            "slowest_5_experiments": [{}] * 5,
        }
        deliverable.write_text(json.dumps(artifact))
        with patch("scripts.experiment_854_milestone_retro.DELIVERABLE",
                   str(deliverable)):
            with pytest.raises(AssertionError, match="Schema version wrong"):
                assert_deliverable_written()

    def test_assert_deliverable_written_fails_on_wrong_criteria_count(self, tmp_path):
        """assert_deliverable_written must raise when success_criteria list is not length 12."""
        deliverable = tmp_path / "retro.json"
        artifact = {
            "schema": "carnot.operational_retro.v40",
            "milestone": "2026.04.65",
            "experiment": 854,
            "honest_verdict": "test",
            "n_criteria_met": 7,
            "n_criteria_total": 11,
            "success_criteria": [{}] * 11,  # wrong count
            "retros_closed": [],
            "retros_opened": [],
            "retros_still_open": [],
            "improvements_suggested": [],
            "total_wall_time_minutes": 4174,
            "experiments_completed": 762,
            "avg_time_per_experiment_minutes": 5.48,
            "slowest_5_experiments": [{}] * 5,
        }
        deliverable.write_text(json.dumps(artifact))
        with patch("scripts.experiment_854_milestone_retro.DELIVERABLE",
                   str(deliverable)):
            with pytest.raises(AssertionError, match="Expected 12 criteria"):
                assert_deliverable_written()


# ---------------------------------------------------------------------------
# Tests: improvements_suggested
# ---------------------------------------------------------------------------

class TestImprovements:
    def test_improvements_list_is_non_empty(self):
        """IMPROVEMENTS must contain at least one entry."""
        assert len(IMPROVEMENTS) > 0

    def test_all_improvements_have_required_fields(self):
        """Every improvement entry must have priority, action, and rationale."""
        for imp in IMPROVEMENTS:
            assert "priority" in imp, f"Missing priority in: {imp}"
            assert "action" in imp, f"Missing action in: {imp}"
            assert "rationale" in imp, f"Missing rationale in: {imp}"

    def test_immediate_items_present(self):
        """At least one IMMEDIATE-priority improvement must be present."""
        immediates = [i for i in IMPROVEMENTS if i["priority"] == "IMMEDIATE"]
        assert len(immediates) >= 1

    def test_model_download_is_immediate(self):
        """RETRO-SOTA-MODEL-DOWNLOAD fix must be IMMEDIATE priority."""
        actions = " ".join(i["action"] for i in IMPROVEMENTS)
        assert "GGUF" in actions or "model download" in actions.lower()


# ---------------------------------------------------------------------------
# Tests: constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_milestone_experiments_is_12(self):
        """MILESTONE_EXPERIMENTS must be 12 (11 experiments + 1 retro)."""
        assert MILESTONE_EXPERIMENTS == 12

    def test_experiment_cap_is_700(self):
        """EXPERIMENT_CAP must be 700 per CLAUDE.md governance rule."""
        assert EXPERIMENT_CAP == 700

    def test_prior_total_matches_64_retro(self):
        """PRIOR_TOTAL_WALL_TIME_MINUTES must match .64 augmented retro value (3971)."""
        assert PRIOR_TOTAL_WALL_TIME_MINUTES == 3971

    def test_prior_experiments_matches_64_retro(self):
        """PRIOR_EXPERIMENTS_COMPLETED must match .64 augmented retro value (750)."""
        assert PRIOR_EXPERIMENTS_COMPLETED == 750

    def test_milestone_wall_time_shorter_than_64(self):
        """Milestone .65 wall time must be shorter than .64 (improvement asserted)."""
        assert MILESTONE_WALL_TIME_MINUTES < PRIOR_MILESTONE_WALL_TIME_MINUTES
