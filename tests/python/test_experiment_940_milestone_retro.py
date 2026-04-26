"""Tests for Experiment 940: Milestone 2026.04.72 Retrospective.

Verifies that the retro script correctly evaluates all 12 success criteria,
builds a valid deliverable, and produces accurate pass/fail counts.

Spec: REQ-INFRA-073, SCENARIO-INFRA-073

Why these tests:
  The retro is the governance record for milestone .72. Incorrect criteria
  counts or wrong pass/fail flags would mislead the .73 planner. These tests
  exercise the evaluation logic with controlled stub data, catching lookup
  bugs and threshold errors before the conductor archives the milestone.
"""

from __future__ import annotations

from unittest.mock import patch

from scripts.experiment_940_milestone_retro_72 import build_artifact, evaluate_criteria

# ---------------------------------------------------------------------------
# Stub data builders — one function per experiment, composable.
# ---------------------------------------------------------------------------


def _r929(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "preflight_complete", "status": "success"}
    if overrides:
        d.update(overrides)
    return d


def _r930(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "math_repair_zero", "signed_improvement": 0.0, "status": "success"}
    if overrides:
        d.update(overrides)
    return d


def _r931(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "blocked_gate_check_failed", "status": "blocked"}
    if overrides:
        d.update(overrides)
    return d


def _r932(overrides: dict | None = None) -> dict:
    d = {
        "honest_verdict": "dualgpu_speedup_confirmed",
        "observed_speedup": 1.96,
        "status": "success",
    }
    if overrides:
        d.update(overrides)
    return d


def _r933(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "hf_published", "hf_authenticated": True, "status": "success"}
    if overrides:
        d.update(overrides)
    return d


def _r934(overrides: dict | None = None) -> dict:
    d = {
        "honest_verdict": "ipfs_mirror_established",
        "ipfs_cid_vjepa": "QmTkGjpN5fYNnC3g8Gx8sPWHZJKkw8oGVDKwWT6sZbVaGN",
        "status": "success",
    }
    if overrides:
        d.update(overrides)
    return d


def _r935(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "tier2_code_memory_works", "status": "success"}
    if overrides:
        d.update(overrides)
    return d


def _r936(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "real_data_improves_over_synthetic", "status": "success"}
    if overrides:
        d.update(overrides)
    return d


def _r937(overrides: dict | None = None) -> dict:
    d = {
        "honest_verdict": "symbolic_kan_viable",
        "auc_symbolic": 0.9344,
        "delta_auc": 0.7136,
        "status": "success",
    }
    if overrides:
        d.update(overrides)
    return d


def _r938(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "tier28_wired", "status": "success"}
    if overrides:
        d.update(overrides)
    return d


def _r939(overrides: dict | None = None) -> dict:
    d = {"honest_verdict": "blocked_gate_check_failed", "status": "blocked"}
    if overrides:
        d.update(overrides)
    return d


def _stub_loads(
    r929_kw=None,
    r930_kw=None,
    r931_kw=None,
    r932_kw=None,
    r933_kw=None,
    r934_kw=None,
    r935_kw=None,
    r936_kw=None,
    r937_kw=None,
    r938_kw=None,
    r939_kw=None,
):
    """
    Return a mapping from filename substring to stub dict so we can patch
    the private _load() function without touching disk.
    """
    return {
        "929": _r929(r929_kw),
        "930": _r930(r930_kw),
        "931": _r931(r931_kw),
        "932": _r932(r932_kw),
        "933": _r933(r933_kw),
        "934": _r934(r934_kw),
        "935": _r935(r935_kw),
        "936": _r936(r936_kw),
        "937": _r937(r937_kw),
        "938": _r938(r938_kw),
        "939": _r939(r939_kw),
    }


def _patch_load(stubs: dict):
    """
    Context-manager helper: patches scripts.experiment_940_milestone_retro_72._load
    so it returns stubs keyed by experiment-number substring.
    """

    def fake_load(filename: str) -> dict:
        for key, data in stubs.items():
            if f"_{key}_" in filename or filename.startswith(key):
                return data
        raise FileNotFoundError(f"No stub for {filename}")

    return patch("scripts.experiment_940_milestone_retro_72._load", side_effect=fake_load)


# ---------------------------------------------------------------------------
# Tests: individual criteria
# ---------------------------------------------------------------------------


class TestEvaluateCriteria:
    """SCENARIO-INFRA-073: per-criterion evaluation logic."""

    def test_baseline_passes_10_of_12(self):
        """Baseline stubs match actual .72 run: 10/12 criteria pass."""
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert sum(results.values()) == 10

    # Criterion 1
    def test_preflight_complete_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["preflight_complete"] is True

    def test_preflight_complete_false(self):
        stubs = _stub_loads(r929_kw={"honest_verdict": "preflight_failed"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["preflight_complete"] is False

    # Criterion 2
    def test_math_repair_working_false_on_zero_improvement(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["math_repair_working"] is False

    def test_math_repair_working_true_on_positive(self):
        stubs = _stub_loads(r930_kw={"signed_improvement": 0.05})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["math_repair_working"] is True

    # Criterion 3
    def test_combined_pipeline_viable_true_when_gated(self):
        """Gated-blocked is an accepted state — criterion passes."""
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["combined_pipeline_viable"] is True

    def test_combined_pipeline_viable_false_when_not_blocked_and_not_improved(self):
        """If Exp 931 ran but produced no improvement, criterion fails."""
        stubs = _stub_loads(r931_kw={"honest_verdict": "no_improvement", "status": "success"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["combined_pipeline_viable"] is False

    # Criterion 4
    def test_dualgpu_confirmed_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["dualgpu_throughput_confirmed"] is True

    def test_dualgpu_confirmed_false(self):
        stubs = _stub_loads(r932_kw={"honest_verdict": "dualgpu_timeout", "observed_speedup": 0.8})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["dualgpu_throughput_confirmed"] is False

    # Criterion 5
    def test_hf_published_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["hf_published"] is True

    def test_hf_published_false_when_not_authenticated(self):
        stubs = _stub_loads(r933_kw={"hf_authenticated": False, "honest_verdict": "auth_failed"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["hf_published"] is False

    # Criterion 6
    def test_ipfs_mirror_established_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["ipfs_mirror_established"] is True

    def test_ipfs_mirror_established_false_when_cid_none(self):
        stubs = _stub_loads(r934_kw={"ipfs_cid_vjepa": None})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["ipfs_mirror_established"] is False

    # Criterion 7
    def test_tier2_code_memory_works_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["tier2_code_memory_works"] is True

    def test_tier2_code_memory_works_accepts_partial(self):
        stubs = _stub_loads(r935_kw={"honest_verdict": "partial"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["tier2_code_memory_works"] is True

    def test_tier2_code_memory_works_false(self):
        stubs = _stub_loads(r935_kw={"honest_verdict": "blocked_gate_check_failed"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["tier2_code_memory_works"] is False

    # Criterion 8
    def test_kan_tier4_real_data_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["kan_tier4_real_data"] is True

    def test_kan_tier4_real_data_false_when_blocked(self):
        stubs = _stub_loads(r936_kw={"honest_verdict": "blocked_gate_check_failed"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["kan_tier4_real_data"] is False

    # Criterion 9
    def test_symbolic_kan_viable_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["symbolic_kan_viable"] is True

    def test_symbolic_kan_viable_false_below_threshold(self):
        stubs = _stub_loads(r937_kw={"auc_symbolic": 0.65})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["symbolic_kan_viable"] is False

    # Criterion 10
    def test_tier28_wired_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["tier28_wired"] is True

    def test_tier28_wired_accepts_no_activation_variant(self):
        stubs = _stub_loads(r938_kw={"honest_verdict": "tier28_wired_no_activation"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["tier28_wired"] is True

    def test_tier28_wired_false(self):
        stubs = _stub_loads(r938_kw={"honest_verdict": "tier28_integration_failed"})
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["tier28_wired"] is False

    # Criterion 11
    def test_sc_energy_viable_false_when_blocked(self):
        """Exp 939 blocked → no auc field → criterion fails."""
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["sc_energy_viable"] is False

    def test_sc_energy_viable_true_when_above_threshold(self):
        stubs = _stub_loads(
            r939_kw={"honest_verdict": "sc_energy_viable", "auc": 0.85, "status": "success"}
        )
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["sc_energy_viable"] is True

    # Criterion 12
    def test_retro_complete_always_true(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            results, _ = evaluate_criteria()
        assert results["retro_complete"] is True


# ---------------------------------------------------------------------------
# Tests: build_artifact output shape
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """SCENARIO-INFRA-073: deliverable artifact structure."""

    def test_required_fields_present(self):
        """All required schema fields must be present in the artifact."""
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        required = [
            "experiment",
            "milestone",
            "title",
            "run_date",
            "status",
            "honest_verdict",
            "n_criteria_met",
            "n_criteria_total",
            "criteria_results",
            "criteria_details",
            "open_retros_entering_73",
            "headline_findings",
        ]
        for field in required:
            assert field in artifact, f"Missing required field: {field}"

    def test_experiment_number_is_940(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert artifact["experiment"] == 940

    def test_milestone_label(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert artifact["milestone"] == "2026.04.72"

    def test_n_criteria_total_is_12(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert artifact["n_criteria_total"] == 12

    def test_n_criteria_met_matches_sum(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert artifact["n_criteria_met"] == sum(artifact["criteria_results"].values())

    def test_honest_verdict_is_milestone_complete(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert artifact["honest_verdict"] == "milestone_complete"

    def test_status_is_success(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert artifact["status"] == "success"

    def test_open_retros_is_list(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert isinstance(artifact["open_retros_entering_73"], list)

    def test_headline_findings_nonempty(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert len(artifact["headline_findings"]) >= 1

    def test_criteria_details_has_all_12_keys(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        assert len(artifact["criteria_details"]) == 12

    def test_each_detail_has_passed_field(self):
        stubs = _stub_loads()
        with _patch_load(stubs):
            artifact = build_artifact()
        for name, detail in artifact["criteria_details"].items():
            assert "passed" in detail, f"criteria_details[{name}] missing 'passed' field"
