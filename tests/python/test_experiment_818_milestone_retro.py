"""Tests for Experiment 818 — Milestone 2026.04.62 Operational Retrospective.

Covers every function in scripts/experiment_818_milestone_retro.py to ensure
100% branch coverage on newly-added code.

All tests use synthetic fixture artifacts written into a tmp_path so the
real results/ directory is never touched.

Spec: REQ-METRICS-010, SCENARIO-RETRO-034
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_818_milestone_retro as exp818  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_artifacts() -> dict[int, dict]:
    """Return a dict of minimal passing artifacts for all 12 milestone experiments.

    Values are chosen so every success criterion evaluates to True, giving a
    clean baseline for individual override tests.
    """
    return {
        806: {"experiment": 806, "honest_verdict": "prereqs_gate_ready", "duration_s": 0.5},
        807: {
            "experiment": 807,
            "honest_verdict": "tools_installed_synthesis_clean",
            "duration_s": 10.0,
        },
        808: {
            "experiment": 808,
            "honest_verdict": "jepa_v22_above_gate",
            "ood_auc": 0.80,
            "duration_s": 600.0,
        },
        809: {
            "experiment": 809,
            "honest_verdict": "rapbm_ood_improved",
            "ood_auc": 0.82,
            "duration_s": 240.0,
        },
        810: {
            "experiment": 810,
            "honest_verdict": "retro_028_closed",
            "retro_028_closed": True,
            "duration_s": 120.0,
        },
        811: {"experiment": 811, "honest_verdict": "code_repair_positive", "duration_s": 10.0},
        812: {
            "experiment": 812,
            "honest_verdict": "injection_works",
            "mean_energy_delta_pct_errors": 0.15,
            "mean_energy_delta_pct_clean": -0.10,
            "duration_s": 2.0,
        },
        813: {
            "experiment": 813,
            "honest_verdict": "injection_live",
            "retro_constraint_zero_delta_closed": True,
            "duration_s": 5.0,
        },
        814: {"experiment": 814, "honest_verdict": "tier1_relay_works_live", "duration_s": 5.0},
        815: {"experiment": 815, "honest_verdict": "vg_search_effective", "duration_s": 0.5},
        816: {"experiment": 816, "honest_verdict": "synthesis_clean_n32", "duration_s": 9.0},
        817: {
            "experiment": 817,
            "honest_verdict": "arbiter_correct",
            "arbiter_accuracy": 1.0,
            "duration_s": 0.016,
        },
    }


def _write_all(tmp_results: Path, artifacts: dict[int, dict]) -> None:
    """Write all artifact dicts into tmp_results with canonical filenames."""
    _filenames = {
        806: "experiment_806_milestone_prereqs_gate.json",
        807: "experiment_807_oss_cad_suite_install.json",
        808: "experiment_808_jepa_v22_retrain.json",
        809: "experiment_809_jepa_v22_rapbm.json",
        810: "experiment_810_gemma4_oom_fix_v5.json",
        811: "experiment_811_sota_gguf_code_repair_v4.json",
        812: "experiment_812_ising_constraint_injection.json",
        813: "experiment_813_constraint_addition_live.json",
        814: "experiment_814_fr11_tier1_live_relay.json",
        815: "experiment_815_vg_search_scheduling.json",
        816: "experiment_816_kv260_synthesis_v2.json",
        817: "experiment_817_multi_agent_arbiter.json",
    }
    for exp_id, payload in artifacts.items():
        fn = _filenames.get(exp_id, f"experiment_{exp_id}_fixture.json")
        (tmp_results / fn).write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# load_artifact
# ---------------------------------------------------------------------------


def test_load_artifact_returns_empty_dict_when_missing(tmp_path: Path) -> None:
    """load_artifact returns {} when no file for the experiment exists.

    Spec: REQ-METRICS-010
    """
    result = exp818.load_artifact(806, results_dir=tmp_path)
    assert result == {}


def test_load_artifact_returns_empty_dict_on_corrupt_json(tmp_path: Path) -> None:
    """load_artifact returns {} when the JSON is malformed.

    Spec: REQ-METRICS-010
    """
    (tmp_path / "experiment_806_milestone_prereqs_gate.json").write_text("{bad json}")
    result = exp818.load_artifact(806, results_dir=tmp_path)
    assert result == {}


def test_load_artifact_returns_empty_dict_when_json_is_list(tmp_path: Path) -> None:
    """load_artifact returns {} when the JSON root is a list, not an object.

    Spec: REQ-METRICS-010
    """
    (tmp_path / "experiment_807_oss_cad_suite_install.json").write_text("[1, 2, 3]")
    result = exp818.load_artifact(807, results_dir=tmp_path)
    assert result == {}


def test_load_artifact_succeeds_with_valid_json(tmp_path: Path) -> None:
    """load_artifact returns the dict when the JSON is valid.

    Spec: REQ-METRICS-010
    """
    payload = {"experiment": 810, "retro_028_closed": True}
    (tmp_path / "experiment_810_gemma4_oom_fix_v5.json").write_text(json.dumps(payload))
    result = exp818.load_artifact(810, results_dir=tmp_path)
    assert result["retro_028_closed"] is True


def test_load_artifact_skips_operational_retro_files(tmp_path: Path) -> None:
    """load_artifact ignores files that contain 'operational_retro' in their name.

    Spec: REQ-METRICS-010
    """
    (tmp_path / "experiment_806_operational_retro.json").write_text('{"experiment": 806}')
    result = exp818.load_artifact(806, results_dir=tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# load_all_artifacts
# ---------------------------------------------------------------------------


def test_load_all_artifacts_returns_12_keys(tmp_path: Path) -> None:
    """load_all_artifacts returns a dict with exactly 12 experiment keys.

    Spec: REQ-METRICS-010
    """
    arts = exp818.load_all_artifacts(results_dir=tmp_path)
    assert set(arts.keys()) == set(exp818._MILESTONE_EXPS)
    assert len(arts) == 12


# ---------------------------------------------------------------------------
# evaluate_success_criteria — all-pass baseline
# ---------------------------------------------------------------------------


def test_evaluate_success_criteria_all_pass() -> None:
    """All 9 criteria pass when every artifact meets its threshold.

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    artifacts = _minimal_artifacts()
    criteria = exp818.evaluate_success_criteria(artifacts)
    assert all(criteria.values()), f"Expected all True, got: {criteria}"
    assert len(criteria) == 9


def test_evaluate_success_criteria_all_fail_on_empty() -> None:
    """All 9 criteria fail when all artifacts are empty dicts.

    Spec: REQ-METRICS-010
    """
    artifacts = {eid: {} for eid in exp818._MILESTONE_EXPS}
    criteria = exp818.evaluate_success_criteria(artifacts)
    assert not any(criteria.values()), f"Expected all False, got: {criteria}"


# ---------------------------------------------------------------------------
# Individual criterion tests
# ---------------------------------------------------------------------------


def test_criterion_prereqs_gate_requires_exact_verdict() -> None:
    """prereqs_gate_implemented requires honest_verdict == 'prereqs_gate_ready'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[806]["honest_verdict"] = "prereqs_incomplete"
    assert not exp818.evaluate_success_criteria(arts)["prereqs_gate_implemented"]

    arts[806]["honest_verdict"] = "prereqs_gate_ready"
    assert exp818.evaluate_success_criteria(arts)["prereqs_gate_implemented"]


def test_criterion_fpga_tools_installed_accepts_both_verdicts() -> None:
    """fpga_tools_installed accepts 'tools_installed_synthesis_clean' or 'already_installed'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[807]["honest_verdict"] = "already_installed"
    assert exp818.evaluate_success_criteria(arts)["fpga_tools_installed"]

    arts[807]["honest_verdict"] = "tools_not_found"
    assert not exp818.evaluate_success_criteria(arts)["fpga_tools_installed"]


def test_criterion_jepa_v22_ood_viable_boundary() -> None:
    """jepa_v22_ood_viable uses >= 0.75 threshold.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[808]["ood_auc"] = 0.7499
    assert not exp818.evaluate_success_criteria(arts)["jepa_v22_ood_viable"]

    arts[808]["ood_auc"] = 0.75
    assert exp818.evaluate_success_criteria(arts)["jepa_v22_ood_viable"]


def test_criterion_retro_028_closed_requires_true_flag() -> None:
    """retro_028_closed is True only when Exp 810 retro_028_closed field is True.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[810]["retro_028_closed"] = False
    assert not exp818.evaluate_success_criteria(arts)["retro_028_closed"]

    arts[810]["retro_028_closed"] = True
    assert exp818.evaluate_success_criteria(arts)["retro_028_closed"]


def test_criterion_sota_code_repair_positive_requires_exact_verdict() -> None:
    """sota_code_repair_positive requires honest_verdict == 'code_repair_positive'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[811]["honest_verdict"] = "blocked_model_load_failed"
    assert not exp818.evaluate_success_criteria(arts)["sota_code_repair_positive"]


def test_criterion_constraint_injection_wired_requires_injection_works() -> None:
    """constraint_injection_wired requires honest_verdict == 'injection_works'.

    injection_negative_delta (the actual .62 result) must NOT satisfy this.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[812]["honest_verdict"] = "injection_negative_delta"
    assert not exp818.evaluate_success_criteria(arts)["constraint_injection_wired"]

    arts[812]["honest_verdict"] = "injection_works"
    assert exp818.evaluate_success_criteria(arts)["constraint_injection_wired"]


def test_criterion_constraint_addition_live_requires_true_flag() -> None:
    """constraint_addition_live requires retro_constraint_zero_delta_closed == True.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[813]["retro_constraint_zero_delta_closed"] = False
    assert not exp818.evaluate_success_criteria(arts)["constraint_addition_live"]


def test_criterion_tier1_relay_live_requires_exact_verdict() -> None:
    """tier1_relay_live requires honest_verdict == 'tier1_relay_works_live'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[814]["honest_verdict"] = "blocked_no_delta"
    assert not exp818.evaluate_success_criteria(arts)["tier1_relay_live"]


def test_criterion_kv260_synthesis_clean_accepts_both_variants() -> None:
    """kv260_synthesis_clean accepts 'synthesis_clean_n32' or 'synthesis_clean_n32_n64'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[816]["honest_verdict"] = "synthesis_clean_n32_n64"
    assert exp818.evaluate_success_criteria(arts)["kv260_synthesis_clean"]

    arts[816]["honest_verdict"] = "synthesis_failed"
    assert not exp818.evaluate_success_criteria(arts)["kv260_synthesis_clean"]


# ---------------------------------------------------------------------------
# compute_wall_time
# ---------------------------------------------------------------------------


def test_compute_wall_time_improvement_when_faster() -> None:
    """improvement is True when total wall time is less than previous milestone (9.393 min).

    Spec: REQ-METRICS-010
    """
    # 12 experiments × 1s each = 12s = 0.2 min < 9.393 min → improvement
    arts = {eid: {"duration_s": 1.0} for eid in exp818._MILESTONE_EXPS}
    wt = exp818.compute_wall_time(arts)
    assert wt["improvement"] is True


def test_compute_wall_time_regression_when_slower() -> None:
    """improvement is False when total wall time exceeds previous milestone.

    Spec: REQ-METRICS-010
    """
    # 12 experiments × 60s each = 720s = 12 min > 9.393 min → regression
    arts = {eid: {"duration_s": 60.0} for eid in exp818._MILESTONE_EXPS}
    wt = exp818.compute_wall_time(arts)
    assert wt["improvement"] is False


def test_compute_wall_time_uses_elapsed_minutes_for_timed_out() -> None:
    """compute_wall_time uses elapsed_minutes * 60 for timed-out experiments.

    Spec: REQ-METRICS-010
    """
    arts = {eid: {"duration_s": 0.0} for eid in exp818._MILESTONE_EXPS}
    arts[808]["timed_out"] = True
    arts[808]["elapsed_minutes"] = 45.0
    wt = exp818.compute_wall_time(arts)
    # elapsed_minutes=45.0 → 45*60=2700s → 45.0 min from 808 alone
    assert wt["total_wall_time_min"] == pytest.approx(45.0, abs=0.01)


def test_compute_wall_time_empty_artifacts_gives_zero_mean() -> None:
    """compute_wall_time handles zero-ran experiments without division by zero.

    Spec: REQ-METRICS-010
    """
    arts: dict[int, dict] = {}
    wt = exp818.compute_wall_time(arts)
    assert wt["mean_min_per_experiment"] == 0.0


# ---------------------------------------------------------------------------
# rank_experiments_by_duration
# ---------------------------------------------------------------------------


def test_rank_experiments_slowest_first() -> None:
    """rank_experiments_by_duration returns experiments slowest-first.

    Spec: REQ-METRICS-010
    """
    arts = {
        806: {"duration_s": 1.0, "title": "fast"},
        808: {"duration_s": 600.0, "title": "slow"},
        815: {"duration_s": 10.0, "title": "mid"},
    }
    ranked = exp818.rank_experiments_by_duration(arts)
    assert ranked[0]["exp_id"] == 808
    assert ranked[-1]["exp_id"] == 806


def test_rank_experiments_includes_timed_out_duration() -> None:
    """rank_experiments_by_duration converts elapsed_minutes to seconds for ordering.

    Spec: REQ-METRICS-010
    """
    arts = {
        806: {"duration_s": 60.0},
        808: {"timed_out": True, "elapsed_minutes": 45.0, "duration_s": 0.0},
    }
    ranked = exp818.rank_experiments_by_duration(arts)
    assert ranked[0]["exp_id"] == 808  # 45*60=2700s > 60s


# ---------------------------------------------------------------------------
# classify_retros
# ---------------------------------------------------------------------------


def test_classify_retros_closes_retro028_when_flag_true() -> None:
    """classify_retros closes RETRO-028 when Exp 810 retro_028_closed=True.

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    retros = exp818.classify_retros(arts)
    assert any("RETRO-028" in r and "resolved" in r for r in retros["retros_closed"])
    assert not any("RETRO-028" in r for r in retros["retros_still_open"])


def test_classify_retros_keeps_retro028_open_when_false() -> None:
    """classify_retros keeps RETRO-028 open when retro_028_closed=False.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[810]["retro_028_closed"] = False
    retros = exp818.classify_retros(arts)
    assert any("RETRO-028" in r for r in retros["retros_still_open"])
    assert not any("RETRO-028" in r and "resolved" in r for r in retros["retros_closed"])


def test_classify_retros_closes_kv260_tools_when_installed() -> None:
    """classify_retros closes RETRO-KV260-TOOLS-UNAVAILABLE on tools_installed verdict.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[807]["honest_verdict"] = "tools_installed_synthesis_clean"
    retros = exp818.classify_retros(arts)
    assert any("RETRO-KV260-TOOLS-UNAVAILABLE" in r for r in retros["retros_closed"])


def test_classify_retros_keeps_kv260_tools_open_when_not_installed() -> None:
    """classify_retros keeps RETRO-KV260-TOOLS-UNAVAILABLE open on failure verdict.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[807]["honest_verdict"] = "tools_not_found"
    retros = exp818.classify_retros(arts)
    assert any("RETRO-KV260-TOOLS-UNAVAILABLE" in r for r in retros["retros_still_open"])


def test_classify_retros_closes_constraint_zero_delta_when_injection_works() -> None:
    """classify_retros closes RETRO-CONSTRAINT-ZERO-DELTA when Exp 812 is injection_works.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[812]["honest_verdict"] = "injection_works"
    retros = exp818.classify_retros(arts)
    assert any("RETRO-CONSTRAINT-ZERO-DELTA" in r for r in retros["retros_closed"])


def test_classify_retros_keeps_constraint_delta_open_on_negative_delta() -> None:
    """classify_retros keeps RETRO-CONSTRAINT-ZERO-DELTA open on injection_negative_delta.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[812]["honest_verdict"] = "injection_negative_delta"
    arts[812]["mean_energy_delta_pct_errors"] = -0.2884
    arts[812]["mean_energy_delta_pct_clean"] = -0.2884
    retros = exp818.classify_retros(arts)
    assert any("RETRO-CONSTRAINT-ZERO-DELTA" in r for r in retros["retros_still_open"])


def test_classify_retros_closes_tier1_when_relay_works() -> None:
    """classify_retros closes RETRO-TIER1-PLATEAU when Exp 814 verdict is tier1_relay_works_live.

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[814]["honest_verdict"] = "tier1_relay_works_live"
    retros = exp818.classify_retros(arts)
    assert any("RETRO-TIER1-PLATEAU" in r for r in retros["retros_closed"])


def test_classify_retros_keeps_tier1_open_when_blocked() -> None:
    """classify_retros keeps RETRO-TIER1-PLATEAU open when Exp 814 is blocked.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[814]["honest_verdict"] = "blocked_no_delta"
    retros = exp818.classify_retros(arts)
    assert any("RETRO-TIER1-PLATEAU" in r for r in retros["retros_still_open"])


def test_classify_retros_closes_jepa_ood_when_above_gate() -> None:
    """classify_retros closes RETRO-JEPA-V21-OOD-BELOW-GATE when best ood_auc >= 0.75.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[808]["ood_auc"] = 0.80
    retros = exp818.classify_retros(arts)
    assert any("RETRO-JEPA-V21-OOD-BELOW-GATE" in r for r in retros["retros_closed"])


def test_classify_retros_keeps_jepa_ood_open_when_below_gate() -> None:
    """classify_retros keeps RETRO-JEPA-V21-OOD-BELOW-GATE open when best ood_auc < 0.75.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[808]["ood_auc"] = 0.2
    arts[809]["ood_auc"] = 0.5
    retros = exp818.classify_retros(arts)
    assert any("RETRO-JEPA-V21-OOD-BELOW-GATE" in r for r in retros["retros_still_open"])


def test_classify_retros_opens_gguf_cache_import_when_blocked() -> None:
    """classify_retros opens RETRO-GGUF-CACHE-IMPORT when Exp 811 is blocked_model_load_failed.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[811]["honest_verdict"] = "blocked_model_load_failed"
    arts[811]["blocked_reason"] = "No module named 'carnot.pipeline.gguf_cache'"
    retros = exp818.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-GGUF-CACHE-IMPORT" in new_ids


def test_classify_retros_does_not_open_gguf_cache_when_not_blocked() -> None:
    """classify_retros does not open RETRO-GGUF-CACHE-IMPORT when code repair succeeded.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[811]["honest_verdict"] = "code_repair_positive"
    retros = exp818.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-GGUF-CACHE-IMPORT" not in new_ids


def test_classify_retros_opens_injection_no_discrimination_when_identical_deltas() -> None:
    """classify_retros opens RETRO-ISING-INJECTION-NO-DISCRIMINATION when error delta == clean delta.

    The actual .62 result shows identical energy changes for errors and clean responses,
    meaning the injection has no discriminating power.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[812]["honest_verdict"] = "injection_negative_delta"
    arts[812]["mean_energy_delta_pct_errors"] = -0.2884
    arts[812]["mean_energy_delta_pct_clean"] = -0.2884
    retros = exp818.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-ISING-INJECTION-NO-DISCRIMINATION" in new_ids


def test_classify_retros_does_not_open_injection_no_disc_when_injection_works() -> None:
    """classify_retros does not open RETRO-ISING-INJECTION-NO-DISCRIMINATION when injection_works.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[812]["honest_verdict"] = "injection_works"
    arts[812]["mean_energy_delta_pct_errors"] = 0.15
    arts[812]["mean_energy_delta_pct_clean"] = -0.10
    retros = exp818.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-ISING-INJECTION-NO-DISCRIMINATION" not in new_ids


def test_classify_retros_opens_arbiter_flat_energy_when_all_scores_zero() -> None:
    """classify_retros opens RETRO-ARBITER-FLAT-ENERGY when arbiter accuracy < 0.5.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[817]["honest_verdict"] = "arbiter_incorrect"
    arts[817]["arbiter_accuracy"] = 0.3333
    retros = exp818.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-ARBITER-FLAT-ENERGY" in new_ids


def test_classify_retros_does_not_open_arbiter_flat_when_accurate() -> None:
    """classify_retros does not open RETRO-ARBITER-FLAT-ENERGY when arbiter accuracy >= 0.5.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[817]["honest_verdict"] = "arbiter_correct"
    arts[817]["arbiter_accuracy"] = 1.0
    retros = exp818.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-ARBITER-FLAT-ENERGY" not in new_ids


# ---------------------------------------------------------------------------
# build_honest_verdict
# ---------------------------------------------------------------------------


def test_build_honest_verdict_contains_criteria_counts() -> None:
    """build_honest_verdict embeds the N_of_M criteria count in the output string.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    criteria = exp818.evaluate_success_criteria(arts)
    wall_time = exp818.compute_wall_time(arts)
    retros = exp818.classify_retros(arts)
    verdict = exp818.build_honest_verdict(criteria, wall_time, retros, arts)
    met = sum(1 for v in criteria.values() if v)
    assert f"{met}_of_9" in verdict


def test_build_honest_verdict_contains_improvement_or_regression() -> None:
    """build_honest_verdict contains 'IMPROVEMENT' or 'REGRESSION' based on wall time.

    Spec: REQ-METRICS-010
    """
    arts = {eid: {"duration_s": 1.0} for eid in exp818._MILESTONE_EXPS}
    arts[808] = {"ood_auc": 0.5, "duration_s": 1.0}
    arts[809] = {"ood_auc": 0.5, "duration_s": 1.0}
    arts[817] = {"arbiter_accuracy": 0.5, "duration_s": 1.0}
    criteria = exp818.evaluate_success_criteria(arts)
    wall_time = exp818.compute_wall_time(arts)
    retros = exp818.classify_retros(arts)
    verdict = exp818.build_honest_verdict(criteria, wall_time, retros, arts)
    assert "IMPROVEMENT" in verdict or "REGRESSION" in verdict


# ---------------------------------------------------------------------------
# Full run integration (uses tmp_path to avoid touching real results/)
# ---------------------------------------------------------------------------


def test_run_produces_valid_deliverable(tmp_path: Path) -> None:
    """run() writes a valid JSON artifact with all required schema fields.

    This is an integration test that exercises the full pipeline with fixtures.

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    tmp_results = tmp_path / "results"
    tmp_results.mkdir()
    artifacts = _minimal_artifacts()
    _write_all(tmp_results, artifacts)

    deliverable = tmp_path / "operational_retro_2026_04_62.json"
    result = exp818.run(deliverable=deliverable, results_dir=tmp_results)

    assert deliverable.exists(), "Deliverable file must be written to disk"
    on_disk = json.loads(deliverable.read_text())

    for fld in [
        "milestone",
        "experiment_range",
        "n_experiments",
        "criteria_met_count",
        "criteria_total",
        "retros_closed",
        "retros_opened",
        "retros_still_open",
        "slowest_experiment",
        "fastest_experiment",
        "improvements_suggested",
        "estimated_time_savings_pct",
        "honest_verdict",
        "status",
    ]:
        assert fld in on_disk, f"Missing required field: {fld}"

    assert on_disk["status"] == "success"
    assert on_disk["criteria_total"] == 9
    assert on_disk["milestone"] == "2026.04.62"
    # build_result() overwrites schema with a sorted list of output keys (template contract)
    assert isinstance(on_disk["schema"], list)
    assert "milestone" in on_disk["schema"]


def test_run_criteria_met_count_equals_nine_on_all_pass(tmp_path: Path) -> None:
    """criteria_met_count equals 9 when all artifacts satisfy their thresholds.

    Spec: REQ-METRICS-010
    """
    tmp_results = tmp_path / "results"
    tmp_results.mkdir()
    _write_all(tmp_results, _minimal_artifacts())
    deliverable = tmp_path / "retro.json"
    result = exp818.run(deliverable=deliverable, results_dir=tmp_results)
    assert result["criteria_met_count"] == 9


def test_run_criteria_met_count_is_four_on_actual_artifacts() -> None:
    """criteria_met_count == 4 on the real .62 experiment results.

    The four passing criteria are:
      - prereqs_gate_implemented (Exp 806 prereqs_gate_ready)
      - fpga_tools_installed (Exp 807 tools_installed_synthesis_clean)
      - retro_028_closed (Exp 810 retro_028_closed=True)
      - kv260_synthesis_clean (Exp 816 synthesis_clean_n32)

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    real_deliverable = REPO_ROOT / "results" / "operational_retro_2026_04_62.json"
    if not real_deliverable.exists():
        pytest.skip("Real deliverable not found — run experiment_818_milestone_retro.py first")
    artifact = json.loads(real_deliverable.read_text())
    assert artifact["criteria_met_count"] == 4
    assert artifact["criteria_total"] == 9


def test_run_slowest_experiment_is_exp_808(tmp_path: Path) -> None:
    """slowest_experiment.exp_id == 808 on the .62 artifacts (Exp 808 at 597.6s).

    Spec: REQ-METRICS-010
    """
    real_deliverable = REPO_ROOT / "results" / "operational_retro_2026_04_62.json"
    if not real_deliverable.exists():
        pytest.skip("Real deliverable not found — run experiment_818_milestone_retro.py first")
    artifact = json.loads(real_deliverable.read_text())
    assert artifact["slowest_experiment"]["exp_id"] == 808
