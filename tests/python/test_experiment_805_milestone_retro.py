"""Tests for Experiment 805 — Milestone 2026.04.61 Operational Retrospective.

Covers every function in scripts/experiment_805_milestone_retro.py to ensure
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

import experiment_805_milestone_retro as exp805  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifact(directory: Path, exp_id: int, filename: str, payload: dict) -> None:
    """Write a JSON fixture artifact into the given directory."""
    (directory / filename).write_text(json.dumps(payload))


def _minimal_artifacts() -> dict[int, dict]:
    """Return a dict of minimal passing artifacts for all 12 milestone experiments.

    Values are chosen so every success criterion evaluates to True, giving a
    clean baseline for individual override tests.
    """
    return {
        793: {"experiment": 793, "honest_verdict": "1_unguarded_sites_found", "duration_s": 1.0},
        794: {
            "experiment": 794,
            "honest_verdict": "tools_installed_synthesis_clean",
            "tools_installed": True,
            "duration_s": 2.0,
        },
        795: {"experiment": 795, "honest_verdict": "retro_028_closed", "duration_s": 60.0},
        796: {"experiment": 796, "honest_verdict": "code_repair_positive", "duration_s": 5.0},
        797: {
            "experiment": 797,
            "honest_verdict": "multi_source_corpus_adequate",
            "n_labeled_total": 150,
            "duration_s": 3.0,
        },
        798: {
            "experiment": 798,
            "honest_verdict": "cpmi_augmentation_adequate",
            "augmentation_ratio": 3.0,
            "duration_s": 1.0,
        },
        799: {
            "experiment": 799,
            "honest_verdict": "jepa_v21_above_gate",
            "ood_auc": 0.80,
            "duration_s": 30.0,
        },
        800: {
            "experiment": 800,
            "honest_verdict": "retrieval_auc_exceeds_target",
            "retrieval_auc_plain": 0.95,
            "duration_s": 4.0,
        },
        801: {
            "experiment": 801,
            "honest_verdict": "constraint_addition_positive",
            "constraint_addition_delta_overall": 0.05,
            "duration_s": 4.0,
        },
        802: {"experiment": 802, "honest_verdict": "tier1_relay_works", "duration_s": 4.0},
        803: {"experiment": 803, "honest_verdict": "hf_models_published", "duration_s": 2.0},
        804: {
            "experiment": 804,
            "honest_verdict": "synthesis_complete",
            "tools_installed": True,
            "duration_s": 0.0,
        },
    }


def _write_all(tmp_results: Path, artifacts: dict[int, dict]) -> None:
    """Write all artifact dicts into tmp_results with canonical filenames."""
    _filenames = {
        793: "experiment_793_manifest_full_scope_audit.json",
        794: "experiment_794_fpga_toolchain_install.json",
        795: "experiment_795_gemma4_oom_fix_v4.json",
        796: "experiment_796_sota_gguf_code_repair_v3.json",
        797: "experiment_797_jepa_v21_data_collection.json",
        798: "experiment_798_cpmi_pairs.json",
        799: "experiment_799_jepa_v21_retrain.json",
        800: "experiment_800_embedding_constraint_store.json",
        801: "experiment_801_embedding_constraint_addition.json",
        802: "experiment_802_fr11_embedding_relay.json",
        803: "experiment_803_hf_publish_v2.json",
        804: "experiment_804_kv260_synthesis_attempt.json",
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
    result = exp805.load_artifact(793, results_dir=tmp_path)
    assert result == {}


def test_load_artifact_returns_empty_dict_on_corrupt_json(tmp_path: Path) -> None:
    """load_artifact returns {} when the JSON is malformed.

    A corrupt artifact is as bad as a missing one — we cannot draw conclusions.

    Spec: REQ-METRICS-010
    """
    (tmp_path / "experiment_793_manifest_full_scope_audit.json").write_text("{bad json}")
    result = exp805.load_artifact(793, results_dir=tmp_path)
    assert result == {}


def test_load_artifact_returns_empty_dict_when_json_is_list(tmp_path: Path) -> None:
    """load_artifact returns {} when the JSON root is a list, not an object.

    Some multi-record artifacts (e.g. triples files) are JSON arrays.  The retro
    code requires a dict per experiment — a list is treated as not_run.

    Spec: REQ-METRICS-010
    """
    (tmp_path / "experiment_798_cpmi_pairs.json").write_text("[1, 2, 3]")
    result = exp805.load_artifact(798, results_dir=tmp_path)
    assert result == {}


def test_load_artifact_succeeds_with_valid_json(tmp_path: Path) -> None:
    """load_artifact returns the dict when the JSON is valid.

    Spec: REQ-METRICS-010
    """
    payload = {"experiment": 797, "n_labeled_total": 300}
    (tmp_path / "experiment_797_jepa_v21_data_collection.json").write_text(json.dumps(payload))
    result = exp805.load_artifact(797, results_dir=tmp_path)
    assert result["n_labeled_total"] == 300


def test_load_artifact_skips_operational_retro_files(tmp_path: Path) -> None:
    """load_artifact ignores files that contain 'operational_retro' in their name.

    This prevents the retro artifact from being loaded as its own input.

    Spec: REQ-METRICS-010
    """
    (tmp_path / "experiment_793_operational_retro.json").write_text('{"experiment": 793}')
    result = exp805.load_artifact(793, results_dir=tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# load_all_artifacts
# ---------------------------------------------------------------------------


def test_load_all_artifacts_returns_12_keys(tmp_path: Path) -> None:
    """load_all_artifacts returns a dict with exactly 12 experiment keys.

    Spec: REQ-METRICS-010
    """
    arts = exp805.load_all_artifacts(results_dir=tmp_path)
    assert set(arts.keys()) == {793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804}


# ---------------------------------------------------------------------------
# evaluate_success_criteria — all-pass baseline
# ---------------------------------------------------------------------------


def test_evaluate_success_criteria_all_pass(tmp_path: Path) -> None:
    """All 10 criteria pass when every artifact meets its threshold.

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    artifacts = _minimal_artifacts()
    criteria = exp805.evaluate_success_criteria(artifacts)
    assert all(criteria.values()), f"Expected all True, got: {criteria}"
    assert len(criteria) == 10


def test_evaluate_success_criteria_all_fail_on_empty(tmp_path: Path) -> None:
    """All 10 criteria fail when all artifacts are empty dicts.

    Spec: REQ-METRICS-010
    """
    artifacts = {eid: {} for eid in exp805._MILESTONE_EXPS}
    criteria = exp805.evaluate_success_criteria(artifacts)
    assert not any(criteria.values()), f"Expected all False, got: {criteria}"


# ---------------------------------------------------------------------------
# evaluate_success_criteria — individual criterion tests
# ---------------------------------------------------------------------------


def test_criterion_retro_028_closed_requires_exact_verdict() -> None:
    """retro_028_closed is True only when honest_verdict == 'retro_028_closed'.

    'partial_success' (the actual .61 result) must NOT satisfy this criterion.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[795]["honest_verdict"] = "partial_success"
    assert not exp805.evaluate_success_criteria(arts)["retro_028_closed"]

    arts[795]["honest_verdict"] = "retro_028_closed"
    assert exp805.evaluate_success_criteria(arts)["retro_028_closed"]


def test_criterion_sota_code_repair_positive_requires_exact_verdict() -> None:
    """sota_code_repair_positive is True only for honest_verdict == 'code_repair_positive'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[796]["honest_verdict"] = "gated_retro028_not_closed"
    assert not exp805.evaluate_success_criteria(arts)["sota_code_repair_positive"]


def test_criterion_jepa_v21_data_adequate_boundary() -> None:
    """jepa_v21_data_adequate uses >= 80 threshold.

    n_labeled_total=79 fails; 80 passes.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[797]["n_labeled_total"] = 79
    assert not exp805.evaluate_success_criteria(arts)["jepa_v21_data_adequate"]

    arts[797]["n_labeled_total"] = 80
    assert exp805.evaluate_success_criteria(arts)["jepa_v21_data_adequate"]


def test_criterion_cpmi_augmentation_works_boundary() -> None:
    """cpmi_augmentation_works uses >= 2.0 threshold.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[798]["augmentation_ratio"] = 1.99
    assert not exp805.evaluate_success_criteria(arts)["cpmi_augmentation_works"]

    arts[798]["augmentation_ratio"] = 2.0
    assert exp805.evaluate_success_criteria(arts)["cpmi_augmentation_works"]


def test_criterion_jepa_v21_ood_viable_boundary() -> None:
    """jepa_v21_ood_viable uses >= 0.75 threshold.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[799]["ood_auc"] = 0.7499
    assert not exp805.evaluate_success_criteria(arts)["jepa_v21_ood_viable"]

    arts[799]["ood_auc"] = 0.75
    assert exp805.evaluate_success_criteria(arts)["jepa_v21_ood_viable"]


def test_criterion_embedding_retrieval_works_uses_retrieval_auc_plain() -> None:
    """embedding_retrieval_works uses retrieval_auc_plain > 0.70.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[800]["retrieval_auc_plain"] = 0.70  # exactly 0.70 — NOT > 0.70
    assert not exp805.evaluate_success_criteria(arts)["embedding_retrieval_works"]

    arts[800]["retrieval_auc_plain"] = 0.71
    assert exp805.evaluate_success_criteria(arts)["embedding_retrieval_works"]


def test_criterion_embedding_retrieval_falls_back_to_retrieval_auc() -> None:
    """embedding_retrieval_works falls back to retrieval_auc when retrieval_auc_plain absent.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[800] = {"retrieval_auc": 0.80}
    assert exp805.evaluate_success_criteria(arts)["embedding_retrieval_works"]


def test_criterion_constraint_addition_positive() -> None:
    """constraint_addition_positive requires delta_overall > 0.0.

    0.0 (the actual .61 result) must fail.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[801]["constraint_addition_delta_overall"] = 0.0
    assert not exp805.evaluate_success_criteria(arts)["constraint_addition_positive"]


def test_criterion_tier1_relay_works_requires_exact_verdict() -> None:
    """tier1_relay_works requires honest_verdict == 'tier1_relay_works'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[802]["honest_verdict"] = "tier1_plateau_persists"
    assert not exp805.evaluate_success_criteria(arts)["tier1_relay_works"]


def test_criterion_fpga_tools_installed() -> None:
    """fpga_tools_installed requires tools_installed == True.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[794]["tools_installed"] = False
    assert not exp805.evaluate_success_criteria(arts)["fpga_tools_installed"]


def test_criterion_kv260_synthesis_attempted_fails_on_tools_not_installed() -> None:
    """kv260_synthesis_attempted fails when honest_verdict == 'tools_not_installed'.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[804]["honest_verdict"] = "tools_not_installed"
    assert not exp805.evaluate_success_criteria(arts)["kv260_synthesis_attempted"]

    arts[804]["honest_verdict"] = "synthesis_complete"
    assert exp805.evaluate_success_criteria(arts)["kv260_synthesis_attempted"]


# ---------------------------------------------------------------------------
# compute_wall_time
# ---------------------------------------------------------------------------


def test_compute_wall_time_improvement_when_faster() -> None:
    """improvement is True when total wall time is less than previous milestone.

    Spec: REQ-METRICS-010
    """
    # Previous milestone wall time is ~2.2 min
    # 60 seconds total = 1.0 min < 2.2 min → improvement
    arts = {eid: {"duration_s": 5.0} for eid in exp805._MILESTONE_EXPS}
    wt = exp805.compute_wall_time(arts)
    assert wt["improvement"] is True
    assert wt["total_wall_time_min"] == pytest.approx(1.0, abs=0.01)


def test_compute_wall_time_regression_when_slower() -> None:
    """improvement is False when total wall time exceeds previous milestone.

    Spec: REQ-METRICS-010
    """
    # 12 experiments × 30s each = 360s = 6.0 min > 2.2 min → regression
    arts = {eid: {"duration_s": 30.0} for eid in exp805._MILESTONE_EXPS}
    wt = exp805.compute_wall_time(arts)
    assert wt["improvement"] is False


def test_compute_wall_time_uses_elapsed_minutes_for_timed_out() -> None:
    """compute_wall_time uses elapsed_minutes * 60 for timed-out experiments.

    duration_s from watchdog artifacts is unreliable when timed out; elapsed_minutes
    is the authoritative field in that case.

    Spec: REQ-METRICS-010
    """
    arts = {eid: {"duration_s": 0.0} for eid in exp805._MILESTONE_EXPS}
    arts[799]["timed_out"] = True
    arts[799]["elapsed_minutes"] = 45.0
    wt = exp805.compute_wall_time(arts)
    # elapsed_minutes=45.0 → 45*60=2700s → 45.0 min contributed from 799
    assert wt["total_wall_time_min"] == pytest.approx(45.0, abs=0.01)


def test_compute_wall_time_empty_artifacts_gives_zero_mean() -> None:
    """compute_wall_time handles zero-ran experiments without division by zero.

    Spec: REQ-METRICS-010
    """
    arts: dict[int, dict] = {}
    wt = exp805.compute_wall_time(arts)
    assert wt["mean_min_per_experiment"] == 0.0


# ---------------------------------------------------------------------------
# rank_experiments_by_duration
# ---------------------------------------------------------------------------


def test_rank_experiments_slowest_first() -> None:
    """rank_experiments_by_duration returns experiments slowest-first.

    Spec: REQ-METRICS-010
    """
    arts = {
        793: {"duration_s": 1.0, "title": "fast"},
        799: {"duration_s": 300.0, "title": "slow"},
        800: {"duration_s": 10.0, "title": "mid"},
    }
    ranked = exp805.rank_experiments_by_duration(arts)
    assert ranked[0]["exp_id"] == 799
    assert ranked[-1]["exp_id"] == 793


def test_rank_experiments_includes_timed_out_duration() -> None:
    """rank_experiments_by_duration converts elapsed_minutes to seconds for ordering.

    Spec: REQ-METRICS-010
    """
    arts = {
        793: {"duration_s": 60.0},
        799: {"timed_out": True, "elapsed_minutes": 45.0, "duration_s": 0.0},
    }
    ranked = exp805.rank_experiments_by_duration(arts)
    assert ranked[0]["exp_id"] == 799  # 45*60=2700s > 60s


# ---------------------------------------------------------------------------
# classify_retros
# ---------------------------------------------------------------------------


def test_classify_retros_closes_retro028_when_verdict_matches() -> None:
    """classify_retros closes RETRO-028 when Exp 795 verdict is retro_028_closed.

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    retros = exp805.classify_retros(arts)
    assert any("RETRO-028" in r and "resolved" in r for r in retros["retros_closed"])
    assert not any("RETRO-028" in r for r in retros["retros_still_open"])


def test_classify_retros_keeps_retro028_open_on_partial_success() -> None:
    """classify_retros keeps RETRO-028 open when verdict is partial_success.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[795]["honest_verdict"] = "partial_success"
    retros = exp805.classify_retros(arts)
    assert any("RETRO-028" in r for r in retros["retros_still_open"])
    assert not any("RETRO-028" in r and "resolved" in r for r in retros["retros_closed"])


def test_classify_retros_closes_jepa_data_when_n_labeled_sufficient() -> None:
    """classify_retros partially closes RETRO-JEPA-V20-NO-DATA when n_labeled_total >= 80.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[797]["n_labeled_total"] = 300
    retros = exp805.classify_retros(arts)
    assert any("RETRO-JEPA-V20-NO-DATA" in r for r in retros["retros_closed"])


def test_classify_retros_opens_jepa_v21_ood_below_gate() -> None:
    """classify_retros opens RETRO-JEPA-V21-OOD-BELOW-GATE when ood_auc < 0.75.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[799]["ood_auc"] = 0.2444
    arts[799]["honest_verdict"] = "jepa_v21_below_gate"
    retros = exp805.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-JEPA-V21-OOD-BELOW-GATE" in new_ids


def test_classify_retros_does_not_open_jepa_v21_when_above_gate() -> None:
    """classify_retros does not open RETRO-JEPA-V21-OOD-BELOW-GATE when ood_auc >= 0.75.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[799]["ood_auc"] = 0.80
    retros = exp805.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-JEPA-V21-OOD-BELOW-GATE" not in new_ids


def test_classify_retros_opens_tier1_plateau_when_verdict_indicates() -> None:
    """classify_retros opens RETRO-TIER1-PLATEAU when Exp 802 verdict is tier1_plateau_persists.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[802]["honest_verdict"] = "tier1_plateau_persists"
    retros = exp805.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-TIER1-PLATEAU" in new_ids


def test_classify_retros_does_not_open_tier1_plateau_when_works() -> None:
    """classify_retros does not open RETRO-TIER1-PLATEAU when tier1 relay works.

    Spec: SCENARIO-RETRO-034
    """
    arts = _minimal_artifacts()
    arts[802]["honest_verdict"] = "tier1_relay_works"
    retros = exp805.classify_retros(arts)
    new_ids = [r["id"] for r in retros["retros_opened"]]
    assert "RETRO-TIER1-PLATEAU" not in new_ids


def test_classify_retros_closes_constraint_zero_delta_when_positive() -> None:
    """classify_retros closes RETRO-CONSTRAINT-ZERO-DELTA when delta > 0.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[801]["constraint_addition_delta_overall"] = 0.05
    retros = exp805.classify_retros(arts)
    assert any("RETRO-CONSTRAINT-ZERO-DELTA" in r for r in retros["retros_closed"])


def test_classify_retros_keeps_constraint_delta_open_when_zero() -> None:
    """classify_retros keeps RETRO-CONSTRAINT-ZERO-DELTA open when delta == 0.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[801]["constraint_addition_delta_overall"] = 0.0
    retros = exp805.classify_retros(arts)
    assert any("RETRO-CONSTRAINT-ZERO-DELTA" in r for r in retros["retros_still_open"])


def test_classify_retros_closes_kv260_tools_when_installed() -> None:
    """classify_retros closes RETRO-KV260-TOOLS-UNAVAILABLE when tools_installed=True.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[794]["tools_installed"] = True
    retros = exp805.classify_retros(arts)
    assert any("RETRO-KV260-TOOLS-UNAVAILABLE" in r for r in retros["retros_closed"])


def test_classify_retros_keeps_kv260_tools_open_when_missing() -> None:
    """classify_retros keeps RETRO-KV260-TOOLS-UNAVAILABLE open when not installed.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    arts[794]["tools_installed"] = False
    retros = exp805.classify_retros(arts)
    assert any("RETRO-KV260-TOOLS-UNAVAILABLE" in r for r in retros["retros_still_open"])


# ---------------------------------------------------------------------------
# build_honest_verdict
# ---------------------------------------------------------------------------


def test_build_honest_verdict_contains_criteria_counts() -> None:
    """build_honest_verdict embeds the N_of_M criteria count in the output string.

    Spec: REQ-METRICS-010
    """
    arts = _minimal_artifacts()
    criteria = exp805.evaluate_success_criteria(arts)
    wall_time = exp805.compute_wall_time(arts)
    retros = exp805.classify_retros(arts)
    verdict = exp805.build_honest_verdict(criteria, wall_time, retros)
    met = sum(1 for v in criteria.values() if v)
    assert f"{met}_of_10" in verdict


def test_build_honest_verdict_contains_improvement_or_regression() -> None:
    """build_honest_verdict contains 'IMPROVEMENT' or 'REGRESSION' based on wall time.

    Spec: REQ-METRICS-010
    """
    arts = {eid: {"duration_s": 1.0} for eid in exp805._MILESTONE_EXPS}
    criteria = exp805.evaluate_success_criteria(arts)
    wall_time = exp805.compute_wall_time(arts)
    retros = exp805.classify_retros(arts)
    verdict = exp805.build_honest_verdict(criteria, wall_time, retros)
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

    deliverable = tmp_path / "operational_retro_2026_04_61.json"
    result = exp805.run(deliverable=deliverable, results_dir=tmp_results)

    assert deliverable.exists(), "Deliverable file must be written to disk"
    on_disk = json.loads(deliverable.read_text())

    # Required schema fields
    for field in [
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
        assert field in on_disk, f"Missing required field: {field}"

    assert on_disk["status"] == "success"
    assert on_disk["criteria_total"] == 10
    assert on_disk["milestone"] == "2026.04.61"
    assert on_disk["schema"] == "carnot.operational_retro.v36"


def test_run_criteria_met_count_matches_all_pass_baseline(tmp_path: Path) -> None:
    """criteria_met_count equals 10 when all artifacts satisfy their thresholds.

    Spec: REQ-METRICS-010
    """
    tmp_results = tmp_path / "results"
    tmp_results.mkdir()
    _write_all(tmp_results, _minimal_artifacts())
    deliverable = tmp_path / "retro.json"
    result = exp805.run(deliverable=deliverable, results_dir=tmp_results)
    assert result["criteria_met_count"] == 10


def test_run_criteria_met_count_is_three_on_actual_artifacts() -> None:
    """criteria_met_count == 3 on the real .61 experiment results.

    The three passing criteria are jepa_v21_data_adequate, cpmi_augmentation_works,
    and embedding_retrieval_works.  All others failed per their honest_verdicts.

    This test reads the live results/ directory — it passes when the real
    deliverable was produced by running the script.

    Spec: REQ-METRICS-010, SCENARIO-RETRO-034
    """
    real_deliverable = REPO_ROOT / "results" / "operational_retro_2026_04_61.json"
    if not real_deliverable.exists():
        pytest.skip("Real deliverable not found — run experiment_805_milestone_retro.py first")
    artifact = json.loads(real_deliverable.read_text())
    assert artifact["criteria_met_count"] == 3
    assert artifact["criteria_total"] == 10


def test_run_slowest_experiment_is_exp_799(tmp_path: Path) -> None:
    """slowest_experiment.exp_id == 799 on the actual .61 artifact (325s).

    Spec: REQ-METRICS-010
    """
    real_deliverable = REPO_ROOT / "results" / "operational_retro_2026_04_61.json"
    if not real_deliverable.exists():
        pytest.skip("Real deliverable not found")
    artifact = json.loads(real_deliverable.read_text())
    assert artifact["slowest_experiment"]["exp_id"] == 799
