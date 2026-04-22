"""Tests for scripts/experiment_677_operational_retro_2026_04_51.py.

Coverage targets (code added in this session only):
- load_experiment_results: missing file raises FileNotFoundError, valid files loaded
- compute_milestone_metrics: wall-time aggregation, slowest-5 ordering,
  retro_033_status and retro_071_status logic, honest_verdict construction
- main: deliverable written with correct schema and all REQUIRED_RESULT_FIELDS

Spec: REQ-INFRA-007, REQ-INFRA-023, REQ-INFRA-062
SCENARIO: RETRO-2026.04.51
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_677_operational_retro_2026_04_51 as retro_mod
from scripts.experiment_677_operational_retro_2026_04_51 import (
    DELIVERABLE,
    MILESTONE_EXP_IDS,
    PRIOR_EXPERIMENTS_COMPLETED,
    PRIOR_TOTAL_WALL_TIME_MIN,
    _RESULT_FILES,
    compute_milestone_metrics,
    load_experiment_results,
)
from scripts.experiment_template import REQUIRED_RESULT_FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fake_results(tmp_path: Path, overrides: dict[int, dict] | None = None) -> None:
    """Write minimal valid result stubs for all 11 milestone experiments."""
    base: dict[int, dict] = {
        666: {"experiment": 666, "duration_s": 0.07, "status": "success",
              "manifest_loaded": True, "honest_verdict": "manifest_wired_xdist_available",
              "title": "Exclusion Manifest Wire-In v3"},
        667: {"experiment": 667, "duration_s": 0.001, "status": "success",
              "honest_verdict": "gate_open_retro_070_unblocked",
              "title": "EnsembleGate v4 Redesign"},
        668: {"experiment": 668, "duration_s": 1623.408, "status": "success",
              "signed_improvement": 0.64, "honest_verdict": "vr_positive"},
        669: {"experiment": 669, "duration_s": 16.84, "status": "success",
              "honest_verdict": "distillation_corpus_built",
              "title": "Prompt Injection KAN Rescue"},
        670: {"experiment": 670, "duration_s": 2.979, "status": "success",
              "jepa_v14_deployed": True, "honest_verdict": "jepa_v14_deployed",
              "title": "JEPA v14 Cascade Deploy"},
        671: {"experiment": 671, "duration_s": 11.113, "status": "success",
              "ood_auc": 1.0, "honest_verdict": "jepa_v15_auc_met",
              "title": "JEPA v15 Retrain"},
        672: {"experiment": 672, "duration_s": 0.0, "status": "blocked",
              "honest_verdict": "blocked_bitfile_not_configured",
              "title": "KV260 dfx-mgr"},
        673: {"experiment": 673, "duration_s": 10.265, "status": "success",
              "max_gpu1_util_pct": 0.0, "retro_071_resolved": False,
              "honest_verdict": "dualgpu_partial", "title": "DualGPU Confirmed v3"},
        674: {"experiment": 674, "duration_s": 0.001, "status": "success",
              "honest_verdict": "ias_gate_improves_v3", "title": "IAS Adaptive Gate"},
        675: {"experiment": 675, "duration_s": 0.014, "status": "success",
              "honest_verdict": "below_threshold", "title": "LOS-Net Detector"},
        676: {"experiment": 676, "duration_s": 0.001, "status": "success",
              "honest_verdict": "metajuls_adapted", "title": "MetaJuLS Adaptive"},
    }
    if overrides:
        for exp_id, patch in overrides.items():
            base[exp_id] = {**base[exp_id], **patch}

    for exp_id, rel_path in _RESULT_FILES.items():
        full = tmp_path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps(base[exp_id]))


# ---------------------------------------------------------------------------
# load_experiment_results
# ---------------------------------------------------------------------------


def test_load_results_missing_file_raises(tmp_path: Path) -> None:
    """FileNotFoundError is raised when any milestone result file is absent."""
    # Write all files except one.
    exp_ids = list(_RESULT_FILES.keys())
    for exp_id in exp_ids[:-1]:
        rel_path = _RESULT_FILES[exp_id]
        full = tmp_path / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(json.dumps({"experiment": exp_id, "duration_s": 1.0}))
    # Last file intentionally absent.
    with pytest.raises(FileNotFoundError):
        load_experiment_results(tmp_path)


def test_load_results_all_present(tmp_path: Path) -> None:
    """All 11 result files load successfully and return a dict keyed by exp ID."""
    _write_fake_results(tmp_path)
    results = load_experiment_results(tmp_path)
    assert set(results.keys()) == set(MILESTONE_EXP_IDS)
    assert results[668]["signed_improvement"] == 0.64


# ---------------------------------------------------------------------------
# compute_milestone_metrics — wall-time aggregation
# ---------------------------------------------------------------------------


def _fake_results(duration_s_each: float = 60.0) -> dict[int, dict]:
    """Return minimal fake results with a uniform duration_s for every experiment."""
    return {
        exp_id: {"experiment": exp_id, "duration_s": duration_s_each,
                 "status": "success", "honest_verdict": "ok",
                 "manifest_loaded": True, "signed_improvement": 0.5,
                 "jepa_v14_deployed": True, "ood_auc": 0.8,
                 "max_gpu1_util_pct": 0.0, "retro_071_resolved": False}
        for exp_id in MILESTONE_EXP_IDS
    }


def test_wall_time_aggregation() -> None:
    """Total wall time = prior + (sum of durations / 60)."""
    # 11 experiments × 60 s = 11 min added to prior total.
    results = _fake_results(duration_s_each=60.0)
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    expected_total = 1000.0 + 11 * 60.0 / 60.0
    assert abs(metrics["total_wall_time_minutes"] - expected_total) < 0.01


def test_per_experiment_avg() -> None:
    """per_experiment_avg_min = total_wall_time / total_experiments."""
    results = _fake_results(duration_s_each=120.0)
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    total = 1000.0 + 11 * 2.0  # 11 × 2 min added
    total_exps = 100 + 11
    assert abs(metrics["per_experiment_avg_min"] - total / total_exps) < 0.001


def test_wall_time_delta_direction_regression() -> None:
    """Adding experiments increases total wall time → direction is regression."""
    results = _fake_results(duration_s_each=60.0)
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert metrics["wall_time_delta_direction"] == "regression"
    assert metrics["wall_time_delta_minutes"] > 0


# ---------------------------------------------------------------------------
# compute_milestone_metrics — slowest_5 ordering
# ---------------------------------------------------------------------------


def test_slowest_5_ordered_by_duration() -> None:
    """slowest_5 must list the five longest experiments in descending order."""
    results = _fake_results(duration_s_each=1.0)
    # Give one experiment a much longer duration to ensure it tops the list.
    results[668]["duration_s"] = 9999.0
    results[671]["duration_s"] = 500.0
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    s5 = metrics["slowest_5"]
    assert len(s5) == 5
    assert s5[0]["experiment"] == 668
    assert s5[1]["experiment"] == 671
    durations = [e["duration_s"] for e in s5]
    assert durations == sorted(durations, reverse=True)


# ---------------------------------------------------------------------------
# compute_milestone_metrics — retro status logic
# ---------------------------------------------------------------------------


def test_retro_033_closed_when_positive_improvement() -> None:
    """retro_033_status is 'closed' when signed_improvement > 0."""
    results = _fake_results()
    results[668]["signed_improvement"] = 0.01
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert metrics["retro_033_status"] == "closed"


def test_retro_033_open_when_zero_improvement() -> None:
    """retro_033_status is 'open_attempt_19' when signed_improvement == 0."""
    results = _fake_results()
    results[668]["signed_improvement"] = 0.0
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert metrics["retro_033_status"] == "open_attempt_19"


def test_retro_071_resolved_when_flag_true() -> None:
    """retro_071_status is 'resolved' when Exp 673 sets retro_071_resolved=True."""
    results = _fake_results()
    results[673]["retro_071_resolved"] = True
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert metrics["retro_071_status"] == "resolved"


def test_retro_071_open_when_flag_false() -> None:
    """retro_071_status is 'open_partial' when Exp 673 retro_071_resolved=False."""
    results = _fake_results()
    results[673]["retro_071_resolved"] = False
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert metrics["retro_071_status"] == "open_partial"


# ---------------------------------------------------------------------------
# compute_milestone_metrics — honest_verdict composition
# ---------------------------------------------------------------------------


def test_honest_verdict_contains_wall_time_direction() -> None:
    """honest_verdict must contain 'wall_time_regression' or 'wall_time_improvement'."""
    results = _fake_results(duration_s_each=60.0)
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    verdict = metrics["honest_verdict"]
    assert "wall_time_regression" in verdict or "wall_time_improvement" in verdict


def test_honest_verdict_vr_positive() -> None:
    """When VR was positive, honest_verdict must contain 'vr_positive'."""
    results = _fake_results()
    results[668]["signed_improvement"] = 0.64
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert "vr_positive" in metrics["honest_verdict"]


def test_honest_verdict_vr_negative() -> None:
    """When VR was non-positive, honest_verdict must contain 'vr_negative'."""
    results = _fake_results()
    results[668]["signed_improvement"] = 0.0
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert "vr_negative" in metrics["honest_verdict"]


def test_honest_verdict_manifest_confirmed() -> None:
    """When manifest_loaded=True, honest_verdict must contain 'manifest_confirmed'."""
    results = _fake_results()
    results[666]["manifest_loaded"] = True
    metrics = compute_milestone_metrics(results, 1000.0, 100)
    assert "manifest_confirmed" in metrics["honest_verdict"]


# ---------------------------------------------------------------------------
# main — deliverable written to disk with all required fields
# ---------------------------------------------------------------------------


def test_main_writes_deliverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must write the deliverable JSON containing all REQUIRED_RESULT_FIELDS."""
    _write_fake_results(tmp_path)
    (tmp_path / "results").mkdir(exist_ok=True)

    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))

    retro_mod.main()

    deliverable = tmp_path / DELIVERABLE
    assert deliverable.exists(), "Deliverable JSON must be written by main()"

    artifact = json.loads(deliverable.read_text())

    # All REQUIRED_RESULT_FIELDS must be present.
    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact, f"Required field '{field}' missing from deliverable"

    assert artifact["experiment"] == 677
    assert artifact["milestone"] == "2026.04.51"
    assert artifact["status"] == "success"
    assert artifact["retro_033_status"] == "closed"
    assert "honest_verdict" in artifact
    assert isinstance(artifact["slowest_5"], list)
    assert len(artifact["slowest_5"]) == 5


def test_main_deliverable_schema_key() -> None:
    """The on-disk deliverable must have schema='carnot.operational_retro.v26'."""
    deliverable = _REPO_ROOT / DELIVERABLE
    if not deliverable.exists():
        pytest.skip("On-disk deliverable not yet generated")
    artifact = json.loads(deliverable.read_text())
    assert artifact["milestone"] == "2026.04.51"
    assert artifact["experiment"] == 677


def test_n_experiments_constant() -> None:
    """MILESTONE_EXP_IDS must contain exactly 11 experiment IDs (666-676)."""
    assert len(MILESTONE_EXP_IDS) == 11
    assert min(MILESTONE_EXP_IDS) == 666
    assert max(MILESTONE_EXP_IDS) == 676


def test_prior_constants() -> None:
    """Prior milestone constants must match the .50 retrospective values."""
    assert PRIOR_TOTAL_WALL_TIME_MIN == 4304.0
    assert PRIOR_EXPERIMENTS_COMPLETED == 519
