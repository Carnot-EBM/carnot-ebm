"""Tests for scripts/experiment_651_retro_2026_04_49.py — Milestone 2026.04.49 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: all v24 success criteria boolean branches, wall-time aggregation,
  closure rate, honest_verdict variants, open_retro_items carry-forward,
  top_priorities_for_50 (all three VR priority branches)
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.49
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_651_retro_2026_04_49 as retro_mod
from scripts.experiment_651_retro_2026_04_49 import (
    DELIVERABLE,
    EXP_ID,
    MILESTONE,
    SCHEMA,
    _MILESTONE_RESULTS,
    _RETROS_OPEN_AT_MILESTONE_START,
    _load_result,
    compute_retro,
)


# ---------------------------------------------------------------------------
# _load_result
# ---------------------------------------------------------------------------


def test_load_result_missing_file(tmp_path: Path) -> None:
    """A missing file returns an empty dict rather than raising an exception."""
    result = _load_result(str(tmp_path / "nonexistent.json"))
    assert result == {}


def test_load_result_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file with malformed JSON returns an empty dict."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    result = _load_result("bad.json")
    assert result == {}


def test_load_result_valid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid JSON file is loaded and returned as a dict."""
    data = {"status": "success", "duration_s": 10.5}
    good = tmp_path / "good.json"
    good.write_text(json.dumps(data))
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    result = _load_result("good.json")
    assert result == data


# ---------------------------------------------------------------------------
# Helper: write fake experiment results to tmp_path
# ---------------------------------------------------------------------------


def _write_fake_results(
    tmp_path: Path,
    *,
    conductor_consulted: bool = True,
    hermes_v2_recall: float = 0.0,
    causal_recall: float = 0.36,
    ensemble_recall: float = 0.36,
    retro_070_resolved: bool = True,
    gate_open: bool = True,
    signed_improvement: float = 0.0,
    retro_033_resolved: bool = False,
    fr11_real_violations_confirmed: bool = False,
    calibration_target_met: bool = True,
    ece_after: float = 0.023,
    otv_viable: bool = False,
    inertia_faster: bool = False,
    dualgpu_proven: bool = False,
    retro_071_resolved: bool = False,
    retro_057_resolved: bool = False,
    multilevel_sparse_vs_dense_error: float = 13.01,
) -> None:
    """Write minimal fake JSON files for each upstream experiment in _MILESTONE_RESULTS."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    # exp640 — Pre-Flight Infra v2
    (results_dir / "experiment_640_preflght_infra.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.035,
            "conductor_consulted": conductor_consulted,
            "manifest_wired": conductor_consulted,
        })
    )
    # exp641 — HERMES v2 Live Loop
    (results_dir / "experiment_641_hermes_v2_live.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 10.797,
            "hermes_v2_recall": hermes_v2_recall,
        })
    )
    # exp642 — Causal Verifier
    (results_dir / "experiment_642_causal_verifier.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.005,
            "causal_recall": causal_recall,
        })
    )
    # exp643 — Ensemble Gate v2
    (results_dir / "experiment_643_ensemble_gate_v2.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.036,
            "ensemble_recall": ensemble_recall,
            "retro_070_resolved": retro_070_resolved,
            "gate_open": gate_open,
        })
    )
    # exp644 — Live VR Attempt #17
    (results_dir / "experiment_644_live_vr_attempt_17.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 10.175,
            "signed_improvement": signed_improvement,
            "retro_033_resolved": retro_033_resolved,
        })
    )
    # exp645 — Tier1 FR-11 Relay
    (results_dir / "experiment_645_tier1_fr11_relay.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.001,
            "fr11_real_violations_confirmed": fr11_real_violations_confirmed,
        })
    )
    # exp646 — JEPA v14 Platt Scaling
    (results_dir / "experiment_646_jepa_v14_platt.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 51.633,
            "calibration_target_met": calibration_target_met,
            "ece_after": ece_after,
        })
    )
    # exp647 — OTV Verifier
    (results_dir / "experiment_647_otv_verifier.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 5.406,
            "otv_viable": otv_viable,
        })
    )
    # exp648 — Parallel Ising Inertia
    (results_dir / "experiment_648_parallel_ising_inertia.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 15.813,
            "inertia_faster": inertia_faster,
        })
    )
    # exp649 — DualGPU 13B Proof v2
    (results_dir / "experiment_649_dualgpu_13b_v2.json").write_text(
        json.dumps({
            "status": "blocked" if not dualgpu_proven else "success",
            "duration_s": 0.053,
            "dualgpu_proven": dualgpu_proven,
            "retro_071_resolved": retro_071_resolved,
        })
    )
    # exp650 — KAEM Multilevel Sparse
    (results_dir / "experiment_650_kaem_multilevel_sparse.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 274.194,
            "retro_057_resolved": retro_057_resolved,
            "multilevel_sparse_vs_dense_error": multilevel_sparse_vs_dense_error,
        })
    )


# ---------------------------------------------------------------------------
# compute_retro — primary success criteria
# ---------------------------------------------------------------------------


def test_retro_033_not_resolved_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is False when Exp 644 signed_improvement=0.0."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is False


def test_retro_033_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is True when Exp 644 sets retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.05)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is True


def test_retro_070_resolved_when_ensemble_recall_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_070_resolved is True when ensemble_recall >= 0.30."""
    _write_fake_results(tmp_path, ensemble_recall=0.36, retro_070_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_070_resolved"] is True
    assert retro["ensemble_recall"] == pytest.approx(0.36, abs=1e-6)


def test_retro_070_not_resolved_when_recall_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_070_resolved is False when ensemble_recall < 0.30."""
    _write_fake_results(tmp_path, ensemble_recall=0.12, retro_070_resolved=False, gate_open=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_070_resolved"] is False


def test_retro_071_not_resolved_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_071_resolved is False when Exp 649 dualgpu_proven=False."""
    _write_fake_results(tmp_path, dualgpu_proven=False, retro_071_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_071_resolved"] is False
    assert retro["dualgpu_proven"] is False


def test_retro_071_resolved_when_proven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_071_resolved is True when Exp 649 sets retro_071_resolved=True."""
    _write_fake_results(tmp_path, dualgpu_proven=True, retro_071_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_071_resolved"] is True
    assert retro["dualgpu_proven"] is True


def test_retro_057_not_resolved_when_error_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_057_resolved is False when multilevel_sparse_vs_dense_error >> 5%."""
    _write_fake_results(tmp_path, retro_057_resolved=False, multilevel_sparse_vs_dense_error=13.01)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_057_resolved"] is False


def test_retro_057_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_057_resolved is True when Exp 650 sets retro_057_resolved=True."""
    _write_fake_results(tmp_path, retro_057_resolved=True, multilevel_sparse_vs_dense_error=0.03)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_057_resolved"] is True


def test_jepa_v14_calibrated_when_target_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jepa_v14_calibrated is True when Exp 646 calibration_target_met=True."""
    _write_fake_results(tmp_path, calibration_target_met=True, ece_after=0.023)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["jepa_v14_calibrated"] is True


def test_jepa_v14_not_calibrated_when_ece_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jepa_v14_calibrated is False when calibration_target_met=False."""
    _write_fake_results(tmp_path, calibration_target_met=False, ece_after=0.191)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["jepa_v14_calibrated"] is False


def test_otv_viable_false_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """otv_viable is False when Exp 647 otv_viable=False."""
    _write_fake_results(tmp_path, otv_viable=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["otv_viable"] is False


def test_otv_viable_true_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """otv_viable is True when Exp 647 otv_viable=True."""
    _write_fake_results(tmp_path, otv_viable=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["otv_viable"] is True


def test_manifest_wired_true_when_conductor_consulted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """manifest_wired is True when Exp 640 conductor_consulted=True."""
    _write_fake_results(tmp_path, conductor_consulted=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["manifest_wired"] is True


def test_fr11_not_confirmed_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is False when Exp 645 fr11_real_violations_confirmed=False."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is False


def test_fr11_confirmed_when_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is True when Exp 645 fr11_real_violations_confirmed=True."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is True


def test_inertia_faster_false_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inertia_faster is False when Exp 648 inertia_faster=False."""
    _write_fake_results(tmp_path, inertia_faster=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["inertia_faster"] is False


def test_hermes_v2_recall_from_exp641(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hermes_v2_recall is read from Exp 641."""
    _write_fake_results(tmp_path, hermes_v2_recall=0.0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["hermes_v2_recall"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_retro — honest_verdict logic
# ---------------------------------------------------------------------------


def test_honest_verdict_vr_17_succeeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='vr_17_succeeded_033_closed' when retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "vr_17_succeeded_033_closed"


def test_honest_verdict_retro_070_closed_jepa_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='retro_070_closed_jepa_calibrated_vr17_blocked' when both close but 033 open."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=True,
        calibration_target_met=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "retro_070_closed_jepa_calibrated_vr17_blocked"


def test_honest_verdict_retro_070_closed_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='retro_070_closed_vr17_blocked' when only 070 resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=True,
        calibration_target_met=False,
        ece_after=0.191,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "retro_070_closed_vr17_blocked"


def test_honest_verdict_jepa_calibrated_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='jepa_calibrated_all_vr_retros_carry' when only JEPA closed."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=False,
        gate_open=False,
        ensemble_recall=0.12,
        calibration_target_met=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "jepa_calibrated_all_vr_retros_carry"


def test_honest_verdict_no_retros_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='no_retros_closed' when nothing resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=False,
        gate_open=False,
        ensemble_recall=0.12,
        calibration_target_met=False,
        ece_after=0.191,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "no_retros_closed"


# ---------------------------------------------------------------------------
# compute_retro — retro_closure_rate and open_retro_count
# ---------------------------------------------------------------------------


def test_closure_rate_when_two_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 2/11 when RETRO-070 and RETRO-060 both closed."""
    _write_fake_results(tmp_path, retro_070_resolved=True, calibration_target_met=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = round(2 / len(_RETROS_OPEN_AT_MILESTONE_START), 3)
    assert retro["retro_closure_rate"] == pytest.approx(expected, abs=1e-6)


def test_closure_rate_zero_when_nothing_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 0 when no RETROs are resolved this milestone."""
    _write_fake_results(
        tmp_path,
        retro_070_resolved=False,
        gate_open=False,
        ensemble_recall=0.12,
        calibration_target_met=False,
        ece_after=0.191,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_closure_rate"] == 0.0


def test_open_retro_count_decreases_when_retros_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_retro_count = 9 when 2 of 11 RETROs are closed (070 + 060)."""
    _write_fake_results(tmp_path, retro_070_resolved=True, calibration_target_met=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["open_retro_count"] == len(_RETROS_OPEN_AT_MILESTONE_START) - 2


def test_new_retro_items_empty_for_milestone_49(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """new_retro_items is empty for .49."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["new_retro_items"] == []


def test_open_retro_items_contains_retro_033_when_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-033 appears in open_retro_items when not resolved."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-033" in ids


def test_retro_033_absent_from_open_items_when_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-033 is NOT in open_retro_items when retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.05)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-033" not in ids


def test_open_retro_items_contains_retro_071_when_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-071 appears in open_retro_items when not resolved."""
    _write_fake_results(tmp_path, dualgpu_proven=False, retro_071_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-071" in ids


def test_retro_071_absent_from_open_items_when_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-071 is NOT in open_retro_items when retro_071_resolved=True."""
    _write_fake_results(tmp_path, dualgpu_proven=True, retro_071_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-071" not in ids


def test_open_retro_items_contains_retro_070_when_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-070 appears in open_retro_items when not resolved."""
    _write_fake_results(
        tmp_path, retro_070_resolved=False, gate_open=False, ensemble_recall=0.12
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-070" in ids


def test_retro_070_absent_from_open_items_when_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-070 is NOT in open_retro_items when retro_070_resolved=True."""
    _write_fake_results(tmp_path, retro_070_resolved=True, ensemble_recall=0.36)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-070" not in ids


def test_open_retro_items_contains_retro_057_when_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-057 appears in open_retro_items when not resolved."""
    _write_fake_results(tmp_path, retro_057_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-057" in ids


def test_retro_060_absent_when_jepa_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-060 is NOT in open_retro_items when jepa_v14_calibrated=True."""
    _write_fake_results(tmp_path, calibration_target_met=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-060" not in ids


def test_retro_060_present_when_jepa_not_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-060 appears in open_retro_items when jepa_v14_calibrated=False."""
    _write_fake_results(tmp_path, calibration_target_met=False, ece_after=0.191)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-060" in ids


# ---------------------------------------------------------------------------
# compute_retro — top_priorities_for_50 branches
# ---------------------------------------------------------------------------


def test_top_priorities_for_50_has_three_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """top_priorities_for_50 always contains exactly 3 entries."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert len(retro["top_priorities_for_50"]) == 3


def test_top_priority_scale_200q_when_033_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_033 is resolved, top priority is scaling to 200q Wilson CI."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_50"][0]
    assert "200q" in first or "Wilson" in first


def test_top_priority_run_vr17_when_070_resolved_033_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_070 is resolved but retro_033 is not, top priority is 'Run VR #17 immediately.'"""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=True,
        ensemble_recall=0.36,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_50"][0]
    assert "#17" in first or "VR" in first


def test_top_priority_structured_format_when_both_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both retro_033 and retro_070 are not resolved, top priority is structured-format."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=False,
        gate_open=False,
        ensemble_recall=0.12,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_50"][0]
    assert "structured" in first.lower() or "format" in first.lower() or "equation" in first.lower()


def test_top_priority_fpga_always_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FPGA Vivado is always the second priority."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    second = retro["top_priorities_for_50"][1]
    assert "FPGA" in second or "Vivado" in second or "KV260" in second or "TCL" in second


def test_top_priority_jepa_cascade_when_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Third priority is 'Deploy JEPA v14 + Platt in cascade' when calibrated."""
    _write_fake_results(tmp_path, calibration_target_met=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    third = retro["top_priorities_for_50"][2]
    assert "JEPA" in third or "Platt" in third or "cascade" in third


def test_top_priority_isotonic_when_not_calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Third priority is isotonic regression when jepa not calibrated."""
    _write_fake_results(tmp_path, calibration_target_met=False, ece_after=0.191)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    third = retro["top_priorities_for_50"][2]
    assert "isotonic" in third.lower() or "regression" in third.lower()


# ---------------------------------------------------------------------------
# compute_retro — wall time and experiment counts
# ---------------------------------------------------------------------------


def test_n_experiments_is_twelve_when_all_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_experiments_run = 12 when all 11 upstream files exist plus this retro."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_experiments_run"] == 12


def test_n_not_run_zero_when_all_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_not_run = 0 when all upstream result files are present."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_not_run"] == 0


def test_n_not_run_increments_on_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_not_run >= 1 when one upstream result file is missing."""
    _write_fake_results(tmp_path)
    (tmp_path / "results" / "experiment_650_kaem_multilevel_sparse.json").unlink()
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_not_run"] >= 1


def test_mean_time_min_is_wall_over_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mean_time_min = total_wall_time_minutes / n_experiments_run."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = round(retro["total_wall_time_minutes"] / retro["n_experiments_run"], 3)
    assert retro["mean_time_min"] == pytest.approx(expected, abs=1e-6)


def test_schema_and_milestone_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema and milestone fields are set to the expected v24 values."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["schema"] == SCHEMA
    assert retro["milestone"] == MILESTONE


# ---------------------------------------------------------------------------
# main() integration — deliverable written with required fields
# ---------------------------------------------------------------------------


def test_main_writes_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes a valid JSON artifact at DELIVERABLE path with all required fields."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(exist_ok=True)

    retro_mod.main()

    deliverable = tmp_path / DELIVERABLE
    assert deliverable.exists(), f"Deliverable not written: {DELIVERABLE}"
    artifact = json.loads(deliverable.read_text())

    required_fields = [
        "experiment", "title", "run_date", "started_at", "finished_at",
        "duration_s", "status", "schema",
        "retro_033_resolved",
        "retro_057_resolved",
        "retro_070_resolved",
        "retro_071_resolved",
        "jepa_v14_calibrated",
        "otv_viable",
        "manifest_wired",
        "fr11_confirmed",
        "inertia_faster",
        "dualgpu_proven",
        "hermes_v2_recall",
        "ensemble_recall",
        "retro_closure_rate",
        "open_retro_count",
        "new_retro_items",
        "open_retro_items",
        "top_priorities_for_50",
        "honest_verdict",
        "env_autofix",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["schema"] == SCHEMA
    assert artifact["env_autofix"] is True
    assert artifact["honest_verdict"] in {
        "vr_17_succeeded_033_closed",
        "retro_070_closed_jepa_calibrated_vr17_blocked",
        "retro_070_closed_vr17_blocked",
        "jepa_calibrated_all_vr_retros_carry",
        "no_retros_closed",
    }
    assert artifact["experiment"] == EXP_ID
    assert artifact["milestone"] == MILESTONE
