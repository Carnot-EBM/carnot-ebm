"""Tests for scripts/experiment_639_retro_2026_04_48.py — Milestone 2026.04.48 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: all v23 success criteria boolean branches, wall-time aggregation,
  closure rate, honest_verdict variants, open_retro_items carry-forward,
  top_priorities_for_49 (all three VR priority branches), multilevel_wins fallback key
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.48
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_639_retro_2026_04_48 as retro_mod
from scripts.experiment_639_retro_2026_04_48 import (
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
    data = {"status": "success", "duration_s": 42.0}
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
    interwhen_recall: float = 0.12,
    gate_open: bool = False,
    retro_070_resolved: bool = False,
    retro_033_resolved: bool = False,
    signed_improvement: float = 0.0,
    calibration_improved: bool = False,
    v14_ece: float = 0.132,
    v14_ood_auc: float = 0.912,
    dualgpu_proven: bool = False,
    retro_071_resolved: bool = False,
    hermes_improvement: bool = True,
    hermes_recall: float = 0.12,
    hermes_fp_rate: float = 0.2,
    multilevel_wins: bool = False,
    multilevel_faster: bool = False,
    adaptrack_improves_recall: bool = False,
    adaptrack_recall: float = 0.08,
    tcl_v2_written: str | None = "hardware/kv260/synth_ising_v2.tcl",
    retro_057_resolved: bool = False,
    sparse_vs_dense_error: float = 0.429,
    fr11_real_violations_confirmed: bool = False,
) -> None:
    """Write minimal fake JSON files for each upstream experiment in _MILESTONE_RESULTS."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    # exp627 — InterWhenMonitor
    (results_dir / "experiment_627_interwhen_monitor.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.026,
            "interwhen_recall": interwhen_recall,
            "gate_open": gate_open,
        })
    )
    # exp628 — ORACLE FOVER v5
    (results_dir / "experiment_628_oracle_fover_v5.json").write_text(
        json.dumps({"status": "success", "duration_s": 0.006, "corpus_ready": True})
    )
    # exp629 — InterwhenDiagnostic Gate
    (results_dir / "experiment_629_interwhen_diagnostic.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.209,
            "interwhen_recall_primary": interwhen_recall,
            "gate_open": gate_open,
            "retro_070_resolved": retro_070_resolved,
        })
    )
    # exp630 — Live VR #16
    (results_dir / "experiment_630_live_vr_attempt_16.json").write_text(
        json.dumps({
            "status": "blocked" if not retro_033_resolved else "success",
            "duration_s": 0.0,
            "gate_open": retro_033_resolved,
            "signed_improvement": signed_improvement,
            "retro_033_resolved": retro_033_resolved,
        })
    )
    # exp631 — JEPA v14 Oracle
    (results_dir / "experiment_631_jepa_v14_oracle.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 179.237,
            "calibration_improved": calibration_improved,
            "v14_ece": v14_ece,
            "v14_ood_auc": v14_ood_auc,
        })
    )
    # exp632 — DualGPU 13B Proof
    (results_dir / "experiment_632_dualgpu_13b_proof.json").write_text(
        json.dumps({
            "status": "blocked" if not dualgpu_proven else "success",
            "duration_s": 255.0,
            "dualgpu_proven": dualgpu_proven,
            "retro_071_resolved": retro_071_resolved,
        })
    )
    # exp633 — HERMES Adapter
    (results_dir / "experiment_633_hermes_adapter.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.029,
            "hermes_improvement": hermes_improvement,
            "hermes_recall": hermes_recall,
            "hermes_fp_rate": hermes_fp_rate,
        })
    )
    # exp634 — Multilevel KAN KAEMEnergy
    exp634_data: dict = {
        "status": "success",
        "duration_s": 396.031,
        "multilevel_faster": multilevel_faster,
    }
    if multilevel_wins:
        exp634_data["multilevel_wins"] = multilevel_wins
    (results_dir / "experiment_634_multilevel_kan_kaem.json").write_text(
        json.dumps(exp634_data)
    )
    # exp635 — AdapTrack Backtrack
    (results_dir / "experiment_635_adaptrack_backtrack.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.022,
            "adaptrack_improves_recall": adaptrack_improves_recall,
            "adaptrack_recall": adaptrack_recall,
        })
    )
    # exp636 — FPGA TCL v2
    exp636_data: dict = {
        "status": "success",
        "duration_s": 8.794,
    }
    if tcl_v2_written is not None:
        exp636_data["tcl_v2_written"] = tcl_v2_written
    (results_dir / "experiment_636_fpga_tcl_v2.json").write_text(
        json.dumps(exp636_data)
    )
    # exp637 — LowRankKAEM Sparse
    (results_dir / "experiment_637_lowrank_kaem_sparse.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 267.365,
            "retro_057_resolved": retro_057_resolved,
            "sparse_vs_dense_error": sparse_vs_dense_error,
        })
    )
    # exp638 — FR-11 Self-Learning Relay
    (results_dir / "experiment_638_tier1_fr11_relay.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.004,
            "fr11_real_violations_confirmed": fr11_real_violations_confirmed,
        })
    )


# ---------------------------------------------------------------------------
# compute_retro — primary success criteria
# ---------------------------------------------------------------------------


def test_retro_033_not_resolved_when_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is False when Exp 630 is blocked."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is False


def test_retro_033_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is True when Exp 630 sets retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.05)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is True


def test_retro_070_not_resolved_when_recall_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_070_resolved is False when Exp 629 interwhen_recall_primary=0.12."""
    _write_fake_results(tmp_path, interwhen_recall=0.12, retro_070_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_070_resolved"] is False
    assert retro["interwhen_recall"] == pytest.approx(0.12, abs=1e-6)


def test_retro_070_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_070_resolved is True when Exp 629 sets retro_070_resolved=True."""
    _write_fake_results(tmp_path, interwhen_recall=0.25, retro_070_resolved=True, gate_open=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_070_resolved"] is True


def test_retro_071_not_resolved_when_model_load_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_071_resolved is False when Exp 632 dualgpu_proven=False."""
    _write_fake_results(tmp_path, dualgpu_proven=False, retro_071_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_071_resolved"] is False
    assert retro["dualgpu_proven"] is False


def test_retro_071_resolved_when_dualgpu_proven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_071_resolved is True when Exp 632 sets retro_071_resolved=True."""
    _write_fake_results(tmp_path, dualgpu_proven=True, retro_071_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_071_resolved"] is True


def test_retro_057_not_resolved_when_error_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_057_resolved is False when sparse_vs_dense_error=0.429."""
    _write_fake_results(tmp_path, sparse_vs_dense_error=0.429, retro_057_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_057_resolved"] is False
    assert retro["sparse_vs_dense_error"] == pytest.approx(0.429, abs=1e-6)


def test_retro_057_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_057_resolved is True when Exp 637 sets retro_057_resolved=True."""
    _write_fake_results(tmp_path, sparse_vs_dense_error=0.04, retro_057_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_057_resolved"] is True


def test_jepa_v14_not_calibrated_when_ece_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jepa_v14_calibrated is False when calibration_improved=False (v14_ece=0.132)."""
    _write_fake_results(tmp_path, calibration_improved=False, v14_ece=0.132)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["jepa_v14_calibrated"] is False
    assert retro["v14_ece"] == pytest.approx(0.132, abs=1e-4)


def test_jepa_v14_calibrated_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jepa_v14_calibrated is True when Exp 631 sets calibration_improved=True."""
    _write_fake_results(tmp_path, calibration_improved=True, v14_ece=0.08)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["jepa_v14_calibrated"] is True


def test_fr11_not_confirmed_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is False when Exp 638 fr11_real_violations_confirmed=False."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is False


def test_fr11_confirmed_when_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is True when Exp 638 fr11_real_violations_confirmed=True."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is True


def test_hermes_improves_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hermes_improves is True when Exp 633 hermes_improvement=True."""
    _write_fake_results(tmp_path, hermes_improvement=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["hermes_improves"] is True


def test_hermes_improves_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hermes_improves is False when Exp 633 hermes_improvement=False."""
    _write_fake_results(tmp_path, hermes_improvement=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["hermes_improves"] is False


def test_multilevel_wins_false_via_multilevel_faster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """multilevel_wins uses multilevel_faster as fallback when multilevel_wins key is absent."""
    _write_fake_results(tmp_path, multilevel_faster=False)  # no multilevel_wins key
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["multilevel_wins"] is False


def test_multilevel_wins_true_via_explicit_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """multilevel_wins is True when the experiment sets multilevel_wins=True explicitly."""
    _write_fake_results(tmp_path, multilevel_wins=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["multilevel_wins"] is True


def test_adaptrack_improves_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """adaptrack_improves is False when Exp 635 adaptrack_improves_recall=False."""
    _write_fake_results(tmp_path, adaptrack_improves_recall=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["adaptrack_improves"] is False


def test_adaptrack_improves_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """adaptrack_improves is True when Exp 635 adaptrack_improves_recall=True."""
    _write_fake_results(tmp_path, adaptrack_improves_recall=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["adaptrack_improves"] is True


def test_fpga_tcl_updated_true_when_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_tcl_updated is True when Exp 636 tcl_v2_written is not None."""
    _write_fake_results(tmp_path, tcl_v2_written="hardware/kv260/synth_ising_v2.tcl")
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_tcl_updated"] is True


def test_fpga_tcl_updated_false_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_tcl_updated is False when Exp 636 does not set tcl_v2_written."""
    _write_fake_results(tmp_path, tcl_v2_written=None)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_tcl_updated"] is False


# ---------------------------------------------------------------------------
# compute_retro — honest_verdict logic
# ---------------------------------------------------------------------------


def test_honest_verdict_first_positive_vr_achieved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='first_positive_vr_achieved' when retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "first_positive_vr_achieved"


def test_honest_verdict_hermes_improved_gate_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='hermes_improved_gate_open_vr16_ready' when hermes improves and gate open."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        hermes_improvement=True,
        retro_070_resolved=True,
        gate_open=True,
        interwhen_recall=0.25,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "hermes_improved_gate_open_vr16_ready"


def test_honest_verdict_hermes_improved_all_retros_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='hermes_improved_all_retros_carry' when hermes improves but gate still closed."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        hermes_improvement=True,
        retro_070_resolved=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "hermes_improved_all_retros_carry"


def test_honest_verdict_no_retros_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='no_retros_closed' when nothing is resolved and hermes doesn't improve."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=False,
        hermes_improvement=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "no_retros_closed"


# ---------------------------------------------------------------------------
# compute_retro — retro_closure_rate and open_retro_count
# ---------------------------------------------------------------------------


def test_closure_rate_zero_when_nothing_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 0 when no RETROs are resolved this milestone."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_closure_rate"] == 0.0


def test_open_retro_count_eleven_when_nothing_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_retro_count = 11 when no RETROs closed and no new ones opened."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["open_retro_count"] == len(_RETROS_OPEN_AT_MILESTONE_START)


def test_new_retro_items_empty_for_milestone_48(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """new_retro_items is empty for .48 — RETRO-071 was already opened in .47."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["new_retro_items"] == []


def test_open_retro_items_contains_retro_033(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-033 appears in open_retro_items when not resolved."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-033" in ids


def test_open_retro_items_contains_retro_070(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-070 appears in open_retro_items when not resolved."""
    _write_fake_results(tmp_path, retro_070_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-070" in ids


def test_open_retro_items_contains_retro_071(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-071 appears in open_retro_items when not resolved."""
    _write_fake_results(tmp_path, dualgpu_proven=False, retro_071_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-071" in ids


def test_retro_070_absent_from_open_items_when_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-070 is NOT in open_retro_items when retro_070_resolved=True."""
    _write_fake_results(tmp_path, retro_070_resolved=True, gate_open=True, interwhen_recall=0.25)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-070" not in ids


def test_retro_071_absent_from_open_items_when_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-071 is NOT in open_retro_items when retro_071_resolved=True."""
    _write_fake_results(tmp_path, dualgpu_proven=True, retro_071_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-071" not in ids


# ---------------------------------------------------------------------------
# compute_retro — top_priorities_for_49 branches
# ---------------------------------------------------------------------------


def test_top_priorities_for_49_has_three_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """top_priorities_for_49 always contains exactly 3 entries."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert len(retro["top_priorities_for_49"]) == 3


def test_top_priority_hermes_v2_when_both_retros_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both retro_033 and retro_070 are not resolved, top priority is HERMES v2 pipeline."""
    _write_fake_results(tmp_path, retro_033_resolved=False, retro_070_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_49"][0]
    assert "HERMES" in first or "RETRO-070" in first


def test_top_priority_scale_200q_when_033_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_033 is resolved, top priority is scaling to 200q Wilson CI."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_49"][0]
    assert "200q" in first or "Wilson" in first


def test_top_priority_vr_17_when_070_resolved_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_070 is resolved but not retro_033, top priority is run VR #17."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=True,
        gate_open=True,
        interwhen_recall=0.25,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_49"][0]
    assert "#17" in first or "VR" in first


def test_top_priority_jepa_calibration_always_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JEPA v14 ECE calibration is always the second priority."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    second = retro["top_priorities_for_49"][1]
    assert "JEPA" in second or "calibration" in second or "Platt" in second


def test_top_priority_kv260_always_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KV260 FPGA synthesis is always the third priority."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    third = retro["top_priorities_for_49"][2]
    assert "KV260" in third or "Vivado" in third or "FPGA" in third


# ---------------------------------------------------------------------------
# compute_retro — wall time and experiment counts
# ---------------------------------------------------------------------------


def test_n_experiments_is_thirteen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_experiments_run = 13 when all 12 upstream files exist plus this retro."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_experiments_run"] == 13


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
    (tmp_path / "results" / "experiment_638_tier1_fr11_relay.json").unlink()
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_not_run"] >= 1


def test_mean_time_min_is_wall_time_over_n_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mean_time_min = total_wall_time_minutes / n_experiments (13 total)."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    n_total = len(_MILESTONE_RESULTS) + 1
    expected = round(retro["total_wall_time_minutes"] / n_total, 3)
    assert retro["mean_time_min"] == pytest.approx(expected, abs=1e-6)


def test_schema_and_milestone_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema and milestone fields are set to the expected v23 values."""
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
        "fr11_confirmed",
        "multilevel_wins",
        "adaptrack_improves",
        "hermes_improves",
        "fpga_tcl_updated",
        "retro_closure_rate",
        "open_retro_count",
        "new_retro_items",
        "open_retro_items",
        "top_priorities_for_49",
        "honest_verdict",
        "env_autofix",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["schema"] == SCHEMA
    assert artifact["env_autofix"] is True
    assert artifact["honest_verdict"] in {
        "first_positive_vr_achieved",
        "hermes_improved_gate_open_vr16_ready",
        "hermes_improved_all_retros_carry",
        "no_retros_closed",
    }
    assert artifact["experiment"] == EXP_ID
    assert artifact["milestone"] == MILESTONE
