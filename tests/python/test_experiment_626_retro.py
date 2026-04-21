"""Tests for scripts/experiment_626_retro_2026_04_47.py — Milestone 2026.04.47 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: all v22 success criteria boolean branches, wall-time aggregation,
  closure rate, honest_verdict variants, new_retro_items (RETRO-071 conditional),
  open_retro_items carry-forward, top_priorities_for_48 (all three VR priority branches)
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.47
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_626_retro_2026_04_47 as retro_mod
from scripts.experiment_626_retro_2026_04_47 import (
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
    gpu1_utilization_confirmed: bool = False,
    retro_070_resolved: bool = False,
    v1_recall_616: float = 0.04,
    gate_open_616: bool = False,
    best_extractor_617: str | None = None,
    v13_ood_auc: float = 0.8676,
    v13_ece: float = 0.2073,
    calibration_improved: bool = False,
    symcode_live_auc: float = 0.804,
    retro_069_resolved: bool = True,
    retro_033_resolved: bool = False,
    signed_improvement: float = 0.0,
    adaptation_effective: bool = True,
    nup_v6_wired: bool = True,
    cascade_latency_ms: float = 1.27,
    trust_recall: float = 0.0,
    best_extractor_623: str = "llm_v1",
    synthesis_succeeded: str = "not_attempted",
    simulation_validated: bool = True,
    fr11_real_violations_confirmed: bool = False,
) -> None:
    """Write minimal fake JSON files for each upstream experiment in _MILESTONE_RESULTS."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    # exp614 — DualGPU exclusion manifest
    (results_dir / "experiment_614_exclusion_manifest_dualgpu.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 3.662,
            "gpu1_utilization_confirmed": gpu1_utilization_confirmed,
        })
    )
    # exp615 — live corpus v3
    (results_dir / "experiment_615_live_corpus_v3.json").write_text(
        json.dumps({"status": "success", "duration_s": 0.0})
    )
    # exp616 — LLMAsExtractorV1
    (results_dir / "experiment_616_llm_extractor_v1.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.0,
            "v1_recall": v1_recall_616,
            "gate_open": gate_open_616,
            "retro_070_resolved": retro_070_resolved,
        })
    )
    # exp617 — extractor diagnostic v5 (timed out — best_extractor absent unless overridden)
    exp617_data: dict = {"status": "timed_out"}
    if best_extractor_617 is not None:
        exp617_data["best_extractor"] = best_extractor_617
        exp617_data["duration_s"] = 1800.0
    (results_dir / "experiment_617_extractor_diagnostic_v5.json").write_text(
        json.dumps(exp617_data)
    )
    # exp618 — JEPA v13 CAPO
    (results_dir / "experiment_618_jepa_v13_capo.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 119.153,
            "v13_ood_auc": v13_ood_auc,
            "v13_ece": v13_ece,
            "calibration_improved": calibration_improved,
        })
    )
    # exp619 — SymCode DSVD
    (results_dir / "experiment_619_dsvd_symcode.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.004,
            "symcode_live_auc": symcode_live_auc,
            "retro_069_resolved": retro_069_resolved,
        })
    )
    # exp620 — live VR attempt #15
    (results_dir / "experiment_620_live_vr_attempt_15.json").write_text(
        json.dumps({
            "status": "blocked" if not retro_033_resolved else "success",
            "duration_s": 0.0,
            "gate_open": retro_033_resolved,
            "signed_improvement": signed_improvement,
            "retro_033_resolved": retro_033_resolved,
        })
    )
    # exp621 — MetaJuLS adaptation
    (results_dir / "experiment_621_metajuls_adaptation.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.012,
            "adaptation_effective": adaptation_effective,
        })
    )
    # exp622 — NUP v6 cascade
    (results_dir / "experiment_622_nup_v6_cascade.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.14,
            "nup_v6_wired": nup_v6_wired,
            "cascade_latency_ms": cascade_latency_ms,
        })
    )
    # exp623 — TRUST agents
    (results_dir / "experiment_623_trust_agents.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.006,
            "trust_recall": trust_recall,
            "best_extractor": best_extractor_623,
        })
    )
    # exp624 — KV260 Vivado v2
    (results_dir / "experiment_624_kv260_vivado_v2.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 2.545,
            "synthesis_succeeded": synthesis_succeeded,
            "simulation_validated": simulation_validated,
        })
    )
    # exp625 — FR-11 relay
    (results_dir / "experiment_625_tier1_fr11_relay.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.0,
            "fr11_real_violations_confirmed": fr11_real_violations_confirmed,
        })
    )


# ---------------------------------------------------------------------------
# compute_retro — primary success criteria
# ---------------------------------------------------------------------------


def test_retro_033_not_resolved_when_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is False when Exp 620 is blocked."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is False


def test_retro_033_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is True when Exp 620 sets retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.05)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is True


def test_retro_069_resolved_from_exp619(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_069_resolved is True when Exp 619 sets retro_069_resolved=True."""
    _write_fake_results(tmp_path, symcode_live_auc=0.804, retro_069_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_069_resolved"] is True
    assert retro["symcode_live_auc"] == pytest.approx(0.804, abs=1e-6)


def test_retro_069_not_resolved_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_069_resolved is False when Exp 619 does not set the flag."""
    _write_fake_results(tmp_path, symcode_live_auc=0.30, retro_069_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_069_resolved"] is False


def test_retro_070_not_resolved_when_recall_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_070_resolved is False when Exp 616 v1_recall=0.04."""
    _write_fake_results(tmp_path, v1_recall_616=0.04, retro_070_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_070_resolved"] is False
    assert retro["v1_recall"] == pytest.approx(0.04, abs=1e-6)


def test_retro_070_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_070_resolved is True when Exp 616 sets retro_070_resolved=True."""
    _write_fake_results(tmp_path, v1_recall_616=0.25, retro_070_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_070_resolved"] is True


def test_nup_v6_deployed_from_exp622(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nup_v6_deployed is True when Exp 622 sets nup_v6_wired=True."""
    _write_fake_results(tmp_path, nup_v6_wired=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["nup_v6_deployed"] is True


def test_nup_v6_not_deployed_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nup_v6_deployed is False when Exp 622 does not set nup_v6_wired."""
    _write_fake_results(tmp_path, nup_v6_wired=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["nup_v6_deployed"] is False


def test_fr11_confirmed_from_exp625(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is True when Exp 625 fr11_real_violations_confirmed=True."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is True


def test_fr11_not_confirmed_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is False when Exp 625 fr11_real_violations_confirmed=False."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is False


def test_jepa_v13_calibrated_from_exp618(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jepa_v13_calibrated is True when Exp 618 sets calibration_improved=True."""
    _write_fake_results(tmp_path, calibration_improved=True, v13_ece=0.08)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["jepa_v13_calibrated"] is True


def test_jepa_v13_not_calibrated_when_ece_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jepa_v13_calibrated is False when calibration_improved=False (v13_ece=0.207)."""
    _write_fake_results(tmp_path, calibration_improved=False, v13_ece=0.2073)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["jepa_v13_calibrated"] is False
    assert retro["v13_ece"] == pytest.approx(0.2073, abs=1e-4)


def test_adaptation_effective_from_exp621(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """adaptation_effective is True when Exp 621 sets adaptation_effective=True."""
    _write_fake_results(tmp_path, adaptation_effective=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["adaptation_effective"] is True


def test_adaptation_not_effective_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """adaptation_effective is False when Exp 621 does not set the flag."""
    _write_fake_results(tmp_path, adaptation_effective=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["adaptation_effective"] is False


def test_best_extractor_unknown_when_exp617_timed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """best_extractor='unknown' when Exp 617 timed out and has no best_extractor field."""
    _write_fake_results(tmp_path, best_extractor_617=None)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["best_extractor"] == "unknown"


def test_best_extractor_set_when_exp617_has_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """best_extractor reflects Exp 617 field when present."""
    _write_fake_results(tmp_path, best_extractor_617="llm_v1")
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["best_extractor"] == "llm_v1"


def test_dualgpu_confirmed_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dualgpu_confirmed is False when Exp 614 gpu1_utilization_confirmed=False."""
    _write_fake_results(tmp_path, gpu1_utilization_confirmed=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dualgpu_confirmed"] is False


def test_dualgpu_confirmed_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dualgpu_confirmed is True when Exp 614 gpu1_utilization_confirmed=True."""
    _write_fake_results(tmp_path, gpu1_utilization_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dualgpu_confirmed"] is True


def test_simulation_validated_from_exp624(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """simulation_validated reflects Exp 624 simulation_validated field."""
    _write_fake_results(tmp_path, simulation_validated=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["simulation_validated"] is True


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


def test_honest_verdict_symcode_closed_nup_deployed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='symcode_closed_nup_deployed_recall_still_blocked' when 069 resolved and NUP deployed."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_069_resolved=True,
        nup_v6_wired=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "symcode_closed_nup_deployed_recall_still_blocked"


def test_honest_verdict_symcode_closed_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='symcode_closed_recall_still_blocked' when 069 resolved but NUP not deployed."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_069_resolved=True,
        nup_v6_wired=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "symcode_closed_recall_still_blocked"


def test_honest_verdict_no_retros_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='no_retros_closed' when nothing is resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_069_resolved=False,
        nup_v6_wired=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "no_retros_closed"


# ---------------------------------------------------------------------------
# compute_retro — retro_closure_rate and open_retro_count
# ---------------------------------------------------------------------------


def test_closure_rate_one_over_eleven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 1/11 when only RETRO-069 is resolved."""
    _write_fake_results(tmp_path, retro_069_resolved=True, gpu1_utilization_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = round(1 / len(_RETROS_OPEN_AT_MILESTONE_START), 3)
    assert retro["retro_closure_rate"] == pytest.approx(expected, abs=1e-6)


def test_closure_rate_zero_when_nothing_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 0 when no RETROs are resolved this milestone."""
    _write_fake_results(
        tmp_path,
        retro_069_resolved=False,
        gpu1_utilization_confirmed=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_closure_rate"] == 0.0


def test_open_retro_count_with_closure_and_new_retro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_retro_count = 11 - 1 (069 closed) + 1 (071 opened) = 11."""
    _write_fake_results(
        tmp_path,
        retro_069_resolved=True,
        gpu1_utilization_confirmed=False,  # triggers RETRO-071
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = len(_RETROS_OPEN_AT_MILESTONE_START) - 1 + 1
    assert retro["open_retro_count"] == expected


def test_open_retro_count_no_new_when_dualgpu_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_retro_count = 11 - 1 = 10 when 069 closed and dualgpu confirmed (no RETRO-071)."""
    _write_fake_results(
        tmp_path,
        retro_069_resolved=True,
        gpu1_utilization_confirmed=True,  # no RETRO-071
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = len(_RETROS_OPEN_AT_MILESTONE_START) - 1
    assert retro["open_retro_count"] == expected


# ---------------------------------------------------------------------------
# compute_retro — new_retro_items (RETRO-071 conditional)
# ---------------------------------------------------------------------------


def test_retro_071_added_when_dualgpu_not_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-071 is added to new_retro_items when dualgpu_confirmed=False."""
    _write_fake_results(tmp_path, gpu1_utilization_confirmed=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-071" in ids


def test_retro_071_not_added_when_dualgpu_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-071 is NOT added to new_retro_items when dualgpu_confirmed=True."""
    _write_fake_results(tmp_path, gpu1_utilization_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-071" not in ids


def test_new_retro_item_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each new_retro_item has id, title, carry_count, description, priority fields."""
    _write_fake_results(tmp_path, gpu1_utilization_confirmed=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    for item in retro["new_retro_items"]:
        assert "id" in item
        assert "title" in item
        assert "carry_count" in item
        assert "description" in item
        assert "priority" in item


# ---------------------------------------------------------------------------
# compute_retro — top_priorities_for_48 branches
# ---------------------------------------------------------------------------


def test_top_priorities_for_48_has_three_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """top_priorities_for_48 always contains exactly 3 entries."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert len(retro["top_priorities_for_48"]) == 3


def test_top_priority_interwhen_when_both_retros_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both retro_033 and retro_070 are not resolved, top priority is interwhen + ORACLE."""
    _write_fake_results(tmp_path, retro_033_resolved=False, retro_070_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_48"][0]
    assert "interwhen" in first or "ORACLE" in first


def test_top_priority_scale_200q_when_033_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_033 is resolved, top priority is scaling to 200q Wilson CI."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_48"][0]
    assert "200q" in first or "Wilson" in first


def test_top_priority_vr_attempt_when_070_resolved_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_070 is resolved but not retro_033, top priority is run VR attempt #16."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_070_resolved=True,
        v1_recall_616=0.25,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_48"][0]
    assert "attempt #16" in first or "VR" in first


def test_top_priority_jepa_v14_always_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JEPA v14 training on ORACLE corpus is always the second priority."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    second = retro["top_priorities_for_48"][1]
    assert "JEPA v14" in second or "ORACLE" in second


def test_top_priority_kv260_always_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KV260 FPGA synthesis is always the third priority."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    third = retro["top_priorities_for_48"][2]
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
    (tmp_path / "results" / "experiment_625_tier1_fr11_relay.json").unlink()
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
    """Schema and milestone fields are set to the expected v22 values."""
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
        "retro_069_resolved",
        "retro_070_resolved",
        "nup_v6_deployed",
        "fr11_confirmed",
        "jepa_v13_calibrated",
        "adaptation_effective",
        "best_extractor",
        "dualgpu_confirmed",
        "retro_closure_rate",
        "open_retro_count",
        "new_retro_items",
        "open_retro_items",
        "top_priorities_for_48",
        "honest_verdict",
        "env_autofix",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["schema"] == SCHEMA
    assert artifact["env_autofix"] is True
    assert artifact["honest_verdict"] in {
        "first_positive_vr_achieved",
        "symcode_closed_nup_deployed_recall_still_blocked",
        "symcode_closed_recall_still_blocked",
        "no_retros_closed",
    }
    assert artifact["experiment"] == EXP_ID
