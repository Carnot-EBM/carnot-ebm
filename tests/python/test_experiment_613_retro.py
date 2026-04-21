"""Tests for scripts/experiment_613_retro_2026_04_46.py — Milestone 2026.04.46 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: all nine success criteria boolean branches, wall-time aggregation,
  closure rate, honest_verdict variants, new_retro_items (RETRO-070 conditional),
  open_retro_items carry-forward, top_priorities_for_47 (both VR priority branches)
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.46
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_613_retro_2026_04_46 as retro_mod
from scripts.experiment_613_retro_2026_04_46 import (
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
    retro_067_resolved: bool = True,
    n_new_pairs: int = 200,
    v4_recall: float = 0.04,
    retro_068_resolved_flag: bool = False,
    post_finetune_val_auc: float = 0.158,
    retro_069_resolved_flag: bool = False,
    best_recall: float = 0.04,
    ilv_recall: float = 0.08,
    v12_ood_auc: float = 0.5,
    fr11_generalization_confirmed: bool = True,
    v6_val_auc: float = 0.9642857142857143,
    retro_049_resolved: bool = True,
    retro_033_resolved: bool = False,
    signed_improvement: float = 0.0,
    dwave_backend_registered: bool = True,
    fr11_real_violations_confirmed: bool = False,
    probe_viable: bool = False,
    synchronous_rtl_created: bool = True,
) -> None:
    """Write minimal fake JSON files for each upstream experiment in _MILESTONE_RESULTS."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    # exp600 — prior retro (only duration_s matters for wall time)
    (results_dir / "experiment_600_retro_2026_04_45.json").write_text(
        json.dumps({"status": "success", "duration_s": 0.0, "retro_033_resolved": False})
    )
    # exp601 — exclusion manifest verification
    (results_dir / "experiment_601_exclusion_manifest_verification.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 6.685,
            "retro_067_resolved": retro_067_resolved,
        })
    )
    # exp602 — live corpus expansion
    (results_dir / "experiment_602_live_corpus_v2.json").write_text(
        json.dumps({"status": "success", "duration_s": 2.658, "n_new_pairs": n_new_pairs})
    )
    # exp603 — CoACEV4 live
    (results_dir / "experiment_603_coace_v4_live.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 938.48,
            "v4_recall": v4_recall,
            "gate_open": False,
            "retro_068_resolved": retro_068_resolved_flag,
        })
    )
    # exp604 — DSVD live fine-tuning
    (results_dir / "experiment_604_dsvd_live_finetuning.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.403,
            "post_finetune_val_auc": post_finetune_val_auc,
            "gate_open": False,
            "retro_069_resolved": retro_069_resolved_flag,
        })
    )
    # exp605 — extractor diagnostic v4
    (results_dir / "experiment_605_extractor_diagnostic_v4.json").write_text(
        json.dumps({"status": "success", "duration_s": 0.126, "best_recall": best_recall})
    )
    # exp606 — interleaved logic
    (results_dir / "experiment_606_interleaved_logic.json").write_text(
        json.dumps({"status": "success", "duration_s": 0.015, "ilv_recall": ilv_recall})
    )
    # exp607 — JEPA v12 OOD
    (results_dir / "experiment_607_jepa_v12_ood.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 19.904,
            "v12_ood_auc": v12_ood_auc,
            "fr11_generalization_confirmed": fr11_generalization_confirmed,
        })
    )
    # exp608 — NUP Probe v6
    (results_dir / "experiment_608_nup_probe_v6.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 7.516,
            "v6_val_auc": v6_val_auc,
            "retro_049_resolved": retro_049_resolved,
        })
    )
    # exp609 — live VR CoACEV4
    (results_dir / "experiment_609_live_vr_coace_v4.json").write_text(
        json.dumps({
            "status": "blocked" if not retro_033_resolved else "success",
            "duration_s": 0.0,
            "signed_improvement": signed_improvement,
            "retro_033_resolved": retro_033_resolved,
        })
    )
    # exp610 — D-Wave wire-in
    (results_dir / "experiment_610_dwave_wire_in.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 24.813,
            "dwave_backend_registered": dwave_backend_registered,
        })
    )
    # exp611 — FR-11 real violations v5
    (results_dir / "experiment_611_flip_fr11_v5.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.0,
            "fr11_real_violations_confirmed": fr11_real_violations_confirmed,
        })
    )
    # exp612 — FACT-E p-bit
    (results_dir / "experiment_612_fact_e_pbit.json").write_text(
        json.dumps({
            "status": "success",
            "duration_s": 0.002,
            "probe_viable": probe_viable,
            "synchronous_rtl_created": synchronous_rtl_created,
        })
    )


# ---------------------------------------------------------------------------
# compute_retro — success criteria
# ---------------------------------------------------------------------------


def test_retro_033_not_resolved_when_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is False when Exp 609 is blocked."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is False


def test_retro_033_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is True when Exp 609 sets retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.05)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is True


def test_retro_049_resolved_when_nup_v6_auc_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_049_resolved is True when Exp 608 sets retro_049_resolved=True."""
    _write_fake_results(tmp_path, retro_049_resolved=True, v6_val_auc=0.964)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_049_resolved"] is True


def test_retro_049_not_resolved_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_049_resolved is False when Exp 608 does not set the flag."""
    _write_fake_results(tmp_path, retro_049_resolved=False, v6_val_auc=0.739)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_049_resolved"] is False


def test_retro_067_resolved_from_exp601(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_067_resolved is True when Exp 601 sets retro_067_resolved=True."""
    _write_fake_results(tmp_path, retro_067_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_067_resolved"] is True
    assert retro["manifest_verified"] is True


def test_retro_067_not_resolved_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_067_resolved is False when Exp 601 does not set the flag."""
    _write_fake_results(tmp_path, retro_067_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_067_resolved"] is False
    assert retro["manifest_verified"] is False


def test_retro_068_not_resolved_when_recall_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_068_resolved is False when v4_recall=0.04 (gate_open=False)."""
    _write_fake_results(tmp_path, v4_recall=0.04, retro_068_resolved_flag=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_068_resolved"] is False
    assert retro["coace_v4_recall"] == pytest.approx(0.04, abs=1e-6)


def test_retro_068_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_068_resolved is True when Exp 603 sets retro_068_resolved=True."""
    _write_fake_results(tmp_path, v4_recall=0.25, retro_068_resolved_flag=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_068_resolved"] is True


def test_retro_069_not_resolved_when_auc_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_069_resolved is False when post_finetune_val_auc=0.158."""
    _write_fake_results(tmp_path, post_finetune_val_auc=0.158, retro_069_resolved_flag=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_069_resolved"] is False
    assert retro["dsvd_live_auc"] == pytest.approx(0.158, abs=1e-6)


def test_retro_069_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_069_resolved is True when Exp 604 sets retro_069_resolved=True."""
    _write_fake_results(tmp_path, post_finetune_val_auc=0.85, retro_069_resolved_flag=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_069_resolved"] is True


def test_fr11_confirmed_from_exp607(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is True when Exp 607 sets fr11_generalization_confirmed=True."""
    _write_fake_results(
        tmp_path, fr11_generalization_confirmed=True, fr11_real_violations_confirmed=False
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is True


def test_fr11_confirmed_from_exp611_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is True when Exp 611 sets fr11_real_violations_confirmed=True."""
    _write_fake_results(
        tmp_path, fr11_generalization_confirmed=False, fr11_real_violations_confirmed=True
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is True


def test_fr11_not_confirmed_when_both_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_confirmed is False when neither Exp 607 nor Exp 611 confirms it."""
    _write_fake_results(
        tmp_path, fr11_generalization_confirmed=False, fr11_real_violations_confirmed=False
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_confirmed"] is False


def test_dwave_wired_from_exp610(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dwave_wired is True when Exp 610 sets dwave_backend_registered=True."""
    _write_fake_results(tmp_path, dwave_backend_registered=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dwave_wired"] is True


def test_dwave_not_wired_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dwave_wired is False when Exp 610 does not register D-Wave backend."""
    _write_fake_results(tmp_path, dwave_backend_registered=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dwave_wired"] is False


def test_corpus_expanded_when_n_new_pairs_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """corpus_expanded is True when Exp 602 n_new_pairs >= 80."""
    _write_fake_results(tmp_path, n_new_pairs=200)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["corpus_expanded"] is True


def test_corpus_not_expanded_when_n_new_pairs_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """corpus_expanded is False when Exp 602 n_new_pairs < 80."""
    _write_fake_results(tmp_path, n_new_pairs=50)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["corpus_expanded"] is False


def test_corpus_expanded_at_exact_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """corpus_expanded is True when n_new_pairs == 80 (exact boundary)."""
    _write_fake_results(tmp_path, n_new_pairs=80)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["corpus_expanded"] is True


def test_nup_v6_auc_reflects_exp608(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """nup_v6_auc reflects Exp 608's v6_val_auc field."""
    _write_fake_results(tmp_path, v6_val_auc=0.9642857142857143)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["nup_v6_auc"] == pytest.approx(0.9642857142857143, abs=1e-6)


def test_fpga_rtl_created_from_exp612(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_rtl_created is True when Exp 612 sets synchronous_rtl_created=True."""
    _write_fake_results(tmp_path, synchronous_rtl_created=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_rtl_created"] is True


# ---------------------------------------------------------------------------
# compute_retro — honest_verdict logic
# ---------------------------------------------------------------------------


def test_honest_verdict_first_positive_achieved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='first_positive_achieved' when retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "first_positive_achieved"


def test_honest_verdict_probe_and_manifest_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='probe_and_manifest_closed_recall_still_blocked' when 049 and 067 resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_049_resolved=True,
        retro_067_resolved=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "probe_and_manifest_closed_recall_still_blocked"


def test_honest_verdict_partial_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='partial_progress_recall_still_blocked' when only one of 049/067 resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_049_resolved=True,
        retro_067_resolved=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "partial_progress_recall_still_blocked"


def test_honest_verdict_no_retros_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='no_retros_closed' when nothing is resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved=False,
        retro_049_resolved=False,
        retro_067_resolved=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "no_retros_closed"


# ---------------------------------------------------------------------------
# compute_retro — retro_closure_rate and open_retro_count
# ---------------------------------------------------------------------------


def test_closure_rate_two_over_twelve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 2/12 when RETRO-049 and RETRO-067 are resolved."""
    _write_fake_results(tmp_path, retro_049_resolved=True, retro_067_resolved=True)
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
        retro_049_resolved=False,
        retro_067_resolved=False,
        retro_033_resolved=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_closure_rate"] == 0.0


def test_open_retro_count_decreases_on_closures_increases_on_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_retro_count = open_at_start - closed + new (2 closed, 1 new RETRO-070)."""
    _write_fake_results(
        tmp_path,
        retro_049_resolved=True,
        retro_067_resolved=True,
        retro_033_resolved=False,  # RETRO-070 opens
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = len(_RETROS_OPEN_AT_MILESTONE_START) - 2 + 1
    assert retro["open_retro_count"] == expected


def test_open_retro_count_no_new_when_033_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_retro_count = open_at_start - closed (no RETRO-070 when 033 resolved)."""
    _write_fake_results(
        tmp_path,
        retro_049_resolved=True,
        retro_067_resolved=True,
        retro_033_resolved=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    # 3 closed (049, 067, 033), 0 new RETROs
    expected = len(_RETROS_OPEN_AT_MILESTONE_START) - 3
    assert retro["open_retro_count"] == expected


# ---------------------------------------------------------------------------
# compute_retro — new_retro_items (RETRO-070 conditional)
# ---------------------------------------------------------------------------


def test_retro_070_added_when_033_still_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-070 is added to new_retro_items when retro_033 is still not resolved."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-070" in ids


def test_retro_070_not_added_when_033_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RETRO-070 is NOT added to new_retro_items when retro_033 is resolved."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-070" not in ids


def test_new_retro_item_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each new_retro_item has id, title, carry_count, description, priority fields."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    for item in retro["new_retro_items"]:
        assert "id" in item
        assert "title" in item
        assert "carry_count" in item
        assert "description" in item
        assert "priority" in item


# ---------------------------------------------------------------------------
# compute_retro — top_priorities_for_47 branches
# ---------------------------------------------------------------------------


def test_top_priorities_for_47_has_three_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """top_priorities_for_47 always contains exactly 3 entries."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert len(retro["top_priorities_for_47"]) == 3


def test_top_priority_llm_extractor_when_033_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_033 is not resolved, top priority is LLM-as-extractor redesign."""
    _write_fake_results(tmp_path, retro_033_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_47"][0]
    assert "LLM-as-extractor" in first or "Qwen3.5-0.8B" in first


def test_top_priority_scale_200q_when_033_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retro_033 is resolved, top priority is scaling to 200q Wilson CI."""
    _write_fake_results(tmp_path, retro_033_resolved=True, signed_improvement=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    first = retro["top_priorities_for_47"][0]
    assert "200q" in first or "Wilson" in first


# ---------------------------------------------------------------------------
# compute_retro — wall time and experiment counts
# ---------------------------------------------------------------------------


def test_n_experiments_is_fourteen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_experiments_run = 14 when all 13 upstream files exist plus this retro."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_experiments_run"] == 14


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
    (tmp_path / "results" / "experiment_612_fact_e_pbit.json").unlink()
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_not_run"] >= 1


def test_mean_time_min_is_wall_time_over_n_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mean_time_min = total_wall_time_minutes / n_experiments (14 total)."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    n_total = len(_MILESTONE_RESULTS) + 1
    expected = round(retro["total_wall_time_minutes"] / n_total, 3)
    assert retro["mean_time_min"] == pytest.approx(expected, abs=1e-6)


def test_schema_and_milestone_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema and milestone fields are set to the expected v21 values."""
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
        "retro_049_resolved",
        "retro_067_resolved",
        "retro_068_resolved",
        "retro_069_resolved",
        "fr11_confirmed",
        "dwave_wired",
        "corpus_expanded",
        "manifest_verified",
        "coace_v4_recall",
        "dsvd_live_auc",
        "nup_v6_auc",
        "retro_closure_rate",
        "open_retro_count",
        "new_retro_items",
        "open_retro_items",
        "top_priorities_for_47",
        "honest_verdict",
        "env_autofix",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["schema"] == SCHEMA
    assert artifact["env_autofix"] is True
    assert artifact["honest_verdict"] in {
        "first_positive_achieved",
        "probe_and_manifest_closed_recall_still_blocked",
        "partial_progress_recall_still_blocked",
        "no_retros_closed",
    }
    assert artifact["experiment"] == EXP_ID
