"""Tests for scripts/experiment_600_retro_2026_04_45.py — Milestone 2026.04.45 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: all eight success criteria boolean branches, wall-time aggregation,
  closure rate, honest_verdict variants, new_retro_items structure,
  open_retro_items carry-forward, top_priorities_for_46
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.45
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_600_retro_2026_04_45 as retro_mod
from scripts.experiment_600_retro_2026_04_45 import (
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
    npu_iron_available: bool = False,
    npu_ninja_available: bool = False,
    v3_recall: float = 0.04,
    retro_066_resolved_flag: bool = False,
    dsvd_live_auc: float = 0.586,
    v12_val_auc: float = 1.0,
    retro_063_validated_flag: bool = True,
    retro_033_resolved_594: bool = False,
    signed_improvement_594: float = 0.0,
    retro_033_resolved_595: bool = False,
    signed_improvement_595: float = 0.0,
    retro_038_resolved_flag: bool = False,
    fr11_real_violations_confirmed: bool = False,
    hisr_credit_assignment_correct: bool = True,
    dwave_available: bool = True,
    vivado_status: str = "not_installed",
    bitfile_built: object = None,
    nup_v5_auc: float = 0.739,
) -> None:
    """Write minimal fake JSON files for each upstream experiment."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    files = {
        "experiment_589_exclusion_manifest_wire_in.json": {
            "status": "success",
            "duration_s": 0.747,
            "npu_iron_available": npu_iron_available,
            "npu_ninja_available": npu_ninja_available,
        },
        "experiment_590_live_assertion.json": {
            "status": "success",
            "duration_s": 0.001,
        },
        "experiment_591_coace_v3_live.json": {
            "status": "success",
            "duration_s": 0.016,
            "v3_recall": v3_recall,
            "gate_open": retro_066_resolved_flag,
            "retro_066_resolved": retro_066_resolved_flag,
        },
        "experiment_592_dsvd_live_val.json": {
            "status": "success",
            "duration_s": 0.166,
            "dsvd_live_auc": dsvd_live_auc,
            "gate_open": dsvd_live_auc >= 0.80,
        },
        "experiment_593_jepa_v12_retrain.json": {
            "status": "success",
            "duration_s": 7.703,
            "v12_val_auc": v12_val_auc,
            "retro_063_validated": retro_063_validated_flag,
        },
        "experiment_594_live_vr_coace_v3.json": {
            "status": "blocked" if not retro_033_resolved_594 else "success",
            "duration_s": 0.0,
            "signed_improvement": signed_improvement_594,
            "retro_033_resolved": retro_033_resolved_594,
        },
        "experiment_595_live_vr_dsvd.json": {
            "status": "blocked" if not retro_033_resolved_595 else "success",
            "duration_s": 0.0,
            "signed_improvement": signed_improvement_595,
            "retro_033_resolved": retro_033_resolved_595,
            "gate_open": False,
            "dsvd_live_auc": dsvd_live_auc,
        },
        "experiment_596_live_200q_wilson.json": {
            "status": "blocked",
            "duration_s": 0.0,
            "signed_improvement": 0.0,
            "wilson_lower_ci": None,
            "retro_038_resolved": retro_038_resolved_flag,
        },
        "experiment_597_fr11_real_violations_v4.json": {
            "status": "success",
            "duration_s": 0.001,
            "fr11_real_violations_confirmed": fr11_real_violations_confirmed,
        },
        "experiment_598_hisr_dwave.json": {
            "status": "success",
            "duration_s": 9.904,
            "hisr_credit_assignment_correct": hisr_credit_assignment_correct,
            "dwave_available": dwave_available,
        },
        "experiment_599_vivado_grpo_nup.json": {
            "status": "success",
            "duration_s": 0.363,
            "vivado_status": vivado_status,
            "bitfile_built": bitfile_built,
            "nup_v5_auc": nup_v5_auc,
        },
    }

    for fname, data in files.items():
        (results_dir / fname).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# compute_retro — success criteria
# ---------------------------------------------------------------------------


def test_retro_033_not_resolved_when_both_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is False when both Exp 594 and Exp 595 are blocked."""
    _write_fake_results(tmp_path, retro_033_resolved_594=False, retro_033_resolved_595=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is False


def test_retro_033_resolved_when_exp594_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is True when Exp 594 reports retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved_594=True, signed_improvement_594=0.05)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is True


def test_retro_033_resolved_when_exp595_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is True when Exp 595 reports retro_033_resolved=True."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved_594=False,
        retro_033_resolved_595=True,
        signed_improvement_595=0.03,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is True


def test_retro_038_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_038_resolved is False when Exp 596 is blocked."""
    _write_fake_results(tmp_path, retro_038_resolved_flag=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_038_resolved"] is False


def test_retro_038_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_038_resolved is True when Exp 596 sets retro_038_resolved=True."""
    _write_fake_results(tmp_path, retro_038_resolved_flag=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_038_resolved"] is True


def test_retro_066_not_resolved_when_recall_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_066_resolved is False when v3_recall=0.04 (gate_open=False)."""
    _write_fake_results(tmp_path, v3_recall=0.04, retro_066_resolved_flag=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_066_resolved"] is False


def test_retro_066_resolved_when_gate_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_066_resolved is True when Exp 591 sets retro_066_resolved=True."""
    _write_fake_results(tmp_path, v3_recall=0.35, retro_066_resolved_flag=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_066_resolved"] is True


def test_retro_063_validated_from_exp593(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_063_validated is True when Exp 593 sets retro_063_validated=True."""
    _write_fake_results(tmp_path, retro_063_validated_flag=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_063_validated"] is True


def test_retro_063_not_validated_when_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_063_validated is False when Exp 593 does not set the flag."""
    _write_fake_results(tmp_path, retro_063_validated_flag=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_063_validated"] is False


def test_fr11_improved_when_violations_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_improved is True when Exp 597 fr11_real_violations_confirmed=True."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_improved"] is True


def test_fr11_not_improved_when_gate_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_improved is False when gate is closed (fr11_real_violations_confirmed=False)."""
    _write_fake_results(tmp_path, fr11_real_violations_confirmed=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_improved"] is False


def test_dsvd_live_validated_when_auc_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dsvd_live_validated is True when dsvd_live_auc >= 0.80."""
    _write_fake_results(tmp_path, dsvd_live_auc=0.85)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dsvd_live_validated"] is True


def test_dsvd_live_not_validated_when_auc_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dsvd_live_validated is False when dsvd_live_auc < 0.80."""
    _write_fake_results(tmp_path, dsvd_live_auc=0.586)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dsvd_live_validated"] is False


def test_dsvd_live_not_validated_at_exact_threshold_minus_epsilon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dsvd_live_validated is False when dsvd_live_auc is just below 0.80."""
    _write_fake_results(tmp_path, dsvd_live_auc=0.799)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dsvd_live_validated"] is False


def test_npu_unblocked_when_iron_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """npu_unblocked is True when npu_iron_available=True."""
    _write_fake_results(tmp_path, npu_iron_available=True, npu_ninja_available=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["npu_unblocked"] is True


def test_npu_unblocked_when_ninja_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """npu_unblocked is True when npu_ninja_available=True."""
    _write_fake_results(tmp_path, npu_iron_available=False, npu_ninja_available=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["npu_unblocked"] is True


def test_npu_still_blocked_when_neither_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """npu_unblocked is False when neither npu_iron nor npu_ninja is available."""
    _write_fake_results(tmp_path, npu_iron_available=False, npu_ninja_available=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["npu_unblocked"] is False


def test_fpga_progress_when_bitfile_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_progress is True when bitfile_built=True."""
    _write_fake_results(tmp_path, bitfile_built=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_progress"] is True


def test_fpga_not_progressed_when_bitfile_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_progress is False when bitfile_built=None (Vivado not installed)."""
    _write_fake_results(tmp_path, bitfile_built=None)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_progress"] is False


def test_dwave_available_from_exp598(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dwave_available reflects Exp 598's dwave_available field."""
    _write_fake_results(tmp_path, dwave_available=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dwave_available"] is True


def test_hisr_credit_correct_from_exp598(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hisr_credit_correct reflects Exp 598's hisr_credit_assignment_correct field."""
    _write_fake_results(tmp_path, hisr_credit_assignment_correct=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["hisr_credit_correct"] is True


# ---------------------------------------------------------------------------
# compute_retro — honest_verdict logic
# ---------------------------------------------------------------------------


def test_honest_verdict_first_positive_achieved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='first_positive_achieved' when retro_033_resolved=True."""
    _write_fake_results(tmp_path, retro_033_resolved_594=True, signed_improvement_594=0.06)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "first_positive_achieved"


def test_honest_verdict_infrastructure_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='infrastructure_progress_no_accuracy_gain' when 033 not resolved but 063 validated and dwave available."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved_594=False,
        retro_033_resolved_595=False,
        retro_063_validated_flag=True,
        dwave_available=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "infrastructure_progress_no_accuracy_gain"


def test_honest_verdict_recall_fixed_no_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='recall_fixed_no_positive' when retro_066_resolved=True but 033 not resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved_594=False,
        retro_033_resolved_595=False,
        retro_063_validated_flag=False,
        dwave_available=False,
        retro_066_resolved_flag=True,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "recall_fixed_no_positive"


def test_honest_verdict_recall_still_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='recall_still_blocked_all_retros_open' when nothing is resolved."""
    _write_fake_results(
        tmp_path,
        retro_033_resolved_594=False,
        retro_033_resolved_595=False,
        retro_063_validated_flag=False,
        dwave_available=False,
        retro_066_resolved_flag=False,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "recall_still_blocked_all_retros_open"


# ---------------------------------------------------------------------------
# compute_retro — retro_closure_rate
# ---------------------------------------------------------------------------


def test_closure_rate_zero_when_nothing_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 0 when no RETROs are resolved this milestone."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_closure_rate"] == 0.0


def test_open_retro_count_increases_for_new_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_retro_count = len(open_at_start) - closed + new (2 new RETROs opened in .45)."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    # 0 closed, 2 new: len(_RETROS_OPEN_AT_MILESTONE_START) + 2
    expected = len(_RETROS_OPEN_AT_MILESTONE_START) + 2
    assert retro["open_retro_count"] == expected


# ---------------------------------------------------------------------------
# compute_retro — wall time and experiment counts
# ---------------------------------------------------------------------------


def test_n_experiments_is_twelve(
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
    (tmp_path / "results" / "experiment_599_vivado_grpo_nup.json").unlink()
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_not_run"] >= 1


def test_mean_time_min_is_wall_time_over_n_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mean_time_min = total_wall_time_minutes / n_experiments (12 total)."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    n_total = len(_MILESTONE_RESULTS) + 1
    expected = round(retro["total_wall_time_minutes"] / n_total, 3)
    assert retro["mean_time_min"] == pytest.approx(expected, abs=1e-6)


def test_wall_time_vs_prior_is_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wall_time_vs_prior_delta_minutes is negative (this milestone is faster than .44)."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    # This milestone's experiments are all <10s; prior was 4654 min.
    assert retro["wall_time_vs_prior_delta_minutes"] < 0


# ---------------------------------------------------------------------------
# compute_retro — structural checks
# ---------------------------------------------------------------------------


def test_new_retro_items_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each new_retro_item has id, title, carry_count, description, priority fields."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    for item in retro["new_retro_items"]:
        assert "id" in item
        assert "title" in item
        assert "carry_count" in item
        assert "description" in item
        assert "priority" in item


def test_new_retro_items_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New retro items opened in .45 are RETRO-068 and RETRO-069."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-068" in ids
    assert "RETRO-069" in ids


def test_top_priorities_for_46_has_three_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """top_priorities_for_46 contains exactly 3 entries."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert len(retro["top_priorities_for_46"]) == 3


def test_schema_and_milestone_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema and milestone fields are set to the expected v20 values."""
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
        "retro_038_resolved",
        "retro_066_resolved",
        "retro_063_validated",
        "fr11_improved",
        "dsvd_live_validated",
        "npu_unblocked",
        "fpga_progress",
        "dwave_available",
        "hisr_credit_correct",
        "retro_closure_rate",
        "open_retro_count",
        "new_retro_items",
        "open_retro_items",
        "top_priorities_for_46",
        "honest_verdict",
        "env_autofix",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["schema"] == SCHEMA
    assert artifact["env_autofix"] is True
    assert artifact["honest_verdict"] in {
        "first_positive_achieved",
        "infrastructure_progress_no_accuracy_gain",
        "recall_fixed_no_positive",
        "recall_still_blocked_all_retros_open",
    }
