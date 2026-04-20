"""Tests for scripts/experiment_588_retro_2026_04_44.py — Milestone 2026.04.44 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: all ten success criteria boolean branches, wall-time aggregation,
  closure rate, honest_verdict variants, new_retro_items structure,
  open_retro_items carry-forward, top_priorities_for_45
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.44
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_588_retro_2026_04_44 as retro_mod
from scripts.experiment_588_retro_2026_04_44 import (
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
    exclusion_manifest_built: bool = True,
    retro_064_partial_flag: bool = False,
    retro_064_resolved_flag: bool = False,
    v2_recall_581: float = 0.059,
    v11_auc: float = 1.0,
    n_pairs_578: int = 100,
    signed_improvement_582: float = 0.0,
    inference_mode_582: str = "blocked_gate_closed_recall_too_low",
    fr11_improved: bool = False,
    bitfile_built: bool = False,
    vivado_available: bool = False,
    formula_interpretable: bool = True,
    tier_2_5_viable: bool = True,
) -> None:
    """Write minimal fake JSON files for each upstream experiment."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    files = {
        "experiment_575_exclusion_manifest.json": {
            "status": "success",
            "duration_s": 0.001,
            "exclusion_manifest_built": exclusion_manifest_built,
        },
        "experiment_576_coace_recall_boost.json": {
            "status": "success",
            "duration_s": 0.006,
            "v2_recall": 0.867,
            "retro_064_partial": True,
            "retro_064_resolved": True,
        },
        "experiment_577_jepa_cpmi_pairs.json": {
            "status": "success",
            "duration_s": 0.099,
            "n_real_pairs": 9,
        },
        "experiment_578_live_data_a_v3.json": {
            "status": "success",
            "duration_s": 66.56,
            "n_pairs_collected": n_pairs_578,
            "inference_mode": "live_gpu",
            "retro_062_resolved": n_pairs_578 >= 40,
        },
        "experiment_579_live_data_c.json": {
            "status": "blocked",
            "n_pairs_collected": 0,
        },
        "experiment_580_jepa_v11_retrain.json": {
            "status": "success",
            "duration_s": 1.961,
            "v11_auc": v11_auc,
            "retro_063_resolved": v11_auc > 0.5,
            "fr11_retrain_complete": True,
        },
        "experiment_581_coace_recall_diagnostic_v2.json": {
            "status": "success",
            "duration_s": 0.005,
            "v2_recall": v2_recall_581,
            "retro_064_partial": retro_064_partial_flag,
            "retro_064_resolved": retro_064_resolved_flag,
            "gate_open": retro_064_partial_flag,
        },
        "experiment_582_live_vr_coace_v2.json": {
            "status": "blocked",
            "duration_s": 0.0,
            "signed_improvement": signed_improvement_582,
            "inference_mode": inference_mode_582,
            "retro_033_resolved": (
                signed_improvement_582 > 0.0 and inference_mode_582 == "live_gpu"
            ),
        },
        "experiment_583_fr11_real_violations_v3.json": {
            "status": "blocked",
            "duration_s": 0.001,
            "fr11_improved": fr11_improved,
        },
        "experiment_584_kv260_synthesis.json": {
            "bitfile_built": bitfile_built,
            "vivado_available": vivado_available,
        },
        "experiment_585_kv260_live_benchmark_v3.json": {
            "status": "blocked",
            "bitfile_built": False,
        },
        "experiment_586_symbolic_kan_energy.json": {
            "status": "success",
            "duration_s": 1.983,
            "formula_interpretable": formula_interpretable,
        },
        "experiment_587_dsvd_adapter.json": {
            "status": "success",
            "duration_s": 0.093,
            "tier_2_5_viable": tier_2_5_viable,
            "dsvd_auc": 0.976,
        },
    }

    for fname, data in files.items():
        (results_dir / fname).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# compute_retro — success criteria
# ---------------------------------------------------------------------------


def test_retro_056_resolved_when_manifest_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_056_resolved is True when Exp 575 exclusion_manifest_built=True."""
    _write_fake_results(tmp_path, exclusion_manifest_built=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_056_resolved"] is True


def test_retro_056_not_resolved_when_manifest_not_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_056_resolved is False when exclusion_manifest_built=False."""
    _write_fake_results(tmp_path, exclusion_manifest_built=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_056_resolved"] is False


def test_retro_064_partial_from_exp581_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_064_partial reflects Exp 581's retro_064_partial field."""
    _write_fake_results(tmp_path, retro_064_partial_flag=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_064_partial"] is True


def test_retro_064_partial_false_when_recall_too_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_064_partial is False when Exp 581 gate is closed (v2_recall=0.059)."""
    _write_fake_results(tmp_path, retro_064_partial_flag=False, v2_recall_581=0.059)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_064_partial"] is False


def test_retro_064_resolved_from_exp581_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_064_resolved reflects Exp 581's retro_064_resolved field."""
    _write_fake_results(tmp_path, retro_064_resolved_flag=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_064_resolved"] is True


def test_retro_063_resolved_when_v11_auc_above_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_063_resolved is True when Exp 580 v11_auc > 0.5."""
    _write_fake_results(tmp_path, v11_auc=1.0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_063_resolved"] is True


def test_retro_063_not_resolved_when_v11_auc_at_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_063_resolved is False when v11_auc=0.5 (boundary exclusive)."""
    _write_fake_results(tmp_path, v11_auc=0.5)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_063_resolved"] is False


def test_retro_063_not_resolved_when_v11_auc_below_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_063_resolved is False when v11_auc < 0.5."""
    _write_fake_results(tmp_path, v11_auc=0.4)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_063_resolved"] is False


def test_retro_062_resolved_when_enough_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_062_resolved is True when n_pairs_collected >= 40."""
    _write_fake_results(tmp_path, n_pairs_578=100)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_062_resolved"] is True


def test_retro_062_not_resolved_when_too_few_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_062_resolved is False when n_pairs_collected < 40."""
    _write_fake_results(tmp_path, n_pairs_578=10)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_062_resolved"] is False


def test_retro_033_resolved_when_positive_live_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is True only when signed_improvement > 0 AND inference_mode='live_gpu'."""
    _write_fake_results(
        tmp_path, signed_improvement_582=0.04, inference_mode_582="live_gpu"
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is True


def test_retro_033_not_resolved_when_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is False when Exp 582 is blocked (gate closed)."""
    _write_fake_results(
        tmp_path,
        signed_improvement_582=0.0,
        inference_mode_582="blocked_gate_closed_recall_too_low",
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is False


def test_retro_033_not_resolved_when_positive_but_not_live_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_033_resolved is False when improvement > 0 but not live_gpu mode."""
    _write_fake_results(
        tmp_path, signed_improvement_582=0.04, inference_mode_582="simulated"
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_033_resolved"] is False


def test_fr11_improved_from_exp583(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_improved reflects Exp 583's fr11_improved field."""
    _write_fake_results(tmp_path, fr11_improved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_improved"] is True


def test_fr11_not_improved_when_gate_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_improved is False when Exp 583 gate was closed."""
    _write_fake_results(tmp_path, fr11_improved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_improved"] is False


def test_fpga_progress_when_bitfile_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_progress is True when bitfile_built=True."""
    _write_fake_results(tmp_path, bitfile_built=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_progress"] is True


def test_fpga_progress_when_vivado_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_progress is True when vivado_available=True."""
    _write_fake_results(tmp_path, vivado_available=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_progress"] is True


def test_fpga_progress_false_when_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_progress is False when neither bitfile_built nor vivado_available."""
    _write_fake_results(tmp_path, bitfile_built=False, vivado_available=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_progress"] is False


def test_symbolic_viable_from_exp586(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbolic_viable reflects Exp 586's formula_interpretable field."""
    _write_fake_results(tmp_path, formula_interpretable=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["symbolic_viable"] is True


def test_symbolic_not_viable_when_formula_not_interpretable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symbolic_viable is False when formula_interpretable=False."""
    _write_fake_results(tmp_path, formula_interpretable=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["symbolic_viable"] is False


def test_dsvd_viable_from_exp587(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dsvd_viable reflects Exp 587's tier_2_5_viable field."""
    _write_fake_results(tmp_path, tier_2_5_viable=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dsvd_viable"] is True


def test_dsvd_not_viable_when_flag_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dsvd_viable is False when tier_2_5_viable=False."""
    _write_fake_results(tmp_path, tier_2_5_viable=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["dsvd_viable"] is False


# ---------------------------------------------------------------------------
# compute_retro — honest_verdict logic
# ---------------------------------------------------------------------------


def test_honest_verdict_first_positive_achieved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='first_positive_achieved' when retro_033_resolved=True."""
    _write_fake_results(
        tmp_path, signed_improvement_582=0.06, inference_mode_582="live_gpu"
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "first_positive_achieved"


def test_honest_verdict_recall_fixed_no_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='recall_fixed_no_positive' when retro_064_partial=True but 033 not resolved."""
    _write_fake_results(tmp_path, retro_064_partial_flag=True, signed_improvement_582=0.0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "recall_fixed_no_positive"


def test_honest_verdict_recall_still_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='recall_still_blocked' when neither retro_033 nor retro_064_partial resolved."""
    _write_fake_results(tmp_path, retro_064_partial_flag=False, signed_improvement_582=0.0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "recall_still_blocked"


# ---------------------------------------------------------------------------
# compute_retro — retro_closure_rate
# ---------------------------------------------------------------------------


def test_closure_rate_three_of_eleven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 3/11 when RETRO-056, RETRO-062, RETRO-063 are closed."""
    _write_fake_results(
        tmp_path,
        exclusion_manifest_built=True,
        n_pairs_578=100,
        v11_auc=1.0,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = round(3 / len(_RETROS_OPEN_AT_MILESTONE_START), 3)
    assert retro["retro_closure_rate"] == pytest.approx(expected, abs=1e-6)


def test_closure_rate_zero_when_nothing_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 0 when no RETROs are resolved this milestone."""
    _write_fake_results(
        tmp_path,
        exclusion_manifest_built=False,
        n_pairs_578=0,
        v11_auc=0.4,
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_closure_rate"] == 0.0


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
    (tmp_path / "results" / "experiment_587_dsvd_adapter.json").unlink()
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
    """New retro items opened in .44 are RETRO-066 and RETRO-067."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-066" in ids
    assert "RETRO-067" in ids


def test_top_priorities_for_45_has_three_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """top_priorities_for_45 contains exactly 3 entries."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert len(retro["top_priorities_for_45"]) == 3


def test_schema_and_milestone_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema and milestone fields are set to the expected v19 values."""
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
        "retro_056_resolved",
        "retro_064_partial", "retro_064_resolved",
        "retro_063_resolved",
        "retro_062_resolved",
        "retro_033_resolved",
        "fr11_improved",
        "fpga_progress",
        "symbolic_viable",
        "dsvd_viable",
        "retro_closure_rate",
        "new_retro_items",
        "top_priorities_for_45",
        "honest_verdict",
        "env_autofix",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["schema"] == SCHEMA
    assert artifact["env_autofix"] is True
    assert artifact["honest_verdict"] in {
        "first_positive_achieved",
        "recall_fixed_no_positive",
        "recall_still_blocked",
    }
