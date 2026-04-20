"""Tests for scripts/experiment_574_retro_2026_04_43.py — Milestone 2026.04.43 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: success criteria evaluation, wall-time aggregation, closure rate,
  honest_verdict variants (root_cause_fixed / partial_fix / both_still_blocked),
  new_retro_items structure, open_retro_items carry-forward, top_priorities_for_44
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.43
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_574_retro_2026_04_43 as retro_mod
from scripts.experiment_574_retro_2026_04_43 import (
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
    n_pairs_563: int = 0,
    retro_061_resolved: bool = True,
    coace_tp_rate: float = 0.0588,
    retro_060_resolved: bool = False,
    v10_auc: float = 0.4444,
    hardware_latency_us: float | None = None,
    fpga_alive: bool = False,
    signed_improvement: float = 0.0,
    fr11_confirmed: bool = True,
) -> None:
    """Write minimal fake JSON files for each upstream experiment."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    files = {
        "experiment_563_live_data_a_v2.json": {
            "status": "blocked",
            "duration_s": 0.0,
            "n_pairs_collected": n_pairs_563,
            "inference_mode": "gpu_required",
            "retro_062_resolved": n_pairs_563 >= 40,
        },
        "experiment_564_coace_extractor.json": {
            "status": "success",
            "duration_s": 0.005,
        },
        "experiment_565_coace_live_diagnostic.json": {
            "status": "success",
            "duration_s": 0.005,
            "coace_tp_rate": coace_tp_rate,
            "retro_061_resolved": retro_061_resolved,
        },
        "experiment_566_jepa_pure_margin.json": {
            "status": "success",
            "duration_s": 13.03,
        },
        "experiment_567_jepa_v10_retrain.json": {
            "status": "success",
            "duration_s": 14.528,
            "v10_auc": v10_auc,
            "retro_060_resolved": retro_060_resolved,
        },
        "experiment_568_kv260_bringup_v2.json": {
            "status": "success",
            "duration_s": 3.865,
            "hardware_latency_us": hardware_latency_us,
            "fpga_alive": fpga_alive,
        },
        "experiment_569_live_vr_coace.json": {
            "status": "success",
            "duration_s": 69.583,
            "signed_improvement": signed_improvement,
        },
        "experiment_570_fr11_real_violations.json": {
            "status": "success",
            "duration_s": 0.002,
            "fr11_real_violations_confirmed": fr11_confirmed,
        },
        "experiment_571_hallufield_tier0e.json": {
            "status": "success",
            "duration_s": 3.354,
            "hallufield_auc": 0.9737,
        },
        "experiment_572_pra_eorm_beam_search.json": {
            "status": "success",
            "duration_s": 0.001,
            "pra_viable": True,
        },
        "experiment_573_energy_per_token_calibration.json": {
            "status": "success",
            "duration_s": 25.56,
            "rapl_available": False,
            "calibration_viable": False,
        },
    }

    for fname, data in files.items():
        (results_dir / fname).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# compute_retro — success criteria
# ---------------------------------------------------------------------------


def test_retro_062_not_resolved_when_zero_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_062_resolved is False when n_pairs_collected=0."""
    _write_fake_results(tmp_path, n_pairs_563=0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_062_resolved"] is False


def test_retro_062_resolved_when_enough_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_062_resolved is True when n_pairs_collected >= 40."""
    _write_fake_results(tmp_path, n_pairs_563=45)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_062_resolved"] is True


def test_retro_061_resolved_from_result_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_061_resolved reflects the retro_061_resolved field from Exp 565."""
    _write_fake_results(tmp_path, retro_061_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_061_resolved"] is True


def test_retro_061_not_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_061_resolved is False when the Exp 565 flag is False."""
    _write_fake_results(tmp_path, retro_061_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_061_resolved"] is False


def test_retro_060_not_resolved_when_auc_below_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_060_resolved is False when v10_auc=0.4444 < 0.5."""
    _write_fake_results(tmp_path, retro_060_resolved=False, v10_auc=0.4444)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_060_resolved"] is False


def test_retro_060_resolved_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_060_resolved is True when Exp 567 sets the flag."""
    _write_fake_results(tmp_path, retro_060_resolved=True, v10_auc=0.75)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_060_resolved"] is True


def test_fpga_alive_false_when_latency_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_alive is False when hardware_latency_us is None."""
    _write_fake_results(tmp_path, hardware_latency_us=None)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_alive"] is False


def test_fpga_alive_true_when_latency_under_100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_alive is True when hardware_latency_us is less than 100."""
    _write_fake_results(tmp_path, hardware_latency_us=50.0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_alive"] is True


def test_fpga_not_alive_when_latency_exactly_100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fpga_alive is False when hardware_latency_us is exactly 100 (boundary exclusive)."""
    _write_fake_results(tmp_path, hardware_latency_us=100.0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fpga_alive"] is False


def test_live_vr_positive_false_when_zero_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """live_vr_positive is False when signed_improvement=0.0."""
    _write_fake_results(tmp_path, signed_improvement=0.0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["live_vr_positive"] is False


def test_live_vr_positive_true_when_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """live_vr_positive is True when signed_improvement > 0."""
    _write_fake_results(tmp_path, signed_improvement=0.04)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["live_vr_positive"] is True


def test_fr11_real_violations_from_exp570(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fr11_real_violations reflects the confirmed flag from Exp 570."""
    _write_fake_results(tmp_path, fr11_confirmed=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["fr11_real_violations"] is True


# ---------------------------------------------------------------------------
# compute_retro — honest_verdict logic
# ---------------------------------------------------------------------------


def test_honest_verdict_partial_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='partial_fix' when only retro_061 is resolved."""
    _write_fake_results(tmp_path, retro_061_resolved=True, retro_060_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "partial_fix"


def test_honest_verdict_both_still_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='both_still_blocked' when neither RETRO is resolved."""
    _write_fake_results(tmp_path, retro_061_resolved=False, retro_060_resolved=False)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "both_still_blocked"


def test_honest_verdict_root_cause_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='root_cause_fixed' when both RETRO-060 and RETRO-061 are resolved."""
    _write_fake_results(tmp_path, retro_061_resolved=True, retro_060_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "root_cause_fixed"


def test_honest_verdict_partial_fix_when_only_060_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """honest_verdict='partial_fix' when only retro_060 is resolved."""
    _write_fake_results(tmp_path, retro_061_resolved=False, retro_060_resolved=True)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["honest_verdict"] == "partial_fix"


# ---------------------------------------------------------------------------
# compute_retro — retro_closure_rate
# ---------------------------------------------------------------------------


def test_closure_rate_one_of_nine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 1/9 when only RETRO-061 is closed."""
    _write_fake_results(tmp_path, retro_061_resolved=True, retro_060_resolved=False, n_pairs_563=0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = round(1 / len(_RETROS_OPEN_AT_MILESTONE_START), 3)
    assert retro["retro_closure_rate"] == pytest.approx(expected, abs=1e-6)


def test_closure_rate_zero_when_nothing_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure rate = 0 when no RETROs are resolved."""
    _write_fake_results(tmp_path, retro_061_resolved=False, retro_060_resolved=False, n_pairs_563=0)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["retro_closure_rate"] == 0.0


# ---------------------------------------------------------------------------
# compute_retro — wall time and experiment counts
# ---------------------------------------------------------------------------


def test_n_experiments_is_twelve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_experiments = 12 (11 upstream + this retro)."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_experiments"] == 12


def test_wall_time_sums_upstream_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """total_wall_time_minutes reflects the sum of upstream duration_s values."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    # Sum from fake data: 0+0.005+0.005+13.03+14.528+3.865+69.583+0.002+3.354+0.001+25.56
    expected_seconds = 0.0 + 0.005 + 0.005 + 13.03 + 14.528 + 3.865 + 69.583 + 0.002 + 3.354 + 0.001 + 25.56
    expected_minutes = round(expected_seconds / 60.0, 3)
    assert retro["total_wall_time_minutes"] == pytest.approx(expected_minutes, abs=0.01)


def test_mean_time_min_is_wall_time_over_n_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mean_time_min = total_wall_time_minutes / n_experiments."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    expected = round(retro["total_wall_time_minutes"] / retro["n_experiments"], 3)
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
    """New retro items open in .43 are RETRO-063, RETRO-064, RETRO-065."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-063" in ids
    assert "RETRO-064" in ids
    assert "RETRO-065" in ids


def test_top_priorities_for_44_has_three_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """top_priorities_for_44 contains exactly 3 entries."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert len(retro["top_priorities_for_44"]) == 3


def test_schema_and_milestone_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema and milestone fields are set to the expected v18 values."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["schema"] == SCHEMA
    assert retro["milestone"] == MILESTONE


# ---------------------------------------------------------------------------
# Missing upstream experiments
# ---------------------------------------------------------------------------


def test_missing_experiment_increments_n_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When one upstream result file is absent, n_missing >= 1."""
    # Write all results except Exp 573.
    _write_fake_results(tmp_path)
    (tmp_path / "results" / "experiment_573_energy_per_token_calibration.json").unlink()
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    retro = compute_retro()
    assert retro["n_missing"] >= 1


# ---------------------------------------------------------------------------
# main() integration — deliverable written with required fields
# ---------------------------------------------------------------------------


def test_main_writes_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes a valid JSON artifact at DELIVERABLE path with all required fields."""
    _write_fake_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    # Set CARNOT_REPO_ROOT so ExperimentTemplate._repo_root also resolves to tmp_path.
    # Without this, assert_deliverable_written() checks the real repo path and fails
    # if the file hasn't been written there yet.
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(exist_ok=True)

    retro_mod.main()

    deliverable = tmp_path / DELIVERABLE
    assert deliverable.exists(), f"Deliverable not written: {DELIVERABLE}"
    artifact = json.loads(deliverable.read_text())

    required_fields = [
        "experiment", "title", "run_date", "started_at", "finished_at",
        "duration_s", "status", "schema",
        "retro_062_resolved", "retro_061_resolved", "retro_060_resolved",
        "fpga_alive", "live_vr_positive", "fr11_real_violations",
        "retro_closure_rate", "new_retro_items", "top_priorities_for_44",
        "honest_verdict", "env_autofix",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["schema"] == SCHEMA
    assert artifact["env_autofix"] is True
    assert artifact["honest_verdict"] in {"root_cause_fixed", "partial_fix", "both_still_blocked"}
