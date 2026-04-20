"""Tests for scripts/experiment_562_retro_2026_04_42.py — Milestone 2026.04.42 retrospective.

Coverage targets (targeted coverage of code added in this session only):
- _load_result: missing path, invalid JSON, valid JSON
- compute_retro: success criteria evaluation, wall-time aggregation, closure rate,
  headline_result variants (broken / partial / intact), top3_slowest ordering,
  new_retro_items structure, open_retro_items carry-forward
- main: artifact written to disk, schema set correctly, all required fields present

Spec: REQ-INFRA-058, REQ-INFRA-076
SCENARIO: RETRO-2026.04.42
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_562_retro_2026_04_42 as retro_mod
from scripts.experiment_562_retro_2026_04_42 import (
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
    # Monkeypatch _REPO_ROOT so the relative path resolves correctly.
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
# Helper: build a minimal set of fake experiment results
# ---------------------------------------------------------------------------


def _minimal_results(
    tmp_path: Path,
    *,
    exclusion_manifest: bool = True,
    live_a_status: str = "blocked",
    live_b_status: str = "success",
    n_labeled: int = 132,
    entropy: float = 1.5019,
    retro_058_data_ready: bool = True,
    jepa_auc: float = 0.4286,
    retro_056_closed: bool = False,
    kaem_energy_mad: float | None = None,
    retro_057_closed: bool = False,
    fr11_real_data: bool = True,
) -> dict[str, Any]:
    """Write fake result JSON files to tmp_path and return a results mapping."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    files: dict[str, tuple[str, dict]] = {
        "549": ("experiment_549_exclusion_manifest.json", {
            "status": "success",
            "exclusion_manifest_created": exclusion_manifest,
            "duration_s": 0.069,
            "honest_verdict": "retro_059_closed",
        }),
        "550": ("experiment_550_batching_real_migration.json", {
            "status": "success",
            "duration_s": 0.146,
            "scripts_migrated": ["experiment_308", "experiment_260"],
            "honest_verdict": "batching_migration_complete",
        }),
        "551": ("experiment_551_live_data_a.json", {
            "status": live_a_status,
            "inference_mode": "gpu_required" if live_a_status == "blocked" else "live_gpu",
            "n_pairs_collected": 0 if live_a_status == "blocked" else 50,
            "duration_s": 0.0,
            "honest_verdict": "gpu_required" if live_a_status == "blocked" else "live_data_collected",
        }),
        "552": ("experiment_552_live_data_b.json", {
            "status": live_b_status,
            "inference_mode": "live_gpu" if live_b_status == "success" else "gpu_required",
            "n_pairs_collected": 100 if live_b_status == "success" else 0,
            "mean_latency_s": 2.81,
            "duration_s": 143.974,
            "honest_verdict": "live_data_collected",
        }),
        "553": ("experiment_553_fover_corpus_v2.json", {
            "status": "success",
            "n_pairs_after_balance": n_labeled,
            "constraint_type_entropy_after": entropy,
            "retro_058_data_ready": retro_058_data_ready,
            "n_sources_merged": 3,
            "duration_s": 0.011,
            "honest_verdict": "corpus_ready",
        }),
        "554": ("experiment_554_extraction_diagnostic.json", {
            "status": "success",
            "duration_s": 0.008,
            "honest_verdict": "diagnostic_complete",
        }),
        "555": ("experiment_555_confidence_weighted.json", {
            "status": "success",
            "duration_s": 0.001,
            "honest_verdict": "marginal_improvement",
        }),
        "556": ("experiment_556_eorm_grpo_retrain.json", {
            "status": "success",
            "inference_mode": "real_data",
            "n_training_pairs": 132,
            "n_contrastive_triples": 9,
            "before_auc": 1.0,
            "after_auc": 1.0,
            "auc_improvement": 0.0,
            "duration_s": 175.138,
            "honest_verdict": "real_data_improvement",
        }),
        "557": ("experiment_557_jepa_v9_retrain.json", {
            "status": "success",
            "inference_mode": "real_data",
            "final_auc": jepa_auc,
            "corpus_entropy": entropy,
            "n_train": 85,
            "retro_056_closed": retro_056_closed,
            "duration_s": 11.284,
            "honest_verdict": "jepa_still_inverted",
        }),
        "558": ("experiment_558_internal_probe_real.json", {
            "status": "success",
            "n_labeled": 132,
            "probe_auc": 0.5217,
            "eorm_auc_for_comparison": 1.0,
            "probe_viable": False,
            "duration_s": 0.023,
            "honest_verdict": "probe_not_viable",
        }),
        "559": ("experiment_559_lowrank_kaem_calibration.json", {
            "status": "success",
            "energy_mad_at_optimal": kaem_energy_mad,
            "retro_057_closed": retro_057_closed,
            "duration_s": 63.17,
            "honest_verdict": "calibration_insufficient",
        }),
        "560": ("experiment_560_latent_cot_calibrator.json", {
            "status": "success",
            "inference_mode": "real_data_556",
            "violation_rate_delta": 0.0,
            "duration_s": 17.264,
            "honest_verdict": "calibration_neutral",
        }),
        "561": ("experiment_561_tier1_relay_real.json", {
            "status": "success",
            "inference_mode": "real_data",
            "fr11_real_data": fr11_real_data,
            "n_responses": 25,
            "constraints_added": [],
            "duration_s": 0.023,
            "honest_verdict": "real_data_no_improvement",
        }),
    }
    for _exp_id, (filename, data) in files.items():
        (results_dir / filename).write_text(json.dumps(data))

    return files


# ---------------------------------------------------------------------------
# compute_retro — baseline (partial outcome)
# ---------------------------------------------------------------------------


def test_compute_retro_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial scenario: live_50q_a blocked → synthetic_barrier_partial."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    assert result["honest_verdict"] == "synthetic_barrier_partial"
    assert result["exclusion_manifest_created"] is True
    assert result["live_50q_a_completed"] is False
    assert result["live_50q_b_completed"] is True
    assert result["fover_corpus_v2_n_labeled"] == 132
    assert result["fover_corpus_v2_entropy"] == pytest.approx(1.5019, abs=1e-4)
    assert result["fover_corpus_v2_ready"] is True
    assert result["retro_056_closed"] is False
    assert result["retro_057_closed"] is False
    assert result["retro_058_data_ready"] is True
    assert result["retro_059_resolved"] is True
    assert result["fr11_real_data_relay"] is True
    assert result["schema"] == SCHEMA
    assert result["milestone"] == MILESTONE


def test_compute_retro_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All three criteria met → synthetic_barrier_broken."""
    _minimal_results(tmp_path, live_a_status="success", n_labeled=100, entropy=1.5)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    assert result["honest_verdict"] == "synthetic_barrier_broken"
    assert result["live_50q_a_completed"] is True
    assert result["live_50q_b_completed"] is True
    assert result["fover_corpus_v2_n_labeled"] == 100


def test_compute_retro_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both live halves blocked and n_labeled < 100 → synthetic_barrier_intact."""
    _minimal_results(
        tmp_path, live_a_status="blocked", live_b_status="blocked", n_labeled=50
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    assert result["honest_verdict"] == "synthetic_barrier_intact"
    assert result["fover_corpus_v2_ready"] is False


# ---------------------------------------------------------------------------
# compute_retro — counts and wall time
# ---------------------------------------------------------------------------


def test_compute_retro_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_experiments=14 (13 upstream + this retro), n_deferred_to_gpu=1."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    assert result["n_experiments"] == 14
    assert result["n_timed_out"] == 0
    assert result["n_deferred_to_gpu"] == 1  # Exp 551 blocked by GPU gate
    assert result["n_missing"] == 0


def test_compute_retro_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Total wall time computed correctly from duration_s fields."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    # Expected: sum of all duration_s values in _minimal_results
    expected_durations = [
        0.069, 0.146, 0.0, 143.974, 0.011, 0.008, 0.001,
        175.138, 11.284, 0.023, 63.17, 17.264, 0.023,
    ]
    expected_total_min = round(sum(expected_durations) / 60.0, 3)
    assert result["total_wall_time_minutes"] == pytest.approx(expected_total_min, abs=0.01)
    assert result["average_minutes_per_experiment"] == pytest.approx(
        expected_total_min / 14.0, abs=0.01
    )


# ---------------------------------------------------------------------------
# compute_retro — RETRO closure rate
# ---------------------------------------------------------------------------


def test_compute_retro_closure_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retro_059 + retro_058 closed = 2 / 8 open at start = 0.25."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    # 2 closed (RETRO-059 via exclusion manifest, RETRO-058 via data_ready)
    # 8 open at milestone start
    assert result["retro_closure_rate"] == pytest.approx(2 / 8, abs=0.001)


def test_compute_retro_closure_rate_none_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If neither retro_058 nor retro_059 met, closure rate is 0."""
    _minimal_results(
        tmp_path, exclusion_manifest=False, retro_058_data_ready=False
    )
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    assert result["retro_closure_rate"] == 0.0


# ---------------------------------------------------------------------------
# compute_retro — top3 slowest
# ---------------------------------------------------------------------------


def test_compute_retro_top3_slowest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Top-3 slowest experiments are returned in descending duration order."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    top3 = result["top3_slowest_experiments"]
    assert len(top3) == 3
    # First entry should be the slowest (Exp 556: 175.138s)
    assert top3[0]["duration_s"] >= top3[1]["duration_s"] >= top3[2]["duration_s"]
    assert top3[0]["exp_id"] == "556"  # 175.138s
    assert top3[1]["exp_id"] == "552"  # 143.974s
    assert top3[2]["exp_id"] == "559"  # 63.17s


# ---------------------------------------------------------------------------
# compute_retro — structural validation of new_retro_items and open_retro_items
# ---------------------------------------------------------------------------


def test_new_retro_items_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each new RETRO item has id, title, description, priority, carry_count."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    for item in result["new_retro_items"]:
        assert "id" in item, f"Missing 'id' in {item}"
        assert "title" in item, f"Missing 'title' in {item}"
        assert "description" in item, f"Missing 'description' in {item}"
        assert "priority" in item, f"Missing 'priority' in {item}"
        assert item["priority"] in ("critical", "high", "medium", "low")
    # Expect exactly 3 new RETROs (RETRO-060, 061, 062)
    ids = [item["id"] for item in result["new_retro_items"]]
    assert "RETRO-060" in ids
    assert "RETRO-061" in ids
    assert "RETRO-062" in ids


def test_open_retro_items_carry_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carry-forward open RETROs include all unresolved items from prior milestone."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    open_ids = {item["id"] for item in result["open_retro_items"]}
    # RETRO-059 and RETRO-058 are closed this milestone — should NOT appear as open
    assert "RETRO-059" not in open_ids
    assert "RETRO-058" not in open_ids
    # These remain open
    assert "RETRO-033" in open_ids
    assert "RETRO-038" in open_ids
    assert "RETRO-056" in open_ids
    assert "RETRO-057" in open_ids
    # New items from this milestone
    assert "RETRO-060" in open_ids
    assert "RETRO-061" in open_ids
    assert "RETRO-062" in open_ids


# ---------------------------------------------------------------------------
# compute_retro — missing experiment files
# ---------------------------------------------------------------------------


def test_compute_retro_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing result files increment n_missing rather than crashing."""
    # Write no result files at all
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)

    result = compute_retro()

    assert result["n_missing"] == 13  # all 13 upstream experiments missing
    # Should still produce a valid structure
    assert result["honest_verdict"] == "synthetic_barrier_intact"
    assert result["exclusion_manifest_created"] is False


# ---------------------------------------------------------------------------
# main — artifact written with correct schema and all required fields
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = [
    "experiment",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "schema",
    "milestone",
    "n_experiments",
    "n_completed",
    "n_timed_out",
    "n_deferred_to_gpu",
    "n_missing",
    "total_wall_time_minutes",
    "average_minutes_per_experiment",
    "exclusion_manifest_created",
    "live_50q_a_completed",
    "live_50q_b_completed",
    "fover_corpus_v2_n_labeled",
    "fover_corpus_v2_entropy",
    "fover_corpus_v2_ready",
    "retro_056_closed",
    "retro_057_closed",
    "retro_058_data_ready",
    "retro_059_resolved",
    "retro_closure_rate",
    "fr11_real_data_relay",
    "jepa_v9_auc",
    "kaem_energy_mad_at_optimal",
    "top3_slowest_experiments",
    "headline_results",
    "new_retro_items",
    "open_retro_items",
    "meta_reflection",
    "honest_verdict",
    "env_autofix",
]


def test_main_writes_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes the deliverable JSON with all required schema fields."""
    _minimal_results(tmp_path)
    results_dir = tmp_path / "results"
    deliverable_path = results_dir / "experiment_562_retro_2026_04_42.json"

    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        retro_mod, "DELIVERABLE", "results/experiment_562_retro_2026_04_42.json"
    )

    retro_mod.main()

    assert deliverable_path.exists(), "Deliverable JSON was not written by main()"
    artifact = json.loads(deliverable_path.read_text())

    for field in _REQUIRED_FIELDS:
        assert field in artifact, f"Required field '{field}' missing from artifact"

    assert artifact["schema"] == SCHEMA
    assert artifact["milestone"] == MILESTONE
    assert artifact["experiment"] == EXP_ID
    assert artifact["status"] == "success"
    assert artifact["env_autofix"] is True


def test_main_schema_is_string_not_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """schema field must be the v17 string identifier, not the sorted-keys list
    that build_result() inserts by default."""
    _minimal_results(tmp_path)
    monkeypatch.setattr(retro_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        retro_mod, "DELIVERABLE", "results/experiment_562_retro_2026_04_42.json"
    )

    retro_mod.main()

    artifact = json.loads(
        (tmp_path / "results" / "experiment_562_retro_2026_04_42.json").read_text()
    )
    assert isinstance(artifact["schema"], str), "schema must be a string, not a list"
    assert artifact["schema"] == "carnot.operational_retro.v17"


# ---------------------------------------------------------------------------
# Module-level constant validation
# ---------------------------------------------------------------------------


def test_milestone_results_count() -> None:
    """_MILESTONE_RESULTS covers exactly 13 upstream experiments (549-561)."""
    exp_ids = [eid for eid, _ in _MILESTONE_RESULTS]
    assert len(exp_ids) == 13
    assert exp_ids[0] == "549"
    assert exp_ids[-1] == "561"


def test_retros_open_at_start_count() -> None:
    """8 RETROs were open at the start of milestone .42."""
    assert len(_RETROS_OPEN_AT_MILESTONE_START) == 8
    assert "RETRO-059" in _RETROS_OPEN_AT_MILESTONE_START
    assert "RETRO-058" in _RETROS_OPEN_AT_MILESTONE_START
    assert "RETRO-056" in _RETROS_OPEN_AT_MILESTONE_START
    assert "RETRO-057" in _RETROS_OPEN_AT_MILESTONE_START
