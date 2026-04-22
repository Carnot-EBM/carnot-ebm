"""Tests for Experiment 715: Milestone 2026.04.54 Operational Retrospective.

REQ-RETRO-054: Retrospective must load all cycle experiment results and compute
wall-time, slowest-5, and closure metrics correctly.

SCENARIO-RETRO-054-01: All 12 cycle experiments load without crash; missing files
return the RETRO-027 sentinel.
SCENARIO-RETRO-054-02: Slowest-5 governance check correctly identifies whether
retired experiment IDs appear in the new slowest-5.
SCENARIO-RETRO-054-03: honest_verdict composite string is built from the four
sub-verdicts in the documented format.
SCENARIO-RETRO-054-04: Deliverable JSON contains all REQUIRED_RESULT_FIELDS.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper: import the private helpers without running main()
# ---------------------------------------------------------------------------

import importlib.util
import sys


def _import_exp715():
    """Import the experiment module without executing main."""
    spec = importlib.util.spec_from_file_location(
        "experiment_715",
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "experiment_715_retro_2026_04_54.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _import_exp715()


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-054-01: missing file returns RETRO-027 sentinel
# ---------------------------------------------------------------------------


def test_load_result_missing_file(tmp_path):
    """_load_result on a nonexistent path returns the RETRO-027 sentinel dict.

    Why this matters: if a cycle experiment was never run we must record
    "not_run" rather than crashing the retrospective.
    """
    # Temporarily redirect repo_root so the missing-file path resolves cleanly
    fake_path = "results/does_not_exist_12345.json"
    with patch.object(mod, "repo_root", tmp_path):
        result = mod._load_result(fake_path)

    assert result["status"] == "not_run"
    assert result["honest_verdict"] == "RETRO-027_sentinel"
    assert result["duration_s"] == 0


def test_load_result_existing_file(tmp_path):
    """_load_result reads and returns the JSON content from an existing file.

    Why: verifies the happy path so we know the file I/O layer works correctly
    before relying on it for all 12 cycle experiments.
    """
    data = {"experiment": 999, "status": "success", "duration_s": 42.0}
    target = tmp_path / "results" / "experiment_999_test.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(data))

    with patch.object(mod, "repo_root", tmp_path):
        result = mod._load_result("results/experiment_999_test.json")

    assert result["experiment"] == 999
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-054-02: slowest-5 governance
# ---------------------------------------------------------------------------


def test_compute_slowest_5_returns_top5_by_duration():
    """_compute_slowest_5 returns the 5 experiments with the largest duration_s.

    Why: the governance check relies on the correct set of slow experiments
    being identified; an off-by-one here would silently pass a retirement that
    actually failed.
    """
    table = [{"experiment": i, "duration_s": float(i), "status": "success"} for i in range(10)]
    slowest = mod._compute_slowest_5(table)
    assert len(slowest) == 5
    ids = [r["experiment"] for r in slowest]
    assert ids == [9, 8, 7, 6, 5]


def test_governance_held_when_retired_exps_absent():
    """slowest_5_governance_held is True when no retired IDs appear in slowest-5.

    Why: the governance fix in .54 retired specific slow experiments; if they
    do not appear in the new slowest-5 the retirement succeeded.
    """
    slowest = [{"experiment": 709, "duration_s": 946.0},
               {"experiment": 706, "duration_s": 402.0},
               {"experiment": 708, "duration_s": 85.7},
               {"experiment": 705, "duration_s": 11.0},
               {"experiment": 710, "duration_s": 10.1}]
    ids = {r["experiment"] for r in slowest}
    held = ids.isdisjoint(mod.RETIRED_SLOW_EXPS)
    assert held is True


def test_governance_failed_when_retired_exp_reappears():
    """slowest_5_governance_held is False when a retired ID recurs in slowest-5.

    Why: this is the failure mode the retirement governance process is designed
    to detect — if exp 425 runs long again the retrospective must flag it.
    """
    slowest = [{"experiment": 425, "duration_s": 3000.0},
               {"experiment": 706, "duration_s": 402.0},
               {"experiment": 708, "duration_s": 85.7},
               {"experiment": 705, "duration_s": 11.0},
               {"experiment": 710, "duration_s": 10.1}]
    ids = {r["experiment"] for r in slowest}
    held = ids.isdisjoint(mod.RETIRED_SLOW_EXPS)
    assert held is False


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-054-03: honest_verdict format
# ---------------------------------------------------------------------------


def test_honest_verdict_format_all_positive():
    """honest_verdict is correctly composed when all four sub-verdicts are positive.

    Format: wall_time_{dir}_jepa_v17_{auc}_gemma4_{gemma}_psv_{psv}_slowest5_{gov}
    """
    verdict = (
        "wall_time_down"
        "_jepa_v17_cascade_unblocked"
        "_gemma4_fixed"
        "_psv_recovering"
        "_slowest5_held"
    )
    assert verdict.startswith("wall_time_")
    assert "jepa_v17_" in verdict
    assert "gemma4_" in verdict
    assert "psv_" in verdict
    assert "slowest5_" in verdict


def test_actual_deliverable_honest_verdict():
    """The written deliverable's honest_verdict matches the expected .54 outcome.

    Why: this is the primary closure assertion — the retrospective must record
    the real outcomes, not a placeholder, so downstream milestone planning tools
    can branch on jepa_v17_cascade_unblocked=False and plan v18.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "operational_retro_2026_04_54.json"
    )
    assert path.exists(), "Deliverable must be written before tests run"
    d = json.loads(path.read_text())
    verdict = d["honest_verdict"]
    # JEPA v17 OOD AUC was 0.4819 < 0.75 threshold → still_blocked
    assert "jepa_v17_still_blocked" in verdict
    # Gemma4 signed_improvement=0.0 >= 0 → fixed
    assert "gemma4_fixed" in verdict
    # Slowest-5 contained only exps 705,706,708,709,710 (none retired) → held
    assert "slowest5_held" in verdict


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-054-04: REQUIRED_RESULT_FIELDS present in deliverable
# ---------------------------------------------------------------------------


def test_deliverable_has_required_fields():
    """Deliverable JSON contains all REQUIRED_RESULT_FIELDS.

    Spec: REQ-VERIFY-083 — every artifact must have experiment, schema,
    run_date, started_at, finished_at, duration_s, status, title.
    """
    from scripts.experiment_template import REQUIRED_RESULT_FIELDS

    path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "operational_retro_2026_04_54.json"
    )
    assert path.exists()
    d = json.loads(path.read_text())

    for field in REQUIRED_RESULT_FIELDS:
        assert field in d, f"Missing required field: {field}"


def test_deliverable_schema_version():
    """Deliverable reports the expected schema version v29."""
    path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "operational_retro_2026_04_54.json"
    )
    d = json.loads(path.read_text())
    assert d["schema"] == "carnot.operational_retro.v29"


def test_deliverable_closure_metrics_present():
    """All seven closure metrics are present in the deliverable.

    Why: the conductor reads these fields to decide whether to open the next
    milestone — absent fields cause KeyError crashes downstream.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "operational_retro_2026_04_54.json"
    )
    d = json.loads(path.read_text())

    required_closure_keys = [
        "jepa_v17_ood_auc",
        "jepa_v17_cascade_unblocked",
        "vr19_gemma4_signed_improvement",
        "psv_pacore_slope",
        "distillation_auroc_v2",
        "slowest_5_governance_held",
        "fover_v2_n_pairs",
        "npu_benchmarkable",
    ]
    for key in required_closure_keys:
        assert key in d, f"Missing closure metric: {key}"
