"""Tests for Exp 689 retrospective deliverable — Milestone 2026.04.52 (Exps 678-688).

Verifies the retro JSON is structurally valid and contains the required fields
for schema carnot.operational_retro.v27.  Checks that key open questions are
answered honestly: VR win at 200q, RETRO-071 closure, manifest confirmation,
FR-11 wiring, and dual-GPU speedup measurement.

Spec: REQ-INFRA-058, REQ-INFRA-076, REQ-VERIFY-083,
      SCENARIO-INFRA-069, SCENARIO-INFRA-075
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DELIVERABLE = Path("results/operational_retro_2026_04_52.json")

# All fields required by the v27 schema for this retrospective.
REQUIRED_FIELDS = [
    "experiment",
    "schema_version",
    "milestone",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "honest_verdict",
    "experiment_table",
    "cycle_duration_s",
    "cycle_wall_time_minutes",
    "total_wall_time_minutes",
    "experiments_completed",
    "cycle_experiments",
    "avg_time_per_experiment_minutes",
    "slowest_5",
    "manifest_consulted",
    "vr_200q_signed_improvement",
    "vr_200q_validated",
    "humaneval_vr_improvement",
    "adversarial_vr_robust",
    "jepa_v15_true_ood_auc",
    "fr11_real_positives_wired",
    "retro_071_resolved",
    "dualgpu_retrain_speedup",
    "fover_formal_v1_n_labels",
    "psv_iterations_completed",
    "retro_071_status",
    "manifest_status",
    "vr_win_validation",
    "wall_time_direction",
    "gpu_state_at_close",
    "milestone_history",
]


@pytest.fixture(scope="module")
def retro() -> dict:
    """Load the retrospective deliverable JSON."""
    assert DELIVERABLE.exists(), f"Deliverable missing: {DELIVERABLE}"
    return json.loads(DELIVERABLE.read_text())


def test_schema_version(retro: dict) -> None:
    # v27 is the schema version for milestone 2026.04.52 retrospectives.
    # build_result() sets "schema" to a sorted key list; version is in "schema_version".
    assert retro["schema_version"] == "carnot.operational_retro.v27"


def test_milestone_identifier(retro: dict) -> None:
    assert retro["milestone"] == "2026.04.52"


def test_experiment_id(retro: dict) -> None:
    assert retro["experiment"] == 689


def test_status_success(retro: dict) -> None:
    assert retro["status"] == "success"


def test_required_fields_present(retro: dict) -> None:
    # Every field in REQUIRED_FIELDS must appear in the artifact.
    missing = [f for f in REQUIRED_FIELDS if f not in retro]
    assert missing == [], f"Missing required fields: {missing}"


def test_vr_200q_win_confirmed(retro: dict) -> None:
    # Exp 679 reported signed_improvement=1.0 — the VR win at 200q is validated.
    assert retro["vr_200q_signed_improvement"] > 0
    assert retro["vr_200q_validated"] is True
    assert retro["vr_win_validation"] == "confirmed"


def test_retro_071_closed(retro: dict) -> None:
    # Exp 684 and 685 both confirmed retro_071_resolved=True — finally closed.
    assert retro["retro_071_resolved"] is True
    assert retro["retro_071_status"] == "closed"


def test_manifest_consulted(retro: dict) -> None:
    # Exp 678 confirmed conductor_consulted=True — manifest check is wired.
    assert retro["manifest_consulted"] is True
    assert retro["manifest_status"] == "confirmed"


def test_fr11_real_positives_wired(retro: dict) -> None:
    # Exp 683 confirmed fr11_real_positives_confirmed=True.
    assert retro["fr11_real_positives_wired"] is True


def test_dualgpu_speedup_positive(retro: dict) -> None:
    # Exp 685 measured speedup=2.0175 — dual-GPU retraining is faster than sequential.
    assert retro["dualgpu_retrain_speedup"] > 1.0


def test_jepa_ood_auc_below_random(retro: dict) -> None:
    # Exp 682 returned true_ood_auc=0.4751 — JEPA v1.5 OOD detection still below random.
    assert retro["jepa_v15_true_ood_auc"] < 0.5


def test_psv_iterations_completed(retro: dict) -> None:
    # Exp 688 completed 10 PSV self-play iterations.
    assert retro["psv_iterations_completed"] == 10


def test_experiment_table_has_all_11(retro: dict) -> None:
    # The table must account for all 11 experiments 678-688 including the missing 686.
    table = retro["experiment_table"]
    assert len(table) == 11
    ids = {e["experiment"] for e in table}
    assert ids == {678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688}


def test_exp_686_not_run(retro: dict) -> None:
    # Exp 686 result file is absent — should be recorded as not_run (RETRO-027 sentinel).
    table = retro["experiment_table"]
    exp686 = next(e for e in table if e["experiment"] == 686)
    assert exp686["status"] == "not_run"


def test_slowest_5_present(retro: dict) -> None:
    # Must identify the 5 slowest experiments in this cycle.
    slowest = retro["slowest_5"]
    assert len(slowest) == 5
    for entry in slowest:
        assert "experiment" in entry
        assert "duration_s" in entry


def test_wall_time_accounting(retro: dict) -> None:
    # Cumulative wall-time must exceed the .51 baseline of 4231 minutes.
    assert retro["total_wall_time_minutes"] >= 4231
    assert retro["experiments_completed"] > 532


def test_milestone_history_includes_52(retro: dict) -> None:
    # The history list must include the new 2026.04.52 entry.
    history = retro["milestone_history"]
    milestones = [h["milestone"] for h in history]
    assert "2026.04.52" in milestones
    assert "2026.04.51" in milestones


def test_gpu_state_at_close_fields(retro: dict) -> None:
    # GPU state must include VRAM readings for both GPUs.
    gpu = retro["gpu_state_at_close"]
    assert "gpu0_vram_mb" in gpu
    assert "gpu1_vram_mb" in gpu
    assert "gpu_close_clean" in gpu


def test_honest_verdict_nonempty(retro: dict) -> None:
    # The honest_verdict must be a non-empty string encoding key outcomes.
    verdict = retro["honest_verdict"]
    assert isinstance(verdict, str)
    assert len(verdict) > 10
    assert "vr_" in verdict
    assert "retro071_" in verdict
    assert "manifest_" in verdict
