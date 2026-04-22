"""Tests for Exp 702 retrospective deliverable — Milestone 2026.04.53 (Exps 690-701).

Verifies the retro JSON is structurally valid and answers the four key questions
the milestone was chartered to resolve:
  1. JEPA v16 OOD AUC threshold (cascade unblock check).
  2. PSV real self-play FP trend (improving vs degrading).
  3. KAN distillation duration invariant (teacher_s >= corpus*0.5).
  4. VR cross-model delta vs Exp 679 baseline.

Spec: REQ-INFRA-058, REQ-INFRA-076, REQ-VERIFY-083,
      SCENARIO-INFRA-069, SCENARIO-INFRA-075
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DELIVERABLE = Path("results/operational_retro_2026_04_53.json")

# Minimum required top-level keys that every v28 retro must contain.
REQUIRED_FIELDS = [
    "experiment",
    "schema",
    "milestone",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "honest_verdict",
    "cycle_data",
    "total_wall_time_minutes",
    "experiments_completed",
    "avg_time_per_experiment_minutes",
    "slowest_experiments",
    "key_metrics",
    "open_retros",
    "gpu_state_at_close",
    "fr11_continuous_learning",
]


@pytest.fixture(scope="module")
def retro() -> dict:
    """Load the retrospective deliverable JSON."""
    assert DELIVERABLE.exists(), f"Deliverable missing: {DELIVERABLE}"
    return json.loads(DELIVERABLE.read_text())


def test_schema_version(retro: dict) -> None:
    # v28 is the schema version for milestone 2026.04.53 retrospectives.
    assert retro["schema"] == "carnot.operational_retro.v28"


def test_milestone_identifier(retro: dict) -> None:
    assert retro["milestone"] == "2026.04.53"


def test_experiment_id(retro: dict) -> None:
    assert retro["experiment"] == 702


def test_status_success(retro: dict) -> None:
    assert retro["status"] == "success"


def test_required_fields_present(retro: dict) -> None:
    # Every field in REQUIRED_FIELDS must appear in the artifact.
    missing = [f for f in REQUIRED_FIELDS if f not in retro]
    assert missing == [], f"Missing required fields: {missing}"


def test_cycle_covers_all_12_experiments(retro: dict) -> None:
    # Cycle spans Exps 690-701 — 12 experiments exactly.
    table = retro["cycle_data"]["experiment_table"]
    assert len(table) == 12
    ids = {e["experiment"] for e in table}
    assert ids == set(range(690, 702))


def test_jepa_v16_ood_auc_recorded(retro: dict) -> None:
    # JEPA v16 OOD AUC from Exp 698 must be present and non-negative.
    auc = retro["key_metrics"]["jepa_v16_ood_auc"]
    assert isinstance(auc, float)
    assert auc >= 0.0


def test_jepa_cascade_still_blocked(retro: dict) -> None:
    # Exp 698 measured v16_ood_auc=0.4759 — below the 0.75 unblock threshold.
    assert retro["key_metrics"]["jepa_v16_ood_auc"] < 0.75
    assert retro["fr11_continuous_learning"]["jepa_v16_cascade_unblocked"] is False
    assert retro["open_retros"]["RETRO-CRITICAL"]["status"] == "open"


def test_psv_fp_trend_recorded(retro: dict) -> None:
    # PSV real self-play FP trend slope from Exp 697 must be present.
    slope = retro["key_metrics"]["psv_real_fp_trend_slope"]
    assert isinstance(slope, float)


def test_psv_fp_degrading(retro: dict) -> None:
    # Exp 697 measured slope=0.004242 (positive = degrading, not improving).
    assert retro["key_metrics"]["psv_real_fp_trend_slope"] > 0
    assert retro["fr11_continuous_learning"]["psv_real_fp_improving"] is False


def test_distillation_invariant_confirmed(retro: dict) -> None:
    # Exp 690 had teacher_inference_duration_s=6256.2, corpus_size=200.
    # 6256.2 >= 200*0.5 = 100 → invariant holds.
    assert retro["key_metrics"]["distillation_invariant_confirmed"] is True
    assert retro["open_retros"]["RETRO-DISTILLATION"]["status"] == "confirmed"


def test_cross_dataset_auroc_high(retro: dict) -> None:
    # Exp 691 mean_auroc=0.9585 — well above the 0.75 publishable threshold.
    assert retro["key_metrics"]["cross_dataset_mean_auroc"] > 0.9


def test_vr_cross_model_delta_present(retro: dict) -> None:
    # Exp 694 cross_model_delta=-1.8; must be recorded in key_metrics.
    delta = retro["key_metrics"]["vr_cross_model_delta"]
    assert delta is not None


def test_publication_ready(retro: dict) -> None:
    assert retro["key_metrics"]["publication_ready"] is True


def test_retro_072_still_open(retro: dict) -> None:
    # Exp 701 was blocked — Vivado not installed; RETRO-072 remains open.
    assert retro["open_retros"]["RETRO-072"]["status"] == "open"
    assert retro["key_metrics"]["retro_072_resolved"] is False


def test_slowest_5_present(retro: dict) -> None:
    slowest = retro["slowest_experiments"]
    assert len(slowest) == 5
    for entry in slowest:
        assert "experiment" in entry
        assert "duration_s" in entry


def test_gpu_state_fields(retro: dict) -> None:
    gpu = retro["gpu_state_at_close"]
    assert "gpu0_vram_mb" in gpu
    assert "gpu1_vram_mb" in gpu
    assert "gpu_close_clean" in gpu


def test_honest_verdict_encodes_outcomes(retro: dict) -> None:
    # The verdict string must encode the four key research decisions.
    verdict = retro["honest_verdict"]
    assert isinstance(verdict, str)
    assert len(verdict) > 10
    # Must reflect JEPA still blocked
    assert "still_blocked" in verdict
    # Must reflect distillation confirmed
    assert "confirmed" in verdict


def test_wall_time_accounting(retro: dict) -> None:
    # Cumulative completed experiments must exceed .52 baseline of 538.
    assert retro["experiments_completed"] > 538
    assert retro["total_wall_time_minutes"] > 0
