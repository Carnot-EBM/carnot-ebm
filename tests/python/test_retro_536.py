"""Tests for Exp 536 retrospective deliverable.

Verifies the .40 milestone retro JSON is structurally valid and
contains the required fields for schema carnot.operational_retro.v15.

Spec: REQ-INFRA-058, SCENARIO-INFRA-069
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DELIVERABLE = Path("results/experiment_536_retro_2026_04_40.json")
REQUIRED_FIELDS = [
    "experiment", "schema", "milestone", "title", "run_date",
    "started_at", "finished_at", "duration_s", "status",
    "retro_053_resolved", "retro_033_closed", "retro_038_closed",
    "gpu1_routing_verified", "tier0c_wired", "tier0d_wired",
    "eorm_rectification_positive", "lowrank_kaem_viable",
    "energy_steering_viable", "potts_viable", "fr11_live_relay",
    "n_experiments", "n_completed", "n_timed_out", "n_deferred_to_gpu",
    "n_missing", "total_wall_time_minutes", "average_minutes_per_experiment",
    "retro_closure_rate", "headline_results", "new_retro_items",
    "open_retro_items", "meta_reflection", "honest_verdict",
]


@pytest.fixture(scope="module")
def retro() -> dict:
    assert DELIVERABLE.exists(), f"Deliverable missing: {DELIVERABLE}"
    return json.loads(DELIVERABLE.read_text())


def test_schema(retro):
    assert retro["schema"] == "carnot.operational_retro.v15"


def test_milestone(retro):
    assert retro["milestone"] == "2026.04.40"


def test_experiment_id(retro):
    assert retro["experiment"] == 536


def test_required_fields_present(retro):
    missing = [f for f in REQUIRED_FIELDS if f not in retro]
    assert missing == [], f"Missing fields: {missing}"


def test_status_success(retro):
    assert retro["status"] == "success"


def test_retro_053_resolved(retro):
    # Exp 526 confirmed RETRO-053 fix
    assert retro["retro_053_resolved"] is True


def test_retro_033_still_open(retro):
    # Exp 527 timed out — 9th consecutive miss
    assert retro["retro_033_closed"] is False


def test_fr11_live_relay_achieved(retro):
    # Exp 535 achieved FR-11 live relay
    assert retro["fr11_live_relay"] is True


def test_tier_wiring(retro):
    assert retro["tier0c_wired"] is True
    assert retro["tier0d_wired"] is True


def test_lowrank_viable(retro):
    assert retro["lowrank_kaem_viable"] is True


def test_n_experiments(retro):
    assert retro["n_experiments"] == 11


def test_timed_out_count(retro):
    # Exp 527 timed out
    assert retro["n_timed_out"] == 1


def test_deferred_count(retro):
    # Exp 528 gpu_required
    assert retro["n_deferred_to_gpu"] == 1


def test_new_retro_items_contain_retro055(retro):
    ids = [item["id"] for item in retro["new_retro_items"]]
    assert "RETRO-055" in ids, f"RETRO-055 not in new_retro_items: {ids}"


def test_open_retro_items_not_empty(retro):
    assert len(retro["open_retro_items"]) >= 4


def test_meta_reflection_keys(retro):
    meta = retro["meta_reflection"]
    for key in ("top_3_bottlenecks", "top_3_improvements_for_41", "credibility_verdict",
                 "wall_time_note", "closures_achieved", "retro_033_miss_diagnosis"):
        assert key in meta, f"Missing meta key: {key}"


def test_honest_verdict(retro):
    assert retro["honest_verdict"] == "milestone_complete"


def test_headline_results_structure(retro):
    hr = retro["headline_results"]
    assert "live_100q_v8" in hr
    assert "live_200q_v7" in hr
    assert "jepa_live_retrain_v7" in hr
    assert hr["live_100q_v8"]["status"] == "timed_out"
