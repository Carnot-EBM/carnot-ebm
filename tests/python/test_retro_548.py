"""Tests for Exp 548 retrospective deliverable — Milestone 2026.04.41.

Verifies the retro JSON is structurally valid and contains
the required fields for schema carnot.operational_retro.v16.

Spec: REQ-INFRA-058, REQ-INFRA-076, SCENARIO-INFRA-069, SCENARIO-INFRA-075
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DELIVERABLE = Path("results/experiment_548_retro_2026_04_41.json")

# All fields required by the v16 schema.
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
    "retro_054_resolved",
    "retro_055_resolved",
    "retro_033_closed",
    "retro_038_closed",
    "fr11_live_relay",
    "jepa_v8_auc",
    "grpo_eorm_improved",
    "grpo_eorm_honest_verdict",
    "lowrank_kaem_wired",
    "lowrank_kaem_energy_within_5pct",
    "legacy_scripts_modernized",
    "n_experiments",
    "n_completed",
    "n_timed_out",
    "n_deferred_to_gpu",
    "n_missing",
    "total_wall_time_minutes",
    "average_minutes_per_experiment",
    "retro_closure_rate",
    "headline_results",
    "new_retro_items",
    "open_retro_items",
    "meta_reflection",
    "honest_verdict",
    "env_autofix",
]


@pytest.fixture(scope="module")
def retro() -> dict:
    assert DELIVERABLE.exists(), f"Deliverable missing: {DELIVERABLE}"
    return json.loads(DELIVERABLE.read_text())


def test_schema(retro: dict) -> None:
    assert retro["schema"] == "carnot.operational_retro.v16"


def test_milestone(retro: dict) -> None:
    assert retro["milestone"] == "2026.04.41"


def test_experiment_id(retro: dict) -> None:
    assert retro["experiment"] == 548


def test_status_success(retro: dict) -> None:
    assert retro["status"] == "success"


def test_required_fields_present(retro: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in retro]
    assert missing == [], f"Missing fields: {missing}"


def test_retro_054_resolved(retro: dict) -> None:
    # Exp 537 implemented teardown() + atexit registration, closing 5-milestone debt.
    assert retro["retro_054_resolved"] is True


def test_retro_055_resolved(retro: dict) -> None:
    # Exp 538 confirmed env_autofix value-check fix working in live_gpu mode.
    assert retro["retro_055_resolved"] is True


def test_retro_033_still_open(retro: dict) -> None:
    # Attempt #10 via Exp 538 — signed_improvement=0.0, not closed.
    assert retro["retro_033_closed"] is False


def test_retro_038_still_open(retro: dict) -> None:
    # Attempt #8 via Exp 539 — signed_improvement=0.0, Wilson CI spans zero.
    assert retro["retro_038_closed"] is False


def test_fr11_live_relay_not_achieved(retro: dict) -> None:
    # Exp 543 ran but final_auc=0.444 (below random) — FR-11 not satisfied.
    assert retro["fr11_live_relay"] is False


def test_jepa_v8_auc_below_random(retro: dict) -> None:
    # AUC < 0.5 means the verifier is anti-correlated with correctness labels.
    assert retro["jepa_v8_auc"] < 0.5


def test_lowrank_kaem_wired_but_inaccurate(retro: dict) -> None:
    # Wired as default tier but energy accuracy failed the 5% tolerance test.
    assert retro["lowrank_kaem_wired"] is True
    assert retro["lowrank_kaem_energy_within_5pct"] is False


def test_legacy_scripts_modernized_count(retro: dict) -> None:
    # Exp 547 audited 5 scripts; all 5 were already fully_modern.
    assert retro["legacy_scripts_modernized"] == 5


def test_experiment_counts(retro: dict) -> None:
    # All 11 milestone experiments (537-547) completed successfully.
    assert retro["n_experiments"] == 11
    assert retro["n_completed"] == 11
    assert retro["n_timed_out"] == 0
    assert retro["n_deferred_to_gpu"] == 0
    assert retro["n_missing"] == 0


def test_wall_time_positive(retro: dict) -> None:
    assert retro["total_wall_time_minutes"] > 0
    assert retro["average_minutes_per_experiment"] > 0


def test_retro_closure_rate_range(retro: dict) -> None:
    # 1 of 5 pre-existing RETROs closed (RETRO-054) => 0.2
    rate = retro["retro_closure_rate"]
    assert 0.0 <= rate <= 1.0
    assert abs(rate - 0.2) < 0.01


def test_headline_results_is_dict(retro: dict) -> None:
    assert isinstance(retro["headline_results"], dict)
    assert len(retro["headline_results"]) == 11  # one per experiment 537-547


def test_new_retro_items_count(retro: dict) -> None:
    # RETRO-056 through RETRO-059 were opened this milestone.
    assert len(retro["new_retro_items"]) >= 4


def test_open_retro_items_has_retro_033(retro: dict) -> None:
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-033" in ids


def test_open_retro_items_has_retro_038(retro: dict) -> None:
    ids = [item["id"] for item in retro["open_retro_items"]]
    assert "RETRO-038" in ids


def test_meta_reflection_structure(retro: dict) -> None:
    mr = retro["meta_reflection"]
    assert "top_3_bottlenecks" in mr
    assert "top_3_improvements_for_42" in mr
    assert "credibility_verdict" in mr
    assert "wall_time_42_estimate_minutes" in mr
    assert len(mr["top_3_bottlenecks"]) == 3
    assert len(mr["top_3_improvements_for_42"]) == 3


def test_honest_verdict_milestone_partial(retro: dict) -> None:
    # RETRO-054 closed but RETRO-033/038 remain open; no publishable headline result.
    assert retro["honest_verdict"] == "milestone_partial"


def test_env_autofix_applied(retro: dict) -> None:
    assert retro["env_autofix"] is True
