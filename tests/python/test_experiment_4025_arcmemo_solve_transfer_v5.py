"""Tests for Exp 4025 ArcMemo solve-transfer v5.

Spec refs: REQ-PHASE4-033, SCENARIO-PHASE4-033.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot.agentic.arc_arcmemo_solve_transfer_v5 import (
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_transfer_artifact,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4025_arcmemo_solve_transfer_v5 as exp  # noqa: E402


def _exp4021_payload() -> dict[str, object]:
    return {
        "experiment": "experiment_4021_heuristic_search_over_verified_wm",
        "honest_verdict": "complete: search_layer_solved_r11l_L4_real_env_confirmed",
        "new_levels_solved_this_task": 1,
        "real_env_confirmed": True,
        "executed_real_env_actions": 16,
        "nodes_expanded": 3,
        "model_reuse_note": "reused offline verified env-copy simulator plus Exp 4020 sandboxed is_goal; no new induction",
        "inference_substrate": "offline_arc_agi3_verified_env_copy_simulator_exp4020_goal_predicate_coded_heuristic_mpc_replanning",
    }


def _exp4024_payload(*, solved: bool = True, baseline: int = 55, seeded: int = 5) -> dict[str, object]:
    return {
        "experiment": "experiment_4024_fifth_game_explore_first",
        "honest_verdict": (
            "success: fifth_game_solved_cd82-fb555c5d_at_action_5"
            if solved
            else "complete: fifth_game_no_solve_cd82-fb555c5d_level_counter_did_not_increment"
        ),
        "game_solved": solved,
        "real_env_confirmed": solved,
        "candidate_baseline_actions": baseline,
        "first_solve_at_action": seeded if solved else -1,
        "target_game": "cd82-fb555c5d",
        "inference_substrate": "offline_arc_agi3_cd82_explore_first_region_fill_induction",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_req_phase4_033_spec_declares_exp4025_contract() -> None:
    """REQ-PHASE4-033: OpenSpec declares Exp 4025 and required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-033" in spec
    assert "SCENARIO-PHASE4-033" in spec
    assert "experiment_4025_arcmemo_solve_transfer_v5.json" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_033_summed_seeded_cost_is_cheaper() -> None:
    """SCENARIO-PHASE4-033: the seeded aggregate must reduce future solve cost."""

    artifact = build_transfer_artifact(
        exp4021=_exp4021_payload(),
        exp4024=_exp4024_payload(),
        duration_s=0.25,
    )

    assert artifact["solve_transfer_win"] is True
    assert artifact["actions_cold"] == 71
    assert artifact["actions_seeded"] == 21
    assert artifact["induction_calls_cold"] == 2
    assert artifact["induction_calls_seeded"] == 1
    assert artifact["honest_verdict"] == "success: arcmemo_v5_transfer_71to21_actions"
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact_schema_errors(artifact) == []

    per_content = {row["content_id"]: row for row in artifact["per_content_costs"]}
    assert per_content["exp4021"]["actions_cold"] == 16
    assert per_content["exp4021"]["actions_seeded"] == 16
    assert per_content["exp4024"]["actions_cold"] == 55
    assert per_content["exp4024"]["actions_seeded"] == 5


def test_scenario_phase4_033_no_transfer_when_upstream_missing_or_not_cheaper() -> None:
    """SCENARIO-PHASE4-033: missing or equal-cost upstream evidence cannot claim transfer."""

    exp4021_without_memory_reuse = dict(_exp4021_payload())
    exp4021_without_memory_reuse["model_reuse_note"] = "fresh induction required"
    missing = build_transfer_artifact(exp4021=None, exp4024=_exp4024_payload(), duration_s=0.1)
    unconfirmed_4021 = dict(_exp4021_payload())
    unconfirmed_4021["real_env_confirmed"] = False
    unconfirmed_4024 = build_transfer_artifact(
        exp4021=_exp4021_payload(),
        exp4024=_exp4024_payload(solved=False),
        duration_s=0.1,
    )
    equal_cost = build_transfer_artifact(
        exp4021=exp4021_without_memory_reuse,
        exp4024=_exp4024_payload(baseline=5, seeded=5),
        duration_s=0.1,
    )
    no_action_key = dict(_exp4021_payload())
    no_action_key.pop("executed_real_env_actions")
    no_action_key.pop("action_count", None)
    zero_action_fallback = build_transfer_artifact(
        exp4021=no_action_key,
        exp4024=_exp4024_payload(),
        duration_s=0.1,
    )

    assert missing["solve_transfer_win"] is False
    assert missing["honest_verdict"] == "complete: arcmemo_v5_no_transfer_missing_upstream_artifacts"
    assert missing["actions_cold"] == 0
    assert missing["actions_seeded"] == 0
    assert artifact_schema_errors(missing) == []

    unconfirmed = build_transfer_artifact(
        exp4021=unconfirmed_4021,
        exp4024=_exp4024_payload(),
        duration_s=0.1,
    )
    assert unconfirmed["honest_verdict"] == "complete: arcmemo_v5_no_transfer_upstream_not_real_env_confirmed"
    assert unconfirmed_4024["honest_verdict"] == (
        "complete: arcmemo_v5_no_transfer_upstream_not_real_env_confirmed"
    )
    assert equal_cost["solve_transfer_win"] is False
    assert equal_cost["actions_cold"] == 21
    assert equal_cost["actions_seeded"] == 21
    assert equal_cost["honest_verdict"] == "complete: arcmemo_v5_no_transfer_seeded_not_cheaper"
    assert artifact_schema_errors(equal_cost) == []
    assert zero_action_fallback["per_content_costs"][0]["actions_cold"] == 0


def test_req_phase4_033_schema_rejects_non_bare_required_fields() -> None:
    """REQ-PHASE4-033: required fields stay bare JSON scalars."""

    artifact = build_transfer_artifact(
        exp4021=_exp4021_payload(),
        exp4024=_exp4024_payload(),
        duration_s=0.25,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "finished"
    bad["solve_transfer_win"] = 1
    bad["actions_cold"] = "71"
    bad["actions_seeded"] = 21.0
    bad["inference_substrate"] = None

    errors = artifact_schema_errors(bad)
    missing = artifact_schema_errors({})

    assert any("honest_verdict" in error for error in errors)
    assert any("solve_transfer_win" in error for error in errors)
    assert any("actions_cold" in error for error in errors)
    assert any("actions_seeded" in error for error in errors)
    assert any("inference_substrate" in error for error in errors)
    assert any("missing required field honest_verdict" in error for error in missing)


def test_runner_writes_exp4025_result_json(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-033: runner writes the stable exp4025 JSON deliverable."""

    _write_json(tmp_path / "results" / "experiment_4021_heuristic_search_over_verified_wm.json", _exp4021_payload())
    _write_json(tmp_path / "results" / "experiment_4024_fifth_game_explore_first.json", _exp4024_payload())
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    written = tmp_path / "results" / "experiment_4025_arcmemo_solve_transfer_v5.json"
    assert artifact["honest_verdict"] == "success: arcmemo_v5_transfer_71to21_actions"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
