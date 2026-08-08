"""Tests for Exp6215 object-relative trajectory transfer A/B.

Spec refs: REQ-ARC-WMTE-6215,
SCENARIO-ARC-WMTE-6215-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6215-WITHIN-GAME-ONLY,
SCENARIO-ARC-WMTE-6215-CANONICAL-LIVE-AGENT,
SCENARIO-ARC-WMTE-6215-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import os
from pathlib import Path

from carnot import experiment_6215_arc_object_relative_trajectory_transfer_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _mutation_receipts() -> list[dict[str, object]]:
    return [
        {"name": "transfer_fire_counter_removed", "killed": True},
        {"name": "fallback_avoidance_counter_removed", "killed": True},
        {"name": "forbidden_access_guard_removed", "killed": True},
    ]


def test_req_arc_wmte_6215_spec_declares_fields_and_scenarios() -> None:
    """REQ-ARC-WMTE-6215: OpenSpec names the artifact and scenarios."""

    section = "REQ-ARC-WMTE-6215" + SPEC.read_text(encoding="utf-8").split(
        "### REQ-ARC-WMTE-6215", 1
    )[1]

    for marker in (
        "REQ-ARC-WMTE-6215",
        "SCENARIO-ARC-WMTE-6215-REGISTRY-PRECHECK",
        "SCENARIO-ARC-WMTE-6215-WITHIN-GAME-ONLY",
        "SCENARIO-ARC-WMTE-6215-CANONICAL-LIVE-AGENT",
        "SCENARIO-ARC-WMTE-6215-ARTIFACT-GUARDS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6215_live_agent_treatment_avoids_fallback(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6215-CANONICAL-LIVE-AGENT: transfer fires before LLM fallback."""

    cell = mod.fixture_level_boundary_cell("ls20", 621500, boundary=1)
    pair = mod.run_matched_live_cell(cell, raw_root=tmp_path)

    control = pair["arms"]["control"]
    treatment = pair["arms"]["treatment"]
    assert pair["within_game_only"] is True
    assert control["llm_induction_calls"] == 1
    assert treatment["llm_induction_calls"] == 0
    assert pair["avoided_llm_induction_calls"] == 1
    assert treatment["engine_source"] == "object_relative_trajectory_transfer"
    assert treatment["trajectory_transfer"]["transfer_confident"] is True
    assert treatment["trajectory_transfer"]["matched_pairs"] > 0
    assert treatment["score"] == control["score"]
    for receipt in pair["raw_event_paths_and_hashes"]:
        assert Path(receipt["path"]).is_file()
        assert str(receipt["sha256"]).startswith("sha256:")


def test_scenario_arc_wmte_6215_artifact_guards_and_required_fields(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6215-ARTIFACT-GUARDS: no solve or registry credit."""

    artifact = mod.build_artifact(
        date="20260808",
        games=("ls20", "s5i5", "tu93"),
        seeds=(621500,),
        raw_root=tmp_path,
        mutation_receipts=_mutation_receipts(),
        test_commands=["unit-fixture"],
        test_exit_codes={"unit-fixture": 0},
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_ready"
    assert "solve_provenance" not in artifact
    assert artifact["solve_claimed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["level_credit_delta"] == 0
    assert artifact["registry_update_count"] == 0
    assert artifact["treatment_fire_and_reason_counts"]["total"] >= mod.SUPPORT_FLOOR
    assert artifact["avoided_llm_induction_calls"]["total"] >= mod.SUPPORT_FLOOR
    assert artifact["ab_complete_score"] == 1.0
    assert artifact["trajectory_transfer_promotion_ready_score"] == 1.0
    assert artifact["inference_substrate"]["legacy_models_contributed_rows"] == 0
    assert artifact["model_specs"][0]["hf_id"] == mod.CANONICAL_MODEL_HF_ID
    assert all(
        type(value) is int and value == 0
        for value in artifact[
            "prior_game_cross_game_source_bfs_adapter_registry_hidden_state_access_counts"
        ].values()
    )
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_arc_wmte_6215_zero_fire_is_instrument_failure(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6215-CANONICAL-LIVE-AGENT: zero fire is not a null."""

    artifact = mod.build_artifact(
        date="20260808",
        games=("ls20", "s5i5", "tu93"),
        seeds=(621500,),
        raw_root=tmp_path,
        mutation_receipts=_mutation_receipts(),
        force_zero_treatment_fire=True,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "instrument_failure_zero_treatment_fire"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["trajectory_transfer_promotion_ready_score"] < 1.0


def test_req_arc_wmte_6215_defensive_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6215: branch guards stay explicit and deterministic."""

    os.environ["EXP6215_TEMP_RESTORE"] = "old"
    with mod._temporary_env({"EXP6215_TEMP_RESTORE": "new"}):
        assert os.environ["EXP6215_TEMP_RESTORE"] == "new"
    assert os.environ["EXP6215_TEMP_RESTORE"] == "old"
    os.environ.pop("EXP6215_TEMP_RESTORE", None)

    assert mod.paired_clustered_intervals({})["n_games"] == 0
    assert mod._ready_score([]) == 0.0
    assert (
        mod.classify_status(
            {
                "total": 1,
                "accepted": 1,
                "support_count": 1,
                "support_floor": 3,
                "mutation_proven": True,
            }
        )
        == "instrument_failure_support_floor"
    )
    assert (
        mod.classify_status(
            {
                "total": 3,
                "accepted": 3,
                "support_count": 3,
                "support_floor": 3,
                "mutation_proven": False,
            }
        )
        == "instrument_failure_mutation_not_killed"
    )
    assert (
        mod.classify_status(
            {
                "total": 3,
                "accepted": 2,
                "support_count": 3,
                "support_floor": 3,
                "mutation_proven": True,
            }
        )
        == "instrument_failure_verifier_acceptance_floor"
    )
    assert (
        mod.classify_status(
            {
                "total": 3,
                "accepted": 3,
                "support_count": 3,
                "support_floor": 3,
                "mutation_proven": True,
            },
            {"total": 2},
        )
        == "instrument_failure_avoided_induction_floor"
    )
    assert all(row["killed"] for row in mod.run_mutation_tests())
    assert mod._validate_zero_credit(
        {"solve_claimed": False, "level_credit_delta": 0, "registry_update_count": 0}
    )
    harmful = mod.harmful_regression_count_and_games(
        {
            "g": {
                "treatment_minus_control_score": -0.03,
            }
        },
        {
            "g": {
                "control": {"wall_s": 1.0},
                "treatment": {"wall_s": 3.0},
            }
        },
        {
            "harmful_if_score_delta_lt": -0.02,
            "harmful_if_wall_cost_ratio_gt": 2.0,
        },
    )
    assert harmful["games"] == ["g"]
    assert mod.write_artifact({"ok": True}, path=tmp_path / "artifact.json").is_file()
