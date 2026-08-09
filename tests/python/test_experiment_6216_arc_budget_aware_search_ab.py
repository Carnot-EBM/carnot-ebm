"""Tests for Exp6216 budget-aware search matched A/B.

Spec refs: REQ-ARC-WMTE-6216,
SCENARIO-ARC-WMTE-6216-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6216-MATCHED-STEPWISE-ARMS,
SCENARIO-ARC-WMTE-6216-CALIBRATION-AND-DEADLINE-GATE,
SCENARIO-ARC-WMTE-6216-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import os
from pathlib import Path

from carnot import experiment_6216_arc_budget_aware_search_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _mutation_receipts() -> list[dict[str, object]]:
    return [
        {"name": "consumer_fire_counter_removed", "killed": True},
        {"name": "estimator_calibration_guard_removed", "killed": True},
        {"name": "forbidden_access_guard_removed", "killed": True},
    ]


def test_req_arc_wmte_6216_spec_declares_fields_and_scenarios() -> None:
    """REQ-ARC-WMTE-6216: OpenSpec names the artifact and scenarios."""

    section = "REQ-ARC-WMTE-6216" + SPEC.read_text(encoding="utf-8").split(
        "### REQ-ARC-WMTE-6216", 1
    )[1]

    for marker in (
        "REQ-ARC-WMTE-6216",
        "SCENARIO-ARC-WMTE-6216-REGISTRY-PRECHECK",
        "SCENARIO-ARC-WMTE-6216-MATCHED-STEPWISE-ARMS",
        "SCENARIO-ARC-WMTE-6216-CALIBRATION-AND-DEADLINE-GATE",
        "SCENARIO-ARC-WMTE-6216-ARTIFACT-GUARDS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6216_stepwise_treatment_fires_and_reshapes_search(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6216-MATCHED-STEPWISE-ARMS: enabled arm changes frontier cost."""

    cell = mod.fixture_hud_support_cell("r11l", 621600)
    pair = mod.run_matched_stepwise_cell(cell, raw_root=tmp_path)

    assert pair["hud_support"]["evidence"]["verdict"] == "admit"
    assert pair["hud_support"]["estimate"]["verdict"] == "estimate"
    assert pair["arms"]["aa_control_a"]["selected_node"] == pair["arms"]["aa_control_b"][
        "selected_node"
    ]
    assert pair["arms"]["control"]["consumer_call_count"] == 0
    assert pair["arms"]["treatment"]["consumer_call_count"] > 0
    assert pair["arms"]["control"]["selected_node"] == "risky_long_path"
    assert pair["arms"]["treatment"]["selected_node"] == "safe_short_path"
    assert pair["arms"]["treatment"]["budget_aware_search_enabled"] is True
    assert pair["arms"]["treatment"]["live_policy_class"] == "E3AgentPolicy"
    assert pair["arms"]["treatment"]["explorer_class"] == "StepwiseExplorer"
    assert pair["arms"]["treatment"]["score"] >= pair["arms"]["control"]["score"]
    assert pair["arms"]["treatment"]["deadline_miss"] is False
    for receipt in pair["raw_event_paths_and_hashes"]:
        assert Path(receipt["path"]).is_file()
        assert str(receipt["sha256"]).startswith("sha256:")


def test_scenario_arc_wmte_6216_artifact_guards_and_required_fields(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6216-ARTIFACT-GUARDS: no solve or registry credit."""

    artifact = mod.build_artifact(
        date="20260808",
        games=("r11l", "sc25", "s5i5"),
        seeds=(621600,),
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
    assert artifact["consumer_fire_counts"]["treatment_total"] >= mod.SUPPORT_FLOOR
    assert artifact["consumer_fire_counts"]["control_total"] == 0
    assert artifact["deadline_miss_counts"]["treatment"] == 0
    assert artifact["ab_complete_score"] == 1.0
    assert artifact["budget_aware_promotion_ready_score"] == 1.0
    assert artifact["inference_substrate"]["legacy_models_contributed_rows"] == 0
    assert artifact["model_specs"][0]["hf_id"] == mod.CANONICAL_MODEL_HF_ID
    assert all(
        type(value) is int and value == 0
        for value in artifact["source_bfs_adapter_registry_hidden_state_access_counts"].values()
    )
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_arc_wmte_6216_nonfire_or_miscalibration_is_instrument_failure(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6216-CALIBRATION-AND-DEADLINE-GATE: bad instruments block."""

    nonfire = mod.build_artifact(
        date="20260808",
        games=("r11l", "sc25", "s5i5"),
        seeds=(621600,),
        raw_root=tmp_path / "nonfire",
        mutation_receipts=_mutation_receipts(),
        force_zero_consumer_fire=True,
    )
    mod.validate_artifact(nonfire)
    assert nonfire["status"] == "instrument_failure_zero_consumer_fire"
    assert nonfire["honest_verdict"].startswith("blocked:")

    miscalibrated = mod.build_artifact(
        date="20260808",
        games=("r11l", "sc25", "s5i5"),
        seeds=(621600,),
        raw_root=tmp_path / "miscalibrated",
        mutation_receipts=_mutation_receipts(),
        force_estimator_error=True,
    )
    mod.validate_artifact(miscalibrated)
    assert miscalibrated["status"] == "instrument_failure_estimator_miscalibrated"
    assert miscalibrated["budget_aware_promotion_ready_score"] < 1.0


def test_req_arc_wmte_6216_defensive_helpers(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6216: branch guards stay explicit and deterministic."""

    os.environ["EXP6216_TEMP_RESTORE"] = "old"
    with mod.temporary_env({"EXP6216_TEMP_RESTORE": "new"}):
        assert os.environ["EXP6216_TEMP_RESTORE"] == "new"
    assert os.environ["EXP6216_TEMP_RESTORE"] == "old"
    with mod.temporary_env({"EXP6216_TEMP_RESTORE": None}):
        assert "EXP6216_TEMP_RESTORE" not in os.environ
    assert os.environ["EXP6216_TEMP_RESTORE"] == "old"
    os.environ.pop("EXP6216_TEMP_RESTORE", None)

    assert mod.paired_clustered_intervals({})["n_games"] == 0
    assert mod.ready_score([]) == 0.0
    assert (
        mod.classify_status(
            {
                "treatment_total": 1,
                "support_count": 1,
                "support_floor": 3,
                "mutation_proven": True,
            },
            {"max_abs_error": 0.0, "tolerance": 0.75},
        )
        == "instrument_failure_support_floor"
    )
    assert (
        mod.classify_status(
            {
                "treatment_total": 3,
                "support_count": 3,
                "support_floor": 3,
                "mutation_proven": False,
            },
            {"max_abs_error": 0.0, "tolerance": 0.75},
        )
        == "instrument_failure_mutation_not_killed"
    )
    assert (
        mod.classify_status(
            {
                "treatment_total": 3,
                "support_count": 3,
                "support_floor": 3,
                "mutation_proven": True,
            },
            {"max_abs_error": 1.0, "tolerance": 0.75},
        )
        == "instrument_failure_estimator_miscalibrated"
    )
    assert all(row["killed"] for row in mod.run_mutation_tests())
    assert mod.validate_zero_credit(
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
