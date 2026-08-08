"""Tests for Exp6214 held-out object-delta A/B.

Spec refs: REQ-ARC-WMTE-6214,
SCENARIO-ARC-WMTE-6214-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6214-MATCHED-ARMS,
SCENARIO-ARC-WMTE-6214-TREATMENT-FIRE,
SCENARIO-ARC-WMTE-6214-ARTIFACT-GUARDS.
"""

from __future__ import annotations

from pathlib import Path

from carnot import experiment_6214_arc_object_delta_heldout_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _mutation_receipts() -> list[dict[str, object]]:
    return [
        {"name": "prompt_delta_hook_removed", "killed": True},
        {"name": "treatment_fire_counter_removed", "killed": True},
        {"name": "registry_update_guard_removed", "killed": True},
    ]


def test_req_arc_wmte_6214_spec_declares_fields_and_scenarios() -> None:
    """REQ-ARC-WMTE-6214: OpenSpec names the artifact and scenarios."""

    section = "REQ-ARC-WMTE-6214" + SPEC.read_text(encoding="utf-8").split(
        "### REQ-ARC-WMTE-6214", 1
    )[1]

    for marker in (
        "REQ-ARC-WMTE-6214",
        "SCENARIO-ARC-WMTE-6214-REGISTRY-PRECHECK",
        "SCENARIO-ARC-WMTE-6214-MATCHED-ARMS",
        "SCENARIO-ARC-WMTE-6214-TREATMENT-FIRE",
        "SCENARIO-ARC-WMTE-6214-ARTIFACT-GUARDS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6214_matched_arms_only_add_delta_block() -> None:
    """SCENARIO-ARC-WMTE-6214-MATCHED-ARMS: treatment changes only object input."""

    receipts = mod.render_matched_arm_prompts(
        "fixture",
        mod.fixture_transitions(),
        cell=1,
    )

    assert receipts["aa_control"]["identical"] is True
    assert receipts["control"]["has_static_object_block"] is True
    assert receipts["control"]["has_object_delta_block"] is False
    assert receipts["treatment"]["has_static_object_block"] is True
    assert receipts["treatment"]["has_object_delta_block"] is True
    assert receipts["treatment"]["control_is_prefix"] is True
    assert receipts["object_delta_only_change"] is True


def test_scenario_arc_wmte_6214_artifact_guards_and_required_fields(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6214-ARTIFACT-GUARDS: no solve or registry credit."""

    artifact = mod.build_artifact(
        date="20260808",
        games=("ls20", "s5i5", "tu93"),
        seeds=(621400,),
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
    assert artifact["duplicate_solve_target_count"] == 0
    assert artifact["level_credit_delta"] == 0
    assert artifact["registry_update_count"] == 0
    assert artifact["treatment_fire_counts"]["total"] > 0
    assert artifact["ab_complete_score"] == 1.0
    assert artifact["object_delta_promotion_ready_score"] == 1.0
    assert all(
        type(value) is int and value == 0
        for value in artifact["source_bfs_adapter_registry_hidden_state_access_counts"].values()
    )
    for row in artifact["raw_induction_paths_and_hashes"]:
        assert Path(row["path"]).is_file()
        assert str(row["sha256"]).startswith("sha256:")


def test_scenario_arc_wmte_6214_zero_fire_is_instrument_failure(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6214-TREATMENT-FIRE: zero fire is not a null."""

    artifact = mod.build_artifact(
        date="20260808",
        games=("ls20", "s5i5", "tu93"),
        seeds=(621400,),
        raw_root=tmp_path,
        mutation_receipts=_mutation_receipts(),
        force_zero_treatment_fire=True,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "instrument_failure_zero_treatment_fire"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["object_delta_promotion_ready_score"] < 1.0


def test_req_arc_wmte_6214_mutation_gate_controls_ready_score(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6214: mutation evidence is required for promotion readiness."""

    artifact = mod.build_artifact(
        date="20260808",
        games=("ls20", "s5i5", "tu93"),
        seeds=(621400,),
        raw_root=tmp_path,
        mutation_receipts=[{"name": "prompt_delta_hook_removed", "killed": False}],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "instrument_failure_mutation_not_killed"
    assert artifact["ab_complete_score"] == 1.0
    assert artifact["object_delta_promotion_ready_score"] < 1.0


def test_req_arc_wmte_6214_defensive_branches(monkeypatch) -> None:
    """REQ-ARC-WMTE-6214: edge branches stay explicit and deterministic."""

    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "1")
    monkeypatch.setenv("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", "0")
    mod.render_matched_arm_prompts("fixture", mod.fixture_transitions(), cell=1)
    assert mod.sign_test_two_sided([])["test_was_possible"] is False
    assert mod._comb(1, 2) == 0
    assert mod.paired_clustered_intervals({})["n_games"] == 0
    assert (
        mod.classify_status(
            {
                "total": 1,
                "support_count": 1,
                "support_floor": 3,
                "mutation_proven": True,
            }
        )
        == "instrument_failure_support_floor"
    )
    assert (
        mod.harmful_regression_count_and_games(
            {
                "g": {
                    "treatment_minus_control_change_fidelity": -0.03,
                }
            },
            {
                "g": {
                    "control": {"wall_s": 1.0},
                    "treatment": {"wall_s": 1.0},
                }
            },
            {
                "harmful_if_change_fidelity_delta_lt": -0.02,
                "harmful_if_wall_cost_ratio_gt": 2.0,
            },
        )["games"]
        == ["g"]
    )
    assert mod._validate_zero_credit(
        {"solve_claimed": False, "level_credit_delta": 0, "registry_update_count": 0}
    )
