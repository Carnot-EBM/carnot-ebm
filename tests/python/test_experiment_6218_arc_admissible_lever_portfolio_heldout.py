"""Tests for Exp6218 admissible ARC lever portfolio gate.

Spec refs: REQ-ARC-WMTE-6218,
SCENARIO-ARC-WMTE-6218-UPSTREAM-RECOMPUTE,
SCENARIO-ARC-WMTE-6218-STRUCTURED-SKIP,
SCENARIO-ARC-WMTE-6218-SELECTION-RULE,
SCENARIO-ARC-WMTE-6218-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_6218_arc_admissible_lever_portfolio_heldout as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def test_req_arc_wmte_6218_spec_declares_fields_and_scenarios() -> None:
    """REQ-ARC-WMTE-6218: the OpenSpec names fields and scenarios."""

    section = "REQ-ARC-WMTE-6218" + SPEC.read_text(encoding="utf-8").split(
        "### REQ-ARC-WMTE-6218", 1
    )[1]

    for marker in (
        "REQ-ARC-WMTE-6218",
        "SCENARIO-ARC-WMTE-6218-UPSTREAM-RECOMPUTE",
        "SCENARIO-ARC-WMTE-6218-STRUCTURED-SKIP",
        "SCENARIO-ARC-WMTE-6218-SELECTION-RULE",
        "SCENARIO-ARC-WMTE-6218-ARTIFACT-GUARDS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6218_committed_upstreams_recompute_to_skip(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6218-STRUCTURED-SKIP: one clean lever is not enough."""

    artifact = mod.build_artifact(
        date="20260809",
        output_path=tmp_path / "experiment_6218.json",
        test_commands=["focused"],
        test_exit_codes={"focused": 0},
        started=0.0,
        now=lambda: 1.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "skipped_less_than_two_eligible_levers"
    assert artifact["structured_skip_reason"]["eligible_count"] == 1
    assert artifact["selected_levers"] == []
    assert artifact["combination_count_tested"] == 0
    assert artifact["inference_substrate"]["model_load_attempted"] is False
    assert artifact["preregistered_heldout_game_seed_matrix"]["opened"] is False
    assert artifact["matched_baseline_single_and_pair_configs"]["built"] is False

    eligibility = artifact["eligible_and_ineligible_levers_with_reasons"]
    assert eligibility["eligible"] == ["exp6215_object_relative_trajectory_transfer"]
    assert "artifact_flagged_by_exp6197" in eligibility["ineligible"]["exp6214_object_delta"]
    assert "artifact_flagged_by_exp6197" in eligibility["ineligible"]["exp6216_budget_aware_search"]
    assert "terminal_class_skipped_not_admissible" in eligibility["ineligible"]["exp6217_gemma31_think"]


def test_scenario_arc_wmte_6218_selection_rule_freezes_one_pair() -> None:
    """SCENARIO-ARC-WMTE-6218-SELECTION-RULE: only the top-two pair is selected."""

    gates = [
        mod.synthetic_gate("exp1", primary_quality_delta=0.2, efficiency_gain=1.0),
        mod.synthetic_gate("exp2", primary_quality_delta=0.1, efficiency_gain=4.0),
        mod.synthetic_gate("exp3", primary_quality_delta=0.26, efficiency_gain=0.0),
    ]

    selected = mod.select_top_two_levers(gates)

    assert [row["lever_id"] for row in selected] == ["exp2", "exp3"]
    assert selected[0]["selection_utility"] > selected[1]["selection_utility"]
    assert mod.combination_count_for_selection(selected) == 1


def test_scenario_arc_wmte_6218_synthetic_non_skip_shape(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6218-SELECTION-RULE: two eligible levers open one matrix."""

    selected = [
        mod.synthetic_gate("exp_a", primary_quality_delta=0.3, efficiency_gain=0.0),
        mod.synthetic_gate("exp_b", primary_quality_delta=0.2, efficiency_gain=1.0),
    ]
    artifact = mod.build_artifact(
        date="20260809",
        precomputed_gates=selected,
        output_path=tmp_path / "experiment_6218.json",
        test_commands=["focused"],
        test_exit_codes={"focused": 0},
        started=0.0,
        now=lambda: 2.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_pair_frozen_not_executed"
    assert artifact["combination_count_tested"] == 1
    assert len(artifact["selected_levers"]) == 2
    assert artifact["preregistered_heldout_game_seed_matrix"]["opened"] is True
    assert artifact["matched_baseline_single_and_pair_configs"]["built"] is True


def test_scenario_arc_wmte_6218_artifact_guards_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6218-ARTIFACT-GUARDS: zero-credit fields are bare."""

    artifact = mod.build_artifact(
        date="20260809",
        output_path=tmp_path / "experiment_6218.json",
        test_commands=["focused"],
        test_exit_codes={"focused": 0},
        started=0.0,
        now=lambda: 1.0,
    )
    out = mod.write_artifact(artifact, path=tmp_path / "experiment_6218.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["default_flip_count"] == 0
    assert loaded["solve_claimed"] is False
    assert loaded["level_credit_delta"] == 0
    assert loaded["registry_update_count"] == 0
    assert loaded["verifier_is_oracle"] is False
    assert all(
        type(value) is int and value == 0
        for value in loaded["source_bfs_adapter_registry_hidden_state_access_counts"].values()
    )
    assert set(loaded["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(loaded["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert loaded["reproducibility_checksum"] == mod.payload_checksum(loaded)


def test_req_arc_wmte_6218_malformed_payload_is_rejected() -> None:
    """REQ-ARC-WMTE-6218: artifact validation fails closed."""

    artifact = mod.build_artifact(started=0.0, now=lambda: 1.0)
    artifact["combination_count_tested"] = 2

    try:
        mod.validate_artifact(artifact)
    except ValueError as exc:
        assert "combination_count_tested" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validate_artifact accepted an invalid combination count")


def test_scenario_arc_wmte_6218_missing_raw_receipts_block_eligibility(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6218-UPSTREAM-RECOMPUTE: missing raw receipts reject a lever."""

    payload = {
        "status": "complete_ready",
        "honest_verdict": "complete: synthetic",
        "treatment_fire_counts": {
            "total": 3,
            "support_count": 3,
            "support_floor": 3,
            "mutation_proven": True,
        },
        "object_delta_promotion_ready_score": 1.0,
        "harmful_regression_count_and_games": {
            "count": 0,
            "games": [],
            "losing_games_reported_not_hidden": [],
        },
        "change_and_goal_fidelity_by_arm_game": {
            "g": {
                "control": {"wall_s": 1.0},
                "treatment": {"wall_s": 1.0},
                "treatment_minus_control_change_fidelity": 0.1,
            }
        },
        "solve_claimed": False,
        "level_credit_delta": 0,
        "registry_update_count": 0,
    }
    artifact_path = tmp_path / "synthetic.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    gate = mod.recompute_lever_gate(
        {
            "lever_id": "synthetic_missing_raw",
            "experiment": 1,
            "requirement": mod.REQUIREMENT,
            "path": "synthetic.json",
            "raw_dir": "missing_raw_dir",
            "quality_score_field": "object_delta_promotion_ready_score",
            "metric_field": "change_and_goal_fidelity_by_arm_game",
            "fire_field": "treatment_fire_counts",
        },
        root=tmp_path,
    )

    assert gate["eligible"] is False
    assert "raw_receipts_missing_or_empty" in gate["ineligible_reasons"]


def test_req_arc_wmte_6218_validation_fail_closed_branches() -> None:
    """REQ-ARC-WMTE-6218: every guard rejects malformed artifact state."""

    def expect_error(artifact: dict[str, object], message: str) -> None:
        try:
            mod.validate_artifact(artifact)
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"validate_artifact accepted {message}")

    base = mod.build_artifact(started=0.0, now=lambda: 1.0)

    missing = dict(base)
    missing.pop("status")
    expect_error(missing, "missing fields")

    bad_provenance = dict(base)
    bad_provenance["field_provenance"] = {}
    expect_error(bad_provenance, "field_provenance incomplete")

    bad_principles = dict(base)
    bad_principles["field_principles"] = {}
    expect_error(bad_principles, "field_principles incomplete")

    bad_zero = dict(base)
    bad_zero["default_flip_count"] = 1
    expect_error(bad_zero, "default_flip_count must be bare 0")

    bad_false = dict(base)
    bad_false["solve_claimed"] = True
    expect_error(bad_false, "solve_claimed must be bare false")

    bad_skip_combo = dict(base)
    bad_skip_combo["combination_count_tested"] = 1
    expect_error(bad_skip_combo, "combination_count_tested must be 0 on skip")

    bad_forbidden = dict(base)
    bad_forbidden["source_bfs_adapter_registry_hidden_state_access_counts"] = {
        "source_reads": 1
    }
    expect_error(bad_forbidden, "forbidden counts must be bare zeros")

    bad_registry = dict(base)
    bad_registry["registry_precheck_and_hash_before_after"] = {
        "registry_hash_before": "sha256:a",
        "registry_hash_after": "sha256:b",
    }
    expect_error(bad_registry, "registry hash changed")

    bad_legacy = dict(base)
    bad_legacy["inference_substrate"] = {
        **dict(base["inference_substrate"]),
        "legacy_models_contributed_rows": 1,
    }
    expect_error(bad_legacy, "legacy model rows must be zero")

    bad_checksum = dict(base)
    bad_checksum["status"] = "skipped_checksum_changed"
    expect_error(bad_checksum, "checksum mismatch")

    bad_verdict = dict(base)
    bad_verdict["honest_verdict"] = "running"
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    expect_error(bad_verdict, "honest verdict prefix invalid")
