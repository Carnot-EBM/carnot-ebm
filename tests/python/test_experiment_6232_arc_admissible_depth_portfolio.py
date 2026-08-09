"""Tests for Exp6232 ARC admissible depth portfolio.

Spec refs: REQ-ARC-WMTE-6232,
SCENARIO-ARC-WMTE-6232-PRECONDITION-LEDGER,
SCENARIO-ARC-WMTE-6232-TERMINAL-SKIP,
SCENARIO-ARC-WMTE-6232-PORTFOLIO-RUN,
SCENARIO-ARC-WMTE-6232-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_6232_arc_admissible_depth_portfolio as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def test_req_arc_wmte_6232_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-6232: OpenSpec names the Exp6232 fields and scenarios."""

    spec = SPEC.read_text(encoding="utf-8")
    section = "REQ-ARC-WMTE-6232" + spec.rsplit("### REQ-ARC-WMTE-6232", 1)[1]

    for marker in (
        "REQ-ARC-WMTE-6232",
        "SCENARIO-ARC-WMTE-6232-PRECONDITION-LEDGER",
        "SCENARIO-ARC-WMTE-6232-TERMINAL-SKIP",
        "SCENARIO-ARC-WMTE-6232-PORTFOLIO-RUN",
        "SCENARIO-ARC-WMTE-6232-ARTIFACT-GUARDS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6232_current_v539_artifacts_skip_before_model_load(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6232-TERMINAL-SKIP: current evidence has no pair."""

    artifact = mod.build_artifact(
        date="20260809",
        output_path=tmp_path / "experiment_6232.json",
        test_commands=["focused"],
        test_exit_codes={"focused": 0},
        started=0.0,
        now=lambda: 1.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "skipped_less_than_two_eligible_depth_levers"
    assert artifact["eligible_lever_count"] == 0
    assert artifact["selected_levers"] == []
    assert artifact["model_loaded"] is False
    assert artifact["exact_skip_reason"] == (
        "fewer_than_two_independent_unflagged_treatment_active_depth_levers:0<2"
    )
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["preregistered_portfolio_game_seed_matrix"]["opened"] is False
    assert artifact["matched_arm_configuration"]["built"] is False

    ledger = {row["lever_id"]: row for row in artifact["lever_eligibility_ledger"]}
    assert ledger["exp6215_object_relative_trajectory_transfer"]["default_stack_component"] is True
    assert "already_default_on_not_counted" in ledger[
        "exp6215_object_relative_trajectory_transfer"
    ]["ineligible_reasons"]
    assert "terminal_class_flagged_not_admissible" in ledger[
        "exp6216_budget_aware_search"
    ]["ineligible_reasons"]
    assert "terminal_class_skipped_not_admissible" in ledger[
        "exp6229_bounded_reinduction"
    ]["ineligible_reasons"]
    assert "terminal_class_skipped_not_admissible" in ledger[
        "exp6230_prompt_enrichment"
    ]["ineligible_reasons"]
    assert "artifact_missing" in ledger["exp6231_depth_lever"]["ineligible_reasons"]


def test_scenario_arc_wmte_6232_synthetic_portfolio_freezes_pair(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6232-PORTFOLIO-RUN: two independent levers open one run."""

    ledgers = [
        mod.synthetic_eligible_lever("a_depth", mechanism="bounded_reinduction", utility=0.4),
        mod.synthetic_eligible_lever("b_depth", mechanism="prompt_enrichment", utility=0.3),
        mod.synthetic_eligible_lever("c_depth", mechanism="graded_goal_bias", utility=0.2),
    ]
    artifact = mod.build_artifact(
        date="20260809",
        precomputed_ledger=ledgers,
        output_path=tmp_path / "experiment_6232.json",
        test_commands=["focused"],
        test_exit_codes={"focused": 0},
        started=0.0,
        now=lambda: 2.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete_portfolio_pair_frozen_not_executed_by_unit_fixture"
    assert artifact["eligible_lever_count"] == 3
    assert [row["lever_id"] for row in artifact["selected_levers"]] == ["a_depth", "b_depth"]
    assert artifact["model_loaded"] is False
    assert artifact["exact_skip_reason"] is None
    assert artifact["preregistered_portfolio_game_seed_matrix"]["opened"] is True
    assert artifact["matched_arm_configuration"]["built"] is True
    assert artifact["matched_arm_configuration"]["current_default_stack"]["trajectory_transfer"] is True
    assert artifact["matched_arm_configuration"]["current_default_stack"]["budget_aware_search"] is True


def test_scenario_arc_wmte_6232_duplicate_mechanisms_do_not_double_count() -> None:
    """SCENARIO-ARC-WMTE-6232-PRECONDITION-LEDGER: duplicate mechanisms collapse."""

    ledgers = mod.apply_independence_gate(
        [
            mod.synthetic_eligible_lever("a1", mechanism="same_mechanism", utility=0.2),
            mod.synthetic_eligible_lever("a2", mechanism="same_mechanism", utility=0.5),
            mod.synthetic_eligible_lever("b1", mechanism="other_mechanism", utility=0.1),
        ]
    )

    rows = {row["lever_id"]: row for row in ledgers}
    assert rows["a2"]["eligible"] is True
    assert rows["a1"]["eligible"] is False
    assert "duplicate_mechanism_not_counted" in rows["a1"]["ineligible_reasons"]
    assert len(mod.select_levers(ledgers)) == 2


def test_scenario_arc_wmte_6232_artifact_guards_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6232-ARTIFACT-GUARDS: required guard fields are bare."""

    artifact = mod.build_artifact(
        date="20260809",
        output_path=tmp_path / "experiment_6232.json",
        test_commands=["focused"],
        test_exit_codes={"focused": 0},
        started=0.0,
        now=lambda: 1.0,
    )
    out = mod.write_artifact(artifact, path=tmp_path / "experiment_6232.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["model_loaded"] is False
    assert loaded["registry_update_count"] == 0
    assert loaded["verifier_is_oracle"] is False
    assert loaded["solve_provenance"] == "live_agent_self_discovery"
    assert all(
        type(value) is int and value == 0
        for value in loaded["source_bfs_adapter_hidden_state_registry_access_counts"].values()
    )
    assert set(loaded["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(loaded["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert loaded["reproducibility_checksum"] == mod.payload_checksum(loaded)


def test_req_arc_wmte_6232_validation_rejects_bad_artifact_state() -> None:
    """REQ-ARC-WMTE-6232: malformed artifacts fail closed."""

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

    bad_model = dict(base)
    bad_model["model_loaded"] = True
    expect_error(bad_model, "model_loaded must be bare false")

    bad_registry = dict(base)
    bad_registry["registry_update_count"] = 1
    expect_error(bad_registry, "registry_update_count must be bare 0")

    bad_solve = dict(base)
    bad_solve["solve_provenance"] = "offline_replay"
    expect_error(bad_solve, "solve_provenance invalid")

    bad_forbidden = dict(base)
    bad_forbidden["source_bfs_adapter_hidden_state_registry_access_counts"] = {"source_reads": 1}
    expect_error(bad_forbidden, "forbidden counts must be bare zeros")

    bad_hash = dict(base)
    bad_hash["registry_precheck_and_hash_before_after"] = {
        "registry_hash_before": "sha256:a",
        "registry_hash_after": "sha256:b",
    }
    expect_error(bad_hash, "registry hash changed")

    bad_verdict = dict(base)
    bad_verdict["honest_verdict"] = "skipped"
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    expect_error(bad_verdict, "honest verdict prefix invalid")

    bad_checksum = dict(base)
    bad_checksum["status"] = "changed"
    expect_error(bad_checksum, "checksum mismatch")

    bad_oracle = dict(base)
    bad_oracle["verifier_is_oracle"] = True
    expect_error(bad_oracle, "verifier_is_oracle must be bare false")


def test_req_arc_wmte_6232_fixture_artifact_exercises_fail_closed_gates(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6232: flagged fixture rows preserve losses and raw receipts."""

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    payload = {
        "status": "complete_ready",
        "honest_verdict": "complete: synthetic_flagged_depth_measurement",
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {"severity": "critical", "kind": "TEST", "detail": "synthetic"}
        ],
        "treatment_fire_counts": {
            "total": 2,
            "support_count": 2,
            "support_floor": 1,
            "mutation_proven": True,
        },
        "synthetic_ready_score": 1.0,
        "admission_and_level_depth_by_arm_game": {
            "loss_game": {
                "control": {"depth": 2},
                "treatment": {"depth": 1},
                "loss_reported": True,
                "treatment_minus_control_wall_s": 1.0,
                "treatment_minus_control_tokens": 5,
                "treatment_minus_control_actions": 1,
            }
        },
        "harmful_regression_count_and_games": {"count": 1, "games": ["loss_game"]},
        "solve_claimed": False,
        "level_credit_delta": 0,
        "registry_update_count": 0,
        "raw_prompt_output_engine_replay_hashes": [{"path": "raw.json", "sha256": "sha256:1"}],
        "raw_induction_paths_and_hashes": [{"path": "engine.py", "sha256": "sha256:2"}],
        "source_bfs_adapter_registry_hidden_state_access_counts": {"source_reads": 1},
    }
    (result_dir / "experiment_999_actual.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    row = mod.recompute_lever_eligibility(
        {
            "lever_id": "synthetic_flagged",
            "experiment": 999,
            "requirement": mod.REQUIREMENT,
            "mechanism": "synthetic",
            "path": "results/fallback.json",
            "path_glob": "results/experiment_999*.json",
            "default_stack_component": False,
            "fire_field": "treatment_fire_counts",
            "quality_fields": ("synthetic_ready_score",),
            "metric_fields": ("admission_and_level_depth_by_arm_game",),
        },
        root=tmp_path,
    )

    assert row["artifact"]["path"].endswith("experiment_999_actual.json")
    assert "flagged_adversarial_true" in row["ineligible_reasons"]
    assert "unresolved_critical_corrigendum_present" in row["ineligible_reasons"]
    assert "safety_gate_failed" in row["ineligible_reasons"]
    assert "forbidden_access_nonzero" in row["ineligible_reasons"]
    assert row["effect_gate"]["losing_games"] == ["loss_game"]
    assert row["raw_prompt_output_engine_replay_hashes"]["raw_count"] == 2
