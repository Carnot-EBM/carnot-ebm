"""Tests for Exp 4862 ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4862,
SCENARIO-ARC-WMTE-4862-ROTATED-TARGET,
SCENARIO-ARC-WMTE-4862-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4862-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4862_levelup_attempt as exp4862


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 1
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - g50t: clone_replay_L2_route_reached_distance_12_no_bank
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - wa30: hidden-state-bound registry row
- game: r11l
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - r11l: prefix_rooted_graph_search_stalled_at_L1
reproducible_total_levels: 65
"""


def _recommendation(game: str = "r11l") -> dict[str, object]:
    return {
        "target_game": game,
        "recommended": [{"game": "lp85", "similarity": 8.0}],
        "selected_generic_operators": [{"operator": "click_template_alignment"}],
        "cautions": ["avoid known no-grounded-delta walls"],
    }


def _preconditions(game: str = "r11l") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade": {"ok": True, "check": "arc_solver_kit.offline_arcade()"},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_offline_env": {"game": game, "ok": True},
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def _success_loop_result(game: str = "r11l", reached_level: int = 2) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": True,
        "reproduced_levels": reached_level,
        "solve_provenance": "live_agent_self_discovery",
        "mode": "standing_arc_loop_adaptered",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": True,
        },
        "solution_labels": ["L1-prefix", "L2-handle-average-tail"],
    }


def test_req_arc_wmte_4862_spec_declares_rotated_contract() -> None:
    """REQ-ARC-WMTE-4862: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4862",
        "SCENARIO-ARC-WMTE-4862-ROTATED-TARGET",
        "SCENARIO-ARC-WMTE-4862-REPRODUCTION-GATE",
        "SCENARIO-ARC-WMTE-4862-STABLE-ARTIFACT",
        exp4862.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in exp4862.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4862_selects_r11l_and_excludes_recent_targets() -> None:
    """SCENARIO-ARC-WMTE-4862-ROTATED-TARGET: rotate off s5i5/ka59."""

    selection = exp4862.select_rotation_target(
        yaml.safe_load(_registry_text()),
        approach_recommendation=_recommendation("r11l"),
    )

    assert selection["game"] == "r11l"
    assert selection["prior_level"] == 1
    assert selection["target_level"] == 2
    assert selection["reason"] == "grounded_click_template_handle_average_delta"
    assert selection["excluded_previous_targets"] == ["s5i5", "ka59"]
    assert selection["approach_recommendation"] == _recommendation("r11l")
    assert [row["game"] for row in selection["candidate_audit"]] == [
        "g50t",
        "wa30",
        "r11l",
    ]
    assert selection["candidate_audit"][0]["status"] == "skip_prior_no_bank_wall"
    assert selection["candidate_audit"][1]["status"] == "skip_hidden_state_bound"
    assert selection["candidate_audit"][2]["status"] == "selected"
    assert all(
        row["game"] not in {"s5i5", "ka59"} for row in selection["candidate_audit"]
    )


def test_scenario_arc_wmte_4862_summarizes_successful_reproduction_gate() -> None:
    """SCENARIO-ARC-WMTE-4862-REPRODUCTION-GATE: a new gate depth banks one."""

    selection = exp4862.select_rotation_target(
        yaml.safe_load(_registry_text()),
        approach_recommendation=_recommendation("r11l"),
    )
    attempt = exp4862.summarize_loop_attempt(
        selection=selection,
        loop_result=_success_loop_result(),
        loop_result_path="results/arc_loop_solve_r11l.json",
    )

    assert attempt["game"] == "r11l"
    assert attempt["prior_level"] == 1
    assert attempt["reached_level"] == 2
    assert attempt["offline_reproduced_new_depth"] is True
    assert attempt["new_levels_banked"] == 1
    assert attempt["residual_cause"] == "banked_new_level"
    assert "handle-average" in attempt["dead_end"]


def test_req_arc_wmte_4862_builds_success_artifact_and_writes_json(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4862: success artifact records the banked reproducible level."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4862.select_rotation_target(
        registry,
        approach_recommendation=_recommendation("r11l"),
    )
    attempts = [
        exp4862.summarize_loop_attempt(
            selection=selection,
            loop_result=_success_loop_result(),
            loop_result_path="results/arc_loop_solve_r11l.json",
        )
    ]

    artifact = exp4862.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("r11l"),
    )
    output = exp4862.write_artifact(artifact, tmp_path / "experiment_4862_levelup_attempt.json")
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["honest_verdict"] == "success_r11l_levelup_banked"
    assert saved["solve_provenance"] == "live_agent_self_discovery"
    assert saved["target_game"] == "r11l"
    assert saved["offline_reproduced"] is True
    assert saved["reproduced_levels"] == 2
    assert saved["new_levels_banked"] == 1
    assert saved["inference_substrate"] == "adaptered_replay_no_induction"
    assert saved["retire_if_same_verdict"] is True
    assert saved["registry_update"]["updated"] is True
    assert saved["registry_update"]["reproducible_total_levels_after"] == 66
    assert saved["schema_errors"] == []
    assert exp4862.artifact_schema_errors(saved) == []


def test_req_arc_wmte_4862_blocks_missing_target_env() -> None:
    """REQ-ARC-WMTE-4862: missing target environments produce blocked artifacts."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4862.select_rotation_target(registry)
    preconditions = _preconditions("r11l")
    preconditions["target_offline_env"] = {"game": "r11l", "ok": False}

    artifact = exp4862.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[],
        preconditions_checked=preconditions,
    )

    assert artifact["honest_verdict"] == "blocked_r11l_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4862_covers_defensive_no_bank_branches() -> None:
    """REQ-ARC-WMTE-4862: no-bank residuals never increment the registry total."""

    registry = yaml.safe_load(
        """schema_version: 1
games:
- game: g50t
  levels_reproduced: 2
- game: wa30
  levels_reproduced: 2
- game: r11l
  levels_reproduced: 2
reproducible_total_levels: 7
"""
    )
    selection = exp4862.select_rotation_target(registry)

    assert selection["game"] == "none"
    assert selection["candidate_audit"][0]["status"] == "skip_not_l1_only"

    lower_priority = exp4862.select_rotation_target(
        yaml.safe_load(
            """schema_version: 1
games:
- game: g50t
  levels_reproduced: 1
- game: wa30
  levels_reproduced: 2
- game: r11l
  levels_reproduced: 2
reproducible_total_levels: 5
"""
        )
    )
    assert lower_priority["candidate_audit"][0]["status"] == "candidate_unselected"

    existing_depth = exp4862.summarize_loop_attempt(
        selection={"game": "r11l", "prior_level": 1, "target_level": 2},
        loop_result={
            "offline_reproduced": True,
            "reproduction_gate": {"reached_level": "not-an-int", "reproduced": True},
        },
        loop_result_path="results/arc_loop_solve_r11l.json",
    )
    failed_gate = exp4862.summarize_loop_attempt(
        selection={"game": "r11l", "prior_level": 1, "target_level": 2},
        loop_result={"offline_reproduced": False, "reached_level": 2},
        loop_result_path="results/arc_loop_solve_r11l.json",
    )
    needs_re = exp4862.summarize_loop_attempt(
        selection={"game": "r11l", "prior_level": 1, "target_level": 2},
        loop_result={"status": "needs_per_game_RE"},
        loop_result_path="results/arc_loop_solve_r11l.json",
    )
    artifact_existing = exp4862.build_artifact(
        registry=registry,
        selection={"game": "r11l", "prior_level": 1, "target_level": 2},
        attempts=[existing_depth, failed_gate],
        preconditions_checked=_preconditions("r11l"),
    )
    artifact_needs_re = exp4862.build_artifact(
        registry=registry,
        selection={"game": "r11l", "prior_level": 1, "target_level": 2},
        attempts=[needs_re],
        preconditions_checked=_preconditions("r11l"),
    )
    artifact_no_attempts = exp4862.build_artifact(
        registry=registry,
        selection={"game": "r11l", "prior_level": 1, "target_level": 2},
        attempts=[],
        preconditions_checked=_preconditions("r11l"),
    )

    assert existing_depth["residual_cause"] == "reproduced_existing_or_lower_level"
    assert failed_gate["residual_cause"] == "offline_reproduction_failed"
    assert needs_re["residual_cause"] == "needs_per_game_RE"
    assert artifact_existing["honest_verdict"].endswith("_residual_existing_depth")
    assert artifact_needs_re["honest_verdict"].endswith("_residual_needs_per_game_RE")
    assert artifact_no_attempts["honest_verdict"].endswith("_residual_no_attempts")
    assert artifact_existing["schema_errors"] == []
    assert artifact_needs_re["schema_errors"] == []
    assert artifact_no_attempts["schema_errors"] == []


def test_req_arc_wmte_4862_schema_guards_required_contract() -> None:
    """REQ-ARC-WMTE-4862: schema validation rejects overclaims and drift."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4862.select_rotation_target(registry)
    attempts = [
        exp4862.summarize_loop_attempt(
            selection=selection,
            loop_result=_success_loop_result(),
            loop_result_path="results/arc_loop_solve_r11l.json",
        )
    ]
    artifact = exp4862.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("r11l"),
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    assert "missing_field:honest_verdict" in exp4862.artifact_schema_errors(missing)

    bad_principle = dict(artifact, field_principles=dict(artifact["field_principles"]))
    bad_principle["field_principles"]["target_game"] = {"principle": "wrong"}
    assert "missing_principle:target_game" in exp4862.artifact_schema_errors(bad_principle)

    invalid_checksum = dict(artifact, reproducibility_checksum="not-hex")
    assert "invalid_reproducibility_checksum" in exp4862.artifact_schema_errors(invalid_checksum)

    checksum_mismatch = dict(artifact, random_seed=1)
    assert "checksum_mismatch" in exp4862.artifact_schema_errors(checksum_mismatch)

    drifted = dict(
        artifact,
        honest_verdict="other",
        solve_provenance="outer_loop_re",
        target_game="s5i5",
        inference_substrate="live_llm_inference",
        verifier_is_oracle=False,
        offline_reproduced=False,
        retire_if_same_verdict=False,
        experiment="wrong",
        schema="wrong",
        spec_refs=[],
        result_path="wrong",
    )
    errors = set(exp4862.artifact_schema_errors(drifted))

    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "solve_provenance_mismatch" in errors
    assert "rotated_target_must_not_be_s5i5_or_ka59" in errors
    assert "inference_substrate_mismatch" in errors
    assert "verifier_is_oracle_must_be_true" in errors
    assert "bank_without_offline_reproduction" in errors
    assert "retire_if_same_verdict_must_be_true" in errors
    assert "experiment_mismatch" in errors
    assert "schema_mismatch" in errors
    assert "spec_refs_mismatch" in errors
    assert "result_path_mismatch" in errors

    no_bank_overclaim = dict(artifact, new_levels_banked=0)
    assert "offline_reproduced_true_without_new_bank" in exp4862.artifact_schema_errors(
        no_bank_overclaim
    )

    missing_target = dict(artifact, target_game="")
    assert "target_game_missing" in exp4862.artifact_schema_errors(missing_target)
