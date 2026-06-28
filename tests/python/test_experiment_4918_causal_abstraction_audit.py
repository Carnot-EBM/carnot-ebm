"""Tests for Exp 4918 causal-abstraction diagnostic audit.

Spec refs: REQ-ARC-WMTE-4918,
SCENARIO-ARC-WMTE-4918-A1-DIAGNOSTIC-AUDIT,
SCENARIO-ARC-WMTE-4918-NAMED-FAILURES,
SCENARIO-ARC-WMTE-4918-BLOCKED-A1-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4918_causal_abstraction_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _observable_proof(variable: str) -> dict[str, Any]:
    extractors = {
        "visible_grid_hash": "frame.grid -> sha256",
        "action_id": "candidate.action_id",
        "action_data": "candidate.data",
        "changed_cell_value_basis": "frame grid plus executed transition delta",
        "visible_level_before": "env/frame levels_completed",
    }
    return {
        "observable": True,
        "extractor": extractors[variable],
        "proof": f"{variable} extracted from {extractors[variable]} on observed transition samples",
    }


def _hidden_proof() -> dict[str, Any]:
    return {
        "observable": False,
        "extractor": None,
        "proof": "No ARC frame/env extractor exposes the banked winning-prefix automaton index.",
    }


def _a1_row(game: str, *, role: str = "failed", hidden: bool = True) -> dict[str, Any]:
    variables = ["visible_grid_hash", "action_id", "action_data", "changed_cell_value_basis"]
    if role == "positive_control":
        variables.append("visible_level_before")
    if hidden:
        variables.append("winning_prefix_order_state")
    proofs = {
        variable: _hidden_proof()
        if variable == "winning_prefix_order_state"
        else _observable_proof(variable)
        for variable in variables
    }
    observable = {variable: bool(proof["observable"]) for variable, proof in proofs.items()}
    return {
        "game": game,
        "role": role,
        "required_variables": variables,
        "observable_from_interface": observable,
        "classification": "HIDDEN_STATE" if hidden else "OBSERVABLE_GAP",
        "evidence": {
            "targets": ["changed_cell_value", "progress_to_goal"],
            "transition_count": 6,
            "changed_transition_count": 3,
            "progress_transition_count": 0,
            "engine_loaded": True,
            "observability_proofs": proofs,
            "live_path_methods_called": ["arc_executable_world_model.load_engine"],
        },
    }


def _a1_artifact(*, fork: str = "WALL_IS_HIDDEN_STATE") -> dict[str, Any]:
    failed = {game: _a1_row(game) for game in ("cd82", "cn04", "ls20")}
    controls = {
        "tu93": _a1_row("tu93", role="positive_control", hidden=False),
        "ar25": _a1_row("ar25", role="positive_control", hidden=False),
    }
    return {
        "experiment_id": 4914,
        "honest_verdict": "complete_causal_abstraction_hidden_state_representation_invariant_closure",
        "fork_verdict": fork,
        "per_game_causal_abstraction": failed,
        "minimal_abstraction_is_observable_subset": False,
        "positive_control_games": ["tu93", "ar25"],
        "positive_control_rows": controls,
        "positive_control_classifies_observable": True,
        "is_decision_need_table_in_disguise": False,
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "causal_abstraction_config": {
            "classification_only": True,
            "failed_games": ["cd82", "cn04", "ls20"],
            "targets": ["changed_cell_value", "progress_to_goal"],
        },
        "preconditions_checked": {
            "a1_baseline": {
                "ok": True,
                "failed_games": ["cd82", "cn04", "ls20"],
                "path": "results/experiment_4903_env_grounded_location_pruned_search.json",
            }
        },
    }


def _exp4903_artifact() -> dict[str, Any]:
    return {
        "experiment_id": 4903,
        "fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION",
        "per_game_first_win": {
            game: {
                "game": game,
                "bucket": "NEVER_ENUMERATED",
                "baseline_bucket": "NEVER_ENUMERATED",
                "first_win_env_grounded": 0.0,
                "migrated": False,
                "best_path_len": 7,
                "states_expanded": 8,
                "real_env_value_reads": 24,
                "change_value_predictions_used": 0,
                "live_path_methods_called": [
                    "StepwiseExplorer.action_prior",
                    "arc_executable_world_model.load_engine",
                    "arc_executable_world_model.plan_in_model",
                ],
            }
            for game in ("cd82", "cn04", "ls20")
        },
    }


def _source_text(*, collect_transitions: bool = True) -> str:
    if not collect_transitions:
        return """
def default_game_classifier(game, role, exp4903_row, transitions_per_game, random_seed, solved_reproduced_level):
    transitions = [{"placeholder": True}]
    return classify_game_causal_abstraction(game=game, transitions=transitions, exp4903_row=exp4903_row, role=role, engine_loaded=True)
"""
    return """
def default_game_classifier(game, role, exp4903_row, transitions_per_game, random_seed, solved_reproduced_level):
    from carnot.agentic.arc_executable_world_model import collect_transitions, load_engine
    load_engine(game)
    transitions, _cell = collect_transitions(game, n=int(transitions_per_game), seed=int(random_seed))
    return classify_game_causal_abstraction(game=game, transitions=transitions, exp4903_row=exp4903_row, role=role, engine_loaded=True, solved_reproduced_level=solved_reproduced_level)
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4918_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4918: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4918",
        "SCENARIO-ARC-WMTE-4918-A1-DIAGNOSTIC-AUDIT",
        "SCENARIO-ARC-WMTE-4918-NAMED-FAILURES",
        "SCENARIO-ARC-WMTE-4918-BLOCKED-A1-ARTIFACT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4918_clean_diagnostic_is_trusted() -> None:
    """SCENARIO-ARC-WMTE-4918-A1-DIAGNOSTIC-AUDIT: all six audit checks pass."""

    artifact = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_source_text(),
        exp4903_artifact=_exp4903_artifact(),
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == "complete_a1_causal_abstraction_audited"
    assert artifact["checks"] == {check: True for check in mod.CHECK_NAMES}
    assert artifact["a1_diagnostic_trustworthy"] is True
    assert artifact["a1_failure_reasons"] == []
    assert len(artifact["observable_claims_spot_checked"]) >= 2
    assert {row["variable"] for row in artifact["observable_claims_spot_checked"]} >= {
        "visible_grid_hash",
        "action_id",
    }
    assert all(row["passed"] for row in artifact["transition_cross_checks"])
    assert artifact["numbers_match_fork_evidence"]["computed_fork_verdict"] == "WALL_IS_HIDDEN_STATE"
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4918_named_failures_are_not_trusted() -> None:
    """SCENARIO-ARC-WMTE-4918-NAMED-FAILURES: every failed check names its reason."""

    a1 = _a1_artifact(fork="WALL_IS_OBSERVABLE_VARIABLE_GAP")
    a1["is_decision_need_table_in_disguise"] = True
    a1["planner_blind_to_banked_answer"] = False
    a1["verifier_is_oracle"] = True
    a1["positive_control_rows"]["tu93"] = _a1_row("tu93", role="positive_control", hidden=True)
    a1["positive_control_classifies_observable"] = False
    a1["per_game_causal_abstraction"]["cd82"]["evidence"]["observability_proofs"][
        "visible_grid_hash"
    ] = {"observable": True, "extractor": None, "proof": "claimed without extractor"}
    exp4903 = _exp4903_artifact()
    exp4903["per_game_first_win"]["cd82"]["change_value_predictions_used"] = 1

    artifact = mod.audit_sources(
        a1_artifact=a1,
        a1_source_text=_source_text(collect_transitions=False),
        exp4903_artifact=exp4903,
        duration_s=1.0,
    )

    assert artifact["a1_diagnostic_trustworthy"] is False
    assert artifact["checks"] == {check: False for check in mod.CHECK_NAMES}
    assert set(artifact["a1_failure_reasons"]) >= {
        "real_transitions_failed",
        "decision_need_table_in_disguise",
        "observable_claims_unverified",
        "positive_control_not_observable",
        "oracle_or_planner_blind_failed",
        "numbers_do_not_match_fork",
    }
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4918_blocked_preconditions_write_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4918-BLOCKED-A1-ARTIFACT: missing inputs block trust."""

    _write_json(tmp_path / mod.EXP4903_ARTIFACT_RELATIVE_PATH, _exp4903_artifact())
    artifact = mod.run(root=tmp_path, write=True, now=iter([10.0, 10.5]).__next__)

    assert artifact["honest_verdict"] == "blocked_a1_artifact_missing"
    assert artifact["a1_diagnostic_trustworthy"] is False
    assert artifact["checks"] == {check: False for check in mod.CHECK_NAMES}
    assert "missing_experiment_4914_artifact" in artifact["a1_failure_reasons"]
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4918_run_reads_artifacts_and_writes_result(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4918: run reads cached A1/4903 inputs and writes the audit."""

    _write_json(tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH, _a1_artifact())
    (tmp_path / mod.A1_SCRIPT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.A1_SCRIPT_RELATIVE_PATH).write_text(_source_text(), encoding="utf-8")
    _write_json(tmp_path / mod.EXP4903_ARTIFACT_RELATIVE_PATH, _exp4903_artifact())

    artifact = mod.run(root=tmp_path, write=True, now=iter([20.0, 22.0]).__next__)

    assert artifact["honest_verdict"] == "complete_a1_causal_abstraction_audited"
    assert artifact["a1_diagnostic_trustworthy"] is True
    assert artifact["duration_s"] >= 1.0
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4918_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4918: final artifact is the requested audit deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert set(artifact["checks"]) == set(mod.CHECK_NAMES)
    assert artifact["a1_diagnostic_trustworthy"] is all(artifact["checks"].values())
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert len(artifact["observable_claims_spot_checked"]) >= 2
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
