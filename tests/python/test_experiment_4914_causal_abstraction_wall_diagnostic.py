"""Tests for Exp 4914 causal-abstraction wall diagnostic.

Spec refs: REQ-ARC-WMTE-4914,
SCENARIO-ARC-WMTE-4914-CLASSIFICATION-REPORT,
SCENARIO-ARC-WMTE-4914-POSITIVE-CONTROL,
SCENARIO-ARC-WMTE-4914-FORK-VERDICT,
SCENARIO-ARC-WMTE-4914-PARTIAL-CHECKPOINT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4914_causal_abstraction_wall_diagnostic as mod
from carnot.agentic.arc_executable_world_model import Transition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _transition(progress: bool = False) -> Transition:
    grid = np.array([[0, 0], [0, 0]], dtype=int)
    next_grid = np.array([[7, 0], [0, 0]], dtype=int)
    return Transition(
        grid=grid,
        action=6,
        data={"x": 0, "y": 0},
        next_grid=next_grid,
        level_before=0,
        level_after=1 if progress else 0,
    )


def _generator_ok(backend: str = "gpu0_cuda") -> dict[str, Any]:
    return {
        "ok": True,
        "generator_backend": backend,
        "backend": backend,
        "server": f"/fake/{backend}/llama-server",
        "model": "Qwen3.5-9B-MTP",
        "igpu_required": False,
        "launch_env_cuda_visible_devices": "0" if backend == "gpu0_cuda" else None,
    }


def _a1_artifact() -> dict[str, Any]:
    return {
        "experiment_id": 4903,
        "fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION",
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "per_game_first_win": {
            game: {
                "game": game,
                "bucket": "NEVER_ENUMERATED",
                "baseline_bucket": "NEVER_ENUMERATED",
                "migrated": False,
                "first_win_env_grounded": 0.0,
                "best_path_len": 6,
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
        "positive_control_result": {
            "game": "tu93",
            "location_ranker_non_degenerate": True,
            "true_changing_action_rank": 1,
        },
    }


def _row(
    game: str,
    *,
    classification: str = "HIDDEN_STATE",
    hidden: bool = True,
) -> dict[str, Any]:
    variables = ["visible_grid_hash", "action_id", "action_data"]
    observable = {name: True for name in variables}
    if hidden:
        variables.append("winning_prefix_order_state")
        observable["winning_prefix_order_state"] = False
    return {
        "game": game,
        "role": "failed" if game not in {"tu93", "ar25"} else "positive_control",
        "required_variables": variables,
        "observable_from_interface": observable,
        "classification": classification,
        "evidence": {
            "transition_count": 1,
            "engine_loaded": True,
            "observability_proofs": {
                key: {
                    "observable": value,
                    "extractor": "test_extractor" if value else None,
                    "proof": "test proof",
                }
                for key, value in observable.items()
            },
        },
    }


def test_req_arc_wmte_4914_spec_declares_causal_abstraction_contract() -> None:
    """REQ-ARC-WMTE-4914: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4914",
        "SCENARIO-ARC-WMTE-4914-CLASSIFICATION-REPORT",
        "SCENARIO-ARC-WMTE-4914-POSITIVE-CONTROL",
        "SCENARIO-ARC-WMTE-4914-FORK-VERDICT",
        "SCENARIO-ARC-WMTE-4914-PARTIAL-CHECKPOINT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4914_classification_report_marks_hidden_prefix_state() -> None:
    """SCENARIO-ARC-WMTE-4914-CLASSIFICATION-REPORT: failed games classify variables, not values."""

    row = {
        "game": "cn04",
        "bucket": "NEVER_ENUMERATED",
        "baseline_bucket": "NEVER_ENUMERATED",
        "migrated": False,
        "first_win_env_grounded": 0.0,
        "best_path_len": 7,
        "states_expanded": 8,
        "real_env_value_reads": 24,
        "change_value_predictions_used": 0,
    }

    classified = mod.classify_game_causal_abstraction(
        game="cn04",
        transitions=[_transition()],
        exp4903_row=row,
        role="failed",
        engine_loaded=True,
    )

    assert classified["classification"] == "HIDDEN_STATE"
    assert "winning_prefix_order_state" in classified["required_variables"]
    assert classified["observable_from_interface"]["winning_prefix_order_state"] is False
    assert classified["evidence"]["targets"] == ["changed_cell_value", "progress_to_goal"]
    assert "decision_need" not in json.dumps(classified).lower()


def test_scenario_arc_wmte_4914_positive_controls_must_be_observable() -> None:
    """SCENARIO-ARC-WMTE-4914-POSITIVE-CONTROL: solved controls gate the lens."""

    tu93 = mod.classify_game_causal_abstraction(
        game="tu93",
        transitions=[_transition(progress=True)],
        exp4903_row={"game": "tu93", "location_ranker_non_degenerate": True},
        role="positive_control",
        engine_loaded=True,
        solved_reproduced_level=5,
    )
    ar25 = mod.classify_game_causal_abstraction(
        game="ar25",
        transitions=[_transition(progress=True)],
        exp4903_row={"game": "ar25", "reproduced_levels": 3},
        role="positive_control",
        engine_loaded=True,
        solved_reproduced_level=3,
    )

    assert tu93["classification"] == "OBSERVABLE_GAP"
    assert ar25["classification"] == "OBSERVABLE_GAP"
    assert mod._positive_controls_observable({"tu93": tu93, "ar25": ar25}) is True
    assert mod._positive_controls_observable({"tu93": {**tu93, "classification": "HIDDEN_STATE"}}) is False


def test_scenario_arc_wmte_4914_fork_verdict_names_hidden_or_observable_wall() -> None:
    """SCENARIO-ARC-WMTE-4914-FORK-VERDICT: table support determines the closure fork."""

    hidden = mod.build_artifact(
        per_game_causal_abstraction={
            "cd82": _row("cd82"),
            "cn04": _row("cn04"),
            "ls20": _row("ls20"),
        },
        positive_control_rows={
            "tu93": _row("tu93", classification="OBSERVABLE_GAP", hidden=False),
            "ar25": _row("ar25", classification="OBSERVABLE_GAP", hidden=False),
        },
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=75.0,
        partial=False,
        checkpoint_emitted=True,
    )
    observable = mod.build_artifact(
        per_game_causal_abstraction={
            "cd82": _row("cd82", classification="OBSERVABLE_GAP", hidden=False),
            "cn04": _row("cn04", classification="OBSERVABLE_GAP", hidden=False),
            "ls20": _row("ls20", classification="OBSERVABLE_GAP", hidden=False),
        },
        positive_control_rows={
            "tu93": _row("tu93", classification="OBSERVABLE_GAP", hidden=False),
            "ar25": _row("ar25", classification="OBSERVABLE_GAP", hidden=False),
        },
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=75.0,
        partial=False,
        checkpoint_emitted=True,
    )
    retired = mod.build_artifact(
        per_game_causal_abstraction={"cd82": _row("cd82")},
        positive_control_rows={
            "tu93": _row("tu93", classification="HIDDEN_STATE", hidden=True),
            "ar25": _row("ar25", classification="OBSERVABLE_GAP", hidden=False),
        },
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=75.0,
        partial=False,
        checkpoint_emitted=True,
    )

    assert hidden["fork_verdict"] == "WALL_IS_HIDDEN_STATE"
    assert hidden["honest_verdict"] == "complete_causal_abstraction_hidden_state_representation_invariant_closure"
    assert hidden["minimal_abstraction_is_observable_subset"] is False
    assert observable["fork_verdict"] == "WALL_IS_OBSERVABLE_VARIABLE_GAP"
    assert observable["minimal_abstraction_is_observable_subset"] is True
    assert observable["honest_verdict"].startswith("complete_causal_abstraction_observable_variable_gap_")
    assert retired["fork_verdict"] == "DIAGNOSTIC_DEGENERATE_RETIRED"
    assert retired["positive_control_classifies_observable"] is False
    assert mod.artifact_schema_errors(hidden) == []
    assert mod.artifact_schema_errors(observable) == []
    assert mod.artifact_schema_errors(retired) == []


def test_req_arc_wmte_4914_schema_errors_are_explicit() -> None:
    """REQ-ARC-WMTE-4914: malformed artifacts fail closed with named errors."""

    artifact = mod.build_artifact(
        per_game_causal_abstraction={
            "cd82": _row("cd82"),
            "cn04": _row("cn04"),
            "ls20": _row("ls20"),
        },
        positive_control_rows={
            "tu93": _row("tu93", classification="OBSERVABLE_GAP", hidden=False),
            "ar25": _row("ar25", classification="OBSERVABLE_GAP", hidden=False),
        },
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=75.0,
        partial=False,
        checkpoint_emitted=True,
    )
    malformed = dict(artifact)
    malformed.update(
        {
            "honest_verdict": "maybe",
            "fork_verdict": "MAYBE",
            "per_game_causal_abstraction": {"cd82": {"classification": "MAYBE"}},
            "minimal_abstraction_is_observable_subset": "no",
            "positive_control_classifies_observable": "yes",
            "is_decision_need_table_in_disguise": True,
            "planner_blind_to_banked_answer": False,
            "verifier_is_oracle": True,
            "live_path_reachable": False,
            "generator_backend": "cpu",
            "solve_provenance": "live_agent_self_discovery",
            "checkpoint_emitted": "yes",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "model_specs": [],
            "reproducibility_checksum": "sha256:bad",
        }
    )

    errors = mod.artifact_schema_errors(malformed)

    assert "honest_verdict_terminal_prefix" in errors
    assert "fork_verdict" in errors
    assert "per_game_causal_abstraction.cd82.classification" in errors
    assert "minimal_abstraction_is_observable_subset" in errors
    assert "positive_control_classifies_observable" in errors
    assert "is_decision_need_table_in_disguise" in errors
    assert "planner_blind_to_banked_answer" in errors
    assert "verifier_is_oracle" in errors
    assert "live_path_reachable" in errors
    assert "generator_backend" in errors
    assert "solve_provenance" in errors
    assert "checkpoint_emitted" in errors
    assert "inference_substrate" in errors
    assert "model_specs" in errors
    assert "reproducibility_checksum" in errors


def test_scenario_arc_wmte_4914_partial_checkpoint_and_blocked_preconditions(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4914-PARTIAL-CHECKPOINT: blocked and capped runs stay valid."""

    common = {
        "root": tmp_path,
        "a1_artifact_loader": lambda _root: _a1_artifact(),
        "live_path_checker": lambda _root: True,
        "game_classifier": lambda game, role, **_kwargs: _row(
            game,
            classification="OBSERVABLE_GAP" if role == "positive_control" else "HIDDEN_STATE",
            hidden=role != "positive_control",
        ),
        "write": False,
    }
    blocked_arcade = mod.run(
        **common,
        offline_arcade_checker=lambda: False,
        generator_checker=_generator_ok,
        now=iter([1.0, 1.1]).__next__,
    )
    blocked_generator = mod.run(
        **{**common, "now": iter([2.0, 2.1]).__next__},
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: {"ok": False, "detail": "missing_qwen"},
    )
    blocked_a1 = mod.run(
        **{**common, "a1_artifact_loader": lambda _root: None, "now": iter([3.0, 3.1]).__next__},
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
    )
    partial = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        game_classifier=lambda game, role, **_kwargs: _row(
            game,
            classification="OBSERVABLE_GAP" if role == "positive_control" else "HIDDEN_STATE",
            hidden=role != "positive_control",
        ),
        now=iter([4.0, 4.1, 4.2, 5.0]).__next__,
        write=True,
        soft_elapsed_budget_s=0.05,
        failed_games=("cd82", "cn04", "ls20"),
    )

    assert blocked_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert blocked_generator["honest_verdict"] == "blocked_generator_unavailable"
    assert blocked_a1["honest_verdict"] == "blocked_a1_baseline_missing"
    assert partial["partial"] is True
    assert partial["checkpoint_emitted"] is True
    assert (tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82.json").exists()
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == partial
    assert mod.artifact_schema_errors(partial) == []


def test_req_arc_wmte_4914_run_full_resume(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4914: run aggregates failed games and resumes checkpoints."""

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        game_classifier=lambda game, role, **_kwargs: _row(
            game,
            classification="OBSERVABLE_GAP" if role == "positive_control" else "HIDDEN_STATE",
            hidden=role != "positive_control",
        ),
        now=iter([10.0, 10.1, 10.2, 10.3, 10.4, 75.0]).__next__,
        write=True,
        failed_games=("cd82", "cn04", "ls20"),
    )
    resumed = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        game_classifier=lambda *_args, **_kwargs: pytest.fail("checkpoint should be reused"),
        now=iter([80.0, 80.1]).__next__,
        write=False,
        failed_games=("cd82", "cn04", "ls20"),
    )

    assert artifact["fork_verdict"] == "WALL_IS_HIDDEN_STATE"
    assert artifact["positive_control_games"] == ["tu93", "ar25"]
    assert artifact["duration_s"] > 60.0
    assert resumed["n_games_measured"] == 3
    assert resumed["checkpoint_emitted"] is True
    assert mod.artifact_schema_errors(resumed) == []


def test_req_arc_wmte_4914_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4914: final artifact is the requested diagnostic deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert len(artifact["per_game_causal_abstraction"]) >= 3
    assert artifact["positive_control_games"] == ["tu93", "ar25"]
    assert artifact["positive_control_classifies_observable"] is True
    assert artifact["is_decision_need_table_in_disguise"] is False
    assert artifact["planner_blind_to_banked_answer"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["generator_backend"] in {"gpu0_cuda", "igpu_hip"}
    assert artifact["model_specs"]["name"] == "Qwen3.5-9B-MTP"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["duration_s"] > 60.0
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
