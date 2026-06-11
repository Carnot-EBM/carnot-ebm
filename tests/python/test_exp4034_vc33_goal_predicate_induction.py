"""Tests for Exp 4034 vc33 held-out goal predicate induction.

Spec refs: REQ-PHASE4-036, SCENARIO-PHASE4-036.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import carnot.agentic.arc_vc33_goal_predicate_induction as mod
from carnot.agentic.arc_vc33_goal_predicate_induction import (
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    GoalExample,
    artifact_schema_errors,
    blocked_artifact,
    build_goal_induction_artifact,
    collect_observed_vc33_levelup_examples,
    compile_goal_predicate,
    evaluate_predicate,
    induce_goal_predicate_code,
    precondition_errors,
    run,
    split_examples_by_level,
    vc33_grid_state_features,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _grid_with_pair(*, aligned: bool, color: int = 11) -> np.ndarray:
    grid = np.full((16, 16), 3, dtype=np.int16)
    grid[2:4, 2:4] = color
    second_x = 2 if aligned else 8
    grid[10:12, second_x : second_x + 2] = color
    return grid


def _example(level: int, row: int, *, aligned: bool, is_goal: bool, color: int = 11) -> GoalExample:
    return GoalExample(
        state=vc33_grid_state_features(_grid_with_pair(aligned=aligned, color=color)),
        is_goal=is_goal,
        level=level,
        row_index=row,
    )


def _observed_examples() -> list[GoalExample]:
    return [
        _example(1, 0, aligned=False, is_goal=False, color=11),
        _example(1, 1, aligned=True, is_goal=True, color=11),
        _example(2, 0, aligned=False, is_goal=False, color=14),
        _example(2, 1, aligned=True, is_goal=True, color=14),
        _example(2, 2, aligned=False, is_goal=False, color=15),
    ]


def _write_vc33_preconditions(root: Path) -> None:
    (root / "results").mkdir(parents=True)
    (root / "results" / "arc3_vc33_world_model_program.py").write_text("def predict(grid, action):\n    return grid\n")
    (root / "results" / "arc3_codex_policy_vc33.json").write_text(json.dumps({"game": "vc33-5430563c"}))
    (root / "results" / "arc3_graph_explore_vc33.json").write_text(json.dumps({"n_transitions": 1}))
    (root / "results" / "world_model_vc33.json").write_text(json.dumps({"edges": {"e": {}}, "n_transitions": 1}))


def test_req_phase4_036_spec_declares_vc33_goal_predicate_contract() -> None:
    """REQ-PHASE4-036: OpenSpec declares Exp 4034 and its required artifact fields."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-036" in spec
    assert "SCENARIO-PHASE4-036" in spec
    assert "experiment_4034_vc33_goal_predicate_induction.json" in spec
    assert "blocked_vc33_world_model_missing" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_036_extracts_grid_goal_features_without_label_leakage() -> None:
    """SCENARIO-PHASE4-036: visible vc33 grid features separate aligned target pairs."""
    early = vc33_grid_state_features(_grid_with_pair(aligned=False, color=11))
    solved = vc33_grid_state_features(_grid_with_pair(aligned=True, color=11))

    assert early["target_color_pairs"] == 1
    assert early["misaligned_target_pairs"] == 1
    assert solved["target_color_pairs"] == 1
    assert solved["misaligned_target_pairs"] == 0
    assert "levels_completed" not in early
    assert "level_completed" not in solved
    try:
        vc33_grid_state_features(np.zeros((1, 2, 3), dtype=np.int16))
    except ValueError as exc:
        assert "2D grid" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("3D grids must be rejected")


def test_scenario_phase4_036_induces_sandboxed_predicate_and_heldout_metrics() -> None:
    """SCENARIO-PHASE4-036: held-out early/late states drive precision and recall."""
    train, heldout = split_examples_by_level(_observed_examples(), heldout_level_count=1)
    code = induce_goal_predicate_code(train)
    predicate = compile_goal_predicate(code)
    metrics = evaluate_predicate(predicate, heldout)

    assert 'state["misaligned_target_pairs"] == 0' in code
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    mixed_metrics = evaluate_predicate(lambda state: state["target_color_pairs"] > 0, heldout)
    assert mixed_metrics["false_positives"] == 2
    assert split_examples_by_level([], heldout_level_count=1) == ([], [])
    assert split_examples_by_level(_observed_examples(), heldout_level_count=0)[1] == []


def test_req_phase4_036_artifact_schema_requires_bare_recall_and_terminal_verdict() -> None:
    """REQ-PHASE4-036: Exp 4034 artifacts keep precision and recall as bare floats."""
    artifact = build_goal_induction_artifact(_observed_examples(), duration_s=0.125)

    assert artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete: vc33_goal_predicate_induced_heldout_precision_1.000"
    assert artifact["goal_predicate_heldout_precision"] == 1.0
    assert artifact["goal_predicate_heldout_recall"] == 1.0
    assert artifact["game"] == "vc33"
    assert artifact["n_levelup_transitions"] == 2
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE

    bad = dict(artifact)
    bad["goal_predicate_heldout_recall"] = {"value": 1.0}
    bad["honest_verdict"] = "done"
    errors = artifact_schema_errors(bad)

    assert any("goal_predicate_heldout_recall" in err for err in errors)
    assert any("honest_verdict" in err for err in errors)


def test_req_phase4_036_reports_nonseparable_and_schema_failure_modes(tmp_path: Path) -> None:
    """REQ-PHASE4-036: non-separable labels and malformed artifacts fail honestly."""
    insufficient = build_goal_induction_artifact([_example(1, 0, aligned=True, is_goal=True)])
    train_bad = build_goal_induction_artifact(
        [
            _example(1, 0, aligned=True, is_goal=True),
            _example(1, 1, aligned=True, is_goal=False),
            _example(2, 0, aligned=True, is_goal=True),
            _example(2, 1, aligned=False, is_goal=False),
        ]
    )
    heldout_bad = build_goal_induction_artifact(
        [
            _example(1, 0, aligned=False, is_goal=False),
            _example(1, 1, aligned=True, is_goal=True),
            _example(2, 0, aligned=False, is_goal=True),
            _example(2, 1, aligned=True, is_goal=False),
        ]
    )
    bad_schema = {
        "goal_predicate_heldout_precision": 1.0,
        "goal_predicate_heldout_recall": 1.0,
        "goal_predicate_code": [],
        "game": 3,
        "n_levelup_transitions": "2",
        "inference_substrate": None,
    }

    assert insufficient["honest_verdict"].endswith("insufficient_levelup_transitions")
    assert train_bad["honest_verdict"].endswith("train_examples")
    assert "not_separable_heldout_precision" in heldout_bad["honest_verdict"]
    try:
        induce_goal_predicate_code([])
    except ValueError as exc:
        assert "not_separable" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("empty training labels must not induce a predicate")
    assert any("missing required field honest_verdict" in err for err in artifact_schema_errors(bad_schema))
    assert any("goal_predicate_code" in err for err in artifact_schema_errors(bad_schema))
    assert any("n_levelup_transitions" in err for err in artifact_schema_errors(bad_schema))
    assert blocked_artifact(0.0)["honest_verdict"] == "blocked_vc33_world_model_missing"
    _write_vc33_preconditions(tmp_path)
    (tmp_path / "results" / "world_model_vc33.json").write_text(json.dumps({"edges": {}, "n_transitions": 0}))
    assert any("no saved transitions" in err for err in precondition_errors(tmp_path))
    (tmp_path / "results" / "arc3_codex_policy_vc33.json").write_text(json.dumps([]))
    assert any("arc3_codex_policy_vc33" in err for err in precondition_errors(tmp_path))


def test_req_phase4_036_internal_helpers_cover_edge_paths() -> None:
    """REQ-PHASE4-036: helper branches stay explicit for small edge cases."""
    frame = SimpleNamespace(frame=np.zeros((2, 2), dtype=np.int16))
    env = SimpleNamespace(_game=SimpleNamespace(_current_level_index=3))

    assert mod._numeric_feature_names([]) == []
    assert mod._literal(True) == "True"
    assert mod._literal(False) == "False"
    assert mod._levels_completed(object(), env) == 3
    assert mod._frame_stack(np.zeros((2, 2), dtype=np.int16)).shape == (1, 2, 2)
    assert mod._frame_stack(frame).shape == (1, 2, 2)


def test_scenario_phase4_036_run_blocks_when_vc33_world_model_missing(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-036: missing vc33 world-model substrate stops honestly."""
    artifact = run(repo_root=tmp_path, write=False, collect_examples=lambda: _observed_examples())

    assert artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "blocked_vc33_world_model_missing"
    assert artifact["goal_predicate_heldout_precision"] == 0.0
    assert artifact["goal_predicate_heldout_recall"] == 0.0
    assert artifact["goal_predicate_code"] == ""


def test_scenario_phase4_036_run_writes_mocked_observed_label_artifact(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-036: runner writes the induced vc33 predicate artifact."""
    _write_vc33_preconditions(tmp_path)

    artifact = run(repo_root=tmp_path, write=True, collect_examples=lambda: _observed_examples())
    output = tmp_path / "results" / "experiment_4034_vc33_goal_predicate_induction.json"

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith("complete: vc33_goal_predicate_induced")


def test_scenario_phase4_036_real_offline_collector_produces_levelup_examples() -> None:
    """SCENARIO-PHASE4-036: compact observed vc33 replays provide held-out labels."""
    examples = collect_observed_vc33_levelup_examples(REPO)
    artifact = build_goal_induction_artifact(examples, duration_s=0.0)

    assert sum(example.is_goal for example in examples) == 2
    assert artifact_schema_errors(artifact) == []
    assert artifact["goal_predicate_heldout_precision"] == 1.0
    assert artifact["goal_predicate_heldout_recall"] == 1.0
