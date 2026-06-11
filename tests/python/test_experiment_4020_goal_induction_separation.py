"""Tests for Exp 4020 held-out ARC-AGI-3 goal predicate separation.

Spec coverage: REQ-PHASE4-029, SCENARIO-PHASE4-029.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.agentic.arc_goal_predicate_separation import (
    GoalExample,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_goal_induction_artifact,
    compile_goal_predicate,
    derive_examples_from_verifier_artifact,
    evaluate_predicate,
    induce_goal_predicate_code,
    main,
    run,
    split_examples_by_level,
    write_artifact,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _cached_r11l_verifier_artifact() -> dict[str, object]:
    return {
        "game": "r11l-495a7899",
        "honest_verdict": "success: verifier_validated_reinduction_advanced_r11l_to_L3",
        "inference_substrate": "offline_arc_agi3_gap4_executed_consistency_verifier_validated_reinduction",
        "real_env_confirmed": True,
        "ACCURACY_levels_solved": 3,
        "level_summaries": [
            {"level": 1, "levels_completed_after": 1},
            {"level": 2, "levels_completed_after": 2},
            {"level": 3, "levels_completed_after": 3},
        ],
        "per_level": [
            {"level": 2, "levels_completed_after": 2},
            {"level": 3, "levels_completed_after": 3},
        ],
        "solve_log": [
            {"level": 1, "group_id": "pumlzd", "target_after_collides": False},
            {"level": 1, "group_id": "pumlzd", "target_after_collides": True},
            {"level": 2, "group_id": "pumlzd", "target_after_collides": True},
            {"level": 2, "group_id": "orrqlj", "target_after_collides": False},
            {"level": 2, "group_id": "orrqlj", "target_after_collides": True},
            {"level": 3, "group_id": "pumlzd", "target_after_collides": True},
            {"level": 3, "group_id": "grhcew", "target_after_collides": False},
            {"level": 3, "group_id": "grhcew", "target_after_collides": True},
        ],
    }


def test_req_phase4_029_spec_declares_goal_induction_artifact_fields() -> None:
    """REQ-PHASE4-029: OpenSpec declares Exp 4020 and required artifact fields."""
    spec = SPEC_PATH.read_text("utf-8")

    assert "REQ-PHASE4-029" in spec
    assert "SCENARIO-PHASE4-029" in spec
    assert "without running a new environment exploration sweep" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_phase4_029_induces_restricted_goal_code_from_cached_levelups() -> None:
    """SCENARIO-PHASE4-029: cached r11l level-ups induce an exact held-out predicate."""
    examples = derive_examples_from_verifier_artifact(_cached_r11l_verifier_artifact())
    train, heldout = split_examples_by_level(examples, heldout_level_count=1)
    code = induce_goal_predicate_code(train)
    predicate = compile_goal_predicate(code)
    metrics = evaluate_predicate(predicate, heldout)

    assert sum(example.is_goal for example in examples) == 3
    assert all("levels_completed" not in example.state for example in examples)
    assert 'state["unsatisfied_targets"] == 0' in code
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_req_phase4_029_sandbox_rejects_unsafe_goal_code() -> None:
    """REQ-PHASE4-029: Exp 4020 executes predicates only in the restricted sandbox."""
    safe = compile_goal_predicate('def is_goal(state):\n    return state["unsatisfied_targets"] == 0\n')

    assert safe({"unsatisfied_targets": 0}) is True
    assert safe({"unsatisfied_targets": 1}) is False
    with pytest.raises(ValueError, match="restricted"):
        compile_goal_predicate("import os\n\ndef is_goal(state):\n    return True\n")
    with pytest.raises(ValueError, match="restricted"):
        compile_goal_predicate('def is_goal(state):\n    return state.get("unsatisfied_targets") == 0\n')


def test_scenario_phase4_029_artifact_schema_and_write_path(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-029: Exp 4020 writes a valid JSON artifact from cached labels."""
    artifact = build_goal_induction_artifact(
        _cached_r11l_verifier_artifact(),
        source_artifact="results/experiment_3992_incremental_levels_verifier_validated.json",
        seed=4020,
        duration_s=0.25,
    )
    output = write_artifact(artifact, tmp_path / "experiment_4020_goal_induction_separation.json")

    assert artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete: goal_predicate_induced_heldout_precision_1.000"
    assert artifact["goal_predicate_heldout_precision"] == 1.0
    assert artifact["game"] == "r11l-495a7899"
    assert artifact["n_levelup_transitions"] == 3
    assert "no_new_env_exploration_sweep" in artifact["inference_substrate"]
    assert json.loads(output.read_text("utf-8")) == artifact


def test_scenario_phase4_029_reports_not_separable_for_insufficient_levelups() -> None:
    """SCENARIO-PHASE4-029: fewer than two train level-ups blocks induction honestly."""
    source = _cached_r11l_verifier_artifact()
    source["level_summaries"] = [{"level": 1, "levels_completed_after": 1}]
    source["per_level"] = []
    source["solve_log"] = [
        {"level": 1, "group_id": "pumlzd", "target_after_collides": False},
        {"level": 1, "group_id": "pumlzd", "target_after_collides": True},
    ]

    artifact = build_goal_induction_artifact(source, duration_s=0.1)

    assert artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete: goal_predicate_not_separable_insufficient_levelup_transitions"
    assert artifact["goal_predicate_heldout_precision"] == 0.0
    assert artifact["goal_predicate_code"] == ""
    assert artifact["n_levelup_transitions"] == 1


def test_req_phase4_029_derivation_skips_unusable_rows_and_legacy_l1_fields() -> None:
    """REQ-PHASE4-029: cached rows without richer collision fields remain usable."""
    source = {
        "game": "r11l-495a7899",
        "level_summaries": [
            {"level": "oops", "levels_completed_after": "1"},
            {"level": "1", "levels_completed_after": "1"},
        ],
        "solve_log": [
            {"group_id": "ignored"},
            {"level": "oops", "group_id": "ignored"},
            {"level": 1, "group_id": "pumlzd"},
            {"level": 1, "group_id": "pumlzd"},
            {"level": 2, "group_id": "unfinished", "target_after_collides": True},
        ],
    }

    examples = derive_examples_from_verifier_artifact(source)

    assert len(examples) == 2
    assert examples[0].is_goal is False
    assert examples[0].state["unsatisfied_targets"] == 1
    assert examples[1].is_goal is True
    assert examples[1].state["unsatisfied_targets"] == 0
    assert split_examples_by_level(examples, heldout_level_count=0)[1] == []
    assert split_examples_by_level([], heldout_level_count=1) == ([], [])


def test_req_phase4_029_inducer_covers_threshold_and_not_separable_cases() -> None:
    """REQ-PHASE4-029: separability induction handles >= thresholds and failures."""
    threshold_train = [
        GoalExample({"score": 5}, True, 1, 0),
        GoalExample({"score": 6}, True, 2, 0),
        GoalExample({"score": 2}, False, 1, 1),
        GoalExample({"score": 1}, False, 2, 1),
    ]
    code = induce_goal_predicate_code(threshold_train)

    assert 'state["score"] >= 5' in code
    with pytest.raises(ValueError, match="not_separable"):
        induce_goal_predicate_code(
            [
                GoalExample({"overlap": 1}, True, 1, 0),
                GoalExample({"overlap": 1}, False, 1, 1),
            ]
        )
    with pytest.raises(ValueError, match="not_separable"):
        induce_goal_predicate_code([])


def test_req_phase4_029_sandbox_rejects_malformed_signatures_and_lookups() -> None:
    """REQ-PHASE4-029: sandbox failures are explicit for malformed generated code."""
    bad_snippets = [
        "def is_goal(",
        "def is_goal(state):\n    return",
        "def nope(state):\n    return True\n",
        "def is_goal(state, extra):\n    return True\n",
        "@state\ndef is_goal(state):\n    return True\n",
        "def is_goal(state):\n    return other == 0\n",
        'def is_goal(state):\n    return other["x"] == 0\n',
        'def is_goal(state):\n    return state["x"]["y"] == 0\n',
        "def is_goal(state):\n    return state[0] == 0\n",
    ]

    for snippet in bad_snippets:
        with pytest.raises(ValueError, match="restricted"):
            compile_goal_predicate(snippet)


def test_req_phase4_029_evaluation_counts_early_and_late_fires() -> None:
    """REQ-PHASE4-029: held-out precision drops for early fires and recall drops for late fires."""
    examples = [
        GoalExample({"flag": 1}, True, 1, 0),
        GoalExample({"flag": 1}, False, 1, 1),
        GoalExample({"flag": 0}, True, 2, 0),
    ]

    metrics = evaluate_predicate(lambda state: state["flag"] == 1, examples)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_req_phase4_029_artifact_schema_reports_type_errors() -> None:
    """REQ-PHASE4-029: schema validation rejects hidden or non-bare fields."""
    bad = {
        "honest_verdict": "done",
        "goal_predicate_heldout_precision": "1.0",
        "goal_predicate_code": [],
        "game": 123,
        "n_levelup_transitions": "3",
        "inference_substrate": None,
    }
    missing_errors = artifact_schema_errors({})
    type_errors = artifact_schema_errors(bad)

    assert any("missing required field honest_verdict" in err for err in missing_errors)
    assert any("honest_verdict" in err for err in type_errors)
    assert any("goal_predicate_heldout_precision" in err for err in type_errors)
    assert any("goal_predicate_code" in err for err in type_errors)
    assert any("game" in err for err in type_errors)
    assert any("n_levelup_transitions" in err for err in type_errors)
    assert any("inference_substrate" in err for err in type_errors)


def test_scenario_phase4_029_reports_train_and_heldout_nonseparation() -> None:
    """SCENARIO-PHASE4-029: non-separable train or held-out splits get honest verdicts."""
    train_bad = _cached_r11l_verifier_artifact()
    train_bad["level_summaries"] = [
        {"level": 1, "levels_completed_after": 1},
        {"level": 2, "levels_completed_after": 2},
        {"level": 3, "levels_completed_after": 3},
    ]
    train_bad["per_level"] = []
    train_bad["solve_log"] = [
        {"level": 1, "group_id": "a", "target_after_collides": True},
        {"level": 2, "group_id": "b", "target_after_collides": True},
        {"level": 3, "group_id": "c", "target_after_collides": True},
    ]
    heldout_bad = _cached_r11l_verifier_artifact()
    heldout_bad["solve_log"][-1]["target_after_collides"] = False

    train_artifact = build_goal_induction_artifact(train_bad)
    heldout_artifact = build_goal_induction_artifact(heldout_bad)

    assert train_artifact["honest_verdict"] == "complete: goal_predicate_not_separable_train_examples"
    assert heldout_artifact["honest_verdict"] == (
        "complete: goal_predicate_not_separable_heldout_precision_0.000_recall_0.000"
    )
    assert heldout_artifact["heldout_false_negatives"] == 1


def test_scenario_phase4_029_run_and_cli_write_cached_artifact(tmp_path: Path, monkeypatch, capsys) -> None:
    """SCENARIO-PHASE4-029: module runner and CLI write the same artifact schema."""
    source = tmp_path / "source.json"
    first_output = tmp_path / "first.json"
    cli_output = tmp_path / "cli.json"
    missing_source = tmp_path / "missing.json"
    source.write_text(json.dumps(_cached_r11l_verifier_artifact()), "utf-8")

    artifact = run(source_path=source, output_path=first_output, write=True)
    missing = run(source_path=missing_source, output_path=tmp_path / "missing_out.json", write=False)
    monkeypatch.setattr(sys, "argv", ["exp4020", "--source", str(source), "--output", str(cli_output)])
    main()

    assert first_output.exists()
    assert artifact_schema_errors(artifact) == []
    assert missing["honest_verdict"] == "complete: goal_predicate_not_separable_insufficient_levelup_transitions"
    assert cli_output.exists()
    assert "goal_predicate_induced" in capsys.readouterr().out
