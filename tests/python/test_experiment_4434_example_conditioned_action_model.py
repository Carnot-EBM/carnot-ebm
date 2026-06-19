"""Tests for Exp 4434 example-conditioned E3 world-model synthesis.

Spec refs: REQ-REPORT-4434, SCENARIO-REPORT-4434.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from carnot import experiment_4434_example_conditioned_action_model as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_fixture_repo(root: Path, *, examples: int = 4, env: bool = True) -> None:
    if env:
        (root / "environment_files" / mod.TARGET_GAME / "fixture").mkdir(parents=True)
    for game in mod.SOLVED_EXAMPLE_GAMES[:examples]:
        model_path = root / "results" / "arc_e3" / game / "world_model.py"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(
            f"def engine(grid, action, data):\n    return grid\n\nGAME = {game!r}\n",
            encoding="utf-8",
        )


def _ok_preconditions(example_count: int = 4) -> dict[str, Any]:
    return {
        "offline_env_files_present": True,
        "target_env_present": True,
        "codex_world_model_proposer": True,
        "existing_world_models": example_count,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def test_req_report_4434_spec_declares_artifact_contract() -> None:
    """REQ-REPORT-4434: OpenSpec names the cold-control artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4434" in spec
    assert "SCENARIO-REPORT-4434" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "world_model_accuracy_with_examples" in spec
    assert "world_model_accuracy_cold" in spec
    assert "verifier_is_oracle=true" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4434_gathers_world_model_examples_and_excludes_target(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4434: solved world_model.py files form the few-shot corpus."""

    _write_fixture_repo(tmp_path)
    target_path = tmp_path / mod.WORLD_MODEL_RELATIVE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    examples = mod.gather_world_model_examples(tmp_path)

    assert [row["game"] for row in examples] == list(mod.SOLVED_EXAMPLE_GAMES)
    assert all(row["game"] != mod.TARGET_GAME for row in examples)
    assert all(row["sha256"] and row["relative_path"].endswith("world_model.py") for row in examples)
    assert examples[0]["source_chars"] > 0


def test_req_report_4434_filesystem_probe_and_defensive_branches(tmp_path: Path) -> None:
    """REQ-REPORT-4434: probes and defensive model branches stay deterministic."""

    _write_fixture_repo(tmp_path, examples=1)
    examples = mod.gather_world_model_examples(
        tmp_path,
        example_games=(mod.TARGET_GAME, "missing_game", mod.SOLVED_EXAMPLE_GAMES[0]),
    )
    probe = mod.precondition_probe(tmp_path)
    blocked_probe = mod.precondition_probe(tmp_path, proposer="not-codex")

    assert [row["game"] for row in examples] == [mod.SOLVED_EXAMPLE_GAMES[0]]
    assert probe["existing_world_models"] == 1
    assert probe["ok"] is False
    assert blocked_probe["codex_world_model_proposer"] is False
    assert mod.first_precondition_miss({**_ok_preconditions(), "existing_world_models": "bad"}) == "few_shot_world_models"

    grid = np.zeros((4, 4), dtype=int)
    assert np.array_equal(mod.cold_engine(grid, 6, {"x": "bad", "y": 1}), grid)
    one_dimensional = np.asarray([1, 2, 3])
    assert np.array_equal(mod.conditioned_engine(one_dimensional, 1, {}), one_dimensional)


def test_scenario_report_4434_positive_control_writes_world_model_and_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4434: conditioning is measured against a cold arm."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["world_model_accuracy_with_examples"] == 1.0
    assert artifact["world_model_accuracy_cold"] < artifact["world_model_accuracy_with_examples"]
    assert artifact["accuracy_margin"] >= mod.REAL_MARGIN
    assert artifact["verifier_is_oracle"] is True
    assert artifact["model_specs"]["no_3090_inference"] is True
    assert artifact["model_specs"]["leaderboard_submission"] is False
    assert artifact["active_data_collection"]["balanced_actions"] is True
    assert artifact["active_data_collection"]["deadly_avoided"] is True
    assert artifact["active_data_collection"]["object_config_signature_count"] >= 2
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["world_model_output_path"] == mod.WORLD_MODEL_RELATIVE_PATH
    assert mod.artifact_schema_errors(artifact) == []

    written_model = tmp_path / mod.WORLD_MODEL_RELATIVE_PATH
    assert written_model.exists()
    assert "example-conditioned cn04 world model" in written_model.read_text(encoding="utf-8")

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["world_model_accuracy_with_examples"] == 1.0
    assert written["world_model_accuracy_cold"] == artifact["world_model_accuracy_cold"]


def test_req_report_4434_offline_reproduction_can_satisfy_gate(tmp_path: Path) -> None:
    """REQ-REPORT-4434: reproduced levels are bare ints from the reproduction gate."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        plan_fn=lambda _metrics: ["6"],
        reproduce_fn=lambda plan: {
            "reproduced": bool(plan),
            "reached_level": 1,
            "mode": "offline_reproduction_gate_fixture",
        },
        now=lambda: 20.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["explore_verify_plan"]["plan"] == ["6"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4434_no_help_result_is_complete_with_residual_gap(tmp_path: Path) -> None:
    """REQ-REPORT-4434: no-help findings are terminal complete, not partial."""

    _write_fixture_repo(tmp_path)
    cases = mod.build_active_data_cases()
    cold_metrics = mod.evaluate_world_model(cases, mod.conditioned_engine)
    with_metrics = mod.evaluate_world_model(cases, mod.conditioned_engine)
    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions=_ok_preconditions(),
        examples=mod.gather_world_model_examples(tmp_path),
        active_data_cases=cases,
        cold_metrics=cold_metrics,
        with_examples_metrics=with_metrics,
        reproduction_result={"reproduced": False, "reached_level": 0, "mode": "fixture_no_plan"},
        plan=[],
        started_at=4.0,
        ended_at=6.0,
    )

    assert artifact["honest_verdict"] == "complete: example_conditioning_no_help_missing_world_model_gap"
    assert artifact["offline_reproduced"] is False
    assert artifact["missing_verifier_gaps"] == [mod.NO_HELP_GAP]
    assert artifact["duration_s"] == 2.0
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"offline_env_files_present": False}, "offline_env_files"),
        ({"target_env_present": False}, "offline_env_cn04"),
        ({"codex_world_model_proposer": False}, "codex_world_model_proposer"),
        ({"existing_world_models": 1}, "few_shot_world_models"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4434_precondition_miss_names_each_blocking_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4434: each missing resource stops before synthesis."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}) == expected


def test_req_report_4434_blocked_artifact_has_no_fabricated_metrics(tmp_path: Path) -> None:
    """REQ-REPORT-4434: missing resources block without writing a target model."""

    _write_fixture_repo(tmp_path, env=False)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "offline_env_files_present": False, "ok": False},
        now=lambda: 1.0,
    )

    assert artifact["honest_verdict"] == "complete: blocked_offline_env_files"
    assert artifact["world_model_accuracy_with_examples"] is None
    assert artifact["world_model_accuracy_cold"] is None
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert not (tmp_path / mod.WORLD_MODEL_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4434_active_data_and_accuracy_are_deterministic() -> None:
    """SCENARIO-REPORT-4434: oracle active data balances actions and measures exact matches."""

    cases = mod.build_active_data_cases()
    summary = mod.summarize_active_data(cases)
    cold = mod.evaluate_world_model(cases, mod.cold_engine)
    conditioned = mod.evaluate_world_model(cases, mod.conditioned_engine)

    assert summary["action_counts"] == {str(action): 2 for action in range(1, 8)}
    assert summary["balanced_actions"] is True
    assert summary["deadly_avoided"] is True
    assert cold["correct"] < conditioned["correct"] == conditioned["total"]
    assert conditioned["accuracy"] == 1.0

    click_case = next(case for case in cases if case["action"] == 6)
    before = np.asarray(click_case["before"])
    observed = mod.conditioned_engine(before, 6, click_case["data"])
    assert int(observed[click_case["data"]["y"], click_case["data"]["x"]]) == 0


def test_req_report_4434_schema_rejects_partial_type_drift_and_false_positive_control() -> None:
    """REQ-REPORT-4434: schema errors expose invalid terminal artifacts."""

    missing_errors = mod.artifact_schema_errors({})
    assert "missing honest_verdict" in missing_errors
    assert "reproduced_levels must be bare int" in missing_errors
    assert "offline_reproduced must be bare bool" in missing_errors

    bad = {
        "honest_verdict": "partial: not terminal",
        "reproduced_levels": "0",
        "offline_reproduced": "false",
        "world_model_accuracy_with_examples": 0.4,
        "world_model_accuracy_cold": 0.4,
        "missing_verifier_gaps": [],
        "random_seed": "4434",
        "reproducibility_checksum": "z" * 64,
        "verifier_is_oracle": False,
        "model_specs": {"no_3090_inference": False, "leaderboard_submission": True},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "missing_verifier_gaps must list the residual gap when neither gate passes" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be hex" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "model_specs.no_3090_inference must be true" in errors
    assert "model_specs.leaderboard_submission must be false" in errors

    blocked = {**bad, "honest_verdict": "complete: blocked_offline_env_files"}
    blocked["world_model_accuracy_with_examples"] = 0.1
    blocked["world_model_accuracy_cold"] = None
    blocked_errors = mod.artifact_schema_errors(blocked)
    assert "blocked artifacts must not fabricate accuracy metrics" in blocked_errors

    impossible_reproduction = {
        **bad,
        "honest_verdict": "success: impossible",
        "reproduced_levels": 0,
        "offline_reproduced": True,
        "world_model_accuracy_with_examples": 1.0,
        "world_model_accuracy_cold": 0.0,
        "missing_verifier_gaps": [],
        "random_seed": mod.RANDOM_SEED,
        "reproducibility_checksum": "0" * 64,
        "verifier_is_oracle": True,
        "field_principles": {**mod.FIELD_PRINCIPLES, "random_seed": {"principle": "wrong"}},
        "model_specs": {"no_3090_inference": True, "leaderboard_submission": False},
    }
    impossible_errors = mod.artifact_schema_errors(impossible_reproduction)
    assert "offline_reproduced true requires reproduced_levels >= 1" in impossible_errors
    assert "field_principles.random_seed must match REQ-REPORT-4434" in impossible_errors


def test_req_report_4434_write_artifact_rejects_schema_errors(tmp_path: Path) -> None:
    """REQ-REPORT-4434: invalid artifacts are not written to the deliverable path."""

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: invalid"})
