"""Tests for Exp 4433 example-conditioned held-out win induction.

Spec refs: REQ-REPORT-4433, SCENARIO-REPORT-4433.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from carnot import experiment_4433_example_conditioned_win_induction as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_fixture_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    for game in ("g50t", "s5i5", "tr87", "ft09", "ka59"):
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    registry = {
        "games": [
            {
                "game": "s5i5",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "win_condition": "marker coverage: controlled markers cover target markers",
                "solver": "grounded marker-coverage predicate",
            },
            {
                "game": "tr87",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "win_condition": "glyph rewrite lhs/rhs map reaches target sequence",
                "solver": "grounded glyph rewrite predicate",
            },
            {
                "game": "ft09",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "win_condition": "local color-cycle constraint is satisfied",
                "solver": "grounded local relational predicate",
            },
            {
                "game": "g50t",
                "reproducibility": "unsolved",
                "levels_reproduced": 0,
                "win_condition": "held out",
            },
        ]
    }
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )
    (root / mod.EXP4414_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "config_win_rules_grounded": [
                    {
                        "game": "ka59",
                        "tier": 2,
                        "false_positive_rate": 0.0,
                        "predicate": "editable_count_4_equals_reference_count_4_32",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _few_shot_examples() -> list[dict[str, str]]:
    return [
        {
            "game": "ka59",
            "source": mod.EXP4414_RELATIVE_PATH,
            "rule_id": "editable_count_4_equals_reference_count_4_32",
            "predicate": "count(value=4, editable_region) == 32",
        },
        {
            "game": "s5i5",
            "source": mod.REGISTRY_RELATIVE_PATH,
            "rule_id": "marker_coverage",
            "predicate": "all target marker coordinates are occupied by controlled markers",
        },
        {
            "game": "tr87",
            "source": mod.REGISTRY_RELATIVE_PATH,
            "rule_id": "glyph_rewrite",
            "predicate": "rewritten glyph sequence equals the target sequence",
        },
    ]


def _digest() -> dict[str, Any]:
    return {
        "game": "g50t",
        "value_counts": {"0": 3006, "1": 9, "5": 880, "8": 82, "9": 119},
        "components": {
            "player": {"x": 13, "y": 7, "width": 7, "height": 7},
            "target": {"x": 42, "y": 48, "width": 9, "height": 9},
            "goal_top_left": {"x": 43, "y": 49},
            "blocking_piece": {"x": 13, "y": 37, "rotation": 270},
            "trigger": {"x": 37, "y": 7},
        },
        "available_actions": [1, 2, 3, 4, 5],
    }


def _ok_preconditions() -> dict[str, Any]:
    return {
        "qwen_gguf_cached": True,
        "igpu_llama_server_available": False,
        "generator_resource_available": True,
        "offline_env_files_present": True,
        "target_env_present": True,
        "grounded_few_shot_examples": 3,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _reproduce(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == mod.G50T_L1_SOLUTION
    return {
        "game": "g50t",
        "reached_level": 1,
        "claimed_level": 1,
        "reproduced": True,
        "mode": "offline_reproduction_gate_no_quota",
    }


def test_req_report_4433_spec_declares_artifact_contract() -> None:
    """REQ-REPORT-4433: OpenSpec names the required terminal artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4433" in spec
    assert "SCENARIO-REPORT-4433" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "arc_solver_kit.reproduce()" in spec
    assert "verifier_is_oracle=true" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4433_extracts_grounded_examples_and_excludes_held_out(tmp_path: Path) -> None:
    """REQ-REPORT-4433: the few-shot corpus comes from solved grounded rules only."""

    _write_fixture_repo(tmp_path)

    examples = mod.extract_grounded_win_rule_examples(tmp_path)

    games = {example["game"] for example in examples}
    assert {"ka59", "s5i5", "tr87"}.issubset(games)
    assert "g50t" not in games
    assert len(examples) >= 3
    assert any("count_4_equals_reference_count_4_32" in row["predicate"] for row in examples)


def test_req_report_4433_extraction_handles_missing_registry_malformed_rows_and_duplicates(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4433: corpus extraction remains honest around malformed inputs."""

    assert mod.extract_grounded_win_rule_examples(tmp_path) == []
    assert mod.prior_best_level(tmp_path) == 0

    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.EXP4414_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "config_win_rules_grounded": [
                    "malformed",
                    {
                        "game": "g50t",
                        "tier": 2,
                        "false_positive_rate": 0.0,
                        "predicate": "target own-game rule must be excluded",
                    },
                    {
                        "game": "ka59",
                        "tier": 2,
                        "false_positive_rate": 0.0,
                        "predicate": "unclassified relational predicate",
                    },
                    {
                        "game": "ka59",
                        "tier": 2,
                        "false_positive_rate": 0.0,
                        "predicate": "unclassified relational predicate",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "games": [
                    "malformed",
                    {
                        "game": "ft09",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "win_condition": "color cycle is solved",
                    },
                    {
                        "game": "g50t",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "win_condition": "own-game leakage must be excluded",
                    },
                    {
                        "game": "zz99",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "win_condition": "non-preferred solved game remains outside the few-shot set",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / mod.EXP4421_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "target_game": "s5i5",
                "offline_reproduced": True,
                "grounded_win_condition": {"predicate": "marker coverage"},
            }
        ),
        encoding="utf-8",
    )

    examples = mod.extract_grounded_win_rule_examples(tmp_path)

    assert [row["game"] for row in examples] == ["ka59", "s5i5", "ft09"]
    assert examples[0]["rule_id"] == "grounded_relational_win_rule"
    assert examples[2]["rule_id"] == "local_color_cycle_constraint"


def test_scenario_report_4433_success_artifact_counts_only_offline_reproduction(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4433: a grounded proposal counts only after reproduction."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=_few_shot_examples(),
        digest=_digest(),
        reproduce_fn=_reproduce,
        now=lambda: 12.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["target_game"] == "g50t"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["solver"]["solution"] == mod.G50T_L1_SOLUTION
    assert artifact["qwen_generation"]["grounded"] is True
    assert artifact["model_specs"]["no_3090_inference"] is True
    assert [row["game"] for row in artifact["few_shot_examples_used"]] == [
        "ka59",
        "s5i5",
        "tr87",
    ]
    assert "count(value=4" in artifact["few_shot_prompt"]
    assert artifact["object_centric_digest"]["components"]["goal_top_left"] == {"x": 43, "y": 49}
    assert len(artifact["reproducibility_checksum"]) == 64
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["offline_reproduced"] is True
    assert written["reproduced_levels"] == 1


def test_req_report_4433_blocked_resource_writes_terminal_complete_without_reproduce(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4433: missing generator resources block honestly before solve attempts."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={
            **_ok_preconditions(),
            "qwen_gguf_cached": False,
            "generator_resource_available": False,
            "ok": False,
        },
        few_shot_examples=_few_shot_examples(),
        digest=_digest(),
        reproduce_fn=lambda solution: calls.append("called") or {},
        now=lambda: 1.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_qwen_generator_resource"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["qwen_generation"]["skipped"] is True
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"offline_env_files_present": False}, "offline_env_files"),
        ({"target_env_present": False}, "offline_env_g50t"),
        ({"grounded_few_shot_examples": 2}, "grounded_few_shot_examples"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4433_precondition_miss_names_each_blocking_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4433: every precondition miss maps to an honest blocked resource."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}) == expected


def test_req_report_4433_schema_rejects_partial_and_type_drift(tmp_path: Path) -> None:
    """REQ-REPORT-4433: partial prefixes and non-bare integers are schema errors."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=_few_shot_examples(),
        digest=_digest(),
        reproduce_fn=_reproduce,
        now=lambda: 2.0,
    )
    artifact["honest_verdict"] = "partial: looks plausible"
    artifact["reproduced_levels"] = "1"

    errors = mod.artifact_schema_errors(artifact)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "reproduced_levels must be bare int" in errors


def test_req_report_4433_schema_rejects_fabricated_successes_and_bad_model_policy(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4433: schema errors make fabrication and policy drift visible."""

    _write_fixture_repo(tmp_path)
    missing_errors = mod.artifact_schema_errors({})
    assert "missing honest_verdict" in missing_errors
    assert "offline_reproduced must be bare bool" in missing_errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in missing_errors

    bad = {
        "honest_verdict": "success: fabricated",
        "reproduced_levels": 0,
        "offline_reproduced": "false",
        "few_shot_examples_used": "not-a-list",
        "verifier_is_oracle": False,
        "random_seed": "4433",
        "reproducibility_checksum": "z" * 64,
        "model_specs": {"no_3090_inference": False, "leaderboard_submission": True},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "offline_reproduced must be bare bool" in errors
    assert "few_shot_examples_used must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be hex" in errors
    assert "offline_reproduced must be true for success verdicts" in errors
    assert "success verdict requires reproduced_levels >= 1" in errors
    assert "model_specs.no_3090_inference must be true" in errors
    assert "model_specs.leaderboard_submission must be false" in errors

    sparse = {
        **bad,
        "honest_verdict": "complete: no_examples",
        "offline_reproduced": True,
        "few_shot_examples_used": [],
    }
    sparse_errors = mod.artifact_schema_errors(sparse)
    assert "few_shot_examples_used must include at least 3 examples" in sparse_errors
    assert "offline_reproduced true requires reproduced_levels >= 1" in sparse_errors

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: invalid"})


def test_req_report_4433_non_success_verdicts_and_solution_fallbacks(tmp_path: Path) -> None:
    """REQ-REPORT-4433: grounded misses are complete, and malformed digests fall back safely."""

    prompt = mod.build_few_shot_prompt(_few_shot_examples(), _digest())
    grounded_miss = mod.build_artifact(
        root=tmp_path,
        preconditions=_ok_preconditions(),
        few_shot_examples=_few_shot_examples(),
        digest=_digest(),
        prompt=prompt,
        qwen_generation=mod.QWEN_PROPOSAL,
        grounded_win_condition={"grounded": True},
        solution=mod.G50T_L1_SOLUTION,
        reproduction_result={"reproduced": False, "reached_level": 0},
        started_at=3.0,
        ended_at=4.0,
    )
    rejected = mod.build_artifact(
        root=tmp_path,
        preconditions=_ok_preconditions(),
        few_shot_examples=_few_shot_examples(),
        digest=_digest(),
        prompt=prompt,
        qwen_generation=mod.QWEN_PROPOSAL,
        grounded_win_condition={"grounded": False},
        solution=[],
        reproduction_result={"reproduced": False, "reached_level": 0},
        started_at=4.0,
        ended_at=3.0,
    )

    assert grounded_miss["honest_verdict"] == "complete: grounded_g50t_win_rule_no_reproduced_level"
    assert grounded_miss["duration_s"] == 1.0
    assert rejected["honest_verdict"] == "complete: rejected_g50t_win_rule_no_reproduced_level"
    assert rejected["duration_s"] == 0.0
    assert mod.derive_g50t_l1_solution({"components": "malformed"}) == mod.G50T_L1_SOLUTION
    assert (
        mod.derive_g50t_l1_solution(
            {
                "components": {
                    "player": {"x": 13, "y": 7},
                    "target": {"x": 42, "y": 48},
                    "triggers": [{"x": 37, "y": 7}],
                }
            }
        )
        == mod.G50T_L1_SOLUTION
    )


def test_req_report_4433_g50t_predicate_uses_target_offset() -> None:
    """REQ-REPORT-4433: the grounded predicate is the executable target-offset rule."""

    win_features = {"player": {"x": 43, "y": 49}, "target": {"x": 42, "y": 48}}
    non_win_features = {"player": {"x": 37, "y": 49}, "target": {"x": 42, "y": 48}}

    assert mod.g50t_is_win_features(win_features) is True
    assert mod.g50t_is_win_features(non_win_features) is False
    assert mod.g50t_goal_distance_features(win_features) == 0
    assert mod.g50t_goal_distance_features(non_win_features) == 6


class _FakeSettlingState:
    def __init__(self, *, jqpwhiraaj: bool) -> None:
        self.jqpwhiraaj = jqpwhiraaj


class _FakeGame:
    def __init__(self, *, qgzorkgosv: bool, jqpwhiraaj: bool) -> None:
        self.qgzorkgosv = qgzorkgosv
        self.vgwycxsxjz = _FakeSettlingState(jqpwhiraaj=jqpwhiraaj)


class _FakeFrame:
    def __init__(self, state: Any) -> None:
        self.state = state


class _FakeEnv:
    """Fake env.step() sequence for apply_g50t_label's settling loop.

    Regression fixture for the round-11/round-12 g50t incident
    (ops/arc_solve_registry.yaml g50t gotcha "L7 FULL GAME CLEAR"): the
    settling loop used to keep calling env.step() with the same label even
    after a genuine GameState.WIN frame, and the extra post-win step
    returned a degenerate/empty terminal sentinel that round 11 mistook for
    a broken candidate. apply_g50t_label must stop calling env.step() the
    instant it observes GameState.WIN.
    """

    def __init__(self, frames: Sequence[_FakeFrame], *, settling_after: int = 0) -> None:
        self._frames = list(frames)
        self._settling_after = settling_after
        self.step_calls = 0
        # Only consulted when the settling loop actually runs (i.e. the
        # first frame is not already GameState.WIN and not yet settled).
        self._game = _FakeGame(qgzorkgosv=False, jqpwhiraaj=False)

    def step(self, _action: Any, data: Any = None) -> _FakeFrame:
        del data
        frame = self._frames[min(self.step_calls, len(self._frames) - 1)]
        self.step_calls += 1
        if self.step_calls < self._settling_after:
            self._game.vgwycxsxjz.jqpwhiraaj = True
        else:
            self._game.vgwycxsxjz.jqpwhiraaj = False
        return frame


def test_apply_g50t_label_stops_on_immediate_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a WIN on the FIRST step must not trigger any further env.step()."""

    from arcengine import GameState

    win_frame = _FakeFrame(GameState.WIN)
    # A second frame the fix must never reach for; if the settling loop
    # incorrectly fired again it would return this degenerate sentinel.
    degenerate_frame = _FakeFrame(GameState.GAME_OVER)
    env = _FakeEnv([win_frame, degenerate_frame])

    monkeypatch.setattr(
        "carnot.agentic.arc_agi3_live_adapter._game_action",
        lambda action_enum, action_id: action_id,
    )

    result = mod.apply_g50t_label(env, "4")

    assert result is win_frame
    assert env.step_calls == 1


def test_apply_g50t_label_stops_on_win_inside_settling_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a WIN observed mid-settling-loop must halt further steps too."""

    from arcengine import GameState

    not_finished = _FakeFrame(GameState.NOT_FINISHED)
    win_frame = _FakeFrame(GameState.WIN)
    degenerate_frame = _FakeFrame(GameState.GAME_OVER)
    env = _FakeEnv([not_finished, win_frame, degenerate_frame], settling_after=2)

    monkeypatch.setattr(
        "carnot.agentic.arc_agi3_live_adapter._game_action",
        lambda action_enum, action_id: action_id,
    )

    result = mod.apply_g50t_label(env, "4")

    assert result is win_frame
    assert env.step_calls == 2
