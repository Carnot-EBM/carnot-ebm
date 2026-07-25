"""Tests for Exp 4512 imitation action prior.

Spec refs: REQ-ARC-FCP-4512, SCENARIO-ARC-FCP-4512.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_4512_imitation_action_prior as exp4512
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic import arc_graph_explore
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_competition_agent import StepwiseExplorer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(value: int = 0) -> SimpleNamespace:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[1, 1] = int(value)
    grid[6, 6] = int(value) + 1
    return SimpleNamespace(frame=grid, available_actions=[1, 2, 3, 6])


def _example(
    game_id: str,
    action_id: int,
    changed: bool = True,
    *,
    x: int | None = None,
    y: int | None = None,
    value: int = 1,
) -> fcp.FrameActionEffectExample:
    frame = _frame(value)
    return fcp.FrameActionEffectExample(
        frame=frame,
        action_id=action_id,
        x=x,
        y=y,
        frame_delta=0.25 if changed else 0.0,
        level_progress=1.0 if changed else 0.0,
        state_key=fcp.frame_state_key(frame),
        env=game_id,
        step_index=1,
    )


def _write_effect_shards(root: Path, rows: list[dict[str, object]]) -> None:
    data_dir = root / exp4512.HUMAN_REPLAY_DATA_RELATIVE_DIR
    shard_dir = data_dir / "shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "train-00000.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    shard_path.write_text(payload, encoding="utf-8")
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "carnot.arc_human_replay.frame_action_delta.v1",
                "example_count": len(rows),
                "shard_count": 1,
                "shards": [{"path": "shards/train-00000.jsonl", "rows": len(rows)}],
                "source_metadata": {"source_kind": "unit_fixture"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _row(action_id: int, *, frame: list[list[int]], changed: bool = True) -> dict[str, object]:
    return {
        "schema": "carnot.arc_human_replay.frame_action_delta.v1",
        "env": "aa00",
        "guid": "fixture",
        "step_index": 1,
        "frame": frame,
        "action": {"id": str(action_id), "data": {}},
        "frame_delta": 1.0 if changed else 0.0,
        "level_progress": 1.0 if changed else 0.0,
    }


def test_req_arc_fcp_4512_spec_declares_imitation_artifact_contract() -> None:
    """REQ-ARC-FCP-4512: OpenSpec anchors the imitation prior artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4512" in spec
    assert "SCENARIO-ARC-FCP-4512" in spec
    assert exp4512.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4512.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4512_prior_orders_and_prunes_bottom_quantile() -> None:
    """SCENARIO-ARC-FCP-4512: prior likelihood orders and prunes candidates."""

    frame = _frame(3)
    frame_class = exp4512.frame_class_key(frame)
    prior = exp4512.ActionTypeSequencePrior(
        prior_source="unit",
        frame_class_action_counts={frame_class: {2: 2.0, 6: 8.0}},
        marginal_action_counts={2: 2.0, 3: 1.0, 6: 8.0},
        sequence_action_counts={1: {2: 1.0, 6: 9.0}},
        history=(1,),
    )
    candidates = [
        ArcAction(3, None, "low_keyboard"),
        ArcAction(2, None, "medium_keyboard"),
        ArcAction(6, {"x": 6, "y": 6}, "human_like_click"),
    ]

    ranked = fcp.rank_arc_actions(frame, candidates, prior=prior)
    kept, diagnostics = fcp.prune_arc_actions_by_prior_quantile(
        frame,
        candidates,
        prior=prior,
        prune_quantile=1 / 3,
    )

    assert ranked[0].source == "human_like_click"
    assert [candidate.source for candidate in kept] == ["medium_keyboard", "human_like_click"]
    assert diagnostics["enabled"] is True
    assert diagnostics["pruned_count"] == 1
    assert diagnostics["forced_keep_count"] == 0


def test_req_arc_fcp_4512_rich_candidates_and_explorer_accept_action_prior(
    monkeypatch,
) -> None:
    """REQ-ARC-FCP-4512: explorer candidate expansion passes action prior through."""

    frame = _frame(4)
    frame_class = exp4512.frame_class_key(frame)
    prior = exp4512.ActionTypeSequencePrior(
        prior_source="unit",
        frame_class_action_counts={frame_class: {1: 1.0, 6: 9.0}},
        marginal_action_counts={1: 1.0, 6: 9.0},
    )
    monkeypatch.setattr(
        arc_graph_explore,
        "_components_detailed",
        lambda _grid: [(1, 1, 1, 1), (6, 6, 1, 2)],
    )

    ranked = arc_graph_explore.rich_action_candidates(
        frame,
        by_salience=False,
        action_prior=prior,
        action_prior_prune_quantile=0.50,
    )
    explorer = StepwiseExplorer(
        action_prior=prior,
        action_prior_prune_quantile=0.50,
    )
    explorer_ranked = explorer._candidates(frame)

    assert ranked[0].action_id == 6
    assert ranked[0].data == {"x": 1, "y": 1}
    assert all(candidate.action_id != 3 for candidate in ranked)
    # 2026-07-25: rows now carry an additive 'tier' annotation when the frontier tier barrier
    # is enabled (shipped ON -- see the flag block in arc_competition_agent.py). Assert the
    # MEANINGFUL fields rather than exact dict equality, which was brittle to any new annotation.
    assert explorer_ranked[0]["action"] == 6
    assert explorer_ranked[0]["data"] == {"x": 1, "y": 1}


def test_req_arc_fcp_4512_uses_human_replays_before_fallback(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4512: human replay rows are primary when the corpus is loadable."""

    frame = _frame(5).frame.tolist()
    _write_effect_shards(
        tmp_path,
        [
            _row(1, frame=frame, changed=False),
            _row(6, frame=frame, changed=True),
        ],
    )

    def fallback_collector(**_kwargs):
        raise AssertionError("fallback should not run when human shards are loadable")

    bundle = exp4512.build_imitation_prior(
        root=tmp_path,
        fallback_collector=fallback_collector,
    )

    assert bundle.prior_source == "human_replay_corpus"
    assert bundle.summary["human_examples_loaded"] == 2
    assert bundle.summary["prior_examples_used"] == 1
    assert bundle.prior.marginal_action_counts == {6: 1.0}


def test_req_arc_fcp_4512_falls_back_to_self_supervised_marginal(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4512: missing human rows use the offline-arcade fallback."""

    fallback_examples = [
        _example("aa00", 1, changed=False, value=1),
        _example("bb00", 2, changed=True, value=2),
    ]

    def fallback_collector(**_kwargs):
        return fallback_examples, {
            "transition_count": 2,
            "changed_count": 1,
            "corpus_source": "self_supervised_offline_arcade_transitions",
        }

    bundle = exp4512.build_imitation_prior(
        root=tmp_path,
        fallback_collector=fallback_collector,
    )

    assert bundle.prior_source == "self_supervised_marginal_fallback"
    assert bundle.summary["fallback_examples_loaded"] == 2
    assert bundle.prior.marginal_action_counts == {2: 1.0}


def test_scenario_arc_fcp_4512_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4512: run writes required JSON and solve-rate guard fields."""

    prior = exp4512.ActionTypeSequencePrior(
        prior_source="human_replay_corpus",
        frame_class_action_counts={},
        marginal_action_counts={6: 3.0},
    )
    bundle = exp4512.PriorBundle(
        prior=prior,
        prior_source="human_replay_corpus",
        examples=[
            _example("aa00", 6, changed=True, x=6, y=6),
            _example("bb00", 6, changed=True, x=5, y=5),
        ],
        summary={"prior_examples_used": 2, "human_examples_loaded": 2},
    )

    def prior_builder(**_kwargs):
        return bundle

    def measure_gate(**_kwargs):
        return {
            "baseline": {
                "solved_count": 4,
                "median_actions_on_solved": exp4512.BASELINE_MEDIAN_ACTIONS,
                "games": ["aa00", "bb00"],
                "per_game": [
                    {"game": "aa00", "solved": True, "actions": 7792},
                    {"game": "bb00", "solved": False, "actions": 8000},
                ],
            },
            "with_prior": {
                "solved_count": 4,
                "median_actions_on_solved": 120.0,
                "games": ["aa00", "bb00"],
                "per_game": [
                    {
                        "game": "aa00",
                        "solved": True,
                        "actions": 120,
                        "reproduced": True,
                    },
                    {
                        "game": "bb00",
                        "solved": False,
                        "actions": 8000,
                        "reproduced": None,
                    },
                ],
            },
            "measurement_script": exp4512.LOCAL_GATE_RELATIVE_PATH,
        }

    artifact = exp4512.run(
        root=tmp_path,
        write=True,
        prior_builder=prior_builder,
        measure_gate=measure_gate,
        random_seed=4512,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "success: imitation_prior_median_actions_120_below_7760"
    assert artifact["median_actions_baseline"] == exp4512.BASELINE_MEDIAN_ACTIONS
    assert artifact["median_actions_with_prior"] == 120.0
    assert artifact["solve_rate_baseline"] == 4
    assert artifact["solve_rate_with_prior"] == 4
    assert artifact["prior_source"] == "human_replay_corpus"
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert exp4512.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4512.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]


def test_req_arc_fcp_4512_artifact_schema_rejects_bad_fields() -> None:
    """REQ-ARC-FCP-4512: schema rejects unsafe or non-terminal artifacts."""

    artifact = exp4512.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        prior_summary={"prior_examples_used": 1},
        gate_metrics={
            "baseline": {"solved_count": 4, "median_actions_on_solved": 7760},
            "with_prior": {"solved_count": 4, "median_actions_on_solved": 100},
        },
        positive_control={"actions_reduced": True},
        prior_source="human_replay_corpus",
        random_seed=4512,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=1.0,
    )

    assert exp4512.artifact_schema_errors(artifact) == []

    bad_verdict = {**artifact, "honest_verdict": "blocked"}
    assert any("terminal prefix" in error for error in exp4512.artifact_schema_errors(bad_verdict))

    dropped_success = {
        **artifact,
        "honest_verdict": "success: imitation_prior_median_actions_100_below_7760",
        "solve_rate_with_prior": 3,
    }
    assert any("solve-rate" in error for error in exp4512.artifact_schema_errors(dropped_success))

    bad_prior_source = {**artifact, "prior_source": "silent"}
    assert any("prior_source" in error for error in exp4512.artifact_schema_errors(bad_prior_source))

    missing = dict(artifact)
    missing.pop("local_gate_metrics")
    assert any("missing required field" in error for error in exp4512.artifact_schema_errors(missing))

    bad_substrate = {**artifact, "inference_substrate": "live_llm_inference"}
    assert any("substrate" in error for error in exp4512.artifact_schema_errors(bad_substrate))

    bad_principles = {**artifact, "field_principles": {}}
    assert any("field_principles" in error for error in exp4512.artifact_schema_errors(bad_principles))

    bad_baseline = {**artifact, "median_actions_baseline": 1}
    assert any("7760" in error for error in exp4512.artifact_schema_errors(bad_baseline))

    no_control = {**artifact, "positive_control_passed": False}
    assert any("positive_control" in error for error in exp4512.artifact_schema_errors(no_control))

    no_guard = {**artifact, "false_negative_risk_checked": False}
    assert any("false_negative" in error for error in exp4512.artifact_schema_errors(no_guard))

    no_checksum = {**artifact, "reproducibility_checksum": "0" * 64}
    assert any("checksum" in error for error in exp4512.artifact_schema_errors(no_checksum))


def test_req_arc_fcp_4512_defensive_helper_paths(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4512: edge helpers stay deterministic and honest."""

    empty = SimpleNamespace(frame=np.zeros((0, 0), dtype=np.int16), available_actions=[])
    assert exp4512.frame_class_key(empty) == "empty"
    assert exp4512._probability(1, {}, 0.1) == 0.0

    first = _example("aa00", 1, changed=True, value=1)
    second = replace(_example("aa00", 6, changed=True, x=6, y=6, value=1), step_index=2)
    prior = exp4512.build_prior_from_effect_examples(
        [first, second],
        prior_source="human_replay_corpus",
    )
    assert prior.sequence_action_counts == {1: {6: 1.0}}
    assert exp4512._json_action_label(6, {"x": 1, "y": 2}) == (
        '{"action": 6, "data": {"x": 1, "y": 2}}'
    )

    data_dir = tmp_path / exp4512.HUMAN_REPLAY_DATA_RELATIVE_DIR
    data_dir.mkdir(parents=True)
    (data_dir / "manifest.json").write_text(
        json.dumps({"example_count": 0, "shard_count": 0, "shards": []}),
        encoding="utf-8",
    )
    examples, summary = exp4512.load_human_replay_examples(tmp_path)
    assert examples == []
    assert summary["human_manifest_present"] is True
    assert summary["human_examples_loaded"] == 0

    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    (ops_dir / "arc-submission-baseline.json").write_text(
        json.dumps({"solved_count": 1, "median_actions_on_solved": 7760, "games": ["aa00"]}),
        encoding="utf-8",
    )
    assert exp4512.load_gate_baseline(tmp_path)["solved_count"] == 1

    assert exp4512.false_negative_risk_guard({"actions_reduced": False}, {}) == (
        "positive_control_failed_null_uninterpretable"
    )
    assert exp4512._gate_value({}, "missing", "field") is None
    assert exp4512.false_negative_risk_guard(
        {"actions_reduced": True},
        {
            "baseline": {"solved_count": 4, "median_actions_on_solved": 7760},
            "with_prior": {"solved_count": 4, "median_actions_on_solved": 8000},
        },
    ) == "positive_control_passed_null_interpretable"

    blocked = exp4512.build_artifact(
        preconditions_checked={"offline_arcade_import": False},
        prior_summary={"prior_examples_used": 0},
        gate_metrics={
            "baseline": {"solved_count": 4, "median_actions_on_solved": 7760},
            "with_prior": {"solved_count": 0, "median_actions_on_solved": None},
        },
        positive_control={"actions_reduced": True},
        prior_source="self_supervised_marginal_fallback",
        random_seed=4512,
        reproducibility_checksum="sha256:" + "1" * 64,
        duration_s=0.0,
    )
    assert blocked["honest_verdict"] == "complete: blocked_offline_arcade_import_failed"

    solve_drop = exp4512.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        prior_summary={"prior_examples_used": 1},
        gate_metrics={
            "baseline": {"solved_count": 4, "median_actions_on_solved": 7760},
            "with_prior": {"solved_count": 3, "median_actions_on_solved": 100},
        },
        positive_control={"actions_reduced": True},
        prior_source="human_replay_corpus",
        random_seed=4512,
        reproducibility_checksum="sha256:" + "2" * 64,
        duration_s=0.0,
    )
    assert solve_drop["honest_verdict"] == "complete: imitation_prior_solve_rate_guard_failed"

    null = exp4512.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        prior_summary={"prior_examples_used": 1},
        gate_metrics={
            "baseline": {"solved_count": 4, "median_actions_on_solved": 7760},
            "with_prior": {"solved_count": 4, "median_actions_on_solved": 8000},
        },
        positive_control={"actions_reduced": True},
        prior_source="human_replay_corpus",
        random_seed=4512,
        reproducibility_checksum="sha256:" + "3" * 64,
        duration_s=0.0,
    )
    assert null["honest_verdict"] == "complete: imitation_prior_no_reduction_honest_null"


def test_scenario_arc_fcp_4512_run_blocks_when_offline_arcade_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-4512: missing arcade exits with a blocked terminal artifact."""

    monkeypatch.setattr(
        exp4512,
        "check_preconditions",
        lambda _root: {"offline_arcade_import": False},
    )

    def prior_builder(**_kwargs):
        raise AssertionError("prior builder should not run without offline arcade")

    artifact = exp4512.run(
        root=tmp_path,
        write=False,
        prior_builder=prior_builder,
        random_seed=4512,
        now=lambda: 5.0,
    )

    assert artifact["honest_verdict"] == "complete: blocked_offline_arcade_import_failed"


def test_scenario_arc_fcp_4512_run_raises_on_invalid_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-4512: schema errors prevent writing invalid results."""

    monkeypatch.setattr(
        exp4512,
        "positive_control",
        lambda: {"actions_reduced": False},
    )
    bundle = exp4512.PriorBundle(
        prior=exp4512.ActionTypeSequencePrior(
            prior_source="human_replay_corpus",
            frame_class_action_counts={},
            marginal_action_counts={},
        ),
        prior_source="human_replay_corpus",
        examples=[],
        summary={"prior_examples_used": 0},
    )

    def prior_builder(**_kwargs):
        return bundle

    def measure_gate(**_kwargs):
        return {
            "baseline": {"solved_count": 4, "median_actions_on_solved": 7760},
            "with_prior": {"solved_count": 4, "median_actions_on_solved": 8000},
        }

    with pytest.raises(ValueError, match="positive_control"):
        exp4512.run(
            root=tmp_path,
            write=False,
            prior_builder=prior_builder,
            measure_gate=measure_gate,
            random_seed=4512,
        )
