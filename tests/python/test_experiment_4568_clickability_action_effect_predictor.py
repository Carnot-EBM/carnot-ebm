"""Tests for Exp 4568 pooled clickability action-effect predictor.

Spec refs: REQ-ARC-FCP-4568, SCENARIO-ARC-FCP-4568.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carnot import experiment_4568_clickability_action_effect_predictor as exp4568
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic import arc_graph_explore
from carnot.agentic.arc_agi3_live_adapter import ArcAction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(value: int = 0, *, available: tuple[int, ...] = (1, 2, 6)) -> SimpleNamespace:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[1, 1] = int(value)
    grid[6, 6] = int(value) + 1
    return SimpleNamespace(frame=grid, available_actions=list(available))


def _example(
    env: str,
    action_id: int,
    changed: bool,
    *,
    x: int | None = None,
    y: int | None = None,
    frame: SimpleNamespace | None = None,
    source: str = "human_replay",
) -> fcp.FrameActionEffectExample:
    source_frame = frame or _frame()
    return fcp.FrameActionEffectExample(
        frame=source_frame,
        action_id=action_id,
        x=x,
        y=y,
        frame_delta=1.0 if changed else 0.0,
        level_progress=1.0 if changed else 0.0,
        state_key=fcp.frame_state_key(source_frame),
        env=env,
        feature_source=source,
    )


class ScoreMap:
    def __init__(self, scores: dict[tuple[int, tuple[tuple[str, int], ...]], float]) -> None:
        self.scores = scores

    def candidate_score(self, _frame: object, candidate: ArcAction) -> float:
        data = tuple(sorted((candidate.data or {}).items()))
        return float(self.scores.get((int(candidate.action_id), data), 0.0))


def test_req_arc_fcp_4568_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-4568: OpenSpec anchors the pooled clickability artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4568" in spec
    assert "SCENARIO-ARC-FCP-4568" in spec
    assert exp4568.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4568.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4568_loads_human_and_transition_corpora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ) -> None:
    """REQ-ARC-FCP-4568: pooled corpus uses both local frame-action sources."""

    human_dir = tmp_path / exp4568.HUMAN_REPLAY_RELATIVE_DIR
    human_dir.mkdir(parents=True)
    (human_dir / "manifest.json").write_text(
        json.dumps({"example_count": 3, "shard_count": 1}),
        encoding="utf-8",
    )
    transition_dir = tmp_path / exp4568.TRANSITION_CORPUS_RELATIVE_DIR
    transition_dir.mkdir(parents=True)
    grid = np.zeros((2, 8, 8), dtype=np.int16)
    next_grid = grid.copy()
    next_grid[0, 3, 2] = 1
    np.savez_compressed(
        transition_dir / "toy.npz",
        grids=grid,
        next_grids=next_grid,
        actions=np.asarray([6, 1], dtype=np.int16),
        xs=np.asarray([2, -1], dtype=np.int16),
        ys=np.asarray([3, -1], dtype=np.int16),
        lb=np.asarray([0, 0], dtype=np.int16),
        la=np.asarray([0, 1], dtype=np.int16),
    )
    human_example = _example("human_game", 6, True, x=6, y=6)
    monkeypatch.setattr(
        exp4568,
        "load_frame_action_effect_examples",
        lambda _path, limit=None: [human_example],
    )

    pooled = exp4568.load_pooled_examples(tmp_path)

    assert len(pooled.examples) == 3
    assert pooled.source_counts == {"human_replay": 1, "transition_corpus": 2}
    assert pooled.metadata["human_replay_manifest_examples"] == 3
    assert pooled.metadata["transition_npz_count"] == 1
    assert pooled.games == ["human_game", "toy"]
    transition_rows = [row for row in pooled.examples if row.feature_source == "arc_transition_corpus"]
    assert transition_rows[0].env == "toy"
    assert transition_rows[0].changed is True
    assert transition_rows[0].x == 2
    assert transition_rows[0].y == 3
    assert transition_rows[1].level_progress == 1.0


def test_scenario_arc_fcp_4568_measurement_reports_positive_actions_delta() -> None:
    """SCENARIO-ARC-FCP-4568: predictor ordering reports baseline-minus-treatment delta."""

    frame_a = _frame(1)
    frame_b = _frame(2)
    examples = [
        _example("aa00", 6, False, x=1, y=1, frame=frame_a),
        _example("aa00", 6, True, x=6, y=6, frame=frame_a),
        _example("bb00", 1, False, frame=frame_b),
        _example("bb00", 2, True, frame=frame_b),
    ]
    scorer = ScoreMap(
        {
            (6, (("x", 1), ("y", 1))): 0.05,
            (6, (("x", 6), ("y", 6))): 0.95,
            (1, ()): 0.10,
            (2, ()): 0.90,
        }
    )

    metrics = exp4568.measure_actions_to_first_levelup(
        examples,
        scorer=scorer,
        n_bootstrap=0,
    )

    assert metrics["median_actions_to_first_levelup_baseline"] == 2.0
    assert metrics["median_actions_to_first_levelup_with_predictor"] == 1.0
    assert metrics["actions_delta"] == 1.0
    assert metrics["actions_delta_ci"] == [1.0, 1.0]
    assert metrics["solve_rate_preserved"] is True
    assert metrics["action_reduction"] is True
    assert exp4568.bootstrap_actions_delta_ci([], n_bootstrap=0) == [0.0, 0.0]


def test_req_arc_fcp_4568_rich_action_candidates_accepts_predictor_ranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-FCP-4568: rich_action_candidates orders by frame-change scorer."""

    frame = _frame()
    monkeypatch.setattr(
        arc_graph_explore,
        "_components_detailed",
        lambda _grid: [(1, 1, 1, 1), (6, 6, 1, 2)],
    )
    scorer = ScoreMap(
        {
            (1, ()): 0.10,
            (2, ()): 0.10,
            (6, (("x", 1), ("y", 1))): 0.20,
            (6, (("x", 6), ("y", 6))): 0.90,
        }
    )

    ranked = arc_graph_explore.rich_action_candidates(
        frame,
        by_salience=False,
        frame_change_scorer=scorer,
    )

    assert [(candidate.action_id, candidate.data) for candidate in ranked] == [
        (6, {"x": 6, "y": 6}),
        (6, {"x": 1, "y": 1}),
        (1, None),
        (2, None),
    ]


def test_scenario_arc_fcp_4568_artifact_schema_success_and_null() -> None:
    """SCENARIO-ARC-FCP-4568: artifact gates success and annotates no-gain nulls."""

    success = exp4568.build_artifact(
        preconditions_checked=exp4568.ok_preconditions_for_tests(),
        corpus_summary={"corpus_examples_loaded": 4, "source_counts": {"human_replay": 2}},
        training_summary={"examples_seen": 4, "batches_trained": 1},
        ranking_metrics={
            "median_actions_to_first_levelup_baseline": 2.0,
            "median_actions_to_first_levelup_with_predictor": 1.0,
            "actions_delta": 1.0,
            "actions_delta_ci": [1.0, 1.0],
            "solve_rate_baseline": 1.0,
            "solve_rate_with_predictor": 1.0,
            "solve_rate_preserved": True,
            "action_reduction": True,
            "paired_delta_count": 2,
        },
        generic_transfer={"generic_transfer_rate_with_predictor": 0.04},
        positive_control={"actions_reduced": True},
        random_seed=4568,
        reproducibility_checksum="sha256:" + "1" * 64,
        duration_s=1.0,
    )

    assert (
        success["honest_verdict"]
        == "success: clickability_predictor_actions_to_levelup_1_below_blind"
    )
    assert success["verifier_is_oracle"] is False
    assert success["actions_delta"] == 1.0
    assert success["chosen_submitted_config"] == "enable_clickability_predictor_ranker"
    assert success["null_delta_methodology_note"] is None
    assert exp4568.artifact_schema_errors(success) == []

    null = exp4568.build_artifact(
        preconditions_checked=exp4568.ok_preconditions_for_tests(),
        corpus_summary={"corpus_examples_loaded": 4},
        training_summary={"examples_seen": 4},
        ranking_metrics={
            "median_actions_to_first_levelup_baseline": 2.0,
            "median_actions_to_first_levelup_with_predictor": 2.0,
            "actions_delta": 0.0,
            "actions_delta_ci": [0.0, 0.0],
            "solve_rate_baseline": 1.0,
            "solve_rate_with_predictor": 1.0,
            "solve_rate_preserved": True,
            "action_reduction": False,
        },
        generic_transfer={"generic_transfer_rate_with_predictor": 0.04},
        positive_control={"actions_reduced": True},
        random_seed=4568,
        reproducibility_checksum="sha256:" + "2" * 64,
        duration_s=1.0,
    )

    assert (
        null["honest_verdict"]
        == "complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened"
    )
    assert null["false_negative_risk_checked"] is True
    assert null["chosen_submitted_config"] == "unchanged"
    assert null["null_delta_methodology_note"]
    assert null["missing_verifier_gaps"]
    assert exp4568.artifact_schema_errors(null) == []

    bad = {**success, "verifier_is_oracle": True}
    assert any("verifier_is_oracle" in error for error in exp4568.artifact_schema_errors(bad))
    fake_success = {**success, "actions_delta": 0.0}
    assert any("positive actions_delta" in error for error in exp4568.artifact_schema_errors(fake_success))


def test_scenario_arc_fcp_4568_run_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4568: run trains, measures, validates, and writes JSON."""

    examples = [
        _example("aa00", 6, False, x=1, y=1),
        _example("aa00", 6, True, x=6, y=6),
        _example("bb00", 1, False),
        _example("bb00", 2, True),
    ]
    pooled = exp4568.PooledCorpus(
        examples=examples,
        source_counts={"human_replay": 2, "transition_corpus": 2},
    )

    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            heatmap = torch.zeros((tensor.shape[0], 1, 8, 8), dtype=torch.float32)
            directional = torch.full((tensor.shape[0], 5), 0.5, dtype=torch.float32)
            return heatmap, directional

    scorer = ScoreMap(
        {
            (6, (("x", 1), ("y", 1))): 0.10,
            (6, (("x", 6), ("y", 6))): 0.90,
            (1, ()): 0.20,
            (2, ()): 0.80,
        }
    )

    artifact = exp4568.run(
        root=tmp_path,
        write=True,
        preconditions_checked=exp4568.ok_preconditions_for_tests(),
        load_examples=lambda _root, human_limit=None, transition_limit=None: pooled,
        train_model=lambda rows, **_kwargs: (DummyModel(), {"examples_seen": len(rows)}),
        scorer_factory=lambda _model, **_kwargs: scorer,
        generic_transfer_loader=lambda _root: {"generic_transfer_rate_with_predictor": 0.04},
        random_seed=4568,
        frame_size=8,
        num_colors=8,
        n_bootstrap=0,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["corpus_summary"]["source_counts"] == {
        "human_replay": 2,
        "transition_corpus": 2,
    }
    assert artifact["positive_control_passed"] is True
    assert artifact["generic_transfer_rate_with_predictor"] == 0.04
    assert exp4568.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4568.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
