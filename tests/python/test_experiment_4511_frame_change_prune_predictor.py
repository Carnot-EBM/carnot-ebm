"""Tests for Exp 4511 self-supervised frame-change pruning.

Spec refs: REQ-ARC-FCP-4511, SCENARIO-ARC-FCP-4511.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from carnot import experiment_4511_frame_change_prune_predictor as exp4511
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic import arc_graph_explore
from carnot.agentic.arc_agi3_live_adapter import ArcAction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(value: int = 0) -> SimpleNamespace:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[1, 1] = int(value)
    grid[6, 6] = int(value) + 1
    return SimpleNamespace(frame=grid, available_actions=[1, 6])


def _example(
    game_id: str,
    action_id: int,
    changed: bool,
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
    )


class ScoreMap:
    def __init__(self, scores: dict[tuple[int, tuple[tuple[str, int], ...]], float]) -> None:
        self.scores = scores

    def candidate_score(self, _frame: object, candidate: ArcAction) -> float:
        data = tuple(sorted((candidate.data or {}).items()))
        return float(self.scores.get((int(candidate.action_id), data), 0.0))


def test_req_arc_fcp_4511_spec_declares_pruning_artifact_contract() -> None:
    """REQ-ARC-FCP-4511: OpenSpec anchors the pruning experiment and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4511" in spec
    assert "SCENARIO-ARC-FCP-4511" in spec
    assert exp4511.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4511.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4511_prunes_predicted_noops_before_ranking() -> None:
    """SCENARIO-ARC-FCP-4511: no-op candidates below threshold are removed."""

    frame = _frame()
    candidates = [
        ArcAction(6, {"x": 1, "y": 1}, "noop_click"),
        ArcAction(6, {"x": 6, "y": 6}, "changing_click"),
        ArcAction(1, None, "keyboard_noop"),
    ]
    scorer = ScoreMap(
        {
            (6, (("x", 1), ("y", 1))): 0.05,
            (6, (("x", 6), ("y", 6))): 0.91,
            (1, ()): 0.02,
        }
    )

    kept, diagnostics = fcp.prune_arc_actions(
        frame,
        candidates,
        scorer=scorer,
        threshold=0.50,
    )

    assert kept == [candidates[1]]
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["kept_count"] == 1
    assert diagnostics["pruned_count"] == 2
    assert diagnostics["forced_keep_count"] == 0


def test_scenario_arc_fcp_4511_prune_guard_keeps_best_candidate() -> None:
    """SCENARIO-ARC-FCP-4511: pruning never returns an empty candidate list."""

    frame = _frame()
    candidates = [
        ArcAction(1, None, "weak_keyboard"),
        ArcAction(6, {"x": 2, "y": 2}, "less_weak_click"),
    ]
    scorer = ScoreMap({(1, ()): 0.01, (6, (("x", 2), ("y", 2))): 0.09})

    kept, diagnostics = fcp.prune_arc_actions(
        frame,
        candidates,
        scorer=scorer,
        threshold=0.95,
    )

    assert kept == [candidates[1]]
    assert diagnostics["forced_keep_count"] == 1
    assert diagnostics["kept_count"] == 1


def test_req_arc_fcp_4511_rich_action_candidates_accepts_prune_threshold(
    monkeypatch,
) -> None:
    """REQ-ARC-FCP-4511: rich_action_candidates wires the pruning threshold."""

    frame = _frame()
    monkeypatch.setattr(
        arc_graph_explore,
        "_components_detailed",
        lambda _grid: [(1, 1, 1, 1), (6, 6, 1, 2)],
    )
    scorer = ScoreMap(
        {
            (1, ()): 0.01,
            (6, (("x", 1), ("y", 1))): 0.10,
            (6, (("x", 6), ("y", 6))): 0.88,
        }
    )

    pruned = arc_graph_explore.rich_action_candidates(
        frame,
        by_salience=False,
        frame_change_scorer=scorer,
        frame_change_prune_threshold=0.50,
    )

    assert [(candidate.action_id, candidate.data) for candidate in pruned] == [
        (6, {"x": 6, "y": 6})
    ]


def test_req_arc_fcp_4511_heldout_noop_metrics() -> None:
    """REQ-ARC-FCP-4511: held-out no-op precision and recall are cross-game metrics."""

    examples = [
        _example("aa00", 6, False, x=1, y=1, value=1),
        _example("aa00", 6, True, x=6, y=6, value=1),
        _example("bb00", 1, False, value=2),
        _example("bb00", 2, True, value=2),
    ]
    scorer = ScoreMap(
        {
            (6, (("x", 1), ("y", 1))): 0.05,
            (6, (("x", 6), ("y", 6))): 0.95,
            (1, ()): 0.10,
            (2, ()): 0.90,
        }
    )

    metrics = exp4511.heldout_noop_metrics(examples, scorer=scorer, threshold=0.50)

    assert metrics["heldout_transition_count"] == 4
    assert metrics["heldout_noop_precision"] == 1.0
    assert metrics["heldout_noop_recall"] == 1.0


def test_scenario_arc_fcp_4511_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4511: run writes required result JSON without over-claiming."""

    examples = [
        _example("aa00", 6, False, x=1, y=1, value=1),
        _example("aa00", 6, True, x=6, y=6, value=1),
        _example("bb00", 1, False, value=2),
        _example("bb00", 2, True, value=2),
    ]

    def collect_fixture(**_kwargs):
        return examples, {
            "game_count": 2,
            "transition_count": len(examples),
            "changed_count": 2,
            "noop_count": 2,
        }

    def gate_fixture(**_kwargs):
        return {
            "baseline": {
                "solved_count": 2,
                "median_actions_on_solved": exp4511.BASELINE_MEDIAN_ACTIONS,
                "per_game": [{"game": "aa00", "solved": True, "actions": 8000}],
            },
            "with_prune": {
                "solved_count": 2,
                "median_actions_on_solved": 100.0,
                "per_game": [{"game": "aa00", "solved": True, "actions": 100}],
            },
            "measurement_script": exp4511.LOCAL_GATE_RELATIVE_PATH,
        }

    artifact = exp4511.run(
        root=tmp_path,
        write=True,
        collect_corpus=collect_fixture,
        measure_gate=gate_fixture,
        train_epochs=1,
        batch_size=2,
        hidden_channels=4,
        frame_size=8,
        num_colors=8,
        random_seed=4511,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "success: frame_change_prune_median_actions_100_below_7760"
    assert artifact["median_actions_baseline"] == exp4511.BASELINE_MEDIAN_ACTIONS
    assert artifact["median_actions_with_prune"] == 100.0
    assert artifact["solve_rate_baseline"] == 2
    assert artifact["solve_rate_with_prune"] == 2
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["random_seed"] == 4511
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert exp4511.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4511.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]


def test_req_arc_fcp_4511_artifact_schema_rejects_bad_fields() -> None:
    """REQ-ARC-FCP-4511: schema rejects non-terminal verdicts and unsafe success claims."""

    artifact = exp4511.build_artifact(
        preconditions_checked={
            "offline_arcade_import": True,
            "torch_import": True,
            "torch_version": torch.__version__,
        },
        corpus_summary={"game_count": 2, "transition_count": 4},
        training_summary={"batches_trained": 1},
        heldout_metrics={
            "heldout_transition_count": 4,
            "heldout_noop_precision": 1.0,
            "heldout_noop_recall": 1.0,
        },
        gate_metrics={
            "baseline": {"solved_count": 2, "median_actions_on_solved": 7760},
            "with_prune": {"solved_count": 2, "median_actions_on_solved": 100},
            "measurement_script": exp4511.LOCAL_GATE_RELATIVE_PATH,
        },
        positive_control={"actions_reduced": True},
        prune_threshold=0.5,
        random_seed=4511,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=1.0,
    )
    assert exp4511.artifact_schema_errors(artifact) == []

    bad_verdict = {**artifact, "honest_verdict": "blocked"}
    assert any("terminal prefix" in error for error in exp4511.artifact_schema_errors(bad_verdict))

    dropped_success = {
        **artifact,
        "honest_verdict": "success: frame_change_prune_median_actions_100_below_7760",
        "solve_rate_with_prune": 1,
    }
    assert any("solve-rate" in error for error in exp4511.artifact_schema_errors(dropped_success))

    no_checksum = {**artifact, "reproducibility_checksum": "0" * 64}
    assert any("checksum" in error for error in exp4511.artifact_schema_errors(no_checksum))
