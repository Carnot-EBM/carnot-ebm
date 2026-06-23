"""Tests for Exp 4629 live action-effect predictor graduation.

Spec refs: REQ-ARC-FCP-4629, SCENARIO-ARC-FCP-4629.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_4629_graduate_action_effect_predictor_live as exp4629
from carnot.agentic import arc_competition_agent as comp
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic import arc_graph_explore
from carnot.agentic.arc_agi3_live_adapter import ArcAction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(*, available: tuple[int, ...] = (1, 2, 6)) -> SimpleNamespace:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[1, 1] = 1
    grid[6, 6] = 2
    return SimpleNamespace(frame=grid, available_actions=list(available))


class ScoreMap:
    def __init__(self, scores: dict[tuple[int, tuple[tuple[str, int], ...]], float]) -> None:
        self.scores = scores

    def candidate_score(self, _frame: object, candidate: object) -> float:
        action_id = int(getattr(candidate, "action_id", getattr(candidate, "action", 0)) or 0)
        data = getattr(candidate, "data", None) or {}
        if isinstance(candidate, dict):
            action_id = int(candidate.get("action_id", candidate.get("action", 0)) or 0)
            data = candidate.get("data") or {}
        return float(self.scores.get((action_id, tuple(sorted(data.items()))), 0.0))


def test_req_arc_fcp_4629_spec_declares_live_artifact_fields() -> None:
    """REQ-ARC-FCP-4629: OpenSpec anchors the live graduation artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4629" in spec
    assert "SCENARIO-ARC-FCP-4629" in spec
    assert exp4629.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4629.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4629_live_scorer_loads_cached_action_effect_memory(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4629: the live scorer is assembled from local transition effects."""

    corpus = tmp_path / exp4629.TRANSITION_CORPUS_RELATIVE_DIR
    corpus.mkdir(parents=True)
    grid = np.zeros((3, 64, 64), dtype=np.int16)
    next_grid = grid.copy()
    next_grid[0, 48, 48] = 9
    np.savez_compressed(
        corpus / "toy.npz",
        grids=grid,
        next_grids=next_grid,
        actions=np.asarray([6, 6, 1], dtype=np.int16),
        xs=np.asarray([48, 1, -1], dtype=np.int16),
        ys=np.asarray([48, 1, -1], dtype=np.int16),
        lb=np.asarray([0, 0, 0], dtype=np.int16),
        la=np.asarray([1, 0, 0], dtype=np.int16),
    )

    scorer = fcp.load_live_action_effect_scorer(root=tmp_path)

    assert scorer is not None
    assert scorer.source == "persistent_aem_plus_optional_cnn"
    good = ArcAction(6, {"x": 48, "y": 48}, "good")
    bad = ArcAction(6, {"x": 1, "y": 1}, "bad")
    assert scorer.candidate_score(_frame(), good) > scorer.candidate_score(_frame(), bad)


def test_scenario_arc_fcp_4629_action_effect_ranker_is_final_after_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-FCP-4629: router order is only the tie-break for effect scores."""

    frame = _frame()
    monkeypatch.setattr(
        arc_graph_explore,
        "_components_detailed",
        lambda _grid: [(1, 1, 1, 1), (6, 6, 1, 2)],
    )

    class PreferNoopRouter:
        def rank(self, _frame: object, candidates: list[ArcAction], **_: object) -> list[ArcAction]:
            return list(reversed(candidates))

    scorer = ScoreMap(
        {
            (1, ()): 0.05,
            (2, ()): 0.05,
            (6, (("x", 1), ("y", 1))): 0.10,
            (6, (("x", 6), ("y", 6))): 0.95,
        }
    )

    ranked = arc_graph_explore.rich_action_candidates(
        frame,
        by_salience=False,
        frame_change_scorer=scorer,
        candidate_router=PreferNoopRouter(),
    )

    assert (ranked[0].action_id, ranked[0].data) == (6, {"x": 6, "y": 6})


def test_scenario_arc_fcp_4629_submitted_e3_default_loads_live_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-FCP-4629: the scored E3 path receives the graduated scorer."""

    scorer = object()
    monkeypatch.setattr(comp, "_load_submitted_frame_change_scorer", lambda: scorer)

    policy = comp.E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)

    assert comp.SUBMITTED_AGENT_CONFIG["frame_change_predictor_enabled"] is True
    assert comp.SUBMITTED_AGENT_CONFIG["frame_change_ranking_mode"] == (
        "persistent_aem_plus_optional_cnn"
    )
    assert policy.explorer.frame_change_scorer is scorer
    assert policy.explorer.frame_change_prune_threshold is None


def test_scenario_arc_fcp_4629_measurement_and_artifact_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4629: matched bare control gates a live efficiency claim."""

    rows = [
        {
            "game": "aa00",
            "state_key": "s1",
            "action_id": 6,
            "x": 1,
            "y": 1,
            "changed": False,
        },
        {
            "game": "aa00",
            "state_key": "s1",
            "action_id": 6,
            "x": 6,
            "y": 6,
            "changed": True,
            "level_progress": 1.0,
        },
        {"game": "bb00", "state_key": "s2", "action_id": 1, "changed": False},
        {"game": "bb00", "state_key": "s2", "action_id": 2, "changed": True},
    ]
    scorer = ScoreMap(
        {
            (6, (("x", 1), ("y", 1))): 0.05,
            (6, (("x", 6), ("y", 6))): 0.95,
            (1, ()): 0.10,
            (2, ()): 0.90,
        }
    )

    metrics = exp4629.measure_live_action_efficiency(
        rows,
        scorer=scorer,
        n_bootstrap=0,
    )

    assert metrics["median_actions_to_first_levelup_bare"] == 2.0
    assert metrics["median_actions_to_first_levelup_predictor"] == 1.0
    assert metrics["actions_delta"] == 1.0
    assert metrics["actions_delta_ci"] == [1.0, 1.0]
    assert metrics["first_win_rate_delta"] == 1.0
    assert metrics["solve_rate_preserved"] is True

    artifact = exp4629.build_artifact(
        preconditions_checked=exp4629.ok_preconditions_for_tests(),
        training_summary={"cnn_batches_trained": 1, "memory_row_count": 4},
        live_measurement=metrics,
        live_path_reachable=True,
        orphan_lint_green=True,
        parity_test_green=True,
        random_seed=4629,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["chosen_submitted_config"] == (
        "frame_change_predictor_enabled:persistent_aem_plus_optional_cnn"
    )
    assert exp4629.artifact_schema_errors(artifact) == []

    path = exp4629.write_artifact(artifact, root=tmp_path)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
