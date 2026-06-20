"""Tests for Exp 4490 human replay frame-change predictor.

Spec refs: REQ-ARC-FCP-4490, REQ-ARC-FCP-4491, REQ-ARC-FCP-4492,
SCENARIO-ARC-FCP-4490, SCENARIO-ARC-FCP-4491.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carnot import experiment_4490_human_replay_frame_change_predictor as exp4490
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic import arc_graph_explore
from carnot.agentic.arc_agi3_live_adapter import ArcAction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(grid: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(frame=grid, available_actions=[1, 6])


def test_req_arc_fcp_4490_spec_declares_frame_only_artifact_contract() -> None:
    """REQ-ARC-FCP-4490: OpenSpec names the frame-only predictor and result contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in (
        "REQ-ARC-FCP-4490",
        "REQ-ARC-FCP-4491",
        "REQ-ARC-FCP-4492",
        "SCENARIO-ARC-FCP-4490",
        "SCENARIO-ARC-FCP-4491",
    ):
        assert ref in spec
    assert "raw rendered replay frames" in spec
    assert "env._game" in spec
    assert exp4490.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4490.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4490_frame_tensor_and_cnn_heads_are_live_legal() -> None:
    """REQ-ARC-FCP-4490: model inputs come from frame pixels and expose both heads."""

    grid = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int16)
    tensor = fcp.frame_to_tensor(_frame(grid), num_colors=4, size=8)

    assert tensor.shape == (4, 8, 8)
    assert torch.allclose(tensor.sum(dim=0), torch.ones((8, 8)))

    model = fcp.SmallFrameChangeCNN(num_colors=4, hidden_channels=4)
    click_heatmap, directional_change = model(tensor.unsqueeze(0))

    assert click_heatmap.shape == (1, 1, 8, 8)
    assert directional_change.shape == (1, 5)


def test_scenario_arc_fcp_4490_behavior_prior_ranks_changing_click_first() -> None:
    """SCENARIO-ARC-FCP-4490: state-conditioned click prior beats no-op order."""

    frame = _frame(np.zeros((8, 8), dtype=np.int16))
    frame_key = fcp.frame_state_key(frame)
    candidates = [
        ArcAction(6, {"x": 1, "y": 1}, "legacy_first_noop"),
        ArcAction(6, {"x": 6, "y": 6}, "changing_click"),
        ArcAction(1, None, "keyboard"),
    ]
    prior = fcp.BehaviorActionPrior(
        marginal_action_counts={1: 1, 6: 4},
        state_click_counts={frame_key: {(6, 6): 30}},
    )

    ranked = fcp.rank_arc_actions(frame, candidates, prior=prior)

    assert ranked[0] is candidates[1]
    assert [candidate.source for candidate in ranked[1:]] == ["legacy_first_noop", "keyboard"]


def test_req_arc_fcp_4491_prior_examples_scorers_and_tie_fallbacks() -> None:
    """REQ-ARC-FCP-4491: prior/scorer helpers rank actions and preserve fallbacks."""

    frame = _frame(np.zeros((8, 8), dtype=np.int16))
    frame_key = fcp.frame_state_key(frame)
    candidates = [
        ArcAction(1, None, "keyboard_one"),
        ArcAction(6, {"x": 7, "y": 7}, "click"),
        ArcAction(7, None, "unused"),
    ]
    prior = fcp.BehaviorActionPrior.from_examples(
        [
            {"state_key": frame_key, "action_id": 1},
            {"state_key": frame_key, "action_id": 6, "x": 7, "y": 7},
            {"state_key": frame_key, "action_id": 6, "x": 7, "y": 7},
        ]
    )

    assert prior.score(frame, candidates[1]) > prior.score(frame, candidates[0])
    assert fcp.rank_arc_actions(frame, candidates) == candidates
    assert (
        fcp.rank_arc_actions(frame, candidates, scorer=lambda _frame, cand: cand.action_id)[0]
        is candidates[2]
    )
    with pytest.raises(TypeError, match="candidate_score"):
        fcp.rank_arc_actions(frame, candidates, scorer=object())
    assert fcp.efficiency_score(0, 4) == 0.0
    assert fcp.efficiency_score(1, 0) == 0.0


def test_req_arc_fcp_4490_frame_change_scorer_maps_clicks_and_directions() -> None:
    """REQ-ARC-FCP-4490: scorer reads click heatmaps and ACTION1-5 heads."""

    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = tensor.shape[0]
            heatmap = torch.zeros((batch, 1, 8, 8), dtype=torch.float32)
            heatmap[:, :, 7, 7] = 0.9
            directional = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32).repeat(
                batch, 1
            )
            return heatmap, directional

    frame = _frame(np.zeros((8, 8), dtype=np.int16))
    scorer = fcp.FrameChangeScorer(DummyModel(), num_colors=4, size=8)

    assert scorer.candidate_score(frame, ArcAction(6, {"x": 7, "y": 7}, "click")) == pytest.approx(
        0.9
    )
    assert scorer.candidate_score(frame, ArcAction(5, None, "direction")) == pytest.approx(0.5)
    assert scorer.candidate_score(frame, ArcAction(7, None, "other")) == 0.0
    ranked = fcp.rank_arc_actions(
        frame,
        [
            ArcAction(5, None, "direction"),
            ArcAction(6, {"x": 7, "y": 7}, "click"),
        ],
        scorer=scorer,
    )
    assert ranked[0].source == "click"


def test_req_arc_fcp_4491_rich_action_candidates_accepts_optional_ranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-FCP-4491: rich_action_candidates wires optional ranking with legacy fallback."""

    grid = np.zeros((8, 8), dtype=np.int16)
    grid[1, 1] = 2
    grid[6, 6] = 3
    frame = SimpleNamespace(frame=grid, available_actions=[6])
    frame_key = fcp.frame_state_key(frame)

    monkeypatch.setattr(
        arc_graph_explore,
        "_components_detailed",
        lambda _grid: [(1, 1, 1, 2), (6, 6, 1, 3)],
    )

    legacy = arc_graph_explore.rich_action_candidates(frame, by_salience=False)
    prior = fcp.BehaviorActionPrior(state_click_counts={frame_key: {(6, 6): 12}})
    ranked = arc_graph_explore.rich_action_candidates(
        frame,
        by_salience=False,
        action_prior=prior,
    )

    assert [candidate.data for candidate in legacy] == [{"x": 1, "y": 1}, {"x": 6, "y": 6}]
    assert [candidate.data for candidate in ranked] == [{"x": 6, "y": 6}, {"x": 1, "y": 1}]


def test_scenario_arc_fcp_4490_positive_control_detects_efficiency_win() -> None:
    """SCENARIO-ARC-FCP-4490: positive control reports fewer actions after ranking."""

    result = fcp.evaluate_positive_control()

    assert result["baseline_actions_to_first_levelup"] == 3
    assert result["ranked_actions_to_first_levelup"] == 1
    assert result["actions_reduced"] is True
    assert result["implied_efficiency_delta"] > 0.0


def test_scenario_arc_fcp_4491_missing_corpus_artifact_is_terminal_and_schema_valid(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-4491: absent corpus writes an honest non-fabricated artifact."""

    preconditions = {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": True,
        "torch_import": True,
        "torch_version": torch.__version__,
        "human_replay_corpus_present": False,
        "human_replay_corpus_paths": [],
        "weights_input_present": False,
        "official_license_verified_for_bundled_weights": False,
        "leaderboard_submission": False,
        "env_game_access_blocked": True,
        "ok": False,
    }

    artifact = exp4490.run(
        root=tmp_path,
        preconditions_checked=preconditions,
        write=True,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "complete: blocked_human_replay_corpus_not_cached"
    assert artifact["inference_substrate"] == exp4490.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["torch_import"] is True
    assert artifact["weights_bundled"] is False
    assert artifact["trained_on_human_corpus"] is False
    assert artifact["heldout_median_actions_before"] is None
    assert artifact["heldout_median_actions_after"] is None
    assert artifact["solve_rate_dropped"] is False
    assert exp4490.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4490.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]


def test_req_arc_fcp_4492_preconditions_and_corpus_discovery(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4492: preconditions record checked resources and corpus paths."""

    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    preconditions = exp4490.check_preconditions(tmp_path)

    assert preconditions["agents_md_read"] is True
    assert preconditions["codex_md_read"] is True
    assert preconditions["offline_arcade_import"] is True
    assert preconditions["torch_import"] is True
    assert preconditions["human_replay_corpus_present"] is False
    assert preconditions["ok"] is False

    replay_dir = tmp_path / "environment_files" / "aa00" / "replays"
    replay_dir.mkdir(parents=True)
    (replay_dir / "aa00-fixture.json").write_text("{}", encoding="utf-8")
    (tmp_path / "action_effect_dict.npz").write_bytes(b"fixture")

    discovered = exp4490.discover_human_replay_corpus(tmp_path)
    assert discovered["human_replay_corpus_present"] is True
    assert discovered["raw_replay_json_count"] == 1
    assert discovered["action_effect_npz_paths"] == ["action_effect_dict.npz"]


def test_req_arc_fcp_4492_artifact_schema_rejects_fabrication(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4492: schema catches malformed or fabricated artifact fields."""

    preconditions = {
        "offline_arcade_import": True,
        "torch_import": True,
        "human_replay_corpus_present": False,
        "ok": False,
    }
    valid = exp4490.run(
        root=tmp_path,
        preconditions_checked=preconditions,
        write=False,
        now=lambda: 1.0,
    )

    mutations = [
        (lambda item: item.pop("solve_rate_dropped"), "missing required"),
        (lambda item: item.__setitem__("honest_verdict", "blocked"), "terminal prefix"),
        (
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
            "inference_substrate",
        ),
        (lambda item: item.__setitem__("preconditions_checked", []), "preconditions_checked"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (
            lambda item: item.update(
                weights_bundled=True,
                official_license_verified_for_bundled_weights=False,
            ),
            "bundled weights",
        ),
        (
            lambda item: item.__setitem__("positive_control", {"actions_reduced": False}),
            "positive_control",
        ),
        (
            lambda item: item.update(
                heldout_median_actions_before=3,
                heldout_median_actions_after=2,
                solve_rate_dropped=True,
                trained_on_human_corpus=True,
            ),
            "solve rate",
        ),
        (
            lambda item: item.update(
                heldout_median_actions_before=3,
                heldout_median_actions_after=2,
                implied_efficiency_delta=0.1,
            ),
            "missing-corpus",
        ),
    ]
    for mutate, expected in mutations:
        artifact = dict(valid)
        mutate(artifact)
        assert any(expected in error for error in exp4490.artifact_schema_errors(artifact))

    offline_blocked = exp4490.run(
        root=tmp_path,
        preconditions_checked={**preconditions, "offline_arcade_import": False},
        write=False,
        now=lambda: 2.0,
    )
    torch_blocked = exp4490.run(
        root=tmp_path,
        preconditions_checked={**preconditions, "torch_import": False},
        write=False,
        now=lambda: 3.0,
    )
    trained_null = exp4490.run(
        root=tmp_path,
        preconditions_checked={**preconditions, "human_replay_corpus_present": True, "ok": True},
        write=False,
        now=lambda: 4.0,
    )

    assert offline_blocked["honest_verdict"] == "complete: blocked_offline_arcade_import_failed"
    assert torch_blocked["honest_verdict"] == "complete: blocked_torch_missing"
    assert (
        trained_null["honest_verdict"]
        == "complete: human_replay_frame_change_predictor_ready_honest_null"
    )
    assert trained_null["trained_on_human_corpus"] is True


def test_req_arc_fcp_4492_run_raises_when_schema_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-FCP-4492: run refuses to write an invalid artifact."""

    monkeypatch.setattr(
        exp4490,
        "evaluate_positive_control",
        lambda: {"actions_reduced": False},
    )
    with pytest.raises(ValueError, match="positive_control"):
        exp4490.run(
            root=tmp_path,
            preconditions_checked={
                "offline_arcade_import": True,
                "torch_import": True,
                "human_replay_corpus_present": False,
                "ok": False,
            },
            write=False,
            now=lambda: 1.0,
        )
