"""Tests for Exp 4547 cached human-replay frame-change CNN ranker.

Spec refs: REQ-ARC-FCP-4547, SCENARIO-ARC-FCP-4547.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carnot import experiment_4547_frame_change_predictor as exp4547
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic import arc_graph_explore
from carnot.agentic.arc_agi3_live_adapter import ArcAction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(value: int = 0) -> SimpleNamespace:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[1, 1] = int(value)
    grid[6, 6] = int(value) + 1
    return SimpleNamespace(frame=grid, available_actions=[1, 2, 6])


def _example(
    env: str,
    action_id: int,
    changed: bool,
    *,
    x: int | None = None,
    y: int | None = None,
    frame: SimpleNamespace | None = None,
) -> fcp.FrameActionEffectExample:
    source_frame = frame or _frame()
    return fcp.FrameActionEffectExample(
        frame=source_frame,
        action_id=action_id,
        x=x,
        y=y,
        frame_delta=1.0 if changed else 0.0,
        level_progress=0.0,
        state_key=fcp.frame_state_key(source_frame),
        env=env,
    )


class ScoreMap:
    def __init__(self, scores: dict[tuple[int, tuple[tuple[str, int], ...]], float]) -> None:
        self.scores = scores

    def candidate_score(self, _frame: object, candidate: ArcAction) -> float:
        data = tuple(sorted((candidate.data or {}).items()))
        return float(self.scores.get((int(candidate.action_id), data), 0.0))


def test_req_arc_fcp_4547_spec_declares_cnn_ranker_artifact_contract() -> None:
    """REQ-ARC-FCP-4547: OpenSpec anchors the experiment and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4547" in spec
    assert "SCENARIO-ARC-FCP-4547" in spec
    assert exp4547.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4547.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4547_auroc_positive_control_beats_trivial_baseline() -> None:
    """REQ-ARC-FCP-4547: held-out delta AUROC is measured against a 0.5 baseline."""

    examples = [
        _example("aa00", 6, False, x=1, y=1),
        _example("aa00", 6, True, x=6, y=6),
        _example("bb00", 1, False),
        _example("bb00", 2, True),
    ]
    scorer = ScoreMap(
        {
            (6, (("x", 1), ("y", 1))): 0.10,
            (6, (("x", 6), ("y", 6))): 0.90,
            (1, ()): 0.20,
            (2, ()): 0.80,
        }
    )

    metrics = exp4547.heldout_delta_metrics(examples, scorer=scorer)

    assert metrics["cnn_held_out_delta_auroc"] == pytest.approx(1.0)
    assert metrics["trivial_delta_auroc"] == 0.5
    assert metrics["positive_control_passed"] is True
    assert exp4547.binary_auroc([0, 1], [1.0, 0.0]) == pytest.approx(0.0)
    assert exp4547.binary_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert exp4547.binary_auroc([1, 1], [0.8, 0.9]) == pytest.approx(0.5)


def test_scenario_arc_fcp_4547_cnn_ranker_reduces_matched_candidate_order() -> None:
    """SCENARIO-ARC-FCP-4547: CNN ordering beats blind order without solve-rate drop."""

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

    metrics = exp4547.measure_ranked_candidate_groups(examples, scorer=scorer)

    assert metrics["median_actions_to_first_levelup_blind"] == 2.0
    assert metrics["median_actions_to_first_levelup_cnn"] == 1.0
    assert metrics["solve_rate_blind"] == 1.0
    assert metrics["solve_rate_cnn"] == 1.0
    assert metrics["solve_rate_preserved"] is True
    assert metrics["action_reduction"] is True

    no_group = exp4547.measure_ranked_candidate_groups(
        [_example("solo", 1, False), _example("solo", 2, False)],
        scorer=scorer,
        min_candidates=3,
    )
    assert no_group["heldout_group_count"] == 0
    assert no_group["median_actions_to_first_levelup_blind"] is None
    no_target = exp4547.measure_ranked_candidate_groups(
        [_example("solo", 1, False), _example("solo", 2, False)],
        scorer=scorer,
        min_candidates=2,
    )
    assert no_target["heldout_group_count"] == 0
    assert exp4547._actions_to_first_levelup([ArcAction(1, None, "noop")]) is None


def test_req_arc_fcp_4547_rich_action_candidates_accepts_cnn_ranker(monkeypatch) -> None:
    """REQ-ARC-FCP-4547: rich_action_candidates ranks by CNN scorer with legacy ties."""

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


def test_req_arc_fcp_4547_loading_split_and_training_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-FCP-4547: helper branches keep corpus handling deterministic."""

    data_dir = tmp_path / exp4547.DATA_RELATIVE_DIR
    data_dir.mkdir(parents=True)
    (data_dir / "manifest.json").write_text(
        json.dumps({"example_count": 3, "shard_count": 1, "shards": [], "source_metadata": {}}),
        encoding="utf-8",
    )

    assert exp4547.load_corpus_manifest(tmp_path)["example_count"] == 3
    assert exp4547.load_corpus_manifest(tmp_path / "missing")["example_count"] == 0

    monkeypatch.setattr(exp4547, "load_frame_action_effect_examples", lambda path, limit=None: [])
    assert exp4547.load_cached_examples(tmp_path, limit=2) == []

    one_game = [_example("aa00", 1, True)]
    assert exp4547.split_train_heldout_by_game(one_game) == (one_game, one_game)

    changed_only = [_example("aa00", 1, True), _example("bb00", 2, True)]
    assert exp4547.balanced_training_subset(changed_only, max_examples=1) == changed_only[:1]

    class DummyModel(torch.nn.Module):
        def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.zeros((tensor.shape[0], 1, 8, 8), dtype=torch.float32),
                torch.zeros((tensor.shape[0], 5), dtype=torch.float32),
            )

    scorer = exp4547._scorer_from_model(DummyModel(), num_colors=8, frame_size=8)
    assert isinstance(scorer, fcp.FrameChangeScorer)


def test_scenario_arc_fcp_4547_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4547: run writes the required JSON after mocked training."""

    examples = [
        _example("aa00", 6, False, x=1, y=1),
        _example("aa00", 6, True, x=6, y=6),
        _example("bb00", 1, False),
        _example("bb00", 2, True),
    ]

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

    def load_fixture(_root: Path, *, limit: int | None = None):
        assert limit is None
        return examples

    def train_fixture(_examples, **_kwargs):
        return DummyModel(), {"examples_seen": len(_examples), "batches_trained": 1}

    def scorer_fixture(_model, **_kwargs):
        return scorer

    artifact = exp4547.run(
        root=tmp_path,
        write=True,
        load_examples=load_fixture,
        train_model=train_fixture,
        scorer_factory=scorer_fixture,
        random_seed=4547,
        frame_size=8,
        num_colors=8,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "success: frame_change_cnn_median_actions_reduced_1"
    assert artifact["median_actions_to_first_levelup_blind"] == 2.0
    assert artifact["median_actions_to_first_levelup_cnn"] == 1.0
    assert artifact["solve_rate_preserved"] is True
    assert artifact["cnn_held_out_delta_auroc"] == pytest.approx(1.0)
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["random_seed"] == 4547
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert exp4547.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4547.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]


def test_req_arc_fcp_4547_artifact_schema_rejects_missing_or_unsafe_fields() -> None:
    """REQ-ARC-FCP-4547: schema rejects bad verdicts, drops, and unchecked nulls."""

    artifact = exp4547.build_artifact(
        preconditions_checked={"offline_arcade_import": True, "corpus_cached": True},
        corpus_summary={"corpus_examples_loaded": 4},
        training_summary={"batches_trained": 1},
        delta_metrics={
            "cnn_held_out_delta_auroc": 0.75,
            "trivial_delta_auroc": 0.5,
            "positive_control_passed": True,
            "heldout_transition_count": 4,
        },
        ranking_metrics={
            "median_actions_to_first_levelup_blind": 2.0,
            "median_actions_to_first_levelup_cnn": 2.0,
            "solve_rate_blind": 1.0,
            "solve_rate_cnn": 1.0,
            "solve_rate_preserved": True,
            "action_reduction": False,
        },
        positive_control={"actions_reduced": True},
        random_seed=4547,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: frame_change_cnn_no_action_reduction_honest_null"
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["secondary_input"]["topic"] == "hidden_field_state_hash_probe"
    assert exp4547.artifact_schema_errors(artifact) == []

    bad_verdict = {**artifact, "honest_verdict": "null"}
    assert any("terminal prefix" in error for error in exp4547.artifact_schema_errors(bad_verdict))

    dropped_success = {
        **artifact,
        "honest_verdict": "success: frame_change_cnn_median_actions_reduced_1",
        "solve_rate_preserved": False,
    }
    assert any("solve-rate" in error for error in exp4547.artifact_schema_errors(dropped_success))

    unchecked_null = {**artifact, "positive_control_passed": False, "false_negative_risk_checked": True}
    assert any("false_negative" in error for error in exp4547.artifact_schema_errors(unchecked_null))

    missing = dict(artifact)
    missing.pop("ranking_metrics")
    assert any("missing required field ranking_metrics" in error for error in exp4547.artifact_schema_errors(missing))

    wrong_substrate = {**artifact, "inference_substrate": "live_llm_inference"}
    assert any("inference_substrate" in error for error in exp4547.artifact_schema_errors(wrong_substrate))

    wrong_principles = {**artifact, "field_principles": {}}
    assert any("field_principles" in error for error in exp4547.artifact_schema_errors(wrong_principles))

    wrong_preconditions = {**artifact, "preconditions_checked": []}
    assert any("preconditions" in error for error in exp4547.artifact_schema_errors(wrong_preconditions))

    wrong_checksum = {**artifact, "reproducibility_checksum": "0" * 64}
    assert any("checksum" in error for error in exp4547.artifact_schema_errors(wrong_checksum))

    false_success = {
        **artifact,
        "honest_verdict": "success: frame_change_cnn_median_actions_reduced_2",
        "median_actions_to_first_levelup_cnn": 2.0,
        "median_actions_to_first_levelup_blind": 2.0,
        "solve_rate_preserved": True,
    }
    assert any("lower CNN median" in error for error in exp4547.artifact_schema_errors(false_success))

    offline = exp4547.build_artifact(
        preconditions_checked={"offline_arcade_import": False},
        corpus_summary={},
        training_summary={},
        delta_metrics={"positive_control_passed": True},
        ranking_metrics={"solve_rate_preserved": True},
        positive_control={"actions_reduced": True},
        random_seed=4547,
        reproducibility_checksum="sha256:" + "1" * 64,
        duration_s=0.0,
    )
    assert offline["honest_verdict"] == "complete: blocked_offline_arcade_import_failed"

    no_torch = exp4547.build_artifact(
        preconditions_checked={"torch_import": False},
        corpus_summary={},
        training_summary={},
        delta_metrics={"positive_control_passed": True},
        ranking_metrics={"solve_rate_preserved": True},
        positive_control={"actions_reduced": True},
        random_seed=4547,
        reproducibility_checksum="sha256:" + "2" * 64,
        duration_s=0.0,
    )
    assert no_torch["honest_verdict"] == "complete: blocked_torch_missing"

    no_corpus = exp4547.build_artifact(
        preconditions_checked={"corpus_cached": False},
        corpus_summary={},
        training_summary={},
        delta_metrics={"positive_control_passed": True},
        ranking_metrics={"solve_rate_preserved": True},
        positive_control={"actions_reduced": True},
        random_seed=4547,
        reproducibility_checksum="sha256:" + "3" * 64,
        duration_s=0.0,
    )
    assert no_corpus["honest_verdict"] == "complete: blocked_human_replay_corpus_not_cached"

    failed_control = exp4547.build_artifact(
        preconditions_checked={},
        corpus_summary={},
        training_summary={},
        delta_metrics={"positive_control_passed": False},
        ranking_metrics={"solve_rate_preserved": True},
        positive_control={"actions_reduced": True},
        random_seed=4547,
        reproducibility_checksum="sha256:" + "4" * 64,
        duration_s=0.0,
    )
    assert failed_control["honest_verdict"] == "complete: frame_change_cnn_positive_control_failed"
    assert failed_control["secondary_input"] is None

    dropped = exp4547.build_artifact(
        preconditions_checked={},
        corpus_summary={},
        training_summary={},
        delta_metrics={"positive_control_passed": True},
        ranking_metrics={"solve_rate_preserved": False},
        positive_control={"actions_reduced": True},
        random_seed=4547,
        reproducibility_checksum="sha256:" + "5" * 64,
        duration_s=0.0,
    )
    assert dropped["honest_verdict"] == "complete: frame_change_cnn_solve_rate_guard_failed"


def test_scenario_arc_fcp_4547_empty_run_and_schema_error_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-FCP-4547: empty corpus reports a guarded positive-control failure."""

    artifact = exp4547.run(
        root=tmp_path,
        write=False,
        preconditions_checked={"offline_arcade_import": True, "torch_import": True, "corpus_cached": True},
        load_examples=lambda _root, limit=None: [],
        random_seed=4547,
        now=lambda: 5.0,
    )

    assert artifact["honest_verdict"] == "complete: frame_change_cnn_positive_control_failed"
    assert artifact["training_summary"]["batches_trained"] == 0
    assert artifact["cnn_held_out_delta_auroc"] == 0.5
    assert artifact["false_negative_risk_checked"] is False

    monkeypatch.setattr(exp4547, "artifact_schema_errors", lambda _artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        exp4547.run(
            root=tmp_path,
            write=False,
            preconditions_checked={"offline_arcade_import": True, "torch_import": True, "corpus_cached": True},
            load_examples=lambda _root, limit=None: [],
            random_seed=4547,
            now=lambda: 5.0,
        )
