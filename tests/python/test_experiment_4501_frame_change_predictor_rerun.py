"""Tests for Exp 4501 frame-change predictor rerun.

Spec refs: REQ-ARC-FCP-4501, SCENARIO-ARC-FCP-4501.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carnot import experiment_4501_frame_change_predictor_rerun as exp4501
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic.arc_agi3_live_adapter import ArcAction


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _grid(value: int = 0) -> list[list[int]]:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[2, 2] = int(value)
    return grid.tolist()


def _row(
    action_id: int,
    *,
    x: int | None = None,
    y: int | None = None,
    frame_delta: float = 0.0,
    frame: list[list[int]] | None = None,
    env: str = "aa00",
) -> dict[str, object]:
    data = {"x": int(x), "y": int(y)} if x is not None and y is not None else {}
    return {
        "schema": "carnot.arc_human_replay.frame_action_delta.v1",
        "env": env,
        "guid": "fixture",
        "step_index": 1,
        "frame": frame or _grid(),
        "action": {"id": str(action_id), "data": data},
        "frame_delta": float(frame_delta),
        "level_progress": 1.0 if frame_delta > 0 else 0.0,
    }


def _write_effect_shards(root: Path, rows: list[dict[str, object]]) -> None:
    data_dir = root / exp4501.DATA_RELATIVE_DIR
    shard_dir = data_dir / "shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "train-00000.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    shard_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "schema": "carnot.arc_human_replay.frame_action_delta.v1",
        "example_count": len(rows),
        "shard_count": 1,
        "shards": [{"path": "shards/train-00000.jsonl", "rows": len(rows), "sha256": digest}],
        "source_metadata": {
            "source_kind": "unit_fixture",
            "official_license_verified": False,
        },
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_req_arc_fcp_4501_spec_declares_rerun_artifact_contract() -> None:
    """REQ-ARC-FCP-4501: OpenSpec anchors the rerun and required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4501" in spec
    assert "SCENARIO-ARC-FCP-4501" in spec
    assert exp4501.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4501.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4501_normalizes_raw_frame_rows_and_prior() -> None:
    """REQ-ARC-FCP-4501: shard rows become frame-only examples and behavior priors."""

    frame = _grid(3)
    examples = list(
        fcp.normalize_frame_action_effect_rows(
            [
                _row(1, frame_delta=0.0, frame=frame),
                _row(6, x=6, y=6, frame_delta=0.25, frame=frame),
            ]
        )
    )

    assert [example.action_id for example in examples] == [1, 6]
    assert examples[1].x == 6
    assert examples[1].y == 6
    assert examples[1].changed is True
    assert examples[1].feature_source == "raw_frame_shard_recomputed"

    prior = fcp.build_behavior_prior_from_effect_examples(examples)
    candidates = [
        ArcAction(1, None, "legacy_noop"),
        ArcAction(6, {"x": 6, "y": 6}, "human_click"),
    ]
    ranked = fcp.rank_arc_actions(SimpleNamespace(frame=np.asarray(frame)), candidates, prior=prior)

    assert ranked[0].source == "human_click"


def test_req_arc_fcp_4501_training_step_uses_cnn_heads() -> None:
    """REQ-ARC-FCP-4501: small torch CNN trains on normalized action-effect labels."""

    rows = [
        _row(6, x=6, y=6, frame_delta=1.0, frame=_grid(1)),
        _row(6, x=1, y=1, frame_delta=0.0, frame=_grid(1)),
        _row(1, frame_delta=1.0, frame=_grid(2)),
        _row(2, frame_delta=0.0, frame=_grid(2)),
    ]
    examples = list(fcp.normalize_frame_action_effect_rows(rows))

    model, summary = fcp.train_frame_change_model(
        examples,
        num_colors=4,
        size=8,
        hidden_channels=4,
        epochs=2,
        batch_size=2,
        learning_rate=0.02,
        seed=4501,
    )

    assert summary["examples_used"] == 4
    assert summary["batches_trained"] > 0
    assert summary["initial_loss"] is not None
    assert summary["final_loss"] is not None
    assert torch.isfinite(torch.tensor(float(summary["final_loss"])))

    scorer = fcp.FrameChangeScorer(model, num_colors=4, size=8)
    score = scorer.candidate_score(
        SimpleNamespace(frame=np.asarray(_grid(1))),
        ArcAction(6, {"x": 6, "y": 6}, "click"),
    )
    assert 0.0 <= score <= 1.0


def test_scenario_arc_fcp_4501_candidate_order_proxy_and_false_negative_guard() -> None:
    """SCENARIO-ARC-FCP-4501: held-out proxy keeps solve rate and detects null risk."""

    frame = _grid(4)
    examples = list(
        fcp.normalize_frame_action_effect_rows(
            [
                _row(1, frame_delta=0.0, frame=frame),
                _row(6, x=6, y=6, frame_delta=1.0, frame=frame),
            ]
        )
    )
    prior = fcp.build_behavior_prior_from_effect_examples(examples)
    metric = fcp.evaluate_replay_candidate_order(examples, prior=prior)

    assert metric["heldout_median_actions_before"] == 2.0
    assert metric["heldout_median_actions_after"] == 1.0
    assert metric["solve_rate_before"] == 1.0
    assert metric["solve_rate_after"] == 1.0
    assert metric["solve_rate_dropped"] is False
    assert metric["implied_efficiency_delta"] > 0.0

    positive_control = fcp.evaluate_positive_control()
    assert exp4501.false_negative_risk_guard(metric, positive_control) == (
        "positive_control_passed_candidate_order_gain"
    )
    null_metric = {**metric, "implied_efficiency_delta": 0.0}
    assert exp4501.false_negative_risk_guard(null_metric, positive_control) == (
        "positive_control_passed_null_interpretable"
    )


def test_scenario_arc_fcp_4501_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4501: rerun artifact records corpus shortfall and null guard."""

    frame = _grid(5)
    _write_effect_shards(
        tmp_path,
        [
            _row(1, frame_delta=0.0, frame=frame),
            _row(6, x=6, y=6, frame_delta=1.0, frame=frame),
            _row(1, frame_delta=0.0, frame=_grid(6)),
            _row(6, x=5, y=5, frame_delta=1.0, frame=_grid(6)),
        ],
    )
    preconditions = {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": True,
        "torch_import": True,
        "torch_version": torch.__version__,
        "training_shards_present": True,
        "action_effect_npz_present": False,
        "ok": True,
    }

    artifact = exp4501.run(
        root=tmp_path,
        preconditions_checked=preconditions,
        write=True,
        train_limit=4,
        epochs=1,
        batch_size=2,
        hidden_channels=4,
        frame_size=8,
        num_colors=8,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["expected_action_effect_examples"] == 14672
    assert artifact["corpus_examples_loaded"] == 4
    assert artifact["feature_source"] == "raw_frame_shard_recomputed"
    assert artifact["behavior_prior_emitted"] is True
    assert artifact["weights_bundled"] is False
    assert artifact["solve_rate_dropped"] is False
    assert artifact["false_negative_risk_guard"] in {
        "positive_control_passed_candidate_order_gain",
        "positive_control_passed_null_interpretable",
    }
    assert exp4501.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4501.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["corpus_examples_loaded"] == 4


def test_req_arc_fcp_4501_artifact_schema_rejects_bad_fields(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4501: schema rejects non-terminal verdicts and solve-rate drops."""

    artifact = exp4501.build_artifact(
        preconditions_checked={"offline_arcade_import": True, "torch_import": True},
        training_summary={"examples_used": 2, "initial_loss": 1.0, "final_loss": 0.9},
        heldout_metrics={
            "heldout_median_actions_before": 2.0,
            "heldout_median_actions_after": 1.0,
            "implied_efficiency_delta": 0.75,
            "solve_rate_before": 1.0,
            "solve_rate_after": 1.0,
            "solve_rate_dropped": False,
            "heldout_group_count": 1,
        },
        positive_control={"actions_reduced": True, "implied_efficiency_delta": 0.5},
        corpus_examples_loaded=2,
        corpus_manifest={"example_count": 2, "shard_count": 1, "shards": []},
        prior_summary={"marginal_action_counts": {"1": 1, "6": 1}},
        started=1.0,
        finished=2.0,
    )
    assert exp4501.artifact_schema_errors(artifact) == []

    bad_verdict = {**artifact, "honest_verdict": "blocked"}
    assert any("terminal prefix" in error for error in exp4501.artifact_schema_errors(bad_verdict))

    dropped = {**artifact, "solve_rate_dropped": True}
    assert any("solve rate" in error for error in exp4501.artifact_schema_errors(dropped))

    missing = dict(artifact)
    missing.pop("preconditions_checked")
    assert any("missing required field" in error for error in exp4501.artifact_schema_errors(missing))

    wrong_substrate = {**artifact, "inference_substrate": "live_llm_inference"}
    assert any("inference_substrate" in error for error in exp4501.artifact_schema_errors(wrong_substrate))

    wrong_principles = {**artifact, "field_principles": {}}
    assert any("field_principles" in error for error in exp4501.artifact_schema_errors(wrong_principles))

    wrong_expected = {**artifact, "expected_action_effect_examples": 10}
    assert any("14672" in error for error in exp4501.artifact_schema_errors(wrong_expected))

    wrong_count = {**artifact, "corpus_examples_loaded": -1}
    assert any("non-negative" in error for error in exp4501.artifact_schema_errors(wrong_count))

    wrong_feature = {**artifact, "feature_source": "feature_keys"}
    assert any("raw-frame" in error for error in exp4501.artifact_schema_errors(wrong_feature))

    no_prior = {**artifact, "behavior_prior_emitted": False}
    assert any("behavior_prior_emitted" in error for error in exp4501.artifact_schema_errors(no_prior))

    weights = {**artifact, "weights_bundled": True}
    assert any("weights_bundled" in error for error in exp4501.artifact_schema_errors(weights))

    bad_positive = {**artifact, "positive_control": {"actions_reduced": False}}
    assert any("positive_control" in error for error in exp4501.artifact_schema_errors(bad_positive))

    bad_guard = {**artifact, "false_negative_risk_guard": "unknown"}
    assert any("false_negative_risk_guard" in error for error in exp4501.artifact_schema_errors(bad_guard))


def test_req_arc_fcp_4501_preconditions_manifest_and_blocked_run(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4501: preconditions record missing shards without fabricating training."""

    manifest = exp4501.load_corpus_manifest(tmp_path)
    assert manifest["example_count"] == 0
    assert manifest["shard_count"] == 0

    preconditions = exp4501.check_preconditions(tmp_path)
    assert preconditions["offline_arcade_import"] is True
    assert preconditions["torch_import"] is True
    assert preconditions["training_shards_present"] is False
    assert preconditions["action_effect_npz_present"] is False
    assert preconditions["ok"] is False

    with pytest.raises(ValueError, match="behavior_prior_emitted"):
        exp4501.run(
            root=tmp_path,
            preconditions_checked=preconditions,
            write=False,
            now=lambda: 1.0,
        )


def test_req_arc_fcp_4501_verdict_and_guard_branches() -> None:
    """REQ-ARC-FCP-4501: verdicts cover blocked, exact-corpus, and null branches."""

    positive = {"actions_reduced": True, "implied_efficiency_delta": 0.5}
    failed_control = {"actions_reduced": False}
    null_metric = {
        "heldout_median_actions_before": None,
        "heldout_median_actions_after": None,
        "implied_efficiency_delta": None,
        "solve_rate_before": 0.0,
        "solve_rate_after": 0.0,
        "solve_rate_dropped": False,
        "heldout_group_count": 0,
    }
    gain_metric = {**null_metric, "implied_efficiency_delta": 0.25}
    dropped_metric = {**gain_metric, "solve_rate_dropped": True}
    base_preconditions = {
        "offline_arcade_import": True,
        "torch_import": True,
        "action_effect_npz_present": False,
    }

    assert exp4501.false_negative_risk_guard(null_metric, failed_control) == (
        "positive_control_failed_null_uninterpretable"
    )

    offline = exp4501.build_artifact(
        preconditions_checked={**base_preconditions, "offline_arcade_import": False},
        training_summary={"batches_trained": 1},
        heldout_metrics=null_metric,
        positive_control=positive,
        corpus_examples_loaded=2,
        corpus_manifest={},
        prior_summary={"marginal_action_counts": {"1": 1}},
    )
    assert offline["honest_verdict"] == "complete: blocked_offline_arcade_import_failed"

    no_torch = exp4501.build_artifact(
        preconditions_checked={**base_preconditions, "torch_import": False},
        training_summary={"batches_trained": 1},
        heldout_metrics=null_metric,
        positive_control=positive,
        corpus_examples_loaded=2,
        corpus_manifest={},
        prior_summary={"marginal_action_counts": {"1": 1}},
    )
    assert no_torch["honest_verdict"] == "complete: blocked_torch_missing"

    no_corpus = exp4501.build_artifact(
        preconditions_checked=base_preconditions,
        training_summary={"batches_trained": 0},
        heldout_metrics=null_metric,
        positive_control=positive,
        corpus_examples_loaded=0,
        corpus_manifest={},
        prior_summary={"marginal_action_counts": {"1": 1}},
    )
    assert no_corpus["honest_verdict"] == "complete: blocked_staged_frame_shards_missing"

    dropped = exp4501.build_artifact(
        preconditions_checked=base_preconditions,
        training_summary={"batches_trained": 1},
        heldout_metrics=dropped_metric,
        positive_control=positive,
        corpus_examples_loaded=2,
        corpus_manifest={},
        prior_summary={"marginal_action_counts": {"1": 1}},
    )
    assert dropped["honest_verdict"] == "complete: frame_change_predictor_rerun_solve_rate_guard_failed"

    exact_gain = exp4501.build_artifact(
        preconditions_checked={**base_preconditions, "action_effect_npz_present": True},
        training_summary={"batches_trained": 1},
        heldout_metrics=gain_metric,
        positive_control=positive,
        corpus_examples_loaded=exp4501.EXPECTED_ACTION_EFFECT_EXAMPLES,
        corpus_manifest={},
        prior_summary={"marginal_action_counts": {"1": 1}},
    )
    assert exact_gain["honest_verdict"] == "complete: frame_change_predictor_rerun_exact_corpus_proxy_gain"

    exact_null = exp4501.build_artifact(
        preconditions_checked={**base_preconditions, "action_effect_npz_present": True},
        training_summary={"batches_trained": 1},
        heldout_metrics=null_metric,
        positive_control=positive,
        corpus_examples_loaded=exp4501.EXPECTED_ACTION_EFFECT_EXAMPLES,
        corpus_manifest={},
        prior_summary={"marginal_action_counts": {"1": 1}},
    )
    assert exact_null["honest_verdict"] == "complete: frame_change_predictor_rerun_exact_corpus_honest_null"


def test_req_arc_fcp_4501_split_uses_heldout_states() -> None:
    """REQ-ARC-FCP-4501: larger reruns split by state key for held-out measurement."""

    rows = [_row(1, frame_delta=0.0, frame=_grid(index % 7)) for index in range(25)]
    examples = list(fcp.normalize_frame_action_effect_rows(rows))
    train, heldout = exp4501._split_train_heldout(examples)

    assert train
    assert heldout
    assert {example.state_key for example in train} != {example.state_key for example in heldout}
