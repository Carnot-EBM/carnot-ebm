"""Tests for Exp 4665 DAgger-lite distribution-shift value routing.

Spec refs: REQ-LEARN-4665, SCENARIO-LEARN-4665-DAGGER-DATA,
SCENARIO-LEARN-4665-LIVE-ROUTE, SCENARIO-LEARN-4665-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _attempt(
    signature: str,
    *,
    mode: str,
    first_win: bool,
    reached_level: int,
) -> dict[str, Any]:
    return {
        "game": signature.split("~", 1)[0],
        "variant_signature": signature,
        "variant": 1,
        "kind": "color",
        "attempted": True,
        "first_win": bool(first_win),
        "solved": bool(first_win),
        "reached_level": int(reached_level),
        "actions": 7,
        "actions_to_first_levelup": 7 if first_win else None,
        "solution_labels": ['{"action":1,"data":null}'] if first_win else [],
        "policy_mode": mode,
        "timed_out": False,
        "reproduction_gate": {
            "reproduced": bool(first_win),
            "reached_level": int(reached_level),
        },
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade": True,
        "value_learner_import": True,
        "agent_import": True,
        "b1_artifact_present": True,
        "a1_artifact_present": True,
        "spec_has_req_4665": True,
        "live_llm_inference": False,
    }


def test_req_learn_4665_spec_declares_dagger_contract() -> None:
    """REQ-LEARN-4665: OpenSpec declares the DAgger-lite artifact contract."""

    from carnot import experiment_4665_dagger_distribution_shift_value_routing as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4665" in spec
    assert "SCENARIO-LEARN-4665-DAGGER-DATA" in spec
    assert "SCENARIO-LEARN-4665-LIVE-ROUTE" in spec
    assert "SCENARIO-LEARN-4665-ARTIFACT" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_learn_4665_dagger_rows_relabel_and_train_value_head() -> None:
    """SCENARIO-LEARN-4665-DAGGER-DATA: live frontier rows are relabeled and learned."""

    from carnot import experiment_4665_dagger_distribution_shift_value_routing as mod
    from carnot.agentic.arc_value_learner import fit_dagger_win_reachability_value_head

    winning_labels = [mod.path_action_label({"action": 1, "data": None})]
    frontier_rows = [
        {
            "source": "alive_frontier",
            "features": [0.0, 0.1],
            "path": [{"action": 1, "data": None}],
            "label": 0.0,
        },
        {
            "source": "alive_frontier",
            "features": [5.0, 5.0],
            "path": [{"action": 2, "data": None}],
            "label": 1.0,
        },
    ]

    relabeled = mod.relabel_frontier_rows(frontier_rows, winning_labels=winning_labels)
    assert [row["label"] for row in relabeled] == [1.0, 0.0]

    aggregate = mod.aggregate_dagger_rows(
        winning_rows=[
            {"source": "winning_path", "features": [0.0, 0.0], "label": 1.0},
            {"source": "winning_path", "features": [0.2, 0.1], "label": 1.0},
        ],
        frontier_rows=relabeled,
    )
    assert aggregate["positive_count"] == 3
    assert aggregate["negative_count"] == 1
    assert aggregate["frontier_count"] == 2

    head = fit_dagger_win_reachability_value_head(
        [row["features"] for row in aggregate["rows"]],
        [row["label"] for row in aggregate["rows"]],
        iters=200,
        lr=0.4,
    )
    assert head.verifier_is_oracle is False
    assert head.proba_features([0.0, 0.0]) > head.proba_features([5.0, 5.0])
    assert head.cost_features([0.0, 0.0]) < head.cost_features([5.0, 5.0])


def test_scenario_learn_4665_live_route_prefers_dagger_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-4665-LIVE-ROUTE: submitted loader prefers corrected head."""

    from carnot.agentic import arc_competition_agent as comp
    from carnot.agentic.arc_value_learner import (
        cross_game_feature_names_v3_value_routing,
        fit_dagger_win_reachability_value_head,
    )

    width = len(cross_game_feature_names_v3_value_routing())
    pos = [[0.0] * width, [0.1] * width]
    neg = [[4.0] * width, [5.0] * width]
    head = fit_dagger_win_reachability_value_head(
        pos + neg,
        [1.0, 1.0, 0.0, 0.0],
        iters=120,
    )
    checkpoint = tmp_path / comp.DAGGER_VALUE_HEAD_RELATIVE_PATH
    head.save(checkpoint, meta={"spec_refs": ["REQ-LEARN-4665"]})

    monkeypatch.setattr(comp, "load_live_spatial_value_head", lambda *args, **kwargs: None)
    loaded = comp._load_linear_cross_game_value_head(root=tmp_path)

    assert loaded is not None
    assert loaded.feature_subset == comp.SUBMITTED_VALUE_HEAD_FEATURE_SUBSET
    assert loaded.verifier_is_oracle is False
    assert loaded.cost_features([0.0] * width) < loaded.cost_features([5.0] * width)


def test_scenario_learn_4665_artifact_schema_records_null_and_shift_drop() -> None:
    """SCENARIO-LEARN-4665-ARTIFACT: no-lift null still records mechanism evidence."""

    from carnot import experiment_4665_dagger_distribution_shift_value_routing as mod

    corrected = mod.measurement_from_attempts(
        [
            _attempt("aa00~color01", mode="corrected", first_win=True, reached_level=1),
            _attempt("bb00~color01", mode="corrected", first_win=False, reached_level=0),
        ]
    )
    baseline = mod.measurement_from_attempts(
        [
            _attempt("aa00~color01", mode="baseline", first_win=True, reached_level=1),
            _attempt("bb00~color01", mode="baseline", first_win=False, reached_level=0),
        ]
    )

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        corrected_measurement=corrected,
        baseline_measurement=baseline,
        dagger_dataset={
            "rows": [],
            "positive_count": 3,
            "negative_count": 2,
            "frontier_count": 2,
        },
        distribution_shift_before=0.699108,
        distribution_shift_after=0.2,
        b1_artifact={"distribution_shift_score": 0.699108},
        a1_artifact={"bare_control_passed": True},
        parity_test={"passed": True},
        orphan_lint={"passed": True},
        model_checkpoint="models/arc_dagger_value_routing_v3.json",
        duration_s=1.0,
    )

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == (
        "complete: dagger_distribution_corrected_no_live_lift_residual_logged."
    )
    assert artifact["shift_score_delta"] == pytest.approx(-0.499108)
    assert artifact["first_win_rate_delta"] == 0.0
    assert artifact["null_methodology_note"]
    assert artifact["residual_bridge_gap"] == "missing_verifier_gap_live_frontier_not_separated"


def test_scenario_learn_4665_stepwise_explorer_logs_search_distribution_rows() -> None:
    """SCENARIO-LEARN-4665-DAGGER-DATA: live explorer exposes frontier sample rows."""

    from carnot.agentic.arc_competition_agent import StepwiseExplorer

    explorer = StepwiseExplorer(
        online_discriminative=True,
        discriminative_featurizer=lambda frame, previous_frame=None: [
            float(frame.value),
            1.0 if previous_frame is not None else 0.0,
        ],
    )
    explorer._record_discriminative_sample(
        SimpleNamespace(value=2.0),
        previous_frame=SimpleNamespace(value=1.0),
        label=1,
        source="unit_frontier",
        node_hash="hash-a",
        path=[{"action": 1, "data": None}],
    )

    rows = explorer.search_distribution_samples()
    assert rows == [
        {
            "features": [2.0, 1.0],
            "label": 1.0,
            "source": "unit_frontier",
            "node_hash": "hash-a",
            "path": [{"action": 1, "data": None}],
        }
    ]
