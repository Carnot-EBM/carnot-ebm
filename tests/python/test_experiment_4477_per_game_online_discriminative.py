"""Tests for Exp 4477 per-game online discriminative frontier pruning.

Spec refs: REQ-PHASE4-4477, SCENARIO-PHASE4-4477.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4477_per_game_online_discriminative as mod
from carnot.agentic.arc_competition_agent import StepwiseExplorer
from carnot.agentic.arc_value_learner import DiscriminativeVerifier


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


class _Frame:
    def __init__(self, value: int, *, state: str = "", actions: list[int] | None = None) -> None:
        self.frame = np.array([[value]], dtype=np.int16)
        self.state = state
        self.available_actions = actions if actions is not None else [1]
        self.levels_completed = 0


def _feature(frame: Any) -> list[float]:
    return [float(np.asarray(frame.frame)[0, 0])]


def test_req_phase4_4477_spec_declares_online_discriminative_contract() -> None:
    """REQ-PHASE4-4477: OpenSpec names the online verifier and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4477" in spec
    assert "SCENARIO-PHASE4-4477" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "DiscriminativeVerifier" in spec
    assert "P(on-path)" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_4477_discriminative_verifier_scores_feature_vectors() -> None:
    """REQ-PHASE4-4477: the classifier can score cached frontier feature vectors."""

    verifier = DiscriminativeVerifier(lambda frame: frame).fit(
        [[0.0], [0.1], [9.0], [10.0]],
        [1.0, 1.0, 0.0, 0.0],
        iters=300,
        lr=0.3,
    )

    assert verifier.proba_features([0.0]) > 0.75
    assert verifier.proba_features([10.0]) < 0.25


def test_scenario_phase4_4477_stepwise_collects_negatives_and_prunes_frontier() -> None:
    """SCENARIO-PHASE4-4477: game-over negatives train a per-game online pruner."""

    explorer = StepwiseExplorer(
        online_discriminative=True,
        discriminative_featurizer=_feature,
        discriminative_min_positives=1,
        discriminative_min_negatives=1,
        discriminative_fit_iters=300,
        discriminative_prune_threshold=0.55,
    )
    alive = _Frame(0)
    game_over = _Frame(10, state="GAME_OVER")

    explorer._ingest(alive)
    explorer.awaiting = {"origin": explorer.cur, "action": 1, "data": None}
    explorer._ingest(game_over)
    diagnostics = explorer.online_discriminator_diagnostics()

    assert diagnostics["trained"] is True
    assert diagnostics["positive_samples"] == 1
    assert diagnostics["negative_samples"] == 1
    assert diagnostics["negative_sources"]["game_over"] == 1

    explorer.graph = {
        "bad": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 2, "data": None}],
            "value": 0.0,
            "discriminative_features": [10.0],
        },
        "good": {
            "path": [{"action": 3, "data": None}],
            "untested": [{"action": 4, "data": None}],
            "value": 0.0,
            "discriminative_features": [0.0],
        },
    }

    assert explorer._frontier() == "good"
    assert explorer.graph["bad"]["on_path_proba"] < 0.55
    assert explorer.graph["good"]["on_path_proba"] > 0.55
    assert explorer.online_discriminator_diagnostics()["frontier_pruned"] >= 1


def test_req_phase4_4477_artifact_schema_and_writer(tmp_path: Path) -> None:
    """REQ-PHASE4-4477: artifact validates required fields and writes JSON."""

    rows = [
        mod.PerGameComparison(
            game="heldout-a",
            baseline_solved=False,
            online_solved=True,
            baseline_actions_to_first_levelup=None,
            online_actions_to_first_levelup=7,
            baseline_reached_level=0,
            online_reached_level=1,
            baseline_actions_spent=12,
            online_actions_spent=7,
            online_verifier={"trained": True, "negative_samples": 2},
        )
    ]
    artifact = mod.build_artifact(
        per_game_results=rows,
        preconditions_checked=mod.Preconditions(
            offline_fixtures_present=True,
            arc_solver_kit_importable=True,
            arcengine_importable=True,
            submitted_to_leaderboard=False,
        ),
    )

    assert (
        artifact["honest_verdict"] == "success: per_game_online_discriminative_improves_solve_rate"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["solve_rate_delta"] == pytest.approx(1.0)
    assert artifact["actions_to_first_levelup_delta"] == -5
    assert mod.artifact_schema_errors(artifact) == []

    path = mod.write_artifact(tmp_path, artifact)
    assert path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


def test_req_phase4_4477_schema_rejects_fabrication() -> None:
    """REQ-PHASE4-4477: invalid terminal fields and solve claims fail closed."""

    artifact = mod.build_artifact(per_game_results=[])
    bad = {
        **artifact,
        "honest_verdict": "partial: still_running",
        "inference_substrate": "live_llm_inference",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "preconditions_checked": [],
        "submitted_to_leaderboard": True,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "loose"}},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must be verifier_ensemble_against_cached_candidates" in errors
    assert "offline_reproduced true requires reproduced_levels >= 1" in errors
    assert "preconditions_checked must be a dict" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-PHASE4-4477" in errors
