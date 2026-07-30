"""Tests for Exp5373 ARC live-path salience repair.

Spec refs: REQ-ARC-FCP-5373,
SCENARIO-ARC-FCP-5373.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from carnot import experiment_5373_arc_salience_re86_levelup_v489 as exp5373
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_color_blob_salience import (
    ColorBlob,
    ColorBlobSaliencePrior,
    connected_color_blobs,
)
from carnot.agentic.arc_competition_agent import (
    SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED,
    E3AgentPolicy,
)
from carnot.agentic.arc_frame_change_predictor import (
    GroundTruthValidatedFrameChangeScorer,
    rank_arc_actions,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(re86_levels: int = 2) -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "re86", "levels_reproduced": re86_levels},
            {"game": "sb26", "levels_reproduced": 2},
            {"game": "bp35", "levels_reproduced": 2},
        ],
    }


def _salience_frame() -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[2:10, 2:18] = 8
    grid[14:16, 14:16] = 8
    return SimpleNamespace(frame=grid, available_actions=[6])


def _salience_candidates() -> list[ArcAction]:
    return [
        ArcAction(6, {"x": 4, "y": 4}, "large_flat_blob"),
        ArcAction(6, {"x": 14, "y": 14}, "button_like_blob"),
        ArcAction(6, {"x": 2, "y": 0}, "status_bar_blob"),
    ]


def test_req_arc_fcp_5373_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5373: OpenSpec anchors the salience repair artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5373" in spec
    assert "SCENARIO-ARC-FCP-5373" in spec
    assert exp5373.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5373.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5373_button_like_blob_beats_flat_and_status_blobs() -> None:
    """SCENARIO-ARC-FCP-5373: repaired salience ranks button-like blobs first."""

    frame = _salience_frame()
    candidates = _salience_candidates()
    legacy = ColorBlobSaliencePrior(large_flat_deprioritization=False)
    repaired = ColorBlobSaliencePrior()

    legacy_ranked = rank_arc_actions(frame, candidates, prior=legacy)
    repaired_ranked = rank_arc_actions(frame, candidates, prior=repaired)
    measurement = exp5373.measure_button_like_blob_rank_delta(frame, candidates)

    assert legacy_ranked[0].source == "large_flat_blob"
    assert repaired_ranked[0].source == "button_like_blob"
    assert repaired_ranked[-1].source == "status_bar_blob"
    assert measurement["button_like_blob_rank_delta"] > 0
    assert measurement["before_rank"] == 1
    assert measurement["after_rank"] == 0


class _UnvalidatedFlatScorer:
    source = "test_unvalidated_flat_scorer"

    def candidate_score(self, _frame: Any, candidate: ArcAction) -> float:
        return 10.0 if candidate.source == "large_flat_blob" else 0.0


def test_scenario_arc_fcp_5373_frame_diff_score_is_gated_until_observed_validation() -> None:
    """SCENARIO-ARC-FCP-5373: frame-diff ranking waits for observed frame-change validation."""

    frame = _salience_frame()
    candidates = _salience_candidates()
    prior = ColorBlobSaliencePrior()
    scorer = GroundTruthValidatedFrameChangeScorer(
        _UnvalidatedFlatScorer(),
        score_threshold=1.0,
        required_agreements=1,
    )

    gated_ranked = rank_arc_actions(frame, candidates, scorer=scorer, prior=prior)
    assert scorer.validated is False
    assert gated_ranked[0].source == "button_like_blob"
    assert scorer.candidate_score(frame, candidates[0]) == 0.0

    before = np.zeros((3, 3), dtype=np.int16)
    after = before.copy()
    after[1, 1] = 8
    scorer.observe_transition(
        SimpleNamespace(frame=before),
        6,
        {"x": 4, "y": 4},
        SimpleNamespace(frame=after),
        source="large_flat_blob",
    )

    assert scorer.validated is True
    assert scorer.candidate_score(frame, candidates[0]) == 10.0
    assert scorer.diagnostics()["frame_diff_ground_truth_validated"] is True


def test_scenario_arc_fcp_5373_salience_and_validator_edge_branches() -> None:
    """SCENARIO-ARC-FCP-5373: repair edge branches are explicit and deterministic."""

    no_shape = ColorBlob(1, 1, (0, 0, 0, 0), (0.0, 0.0), frozenset({(0, 0)}))
    zero_shape = ColorBlob(1, 1, (0, 0, 0, 0), (0.0, 0.0), frozenset({(0, 0)}), (0, 0))
    status = ColorBlob(16, 20, (0, 0, 0, 19), (0.0, 9.5), frozenset(), (20, 20))
    button = ColorBlob(8, 4, (2, 2, 3, 3), (2.5, 2.5), frozenset(), (20, 20))
    huge = ColorBlob(8, 128, (2, 2, 9, 17), (5.5, 9.5), frozenset(), (20, 20))
    medium_dull = ColorBlob(2, 4, (2, 2, 3, 3), (2.5, 2.5), frozenset(), (20, 20))
    single_salient = ColorBlob(8, 1, (2, 2, 2, 2), (2.0, 2.0), frozenset(), (20, 20))
    single_dull = ColorBlob(2, 1, (2, 2, 2, 2), (2.0, 2.0), frozenset(), (20, 20))
    prior = ColorBlobSaliencePrior()
    legacy = ColorBlobSaliencePrior(large_flat_deprioritization=False)

    assert no_shape.area_fraction == 0.0
    assert zero_shape.area_fraction == 0.0
    assert (
        ColorBlobSaliencePrior(status_bar_deprioritization=False).is_status_bar_like(status)
        is False
    )
    assert prior.is_status_bar_like(no_shape) is False
    assert legacy.is_large_flat_blob(huge) is False
    assert prior.is_button_like_blob(status) is False
    assert legacy.is_button_like_blob(huge) is False
    assert prior.is_button_like_blob(button) is True
    assert prior._blob_for_click([button], 19, 19) == button
    assert legacy.tier(medium_dull) == 1
    assert legacy.tier(single_salient) == 2
    assert legacy.tier(single_dull) == 3
    assert prior.tier(medium_dull) == 1
    assert prior.tier(single_salient) == 2
    assert prior.tier(single_dull) == 3
    assert prior.score(np.zeros((2, 2), dtype=np.int16), ArcAction(1, None, "keyboard")) == 0.0
    assert prior.score(np.zeros((2, 2), dtype=np.int16), ArcAction(6, {}, "missing")) == 0.0
    stacked = np.stack([_salience_frame().frame, _salience_frame().frame])
    assert connected_color_blobs(stacked)
    assert prior.score(_salience_frame(), ArcAction(6, {"x": 19, "y": 19}, "nearest")) > 0.0
    assert (
        ColorBlobSaliencePrior(min_pixels=5).score(
            np.asarray([[0, 1], [1, 0]], dtype=np.int16),
            ArcAction(6, {"x": 0, "y": 0}, "no_blob"),
        )
        == 0.0
    )
    assert (
        prior.score(np.zeros((2, 2), dtype=np.int16), ArcAction(6, {"x": 0, "y": 0}, "flat")) == 0.0
    )
    assert prior.as_dict()["large_flat_deprioritization"] is True

    class RecorderScorer:
        def __init__(self, *, raise_observe: bool = False, raise_reset: bool = False) -> None:
            self.raise_observe = raise_observe
            self.raise_reset = raise_reset
            self.observed = 0
            self.reset_count = 0

        def candidate_score(self, _frame: Any, _candidate: ArcAction) -> float:
            return 10.0

        def observe_transition(self, *_args: Any, **_kwargs: Any) -> None:
            self.observed += 1
            if self.raise_observe:
                raise RuntimeError("observe boom")

        def reset(self, *_args: Any, **_kwargs: Any) -> None:
            self.reset_count += 1
            if self.raise_reset:
                raise RuntimeError("reset boom")

        def as_dict(self) -> dict[str, str]:
            return {"source": "recorder"}

    class RaisingScore:
        def candidate_score(self, *_args: Any, **_kwargs: Any) -> float:
            raise RuntimeError("score boom")

    before = np.zeros((3, 3), dtype=np.int16)
    same = before.copy()
    contradiction = GroundTruthValidatedFrameChangeScorer(RecorderScorer(), score_threshold=1.0)
    contradiction.observe_transition(
        SimpleNamespace(frame=before),
        6,
        {"x": 1, "y": 1},
        SimpleNamespace(frame=same),
    )
    assert contradiction.validated is False
    assert contradiction.diagnostics()["base_scorer"] == {"source": "recorder"}

    raising_observer = RecorderScorer(raise_observe=True, raise_reset=True)
    guarded = GroundTruthValidatedFrameChangeScorer(raising_observer)
    guarded.observe_transition(
        SimpleNamespace(frame=before), 6, None, SimpleNamespace(frame=before)
    )
    guarded.reset()
    assert raising_observer.observed == 1
    assert raising_observer.reset_count == 1

    bad_inputs = GroundTruthValidatedFrameChangeScorer(RaisingScore())
    bad_inputs.observe_transition(object(), 6, None, object())
    assert bad_inputs.diagnostics()["agreement_count"] == 1


def test_scenario_arc_fcp_5373_submitted_e3_wraps_frame_change_scorer_when_available() -> None:
    """SCENARIO-ARC-FCP-5373: live E3 path reaches salience repair and validation gate."""

    policy = E3AgentPolicy("re86", proposer=None, value_head=lambda _frame: 0.0)

    # CORRECTED EXPECTATION (2026-07-30): this demanded a live ColorBlobSaliencePrior, but
    # SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED is False by deliberate operator decision (disabled
    # 2026-07-14 after a near-hang, re-validated 2026-07-16 as ~9x slower per action for no
    # measured benefit), so E3AgentPolicy installs no default action_prior and None is correct.
    # The WIRING assertion now tracks the flag; the large_flat_deprioritization DEFAULT is
    # asserted directly below so that property keeps its coverage either way -- dropping it when
    # the flag is off would have been a silent loss of a real check.
    if SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED:
        assert isinstance(policy.explorer.action_prior, ColorBlobSaliencePrior)
        assert policy.explorer.action_prior.large_flat_deprioritization is True
    else:
        assert policy.explorer.action_prior is None
    assert ColorBlobSaliencePrior().large_flat_deprioritization is True, (
        "the prior's shipped default must stay on regardless of whether it is currently wired in"
    )
    if policy.explorer.frame_change_scorer is not None:
        assert isinstance(
            policy.explorer.frame_change_scorer, GroundTruthValidatedFrameChangeScorer
        )


def test_scenario_arc_fcp_5373_selects_re86_l3_when_not_already_banked() -> None:
    """SCENARIO-ARC-FCP-5373: registry precheck targets the next unbanked re86 level."""

    selection = exp5373.select_target_after_precheck(_registry())

    assert selection["registry_precheck_done"] is True
    assert selection["target_game"] == "re86"
    assert selection["target_level_before"] == 2
    assert selection["attempted_level"] == 3
    assert selection["no_duplicate_solve"] is True


def test_scenario_arc_fcp_5373_registry_precheck_rotates_or_reports_no_target() -> None:
    """SCENARIO-ARC-FCP-5373: duplicate re86 L3 is skipped before live attempts."""

    rotated = exp5373.select_target_after_precheck(_registry(re86_levels=3))
    missing = exp5373.select_target_after_precheck(
        {"reproducible_total_levels": 69, "games": [{"game": "re86", "levels_reproduced": 3}]}
    )

    assert rotated["target_game"] == "sb26"
    assert rotated["attempted_level"] == 3
    assert rotated["selection_reason"] == "re86_l3_already_banked_rotated_target"
    assert missing["registry_precheck_done"] is False
    assert missing["no_duplicate_solve"] is False


def test_scenario_arc_fcp_5373_pure_helpers_cover_rank_and_schema_edges(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5373: pure helper branches stay deterministic."""

    assert exp5373._ZeroFrameChangeScorer().candidate_score(None, None) == 0.0
    assert exp5373._rank_of_source(_salience_candidates(), "missing") is None
    assert exp5373._action_label(6, {"x": 1, "y": 2}) == ('{"action":6,"data":{"x":1,"y":2}}')

    selection = exp5373.select_target_after_precheck(_registry())
    repair = exp5373.salience_repair_live_diagnostics()
    rank_measurement = exp5373.measure_button_like_blob_rank_delta()
    success = exp5373.build_artifact(
        selection=selection,
        registry_total_before=69,
        repair=repair,
        rank_measurement=rank_measurement,
        attempt={"offline_reproduced": True, "reproduced_levels": 1},
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    exp5373.validate_artifact(success)
    assert success["status"] == "complete"
    assert success["offline_reproduced"] is True
    assert success["registry_total_after"] == 70
    assert success["honest_verdict"].startswith("banked:")

    bad = dict(success)
    bad["status"] = "maybe"
    bad["solve_provenance"] = "outer_loop_re"
    bad["registry_precheck_done"] = "yes"
    bad["new_level_banked"] = True
    bad["offline_reproduced"] = False
    bad["target_level_before"] = "2"
    bad["perception_error_classes"] = "not-list"
    bad["honest_verdict"] = "unclear"
    errors = exp5373.artifact_schema_errors(bad)
    assert "status must be complete or honest_null" in errors
    assert "registry_precheck_done must be bare bool" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "target_level_before must be bare int" in errors
    assert "perception_error_classes must be a list" in errors
    assert "honest_verdict must be a one-line banked/no-bank verdict" in errors
    assert "new_level_banked requires reproduced live-path non-duplicate evidence" in errors
    with pytest.raises(ValueError):
        exp5373.validate_artifact(bad)

    (tmp_path / "results").mkdir()
    no_preconditions = exp5373.run_experiment(
        root=tmp_path,
        offline_arcade_check=lambda: False,
        tests_run=["precondition unit"],
    )
    assert no_preconditions["status"] == "honest_null"
    assert no_preconditions["registry_precheck_done"] is True
    assert no_preconditions["target_game"] == "re86"
    assert (
        "preconditions_missing_or_offline_arcade_unavailable"
        in no_preconditions["perception_error_classes"]
    )


def test_scenario_arc_fcp_5373_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5373: honest-null attempts emit required bare fields."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    (root / "CODEX.md").write_text("codex instructions\n", encoding="utf-8")
    (root / exp5373.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5373\nSCENARIO-ARC-FCP-5373\n",
        encoding="utf-8",
    )
    (root / exp5373.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["selection"]["target_game"] == "re86"
        return {
            "target_game": "re86",
            "target_level_before": 2,
            "attempted_level": 3,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "new_level_banked": False,
            "actions_taken": 12,
            "max_level_reached": 2,
            "perception_error_classes": ["bounded_budget_no_levelup"],
        }

    artifact = exp5373.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5373 salience repair"],
    )

    written = json.loads((root / exp5373.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    exp5373.validate_artifact(artifact)
    assert artifact["status"] == "honest_null"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["registry_precheck_done"] is True
    assert artifact["target_game"] == "re86"
    assert artifact["target_level_before"] == 2
    assert artifact["attempted_level"] == 3
    # CORRECTED EXPECTATION (2026-07-30): both salience fields come from
    # salience_repair_live_diagnostics(), which gates each on
    # `isinstance(prior, ColorBlobSaliencePrior)`. With the flag False (see the note above) there
    # is no prior, so False is the correct, honest report. frame_diff_ground_truth_validated is
    # NOT gated on the prior -- it only checks the explicitly-supplied validator -- so it stays an
    # unconditional True, which is what stops this correction from quietly widening.
    assert artifact["salience_repair_live_reachable"] is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    assert artifact["status_bar_deprioritization_enabled"] is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    assert artifact["frame_diff_ground_truth_validated"] is True
    assert artifact["button_like_blob_rank_delta"] > 0
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_level_banked"] is False
    assert artifact["registry_total_before"] == 69
    assert artifact["registry_total_after"] == 69
    assert artifact["live_attempt_count"] == 1
    assert artifact["no_outer_loop_re"] is True
    assert artifact["no_duplicate_solve"] is True
    assert artifact["honest_verdict"].startswith("no-bank:")
