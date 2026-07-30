"""Tests for Exp 5360 ARC perception/salience level-up attempt.

Spec refs: REQ-ARC-FCP-5360,
SCENARIO-ARC-FCP-5360.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from carnot import experiment_5360_arc_perception_salience_levelup_attempt_v488 as exp5360
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_color_blob_salience import (
    ColorBlob,
    ColorBlobSaliencePrior,
    connected_color_blobs,
)
from carnot.agentic.arc_competition_agent import (
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED,
    E3AgentPolicy,
)
from carnot.agentic.arc_frame_change_predictor import rank_arc_actions


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame() -> SimpleNamespace:
    grid = np.zeros((10, 10), dtype=np.int16)
    grid[1:6, 1:5] = 2
    grid[7:9, 7:9] = 8
    grid[0, 0:2] = 16
    return SimpleNamespace(frame=grid, available_actions=[6])


def _registry() -> dict[str, object]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "re86", "levels_reproduced": 2, "dead_ends": []},
            {"game": "sb26", "levels_reproduced": 2, "dead_ends": []},
            {
                "game": "bp35",
                "levels_reproduced": 2,
                "dead_ends": ["Exp4970 bp35 no-bank no_grounded_l3_delta"],
            },
            {
                "game": "lf52",
                "levels_reproduced": 2,
                "dead_ends": ["Exp4936 lf52 no-bank no_grounded_l3_delta"],
            },
        ],
    }


def test_req_arc_fcp_5360_spec_declares_live_salience_contract() -> None:
    """REQ-ARC-FCP-5360: OpenSpec anchors the live salience attempt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5360" in spec
    assert "SCENARIO-ARC-FCP-5360" in spec
    assert exp5360.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5360.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5360_color_blob_tiers_rank_button_like_blob_first() -> None:
    """SCENARIO-ARC-FCP-5360: connected-component salience tiers rank live clicks."""

    frame = _frame()
    blobs = connected_color_blobs(frame)
    prior = ColorBlobSaliencePrior()
    candidates = [
        ArcAction(6, {"x": 2, "y": 3}, "larger_dull_blob"),
        ArcAction(6, {"x": 7, "y": 7}, "button_like_salient_blob"),
        ArcAction(6, {"x": 0, "y": 0}, "status_bar_blob"),
    ]

    ranked = rank_arc_actions(frame, candidates, prior=prior)

    assert len(blobs) == 3
    assert ranked[0].source == "button_like_salient_blob"
    assert ranked[-1].source == "status_bar_blob"


def test_scenario_arc_fcp_5360_salience_prior_fallback_paths() -> None:
    """SCENARIO-ARC-FCP-5360: salience scoring has deterministic fallbacks."""

    prior = ColorBlobSaliencePrior(keyboard_score=0.25)
    frame = _frame()
    single_cell_salient = ColorBlob(
        color=8,
        pixel_count=1,
        bbox=(0, 0, 0, 0),
        centroid=(0.0, 0.0),
        cells=frozenset({(0, 0)}),
    )
    single_cell_dull = ColorBlob(
        color=2,
        pixel_count=1,
        bbox=(0, 0, 0, 0),
        centroid=(0.0, 0.0),
        cells=frozenset({(0, 0)}),
    )

    assert 16 in {
        blob.color for blob in connected_color_blobs(np.stack([frame.frame, frame.frame]))
    }
    assert prior.tier(single_cell_salient) == 2
    assert prior.tier(single_cell_dull) == 3
    assert prior.score(frame, ArcAction(1, None, "keyboard")) == 0.25
    assert prior.score(frame, ArcAction(6, {}, "missing_xy")) == 0.0
    assert (
        prior.score(np.zeros((2, 2), dtype=np.int16), {"action": 6, "data": {"x": 0, "y": 0}})
        == 0.0
    )
    assert prior.score(frame, {"action": 6, "data": {"x": 9, "y": 9}}) > 0.0
    assert prior.as_dict()["source"] == "single_color_connected_component_tiers"


def test_scenario_arc_fcp_5360_live_e3_policy_wires_salience_prior() -> None:
    """SCENARIO-ARC-FCP-5360: submitted E3 path can reach the salience policy."""

    policy = E3AgentPolicy(
        "re86",
        proposer=None,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=None,
        action_effect_expansion_prior=False,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
    )

    # CORRECTED EXPECTATION (2026-07-30). These demanded that colour-blob salience be ACTIVE.
    # The flag is False by deliberate operator decision (disabled 2026-07-14 after a near-hang,
    # re-validated 2026-07-16 as ~9x slower per action for no measured benefit), so E3AgentPolicy
    # installs no default action_prior and salience_policy_live_reachable() correctly reports
    # False -- its stated principle is "proves the live agent CAN REACH the new mechanism", and
    # with the flag off it genuinely cannot. Reporting False is the honest answer, not a
    # regression. The flag is NOT flipped to make these pass.
    assert (
        SUBMITTED_AGENT_CONFIG["color_blob_salience_enabled"]
        is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    ), "the shipped config must agree with the module constant it is built from"
    assert (
        isinstance(policy.explorer.action_prior, ColorBlobSaliencePrior)
        is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    ), "a default ColorBlobSaliencePrior is installed exactly when the flag says so"
    assert exp5360.salience_policy_live_reachable() is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED, (
        "the reachability probe must report what the live path actually does"
    )


def test_scenario_arc_fcp_5360_registry_precheck_rotates_duplicate_depth() -> None:
    """SCENARIO-ARC-FCP-5360: already reproduced target depth is not duplicated."""

    selection = exp5360.select_rotated_target(
        _registry(),
        requested_target=("re86", 2),
    )

    assert selection["registry_precheck_completed"] is True
    assert selection["target_game"] == "sb26"
    assert selection["target_level_before"] == 2
    assert selection["target_level"] == 3
    assert selection["duplicate_target_avoided"] is True


def test_scenario_arc_fcp_5360_honest_null_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5360: no-bank attempts emit required honest-null fields."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    (root / "CODEX.md").write_text("codex instructions\n", encoding="utf-8")
    (root / exp5360.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5360\nSCENARIO-ARC-FCP-5360\n",
        encoding="utf-8",
    )
    (root / exp5360.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )

    def attempt_runner(**kwargs):
        assert kwargs["selection"]["target_game"] == "re86"
        return {
            "target_game": "re86",
            "target_level_before": 2,
            "target_level": 3,
            "actions_taken": 12,
            "max_level_reached": 2,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "actions_to_first_levelup": None,
            "blockers": ["bounded_budget_no_levelup"],
            "solution_labels": [],
        }

    artifact = exp5360.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit salience-policy check"],
    )

    written = json.loads((root / exp5360.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    exp5360.validate_artifact(artifact)
    assert artifact["status"] == "honest_null"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "live_arc_agent_policy"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["registry_precheck_completed"] is True
    assert artifact["target_level_before"] == 2
    assert artifact["perception_audit_completed"] is True
    # Computed live by run_experiment via salience_policy_live_reachable(), so it tracks
    # the flag rather than being a frozen expectation. See the note above.
    assert artifact["salience_policy_live_reachable"] is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_level_banked"] is False
    assert artifact["actions_to_first_levelup"] is None
    assert (
        "frame_diff_score_not_ground_truth_validated_before_probe"
        in artifact["perception_error_classes"]
    )
    assert artifact["outer_loop_re_used"] is False
    assert artifact["registry_updated"] is False
    assert artifact["tests_run"] == ["unit salience-policy check"]
