"""Tests for Exp5397 ARC blob-salience live-path attempt.

Spec refs: REQ-ARC-FCP-5397,
SCENARIO-ARC-FCP-5397.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from carnot import experiment_5397_arc_blob_salience_live_path_v491 as exp5397
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior
from carnot.agentic.arc_competition_agent import (
    SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED,
    E3AgentPolicy,
)
from carnot.agentic.arc_graph_explore import rich_action_candidates


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


def _blob_frame() -> SimpleNamespace:
    grid = np.zeros((20, 20), dtype=np.int16)
    grid[0, :] = 16
    grid[2:10, 2:18] = 8
    grid[14:16, 14:16] = 9
    return SimpleNamespace(frame=grid, available_actions=[6])


def _blob_candidates() -> list[ArcAction]:
    return [
        ArcAction(6, {"x": 4, "y": 4}, "large_flat_blob"),
        ArcAction(6, {"x": 14, "y": 14}, "button_like_blob"),
        ArcAction(6, {"x": 3, "y": 0}, "status_bar_blob"),
    ]


def test_req_arc_fcp_5397_spec_declares_required_fields() -> None:
    """REQ-ARC-FCP-5397: OpenSpec anchors the v491 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5397" in spec
    assert "SCENARIO-ARC-FCP-5397" in spec
    assert exp5397.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5397.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5397_blob_prior_emits_status_masked_tiers() -> None:
    """SCENARIO-ARC-FCP-5397: connected components emit auditable salience tiers."""

    prior = ColorBlobSaliencePrior()
    rows = prior.tier_rows(_blob_frame())
    action_rows = prior.action_tier_rows(_blob_frame(), _blob_candidates())

    assert rows[0]["tier"] == 0
    assert rows[0]["button_like"] is True
    assert rows[0]["status_bar"] is False
    assert any(row["tier"] == 4 and row["status_bar"] for row in rows)
    assert rows[-1]["status_bar"] is True
    assert action_rows[0]["source"] == "button_like_blob"
    assert action_rows[0]["tier"] == 0
    assert action_rows[-1]["source"] == "status_bar_blob"
    assert action_rows[-1]["tier"] == 4
    assert prior.as_dict()["salience_tiers_emitted"] is True


def test_scenario_arc_fcp_5397_generation_stage_blob_tiers_beat_click_cap() -> None:
    """SCENARIO-ARC-FCP-5397: tiered points are generated before max_click truncation."""

    frame = _blob_frame()
    prior = ColorBlobSaliencePrior()

    legacy = rich_action_candidates(frame, max_click=1, action_prior=None)
    tiered = rich_action_candidates(frame, max_click=1, action_prior=prior)

    assert legacy[0].data != {"x": 14, "y": 14}
    assert tiered[0].data == {"x": 14, "y": 14}


def test_scenario_arc_fcp_5397_live_e3_policy_logs_salience_tiers() -> None:
    """SCENARIO-ARC-FCP-5397: live E3 generation path emits blob tier diagnostics."""

    # CORRECTED EXPECTATION (2026-07-30). This asserted that the DEFAULT live policy emits tier
    # diagnostics. It does not, and should not: SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED is False by
    # deliberate operator decision (disabled 2026-07-14 after a near-hang, re-validated 2026-07-16
    # as ~9x slower per action for no measured benefit), so no default action_prior is installed
    # and there is nothing to rank or report.
    #
    # The SCENARIO's real claim -- the live generation path emits blob tier diagnostics when the
    # salience prior is wired in -- is preserved and still asserted, by wiring the prior through
    # E3AgentPolicy's own public `action_prior=` parameter. That is a genuine live-path
    # configuration, not a stub: the same slot the flag fills when it is on. The shipped
    # default-off behaviour is then asserted separately, so this test now pins BOTH states instead
    # of silently assuming the one that is currently switched off.
    def _policy(**kwargs):
        return E3AgentPolicy(
            "re86",
            proposer=None,
            value_head=None,
            frame_change_scorer=None,
            candidate_router=None,
            action_effect_expansion_prior=False,
            goal_bias=None,
            goal_candidate_guidance=False,
            active_probe_controller=False,
            **kwargs,
        )

    wired = _policy(action_prior=ColorBlobSaliencePrior())
    candidates = wired.explorer._candidates(_blob_frame())  # noqa: SLF001 - live hook fixture
    diagnostics = wired.explorer.action_salience_diagnostics()

    assert candidates[0]["data"] == {"x": 14, "y": 14}
    assert diagnostics["connected_component_salience_enabled"] is True
    assert diagnostics["salience_tiers_emitted"] is True
    assert diagnostics["generation_stage_action_prioritization"] is True
    assert diagnostics["tier_rows"][0]["tier"] == 0

    # And the shipped default: whatever the flag says, the default policy's diagnostics must agree
    # with it.
    #
    # THE CANDIDATE GENERATION ON THE NEXT LINE IS LOAD-BEARING, not setup noise (corrected
    # 2026-07-30). Without it this assertion was NON-CAUSAL -- the exact "green for the wrong
    # reason" shape this project keeps getting bitten by. `connected_component_salience_enabled` is
    # reported from what the explorer OBSERVED while ranking, so on a policy that has never
    # generated candidates it is False because nothing ran, NOT because the flag is off. Verified
    # directly: a policy with ColorBlobSaliencePrior EXPLICITLY wired also reports False before its
    # first `_candidates()` call. So the pre-correction version of this assertion would have passed
    # with the flag flipped either way, and an earlier comment here claiming it "would have caught
    # the drift" was simply false.
    #
    # Generating candidates first makes it discriminate: False for the default policy, True for the
    # wired one (both confirmed). Coverage of the drift itself is not resting on this line either --
    # a mutation run showed seven other tests catch a flag flip -- but a test that cannot fail is
    # worse than no test, because it reads as protection that is not there.
    default_policy = _policy()
    default_policy.explorer._candidates(_blob_frame())  # noqa: SLF001 - live hook fixture
    default_diagnostics = default_policy.explorer.action_salience_diagnostics()
    assert (
        default_diagnostics["connected_component_salience_enabled"]
        is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    )


def test_scenario_arc_fcp_5397_registry_precheck_selects_re86_l3_or_rotates() -> None:
    """SCENARIO-ARC-FCP-5397: registry precheck avoids duplicate solved levels."""

    selected = exp5397.select_target_after_precheck(_registry(re86_levels=2))
    rotated = exp5397.select_target_after_precheck(_registry(re86_levels=3))

    assert selected["registry_precheck_done"] is True
    assert selected["target_game"] == "re86"
    assert selected["attempted_level"] == 3
    assert selected["duplicate_solve_avoided"] is True
    assert rotated["target_game"] == "sb26"
    assert rotated["duplicate_solve_avoided"] is True


def test_scenario_arc_fcp_5397_artifact_success_and_honest_null_gates() -> None:
    """SCENARIO-ARC-FCP-5397: only reproduced live self-discovery receives credit."""

    selection = exp5397.select_target_after_precheck(_registry())
    diagnostics = exp5397.blob_salience_live_diagnostics()
    no_bank = exp5397.build_artifact(
        selection=selection,
        registry_total_before=69,
        live_diagnostics=diagnostics,
        attempt={
            "offline_reproduced": False,
            "max_level_reached": 2,
            "failure_mode": "bounded_budget_no_levelup",
        },
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5397.validate_artifact(no_bank)
    assert no_bank["status"] == "honest_null"
    assert no_bank["milestone"] == "2026.07.491"
    assert no_bank["target_game"] == "re86"
    assert no_bank["attempted_level"] == 3
    assert no_bank["duplicate_solve_avoided"] is True
    # Computed by blob_salience_live_diagnostics() from the DEFAULT policy, so these
    # track SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED. See the note in the live-path test.
    assert no_bank["connected_component_salience_enabled"] is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    assert no_bank["salience_tiers_emitted"] is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    assert no_bank["per_game_adapter_used"] is False
    assert no_bank["offline_bfs_used"] is False
    assert no_bank["outer_loop_re_used"] is False
    assert no_bank["offline_reproduced"] is False
    assert no_bank["reproduced_levels"] == 0
    assert no_bank["new_level_banked"] is False
    assert no_bank["failure_mode"] == "bounded_budget_no_levelup"
    assert no_bank["honest_verdict"].startswith("honest_null:")

    success = exp5397.build_artifact(
        selection=selection,
        registry_total_before=69,
        live_diagnostics=diagnostics,
        attempt={
            "offline_reproduced": True,
            "max_level_reached": 3,
            "solution_labels": ['{"action":6,"data":{"x":14,"y":14}}'],
        },
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5397.validate_artifact(success)
    assert success["status"] == "complete"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["new_level_banked"] is True
    assert success["failure_mode"] is None
    assert success["honest_verdict"].startswith("complete:")

    bad = dict(success)
    bad["status"] = "complete"
    bad["solve_provenance"] = "outer_loop_re"
    bad["offline_bfs_used"] = True
    bad["reproduced_levels"] = 0
    errors = exp5397.artifact_schema_errors(bad)

    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "offline_bfs_used must be false" in errors
    assert "complete artifact requires reproduced_levels >= 1" in errors
    with pytest.raises(ValueError):
        exp5397.validate_artifact(bad)


def test_scenario_arc_fcp_5397_helper_and_schema_edge_branches(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5397: helper fallbacks stay explicit and deterministic."""

    missing_first = exp5397.select_target_after_precheck(
        _registry(re86_levels=3),
        alternates=("missing", "sb26"),
    )
    no_alternate = exp5397.select_target_after_precheck(
        {"games": [{"game": "re86", "levels_reproduced": 3}]},
        alternates=(),
    )

    assert missing_first["target_game"] == "sb26"
    assert no_alternate["selection_reason"] == "no_alternate_available_no_duplicate_attempted"
    assert exp5397._action_label(6, {"x": 1, "y": 2}) == (  # noqa: SLF001
        '{"action":6,"data":{"x":1,"y":2}}'
    )
    assert (
        exp5397._new_reproduced_levels(  # noqa: SLF001
            attempt={"offline_reproduced": True, "new_reproduced_levels": 2},
            target_level_before=2,
            attempted_level=3,
        )
        == 2
    )
    assert (
        exp5397._new_reproduced_levels(  # noqa: SLF001
            attempt={"offline_reproduced": True, "max_level_reached": 2},
            target_level_before=2,
            attempted_level=3,
        )
        == 0
    )

    diagnostics = exp5397.blob_salience_live_diagnostics()
    default_null = exp5397.build_artifact(
        selection=exp5397.select_target_after_precheck(_registry()),
        registry_total_before=69,
        live_diagnostics=diagnostics,
        attempt={"offline_reproduced": False},
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    default_blocked = exp5397.build_artifact(
        selection=exp5397.select_target_after_precheck(_registry()),
        registry_total_before=69,
        live_diagnostics=diagnostics,
        attempt={"blocked": True},
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    exp5397.validate_artifact(default_null)
    exp5397.validate_artifact(default_blocked)
    assert default_null["failure_mode"] == "bounded_budget_no_levelup"
    assert default_blocked["failure_mode"] == "missing_harness_access"

    invalid = dict(default_null)
    invalid.update(
        {
            "status": "maybe",
            "milestone": "2026.07.490",
            "registry_precheck_done": False,
            "duplicate_solve_avoided": "yes",
            "attempted_level": "3",
            "offline_reproduced": True,
            "new_level_banked": True,
            "failure_mode": "",
            "honest_verdict": "unclear",
        }
    )
    errors = exp5397.artifact_schema_errors(invalid)
    assert "status must be complete, honest_null, or blocked" in errors
    assert "milestone must be 2026.07.491" in errors
    assert "duplicate_solve_avoided must be bare bool" in errors
    assert "registry_precheck_done must be true" in errors
    assert "attempted_level must be bare int" in errors
    assert "non-complete artifact cannot set offline_reproduced true" in errors
    assert "new_level_banked requires complete status" in errors
    assert "honest_null or blocked artifact requires concise failure_mode" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors

    invalid_complete = dict(default_null)
    invalid_complete.update(
        {
            "status": "complete",
            "offline_reproduced": False,
            "new_level_banked": False,
            "reproduced_levels": 1,
            "failure_mode": "should_be_null",
            "honest_verdict": "complete: invalid",
        }
    )
    complete_errors = exp5397.artifact_schema_errors(invalid_complete)
    assert "complete artifact requires offline_reproduced true" in complete_errors
    assert "complete artifact requires new_level_banked true" in complete_errors
    assert "complete artifact requires failure_mode null" in complete_errors

    missing_preconditions = exp5397.run_experiment(
        root=tmp_path,
        offline_arcade_check=lambda: True,
        tests_run=["missing preconditions unit"],
    )
    assert missing_preconditions["status"] == "blocked"
    assert missing_preconditions["live_attempt_count"] == 0

    ready_root = tmp_path / "ready"
    (ready_root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (ready_root / "ops").mkdir()
    (ready_root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (ready_root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (ready_root / exp5397.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5397\n",
        encoding="utf-8",
    )
    (ready_root / exp5397.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    arcade_blocked = exp5397.run_experiment(
        root=ready_root,
        offline_arcade_check=lambda: False,
        tests_run=["arcade blocked unit"],
    )
    assert arcade_blocked["status"] == "blocked"
    assert arcade_blocked["preconditions_checked"]["offline_arcade_available"] is False


def test_scenario_arc_fcp_5397_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5397: runner writes stable JSON with required bare fields."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5397.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5397\nSCENARIO-ARC-FCP-5397\n",
        encoding="utf-8",
    )
    (root / exp5397.REGISTRY_RELATIVE_PATH).write_text(
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
            "new_level_banked": False,
            "actions_taken": 8,
            "max_level_reached": 2,
            "failure_mode": "bounded_budget_no_levelup",
        }

    artifact = exp5397.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5397 blob salience"],
    )
    written = json.loads((root / exp5397.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["status"] == "honest_null"
    assert artifact["registry_precheck_done"] is True
    assert artifact["duplicate_solve_avoided"] is True
    assert artifact["live_agent_policy_modified"] is True
    # Computed by blob_salience_live_diagnostics() from the DEFAULT policy, so these
    # track SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED. See the note in the live-path test.
    assert artifact["connected_component_salience_enabled"] is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    assert artifact["salience_tiers_emitted"] is SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED
    assert artifact["live_attempt_count"] == 1
    assert artifact["tests_run"] == ["unit 5397 blob salience"]
