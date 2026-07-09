"""Tests for Exp5493 ARC trajectory target precheck.

Spec refs: REQ-ARC-FCP-5493,
SCENARIO-ARC-FCP-5493.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5493_arc_trajectory_target_precheck_v498 as exp5493


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry() -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "ka59", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "cn04", "reproducibility": "reproduced", "levels_reproduced": 3},
            {"game": "re86", "reproducibility": "reproduced", "levels_reproduced": 2},
            {
                "game": "dc22",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
                "mechanic_class": "config_toggle_navigation",
                "action_model": "Keyboard movement plus ACTION6 visible toggle clicks.",
                "learned_verifier_checkpoint": "models/arc_verifier_dc22.json",
                "dead_ends": ["Exp4894 dc22 no-bank duplicate_depth"],
            },
            {
                "game": "g50t",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
                "dead_ends": ["Exp4981 g50t no-bank no_grounded_l3_delta"],
            },
        ],
    }


def _manifest() -> dict[str, Any]:
    return {
        "retired_extras": [
            "non-mapping retired row",
            {
                "id": "generation_axis_exploration_signal_retired_exp5154_v473",
                "experiment_scope": (
                    "ARC first-contact candidate-generation exploration-signal tweaks "
                    "after novelty, program-synthesis filter, and energy-as-fitness QD nulls"
                ),
                "blocked_patterns": [
                    "novelty-bonus first-contact rerun",
                    "energy-as-fitness first-contact rerun",
                ],
            }
        ]
    }


def test_req_arc_fcp_5493_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5493: OpenSpec anchors the registry-only artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5493" in spec
    assert "SCENARIO-ARC-FCP-5493" in spec
    assert exp5493.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5493.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5493_selects_dc22_after_exclusions() -> None:
    """SCENARIO-ARC-FCP-5493: recent no-bank and retired lanes are skipped."""

    selection = exp5493.select_trajectory_target(_registry(), _manifest())

    assert selection["blocked"] is False
    assert selection["selected_game"] == "dc22"
    assert selection["selected_target_level"] == 3
    assert selection["prior_levels_reproduced"] == 2
    assert selection["duplicate_solve_avoided"] is True
    assert selection["candidate_audit"]["sb26:L3"]["decision"] == "rejected_recent_no_bank"
    assert selection["candidate_audit"]["bp35:L3"]["decision"] == "rejected_recent_no_bank"
    assert selection["candidate_audit"]["ka59:L2"]["decision"] == "rejected_recent_no_bank"
    assert selection["candidate_audit"]["cn04:L4"]["decision"] == "rejected_recent_no_bank"
    assert selection["candidate_audit"]["re86:L3"]["decision"] == "rejected_recent_no_bank"
    assert selection["candidate_audit"]["dc22:L3"]["decision"] == "selected"
    assert "LiveCoExLandmarkFrontierGenerator" in selection["proposed_live_mechanism"]
    assert "runtime_action_effect_observations" in selection["trajectory_induction_preconditions"]


def test_scenario_arc_fcp_5493_rejects_duplicates_and_retired_scopes() -> None:
    """SCENARIO-ARC-FCP-5493: duplicate targets and retired mechanisms fail closed."""

    candidates = [
        exp5493.TrajectoryCandidate(
            game="zz99",
            target_level=1,
            proposed_live_mechanism="LiveCoExLandmarkFrontierGenerator trajectory induction",
            trajectory_induction_preconditions=("runtime_action_effect_observations",),
            priority=0,
            live_mechanism_hooks=("hook",),
        ),
        exp5493.TrajectoryCandidate(
            game="dc22",
            target_level=2,
            proposed_live_mechanism="LiveCoExLandmarkFrontierGenerator trajectory induction",
            trajectory_induction_preconditions=("runtime_action_effect_observations",),
            priority=1,
        ),
        exp5493.TrajectoryCandidate(
            game="g50t",
            target_level=3,
            proposed_live_mechanism="novelty-only first-contact exploration signal",
            trajectory_induction_preconditions=("runtime_action_effect_observations",),
            priority=2,
        ),
        exp5493.TrajectoryCandidate(
            game="dc22",
            target_level=3,
            proposed_live_mechanism="LiveCoExLandmarkFrontierGenerator trajectory induction",
            trajectory_induction_preconditions=("runtime_action_effect_observations",),
            priority=3,
        ),
    ]

    selection = exp5493.select_trajectory_target(
        _registry(),
        _manifest(),
        candidates=candidates,
        recent_no_bank_targets=(),
    )

    assert selection["blocked"] is True
    assert selection["selected_game"] == ""
    assert selection["selected_target_level"] == 0
    assert selection["candidate_audit"]["zz99:L1"]["decision"] == (
        "rejected_missing_reproduced_registry_row"
    )
    assert selection["candidate_audit"]["dc22:L2"]["decision"] == "rejected_duplicate"
    assert selection["candidate_audit"]["g50t:L3"]["decision"] == "rejected_retired_scope"
    assert selection["candidate_audit"]["dc22:L3"]["decision"] == "rejected_missing_live_trajectory_hooks"
    assert selection["blocker"] == "no_eligible_trajectory_target"
    assert exp5493._exp5494_command({"selected_game": "", "selected_target_level": 0}) == ""  # noqa: SLF001
    assert exp5493._as_int("not-an-int", 4) == 4  # noqa: SLF001


def test_scenario_arc_fcp_5493_artifact_schema_ready_and_blocked() -> None:
    """SCENARIO-ARC-FCP-5493: artifacts expose required ready and no-target fields."""

    ready = exp5493.build_artifact(
        selection=exp5493.select_trajectory_target(_registry(), _manifest()),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    blocked = exp5493.build_artifact(
        selection={
            "blocked": True,
            "blocker": "no_eligible_trajectory_target",
            "selected_game": "",
            "selected_target_level": 0,
            "prior_levels_reproduced": 0,
            "proposed_live_mechanism": "",
            "trajectory_induction_preconditions": [],
            "duplicate_solve_avoided": True,
            "candidate_audit": {},
            "levels_reproduced_by_candidate_game": {},
        },
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5493.validate_artifact(ready)
    exp5493.validate_artifact(blocked)
    assert ready["registry_path"] == exp5493.REGISTRY_RELATIVE_PATH
    assert ready["excluded_recent_no_bank_targets"] == list(exp5493.RECENT_NO_BANK_TARGETS)
    assert ready["duplicate_solve_avoided"] is True
    assert ready["selected_game"] == "dc22"
    assert ready["selected_target_level"] == 3
    assert ready["prior_levels_reproduced"] == 2
    assert ready["offline_bfs_used"] is False
    assert ready["per_game_adapter_used"] is False
    assert ready["arc_trajectory_precheck_ready"] is True
    assert ready["inference_substrate"] == "registry_precheck_no_solve"
    assert ready["honest_verdict"].startswith("complete:")
    assert blocked["selected_game"] == ""
    assert blocked["selected_target_level"] == 0
    assert blocked["arc_trajectory_precheck_ready"] is False
    assert blocked["honest_verdict"].startswith("blocked:")


def test_scenario_arc_fcp_5493_schema_rejects_bad_claims() -> None:
    """SCENARIO-ARC-FCP-5493: schema rejects prohibited paths and non-bare fields."""

    artifact = exp5493.build_artifact(
        selection=exp5493.select_trajectory_target(_registry(), _manifest()),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "registry_path": "ops/wrong.yaml",
        "excluded_recent_no_bank_targets": ["sb26:L3"],
        "duplicate_solve_avoided": "true",
        "selected_game": 7,
        "selected_target_level": "3",
        "prior_levels_reproduced": "2",
        "proposed_live_mechanism": "energy-as-fitness quality-diversity rerun",
        "trajectory_induction_preconditions": "runtime",
        "offline_bfs_used": True,
        "per_game_adapter_used": True,
        "arc_trajectory_precheck_ready": "true",
        "inference_substrate": "arc_live_agent_self_discovery",
        "honest_verdict": "complete: solved dc22 L3",
    }

    errors = exp5493.artifact_schema_errors(invalid)
    list_errors = exp5493.artifact_schema_errors(
        {**artifact, "excluded_recent_no_bank_targets": "sb26:L3"}
    )
    type_errors = exp5493.artifact_schema_errors(
        {**artifact, "proposed_live_mechanism": 7}
    )
    ready_errors = exp5493.artifact_schema_errors(
        {
            **artifact,
            "selected_game": "",
            "selected_target_level": 2,
            "prior_levels_reproduced": 2,
            "trajectory_induction_preconditions": [],
            "arc_trajectory_precheck_ready": True,
            "honest_verdict": "pending",
        }
    )

    assert "registry_path must be ops/arc_solve_registry.yaml" in errors
    assert "excluded_recent_no_bank_targets missing bp35:L3" in errors
    assert "excluded_recent_no_bank_targets must be a list" in list_errors
    assert "duplicate_solve_avoided must be bare bool" in errors
    assert "selected_game must be a string" in errors
    assert "selected_target_level must be bare int" in errors
    assert "prior_levels_reproduced must be bare int" in errors
    assert "proposed_live_mechanism must be a string" in type_errors
    assert "proposed_live_mechanism must not match retired exploration-signal scope" in errors
    assert "trajectory_induction_preconditions must be a list" in errors
    assert "offline_bfs_used must be false" in errors
    assert "per_game_adapter_used must be false" in errors
    assert "arc_trajectory_precheck_ready must be bare bool" in errors
    assert "arc_trajectory_precheck_ready requires selected_game" in ready_errors
    assert (
        "arc_trajectory_precheck_ready requires selected_target_level > prior_levels_reproduced"
        in ready_errors
    )
    assert "arc_trajectory_precheck_ready requires trajectory_induction_preconditions" in ready_errors
    assert "inference_substrate must be registry_precheck_no_solve" in errors
    assert "honest_verdict must start with complete: or blocked:" in ready_errors
    assert "honest_verdict must not claim a solve" in errors
    with pytest.raises(ValueError):
        exp5493.validate_artifact(invalid)


def test_scenario_arc_fcp_5493_run_experiment_writes_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5493: runner writes the required deliverable JSON."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5493.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5493\nSCENARIO-ARC-FCP-5493\n",
        encoding="utf-8",
    )
    (root / exp5493.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (root / exp5493.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        yaml.safe_dump(_manifest()),
        encoding="utf-8",
    )
    (root / exp5493.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "ARC standing floor; Exp5154 retired novelty/curiosity and archive granularity reruns.",
        encoding="utf-8",
    )
    (root / exp5493.EXP5479_RELATIVE_PATH).write_text(
        json.dumps({"selected_game": "lf52", "selected_target_level": 3}),
        encoding="utf-8",
    )
    (root / exp5493.EXP5480_RELATIVE_PATH).write_text(
        json.dumps({"game": "sb26", "target_level": 3, "new_level_banked": False}),
        encoding="utf-8",
    )

    artifact = exp5493.run_experiment(root=root, tests_run=["unit 5493"])
    written = json.loads((root / exp5493.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["selected_game"] == "dc22"
    assert artifact["selected_target_level"] == 3
    assert artifact["prior_levels_reproduced"] == 2
    assert artifact["arc_trajectory_precheck_ready"] is True
    assert artifact["inference_substrate"] == "registry_precheck_no_solve"
    assert artifact["tests_run"] == ["unit 5493"]


def test_scenario_arc_fcp_5493_missing_registry_defaults(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5493: missing registry loads as an empty metric surface."""

    assert exp5493.load_registry(tmp_path) == {"reproducible_total_levels": 0, "games": []}
