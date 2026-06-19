"""Tests for Exp 4423 generic ARC first-contact breadth.

Spec refs: REQ-REPORT-4423, SCENARIO-REPORT-4423.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_4423_generic_first_contact_breadth as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_fixture_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "environment_files" / "bp35" / "00000000").mkdir(parents=True, exist_ok=True)
    (root / "environment_files" / "vc33" / "5430563c").mkdir(parents=True, exist_ok=True)
    (root / "environment_files" / "r11l" / "495a7899").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "updated": "2026-06-19",
                "general_gotchas": [{"id": "level_on_frame_not_game"}],
                "games": [
                    {
                        "game": "r11l",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "solver": "registry-r11l",
                    }
                ],
                "reproducible_total_levels": 35,
                "reproducible_total_games": 18,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "results" / "arc3_full_pass_scorecard.json").write_text(
        json.dumps(
            {
                "per_game": [
                    {"game": "bp35", "class": "FAIL_EXPLORATION", "dur_s": 7.0},
                    {"game": "vc33", "class": "SOLVED", "dur_s": 2.5},
                    {"game": "r11l", "class": "SOLVED", "dur_s": 1.0},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _recommendation(game: str) -> dict[str, object]:
    return {
        "target_game": game,
        "recommended": [
            {
                "game": "r11l",
                "similarity": 3.5,
                "solver": "registry-r11l",
                "win_condition": "click-to-template",
                "action_model": "ACTION6 click",
            }
        ],
        "strategy": {"routed_mechanic": "graph_explore", "solver": "arc_graph_explore"},
        "general_gotchas": [{"id": "level_on_frame_not_game"}],
    }


def _success_loop(game: str, root: Path) -> dict[str, object]:
    return {
        "game": game,
        "mode": "standing_arc_loop_graph_explore_no_quota",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reached_level": 1,
        "solution_labels": ['{"action": 4}', '{"action": 5}'],
        "verifier_is_oracle": False,
    }


def _no_advance_loop(game: str, root: Path) -> dict[str, object]:
    return {
        "game": game,
        "status": "needs_per_game_RE",
        "mode": "standing_arc_loop_routing_only",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "transfer_recommendation": [{"game": "r11l"}],
    }


def test_req_report_4423_spec_declares_generic_breadth_contract() -> None:
    """REQ-REPORT-4423: OpenSpec declares the required artifact and routing fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4423" in spec
    assert "SCENARIO-REPORT-4423" in spec
    assert "experiment_4423_generic_first_contact_breadth.json" in spec
    assert "arc_solve_learning.recommend_approach(game)" in spec
    assert "Exp 4421" in spec
    assert "Exp 4422" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4423_candidate_selection_prefers_unregistered_candidates(tmp_path: Path) -> None:
    """REQ-REPORT-4423: candidates are absent from the current registry or fail-exploration."""

    _write_fixture_repo(tmp_path)

    candidates = mod.select_candidate_games(tmp_path)

    assert [candidate.game for candidate in candidates] == ["vc33", "bp35"]
    assert candidates[0].reason == "unseen_not_in_registry"
    assert "FAIL_EXPLORATION" in candidates[1].signals
    assert "r11l" not in {candidate.game for candidate in candidates}


def test_scenario_report_4423_success_routes_before_standing_loop_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4423: recommendation precedes loop and success requires a new reproduced game."""

    _write_fixture_repo(tmp_path)
    call_order: list[str] = []

    def recommend(game: str) -> dict[str, object]:
        call_order.append(f"recommend:{game}")
        return _recommendation(game)

    def loop(game: str, root: Path) -> dict[str, object]:
        call_order.append(f"loop:{game}")
        return _success_loop(game, root)

    artifact = mod.run(
        root=tmp_path,
        target_game="vc33",
        recommend_fn=recommend,
        standing_loop_fn=loop,
        write_registry=False,
        now=lambda: 10.0,
    )

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert result_path.exists()
    assert call_order == ["recommend:vc33", "loop:vc33"]
    assert artifact["honest_verdict"] == "success: generic_first_contact_vc33_L1_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["attempted_games"] == ["vc33"]
    assert {option["id"] for option in artifact["routing_options"]} >= {
        "exp4421_config_rule_unseen",
        "exp4422_glyph_rewrite_pixels",
    }
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(result_path.read_text(encoding="utf-8"))["target_game"] == "vc33"


def test_scenario_report_4423_partial_logs_gap_and_registry_dead_end(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4423: routed no-advance is valid only with gap and registry dead-end."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        target_game="bp35",
        recommend_fn=_recommendation,
        standing_loop_fn=_no_advance_loop,
        write_registry=True,
        now=lambda: 10.0,
    )

    assert (
        artifact["honest_verdict"]
        == "complete: generic_first_contact_bp35_routed_no_new_level_gap_logged"
    )
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == "GAP-4423-BP35-UNSELECTABLE-FIRST-CONTACT"
    assert artifact["dead_ends_recorded"] == ["bp35"]
    assert mod.artifact_schema_errors(artifact) == []

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    bp35 = next(entry for entry in registry["games"] if entry["game"] == "bp35")
    assert bp35["reproducibility"] == "unsolved"
    assert bp35["dead_ends"][0]["gap_id"] == "GAP-4423-BP35-UNSELECTABLE-FIRST-CONTACT"
    assert "registry-r11l" in bp35["dead_ends"][0]["routed_recipe"]["solver"]


def test_req_report_4423_schema_rejects_fabricated_success_and_missing_gap(tmp_path: Path) -> None:
    """REQ-REPORT-4423: success and partial artifacts must carry their evidence."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        target_game="vc33",
        recommend_fn=_recommendation,
        standing_loop_fn=_success_loop,
        write_registry=False,
        now=lambda: 10.0,
    )
    fabricated = {
        **artifact,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_was_new_to_registry": False,
        "routing_options": [],
        "reproducibility_checksum": "bad",
    }
    errors = mod.artifact_schema_errors(fabricated)

    assert "success verdict requires offline_reproduced true" in errors
    assert "success verdict requires reproduced_levels>=1" in errors
    assert "success verdict requires target_was_new_to_registry true" in errors
    assert "routing_options must include exp4421 and exp4422" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors

    complete_no_level = {
        **artifact,
        "honest_verdict": "complete: no gap",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "missing_verifier_gaps": [],
    }
    assert "complete no-new-level verdict requires missing_verifier_gaps" in mod.artifact_schema_errors(
        complete_no_level
    )


def test_req_report_4423_defensive_paths_and_schema_errors(tmp_path: Path) -> None:
    """REQ-REPORT-4423: malformed inputs block or degrade without fabricating progress."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod.load_registry(tmp_path) == {"games": []}
    assert mod._registered_game_names({"games": "bad"}) == set()
    assert mod._environment_games(tmp_path / "no-env-root") == set()

    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "environment_files" / "extra" / "00000000").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text("games: bad\n", encoding="utf-8")
    (tmp_path / "results" / "arc3_full_pass_scorecard.json").write_text(
        json.dumps({"per_game": [None, {"game": "ghost", "class": "FAIL_EXPLORATION"}]}),
        encoding="utf-8",
    )
    candidates = mod.select_candidate_games(tmp_path)
    assert [candidate.game for candidate in candidates] == ["extra"]

    options = mod.routing_options_for("x", {"recommended": ["bad"]})
    assert options[0]["game"] == ""
    assert mod._closest_recipe({}) == {}
    glyph_gap = mod.missing_gap_for(
        "tr87",
        {"recommended": []},
        {"mode": "routing_only"},
        mod.routing_options_for("tr87", {"recommended": []}),
    )
    assert "Exp 4422" in glyph_gap["candidate_design"]

    mod.record_dead_end(tmp_path, "extra", {**glyph_gap, "gap_id": "GAP-4423-EXTRA"})
    repaired = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert next(entry for entry in repaired["games"] if entry["game"] == "extra")["dead_ends"][0][
        "gap_id"
    ] == "GAP-4423-EXTRA"

    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.write_text(
        yaml.safe_dump(
            {
                "games": [
                    {
                        "game": "tr87",
                        "reproducibility": "unsolved",
                        "dead_ends": "bad",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    mod.record_dead_end(tmp_path, "tr87", glyph_gap)
    mod.record_dead_end(tmp_path, "tr87", glyph_gap)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    tr87 = next(entry for entry in registry["games"] if entry["game"] == "tr87")
    assert len(tr87["dead_ends"]) == 1

    blocked = mod._build_artifact(
        root=tmp_path,
        candidates=[],
        target=None,
        recommendation={},
        routing_options=mod.routing_options_for("", {}),
        loop_result={"offline_reproduced": False, "reproduced_levels": 0},
        preconditions_checked={},
        registry_before={"games": []},
        missing_gaps=[],
        dead_ends_recorded=[],
        started_at=2.0,
        ended_at=1.0,
    )
    assert blocked["honest_verdict"] == "blocked: generic_first_contact_no_candidate"

    malformed = {
        "honest_verdict": 4423,
        "offline_reproduced": "false",
        "reproduced_levels": "0",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": "false",
        "reproducibility_checksum": "bad",
        "routing_options": None,
        "target_was_new_to_registry": "yes",
        "attempts": {},
        "preconditions_checked": [],
        "field_principles": [],
    }
    errors = mod.artifact_schema_errors(malformed)
    assert "missing honest_verdict" not in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be bare bool" in errors
    assert "target_was_new_to_registry must be bare bool" in errors
    assert "attempts must be list" in errors
    assert "preconditions_checked must be dict" in errors
    assert "field_principles must be dict" in errors

    wrong_principle = {
        **blocked,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": "wrong"},
    }
    assert "field_principles missing exact honest_verdict" in mod.artifact_schema_errors(wrong_principle)

    with pytest.raises(ValueError, match="missing offline_reproduced"):
        mod.write_artifact(tmp_path, {"honest_verdict": "bad"})

    _write_fixture_repo(tmp_path)
    picked = mod.run(
        root=tmp_path,
        recommend_fn=_recommendation,
        standing_loop_fn=_success_loop,
        write_registry=False,
        now=lambda: 10.0,
    )
    assert picked["target_game"] == "vc33"

    def boom(game: str, root: Path) -> dict[str, object]:
        raise RuntimeError("unit loop failure")

    failed = mod.run(
        root=tmp_path,
        target_game="bp35",
        recommend_fn=_recommendation,
        standing_loop_fn=boom,
        write_registry=True,
        now=lambda: 10.0,
    )
    assert failed["missing_verifier_gaps"][0]["failure_mode"] == "standing_loop_exception_RuntimeError"
