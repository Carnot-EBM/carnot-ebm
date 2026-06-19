"""Tests for Exp 4436 deepen + primitive consolidation.

Spec refs: REQ-REPORT-4436, SCENARIO-REPORT-4436.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot import experiment_4436_deepen_plus_primitive_consolidation as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def test_req_report_4436_spec_declares_gate_and_operator_contract() -> None:
    """REQ-REPORT-4436: OpenSpec names the artifact gate and generic operators."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-4436",
        "SCENARIO-REPORT-4436",
        "experiment_4436_deepen_plus_primitive_consolidation.json",
        "greedy glyph/rewrite matcher",
        "config-rule grounding helper",
        "graph-explore A* priority helper",
        "object-centric digest helper",
        "active-data collection planner",
        "no_regression",
    ):
        assert marker in spec


def test_req_report_4436_glyph_rewrite_operator_covers_tr87_patterns() -> None:
    """REQ-REPORT-4436: the tr87 rewrite matcher is a generic solver-kit operator."""

    rules = [
        (("A4",), ("B3",)),
        (("B3",), ("C1", "C5", "C1")),
        (("C3", "C3"), ("A6", "A1")),
    ]

    assert kit.greedy_rewrite(("A4",), rules) == ("B3",)
    assert kit.greedy_rewrite(("B3",), rules) == ("C1", "C5", "C1")
    assert kit.greedy_rewrite(("C3", "C3"), rules) == ("A6", "A1")
    assert kit.greedy_rewrite(("Z9",), rules) is None

    two_pass = kit.greedy_rewrite(("A4",), rules, passes=2)
    assert two_pass == ("C1", "C5", "C1")
    assert kit.sequence_cyclic_distance([1, 7, 3], [7, 1, 3], modulus=7) == 2.0


def test_req_report_4436_generic_config_digest_active_data_helpers() -> None:
    """REQ-REPORT-4436: config grounding, object digest, and active-data helpers compose."""

    grounded = kit.ground_marker_coverage_rule(
        controlled_markers=[(9, 33), (30, 9)],
        target_markers=[(9, 51), (51, 9)],
        step=3,
        horizontal_label="h_extend",
        vertical_label="v_extend",
    )
    assert grounded["solution"] == ["h_extend"] * 7 + ["v_extend"] * 6
    assert grounded["predicate_satisfied"] is True

    grid = np.array(
        [
            [0, 0, 0, 2],
            [0, 1, 1, 2],
            [0, 0, 1, 0],
            [3, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    digest = kit.object_centric_digest(grid)
    assert digest["shape"] == [4, 4]
    assert digest["background_color"] == 0
    assert digest["component_count"] == 3
    assert digest["components"][0]["area"] == 3

    plan = kit.active_data_collection_plan(
        action_labels=["left", "right"],
        object_signatures=[c["signature"] for c in digest["components"]],
        max_cases_per_action=2,
    )
    assert [row["action"] for row in plan] == ["left", "left", "right", "right"]
    assert {row["selection_policy"] for row in plan} == {"balanced_action_object_coverage"}


def test_req_report_4436_operator_registry_and_selector_are_standing_loop_ready() -> None:
    """SCENARIO-REPORT-4436: the standing loop can select consolidated operators."""

    operators = {row.operator: row for row in kit.primitive_operator_registry()}

    assert set(operators) >= {
        "glyph_rewrite_matcher",
        "config_rule_grounding",
        "graph_astar_action_cost",
        "object_centric_digest",
        "active_data_collection",
    }
    assert operators["glyph_rewrite_matcher"].derived_from_games == ("tr87",)
    assert "s5i5" in operators["config_rule_grounding"].derived_from_games

    selected = kit.select_primitive_operators(mechanic_class="config_substitution")
    assert [row.operator for row in selected][:2] == [
        "glyph_rewrite_matcher",
        "graph_astar_action_cost",
    ]


def test_req_report_4436_artifact_schema_requires_deepen_and_no_regression(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4436: artifacts count only deepened and non-regressed reproductions."""

    artifact = mod.build_artifact(
        deepened_game="tu93",
        prior_reproduced_levels=4,
        target_level=5,
        deepened_reproduction={
            "game": "tu93",
            "claimed_level": 5,
            "reached_level": 5,
            "reproduced": True,
        },
        prior_reproductions=[
            {"game": "tr87", "claimed_level": 6, "reached_level": 7, "reproduced": True},
            {"game": "s5i5", "claimed_level": 1, "reached_level": 1, "reproduced": True},
        ],
        primitives=kit.primitive_operator_registry(),
        preconditions_checked={"offline_env_files_present": True},
    )

    assert artifact["honest_verdict"] == "success: tu93_L5_deepened_primitives_consolidated"
    assert artifact["reproduced_levels"] == 5
    assert artifact["offline_reproduced"] is True
    assert artifact["no_regression"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert mod.artifact_schema_errors(artifact) == []

    out = mod.write_artifact(tmp_path, artifact)
    assert json.loads(out.read_text(encoding="utf-8"))["primitives_consolidated"][0]["operator"]


def test_req_report_4436_tu93_solution_labels_append_replayed_l5_suffix() -> None:
    """SCENARIO-REPORT-4436: tu93 deepening is rooted in the reproduced L4 prefix."""

    labels = mod.deepened_solution_labels(REPO)

    assert len(labels) == 93
    assert labels[-len(mod.TU93_L5_SUFFIX_ACTIONS) :] == [
        json.dumps({"action": action}) for action in mod.TU93_L5_SUFFIX_ACTIONS
    ]


def test_req_report_4436_artifact_utility_branches(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-4436: schema/load helpers reject bad inputs deterministically."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._load_json(list_json) == {}
    registry_path = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry_path.parent.mkdir()
    registry_path.write_text("[]\n", encoding="utf-8")
    assert mod._load_registry(tmp_path) == {"games": []}
    registry_path.write_text("games: [\n", encoding="utf-8")
    assert mod._load_registry(tmp_path) == {"games": []}
    assert mod._as_int("not-int") == 0

    assert mod.primitives_as_rows([{"operator": "x", "derived_from_games": ["g"]}, "raw"]) == [
        {"operator": "x", "derived_from_games": ["g"]},
        {"operator": "raw", "derived_from_games": []},
    ]

    env_dir = tmp_path / "environment_files"
    env_dir.mkdir()
    (env_dir / "sentinel").write_text("x", encoding="utf-8")
    registry_path.write_text("games:\n- game: tu93\n  levels_reproduced: '4'\n", encoding="utf-8")
    assert mod.preconditions(tmp_path)["offline_env_files_present"] is True
    assert mod.prior_best_level(tmp_path, "tu93") == 4
    assert mod.prior_best_level(tmp_path, "missing") == 0

    blocked = mod.build_artifact(
        deepened_game="tu93",
        prior_reproduced_levels=4,
        target_level=5,
        deepened_reproduction={"game": "tu93", "reached_level": 4, "reproduced": False},
        prior_reproductions=[],
        primitives=[],
        preconditions_checked={"offline_env_files_present": False},
    )
    assert blocked["honest_verdict"] == "blocked_offline_env_files_missing"

    regressed = mod.build_artifact(
        deepened_game="tu93",
        prior_reproduced_levels=4,
        target_level=5,
        deepened_reproduction={"game": "tu93", "reached_level": 5, "reproduced": True},
        prior_reproductions=[{"game": "x", "claimed_level": 1, "reached_level": 0, "reproduced": False}],
        primitives=kit.primitive_operator_registry(),
        preconditions_checked={"offline_env_files_present": True},
    )
    assert regressed["honest_verdict"] == "complete: tu93_L5_deepened_but_regression_detected"

    no_deeper = mod.build_artifact(
        deepened_game="tu93",
        prior_reproduced_levels=4,
        target_level=5,
        deepened_reproduction={"game": "tu93", "reached_level": 4, "reproduced": True},
        prior_reproductions=[],
        primitives=kit.primitive_operator_registry(),
        preconditions_checked={"offline_env_files_present": True},
    )
    assert no_deeper["honest_verdict"] == "complete: tu93_no_deeper_level_primitives_consolidated"

    invalid = {
        "honest_verdict": "not-terminal",
        "reproduced_levels": 0,
        "offline_reproduced": "false",
        "primitives_consolidated": [],
        "no_regression": "false",
        "random_seed": 4436,
        "reproducibility_checksum": "0" * 64,
        "verifier_is_oracle": True,
    }
    errors = mod.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "no_regression must be bare bool" in errors
    assert "primitives_consolidated must be non-empty list" in errors

    missing_errors = mod.artifact_schema_errors({})
    assert "missing honest_verdict" in missing_errors
    with np.testing.assert_raises(ValueError):
        mod.write_artifact(tmp_path, blocked)

    import carnot.agentic.arc_game_adapters as adapters

    monkeypatch.setattr(adapters, "get_adapter", lambda _game: None)
    missing = mod.reproduce_deepened_tu93(tmp_path, claimed_level=5)
    assert missing["mode"] == "missing_tu93_adapter_or_solution"


def test_req_report_4436_tu93_deepened_path_reproduces_level_5() -> None:
    """SCENARIO-REPORT-4436: tu93 L5 is confirmed by the offline reproduction gate."""

    result = mod.reproduce_deepened_tu93(REPO, claimed_level=5)

    assert result["reproduced"] is True
    assert result["reached_level"] == 5
    assert result["solution_action_count"] == 93
    assert result["deepened_suffix_action_count"] == len(mod.TU93_L5_SUFFIX_ACTIONS)


def test_req_report_4436_schema_rejects_fabricated_success() -> None:
    """REQ-REPORT-4436: malformed or ungated success artifacts fail validation."""

    artifact = {
        "honest_verdict": "success: tr87_L7_deepened_primitives_consolidated",
        "reproduced_levels": "7",
        "offline_reproduced": False,
        "primitives_consolidated": [{"operator": "glyph_rewrite_matcher"}],
        "no_regression": False,
        "random_seed": "4436",
        "reproducibility_checksum": "x",
        "verifier_is_oracle": False,
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "reproduced_levels must be bare int" in errors
    assert "success verdict requires offline_reproduced true" in errors
    assert "success verdict requires no_regression true" in errors
    assert "primitives_consolidated rows require derived_from_games" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "verifier_is_oracle must be true" in errors
