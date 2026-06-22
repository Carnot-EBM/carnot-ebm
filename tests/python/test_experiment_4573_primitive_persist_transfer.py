"""Tests for Exp 4573 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4573, SCENARIO-ARC-WMTE-4573.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from carnot import experiment_4573_primitive_persist_transfer as exp4573
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _row(
    game: str,
    state_key: str,
    action_id: int,
    *,
    changed: bool,
    level_progress: bool = False,
    x: int | None = None,
    y: int | None = None,
) -> dict[str, object]:
    return {
        "game": game,
        "env": game,
        "state_key": state_key,
        "action_id": action_id,
        "x": x,
        "y": y,
        "changed": changed,
        "level_progress": 1.0 if level_progress else 0.0,
    }


def test_req_arc_wmte_4573_spec_declares_action_effect_transfer_contract() -> None:
    """REQ-ARC-WMTE-4573: OpenSpec declares the persisted memory primitive."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4573",
        "SCENARIO-ARC-WMTE-4573",
        "PersistentAEM",
        exp4573.PRIMITIVE_OPERATOR,
        exp4573.RESULT_RELATIVE_PATH,
        exp4573.PRIMITIVE_GOTCHA_ID,
        "a CNN predictor forward pass is NOT live_llm_inference",
    ):
        assert marker in spec
    for field, principle in exp4573.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4573_solver_kit_memory_ranks_effective_action() -> None:
    """REQ-ARC-WMTE-4573: PersistentAEM ranks candidates by cross-game effect."""

    memory = kit.PersistentAEM.from_effect_rows(
        [
            _row("train", "s1", 1, changed=False),
            _row("train", "s2", 6, changed=True, level_progress=True, x=48, y=48),
            _row("heldout", "s3", 1, changed=True, level_progress=True),
            SimpleNamespace(env="obj", action=5, changed=True, level_progress="bad"),
            {"game": "bad", "action_id": "bad", "changed": True},
            {"game": "bad", "changed": True},
        ],
        exclude_games=("heldout",),
    )

    result = kit.persistent_action_effect_memory_operator(
        [
            {"candidate_id": "first_noop", "action_id": 1, "reaches_levelup": False},
            {
                "candidate_id": "second_levelup",
                "action_id": 6,
                "data": {"x": 49, "y": 49},
                "reaches_levelup": True,
            },
        ],
        memory=memory,
    )

    assert result["operator"] == exp4573.PRIMITIVE_OPERATOR
    assert result["memory"]["excluded_games"] == ["heldout"]
    assert [row["candidate_id"] for row in result["ranked_candidates"]] == [
        "second_levelup",
        "first_noop",
    ]
    assert result["actions_to_first_levelup_before"] == 2
    assert result["actions_to_first_levelup_after"] == 1
    assert result["actions_reduced"] == 1
    assert result["value_added"] is True

    assert memory.candidate_score({"action": 5}) > 0.0
    assert memory.candidate_score({"action_id": "bad"}) == 0.0
    assert memory.candidate_score({"action_id": 6, "data": {"x": "bad", "y": 1}}) > 0.0
    assert memory._ratio("missing") == 0.0  # noqa: SLF001
    assert kit.PersistentAEM({"empty": {"total": 0.0}})._ratio("empty") == 0.0  # noqa: SLF001

    fallback_targets = kit.persistent_action_effect_memory_operator(
        [
            {"candidate_id": "goal", "action_id": 1, "reaches_goal": True},
            {"candidate_id": "level", "action_id": 2, "level_progress": 1.0},
            {"candidate_id": "none", "action_id": 3},
        ],
        memory=memory,
    )
    assert fallback_targets["actions_to_first_levelup_before"] == 1

    no_target = kit.persistent_action_effect_memory_operator(
        [{"candidate_id": "none", "action_id": 3}],
        memory=memory,
    )
    assert no_target["actions_to_first_levelup_before"] is None
    assert no_target["value_added"] is False


def test_req_arc_wmte_4573_routing_and_registry_surface_memory_primitive() -> None:
    """REQ-ARC-WMTE-4573: routing and registry expose the reusable AEM operator."""

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert exp4573.PRIMITIVE_OPERATOR in operators

    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert exp4573.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("tu93")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert exp4573.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == exp4573.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == exp4573.PRIMITIVE_OPERATOR
    assert "leave-one-game" in gotchas[0]["note"]


def test_req_arc_wmte_4573_selects_a1_memory_over_a2_generation_null() -> None:
    """REQ-ARC-WMTE-4573: A1 wins as best-characterized primitive-as-built."""

    decision = exp4573.select_primitive_from_upstreams(
        a1_artifact={
            "honest_verdict": "complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened",
            "actions_delta": 0.0,
            "positive_control_passed": True,
            "positive_control": {"actions_reduced": True},
        },
        a2_artifact={
            "honest_verdict": "complete: verifier_guided_expansion_no_value_honest_null_generation_gap_sharpened",
            "winner_generated": {"generated_count": 0},
        },
    )

    assert decision["source"] == "A1_action_effect_predictor"
    assert decision["operator"] == exp4573.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == exp4573.PRIMITIVE_GOTCHA_ID
    assert decision["persisted_as_best_characterized_null"] is True
    assert decision["positive_control_passed"] is True


def test_req_arc_wmte_4573_leave_one_game_transfer_reports_actions_reduced() -> None:
    """REQ-ARC-WMTE-4573: transfer measurement excludes the target game from memory."""

    rows = [
        _row("train_a", "a1", 1, changed=False),
        _row("train_a", "a2", 6, changed=True, level_progress=True, x=50, y=50),
        _row("zz99", "target_state", 1, changed=False),
        _row("zz99", "target_state", 6, changed=True, level_progress=True, x=50, y=50),
        _row("zz99", "other_state", 1, changed=True, level_progress=True),
    ]

    result = exp4573.measure_action_effect_memory_transfer_game("zz99", effect_rows=rows)

    assert result["game"] == "zz99"
    assert result["excluded_from_memory"] is True
    assert result["value_added"] is True
    assert result["transfer_value"]["actions_to_first_levelup_baseline"] == 2.0
    assert result["transfer_value"]["actions_to_first_levelup_with_memory"] == 1.0
    assert result["transfer_value"]["actions_reduced"] == 1.0
    assert result["transfer_value"]["representation_transfer"] is True


def test_scenario_arc_wmte_4573_artifact_schema_records_success_and_null(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4573: artifact schema records value-add or honest null."""

    decision = {
        "source": "A1_action_effect_predictor",
        "operator": exp4573.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": exp4573.PRIMITIVE_GOTCHA_ID,
        "inference_substrate": exp4573.INFERENCE_SUBSTRATE,
    }
    success = exp4573.build_artifact(
        upstream_decision=decision,
        preconditions_checked={"ok": True, "offline_arcade_import_smoke": True},
        transfer_results=[
            {
                "game": "zz99",
                "value_added": True,
                "transfer_value": {
                    "actions_reduced": 1.0,
                    "representation_transfer": True,
                    "winner_generated": False,
                },
                "offline_reproduced_new_level": False,
                "dead_end": "",
            },
            {
                "game": "yy88",
                "value_added": False,
                "transfer_value": {
                    "actions_reduced": 0.0,
                    "representation_transfer": True,
                    "winner_generated": False,
                },
                "offline_reproduced_new_level": False,
                "dead_end": "candidate generation remained the bottleneck",
            },
        ],
        registry_updated=True,
        random_seed=4573,
        duration_s=0.0,
    )

    assert success["honest_verdict"] == "success: primitive_persisted_transfer_zz99_value_added"
    assert success["offline_reproduced"] is False
    assert success["new_levels_banked"] == 0
    assert exp4573.artifact_schema_errors(success) == []

    out = exp4573.write_artifact(success, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == success

    null = exp4573.build_artifact(
        upstream_decision=decision,
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "zz99",
                "value_added": False,
                "transfer_value": {"actions_reduced": 0.0, "representation_transfer": False},
                "offline_reproduced_new_level": False,
                "dead_end": "no target candidate was generated",
            },
            {
                "game": "yy88",
                "value_added": False,
                "transfer_value": {"actions_reduced": 0.0, "representation_transfer": False},
                "offline_reproduced_new_level": False,
                "dead_end": "no target candidate was generated",
            },
        ],
        registry_updated=True,
        random_seed=4573,
        duration_s=0.0,
    )

    assert null["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert null["transfer_dead_ends"]["zz99"] == "no target candidate was generated"
    assert exp4573.artifact_schema_errors(null) == []

    errors = exp4573.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert f"primitive_persisted must name {exp4573.PRIMITIVE_OPERATOR}" in errors
    assert "transfer_games must contain at least two games" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors


def test_req_arc_wmte_4573_defensive_branches_are_covered(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4573: defensive null paths remain deterministic."""

    assert exp4573._as_float("bad") == 0.0  # noqa: SLF001
    assert exp4573._as_int("bad") == 0  # noqa: SLF001
    assert exp4573._frame_delta_fraction(np.zeros((1, 1)), np.zeros((2, 2))) == 1.0  # noqa: SLF001
    assert exp4573._frame_delta_fraction(np.zeros((0,)), np.zeros((0,))) == 0.0  # noqa: SLF001

    generated = exp4573.select_primitive_from_upstreams(
        a1_artifact={"actions_delta": 0.0, "positive_control_passed": False},
        a2_artifact={"winner_generated": {"generated_count": 1}},
    )
    assert "A2 generated winners" in generated["selection_rationale"]
    neither = exp4573.select_primitive_from_upstreams(a1_artifact={}, a2_artifact={})
    assert "Both A1/A2" in neither["selection_rationale"]

    obj = SimpleNamespace(
        env="objgame",
        state_key="objstate",
        action=6,
        x="bad",
        y=1,
        changed=None,
        level_progress=0.0,
        frame_delta=1.0,
    )
    assert exp4573._row_game(obj) == "objgame"  # noqa: SLF001
    assert exp4573._row_state_key(obj) == "objstate"  # noqa: SLF001
    assert exp4573._row_action_id({"action": 5}) == 5  # noqa: SLF001
    assert exp4573._row_action_id({"action_id": "bad"}) is None  # noqa: SLF001
    assert exp4573._row_xy({"x": "bad", "y": 1}) is None  # noqa: SLF001
    assert exp4573._row_effective_target(obj) is True  # noqa: SLF001

    no_rows = exp4573.measure_action_effect_memory_transfer_game("none", effect_rows=[])
    assert no_rows["dead_end"].startswith("no cached action-effect rows")
    no_trainable = exp4573.measure_action_effect_memory_transfer_game(
        "bad",
        effect_rows=[
            {"game": "bad", "state_key": "", "action_id": 1, "changed": True},
            _row("bad", "s", 6, changed=True),
        ],
    )
    assert no_trainable["dead_end"].startswith("target rows were present")
    no_target_group = exp4573.measure_action_effect_memory_transfer_game(
        "noop",
        effect_rows=[
            _row("train", "t", 6, changed=True, x=1, y=1),
            _row("noop", "s", 1, changed=False),
            _row("noop", "s", 2, changed=False),
        ],
    )
    assert no_target_group["dead_end"].startswith("cached rows did not contain")
    no_value = exp4573.measure_action_effect_memory_transfer_game(
        "flat",
        effect_rows=[
            _row("train", "t", 1, changed=True),
            _row("train", "u", 2, changed=False),
            _row("flat", "s", 1, changed=True),
            _row("flat", "s", 2, changed=False),
        ],
    )
    assert no_value["dead_end"].startswith("PersistentAEM transferred")

    blocked = exp4573.build_artifact(
        upstream_decision={
            "source": "A1_action_effect_predictor",
            "operator": exp4573.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": exp4573.PRIMITIVE_GOTCHA_ID,
            "inference_substrate": exp4573.INFERENCE_SUBSTRATE,
        },
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=4573,
        duration_s=None,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert exp4573.artifact_schema_errors(blocked) == []

    bad_primitive = dict(blocked)
    bad_primitive["primitive_persisted"] = {
        "operator": exp4573.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": "wrong",
    }
    bad_primitive["reproducibility_checksum"] = exp4573.payload_checksum(bad_primitive)
    assert "primitive_persisted must name the 4573 registry general_gotcha" in (
        exp4573.artifact_schema_errors(bad_primitive)
    )

    bad_success = dict(blocked)
    bad_success["honest_verdict"] = "success: primitive_persisted_transfer_x_value_added"
    bad_success["transfer_games"] = ["x", "y"]
    bad_success["transfer_value_per_game"] = {"x": {"value_added": False}}
    bad_success["offline_reproduced"] = True
    bad_success["new_levels_banked"] = 0
    bad_success["reproducibility_checksum"] = exp4573.payload_checksum(bad_success)
    success_errors = exp4573.artifact_schema_errors(bad_success)
    assert "success requires at least one transfer value_added=true" in success_errors
    assert "offline_reproduced=true requires at least one new level banked" in success_errors

    tampered = dict(blocked)
    tampered["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in exp4573.artifact_schema_errors(
        tampered
    )

    with pytest.raises(ValueError, match="missing required field"):
        exp4573.write_artifact({}, root=tmp_path)
