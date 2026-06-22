"""Tests for Exp 4584 primitive persistence and transfer.

Spec refs: REQ-CAPSTONE-4584, SCENARIO-CAPSTONE-4584,
SCENARIO-CAPSTONE-4584-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4584_primitive_persist_transfer as mod
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _action(x: int, y: int) -> dict[str, Any]:
    return {"action": 6, "data": {"x": x, "y": y}}


def _target(expected: Mapping[str, Any]):
    def target(actions: list[dict[str, Any]], _game: str, _mode: str) -> bool:
        return bool(actions and actions[0] == expected)

    return target


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_capstone_4584_spec_declares_primitive_transfer_contract() -> None:
    """REQ-CAPSTONE-4584: OpenSpec declares the persisted primitive contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4584",
        "SCENARIO-CAPSTONE-4584",
        "SCENARIO-CAPSTONE-4584-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4584_solver_kit_env_adaptive_operator_recovers_drift() -> None:
    """REQ-CAPSTONE-4584: env-adaptive resolve beats frozen replay under drift."""

    result = kit.env_adaptive_resolve_operator(
        ["h_extend"],
        game="s5i5",
        frozen_resolver={"h_extend": _action(47, 21)},
        adaptive_resolver={"h_extend": _action(50, 23)},
        target_predicate=_target(_action(50, 23)),
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["game"] == "s5i5"
    assert result["frozen_reached"] is False
    assert result["adaptive_reached"] is True
    assert result["drift_recovered"] is True
    assert result["value_added"] is True
    assert result["adaptive_actions"] == [_action(50, 23)]

    missing = kit.env_adaptive_resolve_operator(
        ["unknown"],
        game="s5i5",
        frozen_resolver={},
        adaptive_resolver={},
        target_predicate=_target(_action(1, 1)),
    )
    assert missing["value_added"] is False
    assert missing["dead_end"].startswith("adaptive resolver produced no replayable actions")


def test_req_capstone_4584_routing_and_registry_surface_env_adaptive_operator() -> None:
    """REQ-CAPSTONE-4584: reusable operator is selected and recorded in registry."""

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert mod.PRIMITIVE_OPERATOR in operators

    selected = kit.select_primitive_operators(mechanic_class="config_toggle", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("ft09")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert mod.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "drift" in gotchas[0]["note"].lower()


def test_req_capstone_4584_selects_a1_env_adaptive_as_strongest_signal() -> None:
    """REQ-CAPSTONE-4584: A1 env-adaptive re-solve wins over A3/A4 nulls."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact={
            "honest_verdict": "success: live_submittable_count_53_above_33",
            "count_delta": 20,
            "env_adaptive_resolve_recovered": ["sc25"],
        },
        a3_artifact={
            "honest_verdict": "complete: feature_router_no_value_honest_null_transfer_gap_sharpened",
            "transfer_delta": 0.0,
            "winner_generated": {"generated_count": 1},
        },
        a4_artifact={
            "honest_verdict": "complete: diversity_floor_no_transfer_honest_null_gap_sharpened",
            "firstwin_delta": 0,
        },
    )

    assert decision["source"] == "A1_env_adaptive_resolve"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["measured_signal"] > 0.0
    assert decision["source_tuning_games"] == ["sc25"]


def test_req_capstone_4584_transfer_measurement_reports_per_game_value() -> None:
    """REQ-CAPSTONE-4584: transfer records drift recovery for untuned games."""

    cases = [
        mod.TransferCase(
            game="s5i5",
            labels=("h_extend",),
            frozen_resolver={"h_extend": _action(47, 21)},
            adaptive_resolver={"h_extend": _action(50, 23)},
            expected_first_action=_action(50, 23),
            existing_reproduced_level=1,
        ),
        mod.TransferCase(
            game="ft09",
            labels=("click:36,36",),
            frozen_resolver={"click:36,36": _action(36, 36)},
            adaptive_resolver={"click:36,36": _action(39, 41)},
            expected_first_action=_action(39, 41),
            existing_reproduced_level=1,
        ),
    ]

    results = mod.measure_env_adaptive_transfer(cases)

    assert [row["game"] for row in results] == ["s5i5", "ft09"]
    assert all(row["value_added"] is True for row in results)
    assert results[0]["transfer_value"]["drift_recovered"] is True
    assert results[0]["transfer_value"]["offline_reproduced_new_level"] is False
    assert results[0]["transfer_value"]["existing_reproduced_level"] == 1


def test_scenario_capstone_4584_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4584: artifact schema records value-add or honest null."""

    decision = {
        "source": "A1_env_adaptive_resolve",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "source_tuning_games": ["sc25"],
        "selection_rationale": "A1 recovered drift.",
    }
    transfer_results = [
        {
            "game": "s5i5",
            "value_added": True,
            "transfer_value": {
                "drift_recovered": True,
                "winning_approach_selected": False,
                "win_reached": False,
                "offline_reproduced_new_level": False,
            },
            "dead_end": "",
        },
        {
            "game": "ft09",
            "value_added": False,
            "transfer_value": {
                "drift_recovered": False,
                "winning_approach_selected": False,
                "win_reached": False,
                "offline_reproduced_new_level": False,
            },
            "dead_end": "no env-discovered coordinate map was available",
        },
    ]

    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={"A1_env_adaptive_resolve": {"measured_signal": 1.0}},
        preconditions_checked={"ok": True},
        transfer_results=transfer_results,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "success: primitive_persisted_transfer_s5i5_value_added"
    assert artifact["primitive_persisted"]["operator"] == mod.PRIMITIVE_OPERATOR
    assert artifact["offline_reproduced"]["new_levels_banked"] == 0
    assert artifact["new_levels_banked"] == 0
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    null_artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[dict(row, value_added=False, dead_end="no transfer") for row in transfer_results],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert null_artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert mod.artifact_schema_errors(null_artifact) == []

    errors = mod.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "transfer_games must contain at least two games" in errors


def test_scenario_capstone_4584_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4584: run writes a stable result from upstream artifacts."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    registry = {
        "schema_version": 1,
        "general_gotchas": [
            {"id": mod.PRIMITIVE_GOTCHA_ID, "operator": mod.PRIMITIVE_OPERATOR, "note": "fixture"}
        ],
        "games": [],
        "reproducible_total_levels": 54,
    }
    (tmp_path / "ops").mkdir()
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    _write_json(
        tmp_path / mod.A1_RELATIVE_PATH,
        {
            "honest_verdict": "success: live_submittable_count_53_above_33",
            "count_delta": 20,
            "env_adaptive_resolve_recovered": ["sc25"],
        },
    )
    _write_json(
        tmp_path / mod.A3_RELATIVE_PATH,
        {"honest_verdict": "complete: feature_router_no_value_honest_null", "transfer_delta": 0.0},
    )
    _write_json(
        tmp_path / mod.A4_RELATIVE_PATH,
        {"honest_verdict": "complete: diversity_floor_no_transfer_honest_null", "firstwin_delta": 0},
    )

    artifact = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: True,
        now=iter([5.0, 5.25]).__next__,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["transfer_games"] == ["s5i5", "ft09", "sb26"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    blocked = mod.build_artifact(
        selected_upstream={"operator": mod.PRIMITIVE_OPERATOR, "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID},
        upstream_signals={},
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []

    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, root=tmp_path)


def test_req_capstone_4584_defensive_branches_are_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4584: defensive branches stay deterministic and honest."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": "bad"}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._as_int(True) == 0
    assert mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )["offline_arcade"] is False

    a3_selected = mod.select_primitive_from_upstreams(
        a1_artifact={},
        a3_artifact={"transfer_delta": 2.0},
        a4_artifact={},
    )
    assert "largest numeric signal" in a3_selected["selection_rationale"]
    all_null = mod.select_primitive_from_upstreams(a1_artifact={}, a3_artifact={}, a4_artifact={})
    assert "All upstreams were value-null" in all_null["selection_rationale"]

    decision = {
        "source": "A1_env_adaptive_resolve",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "source_tuning_games": ["sc25"],
    }
    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "zz99",
                "value_added": True,
                "transfer_value": {
                    "drift_recovered": True,
                    "existing_reproduced_level": 1,
                    "offline_reproduced_new_level": True,
                },
                "dead_end": "",
            },
            {
                "game": "yy88",
                "value_added": False,
                "transfer_value": {"existing_reproduced_level": 0},
                "dead_end": "no adaptive map",
            },
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert artifact["offline_reproduced"]["new_levels_banked"] == 1
    assert mod.artifact_schema_errors(artifact) == []

    wrong_gotcha = dict(artifact)
    wrong_gotcha["primitive_persisted"] = {
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": "wrong",
    }
    wrong_gotcha["reproducibility_checksum"] = mod.payload_checksum(wrong_gotcha)
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    no_value_success = dict(artifact)
    no_value_success["transfer_value_per_game"] = {"zz99": {"value_added": False}}
    no_value_success["reproducibility_checksum"] = mod.payload_checksum(no_value_success)
    assert "success requires at least one transfer value_added=true" in mod.artifact_schema_errors(
        no_value_success
    )

    mismatched_offline = dict(artifact)
    mismatched_offline["offline_reproduced"] = {
        "new_levels_banked": 1,
        "new_level_records": [],
    }
    mismatched_offline["reproducibility_checksum"] = mod.payload_checksum(mismatched_offline)
    assert "offline_reproduced new_levels_banked must match records" in mod.artifact_schema_errors(
        mismatched_offline
    )

    tampered = dict(artifact)
    tampered["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in mod.artifact_schema_errors(
        tampered
    )

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"honest_verdict": "bad"})
    with pytest.raises(ValueError, match="honest_verdict must start"):
        mod.run(tmp_path, offline_arcade_checker=lambda: True, write=False)

    assert kit._normalise_resolved_action(True) is None  # noqa: SLF001
    assert kit._normalise_resolved_action(3) == {"action": 3}  # noqa: SLF001
    assert kit._normalise_resolved_action("bad") is None  # noqa: SLF001
    assert kit._normalise_resolved_action({"x": 4, "y": 5}) == {  # noqa: SLF001
        "action": 6,
        "data": {"x": 4, "y": 5},
    }
    assert kit._normalise_resolved_action({"action": "bad"}) is None  # noqa: SLF001
    assert kit._normalise_resolved_action({"action": 6, "data": {"x": "bad", "y": 1}}) is None  # noqa: SLF001
    assert kit._resolve_action_with(None, {"action": 2}) == {"action": 2}  # noqa: SLF001
    assert kit._resolve_action_with({"1": {"action": 1}}, 1) == {"action": 1}  # noqa: SLF001
    assert kit._resolve_action_with(lambda _label: (_ for _ in ()).throw(RuntimeError("x")), "a") is None  # noqa: SLF001
    assert kit._call_resolve_target(None, [{"action": 1}], game="g", mode="m") is True  # noqa: SLF001
    assert kit._call_resolve_target(lambda _a, _b, _c, _d: True, [], game="g", mode="m") is False  # noqa: SLF001
    for kwargs in (
        {"mechanic_class": "cast_grid"},
        {"mechanic_class": "color_match"},
        {"mechanic_class": "sprite_overlay"},
        {"mechanic_class": "glyph"},
        {"mechanic_class": "program_editor"},
        {"mechanic_class": "object_motion"},
        {"mechanic_class": "unknown"},
    ):
        assert kit.select_primitive_operators(**kwargs)

    no_target = kit.env_adaptive_resolve_operator(
        [{"action": 1}],
        adaptive_resolver=None,
    )
    assert no_target["adaptive_reached"] is True
    not_reached = kit.env_adaptive_resolve_operator(
        ["a"],
        adaptive_resolver={"a": {"action": 1}},
        target_predicate=lambda _actions: False,
    )
    assert not_reached["dead_end"].startswith("adaptive resolver produced actions")
    frozen_already = kit.env_adaptive_resolve_operator(
        ["a"],
        frozen_resolver={"a": {"action": 1}},
        adaptive_resolver={"a": {"action": 1}},
        target_predicate=lambda _actions, _game: True,
    )
    assert frozen_already["dead_end"].startswith("frozen replay already reached")
