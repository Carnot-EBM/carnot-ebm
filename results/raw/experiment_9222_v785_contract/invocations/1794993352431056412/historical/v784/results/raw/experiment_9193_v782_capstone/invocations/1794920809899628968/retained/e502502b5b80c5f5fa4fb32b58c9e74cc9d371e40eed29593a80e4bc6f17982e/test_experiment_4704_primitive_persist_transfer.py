"""Tests for Exp 4704 .434 primitive persistence and transfer characterization.

Spec refs: REQ-ARC-WMTE-4704,
SCENARIO-ARC-WMTE-4704-PERSIST-STRONGEST-COMPONENT,
SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4704_primitive_persist_transfer as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _grid() -> list[list[int]]:
    return [
        [0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [0, 3, 3, 0, 0],
    ]


def _a1_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: object_centric_perception_no_new_level_residual_offpath_calibration_insufficient",
        "chosen_submitted_config": "unchanged",
        "generic_agent_reached_level": 0,
        "reproduced_levels": 0,
        "perception_is_the_wall": True,
        "proposal_coverage_by_representation": {
            "order1": {"coverage": 0.75, "covered_steps": 3, "total_steps": 4},
            "object_centric": {
                "coverage": 1.0,
                "covered_steps": 4,
                "total_steps": 4,
                "deployable": True,
            },
            "upper_bound_ceiling": {"coverage": 1.0, "deployable": False},
        },
        "target_arm_results": {
            "object_centric": {
                "game": "r11l",
                "reached_level": 0,
                "offline_reproduced": False,
                "object_centric_proposal_diagnostics": {
                    "enabled": True,
                    "last_slot_count": 175,
                    "augmented_candidates": 6913,
                    "candidate_scores": 8477,
                    "offpath_calibrated": True,
                    "representation": "connected_components_object_slots_plus_correspondence_action_context",
                    "verifier_is_oracle": False,
                },
            },
            "order1_ablation": {
                "game": "r11l",
                "reached_level": 0,
                "offline_reproduced": False,
            },
        },
        "offline_reproduced": False,
    }


def _a2_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: amortized_prior_go_explore_no_coverage_gain_residual_logged",
        "chosen_submitted_config": "unchanged",
        "candidate_generation_coverage_with_prior": 0.0,
        "candidate_generation_coverage_no_prior_baseline": 0.0,
        "coverage_delta": 0.0,
        "first_win_rate_delta": 0.0,
        "target_arm_results": {
            "with_prior": [
                {
                    "game": "bp35",
                    "reached_level": 0,
                    "offline_reproduced": False,
                    "amortized_prior_diagnostics": {
                        "enabled": True,
                        "trace_count": 11,
                        "learned_family_keys": ["step0:click"],
                        "verifier_is_oracle": False,
                    },
                    "go_explore_archive_diagnostics": {
                        "enabled": True,
                        "stored_cells": 0,
                        "selected_prefixes": 0,
                        "verifier_is_oracle": False,
                    },
                }
            ],
            "no_prior": [{"game": "bp35", "reached_level": 0, "offline_reproduced": False}],
        },
        "offline_reproduced": False,
        "residual_bridge_gap": "archive_expands_dead_cells_no_goal_gradient",
    }


def test_req_arc_wmte_4704_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4704: OpenSpec declares the persistence/transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4704",
        "SCENARIO-ARC-WMTE-4704-PERSIST-STRONGEST-COMPONENT",
        "SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4704_solver_kit_operator_builds_object_representation() -> None:
    """REQ-ARC-WMTE-4704: the primitive builds oracle-distinct object rows."""

    result = kit.object_centric_representation_builder_operator(
        [
            {
                "game": "heldout_a",
                "grid": _grid(),
                "object_centric_coverage": 0.5,
                "order1_coverage": 0.25,
                "first_win_rate_delta": 0.0,
            },
            {"game": "empty", "grid": [[0, 0], [0, 0]]},
            {"game": "bad", "grid": [1, 2, 3]},
        ],
        min_component_count=1,
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["verifier_is_oracle"] is False
    assert result["representation_row_count"] == 3
    assert result["usable_representation_count"] == 1
    assert result["rejected_representation_count"] == 2
    assert result["coverage_ready"] is True
    assert result["best_coverage_delta"] == 0.25
    assert result["object_centric_representations"][0]["game"] == "heldout_a"
    assert result["object_centric_representations"][0]["usable"] is True

    empty = kit.object_centric_representation_builder_operator([])
    assert empty["coverage_ready"] is False
    assert empty["residual"] == "no_object_centric_rows"

    no_components = kit.object_centric_representation_builder_operator(
        [{"game": "blank", "grid": [[0, 0], [0, 0]]}]
    )
    assert no_components["coverage_ready"] is False
    assert no_components["residual"] == "no_usable_object_centric_representation"


def test_req_arc_wmte_4704_routing_and_registry_surface_persisted_operator() -> None:
    """REQ-ARC-WMTE-4704: solver-kit selection and registry expose the primitive."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "object-centric representation" in gotchas[0]["note"]
    assert "latest_exp4704_transfer" in gotchas[0]


def test_scenario_arc_wmte_4704_selects_a1_representation_when_a1_a2_null() -> None:
    """SCENARIO-ARC-WMTE-4704-PERSIST-STRONGEST-COMPONENT: null A1/A2 persists A1."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert decision["source"] == "A1_object_centric_representation_builder"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["upstream_signal_rank"][0]["source"] == (
        "A1_object_centric_representation_builder"
    )
    assert "both A1 and A2 were value-null" in decision["selection_rationale"]

    a1_success = dict(
        _a1_artifact(),
        honest_verdict="success: object_centric_perception_generic_agent_new_level_r11l_L1",
        generic_agent_reached_level=1,
        reproduced_levels=1,
    )
    assert (
        mod.select_primitive_from_upstreams(a1_artifact=a1_success, a2_artifact=_a2_artifact())[
            "source"
        ]
        == "A1_object_centric_perception_operator"
    )

    a2_success = dict(
        _a2_artifact(),
        honest_verdict="success: amortized_prior_go_explore_coverage_up_heldout_firstwin_lift_bp35",
        coverage_delta=1.0,
        first_win_rate_delta=1.0,
        live_first_win_rate_with_prior=1.0,
    )
    assert (
        mod.select_primitive_from_upstreams(a1_artifact={}, a2_artifact=a2_success)["source"]
        == "A2_amortized_prior_go_explore_archive"
    )


def test_scenario_arc_wmte_4704_transfer_measurement_reports_cached_null_and_value() -> None:
    """SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT: rows report value or null."""

    null = mod.measure_transfer_game(
        "cd82",
        frame_row_provider=lambda _game: {"game": "cd82", "grid": _grid()},
    )

    assert null["game"] == "cd82"
    assert null["value_added"] is False
    assert null["transfer_value"]["candidate_generation_coverage_delta"] == 0.0
    assert null["transfer_value"]["first_win_rate_delta"] == 0.0
    assert null["transfer_value"]["live_solve_rate_delta"] == 0.0
    assert null["transfer_value"]["offline_reproduced_new_level"] is False
    assert null["transfer_value"]["usable_representation_count"] == 1
    assert "no cached winning-prefix coverage lift" in null["dead_end"]

    lifted = mod.measure_transfer_game(
        "dc22",
        frame_row_provider=lambda _game: {
            "game": "dc22",
            "grid": _grid(),
            "object_centric_coverage": 1.0,
            "order1_coverage": 0.0,
            "first_win_rate_delta": 1.0,
            "live_solve_rate_delta": 1.0,
            "offline_reproduced_new_level": True,
        },
    )
    assert lifted["value_added"] is True
    assert lifted["transfer_value"]["candidate_generation_coverage_delta"] == 1.0
    assert lifted["transfer_value"]["first_win_rate_delta"] == 1.0
    assert lifted["transfer_value"]["live_solve_rate_delta"] == 1.0
    assert lifted["transfer_value"]["offline_reproduced_new_level"] is True

    explicit = mod.measure_transfer_game(
        "g50t",
        frame_row_provider=lambda _game: {
            "game": "g50t",
            "grid": _grid(),
            "candidate_generation_coverage_delta": 0.5,
        },
    )
    assert explicit["transfer_value"]["candidate_generation_coverage_delta"] == 0.5

    cached_delta = mod.measure_transfer_game(
        "g50t",
        frame_row_provider=lambda _game: {"game": "g50t", "grid": _grid(), "coverage_delta": 0.25},
    )
    assert cached_delta["transfer_value"]["candidate_generation_coverage_delta"] == 0.25

    provider_error = mod.measure_transfer_game(
        "g50t",
        frame_row_provider=lambda _game: (_ for _ in ()).throw(RuntimeError("missing frame")),
    )
    assert "cached frame unavailable" in provider_error["dead_end"]

    no_representation = mod.measure_transfer_game(
        "g50t",
        frame_row_provider=lambda _game: {"game": "g50t", "grid": [[0, 0], [0, 0]]},
    )
    assert no_representation["dead_end"] == (
        "no usable object-centric representation for this held-out game"
    )


def test_scenario_arc_wmte_4704_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT: artifact schema records transfer."""

    decision = {
        "source": "A1_object_centric_representation_builder",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "selection_rationale": "fixture",
    }
    rows = [
        {
            "game": game,
            "value_added": False,
            "transfer_value": {
                "candidate_generation_coverage_delta": 0.0,
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "offline_reproduced_new_level": False,
            },
            "operator_result": {"operator": mod.PRIMITIVE_OPERATOR},
            "dead_end": "null",
        }
        for game in ("cd82", "dc22", "g50t")
    ]

    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["offline_reproduced_new_level"] is False
    assert "object-centric representation" in artifact["residual_dead_end"]
    assert mod.artifact_schema_errors(artifact) == []
    assert (
        json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8"))
        == artifact
    )

    success_rows = [dict(row) for row in rows]
    success_rows[0] = {
        **success_rows[0],
        "value_added": True,
        "dead_end": "",
        "transfer_value": {
            **success_rows[0]["transfer_value"],
            "candidate_generation_coverage_delta": 1.0,
            "offline_reproduced_new_level": True,
        },
    }
    success = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=success_rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.5,
    )
    assert success["honest_verdict"] == "success: primitive_persisted_transfer_value_characterized"
    assert success["residual_dead_end"] == ""
    assert mod.artifact_schema_errors(success) == []


def test_scenario_arc_wmte_4704_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT: run writes a stable artifact."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "general_gotchas": [
                    {
                        "id": mod.PRIMITIVE_GOTCHA_ID,
                        "operator": mod.PRIMITIVE_OPERATOR,
                        "note": "fixture",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())

    artifact = mod.run(
        tmp_path,
        transfer_games=("cd82", "dc22", "g50t"),
        offline_arcade_checker=lambda: True,
        frame_row_provider=lambda game: {"game": game, "grid": _grid()},
        now=iter([10.0, 10.25]).__next__,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["transfer_games"] == ["cd82", "dc22", "g50t"]
    assert artifact["duration_s"] == 1.0
    assert artifact["preconditions_checked"]["ok"] is True
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4704_defensive_branches_are_schema_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4704: malformed inputs stay explicit and checksum-gated."""

    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})
    assert mod._load_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert mod._load_json(bad) == {}
    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": "bad"}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._as_int(False) == 0
    assert mod._as_int("bad") == 0
    assert mod._coverage_delta({"object_centric": {"coverage": 0.75}}) == 0.75
    assert mod._coverage_delta({"object_centric": "bad"}) == 0.0

    checks = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]

    blocked = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: False,
        frame_row_provider=lambda game: {"game": game, "grid": _grid()},
        now=iter([1.0, 1.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []

    malformed = mod.build_artifact(
        selected_upstream={
            "source": "A1_object_centric_representation_builder",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": game,
                "value_added": game == "aa00",
                "transfer_value": {"offline_reproduced_new_level": game == "aa00"},
                "dead_end": "" if game == "aa00" else "null",
            }
            for game in ("aa00", "bb00", "cc00")
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert malformed["offline_reproduced_new_level"] is True

    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "bad"
    malformed["verifier_is_oracle"] = True
    malformed["primitive_persisted"] = {}
    malformed["transfer_games"] = []
    malformed["transfer_value_per_game"] = []
    malformed["offline_reproduced_new_level"] = "yes"
    malformed["residual_dead_end"] = []
    malformed["random_seed"] = "bad"
    malformed["registry_updated"] = "yes"
    malformed["field_principles"] = {}
    malformed["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match REQ-ARC-WMTE-4704" in errors
    assert "verifier_is_oracle must be false" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "offline_reproduced_new_level must be a bare bool" in errors
    assert "reproducibility_checksum must match artifact content" in errors

    wrong_gotcha = mod.build_artifact(
        selected_upstream={
            "source": "A1_object_centric_representation_builder",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": "wrong",
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {"game": game, "value_added": False, "transfer_value": {}, "dead_end": "null"}
            for game in ("aa00", "bb00", "cc00")
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=1.0,
    )
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    success_without_value = mod.build_artifact(
        selected_upstream={
            "source": "A1_object_centric_representation_builder",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {"game": game, "value_added": False, "transfer_value": {}, "dead_end": "null"}
            for game in ("aa00", "bb00", "cc00")
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=1.0,
    )
    success_without_value["honest_verdict"] = (
        "success: primitive_persisted_transfer_value_characterized"
    )
    success_without_value["reproducibility_checksum"] = mod.payload_checksum(success_without_value)
    assert "success requires at least one transfer value_added=true" in mod.artifact_schema_errors(
        success_without_value
    )

    offline_mismatch = dict(success_without_value)
    offline_mismatch["honest_verdict"] = "complete: primitive_persisted_transfer_null_characterized"
    offline_mismatch["offline_reproduced"] = {
        "new_levels_banked": 2,
        "new_level_records": [],
    }
    offline_mismatch["reproducibility_checksum"] = mod.payload_checksum(offline_mismatch)
    assert "offline_reproduced new_levels_banked must match records" in mod.artifact_schema_errors(
        offline_mismatch
    )

    with pytest.raises(ValueError):
        mod.write_artifact({"honest_verdict": "bad"}, root=tmp_path)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced error"])
    with pytest.raises(ValueError, match="forced error"):
        mod.run(
            tmp_path,
            offline_arcade_checker=lambda: False,
            frame_row_provider=lambda game: {"game": game, "grid": _grid()},
            now=iter([2.0, 2.1]).__next__,
            write=False,
        )
