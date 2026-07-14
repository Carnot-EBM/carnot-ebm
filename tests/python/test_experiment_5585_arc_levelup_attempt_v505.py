"""Tests for Exp5585 V505 standing-loop ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-5585-LEVELUP,
SCENARIO-ARC-WMTE-5585-LEVELUP-ROTATED-TARGET,
SCENARIO-ARC-WMTE-5585-LEVELUP-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-5585-LEVELUP-STABLE-ARTIFACT.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5585_arc_levelup_attempt_v505 as exp5585


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / exp5585.SPEC_RELATIVE_PATH
RESULT_PATH = REPO / exp5585.RESULT_RELATIVE_PATH


def _registry() -> dict[str, Any]:
    return {
        "reproducible_total_levels": 177,
        "games": [
            {"game": "ar25", "levels_reproduced": 8, "full_game_clear": True},
            {"game": "lp85", "levels_reproduced": 8, "full_game_clear": True},
            {"game": "lf52", "levels_reproduced": 6, "full_game_clear": None},
            {"game": "sp80", "levels_reproduced": 6, "full_game_clear": True},
        ],
    }


def _loop_result(*, reproduced_levels: int = 2, reproduced: bool = True) -> dict[str, Any]:
    return {
        "game": "lf52",
        "reached_level": reproduced_levels,
        "offline_reproduced": reproduced,
        "reproduced_levels": reproduced_levels,
        "solve_provenance": "development_proxy",
        "states_expanded": 42,
        "reproduction_gate": {
            "game": "lf52",
            "claimed_level": reproduced_levels,
            "reached_level": reproduced_levels,
            "reproduced": reproduced,
        },
        "selected_generic_operators": [{"operator": "per_level_reinduction_operator"}],
        "solution_labels": ["a", "b"],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def test_req_arc_wmte_5585_levelup_spec_declares_required_fields() -> None:
    """REQ-ARC-WMTE-5585-LEVELUP: OpenSpec anchors the V505 ARC receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5585-LEVELUP") :]

    assert exp5585.RESULT_RELATIVE_PATH in section
    for ref in exp5585.SPEC_REFS:
        assert ref in section
    for field, principle in exp5585.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_arc_wmte_5585_levelup_rotates_to_shallow_nonfull_target() -> None:
    """SCENARIO-ARC-WMTE-5585-LEVELUP-ROTATED-TARGET: avoid recent and auto targets."""

    target = exp5585.select_rotated_target(
        _registry(),
        recent_targets=("lp85", "ar25"),
        auto_game="ar25",
    )

    assert target["game"] == "lf52"
    assert target["prior_reproduced_level"] == 6
    assert target["target_level"] == 7
    assert target["selection_reason"] == "shallowest_non_full_clear_rotated_target"
    assert target["skipped_recent_targets"] == ["lp85", "ar25"]
    assert target["auto_game"] == "ar25"


def test_scenario_arc_wmte_5585_levelup_target_selection_fallbacks() -> None:
    """SCENARIO-ARC-WMTE-5585-LEVELUP-ROTATED-TARGET: first-contact and fallback paths."""

    unsolved = {
        "games": [
            {"game": "zz99", "levels_reproduced": 0, "reproducibility": "unreproduced"},
            {"game": "aa11", "levels_reproduced": 0, "reproducibility": "unreproduced"},
        ]
    }
    all_full = {
        "games": [
            {"game": "bb22", "levels_reproduced": 4, "full_game_clear": True},
            {"game": "aa11", "levels_reproduced": 5, "full_game_clear": True},
        ]
    }

    assert exp5585.select_rotated_target(unsolved)["game"] == "aa11"
    fallback = exp5585.select_rotated_target(all_full, recent_targets=("aa11",))
    empty = exp5585.select_rotated_target({"games": []})

    assert fallback["game"] == "bb22"
    assert fallback["selection_reason"] == "shallowest_available_rotated_target"
    assert empty["game"] == ""
    assert empty["selection_reason"] == "blocked_empty_registry"


def test_req_arc_wmte_5585_recent_targets_and_missing_readers(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5585-LEVELUP: helper parsing is deterministic and fail-closed."""

    registry_path = tmp_path / "registry.yaml"
    json_path = tmp_path / "row.json"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "reproducible_total_levels": "9",
                "games": [
                    {"game": "a", "latest_exp10_levelup_attempt": {}},
                    {"game": "b", "latest_exp_no_digits": {}},
                    {"game": "a", "latest_exp12_levelup_attempt": {}},
                ],
            }
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    assert exp5585.read_json(tmp_path / "missing.json") == {}
    assert exp5585.read_json(json_path) == {"ok": True}
    assert exp5585.read_yaml(tmp_path / "missing.yaml") == {
        "reproducible_total_levels": 0,
        "games": [],
    }
    assert exp5585.read_yaml(registry_path)["reproducible_total_levels"] == "9"
    assert exp5585.recent_levelup_targets(exp5585.read_yaml(registry_path), limit=3) == [
        "a",
        "b",
    ]
    assert exp5585._as_int("not-int", 7) == 7


def test_scenario_arc_wmte_5585_levelup_gate_rejects_duplicate_depth() -> None:
    """SCENARIO-ARC-WMTE-5585-LEVELUP-REPRODUCTION-GATE: reproduced duplicate is no bank."""

    target = exp5585.select_rotated_target(
        _registry(),
        recent_targets=("lp85", "ar25"),
        auto_game="ar25",
    )
    artifact = exp5585.build_artifact(
        registry=_registry(),
        target=target,
        loop_result=_loop_result(reproduced_levels=2),
        auto_result={"game": "ar25", "offline_reproduced": True, "reproduced_levels": 3},
        registry_updated=True,
        command=[".venv/bin/python", "scripts/arc_loop_solve.py", "--game", "lf52"],
        auto_command=[".venv/bin/python", "scripts/arc_loop_solve.py", "--auto"],
        loop_artifact="results/arc_loop_solve_lf52.json",
        auto_artifact="results/arc_loop_solve_ar25.json",
        tests_run=["unit"],
        duration_s=1.0,
    )

    exp5585.validate_artifact(artifact)
    assert artifact["game_targeted"] == "lf52"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["prior_reproduced_level"] == 6
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_total_before"] == 177
    assert artifact["registry_total_after"] == 177
    assert artifact["registry_updated"] is True
    assert artifact["honest_verdict"].startswith("complete: no_new_arc_level_banked")
    assert "standing_loop_reproduced_only_l2_vs_prior_l6" in artifact["dead_ends_found"]


def test_scenario_arc_wmte_5585_levelup_fallback_gate_and_bad_provenance() -> None:
    """SCENARIO-ARC-WMTE-5585-LEVELUP-REPRODUCTION-GATE: fallback booleans stay honest."""

    target = exp5585.select_rotated_target(_registry(), recent_targets=(), auto_game="ar25")
    loop = {
        "game": "lf52",
        "offline_reproduced": False,
        "reproduced_levels": 99,
        "solve_provenance": "bad_value",
        "solution_labels": [],
    }
    artifact = exp5585.build_artifact(
        registry=_registry(),
        target=target,
        loop_result=loop,
        auto_result={},
        registry_updated=False,
        command=["cmd"],
        auto_command=["auto"],
        loop_artifact="loop.json",
        auto_artifact="auto.json",
        tests_run=["unit"],
        duration_s=0.0,
    )

    exp5585.validate_artifact(artifact)
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["solve_provenance"] == "development_proxy"


def test_scenario_arc_wmte_5585_levelup_checksum_is_stable() -> None:
    """SCENARIO-ARC-WMTE-5585-LEVELUP-STABLE-ARTIFACT: checksum covers gate evidence."""

    target = exp5585.select_rotated_target(_registry(), recent_targets=(), auto_game="ar25")
    artifact = exp5585.build_artifact(
        registry=_registry(),
        target=target,
        loop_result=_loop_result(reproduced_levels=7),
        auto_result={"game": "ar25", "offline_reproduced": True, "reproduced_levels": 3},
        registry_updated=True,
        command=["cmd"],
        auto_command=["auto"],
        loop_artifact="results/arc_loop_solve_lf52.json",
        auto_artifact="results/arc_loop_solve_ar25.json",
        tests_run=["unit"],
        duration_s=1.0,
    )
    changed = {**artifact, "new_levels_banked": 0}

    exp5585.validate_artifact(artifact)
    assert artifact["new_levels_banked"] == 1
    assert artifact["registry_total_after"] == 178
    assert artifact["honest_verdict"].startswith("complete: arc_levelup_banked")
    assert artifact["reproducibility_checksum"] == exp5585.compute_checksum(artifact)
    assert changed["reproducibility_checksum"] != exp5585.compute_checksum(changed)


def test_scenario_arc_wmte_5585_levelup_validation_errors() -> None:
    """SCENARIO-ARC-WMTE-5585-LEVELUP-STABLE-ARTIFACT: malformed artifacts fail closed."""

    target = exp5585.select_rotated_target(_registry(), recent_targets=(), auto_game="ar25")
    artifact = exp5585.build_artifact(
        registry=_registry(),
        target=target,
        loop_result=_loop_result(reproduced_levels=7),
        auto_result={},
        registry_updated=True,
        command=["cmd"],
        auto_command=["auto"],
        loop_artifact="loop.json",
        auto_artifact="auto.json",
        tests_run=["unit"],
        duration_s=1.0,
    )

    for bad in (
        {k: v for k, v in artifact.items() if k != "game_targeted"},
        {**artifact, "field_principles": []},
        {
            **artifact,
            "field_principles": {
                **artifact["field_principles"],
                "game_targeted": "wrong",
            },
        },
        {**artifact, "solve_provenance": "bad"},
        {**artifact, "reproducibility_checksum": "bad"},
        {**artifact, "offline_reproduced": False},
    ):
        with pytest.raises(ValueError):
            exp5585.validate_artifact(bad)


def test_req_arc_wmte_5585_write_main_and_recommendation_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-5585-LEVELUP: CLI writer reads loop outputs from one repo root."""

    root = tmp_path
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / exp5585.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "reproducible_total_levels": 177,
                "games": [
                    {
                        "game": "lf52",
                        "levels_reproduced": 6,
                        "full_game_clear": None,
                    },
                    {
                        "game": "lp85",
                        "levels_reproduced": 8,
                        "full_game_clear": True,
                        "latest_exp5040_levelup_attempt": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "results" / "arc_loop_solve_ar25.json").write_text(
        json.dumps({"game": "ar25", "offline_reproduced": True, "reproduced_levels": 3}),
        encoding="utf-8",
    )
    (root / "results" / "arc_loop_solve_lf52.json").write_text(
        json.dumps(_loop_result()),
        encoding="utf-8",
    )
    original_recommendation = exp5585._recommendation
    monkeypatch.setattr(exp5585, "REPO", root)
    monkeypatch.setattr(exp5585, "_recommendation", lambda game: {"recommended": game})

    assert exp5585.main() == 0
    written = json.loads((root / exp5585.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    exp5585.validate_artifact(written)
    assert written["transfer_routing_recommendation"] == {"recommended": "lf52"}

    original_import = builtins.__import__
    monkeypatch.setattr(exp5585, "_recommendation", original_recommendation)

    def _raise_import(name, *args, **kwargs):
        if name == "carnot.agentic":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise_import)
    assert exp5585._recommendation("lf52") == {}


def test_req_arc_wmte_5585_repository_artifact_records_honest_attempt() -> None:
    """REQ-ARC-WMTE-5585-LEVELUP: checked-in artifact records the real V505 run."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    exp5585.validate_artifact(artifact)
    assert artifact["game_targeted"] == "lf52"
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["prior_reproduced_level"] == 6
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_updated"] is True
    assert artifact["standing_loop"]["loop_artifact"] == "results/arc_loop_solve_lf52.json"
