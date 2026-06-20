"""Tests for Exp 4503 HUD/register E3 L2 deepening.

Spec refs: REQ-ARC-WMTE-4503, SCENARIO-ARC-WMTE-4503.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4503_hud_register_deepen_l2 as exp4503


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions() -> dict[str, object]:
    return {
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "test",
    }


def test_req_arc_wmte_4503_spec_declares_incremental_hud_register_artifact() -> None:
    """REQ-ARC-WMTE-4503: OpenSpec names the 4503 L2 artifact and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in ("REQ-ARC-WMTE-4503", "SCENARIO-ARC-WMTE-4503"):
        assert ref in spec
    assert exp4503.RESULT_RELATIVE_PATH in spec
    for phrase in (
        "(grid, registers)",
        "hud_count",
        "undo_stack_depth",
        "target_game",
        "reproduction_gate",
        "solution_labels",
        "offline_reproduced=true",
        "reproduced_levels >= 1",
        "beyond L1",
    ):
        assert phrase in spec
    for field, principle in exp4503.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4503_registered_e3_state_carries_grid_and_registers() -> None:
    """REQ-ARC-WMTE-4503: E3 state is keyed by grid and register payload."""

    grid = np.zeros((3, 3), dtype=np.int16)
    grid[0, 0] = 4
    state = exp4503.make_e3_state("ka59", grid, {"hud_count": 0})
    blocked = exp4503.make_e3_state("ka59", grid, {"hud_count": 1})

    assert np.array_equal(state.grid, grid)
    assert state.registers == {"hud_count": 0}
    assert exp4503.e3_state_key(state) != exp4503.e3_state_key(blocked)
    assert exp4503.e3_state_key(state) == exp4503.e3_state_key((grid, {"hud_count": 0}))
    assert exp4503.induce_hud_registers("ka59", grid) == {"hud_count": 1}
    assert exp4503.induce_hud_registers("ar25", grid, {"undo_stack": [1, 2]}) == {
        "undo_stack_depth": 2
    }
    assert exp4503.induce_hud_registers("other", grid) == {}


def test_scenario_arc_wmte_4503_is_level_complete_reads_registered_hud_count() -> None:
    """SCENARIO-ARC-WMTE-4503: completion predicate reads the register scalar."""

    grid = np.zeros((3, 3), dtype=np.int16)
    grid[0, 0] = 4

    grid_only = exp4503.make_e3_state("ka59", grid, {})
    register_done = exp4503.make_e3_state("ka59", grid, {"hud_count": 0})
    register_blocked = exp4503.make_e3_state("ka59", grid, {"hud_count": 2})

    assert exp4503.registered_is_level_complete("ka59", grid_only) is False
    assert exp4503.registered_is_level_complete("ka59", register_done) is True
    assert exp4503.registered_is_level_complete("ka59", register_blocked) is False

    ar25_blocked = np.zeros((2, 2), dtype=np.int16)
    ar25_blocked[0, 0] = 1
    ar25_blocked[0, 1] = 1
    assert exp4503.registered_is_level_complete("ar25", ar25_blocked) is False
    assert exp4503.registered_is_level_complete("ar25", np.zeros((2, 2), dtype=np.int16)) is True
    with pytest.raises(ValueError, match="unsupported game"):
        exp4503.registered_is_level_complete("bad", register_done)


def test_req_arc_wmte_4503_artifact_success_requires_new_level_beyond_l1(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4503: success requires a reproduced L2 level beyond prior L1."""

    goal_report = exp4503.build_goal_accountability_report()
    artifact = exp4503.build_artifact(
        preconditions_checked=_preconditions(),
        target_game="ka59",
        solution_labels=["4", "2"],
        reproduction_gate={"game": "ka59", "reached_level": 2, "claimed_level": 2, "reproduced": True},
        goal_report=goal_report,
        tests_pass=True,
        residual_blockers=[],
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["goal_predicate_heldout_score"] == pytest.approx(1.0)
    assert artifact["grid_only_goal_predicate_heldout_score"] < 1.0
    assert artifact["schema_errors"] == []
    assert exp4503.artifact_schema_errors(artifact) == []

    out = exp4503.write_artifact(artifact, root=tmp_path)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    mutations = [
        (lambda item: item.pop("honest_verdict"), "missing required"),
        (lambda item: item.__setitem__("honest_verdict", "validated"), "terminal prefix"),
        (lambda item: item.__setitem__("inference_substrate", "live_llm_inference"), "inference_substrate"),
        (lambda item: item.__setitem__("preconditions_checked", []), "preconditions_checked"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles"),
        (lambda item: item.__setitem__("target_game", "bad"), "target_game"),
        (lambda item: item.__setitem__("reproduction_gate", []), "reproduction_gate"),
        (lambda item: item.__setitem__("solution_labels", "bad"), "solution_labels"),
        (lambda item: item.__setitem__("goal_predicate_heldout_score", None), "goal_predicate"),
        (lambda item: item.__setitem__("grid_only_goal_predicate_heldout_score", None), "grid_only"),
        (lambda item: item.__setitem__("offline_reproduced", False), "success artifact"),
        (lambda item: item.__setitem__("reproduced_levels", 0), "success artifact"),
    ]
    for mutate, expected in mutations:
        changed = dict(artifact)
        mutate(changed)
        assert any(expected in error for error in exp4503.artifact_schema_errors(changed))

    fabricated = dict(artifact)
    fabricated["offline_reproduced"] = False
    with pytest.raises(ValueError, match="success artifact"):
        exp4503.write_artifact(fabricated, root=tmp_path)


def test_scenario_arc_wmte_4503_runner_writes_injected_l2_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4503: injected offline replay success writes stable JSON."""

    def fake_runner(game: str, labels: tuple[str, ...], apply_fn: object, claimed_level: int):
        return {
            "game": game,
            "reached_level": claimed_level,
            "claimed_level": claimed_level,
            "reproduced": bool(labels),
        }

    artifact = exp4503.run_experiment(
        root=tmp_path,
        reproduction_runner=fake_runner,
        preconditions_checked=_preconditions(),
        tests_pass=True,
        target_game="ka59",
    )
    written = json.loads((tmp_path / exp4503.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["target_game"] == "ka59"


def test_scenario_arc_wmte_4503_runner_reports_honest_residual_when_l2_blocks(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4503: blocked L2 emits residual, not a fabricated success."""

    expected_l2_tail = ("3", "3", "5", "2", "2", "2", "2", "2", "2", "2", "2")

    def fake_runner(game: str, labels: tuple[str, ...], apply_fn: object, claimed_level: int):
        assert game == "ar25"
        assert labels[-len(expected_l2_tail) :] == expected_l2_tail
        return {
            "game": game,
            "reached_level": 1,
            "claimed_level": claimed_level,
            "reproduced": False,
            "residual": f"{game}_l2_not_reproduced",
        }

    artifact = exp4503.run_experiment(
        root=tmp_path,
        reproduction_runner=fake_runner,
        preconditions_checked=_preconditions(),
        tests_pass=True,
        target_game="ar25",
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["residual_blockers"] == ["ar25_l2_not_reproduced"]
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4503_runner_rejects_missing_resource_preconditions() -> None:
    """REQ-ARC-WMTE-4503: missing import or torch preconditions block before replay."""

    with pytest.raises(RuntimeError, match="blocked_offline_arcade_import_smoke"):
        exp4503.ensure_preconditions_ready({"offline_arcade_import_smoke": False, "torch_import": True})
    with pytest.raises(RuntimeError, match="blocked_torch_import"):
        exp4503.ensure_preconditions_ready({"offline_arcade_import_smoke": True, "torch_import": False})
