"""Spec refs: REQ-ARC-WMTE-6213,
SCENARIO-ARC-WMTE-6213-TRANSLATION,
SCENARIO-ARC-WMTE-6213-HUD-REJECTION,
SCENARIO-ARC-WMTE-6213-FAIL-OPEN,
SCENARIO-ARC-WMTE-6213-PROMPT-WIRING.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition, induce_prompt


def _transition(
    before: np.ndarray,
    after: np.ndarray,
    *,
    action: int = 4,
    data: dict | None = None,
) -> Transition:
    return Transition(
        grid=before,
        action=action,
        data=data,
        next_grid=after,
        level_before=0,
        level_after=0,
    )


def _put_l(grid: np.ndarray, row: int, col: int, color: int) -> None:
    grid[row, col] = color
    grid[row + 1, col] = color
    grid[row, col + 1] = color


def _put_domino(grid: np.ndarray, row: int, col: int, color: int) -> None:
    grid[row, col] = color
    grid[row + 1, col] = color


def test_req_arc_wmte_6213_components_are_four_connected() -> None:
    from carnot.agentic.arc_object_delta_perception import extract_components

    grid = np.zeros((4, 4), dtype=np.int16)
    grid[1, 1] = 2
    grid[2, 2] = 2

    twos = [row for row in extract_components(grid) if row["color"] == 2]

    assert len(twos) == 2
    assert [row["area"] for row in twos] == [1, 1]


def test_req_arc_wmte_6213_translation_matches_and_relations_are_invariant() -> None:
    from carnot.agentic.arc_object_delta_perception import build_object_delta_table

    before = np.zeros((8, 10), dtype=np.int16)
    after = np.zeros_like(before)
    _put_l(before, 1, 1, 2)
    _put_domino(before, 4, 2, 3)
    _put_l(after, 2, 3, 2)
    _put_domino(after, 5, 4, 3)

    row = build_object_delta_table([_transition(before, after)])["transitions"][0]
    matches_by_color = {match["before_component"]["color"]: match for match in row["matches"]}

    assert matches_by_color[2]["centroid_delta"] == [1.0, 2.0]
    assert matches_by_color[3]["centroid_delta"] == [1.0, 2.0]
    relation = next(rel for rel in row["relations"] if set(rel["before_pair_colors"]) == {2, 3})
    assert relation["relation_invariant"] is True
    assert relation["relation_delta"] == [0.0, 0.0]


def test_req_arc_wmte_6213_stable_objects_emit_zero_delta() -> None:
    from carnot.agentic.arc_object_delta_perception import build_object_delta_table

    grid = np.zeros((5, 5), dtype=np.int16)
    _put_l(grid, 1, 1, 6)

    row = build_object_delta_table([_transition(grid, grid.copy(), action=1)])["transitions"][0]
    match = next(m for m in row["matches"] if m["before_component"]["color"] == 6)

    assert match["centroid_delta"] == [0.0, 0.0]
    assert match["delta_kind"] == "stable"
    assert row["changed_cell_count"] == 0


def test_req_arc_wmte_6213_splits_and_merges_do_not_fabricate_identity() -> None:
    from carnot.agentic.arc_object_delta_perception import build_object_delta_table

    before = np.zeros((6, 6), dtype=np.int16)
    after = np.zeros_like(before)
    before[1:3, 1:3] = 4
    after[1, 1:3] = 4
    after[4, 1:3] = 4

    row = build_object_delta_table([_transition(before, after)])["transitions"][0]

    assert not [m for m in row["matches"] if m["before_component"]["color"] == 4]
    assert [c["color"] for c in row["removed_before_components"]].count(4) == 1
    assert [c["color"] for c in row["created_after_components"]].count(4) == 2


def test_req_arc_wmte_6213_ambiguous_same_shape_matches_fail_open() -> None:
    from carnot.agentic.arc_object_delta_perception import build_object_delta_table

    before = np.zeros((8, 12), dtype=np.int16)
    after = np.zeros_like(before)
    before[1:3, 1:3] = 5
    before[5:7, 1:3] = 5
    after[1:3, 3:5] = 5
    after[5:7, 3:5] = 5

    row = build_object_delta_table([_transition(before, after)])["transitions"][0]

    assert not [m for m in row["matches"] if m["before_component"]["color"] == 5]
    assert row["ambiguous_matches"]
    assert row["ambiguous_matches"][0]["fail_open"] is True
    assert row["ambiguous_matches"][0]["before_count"] == 2
    assert row["ambiguous_matches"][0]["after_count"] == 2


def test_req_arc_wmte_6213_hud_strip_components_are_rejected_conservatively() -> None:
    from carnot.agentic.arc_object_delta_perception import build_object_delta_table

    before = np.zeros((6, 8), dtype=np.int16)
    before[0, :] = 7
    before[0, 4] = 8
    before[1, 4] = 8
    before[4, 2] = 3
    after = before.copy()
    after[0, 0] = 9

    row = build_object_delta_table([_transition(before, after)])["transitions"][0]

    assert row["hud_rejection"]["admitted"] is True
    assert row["hud_rejection"]["edge"] == "top"
    assert not [c for c in row["before_components"] if c["bbox"][0] == 0 and c["bbox"][2] == 0]
    assert any(c["color"] == 8 and c["bbox"] == [0, 4, 1, 4] for c in row["before_components"])
    assert row["hud_rejected_component_counts"]["before"] >= 1


def test_req_arc_wmte_6213_hud_rejection_fails_open_when_playfield_also_moves() -> None:
    from carnot.agentic.arc_object_delta_perception import build_object_delta_table

    before = np.zeros((6, 8), dtype=np.int16)
    before[0, :] = 7
    before[4, 2] = 3
    after = before.copy()
    after[0, 0] = 9
    after[4, 3] = 3
    after[4, 2] = 0

    row = build_object_delta_table([_transition(before, after)])["transitions"][0]

    assert row["hud_rejection"]["admitted"] is False
    assert any(c["bbox"] == [0, 0, 0, 7] for c in row["before_components"])


def test_req_arc_wmte_6213_serialization_is_deterministic() -> None:
    from carnot.agentic.arc_object_delta_perception import (
        build_object_delta_table,
        object_delta_block,
        object_delta_table_json,
    )

    before = np.zeros((7, 7), dtype=np.int16)
    after = np.zeros_like(before)
    _put_l(before, 1, 1, 2)
    _put_l(after, 2, 2, 2)
    trans = [_transition(before, after, action=6, data={"x": 2, "y": 1})]

    assert build_object_delta_table(trans) == build_object_delta_table(trans)
    assert object_delta_table_json(build_object_delta_table(trans)) == object_delta_table_json(
        build_object_delta_table(trans)
    )
    assert object_delta_block(trans) == object_delta_block(trans)


def test_req_arc_wmte_6213_fail_open_edges_and_caps_are_serialized() -> None:
    from carnot.agentic.arc_object_delta_perception import (
        _dominant_fraction,
        admitted_hud_strip,
        build_object_delta_table,
        extract_components,
        object_delta_block,
    )

    assert "no transitions" in object_delta_block([])
    assert admitted_hud_strip([])[1]["reason"] == "no_transitions"
    assert _dominant_fraction(np.asarray([], dtype=np.int16)) == 0.0

    bad_shape = Transition(
        grid=np.zeros((2, 2), dtype=np.int16),
        action=1,
        data=None,
        next_grid=np.zeros((3, 3), dtype=np.int16),
        level_before=0,
        level_after=0,
    )
    assert admitted_hud_strip([bad_shape])[1]["reason"] == "shape_mismatch"

    with np.testing.assert_raises(ValueError):
        extract_components(np.zeros((0,), dtype=np.int16))

    before = np.arange(16, dtype=np.int16).reshape(4, 4)
    after = before.copy()
    after[1, 1] = 99
    noisy = _transition(before, after, data={object(): "mixed-key"})
    table = build_object_delta_table([noisy, noisy], max_transitions=1)

    assert table["transition_count"] == 1
    assert table["hud_rejection"]["admitted"] is False
    assert isinstance(table["transitions"][0]["data"], str)


def test_req_arc_wmte_6213_prompt_flag_is_default_off_and_independent(monkeypatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", raising=False)
    monkeypatch.delenv("CARNOT_ARC_OBJECT_PERCEPTION", raising=False)
    grid = np.zeros((5, 5), dtype=np.int16)
    _put_l(grid, 1, 1, 2)
    trans = [_transition(grid, grid.copy())]

    default_prompt = induce_prompt("xx", trans, 1)
    assert "OBJECT STRUCTURE" in default_prompt
    assert "OBJECT DELTA PERCEPTION" not in default_prompt

    monkeypatch.setenv("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", "1")
    delta_prompt = induce_prompt("xx", trans, 1)
    assert "OBJECT STRUCTURE" in delta_prompt
    assert "OBJECT DELTA PERCEPTION" in delta_prompt

    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "0")
    independent_prompt = induce_prompt("xx", trans, 1)
    assert "OBJECT STRUCTURE" not in independent_prompt
    assert "OBJECT DELTA PERCEPTION" in independent_prompt


def test_req_arc_wmte_6213_prompt_hook_fails_open_exactly(monkeypatch) -> None:
    from carnot.agentic import arc_object_delta_perception as odp

    grid = np.zeros((5, 5), dtype=np.int16)
    _put_l(grid, 1, 1, 2)
    trans = [_transition(grid, grid.copy())]

    monkeypatch.setenv("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", "0")
    base_prompt = induce_prompt("xx", trans, 1)

    def _raise(_transitions):
        raise RuntimeError("forced serializer failure")

    monkeypatch.setattr(odp, "object_delta_block", _raise)
    monkeypatch.setenv("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", "1")

    assert induce_prompt("xx", trans, 1) == base_prompt


def test_req_arc_wmte_6213_module_is_in_live_import_closure() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "arc_orphan_solver_lint.py"
    spec = importlib.util.spec_from_file_location("arc_orphan_solver_lint", script_path)
    assert spec and spec.loader
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)

    closure = lint._closure(lint.ENTRYPOINTS)

    assert "arc_executable_world_model" in closure
    assert "arc_object_delta_perception" in closure


def test_req_arc_wmte_6213_experiment_artifact_contract() -> None:
    from carnot import experiment_6213_arc_object_delta_perception_wiring as exp

    artifact = exp.build_artifact(
        date="20260808",
        mutation_receipts=[
            {"name": "prompt_hook_deleted", "killed": True},
            {"name": "identity_normalization_removed", "killed": True},
            {"name": "hud_rejection_disabled", "killed": True},
            {"name": "ambiguity_guard_removed", "killed": True},
        ],
        test_commands=["unit-fixture"],
        test_exit_codes={"unit-fixture": 0},
    )

    exp.validate_artifact(artifact)
    assert artifact["solve_claimed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["object_delta_wiring_ready_score"] == 1.0
    assert all(
        type(value) is int and value == 0
        for value in artifact["source_bfs_adapter_registry_hidden_state_access_counts"].values()
    )


def test_req_arc_wmte_6213_experiment_env_restore_and_empty_score(monkeypatch) -> None:
    from carnot import experiment_6213_arc_object_delta_perception_wiring as exp

    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "1")
    monkeypatch.setenv("CARNOT_ARC_OBJECT_DELTA_PERCEPTION", "0")

    exp._with_prompt_env("1", "0")

    assert exp._ready_score([]) == 0.0
    assert os.environ["CARNOT_ARC_OBJECT_PERCEPTION"] == "1"
    assert os.environ["CARNOT_ARC_OBJECT_DELTA_PERCEPTION"] == "0"
