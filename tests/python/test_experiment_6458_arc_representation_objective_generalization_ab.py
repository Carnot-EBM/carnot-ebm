"""Tests for Exp6458 ARC representation-objective generalization.

Spec refs: REQ-ARC-ARM-6458,
SCENARIO-ARC-ARM-6458-PRECONDITIONS,
SCENARIO-ARC-ARM-6458-DISJOINT-TUNING-HELD,
SCENARIO-ARC-ARM-6458-MATCHED-ARMS,
SCENARIO-ARC-ARM-6458-CHECKPOINT-RESUME,
SCENARIO-ARC-ARM-6458-ROWS-RECOMPUTE,
SCENARIO-ARC-ARM-6458-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6458-NO-SOLVE-OR-PROMOTION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_6458_arc_representation_objective_generalization_ab as exp6458


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / exp6458.ARC_SPEC_RELATIVE_PATH


def _trace_root(tmp_path: Path) -> Path:
    root = tmp_path / "arc_transition_corpus"
    root.mkdir()
    for game_index, game in enumerate(("aa01", "bb02", "cc03", "dd04")):
        grids = []
        next_grids = []
        actions = []
        xs = []
        ys = []
        lb = []
        la = []
        for row_index, action in enumerate((6, 4, 6, 3, 6, 2, 6, 5)):
            grid = np.zeros((4, 4), dtype=np.int16)
            grid[0, 0] = game_index
            after = grid.copy()
            after[1 + (row_index % 3), 1 + (action % 3)] = action
            grids.append(grid)
            next_grids.append(after)
            actions.append(action)
            xs.append(1 + row_index)
            ys.append(2 + row_index)
            lb.append(0)
            la.append(0)
        np.savez(
            root / f"{game}.npz",
            grids=np.asarray(grids, dtype=np.int16),
            next_grids=np.asarray(next_grids, dtype=np.int16),
            actions=np.asarray(actions, dtype=np.int64),
            xs=np.asarray(xs, dtype=np.int64),
            ys=np.asarray(ys, dtype=np.int64),
            lb=np.asarray(lb, dtype=np.int64),
            la=np.asarray(la, dtype=np.int64),
        )
    return root


def _tests_run() -> list[dict[str, Any]]:
    return [{"command": exp6458.FOCUSED_TEST_COMMAND, "exit_code": 0}]


def test_req_arc_arm_6458_spec_declares_artifact_contract() -> None:
    """REQ-ARC-ARM-6458: OpenSpec names all required fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-ARM-6458") :]
    for marker in (
        "SCENARIO-ARC-ARM-6458-PRECONDITIONS",
        "SCENARIO-ARC-ARM-6458-DISJOINT-TUNING-HELD",
        "SCENARIO-ARC-ARM-6458-MATCHED-ARMS",
        "SCENARIO-ARC-ARM-6458-CHECKPOINT-RESUME",
        "SCENARIO-ARC-ARM-6458-ROWS-RECOMPUTE",
        "SCENARIO-ARC-ARM-6458-ATTACKS-FAIL-CLOSED",
        "SCENARIO-ARC-ARM-6458-NO-SOLVE-OR-PROMOTION",
        exp6458.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6458.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_arm_6458_preconditions_and_rosters_are_disjoint(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6458-DISJOINT-TUNING-HELD: splits are frozen."""

    trace_root = _trace_root(tmp_path)
    rosters = exp6458.freeze_rosters(trace_root, tuning_count=1, safety_count=1)
    preconditions = exp6458.preconditions_checked(
        trace_root=trace_root,
        checkpoint_path=tmp_path / "checkpoint.json",
        budgets=exp6458.ShardBudgets(max_prefixes_per_game=2),
        rosters=rosters,
    )

    assert rosters["disjointness"]["tuning_held_disjoint"] is True
    assert rosters["disjointness"]["all_splits_disjoint"] is True
    assert rosters["held_games"]
    assert rosters["manifest_hash"].startswith("sha256:")
    assert preconditions["readable_observation_action_traces"]["available"] is True
    assert preconditions["canonical_live_path_imports"]["available"] is True
    assert preconditions["writable_atomic_checkpoints"]["available"] is True
    assert preconditions["game_source_access_count"] == 0


def test_scenario_arc_arm_6458_run_writes_rows_and_checkpoints(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6458-MATCHED-ARMS: held cells cover all arms."""

    trace_root = _trace_root(tmp_path)
    result_path = tmp_path / exp6458.RESULT_RELATIVE_PATH.name
    checkpoint_path = tmp_path / "exp6458.checkpoint.json"
    artifact = exp6458.run(
        date="20260815",
        trace_root=trace_root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        budgets=exp6458.ShardBudgets(max_prefixes_per_game=2, max_cell_s=1.0),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=True,
        run_adversarial=False,
    )

    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6458.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["no_game_or_level_solve_claim"] is True
    assert artifact["solve_registry_unchanged"] is True
    assert artifact["game_source_access_count"] == 0
    assert artifact["offline_ground_truth_bfs_count"] == 0
    assert artifact["per_game_adapter_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert "solve_provenance" not in artifact
    assert checkpoint["completed_cell_count"] == len(artifact["per_unit_rows"])
    assert {row["arm"] for row in artifact["per_unit_rows"]} == set(exp6458.ARMS)
    assert all(row["chosen_action"] in row["legal_action_set"] for row in artifact["per_unit_rows"])
    assert all(row["checkpoint_receipt"]["written"] is True for row in artifact["per_unit_rows"])
    exp6458.validate_artifact(artifact)


def test_scenario_arc_arm_6458_resume_skips_completed_cells(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6458-CHECKPOINT-RESUME: completed cells are not repeated."""

    trace_root = _trace_root(tmp_path)
    checkpoint_path = tmp_path / "resume.checkpoint.json"
    partial = exp6458.run(
        date="20260815",
        trace_root=trace_root,
        result_path=tmp_path / "partial.json",
        checkpoint_path=checkpoint_path,
        budgets=exp6458.ShardBudgets(max_prefixes_per_game=2, max_cells=1),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=True,
        run_adversarial=False,
    )
    resumed = exp6458.run(
        date="20260815",
        trace_root=trace_root,
        result_path=tmp_path / "complete.json",
        checkpoint_path=checkpoint_path,
        budgets=exp6458.ShardBudgets(max_prefixes_per_game=2),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=True,
        run_adversarial=False,
    )

    assert partial["status"] == "complete_partial"
    assert partial["resume_and_terminal_partial_receipts"]["terminal_partial_written"] is True
    assert resumed["resume_and_terminal_partial_receipts"]["resume_skipped_completed_cells"] >= 1
    assert resumed["resume_and_terminal_partial_receipts"]["completed_cell_repetition_count"] == 0
    assert len(resumed["per_unit_rows"]) > len(partial["per_unit_rows"])


def test_scenario_arc_arm_6458_rows_recompute_and_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6458-ROWS-RECOMPUTE: aggregate drift is rejected."""

    artifact = exp6458.run(
        date="20260815",
        trace_root=_trace_root(tmp_path),
        result_path=tmp_path / "artifact.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        budgets=exp6458.ShardBudgets(max_prefixes_per_game=2),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=False,
        run_adversarial=False,
    )
    recomputed = exp6458.recompute_aggregates(artifact["per_unit_rows"])

    assert recomputed["collision_rates_by_arm"] == artifact["collision_rates_by_arm"]
    assert recomputed["legal_action_coverage_by_arm"] == artifact["legal_action_coverage_by_arm"]
    assert recomputed["held_next_state_reachability_by_arm"] == artifact[
        "held_next_state_reachability_by_arm"
    ]
    assert all(row["fail_closed"] is True for row in artifact["attack_matrix"])

    attacked = copy.deepcopy(artifact)
    attacked["collision_rates_by_arm"][exp6458.BASELINE_ARM]["rate"] = 0.0
    with pytest.raises(ValueError, match="aggregate_row_mismatch"):
        exp6458.validate_artifact(attacked)


def test_scenario_arc_arm_6458_no_solve_validation_rejects_forbidden_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6458-NO-SOLVE-OR-PROMOTION: forbidden drift fails."""

    artifact = exp6458.run(
        date="20260815",
        trace_root=_trace_root(tmp_path),
        result_path=tmp_path / "artifact.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        budgets=exp6458.ShardBudgets(max_prefixes_per_game=1),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=False,
        run_adversarial=False,
    )

    for field, value, message in (
        ("no_game_or_level_solve_claim", False, "no_game_or_level_solve_claim"),
        ("solve_registry_unchanged", False, "solve_registry_unchanged"),
        ("game_source_access_count", 1, "game_source_access_count"),
        ("offline_ground_truth_bfs_count", 1, "offline_ground_truth_bfs_count"),
        ("per_game_adapter_count", 1, "per_game_adapter_count"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ):
        attacked = copy.deepcopy(artifact)
        attacked[field] = value
        attacked["reproducibility_checksum"] = exp6458.payload_checksum(attacked)
        with pytest.raises(ValueError, match=message):
            exp6458.validate_artifact(attacked)

    missing_principle = copy.deepcopy(artifact)
    missing_principle["field_principles"].pop("combined_improves_over_single_change_arms")
    missing_principle["reproducibility_checksum"] = exp6458.payload_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        exp6458.validate_artifact(missing_principle)


def test_scenario_arc_arm_6458_cli_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6458-PRECONDITIONS: CLI writes a terminal artifact."""

    rc = exp6458.main(
        [
            "--date",
            "20260815",
            "--trace-root",
            str(_trace_root(tmp_path)),
            "--out",
            str(tmp_path / "cli.json"),
            "--checkpoint",
            str(tmp_path / "cli.checkpoint.json"),
            "--max-prefixes-per-game",
            "1",
            "--tuning-count",
            "1",
            "--safety-count",
            "1",
            "--skip-adversarial",
        ]
    )

    payload = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["honest_verdict"].startswith(("success:", "complete:"))
