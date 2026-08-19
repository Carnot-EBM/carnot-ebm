"""Tests for Exp6471 ARC generic safety shield.

Spec refs: REQ-ARC-ARM-6471,
SCENARIO-ARC-ARM-6471-PRECHECK-AND-FREEZE,
SCENARIO-ARC-ARM-6471-GENERIC-SHIELD,
SCENARIO-ARC-ARM-6471-MATCHED-ROWS,
SCENARIO-ARC-ARM-6471-CHECKPOINT-RESUME,
SCENARIO-ARC-ARM-6471-ROWS-RECOMPUTE,
SCENARIO-ARC-ARM-6471-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6471-NO-SOLVE-OR-PROMOTION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_6471_arc_generic_safety_shield_objective_ab as exp6471


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / exp6471.ARC_SPEC_RELATIVE_PATH


def _trace_root(tmp_path: Path) -> Path:
    root = tmp_path / "arc_transition_corpus"
    root.mkdir()
    for game_index, game in enumerate(("aa01", "bb02", "g50t", "su15")):
        grids = []
        next_grids = []
        actions = []
        xs = []
        ys = []
        for row_index, action in enumerate((6, 4, 6, 3, 2, 2, 3, 2, 4, 5)):
            grid = np.zeros((4, 4), dtype=np.int16)
            grid[0, 0] = game_index
            grid[0, 1] = row_index % 3
            after = grid.copy()
            after[1 + (row_index % 3), 1 + (action % 3)] = action
            grids.append(grid)
            next_grids.append(after)
            actions.append(action)
            xs.append(1 + row_index)
            ys.append(2 + row_index)
        np.savez(
            root / f"{game}.npz",
            grids=np.asarray(grids, dtype=np.int16),
            next_grids=np.asarray(next_grids, dtype=np.int16),
            actions=np.asarray(actions, dtype=np.int64),
            xs=np.asarray(xs, dtype=np.int64),
            ys=np.asarray(ys, dtype=np.int64),
        )
    return root


def _tests_run() -> list[dict[str, Any]]:
    return [{"command": exp6471.FOCUSED_TEST_COMMAND, "exit_code": 0}]


def test_req_arc_arm_6471_spec_declares_artifact_contract() -> None:
    """REQ-ARC-ARM-6471: OpenSpec names all required fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-ARM-6471") :]
    for marker in (
        "SCENARIO-ARC-ARM-6471-PRECHECK-AND-FREEZE",
        "SCENARIO-ARC-ARM-6471-GENERIC-SHIELD",
        "SCENARIO-ARC-ARM-6471-MATCHED-ROWS",
        "SCENARIO-ARC-ARM-6471-CHECKPOINT-RESUME",
        "SCENARIO-ARC-ARM-6471-ROWS-RECOMPUTE",
        "SCENARIO-ARC-ARM-6471-ATTACKS-FAIL-CLOSED",
        "SCENARIO-ARC-ARM-6471-NO-SOLVE-OR-PROMOTION",
        exp6471.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6471.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_arm_6471_generic_shield_uses_runtime_features() -> None:
    """SCENARIO-ARC-ARM-6471-GENERIC-SHIELD: no game feature is needed."""

    vetoed = exp6471.generic_shield_decision(
        prior_action_history=[2, 4, 4, 3, 2, 2],
        objective_action=6,
        baseline_action=2,
        legal_actions=[1, 2, 3, 4, 5, 6],
        shuffled=False,
    )
    allowed = exp6471.generic_shield_decision(
        prior_action_history=[2, 4, 4, 3, 2, 6],
        objective_action=6,
        baseline_action=2,
        legal_actions=[1, 2, 3, 4, 5, 6],
        shuffled=False,
    )
    shuffled = exp6471.generic_shield_decision(
        prior_action_history=[2, 4, 4, 3, 2, 2],
        objective_action=6,
        baseline_action=2,
        legal_actions=[1, 2, 3, 4, 5, 6],
        shuffled=True,
        shuffle_key="g50t:00001:6458001",
    )

    assert vetoed == {
        "chosen_action": 2,
        "shield_applied": True,
        "shield_reason": "mature_non_click_history_veto",
        "fallback_action": 2,
    }
    assert allowed["chosen_action"] == 6
    assert allowed["shield_reason"] == "objective_allowed"
    assert shuffled["shield_reason"] in {
        "shuffled_control_veto",
        "shuffled_control_allow",
    }


def test_scenario_arc_arm_6471_run_writes_rows_and_exact_aggregates(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6471-MATCHED-ROWS: every fold has all arms."""

    result_path = tmp_path / exp6471.RESULT_RELATIVE_PATH.name
    checkpoint_path = tmp_path / "exp6471.checkpoint.json"
    artifact = exp6471.run(
        date="20260819",
        trace_root=_trace_root(tmp_path),
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        budgets=exp6471.ShardBudgets(max_prefixes_per_game=2),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=True,
        run_adversarial=False,
        progress=False,
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    recomputed = exp6471.canonical_row_reducer(artifact["per_unit_rows"], artifact)

    assert loaded == artifact
    assert set(exp6471.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["no_solve_claim"] is True
    assert artifact["verifier_is_oracle"] is False
    assert "solve_provenance" not in artifact
    assert {row["arm"] for row in artifact["per_unit_rows"]} == set(exp6471.ARMS)
    assert all(row["decision"]["chosen_action"] in row["legal_action_set"] for row in artifact["per_unit_rows"])
    assert all(row["timing"]["checkpoint_written"] is True for row in artifact["per_unit_rows"])
    assert checkpoint["completed_cell_count"] == len(artifact["per_unit_rows"])
    for field in exp6471.CANONICAL_AGGREGATE_FIELDS:
        assert artifact[field] == recomputed[field]
    exp6471.validate_artifact(artifact)


def test_scenario_arc_arm_6471_resume_skips_completed_cells(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6471-CHECKPOINT-RESUME: partial rows are not repeated."""

    trace_root = _trace_root(tmp_path)
    checkpoint_path = tmp_path / "resume.checkpoint.json"
    partial = exp6471.run(
        date="20260819",
        trace_root=trace_root,
        result_path=tmp_path / "partial.json",
        checkpoint_path=checkpoint_path,
        budgets=exp6471.ShardBudgets(max_prefixes_per_game=2, max_cells=3),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=True,
        run_adversarial=False,
        progress=False,
    )
    resumed = exp6471.run(
        date="20260819",
        trace_root=trace_root,
        result_path=tmp_path / "complete.json",
        checkpoint_path=checkpoint_path,
        budgets=exp6471.ShardBudgets(max_prefixes_per_game=2),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=True,
        run_adversarial=False,
        progress=False,
    )

    assert partial["status"] == "complete_partial"
    assert partial["checkpoint_and_resume_receipts"]["terminal_partial_written"] is True
    assert resumed["checkpoint_and_resume_receipts"]["resume_skipped_completed_cells"] >= 3
    assert resumed["checkpoint_and_resume_receipts"]["completed_cell_repetition_count"] == 0
    assert len(resumed["per_unit_rows"]) > len(partial["per_unit_rows"])


def test_scenario_arc_arm_6471_rows_recompute_and_attacks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6471-ROWS-RECOMPUTE: aggregate drift is rejected."""

    artifact = exp6471.run(
        date="20260819",
        trace_root=_trace_root(tmp_path),
        result_path=tmp_path / "artifact.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        budgets=exp6471.ShardBudgets(max_prefixes_per_game=2),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=False,
        run_adversarial=False,
        progress=False,
    )

    assert all(row["fail_closed"] is True for row in artifact["attack_matrix"])
    attacked = copy.deepcopy(artifact)
    first_arm = exp6471.BASELINE_ARM
    attacked["reachability_by_arm"][first_arm]["reachable"] = -1
    attacked["reproducibility_checksum"] = exp6471.payload_checksum(attacked)
    with pytest.raises(ValueError, match="aggregate_row_mismatch:reachability_by_arm"):
        exp6471.validate_artifact(attacked)


def test_scenario_arc_arm_6471_no_solve_validation_rejects_forbidden_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6471-NO-SOLVE-OR-PROMOTION: forbidden drift fails."""

    artifact = exp6471.run(
        date="20260819",
        trace_root=_trace_root(tmp_path),
        result_path=tmp_path / "artifact.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        budgets=exp6471.ShardBudgets(max_prefixes_per_game=1),
        tuning_count=1,
        safety_count=1,
        tests_run=_tests_run(),
        write=False,
        run_adversarial=False,
        progress=False,
    )

    for field, value, message in (
        ("no_solve_claim", False, "no_solve_claim"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ):
        attacked = copy.deepcopy(artifact)
        attacked[field] = value
        attacked["reproducibility_checksum"] = exp6471.payload_checksum(attacked)
        with pytest.raises(ValueError, match=message):
            exp6471.validate_artifact(attacked)

    attacked = copy.deepcopy(artifact)
    attacked["source_and_adapter_access_receipts"]["game_source_access_count"] = 1
    attacked["reproducibility_checksum"] = exp6471.payload_checksum(attacked)
    with pytest.raises(ValueError, match="game_source_access_count"):
        exp6471.validate_artifact(attacked)

    missing_field = copy.deepcopy(artifact)
    missing_field.pop("status")
    missing_field["reproducibility_checksum"] = exp6471.payload_checksum(missing_field)
    with pytest.raises(ValueError, match="missing required field status"):
        exp6471.validate_artifact(missing_field)

    solve_claim = copy.deepcopy(artifact)
    solve_claim["solve_provenance"] = "development_proxy"
    solve_claim["reproducibility_checksum"] = exp6471.payload_checksum(solve_claim)
    with pytest.raises(ValueError, match="solve_provenance"):
        exp6471.validate_artifact(solve_claim)

    missing_principle = copy.deepcopy(artifact)
    missing_principle["field_principles"].pop("status")
    missing_principle["reproducibility_checksum"] = exp6471.payload_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles missing status"):
        exp6471.validate_artifact(missing_principle)

    missing_provenance = copy.deepcopy(artifact)
    missing_provenance["field_provenance"].pop("status")
    missing_provenance["reproducibility_checksum"] = exp6471.payload_checksum(missing_provenance)
    with pytest.raises(ValueError, match="field_provenance missing status"):
        exp6471.validate_artifact(missing_provenance)

    duplicate = copy.deepcopy(artifact)
    duplicate["per_unit_rows"].append(copy.deepcopy(duplicate["per_unit_rows"][0]))
    duplicate["reproducibility_checksum"] = exp6471.payload_checksum(duplicate)
    with pytest.raises(ValueError, match="duplicate per_unit_rows"):
        exp6471.validate_artifact(duplicate)

    protected = copy.deepcopy(artifact)
    first_path = next(iter(protected["protected_files_unchanged"]))
    protected["protected_files_unchanged"][first_path]["unchanged"] = False
    protected["reproducibility_checksum"] = exp6471.payload_checksum(protected)
    with pytest.raises(ValueError, match="protected_files_unchanged"):
        exp6471.validate_artifact(protected)

    checksum = copy.deepcopy(artifact)
    checksum["duration_s"] = checksum["duration_s"] + 1.0
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6471.validate_artifact(checksum)

    ready_gate = copy.deepcopy(artifact)
    ready_gate["arc_safety_shield_ready_score"] = 1.0
    ready_gate["gate_check_summary"]["all_ready_gates_passed"] = False
    ready_gate["reproducibility_checksum"] = exp6471.payload_checksum(ready_gate)
    with pytest.raises(ValueError, match="ready_score gate mismatch"):
        exp6471.validate_artifact(ready_gate)

    assert "unmet gates" in exp6471._honest_verdict(False, False, ["gate"])


def test_scenario_arc_arm_6471_cli_main_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6471-PRECHECK-AND-FREEZE: CLI writes an artifact."""

    rc = exp6471.main(
        [
            "--date",
            "20260819",
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
