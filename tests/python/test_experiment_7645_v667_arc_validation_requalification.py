"""REQ-REPORT-7645 CPU checks for ARC goal validation requalification."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from carnot import experiment_7645_v667_arc_validation_requalification as exp
from carnot.agentic import arc_executable_world_model as e3
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.current_work_receipt import sha256_file


@pytest.mark.parametrize(
    ("case", "mask", "states", "terminal", "expected"),
    [
        ("empty", [[False, False]], [[0, 7]], 7, True),
        ("full", [[True, True]], [[0, 7]], 7, True),
        ("terminal_alias", [[False, True]], [[0, 1], [0, 7]], 7, True),
        ("true_alias", [[False, True]], [[0, 1], [0, 2]], 7, False),
        ("ordinary_duplicate", [[False, False]], [[0, 1], [0, 1]], 7, False),
    ],
)
def test_goal_before_duplicate_in_live_planner(
    monkeypatch, case, mask, states, terminal, expected
) -> None:
    """SCENARIO-REPORT-7645-REGRESSION tests full-grid goal before duplicate skip."""
    monkeypatch.setattr(
        e3,
        "_model_candidates",
        lambda _grid: [{"action": index + 1, "data": None} for index in range(len(states))],
    )
    goal_calls = []

    def engine(_grid, action, _data):
        return np.asarray(states[action - 1], dtype=np.int16).reshape(1, 2)

    def goal(grid):
        goal_calls.append(np.asarray(grid).tolist())
        return int(grid[0, 1]) == terminal

    diagnostics = {}
    plan = e3.plan_in_model(
        engine,
        goal,
        np.asarray([[0, 0]], dtype=np.int16),
        dedup_mask=np.asarray(mask),
        diagnostics=diagnostics,
        max_nodes=len(states),
        max_depth=1,
    )
    assert bool(plan) is expected, case
    if case == "terminal_alias":
        assert goal_calls[:2] == [[[0, 1]], [[0, 7]]]
        assert diagnostics["termination_reason"] == "plan_found"
    if case == "true_alias":
        assert diagnostics["hud_dedup_states_merged"] >= 1


def test_measured_rows_and_stage2_mask() -> None:
    """SCENARIO-REPORT-7645-REGRESSION records actual mask dimensions and E3 replay."""
    rows, stage2 = exp.measure_goal_guard_rows()
    assert {row["unit_id"] for row in rows} >= {
        "terminal_alias",
        "empty_mask",
        "full_mask",
        "true_alias",
        "ordinary_duplicate",
    }
    assert all(row["passed"] for row in rows)
    assert stage2["frame_shape"] == [64, 64]
    assert stage2["logical_shape"] == [64, 64]
    assert stage2["mask_cells"] == 64
    assert stage2["source"] == "edge_bar_detector_req5960_stage2_confirmed"
    assert any(row["route"] == "E3AgentPolicy._call_plan_in_model" for row in rows)


def test_validation_parent_and_closed_venue(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7645-VALIDATION creates pytest parent and rejects dict venue."""
    root = Path(__file__).resolve().parents[2]
    private = tmp_path / "new-private"
    commands = exp.build_validation_commands(root, private)
    assert (private / "pytest").is_dir()
    assert (private / "coverage").is_dir()
    assert any("--basetemp=" in " ".join(row.argv) for row in commands)
    artifact = exp.build_artifact([], {}, [], [], {}, 0.1, [])
    assert artifact["execution_venue"] == "host"
    assert artifact["execution_venue_details"]["owned_pid"] > 0
    assert exp.validate_artifact(artifact) == []
    invalid = copy.deepcopy(artifact)
    invalid["execution_venue"] = {"host": "unclosed"}
    assert "execution_venue" in exp.validate_artifact(invalid)


def test_custody_and_cold_reduction(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7645-CUSTODY keeps readiness distinct from benefit."""
    rows, stage2 = exp.measure_goal_guard_rows()
    artifact = exp.build_artifact(
        rows,
        stage2,
        [{"passed": True}],
        [],
        {},
        1.0,
        [{"name": "focused_pytest", "exit_code": 0, "passed": True}],
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["planner_goal_guard_ready_score"] == 1
    assert artifact["acceptance_gate_results"]["probability_benefit"]["passed"] is False
    assert artifact["sample_size_budget"]["observed_independent_groups"] == len(
        {row["unit_id"] for row in rows}
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, artifact)
    assert exp.cold_replay(candidate) == []
    changed = json.loads(candidate.read_text())
    changed["rows"][0]["passed"] = False
    atomic_json(candidate, changed)
    assert exp.cold_replay(candidate)


def test_blocked_operand_is_exact() -> None:
    """REQ-REPORT-7645 makes an absent external input a complete blocked result."""
    missing = {
        "check": "source_input",
        "upstream": "producer",
        "path": "x.json",
        "field": "is_file",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    artifact = exp.build_artifact([], {}, [missing], [], {}, 0.1, [])
    assert artifact["honest_verdict"] == "complete_blocked_source_input"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["observed"] is False


def test_preconditions_authenticate_inputs_and_keep_output_planned(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7645-VALIDATION checks real files, never the new result."""
    root = Path(__file__).resolve().parents[2]
    old_hash = sha256_file(root / exp.OLD_RESULT)
    checks, hashes = exp.collect_preconditions(root)
    assert all(row["passed"] for row in checks)
    assert exp.RESULT.as_posix() in hashes["planned_outputs_not_inputs"]
    assert exp.RESULT.as_posix() not in hashes["producer_files"]
    assert sha256_file(root / exp.OLD_RESULT) == old_hash
    missing, missing_hashes = exp.collect_preconditions(tmp_path)
    assert missing[0]["observed"] is False
    assert "AGENTS.md" in missing_hashes["missing_inputs"]
    assert missing[-2]["passed"] is False


def test_reader_mutations_and_command_manifest(tmp_path: Path, capsys) -> None:
    """SCENARIO-REPORT-7645-VALIDATION checks closed schema and exact readers."""
    artifact = exp.build_artifact([], {}, [], [], {}, 0.1, [])
    exp.progress(time.monotonic(), "fixture", "end", unit=1)
    assert "phase=fixture" in capsys.readouterr().out
    manifest = exp.affected_validation_manifest()
    assert manifest["coverage_required_percent"] == 100
    assert manifest["tests"] == [exp.TEST.as_posix()]
    commands = exp.build_terminal_commands(Path.cwd(), tmp_path / "candidate.json")
    assert {row.name for row in commands} == {
        "declared_entrypoint_validate",
        "fresh_process_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    assert exp.parse_args(["--date", exp.RUN_DATE]).output == exp.RESULT
    invalid = copy.deepcopy(artifact)
    invalid["honest_verdict"] = "unfinished"
    invalid["MODEL_SPECS"] = ["model"]
    invalid["rows"] = [{"unit_id": "mutated", "passed": True}]
    invalid.pop("reproducibility_checksum")
    assert {
        "honest_verdict",
        "model_declaration",
        "goal_guard_rows",
        "independent_reduction",
        "reproducibility_checksum",
    } <= set(exp.validate_artifact(invalid))


def test_unresolved_stage2_and_env_restoration(monkeypatch) -> None:
    """SCENARIO-REPORT-7645-REGRESSION rejects absent live masks and restores flags."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "prior")
    _, stage2 = exp.measure_goal_guard_rows()
    assert stage2["mask_cells"] == 64
    assert exp.os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] == "prior"

    class NoMaskExplorer:
        hud_mask = None

        def __init__(self, **_kwargs):
            pass

        def _ingest(self, _frame):
            pass

    monkeypatch.setattr(exp.agent, "StepwiseExplorer", NoMaskExplorer)
    with pytest.raises(RuntimeError, match="stage2_mask_unresolved"):
        exp.measure_goal_guard_rows()
