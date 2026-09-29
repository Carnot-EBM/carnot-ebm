"""Current V678 runner checks for REQ-REPORT-7803 and REQ-ARC-WMTE-7803."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from carnot.experiment_7803_v678_arc_runner_qualification import (
    classify_runner,
    freeze_panel,
    positive_selector_fixture,
)


def _sdk_rows() -> list[dict]:
    return [
        {
            "game": game,
            "arm": arm,
            "error": None,
            "new_solve_credit": False,
            "counts": {"sdk_transitions": 2},
            "policy_entry": {"policy_class": "E3AgentPolicy"},
            "actions": [
                {"induction_attempt_count": 0, "actual_observation": {"frame_sha256": "a"}}
            ],
        }
        for game in ("r11l", "cd82")
        for arm in ("off", "total", "organic")
    ]


def test_scenario_report_7803_gate_requires_all_checks_and_two_games() -> None:
    """SCENARIO-REPORT-7803-GATE: transport and each current check are necessary."""
    required = ("focused_pytest", "cold_reduce", "adversarial_verify")
    receipts = [
        {"name": name, "exit_code": 0, "passed": True, "timed_out": False} for name in required
    ]
    assert classify_runner(receipts, required, _sdk_rows()) == (True, [])
    assert classify_runner(receipts[:-1], required, _sdk_rows()) == (
        False,
        ["adversarial_verify"],
    )
    assert classify_runner([*receipts, receipts[0]], required, _sdk_rows()) == (
        False,
        ["focused_pytest"],
    )
    failed = [*receipts[:-1], dict(receipts[-1], exit_code=1, passed=False)]
    assert classify_runner(failed, required, _sdk_rows()) == (False, ["adversarial_verify"])
    bad = _sdk_rows()[:-1]
    assert classify_runner(receipts, required, bad) == (False, ["sdk_transport"])
    bad = _sdk_rows()
    bad[0]["actions"][0]["induction_attempt_count"] = 1
    assert classify_runner(receipts, required, bad) == (False, ["sdk_transport"])


def test_scenario_arc_wmte_7803_selector_replay_changes_choice() -> None:
    """SCENARIO-ARC-WMTE-7803-SELECTOR: replay cannot masquerade as organic."""
    fixture = positive_selector_fixture()
    assert fixture["off_archive"] is None
    assert fixture["total_prefix"] == [{"action": 2, "data": None}]
    assert fixture["organic_prefix"] == [{"action": 3, "data": None}]
    assert fixture["organic_seen_before_replay"] == fixture["organic_seen_after_replay"]
    assert fixture["replay_seen_after_replay"] > 0


def test_scenario_arc_wmte_7803_panel_is_exact_historical_fixture() -> None:
    """SCENARIO-ARC-WMTE-7803-PANEL: schedule stays frozen before compute."""
    source = Path(
        "results/raw/experiment_7790_v677_arc_runner_qualification/arc_panel_manifest.json"
    )
    panel = freeze_panel(source)
    assert panel["games"] == ["cd82", "dc22", "lf52", "m0r0", "sk48", "tn36", "sb26", "sc25"]
    assert len(panel["rows"]) == 48
    assert {row["status"] for row in panel["rows"]} == {"unstarted"}
    with pytest.raises(ValueError, match="panel_changed"):
        freeze_panel(source, expected_games=("wrong",))


def test_scenario_report_7803_cli_never_calls_historical_main() -> None:
    """SCENARIO-REPORT-7803-CLI: new CLI has no inherited main invocation."""
    script = Path("scripts/experiments/experiment_7803_v678_arc_runner_qualification.py")
    tree = ast.parse(script.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr in {"main", "run_experiment"}
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"previous", "prior", "historical"}
        for call in calls
    )
