"""Tests for Exp 4518 canonical ARC local submission metric harness.

Spec refs: REQ-ARC-FCP-4518, SCENARIO-ARC-FCP-4518.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4518_metric_harness_canonical as exp4518


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _baseline() -> dict[str, object]:
    return {
        "policy": "e3",
        "games": list(exp4518.CANONICAL_GAME_SET),
        "action_metric": {
            "field": "actions",
            "definition": "total_actions_on_solved_games",
        },
        "solved_count": 4,
        "solved_games": list(exp4518.CANONICAL_CORE_GAMES),
        "actions_by_game": {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731},
        "median_actions_on_solved": 7760.0,
        "per_game": [
            {"game": "lp85", "solved": True, "actions": 7792},
            {"game": "m0r0", "solved": True, "actions": 7789},
            {"game": "sp80", "solved": True, "actions": 7724},
            {"game": "vc33", "solved": True, "actions": 7731},
            {"game": "cd82", "solved": False, "actions": 7799},
            {"game": "ft09", "solved": False, "actions": 7963},
            {"game": "su15", "solved": False, "actions": 7786},
            {"game": "ls20", "solved": False, "actions": 7821},
        ],
    }


def _dashboard_row() -> dict[str, object]:
    return {
        "lever": "positive_control",
        "metric_action_field": "actions",
        "median_actions_on_core": 7150.0,
        "baseline_median_actions_on_core": 7760.0,
        "actions_saved_vs_baseline": 610.0,
        "core_solves_preserved": True,
        "bonus_solves": ["ft09"],
        "verdict_pass": True,
        "verdict": "PASS (IMPROVED): fixture",
    }


def test_req_arc_fcp_4518_spec_declares_canonical_metric_contract() -> None:
    """REQ-ARC-FCP-4518: OpenSpec anchors the canonical harness artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4518" in spec
    assert "SCENARIO-ARC-FCP-4518" in spec
    assert exp4518.RESULT_RELATIVE_PATH in spec
    assert "actions_to_first_levelup" in spec
    assert "total `actions`" in spec
    for field, principle in exp4518.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4518_builds_terminal_artifact_with_required_fields() -> None:
    """REQ-ARC-FCP-4518: artifact schema records guards, tests, and sample row."""
    artifact = exp4518.build_artifact(
        preconditions_checked={
            "agents_md_read": True,
            "codex_md_read": True,
            "offline_arcade_precondition": True,
            "gate_help_precondition": True,
        },
        baseline=_baseline(),
        baseline_guard={"ok": True, "errors": []},
        headroom={
            "selected_default_budget": 8000,
            "measured": True,
            "rows": [
                {
                    "budget": 8000,
                    "comparison_budget": 12000,
                    "solved_games": list(exp4518.CANONICAL_CORE_GAMES),
                    "comparison_solved_games": list(exp4518.CANONICAL_CORE_GAMES),
                    "stable_vs_1_5x": True,
                }
            ],
        },
        positive_control={"passed": True, "dashboard_row": _dashboard_row()},
        sample_dashboard_row=_dashboard_row(),
        tests_added_pass={
            "command": ".venv/bin/pytest tests/python/test_arc_submission_gate_verdict.py "
            "tests/python/test_experiment_4518_metric_harness_canonical.py -q --no-cov",
            "passed": True,
        },
        duration_s=0.25,
    )

    assert exp4518.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "shipped: metric_harness_canonical_ci_guarded"
    assert artifact["canonical_game_set"] == list(exp4518.CANONICAL_GAME_SET)
    assert artifact["canonical_baseline"]["median_actions_on_core"] == 7760.0
    assert artifact["positive_control_passed"] is True
    assert artifact["sample_dashboard_row"]["actions_saved_vs_baseline"] == 610.0


def test_scenario_arc_fcp_4518_run_writes_json_with_injected_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4518: run() writes the terminal artifact from gate helpers."""

    class FakeGate:
        CANONICAL_GAME_SET = exp4518.CANONICAL_GAME_SET
        CANONICAL_BASELINE_MEDIAN_ACTIONS = 7760.0
        DEFAULT_BUDGET = 8000

        @staticmethod
        def validate_canonical_baseline(_baseline: dict[str, object]) -> dict[str, object]:
            return {"ok": True, "errors": []}

        @staticmethod
        def positive_control(_baseline: dict[str, object]) -> dict[str, object]:
            return {"passed": True, "dashboard_row": _dashboard_row()}

        @staticmethod
        def dashboard_row(
            _cur: dict[str, object],
            _base: dict[str, object],
            *,
            lever: str,
        ) -> dict[str, object]:
            row = _dashboard_row()
            row["lever"] = lever
            return row

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc-submission-baseline.json").write_text(
        json.dumps(_baseline()),
        encoding="utf-8",
    )

    artifact = exp4518.run(
        root=tmp_path,
        gate=FakeGate,
        write=True,
        preconditions_checked={
            "agents_md_read": True,
            "codex_md_read": True,
            "offline_arcade_precondition": True,
            "gate_help_precondition": True,
        },
        measure_headroom=lambda *_args, **_kwargs: {
            "selected_default_budget": 8000,
            "measured": True,
            "rows": [
                {
                    "budget": 8000,
                    "comparison_budget": 12000,
                    "solved_games": list(exp4518.CANONICAL_CORE_GAMES),
                    "comparison_solved_games": list(exp4518.CANONICAL_CORE_GAMES),
                    "stable_vs_1_5x": True,
                }
            ],
        },
        tests_added_pass={"command": "fixture", "passed": True},
        now=lambda: 1.0,
    )

    out = tmp_path / exp4518.RESULT_RELATIVE_PATH
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == artifact
    assert artifact["positive_control_passed"] is True
    assert artifact["canonical_default_budget"] == 8000
