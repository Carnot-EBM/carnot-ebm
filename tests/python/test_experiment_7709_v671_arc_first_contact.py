"""REQ-REPORT-7709 and REQ-ARC-WMTE-7709 bounded evidence tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from carnot import experiment_7709_v671_arc_first_contact as exp


def test_schedule_authenticates_exact_two_exp7708_games() -> None:
    """SCENARIO-REPORT-7709-BLOCKED: a mutated schedule cannot open the gate."""
    schedule = {
        "selection_salt": "v671-arc-20260926",
        "rows": [
            {"game": "wa30", "adapter_withheld": True, "withheld_inputs": list(exp.WITHHELD)},
            {"game": "lf52", "adapter_withheld": True, "withheld_inputs": list(exp.WITHHELD)},
        ],
    }
    assert exp.schedule_check(schedule)["passed"] is True
    schedule["rows"][1]["adapter_withheld"] = False
    check = exp.schedule_check(schedule)
    assert check["passed"] is False
    assert check["field"] == "rows"
    assert check["upstream"] == "experiment_7708"


def test_goal_support_requires_reachable_firing_and_sdk_progress() -> None:
    """SCENARIO-ARC-WMTE-7709-GOAL-JOIN: outcome truth comes from SDK levels."""
    attempts = [
        {"attempt_id": "a", "goal_predicate": "g1", "accepted": True},
        {"attempt_id": "b", "goal_predicate": "g2", "accepted": False},
    ]
    actions = [
        {"action_index": 1, "level_before": 0, "level_after": 0, "goal_firings": ["a"]},
        {"action_index": 2, "level_before": 0, "level_after": 1, "goal_firings": []},
    ]
    support = exp.goal_support(attempts, actions)
    assert support[0]["reachable_firing"] is True
    assert support[0]["sdk_confirmed_level_progress"] is True
    assert support[1]["reachable_firing"] is False
    assert support[1]["recall"] == "unknown"


def test_cold_reduction_rejects_request_and_observation_gaps(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7709-REDUCE: every current call and action must join."""
    rows = [
        {
            "episode_id": f"{game}:live",
            "game": game,
            "arm": "adapter_withheld",
            "actions": [{"action_index": 1, "observation_id": f"{game}:o1"}],
            "observations": [{"observation_id": f"{game}:o1", "level": 0}],
            "requests": [],
            "induction_attempts": [],
            "censoring": "action_limit",
            "exclusions": [],
        }
        for game in ("wa30", "lf52")
    ]
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps({"rows": rows}))
    reduction = exp.cold_reduce(raw)
    assert reduction["joined_actions"] == 2
    assert reduction["observed_games"] == 2
    assert reduction["goal_recall"] == "unknown"
    rows[0]["actions"][0]["observation_id"] = "missing"
    raw.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError, match="observation_join"):
        exp.cold_reduce(raw)


def test_blocked_artifact_has_exact_gate_and_no_current_model() -> None:
    """SCENARIO-REPORT-7709-BLOCKED: absent upstream is terminal, not partial."""
    check = exp.gate_check(
        "exp7708_ready", "experiment_7708", "results/7708.json", "arc_runner_ready_score", 1, 0
    )
    artifact = exp.build_artifact(
        checks=[check],
        rows=[],
        reduction=None,
        hashes={"missing_custody": ["results/7708.json"]},
        receipts=[],
        duration_s=1.0,
        run_date="20260926",
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_checks"] == [check]
    assert artifact["MODEL_SPECS"] == []
    assert artifact["planned_MODEL_SPECS"] == [exp.MODEL_ID]
    assert artifact["invocation_counts"]["generations"] == 0
    assert artifact["acceptance_gate_results"]["readiness"]["passed"] is False


def test_two_public_games_cannot_make_probability_or_utility_claim() -> None:
    """REQ-REPORT-7709: observed level changes remain descriptive."""
    rows = [
        {
            "episode_id": f"{game}:live",
            "game": game,
            "arm": "adapter_withheld",
            "actions": [],
            "observations": [],
            "requests": [],
            "induction_attempts": [],
            "censoring": None,
            "exclusions": [],
            "peak_level": level,
        }
        for game, level in (("wa30", 1), ("lf52", 0))
    ]
    reduction = exp.reduce_rows(rows)
    artifact = exp.build_artifact(
        checks=[],
        rows=rows,
        reduction=reduction,
        hashes={},
        receipts=[{"name": "required", "exit_code": 0}],
        duration_s=100.0,
        run_date="20260926",
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_measurement_complete_score"] == 1
    assert artifact["acceptance_gate_results"]["probability"]["passed"] is False
    assert artifact["acceptance_gate_results"]["utility"]["passed"] is False
    assert artifact["new_solve_credit"] is False


def test_current_preflight_carries_authenticated_gate_and_resource_operands() -> None:
    """SCENARIO-REPORT-7709-BLOCKED: inspect bytes and CUDA without loading Qwen."""
    checks, hashes, context = exp.collect_preconditions(exp.ROOT, time.monotonic())
    by_name = {row["check"]: row for row in checks}
    assert by_name["exp7708_arc_runner_ready_score"]["passed"] is True
    assert by_name["exp7708_schedule"]["passed"] is True
    assert by_name["mandated_qwen_cache"]["passed"] is True
    assert by_name["exclusive_cuda_capacity"]["field"] == "exclusive_device_available"
    assert hashes["pre_gate_receipts"][str(exp.SCHEDULE)].startswith("sha256:")
    assert context["model_path"].is_file()


def test_frozen_validation_and_terminal_commands_are_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-7709: affected scope and exact-candidate readers stay fixed."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    seen: list[list[str]] = []

    def fake_run_commands(_root: Path, commands: object, **_kwargs: object) -> list[dict]:
        names = [command.name for command in commands]
        seen.append(names)
        return [{"name": name, "exit_code": 0, "passed": True} for name in names]

    monkeypatch.setattr(scope, "run_commands", fake_run_commands)
    receipts = exp._validation(exp.ROOT, tmp_path, time.monotonic())
    assert len(receipts) == 12
    assert seen[0][-4:] == ["e2e_009", "e2e_011", "e2e_013", "e2e_009_smoke"]
    assert "changed_module_coverage_report" in seen[0]
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    terminal = exp._terminal(exp.ROOT, candidate, tmp_path, time.monotonic())
    assert [row["name"] for row in terminal] == [
        "cold_reduction", "adversarial_verify", "verdict_row_consistency_strict"
    ]


def test_main_cold_read_rejects_mutated_raw_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7709-REDUCE: a fresh process must see exact raw rows."""
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(exp, "RAW", Path("raw"))
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/episode_rows.json").write_text('{"rows": []}')
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"rows": [], "reduction": null}')
    assert exp.main(["--cold-read", str(candidate)]) == 0
    candidate.write_text('{"rows": [{"game": "forged"}], "reduction": null}')
    with pytest.raises(ValueError, match="cold_raw_rows_mismatch"):
        exp.main(["--cold-read", str(candidate)])


@pytest.mark.parametrize("blocked", [True, False])
def test_orchestration_publishes_exact_current_disposition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blocked: bool
) -> None:
    """REQ-REPORT-7709: an external gate closes blocked; two saved rows stay null."""
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(exp, "RAW", Path("raw"))
    calls: list[str] = []
    check = exp.gate_check("external_gate", "producer", "producer.json", "ready", True, not blocked)

    def preflight(_root: Path, _started: float) -> tuple[list[dict], dict, dict]:
        calls.append("preflight")
        return [check], {"producer_files": {}, "pre_gate_receipts": {}, "missing_custody": []}, {}

    def live(_root: Path, _context: dict, _started: float) -> tuple[list[dict], dict]:
        calls.append("live")
        rows = [
            {"episode_id": f"{game}:live", "game": game, "arm": "adapter_withheld",
             "actions": [], "observations": [], "requests": [], "induction_attempts": [],
             "censoring": "censored_action_limit", "exclusions": [], "peak_level": 0}
            for game in ("wa30", "lf52")
        ]
        exp.atomic_json(tmp_path / "raw/episode_rows.json", {"rows": rows})
        return rows, {"loads_attempted": 1, "loads_completed": 1, "gpu_uuid": "GPU-fixture",
                      "model_path": "/tmp/model", "gguf_sha256": "sha256:model",
                      "server_path": "/tmp/server", "server_sha256": "sha256:server"}

    monkeypatch.setattr(exp, "collect_preconditions", preflight)
    monkeypatch.setattr(exp, "run_live", live)
    monkeypatch.setattr(exp, "_validation", lambda *_: [{"name": "required", "exit_code": 0}])
    monkeypatch.setattr(exp, "_terminal", lambda *_: [
        {"name": "cold_reduction", "exit_code": 0},
        {"name": "adversarial_verify", "exit_code": 0},
        {"name": "verdict_row_consistency_strict", "exit_code": 0},
    ])
    output = tmp_path / "out.json"
    result = exp.run_experiment(tmp_path, "20260926", output)
    assert json.loads(output.read_text())["honest_verdict"] == result["honest_verdict"]
    assert result["verdict_class"] == ("blocked" if blocked else "null")
    assert calls == (["preflight"] if blocked else ["preflight", "live"])
    assert (tmp_path / "raw/validation_receipts.json").is_file()
