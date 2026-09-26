"""REQ-REPORT-7722 and REQ-ARC-WMTE-7722 evidence recovery tests."""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from carnot.agentic import arc_evidence_recovery as evidence
from carnot import experiment_7722_v672_arc_evidence_recovery as exp


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/experiment_7709_v671_arc_first_contact"


def test_cold_join_original_history() -> None:
    """SCENARIO-REPORT-7722-RAW-JOIN: all original actions and Qwen bytes join."""
    recovered = evidence.recover_history(ROOT, RAW)
    assert recovered["failed_checks"] == []
    assert [row["game"] for row in recovered["rows"]] == ["wa30", "lf52"]
    assert [row["actions"] for row in recovered["rows"]] == [128, 128]
    assert sum(row["observed_level_ups"] for row in recovered["rows"]) == 0
    assert [row["observed_level_progress_rate"] for row in recovered["rows"]] == [0.0, 0.0]
    assert sum(row["model_calls"] for row in recovered["rows"]) == 4
    assert all(row["censoring"] == "censored_action_limit" for row in recovered["rows"])
    assert all(row["accepted_engines"] == 0 for row in recovered["rows"])
    assert all(row["goal_recall"] == "unknown" for row in recovered["rows"])
    assert recovered["registry_precheck"]["wa30"]["levels_reproduced"] == 9
    assert recovered["registry_precheck"]["lf52"]["levels_reproduced"] == 10


def test_reasoning_only_length_rejection_comes_from_response_bytes() -> None:
    """REQ-ARC-WMTE-7722: parser and truncation reasons need raw response bytes."""
    response = json.loads((RAW / "wa30__live/requests/00_response.json").read_text())
    diagnosis = evidence.classify_response(response)
    assert diagnosis["finish_reason"] == "length"
    assert diagnosis["final_content_chars"] == 0
    assert diagnosis["reasoning_chars"] > 0
    assert diagnosis["completion_tokens"] == 4096
    assert diagnosis["acceptance_reason"] == "reasoning_only_truncated_no_engine_code"


def test_missing_response_is_exact_external_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7722-RAW-JOIN: a missing response cannot be inferred."""
    request = json.loads((RAW / "wa30__live/requests/00_request.json").read_text())
    check = evidence.authenticate_bytes(
        tmp_path / "missing.json", "sha256:absent", "experiment_7709", "response_sha256"
    )
    assert check == {
        "check": "historical_input_bytes",
        "upstream": "experiment_7709",
        "path": str(tmp_path / "missing.json"),
        "field": "response_sha256",
        "operator": "==",
        "expected": "sha256:absent",
        "observed": None,
        "passed": False,
    }
    assert isinstance(request, dict)


def test_supervisor_deduplicates_trace_and_keeps_zero_firings() -> None:
    """REQ-ARC-WMTE-7722: duplicate receipts cannot inflate outcome support."""
    receipt = {
        "trace_id": "one",
        "game": "wa30",
        "firings": 0,
        "resolved_by_levelup": 0,
        "actions_to_levelup": [],
    }
    summary = evidence.summarize_supervisor([receipt, dict(receipt)])
    assert summary["unique_traces"] == 1
    assert summary["duplicate_traces"] == 1
    assert summary["per_game"]["wa30"]["firings"] == 0
    assert summary["per_game"]["wa30"]["resolved_by_levelup"] == 0


def test_terminal_result_keeps_prior_disqualification_and_no_benefit() -> None:
    """SCENARIO-REPORT-7722-TERMINAL: historical evidence is a narrow null."""
    recovered = evidence.recover_history(ROOT, RAW)
    result = exp.build_artifact(recovered, [], "20260926", 1.0)
    assert result["honest_verdict"].startswith("complete_")
    assert result["verdict_class"] in {"null", "disqualified"}
    assert result["prior_verdict"] == "complete_disqualified_required_validation"
    assert result["MODEL_SPECS"] == []
    assert result["model_invoked"] is False
    assert result["inference_substrate_class"] == "no_model_load"
    assert result["new_solve_credit"] is False
    assert result["acceptance_gate_results"]["probability"]["passed"] is None
    assert result["arc_evidence_ready_score"] == 0
    assert result["source_artifact_hashes"]["flagged_historical_evidence"]


def test_failed_validation_disqualifies_readiness() -> None:
    """SCENARIO-REPORT-7722-TERMINAL: failed checks zero the gate."""
    recovered = evidence.recover_history(ROOT, RAW)
    result = exp.build_artifact(
        recovered, [{"name": "focused_pytest", "exit_code": 1}], "20260926", 1.0
    )
    assert result["verdict_class"] == "disqualified"
    assert result["arc_evidence_ready_score"] == 0


def test_recovery_rejects_mutated_observation_join(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7722-RAW-JOIN: a forged SDK observation is detected."""
    rows = json.loads((RAW / "episode_rows.json").read_text())
    rows["rows"][0]["actions"][0]["observation_id"] = "forged"
    with pytest.raises(ValueError, match="observation_join"):
        evidence.join_episode_rows(rows["rows"])


def test_preflight_authenticates_current_source() -> None:
    """REQ-REPORT-7722: current requirements and code bytes are checked."""
    checks, hashes = exp.preflight(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert hashes["python/carnot/agentic/arc_evidence_recovery.py"].startswith("sha256:")
    assert hashes["openspec/capabilities/research-reporting/spec.md"].startswith("sha256:")


def test_build_artifact_pass_and_blocked_gate() -> None:
    """SCENARIO-REPORT-7722-TERMINAL: a passing scope permits only a null."""
    from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

    recovered = evidence.recover_history(ROOT, RAW)
    names = (*REQUIRED_CHECK_NAMES, "e2e_009", "e2e_011", "e2e_013", "e2e_009_smoke")
    receipts = [{"name": name, "exit_code": 0, "passed": True} for name in names]
    result = exp.build_artifact(recovered, receipts, "20260926", 1.0)
    assert result["verdict_class"] == "null"
    assert result["arc_evidence_ready_score"] == 1
    assert result["historical_generalization_report"]["observed_level_ups"] == 0
    assert result["supervisor_outcomes"]["per_game"]["wa30"]["firings"] == 1
    assert result["supervisor_outcomes"]["per_game"]["wa30"]["applied_redirections"] == 0
    assert result["sample_size_budget"]["censored_families"] == 2
    blocked = dict(recovered)
    blocked["checks"] = [
        evidence.authenticate_bytes(ROOT / "absent", "sha256:expected", "external", "sha256")
    ]
    result = exp.build_artifact(blocked, receipts, "20260926", 1.0)
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"]["failed_checks"][0]["observed"] is None
    assert result["arc_evidence_ready_score"] == 0


def test_frozen_validation_and_terminal_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7722: bounded commands use the declared affected scope."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    seen: list[list[str]] = []

    def fake_commands(_root: Path, commands: object, **_kwargs: object) -> list[dict]:
        names = [command.name for command in commands]
        seen.append(names)
        for command in commands:
            for arg in command.argv:
                if arg.startswith("--basetemp="):
                    assert Path(arg.split("=", 1)[1]).parent.is_dir()
        return [{"name": name, "exit_code": 0, "passed": True} for name in names]

    monkeypatch.setattr(scope, "run_commands", fake_commands)
    receipts = exp._validation(ROOT, tmp_path, time.monotonic())
    assert len(receipts) == 13
    assert seen[0][-5:] == ["e2e_009", "e2e_011", "e2e_013", "e2e_009_smoke", "full_python_suite"]
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    terminal = exp._terminal(ROOT, candidate, tmp_path, time.monotonic())
    assert [row["name"] for row in terminal] == [
        "cold_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]


@pytest.mark.parametrize("mode", ["pass", "blocked", "terminal_fail", "resume"])
def test_entrypoint_publishes_exact_disposition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7722-TERMINAL: run and cold CLI use the real raw reducer."""
    from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

    recovered = evidence.recover_history(ROOT, RAW)
    names = (*REQUIRED_CHECK_NAMES, "e2e_009", "e2e_011", "e2e_013", "e2e_009_smoke")
    receipts = [{"name": name, "exit_code": 0, "passed": True} for name in names]
    monkeypatch.setattr(exp, "RAW", Path("raw"))
    monkeypatch.setattr(exp, "preflight", lambda _root: ([], {}))
    if mode == "blocked":
        recovered = dict(recovered)
        recovered["checks"] = [
            evidence.authenticate_bytes(
                tmp_path / "missing", "sha256:missing", "external", "sha256"
            )
        ]
        recovered["failed_checks"] = recovered["checks"]
    monkeypatch.setattr(exp, "recover_history", lambda *_: recovered)
    monkeypatch.setattr(exp, "_validation", lambda *_: receipts)
    if mode == "resume":
        monkeypatch.setattr(exp, "read_suite_debt", lambda *_: {"name": "full_python_suite"})
    terminal = [
        {"name": "cold_reduction", "exit_code": 0, "passed": True},
        {
            "name": "adversarial_verify",
            "exit_code": int(mode == "terminal_fail"),
            "passed": mode != "terminal_fail",
        },
        {"name": "verdict_row_consistency_strict", "exit_code": 0, "passed": True},
    ]
    monkeypatch.setattr(exp, "_terminal", lambda *_: terminal)
    output = tmp_path / "result.json"
    suite_path = (
        ROOT
        / exp.RESULT.parent
        / "raw/experiment_7722_v672_arc_evidence_recovery/validation_receipts_preterminal.json"
    )
    result = exp.run_experiment(
        tmp_path, "20260926", output, suite_debt_path=suite_path if mode == "resume" else None
    )
    assert json.loads(output.read_text())["honest_verdict"] == result["honest_verdict"]
    assert result["verdict_class"] == (
        "blocked" if mode == "blocked" else "disqualified" if mode == "terminal_fail" else "null"
    )
    if mode == "pass":
        monkeypatch.setattr(exp, "recover_history", lambda *_: evidence.recover_history(ROOT, RAW))
        exp.cold_read(tmp_path / "raw/terminal_candidate.json", ROOT)
        assert exp.main(["--cold-read", str(tmp_path / "raw/terminal_candidate.json")]) == 0
        assert "joined_actions" in capsys.readouterr().out


def test_sdk_join_rejects_identity_and_level_mutations() -> None:
    """REQ-ARC-WMTE-7722: SDK identity, uniqueness and level are hard joins."""
    original = json.loads((RAW / "episode_rows.json").read_text())["rows"]
    with pytest.raises(ValueError, match="frozen_game_identity"):
        evidence.join_episode_rows(original[:1])
    duplicate = json.loads(json.dumps(original))
    duplicate[0]["observations"][1]["observation_id"] = duplicate[0]["observations"][0][
        "observation_id"
    ]
    with pytest.raises(ValueError, match="duplicate_observation"):
        evidence.join_episode_rows(duplicate)
    wrong_level = json.loads(json.dumps(original))
    wrong_level[0]["observations"][0]["level"] = 9
    with pytest.raises(ValueError, match="level_observation_mismatch"):
        evidence.join_episode_rows(wrong_level)


def test_missing_archive_and_wrong_schema_are_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7722-RAW-JOIN: missing ledgers and wrong games fail closed."""
    missing = evidence.recover_history(ROOT, tmp_path / "empty")
    assert missing["failed_checks"]
    assert missing["rows"] == []
    raw = tmp_path / "wrong"
    raw.mkdir()
    (raw / "actions.jsonl").symlink_to(RAW / "actions.jsonl")
    (raw / "episode_rows.json").write_text('{"rows": [{"game": "wrong"}]}')
    wrong = evidence.recover_history(ROOT, raw)
    assert wrong["rows"] == []
    assert any(row["check"] == "historical_schema" for row in wrong["failed_checks"])


def test_missing_request_and_supervisor_bytes_are_not_regenerated(tmp_path: Path) -> None:
    """REQ-REPORT-7722: partial history retains exact missing custody."""
    raw = tmp_path / "partial"
    raw.mkdir()
    for name in ("actions.jsonl", "episode_rows.json"):
        (raw / name).symlink_to(RAW / name)
    for game in ("wa30", "lf52"):
        target = raw / f"{game}__live/requests"
        target.mkdir(parents=True)
        for source in (RAW / f"{game}__live/requests").glob("*.json"):
            if source.name != "00_response.json" or game != "wa30":
                (target / source.name).symlink_to(source)
        if game == "lf52":
            seam = raw / f"episodes/{game}/seam_events.jsonl"
            seam.parent.mkdir(parents=True)
            seam.symlink_to(RAW / f"episodes/{game}/seam_events.jsonl")
    partial = evidence.recover_history(ROOT, raw)
    assert len(partial["rows"]) == 2
    assert len(partial["failed_checks"]) == 2
    assert any("00_response.json" in path for path in partial["hashes"]["missing_custody"])
    assert any("seam_events.jsonl" in path for path in partial["hashes"]["missing_custody"])


def test_supervisor_selection_count_rejects_incomplete_trace(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7722: a partial supervisor receipt cannot report zero."""
    seam = tmp_path / "seam.jsonl"
    seam.write_text(
        json.dumps(
            {"seam": "supervisor_arm_selection", "event": "selection", "supervisor_fired": False}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="supervisor_selection_count"):
        evidence._supervisor_receipt(seam, "wa30", [{}, {}])


def test_cold_reader_rejects_mutation_and_missing_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7722-TERMINAL: replay rejects both forged rows and gaps."""
    recovered = evidence.recover_history(ROOT, RAW)
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"rows": []}')
    monkeypatch.setattr(exp, "recover_history", lambda *_: recovered)
    with pytest.raises(ValueError, match="cold_historical_rows_mismatch"):
        exp.cold_read(candidate, ROOT)
    candidate.write_text(
        json.dumps(
            {
                "rows": [
                    {**row, "supervisor": recovered["supervisor"]["per_game"][row["game"]]}
                    for row in recovered["rows"]
                ]
            }
        )
    )
    missing = dict(recovered)
    missing["failed_checks"] = [{"check": "missing"}]
    monkeypatch.setattr(exp, "recover_history", lambda *_: missing)
    with pytest.raises(ValueError, match="cold_historical_inputs_missing"):
        exp.cold_read(candidate, ROOT)


def test_main_normal_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-7722: the thin CLI dispatches the owned run and date."""
    seen: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(
        exp, "run_experiment", lambda root, date, output: seen.append((root, date, output))
    )
    assert exp.main(["--date", "20260926"]) == 0
    assert seen == [(tmp_path, "20260926", tmp_path / exp.RESULT)]


def test_original_pre_gate_receipts_are_authenticated(tmp_path: Path) -> None:
    """REQ-REPORT-7722: missing Exp7708 custody remains an exact block."""
    old = tmp_path / "results/experiment_7709_v671_arc_first_contact.json"
    old.parent.mkdir(parents=True)
    old.symlink_to(ROOT / "results/experiment_7709_v671_arc_first_contact.json")
    registry = tmp_path / "ops/arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True)
    registry.symlink_to(ROOT / "ops/arc_solve_registry.yaml")
    recovered = evidence.recover_history(tmp_path, RAW)
    assert any(row["field"] == "pre_gate_sha256" for row in recovered["failed_checks"])
    assert recovered["hashes"]["missing_custody"]


def test_original_artifact_rows_anchor_raw_history(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7722-RAW-JOIN: consistent raw files cannot rewrite V671."""
    raw = tmp_path / "changed"
    raw.mkdir()
    (raw / "actions.jsonl").symlink_to(RAW / "actions.jsonl")
    rows = json.loads((RAW / "episode_rows.json").read_text())
    rows["rows"][0]["censoring"] = "edited"
    (raw / "episode_rows.json").write_text(json.dumps(rows))
    recovered = evidence.recover_history(ROOT, raw)
    assert any(row["check"] == "original_rows_match" for row in recovered["failed_checks"])


def test_malformed_history_publishes_blocked_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7722-RAW-JOIN: invalid old JSON becomes an exact blocked row."""
    monkeypatch.setattr(exp, "RAW", Path("raw"))
    monkeypatch.setattr(exp, "preflight", lambda _root: ([], {}))
    monkeypatch.setattr(
        exp,
        "recover_history",
        lambda *_: (_ for _ in ()).throw(ValueError("invalid original rows")),
    )
    result = exp.run_experiment(tmp_path, "20260926", tmp_path / "blocked.json")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"]["failed_checks"][0]["field"] == "raw_rows_and_joins"
    assert result["arc_evidence_ready_score"] == 0


def test_resume_suite_debt_requires_original_log_hash(tmp_path: Path) -> None:
    """REQ-REPORT-7722: a prior failed global-suite receipt may be reused exactly once."""
    source = ROOT / exp.RAW / "validation_receipts_preterminal.json"
    receipt = exp.read_suite_debt(source, ROOT)
    assert receipt["name"] == "full_python_suite"
    assert receipt["exit_code"] != 0
    altered = tmp_path / "altered.json"
    altered.write_text(
        json.dumps(
            [
                {
                    "name": "full_python_suite",
                    "exit_code": 2,
                    "command": "forged",
                    "log_path": "/tmp/missing",
                    "log_sha256": "sha256:forged",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="suite_debt_receipt_invalid"):
        exp.read_suite_debt(altered, ROOT)
    altered.write_text("[]")
    with pytest.raises(ValueError, match="suite_debt_receipt_invalid"):
        exp.read_suite_debt(altered, ROOT)


def test_resumed_validation_reuses_global_debt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7722: only scoped checks rerun after one full-suite attempt."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    seen: list[str] = []

    def fake_run(_root: Path, commands: object, **_kwargs: object) -> list[dict]:
        seen.extend(command.name for command in commands)
        return [{"name": name, "exit_code": 0, "passed": True} for name in seen]

    monkeypatch.setattr(scope, "run_commands", fake_run)
    debt = exp.read_suite_debt(ROOT / exp.RAW / "validation_receipts_preterminal.json", ROOT)
    receipts = exp._validation(ROOT, tmp_path, time.monotonic(), debt)
    assert "full_python_suite" not in seen
    assert receipts[-1] == debt


def test_main_resume_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-7722: CLI makes reuse of one hashed failed receipt explicit."""
    seen: list[dict] = []
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(exp, "run_experiment", lambda *_args, **kwargs: seen.append(kwargs))
    path = tmp_path / "receipt.json"
    assert exp.main(["--reuse-full-suite-receipt", str(path)]) == 0
    assert seen == [{"suite_debt_path": path}]
