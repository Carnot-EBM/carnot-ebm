"""REQ-REPORT-7868-V683: CPU receipt and four-view contract."""

import json
from pathlib import Path

import pytest


def row(text: str = "Café. Next. Same. Next. Café.", answer: str = "Café. More.") -> dict:
    from carnot.verify.source_interventions import digest

    return {
        "family_id": "case-1",
        "complete_source": text,
        "complete_response": answer,
        "source_sha256": digest(text.encode()),
        "response_sha256": digest(answer.encode()),
    }


def reply(witness: int = 2, risk: float = 0.3) -> str:
    return json.dumps({"source_sentence_id": witness, "unsupported_probability": risk})


def test_scenario_report_7868_views_and_bytes() -> None:
    """A nominated ID selects original bytes even when sentence text repeats."""
    from carnot.verify import intervention_protocol_7868 as protocol

    item = row()
    result = protocol.fixture_case(item, reply(), "stop", 68301)
    assert result["status"] == "completed"
    assert result["syntax_valid"] is True
    assert result["source_byte_fidelity"] is True
    assert result["semantic_sensitivity"] is None
    assert result["matching_error"] == 0
    assert len(result["requests"]) == 4
    bodies = [json.loads(request["messages"][1]["content"]) for request in result["requests"]]
    assert [body["visible_source_sentence_ids"] for body in bodies] == [
        [0, 1, 2, 3, 4],
        [2],
        [1, 2, 3],
        [0, 2],
    ] or [body["visible_source_sentence_ids"] for body in bodies] == [
        [0, 1, 2, 3, 4],
        [2],
        [1, 2, 3],
        [2, 4],
    ]
    assert {body["original_answer"] for body in bodies} == {"Café."}
    assert bodies[0]["source_sentence_offsets"][0]["end_byte"] == len("Café. ".encode())
    assert protocol.audit_views(item, result["requests"]) is True
    bodies[1]["complete_source"] = "Wrong. "
    result["requests"][1]["messages"][1]["content"] = json.dumps(bodies[1])
    assert protocol.audit_views(item, result["requests"]) is False


@pytest.mark.parametrize(
    ("response", "finish", "expected", "syntax"),
    [
        ("{", "stop", "invalid_parse", False),
        (reply(), "length", "invalid_parse", False),
        (reply(99), "stop", "invalid_witness", True),
        (json.dumps({"source_sentence_id": 2}), "stop", "invalid_parse", False),
    ],
)
def test_scenario_report_7868_reply_failures(
    response: str, finish: str, expected: str, syntax: bool
) -> None:
    """Syntax and witness range are separate checks with retained outcomes."""
    from carnot.verify import intervention_protocol_7868 as protocol

    result = protocol.fixture_case(row(), response, finish, 68301)
    assert result["status"] == expected
    assert result["syntax_valid"] is syntax
    assert result["source_byte_fidelity"] is True
    assert len(result["requests"]) == 1


def test_scenario_report_7868_exclusions() -> None:
    """Missing disjoint context and incomplete answer remain explicit."""
    from carnot.verify import intervention_protocol_7868 as protocol

    unmatched = protocol.fixture_case(row("First. Middle. Last."), reply(1), "stop", 68301)
    assert unmatched["status"] == "excluded_no_matched_context"
    assert len(unmatched["requests"]) == 3
    incomplete = protocol.fixture_case(row(answer="No full stop"), reply(), "stop", 68301)
    assert incomplete["status"] == "excluded_no_complete_sentence"
    assert incomplete["requests"] == []
    empty = protocol.fixture_case(row(""), reply(), "stop", 68301)
    assert empty["status"] == "excluded_empty_source"


def test_scenario_report_7868_methodology_real_validator(tmp_path: Path) -> None:
    """The exact terminal bytes pass the unchanged methodology detector."""
    from carnot.experiment_7868_v683_intervention_protocol import methodology_fixture
    from carnot.reporting.current_work_receipt import atomic_json
    from scripts.adversarial_verify import verify_artifact

    artifact = tmp_path / "experiment_7868_v683_intervention_protocol.json"
    atomic_json(artifact, methodology_fixture())
    report = verify_artifact(artifact)
    assert not any(flag["kind"] == "METHODOLOGY_MISSING" for flag in report["flags"])


def test_scenario_report_7868_frozen_plan_and_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every current check has one frozen argv and the old failure remains visible."""
    from carnot import experiment_7868_v683_intervention_protocol as exp

    plan = exp.validation_plan()
    assert len(plan["required_checks"]) == len(plan["commands"])
    assert {
        "affected_pytest",
        "coverage_report",
        "full_pytest",
        "adversarial_verify",
        "strict_rows",
    } <= set(plan["required_checks"])
    checks, hashes, rows = exp.preflight()
    assert rows and all(check["passed"] for check in checks)
    assert hashes["prior_exp7854"]["exposure_status"] == "historical"
    assert (
        exp.operand("missing", Path("/tmp/carnot-7868-absent"), "exists", True, False)[
            "artifact_sha256"
        ]
        is None
    )
    monkeypatch.setattr(exp, "PUBLIC", Path("/tmp/carnot-7868-absent"))
    blocked, _, no_rows = exp.preflight()
    assert no_rows == []
    assert any(not check["passed"] for check in blocked)


def test_scenario_report_7868_checkpoint_and_replay(tmp_path: Path) -> None:
    """A second run reuses exact checkpoints and replay rejects changed rows."""
    from carnot import experiment_7868_v683_intervention_protocol as exp

    path = tmp_path / "fixture.json"
    first = exp.fixture_e2e(path, tmp_path / "checkpoints")
    assert first["checkpoint_hits"] == 0
    assert first["model_calls"] == 0
    assert {item["status"] for item in first["rows"]} >= {
        "completed",
        "invalid_parse",
        "invalid_witness",
        "excluded_empty_source",
        "excluded_no_matched_context",
        "excluded_no_complete_sentence",
    }
    second = exp.fixture_e2e(path, tmp_path / "checkpoints")
    assert second["checkpoint_hits"] == 24
    assert exp.cold_replay(path) == {"families": 24, "rows": 24}
    bad = json.loads(path.read_text())
    bad["rows"][0]["status"] = "forged"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="fixture_row_drift"):
        exp.cold_replay(path)
    bad["rows"] = []
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="fixture_count_drift"):
        exp.cold_replay(path)
    family, response, finish = exp.fixture_family(0)
    identity = exp.canonical_hash(
        {"row": family, "reply": response, "finish": finish, "seed": exp.SEED}
    )
    checkpoint = tmp_path / "checkpoints" / f"{identity[7:]}.json"
    saved = json.loads(checkpoint.read_text())
    saved["identity"] = "wrong"
    checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checkpoint_drift"):
        exp.fixture_e2e(path, tmp_path / "checkpoints")


def test_scenario_report_7868_per_arm_primitive_rows(tmp_path: Path) -> None:
    """Every planned arm retains its own status without inventing model calls."""
    from carnot import experiment_7868_v683_intervention_protocol as exp

    fixture = exp.fixture_e2e(tmp_path / "fixture.json", tmp_path / "checkpoints")
    rows = exp.primitive_rows(fixture["rows"])
    assert len(rows) == 96
    assert len({r["family_id"] for r in rows}) == 24
    assert {r["arm"] for r in rows} == {
        "full_source",
        "witness_only",
        "witness_neighbors",
        "matched_control",
    }
    assert all(r["family"] == r["family_id"] and r["seed"] == 68301 for r in rows)
    assert all("status" in r and "source_byte_fidelity" in r for r in rows)
    assert rows[1]["status"] == "constructed"
    assert rows[7]["status"] == "unstarted_invalid_witness"
    assert rows[8]["censored"] is True
    assert rows[23]["status"] == "excluded_no_matched_context"
    candidate = {
        "schema": "carnot.exp7868.intervention_result.v1",
        "rows": rows,
        "fixture_rows_path": str(tmp_path / "fixture.json"),
        "fixture_rows_sha256": exp.sha256_file(tmp_path / "fixture.json"),
    }
    exp.atomic_json(tmp_path / "candidate.json", candidate)
    assert exp.cold_replay(tmp_path / "candidate.json") == {"families": 24, "rows": 96}
    candidate["rows"][0]["status"] = "forged"
    exp.atomic_json(tmp_path / "candidate.json", candidate)
    with pytest.raises(ValueError, match="fixture_row_drift"):
        exp.cold_replay(tmp_path / "candidate.json")


def test_scenario_report_7868_fixture_hash_drift(tmp_path: Path) -> None:
    """Cold replay rejects a fixture whose bytes changed after candidate sealing."""
    from carnot import experiment_7868_v683_intervention_protocol as exp

    fixture_path = tmp_path / "fixture.json"
    fixture = exp.fixture_e2e(fixture_path, tmp_path / "checkpoints")
    candidate_path = tmp_path / "candidate.json"
    exp.atomic_json(
        candidate_path,
        {
            "schema": "carnot.exp7868.intervention_result.v1",
            "rows": exp.primitive_rows(fixture["rows"]),
            "fixture_rows_path": str(fixture_path),
            "fixture_rows_sha256": exp.sha256_file(fixture_path),
        },
    )
    fixture_path.write_bytes(fixture_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fixture_hash_drift"):
        exp.cold_replay(candidate_path)


def test_scenario_report_7868_code_hash_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new protocol checksum cannot reuse a checkpoint from older code."""
    from carnot import experiment_7868_v683_intervention_protocol as exp

    fixture_path = tmp_path / "fixture.json"
    checkpoints = tmp_path / "checkpoints"
    exp.fixture_e2e(fixture_path, checkpoints)
    original_sha256_file = exp.sha256_file
    monkeypatch.setattr(
        exp,
        "sha256_file",
        lambda path: (
            "sha256:changed" if path == Path(exp.protocol.__file__) else original_sha256_file(path)
        ),
    )
    with pytest.raises(ValueError, match="checkpoint_drift"):
        exp.fixture_e2e(fixture_path, checkpoints)


def test_scenario_report_7868_child_sealing_and_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Closed logs are hash named, and command drift fails the dispatcher."""
    import time

    from carnot import experiment_7868_v683_intervention_protocol as exp

    monkeypatch.setattr(exp, "PRIVATE", tmp_path / "private")
    log = tmp_path / "child.log"
    log.write_text("closed child output")
    receipt = {"name": "one", "log_path": str(log), "command_argv": ["one"], "passed": True}
    first = exp.seal_receipt(receipt, 0)
    assert Path(first["log_path"]).read_text() == log.read_text()
    assert exp.seal_receipt(receipt, 0) == first
    Path(first["log_path"]).write_text("drift")
    with pytest.raises(ValueError, match="sealed_log_drift"):
        exp.seal_receipt(receipt, 0)
    monkeypatch.setattr(exp, "seal_receipt", lambda row, _index: row)
    command = {
        "name": "one",
        "argv": [
            "one",
            f"--basetemp={tmp_path / 'base' / 'x'}",
            f"--data-file={tmp_path / 'coverage' / 'c'}",
        ],
        "classification": "required",
        "timeout_s": 2,
    }

    def run(_root: Path, specs: object, **_kwargs: object) -> list[dict]:
        return [{**receipt, "command_argv": list(specs[0].argv)}]

    monkeypatch.setattr(exp, "run_commands", run)
    assert (
        exp.validate({"commands": [command]}, time.monotonic())[0]["classification"] == "required"
    )
    assert (tmp_path / "base").is_dir()
    assert (tmp_path / "coverage").is_dir()
    monkeypatch.setattr(exp, "run_commands", lambda *_args, **_kwargs: [receipt])
    with pytest.raises(ValueError, match="child_command_drift"):
        exp.validate({"commands": [command]}, time.monotonic())


def test_scenario_report_7868_receipt_verdicts(tmp_path: Path) -> None:
    """Blocked inputs, unfinished work and failed children have distinct verdicts."""
    import time

    from carnot import experiment_7868_v683_intervention_protocol as exp

    plan = exp.validation_plan()
    plan["required_checks"] = ["one"]
    start = time.monotonic_ns()
    empty = exp.build_artifact([], {}, [], None, None, plan, [], start, [], False)
    assert empty["verdict_class"] == "partial"
    receipt = {
        "name": "one",
        "classification": "required",
        "passed": True,
        "command_argv": ["true"],
        "log_path": str(tmp_path / "child.log"),
    }
    (tmp_path / "child.log").write_text("done")
    good = exp.build_artifact([], {}, [], None, None, plan, [receipt], start, [], False)
    assert good["verdict_class"] == "circular_positive"
    assert good["model_invocation_counts"]["calls"] == 0
    failed = exp.build_artifact(
        [], {}, [], None, None, plan, [{**receipt, "passed": False}], start, [], False
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["gate_check_summary"][0]["upstream_id"] == "current_validation"
    blocked = exp.build_artifact(
        [exp.operand("up", tmp_path / "missing", "exists", True, False)],
        {},
        [],
        None,
        None,
        plan,
        [],
        start,
        [],
        False,
    )
    assert blocked["verdict_class"] == "blocked"
    flagged = exp.build_artifact([], {}, [], None, None, plan, [receipt], start, [], True)
    assert flagged["verdict_class"] == "disqualified"


@pytest.mark.parametrize("mode", ["good", "failed_terminal", "bad_initial", "bad_final"])
def test_scenario_report_7868_owned_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    """The private run publishes success or a failed exact terminal check."""
    from carnot import experiment_7868_v683_intervention_protocol as exp

    private = tmp_path / "private"
    output = tmp_path / "results" / "experiment_7868_v683_intervention_protocol.json"
    monkeypatch.setattr(exp, "PRIVATE", private)
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", output)
    plan = exp.validation_plan()
    plan["required_checks"] = ["adversarial_verify"]
    monkeypatch.setattr(exp, "validation_plan", lambda: plan)
    adverse = tmp_path / "adverse.log"
    adverse.write_text("bad" if mode == "bad_initial" else '{"flagged_count": 0}')
    final_adverse = tmp_path / "final_adverse.log"
    final_adverse.write_text("bad" if mode == "bad_final" else '{"flagged_count": 0}')
    receipt = {
        "name": "adversarial_verify",
        "classification": "required",
        "passed": True,
        "command_argv": ["verify"],
        "log_path": str(adverse),
    }
    monkeypatch.setattr(exp, "validate", lambda _plan, _started: [receipt])
    monkeypatch.setattr(exp, "seal_receipt", lambda row, _index: row)

    def terminal(_root: Path, _commands: object, **_kwargs: object) -> list[dict]:
        failed = {**receipt, "name": "terminal_strict_rows", "passed": mode != "failed_terminal"}
        return [{**receipt, "name": "terminal_adversarial", "log_path": str(final_adverse)}, failed]

    monkeypatch.setattr(exp, "run_commands", terminal)
    result = exp.run_experiment("20260929")
    assert json.loads(output.read_text()) == result
    bad = mode != "good"
    assert result["verdict_class"] == ("disqualified" if bad else "circular_positive")
    assert result["intervention_protocol_ready_score"] == (0 if bad else 1)
    assert result["sample_size_budget"]["independent"] == 24
    assert Path(result["protocol_manifest_path"]).is_file()
    assert result["semantic_sensitivity"] is None
    if mode == "failed_terminal":
        assert result["gate_check_summary"][-1]["artifact_field"] == "terminal_strict_rows.passed"
    with pytest.raises(ValueError, match="validation_manifest_drift"):
        (tmp_path / "raw" / "validation_command_manifest.json").write_text("{}")
        exp.run_experiment("20260929")


def test_scenario_report_7868_blocked_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An external absence terminates without calling the fixture or a child."""
    from carnot import experiment_7868_v683_intervention_protocol as exp

    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(exp, "PUBLIC", tmp_path / "missing.jsonl")
    result = exp.run_experiment("20260929")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"][0]["artifact_field"] == "exists"
    assert result["rows"] == []


def test_scenario_report_7868_cli_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Actual CLI success and failure return the observed process status."""
    import os
    import subprocess

    from carnot import experiment_7868_v683_intervention_protocol as exp

    script = exp.ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
    env = {
        **os.environ,
        "PYTHONPATH": f"{exp.ROOT / 'python'}:{exp.ROOT}",
        "CARNOT_FORCE_LIVE": "1",
        "JAX_PLATFORMS": "cpu",
    }
    fixture = tmp_path / "cli.json"
    good = subprocess.run(
        [
            str(exp.ROOT / ".venv/bin/python"),
            "-u",
            str(script),
            "--date",
            "20260929",
            "--fixture-e2e",
            str(fixture),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert good.returncode == 0
    replay = subprocess.run(
        [
            str(exp.ROOT / ".venv/bin/python"),
            "-u",
            str(script),
            "--date",
            "20260929",
            "--cold-replay",
            str(fixture),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert replay.returncode == 0
    wrong = subprocess.run(
        [str(exp.ROOT / ".venv/bin/python"), "-u", str(script), "--date", "20260928"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert wrong.returncode != 0 and "run_date_mismatch" in wrong.stderr
    assert exp.main(["--date", "20260929", "--fixture-e2e", str(tmp_path / "direct.json")]) == 0
    assert exp.main(["--date", "20260929", "--cold-replay", str(tmp_path / "direct.json")]) == 0
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.main(["--date", "20260928"])
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.run_experiment("20260928")
    monkeypatch.setattr(exp, "run_experiment", lambda _date: {"verdict_class": "disqualified"})
    assert exp.main(["--date", "20260929"]) == 1
