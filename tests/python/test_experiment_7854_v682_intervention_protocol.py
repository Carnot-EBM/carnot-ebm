"""REQ-REPORT-7854-V682: frozen requests and honest fixture receipts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def family(index: int = 0, source: str | None = None) -> dict[str, str]:
    """Give each fixture a distinct source without importing model weights."""
    from carnot.verify.source_interventions import digest

    text = (
        source
        if source is not None
        else (
            f"Café {index} first. Second {index} fact. Third {index} fact. "
            f"Fourth {index} fact. Fifth {index} fact."
        )
    )
    answer = f"Café {index} first. This must not enter the request."
    return {
        "family_id": f"fixture-{index:02d}",
        "complete_source": text,
        "complete_response": answer,
        "source_sha256": digest(text.encode()),
        "response_sha256": digest(answer.encode()),
    }


def tokenizer(text: str) -> int:
    """Count fixture words through the injected tokenizer interface."""
    return len(text.split())


def reply(payload: dict[str, object], *, bad: str | None = None) -> dict[str, object]:
    """Nominate the visible original ID in a scripted JSON completion."""
    content = json.loads(payload["messages"][1]["content"])  # type: ignore[index]
    ids = content["visible_source_sentence_ids"]
    witness = 2 if content["arm"] == "full_source" else ids[0]
    text = json.dumps({"unsupported_probability": 0.2, "source_sentence_id": witness})
    return {
        "model": payload["model"],
        "choices": [
            {
                "message": {"content": "{" if bad == "json" else text},
                "finish_reason": "length" if bad == "length" else "stop",
            }
        ],
        "usage": {"completion_tokens": 7},
    }


def test_scenario_report_7854_requests_exact_views() -> None:
    """Four arms copy authenticated original bytes and the first answer sentence."""
    from carnot.verify import context_sufficiency_7854 as context

    row = family()
    frozen = context.freeze_protocol(seed=68201)
    requests = [context.build_request(row, arm, 2, 4, frozen, tokenizer) for arm in context.ARMS]
    bodies = [json.loads(request["messages"][1]["content"]) for request in requests]
    assert [body["arm"] for body in bodies] == list(context.ARMS)
    assert [body["visible_source_sentence_ids"] for body in bodies] == [
        [0, 1, 2, 3, 4],
        [2],
        [1, 2, 3],
        [2, 4],
    ]
    assert all(body["original_answer"] == "Café 0 first." for body in bodies)
    assert all(request["max_tokens"] == 128 for request in requests)
    assert bodies[0]["complete_source"] == row["complete_source"]
    assert bodies[1]["complete_source"] == "Third 0 fact. "
    assert bodies[3]["complete_source"] == "Third 0 fact. Fifth 0 fact."
    assert frozen["maximum_model_calls"] == 192
    assert frozen["cache_resolver"] == "carnot.inference.sota_models.cached_current_model"


def test_scenario_report_7854_bad_requests_and_replies() -> None:
    """Malformed output and invalid source IDs never become a valid nomination."""
    from carnot.verify import context_sufficiency_7854 as context

    row = family(source="Repeated. Repeated. Middle. Repeated. Repeated.")
    frozen = context.freeze_protocol(seed=68201)
    request = context.build_request(row, "full_source", None, None, frozen, tokenizer)
    assert json.loads(request["messages"][1]["content"])["visible_source_sentence_ids"] == [
        0,
        1,
        2,
        3,
        4,
    ]
    for bad in ("json", "length"):
        assert (
            context.parse_response(reply(request, bad=bad), request["model"], [0, 1, 2, 3, 4])[
                "status"
            ]
            == "invalid_parse"
        )
    wrong = {**reply(request), "model": "wrong"}
    assert (
        context.parse_response(wrong, request["model"], [0, 1, 2, 3, 4])["status"] == "wrong_model"
    )
    assert (
        context.parse_response({"model": request["model"]}, request["model"], [0])["status"]
        == "invalid_parse"
    )
    with pytest.raises(ValueError, match="invalid_witness"):
        context.build_request(row, "witness_only", 99, None, frozen, tokenizer)
    with pytest.raises(ValueError, match="invalid_control"):
        context.build_request(row, "matched_control", 2, 2, frozen, tokenizer)
    with pytest.raises(ValueError, match="unplanned_arm"):
        context.build_request(row, "invented", 2, 4, frozen, tokenizer)
    frozen["context_ceiling_tokens"] = 1
    with pytest.raises(ValueError, match="context_budget"):
        context.build_request(row, "full_source", None, None, frozen, tokenizer)


def test_scenario_report_7854_family_failures() -> None:
    """The dispatcher keeps failure, exclusion and censoring rows in place."""
    from carnot.verify import context_sufficiency_7854 as context

    frozen = context.freeze_protocol(seed=68201)
    normal = context.capture_family(family(), frozen, reply, tokenizer)
    assert [item["status"] for item in normal] == ["completed"] * 4
    assert normal[0]["source_sentence_id"] == 2
    assert normal[3]["control_sentence_id"] in {0, 4}
    assert all(item["request_sha256"].startswith("sha256:") for item in normal)
    assert all(item["request_bytes"] for item in normal)
    assert [item["arm"] for item in normal] == list(context.ARMS)

    missing = context.capture_family(family(source=""), frozen, reply, tokenizer)
    assert all(item["status"] == "excluded_empty_source" for item in missing)
    assert all(item["excluded"] for item in missing)
    malformed = context.capture_family(
        family(), frozen, lambda payload: reply(payload, bad="json"), tokenizer
    )
    assert malformed[0]["status"] == "invalid_parse"
    assert all(item["status"] == "unstarted_invalid_witness" for item in malformed[1:])
    truncated = context.capture_family(
        family(), frozen, lambda payload: reply(payload, bad="length"), tokenizer
    )
    assert truncated[0]["status"] == "invalid_parse"
    invalid = context.capture_family(
        family(),
        frozen,
        lambda payload: {
            **reply(payload),
            "choices": [
                {
                    "message": {
                        "content": '{"unsupported_probability":0.2,"source_sentence_id":99}'
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        tokenizer,
    )
    assert invalid[0]["status"] == "invalid_witness"

    def middle_reply(payload: dict[str, object]) -> dict[str, object]:
        result = reply(payload)
        if json.loads(payload["messages"][1]["content"])["arm"] == "full_source":  # type: ignore[index]
            result["choices"][0]["message"]["content"] = (  # type: ignore[index]
                '{"unsupported_probability":0.2,"source_sentence_id":1}'
            )
        return result

    narrow = context.capture_family(
        family(source="One. Two. Three."), frozen, middle_reply, tokenizer
    )
    assert narrow[3]["status"] == "excluded_no_matched_context"
    assert narrow[3]["excluded"]
    timed = context.capture_family(
        family(), frozen, lambda payload: (_ for _ in ()).throw(TimeoutError()), tokenizer
    )
    assert timed[0]["status"] == "timeout" and timed[0]["censored"]
    interrupted = context.capture_family(
        family(), frozen, lambda payload: (_ for _ in ()).throw(InterruptedError()), tokenizer
    )
    assert interrupted[0]["status"] == "interrupted" and interrupted[0]["censored"]
    frozen["context_ceiling_tokens"] = 1
    overlong = context.capture_family(family(), frozen, reply, tokenizer)
    assert all(item["status"] == "excluded_context_budget" for item in overlong)


def test_scenario_report_7854_fixture_restart_and_cold_replay(tmp_path: Path) -> None:
    """Twenty-four families resume by hash and replay every captured request byte."""
    from carnot import experiment_7854_v682_intervention_protocol as run

    path = tmp_path / "fixture.json"
    checkpoint = tmp_path / "checkpoints"
    first = run.fixture_e2e(path, checkpoint)
    assert first["independent_families"] == 24
    assert len(first["rows"]) == 96
    assert {item["status"] for item in first["rows"]} >= {
        "completed",
        "invalid_parse",
        "timeout",
        "excluded_empty_source",
    }
    assert run.cold_replay(path)["rows"] == 96
    second = run.fixture_e2e(path, checkpoint)
    assert second["checkpoint_hits"] == 24
    changed = json.loads(path.read_text())
    changed["rows"][0]["request_sha256"] = "sha256:wrong"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="request_drift"):
        run.cold_replay(path)


def test_scenario_report_7854_request_custody_errors() -> None:
    """Source or answer changes and invalid control IDs fail before transport."""
    from carnot.verify import context_sufficiency_7854 as context

    row = family()
    frozen = context.freeze_protocol(seed=68201)
    assert context.choose_control(row, 99, tokenizer, seed=68201) is None
    changed = {**row, "complete_source": "Changed."}
    with pytest.raises(ValueError, match="source_drift"):
        context.build_request(changed, "full_source", None, None, frozen, tokenizer)
    changed = {**row, "complete_response": "Changed."}
    with pytest.raises(ValueError, match="answer_drift"):
        context.build_request(changed, "full_source", None, None, frozen, tokenizer)
    with pytest.raises(ValueError, match="invalid_control"):
        context.build_request(row, "matched_control", 2, 99, frozen, tokenizer)


def test_scenario_report_7854_preflight_and_natural_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """External gates authenticate all 64 rows before selecting 48 families."""
    from carnot import experiment_7854_v682_intervention_protocol as run

    public, checks, hashes = run.preflight()
    assert len(public) == 64 and all(item["passed"] for item in checks)
    assert hashes["historical"]["exposure_status"] == "historical"
    rows, frozen = run.prepare_sources(public, tmp_path / "protocol.json")
    assert len(rows) == 192 and frozen["maximum_model_calls"] == 192
    assert len({item["family_id"] for item in rows}) == 48
    with pytest.raises(ValueError, match="family_count_drift"):
        run.prepare_sources(public[:2], tmp_path / "invalid.json")
    monkeypatch.setattr(run, "EXTERNAL", {**run.EXTERNAL, "public": tmp_path / "missing.jsonl"})
    blocked, failed, _ = run.preflight()
    assert blocked == [] and any(not item["passed"] for item in failed)


def test_scenario_report_7854_cold_mutations(tmp_path: Path) -> None:
    """Cold replay detects source, answer, response and parser changes."""
    from carnot import experiment_7854_v682_intervention_protocol as run
    from carnot.verify.source_interventions import digest

    path = tmp_path / "fixture.json"
    run.fixture_e2e(path, tmp_path / "checkpoints")
    original = json.loads(path.read_text())
    edits = [
        (lambda data: data["rows"].pop(), "fixture_count_drift"),
        (lambda data: data["families"][0].update(complete_source="wrong"), "source_drift"),
        (lambda data: data["families"][0].update(complete_response="wrong"), "answer_drift"),
        (lambda data: data["rows"][0].update(response_sha256="sha256:wrong"), "response_drift"),
        (lambda data: data["rows"][0].update(status="wrong"), "parser_drift"),
    ]
    for edit, reason in edits:
        data = json.loads(json.dumps(original))
        edit(data)
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match=reason):
            run.cold_replay(path)
    data = json.loads(json.dumps(original))
    payload = json.loads(data["rows"][0]["request_bytes"])
    payload["temperature"] = 0.1
    request = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    data["rows"][0].update(request_bytes=request, request_sha256=digest(request.encode()))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="request_drift"):
        run.cold_replay(path)


def test_scenario_report_7854_checkpoint_drift(tmp_path: Path) -> None:
    """A saved family with the wrong identity cannot be silently resumed."""
    from carnot import experiment_7854_v682_intervention_protocol as run

    output = tmp_path / "fixture.json"
    root = tmp_path / "checkpoints"
    run.fixture_e2e(output, root)
    first = next(root.glob("*.json"))
    contents = json.loads(first.read_text())
    contents["identity"] = "wrong"
    first.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="checkpoint_drift"):
        run.fixture_e2e(output, root)


def test_scenario_report_7854_candidate_and_verdicts(tmp_path: Path) -> None:
    """Cold natural rows and readiness come from source and required receipts."""
    from carnot import experiment_7854_v682_intervention_protocol as run
    from carnot.reporting.current_work_receipt import atomic_json

    public, checks, hashes = run.preflight()
    rows, _ = run.prepare_sources(public, tmp_path / "protocol.json")
    fixture = tmp_path / "fixture.json"
    run.fixture_e2e(fixture, tmp_path / "checkpoints")
    artifact = run.build_artifact(
        checks, hashes, rows, fixture, tmp_path / "protocol.json", [], 0.0, [], False
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, artifact)
    assert run.reduce_candidate(candidate) == {"families": 48, "rows": 192, "fixture_rows": 96}
    for key, value, reason in (
        ("experiment_id", 1, "wrong_result_owner"),
        ("rows", [], "row_drift"),
        ("fixture_rows_sha256", "sha256:wrong", "fixture_drift"),
    ):
        changed = {**artifact, key: value}
        atomic_json(candidate, changed)
        with pytest.raises(ValueError, match=reason):
            run.reduce_candidate(candidate)
    manifest = json.loads((run.RAW / "validation_command_manifest.json").read_text())
    log = tmp_path / "child.log"
    log.write_text("pass\n")
    receipts = [
        {
            "name": name,
            "classification": "required",
            "passed": True,
            "log_path": str(log),
            "command_argv": ["true"],
        }
        for name in manifest["required_checks"]
    ]
    ready = run.build_artifact(
        checks, hashes, rows, fixture, tmp_path / "protocol.json", receipts, 0.0, [], False
    )
    assert ready["verdict_class"] == "circular_positive"
    assert ready["intervention_protocol_ready_score"] == 1
    assert ready["target_model"] == "none (no model loaded or invoked)"
    assert ready["repository_health"]["prior_required_failure"]["failed_checks"] == [
        "adversarial_verify"
    ]
    failed = run.build_artifact(
        checks,
        hashes,
        rows,
        fixture,
        tmp_path / "protocol.json",
        [{**receipts[0], "passed": False}],
        0.0,
        [],
        True,
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["gate_check_summary"][0]["artifact_field"].endswith(".passed")
    blocked = run.build_artifact(
        [{**checks[0], "passed": False}], hashes, [], None, None, [], 0.0, [], False
    )
    assert blocked["verdict_class"] == "blocked" and blocked["gate_check_summary"]


def test_scenario_report_7854_dispatch_and_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owned child keeps its true argv and closed log at a byte hash path."""
    from carnot import experiment_7854_v682_intervention_protocol as run

    monkeypatch.setattr(run, "PRIVATE", tmp_path)
    source_log = tmp_path / "original.log"
    source_log.write_text("child exited\n")
    command = {
        "name": "small",
        "argv": ["true", "--basetemp=" + str(tmp_path / "base/x")],
        "classification": "required",
        "timeout_s": 5,
    }

    def child(root: Path, specs: object, **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "name": "small",
                "command_argv": command["argv"],
                "log_path": str(source_log),
                "passed": True,
                "exit_code": 0,
            }
        ]

    monkeypatch.setattr(run, "run_commands", child)
    receipts = run.validate({"commands": [command]}, 0.0)
    assert receipts[0]["log_sha256"] == run.sha256_file(source_log)
    assert (tmp_path / "base").is_dir()
    assert run.seal_receipt({"name": "small", "log_path": str(source_log)}, 0)["log_sha256"]
    Path(receipts[0]["log_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="sealed_log_drift"):
        run.seal_receipt({"name": "small", "log_path": str(source_log)}, 0)
    monkeypatch.setattr(
        run,
        "run_commands",
        lambda *args, **kwargs: [
            {"name": "small", "command_argv": ["wrong"], "log_path": str(source_log)}
        ],
    )
    with pytest.raises(ValueError, match="child_command_drift"):
        run.validate({"commands": [command]}, 0.0)


def test_scenario_report_7854_failed_gate_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present but disqualified source is a failed value, not a missing file."""
    from carnot import experiment_7854_v682_intervention_protocol as run
    from carnot.reporting.current_work_receipt import atomic_json

    producer = json.loads(run.EXTERNAL["producer"].read_text())
    producer["development_cohort_ready_score"] = 0
    copy = tmp_path / "producer.json"
    atomic_json(copy, producer)
    monkeypatch.setattr(run, "EXTERNAL", {**run.EXTERNAL, "producer": copy})
    public, checks, _ = run.preflight()
    assert public == []
    assert any(
        item["artifact_field"] == "development_cohort_ready_score" and not item["passed"]
        for item in checks
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, {"experiment_id": 7854, "task_id": "exp7854-intervention-protocol"})
    with pytest.raises(ValueError, match="source_drift"):
        run.reduce_candidate(candidate)


def test_scenario_report_7854_full_parent_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent publishes terminal rows after exact child exits, or blocks early."""
    from carnot import experiment_7854_v682_intervention_protocol as run
    from carnot.reporting.current_work_receipt import sha256_file

    raw = tmp_path / "raw"
    raw.mkdir()
    manifest_path = raw / "validation_command_manifest.json"
    manifest_path.write_bytes((run.RAW / "validation_command_manifest.json").read_bytes())
    private = tmp_path / "private"
    output = tmp_path / "terminal.json"
    monkeypatch.setattr(run, "RAW", raw)
    monkeypatch.setattr(run, "PRIVATE", private)
    monkeypatch.setattr(run, "OUTPUT", output)
    plan = json.loads(manifest_path.read_text())
    adverse_log = tmp_path / "adverse.log"
    adverse_log.write_text('{"flagged_count":0}')
    receipts = [
        {
            "name": name,
            "classification": "required",
            "passed": True,
            "log_path": str(adverse_log),
            "command_argv": ["true"],
            "exit_code": 0,
        }
        for name in plan["required_checks"]
    ]
    monkeypatch.setattr(run, "validate", lambda plan, started: receipts)
    result = run.run_experiment("20260929")
    assert result["verdict_class"] == "circular_positive"
    assert result["flagged_adversarial"] is False
    assert json.loads(output.read_text())["sample_size_budget"]["intended"] == 192
    assert (
        json.loads((private / "pending_candidate.json").read_text())["verdict_class"] == "partial"
    )
    assert run.reduce_candidate(private / "pending_candidate.json")["rows"] == 192
    adverse_log.write_text("not JSON")
    result = run.run_experiment("20260929")
    assert result["verdict_class"] == "disqualified" and result["flagged_adversarial"]
    monkeypatch.setattr(
        run,
        "preflight",
        lambda: ([], [run.operand("producer", tmp_path / "absent", "exists", True, False)], {}),
    )
    result = run.run_experiment("20260929")
    assert result["verdict_class"] == "blocked"
    with pytest.raises(ValueError, match="run_date_mismatch"):
        run.run_experiment("20260928")
    monkeypatch.setattr(run, "PLAN_SHA256", "sha256:wrong")
    with pytest.raises(ValueError, match="validation_manifest_drift"):
        run.run_experiment("20260929")
    monkeypatch.setattr(run, "PLAN_SHA256", sha256_file(manifest_path))
    plan["schema"] = "wrong"
    manifest_path.write_text(json.dumps(plan))
    monkeypatch.setattr(run, "PLAN_SHA256", sha256_file(manifest_path))
    with pytest.raises(ValueError, match="validation_manifest_drift"):
        run.run_experiment("20260929")


def test_scenario_report_7854_cli_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The direct CLI exercises fixture, cold, date-error and parent exit routes."""
    from carnot import experiment_7854_v682_intervention_protocol as run

    path = tmp_path / "fixture.json"
    assert run.main(["--date", "20260929", "--fixture-e2e", str(path)]) == 0
    assert run.main(["--date", "20260929", "--cold-replay", str(path)]) == 0
    with pytest.raises(ValueError, match="run_date_mismatch"):
        run.main(["--date", "20260928"])
    monkeypatch.setattr(run, "reduce_candidate", lambda path: {"rows": 192})
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"schema":"carnot.exp7854.intervention_result.v1"}')
    assert run.main(["--date", "20260929", "--cold-replay", str(candidate)]) == 0
    monkeypatch.setattr(run, "run_experiment", lambda date: {"verdict_class": "disqualified"})
    assert run.main(["--date", "20260929"]) == 1
