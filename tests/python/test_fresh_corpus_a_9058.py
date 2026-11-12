"""REQ-REPORT-9058-V773: private first-response custody and publication controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import fresh_corpus_a_9058 as d
from carnot.reporting import fresh_corpus_runner_9058 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.experiment_7303_validation_scope import CommandSpec


def manifest():
    """Private mechanics reuse frozen identities without claiming model observations."""
    primary = json.loads((d.ROOT / d.ROSTER).read_bytes())
    return json.loads(Path(primary["fresh_role_manifest"]["path"]).read_bytes())


def private_argv(*args):
    """Measure the real private CLI in the same invocation coverage database."""
    command = list(runner.foreign_argv(*args))
    if os.environ.get("COVERAGE_RCFILE"):
        command[6:7] = ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    return command


def test_missing_changed_and_real_authority(tmp_path):
    """SCENARIO-REPORT-9058-PRIVATE-QUALIFICATION: missing and altered bytes are distinct."""
    missing = d.authenticate(tmp_path, tmp_path / "raw")
    assert missing["checks"][0]["observed"] == "missing_file"
    target = tmp_path / d.ROSTER
    target.parent.mkdir(parents=True)
    atomic_json(target, {})
    changed = d.authenticate(tmp_path, tmp_path / "changed")
    assert changed["checks"][0]["observed"] == d.p.reference(target)["sha256"]
    evidence = d.authenticate(d.ROOT, tmp_path / "real")
    assert all(r["passed"] for r in evidence["checks"] if r["check"] != "no_response_exposure")
    assert len(evidence["units"]) == 512
    assert not any("expected" in u for u in evidence["units"])


def test_first_attempt_and_lost_response(tmp_path):
    """SCENARIO-REPORT-9058-FIRST-RESPONSE-CUSTODY: intent spends a request permanently."""
    m = manifest()
    units = d.source.scheduled(m, "A")[:3]
    m["graph_manifest"] = [
        g for g in m["graph_manifest"] if g["unit_id"] in {u["unit_id"] for u in units}
    ]
    state = d.c.Checkpoint(m, canonical_hash(m), tmp_path / "custody")
    state.start(units[0]["unit_id"])
    state = d.c.Checkpoint(m, canonical_hash(m), tmp_path / "custody")
    assert state.state["status"][units[0]["unit_id"]] == "censored"
    calls = []

    def transport(payload, timeout):
        calls.append(payload)
        if len(calls) == 2:
            raise TimeoutError("private_timeout")
        return dict(id="private", usage=dict(completion_tokens=1))

    rows = d.collect(units, transport, tmp_path / "collection", {}, checkpoint=state)
    assert len(calls) == 2 and len(rows) == 3
    assert rows[0]["status"] == "censored" and rows[1]["status"] == "completed"
    assert rows[2]["status"] == "failed"
    assert all(set(r) <= d.PUBLIC_FIELDS for r in rows)
    again = d.collect(units, transport, tmp_path / "collection", {}, checkpoint=state)
    assert again == rows and len(calls) == 2
    with pytest.raises(ValueError, match="duplicate"):
        state.start(units[1]["unit_id"])
    with pytest.raises(ValueError, match="transport_schema"):
        state.finish(units[0]["unit_id"], dict(output=dict(correct=True)))


def test_budget_censors_without_transport(tmp_path):
    """REQ-REPORT-9058-V773: the scheduled denominator survives budget exhaustion."""
    m = manifest()
    units = d.source.scheduled(m, "A")[:1]
    state = d.c.Checkpoint(m, canonical_hash(m), tmp_path / "custody")
    rows = d.collect(units, None, tmp_path / "collection", {}, checkpoint=state, budget_s=0)
    assert rows[0]["status"] == "censored"
    assert not state.state["starts"]


def test_sixteen_checkpoint_and_changed_upstream(tmp_path):
    """REQ-REPORT-9058-V773: durable batch progress and changed operands cannot disappear."""
    m = manifest()
    state = d.c.Checkpoint(m, canonical_hash(m), tmp_path / "custody")
    units = d.source.scheduled(m, "A")[:16]
    rows = d.collect(
        units,
        lambda *_: dict(id="private", usage=dict(completion_tokens=1)),
        tmp_path / "collection",
        {},
        checkpoint=state,
    )
    assert len(rows) == 16
    original = d.p.reference
    upstream = json.loads((d.ROOT / d.ROSTER).read_bytes())["fresh_role_manifest"]["path"]

    def changed(path):
        return (
            dict(path=str(path), sha256="sha256:private-changed")
            if str(path) == upstream
            else original(path)
        )

    with patch.object(d.p, "reference", side_effect=changed):
        assert any(not r["passed"] for r in d.authenticate(d.ROOT, tmp_path / "changed")["checks"])


def test_freeze_uses_owned_transport_and_private_rows(tmp_path):
    """SCENARIO-REPORT-9058-FIRST-RESPONSE-CUSTODY: runtime gets only scheduled pending units."""
    m = manifest()
    units = d.source.scheduled(m, "A")[:1]
    evidence = dict(
        checks=[], references=[], units=units, manifest=m, state_hashes=m["state_hashes"]
    )

    def run(pending, raw, task_id):
        assert task_id == d.TASK and pending == units
        d.runtime.collect(
            pending,
            lambda *_: dict(id="private", usage=dict(completion_tokens=1)),
            raw / "collection",
            {},
        )
        return dict(
            counts=dict(d.p.ZERO_INVOCATION_COUNTS),
            ledger=[],
            checks=[
                dict(check="private_block", artifact_field="runtime", expected=True, observed=False)
            ],
            owned_failure=False,
            model_receipt={},
        )

    with (
        patch.object(d, "authenticate", return_value=evidence),
        patch.object(d.p, "environment", return_value=dict(checks=[])),
        patch.object(d.runtime, "run", side_effect=run),
    ):
        work = runner.freeze(tmp_path, tmp_path / "work", False)
    assert work["rows"][0]["status"] == "completed"
    assert set(work["rows"][0]) <= d.PUBLIC_FIELDS
    assert not work["runtime"]["checks"][0]["passed"]
    with (
        patch.object(d, "authenticate", return_value=evidence),
        patch.object(d.p, "environment", return_value=dict(checks=[])),
    ):
        fixture = runner.freeze(tmp_path, tmp_path / "fixture", True)
    assert fixture["rows"][0]["status"] == "censored"


def test_build_disqualifies_and_bounds_model_credit(tmp_path):
    """REQ-REPORT-9058-V773: fake completed headlines cannot replace actual model calls."""
    work = runner.freeze(tmp_path, tmp_path / "raw", True)
    work.update(raw_references=[], fixture_only=False)
    validation = dict(scope={}, receipts=[], coverage={})
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "disqualified" and value["shard_ready_score"] == 0
    work["evidence"]["checks"] = work["environment"]["checks"] = []
    assert d.build(work, validation, True)["verdict_class"] == "null"


def test_main_validation_success_and_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9058-PRIVATE-QUALIFICATION: current failed checks prevent publication."""
    original_freeze = runner.freeze
    observed = []

    def frozen(root, raw, fixture):
        work = original_freeze(tmp_path / "absent", raw, fixture)
        return work

    command = CommandSpec("private_check", ("true",), "current_scoped", 1)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "freeze", frozen)
    monkeypatch.setattr(runner, "plan", lambda _: ([command], dict(frozen_ns=0)))
    monkeypatch.setattr(runner.checks, "run_check", lambda *args: dict(passed=True))
    monkeypatch.setattr(runner.checks, "receipts_pass", lambda _: True)
    monkeypatch.setattr(runner, "measured", lambda *args: dict(rows=[]))
    monkeypatch.setattr(runner, "verify_measured", lambda *args: True)
    monkeypatch.setattr(runner.terminal, "scanners", lambda *args: dict(passed=True))
    monkeypatch.delenv("COVERAGE_RCFILE", raising=False)
    assert runner.main([]) == 0
    output = tmp_path / "results" / (d.NAME + ".json")
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and value["verdict_class"] == "blocked"
    with patch.object(runner.checks, "receipts_pass", return_value=False):
        monkeypatch.setenv("COVERAGE_RCFILE", "private-preserved")
        assert runner.main([]) == 1
    assert json.loads(output.read_bytes()) == value
    monkeypatch.setattr(runner, "replay", lambda _: dict(passed=True))
    assert runner.main(["--cold-replay", str(output)]) == 0
    with pytest.raises(SystemExit):
        runner.main(["--fixture-e2e", str(output)])


def test_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-9058-PRIVATE-QUALIFICATION: cold replay recomputes primitive summaries."""
    output = tmp_path / (d.NAME + ".json")
    receipt = runner.child(
        "private_cli",
        private_argv("--input-root", str(tmp_path / "absent"), "--fixture-e2e", str(output)),
        tmp_path / "logs",
        deadline=90,
    )
    assert receipt["passed"]
    assert runner.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["shard_ready_score"] == 0
    assert value["inference_substrate_class"] == "no_model_load"
    value["completed_count"] = 512
    value["reproducibility_checksum"] = d.p.checksum(value)
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    receipt = runner.child(
        "tamper_cli",
        private_argv("--cold-replay", str(output)),
        tmp_path / "tamper",
        deadline=90,
        expected=1,
    )
    assert receipt["passed"]


def test_primitive_replay_and_validation_drift(tmp_path):
    """SCENARIO-REPORT-9058-PRIVATE-QUALIFICATION: original responses defeat rehashed work tampering."""
    from carnot.verify.qwen_development_capture_7995 import Ledger

    raw = tmp_path / "raw"
    work = runner.freeze(tmp_path / "missing", raw, True)
    m = manifest()
    units = d.source.scheduled(m, "A")[:2]
    state = d.c.Checkpoint(m, canonical_hash(m), raw / "custody")
    ledger = Ledger(raw / "runtime/invocation_ledger.json")

    def transport(payload, timeout):
        value = dict(id="private", usage=dict(completion_tokens=1))
        if ledger.rows:
            raise TimeoutError("private_failed_transport")
        ledger.start("generation", units[0]["unit_id"], payload)
        ledger.finish(units[0]["unit_id"], "completed", value)
        return value

    work["rows"] = d.collect(units, transport, raw / "runtime/collection", {}, checkpoint=state)
    work["evidence"].update(manifest=m, units=units)
    work["runtime"].update(counts=ledger.counts(), ledger=ledger.rows)
    work["accepted"] = False
    validation = dict(scope=dict(frozen_ns=0), receipts=[], coverage={})
    output = tmp_path / "candidate.json"

    def save():
        atomic_json(raw / "work.json", {k: v for k, v in work.items() if k != "raw_references"})
        atomic_json(raw / "validation.json", validation)
        work["raw_references"] = [
            d.p.reference(raw / "work.json"),
            d.p.reference(raw / "validation.json"),
        ]
        atomic_json(output, d.build(work, validation, work["accepted"]))

    save()
    assert runner.replay(output)["passed"]
    work["rows"][0]["value"] = 0
    save()
    assert runner.replay(output)["reason"] == "transport_reduction_drift"
    work["rows"] = d.public_rows(units, state)
    work["runtime"]["ledger"][0]["response_sha256"] = "sha256:private-forged"
    save()
    assert runner.replay(output)["reason"] == "response_receipt_drift"
    work["runtime"]["ledger"][0]["response_sha256"] = canonical_hash(
        dict(id="private", usage=dict(completion_tokens=1))
    )
    work["accepted"] = True
    save()
    assert runner.replay(output)["reason"] == "validation_drift"
    value = json.loads(output.read_bytes())
    value["code_config_hashes"][0]["sha256"] = "sha256:private-changed"
    atomic_json(output, value)
    assert runner.replay(output)["reason"] == "primitive_bytes"


def test_frozen_plan_and_inprocess_private_publication(tmp_path):
    """SCENARIO-REPORT-9058-PRIVATE-QUALIFICATION: exact commands precede private publication."""
    commands, scope = runner.plan(tmp_path / "plan")
    assert {c.name for c in commands} == set(scope["required_check_names"])
    assert scope["statement_percent"] == 100 and scope["excluded_statements"] == 0
    output = tmp_path / (d.NAME + ".json")
    assert (
        runner.main(["--input-root", str(tmp_path / "absent"), "--fixture-e2e", str(output)]) == 0
    )
    assert runner.replay(output)["passed"]
