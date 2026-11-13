"""REQ-REPORT-9069-V774, SCENARIO-REPORT-9069-V774: custody must fail closed."""

from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import source_custody_9069 as d
from carnot.reporting import source_custody_runner_9069 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_original_sources_and_missing_authority(tmp_path):
    """REQ-REPORT-9069-V774: intact transport does not replace missing access authority."""
    evidence = d.audit(d.ROOT, tmp_path / "private")
    assert len(evidence["rows"]) == 1024
    assert evidence["role_counts"] == dict(calibration=128, test=128, online=768)
    assert evidence["a_transport_count"] == 512
    assert evidence["b_attempt_inventory"]["attempts"] == []
    assert not evidence["a_ready"] and not evidence["b_ready"]
    assert any(r["observed"] == "missing_field" for r in evidence["checks"])
    assert not evidence["exposure_inventory"]["disclosures"]
    assert all("answer" not in r and "labels" not in r for r in evidence["rows"])


def test_missing_changed_and_rehashed_source(tmp_path):
    """SCENARIO-REPORT-9069-V774: independent pins defeat locally consistent rehashes."""
    missing = d.audit(tmp_path, tmp_path / "raw")
    assert missing["checks"][0]["observed"] == "missing_file"
    target = tmp_path / d.ROSTER
    target.parent.mkdir(parents=True)
    atomic_json(target, dict(state_hashes={}, reproducibility_checksum="rehash"))
    changed = d.audit(tmp_path, tmp_path / "changed")
    assert changed["checks"][0]["observed"] == d.p.reference(target)["sha256"]
    assert not changed["checks"][0]["passed"]


def test_disclosure_roles_and_tokenizer_controls(tmp_path):
    """SCENARIO-REPORT-9069-V774: controls inspect operands rather than ready headlines."""
    unit = dict(unit_id="private-test", role="test", shard="A")
    safe = dict(unit_id="private-test", condition="test", metric="transport_completed", value=1)
    assert d.disclosures(safe, {unit["unit_id"]}) == []
    for changed in (
        dict(safe, answer="secret"),
        dict(role="test", labels=[1]),
        dict(safe, metric="correct"),
    ):
        assert d.disclosures(changed, {unit["unit_id"]})
    manifest = dict(
        graph_manifest=[unit],
        state_hashes={},
        first_response_budget=1,
        generator_settings=d.source.SETTINGS,
    )
    primary = dict(state_hashes={}, model_tokenizer_hashes=dict(d.TOKENIZER))
    assert all(
        r["passed"]
        for r in d.contract_checks(manifest, primary, tmp_path, expected_roles={"test": 1})
    )
    changed = deepcopy(manifest)
    changed["graph_manifest"][0]["role"] = "calibration"
    assert not all(
        r["passed"]
        for r in d.contract_checks(changed, primary, tmp_path, expected_roles={"test": 1})
    )
    primary["model_tokenizer_hashes"]["tokenizer"] = "changed"
    assert not all(
        r["passed"]
        for r in d.contract_checks(manifest, primary, tmp_path, expected_roles={"test": 1})
    )


def test_b_orphan_and_intent_inventory(tmp_path):
    """SCENARIO-REPORT-9069-V774: orphan wire bytes count as attempts without a primary."""
    folder = tmp_path / "results/raw/experiment_9059_private/invocations/1/private-b"
    folder.mkdir(parents=True)
    atomic_json(folder / "wire_response.json", dict(unit_id="private-b", content="sealed"))
    atomic_json(folder / "intent.json", dict(unit_id="private-b"))
    atomic_json(
        folder.parent / "checkpoint.json", dict(starts=["private-b"], transport={}, status={})
    )
    inventory = d.attempt_inventory(tmp_path, {"private-b"})
    assert len(inventory["attempts"]) == 3
    assert all(r["unit_id"] == "private-b" for r in inventory["attempts"])
    assert all("content" not in r for r in inventory["attempts"])


def test_build_blocks_and_disqualifies(tmp_path):
    """REQ-REPORT-9069-V774: failed qualification cannot grant either readiness score."""
    work = runner.freeze(tmp_path / "absent", tmp_path / "raw", True)
    work["raw_references"] = []
    validation = dict(scope={}, receipts=[], coverage={})
    blocked = d.build(work, validation, True)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["a_custody_ready_score"] == blocked["b_collection_ready_score"] == 0
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    assert blocked["model_invocation_counts"] == d.ZERO_INVOCATION_COUNTS


def test_private_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-9069-V774: actual foreign CLI and rehashed tamper rejection."""
    output = tmp_path / (d.NAME + ".json")
    argv = list(
        runner.foreign_argv("--input-root", str(tmp_path / "absent"), "--fixture-e2e", str(output))
    )
    if os.environ.get("COVERAGE_RCFILE"):
        argv[6:7] = ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    receipt = runner.child("private_cli", argv, tmp_path / "logs", deadline=90)
    assert receipt["passed"]
    assert runner.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    value["a_custody_ready_score"] = 1
    value["reproducibility_checksum"] = d.p.checksum(value)
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    receipt = runner.child(
        "rehashed_replay",
        list(runner.foreign_argv("--cold-replay", str(output))),
        tmp_path / "replay",
        deadline=90,
        expected=1,
    )
    assert receipt["passed"]
    with pytest.raises(SystemExit), patch.object(runner, "freeze", side_effect=AssertionError):
        runner.main(["--fixture-e2e", str(d.ROOT / "unsafe.json")])


def fixture_source(tmp_path):
    """Keep synthetic sealed bytes private while exercising the complete audit path."""
    raw = tmp_path / "producer"
    manifest = dict(
        graph_manifest=[
            dict(
                unit_id=f"private-{s}",
                graph_id=f"graph-{s}",
                shard=s,
                role="test",
                questions=[dict(unit_id=f"private-{s}", prompt="private")],
            )
            for s in ("A", "B")
        ],
        state_hashes={},
        first_response_budget=1,
        generator_settings=d.source.SETTINGS,
    )
    mpath = raw / "manifest.json"
    atomic_json(mpath, manifest)
    folder = d.a.runtime.folder(raw / "runtime/collection", 0, manifest["graph_manifest"][0])
    response = dict(id="private-request", usage=dict(completion_tokens=1))
    atomic_json(folder / "response.json", dict(raw_response=response, duration_s=1.0))
    atomic_json(
        folder / "intent.json",
        dict(
            unit_id="private-A",
            request=dict(d.source.SETTINGS, messages=[dict(role="user", content="private")]),
        ),
    )
    transport = dict(
        response_sha256=d.p.reference(folder / "response.json")["sha256"],
        request_id="private-request",
        token_count=1,
        duration_s=1.0,
        status="completed",
    )
    checkpoint = raw / "checkpoint.json"
    atomic_json(
        checkpoint,
        dict(
            manifest_sha256=d.canonical_hash(manifest),
            starts=["private-A"],
            transport={"private-A": transport},
            status={"private-A": "completed", "private-B": "pending"},
        ),
    )
    work = raw / "work.json"
    atomic_json(
        work,
        dict(
            raw=str(raw),
            runtime=dict(
                ledger=[
                    dict(
                        call_id="private-A",
                        operation="generation",
                        response_sha256=d.canonical_hash(response),
                    )
                ]
            ),
        ),
    )
    values = {
        d.ROSTER: dict(fresh_role_manifest=d.p.reference(mpath)),
        d.A: dict(
            custody_reference=d.p.reference(checkpoint),
            raw_shard_hashes=[d.p.reference(work)],
            state_hashes={},
            model_tokenizer_hashes=dict(d.TOKENIZER),
            rows=[dict(unit_id="private-A", **transport)],
        ),
        d.CAPSTONE: {},
    }
    for name, value in values.items():
        atomic_json(tmp_path / name, value)
    return {name: d.p.reference(tmp_path / name)["sha256"] for name in values}


def test_private_disclosure_access_and_attempt_closes_confirmation(tmp_path):
    """SCENARIO-REPORT-9069-V774: an access event or B orphan cannot hide behind a primary."""
    pins = fixture_source(tmp_path)
    logs = tmp_path / "results/raw/experiment_9058_private/label_access.jsonl"
    logs.parent.mkdir(parents=True)
    logs.write_text(
        json.dumps(dict(role="test", authorized=True))
        + "\n"
        + json.dumps(dict(role="test", authorized=False))
        + "\n"
    )
    atomic_json(
        tmp_path / "results/raw/experiment_9059_private/private-B/wire_response.json",
        dict(unit_id="private-B"),
    )
    with (
        patch.object(d, "PINS", pins),
        patch.object(
            d.methods,
            "authenticate",
            return_value=dict(checks=[], references=[], failures=[], state_hashes={}),
        ),
    ):
        value = d.audit(tmp_path, tmp_path / "audit")
    assert value["a_transport_count"] == 1
    assert value["exposure_inventory"]["disclosures"]
    assert value["b_attempt_inventory"]["attempts"]
    assert next(r for r in value["rows"] if r["arm"] == "B")["disposition"] == "attempted"


def test_immutable_operand_and_original_method_failures(tmp_path):
    """REQ-VERIFY-9069-V774: changed primitives stop the audit before response access."""
    pins = fixture_source(tmp_path)
    with (
        patch.object(d, "PINS", pins),
        patch.object(
            d.methods,
            "authenticate",
            return_value=dict(checks=[], references=[], failures=[dict(passed=False)]),
        ),
    ):
        assert not d.audit(tmp_path, tmp_path / "method-failure")["rows"]
    (tmp_path / "producer/work.json").write_bytes(b"changed")
    with patch.object(d, "PINS", pins):
        value = d.audit(tmp_path, tmp_path / "changed")
    assert not value["rows"]
    assert any(not r["passed"] for r in value["checks"])


def test_validation_success_failure_and_replay_integrity(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9069-V774: failed owned checks preserve the last checked primary."""
    original_freeze = runner.freeze
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "freeze",
        lambda root, raw, fixture: original_freeze(tmp_path / "absent", raw, fixture),
    )
    command = runner.checks.CommandSpec("private_check", ("true",), "current_scoped", 1)
    monkeypatch.setattr(runner, "plan", lambda _: ([command], dict(frozen_ns=0)))
    monkeypatch.setattr(runner.checks, "run_check", lambda *args: dict(passed=True))
    monkeypatch.setattr(runner.checks, "receipts_pass", lambda _: True)
    monkeypatch.setattr(runner, "measured", lambda *args: dict(rows=[]))
    monkeypatch.setattr(runner, "verify_measured", lambda *args: True)
    monkeypatch.setattr(runner.terminal, "scanners", lambda *args: dict(passed=True))
    assert runner.main([]) == 0
    output = tmp_path / "results" / (d.NAME + ".json")
    assert runner.replay(output)["passed"]
    before = output.read_bytes()
    with patch.object(runner.checks, "receipts_pass", return_value=False):
        assert not runner.replay(output)["passed"]
        assert runner.main([]) == 1
    assert output.read_bytes() == before
    value = json.loads(before)
    Path(value["raw_shard_hashes"][0]["path"]).write_bytes(b"changed")
    assert not runner.replay(output)["passed"]
    with patch.object(runner, "replay", return_value=dict(passed=True)):
        assert runner.main(["--cold-replay", str(output)]) == 0


def test_inventory_drift_and_null_build(tmp_path):
    """SCENARIO-REPORT-9069-V774: a new invocation invalidates the frozen pending inventory."""
    work = runner.freeze(tmp_path / "absent", tmp_path / "raw", True)
    source_checks = work["evidence"]["checks"]
    work["evidence"]["checks"] = []
    work["environment"]["checks"] = []
    work["accepted"] = False
    work["evidence"]["b_attempt_inventory"] = dict(nonempty=True)
    work["raw_references"] = []
    validation = dict(scope={}, receipts=[], coverage={})
    assert d.build(work, validation, True)["verdict_class"] == "null"
    work["evidence"]["checks"] = source_checks
    atomic_json(Path(work["raw"]) / "work.json", work)
    atomic_json(Path(work["raw"]) / "validation.json", validation)
    work["raw_references"] = [d.p.reference(Path(work["raw"]) / "work.json")]
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, d.build(work, validation, False))
    assert runner.replay(output)["reason"] == "invocation_inventory_drift"


def test_rehashed_source_gate_and_health_receipt(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9069-V774: coordinated local rehashing cannot change source authority."""
    health = tmp_path / "health.json"
    atomic_json(health, dict(passed=False, exit_code=2, timed_out=False))
    monkeypatch.setattr(runner, "HEALTH", health)
    work = runner.freeze(tmp_path / "absent", tmp_path / "raw", True)
    assert work["repository_health"]["current_suite"]["exit_code"] == 2
    work.update(accepted=False, raw_references=[])
    work["evidence"]["checks"][0]["expected"] = "sha256:locally-rehashed"
    raw = Path(work["raw"])
    validation = dict(scope={}, receipts=[], coverage={})
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [d.p.reference(raw / "work.json")]
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, d.build(work, validation, False))
    assert runner.replay(output)["reason"] == "independent_source_binding"
