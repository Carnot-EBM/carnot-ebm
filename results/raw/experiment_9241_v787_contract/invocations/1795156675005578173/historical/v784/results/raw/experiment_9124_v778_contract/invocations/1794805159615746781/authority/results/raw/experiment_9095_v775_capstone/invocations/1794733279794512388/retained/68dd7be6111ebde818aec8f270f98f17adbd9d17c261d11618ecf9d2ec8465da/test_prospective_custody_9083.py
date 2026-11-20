"""REQ-REPORT-9083-V775, SCENARIO-REPORT-9083-V775-ENFORCEMENT: fail closed."""

from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import prospective_custody_9083 as d


def request(slot="fixture-A"):
    """An exact scripted request tests custody without loading a model."""
    return dict(
        slot=slot,
        graph=slot,
        shard="A",
        role="test",
        split="test",
        prompt="scripted private fixture",
        model="scripted_no_model",
        seed=775083,
    )


def test_namespace_and_positive_fixture(tmp_path):
    """SCENARIO-REPORT-9083-V775: real OS controls qualify finite supervised scope."""
    value = d.qualify(tmp_path / "private")
    assert value["custody_ready_score"] == 1
    assert all(row["passed"] for row in value["bypass_control_rows"])
    assert set(d.CONTROLS) == {r["name"] for r in value["bypass_control_rows"]}
    assert value["complete_access_authority_scope"] == "supervised_fixture_invocations_only"
    assert d.Ledger.replay(value["ledger_reference"])["completed"] == 1
    collector = json.loads(Path(value["ledger_reference"]["path"]).read_bytes())
    worker = json.loads(
        Path(value["enforcement_receipts"][0]["ledger_reference"]["path"]).read_bytes()
    )
    collection_rows = d.Ledger(Path(collector["ledger"]["path"]).parent).events()
    worker_rows = d.Ledger(Path(worker["ledger"]["path"]).parent).events()
    assert collection_rows[1]["monotonic_ns"] < worker_rows[0]["monotonic_ns"]
    payload = json.loads(worker_rows[0]["payload"]["prompt"])
    assert payload["features"] == collection_rows[1]["payload"]["features"]
    assert len(payload["paths"]) == 4 and "raw_response" not in payload


def test_start_before_collector_and_crash_spends_slot(tmp_path):
    """REQ-REPORT-9083-V775: durable attempted slots never become free retries."""
    ledger = d.Ledger(tmp_path / "ledger")

    def collect():
        assert len(ledger.events()) == 1
        raise RuntimeError("scripted transport failed")

    with pytest.raises(RuntimeError, match="scripted transport failed"):
        ledger.collect(request(), collect)
    assert ledger.summary() == dict(attempted=1, completed=0, failed=1, censored=0)
    ledger.start(request("lost"))
    restored = d.Ledger(tmp_path / "ledger")
    assert restored.summary()["censored"] == 1
    with pytest.raises(ValueError, match="duplicate_slot"):
        restored.start(request("lost"))
    with pytest.raises(ValueError, match="missing_request_start"):
        restored.finish("orphan", {"response": "retained"})
    with pytest.raises(ValueError, match="duplicate_slot"):
        restored.start(request("orphan"))
    assert restored.summary()["attempted"] == 3
    assert restored.summary()["censored"] == 2
    assert any(r["event"] == "rejected" for r in restored.events())


def test_replay_rejects_reordered_rehashed_and_orphan(tmp_path):
    """SCENARIO-REPORT-9083-V775: the independent seal detects rewritten history."""
    ledger = d.Ledger(tmp_path / "ledger")
    ledger.collect(request(), lambda: {"response": "scripted"})
    ref = ledger.seal()
    saved = ledger.path.read_bytes()
    rows = ledger.events()
    ledger.path.write_text("\n".join(json.dumps(r) for r in reversed(rows)) + "\n")
    with pytest.raises(ValueError):
        d.Ledger.replay(ref)
    ledger.path.write_bytes(saved)
    rows[0]["payload"]["seed"] += 1
    ledger.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError):
        d.Ledger.replay(ref)
    ledger.path.write_bytes(saved)
    assert d.Ledger.replay(ref)["completed"] == 1


def test_bad_request_view_entrypoint_and_private_storage(tmp_path):
    """REQ-REPORT-9083-V775: malformed inputs cannot widen the worker authority."""
    ledger = d.Ledger(tmp_path / "ledger")
    changed = request()
    changed.pop("prompt")
    with pytest.raises(ValueError, match="request_schema"):
        ledger.start(changed)
    for view in [
        {"features": [float("nan")]},
        {"features": [1], "truth": 1},
        {"features": [True]},
        {"features": "raw"},
    ]:
        with pytest.raises(ValueError, match="numeric_view"):
            d.namespace(tmp_path / "worker", view)
    with pytest.raises(ValueError, match="entrypoint"):
        d.namespace(tmp_path / "worker", {"features": [1]}, entrypoint="alternate.py")
    folder = tmp_path / "unsafe"
    folder.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="private_storage"):
        d.Ledger(folder)


def test_external_block_and_readiness_reduction(tmp_path):
    """REQ-REPORT-9083-V775: missing external inputs terminate with exact operands."""
    evidence = d.authenticate(tmp_path / "absent", tmp_path / "private")
    assert evidence["gate_check_summary"][0]["observed"] == "missing_file"
    work = dict(
        evidence=evidence,
        qualification={},
        duration_s=0.1,
        raw_references=[],
        code_config_hashes=[],
        current_work_receipts={},
    )
    value = d.build(work, {"passed": True, "receipts": []})
    assert value["verdict_class"] == "blocked"
    assert value["custody_ready_score"] == value["roster_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert value["independent_generalization_score"] == 0
    assert d.build(work, {"passed": False, "receipts": []})["verdict_class"] == "disqualified"
    changed = deepcopy(value)
    changed["custody_ready_score"] = 1
    assert value != changed


def test_namespace_unavailable_has_no_fallback(tmp_path):
    """REQ-REPORT-9083-V775: an unavailable OS boundary cannot earn readiness."""
    with patch.object(d.shutil, "which", return_value=None):
        result = d.namespace(tmp_path / "worker", {"features": [1]})
    assert not result["passed"] and result["reason"] == "missing_namespace_runtime"


def test_private_cli_block_cold_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-9083-V775: actual standalone CLI works outside the checkout."""
    from carnot.reporting import prospective_custody_runner_9083 as runner

    output = tmp_path / (d.NAME + ".json")
    result = runner.child(
        "private_block",
        runner.foreign("--input-root", str(tmp_path / "absent"), "--fixture-e2e", str(output)),
        tmp_path / "logs",
        deadline=90,
    )
    assert result["passed"]
    assert runner.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    value["custody_ready_score"] = 1
    d.atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    result = runner.child(
        "tampered",
        runner.foreign("--cold-replay", str(output)),
        tmp_path / "tamper",
        expected=1,
        deadline=90,
    )
    assert result["passed"]
    assert runner.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--fixture-e2e", str(d.ROOT / "unsafe.json")])
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261114"])


def test_positive_cli_fixture_and_script_entrypoint(tmp_path):
    """REQ-REPORT-9083-V775: fixture custody succeeds without a production roster."""
    import runpy
    from carnot.reporting import prospective_custody_runner_9083 as runner

    output = tmp_path / (d.NAME + ".json")
    assert runner.main(["--fixture-e2e", str(output), "--input-root", str(tmp_path)]) == 0
    value = json.loads(output.read_bytes())
    assert value["fixture_custody_ready_score"] == 1
    assert value["roster_ready_score"] == value["current_llm_invocations"] == 0
    assert runner.replay(output)["passed"]
    with patch.object(runner, "main", return_value=0), pytest.raises(SystemExit) as end:
        runpy.run_path(str(d.ROOT / d.CLI), run_name="__main__")
    assert end.value.code == 0


def test_authenticated_history_and_exact_missing_operands(tmp_path):
    """REQ-REPORT-9083-V775: original roster bytes cannot repair missing historical coverage."""
    evidence = d.authenticate(d.ROOT, tmp_path / "history")
    failed = [r for r in evidence["gate_check_summary"] if not r["passed"]]
    assert [r["check"] for r in failed] == ["historical_attempt_topology_coverage"]
    assert failed[0]["observed"] == "missing_field"
    inventory = json.loads(
        Path(evidence["historical_attempt_exclusion_reference"]["path"]).read_bytes()
    )
    assert inventory["excluded_v773_shards"] == ["A", "B"]
    assert inventory["retained_proposal_count"] == 47987
    assert len(inventory["frozen_rosters"][0]["identities"]) == 1024
    assert not inventory["global_completeness_claim"]
    assert d.gate(tmp_path / "absent", "ready", 1, 0, "zero")["observed"] == 0
    assert not d.gate(tmp_path / "absent", "ready", 1, True, "type")["passed"]


def test_chain_seal_and_namespace_failure_paths(tmp_path):
    """SCENARIO-REPORT-9083-V775: changed chain/seal bytes and isolation failures fail closed."""
    ledger = d.Ledger(tmp_path / "ledger")
    ledger.collect(request(), lambda: {"response": "scripted"})
    sealed = ledger.seal()
    with pytest.raises(ValueError, match="sealed_ledger"):
        ledger.append("rejected", "new", {})
    saved = ledger.path.read_bytes()
    rows = ledger.events()
    rows[0]["index"] += 1
    ledger.path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="ledger_chain"):
        ledger.events()
    ledger.path.write_bytes(saved)
    seal_path = Path(sealed["path"])
    value = json.loads(seal_path.read_bytes())
    value["summary"]["completed"] = 3
    d.atomic_json(seal_path, value)
    with pytest.raises(ValueError, match="seal_bytes"):
        d.Ledger.replay(sealed)
    with pytest.raises(ValueError, match="ledger_summary"):
        d.Ledger.replay(d.reference(seal_path))
    with patch.object(d.subprocess, "run", side_effect=OSError("unavailable")):
        assert not d.namespace(tmp_path / "failed", {"features": [1]})["passed"]
    result = d.subprocess.CompletedProcess([], 0, json.dumps({"uid": 65534}), "")
    with patch.object(d.subprocess, "run", return_value=result):
        assert not d.namespace(tmp_path / "omitted", {"features": [1]})["passed"]
    result = d.subprocess.CompletedProcess([], 1, "", "denied")
    with patch.object(d.subprocess, "run", return_value=result):
        assert not d.namespace(tmp_path / "exit1", {"features": [1]})["passed"]
    assert not d.refused(lambda: None)


def test_failed_fixture_never_ready(tmp_path):
    """REQ-REPORT-9083-V775: unavailable namespace or omitted probes prevents readiness."""
    with patch.object(d.shutil, "which", return_value=None):
        assert d.qualify(tmp_path / "unavailable")["custody_ready_score"] == 0
    unchecked = tmp_path / "unchecked/extra/invocation"
    unchecked.mkdir(parents=True)
    (tmp_path / "unchecked").chmod(0o700)
    d.atomic_json(unchecked / "seal.json", {"unchecked": True})
    assert d.qualify(tmp_path / "unchecked")["custody_ready_score"] == 0
    unsealed = tmp_path / "unsealed/extra/invocation"
    unsealed.mkdir(parents=True)
    (tmp_path / "unsealed").chmod(0o700)
    (unsealed / "ledger.jsonl").write_text('{"event":"start"}\n')
    assert d.qualify(tmp_path / "unsealed")["custody_ready_score"] == 0


def test_runner_receipts_scanners_and_production_branches(tmp_path):
    """REQ-REPORT-9083-V775: current failed checks preserve the primary instead of publishing."""
    from carnot.reporting import prospective_custody_runner_9083 as runner

    scope = runner.plan(tmp_path)
    for command in scope["commands"]:
        if command["name"] in {"owned", "consumers", "e2e018020021"}:
            assert "--noconftest" in command["argv"]
    assert not runner.checks_pass(dict(scope=scope, receipts=[], coverage={}))
    receipts = []
    for command in scope["commands"]:
        log = tmp_path / (command["name"] + ".log")
        log.write_text("private")
        receipts.append(
            dict(
                passed=True,
                argv=command["argv"],
                started_monotonic_ns=scope["frozen_monotonic_ns"],
                stdout_path=str(log),
                stderr_path=str(log),
                stdout_sha256=d.sha256_file(log),
                stderr_sha256=d.sha256_file(log),
            )
        )
    validation = dict(scope=scope, receipts=receipts, coverage={})
    with patch.object(runner, "verify_measured", return_value=True):
        assert runner.checks_pass(validation)
        saved = scope["code_test_hashes"][0]["sha256"]
        scope["code_test_hashes"][0]["sha256"] = "changed"
        assert not runner.checks_pass(validation)
        scope["code_test_hashes"][0]["sha256"] = saved
    with patch.object(runner, "child", return_value={"passed": True}):
        assert runner.scanners(tmp_path / "candidate.json", tmp_path / "scanners")["passed"]
    with (
        patch.object(runner, "child", return_value={"passed": False}),
        patch.object(runner, "measured", return_value={}),
        patch.object(runner, "checks_pass", return_value=False),
    ):
        assert runner.main(["--input-root", str(tmp_path)]) == 1

    def publish(output, value, validator):
        assert validator(tmp_path / "candidate.json")["passed"]
        return {"passed": True}

    with (
        patch.object(runner, "child", return_value={"passed": True}),
        patch.object(runner, "measured", return_value={}),
        patch.object(runner, "checks_pass", return_value=True),
        patch.object(runner, "scanners", return_value={"passed": True}),
        patch.object(runner, "publish_primary", side_effect=publish),
    ):
        assert runner.main(["--input-root", str(tmp_path)]) == 0


def test_replay_authenticates_primitives_worker_and_validation(tmp_path):
    """SCENARIO-REPORT-9083-V775: even rehashed validation and worker changes are rejected."""
    from carnot.reporting import prospective_custody_runner_9083 as runner

    output = tmp_path / (d.NAME + ".json")
    runner.main(["--fixture-e2e", str(output), "--input-root", str(tmp_path)])
    value = json.loads(output.read_bytes())
    for field in ("worker_code_namespace_policy_reference", "invocation_inventory_reference"):
        sidecar = Path(value[field]["path"])
        saved = sidecar.read_bytes()
        sidecar.write_bytes(saved + b"\n")
        assert runner.replay(output)["reason"] == "custody_reference_bytes"
        sidecar.write_bytes(saved)
    worker = Path(value["enforcement_receipts"][0]["worker_reference"]["path"])
    original = worker.read_bytes()
    worker.chmod(0o600)
    worker.write_text("changed")
    assert runner.replay(output)["reason"] == "worker_bytes"
    worker.write_bytes(original)
    raw = Path(value["raw_shard_hashes"][0]["path"])
    work = json.loads(raw.read_bytes())
    work["fixture_only"] = False
    d.atomic_json(raw, work)
    assert runner.replay(output)["reason"] == "primitive_bytes"
    value["raw_shard_hashes"][0] = d.reference(raw)
    d.atomic_json(output, value)
    with patch.object(runner, "checks_pass", return_value=False):
        assert runner.replay(output)["reason"] == "validation_receipts"
    original_ref = d.reference

    def changed(path):
        ref = original_ref(path)
        if path.name == "fresh_role_manifest.json":
            ref["sha256"] = "changed"
        return ref

    with patch.object(d, "reference", side_effect=changed):
        evidence = d.authenticate(d.ROOT, tmp_path / "changed_history")
    assert any(
        r["observed"] == "changed" and not r["passed"] for r in evidence["gate_check_summary"]
    )


def test_original_prefix_fixture_preserves_producer_and_controls(tmp_path):
    """REQ-REPORT-9083-V775: historical source assertions use original prefix context."""
    from types import SimpleNamespace
    from carnot.reporting import prospective_custody_runner_9083 as runner
    from carnot.reporting import prospective_roster_9055 as old

    original = old.authenticate
    item = SimpleNamespace(
        nodeid="tests/python/test_prospective_roster_9055.py::test_authenticated_original_bytes_and_source_inventory",
        module=SimpleNamespace(d=old),
    )
    hook = runner.pytest_runtest_call(item)
    next(hook)
    try:
        evidence = old.authenticate(old.ROOT, tmp_path / "historical")
        assert not evidence["failures"]
        assert old.ROOT == d.ROOT
        negative = tmp_path / "negative/results/raw/experiment_9058_private/intent.json"
        d.atomic_json(negative, {"started": True})
        refused = old.authenticate(tmp_path / "negative", tmp_path / "negative_audit")
        assert any(r["check"] == "no_collection_since_prefix" for r in refused["failures"])
    finally:
        with pytest.raises(StopIteration):
            next(hook)
    assert old.authenticate is original
    hook = runner.pytest_runtest_call(SimpleNamespace(nodeid="other"))
    next(hook)
    with pytest.raises(StopIteration):
        next(hook)
