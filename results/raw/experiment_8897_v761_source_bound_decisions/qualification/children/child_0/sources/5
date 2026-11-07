"""REQ-REPORT-8897: typed-source coverage cannot stand in for record truth."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify.source_bound_witness import verify_source_bound_witness
from carnot.verify.size_aware_witness import verify_size_aware_witness
from carnot.reporting import source_bound_decisions_8897 as m
from carnot.reporting import source_bound_runner_8897 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def ledger_request():
    """SCENARIO-REPORT-8897-CONTRACT: bytes and IDs describe the supplied typed source."""
    payload = dict(facts=[["a", "r", "b", True]], query=[["a", "r", "b"]])
    records = [dict(record_id="first", fact=payload["facts"][0])]
    source = json.dumps(dict(records=records), sort_keys=True, separators=(",", ":")).encode()
    ledger = dict(records=records, source_sha256="sha256:" + hashlib.sha256(source).hexdigest())
    return payload, dict(record_ids=["first"], source_ledger=ledger, source_bytes=source)


def test_separate_claims_and_unchanged_decisions():
    """REQ-REPORT-8897: complete supplied records do not establish text semantics."""
    payload, kwargs = ledger_request()
    original = deepcopy((payload, kwargs))
    absent = verify_source_bound_witness(payload)
    assert absent["record_decision"] == verify_size_aware_witness(payload)["decision"] == "support"
    assert absent["source_coverage_status"] == absent["source_semantics_status"] == "unknown"
    complete = verify_source_bound_witness(payload, **kwargs)
    assert complete["source_coverage_status"] == "complete"
    assert complete["source_semantics_status"] == "unknown"
    assert complete["ledger_errors"] == []
    empty = dict(facts=[], query=payload["query"])
    incomplete = verify_source_bound_witness(empty, **dict(kwargs, record_ids=[]))
    assert incomplete["source_coverage_status"] == "incomplete"
    assert incomplete["record_decision"] == "insufficient"
    exhausted = verify_source_bound_witness(payload, **kwargs, cap=0)
    assert exhausted["budget_exhausted"] and exhausted["record_decision"] == "insufficient"
    assert exhausted["source_coverage_status"] == "complete"
    negative = deepcopy(payload)
    negative["facts"][0][3] = False
    assert verify_source_bound_witness(negative)["record_decision"] == "refute"
    assert (payload, kwargs) == original
    with pytest.raises(ValueError):
        verify_source_bound_witness(dict(payload, hidden_truth="support"))


def test_ledger_order_and_empty_source():
    """REQ-REPORT-8897: ledger order cannot change equality of supplied typed records."""
    pair = m.freeze(set(), count=1)[0]["pairs"][0]
    ledger = deepcopy(pair["ledger"])
    ledger["records"].reverse()
    result = verify_source_bound_witness(
        pair["input"],
        record_ids=pair["record_ids"],
        source_ledger=ledger,
        source_bytes=bytes.fromhex(pair["source_hex"]),
    )
    assert result["source_coverage_status"] == "complete"
    source = b'{"records":[]}'
    result = verify_source_bound_witness(
        dict(facts=[], query=[["a", "r", "b"]]),
        record_ids=[],
        source_ledger=dict(
            records=[], source_sha256="sha256:" + hashlib.sha256(source).hexdigest()
        ),
        source_bytes=source,
    )
    assert (
        result["record_decision"] == "insufficient"
        and result["source_coverage_status"] == "complete"
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_ids",
        "duplicate_ids",
        "unknown_id",
        "altered_sign",
        "bad_hash",
        "source_bytes",
        "ledger_duplicate",
        "ledger_missing_id",
        "source_disagreement",
        "ledger_shape",
        "ledger_record_shape",
        "ledger_fact_shape",
        "source_type",
    ],
)
def test_malformed_provenance(corruption):
    """SCENARIO-REPORT-8897-CONTRACT: malformed provenance cannot grant coverage."""
    payload, kwargs = ledger_request()
    if corruption == "missing_ids":
        kwargs["record_ids"] = None
    elif corruption == "duplicate_ids":
        payload["facts"] *= 2
        kwargs["record_ids"] *= 2
    elif corruption == "unknown_id":
        kwargs["record_ids"] = ["foreign"]
    elif corruption == "altered_sign":
        payload = deepcopy(payload)
        payload["facts"][0][3] = False
    elif corruption == "bad_hash":
        kwargs["source_ledger"]["source_sha256"] = "sha256:bad"
    elif corruption == "source_bytes":
        kwargs["source_bytes"] += b" "
    elif corruption == "ledger_duplicate":
        kwargs["source_ledger"]["records"] *= 2
    elif corruption == "ledger_missing_id":
        kwargs["source_ledger"]["records"][0]["record_id"] = ""
    elif corruption == "source_disagreement":
        kwargs["source_ledger"]["records"][0]["fact"] = ["x", "r", "b", True]
    elif corruption == "ledger_shape":
        kwargs["source_ledger"] = []
    elif corruption == "ledger_record_shape":
        kwargs["source_ledger"]["records"] = [None]
    elif corruption == "ledger_fact_shape":
        kwargs["source_ledger"]["records"][0]["fact"] = ["a", "r", "b", 1]
    else:
        kwargs["source_bytes"] = "quoted natural text"
    result = verify_source_bound_witness(payload, **kwargs)
    assert result["source_coverage_status"] == "invalid"
    assert result["source_semantics_status"] == "unknown"
    assert result["ledger_errors"]
    assert result["record_decision"] == verify_size_aware_witness(payload)["decision"]


def test_frozen_pairs_and_reductions():
    """REQ-REPORT-8897: hidden conflict truth cannot leak into identical visible inputs."""
    worlds = m.freeze(set(), count=1)
    assert len(worlds) == 4 and worlds == m.freeze(set(), count=1)
    requests = m.requests(worlds)
    assert len(requests) == 4 * 5 * len(m.ARMS)
    observations = [m.observe(request) for request in requests]
    result = m.evaluate(worlds, observations)
    assert all(result["controls"].values())
    assert result["unsupported_claim_count"] == 0
    assert len(result["rows"]) == len(requests)
    assert len(result["ledger_validation_rows"]) == len(requests)
    for world in worlds:
        complete, *_, omitted = world["pairs"]
        assert complete["input"] == omitted["input"]
        assert m.old.truth(complete["source"]) != m.old.truth(omitted["source"])
    assert any(
        row["record_decision"] == "support" and row["source_coverage_status"] == "incomplete"
        for row in result["rows"]
    )
    damaged = deepcopy(observations)
    damaged[0]["record_decision"] = "refute"
    assert not all(m.evaluate(worlds, damaged)["controls"].values())
    damaged[0]["source_coverage_status"] = "complete"
    damaged[0]["source_semantics_status"] = "complete"
    assert m.evaluate(worlds, damaged)["unsupported_claim_count"] > 0
    assert len(m.scheduled(worlds, "blocked", "censored")) == len(requests)
    with pytest.raises(ValueError):
        m.evaluate(worlds, observations[:-1])


def cli(tmp_path, *args):
    """REQ-REPORT-8897: invoke the actual script without inherited checkout imports."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8897-CONTRACT: private success cannot self-qualify publication."""
    output = tmp_path / (m.NAME + ".json")
    result = cli(
        tmp_path,
        "--date",
        "20261107",
        "--private-output",
        "--output",
        str(output),
        "--private-world-limit",
        "1",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["source_boundary_ready_score"] == 0
    assert all(value["acceptance_gates"][g] for g in ["semantic_parity", "provenance_rejection"])
    assert value["installed_callable_receipt"]["passed"]
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    changed = deepcopy(value)
    changed["source_boundary_ready_score"] = 1
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", str(output)).returncode == 1
    atomic_json(output, value)
    raw_operand = Path(value["work_reference"]["path"]).parent / "worlds.json"
    raw_operand.chmod(0o600)
    raw_operand.write_text("{}")
    assert not r.replay(output)
    for arguments in [["--private-output"], ["--private-world-limit", "0"], ["--date", "20261106"]]:
        with pytest.raises(SystemExit) as rejected:
            r.main(arguments)
        assert rejected.value.code == 2


def test_external_authority_diagnostics(tmp_path):
    """REQ-REPORT-8897: absent, absent-field, observed-zero and rejected inputs differ."""
    raw = tmp_path / "raw"
    path = tmp_path / (r.SOURCE + ".json")
    assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "missing_file"
    for value, check in [
        ({}, "missing_field"),
        ({"boundary_rows_ready_score": 0}, "observed_zero"),
        ({"boundary_rows_ready_score": 1}, "rejected_producer"),
    ]:
        atomic_json(path, value)
        assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == check
    work = r.prepare(tmp_path / "blocked", tmp_path / "missing", 1)
    assert work["rows"] and all(row["status"] == "censored" for row in work["rows"])
    atomic_json(tmp_path / "blocked/work.json", work)
    validation = r.unqualified()
    atomic_json(tmp_path / "blocked/validation.json", validation)
    wr, vr = (
        r.reference(tmp_path / "blocked/work.json"),
        r.reference(tmp_path / "blocked/validation.json"),
    )
    value = r.artifact(work, validation, "20261107", wr, vr)
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    output = tmp_path / (m.NAME + ".json")
    atomic_json(output, value)
    assert r.replay(output)
    assert not r.replay(tmp_path / "absent.json")


def test_owned_failures_worker_and_validation(tmp_path, monkeypatch):
    """REQ-REPORT-8897: owned execution failures are retained and disqualified."""
    import runpy
    from scripts.experiments import source_bound_worker_8897 as worker

    requests = m.requests(m.freeze(set(), count=1))
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    atomic_json(request, dict(requests=requests))
    worker.main([str(request), str(response)])
    monkeypatch.setattr(sys, "argv", [str(m.ROOT / m.OWNED[-2]), str(request), str(response)])
    runpy.run_path(str(m.ROOT / m.OWNED[-2]), run_name="__main__")
    assert len(json.loads(response.read_bytes())["rows"]) == len(requests)
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="installed"):
        r.installed(tmp_path / "install-failed", requests)
    work = r.prepare(tmp_path / "owned-failed", m.ROOT / "results", 1)
    assert work["internal_errors"] and all(row["status"] == "failed" for row in work["rows"])
    from types import SimpleNamespace

    monkeypatch.setattr(
        r,
        "namespace",
        lambda *a: SimpleNamespace(
            validate=lambda *a: dict(called=True), terminal=lambda *a: dict(passed=True)
        ),
    )
    assert r.validate(tmp_path, tmp_path) == dict(called=True)
    assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]


def test_rehashed_private_work_cannot_invent_claims(tmp_path):
    """SCENARIO-REPORT-8897-CONTRACT: recompute claims instead of trusting aggregate hashes."""
    work = r.prepare(tmp_path / "raw", m.ROOT / "results", 1)
    validation = r.unqualified()
    atomic_json(tmp_path / "raw/work.json", work)
    atomic_json(tmp_path / "raw/validation.json", validation)
    wr, vr = r.reference(tmp_path / "raw/work.json"), r.reference(tmp_path / "raw/validation.json")
    output = tmp_path / (m.NAME + ".json")
    atomic_json(output, r.artifact(work, validation, "20261107", wr, vr))
    assert r.replay(output)
    work["rows"][0]["record_decision"] = "refute"
    atomic_json(tmp_path / "raw/work.json", work)
    wr = r.reference(tmp_path / "raw/work.json")
    atomic_json(output, r.artifact(work, validation, "20261107", wr, vr))
    assert not r.replay(output)


def test_execution_design_is_required(tmp_path, monkeypatch):
    """REQ-REPORT-8897: current execution authority must be bound before measurement."""
    monkeypatch.setattr(r, "DESIGN", str(tmp_path / "missing-design.json"))
    assert (
        r.preconditions(tmp_path / "missing", m.ROOT / "results")["diagnostics"][-1]["check"]
        == "missing_file"
    )
    atomic_json(tmp_path / "changed.json", dict(changed=True))
    monkeypatch.setattr(r, "DESIGN", str(tmp_path / "changed.json"))
    assert (
        r.preconditions(tmp_path / "changed", m.ROOT / "results")["diagnostics"][-1]["check"]
        == "rejected_producer"
    )


def test_hashed_text_and_malformed_typed_sources():
    """SCENARIO-REPORT-8897-CONTRACT: matching a text hash cannot grant source semantics."""
    for source in [
        b"quoted natural text",
        b"[]",
        b'{"records": [], "records":[]}',
        b'{"records":[],"extra":1}',
        b'{"records":[null]}',
    ]:
        payload, kwargs = ledger_request()
        kwargs["source_bytes"] = source
        kwargs["source_ledger"]["source_sha256"] = "sha256:" + hashlib.sha256(source).hexdigest()
        result = verify_source_bound_witness(payload, **kwargs)
        assert result["source_coverage_status"] == "invalid"
        assert result["source_semantics_status"] == "unknown"


def test_installed_provenance_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8897: a good child exit cannot excuse a foreign import or changed wheel."""
    requests = m.requests(m.freeze(set(), count=1))
    original_hash = r.sha256_file
    monkeypatch.setattr(
        r,
        "sha256_file",
        lambda path: "sha256:changed" if path.suffix == ".whl" else original_hash(path),
    )
    with pytest.raises(ValueError, match="original_bytes"):
        r.installed(tmp_path / "changed", requests)
    monkeypatch.setattr(r, "sha256_file", original_hash)
    raw = tmp_path / "foreign"

    def foreign(name, argv, logs, **kwargs):
        if name == "installed_callable":
            atomic_json(
                raw / "responses.json",
                dict(
                    rows=[],
                    imported=[
                        dict(path="/tmp/foreign.py", sha256="changed"),
                        dict(path="/tmp/foreign.py", sha256="changed"),
                    ],
                ),
            )
        return dict(passed=True)

    monkeypatch.setattr(r, "stream", foreign)
    with pytest.raises(ValueError, match="import_drift"):
        r.installed(raw, requests)


def test_historical_recovery_rejects_bad_bytes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8897-CONTRACT: historical recovery must match the original digest."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        r.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=b"changed", returncode=0)
    )
    result = r.preconditions(tmp_path / "raw", m.ROOT / "results")
    assert not result["external_ready"]
    assert "historical_input_drift" in result["diagnostics"][0]["error"]


def test_replay_rejects_world_receipt_and_validation_changes(tmp_path, monkeypatch):
    """REQ-REPORT-8897: a new aggregate digest cannot excuse changed execution evidence."""
    from types import SimpleNamespace

    raw = tmp_path / "raw"
    work = r.prepare(raw, tmp_path / "missing", 1)
    validation = r.unqualified()
    output = tmp_path / (m.NAME + ".json")

    def write():
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        atomic_json(
            output,
            r.artifact(
                work,
                validation,
                "20261107",
                r.reference(raw / "work.json"),
                r.reference(raw / "validation.json"),
            ),
        )

    original = deepcopy(work)
    work["worlds"][0]["edges"].append([9, 9])
    write()
    assert not r.replay(output)
    work = deepcopy(original)
    work["current_work_receipt"]["duration_s"] = -1
    write()
    assert not r.replay(output)
    work = deepcopy(original)
    validation.update(union=dict(passed=True), coverage=dict(complete=True), receipts=[])
    write()
    assert not r.replay(output)
    monkeypatch.setattr(
        r, "namespace", lambda *a: SimpleNamespace(validation_evidence=lambda v: True)
    )
    assert r.replay(output)


def test_publication_failure_stays_blocked_and_keeps_health(tmp_path, monkeypatch):
    """REQ-REPORT-8897: failed external inputs cannot gain readiness through validation."""
    work = r.prepare(tmp_path / "source", tmp_path / "missing", 1)
    monkeypatch.setattr(r, "prepare", lambda *a: work)
    monkeypatch.setattr(r, "validate", lambda *a: r.unqualified())
    monkeypatch.setattr(r, "terminal", lambda *a: dict(passed=True))
    health = tmp_path / "health.json"
    atomic_json(health, dict(status="failed", scope="unrelated_global_health"))
    monkeypatch.setattr(r, "HEALTH", health)
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["repository_health"]["status"] == "failed"
    assert any(
        Path(ref["path"]).name == "repository_health.json" for ref in value["raw_shard_hashes"]
    )
    assert value["verdict_class"] == "blocked" and value["source_boundary_ready_score"] == 0
