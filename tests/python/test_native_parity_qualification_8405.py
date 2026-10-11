"""REQ-REPORT-8405 / REQ-VERIFY-8405: real typing, native and coverage controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys

import coverage
import pytest

from carnot.reporting import native_evidence_8391 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_exact_typing_repair(tmp_path):
    """SCENARIO-VERIFY-8405-TYPING-COVERAGE: strict typing accepts the real summary."""
    receipt = child(
        "typing_regression",
        [
            str(old.ROOT / ".venv/bin/mypy"),
            "--config-file=/dev/null",
            "--strict",
            "--follow-imports=skip",
            "--ignore-missing-imports",
            *old.OWNED,
        ],
        tmp_path,
        deadline=60,
    )
    assert receipt["passed"], Path(receipt["stdout_path"]).read_text()
    original = json.loads(
        (old.ROOT / "results/experiment_8391_v723_native_evidence_qualification.json").read_bytes()
    )
    failed = next(r for r in original["validation_receipts"] if r["name"] == "strict_mypy")
    assert not failed["passed"]
    assert 'Need type annotation for "summary"' in Path(failed["stdout_path"]).read_text()


@pytest.fixture(scope="module")
def e():
    """REQ-REPORT-8405: import the new adapter only after its test exists."""
    from carnot.reporting import native_parity_8405

    return native_parity_8405


@pytest.fixture(scope="module")
def measured(e, tmp_path_factory):
    """REQ-VERIFY-8405: execute shipped compiled arithmetic, never mock measurements."""
    private = tmp_path_factory.mktemp("native8405")
    return e.measure(e.ROOT, private / "raw", private)


def test_native_operands_and_reduction(e, measured):
    """REQ-VERIFY-8405: retain the entire frozen finite panel and ownership boundary."""
    assert measured["current_authority"]["activated"]
    assert measured["extension"]["actual_loaded"]
    assert measured["native_invocation_count"] > 0
    assert len(measured["rows"]) == 4564
    assert sum(r["action_mismatch"] or 0 for r in measured["rows"]) == 0
    assert measured["coefficient_reduction"]["recomputed"]
    assert measured["coefficient_reduction"]["deliberate_error_rejected"]
    assert measured["coefficient_reduction"]["paired_update_count"] == 192
    assert measured["coefficient_reduction"]["max_error"] == 0
    assert measured["historical_consumers"]["ready"]
    assert measured["coverage_failure_reproduction"]["passed"]


def test_bound_inputs_and_missing(e, tmp_path, monkeypatch):
    """REQ-REPORT-8405: current task and pinned protocol mutations never authorize work."""
    protocol, refs, gates = e.bind(e.ROOT, tmp_path / "valid")
    assert protocol and refs and not gates
    monkeypatch.setattr(e, "TASK_PIN", "sha256:wrong")
    assert e.bind(e.ROOT, tmp_path / "wrong")[2]
    monkeypatch.undo()
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["native_invocation_count"] == 0
    assert value["gate_check_summary"][0]["observed"] is None
    assert value["censored_count"] == 4564
    source = e.current.PROTOCOL
    actual = e.reference
    monkeypatch.setattr(
        e,
        "reference",
        lambda p: dict(actual(p), sha256="sha256:wrong") if p == e.ROOT / source else actual(p),
    )
    assert any(g["check"] == "v724_input_hash" for g in e.bind(e.ROOT, tmp_path / "mutated")[2])


def test_reduction_negative(e, measured):
    """SCENARIO-REPORT-8405-REPLAY: false zero, missing and unequal operands fail."""
    rows = deepcopy(measured["rows"])
    row = next(r for r in rows if r["kind"] == "numeric_update")
    row["native_coefficients"][0] += 0.125
    assert not e.coefficient_reduction(rows)["recomputed"]
    assert not e.coefficient_reduction([])["recomputed"]
    work = dict(measured, rows=rows)
    assert e.build(work, [dict(passed=True)])["verdict_class"] == "disqualified"


def test_frozen_plan(e, tmp_path, monkeypatch):
    """REQ-VERIFY-8405: invocation checks use genuine child coverage and private consumers."""
    for key in ("COVERAGE_RCFILE", "COVERAGE_FILE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    plan = e.plan(tmp_path)
    assert not any(p["scope"] == "global" for p in plan)
    assert "--no-cov" in plan[0]["argv"]
    assert any(p["name"] == "private_E2E018_consumers" for p in plan)
    assert "--keep" in next(p for p in plan if p["name"] == "coverage_combine")["argv"]
    assert (
        "python/carnot/reporting/native_evidence_8391.py"
        in next(p for p in plan if p["name"] == "strict_mypy")["argv"]
    )


def test_real_shard_identity(e, tmp_path):
    """SCENARIO-VERIFY-8405-TYPING-COVERAGE: genuine SQLite identities precede combine."""
    data = coverage.CoverageData(basename=str(tmp_path / ".coverage.unit"))
    data.add_lines({str(e.ROOT / e.OWNED[0]): {1}})
    data.write()
    assert e.shard_identity(Path(data.data_filename()))["measured_files"]
    corrupt = tmp_path / "corrupt"
    corrupt.write_bytes(b"not a coverage database")
    with pytest.raises((ValueError, coverage.CoverageException)):
        e.shard_identity(corrupt)
    empty = tmp_path / "empty.sqlite"
    empty.write_bytes(Path(data.data_filename()).read_bytes())
    with sqlite3.connect(empty) as database:
        database.execute("DELETE FROM line_bits")
        database.execute("DELETE FROM file")
    assert e.shard_identity(empty)["measured_files"] == []
    assert e.retain_shards(tmp_path) == 1
    assert not e.coverage_evidence(tmp_path)["passed"]


def test_replay_and_cli_controls(e, measured, tmp_path):
    """SCENARIO-REPORT-8405-REPLAY: real children reject invented invocation counts."""
    path = tmp_path / (e.NAME + ".json")
    value = e.build(measured, [dict(passed=True)])
    assert value["native_parity_ready_score"] == 0
    atomic_json(path, value)
    assert e.replay(path)
    assert not e.replay(tmp_path / "absent")
    bad = deepcopy(value)
    bad["native_invocation_count"] += 1
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(tmp_path / "tamper.json", bad)
    for name, args, expected in [
        ("valid", ["--cold-replay", str(path)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("error", ["--deliberate-error"], 1),
        ("date", ["--date", "20000101"], 2),
        ("tamper", ["--cold-replay", str(tmp_path / "tamper.json")], 1),
        ("custody", ["--coverage-custody", str(tmp_path)], 1),
        ("custody_bad", ["--coverage-custody"], 1),
    ]:
        assert child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            tmp_path / "logs",
            expected=expected,
            deadline=120,
        )["passed"]
    assert e.main(["--coverage-custody"]) == 1


def test_private_missing_publication(e, tmp_path, monkeypatch):
    """REQ-REPORT-8405: a real private CLI publishes honest absence and preserves history."""
    before = {p: e.reference(e.ROOT / p) for p in e.PRESERVED}
    monkeypatch.setattr(e, "plan", lambda private: [])
    output = tmp_path / (e.NAME + ".json")
    assert (
        e.main(["--date", "20261011", "--root", str(tmp_path / "absent"), "--output", str(output)])
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["run_date"] == "20261011"
    assert before == {p: e.reference(e.ROOT / p) for p in e.PRESERVED}


def test_gates_and_actual_coverage(e, measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8405: coverage reports must match actual databases and receipt failures."""
    with monkeypatch.context() as patch:
        patch.setattr(e, "ROOT", tmp_path)
        patch.setattr(e, "OWNED", ["probe.py"])
        patch.setattr(e.old, "ROOT", tmp_path)
        patch.setattr(e.old, "OWNED", ["probe.py"])
        probe = tmp_path / "probe.py"
        probe.write_text('print("child")\n')
        config = tmp_path / "probe.ini"
        config.write_text(
            f"[run]\nparallel=true\ndata_file={tmp_path / '.coverage'}\ninclude={probe}\n"
        )
        patch.setenv("COVERAGE_FILE", str(tmp_path / ".coverage"))
        patch.setenv("COVERAGE_RCFILE", str(config))
        for i in range(2):
            assert child(
                f"shard_{i}",
                [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config), str(probe)],
                tmp_path / "logs",
            )["passed"]
        assert e.retain_shards(tmp_path) == 0
        for name, args in [
            ("combine", ["combine", "--keep", "--rcfile=" + str(config), str(tmp_path)]),
            ("json", ["json", "--rcfile=" + str(config), "-o", str(tmp_path / "coverage.json")]),
        ]:
            assert child(name, [sys.executable, "-m", "coverage", *args], tmp_path / "logs")[
                "passed"
            ]
        assert e.old.coverage_evidence(tmp_path)["passed"]
        assert e.coverage_evidence(tmp_path)["passed"]
        identities = tmp_path / "coverage_shards/identities.json"
        original = identities.read_bytes()
        changed = json.loads(original)
        changed["invocation"] = "wrong invocation"
        atomic_json(identities, changed)
        assert not e.coverage_evidence(tmp_path)["passed"]
        identities.write_bytes(original)
        report = json.loads((tmp_path / "coverage.json").read_bytes())
        next(iter(report["files"].values()))["summary"]["num_statements"] += 1
        atomic_json(tmp_path / "coverage.json", report)
        assert not e.old.coverage_evidence(tmp_path)["passed"]
        assert not e.coverage_evidence(tmp_path)["passed"]
    monkeypatch.setattr(
        e, "coverage_evidence", lambda private: dict(passed=True, manifest=[], mode="test_control")
    )
    assert e.build(measured, [dict(passed=True)])["native_parity_ready_score"] == 1
    assert e.build(measured, [dict(passed=False)])["verdict_class"] == "disqualified"


def test_authority_tamper(e, measured, tmp_path):
    """SCENARIO-REPORT-8405-REPLAY: rehashed authority cannot replace independent design."""
    work = deepcopy(measured)
    work["current_authority"]["activated"] = False
    atomic_json(tmp_path / "measurement.json", work)
    value = e.build(work, [dict(passed=True)])
    value["measurement_reference"] = e.reference(tmp_path / "measurement.json")
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(tmp_path / "candidate.json", value)
    assert not e.replay(tmp_path / "candidate.json")
    work = deepcopy(measured)
    primitive = json.loads(Path(work["primitive_reference"]["path"]).read_bytes())
    primitive["rows"][0]["native_probability"] += 0.125
    atomic_json(tmp_path / "primitive.json", primitive)
    work["primitive_reference"] = e.reference(tmp_path / "primitive.json")
    work["rows"] = primitive["rows"]
    atomic_json(tmp_path / "measurement.json", work)
    value = e.build(work, [dict(passed=True)])
    value["measurement_reference"] = e.reference(tmp_path / "measurement.json")
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(tmp_path / "candidate.json", value)
    assert not e.replay(tmp_path / "candidate.json")


def test_unchanged_finding_consumer(e, measured, tmp_path):
    """REQ-REPORT-8405: a real paired proof cannot broaden the shipped finding policy."""
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, e.build(measured, [dict(passed=True)]))
    found = e.audit(candidate, tmp_path / "finding", {})
    assert not found["passed"]
    assert found["findings"][0]["kind"] == "IMPLAUSIBLE_PERFECT"
    assert not found["dispositions"][0]["resolved"]
    assert found["dispositions"][0]["evidence"]["recomputed"]
