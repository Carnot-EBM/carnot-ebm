"""REQ-REPORT-8391 / REQ-VERIFY-8391: real children and immutable operands."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting import native_evidence_8391 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """REQ-VERIFY-8391: exercise the actual extension and frozen finite panel."""
    private = tmp_path_factory.mktemp("native8391")
    return e.measure(e.ROOT, private / "raw", private)


def test_native_and_authority(measured):
    """REQ-VERIFY-8391: private E2E-003 crosses compiled PyO3 entrypoints."""
    assert measured["current_authority"]["activated"]
    assert measured["extension"]["actual_loaded"]
    assert measured["native_invocation_count"] > 0
    assert measured["binding_copy_bytes"] > 0
    assert len(measured["rows"]) == 4564
    assert sum(r["action_mismatch"] or 0 for r in measured["rows"]) == 0
    assert measured["historical_consumers"]["missing_design_stays_blocked"]
    assert measured["historical_consumers"]["verdict_class"] == "blocked"
    assert measured["historical_consumers"]["ready"]
    assert measured["coverage_failure_reproduction"]["passed"]


def test_coverage_required(measured, tmp_path):
    """SCENARIO-VERIFY-8391-COVERAGE: no instrumentation cannot qualify parity."""
    value = e.build(measured, [dict(passed=True)])
    assert value["native_parity_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    private = tmp_path / "coverage"
    private.mkdir()
    assert not e.coverage_evidence(private)["passed"]
    atomic_json(
        private / "coverage.json", dict(files={}, totals=dict(num_statements=1, missing_lines=0))
    )
    assert not e.coverage_evidence(private)["passed"]


def test_missing_input(tmp_path):
    """REQ-REPORT-8391: external absence stays blocked with no invented calls."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["native_parity_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"] is None


def test_replay_and_tamper(measured, tmp_path):
    """SCENARIO-REPORT-8391-REPLAY: rehashed claims cannot replace primitives."""
    path = tmp_path / (e.NAME + ".json")
    value = e.build(measured, [dict(passed=True)])
    atomic_json(path, value)
    assert e.replay(path)
    assert not e.replay(tmp_path / "absent")
    for field in ("native_invocation_count", "completed_count", "native_parity_ready_score"):
        bad = deepcopy(value)
        bad[field] += 1
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    atomic_json(path, value)
    for name, args, expected in [
        ("valid", ["--cold-replay", str(path)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("error", ["--deliberate-error"], 1),
        ("date", ["--date", "20000101"], 2),
    ]:
        receipt = child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            tmp_path / "logs",
            expected=expected,
            deadline=120,
        )
        assert receipt["passed"]


def test_plan(tmp_path, monkeypatch):
    """REQ-VERIFY-8391: freeze coverage ownership and named private consumers."""
    for key in ("COVERAGE_RCFILE", "COVERAGE_FILE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    plan = e.plan(tmp_path)
    assert not any(p["scope"] == "global" for p in plan)
    assert any(p["name"] == "private_E2E018_consumers" for p in plan)
    combine = next(p for p in plan if p["name"] == "coverage_combine")
    assert "--keep" in combine["argv"]
    assert "--no-cov" in plan[0]["argv"]


def test_private_main_and_source_isolation(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8391-REPLAY: real missing-input publication preserves sources."""
    before = {p: e.reference(e.ROOT / p) for p in e.PRESERVED}
    monkeypatch.setattr(e, "plan", lambda private: [])
    path = tmp_path / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(path)]) == 0
    assert json.loads(path.read_bytes())["verdict_class"] == "blocked"
    assert before == {p: e.reference(e.ROOT / p) for p in e.PRESERVED}


def test_real_coverage_database_and_shards(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8391-COVERAGE: real children prove both report and data checks."""
    probe = tmp_path / "probe.py"
    probe.write_text('print("real child coverage")\n')
    config = tmp_path / "probe.ini"
    config.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(tmp_path / ".coverage")
        + "\ninclude="
        + str(probe)
        + "\n"
    )
    monkeypatch.setenv("COVERAGE_FILE", str(tmp_path / ".coverage"))
    for i in range(2):
        receipt = child(
            f"coverage_child_{i}",
            [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config), str(probe)],
            tmp_path / "logs",
        )
        assert receipt["passed"]
    assert e.retain_shards(tmp_path) == 0
    assert e.retain_shards(tmp_path / "absent") == 1
    assert e.main(["--coverage-custody", str(tmp_path)]) == 0
    assert e.main(["--coverage-custody"]) == 1
    for name, args in [
        ("combine", ["combine", "--keep", "--rcfile=" + str(config), str(tmp_path)]),
        ("json", ["json", "--rcfile=" + str(config), "-o", str(tmp_path / "coverage.json")]),
    ]:
        assert child(name, [sys.executable, "-m", "coverage", *args], tmp_path / "logs")["passed"]
    monkeypatch.setattr(e, "ROOT", tmp_path)
    monkeypatch.setattr(e, "OWNED", ["probe.py"])
    report = json.loads((tmp_path / "coverage.json").read_bytes())
    report["files"] = {str(probe): next(iter(report["files"].values()))}
    atomic_json(tmp_path / "coverage.json", report)
    assert e.coverage_evidence(tmp_path)["passed"]
    report = json.loads((tmp_path / "coverage.json").read_bytes())
    report["files"][str(probe)]["summary"]["num_statements"] += 1
    atomic_json(tmp_path / "coverage.json", report)
    assert not e.coverage_evidence(tmp_path)["passed"]


def test_reduction_gate_branches(measured, monkeypatch, tmp_path):
    """REQ-VERIFY-8391: failed consumers and missing instrumentation clear readiness."""
    shard = tmp_path / "unit.coverage"
    shard.write_bytes(b"unit reduction control; never measured instrumentation")
    monkeypatch.setattr(
        e,
        "coverage_evidence",
        lambda private: dict(
            passed=True, mode="unit_control", manifest=[e.reference(shard)], report_reference=None
        ),
    )
    value = e.build(measured, [dict(passed=True, name="coverage_unit_control")])
    assert value["verdict_class"] == "circular_positive"
    assert value["native_parity_ready_score"] == 1
    assert e.build(measured, [dict(passed=False)])["verdict_class"] == "disqualified"


def test_authenticated_input_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8391: changed protocol and operand bytes cannot grant authority."""
    actual = e.reference
    protocol = json.loads((e.ROOT / e.base.authority.PROTOCOL).read_bytes())
    target = Path(protocol["checkpoint"]["path"])
    monkeypatch.setattr(
        e,
        "reference",
        lambda p: dict(actual(p), sha256="sha256:wrong") if p == target else actual(p),
    )
    _, _, gates = e.bind(e.ROOT, tmp_path / "operand")
    assert any(g["check"] == "operand_hash" for g in gates)
    target = e.ROOT / e.authority.METHODS
    _, _, gates = e.bind(e.ROOT, tmp_path / "protocol")
    assert any(g["check"] == "input_hash" for g in gates)


def test_rehashed_authority_and_primitive_controls(measured, tmp_path):
    """SCENARIO-REPORT-8391-REPLAY: independently recomputed operands defeat rehashing."""
    candidate = tmp_path / "candidate.json"
    changed = deepcopy(measured)
    changed["current_authority"]["activated"] = False
    atomic_json(tmp_path / "measurement.json", changed)
    original = e.build(changed, [dict(passed=True)])
    original["measurement_reference"] = e.reference(tmp_path / "measurement.json")
    original.pop("reproducibility_checksum")
    original["reproducibility_checksum"] = canonical_hash(original)
    atomic_json(candidate, original)
    assert not e.replay(candidate)
    primitive = json.loads(Path(measured["primitive_reference"]["path"]).read_bytes())
    primitive["rows"][0]["native_probability"] += 0.125
    atomic_json(tmp_path / "primitive.json", primitive)
    changed = deepcopy(measured)
    changed["primitive_reference"] = e.reference(tmp_path / "primitive.json")
    atomic_json(tmp_path / "measurement.json", changed)
    value = e.build(changed, [dict(passed=True)])
    value["measurement_reference"] = e.reference(tmp_path / "measurement.json")
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(candidate, value)
    assert not e.replay(candidate)
