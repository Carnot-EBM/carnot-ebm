"""REQ-REPORT-8715 and REQ-VERIFY-8715: audit independent constructed semantics."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import verifier_completeness_8715 as e
from carnot.reporting import verifier_completeness_runner_8715 as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def worlds():
    """SCENARIO-REPORT-8715-SEMANTICS: implementation uses only the90 training IDs."""
    p = e.upstream.load(
        dict(path=str(e.ROOT / e.upstream.PROTOCOL), sha256=e.upstream.PROTOCOL_PIN)
    )
    rows = e.load(p["relational_evidence_protocol"]["worlds_reference"])["worlds"]
    return [w for i, w in enumerate(rows) if (i + 1) % 4]


def test_semantics_and_probes(tmp_path):
    """SCENARIO-REPORT-8715-SEMANTICS: no tested verifier supplies semantic labels."""
    training = worlds()
    frozen = e.freeze(training)
    assert len(frozen["worlds"]) == 90
    measured = e.measure(frozen, tmp_path)
    assert len(measured["rows"]) == 540
    assert all(x["extended_passed"] for x in measured["rows"] if x["variant"] == "correct")
    assert all(
        not x["baseline_passed"]
        for x in measured["rows"]
        if x["variant"] in {"reject_all", "accept_all"}
    )
    assert any(
        x["extended_detected"] and not x["baseline_detected"]
        for x in measured["defect_detection_rows"]
    )
    assert all(x["extended_probes"] > x["baseline_probes"] for x in measured["probe_cost_rows"])
    assert measured["unresolved_defect_witnesses"]
    assert any(
        w["witness"]
        and w["witness"]["transform"] == "required_role"
        and w["witness"]["expected"] == 0
        for w in measured["unresolved_defect_witnesses"]
    )
    assert e.oracle([], None) == -1
    with pytest.raises(ValueError, match="variant"):
        e.verifier("invalid", [], None)
    assert e.verifier("correct", [[0, 1, 1]], None) == -1


def test_inputs(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8715-CUSTODY: missing, changed and flagged producers cannot qualify."""
    branch = e.inputs(e.ROOT)
    assert branch["passed"], branch["failures"]
    assert e.authority.authority(e.ROOT)["ready"]
    assert not e.inputs(tmp_path)["passed"]
    monkeypatch.setattr(e, "PRODUCER_PIN", "sha256:changed")
    assert not e.inputs(e.ROOT)["passed"]


def test_private_cli_and_replay(tmp_path):
    """SCENARIO-VERIFY-8715-CUSTODY: execute actual private publication and rejection children."""
    output = tmp_path / (e.NAME + ".json")
    args = [sys.executable, "-u", e.CLI, "--date", "20261102"]
    assert child("private", args + ["--private-output", str(output)], tmp_path / "logs")["passed"]
    assert r.replay(output)
    original = output.read_bytes()
    value = json.loads(original)
    assert value["verdict_class"] == "null"
    assert value["completed_count"] == 720
    assert len(value["heldout_world_ids"]) == 30
    for mode in ["checksum", "report", "seal", "source", "primitives", "input"]:
        value = json.loads(original)
        primitive = Path(value["primitive_reference"]["path"])
        pristine = primitive.read_bytes()
        work = json.loads(pristine)
        if mode == "checksum":
            value["reproducibility_checksum"] = "changed"
        elif mode == "report":
            value["probe_rows"][0]["actual"] = 99
            value["reproducibility_checksum"] = e.checksum(value)
        else:
            if mode == "seal":
                work["duration_s"] += 1
            elif mode == "source":
                work["sources"][0]["sha256"] = "changed"
            elif mode == "primitives":
                work["frozen"]["worlds"][0]["probes"][0]["expected"] = 99
            else:
                work["inputs"]["passed"] = False
            if mode != "seal":
                seal = tmp_path / ("seal-" + mode + ".json")
                atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
                work["seal_reference"] = e.reference(seal)
            atomic_json(primitive, work)
            value["primitive_reference"] = e.reference(primitive)
            value["reproducibility_checksum"] = e.checksum(value)
        atomic_json(output, value)
        assert not r.replay(output), mode
        primitive.write_bytes(pristine)
    output.write_bytes(original)
    assert child(
        "reject", args + ["--cold-replay", str(tmp_path / "absent")], tmp_path / "logs", expected=1
    )["passed"]
    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--private-output", str(e.ROOT / "results" / "unsafe.json")]) == 2
    assert r.main(["--cold-replay", str(output)]) == 0


def test_report_verdicts_and_empty_measurement(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8715-SEMANTICS: readiness survives zero gain but never failed checks."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, private=True) == 0
    value = json.loads(output.read_bytes())
    work = e.load(value["primitive_reference"])
    work["private_control"] = False
    assert e.build(work, output)["verdict_class"] == "disqualified"
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    assert e.build(work, output)["verdict_class"] == "circular_positive"
    work["inputs"]["passed"] = False
    assert e.build(work, output)["verdict_class"] == "blocked"
    work["inputs"]["passed"] = True
    for row in work["measurement"]["defect_detection_rows"]:
        row["baseline_detected"] = row["extended_detected"]
    assert e.build(work, output)["verdict_class"] == "null"
    work["measurement"] = e.measure(e.freeze([]), tmp_path / "empty")
    work["resources"]["ready"] = False
    blocked = e.build(work, output)
    assert blocked["censored_count"] == 720
    assert len(blocked["rows"]) == 720
    assert not blocked["verifier_audit_ready_score"]
    monkeypatch.setattr(r, "run", lambda *args, **kwargs: 0)
    assert r.main([]) == 0


def test_validation_custody_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8715-CUSTODY: exercise real child retention and coverage rejection."""
    monkeypatch.setattr(e, "TESTS", ["tests/python/test_primary_publication_7928.py"])
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = r.validate(tmp_path / "validation", scratch, [e.reference(e.ROOT / e.OWNED[0])])
    assert result["custody"]["passed"], result["custody"]
    assert not result["coverage"]["complete"]
    assert all(x["passed"] for x in result["receipts"])
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, private=True) == 0
    value = json.loads(output.read_bytes())
    primitive = Path(value["primitive_reference"]["path"])
    work = e.load(value["primitive_reference"])
    work.update(private_control=False, validation=result)

    def save():
        seal = tmp_path / "seal.json"
        atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
        work["seal_reference"] = e.reference(seal)
        atomic_json(primitive, work)
        atomic_json(output, e.build(work, output))

    save()
    assert r.replay(output)
    work["validation"]["coverage"]["complete"] = True
    save()
    assert not r.replay(output)
    work["validation"]["coverage"]["complete"] = False
    work["validation"]["custody"]["passed"] = False
    save()
    assert not r.replay(output)
    monkeypatch.setattr(r.shared, "CONSUMERS", ["absent::node"])
    assert not r.validate(tmp_path / "empty", tmp_path, [])["custody"]["passed"]


def test_measurement_and_caps(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8715-CUSTODY: reject changed measurements and stop overruns."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, private=True) == 0
    value = json.loads(output.read_bytes())
    primitive = Path(value["primitive_reference"]["path"])
    work = e.load(value["primitive_reference"])
    work["measurement"]["probe_rows"][0]["actual"] = 99
    seal = tmp_path / "changed-seal.json"
    atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
    work["seal_reference"] = e.reference(seal)
    atomic_json(primitive, work)
    value["primitive_reference"] = e.reference(primitive)
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not r.replay(output)
    monkeypatch.setattr(e, "measure", lambda *args: dict(rows=[]))
    monkeypatch.setattr(r.time, "monotonic_ns", iter([0, 3601 * 10**9]).__next__)
    with pytest.raises(ValueError, match="measurement_cap"):
        r.run(tmp_path / "cap" / (e.NAME + ".json"), private=True)
