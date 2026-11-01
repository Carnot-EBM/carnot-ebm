"""REQ-REPORT-8701: retain lexical counterexamples without claiming natural truth."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import relational_feature_8701 as e
from carnot.reporting import relational_feature_runner_8701 as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def operands():
    """Use immutable supplied worlds and checkpoint, never output-derived truth."""
    p = e.upstream.load(
        dict(path=str(e.ROOT / e.upstream.PROTOCOL), sha256=e.upstream.PROTOCOL_PIN)
    )
    return p, e.upstream.load(p["relational_evidence_protocol"]["worlds_reference"])["worlds"]


def test_sqlite_and_error_categories(tmp_path):
    """SCENARIO-REPORT-8701-CAPACITY: missing facts escalate and faults stay distinct."""
    db = tmp_path / "world.sqlite"
    assert e.sql_decision(db, [[0, 1, 1]], [0, 1, 1]) == 1
    assert e.sql_decision(db, [[0, 1, 1]], [0, 1, 0]) == 0
    assert e.sql_decision(db, [], [0, 1, 1]) == -1
    assert e.sql_decision(db, [[0, 1, 1]], [[0, 1, 1], [0, 2, 0]]) == -1
    assert e.sql_decision(db, [], None) == -1
    assert e.diagnose([[0, 1, 1]], [], [0, 1, 1], [0, 1, 1], -1, -1) == ["extraction_omission"]
    assert e.diagnose([[0, 1, 1]], [[0, 1, 0]], [0, 1, 1], [0, 1, 1], 0, 0) == ["wrong_relation"]
    assert e.diagnose([], [], [0, 1, 1], [1, 0, 1], -1, -1) == ["wrong_query"]
    assert e.diagnose([], [], None, None, 0, -1) == ["wrong_decision"]
    assert e.diagnose([], [], None, None, -1, -1) == []


def test_full_corpus_and_missing_numerical_operand(tmp_path):
    """SCENARIO-REPORT-8701-CAPACITY: all views, chunks and oracle controls survive."""
    p, worlds = operands()
    result = e.measure(worlds, e.upstream.load(p["head_checkpoint_reference"]), tmp_path)
    assert len(result["rows"]) == 480
    assert result["symbolic_capacity_passed"]
    assert result["numerical_completed_count"] == 0
    assert len(result["view_alignment_rows"]) == 120
    assert len(result["collision_rows"]) == 120
    assert all(x["exact_equal"] and x["opposite_truth"] for x in result["collision_rows"])
    assert all(x["chunk_parity"] and x["truth_equivalent"] for x in result["view_alignment_rows"])
    assert {x["arm"] for x in result["symbolic_control_rows"]} == {
        "complete_records",
        "deleted_required_records",
        "reversed_relations",
    }
    assert {c for x in result["extraction_error_rows"] for c in x["categories"]} == {
        "extraction_omission",
        "wrong_relation",
    }
    assert {c for x in result["query_error_rows"] for c in x["categories"]} == {
        "wrong_query",
        "wrong_decision",
    }
    assert all(x["head"]["operand_state"] == "missing_field" for x in result["rows"])


def test_frozen_head_and_corrupt_views(tmp_path):
    """SCENARIO-REPORT-8701-CAPACITY: authentic arithmetic needs all nine operands."""
    p, worlds = operands()
    head = e.upstream.load(p["head_checkpoint_reference"])
    assert e.numerical([0.0] * 8, None, head)["probability"] is None
    actual = e.numerical([0.0] * 8, 0.5, head)
    assert 0 <= actual["probability"] <= 1
    assert actual["decision"] in {0, 1, -1}
    changed = deepcopy(worlds[:1])
    changed[0]["views"][0]["truth"] = 0
    result = e.measure(changed, head, tmp_path)
    assert not result["symbolic_capacity_passed"]


def test_branch_custody(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8701-CUSTODY: original inputs and accepted producer bind readiness."""
    inputs = e.inputs(e.ROOT)
    assert inputs["passed"], inputs["failures"]
    assert e.authority.authority(e.ROOT)["ready"]
    monkeypatch.setattr(e, "CORPUS_PIN", "changed")
    assert not e.inputs(e.ROOT)["passed"]
    assert not e.inputs(tmp_path)["passed"]


def test_private_cli_replay_and_guards(tmp_path):
    """SCENARIO-REPORT-8701-CUSTODY: actual private success and rehashed rejection."""
    output = tmp_path / (e.NAME + ".json")
    args = [sys.executable, "-u", e.CLI]
    assert child("private", args + ["--private-output", str(output)], tmp_path / "logs")["passed"]
    assert r.replay(output)
    original = output.read_bytes()
    value = json.loads(original)
    assert value["verdict_class"] == "null"
    value["rows"][0]["truth"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert child("reject", args + ["--cold-replay", str(output)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    output.write_bytes(original)
    primitive = Path(json.loads(original)["primitive_reference"]["path"])
    pristine = primitive.read_bytes()
    work = json.loads(pristine)
    work["measurement"]["rows"][0]["truth"] = 0
    atomic_json(
        Path(work["seal_reference"]["path"]),
        {k: v for k, v in work.items() if k != "seal_reference"},
    )
    work["seal_reference"] = e.reference(Path(work["seal_reference"]["path"]))
    atomic_json(primitive, work)
    value = json.loads(original)
    value["primitive_reference"] = e.reference(primitive)
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not r.replay(output)
    primitive.unlink()
    assert not r.replay(output)
    assert r.main(["--date", "bad"]) == 2
    assert r.main(["--private-output", str(e.ROOT / "results/forbidden.json")]) == 2
    assert r.main(["--cold-replay", str(output)]) == 1


def test_build_failure_classes(tmp_path):
    """SCENARIO-REPORT-8701-CUSTODY: external absence blocks, owned failures disqualify."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, private=True) == 0
    work = e.load(json.loads(output.read_bytes())["primitive_reference"])
    work["private_control"] = False
    assert e.build(work, output)["verdict_class"] == "disqualified"
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    assert e.build(work, output)["verdict_class"] == "blocked"
    work["measurement"]["numerical_completed_count"] = 480
    assert e.build(work, output)["verdict_class"] == "circular_positive"
    work["measurement"]["symbolic_capacity_passed"] = False
    assert e.build(work, output)["symbolic_control_ready_score"] == 0
    work["resources"]["ready"] = False
    assert any(
        g["check"] == "disk_memory_scratch" for g in e.build(work, output)["gate_check_summary"]
    )


def test_registered_operand_and_corpus_replacement(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8701-CUSTODY: rehashed registered values cannot change assertions."""
    branch = e.upstream.inputs(e.ROOT)
    protocol, _ = operands()
    protocol["relational_evidence_protocol"]["chunk_sizes"] = [1]
    path = tmp_path / e.upstream.PROTOCOL
    path.parent.mkdir(parents=True)
    atomic_json(path, protocol)
    monkeypatch.setattr(e.upstream, "PROTOCOL_PIN", e.reference(path)["sha256"])
    monkeypatch.setattr(e.upstream, "inputs", lambda root: deepcopy(branch))
    corpus = e.load(dict(path=str(e.ROOT / e.CORPUS), sha256=e.CORPUS_PIN))
    corpus["rows"][0]["actual_truth"] = 0
    target = tmp_path / e.CORPUS
    target.parent.mkdir(parents=True)
    atomic_json(target, corpus)
    monkeypatch.setattr(e, "CORPUS_PIN", e.reference(target)["sha256"])
    result = e.inputs(tmp_path)
    assert not result["passed"]
    assert {g["check"] for g in result["failures"]} >= {
        "registered_method_values",
        "original_480_rows",
    }


def test_validation_retention_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8701-CUSTODY: retain real disjoint children and reject bad custody."""
    monkeypatch.setattr(e, "TESTS", ["tests/python/test_primary_publication_7928.py"])
    raw, scratch = tmp_path / "validation", tmp_path / "scratch"
    scratch.mkdir()
    result = r.validate(raw, scratch, [e.reference(e.ROOT / e.OWNED[0])])
    assert result["custody"]["passed"], result["custody"]
    assert result["coverage"]["statements"] > 0
    assert not result["coverage"]["complete"]
    assert all(x["passed"] for x in result["receipts"])
    # A mechanics replay uses the original world operands; it cannot claim readiness.
    output = tmp_path / (e.NAME + ".json")
    r.run(output, private=True)
    original = json.loads(output.read_bytes())
    work = e.load(original["primitive_reference"])
    work.update(private_control=False, validation=result)

    def save():
        seal = tmp_path / "seal.json"
        atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
        work["seal_reference"] = e.reference(seal)
        atomic_json(Path(original["primitive_reference"]["path"]), work)
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
    monkeypatch.setattr(r, "CONSUMERS", ["absent::node"])
    assert not r.validate(tmp_path / "empty", tmp_path, [])["custody"]["passed"]


def test_replay_early_boundaries(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8701-CUSTODY: malformed, unsealed and changed sources fail early."""
    output = tmp_path / "bad.json"
    output.write_text("{}")
    assert not r.replay(output)
    atomic_json(output, dict(reproducibility_checksum="bad"))
    assert not r.replay(output)
    fake_work = dict(
        seal_reference={},
        sources=[],
        inputs=dict(references=[]),
        raw_references=[],
        validation=dict(raw_shard_hashes=[]),
    )
    value = dict(primitive_reference={})
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    monkeypatch.setattr(e, "load", lambda ref: fake_work if ref == {} else {})
    assert not r.replay(output)
    fake_work["seal_reference"] = dict(path="seal")
    seal = {k: v for k, v in fake_work.items() if k != "seal_reference"}
    monkeypatch.setattr(e, "load", lambda ref: fake_work if ref == {} else seal)
    fake_work["sources"] = [dict(path="/absent", sha256="bad")]
    seal["sources"] = fake_work["sources"]
    assert not r.replay(output)
    fake_work["sources"] = []
    seal["sources"] = []
    monkeypatch.setattr(e, "inputs", lambda root: {})
    assert not r.replay(output)


def test_measurement_cap(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8701-CUSTODY: measurement overrun stops before validation."""
    monkeypatch.setattr(e, "measure", lambda *args: dict(rows=[], symbolic_control_rows=[]))
    monkeypatch.setattr(r.time, "monotonic_ns", iter([0, 3601 * 10**9]).__next__)
    with pytest.raises(ValueError, match="measurement_cap"):
        r.run(tmp_path / (e.NAME + ".json"), private=True)


def test_absent_corpus_retains_intended_units(tmp_path):
    """SCENARIO-REPORT-8701-CUSTODY: external absence retains all censored intended slots."""
    result = e.measure([], {}, tmp_path)
    assert len(result["rows"]) == 480
    assert all(r["status"] == "censored" and r["censored"] for r in result["rows"])
    assert not result["symbolic_capacity_passed"]
