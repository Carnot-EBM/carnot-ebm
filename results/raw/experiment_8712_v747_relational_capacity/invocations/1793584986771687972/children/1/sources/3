"""REQ-REPORT-8712: model-free capacity cannot supply missing model probabilities."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import relational_capacity_8712 as e
from carnot.reporting import relational_capacity_runner_8712 as r
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child


def worlds():
    """Read original pinned worlds so test labels do not come from measured output."""
    protocol = e.load(dict(path=str(e.ROOT / e.upstream.PROTOCOL), sha256=e.upstream.PROTOCOL_PIN))
    return e.load(protocol["relational_evidence_protocol"]["worlds_reference"])["worlds"]


def test_capacity(tmp_path):
    """SCENARIO-REPORT-8712-CAPACITY: preserve all views, chunk repeats and SQL controls."""
    data = e.measure(worlds(), tmp_path)
    assert len(data["rows"]) == 480
    assert len(data["lexical_rows"]) == len(data["symbolic_rows"]) == 1440
    assert data["relational_passed"] and data["symbolic_passed"]
    assert len(data["collision_witnesses"]) == 120
    assert all(w["exact_equal"] and w["opposite_truth"] for w in data["collision_witnesses"])
    assert len(data["wording_changes"]) == 120
    assert all(w["same_truth"] and w["query_equivalent"] for w in data["wording_changes"])
    assert any(not w["lexical_equal"] for w in data["wording_changes"])
    assert all(len(x["values"]) == 8 for x in data["lexical_rows"])
    assert all(
        x["decision"] == -1 for x in data["symbolic_rows"] if x["arm"] == "deleted_required_records"
    )
    assert all(x["passed"] for x in data["symbolic_rows"])
    assert all("p2_hallucination_probability" not in x for x in data["rows"])
    assert e.MODEL_SPECS == []


def test_failed_and_absent_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8712-CAPACITY: wrong controls fail; absent inputs retain every slot."""
    changed = deepcopy(worlds()[:1])
    changed[0]["views"][0]["truth"] = 0
    assert not e.measure(changed, tmp_path / "wrong")["relational_passed"]
    original_sql = e.sql_decision

    def wrong_sql(*args):
        original_sql(*args)
        return 1

    monkeypatch.setattr(e, "sql_decision", wrong_sql)
    assert not e.measure(worlds()[:1], tmp_path / "broken")["symbolic_passed"]
    data = e.measure([], tmp_path / "absent")
    assert len(data["rows"]) == 480
    assert len(data["lexical_rows"]) == len(data["symbolic_rows"]) == 1440
    assert all(x["status"] == "censored" for x in data["rows"])
    assert not data["relational_passed"] and not data["symbolic_passed"]


def test_input_custody(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8712-CUSTODY: original corpus custody is independent of learning."""
    assert e.inputs(e.ROOT)["passed"]
    assert e.authority.authority(e.ROOT)["ready"]
    monkeypatch.setattr(e, "CORPUS_PIN", "changed")
    assert not e.inputs(e.ROOT)["passed"]
    assert not e.inputs(tmp_path)["passed"]


def test_private_cli_replay_and_guards(tmp_path):
    """SCENARIO-REPORT-8712-CUSTODY: private success and rehashed claims get real exits."""
    output = tmp_path / (e.NAME + ".json")
    args = [sys.executable, "-u", e.CLI, "--date", "20261102"]
    assert child("private", args + ["--private-output", str(output)], tmp_path / "logs")["passed"]
    original = output.read_bytes()
    value = json.loads(original)
    assert r.replay(output)
    assert reader_receipt(e.TASK, output.parent, field="relational_rows_ready_score", expected=0)[
        "passed"
    ]
    assert value["numerical_head_comparison_status"] == "blocked_missing_ninth_input"
    assert value["missing_probability_count"] == 480
    value["rows"][0]["truth"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert child("reject", args + ["--cold-replay", str(output)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    output.write_bytes(original)
    value = json.loads(original)
    primitive = Path(value["primitive_reference"]["path"])
    work = e.load(value["primitive_reference"])
    work["measurement"]["rows"][0]["truth"] = 0
    seal = Path(work["seal_reference"]["path"])
    e.atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
    work["seal_reference"] = e.reference(seal)
    e.atomic_json(primitive, work)
    value["primitive_reference"] = e.reference(primitive)
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert not r.replay(output)
    primitive.unlink()
    assert not r.replay(output)
    assert r.main(["--date", "bad"]) == 2
    assert r.main(["--private-output", str(e.ROOT / "results/forbidden.json")]) == 2
    assert r.main(["--cold-replay", str(output)]) == 1


def test_build_classes_and_gates(tmp_path):
    """SCENARIO-REPORT-8712-CUSTODY: each gate fails independently and owned faults disqualify."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, private=True) == 0
    work = e.load(json.loads(output.read_bytes())["primitive_reference"])
    assert e.build(work, output)["verdict_class"] == "null"
    work["private_control"] = False
    assert e.build(work, output)["verdict_class"] == "disqualified"
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    value = e.build(work, output)
    assert value["verdict_class"] == "circular_positive"
    assert value["relational_rows_ready_score"] == value["symbolic_control_ready_score"] == 1
    work["runtime"]["ready"] = 0
    assert e.build(work, output)["relational_rows_ready_score"] == 1
    work["measurement"]["symbolic_passed"] = False
    assert e.build(work, output)["verdict_class"] == "disqualified"
    work["measurement"]["symbolic_passed"] = True
    work["inputs"]["passed"] = False
    work["inputs"]["failures"] = [e.authority.gate(tmp_path / "absent", "corpus", 1, None)]
    assert e.build(work, output)["verdict_class"] == "blocked"
    work["resources"]["ready"] = False
    assert e.build(work, output)["relational_rows_ready_score"] == 0


def test_registered_rows_cannot_be_replaced(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8712-CUSTODY: a rehashed corpus still has to equal original worlds."""
    branch = e.upstream.inputs(e.ROOT)
    monkeypatch.setattr(e.upstream, "inputs", lambda root: deepcopy(branch))
    corpus = e.load(dict(path=str(e.ROOT / e.CORPUS), sha256=e.CORPUS_PIN))
    corpus["rows"][0]["actual_truth"] = 0
    path = tmp_path / e.CORPUS
    path.parent.mkdir(parents=True)
    e.atomic_json(path, corpus)
    monkeypatch.setattr(e, "CORPUS_PIN", e.reference(path)["sha256"])
    result = e.inputs(tmp_path)
    assert not result["passed"]
    assert any(g["check"] == "original_480_rows" for g in result["failures"])


def test_validation_retention_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8712-CUSTODY: exercise real children without recursive producer runs."""
    monkeypatch.setattr(e, "TESTS", ["tests/python/test_primary_publication_7928.py"])
    raw, scratch = tmp_path / "checks", tmp_path / "scratch"
    scratch.mkdir()
    result = r.validate(raw, scratch, [e.reference(e.ROOT / e.OWNED[0])])
    assert result["custody"]["passed"], result["custody"]
    assert all(x["passed"] for x in result["receipts"])
    assert not result["coverage"]["complete"]
    output = tmp_path / (e.NAME + ".json")
    r.run(output, private=True)
    value = json.loads(output.read_bytes())
    work = e.load(value["primitive_reference"])
    work.update(private_control=False, validation=result)

    def save():
        seal = tmp_path / "seal.json"
        e.atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
        work["seal_reference"] = e.reference(seal)
        e.atomic_json(Path(value["primitive_reference"]["path"]), work)
        e.atomic_json(output, e.build(work, output))

    save()
    assert r.replay(output)
    work["runtime"]["ready"] = 1 - work["runtime"]["ready"]
    save()
    assert not r.replay(output)
    work["runtime"]["ready"] = 1 - work["runtime"]["ready"]
    work["validation"]["coverage"]["complete"] = True
    save()
    assert not r.replay(output)
    work["validation"]["coverage"]["complete"] = False
    work["validation"]["custody"]["passed"] = False
    save()
    assert not r.replay(output)
    monkeypatch.setattr(r, "CONSUMERS", ["absent::node"])
    assert not r.validate(tmp_path / "empty", tmp_path, [])["custody"]["passed"]
    result["custody"]["passed"] = True
    monkeypatch.setattr(r, "validate", lambda *args: result)
    nonprivate = tmp_path / "nonprivate" / (e.NAME + ".json")
    assert r.run(nonprivate) == 0
    assert json.loads(nonprivate.read_bytes())["verdict_class"] == "disqualified"


def test_replay_early_boundaries(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8712-CUSTODY: malformed evidence and missing source operands fail."""
    output = tmp_path / "bad.json"
    output.write_text("{}")
    assert not r.replay(output)
    e.atomic_json(output, dict(reproducibility_checksum="bad"))
    assert not r.replay(output)
    work = dict(
        seal_reference={},
        sources=[],
        inputs=dict(references=[]),
        raw_references=[],
        validation=dict(raw_shard_hashes=[]),
    )
    value = dict(primitive_reference={})
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    monkeypatch.setattr(e, "load", lambda ref: work if ref == {} else {})
    assert not r.replay(output)
    work["seal_reference"] = dict(path="seal")
    seal = {k: v for k, v in work.items() if k != "seal_reference"}
    monkeypatch.setattr(e, "load", lambda ref: work if ref == {} else seal)
    work["sources"] = [dict(path="/absent", sha256="bad")]
    seal["sources"] = work["sources"]
    assert not r.replay(output)
    work["sources"] = []
    seal["sources"] = []
    monkeypatch.setattr(e, "inputs", lambda root: {})
    assert not r.replay(output)


def test_measurement_cap(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8712-CUSTODY: a measurement overrun stops before validation."""
    monkeypatch.setattr(
        r.shared,
        "preconditions",
        lambda scratch: dict(
            disk_backed=False, writable=False, free_memory_kib=0, free_disk_bytes=0
        ),
    )
    monkeypatch.setattr(
        e, "measure", lambda *args: dict(rows=[], lexical_rows=[], symbolic_rows=[])
    )
    monkeypatch.setattr(r.time, "monotonic_ns", iter([0, 3601 * 10**9]).__next__)
    with pytest.raises(ValueError, match="measurement_cap"):
        r.run(tmp_path / (e.NAME + ".json"), private=True)
