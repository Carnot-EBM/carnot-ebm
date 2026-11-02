"""REQ-REPORT-8721 and REQ-VERIFY-8721: independent costs never imply FPGA execution."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import hardware_operation_boundary_8721 as e
from carnot.reporting import hardware_operation_inputs_8721 as i
from carnot.reporting import hardware_operation_runner_8721 as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child


def test_current_independent_operands():
    """SCENARIO-REPORT-8721-MAPPING: missing learning does not suppress batch or CPU parity."""
    evidence = i.authenticate(i.ROOT)
    assert evidence["authority"]["ready"]
    assert len(evidence["authority"]["rows"]) == 14
    assert evidence["board_qualified"], evidence["failures"]
    assert [b["qualified"] for b in evidence["branches"]] == [False, True, True]
    assert evidence["branches"][0]["checks"][0]["operand_state"] == "missing_file"
    assert len(evidence["branches"][1]["value"]["complete_cost_rows"]) == 780
    assert len(evidence["branches"][2]["value"]["operation_partition_rows"]) == 100


def test_gates_and_absence(tmp_path):
    """REQ-REPORT-8721: missing operands, zero and changed bytes are separate evidence."""
    path = tmp_path / "operand.json"
    absent = i.reference(path)
    i.load_bytes(absent)
    assert i.gate(absent, "score", 1, "producer")["operand_state"] == "missing_file"
    for value, state in [({}, "missing_field"), ({"score": 0}, "observed_zero")]:
        atomic_json(path, value)
        assert i.gate(i.reference(path), "score", 1, "p")["operand_state"] == state
    with pytest.raises(ValueError, match="changed_absence"):
        i.load_bytes(absent)
    ref = i.reference(path)
    atomic_json(path, {"score": 1})
    assert i.gate(ref, "score", 1, "p")["operand_state"] == "changed_bytes"


def test_versioned_document_bytes(tmp_path):
    """REQ-REPORT-8721: frozen original documents survive later additive spec updates."""
    old, snapshot = tmp_path / "spec.md", tmp_path / "source_closure" / "spec.md"
    old.write_text("original")
    snapshot.parent.mkdir()
    snapshot.write_text("original")
    ref = i.reference(old)
    old.write_text("later additive requirement")
    assert i.versioned_reference(ref, [i.reference(snapshot)]) == i.reference(snapshot)
    with pytest.raises(ValueError, match="changed_bytes"):
        i.versioned_reference(ref, [])
    source = tmp_path / "module.py"
    source.write_text("original")
    code = i.reference(source)
    source.write_text("modified")
    with pytest.raises(ValueError, match="changed_bytes"):
        i.versioned_reference(code, [i.reference(snapshot)])


def test_private_cli_and_sealed_replay(tmp_path):
    """SCENARIO-VERIFY-8721-REPLAY: execute real private CLI and reject rehashed work."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    logs = tmp_path / "logs"
    assert child(
        "private",
        argv + ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)],
        logs,
    )["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["hardware_mapping_ready_score"] == value["physical_hardware_ready_score"] == 0
    assert len(value["operation_support_rows"]) == len(e.OPERATIONS) * 3
    assert child("replay", argv + ["--cold-replay", str(output)], logs)["passed"]
    path = Path(value["primitive_reference"]["path"])
    original = path.read_bytes()
    work = json.loads(original)
    work["run_date"] = "forged"
    atomic_json(path, work)
    atomic_json(output, e.build(work, value["validation_receipts"], output))
    assert child("rehashed", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    path.write_bytes(original)
    atomic_json(output, value)
    path.unlink()
    assert child("missing", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    assert child("date", argv + ["--date", "bad"], logs, expected=1)["passed"]
    assert child("guard", argv + ["--private-control"], logs, expected=1)["passed"]
    assert r.main(["--date", "20260230"]) == 1


def test_build_boundaries(tmp_path):
    """REQ-REPORT-8721: mapping may qualify while all physical and learning claims block."""
    work = r.measure(tmp_path / "raw" / "invocations" / "unit", "20261102", tmp_path, True)
    work["private_control"] = False
    work["evidence"]["board_qualified"] = True
    work["evidence"]["prior"]["board_status_rows"] = [{"board": "KV260"}]
    value = e.build(work, [{"passed": True}], tmp_path / (e.NAME + ".json"))
    assert value["hardware_mapping_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["branch_gate_values"] == {f: 0 for _, f in i.BRANCHES}
    assert any(x["support"] == "supported" for x in value["operation_support_rows"])
    assert all(x["conditional_ceiling"] is None for x in value["conditional_acceleration_bounds"])
    assert (
        e.build(work, [{"passed": False}], tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )
    work["resource_failures"] = [{"passed": False}]
    assert (
        e.build(work, [{"passed": True}], tmp_path / (e.NAME + ".json"))[
            "hardware_mapping_ready_score"
        ]
        == 0
    )


def test_admission_rejects_flags_and_substitutions(tmp_path, monkeypatch):
    """REQ-REPORT-8721: a ready scalar cannot replace producer qualification or identity."""
    path = tmp_path / "results" / "experiment_8719_v747_compiled_sparse_update.json"
    task = dict(id="exp8719-compiled-sparse-update", deliverable=str(path.relative_to(tmp_path)))
    value = dict(
        experiment_id=8719,
        task_id=task["id"],
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        native_update_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        acceptance_gates={"owned": True},
        adversarial_findings=[],
    )
    monkeypatch.setattr(i, "CURRENT_PINS", {})
    monkeypatch.setattr(i, "producer_operands", lambda *a: [])
    publish_primary(path, value, lambda p: dict(passed=True))
    assert i.admit(tmp_path, task, "native_update_ready_score")["qualified"]
    for changes in [
        dict(flagged_adversarial=True),
        dict(required_checks_passed=False),
        dict(task_id="exp8719-forged"),
        dict(acceptance_gates={"owned": False}),
        dict(acceptance_gates=[]),
        dict(verdict_class="blocked"),
        dict(adversarial_findings=[dict(severity="warning")]),
    ]:
        path.unlink()
        publish_primary(path, {**value, **changes}, lambda p: dict(passed=True))
        assert not i.admit(tmp_path, task, "native_update_ready_score")["qualified"]
    publish_primary(
        path,
        {**value, "acceptance_gates": [dict(name="owned", passed=True)]},
        lambda p: dict(passed=True),
    )
    assert i.admit(tmp_path, task, "native_update_ready_score")["qualified"]
    monkeypatch.setattr(i, "CURRENT_PINS", {8719: "0" * 64})
    assert not i.admit(tmp_path, task, "native_update_ready_score")["qualified"]


def test_producer_construction_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8721-REPLAY: producer seals and terminal identity cannot be rehashed away."""
    from carnot.reporting import compiled_sparse_study_8719 as native
    from carnot.reporting import installed_batch_cost_runner_8717 as batch

    path = tmp_path / "results" / "experiment_8719_fixture.json"
    path.parent.mkdir()
    path.write_text("{}")
    raw = path.parent / "raw" / path.stem
    work_path, validation_path, final_path = (
        raw / "work.json",
        raw / "validation.json",
        raw / "final.json",
    )
    atomic_json(work_path, dict(private_control=False))
    atomic_json(validation_path, dict(receipts=[dict(passed=True)]))
    atomic_json(final_path, dict(publication=dict(primary_sha256=i.reference(path)["sha256"])))
    value = dict(
        experiment_id=8719,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        primitive_reference=i.reference(work_path),
        validation_reference=i.reference(validation_path),
        terminal_validation_sidecar_path=str(final_path),
        invocation_id="unit",
        validation_receipts=[dict(passed=True)],
    )
    for folder, field in [("seals", "measurement"), ("validation_seals", "validation")]:
        atomic_json(
            raw / folder / "unit.json",
            {
                field: value[
                    "primitive_reference" if field == "measurement" else "validation_reference"
                ]
            },
        )
    monkeypatch.setattr(native, "build", lambda *a: value)
    assert i.producer_operands(value, path)
    monkeypatch.setattr(native, "build", lambda *a: {})
    with pytest.raises(ValueError, match="original_native_construction"):
        i.producer_operands(value, path)
    atomic_json(raw / "seals" / "unit.json", dict(measurement={}))
    with pytest.raises(ValueError, match="original_independent_seal"):
        i.producer_operands(value, path)
    monkeypatch.setattr(batch, "replay", lambda *a: False)
    with pytest.raises(ValueError, match="original_batch_construction"):
        i.producer_operands({**value, "experiment_id": 8717}, path)
    with pytest.raises(ValueError, match="unregistered_learning_producer"):
        i.producer_operands({**value, "experiment_id": 8713}, path)
    atomic_json(
        final_path, dict(passed=False, publication=dict(primary_sha256=i.reference(path)["sha256"]))
    )
    with pytest.raises(ValueError, match="producer_final_identity"):
        i.producer_operands(value, path)
    atomic_json(final_path, dict(publication=dict(primary_sha256="forged")))
    with pytest.raises(ValueError, match="producer_final_identity"):
        i.producer_operands(value, path)


def test_resources_deadline_heartbeat_and_failed_child(tmp_path, monkeypatch):
    """REQ-VERIFY-8721: resources, real clocks and failed children cannot grant readiness."""
    pre = dict(
        disk_backed=False, tools_available=False, free_disk_bytes=0, memory_available_bytes=0
    )

    def resource_probe(scratch):
        (scratch / "write_probe").write_bytes(b"private disk scratch")
        return deepcopy(pre)

    monkeypatch.setattr(r, "resources", resource_probe)
    work = r.measure(tmp_path / "resource" / "invocations" / "unit", "20261102", tmp_path, True)
    assert work["resource_failures"]
    ticks = iter([0, 3601 * 10**9])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        r.measure(tmp_path / "deadline" / "invocations" / "unit", "20261102", tmp_path, True)
    monkeypatch.undo()
    waits = iter([False, True])

    class Stop:
        def wait(self, seconds):
            assert seconds <= 60
            return next(waits)

    r.heartbeat(Stop())
    monkeypatch.setattr(r, "BASE_VALIDATE", lambda *a: [dict(passed=False)])
    assert r.validate(tmp_path) == [dict(passed=False)]
    monkeypatch.setattr(r, "BASE_RUN", lambda *a: 1)
    assert r.main([]) == 1


def test_flagged_historical_authority(tmp_path, monkeypatch):
    """REQ-REPORT-8721: a historical producer flag also forbids mapping readiness."""
    path = tmp_path / "board.json"
    atomic_json(path, dict(required_checks_passed=False, flagged_adversarial=True))
    monkeypatch.setattr(i, "PINS", {"board.json": i.reference(path)["sha256"][7:]})
    monkeypatch.setattr(i, "terminal", lambda *a: [])
    assert not i.authenticate(tmp_path)["board_qualified"]
