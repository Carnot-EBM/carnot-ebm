"""REQ-REPORT-8761 and REQ-VERIFY-8761: independent costs and sealed readers."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import hardware_cost_boundaries_8761 as a
from carnot.reporting import hardware_cost_inputs_8761 as source
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def branches():
    """Private branches keep absent siblings visible without acquiring devices."""
    return [
        dict(task_id=str(n), field="ready", qualified=False, checks=[], refs=[], value={})
        for n in range(3)
    ]


def test_current_independent_sources():
    """SCENARIO-REPORT-8761-CLOCKS: installed costs survive absent learning/durability."""
    evidence = source.authenticate(a.e.ROOT)
    assert evidence["authority"]["ready"]
    assert len(evidence["authority"]["rows"]) == 14
    assert evidence["board_qualified"], evidence["failures"]
    assert [b["qualified"] for b in evidence["branches"]] == [False, True, False]
    assert len(evidence["branches"][1]["value"]["complete_cost_rows"]) == 520
    assert evidence["historical_disqualified_producers"][0]["verdict_class"] == "disqualified"
    assert evidence["branches"][2]["scheduling_disposition"]["science_evidence"] is False
    result = a.cost_rows(evidence["branches"], [])
    assert len(result["boundary_cost_rows"]) == 520
    assert len(result["conditional_acceleration_bounds"]) == 120
    assert all(
        r["physical_transfer_duration_s"] is None and not r["includes_measured_host_and_transfer"]
        for r in result["conditional_acceleration_bounds"]
    )
    assert all(r["conditional_ceiling"] is None for r in result["conditional_acceleration_bounds"])


@pytest.mark.parametrize(
    "change,state",
    [
        ({}, "qualified"),
        ({"ready": None}, "observed"),
        ({"ready": 0}, "observed_zero"),
        ({"drop": True}, "missing_field"),
        ({"required_checks_passed": False}, "rejected_producer"),
    ],
)
def test_admission_states(tmp_path, monkeypatch, change, state):
    """REQ-VERIFY-8761: missing fields, zeros and rejected producers differ."""
    task = dict(id="exp8759-private", deliverable="private.json")
    value = dict(
        experiment_id=8759,
        task_id=task["id"],
        ready=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        acceptance_gates={"owned": True},
        verdict_class="null",
        adversarial_findings=[],
    )
    value.update(change)
    if value.pop("drop", False):
        value.pop("ready")
    atomic_json(tmp_path / task["deliverable"], value)
    monkeypatch.setattr(source.i, "terminal", lambda *args: [])
    monkeypatch.setattr(source.i, "producer_operands", lambda *args: [])
    row = source.admit(tmp_path, task, "ready")
    assert row["qualified"] is (state == "qualified")
    if not row["qualified"]:
        assert row["checks"][-1]["operand_state"] == state


def test_cost_parent_and_support():
    """SCENARIO-REPORT-8761-CLOCKS: parent clocks and hardware compatibility both matter."""
    b = branches()
    spans = [
        dict(
            operation="quadratic_Ising_energy_k_max_5",
            duration_s=0.9,
            parent_duration_s=1,
            process_id="p",
            parent="complete",
            clock_domain="same",
        ),
        dict(
            operation="host_features",
            duration_s=0.1,
            parent_duration_s=1,
            process_id="p",
            parent="complete",
            clock_domain="same",
        ),
    ]
    b[0].update(qualified=True, value=dict(complete_cost_rows=spans))
    support = [dict(board="KV260", operation=r["operation"], support="supported") for r in spans]
    result = a.cost_rows(b, support)
    assert result["conditional_acceleration_bounds"][0]["conditional_ceiling"] == pytest.approx(10)
    assert result["conditional_acceleration_bounds"][0][
        "minimum_additional_compatible_fraction_100x"
    ] == pytest.approx(0.09)
    assert {r["support"] for r in result["operation_support_rows"]} <= {
        "supported",
        "unsupported",
        "unknown",
    }
    spans[1]["clock_domain"] = "different"
    assert result["boundary_cost_rows"][0]["clock_domain"] == "same"
    assert all(
        r["conditional_ceiling"] is None
        for r in a.cost_rows(b, support)["conditional_acceleration_bounds"]
    )
    spans[1]["clock_domain"] = "same"
    for change, message in [
        ({"duration_s": 0.2}, "non_disjoint"),
        ({"nested_parent_span": True}, "nested_parent"),
        ({"duration_s": float("nan")}, "non_finite"),
    ]:
        bad = deepcopy(b)
        bad[0]["value"]["complete_cost_rows"][1].update(change)
        with pytest.raises(ValueError, match=message):
            a.cost_rows(bad, support)
    spans[0].update(start_ns=2, end_ns=4)
    spans[1].update(start_ns=1, end_ns=3)
    with pytest.raises(ValueError, match="overlapping"):
        a.cost_rows(b, support)


def test_mapping_not_benefit(tmp_path):
    """REQ-REPORT-8761: administrative completion needs no physical operation."""
    work = a.r.measure(tmp_path / "raw/invocations/private", "20261103", tmp_path, True)
    work["private_control"] = False
    work["evidence"]["board_qualified"] = True
    output = tmp_path / (a.e.NAME + ".json")
    value = a.build(work, [dict(passed=True)], output)
    assert value["hardware_mapping_ready_score"] == 1
    assert all(value["acceptance_gates"].values())
    assert not any(value["scientific_promotion_gates"].values())
    assert (
        value["physical_hardware_ready_score"] == value["complete_100x_learning_ready_score"] == 0
    )
    assert value["device_commands_executed"] == []
    assert value["MODEL_SPECS"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["repository_health_reference"]["proves_complete_full_suite_pass"] is False
    assert value["verdict_class"] == "blocked"
    assert a.build(work, [dict(passed=False)], output)["verdict_class"] == "disqualified"
    for branch in work["evidence"]["branches"]:
        branch["qualified"] = True
    assert a.build(work, [dict(passed=True)], output)["verdict_class"] == "null"


def test_actual_private_cli(tmp_path):
    """SCENARIO-VERIFY-8761-REPLAY: real child publication and cold replay resist edits."""
    argv = [sys.executable, "-u", str(a.e.ROOT / a.e.CLI)]
    output, logs = tmp_path / (a.e.NAME + ".json"), tmp_path / "logs"
    receipt = child(
        "private",
        argv + ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)],
        logs,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert child("replay", argv + ["--cold-replay", str(output)], logs)["passed"]
    value = json.loads(output.read_bytes())
    primitive = Path(value["primitive_reference"]["path"])
    work = json.loads(primitive.read_bytes())
    work["run_date"] = "forged"
    atomic_json(primitive, work)
    atomic_json(output, a.build(work, value["validation_receipts"], output))
    assert child("tamper", argv + ["--cold-replay", str(output)], logs, expected=1)["passed"]
    assert child("date", argv + ["--date", "20261399"], logs, expected=1)["passed"]
    assert child("guard", argv + ["--private-control"], logs, expected=1)["passed"]


def test_producer_replay_controls(tmp_path, monkeypatch):
    """REQ-VERIFY-8761: final identity and registered original replay both qualify costs."""
    from carnot.reporting import feature_reuse_cost_runner_8757 as producer

    path, final = tmp_path / "primary.json", tmp_path / "terminal.json"
    atomic_json(path, {})
    atomic_json(final, dict(publication=dict(primary_sha256=source.i.reference(path)["sha256"])))
    value = dict(
        experiment_id=8757,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        primitive_reference=source.i.reference(path),
        terminal_validation_sidecar_path=str(final),
    )
    monkeypatch.setattr(producer, "replay", lambda *args: False)
    with pytest.raises(ValueError, match="original_installed_cost_construction"):
        source.producer_operands(value, path)
    with pytest.raises(ValueError, match="unregistered_producer"):
        source.producer_operands(dict(value, experiment_id=8752), path)
    atomic_json(final, dict(publication=dict(primary_sha256="wrong")))
    with pytest.raises(ValueError, match="producer_final_identity"):
        source.producer_operands(value, path)


def test_missing_board_and_flagged_source(tmp_path, monkeypatch):
    """REQ-VERIFY-8761: pinned bytes alone cannot qualify a flagged producer."""
    path = tmp_path / "flagged.json"
    atomic_json(path, dict(required_checks_passed=False, flagged_adversarial=True))
    monkeypatch.setattr(source.i, "PINS", {path.name: source.i.reference(path)["sha256"][7:]})
    monkeypatch.setattr(source.i, "terminal", lambda *args: [])
    evidence = source.authenticate(tmp_path)
    assert not evidence["board_qualified"]
    assert evidence["failures"][-1]["reason"] == "flagged_original_board_evidence"


def test_scoped_wrapper_and_collection_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8761: exact node union is required and health remains separate."""
    runner = a.r
    raw = tmp_path / "raw/invocations/test"
    raw.mkdir(parents=True)
    (raw / "validation").mkdir()
    nodes = ["tests/private.py::test_req"]
    atomic_json(raw / "frozen_nodes.json", dict(nodes=nodes))
    atomic_json(raw / "measurement.json", dict(frozen_test_nodes=nodes))
    atomic_json(raw / "validation/collected_nodes.json", nodes)
    monkeypatch.setattr(runner, "BASE_VALIDATE", lambda raw: [])
    assert runner.validate(raw)[-1]["passed"]
    atomic_json(raw / "validation/collected_nodes.json", [])
    assert not runner.validate(raw)[-1]["passed"]
    monkeypatch.setattr(
        runner, "child", lambda *args, **kwargs: dict(passed=False, stdout_path=str(raw / "empty"))
    )
    (raw / "empty").write_text("")
    monkeypatch.setattr(
        runner,
        "BASE_MEASURE",
        lambda *args: dict(source_refs=[], resource_failures=[], code_refs=[]),
    )
    work = runner.measure(raw, "20261103", tmp_path, True)
    assert work["resource_failures"][-1]["operand_state"] == "failed_owned_validation"
    assert runner.main(["--date", "20261399"]) == 1
