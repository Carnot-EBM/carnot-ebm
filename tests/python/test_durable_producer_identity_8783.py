"""REQ-REPORT-8783, REQ-VERIFY-8783: task identity is a checked operand."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import durable_producer_identity_8783 as e
from carnot.reporting import durable_producer_runner_8783 as runner


def test_exact_descriptor_and_roster(tmp_path):
    """SCENARIO-REPORT-8783-IDENTITY: a same-number alias must fail publication."""
    descriptor = e.descriptor("private-invocation")
    assert descriptor["task_id"] == "exp8783-durable-producer-identity"
    assert descriptor["MODEL_SPECS"] == []
    assert len(e.roster()) == 3
    assert {r["arm"] for r in e.roster()} == set(e.ARMS)
    assert all(r["offset"] == 2032 and r["updates"] == 32 for r in e.roster())
    candidate = dict(descriptor, honest_verdict="complete_null_fixture", verdict_class="null")
    e.exact_identity(candidate, descriptor)
    for key, changed in [("task_id", "exp8783-ordinary-update-cost"), ("invocation_id", "old")]:
        with pytest.raises(ValueError, match="producer_identity"):
            e.exact_identity(dict(candidate, **{key: changed}), descriptor)
    with pytest.raises(ValueError, match="producer_descriptor"):
        e.exact_identity(candidate, dict(descriptor, task_id="exp8772-ordinary-update-cost"))
    assert not e.projection([10**11] * 4, 6 * 10**11)["admitted"]
    assert e.projection([10**6] * 4, 10**8)["durable_updates"] == 96


def test_rehashed_alias_rejected_by_consumers(tmp_path):
    """SCENARIO-REPORT-8783-IDENTITY: real readers cannot authenticate an alias."""
    descriptor = e.descriptor("reader-fixture")
    value = dict(
        descriptor,
        producer_descriptor=descriptor,
        durable_producer_ready_score=1,
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
    )
    rows = runner.consumer_controls(value, tmp_path)
    assert all(r["passed"] for r in rows)
    assert {r["case"] for r in rows} == {"exact", "same_number_alias", "historical_alias"}
    assert all(r["candidate_reference"]["sha256"] for r in rows)


def test_cli_rejects_bad_date_worker_and_private_output(tmp_path):
    """REQ-VERIFY-8783: malformed requests cannot touch a production primary."""
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--worker", str(tmp_path / "absent")]) == 1
    with pytest.raises(ValueError, match="private_output"):
        runner.run(e.ROOT / "results" / (e.NAME + ".json"), "20261103", private=True)


@pytest.fixture(scope="module")
def admitted():
    """REQ-REPORT-8783: upstream operands are read only and never edited by tests."""
    value = e.inputs()
    assert not value["failures"] and value["admitted"]
    value["producer_descriptor"] = e.descriptor("private-worker")
    return value


def test_archived_inputs_and_worker_descriptor(admitted, tmp_path):
    """REQ-REPORT-8783: archive authority and retained controls remain mandatory."""
    assert admitted["original_authority_rows"][0]["passed"]
    assert len(admitted["retained_controls"]["threshold_control_rows"]) == 15
    assert all(r["passed"] for r in admitted["retained_controls"]["restart_rows"])
    bad = dict(admitted["producer_descriptor"], task_id="exp8783-wrong")
    with pytest.raises(ValueError, match="producer_descriptor"):
        runner.worker(dict(producer_descriptor=bad), tmp_path / "never")
    assert not (tmp_path / "never").exists()


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """REQ-VERIFY-8783: all real CLI controls write only under private scratch."""
    root = tmp_path_factory.mktemp("identity8783")
    output = root / (e.NAME + ".json")
    receipt = runner.child(
        "private_cli",
        [
            sys.executable,
            "-u",
            e.CLI,
            "--date",
            "20261103",
            "--output",
            str(output),
            "--private-control",
        ],
        root / "logs",
        deadline=450,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    return output


def test_private_cli_replay_and_exact_restart(private, tmp_path):
    """SCENARIO-VERIFY-8783-REPLAY: real children cover all three restart arms."""
    value = json.loads(private.read_bytes())
    assert value["task_id"] == e.TASK and value["completed_count"] == 3
    assert len(value["parity_rows"]) == 96 and all(r["passed"] for r in value["parity_rows"])
    assert all(r["passed"] for r in value["restart_rows"])
    assert value["durable_producer_ready_score"] == 0 and value["private_control"]
    runner.replay(private)
    assert runner.main(["--cold-replay", str(private)]) == 0
    changed = tmp_path / "rehashed.json"
    for key, mutation in [("task_id", "exp8783-ordinary-update-cost"), ("invocation_id", "old")]:
        e.atomic_json(changed, dict(value, **{key: mutation}))
        with pytest.raises(ValueError, match="producer_identity"):
            runner.replay(changed)
    bad = deepcopy(value)
    bad["producer_descriptor"]["task_id"] = bad["task_id"] = "exp8772-ordinary-update-cost"
    e.atomic_json(changed, bad)
    with pytest.raises(ValueError, match="producer_descriptor"):
        runner.replay(changed)


def test_readiness_and_failures(private):
    """REQ-REPORT-8783: only coverage, consumer and parity acceptance release readiness."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    work["private_control"] = False
    receipts = [dict(name="coverage_report", passed=True)]
    assert e.build(work, receipts, private)["durable_producer_ready_score"] == 1
    assert (
        e.build(work, [dict(name="failed_owned", passed=False)], private)["verdict_class"]
        == "disqualified"
    )
    assert e.build(work, [], private)["durable_producer_ready_score"] == 0
    work["window_rows"][0]["update_rows"][0]["state_hash"] = "changed"
    assert e.build(work, receipts, private)["durable_producer_ready_score"] == 0
    work["failures"] = [dict(passed=False)]
    assert e.build(work, receipts, private)["verdict_class"] == "blocked"


def test_input_rejection(monkeypatch):
    """REQ-VERIFY-8783: absent or altered authority never silently admits old receipts."""
    original = e.measurement.inputs
    monkeypatch.setattr(
        e.measurement, "inputs", lambda: dict(failures=[dict(passed=False)], refs=[])
    )
    assert e.inputs()["failures"]
    monkeypatch.setattr(e.measurement, "inputs", original)
    monkeypatch.setattr(e, "parse_design", lambda *a, **k: ([], []))
    assert e.inputs()["failures"][-1]["operand_state"] == "rejected_producer"


def test_public_orchestration_keeps_repository_health_separate(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8783: unrelated repository failure cannot become a scoped pass claim."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    work["private_control"] = False
    monkeypatch.setattr(runner, "measure", lambda *a, **kw: deepcopy(work))
    monkeypatch.setattr(runner, "seal", lambda work: None)
    monkeypatch.setattr(
        runner.runtime, "validate", lambda raw: [dict(name="coverage_report", passed=True)]
    )
    monkeypatch.setattr(
        runner, "child", lambda *a, **kw: dict(passed=False, scope="repository_health")
    )
    monkeypatch.setattr(runner.runtime, "terminal", lambda *a: dict(passed=True))
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(output, "20261103") == 0
    published = json.loads(output.read_bytes())
    assert published["durable_producer_ready_score"] == 1
    assert not published["repository_health_reference"]["full_suite_pass_claimed"]
    assert not published["repository_health_reference"]["current_receipt"]["passed"]


def test_replay_rejects_descriptor_drift(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8783-REPLAY: candidate rehashing cannot alter sealed identity."""
    value = json.loads(private.read_bytes())
    original = e.read

    def changed(ref):
        result = deepcopy(original(ref))
        if ref == value["primitive_reference"]:
            result["producer_descriptor"]["invocation_id"] = "other"
        return result

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError, match="producer_descriptor_drift"):
        runner.replay(private)


def test_mutated_task_contract_rejects(monkeypatch):
    """REQ-REPORT-8783: a full-object contract edit cannot bless a producer alias."""
    original = e.parse_design

    def changed(*args, **kwargs):
        table, tasks = original(*args, **kwargs)
        tasks[0]["title"] += " changed"
        return table, tasks

    monkeypatch.setattr(e, "parse_design", changed)
    assert e.inputs()["failures"][-1]["observed"] == "exact_current_task_contract"


def test_retained_control_failure(admitted, monkeypatch):
    """REQ-REPORT-8783: old threshold and restart assertions cannot be dropped."""
    changed = deepcopy(admitted)
    changed["qualified"]["restart_rows"][0]["passed"] = False
    monkeypatch.setattr(e.measurement, "inputs", lambda: changed)
    assert e.inputs()["failures"][-1]["observed"] == "retained_lifecycle_controls"


@pytest.mark.parametrize(
    "suffix,message,replacement",
    [
        ("measurement_seal.json", "independent_measurement_seal", {"measurement": {}}),
        ("validation_seal.json", "independent_validation_seal", {"validation": {}}),
        ("measurement.json", "clock_or_method_drift", {"duration_s": -1}),
        ("window_rows.json", "raw_window_drift", {"rows": []}),
    ],
)
def test_original_replay_assertions(private, monkeypatch, suffix, message, replacement):
    """SCENARIO-VERIFY-8783-REPLAY: retain each original independent seal assertion."""
    original = e.measurement.read

    def changed(ref):
        value = original(ref)
        return dict(value, **replacement) if ref["path"].endswith("/" + suffix) else value

    monkeypatch.setattr(e.measurement, "read", changed)
    with pytest.raises(ValueError, match=message):
        runner.replay(private)


def test_failed_consumer_disqualifies(private):
    """REQ-VERIFY-8783: failed owned consumer execution is not a scientific null."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    work["private_control"] = False
    work["consumer_admission_rows"][0]["passed"] = False
    result = e.build(work, [dict(name="coverage_report", passed=True)], private)
    assert result["verdict_class"] == "disqualified"
    assert result["durable_producer_ready_score"] == 0
    assert not result["required_checks_passed"]
