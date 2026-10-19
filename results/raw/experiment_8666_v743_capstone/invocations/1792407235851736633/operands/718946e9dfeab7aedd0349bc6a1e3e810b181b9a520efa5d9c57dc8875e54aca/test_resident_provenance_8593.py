"""REQ-REPORT-8593 / REQ-VERIFY-8593: real head identity precedes readiness."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import resident_provenance_8593 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_original_descriptor_and_mutations(tmp_path):
    """SCENARIO-REPORT-8593-DESCRIPTOR: repaired hashes cannot replace original authority."""
    descriptor = e.descriptor(e.ROOT)
    assert descriptor["parameter_count"] == 109
    assert len(descriptor["feature_order"]) == 9
    assert descriptor["dtype"] == "IEEE754 binary64"
    assert e.check_descriptor(descriptor)
    assert not e.check_descriptor(None)
    for key, value in [("parameter_count", 108), ("scaler_sha256", "sha256:wrong")]:
        changed = dict(descriptor, **{key: value})
        changed["descriptor_sha256"] = e.canonical_hash(
            {k: v for k, v in changed.items() if k != "descriptor_sha256"}
        )
        assert not e.check_descriptor(changed)
    with pytest.raises((OSError, ValueError)):
        e.descriptor(tmp_path)


def test_classifier_and_private_publication(tmp_path):
    """SCENARIO-VERIFY-8593-REPLAY: real warning exit and publication refusal stay visible."""
    result = e.reproduce(tmp_path, e.descriptor(e.ROOT))
    assert result["original"]["cli"]["exit_code"] == 1
    assert result["original"]["publication_refused"]
    assert result["corrected"]["cli"]["exit_code"] == 0
    assert result["corrected"]["flags"] == []
    assert result["original"]["classifier_inputs"]["target_model"] is None
    assert result["original"]["flags"][0]["kind"] == "METHODOLOGY_MISSING"


def test_replay_missing_and_owned_failure(tmp_path):
    """REQ-VERIFY-8593: absent primitives and invented checks never qualify."""
    assert not e.replay(tmp_path / "absent")
    work = e.measure(tmp_path / "raw", "20261018", root=tmp_path)
    value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    failed = e.artifact(work, [dict(name="owned", passed=False)], tmp_path / "failed.json")
    assert failed["verdict_class"] == "disqualified"
    assert value["run_date"] == "20261018"


def test_real_cli_rejections_and_date(tmp_path):
    """REQ-VERIFY-8593: direct CLI paths run outside the interpreter under test."""
    for name, args, expected in [
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("consume", ["--consume", str(tmp_path / "absent")], 1),
        ("bad_date", ["--date", "20260230"], 2),
    ]:
        receipt = child(
            name, [sys.executable, str(e.ROOT / e.CLI), *args], tmp_path / "logs", expected=expected
        )
        assert receipt["passed"]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """REQ-REPORT-8593: one genuine immutable reproduction supplies private test operands."""
    raw = tmp_path_factory.mktemp("provenance")
    work = e.measure(raw, "20261017")
    assert not work["gate_check_summary"]
    atomic_json(raw / "head_descriptor.json", work["target_model"])
    return work


def test_immutable_replay_and_real_controls(measured, tmp_path):
    """SCENARIO-VERIFY-8593-REPLAY: fresh children reject repaired summaries and descriptors."""
    from carnot.reporting import resident_runner_8593 as runner

    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(measured, [], output)
    atomic_json(output, value)
    assert e.replay(output)
    assert e.consume(output, require_ready=False)["passed"]
    with pytest.raises(ValueError, match="not_ready"):
        e.consume(output)
    controls = runner.controls(deepcopy(measured), [], tmp_path / "control", output)
    assert len(controls) == 7 and all(r["passed"] for r in controls)
    for field, observed in [("target_model", None), ("completed_count", 12)]:
        changed = dict(value, **{field: observed})
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    value["reproducibility_checksum"] = "wrong"
    atomic_json(output, value)
    assert not e.replay(output)
    assert not e.receipts_valid(measured, [])


def test_rehashed_primitive_and_original_byte_drift(measured, tmp_path):
    """REQ-VERIFY-8593: repairing work hashes cannot change historical numerical operands."""
    work = deepcopy(measured)
    raw = tmp_path / "raw"
    work["raw"] = str(raw)
    atomic_json(raw / "head_descriptor.json", work["target_model"])
    original_path = raw / "wrong_original.json"
    changed = e.original()
    changed["random_seed"] += 1
    atomic_json(original_path, changed)
    work["reproduction"]["original"]["candidate"] = e.reference(original_path)
    atomic_json(raw / "work.json", work)
    value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert not e.replay(path)
    work["target_model"]["parameter_count"] = 108
    atomic_json(raw / "work.json", work)
    value = e.artifact(work, [], path)
    atomic_json(path, value)
    assert not e.replay(path)


def test_owned_checks_and_qualification_boundary(measured, tmp_path, monkeypatch):
    """REQ-REPORT-8593: qualification and blocked states stay separate from scientific benefit."""
    path = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(e, "REQUIRED", {"unit_fixture"})
    monkeypatch.setattr(e, "receipts_valid", lambda *args: True)
    value = e.artifact(measured, [dict(name="unit_fixture", passed=True)], path)
    assert value["resident_measurement_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert value["generalized_learning_benefit_score"] == 0
    atomic_json(path, value)
    assert e.consume(path)["passed"]
    from carnot.reporting import resident_runner_8593 as runner

    assert runner.main(["--consume", str(path)]) == 0
    assert runner.main(["--check-descriptor", str(path)]) == 0
    atomic_json(path, dict(target_model={}))
    assert runner.main(["--check-descriptor", str(path)]) == 1
    assert runner.main(["--check-descriptor", str(tmp_path / "missing")]) == 1
    assert runner.date_value("20261018") == "20261018"
    for bad in ["20261301", "2026111"]:
        with pytest.raises(Exception, match="calendar"):
            runner.date_value(bad)


def test_runner_flow_private_success_and_blocked(measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8593: publication paths preserve real checks even when unit validation is isolated."""
    from carnot.reporting import resident_runner_8593 as runner

    def measured_copy(raw, date):
        work = deepcopy(measured)
        work.update(raw=str(raw), run_date=date)
        atomic_json(raw / "work.json", work)
        return work

    monkeypatch.setattr(runner.e, "measure", measured_copy)
    monkeypatch.setattr(runner.validation, "validate", lambda *args: [])
    monkeypatch.setattr(runner.validation, "finish", lambda *args: [])
    monkeypatch.setattr(runner, "controls", lambda *args: [])
    output = tmp_path / "private" / (e.NAME + ".json")
    assert runner.run(output, "20261018") == 0
    value = json.loads(output.read_bytes())
    assert value["run_date"] == "20261018"
    assert value["resident_measurement_ready_score"] == 0
    private = value["private_publication_receipts"][0]
    assert e.receipts_valid(dict(private_publication_receipts=[private]), [])
    assert not e.receipts_valid({}, [dict(name="invented", passed=True)])
    monkeypatch.setattr(runner, "run", lambda *args: 0)
    assert runner.main(["--date", "20261018", "--private-output", str(output)]) == 0


@pytest.mark.parametrize("mode", ["pass", "missing", "failed"])
def test_bounded_validation_collection_and_finish(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8593: frozen node unions reject missing or failed JUnit outcomes."""
    from carnot.reporting import resident_validation_8593 as validation

    suites = ["tests/python/test_fixture0.py", "tests/python/test_fixture1.py"]
    monkeypatch.setattr(validation, "SUITES", suites)
    raw = tmp_path / mode
    work = dict(original_node_union=dict(collected=[suites[0] + "::test_unit"]))

    def fake_child(name, argv, logs, **options):
        logs.mkdir(parents=True, exist_ok=True)
        stdout = logs / (name + ".stdout")
        stdout.write_text("")
        if name.startswith("collect_"):
            index = int(name.split("_")[-1])
            stdout.write_text(suites[index] + "::test_unit\n")
        if name.startswith("shard_"):
            index = int(name.split("_")[-1])
            junit = Path(next(s.split("=", 1)[1] for s in argv if s.startswith("--junitxml=")))
            if mode != "missing":
                failure = '<failure message="owned"/>' if mode == "failed" else ""
                junit.write_text(
                    f'<testsuite><testcase classname="tests.python.test_fixture{index}" name="test_unit">{failure}</testcase></testsuite>'
                )
        if name == "coverage_json":
            destination = Path(argv[-1])
            atomic_json(destination, dict(totals=dict(percent_covered=100)))
        return dict(name=name, passed=True, stdout_path=str(stdout))

    monkeypatch.setattr(validation, "child", fake_child)
    receipts = validation.validate(work, raw)
    assert receipts[-1]["passed"] == (mode == "pass")
    assert work["exact_union_receipt"]["historical_union_preserved"]
    assert validation.finish(work, raw)[-1]["passed"]


def test_measure_resource_error_and_blocked_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8593-REPLAY: external resource failure has a reproducible blocked record."""
    from carnot.reporting import resident_runner_8593 as runner

    monkeypatch.setattr(
        e.methods, "authority", lambda root: (_ for _ in ()).throw(ValueError("external_absence"))
    )
    work = e.measure(tmp_path / "work", "20261018")
    assert work["gate_check_summary"][0]["observed"] == "external_absence"
    path = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [], path)
    atomic_json(path, value)
    assert e.replay(path)
    monkeypatch.setattr(runner.e, "measure", lambda raw, date: dict(work, raw=str(raw)))
    monkeypatch.setattr(runner.validation, "validate", lambda *args: [])
    monkeypatch.setattr(runner.validation, "finish", lambda *args: [])
    monkeypatch.setattr(runner, "controls", lambda *args: [])
    assert runner.run(tmp_path / "blocked" / path.name, "20261018") == 0
