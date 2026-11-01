"""REQ-REPORT-8692, REQ-VERIFY-8692: private CLI, producer and immutable replay controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from carnot.reporting import spline_precision_study_8692 as s
from carnot.reporting import spline_precision_runner_8692 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from test_spline_precision_8692 import head


def test_authenticated_inputs_and_gate_distinctions(tmp_path):
    """SCENARIO-REPORT-8692-TERMINAL: zero and missing are different observed operands."""
    inputs = s.authenticate()
    assert len(inputs["heads"]) == 5
    assert not inputs["failures"]
    ref = s.reference(tmp_path / "absent")
    assert s.check(ref, "x", 1)["operand_state"] == "missing_file"
    atomic_json(tmp_path / "absent", {})
    ref = s.reference(tmp_path / "absent")
    assert s.check(ref, "x", 1)["operand_state"] == "missing_field"
    atomic_json(tmp_path / "absent", dict(x=0))
    ref = s.reference(tmp_path / "absent")
    assert s.check(ref, "x", 1)["operand_state"] == "observed_zero"
    atomic_json(tmp_path / "absent", dict(x=1))
    assert s.check(ref, "x", 1)["operand_state"] == "changed_bytes"
    assert s.check(s.reference(tmp_path / "absent"), "x", 1)["passed"]


def test_reduction_primary_null_failed_and_blocked(tmp_path):
    """SCENARIO-REPORT-8692-TERMINAL: descriptive widths cannot replace primary."""
    data = s.evaluate([head()], s.numeric.vectors([head()]))
    assert len(data["precision_rows"]) == 3 * 256 * 2
    assert len(data["trajectory_rows"]) == 3 * 4 * 256
    assert data["primary_passed"]
    work = dict(
        data,
        failures=[],
        run_date="20261101",
        source_refs=[],
        code_refs=[],
        preconditions={},
        started_at_utc="x",
        ended_at_utc="y",
        duration_s=1.0,
        phase_spans=[],
        raw=str(tmp_path),
        historical_model_provenance={},
    )
    output = tmp_path / (s.NAME + ".json")
    before = canonical_hash(work)
    good = s.build(work, [dict(passed=True)], output)
    assert good["precision_mapping_ready_score"] == 1
    assert s.build(work, [dict(passed=False)], output)["verdict_class"] == "disqualified"
    assert s.build(dict(work, primary_passed=False), [], output)["verdict_class"] == "null"
    external = s.build(
        work,
        [
            dict(
                passed=False,
                coverage_role="consumers",
                name="old_e2e",
                stdout_path="/var/tmp/old_e2e.stdout",
                stdout_sha256="sha256:old",
                actual_exit=1,
            )
        ],
        output,
    )
    assert external["verdict_class"] == "blocked"
    assert external["required_checks_passed"]
    assert external["precision_mapping_ready_score"] == 0
    blocked = s.build(dict(work, failures=[dict(reason="external")]), [], output)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["precision_mapping_ready_score"] == 0
    assert canonical_hash(work) == canonical_hash(deepcopy(work))
    assert canonical_hash(work) == before


def test_private_destination_and_date_guards(tmp_path):
    """SCENARIO-VERIFY-8692-CUSTODY: aliases and direct calls cannot publish fixtures."""
    assert r.main(["--date", "bad"]) == 1
    assert (
        r.main(["--private-control", "--output", str(s.ROOT / "results" / (s.NAME + ".json"))]) == 1
    )
    alias = tmp_path / "alias"
    alias.symlink_to(s.ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match="private"):
        r.run(alias / "results" / (s.NAME + ".json"), "20261101", True, False)


@pytest.fixture
def retained_cli(tmp_path):
    """SCENARIO-VERIFY-8692-CUSTODY: preserve actual nested child logs before test cleanup."""
    yield tmp_path
    directory = os.environ.get("CARNOT_8692_CLI_RETENTION")
    if directory:
        for receipt in tmp_path.rglob("*.receipt.json"):
            value = json.loads(receipt.read_text())
            for source in [receipt, *[Path(value[k]) for k in ["stdout_path", "stderr_path"]]]:
                target = Path(directory) / tmp_path.name / source.relative_to(tmp_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


def test_real_private_cli_and_fresh_replay(tmp_path, retained_cli):
    """SCENARIO-VERIFY-8692-CUSTODY: real subprocesses retain immutable failure evidence."""
    output = tmp_path / (s.NAME + ".json")
    argv = [sys.executable, "-u", str(s.ROOT / s.CLI)]
    assert child(
        "private",
        argv + ["--private-control", "--output", str(output)],
        tmp_path / "logs",
        deadline=180,
    )["passed"]
    assert child("replay", argv + ["--cold-replay", str(output)], tmp_path / "logs", deadline=180)[
        "passed"
    ]
    original = json.loads(output.read_text())
    value = deepcopy(original)
    value["precision_mapping_ready_score"] = 99
    atomic_json(output, value)
    assert child(
        "reject", argv + ["--cold-replay", str(output)], tmp_path / "logs", expected=1, deadline=30
    )["passed"]
    atomic_json(output, original)
    primitive = Path(original["primitive_reference"]["path"])
    saved = primitive.read_bytes()
    work = json.loads(saved)
    work["primary_passed"] = False
    atomic_json(primitive, work)
    value = deepcopy(original)
    value["primitive_reference"] = s.reference(primitive)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="seal"):
        r.replay(output)
    primitive.write_bytes(saved)
    atomic_json(output, original)
    primitive.unlink()
    with pytest.raises(OSError):
        r.replay(output)
    failed = tmp_path / "failed" / (s.NAME + ".json")
    assert child(
        "failed",
        argv + ["--private-control", "--failed-child", "--output", str(failed)],
        tmp_path / "logs",
        expected=1,
        deadline=180,
    )["passed"]
    assert json.loads(failed.read_text())["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    "mutation", ["flag", "construction", "terminal", "rule", "head", "missing"]
)
def test_producer_rejections(monkeypatch, mutation):
    """REQ-REPORT-8692: a changed or flagged producer cannot authorize numerical readiness."""
    real_read = s.read
    if mutation == "head":
        from carnot.verify import v726_criterion_energy_fit as producer

        protocol = real_read(s.reference(s.ROOT / s.PROTOCOL))
        upstream = real_read(protocol["original_producers"]["8431"])
        monkeypatch.setattr(producer, "build", lambda *args: upstream)

    def changed(ref):
        """Mutate one independently authenticated object after its ordinary byte check."""
        if mutation == "missing":
            raise FileNotFoundError("controlled missing protocol")
        value = real_read(ref)
        if mutation == "flag" and value.get("experiment_id") == 8431:
            value["flagged_adversarial"] = True
        if mutation == "construction" and "result" in value:
            value["result"] = {}
        if mutation == "terminal" and "publication" in value:
            value["consumer"]["passed"] = False
        if mutation == "rule" and "objective_updates" in value:
            value["objective_updates"]["learning_rate"] = 0.2
        if mutation == "head" and "heads" in value:
            value["heads"]["spline9"][0]["parameter_count"] = 108
        if mutation == "head" and "result" in value:
            value["result"]["heads"]["spline9"][0]["parameter_count"] = 108
        return value

    monkeypatch.setattr(s, "read", changed)
    result = s.authenticate()
    assert not result["heads"]
    assert result["failures"]
    if mutation == "head":
        assert result["failures"][-1]["observed"] == "original_nine_input_head"


def test_blocked_measurement_and_replay_controls(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8692-CUSTODY: missing CPU resources block; corrupted seals reject."""
    monkeypatch.setattr(
        s,
        "authenticate",
        lambda: dict(heads=[], refs=[], failures=[], historical_model_provenance={}),
    )
    real_resources = r.resources

    def unavailable(path):
        """Exercise the actual scratch probe while making the memory precondition fail."""
        return dict(real_resources(path), memory_available_bytes=0)

    monkeypatch.setattr(r, "resources", unavailable)
    raw = tmp_path / "raw" / s.NAME / "invocations" / "control"
    work = r.measure(raw, "20261101")
    assert work["failures"][0]["upstream"] == "host"
    assert not work["primary_passed"]
    output = tmp_path / (s.NAME + ".json")
    validation_operand = tmp_path / "validation_operand.txt"
    validation_operand.write_text("actual retained validation operand")
    receipts = [
        dict(
            passed=True,
            references=[s.reference(validation_operand)],
            shards=[dict(reference=s.reference(validation_operand))],
        )
    ]
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(
        raw.parents[1] / "validation_seals" / "control.json",
        dict(validation=s.reference(raw / "validation.json")),
    )
    atomic_json(output, s.build(work, receipts, output))
    r.replay(output)
    report = json.loads(output.read_text())
    atomic_json(output, dict(report, precision_mapping_ready_score=99))
    with pytest.raises(ValueError, match="report_drift"):
        r.replay(output)
    atomic_json(output, report)
    saved_validation = (raw / "validation.json").read_bytes()
    atomic_json(raw / "validation.json", dict(receipts=[dict(passed=False)]))
    atomic_json(output, s.build(work, [dict(passed=False)], output))
    with pytest.raises(ValueError, match="independent_validation_seal"):
        r.replay(output)
    (raw / "validation.json").write_bytes(saved_validation)
    atomic_json(output, report)
    validation_operand.write_text("changed validation bytes")
    with pytest.raises(ValueError, match="operand_changed_bytes"):
        r.replay(output)
    validation_operand.write_text("actual retained validation operand")
    original = deepcopy(work)
    seal_path = raw.parents[1] / "seals" / "control.json"

    def replace_work(changes):
        """Rebind the trusted test seal only to reach each inner replay guard."""
        updated = dict(original, **changes)
        atomic_json(raw / "measurement.json", updated)
        atomic_json(
            seal_path,
            dict(
                measurement=s.reference(raw / "measurement.json"), code_refs=original["code_refs"]
            ),
        )
        atomic_json(output, s.build(updated, [], output))

    replace_work(dict(code_refs=[]))
    with pytest.raises(ValueError, match="source_seal"):
        r.replay(output)
    replace_work(dict(duration_s=-1))
    with pytest.raises(ValueError, match="clock_drift"):
        r.replay(output)
    replace_work({})
    actual_read = s.read

    def changed_heads(ref):
        """A changed geometry must fail before numerical reduction can run."""
        return (
            dict(heads=[head()])
            if Path(ref["path"]).name == "initial_heads.json"
            else actual_read(ref)
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(s, "read", changed_heads)
        scoped.setattr(s.numeric, "vectors", lambda heads: dict(changed=True))
        with pytest.raises(ValueError, match="geometry_drift"):
            r.replay(output)
    monkeypatch.setattr(s, "evaluate", lambda *args: dict(original, primary_passed=True))
    with pytest.raises(ValueError, match="numerical_replay_drift"):
        r.replay(output)
    with pytest.raises(ValueError, match="operand_changed_bytes"):
        r.verify_bytes(dict(path=str(output), sha256="sha256:wrong"))


def test_measurement_cap_and_validation_binding(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8692-CUSTODY: reserve validation time and bind the shared runner."""
    monkeypatch.setattr(
        s,
        "authenticate",
        lambda: dict(heads=[], refs=[], failures=[], historical_model_provenance={}),
    )
    ticks = iter([0, 3601 * 10**9])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        r.measure(tmp_path / "invocation", "20261101")

    def validation(raw):
        """The wrapper must supply this experiment's own paths to the retained collector."""
        assert r.validation.e is s
        assert raw == tmp_path
        retained = Path(os.environ["CARNOT_8692_CLI_RETENTION"]) / "control.receipt.json"
        atomic_json(retained, dict(actual_exit=0))
        (raw / "validation").mkdir(exist_ok=True)
        (raw / "validation" / "control.xml").write_text("<testsuite/>")
        return [dict(passed=True)]

    monkeypatch.setattr(r.validation, "validate", validation)
    receipts = r.validate(tmp_path)
    assert receipts[0] == dict(passed=True)
    assert receipts[1]["passed"]
