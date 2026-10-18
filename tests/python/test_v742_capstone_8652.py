"""REQ-REPORT-8652 / REQ-VERIFY-8652: original evidence controls reconciliation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v742_capstone as runner
from carnot.reporting import v742_capstone_evidence as e
from carnot.reporting import v742_capstone_report as report
from carnot.reporting import v742_capstone_samples as samples


def cli(*args):
    """A real child tests standalone imports without relying on the caller's path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-VERIFY-8652-REJECTION: use private actual publication bytes."""
    output = tmp_path_factory.mktemp("v742") / (e.NAME + ".json")
    result = cli("--date", "20261018", "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return output, json.loads(output.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8652-ACCOUNTING: scheduling completion grants no science credit."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["intended_count"] == value["completed_count"] == len(value["rows"]) == 13
    assert value["executed_producer_count"] == 7
    assert value["administrative_complete"] and value["required_checks_passed"]
    assert value["verdict_class"] == "blocked" and value["run_date"] == "20261018"
    assert len(value["continuation_decisions"]) == 13 and value["prd_gap_decisions"]
    assert not value["publication_authorized"]
    assert all(value[k] == v and type(v) is int for k, v in value["branch_gate_values"].items())
    assert all(value[k] == 0 for k in ["nfr01_score", "generalized_learning_benefit_score"])
    assert any(r["disposition"] == "conductor_cascade_skip" for r in value["rows"])
    assert any(r["operand_state"] == "missing_path" for r in value["gate_edge_rows"])
    assert any(r["operand_state"] == "observed_zero" for r in value["gate_edge_rows"])
    assert {8641, 8642, 8646, 8650} <= {r["experiment_id"] for r in value["quarantined_inputs"]}
    assert set(value) <= set(value["field_principles"])
    assert set(["G1", "G2", "G3", "G4"]) <= set(value)
    assert value["deterministic_claim_samples"]["arc_reset_transfer"]["complete_horizon_count"] == 0


def test_cli_rejections(private, tmp_path):
    """REQ-VERIFY-8652: owned child failure disqualifies; malformed calls reject."""
    output = tmp_path / (e.NAME + ".json")
    result = cli("--output", output, "--private-control", "--failed-child")
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    for args in [
        ["--date", "123"],
        ["--date", "20260230"],
        ["--private-control"],
        ["--cold-replay", tmp_path / "missing"],
    ]:
        assert cli(*args).returncode == 1


@pytest.mark.parametrize(
    "field", ["rows", "duration_s", "protocol", "samples", "publication", "receipt", "source"]
)
def test_rehashed_tampering(private, tmp_path, field):
    """REQ-VERIFY-8652: repaired checksums cannot authorize changed original operands."""
    _, value = private
    changed = deepcopy(value)
    work = deepcopy(e.read(value["primitive_reference"]))
    if field == "receipt":
        changed["validation_receipts"][0]["duration_s"] += 1
    elif field == "source":
        work["source_refs"][0]["sha256"] = "wrong"
    elif field == "rows":
        changed["rows"][0]["eligible"] = False
    elif field == "duration_s":
        work["duration_s"] += 1
    elif field == "protocol":
        work["protocol"]["milestone"] = "wrong"
    elif field == "samples":
        work["samples"]["arc_reset_transfer"]["complete_horizon_count"] += 1
    else:
        work["publication"]["paper_ready"] = not work["publication"]["paper_ready"]
    e.atomic_json(tmp_path / "work.json", work)
    changed["primitive_reference"] = e.reference(tmp_path / "work.json")
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(tmp_path / "candidate.json", changed)
    assert cli("--cold-replay", tmp_path / "candidate.json").returncode == 1


def test_owned_failure_and_adapters(private, tmp_path):
    """REQ-VERIFY-8652: failed resources and validators cannot qualify owned work."""
    _, value = private
    work = e.read(value["primitive_reference"])
    assert report.build(work, [], tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"
    with patch.object(runner.shipped, "validate", return_value=[]):
        assert runner.validate(tmp_path) == [
            dict(name="preserved_child_coverage", passed=False, shards=[])
        ]
    with patch.object(runner.shipped, "terminal", return_value={"passed": False}):
        assert runner.terminal(tmp_path, tmp_path) == {"passed": False}


def test_resource_and_snapshot_rejections(private, tmp_path):
    """REQ-VERIFY-8652: actual missing files, changed snapshots and expired clocks reject."""
    from types import SimpleNamespace

    _, value = private
    work = e.read(value["primitive_reference"])
    source = tmp_path / "source"
    source.write_bytes(b"source")
    ref = e.freeze(source, tmp_path / "raw")
    Path(ref["snapshot_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable_snapshot_changed"):
        e.freeze(source, tmp_path / "raw")
    assert not e.freeze(tmp_path / "missing", tmp_path)["exists"]

    def resources(directory):
        (directory / "write_probe").write_bytes(b"private disk scratch")
        return dict(work["preconditions"], tools_available=False)

    with (
        patch.object(e, "resources", side_effect=resources),
        patch.object(e, "slots", return_value=[]),
        patch.object(e, "sample", return_value=(work["samples"], work["branch_receipts"][0])),
        patch.object(e, "child", return_value=work["publication_receipt"]),
    ):
        assert e.measure(tmp_path / "resource", "20261018")["precondition_failures"]
        with patch.object(
            e, "time", SimpleNamespace(monotonic_ns=iter([0, 3600_000_000_000]).__next__)
        ):
            with pytest.raises(TimeoutError, match="measurement_deadline"):
                e.measure(tmp_path / "expired", "20261018")
    with patch.object(e, "child", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="sample_child_failed"):
            e.sample(tmp_path / "child")


@pytest.mark.parametrize("kind", ["retention", "feedback", "streaming", "producer"])
def test_sample_source_rejections(tmp_path, kind):
    """REQ-VERIFY-8652: unbound arithmetic operands cannot be admitted as diagnostics."""
    original = e.methods.checked

    def checked(ref):
        name = Path(ref["path"]).name
        reject = {"retention": "retention", "feedback": "released", "streaming": ".jsonl"}.get(
            kind, "never"
        )
        return False if reject in name else original(ref)

    with patch.object(e.methods, "checked", side_effect=checked):
        if kind == "producer":
            with patch.object(e, "ROOT", tmp_path):
                with pytest.raises(ValueError, match="original_producer_identity"):
                    samples.reduce()
        else:
            with pytest.raises(ValueError):
                samples.reduce()


def test_validation_copy_control(tmp_path):
    """REQ-VERIFY-8652: validation copies child bytes before the combine command runs."""
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\n")
    source = tmp_path / ".coverage.child"
    source.write_bytes(b"SQLite original child bytes")

    def validate(raw):
        return [
            runner.shipped.child(
                "coverage_combine", ["combine", "--rcfile=" + str(config)], raw, deadline=60
            ),
            runner.shipped.child("tests_0", ["test"], raw, deadline=60),
        ]

    with (
        patch.object(runner.shipped, "validate", side_effect=validate),
        patch.object(runner, "child", return_value=dict(passed=True)) as child,
    ):
        receipts = runner.validate(tmp_path / "raw")
    assert receipts[-1]["passed"]
    assert Path(receipts[-1]["shards"][0]["path"]).read_bytes() == source.read_bytes()
    assert child.call_args_list[0].args[1][-1] == str(tmp_path / "combine_copies")
    assert child.call_args_list[1].kwargs["deadline"] == 600


@pytest.mark.parametrize("kind", ["missing", "receipt", "stream", "unbound"])
def test_replay_receipt_rejections(private, tmp_path, kind):
    """REQ-VERIFY-8652: original receipt bytes and missing-state custody remain binding."""
    _, original = private
    value = deepcopy(original)
    work = deepcopy(e.read(value["primitive_reference"]))
    if kind == "missing":
        present = tmp_path / "present"
        present.write_bytes(b"unexpected")
        work["source_refs"].append(dict(path=str(present), exists=False, sha256=None))
    elif kind == "unbound":
        value["validation_receipts"].append(dict(name="never_executed", passed=True))
    elif kind == "receipt":
        value["validation_receipts"][0]["duration_s"] += 1
    else:
        receipt = value["validation_receipts"][0]
        stdout = tmp_path / (receipt["name"] + ".stdout")
        stdout.write_text("changed output")
        receipt["stdout_path"] = str(stdout)
        e.atomic_json(tmp_path / (receipt["name"] + ".receipt.json"), receipt)
    e.atomic_json(tmp_path / "work.json", work)
    e.atomic_json(tmp_path / "validation.json", dict(receipts=value["validation_receipts"]))
    value.update(
        primitive_reference=e.reference(tmp_path / "work.json"),
        validation_reference=e.reference(tmp_path / "validation.json"),
    )
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(tmp_path / "candidate.json", value)
    with (
        patch.object(e, "sample", return_value=(work["samples"], {})),
        patch.object(e, "slots", return_value=work["slots"]),
        patch.object(e, "bindings", return_value=work["declared_gates"]),
    ):
        with pytest.raises(ValueError):
            report.replay(tmp_path / "candidate.json")


def test_missing_prior_does_not_become_null(private):
    """REQ-REPORT-8652: absent historical science has a missing cause, never a null verdict."""
    _, value = private
    work = deepcopy(e.read(value["primitive_reference"]))
    work["protocol"]["task_map_v742"][0]["prior_failures"] = [
        dict(experiment_id="exp9999-missing", verdict="registered failure")
    ]
    prior = e.read(e.reference(e.ROOT / "results/experiment_8639_v741_capstone.json"))
    decisions = report.decisions(work, value["rows"], prior)
    assert (
        decisions[0]["actual_prior_outcomes"][0]["literal_verdict"]
        == "complete_blocked_missing_historical_producer"
    )


def test_exact_fallback_and_failure_causes(tmp_path):
    """REQ-REPORT-8652: exact fallback identity and original child failures bind causes."""
    task = dict(
        id="exp9999-example-task", deliverable="results/experiment_9999_v742_example_task.json"
    )
    with patch.object(e, "ROOT", tmp_path):
        assert e.locate(task) == tmp_path / task["deliverable"]
        (tmp_path / "results").mkdir()
        fallback = tmp_path / "results/experiment_9999_example_task.json"
        fallback.write_text("{}")
        assert e.locate(task) == fallback
        (tmp_path / "results/experiment_9999_unrelated.json").write_text("{}")
        with pytest.raises(ValueError, match="unexpected_primary_identity"):
            e.locate(task)
    stdout = tmp_path / "failure.stdout"
    stdout.write_text("E carnot.testing.guard.TrackedResultWriteError: tracked results\n")
    value = dict(
        validation_receipts=[
            dict(name="child", passed=False, exit_code=-9, stdout_path=str(stdout))
        ]
    )
    causes = e.failure_evidence(value)
    assert causes[0]["exit_code"] == -9
    assert "TrackedResultWriteError" in causes[0]["failure_lines"][0]
