"""REQ-REPORT-8639 / REQ-VERIFY-8639: original operands control terminal claims."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v741_capstone as runner
from carnot.reporting import v741_capstone_evidence as e
from carnot.reporting import v741_capstone_report as report
from carnot.reporting import v741_capstone_samples as samples


def cli(*args):
    """Real children exercise imports and rejection paths outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-VERIFY-8639-CONTROLS: retain actual private publication evidence."""
    output = tmp_path_factory.mktemp("capstone") / (e.NAME + ".json")
    result = cli("--date", "20261018", "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return output, json.loads(output.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8639-ACCOUNTING: completion grants no new science credit."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["task_count"] == value["completed_count"] == len(value["rows"]) == 13
    assert value["administrative_complete"] and value["required_checks_passed"]
    assert value["verdict_class"] == "blocked"
    assert value["executed_producer_count"] == 7
    assert value["run_date"] == "20261018"
    assert len(value["continuation_decisions"]) == 13
    assert not value["publication_authorized"]
    assert all(value[k] == v for k, v in value["branch_gate_values"].items())
    assert all(value[k] == 0 for k in value["branch_gate_values"])
    assert {8616, 8629, 8634, 8638} <= {r["experiment_id"] for r in value["quarantined_inputs"]}
    assert set(value) <= set(value["field_principles"])
    assert value["cited_upstream_artifacts"] == value["source_artifact_hashes"]
    assert value["deterministic_claim_samples"]["v740"]["released_feedback"]["matches_original"]
    assert all(r["matches_original"] for r in value["deterministic_claim_samples"]["streaming"])
    assert value["deterministic_claim_samples"]["arc"]["matches_original"]
    assert any(r["operand_state"] == "missing_path" for r in value["declared_upstream_gate_rows"])
    assert any(r["operand_state"] == "observed_zero" for r in value["declared_upstream_gate_rows"])


def test_cli_errors(private, tmp_path):
    """REQ-VERIFY-8639: failed owned children disqualify and malformed calls reject."""
    _, value = private
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
    changed = dict(value, task_count=99)
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(tmp_path / "changed.json", changed)
    assert cli("--cold-replay", tmp_path / "changed.json").returncode == 1


@pytest.mark.parametrize(
    "kind",
    ["slots", "samples", "contract", "duration_s", "source", "protocol", "receipt", "publication"],
)
def test_rehashed_tampering(private, tmp_path, kind):
    """REQ-VERIFY-8639: repaired hashes do not make altered claims authoritative."""
    _, value = private
    work = deepcopy(e.read(value["primitive_reference"]))
    receipts = deepcopy(value["validation_receipts"])
    if kind == "slots":
        work["slots"][0]["eligible"] = False
    elif kind == "samples":
        work["samples"]["arc"]["matches_original"] = False
    elif kind == "contract":
        work["contract"]["ready"] = False
    elif kind == "duration_s":
        work["duration_s"] += 1
    elif kind == "source":
        work["code_refs"][0]["sha256"] = "wrong"
    elif kind == "protocol":
        work["protocol"]["milestone"] = "wrong"
    elif kind == "receipt":
        receipts[0]["duration_s"] += 1
    else:
        work["publication"]["paper_ready"] = not work["publication"]["paper_ready"]
    e.atomic_json(tmp_path / "measurement.json", work)
    changed = dict(
        value,
        primitive_reference=e.reference(tmp_path / "measurement.json"),
        validation_receipts=receipts,
    )
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(tmp_path / "candidate.json", changed)
    with pytest.raises(ValueError):
        report.replay(tmp_path / "candidate.json")


def test_owned_failure_controls(private, tmp_path):
    """REQ-VERIFY-8639: resource limits and failed validation cannot qualify work."""
    _, value = private
    work = e.read(value["primitive_reference"])
    assert report.build(work, [], tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"
    assert (
        report.build(
            work, [dict(name="never_executed", passed=True)], tmp_path / (e.NAME + ".json")
        )["verdict_class"]
        == "disqualified"
    )
    altered = deepcopy(work)
    altered["samples"]["v740"]["retention_maximum"]["matches_original"] = False
    assert (
        report.build(altered, work["branch_receipts"], tmp_path / (e.NAME + ".json"))[
            "verdict_class"
        ]
        == "disqualified"
    )
    with (
        patch.object(runner.shipped, "validate", return_value=[{"passed": False}]),
        patch.object(runner.shipped, "terminal", return_value={"passed": False}),
    ):
        assert runner.validate(tmp_path) == [{"passed": False}]
        assert runner.terminal(tmp_path, tmp_path) == {"passed": False}

    def resources(directory):
        (directory / "write_probe").write_bytes(b"private disk scratch")
        return dict(work["preconditions"], tools_available=False)

    with (
        patch.object(e, "resources", side_effect=resources),
        patch.object(e, "slots", return_value=[]),
        patch.object(e, "sample", return_value=(work["samples"], work["branch_receipts"][0])),
        patch.object(e, "child", return_value=work["publication_receipt"]),
    ):
        measured = e.measure(tmp_path / "resource", "20261018")
        assert measured["precondition_failures"][0]["observed"] is False
        with patch.object(
            e, "time", SimpleNamespace(monotonic_ns=iter([0, 3600_000_000_000]).__next__)
        ):
            with pytest.raises(TimeoutError):
                e.measure(tmp_path / "expired", "20261018")
    with patch.object(e.previous, "child", return_value={"passed": False}):
        with pytest.raises(ValueError, match="sample_child_failed"):
            e.sample(tmp_path / "failed")
    with patch.object(samples.e, "ROOT", tmp_path):
        with pytest.raises(ValueError, match="original_producer_identity"):
            samples.producer(8633)
    original = e.methods.sha256_file
    with (
        patch.object(samples.historical, "reduce", return_value={}),
        patch.object(
            e.methods,
            "sha256_file",
            side_effect=lambda p: "wrong" if str(p).endswith(".jsonl") else original(p),
        ),
    ):
        with pytest.raises(ValueError, match="streaming_window_hash"):
            samples.reduce()


def test_gate_and_authority_rejections(private, tmp_path):
    """REQ-VERIFY-8639: missing fields, conflicting primaries and invalid seals reject."""
    from carnot.reporting import v741_capstone_inputs as inputs

    _, value = private
    work = e.read(value["primitive_reference"])
    tasks = work["protocol"]["task_map_v741"]
    forged = dict(value, validation_receipts=[dict(name="never_executed", passed=True)])
    e.atomic_json(tmp_path / "forged.json", forged)
    with pytest.raises(ValueError, match="unbound_validation_receipt"):
        report.replay(tmp_path / "forged.json")
    with patch.object(inputs, "read_bound_sidecar", side_effect=ValueError("seal")):
        row = inputs.resolve(tasks[0], tmp_path / "seal", [])
        assert not row["eligible"] and not row["terminal_sealed"]
    with patch.object(inputs.methods, "ROOT", tmp_path):
        directory = tmp_path / "results"
        directory.mkdir()
        for suffix in ["a", "b"]:
            (directory / ("experiment_9999_" + suffix + ".json")).write_text("{}")
        with pytest.raises(ValueError, match="ambiguous_original_primary"):
            inputs.locate(dict(id="exp9999-test", deliverable="missing"))
    changed = deepcopy(work)
    changed["declared_gates"][0]["observed"] = "forged"
    e.atomic_json(tmp_path / "gate.json", changed)
    altered = dict(value, primitive_reference=e.reference(tmp_path / "gate.json"))
    e.atomic_json(tmp_path / "candidate.json", altered)
    with pytest.raises(ValueError, match="declared_gate_drift"):
        report.replay(tmp_path / "candidate.json")
    changed["protocol"]["task_map_v741"][0]["prior_failures"] = [
        dict(experiment_id="exp9998-missing")
    ]
    decisions = report.decisions(
        changed,
        value["rows"],
        dict(rows=[], quarantined_inputs=[], qualified_scientific_results=[]),
    )
    assert (
        decisions[0]["actual_prior_outcomes"][0]["literal_verdict"]
        == "complete_blocked_missing_primary"
    )
    assert inputs.cause(dict(honest_verdict="complete_blocked_v741_runtime")) == inputs.cause(
        dict(honest_verdict="complete_blocked_v740_runtime")
    )


def test_conductor_scope(tmp_path):
    """REQ-REPORT-8639: old receipts and current START rows cannot close a missing slot."""
    from carnot.reporting import v741_capstone_inputs as inputs

    task = dict(
        id="exp9997-missing",
        title="Unique absent producer",
        deliverable="results/missing-9997.json",
    )
    lines = [
        "| yesterday | Unique absent producer | GATE_BLOCK | retired |",
        "| now | Milestone 2026.10.741 activated | OK | 13 tasks queued |",
        "| now | Unique absent producer | START | running |",
    ]
    assert inputs.resolve(task, tmp_path / "pending", lines)["status"] == "missing"
    lines += ["| now | Unique absent producer | GATE_BLOCK | upstream retired |"]
    assert inputs.resolve(task, tmp_path / "terminal", lines)["status"] == "completed"


def test_validation_budget(tmp_path):
    """REQ-VERIFY-8639: coverage overhead gets a bounded budget without dropping nodes."""
    with patch.object(runner, "child", return_value=dict(passed=True)) as measured:
        assert runner.validation_child("tests_0", ["pytest"], tmp_path, deadline=240)["passed"]
        assert measured.call_args.kwargs["deadline"] == 600
        assert runner.validation_child("collection", ["pytest"], tmp_path, deadline=90)["passed"]
        assert measured.call_args.kwargs["deadline"] == 90
