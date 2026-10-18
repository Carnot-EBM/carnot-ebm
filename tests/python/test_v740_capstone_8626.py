"""REQ-REPORT-8626 / REQ-VERIFY-8626: original evidence limits capstone claims."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v740_capstone as runner
from carnot.reporting import v740_capstone_evidence as e
from carnot.reporting import v740_capstone_report as report
from carnot.reporting import v740_capstone_samples as samples


def cli(*args):
    """Standalone children expose imports and rejection paths hidden by unit mocks."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-VERIFY-8626-CONTROLS: real original operands drive private publication."""
    output = tmp_path_factory.mktemp("capstone") / (e.NAME + ".json")
    result = cli("--date", "20261018", "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return output, json.loads(output.read_bytes())


def test_accounting_and_replay(private, tmp_path):
    """SCENARIO-REPORT-8626-ACCOUNTING: twelve dispositions provide zero generalization."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["task_count"] == len(value["rows"]) == 12
    assert value["completed_count"] == 12
    assert len(value["continuation_decisions"]) == 12
    assert value["administrative_complete"]
    assert value["verdict_class"] == "blocked"
    assert value["run_date"] == "20261018"
    assert value["independent_generalization_score"] == value["nfr01_score"] == 0
    assert value["complete_cost_benefit_score"] == 1
    assert value["learning_execution_ready_score"] == 0
    assert not value["publication_authorized"]
    assert value["archive_authority"]["completed_archive_ends_at"] == "2026.10.738"
    assert {8616, 8603} <= {r["experiment_id"] for r in value["quarantined_inputs"]}
    assert set(value) <= set(value["field_principles"])
    reduced, receipt = e.sample(tmp_path)
    assert receipt["passed"]
    assert reduced == value["deterministic_claim_samples"]
    assert reduced["triviaplus_coverage"]["usable_clusters"] == 90
    assert all(r["matches_original"] for r in reduced["complete_lean_cost"]["rows"])
    assert reduced["released_feedback"]["matches_original"]


def test_cli_rejections(private, tmp_path):
    """SCENARIO-VERIFY-8626-CONTROLS: missing and failed owned work cannot qualify."""
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
    for field in ["task_count", "nfr01_score", "complete_cost_benefit_score"]:
        changed = dict(value, **{field: 999})
        changed["reproducibility_checksum"] = e.checksum(changed)
        e.atomic_json(tmp_path / "tampered.json", changed)
        with pytest.raises(ValueError):
            report.replay(tmp_path / "tampered.json")
    assert cli("--cold-replay", tmp_path / "tampered.json").returncode == 1


@pytest.mark.parametrize(
    "kind",
    ["slots", "samples", "contract", "duration_s", "source", "protocol", "receipt", "publication"],
)
def test_primitive_tamper(private, tmp_path, kind):
    """REQ-VERIFY-8626: repaired envelopes cannot replace independent original operands."""
    _, value = private
    work = deepcopy(e.read(value["primitive_reference"]))
    receipts = deepcopy(value["validation_receipts"])
    if kind == "slots":
        work["slots"][0]["eligible"] = False
    elif kind == "samples":
        work["samples"]["triviaplus_coverage"]["usable_clusters"] = 999
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


def test_adapters_and_owned_failures(private, tmp_path):
    """REQ-VERIFY-8626: validation adapters preserve failed exits and empty receipts."""
    _, value = private
    work = e.read(value["primitive_reference"])
    assert e.gate(tmp_path, "owned", True, False)["upstream"] == e.TASK
    assert report.build(work, [], tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"
    with (
        patch.object(runner.shipped, "validate", return_value=[{"passed": False}]),
        patch.object(runner.shipped, "terminal", return_value={"passed": False}),
    ):
        assert runner.validate(tmp_path) == [{"passed": False}]
        assert runner.terminal(tmp_path, tmp_path) == {"passed": False}


def test_measurement_rejections(private, tmp_path):
    """REQ-VERIFY-8626: resource failure, child failure and expired clocks stay failures."""
    _, value = private
    work = e.read(value["primitive_reference"])

    def resources(directory):
        (directory / "write_probe").write_bytes(b"private disk scratch")
        return dict(work["preconditions"], tools_available=False)

    with (
        patch.object(e, "resources", side_effect=resources),
        patch.object(e, "slots", return_value=[]),
        patch.object(e, "sample", return_value=(work["samples"], work["branch_receipts"][0])),
        patch.object(e, "child", return_value=work["publication_receipt"]),
    ):
        measured = e.measure(tmp_path / "resources", "20261018")
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
            samples.producer(8620)


@pytest.mark.parametrize(
    "filename,reason",
    [
        ("released_feedback.json", "released_feedback_hash"),
        ("causal_brier_runner_8582.py", "deadline_source_hash"),
    ],
)
def test_original_hash_failures(filename, reason):
    """REQ-VERIFY-8626: original source and released feedback retain independent pins."""
    original = e.methods.sha256_file
    with (
        patch.object(samples.historical, "reduce", return_value={}),
        patch.object(
            e.methods,
            "sha256_file",
            side_effect=lambda p: "wrong" if Path(p).name == filename else original(p),
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            samples.reduce()
