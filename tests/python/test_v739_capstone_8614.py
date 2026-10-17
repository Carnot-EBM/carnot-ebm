"""REQ-REPORT-8614 / REQ-VERIFY-8614: original evidence controls completion."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v739_capstone as runner
from carnot.reporting import v739_capstone_evidence as e
from carnot.reporting import v739_capstone_report as report
from carnot.reporting import v739_capstone_samples as samples


def cli(*args):
    """Real children expose standalone imports and checked publication errors."""
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
    root = tmp_path_factory.mktemp("v739-capstone")
    output = root / (e.NAME + ".json")
    result = cli("--date", "20261017", "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return output, json.loads(output.read_bytes())


def test_accounting(private):
    """SCENARIO-REPORT-8614-ACCOUNTING: thirteen dispositions preserve missing science."""
    output, value = private
    assert value["task_count"] == value["completed_count"] == 13
    assert value["administrative_complete"] and value["required_checks_passed"]
    assert value["verdict_class"] == "blocked" and value["failed_count"] == 1
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8602, 8615))
    assert sum(r["disposition"] == "pre_gate_receipt" for r in value["rows"]) == 5
    assert sum(r["disposition"] == "missing_primary_conductor_skip" for r in value["rows"]) == 2
    assert not value["publication_authorized"] and value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["generalized_learning_benefit_score"] == 0
    assert {8529, 8559} <= {r["experiment_id"] for r in value["preserved_nulls"]}
    assert {8582, 8584, 8592, 8603} <= {r["experiment_id"] for r in value["quarantined_inputs"]}
    assert value["qualification_failure_evidence"]["missing_lines"] == [94]
    assert set(value) <= set(value["field_principles"])
    report.replay(output)
    assert cli("--cold-replay", output).returncode == 0


def test_samples(private):
    """SCENARIO-REPORT-8614-SAMPLES: arithmetic never rehabilitates failed measurements."""
    output, value = private
    sample_path = output.parent / "test-samples.json"
    assert cli("--sample-output", sample_path).returncode == 0
    reduced = json.loads(sample_path.read_bytes())
    assert reduced == value["deterministic_claim_samples"]
    assert all(r["matches_original"] for r in reduced["causal_losses"])
    assert reduced["retention_maximum"]["matches_original"]
    assert reduced["arc_horizons"]["resolved_count"] == 0
    assert reduced["triviaplus_coverage"]["observed_questions"] is None
    assert reduced["complete_lean_cost"]["total_s"] is None
    assert all(r["matches_original"] for r in reduced["lean_prototype_sessions"])


def test_cli_controls(private, tmp_path):
    """SCENARIO-VERIFY-8614-CONTROLS: real missing and owned failure paths stay terminal."""
    _, value = private
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    for args, code in [
        (["--date", "20260230"], 1),
        (["--date", "123"], 1),
        (["--private-control"], 1),
        (["--cold-replay", tmp_path / "absent"], 1),
    ]:
        assert cli(*args).returncode == code
    for field, replacement in [
        ("task_count", 99),
        ("preserved_nulls", []),
        ("continuation_decisions", {}),
    ]:
        changed = dict(value, **{field: replacement})
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = tmp_path / (field + ".json")
        e.atomic_json(path, changed)
        with pytest.raises(ValueError):
            report.replay(path)
    assert cli("--cold-replay", path).returncode == 1


def test_rehashed_primitives(private, tmp_path):
    """REQ-REPORT-8614-REPLAY: rehashed operand, receipt and claim tampering fails."""
    _, value = private
    for kind in [
        "row",
        "sample",
        "contract",
        "clock",
        "source",
        "receipt",
        "protocol",
        "publication",
    ]:
        work = deepcopy(e.read(value["primitive_reference"]))
        receipts = deepcopy(value["validation_receipts"])
        if kind == "row":
            work["slots"][0]["eligible"] = True
        elif kind == "sample":
            work["samples"]["triviaplus_coverage"]["observed_questions"] = 128
        elif kind == "contract":
            work["contract"]["ready"] = True
        elif kind == "clock":
            work["duration_s"] += 1
        elif kind == "source":
            work["code_refs"][0]["sha256"] = "wrong"
        elif kind == "publication":
            work["publication"]["paper_ready"] = not work["publication"]["paper_ready"]
        elif kind == "receipt":
            receipts[0]["duration_s"] += 1
        else:
            work["protocol"]["archive_authority"] = "invented"
        raw = tmp_path / kind
        e.atomic_json(raw / "measurement.json", work)
        candidate = report.build(work, receipts, raw / (e.NAME + ".json"))
        candidate["primitive_reference"] = e.reference(raw / "measurement.json")
        candidate["reproducibility_checksum"] = e.checksum(candidate)
        e.atomic_json(raw / "candidate.json", candidate)
        with pytest.raises(ValueError):
            report.replay(raw / "candidate.json")


def test_evidence_boundaries(private, tmp_path):
    """REQ-VERIFY-8614: missing primaries and broken seals cannot become eligible."""
    task = e.methods.protocol()["task_map_v739"][4]
    with patch.object(e, "_find_artifact_by_task_id", return_value=None):
        row = e.resolve(task, tmp_path, [])
    assert row["verdict_class"] == "blocked" and not row["producer_executed"]
    assert row["status"] == "missing" and row["primary_reference"]["sha256"] is None
    task = e.methods.protocol()["task_map_v739"][0]
    original = e.read(e.reference(e.ROOT / task["deliverable"]))
    with patch.object(e, "read_bound_sidecar", side_effect=ValueError("stale")):
        assert not e.resolve(task, tmp_path, [])["eligible"]
    with patch.object(
        e,
        "read",
        return_value=dict(original, verdict_class="positive", acceptance_gates={"bad": False}),
    ):
        assert not e.resolve(task, tmp_path, [])["eligible"]


def test_measurement_failures(private, tmp_path):
    """REQ-VERIFY-8614: resource failures and measured deadlines remain explicit."""
    _, value = private
    work = e.read(value["primitive_reference"])
    with patch.object(e, "child", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="sample_child_failed"):
            e.sample(tmp_path / "sample-failure")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "write_probe").write_bytes(b"private disk scratch")
    resource = dict(work["preconditions"], tools_available=False)
    with (
        patch.object(e.tempfile, "mkdtemp", return_value=str(scratch)),
        patch.object(e, "resources", return_value=resource),
        patch.object(e, "slots", return_value=[]),
        patch.object(e, "sample", return_value=(work["samples"], work["branch_receipts"][0])),
        patch.object(e, "child", return_value=work["publication_receipt"]),
    ):
        measured = e.measure(tmp_path / "resource", "20261017")
        assert measured["precondition_failures"][0]["field"] == "disk_memory_tools"
        with patch.object(e.time, "monotonic_ns", side_effect=[0, 3600_000_000_000]):
            with pytest.raises(TimeoutError, match="measurement_deadline"):
                e.measure(tmp_path / "deadline", "20261017")


def test_validation_adapter_and_default_run(private, tmp_path):
    """REQ-VERIFY-8614: the normal route executes the unchanged validation adapter."""
    _, value = private
    with patch.object(runner.shipped, "validate", return_value=[dict(passed=True)]):
        assert runner.validate(tmp_path) == [dict(passed=True)]
    work = e.read(value["primitive_reference"])
    with (
        patch.object(e, "measure", return_value=work),
        patch.object(runner, "validate", return_value=value["validation_receipts"][:1]),
        patch.object(runner, "terminal", return_value=dict(passed=True)),
    ):
        assert runner.run(tmp_path / (e.NAME + ".json"), "20261017", False, False) == 0


def test_replay_stream_rejection(private):
    """REQ-REPORT-8614-REPLAY: authentic receipt envelopes still require exact streams."""
    output, value = private
    work = e.read(value["primitive_reference"])
    sha256_file = e.methods.sha256_file
    stream = value["validation_receipts"][0]["stderr_path"]

    def changed(path):
        return "wrong" if str(path) == stream else sha256_file(path)

    with patch.object(e.methods, "sha256_file", side_effect=changed):
        with pytest.raises(ValueError, match="receipt_stream_drift"):
            report.replay(output)
    with patch.object(e, "sample", return_value=(work["samples"], {})):
        with patch.object(e, "slots", return_value=[]):
            with pytest.raises(ValueError, match="reduction_drift"):
                report.replay(output)


def test_retention_rejects_changed_bytes(private, tmp_path):
    """SCENARIO-REPORT-8614-SAMPLES: independently authenticate retention operands."""
    # Stub the heavy ARC import only for this early hash rejection path.
    from types import SimpleNamespace

    with patch.dict(
        sys.modules, {"carnot.reporting.arc_outcome_identifiability_8599": SimpleNamespace()}
    ):
        with patch.object(e.methods, "sha256_file", return_value="wrong"):
            with pytest.raises(ValueError):
                samples.reduce()


def test_archive_and_prior_comparison():
    """REQ-REPORT-8614: planning lag and actual prior failures must remain distinct."""
    assert report.ARCHIVE_AUTHORITY["completed_archive_ends_at"] == "2026.10.737"
    task = dict(
        id="exp8605-causal-brier-stream",
        prior_failures=[
            dict(
                experiment_id="exp8594-causal-brier-stream",
                verdict="producer_absent_after_upstream_gate",
            )
        ],
    )
    work = dict(
        protocol=dict(task_map_v739=[task, {}], original_producers={}),
        slots=[dict(honest_verdict="complete_blocked_current", verdict_class="blocked")],
    )
    prior = dict(
        rows=[
            dict(
                experiment_id=8594,
                honest_verdict="blocked_gate_check_failed",
                checks=[dict(field="methods_ready_score", observed=0)],
            )
        ],
        quarantined_inputs=[],
        qualified_scientific_results=[],
    )
    decision = report.decisions(work, prior)[0]
    assert decision["actual_prior_outcomes"][0]["honest_verdict"] == "blocked_gate_check_failed"
    assert decision["actual_prior_outcomes"][0]["failed_checks"][0]["observed"] == 0


def test_owned_receipt_binding(private, tmp_path):
    """REQ-VERIFY-8614: inline receipt edits cannot clear failed owned checks."""
    _, value = private
    changed = deepcopy(value)
    changed["validation_receipts"][0]["passed"] = False
    changed["reproducibility_checksum"] = e.checksum(changed)
    path = tmp_path / "edited-receipts.json"
    e.atomic_json(path, changed)
    with pytest.raises(ValueError, match="validation_receipts_drift"):
        report.replay(path)
    work = deepcopy(e.read(value["primitive_reference"]))
    work["samples"]["causal_losses"][0]["matches_original"] = False
    rejected = report.build(work, [], tmp_path / (e.NAME + ".json"))
    assert rejected["verdict_class"] == "disqualified"
    assert any(r["field"] == "owned_receipts_and_samples" for r in rejected["gate_check_summary"])
