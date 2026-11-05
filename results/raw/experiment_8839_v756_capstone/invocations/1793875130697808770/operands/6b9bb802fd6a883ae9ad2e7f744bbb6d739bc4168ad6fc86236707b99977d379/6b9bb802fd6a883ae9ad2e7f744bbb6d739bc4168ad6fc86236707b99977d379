"""REQ-REPORT-8838 / REQ-VERIFY-8838: costs do not imply fabric support."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_budget_8838 as m


def producer(branch):
    """SCENARIO-REPORT-8838-CONTRACT: small operands keep failure controls cheap."""
    value = dict(
        experiment_id=8835 if branch == "whole" else 8831,
        required_checks_passed=True,
        verdict_class="null",
        honest_verdict="complete_null_fixture",
    )
    if branch == "whole":
        row = dict(
            unit_id="window",
            cluster_id="stream",
            arm="read_only",
            seed=1,
            status="completed",
            denominator=1,
            complete_ns=2000000000,
            segments=[
                dict(
                    predictions=[dict(primitive_checks=5)],
                    releases=[
                        dict(
                            optimizer_steps=1,
                            coefficient_committed=False,
                            template_added=False,
                            feedback_verifier_calls=1,
                        )
                    ],
                )
            ],
        )
        value.update(whole_cost_ready_score=1, window_rows=[row], paired_cost_intervals=[])
    else:
        value.update(
            evidence_rows_ready_score=1,
            rows=[
                dict(
                    unit_id="w/intact",
                    cluster_id="w",
                    arm="local_then_joint",
                    seed=1,
                    status="completed",
                    denominator=1,
                    metric=1,
                    missing_reason=None,
                    operation_counts=dict(join_attempts=0, lookup_attempts=7),
                    primitive_operations=7,
                    wall_time_s=0.001,
                )
            ],
            schedule_benefit_score=0,
        )
    return value


def test_independent_rows_and_target():
    """SCENARIO-REPORT-8838-CONTRACT: a missing sibling never hides valid costs."""
    whole = m.reduce({"whole": producer("whole")})
    assert whole["complete_cost_refs"][0]["complete_operation_s"] == 2
    assert whole["target_latency_requirements"][0]["target_latency_s"] == 0.02
    assert whole["rows"][0]["operation_counts"]["dense_gradient_calls"] == 1
    assert whole["rows"][0]["operation_counts"]["parameter_commits"] == 0
    assert whole["board_operation_rows"][0]["paid_host_counts"]["primitive_checks"] == 5
    witness = m.reduce({"witness": producer("witness")})
    assert witness["paid_witness_rows"][0]["operation_counts"]["join_attempts"] == 0
    assert len(witness["board_operation_rows"]) == 3 * len(m.OPERATIONS)
    assert witness["sampling_call_count"] == 0
    assert witness["physical_evidence_state"]["GateMate"]["retry_allowed"] is False
    assert not witness["complete_cost_refs"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"whole_cost_ready_score": None}, "missing_field"),
        ({"whole_cost_ready_score": 0}, "observed_zero"),
        ({"required_checks_passed": False}, "rejected_producer"),
        ({"flagged_adversarial": True}, "rejected_producer"),
        ({"window_rows": []}, "rejected_producer"),
    ],
)
def test_admission(change, reason):
    """REQ-VERIFY-8838: missing fields, zero and rejected producers stay distinct."""
    value = producer("whole")
    value.update(change)
    assert m.admission(value, "whole") == reason
    assert m.admission(producer("whole"), "whole") is None


def test_snapshot_and_replay(tmp_path):
    """SCENARIO-VERIFY-8838-REPLAY: original bytes and reductions both constrain replay."""
    work = m.collect(tmp_path / "raw")
    assert set(work["evidence"]) == {"whole", "witness"}
    validation = dict(passed=False, receipts=[], coverage=dict(complete=False))
    value = m.build(work, validation)
    assert value["verdict_class"] == "disqualified"
    path = tmp_path / (m.NAME + ".json")
    m.atomic_json(path, value)
    assert m.replay(path)["passed"]
    altered = deepcopy(value)
    altered["current_work_receipt"]["duration_s"] = 99
    m.atomic_json(path, altered)
    with pytest.raises(ValueError, match="current_receipt_drift"):
        m.replay(path)
    altered = deepcopy(value)
    fake = tmp_path / "fake.json"
    m.atomic_json(fake, producer("whole"))
    altered["phase_work"]["evidence"]["whole"] = dict(
        path=str(fake), sha256=m.sha256_file(fake), original_path=str(fake)
    )
    altered = m.build(altered["phase_work"], validation)
    m.atomic_json(path, altered)
    with pytest.raises(ValueError, match="authority_pin_drift"):
        m.replay(path)
    altered = deepcopy(value)
    altered["source_artifact_hashes"] = altered["source_artifact_hashes"][1:]
    m.atomic_json(path, altered)
    # A source-only entry can be absent without altering the producer manifest.
    operand = next(
        r
        for r in value["source_artifact_hashes"]
        if r["original_path"].endswith("measurement.json")
    )
    altered["source_artifact_hashes"] = [r for r in value["source_artifact_hashes"] if r != operand]
    m.atomic_json(path, altered)
    with pytest.raises(ValueError, match="operand_manifest_drift"):
        m.replay(path)
    altered = deepcopy(value)
    altered["source_artifact_hashes"][0]["original_path"] = str(m.ROOT / next(iter(m.HISTORY)))
    m.atomic_json(path, altered)
    with pytest.raises(ValueError, match="authority_pin_drift"):
        m.replay(path)
    altered = deepcopy(value)
    altered["sampling_call_count"] = 1
    m.atomic_json(path, altered)
    with pytest.raises(ValueError, match="reduction_drift"):
        m.replay(path)
    m.atomic_json(path, value)
    ref = work["refs"][0]
    Path(ref["path"]).write_text("changed")
    with pytest.raises(ValueError, match="operand_hash_drift"):
        m.replay(path)


def test_cold_cli(tmp_path):
    """SCENARIO-VERIFY-8838-REPLAY: the actual script bootstraps outside the checkout."""
    output = tmp_path / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    def invoke(args):
        return subprocess.run(
            command + args, cwd="/tmp", env=env, capture_output=True, text=True, timeout=90
        )

    result = invoke(["--private-output", str(output), "--date", "20261105"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert invoke(["--replay", str(output)]).returncode == 0
    value = json.loads(output.read_text())
    value["duration_s"] = 99
    m.atomic_json(output, value)
    assert invoke(["--replay", str(output)]).returncode == 1
    assert invoke(["--date", "20261104"]).returncode == 2
    assert m.main(["--replay", str(tmp_path / "absent")]) == 1


def test_external_admission_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8838: each failed external gate keeps its exact path and cause."""
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "SOURCES", [])
    monkeypatch.setattr(m, "HISTORY", {})
    result = m.collect(tmp_path / "raw")
    assert {f["missing_reason"] for f in result["failures"]} == {"missing_file"}
    monkeypatch.setattr(m, "HISTORY", {"absent-boundary.json": "sha256:absent"})
    assert (
        m.collect(tmp_path / "missing-authority")["failures"][0]["missing_reason"] == "missing_file"
    )
    monkeypatch.setattr(m, "HISTORY", {})
    file = tmp_path / "producer.json"
    value = producer("whole")
    value.update(whole_cost_ready_score=0)
    m.atomic_json(file, value)
    monkeypatch.setattr(
        m,
        "PRODUCERS",
        dict(whole=("producer.json", m.sha256_file(file), "whole_cost_ready_score", 8835)),
    )
    result = m.collect(tmp_path / "zero")
    assert result["failures"][0]["missing_reason"] == "observed_zero"
    value.update(whole_cost_ready_score=1)
    m.atomic_json(file, value)
    result = m.collect(tmp_path / "drift")
    assert result["failures"][0]["missing_reason"] == "rejected_producer"
    with pytest.raises(ValueError, match="operand_hash_drift"):
        m.retain(file, tmp_path, "sha256:wrong")
    spec = tmp_path / "openspec/capabilities/private/spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("current")
    monkeypatch.setattr(m.subprocess, "check_output", lambda *args, **kwargs: b"old")
    with pytest.raises(ValueError, match="operand_hash_drift"):
        m.retain(spec, tmp_path / "recovered", "sha256:wrong")


def test_readiness_and_public_main(tmp_path, monkeypatch):
    """REQ-VERIFY-8838: owned failure suppresses readiness; external absence stays blocked."""
    from carnot.reporting import hardware_budget_validation_8838 as v
    from carnot.reporting import primary_publication as publication

    work = dict(
        evidence={},
        refs=[],
        failures=[],
        checks=dict(
            repository_virtualenv=True,
            private_output_writable=True,
            free_disk_bytes=999999999,
            minimum_free_bytes=1,
        ),
        started_ns=1,
        ended_ns=1001,
        owner_pid=os.getpid(),
    )
    validation = dict(passed=True, receipts=[], coverage=dict(complete=True))
    assert m.build(work, validation)["verdict_class"] == "null"
    work["failures"] = [dict(check="missing_file")]
    assert m.build(work, validation)["verdict_class"] == "blocked"
    work["checks"]["repository_virtualenv"] = False
    assert m.build(work, validation)["verdict_class"] == "blocked"
    work["internal_failure"] = True
    assert m.build(work, validation)["verdict_class"] == "disqualified"
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "collect", lambda raw: work)
    monkeypatch.setattr(v, "qualify", lambda raw: validation)
    monkeypatch.setattr(v, "terminal", lambda path, raw: dict(passed=True))
    monkeypatch.setattr(
        publication, "publish_primary", lambda output, value, validator: validator(output)
    )
    assert m.main(["--date", "20261105"]) == 0


def test_validation_custody(tmp_path, monkeypatch):
    """REQ-VERIFY-8838: altered logs and self-reported readiness cannot qualify."""
    log = tmp_path / "check.log"
    log.write_text("real output")
    state = dict(
        passed=False,
        coverage=dict(complete=False),
        receipts=[
            dict(
                name="failed_control",
                log_path=str(log),
                log_sha256=m.sha256_file(log),
                passed=False,
            )
        ],
    )
    m.verify_validation(state)
    log.write_text("changed output")
    with pytest.raises(ValueError, match="validation_log_drift"):
        m.verify_validation(state)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        m.verify_validation(dict(passed=True, coverage=dict(complete=True), receipts=[]))
    from coverage import Coverage

    source = tmp_path / "measured.py"
    source.write_text("counter = 1\n")
    tracer = Coverage(
        config_file=False, include=[str(source)], data_file=str(tmp_path / ".coverage")
    )
    tracer.start()
    exec(compile(source.read_text(), str(source), "exec"), {})
    tracer.stop()
    report = tmp_path / "coverage.json"
    tracer.json_report(outfile=str(report))
    data = json.loads(report.read_text())
    monkeypatch.setattr(m, "OWNED", [str(source)])
    names = [
        "owned_pytest",
        "private_e2e018_021",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    ]
    state = dict(
        passed=True,
        receipts=[
            dict(name=n, log_path=str(log), log_sha256=m.sha256_file(log), passed=True, exit_code=0)
            for n in names
        ],
        coverage=dict(path=str(report), sha256=m.sha256_file(report), files=data["files"]),
    )
    m.verify_validation(state)
    state["coverage"]["sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="coverage_hash_drift"):
        m.verify_validation(state)
    state["coverage"].update(sha256=m.sha256_file(report), files={})
    with pytest.raises(ValueError, match="coverage_statement_drift"):
        m.verify_validation(state)


def test_scanner_metadata_boundaries():
    """REQ-VERIFY-8838: clocks and numeric formats are metadata, not measured outcomes."""
    from scripts import adversarial_verify, verdict_row_consistency_lint

    work = dict(
        evidence={},
        refs=[],
        failures=[],
        checks=dict(
            repository_virtualenv=True,
            private_output_writable=True,
            free_disk_bytes=999999999,
            minimum_free_bytes=1,
        ),
        started_ns=553422534819890,
        ended_ns=553423534819890,
        owner_pid=os.getpid(),
    )
    value = m.build(work, dict(passed=False, receipts=[], coverage=dict(complete=False)))
    flags = []
    adversarial_verify.check_tautology(value, flags)
    assert not flags
    assert not verdict_row_consistency_lint.check_all_rows_null(value, value["rows"])
    assert isinstance(value["board_operation_rows"][0]["requirements"]["precision"], str)
