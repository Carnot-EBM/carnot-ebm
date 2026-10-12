"""REQ-REPORT-8461: qualification cannot become a learning benefit claim."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import objective_checkpoint_8461 as q
from carnot.reporting import objective_checkpoint_runner_8461 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.verify import objective_checkpoint_8461 as e


def test_manifest_and_cli_failures(tmp_path):
    """SCENARIO-VERIFY-8461-VALIDATION: bounded genuine child coverage includes failure paths."""
    plan = q.manifest(tmp_path / "plan")
    assert q.REQUIRED_CHECKS <= {r["name"] for r in plan}
    assert "_exit" in (tmp_path / "plan/coverage.ini").read_text()
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert (
        runner.main(
            ["--worker", str(tmp_path / "store"), "--configuration", str(tmp_path / "missing")]
        )
        == 1
    )
    cfg = tmp_path / "cfg.json"
    atomic_json(cfg, e.config(e.legacy.head()))
    assert runner.main(["--worker", str(tmp_path / "store"), "--configuration", str(cfg)]) == 0
    final = tmp_path / "store/final.json"
    assert runner.main(["--state-replay", str(final), "--configuration", str(cfg)]) == 0
    changed = json.loads(final.read_bytes())
    changed["head"]["parameters"][0] += 1
    atomic_json(final, changed)
    assert runner.main(["--state-replay", str(final), "--configuration", str(cfg)]) == 1


def test_blocked_measure_and_rehashed_tampering(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8461-TERMINAL: external absence stays distinct from numeric zero."""
    raw = tmp_path / "raw"
    work = q.measure(tmp_path / "absent-root", raw, tmp_path / "private")
    receipts = [
        child(n, [sys.executable, "-c", "print('private receipt')"], raw / "checks", deadline=10)
        for n in sorted(q.REQUIRED_CHECKS)
    ]
    output = tmp_path / (q.NAME + ".json")
    value = q.build(work, receipts, raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["objective_kernel_ready_score"] == 0
    assert value["objective_stream_ready_score"] == 0
    assert value["gate_check_summary"]
    assert value["censored_count"] == 126
    atomic_json(output, value)
    assert q.replay(output)

    def absent_authority(root, raw):
        raise FileNotFoundError("required_authority")

    monkeypatch.setattr(q.evidence.methods, "authority", absent_authority)
    assert q.replay(output)
    changed = deepcopy(value)
    changed["objective_kernel_ready_score"] = 1
    changed["reproducibility_checksum"] = q.checksum(changed)
    atomic_json(output, changed)
    assert not q.replay(output)
    assert not q.replay(tmp_path / "missing")
    disqualified = q.build(work, [], raw, output)
    assert disqualified["verdict_class"] == "disqualified"
    assert not disqualified["required_checks_passed"]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-VERIFY-8461-RECOVERY: original sealed producers supply the real full head."""
    parent = tmp_path_factory.mktemp("8461-measured")
    return q.measure(q.ROOT, parent / "raw", parent / "private"), parent


def test_qualified_primitive_replay(measured):
    """SCENARIO-REPORT-8461-TERMINAL: real states qualify only with every owned check."""
    work, parent = measured
    assert not work["failures"]
    primitive = q.read(work["primitive_reference"])
    assert len(primitive["restart_rows"]) == 36
    assert len(primitive["process_exit_rows"]) == 32
    assert all(r["passed"] for r in primitive["restart_rows"])
    receipts = [
        child(n, [sys.executable, "-c", "print('private receipt')"], parent / "checks", deadline=10)
        for n in sorted(q.REQUIRED_CHECKS)
    ]
    output = parent / (q.NAME + ".json")
    value = q.build(work, receipts, parent / "raw", output)
    assert value["objective_kernel_ready_score"] == 1
    atomic_json(output, value)
    assert q.replay(output)
    changed = deepcopy(value)
    changed["derivative_rows"][0]["absolute_error"] = 999
    changed["reproducibility_checksum"] = q.checksum(changed)
    atomic_json(output, changed)
    assert not q.replay(output)


def test_private_cli_publication_and_date(tmp_path):
    """SCENARIO-REPORT-8461-TERMINAL: real consumers accept honest missing-operand accounting."""
    output = tmp_path / "results" / (q.NAME + ".json")
    assert runner.run(tmp_path / "absent", output, tmp_path / "private", validate=False) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert q.replay(output)
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261011"])


@pytest.mark.parametrize(
    "operand",
    [
        "code",
        "authority",
        "head",
        "derivative",
        "state",
        "hash",
        "exit",
        "receipt",
        "log",
        "pass",
        "authority_object",
        "primary_pin",
        "transition",
    ],
)
def test_rehashed_primitive_and_receipt_controls(tmp_path, measured, operand):
    """SCENARIO-REPORT-8461-TERMINAL: rebuilds reject repaired hashes over incorrect operands."""
    import shutil

    work, parent = measured
    work = deepcopy(work)
    primitive = deepcopy(q.read(work["primitive_reference"]))
    receipt = child(
        "required",
        [sys.executable, "-c", 'print("independent log")'],
        tmp_path / "logs",
        deadline=10,
    )
    if operand == "code":
        work["code_refs"] = work["code_refs"][1:]
    elif operand == "authority":
        work["authority"]["activated"] = False
    elif operand == "authority_object":
        work["authority"]["tasks"][0]["MODEL_SPECS"] = ["forged"]
    elif operand == "primary_pin":
        ref = next(
            r
            for r in work["source_refs"]
            if r["path"].endswith("experiment_8434_v726_capacity_online_learning.json")
        )
        changed_primary = q.read(ref)
        changed_primary["tampered"] = True
        path = tmp_path / "experiment_8434_v726_capacity_online_learning.json"
        atomic_json(path, changed_primary)
        ref.update(q.reference(path))
    elif operand == "transition":
        primitive["transition_rows"][0]["observed_error"] = "forged"
    elif operand == "pass":
        receipt["passed"] = False
    elif operand == "head":
        primitive["head"]["parameters"][0] += 0.1
    elif operand == "derivative":
        primitive["derivative_rows"][0]["absolute_error"] = 999
    elif operand == "state":
        state = deepcopy(q.read(primitive["restart_rows"][0]["final_state"]))
        state["head"]["parameters"][0] += 0.1
        atomic_json(tmp_path / "state.json", state)
        primitive["restart_rows"][0]["final_state"] = q.reference(tmp_path / "state.json")
    elif operand == "hash":
        primitive["restart_rows"][0]["logical_state_hash"] = "wrong"
    elif operand == "exit":
        primitive["process_exit_rows"][0]["actual_exit"] = 0
    elif operand == "receipt":
        receipt["duration_s"] += 1
    else:
        receipt["stdout_sha256"] = "wrong"
        atomic_json(Path(receipt["stdout_path"]).with_suffix(".receipt.json"), receipt)
    atomic_json(tmp_path / "primitives.json", primitive)
    work["primitive_reference"] = q.reference(tmp_path / "primitives.json")
    shutil.copyfile(parent / "raw/update_adapter.py", tmp_path / "update_adapter.py")
    atomic_json(tmp_path / "measurement.json", work)
    output = tmp_path / (q.NAME + ".json")
    value = q.build(work, [receipt], tmp_path, output)
    atomic_json(output, value)
    assert not q.replay(output)


def test_external_resource_and_deadline_gates(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8461-VALIDATION: failed resource gates and measured deadlines stay visible."""
    from carnot.reporting import objective_checkpoint_evidence_8461 as evidence

    monkeypatch.setattr(evidence.inherited, "filesystem_type", lambda path: "tmpfs")
    ticks = iter([0, 3_601_000_000_000, 3_602_000_000_000, 3_603_000_000_000])
    monkeypatch.setattr(evidence.time, "monotonic_ns", lambda: next(ticks))
    work = evidence.measure(tmp_path / "missing", tmp_path / "raw", tmp_path / "private")
    assert any(r["check"] == "cpu_resources" for r in work["failures"])
    assert any(r["check"] == "measurement_cap_s" for r in work["failures"])


def test_foreign_task_authority_is_rejected(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8461-TERMINAL: an activated foreign task cannot authorize this output."""
    from carnot.reporting import objective_checkpoint_evidence_8461 as evidence

    monkeypatch.setattr(
        evidence.methods,
        "authority",
        lambda root, raw: dict(
            activated=True,
            gate_check_summary=[],
            tasks=[dict(id=q.TASK, deliverable="foreign.json")],
        ),
    )
    value = evidence.authenticate(tmp_path, tmp_path / "raw")
    assert not value["authority"]["activated"]
    assert any(r["check"] == "current_full_authority" for r in value["failures"])


def test_original_producer_rejection_is_recorded(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8461-TERMINAL: a child with an ineligible producer cannot supply a head."""
    from carnot.reporting import objective_checkpoint_evidence_8461 as evidence

    def rejecting_child(name, argv, logs, **kwargs):
        result = argv[-1]
        program = 'import json,sys;json.dump(dict(qualified=False,gate_check_summary=[]),open(sys.argv[1],"w"))'
        return child(name, [sys.executable, "-c", program, result], logs, deadline=20)

    monkeypatch.setattr(evidence, "child", rejecting_child)
    value = evidence.authenticate(q.ROOT, tmp_path / "raw")
    assert value["head"] is None
    assert any(
        r["check"] == "original_head_and_checkpoint_qualification" for r in value["failures"]
    )


def test_real_cli_validation_copies_genuine_coverage(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8461-VALIDATION: real command failure stays disqualified with genuine shards."""
    original = q.manifest

    def private_plan(private):
        original(private)
        rc = "--rcfile=" + str(private / "coverage.ini")
        archive = next(row for row in original(private) if row["name"] == "coverage_archive")
        rows = [
            dict(
                name=n,
                argv=[sys.executable, "-m", "coverage", *args],
                expected=exit_code,
                deadline=30,
                scope="owned",
            )
            for n, args, exit_code in [
                ("coverage_probe", ["run", rc, str(q.ROOT / q.CLI), "--deliberate-error"], 7),
                ("coverage_combine", ["combine", rc, "--keep"], 0),
                ("coverage_json", ["json", rc, "-o", str(private / "coverage.json")], 0),
            ]
        ]
        rows.insert(1, archive)
        return rows

    monkeypatch.setattr(q, "manifest", private_plan)
    output = tmp_path / "results" / (q.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert any(".coverage" in r["path"] for r in value["raw_shard_hashes"])
    assert q.replay(output)


def test_recovery_runtime_paths_are_private(measured):
    """SCENARIO-VERIFY-8461-RECOVERY: primitive copies outlive private child checkpoints."""
    work, parent = measured
    workers = [r for r in work["process_receipts"] if "--worker" in r["argv"]]
    assert workers
    assert all(
        Path(r["argv"][r["argv"].index("--worker") + 1]).is_relative_to(parent / "private")
        for r in workers
    )


def test_recovery_deadline_stops_before_child(tmp_path):
    """SCENARIO-VERIFY-8461-VALIDATION: an expired measurement budget cannot start another child."""
    from carnot.reporting import objective_checkpoint_evidence_8461 as evidence

    rows, exits, receipts = evidence.recovery(
        tmp_path / "raw", e.legacy.head(), tmp_path / "private", deadline_ns=0
    )
    assert not rows and not exits and not receipts


def test_failed_child_stops_recovery(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8461-VALIDATION: a real failed worker stops the remaining panel."""
    import time
    from carnot.reporting import objective_checkpoint_evidence_8461 as evidence

    def failing_child(name, argv, logs, **kwargs):
        program = "import sys;from carnot.reporting.objective_checkpoint_runner_8461 import main;main(sys.argv[1:]);sys.exit(7)"
        return child(name, [sys.executable, "-u", "-c", program, *argv[3:]], logs, deadline=30)

    monkeypatch.setattr(evidence, "child", failing_child)
    rows, exits, receipts = evidence.recovery(
        tmp_path / "raw",
        e.legacy.head(),
        tmp_path / "private",
        deadline_ns=time.monotonic_ns() + 60 * 10**9,
    )
    assert len(rows) == 1 and rows[0]["passed"] is False
    assert not exits and receipts[0]["exit_code"] == 7
    atomic_json(tmp_path / "failed_rows.json", rows)
