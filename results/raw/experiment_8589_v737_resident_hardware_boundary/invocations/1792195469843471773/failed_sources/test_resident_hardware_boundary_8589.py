"""REQ-REPORT-8589, REQ-VERIFY-8589: qualify mapping without inventing resident cost."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import resident_hardware_boundary_8589 as e
from carnot.reporting import resident_hardware_evidence_8589 as evidence
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    """Real children reveal import, replay and publication failures hidden by mocks."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=180)


def test_original_evidence_and_unqualified_resident(tmp_path):
    """SCENARIO-REPORT-8589-BOUNDARY: failed upstream checks cannot supply ceilings."""
    source = evidence.authenticate(e.ROOT)
    assert source["board"]["qualified"]
    assert not source["cost"]["qualified"]
    assert source["cost"]["checks"]
    assert all(r["zero_cost_ceiling"] is None for r in evidence.ceilings(source["cost"]))
    assert len(evidence.matrix(source)) == 50


def test_absent_and_changed_original(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8589-REPLAY: missing and rehashed foreign originals remain blocked."""
    source = evidence.authenticate(tmp_path)
    assert not source["board"]["qualified"]
    assert source["board"]["checks"][0]["observed"] is None
    path = tmp_path / evidence.BOARD
    atomic_json(path, dict(experiment_id=8563))
    assert not evidence.authenticate(tmp_path)["board"]["qualified"]
    monkeypatch.setitem(evidence.PINS, evidence.BOARD, e.reference(path)["sha256"])
    assert not evidence.authenticate(tmp_path)["board"]["qualified"]


def test_qualified_disjoint_ceiling_and_partition_rejection():
    """REQ-REPORT-8589: host and frame costs survive ideal arithmetic removal."""
    rows = []
    for length in [1, 4, 16]:
        for batch in [1, 8]:
            for repeat in range(10):
                rows.append(
                    dict(
                        length=length,
                        batch=batch,
                        repeat=repeat,
                        arm="resident_python",
                        total_s=100.0,
                        seconds={op: 10.0 for op in evidence.OPERATIONS},
                    )
                )
    cost = dict(qualified=True, cells=rows)
    evidence.partition(rows)
    ceilings = evidence.ceilings(cost)
    assert len(ceilings) == 12
    assert all(r["zero_cost_ceiling"] < 10 for r in ceilings)
    assert all(r["fixed_host_clock_s"] >= 700 for r in ceilings)
    rows[0]["total_s"] = 99.0
    with pytest.raises(ValueError, match="disjoint_complete_cost"):
        evidence.partition(rows)


def test_private_cli_missing_owned_failure_and_rehashed_tamper(tmp_path):
    """SCENARIO-VERIFY-8589-PUBLICATION: genuine children preserve terminal rejection paths."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        args = ["--private-control", "--root", tmp_path / "absent", "--output", output]
        if failed:
            args.append("--owned-failure")
        result = cli(tmp_path, *args)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert value["physical_hardware_ready_score"] == 0
        assert value["hardware_boundary_ready_score"] == 0
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        value["conditional_ceiling_rows"][0]["zero_cost_ceiling"] = 11
        value["reproducibility_checksum"] = e.checksum(value)
        atomic_json(output, value)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--deliberate-error").returncode == 7
    assert cli(tmp_path, "--date", "20261015").returncode == 2


def test_qualified_cost_adapter_and_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8589: only complete original cost reductions can populate ceiling operands."""
    value = dict(
        rows=[{}] * 240,
        completed_count=240,
        work_reference={},
        primary_contrast=dict(full_parity=True),
        lifecycle_contrast=dict(full_parity=True),
    )
    pairs, clocks = [], []
    for length in [1, 4, 16]:
        for batch in [1, 8]:
            for repeat in range(10):
                key = f"{length}-{batch}-{repeat}"
                pairs.append(
                    dict(
                        unit_id=key,
                        length=length,
                        batch=batch,
                        repeat=repeat,
                        arm="resident_python",
                        total_session_wall_s=100.0,
                    )
                )
                clocks.extend(
                    dict(unit_id=key, operation=op, duration_s=10.0) for op in evidence.MAPPING
                )
    pairs.append(dict(arm="cold_python"))
    monkeypatch.setattr(
        evidence, "original", lambda *_: dict(qualified=True, checks=[], refs=[], value=value)
    )
    monkeypatch.setattr(evidence.resident, "consume", lambda *_: dict(passed=True))
    monkeypatch.setattr(evidence.resident, "read", lambda *_: {})
    monkeypatch.setattr(
        evidence.resident,
        "reduce",
        lambda *_: dict(paired_session_rows=pairs, operation_cost_rows=clocks),
    )
    source = evidence.authenticate(tmp_path)
    assert source["cost"]["qualified"]
    assert len(evidence.matrix(source)[0]["qualified_cost_cells"]) == 60
    value["primary_contrast"]["full_parity"] = False
    assert not evidence.authenticate(tmp_path)["cost"]["qualified"]
    with pytest.raises(ValueError, match="complete_resident_roster"):
        evidence.partition([])


def test_original_binding_and_primitive_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8589-REPLAY: repaired summaries cannot authorize failed bindings or changed primitives."""
    value = json.loads((e.ROOT / evidence.BOARD).read_bytes())
    path = tmp_path / evidence.BOARD
    atomic_json(path, value)
    monkeypatch.setitem(evidence.PINS, evidence.BOARD, e.reference(path)["sha256"])
    monkeypatch.setattr(evidence, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
    assert not evidence.original(tmp_path, evidence.BOARD, "hardware_boundary_ready_score")[
        "qualified"
    ]
    monkeypatch.setattr(evidence, "read_bound_sidecar", lambda *_: dict(report=dict(passed=True)))
    value["work_reference"] = dict(path=str(tmp_path / "absent"), sha256="sha256:bad")
    atomic_json(path, value)
    monkeypatch.setitem(evidence.PINS, evidence.BOARD, e.reference(path)["sha256"])
    result = evidence.original(tmp_path, evidence.BOARD, "hardware_boundary_ready_score")
    assert not result["qualified"]
    assert result["checks"][0]["passed"] is False


def test_bounded_timeout_and_manifest(tmp_path, monkeypatch):
    """REQ-VERIFY-8589: timeout reports missing evidence while the scoped manifest preserves original nodes."""
    from carnot.reporting import resident_hardware_runner_8589 as runner

    def expired(*_):
        e.scan_timeout(14, None)

    monkeypatch.setattr(e.evidence, "authenticate", expired)
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    assert not work["evidence"]["board"]["qualified"]
    assert work["evidence"]["board"]["checks"][0]["passed"] is False
    monkeypatch.setattr(
        runner.validation,
        "manifest",
        lambda *_: [
            dict(collected_node_ids=["original"], argv=["pytest", "-q"]),
            dict(argv=["ruff"]),
        ],
    )
    assert runner.manifest(tmp_path)[0]["argv"] == ["pytest", "-vv"]
    with pytest.raises(SystemExit) as error:
        runner.main(["--private-control"])
    assert error.value.code == 2


def test_cold_replay_rehashes_and_earned_mapping(tmp_path):
    """SCENARIO-REPORT-8589-REPLAY: original pins defeat repaired work, operand, code and log claims."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, tmp_path)
    work_path = raw / "measurement.json"
    output = tmp_path / (e.NAME + ".json")
    receipts = [dict(name="private_test_receipt", passed=True)]

    def candidate(current_work):
        atomic_json(work_path, current_work)
        value = e.build(current_work, receipts, raw, output)
        atomic_json(output, value)
        return value

    value = candidate(work)
    assert value["hardware_boundary_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert e.replay(output)
    value["work_reference"]["sha256"] = "sha256:changed"
    atomic_json(output, value)
    assert not e.replay(output)
    changed = json.loads(json.dumps(work))
    changed["evidence"]["cost"]["qualified"] = True
    changed["evidence"]["cost"]["cells"] = []
    candidate(changed)
    assert not e.replay(output)
    candidate(work)
    operands = Path(work["operand_reference"]["path"])
    atomic_json(operands, {})
    changed = json.loads(json.dumps(work))
    changed["operand_reference"] = e.reference(operands)
    candidate(changed)
    assert not e.replay(output)
    atomic_json(operands, work["evidence"])
    candidate(work)
    log = tmp_path / "receipt.log"
    log.write_text("original")
    receipts[0].update(stdout_path=str(log), stdout_sha256=e.reference(log)["sha256"])
    candidate(work)
    log.write_text("changed")
    assert not e.replay(output)
    missing = json.loads(json.dumps(work))
    missing["code_config_hashes"][0]["path"] = str(tmp_path / "absent")
    candidate(missing)
    assert not e.replay(output)
