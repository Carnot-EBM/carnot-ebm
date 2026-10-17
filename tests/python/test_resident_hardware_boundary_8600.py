"""REQ-REPORT-8600, REQ-VERIFY-8600: resident mapping preserves original authority."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import resident_hardware_boundary_8600 as e
from carnot.reporting import resident_hardware_evidence_8600 as evidence
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    """Real children exercise imports and publication beyond in-process adapters."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=180)


def test_original_cost_and_scope():
    """SCENARIO-REPORT-8600-BOUNDARY: original qualified costs support conditional ceilings only."""
    source = evidence.authenticate(e.ROOT)
    assert source["board"]["qualified"]
    assert source["resident_qualification"]["qualified"]
    assert source["cost"]["qualified"], source["cost"]["checks"]
    assert len(source["cost"]["cells"]) == 60
    assert len(evidence.matrix(source)) == 50
    assert all(r["zero_cost_ceiling"] < 10 for r in evidence.ceilings(source["cost"]))


def test_missing_changed_and_rehashed_original(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8600-REPLAY: independent pins reject repaired producer bytes."""
    source = evidence.authenticate(tmp_path)
    assert not source["cost"]["qualified"]
    assert source["cost"]["checks"][0]["observed"] is None
    assert all(r["zero_cost_ceiling"] is None for r in evidence.ceilings(source["cost"]))
    path = tmp_path / evidence.COST
    atomic_json(path, {})
    assert not evidence.original(tmp_path, evidence.COST, "resident_cost_ready_score")["qualified"]
    monkeypatch.setattr(evidence, "COST_PIN", e.reference(path)["sha256"])
    assert not evidence.original(tmp_path, evidence.COST, "resident_cost_ready_score")["qualified"]


def test_real_private_children(tmp_path):
    """SCENARIO-VERIFY-8600-PUBLICATION: failures and repaired summaries remain terminal."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        args = ["--private-control", "--root", tmp_path / "absent", "--output", output]
        if failed:
            args.append("--owned-failure")
        result = cli(tmp_path, *args, "--date", "20261018")
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["run_date"] == "20261018"
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert value["physical_hardware_ready_score"] == 0
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        value["conditional_ceiling_rows"][0]["zero_cost_ceiling"] = 11
        value["reproducibility_checksum"] = e.checksum(value)
        atomic_json(output, value)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--deliberate-error").returncode == 7
    assert cli(tmp_path, "--date", "invalid").returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2


def test_mapping_and_primitive_tamper(tmp_path):
    """SCENARIO-REPORT-8600-REPLAY: fresh reconstruction defeats self-consistent primitive changes."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, tmp_path, "20261017")
    output = tmp_path / (e.NAME + ".json")
    receipts = [dict(name="private", passed=True)]

    def candidate(current):
        atomic_json(raw / "measurement.json", current)
        value = e.build(current, receipts, raw, output)
        atomic_json(output, value)
        return value

    original_work = deepcopy(work)
    value = candidate(work)
    assert work == original_work
    assert e.build(work, receipts, raw, output) == value
    assert value["hardware_boundary_ready_score"] == 1
    assert value["acquisition_decision"]["cpu_residence_removes_bottleneck"] is True
    assert value["target_model"]["parameter_count"] == 109
    assert e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["cost"]["cells"][0]["total_s"] += 1
    atomic_json(Path(changed["operand_reference"]["path"]), changed["evidence"])
    changed["operand_reference"] = e.reference(Path(changed["operand_reference"]["path"]))
    candidate(changed)
    assert not e.replay(output)


def test_manifest_and_date_controls(tmp_path, monkeypatch):
    """REQ-VERIFY-8600: original union and date-independent controls remain explicit."""
    from carnot.reporting import resident_hardware_runner_8600 as runner

    monkeypatch.setattr(runner, "_manifest", lambda *_: [dict(argv=["pytest"])])
    assert runner.manifest(tmp_path) == [dict(argv=["pytest"])]
    assert runner.main(["--date", "20260230"]) == 2
    with pytest.raises(SystemExit):
        runner.main(["--private-control"])


def test_owned_original_field_and_publication_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8600: rehashed field and terminal failures cannot qualify fresh clocks."""
    path = tmp_path / evidence.COST
    value = json.loads((e.ROOT / evidence.COST).read_bytes())
    value["required_checks_passed"] = False
    atomic_json(path, value)
    monkeypatch.setattr(evidence, "COST_PIN", e.reference(path)["sha256"])
    assert not evidence.original(tmp_path, evidence.COST, "resident_cost_ready_score")["qualified"]
    value["required_checks_passed"] = True
    atomic_json(path, value)
    monkeypatch.setattr(evidence, "COST_PIN", e.reference(path)["sha256"])
    monkeypatch.setattr(evidence, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
    assert not evidence.original(tmp_path, evidence.COST, "resident_cost_ready_score")["qualified"]
