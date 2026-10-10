"""REQ-VERIFY-8353 / REQ-REPORT-8353: qualify readers without inferred GPU repair."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import yaml

from carnot.verify import runtime_reader_8353 as q
from carnot.verify import runtime_reader_execution_8353 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference


def cli(tmp_path, *args):
    """Run actual script children so dispatch and coverage include fresh imports."""
    return subprocess.run(
        [str(q.ROOT / ".venv/bin/python"), "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("milestone", [q.MILESTONE, "2026.10.717"])
def test_authority(tmp_path, milestone):
    """SCENARIO-VERIFY-8353-AUTHORITY: full bytes accept old/current and reject drift."""
    root = q.private_authority(tmp_path / "root", milestone)
    assert cli(tmp_path, "--authority", milestone, "--root", root).returncode == 0
    active = root / "research-roadmap.yaml"
    value = yaml.safe_load(active.read_bytes())
    value["milestone"] = "wrong"
    active.write_text(yaml.safe_dump(value))
    assert cli(tmp_path, "--authority", milestone, "--root", root).returncode == 1
    value["milestone"] = milestone
    value["tasks"][0]["prompt"] += " corrupt"
    active.write_text(yaml.safe_dump(value))
    assert cli(tmp_path, "--authority", milestone, "--root", root).returncode == 1
    assert not q.authority(tmp_path / "absent", tmp_path / "missing", milestone)["activated"]


def test_measured_source_closure_and_boundary(tmp_path):
    """SCENARIO-VERIFY-8353-BOUNDARY: immutable failures precede any natural probe."""
    source = json.loads((q.ROOT / q.UPSTREAM).read_bytes())
    identity = json.loads(Path(source["runtime_binding_path"]).read_bytes())
    with (
        patch.object(q.old, "inventory", return_value=(identity, [])),
        patch.object(
            q.old, "probe_changed", side_effect=AssertionError("unchanged probe forbidden")
        ),
    ):
        work = q.measure(q.ROOT, tmp_path / "raw")
    assert work["historical_fixture_manifest"]
    assert work["historical"]["verdict_class"] == "disqualified"
    assert not work["diagnostic"]["rows"]
    work["boundary_path"] = str(tmp_path / "raw/measurement.json")
    atomic_json(tmp_path / "raw/measurement.json", work)
    value = q.build(work, tmp_path / "raw", [dict(passed=True, scope="owned")])
    assert value["runtime_reader_ready_score"] == 1
    assert value["runtime_changed_score"] == value["cuda_context_ready_score"] == 0
    assert value["honest_verdict"] == "complete_blocked_cuda_environment_unchanged"
    assert value["runtime_identity"] == identity
    work["current"]["driver"] = "constructed changed control"
    with (
        patch.object(q.old, "inventory", return_value=(work["current"], [])),
        patch.object(q.old, "probe_changed", return_value=work["diagnostic"]) as probe,
    ):
        q.measure(q.ROOT, tmp_path / "changed")
    assert probe.call_count == 1
    with patch.object(q.old, "inventory", side_effect=AssertionError("missing source")):
        missing = q.measure(tmp_path / "absent", tmp_path / "missing_measure")
    assert not missing["authority"]["activated"]
    with (
        patch.object(q, "closure", side_effect=ValueError("corrupted closure")),
        patch.object(q.old, "inventory", return_value=(identity, [])),
    ):
        corrupted = q.measure(q.ROOT, tmp_path / "corrupted_closure")
    assert any(c.get("field") == "historical_source_closure" for c in corrupted["checks"])


def test_cli_replay_tamper_and_failure(tmp_path):
    """SCENARIO-REPORT-8353-REPLAY: cold children reject changed headlines and closure."""
    output = tmp_path / (q.NAME + ".json")
    run = cli(tmp_path, "--private-run", "--root", tmp_path / "absent", "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8353 and value["verdict_class"] == "blocked"
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for field, replacement in [("runtime_changed_score", 1), ("experiment_id", 8340)]:
        bad = deepcopy(value)
        bad[field] = replacement
        bad["reproducibility_checksum"] = q.checksum(bad)
        atomic_json(output, bad)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    work["historical_fixture_manifest"].append(dict(path="/absent", sha256="invalid"))
    atomic_json(Path(value["primitive_reference"]["path"]), work)
    bad = deepcopy(value)
    bad["primitive_reference"] = reference(Path(value["primitive_reference"]["path"]))
    bad["raw_shard_hashes"][0] = bad["primitive_reference"]
    bad["reproducibility_checksum"] = q.checksum(bad)
    atomic_json(output, bad)
    assert not q.replay(output)
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--date", "wrong").returncode == 2
    assert cli(tmp_path, "--private-run").returncode == 2


def test_frozen_plan(tmp_path):
    """SCENARIO-VERIFY-8353-AUTHORITY: controls use real versioned argv before work."""
    commands = e.manifest(tmp_path)
    assert all("tests/python" not in c["argv"] for c in commands)
    assert {"valid_old", "valid_current", "wrong_milestone", "corrupt_contract"} <= {
        c["name"] for c in commands
    }
    old = next(c for c in commands if c["name"] == "valid_old")
    assert "2026.10.717" in old["argv"]
    consumer = next(c for c in commands if c["name"] == "E2E018_runtime_consumers")
    assert "tests/python/test_runtime_change_boundary_8307.py" in consumer["argv"]
    assert "--fail-under=100" in next(c for c in commands if c["name"] == "coverage_json")["argv"]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_main_reader_before_measurement(tmp_path, monkeypatch, exit_code):
    """SCENARIO-REPORT-8353-REPLAY: real child failures precede runtime permission."""
    original = q.measure
    observed = []

    def plan(private):
        atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=100)))
        return [
            dict(
                name="private_child",
                argv=[e.PY, "-c", f"raise SystemExit({exit_code})"],
                deadline=15,
                expected=0,
                scope="owned",
            )
        ]

    def measure(root, raw, *, reader_checks_passed=True):
        observed.append(reader_checks_passed)
        return original(root, raw, reader_checks_passed=reader_checks_passed)

    monkeypatch.setattr(e, "manifest", plan)
    monkeypatch.setattr(q, "measure", measure)
    output = tmp_path / (q.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    assert observed == [exit_code == 0]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("blocked" if exit_code == 0 else "disqualified")
    assert q.replay(output)


def test_closure_missing_and_rejected_terminal(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8353-BOUNDARY: missing bound bytes never become zero evidence."""
    primary = tmp_path / q.UPSTREAM
    atomic_json(
        primary, dict(primitive_reference=dict(path=str(tmp_path / "absent"), sha256="invalid"))
    )
    with pytest.raises(FileNotFoundError):
        q.closure(tmp_path, tmp_path / "raw")
    monkeypatch.setattr(
        q.typed, "read_bound_sidecar", lambda primary, sidecar: dict(report=dict(passed=False))
    )
    with pytest.raises(ValueError, match="terminal_rejected"):
        q.closure(q.ROOT, tmp_path / "rejected")
