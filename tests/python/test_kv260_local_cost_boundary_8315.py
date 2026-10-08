"""REQ-REPORT-8315 and REQ-VERIFY-8315: private dependency and replay gates."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import kv260_local_cost_boundary_8315 as e
from carnot.reporting import kv260_local_cost_execution_8315 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary


def source(root: Path, name: str, **fields: object) -> Path:
    """Use the real publisher so fixture sidecars follow the production contract."""
    identity = int(name.split("_")[1])
    path = root / "results" / (name + ".json")
    value = dict(
        experiment_id=identity,
        task_id=f"exp{identity}-fixture",
        honest_verdict="complete_null_private",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        raw_shard_hashes=[],
        **fields,
    )
    terminal = path.parent / "raw" / path.stem / "terminal.json"
    value["terminal_validation_sidecar_path"] = str(terminal)
    publication = publish_primary(path, value, lambda _: dict(passed=True))
    atomic_json(terminal, dict(publication=publication, normal_process_exit=True))
    return path


def test_authentication(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8315-BOUNDARY: readiness never replaces byte custody."""
    work = dict(checks=[], refs=[])
    absent = tmp_path / "results" / (e.SOURCES["natural"][0] + ".json")
    assert e.probe(absent, tmp_path / "raw", work, "trajectory_ready_score") is None
    path = source(tmp_path, e.SOURCES["fixture"][0], local_kernel_ready_score=1)
    assert e.probe(path, tmp_path / "raw", work, "local_kernel_ready_score") is not None
    value = json.loads(path.read_text())
    shard = tmp_path / "primitive.json"
    atomic_json(shard, dict(event=1))
    value["raw_shard_hashes"] = [dict(path=str(shard), sha256=sha256_file(shard))]
    terminal = Path(value["terminal_validation_sidecar_path"])
    publication = publish_primary(path, value, lambda _: dict(passed=True))
    atomic_json(terminal, dict(publication=publication))
    assert e.probe(path, tmp_path / "raw", work, "local_kernel_ready_score") is not None
    shard.write_text("{}")
    assert e.probe(path, tmp_path / "raw", work, "local_kernel_ready_score") is None
    assert not work["checks"][-1]["passed"]
    path.write_text("invalid")
    assert e.probe(path, tmp_path / "raw", work, "local_kernel_ready_score") is None


def test_block_and_recovery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8315-BLOCKED: receipts determine owned readiness."""
    historical = source(tmp_path, e.SOURCES["hardware"][0], kv260_obligation=dict(k_max=5))
    work = e.measure(tmp_path, tmp_path / "raw")
    assert work["hardware"]["kv260_obligation"]["k_max"] == 5
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert value["verdict_class"] == "blocked"
    assert value["complete_cost_rows"] == []
    assert value["compatible_fraction"] is None
    assert len(value["rows"]) == 4
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    assert e.build(work, tmp_path / "raw", [])["required_checks_passed"] is False
    with pytest.raises(ValueError, match="qualified_operands"):
        e.build(dict(work, checks=[dict(passed=True)]), tmp_path / "raw", [dict(passed=True)])
    log = tmp_path / "stdout.log"
    log.write_text("real private output")
    value = e.build(
        work,
        tmp_path / "raw",
        [dict(passed=True, stdout_path=str(log), stdout_sha256=sha256_file(log))],
    )
    atomic_json(candidate, value)
    assert e.replay(candidate)
    log.write_text("changed")
    assert not e.replay(candidate)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    value["cpu_cost_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    atomic_json(candidate, {})
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "absent")
    historical.write_text("{}")
    assert not e.replay(candidate)


def test_direct_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8315-BLOCKED: the actual CLI owns failure and replay."""
    output = tmp_path / "private" / (e.NAME + ".json")
    argv = ["--root", str(tmp_path / "absent"), "--fixture-output", str(output)]
    assert runner.main(argv) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert (
        runner.main(["--root", str(tmp_path), "--worker-output", str(tmp_path / "work.json")]) == 0
    )
    with pytest.raises(SystemExit, match="2"):
        runner.main(["--date", "20000101"])
    with pytest.raises(SystemExit, match="2"):
        runner.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    args = [sys.executable, str(e.ROOT / e.CLI), "--cold-replay", str(output)]
    if "COVERAGE_RCFILE" in os.environ:
        args[1:1] = ["-m", "coverage", "run"]
    child = subprocess.run(args, cwd=tmp_path, capture_output=True, timeout=30)
    assert child.returncode == 0
    assert b"replay_passed" in child.stdout
    value = json.loads(output.read_text())
    value["current_device_execution_count"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    child = subprocess.run(args, cwd=tmp_path, capture_output=True, timeout=30)
    assert child.returncode == 1


def test_manifest_and_failed_checks(tmp_path: Path) -> None:
    """REQ-REPORT-8315: frozen argv includes private costs and exact validators."""
    specs = runner.manifest(tmp_path, tmp_path / "candidate.json")
    assert "repository_health" not in specs
    assert specs["commands"][-1]["argv"][2] == "--files"
    assert specs["commands"][0]["deadline_s"] <= 480
    with patch.object(runner.qualified, "main", return_value=0):
        assert runner.main([]) == 0


def test_copy_race_and_transcript(tmp_path: Path) -> None:
    """REQ-VERIFY-8315: a changed primitive cannot become authenticated history."""
    path = tmp_path / "transcript.json"
    atomic_json(path, dict(kv260_overlay_loaded="historical_overlay"))
    with patch.object(e, "sha256_file", side_effect=["sha256:first", "sha256:changed"]):
        with pytest.raises(ValueError, match="source_changed"):
            e.pin(path, tmp_path / "race", [])
    source(
        tmp_path,
        e.SOURCES["hardware"][0],
        kv260_obligation=dict(
            historical=dict(source_transcript=str(path), source_transcript_sha256=sha256_file(path))
        ),
    )
    work = e.measure(tmp_path, tmp_path / "valid")
    assert work["hardware_transcript"]["kv260_overlay_loaded"] == "historical_overlay"
    path.write_text("{}")
    work = e.measure(tmp_path, tmp_path / "invalid")
    assert work["hardware"] == {}
    assert not work["checks"][-1]["passed"]
