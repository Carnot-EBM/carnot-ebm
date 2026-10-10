"""REQ-REPORT-8368: private children and replay preserve the failed runtime boundary."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import runtime_closure_8368 as q
from carnot.verify import runtime_closure_execution_8368 as e


def cli(tmp_path, *args):
    """Fresh processes test dispatch and import coverage using real script bytes."""
    return subprocess.run(
        [str(q.ROOT / ".venv/bin/python"), "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_authority_and_manifest(tmp_path):
    """SCENARIO-VERIFY-8368-EXECUTION: complete historical and current authority are distinct."""
    for milestone in (q.MILESTONE, "2026.10.717"):
        root = q.private_authority(tmp_path / milestone, milestone)
        assert cli(tmp_path, "--authority", milestone, "--root", root).returncode == 0
        (root / "research-roadmap.yaml").write_text("milestone: wrong\ntasks: []\n")
        assert cli(tmp_path, "--authority", milestone, "--root", root).returncode == 1
    private = tmp_path / "plan"
    private.mkdir()
    plan = e.manifest(private)
    assert any(r["name"] == "E2E018_runtime_consumers" for r in plan)
    assert all("tests/python" not in r["argv"] for r in plan)
    assert not q.authority(tmp_path / "absent", tmp_path / "missing")["activated"]


def test_qualification_and_no_unchanged_retry(tmp_path):
    """SCENARIO-VERIFY-8368-TYPED: real identities cannot be reopened by a reader hash."""
    source = json.loads((q.ROOT / q.UPSTREAM).read_bytes())
    previous = json.loads(Path(source["runtime_binding_path"]).read_bytes())
    with (
        patch.object(q.old, "inventory", return_value=(previous, [])),
        patch.object(q.old, "probe_changed", side_effect=AssertionError("unchanged retry")),
    ):
        work = q.measure(q.ROOT, tmp_path / "raw")
    atomic_json(tmp_path / "raw/measurement.json", work)
    value = q.build(work, tmp_path / "raw", [dict(passed=True, scope="owned")])
    assert value["runtime_reader_ready_score"] == 1
    assert (
        value["runtime_changed_score"]
        == value["cuda_context_ready_score"]
        == value["current_probe_count"]
        == 0
    )
    assert value["historical_dispositions"][0]["verdict_class"] == "disqualified"
    assert value["typed_reference_rows"]
    with (
        patch.object(q.typed, "closure", side_effect=FileNotFoundError("real missing operand")),
        patch.object(q.old, "inventory", return_value=(previous, [])),
    ):
        blocked = q.measure(q.ROOT, tmp_path / "blocked")
    assert blocked["first_failed_operand"]


@pytest.mark.parametrize("exit_code", [0, 1])
def test_private_cli_and_rehashed_substitution(tmp_path, monkeypatch, exit_code):
    """SCENARIO-REPORT-8368-REPLAY: real failed children clear qualification."""

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

    monkeypatch.setattr(e, "manifest", plan)
    output = tmp_path / (q.NAME + ".json")
    assert (
        e.main(["--date", "20261010", "--root", str(tmp_path / "missing"), "--output", str(output)])
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("blocked" if exit_code == 0 else "disqualified")
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    altered_receipt = deepcopy(value)
    next(
        r
        for r in altered_receipt["validation_receipts"]
        if r["name"] == "legitimate_info_qualified"
    )["exit_code"] = 2
    altered_receipt["reproducibility_checksum"] = q.checksum(altered_receipt)
    atomic_json(output, altered_receipt)
    assert not q.replay(output)
    for field in (
        "runtime_changed_score",
        "experiment_id",
        "typed_reference_rows",
        "historical_dispositions",
    ):
        bad = deepcopy(value)
        bad[field] = 9
        bad["reproducibility_checksum"] = q.checksum(bad)
        atomic_json(output, bad)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    primitive = Path(value["primitive_reference"]["path"])
    work = json.loads(primitive.read_bytes())
    work["historical_fixture_manifest"].append(
        dict(path=str(tmp_path / "substitution"), sha256="sha256:" + "a" * 64)
    )
    atomic_json(primitive, work)
    bad = deepcopy(value)
    bad["primitive_reference"] = q.reference(primitive)
    bad["raw_shard_hashes"][0] = bad["primitive_reference"]
    bad["reproducibility_checksum"] = q.checksum(bad)
    atomic_json(output, bad)
    assert not q.replay(output)
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    assert cli(tmp_path, "--date", "wrong").returncode != 0
    assert cli(tmp_path, "--date").returncode != 0
    assert cli(tmp_path, "--private-run").returncode == 2


def test_private_run(tmp_path):
    """SCENARIO-VERIFY-8368-EXECUTION: a constructed private run grants no reader readiness."""
    output = tmp_path / (q.NAME + ".json")
    result = cli(tmp_path, "--private-run", "--root", tmp_path / "absent", "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cli(tmp_path, "--cold-replay", output).returncode == 0


def test_replay_primitive_controls(tmp_path):
    """SCENARIO-REPORT-8368-REPLAY: pinned source, authority and child meaning resist rehashing."""
    from carnot.reporting.v709_execution import child

    raw = tmp_path / "raw"
    previous = json.loads(
        Path(json.loads((q.ROOT / q.UPSTREAM).read_bytes())["runtime_binding_path"]).read_bytes()
    )
    with patch.object(q.old, "inventory", return_value=(previous, [])):
        work = q.measure(q.ROOT, raw)
    receipt = child("control", [e.PY, "-c", "print('{}')"], raw / "children", deadline=15)
    output = tmp_path / (q.NAME + ".json")

    def write(changed):
        atomic_json(raw / "measurement.json", changed)
        value = q.build(changed, raw, [receipt])
        atomic_json(output, value)
        return value

    write(work)
    assert q.replay(output)
    bad = deepcopy(work)
    bad["historical_dispositions"][0]["verdict_class"] = "positive"
    write(bad)
    assert not q.replay(output)
    bad = deepcopy(work)
    bad["historical_fixture_manifest"] = [
        r for r in bad["historical_fixture_manifest"] if r["path"] != str(q.ROOT / list(q.PINS)[0])
    ]
    bad["historical_dispositions"] = bad["historical_dispositions"][1:]
    write(bad)
    assert not q.replay(output)
    bad = deepcopy(work)
    bad["root"] = str(tmp_path / "relabeled")
    bad["historical_dispositions"] = []
    write(bad)
    assert not q.replay(output)
    bad.pop("baseline_reference")
    bad.pop("baseline_source_reference")
    write(bad)
    assert not q.replay(output)
    failed = child("failed", [e.PY, "-c", "raise SystemExit(2)"], raw / "children", deadline=15)
    false_pass = dict(failed, passed=True)
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, q.build(work, raw, [false_pass]))
    assert not q.replay(output)
    for field in ("previous", "current"):
        bad = deepcopy(work)
        bad[field]["driver"] = "constructed corruption"
        write(bad)
        assert not q.replay(output)
    bad = deepcopy(work)
    bad["authority"]["tasks"] = []
    write(bad)
    assert not q.replay(output)
    bad = deepcopy(work)
    original = next(
        r for r in bad["historical_fixture_manifest"] if r["path"] == str(q.ROOT / q.UPSTREAM)
    )
    substitution = tmp_path / "substitution.json"
    atomic_json(substitution, dict(constructed=True))
    original.update(snapshot_path=str(substitution), sha256=q.reference(substitution)["sha256"])
    write(bad)
    assert not q.replay(output)
    for devices in (
        [dict(index="0", uuid="control", name="control", driver_version="control")],
        [],
    ):
        bad = deepcopy(work)
        observation = child(
            "inventory",
            [e.PY, "-c", "print('0,control,control,control')"],
            raw / "inventory",
            deadline=15,
        )
        bad["observation_receipts"] = [observation]
        bad["current"]["devices"] = devices
        binding = tmp_path / "constructed_binding.json"
        atomic_json(binding, bad["current"])
        bad["current_reference"] = q.reference(binding)
        write(bad)
        assert q.replay(output) == bool(devices)
    bad = deepcopy(work)
    bad["diagnostic"]["rows"] = [
        dict(layer="driver", binding="control", primitive={}, receipt=receipt)
    ]
    write(bad)
    assert q.replay(output)
    bad["diagnostic"]["rows"][0]["primitive"] = dict(constructed="different bytes")
    write(bad)
    assert not q.replay(output)
    with (
        patch.object(q, "sha256_file", return_value="sha256:" + "a" * 64),
        patch.object(q.old, "inventory", return_value=(previous, [])),
    ):
        assert q.measure(q.ROOT, tmp_path / "substituted")["first_failed_operand"]
