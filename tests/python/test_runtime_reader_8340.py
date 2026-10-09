"""REQ-VERIFY-8340 / REQ-REPORT-8340: reader success cannot imply CUDA health."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import yaml

from carnot.verify import runtime_reader_8340 as q
from carnot.verify import runtime_reader_execution_8340 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import v718_replay_history as history


def cli(tmp_path, *args):
    """Real children inherit private coverage but never borrow live output paths."""
    return subprocess.run(
        [str(q.ROOT / ".venv/bin/python"), "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )


def authorities(tmp_path, milestone):
    """Copy complete versioned design bytes and construct matching private activation."""
    root = tmp_path / milestone
    design = q.design(root, milestone)
    design.parent.mkdir(parents=True)
    design.write_bytes(q.design(q.ROOT, milestone).read_bytes())
    tasks = q.parse_design(design.read_text(), milestone=milestone)[1]
    (root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(dict(milestone=milestone, tasks=tasks))
    )
    return root


@pytest.mark.parametrize("milestone", [q.MILESTONE, "2026.10.716"])
def test_authority_children(tmp_path, milestone):
    """SCENARIO-VERIFY-8340-AUTHORITY: actual children qualify full old/current contracts."""
    root = authorities(tmp_path, milestone)
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
    assert not q.authority(tmp_path / "missing", tmp_path / "raw", milestone)["activated"]


def data(tmp_path):
    """Only mechanics use constructed runtime bytes, with no live readiness claim."""
    from test_runtime_change_boundary_8307 import fixture

    return fixture(tmp_path)


def test_change_requires_identity(tmp_path):
    """SCENARIO-VERIFY-8340-BOUNDARY: clocks and renamed paths never reopen CUDA."""
    d = data(tmp_path)
    d["current"]["date"] = "20261009"
    d["current"]["libraries"][0]["path"] = "/renamed/library"
    assert not any(r["changed"] for r in q.changes(d))
    assert q.reduce(d, True)["honest_verdict"] == "complete_blocked_cuda_environment_unchanged"
    d["current"]["driver"] = "new-driver"
    assert q.reduce(d, True)["runtime_changed_score"] == 1
    assert q.reduce(d, True)["cuda_context_ready_score"] == 0
    d["current"]["driver"] = None
    assert not any(r["changed"] for r in q.changes(d))
    assert q.reduce(d, False)["verdict_class"] == "disqualified"
    d["checks"] = [q.old.gate(tmp_path / "missing", "exists", True, False)]
    assert q.reduce(d, True)["verdict_class"] == "blocked"


def test_preserved_historical_binding(tmp_path):
    """SCENARIO-VERIFY-8340-AUTHORITY: all old assertions consume preserved V716."""
    from carnot.verify import runtime_localization_8290 as historical

    assert historical.DESIGN == history.OLD_DESIGN
    from test_runtime_localization_8290 import test_authority

    test_authority(tmp_path)


def test_finding_consumer_fail_closed(tmp_path):
    """SCENARIO-VERIFY-8340-REPLAY: all unknown/error/warning reports block readiness."""
    path = tmp_path / "candidate.json"
    atomic_json(path, {})
    report = dict(
        candidate_sha256=sha256_file(path),
        verifier_sha256=history.verifier_hash(),
        reports=[dict(loaded=True, artifact=str(path), flag_count=0, flags=[])],
    )
    assert history.consume(report, path, 0, {})["passed"]
    for exit_code in [1, 2, -9]:
        assert not history.consume(report, path, exit_code, {})["passed"]
    for severity, kind in [
        ("warn", "IMPLAUSIBLE_PERFECT"),
        ("info", "unknown"),
        ("critical", "IMPLAUSIBLE_PERFECT"),
    ]:
        bad = deepcopy(report)
        bad["reports"][0].update(flag_count=1, flags=[dict(severity=severity, kind=kind)])
        assert not history.consume(bad, path, 1, {})["passed"]
    for bad in [
        {},
        dict(report, reports="malformed"),
        dict(report, candidate_sha256="wrong"),
        dict(report, reports=[dict(flags=["malformed"])]),
    ]:
        assert not history.consume(bad, path, 0, {})["passed"]


def test_private_cli_replay_and_recovery(tmp_path):
    """SCENARIO-REPORT-8340-PUBLISH: real cold children reject rehashed score tampering."""
    output = tmp_path / (q.NAME + ".json")
    run = cli(tmp_path, "--private-run", "--root", tmp_path / "missing", "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["MODEL_SPECS"] == [] and value["runtime_reader_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["runtime_changed_score"] = 1
    value["reproducibility_checksum"] = q.checksum(value)
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    assert cli(tmp_path, "--date", "wrong").returncode == 2
    assert cli(tmp_path, "--private-run").returncode == 2


def test_measure_live_identity_and_changed_gate(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8340-BOUNDARY: authenticated Exp8307 failure remains usable."""
    current = json.loads(
        Path(json.loads((q.ROOT / q.UPSTREAM).read_bytes())["runtime_binding_path"]).read_bytes()
    )
    monkeypatch.setattr(q.old, "inventory", lambda previous, raw: (deepcopy(current), []))
    d = q.measure(q.ROOT, tmp_path / "same")
    assert not any(row["changed"] for row in q.changes(d))
    assert d["historical"]["verdict_class"] == "disqualified"
    assert all(c.get("passed", False) for c in d["checks"]), d["checks"]
    current["driver"] = "private-changed-driver"
    probe = dict(rows=[], checks=[], lease_receipt={}, cleanup_receipt={})
    with patch.object(q.old, "probe_changed", return_value=probe) as run:
        changed = q.measure(q.ROOT, tmp_path / "changed")
    assert run.call_count == 1 and q.reduce(changed, True)["runtime_changed_score"] == 1
    with patch.object(q, "checked", side_effect=ValueError("corrupt source")):
        bad = q.measure(q.ROOT, tmp_path / "bad")
    assert any(c.get("field") == "runtime_identity_custody" for c in bad["checks"])
    terminal = tmp_path / "bad_terminal.json"
    atomic_json(terminal, dict(normal_process_exit=False, report=dict(passed=False)))
    with patch.object(
        q, "authenticate", return_value=dict(terminal_validation_sidecar_path=str(terminal))
    ):
        assert (
            q.measure(q.ROOT, tmp_path / "badterminal")["checks"][-2]["field"]
            == "runtime_identity_custody"
        )

    raw = tmp_path / "same"
    stream = tmp_path / "validation.stdout"
    stream.write_text("actual receipt")
    receipt = dict(
        passed=True,
        stdout_path=str(stream),
        stderr_path=str(stream),
        stdout_sha256=sha256_file(stream),
        stderr_sha256=sha256_file(stream),
    )

    def candidate(work):
        atomic_json(raw / "measurement.json", work)
        path = tmp_path / (q.NAME + ".json")
        atomic_json(path, q.build(work, raw, [receipt]))
        return path

    assert q.replay(candidate(d))
    for mutate in [
        lambda w: w["previous"].update(driver="invented"),
        lambda w: w["current"].update(driver="invented"),
        lambda w: w["authority"]["tasks"][0].update(prompt="invented"),
    ]:
        wrong = deepcopy(d)
        mutate(wrong)
        assert not q.replay(candidate(wrong))


def test_manifest_live_driver_and_failure_recovery(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8340-PUBLISH: real validators retain a failed owned check."""
    private = tmp_path / "plan"
    private.mkdir()
    plan = e.manifest(private)
    assert {p["name"] for p in plan} >= {
        "owned_coverage",
        "coverage_json",
        "valid_old",
        "valid_current",
    }
    assert all(p["deadline"] <= 360 for p in plan)
    output = tmp_path / (q.NAME + ".json")

    def tiny_manifest(scratch):
        atomic_json(scratch / "coverage.json", dict(totals=dict(percent_covered=100)))
        return []

    monkeypatch.setattr(e, "manifest", tiny_manifest)
    original = e.qualified.check
    failed = []

    def reject_once(spec, logs):
        receipt = original(spec, logs)
        if spec["name"] == "cold_replay" and not failed:
            failed.append(True)
            receipt["passed"] = False
        return receipt

    monkeypatch.setattr(e.qualified, "check", reject_once)
    assert e.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and not value["required_checks_passed"]
    assert value["runtime_reader_ready_score"] == value["cuda_context_ready_score"] == 0
    assert q.replay(output)
    for key, operand in [
        ("experiment_id", 0),
        ("MODEL_SPECS", [{}]),
        ("required_checks_passed", True),
    ]:
        wrong = deepcopy(value)
        wrong[key] = operand
        wrong["reproducibility_checksum"] = q.checksum(wrong)
        atomic_json(output, wrong)
        assert not q.replay(output)


def test_inventory_and_primitive_replay_bindings(tmp_path):
    """SCENARIO-VERIFY-8340-REPLAY: changing raw identity or stream custody is rejected."""
    raw = tmp_path / "raw"
    raw.mkdir()
    d = data(tmp_path)
    d.update(
        authority=dict(activated=False),
        refs=[],
        current_reference=q.reference(tmp_path / "operand"),
    )
    binding = raw / "binding.json"
    atomic_json(binding, d["current"])
    d["current_reference"] = q.reference(binding)
    stream = raw / "stream"
    stream.write_text("private")
    receipt = dict(
        passed=True,
        scope="owned",
        stdout_path=str(stream),
        stderr_path=str(stream),
        stdout_sha256=sha256_file(stream),
        stderr_sha256=sha256_file(stream),
    )

    def candidate():
        atomic_json(raw / "measurement.json", d)
        value = q.build(d, raw, [receipt])
        path = tmp_path / (q.NAME + ".json")
        atomic_json(path, value)
        return path

    assert q.replay(candidate())
    d["observation_receipts"] = [receipt]
    assert not q.replay(candidate())
    stream.write_text("0, GPU-abc, private, driver-a\n")
    receipt.update(stdout_sha256=sha256_file(stream), stderr_sha256=sha256_file(stream))
    d["current"]["devices"] = [
        dict(index="0", uuid="GPU-abc", name="private", driver_version="driver-a")
    ]
    atomic_json(binding, d["current"])
    d["current_reference"] = q.reference(binding)
    assert q.replay(candidate())
    d["diagnostic"]["rows"] = [
        dict(layer="driver", binding="explicit_uuid", primitive={}, receipt=receipt)
    ]
    assert not q.replay(candidate())
    stream.write_text("{}\n")
    receipt.update(stdout_sha256=sha256_file(stream), stderr_sha256=sha256_file(stream))
    d["observation_receipts"] = []
    assert q.replay(candidate())
    d["diagnostic"]["rows"][0]["primitive"] = dict(invented=True)
    assert not q.replay(candidate())
    d["diagnostic"]["rows"][0]["primitive"] = {}
    receipt["stdout_sha256"] = "wrong"
    assert not q.replay(candidate())
