"""REQ-REPORT-8331 / REQ-VERIFY-8331: preserve missing science and stable replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v718_capstone as runner
from carnot.reporting import v718_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(root):
    root.mkdir()
    for name in [e.DESIGN, e.ACTIVE, e.PROTOCOL]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    return root


def cli(parent, *args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_absent_science(tmp_path):
    """SCENARIO-REPORT-8331-SCIENCE: absent units keep every intended denominator."""
    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    value = e.build(
        work, [dict(passed=True, scope="owned")], tmp_path / "raw", tmp_path / (e.NAME + ".json")
    )
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8318, 8332))
    assert value["missing_output_count"] == 13
    assert all(r["honest_verdict"] is None for r in value["rows"][:-1])
    assert value["verdict_class"] == "blocked"
    assert value["H1"]["intended_count"] == 128
    assert value["H2"]["intended_count"] == 88
    assert value["H2"]["retention_windows"] == [0, 32, 64, 96]
    assert value["H1"]["statistics"] is value["H2"]["statistics"] is None
    assert value["capacity_scope"]["included_in_H1_H2"] is False
    assert value["science_ready_score"] == 0
    assert value["capstone_execution_ready_score"] == 1
    assert (
        e.build(work, [], tmp_path / "raw", tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_private_cli_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8331-CLI: final candidate, self and validation are replayed."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli(
        tmp_path, "--date", "20261009", "--root", root, "--output", output, "--private-fixture"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert value["run_date"] == "20261009"
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert (root / "docs/research-notes/v718-outcomes.md").exists()
    for field in ["self", "validation", "checksum", "gate", "model"]:
        changed = deepcopy(value)
        if field == "self":
            changed["rows"][-1]["verdict_class"] = "positive"
        elif field == "validation":
            changed["validation_receipts"][0]["passed"] = False
        elif field == "gate":
            changed["paper_ready"] = not changed["paper_ready"]
        elif field == "model":
            changed["MODEL_SPECS"] = [{}]
        changed["reproducibility_checksum"] = (
            canonical_hash({k: v for k, v in changed.items() if k != "reproducibility_checksum"})
            if field != "checksum"
            else "invalid"
        )
        path = tmp_path / (field + ".json")
        atomic_json(path, changed)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--date", "20260101").returncode == 2
    assert not e.replay(tmp_path / "absent")


def test_manifest(tmp_path):
    """REQ-VERIFY-8331: scoped coverage and private E2Es freeze before measurement."""
    plan = runner.manifest(tmp_path)
    assert {p["name"] for p in plan} >= {"private_E2E021", "private_E2E018_consumers"}
    assert "patch=subprocess" in (tmp_path / "coverage.ini").read_text()
    assert all("tests/python" not in p["argv"] for p in plan)


def test_actual_mixed_history(tmp_path):
    """SCENARIO-REPORT-8331-ACCOUNTING: current and historical failures stay distinct."""
    if os.environ.get("CARNOT_8331_MIXED_CHILD") != "1":
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['test_actual_mixed_history'](Path(sys.argv[2]))",
                str(Path(__file__).absolute()),
                str(tmp_path),
            ],
            env=dict(os.environ, CARNOT_8331_MIXED_CHILD="1"),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    work = e.measure(e.ROOT, tmp_path / "actual")
    value = e.build(
        work, [dict(passed=True, scope="owned")], tmp_path / "actual", tmp_path / (e.NAME + ".json")
    )
    assert value["actual_executed_task_count"] == 5
    assert value["missing_output_count"] == 6
    assert value["pre_gate_count"] == 3
    assert value["historical_v717"]["actual_executed_task_count"] == 8
    assert value["historical_v717"]["pre_gate_count"] == 2
    assert value["historical_v717"]["missing_output_count"] == 4
    assert any(r["disposition"] == "authenticated_upstream_failure" for r in value["rows"])
    current_passed = all(r["passed"] for r in value["branch_replay_receipts"])
    assert value["capstone_execution_ready_score"] == int(current_passed)
    assert value["verdict_class"] == ("blocked" if current_passed else "disqualified")
    assert value["science_ready_score"] == 0
    assert any(r["same_verdict_entries"] for r in value["retirements"])
    changed = deepcopy(work)
    changed["tasks"][0]["title"] = "changed"
    from carnot.reporting.v718_capstone_reduction import reduce

    with pytest.raises(ValueError, match="contract"):
        reduce(changed, [])
    changed = deepcopy(work)
    changed["inputs"][0]["reference"] = dict(changed["inputs"][0]["reference"], path="foreign")
    with pytest.raises(ValueError, match="input_reference"):
        reduce(changed, [])
    changed = deepcopy(work)
    next(r for r in changed["references"] if "expected_sha256" in r)["expected_sha256"] = (
        "sha256:changed"
    )
    assert any(
        g["artifact_field"] == "primitive_sha256"
        for g in reduce(changed, [dict(passed=True, scope="owned")])["gate_check_summary"]
    )
    changed = deepcopy(work)
    upstream = e.old.read(changed["inputs"][0]["reference"])
    ref = next(r for r in changed["references"] if r["path"] == upstream["work_reference"]["path"])
    primitive = e.old.read(ref)
    primitive["history"]["reduction_sha256"] = "sha256:wrong"
    path = tmp_path / "historical_tamper.json"
    atomic_json(path, primitive)
    ref.update(snapshot_path=str(path), sha256=e.sha256_file(path))
    with pytest.raises(ValueError, match="historical_reduction_drift"):
        reduce(changed, [])


def test_absent_authority_pregate_and_primitive_error(tmp_path):
    """SCENARIO-REPORT-8331-ACCOUNTING: gate receipts never invent producer verdicts."""
    root = fixture(tmp_path / "root")
    tasks = e.parse_design((root / e.DESIGN).read_text(), milestone=e.MILESTONE)[1]
    task = tasks[2]
    gate = task["gated_on"][0]
    source = root / tasks[0]["deliverable"]
    source.parent.mkdir()
    atomic_json(source, {})
    path = root / "results/experiment_8320_conductor_gate.json"
    bound = dict(
        upstream=gate["upstream"],
        artifact_field=gate["artifact_field"],
        op=gate["op"],
        expected=gate["value"],
        actual=None,
        passed=False,
        artifact_path=str(source),
        artifact_sha256=e.sha256_file(source),
    )
    atomic_json(
        path,
        dict(
            experiment=8320,
            task_id=task["id"],
            schema="blocked_gate_check_v1",
            blocked_at_layer="conductor_pre_gate",
            gates_evaluated=[bound],
        ),
    )
    work = e.measure(root, tmp_path / "pre_gate")
    from carnot.reporting.v718_capstone_reduction import reduce

    value = reduce(work, [dict(passed=True, scope="owned")])
    assert value["pre_gate_count"] == 1
    assert value["rows"][2]["honest_verdict"] is None
    changed = deepcopy(work)
    ref = changed["inputs"][2]["reference"]
    body = e.old.read(ref)
    body["gates_evaluated"][0]["artifact_sha256"] = "sha256:wrong"
    corrupt = tmp_path / "bad_gate.json"
    atomic_json(corrupt, body)
    ref.update(snapshot_path=str(corrupt), sha256=e.sha256_file(corrupt))
    assert (
        reduce(changed, [dict(passed=True, scope="owned")])["rows"][2]["disposition"]
        == "unbound_pre_gate"
    )
    (root / e.DESIGN).unlink()
    (root / e.ACTIVE).unlink()
    absent = e.measure(root, tmp_path / "absent")
    assert reduce(absent, [dict(passed=True, scope="owned")])["capstone_execution_ready_score"] == 0
    (root / e.DESIGN).write_text(
        (e.ROOT / e.DESIGN).read_text().replace("exp8318-contract-replay", "exp9999-wrong")
    )
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "wrong")


def test_owned_failure_recovery_and_scratch(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8331-CLI: retain real child failure and honest recovery."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    good_manifest = runner.manifest

    def failed_manifest(private):
        return [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-u", "-c", "raise SystemExit(1)"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ]

    original = runner.qualified.audit
    calls = []

    def reject_once(candidate, logs, proof):
        report = original(candidate, logs, proof)
        calls.append(candidate)
        if len(calls) == 1:
            report["passed"] = report["receipt"]["passed"] = False
        return report

    monkeypatch.setattr(runner, "manifest", failed_manifest)
    monkeypatch.setattr(runner.qualified, "audit", reject_once)
    assert runner.main(["--date", "20261009", "--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    assert e.replay(output)
    assert len(calls) == 2
    original_read = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: b"wrong" if p.name == "write_probe" else original_read(p)
    )
    with pytest.raises(ValueError, match="private_scratch"):
        good_manifest(tmp_path)


def test_protocol_history_and_resource_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8331: protocol and resource failure cannot gain readiness."""
    from carnot.reporting.v718_capstone_reduction import reduce

    root = fixture(tmp_path / "root")
    (root / e.PROTOCOL).write_text("{}")
    work = e.measure(root, tmp_path / "bad_protocol")
    assert reduce(work, [dict(passed=True, scope="owned")])["capstone_execution_ready_score"] == 0
    assert any(
        g["artifact_field"] == "protocol_sha256"
        for g in reduce(work, [dict(passed=True, scope="owned")])["gate_check_summary"]
    )
    assert runner.main(["--date"]) == 2
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: type("Disk", (), dict(free=0))())
    assert runner.preflight([])[0]["artifact_field"] == "private_scratch_capacity_bytes"
    assert (
        runner.preflight([dict(argv=["/absent/tool"])])[0]["artifact_field"]
        == "executable_available"
    )


def test_post_publication_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8331-CLI: a failing published replay never returns success."""
    root = fixture(tmp_path / "root")
    monkeypatch.setattr(e, "ROOT", root)
    # The real private CLI is qualified by the preceding tests. This isolates
    # the adapter's final supervision response from the existing publisher.
    monkeypatch.setattr(runner.runner, "main", lambda args: 0)
    monkeypatch.setattr(runner, "child", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="published_reduction_drift"):
        runner.main([])


def test_missing_tool_closes_blocked(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8331-CLI: missing external tools close without fake checks."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(
        runner,
        "manifest",
        lambda private: [
            dict(name="absent_tool", argv=["/absent/tool"], scope="owned", expected=0, deadline=10)
        ],
    )
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_tool"
    assert value["capstone_execution_ready_score"] == 0
    assert e.replay(output)
