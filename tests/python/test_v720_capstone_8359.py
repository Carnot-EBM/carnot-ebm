"""REQ-REPORT-8359 / REQ-VERIFY-8359: bound every V720 disposition to bytes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pytest

from carnot.reporting import v720_capstone_evidence as e
from carnot.reporting import v720_capstone as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v710_contract_replay import snapshot


def fixture(root):
    """Private authority copies authorize mechanics, without inventing science."""
    for name in [e.DESIGN, e.STAGED, e.ACTIVE, e.PROTOCOL]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        source = e.ROOT / name
        path.write_bytes((source if source.is_file() else e.ROOT / e.ACTIVE).read_bytes())
    return root


def cli(parent, *args):
    """A real process tests script imports, dispatch and bounded normal exit."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def test_private_cli_and_rehashed_controls(tmp_path):
    """SCENARIO-VERIFY-8359-REPLAY: missing science cannot become a null."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli(
        tmp_path, "--date", "20261009", "--root", root, "--output", output, "--private-fixture"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8346, 8360))
    assert value["missing_output_count"] == 13
    assert all(r["honest_verdict"] is None for r in value["rows"][:-1])
    assert value["verdict_class"] == "blocked"
    assert value["H1"]["intended_count"] == 128
    assert value["H2"]["retention_windows"] == [0, 32, 64, 96]
    assert value["science_ready_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["memory_bounds"]["growth_mb"] == 500
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for key in ["rows", "validation_receipts", "paper_ready", "H1"]:
        changed = deepcopy(value)
        if key == "rows":
            changed[key][-1]["verdict_class"] = "positive"
        elif key == "validation_receipts":
            changed[key][0]["passed"] = False
        elif key == "H1":
            changed[key]["intended_count"] = 96
        else:
            changed[key] = not changed[key]
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    assert not e.replay(tmp_path / "absent")
    assert cli(tmp_path, "--date", "wrong").returncode == 2


def test_owned_failure_and_primitive_tamper(tmp_path):
    """SCENARIO-REPORT-8359-ACCOUNTING: current failures disqualify."""
    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    value = e.build(work, [], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["tasks"][0]["title"] = "foreign"
    with pytest.raises(ValueError, match="contract"):
        e.reduce(changed, [])
    changed = deepcopy(work)
    changed["inputs"][0]["summary"]["row"]["honest_verdict"] = "complete_null_fake"
    with pytest.raises(ValueError, match="primitive"):
        e.reduce(changed, [])


def test_manifest_and_preflight(tmp_path):
    """REQ-VERIFY-8359: freeze owned coverage and private E2Es before work."""
    plan = runner.manifest(tmp_path)
    assert {p["name"] for p in plan} >= {"private_E2E018", "private_E2E021"}
    assert (
        str(e.ROOT / e.CLI) in (tmp_path / "coverage.ini").read_text()
        or e.CLI in (tmp_path / "coverage.ini").read_text()
    )
    assert runner.preflight([dict(argv=["/absent/tool"])])
    assert all(
        p["argv"][-1].startswith("--basetemp=/tmp/exp8359-")
        for p in plan
        if p["name"] in {"private_E2E018", "private_E2E021", "unchanged_consumers"}
    )


def test_current_fourteen_and_science(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8359-ACCOUNTING: sealed diagnostics preserve failures."""
    execute = e.execute
    monkeypatch.setattr(
        e,
        "execute",
        lambda plan, logs: execute([p for p in plan if p["name"] == "publication_gate"], logs),
    )
    work = e.measure(e.ROOT, tmp_path / "actual")
    value = e.build(
        work, [dict(passed=True, scope="owned")], tmp_path / "actual", tmp_path / (e.NAME + ".json")
    )
    assert len(value["rows"]) == 14
    assert (
        value["actual_executed_task_count"]
        + value["pre_gate_count"]
        + value["missing_output_count"]
        == 14
    )
    assert value["rows"][4]["verdict_class"] == "disqualified"
    assert value["rows"][5]["verdict_class"] == "disqualified"
    assert value["H1"]["qualified_count"] == 97
    assert value["H2"]["qualified_count"] == 67
    assert value["H1"]["status"] == "measured_unqualified_diagnostic"
    assert value["H2"]["status"] == "measured_unqualified_diagnostic"
    assert len(value["H2"]["retention_window_bounds"]) == 4
    assert value["science_ready_score"] == 0
    assert value["memory_measurements"]["retained_payload_bytes"] < 8_000_000
    assert value["static_ready_score"] == 1
    assert len(value["three_prd_gaps"]) == 3
    assert all(not g["closed"] for g in value["three_prd_gaps"])
    changed = deepcopy(work)
    changed["memory_measurements"]["parent_growth_mb"] = 501
    assert (
        e.reduce(changed, [dict(passed=True, scope="owned")])["capstone_execution_ready_score"] == 0
    )
    changed = deepcopy(work)
    changed["inputs"][4]["summary"]["selected"]["independent_reduction"]["qualified_count"] = 128
    changed["inputs"][4]["summary_sha256"] = canonical_hash(changed["inputs"][4]["summary"])
    candidate = e.build(
        changed,
        [dict(passed=True, scope="owned")],
        tmp_path / "tamper",
        tmp_path / (e.NAME + ".json"),
    )
    path = tmp_path / "tampered.json"
    atomic_json(path, candidate)
    assert not e.replay(path)


def test_worker_and_invalid_authority(tmp_path):
    """SCENARIO-VERIFY-8359-REPLAY: worker dispatch and authority failures are real."""
    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    request, output = tmp_path / "request.json", tmp_path / "worker.json"
    atomic_json(request, dict(task=work["tasks"][0], item=work["inputs"][0]))
    assert cli(tmp_path, "--worker-request", request, "--worker-output", output).returncode == 0
    assert json.loads(output.read_bytes())["row"]["missing"]
    changed = deepcopy(work)
    changed["references"][0] = snapshot(tmp_path / "absent", tmp_path / "ref", "absent")
    assert not e.authority(changed)["activated"]
    changed = deepcopy(work)
    changed["references"][0]["sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="hash_drift"):
        e.reduce(changed, [])


def test_private_failed_child_publication(tmp_path):
    """SCENARIO-VERIFY-8359-REPLAY: an owned child failure cannot earn readiness."""
    from carnot.reporting.v709_execution import child
    from carnot.reporting import v720_replay_execution as terminal

    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    failed = child(
        "deliberate_failure",
        [
            sys.executable,
            "-u",
            "-c",
            "print('deliberate failed check', flush=True); raise SystemExit(7)",
        ],
        tmp_path / "failed",
        expected=0,
        deadline=5,
    )
    value = e.build(work, [failed], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    runner.publish(value, tmp_path / (e.NAME + ".json"), tmp_path / "raw")
    assert (
        json.loads((tmp_path / (e.NAME + ".json")).read_bytes())["verdict_class"] == "disqualified"
    )
    with TemporaryDirectory(prefix="exp8359-mechanical-", dir="/tmp") as directory:
        controls = terminal.operand_controls(Path(directory), tmp_path / "controls")
    assert len(controls) == 12
    assert all(r["passed"] for r in controls)


def test_default_dispatch_and_date(tmp_path, monkeypatch):
    """REQ-VERIFY-8359: default identity and rejected dates cannot escape dispatch."""
    called = []
    monkeypatch.setattr(runner.runner, "main", lambda args: 0)
    monkeypatch.setattr(runner, "write_note", lambda p: called.append(p))
    monkeypatch.setattr(e, "ROOT", tmp_path)
    assert runner.main([]) == 0
    assert called == [tmp_path / "results" / (e.NAME + ".json")]
    assert runner.main(["--date"]) == 2


def test_wrong_task_range(tmp_path):
    """SCENARIO-REPORT-8359-ACCOUNTING: exactly the fourteen authorized slots exist."""
    root = fixture(tmp_path / "root")
    design = root / e.DESIGN
    design.write_text(design.read_text().replace("exp8346", "exp9346"))
    with pytest.raises(ValueError, match="exact_fourteen_task_contract"):
        e.measure(root, tmp_path / "raw")


def test_provenance_shapes():
    """REQ-REPORT-8359: compact mixed historical formats without losing call counts."""
    value = [
        [dict(imported_counts=dict(generate=211), model_receipt=dict(log="long"))],
        "inherited pin only",
    ]
    assert e.compact_provenance(value) == [
        [dict(imported_counts=dict(generate=211))],
        "inherited pin only",
    ]
    assert value[0][0]["model_receipt"] == dict(log="long")
