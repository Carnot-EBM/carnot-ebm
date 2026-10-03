"""REQ-REPORT-8055: guarded decisions, custody and actual CLI exits stay bounded."""

import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import hardware_guard_8055 as h
from carnot import experiment_8055_v697_hardware_guard_boundary as e


def fixture():
    """SCENARIO-REPORT-8055-GUARDS: destructive controls do not enlarge support."""
    head = dict(parameters=[0.0, 0.0], decay_scale=1.0, calibration=[0.0, 1.0])
    cases = []
    for value, delta, arm, size in [
        (0, 0, "feedback_constrained", 8),
        (0, 10, "feedback_constrained", 8),
        (10, 0, "feedback_constrained", 8),
        (0, 1, "feedback_constrained", 2),
        (-2.197224577, 0, "unconstrained", 8),
        (1e8, 0, "feedback_constrained", 8),
    ]:
        cases.append(
            dict(
                identity=str(len(cases)),
                arm=arm,
                seed=0,
                slot=0,
                before=[value, 0],
                delta=[delta, 0],
                natural=False,
                guards=[[0, i % 2] for i in range(size)],
                guard_design=[[1.0, 0.0]] * size,
            )
        )
    return dict(head=head, cases=cases, boards=[], checks=[], references=[], costs=[])


def test_guards_and_bounds():
    """SCENARIO-REPORT-8055-GUARDS: intervals certify predicates before commit."""
    data = fixture()
    result = h.reduce(data)
    assert result["numeric_branch_status"] == "qualified"
    assert result["guard_fallback_ready_score"] == 1
    assert all(r["passed"] for r in result["acceptance_parity_rows"])
    assert all(r["contained"] for r in result["interval_containment_rows"])
    assert any(r["overflow"] for r in result["guard_fallback_rows"])
    assert result["independent_count"] == 0
    assert result["prediction_fallback_denominator"] > 0
    assert result["positive_control_results"]["working"]
    missing = h.reduce(dict(data, cases=[]))
    assert missing["numeric_branch_status"] == "blocked"
    assert missing["verdict_class"] == "blocked"
    with pytest.raises(TimeoutError):
        h.numeric(data, budget_s=-1)
    for bounds, expected in [([0, 1], None), ([-2, -1], False), ([1, 2], True)]:
        assert h.predicate(bounds) is expected
    with pytest.raises(ValueError):
        h.scan(data["head"], np.array([float("nan"), 0]), np.ones((8, 2)), np.zeros(8), 0.001)


def test_costs():
    """SCENARIO-REPORT-8055-COSTS: serial work limits hypothetical arithmetic."""
    row = dict(
        arm="python",
        condition="feedback_constrained",
        natural=True,
        excluded=False,
        repetition=0,
        transaction_class="accepted",
        transaction_ns=1000,
        components=dict(gradient_arithmetic_ns=100, guard_scans_ns=200, storage_fsync_ns=400),
    )
    bounds = h.costs([row], 0.5)
    assert bounds[0]["speedup_bound"] == pytest.approx(1000 / 802)
    assert bounds[0]["device_transfer_ns"] is None
    assert not h.costs([dict(row, excluded=True)], 0)
    with pytest.raises(ValueError):
        h.costs([dict(row, transaction_ns=0)], 0)


def test_current_and_missing_custody(tmp_path):
    """REQ-REPORT-8055: each board authenticates independently of numeric work."""
    data = h.load(e.ROOT, tmp_path / "custody")
    assert len(data["cases"]) == 3960
    assert all(r["custody_valid"] for r in data["boards"])
    assert data["costs"]
    absent = h.load(tmp_path, tmp_path / "absent")
    assert not absent["cases"] and any(not r["passed"] for r in absent["checks"])


def test_cli_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8055-TERMINAL: valid/null/blocked/tampered children exit."""
    source = tmp_path / "fixture.json"
    e.atomic_json(source, fixture())
    output = tmp_path / (e.NAME + ".json")
    argv = ["--fixture-input", str(source), "--output", str(output), "--validation-worker"]
    assert e.main(argv) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    value["guard_fallback_ready_score"] = 0
    e.atomic_json(output, value)
    assert e.main(["--cold-replay", str(output)]) == 1
    assert e.main(["--date", "wrong"]) == 1
    assert e.main(["--root", str(tmp_path), "--output", str(output), "--validation-worker"]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert e.main(["--fixture-input", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(sys, "argv", [e.SCRIPT, *argv])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert exit_info.value.code == 0
    environment = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    cli_receipts = []
    for args, expected in [
        (argv, 0),
        (["--cold-replay", str(output)], 0),
        (["--date", "wrong"], 1),
        (["--root", str(tmp_path), "--output", str(output), "--validation-worker"], 0),
    ]:
        child = subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.SCRIPT), *args],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == expected, child.stdout + child.stderr
        cli_receipts.append(
            dict(
                argv=child.args,
                expected_exit_code=expected,
                actual_exit_code=child.returncode,
                stdout=child.stdout,
                stderr=child.stderr,
            )
        )
    value = json.loads(output.read_bytes())
    value["completed_count"] += 1
    e.atomic_json(output, value)
    child = subprocess.run(
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.SCRIPT),
            "--cold-replay",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 1
    cli_receipts.append(
        dict(
            argv=child.args,
            expected_exit_code=1,
            actual_exit_code=child.returncode,
            stdout=child.stdout,
            stderr=child.stderr,
        )
    )
    if os.environ.get("CARNOT_8055_PRIVATE_CLI_RECEIPTS"):
        e.atomic_json(Path(os.environ["CARNOT_8055_PRIVATE_CLI_RECEIPTS"]), dict(rows=cli_receipts))


def test_replay_tamper(tmp_path):
    """SCENARIO-REPORT-8055-TERMINAL: byte and aggregate mutations fail closed."""
    source = tmp_path / "fixture.json"
    e.atomic_json(source, fixture())
    output = tmp_path / (e.NAME + ".json")
    e.main(["--fixture-input", str(source), "--output", str(output), "--validation-worker"])
    value = json.loads(output.read_bytes())
    plan = Path(value["replay_input_reference"]["path"])
    original = plan.read_bytes()
    plan.write_bytes(original + b" ")
    with pytest.raises(ValueError):
        e.replay(output)
    plan.write_bytes(original)
    assert e.replay(output)["passed"]


def test_validation_and_publication_failure_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8055-TERMINAL: missing coverage and bad exits suppress readiness."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    specs = [e.CommandSpec("small_child", (sys.executable, "-c", "print(1)"), "owned", 10)]
    receipts, counts = e.validate(specs, tmp_path / "validation", scratch)
    assert receipts[0]["actual_exit_code"] == 0 and counts == {}
    e.atomic_json(scratch / "private_cli.json", dict(rows=[]))
    e.atomic_json(
        scratch / "coverage.json", dict(files={"sample": dict(summary={"missing_lines": 1})})
    )
    assert e.validate(specs, tmp_path / "validation2", scratch)[1]["sample"]["missing_lines"] == 1
    source = tmp_path / "fixture.json"
    e.atomic_json(source, fixture())
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(e, "validate", lambda *args: ([], {}))
    assert e.main(["--fixture-input", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert e.replay(output)["passed"]
    reports = iter([dict(passed=True), dict(passed=False)])
    monkeypatch.setattr(e, "terminal", lambda path: next(reports))
    assert (
        e.main(["--fixture-input", str(source), "--output", str(output), "--validation-worker"])
        == 1
    )


def test_empty_guard_and_trajectory_contract(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8055-GUARDS: empty guards hold state and corrupt ledgers block."""
    data = fixture()
    data["cases"][0].update(guards=[], guard_design=[])
    assert h.reduce(data)["acceptance_parity_rows"][0]["passed"]

    def corrupt(*args):
        raise ValueError("preserved_original_trajectory_failure")

    monkeypatch.setattr(h, "workloads", corrupt)
    loaded = h.load(e.ROOT, tmp_path)
    assert any(r["observed"] == "preserved_original_trajectory_failure" for r in loaded["checks"])
    assert h.reduce(loaded)["numeric_branch_status"] == "blocked"
