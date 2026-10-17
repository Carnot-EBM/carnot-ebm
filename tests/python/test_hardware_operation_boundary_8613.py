"""REQ-REPORT-8613, REQ-VERIFY-8613: original authority precedes cost mapping."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import hardware_operation_boundary_8613 as e
from carnot.reporting import hardware_operation_evidence_8613 as evidence
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    """Real children cover the standalone entry and fresh replay imports."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    active = coverage.Coverage.current()
    if active is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(active.config.config_file)]
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=90)


def test_original_mapping(tmp_path):
    """SCENARIO-REPORT-8613-BOUNDARY: historical costs and physical readiness stay separate."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path, "20261017")
    value = e.build(
        work, [dict(name="owned", passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json")
    )
    assert work["evidence"]["cost"]["qualified"]
    assert value["board_boundary_recorded_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert value["target_model"]["parameter_count"] == 109
    assert len(value["resident_inference_operation_rows"]) == 50
    assert len(value["delayed_learning_operation_rows"]) == 65
    assert all(r["host_clock_s"] is None for r in value["learning_locality_rows"])
    assert all(r["zero_cost_ceiling"] < 10 for r in value["conditional_ceiling_rows"])
    assert value["qualified_cost_reference"]["historical_boundary"]
    assert value["model_invocation_counts"]["current_llm_invocations"] == 0
    assert value["verdict_class"] == "blocked"
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    value["conditional_ceiling_rows"][0]["zero_cost_ceiling"] = 20
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    value = e.build(work, [dict(name="owned", passed=True)], tmp_path / "raw", output)
    altered = deepcopy(work)
    altered["evidence"]["cost"]["cells"][0]["total_s"] += 1
    atomic_json(tmp_path / "raw/measurement.json", altered)
    value["work_reference"] = e.reference(tmp_path / "raw/measurement.json")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_missing_original_and_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8613-REPLAY: a changed pin cannot repair original identity or publication."""
    assert not evidence.sealed(tmp_path, evidence.BOARD, "hardware_boundary_ready_score")[
        "qualified"
    ]
    path = tmp_path / evidence.BOARD
    atomic_json(path, {})
    assert not evidence.sealed(tmp_path, evidence.BOARD, "hardware_boundary_ready_score")[
        "qualified"
    ]
    monkeypatch.setitem(evidence.PINS, evidence.BOARD, e.reference(path)["sha256"])
    assert not evidence.sealed(tmp_path, evidence.BOARD, "hardware_boundary_ready_score")[
        "qualified"
    ]
    original = json.loads((e.ROOT / evidence.BOARD).read_bytes())
    for field, mutation in [
        ("required_checks_passed", False),
        ("hardware_boundary_ready_score", 0),
        ("flagged_adversarial", True),
    ]:
        changed = dict(original, **{field: mutation})
        atomic_json(path, changed)
        monkeypatch.setitem(evidence.PINS, evidence.BOARD, e.reference(path)["sha256"])
        assert not evidence.sealed(tmp_path, evidence.BOARD, "hardware_boundary_ready_score")[
            "qualified"
        ]
    atomic_json(path, original)
    monkeypatch.setitem(evidence.PINS, evidence.BOARD, e.reference(path)["sha256"])
    monkeypatch.setattr(evidence, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
    assert not evidence.sealed(tmp_path, evidence.BOARD, "hardware_boundary_ready_score")[
        "qualified"
    ]


def test_private_cli_controls(tmp_path):
    """SCENARIO-VERIFY-8613-CONTROLS: actual failures disqualify and missing costs stay null."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        args = ["--private-control", "--root", tmp_path / "absent", "--output", output]
        result = cli(
            tmp_path, *args, *(["--owned-failure"] if failed else []), "--date", "20261017"
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert all(r["zero_cost_ceiling"] is None for r in value["conditional_ceiling_rows"])
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    for date in ["bad", "20260230"]:
        assert cli(tmp_path, "--date", date).returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2
    assert cli(tmp_path, "--deliberate-error").returncode == 7


def test_partition_rejection():
    """REQ-REPORT-8613: overlapping clocks never authorize an acceleration ceiling."""
    with pytest.raises(ValueError):
        evidence.partition([])


def test_owned_rejection_branches(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8613-REPLAY: corrupt clocks, authority substitutions and timeout fail closed."""
    from carnot.reporting import hardware_operation_runner_8613 as runner

    with pytest.raises(TimeoutError):
        e.scan_timeout(0, None)
    with pytest.raises(ValueError):
        evidence.checked(dict(path=str(tmp_path / "missing"), sha256="sha256:missing"))
    with pytest.raises(ValueError):
        evidence.checked(
            dict(path=str(e.ROOT / evidence.authority.ACTIVE), sha256="sha256:changed")
        )
    original = evidence.sealed(e.ROOT, evidence.COST, "resident_cost_ready_score")
    original["value"]["primary_contrast"]["full_parity"] = False
    monkeypatch.setattr(
        evidence,
        "sealed",
        lambda root, name, field: (
            deepcopy(original)
            if name == evidence.COST
            else dict(qualified=False, value=None, refs=[], checks=[])
        ),
    )
    assert not evidence.authenticate(tmp_path)["cost"]["qualified"]
    original["value"]["primary_contrast"]["full_parity"] = True
    original["value"]["paired_session_rows"] = []
    assert not evidence.authenticate(tmp_path)["cost"]["qualified"]
    monkeypatch.setattr(evidence, "authenticate", lambda *_: (_ for _ in ()).throw(TimeoutError()))
    work = e.measure(tmp_path, tmp_path / "raw", tmp_path, "20261017")
    assert not work["evidence"]["board"]["qualified"]
    monkeypatch.setattr(runner.plans, "manifest", lambda *_: [dict(argv=["pytest", "-q"])])
    assert runner.manifest(tmp_path) == [dict(argv=["pytest", "-vv"])]


def test_full_runner_copy_and_union(tmp_path, monkeypatch):
    """REQ-VERIFY-8613: full-mode plumbing retains coverage and original union receipts."""
    from carnot.reporting import hardware_operation_runner_8613 as runner

    private = tmp_path / "private"
    private.mkdir()
    # These fixture bytes exercise copying only; actual coverage is required by the real plan.
    atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=100)))
    (private / ".coverage.fixture").write_bytes(b"copy control")
    plan = [
        dict(
            name="actual",
            argv=[sys.executable, "-c", 'print("actual child")'],
            deadline=30,
            expected=0,
            scope="owned",
        )
    ]
    monkeypatch.setattr(runner, "manifest", lambda *_: plan)
    monkeypatch.setattr(runner, "node_union", lambda *_: dict(passed=True, scope="copy fixture"))
    output = tmp_path / "full" / (e.NAME + ".json")
    assert (
        runner.run(
            tmp_path / "missing",
            output,
            private,
            control=False,
            owned_failure=False,
            date="20261017",
        )
        == 0
    )
    assert list(output.parent.glob("raw/**/coverage.json"))


def test_future_paths_and_immutable_checks(tmp_path):
    """REQ-REPORT-8613: canonical missing producers retain their exact conductor diagnostics."""
    source = evidence.authenticate(e.ROOT)
    assert source["lean"]["requested_path"].endswith("experiment_8610_v739_lean_resident_cost.json")
    assert source["lean"]["diagnostic_reference"]["path"].endswith(
        "experiment_8610_lean_resident_cost.json"
    )
    assert not source["lean"]["qualified"]
    assert any(
        g["field"] == "lean_measurement_ready_score" and g["observed"] == 0
        for g in source["lean"]["checks"]
    )
    work = e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path, "20261017")
    output = tmp_path / (e.NAME + ".json")
    work["contract"]["ready"] = True
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(output, e.build(work, [dict(name="owned", passed=True)], tmp_path / "raw", output))
    assert not e.replay(output)
