"""REQ-REPORT-8625, REQ-VERIFY-8625: keep hardware claims within original evidence."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_boundary_8625 as e
from carnot.reporting import hardware_operation_evidence_8625 as evidence
from carnot.reporting.current_work_receipt import atomic_json


def test_current_and_historical_costs(tmp_path):
    """SCENARIO-REPORT-8625-BOUNDARY: complete lean costs never absorb learning costs."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path, "20261018")
    value = e.build(
        work, [dict(name="owned", passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json")
    )
    assert work["evidence"]["board"]["qualified"]
    assert work["evidence"]["lean"]["qualified"]
    assert value["qualified_cost_reference"]["reference"]["path"].endswith(evidence.LEAN)
    assert not value["qualified_cost_reference"]["historical_boundary"]
    assert {r["arm"] for r in work["evidence"]["cost"]["cells"]} == {"lean_python"}
    assert value["hardware_boundary_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert value["NPU_qualified"] == value["TSU_qualified"] == 0
    assert len(value["resident_inference_operation_rows"]) == 55
    assert len(value["delayed_learning_operation_rows"]) == 70
    assert all(r["host_clock_s"] is None for r in value["learning_locality_rows"])
    assert all(1 <= r["zero_cost_ceiling"] < 10 for r in value["conditional_ceiling_rows"])
    assert value["target_model"]["parameter_count"] == 109
    assert value["model_invocation_counts"]["current_llm_invocations"] == 0
    assert value["verdict_class"] == "blocked"
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    value["conditional_ceiling_rows"][0]["zero_cost_ceiling"] = 20
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["cost"]["cells"][0]["total_s"] += 1
    atomic_json(tmp_path / "raw/measurement.json", changed)
    atomic_json(
        output, e.build(changed, [dict(name="owned", passed=True)], tmp_path / "raw", output)
    )
    assert not e.replay(output)


def test_clock_rejections_and_fallback(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8625-BOUNDARY: overlapping clocks and missing slots cannot make ceilings."""
    original = evidence.sealed(e.ROOT, evidence.LEAN, "lean_cost_ready_score")
    assert original["qualified"]
    cells = evidence.cells(original)
    evidence.partition(cells)
    for mutate in [
        lambda c: c.pop(),
        lambda c: c[0]["seconds"].update(offset=-1),
        lambda c: c[0].update(total_s=999),
    ]:
        changed = deepcopy(cells)
        mutate(changed)
        with pytest.raises(ValueError):
            evidence.partition(changed)
    sealed = evidence.sealed
    monkeypatch.setattr(
        evidence,
        "sealed",
        lambda root, name, field: (
            dict(qualified=False, value=None, refs=[], checks=[])
            if name == evidence.LEAN
            else sealed(root, name, field)
        ),
    )
    source = evidence.authenticate(e.ROOT)
    assert source["cost"]["qualified"] and not source["lean"]["qualified"]
    assert {c["arm"] for c in source["cost"]["cells"]} == {"resident_python"}
    assert not evidence.sealed(tmp_path, evidence.COST, "resident_cost_ready_score")["qualified"]


def cli(tmp_path, *args):
    """A real process proves standalone imports and exercises the script's entry statements."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_private_cli_missing_and_owned_failure(tmp_path):
    """SCENARIO-VERIFY-8625-REPLAY: external absence blocks; actual owned child failure disqualifies."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        result = cli(
            tmp_path,
            "--private-control",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
            "--date",
            "20261018",
            *(["--owned-failure"] if failed else []),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert value["hardware_boundary_ready_score"] == 0
        assert all(r["zero_cost_ceiling"] is None for r in value["conditional_ceiling_rows"])
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    for date in ["bad", "20260230"]:
        assert cli(tmp_path, "--date", date).returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2


def test_rejected_originals(tmp_path, monkeypatch):
    """REQ-REPORT-8625: changed pins cannot bypass identity, readiness or bound publication."""
    with pytest.raises(ValueError):
        evidence.checked(dict(path=str(tmp_path / "missing"), sha256="sha256:missing"))
    original = json.loads((e.ROOT / evidence.BOARD).read_bytes())
    path = tmp_path / evidence.BOARD
    for mutation in [{"required_checks_passed": False}, {"hardware_boundary_ready_score": True}]:
        atomic_json(path, dict(original, **mutation))
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


def test_primitive_and_parity_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8625-BOUNDARY: primitive overlap and bad parity reject lean clocks."""
    original = evidence.sealed(e.ROOT, evidence.LEAN, "lean_cost_ready_score")
    row = next(r for r in original["value"]["paired_session_rows"] if r["arm"] == "lean_python")
    primitive = json.loads(Path(row["primitive_reference"]["path"]).read_bytes())
    for kind in ["exit", "overlap"]:
        changed = deepcopy(primitive)
        if kind == "exit":
            changed["exit_code"] = 7
        else:
            changed["operations"][1]["start_ns"] += 1
        atomic_json(tmp_path / "primitive.json", changed)
        monkeypatch.setattr(
            evidence, "checked", lambda _: dict(path=str(tmp_path / "primitive.json"))
        )
        with pytest.raises(ValueError):
            evidence.cells(original)
    monkeypatch.undo()
    sealed = evidence.sealed
    original["value"]["primary_contrast"]["full_parity"] = False
    monkeypatch.setattr(
        evidence,
        "sealed",
        lambda root, name, field: (
            deepcopy(original) if name == evidence.LEAN else sealed(root, name, field)
        ),
    )
    source = evidence.authenticate(e.ROOT)
    assert source["cost"]["qualified"] and not source["lean"]["qualified"]
    with pytest.raises(ValueError):
        evidence.partition([dict(seconds={})])


def test_changed_authority_rejected(tmp_path):
    """SCENARIO-VERIFY-8625-REPLAY: a self-rehashed authority summary cannot grant current methods."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path, "20261018")
    output = tmp_path / (e.NAME + ".json")
    work["contract"]["ready"] = True
    atomic_json(tmp_path / "raw/measurement.json", work)
    atomic_json(output, e.build(work, [dict(name="owned", passed=True)], tmp_path / "raw", output))
    assert not e.replay(output)


def test_validation_plumbing_and_full_mode(tmp_path, monkeypatch):
    """REQ-VERIFY-8625: full orchestration retains exact collection and actual private child logs."""
    from carnot.reporting import hardware_operation_runner_8625 as runner

    calls = []

    def fake_child(name, argv, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / (name + ".stdout")
        if name.startswith("collect"):
            log.write_text(e.TEST + "::test_control\n")
        else:
            log.write_text("")
        if name.startswith("tests"):
            xml = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")))
            xml.write_text(
                '<testsuite><testcase classname="tests.python.test_hardware_operation_boundary_8625" name="test_control"/></testsuite>'
            )
        if name == "coverage_report":
            atomic_json(
                folder / "coverage.json",
                dict(
                    files={
                        p: dict(missing_lines=[], summary=dict(num_statements=1)) for p in e.OWNED
                    }
                ),
            )
        calls.append(name)
        return dict(name=name, passed=True, stdout_path=str(log), duration_s=0)

    with monkeypatch.context() as context:
        context.setattr(runner, "child", fake_child)
        context.setattr(runner, "SUITES", [e.TEST])
        receipts = runner.validate(tmp_path / "plumbing")
    assert all(r["passed"] for r in receipts)
    assert "strict_mypy" in calls
    monkeypatch.setattr(runner, "validate", lambda _: receipts)
    assert (
        runner.run(
            tmp_path / "absent",
            tmp_path / "full" / (e.NAME + ".json"),
            tmp_path,
            "20261018",
            False,
            False,
        )
        == 0
    )


def test_rehashed_owned_receipt_rejected(tmp_path):
    """SCENARIO-VERIFY-8625-REPLAY: rewriting a failed receipt cannot grant owned acceptance."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(
        tmp_path,
        "--private-control",
        "--owned-failure",
        "--root",
        tmp_path / "absent",
        "--output",
        output,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    work_path = Path(value["work_reference"]["path"])
    work = json.loads(work_path.read_bytes())
    forged = deepcopy(value["validation_receipts"])
    forged[0]["passed"] = True
    atomic_json(output, e.build(work, forged, work_path.parent, output))
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
