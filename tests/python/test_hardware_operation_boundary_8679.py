"""REQ-REPORT-8679, REQ-VERIFY-8679: independent cost admission and sealed replay."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_boundary_8679 as e
from carnot.reporting import hardware_boundary_costs_8679 as costs
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary


def producer(root, index=2, gates=None):
    """Use real primary publication so fixtures exercise the actual terminal contract."""
    number, version, suffix, field, boundary = e.PRODUCERS[index]
    output = root / f"results/experiment_{number}_{version}_{suffix}.json"
    raw = output.parent / "raw" / output.stem
    operations = e.old.previous.REQUIRED[boundary]
    spans = [
        dict(operation=op, start_ns=i * 100, end_ns=(i + 1) * 100)
        for i, op in enumerate(operations)
    ]
    primitive = raw / "cost.json"
    atomic_json(
        primitive,
        dict(
            boundary=boundary,
            cohort="private",
            arm="cpu",
            exit_code=0,
            start_ns=0,
            end_ns=len(spans) * 100,
            spans=spans,
        ),
    )
    value = dict(
        experiment_id=number,
        task_id=f"exp{number}-{suffix.replace('_', '-')}",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        acceptance_gates=gates or {"owned_checks": True},
        adversarial_findings=[],
        operation_cost_closure=[e.reference(primitive)],
        **{field: 1},
    )
    publish_primary(output, value, lambda p: dict(passed=True))
    return output, primitive, value


def cli(tmp_path, *args):
    """Run actual children outside the checkout without invoking recursive validation."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_shapes_and_historical_typeerror():
    """SCENARIO-REPORT-8679-ADMISSION: adapt original forms and retain scientific nulls."""
    boolean = {"owned_checks": True, "paired_costs": True}
    named = [
        {"name": "qualified_complete_measurement", "passed": True},
        {"name": "NFR-01", "passed": False},
    ]
    with pytest.raises(TypeError, match="string indices"):
        any(g["name"] == "qualified_complete_measurement" for g in boolean)
    assert costs.acceptance(boolean)["qualified"]
    assert costs.acceptance(named)["qualified"]
    assert costs.acceptance(named)["failures"] == ["NFR-01"]
    for malformed in [
        None,
        [],
        {},
        [True],
        [dict(name="x", passed=1)],
        [dict(name="x", passed=True)] * 2,
        {"x": 1},
        {1: True},
    ]:
        with pytest.raises(ValueError):
            costs.acceptance(malformed)
    assert not costs.acceptance({"owned_checks": False})["qualified"]
    assert not costs.acceptance([dict(name="unknown", passed=True)])["qualified"]


def test_independent_admission_null_and_missing_states(tmp_path):
    """REQ-REPORT-8679: a missing learner cannot veto qualified batching costs."""
    gates = [
        dict(name="qualified_complete_measurement", passed=True),
        dict(name="NFR-01", passed=False),
    ]
    output, primitive, original = producer(tmp_path, gates=gates)
    branch = e.admit(tmp_path, e.PRODUCERS[2])
    assert branch["qualified"] and branch["acceptance_shape"]["failures"] == ["NFR-01"]
    assert branch["cells"][0]["conditional_ceiling"] is None
    assert not e.admit(tmp_path, e.PRODUCERS[0])["qualified"]
    for mutation, state in [
        ({"batch_cost_ready_score": 0}, "observed_zero"),
        ({"batch_cost_ready_score": None}, "observed"),
    ]:
        publish_primary(output, {**original, **mutation}, lambda p: dict(passed=True))
        assert e.admit(tmp_path, e.PRODUCERS[2])["checks"][0]["observed_state"] == state
    missing = dict(original)
    del missing["batch_cost_ready_score"]
    publish_primary(output, missing, lambda p: dict(passed=True))
    assert e.admit(tmp_path, e.PRODUCERS[2])["checks"][0]["observed_state"] == "missing_field"
    publish_primary(output, original, lambda p: dict(passed=True))
    primitive.write_text("{}")
    assert e.admit(tmp_path, e.PRODUCERS[2])["checks"][0]["observed_state"] == "changed_bytes"


def test_real_source_and_hardware_scope(tmp_path):
    """SCENARIO-REPORT-8679-HARDWARE: original source costs survive absent future siblings."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path, "20261101")
    source = work["evidence"]["branches"][1]
    assert source["qualified"] and len(source["cells"]) == 1140
    assert all(c["conditional_ceiling"] is None for c in source["cells"])
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["hardware_mapping_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert value["device_commands_executed"] == []
    assert value["preserved_quarantines"]["Exp8664"]["flagged_adversarial"] is True


def test_private_cli_replay_rejection_and_owned_failure(tmp_path):
    """SCENARIO-VERIFY-8679-REPLAY: real child success/failure and rehashed tampering reject."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        result = cli(
            tmp_path,
            "--private-control",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
            *(["--owned-failure"] if failed else []),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        work_path = Path(value["work_reference"]["path"])
        work = json.loads(work_path.read_bytes())
        work["evidence"]["branches"][0]["qualified"] = True
        atomic_json(work_path, work)
        atomic_json(output, e.build(work, value["validation_receipts"], work_path.parent, output))
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    for date in ["bad", "20260230"]:
        assert cli(tmp_path, "--date", date).returncode == 2
    assert cli(tmp_path, "--private-control").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1


def test_rejection_shapes_flags_and_spans(tmp_path):
    """REQ-REPORT-8679: unknown shapes, critical flags and malformed costs never qualify."""
    output, primitive, original = producer(tmp_path)
    for changed in [
        dict(required_checks_passed=False),
        dict(flagged_adversarial=True),
        dict(acceptance_gates={"owned": False}),
        dict(acceptance_gates=[True]),
        dict(adversarial_findings=[dict(severity="critical")]),
        dict(
            adversarial_findings={"reports": [{"flags": [dict(severity="info")]}]},
            flagged_adversarial=False,
        ),
        dict(task_id="exp8675-wrong"),
        dict(operation_cost_closure=[]),
        dict(operation_cost_closure=[e.reference(e.ROOT / "CODEX.md")]),
        dict(operation_cost_closure=[e.reference(primitive)] * 2),
    ]:
        output.unlink()
        publish_primary(output, {**original, **changed}, lambda p: dict(passed=True))
        qualified = "reports" in changed.get("adversarial_findings", {})
        assert e.admit(tmp_path, e.PRODUCERS[2])["qualified"] is qualified
    for value in [None, [True], [dict(severity="unknown")]]:
        with pytest.raises(ValueError):
            costs.flags(value)
    assert costs.flags({"reports": [{"flags": [dict(severity="critical")]}]})
    for fraction, expected in [(0, 1), (2, 2), (4, None)]:
        result = costs.conditional(dict(total_s=4, compatible_s=fraction), True)
        assert result["conditional_ceiling"] == expected
    assert (
        costs.conditional(dict(total_s=4, compatible_s=None), True)["conditional_ceiling"] is None
    )
    for total, part in [(0, 0), (1, -1), (1, 2)]:
        with pytest.raises(ValueError):
            costs.conditional(dict(total_s=total, compatible_s=part), True)


def test_frozen_seal_and_terminal_checks(tmp_path, monkeypatch):
    """REQ-VERIFY-8679: exact final bytes and the independent seal control admission."""
    from carnot.reporting import hardware_boundary_runner_8679 as runner

    work = e.measure(tmp_path, tmp_path / "raw", tmp_path, "20261101")
    output = tmp_path / (e.NAME + ".json")
    receipts = [dict(passed=True)]
    atomic_json(output, e.build(work, receipts, tmp_path / "raw", output))
    assert e.replay(output)
    changed = deepcopy(json.loads(output.read_bytes()))
    changed["hardware_mapping_ready_score"] = 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    atomic_json(tmp_path / "raw/measurement.seal.json", dict(path="foreign", sha256="bad"))
    assert not e.replay(output)
    monkeypatch.setattr(runner, "validate", lambda raw: [dict(name="owned", passed=True)])
    assert (
        runner.run(
            tmp_path, tmp_path / "ok" / (e.NAME + ".json"), tmp_path, "20261101", False, False
        )
        == 0
    )
    monkeypatch.setattr(runner, "terminal", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.run(
            tmp_path, tmp_path / "bad" / (e.NAME + ".json"), tmp_path, "20261101", False, False
        )


def test_terminal_reader_receipt_and_unknown_closure(tmp_path, monkeypatch):
    """REQ-REPORT-8679: exact readers and terminal sidecars remain mandatory."""
    output, _, value = producer(tmp_path)
    side = (
        output.parent
        / "raw"
        / output.stem
        / "validators"
        / (e.reference(output)["sha256"][7:] + ".json")
    )
    old_side = side.read_bytes()
    atomic_json(
        side,
        dict(
            primary_sha256=e.reference(output)["sha256"],
            primary_path=str(output),
            report=dict(passed=False),
        ),
    )
    assert not e.admit(tmp_path, e.PRODUCERS[2])["qualified"]
    side.write_bytes(old_side)
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=False))
    assert not e.admit(tmp_path, e.PRODUCERS[2])["qualified"]
    monkeypatch.undo()
    del value["operation_cost_closure"]
    publish_primary(output, value, lambda p: dict(passed=True))
    assert e.admit(tmp_path, e.PRODUCERS[2])["checks"][0]["observed_state"] == "missing_field"


def test_source_operand_rejections(tmp_path):
    """REQ-REPORT-8679: sealed source clocks reject roster, byte and endpoint substitutions."""
    original = json.loads(
        (e.ROOT / "results/experiment_8659_v743_complete_source_comparison.json").read_bytes()
    )
    authority = json.loads(e.checked(original["primitive_reference"]).read_bytes())
    path = tmp_path / "authority.json"
    for kind in ["roster", "cost", "endpoints"]:
        work = deepcopy(authority)
        rows = work["report"]["complete_extraction_cost_rows"]
        if kind == "roster":
            rows.pop()
        elif kind == "cost":
            rows[0]["duration_s"] += 1
        else:
            rows[0]["receipt"]["exit_code"] = 7
        atomic_json(path, work)
        changed = {
            **original,
            "primitive_reference": e.reference(path),
            "complete_extraction_cost_rows": rows,
        }
        with pytest.raises(
            ValueError, match={"roster": "roster", "cost": "bytes", "endpoints": "endpoints"}[kind]
        ):
            costs.source_cells(changed, [])


def test_validation_wrapper_binds_new_owned_sources(tmp_path, monkeypatch):
    """REQ-VERIFY-8679: consumer-only coverage cannot certify this producer."""
    from carnot.reporting import hardware_boundary_runner_8679 as runner
    from carnot.reporting import hardware_evidence_validation_8665 as validation

    def inspect(raw):
        assert validation.e is e
        assert validation.SUITES[0] == e.TEST
        assert raw == tmp_path
        return [dict(passed=True)]

    monkeypatch.setattr(validation, "validate", inspect)
    assert runner.validate(tmp_path) == [dict(passed=True)]


def test_disjoint_opaque_boundary_and_invalid_endpoints(tmp_path):
    """SCENARIO-REPORT-8679-ADMISSION: binding time is not isolated spline arithmetic."""
    output, primitive, value = producer(tmp_path)
    source = json.loads(primitive.read_bytes())
    for span in source["spans"]:
        if span["operation"] == "deterministic_spline_109":
            span["operation"] = "binding_residual"
    atomic_json(primitive, source)
    del value["operation_cost_closure"]
    value["complete_operation_spans"] = [dict(primitive_reference=e.reference(primitive))]
    publish_primary(output, value, lambda p: dict(passed=True))
    admitted = e.admit(tmp_path, e.PRODUCERS[2])
    assert admitted["qualified"]
    assert admitted["cells"][0]["compatible_s"] is None
    assert admitted["cells"][0]["conditional_ceiling"] is None
    for mutation in [dict(exit_code=7), dict(spans=[]), dict(end_ns=0)]:
        with pytest.raises(ValueError):
            costs.reduce_spans({**source, **mutation}, "installed_inference")
    bad = deepcopy(source)
    bad["spans"][1]["start_ns"] += 1
    with pytest.raises(ValueError, match="disjoint"):
        costs.reduce_spans(bad, "installed_inference")
    zero = dict(
        boundary="installed_inference",
        exit_code=0,
        start_ns=0,
        end_ns=0,
        spans=[dict(operation="binding_residual", start_ns=0, end_ns=0)],
    )
    with pytest.raises(ValueError, match="nonpositive"):
        costs.reduce_spans(zero, "installed_inference")


def test_snapshot_cross_source_and_disk_backing_reject(tmp_path, monkeypatch):
    """REQ-VERIFY-8679: snapshots need matching originals and scratch must be disk-backed."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, tmp_path, "20261101")
    work["snapshots"] = [
        dict(
            path="missing",
            sha256="changed",
            original_reference=dict(path="missing", sha256="original"),
        )
    ]
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "measurement.seal.json", e.reference(raw / "measurement.json"))
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    monkeypatch.setattr(
        e.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "tmpfs tmpfs"})()
    )
    with pytest.raises(ValueError, match="disk_backed"):
        e.measure(tmp_path, tmp_path / "other", tmp_path, "20261101")


def test_rehashed_measurement_cannot_replace_original_producer(tmp_path):
    """REQ-VERIFY-8679: even a replaced local seal cannot invent original measured costs."""
    producer(tmp_path)
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, tmp_path, "20261101")
    output = tmp_path / (e.NAME + ".json")
    receipts = [dict(passed=True)]
    atomic_json(output, e.build(work, receipts, raw, output))
    assert e.replay(output)
    work["evidence"]["branches"][2]["cells"][0]["total_s"] *= 2
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "measurement.seal.json", e.reference(raw / "measurement.json"))
    atomic_json(output, e.build(work, receipts, raw, output))
    assert not e.replay(output)
