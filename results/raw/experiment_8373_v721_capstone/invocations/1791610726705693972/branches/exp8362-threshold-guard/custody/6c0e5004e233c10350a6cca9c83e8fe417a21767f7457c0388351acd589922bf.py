"""REQ-VERIFY-8362 / REQ-REPORT-8362: exact decisions need a proved bound."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.verify import threshold_guard_8362 as n
from carnot.reporting import threshold_guard_8362 as e
from carnot.reporting import threshold_guard_execution_8362 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def operands(tmp_path_factory):
    """REQ-REPORT-8362: use original coefficients, never replacement observations."""
    return e.authenticate(e.ROOT, tmp_path_factory.mktemp("threshold-inputs"))


def test_analytic_cells_and_reference(operands):
    """SCENARIO-VERIFY-8362-BOUND: exact extrema agree with an independent polynomial."""
    head, table = operands["head"], np.asarray(operands["table"], dtype="<i2")
    bound = n.certificate(head, table)
    assert len(bound["cells"]) == 4
    assert all(len(feature) == 64 for feature in bound["cells"])
    assert bound["spline_proof_passed"]
    assert bound["independent_extrema_passed"]
    assert not bound["numerical_policy_proof_passed"]
    assert any(len(cell["split_points"]) > 2 for cell in bound["cells"][0])
    assert all(cell["logit_error_bound"] >= 1 / 4096 for f in bound["cells"] for cell in f)
    x, kinds = n.panel(head)
    assert kinds.count("random") == 4096
    assert "nextafter_threshold" in kinds and "nextafter_knot" in kinds
    direct, actual = n.old.direct(head, x), n.old.lookup(head, x, table, "int16", "linear")
    assert (
        sum(n.action(float(a)) != n.action(float(b)) for a, b in zip(direct[:4174], actual[:4174]))
        == 3
    )
    rows = n.evaluate(head, table, x, kinds)
    assert all(
        r["probability_interval"][0] <= r["direct_probability"] <= r["probability_interval"][1]
        for r in rows
    )
    assert all(r["guarded_action"] == r["direct_action"] for r in rows)
    assert not any(r["fast_path"] for r in rows)
    assert any(r["empirical_fast_path"] for r in rows[:4096])


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (0, 0.249, "accept"),
        (0.751, 1, "reject"),
        (0.25, 0.75, "escalate"),
        (0.2, 0.25, None),
        (0.75, 0.8, None),
        (0, 1, None),
    ],
)
def test_strict_regions(lo, hi, expected):
    """SCENARIO-VERIFY-8362-PROOF: strict confident regions leave ties to escalation."""
    assert n.region(lo, hi) == expected
    assert n.action(0.25) == n.action(0.75) == "escalate"


@pytest.mark.parametrize(
    "bad",
    [
        [float("nan"), 0, 0, 0, 0],
        [float("inf"), 0, 0, 0, 0],
        [0, -0.1, 0, 0, 0],
        [0, 1.1, 0, 0, 0],
        [0] * 4,
    ],
)
def test_invalid_features(operands, bad):
    """SCENARIO-VERIFY-8362-BOUND: invalid natural features never enter table indexing."""
    with pytest.raises(ValueError, match="features"):
        n.guard(operands["head"], np.array(bad), np.asarray(operands["table"], dtype="<i2"), None)


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_invalid_temperature(operands, temperature):
    """SCENARIO-VERIFY-8362-BOUND: undefined policy temperature is an explicit rejection."""
    with pytest.raises(ValueError, match="temperature"):
        n.guard(
            dict(operands["head"], temperature=temperature),
            np.zeros(5),
            np.asarray(operands["table"], dtype="<i2"),
            None,
        )


def test_bound_and_table_faults(operands):
    """SCENARIO-VERIFY-8362-BOUND: rehashed malformed certificates cannot authorize a fast result."""
    head, table = operands["head"], np.asarray(operands["table"], dtype="<i2")
    cert = n.certificate(head, table)
    x = np.array([0, 0.5, 0.5, 0.5, 0.5])
    under = deepcopy(cert)
    under["cells"][0][0]["logit_error_bound"] = 0
    stale = dict(cert, version="stale")
    for supplied, data, reason in [
        (None, table, "missing_bounds"),
        (stale, table, "stale_version"),
        (under, table, "underbound_or_changed_certificate"),
        (cert, table[::-1], "swapped_table"),
        (cert, np.full((4, 65), 32767, dtype="<i2"), "saturated_table"),
    ]:
        row = n.guard(head, x, data, supplied)
        assert reason in row["fallback_reason"]
        assert row["guarded_action"] == row["direct_action"]
        assert not row["fast_path"]
    row = n.guard(head, np.array([1e308, 0.5, 0.5, 0.5, 0.5]), table, cert)
    assert row["fallback_reason"] == "saturation_or_overflow"
    invalid = dict(head, coefficients=[float("nan"), *head["coefficients"][1:]])
    with pytest.raises(ValueError, match="coefficients"):
        n.certificate(invalid, table)


def test_missing_input_and_real_failure(tmp_path):
    """SCENARIO-REPORT-8362-REPLAY: missing is not measured zero and a real failed child stays failed."""
    with pytest.raises(e.OperandError):
        e.authenticate(tmp_path, tmp_path / "raw")
    receipt = child(
        "deliberate_failure",
        [sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path / "logs",
        deadline=10,
    )
    assert receipt["exit_code"] == 7 and not receipt["passed"]


@pytest.fixture(scope="module")
def measured(operands, tmp_path_factory):
    """REQ-REPORT-8362: private natural operand copies back all replay controls."""
    raw = tmp_path_factory.mktemp("threshold-measured")
    work = e.measure(operands, raw)
    atomic_json(raw / "manifest.json", {"commands": []})
    work.update(
        failures=[],
        coverage={
            "totals": {"percent_covered": 100},
            "files": {p: {"summary": {"missing_lines": 0}} for p in e.OWNED},
        },
        manifest_reference=e.reference(raw / "manifest.json"),
        code_refs=[],
        duration_s=2.0,
    )
    receipt = child(
        "primitive_child", [sys.executable, "-c", "print('real child')"], raw / "logs", deadline=10
    )
    atomic_json(raw / "measurement.json", work)
    output = raw / "results" / (e.NAME + ".json")
    return work, [receipt], raw, output


def test_replay_and_typed_qualification(measured):
    """SCENARIO-REPORT-8362-REPLAY: valid primitive meaning cannot certify unknown libm error."""
    work, receipts, raw, output = measured
    value = e.build(work, receipts, raw, output)
    assert value["required_checks_passed"] and value["verdict_class"] == "null"
    assert value["guard_ready_score"] == value["guard_useful_score"] == 0
    assert value["original_table_flip_count"] == 3
    assert value["completed_count"] == value["intended_count"] == 4218
    path = raw / "valid-test.json"
    atomic_json(path, value)
    assert e.replay(path)
    assert not e.replay(raw / "absent.json")
    bad = dict(value, guard_ready_score=1)
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(path, bad)
    assert not e.replay(path)
    failed = dict(receipts[0], passed=False, exit_code=7)
    assert e.build(work, [failed], raw, output)["verdict_class"] == "disqualified"
    blocked = dict(work, failures=[{"field": "missing"}], numeric_summary={}, primitive_refs=[])
    assert e.build(blocked, [], raw, output)["verdict_class"] == "blocked"
    assert (
        e.build(dict(work, coverage={}), receipts, raw, output)["verdict_class"] == "disqualified"
    )
    e.write_note(value, raw / "note.md")
    assert "empirical" in (raw / "v721-threshold-error-bound.md").read_text()


def test_rehashed_primitive_and_operand_faults(measured, monkeypatch):
    """SCENARIO-REPORT-8362-REPLAY: recomputation rejects repaired hashes and fake aggregate zeros."""
    work, receipts, raw, output = measured
    real_reconstruct = e.reconstruct
    primitives = json.loads(Path(work["primitive_refs"][0]["path"]).read_bytes())
    monkeypatch.setattr(
        e,
        "reconstruct",
        lambda inputs: real_reconstruct(inputs) if inputs.get("force_real") else primitives,
    )
    for label in [
        "checkpoint",
        "activated",
        "work_hash",
        "input_hash",
        "log_hash",
        "head",
        "task",
        "protocol",
        "table",
        "authority",
        "primitives",
        "summary",
    ]:
        changed = deepcopy(work)
        rs = deepcopy(receipts)
        if label == "checkpoint":
            checkpoint_ref = next(
                r for r in changed["inputs"]["refs"] if r["source_sha256"] == e.old.CHECKPOINT
            )
            checkpoint = json.loads(Path(checkpoint_ref["path"]).read_bytes())
            altered_head = next(h for h in checkpoint["heads"] if h["arm"] == "spline34")
            altered_head["coefficients"][0] += 0.01
            private = raw / "rehashed-checkpoint.json"
            atomic_json(private, checkpoint)
            checkpoint_ref["path"], checkpoint_ref["sha256"] = (
                str(private),
                e.reference(private)["sha256"],
            )
            changed["inputs"]["head"] = altered_head
        if label == "activated":
            import yaml

            activated_ref = next(
                r
                for r in changed["inputs"]["refs"]
                if Path(r["source_path"]).name == "research-roadmap.yaml"
            )
            activated = yaml.safe_load(Path(activated_ref["path"]).read_bytes())
            next(t for t in activated["tasks"] if t["id"] == e.TASK)["prompt"] += "tamper"
            private = raw / "rehashed-active.yaml"
            private.write_text(yaml.safe_dump(activated))
            activated_ref["path"], activated_ref["sha256"] = (
                str(private),
                e.reference(private)["sha256"],
            )
        if label == "input_hash":
            changed["code_refs"] = [{"path": str(raw / "manifest.json"), "sha256": "bad"}]
        if label == "log_hash":
            rs[0]["stdout_sha256"] = "bad"
        if label == "head":
            changed["inputs"]["head"]["coefficients"][0] += 1
        if label == "task":
            changed["inputs"]["task"]["prompt"] += "tamper"
        if label == "protocol":
            changed["inputs"]["protocol"]["head"]["degree"] = 2
        if label == "table":
            changed["inputs"]["table"][0][0] += 1
        if label == "authority":
            authority = next(
                r
                for r in changed["inputs"]["refs"]
                if Path(r["source_path"]).name == "research-roadmap-vNEXT.md"
            )
            private = raw / "bad-authority.md"
            private.write_text(
                Path(authority["path"])
                .read_text()
                .replace('"id": "exp8362-threshold-guard"', '"id": "exp9999-threshold-guard"')
            )
            authority["path"], authority["sha256"] = str(private), e.reference(private)["sha256"]
        if label == "primitives":
            private = raw / "bad-primitives.json"
            bad = deepcopy(primitives)
            bad["rows"][0]["guarded_action"] = "tamper"
            atomic_json(private, bad)
            changed["primitive_refs"] = [e.reference(private)]
        if label == "summary":
            changed["numeric_summary"]["probability_error_max"] += 0.1
        atomic_json(raw / "measurement.json", changed)
        value = e.build(changed, rs, raw, output)
        if label == "work_hash":
            value["work_reference"]["sha256"] = "bad"
        path = raw / (label + "-test.json")
        atomic_json(path, value)
        assert not e.replay(path), label
    atomic_json(raw / "measurement.json", work)
    # Exact digest and complete task equality differ: the design must also
    # match the stored task even if a private source reference was repaired.
    monkeypatch.setattr(
        e,
        "parse_design",
        lambda *args, **kwargs: ({}, [dict(work["inputs"]["task"], prompt="changed")]),
    )
    atomic_json(raw / "source-drift-test.json", e.build(work, receipts, raw, output))
    assert not e.replay(raw / "source-drift-test.json")


def test_sidecar_and_input_rejections(operands, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8362-REPLAY: invalid terminal seals are named external blocks."""
    with monkeypatch.context() as patch:
        patch.setattr(
            e, "read_bound_sidecar", lambda *args: (_ for _ in ()).throw(ValueError("bad_seal"))
        )
        with pytest.raises(e.OperandError, match="bound_terminal_sidecar"):
            e.authenticate(e.ROOT, tmp_path / "bad-seal")
    raw = tmp_path / "private"
    with monkeypatch.context() as patch:
        patch.setattr(e, "read_bound_sidecar", lambda *args: {"report": {"passed": False}})
        with pytest.raises(e.OperandError, match="report.passed"):
            e.authenticate(e.ROOT, raw)
    head = operands["head"]
    table = np.asarray(operands["table"], dtype="<i2")
    cert = n.certificate(head, table)
    row = n.guard(head, np.zeros(5), table.astype(float), cert)
    assert row["fallback_reason"] == "table_format"
    overflow = dict(head, coefficients=[4, *head["coefficients"][1:]])
    row = n.guard(
        overflow, np.array([1e308, 0.5, 0.5, 0.5, 0.5]), table, n.certificate(overflow, table)
    )
    assert row["fallback_reason"] == "saturation_or_overflow"


def test_private_e2e018(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8362-EXECUTION: real children and unchanged readers exercise private publication."""
    from carnot.reporting import spline_table_execution_8352 as base
    from carnot.reporting.primary_publication import reader_receipt, read_bound_sidecar

    # Unit-only coverage data exercises lifecycle gates. Production obtains
    # its coverage from the real frozen coverage commands, never this fixture.
    def private_plan(private):
        report = {
            "totals": {"percent_covered": 100},
            "files": {p: {"summary": {"missing_lines": 0}} for p in e.OWNED},
        }
        code = (
            "from pathlib import Path; Path("
            + repr(str(private / "coverage.json"))
            + ").write_text("
            + repr(json.dumps(report))
            + "); print('private lifecycle child')"
        )
        return [
            {
                "name": "private_child",
                "argv": [sys.executable, "-c", code],
                "expected": 0,
                "deadline": 20,
                "scope": "owned",
            }
        ]

    monkeypatch.setattr(runner, "manifest", private_plan)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--date", "20261010", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert value["guard_ready_score"] == 0 and value["verdict_class"] == "null"
    assert reader_receipt(e.TASK, output.parent, field="required_checks_passed", expected=True)[
        "passed"
    ]
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert read_bound_sidecar(output, Path(terminal["publication"]["sidecar_path"]))["report"][
        "passed"
    ]
    assert all(r["passed"] for r in terminal["checks"])
    assert runner.main(["--cold-replay", str(output)]) == 0
    with pytest.raises(SystemExit, match="date must"):
        runner.main(["--date", "20261009"])
    with pytest.raises(SystemExit, match="date must"):
        runner.main(["--date"])
    # A real missing-root CLI covers the thin runner outside the checkout.
    receipt = child(
        "blocked_cli",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--date",
            "20261010",
            "--root",
            str(tmp_path / "absent-root"),
            "--output",
            str(tmp_path / "blocked" / (e.NAME + ".json")),
        ],
        tmp_path / "cli-logs",
        deadline=120,
    )
    assert receipt["passed"]
    blocked = json.loads((tmp_path / "blocked" / (e.NAME + ".json")).read_bytes())
    assert blocked["verdict_class"] == "blocked" and blocked["gate_check_summary"]
    # Freeze the full production command shape separately from the tiny
    # lifecycle fixture, including its private E2E consumer requirement.
    monkeypatch.undo()
    plan = runner.manifest(tmp_path / "manifest") if (tmp_path / "manifest").mkdir() is None else []
    assert any(s["name"] == "private_E2E018_guard" for s in plan)
    assert not any(s["argv"][-1] == "tests/python" for s in plan)
    assert base.e is not e


def test_cold_primitive_control(measured):
    """SCENARIO-REPORT-8362-REPLAY: a real child rejects self-consistently rehashed vector evidence."""
    work, receipts, raw, output = measured
    controls = runner.controls(e.build(work, receipts, raw, output), raw / "primitive-control")
    assert all(r["passed"] for r in controls)
    assert any(r["name"] == "cold_primitive_tamper" and r["exit_code"] == 1 for r in controls)
