"""REQ-VERIFY-8381 / REQ-REPORT-8381: rational actions have a separate contract."""

from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.verify import dyadic_logit_v1 as n
from carnot.reporting import logit_policy_certificate_8381 as e
from carnot.reporting import logit_policy_execution_8381 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def operands(tmp_path_factory):
    """REQ-REPORT-8381: private copies retain the actual historical operands."""
    return e.authenticate(e.ROOT, tmp_path_factory.mktemp("logit-inputs"))


def test_exact_regions_and_outward():
    """SCENARIO-VERIFY-8381-BOUND: ties escalate and outward conversion encloses."""
    for z, expected in [
        (-n.L - F(1, 2**52), "accept"),
        (-n.L, "escalate"),
        (n.L, "escalate"),
        (n.L + F(1, 2**52), "reject"),
    ]:
        assert n.region(z, z) == expected
    assert n.region(-n.L - 1, -n.L) is None
    assert n.region(n.L, n.L + 1) is None
    assert F(n.outward(F(1, 3), -1)) <= F(1, 3)
    assert F(n.outward(F(1, 3), 1)) >= F(1, 3)


def test_certificate_panel_and_faults(operands):
    """SCENARIO-VERIFY-8381-BOUND: independent polynomials reject unsafe bounds."""
    head, table = operands["head"], np.asarray(operands["table"], dtype="<i2")
    cert = n.certificate(head, table)
    assert cert["independent_remainder_audit"]
    rows = n.evaluate(head, table)
    assert sum(r["kind"] == "random" for r in rows) == 4096
    assert all(not r["interval_escape"] and r["action"] == r["reference_action"] for r in rows)
    assert any(r["kind"] == "exact_threshold" for r in rows)
    assert all(r["action"] == "escalate" for r in rows if r["kind"] == "exact_threshold")
    assert sum(r["fast_path"] for r in rows[:4096]) / 4096 >= 0.5
    x = [F(0), *[F(1, 2)] * 4]
    under = deepcopy(cert)
    under["cells"][0][0] = "0"
    for supplied, data in [
        (None, table),
        (under, table),
        (dict(cert, rounding="nearest"), table),
        (dict(cert, version="stale"), table),
        (cert, table[::-1]),
        (cert, np.full((4, 65), 32767, dtype="<i2")),
        (cert, table.astype(float)),
    ]:
        assert not n.guard(head, x, data, supplied)["fast_path"]
    assert not n.guard(dict(head, temperature=3), x, table, cert)["fast_path"]
    assert not n.guard(head, [F(10**400), *x[1:]], table, cert)["fast_path"]
    with pytest.raises(ValueError):
        n.guard(head, [F(0)] * 4, table, cert)
    with pytest.raises(ValueError):
        n.guard(head, [F(0), F(-1), *x[2:]], table, cert)


def test_missing_and_child(tmp_path):
    """SCENARIO-VERIFY-8381-CLI: actual child failure and absence stay distinct."""
    with pytest.raises(e.OperandError):
        e.authenticate(tmp_path, tmp_path / "raw")
    r = child("deliberate", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path, deadline=10)
    assert not r["passed"] and r["exit_code"] == 7


def test_measure_build_replay(operands, tmp_path):
    """SCENARIO-REPORT-8381-REPLAY: repaired hashes cannot change scientific meaning."""
    work = e.measure(operands, tmp_path)
    atomic_json(tmp_path / "manifest.json", {"commands": []})
    work.update(
        manifest_reference=e.reference(tmp_path / "manifest.json"),
        code_refs=[],
        failures=[],
        coverage={
            "totals": {"percent_covered": 100},
            "files": {p: {"summary": {"missing_lines": 0}} for p in e.OWNED},
        },
    )
    receipts = [child("real", [sys.executable, "-c", "print('real')"], tmp_path / "logs")]
    atomic_json(tmp_path / "measurement.json", work)
    output = tmp_path / "results" / (e.NAME + ".json")
    value = e.build(work, receipts, tmp_path, output)
    assert value["verdict_class"] == "circular_positive" and value["action_certificate_ready_score"]
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    assert not e.replay(tmp_path / "absent")
    changed = dict(value, action_certificate_ready_score=0)
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(path, changed)
    assert not e.replay(path)
    assert (
        e.build(work, [dict(receipts[0], passed=False)], tmp_path, output)["verdict_class"]
        == "disqualified"
    )
    assert (
        e.build(
            dict(work, failures=[{"field": "missing"}], numeric_summary={}, primitive_refs=[]),
            [],
            tmp_path,
            output,
        )["verdict_class"]
        == "blocked"
    )
    assert (
        e.build(dict(work, coverage={}), receipts, tmp_path, output)["verdict_class"]
        == "disqualified"
    )
    e.write_note(value, tmp_path / "note.md")
    assert "dyadic_logit_v1" in (tmp_path / "v722-logit-policy-certificate.md").read_text()


def test_cli_lifecycle(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8381-CLI: real private publication includes bounded cold children."""
    frozen_plan = runner.manifest(tmp_path / "frozen")
    assert any(r["name"] == "threshold_consumers" for r in frozen_plan)
    # This lifecycle fixture intentionally has no qualifying validation report.
    # Production uses the genuine frozen coverage commands, which run separately.
    monkeypatch.setattr(runner, "manifest", lambda private: [])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--date", "20261010", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["action_certificate_ready_score"]
    assert runner.main(["--cold-replay", str(output)]) == 0
    with pytest.raises(SystemExit, match="date must"):
        runner.main(["--date", "20261009"])
    with pytest.raises(SystemExit, match="date must"):
        runner.main(["--date"])
    private_output = tmp_path / "missing-results" / (e.NAME + ".json")
    receipt = child(
        "missing_cli",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(tmp_path / "absent-root"),
            "--output",
            str(private_output),
        ],
        tmp_path / "cli-logs",
        deadline=120,
    )
    assert receipt["passed"]
    blocked = json.loads(private_output.read_bytes())
    assert blocked["verdict_class"] == "blocked" and blocked["completed_count"] == 0
    assert blocked["gate_check_summary"][0]["observed"] is False
    monkeypatch.setattr("pytest.main", lambda args: 0)
    assert runner.historical_consumers(tmp_path / "preserved-authority") == 0


def test_rehashed_primitive_faults(operands, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8381-REPLAY: every operand and log is independently sealed."""
    work = e.measure(operands, tmp_path)
    atomic_json(tmp_path / "manifest.json", {"commands": []})
    work.update(
        manifest_reference=e.reference(tmp_path / "manifest.json"), code_refs=[], failures=[]
    )
    receipts = [child("seal", [sys.executable, "-c", "print('seal')"], tmp_path / "logs")]
    output = tmp_path / "results" / (e.NAME + ".json")
    real = e.reconstruct
    primitives = json.loads(Path(work["primitive_refs"][0]["path"]).read_bytes())
    monkeypatch.setattr(e, "reconstruct", lambda inputs: primitives)
    for name in [
        "work_hash",
        "input_hash",
        "log_hash",
        "task",
        "head",
        "table",
        "updates",
        "primitive",
        "summary",
        "pin",
    ]:
        changed, rs = deepcopy(work), deepcopy(receipts)
        if name == "input_hash":
            changed["code_refs"] = [{"path": str(tmp_path / "manifest.json"), "sha256": "bad"}]
        if name == "log_hash":
            rs[0]["stdout_sha256"] = "bad"
        if name in ["head", "task", "table", "updates"]:
            if name == "head":
                changed["inputs"]["head"]["temperature"] = 3
            elif name == "task":
                changed["inputs"]["task"]["prompt"] += " changed"
            elif name == "table":
                changed["inputs"]["table"][0][0] += 1
            else:
                changed["inputs"]["refreshed_coefficients"][0][0] += 1
        if name == "pin":
            changed["inputs"]["refs"] = [
                r for r in changed["inputs"]["refs"] if r["source_sha256"] != e.PINS[e.UPSTREAM]
            ]
        if name == "primitive":
            bad = deepcopy(primitives)
            bad["rows"][0]["action"] = "changed"
            atomic_json(tmp_path / "changed-primitives.json", bad)
            changed["primitive_refs"] = [e.reference(tmp_path / "changed-primitives.json")]
        if name == "summary":
            changed["numeric_summary"]["fast_path_fraction"] = 0
        atomic_json(tmp_path / "measurement.json", changed)
        value = e.build(changed, rs, tmp_path, output)
        if name == "work_hash":
            value["work_reference"]["sha256"] = "bad"
        path = tmp_path / (name + "-candidate.json")
        atomic_json(path, value)
        assert not e.replay(path), name
    monkeypatch.setattr(e, "reconstruct", real)


def test_source_seal_failure_and_endpoint_overflow(operands, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8381-BOUND: invalid seals and infinite endpoints cannot certify."""
    head = dict(operands["head"], coefficients=[4.0, *operands["head"]["coefficients"][1:]])
    table = n.previous.old.table(head, 65, "int16")[0]
    x = [F(float(np.finfo(float).max)) / 2, *[F(1, 2)] * 4]
    assert not n.guard(head, x, table, n.certificate(head, table))["fast_path"]
    monkeypatch.setattr(
        e, "read_bound_sidecar", lambda *args: (_ for _ in ()).throw(ValueError("seal"))
    )
    with pytest.raises(e.OperandError, match="bound_terminal_sidecar"):
        e.authenticate(e.ROOT, tmp_path)


def test_actual_rounding_fault(operands, monkeypatch):
    """SCENARIO-VERIFY-8381-BOUND: the runtime checks enclosure independently of float conversion."""
    head, table = operands["head"], np.asarray(operands["table"], dtype="<i2")
    cert = n.certificate(head, table)
    x = [0.0, 0.5, 0.5, 0.5, 0.5]
    assert n.guard(head, x, table, cert)["fast_path"]
    monkeypatch.setattr(n, "outward", lambda value, direction: 1000.0 * -direction)
    row = n.guard(head, x, table, cert)
    assert not row["fast_path"] and row["fallback_reason"] == "altered_rounding_contract"
