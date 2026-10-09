"""REQ-REPORT-8328 / REQ-VERIFY-8328: qualification, live joins and private CLI."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_frontier_8328 as m
from carnot.reporting import arc_supervisor_execution_8328 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from test_arc_authoritative_frontier_8215 import fixture
from test_arc_supervisor_frontier_8243 import current


def test_qualified_empty_and_native_delta(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8328-DELTA: sealed qualification and exact clock are reused."""
    previous = m.authenticate(m.FRONTIER)
    locator = tmp_path / "locator.json"
    atomic_json(locator, m.reader.authority.discover())
    value = m.inspect(locator, previous, tmp_path / "adapter")
    assert value["unchanged_authority"] and value["new_outcome_count"] == 0
    rows = [
        current(g, a, a == "drop_goal_bias")
        for g in ("cd82", "r11l", "ar25")
        for a in ("drop_goal_bias", "allow_reinduction")
        for _ in range(5)
    ]
    for i, row in enumerate(rows):
        row.update(seed=i, finished_at="2026-10-09T01:00:00Z")
    delta = m.inspect(fixture(tmp_path / "native", rows), previous, tmp_path / "new")
    assert delta["new_outcome_count"] == 30 and delta["independent_count"] == 3
    assert delta["proposed_arm_change"] and delta["selection_propensity"]["status"] == "missing"
    assert m.reader.summarize(delta["rows"][1:], {})["proposed_arm_change"] is None
    assert m.recommend(delta)[0]["causal_superiority"] is False
    assert m.recommend(value) == []


@pytest.mark.parametrize("mutation", ["missing", "bytes", "sidecar", "qualification", "code"])
def test_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    """SCENARIO-REPORT-8328-DELTA: missing and altered evidence cannot qualify."""
    source = m.FRONTIER
    if mutation == "missing":
        source = tmp_path / source.name
    elif mutation == "bytes":
        source = tmp_path / source.name
        source.write_text("{}")
    elif mutation == "sidecar":
        monkeypatch.setattr(m, "read_bound_sidecar", lambda *a: {})
    elif mutation == "qualification":
        monkeypatch.setattr(m, "PIN", sha256_file(source))
        monkeypatch.setattr(m, "json_document", lambda p: {})
    else:
        monkeypatch.setattr(m, "READERS", ["absent.py"])
    with pytest.raises((OSError, ValueError, KeyError)):
        m.authenticate(source)


def test_measure_build_replay_and_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8328-REPLAY: semantic recomputation rejects rehashed invention."""
    work = m.measure(tmp_path / "raw", tmp_path)
    output = tmp_path / e.NAME.replace(".py", ".json")
    receipts = [dict(name="proof", passed=True)]
    value = e.build(work, receipts, tmp_path / "raw", output)
    assert e.replay(value) and value["arc_reader_ready_score"] == 1
    for change in [
        dict(new_outcome_count=1),
        dict(arc_reader_ready_score=0),
        dict(solve_claims=["fake"]),
        dict(MODEL_SPECS=[{}]),
    ]:
        assert not e.replay(dict(value, **change))
    original = deepcopy(work)
    work["delta"]["new_outcome_count"] = 99
    atomic_json(Path(value["work_reference"]["path"]), work)
    value["work_reference"]["sha256"] = sha256_file(Path(value["work_reference"]["path"]))
    value["reproducibility_checksum"] = canonical_hash(work)
    assert not e.replay(value)
    atomic_json(Path(value["work_reference"]["path"]), original)
    failed = e.build(original, [dict(name="failure", passed=False)], tmp_path / "raw", output)
    assert failed["verdict_class"] == "disqualified" and e.replay(failed)
    monkeypatch.setattr(m, "FRONTIER", tmp_path / "missing.json")
    blocked = m.measure(tmp_path / "blocked", tmp_path)
    result = e.build(blocked, receipts, tmp_path / "blocked", output)
    assert result["verdict_class"] == "blocked" and result["gate_check_summary"]
    assert e.replay(result)


def test_children_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8328-CLI: real failures, timeout and recovery retain receipts."""
    for name, code, deadline, expected in [
        ("fail", "raise SystemExit(1)", 5, 1),
        ("timeout", "while True: pass", 0.02, 0),
        ("recover", "print('ok', flush=True)", 5, 0),
    ]:
        receipt = child(
            name,
            [sys.executable, "-u", "-c", code],
            tmp_path,
            deadline=deadline,
            expected=expected,
            heartbeat=0.01,
        )
        assert receipt["passed"] == (name != "timeout")
    runner = runpy.run_path(str(m.ROOT / e.CLI))
    main = runner["main"]
    with pytest.raises(SystemExit):
        main(["--date", "20261008"])
    with pytest.raises(SystemExit):
        main(["--private-e2e", "--output", str(e.OUTPUT)])
    output = tmp_path / e.OUTPUT.name
    assert main(["--private-e2e", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["fixture_claim_scope"] and value["arc_reader_ready_score"] == 0
    receipt = child(
        "cold", [sys.executable, "-u", str(m.ROOT / e.CLI), "--cold-replay", str(output)], tmp_path
    )
    assert receipt["passed"]
    value["solve_claims"] = ["invented"]
    atomic_json(output, value)
    assert main(["--cold-replay", str(output)]) == 1
    monkeypatch.setattr(
        e,
        "plan",
        lambda p: [
            dict(
                name="failed",
                argv=[sys.executable, "-c", "exit(1)"],
                deadline=5,
                expected=0,
                scope="owned",
            )
        ],
    )
    assert main(["--output", str(tmp_path / "failure" / e.OUTPUT.name)]) == 1

    def covered_plan(p: Path) -> list[dict[str, Any]]:
        atomic_json(p / "coverage.json", {})
        (p / ".coverage").write_bytes(b"private branch control")
        return []

    monkeypatch.setattr(e, "plan", covered_plan)
    assert main(["--output", str(tmp_path / "recovery" / e.OUTPUT.name)]) == 0


def test_failure_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8328-CLI: preconditions and terminal recovery fail closed."""
    specs = e.plan(tmp_path)
    assert {"coverage_report", "private_E2E017", "private_E2E018_consumers"} <= {
        s["name"] for s in specs
    }
    assert all(s["deadline"] <= 240 for s in specs)
    work = m.measure(tmp_path / "raw", tmp_path)
    output = tmp_path / e.OUTPUT.name
    value = e.build(work, [], tmp_path / "raw", output)
    assert not e.replay(dict(value, work_reference=dict(path="/tmp/absent-8328", sha256="bad")))
    assert not e.replay(dict(value, work_reference=dict(value["work_reference"], sha256="bad")))
    assert not e.replay(dict(value, reproducibility_checksum="bad"))
    saved = next(iter(work["snapshots"]))
    original = Path(saved).read_bytes()
    Path(saved).write_bytes(b"tampered")
    assert not e.replay(value)
    Path(saved).write_bytes(original)
    first = True
    original_child = e.child

    def fail_once(name: str, argv: list[str], logs: Path, **kwargs: Any) -> dict[str, Any]:
        nonlocal first
        if name == "terminal_cold" and first:
            first = False
            argv = [sys.executable, "-u", "-c", "raise SystemExit(1)"]
        return original_child(name, argv, logs, **kwargs)

    monkeypatch.setattr(e, "child", fail_once)
    e.publish(value, work, output, tmp_path / "raw")
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        e, "publish_primary", lambda *a: (_ for _ in ()).throw(ValueError("other_failure"))
    )
    with pytest.raises(ValueError, match="other_failure"):
        e.publish(value, work, output, tmp_path / "raw")
    monkeypatch.setattr(
        m.authority, "authority", lambda *a: dict(activated=False, gate_check_summary=[])
    )
    assert m.measure(tmp_path / "bad-authority", tmp_path)["failures"]
    assert m.measure(tmp_path / "bad-scratch", m.ROOT)["failures"]
    original_document = m.json_document
    monkeypatch.setattr(
        m,
        "json_document",
        lambda p: (
            dict(original_document(p), primary_sha256="changed")
            if p.name == "terminal_reports.json"
            else original_document(p)
        ),
    )
    with pytest.raises(ValueError, match="terminal_report"):
        m.authenticate(m.FRONTIER)


def test_missing_tools_and_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8328-CLI: absent operands block before measurement."""
    monkeypatch.setattr(e, "plan", lambda p: [])
    original = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda p: False if p == m.ROOT / ".venv/bin/ruff" else original(p)
    )
    output = tmp_path / e.OUTPUT.name
    assert e.run(output, tmp_path) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    monkeypatch.setattr(m.native.authority, "REGISTRY", tmp_path / "absent-registry.yaml")
    assert m.measure(tmp_path / "missing-registry", tmp_path)["failures"]
