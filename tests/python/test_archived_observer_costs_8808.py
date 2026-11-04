"""REQ-REPORT-8808, REQ-VERIFY-8808: original authority precedes timing."""

from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import archived_observer_costs_8808 as e
from carnot.reporting import archived_observer_runner_8808 as runner


def test_orders_and_absolute_gate():
    """SCENARIO-REPORT-8808-WINDOWS: neither sign of instability is a gain."""
    rows = e.roster()
    assert len(rows) == 36 and rows == e.roster()
    for arm in e.ARMS:
        assert {
            tuple(
                r["condition"]
                for r in rows
                if (r["arm"], r["repeat"], r["offset"]) == (arm, rep, off)
            )
            for rep, off in itertools.product(range(2), e.OFFSETS)
        } == set(itertools.permutations(e.CONDITIONS))
    measured = [dict(r, status="completed", complete_ns=100) for r in rows]
    assert all(r["disturbance_passed"] for r in e.pairs(measured))
    measured[0]["complete_ns"] = 59
    assert not all(r["disturbance_passed"] for r in e.pairs(measured))
    assert not any(r["disturbance_passed"] for r in e.pairs([]))


def test_immutable_resolution(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8808-AUTHORITY: mutable vNEXT is never an operand."""
    source = tmp_path / "original.md"
    source.write_text("original preserved bytes")
    ref = e.reference(source)
    monkeypatch.setattr(e, "AUTHORITIES", {"751": ref, "752": ref})
    called = []

    def original():
        called.append(e.previous.previous.identity.contract.DESIGN)
        return dict(refs=[], failures=[])

    monkeypatch.setattr(e.previous, "inputs", original)
    value = e.inputs()
    assert not value["failures"]
    assert called == [str(source)]
    source.write_text("changed bytes")
    assert e.inputs()["failures"][0]["operand_state"] == "rejected_producer"
    source.unlink()
    assert e.inputs()["failures"][0]["operand_state"] == "missing_file"


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    """SCENARIO-REPORT-8808-IMMUTABLE: fixtures cannot authorize production."""
    path = tmp_path_factory.mktemp("blocked-operands") / "packet.json"
    e.atomic_json(
        path,
        dict(
            refs=[],
            failures=[
                e.failure(
                    dict(path="/private/missing", sha256=None),
                    "contract",
                    "authenticated",
                    None,
                    "missing_file",
                )
            ],
        ),
    )
    return path


@pytest.fixture(scope="module")
def private(packet, tmp_path_factory):
    """SCENARIO-VERIFY-8808-SEALED: actual CLI bytes pass terminal checks."""
    tmp_path = tmp_path_factory.mktemp("blocked-cli")
    output = tmp_path / (e.NAME + ".json")
    receipt = runner.child(
        "private",
        [sys.executable, "-u", e.CLI, "--private-input", str(packet), "--output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    return output


def test_private_replay_and_tampering(private, tmp_path):
    """SCENARIO-REPORT-8808-REPLAY: replay rebuilds, never trusts aggregates."""
    value = json.loads(private.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["intended_count"] == value["censored_count"] == 36
    assert value["component_cost_ready_score"] == value["observer_rows_ready_score"] == 0
    assert runner.main(["--cold-replay", str(private)]) == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), "--cold-replay", str(private)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    changed = tmp_path / "changed.json"
    for field, new in [("task_id", "exp8808-alias"), ("component_cost_ready_score", 1)]:
        e.atomic_json(changed, dict(value, **{field: new}))
        assert runner.main(["--cold-replay", str(changed)]) == 1
    primitive = Path(value["primitive_reference"]["path"])
    original_bytes = primitive.read_bytes()
    work = e.read(value["primitive_reference"])
    work["failures"][0]["observed"] = "forged"
    e.atomic_json(primitive, work)
    e.atomic_json(changed, dict(value, primitive_reference=e.reference(primitive)))
    assert runner.main(["--cold-replay", str(changed)]) == 1
    primitive.write_bytes(original_bytes)


def test_build_failure_and_cli_guards(private, packet, tmp_path):
    """REQ-VERIFY-8808: failed owned checks cannot become external blocking."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    assert e.build(work, [], private)["verdict_class"] == "disqualified"
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--worker", str(packet)]) == 1
    assert runner.main(["--private-input", str(packet)]) == 1
    assert (
        runner.main(
            [
                "--private-input",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / (e.NAME + ".json")),
            ]
        )
        == 1
    )


@pytest.fixture(scope="module")
def durable_packet(tmp_path_factory):
    """SCENARIO-REPORT-8808-WINDOWS: reuse original private durable operands."""
    from test_durable_cost_components_8784 import packet as original_packet

    path = original_packet.__wrapped__(tmp_path_factory)
    admitted = e.read(e.reference(path))
    admitted["private_roster"] = [r for r in e.roster() if r["repeat"] == 0 and r["offset"] == 0]
    e.atomic_json(path, admitted)
    return path


@pytest.fixture(scope="module")
def successful(durable_packet, tmp_path_factory):
    """SCENARIO-VERIFY-8808-SEALED: real children cover all observer conditions."""
    root = tmp_path_factory.mktemp("observer-success")
    output = root / (e.NAME + ".json")
    receipt = runner.child(
        "success",
        [
            sys.executable,
            "-u",
            e.CLI,
            "--private-input",
            str(durable_packet),
            "--output",
            str(output),
        ],
        root / "logs",
        deadline=450,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    return output


def test_real_workers_states_and_components(successful):
    """SCENARIO-REPORT-8808-WINDOWS: observer arms retain every state and output."""
    value = json.loads(successful.read_bytes())
    assert value["completed_count"] == 6
    assert all(r["passed"] for r in value["state_parity_rows"])
    assert all(r["passed"] for r in value["parent_clock_reconciliation_rows"])
    assert value["positive_control_rows"][0]["passed"]
    assert value["component_cost_ready_score"] == 0
    for row in value["rows"]:
        assert row["restart_verified"] and len(row["update_rows"]) == 32
        assert all(bool(s["spans"]) == (row["condition"] == "clock") for s in row["segments"])
        assert all(
            bool(s["observer_dispatches"]) == (row["condition"] != "plain") for s in row["segments"]
        )
    assert e.projection([10**6] * 4, 10**8)["admitted"]
    assert not e.projection([10**11] * 4, 5 * 10**11)["admitted"]
    work = e.read(value["primitive_reference"])
    source = {(r["arm"], r["condition"]): r for r in work["window_rows"]}
    equal_parent_ns = (
        max(
            sum(c["ended_monotonic_ns"] - c["started_monotonic_ns"] for c in row["commands"])
            for row in source.values()
        )
        + 100
    )
    work["roster"] = e.roster()
    work["window_rows"] = [
        dict(deepcopy(source[r["arm"], r["condition"]]), **r, complete_ns=equal_parent_ns)
        for r in e.roster()
    ]
    work["private_control"] = False
    receipts = [dict(name="coverage_report", passed=True)]
    report = e.build(work, receipts, successful)
    assert report["observer_rows_ready_score"] == report["component_cost_ready_score"] == 1
    work["window_rows"][0]["segments"][0]["checkpoint"]["cursor"] = -1
    assert e.build(work, receipts, successful)["verdict_class"] == "disqualified"


def test_measure_resource_failure(packet, tmp_path, monkeypatch):
    """REQ-VERIFY-8808: an unchanged unavailable resource blocks before probing."""
    monkeypatch.setattr(
        runner.runtime, "resources", lambda path: dict(disk_backed=False, tools_available=False)
    )
    work = runner.measure(tmp_path / "blocked", "20261104", packet)
    assert len(work["failures"]) == 2
    expected = [dict(name="coverage_report", passed=True)]

    def validate(raw):
        assert raw == tmp_path and runner.checks.e is e
        return expected

    monkeypatch.setattr(runner.checks, "validate", validate)
    assert runner.validate(tmp_path) == expected


def test_public_health_is_separate(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8808: global debt cannot be relabeled as owned success."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    work["raw"] = str(tmp_path / "public-control-raw")
    monkeypatch.setattr(runner, "measure", lambda *args: deepcopy(work))
    monkeypatch.setattr(runner, "validate", lambda raw: [dict(name="coverage_report", passed=True)])
    monkeypatch.setattr(
        runner, "child", lambda *args, **kwargs: dict(passed=False, scope="repository_health")
    )
    monkeypatch.setattr(runner, "terminal", lambda *args: dict(passed=True))
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(output, "20261104") == 0
    assert not json.loads(output.read_bytes())["repository_health_reference"][
        "full_suite_pass_claimed"
    ]
    monkeypatch.setattr(
        runner, "validate", lambda raw: [dict(name="coverage_report", passed=False)]
    )
    assert runner.run(output, "20261104") == 1


def test_fork_failure_timeout_and_direct_worker(durable_packet, tmp_path, monkeypatch):
    """REQ-VERIFY-8808: real child failure and timeout retain their receipts."""
    missing = [
        sys.executable,
        e.CLI,
        "--worker",
        str(tmp_path / "missing"),
        "--work-dir",
        str(tmp_path / "work"),
    ]
    assert not runner.segment_child("missing", missing, tmp_path / "logs")["passed"]
    import time

    with monkeypatch.context() as patcher:
        patcher.setattr(runner, "worker", lambda *args: time.sleep(1))
        assert not runner.segment_child("timeout", missing, tmp_path / "logs", deadline=0.01)[
            "passed"
        ]
    admitted = e.read(e.reference(durable_packet))
    request = tmp_path / "request.json"
    e.atomic_json(request, dict(e.request(admitted, 0, "python", 4), condition="plain"))
    assert runner.main(["--worker", str(request), "--work-dir", str(tmp_path / "direct")]) == 0


def test_admission_uses_no_mutable_alias(monkeypatch):
    """SCENARIO-REPORT-8808-IMMUTABLE: original checks run with a vNEXT read trap."""
    mutable = e.ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
    original_read = Path.open

    def checked(path, *args, **kwargs):
        assert path != mutable, "mutable_vNEXT_read"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked)
    value = e.inputs()
    assert not value["failures"]
    assert all(r["passed"] for r in value["retained_controls"]["restart_rows"])
    assert len(value["retained_controls"]["threshold_control_rows"]) == 15


def test_public_measurement_cites_exact_operands(private, tmp_path, monkeypatch):
    """REQ-REPORT-8808: current citations bind the producer fields actually imported."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    raw = tmp_path / "public-measure-control"
    work["raw"] = str(raw)
    monkeypatch.setattr(e, "inputs", lambda: dict(refs=[], failures=work["failures"]))
    monkeypatch.setattr(runner.runtime, "measure", lambda *args: deepcopy(work))
    observed = runner.measure(raw, "20261104")
    assert not observed["private_control"]
    assert [r[0] for r in observed["citations"]] == [8783, 8784, 8796]
    assert all(r[1] and r[2]["sha256"].startswith("sha256:") for r in observed["citations"])
