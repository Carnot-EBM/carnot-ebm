"""REQ-REPORT-8796, REQ-VERIFY-8796: blocked admission cannot manufacture timing."""

from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import observer_intervention_8796 as e
from carnot.reporting import observer_intervention_runner_8796 as runner


def test_frozen_orders_and_absolute_disturbance():
    """SCENARIO-REPORT-8796-DISTURBANCE: faster observers can be unstable too."""
    rows = e.roster()
    assert rows == e.roster() and len(rows) == 36
    assert {r["offset"] for r in rows} == {0, 2032, 4064}
    for arm in e.ARMS:
        orders = [
            tuple(
                r["condition"]
                for r in rows
                if r["arm"] == arm and r["repeat"] == rep and r["offset"] == offset
            )
            for rep in range(2)
            for offset in e.OFFSETS
        ]
        assert set(orders) == set(itertools.permutations(e.CONDITIONS))
    observed = [dict(r, complete_ns=100, status="completed") for r in rows]
    observed[1]["complete_ns"] = 59
    pairs = e.pairs(observed)
    assert len(pairs) == 12 and any(not p["disturbance_passed"] for p in pairs)
    assert any(
        p["clock_absolute_fraction"] == pytest.approx(0.41)
        or p["sham_absolute_fraction"] == pytest.approx(0.41)
        for p in pairs
    )
    assert not any(p["disturbance_passed"] for p in e.pairs([]))


@pytest.fixture
def packet(tmp_path):
    """SCENARIO-REPORT-8796-BLOCKED: private failures never read real primaries."""
    path = tmp_path / "packet.json"
    e.atomic_json(
        path,
        dict(
            refs=[],
            failures=[
                e.failure(
                    dict(path="/private/absent", sha256=None),
                    "contract",
                    "authenticated",
                    None,
                    "missing_file",
                )
            ],
        ),
    )
    return path


@pytest.fixture
def private(packet, tmp_path):
    """SCENARIO-VERIFY-8796-REPLAY: exercise the actual CLI and terminal scanners."""
    output = tmp_path / (e.NAME + ".json")
    receipt = runner.child(
        "private_cli",
        [sys.executable, "-u", e.CLI, "--private-input", str(packet), "--output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    return output


def test_private_cli_cold_replay_and_rehashed_tampering(private, tmp_path):
    """SCENARIO-VERIFY-8796-REPLAY: independent seals precede summary rebuilding."""
    value = json.loads(private.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 36
    assert value["observer_rows_ready_score"] == value["component_cost_ready_score"] == 0
    assert len(value["paired_observer_rows"]) == 12
    assert runner.main(["--cold-replay", str(private)]) == 0
    changed = tmp_path / "changed.json"
    for field, observed in [("task_id", "exp8796-alias"), ("observer_rows_ready_score", 1)]:
        e.atomic_json(changed, dict(value, **{field: observed}))
        assert runner.main(["--cold-replay", str(changed)]) == 1
    primitive = Path(value["primitive_reference"]["path"])
    saved = primitive.read_bytes()
    work = json.loads(saved)
    work["failures"][0]["observed"] = "tampered"
    e.atomic_json(primitive, work)
    e.atomic_json(changed, dict(value, primitive_reference=e.reference(primitive)))
    assert runner.main(["--cold-replay", str(changed)]) == 1
    primitive.write_bytes(saved)


def test_build_and_scope(private):
    """REQ-REPORT-8796: failed owned validation disqualifies even blocked science."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    assert e.build(work, [], private)["verdict_class"] == "disqualified"
    work["private_control"] = False
    assert (
        e.build(work, [dict(name="coverage_report", passed=True)], private)["verdict_class"]
        == "blocked"
    )
    assert e.build(work, [dict(name="coverage_report", passed=False)], private)[
        "flagged_adversarial"
    ]
    work["failures"] = []
    with pytest.raises(ValueError, match="failed_admission_required"):
        e.build(work, [dict(name="coverage_report", passed=True)], private)


def test_bad_cli_and_private_scope(packet, tmp_path):
    """REQ-VERIFY-8796: malformed or production fixture writes fail closed."""
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--private-input", str(packet)]) == 1
    assert (
        runner.main(
            [
                "--private-input",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / (e.NAME + ".json")),
            ]
        )
        == 1
    )


def test_blocked_measurement_refuses_admitted_input(packet, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8796-BLOCKED: do not silently reuse the old observer."""
    admitted = dict(failures=[], refs=[])
    monkeypatch.setattr(e, "inputs", lambda: admitted)
    with pytest.raises(ValueError, match="qualified_observer_execution_required"):
        runner.measure(tmp_path / "admitted", "20261104")
    monkeypatch.setattr(
        runner, "resources", lambda p: dict(disk_backed=False, tools_available=False)
    )
    work = runner.measure(tmp_path / "missing-resources", "20261104", packet)
    assert len(work["failures"]) == 2 and not work["resource_projection"]["admitted"]


def test_owned_validation_delegation(tmp_path, monkeypatch):
    """REQ-VERIFY-8796: the existing validator receives exactly this owned scope."""
    expected = [dict(name="coverage_report", passed=True)]

    def check(raw):
        assert raw == tmp_path and runner.checks.e is e
        assert runner.checks.E2E == runner.E2E
        return expected

    monkeypatch.setattr(runner.checks, "validate", check)
    assert runner.validate(tmp_path) == expected


@pytest.mark.parametrize(
    "mode", ["valid", "missing", "field", "zero", "alias", "terminal", "rows", "array"]
)
def test_upstream_authentication(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8796: keep absent, zero and rejected source states distinct."""
    source = tmp_path / "upstream.json"
    primitives = tmp_path / "primitives.json"
    e.atomic_json(primitives, dict(window_rows=[dict(index=i) for i in range(36)]))
    prior = dict(
        experiment_id=8784,
        task_id=e.previous.TASK,
        durable_cost_rows_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        rows=[dict(index=i) for i in range(36)],
        observer_overhead_rows=[dict(overhead_fraction=-0.41, passed=True)],
        primitive_reference=e.reference(primitives),
        terminal_validation_sidecar_path=str(tmp_path / "terminal.json"),
    )
    if mode == "field":
        prior.pop("durable_cost_rows_ready_score")
    if mode == "zero":
        prior["durable_cost_rows_ready_score"] = 0
    if mode == "alias":
        prior["task_id"] = "alias"
    if mode == "rows":
        prior["rows"] = []
    e.atomic_json(source, [] if mode == "array" else prior)
    pin = e.reference(source)["sha256"]
    e.atomic_json(
        tmp_path / "terminal.json",
        dict(passed=mode != "terminal", publication=dict(primary_sha256=pin)),
    )
    monkeypatch.setattr(e, "UPSTREAM", source)
    monkeypatch.setattr(e, "PIN", pin)
    monkeypatch.setattr(e.previous, "inputs", lambda: dict(failures=[], refs=[]))
    if mode == "missing":
        source.unlink()
    result = e.inputs()
    assert bool(result["failures"]) == (mode != "valid")
    if mode == "valid":
        assert not result["historical_observer_diagnostics"][0]["absolute_disturbance_passed"]
    if result["failures"]:
        assert set(
            ["upstream", "path", "sha256", "check", "field", "operator", "expected", "observed"]
        ).issubset(result["failures"][0])


def test_public_orchestration_separates_health(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8796: repository health is never rewritten as a scoped pass."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    monkeypatch.setattr(runner, "measure", lambda *a: deepcopy(work))
    monkeypatch.setattr(runner, "validate", lambda raw: [dict(name="coverage_report", passed=True)])
    monkeypatch.setattr(
        runner, "child", lambda *a, **kw: dict(passed=False, scope="repository_health")
    )
    monkeypatch.setattr(runner, "terminal", lambda *a: dict(passed=True))
    out = tmp_path / (e.NAME + ".json")
    assert runner.run(out, "20261104") == 0
    assert not json.loads(out.read_bytes())["repository_health_reference"][
        "full_suite_pass_claimed"
    ]
    monkeypatch.setattr(
        runner, "validate", lambda raw: [dict(name="coverage_report", passed=False)]
    )
    assert runner.run(out, "20261104") == 1


def test_replay_checks_clock_validation_seals_and_child_logs(private, tmp_path):
    """SCENARIO-VERIFY-8796-REPLAY: actual child streams and clock operands are bound."""
    value = json.loads(private.read_bytes())
    raw = Path(value["primitive_reference"]["path"]).parent
    work = e.read(value["primitive_reference"])
    saved = {
        p: p.read_bytes()
        for p in [
            raw / "measurement.json",
            raw / "measurement_seal.json",
            raw / "validation.json",
            raw / "validation_seal.json",
        ]
    }
    candidate = tmp_path / "replay.json"
    receipt = runner.child(
        "tiny_child",
        [sys.executable, "-c", "print('child receipt')"],
        tmp_path / "logs",
        deadline=30,
    )
    try:
        e.atomic_json(raw / "validation.json", dict(receipts=[receipt]))
        e.atomic_json(
            raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json"))
        )
        updated = e.build(work, [receipt], Path(value["publication_output"]))
        e.atomic_json(candidate, updated)
        assert runner.main(["--cold-replay", str(candidate)]) == 0
        e.atomic_json(
            raw / "validation_seal.json", dict(validation=dict(path="wrong", sha256="wrong"))
        )
        assert runner.main(["--cold-replay", str(candidate)]) == 1
        e.atomic_json(
            raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json"))
        )
        work["duration_s"] += 1
        runner.seal(work)
        e.atomic_json(candidate, e.build(work, [receipt], Path(value["publication_output"])))
        assert runner.main(["--cold-replay", str(candidate)]) == 1
    finally:
        for path, content in saved.items():
            path.write_bytes(content)


def test_failed_source_bytes_are_retained(packet, tmp_path):
    """SCENARIO-REPORT-8796-BLOCKED: changed authority remains available for diagnosis."""
    admitted = e.read(e.reference(packet))
    source = tmp_path / "rejected-contract.md"
    source.write_text("a different contract")
    admitted["failures"][0]["path"] = str(source)
    e.atomic_json(packet, admitted)
    work = runner.measure(tmp_path / "retained", "20261104", packet)
    frozen = next(r for r in work["source_refs"] if r.get("original_path") == str(source))
    assert Path(frozen["path"]).read_bytes() == source.read_bytes()
