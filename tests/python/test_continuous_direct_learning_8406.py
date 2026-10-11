"""REQ-REPORT-8406 / REQ-VERIFY-8406: qualify serving without reopening H2."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import continuous_direct_learning_8406 as e
from carnot.verify import continuous_direct_learning_8406 as k
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture
def bundle(tmp_path):
    """SCENARIO-VERIFY-8406-CAUSAL: constructed data qualifies mechanics only."""
    rows = [
        dict(slot=i, unit_id=str(i), source_cluster_id=str(i), x=[0, 0.2, 0.3, 0.4, 0.5])
        for i in range(1, 129)
    ]
    rows[13]["x"] = None
    labels = tmp_path / "labels.json"
    atomic_json(
        labels,
        dict(
            rows=[
                dict(slot=i, unit_id=str(i), source_cluster_id=str(i), y=i % 2)
                for i in range(1, 129)
            ]
        ),
    )
    return dict(
        head=dict(
            arm="spline34",
            coefficients=[0.0] * 34,
            temperature=2.0,
            knots=k.s.evaluator.kernel.KNOTS,
            degree=3,
        ),
        slots=rows,
        labels=e.reference(labels),
    )


def test_causal_state_and_feedback_controls(bundle, tmp_path):
    """SCENARIO-VERIFY-8406-CAUSAL: controls cannot silently alter the issued ledger."""
    state = k.run(bundle, tmp_path / "memory", "in_memory_uninterrupted", stop=12)
    assert len(state["issued"]) == 12 and len(state["applied"]) == 4
    assert len(state["pending"]) == 8
    assert state["head"]["coefficients"][:2] == [0.0, 0.0]
    assert any(state["head"]["coefficients"][2:])
    assert all(r["passed"] for r in k.controls(bundle, state))
    steps = k.read_steps(tmp_path / "memory")
    assert k.check_steps(bundle, steps) == state
    altered = deepcopy(steps)
    altered[-1]["state"]["head"]["coefficients"][2] += 0.1
    with pytest.raises(ValueError):
        k.check_steps(bundle, altered)


def test_real_hard_exits_and_exact_resume(bundle, tmp_path):
    """SCENARIO-VERIFY-8406-CHILD-COVERAGE: two actual exits preserve complete state."""
    source = tmp_path / "bundle.json"
    atomic_json(source, bundle)
    expected = k.run(bundle, tmp_path / "memory", "in_memory_uninterrupted")
    destination = tmp_path / "resumed"
    argv = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--worker",
        str(source),
        "--state-directory",
        str(destination),
        "--variant",
        "durable_crash_resume",
    ]
    for crash in (32, 64, 0):
        receipt = child(
            "crash" + str(crash),
            argv + ["--crash", str(crash)],
            tmp_path / "logs",
            expected=73 if crash else 0,
            deadline=300,
        )
        assert receipt["passed"]
    actual = json.loads((destination / "final.json").read_bytes())
    assert actual == expected
    assert k.check_steps(bundle, k.read_steps(destination)) == expected
    assert k.run(bundle, tmp_path / "durable", "durable_uninterrupted") == expected


def test_missing_gate_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-8406-CUSTODY: missing operands differ from zero readiness."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["continuous_service_ready_score"] == 0
    assert len(value["rows"]) == 864 and value["completed_count"] == 0
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    paths = {g["artifact_path"] for g in value["gate_check_summary"]}
    assert all(str(tmp_path / "absent" / name) in paths for name in e.PINS)
    assert e.build(work, [dict(passed=False)])["verdict_class"] == "disqualified"
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    bad = deepcopy(value)
    bad["completed_count"] += 1
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(candidate, bad)
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "missing")


def test_authenticate_pinned_natural_inputs(tmp_path):
    """SCENARIO-REPORT-8406-CUSTODY: current authority and old operands stay byte-bound."""
    work = dict(refs=[], gate_check_summary=[])
    natural = e.authenticate(e.ROOT, tmp_path, work)
    assert len(natural["slots"]) == 128
    assert sum(r["x"] is None for r in natural["slots"][:96]) == 22
    assert work["current_authority"]["activated"]
    assert natural["head"]["arm"] == "spline34"


@pytest.fixture
def natural_work(tmp_path):
    """SCENARIO-REPORT-8406-CUSTODY: private replay uses authentic source bytes."""
    raw = tmp_path / "natural"
    work = dict(refs=[], gate_check_summary=[])
    bundle = e.authenticate(e.ROOT, raw, work)
    directory = raw / "fixture_memory"
    k.run(bundle, directory, k.VARIANTS[0])
    work.update(
        root=str(e.ROOT),
        raw=str(raw),
        bundle=bundle,
        streams=[
            dict(seed=s, variant=v, directory=str(directory)) for s in k.SEEDS for v in k.VARIANTS
        ],
        child_receipts=[dict(passed=True)],
        phase_spans=[],
        duration_s=1.0,
        preconditions_checked=True,
        work_path=str(raw / "work.json"),
    )
    atomic_json(Path(work["work_path"]), work)
    return work


def test_positive_cold_replay_and_primitive_tamper(natural_work, tmp_path):
    """SCENARIO-REPORT-8406-CUSTODY: repaired hashes cannot replace causal operands."""
    value = e.build(natural_work, [dict(passed=True)])
    assert value["exact_resume_score"] == 1 and value["independent_count"] == 74
    assert value["completed_count"] == 666 and value["excluded_count"] == 198
    assert value["coefficient_change_count"] > 0
    assert len(value["retention_rows"]) == 1152 and len(value["pending_feedback_rows"]) == 72
    assert all(i >= 2 for i in value["touched_coefficients"])
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert e.build(natural_work, [dict(passed=False)])["verdict_class"] == "disqualified"
    changed = deepcopy(natural_work)
    changed["bundle"]["slots"][0]["x"][0] += 1
    atomic_json(Path(changed["work_path"]), changed)
    with pytest.raises(ValueError):
        e.build(changed, [dict(passed=True)])
    forged = deepcopy(value)
    forged["work_reference"] = e.reference(Path(changed["work_path"]))
    forged.pop("reproducibility_checksum")
    forged["reproducibility_checksum"] = canonical_hash(forged)
    atomic_json(candidate, forged)
    assert not e.replay(candidate)
    changed = deepcopy(natural_work)
    changed["current_authority"]["tasks"][0]["id"] = "foreign-task"
    atomic_json(Path(changed["work_path"]), changed)
    atomic_json(candidate, e.build(changed, [dict(passed=True)]))
    assert not e.replay(candidate)


def test_service_failure_paths(bundle, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8406-CAUSAL: owned corruption cannot become a successful recovery."""
    with pytest.raises(ValueError, match="service_configuration"):
        k.run(bundle, tmp_path / "bad", "unknown")
    steps_dir = tmp_path / "steps"
    state = k.run(bundle, steps_dir, k.VARIANTS[0], stop=12)
    altered = k.read_steps(steps_dir)
    altered[0]["issue"]["probability"] = 0.1
    with pytest.raises(ValueError, match="issued_state_drift"):
        k.check_steps(bundle, altered)
    with monkeypatch.context() as m:
        m.setattr(k.Labels, "release", lambda *args, **kw: dict(y=0))
        with pytest.raises(ValueError, match="tripwire_did_not_fire"):
            k.controls(bundle, state)
    with monkeypatch.context() as m:
        ticks = iter([0.0, 601.0])
        m.setattr(k.time, "monotonic", lambda: next(ticks))
        with pytest.raises(TimeoutError):
            k.run(bundle, tmp_path / "timeout", k.VARIANTS[0], stop=1)

    class WrongStore:
        def __init__(self, path, frozen):
            self.state = k.s.initial(frozen["head"])

        def initialize(self):
            pass

        def read(self):
            return self.state

        def apply(self, event):
            return dict(wrong=True)

    with monkeypatch.context() as m:
        m.setattr(k, "QualifiedStore", WrongStore)
        with pytest.raises(ValueError, match="acknowledged_state_drift"):
            k.run(bundle, tmp_path / "ack", k.VARIANTS[1], stop=1)
        m.setattr(WrongStore, "read", lambda self: {})
        with pytest.raises(ValueError, match="restart_state_drift"):
            k.run(bundle, tmp_path / "restart", k.VARIANTS[1], stop=1)
    original = k.s.transition

    def mutation(state, event):
        after, result = original(state, event)
        if event["kind"] == "feedback":
            after["issued"]["issue-9"]["probability"] = -1
        return after, result

    with monkeypatch.context() as m:
        m.setattr(k.s, "transition", mutation)
        with pytest.raises(ValueError, match="issued_prediction_mutation"):
            k.run(bundle, tmp_path / "mutation", k.VARIANTS[0], stop=9)


def test_authentication_failure_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8406-CUSTODY: explicit zeros and changed pinned heads fail closed."""
    original = e.bind
    raw = tmp_path / "bad_hash"
    operand = tmp_path / "operand.json"
    atomic_json(operand, {"value": 1})
    missing = dict(refs=[], gate_check_summary=[])
    with pytest.raises(ValueError):
        e.bind(dict(path=str(operand), sha256="invalid"), raw, missing)
    assert missing["gate_check_summary"][0]["observed"] == e.reference(operand)["sha256"]
    for field in ("direct_state_ready_score", "frozen_head", "closed_h2"):

        def bad_bind(ref, raw, work, *, parse=True):
            value = original(ref, raw, work, parse=parse)
            if parse and "8404_" in ref["path"] and field == "direct_state_ready_score":
                value["direct_state_ready_score"] = 0
            if parse and ref["path"].endswith(e.authority.PROTOCOL) and field == "frozen_head":
                value["head"]["coefficients"][2] += 0.1
            if parse and "8361_" in ref["path"] and field == "closed_h2":
                value["H2"]["block_bootstrap_summary"]["mean_gain"] = 0.1
            return value

        with monkeypatch.context() as m:
            m.setattr(e, "bind", bad_bind)
            with pytest.raises(ValueError):
                e.authenticate(e.ROOT, tmp_path / field, dict(refs=[], gate_check_summary=[]))


def test_measured_loop_and_reduction_failures(natural_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8406-CUSTODY: mocked unit transport does not qualify deployment."""
    directory = natural_work["streams"][0]["directory"]

    # Unit transport returns existing fixture states. Genuine process exits are
    # verified separately, and this fixture never enters the production receipt.
    def fixture_child(name, argv, logs, **kwargs):
        import shutil

        destination = Path(argv[argv.index("--state-directory") + 1])
        if not destination.exists():
            shutil.copytree(directory, destination)
        return dict(passed=True)

    monkeypatch.setattr(e, "child", fixture_child)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    measured = e.measure(e.ROOT, tmp_path / "measured", private)
    assert len(measured["streams"]) == 9
    assert len(measured["child_receipts"]) == 15
    with monkeypatch.context() as m:
        m.setattr(e.os, "access", lambda *args: False)
        missing = e.measure(e.ROOT, tmp_path / "no_tools", private)
        assert not missing["streams"] and missing["gate_check_summary"]
    monkeypatch.setattr(e, "child", lambda *args, **kw: dict(passed=False))
    failed = e.measure(e.ROOT, tmp_path / "failed", private)
    assert not failed["streams"]
    assert e.build(failed, [dict(passed=True)])["verdict_class"] == "disqualified"
    path = Path(directory) / "final.json"
    prior = path.read_bytes()
    atomic_json(path, {})
    with pytest.raises(ValueError, match="final_state_drift"):
        e.reduction(natural_work)
    path.write_bytes(prior)
    last = Path(directory) / "steps/096.json"
    last.rename(last.with_suffix(".hidden"))
    with pytest.raises(ValueError, match="incomplete_stream"):
        e.reduction(natural_work)


def test_real_publication_and_cli_controls(natural_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8406-PUBLICATION: actual consumers check a private fixture primary."""
    from carnot.reporting import continuous_direct_runner_8406 as r

    work = deepcopy(natural_work)
    work["work_path"] = str(tmp_path / "fixture_work.json")
    monkeypatch.setattr(r, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(e, "measure", lambda *args: deepcopy(work))

    def fixture_plan(private):
        source = private / "coverage_fixture.py"
        source.write_text("result = 1\n")
        config = private / "fixture.ini"
        config.write_text("[run]\ninclude=" + str(source) + "\n[report]\nexclude_lines=\n")
        # Trace a real child so creating fixture coverage cannot suspend the
        # parent collector while the runner itself is under measurement.
        for name, args in [
            (
                "fixture_trace",
                [
                    "run",
                    "--rcfile=" + str(config),
                    "--data-file=" + str(private / ".coverage"),
                    str(source),
                ],
            ),
            (
                "fixture_report",
                [
                    "json",
                    "--rcfile=" + str(config),
                    "--data-file=" + str(private / ".coverage"),
                    "-o",
                    str(private / "coverage.json"),
                ],
            ),
        ]:
            assert child(
                name,
                [sys.executable, "-m", "coverage", *args],
                private / "fixture_logs",
                deadline=30,
            )["passed"]
        return [
            dict(
                name="real_deliberate_error",
                argv=[sys.executable, "-u", str(e.ROOT / e.CLI), "--deliberate-error"],
                deadline=30,
                expected=1,
                scope="owned",
            )
        ]

    monkeypatch.setattr(r, "plan", fixture_plan)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["historical_h2_gain"] == 0 and value["continuous_service_ready_score"] == 0
    assert value["required_checks_passed"]
    assert r.main(["--cold-replay", str(output)]) == 0
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert r.main(["--deliberate-error"]) == 1
    with pytest.raises(SystemExit):
        r.main(["--worker", str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        r.main(["--date", "20261010"])
    monkeypatch.setattr(
        r, "plan", lambda private: (_ for _ in ()).throw(ValueError("owned_failure"))
    )
    assert r.main(["--output", str(tmp_path / "failure.json")]) == 1


def test_frozen_manifest_and_replay_headers(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8406-CHILD-COVERAGE: only invocation shards and scoped commands count."""
    from carnot.reporting import continuous_direct_runner_8406 as r

    monkeypatch.setenv("COVERAGE_RCFILE", "")
    monkeypatch.setenv("COVERAGE_FILE", "")
    frozen = r.plan(tmp_path)
    assert {c["name"] for c in frozen} >= {
        "private_E2E018_consumers",
        "private_E2E020",
        "private_E2E021",
    }
    assert "_exit" in (tmp_path / "coverage.ini").read_text()
    assert "--keep" in next(c["argv"] for c in frozen if c["name"] == "coverage_combine")
    path = tmp_path / "invalid.json"
    atomic_json(path, dict(experiment_id=8406, reproducibility_checksum="invalid"))
    assert not e.replay(path)
