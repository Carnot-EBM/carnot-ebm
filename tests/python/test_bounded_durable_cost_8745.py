"""REQ-REPORT-8745, REQ-VERIFY-8745: preserve fixed windows and real durability."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import bounded_durable_cost_8745 as e
from carnot.reporting import bounded_durable_runner_8745 as runner


def test_fixed_plan_and_projection():
    """SCENARIO-REPORT-8745-ADMISSION: outcomes cannot select cheaper windows."""
    rows = e.roster()
    assert len(rows) == 90 and sum(r["updates"] for r in rows) == 2880
    assert {r["offset"] for r in rows} == {0, 2032, 4064}
    assert {r["repeat"] for r in rows} == set(range(10))
    assert rows == e.roster()
    assert e.projection([1_000_000_000] * 4, 6_000_000_000)["projected_with_reserve_s"] == 3900
    assert not e.projection([1_000_000_000] * 4, 2_000_000_000)["admitted"]
    assert e.projection([100_000_000] * 4, 1_000_000_000)["admitted"]
    with pytest.raises(ValueError, match="probe_contract"):
        e.projection([1], 0)


@pytest.fixture(scope="module")
def authenticated():
    """REQ-VERIFY-8745: use retained source bytes, not the retired live V748 gate."""
    return e.inputs()


def test_actual_original_trajectory(authenticated, tmp_path):
    """REQ-REPORT-8745: real original rows have no middle renormalization."""
    assert not authenticated["failures"]
    assert authenticated["historical_probe"]["projected_minimum_measurement_s"] == 71507.5252224
    states, costs, gate = e.starting_states(authenticated, tmp_path)
    assert [r["offset"] for r in states] == [0, 2032, 4064]
    assert all(r["passed"] for r in states)
    assert costs[-1]["excluded_arithmetic_prefix_updates"] == 4064
    assert gate["observed"] == 0 and gate["expected"] == 1 and not gate["passed"]
    assert gate["path"] and gate["sha256"] and gate["operator"] == ">="
    changed = deepcopy(authenticated)
    changed["trace"][0]["state_hash"] = "changed"
    with pytest.raises(ValueError, match="original_parity_trace"):
        e.starting_states(changed, tmp_path / "changed")


def test_authentication_rejections(monkeypatch):
    """REQ-VERIFY-8745: missing fields and measured zero have distinct evidence."""
    original = e.read
    for mode in ["missing", "field", "zero", "probe"]:
        with monkeypatch.context() as m:

            def altered(ref):
                value = deepcopy(original(ref))
                if ref["path"].endswith(e.PRODUCERS[0][0]):
                    if mode == "missing":
                        raise FileNotFoundError(ref["path"])
                    if mode == "field":
                        value.pop("resident_update_ready_score")
                    if mode == "zero":
                        value["resident_update_ready_score"] = 0
                if ref["path"].endswith(e.PRODUCERS[1][0]) and mode == "probe":
                    value["durability_probe"]["observed_updates"] = 0
                return value

            m.setattr(e, "read", altered)
            failure = e.inputs()["failures"][-1]
            assert failure["path"] and failure["sha256"] and failure["field"]
            assert failure["operand_state"] == (
                "missing_file" if mode == "missing" else "invalid_or_changed_operand"
            )


def test_report_keeps_censored_costs(tmp_path):
    """REQ-REPORT-8745: a blocked prerequisite grants neither cost nor benefit."""
    work = dict(
        raw=str(tmp_path),
        run_date="20261102",
        roster=e.roster(),
        failures=[dict(passed=False)],
        source_refs=[],
        code_refs=[],
        raw_refs=[],
        start_ns=1,
        end_ns=2,
        duration_s=1e-9,
        started_at_utc="a",
        ended_at_utc="b",
        phase_spans=[],
        preconditions={},
        starting_state_rows=[],
        excluded_prefix_cost_rows=[],
        durability_probe={},
        resource_projection={},
        historical_probe={},
        loaded_extension_reference={},
    )
    e.atomic_json(tmp_path / "measurement.json", work)
    e.atomic_json(tmp_path / "validation.json", dict(receipts=[]))
    value = e.build(work, [], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["completed_count"] == 0
    assert value["censored_count"] == 90 and value["bounded_update_cost_ready_score"] == 0
    assert all(r["complete_ns"] is None for r in value["rows"])
    assert not value["long_run_claim_allowed"] and value["paired_window_intervals"] == {}
    assert (
        e.build(work, [dict(passed=False)], tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_projection_and_probe_worker(tmp_path, authenticated):
    """REQ-VERIFY-8745: actual disk updates keep both original Store barriers."""
    job = tmp_path / "job.json"
    e.atomic_json(job, authenticated["job"])
    assert runner.main(["--probe-job", str(job), "--work-dir", str(tmp_path / "probe")]) == 0
    summary = json.loads((tmp_path / "probe/summary-0-4.json").read_bytes())
    assert summary["updates"] == 4 and summary["durable_commits"] == 8
    assert all(r > 0 for r in summary["latencies_ns"])
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--probe-job", str(job)]) == 1


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """REQ-VERIFY-8745: private publication retains real probe and fresh replay."""
    output = tmp_path_factory.mktemp("bounded8745") / (e.NAME + ".json")
    assert runner.run(output, "20261102", private=True) == 0
    return output


def test_private_publication_and_replay(private, tmp_path):
    """SCENARIO-VERIFY-8745-CUSTODY: outer edits cannot change sealed results."""
    value = json.loads(private.read_bytes())
    assert value["verdict_class"] == "blocked"
    runner.replay(private)
    changed = dict(value, bounded_resident_benefit_score=1)
    target = tmp_path / "changed.json"
    e.atomic_json(target, changed)
    with pytest.raises(ValueError, match="report_drift"):
        runner.replay(target)
    work_path = Path(value["primitive_reference"]["path"])
    saved = work_path.read_bytes()
    try:
        work = json.loads(saved)
        work["duration_s"] += 1
        e.atomic_json(work_path, work)
        changed = dict(value, primitive_reference=e.reference(work_path))
        e.atomic_json(target, changed)
        with pytest.raises(ValueError, match="independent_measurement_seal"):
            runner.replay(target)
    finally:
        work_path.write_bytes(saved)
    receipt = runner.child(
        "fresh_replay",
        [sys.executable, e.CLI, "--cold-replay", str(private)],
        tmp_path / "cold",
        deadline=120,
    )
    assert receipt["passed"]


def test_replay_seals_and_clocks(private, monkeypatch):
    """REQ-VERIFY-8745: frozen methods, clocks and validation are separate seals."""
    original = e.read
    for mode in ["clock", "validation"]:
        with monkeypatch.context() as m:

            def altered(ref):
                result = deepcopy(original(ref))
                if mode == "clock" and ref["path"].endswith("/measurement.json"):
                    result["duration_s"] += 1
                if mode == "validation" and ref["path"].endswith("/validation_seal.json"):
                    result["validation"] = {}
                return result

            m.setattr(e, "read", altered)
            with pytest.raises(
                ValueError,
                match="clock_or_method_drift" if mode == "clock" else "independent_validation_seal",
            ):
                runner.replay(private)


def test_blocked_admission_and_owned_failures(tmp_path, authenticated, monkeypatch):
    """REQ-VERIFY-8745: no resource, native-load or probe failure grants readiness."""
    actual_resources = runner.resources
    with monkeypatch.context() as m:
        m.setattr(e, "inputs", lambda: dict(deepcopy(authenticated), failures=[dict(passed=False)]))
        m.setattr(runner, "resources", lambda path: dict(actual_resources(path), disk_backed=False))
        work = runner.measure(tmp_path / "absent", "20261102")
        assert len(work["failures"]) == 2 and not work["durability_probe"]
    with monkeypatch.context() as m:
        m.setattr(runner, "child", lambda *a, **kw: dict(passed=False))
        with pytest.raises(ValueError, match="durability_probe_child_failed"):
            runner.measure(tmp_path / "child", "20261102")
    with monkeypatch.context() as m:
        m.setattr(
            e.previous.numerical.resident,
            "load",
            lambda path: type("Wrong", (), {"__file__": __file__})(),
        )
        with pytest.raises(ValueError, match="actual_loaded_binary"):
            runner.measure(tmp_path / "binary", "20261102")
    with monkeypatch.context() as m:
        m.setattr(e, "projection", lambda *a: dict(admitted=False, projected_with_reserve_s=4000))
        work = runner.measure(tmp_path / "budget", "20261102")
        assert work["failures"][-1]["observed"] == 4000


def test_validation_main_and_private_boundaries(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8745: real CLI contract rejects unsafe private destinations."""
    with pytest.raises(ValueError, match="private_output"):
        runner.run(e.ROOT / "results" / (e.NAME + ".json"), "20261102", private=True)
    monkeypatch.setattr(runner.checks, "validate", lambda raw: [dict(passed=True)])
    assert runner.validate(tmp_path) == [dict(passed=True)]
    monkeypatch.setattr(runner, "run", lambda *a, **kw: 0)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    assert runner.main(["--cold-replay", str(private)]) == 0


@pytest.mark.parametrize(
    "mode,message",
    [
        ("owned", "historical_owned_validation"),
        ("terminal", "historical_terminal_binding"),
        ("validation", "historical_validation_binding"),
        ("private", "private_upstream"),
        ("boundary", "original_4096_trajectory"),
        ("binary", "historical_binary_not_loaded"),
        ("job", "original_durable_job"),
    ],
)
def test_historical_custody_rejections(monkeypatch, mode, message):
    """REQ-VERIFY-8745: every upstream seal is checked before new storage work."""
    original = e.read

    def altered(ref):
        value = deepcopy(original(ref))
        path = ref["path"]
        if path.endswith(e.PRODUCERS[0][0]):
            if mode == "owned":
                value["required_checks_passed"] = False
            if mode == "binary":
                value["loaded_extension_reference"]["actual_loaded"] = False
        if "resident_sparse_prototype" in path:
            if path.endswith("/terminal_validation.json") and mode == "terminal":
                value["publication"]["primary_sha256"] = "changed"
            if path.endswith("/validation.json") and mode == "validation":
                value["receipts"] = []
            if path.endswith("/measurement.json") and mode == "private":
                value["private_control"] = True
            if path.endswith("/jobs/00.json") and mode == "boundary":
                value["boundary"] = True
        if "resident_update_cost" in path and path.endswith("/job-0.json") and mode == "job":
            value["updates"] = 1
        return value

    monkeypatch.setattr(e, "read", altered)
    assert e.inputs()["failures"][-1]["observed"] == message


def test_public_validation_failure_is_terminal(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8745: a failed full-suite receipt disqualifies the public report."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(runner, "validate", lambda raw: [dict(passed=True)])
    monkeypatch.setattr(runner, "child", lambda *a, **kw: dict(passed=False))
    observed = []

    def publication(output, result, validator):
        observed.append(result)
        return dict(primary_sha256="private_failure_control")

    monkeypatch.setattr(runner, "publish_primary", publication)
    assert runner.run(tmp_path / (e.NAME + ".json"), "20261102") == 1
    assert observed[0]["verdict_class"] == "disqualified"
