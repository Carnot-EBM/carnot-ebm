"""REQ-REPORT-8772, REQ-VERIFY-8772: ordinary windows need real durable evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import ordinary_durable_cost_8772 as e
from carnot.reporting import ordinary_durable_runner_8772 as runner


def test_plan_and_projection():
    """SCENARIO-REPORT-8772-ORDINARY: never shrink the fixed paired roster."""
    assert len(e.roster()) == 90
    assert sum(r["updates"] for r in e.roster()) == 2880
    assert e.roster() == e.roster()
    assert not e.projection([10**9] * 4, 6 * 10**9)["admitted"]
    assert e.projection([10**8] * 4, 10**9)["admitted"]
    with pytest.raises(ValueError, match="probe_contract"):
        e.projection([], 0)


def test_authenticated_inputs(tmp_path):
    """REQ-REPORT-8772: actual archived ordinary readiness admits zero dense events."""
    admitted = e.inputs()
    assert not admitted["failures"]
    assert admitted["qualified"]["ordinary_window_ready_score"] == 1
    assert admitted["qualified"]["natural_dense_event_count"] == 0
    starts, costs, gate = e.starting_states(admitted, tmp_path)
    assert gate["passed"] and len(starts) == len(costs) == 3
    assert [r["offset"] for r in starts] == [0, 2032, 4064]


@pytest.mark.parametrize("mode", ["missing", "field", "zero", "rejected", "protocol", "binary"])
def test_upstream_rejection(mode, monkeypatch):
    """SCENARIO-VERIFY-8772-CUSTODY: reject precise upstream operand failures."""
    original = e.read

    def altered(ref):
        value = deepcopy(original(ref))
        if ref["path"] == str(e.UPSTREAM):
            if mode == "missing":
                raise FileNotFoundError(ref["path"])
            if mode == "field":
                value.pop("ordinary_window_ready_score")
            if mode == "zero":
                value["ordinary_window_ready_score"] = 0
            if mode == "rejected":
                value["required_checks_passed"] = False
            if mode == "binary":
                value["code_config_hashes"][0]["sha256"] = "changed"
        if mode == "protocol" and ref["path"].endswith("/cost_protocol.json"):
            value["updates"] = 31
        return value

    monkeypatch.setattr(e, "read", altered)
    result = e.inputs()
    assert result["failures"] and not result.get("admitted", False)
    failure = result["failures"][-1]
    assert all(
        k in failure
        for k in [
            "upstream",
            "path",
            "sha256",
            "check",
            "field",
            "operator",
            "expected",
            "observed",
        ]
    )


def test_reject_incomplete_and_skipped_fsync():
    """REQ-VERIFY-8772: incomplete windows and fake durability cannot qualify."""
    with pytest.raises(ValueError, match="durable_window"):
        e.check_window(dict(updates=31, durable_commits=64, fsync_calls=128), 32)
    with pytest.raises(ValueError, match="durable_window"):
        e.check_window(dict(updates=32, durable_commits=64, fsync_calls=0), 32)
    assert e.check_window(dict(updates=32, durable_commits=64, fsync_calls=256), 32)


def test_real_probe_child(tmp_path):
    """REQ-VERIFY-8772: real CLI child uses private disk and shipped Store barriers."""
    admitted = e.inputs()
    request = tmp_path / "request.json"
    e.atomic_json(request, e.request(admitted, 0, "python", 4))
    receipt = runner.child(
        "probe",
        [sys.executable, e.CLI, "--worker", str(request), "--work-dir", str(tmp_path / "work")],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"]
    result = json.loads((tmp_path / "work/result.json").read_bytes())
    assert result["updates"] == 4 and result["durable_commits"] == 8
    assert e.check_window(result, 4)


def test_cli_errors(tmp_path):
    """REQ-VERIFY-8772: invalid CLI inputs never write production evidence."""
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--worker", str(tmp_path / "absent")]) == 1
    with pytest.raises(ValueError, match="private_output"):
        runner.run(e.ROOT / "results" / (e.NAME + ".json"), "20261103", private=True)


def test_paired_repeat_bootstrap():
    """REQ-REPORT-8772: sample the same repeat blocks for both comparisons."""
    assert e.intervals([], [[0] * 10]) == {}
    rows = [
        dict(
            r,
            status="completed",
            complete_ns=100 if r["arm"] == "resident_native" else 120,
            p95_ns=100,
        )
        for r in e.roster()
    ]
    value = e.intervals(rows, [[i] * 10 for i in range(10)])
    assert value["json_native"]["lower95"] == pytest.approx(1.2)
    assert value["python"]["p95_ratio"] == 1


@pytest.fixture(scope="module")
def admitted():
    """REQ-VERIFY-8772: retain real immutable inputs for private controls."""
    value = e.inputs()
    assert value["admitted"]
    return value


def test_remaining_input_failures(monkeypatch):
    """REQ-VERIFY-8772: independent producer seals and prior inputs remain mandatory."""
    original = e.read
    with monkeypatch.context() as m:

        def altered(ref):
            value = deepcopy(original(ref))
            if ref["path"].endswith("/terminal_validation.json"):
                value["publication"]["primary_sha256"] = "changed"
            return value

        m.setattr(e, "read", altered)
        assert e.inputs()["failures"][-1]["observed"] == "upstream_terminal_binding"
    with monkeypatch.context() as m:
        m.setattr(e.previous, "inputs", lambda: dict(failures=[dict(passed=False)]))
        assert e.inputs()["failures"] == [dict(passed=False)]


def test_prefix_tampering(admitted, tmp_path, monkeypatch):
    """REQ-VERIFY-8772: an imported ordinary state must equal its archived state."""
    original = e.read

    def altered(ref):
        value = original(ref)
        return dict(value, maximum=-1) if ref["path"].startswith(str(tmp_path)) else value

    monkeypatch.setattr(e, "read", altered)
    with pytest.raises(ValueError, match="ordinary_state_identity"):
        e.starting_states(admitted, tmp_path)


@pytest.mark.parametrize("arm", e.ARMS)
def test_real_import_resume_and_native(arm, admitted, tmp_path):
    """REQ-VERIFY-8772: real native/Python import and durable restart cross the ABI."""
    request = e.request(admitted, 2032, arm, 1)
    first = runner.worker(request, tmp_path)
    second = runner.worker(dict(request, start=2033, stop=2034, resume=True), tmp_path)
    assert first["updates"] == second["updates"] == 1
    assert first["rows"][0]["state_hash"] == admitted["trace"][2032]["state_hash"]
    assert second["rows"][0]["state_hash"] == admitted["trace"][2033]["state_hash"]
    assert first["rows"][0]["arithmetic_ns"] > 0


def test_wrong_loaded_binary(admitted, tmp_path, monkeypatch):
    """REQ-VERIFY-8772: a binary path alone cannot establish loaded identity."""
    monkeypatch.setattr(
        e.previous.previous.numerical.resident,
        "load",
        lambda path: type("Wrong", (), {"__file__": __file__})(),
    )
    with pytest.raises(ValueError, match="loaded_binary_identity"):
        runner.worker(e.request(admitted, 0, "json_native", 1), tmp_path)


def test_cell_failure_and_real_restart(admitted, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8772-ORDINARY: failed cells keep all intended update slots."""
    spec = next(r for r in e.roster() if r["offset"] == 2032 and r["arm"] == "python")
    with monkeypatch.context() as m:
        m.setattr(runner, "child", lambda *a, **kw: dict(passed=False))
        failed = runner.cell(spec, admitted, tmp_path / "failed")
    assert failed["status"] == "failed" and len(failed["update_rows"]) == 32
    actual = runner.cell(spec, admitted, tmp_path / "real")
    assert actual["status"] == "completed" and len(actual["commands"]) == 2
    assert actual["update_rows"][15]["update_index"] == 2048
    assert actual["update_rows"][16]["update_index"] == 2049


@pytest.fixture(scope="module")
def private(tmp_path_factory, admitted):
    """REQ-VERIFY-8772: private negative admission retains a real durable probe."""
    from unittest.mock import patch

    output = tmp_path_factory.mktemp("ordinary8772") / (e.NAME + ".json")

    def blocked(*args):
        return dict(admitted=False, projected_with_reserve_s=4000)

    with patch.object(e, "projection", blocked):
        assert runner.run(output, "20261103", private=True) == 0
    return output


def test_private_replay_and_mutations(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8772-CUSTODY: reject candidate edits and rehashed primitives."""
    value = json.loads(private.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 90
    assert len(value["per_update_rows"]) == 2880
    runner.replay(private)
    target = tmp_path / "changed.json"
    e.atomic_json(target, dict(value, ordinary_update_cost_ready_score=1))
    with pytest.raises(ValueError, match="report_drift"):
        runner.replay(target)
    original = e.read
    for suffix, message, change in [
        ("measurement_seal.json", "independent_measurement_seal", dict(measurement={})),
        ("validation_seal.json", "independent_validation_seal", dict(validation={})),
        ("measurement.json", "clock_or_method_drift", dict(duration_s=-1)),
        ("window_rows.json", "raw_window_drift", dict(rows=[])),
    ]:
        with monkeypatch.context() as m:

            def altered(ref):
                result = original(ref)
                return dict(result, **change) if ref["path"].endswith("/" + suffix) else result

            m.setattr(e, "read", altered)
            with pytest.raises(ValueError, match=message):
                runner.replay(private)
    assert runner.main(["--cold-replay", str(private)]) == 0


def test_build_acceptance(private, tmp_path):
    """REQ-REPORT-8772: valid nulls release costs; owned failures suppress readiness."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    assert e.build(work, [dict(passed=False)], private)["verdict_class"] == "disqualified"
    complete = deepcopy(work)
    complete["failures"] = []
    for row in complete["window_rows"]:
        row.update(status="completed", passed=True, censored=False, complete_ns=100, p95_ns=100)
        row["segments"] = [dict(updates=32, durable_commits=64, fsync_calls=256)]
        for update in row["update_rows"]:
            update.update(status="completed", passed=True, censored=False)
    null = e.build(complete, [], private)
    assert null["verdict_class"] == "null" and null["ordinary_update_cost_ready_score"] == 1
    for row in complete["window_rows"]:
        row["complete_ns"] = 80 if row["arm"] == "resident_native" else 100
    positive = e.build(complete, [], private)
    assert positive["bounded_durable_benefit_score"] == 1


def test_measure_blocked_storage_and_missing_inputs(tmp_path, monkeypatch):
    """REQ-VERIFY-8772: RAM-backed storage cannot qualify durable costs."""
    original = runner.resources
    monkeypatch.setattr(e, "inputs", lambda: dict(failures=[dict(passed=False)], refs=[]))
    monkeypatch.setattr(runner, "resources", lambda path: dict(original(path), disk_backed=False))
    work = runner.measure(tmp_path / "blocked", "20261103")
    assert len(work["failures"]) == 2 and len(work["window_rows"]) == 90
    assert not work["durability_probe"]


def test_measure_admitted_cells_and_probe_failure(admitted, tmp_path, monkeypatch):
    """REQ-VERIFY-8772: admission keeps every cell; failed probe cannot launch them."""
    original_child = runner.child
    monkeypatch.setattr(e, "inputs", lambda: admitted)
    monkeypatch.setattr(e, "starting_states", lambda *a: ([], [], dict(passed=True)))
    monkeypatch.setattr(e, "projection", lambda *a: dict(admitted=True, projected_with_reserve_s=1))
    monkeypatch.setattr(
        runner,
        "cell",
        lambda spec, *a: dict(spec, status="completed", update_rows=[], commands=[], segments=[]),
    )
    work = runner.measure(tmp_path / "admitted", "20261103")
    assert len(work["window_rows"]) == 90 and not work["failures"]

    def failed_probe(name, *args, **kwargs):
        return (
            dict(passed=False) if name == "durable_probe" else original_child(name, *args, **kwargs)
        )

    monkeypatch.setattr(runner, "child", failed_probe)
    with pytest.raises(ValueError, match="durable_probe_failed"):
        runner.measure(tmp_path / "failed-probe", "20261103")
    monkeypatch.setattr(
        runner, "child", lambda *a, **kw: dict(passed=False, stdout_path=str(tmp_path / "empty"))
    )
    (tmp_path / "empty").write_text("")
    with pytest.raises(ValueError, match="premeasurement_node_collection"):
        runner.measure(tmp_path / "collection", "20261103")


def test_public_health_separate_and_cli(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8772: failed global health never impersonates owned acceptance."""
    value = json.loads(private.read_bytes())
    saved_work = e.read(value["primitive_reference"])

    def measured(raw, date):
        raw.mkdir(parents=True)
        work = dict(deepcopy(saved_work), raw=str(raw))
        e.atomic_json(raw / "bootstrap.json", dict(draws=[]))
        e.atomic_json(raw / "measurement.json", work)
        return work

    monkeypatch.setattr(runner, "measure", measured)
    monkeypatch.setattr(runner.checks, "validate", lambda raw: [dict(name="owned", passed=True)])
    monkeypatch.setattr(
        runner, "child", lambda *a, **kw: dict(passed=False, scope="repository_health")
    )
    observed = []

    def publish(output, result, validator):
        observed.append(result)
        return dict(passed=True)

    monkeypatch.setattr(runner, "publish_primary", publish)
    assert runner.run(tmp_path / (e.NAME + ".json"), "20261103") == 0
    assert observed[0]["required_checks_passed"]
    assert not observed[0]["repository_health_reference"]["current_receipt"]["passed"]
    assert runner.main(["--output", str(tmp_path / "cli" / (e.NAME + ".json"))]) == 0


def test_completed_flags_cannot_hide_missing_updates(private):
    """REQ-VERIFY-8772: complete flags alone cannot qualify an incomplete window."""
    value = json.loads(private.read_bytes())
    work = deepcopy(e.read(value["primitive_reference"]))
    work["failures"] = []
    for row in work["window_rows"]:
        row.update(status="completed", passed=True, censored=False, complete_ns=100, p95_ns=100)
    assert e.build(work, [], private)["ordinary_update_cost_ready_score"] == 0


def test_replay_authenticates_child_logs(private, monkeypatch):
    """REQ-VERIFY-8772: raw command logs are checked before receipt reduction."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    receipt = work["durability_probe"]["process_receipt"]
    original = e.read

    def altered(ref):
        result = original(ref)
        return dict(receipts=[receipt]) if ref["path"].endswith("/validation.json") else result

    monkeypatch.setattr(e, "read", altered)
    with pytest.raises(ValueError, match="report_drift"):
        runner.replay(private)
