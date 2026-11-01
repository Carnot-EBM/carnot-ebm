"""REQ-REPORT-8691: censorship cannot select a new supervisor policy."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_censoring_decision_8691 as e
from carnot.reporting import arc_censoring_runner_8691 as runner


def episode(status="censored", game="private-game"):
    """Constructed outcomes test accounting without claiming live generalization."""
    count = 120 if status == "no_progress_within_horizon" else 3
    seq = [
        dict(action_index=i, action={"kind": 1}, observation={"level": 0})
        for i in range(1, count + 1)
    ]
    if status == "level_progress_observed":
        seq[-1]["observation"]["level"] = 1
    return dict(
        unit=dict(game=game, seed=11, reset_provenance="on", episode_id=game + ":11:on"),
        actions=[x["action"] for x in seq],
        delivery_rows=[
            dict(
                action_index=x["action_index"],
                action=x["action"],
                owner="private",
                status="canonical",
            )
            for x in seq
        ],
        reset_provenance=dict(
            reset_request_rows=[], terminal_reason_rows=[], callback_sequence_rows=[]
        ),
        observer_rows=[
            dict(
                redirect_id="private:0",
                arm="force_exploration_diversity",
                selected=True,
                applied=True,
                application_index=1,
                level_before=0,
                horizon=120,
                horizon_status=status,
                observed_horizon=count,
                sequence=seq,
                level_progress=int(status == "level_progress_observed"),
                end_reason="cutoff",
            )
        ],
    )


@pytest.mark.parametrize(
    "status,outcome",
    [
        ("censored", "censored"),
        ("no_progress_within_horizon", "no_progress"),
        ("level_progress_observed", "positive"),
        ("missing", "missing"),
    ],
)
def test_outcome_fixtures(status, outcome):
    """SCENARIO-REPORT-8691-DECISION: all four outcome classes stay distinct."""
    ep = episode(status)
    if status == "missing":
        ep["observer_rows"] = []
    reduced = e.reduce([ep])
    assert reduced["censoring_rows"][0]["outcome"] == outcome
    assert reduced["policy_decision"] == "no_supported_policy_change"
    assert reduced["independent_count"] == 1


@pytest.mark.parametrize(
    "change", ["promotion", "count", "callback", "horizon", "level", "duplicate"]
)
def test_contradictory_receipts(change):
    """SCENARIO-REPORT-8691-DECISION: contradictions cannot promote censored evidence."""
    ep = episode()
    row = ep["observer_rows"][0]
    if change == "promotion":
        row["horizon_status"] = "no_progress_within_horizon"
    elif change == "count":
        row["observed_horizon"] += 1
    elif change == "callback":
        ep["delivery_rows"][0]["action"] = {"kind": 2}
    elif change == "horizon":
        row["horizon"] = 121
    elif change == "level":
        row["horizon_status"] = "level_progress_observed"
    else:
        ep["delivery_rows"].append(deepcopy(ep["delivery_rows"][0]))
    with pytest.raises(ValueError):
        e.reduce([ep])


def cli(*args):
    """REQ-VERIFY-8691: the real standalone process owns private success/rejection."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """REQ-VERIFY-8691: use real original inputs without changing public artifacts."""
    path = tmp_path_factory.mktemp("arc8691") / (e.NAME + ".json")
    result = cli("--output", path, "--private-control", "--date", "20261101")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_private_decision_and_replay(private):
    """SCENARIO-VERIFY-8691-REPLAY: the fixed closed panel yields no policy change."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["arc_decision_ready_score"] == 1
    assert value["policy_decision"] == "no_supported_policy_change"
    assert value["completed_count"] == value["intended_count"] == 16
    assert value["censored_count"] == 16 and value["complete_horizon_count"] == 0
    assert value["independent_count"] == 4 and value["verdict_class"] == "null"
    assert value["model_invocation_counts"]["language_model"] == 0
    assert value["headline_solve_credit"] == 0
    assert set(value) <= set(value["field_principles"])
    work = e.read(value["primitive_reference"])
    assert set(e.read(work["episode_refs"][0])) == {
        "unit",
        "actions",
        "delivery_rows",
        "reset_provenance",
        "observer_rows",
    }


@pytest.mark.parametrize("mode", ["promotion", "primitive", "local_seal", "missing"])
def test_rehashed_rejections(private, tmp_path, mode):
    """SCENARIO-VERIFY-8691-REPLAY: self-consistent replacement hashes grant no authority."""
    _, original = private
    value = deepcopy(original)
    if mode == "promotion":
        value["policy_decision"] = "raise_priority"
    else:
        work = e.read(value["primitive_reference"])
        work["run_date"] = "20261102"
        path = tmp_path / "measurement.json"
        e.atomic_json(path, work)
        value["primitive_reference"] = e.reference(path)
        if mode == "local_seal":
            e.atomic_json(
                path.with_name("measurement_seal.json"),
                {"measurement": value["primitive_reference"]},
            )
        if mode == "missing":
            path.unlink()
    value["reproducibility_checksum"] = e.checksum(value)
    target = tmp_path / "candidate.json"
    e.atomic_json(target, value)
    assert cli("--cold-replay", target).returncode == 1


def test_gate_operand_states(tmp_path):
    """REQ-VERIFY-8691: absence, zero, altered bytes and a flagged producer differ."""
    path = tmp_path / "operand.json"
    assert e.observe(path, "up", "ready", "missing")["operand_state"] == "missing_file"
    for value, expected in [
        ({}, "missing_field"),
        ({"ready": 0}, "observed_zero"),
        ({"ready": 1}, "changed_bytes"),
    ]:
        e.atomic_json(path, value)
        assert e.observe(path, "up", "ready", "wrong")["operand_state"] == expected
    e.atomic_json(path, dict(ready=1, required_checks_passed=True, flagged_adversarial=True))
    assert not e.observe(path, "up", "ready", e.sha256_file(path))["passed"]
    with pytest.raises(ValueError, match="changed_bytes"):
        e.read(dict(path=str(path), sha256="wrong"))


def test_authentication_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8691: producer construction, source, registry and checkpoint are independent gates."""
    monkeypatch.setattr(e, "ROOT", tmp_path)
    assert not e.authenticate()[0]
    source = tmp_path / "source"
    source.write_bytes(b"owned")
    registry = tmp_path / "registry.yaml"
    registry.write_text("games: []")
    head = tmp_path / "head"
    head.write_bytes(b"numeric checkpoint")
    workpath = tmp_path / "work.json"
    e.atomic_json(workpath, dict(references=[], numerical_head_reference=e.reference(head)))
    side = tmp_path / "side.json"
    e.atomic_json(side, dict(publication=dict(sidecar_path=str(side)), passed=True))
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=True)))
    values = []
    for producer, field in (
        (e.ownership, "observer_delivery_ready_score"),
        (e.transfer, "arc_rows_ready_score"),
    ):
        value = dict(
            terminal_validation_sidecar_path=str(side),
            work_reference=e.reference(workpath),
            validation_receipts=[],
            code_config_hashes={str(source): e.sha256_file(source)},
            registry_precheck=dict(
                checked_before_execution=True,
                policy_received_registry=False,
                path=str(registry),
                sha256=e.sha256_file(registry),
            ),
            required_checks_passed=True,
            flagged_adversarial=False,
        )
        value[field] = 1
        values.append(value)
        monkeypatch.setattr(producer, "build", lambda *args, v=value: v)
    paths = [tmp_path / "results" / (p.NAME + ".json") for p in (e.ownership, e.transfer)]

    def write():
        for path, value, number in zip(paths, values, ("8677", "8678"), strict=True):
            e.atomic_json(path, value)
            monkeypatch.setitem(e.PINS, number, e.sha256_file(path))

    write()
    assert e.authenticate()[0] == []
    for field, bad, reason in [
        ("code_config_hashes", {str(source): "bad"}, "producer_source"),
        (
            "registry_precheck",
            dict(checked_before_execution=False, policy_received_registry=False),
            "registry_precheck",
        ),
    ]:
        old = values[0][field]
        values[0][field] = bad
        write()
        with pytest.raises(ValueError, match=reason):
            e.authenticate()
        values[0][field] = old
    write()
    registry.write_text("changed")
    with pytest.raises(ValueError, match="registry_changed"):
        e.authenticate()
    registry.write_text("games: []")
    head.write_bytes(b"changed")
    with pytest.raises(ValueError, match="numerical_checkpoint"):
        e.authenticate()
    head.write_bytes(b"numeric checkpoint")
    e.atomic_json(side, dict(publication=dict(sidecar_path=str(side)), passed=False))
    with pytest.raises(ValueError, match="producer_terminal"):
        e.authenticate()


def test_owned_failure_and_cli_guards(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8691: an actual failed child disqualifies; private mode cannot write public results."""
    path, value = private
    work = e.read(value["primitive_reference"])
    assert e.build(work, [], path)["verdict_class"] == "disqualified"
    for args in [
        ("--date", "123"),
        ("--date", "20260230"),
        ("--private-control",),
        ("--cold-replay", tmp_path / "absent"),
    ]:
        assert cli(*args).returncode == 1
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(runner.validation, "validate", lambda *a: [dict(name="bad", passed=False)])
    monkeypatch.setattr(runner, "publish_primary", lambda *a: {})
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **kw: dict(passed=True))
    assert runner.run(tmp_path / (e.NAME + ".json"), "20261101", False, True) == 1
    monkeypatch.setattr(runner.validation, "validate", lambda *a: [dict(passed=True)])
    assert runner.validate(tmp_path) == [dict(passed=True)]


@pytest.mark.parametrize("private,failed_child", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("alias", [False, True])
def test_private_output_guard_before_measurement(
    tmp_path, monkeypatch, private, failed_child, alias
):
    """SCENARIO-VERIFY-8691-PRIVATE-OUTPUT: direct calls reject resolved checkout paths."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(e, "ROOT", checkout)
    destination = checkout
    if alias:
        destination = tmp_path / "checkout-alias"
        destination.symlink_to(checkout, target_is_directory=True)

    def unexpected_measurement(*args):
        raise AssertionError("private output must be rejected before measurement")

    monkeypatch.setattr(runner, "measure", unexpected_measurement)
    with pytest.raises(ValueError, match="private controls require output outside checkout"):
        runner.run(destination / (e.NAME + ".json"), "20261101", private, failed_child)
    assert list(checkout.iterdir()) == []


@pytest.mark.parametrize("control", ["--private-control", "--failed-child"])
def test_private_cli_rejects_checkout_alias(tmp_path, control):
    """SCENARIO-VERIFY-8691-PRIVATE-OUTPUT: standalone controls cannot follow public aliases."""
    alias = tmp_path / "checkout-alias"
    alias.symlink_to(e.ROOT, target_is_directory=True)
    public = e.ROOT / "results" / (e.NAME + ".json")
    before = public.read_bytes() if public.exists() else None
    result = cli(control, "--output", alias / "results" / public.name)
    assert result.returncode == 1
    assert "private controls require output outside checkout" in result.stderr
    assert (public.read_bytes() if public.exists() else None) == before


@pytest.mark.parametrize("private,failed_child", [(True, False), (False, True), (True, True)])
def test_private_public_symlink_to_external_rejected(tmp_path, monkeypatch, private, failed_child):
    """SCENARIO-VERIFY-8691-PRIVATE-OUTPUT: external targets cannot expose private public aliases."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(e, "ROOT", checkout)
    external = tmp_path / "external.json"
    output = checkout / (e.NAME + ".json")
    output.symlink_to(external)

    def unexpected_measurement(*args):
        raise AssertionError("public aliases must be rejected before measurement")

    monkeypatch.setattr(runner, "measure", unexpected_measurement)
    with pytest.raises(ValueError, match="private controls require output outside checkout"):
        runner.run(output, "20261101", private, failed_child)
    assert output.is_symlink() and not external.exists()
    assert list(checkout.iterdir()) == [output]


def test_private_failed_child_disqualifies(tmp_path):
    """REQ-VERIFY-8691: a real failing child alone clears readiness and preserves its receipt."""
    output = tmp_path / (e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", output)
    assert result.returncode == 1
    raw = tmp_path / "raw" / e.NAME
    value = json.loads((raw / "terminal_candidate.json").read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["arc_decision_ready_score"] == 0
    assert value["required_checks_passed"] is False
    receipt = next(r for r in value["validation_receipts"] if r["name"] == "failed_child")
    assert receipt["actual_exit"] == receipt["exit_code"] == 1
    assert receipt["expected_exit"] == 0
    assert receipt["passed"] is False and receipt["timed_out"] is False
    assert e.read(value["validation_reference"])["receipts"] == value["validation_receipts"]
    for stream in ("stdout", "stderr"):
        assert e.sha256_file(Path(receipt[stream + "_path"])) == receipt[stream + "_sha256"]


def test_blocked_measurement(tmp_path, monkeypatch):
    """REQ-VERIFY-8691: unavailable external inputs and CPU resources produce terminal blocked."""
    monkeypatch.setattr(
        runner,
        "resources",
        lambda *a: dict(
            disk_backed=False, tools_available=True, free_disk_bytes=0, memory_available_bytes=0
        ),
    )
    monkeypatch.setattr(
        e, "authenticate", lambda: (_ for _ in ()).throw(FileNotFoundError("external operand"))
    )
    raw = tmp_path / "raw" / e.NAME / "invocations" / "blocked"
    work = runner.measure(raw, "20261101")
    e.atomic_json(raw / "validation.json", dict(receipts=[]))
    result = e.build(work, [dict(name="private", passed=True)], tmp_path / (e.NAME + ".json"))
    assert result["verdict_class"] == "blocked" and result["arc_decision_ready_score"] == 0
    assert result["gate_check_summary"] and result["completed_count"] == 0
    monkeypatch.setattr(runner.time, "monotonic_ns", iter([0, 3601_000_000_000]).__next__)
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        runner.measure(tmp_path / "timeout", "20261101")


@pytest.mark.parametrize("mode", ["source", "operand", "clock", "receipts", "log"])
def test_replay_boundaries(private, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8691-REPLAY: immutable source, clocks and child streams are mandatory."""
    path, original = private
    value = deepcopy(original)
    work = deepcopy(e.read(value["primitive_reference"]))
    receipts = deepcopy(value["validation_receipts"])
    if mode == "source":
        work["code_refs"] = []
    elif mode == "operand":
        work["source_refs"][0]["sha256"] = "changed"
    elif mode == "clock":
        work["end_ns"] += 1
    elif mode == "receipts":
        receipts = []
    else:
        receipts[0].update(stdout_path=str(path), stdout_sha256="changed")
        value["validation_receipts"] = receipts
    read = e.read

    def altered(ref):
        if ref == value["primitive_reference"]:
            return work
        if ref == value["validation_reference"]:
            return dict(receipts=receipts)
        return read(ref)

    monkeypatch.setattr(e, "read", altered)
    target = tmp_path / "candidate.json"
    e.atomic_json(target, value)
    with pytest.raises(ValueError):
        runner.replay(target)


def test_unknown_and_fabricated_positive():
    """SCENARIO-REPORT-8691-DECISION: observed frames must support a reported positive."""
    ep = episode()
    ep["observer_rows"][0]["horizon_status"] = "pending"
    with pytest.raises(ValueError, match="unknown_outcome"):
        e.reduce([ep])
    ep["observer_rows"][0].update(horizon_status="level_progress_observed", level_progress=1)
    with pytest.raises(ValueError, match="false_promotion"):
        e.reduce([ep])
