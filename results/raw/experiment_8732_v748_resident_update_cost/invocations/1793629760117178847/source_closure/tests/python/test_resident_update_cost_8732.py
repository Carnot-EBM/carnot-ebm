"""REQ-REPORT-8732, REQ-VERIFY-8732: complete costs require matched durable work."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import resident_update_cost_8732 as e
from carnot.reporting import resident_update_runner_8732 as runner
from carnot.verify import resident_update_cost_8732 as n
from test_spline_lifecycle_8706 import fixture


def test_roster_and_paired_intervals():
    """SCENARIO-REPORT-8732-COST: repeats pair processes, never independent cases."""
    roster = e.roster()
    assert len(roster) == 30
    assert sum(r["updates"] for r in roster) == 122880
    assert (
        roster == e.roster()
        and len({tuple(r["arm"] for r in roster[i : i + 3]) for i in range(0, 30, 3)}) > 1
    )
    rows = [dict(r, complete_ns=(2 if r["arm"] == "python" else 1) * 10**9) for r in roster]
    interval = e.intervals(rows)
    assert interval["resident_over_json"]["lower95"] == 1
    assert interval["resident_over_python"]["lower95"] == 2
    assert interval["draws"] == 10000


@pytest.mark.parametrize("arm", e.ARMS)
def test_durable_restart_and_exact_controls(tmp_path, arm):
    """REQ-VERIFY-8732: the same store persists prediction and acknowledged state."""
    job = dict(fixture(), updates=8)
    extension = n.resident.load(Path(n.resident.DEFAULT_EXTENSION))
    full = n.drive(job, tmp_path / "full", arm, 0, 8, extension)
    n.drive(job, tmp_path / "split", arm, 0, 4, extension)
    split = n.drive(job, tmp_path / "split", arm, 4, 8, extension)
    assert split["checkpoint"] == full["checkpoint"]
    assert full["durable_commits"] == 16
    boundary = dict(job, boundary=True, boundary_offset=-2)
    dense = n.drive(boundary, tmp_path / "dense", arm, 0, 2, extension)
    assert dense["renormalization_writes"] == 109
    assert full["latencies_ns"] and full["kernel_ns"] > 0
    with pytest.raises(ValueError, match="restart_identity"):
        n.drive(job, tmp_path / "split", arm, 3, 8, extension)


def test_authentication_and_missing_fields(monkeypatch):
    """REQ-REPORT-8732: qualified source/binary identity cannot be inferred from zero."""
    admitted = e.inputs()
    assert not admitted["failures"] and len(admitted["contract"]["rows"]) == 14
    original = e.read
    for mode in [
        "missing",
        "score",
        "zero",
        "binary",
        "validation",
        "source",
        "private",
        "durable_score",
        "durable_flag",
    ]:
        with monkeypatch.context() as m:

            def altered(ref):
                if ref["path"].endswith(e.PRODUCER):
                    if mode == "missing":
                        raise FileNotFoundError(ref["path"])
                    d = deepcopy(original(ref))
                    if mode == "score":
                        d.pop("resident_update_ready_score")
                    if mode == "zero":
                        d["resident_update_ready_score"] = 0
                    if mode == "binary":
                        d["loaded_extension_reference"]["sha256"] = "bad"
                    if mode == "validation":
                        d["validation_receipts"] = []
                    if mode == "source":
                        d["code_config_hashes"][0]["sha256"] = "bad"
                    return d
                result = deepcopy(original(ref))
                if (
                    mode == "private"
                    and ref["path"].endswith("/measurement.json")
                    and "resident_sparse_prototype" in ref["path"]
                ):
                    result["private_control"] = True
                if ref["path"].endswith("experiment_8433_v726_block_checkpoint_qualification.json"):
                    if mode == "durable_score":
                        result["block_store_ready_score"] = 0
                    if mode == "durable_flag":
                        result["required_checks_passed"] = False
                return result

            m.setattr(e, "read", altered)
            failed = e.inputs()["failures"]
            assert failed and failed[-1]["path"] and failed[-1]["field"]


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """Real subprocess evidence covers the script's public entrypoint and restart."""
    out = tmp_path_factory.mktemp("cost8732") / (e.NAME + ".json")
    assert runner.run(out, "20261102", True) == 0
    return out


def test_private_replay_and_rehashed_tampering(private, tmp_path):
    """SCENARIO-VERIFY-8732-REPLAY: a new outer hash cannot authorize edited rows."""
    value = json.loads(private.read_bytes())
    assert value["completed_count"] == 3 and value["update_cost_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and not value["physical_fpga_execution"]
    runner.replay(private)
    command = [sys.executable, str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--cold-replay", str(private)], 0),
        (["--date", "bad"], 1),
        (["--private-control", "--output", str(e.ROOT / "results" / (e.NAME + ".json"))], 1),
        (["--worker", "missing"], 1),
    ]:
        result = subprocess.run(command + args, capture_output=True, timeout=120)
        assert result.returncode == expected, result.stderr.decode()
    changed = deepcopy(value)
    changed["resident_speedup_score"] = 1
    target = tmp_path / "tampered.json"
    e.atomic_json(target, changed)
    with pytest.raises(ValueError, match="report_drift"):
        runner.replay(target)
    primitive = Path(value["primitive_reference"]["path"])
    saved = primitive.read_bytes()
    work = e.read(value["primitive_reference"])
    work["duration_s"] += 1
    e.atomic_json(primitive, work)
    changed = deepcopy(value)
    changed["primitive_reference"] = e.reference(primitive)
    e.atomic_json(target, changed)
    try:
        with pytest.raises(ValueError, match="independent_measurement_seal"):
            runner.replay(target)
    finally:
        primitive.write_bytes(saved)


def test_terminal_outcomes_and_scores(private, monkeypatch):
    """REQ-REPORT-8732: readiness, null, blocked and benefit are independent outcomes."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    assert e.build(work, [dict(passed=False)], private)["verdict_class"] == "disqualified"
    blocked = deepcopy(work)
    blocked.update(process_rows=[], failures=[dict(path="/absent", sha256=None, observed=None)])
    outcome = e.build(blocked, [], private)
    assert outcome["verdict_class"] == "blocked" and outcome["paired_intervals"] == {}
    assert all(r["complete_ns"] is None for r in outcome["rows"])
    positive = deepcopy(work)
    positive["private_control"] = False
    for row in positive["process_rows"]:
        row["complete_ns"] = {
            "python": 20 * 10**9,
            "json_native": 2 * 10**9,
            "resident_native": 10**9,
        }[row["arm"]]
    monkeypatch.setattr(e.np, "quantile", lambda *args: 100.0)
    outcome = e.build(positive, [], private)
    assert outcome["resident_speedup_score"] == outcome["complete_10x_score"] == 1
    assert outcome["verdict_class"] == "circular_positive"


def test_resource_block_and_child_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8732: missing resources block, while failed owned children reject."""
    admitted = e.inputs()
    monkeypatch.setattr(e, "inputs", lambda: deepcopy(admitted))
    real_resources = runner.resources

    def missing(path):
        return dict(real_resources(path), disk_backed=False)

    with monkeypatch.context() as m:
        m.setattr(runner, "resources", missing)
        work = runner.measure(tmp_path / "blocked" / "invocations" / "1", "20261102", True)
        assert work["failures"] and not work["process_rows"]
    with monkeypatch.context() as m:
        m.setattr(runner, "child", lambda *args, **kwargs: dict(passed=False))
        with pytest.raises(ValueError, match="failed_or_killed_measurement_child"):
            runner.measure(tmp_path / "failed" / "invocations" / "2", "20261102", True)


def test_replay_rejection_branches(private, monkeypatch):
    """SCENARIO-VERIFY-8732-REPLAY: seals protect clocks, configuration and sources."""
    original = e.read
    for mode, message in [
        ("source", "independent_source_or_roster_seal"),
        ("clock", "clock_drift"),
        ("freeze", "premeasurement_drift"),
        ("validation", "independent_validation_seal"),
    ]:
        with monkeypatch.context() as m:

            def altered(ref):
                result = deepcopy(original(ref))
                if ref["path"].endswith("/measurement.json"):
                    if mode == "source":
                        result["roster"] = []
                    if mode == "clock":
                        result["duration_s"] += 1
                if ref["path"].endswith("/premeasurement_seal.json") and mode == "freeze":
                    result["roster"] = []
                if "/validation_seals/" in ref["path"] and mode == "validation":
                    result["validation"] = {}
                return result

            m.setattr(e, "read", altered)
            with pytest.raises(ValueError, match=message):
                runner.replay(private)
    value = json.loads(private.read_bytes())
    shard = Path(value["raw_shard_hashes"][0]["path"])
    saved = shard.read_bytes()
    shard.unlink()
    try:
        with pytest.raises(OSError):
            runner.replay(private)
    finally:
        shard.write_bytes(saved)


def test_validation_public_failure_and_main(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8732: owned failure cannot qualify; private paths stay private."""
    monkeypatch.setattr(
        runner.checks, "validate", lambda raw: [dict(name="unit_control", passed=True)]
    )
    assert runner.validate(tmp_path)[0]["passed"]
    work = e.read(json.loads(private.read_bytes())["primitive_reference"])
    monkeypatch.setattr(runner, "measure", lambda *args: work)
    monkeypatch.setattr(runner, "validate", lambda raw: [dict(name="failed_owned", passed=False)])
    monkeypatch.setattr(
        runner, "publish_primary", lambda output, value, validator: dict(primary_sha256="control")
    )
    assert runner.run(tmp_path / (e.NAME + ".json"), "20261102", False) == 1
    monkeypatch.setattr(runner, "run", lambda *args: 0)
    assert runner.main(["--private-control", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setattr(
        runner,
        "child",
        lambda *args, **kwargs: dict(passed=False, stdout_path=str(tmp_path / "findings.json")),
    )
    (tmp_path / "findings.json").write_text(json.dumps(dict(flagged_count=1)))
    assert not runner.terminal(tmp_path / "candidate.json", tmp_path)["passed"]


def test_durability_budget_precondition(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8732-DURABILITY-BUDGET: a slow durable host cannot finish."""
    admitted = e.inputs()
    monkeypatch.setattr(e, "inputs", lambda: deepcopy(admitted))
    real_drive = e.numerical.drive

    def slow(*args):
        result = real_drive(*args)
        result["latencies_ns"] = [10**9] * 4
        return result

    monkeypatch.setattr(e.numerical, "drive", slow)
    work = runner.measure(tmp_path / "budget" / "invocations" / "1", "20261102", False)
    assert not work["process_rows"]
    gate = work["failures"][-1]
    assert gate["field"] == "projected_complete_measurement_s" and gate["observed"] > 3600
    assert gate["path"] and gate["sha256"] and gate["operator"] == "<="
    assert work["durability_probe"]["observed_updates"] == 4
