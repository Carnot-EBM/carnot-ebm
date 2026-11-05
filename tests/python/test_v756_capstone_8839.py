"""REQ-REPORT-8839, REQ-VERIFY-8839: custody precedes scientific aggregation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from coverage import Coverage

from carnot.reporting import v756_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    return c.freeze(c.ROOT, tmp_path_factory.mktemp("capstone8839"), "20261105")


def test_accounting(frozen):
    """SCENARIO-REPORT-8839-CONTRACT: skips and rejected work cannot license science."""
    value = c.build(frozen, validation())
    assert value["experiment_id"] == 8839 and value["task_id"] == c.TASK
    assert len(value["rows"]) == len(value["full_task_objects"]) == 13
    assert value["reconciliation_ready_score"] == 1 and value["verdict_class"] == "blocked"
    assert value["natural_generalization_count"] == 0
    assert value["historical_counts"] == dict(
        constructed_measurement_count=2160,
        independent_constructed_unit_count=240,
        natural_generalization_count=0,
    )
    assert value["constructed_measurement_count"] > value["independent_constructed_unit_count"] > 0
    assert not value["publication_authorized"] and not value["activation_authorized"]
    slots = {r["experiment_id"]: r for r in value["rows"]}
    assert slots[8834]["primary_honest_verdict"] is None
    assert slots[8832]["disposition"] == "quarantined_evidence"
    assert not slots[8827]["admitted"] and not slots[8830]["admitted"]
    assert not slots[8831]["admitted"]
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"
    assert set(value) <= set(value["field_principles"])


def test_reducers(frozen):
    """REQ-REPORT-8839: admitted nulls count; derived hardware rows do not count twice."""
    by_id = {r["experiment_id"]: r for r in frozen["slots"]}
    for n in [8828, 8829, 8835, 8838]:
        assert by_id[n]["admitted"]
        v = json.loads(Path(by_id[n]["primary_reference"]["path"]).read_bytes())
        reduced = c.reduce(v)
        assert reduced["natural_generalization_count"] == 0
        assert reduced["registered_estimands"]
    assert by_id[8838]["reduction"]["constructed_measurement_count"] == 0
    v = json.loads(Path(by_id[8831]["primary_reference"]["path"]).read_bytes())
    assert c.reduce(v)["registered_estimands"]["schedule_benefit_score"] == 0
    assert c.reduce(dict(experiment_id=999, rows=[]))["constructed_measurement_count"] == 0


def test_retirement_exact_cause():
    """SCENARIO-VERIFY-8839-TAMPER: matching classes cannot justify exact-scope retirement."""
    prior = dict(
        verdict="complete_blocked_missing_runtime",
        retire_if_same_verdict=True,
        addressed_by="new runtime receipt",
    )
    current = dict(
        primary_honest_verdict="complete_blocked_missing_other",
        disposition="admitted_primary",
        verdict_class="blocked",
    )
    assert not c.retirement(prior, current, prior["verdict"])["retire_exact_scope"]
    current["primary_honest_verdict"] = prior["verdict"]
    assert c.retirement(prior, current, prior["verdict"])["retire_exact_scope"]
    current["primary_honest_verdict"] = None
    assert not c.retirement(prior, current, None)["retire_exact_scope"]


def test_replay_tamper(frozen, tmp_path):
    """SCENARIO-VERIFY-8839-TAMPER: repaired outer checksums cannot repair false rows."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    for field in [
        "constructed_measurement_count",
        "natural_generalization_count",
        "reconciliation_ready_score",
    ]:
        changed = deepcopy(value)
        changed[field] += 1
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(path, changed)
        assert not c.replay(path)["passed"]
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    c.atomic_json(path, changed)
    assert not c.replay(path)["passed"]
    for mode in ["contract", "reduction", "admission"]:
        work = deepcopy(frozen)
        if mode == "contract":
            work["contract"]["tasks"][0]["title"] += "tamper"
        elif mode == "reduction":
            next(r for r in work["slots"] if r["admitted"])["reduction"][
                "constructed_measurement_count"
            ] += 1
        else:
            work["slots"][0]["admitted"] = True
        c.atomic_json(tmp_path / "work.json", work)
        changed = deepcopy(value)
        changed["work_reference"] = c.reference(tmp_path / "work.json")
        c.atomic_json(path, changed)
        assert not c.replay(path)["passed"]
    path.write_text("{")
    assert not c.replay(path)["passed"]


def test_preconditions_and_admission(tmp_path, frozen, monkeypatch):
    """REQ-VERIFY-8839: internal and log custody failures are disqualifying."""
    with pytest.raises(ValueError, match="authority"):
        c.freeze(tmp_path, tmp_path / "raw", "20261105")
    with pytest.raises(ValueError, match="disk"):
        monkeypatch.setattr(c.shutil, "disk_usage", lambda p: type("Disk", (), dict(free=0))())
        c.preflight(c.ROOT, tmp_path / "raw")
    assert not c.admit(tmp_path / "missing", "exp0-missing")["admitted"]
    slot = next(r for r in frozen["slots"] if r["experiment_id"] == 8835)
    v = json.loads(Path(slot["primary_reference"]["path"]).read_bytes())
    v["validation_receipts"] = [
        dict(passed=True, log_path=str(tmp_path / "missing.log"), log_sha256="sha256:bad")
    ]
    p = tmp_path / "experiment_8835_fixture.json"
    c.atomic_json(p, v)
    assert not c.admit(p, v["task_id"])["admitted"]


def test_cli_replay(frozen, tmp_path):
    """REQ-VERIFY-8839: real interpreter replay outside checkout needs no PYTHONPATH."""
    from carnot.reporting import v756_capstone_runner as runner

    assert runner.main(["--date", "bad"]) == 2
    assert runner.main(["--replay", str(tmp_path / "missing")]) == 1
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(frozen, validation()))
    assert runner.main(["--replay", str(path)]) == 0
    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    current = Coverage.current()
    data = tmp_path / ".coverage"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(config),
            "--data-file",
            str(data),
            "--parallel-mode",
            str(c.ROOT / c.CLI),
            "--replay",
            str(path),
        ],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    child_coverage = Coverage(data_file=str(data), config_file=str(config))
    child_coverage.combine(data_paths=[str(tmp_path)], strict=True, keep=True)
    if current:
        current.get_data().update(child_coverage.get_data())


def test_private_publication(frozen, tmp_path, monkeypatch):
    """REQ-VERIFY-8839: publication rejection and failed owned checks retain terminal evidence."""
    from types import SimpleNamespace

    from carnot.reporting import v756_capstone_runner as runner

    monkeypatch.setattr(
        runner,
        "Coverage",
        lambda **kwargs: SimpleNamespace(set_option=lambda *a: None, start=lambda: None),
    )
    monkeypatch.setattr(c, "freeze", lambda *args: deepcopy(frozen))
    monkeypatch.setattr(runner.checks, "validate", lambda *args: validation())
    gate_output = tmp_path / "gates.json"
    c.atomic_json(gate_output, dict(paper_ready=False, unmet_gates=["G2"]))
    monkeypatch.setattr(
        runner.checks,
        "run_check",
        lambda *args: dict(
            passed=True,
            stdout_path=str(gate_output),
            stdout_sha256=c.reference(gate_output)["sha256"],
        ),
    )
    monkeypatch.setattr(runner.checks, "terminal", lambda p, r: dict(passed=c.replay(p)["passed"]))
    health_path = tmp_path / "health.json"
    health_log = tmp_path / "health.log"
    health_log.write_text("fixture health debt")
    c.atomic_json(health_path, dict(log_path=str(health_log), passed=False))
    original = runner.Path
    monkeypatch.setattr(
        runner,
        "Path",
        lambda p: health_path if str(p) == "/tmp/exp8839-global/receipt.json" else original(p),
    )
    output = tmp_path / "results" / (c.NAME + ".json")
    assert runner.main(["--private-output", str(output), "--raw-dir", str(tmp_path / "raw")]) == 0
    assert json.loads(output.read_bytes())["reconciliation_ready_score"] == 1
    monkeypatch.setattr(runner.checks, "terminal", lambda *args: dict(passed=False))
    assert runner.run(output, tmp_path / "raw", c.ROOT, "20261105") == 1
    monkeypatch.setattr(runner.checks, "terminal", lambda *args: dict(passed=True))
    monkeypatch.setattr(runner.checks, "validate", lambda *args: validation(False))
    assert runner.run(output, tmp_path / "raw", c.ROOT, "20261105") == 1
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_reduction_and_venv_failures(frozen, tmp_path):
    """SCENARIO-VERIFY-8839-TAMPER: changed measured rows and absent internal Python fail closed."""
    with pytest.raises(ValueError, match="virtualenv"):
        c.preflight(tmp_path, tmp_path / "raw")
    for number, field, message in [
        (8828, "rows", "learning_row"),
        (8829, "finite_support_rows", "support_row"),
    ]:
        slot = next(r for r in frozen["slots"] if r["experiment_id"] == number)
        value = json.loads(Path(slot["primary_reference"]["path"]).read_bytes())
        value[field] = []
        with pytest.raises(ValueError, match=message):
            c.reduce(value)


def test_additional_log_rejection(frozen, tmp_path, monkeypatch):
    """REQ-VERIFY-8839: log custody can reject an otherwise admitted producer."""
    original = c.shared.outcome

    def claimed(*args):
        row = original(*args)
        if row["experiment_id"] == 8827:
            row["disposition"] = "admitted_primary"
        return row

    monkeypatch.setattr(c.shared, "outcome", claimed)
    result = c.freeze(c.ROOT, tmp_path / "raw", "20261105")
    assert result["slots"][0]["disposition"] == "rejected_producer"


def test_frozen_validation_budget(tmp_path, monkeypatch):
    """REQ-VERIFY-8839: owned coverage gets a fixed budget before child execution."""
    from carnot.reporting import v756_capstone_runner as runner

    monkeypatch.setattr(
        runner,
        "original_check",
        lambda name, argv, raw, deadline, scope: dict(deadline=deadline, scope=scope),
    )
    assert runner.run_check("owned_tests", [], tmp_path)["deadline"] == 600
    assert runner.run_check("cold_replay", [], tmp_path)["deadline"] == 120
