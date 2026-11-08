"""REQ-REPORT-8916; SCENARIO-REPORT-8916-CONTRACT: keep scoped evidence honest."""

from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from coverage import Coverage
import pytest

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import v762_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


def cold(argv, raw, expected=0):
    """Measure real child statements outside the checkout without PYTHONPATH."""
    raw.mkdir()
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    data = raw / ".coverage"
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(config),
            "--data-file",
            str(data),
            "--parallel-mode",
            *argv,
        ],
        cwd="/var/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        timeout=600,
    )
    assert result.returncode == expected
    combined = Coverage(data_file=str(data), config_file=str(config))
    combined.combine(data_paths=[str(raw)], strict=True, keep=True)
    if current := Coverage.current():
        current.get_data().update(combined.get_data())


@pytest.fixture(scope="module")
def frozen():
    with TemporaryDirectory(prefix="capstone8916-", dir="/tmp") as directory:
        raw = Path(directory)
        cold([str(Path(__file__).absolute()), str(raw)], raw / "child")
        yield json.loads((raw / "work.json").read_bytes())


def test_accounting(frozen):
    """REQ-REPORT-8916: complete accounting preserves blocked, null and rejected science."""
    value = c.build(frozen, validation())
    assert [r["experiment_id"] for r in value["outcome_rows"]] == list(range(8904, 8917))
    assert value["completed_count"] == value["intended_count"] == 13
    assert value["reconciliation_ready_score"] == value["contract_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["independent_count"] == value["independent_generalization_score"] == 0
    assert value["numerical_training_calls"] == 0
    assert len(value["continuation_rows"]) == 13
    assert len(value["independent_denominator_rows"]) == 12
    branches = {r["task_id"]: r for r in value["branch_conclusions"]}
    assert branches["exp8913-qwen-natural-canary"]["verdict_class"] == "blocked"
    assert branches["exp8913-qwen-natural-canary"]["scheduler_receipts"]
    assert branches["exp8914-guarded-durable-cost"]["verdict_class"] == "disqualified"
    assert not branches["exp8914-guarded-durable-cost"]["admitted"]
    assert (
        branches["exp8911-guarded-benefit-audit"]["reduction"]["registered_estimands"][
            "guarded_benefit_score"
        ]
        == 0
    )
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"
    changed = deepcopy(frozen)
    changed["slots"][0]["accounted"] = False
    assert c.build(changed, validation())["reconciliation_ready_score"] == 0


def test_replay(frozen, tmp_path):
    """SCENARIO-REPORT-8916-CONTRACT: rehashing cannot authorize changed conclusions."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    cold([str(c.ROOT / c.CLI), "--replay", str(path)], tmp_path / "good")
    value["reconciliation_ready_score"] = 9
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    cold([str(c.ROOT / c.CLI), "--replay", str(path)], tmp_path / "bad", 1)
    assert not c.replay(tmp_path / "missing")["passed"]
    assert c.main(["--date", "bad"]) == 2
    with pytest.raises(FileExistsError):
        c.freeze(c.ROOT, Path(frozen["raw"]), "20261108")


@pytest.mark.parametrize("number", [8905, 8906, 8907, 8908, 8909, 8910, 8911, 8912, 8915])
def test_reductions(number):
    """REQ-REPORT-8916: original raw rows constrain every imported scientific headline."""
    value = json.loads(
        next((c.ROOT / "results").glob(f"experiment_{number}_v762_*.json")).read_bytes()
    )
    result = c.reduce(value)
    assert result["verdict_class"] == value["verdict_class"]
    changed = deepcopy(value)
    changed["rows"] = []
    with pytest.raises(ValueError):
        c.reduce(changed)


def test_external_and_custody(frozen, tmp_path):
    """REQ-REPORT-8916: absent science, observed zero and invented skips are distinct."""
    path = tmp_path / "gate.json"
    assert c.base.shared.gate(path, "external", "ready", 1)["missing_reason"] == "missing_file"
    c.atomic_json(path, {})
    assert c.base.shared.gate(path, "external", "ready", 1)["missing_reason"] == "missing_field"
    c.atomic_json(path, dict(ready=0))
    assert c.base.shared.gate(path, "external", "ready", 1)["missing_reason"] == "observed_zero"
    c.atomic_json(path, dict(ready=True))
    assert not c.base.shared.gate(path, "external", "ready", 1)["passed"]
    assert not c.admit(path, "external")["admitted"]
    slot = deepcopy(frozen["slots"][9])
    task = frozen["contract"]["tasks"][9]
    slot["scheduler_receipts"] = []
    slot["conductor_lines"] = ["invented"]
    assert not c.accounted(slot, task)
    assert c.summary(dict(gate_check_summary="diagnostic"))["gate_check_summary"] == []


def test_admission_controls(tmp_path):
    """SCENARIO-REPORT-8916-CONTRACT: wrong identities and changed operands fail closed."""
    path = tmp_path / "experiment_9999_fixture.json"
    operand = tmp_path / "operand"
    operand.write_text("sealed")
    value = dict(
        experiment_id=9999,
        task_id="exp9999-fixture",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        owned_statement_coverage=dict(complete=True, percent=100),
        validation_receipts=[dict(passed=True)],
        raw_shard_hashes=[c.reference(operand)],
    )
    c.atomic_json(path, value)
    assert c.admit(path, value["task_id"])["admitted"]
    operand.write_text("changed")
    assert not c.admit(path, value["task_id"])["admitted"]
    value["raw_shard_hashes"] = []
    c.atomic_json(path, value)
    assert not c.admit(path, value["task_id"])["admitted"]
    value.update(experiment_id=8904, task_id="exp8904-branch-contract")
    path = tmp_path / "experiment_8904_fixture.json"
    c.atomic_json(path, value)
    assert not c.admit(path, value["task_id"])["admitted"]


def test_heartbeat_and_deadlines(frozen, monkeypatch, tmp_path, capsys):
    """REQ-REPORT-8916: observable custody and child deadlines never extend the hard cap."""
    pending = Mock()
    pending.done.side_effect = [False, True]
    pending.result.side_effect = [TimeoutError, deepcopy(frozen)]
    monkeypatch.setattr(
        c,
        "ThreadPoolExecutor",
        lambda **kwargs: nullcontext(SimpleNamespace(submit=lambda *args: pending)),
    )
    c.freeze(c.ROOT, tmp_path, "20261108")
    assert "phase=operand_custody_wait completed=0 pending=1" in capsys.readouterr().out
    monkeypatch.setattr(
        c.checks, "run_check", lambda name, argv, raw, deadline, scope: dict(deadline=deadline)
    )
    for name, deadline in [("owned_tests", 1200), ("cold_replay", 600), ("lint", 120)]:
        assert c.run_check(name, [], tmp_path)["deadline"] == deadline
    assert c.evidence_path("/tmp/exp8839-global/receipt.json").name == "repository_health.json"
    assert c.evidence_path(tmp_path) == tmp_path


def test_scientific_null_and_semantic_drift(frozen, monkeypatch, tmp_path):
    """REQ-REPORT-8916: a valid null stays null and repaired hashes cannot change science."""
    work = deepcopy(frozen)
    for slot in work["slots"]:
        if not slot["admitted"]:
            slot["reduction"] = c.reduce(dict(experiment_id=0, rows=[]))
        slot.update(admitted=True, verdict_class="null")
    assert c.build(work, validation())["verdict_class"] == "null"
    value = json.loads(
        (c.ROOT / "results/experiment_8905_v762_guarded_update_methods.json").read_bytes()
    )
    value["candidate_ready_score"] = 1 - value["candidate_ready_score"]
    with pytest.raises(ValueError, match="registered_estimand_drift"):
        c.reduce(value)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(frozen, validation()))
    monkeypatch.setattr(c, "legacy_replay", lambda path: dict(passed=True))
    monkeypatch.setattr(c, "accounted", lambda slot, task: False)
    assert not c.replay(path)["passed"]


if __name__ == "__main__":
    c.freeze(c.ROOT, Path(sys.argv[1]), "20261108")
