"""REQ-REPORT-8877; SCENARIO-REPORT-8877-CONTRACT: preserve terminal evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

import pytest
from coverage import Coverage

from carnot.reporting import v759_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    raw = tmp_path_factory.mktemp("capstone8877")
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    data = raw / ".coverage"
    child = subprocess.run(
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
            str(Path(__file__).absolute()),
            str(raw),
        ],
        cwd="/var/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        text=True,
        timeout=420,
    )
    assert child.returncode == 0
    combined = Coverage(data_file=str(data), config_file=str(config))
    combined.combine(data_paths=[str(raw)], strict=True, keep=True)
    if current := Coverage.current():
        current.get_data().update(combined.get_data())
    return json.loads((raw / "work.json").read_bytes())


def test_terminal_accounting(frozen):
    """REQ-REPORT-8877: absence and rejected science still occupy their exact slots."""
    assert c.runner.runner.c.freeze is c.freeze
    value = c.build(frozen, validation())
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8866, 8878))
    assert value["verdict_class"] == "blocked"
    assert value["reconciliation_ready_score"] == 1
    assert not value["contract_ready_score"]
    assert value["intended_count"] == value["completed_count"] == 12
    assert value["paper_ready"] == all(
        g["pass"] for g in value["historical_publication_gates"]["gates"].values()
    )
    assert value["numerical_training_calls"] == 0
    assert value["historical_v758"]["verdict_class"] == "blocked"
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"
    branches = {r["task_id"]: r for r in value["branch_conclusions"]}
    assert branches["exp8871-size-aware-verifier"]["verifier_is_oracle"] is True
    assert branches["exp8871-size-aware-verifier"]["verdict_class"] == "circular_positive"
    assert (
        branches["exp8870-conditional-risk-evaluation"]["reduction"]["registered_estimands"][
            "robustness_advantage_score"
        ]
        == 0
    )
    for n in (8869, 8872, 8873, 8875):
        row = next(r for r in value["rows"] if r["experiment_id"] == n)
        assert row["verdict_class"] == "blocked"
        assert row["scheduler_receipts"] or row["conductor_lines"]
    assert all(not r["broader_hypothesis_retired"] for r in value["retirement_recommendations"])
    assert all(
        "prior_verdict_class" in r and "verifier_is_oracle" in r
        for r in value["retirement_recommendations"]
    )
    assert frozen["preconditions"]["producer_invocation"]["executable"] == sys.executable


def test_replay_and_tamper(frozen, tmp_path):
    """SCENARIO-REPORT-8877-CONTRACT: rehashing a false headline never creates evidence."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    changed = deepcopy(value)
    changed["reconciliation_ready_score"] = 9
    changed["reproducibility_checksum"] = c.checksum(changed)
    c.atomic_json(path, changed)
    assert not c.replay(path)["passed"]
    assert not c.replay(tmp_path / "missing")["passed"]


def test_skip_custody(frozen, tmp_path):
    """REQ-REPORT-8877: a reported skip must exist in the original hashed log."""
    slot = deepcopy(next(r for r in frozen["slots"] if r["experiment_id"] == 8872))
    task = next(t for t in frozen["contract"]["tasks"] if t["id"] == slot["task_id"])
    assert c.accounted(slot, task)
    slot["conductor_lines"] = ["invented skip"]
    assert not c.accounted(slot, task)
    slot["conductor_reference"] = dict(path=str(tmp_path / "missing"), sha256=None)
    assert not c.accounted(slot, task)
    work = deepcopy(frozen)
    work["raw"] = str(tmp_path)
    next(r for r in work["slots"] if r["experiment_id"] == 8872)["conductor_lines"] = [
        "invented skip"
    ]
    shutil.copytree(Path(frozen["raw"]) / "authority_view", tmp_path / "authority_view")
    c.atomic_json(tmp_path / "work.json", work)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, validation()))
    assert not c.replay(path)["passed"]


def test_reduction_controls(frozen):
    """REQ-REPORT-8877: reduction rejects altered rows and preserves declared classes."""
    for n in (8868, 8870, 8871, 8874):
        task = next(t for t in frozen["contract"]["tasks"] if t["id"].startswith(f"exp{n}-"))
        value = json.loads((c.ROOT / task["deliverable"]).read_bytes())
        result = c.reduce(value)
        assert result["verdict_class"] == value["verdict_class"]
        assert result["verifier_is_oracle"] == value["verifier_is_oracle"]
        if n in (8870, 8874):
            changed = deepcopy(value)
            if n == 8870:
                changed["group_count_rows"] = []
            else:
                changed["size_cost_benefit_score"] = 99
            with pytest.raises(ValueError):
                c.reduce(changed)
            value["rows"] = []
            with pytest.raises(ValueError, match="rows_drift"):
                c.reduce(value)
    assert c.reduce(dict(experiment_id=0, rows=[]))["world_ids"] == []
    assert c.summary(dict(gate_check_summary="scheduler diagnostic"))["gate_check_summary"] == []


def test_cli(frozen, tmp_path):
    """REQ-REPORT-8877: actual CLI replay works without checkout path injection."""
    from carnot.reporting.v759_capstone import main

    assert main(["--date", "bad"]) == 2
    assert main(["--replay", str(tmp_path / "missing")]) == 1
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(frozen, validation()))
    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines=\n"
    )
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
        cwd="/var/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        text=True,
        timeout=420,
    )
    assert result.returncode == 0
    child = Coverage(data_file=str(data), config_file=str(config))
    child.combine(data_paths=[str(tmp_path)], strict=True, keep=True)
    if current := Coverage.current():
        current.get_data().update(child.get_data())


if __name__ == "__main__":
    c.freeze(c.ROOT, Path(sys.argv[1]), "20261106")
