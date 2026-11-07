"""REQ-REPORT-8903; SCENARIO-REPORT-8903-CONTRACT: preserve every claim boundary."""

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

import pytest
from coverage import Coverage

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import v761_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


def cold(argv, raw, expected):
    """Measure real child statements, including cold replay outside the checkout."""
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
    with TemporaryDirectory(prefix="capstone8903-", dir="/tmp") as directory:
        raw = Path(directory)
        cold([str(Path(__file__).absolute()), str(raw)], raw / "child", 0)
        yield json.loads((raw / "work.json").read_bytes())


def test_terminal_accounting(frozen):
    """REQ-REPORT-8903: completed administration coexists with blocked science."""
    value = c.build(frozen, validation())
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8891, 8904))
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["intended_count"] == value["completed_count"] == 13
    assert value["natural_generalization_score"] == value["independent_count"] == 0
    assert value["numerical_training_calls"] == 0
    assert not value["activation_authorized"]
    assert value["historical_v760"]["verdict_class"] == "blocked"
    assert len(value["independent_denominators"]) == 12
    branches = {r["task_id"]: r for r in value["branch_conclusions"]}
    assert branches["exp8891-branch-contract"]["verdict_class"] == "disqualified"
    assert not branches["exp8891-branch-contract"]["admitted"]
    assert branches["exp8894-buffered-clock-methods"]["admitted"]
    assert branches["exp8899-buffered-cost-attribution"]["admitted"]
    assert branches["exp8900-qwen-source-boundary"]["scheduler_receipts"]
    assert branches["exp8900-qwen-source-boundary"]["inference_substrate_class"] == "blocked_no_run"
    assert (
        branches["exp8891-branch-contract"]["exact_scope_retirement"] == "repair_owned_validation"
    )
    assert (
        branches["exp8898-signed-benefit-audit"]["reduction"]["registered_estimands"][
            "feature_specific_benefit_score"
        ]
        == 0
    )
    assert (
        branches["exp8896-shared-terminal-evaluation"]["reduction"]["registered_estimands"][
            "shared_union_benefit_score"
        ]
        == 0
    )
    assert (
        branches["exp8896-shared-terminal-evaluation"]["reduction"]["registered_estimands"][
            "standalone_label_benefit_score"
        ]
        == 1
    )
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"
    assert all(r["changed_mechanism_needed"] for r in value["next_decisions"])
    assert value["gate_check_summary"]


def test_replay_and_tamper(frozen, tmp_path):
    """SCENARIO-REPORT-8903-CONTRACT: rehashed aggregates and invented skips fail."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    cold([str(c.ROOT / c.CLI), "--replay", str(path)], tmp_path / "valid", 0)
    value["reconciliation_ready_score"] = 9
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    cold([str(c.ROOT / c.CLI), "--replay", str(path)], tmp_path / "changed", 1)
    assert not c.replay(tmp_path / "missing")["passed"]
    assert c.main(["--date", "bad"]) == 2
    assert c.main(["--replay", str(tmp_path / "missing")]) == 1
    with pytest.raises(FileExistsError):
        c.freeze(c.ROOT, Path(frozen["raw"]), "20261107")
    slot = deepcopy(next(s for s in frozen["slots"] if s["experiment_id"] == 8900))
    task = next(t for t in frozen["contract"]["tasks"] if t["id"] == slot["task_id"])
    assert c.accounted(slot, task)
    slot["scheduler_receipts"] = []
    slot["conductor_lines"] = ["invented skip"]
    assert not c.accounted(slot, task)


def test_custody_heartbeat(frozen, monkeypatch, tmp_path, capsys):
    """REQ-REPORT-8903: copying operands emits progress without renewing a deadline."""
    pending = Mock()
    pending.done.side_effect = [False, True]
    pending.result.side_effect = [TimeoutError, deepcopy(frozen)]
    monkeypatch.setattr(
        c,
        "ThreadPoolExecutor",
        lambda **kwargs: nullcontext(SimpleNamespace(submit=lambda *args: pending)),
    )
    c.freeze(c.ROOT, tmp_path, "20261107")
    assert "phase=operand_custody_wait completed=0 pending=1" in capsys.readouterr().out


@pytest.mark.parametrize("number", [8892, 8893, 8894, 8895, 8896, 8897, 8898, 8899, 8901])
def test_reductions(number):
    """REQ-REPORT-8903: retained rows bind independently recomputed endpoints."""
    value = json.loads(
        next((c.ROOT / "results").glob(f"experiment_{number}_v761_*.json")).read_bytes()
    )
    reduced = c.reduce(value)
    assert reduced["verdict_class"] == value["verdict_class"]
    assert reduced["verifier_is_oracle"] == value["verifier_is_oracle"]
    changed = deepcopy(value)
    changed["rows"] = []
    with pytest.raises(ValueError):
        c.reduce(changed)


def test_absence_and_deadlines(monkeypatch, tmp_path):
    """REQ-REPORT-8903: absence, missing field, zero and owned failure stay distinct."""
    path = tmp_path / "gate.json"
    assert c.base.shared.gate(path, "external", "ready", 1)["missing_reason"] == "missing_file"
    c.atomic_json(path, {})
    assert c.base.shared.gate(path, "external", "ready", 1)["missing_reason"] == "missing_field"
    c.atomic_json(path, dict(ready=0))
    assert c.base.shared.gate(path, "external", "ready", 1)["missing_reason"] == "observed_zero"
    assert not c.admit(path, "external")["admitted"]
    assert c.summary(dict(gate_check_summary="diagnostic"))["gate_check_summary"] == []
    assert c.reduce(dict(experiment_id=0, rows=[]))["constructed_unit_ids"] == []
    assert c.runner.runner.Path("/tmp/exp8839-global/receipt.json").name == "repository_health.json"
    assert c.runner.runner.Path(tmp_path) == tmp_path
    monkeypatch.setattr(
        c.checks, "run_check", lambda name, argv, raw, deadline, scope: dict(deadline=deadline)
    )
    for name, deadline in [("owned_tests", 1200), ("cold_replay", 600), ("lint", 120)]:
        assert c.run_check(name, [], tmp_path)["deadline"] == deadline


def test_label_semantics(monkeypatch):
    """REQ-REPORT-8903: rehashed ledgers cannot hide paid-label or inclusion errors."""
    from carnot.reporting import v761_capstone_reduction as r
    from carnot.reporting.shared_terminal_methods_8893 import ledgers

    value = json.loads(
        (c.ROOT / "results/experiment_8896_v761_shared_terminal_evaluation.json").read_bytes()
    )
    original = json.loads(Path(value["measurement_manifest"]["path"]).read_bytes())
    for mode, reason in [
        ("ledger", "label_ledger_drift"),
        ("paid", "paid_labels_or_risk_drift"),
        ("prefix", "final_prefix_inclusion_drift"),
    ]:
        raw = deepcopy(original)
        if mode == "ledger":
            raw["shared_union_costs"][0]["sequential_total_labels"] += 1
        elif mode == "paid":
            raw["rows"][0]["total_labels"] += 1
            raw["standalone_costs"], raw["shared_union_costs"] = ledgers(raw["rows"])
        else:
            row = next(x for x in raw["rows"] if x["protocol"] == "sequential")
            row["final_prefix_hash"] = "forged"
        monkeypatch.setattr(r, "load", lambda ref: raw)
        with pytest.raises(ValueError, match=reason):
            c.reduce(dict(value, rows=raw["rows"]))


def test_source_and_headline_semantics(monkeypatch):
    """REQ-REPORT-8903: copied source decisions and producer headlines remain challengeable."""
    from carnot.reporting import v761_capstone_reduction as r

    value = json.loads(
        (c.ROOT / "results/experiment_8897_v761_source_bound_decisions.json").read_bytes()
    )
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    work["rows"][0]["source_semantics_status"] = "proven"
    with monkeypatch.context() as patch:
        patch.setattr(r, "load", lambda ref: work)
        with pytest.raises(ValueError, match="source_record_boundary_drift"):
            c.reduce(dict(value, rows=work["rows"]))
    value = json.loads(
        (c.ROOT / "results/experiment_8898_v761_signed_benefit_audit.json").read_bytes()
    )
    value["feature_specific_benefit_score"] = 1
    with pytest.raises(ValueError, match="registered_estimand_drift"):
        c.reduce(value)


def test_original_config_resolution(monkeypatch):
    """REQ-REPORT-8903: only the producer's exact historical config bytes admit science."""
    path = c.ROOT / "results/experiment_8894_v761_buffered_clock_methods.json"
    result = c.admit(path, "exp8894-buffered-clock-methods")
    assert result["admitted"], result["errors"]
    assert any(
        r.get("original_path", "").endswith("research-reporting/spec.md")
        for r in result["references"]
    )
    assert any(r["path"] == str(c.ROOT / c.SPEC) for r in result["value"]["code_config_hashes"])
    monkeypatch.setitem(c.SPEC_HISTORY, "exp8894-buffered-clock-methods", ("0" * 64, "unavailable"))
    assert not c.admit(path, "exp8894-buffered-clock-methods")["admitted"]


if __name__ == "__main__":
    c.freeze(c.ROOT, Path(sys.argv[1]), "20261107")
