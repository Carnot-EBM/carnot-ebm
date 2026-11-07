"""REQ-REPORT-8890; SCENARIO-REPORT-8890-CONTRACT: keep evidence scopes intact."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from coverage import Coverage

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import v760_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


def cold_replay(path, raw, expected):
    """Measure a real cold reader without retaining a second large history in pytest."""
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
            str(c.ROOT / c.CLI),
            "--replay",
            str(path),
        ],
        cwd="/var/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        timeout=600,
    )
    assert result.returncode == expected
    child = Coverage(data_file=str(data), config_file=str(config))
    child.combine(data_paths=[str(raw)], strict=True, keep=True)
    if current := Coverage.current():
        current.get_data().update(child.get_data())


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    raw = tmp_path_factory.mktemp("capstone8890")
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
            "-u",
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
        timeout=600,
    )
    assert child.returncode == 0
    combined = Coverage(data_file=str(data), config_file=str(config))
    combined.combine(data_paths=[str(raw)], strict=True, keep=True)
    if current := Coverage.current():
        current.get_data().update(combined.get_data())
    return json.loads((raw / "work.json").read_bytes())


def test_terminal_accounting(frozen):
    """REQ-REPORT-8890: complete accounting includes blocked science and the self-slot."""
    assert all(s["admitted"] for s in frozen["slots"] if s["experiment_id"] != 8887), [
        (s["task_id"], s["admission_errors"]) for s in frozen["slots"] if not s["admitted"]
    ]
    value = c.build(frozen, validation())
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8878, 8891))
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["intended_count"] == value["completed_count"] == 13
    assert value["contract_ready_score"] == 1
    assert value["natural_generalization_score"] == value["independent_count"] == 0
    assert value["numerical_training_calls"] == 0
    assert not value["activation_authorized"]
    assert value["historical_v759"]["contract_ready_score"] == 0
    assert value["paper_ready"] == value["historical_publication_gates"]["paper_ready"]
    assert len(value["independent_denominators"]) == 12
    branches = {r["task_id"]: r for r in value["branch_conclusions"]}
    assert (
        branches["exp8886-label-benefit-audit"]["reduction"]["registered_estimands"][
            "label_benefit_score"
        ]
        == 0
    )
    assert branches["exp8883-evidence-loss-boundary"]["verifier_is_oracle"]
    assert branches["exp8887-qwen-source-canary"]["scheduler_receipts"]
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"
    assert all(r["changed_mechanism_needed"] for r in value["next_decisions"])


def test_replay_and_tamper(frozen, tmp_path):
    """SCENARIO-REPORT-8890-CONTRACT: rehashing cannot hide changed accounting."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    cold_replay(path, tmp_path / "valid", 0)
    value["reconciliation_ready_score"] = 9
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(path, value)
    cold_replay(path, tmp_path / "tampered", 1)
    assert not c.replay(tmp_path / "missing")["passed"]


def test_gate_absences(tmp_path):
    """REQ-REPORT-8890: absent, missing-field and observed-zero gates stay distinct."""
    path = tmp_path / "gate.json"
    assert (
        c.base.shared.gate(path, "exp8887-qwen-source-canary", "ready", 1)["missing_reason"]
        == "missing_file"
    )
    c.atomic_json(path, {})
    assert (
        c.base.shared.gate(path, "exp8887-qwen-source-canary", "ready", 1)["missing_reason"]
        == "missing_field"
    )
    c.atomic_json(path, dict(ready=0))
    assert (
        c.base.shared.gate(path, "exp8887-qwen-source-canary", "ready", 1)["missing_reason"]
        == "observed_zero"
    )
    assert not c.admit(path, "exp8887-qwen-source-canary")["admitted"]


def test_boundary_flags_recomputed(tmp_path):
    """REQ-REPORT-8890: changing both copied rows and error flags still fails semantic recomputation."""
    row = dict(
        unit_id="world/view",
        cluster_id="world",
        arm="learned",
        seed=11,
        metric=0,
        denominator=1,
        status="completed",
        missing_reason=None,
        action="accept",
        candidate="support",
        full_source_truth="refute",
        visible_view_truth="support",
        wrong_source_acceptance=0,
        view_semantic_error=0,
        decisive_coverage=1,
        abstention=0,
    )
    raw = tmp_path / "work.json"
    c.atomic_json(raw, dict(rows=[row]))
    with pytest.raises(ValueError, match="boundary_flags_drift"):
        c.reduce(dict(experiment_id=8883, rows=[row], work_reference=c.reference(raw)))


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("standalone", "label_cost_rows_drift"),
        ("union", "label_union_rows_drift"),
        ("risk", "served_risk_rows_drift"),
        ("unit", "label_unit_rows_drift"),
    ],
)
def test_label_cost_rows_recomputed(tmp_path, mode, reason):
    """REQ-REPORT-8890: paid label counts must reduce from stopping rows rather than copied totals."""
    value = json.loads(
        (c.ROOT / "results/experiment_8885_v760_sequential_label_cost.json").read_bytes()
    )
    result = json.loads(Path(value["measurement_manifest"]["path"]).read_bytes())
    if mode == "standalone":
        result["standalone_costs"][0]["sequential"] += 1
    elif mode == "union":
        result["shared_union_costs"][0]["sequential_total_labels"] += 1
    elif mode == "risk":
        result["rows"][0]["population_risk"] = 9
    else:
        result["rows"][0]["candidate_draws"] += 1
    path = tmp_path / "evaluation.json"
    c.atomic_json(path, result)
    with pytest.raises(ValueError, match=reason):
        c.reduce(
            dict(experiment_id=8885, rows=result["rows"], measurement_manifest=c.reference(path))
        )


def test_reduction_controls():
    """REQ-REPORT-8890: unit rows bind each headline and retain oracle declarations."""
    for n in (8880, 8881, 8882, 8883, 8884, 8885, 8886, 8888):
        value = json.loads(
            next((c.ROOT / "results").glob(f"experiment_{n}_v760_*.json")).read_bytes()
        )
        observed = c.reduce(value)
        assert observed["verdict_class"] == value["verdict_class"]
        assert observed["verifier_is_oracle"] == value["verifier_is_oracle"]
        changed = deepcopy(value)
        changed["rows"] = []
        with pytest.raises(ValueError):
            c.reduce(changed)
    assert c.reduce(dict(experiment_id=0, rows=[]))["constructed_unit_ids"] == []


def test_raw_headline_and_skip_controls(frozen, tmp_path):
    """REQ-REPORT-8890: immutable invocations and authenticated skips prevent invented work."""
    with pytest.raises(FileExistsError):
        c.freeze(c.ROOT, Path(frozen["raw"]), "20261107")
    slot = deepcopy(next(s for s in frozen["slots"] if s["experiment_id"] == 8887))
    task = next(t for t in frozen["contract"]["tasks"] if t["id"] == slot["task_id"])
    assert c.accounted(slot, task)
    slot["scheduler_receipts"] = []
    slot["conductor_lines"] = ["invented skip"]
    assert not c.accounted(slot, task)
    value = json.loads(
        (c.ROOT / "results/experiment_8885_v760_sequential_label_cost.json").read_bytes()
    )
    value["label_savings_score"] = 9
    with pytest.raises(ValueError, match="registered_estimand_drift"):
        c.reduce(value)
    assert c.summary(dict(gate_check_summary="diagnostic"))["gate_check_summary"] == []
    assert c.runner.runner.Path("/tmp/exp8839-global/receipt.json").name == "repository_health.json"
    assert c.runner.runner.Path(tmp_path) == tmp_path


def test_rehashed_skip_custody(frozen, tmp_path):
    """SCENARIO-REPORT-8890-CONTRACT: a recomputed work hash cannot authenticate invented skips."""
    work = dict(frozen, slots=[dict(s) for s in frozen["slots"]])
    shutil.copytree(Path(work["raw"]) / "authority_view", tmp_path / "authority_view")
    work["raw"] = str(tmp_path)
    slot = next(s for s in work["slots"] if s["experiment_id"] == 8887)
    slot["scheduler_receipts"] = []
    slot["conductor_lines"] = ["invented skip"]
    c.atomic_json(tmp_path / "work.json", work)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(work, validation()))
    cold_replay(path, tmp_path / "skip", 1)


def test_child_deadlines(monkeypatch, tmp_path):
    """REQ-REPORT-8890: large custody tests retain a finite budget without heartbeat renewal."""
    monkeypatch.setattr(
        c.checks, "run_check", lambda name, argv, raw, deadline, scope: dict(deadline=deadline)
    )
    for name, deadline in [("owned_tests", 1200), ("cold_replay", 600), ("lint", 120)]:
        assert c.run_check(name, [], tmp_path)["deadline"] == deadline


def test_cli(frozen, tmp_path):
    """REQ-REPORT-8890: actual cold replay works outside the checkout without PYTHONPATH."""
    assert c.main(["--date", "bad"]) == 2
    assert c.main(["--replay", str(tmp_path / "missing")]) == 1
    path = tmp_path / "candidate.json"
    c.atomic_json(path, c.build(frozen, validation()))
    cold_replay(path, tmp_path / "cli", 0)


if __name__ == "__main__":
    c.freeze(c.ROOT, Path(sys.argv[1]), "20261107")
