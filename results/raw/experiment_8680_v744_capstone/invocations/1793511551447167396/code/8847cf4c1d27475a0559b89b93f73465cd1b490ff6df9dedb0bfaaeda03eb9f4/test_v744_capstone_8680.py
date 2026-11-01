"""REQ-REPORT-8680 / REQ-VERIFY-8680: qualification never creates science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v744_capstone_inputs as e
from carnot.reporting import v744_capstone_report as report
from carnot.reporting import v744_capstone as runner


def cli(*args):
    """Standalone imports must work outside the checkout with no ambient path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_gate_states(tmp_path):
    """SCENARIO-REPORT-8680-ACCOUNTING: absence differs from zero and drift."""
    path = tmp_path / "operand.json"
    assert e.observe(path, "up", "ready")["operand_state"] == "missing_file"
    e.atomic_json(path, {})
    assert e.observe(path, "up", "ready")["operand_state"] == "missing_field"
    e.atomic_json(path, {"ready": 0})
    assert e.observe(path, "up", "ready")["operand_state"] == "observed_zero"
    e.atomic_json(path, {"ready": 1})
    assert e.observe(path, "up", "ready")["passed"]
    assert (
        e.observe(path, "up", "ready", expected_hash="sha256:wrong")["operand_state"]
        == "changed_bytes"
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """REQ-VERIFY-8680: real readers and cold children use private outputs."""
    path = tmp_path_factory.mktemp("v744") / (e.NAME + ".json")
    result = cli("--date", "20261101", "--output", path, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_exact_accounting(private):
    """SCENARIO-REPORT-8680-ACCOUNTING: retain all scopes and prior quarantine."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert len(value["task_outcome_rows"]) == value["intended_count"] == 14
    assert value["completed_count"] == 14 and value["independent_count"] == 0
    assert value["reconciliation_ready_score"] == 1 and value["verdict_class"] == "blocked"
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert value["MODEL_SPECS"] == [] and value["current_language_model_calls"] == 0
    assert value["learning_decision_rows"][0]["measured"] is False
    assert value["semantic_witness_rows"][0]["measured"] is False
    assert value["paper_ready"] == all(value[g]["pass"] for g in ("G1", "G2", "G3", "G4"))
    assert len(value["prd_gap_rows"]) == 3
    assert all(type(v) is int and value[k] == v for k, v in value["branch_gate_values"].items())
    assert set(value) <= set(value["field_principles"])
    assert value["historical_quarantines"] and value["preserved_nulls"]
    assert (
        next(r for r in value["historical_stamped_outcomes"] if r["experiment_id"] == 8664)[
            "flagged_adversarial"
        ]
        is True
    )
    arc = value["arc_decision_rows"]
    assert arc["complete_horizon_count"] == 0 and arc["censored_count"] == 16
    assert arc["action_parity_pairs"] == 8 and arc["observer_error_count"] == 0
    assert all(not r["retired_scientific_hypothesis"] for r in value["retirement_decisions"])


@pytest.mark.parametrize("field", ["rows", "primitive", "local_seal", "receipts"])
def test_rehashed_tampering(private, tmp_path, field):
    """SCENARIO-VERIFY-8680-REPLAY: local self-consistent hashes confer no authority."""
    _, original = private
    value = deepcopy(original)
    if field in {"primitive", "local_seal"}:
        work = e.read(value["primitive_reference"])
        work["duration_s"] += 1
        e.atomic_json(tmp_path / "measurement.json", work)
        value["primitive_reference"] = e.reference(tmp_path / "measurement.json")
        if field == "local_seal":
            e.atomic_json(
                tmp_path / "measurement_seal.json", {"measurement": value["primitive_reference"]}
            )
    elif field == "receipts":
        value["validation_receipts"][0]["passed"] = False
    else:
        value["rows"][0]["eligible"] = True
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(tmp_path / "candidate.json", value)
    assert cli("--cold-replay", tmp_path / "candidate.json").returncode == 1


def test_failure_and_date_paths(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8680: owned child failures disqualify and private modes stay private."""
    path, value = private
    work = e.read(value["primitive_reference"])
    assert report.build(work, [], path)["verdict_class"] == "disqualified"
    for args in [
        ["--date", "123"],
        ["--date", "20260230"],
        ["--private-control"],
        ["--cold-replay", tmp_path / "missing"],
    ]:
        assert cli(*args).returncode == 1
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(runner, "validate", lambda *a: [{"name": "owned", "passed": False}])
    monkeypatch.setattr(runner, "publish_primary", lambda *a: {})
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **kw: {"passed": True})
    assert runner.run(tmp_path / (e.NAME + ".json"), "20261101", False, True) == 1


def test_sealed_authority_and_validator_wrapper(tmp_path, monkeypatch):
    """REQ-VERIFY-8680: authority cannot be replaced by rehashed task dictionaries."""
    p = e.protocol()
    assert e.authority(tmp_path / "authority", p)["ready"]
    (tmp_path / "authority" / e.methods.DESIGN).write_text("forged")
    with pytest.raises(ValueError, match="sealed_authority_changed"):
        e.authority(tmp_path / "authority", p)
    forged = deepcopy(p)
    forged["task_map_v744"][0]["prompt"] += "forged"
    monkeypatch.setattr(e, "read", lambda ref: forged)
    with pytest.raises(ValueError, match="independent_task_authority"):
        e.protocol()
    monkeypatch.setattr(runner.previous, "validate", lambda raw: [{"passed": True}])
    assert runner.validate(tmp_path) == [{"passed": True}]


@pytest.mark.parametrize("mode", ["source", "clock", "log", "reduction"])
def test_replay_independent_operands(private, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8680-REPLAY: source, clock, child output and arithmetic remain bound."""
    _, original = private
    value = deepcopy(original)
    work = deepcopy(e.read(value["primitive_reference"]))
    read = e.read
    if mode == "source":
        work["code_refs"] = []
    elif mode == "clock":
        work["end_ns"] += 1
    elif mode == "log":
        work["publication_receipt"]["stdout_sha256"] = "sha256:wrong"
    monkeypatch.setattr(
        e, "read", lambda ref: work if ref == value["primitive_reference"] else read(ref)
    )
    monkeypatch.setattr(runner.reduction, "reduce", lambda *a: ({}, []))
    target = tmp_path / "candidate.json"
    e.atomic_json(target, value)
    with pytest.raises(
        ValueError,
        match={
            "source": "independent_source_seal",
            "clock": "authority_or_clock_drift",
            "log": "child_log_drift",
            "reduction": "cold_reduction_drift",
        }[mode],
    ):
        runner.replay(target)


@pytest.mark.parametrize("mode", ["span", "truth", "arc", "hardware"])
def test_primitive_reduction_rejections(private, tmp_path, monkeypatch, mode):
    """REQ-REPORT-8680: changed spans, constructed truth, callbacks and costs reject."""
    from carnot.reporting import v744_capstone_reduce as reduction

    _, value = private
    work = e.read(value["primitive_reference"])
    values = {
        str(r["experiment_id"]): e.read(r["primary_reference"])
        for r in work["slots"]
        if r["producer_executed"]
    }
    read = e.read
    episode_paths = {
        r["path"]
        for number in ("8677", "8678")
        for r in read(values[number]["work_reference"])["references"]
    }
    world_ref = work["protocol"]["relational_evidence_protocol"]["worlds_reference"]

    def altered(ref):
        if ref["path"] in episode_paths:
            return {}
        observed = read(ref)
        if ref == world_ref:
            if mode == "span":
                observed["worlds"][0]["views"][0]["claim_span"][1] += 1
            if mode == "truth":
                observed["worlds"][0]["views"][0]["truth"] = 99
        return observed

    monkeypatch.setattr(e, "read", altered)
    monkeypatch.setattr(
        reduction.historical,
        "reduce",
        lambda *a: (work["reductions"]["historical"], work["reductions"]["historical_unions"], []),
    )
    monkeypatch.setattr(
        reduction.ownership,
        "reduce",
        lambda *a: (
            {"action_parity_rows": []}
            if mode == "arc"
            else {
                k: values["8677"][k]
                for k in (
                    "action_parity_rows",
                    "callback_ownership_rows",
                    "repeated_delivery_rows",
                    "per_game_results",
                )
            }
        ),
    )
    monkeypatch.setattr(reduction.transfer, "reduce", lambda *a: work["reductions"]["8678"])
    if mode == "hardware":
        monkeypatch.setattr(reduction.hardware, "source_cells", lambda *a: [])
    with pytest.raises(
        ValueError,
        match={
            "span": "original_relational_span_bytes",
            "truth": "independent_constructed_evaluator",
            "arc": "arc_original_primitive_reduction",
            "hardware": "hardware_original_cost_reduction",
        }[mode],
    ):
        reduction.reduce(values, work["protocol"], tmp_path)


@pytest.mark.parametrize("mode", ["unqualified", "publication", "deadline"])
def test_bounded_measurement_rejection(private, tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8680: failed upstream admission and exhausted budgets cannot qualify."""
    from types import SimpleNamespace

    _, value = private
    work = e.read(value["primitive_reference"])
    by_id = {r["task_id"]: deepcopy(r) for r in work["slots"]}
    if mode == "unqualified":
        by_id["exp8677-arc-observation-ownership"]["eligible"] = False
    monkeypatch.setattr(e, "resolve", lambda task, *a: by_id[task["id"]])
    monkeypatch.setattr(runner.reduction, "reduce", lambda *a: (work["reductions"], []))
    pub = tmp_path / "publication.json"
    e.atomic_json(pub, work["publication"])
    monkeypatch.setattr(
        runner, "child", lambda *a, **kw: {"passed": mode != "publication", "stdout_path": str(pub)}
    )
    if mode == "deadline":
        ticks = iter([0, 3600_000_000_000])
        monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic_ns=lambda: next(ticks)))
    with pytest.raises(
        (ValueError, TimeoutError),
        match={
            "unqualified": "unqualified_current_primitive",
            "publication": "publication_gate_execution",
            "deadline": "measurement_deadline",
        }[mode],
    ):
        runner.measure(tmp_path / "invocations" / "control", "20261101")


def test_missing_primitive_and_killed_child(private, tmp_path):
    """REQ-VERIFY-8680: missing immutable snapshots and genuine SIGKILL cannot qualify."""
    path, value = private
    work = e.read(value["primitive_reference"])
    snapshot = Path(work["source_refs"][0]["snapshot_path"])
    original = snapshot.read_bytes()
    snapshot.unlink()
    try:
        assert cli("--cold-replay", path).returncode == 1
    finally:
        snapshot.write_bytes(original)
    receipt = runner.child(
        "killed_control",
        [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
        tmp_path,
        deadline=10,
    )
    assert receipt["exit_code"] == -9 and receipt["passed"] is False
    assert report.build(work, [receipt], path)["verdict_class"] == "disqualified"
