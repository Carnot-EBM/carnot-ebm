"""REQ-REPORT-9095-V775; SCENARIO-REPORT-9095-V775: terminal, honest accounting."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v775_capstone as m
from carnot.reporting import v775_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


def copy_work(work):
    """Copy changed envelopes while immutable large numerical rows remain shared."""
    return dict(
        work,
        tasks=deepcopy(work["tasks"]),
        snapshots=deepcopy(work["snapshots"]),
        values={t: dict(v, value=dict(v["value"])) for t, v in work["values"].items()},
        validation=deepcopy(work["validation"]),
        validation_scope_manifest=deepcopy(work["validation_scope_manifest"]),
        publication_gate=deepcopy(work["publication_gate"]),
        operands=dict(work["operands"]),
    )


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Use actual pinned evidence privately; no producer is run or rewritten."""
    return m.collect(tmp_path_factory.mktemp("capstone9095"), "20261115")


def test_fourteen_outcomes_and_primitive_reductions(work, monkeypatch):
    """SCENARIO-REPORT-9095-V775: siblings survive a terminal external block."""
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    value = m.build(work)
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_declared_gate"
    assert [r["experiment_id"] for r in value["task_outcome_rows"]] == list(range(9082, 9096))
    slots = {r["experiment_id"]: r for r in value["task_outcome_rows"]}
    assert slots[9085]["disposition"] == "authenticated_scheduler_skip"
    assert slots[9085]["primary_status"] == "missing_file"
    assert slots[9087]["disposition"] == "unstarted_upstream_retired"
    assert slots[9093]["verdict_class"] == "null"
    assert slots[9094]["verdict_class"] == "circular_positive"
    assert value["human_summary"]["question_count"] == 300
    assert value["human_summary"]["annotator2_observed_count"] == 300
    assert value["human_summary"]["annotator2_missing_count"] == 600
    assert value["human_summary"]["disagreement_count"] == 29
    assert value["susceptibility_summary"]["counts"]["completed"] == 32
    assert value["cost_summary"]["complete_service_ratios"] == []
    assert value["cost_summary"]["energy_sensor_coverage"]["joules"] is None
    assert len(value["seven_bound_rows"]) == 7 and len(value["ten_bound_rows"]) == 10
    assert all(r["value"] is None for r in value["seven_bound_rows"] + value["ten_bound_rows"])
    assert value["independent_generalization_score"] == value["current_llm_invocations"] == 0
    assert value["MODEL_SPECS"] == [] and not value["current_science_paper_ready"]
    assert value["retirement_recommendations"][0]["recommendation"] == "retire_prospective_chain"
    assert value["archive_lag"]["archive_modified"] is False
    publication = value["publication_gate_receipt"][0]
    assert publication["command_sha256"] == e.canonical_hash(publication["command_argv"])
    assert publication["source_hashes"][0]["original_path"].endswith("scripts/publication_gate.py")
    assert all(r["path"] for r in value["declared_gate_rows"])
    assert all(
        isinstance(r, dict)
        and {
            "check",
            "upstream",
            "path",
            "sha256",
            "artifact_field",
            "op",
            "expected",
            "observed",
            "passed",
        }
        <= r.keys()
        for r in value["gate_check_summary"]
    )


def test_gate_states_and_null_closure():
    """REQ-REPORT-9095-V775: missing fields and zero cannot be renamed success."""
    gate = dict(upstream="exp1-test", artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    assert e.edge(gate, {"exp1-test": dict(value={}, reference={})})["observed"] == "missing_field"
    assert e.edge(gate, {"exp1-test": dict(value=dict(ready=0), reference={})})["observed"] == 0
    assert e.decision("detector", dict(verdict_class="null"))["decision"] == "close_unchanged_scope"
    assert (
        e.decision("cost", dict(verdict_class="null"))["decision"]
        == "retain_inventory_require_complete_service"
    )
    assert (
        e.decision("online", dict(verdict_class="positive"))["decision"]
        == "retain_qualified_result"
    )


def test_replay_and_rehashed_tampering(work, tmp_path):
    """SCENARIO-REPORT-9095-V775: new hashes cannot turn a missing result into a win."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    value["human_summary"]["question_count"] += 1
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert m.replay(path)["reason"] == "reduction_drift"
    assert not m.replay(tmp_path / "absent")["passed"]


def test_blocked_inputs_and_failed_owned_validation(work, tmp_path):
    """REQ-REPORT-9095-V775: owned failure zeros readiness; external failure is terminal."""
    missing = m.collect(tmp_path / "blocked", "20261115", tmp_path / "absent")
    value = m.build(missing)
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["reconciliation_ready_score"] == 0
    assert len(value["task_outcome_rows"]) == 14
    assert m.build(work)["verdict_class"] == "disqualified"
    changed = copy_work(work)
    changed["tasks"][0]["prompt"] += "tampered"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)


def test_private_actual_cli(work, tmp_path):
    """SCENARIO-REPORT-9095-V775: real foreign CLI replays and rejects tampering."""
    path = tmp_path / "work.json"
    atomic_json(path, work)
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9095] before_child", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=90
        )
        print("[test9095] after_child", result.returncode, flush=True)
        return result

    result = call("--from-work", str(path), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    value = json.loads(output.read_bytes())
    value["current_llm_invocations"] = 1
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(output, value)
    assert call("--cold-replay", str(output)).returncode == 1


def test_frozen_validation_and_main_paths(work, tmp_path, monkeypatch):
    """REQ-REPORT-9095-V775: validation failure disqualifies before public replacement."""
    with pytest.raises(SystemExit):
        m.main(["--date", "wrong"])
    with pytest.raises(SystemExit):
        m.main(["--from-work", str(tmp_path / "absent")])
    changed = copy_work(work)
    monkeypatch.setattr(m.runner, "run_check", lambda c, *_: dict(name=c.name, passed=True))
    monkeypatch.setattr(m, "measured", lambda *_: dict(complete=True, rows=[]))
    assert m.validate(changed)["coverage"] == dict(complete=True)
    changed["validation_scope_manifest"]["commands"] = []
    with pytest.raises(ValueError, match="frozen_commands"):
        m.validate(changed)
    monkeypatch.setattr(m, "collect", lambda *_: copy_work(work))
    monkeypatch.setattr(m, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m, "publish_primary", lambda *args: args[2](args[0]))
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=False, receipts=[]))
    assert m.main(["--output", str(output)]) == 1
    assert m.main(["--cold-replay", str(tmp_path / "absent")]) == 1


def test_all_replay_trust_boundaries(work, tmp_path):
    """SCENARIO-REPORT-9095-V775: source, producer, work and gate tampering all fail."""
    target = tmp_path / "candidate.json"
    for mutation, reason in [
        ("pin", "original_pin"),
        ("producer", "producer_drift"),
        ("gate", "publication_gate_drift"),
    ]:
        changed = copy_work(work)
        changed["raw"] = str(tmp_path)
        if mutation == "pin":
            next(iter(changed["snapshots"].values()))["sha256"] = "sha256:changed"
        elif mutation == "producer":
            changed["values"][e.TASKS[1]]["value"]["honest_verdict"] = "complete_null_forged"
        else:
            changed["publication_gate"]["paper_ready"] = False
        atomic_json(tmp_path / "work.json", changed)
        atomic_json(target, m.build(changed))
        assert m.replay(target)["reason"] == reason
    value = m.build(work)
    atomic_json(target, dict(value, reproducibility_checksum="changed"))
    assert m.replay(target)["reason"] == "checksum_drift"
    value["work_reference"]["sha256"] = "sha256:changed"
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(target, value)
    assert m.replay(target)["reason"] == "work_drift"
    changed = copy_work(work)
    saved = tmp_path / "primitive.json"
    atomic_json(saved, dict(original=True))
    changed["operands"]["extra"] = m.reference(saved)
    changed["raw"] = str(tmp_path)
    atomic_json(tmp_path / "work.json", changed)
    atomic_json(target, m.build(changed))
    atomic_json(saved, dict(tampered=True))
    assert m.replay(target)["reason"] == "operand_drift"


def test_changed_primitive_binding_and_publication_failure(tmp_path, monkeypatch):
    """REQ-REPORT-9095-V775: missing raw custody blocks dependent work once."""
    root = tmp_path / "source"
    path = root / "results/experiment_9083_test.json"
    atomic_json(
        path,
        dict(
            experiment_id=9083,
            task_id=e.TASKS[1],
            raw_shard_hashes=[dict(path=str(tmp_path / "absent"), sha256="sha256:absent")],
        ),
    )
    monkeypatch.setattr(e, "PINS", {str(path.relative_to(root)): m.reference(path)["sha256"]})
    result = e.authenticate(root, tmp_path / "raw")
    assert any(r["check"] == "producer_binding" and not r["passed"] for r in result["checks"])
    monkeypatch.setattr(
        m.runner,
        "run_check",
        lambda *_: dict(passed=False, normal_exit=True, log_path=str(tmp_path / "no-log")),
    )
    blocked = m.collect(tmp_path / "blocked", "wrong", root)
    assert "publication_gate" not in blocked
    assert m.build(blocked)["verdict_class"] == "blocked"


def test_private_storage_health_and_blocked_cli(work, tmp_path, monkeypatch):
    """REQ-REPORT-9095-V775: broader failures are hashed context; blocked CLI still completes."""
    out, err = tmp_path / "suite.stdout", tmp_path / "suite.stderr"
    out.write_text("historical failure")
    err.write_text("")
    health = tmp_path / "health.json"
    atomic_json(health, dict(stdout_path=str(out), stderr_path=str(err), passed=False))
    monkeypatch.setattr(m, "HEALTH", health)
    blocked = m.collect(tmp_path / "blocked", "20261115", tmp_path / "absent")
    assert len(blocked["repository_health_references"]) == 3
    source = tmp_path / "blocked_work.json"
    atomic_json(source, blocked)
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [
        sys.executable,
        "-u",
        str(m.ROOT / m.CLI),
        "--from-work",
        str(source),
        "--output",
        str(output),
    ]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9095] blocked_cli_before", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, timeout=90)
    print("[test9095] blocked_cli_after", result.returncode, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
