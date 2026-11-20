"""REQ-REPORT-9179-V781: closure cannot invent missing scientific measurements."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v781_capstone as m
from carnot.reporting import v781_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Original pinned producers supply observations without rerunning old science."""
    return m.collect(tmp_path_factory.mktemp("capstone9179"), "20261117")


def test_fourteen_terminal_outcomes(work, monkeypatch):
    """SCENARIO-REPORT-9179-V781: administrative readiness retains disqualified science."""
    monkeypatch.setattr(m.legacy.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    value = m.build(work)
    assert all(r["passed"] for r in work["checks"])
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert len(value["full_task_objects"]) == len(value["outcome_rows"]) == 14
    slots = {r["experiment_id"]: r for r in value["outcome_rows"]}
    assert list(slots) == list(range(9166, 9180))
    assert slots[9166]["verdict_class"] == "null"
    assert slots[9167]["verdict_class"] == slots[9178]["verdict_class"] == "disqualified"
    assert slots[9168]["scheduler_alternative"]["sha256"]
    assert slots[9168]["primary_status"] == "missing_file"
    assert slots[9169]["scheduler_alternative"] is None
    assert slots[9169]["gate_operands"][0]["observed"] == "missing_field"
    assert slots[9170]["gate_operands"][1]["observed"] == "missing_file"
    assert all(r["verdict_class"] == "blocked" for i, r in slots.items() if 9168 <= i <= 9177)
    assert value["collection_partition"]["answer_count"] == 900
    assert value["collection_partition"]["qid_counts"] == [100] * 3
    assert value["collection_partition"]["union_equals_original"]
    assert value["validation_obligations"]["original"] == 621
    assert value["validation_obligations"]["v779_additions"] == 53
    assert value["validation_obligations"]["current_qualification_credit"] == 0
    assert value["validation_obligations"]["union_count"] >= 674
    assert not value["science_decisions"]["validation"]["cause_demonstrated"]
    assert value["science_decisions"]["static"]["status"] == "unmeasured"
    assert value["science_decisions"]["learning"]["status"] == "unmeasured"
    assert len(value["two_bound_rows"]) == 2 and len(value["seven_bound_rows"]) == 7
    assert all(
        r["absolute_metric"] is None for r in value["two_bound_rows"] + value["seven_bound_rows"]
    )
    assert value["arc_summary"]["completed"] == 124
    assert value["arc_summary"]["recognized"] == 52
    assert value["arc_summary"]["legacy_controls"] == 72
    assert not value["arc_summary"]["live_path_observed"]
    assert value["random_seed"] == 781179
    assert value["MODEL_SPECS"] == []
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert not value["publication_gates"]["current_science_paper_ready"]
    assert value["independent_generalization_score"] == 0
    assert {r["branch"] for r in value["scientific_scope_ledger"]} == {
        "validation",
        "static",
        "learning",
        "cost",
        "ARC",
    }
    assert all(v == "open" for v in value["open_gaps"].values())
    assert not value["activation_authorized"] and not value["external_publication_authorized"]
    assert all(r["next_reopen_condition"] for r in value["continuation_decisions"])
    assert all(r["manifest_mutated"] is False for r in value["retirement_recommendations"])


def test_missing_and_owned_failure(work, tmp_path):
    """REQ-REPORT-9179-V781: unavailable evidence blocks; owned failures disqualify."""
    assert m.build(work)["verdict_class"] == "disqualified"
    blocked = m.collect(tmp_path / "raw", "20261117", tmp_path / "absent")
    value = m.build(blocked)
    assert value["verdict_class"] == "blocked"
    assert len(value["outcome_rows"]) == 14
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += " altered"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)


def test_strict_gate_operands(tmp_path):
    """REQ-REPORT-9179-V781: missing file, missing field and zero differ."""
    gate = dict(upstream="exp1-test", artifact_field="ready", op="==", value=1)
    assert e.edge(gate, {})["observed"] == "missing_file"
    entry = dict(value={}, reference={})
    assert e.edge(gate, {"exp1-test": entry})["observed"] == "missing_field"
    entry["value"]["ready"] = 0
    assert e.edge(gate, {"exp1-test": entry})["observed"] == 0
    entry["value"]["ready"] = True
    assert not e.edge(gate, {"exp1-test": entry})["passed"]
    assert not e.check(tmp_path, "ready", 1, True, "strict")["passed"]


def test_cold_replay_and_tamper(work, tmp_path):
    """SCENARIO-REPORT-9179-V781: rehashing cannot bless changed science or inputs."""
    path = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    for key in ["outcome_rows", "science_decisions", "current_llm_invocations"]:
        changed = dict(value, **{key: 1})
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        assert not m.replay(path)["passed"]
    changed = deepcopy(work)
    next(iter(changed["operands"].values()))["sha256"] = "sha256:changed"
    source = tmp_path / "altered_work.json"
    atomic_json(source, changed)
    tampered = dict(value, work_reference=m.reference(source))
    tampered["reproducibility_checksum"] = m.checksum(tampered)
    atomic_json(path, tampered)
    assert not m.replay(path)["passed"]


def test_private_real_cli(work, tmp_path):
    """SCENARIO-REPORT-9179-V781: E2E-018/021 exercise the real foreign CLI."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]

    def call(*args):
        print("[test9179] child_before", args, flush=True)
        result = subprocess.run(
            [*argv, *args], cwd=tmp_path, env=env, capture_output=True, timeout=180
        )
        print("[test9179] child_after", result.returncode, flush=True)
        return result

    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    result = call("--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert call("--cold-replay", str(output)).returncode == 0
    assert call("--date", "20261116").returncode == 2
    assert call("--from-work", str(source)).returncode == 2
    changed = json.loads(output.read_bytes())
    changed["two_bound_rows"][0]["absolute_metric"] = 1
    changed["reproducibility_checksum"] = m.checksum(changed)
    atomic_json(output, changed)
    assert call("--cold-replay", str(output)).returncode == 1


def test_health_and_publication_refusal(work, tmp_path, monkeypatch):
    """REQ-REPORT-9179-V781: health remains separate and gate-command failure blocks."""
    health = tmp_path / "results/raw" / m.NAME / "development/global_python_suite.receipt.json"
    health.parent.mkdir(parents=True)
    out, err = tmp_path / "stdout", tmp_path / "stderr"
    out.write_text("unrelated failure")
    err.write_text("")
    atomic_json(
        health,
        dict(
            passed=False, timed_out=False, exit_code=1, stdout_path=str(out), stderr_path=str(err)
        ),
    )
    receipt = dict(work["publication_receipts"][0], passed=False)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m.runner, "run_check", lambda *_: receipt)
    changed = m.collect(tmp_path / "raw", "20261117")
    value = m.build(changed)
    assert value["verdict_class"] == "blocked"
    assert value["repository_health"]["exit_code"] == 1
    assert not value["repository_health"]["affects_owned_readiness"]
    assert any(
        r["check"] == "publication_gate_execution" and not r["passed"]
        for r in value["gate_check_summary"]
    )


def test_owned_main_paths(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9179-V781: orchestration enforces affected validation and deadlines."""

    def collect(raw, *_):
        changed = deepcopy(work)
        changed["raw"] = str(raw)
        return changed

    monkeypatch.setattr(m, "collect", collect)
    monkeypatch.setattr(m.legacy, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m.legacy, "verify_measured", lambda *_: True)
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: False)
    assert m.main(["--output", str(output)]) == 1
    assert m.main(["--from-work", str(tmp_path / "missing"), "--output", str(output)]) == 1


def test_exact_retirement_is_installed(work):
    """REQ-REPORT-9179-V781: exact scheduler recurrence retires only unchanged scope."""
    value = m.build(work)
    rows = [r for r in value["retirement_recommendations"] if r["condition_met"]]
    assert len(rows) == 1
    assert rows[0]["task_id"] == "exp9168-forward-qualification"
    assert rows[0]["prior_task_id"] == "exp9154-forward-qualification"
    assert rows[0]["current_verdict"] == "blocked_gate_check_failed"
    assert rows[0]["manifest_retirement_applied"]
    assert rows[0]["next_reopen_condition"]


def test_row_aggregates_cannot_be_invented(work):
    """SCENARIO-REPORT-9179-V781: service counts and donor fractions come from rows."""
    for field in ["cost_summary", "donor_changed_label_fraction"]:
        changed = deepcopy(work)
        prior = changed["values"]["prior:9165"]["value"]
        if field == "cost_summary":
            prior[field]["scientific_request_counts"]["completed"] = 1
        else:
            prior[field] += 0.1
        with pytest.raises(ValueError, match="cost_aggregate|donor_aggregate"):
            m.build(changed)


def test_current_receipt_is_valid(work):
    """REQ-REPORT-9179-V781: operation receipts validate independently of capstone claims."""
    from carnot.reporting.current_work_receipt import validate_current_work_receipt

    value = json.loads(Path(work["current_work_reference"]["path"]).read_bytes())
    assert validate_current_work_receipt(value) == []
