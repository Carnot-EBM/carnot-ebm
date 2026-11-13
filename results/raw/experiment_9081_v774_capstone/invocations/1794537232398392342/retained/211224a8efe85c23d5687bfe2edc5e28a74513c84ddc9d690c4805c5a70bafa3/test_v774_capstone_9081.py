"""REQ-REPORT-9081-V774: private controls preserve unavailable scientific evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from carnot.reporting import v774_capstone as m
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Use original pinned operands; a private run never changes a producer."""
    return m.collect(tmp_path_factory.mktemp("capstone9081"), "20261113")


def test_fourteen_outcomes(work, monkeypatch):
    """SCENARIO-REPORT-9081-V774: accounting completes while external science blocks."""
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    v = m.build(work)
    assert v["reconciliation_ready_score"] == 1 and v["verdict_class"] == "blocked"
    assert v["intended_count"] == v["completed_count"] == len(v["slot_rows"]) == 14
    slots = {r["experiment_id"]: r for r in v["slot_rows"]}
    assert slots[9071]["disposition"] == "authenticated_scheduler_skip"
    assert slots[9071]["honest_verdict"] is None
    assert slots[9071]["scheduler_honest_verdict"] == "blocked_gate_check_failed"
    assert slots[9072]["disposition"] == "missing_primary"
    assert not v["source_completion_rows"][1]["B_qualified_artifact"]
    for key, count in [("seven_bound_rows", 7), ("ten_bound_rows", 10), ("human_bound_rows", 3)]:
        assert len(v[key]) == count
        assert all(r["value"] is None and r["status"] == "unmeasured" for r in v[key])
    assert v["human_bound_rows"][-1]["control"] == "full_linear"
    assert v["arc_summary"]["new_uncensored_episode_count"] == 0
    assert v["publication_scope"] == "historical_FoVer"
    assert v["paper_ready"] == v["historical_publication_gate"]["paper_ready"]
    assert not v["current_science_paper_ready"]
    assert all(r["passed"] for r in v["title_key_rows"])
    assert not any(v["model_invocation_counts"].values())
    assert all("reopening_evidence" in r for r in v["retirement_scope_rows"])


def test_replay_and_mutations(work, tmp_path):
    """SCENARIO-VERIFY-9081-V774: new hashes cannot hide changed operands or summaries."""
    p = tmp_path / "candidate.json"
    value = m.build(work)
    atomic_json(p, value)
    assert m.replay(p)["passed"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI), "--cold-replay", str(p)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9081] foreign_cli_before", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, timeout=60)
    print("[test9081] foreign_cli_after", result.returncode, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    value["human_bound_rows"][0]["value"] = 1
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(p, value)
    assert not m.replay(p)["passed"]


def test_publication_gate_and_checksum(work, tmp_path):
    """SCENARIO-VERIFY-9081-V774: FoVer gate bytes cannot be replaced by a rehashed claim."""
    changed = deepcopy(work)
    changed["publication_gate"]["paper_ready"] = not changed["publication_gate"]["paper_ready"]
    changed["raw"] = str(tmp_path)
    atomic_json(tmp_path / "work.json", changed)
    p = tmp_path / "candidate.json"
    atomic_json(p, m.build(changed))
    assert m.replay(p)["reason"] == "publication_gate_drift"
    value = m.build(work)
    value["reproducibility_checksum"] = "changed"
    atomic_json(p, value)
    assert m.replay(p)["reason"] == "checksum_drift"
    assert m.main(["--cold-replay", str(p)]) == 1
    assert not m.replay(tmp_path / "missing")["passed"]
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += "changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(changed)
    changed = deepcopy(work)
    changed["snapshots"]["research-roadmap.yaml"]["sha256"] = "sha256:changed"
    atomic_json(tmp_path / "work.json", changed)
    value = m.build(work)
    value["work_reference"] = m.reference(tmp_path / "work.json")
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(p, value)
    assert not m.replay(p)["passed"]


def test_absence_and_owned_failure(work, tmp_path, monkeypatch):
    """REQ-REPORT-9081-V774: precondition failures stop before any reduction."""
    blocked = m.collect(tmp_path / "blocked", "20261113", tmp_path / "absent")
    value = m.build(blocked)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert not value["reconciliation_ready_score"] and not blocked["reduction"]
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    assert m.build(work)["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["values"]["exp9076-graph-benefit"] = dict(value={"required_checks_passed": True})
    with pytest.raises(ValueError, match="qualified_raw_reducer_required"):
        m.reduce(changed)
    gate = dict(upstream="absent", artifact_field="score", op="==", value=1)
    assert m.edge(gate, {})["observed"] == "missing_file"
    assert m.edge(gate, {"absent": dict(value={}, reference={})})["observed"] == "missing_field"
    assert m.edge(gate, {"absent": dict(value=dict(score=0), reference={})})["observed"] == 0


def test_main_and_validation(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9081-V774: only checked terminal bytes reach a private primary."""
    with pytest.raises(SystemExit):
        m.main(["--date", "wrong"])
    with pytest.raises(SystemExit):
        m.main(["--from-work", str(tmp_path / "missing")])
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--from-work", str(source), "--output", str(output)]) == 1
    monkeypatch.setattr(m, "collect", lambda *_: deepcopy(work))
    monkeypatch.setattr(m, "validate", lambda _: dict(receipts=[], coverage={}))
    monkeypatch.setattr(m.runner, "receipts_pass", lambda _: True)
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m, "publish_primary", lambda *args: {"passed": args[2](output)["passed"]})
    assert m.main(["--output", str(output)]) == 0
    monkeypatch.setattr(m.runner, "terminal", lambda *_: dict(passed=False, receipts=[]))
    assert m.main(["--output", str(output)]) == 1
    runpy.run_path(str(m.ROOT / m.CLI), run_name="private_import")


def test_measured_validation_control(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-9081-V774: real command receipts bind the frozen scope."""
    w = deepcopy(work)
    w["raw"], w["private"] = str(tmp_path / "raw"), str(tmp_path / "private")
    spec = m.CommandSpec(
        "private_check", (sys.executable, "-c", "print('normal exit')"), "test", 30
    )
    w["validation_scope_manifest"]["commands"] = [
        dict(name=spec.name, argv=list(spec.argv), timeout_s=spec.timeout_s)
    ]
    monkeypatch.setattr(m.runner, "validation_plan", lambda _: [spec])
    monkeypatch.setattr(m, "measured", lambda *_: dict(complete=False, rows=[]))
    result = m.validate(w)
    assert result["receipts"][0]["normal_exit"] and not result["coverage"]["complete"]
