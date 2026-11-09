"""REQ-REPORT-8956: reconciliation must preserve evidence without inventing benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import v765_capstone as m
from carnot.reporting import v765_capstone_sources as s
from carnot.reporting.current_work_receipt import atomic_json


def validation(passed=True):
    """REQ-REPORT-8956: injected receipts exercise private routing only."""
    return dict(passed=passed, coverage=dict(complete=passed, percent=100), receipts=[])


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8956: real original producers supply the unchanged census."""
    return s.collect(tmp_path_factory.mktemp("capstone8956"))


def test_census(work):
    """SCENARIO-REPORT-8956-CONTRACT: thirteen slots preserve nulls and oracle credit."""
    v = m.build(work, validation())
    assert len(v["outcome_rows"]) == 13
    assert [r["task_id"] for r in v["outcome_rows"]] == [t["id"] for t in work["tasks"]]
    assert v["verdict_class"] == "blocked" and v["reconciliation_ready_score"] == 0
    clean = deepcopy(work)
    clean["slots"][1]["qualified"] = True
    assert m.build(clean, validation())["verdict_class"] == "null"
    assert v["completed_count"] == 13 and v["independent_count"] == 0
    assert v["branch_conclusions"][1]["original_assertions"]["flagged_adversarial"] is True
    assert v["scientific_reductions"]["collection"]["independent_error_count"] >= 200
    assert not v["scientific_reductions"]["detector"]["benefit_passed"]
    assert v["scientific_reductions"]["learning"]["gradient_benefit_score"] == 0
    assert len(v["scientific_reductions"]["cost"]["rows"]) == 432
    assert v["numerical_training_calls"] == 0
    assert not any(v["model_invocation_counts"].values())
    assert v["paper_ready_scope"].startswith("FoVer")
    assert v["unmet_gates"] == work["publication_gate"]["unmet_gates"]
    assert m.build(work, validation(False))["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["slots"][0]["accounted"] = False
    assert m.build(changed, validation())["honest_verdict"] == "complete_blocked_v765_capstone"
    changed["tasks"].reverse()
    with pytest.raises(ValueError):
        m.build(changed, validation())


def test_missing_and_drift(tmp_path):
    """REQ-REPORT-8956: absent evidence is a terminal external block, never partial."""
    sources = tmp_path / "sources"
    sources.mkdir()
    for label in ("missing", "changed"):
        if label == "changed":
            (sources / s.PINS[8944][0]).write_text("{}")
        w = s.collect(tmp_path / label, sources=sources)
        v = m.build(w, validation())
        assert v["verdict_class"] == "blocked" and v["reconciliation_ready_score"] == 0
        assert v["gate_check_summary"][0]["check"] == (
            "missing_file" if label == "missing" else "rejected_producer"
        )


def test_admission_and_diagnostics(work):
    """REQ-REPORT-8956: observed one and scientific zero are both present fields."""
    task = work["tasks"][4]
    p = s.load(work["slots"][4]["reference"])
    assert s.admission(p, task) is None
    for changed, reason in [
        ({}, "missing_field"),
        (dict(p, task_id="wrong"), "rejected_producer"),
        (dict(p, required_checks_passed=False), "disqualified_primary"),
        (dict(p, verdict_class="positive"), "rejected_producer"),
    ]:
        assert s.admission(changed, task) == reason
    for value, field, check in [
        ({}, "x", "missing_field"),
        ({"x": 0}, "x", "observed_zero"),
        ({"x": 1}, "x", "observed_value"),
    ]:
        result = s.diagnostic(task, None, check, field, 2, value.get(field))
        assert result["check"] == check and result["observed"] == value.get(field)


def test_replay_and_rehashed_tamper(work, tmp_path):
    """SCENARIO-REPORT-8956-CONTRACT: isolated replay must reject rehashed false claims."""
    path = tmp_path / (m.NAME + ".json")
    v = m.build(work, validation())
    cli = [str(m.ROOT / ".venv/bin/python"), "-u", str(m.ROOT / m.CLI), "--cold-replay", str(path)]
    for field, replacement, expected in [
        (None, None, 0),
        ("outcome_rows", [], 1),
        ("independent_count", 320, 1),
        ("paper_ready", not v["paper_ready"], 1),
        ("rows", [], 1),
    ]:
        changed = dict(v, **({field: replacement} if field else {}))
        changed["reproducibility_checksum"] = m.checksum(changed)
        atomic_json(path, changed)
        result = subprocess.run(cli, cwd=tmp_path, capture_output=True, text=True, timeout=180)
        assert result.returncode == expected, result.stdout + result.stderr
    atomic_json(path, dict(v, reproducibility_checksum="wrong"))
    assert m.main(["--cold-replay", str(path)]) == 1


def test_private_cli(tmp_path):
    """SCENARIO-REPORT-8956-CONTRACT: direct child publication and cold replay stay private."""
    path = tmp_path / (m.NAME + ".json")
    cli = [str(m.ROOT / ".venv/bin/python"), "-u", str(m.ROOT / m.CLI)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, expected in [
        (["--private-output", str(path)], 0),
        (["--cold-replay", str(path)], 0),
        (["--date", "bad"], 2),
    ]:
        result = subprocess.run(
            cli + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == expected, result.stdout + result.stderr
    assert json.loads(path.read_bytes())["verdict_class"] == "disqualified"


def test_checked_publication(work, tmp_path, monkeypatch):
    """REQ-REPORT-8956: exercise final checks and debt retention without public writes."""
    real_root = m.ROOT
    monkeypatch.setattr(m, "ROOT", tmp_path)
    for name in [
        *m.OWNED,
        m.TEST,
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(real_root / name)

    def collected(raw, sources):
        raw.mkdir(parents=True, exist_ok=True)
        return dict(work, raw=str(raw), code=[], references=list(work["references"]))

    monkeypatch.setattr(s, "collect", collected)
    monkeypatch.setattr(m.checker, "validate", lambda raw: validation())
    monkeypatch.setattr(
        m.checker, "terminal", lambda candidate, raw: dict(passed=True, receipts=[])
    )
    log = tmp_path / "global.log"
    log.write_text("existing global collection timeout\n")
    health = tmp_path / "global.json"
    health.write_text(
        json.dumps([dict(log_path=str(log), log_sha256=s.sha256_file(log), passed=False)])
    )
    monkeypatch.setenv("CARNOT_V765_GLOBAL_HEALTH_RECEIPT", str(health))
    assert m.main([]) == 0
    output = tmp_path / "results" / (m.NAME + ".json")
    assert json.loads(output.read_bytes())["global_test_debt"]["current"][0]["passed"] is False
    monkeypatch.setattr(m.checker, "validate", lambda raw: validation(False))
    with pytest.raises(ValueError, match="owned_validation_failed"):
        m.main([])
    with pytest.raises(SystemExit):
        m.main(["--private-output", str(real_root / "results" / (m.NAME + ".json"))])


def test_owned_scope(monkeypatch, tmp_path):
    """REQ-REPORT-8956: no second global-suite attempt hides inside owned checks."""
    calls = []

    def run(root, commands, **kwargs):
        calls.extend(commands)
        return []

    monkeypatch.setattr(m, "run_commands", run)
    unit = m.CommandSpec("unit", ("true",), "owned")
    debt = m.CommandSpec("global", ("pytest",), "repository_health")
    assert m.run_checks(tmp_path, [unit, debt]) == []
    assert calls == [unit]


def test_operand_rejection(monkeypatch, tmp_path):
    """REQ-REPORT-8956: readable primaries cannot excuse lost or replaced raw operands."""

    def reject(ref, catalog):
        raise ValueError("lost_operand")

    monkeypatch.setattr(s, "resolve", reject)
    w = s.collect(tmp_path / "rejected")
    assert all(not r["qualified"] for r in w["slots"])
    assert w["scientific_reductions"] == {}


def test_rehashed_work_refusal(work, tmp_path):
    """SCENARIO-REPORT-8956-CONTRACT: rehashed work cannot invent primary accounting."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    changed["slots"][0]["reference"] = None
    atomic_json(tmp_path / "work.json", changed)
    path = tmp_path / (m.NAME + ".json")
    atomic_json(path, m.build(changed, validation()))
    with pytest.raises(ValueError, match="missing_primary_accounting"):
        m.replay(path)


def test_scheduler_identity(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8956-CONTRACT: actual task and upstream receipts authenticate skips."""
    values = {r["experiment_id"]: s.load(r["reference"]) for r in work["slots"]}
    task = work["tasks"][5]
    value = dict(
        task_id=task["id"],
        experiment_id=8949,
        blocked_at_layer="conductor_pre_gate",
        failed_upstream=work["tasks"][4]["id"],
        failed_field="depth_source_ready_score",
        failed_expected=1,
        failed_operator="==",
        failed_observed=0,
        failed_evidence_sha256=s.PINS[8948][1],
    )
    values[8948]["depth_source_ready_score"] = 0
    refs = {r["experiment_id"]: r["reference"] for r in work["slots"]}
    assert s.scheduler_check(value, task, values, refs)["check"] == "observed_zero"
    values[8948]["depth_source_ready_score"] = value["failed_observed"] = 1
    with pytest.raises(ValueError, match="scheduler_gate_identity"):
        s.scheduler_check(value, task, values, refs)
    values[8948]["depth_source_ready_score"] = value["failed_observed"] = 0
    sources = tmp_path / "sources"
    sources.mkdir()
    pins = dict(s.PINS)
    for n, (name, pin) in pins.items():
        if n in (8948, 8949):
            p = sources / name
            p.write_text(
                json.dumps(
                    values[n] if n == 8948 else dict(value, failed_evidence_sha256=pins[8948][1])
                )
            )
            pins[n] = (name, s.sha256_file(p))
        else:
            (sources / name).symlink_to(s.ROOT / "results" / name)
    monkeypatch.setattr(s, "PINS", pins)
    monkeypatch.setattr(s, "references", lambda v: [])
    monkeypatch.setattr(s, "reductions", lambda v: {})
    w = s.collect(tmp_path / "raw", sources=sources)
    assert w["slots"][5]["disposition"] == "authenticated_scheduler_skip"
    assert w["slots"][5]["gate_check_summary"][0]["observed"] == 0

    path = tmp_path / (m.NAME + ".json")
    atomic_json(path, m.build(w, validation()))
    assert m.replay(path)["passed"]
