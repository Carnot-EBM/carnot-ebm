"""REQ-REPORT-9067-V773: private controls never supply scientific measurements."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v773_capstone as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Use original source bytes, with measurements left absent where unavailable."""
    return m.collect(tmp_path_factory.mktemp("capstone9067"), "20261112")


def receipts(work, tmp_path):
    """Administrative receipt controls remain private and never enter production evidence."""
    w = deepcopy(work)
    w["validation"] = dict(receipts=[], coverage=dict(complete=True))
    tmp_path.mkdir(parents=True, exist_ok=True)
    for c in w["validation_scope_manifest"]["commands"]:
        p = tmp_path / (c["name"] + ".log")
        p.write_text("private administrative control\n")
        w["validation"]["receipts"].append(
            dict(
                name=c["name"],
                passed=True,
                exit_code=0,
                normal_exit=True,
                timed_out=False,
                started_monotonic_ns=w["frozen_ns"] + 1,
                ended_monotonic_ns=w["frozen_ns"] + 2,
                scope_sha256=canonical_hash(w["validation_scope_manifest"]),
                command_argv=c["argv"],
                log_path=str(p),
                log_sha256=m.reference(p)["sha256"],
            )
        )
    return w


def child(tmp_path, *args):
    """Run the actual CLI outside the checkout without ambient import paths."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9067] child_before completed=0 pending=1", flush=True)
    r = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("[test9067] child_after completed=1 pending=0", r.returncode, flush=True)
    return r


def test_exact_accounting(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9067-EXACT-ACCOUNTING: all slots close without invented benefits."""
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    v = m.build(receipts(work, tmp_path))
    assert v["reconciliation_ready_score"] == 1 and v["verdict_class"] == "blocked"
    assert v["intended_count"] == v["completed_count"] == len(v["slot_rows"]) == 14
    assert len(v["full_contract_rows"]) == 14 and all(r["matched"] for r in v["full_contract_rows"])
    slots = {r["experiment_id"]: r for r in v["slot_rows"]}
    assert slots[9057]["honest_verdict"] is slots[9057]["verdict_class"] is None
    assert slots[9057]["disposition"] == "authenticated_scheduler_skip"
    assert slots[9059]["disposition"] == "missing_primary"
    assert slots[9058]["honest_verdict"] == "complete_circular_positive_fresh_corpus_a"
    assert v["fresh_source_summary"]["completed_count"] == 1024
    assert v["fresh_source_summary"]["proposal_count"] == 47987
    assert v["historical_fresh_source_summary"]["completed_count"] == 414
    assert v["human_authority_summary"]["completed_count"] == 4800
    assert not v["human_authority_summary"]["human_source_ready_score"]
    for name, count in [("seven_bound_rows", 7), ("ten_bound_rows", 10), ("four_bound_rows", 4)]:
        assert len(v[name]) == count
        assert all(
            r["status"] == "unmeasured" and r["value"] is None and r["denominator"] == 0
            for r in v[name]
        )
    assert not any(v["model_invocation_counts"].values())
    assert v["MODEL_SPECS"] == v["model_specs"] == v["trained_head_specs"] == []
    assert not v["paper_ready"] and not v["repository_health_passed"]
    assert any(
        r["scope"] == "MDH_400_group_human_authority" and r["status"] == "closed_unchanged"
        for r in v["continuation_evidence"]
    )


@pytest.mark.parametrize(
    "mutation",
    ["valid", "missing", "stale", "failed", "timeout", "log", "command", "duplicate", "code"],
)
def test_receipts_and_foreign_cli(work, tmp_path, mutation):
    """SCENARIO-REPORT-9067-V773: altered receipt evidence refuses readiness."""
    w = receipts(work, tmp_path)
    m.runner.mutate_receipt(w, mutation)
    p = tmp_path / "work.json"
    atomic_json(p, w)
    assert m.receipts_pass(w) == (mutation == "valid")
    assert child(tmp_path, "--check-receipts", p).returncode == int(mutation != "valid")


def test_cold_replay_and_rehashed_tamper(work, tmp_path):
    """SCENARIO-REPORT-9067-EXACT-ACCOUNTING: independent recount rejects new hashes."""
    p = tmp_path / "candidate.json"
    v = m.build(work)
    atomic_json(p, v)
    assert m.replay(p)["passed"]
    assert child(tmp_path, "--cold-replay", p).returncode == 0
    v["fresh_source_summary"]["completed_count"] = 1
    v["reproducibility_checksum"] = m.checksum(v)
    atomic_json(p, v)
    assert not m.replay(p)["passed"]
    assert child(tmp_path, "--cold-replay", p).returncode == 1
    assert not m.replay(tmp_path / "absent")["passed"]
    w = deepcopy(work)
    w["contract"]["tasks"][0]["prompt"] += "changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(w)
    r = deepcopy(work["contract"])
    changed = tmp_path / "changed.md"
    changed.write_bytes(Path(r["design_reference"]["path"]).read_bytes() + b"changed")
    r["design_reference"].update(m.reference(changed))
    assert not m.contract(r)["ready"]


def test_external_absence_and_zero(work, tmp_path, monkeypatch):
    """REQ-REPORT-9067-V773: external absence stays terminal with honest null fields."""
    empty = m.collect(tmp_path / "empty", "20261112", tmp_path / "missing")
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    v = m.build(receipts(empty, tmp_path / "receipts"))
    assert v["reconciliation_ready_score"] == 1 and v["verdict_class"] == "blocked"
    assert v["completed_count"] == 14 and v["censored_count"] == 0
    assert all(r["honest_verdict"] is None for r in v["slot_rows"][:-1])
    gate = dict(upstream="missing", artifact_field="score", op="==", value=1)
    assert m.edge(gate, {})["observed"] == "missing_file"
    assert m.edge(gate, {"missing": dict(value={}, reference={})})["observed"] == "missing_field"
    assert m.edge(gate, {"missing": dict(value=dict(score=0), reference={})})["observed"] == 0
    assert m.reduce(work)["arc_summary"]["new_uncensored_episode_count"] == 0


def test_private_cli_failure_and_publication(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9067-V773: failed owned evidence stays private."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    assert child(tmp_path, "--from-work", source, "--output", output).returncode == 1
    assert not output.exists()
    assert child(tmp_path, "--date", "wrong").returncode == 2
    assert child(tmp_path, "--from-work", source).returncode == 2
    observed = receipts(work, tmp_path / "receipts")

    def collect(raw, *args):
        w = deepcopy(observed)
        w["raw"] = str(raw)
        return w

    monkeypatch.setattr(m, "collect", collect)
    monkeypatch.setattr(m, "validate", lambda _: deepcopy(observed["validation"]))
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    monkeypatch.setattr(m, "terminal", lambda *_: dict(passed=False, receipts=[]))
    assert m.main(["--output", str(output)]) == 1
    monkeypatch.setattr(m, "terminal", lambda *_: dict(passed=True, receipts=[]))
    assert m.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["reconciliation_ready_score"] == 1
    observed["started_ns"] -= 3301 * 10**9
    assert m.main(["--output", str(output)]) == 1
    monkeypatch.setattr(pytest, "main", lambda *a, **kw: 0)
    assert m.main(["--direct-consumers", "private-control"]) == 0


def test_validation_terminal_and_refusals(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9067-EXACT-ACCOUNTING: unchanged scanners and raw bindings reject drift."""
    w = deepcopy(work)
    w["raw"], w["private"] = str(tmp_path / "raw"), str(tmp_path / "private")
    private = Path(w["private"])
    private.mkdir()
    (private / "coverage.ini").write_text("[run]\n")
    spec = m.CommandSpec(
        "private_check", (sys.executable, "-c", "print('actual normal exit')"), "test", 30
    )
    w["validation_scope_manifest"]["commands"] = [
        dict(name=spec.name, argv=list(spec.argv), timeout_s=spec.timeout_s)
    ]
    monkeypatch.setattr(m.runner, "validation_plan", lambda _: [spec])
    report = m.validate(w)
    assert report["receipts"][0]["normal_exit"] and not report["coverage"]["complete"]
    p = tmp_path / "candidate.json"
    atomic_json(p, m.build(work))
    assert m.terminal(p, tmp_path / "terminal", work["validation_scope_manifest"])["passed"]
    for index, reason in [(9, "registered_raw_reducer"), (10, "registered_human_reducer")]:
        w = deepcopy(work)
        w["slots"][index]["qualified"] = True
        with pytest.raises(ValueError, match=reason):
            m.reduce(w)
    w = deepcopy(work)
    task_id = w["tasks"][2]["id"]
    primitive = m.load(w["primitive_references"][task_id])
    primitive["reduction"]["rows"][0]["numerator"] = 100
    source = tmp_path / "primitive.json"
    atomic_json(source, primitive)
    w["primitive_references"][task_id] = m.reference(source)
    with pytest.raises(ValueError, match="primitive_binding"):
        m.reduce(w)


def test_rejected_bytes_and_changed_work(work, tmp_path):
    """REQ-REPORT-9067-V773: rejected primaries stay available without qualifying science."""
    selected = tmp_path / "sources"
    selected.mkdir()
    v = deepcopy(work["values"][work["tasks"][1]["id"]]["original"])
    log = tmp_path / "failed.log"
    log.write_text("private failed producer receipt control\n")
    v["validation_receipts"] = [
        dict(passed=False, log_path=str(log), log_sha256=m.reference(log)["sha256"])
    ]
    atomic_json(selected / Path(work["tasks"][1]["deliverable"]).name, v)
    w = m.collect(tmp_path / "rejected", "20261112", selected)
    assert any(r.get("original_path") == str(log) for r in w["references"])
    assert not w["slots"][1]["qualified"]
    for mutation in ["scheduler", "producer", "checksum"]:
        w = deepcopy(work)
        if mutation == "scheduler":
            w["scheduler_lines"] = []
        else:
            w["values"][w["tasks"][1]["id"]]["original"]["completed_count"] = 0
        source = tmp_path / "changed_work.json"
        atomic_json(source, w)
        v = m.build(work)
        v["work_reference"] = m.reference(source)
        v["reproducibility_checksum"] = "changed" if mutation == "checksum" else m.checksum(v)
        p = tmp_path / "candidate.json"
        atomic_json(p, v)
        assert not m.replay(p)["passed"]
