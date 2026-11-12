"""REQ-REPORT-9053-V772: private accounting controls grant no scientific credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v772_capstone as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Use actual retained bytes; absent measurements remain absent."""
    return m.collect(tmp_path_factory.mktemp("capstone9053"), "20261112")


def receipts(work, tmp_path):
    """Private receipt fixtures test acceptance only and never enter the primary."""
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
    """A real child must import and exit without the checkout as its working directory."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9053] child_before completed=0 pending=1", flush=True)
    r = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("[test9053] child_after completed=1 pending=0", r.returncode, flush=True)
    return r


def test_exact_accounting(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9053-ACCOUNTING: readiness never invents a scientific result."""
    w = receipts(work, tmp_path)
    monkeypatch.setattr(m, "verify_measured", lambda *_: True)
    v = m.build(w)
    assert v["reconciliation_ready_score"] == 1 and v["verdict_class"] == "blocked"
    assert v["completed_count"] == len(v["rows"]) == len(v["slot_rows"]) == 13
    slots = {r["experiment_id"]: r for r in v["slot_rows"]}
    assert all(slots[n]["qualified"] for n in range(9041, 9044))
    assert not slots[9044]["qualified"]
    assert slots[9044]["honest_verdict"] == "complete_blocked_source_identity_ready_score"
    assert slots[9044]["current_evidence_verdict_class"] == "disqualified"
    assert slots[9045]["disposition"] == "authenticated_scheduler_skip"
    assert slots[9045]["honest_verdict"] is slots[9045]["verdict_class"] is None
    assert slots[9050]["disposition"] == "conductor_skip_without_primary"
    assert all(r["denominator"] == 0 and r["value"] is None for r in v["seven_bound_rows"])
    assert len(v["seven_bound_rows"]) == 7 and len(v["ten_bound_rows"]) == 10
    natural = v["natural_feasibility_summary"]
    assert natural["identity"]["category_counts"]["missing_page"] == 206
    assert natural["identity"]["category_counts"]["existing_page_without_qid"] == 9
    assert natural["annotation"]["record_count"] == 2401
    assert natural["annotation"]["natural_pilot_ready_score"] == 0
    assert natural["annotation"]["reported_annotation_structure_ready_score"] == 1
    assert natural["annotation"]["current_participation_authority_ready_score"] == 0
    assert natural["annotation"]["participation_mismatch_count"] > 0
    assert v["fresh_source_summary"]["completed_count"] == 414
    assert v["fresh_source_summary"]["censored_count"] == 610
    assert not v["paper_ready"] and not v["repository_health_passed"]
    assert not any(v["model_invocation_counts"].values())
    assert v["MODEL_SPECS"] == v["model_specs"] == v["trained_head_specs"] == []
    assert len(v["full_contract_rows"]) == 13
    assert all(r["matched"] for r in v["full_contract_rows"])
    assert all("retire_if_same_verdict" in r for r in v["retirement_scope_rows"])
    assert any(r["scope"] == "old_A_B_reuse" for r in v["continuation_evidence"])


@pytest.mark.parametrize(
    "mutation",
    ["valid", "missing", "stale", "failed", "timeout", "log", "command", "duplicate", "code"],
)
def test_receipts_and_foreign_cli(work, tmp_path, mutation):
    """SCENARIO-REPORT-9053-V772: missing or altered normal-exit evidence refuses readiness."""
    w = receipts(work, tmp_path)
    m.runner.mutate_receipt(w, mutation)
    p = tmp_path / "work.json"
    atomic_json(p, w)
    assert m.receipts_pass(w) == (mutation == "valid")
    assert child(tmp_path, "--check-receipts", p).returncode == int(mutation != "valid")


def test_missing_authority_and_rehashed_tamper(work, tmp_path):
    """SCENARIO-REPORT-9053-ACCOUNTING: rehashing cannot change immutable operands."""
    p = tmp_path / "candidate.json"
    v = m.build(work)
    atomic_json(p, v)
    assert m.replay(p)["passed"]
    assert child(tmp_path, "--cold-replay", p).returncode == 0
    v["natural_feasibility_summary"]["identity"]["category_counts"]["missing_page"] = 0
    v["reproducibility_checksum"] = m.checksum(v)
    atomic_json(p, v)
    assert not m.replay(p)["passed"]
    assert child(tmp_path, "--cold-replay", p).returncode == 1
    assert not m.replay(tmp_path / "absent")["passed"]
    w = deepcopy(work)
    w["contract"]["tasks"][0]["prompt"] += "changed"
    with pytest.raises(ValueError, match="task_authority"):
        m.build(w)
    empty = m.collect(tmp_path / "empty_raw", "20261112", tmp_path / "absent")
    assert all(r["primary_status"] == "missing_file" for r in empty["slots"])
    assert m.build(empty)["reconciliation_ready_score"] == 0
    gate = dict(upstream="missing", artifact_field="score", op="==", value=1)
    assert m.edge(gate, {})["observed"] == "missing_file"
    assert m.edge(gate, {"missing": dict(value={}, reference={})})["observed"] == "missing_field"
    assert m.edge(gate, {"missing": dict(value=dict(score=0), reference={})})["observed"] == 0


def test_private_cli_failure_and_publication(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9053-V772: failed owned work stays private; passing checks publish atomically."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (m.NAME + ".json")
    r = child(tmp_path, "--from-work", source, "--output", output)
    assert r.returncode == 1 and not output.exists()
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
    """SCENARIO-REPORT-9053-ACCOUNTING: real scanners and frozen commands reject changed claims."""
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
    w = deepcopy(work)
    w["slots"][7]["qualified"] = True
    with pytest.raises(ValueError, match="registered_raw_reducer"):
        m.reduce(w)
    w = deepcopy(work)
    r = w["contract"]["design_reference"]
    changed = tmp_path / "changed.md"
    changed.write_bytes(Path(r["path"]).read_bytes().replace(b"13 tasks", b"14 tasks", 1))
    r.update(m.reference(changed))
    assert not m.contract(w["contract"])["ready"]


def test_rejected_bytes_and_primitive_refusal(work, tmp_path, monkeypatch):
    """REQ-REPORT-9053-V772: retain failed operands and refuse invented primitive counts."""
    selected = tmp_path / "sources"
    selected.mkdir()
    original = work["values"][work["tasks"][1]["id"]]["original"]
    v = deepcopy(original)
    log = tmp_path / "failed.log"
    log.write_text("private failed producer receipt control\n")
    v["validation_receipts"] = [
        dict(passed=False, log_path=str(log), log_sha256=m.reference(log)["sha256"])
    ]
    atomic_json(selected / Path(work["tasks"][1]["deliverable"]).name, v)
    w = m.collect(tmp_path / "rejected", "20261112", selected)
    assert any(r.get("original_path") == str(log) for r in w["references"])
    assert not w["slots"][1]["qualified"]
    w = deepcopy(work)
    w["values"][w["tasks"][2]["id"]]["original"]["completed_count"] = 0
    with pytest.raises(ValueError, match="natural_primitive_rows"):
        m.reduce(w)
    w = deepcopy(work)
    w["scheduler_lines"] = []
    p = tmp_path / "tampered_work.json"
    atomic_json(p, w)
    v = m.build(work)
    v["work_reference"] = m.reference(p)
    v["reproducibility_checksum"] = m.checksum(v)
    candidate = tmp_path / "tampered.json"
    atomic_json(candidate, v)
    assert not m.replay(candidate)["passed"]
    v["reproducibility_checksum"] = "changed"
    atomic_json(candidate, v)
    assert not m.replay(candidate)["passed"]


def test_rehashed_primitive_cannot_replace_original(work, tmp_path):
    """SCENARIO-REPORT-9053-V772: a newly hashed primitive must still match the pinned producer."""
    w = deepcopy(work)
    task_id = w["tasks"][2]["id"]
    primitive = m.load(w["primitive_references"][task_id])
    primitive["reduction"]["obstruction_rows"][0]["category"] = "invented"
    source = tmp_path / "primitive.json"
    atomic_json(source, primitive)
    w["primitive_references"][task_id] = m.reference(source)
    with pytest.raises(ValueError, match="primitive_binding"):
        m.reduce(w)


def test_arc_support_recounts_retained_events(work):
    """REQ-REPORT-9053-V772: independently derive optional support without running a game."""
    summary = m.reduce(work)["arc_summary"]
    assert summary["source_event_count"] == summary["new_uncensored_episode_count"] == 0
    assert summary["source_reference"]["sha256"]
    assert not summary["ready"] and not summary["qualified_association"]
