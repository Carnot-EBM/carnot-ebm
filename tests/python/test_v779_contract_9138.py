"""REQ-REPORT-9138-V779 / SCENARIO-REPORT-9138-V779: custody and real CLI controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v779_contract as c
from carnot.reporting import v779_contract_evidence as e


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate real retained evidence once without rerunning method development."""
    raw = tmp_path_factory.mktemp("contract9138")
    return c.measure(e.ROOT, raw, "20261116")


def test_source_and_methods(work):
    """REQ-REPORT-9138-V779: imported methods and human authority remain separate."""
    inspected = work["inspection"]
    assert all(r["passed"] for r in inspected["checks"]), [
        r for r in inspected["checks"] if not r["passed"]
    ]
    assert inspected["source_ready"] and inspected["static_ready"] and inspected["online_ready"]
    source = inspected["source"]
    assert source["answer_count"] == 900 and source["qid_count"] == 300
    assert source["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert source["shards"] == [150, 150]
    assert len(inspected["static_protocol"]["arms"]) == 5
    assert len(inspected["online_protocol"]["arms"]) == 6
    assert len(inspected["online_protocol"]["benefit_bounds"]["bounds"]) == 7
    retired = inspected["retired_qualification"]
    assert retired["receipt_ready_score"] == 0 and retired["bounded_scope_retired"]
    assert [retired[k] for k in ["intended_count", "completed_count", "censored_count"]] == [
        621,
        48,
        573,
    ]
    assert len(inspected["previous_outcome_rows"]) == 14
    assert inspected["archive_lag"]["completed_scheduling_frontier"] == "2026.11.778"
    assert inspected["method_receipt_replays"] == dict(static=True, online=True)


def test_full_contract_and_mutations(tmp_path):
    """SCENARIO-REPORT-9138-V779: complete objects reject hidden metadata changes."""
    for name in [e.DESIGN, e.ACTIVE]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    valid = e.authority(tmp_path)
    assert valid["ready"] and len(valid["tasks"]) == 14
    assert [t["id"].split("-")[0] for t in valid["tasks"]] == [f"exp{i}" for i in range(9138, 9152)]
    queue = yaml.safe_load((tmp_path / e.ACTIVE).read_bytes())
    for kind in ["prompt", "gate", "model", "order", "count"]:
        changed = deepcopy(queue)
        if kind == "prompt":
            changed["tasks"][0]["prompt"] += " forged"
        elif kind == "gate":
            changed["tasks"][1]["gated_on"][0]["value"] = 0
        elif kind == "model":
            changed["tasks"][0]["MODEL_SPECS"] = [{"hf_id": "invented"}]
        elif kind == "order":
            changed["tasks"].reverse()
        else:
            changed["tasks"].pop()
        (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(changed))
        assert not e.authority(tmp_path)["ready"]
    (tmp_path / e.ACTIVE).unlink()
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_file"
    (tmp_path / e.ACTIVE).write_text("{}")
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_field"


def test_gates(tmp_path):
    """REQ-REPORT-9138-V779: bool, absent fields and zero are different operands."""
    p = tmp_path / "experiment_9126_methods.json"
    for observed in ["missing_file", "missing_field", 0, False, 1]:
        row = e.check(p, "methods_ready_score", 1, observed, "source_gate")
        assert row["observed"] == observed
        assert row["passed"] == (type(observed) is int and observed == 1)
        assert row["upstream"] == "exp9126"


def test_blocking_and_independent_contract(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9138-V779: absent science cannot invalidate an exact schedule."""
    absent = c.measure(tmp_path / "absent", tmp_path / "blocked", "20261116")
    value = c.build(absent, {})
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 14
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    changed = deepcopy(work)
    changed["inspection"]["source_ready"] = False
    value = c.build(changed, {})
    assert value["contract_ready_score"] == 1 and value["source_ready_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["canonical_task_hash"] == e.canonical_hash(value["full_task_objects"])
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert value["sample_size_budget"]["effective_independent_groups"] == 0


def child(tmp_path, *args):
    """Run the real entry point outside the checkout, including child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9138] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9138] child_after",
        result.returncode,
        result.stdout[-500:],
        result.stderr[-500:],
        flush=True,
    )
    return result


def test_private_cli(work, tmp_path):
    """SCENARIO-REPORT-9138-V779: success, refusal, cold replay and rehashed tamper."""
    source, output = tmp_path / "work.json", tmp_path / (e.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["contract_ready_score"] == 0  # Unvalidated private work cannot claim readiness.
    value["source_ready_score"] = 99
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 1
    assert child(tmp_path, "--from-work", source).returncode == 1
    assert (
        child(
            tmp_path, "--checks-only", "--root", tmp_path / "absent", "--output", output
        ).returncode
        == 0
    )


def test_replay_rejects_owned_drift(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9138-V779: rehashing work cannot hide source or ledger tampering."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    path = tmp_path / (e.NAME + ".json")
    c.atomic_json(tmp_path / "work.json", changed)
    value = c.build(changed, {})
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    assert c.main(["--cold-replay", str(path)]) == 0
    forged = dict(value, source_ready_score=99)
    forged["reproducibility_checksum"] = c.checksum(forged)
    c.atomic_json(path, forged)
    with pytest.raises(ValueError, match="reduction_drift"):
        c.replay(path)
    altered = deepcopy(value)
    altered["work_reference"]["sha256"] = "sha256:forged"
    c.atomic_json(path, altered)
    with pytest.raises(ValueError, match="work_hash"):
        c.replay(path)
    changed["inspection"]["source"]["answer_count"] = 1
    c.atomic_json(tmp_path / "work.json", changed)
    c.atomic_json(path, c.build(changed, {}))
    with pytest.raises(ValueError, match="source_drift"):
        c.replay(path)
    changed["inspection"] = deepcopy(work["inspection"])
    changed["current_work_reference"]["sha256"] = "sha256:forged"
    c.atomic_json(tmp_path / "work.json", changed)
    c.atomic_json(path, c.build(changed, {}))
    with pytest.raises(ValueError, match="current_work_hash"):
        c.replay(path)
    changed["current_work_reference"] = deepcopy(work["current_work_reference"])
    c.atomic_json(tmp_path / "work.json", changed)
    c.atomic_json(path, c.build(changed, {}))
    monkeypatch.setattr(c, "validate_current_work_receipt", lambda *_: ["forged event"])
    with pytest.raises(ValueError, match="current_work_drift"):
        c.replay(path)


def test_qualified_and_failed_validation(work, tmp_path, monkeypatch):
    """REQ-REPORT-9138-V779: production readiness requires actual affected validation."""
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    value = c.build(work, {})
    assert value["verdict_class"] == "null"
    assert all(
        value[k] == 1
        for k in [
            "contract_ready_score",
            "source_ready_score",
            "static_methods_ready_score",
            "online_methods_ready_score",
        ]
    )
    monkeypatch.setattr(c, "receipts_pass", lambda *_: False)
    assert c.build(work, {})["verdict_class"] == "disqualified"
    monkeypatch.setattr(c.lifecycle, "qualify", lambda _: dict(real_delegation=True))
    assert c.qualify(work) == dict(real_delegation=True)
    monkeypatch.setattr(c, "measure", lambda *_: deepcopy(work))
    monkeypatch.setattr(c, "qualify", lambda _: {})
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + 1)
    output = tmp_path / (e.NAME + ".json")
    assert c.main(["--output", str(output)]) == 1
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    monkeypatch.setattr(c, "terminal", lambda *_: dict(passed=True))
    assert c.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["contract_ready_score"] == 1
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + int(3301e9))
    assert c.main(["--output", str(output)]) == 1
    times = iter([work["start_ns"] + 1, work["start_ns"] + int(4801e9)])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(times))
    assert c.main(["--output", str(output)]) == 1


def test_retained_input_refusal(work, tmp_path, monkeypatch):
    """REQ-REPORT-9138-V779: altered producer bytes fail before dependent field access."""
    original = e.ref
    capstone_work = next(
        r["path"]
        for r in work["inspection"]["source_references"]
        if r.get("producer_path", "").endswith("experiment_9097_v776_human_authority.json")
    )

    def bad_human(path):
        result = original(path)
        if str(path) == capstone_work:
            result["sha256"] = "sha256:forged"
        return result

    monkeypatch.setattr(e, "ref", bad_human)
    assert not e.inspect(e.ROOT)["source_ready"]
    capstone = json.loads((e.ROOT / "results/experiment_9137_v778_capstone.json").read_bytes())

    def bad_capstone(path):
        result = original(path)
        if str(path) == capstone["work_reference"]["path"]:
            result["sha256"] = None
        return result

    monkeypatch.setattr(e, "ref", bad_capstone)
    assert not e.inspect(e.ROOT)["source_ready"]
    for name in [e.DESIGN, e.ACTIVE]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    (tmp_path / e.ACTIVE).write_text("tasks: [\n")
    assert not e.authority(tmp_path)["ready"]
