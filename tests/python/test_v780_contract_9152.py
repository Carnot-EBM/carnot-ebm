"""REQ-REPORT-9152-V780 / SCENARIO-REPORT-9152-V780: exact custody and CLI controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v780_contract as c
from carnot.reporting import v780_contract_evidence as e


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Retain real authenticated operands once; fixtures grant no scientific benefit."""
    return c.measure(e.ROOT, tmp_path_factory.mktemp("contract9152"), "20261116")


def test_authority_and_partition(work):
    """REQ-REPORT-9152-V780: scheduling, source and methods have independent gates."""
    i = work["inspection"]
    assert all(r["passed"] for r in i["checks"]), [r for r in i["checks"] if not r["passed"]]
    assert i["source_ready"] and i["static_ready"] and i["online_ready"]
    assert len(i["contract"]["tasks"]) == 14
    assert i["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    rosters = i["execution_rosters"]
    assert [len(r["qids"]) for r in rosters] == [100, 100, 100]
    assert [len(r["rows"]) for r in rosters] == [300, 300, 300]
    assert len({r["unit_id"] for p in rosters for r in p["rows"]}) == 900
    assert i["partition_union_equals_original"]
    assert len(i["static_protocol"]["arms"]) == 5
    assert len(i["static_protocol"]["bounds"]) == 2
    assert len(i["online_protocol"]["arms"]) == 6
    assert len(i["online_protocol"]["benefit_bounds"]["bounds"]) == 7
    assert len(i["previous_outcome_rows"]) == 14
    retired = i["retired_schedule"]
    assert [retired[k] for k in ["original_nodes", "v779_additions", "completed", "censored"]] == [
        621,
        53,
        23,
        651,
    ]
    assert retired["sum_of_watchdog_ceilings_s"] == 27880.142389
    assert retired["complete_runtime_s"] is None
    assert retired["unknown_slow_child_cost_s"] is None
    assert len(retired["original_node_ids"]) == 621
    assert len(retired["new_node_ids"]) == 53
    assert set(retired["completed_node_ids"]).isdisjoint(retired["original_node_ids"])
    assert i["diagnostic_protocol"]["scale_ranking"]["one_class_auroc"] == "unavailable"


def test_authority_mutations(tmp_path):
    """SCENARIO-REPORT-9152-V780: prompt, metadata, order and gates all bind bytes."""
    for name in [e.DESIGN, e.ACTIVE]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    assert e.authority(tmp_path)["ready"]
    original = yaml.safe_load((tmp_path / e.ACTIVE).read_bytes())
    for key in ["prompt", "MODEL_SPECS", "deliverable", "gated_on", "order", "count"]:
        q = deepcopy(original)
        if key == "order":
            q["tasks"].reverse()
        elif key == "count":
            q["tasks"].pop()
        else:
            q["tasks"][1][key] = "forged"
        (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(q))
        assert not e.authority(tmp_path)["ready"]
    (tmp_path / e.ACTIVE).unlink()
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_file"
    (tmp_path / e.ACTIVE).write_text("{}")
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_field"


def test_missing_and_zero(work, tmp_path, monkeypatch):
    """REQ-REPORT-9152-V780: missing operands and zero must never share a verdict operand."""
    absent = c.measure(tmp_path / "absent", tmp_path / "raw", "20261116")
    value = c.build(absent, {})
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 14
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert [r["unit_id"] for r in value["rows"]] == [f"exp{n}" for n in range(9152, 9166)]
    for observed in ["missing_file", "missing_field", 0, False, 1]:
        assert e.check(tmp_path, "score", 1, observed, "gate")["passed"] == (
            type(observed) is int and observed == 1
        )
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    changed = deepcopy(work)
    changed["inspection"]["source_ready"] = False
    value = c.build(changed, {})
    assert value["contract_ready_score"] == 1 and value["source_ready_score"] == 0
    assert changed["inspection"]["protocol_references"] == work["inspection"]["protocol_references"]
    assert value["MODEL_SPECS"] == []
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert all(
        "canonical_task_hash" in t and "previous_outcome_rows" in t
        for t in value["full_task_objects"]
    )
    assert (
        value["shard_c_roster_reference"]
        == value["protocol_references"]["shard_c_roster_reference"]
    )


def child(tmp_path, *args):
    """Actual foreign-process entry points also contribute owned statement coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9152] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9152] child_after",
        result.returncode,
        result.stdout[-300:],
        result.stderr[-300:],
        flush=True,
    )
    return result


def test_private_cli(work, tmp_path):
    """SCENARIO-REPORT-9152-V780: E2E-018/021 success, refusal and rehashed cold tamper."""
    source, output = tmp_path / "work.json", tmp_path / (e.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["contract_ready_score"] == 0
    value["source_ready_score"] = 99
    value["reproducibility_checksum"] = c.legacy.checksum(value)
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


def test_producer_refusal(work, monkeypatch):
    """SCENARIO-REPORT-9152-V780: tampering fails before any dependent label access."""
    original = e.ref
    for suffix in ["experiment_9141_v779_human_contrast_audit.json", "diagnostic_protocol.json"]:

        def forged(path):
            r = original(path)
            if str(path).endswith(suffix):
                r["sha256"] = None
            return r

        monkeypatch.setattr(e, "ref", forged)
        assert not e.inspect(e.ROOT)["source_ready"]


def test_adapter_lifecycle(work, tmp_path, monkeypatch):
    """REQ-REPORT-9152-V780: reused publication validates the new producer identity."""
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    value = c.build(work, {})
    assert value["verdict_class"] == "null"
    assert value["experiment_id"] == 9152 and value["task_id"] == e.TASK
    assert value["contract_ready_score"] == value["online_methods_ready_score"] == 1
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: False)
    assert c.build(work, {})["verdict_class"] == "disqualified"
    monkeypatch.setattr(c.legacy.lifecycle, "qualify", lambda _: dict(delegated=True))
    assert c.qualify(work) == dict(delegated=True)


def test_sidecar_tamper(work, tmp_path):
    """SCENARIO-REPORT-9152-V780: freshly rehashed roster bytes cannot invent population."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    path = tmp_path / "forged_roster.json"
    c.atomic_json(path, dict(qids=["invented"]))
    changed["v780_protocol_references"]["shard_c_roster_reference"] = e.ref(path)
    c.atomic_json(tmp_path / "work.json", changed)
    output = tmp_path / (e.NAME + ".json")
    c.atomic_json(output, c.build(changed, {}))
    with pytest.raises(ValueError, match="protocol_sidecar_drift"):
        c.replay(output)


def test_authenticate_before_labels(monkeypatch):
    """SCENARIO-REPORT-9152-V780: failed new authority stops before historical label access."""
    original = e.ref

    def absent(path):
        r = original(path)
        if str(path).endswith("experiment_9141_v779_human_contrast_audit.json"):
            r["sha256"] = None
        return r

    def forbidden(_):
        raise AssertionError("dependent label access before authority authentication")

    monkeypatch.setattr(e, "ref", absent)
    monkeypatch.setattr(e.prior, "inspect", forbidden)
    assert not e.inspect(e.ROOT)["source_ready"]
