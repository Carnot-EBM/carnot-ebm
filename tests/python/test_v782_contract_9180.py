"""REQ-REPORT-9180-V782 / SCENARIO-REPORT-9180-V782: authority before science."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting.historical_contract_inputs import original_root
from carnot.reporting import v782_contract as c
from carnot.reporting import v782_contract_evidence as e


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Freeze genuine sources once; synthetic controls never become scientific evidence."""
    return c.measure(
        original_root(tmp_path_factory.mktemp("inputs9180"), 782),
        tmp_path_factory.mktemp("contract9180"),
        "20261117",
    )


def test_authority_and_protocol(work):
    """REQ-REPORT-9180-V782: every support, bound and historical failure stays fixed."""
    i = work["inspection"]
    assert all(r["passed"] for r in i["checks"]), [r for r in i["checks"] if not r["passed"]]
    assert i["source_ready"] and i["static_ready"] and i["online_ready"]
    assert i["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert i["source"]["answer_count"] == 900
    assert [len(r["qids"]) for r in i["execution_rosters"]] == [100] * 3
    assert [len(r["rows"]) for r in i["execution_rosters"]] == [300] * 3
    assert len(i["static_protocol"]["arms"]) == 5
    assert len(i["static_protocol"]["bounds"]) == 2
    assert len(i["online_protocol"]["arms"]) == 6
    assert len(i["online_protocol"]["benefit_bounds"]["bounds"]) == 7
    assert len(i["previous_outcome_rows"]) == 14
    assert sum(r["primary_status"] == "missing_file" for r in i["previous_outcome_rows"]) == 10
    o = i["validation_obligations"]
    assert len(o["original_node_ids"]) == 621 and len(o["v779_addition_ids"]) == 53
    assert len(o["node_bindings"]) == 674
    assert len(o["arc_regressions"]) == 10 and o["current_qualification_credit"] == 0
    assert any(r["node_id"].startswith(e.TEST) for r in o["affected_node_assertions"])
    assert i["diagnostic_protocol"]["brier_moments"]["identity_tolerance"] == 1e-12
    p = i["entropy_reference_protocol"]
    assert p["bootstrap_seed"] == 782190 and p["draws"] == 10000
    assert len(p["comparators"]) == 3 and p["tail"] == 0.05 / 3
    assert p["primary_gate_rescue"] is False and p["reference_role"] == "TRAIN150"
    assert i["preserved_v781"]["queue_tasks"] != i["preserved_v781"]["design_tasks"]
    for queued, original in zip(
        i["preserved_v781"]["queue_tasks"], i["preserved_v781"]["design_tasks"], strict=True
    ):
        assert queued["prompt"].format(project_root="/workspace", date="20261117") == original[
            "prompt"
        ].replace("{project_root}", "/workspace").replace("{date}", "20261117")
    assert all(r["assertion_hash"] for r in o["node_bindings"])
    assert any(
        "test_causal_validation_9167.py::" in r["node_id"] for r in o["affected_node_assertions"]
    )
    assert any(
        "test_v781_capstone_9179.py::" in r["node_id"] for r in o["affected_node_assertions"]
    )
    assert all(r["assertion_hash"] for r in o["arc_regressions"])


def test_binding_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9180-V782: fresh byte pins cannot bless changed whole objects."""
    source_root = original_root(tmp_path / "original", 782)
    for name in [e.DESIGN, e.ACTIVE]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source_root / name).read_bytes())
    assert e.authority(tmp_path)["ready"]
    original = yaml.safe_load((tmp_path / e.ACTIVE).read_bytes())
    for field in [
        "prompt",
        "MODEL_SPECS",
        "gated_on",
        "deliverable",
        "inference_substrate_class",
        "order",
        "count",
    ]:
        queue = deepcopy(original)
        if field == "order":
            queue["tasks"].reverse()
        elif field == "count":
            queue["tasks"].pop()
        else:
            queue["tasks"][1][field] = "forged"
        (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(queue))
        frozen = deepcopy(e.FROZEN)
        frozen["pins"][e.ACTIVE] = e.ref(tmp_path / e.ACTIVE)["sha256"]
        monkeypatch.setattr(e, "FROZEN", frozen)
        assert not e.authority(tmp_path)["ready"]
    (tmp_path / e.ACTIVE).unlink()
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_file"
    (tmp_path / e.ACTIVE).write_text("{}")
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_field"
    for observed in ["missing_file", "missing_field", 0, False, 1]:
        assert e.check(tmp_path, "score", 1, observed, "gate")["passed"] == (
            type(observed) is int and observed == 1
        )


def test_build(work, tmp_path, monkeypatch):
    """REQ-REPORT-9180-V782: independent readiness never grants science credit."""
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    value = c.build(work, {})
    assert value["experiment_id"] == 9180 and value["verdict_class"] == "null"
    assert all(
        value[k] == 1
        for k in [
            "contract_ready_score",
            "source_ready_score",
            "static_methods_ready_score",
            "online_methods_ready_score",
        ]
    )
    assert (
        value["MODEL_SPECS"] == []
        and value["current_llm_invocations"] == value["generated_tokens"] == 0
    )
    assert not value["activation_authorized"] and not value["current_science_paper_ready"]
    assert value["entropy_reference_protocol"] == work["inspection"]["entropy_reference_protocol"]
    assert all(
        t["canonical_task_hash"] and "previous_outcome_rows" in t
        for t in value["full_task_objects"]
    )
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: False)
    assert c.build(work, {})["verdict_class"] == "disqualified"
    missing = c.measure(tmp_path / "absent", tmp_path / "raw", "20261117")
    value = c.build(missing, {})
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 14
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert [r["unit_id"] for r in value["rows"]] == [f"exp{n}" for n in range(9180, 9194)]
    monkeypatch.setattr(c.legacy, "qualify", lambda _: dict(delegated=True))
    assert c.qualify(work) == dict(delegated=True)


def child(tmp_path, *args):
    """Exercise the standalone installed import path from private storage with child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9180] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9180] child_after",
        result.returncode,
        result.stdout[-400:],
        result.stderr[-400:],
        flush=True,
    )
    return result


def test_private_cli(work, tmp_path):
    """SCENARIO-REPORT-9180-V782: real E2E-018/021 success, refusal and rehashed tamper."""
    source, output = tmp_path / "work.json", tmp_path / (e.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
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


def test_protocol_tamper(work, tmp_path):
    """SCENARIO-REPORT-9180-V782: new hashes cannot bless a changed CES reference."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    path = tmp_path / "forged.json"
    c.atomic_json(path, dict(invented=True))
    changed["protocol_sidecars"]["entropy_reference_protocol_reference"] = e.ref(path)
    c.atomic_json(tmp_path / "work.json", changed)
    output = tmp_path / (e.NAME + ".json")
    c.atomic_json(output, c.build(changed, {}))
    with pytest.raises(ValueError, match="entropy_protocol_drift"):
        c.replay(output)


@pytest.mark.parametrize("kind", ["primary", "work", "scheduler", "arc_log"])
def test_producer_snapshot_refusal(work, monkeypatch, kind):
    """SCENARIO-REPORT-9180-V782: failed original work bytes block dependent imports."""
    source_root = Path(work["root"])
    original = e.ref
    source = json.loads((e.ROOT / "results/experiment_9166_v781_contract.json").read_bytes())
    capstone = json.loads((e.ROOT / "results/experiment_9179_v781_capstone.json").read_bytes())
    saved = json.loads(Path(capstone["work_reference"]["path"]).read_bytes())
    arc = json.loads(
        Path(
            saved["snapshots"]["results/experiment_9178_v781_arc_mechanic_routing.json"]["path"]
        ).read_bytes()
    )
    operand = {
        "primary": str(source_root / "results/experiment_9166_v781_contract.json"),
        "work": source["work_reference"]["path"],
        "scheduler": saved["snapshots"]["ops/conductor-log.md"]["path"],
        "arc_log": next(
            r["stdout_path"] for r in arc["validation_receipts"] if r["name"] == "direct_consumers"
        ),
    }[kind]

    def tampered(path):
        reference = original(path)
        if str(path) == operand:
            reference["sha256"] = None
        return reference

    monkeypatch.setattr(e, "ref", tampered)

    def forbidden(_):
        raise AssertionError("dependent labels inspected before producer authentication")

    if kind != "arc_log":
        monkeypatch.setattr(e.prior, "inspect", forbidden)
    value = e.inspect(source_root)
    assert not value["source_ready"]
    assert any(r["observed"] == "missing_file" for r in value["checks"])


def test_shared_deadline(monkeypatch):
    """REQ-REPORT-9180-V782: every supervised descendant shares the invocation deadline."""
    seen = []
    monkeypatch.delenv("CARNOT_V782_DEADLINE", raising=False)

    def record(*args, **kwargs):
        seen.append(kwargs["deadline"])
        return {}

    monkeypatch.setattr(c.legacy.lifecycle, "child", record)

    def run(_):
        assert c.os.environ["CARNOT_V782_DEADLINE"] == "4800.0"
        c.legacy.lifecycle.child("owned_tests", [], Path("/tmp"), deadline=600)
        return 0

    monkeypatch.setattr(c, "_main", run)
    clock = iter([0, 4400])
    monkeypatch.setattr(c.time, "monotonic", lambda: next(clock))
    assert c.main([]) == 0 and seen == [100]
    clock = iter([0, 4801])
    monkeypatch.setattr(c.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="shared_deadline"):
        c.main([])


def test_original_source_refusal(work, monkeypatch):
    """SCENARIO-REPORT-9180-V782: original method failure cannot inherit readiness."""
    monkeypatch.setattr(
        e.prior,
        "inspect",
        lambda _: dict(
            checks=[e.check(e.ROOT, "original_source", 1, 0, "original_source")],
            source_references=[],
            source_ready=False,
        ),
    )
    assert not e.inspect(Path(work["root"]))["source_ready"]
