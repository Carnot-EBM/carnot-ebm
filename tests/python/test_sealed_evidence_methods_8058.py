"""REQ-REPORT-8058: exposed method custody never grants scientific benefit."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from carnot import experiment_8058_v698_sealed_evidence_methods as m
from carnot.reporting.current_work_receipt import atomic_json


def test_scenario_report_8058_methods():
    """SCENARIO-REPORT-8058-METHODS: costs, clocks and information stay sealed."""
    v = m.methods()
    assert v["learning"]["arms"] == ["frozen", "unconditional", "reused_guard", "fresh_admission"]
    assert v["learning"]["attempt_slots"] == [64, 128, 192]
    assert v["learning"]["delay"] == 20
    assert v["learning"]["newest_update_rows"] == 32
    assert v["learning"]["minimum_update_rows"] == 16
    assert v["learning"]["admission_rows"] == 12
    assert v["learning"]["cap_steps"] == 12
    assert v["guard"]["alphas"] == [1, 0.5, 0.25, 0.125, 0]
    assert v["guard"]["versus_initial"] == {"brier": 0.01, "typed_cost": 0.02}
    assert v["guard"]["versus_incumbent"] == {"brier": 0, "typed_cost": 0}
    assert v["learning"]["guard_gradients"] is False
    assert m.partition("a") == int(hashlib.sha256(b"a").hexdigest(), 16) % 4
    assert {r["id"] for r in v["method_source_map"]} >= {
        "2607.04223",
        "2609.10873",
        "2602.02634",
        "2511.12828",
    }


def test_scenario_report_8058_inference():
    """SCENARIO-REPORT-8058-INFERENCE: support floors do not move after outcomes."""
    v = m.methods()
    assert v["hypotheses"][2]["comparison"] == "fresh_admission versus reused_guard"
    assert v["safety"]["later_support"] == [80, 10]
    assert v["safety"]["retention_support"] == [48, 8]
    assert v["statistics"]["primary_block"] == 32
    assert v["statistics"]["sensitivity_blocks"] == [16, 64]
    assert v["statistics"]["holm_family"] == ["H1", "H2", "H3"]
    assert v["learning"]["seeds"] == list(range(101, 121))


def test_scenario_report_8058_seal(tmp_path):
    """REQ-REPORT-8058: historical eligibility is metadata, never a current label."""
    plan = m.seal(m.ROOT, tmp_path)
    assert plan["failures"] == []
    assert len(plan["rows"]) == 512
    assert plan["outcome_access_ledger"][-1]["current_outcomes_opened"] == 0
    assert plan["source_valid"] and plan["learning_valid"]
    assert all("y" not in r for r in plan["rows"])
    assert all(r["release_slot"] == r["slot"] + 20 for r in plan["rows"] if r["role"] == "stream")
    atomic_json(tmp_path / "validation.json", {"receipts": [], "coverage": {}})
    value = m.build(plan, tmp_path, [], {}, True, 0.1)
    assert value["source_protocol_ready_score"] == value["learning_protocol_ready_score"] == 0
    assert value["independent_count"] < value["completed_count"]
    assert set(value) - {"field_principles"} <= set(value["field_principles"])


def test_scenario_report_8058_missing_and_mutation(tmp_path):
    """SCENARIO-REPORT-8058-TERMINAL: missing operands retain exact failure fields."""
    plan = m.seal(tmp_path / "missing", tmp_path / "raw")
    assert plan["failures"][0]["field"] == "resource_exists"
    assert plan["failures"][0]["observed"] is False
    value = m.build(plan, tmp_path / "raw", [], {}, False, 0.1)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    plan = m.seal(m.ROOT, tmp_path / "mutated", mutate=True)
    assert any(r["field"] == "source_cluster_overlap" for r in plan["failures"])
    assert not plan["source_valid"] and plan["learning_valid"]


def test_scenario_report_8058_readiness(tmp_path):
    """REQ-REPORT-8058: branch readiness depends on checks, never positive outcomes."""
    plan = m.seal(m.ROOT, tmp_path)
    receipts = [{"name": "required", "passed": True}]
    coverage = {p: {"summary": {"num_statements": 1, "missing_lines": 0}} for p in m.OWNED}
    plan["validation_manifest"] = ["required"]
    value = m.build(plan, tmp_path, receipts, coverage, False, 0.1)
    assert value["verdict_class"] == "null"
    assert value["source_protocol_ready_score"] == value["learning_protocol_ready_score"] == 1
    bad = deepcopy(receipts)
    bad[0]["passed"] = False
    value = m.build(plan, tmp_path, bad, coverage, False, 0.1)
    assert value["verdict_class"] == "disqualified"
    assert value["required_checks_passed"] is False


@pytest.mark.parametrize("route", ["success", "blocked", "mutation"])
def test_scenario_report_8058_cli(tmp_path, route):
    """SCENARIO-REPORT-8058-TERMINAL: real outside-checkout children exit normally."""
    output = tmp_path / route / (m.NAME + ".json")
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI), "--fixture-output", str(output)]
    if route == "blocked":
        cmd += ["--root", str(tmp_path / "absent")]
    if route == "mutation":
        cmd += ["--mutate"]
    config = os.environ.get("CARNOT_8058_COVERAGE_CONFIG")
    if config:
        cmd = [sys.executable, "-m", "coverage", "run", "--rcfile=" + config, *cmd[2:]]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print("8058 CLI before " + route, flush=True)
    started = time.monotonic()
    p = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    print(
        json.dumps(
            dict(
                route=route,
                argv=cmd,
                exit_code=p.returncode,
                duration_s=time.monotonic() - started,
                log_sha256=hashlib.sha256((p.stdout + p.stderr).encode()).hexdigest(),
                log=p.stdout + p.stderr,
            )
        ),
        flush=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("null" if route == "success" else "blocked")
    assert m.replay(output)
    value["eligible_count"] += 1
    atomic_json(output, value)
    assert not m.replay(output)
    assert m.main(["--cold-replay", str(output)]) == 1


def test_scenario_report_8058_production_reduction(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8058-TERMINAL: production reduction honors actual check failures."""
    output = tmp_path / (m.NAME + ".json")
    cov = {p: {"summary": {"num_statements": 1, "missing_lines": 0}} for p in m.OWNED}

    def checks(root, spec, private, durable):
        atomic_json(private / "coverage.json", {"files": cov})
        log = private / "check.log"
        log.write_text("owned check control\n")
        return dict(
            name=spec["name"], passed=True, log_path=str(log), log_sha256=m.sha256_file(log)
        )

    original_config = os.environ.get("CARNOT_8058_COVERAGE_CONFIG")
    monkeypatch.setattr(m, "run_check", checks)
    monkeypatch.setattr(m, "terminal", lambda p: {"passed": True})
    assert m.main(["--output", str(output)]) == 0
    assert m.main(["--output", str(output)]) == 1
    # Validation log disappearance after the private venue closes must be detected.
    assert not m.replay(output)
    monkeypatch.setattr(
        m, "seal", lambda *a, **kw: (_ for _ in ()).throw(ValueError("owned mutation"))
    )
    assert m.main(["--output", str(tmp_path / "bad" / (m.NAME + ".json"))]) == 1
    if original_config:
        monkeypatch.setenv("CARNOT_8058_COVERAGE_CONFIG", original_config)
    else:
        monkeypatch.delenv("CARNOT_8058_COVERAGE_CONFIG", raising=False)


def test_scenario_report_8058_replay_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8058-TERMINAL: readers reject independently edited evidence."""
    output = tmp_path / (m.NAME + ".json")
    monkeypatch.setattr(m, "terminal", lambda p: {"passed": True})
    assert m.main(["--fixture-output", str(output)]) == 0
    assert m.main(["--cold-replay", str(output)]) == 0
    original = json.loads(output.read_text())
    raw = Path(original["terminal_validation_sidecar_path"]).parent
    for key, edit in [
        ("source_artifact_hashes", lambda v: v[0].update(sha256="wrong")),
        ("source_artifact_hashes", lambda v: v[0].update(snapshot_path=str(tmp_path / "absent"))),
        ("code_config_hashes", lambda v: v.update({m.MODULE: "wrong"})),
        ("validation_receipts", lambda v: v.append(dict(log_path=str(output), log_sha256="wrong"))),
    ]:
        value = deepcopy(original)
        edit(value[key])
        atomic_json(output, value)
        assert not m.replay(output)
    plan_path = raw / "seal.json"
    original_plan = json.loads(plan_path.read_text())
    for mutation in ["methods", "slot", "denominator", "remove"]:
        plan = deepcopy(original_plan)
        if mutation == "methods":
            plan["methods"]["learning"]["step"] = 0.02
        elif mutation == "remove":
            plan["rows"].pop()
        else:
            plan["rows"][0][mutation] = -1
        atomic_json(plan_path, plan)
        value = deepcopy(original)
        for ref in value["raw_shard_hashes"]:
            if ref["path"] == str(plan_path):
                ref["sha256"] = m.sha256_file(plan_path)
        atomic_json(output, value)
        assert not m.replay(output)
    atomic_json(plan_path, original_plan)
    atomic_json(output, original)
    assert m.replay(output)
    ref = original["source_artifact_hashes"][0]
    snapshot = Path(ref["snapshot_path"])
    snapshot.write_text("changed snapshot")
    assert not m.replay(output)


def test_scenario_report_8058_missing_bound_operands(tmp_path, monkeypatch):
    """REQ-REPORT-8058: absent sidecars and roles block without invented evidence."""
    # Private copies allow fault injection without rewriting protected history.
    for relative in m.INPUTS:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(m.ROOT / relative)
    parent_values = {
        n: json.loads((m.ROOT / "results" / (n + ".json")).read_text()) for n in m.PARENTS
    }
    for route in ["hash", "terminal", "report", "role", "binding"]:
        pins = {}
        for name, original in parent_values.items():
            value = deepcopy(original)
            primary = tmp_path / "results" / (name + ".json")
            side = tmp_path / (name + "-terminal.json")
            report = tmp_path / "results" / "raw" / name / "report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            value["terminal_validation_sidecar_path"] = str(side)
            if route == "terminal":
                value["terminal_validation_sidecar_path"] = str(tmp_path / "absent")
            if route == "role" and "role_manifests" in value:
                value["role_manifests"]["fit"]["path"] = str(tmp_path / "absent")
            atomic_json(primary, value)
            digest = m.sha256_file(primary)
            pins[name] = "wrong" if route == "hash" else digest
            atomic_json(
                report,
                dict(
                    primary_path=str(primary),
                    primary_sha256="wrong" if route == "binding" else digest,
                    report={"passed": True},
                ),
            )
            atomic_json(
                side,
                dict(
                    primary_path=str(primary),
                    primary_sha256="wrong" if route == "binding" else digest,
                    sidecar_path=str(tmp_path / "absent") if route == "report" else str(report),
                ),
            )
        monkeypatch.setattr(m, "PARENTS", pins)
        plan = m.seal(tmp_path, tmp_path / "evidence" / route)
        assert plan["failures"]
        assert not plan["source_valid"]
