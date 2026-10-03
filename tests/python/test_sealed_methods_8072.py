"""REQ-REPORT-8072: method custody does not assert scientific benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_8072_v699_sealed_methods as m


def test_methods():
    """SCENARIO-REPORT-8072-METHODS: the registered family has exactly two tests."""
    v = m.methods()
    assert v["statistics"]["holm_family"] == ["H1_source_interactions", "H2_projected_learning"]
    assert v["statistics"]["bootstrap_draws"] == 10000
    assert v["statistics"]["primary_block"] == 32
    assert v["source"]["ridge_grid"] == [0.0001, 0.001, 0.01, 0.1, 1]
    assert len(v["source"]["interactions"]) == 3
    assert v["source"]["inputs"] == 9
    assert v["learning"]["arms"] == ["frozen", "unconditional", "ray_fresh", "projected_fresh"]
    assert v["learning"]["attempt_slots"] == [64, 128, 192]
    assert v["learning"]["admission_rows"] == 12
    assert v["projection"]["max_steps"] == 256
    assert v["projection"]["memory_limit"] == 64
    assert {r["id"] for r in v["method_source_map"]} == set(m.PAPERS)


def test_authenticated_seal(tmp_path):
    """REQ-REPORT-8072: original manifests, masks and head survive byte for byte."""
    plan = m.seal(m.ROOT, tmp_path)
    assert not plan["failures"]
    assert len(plan["rows"]) == 512
    assert set(plan["role_manifests"]) == {"fit", "tune", "evaluation", "stream", "retention"}
    assert all("y" not in r for r in plan["rows"])
    assert all(r["release_slot"] == r["slot"] + 20 for r in plan["rows"] if r["role"] == "stream")
    assert plan["qualified_head_sha256"] == m.canonical_hash(plan["qualified_head"])
    for ref in plan["role_manifests"].values():
        assert m.sha256_file(Path(ref["path"])) == ref["sha256"]
    assert plan["outcome_access_ledger"][-1]["private_evaluator_files_opened"] == 0
    assert len(plan["service_matrix"]["rows"]) == 2520
    assert plan["workload_budget_estimate"]["historical_timing_rows"] > 0


def test_readiness_and_external_failures(tmp_path):
    """SCENARIO-REPORT-8072-TERMINAL: owned failures and external blocks differ."""
    plan = m.seal(m.ROOT, tmp_path / "good")
    receipts = [{"name": "owned", "passed": True, "classification": "required"}]
    coverage = {p: {"summary": {"num_statements": 1, "missing_lines": 0}} for p in m.OWNED}
    plan["validation_manifest"] = ["owned"]
    v = m.build(plan, tmp_path / "good", receipts, coverage, False, 0.1)
    assert v["verdict_class"] == "null"
    assert all(v[b + "_protocol_ready_score"] == 1 for b in m.BRANCHES)
    assert set(v) - {"field_principles"} <= set(v["field_principles"])
    bad = deepcopy(receipts)
    bad[0]["passed"] = False
    assert m.build(plan, tmp_path, bad, coverage, False, 0.1)["verdict_class"] == "disqualified"
    assert m.build(plan, tmp_path, receipts, {}, False, 0.1)["required_checks_passed"] is False
    missing = m.seal(tmp_path / "absent", tmp_path / "missing")
    v = m.build(missing, tmp_path, receipts, coverage, True, 0.1)
    assert v["verdict_class"] == "blocked"
    assert v["gate_check_summary"][0]["observed"] is False
    mutated = m.seal(m.ROOT, tmp_path / "mutated", mutate=True)
    mutated["validation_manifest"] = ["owned"]
    v = m.build(mutated, tmp_path, receipts, coverage, False, 0.1)
    assert v["source_protocol_ready_score"] == 0
    assert v["learning_protocol_ready_score"] == v["service_protocol_ready_score"] == 1


def test_timing_estimate_missing_cells():
    """SCENARIO-REPORT-8072-SERVICE: absent cells cannot become zero-time estimates."""
    matrix = m.service_matrix()
    v = m.estimate([], matrix)
    assert all(r["estimate_s"] is None for r in v["partitions"])
    assert len(matrix["rows"]) == 2520
    rows = [
        {
            "mode": "cold",
            "condition": "feedback_constrained",
            "transaction_class": "accepted",
            "arm": "python_cached",
            "transaction_ns": 100000000000,
            "status": "completed",
        }
    ]
    v = m.estimate(rows, matrix)
    assert v["cells"][0]["sample_count"] >= 0


@pytest.mark.parametrize("route", ["success", "blocked", "mutation"])
def test_private_real_cli(tmp_path, route):
    """SCENARIO-REPORT-8072-TERMINAL: real children work without ambient paths."""
    output = tmp_path / route / (m.NAME + ".json")
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI), "--fixture-output", str(output)]
    if route == "blocked":
        cmd += ["--root", str(tmp_path / "missing")]
    if route == "mutation":
        cmd += ["--mutate"]
    config = os.environ.get("CARNOT_8072_COVERAGE_CONFIG")
    if config:
        cmd = [sys.executable, "-m", "coverage", "run", "--rcfile=" + config, *cmd[2:]]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print("8072 CLI before " + route, flush=True)
    p = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    print("8072 CLI after " + route + " exit=" + str(p.returncode), flush=True)
    assert p.returncode == 0, p.stdout + p.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("null" if route == "success" else "blocked")
    assert m.replay(output)
    assert m.main(["--cold-replay", str(output)]) == 0
    value["eligible_count"] += 1
    m.atomic_json(output, value)
    assert m.main(["--cold-replay", str(output)]) == 1


def test_production_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8072: real receipts control readiness and seals cannot be reused."""
    output = tmp_path / (m.NAME + ".json")
    cov = {p: {"summary": {"num_statements": 1, "missing_lines": 0}} for p in m.OWNED}

    def checks(root, spec, private, durable):
        if spec["name"] == "seal_child_normal_exit":
            plan = m.seal(m.ROOT, output.parent / "raw" / output.stem)
            m.atomic_json(output.parent / "raw" / output.stem / "seal.json", plan)
        m.atomic_json(private / "coverage.json", {"files": cov})
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private receipt control\n")
        return dict(spec, passed=True, log_path=str(log), log_sha256=m.sha256_file(log))

    monkeypatch.setattr(m, "run_check", checks)
    monkeypatch.setattr(m, "terminal", lambda p: {"passed": True})
    monkeypatch.setenv(
        "CARNOT_8072_COVERAGE_CONFIG", os.environ.get("CARNOT_8072_COVERAGE_CONFIG", "")
    )
    assert m.main(["--output", str(output)]) == 0
    assert m.main(["--output", str(output)]) == 1
    assert m.replay(output)
    monkeypatch.setattr(m, "run_check", lambda *a: {"passed": False})
    assert m.main(["--output", str(tmp_path / "bad" / (m.NAME + ".json"))]) == 1


def test_replay_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8072-TERMINAL: independent reduction rejects altered evidence."""
    output = tmp_path / (m.NAME + ".json")
    monkeypatch.setattr(m, "terminal", lambda p: {"passed": True})
    monkeypatch.setenv(
        "CARNOT_8072_COVERAGE_CONFIG", os.environ.get("CARNOT_8072_COVERAGE_CONFIG", "")
    )
    assert m.main(["--fixture-output", str(output)]) == 0
    original = json.loads(output.read_text())
    raw = Path(original["terminal_validation_sidecar_path"]).parent
    for key, edit in [
        ("source_artifact_hashes", lambda v: v[0].update(sha256="wrong")),
        ("code_config_hashes", lambda v: v.update({m.MODULE: "wrong"})),
        ("validation_receipts", lambda v: v.append(dict(log_path=str(output), log_sha256="wrong"))),
    ]:
        value = deepcopy(original)
        edit(value[key])
        m.atomic_json(output, value)
        assert not m.replay(output)
    m.atomic_json(output, original)
    seal_path = raw / "seal.json"
    plan = json.loads(seal_path.read_text())
    for mutation in ("methods", "matrix", "row", "eligibility", "head", "estimate"):
        changed = deepcopy(plan)
        if mutation == "methods":
            changed["methods"]["learning"]["step"] = 1
        elif mutation == "matrix":
            changed["service_matrix"]["rows"].pop()
        elif mutation == "row":
            changed["rows"][0]["slot"] = -1
        elif mutation == "eligibility":
            changed["rows"][0]["eligible"] = not changed["rows"][0]["eligible"]
        elif mutation == "head":
            changed["qualified_head"]["parameters"][0] += 1
        else:
            changed["workload_budget_estimate"]["historical_timing_rows"] += 1
        m.atomic_json(seal_path, changed)
        value = deepcopy(original)
        for ref in value["raw_shard_hashes"]:
            if ref["path"] == str(seal_path):
                ref["sha256"] = m.sha256_file(seal_path)
        m.atomic_json(output, value)
        assert not m.replay(output)
    m.atomic_json(seal_path, plan)
    m.atomic_json(output, original)
    assert m.replay(output)
    output.write_text("invalid JSON")
    assert not m.replay(output)


@pytest.mark.parametrize("route", ["hash", "terminal", "report", "binding", "role"])
def test_missing_authenticated_operands(tmp_path, monkeypatch, route):
    """REQ-REPORT-8072: missing resources preserve the exact failed operand."""
    root = tmp_path / "private-root"
    for relative in m.INPUTS:
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(m.ROOT / relative)
    for number in m.PINS:
        if number == 8058:
            continue
        p = next((m.ROOT / "results").glob(f"experiment_{number}_*.json"))
        (root / "results" / p.name).symlink_to(p)
    original = next((m.ROOT / "results").glob("experiment_8058_*.json"))
    primary = root / "results" / original.name
    value = json.loads(original.read_text())
    terminal = tmp_path / "terminal.json"
    report = primary.parent / "raw" / primary.stem / "report.json"
    value["terminal_validation_sidecar_path"] = (
        str(tmp_path / "absent") if route == "terminal" else str(terminal)
    )
    if route == "role":
        value["role_manifests"]["fit"]["path"] = str(tmp_path / "missing-role")
    m.atomic_json(primary, value)
    digest = m.sha256_file(primary)
    m.atomic_json(
        report,
        dict(
            primary_path=str(primary),
            primary_sha256="wrong" if route == "binding" else digest,
            report={"passed": True},
        ),
    )
    m.atomic_json(
        terminal,
        dict(
            publication=dict(
                primary_path=str(primary),
                primary_sha256=digest,
                sidecar_path=str(tmp_path / "absent-report") if route == "report" else str(report),
            )
        ),
    )
    monkeypatch.setattr(m, "PINS", {**m.PINS, 8058: "wrong" if route == "hash" else digest})
    plan = m.seal(root, tmp_path / "raw")
    assert plan["failures"]
    assert not plan["valid"]["source"]


def test_worker_mode(tmp_path):
    """SCENARIO-REPORT-8072-TERMINAL: the seal worker finishes before publication."""
    output = tmp_path / "worker" / "seal.json"
    assert m.main(["--root", str(tmp_path / "missing"), "--seal-output", str(output)]) == 0
    plan = json.loads(output.read_text())
    assert plan["failures"]
    assert plan["phase_spans"][0]["duration_s"] >= 0


def test_full_budget_estimate():
    """SCENARIO-REPORT-8072-SERVICE: an overrun keeps the complete planned matrix."""
    matrix = m.service_matrix()
    rows = [dict(r, status="completed", transaction_ns=100000000000) for r in matrix["rows"]]
    estimates = m.estimate(rows, matrix)
    assert all(r["estimated_overrun"] for r in estimates["partitions"])
    assert all(r["estimate_s"] == 126000 for r in estimates["partitions"])
    assert len(matrix["rows"]) == 2520
