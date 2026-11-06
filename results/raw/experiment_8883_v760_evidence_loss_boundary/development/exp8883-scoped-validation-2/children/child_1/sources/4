"""REQ-REPORT-8883: visible sufficiency cannot certify omitted source evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import evidence_loss_boundary_8883 as m
from carnot.reporting import evidence_loss_runner_8883 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def frozen():
    """Original fitted heads and rules are inputs, never recomputed from challenge labels."""
    value = json.loads((m.ROOT / "results" / (r.SOURCE + ".json")).read_bytes())
    fitted = r.load(value["fit_manifest"])
    rules = r.load(
        r.reference(Path(value["population_manifest"]["path"]).parent / "certificates.json")
    )
    return fitted, m.rules(rules)


def test_source_view_contract():
    """SCENARIO-REPORT-8883-CONTRACT: identical views can hide opposite source sufficiency."""
    worlds = m.freeze(set(), count=1)
    assert worlds == m.freeze(set(), count=1)
    assert len(worlds) == 4
    assert {w["group"] for w in worlds} == set(m.base.GROUPS)
    for w in worlds:
        pairs = w["pairs"]
        assert [p["view"] for p in pairs] == m.VIEWS
        assert pairs[0]["input"] == pairs[-1]["input"]
        assert pairs[0]["source_sha256"] != pairs[-1]["source_sha256"]
        assert all(m.base.group(p["input"]["query"]) == w["group"] for p in pairs)
        assert [m.truth(p["input"]) for p in pairs] == [
            "support",
            "insufficient",
            "support",
            "insufficient",
            "support",
        ]
        assert [m.truth(p["source"]) for p in pairs] == ["support"] * 4 + ["insufficient"]


def test_scoring_and_controls():
    """REQ-REPORT-8883: only visible records reach serving, and oracle errors stay separate."""
    fitted, rules = frozen()
    worlds = m.freeze(set(), count=1)
    result = m.evaluate(worlds, fitted, rules)
    assert len(result["rows"]) == 4 * 5 * 5 * 3
    assert result["controls"]["passed"]
    exact = [x for x in result["rows"] if x["arm"] == "exact_view"]
    assert not any(x["view_semantic_error"] for x in exact)
    assert all(x["wrong_source_acceptance"] for x in exact if x["view"] == "omit_conflict")
    summary = m.summarize(result["rows"])
    assert len(summary["paired_intervals"]) == 15 * 4
    assert all(i["independent_count"] == 4 for i in summary["paired_intervals"])
    assert m.controls(result["rows"][:-1])["passed"] is False
    broken = deepcopy(result["rows"])
    next(x for x in broken if x["arm"] == "exact_view" and x["view"] == "missing_hop")["action"] = (
        "accept"
    )
    assert not m.controls(broken)["passed"]
    assert m.summarize([])["paired_intervals"] == []
    assert len(m.scheduled(worlds, "missing", "censored")) == len(result["rows"])
    assert m.rules({"choices": []}) == []


def test_external_gate_diagnostics(tmp_path):
    """REQ-REPORT-8883: absent, missing, observed-zero and foreign producers differ."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name, (_, field) = next(iter(r.UPSTREAM.items()))
    for value, check in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (name + ".json"), value)
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    assert r.preconditions(tmp_path / "original")["external_ready"]


def cli(tmp_path, *args):
    """Fresh processes must resolve the real package without checkout environment hints."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )


def test_cli_blocked_replay_and_arguments(tmp_path):
    """SCENARIO-REPORT-8883-CONTRACT: blocked evidence replays but cannot gain readiness."""
    path = tmp_path / (m.NAME + ".json")
    run = cli(
        tmp_path, "--private-output", "--output", str(path), "--sources", str(tmp_path / "missing")
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(path.read_bytes())["verdict_class"] == "blocked"
    assert cli(tmp_path, "--cold-replay", str(path)).returncode == 0
    value = json.loads(path.read_bytes())
    value["boundary_rows_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert cli(tmp_path, "--cold-replay", str(path)).returncode == 1
    assert cli(tmp_path, "--private-output").returncode == 2
    assert cli(tmp_path, "--private-world-limit", "0").returncode == 2
    assert cli(tmp_path, "--date", "20261107").returncode == 2


def test_owned_failures_and_visible_only_serving(tmp_path, monkeypatch):
    """REQ-REPORT-8883: changed rules, exhausted budgets and solver errors disqualify."""
    fitted, choices = frozen()
    worlds = m.freeze(set(), count=1)
    with pytest.raises(ValueError, match="frozen_policy_roster"):
        m.evaluate(worlds, fitted, [])
    with pytest.raises(ValueError, match="owned_exact_budget_exhausted"):
        m.evaluate(worlds, fitted, choices, lambda p, **kw: dict(budget_exhausted=True))

    def wrong(payload, **kw):
        assert set(payload) == {"facts", "query"}
        return dict(budget_exhausted=False, decision="refute")

    with pytest.raises(ValueError, match="owned_exact_oracle_disagreement"):
        m.evaluate(worlds, fitted, choices, wrong)
    choices = deepcopy(choices)
    choices[0]["cutoffs"] = dict.fromkeys(m.base.GROUPS)
    assert all(
        x["action"] == "escalate"
        for x in m.evaluate(worlds, fitted, choices)["rows"]
        if x["seed"] == 11 and x["arm"] == "learned"
    )
    monkeypatch.setattr(r, "DESIGN", "missing-8883-design.json")
    assert (
        r.preconditions(tmp_path / "missing-design")["diagnostics"][-1]["check"] == "missing_file"
    )
    atomic_json(tmp_path / "changed-design.json", dict(changed=True))
    monkeypatch.setattr(r, "DESIGN", str(tmp_path / "changed-design.json"))
    assert (
        r.preconditions(tmp_path / "changed-design")["diagnostics"][-1]["check"]
        == "rejected_producer"
    )


def test_worker_and_installed_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8883-CONTRACT: installed requests cannot carry hidden source labels."""
    import runpy
    import shutil
    from scripts.experiments import evidence_loss_worker_8883 as worker

    pairs = [p for w in m.freeze(set(), count=1) for p in w["pairs"]]
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    atomic_json(request, dict(payloads=[p["input"] for p in pairs] * 2))
    worker.main([str(request), str(response)])
    monkeypatch.setattr(sys, "argv", [str(m.ROOT / m.OWNED[2]), str(request), str(response)])
    runpy.run_path(str(m.ROOT / m.OWNED[2]), run_name="__main__")
    assert len(json.loads(response.read_bytes())["rows"]) == 40
    value = json.loads(
        (m.ROOT / "results/experiment_8871_v759_size_aware_verifier.json").read_bytes()
    )
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="owned_installed_child_failure"):
        r.installed(tmp_path / "failed", pairs, value["package_hash"])

    def foreign(name, argv, raw, **kw):
        if name == "exact_views":
            shutil.copyfile(response, argv[-1])
        return dict(passed=True)

    monkeypatch.setattr(r, "stream", foreign)
    with pytest.raises(ValueError, match="owned_installed_import_drift"):
        r.installed(tmp_path / "foreign", pairs, value["package_hash"])


def test_prepare_measurement_and_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8883: real installed semantics replay from frozen operands and reject changes."""
    raw = tmp_path / "raw"
    work = r.prepare(raw, m.ROOT / "results", 1)
    assert not work["internal_errors"]
    assert work["controls"]["passed"]
    validation = r.unqualified()
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    wr, vr = r.reference(raw / "work.json"), r.reference(raw / "validation.json")
    path = tmp_path / (m.NAME + ".json")
    value = r.artifact(work, validation, "20261106", wr, vr)
    atomic_json(path, value)
    assert value["verdict_class"] == "disqualified"
    assert r.replay(path)
    changed = deepcopy(value)
    changed["rows"][0]["wrong_source_acceptance"] = 99
    atomic_json(path, changed)
    assert not r.replay(path)
    atomic_json(path, value)
    original = r.measurement

    def drift(*a, **kw):
        result = original(*a, **kw)
        result["controls"] = {}
        return result

    monkeypatch.setattr(r, "measurement", drift)
    assert not r.replay(path)
    monkeypatch.setattr(r, "measurement", original)
    value["code_config_hashes"][0]["sha256"] = "sha256:wrong"
    atomic_json(path, value)
    assert not r.replay(path)
    assert not r.replay(tmp_path / "absent.json")

    def failure(*a, **kw):
        raise ValueError("owned_control_failed")

    monkeypatch.setattr(r, "measurement", failure)
    monkeypatch.setattr(r, "preconditions", lambda *a: work["preconditions_checked"])
    failed = r.prepare(tmp_path / "failed", m.ROOT / "results", 1)
    assert failed["internal_errors"] == ["owned_control_failed"]
    assert {x["status"] for x in failed["rows"]} == {"failed"}


def test_replay_rejects_rehashed_work_and_validation(tmp_path, monkeypatch):
    """REQ-REPORT-8883: rehashed source transformations and unsupported coverage cannot pass."""
    from types import SimpleNamespace

    raw = tmp_path / "blocked"
    work = r.prepare(raw, tmp_path / "absent", 1)
    validation = r.unqualified()
    path = tmp_path / (m.NAME + ".json")

    def bind(changed, checks):
        atomic_json(raw / "work.json", changed)
        atomic_json(raw / "validation.json", checks)
        value = r.artifact(
            changed,
            checks,
            "20261106",
            r.reference(raw / "work.json"),
            r.reference(raw / "validation.json"),
        )
        atomic_json(path, value)

    changed = deepcopy(work)
    changed["protocol"]["seed"] = 0
    bind(changed, validation)
    assert not r.replay(path)
    bind(work, validation)
    with monkeypatch.context() as patch:
        patch.setattr(r, "validate_current_work_receipt", lambda *a: ["changed"])
        assert not r.replay(path)
    checks = deepcopy(validation)
    checks["union"]["passed"] = True
    checks["coverage"]["complete"] = True
    checks["receipts"] = []
    bind(work, checks)
    assert not r.replay(path)
    with monkeypatch.context() as patch:
        patch.setattr(
            r, "namespace", lambda *a: SimpleNamespace(validation_evidence=lambda *a: True)
        )
        assert r.replay(path)
    assert not m.controls([dict(work["rows"][0], view="foreign")])["passed"]
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_validation_terminal_and_publication_wiring(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8883-CONTRACT: owned validation failure cannot publish a ready claim."""
    from types import SimpleNamespace

    helper = SimpleNamespace(
        validate=lambda *a: dict(marker="validation"), terminal=lambda *a: dict(passed=True)
    )
    with monkeypatch.context() as patch:
        patch.setattr(r, "namespace", lambda *a: helper)
        assert r.validate(tmp_path, tmp_path) == dict(marker="validation")
        assert r.terminal(tmp_path, tmp_path) == dict(passed=True)
    original_prepare = r.prepare
    work = original_prepare(tmp_path / "work", tmp_path / "missing", 1)
    monkeypatch.setattr(r, "prepare", lambda *a: deepcopy(work))
    monkeypatch.setattr(r, "validate", lambda *a: r.unqualified())
    monkeypatch.setattr(r, "terminal", lambda *a: dict(passed=True))
    health = tmp_path / "health/receipt.json"
    atomic_json(health, dict(status="failed", scope="unrelated_global_health"))
    monkeypatch.setattr(r, "HEALTH", health)
    path = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(path)]) == 0
    value = json.loads(path.read_bytes())
    assert not value["required_checks_passed"] and value["boundary_rows_ready_score"] == 0
    assert value["repository_health"]["status"] == "failed"


def test_bounded_collection_allowance(tmp_path, monkeypatch):
    """REQ-REPORT-8883: retain checkout guard startup while bounding collection at180 seconds."""
    monkeypatch.setattr(r, "stream", lambda *a, **kw: kw)
    assert r.validation_stream("collect_once", [], tmp_path, deadline=60)["deadline"] == 180
    assert r.validation_stream("unit", [], tmp_path, deadline=600)["deadline"] == 600


def test_full_roster_and_runtime_gates(tmp_path, monkeypatch):
    """REQ-REPORT-8883: complete paired measurements establish only the constructed boundary."""
    fitted, choices = frozen()
    worlds = m.freeze(set())
    result = m.evaluate(worlds, fitted, choices)
    work = dict(
        result,
        worlds=worlds,
        private_world_limit=None,
        preconditions_checked=dict(
            external_ready=True, resources=dict(passed=True), references=[], diagnostics=[]
        ),
        current_work_receipt=dict(
            inference_substrate="no_model_load",
            inference_substrate_class="no_model_load",
            invocation_counts={},
            duration_s=1,
            phase_spans=[],
        ),
        internal_errors=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        protocol=m.protocol(),
        producer_invocation={},
        installation_receipts=[],
    )
    checks = r.unqualified()
    checks["union"]["passed"] = checks["coverage"]["complete"] = True
    checks["receipts"] = []
    ref = dict(path=str(tmp_path / "unit.json"), sha256="unit-test-only")
    value = r.artifact(work, checks, "20261106", ref, ref)
    assert value["verdict_class"] == "circular_positive"
    assert (
        value["boundary_rows_ready_score"] == 1 and value["source_fidelity_established_score"] == 0
    )
    assert value["independent_count"] == 120 and value["completed_count"] == 9000
    work["worlds"][-1] = work["worlds"][0]
    assert r.artifact(work, checks, "20261106", ref, ref)["boundary_rows_ready_score"] == 0
    raw = tmp_path / "runtime"
    pre = r.preconditions(raw / "authorities")
    assert pre["external_ready"]
    pairs = [p for w in worlds for p in w["pairs"]]
    monkeypatch.setattr(
        r,
        "installed",
        lambda *a: dict(
            rows=[dict(decision=m.truth(p["input"]), budget_exhausted=False) for p in pairs],
            receipts=[],
            package={},
        ),
    )
    clock = iter([0.0, 10000.0, 10001.0])
    monkeypatch.setattr(r.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="owned_runtime_budget"):
        r.measurement(raw, worlds)
