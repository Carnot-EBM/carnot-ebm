"""REQ-REPORT-8063: pool accounting and evidence limits remain diagnostic."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_8063_v698_admission_opportunity_audit as m
from carnot.reporting.current_work_receipt import atomic_json


def test_pool_units():
    """SCENARIO-REPORT-8063-POOLS: selection does not create artificial misses."""
    candidates = [dict(useful=True, selected=i == 0, harmful=False) for i in range(5)]
    assert m.pool_summary(candidates)["missed"] is False
    assert m.pool_summary(candidates)["useful_count"] == 5
    for r in candidates:
        r["selected"] = False
    assert m.pool_summary(candidates)["missed"] is True
    assert m.pool_summary([])["missed"] is None
    assert m.ratio(0, 0) is None
    assert m.ratio(1, 2) == 0.5


def test_bounds():
    """SCENARIO-REPORT-8063-BOUNDS: no-disagreement evidence has finite uncertainty."""
    v = m.feasibility(12, 0, 0, 1)
    assert v["paired_lower"] < 0 < v["paired_upper"]
    assert not v["paired_margin_feasible"]
    assert not v["brier_margin_feasible"]
    assert v["iid_certificate_valid"] is False
    assert m.feasibility(0, 0, 0, 2)["paired_lower"] is None
    assert m.feasibility(1000, 3, 2, 1)["paired_lower"] < 0.01


def test_readiness(tmp_path):
    """REQ-REPORT-8063: failed owned checks cannot claim readiness."""
    plan = m.empty_plan()
    receipts = [dict(name="unit", passed=True)]
    plan["validation_manifest"] = ["unit"]
    cov = {p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in m.OWNED}
    value = m.build(plan, tmp_path, receipts, cov, False, 0.1)
    assert value["admission_audit_ready_score"] == 1
    assert value["verdict_class"] == "null"
    receipts[0]["passed"] = False
    assert m.build(plan, tmp_path, receipts, cov, False, 0.1)["verdict_class"] == "disqualified"
    plan["failures"] = [dict(check="absent", field="resource_exists", observed=False)]
    assert m.build(plan, tmp_path, [], {}, False, 0.1)["verdict_class"] == "blocked"
    assert set(value) - {"field_principles"} <= set(value["field_principles"])


@pytest.mark.parametrize("route", ["success", "blocked", "mutation"])
def test_cli(tmp_path, route):
    """SCENARIO-REPORT-8063-TERMINAL: actual private CLI exits and cold reductions."""
    output = tmp_path / route / (m.NAME + ".json")
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI), "--fixture-output", str(output)]
    if route == "blocked":
        cmd += ["--root", str(tmp_path / "absent")]
    if route == "mutation":
        cmd += ["--mutate"]
    config = os.environ.get("CARNOT_8063_COVERAGE_CONFIG")
    if config:
        cmd = [sys.executable, "-m", "coverage", "run", "--rcfile=" + config, *cmd[2:]]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print("8063 CLI before " + route, flush=True)
    p = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    print("8063 CLI after " + route + "\n" + p.stdout + p.stderr, flush=True)
    assert p.returncode == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("null" if route == "success" else "blocked")
    assert m.replay(output)
    assert m.main(["--cold-replay", str(output)]) == 0
    mutated = deepcopy(value)
    mutated["missed_opportunity_numerator"] += 1
    atomic_json(output, mutated)
    assert not m.replay(output)
    assert m.main(["--cold-replay", str(output)]) == 1


def test_production_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8063-TERMINAL: owned failures disqualify and seals survive reruns."""
    coverage = {p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in m.OWNED}
    monkeypatch.setattr(m, "collect", lambda *a, **k: m.empty_plan())
    monkeypatch.setattr(m, "terminal", lambda p: dict(passed=True))
    failed = []

    def checks(root, spec, private, durable):
        atomic_json(private / "coverage.json", dict(files=coverage))
        atomic_json(
            Path(spec["argv"][-1]) / "evaluation.json",
            dict(rows=[], candidate_pool_rows=[], harmful_admission_rows=[]),
        ) if spec["name"] == "isolated_evaluator" else None
        log = tmp_path / "stub.log"
        log.write_text("Owned path coverage control only; real CLI runs are separate.\n")
        return dict(
            name=spec["name"],
            passed=spec["name"] not in failed,
            log_path=str(log),
            log_sha256=m.sha256_file(log),
        )

    monkeypatch.setattr(m, "run_check", checks)
    output = tmp_path / "success" / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 0
    assert m.main(["--output", str(output)]) == 1
    assert not m.replay(output)
    failed.append("isolated_evaluator")
    output = tmp_path / "failure" / (m.NAME + ".json")
    assert m.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(m, "collect", lambda *a, **k: (_ for _ in ()).throw(ValueError("control")))
    assert m.main(["--output", str(tmp_path / "error" / (m.NAME + ".json"))]) == 1


@pytest.mark.parametrize("route", ["sidecar", "report", "stale", "flagged"])
def test_missing_terminal_operands(tmp_path, monkeypatch, route):
    """REQ-REPORT-8063: absent or changed validators keep exact blocked operands."""
    monkeypatch.setattr(m.prior, "INPUTS", [])
    for label in [m.MODULE, m.CLI, "python/carnot/verify/feedback_constrained_8051.py"]:
        path = tmp_path / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    primary = tmp_path / "results" / "experiment_8051_fixture.json"
    sidecar = tmp_path / "sidecar.json"
    report = tmp_path / "report.json"
    atomic_json(
        primary,
        dict(terminal_validation_sidecar_path=str(sidecar), flagged_adversarial=route == "flagged"),
    )
    digest = m.sha256_file(primary)
    atomic_json(sidecar, dict(publication=dict(sidecar_path=str(report), primary_sha256=digest)))
    report = primary.parent / "raw" / primary.stem / "validators" / "report.json"
    atomic_json(
        report,
        dict(primary_sha256="changed" if route == "stale" else digest, report=dict(passed=True)),
    )
    atomic_json(sidecar, dict(publication=dict(sidecar_path=str(report), primary_sha256=digest)))
    if route == "sidecar":
        sidecar.unlink()
    if route == "report":
        report.unlink()
    monkeypatch.setattr(m, "PARENTS", {primary.stem: digest})
    plan = m.collect(tmp_path, tmp_path / "raw")
    assert plan["failures"]
    assert plan["failures"][0]["path"] in {str(sidecar), str(report)}


def test_private_evaluator(tmp_path):
    """SCENARIO-REPORT-8063-POOLS: raw probabilities cold-reduce without a learner."""
    target = tmp_path / "target.json"
    atomic_json(target, dict(rows=[dict(family_id="u", eligible_y=0)]))
    ref = dict(path=str(target), sha256=m.sha256_file(target))
    candidate = dict(
        pool="p",
        coefficients=[0.0],
        incumbent=[0.0],
        arm="feedback_constrained",
        seed=1,
        slot=0,
        opportunity="gradient/0",
        alpha=1,
        selected=True,
        guard_admissible=True,
    )
    public = [
        dict(role=role, slot=1, unit="u", source="s", vector=[1.0], in_later_mask=True)
        for role in ["later", "retention"]
    ]
    atomic_json(
        tmp_path / "committed_candidates.json",
        dict(
            head=dict(calibration=[0.0, 1.0], parameters=[0.0], decay_scale=1.0),
            candidates=[candidate],
            public=public,
            stream_target=ref,
            retention_target=ref,
        ),
    )
    assert m.main(["--evaluate", str(tmp_path)]) == 0
    value = m.evaluate(tmp_path, cold=True)
    assert value["candidate_pool_rows"][0]["denominator"] == 0
    assert value["harmful_admission_rows"] == []
    assert m.feasibility(12, 12, 0, 1)["paired_upper"] == 1
