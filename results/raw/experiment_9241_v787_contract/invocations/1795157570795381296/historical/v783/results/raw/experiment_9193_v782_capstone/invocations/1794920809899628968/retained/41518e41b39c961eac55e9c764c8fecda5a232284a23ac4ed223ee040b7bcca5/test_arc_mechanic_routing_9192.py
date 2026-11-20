"""REQ-REPORT-9192-V782 / SCENARIO-REPORT-9192-V782: preserve both consumer contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.agentic import arc_solver_kit as kit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "openspec/change-proposals/experiment-9192-inputs.json").read_bytes())


def test_specialists_and_collisions():
    """Original specialists must lead despite colliding names; dispatch stays reachable."""
    for mechanic, prefix in CONFIG["compatibility_prefixes"].items():
        reference = kit.select_primitive_operators(mechanic_class=mechanic)
        assert [o.operator for o in reference][: len(prefix)] == prefix
        assert "approach_dispatcher_operator" in [o.operator for o in reference]
        for game in ["", "sc25", "sb26", "tr87", "re86", "unseen_9192"]:
            assert kit.select_primitive_operators(mechanic_class=mechanic, game=game) == reference


def test_matrix_and_artifact(tmp_path, monkeypatch):
    """Readiness requires affected receipts; missing inputs become censored blocked rows."""
    from carnot.reporting import arc_mechanic_routing_9192 as m

    work = m.measure(ROOT, tmp_path / "raw", "20261117")
    assert len(work["rows"]) == 124 and all(r["passed"] for r in work["rows"])
    assert work["consumer_reachability"]["recommendation_selector_observed"]
    assert work["consumer_reachability"]["dispatcher_reached"]
    value = m.build(work, {})
    assert value["experiment_id"] == 9192 and value["mechanic_routing_ready_score"] == 0
    monkeypatch.setattr(m, "receipts_pass", lambda *_: True)
    value = m.build(work, {})
    assert value["mechanic_routing_ready_score"] == 1
    out = tmp_path / (m.NAME + ".json")
    m.atomic_json(out, value)
    assert m.replay(out)["passed"]
    value["rows"][0]["after"] = ["invented"]
    value["reproducibility_checksum"] = m.checksum(value)
    m.atomic_json(out, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        m.replay(out)
    work = m.measure(tmp_path / "absent", tmp_path / "blocked", "20261117")
    value = m.build(work, {})
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 124


def child(tmp_path, *args):
    """A foreign working directory and absent PYTHONPATH exercise the shipped CLI imports."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/experiments/experiment_9192_v782_arc_mechanic_routing.py"),
        *map(str, args),
    ]
    if env.get("COVERAGE_RCFILE"):
        cmd[1:2] = ["-m", "coverage", "run"]
    print("[9192] child_before", args, flush=True)
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    print("[9192] child_after", result.returncode, flush=True)
    return result


def test_private_cli(tmp_path):
    """E2E-018/021: actual success, refusal, tamper and replay retain terminal evidence."""
    from carnot.reporting import arc_mechanic_routing_9192 as m

    out = tmp_path / (m.NAME + ".json")
    result = child(tmp_path, "--output", out, "--checks-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", out).returncode == 0
    value = json.loads(out.read_bytes())
    value["completed_count"] += 1
    value["reproducibility_checksum"] = m.checksum(value)
    m.atomic_json(out, value)
    assert child(tmp_path, "--cold-replay", out).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 1
    assert child(tmp_path, "--checks-only").returncode == 1
    assert (
        child(tmp_path, "--root", tmp_path / "absent", "--output", out, "--checks-only").returncode
        == 0
    )
    assert json.loads(out.read_bytes())["verdict_class"] == "blocked"


def test_main_and_refusals(tmp_path, monkeypatch):
    """Normal execution must qualify; drift in either sources or rows cannot replay."""
    from carnot.reporting import arc_mechanic_routing_9192 as m

    monkeypatch.setattr(m, "qualify", lambda _: {})
    monkeypatch.setattr(m, "validate", lambda _: {"passed": True})
    out = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(out)]) == 0
    value = json.loads(out.read_bytes())
    value["code_config_hashes"][0]["sha256"] = "forged"
    m.atomic_json(out, value)
    with pytest.raises(ValueError, match="operand_hash"):
        m.replay(out)
    work = m.measure(ROOT, tmp_path / "work", "20261117")
    value = m.build(work, {})
    m.atomic_json(out, value)
    monkeypatch.setattr(m, "inspect", lambda _: {})
    with pytest.raises(ValueError, match="source_drift"):
        m.replay(out)
    monkeypatch.undo()
    monkeypatch.setattr(m, "routing_rows", lambda _: [])
    with pytest.raises(ValueError, match="routing_drift"):
        m.replay(out)


def test_owned_selector_statements():
    """Only new selector statements enter coverage, with zero exclusions."""
    import ast
    from coverage import Coverage
    from carnot.reporting.current_work_receipt import atomic_json

    tracer = Coverage.current()
    if tracer is None:
        return
    tree = ast.parse(Path(kit.__file__).read_text())
    selector = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "select_primitive_operators"
    )
    _, statements, excluded, missing, _ = tracer.analysis2(kit.__file__)
    nodes = [
        n
        for n in ast.walk(selector)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in {"recognized_mechanic", "specialists", "priority"}
            for t in n.targets
        )
    ]
    owned = {line for n in nodes for line in statements if n.lineno <= line <= n.end_lineno}
    guard = next(
        n
        for n in selector.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "recognized_mechanic"
    )
    owned.update(line for line in statements if guard.lineno <= line <= guard.end_lineno)
    assert owned and not owned.intersection(missing) and not owned.intersection(excluded)
    if os.environ.get("COVERAGE_FILE"):
        atomic_json(
            Path(os.environ["COVERAGE_FILE"]).parent
            / "selector_9192_changed_statement_coverage.json",
            dict(
                path=kit.__file__,
                owned_statements=sorted(owned),
                missing=[],
                excluded=[],
                percent=100,
            ),
        )


def test_qualification_and_consumer_tamper(tmp_path, monkeypatch):
    """Qualifying children stay bounded; replay checks measured consumer evidence itself."""
    from carnot.reporting import arc_mechanic_routing_9192 as m

    work = m.measure(ROOT, tmp_path / "raw", "20261117")
    calls = []
    monkeypatch.setattr(m, "child", lambda name, argv, logs, **kw: calls.append((name, kw)) or {})
    monkeypatch.setattr(m, "measured", lambda *_: {"complete": False})
    validation = m.qualify(work)
    assert len(calls) == 7 and len(validation["checks"]) == 6
    assert all(0 < kw["deadline"] <= 600 for _, kw in calls)
    value = m.build(work, {})
    out = tmp_path / (m.NAME + ".json")
    m.atomic_json(out, value)
    monkeypatch.setattr(m, "consumers", lambda: {})
    with pytest.raises(ValueError, match="consumer_drift"):
        m.replay(out)
    work["consumer_reachability"] = {}
    assert m.build(work, {})["verdict_class"] == "disqualified"
