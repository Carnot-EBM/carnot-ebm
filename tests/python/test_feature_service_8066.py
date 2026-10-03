"""REQ-REPORT-8066: private checks bind vectors, complete costs and terminal bytes."""

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import content_addressed_features_8066 as c
from carnot import experiment_8066_v698_content_addressed_feature_service as e


def public():
    """Provide complete public bytes without evaluator targets or learned state."""
    return dict(
        family_id="private",
        source_bytes=b"Water is wet.".hex(),
        answer_bytes=b"Water is wet.".hex(),
    )


def test_cache_integrity_mutations_and_eviction(tmp_path):
    """SCENARIO-REPORT-8066-CACHE: every key operand invalidates old vectors."""
    row = public()
    cache = c.FeatureCache(tmp_path / "cache.sqlite", capacity=2)
    expected = c.features.extract(row)["values"]
    assert cache.get(row)["values"] == expected
    assert cache.events[-1]["status"] == "miss"
    assert cache.get(row)["values"] == expected
    assert cache.events[-1]["status"] == "hit"
    key = cache.key(row)
    for field in ("source_bytes", "answer_bytes"):
        changed = dict(row, **{field: row[field] + "20"})
        assert cache.key(changed) != key
        assert cache.get(changed)["values"] == c.features.extract(changed)["values"]
    assert cache.events[-1]["evicted"] == 1
    cache.close()
    cache = c.FeatureCache(tmp_path / "cache.sqlite", capacity=2)
    cache.get(row)
    cache.db.execute("UPDATE features SET vector='[999]' WHERE key=?", (cache.key(row),))
    cache.db.commit()
    assert cache.get(row)["values"] == expected
    assert cache.events[-1]["status"] == "corrupt_miss"
    for field in ("extractor", "config", "schema", "version"):
        alternate = c.FeatureCache(
            tmp_path / "cache.sqlite", identity=dict(cache.identity, **{field: "changed"})
        )
        assert alternate.key(row) != key
        assert alternate.get(row)["values"] == expected
        alternate.close()
    with pytest.raises(ValueError, match="public_fields"):
        cache.get(dict(row, label=1))
    with pytest.raises(ValueError):
        cache.get(dict(row, source_bytes="zz"))
    with pytest.raises(ValueError, match="cache_capacity"):
        c.FeatureCache(tmp_path / "bad.sqlite", capacity=0)
    assert cache.memory()["resident_bound_bytes"] > 0
    cache.close()


def test_abstention_is_never_cached(tmp_path):
    """SCENARIO-REPORT-8066-CACHE: invalid public extraction fails closed."""
    cache = c.FeatureCache(tmp_path / "cache.sqlite")
    with pytest.raises(ValueError, match="feature_abstention"):
        cache.get(dict(public(), source_bytes=b" ".hex()))
    assert cache.db.execute("SELECT count(*) FROM features").fetchone()[0] == 0
    cache.close()


def test_loaded_transaction_pairs_and_cold_reduction(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8066-TRANSACTIONS: E2E-003/004 use the actual PyO3 binding."""
    data, failures = e.load_inputs(e.ROOT, tmp_path / "inputs")
    assert not failures
    assert all(
        any(w["updates"] for w in data["cases"] if (w["arm"], w["class"]) == group)
        for group in {(w["arm"], w["class"]) for w in data["cases"]}
    )
    from test_guarded_transaction_8053 import data as fixture

    data = fixture.__wrapped__()
    native, receipt = e.old.prior.load_library(data["library_reference"], tmp_path)
    monkeypatch.setitem(e.CONFIG, "warmups", 1)
    monkeypatch.setitem(e.CONFIG, "repetitions", 2)
    measured = e.measure(data, native, tmp_path / "measured")
    assert measured["parity_rows"] and all(r["passed"] for r in measured["parity_rows"])
    assert measured["complete_workload_ratios"] == e.reduce_rows(measured["rows"])
    assert measured["completed_count"] > 0
    assert {r["mode"] for r in measured["rows"]} == set(e.MODES)
    assert measured["feature_service_ready_score"] == 0
    v = e.base([])
    v.update(measured, loaded_library_receipt=receipt, raw_directory=str(tmp_path / "measured"))
    v["raw_shard_hashes"] = [e.reference(tmp_path / "measured" / "observations.json")]
    p = tmp_path / "candidate.json"
    atomic_json(p, v)
    assert e.replay(p)["passed"]
    altered = copy.deepcopy(v)
    altered["complete_workload_ratios"] = []
    atomic_json(p, altered)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(p)
    monkeypatch.setitem(e.CONFIG, "budget_s", -1)
    censored = e.measure(data, native, tmp_path / "censored")
    assert censored["censored_count"] == censored["intended_count"]
    assert not censored["complete_workload_ratios"]
    assert e.load_inputs(tmp_path / "missing", tmp_path / "blocked")[1]


def test_speed_gate_preserves_cold_regression():
    """REQ-REPORT-8066: a warm win cannot hide cold population regression."""
    ratios = [
        dict(mode=mode, arm=arm, lower_95=2.0, ratio=2.0)
        for mode in e.MODES
        for arm in ("python", "native")
    ]
    assert e.speed_gate(ratios)
    ratios[0]["ratio"] = 0.9
    assert not e.speed_gate(ratios)


def test_cli_process_routes_and_crash_restart(tmp_path):
    """SCENARIO-REPORT-8066-TERMINAL/CACHE: outside-checkout real CLIs must exit."""
    import hashlib
    import os
    import subprocess
    import time

    receipts = []
    for action, expected in [
        ("crash", 73),
        ("restart", 0),
        ("crash_pending", 73),
        ("restart", 0),
        ("mutate", 0),
        ("populate", 0),
    ]:
        argv = [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.SCRIPT),
            "--cache-probe",
            action,
            "--cache",
            str(tmp_path / "cache.sqlite"),
        ]
        started = time.monotonic()
        child = subprocess.run(
            argv,
            cwd=tmp_path,
            env=dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu"),
            capture_output=True,
            timeout=30,
        )
        assert child.returncode == expected, child.stderr
        receipts.append(
            dict(
                argv=argv,
                exit_code=child.returncode,
                duration_s=time.monotonic() - started,
                log_hash=hashlib.sha256(child.stdout + child.stderr).hexdigest(),
            )
        )
        if action == "restart":
            assert json.loads(child.stdout.splitlines()[-1])["events"][0]["status"] == "hit"
    atomic_json(tmp_path / "cli-receipts.json", dict(rows=receipts))
    out = tmp_path / (e.NAME + ".json")
    child = subprocess.run(
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.SCRIPT),
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(out.read_text())["verdict_class"] == "blocked"
    child = subprocess.run(
        [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.SCRIPT), "--cold-replay", str(out)],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr


def test_owned_cli_probe_and_terminal_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8066-TERMINAL: unsafe readiness and corrupt reductions fail."""
    import runpy
    import sys

    for action in ("populate", "restart", "mutate"):
        assert e.main(["--cache-probe", action, "--cache", str(tmp_path / "cache.sqlite")]) == 0
    monkeypatch.setattr(e.os, "_exit", lambda code: (_ for _ in ()).throw(RuntimeError(str(code))))
    with pytest.raises(RuntimeError, match="73"):
        e.main(["--cache-probe", "crash", "--cache", str(tmp_path / "cache.sqlite")])
    value = e.base([])
    value.update(feature_service_ready_score=1, required_checks_passed=False)
    p = tmp_path / "unsafe.json"
    atomic_json(p, value)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(p)
    assert e.main(["--cold-replay", str(p)]) == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [e.SCRIPT, "--cache-probe", "populate", "--cache", str(tmp_path / "cache.sqlite")],
    )
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert exited.value.code == 0


def test_main_terminal_dispositions(tmp_path, monkeypatch):
    """REQ-REPORT-8066: terminal credit depends on owned gates, never fixture timing."""
    from test_guarded_transaction_8053 import data as fixture

    data = fixture.__wrapped__()
    data["cases"] = [data["cases"][0]]
    monkeypatch.setitem(e.CONFIG, "warmups", 0)
    monkeypatch.setitem(e.CONFIG, "repetitions", 1)
    source = tmp_path / "fixture.json"
    atomic_json(source, data)
    out = tmp_path / (e.NAME + ".json")
    assert (
        e.main(["--fixture-input", str(source), "--output", str(out), "--validation-worker"]) == 0
    )
    assert e.main(["--cold-replay", str(out)]) == 0
    v = json.loads(out.read_text())
    assert v["verdict_class"] == "circular_positive"
    assert v["feature_service_ready_score"] == 0
    assert (
        e.main(
            [
                "--worker-input",
                str(source),
                "--raw",
                str(tmp_path / "direct-worker"),
                "--output",
                str(tmp_path / "worker.json"),
            ]
        )
        == 0
    )
    assert e.main(["--root", str(tmp_path / "missing"), "--output", str(out)]) == 0
    original = e.run_commands

    def checked_commands(root, commands, **kwargs):
        if commands[0].scope == "measurement":
            return original(root, commands, **kwargs)
        basetemp = Path(
            next(a.split("=", 1)[1] for a in commands[0].argv if a.startswith("--basetemp="))
        )
        atomic_json(basetemp / "private-cli/cli-receipts.json", dict(rows=[]))
        return [dict(name="private_owned_gate", passed=True, exit_code=0, log_path=str(source))]

    monkeypatch.setattr(e, "load_inputs", lambda *args: (data, []))
    monkeypatch.setattr(e, "run_commands", checked_commands)
    monkeypatch.setattr(e, "terminal", lambda path: e.replay(path))
    assert e.main(["--output", str(out)]) == 0
    assert json.loads(out.read_text())["feature_service_ready_score"] == 1
    monkeypatch.setattr(e, "speed_gate", lambda ratios: True)
    assert e.main(["--output", str(out)]) == 0
    assert json.loads(out.read_text())["service_speedup_score"] == 1
    monkeypatch.setitem(e.CONFIG, "budget_s", -1)
    assert e.main(["--output", str(out)]) == 0
    assert json.loads(out.read_text())["verdict_class"] == "disqualified"
    assert any(
        g["field"] == "rows[status=censored].count"
        for g in json.loads(out.read_text())["gate_check_summary"]
    )

    def failed_commands(root, commands, **kwargs):
        if commands[0].scope == "measurement":
            return [dict(name="measurement", passed=False, exit_code=9)]
        return [dict(name="private_owned_gate", passed=False, exit_code=2, log_path=str(source))]

    monkeypatch.setattr(e, "run_commands", failed_commands)
    assert e.main(["--output", str(out)]) == 0
    assert json.loads(out.read_text())["verdict_class"] == "disqualified"
    assert e.main(["--root", str(tmp_path), "--output", str(out)]) == 0


def test_reader_operand_and_component_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8066-TERMINAL: exact failed operands and corrupt spans reject."""
    from test_guarded_transaction_8053 import data as fixture

    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    data, failures = e.load_inputs(e.ROOT, tmp_path / "bad-reader")
    assert any(f["field"] == "report.passed" for f in failures)
    data = fixture.__wrapped__()
    data["cases"] = [data["cases"][0]]
    native, receipt = e.old.prior.load_library(data["library_reference"], tmp_path / "native")
    monkeypatch.setitem(e.CONFIG, "warmups", 0)
    monkeypatch.setitem(e.CONFIG, "repetitions", 1)
    measured = e.measure(data, native, tmp_path / "rows")
    value = e.base([])
    value.update(measured, loaded_library_receipt=receipt, raw_directory=str(tmp_path / "rows"))
    p = tmp_path / "candidate.json"
    raw = tmp_path / "rows/observations.json"
    original = json.loads(raw.read_text())
    altered = copy.deepcopy(original)
    altered["rows"][0]["components"]["feature_gather_ns"] += 1
    atomic_json(raw, altered)
    value["rows"] = altered["rows"]
    atomic_json(p, value)
    with pytest.raises(ValueError, match="component_drift"):
        e.replay(p)
    atomic_json(raw, original)
    value["rows"] = original["rows"]
    atomic_json(p, value)
    monkeypatch.setattr(e.old.m, "restart_decisions", lambda *args: dict(passed=False))
    with pytest.raises(ValueError, match="cold_restart_drift"):
        e.replay(p)


def test_crash_pending_and_failed_mutation_route(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8066-CACHE: uncommitted corruption must survive no restart."""

    def crash(code):
        raise RuntimeError(str(code))

    monkeypatch.setattr(e.os, "_exit", crash)
    with pytest.raises(RuntimeError, match="73"):
        e.main(["--cache-probe", "crash_pending", "--cache", str(tmp_path / "pending.sqlite")])
    original = c.FeatureCache.get

    def broken(self, row):
        result = original(self, row)
        if self.events[-1]["status"] == "corrupt_miss":
            self.events[-1]["status"] = "hit"
        return result

    monkeypatch.setattr(c.FeatureCache, "get", broken)
    with pytest.raises(ValueError, match="mutation_not_rejected"):
        e.main(["--cache-probe", "mutate", "--cache", str(tmp_path / "mutation.sqlite")])


def test_partial_budget_retains_completed_and_censored_units(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8066-TRANSACTIONS: a partial clock budget cannot lose rows."""
    from test_guarded_transaction_8053 import data as fixture

    data = fixture.__wrapped__()
    data["cases"] = [data["cases"][0]]
    native, _ = e.old.prior.load_library(data["library_reference"], tmp_path / "native")
    monkeypatch.setitem(e.CONFIG, "warmups", 0)
    monkeypatch.setitem(e.CONFIG, "repetitions", 1)
    monkeypatch.setitem(e.CONFIG, "budget_s", 1)
    clock = iter([0.0, 0.0])
    monkeypatch.setattr(e, "progress", lambda *args: None)
    monkeypatch.setattr(e.time, "monotonic", lambda: next(clock, 2.0))
    result = e.measure(data, native, tmp_path / "rows")
    assert result["completed_count"] == 4
    assert result["censored_count"] == result["intended_count"] - 4
    assert result["break_even_requests"][0]["requests"] is None
