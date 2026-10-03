"""REQ-REPORT-8071: private token custody and lifetime diagnostic controls."""

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_8071_v699_process_isolation_diagnostic as m
from carnot.reporting.current_work_receipt import atomic_json


def panel():
    """Eight distinct groups make repeated calls distinct from source sample size."""
    return [
        dict(
            family_id=f"f{i}",
            source_cluster_id=f"s{i}",
            role="fit",
            slot=i,
            source_bytes="73",
            answer_bytes="61",
            eligible=True,
            target_tokens=[2, 0],
            response_token_offsets=[[0, 1], [1, 2]],
            views={
                k: dict(tokens=[1, 2, 0], response_start=1, prompt_bytes="70")
                for k in ["full", "no_source"]
            },
        )
        for i in range(8)
    ]


def scorer(view, raw):
    """Owned vocabulary rows let replay detect changes after the score is returned."""
    logits = np.array([[0.0, math.log(2.0), math.log(3.0)], [math.log(3.0), 0.0, math.log(2.0)]])
    return dict(
        m.own_logits(logits, view, raw),
        process_id=42,
        context_identity=str(raw),
        context_closed=True,
        model_build_hashes={"native": "fixture"},
        load_seconds=0,
        forward_seconds=0.01,
        cleanup_seconds=0,
        offload_evidence={},
    )


def test_schedule_and_owned_reduction(tmp_path):
    """SCENARIO-REPORT-8071-SCHEDULE: separated sweeps retain all64 slots."""
    p = panel()
    slots = m.schedule(p)
    assert len(slots) == 64
    assert [r["lifetime_arm"] for r in slots[:4]] == ["current", "fresh_process"] * 2
    assert slots[0]["repeat"] == "A" and slots[32]["repeat"] == "B"
    rows = m.capture(p, scorer, scorer, tmp_path, deadline=math.inf)
    result = m.reduce(p, rows)
    assert result["arm_passed"] == {"current": True, "fresh_process": True}
    assert result["scored_tokens"] == 128
    assert len(result["duplicate_drift_rows"]) == 32
    assert not result["current_failure_reproduced"]
    assert len(list(tmp_path.glob("pass-*/slot.json"))) == 64
    bad = deepcopy(rows)
    bad[0]["token_rows"][0]["logit_position"] += 1
    with pytest.raises(ValueError, match="alignment"):
        m.reduce(p, bad)
    bad = deepcopy(rows)
    bad[0]["token_rows"][0]["log_probability"] += 0.01
    with pytest.raises(ValueError, match="probability"):
        m.reduce(p, bad)
    logits_path = Path(rows[0]["logits_reference"]["path"])
    np.save(logits_path, np.ones((2, 3)))
    with pytest.raises(ValueError, match="hash"):
        m.reduce(p, rows)
    with pytest.raises(ValueError, match="roster"):
        m.reduce(p, rows[:-1])
    with pytest.raises(ValueError):
        m.schedule(p[:-1])


def test_failure_budget_and_drift(tmp_path):
    """SCENARIO-REPORT-8071-CUSTODY: both failing methods can be a complete diagnosis."""
    p = panel()
    calls = 0

    def drift(view, raw):
        nonlocal calls
        calls += 1
        matrix = np.ones((2, 3))
        if calls > 16:
            matrix[0, 2] += 0.01
        return dict(
            m.own_logits(matrix, view, raw),
            process_id=1,
            context_identity=str(raw),
            context_closed=True,
            model_build_hashes={"native": "fixture"},
        )

    rows = m.capture(p, drift, drift, tmp_path / "drift", deadline=math.inf)
    value = m.reduce(p, rows)
    assert value["current_failure_reproduced"]
    assert value["first_divergent_token_rows"]

    def failed(view, raw):
        raise RuntimeError("real child failed")

    rows = m.capture(p, failed, scorer, tmp_path / "failed", deadline=math.inf)
    assert sum(r["status"] == "failed" for r in rows) == 1
    assert sum(r["status"] == "completed" for r in rows) == 32
    assert not m.reduce(p, rows)["arm_passed"]["current"]
    rows = m.capture(p, scorer, scorer, tmp_path / "expired", deadline=0)
    assert all(r["status"] == "censored" for r in rows)
    atomic_json(tmp_path / "panel.json", {"rows": p})


def test_private_cli_success_blocked_failure_replay(tmp_path):
    """SCENARIO-REPORT-8071-TERMINAL: real children exit before private publication."""
    import os
    import subprocess

    fixture = tmp_path / "input.json"
    atomic_json(fixture, {"rows": panel()})
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    for mode, cls in [
        ("success", "circular_positive"),
        ("blocked", "blocked"),
        ("failure", "disqualified"),
    ]:
        out = tmp_path / mode / (m.NAME + ".json")
        cmd = [
            str(m.ROOT / ".venv/bin/python"),
            "-u",
            str(m.CLI),
            "--date",
            "20261003",
            "--fixture-input",
            str(fixture),
            "--fixture-mode",
            mode,
            "--output",
            str(out),
        ]
        p = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, timeout=60)
        assert p.returncode == 0, p.stderr.decode()
        v = json.loads(out.read_text())
        assert v["verdict_class"] == cls
        assert v["process_isolation_ready_score"] == 0
        assert m.replay(out)
        check = subprocess.run(
            cmd[:3] + ["--cold-replay", str(out)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert check.returncode == 0, check.stderr.decode()
        v["completed_count"] = 999
        atomic_json(out, v)
        assert not m.replay(out)
        check = subprocess.run(
            cmd[:3] + ["--cold-replay", str(out)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert check.returncode == 1


def test_preconditions_missing_and_owned_controls(tmp_path):
    """REQ-REPORT-8071: absence is an external blocked operand, never a measured zero."""
    plan = m.preconditions(tmp_path, tmp_path / "raw")
    assert plan["failures"]
    assert all(
        set(["check", "upstream", "path", "hash", "field", "op", "expected", "observed"]) <= set(r)
        for r in plan["failures"]
    )
    assert m.controls(tmp_path / "controls")["passed"]
    with pytest.raises(ValueError):
        m.own_logits(np.zeros((1, 3)), panel()[0]["views"]["full"], tmp_path)
    with pytest.raises(ValueError):
        m.main(
            [
                "--fixture-input",
                str(tmp_path / "absent"),
                "--output",
                str(m.ROOT / "results" / (m.NAME + ".json")),
            ]
        )
    assert not m.replay(tmp_path / "absent")


def test_native_score_and_worker(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8071-CUSTODY: native buffers are copied before context closure."""
    import types

    class Controller:
        def __init__(self, runtime, deadline):
            pass

        def score(self, view, condition):
            result = m.isolation.likelihood(
                np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
                view["tokens"],
                view["response_start"],
            )
            return dict(
                result, context_identity="fresh-1", context_closed=True, native_context_address=123
            )

    monkeypatch.setattr(m.isolation, "NativeController", Controller)
    monkeypatch.setattr(m, "EmbeddedRuntime", lambda model: types.SimpleNamespace(model=model))
    monkeypatch.setattr(m, "build_identity", lambda: {"native": "fixture"})
    monkeypatch.setattr(m, "gpu_observation", lambda: {"devices": [], "fixture": True})
    score = m.native_score(object(), panel()[0]["views"]["full"], tmp_path / "score", math.inf)
    assert score["process_id"] > 0 and score["context_closed"]
    model = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        m, "load_model", lambda plan, raw, deadline: (model, {"load_seconds": 0.01})
    )
    monkeypatch.setattr(m, "gpu_observation", lambda: {"memory": 123})
    plan = dict(panel=panel(), view=panel()[0]["views"]["full"], deadline=time_deadline())
    atomic_json(tmp_path / "plan.json", plan)
    m.worker(tmp_path / "plan.json", tmp_path / "worker.json")
    assert json.loads((tmp_path / "worker.json").read_text())["passed"]
    monkeypatch.setattr(m, "load_model", lambda *a: (_ for _ in ()).throw(RuntimeError("load")))
    m.worker(tmp_path / "plan.json", tmp_path / "failed.json")
    assert not json.loads((tmp_path / "failed.json").read_text())["passed"]


def time_deadline():
    import time

    return time.monotonic() + 120


def test_validation_manifest_and_model_environment(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8071-TERMINAL: checks are frozen before model measurement."""
    specs = m.manifest(tmp_path)
    assert {s["name"] for s in specs} >= {
        "unit_coverage",
        "coverage_report",
        "ruff_check",
        "ruff_format",
        "mypy_strict",
        "scoped_spec",
        "e2e_015",
        "e2e_016_fixture",
        "e2e_016_replay",
        "repository_full_suite",
    }
    assert (
        next(s for s in specs if s["name"] == "repository_full_suite")["classification"]
        == "diagnostic"
    )
    assert all(s["deadline_s"] > 0 for s in specs)
    import sys

    good = m.execute([sys.executable, "-c", "print('done')"], tmp_path / "good", 10)
    assert good["normal_exit"] and good["passed"]
    bad = m.execute(
        [sys.executable, "-c", "import time; time.sleep(10)"], tmp_path / "timeout", 0.05
    )
    assert bad["timed_out"] and not bad["normal_exit"]


@pytest.mark.parametrize("fault", [None, "stat", "build", "tokens", "template", "allocation"])
def test_loader_custody_checks(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8071-CUSTODY: every load checks the same build and token views."""
    import types
    import llama_cpp
    from carnot import experiment_8059_v698_fit_source_scoring as old

    path = tmp_path / "Q4_K_M.gguf"
    path.write_bytes(b"test")
    st = path.stat()
    closed = []
    model = types.SimpleNamespace(
        metadata={"tokenizer.chat_template": "template"}, close=lambda: closed.append(True)
    )
    plan = dict(
        model_path=str(path),
        model_stat=[st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns],
        panel=panel(),
        build={"native": "test"},
        chat_template_sha256=m.canonical_hash("template"),
        gpu_before={"devices": [dict(index=0, used_mb=4)]},
    )
    monkeypatch.setattr(m, "build_identity", lambda: {"native": "test"})
    monkeypatch.setattr(llama_cpp, "Llama", lambda **kw: model)
    monkeypatch.setattr(m, "EmbeddedRuntime", lambda model: model)
    monkeypatch.setattr(old, "freeze", lambda rows, runtime: deepcopy(rows))
    monkeypatch.setattr(m, "gpu_observation", lambda: {"devices": [dict(index=0, used_mb=16000)]})
    if fault == "stat":
        plan["model_stat"][2] = 0
    if fault == "build":
        plan["build"] = {}
    if fault == "tokens":

        def changed(rows, runtime):
            value = deepcopy(rows)
            value[0]["target_tokens"] = [1]
            return value

        monkeypatch.setattr(old, "freeze", changed)
    if fault == "template":
        plan["chat_template_sha256"] = "bad"
    if fault == "allocation":
        monkeypatch.setattr(m, "gpu_observation", lambda: plan["gpu_before"])
    if fault:
        with pytest.raises(ValueError):
            m.load_model(plan, tmp_path, time_deadline())
        if fault in ["tokens", "template", "allocation"]:
            assert closed
    else:
        loaded, receipt = m.load_model(plan, tmp_path, time_deadline())
        assert loaded is model and receipt["status"] == "completed"


def test_real_precondition_binding_and_gpu_tools(tmp_path, monkeypatch):
    """REQ-REPORT-8071: historical failures remain readable without becoming score inputs."""
    value = m.preconditions(m.ROOT, tmp_path / "good")
    assert not value["failures"] and len(value["panel"]) == 8
    monkeypatch.setattr(
        m, "clean_terminal", lambda p: (_ for _ in ()).throw(ValueError("terminal"))
    )
    assert m.preconditions(m.ROOT, tmp_path / "bad")["failures"]
    assert m.build_identity()["native"].startswith("sha256:")
    import subprocess

    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=a[0], returncode=0, stdout="0, uuid, 4, 24000, 0\n", stderr=""
        ),
    )
    assert m.gpu_observation()["devices"][0]["used_mb"] == 4
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=a[0], returncode=1, stdout="", stderr="denied"
        ),
    )
    with pytest.raises(RuntimeError):
        m.gpu_observation()


@pytest.mark.parametrize(
    "fault", [None, "cache", "hash", "build", "gpu", "lease", "load", "child_exit", "child_result"]
)
def test_live_ownership_and_failed_admission(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8071-SCHEDULE: model admission and cleanup retain terminal slot custody."""
    import types
    from llama_cpp import llama_cpp as native
    from carnot.inference import sota_models
    from carnot import gpu_lease_phase_journal as leases

    path = tmp_path / "Q4_K_M.gguf"
    path.write_bytes(b"model")
    history = json.loads(
        (m.ROOT / "results/raw/experiment_8059_v698_fit_source_scoring/runtime.json").read_text()
    )
    build = history["llama_cpp_build"]["native_library_sha256"]
    plan = dict(
        panel=panel(),
        failures=[],
        references=[],
        historical_model=dict(gguf_sha256=m.sha256_file(path), chat_template_sha256="template"),
    )
    gpu = dict(
        devices=[
            dict(index=i, uuid=f"uuid{i}", used_mb=4, free_mb=24000, utilization_pct=0)
            for i in range(2)
        ]
    )
    calls = []

    class Lease:
        def __init__(self, **kw):
            self.device_uuid = kw["device_uuid"]
            self.document = {"phase": "preflight"}

        def transition(self, phase, **kw):
            self.document["phase"] = phase

        def owner_receipt(self):
            return dict(owned=True)

        def release(self):
            calls.append(self.document["phase"])
            return dict(released=True)

    monkeypatch.setattr(
        sota_models, "cached_current_model", lambda: dict(hf_id=m.MODEL, model_path=str(path))
    )
    monkeypatch.setattr(m, "build_identity", lambda: dict(native=build))
    monkeypatch.setattr(m, "gpu_observation", lambda: gpu)
    monkeypatch.setattr(native, "llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(leases.GpuLease, "acquire", lambda **kw: Lease(**kw))
    monkeypatch.setattr(
        m,
        "load_model",
        lambda *a: (types.SimpleNamespace(close=lambda: None), dict(load_seconds=0.01)),
    )
    monkeypatch.setattr(m, "native_score", lambda model, view, raw, deadline: scorer(view, raw))

    def execute(argv, raw, timeout, **kw):
        child = json.loads(Path(argv[-3]).read_text())
        atomic_json(
            Path(argv[-1]),
            dict(
                scorer(child["view"], Path(argv[-1]).parent),
                passed=fault != "child_result",
                error="child_error",
            ),
        )
        return dict(passed=fault != "child_exit", exit_code=1 if fault == "child_exit" else 0)

    monkeypatch.setattr(m, "execute", execute)
    if fault == "cache":
        monkeypatch.setattr(sota_models, "cached_current_model", lambda: None)
    if fault == "hash":
        plan["historical_model"]["gguf_sha256"] = "missing"
    if fault == "build":
        monkeypatch.setattr(m, "build_identity", lambda: dict(native="other"))
    if fault == "gpu":
        monkeypatch.setattr(native, "llama_supports_gpu_offload", lambda: False)
    if fault == "lease":
        monkeypatch.setattr(
            leases.GpuLease,
            "acquire",
            lambda **kw: (_ for _ in ()).throw(leases.LeaseError("busy")),
        )
    if fault == "load":
        monkeypatch.setattr(
            m, "load_model", lambda *a: (_ for _ in ()).throw(RuntimeError("load_failure"))
        )
    rows = m.live(plan, tmp_path / "raw")
    assert len(rows) == 64
    if fault in ["child_exit", "child_result"]:
        assert sum(r["status"] == "failed" for r in rows) == 1
        assert sum(r["status"] == "completed" for r in rows) == 32
    elif fault:
        assert plan["failures"] and all(r["status"] == "censored" for r in rows)
    else:
        assert all(r["status"] == "completed" for r in rows)
        assert calls == ["terminal_complete"] * 2


@pytest.mark.parametrize("mode", ["success", "failure", "blocked"])
def test_direct_fixture_publication(tmp_path, mode):
    """SCENARIO-REPORT-8071-TERMINAL: publish and replay oracle routes in private storage."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, {"rows": panel()})
    out = tmp_path / (m.NAME + ".json")
    args = ["--fixture-input", str(fixture), "--fixture-mode", mode, "--output", str(out)]
    assert m.main(args) == 0
    assert m.main(args) == 0
    assert m.main(["--cold-replay", str(out)]) == 0
    v = json.loads(out.read_text())
    raw = out.parent / "raw" / out.stem
    candidate = raw / "terminal_candidate.json"
    bad = deepcopy(v)
    bad["raw_shard_hashes"][0]["sha256"] = "bad"
    atomic_json(candidate, bad)
    with pytest.raises(ValueError, match="evidence_hash"):
        m.check_candidate(candidate)
    atomic_json(candidate, v)
    bad = deepcopy(v)
    bad["completed_count"] = 999
    atomic_json(candidate, bad)
    with pytest.raises(ValueError, match="completed_count"):
        m.check_candidate(candidate)
    bad = deepcopy(v)
    bad["scored_tokens"] = 999
    atomic_json(candidate, bad)
    with pytest.raises(ValueError, match="reduction_changed"):
        m.check_candidate(candidate)
    assert not m.replay(out)
    atomic_json(candidate, v)
    terminal = Path(v["terminal_validation_sidecar_path"])
    doc = json.loads(terminal.read_text())
    sidecar = Path(doc["sidecar_path"])
    value = json.loads(sidecar.read_text())
    value["report"]["passed"] = False
    atomic_json(sidecar, value)
    assert not m.replay(out)


@pytest.mark.parametrize(
    "blocked,validation_failed", [(False, False), (True, False), (False, True)]
)
def test_direct_live_publication_routes(tmp_path, monkeypatch, blocked, validation_failed):
    """SCENARIO-REPORT-8071-TERMINAL: owned failures remove readiness, external absence stays blocked."""
    fixture = tmp_path / "input.json"
    atomic_json(fixture, {"rows": panel()})

    def preconditions(root, raw):
        atomic_json(raw / "panel.json", {"rows": panel()})
        return dict(
            panel=[] if blocked else panel(),
            failures=[dict(check="missing")] if blocked else [],
            references=[dict(path=str(fixture), sha256=m.sha256_file(fixture))],
        )

    monkeypatch.setattr(m, "preconditions", preconditions)
    monkeypatch.setattr(
        m, "live", lambda p, raw: m.fixture_measure(p["panel"], raw / "forwards", "success")
    )

    def execute(argv, raw, timeout, **kw):
        raw.mkdir(parents=True, exist_ok=True)
        log = raw / "child.log"
        log.write_text("private mocked command")
        if "json" in argv and "-o" in argv:
            atomic_json(Path(argv[argv.index("-o") + 1]), {"files": {}})
        return dict(
            argv=argv,
            exit_code=1 if validation_failed else 0,
            passed=not validation_failed,
            duration_s=0.01,
            log_path=str(log),
            log_sha256=m.sha256_file(log),
            normal_exit=True,
        )

    monkeypatch.setattr(m, "execute", execute)
    out = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(out)]) == 0
    v = json.loads(out.read_text())
    assert v["verdict_class"] == (
        "disqualified" if validation_failed else "blocked" if blocked else "null"
    )
    assert m.replay(out)


def test_worker_dispatch_and_thin_wrapper(tmp_path, monkeypatch):
    """REQ-REPORT-8071: script bootstrap and worker modes share one qualified implementation."""
    import runpy

    calls = []
    monkeypatch.setattr(m, "worker", lambda p, o: calls.append((p, o)))
    assert (
        m.main(
            [
                "--worker",
                str(tmp_path / "plan.json"),
                "--worker-output",
                str(tmp_path / "worker.json"),
            ]
        )
        == 0
    )
    assert calls
    atomic_json(tmp_path / "fixture.json", dict(panel=panel(), mode="success"))
    assert (
        m.main(
            [
                "--fixture-worker",
                str(tmp_path / "fixture.json"),
                "--worker-output",
                str(tmp_path / "output.json"),
            ]
        )
        == 0
    )
    monkeypatch.setattr(m, "main", lambda: 0)
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(m.CLI), run_name="__main__")
    assert e.value.code == 0


def test_aggregate_and_normalization_mutations(tmp_path):
    """SCENARIO-REPORT-8071-CUSTODY: mean agreement cannot override raw normalization custody."""
    rows = m.capture(panel(), scorer, scorer, tmp_path, deadline=math.inf)
    rows[0]["mean_nll"] += 0.1
    with pytest.raises(ValueError, match="aggregate"):
        m.reduce(panel(), rows)


@pytest.mark.parametrize("fault", ["hash", "incomplete", "runtime"])
def test_historical_panel_mutations(tmp_path, monkeypatch, fault):
    """REQ-REPORT-8071: corrupt historical tokens cannot become current scientific inputs."""
    root = tmp_path / "repo"
    primary = root / "results/experiment_8059_v698_fit_source_scoring.json"
    historical = primary.parent / "raw" / primary.stem / "frozen_panel.json"
    rows = panel()
    if fault == "incomplete":
        rows[0]["eligible"] = False
    atomic_json(historical, {"rows": rows})
    runtime = historical.with_name("runtime.json")
    atomic_json(runtime, {})
    atomic_json(
        primary,
        {
            "raw_shard_hashes": [
                dict(
                    path=str(runtime),
                    sha256="bad" if fault == "runtime" else m.sha256_file(runtime),
                ),
                dict(
                    path=str(historical),
                    sha256="bad" if fault == "hash" else m.sha256_file(historical),
                ),
            ]
        },
    )
    monkeypatch.setattr(
        m, "clean_terminal", lambda p: dict(terminal_path=str(primary), validator_path=str(primary))
    )
    v = m.preconditions(root, tmp_path / "raw")
    assert v["failures"][-1]["field"] == "historical_custody"


def test_supervisor_hard_kill_and_heartbeat(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8071-SCHEDULE: an unresponsive child remains bounded without invented units."""
    import subprocess

    waits = []

    class Child:
        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return -9

    ticks = iter([0, 0, 31, 100, 101, 102])
    monkeypatch.setattr(m.time, "monotonic", lambda: next(ticks, 103))
    monkeypatch.setattr(m, "progress", lambda *a: None)
    monkeypatch.setattr(m.time, "sleep", lambda t: None)
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: Child())
    assert m.execute(["child"], tmp_path, 60)["timed_out"]
    assert len(waits) == 3


def test_fixture_failures_and_terminal_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8071-TERMINAL: failed final replay cannot be reported as successful publication."""
    assert any(
        r["status"] == "failed" for r in m.fixture_measure(panel(), tmp_path / "failed", "failure")
    )
    fixture = tmp_path / "input.json"
    atomic_json(fixture, {"rows": panel()})
    monkeypatch.setattr(m, "replay", lambda p: False)
    with pytest.raises(ValueError, match="terminal_cold_replay"):
        m.main(["--fixture-input", str(fixture), "--output", str(tmp_path / (m.NAME + ".json"))])
