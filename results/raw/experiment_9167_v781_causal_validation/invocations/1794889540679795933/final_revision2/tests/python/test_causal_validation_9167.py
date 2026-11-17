"""REQ-REPORT-9167-V781: trace actual descendants and refuse unsupported causality."""

import json
from pathlib import Path
import os
import subprocess
from typing import Any

from carnot.reporting import validation_trace_9167 as trace


def test_descendant_trace_before_imports(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9167-V781: real grandchildren retain identity and completed work."""
    bootstrap = trace.bootstrap(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(bootstrap), CARNOT_9167_TRACE=str(tmp_path))
    result = subprocess.run(
        [
            os.sys.executable,
            "-c",
            'import subprocess,sys; subprocess.run([sys.executable,"-c","print(123)"],check=True)',
        ],
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )
    assert result.stdout.strip() == "123"
    files = sorted(tmp_path.glob("trace-*.jsonl"))
    assert len(files) == 2
    events = [json.loads(line) for p in files for line in p.read_text().splitlines()]
    starts = [e for e in events if e["event"] == "start"]
    assert len(starts) == 2 and starts[1]["ppid"] in {e["pid"] for e in starts}
    assert all(e["argv"] and e["monotonic_ns"] for e in starts)
    ends = [e for e in events if e["event"] == "summary"]
    assert len(ends) == 2 and all(e["overhead_ns"] >= 0 for e in ends)
    assert any(e["event"] == "spawn" for e in events)
    assert any(e["event"] == "child_exit" and e["exit_code"] == 0 for e in events)


def test_proc_permission_failure_is_evidence(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9167-V781: inaccessible process metadata remains an explicit refusal."""
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied"))
    )
    assert trace.proc(os.getpid())["error"].startswith("PermissionError:")


def test_authentication_gates_and_preserved_trace(tmp_path: Path, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-9167-V781: exact original pins precede dependent reads."""
    from carnot.reporting import causal_validation_9167 as runner
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file

    original = tmp_path / "upstream.json"
    trace_path = tmp_path / "old.stderr"
    trace_path.write_text("preserved original stack")
    atomic_json(
        original,
        dict(ready=1, original_trace=dict(path=str(trace_path), sha256=sha256_file(trace_path))),
    )
    monkeypatch.setattr(runner, "INPUTS", {"upstream.json": (sha256_file(original), "ready", 1)})
    missing = runner.authenticate(tmp_path / "absent", tmp_path / "raw1")
    assert missing["checks"][0]["observed"] == "missing_file"
    good = runner.authenticate(tmp_path, tmp_path / "raw2")
    assert all(c["passed"] for c in good["checks"])
    for observed in [0, True, "missing_field"]:
        atomic_json(original, {} if observed == "missing_field" else dict(ready=observed))
        monkeypatch.setattr(
            runner, "INPUTS", {"upstream.json": (sha256_file(original), "ready", 1)}
        )
        work = runner.authenticate(tmp_path, tmp_path / str(observed))
        assert work["checks"][-1]["observed"] == observed
        assert not work["checks"][-1]["passed"]


def test_no_trace_cannot_establish_cause(tmp_path: Path) -> None:
    """REQ-REPORT-9167-V781: no parent timing or empty trace can qualify a correction."""
    from carnot.reporting import causal_validation_9167 as runner

    evidence = runner.trace_evidence(tmp_path)
    assert evidence["cause_demonstrated"] is False
    assert evidence["corrective_execution_passed"] is False
    assert evidence["operation_rows"] == []


def test_blocked_artifact_reduction_and_replay(tmp_path: Path) -> None:
    """REQ-REPORT-9167-V781: external absence remains terminal and tamper-evident."""
    from carnot.reporting import causal_validation_9167 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    work = runner.prepare(tmp_path / "missing", tmp_path / "raw", "20261117")
    atomic_json(tmp_path / "raw/work.json", work)
    value = runner.build(work)
    assert value["verdict_class"] == "blocked"
    assert value["causal_fix_ready_score"] == 0
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    value["causal_fix_ready_score"] = 1
    value["reproducibility_checksum"] = runner.canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(path, value)
    import pytest

    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(path)


def test_preprocessing_reuse_invalidates_changed_bytes_and_config(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """SCENARIO-REPORT-9167-V781: reuse preprocessing, never verdicts or changed inputs."""
    from carnot.reporting import validation_reuse_9167 as reuse
    from scripts import adversarial_verify as av

    calls = []
    payload = dict(schema="probe", nested=dict(x=1))

    def classify(value: dict[str, Any]) -> bool:
        calls.append(json.loads(json.dumps(value)))
        return value["nested"]["x"] == 1

    monkeypatch.setattr(av, "_is_arc_generation_or_exploration_artifact", classify)

    def verify(path: Path) -> dict[str, Any]:
        assert av._is_arc_generation_or_exploration_artifact(payload)
        assert av._is_arc_generation_or_exploration_artifact(payload)
        payload["nested"]["x"] = 2
        assert not av._is_arc_generation_or_exploration_artifact(payload)
        monkeypatch.setattr(av, "_LEVER_EXERCISE_CONTEXT_MARKERS", ("changed",))
        assert not av._is_arc_generation_or_exploration_artifact(payload)
        return dict(actual_checker_executed=True)

    monkeypatch.setattr(av, "_verify_artifact_impl", verify)
    assert reuse.current_report(tmp_path / "input.json")["actual_checker_executed"]
    assert len(calls) == 3
    assert av._is_arc_generation_or_exploration_artifact is classify
    payload["nested"]["x"] = 1
    assert reuse.current_report(tmp_path / "input.json")["actual_checker_executed"]
    assert len(calls) == 5


def test_reuse_binds_checker_revision_and_arc_configuration(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """REQ-REPORT-9167-V781: each effective validator revision/config change executes anew."""
    from carnot.reporting import validation_reuse_9167 as reuse
    from scripts import adversarial_verify as av

    revision = tmp_path / "validator.py"
    revision.write_text("revision one")
    monkeypatch.setattr(av, "__file__", str(revision))
    count = []
    monkeypatch.setattr(
        av, "_is_arc_generation_or_exploration_artifact", lambda d: bool(count.append(1))
    )

    def verify(path: Path) -> dict[str, Any]:
        for stage in range(4):
            if stage == 1:
                revision.write_text("revision two")
            if stage == 2:
                monkeypatch.setattr(av, "_ARC_GAME_IDS", ("changed_arc",))
            av._is_arc_generation_or_exploration_artifact({})
        return {}

    monkeypatch.setattr(av, "_verify_artifact_impl", verify)
    reuse.current_report(tmp_path / "unused")
    assert len(count) == 3


def test_structural_controls_use_preprocessing_without_changing_refusals(monkeypatch: Any) -> None:
    """SCENARIO-REPORT-9167-V781: direct negative checks also execute with identical flags."""
    from carnot.reporting import forward_evidence_runner_9111 as old
    from scripts import adversarial_verify as av

    count = []
    original = av._is_arc_generation_or_exploration_artifact

    def classify(d: dict[str, Any]) -> bool:
        count.append(1)
        return original(d)

    monkeypatch.setattr(av, "_is_arc_generation_or_exploration_artifact", classify)
    diagnostic = dict(
        rows=[{}],
        qid_rows=[dict(denominator=3)],
        accuracy=0.7654321,
        auroc=0.7654321,
        metric_a=0.123456789,
        metric_b=0.123456789,
        metric_c=0.123456789,
    )
    controls = old.structural_controls(diagnostic)
    assert all(c["passed"] for c in controls)
    assert len(count) <= len(controls)


def test_private_cli_success_refusal_tamper_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9167-V781: real private CLIs run outside checkout without PYTHONPATH."""
    from carnot.reporting import causal_validation_9167 as r
    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting.validation_schedule_runner_9139 import foreign_argv

    work = r.prepare(tmp_path / "absent", tmp_path / "work", "20261117")
    path = tmp_path / "work/work.json"
    atomic_json(path, work)
    output = tmp_path / (r.NAME + ".json")
    commands = [
        ("success", foreign_argv(r.CLI, "--from-work", str(path), "--output", str(output)), 0),
        ("cold", foreign_argv(r.CLI, "--replay", str(output)), 0),
        ("date_refusal", foreign_argv(r.CLI, "--date", "bad"), 1),
    ]
    for name, argv, expected in commands:
        assert r.child(name, argv, tmp_path / "logs", expected=expected, deadline=120)["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["causal_fix_ready_score"] == 0
    value["completed_count"] += 1
    value["reproducibility_checksum"] = r.canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert r.child(
        "tamper", foreign_argv(r.CLI, "--replay", str(output)), tmp_path / "logs", expected=1
    )["passed"]
    assert (
        r.main(["--from-work", str(path), "--output", str(r.ROOT / "results" / output.name)]) == 1
    )
    assert r.main(["--replay", str(tmp_path / "absent.json")]) == 1


def test_real_receipts_coverage_and_refusals(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9167-V781: exact logs, complete statements and commands gate qualification."""
    from copy import deepcopy
    import runpy
    from coverage import Coverage
    from carnot.reporting import causal_validation_9167 as r

    monkeypatch.delenv("COVERAGE_FILE", raising=False)
    source = tmp_path / "tiny.py"
    source.write_text("executed = 123\n")
    tracer = Coverage(
        config_file=False,
        data_file=str(tmp_path / ".coverage"),
        include=[str(source)],
        data_suffix=True,
    )
    tracer.start()
    runpy.run_path(str(source))
    tracer.stop()
    tracer.save()
    (tmp_path / "coverage.ini").write_text(
        "[run]\nparallel=true\ndata_file="
        + str(tmp_path / ".coverage")
        + "\n[report]\nexclude_lines=\n"
    )
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "OWNED", [])
    monkeypatch.setattr(r, "SEAM", "tiny.py")
    monkeypatch.setattr(r, "BASELINE", tmp_path / "before.py")
    r.BASELINE.write_text("")
    monkeypatch.setattr(r, "BASELINE_PIN", r.sha256_file(r.BASELINE))
    argv = [os.sys.executable, "-c", 'print("actual child log")']
    spec = dict(name="real", argv=argv)
    monkeypatch.setattr(r, "commands", lambda raw: [spec])
    work = dict(raw=str(tmp_path), commands=[spec])
    r.verify(work, r.time.monotonic() + 120)
    assert work["coverage"]["statement_percent"] == 100
    monkeypatch.setattr(r, "OWNED", [])
    assert r.qualified(work)
    for field, value in [
        ("passed", False),
        ("normal_exit", False),
        ("timed_out", True),
        ("name", "wrong"),
        ("argv", []),
    ]:
        altered = deepcopy(work)
        altered["validation_receipts"][0][field] = value
        assert not r.qualified(altered)
    altered = deepcopy(work)
    altered["commands"] = []
    assert not r.qualified(altered)
    altered = deepcopy(work)
    altered["coverage"]["files"][0]["missing"] = [1]
    assert not r.qualified(altered)
    monkeypatch.setattr(r, "OWNED", ["tiny.py"])
    assert not r.qualified(work)


def test_normal_main_branches(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9167-V781: bounded main always records the owned execution it attempted."""
    from carnot.reporting import causal_validation_9167 as r

    prepare = r.prepare

    def prepared(root: Path, raw: Path, date: str) -> dict[str, Any]:
        work = prepare(r.ROOT, raw, date)
        for c in work["checks"]:
            c["passed"] = True
        return work

    monkeypatch.setattr(r, "prepare", prepared)
    monkeypatch.setattr(
        r,
        "diagnose",
        lambda *a: dict(
            directory=str(tmp_path),
            receipt=dict(timed_out=True, normal_exit=False, deadline_s=1, duration_s=0),
            evidence={},
        ),
    )
    monkeypatch.setattr(r, "verify", lambda *a: None)
    monkeypatch.setattr(r, "child", lambda *a, **k: dict(passed=False))
    published = []
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, validator: published.append(value) or {}
    )
    assert r.main(["--output", str(tmp_path / (r.NAME + ".json"))]) == 0
    assert published[0]["causal_fix_ready_score"] == 0
    monkeypatch.setenv(
        "CARNOT_9167_DIAGNOSIS_REFERENCE",
        json.dumps(r.ref(r.ROOT / "results/raw" / r.NAME / "diagnosis_manifest.json")),
    )
    assert r.main(["--output", str(tmp_path / (r.NAME + ".json"))]) == 0
    assert published[-1]["verdict_class"] == "disqualified"


def test_diagnose_invokes_original_fixture_and_binds_plan(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9167-V781: watchdog ceilings bound actual commands, never elapsed predictions."""
    from carnot.reporting import causal_validation_9167 as r

    calls = []
    monkeypatch.setattr(
        r, "child", lambda n, a, p, **kw: calls.append((a, kw)) or dict(passed=False)
    )
    result = r.diagnose(tmp_path, r.time.monotonic() - 1)
    assert calls[0][0][-1] == r.NODE
    assert calls[0][1]["deadline"] == 0.001
    plan = json.loads((tmp_path / "diagnostic_plan.json").read_bytes())
    assert plan["assertion_reference"]["sha256"]
    assert result["evidence"]["cause_demonstrated"] is False


def test_trace_callbacks_and_heartbeat_without_waiting(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9167-V781: exercise callback accounting and heartbeat scheduling with a fake clock wait."""
    import sys
    from types import SimpleNamespace

    callbacks = {}
    finished = []
    audits = []

    class Event:
        def __init__(self) -> None:
            self.count = 0

        def wait(self, seconds: int) -> bool:
            assert seconds == 60
            self.count += 1
            return self.count > 1

        def set(self) -> None:
            pass

    class Thread:
        def __init__(self, target: Any, daemon: bool) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    monitoring = SimpleNamespace(
        DISABLE=object(),
        events=SimpleNamespace(PY_START=1, PY_RETURN=2),
        use_tool_id=lambda *a: None,
        free_tool_id=lambda *a: None,
        set_events=lambda *a: None,
        set_local_events=lambda *a: None,
        register_callback=lambda tool, event, callback: callbacks.__setitem__(event, callback),
    )
    monkeypatch.setattr(
        trace,
        "sys",
        SimpleNamespace(
            monitoring=monitoring,
            _getframe=sys._getframe,
            _current_frames=sys._current_frames,
            argv=["unit_control"],
            addaudithook=audits.append,
        ),
    )
    monkeypatch.setattr(trace, "threading", SimpleNamespace(Event=Event, Thread=Thread))
    monkeypatch.setattr(trace, "atexit", SimpleNamespace(register=finished.append))
    monkeypatch.setenv("CARNOT_9167_TRACE", str(tmp_path))
    trace.install()

    def measured_work() -> None:
        frame = sys._getframe()
        callbacks[1](frame.f_code, 0)
        callbacks[2](frame.f_code, 0, None)

    measured_work()
    measured_work()
    namespace = dict(sys=sys, callbacks=callbacks)
    exec(
        compile(
            "def wait(self):\n f=sys._getframe()\n callbacks[1](f.f_code,0)\n callbacks[2](f.f_code,0,0)\n",
            "subprocess.py",
            "exec",
        ),
        namespace,
    )
    namespace["wait"](SimpleNamespace(pid=123))
    assert (
        callbacks[1](test_trace_callbacks_and_heartbeat_without_waiting.__code__, 0)
        is monitoring.DISABLE
    )
    audits[0]("open", ("path",))
    audits[0]("subprocess.Popen", ("python", ["python"], "/tmp"))
    finished[0]()
    events = [json.loads(l) for l in next(tmp_path.glob("trace-*.jsonl")).read_text().splitlines()]
    assert any(e["event"] == "heartbeat" for e in events)
    assert any(e["event"] == "return" and e["elapsed_ns"] >= 0 for e in events)
    assert any(e["event"] == "child_exit" and e["child_pid"] == 123 for e in events)
    assert events[-1]["counters"]["audit:open"] == 1


def test_retained_diagnosis_is_disqualified_and_tamper_evident(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9167-V781: real timed-out module evidence cannot become readiness."""
    from copy import deepcopy
    import pytest
    from carnot.reporting import causal_validation_9167 as r
    from carnot.reporting.current_work_receipt import atomic_json

    reference = r.ref(r.ROOT / "results/raw" / r.NAME / "diagnosis_manifest.json")
    diagnosis = r.adopt_diagnosis(reference)
    work = r.prepare(r.ROOT, tmp_path / "work", "20261117")
    work["diagnosis"] = diagnosis
    work["node_events"] = [
        json.loads(l)
        for l in (Path(diagnosis["directory"]) / "original/nodes.jsonl").read_text().splitlines()
    ]
    status = r.causal_status(work)
    assert status["cause_demonstrated"] and not status["ready"]
    assert status["unfinished_module_nodes"]
    atomic_json(tmp_path / "work/work.json", work)
    value = r.build(work)
    assert value["verdict_class"] == "disqualified" and value["causal_fix_ready_score"] == 0
    assert value["attempted_count"] == sum(row["attempted"] for row in value["rows"]) == 2
    assert (
        value["cost_model_rows"][1]["measured_elapsed_s"]
        == diagnosis["corrected_receipt"]["duration_s"]
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert r.replay(path)["passed"]
    from scripts import adversarial_verify as av

    flags: list[Any] = []
    av.check_methodology_present(value, flags)
    assert not flags
    altered = deepcopy(work)
    altered["diagnosis"]["evidence"]["operation_rows"] = []
    with pytest.raises(ValueError, match="diagnosis_drift"):
        r.causal_status(altered)


def test_replay_rejects_input_trace_and_module_scope_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9167-V781: rehashing narrowed scope or invented trace does not qualify."""
    from copy import deepcopy
    import pytest
    from carnot.reporting import causal_validation_9167 as r
    from carnot.reporting.current_work_receipt import atomic_json

    work = r.prepare(r.ROOT, tmp_path / "raw", "20261117")
    work["diagnosis"] = r.adopt_diagnosis(
        r.ref(r.ROOT / "results/raw" / r.NAME / "diagnosis_manifest.json")
    )
    workpath = tmp_path / "raw/work.json"
    atomic_json(workpath, work)
    value = r.build(work)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    altered = deepcopy(work)
    altered["inputs"][next(iter(r.INPUTS))]["contract_ready_score"] = 0
    atomic_json(workpath, altered)
    value["work_reference"] = r.ref(workpath)
    atomic_json(path, value)
    with pytest.raises(ValueError, match="input_drift"):
        r.replay(path)
    altered = deepcopy(work)
    altered["diagnosis"]["evidence"]["operation_rows"] = []
    atomic_json(workpath, altered)
    value["work_reference"] = r.ref(workpath)
    atomic_json(path, value)
    with pytest.raises(ValueError, match="trace_drift"):
        r.replay(path)
    altered = deepcopy(work)
    altered["inputs"][next(iter(r.INPUTS))]["validation_obligations"]["original_node_ids"] = []
    with pytest.raises(ValueError, match="module_scope_drift"):
        r.causal_status(altered)


def test_original_measure_replay_and_all_primitive_tamper_assertions(tmp_path: Path) -> None:
    """REQ-REPORT-9167-V781: every original tamper assertion still executes its current checker."""
    import runpy
    from carnot.reporting import forward_evidence_runner_9111 as old
    from carnot.reporting.current_work_receipt import atomic_json

    work = old.measure(old.authenticate(old.ROOT, tmp_path), tmp_path)
    assert all(row["passed"] for row in work["preserved_detection_rows"])
    path = tmp_path / "candidate.json"
    atomic_json(path, old.build(work, [], {}, "20261115", 0, 1))
    assert old.replay(path)["passed"]
    original = runpy.run_path(str(old.ROOT / old.TESTS[1]))["test_rehashed_primitive_tampering"]
    for change in [
        "checksum",
        "derivation",
        "flags",
        "roster",
        "diagnostic_input",
        "diagnostic_report",
        "structural_control",
    ]:
        original(tmp_path, work, change)
