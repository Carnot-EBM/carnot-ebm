"""REQ-REPORT-8764, REQ-VERIFY-8764: bounded diagnosis never promotes old failures."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import pytest

from carnot.reporting import publication_cost_8764 as e
from carnot.reporting import publication_probe_8764 as p
from carnot.reporting.current_work_receipt import atomic_json


def test_missing_events_and_disjoint_boundaries(tmp_path):
    """SCENARIO-REPORT-8764-NULL: absent event keys leave unknown spans explicit."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"stage":"legacy","boundary":"start"}\n')
    assert p.summarize(path)["unknown_event_count"] == 1
    with p.boundaries(path):
        assert p.identity({"path": str(path), "sha256": "bad"}) is False
    result = p.summarize(path)
    assert result["counts"]["source_read"] >= 1
    assert not result["cache_enabled"]
    assert time.monotonic is p.REAL_CLOCK


def test_supervisor_kills_and_reaps_group(tmp_path):
    """REQ-VERIFY-8764: malformed legacy events cannot block real kill and wait."""
    (tmp_path / "stages.jsonl").write_text('{"stage":"missing_event"}\n')
    row = p.supervise("kill", [sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, 0.05)
    assert row["timed_out"] and row["exit_code"] == -9 and row["cleanup_passed"]
    with pytest.raises(ChildProcessError):
        os.waitpid(row["pid"], os.WNOHANG)
    row = p.supervise("ok", [sys.executable, "-c", "print('real child')"], tmp_path, 10)
    assert row["passed"] and row["cleanup_passed"]


def test_identity_mutation_and_no_verdict_cache(tmp_path):
    """REQ-VERIFY-8764: every read rehashes content, even after a prior success."""
    path = tmp_path / "input.json"
    atomic_json(path, dict(value=1))
    ref = p.ref(path)
    assert p.identity(ref)
    atomic_json(path, dict(value=2))
    assert not p.identity(ref)
    assert not p.identity(dict(path=str(tmp_path / "missing"), sha256="bad"))
    with pytest.raises(ValueError, match="input_drift"):
        p.authenticate([ref])


def test_terminal_null_and_rehashed_attacks(tmp_path):
    """SCENARIO-VERIFY-8764-REPLAY: corrected checksums cannot manufacture readiness."""
    work = e.empty(tmp_path, "20261103")
    work["checks"] = dict(unit=True)
    e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work)
    atomic_json(output, value)
    assert e.replay(output)
    assert value["verdict_class"] == "null" and value["publication_path_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["independent_count"] == 0
    for field, changed in [("publication_path_ready_score", 1), ("duration_s", -1), ("rows", [])]:
        attack = deepcopy(value)
        attack[field] = changed
        attack["reproducibility_checksum"] = e.old.digest(attack)
        atomic_json(output, attack)
        assert not e.replay(output)
    assert not e.replay(tmp_path / "missing")
    work["checks"]["unit"] = False
    assert e.build(work)["verdict_class"] == "disqualified"


def test_clock_and_failed_child_rejection(tmp_path):
    """REQ-VERIFY-8764: elapsed clocks and child success are authenticated operands."""
    work = e.empty(tmp_path, "20261103")
    work["checks"] = dict(unit=True)
    work["receipts"] = [dict(passed=False, normal_exit=False)]
    assert not e.accepted(work)
    work["receipts"] = []
    work["duration_s"] = -2
    assert not e.accepted(work)


def test_frozen_transitive_content(tmp_path):
    """REQ-REPORT-8764: nested source references join the frozen closure."""
    inner = tmp_path / "inner.json"
    outer = tmp_path / "outer.json"
    atomic_json(inner, dict(number=1))
    atomic_json(outer, dict(nested=[p.ref(inner)]))
    refs = p.closure([p.ref(outer)])
    assert {r["path"] for r in refs} == {str(inner), str(outer)}
    p.authenticate(refs)


def test_real_cli_controls(tmp_path):
    """SCENARIO-VERIFY-8764-REPLAY: real standalone children cover exit and replay paths."""
    cli = [sys.executable, str(e.ROOT / e.CLI)]
    for name, args, code in [
        ("date", ["--date", "bad"], 2),
        ("calendar", ["--date", "20260230"], 2),
        ("unsafe", ["--private-output", str(e.ROOT / "bad")], 2),
        ("replay", ["--cold-replay", str(tmp_path / "missing")], 1),
    ]:
        receipt = p.supervise(name, [*cli, *args], tmp_path / name, 20)
        assert receipt["exit_code"] == code


def test_historical_receipts_are_opaque_numeric_operands(tmp_path):
    """REQ-REPORT-8764: old validation histories do not become current executable inputs."""
    retired = tmp_path / "retired.json"
    parent = tmp_path / "parent.json"
    atomic_json(parent, dict(historical=dict(path=str(retired), sha256="retired")))
    assert p.closure([p.ref(parent)], opaque={str(parent)}) == [p.ref(parent)]
    with pytest.raises(ValueError, match="input_drift"):
        p.closure([p.ref(parent)])


def test_conflicting_input_pins(tmp_path):
    """REQ-VERIFY-8764: incompatible pins cannot identify one immutable computation."""
    path = tmp_path / "operand.json"
    atomic_json(path, dict(value=1))
    with pytest.raises(ValueError, match="conflicting_input_identity"):
        p.closure([dict(path=str(path), sha256="other"), p.ref(path)])


def test_original_stages_only_wrap_boundaries(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8764-NULL: stage dispatch leaves original assertions callable."""
    from carnot.reporting import invocation_clock_core_8724 as clock

    plan = tmp_path / "plan.json"
    atomic_json(plan, dict(inputs=[], sources=[], scratch=str(tmp_path), fixture={}))
    calls = []
    monkeypatch.setattr(
        p.stream, "inspect", lambda *a: calls.append("precondition") or dict(failures=[])
    )

    def fixture(*args):
        atomic_json(tmp_path / "fixture/measurement.json", dict(raw=str(tmp_path)))
        calls.append("copy")

    monkeypatch.setattr(clock, "fixture", fixture)
    monkeypatch.setattr(p.stream, "build", lambda *a: dict(measured=False))
    monkeypatch.setattr(p.stream, "replay", lambda *a: True)
    for stage in ["precondition", "copy", "materialize", "cold_replay"]:
        assert p.stage(stage, plan) == 0
    assert calls == ["precondition", "copy"]
    with pytest.raises(ValueError, match="unknown_stage"):
        p.stage("bogus", plan)
    original = p.REAL_CLOCK
    assert time.monotonic is original
    monkeypatch.setattr(p, "REPORT_INTERVAL_NS", 0)
    with p.boundaries(tmp_path / "calls.jsonl"):
        from carnot.reporting import brier_online_learning_8517 as learner
        from carnot.verify import objective_stream_8517 as numeric

        sample = tmp_path / "sample.json"
        atomic_json(sample, dict(value=1))
        assert learner.read(learner.reference(sample)) == dict(value=1)
        with pytest.raises((KeyError, TypeError)):
            numeric.check_step({}, {})
    assert p.summarize(tmp_path / "calls.jsonl")["counts"]["numerical_step_replay"] == 1


def test_diagnostic_plan_and_stop(tmp_path, monkeypatch):
    """REQ-REPORT-8764: one failure censors the rest and never repeats a timeout."""
    plan = tmp_path / "plan.json"
    atomic_json(plan, dict(scratch=str(tmp_path), expected_scan_findings=[]))
    commands = []

    def supervisor(name, argv, raw, budget, **options):
        commands.append(argv)
        event = tmp_path / (name + ".events.jsonl")
        event.write_text('{"event":"counts","counts":{}}\n')
        stdout = tmp_path / (name + ".stdout")
        atomic_json(stdout, dict(reports=[dict(flags=[])]))
        return dict(
            stdout_path=str(stdout),
            passed=True,
            normal_exit=True,
            cleanup_passed=True,
            pid=1,
            duration_s=0.1,
            timed_out=False,
        )

    monkeypatch.setattr(p, "supervise", supervisor)
    work = e.empty(tmp_path / "raw", "20261103")
    Path(work["raw"]).mkdir()
    e.diagnostic(work, plan)
    assert len(commands) == 6 and len(work["stage_rows"]) == 6
    assert "adversarial_verify.py" in commands[3][2]
    assert "--strict" in commands[4]
    monkeypatch.setattr(
        p,
        "supervise",
        lambda *a, **k: dict(
            passed=False,
            normal_exit=False,
            cleanup_passed=True,
            pid=1,
            duration_s=0.1,
            timed_out=True,
        ),
    )
    work = e.empty(tmp_path / "failed", "20261103")
    Path(work["raw"]).mkdir()
    e.diagnostic(work, plan)
    assert len(work["stage_rows"]) == 1 and work["stage_rows"][0]["status"] == "censored"


def test_replay_rejects_seal_log_and_clock_drift(tmp_path):
    """SCENARIO-VERIFY-8764-REPLAY: stale success cannot authenticate changed child evidence."""
    work = e.empty(tmp_path, "20261103")
    row = p.supervise("evidence", [sys.executable, "-c", "pass"], tmp_path / "logs", 10)
    work["receipts"] = [row]
    work["checks"] = dict(unit=True)
    e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work))
    assert e.replay(output)
    for attack in ["seal", "clock", "log"]:
        changed = deepcopy(work)
        if attack == "seal":
            atomic_json(Path(work["raw"]) / "work.json", dict(changed, duration_s=1))
        elif attack == "clock":
            changed["receipts"][0]["duration_s"] = -5
            e.seal(changed)
        else:
            Path(row["stdout_path"]).write_text("changed after success")
            e.seal(changed)
        value = e.build(changed)
        atomic_json(output, value)
        assert not e.replay(output)
        e.seal(work)


@pytest.mark.parametrize("retained", [False, True])
def test_failed_coverage_never_qualifies(tmp_path, monkeypatch, retained):
    """REQ-VERIFY-8764: absent statements remain failures despite successful command exits."""
    from coverage import Coverage
    from types import SimpleNamespace

    work = e.empty(tmp_path, "20261103")
    if retained:
        work["historical_evidence"]["owned_aborted_diagnostics"] = [
            dict(repository_health_receipt=dict(passed=False))
        ]
    data = tmp_path / ".coverage.combined"
    data.write_bytes(b"unit-only coverage adapter")
    meter = SimpleNamespace(
        combine=lambda **k: None,
        save=lambda: None,
        analysis2=lambda name: (name, [1, 2], [], [2], "2"),
    )
    monkeypatch.setattr("coverage.Coverage", lambda **k: meter)
    monkeypatch.setattr(
        p,
        "supervise",
        lambda name, *a: dict(name=name, passed=True, normal_exit=True, cleanup_passed=True),
    )
    e.validate(work)
    assert work["checks"]["new_statement_coverage"] is False
    assert work["repository_health_reference"]["scoped_prerequisite"] is False


def test_normal_publication_with_private_dependencies(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8764-REPLAY: real scanners and fresh replay gate exact final bytes."""
    monkeypatch.setattr(e, "prepare", lambda work: tmp_path / "plan")
    monkeypatch.setattr(
        e,
        "diagnostic",
        lambda work, plan: work["receipts"].append(
            p.supervise("real_boundary", [sys.executable, "-c", "pass"], tmp_path / "child", 10)
        ),
    )
    monkeypatch.setattr(e, "validate", lambda work: work["checks"].update(unit=True))
    output = tmp_path / (e.NAME + ".json")
    assert e.run(output, "20261103") == 0
    assert e.replay(output)
    sidecar = json.loads(
        Path(json.loads(output.read_bytes())["terminal_validation_sidecar_path"]).read_bytes()
    )
    assert sidecar["selected_identity"]["passed"]
    assert e.main(["--stage", "copy"]) == 2
    monkeypatch.setattr(p, "stage", lambda *a: 0)
    assert e.main(["--stage", "copy", "--plan", str(tmp_path / "plan")]) == 0
    monkeypatch.setattr(e, "run", lambda *a, **k: 0)
    assert e.main(["--private-output", str(output)]) == 0


def test_prepare_freezes_private_computational_scope(tmp_path, monkeypatch):
    """REQ-REPORT-8764: full consumed operands freeze while historical validation stays ineligible."""
    source = tmp_path / "source.py"
    source.write_text("SOURCE=1\n")
    fixture = tmp_path / "measurement.json"
    atomic_json(
        fixture,
        dict(
            source_refs=[], code_refs=[p.ref(source)], primitive_files=[], adversarial_findings=[]
        ),
    )
    for name in (
        "stream_rows.json",
        "released_feedback.json",
        "retention_rows.json",
        "coefficient_transitions.json",
    ):
        atomic_json(tmp_path / name, {})
    output = tmp_path / "out"
    output.write_text("phase=cold_replay phase=retention_boundary")
    err = tmp_path / "err"
    err.write_text("")
    prior = tmp_path / "prior.json"
    atomic_json(prior, dict(input_references=[{}, {}, {}, p.ref(fixture)]))
    primary = tmp_path / "results/experiment_8750_v750_deadline_observer_control.json"
    atomic_json(
        primary,
        dict(
            primitive_reference=p.ref(prior),
            verdict_class="disqualified",
            observer_comparison_rows=[
                dict(
                    arm="historical",
                    stage_reference=p.ref(output),
                    receipt=dict(
                        stdout_path=str(output),
                        stdout_sha256=p.ref(output)["sha256"],
                        stderr_path=str(err),
                        stderr_sha256=p.ref(err)["sha256"],
                        parent_samples=p.ref(output),
                    ),
                )
            ],
        ),
    )
    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text("frozen")
    monkeypatch.setattr(e, "ROOT", tmp_path)
    monkeypatch.setattr(e.core, "source_closure", lambda *a: [p.ref(source)])

    def collection(name, argv, directory, budget):
        out = tmp_path / "collection.stdout"
        out.write_text(
            "tests/python/test_publication_cost_8764.py::test_prepare_freezes_private_computational_scope\n"
        )
        return dict(name=name, passed=True, stdout_path=str(out))

    monkeypatch.setattr(p, "supervise", collection)
    raw = tmp_path / "raw"
    raw.mkdir()
    work = e.empty(raw, "20261103")
    plan = e.prepare(work)
    frozen = json.loads(plan.read_bytes())
    assert frozen["correction_selected"] is None and frozen["cache_enabled"] is False
    assert len(frozen["nodes"]) == 1
    assert not work["historical_evidence"]["late_output_eligible"]


def test_duplicate_identity_and_detached_descendant_cleanup(tmp_path):
    """REQ-VERIFY-8764: duplicate identities deduplicate; detached owned children cannot leak."""
    path = tmp_path / "same.json"
    atomic_json(path, dict(value=1))
    assert p.closure([p.ref(path), p.ref(path)]) == [p.ref(path)]
    pidfile = tmp_path / "descendant.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True); "
        f"Path({str(pidfile)!r}).write_text(str(child.pid)); time.sleep(20)"
    )
    row = p.supervise("detached", [sys.executable, "-c", code], tmp_path / "detached", 0.4)
    assert row["timed_out"] and row["cleanup_passed"]
    pid = int(pidfile.read_text())
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_cleanup_race_and_false_success(tmp_path, monkeypatch):
    """REQ-VERIFY-8764: an already killed descendant still permits reaping, never readiness."""
    kill = p.os.kill

    def raced(pid, sig):
        kill(pid, sig)
        raise ProcessLookupError

    monkeypatch.setattr(p.os, "kill", raced)
    row = p.supervise(
        "race", [sys.executable, "-c", "import time;time.sleep(20)"], tmp_path / "race", 0.05
    )
    assert row["timed_out"] and row["cleanup_passed"]
    work = e.empty(tmp_path, "20261103")
    work["checks"] = dict(unit=True)
    work["deadline_ready"] = True
    work["correction_source_refs"] = [dict(cached_success=True)]
    e.seal(work)
    assert e.build(work)["publication_path_ready_score"] == 0
    work["duration_s"] = 1
    assert not e.accepted(work)


def test_expected_warning_exit_is_normal(tmp_path):
    """REQ-VERIFY-8764: an expected scanner warning exit is not a killed or failed child."""
    row = p.supervise(
        "warning", [sys.executable, "-c", "raise SystemExit(1)"], tmp_path, 10, expected=1
    )
    assert row["exit_code"] == 1 and row["normal_exit"] and row["passed"]


def test_original_warning_findings_still_require_exact_match(tmp_path, monkeypatch):
    """REQ-REPORT-8764: scanner warnings preserve original assertions and permit cold replay."""
    plan = tmp_path / "plan.json"
    warning = dict(kind="TAUTOLOGY", severity="warn", detail="same original warning")
    atomic_json(
        plan, dict(scratch=str(tmp_path), expected_scan_exit=1, expected_scan_findings=[warning])
    )

    def supervised(name, argv, raw, budget, **options):
        raw.mkdir(parents=True, exist_ok=True)
        stdout = raw / (name + ".stdout")
        atomic_json(stdout, dict(reports=[dict(flags=[warning])]))
        return dict(
            name=name,
            passed=True,
            normal_exit=True,
            cleanup_passed=True,
            pid=1,
            stdout_path=str(stdout),
            duration_s=0.1,
            timed_out=False,
        )

    monkeypatch.setattr(p, "supervise", supervised)
    work = e.empty(tmp_path / "raw", "20261103")
    Path(work["raw"]).mkdir()
    e.diagnostic(work, plan)
    assert len(work["stage_rows"]) == 6 and all(row["passed"] for row in work["stage_rows"])
    atomic_json(plan, dict(scratch=str(tmp_path), expected_scan_exit=1, expected_scan_findings=[]))
    work = e.empty(tmp_path / "mismatch", "20261103")
    Path(work["raw"]).mkdir()
    e.diagnostic(work, plan)
    assert len(work["stage_rows"]) == 4 and work["stage_rows"][-1]["passed"] is False
