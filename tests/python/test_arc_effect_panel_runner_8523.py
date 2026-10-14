"""REQ-VERIFY-8523: private CLI, child validation and exact byte publication."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import arc_effect_panel_8523 as e
from carnot.reporting import arc_effect_panel_runner_8523 as runner
from carnot.reporting.current_work_receipt import atomic_json

from test_arc_effect_panel_8523 import primitive


def names():
    return [
        dict(name=n, passed=True)
        for n in [
            "owned_tests",
            "coverage_report",
            "ruff_check",
            "ruff_format",
            "strict_mypy",
            "spec_coverage",
        ]
    ]


def test_cli_dates_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8523-REPLAY crosses the real standalone child boundary."""
    for args, expected in [
        (["--date", "20261013"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", str(tmp_path / "missing")], 1),
    ]:
        result = subprocess.run(
            [sys.executable, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stdout + result.stderr
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert runner.main(["--cold-replay", str(bad)]) == 1
    assert runner.main(["--deliberate-error"]) == 7


@pytest.mark.parametrize("failed", [False, True])
def test_private_block_owned_failure_and_consumer(tmp_path, monkeypatch, failed):
    """REQ-VERIFY-8523: actual validators and readers see private terminal bytes."""
    monkeypatch.setattr(
        runner, "validate_checks", lambda *_: [dict(name="private", passed=not failed)]
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if failed else "blocked")
    assert value["arc_effect_panel_ready_score"] == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["passed"] and side["consumer"]["passed"]
    assert side["consumer"]["gates"][0]["artifact_field"] == "arc_effect_panel_ready_score"


def test_validation_adapter_union_and_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8523 freezes collected original nodes and preserves coverage checks."""
    calls = []

    def fake(name, argv, logs, **kwargs):
        calls.append((name, argv))
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (name + ".stdout")
        log.write_text(
            "tests/python/test_private.py::test_original\n" if name.startswith("collect") else ""
        )
        return dict(name=name, passed=True, stdout_path=str(log))

    monkeypatch.setattr(runner, "child", fake)
    rows = runner.validate_checks(tmp_path / "raw", tmp_path / "private")
    assert rows and all(r["passed"] for r in rows)
    assert all(
        "--no-cov" in argv and "0" in argv for name, argv in calls if name.startswith("historical")
    )
    assert any(name == "coverage_report" and "--fail-under=100" in argv for name, argv in calls)
    monkeypatch.setattr(runner, "child", lambda *_a, **_k: dict(passed=False))
    assert not any(
        r["passed"] for r in runner.validate_checks(tmp_path / "failed", tmp_path / "scratch")
    )


def test_measure_bounded_episode_adapter_and_deadline(tmp_path, monkeypatch):
    """REQ-REPORT-8523 keeps all fixed slots; the aggregate deadline cannot be extended."""
    monkeypatch.setattr(e, "authenticate", lambda *_: dict(protocol=e.qualification.protocol()))
    monkeypatch.setattr(e.shipped, "preconditions", lambda *_: ([], {}, {}))
    monkeypatch.setattr(
        e.panel_runtime,
        "sdk",
        lambda: SimpleNamespace(
            available_environments=[
                SimpleNamespace(game_id=g) for g in e.qualification.protocol()["games"]
            ]
        ),
    )
    calls = []

    def execute(name, argv, logs, **kwargs):
        calls.append((name, kwargs))
        unit = json.loads(Path(argv[-1]).read_bytes())
        directory = Path(unit["directory"])
        atomic_json(directory / "episode.json", dict(unit=unit))
        (directory / "steps.jsonl").write_text("")
        return dict(name=name, passed=True)

    monkeypatch.setattr(e, "child", execute)
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "private")
    assert len(work["references"]) == 16
    assert all(k["deadline"] <= 130 for _, k in calls)
    monkeypatch.setattr(e, "time", SimpleNamespace(monotonic=iter([0, 2390, 2391]).__next__))
    work = e.measure(e.ROOT, tmp_path / "stopped", tmp_path / "other")
    assert work["references"] == []


def test_episode_actual_adapter_flags(tmp_path, monkeypatch):
    """REQ-REPORT-8523 sets receipt and induction flags before calling shipped runtime."""
    import os

    path = tmp_path / "unit.json"
    atomic_json(path, dict(directory=str(tmp_path), episode_id="private"))
    monkeypatch.setattr(e.panel_runtime, "sdk", lambda: "sdk")

    def execute(unit, sdk, raw):
        assert os.environ["CARNOT_ARC_EFFECT_RECEIPTS"] == "1"
        assert os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "1"
        assert sdk == "sdk" and raw == tmp_path
        return dict(model_tripwire_attempts=[])

    monkeypatch.setattr(e.runtime, "episode", execute)
    assert e.episode(path) == 0
    assert runner.main(["--episode", str(path)]) == 0


def test_replay_independent_rejection_paths(tmp_path):
    """REQ-VERIFY-8523 rejects every independently pinned envelope after outer rehashing."""
    work = e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path / "private")
    path = tmp_path / "work.json"
    output = tmp_path / "results" / (e.NAME + ".json")
    for case in [
        "clean",
        "frozen",
        "protocol",
        "authority_hash",
        "task",
        "code",
        "source",
        "source_missing",
        "log",
        "aggregate",
    ]:
        changed = deepcopy(work)
        receipts = []
        if case == "frozen":
            changed["panel"]["protocol"]["max_actions"] = 511
        elif case == "protocol":
            changed["panel"]["protocol"]["max_actions"] = 511
            target = tmp_path / "frozen.json"
            atomic_json(target, changed["panel"])
            changed["frozen_reference"] = dict(path=str(target), sha256=e.sha256_file(target))
        elif case in ["authority_hash", "task"]:
            target = tmp_path / "active.yaml"
            target.write_text("tasks: []\n")
            changed["activation_reference"] = dict(
                exists=True,
                snapshot_path=str(target),
                sha256="bad" if case == "authority_hash" else e.sha256_file(target),
            )
            if case == "task":
                changed["failures"] = []
        elif case == "code":
            changed["code_config_hashes"][str(e.ROOT / e.CLI)] = "bad"
        elif case == "source":
            changed["source"] = dict(primary={}, reference=dict(snapshot_path=str(path)))
        elif case == "source_missing":
            changed["failures"] = []
        elif case == "log":
            target = tmp_path / "log"
            target.write_text("changed")
            receipts = [
                dict(name="private", passed=True, stdout_path=str(target), stdout_sha256="bad")
            ]
        atomic_json(path, changed)
        value = e.build(path, receipts, output)
        if case == "aggregate":
            value["headline_solve_credit"] = 1
        assert e.replay(value) == (case == "clean"), case


def test_upstream_independent_field_and_bound_refusals(tmp_path, monkeypatch):
    """REQ-VERIFY-8523: readiness alone cannot replace producer identity and bound bytes."""
    original = e.checked
    primary = original(e.ROOT / e.UPSTREAM, e.UPSTREAM_PIN)
    monkeypatch.setattr(e.qualification, "replay", lambda *_: True)
    monkeypatch.setattr(
        e,
        "checked",
        lambda p, h: (
            dict(primary, arc_effect_receipts_ready_score=0)
            if p == e.ROOT / e.UPSTREAM
            else original(p, h)
        ),
    )
    with pytest.raises(ValueError, match="upstream_field"):
        e.authenticate(e.ROOT, tmp_path)
    monkeypatch.setattr(e, "checked", original)
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
    with pytest.raises(ValueError, match="upstream_bound_consumer"):
        e.authenticate(e.ROOT, tmp_path)
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=True)))
    monkeypatch.setattr(e.qualification, "replay", lambda *_: False)
    with pytest.raises(ValueError, match="upstream_protocol_or_replay"):
        e.authenticate(e.ROOT, tmp_path)


def test_missing_sdk_and_malformed_upstream(tmp_path, monkeypatch):
    """REQ-REPORT-8523: installed-game absence blocks rather than altering the frozen roster."""
    monkeypatch.setattr(e.shipped, "preconditions", lambda *_: ([], {}, {}))
    monkeypatch.setattr(e, "authenticate", lambda *_: dict(protocol=e.qualification.protocol()))
    monkeypatch.setattr(
        e.panel_runtime, "sdk", lambda: (_ for _ in ()).throw(ImportError("missing SDK"))
    )
    work = e.measure(e.ROOT, tmp_path / "no_sdk", tmp_path / "scratch")
    assert work["references"] == [] and len(work["failures"]) == 5
    monkeypatch.setattr(
        e, "authenticate", lambda *_: (_ for _ in ()).throw(ValueError("unqualified"))
    )
    target = tmp_path / e.UPSTREAM
    target.parent.mkdir(parents=True)
    target.write_text("{")
    work = e.measure(tmp_path, tmp_path / "malformed", tmp_path / "other")
    assert work["failures"][0]["observed"] is None


def test_build_live_qualification_and_primitive_disqualification(tmp_path, monkeypatch):
    """REQ-VERIFY-8523: collection readiness is separate from benefit and primitive rejection."""
    work = e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path / "scratch")
    panel = work["panel"]["units"]
    work["references"] = [
        primitive(tmp_path / "episodes", u, mode=u["mode"], changed=u["mode"] == "applied")
        for u in panel
    ]
    monkeypatch.setattr(e.panel_runtime, "reduce_episode", lambda *_: {})
    for ref in work["references"]:
        path = Path(ref["directory"]) / "episode.json"
        terminal = json.loads(path.read_bytes())
        terminal["solve_provenance"] = "live_agent_self_discovery"
        atomic_json(path, terminal)
        ref["terminal_sha256"] = e.sha256_file(path)
    work["failures"] = []
    path = tmp_path / "work.json"
    atomic_json(path, work)
    receipts = [
        *names(),
        *[dict(name=n, passed=True) for n in ["missing", "intentional", "rehashed_tamper"]],
    ]
    output = tmp_path / "results" / (e.NAME + ".json")
    value = e.build(path, receipts, output)
    assert value["arc_effect_panel_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["independent_generalization_score"] == 0
    monkeypatch.setattr(runner, "child", lambda name, *_a, **_k: dict(name=name, passed=True))
    assert all(r["passed"] for r in runner.controls(value, tmp_path / "controls"))
    work["references"][0]["terminal_sha256"] = "invalid"
    atomic_json(path, work)
    value = e.build(path, receipts, output)
    assert value["verdict_class"] == "disqualified" and value["arc_effect_panel_ready_score"] == 0
    assert value["gate_check_summary"][0]["check"] == "primitive_replay"


def test_source_and_code_drift_after_valid_replay(tmp_path):
    """REQ-VERIFY-8523: independent source equality and executable hashes fail closed."""
    work = e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path / "scratch")
    path = tmp_path / "work.json"
    output = tmp_path / "results" / (e.NAME + ".json")
    atomic_json(path, work)
    assert e.replay(e.build(path, [], output))
    changed = deepcopy(work)
    changed["code_config_hashes"] = dict(
        changed["code_config_hashes"], **{str(e.ROOT / e.CLI): "bad"}
    )
    atomic_json(path, changed)
    assert not e.replay(e.build(path, [], output))
    changed = deepcopy(work)
    changed["source"] = dict(reference=dict(snapshot_path=str(e.ROOT / e.UPSTREAM)), primary={})
    atomic_json(path, changed)
    assert not e.replay(e.build(path, [], output))
