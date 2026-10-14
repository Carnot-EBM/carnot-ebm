"""REQ-VERIFY-8522: independent operands qualify exact terminal bytes."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from carnot.reporting import arc_effect_qualification_8522 as e
from carnot.reporting.current_work_receipt import atomic_json

import pytest

from carnot.reporting import arc_effect_runner_8522 as runner


def test_original_authentication_and_protocol(tmp_path):
    """SCENARIO-VERIFY-8522-REPLAY retains original dates and authority."""
    source = e.authenticate(e.ROOT, tmp_path)
    assert source["historical_date"] == "20261014"
    assert source["historical_redirect_count"] == 32
    assert source["historical_applied_missing_count"] == 16
    protocol = e.protocol()
    assert protocol["games"] == ["ar25-0c556536", "ft09-0d8bbf25", "lf52-271a04aa", "tn36-ef4dde99"]
    assert protocol["max_actions"] == 512 and protocol["episode_cap_s"] == 120
    assert protocol["refinement_min_resolved_per_shared_cell"] == 5
    assert protocol["MODEL_SPECS"] == []


def test_fresh_cli_rejections(tmp_path):
    """SCENARIO-VERIFY-8522-REPLAY checks actual bounded child exits."""
    for arguments, expected in [
        (["--date", "20261013"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", str(tmp_path / "missing")], 1),
    ]:
        child = subprocess.run(
            [sys.executable, str(e.ROOT / e.CLI), *arguments],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == expected, child.stdout + child.stderr


def test_reduction_and_tamper(tmp_path):
    """REQ-VERIFY-8522 rejects rehashed summaries and missing primitive operands."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "scratch")
    path = tmp_path / "work.json"
    atomic_json(path, work)
    output = tmp_path / "results" / (e.NAME + ".json")
    value = e.build(path, [], output)
    assert value["arc_effect_receipts_ready_score"] == 0
    assert e.replay(value)
    changed = deepcopy(value)
    changed["headline_solve_credit"] = 1
    assert not e.replay(changed)
    changed = deepcopy(value)
    changed["work_reference"]["path"] = str(tmp_path / "absent")
    assert not e.replay(changed)
    work["fixtures"]["mutation_fixture_rows"][0]["receipts"][0]["applied"] = True
    atomic_json(path, work)
    assert not e.replay(e.build(path, [], output))


def test_independent_rejection_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8522: all evidence operands reject independently after outer rehashing."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "scratch")
    path = tmp_path / "work.json"
    output = tmp_path / "results" / (e.NAME + ".json")
    monkeypatch.setattr(e, "authenticate", lambda *_: work["source"])
    monkeypatch.setattr(e.fixtures, "replay", lambda *_: True)
    from carnot.reporting.current_work_receipt import sha256_file

    for mechanism in ["metadata", "primitive", "code", "log", "protocol", "source", "activation"]:
        changed = deepcopy(work)
        rows = []
        if mechanism == "metadata":
            changed["source"]["historical_redirect_count"] = 33
        elif mechanism == "primitive":
            changed["source"]["references"][0]["expected_sha256"] = "bad"
        elif mechanism == "code":
            changed["code_config_hashes"][str(e.ROOT / e.CLI)] = "bad"
        elif mechanism == "log":
            log = tmp_path / "log"
            log.write_text("changed")
            rows = [dict(name="private", passed=True, stdout_path=str(log), stdout_sha256="bad")]
        elif mechanism == "protocol":
            target = tmp_path / "changed_protocol.json"
            atomic_json(target, dict(changed=True))
            changed["protocol_reference"] = dict(path=str(target), sha256=sha256_file(target))
        elif mechanism == "source":
            changed["source"] = None
            changed["failures"] = []
        elif mechanism == "activation":
            import yaml

            target = tmp_path / "changed_activation.yaml"
            target.write_text(yaml.safe_dump(dict(tasks=[dict(id=e.TASK)])))
            changed["activation_reference"]["snapshot_path"] = str(target)
        atomic_json(path, changed)
        assert not e.replay(e.build(path, rows, output)), mechanism


@pytest.mark.parametrize("failed", [False, True])
def test_private_block_and_owned_failure(tmp_path, monkeypatch, failed):
    """REQ-VERIFY-8522: unchanged absence blocks; owned failures disqualify."""
    monkeypatch.setattr(
        runner, "validate_checks", lambda *_: [dict(name="private", passed=not failed)]
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if failed else "blocked")
    assert value["arc_effect_receipts_ready_score"] == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["passed"] and side["consumer"]["passed"]


def test_private_ready_consumer_and_fresh_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8522-REPLAY: the exact ready field passes both downstream consumers."""
    names = [
        "owned_tests",
        "coverage_report",
        "policy_added_coverage",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    ]
    monkeypatch.setattr(
        runner, "validate_checks", lambda *_: [dict(name=n, passed=True) for n in names]
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive"
    assert value["arc_effect_receipts_ready_score"] == 1
    assert value["live_episodes_measured"] == value["headline_solve_credit"] == 0
    result = subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), "--cold-replay", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["consumer"]["gates"][0]["artifact_field"] == "arc_effect_receipts_ready_score"


def test_validation_adapter_real_plan_and_coverage_boundary(tmp_path, monkeypatch):
    """REQ-VERIFY-8522: owned coverage includes added policy statements and original node shards."""
    calls = []

    def fake_child(name, argv, logs, **kwargs):
        calls.append((name, argv))
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (name + ".stdout")
        log.write_text(
            "tests/python/test_private.py::test_original\n" if name.startswith("collect") else ""
        )
        if name == "coverage_json":
            # The real run measures execution; this fixture only qualifies how
            # the adapter compares added statements against Coverage.py output.
            source = (e.ROOT / e.POLICY).read_text()
            atomic_json(
                logs / "owned_coverage.json",
                {
                    "files": {
                        e.POLICY: {"executed_lines": list(range(1, len(source.splitlines()) + 1))}
                    }
                },
            )
        return dict(name=name, passed=True, stdout_path=str(log))

    monkeypatch.setattr(runner, "child", fake_child)
    rows = runner.validate_checks(tmp_path / "raw", tmp_path / "private")
    assert rows[-1]["name"] == "policy_added_coverage" and rows[-1]["passed"]
    assert any(
        "--include=" in arg for name, argv in calls if name == "coverage_report" for arg in argv
    )
    monkeypatch.setattr(runner.old, "validate_checks", lambda *_: [])
    assert runner.validate_checks(tmp_path / "no_coverage", tmp_path / "other") == []
    monkeypatch.setattr(runner, "child", lambda *_a, **_k: dict(passed=False))
    assert all(
        not r["passed"]
        for r in runner.validate_checks(tmp_path / "fail", tmp_path / "fail_private")
    )


def test_missing_hash_and_history_qualification_refusals(tmp_path, monkeypatch):
    """REQ-VERIFY-8522: original validation and authority are independently required."""
    with pytest.raises(OSError):
        e.authenticate(tmp_path, tmp_path / "raw")
    bad = tmp_path / "bad.json"
    atomic_json(bad, {"changed": True})
    with pytest.raises(ValueError, match="input_hash"):
        e.checked(bad, "sha256:invalid")
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: {"report": {"passed": False}})
    with pytest.raises(ValueError, match="historical_validation"):
        e.authenticate(e.ROOT, tmp_path / "raw")
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: {"report": {"passed": True}})
    original = e.checked

    def changed(path, pin):
        value = original(path, pin)
        if "preconditions_checked" in value:
            value["preconditions_checked"]["task_sha256"] = "changed"
        return value

    monkeypatch.setattr(e, "checked", changed)
    with pytest.raises(ValueError, match="historical_authority"):
        e.authenticate(e.ROOT, tmp_path / "raw")


def test_historical_primitive_and_policy_refusal(tmp_path, monkeypatch):
    """REQ-VERIFY-8522: stored policy bytes and each primitive need original producer hashes."""
    value = e.checked(e.ROOT / e.UPSTREAM, e.UPSTREAM_PIN)
    monkeypatch.setattr(e.historical, "reduce", lambda *_: dict(rows=[]))
    with pytest.raises(ValueError, match="historical_primitives"):
        e.authenticate(e.ROOT, tmp_path / "raw")
    monkeypatch.setattr(e.historical, "reduce", lambda *_: dict(rows=value["rows"]))
    original = e.sha256_file
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "bad" if p.name == "arc_competition_agent.py" else original(p)
    )
    with pytest.raises(ValueError, match="historical_policy_pin"):
        e.authenticate(e.ROOT, tmp_path / "raw")
    work = e.checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
    operand = Path(work["source"]["operands"][0]["terminal"]["snapshot_path"])
    monkeypatch.setattr(e, "sha256_file", lambda p: "bad" if p == operand else original(p))
    with pytest.raises(ValueError, match="historical_primitive_hash"):
        e.authenticate(e.ROOT, tmp_path / "raw")


def test_deadline_and_malformed_replay(tmp_path, monkeypatch):
    """REQ-VERIFY-8522: measurement ends before its validation reserve and bad JSON fails closed."""
    from types import SimpleNamespace

    monkeypatch.setattr(e, "time", SimpleNamespace(monotonic=iter([0, 3601]).__next__))
    monkeypatch.setattr(e.fixtures, "collect", lambda *_: {})
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path / "private")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert runner.main(["--cold-replay", str(bad)]) == 1
