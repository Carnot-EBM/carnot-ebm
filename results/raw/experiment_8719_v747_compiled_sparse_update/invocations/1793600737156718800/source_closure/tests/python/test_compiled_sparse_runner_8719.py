"""REQ-VERIFY-8719: real private CLI, fresh replay and rejection paths."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from unittest.mock import patch

import pytest

from carnot.reporting import compiled_sparse_study_8719 as e
from carnot.reporting import compiled_sparse_runner_8719 as r
from carnot.verify import compiled_sparse_8719 as n


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """A genuine private producer retains raw child evidence for rejection controls."""
    folder = tmp_path_factory.mktemp("native8719")
    output = folder / (e.NAME + ".json")
    assert r.run(output, "20261102", True, Path(n.DEFAULT_EXTENSION)) == 0
    return output


def test_private_publication_and_replay(private):
    """SCENARIO-VERIFY-8719-CUSTODY: success crosses native processes and readers."""
    value = json.loads(private.read_bytes())
    assert value["completed_count"] == 2
    assert value["native_update_ready_score"] == 0
    assert value["restart_rows"][0]["real_exit"] == 73
    r.replay(private)
    command = [sys.executable, str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--cold-replay", str(private)], 0),
        (["--date", "bad"], 1),
        (["--worker", str(private)], 1),
        (["--help"], 0),
        (["--private-control", "--output", str(e.ROOT / "results" / (e.NAME + ".json"))], 1),
    ]:
        child = subprocess.run(command + args, capture_output=True, timeout=120)
        assert child.returncode == expected, child.stderr.decode()


def test_rehashed_and_missing_rejections(private, tmp_path):
    """SCENARIO-VERIFY-8719-CUSTODY: matching self-reported hashes cannot grant trust."""
    original = json.loads(private.read_bytes())
    changed = deepcopy(original)
    changed["native_parity_rows"][0]["probability"] += 0.01
    target = tmp_path / "changed.json"
    e.atomic_json(target, changed)
    with pytest.raises(ValueError, match="report_drift"):
        r.replay(target)
    changed = deepcopy(original)
    changed["primitive_reference"]["sha256"] = "sha256:" + "0" * 64
    e.atomic_json(target, changed)
    with pytest.raises(ValueError, match="independent_measurement_seal"):
        r.replay(target)
    work = e.read(original["primitive_reference"])
    ref = work["raw_refs"][0]
    path = Path(ref["path"])
    saved = path.read_bytes()
    path.unlink()
    try:
        with pytest.raises(OSError):
            r.replay(private)
    finally:
        path.write_bytes(saved)


def test_host_and_entrypoint_errors(monkeypatch, tmp_path):
    """REQ-VERIFY-8719: malformed imports and CLI arguments fail visibly."""
    with patch.object(n.importlib.util, "spec_from_file_location", return_value=None):
        with pytest.raises(ValueError, match="extension_spec"):
            n.load(tmp_path / "bad.so")

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            pass

    spec = importlib.util.spec_from_loader("empty", Loader())
    with patch.object(n.importlib.util, "spec_from_file_location", return_value=spec):
        with pytest.raises(ValueError, match="native_entrypoint"):
            n.load(tmp_path / "empty.so")
    assert r.main(["--date", "20260230"]) == 1
    assert r.main(["--date", "bad"]) == 1
    stop = Event()
    stop.set()
    r.heartbeat(stop)

    class ShortWait:
        def __init__(self):
            self.calls = 0

        def wait(self, seconds):
            self.calls += 1
            return self.calls > 1

    r.heartbeat(ShortWait())


def test_block_and_disqualification(private, tmp_path, monkeypatch):
    """REQ-REPORT-8719: failed owned checks cannot produce native readiness."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    work["private_control"] = False
    blocked = deepcopy(work)
    blocked.update(summaries=[], restart_rows=[], failures=[dict(reason="missing")])
    assert (
        e.build(blocked, [dict(passed=True)], tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "blocked"
    )
    assert e.build(work, [dict(passed=False)], private)["verdict_class"] == "disqualified"
    broken = deepcopy(work)
    broken["summaries"][0]["passed"] = False
    assert e.build(broken, [dict(passed=True)], private)["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        e.previous,
        "authenticate",
        lambda: dict(heads=[], frozen={}, refs=[], failures=[], historical_model_provenance={}),
    )
    monkeypatch.setattr(e, "PIN", "sha256:" + "0" * 64)
    assert e.inputs()["failures"]


def test_owned_rust_checks(tmp_path):
    """REQ-VERIFY-8719: the owned module passes real cargo tests, format and clippy."""
    receipts = r.native_build(tmp_path / "rust")
    assert all(row["passed"] for row in receipts), [
        Path(row["stderr_path"]).read_text() for row in receipts if not row["passed"]
    ]


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-VERIFY-8719: genuine subprocesses flush coverage on committed hard exits."""
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\npatch=subprocess\n")

    def invoke(raw):
        return [
            r.validation.child(
                "adapter", [sys.executable, "-c", "pass", "--rcfile=" + str(config)], raw
            )
        ]

    monkeypatch.setattr(r.validation, "validate", invoke)
    assert r.validate(tmp_path / "logs")[0]["passed"]
    assert "patch=subprocess,_exit" in config.read_text()


def test_preconditions_and_worker_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8719: resource absence and killed workers never grant readiness."""
    real_resources = r.resources
    with monkeypatch.context() as m:
        m.setattr(r, "resources", lambda path: dict(real_resources(path), memory_available_bytes=0))
        work = r.measure(tmp_path / "blocked", "20261102", True, Path(n.DEFAULT_EXTENSION))
        assert work["failures"] and not work["summaries"]
    real_child = r.child
    for failure in ["full_0", "exit_0"]:

        def fail(name, *args, **kwargs):
            return dict(passed=False) if name == failure else real_child(name, *args, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(r, "child", fail)
            with pytest.raises(ValueError, match="failed_or_killed"):
                r.measure(tmp_path / failure, "20261102", True, Path(n.DEFAULT_EXTENSION))


def test_cold_seal_guards(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8719-CUSTODY: every independent seal and recomputation is required."""
    value = json.loads(private.read_bytes())
    work = e.read(value["primitive_reference"])
    real_read = e.read
    for mode, message in [
        ("source", "independent_source_seal"),
        ("clock", "clock_drift"),
        ("validation", "independent_validation_seal"),
        ("native", "native_replay_drift"),
        ("receipts", "report_drift"),
    ]:
        changed = deepcopy(work)
        if mode == "source":
            changed["code_refs"] = []
        if mode == "clock":
            changed["duration_s"] += 1

        def altered(ref):
            if ref == value["primitive_reference"]:
                return changed
            result = real_read(ref)
            if mode == "validation" and "/validation_seals/" in ref["path"]:
                return dict(validation={})
            if mode == "receipts" and ref == value["validation_reference"]:
                result["receipts"][0].update(
                    references=[e.reference(private)],
                    stdout_path=str(private),
                    stdout_sha256=e.reference(private)["sha256"],
                )
            return result

        with monkeypatch.context() as m:
            m.setattr(e, "read", altered)
            if mode == "native":
                real_drive = n.drive

                def drift(*args):
                    result = real_drive(*args)
                    result["checkpoint"] = {}
                    return result

                m.setattr(n, "drive", drift)
            with pytest.raises(ValueError, match=message):
                r.replay(private)
    changed_value = deepcopy(value)
    changed_work = deepcopy(work)
    changed_work["summaries"][0]["passed"] = False
    operand = tmp_path / "rehashed-work.json"
    e.atomic_json(operand, changed_work)
    changed_value["primitive_reference"] = e.reference(operand)
    candidate = tmp_path / "rehashed.json"
    e.atomic_json(candidate, changed_value)
    with pytest.raises(ValueError, match="independent_measurement_seal"):
        r.replay(candidate)


def test_full_cli_success(tmp_path):
    """REQ-VERIFY-8719: success resolves imports outside the checkout without PYTHONPATH."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (e.NAME + ".json")
    command = [
        sys.executable,
        str(e.ROOT / e.CLI),
        "--date",
        "20261102",
        "--private-control",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(output.read_bytes())["completed_count"] == 2


def test_missing_extension_is_terminal_blocked(tmp_path):
    """REQ-VERIFY-8719: external absence produces a complete blocked artifact."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, "20261102", True, tmp_path / "missing.so") == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["native_update_ready_score"] == 0
    assert value["gate_check_summary"][-1]["operand_state"] == "missing_file"
    r.replay(output)
