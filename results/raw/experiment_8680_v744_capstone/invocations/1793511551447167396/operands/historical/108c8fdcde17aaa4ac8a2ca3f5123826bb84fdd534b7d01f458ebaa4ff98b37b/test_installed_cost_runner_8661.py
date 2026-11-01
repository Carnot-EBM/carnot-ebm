"""REQ-VERIFY-8661: private publication and bounded actual child controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from carnot.reporting import installed_cost_8661 as c
from carnot.reporting import installed_cost_runner_8661 as r
from carnot.reporting import installed_cost_session_8661 as s
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_actual_private_session_and_rejection_paths(tmp_path, capsys):
    """SCENARIO-VERIFY-8661-REPLAY: real children preserve typed outputs and clean exit."""
    bundle = c.original_inputs(c.ROOT)
    config = tmp_path / "head.json"
    atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))
    slot = c.roster()[0]
    command = [sys.executable, "-u", str(c.ROOT / "scripts/experiments/resident_worker_8608.py")]
    primitive = s.session(
        command, config, slot, bundle["requests"], tmp_path, tmp_path / "good", heartbeat_s=0
    )
    assert "8661_session_heartbeat" in capsys.readouterr().out
    summary, rows, spans = c.summarize(primitive, bundle)
    assert summary["completed_count"] == 64 and summary["exact_decisions"]
    assert len(rows) == 64
    assert (
        sum(x["end_ns"] - x["start_ns"] for x in spans)
        == primitive["end_ns"] - primitive["start_ns"]
    )
    for key, changed, message in [
        ("exit_code", 9, "session_configuration"),
        ("ready", {}, "status"),
        ("observations", [], "request_count"),
        ("end_ns", 0, "clock_partition"),
    ]:
        bad = deepcopy(primitive)
        bad[key] = changed
        with pytest.raises((ValueError, KeyError), match=message):
            c.summarize(bad, bundle)
    for mutation, message in [
        ("frame", "frame_operands"),
        ("probability", "typed_output_parity"),
        ("decision", "typed_output_parity"),
    ]:
        bad = deepcopy(primitive)
        obs = bad["observations"][0]
        if mutation == "frame":
            obs["request"]["row"]["q"] = 0.999
        else:
            obs["response"]["response"][mutation] = 0.999 if mutation == "probability" else "wrong"
        with pytest.raises(ValueError, match=message):
            c.summarize(bad, bundle)
    with pytest.raises(TimeoutError, match="before_launch"):
        s.session(command, config, slot, [], tmp_path, tmp_path / "late", deadline_ns=0)
    with pytest.raises(ValueError, match="worker_ready"):
        s.session([sys.executable, "-c", "pass"], config, slot, [], tmp_path, tmp_path / "exit")
    with pytest.raises(ValueError, match="worker_response"):
        s.session(
            command,
            config,
            slot,
            [dict(bundle["requests"][0], q=None)],
            tmp_path,
            tmp_path / "bad-request",
        )
    assert json.loads((tmp_path / "bad-request/failed_child.json").read_text())["exit_code"] < 0


def test_private_publication_replay_and_cli(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8661-REPLAY: real terminal CLIs reject rehashed and missing input."""
    from carnot.reporting import installed_cost_validation_8661 as v

    monkeypatch.setattr(
        v, "validate", lambda work, raw: [dict(name=n, passed=True) for n in r.REQUIRED]
    )
    output = tmp_path / (c.NAME + ".json")
    assert r.run(output, "20261019", tmp_path / "missing") == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    authority = value["authority_reference"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(c.ROOT / c.CLI)]
    args = ["--authority", authority["path"], authority["sha256"]]
    for extra, code in [
        (["--cold-replay", str(output), *args], 0),
        (["--cold-replay", str(tmp_path / "missing"), *args], 1),
        (["--cold-replay", str(output)], 1),
        (["--date", "bad"], 1),
        (["--private-output", str(c.ROOT / "unsafe")], 1),
        (["--input-root", str(tmp_path)], 1),
    ]:
        result = subprocess.run(
            argv + extra, cwd=tmp_path, env=env, capture_output=True, timeout=60
        )
        assert result.returncode == code, result.stderr
    monkeypatch.setattr(r, "run", lambda *a: 0)
    assert r.main(["--date", "20261019", "--private-output", str(output)]) == 0


def test_owned_failure_disqualifies_even_when_upstream_missing(tmp_path):
    """REQ-VERIFY-8661: upstream incompleteness cannot hide failed owned checks."""
    work = c.empty_work(tmp_path, "20261019")
    work["gate_check_summary"] = [dict(observed_state="missing_file")]
    work["validation_receipts"] = [dict(name="owned_tests", passed=False)]
    path = tmp_path / "authority.json"
    atomic_json(path, work)
    value = c.artifact(c.reference(path))
    assert value["verdict_class"] == "disqualified" and value["installed_cost_ready_score"] == 0


def test_fresh_install_measurement_deadline_and_failed_cell(tmp_path, monkeypatch):
    """REQ-REPORT-8661: one genuine cell survives later deadline and owned failure."""
    real_session = s.session
    calls = []

    def bounded(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError("fixed measurement deadline")
        return real_session(*args, **kwargs)

    monkeypatch.setattr(s, "session", bounded)
    work = r.measure(tmp_path / "measured", "20261019", c.ROOT)
    assert not work["owned_error"] and not work["gate_check_summary"], work
    assert work["rows"][0]["status"] == "completed"
    assert work["rows"][1]["status"] == "censored"
    assert c.reduce(work)["completed_count"] == 1
    prepared = dict(commands={"private": []}, modules=[])
    config = tmp_path / "unused.json"
    atomic_json(config, {})
    monkeypatch.setattr(r, "prepare", lambda *a: (prepared, config, []))

    def failed(*a, **k):
        raise ValueError("real failed response control")

    monkeypatch.setattr(s, "session", failed)
    failed_work = r.measure(tmp_path / "failed", "20261019", c.ROOT)
    assert failed_work["rows"][0]["status"] == "failed"
    assert failed_work["owned_error"] == "real failed response control"
    for error, state in [
        (FileNotFoundError("missing operand"), "missing_file"),
        (KeyError("missing field"), "missing_field"),
    ]:

        def missing(_):
            raise error

        monkeypatch.setattr(c, "admit", missing)
        blocked = r.measure(tmp_path / state, "20261019", c.ROOT)
        assert blocked["gate_check_summary"][0]["observed_state"] == state
    original_child = r.child

    def memory_control(name, argv, raw, **kwargs):
        receipt = original_child(name, argv, raw, **kwargs)
        if name == "scratch_filesystem":
            Path(receipt["stdout_path"]).write_text("tmpfs")
        return receipt

    monkeypatch.setattr(r, "child", memory_control)
    assert r.measure(tmp_path / "tmpfs", "20261019", c.ROOT)["owned_error"] == "cpu_scratch_memory"


def test_installer_rejects_failed_tools_and_loaded_identity(tmp_path, monkeypatch):
    """REQ-VERIFY-8661: changed native bytes or observed loaded head cannot pass installation."""
    bundle = c.original_inputs(c.ROOT)
    bundle["wheel_reference"] = json.loads(
        (c.ROOT / "results/experiment_8660_v743_installed_compatibility.json").read_bytes()
    )["wheel_reference"]
    extension = bundle["extension"]
    real_ref = c.reference
    control = {"mode": "install"}

    def child(name, argv, raw, **kwargs):
        raw.mkdir(parents=True, exist_ok=True)
        if name == "venv":
            control["site"] = Path(argv[-1]) / "lib/python3.12/site-packages"
            control["site"].mkdir(parents=True)
        out = raw / (name + ".stdout")
        if name.startswith("identity_"):
            site = control["site"]
            ext = site / "carnot_native" / Path(extension["path"]).name
            observed = dict(
                head_hash=canonical_hash(bundle["head"]),
                extension=real_ref(ext),
                modules=[real_ref(c.ROOT / "python/carnot/inference/resident_energy.py")],
                versions={},
            )
            if control["mode"] == "head":
                observed["head_hash"] = "wrong"
            if control["mode"] == "snapshot":
                observed["modules"][0]["sha256"] = "wrong"
            out.write_text(json.dumps(observed))
        return dict(
            name=name,
            passed=control["mode"] != "install"
            and not (control["mode"] == "probe" and name.startswith("identity_")),
            stdout_path=str(out),
        )

    monkeypatch.setattr(r, "child", child)
    for mode, message in [
        ("install", "installation_failed"),
        ("extension", "installed_extension_hash"),
        ("probe", "loaded_identity_probe"),
        ("head", "actual_loaded_head"),
        ("snapshot", "loaded_module_snapshot"),
    ]:
        control["mode"] = mode
        scratch = tmp_path / ("scratch-" + mode)
        raw = tmp_path / ("raw-" + mode)
        with monkeypatch.context() as m:
            if mode == "extension":
                m.setattr(
                    c,
                    "reference",
                    lambda p: (
                        dict(real_ref(p), sha256="wrong")
                        if str(p).startswith(str(scratch))
                        else real_ref(p)
                    ),
                )
            with pytest.raises(ValueError, match=message):
                r.prepare(raw, scratch, bundle)
        if mode in {"probe", "head", "snapshot"}:
            import shutil

            shutil.rmtree(scratch)


def test_clock_ready_shutdown_and_mid_session_deadline_controls(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8661-REPLAY: zero clocks, foreign ready and failed shutdown reject."""
    bundle = c.original_inputs(c.ROOT)
    config = tmp_path / "head.json"
    atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))
    command = [sys.executable, "-u", str(c.ROOT / "scripts/experiments/resident_worker_8608.py")]
    p = s.session(command, config, c.roster()[0], bundle["requests"], tmp_path, tmp_path / "base")
    bad = deepcopy(p)
    bad["ready"]["config_hash"] = "foreign"
    with pytest.raises(ValueError, match="loaded_head_or_extension"):
        c.summarize(bad, bundle)
    bad = deepcopy(p)
    bad["observations"][0]["response"]["response"]["boundary_fallback"] = True
    with pytest.raises(ValueError, match="typed_output_parity"):
        c.summarize(bad, bundle)
    bad = deepcopy(p)
    bad.update(start_ns=0, ready_ns=0, end_ns=0)
    for obs in bad["observations"]:
        obs.update(start_ns=0, end_ns=0)
        for span in obs["response"]["spans"]:
            span.update(start_ns=0, end_ns=0)
    with pytest.raises(ValueError, match="session_clock"):
        c.summarize(bad, bundle)
    original_clock = s.time.monotonic_ns
    values = iter([0, 0, 0, 100_000_000_000])
    with monkeypatch.context() as m:
        m.setattr(s.time, "monotonic_ns", lambda: next(values, 100_000_000_000))
        with pytest.raises(TimeoutError, match="reserved_shutdown"):
            s.session(
                command,
                config,
                c.roster()[0],
                bundle["requests"],
                tmp_path,
                tmp_path / "mid",
                deadline_ns=60_000_000_000,
            )
    assert s.time.monotonic_ns is original_clock
    real_read = s.wire.read_frame

    def failed_shutdown(fd, timeout):
        payload = real_read(fd, timeout)
        if payload and json.loads(payload).get("status") == "shutdown":
            return b'{"status":"error"}'
        return payload

    monkeypatch.setattr(s.wire, "read_frame", failed_shutdown)
    with pytest.raises(ValueError, match="unclean_shutdown"):
        s.session(command, config, c.roster()[0], [], tmp_path, tmp_path / "shutdown")


def test_validation_collects_once_preserves_sqlite_and_denies_missing_sources(
    tmp_path, monkeypatch
):
    """REQ-VERIFY-8661: partial source coverage and killed/missing children cannot qualify."""
    import coverage
    from carnot.reporting import installed_cost_validation_8661 as v

    mode = {"value": "partial"}
    calls = []

    def fake_child(name, argv, folder, **kwargs):
        calls.append(name)
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / (name + ".stdout")
        suites = c.TESTS + v.E2E
        if name == "collection":
            out.write_text(
                "\n".join(suite + "::test_fixture" for suite in suites)
                if mode["value"] != "empty"
                else ""
            )
        elif name.startswith("owned-") or name in {"e2e018", "e2e021"}:
            index = int(name[-1]) if name.startswith("owned-") else 2 if name == "e2e018" else 3
            source = suites[index]
            junit = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")))
            junit.write_text(
                f'<testsuite><testcase classname="{source[:-3].replace("/", ".")}" name="test_fixture"/></testsuite>'
            )
            data = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("COVERAGE_FILE=")))
            # A synthetic partial shard exercises the parser, never the readiness evidence.
            shard = coverage.CoverageData(basename=str(data) + ".synthetic-control")
            shard.add_lines({str(c.ROOT / c.OWNED[0]): {1}})
            shard.write()
            out.write_text("fixture command")
        else:
            out.write_text("fixture check")
        return dict(
            name=name,
            passed=mode["value"] != "killed" or name != "owned-0",
            exit_code=-9 if mode["value"] == "killed" and name == "owned-0" else 0,
            stdout_path=str(out),
        )

    monkeypatch.setattr(v, "child", fake_child)
    for value in ["partial", "killed", "empty"]:
        mode["value"] = value
        calls.clear()
        receipts = v.validate({}, tmp_path / value)
        assert calls.count("collection") == 1
        assert not next(x for x in receipts if x["name"] == "owned_coverage")["passed"]
        if value == "killed":
            assert not next(x for x in receipts if x["name"] == "exact_node_union")["passed"]
        if value == "empty":
            assert not any(x.startswith("owned-") for x in calls)
        else:
            assert next(x for x in receipts if x["name"] == "original_sqlite_unchanged")["passed"]
            manifests = list((tmp_path / value / "validation/shards").glob("*/manifest.json"))
            assert len(manifests) == 4
