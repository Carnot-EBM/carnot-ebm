"""REQ-REPORT-8646 and REQ-VERIFY-8646: original and current bytes stay separate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import installed_compatibility_8646 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_original_hash_failure_and_overlay(tmp_path):
    """SCENARIO-REPORT-8646-ISOLATION: actual children inspect both protocol bytes."""
    observed = e.historical(tmp_path)
    assert observed["failure_reproduced"]
    assert observed["overlay_passed"]
    assert observed["historical_source_reference"]["sha256"] == e.HASH
    assert observed["current_source_reference"]["sha256"] != e.HASH
    assert all(r["passed"] for r in observed["resolved_import_rows"])
    assert len(observed["original_nodes"]) == 122


def test_exact_disjoint_union():
    """SCENARIO-VERIFY-8646-CUSTODY: omitted, repeated and extra nodes fail."""
    assert e.union(["a", "b"], ["b", "a"])["passed"]
    for expected, actual in [([], []), (["a"], []), (["a"], ["a", "a"]), (["a"], ["b"])]:
        assert not e.union(expected, actual)["passed"]


def test_input_missing_is_distinct_from_zero(tmp_path):
    """SCENARIO-REPORT-8646-PUBLICATION: missing bytes never become an observed zero."""
    missing = e.measure(tmp_path / "raw", "20261018", tmp_path)
    assert missing["gate_check_summary"][0]["observed"] == "missing"
    assert not missing.get("owned_execution_failed", False)


def test_current_wheel_and_real_children(tmp_path):
    """SCENARIO-REPORT-8646-ISOLATION: fresh wheel executes actual inference and public routes."""
    work = e.measure(tmp_path / "raw", "20261018", e.ROOT)
    assert not work["gate_check_summary"], work["gate_check_summary"]
    assert e.previous.reduce(work)["completed_count"] >= 128
    assert work["wheel_identity"]["fresh_current_build"]
    assert all(r["passed"] for r in work["isolation_rows"] + work["cleanup_rows"])
    assert work["installed_module_hashes"]
    atomic_json(tmp_path / "measured.json", work)


def test_report_and_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8646-REPLAY: external authority rejects repaired envelopes."""
    from carnot.reporting import installed_report_8646 as r

    work = e.measure(tmp_path / "raw", "20261018", tmp_path)
    work.update(
        validation_receipts=[],
        original_node_union={},
        current_node_union={},
        child_shard_manifest=[],
    )
    authority_path = tmp_path / "authority.json"
    atomic_json(authority_path, work)
    authority = e.reference(authority_path)
    value = r.artifact(authority)
    assert value["verdict_class"] == "blocked"
    assert value["experiment_id"] == 8646
    candidate = tmp_path / (e.NAME + ".json")
    atomic_json(candidate, value)
    r.replay(candidate, authority)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(candidate, changed)
    with pytest.raises(ValueError, match="immutable_reduction_drift"):
        r.replay(candidate, authority)
    for argv in [
        ["--date", "bad"],
        ["--cold-replay", str(candidate)],
        ["--private-output", str(e.ROOT / "bad.json")],
        ["--input-root", str(tmp_path)],
    ]:
        assert r.main(argv) == 1
    monkeypatch.setattr(r, "run", lambda *args: 0)
    assert r.main(["--date", "20261018", "--private-output", str(tmp_path / "out.json")]) == 0
    cli = subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--date", "bad"],
        cwd=tmp_path,
        capture_output=True,
        timeout=60,
    )
    assert cli.returncode == 1


def test_measurement_and_overlay_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8646: rejected identities, missing resources and owned failures stay closed."""
    from carnot.reporting import installed_current_8646 as current

    with monkeypatch.context() as patch:
        patch.setattr(e.shutil, "disk_usage", lambda *args: type("Disk", (), {"free": 0})())
        assert (
            e.measure(tmp_path / "disk", "20261018", e.ROOT)["gate_check_summary"][0]["observed"]
            == "memory_disk_tools"
        )
    with monkeypatch.context() as patch:
        patch.setattr(
            current, "installed", lambda *args: (_ for _ in ()).throw(ValueError("owned_execution"))
        )
        result = e.measure(tmp_path / "owned", "20261018", e.ROOT)
        assert result["owned_execution_failed"]
    with monkeypatch.context() as patch:
        patch.setattr(current, "installed", lambda *args: {})
        patch.setattr(e.previous, "reduce", lambda *args: {})
        ticks = iter([0, 3601_000_000_000, 3602_000_000_000])
        patch.setattr(e.time, "monotonic_ns", lambda: next(ticks))
        patch.setattr(e, "historical", lambda *args: {})
        assert (
            e.measure(tmp_path / "deadline", "20261018", e.ROOT)["gate_check_summary"][0][
                "observed"
            ]
            == "measurement_deadline"
        )
    for bad_exit, bad_identity in [(True, False), (False, True)]:

        def fake(name, argv, logs, **kwargs):
            logs.mkdir(parents=True, exist_ok=True)
            out, err = logs / (name + ".stdout"), logs / (name + ".stderr")
            out.write_text(json.dumps(dict(path="wrong", sha256=e.HASH)))
            err.write_text("input_hash:private")
            return dict(
                name=name,
                passed=not (bad_exit and name == "original_overlay_probe"),
                stdout_path=str(out),
                stderr_path=str(err),
            )

        with monkeypatch.context() as patch:
            patch.setattr(e, "child", fake)
            with pytest.raises(
                ValueError,
                match="original_overlay_execution_failed"
                if bad_exit
                else "original_import_identity",
            ):
                e.historical(tmp_path / f"overlay-{bad_exit}-{bad_identity}")
    monkeypatch.setattr(current, "child", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="current_wheel_build"):
        current.build_wheel(tmp_path / "build", {})


def test_private_terminal_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8646-PUBLICATION: actual terminal readers inspect a private blocked artifact."""
    from carnot.reporting import installed_report_8646 as r
    from carnot.reporting import installed_validation_8646 as v

    original_measure = e.measure

    def measured(raw, run_date, root):
        return original_measure(raw, run_date, tmp_path / "missing-inputs")

    def checked(work, raw):
        work.update(original_node_union={}, current_node_union={}, child_shard_manifest=[])
        return [dict(name=name, passed=True) for name in r.REQUIRED]

    monkeypatch.setattr(e, "measure", measured)
    monkeypatch.setattr(v, "validate", checked)
    # The original readers intentionally reject blocked artifacts as ready inputs.
    monkeypatch.setattr(r, "reader_receipt", lambda *args, **kwargs: dict(passed=True))
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, "20261018", e.ROOT) == 0
    value = json.loads(output.read_bytes())
    assert value["installed_measurement_ready_score"] == 0
    authority = value["authority_reference"]
    env = dict(__import__("os").environ)
    env.pop("PYTHONPATH", None)
    cli = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--cold-replay",
        str(output),
        "--authority",
        authority["path"],
        authority["sha256"],
    ]
    assert (
        subprocess.run(cli, cwd=tmp_path, env=env, capture_output=True, timeout=60).returncode == 0
    )
    monkeypatch.setattr(r, "reader_receipt", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="publication_reader"):
        r.run(tmp_path / "other" / (e.NAME + ".json"), "20261018", e.ROOT)


def test_worker_import_receipt_and_complete_responses(tmp_path):
    """SCENARIO-REPORT-8646-ISOLATION: each entry records its own actual resolved bytes."""
    from carnot.reporting import installed_current_8646 as current

    command = current.worker_command(sys.executable, tmp_path, "private")
    assert "-c" in command
    receipt = subprocess.run(command, cwd=tmp_path, capture_output=True, timeout=60)
    assert receipt.returncode != 0
    paths = list(tmp_path.glob("worker-import-*.json"))
    assert paths
    rows = json.loads(paths[0].read_bytes())
    assert all(row["sha256"] == e.reference(Path(row["path"]))["sha256"] for row in rows)


def test_replay_primitive_hash_and_runtime_cap(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8646-REPLAY: changed sealed files and exceeded reserves disqualify."""
    from carnot.reporting import installed_report_8646 as r
    from carnot.reporting import installed_validation_8646 as v

    work = e.measure(tmp_path / "raw", "20261018", tmp_path)
    work.update(
        validation_receipts=[],
        original_node_union={},
        current_node_union={},
        child_shard_manifest=[],
    )
    operand = tmp_path / "operand"
    operand.write_text("original")
    work["raw_shard_hashes"] = [e.reference(operand)]
    authority_path = tmp_path / "authority.json"
    atomic_json(authority_path, work)
    authority = e.reference(authority_path)
    value = r.artifact(authority)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    operand.write_text("changed")
    with pytest.raises(ValueError, match="replay_input_hash"):
        r.replay(candidate, authority)
    captured = {}
    monkeypatch.setattr(e, "measure", lambda *args: dict(phase_spans=[]))
    monkeypatch.setattr(v, "validate", lambda *args: [])
    ticks = iter([0, 100, 4501_000_000_000])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(ticks))

    def stopped(ref):
        captured.update(e.load(ref))
        raise ValueError("stop_before_publication")

    monkeypatch.setattr(r, "artifact", stopped)
    with pytest.raises(ValueError, match="stop_before_publication"):
        r.run(tmp_path / (e.NAME + ".json"), "20261018", e.ROOT)
    assert captured["validation_receipts"] == [dict(name="runtime_cap", passed=False)]


def test_actual_forced_cleanup_is_explicit(tmp_path):
    """SCENARIO-VERIFY-8646-CUSTODY: a deliberately stopped real child is killed and reaped."""
    from carnot.reporting import installed_current_8646 as current

    config, _ = current.configurations()
    path = tmp_path / "head.json"
    atomic_json(path, config)
    row = current.forced_cleanup(
        current.worker_command(sys.executable, tmp_path, "private"), path, tmp_path
    )
    assert row["passed"] and row["forced_cleanup"] and row["child_reaped"]
    assert row["intentional_failure_control"]
    assert row["exit_code"] < 0
