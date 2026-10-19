"""REQ-VERIFY-8642: real SQLite evidence must survive later validation."""

from copy import deepcopy
import json
from pathlib import Path

from coverage import CoverageData
import pytest

from carnot.reporting import streaming_custody_8642 as c


def database(path, source):
    """Use actual CoverageData bytes so filename-only custody cannot pass."""
    data = CoverageData(basename=str(path))
    data.add_lines({str(source): {1}})
    data.write()


def test_direct_unsuffixed_capture_and_repeated_combine(tmp_path):
    """SCENARIO-VERIFY-8642-CAPTURE: unsuffixed data survives two combines."""
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    base = tmp_path / "child" / ".coverage"
    base.parent.mkdir()
    database(base, source)
    rows = c.capture(base, tmp_path / "raw", [source])
    assert len(rows) == 1 and rows[0]["unsuffixed"]
    assert rows[0]["inventory"][0]["lines"] == [1]
    base.unlink()
    before = deepcopy(rows)
    for i in range(2):
        report = c.combine(rows, [source], tmp_path / f"combine-{i}")
        assert report["passed"] and report["statement_count"] == 1
    assert rows == before
    assert c.authenticate(rows[0]["reference"])


@pytest.mark.parametrize("mode", ["missing", "foreign", "rehashed", "alias", "empty"])
def test_capture_rejects_invalid_custody(tmp_path, mode):
    """SCENARIO-VERIFY-8642-IMMUTABLE: repaired hashes cannot bless foreign data."""
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    base = tmp_path / ".coverage"
    if mode != "missing":
        database(base, source if mode != "foreign" else tmp_path / "foreign.py")
    if mode == "empty":
        data = CoverageData(basename=str(base))
        data.erase()
        data.write()
    if mode in {"missing", "foreign", "empty"}:
        with pytest.raises(ValueError):
            c.capture(base, tmp_path / "raw", [source])
        return
    row = c.capture(base, tmp_path / "raw", [source])[0]
    ref = row["reference"]
    if mode == "rehashed":
        path = Path(ref["path"])
        path.chmod(0o644)
        path.write_bytes(b"changed")
        ref["sha256"] = c.sha256_file(path)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(ref["path"])
        ref["path"] = str(alias)
    assert not c.authenticate(ref)


def test_real_old_hook_failure_and_new_capture(tmp_path):
    """SCENARIO-VERIFY-8642-CAPTURE: execute the historical small-runner defect."""
    row = c.retention_regression(tmp_path)
    assert row["passed"] and row["original_coverage_retention"] is False
    assert row["direct_capture"][0]["unsuffixed"]
    assert row["scratch_cleanup"]


def test_freeze_rejects_changed_content_address(tmp_path):
    """REQ-VERIFY-8642: existing immutable operands cannot be silently replaced."""
    source = tmp_path / "input"
    source.write_text("one")
    ref = c.freeze(source, tmp_path / "raw")
    assert c.freeze(source, tmp_path / "raw") == ref
    target = Path(ref["path"])
    target.chmod(0o644)
    target.write_text("two")
    with pytest.raises(ValueError, match="immutable_collision"):
        c.freeze(source, tmp_path / "raw")
    assert not c.authenticate({"path": str(tmp_path / "absent"), "sha256": "sha256:bad"})


def test_measure_build_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8642-PUBLICATION: immutable measurements survive repeated readers."""
    from carnot.reporting import streaming_execution_8642 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261018")
    work["validation"] = dict(
        receipts=[dict(name="fixture", passed=True)],
        child_shard_manifest=[],
        original_node_union=dict(passed=True),
        coverage=dict(passed=True),
    )
    ref = e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(ref, [], {}, output)
    e.atomic_json(output, value)
    assert e.replay(output) and e.replay(output)
    assert value["streaming_extractor_ready_score"] == 1
    assert value["independent_generalization_score"] == 0
    for mode in ["metric", "measurement", "deleted", "rehashed"]:
        changed = deepcopy(value)
        if mode == "metric":
            changed["streaming_extractor_ready_score"] = 999
        elif mode == "measurement":
            changed["measurement_reference"]["path"] = str(tmp_path / "absent")
        else:
            path = Path(ref["path"])
            original = path.read_bytes()
            path.chmod(0o644)
            path.write_bytes(b"{}")
            if mode == "deleted":
                path.unlink()
            else:
                changed["measurement_reference"]["sha256"] = c.sha256_file(path)
        changed["reproducibility_checksum"] = e.canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        e.atomic_json(output, changed)
        assert not e.replay(output)
        if mode in {"deleted", "rehashed"}:
            path.write_bytes(original)
    failed = e.build(ref, [dict(name="owned", passed=False)], {}, output)
    assert failed["verdict_class"] == "disqualified"


def test_missing_external_is_blocked(tmp_path):
    """REQ-REPORT-8642: an absent operand is missing, never an observed numeric zero."""
    from carnot.reporting import streaming_execution_8642 as e

    work = e.measure(tmp_path / "missing", tmp_path / "raw", "20261018")
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        child_shard_manifest=[],
        original_node_union=dict(passed=True),
        coverage=dict(passed=True),
    )
    ref = e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(ref, [], {}, output)
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] is None
    e.atomic_json(output, value)
    assert e.replay(output)


def test_private_cli_modes_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8642-PUBLICATION: real standalone children and checked readers."""
    import os
    import subprocess
    import sys
    from carnot.reporting import streaming_execution_8642 as e
    from carnot.reporting import streaming_execution_runner_8642 as runner

    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    work_path = tmp_path / "private.json"
    result = subprocess.run(
        cli + ["--date", "20261018", "--measure-only", str(work_path)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert subprocess.run(cli + ["--date", "20261399"], capture_output=True).returncode == 2
    assert (
        subprocess.run(
            cli + ["--cold-replay", str(tmp_path / "absent")], capture_output=True
        ).returncode
        == 1
    )
    monkeypatch.setattr(
        runner,
        "run_checks",
        lambda _: dict(
            receipts=[dict(name="outer", passed=True)],
            child_shard_manifest=[],
            original_node_union=dict(passed=True),
            coverage=dict(passed=True),
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261018", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert e.replay(output) and value["verdict_class"] == "null"
    assert Path(value["terminal_validation_sidecar_path"]).is_file()


@pytest.mark.parametrize("capture_failure", [False, True])
def test_validation_driver_boundaries(tmp_path, monkeypatch, capture_failure):
    """REQ-VERIFY-8642: each child is captured before the next, with failures retained."""
    from carnot.reporting import streaming_execution_8642 as e
    from carnot.reporting import streaming_execution_runner_8642 as runner

    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    suite = "tests/python/test_probe_8210.py"
    nodes = [suite + "::test_one", suite + "::test_two"]
    monkeypatch.setattr(e, "SUITES", [suite])
    monkeypatch.setattr(e, "SOURCES", [source])
    monkeypatch.setattr(e, "OWNED", [str(source)])
    monkeypatch.setattr(e, "TEST", str(source))
    captures = []
    original_capture = c.capture

    def command(name, argv, logs, **kwargs):
        """Script command boundaries only; real SQLite bytes exercise capture and combine."""
        import os

        logs.mkdir(parents=True, exist_ok=True)
        out, err = logs / (name + ".out"), logs / (name + ".err")
        out.write_text("\n".join(nodes) if name == "collection" else "unit boundary")
        err.write_text("")
        if "--context=" + name in argv:
            assert len(captures) == int(name.rsplit("_", 1)[1])
            base = Path(os.environ["COVERAGE_FILE"])
            database(base, source)
            junit = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")))
            which = "test_one" if name.endswith("_0") else "test_two"
            junit.write_text(
                '<testsuite><testcase classname="tests.python.test_probe_8210" name="'
                + which
                + '"/></testsuite>'
            )
        return dict(
            name=name, passed=True, normal_exit=True, stdout_path=str(out), stderr_path=str(err)
        )

    def captured(base, raw, sources):
        captures.append(str(base))
        if capture_failure:
            raise ValueError("injected_missing_child")
        return original_capture(base, raw, sources)

    monkeypatch.setattr(runner, "child", command)
    monkeypatch.setattr(c, "capture", captured)
    value = runner.run_checks(tmp_path / "raw")
    assert len(captures) == 2 and len(set(captures)) == 2
    assert value["original_node_union"]["passed"]
    assert all(c.authenticate(r) for r in value["original_node_union"]["junit_references"])
    assert value["coverage"]["passed"] is (not capture_failure)
    assert all(m["passed"] is (not capture_failure) for m in value["child_shard_manifest"])
    assert value["scratch_cleanup"]


def test_combine_rejects_lost_or_changed_inventory(tmp_path):
    """SCENARIO-VERIFY-8642-IMMUTABLE: a saved manifest cannot substitute for its shard."""
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    base = tmp_path / ".coverage"
    database(base, source)
    rows = c.capture(base, tmp_path / "raw", [source])
    rows[0]["inventory"][0]["lines"] = [999]
    with pytest.raises(ValueError, match="child_custody"):
        c.combine(rows, [source], tmp_path / "combine")


def test_replay_authenticates_primitives_and_original_semantics(tmp_path):
    """SCENARIO-VERIFY-8642-IMMUTABLE: rehashing summaries cannot repair changed operands."""
    from carnot.reporting import streaming_execution_8642 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261018")
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        child_shard_manifest=[],
        original_node_union=dict(passed=True),
        coverage=dict(passed=True),
    )
    source = e.SOURCES[0]
    base = tmp_path / ".coverage"
    database(base, source)
    shard = c.capture(base, tmp_path / "raw", e.SOURCES)[0]
    output = tmp_path / (e.NAME + ".json")
    for mode in ["operand", "authority", "inventory", "roster", "feature", "offset"]:
        changed = deepcopy(work)
        if mode == "operand":
            changed["immutable_refs"][0]["sha256"] = "sha256:bad"
        elif mode == "authority":
            bad_source = tmp_path / "changed_source.py"
            bad_source.write_text("different original source")
            changed["original_authority"][0]["origin_path"] = str(bad_source)
        elif mode == "inventory":
            bad = deepcopy(shard)
            bad["inventory"][0]["lines"] = [999]
            changed["validation"]["child_shard_manifest"] = [dict(passed=True, shards=[bad])]
        elif mode == "roster":
            changed["fixture"]["chunk_rows"].pop()
        elif mode == "feature":
            changed["fixture"]["chunk_rows"][9]["features"]["values"] = [999] * 8
        else:
            row = changed["fixture"]["chunk_rows"][9]
            original = Path(row["primitive"]["path"]).read_bytes()
            bad = tmp_path / "bad.jsonl"
            bad.write_bytes(original.replace(b'"start":0', b'"start":1', 1))
            row["primitive"] = c.freeze(bad, tmp_path / "raw")
        ref = e.seal(changed)
        value = e.build(ref, [], {}, output)
        e.atomic_json(output, value)
        assert not e.replay(output), mode


def test_private_disqualification_and_real_rehashed_child(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8642-PUBLICATION: a failed owned gate publishes zero readiness."""
    import subprocess
    import sys
    from carnot.reporting import streaming_execution_8642 as e
    from carnot.reporting import streaming_execution_runner_8642 as runner

    monkeypatch.setattr(
        runner,
        "run_checks",
        lambda _: dict(
            receipts=[dict(name="failed_owned", passed=False)],
            child_shard_manifest=[],
            original_node_union=dict(passed=True),
            coverage=dict(passed=True),
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261018", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert (
        value["verdict_class"] == "disqualified" and value["streaming_extractor_ready_score"] == 0
    )
    value["streaming_extractor_ready_score"] = 999
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = e.canonical_hash(value)
    e.atomic_json(output, value)
    result = subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(output)],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 1 and b"reduction_drift" in result.stdout


def test_missing_coverage_is_owned_failure(tmp_path):
    """REQ-REPORT-8642: passing command exits cannot conceal missing statement coverage."""
    from carnot.reporting import streaming_execution_8642 as e

    work = e.measure(tmp_path / "missing", tmp_path / "raw", "20261018")
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        child_shard_manifest=[],
        original_node_union=dict(passed=True),
        coverage=dict(passed=False),
    )
    value = e.build(e.seal(work), [], {}, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert any(r["check"] == "owned_validation" for r in value["gate_check_summary"])


def test_fully_rehashed_measurement_replacement_is_rejected(tmp_path):
    """SCENARIO-VERIFY-8642-IMMUTABLE: the invocation anchor survives repaired summaries."""
    from carnot.reporting import streaming_execution_8642 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261018")
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        child_shard_manifest=[],
        original_node_union=dict(passed=True),
        coverage=dict(passed=True),
    )
    original = e.seal(work)
    changed = deepcopy(work)
    changed["duration_s"] += 123
    replacement = tmp_path / "replacement" / "measurement.json"
    e.atomic_json(replacement, changed)
    substituted = c.freeze(replacement, tmp_path / "raw")
    assert substituted["sha256"] != original["sha256"]
    output = tmp_path / (e.NAME + ".json")
    value = e.build(substituted, [], {}, output)
    e.atomic_json(output, value)
    assert not e.replay(output)
