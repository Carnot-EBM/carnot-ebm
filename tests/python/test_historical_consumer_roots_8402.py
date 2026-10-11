"""REQ-REPORT-8402 / REQ-VERIFY-8402: historical passes cannot imply science."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import historical_consumer_roots_8402 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_authentic_roots_and_negative_controls(tmp_path):
    """SCENARIO-REPORT-8402-ROOTS: independently pinned tasks survive rotation."""
    fixtures = e.fixtures(tmp_path / "fixtures")
    assert e.check_fixtures(fixtures)
    assert len(e.historical_tasks(722)) == 14
    assert e.historical_tasks(722)[0]["id"] == "exp8374-contract-methods"
    assert fixtures["722"]["independent_design_available"] is False
    for version in [720, 721]:
        with e.inject(fixtures[str(version)], tmp_path / "writes"):
            value = e.ROOT / e.DESIGN
            assert f"v{version}" in value.read_text().lower()
            assert len(e.parse_design(value.read_text(), milestone=f"2026.10.{version}")[1]) == 14
    for mutation in ["task", "alias", "missing", "root"]:
        bad = deepcopy(fixtures)
        if mutation == "task":
            bad["722"]["tasks"][0]["prompt"] += " altered"
            bad["722"]["tasks_sha256"] = e.tasks_digest(bad["722"]["tasks"])
        elif mutation == "alias":
            bad["721"]["refs"][0]["source_path"] = str(tmp_path / "wrong-alias")
        elif mutation == "missing":
            bad["721"]["refs"][0]["path"] = str(tmp_path / "missing")
        else:
            bad["721"]["root"] = str(tmp_path / "wrong-root")
        assert not e.check_fixtures(bad)
    with pytest.raises(ValueError):
        e.inject({}, tmp_path).__enter__()


def test_contract_and_methods(tmp_path):
    """SCENARIO-REPORT-8402-METHODS: complete full objects and causal tapes are frozen."""
    bound = e.authority(e.ROOT, tmp_path / "authority")
    assert bound["activated"] and len(bound["tasks"]) == 14
    protocol = json.loads((e.ROOT / e.PROTOCOL).read_bytes())
    assert protocol["canonical_tasks_sha256"] == bound["canonical_tasks_sha256"]
    assert protocol["MODEL_SPECS"] == [] and protocol["stream_slots"] == 864
    for ratio in [1, 8, 64]:
        tape = e.cost_tape(ratio)
        assert sum(r["kind"] == "prediction" for r in tape) == 8 * ratio
        releases = [r for r in tape if r["kind"] == "feedback"]
        assert len(releases) == 8
        assert all(r["tick"] == r["issued_tick"] + 8 for r in releases)
        assert max(r["tick"] for r in tape) == 8 * ratio + 8
    assert not e.authority(tmp_path, tmp_path / "absent")["activated"]
    e.progress("test_control", 1, 0)


def test_classification_preserves_failed_families(tmp_path):
    """SCENARIO-VERIFY-8402-CONTROLS: one measured failure zeros only its own gate."""
    pass_log = tmp_path / "pass.stdout"
    pass_log.write_text("tests/python/a.py::test_one PASSED\n1 passed in 0.01s\n")
    fail_log = tmp_path / "fail.stdout"
    fail_log.write_text("tests/python/a.py::test_one FAILED\n1 failed in 0.01s\n")
    empty = tmp_path / "stderr"
    empty.write_bytes(b"")

    def receipt(path, code):
        return dict(
            stdout_path=str(path),
            stdout_sha256=e.sha256_file(path),
            stderr_path=str(empty),
            stderr_sha256=e.sha256_file(empty),
            argv=["pytest", "same_node"],
            actual_exit=code,
            exit_code=code,
            passed=code == 0,
            timed_out=False,
        )

    good, bad = receipt(pass_log, 0), receipt(fail_log, 1)
    assert e.family("direct", [good, good], ["same_node"])["ready"]
    failed = e.family("direct", [bad, bad], ["same_node"])
    assert not failed["ready"] and failed["status"] == "failed"
    assert e.family("runtime", [], ["same_node"])["status"] == "blocked"
    assert not e.family("direct", [good, bad], ["same_node"])["ready"]
    assert e.parse_outcomes(pass_log)["passed"] == ["tests/python/a.py::test_one"]
    assert e.parse_outcomes(fail_log)["failed"] == ["tests/python/a.py::test_one"]


def test_real_child_and_replay_controls(tmp_path):
    """SCENARIO-VERIFY-8402-CONTROLS: exits and cold reads are actual subprocess evidence."""
    from carnot.reporting import historical_consumer_runner_8402 as r

    assert any(p["name"] == "private_E2E021" for p in r.manifest(tmp_path / "plan"))
    output = tmp_path / "results" / (e.NAME + ".json")
    result = child(
        "missing_root",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--date",
            "20261011",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=90,
    )
    assert result["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert e.replay(output)
    for field in ["completed_count", "historical_direct_consumers_ready_score", "rows"]:
        changed = deepcopy(value)
        changed[field] = 999
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        atomic_json(tmp_path / "forged.json", changed)
        assert not e.replay(tmp_path / "forged.json")
    assert not e.replay(tmp_path / "missing.json")
    for label, argv, expected in [
        ("valid", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "missing.json")], 1),
        ("error", ["--deliberate-error"], 7),
        ("date", ["--date", "20261010"], 2),
    ]:
        assert child(
            label,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *argv],
            tmp_path / "cli",
            deadline=60,
            expected=expected,
        )["passed"]
    assert r.main(["--cold-replay", str(output)]) == 0
    with pytest.raises(SystemExit):
        r.main(["--private-control"])


def test_owned_failure_and_primitive_tamper(tmp_path):
    """SCENARIO-VERIFY-8402-CONTROLS: owned failures zero every score; external absence stays blocked."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    log = tmp_path / "positive.stdout"
    log.write_text("tests/python/a.py::test_one PASSED\n")
    err = tmp_path / "stderr"
    err.write_bytes(b"")
    receipt = dict(
        name="owned",
        scope="owned",
        passed=True,
        actual_exit=0,
        exit_code=0,
        stdout_path=str(log),
        stdout_sha256=e.sha256_file(log),
        stderr_path=str(err),
        stderr_sha256=e.sha256_file(err),
        timed_out=False,
    )
    for name in ["direct", "runtime"]:
        work["families"][name] = e.family(name, [receipt, receipt], ["test_one"])
    atomic_json(tmp_path / "raw/measurement.json", work)
    output = tmp_path / "results" / (e.NAME + ".json")
    value = e.build(work, [receipt], tmp_path / "raw", output)
    atomic_json(output, value)
    assert value["required_checks_passed"] and e.replay(output)
    assert value["verdict_class"] == "blocked"
    assert value["historical_direct_consumers_ready_score"] == 1
    assert value["historical_runtime_consumers_ready_score"] == 1
    failed = e.build(work, [dict(receipt, passed=False)], tmp_path / "raw", output)
    assert failed["verdict_class"] == "disqualified"
    assert failed["current_contract_ready_score"] == 0
    assert failed["historical_direct_consumers_ready_score"] == 0
    assert failed["historical_runtime_consumers_ready_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert not any(value["model_invocation_counts"].values())
    changed = deepcopy(work)
    changed["contract"]["tasks"][0]["prompt"] += " forged"
    atomic_json(tmp_path / "raw/measurement.json", changed)
    forged = e.build(changed, [receipt], tmp_path / "raw", output)
    atomic_json(output, forged)
    assert not e.replay(output)
    atomic_json(tmp_path / "raw/measurement.json", work)
    changed = deepcopy(work)
    changed["families"]["direct"]["ready"] = False
    atomic_json(tmp_path / "raw/measurement.json", changed)
    atomic_json(output, e.build(changed, [receipt], tmp_path / "raw", output))
    assert not e.replay(output)


def test_injection_copy_and_current_bytes(tmp_path):
    """SCENARIO-REPORT-8402-ROOTS: copy APIs obey the same explicit pinned dependency root."""
    import shutil
    from carnot.testing import historical_roots_8402 as plugin

    fixtures = e.fixtures(tmp_path / "fixtures")
    with e.inject(fixtures["721"], tmp_path / "nested/writes"):
        copied = tmp_path / "active-copy.yaml"
        shutil.copyfile(e.ROOT / e.ACTIVE, copied)
        assert e.sha256_file(copied) == fixtures["721"]["refs"][1]["sha256"]
    assert plugin.version_for("test_local_consumer_qualification_8347") == "720"
    assert plugin.version_for("test_threshold_guard_8362") == "721"
    assert plugin.version_for("test_v721_capstone_worker_memory_8373") == "721"
    assert plugin.version_for("test_direct_atomic_state_8376") == "722"
    assert plugin.version_for("unrelated") is None


def test_private_family_execution_and_seals(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8402-CONTROLS: real children seal passing families separately."""
    from carnot.reporting import historical_consumer_runner_8402 as r

    monkeypatch.setattr(
        r,
        "manifest",
        lambda p: [
            dict(
                name="real_owned_control",
                argv=[sys.executable, "-u", "-c", "print('actual classification control')"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ],
    )
    monkeypatch.setattr(
        r,
        "family_commands",
        lambda p: dict(
            direct=[
                "tests/python/test_v721_capstone_frozen_aliases_8373.py::test_source_alias_reads_and_writes_use_private_custody"
            ],
            runtime=["tests/python/test_v722_contract_methods_8374.py::test_private_e2e018[match]"],
        ),
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert r.main(["--root", str(e.ROOT), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert e.replay(output) and value["current_contract_ready_score"] == 1
    for name in ["direct", "runtime"]:
        assert e.authenticate_family(output, name)["ready"]
    assert len(value["historical_family_receipts"]) == 2
    work_ref = value["work_reference"]
    original = json.loads(Path(work_ref["path"]).read_bytes())
    altered = deepcopy(original)
    altered["contract"]["contract_rows"][0]["absolute_metric"] = 42
    atomic_json(Path(work_ref["path"]), altered)
    atomic_json(
        output,
        e.build(altered, value["validation_receipts"], Path(work_ref["path"]).parent, output),
    )
    assert not e.replay(output)
    atomic_json(Path(work_ref["path"]), original)
    atomic_json(output, value)
    seal = Path(value["historical_family_receipts"]["direct"]["path"])
    original_seal = seal.read_bytes()
    seal.write_bytes(b"changed")
    assert not e.replay(output)
    with pytest.raises(ValueError):
        e.authenticate_family(output, "direct")
    seal.write_bytes(original_seal)


def test_resource_failure_and_plugin_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8402-ROOTS: unavailable resources and malformed roots fail closed."""
    from types import SimpleNamespace
    from carnot.testing import historical_roots_8402 as plugin

    monkeypatch.setattr(e.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    work = e.measure(tmp_path, tmp_path / "raw")
    assert any(f["artifact_field"] == "private_disk_memory_tools" for f in work["failures"])
    monkeypatch.undo()
    assert not e.check_fixtures({})
    tasks = e.historical_tasks(722)
    with pytest.raises(ValueError):
        e.checked(e.ROOT / e.PROTOCOL, "sha256:wrong")
    assert len(tasks) == 14
    fixtures = e.fixtures(tmp_path / "fixtures")
    manifest = tmp_path / "fixtures/manifest.json"
    options = {
        "--historical-fixture-manifest": str(manifest),
        "--historical-current-root": str(tmp_path / "absent"),
    }
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda k: options[k]),
        module=SimpleNamespace(__name__="test_threshold_guard_8362"),
    )
    with pytest.raises(ValueError, match="explicit_current_root"):
        next(plugin.historical_dependency_root.__wrapped__(request, None))
    atomic_json(manifest, {})
    with pytest.raises(ValueError, match="historical_fixture_custody"):
        next(plugin.historical_dependency_root.__wrapped__(request, None))
    atomic_json(manifest, fixtures)


def test_more_pinned_and_rehashed_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8402-ROOTS: repaired metadata cannot authorize altered authority."""
    fixtures = e.fixtures(tmp_path / "fixtures")
    for field, replacement in [("independent_design_available", False), ("tasks_sha256", "wrong")]:
        bad = deepcopy(fixtures)
        bad["721"][field] = replacement
        assert not e.check_fixtures(bad)
    bad = deepcopy(fixtures)
    bad["721"]["refs"][0]["sha256"] = "wrong"
    assert not e.check_fixtures(bad)
    root = tmp_path / "current"
    for name in [e.DESIGN, e.ACTIVE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    path = root / e.DESIGN
    path.write_text(
        path.read_text().replace(
            '"title": "Bind fourteen tasks', '"title": "Changed fourteen tasks'
        )
    )
    assert not e.authority(root, tmp_path / "assessment")["activated"]
    monkeypatch.setattr(e.re, "findall", lambda *a, **k: [])
    with pytest.raises(ValueError, match="historical_log_counts"):
        e.historical_logs(tmp_path)
    monkeypatch.undo()
    work = e.measure(e.ROOT, tmp_path / "raw")
    receipt = child(
        "classification",
        [sys.executable, "-c", "print('real child')"],
        tmp_path / "logs",
        deadline=10,
    )
    for field in ["source_alias", "fixture", "protocol"]:
        altered = deepcopy(work)
        if field == "source_alias":
            altered["refs"][0]["source_path"] = str(tmp_path / "wrong")
        elif field == "fixture":
            altered["fixtures"]["722"]["tasks"][0]["prompt"] += " rehashed"
        else:
            altered["protocol"]["stream_slots"] += 1
        atomic_json(tmp_path / "raw/measurement.json", altered)
        path = tmp_path / "forged.json"
        atomic_json(path, e.build(altered, [receipt], tmp_path / "raw", path))
        assert not e.replay(path)


def test_failed_family_authentication(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8402-CONTROLS: a genuinely failing family stays unavailable downstream."""
    import coverage
    from carnot.reporting import historical_consumer_runner_8402 as r

    private = tmp_path / "private"
    private.mkdir()
    cov = coverage.Coverage(
        config_file=False,
        data_file=str(private / ".coverage.real"),
        include=[str(e.ROOT / e.OWNED[0])],
    )
    with cov.collect():
        e.progress("real_coverage_for_private_control")
    cov.save()
    cov.json_report(outfile=str(private / "coverage.json"))
    plan = [
        dict(
            name="real_owned_control",
            argv=[sys.executable, "-c", "print('actual owned child')"],
            expected=0,
            deadline=10,
            scope="owned",
        )
    ]
    monkeypatch.setattr(r, "manifest", lambda p: plan)
    node = "tests/python/test_v721_capstone_frozen_aliases_8373.py::test_source_alias_reads_and_writes_use_private_custody"
    monkeypatch.setattr(r, "family_commands", lambda p: dict(direct=[node], runtime=[node]))
    real_history = r.history_plan

    def absent_history(raw):
        rows = real_history(raw)
        for row in rows:
            if row["family"] == "direct":
                row["argv"] = [
                    "--historical-current-root=" + str(tmp_path / "missing")
                    if arg.startswith("--historical-current-root=")
                    else arg
                    for arg in row["argv"]
                ]
        return rows

    monkeypatch.setattr(r, "history_plan", absent_history)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert r.run(e.ROOT, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert value["historical_direct_consumers_ready_score"] == 0
    assert value["historical_runtime_consumers_ready_score"] == 1
    with pytest.raises(ValueError, match="qualified_family_terminal"):
        e.authenticate_family(output, "direct")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    assert work["coverage_shards"]
    ref = work["family_refs"]["runtime"]
    seal = json.loads(Path(ref["path"]).read_bytes())
    seal["terminal_hash"] = "sha256:rehashed_tamper"
    atomic_json(Path(ref["path"]), seal)
    ref["sha256"] = e.sha256_file(Path(ref["path"]))
    atomic_json(Path(value["work_reference"]["path"]), work)
    atomic_json(
        output,
        e.build(
            work, value["validation_receipts"], Path(value["work_reference"]["path"]).parent, output
        ),
    )
    assert not e.replay(output)


def test_historical_child_root_and_original_dates(tmp_path):
    """SCENARIO-REPORT-8402-ROOTS: real nested children receive the same explicit dependency root."""
    import subprocess
    from carnot.reporting import historical_consumer_runner_8402 as r
    from carnot.testing import historical_roots_8402 as plugin

    e.fixtures(tmp_path / "fixtures")
    manifest = tmp_path / "fixtures/manifest.json"
    probe = tmp_path / "root_probe.py"
    probe.write_text(
        "from pathlib import Path\nfrom carnot.reporting.historical_consumer_roots_8402 import ROOT, DESIGN\nassert 'v721' in (ROOT / DESIGN).read_text().lower()\n"
    )
    assert r.historical_exec(manifest, "721", [str(probe)]) == 0
    probe.write_text("raise SystemExit(7)\n")
    assert r.historical_exec(manifest, "721", [str(probe)]) == 7
    probe.write_text("raise SystemExit\n")
    assert r.historical_exec(manifest, "721", [str(probe)]) == 0
    popen = plugin.bound_popen(manifest, "721", subprocess.Popen)
    old_cli = e.ROOT / "scripts/experiments/experiment_8362_v721_threshold_guard.py"
    # Use the shipped CLI path declared by its actual module.
    from carnot.reporting import threshold_guard_8362 as old

    old_cli = old.ROOT / old.CLI
    process = popen(
        [sys.executable, "-u", str(old_cli), "--date", "wrong"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = process.communicate(timeout=60)
    assert process.returncode == 1
    assert b"date must be 20261010" in stderr
    plain = popen([sys.executable, "-c", "raise SystemExit(0)"])
    assert plain.wait(timeout=60) == 0
    atomic_json(manifest, {})
    with pytest.raises(ValueError, match="historical_fixture_custody"):
        r.historical_exec(manifest, "721", [str(probe)])


def test_module_fixture_uses_pinned_epoch(tmp_path):
    """SCENARIO-REPORT-8402-ROOTS: input isolation starts before module-scoped operand fixtures."""
    from carnot.reporting import historical_consumer_runner_8402 as r

    e.fixtures(tmp_path / "fixtures")
    r.rotations(e.ROOT, tmp_path)
    source = tmp_path / "test_threshold_guard_8362_owned_probe.py"
    source.write_text(
        "import pytest\nfrom carnot.reporting.historical_consumer_roots_8402 import ROOT,DESIGN,parse_design\n@pytest.fixture(scope='module')\ndef operands():\n    return parse_design((ROOT/DESIGN).read_text(),milestone='2026.10.721')[1]\ndef test_pinned(operands):\n    assert len(operands)==14\n"
    )
    receipt = child(
        "module_scoped_root",
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            "-p",
            "carnot.testing.historical_roots_8402",
            "--historical-fixture-manifest=" + str(tmp_path / "fixtures/manifest.json"),
            "--historical-current-root=" + str(tmp_path / "current/pinned"),
            str(source),
            "--basetemp=" + str(tmp_path / "probe"),
        ],
        tmp_path / "logs",
        deadline=60,
    )
    assert receipt["passed"]
