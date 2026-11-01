"""REQ-REPORT-8681, REQ-VERIFY-8681: preserve original evidence and qualify fresh work."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml

from carnot.reporting import v744_contract_methods as original
from carnot.reporting import v744_contract_report as old_report
from carnot.reporting import v745_contract_methods as e
from carnot.reporting import v745_contract_report as report
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    """Original authority copies keep the failed producer's source and results unchanged."""
    root = tmp_path_factory.mktemp("original-authority")
    p = original.protocol()
    design = root / original.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes(Path(p["v744_design_reference"]["path"]).read_bytes())
    (root / original.ACTIVE).write_text(
        yaml.safe_dump(dict(milestone=original.MILESTONE, tasks=p["task_map_v744"]))
    )
    raw = root / "raw"
    return original.measure(root, raw, "20261101"), raw


def test_retained_mutation_and_corrected_boundary(legacy, tmp_path):
    """SCENARIO-REPORT-8681-IMMUTABLE: execute retained source, then corrected build/replay."""
    work, raw = legacy
    failure = original.read(original.PRIOR_FAILURE_REFERENCE)
    assert failure["flagged_adversarial"] and not failure["required_checks_passed"]
    ref = next(
        r
        for r in failure["references"]
        if r["path"].endswith(
            "source_before_report_fix/python/carnot/reporting/v744_contract_report.py"
        )
    )
    assert original.checked(ref)
    spec = importlib.util.spec_from_file_location("retained_report", ref["path"])
    retained = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retained)
    checks, output = tmp_path / "checks.json", tmp_path / (original.NAME + ".json")
    original.atomic_json(checks, dict(receipts=[]))
    changed = deepcopy(work)
    before = original.canonical_hash(changed)
    retained.build(changed, original.reference(checks), raw, output)
    assert original.canonical_hash(changed) != before
    failed_log = Path(original.PRIOR_FAILURE_REFERENCE["path"]).parent / "checks/owned_tests.stdout"
    assert "1 failed, 76 passed" in failed_log.read_text()
    shard = tmp_path / "real.sqlite"
    source = Path(original.PRIOR_FAILURE_REFERENCE["path"]).parent / "coverage_shards/originals/0"
    shutil.copyfile(source, shard)
    assert shard.read_bytes()[:16] == b"SQLite format 3\x00"
    original.atomic_json(
        checks,
        dict(
            receipts=[
                dict(
                    name="original_coverage_retention",
                    passed=True,
                    original_references=[original.reference(shard)],
                )
            ]
        ),
    )
    stable = original.canonical_hash(work)
    first = old_report.build(work, original.reference(checks), raw, output)
    assert old_report.build(work, original.reference(checks), raw, output) == first
    assert original.canonical_hash(work) == stable
    original.atomic_json(output, first)
    assert old_report.replay(output)
    assert original.canonical_hash(work) == stable
    command = [
        sys.executable,
        "-u",
        str(original.ROOT / original.CLI),
        "--cold-replay",
        str(output),
    ]
    assert child("original_cold", command, tmp_path / "logs", deadline=120)["passed"]
    shard.write_bytes(b"tampered real shard")
    assert not old_report.replay(output)
    shard.unlink()
    assert not old_report.replay(output)
    assert original.canonical_hash(work) == stable
    print(
        "IMMUTABLE_8681 "
        + json.dumps(
            dict(
                study="original_corrected_boundary",
                before=stable,
                after=original.canonical_hash(work),
                repeated_build=True,
                cold_replay=True,
                retained_real_shard=True,
                missing_rejected=True,
                tampered_rejected=True,
                historical_mutation_reproduced=True,
            )
        ),
        flush=True,
    )


@pytest.fixture
def authority(tmp_path):
    """Independent copies expose authority mutations without writing the live queue."""
    target = tmp_path / e.DESIGN
    target.parent.mkdir(parents=True)
    target.write_bytes((e.ROOT / e.DESIGN).read_bytes())
    (tmp_path / e.ACTIVE).write_bytes((e.ROOT / e.ACTIVE).read_bytes())
    return tmp_path


def test_authority_and_registered_operands(authority):
    """SCENARIO-REPORT-8681-AUTHORITY: full objects and an independent digest decide authority."""
    assert e.authority(authority)["ready"]
    rows = e.mutation_controls(authority)
    assert len(rows) >= 20 and all(r["passed"] for r in rows)
    p = e.protocol()
    old = original.protocol()
    for key in [
        "objective_updates",
        "objective_audit",
        "cohort_partition_reference",
        "relational_evidence_protocol",
        "installed_batching_protocol",
    ]:
        assert p[key] == old[key]
    manifests = e.methods_manifests(p)
    assert {"learning", "extraction", "deployment", "symbolic_canary", "fixed_point"} == set(
        manifests
    )
    assert manifests["symbolic_canary"]["calls"] == 36
    assert manifests["fixed_point"]["vectors"] == 256
    assert all(
        row["status"] == "future_dependency_not_observation"
        for row in p["future_dependencies_v745"]
    )
    (authority / e.DESIGN).unlink()
    assert not e.authority(authority)["ready"]


def test_new_report_immutability_and_tampering(authority, tmp_path):
    """SCENARIO-VERIFY-8681-REPLAY: nested report operands and independent seals survive builds."""
    output = tmp_path / (e.NAME + ".json")
    raw = tmp_path / "raw" / e.NAME / "invocations" / "unit"
    work = e.measure(authority, raw, "20261101")
    checks = raw / "validation.json"
    e.atomic_json(checks, dict(receipts=[dict(name="private_child", passed=True)]))
    before = e.canonical_hash(work)
    value = report.build(work, e.reference(checks), raw, output)
    assert report.build(work, e.reference(checks), raw, output) == value
    assert e.canonical_hash(work) == before
    e.atomic_json(output, value)
    assert report.replay(output)
    assert e.canonical_hash(work) == before
    print(
        "IMMUTABLE_8681 "
        + json.dumps(
            dict(
                study="v745_pure_build",
                before=before,
                after=e.canonical_hash(work),
                repeated_build=True,
                replay=True,
            )
        ),
        flush=True,
    )
    for mode in ["summary", "primitive", "local_seal", "missing", "stream"]:
        changed = deepcopy(value)
        if mode == "summary":
            changed["completed_count"] += 1
        if mode in {"primitive", "local_seal"}:
            forged = deepcopy(work)
            forged["history"]["learning_operands_ready_score"] = 0
            path = raw / (mode + ".json")
            e.atomic_json(path, forged)
            changed["primitive_reference"] = e.reference(path)
            if mode == "local_seal":
                changed["measurement_seal_reference"] = e.reference(path)
        if mode == "missing":
            changed["primitive_reference"]["path"] = str(raw / "missing")
        if mode == "stream":
            changed["validation_receipts"] = []
        changed["reproducibility_checksum"] = e.checksum(changed)
        e.atomic_json(output, changed)
        assert not report.replay(output)
    assert not report.replay(tmp_path / "absent")


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8681-REPLAY: actual private publication, replay and rejection paths run."""
    output = tmp_path / "results" / (e.NAME + ".json")
    command = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private",
        command
        + [
            "--date",
            "20261101",
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=180,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["methods_ready_score"] == 0
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("date", ["--date", "bad"], 2),
        ("error", ["--deliberate-error"], 7),
        ("unsafe", ["--private-control"], 2),
    ]:
        assert child(name, command + args, tmp_path / "logs", deadline=120, expected=expected)[
            "passed"
        ]
    value["completed_count"] += 1
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert child(
        "tampered",
        command + ["--cold-replay", str(output)],
        tmp_path / "logs",
        deadline=120,
        expected=1,
    )["passed"]


def test_manifest_and_failed_child(tmp_path, monkeypatch):
    """REQ-VERIFY-8681: independent original IDs and real failed children retain authority."""
    from carnot.reporting import v745_contract_runner as runner

    def collected(private, suite):
        name = "collect_" + Path(suite).stem
        e.atomic_json(private / "collection" / (name + ".receipt.json"), dict(passed=True))
        return [suite + "::unit"], dict(name=name)

    monkeypatch.setattr(runner.custody.collection, "collect", collected)
    plan = runner.manifest(tmp_path / "private")
    names = [row["name"] for row in plan]
    assert names.index("producer_coverage_report") < next(
        i for i, name in enumerate(names) if name.startswith("test_test_experiment_7891")
    )
    assert {"ruff_check", "ruff_format", "strict_mypy", "coverage_report", "spec_coverage"} <= {
        r["name"] for r in plan
    }
    assert all(
        suite
        in [
            n.split("::")[0]
            for row in plan
            if "junit_path" in row
            for n in row["collected_node_ids"]
        ]
        for suite in e.ORIGINAL_TESTS
    )
    with pytest.raises(ValueError, match="outside"):
        runner.manifest(e.ROOT)
    tiny = tmp_path / "test_owned_child.py"
    tiny.write_text(
        'def test_owned():\n    """REQ-VERIFY-8681: retain a real failed pytest child."""\n    assert False\n'
    )
    junit = tmp_path / "child.xml"
    commands = [
        dict(
            name="test_failed",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-c",
                "/dev/null",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                str(tiny),
                "--junitxml=" + str(junit),
            ],
            junit_path=str(junit),
            deadline=10,
            expected=0,
            scope="owned",
        ),
        dict(
            name="coverage_combine",
            argv=[sys.executable, "-c", "pass", str(tmp_path / "private/coverage_copies")],
            deadline=10,
            expected=0,
            scope="owned",
        ),
    ]
    receipts = runner.execute(commands, tmp_path / "raw/checks")
    assert receipts[0]["exit_code"] == 1 and not receipts[0]["passed"]
    assert e.checked(receipts[0]["junit_reference"])
    assert Path(receipts[0]["junit_reference"]["path"]).read_bytes() == junit.read_bytes()
    assert not receipts[-1]["passed"]


def test_owned_failure_and_mutation(authority, tmp_path):
    """REQ-REPORT-8681: owned failure cannot grant readiness or change original operands."""
    output = tmp_path / (e.NAME + ".json")
    raw = tmp_path / "raw" / e.NAME / "invocations" / "unit"
    work = e.measure(authority, raw, "20261101")
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[dict(name="owned_failure", passed=False)]))
    value = report.build(work, e.reference(checks), raw, output)
    assert value["methods_ready_score"] == 0 and value["verdict_class"] == "disqualified"
    assert value["learning_operands_ready_score"] == 1
    failed = child(
        "actual_failed", [sys.executable, "-c", "raise SystemExit(3)"], raw / "failed", deadline=10
    )
    failed["passed"] = True
    e.atomic_json(checks, dict(receipts=[failed]))
    assert report.build(work, e.reference(checks), raw, output)["verdict_class"] == "disqualified"
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    original_protocol = e.original.protocol
    monkey = pytest.MonkeyPatch()
    p = original_protocol()
    p["task_map_v745"] = []
    read_text = Path.read_text

    def changed(path, *args, **kwargs):
        text = read_text(path, *args, **kwargs)
        return (
            text.replace('"exp8681-contract-methods"', '"forged"')
            if path == e.ROOT / e.DESIGN
            else text
        )

    monkey.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="independent_v745_task_seal"):
        e.protocol()
    monkey.undo()


def test_private_original_source_custody(tmp_path):
    """SCENARIO-REPORT-8681-IMMUTABLE: only exact archived source bytes replace a changed input."""
    from carnot.reporting import v745_private_inputs as private

    with private.binding():
        p = original.protocol()
        ref = next(
            r
            for r in p["protocol_operand_references"]
            if r["path"].endswith("arc_competition_agent.py")
        )
        assert original.checked(ref)
        assert not original.checked(dict(ref, sha256="sha256:" + "0" * 64))
        assert not original.checked(dict(ref, path=str(tmp_path / "missing")))
        assert e.sha256_file(private.ARCHIVE) == ref["sha256"]


def test_full_lifecycle_without_qualification(tmp_path, monkeypatch):
    """REQ-VERIFY-8681: absent original execution cannot qualify the production lifecycle."""
    from carnot.reporting import v745_contract_runner as runner

    def plan(private):
        e.atomic_json(private / "coverage.json", {})
        e.atomic_json(private / "original_node_union.json", dict(suites={}))
        return []

    monkeypatch.setattr(runner, "manifest", plan)
    monkeypatch.setattr(runner, "execute", lambda plan, logs: [])
    monkeypatch.setattr(runner.publication, "controls", lambda value, raw: [])
    monkeypatch.setattr(
        runner.publication, "publish", lambda value, output, raw: e.atomic_json(output, value)
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    assert (
        runner.main(
            ["--date", "20261101", "--root", str(tmp_path / "absent"), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["methods_ready_score"] == 0


@pytest.mark.parametrize("mode", ["valid", "missing", "field", "flag", "changed"])
def test_producer_terminal_custody(tmp_path, mode):
    """REQ-REPORT-8681: flags, missing terminal custody and changed bytes cannot grant a gate."""
    p = e.protocol()
    ref = p["current_producers"]["8580"]
    value = e.read(ref)
    if mode == "field":
        value.pop("current_validator_ready_score")
    if mode == "flag":
        value["flagged_adversarial"] = True
    if mode == "missing":
        value["terminal_validation_sidecar_path"] = str(tmp_path / "absent")
    if mode != "valid":
        path = tmp_path / "producer.json"
        e.atomic_json(path, value)
        ref = e.reference(path)
    if mode == "changed":
        Path(ref["path"]).write_text("{}")
    row = e.terminal_custody(ref, "current_validator_ready_score")
    assert row["passed"] == (mode == "valid")
    if mode == "missing":
        p["current_producers"]["8580"] = ref
        assert e.historical(p)["learning_operands_ready_score"] == 0


def test_historical_absence_stays_branch_local():
    """REQ-REPORT-8681: missing quarantined ARC operands cannot veto admitted CPU operands."""
    value = e.historical(e.protocol())
    assert any(not e.checked(ref) for ref in value["references"])
    assert value["historical_gate_check_summary"]
    assert all(
        value[key] == 1
        for key in [
            "learning_operands_ready_score",
            "extraction_inputs_ready_score",
            "deployment_inputs_ready_score",
        ]
    )


def test_source_and_configuration_seal(authority, tmp_path):
    """REQ-VERIFY-8681: commands and coverage configuration enter the independent source seal."""
    raw, private = tmp_path / "raw", tmp_path / "configuration"
    private.mkdir()
    config = private / "coverage.ini"
    config.write_text("[run]\nparallel=true\n")
    e.atomic_json(
        raw / "execution_manifest.json",
        dict(commands=[], private_control=True, private_scratch=str(private)),
    )
    work = e.measure(authority, raw, "20261101")
    assert work["execution_manifest_reference"] == e.reference(raw / "execution_manifest.json")
    assert e.reference(config) in work["code_refs"]
    assert e.read(e.reference(raw / "measurement_seal.json")) == work


def test_replay_rejects_changed_original_operands(authority, tmp_path):
    """SCENARIO-VERIFY-8681-REPLAY: original seals, checkpoint arithmetic and streams stay authoritative."""
    output = tmp_path / (e.NAME + ".json")
    raw = tmp_path / "raw" / e.NAME / "invocations" / "rejection"
    work = e.measure(authority, raw, "20261101")
    checks, stream = raw / "validation.json", tmp_path / "owned.stdout"
    before = e.canonical_hash(work)
    proof = dict(study="nested_rejection_control", before=before, after=e.canonical_hash(work))
    stream.write_text("IMMUTABLE_8681 " + json.dumps(proof) + "\n")
    e.atomic_json(
        checks,
        dict(
            receipts=[
                dict(
                    name="private_child",
                    passed=True,
                    stdout_path=str(stream),
                    stdout_sha256=e.sha256_file(stream),
                )
            ]
        ),
    )

    def candidate(changed, seal=True):
        e.atomic_json(raw / "measurement.json", changed)
        if seal:
            e.atomic_json(raw / "measurement_seal.json", changed)
        value = report.build(changed, e.reference(checks), raw, output)
        e.atomic_json(output, value)
        return value

    value = candidate(work)
    assert value["immutable_build_rows"] == [proof]
    assert report.replay(output)
    replacement = raw / "replacement-validation.json"
    replacement.write_bytes(checks.read_bytes())
    e.atomic_json(output, report.build(work, e.reference(replacement), raw, output))
    assert not report.replay(output)
    candidate(work)
    changed = deepcopy(work)
    changed["numerical"]["probability"] = 0
    candidate(changed, seal=False)
    assert not report.replay(output)
    candidate(changed)
    assert not report.replay(output)
    changed = deepcopy(work)
    changed["runtime"]["inventory_reference"] = dict(
        path=str(tmp_path / "absent"), sha256="sha256:" + "0" * 64
    )
    candidate(changed)
    assert not report.replay(output)
    value = candidate(work)
    value["raw_shard_hashes"].append(
        dict(path=str(tmp_path / "absent"), sha256="sha256:" + "0" * 64)
    )
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert not report.replay(output)
    candidate(work)
    stream.write_text("changed after accepted child receipt")
    assert not report.replay(output)
    candidate(work)
    assert report.build(work, e.reference(checks), raw, output)["methods_ready_score"] == 0


def test_original_cli_cannot_publish_live():
    """REQ-VERIFY-8681: the historical adapter cannot run an unguarded original producer."""
    from carnot.reporting import v745_private_inputs as private

    with pytest.raises(SystemExit) as error:
        private.main(["--date", "20261101"])
    assert error.value.code == 2


def test_original_snapshot_resolution(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8681-STANDALONE: only intact registered snapshots authenticate."""
    from carnot.reporting import v745_private_inputs as private

    ref = dict(path=str(private.SOURCE), sha256=private.PIN)
    archive = tmp_path / "arc_competition_agent.py"
    archive.write_bytes(private.ARCHIVE.read_bytes())
    monkeypatch.setattr(original, "ARC_ARCHIVE", archive)
    monkeypatch.setattr(private, "ARCHIVE", archive)
    assert original.checked(ref)
    assert not original.checked(dict(ref, sha256="sha256:" + "0" * 64))
    assert not original.checked(dict(ref, path=str(tmp_path / "missing")))
    archive.write_bytes(b"changed retained source")
    assert not original.checked(ref)
    archive.unlink()
    assert not original.checked(ref)
    assert original.checked(original.reference(Path(__file__)))
