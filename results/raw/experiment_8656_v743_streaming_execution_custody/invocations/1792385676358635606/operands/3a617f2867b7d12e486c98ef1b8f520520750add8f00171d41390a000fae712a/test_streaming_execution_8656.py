"""REQ-VERIFY-8656: role inventories keep historical and current execution separate."""

from copy import deepcopy
import json
from pathlib import Path

from coverage import CoverageData
import pytest

from carnot.reporting import streaming_adapter_8656 as a


def database(path, source, lines=(1,)):
    """Retain real SQLite statements and context rather than a fabricated filename."""
    data = CoverageData(basename=str(path))
    data.set_context("actual-child")
    data.add_lines({str(source): set(lines)})
    data.write()


def test_role_capture_and_two_combinations(tmp_path):
    """SCENARIO-VERIFY-8656-CAPTURE: exact current data survives both combines."""
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    role = a.role_inventory("suite", "current_adapter", [source], tmp_path / "raw")
    base = tmp_path / ".coverage"
    database(base, source)
    shards = a.capture(base, tmp_path / "raw", role)
    manifests = [dict(role=role, shards=shards)]
    before = deepcopy(shards)
    base.unlink()
    for i in range(2):
        report = a.combine(manifests, "current_adapter", [source], tmp_path / str(i))
        assert report["passed"] and report["statement_count"] == 1
    assert shards == before
    assert shards[0]["inventory"][0]["contexts"] == {"1": ["actual-child"]}
    assert a.c.authenticate(shards[0]["reference"])


@pytest.mark.parametrize("mode", ["missing", "empty", "foreign", "source", "overwritten"])
def test_capture_refuses_invalid_child(tmp_path, mode):
    """SCENARIO-VERIFY-8656-CAPTURE: no empty or unrelated source waiver exists."""
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    role = a.role_inventory("suite", "streaming_producer", [source], tmp_path / "raw")
    base = tmp_path / ".coverage"
    if mode != "missing":
        database(
            base,
            source if mode != "foreign" else tmp_path / "foreign.py",
            () if mode == "empty" else (1,),
        )
    if mode == "source":
        source.write_text("changed = 2\n")
    if mode == "overwritten":
        base.write_bytes(b"not SQLite")
    with pytest.raises((ValueError, a.DataError)):
        a.capture(base, tmp_path / "raw", role)


def test_consumer_and_historical_data_cannot_supply_current_coverage(tmp_path):
    """REQ-REPORT-8656: consumer statements qualify only the consumer's own work."""
    source = tmp_path / "consumer.py"
    source.write_text("value = 1\n")
    role = a.role_inventory("consumer", "historical_consumer", [source], tmp_path / "raw")
    base = tmp_path / ".coverage"
    database(base, source)
    shards = a.capture(base, tmp_path / "raw", role)
    assert shards
    with pytest.raises(ValueError, match="source_role"):
        a.combine([dict(role=role, shards=shards)], "current_adapter", [source], tmp_path / "c")
    with pytest.raises(ValueError, match="source_role"):
        a.role_inventory("bad", "unknown", [source], tmp_path / "raw")


def test_pinned_history_and_private_views(tmp_path):
    """SCENARIO-REPORT-8656-AUTHORITY: original authority survives current drift."""
    history = a.authority(a.ROOT, tmp_path / "raw")
    assert not history["gates"]
    assert len(history["failed_owned_test_identities"]) == 2
    assert history["driver_mismatch"]["expected"] != history["driver_mismatch"]["observed"]
    assert all(a.c.authenticate(r["reference"]) for r in history["historical"])
    assert a.private_command(history["views"]["8629"], ["true"], tmp_path)[-1] == "true"
    missing = a.authority(tmp_path / "missing", tmp_path / "empty")
    assert missing["gates"][0]["observed"] is None
    assert missing["gates"][0]["missing_reason"] == "missing_file"


@pytest.mark.parametrize(
    "value,reason",
    [
        ({}, "missing_field"),
        ({"methods_ready_score": 0}, "observed_zero"),
        ({"methods_ready_score": 1}, None),
    ],
)
def test_gate_distinguishes_absent_field_and_zero(tmp_path, value, reason):
    """REQ-REPORT-8656: an observed numeric zero is never a missing operand."""
    path = tmp_path / "upstream.json"
    path.write_text(json.dumps(value))
    row = a.gate(path, "methods_ready_score")
    assert row["missing_reason"] == reason
    assert row["passed"] is (reason is None)


def qualified_validation(raw=None):
    """Builder tests supply explicit validation; real outer children own execution evidence."""
    validation = dict(
        receipts=[dict(name="test_boundary", passed=True)],
        original_node_union=dict(passed=True),
        current_node_union=dict(passed=True),
        coverage=dict(passed=True),
        child_shard_manifest=[],
        legacy_failures_reproduced=True,
    )
    if raw is not None:
        from carnot.reporting import streaming_execution_8656 as e

        return e.seal_validation(validation, raw)
    return validation


def test_measure_build_replay_and_unchanged_history(tmp_path):
    """SCENARIO-REPORT-8656-PUBLICATION: sealed measurements produce only readiness."""
    from carnot.reporting import streaming_execution_8656 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261019")
    ref = e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(ref, qualified_validation(tmp_path / "raw"), [], {}, output)
    e.atomic_json(output, value)
    assert value["streaming_extractor_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert e.replay(output)
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert len(value["rows"]) == 33
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert a.authority(e.ROOT, tmp_path / "recheck")["historical"]


@pytest.mark.parametrize("mode", ["missing", "feature", "offset", "mixed", "summary", "anchor"])
def test_rehashed_tampering_is_rejected(tmp_path, mode):
    """SCENARIO-VERIFY-8656-REPLAY: repaired derived hashes cannot change semantic authority."""
    from carnot.reporting import streaming_execution_8656 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261019")
    output = tmp_path / (e.NAME + ".json")
    if mode == "feature":
        work["fixture"]["chunk_rows"][9]["features"]["values"] = [999] * 8
    if mode == "offset":
        row = work["fixture"]["chunk_rows"][9]
        bad = tmp_path / "offset.jsonl"
        bad.write_bytes(
            Path(row["primitive"]["path"]).read_bytes().replace(b'"start":0', b'"start":1', 1)
        )
        row["primitive"] = a.c.freeze(bad, tmp_path / "raw")
    if mode == "mixed":
        work["semantic_source_identity_rows"][0]["current_sha256"] = "sha256:bad"
    ref = e.seal(work)
    value = e.build(ref, qualified_validation(tmp_path / "raw"), [], {}, output)
    if mode == "missing":
        value["immutable_measurement_reference"]["path"] = str(tmp_path / "absent")
    if mode == "summary":
        value["streaming_extractor_ready_score"] = 999
    if mode == "anchor":
        work["duration_s"] += 123
        replacement = tmp_path / "replacement" / "measurement.json"
        e.atomic_json(replacement, work)
        value = e.build(
            a.c.freeze(replacement, tmp_path / "raw"),
            qualified_validation(tmp_path / "raw"),
            [],
            {},
            output,
        )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = e.canonical_hash(value)
    e.atomic_json(output, value)
    assert not e.replay(output)


def test_external_block_and_owned_failure(tmp_path):
    """REQ-REPORT-8656: unavailable future methods block independently of owned validation."""
    from carnot.reporting import streaming_execution_8656 as e

    work = e.measure(tmp_path / "missing", tmp_path / "raw", "20261019")
    ref = e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    validation = qualified_validation(tmp_path / "raw")
    value = e.build(ref, validation, [], {}, output)
    assert value["verdict_class"] == "blocked" and len(value["rows"]) == 33
    assert value["gate_check_summary"][0]["observed"] is None
    e.atomic_json(output, value)
    assert e.replay(output)
    validation["receipts"][0]["passed"] = False
    assert e.build(ref, validation, [], {}, output)["verdict_class"] == "disqualified"


def test_real_private_cli_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8656-PUBLICATION: actual CLI accepts dates and validates exact bytes."""
    import os
    import subprocess
    import sys
    from carnot.reporting import streaming_execution_8656 as e
    from carnot.reporting import streaming_execution_runner_8656 as runner

    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    measured = subprocess.run(
        cli + ["--date", "20261019", "--measure-only", str(tmp_path / "work.json")],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert measured.returncode == 0, measured.stderr.decode()
    assert (
        subprocess.run(cli + ["--date", "20261399"], capture_output=True, timeout=30).returncode
        == 2
    )
    assert (
        subprocess.run(
            cli + ["--cold-replay", str(tmp_path / "absent")], capture_output=True, timeout=30
        ).returncode
        == 1
    )
    monkeypatch.setattr(runner, "run_checks", lambda *_: qualified_validation())
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261019", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert e.replay(output) and value["streaming_extractor_ready_score"] == 1
    assert Path(value["terminal_validation_sidecar_path"]).is_file()


def test_failed_owned_publication(tmp_path, monkeypatch):
    """REQ-REPORT-8656: a failed child publishes an honest zero, never readiness."""
    from carnot.reporting import streaming_execution_8656 as e
    from carnot.reporting import streaming_execution_runner_8656 as runner

    validation = qualified_validation()
    validation["receipts"][0]["passed"] = False
    monkeypatch.setattr(runner, "run_checks", lambda *_: validation)
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261019", "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


@pytest.mark.parametrize("capture_failure", [False, True])
def test_child_boundary_preserves_real_statements(tmp_path, monkeypatch, capture_failure):
    """REQ-VERIFY-8656: the next child cannot overwrite uncaptured data."""
    import os
    from carnot.reporting import streaming_execution_runner_8656 as runner

    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    suite = "tests/python/test_probe.py"
    node = suite + "::test_one"

    def child(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        stdout = logs / (name + ".stdout")
        stdout.write_text(node if name.startswith("collect_") else "actual child")
        if not name.startswith("collect_"):
            if not capture_failure:
                database(Path(os.environ["COVERAGE_FILE"]), source)
            junit = Path(next(s.split("=", 1)[1] for s in argv if s.startswith("--junitxml=")))
            junit.write_text(
                '<testsuite><testcase classname="tests.python.test_probe" name="test_one"/></testsuite>'
            )
        return dict(name=name, passed=True, normal_exit=True, stdout_path=str(stdout))

    monkeypatch.setattr(runner, "child", child)
    reference = a.c.freeze(source, tmp_path / "raw")
    receipts, manifests, plan = runner.child_suite(
        suite,
        "current_adapter",
        [str(source)],
        [dict(origin_path=str(source), reference=reference)],
        tmp_path / "raw",
        tmp_path,
        "probe",
    )
    assert len(receipts) == 2 and len(manifests) == 1 and len(plan) == 2
    assert manifests[0]["passed"] is (not capture_failure)
    assert manifests[0]["node_ids"] == [node]


def test_coverage_roles_and_lost_shard(tmp_path):
    """SCENARIO-VERIFY-8656-CAPTURE: combined current coverage rejects lost evidence."""
    from carnot.reporting import streaming_execution_runner_8656 as runner

    manifests = []
    for label, role_name in [
        ("current8656", "current_adapter"),
        ("current8629", "streaming_producer"),
    ]:
        source = tmp_path / (
            "streaming_evidence.py" if role_name == "streaming_producer" else label + ".py"
        )
        source.write_text("value = 1\n")
        role = a.role_inventory(label, role_name, [source], tmp_path / "raw")
        base = tmp_path / (label + ".coverage")
        database(base, source)
        manifests.append(
            dict(name=label + "_0", role=role, shards=a.capture(base, tmp_path / "raw", role))
        )
    assert runner.coverage_gates(manifests, tmp_path / "raw", tmp_path)["passed"]
    Path(manifests[0]["shards"][0]["reference"]["path"]).unlink()
    assert not runner.coverage_gates(manifests, tmp_path / "raw", tmp_path / "again")["passed"]


def test_real_historical_failure_reproduction(tmp_path):
    """SCENARIO-REPORT-8656-AUTHORITY: real old tests still fail with the changed driver."""
    from carnot.reporting import streaming_execution_runner_8656 as runner

    history = a.authority(a.ROOT, tmp_path / "raw")
    result = runner.reproduce_failures(history, tmp_path / "raw", tmp_path)
    assert result["passed"]
    assert result["consumer_missing_statements"]
    assert result["failed_test_identities"] == history["failed_owned_test_identities"]


@pytest.mark.parametrize("failed_child", [False, True])
def test_execution_orchestration_retains_disjoint_outcomes(tmp_path, monkeypatch, failed_child):
    """REQ-VERIFY-8656: original and current unions preserve failed child status."""
    from carnot.reporting import streaming_execution_runner_8656 as runner

    calls = []

    def suite(suite, role, sources, view, raw, scratch, label):
        calls.append(label)
        node = suite + "::test_one"
        junit = scratch / (label + ".xml")
        classname = suite.removesuffix(".py").replace("/", ".")
        junit.write_text(
            f'<testsuite><testcase classname="{classname}" name="test_one"/></testsuite>'
        )
        inventory = a.role_inventory(suite, role, [a.ROOT / sources[0]], raw)
        receipt = dict(name=label, passed=not failed_child)
        return (
            [receipt],
            [
                dict(
                    name=label + "_0",
                    role=inventory,
                    node_ids=[node],
                    shards=[],
                    passed=not failed_child,
                )
            ],
            [
                dict(name="collect_" + label, collected_node_ids=[node]),
                dict(name=label, collected_node_ids=[node], junit_path=str(junit)),
            ],
        )

    def command(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / (name + ".log")
        path.write_text("actual tool boundary")
        return dict(name=name, passed=True, stdout_path=str(path))

    monkeypatch.setattr(runner, "child_suite", suite)
    monkeypatch.setattr(runner, "child", command)
    monkeypatch.setattr(runner, "coverage_gates", lambda *_: dict(passed=True))
    monkeypatch.setattr(runner, "reproduce_failures", lambda *_: dict(passed=True))
    monkeypatch.setattr(a.c, "retention_regression", lambda *_: dict(passed=True))
    result = runner.run_checks(tmp_path / "raw", {})
    assert len(calls) == 8 and result["scratch_cleanup"]
    assert result["original_node_union"]["passed"] and result["current_node_union"]["passed"]
    assert all(m["passed"] is (not failed_child) for m in result["child_shard_manifest"])
    assert all(a.c.authenticate(ref) for ref in result["immutable_refs"])


def test_missing_historical_authority_never_collects_all_tests(tmp_path):
    """REQ-VERIFY-8656: missing historical nodes cannot launch an unbounded pytest scan."""
    from carnot.reporting import streaming_execution_runner_8656 as runner

    result = runner.reproduce_failures({}, tmp_path / "raw", tmp_path)
    assert not result["passed"] and result["missing_reason"] == "missing_historical_authority"


def test_separately_sealed_validation_rejects_replacement(tmp_path):
    """SCENARIO-VERIFY-8656-REPLAY: a repaired validation hash cannot replace invocation evidence."""
    from carnot.reporting import streaming_execution_8656 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261019")
    ref = e.seal(work)
    validation = qualified_validation(tmp_path / "raw")
    validation["current_node_union"]["passed"] = False
    replacement = tmp_path / "replacement" / "validation.json"
    e.atomic_json(replacement, {k: v for k, v in validation.items() if k != "seal_reference"})
    validation["seal_reference"] = a.c.freeze(replacement, tmp_path / "raw")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(ref, validation, [], {}, output)
    e.atomic_json(output, value)
    assert not e.replay(output)


@pytest.mark.parametrize(
    "mode",
    [
        "operand",
        "authority",
        "historical",
        "roster",
        "role",
        "shard",
        "inventory",
        "validation",
        "validation_operand",
    ],
)
def test_cold_replay_checks_each_custody_boundary(tmp_path, mode):
    """SCENARIO-VERIFY-8656-REPLAY: each independent custody operand has a rejection path."""
    from carnot.reporting import streaming_execution_8656 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261019")
    source = tmp_path / "consumer.py"
    source.write_text("value = 1\n")
    role = a.role_inventory("consumer", "historical_consumer", [source], tmp_path / "raw")
    base = tmp_path / ".coverage"
    database(base, source)
    shards = a.capture(base, tmp_path / "raw", role)
    validation = qualified_validation()
    validation["child_shard_manifest"] = [
        dict(name="consumer0_0", role=role, shards=shards, passed=True)
    ]
    if mode == "operand":
        work["immutable_refs"][0]["sha256"] = "sha256:bad"
    elif mode == "authority":
        work["current_authority"][0]["origin_path"] = str(source)
    elif mode == "historical":
        work["history"]["historical"][0]["reference"]["sha256"] = "sha256:bad"
    elif mode == "roster":
        work["fixture"]["chunk_rows"].pop()
    elif mode == "role":
        validation["child_shard_manifest"][0]["role"]["role"] = "streaming_producer"
    elif mode == "shard":
        shards[0]["reference"]["sha256"] = "sha256:bad"
    elif mode == "inventory":
        shards[0]["inventory"][0]["lines"] = [999]
    elif mode == "validation_operand":
        validation["immutable_refs"] = [dict(path=str(tmp_path / "absent"), sha256="sha256:bad")]
    ref = e.seal(work)
    validation = e.seal_validation(validation, tmp_path / "raw")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(ref, validation, [], {}, output)
    if mode == "validation":
        value["validation_evidence"]["coverage"]["passed"] = False
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = e.canonical_hash(value)
    e.atomic_json(output, value)
    assert not e.replay(output)


def test_cold_replay_retains_valid_consumer_statements(tmp_path):
    """REQ-REPORT-8656: nonempty consumer statements survive replay without producer data."""
    from carnot.reporting import streaming_execution_8656 as e

    work = e.measure(e.ROOT, tmp_path / "raw", "20261019")
    source = tmp_path / "consumer.py"
    source.write_text("value = 1\n")
    role = a.role_inventory("consumer", "historical_consumer", [source], tmp_path / "raw")
    base = tmp_path / ".coverage"
    database(base, source)
    validation = qualified_validation()
    validation["child_shard_manifest"] = [
        dict(
            name="consumer0_0",
            role=role,
            shards=a.capture(base, tmp_path / "raw", role),
            passed=True,
        )
    ]
    ref = e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    e.atomic_json(
        output, e.build(ref, e.seal_validation(validation, tmp_path / "raw"), [], {}, output)
    )
    assert e.replay(output)


def test_pinned_history_rejections_and_missing_mount(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8656-AUTHORITY: original measurement and driver pins are independent."""
    with monkeypatch.context() as patcher:
        patcher.setattr(a.c, "authenticate", lambda _: False)
        with pytest.raises(ValueError, match="historical_measurement"):
            a.authority(a.ROOT, tmp_path / "measurement")
    with monkeypatch.context() as patcher:
        patcher.setattr(a, "ORIGINAL_DRIVER", "sha256:bad")
        with pytest.raises(ValueError, match="historical_driver"):
            a.authority(a.ROOT, tmp_path / "driver")
    with pytest.raises(ValueError, match="historical_source"):
        a.private_command(
            [
                dict(
                    origin_path="absent",
                    reference=dict(path=str(tmp_path / "absent"), sha256="sha256:bad"),
                )
            ],
            ["true"],
            tmp_path,
        )


@pytest.mark.parametrize("mode", ["source", "closure"])
def test_independent_original_source_pins(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8656-AUTHORITY: original source closures cannot inherit repaired hashes."""
    if mode == "source":
        original = a.sha256_file
        monkeypatch.setattr(
            a,
            "sha256_file",
            lambda p: (
                "sha256:bad"
                if str(a.FROZEN) in str(p) and p.name == "streaming_evidence.py"
                else original(p)
            ),
        )
    else:
        original = a.c.authenticate
        monkeypatch.setattr(
            a.c,
            "authenticate",
            lambda r: original(r) if Path(r["path"]).name == "measurement.json" else False,
        )
    with pytest.raises(
        ValueError, match="historical_source_pin" if mode == "source" else "historical_closure"
    ):
        a.authority(a.ROOT, tmp_path / "raw")
