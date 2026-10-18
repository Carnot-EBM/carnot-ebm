"""REQ-REPORT-8641, REQ-VERIFY-8641: real child bytes authorize execution."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

from coverage import CoverageData
import pytest

from carnot.reporting import learning_custody_8641 as c


@pytest.fixture
def sealed(tmp_path):
    """Use an actual instrumented process so invented coverage cannot pass the tests."""
    source = tmp_path / "operand.py"
    source.write_text("answer = 42\nassert answer == 42\n")
    namespace = tmp_path / "child"
    namespace.mkdir()
    config = namespace / "coverage.ini"
    config.write_text("[run]\nparallel=true\ncontext=child0\ninclude=" + str(source) + "\n")
    receipt = c.supervise(
        "child0",
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config), str(source)],
        namespace,
        env={"COVERAGE_FILE": str(namespace / ".coverage")},
        deadline=30,
    )
    manifest = c.seal_child(namespace, [c.ref(source)], receipt, ["fixture::answer"], "original")
    return tmp_path, source, namespace, manifest


def test_actual_child_and_repeat_combine(sealed):
    """SCENARIO-REPORT-8641-CUSTODY: retain suffix or unsuffixed SQLite before combining."""
    root, source, namespace, manifest = sealed
    assert manifest["receipt"]["pid"] > 0 and manifest["receipt"]["passed"]
    assert manifest["files"][0]["contexts"] == ["child0"]
    assert manifest["files"][0]["lines"][str(source)] == [1, 2]
    original = deepcopy(manifest)
    aggregate = c.combine_copies([manifest], root / "aggregate")
    assert aggregate["second_combine_invariance"]["passed"]
    assert c.attest_children([manifest], [c.ref(source)]) == {str(source): [1, 2]}
    assert manifest == original
    data = next(p for p in namespace.iterdir() if p.name.startswith(".coverage"))
    data.rename(namespace / ".coverage")
    assert c.seal_child(namespace, [c.ref(source)], manifest["receipt"], [], "original")["files"]


@pytest.mark.parametrize(
    "attack", ["deleted", "overwritten", "empty", "duplicate", "foreign", "substitution"]
)
def test_custody_rejections(sealed, attack):
    """SCENARIO-REPORT-8641-CUSTODY: repaired envelope hashes cannot replace sealed bytes."""
    root, source, namespace, original = sealed
    manifest = deepcopy(original)
    path = Path(manifest["files"][0]["reference"]["path"])
    if attack == "deleted":
        path.unlink()
    elif attack == "overwritten":
        path.chmod(0o644)
        path.write_bytes(b"overwritten")
    elif attack == "empty":
        path.chmod(0o644)
        path.write_bytes(b"")
    elif attack == "duplicate":
        manifest["files"].append(deepcopy(manifest["files"][0]))
    elif attack == "foreign":
        source.write_text("foreign = True\n")
    else:
        replacement = root / "replacement"
        data = CoverageData(basename=str(replacement))
        data.add_lines({str(source): {1}})
        data.write()
        manifest["files"][0]["reference"] = c.ref(replacement)
        manifest["files"][0]["lines"] = {str(source): [1]}
    with pytest.raises((ValueError, OSError)):
        c.attest_children([manifest], [c.ref(source)])


def test_empty_foreign_and_killed_children(tmp_path):
    """REQ-VERIFY-8641: zero genuine data and killed children fail despite valid filenames."""
    receipt = c.supervise(
        "killed", [sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, deadline=0.05
    )
    assert receipt["timed_out"] and receipt["exit_code"] < 0
    with pytest.raises(ValueError, match="child"):
        c.seal_child(tmp_path, [], receipt, [], "original")
    receipt.update(passed=True, timed_out=False, exit_code=0)
    with pytest.raises(ValueError, match="coverage"):
        c.seal_child(tmp_path, [], receipt, [], "original")
    (tmp_path / ".coverage").write_bytes(b"")
    with pytest.raises(ValueError):
        c.seal_child(tmp_path, [], receipt, [], "original")
    (tmp_path / ".coverage").unlink()
    data = CoverageData(basename=str(tmp_path / ".coverage"))
    data.add_lines({str(tmp_path / "foreign.py"): {1}})
    data.write()
    with pytest.raises(ValueError, match="source"):
        c.seal_child(tmp_path, [], receipt, [], "original")


def test_old_lost_shard_fixture(tmp_path):
    """SCENARIO-REPORT-8641-CUSTODY: demonstrate the old destructive second combine."""
    proof = c.old_failure_fixture(tmp_path)
    assert proof["lost_shards"] and proof["aggregate_overwritten"]
    assert proof["first_lines"] != proof["second_lines"]


def test_cli_rejection_and_replay(tmp_path):
    """SCENARIO-VERIFY-8641-REPLAY: real private CLI and rehashed tampering are bounded."""
    from carnot.reporting import learning_execution_8641 as e

    assert e.main(["--date", "bad"]) == 2
    assert e.main(["--date", "20260230"]) == 2
    assert e.main(["--deliberate-error"]) == 7
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert not e.replay(tmp_path / "missing")
    assert e.MODEL_SPECS == []
    for args, expected in [
        (["--date", "20261019"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", str(tmp_path / "missing")], 1),
    ]:
        row = c.supervise(
            "cli" + str(expected),
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            tmp_path / "logs",
            expected=expected,
            deadline=60,
        )
        assert row["passed"]


def test_report_gates_and_source_authority(tmp_path, monkeypatch):
    """REQ-REPORT-8641: only operand gates authorize a new run; historical failure is preserved."""
    from carnot.reporting import learning_execution_8641 as e
    from carnot.reporting import learning_execution_report_8641 as report

    path = tmp_path / "upstream.json"
    path.write_text(
        json.dumps(
            dict(
                methods_ready_score=1, learning_operands_ready_score=1, required_checks_passed=False
            )
        )
    )
    monkeypatch.setattr(report, "UPSTREAM", path)
    value, rows = report.inputs()
    assert not rows and value["required_checks_passed"] is False
    for values in [
        dict(methods_ready_score=0, learning_operands_ready_score=1),
        dict(methods_ready_score=1),
    ]:
        path.write_text(json.dumps(values))
        _, rows = report.inputs()
        assert rows and rows[0]["observed"] == values.get(rows[0]["field"])
    path.unlink()
    assert report.inputs()[1][0]["observed"] is None
    assert not e.replay(tmp_path / "absent")


def test_statement_and_semantic_rejections(tmp_path):
    """REQ-VERIFY-8641: source-bound statements and real diagnostics remain separate gates."""
    from carnot.reporting import learning_execution_report_8641 as report

    source = tmp_path / "tiny.py"
    source.write_text("x = 1\ny = 2\n")
    assert report.statement_report({str(source): [1, 2]}, [c.ref(source)])["complete"]
    assert not report.statement_report({str(source): [1]}, [c.ref(source)])["complete"]
    assert report.statement_report({}, [c.ref(source)])["files"][str(source)]["missing"] == [1, 2]


def test_sealed_rejection_paths(sealed):
    """REQ-VERIFY-8641: independently re-read source statements, logs and child dispositions."""
    root, source, namespace, original = sealed
    for attack in ("receipt", "log", "duplicate", "data", "empty_statements"):
        manifest = deepcopy(original)
        if attack == "receipt":
            manifest["receipt"]["passed"] = False
        elif attack == "log":
            manifest["receipt"]["stdout_sha256"] = "wrong"
        elif attack == "duplicate":
            manifest["files"] *= 2
        elif attack == "data":
            manifest["files"][0]["lines"] = {str(source): [1]}
        else:
            empty = root / "empty_sqlite"
            data = CoverageData(basename=str(empty))
            data.add_lines({str(source): set()})
            data.write()
            with pytest.raises(ValueError, match="empty"):
                c.inventory(empty, [c.ref(source)])
            continue
        seal = root / (attack + ".json")
        seal.write_text(json.dumps({k: v for k, v in manifest.items() if k != "seal_reference"}))
        manifest["seal_reference"] = c.ref(seal)
        with pytest.raises(ValueError):
            c.attest_children([manifest], [c.ref(source)])
    with pytest.raises(ValueError):
        c.attest_children([], [])
    with pytest.raises(ValueError):
        c.read(dict(path=str(source), sha256="wrong"))
    changed = deepcopy(original)
    changed["files"][0]["reference"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="raw_changed"):
        c.combine_copies([changed], root / "changed_combine")
    Path(original["files"][0]["reference"]["path"]).unlink()
    with pytest.raises(OSError):
        c.combine_copies([original], root / "broken_combine")


@pytest.fixture
def private_work(tmp_path):
    """Synthetic controller inputs exercise dispatch and rejection, never qualify real science."""
    from carnot.reporting import learning_execution_report_8641 as r

    raw = tmp_path / "raw"
    raw.mkdir()
    return dict(
        raw=str(raw),
        run_date="20261018",
        started_monotonic_ns=__import__("time").monotonic_ns(),
        upstream=dict(methods_ready_score=1, learning_operands_ready_score=1),
        input_failures=[],
        input_references=[],
        original_authority=[],
        sources=[c.ref(c.ROOT / p) for p in r.OWNED],
        original_nodes=[f"tests/python/synthetic.py::test_{i}" for i in range(27)],
        mechanics_fixture=dict(path=str(raw / "measurement.json")),
        children=[],
        receipts=[],
        nodes=[],
        outcomes=[],
        duration_s=0,
        phase_spans=[],
        preconditions=dict(private_disk_backed_scratch=True),
    )


def test_controller_plan_and_failed_child(private_work, tmp_path, monkeypatch):
    """REQ-VERIFY-8641: collection partitions stay disjoint and killed children cannot vanish."""
    from carnot.reporting import learning_execution_8641 as e

    work = private_work
    originals = work["original_nodes"] + ["tests/python/synthetic.py::test_fixed_stream"]
    nodes = originals + [e.r.TESTS[0] + "::test_old_lost_shard_fixture"]
    actual_supervise = c.supervise
    calls = []

    def supervised(name, argv, logs, **kwargs):
        calls.append(name)
        if name == "collection":
            receipt = actual_supervise(
                name, [sys.executable, "-c", "print(" + repr("\n".join(nodes)) + ")"], logs
            )
            return receipt
        receipt = actual_supervise(name, [sys.executable, "-c", "pass"], logs)
        receipt.update(passed=False, exit_code=1)
        return receipt

    monkeypatch.setattr(c, "supervise", supervised)
    case = Path(work["raw"]) / "deadline_cases/case-3599/case.json"
    case.parent.mkdir(parents=True)
    case.write_text(json.dumps(dict(observed_elapsed_s=None)))
    monkeypatch.setattr(e.historical, "semantic_cases", lambda rows: False)
    e.execute(work)
    assert not work["union_passed"] and work["pending_nodes"] == originals
    assert len(set(n for group in work["original_groups"] for n in group)) == len(originals)
    assert work["child_errors"] and not work["custody_controls_passed"]
    e.recompute(work, tmp_path / "failed_cold")
    assert not work["union_passed"]
    work["started_monotonic_ns"] = 0
    e.execute(work)
    assert calls.count("owned_tests") == 2


def test_instrumented_real_data_and_recompute(private_work, tmp_path, monkeypatch):
    """REQ-VERIFY-8641: actual source data and JUnit are required, including cold union drift."""
    from carnot.reporting import learning_execution_8641 as e

    source = tmp_path / "source.py"
    source.write_text("value = 1\nassert value == 1\n")
    sources = [c.ref(source)]
    work = private_work

    def supervised(name, argv, logs, **kwargs):
        config = next(x.split("=", 1)[1] for x in argv if x.startswith("--rcfile="))
        receipt = c.supervise_original(
            name,
            [sys.executable, "-m", "coverage", "run", "--rcfile=" + config, str(source)],
            logs,
            **kwargs,
        )
        junit = next(x.split("=", 1)[1] for x in argv if x.startswith("--junitxml="))
        Path(junit).write_text(
            '<testsuite><testcase classname="tests.python.synthetic" name="test_source"/></testsuite>'
        )
        return receipt

    monkeypatch.setattr(c, "supervise_original", c.supervise, raising=False)
    monkeypatch.setattr(c, "supervise", supervised)
    assert e.instrumented(work, "source", ["synthetic"], "original", sources)["passed"]
    assert work["children"] and work["outcomes"][0]["outcome"] == "passed"
    monkeypatch.setattr(e.r, "RUNNER", str(source))
    monkeypatch.setattr(
        e.historical, "map_statements", lambda *a: dict(normalized_statement_rows=[])
    )
    e.recompute(work, tmp_path / "cold")
    assert work["original_coverage"]["complete"] and not work["owned_coverage"]["complete"]
    monkeypatch.setattr(e, "terminal", lambda *a: dict(passed=True))
    supplied = tmp_path / "supplied.json"
    supplied.write_text(json.dumps(work))
    assert e.run(tmp_path / (e.r.NAME + ".json"), "20261018", supplied) == 0
    broken = tmp_path / "drift_data"
    data = CoverageData(basename=str(broken))
    data.add_lines({str(source): {1}})
    data.write()
    with monkeypatch.context() as local:
        local.setattr(c, "combine_copies", lambda *a: dict(reference=c.ref(broken)))
        with pytest.raises(ValueError, match="union_drift"):
            e.recompute(work, tmp_path / "cold_drift")
    work["semantics_passed"] = True
    work["deadline_case_rows"] = []
    with pytest.raises(ValueError, match="semantics"):
        e.recompute(work, tmp_path / "cold2")
    work["sources"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="source"):
        e.recompute(work, tmp_path / "cold3")


def test_private_publication_and_rehashed_replay(private_work, tmp_path):
    """SCENARIO-VERIFY-8641-REPLAY: actual CLI cold-replays immutable seals and rejects tampering."""
    from carnot.reporting import learning_execution_8641 as e

    work = private_work
    raw = Path(work["raw"])
    path = raw / "supplied.json"
    path.write_text(json.dumps(work))
    output = tmp_path / (e.r.NAME + ".json")
    receipt = c.supervise(
        "private",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--private-output",
            str(output),
            "--work",
            str(path),
        ],
        tmp_path / "logs",
        deadline=90,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["learning_execution_ready_score"] == 0 and e.replay(output)
    primitive = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    primitive["children"] = [dict(invented=True)]
    Path(value["primitive_reference"]["path"]).write_text(json.dumps(primitive))
    value["primitive_reference"] = c.ref(Path(value["primitive_reference"]["path"]))
    value["reproducibility_checksum"] = e.r.digest(value)
    output.write_text(json.dumps(value))
    assert not e.replay(output)
    assert c.supervise(
        "tampered",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(output)],
        tmp_path / "logs",
        expected=1,
        deadline=60,
    )["passed"]


def test_prepare_independent_inputs(tmp_path, monkeypatch, private_work):
    """REQ-REPORT-8641: failed historical execution does not veto authenticated numerical work."""
    from carnot.reporting import learning_execution_8641 as e

    base = dict(
        failures=[],
        preconditions={},
        source_refs=[],
        original_nodes=private_work["original_nodes"],
        mechanics_fixture={},
        checkpoint_reference={},
    )
    monkeypatch.setattr(e.historical.base, "prepare", lambda raw: base)
    primary = tmp_path / "methods.json"
    primary.write_text("{}")
    sidecar = tmp_path / "terminal.json"
    sidecar.write_text(
        json.dumps(dict(publication=dict(sidecar_path=str(tmp_path / "bound.json"))))
    )
    upstream = dict(
        methods_ready_score=1,
        learning_operands_ready_score=1,
        required_checks_passed=False,
        terminal_validation_sidecar_path=str(sidecar),
    )
    monkeypatch.setattr(e.r, "UPSTREAM", primary)
    monkeypatch.setattr(e.r, "inputs", lambda: (upstream, []))
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=True)))
    work = e.prepare(tmp_path / "prepared", "20261018")
    assert not work["input_failures"] and work["numerical_inference"]["evaluated_this_invocation"]
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    monkeypatch.setattr(e.r, "TEST_PINS", {"test_learning_clock_8592.py": "wrong"})
    monkeypatch.setattr(e.historical, "PIN_SOURCE", "wrong")
    work = e.prepare(tmp_path / "rejected", "20261018")
    assert len(work["input_failures"]) == 3


def test_dispatch_and_run_validation(private_work, tmp_path, monkeypatch):
    """REQ-VERIFY-8641: normal dispatch executes one controller and retains failed command receipts."""
    from carnot.reporting import learning_execution_8641 as e

    work = private_work
    calls = []
    monkeypatch.setattr(e, "prepare", lambda *a: work)
    monkeypatch.setattr(e, "execute", lambda w: calls.append(w["raw"]))
    actual = c.supervise
    monkeypatch.setattr(
        c,
        "supervise",
        lambda name, argv, logs, **kw: actual(name, [sys.executable, "-c", "pass"], logs),
    )
    monkeypatch.setattr(e, "terminal", lambda *a: dict(passed=True))
    output = tmp_path / (e.r.NAME + ".json")
    assert e.run(output, "20261018") == 0
    assert calls == [work["raw"]] and len(work["receipts"]) == 4
    assert e.main(["--private-output", str(e.ROOT / "bad")]) == 2
    monkeypatch.setattr(e, "run", lambda *a: 0)
    assert e.main(["--date", "20261018"]) == 0


def test_cold_collection_and_receipt_mutations(private_work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8641-REPLAY: JUnit, collection and log changes are independently rejected."""
    from carnot.reporting import learning_execution_8641 as e

    work = private_work
    raw = Path(work["raw"])
    collection = raw / "collection.json"
    collection.write_text(json.dumps(dict(nodes=[], authority=[])))
    work["collection_reference"] = c.ref(collection)
    work["nodes"] = ["different"]
    with pytest.raises(ValueError, match="collection"):
        e.recompute(work, tmp_path / "cold_bad_collection")
    work["nodes"] = []
    junit = raw / "nodes.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.python.synthetic" name="test_case"/></testsuite>'
    )
    work["children"] = [
        dict(kind="unmeasured", node_ids=[], receipt=dict(junit_reference=c.ref(junit)))
    ]
    with pytest.raises(ValueError, match="assignment"):
        e.recompute(work, tmp_path / "cold_bad_assignment")
    work["children"][0]["node_ids"] = ["tests/python/synthetic.py::test_case"]
    e.recompute(work, tmp_path / "cold_actual")
    assert work["outcomes"][0]["passed"]
    junit.write_text("changed")
    with pytest.raises(ValueError, match="junit"):
        e.recompute(work, tmp_path / "cold_bad_junit")
    work["children"] = []
    work.pop("collection_reference")
    work["receipts"] = [c.supervise("proof", [sys.executable, "-c", "pass"], raw / "logs")]
    operand = raw / "input.json"
    operand.write_text("{}")
    work["input_references"] = [c.ref(operand)]
    seal = raw / "invocation_seal.json"
    seal.write_text(json.dumps(e.invocation_binding(work)))
    work["invocation_seal_reference"] = c.ref(seal)
    primitive = raw / "work.json"
    primitive.write_text(json.dumps(work))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(e.r.build(work, tmp_path / (e.r.NAME + ".json"))))
    assert e.replay(candidate)
    Path(work["receipts"][0]["stdout_path"]).write_text("changed")
    assert not e.replay(candidate)


def test_terminal_consumers(private_work, tmp_path):
    """REQ-VERIFY-8641: run unchanged scanners and the cold CLI with retained severities."""
    from carnot.reporting import learning_execution_8641 as e

    work = private_work
    raw = Path(work["raw"])
    supplied = raw / "supplied.json"
    supplied.write_text(json.dumps(work))
    output = tmp_path / (e.r.NAME + ".json")
    assert e.run(output, "20261018", supplied) == 0
    assert e.terminal(output, tmp_path / "consumer_repeat")["passed"]


def test_empty_auxiliary_shards_do_not_authorize_execution(sealed):
    """SCENARIO-REPORT-8641-CUSTODY: inventory empty helper shards but require real child statements."""
    root, source, namespace, original = sealed
    mixed = root / "mixed"
    mixed.mkdir()
    for path in namespace.iterdir():
        if path.is_file() and path.name.startswith(".coverage"):
            shutil.copyfile(path, mixed / path.name)
    namespace = mixed
    empty = CoverageData(basename=str(namespace / ".coverage.zero"))
    empty.add_lines({str(source): set()})
    empty.write()
    manifest = c.seal_child(
        namespace, [c.ref(source)], original["receipt"], ["fixture::answer"], "original"
    )
    assert len(manifest["files"]) == 2
    assert c.attest_children([manifest], [c.ref(source)]) == {str(source): [1, 2]}
    zero = root / "zero"
    zero.mkdir()
    shutil.copyfile(namespace / ".coverage.zero", zero / ".coverage.zero")
    with pytest.raises(ValueError, match="zero_real"):
        c.seal_child(zero, [c.ref(source)], original["receipt"], [], "original")


def test_original_assertion_scope_is_inventoried(private_work, tmp_path, monkeypatch):
    """REQ-VERIFY-8641: numerical-only children retain real assertion data; runner remains83 statements."""
    from carnot.reporting import learning_execution_8641 as e

    work = private_work
    source = tmp_path / "original_assertions.py"
    source.write_text("assert 2 + 2 == 4\n")
    work["original_authority"] = [c.ref(source)]
    calls = []
    actual = c.supervise

    def collection(name, argv, logs, **kwargs):
        return actual(
            name, [sys.executable, "-c", "print('tests/python/synthetic.py::test_scope')"], logs
        )

    def instrumented(w, name, nodes, kind, sources):
        calls.append((kind, sources))
        return dict(passed=False)

    monkeypatch.setattr(c, "supervise", collection)
    monkeypatch.setattr(e, "instrumented", instrumented)
    e.execute(work)
    original = next(sources for kind, sources in calls if kind == "original")
    assert c.ref(source) in original
    assert c.ref(e.ROOT / e.r.RUNNER) in original
