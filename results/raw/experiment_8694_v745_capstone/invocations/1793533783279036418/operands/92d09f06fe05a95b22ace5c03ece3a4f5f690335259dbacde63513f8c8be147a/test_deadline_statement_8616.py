"""REQ-REPORT-8616, REQ-VERIFY-8616: source accounting cannot replace semantics."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType
from unittest.mock import patch

import pytest

from carnot.reporting import deadline_statement_8616 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Read immutable authority once; no historical controller or training is run."""
    raw = tmp_path_factory.mktemp("statement8616") / "raw"
    supplied = os.environ.get("CARNOT_8582_MEASUREMENT")
    current = Path(supplied).parent.parent / "work.json" if supplied else None
    if current and current.is_file() and "8616" in str(current):
        work = json.loads(current.read_bytes())
        work.update(
            raw=str(raw),
            started_monotonic_ns=time.monotonic_ns(),
            receipts=[],
            node_union={},
            coverage_reference=None,
            current_mapping={},
            shards=[],
            deadline_case_rows=[],
            semantic_mutation_rows=[],
            junit_references=[],
        )
        work["current_source_reference"] = [
            e.ref(Path(r["path"])) for r in work["current_source_reference"]
        ]
        atomic_json(raw / "work.json", work)
        return work
    return e.prepare(raw)


def test_historical_mapping_and_rejections(prepared, tmp_path):
    """SCENARIO-REPORT-8616-MAPPING: retain94 while proving executed statement92."""
    evidence = prepared["statement_diagnosis"]
    fresh = e.prepare(tmp_path / "fresh-source-preconditions")
    assert fresh["statement_diagnosis"] == evidence
    assert evidence["coverage_version"] == e.coverage.__version__
    assert evidence["historical_raw_statement_count"] == 83
    assert evidence["historical_projected_missing"] == [94]
    assert evidence["raw_statement_anchor_rows"][-1]["raw_anchor"] == 94
    row = evidence["normalized_statement_rows"][-1]
    assert row == dict(raw_anchor=94, normalized_statement=92, executed=True)
    source = e.ref(e.ROOT / e.RUNNER)
    report = e.read(evidence["raw_coverage_reference"])
    mapped = e.map_statements(source, report)
    assert mapped["normalized_statement_rows"][-1] == row
    report["files"][e.RUNNER]["executed_lines"].remove(92)
    with pytest.raises(ValueError, match="unexecuted"):
        e.map_statements(source, report)
    altered = tmp_path / "altered.py"
    altered.write_bytes((e.ROOT / e.RUNNER).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source"):
        e.map_statements(e.ref(altered), report)
    with pytest.raises(ValueError, match="statement"):
        e.map_statements(source, e.read(evidence["raw_coverage_reference"]), anchors=[1])


@pytest.mark.parametrize("elapsed", [3599.0, 3600.0, 3601.0])
def test_real_cases(prepared, tmp_path, elapsed):
    """SCENARIO-VERIFY-8616-CONTROLS: execute original predicates and diagnostics."""
    root = Path(os.environ.get("CARNOT_8616_CASE_ROOT", str(tmp_path)))
    path = root / f"case-{elapsed:g}" / "case.json"
    row = (
        json.loads(path.read_bytes())
        if path.is_file()
        else e.edge.deadline_case(
            prepared["mechanics_fixture"], root / f"case-{elapsed:g}", elapsed
        )
    )
    assert row["passed"] and row["normal_cleanup"] and row["kernel_clock_unchanged"]
    assert row["clock_reads"] == (2 if elapsed < 3600 else 3)
    assert row["observed_elapsed_s"] == (None if elapsed < 3600 else elapsed + 0.25)
    assert e.semantic_cases([row], complete=False)


@pytest.mark.parametrize("diagnostic,reads", [("3600.25", 2), ("deadline_clock() - start + 1", 3)])
def test_actual_removed_read_mutant(prepared, tmp_path, diagnostic, reads):
    """SCENARIO-VERIFY-8616-CONTROLS: statement92 cannot prove diagnostic semantics."""
    original = (e.ROOT / e.RUNNER).read_text()
    mutant = original.replace('"<3600", deadline_clock() - start', '"<3600", ' + diagnostic)
    assert mutant != original
    path = tmp_path / "mutant.py"
    path.write_text(mutant)
    namespace = dict(vars(e.edge.old.runner))
    exec(compile(mutant, str(path), "exec"), namespace)
    clock = e.edge.DeadlineClock([50.0, 3650.0, 3650.25])
    captured = []
    lines = []

    class SealedError(Exception):
        pass

    def seal(path, value):
        if path.name == "measurement.json":
            captured.append(deepcopy(value))
            raise SealedError
        atomic_json(path, value)

    previous_trace = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code.co_filename == str(path) and event == "line":
            lines.append(frame.f_lineno)
        return trace

    try:
        sys.settrace(trace)
        with (
            patch.object(e.edge.old.runner, "deadline_clock", clock),
            patch.object(e.edge.old.runner, "atomic_json", seal),
        ):
            with pytest.raises(SealedError):
                FunctionType(namespace["run"].__code__, vars(e.edge.old.runner))(
                    tmp_path / "output.json",
                    private=True,
                    measurement=Path(prepared["mechanics_fixture"]["path"]),
                )
    finally:
        sys.settrace(previous_trace)
    assert 92 in lines and clock.reads == reads
    observed = [r for r in captured[0]["failures"] if r["field"] == "measurement_deadline"][0][
        "observed"
    ]
    assert clock.reads != 3 or observed != 3600.25
    supplied = os.environ.get("CARNOT_8582_MEASUREMENT")
    evidence = (
        Path(supplied).parent.parent / "semantic_mutations"
        if supplied and "8616" in supplied
        else tmp_path
    )
    atomic_json(
        evidence / ("removed_third_read.json" if reads == 2 else "changed_diagnostic.json"),
        dict(
            rejected=True,
            statement92_executed=92 in lines,
            clock_reads=clock.reads,
            observed_elapsed_s=observed,
        ),
    )


def test_semantic_mutants(prepared, tmp_path):
    """SCENARIO-VERIFY-8616-CONTROLS: missing expiration and diagnostic drift fail."""
    rows = e.read(prepared["historical_work_reference"])["deadline_case_rows"]
    assert e.semantic_cases(rows)
    assert not e.semantic_cases(rows[:1])
    changed = deepcopy(rows)
    changed[-1]["observed_elapsed_s"] = 3600
    assert not e.semantic_cases(changed)
    changed = deepcopy(rows)
    changed[-1]["clock_reads"] = 2
    assert not e.semantic_cases(changed)


def test_private_cli_and_rehashed_controls(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8616: real private publication and cold rejection retain exits."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private",
        [
            *argv,
            "--private-output",
            str(output),
            "--work",
            str(Path(prepared["raw"]) / "work.json"),
        ],
        tmp_path / "logs",
        deadline=180,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert e.replay(output)
    assert json.loads(output.read_bytes())["learning_execution_ready_score"] == 0
    for name, args, expected in [
        ("missing", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("owned", ["--deliberate-error"], 7),
        ("date", ["--date", "bad"], 2),
        ("unsafe", ["--private-output", str(e.ROOT / "bad")], 2),
    ]:
        assert child(name, [*argv, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]
    value = json.loads(output.read_bytes())
    value["learning_execution_ready_score"] = 1
    value["reproducibility_checksum"] = e.base.digest(value)
    atomic_json(tmp_path / "tampered.json", value)
    assert child(
        "tampered",
        [*argv, "--cold-replay", str(tmp_path / "tampered.json")],
        tmp_path / "logs",
        expected=1,
        deadline=60,
    )["passed"]
    assert e.main(["--date", "20260230"]) == 2
    monkeypatch.setattr(e, "run", lambda *a, **k: 0)
    assert e.main(["--date", "20261017"]) == 0
    assert e.main(["--deliberate-error"]) == 7
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_owned_evidence_rejection(prepared, tmp_path):
    """REQ-VERIFY-8616: repaired envelopes cannot bless failed evidence."""
    assert not e.qualified(prepared)
    assert not e.replay(tmp_path / "missing")
    for attack in ("diagnosis", "authority", "source", "failed"):
        work = deepcopy(prepared)
        if attack == "diagnosis":
            work["statement_diagnosis"]["normalized_statement_rows"][-1]["normalized_statement"] = (
                94
            )
        elif attack == "authority":
            work["original_authority"] = {}
        elif attack == "source":
            work["current_source_reference"][0]["sha256"] = "wrong"
        else:
            work["receipts"] = [dict(passed=False)]
        atomic_json(tmp_path / "work.json", work)
        value = e.build(prepared, tmp_path / (e.NAME + ".json"))
        value["primitive_reference"] = e.ref(tmp_path / "work.json")
        value["reproducibility_checksum"] = e.base.digest(value)
        atomic_json(tmp_path / "candidate.json", value)
        assert not e.replay(tmp_path / "candidate.json"), attack
    work = deepcopy(prepared)
    work["deadline_case_rows"] = e.read(work["historical_work_reference"])["deadline_case_rows"]
    work["deadline_case_rows"][-1]["clock_reads"] = 2
    atomic_json(tmp_path / "case-work.json", work)
    atomic_json(
        tmp_path / "case-candidate.json",
        dict(primitive_reference=e.ref(tmp_path / "case-work.json")),
    )
    assert not e.replay(tmp_path / "case-candidate.json")


def test_unresumed_dispatch(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8616: normal dispatch runs this controller exactly once."""
    from carnot.reporting import deadline_statement_validation_8616 as v

    work = deepcopy(prepared)
    work["raw"] = str(tmp_path / "work")
    calls = []
    monkeypatch.setattr(e, "prepare", lambda raw: work)
    monkeypatch.setattr(v, "execute", lambda w: calls.append(w["raw"]))
    assert e.run(tmp_path / (e.NAME + ".json"), "20261017") == 0
    assert calls == [work["raw"]]


def test_controller_plan_and_attestation(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8616: exercise plan accounting and each raw-shard rejection path."""
    import xml.etree.ElementTree as xml

    from coverage import CoverageData
    from coverage.parser import PythonParser

    from carnot.reporting import deadline_statement_validation_8616 as v

    work = deepcopy(prepared)
    work["raw"] = str(tmp_path)
    originals = work["original_nodes"]
    nodes = [
        *e.read(work["historical_work_reference"])["node_union"]["collected"],
        e.TESTS[0] + "::test_historical_mapping_and_rejections",
        e.TESTS[0] + "::test_actual_removed_read_mutant",
    ]

    def fake_child(name, argv, logs, **kwargs):
        logs.mkdir(exist_ok=True)
        stdout = logs / (name + ".stdout")
        stdout.write_text("\n".join(nodes) if name == "collection" else "")
        for arg in argv:
            if arg.startswith("--junitxml="):
                tree = xml.Element("testsuite")
                for node in argv[argv.index(arg) + 1 :]:
                    file, test = node.split("::", 1)
                    xml.SubElement(
                        tree, "testcase", classname=file[:-3].replace("/", "."), name=test
                    )
                xml.ElementTree(tree).write(arg.split("=", 1)[1])
        if name == "coverage":
            atomic_json(tmp_path / "coverage_full.json", dict(files=files))
        return dict(name=name, passed=True, stdout_path=str(stdout))

    files = {}
    data = CoverageData(basename=str(tmp_path / ".coverage.synthetic"))
    for path in [*e.OWNED, e.RUNNER]:
        parser = PythonParser(filename=str(e.ROOT / path))
        parser.parse_source()
        lines = sorted(parser.statements)
        files[path] = dict(
            executed_lines=lines, missing_lines=[], excluded_lines=[], summary=dict(missing_lines=0)
        )
        data.add_lines({str(e.ROOT / path): lines})
    data.write()
    for elapsed in (3599, 3600, 3601):
        row = e.read(prepared["historical_work_reference"])["deadline_case_rows"][(elapsed - 3599)]
        atomic_json(tmp_path / "deadline_cases" / f"case-{elapsed}" / "case.json", row)

    class OuterMeter:
        """Keep the real outer tracer active while testing the child plan in isolation."""

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def save(self):
            pass

    monkeypatch.setattr(v, "Coverage", OuterMeter)
    monkeypatch.setattr(v, "child", fake_child)
    monkeypatch.setattr(v, "clock", lambda: 0)
    v.execute(work)
    # Synthetic shards test the checker only; they never enter the study's child union.
    work["shards"] = [e.ref(tmp_path / ".coverage.synthetic")]
    v.attest(work)
    assert e.qualified(work)
    assert e.build(work, tmp_path / (e.NAME + ".json"))["learning_execution_ready_score"] == 1
    atomic_json(tmp_path / "work.json", work)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(primitive_reference=e.ref(tmp_path / "work.json")))
    with (
        patch.object(e.base, "replay", lambda path: True),
        patch.object(e.stream, "replay", lambda path: True),
    ):
        assert e.replay(candidate)
    for attack in (
        "union",
        "empty",
        "hash",
        "coverage",
        "mapping",
        "mutation",
        "failed",
        "junit",
        "exit",
        "case",
    ):
        changed = deepcopy(work)
        if attack == "union":
            changed["node_union"]["children"] = []
        elif attack == "empty":
            changed["shards"] = []
        elif attack == "hash":
            changed["shards"][0]["sha256"] = "wrong"
        elif attack == "coverage":
            bad = deepcopy(files)
            bad[e.RUNNER]["missing_lines"] = [92]
            atomic_json(tmp_path / "bad.json", dict(files=bad))
            changed["full_coverage_reference"] = e.ref(tmp_path / "bad.json")
        elif attack == "mapping":
            changed["current_mapping"] = {}
        elif attack == "mutation":
            changed["semantic_mutation_rows"] = []
        elif attack == "junit":
            changed["junit_references"][0]["sha256"] = "wrong"
        elif attack == "exit":
            changed["receipts"][0]["exit_code"] = 1
        elif attack == "case":
            changed["deadline_case_rows"][-1]["clock_reads"] = 2
        else:
            changed["node_union"]["outcomes"][0]["outcome"] = "failed"
        with pytest.raises(ValueError):
            v.attest(changed)
        assert not e.qualified(changed)
        atomic_json(tmp_path / "changed-work.json", changed)
        atomic_json(candidate, dict(primitive_reference=e.ref(tmp_path / "changed-work.json")))
        assert not e.replay(candidate)
    work["shards"][0]["path"] = str(tmp_path / "absent")
    assert not e.qualified(work)
    monkeypatch.setattr(v, "clock", lambda: work["started_monotonic_ns"] / 1e9 + 4000)
    with pytest.raises(TimeoutError):
        v.execute(work)


def test_contradicted_diagnosis_stops(prepared, monkeypatch):
    """REQ-REPORT-8616: changed raw operands stop qualification before expensive work."""
    original_ref, original_read = e.ref, e.read

    def changed_ref(path):
        value = original_ref(path)
        if path.name == "coverage_full.json":
            value["sha256"] = "wrong"
        return value

    monkeypatch.setattr(e, "ref", changed_ref)
    with pytest.raises(ValueError, match="historical_coverage"):
        e.diagnosis()
    monkeypatch.setattr(e, "ref", original_ref)

    def changed_read(reference):
        value = original_read(reference)
        if reference["path"].endswith("coverage_full.json"):
            value["files"][e.RUNNER]["missing_lines"] = [92]
        return value

    monkeypatch.setattr(e, "read", changed_read)
    with pytest.raises(ValueError, match="diagnosis_contradicted"):
        e.diagnosis()


def test_validation_reserve_keeps_measurement_boundary():
    """REQ-VERIFY-8616: finished measurements leave900 seconds for real validation."""
    from carnot.reporting import deadline_statement_validation_8616 as v

    assert v.child_deadline(0, 3599.0) == 600
    assert v.child_deadline(0, 4400.0) == 100
    assert v.child_deadline(0, 4500.0) == 1
    assert e.ref(e.ROOT / e.RUNNER)["sha256"] == e.PIN_SOURCE
