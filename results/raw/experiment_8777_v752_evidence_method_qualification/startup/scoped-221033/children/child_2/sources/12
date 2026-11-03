"""REQ-REPORT-8768 / REQ-VERIFY-8768: independent constructed truth controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting import evidence_set_methods_8768 as m
from carnot.reporting import evidence_set_corpus_8768 as corpus
from carnot.reporting import evidence_set_analysis_8768 as analysis
from carnot.reporting import evidence_set_runner_8768 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def frozen():
    """Keep labels separate even in fixtures so the verifier boundary is exercised."""
    return corpus.freeze()


def test_open_world_join_and_conflict():
    """REQ-VERIFY-8768: missing premises differ from explicit negative literals."""
    p = dict(
        facts=[["a", "r", "b", True], ["b", "s", "c", True]],
        query=[["a", "r", "?x"], ["?x", "s", "c"]],
    )
    assert m.decide(p)["decision"] == "support"
    assert m.decide(p)["required_joins"] == ["?x"]
    missing = dict(p, facts=p["facts"][:1])
    assert m.decide(missing)["decision"] == "insufficient"
    assert m.decide(missing)["missing_premises"] == [1]
    disconnected = dict(p, facts=[p["facts"][0], ["d", "s", "c", True]])
    assert m.decide(disconnected)["missing_join"]
    assert m.compare("local_record", disconnected)[0] == "support"
    assert m.compare("joint_symbolic", disconnected)[0] == "insufficient"
    negative = dict(facts=[["a", "r", "b", False]], query=[["a", "r", "b"]])
    assert m.decide(negative)["decision"] == "refute"
    conflict = dict(negative, facts=negative["facts"] + [["a", "r", "b", True]])
    assert m.decide(conflict)["decision"] == "insufficient"
    assert m.decide(conflict)["conflicts"]
    assert corpus.enumerate_truth(conflict)["decision"] == "insufficient"
    assert corpus.enumerate_truth(missing)["satisfiable_outcomes"] == [False, True]
    with pytest.raises(ValueError, match="arm"):
        m.compare("forged", p)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(truth="support"),
        lambda p: p.update(query=[]),
        lambda p: p.update(facts=[["a", "r", "b", 1]]),
        lambda p: p.update(query=[["a", "?r", "b"]]),
        lambda p: p.update(facts=[["?x", "r", "b", True]]),
        lambda p: p.update(query=[["", "r", "b"]]),
    ],
)
def test_parser_rejects_leakage_and_malformed_records(change):
    """SCENARIO-VERIFY-8768-TERMINAL: truth and malformed fields never become features."""
    p = dict(facts=[["a", "r", "b", True]], query=[["a", "r", "b"]])
    change(p)
    with pytest.raises(ValueError):
        m.parse(p)


def test_json_rejection_and_roundtrip():
    """SCENARIO-VERIFY-8768-TERMINAL: duplicate keys and nonfinite JSON reject."""
    p = dict(facts=[], query=[["a", "r", "b"]])
    assert m.decode(json.dumps(p)) == p
    for text in ['{"facts":[],"facts":[],"query":[]}', '{"facts":NaN}', "[]", "{"]:
        with pytest.raises(ValueError):
            m.decode(text)
    with pytest.raises(ValueError, match="input_lists"):
        m.parse(dict(facts={}, query=[["a", "r", "b"]]))
    with pytest.raises(ValueError, match="insufficient_distinct_graphs"):
        corpus.graph_roster(1, 1)


def test_no_oracle_access(monkeypatch):
    """REQ-VERIFY-8768: replacing the label enumerator with a trap cannot affect decisions."""

    def forbidden(*args, **kwargs):
        raise AssertionError("oracle accessed by verifier")

    inputs, _ = corpus.construct()
    monkeypatch.setattr(corpus, "enumerate_truth", forbidden)
    decisions = analysis.measure(inputs[:12])
    assert len(decisions) == 36
    assert analysis.audit(inputs[:12], decisions)


def test_corpus_and_independent_labels(frozen):
    """SCENARIO-REPORT-8768-CORPUS: all 2160 units have independent three-way labels."""
    inputs, labels, sources = frozen
    assert len(inputs) == len(labels) == len(sources) == 2160
    assert sum(p["split"] == "evaluation" for p in inputs) == 1440
    assert len({p["world_id"] for p in inputs if p["split"] == "development"}) == 60
    assert len({p["world_id"] for p in inputs if p["split"] == "evaluation"}) == 120
    assert corpus.disjoint(inputs)
    originals = [
        p for p in inputs if p["condition"] == "complete_support" and p["view"] == "original"
    ]
    assert len({p["structure"] for p in originals}) == 180
    by_id = {r["unit_id"]: r for r in labels}
    expected = dict(
        complete_support="support",
        explicit_refutation="refute",
        missing_required_hop="insufficient",
        disconnected_join="insufficient",
        irrelevant_distractor="support",
        explicit_conflict="insufficient",
    )
    for p, source in zip(inputs, sources, strict=True):
        label = by_id[p["unit_id"]]
        assert label["decision"] == expected[p["condition"]]
        assert m.decide(p["input"])["decision"] == label["decision"]
        assert all(source["document"][a:b] == line for a, b, line in source["spans"])
    forged = deepcopy(inputs)
    forged[-1]["template"] = forged[0]["template"]
    assert not corpus.disjoint(forged)


def test_measurement_statistics_and_negative_controls(tmp_path, frozen):
    """SCENARIO-REPORT-8768-STATISTICS: pair worlds and retain actual invocation costs."""
    inputs, labels, _ = frozen
    decisions = analysis.measure(inputs)
    rows = analysis.score(inputs, decisions, labels)
    assert len(rows) == 6480 and sum(r["split"] == "evaluation" for r in rows) == 4320
    assert all(1 <= r["evaluator_invocations"] <= 5 for r in rows)
    assert analysis.audit(inputs, decisions)
    summary = analysis.summarize(rows)
    assert summary["independent_count"] == 120
    assert summary["statistical_protocol"]["draws"] == 10000
    assert len(summary["paired_intervals"]) == 2
    assert summary["truth_controls_passed"]
    assert summary["correct_control_invariance"]
    assert summary["scientific_promotion"] == all(
        r["lower_simultaneous_97_5"] > 0 for r in summary["paired_intervals"]
    )
    optimal = deepcopy(rows)
    for row in optimal:
        row.update(decision=row["expected"], raw_numerator=0, absolute_metric=1)
    null = analysis.summarize(optimal)
    assert not null["scientific_promotion"]
    assert all(r["lower_simultaneous_97_5"] == 0 for r in null["paired_intervals"])
    assert all(r["failures"] > 0 for r in summary["negative_control_rows"])
    bad = deepcopy(decisions)
    bad[0]["decision"] = "refute"
    assert not analysis.audit(inputs, bad)
    bad = deepcopy(decisions)
    bad[0]["wall_time_s"] = -1
    assert not analysis.audit(inputs, bad)
    assert not analysis.audit(inputs, decisions[:-1])
    assert analysis.protocol() == m.method()["statistical_protocol"]


def test_private_cli_and_rehashed_replay(tmp_path):
    """SCENARIO-VERIFY-8768-TERMINAL: real CLI children and independent seals reject attacks."""
    output = tmp_path / (m.NAME + ".json")
    argv = [sys.executable, str(m.ROOT / m.CLI)]
    if os.environ.get("COVERAGE_RCFILE"):
        argv = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + os.environ["COVERAGE_RCFILE"],
            str(m.ROOT / m.CLI),
        ]
    logs = tmp_path / "logs"
    assert child(
        "private", [*argv, "--private-control", "--output", str(output)], logs, deadline=180
    )["passed"]
    assert runner.replay(output)
    assert child("replay", [*argv, "--cold-replay", str(output)], logs, deadline=180)["passed"]
    saved = output.read_bytes()
    value = json.loads(saved)
    assert value["completed_count"] == 6480
    value["rows"][0]["decision"] = "refute"
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not runner.replay(output)
    assert child("tamper", [*argv, "--cold-replay", str(output)], logs, expected=1, deadline=180)[
        "passed"
    ]
    output.write_bytes(saved)
    primitive = Path(value["primitive_reference"]["path"])
    original = primitive.read_bytes()
    work = json.loads(original)
    work["method_reference"]["sha256"] = "sha256:" + "0" * 64
    atomic_json(primitive, work)
    assert not runner.replay(output)
    primitive.write_bytes(original)
    assert not runner.replay(tmp_path / "absent")
    assert child("date", [*argv, "--date", "20260101"], logs, expected=2, deadline=60)["passed"]
    assert child("unsafe", [*argv, "--private-control"], logs, expected=2, deadline=60)["passed"]


def test_production_dispatch_and_validation_gates(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8768-TERMINAL: real dispatcher and failed owned checks preserve gates."""
    from carnot.reporting import evidence_set_report_8768 as report
    from carnot.reporting.v746_retained_evidence import reference

    def checked(raw, tracer):
        tracer.stop()
        tracer.save()
        return dict(
            receipts=[dict(name="private dispatcher fixture", passed=True)],
            union=dict(passed=True),
            coverage=dict(complete=True),
            raw_references=[],
        )

    monkeypatch.setattr(runner, "validate", checked)
    output = tmp_path / (m.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["evidence_set_methods_ready_score"] == 1
    assert value["verdict_class"] in {"null", "circular_positive"}
    assert runner.replay(output)
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    validation = json.loads(Path(value["validation_reference"]["path"]).read_bytes())
    rejected = deepcopy(validation)
    rejected["receipts"][0]["passed"] = False
    report_value = report.build(work, rejected, output)
    assert report_value["verdict_class"] == "disqualified"
    assert not report_value["evidence_set_methods_ready_score"]
    blocked = deepcopy(work)
    blocked["preconditions_checked"]["passed"] = False
    assert report.build(blocked, validation, output)["verdict_class"] == "blocked"
    seal = Path(work["raw"]) / "independent_seal.json"
    source = tmp_path / "private-source"
    source.write_text("original")
    work["sources"].append(reference(source))
    source.write_text("changed")
    primitive = Path(value["primitive_reference"]["path"])
    atomic_json(primitive, work)
    value["primitive_reference"] = reference(primitive)
    atomic_json(seal, {k: value[k] for k in ["primitive_reference", "validation_reference"]})
    atomic_json(output, value)
    assert not runner.replay(output)
    work["sources"].pop()
    original_seal = seal.read_bytes()
    atomic_json(seal, {})
    assert not runner.replay(output)
    seal.write_bytes(original_seal)
    oracle = Path(work["oracle_reference"]["path"])
    truth = json.loads(oracle.read_bytes())
    truth["rows"][0]["decision"] = "refute"
    atomic_json(oracle, truth)
    work["oracle_reference"] = reference(oracle)
    primitive = Path(value["primitive_reference"]["path"])
    atomic_json(primitive, work)
    value["primitive_reference"] = reference(primitive)
    atomic_json(seal, {k: value[k] for k in ["primitive_reference", "validation_reference"]})
    atomic_json(output, value)
    assert not runner.replay(output)


def test_worker_and_reader_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8768-TERMINAL: actual worker parse and owned process failures reject."""
    inputs = tmp_path / "inputs.json"
    atomic_json(inputs, dict(rows=[]))
    with pytest.raises(SystemExit) as error:
        runner.main(["--decide-input", str(inputs)])
    assert error.value.code == 2
    inputs.write_text('{"rows":[],"rows":[]}')
    with pytest.raises(ValueError, match="duplicate_key"):
        runner.main(
            ["--decide-input", str(inputs), "--decisions", str(tmp_path / "decisions.json")]
        )
    original_child = runner.child

    def fail_after_actual_child(name, argv, logs, **kwargs):
        receipt = original_child(name, argv, logs, **kwargs)
        if name == "development_input_only":
            receipt["passed"] = False
        return receipt

    monkeypatch.setattr(runner, "child", fail_after_actual_child)
    output = tmp_path / "failed" / (m.NAME + ".json")
    with pytest.raises(ValueError, match="input_only_child_rejected"):
        runner.main(["--private-control", "--output", str(output)])
    monkeypatch.setattr(runner, "child", original_child)
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **kw: dict(passed=False))
    output = tmp_path / "reader" / (m.NAME + ".json")
    with pytest.raises(ValueError, match="reader_identity"):
        runner.main(["--private-control", "--output", str(output)])


def test_strict_validation_configuration(tmp_path, monkeypatch):
    """REQ-VERIFY-8768: CLI coverage and strict type scope must survive child dispatch."""
    from carnot.reporting.v746_retained_evidence import reference

    captured = []
    monkeypatch.setattr(
        runner, "child", lambda name, argv, logs, **kw: captured.append(argv) or dict(passed=True)
    )
    runner.validation_child("types", ["mypy", "legacy"], tmp_path)
    assert "--strict" in captured[-1] and "--config-file=/dev/null" in captured[-1]
    runner.validation_child("child_0", [sys.executable, "--rcfile=/tmp/private.ini"], tmp_path)
    assert captured[-1][:2] == ["env", "COVERAGE_RCFILE=/tmp/private.ini"]
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\nparallel=true\n[report]\nexclude_lines=\n")
    monkeypatch.setattr(
        runner, "collect", lambda *a: ([dict(config=reference(config))], dict(passed=True))
    )
    plan, receipt = runner.validation_collect(tmp_path, tmp_path, m.TESTS, m.TESTS, runner.child)
    assert receipt["passed"] and plan[0]["config"] == reference(config)
    assert all(str(m.ROOT / name) in config.read_text() for name in m.OWNED)
    monkeypatch.setattr(
        runner.validation_runner, "validate", lambda raw, tracer, suites: dict(passed=True)
    )
    assert runner.validate(tmp_path, None)["passed"]
