"""REQ-REPORT-8779 / REQ-VERIFY-8779: private truth, costs and terminal custody."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import evidence_set_evaluation_8779 as e
from carnot.reporting import evidence_evaluation_report_8779 as report
from carnot.reporting import evidence_evaluation_runner_8779 as runner
from carnot.reporting import evidence_qualification_8777 as q
from carnot.reporting import evidence_set_corpus_8768 as corpus
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    """Private constructed operands keep every test away from production result files."""
    root = tmp_path_factory.mktemp("upstream8779")
    old, _ = corpus.construct()
    inputs = [
        dict(p, split="replication", input=q.projection(p["input"]))
        for p in old
        if p["split"] == "evaluation"
    ]
    inputs += [
        dict(p, input=q.projection(p["input"]))
        for p in q.new_units("evaluation", corpus.graph_roster(6, 120))
    ]
    labels = [
        dict(unit_id=p["unit_id"], decision=corpus.enumerate_truth(p["input"])["decision"])
        for p in inputs
    ]
    for name, value in [
        ("corpus", dict(rows=inputs)),
        ("oracle", dict(rows=labels)),
        (
            "method",
            dict(
                original_method=original.method(),
                maximum_evaluator_invocations=5,
                local_over_budget="first five premise checks; any refutation refutes; otherwise insufficient",
                MODEL_SPECS=[],
            ),
        ),
    ]:
        atomic_json(root / (name + ".json"), value)
    atomic_json(
        root / "work.json",
        dict(
            corpus_reference=reference(root / "corpus.json"),
            oracle_reference=reference(root / "oracle.json"),
        ),
    )
    value = dict(
        experiment_id=8777,
        task_id=q.TASK,
        evidence_methods_ready_score=1,
        verdict_class="circular_positive",
        required_checks_passed=True,
        flagged_adversarial=False,
        primitive_reference=reference(root / "work.json"),
        frozen_method_reference=reference(root / "method.json"),
        code_config_hashes=[reference(e.ROOT / p) for p in original.OWNED + q.OWNED],
    )
    atomic_json(root / "upstream.json", value)
    return root / "upstream.json"


def test_gate_diagnostics_and_authentication(tmp_path, upstream):
    """SCENARIO-VERIFY-8779-CUSTODY: missing, zero and rejected are distinct operands."""
    assert e.freeze(upstream)["passed"]
    missing = e.freeze(tmp_path / "missing")
    assert missing["diagnostics"][0]["check"] == "missing_file"
    original_value = json.loads(upstream.read_bytes())
    for field, value, check in [
        ("evidence_methods_ready_score", None, "missing_field"),
        ("evidence_methods_ready_score", 0, "observed_zero"),
        ("verdict_class", "disqualified", "rejected_producer"),
    ]:
        altered = deepcopy(original_value)
        if value is None:
            altered.pop(field)
        else:
            altered[field] = value
        target = tmp_path / "producer.json"
        atomic_json(target, altered)
        result = e.freeze(target)
        assert not result["passed"] and result["diagnostics"][0]["check"] == check
        assert {
            "upstream",
            "path",
            "sha256",
            "field",
            "operator",
            "expected",
            "observed",
        } <= result["diagnostics"][0].keys()
    altered = deepcopy(original_value)
    altered["frozen_method_reference"]["sha256"] = "changed"
    atomic_json(target, altered)
    assert not e.freeze(target)["passed"]


@pytest.fixture(scope="module")
def measured(upstream):
    """SCENARIO-REPORT-8779-ROWS: seal decisions before opening independent labels."""
    frozen = e.freeze(upstream)
    decisions = e.measure([p["input"] for p in frozen["inputs"]])
    return frozen, decisions, e.score(frozen["inputs"], frozen["labels"], decisions)


def test_complete_rows_and_separate_strata(measured, monkeypatch):
    """REQ-REPORT-8779: identical inputs, bounded calls and independent error denominators."""
    frozen, decisions, rows = measured
    assert len(rows) == 8640 and e.audit(frozen["inputs"], decisions)
    assert all(
        1 <= r["evaluator_invocations"] <= 5 and r["cpu_time_s"] >= 0 and r["input_size_bytes"] > 0
        for r in rows
    )
    assert all(
        {
            "false_acceptance",
            "valid_alternative_rejection",
            "missing_hop_error",
            "conflict_error",
            "surface",
        }
        <= r.keys()
        for r in rows
    )
    summary = e.summarize(rows)
    assert len(summary["paired_world_rows"]) == 240
    assert summary["exposed_replication_metrics"]["independent_count"] == 120
    assert summary["new_family_metrics"]["independent_count"] == 120
    assert len(summary["paired_intervals"]) == 4
    assert all(r["draws"] == 10000 for r in summary["paired_intervals"])
    perfect = deepcopy(rows)
    for row in perfect:
        row.update(decision=row["expected"], raw_numerator=0, absolute_metric=1)
    assert not e.summarize(perfect)["scientific_promotion"]
    assert not e.audit(frozen["inputs"], decisions[:-1])
    forged = deepcopy(decisions)
    forged[0]["cpu_time_s"] = -1
    assert not e.audit(frozen["inputs"], forged)
    monkeypatch.setattr(corpus, "enumerate_truth", lambda *a: pytest.fail("oracle in arm"))
    assert len(e.measure([frozen["inputs"][0]["input"]])) == 3
    assert e.summarize([])["paired_intervals"] == []


def invoke(tmp_path, *args, expected=0):
    """Real bounded children collect executable CLI coverage and keep logs private."""
    return child(
        "cli_" + str(len(list(tmp_path.glob("*.receipt.json")))),
        [sys.executable, e.CLI, *map(str, args)],
        tmp_path,
        deadline=180,
        expected=expected,
    )


def test_private_cli_and_cold_tampering(tmp_path, upstream):
    """SCENARIO-VERIFY-8779-CUSTODY: real failure publishes and fresh replay rejects tampering."""
    output = tmp_path / (e.NAME + ".json")
    assert invoke(tmp_path, "--upstream", upstream, "--private-control", "--output", output)[
        "passed"
    ]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["evidence_rows_ready_score"] == 0
    assert value["completed_count"] == 8640 and report.replay(output)
    assert invoke(tmp_path, "--cold-replay", output)["passed"]
    saved = output.read_bytes()
    value["rows"][0]["decision"] = "forged"
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not report.replay(output)
    assert invoke(tmp_path, "--cold-replay", output, expected=1)["passed"]
    output.write_bytes(saved)
    work = Path(value["primitive_reference"]["path"])
    old = work.read_bytes()
    atomic_json(work, {})
    assert not report.replay(output)
    work.write_bytes(old)
    raw = work.parent
    seal = raw / "independent_seal.json"
    validation_path = Path(value["validation_reference"]["path"])
    decision_path = raw / "decisions.json"
    originals = {p: p.read_bytes() for p in [output, work, seal, validation_path, decision_path]}
    for mode in ["seal", "validation", "upstream", "source", "decisions"]:
        candidate = json.loads(saved)
        primitive = json.loads(old)
        if mode == "seal":
            atomic_json(seal, {})
        elif mode == "validation":
            atomic_json(validation_path, dict(union={}, coverage=dict(complete=True)))
            candidate["validation_reference"] = reference(validation_path)
        elif mode == "upstream":
            primitive["upstream_path"] = str(tmp_path / "absent")
        elif mode == "source":
            primitive["sources"][0]["sha256"] = "forged"
        else:
            decision_data = json.loads(originals[decision_path])
            decision_data["rows"][0]["cpu_time_s"] = -1
            atomic_json(decision_path, decision_data)
            primitive["decisions_reference"] = reference(decision_path)
            primitive["shards"] = [
                reference(decision_path) if r["path"] == str(decision_path) else r
                for r in primitive["shards"]
            ]
        if mode != "seal":
            atomic_json(work, primitive)
            candidate["primitive_reference"] = reference(work)
            atomic_json(output, candidate)
            atomic_json(
                seal,
                {
                    k: candidate[k]
                    for k in [
                        "primitive_reference",
                        "validation_reference",
                        "frozen_input_reference",
                    ]
                },
            )
        assert not report.replay(output)
        for path, content in originals.items():
            path.write_bytes(content)
    assert not report.replay(tmp_path / "absent")
    assert invoke(tmp_path, "--private-control", "--output", e.ROOT / "unsafe.json", expected=2)[
        "passed"
    ]
    assert invoke(tmp_path, "--date", "wrong", expected=2)["passed"]
    assert invoke(tmp_path, "--decide-input", upstream, expected=2)["passed"]


@pytest.mark.parametrize("mode", ["method", "source", "budget", "count", "overlap", "truth"])
def test_frozen_operand_mutations(tmp_path, upstream, mode):
    """REQ-VERIFY-8779: authenticated bytes alone cannot qualify malformed scientific operands."""
    producer = json.loads(upstream.read_bytes())
    work = json.loads(Path(producer["primitive_reference"]["path"]).read_bytes())
    inputs = json.loads(Path(work["corpus_reference"]["path"]).read_bytes())
    labels = json.loads(Path(work["oracle_reference"]["path"]).read_bytes())
    method = json.loads(Path(producer["frozen_method_reference"]["path"]).read_bytes())
    if mode == "method":
        method["maximum_evaluator_invocations"] = 6
    elif mode == "source":
        producer["code_config_hashes"][0]["sha256"] = "forged"
    elif mode == "budget":
        next(r for r in producer["code_config_hashes"] if r["path"] == str(e.ROOT / q.OWNED[0]))[
            "sha256"
        ] = "forged"
    elif mode == "count":
        inputs["rows"].pop()
    elif mode == "overlap":
        for row in inputs["rows"]:
            row["structure"] = "same"
    else:
        labels["rows"][0]["decision"] = "refute"
    for name, value in [("corpus", inputs), ("oracle", labels), ("method", method)]:
        atomic_json(tmp_path / (name + ".json"), value)
    work.update(
        corpus_reference=reference(tmp_path / "corpus.json"),
        oracle_reference=reference(tmp_path / "oracle.json"),
    )
    atomic_json(tmp_path / "work.json", work)
    producer.update(
        primitive_reference=reference(tmp_path / "work.json"),
        frozen_method_reference=reference(tmp_path / "method.json"),
    )
    atomic_json(tmp_path / "producer.json", producer)
    assert not e.freeze(tmp_path / "producer.json")["passed"]


def test_real_validation_collection(tmp_path, monkeypatch):
    """REQ-VERIFY-8779: exercise real collection, normal exits, JUnit and SQLite retention."""
    monkeypatch.setattr(e, "TESTS", [original.CONSUMERS[0]])
    monkeypatch.setattr(original, "CONSUMERS", [original.CONSUMERS[1]])
    raw, private = tmp_path / "raw", tmp_path / "private"
    raw.mkdir()
    private.mkdir()
    validation = runner.validate(raw, private)
    assert validation["union"]["passed"]
    assert validation["original_node_union"]["passed"]
    assert validation["coverage"]["combined_reference"]
    assert all(r["normal_exit"] for r in validation["receipts"])


def test_normal_entry_and_blocked_publication(tmp_path, upstream, monkeypatch):
    """SCENARIO-VERIFY-8779-CUSTODY: normal entry uses scoped validation; absent upstream blocks."""
    monkeypatch.setattr(
        runner, "validate", lambda raw, private: runner.previous.private_validation(raw)
    )
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--output", str(output), "--upstream", str(upstream)]) == 0
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    validation = json.loads(Path(value["validation_reference"]["path"]).read_bytes())
    qualified = deepcopy(validation)
    qualified.update(
        receipts=[dict(passed=True)], union=dict(passed=True), coverage=dict(complete=True)
    )
    assert report.build(work, qualified, output)["evidence_rows_ready_score"] == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert runner.main(["--output", str(blocked), "--upstream", str(tmp_path / "absent")]) == 0
    blocked_value = json.loads(blocked.read_bytes())
    assert blocked_value["verdict_class"] == "blocked" and blocked_value["independent_count"] == 0
    monkeypatch.setattr(
        runner.yaml, "safe_load", lambda _: dict(retired=[dict(experiment_id=8779)])
    )
    assert not runner.preconditions(tmp_path / "pre")["passed"]
