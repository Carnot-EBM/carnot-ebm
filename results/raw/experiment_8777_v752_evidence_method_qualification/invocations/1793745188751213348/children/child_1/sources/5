"""REQ-REPORT-8777 / REQ-VERIFY-8777: qualify original truth and real CLI failure."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting import evidence_qualification_8777 as q
from carnot.reporting import evidence_archive_8777 as archive
from carnot.reporting import evidence_qualification_report_8777 as report
from carnot.reporting import evidence_qualification_runner_8777 as runner
from carnot.reporting import evidence_set_methods_8768 as original
from carnot.reporting import evidence_set_corpus_8768 as corpus
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference


@pytest.fixture(scope="module")
def frozen():
    """Keep expensive graph construction shared without sharing mutable test operands."""
    return q.freeze()


def test_frozen_worlds_and_truth(frozen):
    """SCENARIO-REPORT-8777-TRUTH: canonical worlds and Boolean labels stay independent."""
    inputs, labels, manifest = frozen
    assert len(inputs) == len(labels) == 5184
    assert manifest["world_counts"] == dict(
        development=60, replication=120, evaluation=120, canary=12, retention=120
    )
    assert manifest["disjoint"] and len(manifest["canonical_worlds"]) == 432
    assert len({r["structure"] for r in manifest["canonical_worlds"]}) == 432
    expected = dict(
        zip(
            corpus.CONDITIONS,
            ["support", "refute", "insufficient", "insufficient", "support", "insufficient"],
            strict=True,
        )
    )
    for unit, label in zip(inputs, labels, strict=True):
        assert label["unit_id"] == unit["unit_id"]
        assert label["decision"] == expected[unit["condition"]]
        assert corpus.enumerate_truth(unit["input"]) == label["oracle"]
        assert original.decide(unit["input"])["decision"] == label["decision"]
    changed = deepcopy(manifest["canonical_worlds"])
    changed[-1]["structure"] = changed[0]["structure"]
    assert not q.disjoint(changed)


def test_five_call_budget_and_no_truth_features(monkeypatch, frozen):
    """REQ-REPORT-8777: actual costs respect equal ceilings and never consult oracle labels."""
    inputs = [
        next(p for p in frozen[0] if p["split"] == s)
        for s in ["development", "evaluation", "retention"]
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("oracle read by arm")

    monkeypatch.setattr(corpus, "enumerate_truth", forbidden)
    rows = q.measure([p["input"] for p in inputs])
    assert len(rows) == 9 and q.audit(inputs, rows)
    assert all(1 <= r["evaluator_invocations"] <= 5 and r["wall_time_s"] >= 0 for r in rows)
    assert [r["decision"] for r in rows[3:6]] == ["insufficient", "support", "support"]
    assert all(
        "v751" not in json.dumps(p["input"]) and "v752" not in json.dumps(p["input"])
        for p in inputs
    )
    rows[0]["decision"] = "forged"
    assert not q.audit(inputs, rows)
    assert not q.audit(inputs, rows[:-1])
    with pytest.raises(ValueError):
        q.measure([dict(inputs[0]["input"], condition="complete_support")])


def test_negative_controls_and_semantic_null(frozen):
    """SCENARIO-REPORT-8777-TRUTH: a complete semantic null still qualifies execution."""
    inputs, labels, _ = frozen
    decisions = q.measure([p["input"] for p in inputs])
    rows = q.score(inputs, labels, decisions)
    summary = q.summarize(rows)
    assert summary["truth_controls_passed"]
    assert all(r["failures"] > 0 for r in summary["negative_control_rows"])
    assert summary["independent_count"] == 120
    assert summary["correct_control_invariance"]
    assert all(p["lower_simultaneous_97_5"] > 0 for p in summary["paired_intervals"])
    optimal = deepcopy(rows)
    for row in optimal:
        row.update(decision=row["expected"], raw_numerator=0, absolute_metric=1)
    assert not q.summarize(optimal)["scientific_promotion"]


def cli(tmp_path, *args, expected=0):
    """Use the real entrypoint so subprocess statements contribute genuine coverage."""
    argv = [sys.executable, q.CLI, *map(str, args)]
    return child(
        "cli_" + str(len(list(tmp_path.glob("*.receipt.json")))),
        argv,
        tmp_path,
        expected=expected,
        deadline=300,
    )


def test_private_cli_and_rehashed_mutations(tmp_path):
    """SCENARIO-VERIFY-8777-CUSTODY: exact private publication and fresh replay resist tampering."""
    output = tmp_path / (q.NAME + ".json")
    assert cli(tmp_path, "--private-control", "--output", output)["passed"]
    assert report.replay(output)
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["evidence_methods_ready_score"] == 0
    assert value["completed_count"] == value["intended_count"] == 15552
    assert cli(tmp_path, "--cold-replay", output)["passed"]
    saved = output.read_bytes()
    value["rows"][0]["decision"] = "refute"
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not report.replay(output)
    assert cli(tmp_path, "--cold-replay", output, expected=1)["passed"]
    output.write_bytes(saved)
    work_path = Path(value["primitive_reference"]["path"])
    work_bytes = work_path.read_bytes()
    work = json.loads(work_bytes)
    work["duration_s"] = -1
    atomic_json(work_path, work)
    assert not report.replay(output)
    work_path.write_bytes(work_bytes)
    assert not report.replay(tmp_path / "absent")
    raw = work_path.parent
    candidates = [
        output,
        work_path,
        raw / "frozen_inputs.json",
        raw / "independent_seal.json",
        raw / "oracle.json",
        raw / "decisions.json",
        raw / "arm_inputs.json",
        raw / "validation.json",
        raw / "v751_authority/historical_cli.py",
    ]
    originals = {p: p.read_bytes() for p in candidates}
    for mode in [
        "seal",
        "frozen",
        "source",
        "archive",
        "validation",
        "oracle_reference",
        "decisions_reference",
    ]:
        value = json.loads(saved)
        work = json.loads(work_bytes)
        frozen_work = json.loads(originals[raw / "frozen_inputs.json"])
        if mode == "seal":
            atomic_json(raw / "independent_seal.json", {})
        elif mode == "source":
            (raw / "arm_inputs.json").write_text("changed sealed source")
        elif mode == "archive":
            (raw / "v751_authority/historical_cli.py").write_text("changed private archive")
        elif mode == "validation":
            invalid = json.loads(originals[raw / "validation.json"])
            invalid["coverage"]["complete"] = True
            atomic_json(raw / "validation.json", invalid)
            value["validation_reference"] = reference(raw / "validation.json")
            atomic_json(output, value)
            atomic_json(
                raw / "independent_seal.json",
                {
                    k: value[k]
                    for k in [
                        "primitive_reference",
                        "validation_reference",
                        "frozen_input_reference",
                    ]
                },
            )
        else:
            if mode == "frozen":
                work["run_date"] = "forged"
            else:
                ref = work[mode]
                operand = Path(ref["path"])
                data = json.loads(operand.read_bytes())
                data["rows"][0]["decision"] = "refute"
                atomic_json(operand, data)
                work[mode] = reference(operand)
                work["shards"] = [work[mode] if r == ref else r for r in work["shards"]]
                if mode in frozen_work:
                    frozen_work[mode] = work[mode]
                    atomic_json(raw / "frozen_inputs.json", frozen_work)
                    work["frozen_input_reference"] = reference(raw / "frozen_inputs.json")
                    value["frozen_input_reference"] = work["frozen_input_reference"]
            atomic_json(work_path, work)
            value["primitive_reference"] = reference(work_path)
            atomic_json(output, value)
            atomic_json(
                raw / "independent_seal.json",
                {
                    k: value[k]
                    for k in [
                        "primitive_reference",
                        "validation_reference",
                        "frozen_input_reference",
                    ]
                },
            )
        assert not report.replay(output), mode
        for path, data in originals.items():
            path.write_bytes(data)
    assert cli(tmp_path, "--date", "20260101", expected=2)["passed"]
    assert cli(tmp_path, "--private-control", expected=2)["passed"]


def test_worker_and_precondition_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8777: missing outputs, changed inputs and external absence cannot promote."""
    source = tmp_path / "input.json"
    atomic_json(source, dict(rows=[dict(facts=[], query=[["a", "r", "b"]])]))
    with pytest.raises(SystemExit):
        runner.main(["--decide-input", str(source)])
    target = tmp_path / "decisions.json"
    assert runner.main(["--decide-input", str(source), "--decisions", str(target)]) == 0
    assert json.loads(target.read_bytes())["oracle_file_reads"] == 0
    source.write_text('{"rows":[],"rows":[]}')
    with pytest.raises(ValueError, match="duplicate_key"):
        runner.main(["--decide-input", str(source), "--decisions", str(target)])
    monkeypatch.setattr(runner, "available", lambda: False)
    assert not runner.preconditions(tmp_path)["passed"]
    monkeypatch.undo()
    actual_child = runner.child

    def rejected(name, argv, logs, **kwargs):
        receipt = actual_child(name, argv, logs, **kwargs)
        if name == "input_only_decisions":
            receipt["passed"] = False
        return receipt

    monkeypatch.setattr(runner, "child", rejected)
    with pytest.raises(ValueError, match="input_only_child_rejected"):
        runner.main(
            ["--private-control", "--output", str(tmp_path / "rejected" / (q.NAME + ".json"))]
        )


def test_original_archive_and_real_failed_cli(tmp_path):
    """REQ-VERIFY-8777: original line345 executes a real failed validation and publishes zero."""
    root = archive.freeze_archive(tmp_path / "archive")
    for name in original.OWNED + original.TESTS:
        assert (root / name).read_bytes() == (q.ROOT / name).read_bytes()
    assert (root / "openspec/change-proposals/research-roadmap-vNEXT.md").read_bytes() == (
        q.ROOT / archive.V751
    ).read_bytes()
    primary_before = reference(q.ROOT / archive.PRIMARY)
    assert child(
        "unsafe_original_output",
        [
            sys.executable,
            str(root / "historical_cli.py"),
            "--private-control",
            "--output",
            str(q.ROOT / archive.PRIMARY),
        ],
        tmp_path / "guard_logs",
        expected=2,
        deadline=60,
    )["passed"]
    assert reference(q.ROOT / archive.PRIMARY) == primary_before
    receipt = archive.failure_control(root, tmp_path / "failure")
    assert receipt["passed"] and receipt["line345_executed"]
    assert receipt["published_verdict"] == "disqualified" and receipt["readiness"] == 0
    from carnot.reporting.collection_contract import collect

    raw, scratch = tmp_path / "checks", tmp_path / "scratch"
    raw.mkdir()
    scratch.mkdir()
    suite = os.path.relpath(root / "original_tests" / Path(original.TESTS[0]).name, q.ROOT)
    plan, collected = collect(scratch, raw, [suite], [suite], child)
    # One real historical assertion is enough to exercise validation custody here;
    # the complete original inventory remains mandatory in the production run.
    plan[0]["nodes"] = plan[0]["nodes"][:1]
    argv = plan[0]["argv"]
    start = argv.index("-q") + 1
    end = next(i for i, arg in enumerate(argv) if arg.startswith("--basetemp="))
    argv[start:end] = plan[0]["nodes"]
    plan[0]["role"] = "original"
    for spec in plan:
        config = Path(spec["config"]["path"])
        config.write_text(
            config.read_text().replace(
                "[run]\n",
                "[run]\ninclude=\n"
                + "\n".join("    " + str(q.ROOT / p) for p in q.OWNED + archive.ORIGINAL_OWNED)
                + "\n",
            )
        )
        spec["config"] = reference(config)
    atomic_json(raw / "frozen_test_plan.json", dict(plan=plan, collection_receipt=collected))
    atomic_json(raw / "failure/control.json", receipt)
    validation = runner.validate(raw, None, root)
    assert validation["union"]["passed"] and validation["original_node_union"]["passed"]
    assert all(r["passed"] for r in validation["receipts"])
    assert not validation["coverage"]["complete"]
    assert report.audit_validation(validation)
    changed = deepcopy(validation)
    changed["coverage"]["files"][0]["missing"].append(999999)
    assert not report.audit_validation(changed)
    changed = deepcopy(validation)
    changed["union"]["node_ids"] = []
    assert not report.audit_validation(changed)
    changed = deepcopy(validation)
    changed["original_node_union"]["node_ids"] = []
    assert not report.audit_validation(changed)
    changed = deepcopy(validation)
    changed["coverage"]["combined_reference"]["sha256"] = "forged"
    assert not report.audit_validation(changed)
    assert not report.audit_validation({})


def test_dispatch_and_gate_reduction(tmp_path, monkeypatch):
    """REQ-VERIFY-8777: owned checks gate publication and cannot turn a block into success."""
    monkeypatch.setattr(
        runner, "validate", lambda raw, tracer, root: runner.private_validation(raw)
    )
    monkeypatch.setattr(
        archive, "failure_control", lambda *args: dict(passed=True, line345_executed=True)
    )
    output = tmp_path / (q.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    validation = json.loads(Path(value["validation_reference"]["path"]).read_bytes())
    validation.update(
        receipts=[dict(passed=True)],
        union=dict(passed=True, node_ids=[]),
        coverage=dict(complete=True),
        original_node_union=dict(passed=True, node_ids=work["expected_original_nodes"]),
    )
    assert report.build(work, validation, output)["evidence_methods_ready_score"] == 1
    work["preconditions_checked"]["passed"] = False
    assert report.build(work, validation, output)["verdict_class"] == "blocked"
    work["preconditions_checked"]["passed"] = True
    validation["receipts"][0]["passed"] = False
    assert report.build(work, validation, output)["verdict_class"] == "disqualified"
