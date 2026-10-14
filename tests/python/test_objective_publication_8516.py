"""REQ-REPORT-8516 and REQ-VERIFY-8516: new bytes retain historical failures."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import objective_publication_8516 as e
from carnot.reporting import objective_publication_runner_8516 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v709_execution import child


def test_original_mismatch_is_preserved(tmp_path):
    """SCENARIO-REPORT-8516-HISTORY: a new seal cannot authorize old changed bytes."""
    before = e.reference(e.OLD_PRIMARY)
    record = e.history_control(tmp_path)
    assert record["flagged_primary_reference"]["sha256"] == e.OLD_PIN
    assert record["sidecar_bound_sha256"] == e.OLD_SEAL
    assert record["stale_sidecar_rejected"]
    assert record["copied_flagged_envelope_rejected"]
    assert record["failed_consumer_reference"]
    assert e.reference(e.OLD_PRIMARY) == before


def test_independent_primitive_grid():
    """SCENARIO-VERIFY-8516-PRIMITIVES: unchanged kernels meet both error limits."""
    head = e.adapter.legacy.head()
    rows = e.primitive_grid(head)
    assert len(rows) == 180 and all(r["passed"] for r in rows)
    assert max(r["absolute_error"] for r in rows) <= 1e-7
    assert max(r["relative_error"] for r in rows) <= 1e-5
    assert {r["temperature"] for r in rows} == {0.5, 1, 2}
    assert {r["logit"] for r in rows} == {-20, -4, 0, 4, 20}
    assert {r["target"] for r in rows} == {0, 1}


def test_bound_publication_rejects_mutation_and_flagged_copy(tmp_path):
    """SCENARIO-REPORT-8516-PUBLICATION: qualification follows sealed consumer bytes."""
    output = tmp_path / (e.NAME + ".json")
    value = dict(
        experiment_id=8516,
        task_id=e.TASK,
        honest_verdict="complete_circular_positive_fixture",
        verdict_class="circular_positive",
        required_checks_passed=True,
        flagged_adversarial=False,
        objective_kernel_ready_score=1,
    )
    publication = publish_primary(output, value, lambda p: dict(passed=True))
    assert e.consumer(output, Path(publication["sidecar_path"]), replay=False)["passed"]
    output.write_bytes(output.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="stale_primary_hash"):
        read_bound_sidecar(output, Path(publication["sidecar_path"]))
    assert not e.consumer(output, Path(publication["sidecar_path"]), replay=False)["passed"]
    value["flagged_adversarial"] = True
    publication = publish_primary(output, value, lambda p: dict(passed=True))
    assert not e.consumer(output, Path(publication["sidecar_path"]), replay=False)["passed"]


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-VERIFY-8516-PRIMITIVES: qualification retains real child exit evidence."""
    parent = tmp_path_factory.mktemp("8516-measured")
    work = e.measure(e.ROOT, parent / "raw", parent / "private")
    receipts = [
        child(
            n,
            [sys.executable, "-c", "print('constructed test receipt')"],
            parent / "receipts",
            deadline=10,
        )
        for n in sorted(e.REQUIRED_CHECKS)
    ]
    return parent, work, receipts


def test_new_qualification_replays_all_primitives(measured):
    """SCENARIO-REPORT-8516-PUBLICATION: new readiness keeps old quarantine intact."""
    parent, work, receipts = measured
    assert not work["failures"]
    value = e.build(work, receipts, parent / "raw", parent / (e.NAME + ".json"))
    assert value["objective_kernel_ready_score"] == 1
    assert value["learning_qualification_ready_score"] == 1
    assert value["completed_count"] == value["intended_count"] == 216
    assert len(value["process_exit_rows"]) == 32
    assert all(r["actual_exit"] == 73 and r["passed"] for r in value["process_exit_rows"])
    assert len(value["restart_checks"]) == 36
    assert value["verdict_class"] == "circular_positive"
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["original_flagged_reference"]["sha256"] == e.OLD_PIN
    assert value["original_sidecar_reference"]["primary_sha256"] == e.OLD_SEAL
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert e.history.read(e.reference(e.OLD_PRIMARY))["flagged_adversarial"]


def test_owned_failure_and_findings_forbid_readiness(measured):
    """SCENARIO-REPORT-8516-PUBLICATION: no sanitized metadata can repair a failed check."""
    parent, original, receipts = measured
    for changed in (
        dict(original, adversarial_findings=[dict(code="AUTHENTICITY")]),
        dict(original, changed_reference_passed=False),
    ):
        value = e.build(changed, receipts, parent / "raw", parent / (e.NAME + ".json"))
        assert value["verdict_class"] == "disqualified"
        assert (
            value["learning_qualification_ready_score"]
            == value["objective_kernel_ready_score"]
            == 0
        )
    assert (
        e.build(original, [], parent / "raw", parent / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_cli_missing_date_and_private_boundaries(tmp_path):
    """SCENARIO-VERIFY-8516-CLI: private CLI controls have genuine process outcomes."""
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261013"])
    with pytest.raises(SystemExit):
        runner.main(["--private-control"])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert (
        runner.main(
            ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["objective_kernel_ready_score"] == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    value["learning_qualification_ready_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert runner.main(["--cold-replay", str(output)]) == 1


def test_new_binding_restores_original_identity():
    """SCENARIO-REPORT-8516-HISTORY: adaptation never rewrites the original producers."""
    before = e.legacy.NAME
    with e.binding():
        assert e.legacy.NAME == e.NAME and e.evidence.TASK == e.TASK
    assert e.legacy.NAME == before


def test_missing_methods_and_mismatch_records(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8516-HISTORY: missing operands differ from observed zero."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "private")
    assert work["failures"]
    assert any(f.get("observed") is None for f in work["failures"])
    receipts = [
        child(
            n,
            [sys.executable, "-c", "print('constructed test receipt')"],
            tmp_path / "receipts",
            deadline=10,
        )
        for n in sorted(e.REQUIRED_CHECKS)
    ]
    value = e.build(work, receipts, tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 216
    monkeypatch.setattr(e, "OLD_PRIMARY", tmp_path / "absent")
    with pytest.raises(FileNotFoundError):
        e.history_control(tmp_path / "missing")


def test_manifest_preserves_exact_disjoint_union(tmp_path):
    """SCENARIO-VERIFY-8516-VALIDATION: collect first and keep every original assertion."""
    plan = runner.manifest(tmp_path)
    ledger = e.read(e.reference(tmp_path / "historical_collection.json"))
    nodes = [n for group in ledger["shards"] for n in group]
    assert nodes == ledger["node_ids"] and len(nodes) == len(set(nodes))
    assert ledger["assertion_sources"] and ledger["total_deadline_s"] == 1800
    shards = [r for r in plan if r["name"].startswith("historical_shard_")]
    assert all(r["deadline"] <= 600 and "--no-cov" in r["argv"] for r in shards)
    assert e.REQUIRED_CHECKS <= {r["name"] for r in plan}
    assert "patch=subprocess,_exit" in (tmp_path / "coverage.ini").read_text()


def candidate(measured, tmp_path):
    """SCENARIO-REPORT-8516-PUBLICATION: fresh private bytes reuse measured immutable operands."""
    import shutil

    parent, work, receipts = measured
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copyfile(parent / "raw/update_adapter.py", raw / "update_adapter.py")
    atomic_json(raw / "measurement.json", work)
    path = tmp_path / (e.NAME + ".json")
    value = e.build(work, receipts, raw, path)
    atomic_json(path, value)
    return path, value


def test_fresh_consumer_and_replay(measured, tmp_path):
    """SCENARIO-REPORT-8516-PUBLICATION: exact new bytes pass actual downstream checks."""
    path, value = candidate(measured, tmp_path)
    assert e.replay(path)
    published = publish_primary(path, value, lambda p: dict(passed=e.replay(p)))
    receipt = e.consumer(path, Path(published["sidecar_path"]))
    assert receipt["passed"] and receipt["primary_sha256"] == published["primary_sha256"]
    side = Path(published["sidecar_path"])
    bound = json.loads(side.read_bytes())
    bound["report"]["passed"] = False
    atomic_json(side, bound)
    assert not e.consumer(path, side, replay=False)["passed"]
    bound["report"]["passed"] = True
    atomic_json(side, bound)
    changed = deepcopy(value)
    changed["learning_qualification_ready_score"] = 99
    changed["reproducibility_checksum"] = e.checksum(changed)
    published = publish_primary(path, changed, lambda p: dict(passed=True))
    assert not e.consumer(path, Path(published["sidecar_path"]))["passed"]


@pytest.mark.parametrize(
    "operand", ["grid", "old_flag", "old_pin", "copied", "control", "fixture", "union"]
)
def test_rehashed_primitive_substitutions(measured, tmp_path, operand):
    """SCENARIO-VERIFY-8516-CLI: repaired outer hashes do not authorize different operands."""
    path, value = candidate(measured, tmp_path)
    if operand == "grid":
        rows = e.read(value["derivative_grid_reference"])
        rows[0]["absolute_error"] = 999
        atomic_json(tmp_path / "changed_grid.json", rows)
        value["derivative_grid_reference"] = e.reference(tmp_path / "changed_grid.json")
    elif operand in ("old_flag", "old_pin", "copied"):
        old = e.read(value["history_reference"])
        if operand == "old_flag":
            old["flagged_primary_reference"]["flagged_adversarial"] = False
        elif operand == "old_pin":
            old["sidecar_bound_sha256"] = "wrong"
        else:
            old["copied_primary_reference"]["path"] = str(tmp_path / "absent")
        atomic_json(tmp_path / "changed_history.json", old)
        value["history_reference"] = e.reference(tmp_path / "changed_history.json")
    elif operand == "control":
        control = e.read(value["changed_reference_reference"])
        control["original_terminal_reference"]["sha256"] = "wrong"
        atomic_json(tmp_path / "changed_control.json", control)
        value["changed_reference_reference"] = e.reference(tmp_path / "changed_control.json")
    elif operand == "fixture":
        atomic_json(tmp_path / "changed_fixture.json", [])
        value["primitive_fixture_reference"] = e.reference(tmp_path / "changed_fixture.json")
    else:
        atomic_json(
            tmp_path / "changed_ledger.json", dict(node_ids=["node"], shards=[["node", "node"]])
        )
        value["historical_collection_reference"] = e.reference(tmp_path / "changed_ledger.json")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(path, value)
    assert not e.replay(path)


def test_worker_historical_modes_and_summary(tmp_path):
    """SCENARIO-VERIFY-8516-VALIDATION: unchanged workers and historical exits remain real."""
    from carnot.reporting.v709_execution import child

    config = tmp_path / "configuration.json"
    atomic_json(config, e.adapter.config(e.adapter.legacy.head()))
    assert runner.main(["--worker", str(tmp_path / "store"), "--configuration", str(config)]) == 0
    assert (
        runner.main(
            ["--state-replay", str(tmp_path / "store/final.json"), "--configuration", str(config)]
        )
        == 0
    )
    script = tmp_path / "private_script.py"
    script.write_text("print('actual private historical script',flush=True)\n")
    assert runner.main(["--historical-exec", str(script)]) == 0
    script.write_text("raise SystemExit(7)\n")
    assert runner.main(["--historical-exec", str(script)]) == 7
    script.write_text("raise SystemExit('private string exit')\n")
    assert runner.historical([str(script)], script=True) == 0
    assert runner.summarize(tmp_path / "absent", tmp_path / "receipts") == 1
    node = "tests/python/test_objective_checkpoint_8489.py::test_log_loss_reference_grid_and_rejection_controls"
    assert runner.main(["--historical-tests", *runner.COMMON, node]) == 0
    original = e.history.ORIGINAL_TESTS[0] + "::test_transition_controls_and_invalid_configuration"
    assert runner.historical([*runner.COMMON, original]) == 0
    ledger = tmp_path / "ledger.json"
    groups = [[node], [node + "-1"], [node + "-2"]]
    atomic_json(ledger, dict(node_ids=[n for group in groups for n in group], shards=groups))
    for i, group in enumerate(groups):
        receipt = child(
            f"historical_shard_{i}",
            [sys.executable, "-c", f"print({group[0]!r}+' PASSED')"],
            tmp_path / "receipts",
            deadline=10,
        )
        assert receipt["passed"]
    assert (
        runner.main(["--historical-summary", str(ledger), "--receipts", str(tmp_path / "receipts")])
        == 0
    )
    row = json.loads((tmp_path / "receipts/historical_shard_0.receipt.json").read_bytes())
    row["timed_out"] = True
    atomic_json(tmp_path / "receipts/historical_shard_0.receipt.json", row)
    assert runner.summarize(ledger, tmp_path / "receipts") == 1


def test_historical_descendant_prefix_and_store_selection(tmp_path):
    """SCENARIO-VERIFY-8516-VALIDATION: old CLI descendants retain coverage prefixes and exits."""
    script = tmp_path / "child.py"
    args = [sys.executable, "-u", str(e.ROOT / e.history.CLI), "--deliberate-error"]
    script.write_text(
        f"import subprocess\nassert subprocess.run({args!r},check=False).returncode == 7\n"
    )
    assert runner.historical([str(script)], script=True) == 0
    node = old_node = (
        "tests/python/test_block_feedback_state_8433.py::test_independent_basis_nonzero_and_frozen"
    )
    rest = "tests/python/test_current_work_receipt.py::test_scenario_report_7395_current_counts_a_failed_owned_load"
    assert runner.historical([*runner.COMMON, node, rest]) == 0


def test_missing_summary_arguments(tmp_path):
    """SCENARIO-VERIFY-8516-CLI: malformed summary requests cannot qualify validation."""
    with pytest.raises(SystemExit):
        runner.main(["--historical-summary", str(tmp_path / "ledger")])


def test_added_derivative_failure_retains_its_row(measured, tmp_path):
    """SCENARIO-VERIFY-8516-PRIMITIVES: failed pinned arithmetic remains a visible failed cell."""
    parent, work, receipts = measured
    changed = deepcopy(work)
    rows = e.read(changed["derivative_grid_reference"])
    rows[0]["passed"] = False
    atomic_json(tmp_path / "grid.json", rows)
    changed["derivative_grid_reference"] = e.reference(tmp_path / "grid.json")
    value = e.build(changed, receipts, parent / "raw", parent / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["failed_count"] == 1


@pytest.mark.parametrize("field", ["bound", "methods_ready_score", "learning_inputs_ready_score"])
def test_methods_bound_consumer_rejections(tmp_path, monkeypatch, field):
    """SCENARIO-REPORT-8516-HISTORY: original bytes and current input gates remain independent."""
    monkeypatch.setattr(e.history, "_AUTHENTICATE", lambda *a: dict(refs=[], failures=[]))
    if field == "bound":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    else:
        original = e.reader_receipt
        monkeypatch.setattr(
            e,
            "reader_receipt",
            lambda task, root, **kw: (
                dict(passed=False) if kw["field"] == field else original(task, root, **kw)
            ),
        )
    assert e.authenticate(e.ROOT, tmp_path)["failures"]


def test_history_mismatch_and_changed_text_rejections(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8516-HISTORY: each immutable historical operand remains necessary."""
    path, value = candidate(measured, tmp_path)
    control = e.read(value["changed_reference_reference"])
    changed = tmp_path / "changed.txt"
    changed.write_text("different primitive text\n")
    control["changed_file_reference"]["path"] = str(changed)
    atomic_json(tmp_path / "control.json", control)
    value["changed_reference_reference"] = e.reference(tmp_path / "control.json")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(path, value)
    assert not e.replay(path)
    monkeypatch.setattr(e, "OLD_SEAL", "wrong")
    with pytest.raises(ValueError, match="original_publication_mismatch"):
        e.history_control(tmp_path / "wrong-seal")


def test_historical_collection_retained_before_replay(tmp_path):
    """SCENARIO-VERIFY-8516-VALIDATION: collection evidence survives private scratch cleanup."""
    private = tmp_path / "private"
    atomic_json(private / "historical_collection.json", dict(node_ids=["node"], shards=[["node"]]))
    work = e.measure(tmp_path / "absent", tmp_path / "raw", private / "checkpoints")
    assert work["historical_collection_reference"]
    receipts = [
        child(
            n,
            [sys.executable, "-c", "print('constructed test receipt')"],
            tmp_path / "receipts",
            deadline=10,
        )
        for n in e.REQUIRED_CHECKS
    ]
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, receipts, tmp_path / "raw", output))
    assert e.replay(output)


def test_runner_plan_and_genuine_coverage_archive(measured, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8516-VALIDATION: real receipt and trace files survive complete publication."""
    import coverage
    import shutil

    parent, original, _ = measured
    private = tmp_path / "private"
    private.mkdir()
    child(
        "actual_collection_fixture",
        [sys.executable, "-c", "print('genuine collection fixture')"],
        private / "collection",
        deadline=10,
    )
    trace = coverage.Coverage(data_file=str(private / ".coverage"), config_file=False)
    trace.start()
    e.checksum(dict(actual_fixture_trace=True))
    trace.stop()
    trace.save()
    trace.json_report(outfile=str(private / "coverage.json"))
    plan = [
        dict(
            name=n,
            argv=[sys.executable, "-c", "print('constructed validation fixture')"],
            deadline=10,
        )
        for n in sorted(e.REQUIRED_CHECKS)
    ]
    plan.insert(
        0,
        dict(
            name="historical_shard_fixture",
            argv=[sys.executable, "-c", "print('constructed shard fixture')"],
            deadline=10,
        ),
    )
    monkeypatch.setattr(runner, "manifest", lambda p: plan)

    def fixture_measure(root, raw, checkpoint):
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / "raw/update_adapter.py", raw / "update_adapter.py")
        return deepcopy(original)

    monkeypatch.setattr(e, "measure", fixture_measure)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.run(e.ROOT, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["learning_qualification_ready_score"] == 1
    assert value["raw_shard_hashes"]
    actual = e.consumer
    monkeypatch.setattr(
        e, "consumer", lambda p, s, **kw: dict(passed=False) if p == output else actual(p, s, **kw)
    )
    with pytest.raises(ValueError, match="post_publication_consumer_failed"):
        runner.run(e.ROOT, output, private)


def test_input_hash_preconditions_and_original_pin(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8516-HISTORY: fixed inputs authenticate before any numeric measurement."""
    monkeypatch.setattr(e.history, "_AUTHENTICATE", lambda *a: dict(refs=[], failures=[]))
    monkeypatch.setattr(e, "FIXTURE_PIN", "wrong")
    result = e.authenticate(e.ROOT, tmp_path / "auth")
    assert any(f["field"] == "sha256" for f in result["failures"])
    monkeypatch.undo()
    path, _ = candidate(measured, tmp_path)
    monkeypatch.setattr(e, "OLD_PIN", "wrong")
    assert not e.replay(path)
