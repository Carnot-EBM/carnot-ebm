"""REQ-REPORT-8415 / REQ-VERIFY-8415: immutable evidence cannot grant new science."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v724_capstone as runner
from carnot.reporting import v724_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def cli(*args):
    """Actual processes expose CLI, publication and memory failures hidden by mocks."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("v724-root")
    for name in [e.DESIGN, e.ACTIVE, e.PROTOCOL]:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((e.ROOT / name).read_bytes())
    output = root.parent / (e.NAME + ".json")
    result = cli("--root", root, "--output", output, "--private-control")
    assert result.returncode == 0, result.stdout + result.stderr
    return root, output, json.loads(output.read_bytes())


def test_private_controls(private, tmp_path):
    """SCENARIO-VERIFY-8415-CONTROLS: external absence blocks; owned failure disqualifies."""
    root, output, value = private
    assert value["verdict_class"] == "blocked"
    assert value["missing_output_count"] == 13
    assert value["actual_executed_task_count"] == 1
    assert value["intended_count"] == len(value["rows"]) == 14
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    assert {"cold_missing", "cold_deliberate_error"} <= {
        r["name"] for r in value["validation_receipts"]
    }
    assert e.replay(output) and cli("--cold-replay", output).returncode == 0
    failed = tmp_path / (e.NAME + ".json")
    result = cli("--root", root, "--output", failed, "--private-control", "--failed-child")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"
    assert cli("--date", "20261010").returncode == 2
    assert cli("--private-control").returncode == 2
    assert cli("--deliberate-error").returncode == 7
    assert cli("--cold-replay", tmp_path / "missing").returncode == 1


@pytest.mark.parametrize(
    "field", ["rows", "H1", "g1", "canonical_tasks_sha256", "validation_receipts"]
)
def test_rehashed_tamper(private, tmp_path, field):
    """SCENARIO-VERIFY-8415-CONTROLS: a new checksum cannot authorize altered conclusions."""
    value = deepcopy(private[2])
    value[field] = "forged"
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "tamper.json"
    atomic_json(path, value)
    assert not e.replay(path)


def test_bound_authority_and_manifest(private, tmp_path):
    """SCENARIO-REPORT-8415-ACCOUNTING: full objects and frozen command vectors bind the task."""
    work = e.load(private[2]["work_reference"])
    changed = deepcopy(work)
    changed["tasks"][0]["prompt"] += "forged"
    with pytest.raises(ValueError, match="authority"):
        e.reduce(changed, private[2]["validation_receipts"])
    with pytest.raises(ValueError, match="authority"):
        e.measure(tmp_path / "missing-root", tmp_path / "missing-raw")
    plan = runner.manifest(tmp_path / "plan")
    assert plan[0]["name"] == "owned_tests"
    assert {"private_E2E018_consumers", "private_E2E021", "full_contract_mutations"} <= {
        p["name"] for p in plan
    }
    assert plan[0]["deadline"] == 900
    assert all(p["deadline"] <= 900 for p in plan)


@pytest.fixture(scope="module")
def natural(tmp_path_factory):
    directory = tmp_path_factory.mktemp("v724-natural")
    return e.measure(e.ROOT, directory / "raw")


def test_natural_sources(natural, private, tmp_path):
    """SCENARIO-REPORT-8415-SCIENCE: replay preserves failures, exposed nulls and branch scope."""
    value = e.build(
        natural,
        private[2]["validation_receipts"],
        tmp_path / "built",
        tmp_path / (e.NAME + ".json"),
    )
    assert value["actual_executed_task_count"] == 12
    assert value["pre_gate_count"] == 2 and value["cascade_skip_count"] == 0
    assert value["missing_output_count"] == 0
    assert value["failed_count"] == 3
    assert value["rows"][10]["verdict_class"] == "disqualified"
    assert value["rows"][10]["original_verdict_class"] == "null"
    assert (
        sum(
            value[k]
            for k in ["completed_count", "failed_count", "censored_count", "excluded_count"]
        )
        == 14
    )
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["H1"]["bootstrap_summary"]["all_intended"]["mean_gain"] == -0.00390625
    assert value["H2"]["block_bootstrap_summary"]["mean_gain"] == 0
    assert value["retirements"] == []
    assert len(value["historical_v723"]["failed_consumer_node_ids"]) == 5
    assert value["historical_v723"]["failed_count"] == 5
    assert not value["repository_health"]["current_suite_launched"]
    assert all(not g["closed"] for g in value["three_prd_gaps"])
    assert all(r["continuation_condition"] for r in value["rows"])
    assert all(r["metrics"] is None or "rows" not in r["metrics"] for r in value["rows"])
    assert value["rows"][5]["metrics"]["primitive_rows_reference"]["json_pointer"] == "/rows"
    assert all(r["memory"]["passed"] for r in natural["slots"] if "memory" in r)
    assert any(
        r.get("source_path", "").endswith("ops/exclusion_manifest.yaml")
        for r in value["source_artifact_hashes"]
    )
    path = tmp_path / "natural.json"
    atomic_json(path, value)
    assert e.replay(path)


def test_worker_authentication_and_receipt_tamper(private, tmp_path):
    """SCENARIO-VERIFY-8415-CONTROLS: fresh invalid requests retain measured memory and real exits."""
    work = e.load(private[2]["work_reference"])
    task = work["tasks"][6]
    source = next((e.ROOT / "results").glob("experiment_8408_*.json"))
    primary = e.freeze(source, tmp_path / "custody")
    request, output = tmp_path / "request.json", tmp_path / "worker.json"
    atomic_json(request, dict(task=task, task_sha256="wrong", primary=primary))
    assert cli("--worker-request", request, "--worker-output", output).returncode == 1
    assert json.loads(output.read_bytes())["memory"]["peak_bound_mb"] == 1500
    atomic_json(request, dict(task=task, task_sha256=canonical_hash(task), primary=primary))
    assert cli("--worker-request", request, "--worker-output", output).returncode == 0
    changed = deepcopy(private[2])
    changed["validation_receipts"][0]["argv"].append("forged")
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "changed-receipt.json"
    atomic_json(path, changed)
    assert not e.replay(path)
    changed["reproducibility_checksum"] = "wrong"
    atomic_json(path, changed)
    assert not e.replay(path)


@pytest.mark.parametrize(
    "before,after,expected", [(1000, 1500, 0), (1500, 1501, 1), (500, 1001, 1)]
)
def test_exact_worker_bounds(private, tmp_path, monkeypatch, before, after, expected):
    """SCENARIO-VERIFY-8415-MEMORY: either exceeded fresh-child bound rejects qualification."""
    task = e.load(private[2]["work_reference"])["tasks"][6]
    primary = e.freeze(
        next((e.ROOT / "results").glob("experiment_8408_*.json")), tmp_path / "custody"
    )
    request, output = tmp_path / "request.json", tmp_path / "worker.json"
    atomic_json(request, dict(task=task, task_sha256=canonical_hash(task), primary=primary))
    samples = iter(
        [
            dict(current_rss_mb=before, peak_rss_mb=before),
            dict(current_rss_mb=after, peak_rss_mb=after),
        ]
    )
    monkeypatch.setattr(e, "memory", lambda: next(samples))
    assert e.worker_process(request, output) == expected
    assert json.loads(output.read_bytes())["memory"]["growth_mb"] == after - before


def test_absence_and_failed_owned_checks(private, tmp_path):
    """SCENARIO-REPORT-8415-ACCOUNTING: a skip supplies no primary, metric or independent example."""
    work = e.load(private[2]["work_reference"])
    task = work["tasks"][0]
    absent = e.freeze(tmp_path / "missing.json", tmp_path / "custody")
    row = e.slot(task, absent, "actual recorded cascade")
    assert row["disposition"] == "cascade_skip" and row["metrics"] is None
    work["memory_measurements"]["parent_growth_mb"] = 501
    assert e.reduce(work, private[2]["validation_receipts"])["verdict_class"] == "disqualified"
    assert e.memory()["measurement_source"] == "/proc/self/status"


def test_historical_and_branch_tamper(natural, private, tmp_path):
    """SCENARIO-VERIFY-8415-HISTORY: changed primitive tasks and historical means cannot grant readiness."""
    for field in ["history", "slots"]:
        work = deepcopy(natural)
        if field == "history":
            work["history"][0]["selected"]["H1"]["bootstrap_summary"]["all_intended"][
                "mean_gain"
            ] = 1
        else:
            work["slots"][0]["task"] = deepcopy(work["slots"][0]["task"])
            work["slots"][0]["task"]["prompt"] += "forged"
        value = e.build(
            work, private[2]["validation_receipts"], tmp_path / field, tmp_path / (e.NAME + ".json")
        )
        path = tmp_path / (field + ".json")
        atomic_json(path, value)
        assert not e.replay(path)


def test_owned_runner_path(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8415-CONTROLS: real private execution archives genuine coverage and validators."""
    import coverage

    scratch = tmp_path / "owned"
    scratch.mkdir(mode=0o700)
    config = scratch / "coverage.ini"
    config.write_text("[run]\nparallel=true\ndata_file=" + str(scratch / ".coverage") + "\n")
    covered = coverage.Coverage(config_file=str(config))
    covered.start()
    compiled = compile("proof = 1\n", str(tmp_path / "coverage-proof.py"), "exec")
    exec(compiled, {})
    covered.stop()
    covered.save()
    monkeypatch.setattr(
        runner,
        "manifest",
        lambda _: [
            dict(
                name="actual_private_child",
                argv=[sys.executable, "-u", "-c", "print('actual child', flush=True)"],
                deadline=10,
                expected=0,
                scope="owned",
            )
        ],
    )
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(private[0], output, scratch) == 0
    assert list((output.parent / "raw" / output.stem).rglob(".coverage*"))
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert any(".coverage" in r["path"] for r in value["raw_shard_hashes"])
    for name, deadline in [("terminal_cold", 240), ("ordinary_child", 10)]:
        receipt = runner.terminal_child(
            name,
            [sys.executable, "-u", "-c", "print('actual child', flush=True)"],
            tmp_path / name,
            deadline=10,
        )
        assert receipt["passed"] and receipt["deadline_s"] == deadline


def test_bound_staged_authority_and_positive_reduction(private, tmp_path):
    """SCENARIO-REPORT-8415-ACCOUNTING: current YAML cannot override the sealed full objects."""
    work = e.load(private[2]["work_reference"])
    stage = tmp_path / e.STAGED
    stage.write_bytes((e.ROOT / e.ACTIVE).read_bytes())
    work["authority_refs"][1] = e.freeze(stage, tmp_path / "custody")
    assert e.authority(work) == work["tasks"]
    import yaml

    mutated = yaml.safe_load(stage.read_bytes())
    mutated["tasks"][0]["prompt"] += "forged"
    stage.write_text(yaml.safe_dump(mutated))
    work["authority_refs"][1] = e.freeze(stage, tmp_path / "changed")
    with pytest.raises(ValueError, match="staged_task_authority"):
        e.authority(work)
    work = e.load(private[2]["work_reference"])
    for slot in work["slots"]:
        slot.update(eligible=True, verdict_class="null", disposition="executed_producer")
    work["checks"] = []
    assert e.reduce(work, private[2]["validation_receipts"])["verdict_class"] == "null"


def test_primary_and_pregate_rejections(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8415-CONTROLS: declared gates and actual operands must agree exactly."""
    task = e.load(private[2]["work_reference"])["tasks"][6]
    primary = e.freeze(
        next((e.ROOT / "results").glob("experiment_8408_*.json")), tmp_path / "custody"
    )
    value = e.load(primary)
    for control, message in [
        ("identity", "pre_gate_identity"),
        ("contract", "pre_gate_contract"),
        ("operand", "pre_gate_operand"),
        ("pin", "primary_pin"),
    ]:
        changed = deepcopy(value)
        ref = deepcopy(primary)
        if control == "identity":
            changed["experiment"] = 0
        elif control == "contract":
            changed["gates_evaluated"][0]["op"] = "!="
        elif control == "operand":
            changed["gates_evaluated"][0]["actual"] = 9
        else:
            ref["sha256"] = "sha256:" + "0" * 64
        with monkeypatch.context() as patch:
            actual_load = e.load
            patch.setattr(
                e, "load", lambda r: changed if r["path"] == primary["path"] else actual_load(r)
            )
            with pytest.raises(ValueError, match=message):
                e.outcome(task, ref, tmp_path / control)
    task = deepcopy(task)
    task["id"] = "exp8404-wrong-task"
    producer = e.freeze(
        next((e.ROOT / "results").glob("experiment_8404_*.json")), tmp_path / "producer"
    )
    with pytest.raises(ValueError, match="task_identity"):
        e.outcome(task, producer, tmp_path / "identity")


def test_sealed_operand_tamper(tmp_path):
    """SCENARIO-VERIFY-8415-CONTROLS: a false source digest never becomes an authenticated operand."""
    source = tmp_path / "source.json"
    atomic_json(source, dict(actual=True))
    with pytest.raises(ValueError, match="source_sha256"):
        e.seal(
            dict(source_artifact_hashes=[dict(path=str(source), sha256="sha256:" + "0" * 64)]),
            tmp_path / "custody",
        )
    aliases = e.references(
        dict(
            authority=dict(
                source_path="/original/authority",
                snapshot_path=str(source),
                sha256=e.freeze(source, tmp_path / "sealed")["sha256"],
            )
        )
    )
    assert aliases[0]["path"] == aliases[0]["source_path"] == "/original/authority"


def test_missing_protocol_ambiguous_primary_and_history(private, tmp_path):
    """SCENARIO-VERIFY-8415-CONTROLS: missing protocols, ambiguous identities and foreign terminals reject before science."""
    import shutil

    for control, expected in [
        ("protocol", "protocol_pin"),
        ("ambiguous", "ambiguous_primary"),
        ("history_hash", "historical_primary_pin"),
        ("history_terminal", "historical_terminal_binding"),
    ]:
        root = tmp_path / control
        shutil.copytree(private[0], root)
        results = root / "results"
        results.mkdir()
        if control == "protocol":
            (root / e.PROTOCOL).unlink()
        elif control == "ambiguous":
            for name in ["experiment_8402_one.json", "experiment_8402_two.json"]:
                atomic_json(results / name, dict(negative_identity_control=True))
        elif control == "history_hash":
            atomic_json(results / "experiment_8361_wrong.json", dict(negative_digest_control=True))
        else:
            source = e.ROOT / "results/experiment_8361_v721_utility_audit_qualification.json"
            shutil.copyfile(source, results / source.name)
        with pytest.raises(ValueError, match=expected):
            e.measure(root, tmp_path / (control + "-raw"))


def test_corrupt_fixture_and_terminal(private, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8415-HISTORY: unrelated roots and failed terminal validation cannot qualify a reader."""
    with monkeypatch.context() as patch:
        patch.setattr(runner.contract, "check_fixtures", lambda _: False)
        with pytest.raises(ValueError, match="historical_fixture_custody"):
            runner.manifest(tmp_path / "wrong-fixture")
    task = e.load(private[2]["work_reference"])["tasks"][0]
    primary = e.freeze(e.ROOT / task["deliverable"], tmp_path / "primary")
    summary = e.outcome(task, primary, tmp_path / "closure")
    original_load = e.load

    def changed_side(ref):
        value = original_load(ref)
        if ref["path"] == summary["closure"][2]["path"]:
            value["report"]["passed"] = False
        return value

    monkeypatch.setattr(e, "load", changed_side)
    with pytest.raises(ValueError, match="terminal_binding"):
        e.outcome(task, primary, tmp_path / "changed-terminal", summary["closure"])


@pytest.mark.parametrize(
    "control", ["historical_verdict", "failed_replay_child", "forged_disposition"]
)
def test_cold_primitive_failures(natural, private, tmp_path, monkeypatch, control):
    """SCENARIO-VERIFY-8415-CONTROLS: independently rehashed primitive work cannot bless history or owned child failures."""
    work = deepcopy(natural)
    if control == "historical_verdict":
        work["history"][0]["honest_verdict"] = "complete_null_forged_history"
    elif control == "forged_disposition":
        work["slots"][0]["disposition"] = "forged"
    value = e.build(
        work, private[2]["validation_receipts"], tmp_path / "built", tmp_path / (e.NAME + ".json")
    )
    path = tmp_path / "primitive.json"
    atomic_json(path, value)
    if control == "failed_replay_child":
        monkeypatch.setattr(e, "invoke", lambda *args, **kwargs: ({}, dict(passed=False)))
    assert not e.replay(path)
