"""REQ-REPORT-8530, REQ-VERIFY-8530: count state effects independently of update counters."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v733_update_accounting as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child


def test_coordinate_semantics():
    """SCENARIO-REPORT-8530-ACCOUNTING: writes, changes and decay have different operands."""
    before = dict(parameters=[1.0, 2.0, 0.0], decay_scale=1.0)
    after = dict(parameters=[2.0, 2.0, 0.0], decay_scale=0.5)
    row = e.coordinate_row(before, after, [0, 0, 1], event="a", namespace="head")
    assert row["physical_coordinate_writes"] == 3
    assert row["changed_coordinate_count"] == 1
    assert row["effective_coefficient_changes"] == 1
    assert row["decay_scale_changes"] == 1
    assert row["changed_coordinates"] == ["head:0"]
    zero = e.coordinate_row(before, before, [0, 0], event="b", namespace="head")
    decay = e.coordinate_row(before, dict(before, decay_scale=0.9), [], event="c", namespace="head")
    total = e.aggregate([row, zero, decay])
    assert total["applied_update_events"] == 3
    assert total["physical_coordinate_writes"] == 5
    assert total["changed_coordinate_count"] == total["unique_changed_coordinates"] == 1
    assert decay["changed_coordinate_count"] == 0 and decay["effective_coefficient_changes"] == 2
    with pytest.raises(ValueError, match="coordinate_contract"):
        e.coordinate_row(before, dict(after, parameters=[]), [], event="d", namespace="head")
    with pytest.raises(ValueError, match="coordinate_contract"):
        e.coordinate_row(before, after, [9], event="d", namespace="head")


@pytest.mark.parametrize("case", ["missing", "zero", "wrong_task", "wrong_hash", "stale", "valid"])
def test_complete_diagnostics(tmp_path, case):
    """SCENARIO-REPORT-8530-DIAGNOSTICS: absence is never coerced to zero."""
    path = tmp_path / "experiment_8516_fixture.json"
    value = dict(
        experiment_id=8516,
        task_id=e.UPSTREAM_TASK,
        honest_verdict="complete_circular_positive_fixture",
        verdict_class="circular_positive",
        learning_qualification_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
    )
    if case == "zero":
        value["learning_qualification_ready_score"] = 0
    if case == "wrong_task":
        value["task_id"] = "exp8516-foreign"
    pub = publish_primary(path, value, lambda p: dict(passed=True))
    pin = e.reference(path)["sha256"]
    side = Path(pub["sidecar_path"])
    if case == "missing":
        path.unlink()
    if case == "wrong_hash":
        pin = "sha256:wrong"
    if case == "stale":
        path.write_bytes(path.read_bytes() + b"\n")
        pin = e.reference(path)["sha256"]
    rows = e.qualification_diagnostics(path, pin, side)
    assert all(e.DIAGNOSTIC_FIELDS <= r.keys() for r in rows)
    assert (not rows) == (case == "valid")
    if case == "missing":
        assert rows[0]["observed"] is None
    if case == "zero":
        assert any(
            r["field"] == "learning_qualification_ready_score" and r["observed"] == 0 for r in rows
        )


def test_normalization_preserves_failure():
    """SCENARIO-REPORT-8530-DIAGNOSTICS: a missing legacy field cannot hide a later zero."""
    original = [
        dict(artifact_field="authority", upstream_id="old", observed=None),
        dict(field="learning_qualification_ready_score", observed=0),
    ]
    rows = e.diagnostics(original)
    assert [r["field"] for r in rows] == ["authority", "learning_qualification_ready_score"]
    assert original[0].get("field") is None and rows[1]["observed"] == 0


def test_real_exit_resume_and_repeated_delivery(tmp_path):
    """SCENARIO-REPORT-8530-ACCOUNTING: an actual process exits73 after a durable release."""
    from carnot.reporting import v733_update_fixture as f

    cfg = f.configuration()
    atomic_json(tmp_path / "cfg.json", cfg)
    argv = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--configuration",
        str(tmp_path / "cfg.json"),
    ]
    baseline = f.worker(cfg, tmp_path / "baseline")
    died = child(
        "exit",
        [*argv, "--fixture-worker", str(tmp_path / "resume"), "--crash"],
        tmp_path / "logs",
        expected=73,
        deadline=60,
    )
    resumed = child(
        "resume",
        [*argv, "--fixture-worker", str(tmp_path / "resume")],
        tmp_path / "logs",
        deadline=60,
    )
    assert died["passed"] and resumed["passed"]
    assert json.loads((tmp_path / "resume/final.json").read_bytes()) == baseline
    rows = json.loads((tmp_path / "resume/accounting.json").read_bytes())
    assert rows == json.loads((tmp_path / "baseline/accounting.json").read_bytes())
    assert len(rows) == 48 and e.aggregate(rows)["changed_coordinate_count"] > 48
    assert f.worker(cfg, tmp_path / "resume") == baseline


def test_historical_reconciliation():
    """SCENARIO-REPORT-8530-ACCOUNTING: original serialized state refutes the1800-change summary."""
    historical = e.historical_accounting()
    assert historical["applied_update_events"] == 1800
    assert historical["changed_coordinate_count"] > 1800
    assert historical["physical_coordinate_writes"] >= historical["changed_coordinate_count"]
    assert historical["effective_coefficient_changes"] >= historical["changed_coordinate_count"]
    assert historical["reconciliation_passed"]
    assert historical["metric_reconciliation_rows"][0]["old_value"] == 1800


def test_cli_rejections_and_private_publication(tmp_path):
    """SCENARIO-VERIFY-8530-CONTROLS: real private bytes replay and reject repaired tampering."""
    from carnot.reporting import v733_update_accounting_runner as runner

    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert runner.main(["--fixture-worker", str(tmp_path)]) == 1
    assert runner.main(["--private-control"]) == 2
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261014"])
    output = tmp_path / (e.NAME + ".json")
    assert (
        runner.main(
            ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["online_adapter_ready_score"] == 0 and value["verdict_class"] == "disqualified"
    assert e.replay(output)
    changed = deepcopy(value)
    changed["changed_coordinate_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    actual = tmp_path / "actual-cli" / (e.NAME + ".json")
    receipt = child(
        "actual_private_cli",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--date",
            "20261015",
            "--private-control",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(actual),
        ],
        tmp_path / "actual_cli_logs",
        deadline=180,
    )
    assert receipt["passed"] and e.replay(actual)


def test_original_diagnostic_assertion_and_bad_sidecar(tmp_path):
    """SCENARIO-REPORT-8530-DIAGNOSTICS: keep the strict original assertion and its zero."""
    path = tmp_path / "inputs/results" / (e.upstream.NAME + ".json")
    atomic_json(path, dict(learning_qualification_ready_score=0))
    result = e.inspect_adapter(tmp_path / "inputs", tmp_path / "raw", tmp_path)
    assert any(
        f["field"] == "learning_qualification_ready_score" and f["observed"] == 0
        for f in result["failures"]
    )
    assert all(e.DIAGNOSTIC_FIELDS <= f.keys() for f in result["failures"])
    primary = tmp_path / "experiment_8516_fixture.json"
    value = dict(
        experiment_id=8516,
        task_id=e.UPSTREAM_TASK,
        honest_verdict="complete_circular_positive_fixture",
        verdict_class="circular_positive",
        learning_qualification_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
    )
    pub = publish_primary(primary, value, lambda p: dict(passed=True))
    side = Path(pub["sidecar_path"])
    bound = json.loads(side.read_bytes())
    bound["report"]["passed"] = False
    atomic_json(side, bound)
    assert (
        e.qualification_diagnostics(primary, e.reference(primary)["sha256"], side)[0]["field"]
        == "bound_publication"
    )


@pytest.fixture(scope="module")
def measured8530(tmp_path_factory):
    """SCENARIO-VERIFY-8530-CONTROLS: run the real adapter once for verdict and tamper cases."""
    directory = tmp_path_factory.mktemp("qualified8530")
    work = e.measure(e.ROOT, directory / "raw")
    assert not work["failures"]
    node = (
        __file__.split("/tests/")[0]
        + "/tests/python/test_v733_update_accounting_8530.py::test_coordinate_semantics"
    )
    relative = node.split("/tests/")[1]
    relative = "tests/" + relative
    receipt = child(
        "minimal_union",
        [sys.executable, "-m", "pytest", "-n", "0", "-o", "addopts=", "--no-cov", "-vv", relative],
        directory / "receipts",
        deadline=60,
    )
    receipt["collected_node_ids"] = [relative]
    names = [
        "owned_tests",
        "coverage_report",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    ]
    receipts = [
        child(
            name,
            [sys.executable, "-c", "print('constructed unit validation receipt')"],
            directory / "receipts",
            deadline=20,
        )
        for name in names
    ] + [receipt]
    work["collected_node_ids"] = [relative]
    atomic_json(directory / "raw/measurement.json", work)
    return directory, work, receipts


def test_verdicts_and_consumer_binding(measured8530):
    """SCENARIO-REPORT-8530-DIAGNOSTICS: qualification cannot survive owned failure or stale bytes."""
    directory, work, receipts = measured8530
    output = directory / (e.NAME + ".json")
    value = e.build(work, receipts, directory / "raw", output)
    assert (
        value["online_adapter_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    )
    pub = publish_primary(output, value, lambda p: dict(passed=e.replay(p)))
    assert e.stream_consumer(output, Path(pub["sidecar_path"]))["passed"]
    output.write_bytes(output.read_bytes() + b"\n")
    assert not e.stream_consumer(output, Path(pub["sidecar_path"]))["passed"]
    changed = deepcopy(work)
    changed["failures"] = [e.old.gate(output, "sha256:missing", "readiness", 1, None)]
    assert e.build(changed, receipts, directory / "raw", output)["verdict_class"] == "blocked"
    changed["adversarial_findings"] = [dict(kind="owned_failure")]
    assert e.build(changed, receipts, directory / "raw", output)["verdict_class"] == "disqualified"
    failed = [dict(r, passed=False) for r in receipts]
    assert e.build(work, failed, directory / "raw", output)["online_adapter_ready_score"] == 0


def test_rehashed_primitives_and_counts_rejected(measured8530, monkeypatch):
    """SCENARIO-VERIFY-8530-CONTROLS: a new checksum cannot conceal independently measured drift."""
    directory, work, receipts = measured8530
    output = directory / "tampered.json"
    primitive = directory / "raw/measurement.json"
    for attack in ["reference", "history", "fixture", "controls"]:
        changed = deepcopy(work)
        if attack == "reference":
            changed["code_refs"][0]["sha256"] = "wrong"
        elif attack == "history":
            changed["history_reference"] = e.reference(directory / "raw/historical_accounting.json")
            monkeypatch.setattr(e, "historical_accounting", lambda: {})
        elif attack == "fixture":
            changed["fixture_rows"][0]["after"]["parameters"][0] += 1
        else:
            changed["control_rows"][0]["changed_coordinate_count"] = 1
        atomic_json(primitive, changed)
        value = e.build(changed, receipts, directory / "raw", output)
        atomic_json(output, value)
        assert not e.replay(output), attack
        monkeypatch.undo()
    atomic_json(primitive, work)
    value = e.build(work, receipts, directory / "raw", output)
    value["validation_receipts"] = deepcopy(receipts)
    value["validation_receipts"][0]["actual_exit"] = 73
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    value = e.build(work, receipts, directory / "raw", output)
    atomic_json(output, value)
    log = Path(receipts[0]["stdout_path"])
    original = log.read_bytes()
    log.write_bytes(original + b"changed log bytes\n")
    assert not e.replay(output)
    log.write_bytes(original)


def test_exact_node_union_and_incomplete_logs(tmp_path, monkeypatch):
    """REQ-VERIFY-8530: start-only, killed and nonterminal shards cannot qualify the union."""
    from carnot.reporting import v733_update_validation as v

    source = e.read(
        dict(path=str(e.ROOT / "results" / (e.old.NAME + ".json")), sha256=e.HISTORICAL_PIN)
    )
    original = e.read(source["primitive_reference"])
    ledger = json.loads((Path(original["raw"]) / "historical_collection.json").read_bytes())
    by_suite = {
        suite: [n for n in ledger["node_ids"] if n.split("::")[0] == suite]
        for suite in {n.split("::")[0] for n in ledger["node_ids"]}
    }
    monkeypatch.setattr(
        v.collection,
        "collect",
        lambda private, suite: (by_suite[suite], dict(name="unit_collection_fixture")),
    )
    plan = v.manifest(tmp_path)
    nodes = [n for p in plan for n in p.get("collected_node_ids", [])]
    assert set(nodes) == set(ledger["node_ids"]) and len(nodes) == len(set(nodes))
    assert all("--no-cov" in p["argv"] for p in plan if p.get("collected_node_ids"))
    log = tmp_path / "start_only.stdout"
    log.write_text("[runner] start\n")
    receipt = dict(
        stdout_path=str(log),
        passed=True,
        timed_out=False,
        collected_node_ids=["tests/python/x.py::test_x"],
    )
    assert not v.union_receipt(["tests/python/x.py::test_x"], [receipt])["passed"]
    log.write_text("tests/python/x.py::test_x PASSED [100%]\n")
    assert v.union_receipt(["tests/python/x.py::test_x"], [receipt])["passed"]
    assert not v.union_receipt(["tests/python/x.py::test_x"], [dict(receipt, timed_out=True)])[
        "passed"
    ]
    monkeypatch.setattr(v.collection, "collect", lambda private, suite: ([], dict(name="empty")))
    with pytest.raises(ValueError, match="original_node_union_changed"):
        v.manifest(tmp_path)


def test_real_private_validation_plan(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8530-CONTROLS: execute and archive a real disjoint private test child."""
    from carnot.reporting import v733_update_accounting_runner as runner
    from carnot.reporting import v733_update_validation as v

    node = "tests/python/test_v733_update_accounting_8530.py::test_coordinate_semantics"

    def fixture_plan(private):
        receipt = child(
            "collect_private",
            [
                sys.executable,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--collect-only",
                "-q",
                node,
            ],
            private,
            deadline=60,
        )
        atomic_json(private / "collection_archive.json", dict(node_ids=[node], receipts=[receipt]))
        atomic_json(private / "coverage.json", dict(scope="unit fixture for archival path"))
        return [
            dict(
                name="private_actual_pytest",
                argv=[
                    sys.executable,
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-vv",
                    node,
                ],
                expected=0,
                deadline=60,
                collected_node_ids=[node],
            )
        ]

    monkeypatch.setattr(v, "manifest", fixture_plan)
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(tmp_path / "absent", output, tmp_path / "private") == 0
    value = json.loads(output.read_bytes())
    assert value["historical_union_receipt"]["passed"]
    assert e.replay(output)


@pytest.mark.parametrize("failure", ["unsealed", "duplicate", "conflict"])
def test_fixture_detects_owned_causal_failures(tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-8530-ACCOUNTING: successful process status cannot hide causal damage."""
    from carnot.reporting import v733_update_fixture as f

    cfg = f.configuration()
    if failure == "unsealed":
        monkeypatch.setattr(f.adapter, "release", lambda *a: {})
    else:
        original_store = f.adapter.Store

        class BrokenStore(original_store):
            calls = 0

            def release(self, block, labels, hook=lambda phase: None):
                self.calls += 1
                if failure == "conflict" and self.calls == 3:
                    return "conflict_accepted"
                result = super().release(block, labels, hook)
                return result + "changed" if failure == "duplicate" and self.calls == 2 else result

        monkeypatch.setattr(f.adapter, "Store", BrokenStore)
    with pytest.raises(
        ValueError,
        match={
            "unsealed": "unsealed_release_accepted",
            "duplicate": "duplicate_release_changed_state",
            "conflict": "conflicting_release_accepted",
        }[failure],
    ):
        f.worker(cfg, tmp_path)


def test_resource_and_zero_methods_diagnostics(measured8530, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8530-CONTROLS: exact observed tools and method zero remain named gates."""
    directory, work, _ = measured8530
    original_read = e.read
    monkeypatch.setattr(
        e,
        "read",
        lambda ref: (
            dict(methods_ready_score=0, learning_inputs_ready_score=1)
            if ref["path"].endswith(e.methods.NAME + ".json")
            else original_read(ref)
        ),
    )
    monkeypatch.setattr(
        e, "historical_accounting", lambda: original_read(work["history_reference"])
    )
    resources = e.methods.base.legacy.resources
    monkeypatch.setattr(
        e.methods.base.legacy,
        "resources",
        lambda p: dict(resources(p), private_disk_backed_scratch=False),
    )
    value = e.measure(e.ROOT, tmp_path / "raw")
    assert any(
        r["field"] == "methods_ready_score" and r["observed"] == 0 for r in value["failures"]
    )
    assert any(
        r["field"] == "memory_disk_tools" and r["observed"]["private_disk_backed_scratch"] is False
        for r in value["failures"]
    )
