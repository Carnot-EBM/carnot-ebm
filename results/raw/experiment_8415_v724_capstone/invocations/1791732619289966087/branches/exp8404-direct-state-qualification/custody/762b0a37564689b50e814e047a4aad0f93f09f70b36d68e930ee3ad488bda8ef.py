"""REQ-REPORT-8404 / REQ-VERIFY-8404: authentic recovery grants no science credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting import direct_state_qualification_8404 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-VERIFY-8404: reuse this invocation's real crash evidence without extra samples."""
    if os.environ.get("CARNOT8404_WORK"):
        return json.loads(Path(os.environ["CARNOT8404_WORK"]).read_bytes())
    private = tmp_path_factory.mktemp("v724-state")
    with e.bindings():
        return e.measure(e.ROOT, private / "raw", private)


def test_genuine_directory_child(work, tmp_path):
    """SCENARIO-VERIFY-8404-DIRECTORY-CHILD: kill after pointer directory fsync in a real child."""
    frozen = Path(work["traces"][0]["path"])
    trace = json.loads(frozen.read_bytes())
    store = e.QualifiedStore(tmp_path / "state", trace)
    store.initialize()
    receipt = child(
        "directory",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--worker",
            str(frozen),
            "--state-directory",
            str(store.path),
            "--barrier",
            "directory_fsync",
        ],
        tmp_path / "logs",
        expected=-9,
        deadline=60,
    )
    assert receipt["passed"] and receipt["actual_exit"] == -9
    marker = json.loads((store.path / "kill.json").read_bytes())
    assert marker["directory_fsync"] is True
    assert marker["visible_state"] == e.s.fold(trace, trace["crash_event_index"] + 1)


def test_all_barriers_and_complete_states(work):
    """REQ-VERIFY-8404: all fifteen barrier pairs recover complete uninterrupted state."""
    assert len(work["rows"]) == 33
    assert all(row["passed"] for row in work["rows"])
    assert work["historical_consumer_receipt"]["classification"]["ready"]
    assert work["current_authority"]["activated"]
    assert all(row["actual_exit"] == -9 for row in work["rows"] if row["barrier"] != "none")
    value = e.build(work, [dict(passed=True)])
    assert value["direct_state_ready_score"] == 0  # No coverage receipt means no readiness.
    assert value["power_loss_certified"] is False
    assert value["directory_fsync_hook_executed"] is True
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )


def test_replay_and_rehashed_controls(work, tmp_path):
    """SCENARIO-REPORT-8404-PUBLICATION: primitive replay rejects a repaired aggregate."""
    candidate = tmp_path / (e.NAME + ".json")
    value = e.build(work, [dict(passed=True)])
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert not e.replay(tmp_path / "absent")
    assert not e.replay_value(dict(value, experiment_id=8390), tmp_path)
    bad = dict(value, direct_state_ready_score=1, reproducibility_checksum="wrong")
    atomic_json(candidate, bad)
    assert not e.replay(candidate)
    for field in ("completed_count", "directory_fsync_hook_executed"):
        bad = deepcopy(value)
        bad[field] = 99
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(candidate, bad)
        assert not e.replay(candidate)
    atomic_json(candidate, value)
    assert all(r["passed"] for r in e.controls(candidate, tmp_path / "controls"))
    assert e.build(work, [dict(passed=False)])["verdict_class"] == "disqualified"


def test_absent_and_changed_producer(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8404-CUSTODY: absence is blocked and never measured as zero."""
    monkeypatch.setenv("CARNOT8404_WORK", os.environ.get("CARNOT8404_WORK", ""))
    assert e.dependencies(tmp_path, tmp_path / "raw")["failures"]
    producer = tmp_path / e.UPSTREAM
    atomic_json(producer, {"historical_direct_consumers_ready_score": 0})
    assert e.dependencies(tmp_path, tmp_path / "raw")["failures"]
    with e.bindings():
        missing = e.measure(tmp_path, tmp_path / "missing-raw", tmp_path)
    value = e.build(missing, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["completed_count"] == 0 and value["censored_count"] == 33
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    assert value["direct_state_ready_score"] == 0


def test_plan_and_cli_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8404-VALIDATION: real commands freeze private coverage and historical roots."""
    monkeypatch.setenv("COVERAGE_RCFILE", os.environ.get("COVERAGE_RCFILE", ""))
    monkeypatch.setenv("COVERAGE_FILE", os.environ.get("COVERAGE_FILE", ""))
    plan = e.plan(tmp_path)
    assert any("--fail-under=100" in c["argv"] for c in plan)
    command = next(c for c in plan if c["name"] == "private_E2E020_direct_consumers")
    assert "carnot.testing.historical_roots_8402" in command["argv"]
    assert all(c["scope"] == "owned" for c in plan)
    for args, expected in [
        (["--deliberate-error"], 1),
        (["--date", "20261010"], 2),
        (["--worker", "missing"], 2),
        (["--probe"], 2),
    ]:
        assert child(
            "cli" + str(len(args)),
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            tmp_path / str(expected) / str(len(args)),
            expected=expected,
        )["passed"]


def test_fresh_corruption_controls(work, tmp_path):
    """REQ-VERIFY-8404: preserve genuine invalid-state child checks and original assertions."""
    assert all(r["passed"] for r in e.state_controls(work, tmp_path))


def test_blocked_family_does_not_measure(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8404-CUSTODY: a real gate failure prevents every crash arm."""
    failure = e.authority.failure(
        e.ROOT / e.UPSTREAM, "historical_direct_consumers_ready_score", 1, 0
    )
    monkeypatch.setattr(
        e,
        "dependencies",
        lambda *args: dict(
            failures=[failure], refs=[], historical_consumer_receipt={}, original_logs=[]
        ),
    )
    with e.bindings():
        missing = e.measure(e.ROOT, tmp_path / "raw", tmp_path)
    assert all(r["status"] == "unstarted" for r in missing["rows"])
    assert e.build(missing, [dict(passed=True)])["verdict_class"] == "blocked"
    monkeypatch.setenv("CARNOT8404_WORK", work["work_path"])


def test_changed_authority_and_family(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8404-CUSTODY: rehashed current authority cannot replace its sealed inputs."""
    changed = deepcopy(work)
    changed["current_authority"]["canonical_tasks_sha256"] = "changed"
    changed["work_path"] = str(tmp_path / "changed-work.json")
    atomic_json(Path(changed["work_path"]), changed)
    path = tmp_path / "candidate.json"
    atomic_json(path, e.build(changed, [dict(passed=True)]))
    assert not e.replay(path)
    monkeypatch.setattr(e.authority, "authenticate_family", lambda *args: {"ready": False})
    atomic_json(path, e.build(work, [dict(passed=True)]))
    assert not e.replay(path)


def test_zero_producer_score(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8404-CUSTODY: an observed zero is distinct from missing upstream bytes."""
    primary = tmp_path / e.UPSTREAM
    atomic_json(primary, {"historical_direct_consumers_ready_score": 0})
    monkeypatch.setattr(e.authority, "checked", lambda path, digest: path)
    result = e.dependencies(tmp_path, tmp_path / "raw")
    assert result["failures"][0]["observed"] == "historical_direct_consumers_ready_score"


def test_real_publication_entry_and_owned_error(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8404-PUBLICATION: actual runner seals real coverage and checked bytes."""
    import coverage
    from carnot.reporting import direct_state_runner_8404 as runner

    copied = deepcopy(work)
    copied["work_path"] = str(tmp_path / "work.json")
    copied["terminal_validation_sidecar_path"] = str(tmp_path / "terminal.json")
    copied["code_config_hashes"] = [
        e.reference(Path(r["path"])) for r in copied["code_config_hashes"]
    ]
    atomic_json(Path(copied["work_path"]), copied)
    monkeypatch.setattr(e, "measure", lambda *args: deepcopy(copied))
    monkeypatch.setattr(runner, "SCRATCH", tmp_path / "scratch")

    def private_plan(private):
        # This is a real statement trace. Its limited coverage cannot authorize
        # readiness; production uses the complete frozen unit/child command set.
        config = private / "coverage.ini"
        config.write_text(
            "[run]\ninclude=" + str(e.ROOT / e.OWNED[0]) + "\n[report]\nexclude_lines=\n"
        )
        trace = coverage.Coverage(config_file=str(config), data_file=str(private / ".coverage"))
        trace.start()
        try:
            e.build(copied, [dict(passed=True)])
        finally:
            trace.stop()
        trace.save()
        trace.json_report(outfile=str(private / "coverage.json"))
        shard = coverage.Coverage(
            config_file=str(config), data_file=str(private / ".coverage.unit")
        )
        shard.get_data().update(trace.get_data())
        shard.save()
        return [
            dict(
                name="real_cli_error",
                argv=[sys.executable, "-u", str(e.ROOT / e.CLI), "--deliberate-error"],
                deadline_s=30,
                scope="owned",
                expected=1,
            )
        ]

    # The runner's normal manifest expects exit0 for validation; use a harmless
    # real cold replay child instead of changing that contract for this fixture.
    def passing_plan(private):
        private_plan(private)
        return [
            dict(
                name="real_cli",
                argv=[sys.executable, "-c", "print('private CLI fixture')"],
                deadline_s=30,
                scope="owned",
            )
        ]

    monkeypatch.setattr(e, "plan", passing_plan)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert e.main(["--date", "20261011", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and value["direct_state_ready_score"] == 0
    assert value["owned_coverage"]["missing_lines"] > 0
    monkeypatch.setattr(e, "plan", lambda private: (_ for _ in ()).throw(ValueError("owned_error")))
    assert e.main(["--output", str(tmp_path / "failure.json")]) == 1


def test_genuine_coverage_report_and_tamper(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8404-VALIDATION: genuine bytes, not reported totals, authorize coverage."""
    import coverage
    import runpy

    # A private one-statement fixture tests the report reader. This fixture is
    # never counted as coverage of the production module or as its qualification.
    source = tmp_path / "covered_fixture.py"
    source.write_text("result = 1\n")
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\ninclude=" + str(source) + "\n[report]\nexclude_lines=\n")
    trace = coverage.Coverage(config_file=str(config), data_file=str(tmp_path / ".coverage"))
    trace.start()
    runpy.run_path(str(source))
    trace.stop()
    trace.save()
    path = tmp_path / "coverage.json"
    trace.json_report(outfile=str(path))
    report = json.loads(path.read_bytes())
    monkeypatch.setattr(e, "OWNED", list(report["files"]))
    genuine = dict(report["totals"], reference=e.reference(path))
    assert e.coverage_ready(genuine)
    assert not e.coverage_ready(dict(genuine, missing_lines=9))
    assert not e.coverage_ready(
        dict(genuine, reference={"path": str(tmp_path / "absent"), "sha256": "bad"})
    )


def test_successful_pinned_dependencies(tmp_path):
    """SCENARIO-REPORT-8404-CUSTODY: authenticate actual producer, family and original failure logs."""
    result = e.dependencies(e.ROOT, tmp_path / "raw")
    assert not result["failures"]
    receipt = result["historical_consumer_receipt"]
    assert receipt["primary"]["sha256"] == e.UPSTREAM_PIN
    assert receipt["classification"]["ready"] is True
    assert len(result["original_logs"]) == 2
    assert all(Path(ref["path"]).is_file() for ref in result["refs"])
