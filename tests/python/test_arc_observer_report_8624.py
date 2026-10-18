"""REQ-VERIFY-8624: qualify reachable mechanics and immutable reductions."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import arc_observer_8624 as e
from carnot.reporting.arc_observer_fixtures_8624 import qualify


@pytest.fixture(scope="module")
def historical_data():
    """SCENARIO-REPORT-8624-HISTORY: authenticate the actual closed panel once."""
    return e.historical(e.ROOT)


def cli(tmp_path, *args):
    """Run real standalone children, including their statements in coverage."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=180)


def test_actual_live_wrapper_preserves_all_actions_and_receipts():
    """SCENARIO-REPORT-8624-PARITY: three mechanics exercise every boundary case."""
    rows = qualify()
    assert len(rows) >= 27
    assert {r["mechanic"] for r in rows} == {"navigation", "toggle", "sequence"}
    assert all(r["passed"] for r in rows)
    assert all(r["model_calls"] == 0 and r["adapter_calls"] == 0 for r in rows)
    assert all(r["wrapper"] == "make_carnot_agent.choose_action" for r in rows)
    assert any(r["overlap_count"] for r in rows)
    assert all(r["equality"]["horizon_status"] for r in rows)


def test_history_reduction_preserves_missing_and_legacy():
    """SCENARIO-REPORT-8624-HISTORY: retained holes never become no progress."""
    events = [
        dict(
            event_id="r",
            arm="a",
            wrapper_action_index=1,
            applied=True,
            observation=dict(level=0),
            resolved_by_levelup=False,
        )
    ]
    steps = [dict(index=i, action=dict(kind=1), observation=dict(level=0)) for i in range(1, 121)]
    assert e.reduce_stream(steps, events)[0]["horizon_status"] == "no_progress_within_horizon"
    steps[30]["observation"] = None
    assert e.reduce_stream(steps, events)[0]["horizon_status"] == "censored"
    events[0]["wrapper_action_index"] = 122
    assert e.reduce_stream(steps, events)[0]["end_reason"] == "missing_application"


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--date", "bad"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", "/var/tmp/absent-8624.json"], 1),
    ],
)
def test_cli_rejection_paths(tmp_path, args, expected):
    """SCENARIO-VERIFY-8624-REPLAY: actual exits preserve failed evidence."""
    assert cli(tmp_path, *args).returncode == expected


def test_private_blocked_owned_failure_and_rehashed_tamper(tmp_path):
    """SCENARIO-VERIFY-8624-REPLAY: blocked and disqualified are distinct."""
    for failure in [False, True]:
        output = tmp_path / str(failure) / (e.NAME + ".json")
        args = [
            "--date",
            "20261017",
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
            "--private-control",
        ]
        if failure:
            args.append("--owned-failure")
        result = cli(tmp_path, *args)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_text())
        assert value["verdict_class"] == ("disqualified" if failure else "blocked")
        assert value["arc_observer_ready_score"] == 0
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        tamper = deepcopy(value)
        tamper["completed_count"] = 991
        tamper.pop("reproducibility_checksum")
        tamper["reproducibility_checksum"] = e.canonical_hash(tamper)
        output.write_text(json.dumps(tamper))
        assert cli(tmp_path, "--cold-replay", output).returncode == 1


def test_input_authentication_rejects_missing_and_changed_original(tmp_path):
    """SCENARIO-VERIFY-8624-REPLAY: envelope hashes cannot replace original pins."""
    with pytest.raises((ValueError, OSError)):
        e.authenticate(tmp_path)
    for name, pin in e.UPSTREAM.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    with pytest.raises(ValueError, match="original_hash"):
        e.authenticate(tmp_path)


def test_original_history_retains_resets_and_zero_resolution(historical_data):
    """SCENARIO-REPORT-8624-HISTORY: historical resets invalidate long horizons."""
    rows = historical_data["horizon_status_rows"]
    assert len(rows) == 24
    assert all(r["horizon_status"] == "censored" and r["end_reason"] == "reset" for r in rows)
    assert all(r["legacy_receipt"]["resolved_by_levelup"] is False for r in rows)
    assert all(r["legacy_receipt"]["actions_to_levelup"] is None for r in rows)
    assert len({r["game"] for r in rows}) == 4


@pytest.mark.parametrize("mutation", ["validation", "task"])
def test_original_authority_and_validation_refuse_changes(monkeypatch, mutation):
    """REQ-VERIFY-8624: valid envelope hashes do not bypass authority or checks."""
    if mutation == "validation":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: {"report": {"passed": False}})
    else:
        monkeypatch.setattr(e, "canonical_hash", lambda *a: "foreign_task")
    with pytest.raises(
        ValueError,
        match="original_" + ("validation" if mutation == "validation" else "authority_task"),
    ):
        e.authenticate(e.ROOT)


def test_build_replay_and_rehashed_primitive_rejections(tmp_path, historical_data, monkeypatch):
    """SCENARIO-VERIFY-8624-REPLAY: self-rehashed science still needs original operands."""
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file
    from carnot.reporting import arc_observer_fixtures_8624 as fixtures

    fake = [dict(game="fixture", passed=True, absolute_metric=1, equality=dict(actions=True))] * 27
    primitive = tmp_path / "primitives.json"
    atomic_json(primitive, dict(fixtures=fake, history=historical_data))
    work = dict(
        run_date="20261017",
        input_root=str(e.ROOT),
        failures=[],
        primitive_reference=dict(path=str(primitive), sha256=sha256_file(primitive)),
        preconditions_checked={},
        duration_s=0.01,
        phase_spans=[],
        source_artifact_hashes={},
        code_config_hashes={},
        raw_shard_hashes={},
        registry_precheck={},
    )
    work_path = tmp_path / "work.json"
    log = tmp_path / "log"
    log.write_text("real retained log")
    receipt = dict(
        name="owned_tests",
        passed=True,
        expected_exit=0,
        exit_code=0,
        duration_s=0.01,
        stdout_path=str(log),
        stderr_path=str(log),
        stdout_sha256=sha256_file(log),
        stderr_sha256=sha256_file(log),
    )
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(e, "historical", lambda root: historical_data)
    monkeypatch.setattr(fixtures, "qualify", lambda: fake)
    atomic_json(work_path, work)
    receipts = [dict(receipt, name=name) for name in e.REQUIRED_CHECKS]
    value = e.build(work_path, receipts, output)
    assert value["verdict_class"] == "null"
    assert e.replay(value)
    for field, altered, expected in [
        ("history", {"altered": True}, "historical_reduction_drift"),
        ("fixtures", [dict(passed=False)], "fixture_parity_drift"),
        (
            "fixtures",
            [dict(passed=True, absolute_metric=1, equality=dict(actions=True))],
            "fixture_reduction_drift",
        ),
    ]:
        atomic_json(primitive, {"fixtures": fake, "history": historical_data, field: altered})
        work["primitive_reference"]["sha256"] = sha256_file(primitive)
        atomic_json(work_path, work)
        changed = deepcopy(value)
        changed["work_reference"]["sha256"] = sha256_file(work_path)
        with pytest.raises(ValueError, match=expected):
            e.replay(changed)
    atomic_json(primitive, dict(fixtures=fake, history=historical_data))
    work["primitive_reference"]["sha256"] = sha256_file(primitive)
    atomic_json(work_path, work)
    value = e.build(work_path, receipts, output)
    log.write_text("changed log")
    with pytest.raises(ValueError, match="original_hash"):
        e.replay(value)


def test_current_authority_refuses_drift_and_preserves_gate_rows(tmp_path, monkeypatch):
    """REQ-VERIFY-8624: current registration differs from immutable old authority."""
    from carnot.reporting import arc_observer_runner_8624 as runner
    import yaml

    valid = runner.authority(e.ROOT)
    assert valid["passed"] and valid["task_sha256"] == e.TASK_PIN
    plan = yaml.safe_load((e.ROOT / "research-roadmap.yaml").read_text())
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(dict(plan, milestone="foreign")))
    design = tmp_path / "openspec/change-proposals/research-roadmap-vNEXT.md"
    design.parent.mkdir(parents=True)
    design.write_bytes(
        (e.ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md").read_bytes()
    )
    with pytest.raises(ValueError, match="current_design_authority"):
        runner.authority(tmp_path)


def test_validation_adapter_retains_hook_scope(tmp_path, monkeypatch):
    """REQ-VERIFY-8624: the original E2E orchestrator remains unchanged."""
    from carnot.reporting import arc_observer_runner_8624 as runner

    monkeypatch.setattr(runner.validation, "validate_checks", lambda *args: [])
    commands = []

    def child(name, argv, *args, **kwargs):
        commands.append(argv)
        return dict(name=name, passed=True)

    monkeypatch.setattr(runner, "child", child)
    assert all(r["passed"] for r in runner.validate_checks(tmp_path, tmp_path / "private"))
    scope = json.loads((tmp_path / "live_statement_scope.json").read_text())
    assert scope["executable_new_statements"] and scope["sha256"]
    assert len(commands) == 2


@pytest.mark.parametrize("bad_authority", [False, True])
def test_regular_controller_is_bounded_and_records_checks(tmp_path, monkeypatch, bad_authority):
    """SCENARIO-VERIFY-8624-REPLAY: exercise orchestration without nested controllers."""
    from carnot.reporting import arc_observer_runner_8624 as runner

    monkeypatch.setattr(runner, "qualify", lambda: [])
    monkeypatch.setattr(e, "historical", lambda root: {})
    if bad_authority:

        def broken(root):
            raise ValueError("altered design")

        monkeypatch.setattr(runner, "authority", broken)
    else:
        monkeypatch.setattr(
            runner,
            "authority",
            lambda root: dict(
                passed=False,
                gates=[
                    dict(
                        passed=False,
                        upstream="exp8615-contract-methods",
                        artifact_path="missing.json",
                        artifact_sha256=None,
                        artifact_field="methods_ready_score",
                        op="==",
                        expected=1,
                        actual=None,
                    )
                ],
            ),
        )
    log = tmp_path / "log"
    log.write_text("retained")
    receipt = dict(
        name="stub",
        passed=True,
        duration_s=0.01,
        stdout_path=str(log),
        stdout_sha256=e.sha256_file(log),
        expected_exit=0,
        exit_code=0,
    )
    monkeypatch.setattr(runner, "validate_checks", lambda *a: [receipt])
    monkeypatch.setattr(runner, "child", lambda *a, **k: receipt)
    seen = []
    monkeypatch.setattr(
        runner, "publish", lambda value, *a: seen.append(value) or dict(passed=True)
    )
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261017", "--output", str(output)]) == 0
    assert seen[0]["verdict_class"] == "blocked"
    assert len(seen[0]["validation_receipts"]) == 4
    assert seen[0]["gate_check_summary"]
    monkeypatch.setattr(runner.time, "monotonic_ns", iter([0, 3601 * 10**9]).__next__)
    with pytest.raises(TimeoutError, match="measurement_cap"):
        runner.main(["--date", "20261017", "--output", str(output)])


def test_private_existing_root_and_publication_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8624-REPLAY: unsafe private roots and bad final validators fail."""
    from carnot.reporting import arc_observer_runner_8624 as runner

    with pytest.raises(SystemExit):
        runner.main(["--private-control", "--root", str(tmp_path)])
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(runner, "child", lambda *a, **k: dict(passed=False))
    value = dict(
        experiment_id=8624,
        task_id=e.TASK,
        honest_verdict="complete_null_control",
        verdict_class="null",
        arc_observer_ready_score=0,
    )
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.publish(value, output, tmp_path / "raw")
    assert not output.exists()


def test_pinned_field_reader_refuses_ambiguous_or_missing_fields(tmp_path):
    """REQ-VERIFY-8624: narrow parsing retains an authenticated unique field identity."""
    path = tmp_path / "original.json"
    path.write_text('{\n  "other": 1\n}\n')
    with pytest.raises(ValueError, match="original_field_identity"):
        e.original_fields(path, e.sha256_file(path), ["redirects"])
    with pytest.raises(ValueError, match="original_hash"):
        e.checked(path, "foreign_hash")
    path.write_text('{\n  "redirects": [')
    with pytest.raises(json.JSONDecodeError):
        e.original_fields(path, e.sha256_file(path), ["redirects"])
