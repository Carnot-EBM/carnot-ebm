"""REQ-VERIFY-8663: immutable operands determine readiness and replay."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_reset_8663 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def private_work(tmp_path):
    """Missing external inputs create a complete blocked engineering artifact."""
    work = dict(
        run_date="20261019",
        failures=[dict(check="missing", observed=None)],
        references=[],
        reproduction=None,
        historical=None,
        preconditions_checked={},
        source_artifact_hashes={},
        code_config_hashes={},
        raw_file_hashes={},
        registry_precheck={},
        historical_model_provenance={},
        target_model=None,
        duration_s=0.01,
        phase_spans=[],
        measurement_receipts=[],
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
    return path


def test_blocked_failed_owned_and_rehashed_tamper(tmp_path):
    """SCENARIO-VERIFY-8663-REPLAY: absent external evidence never means zero."""
    work = private_work(tmp_path)
    output = tmp_path / "results" / (e.NAME + ".json")
    receipts = [dict(name="private_control", passed=True, duration_s=0.01)]
    value = e.build(work, receipts, output)
    assert value["verdict_class"] == "blocked"
    assert value["reset_provenance_ready_score"] == 0
    assert e.replay(value)
    changed = deepcopy(value)
    changed["reset_provenance_ready_score"] = 1
    assert not e.replay(changed)
    assert (
        e.build(work, [dict(name="bad", passed=False, duration_s=0.1)], output)["verdict_class"]
        == "disqualified"
    )
    data = json.loads(work.read_bytes())
    data["failures"] = []
    atomic_json(work, data)
    changed["work_reference"]["sha256"] = sha256_file(work)
    with pytest.raises(ValueError, match="measurement_seal"):
        e.replay(changed)


def test_actual_private_cli_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8663-REPLAY: direct CLI uses private results only."""
    cli = e.ROOT / e.CLI
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(cli)]
    run = subprocess.run(
        argv
        + [
            "--date",
            "20261019",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert (
        subprocess.run(
            argv + ["--cold-replay", str(output)], capture_output=True, timeout=60
        ).returncode
        == 0
    )
    value["completed_count"] = 12
    atomic_json(output, value)
    assert (
        subprocess.run(
            argv + ["--cold-replay", str(output)], capture_output=True, timeout=60
        ).returncode
        == 1
    )
    assert (
        subprocess.run(argv + ["--date", "invalid"], capture_output=True, timeout=60).returncode
        == 2
    )
    assert (
        subprocess.run(
            argv + ["--cold-replay", str(tmp_path / "missing")], capture_output=True, timeout=60
        ).returncode
        == 1
    )


def test_history_and_exact_failure_reproduction(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8663-PAYLOAD: independently pinned originals keep authority."""
    from carnot.reporting.arc_reset_witness_8663 import authenticate, reproduce

    history = authenticate(e.ROOT)
    assert history["missing_causes"] == 236
    assert history["request_rows"] == 688
    assert history["censored_redirects"] == 16
    assert history["action_equal_pairs"] == 8
    result = reproduce(history, tmp_path)
    assert result["passed"]
    assert {row["path"] for row in result["exception_reproduction_rows"]} == {
        "public_hash",
        "begin",
        "sdk_call",
        "make_carnot_agent",
    }
    assert all(row["error"] == "TypeError" for row in result["exception_reproduction_rows"])
    assert result["real_frame_fixture_reference"]["sha256"]
    with pytest.raises(OSError):
        authenticate(tmp_path)
    path = private_work(tmp_path)
    work = json.loads(path.read_bytes())
    work["historical"] = history
    atomic_json(path, work)
    atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
    value = e.build(
        path,
        [dict(name="control", passed=True, duration_s=0.01)],
        tmp_path / "results" / (e.NAME + ".json"),
    )
    assert e.replay(value)
    from carnot.reporting import arc_reset_witness_8663 as witness

    monkeypatch.setattr(witness, "authenticate", lambda _: {})
    with pytest.raises(ValueError, match="historical_drift"):
        e.replay(value)


def test_fresh_process_reproduction_without_prior_recorder_import(tmp_path):
    """SCENARIO-REPORT-8663-PAYLOAD: cold startup cannot rely on test import order."""
    code = "from pathlib import Path; from carnot.reporting.arc_reset_witness_8663 import authenticate,reproduce; from carnot.reporting.arc_reset_8663 import ROOT; import sys; assert reproduce(authenticate(ROOT),Path(sys.argv[1]))['passed']"
    result = subprocess.run(
        [sys.executable, "-u", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def synthetic_episode(arm):
    """Provide reducer operands only; these fixtures are never live witness evidence."""
    row = dict(
        action_index=1,
        reset_request_id="1:1",
        post_observation_sha256="post",
        sdk_status=dict(error=None),
    )
    return dict(
        unit=dict(reset_provenance=arm),
        actions=[dict(kind="RESET")],
        observations=[dict(level=0)],
        rng=[1],
        torch_rng=[2],
        termination=dict(error=None),
        legacy={},
        observer_rows=[],
        sdk_calls=["reset"],
        reset_provenance=dict(
            reset_request_rows=[row] if arm == "on" else [],
            issues=[],
            missing_cause_rows=[],
            callback_sequence_rows=[],
        ),
        reset_observations=[dict(action_index=1, observation_sha256="post")],
        recorder_errors=[],
        model_tripwire_attempts=[],
        sdk_identity=dict(official=True),
        llm_calls=0,
        numerical_head_calls=1,
    )


def test_complete_witness_reduction_and_each_failed_join(tmp_path):
    """SCENARIO-REPORT-8663-WITNESS: a complete join differs from absent evidence."""
    episodes = [synthetic_episode(arm) for arm in ("off", "on")]
    assert e.reduce(episodes)["witnessed"]
    for field, value in [
        ("action_index", 2),
        ("reset_request_id", "2:1"),
        ("post_observation_sha256", "bad"),
        ("sdk_status", None),
        ("sdk_status", dict(error="TimeoutError")),
    ]:
        changed = deepcopy(episodes)
        changed[1]["reset_provenance"]["reset_request_rows"][0][field] = value
        assert not e.reduce(changed)["witnessed"]
    work_path = private_work(tmp_path)
    work = json.loads(work_path.read_bytes())
    work.update(failures=[], reproduction=dict(passed=True), references=[])
    for index, ep in enumerate(episodes):
        path = tmp_path / f"stream{index}.json"
        atomic_json(path, ep)
        work["references"].append(dict(path=str(path), sha256=sha256_file(path)))
    atomic_json(work_path, work)
    atomic_json(
        work_path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(work_path))
    )
    receipts = [dict(name=name, passed=True, duration_s=0.01) for name in e.REQUIRED_CHECKS]
    result = e.build(work_path, receipts, tmp_path / "results" / (e.NAME + ".json"))
    assert result["reset_provenance_ready_score"] == 1
    assert result["verdict_class"] == "null"
    assert e.replay(result)


def test_current_validation_child_roles_and_raw_custody(tmp_path, monkeypatch):
    """REQ-VERIFY-8663: each bounded child retains its uncombined data and JUnit."""
    from carnot.reporting import arc_reset_runner_8663 as runner

    def fake_child(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text(
            "tests/python/test_a.py::test_one\ntests/python/test_a.py::test_two\n"
            if name == "collect_once"
            else ""
        )
        for arg in argv:
            if arg.startswith("--rcfile="):
                config = Path(arg.split("=", 1)[1])
                for line in config.read_text().splitlines():
                    if line.startswith("data_file="):
                        Path(line.split("=", 1)[1] + ".raw").write_bytes(
                            b"private simulated coverage"
                        )
        return dict(name=name, passed=True, duration_s=0.01, stdout_path=str(out))

    monkeypatch.setattr(runner, "child", fake_child)
    receipts = runner.validate_checks(tmp_path / "raw", tmp_path / "private")
    assert e.REQUIRED_CHECKS <= {row["name"] for row in receipts}
    assert all(row["passed"] for row in receipts)
    assert list((tmp_path / "raw/validation/raw_coverage").rglob(".coverage.raw"))


@pytest.mark.parametrize(
    "mode", ["complete", "unreproduced", "missing_head", "bad_history", "timeout"]
)
def test_measurement_boundaries_and_external_failure(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8663: reproduction must precede the bounded live children."""
    from carnot.reporting import arc_reset_runner_8663 as runner

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(e, "ROOT", root)
    for path in e.OWNED + [
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_outcome_observer.py",
        "python/carnot/agentic/arc_live_panel_runtime_8384.py",
        "python/carnot/reporting/arc_reset_live_8649.py",
        "python/carnot/reporting/arc_observer_live_8637.py",
        "tests/python/test_arc_reset_provenance_8649.py",
        *runner.validation.HISTORICAL,
        "ops/arc_solve_registry.yaml",
        *e.PINS,
        runner.HEAD,
    ]:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("games: []" if path.endswith("yaml") else "{}")
    pin = sha256_file(root / runner.HEAD)
    monkeypatch.setattr(runner, "HEAD_PIN", pin)
    if mode == "missing_head":
        (root / runner.HEAD).unlink()
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(
        runner.validation,
        "preconditions",
        lambda *args: ([], {}, {str(root / p): sha256_file(root / p) for p in e.OWNED}),
    )

    def history(_):
        if mode == "bad_history":
            raise ValueError("bad_original")
        return {}

    monkeypatch.setattr(runner.witness, "authenticate", history)
    monkeypatch.setattr(
        runner.witness,
        "reproduce",
        lambda *a: dict(passed=mode != "unreproduced", exception_reproduction_rows=[]),
    )

    def fake_child(name, argv, logs, **kwargs):
        unit = json.loads(Path(argv[-1]).read_bytes())
        atomic_json(
            Path(unit["directory"]) / "witness.json", synthetic_episode(unit["reset_provenance"])
        )
        return dict(name=name, passed=True, duration_s=0.01)

    monkeypatch.setattr(runner, "child", fake_child)
    if mode == "timeout":
        clock = iter([0, 3460 * 10**9])
        monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(clock))
        with pytest.raises(TimeoutError, match="measurement_cap"):
            runner.measure(root, raw, tmp_path / "private", "20261019")
        return
    work = runner.measure(root, raw, tmp_path / "private", "20261019")
    assert len(work["references"]) == (2 if mode == "complete" else 0)
    assert bool(work["failures"]) == (mode != "complete")


def test_episode_public_dispatch_and_main_branches(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8663-WITNESS: the private CLI dispatches the real episode function."""
    from carnot.reporting import arc_reset_runner_8663 as runner
    from carnot.reporting import arc_reset_live_8649 as live
    from carnot.reporting import arc_supervisor_live_panel_8384 as sdk_module
    from carnot.reporting import arc_reset_witness_8663 as witness
    from types import SimpleNamespace

    class SDK:
        pass

    monkeypatch.setattr(sdk_module, "sdk", SDK)

    def collect(unit, sdk, raw):
        result = synthetic_episode("on")
        (raw / "steps.jsonl").write_text(
            json.dumps(dict(index=1, action=dict(name="RESET"), observation=dict(pixels=[[[1]]])))
            + "\n"
        )
        return result

    monkeypatch.setattr(live, "collect", collect)
    path = tmp_path / "unit.json"
    atomic_json(path, dict(directory=str(tmp_path)))
    assert runner.main(["--episode", str(path)]) == 0
    assert json.loads((tmp_path / "witness.json").read_bytes())["reset_observations"]
    with pytest.raises(SystemExit) as error:
        runner.main(["--private-control", "--root", str(tmp_path)])
    assert error.value.code == 2
    monkeypatch.setattr(runner, "run", lambda args: 0)
    assert runner.main([]) == 0


def test_run_current_validation_path_without_recursive_controller(tmp_path, monkeypatch):
    """REQ-VERIFY-8663: current validation returns through the production run path."""
    from types import SimpleNamespace
    from carnot.reporting import arc_reset_runner_8663 as runner

    source = private_work(tmp_path)
    work = json.loads(source.read_bytes())
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(
        runner, "validate_checks", lambda *a: [dict(name="control", passed=True, duration_s=0.1)]
    )
    monkeypatch.setattr(runner, "publish", lambda *a: dict(passed=True))
    args = SimpleNamespace(
        root=tmp_path / "absent",
        output=tmp_path / "results" / (e.NAME + ".json"),
        date="20261019",
        private_control=False,
    )
    assert runner.run(args) == 0
