"""REQ-VERIFY-8649: invocation claims derive from sealed operands and real checks."""

from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from carnot.reporting import arc_reset_8649 as e
from carnot.reporting import arc_reset_runner_8649 as runner
from carnot.reporting import arc_reset_live_8649 as live
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def test_original_history_and_rehashed_rejection(tmp_path):
    """SCENARIO-VERIFY-8649-REPLAY: original public redirects retain their null."""
    source = e.authenticate(e.ROOT)
    assert source["censored_count"] == 16
    assert len(source["observer_action_parity_rows"]) == 8
    with pytest.raises(OSError):
        e.authenticate(tmp_path)
    path = tmp_path / e.UPSTREAM
    path.parent.mkdir(parents=True)
    atomic_json(path, dict(censored_count=0))
    with pytest.raises(ValueError, match="original_hash"):
        e.authenticate(tmp_path)


def test_build_owned_failure_and_rehashed_work(tmp_path):
    """SCENARIO-VERIFY-8649-REPLAY: external missing and owned failure differ."""
    work = dict(
        run_date="20261018",
        failures=[dict(check="missing", observed=None)],
        references=[],
        preconditions_checked={},
        source_artifact_hashes={},
        code_config_hashes={},
        historical_model_provenance={},
        registry_precheck={},
        duration_s=0.01,
        phase_spans=[],
        history=None,
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
    receipt = dict(name="control", passed=True, duration_s=0.01)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(path, [receipt], output)
    assert value["verdict_class"] == "blocked"
    assert e.replay(value)
    changed = deepcopy(value)
    changed["reset_provenance_ready_score"] = 1
    changed["reproducibility_checksum"] = e.canonical_hash(changed)
    assert not e.replay(changed)
    assert e.build(path, [dict(receipt, passed=False)], output)["verdict_class"] == "disqualified"
    work["failures"] = []
    atomic_json(path, work)
    changed["work_reference"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="measurement_seal"):
        e.replay(changed)


def test_real_private_cli_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8649-REPLAY: actual children publish, read and reject tampering."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(e.ROOT / e.CLI), *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert run("--date", "invalid").returncode == 2
    assert run("--cold-replay", tmp_path / "absent.json").returncode == 1
    for owned in [False, True]:
        output = tmp_path / str(owned) / (e.NAME + ".json")
        argv = [
            "--date",
            "20261018",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
            "--private-control",
        ]
        if owned:
            argv.append("--owned-failure")
        result = run(*argv)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if owned else "blocked")
        assert run("--cold-replay", output).returncode == 0
        value["completed_count"] = 999
        value["reproducibility_checksum"] = e.canonical_hash(value)
        atomic_json(output, value)
        assert run("--cold-replay", output).returncode == 1
    assert run("--private-control", "--root", tmp_path).returncode == 2


def test_positive_build_and_controller_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8649: execute orchestration without recursively running validation."""
    from carnot.reporting import arc_reset_runner_8649 as runner

    pair = dict(mechanic="navigation", passed=True)
    arm = dict(
        actions=[],
        rng=[],
        torch_rng=[],
        termination=dict(error=None),
        legacy={},
        observations=[],
        observer_rows=[],
        sdk_calls=[],
        recorder_errors=[],
        llm_calls=0,
        numerical_head_calls=1,
        reset_provenance=dict(
            reset_request_rows=[dict(action_index=1)],
            requester_ownership_rows=[dict(status="unique")],
            terminal_reason_rows=[],
            missing_cause_rows=[],
            issues=[],
        ),
    )
    refs = []
    for name in ["navigation", "toggle", "sequence"]:
        path = tmp_path / (name + ".json")
        atomic_json(path, dict(pair, mechanic=name, off=arm, on=arm))
        refs.append(dict(path=str(path), sha256=sha256_file(path)))
    history = e.checked(e.ROOT / e.UPSTREAM, e.UPSTREAM_PIN)
    work = dict(
        run_date="20261018",
        references=refs,
        history=history,
        failures=[],
        preconditions_checked={},
        source_artifact_hashes={},
        code_config_hashes={},
        historical_model_provenance={},
        registry_precheck={},
        duration_s=0.01,
        phase_spans=[],
        measurement_receipts=[],
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
    output = tmp_path / (e.NAME + ".json")
    value = e.build(
        path, [dict(name=n, passed=True, duration_s=0.01) for n in e.REQUIRED_CHECKS], output
    )
    assert value["reset_provenance_ready_score"] == 1
    assert e.replay(value)
    monkeypatch.setattr(runner, "measure", lambda *args: work)
    monkeypatch.setattr(
        runner.inherited,
        "validate_checks",
        lambda *args: [dict(name="unit", passed=True, duration_s=0)],
    )
    monkeypatch.setattr(runner, "publish", lambda *args: dict(passed=True))
    monkeypatch.setattr(runner, "live_checks", lambda *args: [])
    assert runner.main(["--output", str(output)]) == 0
    from carnot.reporting import arc_reset_live_8649 as live

    monkeypatch.setattr(live, "fixture_pair", lambda *args: dict(passed=True))
    unit = tmp_path / "unit.json"
    atomic_json(unit, dict(directory=str(tmp_path), mechanic="navigation"))
    assert runner.main(["--fixture", str(unit)]) == 0


def test_measure_current_closure_and_rejected_publication(tmp_path, monkeypatch):
    """REQ-VERIFY-8649: snapshot current sources before validation and retain rejection."""
    from carnot.reporting import arc_reset_runner_8649 as runner

    raw = tmp_path / "raw"
    raw.mkdir()

    def child(name, argv, logs, **kwargs):
        if name.startswith("fixture_"):
            unit = json.loads(Path(argv[-1]).read_bytes())
            atomic_json(Path(unit["directory"]) / "pair.json", dict(mechanic=unit["mechanic"]))
        return dict(name=name, passed=True, duration_s=0.001)

    monkeypatch.setattr(runner, "child", child)
    work = runner.measure(e.ROOT, raw, tmp_path / "scratch", "20261018")
    assert len(work["references"]) == 3
    assert not work["failures"]
    assert all(Path(p).is_file() for p in work["code_config_hashes"])
    monkeypatch.setattr(runner, "child", lambda *args, **kwargs: dict(passed=False))
    value = dict(
        experiment_id=8649,
        task_id=e.TASK,
        verdict_class="null",
        honest_verdict="complete_null_control",
        reset_provenance_ready_score=0,
    )
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.publish(value, tmp_path / "reject" / (e.NAME + ".json"), raw)


def test_public_projection_hashes_legacy_without_loading_it(tmp_path):
    """REQ-VERIFY-8649: a projection authenticates exact bytes and avoids pixel custody copies."""
    path = tmp_path / "stream.json"
    atomic_json(path, dict(unit={}, legacy=dict(x=[1, 2]), actions=[1]))
    value = e.public_episode(dict(path=str(path), sha256=sha256_file(path)))
    assert value["actions"] == [1]
    assert isinstance(value["legacy"], str) and len(value["legacy"]) == 64
    original_hash = sha256_file(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="original_hash"):
        e.public_episode(dict(path=str(path), sha256=original_hash))


def test_live_validation_scope_uses_original_diff(tmp_path, monkeypatch):
    """REQ-VERIFY-8649: only executable statements added to the historical agent are owned."""
    from carnot.reporting import arc_reset_runner_8649 as runner

    monkeypatch.setattr(runner, "child", lambda *args, **kwargs: dict(passed=True))
    receipts = runner.live_checks(tmp_path)
    assert len(receipts) == 2 and all(row["passed"] for row in receipts)
    scope = json.loads((tmp_path / "live_statement_scope.json").read_bytes())
    assert len(scope["statements"]) >= 15
    assert scope["original_sha256"] != scope["current_sha256"]


def test_missing_checkpoint_remains_blocked_not_a_snapshot_crash(tmp_path, monkeypatch):
    """REQ-VERIFY-8649: missing CPU operands remain explicit external gate failures."""
    exists, is_file = Path.exists, Path.is_file
    checkpoint = e.ROOT / "results/experiment_4629_live_frame_change_cnn.pt"
    monkeypatch.setattr(Path, "exists", lambda p: False if p == checkpoint else exists(p))
    monkeypatch.setattr(Path, "is_file", lambda p: False if p == checkpoint else is_file(p))
    work = runner.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch", "20261018")
    gate = next(row for row in work["failures"] if row["check"] == "numerical_checkpoint")
    assert gate["observed"] is None and gate["expected"].startswith("sha256:")
    assert not work["references"]
