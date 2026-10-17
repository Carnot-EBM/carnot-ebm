"""REQ-REPORT-8599 and REQ-VERIFY-8599 bind support to live receipts."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import arc_outcome_identifiability_8599 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def cli(tmp_path, *args):
    """Real children expose import failures and preserve command exit semantics."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=300)


def test_windows_do_not_turn_censoring_into_failure():
    """SCENARIO-REPORT-8599-SUPPORT: later redirects compete with attribution."""
    steps = [
        dict(
            index=i,
            action_sha256=str(i),
            observation_sha256=str(i),
            observation=dict(level=0),
            progress=False,
        )
        for i in range(1, 7)
    ]
    events = [
        dict(
            event_id=str(i),
            wrapper_action_index=i,
            resolved_by_levelup=False,
            selected=True,
            applied=True,
            effective_action_policy_change=i == 2,
            before_state=dict(induced=True),
            after_state=dict(induced=i == 2),
        )
        for i in [2, 4]
    ]
    rows = e.windows(steps, events)
    assert rows[0]["observable_post_intervention_actions"] == 5
    assert rows[0]["uncontested_observable_actions"] == 2
    assert rows[0]["competing_redirect_ids"] == ["4"]
    assert rows[0]["outcome_observed"] and rows[0]["right_censored"]
    assert rows[0]["changed_action"] is None
    assert rows[1]["induction_latch_changed"]
    steps[4]["progress"] = True
    events[0]["resolved_by_levelup"] = True
    assert e.windows(steps, events)[0]["observable_post_intervention_actions"] == 4


def test_transfer_requires_each_resolved_shared_cell():
    """REQ-REPORT-8599: repeated seeds do not supply independent games."""
    cells = [
        dict(game=g, mode=m, arm=a, selected_count=2, resolved_count=0)
        for g in e.GAMES
        for m in e.MODES
        for a in e.ARMS
    ]
    folds = e.transfer(cells)
    assert len(folds) == 16 and not any(r["eligible"] for r in folds)
    assert all(r["overlapping_training_game_count"] == 3 for r in folds)
    for r in cells:
        r["resolved_count"] = 5
    assert all(r["eligible"] for r in e.transfer(cells))
    cells[0]["selected_count"] = 0
    assert any(r["overlapping_training_game_count"] == 2 for r in e.transfer(cells))


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """REQ-REPORT-8599: use actual authenticated receipts once, without game execution."""
    raw = tmp_path_factory.mktemp("outcome-primitives")
    result = cli(raw, "--audit-only", raw / "audit.json")
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((raw / "audit.json").read_bytes())


def test_real_panel_matches_independent_counts(measured):
    """REQ-REPORT-8599: preserve all8192 actions and original event IDs."""
    assert len(measured["action_receipt_rows"]) == 8192
    assert len(measured["censoring_rows"]) == 24
    assert sum(r["resolved_count"] for r in measured["intervention_support_rows"]) == 0
    assert sum(r["actual_disabled_attempts"] for r in measured["rows"]) == 24
    assert sum(r["induction_latch_changes"] for r in measured["rows"]) == 8
    assert all(r["peak_level"] == 0 for r in measured["rows"])
    assert len(measured["per_game_results"]) == 4
    assert len(measured["paired_results"]) == 8
    assert all(not r["eligible"] for r in measured["leave_one_game_out_support"])


def test_authentication_rejects_hash_and_readiness(tmp_path, monkeypatch):
    """REQ-VERIFY-8599: independent pins reject changed outer envelopes."""
    with pytest.raises((ValueError, OSError)):
        e.authenticate(tmp_path)
    monkeypatch.setattr(e, "PANEL_PIN", "sha256:bad")
    with pytest.raises(ValueError, match="sealed_hash"):
        e.authenticate(e.ROOT)


@pytest.mark.parametrize("owned_failure", [False, True])
def test_private_missing_and_owned_failure_cli(tmp_path, owned_failure):
    """SCENARIO-VERIFY-8599-REPLAY: actual publication preserves blocked vs disqualified."""
    output = tmp_path / (e.NAME + ".json")
    args = ["--root", tmp_path / "absent", "--output", output, "--private-control"]
    if owned_failure:
        args.append("--owned-failure")
    result = cli(tmp_path, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("disqualified" if owned_failure else "blocked")
    assert value["completed_count"] == 0 and value["intended_count"] == 16
    assert value["arc_outcome_audit_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["completed_count"] = 999
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--deliberate-error").returncode == 7
    assert cli(tmp_path, "--date", "bad").returncode == 2


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("readiness", "upstream_readiness"),
        ("binding", "upstream_publication_binding"),
        ("code", "original_code_hash"),
        ("authority_hash", "original_authority_hash"),
        ("authority_task", "original_authority_task"),
        ("panel", "fixed_live_panel"),
    ],
)
def test_original_authority_rejections(monkeypatch, mutation, expected):
    """REQ-VERIFY-8599: valid outer pins cannot excuse failed authority fields."""
    checked, hashed, canonical = e.checked, e.sha256_file, e.canonical_hash
    primary = checked(e.ROOT / e.PANEL_PATH, e.PANEL_PIN)
    work_path = Path(primary["work_reference"]["path"])
    authority_path = Path(
        checked(work_path, primary["work_reference"]["sha256"])["activation_reference"][
            "snapshot_path"
        ]
    )

    def altered(path, digest):
        value = checked(path, digest)
        if path == e.ROOT / e.PANEL_PATH and mutation == "readiness":
            value["arc_panel_rows_ready_score"] = 0
        if path == work_path and mutation == "panel":
            value["references"] = []
        return value

    monkeypatch.setattr(e, "checked", altered)
    if mutation == "binding":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
    if mutation in ["code", "authority_hash"]:
        code = Path(next(iter(primary["code_config_hashes"])))
        monkeypatch.setattr(
            e,
            "sha256_file",
            lambda p: (
                "sha256:wrong"
                if p == (code if mutation == "code" else authority_path)
                else hashed(p)
            ),
        )
    if mutation == "authority_task":
        monkeypatch.setattr(e, "canonical_hash", lambda _: "sha256:wrong")
    with pytest.raises(ValueError, match=expected):
        e.authenticate(e.ROOT)
    assert callable(canonical)


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("deadline", "measurement_deadline"),
        ("adapter", "live_adapter_withheld"),
        ("row", "original_row_mismatch"),
        ("count", "independent_action_progress_count"),
        ("resolution", "resolution_mismatch"),
        ("pair", "original_pair_mismatch"),
    ],
)
def test_primitive_rejections(tmp_path, monkeypatch, fault, expected):
    """SCENARIO-VERIFY-8599-REPLAY: owned inconsistencies fail before a result exists."""
    row = dict(episode_id="episode", game=e.GAMES[0], seed=11, mode=e.MODES[0], peak_level=0)
    directory = tmp_path / "episode"
    directory.mkdir()
    steps = [
        dict(
            index=i,
            action_sha256=str(i),
            observation_sha256=str(i),
            observation=dict(level=0),
            progress=False,
            action={},
            choose_action_s=0,
            sdk_transition_s=0,
        )
        for i in range(1, 513)
    ]
    if fault == "count":
        steps.pop()
    (directory / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps))
    atomic_json(
        directory / "episode.json",
        dict(adapter_withheld=fault != "adapter", actual_wrapper_path="qualified"),
    )
    ref = dict(directory=str(directory), steps_sha256="sealed")
    original = dict(row, seed=22) if fault == "row" else row
    monkeypatch.setattr(
        e,
        "authenticate",
        lambda _: (dict(rows=[original], paired_results=[1]), dict(references=[ref])),
    )
    events = []
    if fault == "resolution":
        events = [
            dict(
                event_id="event",
                wrapper_action_index=2,
                resolved_by_levelup=True,
                selected=True,
                applied=True,
                effective_action_policy_change=False,
                before_state=dict(induced=True),
                after_state=dict(induced=False),
            )
        ]
    monkeypatch.setattr(e.panel.panel, "reduce_episode", lambda _: (row, events, []))
    monkeypatch.setattr(e.panel.panel, "pair", lambda _: dict(paired_results=[]))
    if fault == "deadline":
        ticks = iter([0, 3601])
        monkeypatch.setattr(e.time, "monotonic", lambda: next(ticks))
    with pytest.raises((ValueError, TimeoutError), match=expected):
        e.audit(tmp_path, tmp_path / "raw")


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-VERIFY-8599: scoped checks keep the original E2E authority intact."""
    from carnot.reporting import arc_outcome_runner_8599 as runner

    def validate(*_):
        assert runner.validation.e is e
        assert runner.validation.TESTS == e.TESTS
        return [dict(name="actual_scope", passed=True)]

    monkeypatch.setattr(runner.validation, "validate_checks", validate)
    assert runner.validate_checks(tmp_path, tmp_path)[0]["passed"]


def test_regular_controller_and_current_date(tmp_path, monkeypatch):
    """REQ-VERIFY-8599: a current date and frozen authority survive normal dispatch."""
    from carnot.reporting import arc_outcome_runner_8599 as runner

    root = tmp_path / "root"
    root.mkdir()
    (root / "research-roadmap.yaml").write_text("authority")
    for name in [e.PANEL_PATH, e.PUBLICATION_PATH]:
        atomic_json(root / name, {})
    failure = dict(
        check="test_missing_input",
        upstream=e.TASK,
        path="missing",
        hash=None,
        field="present",
        operator="==",
        expected=True,
        observed=False,
        passed=False,
    )
    monkeypatch.setattr(runner.validation, "preconditions", lambda *_: ([failure], {}, {}))
    monkeypatch.setattr(e, "audit", lambda *_: {})
    monkeypatch.setattr(runner, "validate_checks", lambda *_: [])
    monkeypatch.setattr(runner, "publish", lambda *_: dict(passed=False))
    assert (
        runner.main(
            [
                "--root",
                str(root),
                "--output",
                str(tmp_path / (e.NAME + ".json")),
                "--date",
                "20261018",
            ]
        )
        == 1
    )
    assert list(tmp_path.glob("raw/*/invocations/*/authority.yaml"))
    with pytest.raises(SystemExit):
        runner.main(["--private-control", "--root", str(root)])


def test_replay_rejects_rehashed_work_and_logs(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8599-REPLAY: recomputation defeats self-consistent rehashing."""
    operand = tmp_path / "operand"
    operand.write_text("immutable")
    raw = tmp_path / "work.json"
    work = dict(
        data=dict(rows=[]),
        source_artifact_hashes={},
        code_config_hashes={},
        raw_shard_hashes={},
        input_root=str(tmp_path),
        replay_scratch=str(tmp_path),
    )
    atomic_json(raw, work)
    value = dict(work_reference=dict(path=str(raw), sha256=e.sha256_file(raw)))
    monkeypatch.setattr(e, "audit", lambda *_: dict(rows=["tampered"]))
    with pytest.raises(ValueError, match="recomputed_primitive_mismatch"):
        e.replay(value)
    work["data"] = {}
    work["raw_shard_hashes"][str(operand)] = "sha256:wrong"
    atomic_json(raw, work)
    value["work_reference"]["sha256"] = e.sha256_file(raw)
    with pytest.raises(ValueError, match="replay_operand_hash"):
        e.replay(value)
    work["raw_shard_hashes"] = {}
    atomic_json(raw, work)
    value["work_reference"]["sha256"] = e.sha256_file(raw)
    value["validation_receipts"] = [dict(stdout_path=str(operand), stdout_sha256="sha256:wrong")]
    with pytest.raises(ValueError, match="validation_log_hash"):
        e.replay(value)


def test_publish_keeps_actual_validator_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8599: a warning or nonzero validator exit cannot become a pass."""
    from carnot.reporting import arc_outcome_runner_8599 as runner

    output = tmp_path / (e.NAME + ".json")
    value = dict(
        experiment_id=8599,
        task_id=e.TASK,
        honest_verdict="complete_blocked_control",
        verdict_class="blocked",
        arc_outcome_audit_ready_score=0,
    )
    monkeypatch.setattr(runner, "child", lambda *_, **__: dict(passed=False, exit_code=2))
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.publish(value, output, tmp_path / "private")
    assert not output.exists()
    assert (
        json.loads((tmp_path / "private/publication_attempt.json").read_bytes())["checks"][0][
            "exit_code"
        ]
        == 2
    )


def test_incomplete_validation_cannot_enable_readiness(tmp_path):
    """REQ-VERIFY-8599: a passing subset cannot qualify a measured diagnostic."""
    work = dict(
        data=dict(rows=[]),
        failures=[],
        run_date="20261017",
        preconditions_checked={},
        duration_s=1,
        phase_spans=[],
        source_artifact_hashes={},
        code_config_hashes={},
        raw_shard_hashes={},
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    value = e.build(path, [dict(name="owned_tests", passed=True)], tmp_path / (e.NAME + ".json"))
    assert not value["required_checks_passed"]
    assert value["verdict_class"] == "disqualified"
    assert value["arc_outcome_audit_ready_score"] == 0
