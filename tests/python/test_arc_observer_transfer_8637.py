"""REQ-REPORT-8637, REQ-VERIFY-8637: natural evidence stays distinct from controls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_observer_transfer_8637 as e


def test_registered_panel_and_original_authentication():
    """SCENARIO-REPORT-8637-PANEL: original producer bytes fix games and budgets."""
    source = e.authenticate(e.ROOT)
    units = e.units(source["protocol"])
    assert len(units) == 16
    assert len({u["game"] for u in units}) == 4
    assert all(u["arm"] == "on" and u["capability_mask"] for u in units)
    assert {u["observer"] for u in units} == {"off", "on"}
    assert all(u["max_actions"] == 512 and u["max_seconds"] == 120 for u in units)


def test_missing_and_rehashed_original_rejected(tmp_path):
    """SCENARIO-VERIFY-8637-REPLAY: envelope hashes do not replace original pins."""
    with pytest.raises(OSError):
        e.authenticate(tmp_path)
    path = tmp_path / next(iter(e.PINS))
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="original_hash"):
        e.authenticate(tmp_path)


def test_horizons_and_support_preserve_censoring_and_zero_cells():
    """SCENARIO-REPORT-8637-PANEL: only120 actual returns complete a window."""
    event = dict(
        event_id="r",
        arm="force_exploration_diversity",
        wrapper_action_index=1,
        applied=True,
        observation=dict(level=0),
    )
    steps = [dict(index=i, action=dict(kind=1), observation=dict(level=0)) for i in range(1, 121)]
    row = e.replay_stream(steps, [event], "cutoff")[0]
    assert row["horizon_status"] == "no_progress_within_horizon"
    assert row["observed_horizon"] == 120
    assert e.replay_stream(steps[:5], [event], "cutoff")[0]["horizon_status"] == "censored"
    steps[3]["observation"]["level"] = 1
    assert (
        e.replay_stream(steps, [event], "cutoff")[0]["horizon_status"] == "level_progress_observed"
    )
    cells, support = e.support([], ["g"])
    assert len(cells) == 3 and all(r["firings"] == 0 for r in cells)
    assert not any(r["descriptive_support"] for r in support)


def test_pair_rejects_every_behavior_change():
    """SCENARIO-REPORT-8637-PANEL: receipt collection cannot alter the policy."""
    left = dict(actions=[1], rng=[2], termination="cap", legacy=dict(a=1), observations=[3])
    assert e.parity(left, deepcopy(left))["passed"]
    for key in left:
        right = deepcopy(left)
        right[key] = None
        assert not e.parity(left, right)["passed"]


@pytest.mark.parametrize("budget", [6, 128])
def test_real_wrapper_recording_is_passive(tmp_path, budget):
    """SCENARIO-REPORT-8637-PANEL: scripted SDK controls use genuine wrapper calls."""
    from carnot.reporting.arc_observer_live_8637 import collect
    from carnot.reporting.arc_observer_fixtures_8624 import frame

    class Env:
        def __init__(self):
            self.index = 0

        def reset(self):
            self.index += 1
            return frame("navigation", self.index, "cutoff")

        def step(self, *args, **kwargs):
            return self.reset()

    class SDK:
        def make(self, *args, **kwargs):
            return Env()

        def open_scorecard(self):
            return "private_control"

    unit = dict(
        episode_id="fixture",
        game="fixture-unknown",
        seed=11,
        arm="on",
        max_actions=budget,
        max_seconds=120,
        observer="off",
        capability_mask=True,
    )
    left = collect(unit, SDK(), tmp_path / "off")
    right = collect(dict(unit, observer="on"), SDK(), tmp_path / "on")
    assert e.parity(left, right)["passed"]
    assert left["llm_calls"] == right["llm_calls"] == 0
    assert len(right["actions"]) == budget


def test_build_blocked_disqualified_and_anchored_replay(tmp_path):
    """SCENARIO-VERIFY-8637-REPLAY: cold reduction rejects rehashed claim changes."""
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file

    work = dict(
        run_date="20261018",
        input_root=str(tmp_path / "absent"),
        failures=[
            dict(
                check="missing",
                upstream="exp8627-contract-methods",
                path="missing",
                hash=None,
                field="sha256",
                operator="==",
                expected="original",
                observed=None,
                passed=False,
            )
        ],
        references=[],
        games=["g"],
        registry_precheck={},
        preconditions_checked={},
        source_artifact_hashes={},
        code_config_hashes={},
        duration_s=0.01,
        phase_spans=[],
        historical_model_provenance={},
        measurement_receipts=[dict(name="private_control")],
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(tmp_path / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
    receipt = dict(name="private_control", passed=True, duration_s=0.01)
    value = e.build(path, [receipt], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["duration_s"] == 0.01
    assert e.replay(value)
    assert (
        e.build(path, [dict(receipt, passed=False)], tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )
    changed = deepcopy(value)
    changed["completed_count"] = 991
    changed["reproducibility_checksum"] = e.canonical_hash(changed)
    assert not e.replay(changed)
    work["failures"] = []
    atomic_json(path, work)
    changed["work_reference"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="measurement_seal"):
        e.replay(changed)


def test_cli_actual_private_children_and_rehashed_rejection(tmp_path):
    """SCENARIO-VERIFY-8637-REPLAY: fresh CLI exercises actual publication readers."""
    import os
    import subprocess
    import sys

    script = e.ROOT / e.CLI
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(script), *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    assert run("--date", "invalid").returncode == 2
    assert run("--deliberate-error").returncode == 7
    assert run("--cold-replay", tmp_path / "missing.json").returncode == 1
    for owned in [False, True]:
        output = tmp_path / str(owned) / (e.NAME + ".json")
        args = [
            "--date",
            "20261018",
            "--root",
            tmp_path / "absent",
            "--output",
            output,
            "--private-control",
        ]
        if owned:
            args.append("--owned-failure")
        result = run(*args)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if owned else "blocked")
        assert run("--cold-replay", output).returncode == 0
        value["completed_count"] = 991
        value["reproducibility_checksum"] = e.canonical_hash(value)
        output.write_text(json.dumps(value))
        assert run("--cold-replay", output).returncode == 1


@pytest.mark.parametrize("mutation", ["readiness", "publication", "protocol"])
def test_authentication_semantic_rejections(monkeypatch, mutation):
    """REQ-VERIFY-8637: original hashes also require qualified field semantics."""
    original = e.checked

    def changed(path, digest):
        value = original(path, digest)
        if mutation == "readiness" and path.name.startswith("experiment_8624"):
            value["arc_observer_ready_score"] = 0
        if mutation == "protocol" and path.name == "v741-methods-protocol.json":
            value["arc_observer_protocol"]["horizon"] = 1
        return value

    monkeypatch.setattr(e, "checked", changed)
    if mutation == "publication":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    with pytest.raises(ValueError):
        e.authenticate(e.ROOT)


def test_cutoff_overlap_and_support_floor():
    """REQ-REPORT-8637: five firings in three games supplies description only."""
    event = dict(
        event_id="r",
        arm="drop_goal_bias",
        wrapper_action_index=1,
        applied=True,
        observation=dict(level=0),
    )
    step = dict(index=1, action=dict(kind=1), observation=dict(level=0))
    assert e.replay_stream([step], [event], "time_limit")[0]["end_reason"] == "time_limit"
    rows = [
        dict(game=g, arm="drop_goal_bias", horizon_status="censored")
        for g in ["a", "b", "c"]
        for o in ["off", "on"]
        for _ in range(5)
    ]
    for i, row in enumerate(rows):
        row["observer"] = ["off", "on"][(i // 5) % 2]
    cells, arms = e.support(rows, ["a", "b", "c", "d"])
    assert arms[0]["descriptive_support"]
    assert cells[0]["observer_counts"] == dict(off=5, on=5)
    assert not arms[0]["refinement_authorized"]
    assert any(c["firings"] == 0 for c in cells)


def test_controller_orchestration_and_private_root_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8637: exercise controller without recursively running validators."""
    from carnot.reporting import arc_observer_runner_8637 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    with pytest.raises(SystemExit):
        runner.main(["--private-control", "--root", str(tmp_path)])
    monkeypatch.setattr(runner, "measure", lambda *args: dict(measurement_receipts=[]))
    monkeypatch.setattr(runner.validation, "validate_checks", lambda *args: [dict(passed=True)])
    monkeypatch.setattr(runner, "validation_bundle", lambda path: path)
    monkeypatch.setattr(e, "build", lambda *args: dict(completed_count=16))
    monkeypatch.setattr(runner, "publish", lambda *args: dict(passed=True))
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261018", "--output", str(output)]) == 0
    from carnot.reporting import arc_observer_live_8637 as live

    monkeypatch.setattr(live, "episode", lambda path: 0)
    assert runner.main(["--episode", str(tmp_path / "unit.json")]) == 0
    monkeypatch.setattr(runner, "child", lambda *args, **kwargs: dict(passed=False))
    monkeypatch.undo()
    monkeypatch.setattr(runner, "child", lambda *args, **kwargs: dict(passed=False))
    value = dict(
        experiment_id=8637,
        task_id=e.TASK,
        honest_verdict="complete_null_control",
        verdict_class="null",
        real_horizon_rows_ready_score=0,
    )
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.publish(value, tmp_path / "reject" / output.name, tmp_path / "raw")


def test_build_complete_panel_and_live_replay_rejection(tmp_path):
    """REQ-REPORT-8637: complete records can qualify only descriptive instrumentation."""
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file

    event = dict(
        event_id="r",
        arm="drop_goal_bias",
        wrapper_action_index=1,
        applied=True,
        observation=dict(level=0),
    )
    steps = [dict(index=1, action=dict(kind=1), observation=dict(level=0))]
    episode = dict(
        actions=[dict(kind=1)],
        observations=[dict(level=0)],
        rng=[1],
        termination=dict(censoring="action_limit", error=None),
        legacy={},
        steps=steps,
        events=[event],
        end_reason="action_limit",
        observer_rows=e.replay_stream(steps, [event], "action_limit"),
        numerical_head_calls=0,
    )
    refs = []
    for g in ["a", "b", "c", "d"]:
        for s in [11, 22]:
            for o in ["off", "on"]:
                path = tmp_path / f"{g}-{s}-{o}.json"
                ep = dict(
                    episode,
                    unit=dict(
                        game=g, seed=s, observer=o, pair_key=f"{g}:{s}", episode_id=f"{g}:{s}:{o}"
                    ),
                )
                atomic_json(path, ep)
                refs.append(dict(path=str(path), sha256=sha256_file(path)))
    work = dict(
        run_date="20261018",
        references=refs,
        games=["a", "b", "c", "d"],
        failures=[],
        historical_model_provenance={},
        registry_precheck={},
        preconditions_checked={},
        source_artifact_hashes={},
        code_config_hashes={},
        duration_s=0.01,
        phase_spans=[],
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(tmp_path / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
    value = e.build(
        path, [dict(name="control", passed=True, duration_s=0)], tmp_path / (e.NAME + ".json")
    )
    assert value["real_horizon_rows_ready_score"] == 1
    assert value["verdict_class"] == "null" and value["refinement_eligible_score"] == 0
    assert len(value["censored_reason_rows"]) == 16
    assert e.replay(value)
    ep["observer_rows"] = []
    atomic_json(Path(refs[-1]["path"]), ep)
    refs[-1]["sha256"] = sha256_file(Path(refs[-1]["path"]))
    atomic_json(path, work)
    with pytest.raises(ValueError, match="live_replay_drift"):
        e.build(path, [], tmp_path / (e.NAME + ".json"))


@pytest.mark.parametrize("case", ["normal", "authentication", "deadline"])
def test_measure_boundaries_and_hash_custody(tmp_path, monkeypatch, case):
    """REQ-VERIFY-8637: prechecks precede real child launch and cap the loop."""
    from carnot.reporting import arc_observer_runner_8637 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    root = tmp_path / "root"
    root.mkdir()
    (root / "research-roadmap.yaml").write_text("tasks: []")
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(runner.validation, "preconditions", lambda *args: ([], {}, {}))
    monkeypatch.setattr(e, "PINS", {})
    source = dict(protocol=dict(games=["a"]), historical_model_provenance={})
    if case == "authentication":

        def authenticate(root):
            raise ValueError("changed original")

        monkeypatch.setattr(e, "PINS", {"missing": "pin", "second": "pin"})
        for name in e.PINS:
            (root / name).write_text("{}")
        monkeypatch.setattr(runner, "sha256_file", lambda path: "pin")
        monkeypatch.setattr(e, "authenticate", authenticate)
    else:
        monkeypatch.setattr(e, "authenticate", lambda root: source)
    unit = dict(
        game="a",
        episode_id="fixture",
        observer="on",
        arm="on",
        seed=11,
        max_actions=1,
        max_seconds=120,
        capability_mask=True,
    )
    monkeypatch.setattr(e, "units", lambda protocol: [unit])

    def child(name, argv, *args, **kwargs):
        u = json.loads(Path(argv[-1]).read_bytes())
        atomic_json(Path(u["directory"]) / "public_stream.json", dict(unit=u))
        return dict(name=name, passed=True)

    monkeypatch.setattr(runner, "child", child)
    if case == "deadline":
        monkeypatch.setattr(runner.time, "monotonic_ns", iter([0, 3601 * 10**9]).__next__)
        with pytest.raises(TimeoutError, match="measurement_cap"):
            runner.measure(root, raw, tmp_path / "private", "20261018", True)
    else:
        value = runner.measure(root, raw, tmp_path / "private", "20261018", True)
        assert bool(value["failures"]) == (case == "authentication")
        assert len(value["references"]) == (0 if case == "authentication" else 1)


def test_live_episode_entry_uses_official_sdk_factory(tmp_path, monkeypatch):
    """REQ-VERIFY-8637: episode entry reports real errors and tripwire attempts."""
    from carnot.reporting import arc_observer_live_8637 as live
    from carnot.reporting import arc_supervisor_live_panel_8384 as sdk_module

    path = tmp_path / "unit.json"
    path.write_text(json.dumps(dict(directory=str(tmp_path))))
    monkeypatch.setattr(sdk_module, "sdk", lambda: object())
    monkeypatch.setattr(
        live,
        "collect",
        lambda *args: dict(termination=dict(error=None), model_tripwire_attempts=[]),
    )
    assert live.episode(path) == 0
    monkeypatch.setattr(
        live,
        "collect",
        lambda *args: dict(termination=dict(error="sdk_error"), model_tripwire_attempts=[]),
    )
    assert live.episode(path) == 1


def test_missing_head_blocks_cpu_panel(tmp_path, monkeypatch):
    """REQ-VERIFY-8637: a missing required numeric head differs from optional GPU absence."""
    from carnot.reporting import arc_observer_runner_8637 as runner

    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(runner.validation, "preconditions", lambda *args: ([], {}, {}))
    monkeypatch.setattr(e, "PINS", {})
    monkeypatch.setattr(e, "authenticate", lambda *args: dict(protocol=dict(games=[])))
    original = runner.sha256_file
    monkeypatch.setattr(
        runner, "sha256_file", lambda path: "foreign" if path.suffix == ".pt" else original(path)
    )
    value = runner.measure(tmp_path, raw, tmp_path / "private", "20261018", False)
    assert value["failures"][0]["check"] == "numerical_head"
    assert not value["references"]


def test_validation_bundle_preserves_original_source_and_primitives(tmp_path, monkeypatch):
    """REQ-VERIFY-8637: validation additions cannot replace measured operands."""
    from carnot.reporting import arc_observer_runner_8637 as runner
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file

    source = tmp_path / "source.py"
    source.write_text("original = 1\n")
    raw = tmp_path / "raw"
    snapshot = raw / "measurement_code/source.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    work = dict(source_artifact_hashes={str(source): sha256_file(source)}, references=[])
    path = raw / "work.json"
    atomic_json(path, work)
    original = sha256_file(path)
    monkeypatch.setattr(e, "ROOT", tmp_path)
    monkeypatch.setattr(e, "OWNED", ["source.py"])
    monkeypatch.setattr(e, "TESTS", [])
    source.write_text("updated = 2\n")
    result = runner.validation_bundle(path)
    assert sha256_file(path) == original
    value = json.loads(result.read_bytes())
    assert value["original_measurement_reference"]["sha256"] == original
    assert value["source_artifact_hashes"][str(snapshot)] == sha256_file(snapshot)
    snapshot.unlink()
    with pytest.raises(ValueError, match="measurement_source_custody"):
        runner.validation_bundle(path)


def test_rehashed_work_cannot_substitute_original_measurement(tmp_path):
    """REQ-VERIFY-8637: a separate immutable original defeats rehashed substitutions."""
    from carnot.reporting.current_work_receipt import atomic_json, sha256_file

    original = tmp_path / "original.json"
    atomic_json(original, dict(references=["original"]))
    work = dict(
        references=[],
        source_artifact_hashes={},
        code_config_hashes={},
        original_measurement_reference=dict(path=str(original), sha256=sha256_file(original)),
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(tmp_path / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
    with pytest.raises(ValueError, match="original_measurement_substitution"):
        e.replay(dict(work_reference=dict(path=str(path), sha256=sha256_file(path))))


def test_original_child_shards_survive_combining(tmp_path, monkeypatch):
    """REQ-VERIFY-8637: preserve actual SQLite child data before any combine."""
    from carnot.reporting import arc_observer_runner_8637 as runner
    from carnot.reporting.current_work_receipt import sha256_file
    import coverage

    private = tmp_path / "private"
    private.mkdir()
    shard = private / ".coverage.original-child"
    data = coverage.CoverageData(basename=str(shard))
    data.add_lines({str(e.ROOT / e.OWNED[0]): {1}})
    data.write()
    original = sha256_file(shard)
    seen = []

    def child(name, argv, *args, **kwargs):
        seen.append(argv)
        return dict(name=name, passed=True)

    def validate(raw, private):
        return [
            runner.validation.child("coverage_combine", ["coverage", "combine", str(private)], raw)
        ]

    monkeypatch.setattr(runner.validation, "validate_checks", validate)
    monkeypatch.setattr(runner, "child", child)
    result = runner.validate_checks(tmp_path / "raw", private)
    assert result[0]["passed"] and "--keep" in seen[0]
    manifest = json.loads((tmp_path / "raw" / "original_child_shards.json").read_bytes())
    assert manifest[0]["sha256"] == original
    assert sha256_file(Path(manifest[0]["path"])) == original
    assert sha256_file(shard) == original
    shard.unlink()
    with pytest.raises(ValueError, match="missing_original_child_shards"):
        runner.validate_checks(tmp_path / "empty", private)
