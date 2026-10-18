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


def test_real_wrapper_recording_is_passive(tmp_path):
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
        max_actions=6,
        max_seconds=120,
        observer="off",
        capability_mask=True,
    )
    left = collect(unit, SDK(), tmp_path / "off")
    right = collect(dict(unit, observer="on"), SDK(), tmp_path / "on")
    assert e.parity(left, right)["passed"]
    assert left["llm_calls"] == right["llm_calls"] == 0
    assert len(right["actions"]) == 6


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
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(tmp_path / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
    receipt = dict(name="private_control", passed=True, duration_s=0.01)
    value = e.build(path, [receipt], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
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
