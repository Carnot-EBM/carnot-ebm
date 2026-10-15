"""REQ-REPORT-8537, REQ-VERIFY-8537: eligibility is distinct from useful progress."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import arc_capability_panel_8537 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_frozen_units():
    """SCENARIO-REPORT-8537-PANEL: no outcome changes the frozen budgets."""
    units = e.units(e.qualification.protocol())
    assert len(units) == 16
    assert len({u["episode_id"] for u in units}) == 16
    assert all(u["max_actions"] == 512 and u["max_seconds"] == 120 for u in units)
    assert sum(u["capability_mask"] for u in units) == 8


def test_missing_panel_and_duplicate():
    """SCENARIO-VERIFY-8537-REJECTION: missing is never a measured zero."""
    units = e.units(e.qualification.protocol())
    result = e.reduce([], units)
    assert len(result["rows"]) == 16
    assert result["independent_count"] == 0
    assert all(r["actual_disabled_attempts"] is None for r in result["rows"])
    with pytest.raises(ValueError, match="duplicate"):
        e.reduce([{"unit": units[0]}, {"unit": units[0]}], units)


def test_paired_efficiency_is_not_mechanical_success():
    """REQ-REPORT-8537: wall cost and levels constrain useful efficiency."""
    units = e.units(e.qualification.protocol())
    rows = [
        dict(
            u,
            status="completed",
            peak_level=0,
            runtime_s=10.0,
            actions_used=512,
            actions_to_progress=None,
            unique_visible_states=3,
            actual_disabled_attempts=2 if not u["capability_mask"] else 1,
            selected_reinduction_count=1 if not u["capability_mask"] else 0,
            selected_disabled_attempts=1 if not u["capability_mask"] else 0,
            censoring="action_limit",
            observation_hashes=["same"],
        )
        for u in units
    ]
    summary = e.pair(rows)
    assert len(summary["per_game_results"]) == 4
    assert summary["useful_efficiency"] is True
    assert summary["zero_progress_headroom"] is True
    rows[1]["runtime_s"] = 11.0
    assert e.pair(rows)["useful_efficiency"] is False
    rows[1]["runtime_s"] = 10.0
    rows[0]["peak_level"] = 1
    assert e.pair(rows)["useful_efficiency"] is False


def test_replay_absent():
    """SCENARIO-REPORT-8537-REPLAY: absence rejects before reduction."""
    assert not e.replay({})


def primitive(tmp_path, unit, *, resolved=False):
    """REQ-VERIFY-8537: fixtures qualify arithmetic, never live evidence."""
    from test_arc_effect_panel_8523 import primitive as original

    ref = original(tmp_path, unit, resolved=resolved)
    directory = Path(ref["directory"])
    terminal = json.loads((directory / "episode.json").read_bytes())
    terminal["supervisor_receipt"]["stagnations_unredirected"] = 1
    terminal["supervisor_receipt"]["unredirected_windows"] = [
        dict(selection_status="no_currently_executable_arm")
    ]
    terminal["supervisor_receipt"]["direct_effect_receipts"][0]["eligibility_predicates"].append(
        dict(
            arm="allow_reinduction",
            eligible=False,
            baseline_eligible=True,
            capability_eligible=False,
            capability_reason="runtime_explicitly_disabled",
        )
    )
    attempts = directory / "attempts.jsonl"
    attempts.write_text('[]\n[]\n[{"skipped":"disabled_by_env"}]\n')
    terminal["attempts_sha256"] = e.sha256_file(attempts)
    atomic_json(directory / "episode.json", terminal)
    ref["terminal_sha256"] = e.sha256_file(directory / "episode.json")
    steps = [json.loads(s) for s in (directory / "steps.jsonl").read_text().splitlines()]
    for step in steps:
        step["progress"] = bool(resolved and step["index"] == 3)
    (directory / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    ref["steps_sha256"] = e.sha256_file(directory / "steps.jsonl")
    return ref


def test_primitive_reduction_and_rehashed_refusals(tmp_path):
    """SCENARIO-VERIFY-8537-REJECTION keeps attempts independently observable."""
    units = e.units(e.qualification.protocol())
    refs = [primitive(tmp_path, u, resolved=True) for u in units]
    summary = e.reduce(refs, units)
    assert summary["independent_count"] == 4
    assert summary["resolved_redirects"] == 16
    assert len(summary["capability_exclusion_rows"]) == 16
    assert all(r["actual_disabled_attempts"] == 1 for r in summary["rows"])
    assert all(r["selected_reinduction_count"] == 0 for r in summary["rows"])
    assert all(r["exhaustion_states"] for r in summary["rows"])
    ref = refs[0]
    path = Path(ref["directory"]) / "episode.json"
    original = json.loads(path.read_bytes())
    bad = deepcopy(original)
    bad["unit"] = {}
    atomic_json(path, bad)
    ref["terminal_sha256"] = e.sha256_file(path)
    with pytest.raises(ValueError, match="identity"):
        e.reduce_episode(ref)
    for contents, message in [
        ("[]\n", "attempt_sequence"),
        ("[]\n[{}]\n[]\n", "attempt_sequence"),
        ("[]\n[]\n[{}]\n", "model_attempt_executed"),
    ]:
        attempts = Path(ref["directory"]) / "attempts.jsonl"
        attempts.write_text(contents)
        bad = deepcopy(original)
        bad["attempts_sha256"] = e.sha256_file(attempts)
        atomic_json(path, bad)
        ref["terminal_sha256"] = e.sha256_file(path)
        with pytest.raises(ValueError, match=message):
            e.reduce_episode(ref)


def test_real_wrapper_replays_capability_snapshots(tmp_path, monkeypatch):
    """REQ-REPORT-8537: constructed observations drive the actual wrapper in both modes."""
    from test_arc_supervisor_live_panel_8384 import Arcade, Frame

    class MovingArcade(Arcade):
        def __init__(self):
            super().__init__()
            self.index = 0

        def step(self, action, *, data=None):
            self.index += 1
            frame = Frame()
            frame.frame[0][self.index % 8][(self.index // 8) % 8] = 1 + self.index // 64
            return frame

    monkeypatch.setattr(e.history.panel_runtime, "sdk", MovingArcade)
    for unit in e.units(e.qualification.protocol())[:2]:
        directory = tmp_path / unit["mode"]
        directory.mkdir()
        unit = dict(unit, game="aa", directory=str(directory), max_actions=121)
        path = directory / "unit.json"
        atomic_json(path, unit)
        assert e.episode(path) == 0
        ref = dict(
            unit=unit,
            directory=str(directory),
            terminal_sha256=e.sha256_file(directory / "episode.json"),
            steps_sha256=e.sha256_file(directory / "steps.jsonl"),
        )
        row, events, exclusions = e.reduce_episode(ref)
        assert row["actions_used"] == 121
        assert row["model_tripwire_passed"]
        assert row["selected_disabled_attempts"] == 0
        assert row["actual_disabled_attempts"] == 1
        assert isinstance(exclusions, list)


def test_mismatched_starts_cannot_qualify_efficiency():
    """REQ-REPORT-8537: matched starts are necessary even for descriptive pairs."""
    rows = []
    for u in e.units(e.qualification.protocol()):
        rows.append(
            dict(
                u,
                status="completed",
                peak_level=0,
                runtime_s=1,
                actions_used=2,
                unique_visible_states=1,
                actual_disabled_attempts=1 if u["capability_mask"] else 2,
                selected_reinduction_count=0,
                selected_disabled_attempts=0,
                actions_to_progress=None,
                censoring="action_limit",
                observation_hashes=["different" if u["capability_mask"] else "initial"],
            )
        )
    assert not e.pair(rows)["useful_efficiency"]
    assert not any(p["matched_initial_state"] for p in e.pair(rows)["paired_results"])


def test_original_authentication_and_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8537: independently pin upstream fields, authority and code."""
    source = e.authenticate(e.ROOT, tmp_path / "clean")
    assert source["primary"]["arc_capability_ready_score"] == 1
    original = e.checked
    for case in ["field", "protocol", "authority"]:

        def changed(path, digest):
            value = original(path, digest)
            if case == "field" and value.get("experiment_id") == 8536:
                value["arc_capability_ready_score"] = 0
            elif case == "protocol" and "protocol" in value:
                value["protocol"]["max_actions"] = 511
            elif case == "authority" and "activation_reference" in value:
                value["activation_reference"]["sha256"] = "bad"
            return value

        with monkeypatch.context() as scoped:
            scoped.setattr(e, "checked", changed)
            with pytest.raises(ValueError):
                e.authenticate(e.ROOT, tmp_path / case)
    with monkeypatch.context() as scoped:
        scoped.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
        with pytest.raises(ValueError, match="bound_consumer"):
            e.authenticate(e.ROOT, tmp_path / "bound")
    with monkeypatch.context() as scoped:
        scoped.setattr(e, "canonical_hash", lambda *_: "bad")
        with pytest.raises(ValueError, match="original_authority"):
            e.authenticate(e.ROOT, tmp_path / "authority_task")
    with pytest.raises(OSError):
        e.authenticate(tmp_path / "missing", tmp_path / "absent")


def test_build_and_replay_owned_refusals(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8537-REPLAY rejects self-consistently rehashed envelopes."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")
    output = tmp_path / "results" / (e.NAME + ".json")
    source = e.authenticate(e.ROOT, tmp_path / "source")
    from carnot.reporting.v721_capstone_evidence import freeze

    for case in [
        "clean",
        "frozen",
        "protocol",
        "task",
        "absent_authority",
        "absent_source",
        "source",
        "code",
        "log",
        "aggregate",
        "source_clean",
    ]:
        changed = deepcopy(work)
        receipts = [dict(name=n, passed=True) for n in e.REQUIRED]
        if case == "frozen":
            changed["panel"]["units"] = []
        elif case == "protocol":
            changed["panel"]["protocol"]["max_actions"] = 511
            target = tmp_path / "frozen.json"
            atomic_json(target, changed["panel"])
            changed["frozen_reference"] = dict(path=str(target), sha256=e.sha256_file(target))
        elif case == "task":
            active = tmp_path / "active.yaml"
            active.write_text("tasks: []\n")
            changed["activation_reference"] = freeze(active, tmp_path / "authority")
            changed["failures"] = []
        elif case in ["absent_authority", "absent_source"]:
            changed["failures"] = []
            if case == "absent_source":
                changed["activation_reference"] = freeze(
                    e.ROOT / "research-roadmap.yaml", tmp_path / "current"
                )
        elif case in ["source", "source_clean"]:
            changed["source"] = deepcopy(source)
            if case == "source":
                changed["source"]["primary"]["headline_solve_credit"] = 1
        elif case == "code":
            changed["code_config_hashes"][str(e.ROOT / e.CLI)] = "bad"
        elif case == "log":
            log = tmp_path / "log"
            log.write_text("changed")
            receipts.append(
                dict(name="private", passed=True, stdout_path=str(log), stdout_sha256="bad")
            )
        path = tmp_path / "work.json"
        atomic_json(path, changed)
        value = e.build(path, receipts, output)
        if case == "aggregate":
            value["headline_solve_credit"] = 1
        assert e.replay(value) == (case in ["clean", "source_clean"]), case
    changed = deepcopy(work)
    changed["references"] = [
        dict(
            unit=e.units(e.qualification.protocol())[0],
            directory=str(tmp_path / "gone"),
            terminal_sha256="bad",
        )
    ]
    atomic_json(path, changed)
    value = e.build(path, receipts, output)
    assert value["verdict_class"] == "disqualified"
    assert value["failed_count"] == 1
    assert not e.replay(value)


def test_complete_cost_is_bound_to_child_clock(tmp_path):
    """REQ-VERIFY-8537: rehashed wall-cost claims must match child clock operands."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")
    ref = primitive(tmp_path / "primitive", work["panel"]["units"][0])
    work["references"] = [dict(ref, complete_wall_s=0.0)]
    work["measurement_receipts"] = [
        dict(
            name="episode_00",
            passed=True,
            duration_s=1.0,
            started_monotonic_ns=10,
            ended_monotonic_ns=1000000010,
        )
    ]
    path = tmp_path / "work.json"
    atomic_json(path, work)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert not e.replay(e.build(path, [], output))


def test_readiness_and_efficiency_are_distinct(tmp_path, monkeypatch):
    """REQ-REPORT-8537: fixture arithmetic can exercise gates without live credit."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")
    work["failures"] = []
    units = work["panel"]["units"]
    summary = e.reduce([], units)
    for row in summary["rows"]:
        row.update(
            status="completed",
            peak_level=0,
            runtime_s=1.0,
            actions_used=512,
            actions_to_progress=None,
            unique_visible_states=3,
            receipt_count=2,
            model_tripwire_passed=True,
            solve_provenance="live_agent_self_discovery",
            observation_hashes=["same"],
            actual_disabled_attempts=1 if row["capability_mask"] else 2,
            selected_reinduction_count=0 if row["capability_mask"] else 1,
            selected_disabled_attempts=0 if row["capability_mask"] else 1,
        )
    summary.update(e.pair(summary["rows"]))
    work["references"] = [
        dict(unit=u, directory=str(tmp_path / str(i)), terminal_sha256=None, steps_sha256=None)
        for i, u in enumerate(units)
    ]
    monkeypatch.setattr(e, "reduce", lambda *_: summary)
    path = tmp_path / "work.json"
    atomic_json(path, work)
    receipts = [dict(name=n, passed=True) for n in e.REQUIRED]
    value = e.build(path, receipts, tmp_path / "results" / (e.NAME + ".json"))
    assert value["arc_capability_panel_ready_score"] == 1
    assert value["capability_efficiency_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    summary["useful_efficiency"] = False
    value = e.build(path, receipts, tmp_path / "results" / (e.NAME + ".json"))
    assert value["arc_capability_panel_ready_score"] == 1
    assert value["capability_efficiency_score"] == 0
    assert value["verdict_class"] == "null"


def test_missing_child_terminal_is_owned_failure():
    """REQ-VERIFY-8537: killed or broken children cannot become skipped successes."""
    units = e.units(e.qualification.protocol())
    ref = dict(unit=units[0], directory="/absent/owned_child", terminal_sha256=None)
    summary = e.reduce([ref], units)
    assert summary["rows"][0]["status"] == "failed"
    assert all(r["status"] == "unstarted" for r in summary["rows"][1:])
