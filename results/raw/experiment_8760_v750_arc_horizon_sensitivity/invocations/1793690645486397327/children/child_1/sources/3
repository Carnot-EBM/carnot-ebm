"""REQ-REPORT-8760 / REQ-VERIFY-8760: deletion cannot create evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_horizon_sensitivity_8760 as s
from carnot.reporting import v709_execution
from carnot.reporting import arc_horizon_bounds_8760 as numeric


def cli(*args):
    """Exercise real script imports and replay from outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(s.e.ROOT / s.e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-VERIFY-8760: fixture publication owns no production bytes."""
    path = tmp_path_factory.mktemp("arc8760") / (s.e.NAME + ".json")
    result = cli("--private-control", "--output", path, "--date", "20261103")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_panel_and_grid(private):
    """SCENARIO-REPORT-8760: retain every original denominator and unknown."""
    path, value = private
    assert value["experiment_id"] == 8760
    assert value["milestone"] == "2026.11.750"
    assert value["intended_count"] == value["censored_count"] == 16
    assert value["completed_count"] == 0 and value["independent_count"] == 4
    assert value["new_event_rows"] == [] and value["new_level_credit"] == 0
    assert value["frontier_before"] == value["frontier_after"]
    assert value["arc_sensitivity_ready_score"] == 0
    assert len(value["censoring_sensitivity_rows"]) == 100
    assert all(r["passed"] for r in value["independent_reduction_rows"])
    assert all(r["independent_count"] == 4 for r in value["independent_reduction_rows"])
    assert all(r["reconstructed_episode_count"] == 16 for r in value["independent_reduction_rows"])
    assert all(r["intended_count"] == 16 for r in value["censoring_sensitivity_rows"])
    assert all(not r["generalization_claim"] for r in value["overlap_rows"])
    assert set(value) <= set(value["field_principles"])
    assert cli("--cold-replay", path).returncode == 0


def test_unknown_zero_and_monotone():
    """REQ-REPORT-8760: unknown intervals retain unobserved counterfactuals."""
    rows = [
        dict(
            episode_id=f"g:{i}",
            game="g",
            arm=s.ARMS[0],
            selected=True,
            applied=True,
            binary_outcome=0,
            masked=False,
        )
        for i in range(5)
    ]
    known = s.bounds(rows)
    assert known["arm_rows"][0]["identified_bounds"] == [0, 0]
    missing = deepcopy(rows)
    missing[0]["binary_outcome"] = None
    reduced = s.bounds(missing)
    assert reduced["arm_rows"][0]["identified_bounds"] == [0, 0.2]
    assert reduced["arm_rows"][1]["identified_bounds"] == [0, 1]
    s.assert_widens(known, reduced)
    with pytest.raises(ValueError, match="manufactured_support"):
        s.assert_widens(reduced, known)
    assert known["arm_rows"][1]["complete_case_mean"] is None
    assert not known["overlap_rows"][0]["generalization_claim"]


def test_reconstruction_count_units():
    """REQ-REPORT-8760: recorder siblings remain one independent game."""
    upstream = s.e.read(dict(path=str(s.e.ROOT / s.e.UPSTREAM), sha256=s.e.PIN))
    work = s.e.read(upstream["primitive_reference"])
    episodes = [s.e.read(ref) for ref in work["episode_refs"][:2]]
    result = numeric.grid(episodes, lambda *args: None)
    assert all(r["independent_count"] == 1 for r in result["independent_reduction_rows"])
    assert all(r["reconstructed_episode_count"] == 2 for r in result["independent_reduction_rows"])


def test_overlap_floor():
    """REQ-REPORT-8760: games, seeds and siblings never multiply independence."""
    rows = [
        dict(
            episode_id=f"{g}:{a}:{i}",
            game=g,
            arm=a,
            selected=True,
            applied=True,
            binary_outcome=i % 2,
            masked=False,
        )
        for g in ["a", "b", "c"]
        for a in s.ARMS[:2]
        for i in range(5)
    ]
    result = s.bounds(rows)
    pair = result["overlap_rows"][0]
    assert pair["admission_passed"] and pair["shared_games"] == ["a", "b", "c"]
    assert not pair["generalization_claim"]
    assert not s.bounds(rows[:-1])["overlap_rows"][0]["admission_passed"]
    assert not s.bounds(rows[:20])["overlap_rows"][0]["admission_passed"]


def test_rehashed_tamper_and_cli_errors(private, tmp_path):
    """SCENARIO-VERIFY-8760: changing a checksum cannot invent a benefit."""
    _, value = private
    value = deepcopy(value)
    value["new_level_credit"] = 1
    value["reproducibility_checksum"] = s.e.checksum(value)
    candidate = tmp_path / "tampered.json"
    s.e.atomic_json(candidate, value)
    assert cli("--cold-replay", candidate).returncode == 1
    for args in [("--date", "bad"), ("--date", "20261303"), ("--private-control",)]:
        assert cli(*args).returncode == 1
    failed = tmp_path / "failed" / (s.e.NAME + ".json")
    result = cli("--private-control", "--failed-child", "--output", failed)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(failed.read_bytes())["verdict_class"] == "disqualified"


def test_external_absence(tmp_path, monkeypatch):
    """REQ-VERIFY-8760: a missing upstream is blocked with an exact operand."""
    monkeypatch.setattr(s, "FRONTIER", "results/absent8760.json")
    output = tmp_path / (s.e.NAME + ".json")
    assert s.runtime.run(output, "20261103", True, False) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][-1]["operand_state"] == "missing_file"


def test_primitive_mask_controls():
    """REQ-REPORT-8760: constructed controls distinguish observed zero and cutoff.

    These tiny private controls exercise positive and negative branches. They
    never enter the scientific panel or its source receipts.
    """
    original = [dict(episode_id="fixture:0", game="fixture", arm=s.ARMS[0])]
    episodes = [
        dict(
            unit=dict(game="fixture"),
            observer_rows=[
                dict(
                    arm=s.ARMS[0],
                    observed_horizon=40,
                    sequence=list(range(40)),
                    horizon_status="level_progress_observed",
                    level_progress=1,
                    end_reason="progress",
                )
            ],
        )
    ]
    saved = deepcopy(episodes)
    assert numeric.reconstruct(episodes, original)[0]["binary_outcome"] == 1
    masked = numeric.perturb(episodes, None, 30, None)
    assert numeric.reconstruct(masked, original)[0]["binary_outcome"] is None
    assert masked[0]["observer_rows"][0]["observed_horizon"] == 30
    assert episodes == saved
    early = deepcopy(episodes)
    early[0]["observer_rows"][0]["observed_horizon"] = 10
    assert (
        numeric.reconstruct(numeric.perturb(early, None, 30, None), original)[0]["binary_outcome"]
        == 1
    )
    zero = deepcopy(episodes)
    zero[0]["observer_rows"][0].update(
        horizon_status="no_progress_within_horizon", observed_horizon=120
    )
    assert numeric.reconstruct(zero, original)[0]["binary_outcome"] == 0


def test_independent_reconstruction_controls(private, monkeypatch):
    """SCENARIO-REPORT-8760: independent disagreement is an owned failure."""
    _, value = private
    work = s.e.read(value["primitive_reference"])
    episodes = [s.e.read(ref) for ref in work["episode_refs"]]
    reducer = numeric.frozen_reduce
    calls = []

    def disagree(inputs):
        result = reducer(inputs)
        if calls:
            result["censoring_rows"][0]["observation_count"] = -1
        calls.append(True)
        return result

    monkeypatch.setattr(numeric, "frozen_reduce", disagree)
    with pytest.raises(ValueError, match="independent_reduction_disagreement"):
        numeric.grid(episodes, lambda *args: None)
    monkeypatch.setattr(numeric, "frozen_reduce", reducer)
    broken = deepcopy(work)
    broken["analysis"]["rows"][0]["binary_outcome"] = 1
    with pytest.raises(ValueError, match="independent_grid_drift"):
        s.build(broken, value["validation_receipts"], Path(value["publication_output"]))


def test_independent_intervals():
    """REQ-REPORT-8760: reconstruct endpoints separately from count formulas."""
    rows = [
        dict(
            game="fixture",
            arm=s.ARMS[0],
            applied=True,
            selected=True,
            binary_outcome=1,
            masked=False,
        )
    ]
    expected = s.bounds(rows)
    bad = deepcopy(expected)
    bad["overlap_rows"][0]["full_roster_effect_bounds"] = [0, 0]
    with pytest.raises(ValueError, match="independent_interval_disagreement"):
        numeric.check_intervals(rows, bad)
    assert numeric.check_intervals([], s.bounds([]))


def test_actual_validation_adapter(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8760: run the real harness on a tiny private pytest child.

    This validates adapter wiring without recursively running the production
    suite. Its measured fixture coverage grants no production science credit.
    """
    source = tmp_path / "test_private_adapter.py"
    source.write_text(
        '"""REQ-VERIFY-8760: private harness control."""\n\n\n'
        "def test_private() -> None:\n    assert 2 + 2 == 4\n"
    )
    (tmp_path / ".venv").symlink_to(s.e.ROOT / ".venv", target_is_directory=True)
    (tmp_path / "scripts").symlink_to(s.e.ROOT / "scripts", target_is_directory=True)
    suite = source.name
    monkeypatch.setattr(v709_execution, "ROOT", tmp_path)
    monkeypatch.setattr(s.checks.a, "ROOT", tmp_path)
    for field, value in [("TESTS", [suite]), ("CONSUMERS", []), ("OWNED", [suite])]:
        monkeypatch.setattr(s.checks.a, field, value)
    raw = tmp_path / "raw"
    plan, collected = s.checks.collect(tmp_path, raw, [suite], [suite], s.checks.child)
    s.e.atomic_json(raw / "preregistered.json", dict(test_plan=plan, collection=collected))
    receipts = s.validate(raw)
    assert all(row["passed"] for row in receipts)
    assert s.e.read(receipts[-3]["details_reference"])["coverage"]["complete"]
    path, value = private
    work = s.e.read(value["primitive_reference"])
    # An isolated harness control may qualify mechanics, never science or solves.
    candidate = s.build(work, receipts, path)
    assert candidate["arc_sensitivity_ready_score"] == 1
    assert candidate["independent_generalization_score"] == candidate["new_level_credit"] == 0


def test_new_receipt_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8760: a rejected external producer cannot advance the frontier."""

    def rejected(*args):
        raise ValueError("private_rejected_producer")

    monkeypatch.setattr(s.e, "delta", rejected)
    work = s.measure(tmp_path / "raw" / "invocations" / "private", "20261103")
    assert work["frontier_before"] == work["frontier_after"]
    assert work["new_event_rows"] == []
    gate = work["failures"][-1]
    assert gate["operand_state"] == "rejected_producer"
    assert gate["field"] == "new_events" and gate["expected"] == "authenticated"
    assert gate["observed"] == "private_rejected_producer"


def test_archived_reader_bindings(tmp_path):
    """REQ-VERIFY-8760: archive dependencies without editing historical assertions."""
    suites, refs = s.scoped_suites(tmp_path, True)
    suite = suites[1]
    original = s.e.ROOT / "tests/python/test_arc_outcome_frontier_8746.py"
    assert Path(suite).read_bytes() == original.read_bytes()
    fixture = tmp_path / "archived_v749"
    assert s.qualified.methods.authority(fixture)["ready"]
    assert refs and all(s.e.sha256_file(Path(r["path"])) == r["sha256"] for r in refs)
    assert s.e.sha256_file(fixture / "ops/arc_solve_registry.yaml") == s.e.sha256_file(
        s.e.ROOT / "ops/arc_solve_registry.yaml"
    )
