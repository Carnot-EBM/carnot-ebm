"""Tests for the conductor pre-test poison-test auto-quarantine guard.

Spec: REQ-AUTORESEARCH-POISON-GUARD — a single agent's broken experiment-specific
test must not cascade-block an entire milestone via the pre-test gate. After a
test fails the smart-subset gate PRETEST_POISON_THRESHOLD times in a row it is
auto-quarantined (moved to tests/python/quarantine/, which conftest excludes),
and the operator is notified.

Incident lineage (the cascades this guard prevents): exp3521 (.325),
exp3544 (.326), exp3612 (.332), exp3827 (.351 -> blocked exp3828-3833, .352
archived empty).
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

import research_conductor as rc


def test_failed_name_to_test_file_parses_nodeid():
    # SCENARIO: a captured "FAILED <file>::<test>" line yields the file path.
    assert (
        rc._failed_name_to_test_file(
            "FAILED tests/python/test_experiment_3827_x.py::test_blocked_no_cuda"
        )
        == "tests/python/test_experiment_3827_x.py"
    )
    assert (
        rc._failed_name_to_test_file("ERROR tests/python/test_exp99_y.py::test_z")
        == "tests/python/test_exp99_y.py"
    )
    assert rc._failed_name_to_test_file("garbage-no-space") is None


def test_only_experiment_tests_are_quarantinable():
    # SCENARIO: experiment-specific tests are eligible; core/shared tests never.
    assert rc._is_auto_quarantinable("tests/python/test_experiment_3827_x.py")
    assert rc._is_auto_quarantinable("tests/python/test_exp99_y.py")
    # Core / shared-module tests must keep blocking the gate (real regressions).
    assert not rc._is_auto_quarantinable("tests/python/test_pipeline_extract.py")
    assert not rc._is_auto_quarantinable("tests/python/test_docs.py")
    assert not rc._is_auto_quarantinable("tests/python/test_cli.py")
    # Already-quarantined paths are not re-flagged.
    assert not rc._is_auto_quarantinable(
        "tests/python/quarantine/test_experiment_3827_x.py"
    )
    # Non-test files / non-python excluded.
    assert not rc._is_auto_quarantinable("python/carnot/verify/foo.py")


def test_quarantine_only_after_threshold_consecutive_failures():
    # SCENARIO: the same experiment test must fail the gate THRESHOLD times in a
    # row before it is quarantined — a single transient failure does not trigger.
    fails = ["FAILED tests/python/test_experiment_3827_x.py::test_blocked_no_cuda"]
    counter: dict[str, int] = {}
    for i in range(1, rc.PRETEST_POISON_THRESHOLD):
        to_q, counter = rc._compute_poison_quarantine_decision(fails, counter)
        assert to_q == [], f"quarantined too early at run {i}"
        assert counter["tests/python/test_experiment_3827_x.py"] == i
    # The threshold-th consecutive failure quarantines and clears the counter.
    to_q, counter = rc._compute_poison_quarantine_decision(fails, counter)
    assert to_q == ["tests/python/test_experiment_3827_x.py"]
    assert "tests/python/test_experiment_3827_x.py" not in counter


def test_counter_resets_when_test_stops_failing():
    # SCENARIO: an intermittent test that passes between failures never reaches
    # the threshold — its counter resets when it is absent from a fail set.
    tf = "tests/python/test_experiment_3827_x.py"
    fails = [f"FAILED {tf}::test_a"]
    _, counter = rc._compute_poison_quarantine_decision(fails, {})
    assert counter[tf] == 1
    # A subsequent gate run where this test does NOT fail drops its count.
    to_q, counter = rc._compute_poison_quarantine_decision(
        ["FAILED tests/python/test_experiment_9999_other.py::test_b"], counter
    )
    assert tf not in counter
    assert to_q == []


def test_core_test_never_quarantined_even_at_high_count():
    # SCENARIO: a core test failing many times is a REAL regression and must
    # never be auto-quarantined (it should keep blocking the gate).
    seeded = {"tests/python/test_pipeline_extract.py": 99}
    to_q, _ = rc._compute_poison_quarantine_decision(
        ["FAILED tests/python/test_pipeline_extract.py::test_x"], seeded
    )
    assert to_q == []
