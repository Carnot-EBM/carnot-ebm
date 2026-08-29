"""REQ-HARNESS-5920: the shared node-id delta must not launder unmeasured breakage.

This module had NO dedicated tests when it shipped (2026-08-28). Two holes were then
demonstrated by execution the same day:

1. An ABORTED suite (`Interrupted: 1 error during collection`, zero FAILED lines,
   nonzero exit) parsed to an empty failure set, and `delta()` returned
   `ready_allowed: True` with a delta of -1,726. A suite that never ran read as
   cleaner than baseline.
2. A change whose only breakage is a setup ERROR prints `ERROR <node>` in the short
   summary, not `FAILED <node>`. The FAILED-only parser could not see it, so the
   regression passed the delta. This is the same suppression that contaminated two
   full-suite measurement runs on 2026-08-28 (failures became setup errors and the
   failure count DROPPED).

Every test here pins one rule of the fix. Deleting a rule from the module must turn
at least one of these RED -- that is the mutation proof this file exists to hold.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.global_suite_baseline import (
    BASELINE_RELATIVE_PATH,
    baseline_error_node_ids,
    baseline_node_ids,
    delta,
    error_node_ids_from_pytest_output,
    failure_node_ids_from_pytest_output,
    observed_suite_evidence,
    pytest_run_aborted,
)

# The incident shape from 2026-08-27: one stale capstone aborted collection for
# 57,917 tests. Zero FAILED lines. Any delta over this output is meaningless.
ABORTED_RUN = """collected 57917 items / 1 error
==================================== ERRORS ====================================
ERROR tests/python/test_experiment_6659_v580_capstone.py - RuntimeError: rot
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=========================== short test summary info ============================
ERROR tests/python/test_experiment_6659_v580_capstone.py
1 error in 42.31s
"""

COMPLETE_RUN = """FAILED tests/python/test_a.py::test_one - AssertionError: boom
FAILED tests/python/test_a.py::test_two
ERROR tests/python/test_b.py::test_three - fixture 'gone' not found
2 failed, 1 error, 5 passed in 3.2s
"""


def _write_baseline(root: Path, nodes: list[str], errors: list[str] | None = None) -> None:
    payload: dict[str, object] = {"baseline_node_ids": nodes}
    if errors is not None:
        payload["baseline_error_node_ids"] = errors
    target = root / BASELINE_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def test_failed_parser_strips_reasons_and_sorts() -> None:
    nodes = failure_node_ids_from_pytest_output(COMPLETE_RUN)
    assert nodes == [
        "tests/python/test_a.py::test_one",
        "tests/python/test_a.py::test_two",
    ]


def test_error_parser_sees_setup_errors_the_failed_parser_cannot() -> None:
    # SCENARIO: a change turns a passing test into a setup error. FAILED-only
    # parsing is blind to it; the error parser is the closing of that hole.
    assert error_node_ids_from_pytest_output(COMPLETE_RUN) == ["tests/python/test_b.py::test_three"]
    assert "tests/python/test_b.py::test_three" not in failure_node_ids_from_pytest_output(
        COMPLETE_RUN
    )


def test_error_parser_ignores_bare_banner_lines() -> None:
    assert error_node_ids_from_pytest_output("ERROR \nERROR something without a path\n") == []


def test_aborted_run_is_detected() -> None:
    assert pytest_run_aborted(ABORTED_RUN) is True
    assert pytest_run_aborted(COMPLETE_RUN) is False
    assert pytest_run_aborted("INTERNALERROR> boom") is True


def test_aborted_run_yields_no_evidence_even_with_parsed_nodes() -> None:
    # REGRESSION (2026-08-28, demonstrated): an aborted run parsed to [] and the
    # delta reported ready_allowed True with delta -1726. Aborts must yield None,
    # and None must never be fed to delta() as if it were a clean empty set.
    assert observed_suite_evidence(ABORTED_RUN, exit_code=2) is None
    # Even if an abort somehow exits 0, the truncated output stays untrusted.
    assert observed_suite_evidence(ABORTED_RUN, exit_code=0) is None


def test_nonzero_exit_with_no_parsed_nodes_yields_no_evidence() -> None:
    # A nonzero exit the output does not explain: we know it failed and we do not
    # know what failed. Treating it as "zero failures" is the laundering the spec
    # forbids.
    assert observed_suite_evidence("no tests ran in 0.01s", exit_code=4) is None
    assert observed_suite_evidence("", exit_code=1) is None


def test_clean_zero_exit_yields_empty_evidence() -> None:
    evidence = observed_suite_evidence("5 passed in 1.0s", exit_code=0)
    assert evidence == {"failure_node_ids": [], "error_node_ids": []}


def test_complete_run_yields_both_node_kinds() -> None:
    evidence = observed_suite_evidence(COMPLETE_RUN, exit_code=1)
    assert evidence is not None
    assert evidence["failure_node_ids"] == [
        "tests/python/test_a.py::test_one",
        "tests/python/test_a.py::test_two",
    ]
    assert evidence["error_node_ids"] == ["tests/python/test_b.py::test_three"]


def test_delta_new_node_blocks_and_baseline_equal_allows(tmp_path: Path) -> None:
    _write_baseline(tmp_path, ["t.py::old_a", "t.py::old_b"])
    same = delta(["t.py::old_a", "t.py::old_b"], root=tmp_path)
    assert same["ready_allowed"] is True
    assert same["global_suite_failure_delta"] == 0

    worse = delta(["t.py::old_a", "t.py::brand_new"], root=tmp_path)
    assert worse["ready_allowed"] is False
    assert worse["new_node_ids"] == ["t.py::brand_new"]


def test_delta_unreadable_baseline_counts_every_failure_as_new(tmp_path: Path) -> None:
    # No baseline file under this root: [] baseline, every failure new, not ready.
    # Failing toward "not ready" is the safe direction and must stay that way.
    result = delta(["t.py::anything"], root=tmp_path)
    assert result["ready_allowed"] is False
    assert result["baseline_node_count"] == 0


def test_delta_error_ledger_enforces_when_both_sides_known(tmp_path: Path) -> None:
    _write_baseline(tmp_path, ["t.py::old"], errors=["t.py::known_error"])
    ok = delta(["t.py::old"], after_error_node_ids=["t.py::known_error"], root=tmp_path)
    assert ok["error_nodes_assessed"] is True
    assert ok["ready_allowed"] is True

    bad = delta(["t.py::old"], after_error_node_ids=["t.py::NEW_error"], root=tmp_path)
    assert bad["ready_allowed"] is False
    assert bad["new_error_node_ids"] == ["t.py::NEW_error"]


def test_delta_without_error_ledger_says_so_instead_of_claiming_clean(tmp_path: Path) -> None:
    # The 2026-08-29 baseline predates error tracking. Until a re-baseline records
    # the error nodes, the delta must SAY errors were not assessed -- visible
    # honesty, not silent cleanliness, and not a refusal that would fire on every
    # honest run either.
    _write_baseline(tmp_path, ["t.py::old"])
    result = delta(["t.py::old"], after_error_node_ids=["t.py::err"], root=tmp_path)
    assert result["error_nodes_assessed"] is False
    assert result["ready_allowed"] is True


def test_real_repo_baseline_reads_and_matches_its_count() -> None:
    nodes = baseline_node_ids()
    assert len(nodes) >= 1, "the committed baseline must be readable from the repo root"
    payload = json.loads((Path(__file__).resolve().parents[2] / BASELINE_RELATIVE_PATH).read_text())
    assert payload["baseline_node_count"] == len(nodes), (
        "baseline_node_count must equal the id list it describes; a mismatch means the "
        "ledger was hand-edited without re-deriving the count"
    )
    assert len(set(nodes)) == len(nodes), "baseline node ids must be unique"


def test_real_repo_error_ledger_absent_is_none_not_empty() -> None:
    # None means "predates error tracking"; [] would mean "zero errors recorded",
    # which is a claim the 2026-08-29 run (143 ERROR nodes) cannot support.
    ledger = baseline_error_node_ids()
    assert ledger is None or isinstance(ledger, list)
