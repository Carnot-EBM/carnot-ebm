"""Three conductor guards hardened after the 2026-09-07 orphan-test incident.

A task that writes its test before its module and then dies leaves a file that
fails at COLLECT time. The pre-test gate skips the NEXT task, not the one that
wrote it. That happened three times in one day, twice to the critical task.

Spec: REQ-CONDUCTOR-RECEIPT-1 (exit-code fallback), REQ-INFRA-067 (pre-test gate)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from research_conductor import (  # noqa: E402
    _counter_after_passing_run,
    _counter_key_file,
    _drop_orphan_tests,
)


class _FakeAudit:
    def __init__(self, details):
        self.failure_details = details


def test_orphan_test_is_dropped_from_the_subset(monkeypatch) -> None:
    """A test importing a module that does not exist cannot run, so it must not gate."""

    import research_conductor as rc

    orphan = "tests/python/test_experiment_9999_never_written.py"
    good = "tests/python/test_adaptive_sleep.py"
    monkeypatch.setattr(
        rc,
        "_orphan_audit",
        lambda paths: _FakeAudit(
            [f"{rc.PROJECT_ROOT}/{orphan}:10: orphan local import carnot.experiment_9999"]
        ),
    )
    assert rc._drop_orphan_tests([good, orphan]) == [good]


def test_a_clean_subset_is_returned_untouched(monkeypatch) -> None:
    """No orphans means the list is passed through, not rebuilt or reordered."""

    import research_conductor as rc

    monkeypatch.setattr(rc, "_orphan_audit", lambda paths: _FakeAudit([]))
    files = ["tests/python/test_a.py", "tests/python/test_b.py"]
    assert rc._drop_orphan_tests(files) == files


def test_orphan_filter_never_breaks_the_gate_it_protects(monkeypatch) -> None:
    """If the auditor raises, the subset passes through unfiltered rather than emptying.

    A guard that fails closed here would stop every task instead of one bad file.
    """

    import research_conductor as rc

    def _boom(paths):
        raise RuntimeError("auditor exploded")

    monkeypatch.setattr(rc, "_orphan_audit", _boom)
    files = ["tests/python/test_adaptive_sleep.py", "tests/python/does_not_matter.py"]
    assert rc._drop_orphan_tests(files) == files


def test_poison_counter_clears_only_the_tests_that_actually_ran() -> None:
    """A test earns a clean slate by RUNNING and passing, not by someone else passing.

    Resetting every counter on any green run is why an intermittently-selected
    poison never reached the threshold: the gate runs a subset, so a test picked
    one run in five had its count zeroed by four unrelated successes in between.
    """

    counter = {
        "tests/python/test_ran.py": 2,
        "tests/python/test_not_in_this_subset.py": 2,
    }
    after = _counter_after_passing_run(counter, {"tests/python/test_ran.py"})
    assert after == {"tests/python/test_not_in_this_subset.py": 2}


def test_poison_counter_key_shapes_resolve_to_a_file() -> None:
    """Keys are recorded from pytest summary lines, which carry a status prefix."""

    assert _counter_key_file("tests/python/test_x.py") == "tests/python/test_x.py"
    assert _counter_key_file("ERROR tests/python/test_x.py") == "tests/python/test_x.py"
    assert _counter_key_file("FAILED tests/python/test_x.py::test_y") == "tests/python/test_x.py"


def test_findings_exit_code_is_not_a_failed_audit(monkeypatch) -> None:
    """A linter's non-zero exit means FINDINGS; only the caller can say if that is ok.

    The adversarial-verify backfill returns 1 when it stamped artifacts. Read as
    failure it BLOCKed milestone activation exactly when the sweep was useful.
    """

    import research_conductor as rc

    # A refused audit writes a BLOCK line to the tracked conductor log. A test
    # must never write the research record, so silence that one call.
    monkeypatch.setattr(rc, "log_step", lambda *a, **k: None)
    exit_one = [sys.executable, "-c", "raise SystemExit(1)"]
    assert rc._run_audit_with_receipt("t", exit_one, None, 60, ok_returncodes=(0, 1)) is True
    assert rc._run_audit_with_receipt("t", exit_one, None, 60) is False


def test_a_real_failure_is_still_a_failure_with_the_wider_codes(monkeypatch) -> None:
    """Widening to (0, 1) must not accept a crash: those return a different code."""

    import research_conductor as rc

    monkeypatch.setattr(rc, "log_step", lambda *a, **k: None)
    crash = [sys.executable, "-c", "raise SystemExit(2)"]
    assert rc._run_audit_with_receipt("t", crash, None, 60, ok_returncodes=(0, 1)) is False


def _calls_inside(function_name: str) -> set[str]:
    """Every function called inside one top-level function, read from the AST.

    Parsed, not grepped: a text search would match the name in a comment or a
    docstring and report wiring that is not there.
    """

    import ast

    source = (Path(__file__).resolve().parents[2] / "scripts" / "research_conductor.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
    raise AssertionError(f"{function_name} not found")


def test_the_orphan_filter_is_actually_wired_into_run_tests() -> None:
    """The helper must be CALLED, not merely defined.

    Deleting the single call site left every other test in this module green, so
    the guard could have become dead code with nothing to notice. That failure
    mode is why this assertion exists rather than trusting the helper tests.
    """

    assert "_drop_orphan_tests" in _calls_inside("run_tests")


def test_the_per_test_counter_decay_is_actually_wired() -> None:
    """Same reasoning: the success path must clear only the tests that ran."""

    calls = _calls_inside("run_tests")
    assert "_counter_after_passing_run" in calls
    assert "_load_poison_counter" in calls
