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


class _Proc:
    def __init__(self, rc: int, out: str = "") -> None:
        self.returncode = rc
        self.stdout = out
        self.stderr = ""


def test_uncollectable_test_is_dropped_whatever_the_cause(monkeypatch) -> None:
    """REQ-CONDUCTOR-RECEIPT-1 sibling: collectability is the property that matters.

    The orphan check catches one cause, a module never written. exp7131's module WAS written
    and was truncated mid-write by the kill, so it existed and could not be imported. Asking
    pytest directly covers both, and the causes not yet seen.
    """

    import research_conductor as rc

    bad = "tests/python/test_experiment_7131_v626_model_facing_csl.py"
    good = "tests/python/test_adaptive_sleep.py"
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: _Proc(2, f"ERROR {bad}\n1 error\n"))
    assert rc._drop_uncollectable_tests([good, bad]) == [good]


def test_collect_errors_reported_with_a_node_id_still_match_the_file(monkeypatch) -> None:
    """pytest may report `ERROR path::node`; the file is what gets excluded."""

    import research_conductor as rc

    bad = "tests/python/test_experiment_7131_v626_model_facing_csl.py"
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: _Proc(2, f"ERROR {bad}::TestX\n"))
    assert rc._drop_uncollectable_tests([bad]) == []


def test_a_clean_collect_leaves_the_subset_untouched(monkeypatch) -> None:
    """Exit zero means everything collected; do not rebuild or reorder the list."""

    import research_conductor as rc

    files = ["tests/python/test_a.py", "tests/python/test_b.py"]
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: _Proc(0, "2 tests collected\n"))
    assert rc._drop_uncollectable_tests(files) == files


def test_the_collectability_filter_fails_open(monkeypatch) -> None:
    """A guard that empties the subset on its own error stops every task, not one bad file."""

    import research_conductor as rc

    def _boom(*a, **k):
        raise OSError("pytest missing")

    monkeypatch.setattr(rc.subprocess, "run", _boom)
    files = ["tests/python/test_a.py"]
    assert rc._drop_uncollectable_tests(files) == files


def test_the_collectability_filter_is_actually_wired() -> None:
    """It must be CALLED from the orphan filter, or it is another guard nothing runs."""

    assert "_drop_uncollectable_tests" in _calls_inside("_drop_orphan_tests")


def test_a_zero_exit_is_trusted_over_stdout_noise(monkeypatch) -> None:
    """When pytest says everything collected, an ERROR-looking line in output is not a verdict.

    Captured logs and test output can contain a line beginning with ERROR. Filtering on that
    text while pytest reports success would drop a healthy test file, which is the same
    over-matching this project keeps finding in guards that read prose instead of a result.
    """

    import research_conductor as rc

    good = "tests/python/test_adaptive_sleep.py"
    noisy = f"ERROR {good}\ncollected 3 items\n"
    monkeypatch.setattr(rc.subprocess, "run", lambda *a, **k: _Proc(0, noisy))
    assert rc._drop_uncollectable_tests([good]) == [good]


def test_planner_prompt_still_demands_a_progress_line() -> None:
    """The planner prompt must keep the progress-line clause (CLAUDE.md, 2026-09-08).

    Measured: of 145 wall-clock timeouts carrying both figures, 100 have silence equal to elapsed,
    meaning the task never emitted a line, and 96 of those died at exactly 1201 s. A silent task
    has an effective budget of 1200 s regardless of its estimate, its stall grace, or the 4800 s
    hard cap. Asserted here so a future edit to the prompt cannot drop it silently, which is the
    failure mode this project keeps recording against guards nobody calls.
    """

    import inspect

    import research_conductor as rc

    src = inspect.getsource(rc._plan_next_milestone)
    assert "progress line" in src, "planner prompt no longer requires a progress line"
    assert "1201s" in src, "the measurement justifying the requirement was removed"


# Spec refs: REQ-CONDUCTOR-PLANNER-1.
#
# 2026-09-09: the SOTA mandate lived in TWO places -- CLAUDE.md and, inlined
# verbatim, the planner prompt. Updating only CLAUDE.md changed nothing: the
# planner ran 8 minutes after that commit and emitted 3 task prompts naming the
# retired model and zero naming the new one. This pins the planner's copy so the
# two cannot drift apart silently again.


def test_planner_prompt_mandates_the_current_sota_model() -> None:
    import inspect

    import research_conductor as rc

    src = inspect.getsource(rc._plan_next_milestone)
    assert "unsloth/Qwen3.8-27B-GGUF" in src
    # the superseded flagship must not reappear as a mandate
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" not in src


# Spec refs: REQ-CONDUCTOR-PLANNER-1.
#
# 2026-09-09: exp7153 ran an honest 26.3s model canary, declared
# model_full_generation (60s floor) and was quarantined as fabrication. The
# taxonomy already had model_bounded_generation at a 10s floor. The planner now
# states which class each shape declares, so the floor is chosen deliberately.


def test_planner_prompt_explains_substrate_class_floors() -> None:
    import inspect

    import research_conductor as rc

    src = inspect.getsource(rc._plan_next_milestone)
    assert "model_bounded_generation" in src
    assert "model_full_generation" in src
    # the incident that motivated it must stay cited, or the rule loses its why
    assert "exp7153" in src
