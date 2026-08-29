"""REQ-HARNESS-5920: the exp5879 debt decision must reach production, and be consumed.

INCIDENT 2026-08-29, and it is why these tests exist rather than a manual check. The first
wiring of the node-id delta into this module was HOLLOW in two independent ways:

  1. `build_artifact` never passed `global_failure_node_ids` to `classify_test_debt`, so
     production always took the fail-closed None path and nothing changed.
  2. `status()` keyed on `unrelated_global_suite_debt` -- the CLASSIFICATION, which stays True
     whenever debt exists -- rather than on `blocks_terminal_ready_status`, the DECISION the
     change computed. So even with perfect evidence the verdict still read
     "blocked: science_ready_but_unrelated_global_suite_debt".

The commit's verification called `classify_test_debt` DIRECTLY with evidence handed in: a path
production never takes. That is the lesson these tests encode -- exercise the CALL SITE, not
the function. Both holes are asserted here so neither can return silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot import experiment_5879_hardness_headroom_taxonomy_corrigendum as exp  # noqa: E402
from carnot.global_suite_baseline import baseline_node_ids  # noqa: E402


def _debt(nodes, *, science_ready: bool = True, owned_ok: bool = True):
    """Classify with a failing global suite and (by default) clean owned checks."""
    commands = (
        list(exp.DEFAULT_TEST_COMMANDS)
        if hasattr(exp, "DEFAULT_TEST_COMMANDS")
        else [exp.FULL_TEST_COMMAND]
    )
    codes = {c: 0 for c in commands}
    codes[exp.FULL_TEST_COMMAND] = 3
    if not owned_ok:
        for c in commands:
            if c != exp.FULL_TEST_COMMAND:
                codes[c] = 1
                break
    return exp.classify_test_debt(
        commands, codes, science_matrix_ready=science_ready, global_failure_node_ids=nodes
    )


def _status_with_score(monkeypatch, debt, score: float) -> str:
    """`hardness_surface_headroom_ready_score` recomputes from real evidence fields rather than
    reading a key, so it is monkeypatched. Fabricating that key looks like it works and
    silently scores 0.0 -- a first version of this test did exactly that and failed for the
    wrong reason."""

    monkeypatch.setattr(exp, "hardness_surface_headroom_ready_score", lambda _a: score)
    return exp.status({"test_debt_classification": debt})


def test_build_artifact_accepts_and_forwards_the_evidence() -> None:
    """Hole 1: the parameter must exist on the CALLER, not only on the function."""
    import inspect

    sig = inspect.signature(exp.build_artifact)
    assert "global_failure_node_ids" in sig.parameters
    src = inspect.getsource(exp.build_artifact)
    assert "global_failure_node_ids=global_failure_node_ids" in src, (
        "build_artifact must FORWARD the evidence to classify_test_debt; accepting it and "
        "dropping it on the floor is the hollow fix this test exists to prevent"
    )


def test_status_consumes_the_decision_not_the_classification(monkeypatch) -> None:
    """Hole 2: with baseline-only debt the verdict must stop saying blocked."""
    debt = _debt(baseline_node_ids())
    assert debt["blocks_terminal_ready_status"] is False
    assert _status_with_score(monkeypatch, debt, 1.0) != "blocked"


def test_a_new_failing_node_still_blocks(monkeypatch) -> None:
    """A regression this task caused must still block, evidence or not."""
    debt = _debt(baseline_node_ids() + ["tests/python/test_regression.py::test_new"])
    assert debt["blocks_terminal_ready_status"] is True
    assert (
        exp.status({"test_debt_classification": debt, "hardness_surface_headroom_ready_score": 1.0})
        == "blocked"
    )


def test_no_evidence_fails_closed(monkeypatch) -> None:
    """No node-id evidence is not the same as no failures."""
    debt = _debt(None)
    assert debt["blocks_terminal_ready_status"] is True
    assert (
        exp.status({"test_debt_classification": debt, "hardness_surface_headroom_ready_score": 1.0})
        == "blocked"
    )


def test_an_owned_failure_still_blocks_even_with_clean_global_debt() -> None:
    """This change must not weaken the module's own verification."""
    debt = _debt(baseline_node_ids(), owned_ok=False)
    assert debt["blocks_terminal_ready_status"] is True


def test_a_legacy_artifact_without_the_decision_field_reads_as_before(monkeypatch) -> None:
    """Artifacts written before this change must not change meaning."""
    monkeypatch.setattr(exp, "hardness_surface_headroom_ready_score", lambda _a: 1.0)
    assert (
        exp.status({"test_debt_classification": {"unrelated_global_suite_debt": True}}) == "blocked"
    )
