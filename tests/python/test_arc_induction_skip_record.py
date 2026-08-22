"""REQ-ARC-WMTE-6610: the single-shot induce skip record distinguishes its two causes.

Origin: the 2026-08-21 zero-world-model A/B. Every game recorded the conflated label
`proposer_failed_or_missing_root`, and the induce note naming the real failure
(`[HIT n_predict=4096 OUTPUT LIMIT]`) was discarded by `ok, _ = ...` at the call site.
The artifact could not say WHICH branch of the disjunction fired.

Fixtures follow test_experiment_4544_llm_proposer_reinduction.py's minimal-policy shape;
CARNOT_ARC_STALL_REFACTOR_LOOP=0 routes the attempt straight to the plain single-shot
path under test. No GPU, no server.

Spec refs: REQ-ARC-WMTE-6610, SCENARIO-ARC-WMTE-6610-1, SCENARIO-ARC-WMTE-6610-2.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from carnot.agentic import arc_competition_agent as agent

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _policy(induce, root_grid):
    policy = agent.E3AgentPolicy(
        "skiprecord",
        proposer=SimpleNamespace(model_specs="stub", induce=induce),
        value_head=lambda _frame: 0.0,
    )
    policy.transitions = [SimpleNamespace(grid=np.array([[0]]))]
    policy.root_grid = root_grid
    policy._pending_induction_reason = "stall"
    return policy


def test_proposer_failure_records_cause_and_note(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-6610-1: not-ok with a root grid present is `proposer_failed`,
    and the induce note -- the thing the old call site discarded -- rides the attempt."""

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")
    note = "missing ('engine',) in output [HIT n_predict=4096 OUTPUT LIMIT before completing]"
    policy = _policy(lambda *_a, **_k: (False, note), np.array([[1]], dtype=np.int16))

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt["skipped"] == "proposer_failed"
    assert "HIT n_predict" in attempt["proposer_note"]


def test_proposer_note_is_truncated_to_300(monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")
    policy = _policy(lambda *_a, **_k: (False, "x" * 500), np.array([[1]], dtype=np.int16))
    policy._induce_and_plan()
    assert policy.induction_attempts[-1]["proposer_note"] == "x" * 300


def test_missing_root_grid_is_distinguishable(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-6610-2: ok with no plan root grid is `missing_plan_start_grid`,
    with no proposer note -- the proposer did nothing wrong."""

    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")
    policy = _policy(lambda *_a, **_k: (True, "ok"), None)

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt["skipped"] == "missing_plan_start_grid"
    assert "proposer_note" not in attempt


def test_both_causes_record_the_combined_label(monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")
    policy = _policy(lambda *_a, **_k: (False, "declined"), None)

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert attempt["skipped"] == "proposer_failed_and_missing_plan_start_grid"
    assert attempt["proposer_note"] == "declined"


def test_harness_lifts_proposer_notes_onto_the_row() -> None:
    """Wiring pin for the lever harness (same source-level style as its own
    run_cell/HUD-projection guard): the row assembly must carry `induction_proposer_notes`
    read from the attempts' `proposer_note`, bounded to 3 -- the carry-the-message rule
    that `induction_tracebacks` already follows."""

    harness = (_REPO_ROOT / "scripts" / "arc_scored_path_lever_harness.py").read_text()
    assert re.search(
        r'row\["induction_proposer_notes"\]\s*=\s*\[\s*'
        r'a\["proposer_note"\] for a in atts if a\.get\("proposer_note"\)\s*'
        r"\]\[\s*:3\s*\]",
        harness,
    ), (
        "run_cell must lift the induce failure notes onto the row; a proposer_failed tally "
        "without its message is how the 2026-08-21 A/B burned five game budgets blind"
    )
