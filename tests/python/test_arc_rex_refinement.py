"""Tests for REQ-ARC-WMTE-6248 (Pinductor-style REx refinement).

CPU-only: no GPU, no network, no engine store. The LLM side is a fake proposer; the
scoring side is an injected callable. Covers SCENARIO-ARC-WMTE-6248-UCB1-CONSULTS-QUALITY,
SCENARIO-ARC-WMTE-6248-QBC-ORDERS-BY-DISAGREEMENT, and
SCENARIO-ARC-WMTE-6248-BUDGET-PARITY-BY-CONSTRUCTION.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from carnot.agentic.arc_rex_refinement import (
    RexNode,
    committee_entropy,
    load_engine_from_source,
    qbc_order_mismatches,
    run_rex,
    ucb1_pick,
)


def _node(idx: int, fid: float, visits: int = 0, parent: int | None = None) -> RexNode:
    return RexNode(idx=idx, source=f"src{idx}", valid_fidelity=fid, parent=parent, n_visit=visits)


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6248-UCB1-CONSULTS-QUALITY
# ---------------------------------------------------------------------------


def test_ucb1_equal_visits_picks_highest_quality() -> None:
    nodes = [_node(0, 0.2), _node(1, 0.7), _node(2, 0.4)]
    assert ucb1_pick(nodes) == 1


def test_ucb1_equal_quality_picks_least_visited() -> None:
    nodes = [_node(0, 0.5, visits=5), _node(1, 0.5, visits=0), _node(2, 0.5, visits=2)]
    assert ucb1_pick(nodes) == 1


def test_ucb1_fresh_node_bonus_is_finite() -> None:
    # A fresh node must not win on an infinite bonus: with c small enough the
    # quality term must dominate over the fresh node's exploration bonus.
    nodes = [_node(0, 0.9, visits=3), _node(1, 0.0, visits=0)]
    assert ucb1_pick(nodes, c=0.01) == 0


def test_ucb1_empty_raises() -> None:
    try:
        ucb1_pick([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty node list")


# ---------------------------------------------------------------------------
# committee_entropy edges
# ---------------------------------------------------------------------------


def test_entropy_zero_when_all_agree() -> None:
    assert committee_entropy([1, 1, 1]) == 0.0


def test_entropy_log_k_when_all_differ() -> None:
    assert abs(committee_entropy([1, 2, 3]) - math.log(3)) < 1e-12


def test_entropy_excludes_crashes() -> None:
    # Two crashed members (None) must not read as two extra distinct predictions.
    assert committee_entropy([7, 7, None, None]) == 0.0


def test_entropy_all_crashed_is_zero() -> None:
    assert committee_entropy([None, None]) == 0.0


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6248-QBC-ORDERS-BY-DISAGREEMENT
# ---------------------------------------------------------------------------


def test_qbc_orders_by_disagreement_highest_first() -> None:
    mismatches = [{"i": 0, "tag": "agreed"}, {"i": 1, "tag": "split"}]
    entropies = {0: 0.0, 1: math.log(3)}
    out = qbc_order_mismatches(mismatches, entropies)
    assert [m["tag"] for m in out] == ["split", "agreed"]


def test_qbc_missing_index_sorts_last_and_stably() -> None:
    mismatches = [{"i": 5}, {"i": 9}, {"i": 2}]
    entropies = {2: 1.0}  # 5 and 9 unmeasured -> 0.0, keep original relative order
    out = qbc_order_mismatches(mismatches, entropies)
    assert [m["i"] for m in out] == [2, 5, 9]


# ---------------------------------------------------------------------------
# load_engine_from_source
# ---------------------------------------------------------------------------


def test_load_engine_from_source_executes() -> None:
    src = "import numpy as np\n\ndef engine(grid, action, data):\n    return grid + 1\n"
    eng = load_engine_from_source(src, tag="t")
    out = eng(np.zeros((2, 2), dtype=int), 1, None)
    assert (np.asarray(out) == 1).all()


def test_load_engine_from_source_bad_source_raises() -> None:
    try:
        load_engine_from_source("this is not python", tag="bad")
    except Exception:
        return
    raise AssertionError("expected import failure to raise")


# ---------------------------------------------------------------------------
# run_rex control flow with a fake proposer (no GPU)
# ---------------------------------------------------------------------------


@dataclass
class _Transition:
    grid: np.ndarray
    action: int
    data: dict | None
    next_grid: np.ndarray
    level_before: int = 0
    level_after: int = 0


def _tiny_transitions(n: int = 3) -> list[_Transition]:
    rng = np.random.default_rng(6248)
    out = []
    for _ in range(n):
        g = rng.integers(0, 3, size=(2, 2))
        out.append(_Transition(grid=g, action=1, data=None, next_grid=g + 1))
    return out


class _FakeStore:
    """In-memory stand-in for the isolated engine-store file."""

    def __init__(self) -> None:
        self.text: str | None = None

    def read(self) -> str | None:
        return self.text

    def write(self, text: str) -> None:
        self.text = text


class _FakeProposer:
    """Writes a deterministic sequence of engine sources into the store."""

    def __init__(self, store: _FakeStore, sources: list[str]) -> None:
        self.store = store
        self.sources = list(sources)
        self.calls: list[tuple[str, Any]] = []

    def _emit(self) -> tuple[bool, str]:
        if not self.sources:
            return False, "exhausted"
        self.store.write(self.sources.pop(0))
        return True, "ok"

    def induce(self, game: str, trans: list, cell: int) -> tuple[bool, str]:
        self.calls.append(("induce", None))
        return self._emit()

    def refactor(self, game: str, vr: Any) -> tuple[bool, str]:
        self.calls.append(("refactor", vr))
        return self._emit()


def _scores_from_table(table: dict[str, float]):
    def score_candidate(source: str) -> dict:
        return {
            "valid_fidelity": table.get(source, 0.0),
            "mismatches": [{"i": 0}, {"i": 1}],
            "valid_accuracy": 0.0,
            "valid_n": 2,
            "valid_n_correct": 0,
        }

    return score_candidate


def _mk_vr(node: RexNode, ordered: list[dict]) -> dict:
    # The experiment passes a real VerifyResult; the loop only forwards it, so a dict
    # is enough here.
    return {"mismatches": ordered, "parent_idx": node.idx}


def _run(
    sources: list[str], table: dict[str, float], *, use_ucb1: bool, use_qbc: bool, budget: int = 3
):
    store = _FakeStore()
    prop = _FakeProposer(store, sources)
    result = run_rex(
        "testgame",
        prop,
        train=_tiny_transitions(2),
        valid=_tiny_transitions(2),
        cell=1,
        budget=budget,
        score_candidate=_scores_from_table(table),
        read_store_source=store.read,
        write_store_source=store.write,
        make_verify_result=_mk_vr,
        use_ucb1=use_ucb1,
        use_qbc=use_qbc,
    )
    return result, prop


# SCENARIO-ARC-WMTE-6248-BUDGET-PARITY-BY-CONSTRUCTION


def test_both_arms_spend_identical_llm_call_budget() -> None:
    sources = ["s0", "s1", "s2"]
    table = {"s0": 0.1, "s1": 0.2, "s2": 0.3}
    linear, prop_a = _run(list(sources), table, use_ucb1=False, use_qbc=False)
    rex, prop_b = _run(list(sources), table, use_ucb1=True, use_qbc=True)
    assert linear["llm_calls"] == rex["llm_calls"] == 3
    assert len(prop_a.calls) == len(prop_b.calls) == 3


def test_linear_arm_always_refines_latest() -> None:
    sources = ["s0", "s1", "s2"]
    # s0 scores HIGHEST -- a UCB1 arm would go back to it; linear must not.
    table = {"s0": 0.9, "s1": 0.1, "s2": 0.1}
    result, _ = _run(sources, table, use_ucb1=False, use_qbc=False)
    parents = [n["parent"] for n in result["nodes"]]
    assert parents == [None, 0, 1]


def test_ucb1_arm_returns_to_best_parent() -> None:
    sources = ["s0", "s1", "s2"]
    # s0 best, s1 poor: with modest c the third proposal's parent is s0 again, not s1.
    table = {"s0": 0.9, "s1": 0.05, "s2": 0.5}
    result, _ = _run(sources, table, use_ucb1=True, use_qbc=False)
    parents = [n["parent"] for n in result["nodes"]]
    assert parents[0] is None
    assert parents[1] == 0
    assert parents[2] == 0  # returned to the strong root instead of chaining onto s1


def test_final_pick_is_argmax_valid_fidelity() -> None:
    sources = ["s0", "s1", "s2"]
    table = {"s0": 0.2, "s1": 0.8, "s2": 0.3}
    result, _ = _run(sources, table, use_ucb1=False, use_qbc=False)
    assert result["final_idx"] == 1
    assert result["final_source"] == "s1"
    assert abs(result["final_valid_fidelity"] - 0.8) < 1e-9


def test_failed_induce_returns_empty_summary_after_one_call() -> None:
    store = _FakeStore()
    prop = _FakeProposer(store, [])  # exhausted immediately -> induce fails
    result = run_rex(
        "testgame",
        prop,
        train=_tiny_transitions(1),
        valid=_tiny_transitions(1),
        cell=1,
        budget=4,
        score_candidate=_scores_from_table({}),
        read_store_source=store.read,
        write_store_source=store.write,
        make_verify_result=_mk_vr,
        use_ucb1=True,
        use_qbc=True,
    )
    assert result["final_idx"] is None
    assert result["llm_calls"] == 1  # no refinement calls after a failed induce


def test_unchanged_source_is_not_added_as_new_node() -> None:
    # The proposer "succeeds" but re-emits the identical source: the loop must not
    # grow the population with a duplicate of its own parent.
    store = _FakeStore()

    class _EchoProposer(_FakeProposer):
        def refactor(self, game: str, vr: Any) -> tuple[bool, str]:
            self.calls.append(("refactor", vr))
            return True, "ok"  # writes nothing: store keeps the parent's source

    prop = _EchoProposer(store, ["s0"])
    result = run_rex(
        "testgame",
        prop,
        train=_tiny_transitions(1),
        valid=_tiny_transitions(1),
        cell=1,
        budget=3,
        score_candidate=_scores_from_table({"s0": 0.4}),
        read_store_source=store.read,
        write_store_source=store.write,
        make_verify_result=_mk_vr,
        use_ucb1=False,
        use_qbc=False,
    )
    assert len(result["nodes"]) == 1
    assert result["llm_calls"] == 3
    kinds = [e["kind"] for e in result["events"]]
    assert kinds.count("no_new_source") == 2
