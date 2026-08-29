"""The global-suite debt ledger may shrink freely and may NEVER grow silently.

WHY (2026-08-28). ops/global_suite_failure_baseline.json legitimises 1,726
pre-existing test failures so the REQ-HARNESS-5920 node-id delta can tell a new
regression from old debt. The file's own prose says "never upward without
stating" -- but prose is not a check, and this ledger's one recorded
re-baseline was 116 -> 1,726, a 15x growth that no mechanism examined. A debt
snapshot that can be quietly regenerated upward is an approval stamp, not a
ledger.

THE RATCHET. The ceiling below is the committed node count. Re-baselining
DOWNWARD (debt paid) needs no ceremony: lower the ceiling with it or leave it.
Re-baselining UPWARD requires editing this constant in the same change, which
makes the growth a typed, reviewable act with a diff a human reads -- the same
"name it explicitly" bar the harness seal sets. If you are here to raise it:
state in your commit message which node ids grew the ledger and why each is
debt rather than a regression your change introduced.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "ops" / "global_suite_failure_baseline.json"

#: The committed debt count. Raising this number IS the ceremony -- see module docstring.
CEILING = 1726

#: Metadata a lazy regeneration must not drop: the honesty prose is load-bearing,
#: because the ledger's meaning ("snapshot, not approval") lives in these fields.
REQUIRED_HONESTY_KEYS = (
    "what_this_is",
    "how_to_shrink",
    "measurement_conditions",
    "supersedes",
)


def ratchet_violations(payload: dict, ceiling: int = CEILING) -> list[str]:
    """Every way a rewritten ledger could lie, as human-readable violations."""
    problems: list[str] = []
    nodes = payload.get("baseline_node_ids")
    if not isinstance(nodes, list) or not nodes:
        problems.append("baseline_node_ids missing or empty")
        return problems
    if len(nodes) > ceiling:
        problems.append(
            f"ledger grew: {len(nodes)} node ids exceed the ratchet ceiling {ceiling}; "
            "growth must be a typed, explained act (edit the CEILING in "
            "test_global_suite_failure_baseline_ratchet.py in the same change)"
        )
    if payload.get("baseline_node_count") != len(nodes):
        problems.append(
            f"baseline_node_count {payload.get('baseline_node_count')!r} does not match "
            f"the {len(nodes)} ids it describes"
        )
    if len(set(map(str, nodes))) != len(nodes):
        problems.append("baseline_node_ids contains duplicates, inflating the excused set")
    for key in REQUIRED_HONESTY_KEYS:
        if key not in payload:
            problems.append(f"honesty metadata dropped: {key}")
    return problems


def test_committed_ledger_respects_the_ratchet() -> None:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ratchet_violations(payload) == []


def test_a_grown_ledger_is_flagged_not_absorbed() -> None:
    # Synthetic upward drift: one node id over the ceiling must be named a violation.
    payload = {
        "baseline_node_ids": [f"t.py::case_{i}" for i in range(CEILING + 1)],
        "baseline_node_count": CEILING + 1,
        "what_this_is": "x",
        "how_to_shrink": "x",
        "measurement_conditions": "x",
        "supersedes": "x",
    }
    problems = ratchet_violations(payload)
    assert any("grew" in p for p in problems)


def test_a_miscounted_or_duplicated_ledger_is_flagged() -> None:
    base = {
        "what_this_is": "x",
        "how_to_shrink": "x",
        "measurement_conditions": "x",
        "supersedes": "x",
    }
    miscounted = {**base, "baseline_node_ids": ["a", "b"], "baseline_node_count": 3}
    assert any("does not match" in p for p in ratchet_violations(miscounted))
    duplicated = {**base, "baseline_node_ids": ["a", "a"], "baseline_node_count": 2}
    assert any("duplicates" in p for p in ratchet_violations(duplicated))


def test_dropping_the_honesty_prose_is_flagged() -> None:
    payload = {"baseline_node_ids": ["a"], "baseline_node_count": 1}
    problems = ratchet_violations(payload)
    assert {p for p in problems if "honesty metadata dropped" in p} == {
        f"honesty metadata dropped: {key}" for key in REQUIRED_HONESTY_KEYS
    }
