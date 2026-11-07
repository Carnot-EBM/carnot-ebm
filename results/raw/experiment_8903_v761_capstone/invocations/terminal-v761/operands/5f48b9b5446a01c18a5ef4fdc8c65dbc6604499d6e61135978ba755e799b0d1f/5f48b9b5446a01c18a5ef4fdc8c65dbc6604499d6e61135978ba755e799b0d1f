"""REQ-REPORT-8871: preserve exact proofs while avoiding small-request indexes."""

from collections import Counter
from copy import deepcopy
from itertools import product
import sys

import pytest

from carnot.verify import _indexed_witness_core as core
from carnot.verify import _indexed_witness_meter as meter
from carnot.verify.indexed_witness import verify_indexed_witness
from carnot.verify.size_aware_witness import DirectFacts, build_direct, verify_size_aware_witness


def witnesses(call, payload):
    """Observe the common solver's completed bindings without changing its code."""
    bindings = []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is core.solve.__code__:
            if arg is not None and arg["decision"] == "support":
                bindings.extend(frame.f_locals["bindings"])

    sys.setprofile(observe)
    try:
        result = call(payload, cap=None)
    finally:
        sys.setprofile(previous)
    return result, Counter(tuple(sorted(b.items())) for b in bindings)


def scan(payload):
    """Independently enumerate fact choices, including duplicate witness multiplicity."""
    choices = [r for r in payload["facts"] if r[3]]
    if any([*r[:3], not r[3]] in payload["facts"] for r in payload["facts"]):
        return Counter()
    found = []
    for rows in product(choices, repeat=len(payload["query"])):
        binding, valid = {}, True
        for predicate, row in zip(payload["query"], rows, strict=True):
            for term, actual in zip(predicate, row[:3], strict=True):
                valid &= (
                    binding.setdefault(term, actual) == actual
                    if term.startswith("?")
                    else term == actual
                )
        if valid:
            found.append(tuple(sorted(binding.items())))
    return Counter(found)


@pytest.mark.parametrize("size", [0, 1, 15, 16, 17, 64])
def test_boundaries_and_complete_witness_sets(size):
    """SCENARIO-REPORT-8871-CONTRACT: dispatch preserves all shared-variable witnesses."""
    records = [["z", "r", "z", True], ["a", "r", "a", True]]
    payload = dict(
        facts=(records * 2 + [["n", "noise", "m", False]] * size)[:size], query=[["?x", "r", "?x"]]
    )
    indexed, expected = witnesses(verify_indexed_witness, payload)
    actual, observed = witnesses(verify_size_aware_witness, payload)
    assert observed == expected == scan(payload)
    for key in ["decision", "witness", "supporting_bindings", "witness_order_difference"]:
        assert actual[key] == indexed[key]
    assert actual["dispatch_branch"] == ("direct" if size <= 16 else "indexed")
    assert actual["primitive_operations"] == sum(actual["operation_counts"].values())
    assert (actual["operation_counts"]["index_build"] == 0) == (size <= 16)


@pytest.mark.parametrize(
    "facts,query",
    [
        ([], [["a", "r", "b"]]),
        ([["a", "r", "b", False]], [["a", "r", "b"]]),
        ([["a", "r", "b", True], ["a", "r", "b", False]], [["a", "r", "b"]]),
        ([["a", "r", "b", True]], [["?x", "r", "?x"]]),
        (
            [["a", "r", "b", True], ["a", "s", "c", True], ["z", "s", "c", True]],
            [["?x", "r", "b"], ["?x", "s", "c"]],
        ),
        (
            [[str(i), "r", str(j), True] for i in range(3) for j in range(3)],
            [["?x", "r", "?y"], ["?z", "r", "?w"]],
        ),
    ],
)
def test_adversaries_and_budget_edges(facts, query):
    """REQ-REPORT-8871: no exhausted primitive may release a partial proof."""
    payload = dict(facts=facts, query=query)
    original, expected = witnesses(verify_indexed_witness, payload)
    full, actual = witnesses(verify_size_aware_witness, payload)
    assert actual == expected == scan(payload)
    assert full["decision"] == original["decision"]
    for call in [verify_indexed_witness, verify_size_aware_witness]:
        result = call(payload, cap=None)
        cost = result["primitive_operations"]
        for cap in [0, 1, cost - 1, cost]:
            limited = call(payload, cap=cap)
            assert limited["primitive_operations"] == cap
            assert limited["budget_exhausted"] == (cap < cost)
            if cap < cost:
                assert limited["decision"] == "insufficient"
                assert limited["witness"] is None and limited["supporting_bindings"] == 0
            else:
                assert limited["decision"] == result["decision"]
    assert core.REQUEST.get() is None
    with pytest.raises(LookupError):
        meter.CURRENT.get()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [],
        dict(facts=[], query=[]),
        dict(facts=(), query=[["a", "r", "b"]]),
        dict(facts=[["a", "r", "b", 1]], query=[["a", "r", "b"]]),
        dict(facts=[["a", "r", "b", "true"]], query=[["a", "r", "b"]]),
        dict(facts=[["?x", "r", "b", True]], query=[["a", "r", "b"]]),
        dict(facts=[], query=[["a", "?r", "b"]]),
        dict(facts=[[" a", "r", "b", True]], query=[["a", "r", "b"]]),
    ],
)
def test_full_validation_before_dispatch(payload):
    """REQ-REPORT-8871: an empty budget cannot hide a malformed record or nonboolean sign."""
    with pytest.raises(ValueError):
        verify_size_aware_witness(payload, cap=0)


@pytest.mark.parametrize("cap", [-1, True, False, 1.5, "3"])
def test_limits(cap):
    """REQ-REPORT-8871: strict resource limits retain the existing API contract."""
    with pytest.raises(ValueError, match="primitive_limit"):
        verify_size_aware_witness(dict(facts=[], query=[["a", "r", "b"]]), cap=cap)


def test_snapshot_lifetime_and_unexpected_failure(monkeypatch):
    """REQ-REPORT-8871: immutable direct snapshots reject foreign request ownership."""
    payload = dict(facts=[["a", "r", "b", True]] * 2, query=[["?x", "r", "b"]])
    counter = meter.Counter(None)
    counter.counts.update(dict.fromkeys(core.EXTRA, 0))
    token, owner = meter.CURRENT.set(counter), core.REQUEST.set(object())
    try:
        direct, _, _ = build_direct(payload)
        payload["facts"][0][0] = "changed"
        assert list(direct.candidates([], {})) == [("a", "r", "b", True)] * 2
        with pytest.raises(AttributeError):
            direct.facts = ()
        foreign = core.REQUEST.set(object())
        try:
            with pytest.raises(ValueError, match="stale_request_index"):
                list(direct.candidates([], {}))
        finally:
            core.REQUEST.reset(foreign)
    finally:
        core.REQUEST.reset(owner)
        meter.CURRENT.reset(token)
    assert isinstance(direct, DirectFacts)
    assert verify_size_aware_witness(deepcopy(payload), cap=None)["witness"] == {"?x": "changed"}


def test_nested_concurrent_and_failure_cleanup(monkeypatch):
    """REQ-REPORT-8871: concurrent and nested calls keep distinct paid request contexts."""
    from concurrent.futures import ThreadPoolExecutor
    from carnot.verify import size_aware_witness as implementation

    payload = dict(facts=[["a", "r", "b", True]], query=[["?x", "r", "b"]])
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(
            r["witness"] == {"?x": "a"} for r in pool.map(verify_size_aware_witness, [payload] * 8)
        )
    original = implementation._DIRECT_SOLVE
    active = False

    def nested(value):
        nonlocal active
        if not active:
            active = True
            owner, counter = core.REQUEST.get(), meter.CURRENT.get()
            assert verify_size_aware_witness(value)["witness"] == {"?x": "a"}
            assert core.REQUEST.get() is owner and meter.CURRENT.get() is counter
            active = False
        return original(value)

    monkeypatch.setattr(implementation, "_DIRECT_SOLVE", nested)
    assert verify_size_aware_witness(payload)["decision"] == "support"

    def fail(value):
        raise RuntimeError("owned_failure")

    monkeypatch.setattr(implementation, "_DIRECT_SOLVE", fail)
    with pytest.raises(RuntimeError, match="owned_failure"):
        verify_size_aware_witness(payload)
    assert core.REQUEST.get() is None
    with pytest.raises(LookupError):
        meter.CURRENT.get()
